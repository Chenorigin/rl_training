"""Warm-start CTS models preserving both source actors at initialization.

The latent is a complete 16D action proposal. The new shared decoder starts as
the identity, so warm starting neither drops the Safe teacher branches nor
changes the student's actions. Raw commands are a separate deployable input:
effective yaw feedback in policy789 cannot identify a raw zero command.
"""
from copy import deepcopy
import math

import torch
from torch import nn
from tensordict import TensorDict
from rsl_rl.models import MLPModel

from task_support.stair_quiet_v1_safe.actor import SafeResidualActor


OBS_DIMS = {"policy": 789, "teacher": 244, "motion_control": 3,
            "locomotion_scan": 187, "safety": 48, "teacher_critic": 285}


def sample_observations(n=2, device="cpu"):
    return TensorDict({key: torch.zeros(n, dim, device=device) for key, dim in OBS_DIMS.items()}, [n])


def teacher_observations(obs):
    return TensorDict({"policy": obs["teacher"], "motion_control": obs["motion_control"],
                       "locomotion_scan": obs["locomotion_scan"], "safety": obs["safety"]},
                      batch_size=obs["policy"].shape[:-1])


def actor_settings(all_terrain_cfg):
    """Optional versioned architecture; legacy reconstruction never enables it."""
    if all_terrain_cfg is None:
        return None
    cfg = all_terrain_cfg.get("actor") if isinstance(all_terrain_cfg, dict) else None
    if not isinstance(cfg, dict) or set(cfg) != {"idle_leg_scale", "new_domain_anchor_scale"}:
        raise ValueError("CTS all-terrain actor settings are incomplete")
    for name, value in cfg.items():
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Invalid CTS all-terrain actor setting: " + name)
    if cfg["idle_leg_scale"] <= 0:
        raise ValueError("CTS all-terrain idle adaptation must have positive bounds")
    return cfg


def idle_network(input_dim, hidden_dims, output_dim=12):
    # Deterministic construction lets strict resume verify the frozen reference
    # teacher, without consuming or depending on the caller's RNG state.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(130013)
        dimensions = [input_dim, *hidden_dims, output_dim]
        layers = []
        for index, (before, after) in enumerate(zip(dimensions, dimensions[1:])):
            layers.append(nn.Linear(before, after))
            if index < len(dimensions)-2:
                layers.append(nn.ELU())
        network = nn.Sequential(*layers)
        nn.init.zeros_(network[-1].weight)
        nn.init.zeros_(network[-1].bias)
    return network


class AllTerrainTeacher(SafeResidualActor):
    """Keep Safe moving behavior and add bounded, zero-initialized idle legs."""
    def __init__(self, obs, *args, idle_leg_scale, **kwargs):
        super().__init__(obs, *args, **kwargs)
        self.idle_leg_scale = float(idle_leg_scale)
        self.idle_mlp = idle_network(244+48, (128, 64))

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        mean = super().forward(obs, masks, hidden_state, stochastic_output=False)
        idle = obs["motion_control"].abs().amax(-1, keepdim=True) <= 1.e-4
        logits = self.idle_mlp(torch.cat((obs["policy"], obs["safety"]), -1))
        bounds = mean.new_tensor([.18, .35, .35]*4) * self.idle_leg_scale
        legs = torch.tanh(logits) * bounds * idle
        mean = mean + torch.cat((legs, torch.zeros_like(mean[:, 12:])), -1)
        if stochastic_output:
            self.distribution.update(mean)
            return self.distribution.sample()
        return mean


class SharedDecoder(nn.Module):
    def __init__(self, hidden_dims, bounds, idle_leg_scale=0., idle_wheels=False):
        super().__init__()
        if len(bounds) != 16 or any(float(v) <= 0 for v in bounds):
            raise ValueError("CTS decoder requires 16 positive action bounds")
        dimensions = [57 + 16, *hidden_dims, 16]
        layers = []
        for index, (before, after) in enumerate(zip(dimensions, dimensions[1:])):
            layers.append(nn.Linear(before, after))
            if index < len(dimensions)-2:
                layers.append(nn.ELU())
        self.network = nn.Sequential(*layers)
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        self.register_buffer("bounds", torch.tensor(bounds, dtype=torch.float32))
        self.idle_leg_scale = float(idle_leg_scale)
        self.idle_mlp = idle_network(57+16, hidden_dims) if idle_leg_scale > 0 else None
        self.idle_wheel_mlp = idle_network(57+16, hidden_dims, 4) if idle_wheels else None

    def forward(self, common: torch.Tensor, latent: torch.Tensor, raw_command: torch.Tensor) -> torch.Tensor:
        correction = torch.tanh(self.network(torch.cat((common, latent), -1))) * self.bounds
        moving = raw_command.abs().amax(-1, keepdim=True) > 1.e-4
        mean = latent + correction * moving
        if self.idle_mlp is not None:
            idle = torch.tanh(self.idle_mlp(torch.cat((common, latent), -1)))
            legs = idle * self.bounds[:12] * self.idle_leg_scale * ~moving
            mean = mean + torch.cat((legs, torch.zeros_like(latent[:, 12:])), -1)
        if self.idle_wheel_mlp is not None:
            # Identity at migration; learned gain in (0,2), not a hard stop.
            gain = 2*torch.sigmoid(self.idle_wheel_mlp(torch.cat((common, latent), -1)))
            wheels = torch.where(moving, mean[:, 12:], mean[:, 12:]*gain)
            mean = torch.cat((mean[:, :12], wheels), -1)
        return mean


class LatentCritic(nn.Module):
    def __init__(self, source_state):
        super().__init__()
        weights = [source_state[f"mlp.{index}.weight"] for index in (0, 2, 4, 6)]
        if weights[0].shape[1] != 288 or weights[-1].shape[0] != 1:
            raise ValueError("CTS warm start requires a 285D + raw-command3 Safe critic")
        layers = []
        for index, weight in enumerate(weights):
            layers.append(nn.Linear(weight.shape[1] + (16 if index == 0 else 0), weight.shape[0]))
            if index < len(weights)-1:
                layers.append(nn.ELU())
        self.mlp = nn.Sequential(*layers)
        expanded = deepcopy(source_state)
        expanded["mlp.0.weight"] = torch.cat((weights[0], weights[0].new_zeros(weights[0].shape[0], 16)), -1)
        self.load_state_dict(expanded, strict=True)

    def forward(self, privileged: torch.Tensor, raw_command: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        # The value objective must never update either policy encoder.
        return self.mlp(torch.cat((privileged, raw_command, latent.detach()), -1)).squeeze(-1)


class CTSStudentExport(nn.Module):
    def __init__(self, student, decoder):
        super().__init__()
        self.encoder = deepcopy(student.as_jit())
        self.decoder = deepcopy(decoder)

    def forward(self, policy: torch.Tensor, raw_command: torch.Tensor) -> torch.Tensor:
        return self.decoder(policy[:, :57], self.encoder(policy), raw_command)


class CTSModels(nn.Module):
    def __init__(self, teacher, student, decoder, critic, log_std, clean_teacher_decoder=False):
        super().__init__()
        self.teacher, self.student = teacher, student
        self.decoder, self.critic = decoder, critic
        self.clean_teacher_decoder = bool(clean_teacher_decoder)
        self.reference_teacher = deepcopy(teacher).requires_grad_(False).eval()
        self.register_buffer("log_std", log_std.detach().clone())

    def teacher_latent(self, obs):
        return self.teacher(teacher_observations(obs))

    def teacher_action(self, obs, latent=None):
        """Complete current teacher action, with versioned common-input semantics."""
        latent = self.teacher_latent(obs) if latent is None else latent
        common = obs['teacher'][:, :57] if self.clean_teacher_decoder else obs['policy'][:, :57]
        return self.decoder(common, latent, obs['motion_control'])

    def student_latent(self, obs):
        return self.student(TensorDict({"policy": obs["policy"]}, batch_size=obs["policy"].shape[:-1]))

    def student_action(self, policy, raw_command):
        """Complete nominal deployment action, including the shared decoder."""
        latent = self.student_latent({"policy": policy})
        return self.decoder(policy[:, :57], latent, raw_command)

    def latent(self, obs, teacher_mask, detach_student=True):
        teacher = self.teacher_latent(obs)
        student = self.student_latent(obs)
        if detach_student:
            student = student.detach()
        return torch.where(teacher_mask[:, None], teacher, student)

    def mean(self, obs, teacher_mask, detach_student=True):
        latent = self.latent(obs, teacher_mask, detach_student)
        common = obs['policy'][:, :57]
        if self.clean_teacher_decoder:
            common = torch.where(teacher_mask[:, None], obs['teacher'][:, :57], common)
        return self.decoder(common, latent, obs["motion_control"]), latent

    def value(self, obs, latent):
        return self.critic(obs["teacher_critic"], obs["motion_control"], latent)

    def student_export(self):
        return CTSStudentExport(self.student, self.decoder)


def build_models(saved_teacher, saved_student, cfg, device="cpu", *, all_terrain_cfg=None):
    """Strict source loading; no optimizer step and no simulator dependencies."""
    manifest = saved_teacher["m20_training_manifest"]
    teacher_cfg = deepcopy(manifest["actor_model_config"])
    teacher_cfg.pop("class_name")
    if teacher_cfg.get("obs_normalization") or teacher_cfg.get("activation") != "elu":
        raise ValueError("CTS requires the current unnormalized ELU Safe teacher")
    obs = sample_observations()
    settings = actor_settings(all_terrain_cfg)
    if settings is None:
        teacher = SafeResidualActor(teacher_observations(obs), {"actor": ["policy"]}, "actor", 16, **teacher_cfg)
        teacher.load_state_dict(saved_teacher["actor_state_dict"], strict=True)
    else:
        teacher = AllTerrainTeacher(teacher_observations(obs), {"actor": ["policy"]}, "actor", 16,
                                    idle_leg_scale=settings["idle_leg_scale"], **teacher_cfg)
        source = saved_teacher["actor_state_dict"]
        missing = {key for key in teacher.state_dict() if key.startswith("idle_mlp.")}
        if set(source) != set(teacher.state_dict())-missing:
            raise ValueError("CTS all-terrain requires an unchanged Safe source teacher")
        teacher.load_state_dict({**teacher.state_dict(), **source}, strict=True)
    from .configuration import train_teacher_base
    full_teacher = train_teacher_base(cfg)
    # v16 trains every teacher policy layer. Exploration std is a separate
    # fixed contract; reference_teacher remains a fully frozen copy.
    for name, parameter in teacher.named_parameters():
        parameter.requires_grad_(
            name.startswith(("residual_mlp.", "safety_mlp.", "idle_mlp."))
            or (full_teacher and name.startswith("mlp.")))
    student_state = saved_student.get("student_state_dict", saved_student.get("actor_state_dict"))
    if student_state is None or student_state["mlp.0.weight"].shape[1] != 789:
        raise ValueError("CTS student warm start requires the complete policy789 actor")
    hidden = [student_state[f"mlp.{i}.weight"].shape[0] for i in (0, 2, 4)]
    student = MLPModel(obs, {"student": ["policy"]}, "student", 16, hidden_dims=hidden,
                       activation="elu", obs_normalization=False,
                       distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1., "std_type": "log"})
    student.load_state_dict(student_state, strict=True)
    student.distribution.requires_grad_(False)
    critic = LatentCritic(saved_teacher["critic_state_dict"])
    from .wall_stability import settings as wall_settings
    decoder = SharedDecoder(cfg["decoder_hidden_dims"], cfg["decoder_bound"],
                            settings["idle_leg_scale"] if settings is not None else 0.,
                            idle_wheels=wall_settings(all_terrain_cfg) is not None)
    log_std = saved_teacher["actor_state_dict"]["distribution.log_std_param"]
    if log_std.shape != (16,) or not torch.isfinite(log_std).all() or not (log_std.exp() > 0).all():
        raise ValueError("Invalid source teacher action standard deviation")
    models = CTSModels(teacher, student, decoder, critic, log_std,
                       clean_teacher_decoder=settings is not None).to(device)
    if not all(torch.isfinite(value).all() for value in models.state_dict().values()):
        raise ValueError("Nonfinite CTS source model")
    return models


def transfer_state(models, source):
    """Validate an init-CTS transfer, allowing only the complete new idle heads.

    Returns a merged state dictionary without mutating the live model. A caller
    can pass it through the ordinary strict algorithm checkpoint validator.
    This is not a resume fallback: optimizer state and contracts stay strict.
    """
    expected = models.state_dict()
    if not isinstance(source, dict):
        raise ValueError("Invalid CTS transfer state")
    missing = set(expected)-set(source)
    families = (('teacher.idle_mlp.', 'reference_teacher.idle_mlp.', 'decoder.idle_mlp.'),
                ('decoder.idle_wheel_mlp.',))
    allowed = set()
    for prefixes in families:
        group = {key for key in expected if key.startswith(prefixes)}
        absent = missing & group
        if absent and absent != group:
            raise ValueError('CTS transfer requires complete zero-output idle heads')
        allowed |= group
    if set(source)-set(expected) or missing-allowed:
        raise ValueError('CTS transfer may add only complete zero-output idle heads')
    for key, value in source.items():
        if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
                or value.shape != expected[key].shape or value.dtype != expected[key].dtype
                or not torch.isfinite(value).all()):
            raise ValueError("Invalid CTS transfer tensor: " + key)
    for name, module in models.named_modules():
        if name in ('teacher.idle_mlp', 'reference_teacher.idle_mlp', 'decoder.idle_mlp', 'decoder.idle_wheel_mlp'):
            if any(key.startswith(name+'.') for key in missing):
                if torch.count_nonzero(module[-1].weight) or torch.count_nonzero(module[-1].bias):
                    raise ValueError('CTS legacy transfer requires zero-output idle heads')
    return {**expected, **source}
