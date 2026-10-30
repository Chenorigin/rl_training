"""Explicit Transition checkpoint/scan contract; old Safe loader stays strict."""
from copy import deepcopy
import hashlib
import json
import torch
from tensordict import TensorDict
from safe_actor_adapter import GROUPS, ROOT, SHARED_SOURCES, SafeMujocoAdapter

CLASS = "task_support.stair_quiet_v1_transition.actor:TransitionResidualActor"
ROUTING = "stair_scan_through_zero_v1"


def load_transition_actor(state, manifest):
    noise, config = manifest.get("noise_refinement", {}), manifest.get("actor_model_config", {})
    contract = noise.get("contract", {})
    if (manifest.get("refinement_version") != "m20_stair_quiet_v1_transition.v1"
            or noise.get("task") != "stair_quiet_v1_transition" or noise.get("version") != 1
            or manifest.get("input_routing_version") != ROUTING or contract.get("input_routing") != ROUTING
            or noise.get("contract_sha256") != hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()):
        raise ValueError("Transition checkpoint task/scan/contract mismatch")
    if (manifest.get("actor_input_groups") != GROUPS or config.get("class_name") != CLASS
            or manifest.get("motion_control_version") != "m20_raw_command_gate_20261017"
            or manifest.get("m20_profile") != "idle_stop_motion_v1" or manifest.get("control_profile") != "legacy_v1"):
        raise ValueError("Transition actor input/control ABI mismatch")
    if (config.get("hidden_dims") != [512, 256, 128] or config.get("residual_hidden_dims") != [128, 64]
            or config.get("activation") != "elu" or config.get("obs_normalization") is not False):
        raise ValueError("Unsupported Transition architecture")
    for relative in (*SHARED_SOURCES, "stair_quiet_v1_transition/actor.py", "stair_quiet_v1_transition/settings.py"):
        key = "source/rl_training/task_support/"+relative
        if noise.get("source_sha256", {}).get(key) != hashlib.sha256((ROOT/key).read_bytes()).hexdigest():
            raise ValueError("Transition shared source mismatch: "+relative)
    if not all(torch.is_tensor(v) and torch.isfinite(v).all() for v in state.values()):
        raise ValueError("Nonfinite Transition state")
    from task_support.stair_quiet_v1_transition.actor import TransitionResidualActor
    from task_support.stair_quiet_v1_transition.settings import TransitionSettings
    settings = TransitionSettings(**contract["environment"]["stair_tail"])
    settings.validate()
    obs = TensorDict({k: torch.zeros(1, v) for k, v in GROUPS.items()}, [1])
    kwargs = deepcopy(config)
    kwargs.pop("class_name")
    actor = TransitionResidualActor(obs, {"actor": ["policy"]}, "actor", 16, **kwargs)
    actor.load_state_dict(state, strict=True)
    actor.eval().requires_grad_(False)
    actor.uses_safe_control = actor.uses_transition_control = True
    actor.m20_training_manifest, actor.safe_settings = manifest, settings
    print("[POLICY] StairQuiet-v1-Transition: stair scan retained through zero command", flush=True)
    return actor


class TransitionMujocoAdapter(SafeMujocoAdapter):
    def inputs(self, observation, requested_command, data, perception=None):
        result = super().inputs(observation, requested_command, data, perception)
        if self.flat:
            # Flat zero behavior must see the original delivered scan, including
            # its sensor corruption/missing cells; no route-corridor fill.
            result["locomotion_scan"] = result["policy"][:, 57:].clone()
        return result

    def metadata(self):
        result = super().metadata()
        result.update(input_routing_version=ROUTING, zero_command_scan="same_delivered_locomotion_scan",
                      flat_scan="original_delivered_policy_scan")
        return result
