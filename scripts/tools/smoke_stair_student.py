"""Bounded student ABI/warm-start check; no learning or checkpoint saving."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/rl_training"))
sys.path.insert(0, str(ROOT / "scripts/reinforcement_learning/rsl_rl"))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
import cli_args
__import__("rl_training.tasks")
from rl_training.tasks.manager_based.locomotion.velocity.mdp import stair_student


def main():
    torch.set_num_threads(1)
    task = "Rough-Deeprobotics-M20-StairStudent-v0"
    cfg = parse_env_cfg(task, device=args.device or "cuda:0", num_envs=4)
    cfg.seed = 42
    cfg.scene.terrain.terrain_generator.num_rows = 2
    cfg.scene.terrain.visual_material = None
    cfg.scene.sky_light = None
    cfg.commands.base_velocity.debug_vis = False
    wrapped = RslRlVecEnvWrapper(gym.make(task, cfg=cfg), clip_actions=100)
    try:
        agent = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
        runner = OnPolicyRunner(wrapped, cli_args.convert_rsl_rl_cfg_dict(agent.to_dict()),
                                log_dir=None, device=cfg.sim.device)
        obs, _ = wrapped.reset()
        assert {k:tuple(v.shape) for k,v in obs.items()} == {
            "policy": (4,618), "critic": (4,618), "teacher": (4,244)}
        assert torch.equal(obs["policy"], obs["critic"])
        assert torch.equal(obs["policy"][:,:244], obs["teacher"])
        forbidden = {"joint_acceleration", "applied_torque", "wheel_contact_force_magnitude", "material_friction"}
        terms = wrapped.unwrapped.observation_manager.active_terms
        assert not forbidden.intersection(terms["critic"])
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)["actor_state_dict"]
        stair_student.initialize_student_actor(runner.alg.actor, state)
        teacher = MLPModel(TensorDict({"policy": obs["teacher"]}, batch_size=[4]),
            {"actor":["policy"]}, "actor",16,hidden_dims=[512,256,128],activation="elu",
            obs_normalization=False,distribution_cfg={"class_name":"GaussianDistribution", "init_std":1., "std_type":"log"}).to(cfg.sim.device)
        teacher.load_state_dict(state, strict=True)
        with torch.inference_mode():
            expected = teacher(TensorDict({"policy":obs["teacher"]},batch_size=[4]))
            actual = runner.alg.actor(obs)
        error = float((expected-actual).abs().max())
        assert error < 1e-5, error
        cfg.student_perception.height_noise_std = .02
        cfg.student_perception.dropout_probability = .25
        for _ in range(8):
            with torch.inference_mode():
                actions = runner.alg.actor(obs)
                obs, reward, _, _ = wrapped.step(actions)
            assert actions.shape == (4,16) and torch.isfinite(actions).all()
            assert torch.isfinite(reward).all() and all(torch.isfinite(x).all() for x in obs.values())
            assert torch.equal(obs["policy"], obs["critic"])
            h,v,c = (obs["policy"][:,sl] for sl in (slice(57,244),slice(244,431),slice(431,618)))
            assert torch.all((v==0)|(v==1)) and torch.all((c>=0)&(c<=1))
            assert torch.all(h[v==0]==0) and torch.all(c[v==0]==0)
        assert (v==0).any() and (v==1).any()
        result = dict(observation_shapes={k:list(x.shape) for k,x in obs.items()},
            critic_terms=terms["critic"], actor_only_warm_start_max_error=error,
            finite=True, steps=8, num_envs=4, trained=False,
            valid_fraction=float(v.mean()), confidence_mean=float(c.mean()))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+"\n")
        print("STUDENT_SMOKE_RESULT="+json.dumps(result),flush=True)
    finally:
        wrapped.close()


try:
    main()
finally:
    app.close()
