import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import isaacgym  # noqa: F401
from isaacgym import gymtorch
import numpy as np
import torch
import yaml


REPO = Path("/mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official")
sys.path.insert(0, str(REPO))

from envs import *  # noqa: F401,F403
from utils.model import ActorCritic


FPS = 50
BODY_NAMES = [
    "Trunk",
    "Hip_Pitch_Left",
    "Hip_Roll_Left",
    "Hip_Yaw_Left",
    "Shank_Left",
    "Ankle_Cross_Left",
    "left_foot_link",
    "Hip_Pitch_Right",
    "Hip_Roll_Right",
    "Hip_Yaw_Right",
    "Shank_Right",
    "Ankle_Cross_Right",
    "right_foot_link",
]


def fixed_cfg(cfg, args):
    cfg["basic"]["headless"] = True
    cfg["viewer"]["record_video"] = False
    cfg["env"]["num_envs"] = 1
    cfg["basic"]["sim_device"] = "cuda:0"
    cfg["basic"]["rl_device"] = "cuda:0"
    cfg["basic"]["seed"] = args.seed
    cfg["runner"]["use_wandb"] = False

    fixed_add0 = {"range": [0.0, 0.0], "operation": "additive", "distribution": "uniform"}
    fixed_scale1 = {"range": [1.0, 1.0], "operation": "scaling", "distribution": "uniform"}
    fixed_friction = {"range": [1.0, 1.0], "operation": "additive", "distribution": "uniform"}
    r = cfg["randomization"]
    r["init_dof_pos"] = None
    r["init_base_pos_xy"] = None
    r["init_base_lin_vel_xy"] = None
    r["kick_interval_s"] = 1e9
    r["push_interval_s"] = 1e9
    r["kick_lin_vel"] = None
    r["kick_ang_vel"] = None
    r["push_force"] = None
    r["push_torque"] = None
    r["dof_stiffness"] = fixed_scale1
    r["dof_damping"] = fixed_scale1
    r["dof_friction"] = fixed_add0
    r["friction"] = fixed_friction
    r["compliance"] = fixed_add0
    r["restitution"] = fixed_add0
    r["base_com"] = fixed_add0
    r["base_mass"] = fixed_scale1
    r["other_com"] = fixed_add0
    r["other_mass"] = fixed_scale1

    cfg["commands"]["curriculum"] = False
    cfg["commands"]["still_proportion"] = 0.0
    cfg["commands"]["resampling_time_s"] = [1e6, 1e6 + 1.0]
    cfg["rewards"]["episode_length_s"] = args.seconds + 5.0
    return cfg


def sample_target(args):
    if random.random() < 0.12:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    speed = math.sqrt(random.random()) * args.max_speed
    theta = random.uniform(-math.pi, math.pi)
    yaw = random.uniform(-args.max_yaw, args.max_yaw)
    if random.random() < 0.2:
        yaw *= 0.35
    return np.array([speed * math.cos(theta), speed * math.sin(theta), yaw], dtype=np.float32)


def make_segments(args):
    segments = []
    t = 0.0
    prev = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    first = min(2.0, args.seconds)
    segments.append({"start": 0.0, "end": first, "prev": prev.copy(), "target": prev.copy()})
    t = first
    while t < args.seconds - 1e-6:
        dur = random.uniform(2.4, 4.6)
        end = min(args.seconds, t + dur)
        target = sample_target(args)
        segments.append({"start": t, "end": end, "prev": prev.copy(), "target": target.copy()})
        prev = target
        t = end
    return segments


def command_at(segments, t, smooth_s):
    seg = segments[-1]
    for candidate in segments:
        if candidate["start"] <= t < candidate["end"]:
            seg = candidate
            break
    u = min(1.0, max(0.0, (t - seg["start"]) / smooth_s))
    u = u * u * (3.0 - 2.0 * u)
    return (1.0 - u) * seg["prev"] + u * seg["target"]


def clean_start(env):
    env.dof_pos[:] = env.default_dof_pos
    env.dof_vel.zero_()
    env.root_states[0, :3] = torch.tensor([0.0, 0.0, 0.72], dtype=torch.float, device=env.device)
    env.root_states[0, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float, device=env.device)
    env.root_states[0, 7:13] = 0.0
    env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(env.dof_state))
    env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.root_states))
    env.gym.refresh_dof_state_tensor(env.sim)
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)
    env.base_pos[:] = env.root_states[:, 0:3]
    env.base_quat[:] = env.root_states[:, 3:7]
    env.filtered_lin_vel.zero_()
    env.filtered_ang_vel.zero_()
    env.commands.zero_()
    env.gait_frequency.zero_()
    env.cmd_resample_time[:] = 10**9
    env._compute_observations()
    return env.obs_buf


def actor_body_names(env):
    try:
        return env.gym.get_actor_rigid_body_names(env.envs[0], env.actor_handles[0])
    except Exception:
        return BODY_NAMES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--seed", type=int, default=20260504)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--smooth_s", type=float, default=0.65)
    parser.add_argument("--max_speed", type=float, default=0.92)
    parser.add_argument("--max_yaw", type=float, default=0.85)
    args = parser.parse_args()

    os.chdir(REPO)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    with open(f"envs/{args.task}.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg = fixed_cfg(cfg, args)
    segments = make_segments(args)

    print("creating env...", flush=True)
    task_class = eval(args.task)
    env = task_class(cfg)
    device = cfg["basic"]["rl_device"]
    model = ActorCritic(env.num_actions, env.num_obs, env.num_privileged_obs).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"], strict=False)
    model.eval()

    env.reset()
    obs = clean_start(env).to(device)

    rigid_names = actor_body_names(env)
    missing = [name for name in BODY_NAMES if name not in rigid_names]
    if missing:
        raise RuntimeError(f"Missing replay bodies: {missing}; available={rigid_names}")
    indices = torch.tensor([rigid_names.index(name) for name in BODY_NAMES], dtype=torch.long, device=env.device)

    frames = []
    reset_steps = []
    total_steps = int(round(args.seconds / env.dt))
    for step in range(total_steps):
        t = step * env.dt
        cmd = command_at(segments, t, args.smooth_s)
        cmd_t = torch.tensor(cmd, dtype=torch.float, device=env.device).view(1, 3)
        env.commands[:, :3] = cmd_t
        gait = 0.0 if float(np.linalg.norm(cmd)) < 0.04 else 1.45
        env.gait_frequency[:] = gait
        env.cmd_resample_time[:] = 10**9
        with torch.no_grad():
            action = model.act(obs).loc
        obs, _, done, _ = env.step(action)
        obs = obs.to(device)
        env.commands[:, :3] = cmd_t
        env.gait_frequency[:] = gait
        env.cmd_resample_time[:] = 10**9
        if bool(done[0].item()):
            reset_steps.append(step)

        bs = env.body_states[0, indices, :7].detach().cpu().numpy()
        p = np.round(bs[:, :3], 4).tolist()
        q_xyzw = bs[:, 3:7]
        q_wxyz = np.stack([q_xyzw[:, 3], q_xyzw[:, 0], q_xyzw[:, 1], q_xyzw[:, 2]], axis=1)
        q = np.round(q_wxyz, 5).tolist()
        frames.append({"p": p, "q": q, "c": np.round(cmd.astype(np.float32), 4).tolist()})

    commands = [
        {"start": round(float(seg["start"]), 3), "end": round(float(seg["end"]), 3), "target": np.round(seg["target"], 4).tolist()}
        for seg in segments
    ]
    payload = {
        "title": f"{args.label} 随机命令长回放",
        "stage": "t1circle_random",
        "stageLabel": f"{args.label} 随机命令长回放",
        "command": {"vx": 0.0, "vy": 0.0, "yaw": 0.0},
        "fps": FPS,
        "bodyNames": BODY_NAMES,
        "frames": frames,
        "commandSegments": commands,
        "meta": {
            "checkpoint": args.checkpoint,
            "seed": args.seed,
            "seconds": args.seconds,
            "policy_dt": env.dt,
            "frames": len(frames),
            "reset_steps": reset_steps,
            "num_command_segments": len(commands),
            "command_sampling": f"circle speed<={args.max_speed}, yaw<={args.max_yaw}, smoothstep transitions",
        },
    }
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(json.dumps(payload["meta"], indent=2), flush=True)
    print(str(out), flush=True)


if __name__ == "__main__":
    main()
