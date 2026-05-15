import argparse
import json
import os
import random

import isaacgym
import numpy as np
import torch
import yaml

from envs import *
from utils.model import ActorCritic


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_cfg(task, num_envs):
    with open(os.path.join("envs", f"{task}.yaml"), "r", encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    cfg["env"]["num_envs"] = num_envs
    cfg["viewer"]["record_video"] = False
    cfg["basic"]["headless"] = True
    cfg["commands"]["curriculum"] = False
    cfg["commands"]["still_proportion"] = 0.0
    cfg["randomization"]["push_force"]["range"] = [0.0, 0.0]
    cfg["randomization"]["push_torque"]["range"] = [0.0, 0.0]
    cfg["randomization"]["kick_lin_vel"]["range"] = [0.0, 0.0]
    cfg["randomization"]["kick_ang_vel"]["range"] = [0.0, 0.0]
    return cfg


def command_cases():
    return [
        ("stand", (0.0, 0.0, 0.0)),
        ("forward_10", (1.0, 0.0, 0.0)),
        ("backward_08", (-0.8, 0.0, 0.0)),
        ("left_08", (0.0, 0.8, 0.0)),
        ("right_08", (0.0, -0.8, 0.0)),
        ("forward_15", (1.5, 0.0, 0.0)),
    ]


def apply_command(env, cmd, steps_ahead):
    env.commands[:, 0] = float(cmd[0])
    env.commands[:, 1] = float(cmd[1])
    env.commands[:, 2] = float(cmd[2])
    speed = (cmd[0] * cmd[0] + cmd[1] * cmd[1]) ** 0.5
    env.gait_frequency[:] = 0.0 if speed < 1.0e-6 and abs(cmd[2]) < 1.0e-6 else 1.5
    env.cmd_resample_time[:] = env.episode_length_buf + steps_ahead + 1000


def summarize(values):
    if not values:
        return {}
    data = torch.cat(values, dim=0).detach().cpu()
    return {
        "mean": float(data.mean()),
        "p95": float(torch.quantile(data, 0.95)),
        "max": float(data.max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--warmup-s", type=float, default=1.0)
    parser.add_argument("--out")
    parser.add_argument("--seed", type=int, default=44)
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = load_cfg(args.task, args.num_envs)
    task_class = eval(cfg["basic"]["task"])
    env = task_class(cfg)
    device = cfg["basic"]["rl_device"]
    model = ActorCritic(env.num_actions, env.num_obs, env.num_privileged_obs).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    obs, infos = env.reset()
    obs = obs.to(device)
    total_steps = int(args.seconds / env.dt)
    warmup_steps = int(args.warmup_s / env.dt)
    results = []
    for name, cmd in command_cases():
        env._reset_idx(torch.arange(env.num_envs, device=device))
        env._compute_observations()
        obs = env.obs_buf.to(device)
        reset_count = torch.zeros(env.num_envs, device=device)
        tilt_values = []
        shoulder_values = []
        for step in range(total_steps):
            apply_command(env, cmd, total_steps)
            with torch.no_grad():
                action = torch.clamp(model.act(obs).loc, -1.0, 1.0)
            obs, rew, done, infos = env.step(action)
            obs = obs.to(device)
            reset_count += done.float()
            apply_command(env, cmd, total_steps)
            if step >= warmup_steps:
                tilt = torch.sqrt(torch.sum(torch.square(env.projected_gravity[:, :2]), dim=1))
                tilt_values.append(tilt)
                if len(env.arm_indices) >= 4:
                    q = env.dof_pos[:, env.arm_indices] - env.default_dof_pos[:, env.arm_indices]
                    shoulder_values.append(torch.abs(q[:, :4]).reshape(-1))
        results.append(
            {
                "case": name,
                "cmd": cmd,
                "reset_events_per_env": float(reset_count.mean()),
                "camera_tilt": summarize(tilt_values),
                "shoulder_abs": summarize(shoulder_values),
            }
        )
    payload = {"task": args.task, "checkpoint": args.checkpoint, "results": results}
    text = json.dumps(payload, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
