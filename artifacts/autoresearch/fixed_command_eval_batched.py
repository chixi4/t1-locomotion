import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, os.getcwd())

import isaacgym  # noqa: F401
import numpy as np
import torch
import yaml

from envs import *  # noqa: F401,F403
from utils.model import ActorCritic


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def zero_randomization_tree(tree):
    if not isinstance(tree, dict):
        return
    if "range" in tree and isinstance(tree["range"], list) and len(tree["range"]) == 2:
        tree["range"] = [1.0, 1.0] if tree.get("operation") == "scaling" else [0.0, 0.0]
    for value in tree.values():
        zero_randomization_tree(value)


def load_cfg(args, num_envs):
    cfg_path = Path("envs") / f"{args.task}.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    cfg["basic"]["task"] = args.task
    cfg["basic"]["checkpoint"] = args.checkpoint
    cfg["basic"]["headless"] = True
    cfg["basic"]["sim_device"] = args.sim_device
    cfg["basic"]["rl_device"] = args.rl_device
    cfg["basic"]["seed"] = args.seed
    cfg["env"]["num_envs"] = num_envs
    cfg["viewer"]["record_video"] = False
    cfg["rewards"]["terminate_height"] = -10.0
    cfg["rewards"]["terminate_vel"] = 1.0e9
    cfg["rewards"]["terminate_contacts_on"] = []
    if args.clean:
        cfg["terrain"]["type"] = "plane"
        zero_randomization_tree(cfg.get("noise", {}))
        zero_randomization_tree(cfg.get("randomization", {}))
        if "friction" in cfg["randomization"]:
            cfg["randomization"]["friction"]["range"] = [1.0, 1.0]
        if "compliance" in cfg["randomization"]:
            cfg["randomization"]["compliance"]["range"] = [0.0, 0.0]
        if "restitution" in cfg["randomization"]:
            cfg["randomization"]["restitution"]["range"] = [0.0, 0.0]
        cfg["randomization"]["kick_interval_s"] = 1.0e9
        cfg["randomization"]["push_interval_s"] = 1.0e9
        cfg["randomization"]["push_duration_s"] = 0.0
    return cfg


def command_suite():
    rt2 = math.sqrt(0.5)
    return [
        ("stand", 0.0, 0.0, 0.0),
        ("forward_0.3", 0.3, 0.0, 0.0),
        ("forward_0.6", 0.6, 0.0, 0.0),
        ("forward_1.0", 1.0, 0.0, 0.0),
        ("backward_0.3", -0.3, 0.0, 0.0),
        ("backward_0.6", -0.6, 0.0, 0.0),
        ("backward_1.0", -1.0, 0.0, 0.0),
        ("left_0.3", 0.0, 0.3, 0.0),
        ("left_0.6", 0.0, 0.6, 0.0),
        ("left_1.0", 0.0, 1.0, 0.0),
        ("right_0.3", 0.0, -0.3, 0.0),
        ("right_0.6", 0.0, -0.6, 0.0),
        ("right_1.0", 0.0, -1.0, 0.0),
        ("diag_fl_0.7", 0.5, 0.5, 0.0),
        ("diag_fr_0.7", 0.5, -0.5, 0.0),
        ("diag_bl_0.7", -0.5, 0.5, 0.0),
        ("diag_br_0.7", -0.5, -0.5, 0.0),
        ("diag_fl_1.0", rt2, rt2, 0.0),
        ("diag_fr_1.0", rt2, -rt2, 0.0),
        ("diag_bl_1.0", -rt2, rt2, 0.0),
        ("diag_br_1.0", -rt2, -rt2, 0.0),
        ("yaw_l_0.5", 0.0, 0.0, 0.5),
        ("yaw_l_1.0", 0.0, 0.0, 1.0),
        ("yaw_r_0.5", 0.0, 0.0, -0.5),
        ("yaw_r_1.0", 0.0, 0.0, -1.0),
        ("fwd0.6_yaw_l0.5", 0.6, 0.0, 0.5),
        ("fwd0.6_yaw_r0.5", 0.6, 0.0, -0.5),
        ("left0.6_yaw_l0.5", 0.0, 0.6, 0.5),
        ("right0.6_yaw_r0.5", 0.0, -0.6, -0.5),
        ("diag_fl0.7_yaw_l0.5", 0.5, 0.5, 0.5),
        ("diag_br0.7_yaw_r0.5", -0.5, -0.5, -0.5),
    ]


def apply_commands(env, commands, moving_mask, gait_frequency):
    env.commands[:, :3] = commands
    env.gait_frequency[:] = moving_mask.float() * gait_frequency
    env.cmd_resample_time[:] = 10**9
    env._compute_observations()
    return env.obs_buf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="T1CircleGridCurriculum")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--envs_per_command", type=int, default=64)
    parser.add_argument("--duration_s", type=float, default=8.0)
    parser.add_argument("--warmup_s", type=float, default=2.0)
    parser.add_argument("--gait_frequency", type=float, default=1.5)
    parser.add_argument("--sim_device", default="cuda:0")
    parser.add_argument("--rl_device", default="cuda:0")
    parser.add_argument("--out_dir", default="artifacts/autoresearch/evals")
    parser.add_argument("--label", default=None)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    set_seed(args.seed)
    suite = command_suite()
    num_commands = len(suite)
    num_envs = num_commands * args.envs_per_command
    cfg = load_cfg(args, num_envs)
    print(
        f"Preparing batched eval task={args.task} checkpoint={args.checkpoint} "
        f"commands={num_commands} envs_per_command={args.envs_per_command} total_envs={num_envs}",
        flush=True,
    )
    task_class = eval(cfg["basic"]["task"])
    env = task_class(cfg)
    device = cfg["basic"]["rl_device"]
    model = ActorCritic(env.num_actions, env.num_obs, env.num_privileged_obs).to(device)
    model_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(model_dict["model"], strict=False)
    model.eval()
    print(f"Environment/model ready dt={env.dt}", flush=True)

    print("Resetting env", flush=True)
    env.reset()
    print("Reset complete", flush=True)
    command_rows = torch.tensor([[vx, vy, wz] for _, vx, vy, wz in suite], dtype=torch.float, device=env.device)
    commands = command_rows.repeat_interleave(args.envs_per_command, dim=0)
    group_ids = torch.arange(num_commands, device=env.device).repeat_interleave(args.envs_per_command)
    moving_mask = commands.abs().sum(dim=1) > 1.0e-8
    obs = apply_commands(env, commands, moving_mask, args.gait_frequency).to(device)
    steps = int(round(args.duration_s / env.dt))
    warmup_steps = int(round(args.warmup_s / env.dt))

    ever_done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    ever_fallen = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    done_counts = torch.zeros(num_commands, dtype=torch.float, device=env.device)
    samples = 0
    keys = [
        "vx",
        "vy",
        "wz",
        "abs_vx_err",
        "abs_vy_err",
        "xy_err",
        "yaw_err",
        "along",
        "orth_abs",
        "base_height",
        "tilt",
        "action_abs",
        "torque_abs",
        "reward",
    ]
    sums = {key: torch.zeros(num_commands, dtype=torch.float, device=env.device) for key in keys}
    max_xy_err = torch.zeros(num_commands, dtype=torch.float, device=env.device)
    max_yaw_err = torch.zeros(num_commands, dtype=torch.float, device=env.device)
    min_height = torch.full((num_commands,), 10.0, dtype=torch.float, device=env.device)

    with torch.no_grad():
        for step in range(steps):
            if step % 100 == 0:
                print(f"step {step}/{steps}", flush=True)
            dist = model.act(obs)
            obs, rew, done, _ = env.step(dist.loc)
            done = done.to(env.device)
            if done.any():
                ever_done |= done
                done_counts.scatter_add_(0, group_ids, done.float())
            obs = apply_commands(env, commands, moving_mask, args.gait_frequency).to(device)
            height_now = env.base_pos[:, 2] - env.terrain.terrain_heights(env.base_pos)
            tilt_now = torch.linalg.norm(env.projected_gravity[:, :2], dim=1)
            ever_fallen |= (height_now < 0.45) | (tilt_now > 0.9)
            if step < warmup_steps:
                continue

            lin = env.filtered_lin_vel[:, :2]
            yaw = env.filtered_ang_vel[:, 2]
            xy_err = torch.linalg.norm(lin - commands[:, :2], dim=1)
            yaw_err = torch.abs(yaw - commands[:, 2])
            xy_norm = torch.linalg.norm(commands[:, :2], dim=1)
            unit = torch.where(xy_norm[:, None] > 1.0e-6, commands[:, :2] / torch.clamp(xy_norm[:, None], min=1.0e-6), torch.zeros_like(commands[:, :2]))
            along = (lin * unit).sum(dim=1)
            orth = lin[:, 0] * unit[:, 1] - lin[:, 1] * unit[:, 0]
            height = env.base_pos[:, 2] - env.terrain.terrain_heights(env.base_pos)
            tilt = torch.linalg.norm(env.projected_gravity[:, :2], dim=1)
            values = {
                "vx": lin[:, 0],
                "vy": lin[:, 1],
                "wz": yaw,
                "abs_vx_err": torch.abs(lin[:, 0] - commands[:, 0]),
                "abs_vy_err": torch.abs(lin[:, 1] - commands[:, 1]),
                "xy_err": xy_err,
                "yaw_err": yaw_err,
                "along": along,
                "orth_abs": torch.abs(orth),
                "base_height": height,
                "tilt": tilt,
                "action_abs": torch.abs(env.actions).mean(dim=1),
                "torque_abs": torch.abs(env.torques).mean(dim=1),
                "reward": rew,
            }
            for key, value in values.items():
                sums[key].scatter_add_(0, group_ids, value)
            for idx in range(num_commands):
                mask = group_ids == idx
                max_xy_err[idx] = torch.maximum(max_xy_err[idx], xy_err[mask].max())
                max_yaw_err[idx] = torch.maximum(max_yaw_err[idx], yaw_err[mask].max())
                min_height[idx] = torch.minimum(min_height[idx], height[mask].min())
            samples += args.envs_per_command

    rows = []
    denom = max(1, samples)
    for idx, (name, vx, vy, wz) in enumerate(suite):
        cmd_xy = math.hypot(vx, vy)
        row = {
            "name": name,
            "cmd_vx": vx,
            "cmd_vy": vy,
            "cmd_wz": wz,
            "cmd_xy_speed": cmd_xy,
            "envs": args.envs_per_command,
            "duration_s": args.duration_s,
            "warmup_s": args.warmup_s,
            "reset_events_per_env": float(done_counts[idx].item() / args.envs_per_command),
            "ever_reset_frac": float(ever_done[group_ids == idx].float().mean().item()),
            "fall_frac": float(ever_fallen[group_ids == idx].float().mean().item()),
            "min_height": float(min_height[idx].item()),
            "max_xy_err": float(max_xy_err[idx].item()),
            "max_yaw_err": float(max_yaw_err[idx].item()),
        }
        for key in keys:
            row[key] = float(sums[key][idx].item() / denom)
        row["speed_ratio"] = row["along"] / cmd_xy if cmd_xy > 1.0e-6 else None
        row["score_tracking"] = math.exp(-(row["xy_err"] ** 2) / 0.25) * math.exp(-(row["yaw_err"] ** 2) / 0.25)
        rows.append(row)
        print(
            f"{name:22s} cmd=({vx:+.3f},{vy:+.3f},{wz:+.3f}) "
            f"actual=({row['vx']:+.3f},{row['vy']:+.3f},{row['wz']:+.3f}) "
            f"xy_err={row['xy_err']:.3f} yaw_err={row['yaw_err']:.3f} "
            f"fall={row['fall_frac']:.3f} ratio={row['speed_ratio'] if row['speed_ratio'] is not None else float('nan'):.3f}",
            flush=True,
        )

    started = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    label = args.label or Path(args.checkpoint).stem
    mode = "clean" if args.clean else "default"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "label": label,
        "checkpoint": args.checkpoint,
        "task": args.task,
        "mode": mode,
        "commands": num_commands,
        "envs_per_command": args.envs_per_command,
        "duration_s": args.duration_s,
        "warmup_s": args.warmup_s,
        "gait_frequency": args.gait_frequency,
        "dt": env.dt,
        "started_at": started,
        "rows": rows,
    }
    json_path = out_dir / f"{label}_{mode}_batched_{started}.json"
    csv_path = out_dir / f"{label}_{mode}_batched_{started}.csv"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"JSON {json_path}", flush=True)
    print(f"CSV {csv_path}", flush=True)


if __name__ == "__main__":
    main()
