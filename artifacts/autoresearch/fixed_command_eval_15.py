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
        if tree.get("operation") == "scaling":
            tree["range"] = [1.0, 1.0]
        else:
            tree["range"] = [0.0, 0.0]
    for value in tree.values():
        zero_randomization_tree(value)


def load_cfg(task, checkpoint, num_envs, headless, sim_device, rl_device, clean):
    cfg_path = Path("envs") / f"{task}.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    cfg["basic"]["task"] = task
    cfg["basic"]["checkpoint"] = checkpoint
    cfg["basic"]["headless"] = headless
    cfg["basic"]["sim_device"] = sim_device
    cfg["basic"]["rl_device"] = rl_device
    cfg["basic"]["seed"] = 1234
    cfg["env"]["num_envs"] = num_envs
    cfg["viewer"]["record_video"] = False
    cfg["commands"]["curriculum"] = False
    cfg["commands"]["still_proportion"] = 0.0
    cfg["commands"]["resampling_time_s"] = [1.0e6, 1.0e6 + 1.0]
    if clean:
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
    rt2_1 = math.sqrt(0.5)
    rt2_15 = 1.5 * math.sqrt(0.5)
    return [
        ("stand", 0.0, 0.0, 0.0),
        ("forward_0.5", 0.5, 0.0, 0.0),
        ("forward_1.0", 1.0, 0.0, 0.0),
        ("forward_1.5", 1.5, 0.0, 0.0),
        ("backward_0.5", -0.5, 0.0, 0.0),
        ("backward_1.0", -1.0, 0.0, 0.0),
        ("backward_1.5", -1.5, 0.0, 0.0),
        ("left_0.5", 0.0, 0.5, 0.0),
        ("left_1.0", 0.0, 1.0, 0.0),
        ("left_1.5", 0.0, 1.5, 0.0),
        ("right_0.5", 0.0, -0.5, 0.0),
        ("right_1.0", 0.0, -1.0, 0.0),
        ("right_1.5", 0.0, -1.5, 0.0),
        ("diag_fl_1.0", rt2_1, rt2_1, 0.0),
        ("diag_fr_1.0", rt2_1, -rt2_1, 0.0),
        ("diag_bl_1.0", -rt2_1, rt2_1, 0.0),
        ("diag_br_1.0", -rt2_1, -rt2_1, 0.0),
        ("diag_fl_1.5", rt2_15, rt2_15, 0.0),
        ("diag_fr_1.5", rt2_15, -rt2_15, 0.0),
        ("diag_bl_1.5", -rt2_15, rt2_15, 0.0),
        ("diag_br_1.5", -rt2_15, -rt2_15, 0.0),
        ("yaw_l_0.75", 0.0, 0.0, 0.75),
        ("yaw_l_1.5", 0.0, 0.0, 1.5),
        ("yaw_r_0.75", 0.0, 0.0, -0.75),
        ("yaw_r_1.5", 0.0, 0.0, -1.5),
        ("fwd1.0_yaw_l0.75", 1.0, 0.0, 0.75),
        ("fwd1.0_yaw_r0.75", 1.0, 0.0, -0.75),
        ("left1.0_yaw_l0.75", 0.0, 1.0, 0.75),
        ("right1.0_yaw_r0.75", 0.0, -1.0, -0.75),
        ("diag_fl1.0_yaw_l0.75", rt2_1, rt2_1, 0.75),
        ("diag_br1.0_yaw_r0.75", -rt2_1, -rt2_1, -0.75),
    ]


def reset_env(env):
    ids = torch.arange(env.num_envs, device=env.device)
    env._reset_idx(ids)
    env.cmd_resample_time[:] = 10**9
    env.time_out_buf[:] = False
    env.reset_buf[:] = False


def apply_command(env, cmd, gait_frequency):
    env.commands[:, 0] = cmd[0]
    env.commands[:, 1] = cmd[1]
    env.commands[:, 2] = cmd[2]
    if abs(cmd[0]) + abs(cmd[1]) + abs(cmd[2]) < 1.0e-8:
        env.gait_frequency[:] = 0.0
    else:
        env.gait_frequency[:] = gait_frequency
    env.cmd_resample_time[:] = 10**9
    env._compute_observations()
    return env.obs_buf


def summarize_command(env, model, name, cmd, steps, warmup_steps, gait_frequency):
    reset_env(env)
    obs = apply_command(env, cmd, gait_frequency).to(model.logstd.device)
    cmd_tensor = torch.tensor(cmd, device=env.device, dtype=torch.float)
    xy_cmd_norm = math.hypot(cmd[0], cmd[1])
    ever_done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    done_count = 0
    samples = 0
    accum = {
        "vx": 0.0,
        "vy": 0.0,
        "wz": 0.0,
        "abs_vx_err": 0.0,
        "abs_vy_err": 0.0,
        "xy_err": 0.0,
        "yaw_err": 0.0,
        "along": 0.0,
        "orth_abs": 0.0,
        "base_height": 0.0,
        "tilt": 0.0,
        "action_abs": 0.0,
        "torque_abs": 0.0,
        "reward": 0.0,
    }
    max_xy_err = 0.0
    max_yaw_err = 0.0

    with torch.no_grad():
        for step in range(steps):
            dist = model.act(obs)
            action = dist.loc
            obs, rew, done, _ = env.step(action)
            obs = obs.to(model.logstd.device)
            done = done.to(env.device)
            if done.any():
                ever_done |= done
                done_count += int(done.sum().item())
            obs = apply_command(env, cmd, gait_frequency).to(model.logstd.device)

            if step < warmup_steps:
                continue

            lin = env.filtered_lin_vel[:, :2]
            yaw = env.filtered_ang_vel[:, 2]
            xy_err = torch.linalg.norm(lin - cmd_tensor[:2], dim=1)
            yaw_err = torch.abs(yaw - cmd_tensor[2])
            accum["vx"] += float(lin[:, 0].mean().item())
            accum["vy"] += float(lin[:, 1].mean().item())
            accum["wz"] += float(yaw.mean().item())
            accum["abs_vx_err"] += float(torch.abs(lin[:, 0] - cmd_tensor[0]).mean().item())
            accum["abs_vy_err"] += float(torch.abs(lin[:, 1] - cmd_tensor[1]).mean().item())
            accum["xy_err"] += float(xy_err.mean().item())
            accum["yaw_err"] += float(yaw_err.mean().item())
            if xy_cmd_norm > 1.0e-6:
                unit = cmd_tensor[:2] / xy_cmd_norm
                along = (lin * unit).sum(dim=1)
                orth = lin[:, 0] * unit[1] - lin[:, 1] * unit[0]
                accum["along"] += float(along.mean().item())
                accum["orth_abs"] += float(torch.abs(orth).mean().item())
            accum["base_height"] += float((env.base_pos[:, 2] - env.terrain.terrain_heights(env.base_pos)).mean().item())
            accum["tilt"] += float(torch.linalg.norm(env.projected_gravity[:, :2], dim=1).mean().item())
            accum["action_abs"] += float(torch.abs(env.actions).mean().item())
            accum["torque_abs"] += float(torch.abs(env.torques).mean().item())
            accum["reward"] += float(rew.mean().item())
            max_xy_err = max(max_xy_err, float(xy_err.max().item()))
            max_yaw_err = max(max_yaw_err, float(yaw_err.max().item()))
            samples += 1

    out = {
        "name": name,
        "cmd_vx": cmd[0],
        "cmd_vy": cmd[1],
        "cmd_wz": cmd[2],
        "cmd_xy_speed": xy_cmd_norm,
        "steps": steps,
        "warmup_steps": warmup_steps,
        "duration_s": steps * env.dt,
        "warmup_s": warmup_steps * env.dt,
        "envs": env.num_envs,
        "reset_events_per_env": done_count / env.num_envs,
        "ever_reset_frac": float(ever_done.float().mean().item()),
        "max_xy_err": max_xy_err,
        "max_yaw_err": max_yaw_err,
    }
    for key, value in accum.items():
        out[key] = value / max(1, samples)
    if xy_cmd_norm > 1.0e-6:
        out["speed_ratio"] = out["along"] / xy_cmd_norm
    else:
        out["speed_ratio"] = None
    out["score_tracking"] = math.exp(-(out["xy_err"] ** 2) / 0.15) * math.exp(-(out["yaw_err"] ** 2) / 0.1)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="T1CircleGridFace6Tight15")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--duration_s", type=float, default=8.0)
    parser.add_argument("--warmup_s", type=float, default=2.0)
    parser.add_argument("--gait_frequency", type=float, default=1.5)
    parser.add_argument("--sim_device", default="cuda:0")
    parser.add_argument("--rl_device", default="cuda:0")
    parser.add_argument("--out_dir", default="artifacts/autoresearch/evals")
    parser.add_argument("--label", default=None)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--max_commands", type=int, default=None)
    args = parser.parse_args()

    set_seed(1234)
    cfg = load_cfg(args.task, args.checkpoint, args.num_envs, True, args.sim_device, args.rl_device, args.clean)
    print(f"Preparing env task={args.task} checkpoint={args.checkpoint} envs={args.num_envs}", flush=True)
    task_class = eval(cfg["basic"]["task"])
    env = task_class(cfg)
    device = cfg["basic"]["rl_device"]
    print(f"Environment ready dt={env.dt}", flush=True)
    model = ActorCritic(env.num_actions, env.num_obs, env.num_privileged_obs).to(device)
    model_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(model_dict["model"], strict=False)
    model.eval()
    print("Model loaded", flush=True)

    obs, _ = env.reset()
    del obs
    steps = int(round(args.duration_s / env.dt))
    warmup_steps = int(round(args.warmup_s / env.dt))
    started = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    label = args.label or Path(args.checkpoint).stem
    mode = "clean" if args.clean else "default"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    suite = command_suite()
    if args.max_commands is not None:
        suite = suite[: args.max_commands]
    for name, vx, vy, wz in suite:
        row = summarize_command(env, model, name, (vx, vy, wz), steps, warmup_steps, args.gait_frequency)
        rows.append(row)
        print(
            f"{name:24s} cmd=({vx:+.3f},{vy:+.3f},{wz:+.3f}) "
            f"actual=({row['vx']:+.3f},{row['vy']:+.3f},{row['wz']:+.3f}) "
            f"xy_err={row['xy_err']:.3f} yaw_err={row['yaw_err']:.3f} "
            f"reset={row['ever_reset_frac']:.3f} ratio={row['speed_ratio'] if row['speed_ratio'] is not None else float('nan'):.3f}",
            flush=True,
        )

    payload = {
        "label": label,
        "checkpoint": args.checkpoint,
        "task": args.task,
        "mode": mode,
        "num_envs": args.num_envs,
        "duration_s": args.duration_s,
        "warmup_s": args.warmup_s,
        "gait_frequency": args.gait_frequency,
        "dt": env.dt,
        "started_at": started,
        "rows": rows,
        "summary": {
            "command_count": len(rows),
            "mean_xy_err": sum(row["xy_err"] for row in rows) / max(1, len(rows)),
            "mean_yaw_err": sum(row["yaw_err"] for row in rows) / max(1, len(rows)),
            "max_reset_frac": max((row["ever_reset_frac"] for row in rows), default=0.0),
            "mean_speed_ratio": sum(row["speed_ratio"] for row in rows if row["speed_ratio"] is not None)
            / max(1, sum(1 for row in rows if row["speed_ratio"] is not None)),
        },
    }
    json_path = out_dir / f"{label}_{mode}_{started}.json"
    csv_path = out_dir / f"{label}_{mode}_{started}.csv"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"JSON {json_path}")
    print(f"CSV {csv_path}")


if __name__ == "__main__":
    main()
