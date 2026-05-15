import argparse
import json
import os
import random
import sys

import numpy as np
import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from isaacgym import gymtorch, gymapi
assert gymtorch and gymapi

import torch

import envs
from utils.model import ActorCritic


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_old_leg_obs(env, last_leg_action):
    commands_scale = torch.tensor(
        [
            env.cfg["normalization"]["lin_vel"],
            env.cfg["normalization"]["lin_vel"],
            env.cfg["normalization"]["ang_vel"],
        ],
        device=env.device,
    )
    gait_active = (env.gait_frequency > 1.0e-8).float()
    return torch.cat(
        (
            env.projected_gravity * env.cfg["normalization"]["gravity"],
            env.base_ang_vel * env.cfg["normalization"]["ang_vel"],
            env.commands[:, :3] * commands_scale,
            (torch.cos(2 * torch.pi * env.gait_process) * gait_active).unsqueeze(-1),
            (torch.sin(2 * torch.pi * env.gait_process) * gait_active).unsqueeze(-1),
            (env.dof_pos[:, env.leg_indices] - env.default_dof_pos[:, env.leg_indices]) * env.cfg["normalization"]["dof_pos"],
            env.dof_vel[:, env.leg_indices] * env.cfg["normalization"]["dof_vel"],
            last_leg_action,
        ),
        dim=-1,
    )


def load_cfg(task, num_envs, seed):
    cfg_path = os.path.join("envs", f"{task}.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    cfg["env"]["num_envs"] = num_envs
    cfg["basic"]["headless"] = True
    cfg["basic"]["seed"] = seed
    cfg["viewer"]["record_video"] = False
    return cfg


def apply_levels(env, levels):
    lin_res = float(env.cfg["commands"].get("linear_speed_resolution", 0.1))
    yaw_res = float(env.cfg["commands"].get("ang_vel_resolution", 0.1))
    env.env_curriculum_level[:] = levels
    env.commands[:, 0] = levels[:, 0].float() * lin_res
    env.commands[:, 1] = levels[:, 1].float() * lin_res
    env.commands[:, 2] = torch.clamp(
        levels[:, 2].float() * yaw_res,
        float(env.cfg["commands"]["ang_vel_yaw"][0]),
        float(env.cfg["commands"]["ang_vel_yaw"][1]),
    )
    env.gait_frequency[:] = 0.5 * (float(env.cfg["commands"]["gait_frequency"][0]) + float(env.cfg["commands"]["gait_frequency"][1]))
    env.cmd_resample_time[:] = 10**9


def sway_score(env, prev_tilt, tilt_weight, ang_vel_weight, tilt_rate_weight):
    tilt = torch.sqrt(torch.sum(torch.square(env.projected_gravity[:, :2]), dim=1))
    tilt_rate = (tilt - prev_tilt) / env.dt
    ang_vel_xy = torch.linalg.norm(env.base_ang_vel[:, :2], dim=1)
    score = tilt_weight * tilt.square() + ang_vel_weight * ang_vel_xy.square() + tilt_rate_weight * tilt_rate.square()
    return torch.nan_to_num(score, nan=0.0, posinf=1.0e6, neginf=0.0), tilt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="T1Shoulder4AntiSwayBaseline_from7000LegFrozen_train1000")
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--warmup_s", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", default="artifacts/autoresearch/fixed_arm_sway_baseline_model7000.pt")
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = load_cfg(args.task, args.num_envs, args.seed)
    task_class = getattr(envs, args.task)
    env = task_class(cfg)
    device = env.device

    leg_model = ActorCritic(len(env.leg_indices), 47, env.num_privileged_obs).to(device)
    checkpoint = torch.load(cfg["basic"]["leg_checkpoint"], map_location=device, weights_only=True)
    leg_model.load_state_dict(checkpoint["model"], strict=True)
    leg_model.eval()

    mask = env.curriculum_mask.bool()
    cells = mask.nonzero(as_tuple=False)
    lin_levels = int(cfg["commands"]["lin_vel_levels"])
    yaw_levels = int(cfg["commands"]["ang_vel_levels"])
    offsets = torch.tensor([lin_levels, lin_levels, yaw_levels], dtype=torch.long, device=device)
    baseline = torch.zeros(mask.shape, dtype=torch.float, device=device)
    counts = torch.zeros(mask.shape, dtype=torch.float, device=device)

    tilt_weight = float(cfg["rewards"].get("sway_tilt_weight", 1.0))
    ang_vel_weight = float(cfg["rewards"].get("sway_ang_vel_xy_weight", 0.25))
    tilt_rate_weight = float(cfg["rewards"].get("sway_tilt_rate_weight", 0.01))
    steps = int(np.ceil(args.seconds / env.dt))
    warmup_steps = int(np.ceil(args.warmup_s / env.dt))
    env_ids = torch.arange(env.num_envs, device=device)
    last_leg_action = torch.zeros(env.num_envs, len(env.leg_indices), dtype=torch.float, device=device)

    with torch.no_grad():
        for start in range(0, cells.shape[0], env.num_envs):
            batch_cells = cells[start : start + env.num_envs]
            active = batch_cells.shape[0]
            full_cells = cells.new_empty((env.num_envs, 3))
            full_cells[:active] = batch_cells
            if active < env.num_envs:
                full_cells[active:] = batch_cells[0]
            levels = full_cells - offsets

            env._reset_idx(env_ids)
            apply_levels(env, levels)
            last_leg_action.zero_()
            score_sum = torch.zeros(env.num_envs, dtype=torch.float, device=device)
            score_count = torch.zeros(env.num_envs, dtype=torch.float, device=device)
            prev_tilt = torch.sqrt(torch.sum(torch.square(env.projected_gravity[:, :2]), dim=1))

            for step in range(steps):
                leg_obs = build_old_leg_obs(env, last_leg_action)
                leg_action = torch.clamp(leg_model.act(leg_obs).loc, -1.0, 1.0)
                full_action = torch.zeros(env.num_envs, env.num_actions, dtype=torch.float, device=device)
                full_action[:, env.leg_indices] = leg_action
                _, _, done, _ = env.step(full_action)
                apply_levels(env, levels)
                score, prev_tilt = sway_score(env, prev_tilt, tilt_weight, ang_vel_weight, tilt_rate_weight)
                if step >= warmup_steps:
                    score_sum += score
                    score_count += 1.0
                last_leg_action[:] = leg_action
                last_leg_action[done] = 0.0

            mean_score = score_sum[:active] / torch.clamp(score_count[:active], min=1.0)
            baseline[batch_cells[:, 0], batch_cells[:, 1], batch_cells[:, 2]] = mean_score
            counts[batch_cells[:, 0], batch_cells[:, 1], batch_cells[:, 2]] = score_count[:active]
            print(f"baseline cells {min(start + active, cells.shape[0])}/{cells.shape[0]}", flush=True)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    payload = {
        "sway": baseline.cpu(),
        "mask": mask.cpu(),
        "counts": counts.cpu(),
        "weights": {
            "tilt": tilt_weight,
            "ang_vel_xy": ang_vel_weight,
            "tilt_rate": tilt_rate_weight,
        },
        "task": args.task,
        "seconds": args.seconds,
        "warmup_s": args.warmup_s,
        "leg_checkpoint": cfg["basic"]["leg_checkpoint"],
    }
    torch.save(payload, args.output)
    values = baseline[mask].detach().cpu()
    summary = {
        "cells": int(mask.sum().item()),
        "seconds": args.seconds,
        "warmup_s": args.warmup_s,
        "mean": float(values.mean().item()),
        "p50": float(values.quantile(0.50).item()),
        "p95": float(values.quantile(0.95).item()),
        "max": float(values.max().item()),
        "output": args.output,
    }
    with open(args.output + ".json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
