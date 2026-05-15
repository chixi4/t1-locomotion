#!/usr/bin/env python3
import argparse
import json
import math
import random
from pathlib import Path

import isaacgym  # noqa: F401
import numpy as np
import torch
import yaml

from envs import *  # noqa: F401,F403
from utils.model import ActorCritic


def zero_randomization_tree(tree):
    if not isinstance(tree, dict):
        return
    if "range" in tree and isinstance(tree["range"], list) and len(tree["range"]) == 2:
        tree["range"] = [1.0, 1.0] if tree.get("operation") == "scaling" else [0.0, 0.0]
    for value in tree.values():
        zero_randomization_tree(value)


def load_cfg(task, num_envs, seconds):
    cfg = yaml.safe_load((Path("envs") / f"{task}.yaml").read_text(encoding="utf-8"))
    cfg["basic"]["task"] = task
    cfg["basic"]["headless"] = True
    cfg["basic"]["sim_device"] = "cuda:0"
    cfg["basic"]["rl_device"] = "cuda:0"
    cfg["basic"]["seed"] = 20260514
    cfg["env"]["num_envs"] = num_envs
    cfg["viewer"]["record_video"] = False
    cfg["runner"]["use_wandb"] = False
    cfg["commands"]["curriculum"] = False
    cfg["commands"]["still_proportion"] = 0.0
    cfg["commands"]["resampling_time_s"] = [1.0e6, 1.0e6 + 1.0]
    cfg["rewards"]["episode_length_s"] = seconds + 5.0
    cfg["terrain"]["type"] = "plane"
    zero_randomization_tree(cfg.get("noise", {}))
    zero_randomization_tree(cfg.get("randomization", {}))
    randomization = cfg.get("randomization", {})
    randomization["kick_interval_s"] = 1.0e9
    randomization["push_interval_s"] = 1.0e9
    randomization["push_duration_s"] = 0.0
    return cfg


def actor_mean_scale_from_cfg(env, cfg):
    by_dof = cfg["algorithm"].get("actor_mean_scale_by_dof")
    if by_dof:
        return [float(by_dof[env.dof_names[dof_idx]]) for dof_idx in env.arm_indices.tolist()]
    return cfg["algorithm"].get("actor_mean_scale")


def load_models(env, cfg, arm_checkpoint, leg_checkpoint):
    device = cfg["basic"]["rl_device"]
    arm = ActorCritic(
        len(env.arm_indices),
        env.num_obs,
        env.num_privileged_obs,
        logstd_init=float(cfg["algorithm"].get("logstd_init", -2.0)),
        actor_mean_scale=actor_mean_scale_from_cfg(env, cfg),
        logstd_min=cfg["algorithm"].get("logstd_min"),
        logstd_max=cfg["algorithm"].get("logstd_max"),
    ).to(device)
    arm.load_state_dict(torch.load(arm_checkpoint, map_location=device, weights_only=True)["model"], strict=False)
    arm.eval()
    leg = ActorCritic(len(env.leg_indices), 47, env.num_privileged_obs).to(device)
    leg.load_state_dict(torch.load(leg_checkpoint, map_location=device, weights_only=True)["model"], strict=True)
    leg.eval()
    return arm, leg


def build_old_leg_obs(env, last_leg_action):
    scale = torch.tensor(
        [env.cfg["normalization"]["lin_vel"], env.cfg["normalization"]["lin_vel"], env.cfg["normalization"]["ang_vel"]],
        device=env.device,
    )
    gait_active = (env.gait_frequency > 1.0e-8).float()
    return torch.cat(
        (
            env.projected_gravity * env.cfg["normalization"]["gravity"],
            env.base_ang_vel * env.cfg["normalization"]["ang_vel"],
            env.commands[:, :3] * scale,
            (torch.cos(2 * torch.pi * env.gait_process) * gait_active).unsqueeze(-1),
            (torch.sin(2 * torch.pi * env.gait_process) * gait_active).unsqueeze(-1),
            (env.dof_pos[:, env.leg_indices] - env.default_dof_pos[:, env.leg_indices]) * env.cfg["normalization"]["dof_pos"],
            env.dof_vel[:, env.leg_indices] * env.cfg["normalization"]["dof_vel"],
            last_leg_action,
        ),
        dim=-1,
    )


def policy_step(env, arm, leg, last_leg_action):
    old_leg_obs = build_old_leg_obs(env, last_leg_action)
    leg_action = torch.clamp(leg.act(old_leg_obs).loc, -1.0, 1.0)
    arm_action = torch.clamp(arm.act(env.obs_buf).loc, -1.0, 1.0)
    full = torch.zeros(env.num_envs, env.num_actions, dtype=torch.float, device=env.device)
    full[:, env.arm_indices] = arm_action
    full[:, env.leg_indices] = leg_action
    return full, arm_action, leg_action


def apply_command(env, cmd, gait_frequency):
    env.commands[:, 0] = cmd[0]
    env.commands[:, 1] = cmd[1]
    env.commands[:, 2] = cmd[2]
    env.gait_frequency[:] = 0.0 if sum(abs(v) for v in cmd) < 1.0e-8 else gait_frequency
    env.cmd_resample_time[:] = 10**9
    env._compute_observations()


def pct(values, q):
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    return float(np.percentile(arr, q))


def corr(a, b):
    if len(a) < 2:
        return 0.0
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if float(np.std(a)) < 1.0e-8 or float(np.std(b)) < 1.0e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def command_suite():
    return [
        ("stand", (0.0, 0.0, 0.0)),
        ("forward_1.0", (1.0, 0.0, 0.0)),
        ("forward_1.5", (1.5, 0.0, 0.0)),
        ("backward_1.0", (-1.0, 0.0, 0.0)),
        ("backward_1.5", (-1.5, 0.0, 0.0)),
        ("left_1.0", (0.0, 1.0, 0.0)),
        ("left_1.5", (0.0, 1.5, 0.0)),
        ("right_1.0", (0.0, -1.0, 0.0)),
        ("right_1.5", (0.0, -1.5, 0.0)),
        ("yaw_l_1.0", (0.0, 0.0, 1.0)),
        ("yaw_r_1.0", (0.0, 0.0, -1.0)),
        ("diag_fl_1.0", (math.sqrt(0.5), math.sqrt(0.5), 0.0)),
        ("diag_fr_1.0", (math.sqrt(0.5), -math.sqrt(0.5), 0.0)),
    ]


def eval_checkpoint(task, checkpoint, leg_checkpoint, num_envs, seconds, warmup_s):
    random.seed(20260514)
    np.random.seed(20260514)
    torch.manual_seed(20260514)
    torch.cuda.manual_seed_all(20260514)
    cfg = load_cfg(task, num_envs, seconds)
    env = eval(task)(cfg)
    arm, leg = load_models(env, cfg, checkpoint, leg_checkpoint)
    steps = int(seconds / env.dt)
    warmup = int(warmup_s / env.dt)
    all_rows = []
    all_resets = 0
    for name, cmd in command_suite():
        env.reset()
        apply_command(env, cmd, 1.5)
        last_leg_action = torch.zeros(env.num_envs, len(env.leg_indices), dtype=torch.float, device=env.device)
        rows = []
        done_count = 0
        with torch.no_grad():
            for step in range(steps):
                full, arm_action, leg_action = policy_step(env, arm, leg, last_leg_action)
                _, _, done, _ = env.step(full)
                apply_command(env, cmd, 1.5)
                last_leg_action[:] = leg_action
                last_leg_action[done] = 0.0
                done_count += int(done.sum().item())
                if step < warmup:
                    continue
                q = (env.dof_pos[:, env.arm_indices] - env.default_dof_pos[:, env.arm_indices]).detach().cpu().numpy()
                feet = env.feet_pos_body.detach().cpu().numpy()
                rows.append(
                    {
                        "cmd": cmd,
                        "tilt": torch.linalg.norm(env.projected_gravity[:, :2], dim=1).detach().cpu().numpy(),
                        "ang_xy": torch.linalg.norm(env.base_ang_vel[:, :2], dim=1).detach().cpu().numpy(),
                        "xy_err": torch.linalg.norm(env.filtered_lin_vel[:, :2] - torch.tensor(cmd[:2], device=env.device), dim=1).detach().cpu().numpy(),
                        "yaw_err": torch.abs(env.filtered_ang_vel[:, 2] - cmd[2]).detach().cpu().numpy(),
                        "q": q,
                        "feet": feet,
                        "arm_sat": (torch.abs(arm_action) > 0.98).float().detach().cpu().numpy(),
                    }
                )
        all_resets += done_count
        all_rows.append({"name": name, "cmd": cmd, "done_count": done_count, "rows": rows})

    def flatten(selector, predicate=lambda item: True):
        out = []
        for item in all_rows:
            if not predicate(item):
                continue
            for row in item["rows"]:
                value = selector(row, item)
                out.extend(np.asarray(value).reshape(-1).tolist())
        return out

    stand = lambda item: item["name"] == "stand"
    sagittal = lambda item: abs(item["cmd"][0]) > 0.4 and abs(item["cmd"][1]) < 0.2
    lateral = lambda item: abs(item["cmd"][1]) > 0.4 and abs(item["cmd"][0]) < 0.2
    moving = lambda item: item["name"] != "stand"

    left_pitch = flatten(lambda row, item: row["q"][:, 0], sagittal)
    right_pitch = flatten(lambda row, item: row["q"][:, 2], sagittal)
    foot_sag = flatten(lambda row, item: row["feet"][:, 0, 0] - row["feet"][:, 1, 0], sagittal)
    left_roll_side = flatten(lambda row, item: row["q"][:, 1], lateral)
    right_roll_side = flatten(lambda row, item: row["q"][:, 3], lateral)
    left_cmd_rows = [item for item in all_rows if lateral(item) and item["cmd"][1] > 0]
    right_cmd_rows = [item for item in all_rows if lateral(item) and item["cmd"][1] < 0]
    left_cmd_right_arm = []
    right_cmd_left_arm = []
    for item in left_cmd_rows:
        for row in item["rows"]:
            left_cmd_right_arm.extend((-row["q"][:, 3]).reshape(-1).tolist())
    for item in right_cmd_rows:
        for row in item["rows"]:
            right_cmd_left_arm.extend(row["q"][:, 1].reshape(-1).tolist())

    result = {
        "checkpoint": checkpoint,
        "num_envs": num_envs,
        "seconds_per_command": seconds,
        "reset_events_per_env": all_resets / float(num_envs),
        "lin_error_mean": float(np.mean(flatten(lambda row, item: row["xy_err"], moving))),
        "yaw_error_mean": float(np.mean(flatten(lambda row, item: row["yaw_err"], moving))),
        "camera_tilt_p95_all": pct(flatten(lambda row, item: row["tilt"]), 95),
        "camera_tilt_p95_moving": pct(flatten(lambda row, item: row["tilt"], moving), 95),
        "camera_ang_xy_rms_moving": float(np.sqrt(np.mean(np.square(flatten(lambda row, item: row["ang_xy"], moving))))),
        "stand_shoulder_abs_p95": pct(flatten(lambda row, item: np.abs(row["q"][:, :4]), stand), 95),
        "stand_elbow_abs_p95": pct(flatten(lambda row, item: np.abs(row["q"][:, 4:8]), stand), 95),
        "stand_waist_abs_p95": pct(flatten(lambda row, item: np.abs(row["q"][:, 8]), stand), 95),
        "pitch_abs_p95_sagittal": pct([abs(x) for x in left_pitch + right_pitch], 95),
        "pitch_lr_antisym_corr": corr(left_pitch, [-x for x in right_pitch]),
        "pitch_left_foot_corr": corr(left_pitch, foot_sag),
        "pitch_common_abs_p95": pct([abs(a + b) for a, b in zip(left_pitch, right_pitch)], 95),
        "side_left_arm_outward_p95_on_right_cmd": pct(right_cmd_left_arm, 95),
        "side_right_arm_outward_p95_on_left_cmd": pct(left_cmd_right_arm, 95),
        "side_outward_abs_p95": pct([max(0.0, x) for x in left_roll_side] + [max(0.0, -x) for x in right_roll_side], 95),
        "side_correct_arm_mean": float(np.mean([x for x in right_cmd_left_arm + left_cmd_right_arm])) if right_cmd_left_arm or left_cmd_right_arm else 0.0,
        "elbow_pitch_abs_p95_moving": pct(flatten(lambda row, item: np.abs(row["q"][:, [4, 6]]), moving), 95),
        "elbow_yaw_abs_p95_moving": pct(flatten(lambda row, item: np.abs(row["q"][:, [5, 7]]), moving), 95),
        "waist_abs_p95_moving": pct(flatten(lambda row, item: np.abs(row["q"][:, 8]), moving), 95),
        "arm_action_saturation_frac": float(np.mean(flatten(lambda row, item: row["arm_sat"]))),
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--leg-checkpoint", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--warmup-s", type=float, default=2.0)
    args = parser.parse_args()
    payload = {
        "task": args.task,
        "leg_checkpoint": args.leg_checkpoint,
        "results": [
            eval_checkpoint(args.task, ckpt, args.leg_checkpoint, args.num_envs, args.seconds, args.warmup_s)
            for ckpt in args.checkpoints
        ],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
