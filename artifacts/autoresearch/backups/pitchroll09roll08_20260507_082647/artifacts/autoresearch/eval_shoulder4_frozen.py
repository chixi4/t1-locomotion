import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import isaacgym  # noqa: F401
from isaacgym import gymtorch
import numpy as np
import torch
import yaml


sys.path.insert(0, os.getcwd())

from envs import *  # noqa: F401,F403
from utils.model import ActorCritic


BODY_NAMES_PREFERRED = [
    "Trunk",
    "AL1",
    "AL2",
    "AR1",
    "AR2",
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


def load_cfg(task, num_envs, seed, clean, seconds):
    with open(Path("envs") / f"{task}.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    cfg["basic"]["task"] = task
    cfg["basic"]["headless"] = True
    cfg["basic"]["sim_device"] = "cuda:0"
    cfg["basic"]["rl_device"] = "cuda:0"
    cfg["basic"]["seed"] = seed
    cfg["env"]["num_envs"] = num_envs
    cfg["viewer"]["record_video"] = False
    cfg["runner"]["use_wandb"] = False
    cfg["commands"]["curriculum"] = False
    cfg["commands"]["sway_curriculum"] = False
    cfg["commands"]["still_proportion"] = 0.0
    cfg["commands"]["resampling_time_s"] = [1.0e6, 1.0e6 + 1.0]
    cfg["rewards"]["episode_length_s"] = seconds + 5.0
    if clean:
        cfg["terrain"]["type"] = "plane"
        zero_randomization_tree(cfg.get("noise", {}))
        zero_randomization_tree(cfg.get("randomization", {}))
        randomization = cfg.get("randomization", {})
        randomization["kick_interval_s"] = 1.0e9
        randomization["push_interval_s"] = 1.0e9
        randomization["push_duration_s"] = 0.0
        for key in ["compliance", "restitution"]:
            if key in randomization and isinstance(randomization[key], dict):
                randomization[key]["range"] = [0.0, 0.0]
        if "friction" in randomization and isinstance(randomization["friction"], dict):
            randomization["friction"]["range"] = [1.0, 1.0]
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


def load_models(env, arm_checkpoint, leg_checkpoint, device, logstd_init, actor_mean_scale=None, logstd_min=None, logstd_max=None):
    arm_model = ActorCritic(
        len(env.arm_indices),
        env.num_obs,
        env.num_privileged_obs,
        logstd_init=logstd_init,
        actor_mean_scale=actor_mean_scale,
        logstd_min=logstd_min,
        logstd_max=logstd_max,
    ).to(device)
    arm_state = torch.load(arm_checkpoint, map_location=device, weights_only=True)
    arm_model.load_state_dict(arm_state["model"], strict=False)
    arm_model.eval()

    leg_model = ActorCritic(len(env.leg_indices), 47, env.num_privileged_obs).to(device)
    leg_state = torch.load(leg_checkpoint, map_location=device, weights_only=True)
    leg_model.load_state_dict(leg_state["model"], strict=True)
    leg_model.eval()
    return arm_model, leg_model


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
            (env.dof_pos[:, env.leg_indices] - env.default_dof_pos[:, env.leg_indices])
            * env.cfg["normalization"]["dof_pos"],
            env.dof_vel[:, env.leg_indices] * env.cfg["normalization"]["dof_vel"],
            last_leg_action,
        ),
        dim=-1,
    )


def policy_step(env, arm_model, leg_model, obs, last_leg_action):
    with torch.no_grad():
        old_leg_obs = build_old_leg_obs(env, last_leg_action)
        leg_action = torch.clamp(leg_model.act(old_leg_obs).loc, -1.0, 1.0)
        arm_action = torch.clamp(arm_model.act(obs).loc, -1.0, 1.0)
        full_action = torch.zeros(env.num_envs, env.num_actions, dtype=torch.float, device=env.device)
        full_action[:, env.arm_indices] = arm_action
        full_action[:, env.leg_indices] = leg_action
    return full_action, arm_action, leg_action


def reset_env(env):
    ids = torch.arange(env.num_envs, device=env.device)
    env._reset_idx(ids)
    env.cmd_resample_time[:] = 10**9
    env.time_out_buf[:] = False
    env.reset_buf[:] = False


def clean_start(env):
    env.dof_pos[:] = env.default_dof_pos
    env.dof_vel.zero_()
    env.root_states[:, :3] = torch.tensor([0.0, 0.0, 0.72], dtype=torch.float, device=env.device)
    env.root_states[:, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float, device=env.device)
    env.root_states[:, 7:13] = 0.0
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


def apply_command(env, cmd, gait_frequency):
    env.commands[:, 0] = cmd[0]
    env.commands[:, 1] = cmd[1]
    env.commands[:, 2] = cmd[2]
    env.gait_frequency[:] = 0.0 if sum(abs(v) for v in cmd) < 1.0e-8 else gait_frequency
    env.cmd_resample_time[:] = 10**9
    env._compute_observations()
    return env.obs_buf


def percentile(values, p):
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(round((len(values) - 1) * p))))
    return values[idx]


def summarize_values(values):
    if not values:
        return {"mean": 0.0, "rms": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(sum(values) / len(values)),
        "rms": float(math.sqrt(sum(v * v for v in values) / len(values))),
        "p95": float(percentile(values, 0.95)),
        "max": float(max(values)),
    }


def collect_sample(env, arm_action, cmd):
    cmd_tensor = torch.tensor(cmd, device=env.device, dtype=torch.float)
    lin = env.filtered_lin_vel[:, :2]
    yaw = env.filtered_ang_vel[:, 2]
    xy_err = torch.linalg.norm(lin - cmd_tensor[:2], dim=1)
    yaw_err = torch.abs(yaw - cmd_tensor[2])
    tilt = torch.linalg.norm(env.projected_gravity[:, :2], dim=1)
    base_ang_xy = torch.linalg.norm(env.base_ang_vel[:, :2], dim=1)
    q = env.dof_pos[:, env.arm_indices] - env.default_dof_pos[:, env.arm_indices]
    pitch_abs = torch.abs(q[:, [0, 2]]).reshape(-1)
    roll_abs = torch.abs(q[:, [1, 3]]).reshape(-1)
    sat = (torch.abs(arm_action) > 0.98).float().reshape(-1)
    xy_cmd_norm = math.hypot(cmd[0], cmd[1])
    if xy_cmd_norm > 1.0e-6:
        unit = cmd_tensor[:2] / xy_cmd_norm
        along = (lin * unit).sum(dim=1)
        speed_ratio = float((along / xy_cmd_norm).mean().item())
    else:
        speed_ratio = None
    return {
        "xy_err": xy_err.detach().cpu().tolist(),
        "yaw_err": yaw_err.detach().cpu().tolist(),
        "tilt": tilt.detach().cpu().tolist(),
        "base_ang_vel_xy": base_ang_xy.detach().cpu().tolist(),
        "shoulder_pitch_abs": pitch_abs.detach().cpu().tolist(),
        "shoulder_roll_abs": roll_abs.detach().cpu().tolist(),
        "arm_action_saturation": sat.detach().cpu().tolist(),
        "speed_ratio": speed_ratio,
        "vx": float(lin[:, 0].mean().item()),
        "vy": float(lin[:, 1].mean().item()),
        "wz": float(yaw.mean().item()),
    }


def summarize_command(env, arm_model, leg_model, name, cmd, steps, warmup_steps, gait_frequency):
    reset_env(env)
    obs = apply_command(env, cmd, gait_frequency).to(env.device)
    last_leg_action = torch.zeros(env.num_envs, len(env.leg_indices), dtype=torch.float, device=env.device)
    ever_done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    done_count = 0
    samples = []
    with torch.no_grad():
        for step in range(steps):
            full_action, arm_action, leg_action = policy_step(env, arm_model, leg_model, obs, last_leg_action)
            obs, _, done, _ = env.step(full_action)
            obs = apply_command(env, cmd, gait_frequency).to(env.device)
            last_leg_action[:] = leg_action
            last_leg_action[done] = 0.0
            if done.any():
                ever_done |= done
                done_count += int(done.sum().item())
            if step >= warmup_steps:
                samples.append(collect_sample(env, arm_action, cmd))

    def flat(key):
        out = []
        for sample in samples:
            out.extend(sample[key])
        return out

    speed_values = [sample["speed_ratio"] for sample in samples if sample["speed_ratio"] is not None]
    row = {
        "name": name,
        "cmd_vx": cmd[0],
        "cmd_vy": cmd[1],
        "cmd_wz": cmd[2],
        "cmd_xy_speed": math.hypot(cmd[0], cmd[1]),
        "steps": steps,
        "warmup_steps": warmup_steps,
        "envs": env.num_envs,
        "reset_events_per_env": done_count / env.num_envs,
        "reset_frac": float(ever_done.float().mean().item()),
        "actual_vx": sum(s["vx"] for s in samples) / max(1, len(samples)),
        "actual_vy": sum(s["vy"] for s in samples) / max(1, len(samples)),
        "actual_wz": sum(s["wz"] for s in samples) / max(1, len(samples)),
        "xy_err": summarize_values(flat("xy_err"))["mean"],
        "yaw_err": summarize_values(flat("yaw_err"))["mean"],
        "speed_ratio": sum(speed_values) / len(speed_values) if speed_values else None,
        "tilt_mean": summarize_values(flat("tilt"))["mean"],
        "tilt_rms": summarize_values(flat("tilt"))["rms"],
        "tilt_p95": summarize_values(flat("tilt"))["p95"],
        "tilt_max": summarize_values(flat("tilt"))["max"],
        "base_ang_vel_xy_rms": summarize_values(flat("base_ang_vel_xy"))["rms"],
        "shoulder_pitch_abs_p95": summarize_values(flat("shoulder_pitch_abs"))["p95"],
        "shoulder_roll_abs_p95": summarize_values(flat("shoulder_roll_abs"))["p95"],
        "arm_action_saturation_frac": summarize_values(flat("arm_action_saturation"))["mean"],
    }
    return row


def sample_target(max_speed, max_yaw):
    if random.random() < 0.12:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    speed = math.sqrt(random.random()) * max_speed
    theta = random.uniform(-math.pi, math.pi)
    yaw = random.uniform(-max_yaw, max_yaw)
    if random.random() < 0.2:
        yaw *= 0.35
    return np.array([speed * math.cos(theta), speed * math.sin(theta), yaw], dtype=np.float32)


def make_segments(seconds, max_speed, max_yaw):
    segments = []
    t = 0.0
    prev = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    first = min(2.0, seconds)
    segments.append({"start": 0.0, "end": first, "prev": prev.copy(), "target": prev.copy()})
    t = first
    while t < seconds - 1.0e-6:
        dur = random.uniform(2.4, 4.6)
        end = min(seconds, t + dur)
        target = sample_target(max_speed, max_yaw)
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


def actor_body_names(env):
    names = env.gym.get_actor_rigid_body_names(env.envs[0], env.actor_handles[0])
    preferred = [name for name in BODY_NAMES_PREFERRED if name in names]
    rest = [name for name in names if name not in preferred]
    return preferred + rest


def random_replay(env, arm_model, leg_model, seconds, warmup_s, gait_frequency, max_speed, max_yaw, smooth_s, fps, checkpoint, seed):
    env.reset()
    obs = clean_start(env).to(env.device)
    last_leg_action = torch.zeros(env.num_envs, len(env.leg_indices), dtype=torch.float, device=env.device)
    segments = make_segments(seconds, max_speed, max_yaw)
    rigid_names = env.gym.get_actor_rigid_body_names(env.envs[0], env.actor_handles[0])
    body_names = actor_body_names(env)
    indices = torch.tensor([rigid_names.index(name) for name in body_names], dtype=torch.long, device=env.device)
    frame_every = max(1, int(round((1.0 / fps) / env.dt)))
    total_steps = int(round(seconds / env.dt))
    warmup_steps = int(round(warmup_s / env.dt))
    frames = []
    reset_steps = []
    all_samples = []
    with torch.no_grad():
        for step in range(total_steps):
            t = step * env.dt
            cmd = command_at(segments, t, smooth_s)
            env.commands[:, :3] = torch.tensor(cmd, dtype=torch.float, device=env.device).view(1, 3)
            env.gait_frequency[:] = 0.0 if float(np.linalg.norm(cmd)) < 0.04 else gait_frequency
            env.cmd_resample_time[:] = 10**9
            env._compute_observations()
            obs = env.obs_buf.to(env.device)
            full_action, arm_action, leg_action = policy_step(env, arm_model, leg_model, obs, last_leg_action)
            obs, _, done, _ = env.step(full_action)
            last_leg_action[:] = leg_action
            last_leg_action[done] = 0.0
            env.commands[:, :3] = torch.tensor(cmd, dtype=torch.float, device=env.device).view(1, 3)
            env.cmd_resample_time[:] = 10**9
            if bool(done[0].item()):
                reset_steps.append(step)
            if step >= warmup_steps:
                sample = collect_sample(env, arm_action, tuple(float(x) for x in cmd))
                all_samples.append(sample)
            if step % frame_every == 0:
                bs = env.body_states[0, indices, :7].detach().cpu().numpy()
                p = np.round(bs[:, :3], 4).tolist()
                q_xyzw = bs[:, 3:7]
                q_wxyz = np.stack([q_xyzw[:, 3], q_xyzw[:, 0], q_xyzw[:, 1], q_xyzw[:, 2]], axis=1)
                q = np.round(q_wxyz, 5).tolist()
                frames.append({"p": p, "q": q, "c": np.round(cmd.astype(np.float32), 4).tolist()})

    def flat(key):
        out = []
        for sample in all_samples:
            out.extend(sample[key])
        return out

    lin_values = flat("xy_err")
    yaw_values = flat("yaw_err")
    replay = {
        "title": "T1Shoulder4 model_900 随机命令长回放",
        "stage": "t1shoulder4_random",
        "stageLabel": "T1Shoulder4 model_900 随机命令长回放",
        "command": {"vx": 0.0, "vy": 0.0, "yaw": 0.0},
        "fps": fps,
        "bodyNames": body_names,
        "frames": frames,
        "commandSegments": [
            {
                "start": round(float(seg["start"]), 3),
                "end": round(float(seg["end"]), 3),
                "target": np.round(seg["target"], 4).tolist(),
            }
            for seg in segments
        ],
        "meta": {
            "checkpoint": checkpoint,
            "seed": seed,
            "seconds": seconds,
            "policy_dt": env.dt,
            "frames": len(frames),
            "reset_steps": reset_steps,
            "num_command_segments": len(segments),
            "command_sampling": f"circle speed<={max_speed}, yaw<={max_yaw}, smoothstep transitions",
        },
    }
    summary = {
        "seconds": seconds,
        "reset_steps": reset_steps,
        "reset_frac": 1.0 if reset_steps else 0.0,
        "linErrorMean": summarize_values(lin_values)["mean"],
        "linErrorRms": summarize_values(lin_values)["rms"],
        "linErrorP95": summarize_values(lin_values)["p95"],
        "yawErrorMean": summarize_values(yaw_values)["mean"],
        "yawErrorRms": summarize_values(yaw_values)["rms"],
        "yawErrorP95": summarize_values(yaw_values)["p95"],
        "tilt_mean": summarize_values(flat("tilt"))["mean"],
        "tilt_rms": summarize_values(flat("tilt"))["rms"],
        "tilt_p95": summarize_values(flat("tilt"))["p95"],
        "tilt_max": summarize_values(flat("tilt"))["max"],
        "base_ang_vel_xy_rms": summarize_values(flat("base_ang_vel_xy"))["rms"],
        "shoulder_pitch_abs_p95": summarize_values(flat("shoulder_pitch_abs"))["p95"],
        "shoulder_roll_abs_p95": summarize_values(flat("shoulder_roll_abs"))["p95"],
        "arm_action_saturation_frac": summarize_values(flat("arm_action_saturation"))["mean"],
    }
    return replay, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="T1Shoulder4SwayMin_from7000LegFrozen_train5000")
    parser.add_argument("--arm_checkpoint", default="logs/2026-05-06-07-55-02/nn/model_900.pth")
    parser.add_argument("--leg_checkpoint", default="logs/2026-05-05-11-09-07/nn/model_4000.pth")
    parser.add_argument("--out_dir", default="artifacts/autoresearch/shoulder4_model900_eval")
    parser.add_argument("--label", default="shoulder4_model900")
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--fixed_duration_s", type=float, default=8.0)
    parser.add_argument("--warmup_s", type=float, default=2.0)
    parser.add_argument("--random_seconds", type=float, default=60.0)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--gait_frequency", type=float, default=1.5)
    parser.add_argument("--max_speed", type=float, default=1.35)
    parser.add_argument("--max_yaw", type=float, default=1.35)
    parser.add_argument("--smooth_s", type=float, default=0.65)
    parser.add_argument("--seed", type=int, default=20260506)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--mode", choices=["fixed", "random", "both"], default="both")
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fixed_json = out_dir / f"{args.label}_fixed_eval.json"
    fixed_csv = out_dir / f"{args.label}_fixed_eval.csv"
    fixed_summary = None
    if args.mode in {"fixed", "both"}:
        cfg = load_cfg(args.task, args.num_envs, args.seed, args.clean, max(args.fixed_duration_s, args.random_seconds))
        print(f"Creating fixed eval env envs={args.num_envs}", flush=True)
        env = eval(args.task)(cfg)
        device = cfg["basic"]["rl_device"]
        arm_model, leg_model = load_models(
            env,
            args.arm_checkpoint,
            args.leg_checkpoint,
            device,
            float(cfg["algorithm"].get("logstd_init", -3.0)),
            cfg["algorithm"].get("actor_mean_scale"),
            cfg["algorithm"].get("logstd_min"),
            cfg["algorithm"].get("logstd_max"),
        )
        print("Fixed eval DOFs", env.dof_names, flush=True)
        print("Fixed eval bodies", env.gym.get_actor_rigid_body_names(env.envs[0], env.actor_handles[0]), flush=True)
        fixed_steps = int(round(args.fixed_duration_s / env.dt))
        warmup_steps = int(round(args.warmup_s / env.dt))
        rows = []
        for name, vx, vy, wz in command_suite():
            row = summarize_command(env, arm_model, leg_model, name, (vx, vy, wz), fixed_steps, warmup_steps, args.gait_frequency)
            rows.append(row)
            print(
                f"{name:24s} xy={row['xy_err']:.3f} yaw={row['yaw_err']:.3f} "
                f"tilt={row['tilt_mean']:.4f}/{row['tilt_max']:.4f} reset={row['reset_frac']:.3f} "
                f"sat={row['arm_action_saturation_frac']:.4f}",
                flush=True,
            )
        moving = [row for row in rows if row["speed_ratio"] is not None]
        fixed_summary = {
            "command_count": len(rows),
            "mean_xy_err": sum(row["xy_err"] for row in rows) / len(rows),
            "mean_yaw_err": sum(row["yaw_err"] for row in rows) / len(rows),
            "mean_speed_ratio": sum(row["speed_ratio"] for row in moving) / max(1, len(moving)),
            "max_reset_frac": max(row["reset_frac"] for row in rows),
            "mean_tilt": sum(row["tilt_mean"] for row in rows) / len(rows),
            "max_tilt": max(row["tilt_max"] for row in rows),
            "tilt_rms_mean": sum(row["tilt_rms"] for row in rows) / len(rows),
            "tilt_p95_mean": sum(row["tilt_p95"] for row in rows) / len(rows),
            "base_ang_vel_xy_rms_mean": sum(row["base_ang_vel_xy_rms"] for row in rows) / len(rows),
            "shoulder_pitch_abs_p95_max": max(row["shoulder_pitch_abs_p95"] for row in rows),
            "shoulder_roll_abs_p95_max": max(row["shoulder_roll_abs_p95"] for row in rows),
            "arm_action_saturation_frac_max": max(row["arm_action_saturation_frac"] for row in rows),
        }
        fixed_payload = {
            "label": args.label,
            "checkpoint": args.arm_checkpoint,
            "leg_checkpoint": args.leg_checkpoint,
            "task": args.task,
            "mode": "clean" if args.clean else "default",
            "num_envs": args.num_envs,
            "duration_s": args.fixed_duration_s,
            "warmup_s": args.warmup_s,
            "rows": rows,
            "summary": fixed_summary,
        }
        fixed_json.write_text(json.dumps(fixed_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        with fixed_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        if args.mode == "fixed":
            print(json.dumps({"fixed_eval": fixed_summary, "files": {"fixed_json": str(fixed_json), "fixed_csv": str(fixed_csv)}}, ensure_ascii=False, indent=2), flush=True)
            return
    elif fixed_json.exists():
        fixed_summary = json.loads(fixed_json.read_text(encoding="utf-8")).get("summary")

    random_summary = None
    replay_json = out_dir / f"{args.label}_random_replay.json"
    random_json = out_dir / f"{args.label}_random_eval.json"
    if args.mode in {"random", "both"}:
        print("Creating random replay env", flush=True)
        replay_cfg = load_cfg(args.task, 1, args.seed, args.clean, args.random_seconds)
        replay_env = eval(args.task)(replay_cfg)
        replay_arm_model, replay_leg_model = load_models(
            replay_env,
            args.arm_checkpoint,
            args.leg_checkpoint,
            replay_cfg["basic"]["rl_device"],
            float(replay_cfg["algorithm"].get("logstd_init", -3.0)),
            replay_cfg["algorithm"].get("actor_mean_scale"),
            replay_cfg["algorithm"].get("logstd_min"),
            replay_cfg["algorithm"].get("logstd_max"),
        )
        replay, random_summary = random_replay(
            replay_env,
            replay_arm_model,
            replay_leg_model,
            args.random_seconds,
            args.warmup_s,
            args.gait_frequency,
            args.max_speed,
            args.max_yaw,
            args.smooth_s,
            args.fps,
            args.arm_checkpoint,
            args.seed,
        )
        replay_json.write_text(json.dumps(replay, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        random_json.write_text(json.dumps(random_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    elif random_json.exists():
        random_summary = json.loads(random_json.read_text(encoding="utf-8"))

    summary = {
        "label": args.label,
        "checkpoint": args.arm_checkpoint,
        "leg_checkpoint": args.leg_checkpoint,
        "fixed_eval": fixed_summary,
        "random_replay_60s": random_summary,
        "files": {
            "fixed_json": str(fixed_json),
            "fixed_csv": str(fixed_csv),
            "random_replay_json": str(replay_json),
            "random_eval_json": str(random_json),
        },
        "created_at": time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()),
    }
    summary_json = out_dir / f"{args.label}_summary.json"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
