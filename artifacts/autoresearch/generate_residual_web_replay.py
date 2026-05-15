import argparse
import csv
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
    "Waist",
    "AL1",
    "AL2",
    "AL3",
    "left_hand_link",
    "AR1",
    "AR2",
    "AR3",
    "right_hand_link",
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

MESH_SPECS = [
    {"body": "Trunk", "file": "../../resources/T1/meshes/Trunk.STL", "pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0], "color": [0.76, 0.76, 0.76]},
    {"body": "Trunk", "file": "../../resources/T1/meshes/H1.STL", "pos": [0.0625, 0.0, 0.243], "quat": [1.0, 0.0, 0.0, 0.0], "color": [0.4, 0.4, 0.4]},
    {"body": "Trunk", "file": "../../resources/T1/meshes/H2.STL", "pos": [0.0625, 0.0, 0.30485], "quat": [1.0, 0.0, 0.0, 0.0], "color": [0.4, 0.4, 0.4]},
    {"body": "Waist", "file": "../../resources/T1/meshes/Waist.STL", "pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0], "color": [0.4, 0.4, 0.4]},
    {"body": "AL1", "file": "../../resources/T1/meshes/AL1.STL", "pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0], "color": [0.76, 0.76, 0.76]},
    {"body": "AL2", "file": "../../resources/T1/meshes/AL2.STL", "pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0], "color": [0.4, 0.4, 0.4]},
    {"body": "AL3", "file": "../../resources/T1/meshes/AL3.STL", "pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0], "color": [0.4, 0.4, 0.4]},
    {"body": "left_hand_link", "file": "../../resources/T1/meshes/left_hand_link.STL", "pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0], "color": [0.4, 0.4, 0.4]},
    {"body": "AR1", "file": "../../resources/T1/meshes/AR1.STL", "pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0], "color": [0.76, 0.76, 0.76]},
    {"body": "AR2", "file": "../../resources/T1/meshes/AR2.STL", "pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0], "color": [0.4, 0.4, 0.4]},
    {"body": "AR3", "file": "../../resources/T1/meshes/AR3.STL", "pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0], "color": [0.4, 0.4, 0.4]},
    {"body": "right_hand_link", "file": "../../resources/T1/meshes/right_hand_link.STL", "pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0], "color": [0.4, 0.4, 0.4]},
    {"body": "Hip_Pitch_Left", "file": "../../resources/T1/meshes/Hip_Pitch_Left.STL", "pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0], "color": [0.76, 0.76, 0.76]},
    {"body": "Hip_Roll_Left", "file": "../../resources/T1/meshes/Hip_Roll_Left.STL", "pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0], "color": [0.4, 0.4, 0.4]},
    {"body": "Hip_Yaw_Left", "file": "../../resources/T1/meshes/Hip_Yaw_Left.STL", "pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0], "color": [0.4, 0.4, 0.4]},
    {"body": "Shank_Left", "file": "../../resources/T1/meshes/Shank_Left.STL", "pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0], "color": [0.4, 0.4, 0.4]},
    {"body": "Ankle_Cross_Left", "file": "../../resources/T1/meshes/Ankle_Cross_Left.STL", "pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0], "color": [0.4, 0.4, 0.4]},
    {"body": "left_foot_link", "file": "../../resources/T1/meshes/left_foot_link.STL", "pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0], "color": [0.4, 0.4, 0.4]},
    {"body": "Hip_Pitch_Right", "file": "../../resources/T1/meshes/Hip_Pitch_Right.STL", "pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0], "color": [0.76, 0.76, 0.76]},
    {"body": "Hip_Roll_Right", "file": "../../resources/T1/meshes/Hip_Roll_Right.STL", "pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0], "color": [0.4, 0.4, 0.4]},
    {"body": "Hip_Yaw_Right", "file": "../../resources/T1/meshes/Hip_Yaw_Right.STL", "pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0], "color": [0.4, 0.4, 0.4]},
    {"body": "Shank_Right", "file": "../../resources/T1/meshes/Shank_Right.STL", "pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0], "color": [0.4, 0.4, 0.4]},
    {"body": "Ankle_Cross_Right", "file": "../../resources/T1/meshes/Ankle_Cross_Right.STL", "pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0], "color": [0.4, 0.4, 0.4]},
    {"body": "right_foot_link", "file": "../../resources/T1/meshes/right_foot_link.STL", "pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0], "color": [0.4, 0.4, 0.4]},
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fixed_cfg(cfg, args):
    cfg["basic"]["headless"] = True
    cfg["viewer"]["record_video"] = False
    cfg["env"]["num_envs"] = 1
    cfg["basic"]["sim_device"] = "cuda:0"
    cfg["basic"]["rl_device"] = "cuda:0"
    cfg["basic"]["seed"] = args.seed
    cfg["runner"]["use_wandb"] = False
    cfg["commands"]["curriculum"] = False
    cfg["commands"]["sway_curriculum"] = False
    cfg["commands"]["still_proportion"] = 0.0
    cfg["commands"]["resampling_time_s"] = [1.0e6, 1.0e6 + 1.0]
    cfg["rewards"]["episode_length_s"] = args.seconds + 8.0

    fixed_add0 = {"range": [0.0, 0.0], "operation": "additive", "distribution": "uniform"}
    fixed_scale1 = {"range": [1.0, 1.0], "operation": "scaling", "distribution": "uniform"}
    fixed_friction = {"range": [1.0, 1.0], "operation": "additive", "distribution": "uniform"}
    randomization = cfg.get("randomization", {})
    for key in ["init_dof_pos", "init_base_pos_xy", "init_base_lin_vel_xy"]:
        randomization[key] = None
    randomization["kick_interval_s"] = 1.0e9
    randomization["push_interval_s"] = 1.0e9
    randomization["kick_lin_vel"] = None
    randomization["kick_ang_vel"] = None
    randomization["push_force"] = None
    randomization["push_torque"] = None
    randomization["dof_stiffness"] = fixed_scale1
    randomization["dof_damping"] = fixed_scale1
    randomization["dof_friction"] = fixed_add0
    randomization["friction"] = fixed_friction
    randomization["compliance"] = fixed_add0
    randomization["restitution"] = fixed_add0
    randomization["base_com"] = fixed_add0
    randomization["base_mass"] = fixed_scale1
    randomization["other_com"] = fixed_add0
    randomization["other_mass"] = fixed_scale1
    return cfg


def dof_scale(env, cfg, key, default=1.0):
    by_dof = cfg["algorithm"].get(key)
    if not by_dof:
        return torch.full((1, env.num_actions), float(default), dtype=torch.float, device=env.device)
    return torch.tensor([float(by_dof[name]) for name in env.dof_names], dtype=torch.float, device=env.device).view(1, -1)


def old_leg_obs(env, last_leg_action):
    commands_scale = torch.tensor(
        [env.cfg["normalization"]["lin_vel"], env.cfg["normalization"]["lin_vel"], env.cfg["normalization"]["ang_vel"]],
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


def command_segments(seconds):
    raw = [
        (0.0, 3.0, (0.0, 0.0, 0.0), "stand"),
        (3.0, 9.0, (1.0, 0.0, 0.0), "forward_10"),
        (9.0, 15.0, (1.5, 0.0, 0.0), "forward_15"),
        (15.0, 21.0, (1.8, 0.0, 0.0), "forward_18"),
        (21.0, 27.0, (-0.8, 0.0, 0.0), "backward_08"),
        (27.0, 33.0, (0.0, 0.8, 0.0), "left_08"),
        (33.0, 39.0, (0.0, -0.8, 0.0), "right_08"),
        (39.0, 46.0, (1.0, 0.45, 0.35), "diagonal_left_turn"),
        (46.0, 53.0, (1.0, -0.45, -0.35), "diagonal_right_turn"),
        (53.0, seconds, (0.0, 0.0, 0.0), "settle"),
    ]
    segments = []
    prev = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    for start, end, target, name in raw:
        if start >= seconds:
            break
        target_np = np.array(target, dtype=np.float32)
        segments.append(
            {
                "start": float(start),
                "end": float(min(end, seconds)),
                "prev": prev.copy(),
                "target": target_np.copy(),
                "name": name,
            }
        )
        prev = target_np
    return segments


def command_at(segments, t, smooth_s):
    seg = segments[-1]
    for candidate in segments:
        if candidate["start"] <= t < candidate["end"]:
            seg = candidate
            break
    u = min(1.0, max(0.0, (t - seg["start"]) / smooth_s))
    u = u * u * (3.0 - 2.0 * u)
    return (1.0 - u) * seg["prev"] + u * seg["target"], seg["name"]


def apply_command(env, cmd):
    cmd_t = torch.tensor(cmd, dtype=torch.float, device=env.device).view(1, 3)
    env.commands[:, :3] = cmd_t
    cmd_mag = float(np.linalg.norm(cmd))
    env.gait_frequency[:] = 0.0 if cmd_mag < 0.04 else 1.5
    env.cmd_resample_time[:] = 10**9


def actor_body_names(env):
    return env.gym.get_actor_rigid_body_names(env.envs[0], env.actor_handles[0])


def q95(values):
    if not values:
        return None
    data = np.concatenate(values)
    return float(np.quantile(data, 0.95))


def rms(values):
    if not values:
        return None
    data = np.concatenate(values)
    return float(np.sqrt(np.mean(np.square(data))))


def index_or_none(names, name):
    try:
        return names.index(name)
    except ValueError:
        return None


def load_confirm(confirm_path):
    if not confirm_path:
        return None
    p = REPO / confirm_path
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    if data.get("results"):
        return data["results"][0]
    return None


def make_fixed_eval(raw):
    if not raw:
        return {}
    return {
        "max_reset_frac": raw.get("reset_events_per_env"),
        "mean_tilt": raw.get("camera_tilt_p95"),
        "max_tilt": raw.get("camera_tilt_p95"),
        "mean_xy_err": raw.get("lin_error_mean"),
        "mean_yaw_err": raw.get("yaw_error_mean"),
        "mean_speed_ratio": max(0.0, 1.0 - float(raw.get("lin_error_mean", 0.0)) / 1.8),
        "raw_confirm": raw,
    }


def write_csv(path, raw):
    if not raw:
        return
    keys = [k for k, v in raw.items() if not isinstance(v, (list, dict))]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerow({k: raw.get(k) for k in keys})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--confirm-json", default="")
    parser.add_argument("--seed", type=int, default=20260514)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--smooth-s", type=float, default=0.8)
    args = parser.parse_args()

    os.chdir(REPO)
    set_seed(args.seed)
    with open(f"envs/{args.task}.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg = fixed_cfg(cfg, args)

    print("creating residual replay env...", flush=True)
    task_class = eval(cfg["basic"]["task"])
    env = task_class(cfg)
    device = cfg["basic"]["rl_device"]

    leg_model = ActorCritic(len(env.leg_indices), 47, env.num_privileged_obs).to(device)
    leg_state = torch.load(cfg["basic"]["leg_checkpoint"], map_location=device, weights_only=True)
    leg_model.load_state_dict(leg_state["model"], strict=True)
    leg_model.eval()

    upper_state = torch.load(cfg["basic"]["upper_checkpoint"], map_location=device, weights_only=True)
    upper_model = ActorCritic(
        len(env.arm_indices),
        env.num_obs,
        env.num_privileged_obs,
        actor_mean_scale=upper_state.get("actor_mean_scale"),
        logstd_min=cfg["algorithm"].get("upper_logstd_min"),
        logstd_max=cfg["algorithm"].get("upper_logstd_max"),
    ).to(device)
    upper_model.load_state_dict(upper_state["model"], strict=False)
    upper_model.eval()

    residual_model = ActorCritic(
        env.num_actions,
        env.num_obs,
        env.num_privileged_obs,
        logstd_init=float(cfg["algorithm"].get("logstd_init", -4.2)),
        actor_mean_scale=[float(cfg["algorithm"]["residual_actor_mean_scale_by_dof"][name]) for name in env.dof_names],
        logstd_min=cfg["algorithm"].get("logstd_min"),
        logstd_max=cfg["algorithm"].get("logstd_max"),
    ).to(device)
    residual_state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    residual_model.load_state_dict(residual_state["model"], strict=False)
    residual_model.eval()
    effect_scale = dof_scale(env, cfg, "residual_effect_scale_by_dof")

    env.reset()
    obs = clean_start(env).to(device)
    last_leg_action = torch.zeros(env.num_envs, len(env.leg_indices), dtype=torch.float, device=device)

    rigid_names = actor_body_names(env)
    missing = [name for name in BODY_NAMES if name not in rigid_names]
    if missing:
        raise RuntimeError(f"Missing replay bodies: {missing}; available={rigid_names}")
    body_indices = torch.tensor([rigid_names.index(name) for name in BODY_NAMES], dtype=torch.long, device=env.device)

    dof_names = list(env.dof_names)
    left_pitch = index_or_none(dof_names, "Left_Shoulder_Pitch")
    right_pitch = index_or_none(dof_names, "Right_Shoulder_Pitch")
    left_roll = index_or_none(dof_names, "Left_Shoulder_Roll")
    right_roll = index_or_none(dof_names, "Right_Shoulder_Roll")

    frames = []
    reset_steps = []
    tilt_samples = []
    lin_err_samples = []
    yaw_err_samples = []
    shoulder_pitch_samples = []
    shoulder_roll_samples = []
    arm_sat_samples = []
    command_segments_out = []
    segments = command_segments(args.seconds)

    total_steps = int(round(args.seconds / env.dt))
    for step in range(total_steps):
        t = step * env.dt
        cmd, cmd_name = command_at(segments, t, args.smooth_s)
        apply_command(env, cmd)
        with torch.no_grad():
            leg_action = torch.clamp(leg_model.act(old_leg_obs(env, last_leg_action)).loc, -1.0, 1.0)
            upper_action = torch.clamp(upper_model.act(obs).loc, -1.0, 1.0)
            base = torch.zeros(env.num_envs, env.num_actions, dtype=torch.float, device=device)
            base[:, env.leg_indices] = leg_action
            base[:, env.arm_indices] = upper_action
            residual = torch.clamp(residual_model.act(obs).loc, -1.0, 1.0) * effect_scale
            action = torch.clamp(base + residual, -1.0, 1.0)
        obs, _, done, _ = env.step(action)
        obs = obs.to(device)
        last_leg_action[:] = action[:, env.leg_indices]
        if bool(done[0].item()):
            reset_steps.append(step)
            last_leg_action.zero_()
        apply_command(env, cmd)

        tilt = torch.sqrt(torch.sum(torch.square(env.projected_gravity[:, :2]), dim=1)).detach().cpu().numpy()
        lin_err = torch.linalg.norm(env.filtered_lin_vel[:, :2] - env.commands[:, :2], dim=1).detach().cpu().numpy()
        yaw_err = torch.abs(env.filtered_ang_vel[:, 2] - env.commands[:, 2]).detach().cpu().numpy()
        tilt_samples.append(tilt)
        lin_err_samples.append(lin_err)
        yaw_err_samples.append(yaw_err)
        arm_sat_samples.append((torch.abs(action[:, env.arm_indices]) > 0.97).float().detach().cpu().numpy().reshape(-1))
        q = (env.dof_pos - env.default_dof_pos).detach().cpu().numpy()[0]
        if left_pitch is not None and right_pitch is not None:
            shoulder_pitch_samples.append(np.abs([q[left_pitch], q[right_pitch]], dtype=np.float32))
        if left_roll is not None and right_roll is not None:
            shoulder_roll_samples.append(np.abs([q[left_roll], q[right_roll]], dtype=np.float32))

        body_state = env.body_states[0, body_indices, :7].detach().cpu().numpy()
        p = np.round(body_state[:, :3], 4).tolist()
        q_xyzw = body_state[:, 3:7]
        q_wxyz = np.stack([q_xyzw[:, 3], q_xyzw[:, 0], q_xyzw[:, 1], q_xyzw[:, 2]], axis=1)
        q_out = np.round(q_wxyz, 5).tolist()
        frames.append({"p": p, "q": q_out, "c": np.round(cmd.astype(np.float32), 4).tolist(), "case": cmd_name})

    for seg in segments:
        command_segments_out.append(
            {
                "start": round(float(seg["start"]), 3),
                "end": round(float(seg["end"]), 3),
                "target": np.round(seg["target"], 4).tolist(),
                "name": seg["name"],
            }
        )

    random_eval = {
        "tilt_rms": rms(tilt_samples),
        "tilt_p95": q95(tilt_samples),
        "tilt_max": float(np.max(np.concatenate(tilt_samples))) if tilt_samples else None,
        "linErrorRms": rms(lin_err_samples),
        "yawErrorRms": rms(yaw_err_samples),
        "shoulder_pitch_abs_p95": q95(shoulder_pitch_samples),
        "shoulder_roll_abs_p95": q95(shoulder_roll_samples),
        "arm_action_saturation_frac": float(np.mean(np.concatenate(arm_sat_samples))) if arm_sat_samples else None,
        "reset_steps": reset_steps,
        "reset_count": len(reset_steps),
    }
    confirm = load_confirm(args.confirm_json)
    fixed_eval = make_fixed_eval(confirm)

    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    replay = {
        "title": f"{args.label} staged command replay",
        "stage": args.label,
        "stageLabel": f"{args.label} staged command replay",
        "command": {"vx": 0.0, "vy": 0.0, "yaw": 0.0},
        "fps": FPS,
        "bodyNames": BODY_NAMES,
        "meshSpecs": MESH_SPECS,
        "frames": frames,
        "commandSegments": command_segments_out,
        "meta": {
            "checkpoint": args.checkpoint,
            "task": args.task,
            "seed": args.seed,
            "seconds": args.seconds,
            "policy_dt": env.dt,
            "frames": len(frames),
            "reset_steps": reset_steps,
            "composition": {
                "leg_checkpoint": cfg["basic"]["leg_checkpoint"],
                "upper_checkpoint": cfg["basic"]["upper_checkpoint"],
                "residual_checkpoint": args.checkpoint,
            },
            "web_label": args.label,
        },
    }
    summary = {
        "label": args.label,
        "task": args.task,
        "checkpoint": args.checkpoint,
        "fixed_eval": fixed_eval,
        "random_replay_60s": random_eval,
        "selected_confirm_eval": confirm,
        "command_segments": command_segments_out,
    }
    (out_dir / f"{args.label}_random_replay.json").write_text(json.dumps(replay, separators=(",", ":")), encoding="utf-8")
    (out_dir / f"{args.label}_random_eval.json").write_text(json.dumps(random_eval, indent=2), encoding="utf-8")
    (out_dir / f"{args.label}_fixed_eval.json").write_text(json.dumps(fixed_eval, indent=2), encoding="utf-8")
    write_csv(out_dir / f"{args.label}_fixed_eval.csv", confirm)
    (out_dir / f"{args.label}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "random_eval": random_eval, "confirm": confirm}, indent=2), flush=True)


if __name__ == "__main__":
    main()
