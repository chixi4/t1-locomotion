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
    cfg["commands"]["sway_curriculum"] = False
    cfg["commands"]["still_proportion"] = 0.0
    for key in ["push_force", "push_torque", "kick_lin_vel", "kick_ang_vel"]:
        if key in cfg["randomization"] and isinstance(cfg["randomization"][key], dict):
            cfg["randomization"][key]["range"] = [0.0, 0.0]
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


def apply_command(env, cmd, steps_ahead):
    env.commands[:, 0] = float(cmd[0])
    env.commands[:, 1] = float(cmd[1])
    env.commands[:, 2] = float(cmd[2])
    speed = (cmd[0] * cmd[0] + cmd[1] * cmd[1]) ** 0.5
    env.gait_frequency[:] = 0.0 if speed < 1.0e-6 and abs(cmd[2]) < 1.0e-6 else 1.5
    env.cmd_resample_time[:] = env.episode_length_buf + steps_ahead + 1000


def q95(x):
    if not x:
        return None
    data = torch.cat(x, dim=0).detach().cpu()
    return float(torch.quantile(data, 0.95))


def mean_cat(x):
    if not x:
        return None
    return float(torch.cat(x, dim=0).mean().detach().cpu())


def corr(a, b):
    if not a or not b:
        return None
    x = torch.cat(a, dim=0).detach().cpu()
    y = torch.cat(b, dim=0).detach().cpu()
    x = x - x.mean()
    y = y - y.mean()
    den = x.std() * y.std()
    if den < 1.0e-6:
        return None
    return float((x * y).mean() / den)


def command_cases():
    return [
        ("stand", (0.0, 0.0, 0.0)),
        ("forward_10", (1.0, 0.0, 0.0)),
        ("backward_08", (-0.8, 0.0, 0.0)),
        ("left_08", (0.0, 0.8, 0.0)),
        ("right_08", (0.0, -0.8, 0.0)),
        ("forward_15", (1.5, 0.0, 0.0)),
        ("forward_18", (1.8, 0.0, 0.0)),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--warmup-s", type=float, default=1.5)
    parser.add_argument("--out")
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--residual-motion-gate", action="store_true")
    args = parser.parse_args()

    all_results = []
    for checkpoint in args.checkpoints:
        set_seed(args.seed)
        cfg = load_cfg(args.task, args.num_envs)
        task_class = eval(cfg["basic"]["task"])
        env = task_class(cfg)
        device = cfg["basic"]["rl_device"]
        leg_model = ActorCritic(len(env.leg_indices), 47, env.num_privileged_obs).to(device)
        leg_model.load_state_dict(torch.load(cfg["basic"]["leg_checkpoint"], map_location=device, weights_only=True)["model"], strict=True)
        leg_model.eval()
        upper_ckpt = torch.load(cfg["basic"]["upper_checkpoint"], map_location=device, weights_only=True)
        upper_model = ActorCritic(
            len(env.arm_indices),
            env.num_obs,
            env.num_privileged_obs,
            actor_mean_scale=upper_ckpt.get("actor_mean_scale"),
            logstd_min=cfg["algorithm"].get("upper_logstd_min"),
            logstd_max=cfg["algorithm"].get("upper_logstd_max"),
        ).to(device)
        upper_model.load_state_dict(upper_ckpt["model"], strict=False)
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
        residual_model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["model"], strict=False)
        residual_model.eval()
        effect_scale = dof_scale(env, cfg, "residual_effect_scale_by_dof")

        obs, infos = env.reset()
        obs = obs.to(device)
        steps = int(args.seconds / env.dt)
        warmup = int(args.warmup_s / env.dt)
        last_leg_action = torch.zeros(env.num_envs, len(env.leg_indices), dtype=torch.float, device=device)
        reset_total = torch.zeros(env.num_envs, dtype=torch.float, device=device)
        lin_err = []
        yaw_err = []
        tilt = []
        ang_xy = []
        stand_shoulder = []
        stand_elbow = []
        stand_waist = []
        pitch_abs = []
        pitch_left = []
        pitch_right_neg = []
        pitch_common = []
        left_side_out = []
        right_side_out = []
        elbow_move = []
        waist_move = []
        residual_abs = []
        case_results = []

        for case_name, cmd in command_cases():
            env._reset_idx(torch.arange(env.num_envs, device=device))
            env._compute_observations()
            obs = env.obs_buf.to(device)
            last_leg_action.zero_()
            case_reset = torch.zeros(env.num_envs, dtype=torch.float, device=device)
            case_lin_err = []
            case_yaw_err = []
            case_tilt = []
            case_ang_xy = []
            case_pitch_abs = []
            case_pitch_left = []
            case_pitch_right_neg = []
            case_pitch_common = []
            case_left_side_out = []
            case_right_side_out = []
            case_residual_abs = []
            for step in range(steps):
                apply_command(env, cmd, steps)
                with torch.no_grad():
                    leg_action = torch.clamp(leg_model.act(old_leg_obs(env, last_leg_action)).loc, -1.0, 1.0)
                    upper_action = torch.clamp(upper_model.act(obs).loc, -1.0, 1.0)
                    base = torch.zeros(env.num_envs, env.num_actions, dtype=torch.float, device=device)
                    base[:, env.leg_indices] = leg_action
                    base[:, env.arm_indices] = upper_action
                    residual = torch.clamp(residual_model.act(obs).loc, -1.0, 1.0) * effect_scale
                    if args.residual_motion_gate:
                        cmd_mag = torch.linalg.norm(env.commands[:, :2], dim=1) + 0.35 * torch.abs(env.commands[:, 2])
                        gate = torch.clamp((cmd_mag - 0.08) / 0.17, min=0.0, max=1.0).view(-1, 1)
                        residual = residual * gate
                    action = torch.clamp(base + residual, -1.0, 1.0)
                obs, rew, done, infos = env.step(action)
                obs = obs.to(device)
                last_leg_action[:] = action[:, env.leg_indices]
                last_leg_action[done] = 0.0
                reset_total += done.float()
                case_reset += done.float()
                apply_command(env, cmd, steps)
                if step < warmup:
                    continue
                moving = abs(cmd[0]) + abs(cmd[1]) + abs(cmd[2]) > 1.0e-6
                lin_sample = torch.linalg.norm(env.filtered_lin_vel[:, :2] - env.commands[:, :2], dim=1)
                yaw_sample = torch.abs(env.filtered_ang_vel[:, 2] - env.commands[:, 2])
                tilt_sample = torch.sqrt(torch.sum(torch.square(env.projected_gravity[:, :2]), dim=1))
                ang_sample = torch.linalg.norm(env.base_ang_vel[:, :2], dim=1)
                lin_err.append(lin_sample)
                yaw_err.append(yaw_sample)
                tilt.append(tilt_sample)
                ang_xy.append(ang_sample)
                case_lin_err.append(lin_sample)
                case_yaw_err.append(yaw_sample)
                case_tilt.append(tilt_sample)
                case_ang_xy.append(ang_sample)
                q_arm = env.dof_pos[:, env.arm_indices] - env.default_dof_pos[:, env.arm_indices]
                residual_sample = torch.abs(residual).reshape(-1)
                residual_abs.append(residual_sample)
                case_residual_abs.append(residual_sample)
                if case_name == "stand":
                    stand_shoulder.append(torch.abs(q_arm[:, :4]).reshape(-1))
                    if len(env.elbow_indices) > 0:
                        stand_elbow.append(torch.abs(env.dof_pos[:, env.elbow_indices] - env.default_dof_pos[:, env.elbow_indices]).reshape(-1))
                    if len(env.waist_indices) > 0:
                        stand_waist.append(torch.abs(env.dof_pos[:, env.waist_indices] - env.default_dof_pos[:, env.waist_indices]).reshape(-1))
                if moving:
                    pitch_sample = torch.abs(q_arm[:, [0, 2]]).reshape(-1)
                    pitch_abs.append(pitch_sample)
                    case_pitch_abs.append(pitch_sample)
                    pitch_left.append(q_arm[:, 0])
                    pitch_right_neg.append(-q_arm[:, 2])
                    pitch_common_sample = torch.abs(q_arm[:, 0] + q_arm[:, 2])
                    pitch_common.append(pitch_common_sample)
                    case_pitch_left.append(q_arm[:, 0])
                    case_pitch_right_neg.append(-q_arm[:, 2])
                    case_pitch_common.append(pitch_common_sample)
                    if len(env.elbow_indices) > 0:
                        elbow_move.append(torch.abs(env.dof_pos[:, env.elbow_indices] - env.default_dof_pos[:, env.elbow_indices]).reshape(-1))
                    if len(env.waist_indices) > 0:
                        waist_move.append(torch.abs(env.dof_pos[:, env.waist_indices] - env.default_dof_pos[:, env.waist_indices]).reshape(-1))
                if case_name == "right_08":
                    side_sample = torch.clamp(q_arm[:, 1], min=0.0)
                    left_side_out.append(side_sample)
                    case_left_side_out.append(side_sample)
                if case_name == "left_08":
                    side_sample = torch.clamp(-q_arm[:, 3], min=0.0)
                    right_side_out.append(side_sample)
                    case_right_side_out.append(side_sample)
            case_results.append(
                {
                    "case": case_name,
                    "cmd": list(cmd),
                    "reset_events_per_env": float(case_reset.mean().detach().cpu()),
                    "lin_error_mean": mean_cat(case_lin_err),
                    "yaw_error_mean": mean_cat(case_yaw_err),
                    "camera_tilt_p95": q95(case_tilt),
                    "camera_ang_xy_rms": float(torch.sqrt(torch.cat(case_ang_xy).square().mean()).detach().cpu()) if case_ang_xy else None,
                    "pitch_abs_p95": q95(case_pitch_abs),
                    "pitch_lr_antisym_corr": corr(case_pitch_left, case_pitch_right_neg),
                    "pitch_common_abs_p95": q95(case_pitch_common),
                    "side_left_out_p95_on_right": q95(case_left_side_out),
                    "side_right_out_p95_on_left": q95(case_right_side_out),
                    "residual_abs_p95": q95(case_residual_abs),
                }
            )

        all_results.append(
            {
                "checkpoint": checkpoint,
                "reset_events_per_env": float(reset_total.mean().detach().cpu()),
                "lin_error_mean": mean_cat(lin_err),
                "yaw_error_mean": mean_cat(yaw_err),
                "camera_tilt_p95": q95(tilt),
                "camera_ang_xy_rms": float(torch.sqrt(torch.cat(ang_xy).square().mean()).detach().cpu()) if ang_xy else None,
                "stand_shoulder_abs_p95": q95(stand_shoulder),
                "stand_elbow_abs_p95": q95(stand_elbow),
                "stand_waist_abs_p95": q95(stand_waist),
                "pitch_abs_p95_moving": q95(pitch_abs),
                "pitch_lr_antisym_corr": corr(pitch_left, pitch_right_neg),
                "pitch_common_abs_p95": q95(pitch_common),
                "side_left_out_p95_on_right": q95(left_side_out),
                "side_right_out_p95_on_left": q95(right_side_out),
                "elbow_abs_p95_moving": q95(elbow_move),
                "waist_abs_p95_moving": q95(waist_move),
                "residual_abs_p95": q95(residual_abs),
                "cases": case_results,
            }
        )
        del env
    payload = {"task": args.task, "results": all_results}
    text = json.dumps(payload, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
