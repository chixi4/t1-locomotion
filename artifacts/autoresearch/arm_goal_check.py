import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "artifacts/autoresearch")

import eval_shoulder4_frozen as ev  # noqa: E402
from envs import *  # noqa: F401,F403,E402
import torch  # noqa: E402


def corrcoef(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or float(np.std(a)) < 1.0e-9 or float(np.std(b)) < 1.0e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def pct(a, q):
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0:
        return 0.0
    return float(np.percentile(a, q))


def run_command(env, arm_model, leg_model, name, cmd, seconds, warmup_s, gait_frequency):
    ev.reset_env(env)
    obs = ev.apply_command(env, cmd, gait_frequency).to(env.device)
    last_leg_action = torch.zeros(env.num_envs, len(env.leg_indices), dtype=torch.float, device=env.device)
    total_steps = int(round(seconds / env.dt))
    warmup_steps = int(round(warmup_s / env.dt))
    rows = []
    done_count = 0
    with torch.no_grad():
        for step in range(total_steps):
            full_action, arm_action, leg_action = ev.policy_step(env, arm_model, leg_model, obs, last_leg_action)
            obs, _, done, _ = env.step(full_action)
            obs = ev.apply_command(env, cmd, gait_frequency).to(env.device)
            last_leg_action[:] = leg_action
            last_leg_action[done] = 0.0
            done_count += int(done.sum().item())
            if step < warmup_steps:
                continue
            q = env.dof_pos[:, env.arm_indices] - env.default_dof_pos[:, env.arm_indices]
            foot_sag = env.feet_pos_body[:, 0, 0] - env.feet_pos_body[:, 1, 0]
            left_out_vel = torch.relu(env.feet_vel_body[:, 0, 1])
            right_out_vel = torch.relu(-env.feet_vel_body[:, 1, 1])
            rows.append(
                torch.stack(
                    [
                        q[:, 0],
                        q[:, 1],
                        q[:, 2],
                        q[:, 3],
                        foot_sag,
                        left_out_vel,
                        right_out_vel,
                    ],
                    dim=1,
                )
                .detach()
                .cpu()
                .numpy()
            )
    data = np.concatenate(rows, axis=0) if rows else np.zeros((0, 7), dtype=np.float32)
    left_pitch = data[:, 0]
    left_roll = data[:, 1]
    right_pitch = data[:, 2]
    right_roll = data[:, 3]
    foot_sag = data[:, 4]
    left_out_vel = data[:, 5]
    right_out_vel = data[:, 6]
    left_outward = left_roll
    right_outward = -right_roll
    return {
        "name": name,
        "cmd": list(cmd),
        "reset_events_per_env": done_count / max(1, env.num_envs),
        "left_pitch": left_pitch,
        "left_roll": left_roll,
        "right_pitch": right_pitch,
        "right_roll": right_roll,
        "foot_sag": foot_sag,
        "left_out_vel": left_out_vel,
        "right_out_vel": right_out_vel,
        "left_outward": left_outward,
        "right_outward": right_outward,
    }


def summarize_checkpoint(rows, event_vel):
    by_name = {row["name"]: row for row in rows}
    stand = by_name["stand"]
    stand_abs = np.concatenate(
        [
            np.abs(stand["left_pitch"]),
            np.abs(stand["right_pitch"]),
            np.abs(stand["left_roll"]),
            np.abs(stand["right_roll"]),
        ]
    )
    pitch_rows = [by_name["forward_1.0"], by_name["backward_1.0"], by_name["forward_1.5"], by_name["backward_1.5"]]
    pitch_left = np.concatenate([row["left_pitch"] for row in pitch_rows])
    pitch_right = np.concatenate([row["right_pitch"] for row in pitch_rows])
    pitch_foot = np.concatenate([row["foot_sag"] for row in pitch_rows])
    pitch_abs = np.concatenate([np.abs(pitch_left), np.abs(pitch_right)])
    pitch_sum = pitch_left + pitch_right

    side_rows = [by_name["left_1.0"], by_name["right_1.0"], by_name["left_1.5"], by_name["right_1.5"]]
    side_left_out = np.concatenate([row["left_outward"] for row in side_rows])
    side_right_out = np.concatenate([row["right_outward"] for row in side_rows])
    side_left_vel = np.concatenate([row["left_out_vel"] for row in side_rows])
    side_right_vel = np.concatenate([row["right_out_vel"] for row in side_rows])
    side_arm_delta = side_left_out - side_right_out
    side_vel_delta = side_right_vel - side_left_vel
    right_event = (side_right_vel > event_vel) & (side_right_vel > side_left_vel * 1.1)
    left_event = (side_left_vel > event_vel) & (side_left_vel > side_right_vel * 1.1)

    def match(mask, side):
        if not np.any(mask):
            return 0.0
        if side == "left_arm":
            return float(np.mean(side_left_out[mask] > side_right_out[mask] + 0.02))
        return float(np.mean(side_right_out[mask] > side_left_out[mask] + 0.02))

    return {
        "max_reset_events_per_env": float(max(row["reset_events_per_env"] for row in rows)),
        "stand_abs_mean": float(np.mean(stand_abs)),
        "stand_abs_p95": pct(stand_abs, 95),
        "stand_pitch_mean": float(np.mean(np.concatenate([stand["left_pitch"], stand["right_pitch"]]))),
        "stand_roll_abs_p95": pct(np.concatenate([np.abs(stand["left_roll"]), np.abs(stand["right_roll"])]), 95),
        "pitch_abs_p95": pct(pitch_abs, 95),
        "pitch_lr_antisym_corr": corrcoef(pitch_left, -pitch_right),
        "pitch_foot_left_corr": corrcoef(pitch_foot, pitch_left),
        "pitch_foot_right_corr": corrcoef(pitch_foot, -pitch_right),
        "pitch_common_abs_p95": pct(np.abs(pitch_sum), 95),
        "side_outward_abs_p95": pct(np.concatenate([np.abs(side_left_out), np.abs(side_right_out)]), 95),
        "side_vel_to_arm_delta_corr": corrcoef(side_vel_delta, side_arm_delta),
        "right_foot_event_frames": int(right_event.sum()),
        "right_foot_event_left_arm_match": match(right_event, "left_arm"),
        "left_foot_event_frames": int(left_event.sum()),
        "left_foot_event_right_arm_match": match(left_event, "right_arm"),
        "side_left_outward_mean": float(np.mean(side_left_out)),
        "side_right_outward_mean": float(np.mean(side_right_out)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--arm_checkpoint", required=True)
    parser.add_argument("--leg_checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--num_envs", type=int, default=128)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--warmup_s", type=float, default=1.0)
    parser.add_argument("--gait_frequency", type=float, default=1.5)
    parser.add_argument("--event_vel", type=float, default=0.18)
    parser.add_argument("--seed", type=int, default=20260513)
    args = parser.parse_args()

    ev.set_seed(args.seed)
    cfg = ev.load_cfg(args.task, args.num_envs, args.seed, True, args.seconds)
    env = eval(args.task)(cfg)
    arm_model, leg_model = ev.load_models(
        env,
        args.arm_checkpoint,
        args.leg_checkpoint,
        cfg["basic"]["rl_device"],
        float(cfg["algorithm"].get("logstd_init", -3.0)),
        ev.actor_mean_scale_from_cfg(env, cfg),
        cfg["algorithm"].get("logstd_min"),
        cfg["algorithm"].get("logstd_max"),
    )
    commands = [
        ("stand", (0.0, 0.0, 0.0)),
        ("forward_1.0", (1.0, 0.0, 0.0)),
        ("backward_1.0", (-1.0, 0.0, 0.0)),
        ("forward_1.5", (1.5, 0.0, 0.0)),
        ("backward_1.5", (-1.5, 0.0, 0.0)),
        ("left_1.0", (0.0, 1.0, 0.0)),
        ("right_1.0", (0.0, -1.0, 0.0)),
        ("left_1.5", (0.0, 1.5, 0.0)),
        ("right_1.5", (0.0, -1.5, 0.0)),
    ]
    rows = [run_command(env, arm_model, leg_model, name, cmd, args.seconds, args.warmup_s, args.gait_frequency) for name, cmd in commands]
    summary = summarize_checkpoint(rows, args.event_vel)
    payload = {
        "checkpoint": args.arm_checkpoint,
        "event_vel": args.event_vel,
        "seconds": args.seconds,
        "warmup_s": args.warmup_s,
        "summary": summary,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
