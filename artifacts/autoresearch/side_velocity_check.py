import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "artifacts/autoresearch")

import eval_shoulder4_frozen as ev  # noqa: E402
import torch  # noqa: E402


def corrcoef(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or float(np.std(a)) < 1.0e-9 or float(np.std(b)) < 1.0e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def summarize_side_command(env, arm_model, leg_model, name, cmd, seconds, warmup_s, gait_frequency, event_vel):
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
            left_roll = q[:, 1]
            right_roll = q[:, 3]
            left_outward = left_roll
            right_outward = -right_roll
            left_out_vel = torch.relu(env.feet_vel_body[:, 0, 1])
            right_out_vel = torch.relu(-env.feet_vel_body[:, 1, 1])
            vel_delta = right_out_vel - left_out_vel
            arm_delta = left_roll + right_roll
            rows.append(
                torch.stack(
                    [left_outward, right_outward, left_out_vel, right_out_vel, vel_delta, arm_delta],
                    dim=1,
                )
                .detach()
                .cpu()
                .numpy()
            )
    data = np.concatenate(rows, axis=0) if rows else np.zeros((0, 6), dtype=np.float32)
    left_outward = data[:, 0]
    right_outward = data[:, 1]
    left_out_vel = data[:, 2]
    right_out_vel = data[:, 3]
    vel_delta = data[:, 4]
    arm_delta = data[:, 5]
    dominance = left_outward - right_outward
    right_event = (right_out_vel > event_vel) & (right_out_vel > left_out_vel * 1.1)
    left_event = (left_out_vel > event_vel) & (left_out_vel > right_out_vel * 1.1)

    def event_payload(mask, desired):
        if not np.any(mask):
            return {
                "frames": 0,
                "dominance_mean": 0.0,
                "desired_match_frac": 0.0,
                "left_outward_mean": 0.0,
                "right_outward_mean": 0.0,
            }
        dom = dominance[mask]
        if desired == "left":
            match = left_outward[mask] > right_outward[mask] + 0.02
        else:
            match = right_outward[mask] > left_outward[mask] + 0.02
        return {
            "frames": int(mask.sum()),
            "dominance_mean": float(np.mean(dom)),
            "desired_match_frac": float(np.mean(match)),
            "left_outward_mean": float(np.mean(left_outward[mask])),
            "right_outward_mean": float(np.mean(right_outward[mask])),
        }

    return {
        "name": name,
        "cmd": cmd,
        "reset_events_per_env": done_count / max(1, env.num_envs),
        "samples": int(data.shape[0]),
        "corr_vel_delta_to_arm_delta": corrcoef(vel_delta, arm_delta),
        "left_outward_mean": float(np.mean(left_outward)) if data.size else 0.0,
        "right_outward_mean": float(np.mean(right_outward)) if data.size else 0.0,
        "dominance_mean": float(np.mean(dominance)) if data.size else 0.0,
        "dominance_p05": float(np.percentile(dominance, 5)) if data.size else 0.0,
        "dominance_p95": float(np.percentile(dominance, 95)) if data.size else 0.0,
        "right_foot_out_event": event_payload(right_event, "left"),
        "left_foot_out_event": event_payload(left_event, "right"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--arm_checkpoint", required=True)
    parser.add_argument("--leg_checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--num_envs", type=int, default=128)
    parser.add_argument("--seconds", type=float, default=6.0)
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
        ("left_1.0", (0.0, 1.0, 0.0)),
        ("right_1.0", (0.0, -1.0, 0.0)),
        ("left_1.5", (0.0, 1.5, 0.0)),
        ("right_1.5", (0.0, -1.5, 0.0)),
    ]
    rows = [
        summarize_side_command(env, arm_model, leg_model, name, cmd, args.seconds, args.warmup_s, args.gait_frequency, args.event_vel)
        for name, cmd in commands
    ]
    payload = {
        "checkpoint": args.arm_checkpoint,
        "event_vel": args.event_vel,
        "seconds": args.seconds,
        "warmup_s": args.warmup_s,
        "rows": rows,
        "mean_corr": float(np.mean([row["corr_vel_delta_to_arm_delta"] for row in rows])),
        "mean_right_event_match": float(np.mean([row["right_foot_out_event"]["desired_match_frac"] for row in rows if row["right_foot_out_event"]["frames"] > 0] or [0.0])),
        "mean_left_event_match": float(np.mean([row["left_foot_out_event"]["desired_match_frac"] for row in rows if row["left_foot_out_event"]["frames"] > 0] or [0.0])),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
