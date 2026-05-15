import json
import math
import sys
from pathlib import Path

import isaacgym  # noqa: F401
import torch

sys.path.insert(0, str(Path("artifacts/autoresearch").resolve()))
import eval_shoulder4_frozen as ev  # noqa: E402


TASK = "T1Shoulder4GaitPhaseNightJLowBaseLockedDynamic_from7000LegFrozen_train400"
ARM_CKPT = "logs/2026-05-12-14-06-21/nn/model_400.pth"
LEG_CKPT = "logs/2026-05-05-11-09-07/nn/model_4000.pth"


def qstats(values):
    if not values:
        return {"mean": 0.0, "p05": 0.0, "p50": 0.0, "p95": 0.0, "range_p95_p05": 0.0}
    x = torch.tensor(values, dtype=torch.float32)
    p = torch.quantile(x, torch.tensor([0.05, 0.50, 0.95]))
    return {
        "mean": float(x.mean().item()),
        "p05": float(p[0].item()),
        "p50": float(p[1].item()),
        "p95": float(p[2].item()),
        "range_p95_p05": float((p[2] - p[0]).item()),
    }


def main():
    commands = [
        ("left_1.0", 0.0, 1.0, 0.0),
        ("right_1.0", 0.0, -1.0, 0.0),
        ("left_1.5", 0.0, 1.5, 0.0),
        ("right_1.5", 0.0, -1.5, 0.0),
        ("diag_fl_1.5", math.sqrt(0.5) * 1.5, math.sqrt(0.5) * 1.5, 0.0),
        ("diag_fr_1.5", math.sqrt(0.5) * 1.5, -math.sqrt(0.5) * 1.5, 0.0),
    ]
    ev.set_seed(20260512)
    cfg = ev.load_cfg(TASK, 192, 20260512, True, 8.0)
    print("creating env", flush=True)
    env = getattr(ev, TASK)(cfg)
    device = cfg["basic"]["rl_device"]
    arm_model, leg_model = ev.load_models(
        env,
        ARM_CKPT,
        LEG_CKPT,
        device,
        float(cfg["algorithm"].get("logstd_init", -3.0)),
        ev.actor_mean_scale_from_cfg(env, cfg),
        cfg["algorithm"].get("logstd_min"),
        cfg["algorithm"].get("logstd_max"),
    )

    rigid_names = env.gym.get_actor_rigid_body_names(env.envs[0], env.actor_handles[0])
    al2_idx = rigid_names.index("AL2")
    ar2_idx = rigid_names.index("AR2")
    trunk_idx = rigid_names.index("Trunk")
    steps = int(round(8.0 / env.dt))
    warmup = int(round(2.0 / env.dt))
    results = []
    for name, vx, vy, wz in commands:
        ev.reset_env(env)
        obs = ev.apply_command(env, (vx, vy, wz), 1.5).to(env.device)
        last_leg_action = torch.zeros(env.num_envs, len(env.leg_indices), dtype=torch.float, device=env.device)
        bucket = {
            "left_pitch": [],
            "right_pitch": [],
            "left_roll": [],
            "right_roll": [],
            "left_al2_rel_z": [],
            "right_ar2_rel_z": [],
            "left_out": [],
            "right_out": [],
            "foot_delta": [],
            "left_extra": [],
            "right_extra": [],
            "left_roll_action": [],
            "right_roll_action": [],
        }
        with torch.no_grad():
            for step in range(steps):
                full_action, arm_action, leg_action = ev.policy_step(env, arm_model, leg_model, obs, last_leg_action)
                obs, _, done, _ = env.step(full_action)
                obs = ev.apply_command(env, (vx, vy, wz), 1.5).to(env.device)
                last_leg_action[:] = leg_action
                last_leg_action[done] = 0.0
                if step < warmup:
                    continue
                q = env.dof_pos[:, env.arm_indices] - env.default_dof_pos[:, env.arm_indices]
                trunk_z = env.body_states[:, trunk_idx, 2]
                left_al2_z = env.body_states[:, al2_idx, 2] - trunk_z
                right_ar2_z = env.body_states[:, ar2_idx, 2] - trunk_z
                left_out = torch.clamp(env.feet_pos_body[:, 0, 1], min=0.0)
                right_out = torch.clamp(-env.feet_pos_body[:, 1, 1], min=0.0)
                foot_delta = left_out - right_out
                foot_norm = float(env.cfg["rewards"].get("shoulder_lateral_foot_out_norm", 0.085))
                left_extra = torch.clamp(foot_delta / max(1.0e-6, foot_norm), min=0.0, max=1.0)
                right_extra = torch.clamp(-foot_delta / max(1.0e-6, foot_norm), min=0.0, max=1.0)
                tensors = {
                    "left_pitch": q[:, 0],
                    "right_pitch": q[:, 2],
                    "left_roll": q[:, 1],
                    "right_roll": q[:, 3],
                    "left_al2_rel_z": left_al2_z,
                    "right_ar2_rel_z": right_ar2_z,
                    "left_out": left_out,
                    "right_out": right_out,
                    "foot_delta": foot_delta,
                    "left_extra": left_extra,
                    "right_extra": right_extra,
                    "left_roll_action": arm_action[:, 1],
                    "right_roll_action": arm_action[:, 3],
                }
                for key, value in tensors.items():
                    bucket[key].extend(value.detach().cpu().tolist())
        row = {"name": name, "cmd": [vx, vy, wz]}
        for key, values in bucket.items():
            row[key] = qstats(values)
        row["roll_mean_abs"] = 0.5 * (abs(row["left_roll"]["mean"]) + abs(row["right_roll"]["mean"]))
        row["roll_dyn_avg"] = 0.5 * (row["left_roll"]["range_p95_p05"] + row["right_roll"]["range_p95_p05"])
        row["al2_ar2_z_dyn_avg"] = 0.5 * (
            row["left_al2_rel_z"]["range_p95_p05"] + row["right_ar2_rel_z"]["range_p95_p05"]
        )
        results.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    out = Path("artifacts/autoresearch/shoulder4_night_j_lr_asym_probe.json")
    out.write_text(json.dumps({"task": TASK, "arm_checkpoint": ARM_CKPT, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"WROTE {out}", flush=True)


if __name__ == "__main__":
    main()
