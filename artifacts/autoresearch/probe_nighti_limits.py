import json
import sys
from pathlib import Path

import isaacgym  # noqa: F401
import torch

sys.path.insert(0, str(Path("artifacts/autoresearch").resolve()))
import eval_shoulder4_frozen as ev  # noqa: E402


TASK = "T1Shoulder4GaitPhaseNightILowBaseHighDynamic_from7000LegFrozen_train400"
ARM_CKPT = "logs/2026-05-12-11-50-54/nn/model_400.pth"
LEG_CKPT = "logs/2026-05-05-11-09-07/nn/model_4000.pth"


def main():
    commands = [
        ("left_1.5", 0.0, 1.5, 0.0),
        ("right_1.5", 0.0, -1.5, 0.0),
        ("diag_fl_1.5", 1.0606601718, 1.0606601718, 0.0),
        ("diag_fr_1.5", 1.0606601718, -1.0606601718, 0.0),
        ("diag_br_1.5", -1.0606601718, -1.0606601718, 0.0),
        ("right1.0_yaw_r0.75", 0.0, -1.0, -0.75),
    ]
    ev.set_seed(20260512)
    cfg = ev.load_cfg(TASK, 256, 20260512, True, 8.0)
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

    steps = int(round(8.0 / env.dt))
    warmup = int(round(2.0 / env.dt))
    results = []
    for name, vx, vy, wz in commands:
        ev.reset_env(env)
        obs = ev.apply_command(env, (vx, vy, wz), 1.5).to(env.device)
        last_leg_action = torch.zeros(env.num_envs, len(env.leg_indices), dtype=torch.float, device=env.device)
        left_min = 1.0e9
        left_max = -1.0e9
        right_min = 1.0e9
        right_max = -1.0e9
        left_near_28 = left_near_295 = right_near_28 = right_near_295 = 0
        left_act_sat = right_act_sat = any_act_sat = 0
        total = 0
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
                left = q[:, 1]
                right = q[:, 3]
                total += left.numel()
                left_min = min(left_min, float(left.min().item()))
                left_max = max(left_max, float(left.max().item()))
                right_min = min(right_min, float(right.min().item()))
                right_max = max(right_max, float(right.max().item()))
                left_near_28 += int((left > 0.28).sum().item())
                left_near_295 += int((left > 0.295).sum().item())
                right_near_28 += int((right < -0.28).sum().item())
                right_near_295 += int((right < -0.295).sum().item())
                left_act_sat += int((arm_action[:, 1] > 0.98).sum().item())
                right_act_sat += int((arm_action[:, 3] < -0.98).sum().item())
                any_act_sat += int((torch.abs(arm_action) > 0.98).any(dim=1).sum().item())
        row = {
            "name": name,
            "left_roll_min": left_min,
            "left_roll_max": left_max,
            "right_roll_min": right_min,
            "right_roll_max": right_max,
            "left_margin_to_0.3": 0.3 - left_max,
            "right_margin_to_-0.3": right_min + 0.3,
            "left_frac_gt_0.28": left_near_28 / total,
            "left_frac_gt_0.295": left_near_295 / total,
            "right_frac_lt_-0.28": right_near_28 / total,
            "right_frac_lt_-0.295": right_near_295 / total,
            "left_roll_action_sat_frac": left_act_sat / total,
            "right_roll_action_sat_frac": right_act_sat / total,
            "any_arm_action_sat_frac": any_act_sat / total,
            "samples": total,
        }
        results.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    out = Path("artifacts/autoresearch/shoulder4_night_i_lowbase_highdynamic400_limit_probe.json")
    out.write_text(json.dumps({"task": TASK, "arm_checkpoint": ARM_CKPT, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"WROTE {out}", flush=True)


if __name__ == "__main__":
    main()
