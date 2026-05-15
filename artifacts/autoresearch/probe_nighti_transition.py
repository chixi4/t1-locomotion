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
    transitions = [
        ("stand_to_forward_1.0", (1.0, 0.0, 0.0)),
        ("stand_to_right_1.5", (0.0, -1.5, 0.0)),
        ("stand_to_diag_fr_1.5", (1.0606601718, -1.0606601718, 0.0)),
    ]
    pre_s = 2.0
    post_s = 4.0
    bin_s = 0.2

    ev.set_seed(20260512)
    cfg = ev.load_cfg(TASK, 256, 20260512, True, pre_s + post_s + 2.0)
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

    steps = int(round((pre_s + post_s) / env.dt))
    bin_steps = max(1, int(round(bin_s / env.dt)))
    results = []
    for name, walk_cmd in transitions:
        ev.reset_env(env)
        obs = ev.apply_command(env, (0.0, 0.0, 0.0), 0.0).to(env.device)
        last_leg_action = torch.zeros(env.num_envs, len(env.leg_indices), dtype=torch.float, device=env.device)
        bins = []
        acc = []
        with torch.no_grad():
            for step in range(steps):
                t = step * env.dt
                cmd = (0.0, 0.0, 0.0) if t < pre_s else walk_cmd
                gait_frequency = 0.0 if t < pre_s else 1.5
                obs = ev.apply_command(env, cmd, gait_frequency).to(env.device)
                full_action, arm_action, leg_action = ev.policy_step(env, arm_model, leg_model, obs, last_leg_action)
                obs, _, done, _ = env.step(full_action)
                last_leg_action[:] = leg_action
                last_leg_action[done] = 0.0
                q = env.dof_pos[:, env.arm_indices] - env.default_dof_pos[:, env.arm_indices]
                left_pitch = q[:, 0]
                left_roll = q[:, 1]
                right_pitch = q[:, 2]
                right_roll = q[:, 3]
                acc.append(
                    {
                        "t": float(t),
                        "left_roll_mean": float(left_roll.mean().item()),
                        "right_roll_mean": float(right_roll.mean().item()),
                        "roll_abs_mean": float(torch.cat([left_roll.abs(), right_roll.abs()]).mean().item()),
                        "roll_abs_p95": float(torch.quantile(torch.cat([left_roll.abs(), right_roll.abs()]), 0.95).item()),
                        "left_pitch_mean": float(left_pitch.mean().item()),
                        "right_pitch_mean": float(right_pitch.mean().item()),
                        "pitch_abs_mean": float(torch.cat([left_pitch.abs(), right_pitch.abs()]).mean().item()),
                        "arm_action_sat_frac": float((torch.abs(arm_action) > 0.98).float().mean().item()),
                    }
                )
                if len(acc) == bin_steps or step == steps - 1:
                    keys = [k for k in acc[0].keys() if k != "t"]
                    row = {
                        "t0": acc[0]["t"],
                        "t1": acc[-1]["t"] + env.dt,
                    }
                    for key in keys:
                        row[key] = sum(item[key] for item in acc) / len(acc)
                    bins.append(row)
                    acc = []
        results.append({"name": name, "walk_cmd": walk_cmd, "pre_s": pre_s, "post_s": post_s, "bins": bins})
        for row in bins:
            if 1.6 <= row["t0"] <= 3.2:
                print(json.dumps({"name": name, **row}, ensure_ascii=False), flush=True)

    out = Path("artifacts/autoresearch/shoulder4_night_i_lowbase_highdynamic400_transition_probe.json")
    out.write_text(json.dumps({"task": TASK, "arm_checkpoint": ARM_CKPT, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"WROTE {out}", flush=True)


if __name__ == "__main__":
    main()
