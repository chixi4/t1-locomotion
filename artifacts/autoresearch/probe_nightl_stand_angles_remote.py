#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())

from artifacts.autoresearch import eval_shoulder4_frozen as ev  # noqa: E402
import torch  # noqa: E402


def stats(values):
    values = sorted(float(v) for v in values)
    if not values:
        return {"mean": 0.0, "p05": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    n = len(values)
    return {
        "mean": sum(values) / n,
        "p05": values[int(0.05 * (n - 1))],
        "p50": values[int(0.50 * (n - 1))],
        "p95": values[int(0.95 * (n - 1))],
        "min": values[0],
        "max": values[-1],
    }


def main():
    task = "T1Shoulder4GaitPhaseNightLFourGroupClean_from7000LegFrozen_train400"
    arm_checkpoint = sys.argv[1] if len(sys.argv) > 1 else "logs/2026-05-12-17-50-29/nn/model_300.pth"
    leg_checkpoint = "logs/2026-05-05-11-09-07/nn/model_4000.pth"
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0

    ev.set_seed(20260512)
    cfg = ev.load_cfg(task, 1, 20260512, True, seconds + 2.0)
    env = getattr(ev, task)(cfg)
    device = cfg["basic"]["rl_device"]
    arm_model, leg_model = ev.load_models(
        env,
        arm_checkpoint,
        leg_checkpoint,
        device,
        float(cfg["algorithm"].get("logstd_init", -3.0)),
        ev.actor_mean_scale_from_cfg(env, cfg),
        cfg["algorithm"].get("logstd_min"),
        cfg["algorithm"].get("logstd_max"),
    )
    obs = ev.clean_start(env).to(env.device)
    last_leg_action = torch.zeros(env.num_envs, len(env.leg_indices), dtype=torch.float, device=env.device)
    steps = int(round(seconds / env.dt))
    samples = []

    q0 = (env.dof_pos[:, env.arm_indices] - env.default_dof_pos[:, env.arm_indices]).detach().cpu()[0].tolist()
    body_names = env.gym.get_actor_rigid_body_names(env.envs[0], env.actor_handles[0])
    body_state0 = env.body_states.detach().cpu()[0]
    trunk_idx = body_names.index("Trunk")
    al2_idx = body_names.index("AL2")
    ar2_idx = body_names.index("AR2")

    with torch.no_grad():
        for step in range(steps):
            env.commands[:, :3] = 0.0
            env.gait_frequency[:] = 0.0
            env.cmd_resample_time[:] = 10**9
            env._compute_observations()
            obs = env.obs_buf.to(env.device)
            full_action, arm_action, leg_action = ev.policy_step(env, arm_model, leg_model, obs, last_leg_action)
            obs, _, done, _ = env.step(full_action)
            last_leg_action[:] = leg_action
            last_leg_action[done] = 0.0
            q = (env.dof_pos[:, env.arm_indices] - env.default_dof_pos[:, env.arm_indices]).detach().cpu()[0].tolist()
            a = arm_action.detach().cpu()[0].tolist()
            samples.append({"t": (step + 1) * env.dt, "q": q, "action": a})

    by_name = {
        "left_pitch": [s["q"][0] for s in samples],
        "left_roll": [s["q"][1] for s in samples],
        "right_pitch": [s["q"][2] for s in samples],
        "right_roll": [s["q"][3] for s in samples],
        "left_roll_action": [s["action"][1] for s in samples],
        "right_roll_action": [s["action"][3] for s in samples],
    }
    trunk = body_state0[trunk_idx, :3].tolist()
    al2 = body_state0[al2_idx, :3].tolist()
    ar2 = body_state0[ar2_idx, :3].tolist()
    result = {
        "checkpoint": arm_checkpoint,
        "seconds": seconds,
        "dt": env.dt,
        "q_order": ["Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Right_Shoulder_Pitch", "Right_Shoulder_Roll"],
        "clean_start_q_rad": q0,
        "first_step_q_rad": samples[0]["q"] if samples else q0,
        "first_step_action": samples[0]["action"] if samples else [],
        "window_q_rad": {name: stats(vals) for name, vals in by_name.items() if not name.endswith("_action")},
        "window_q_deg": {name: {k: v * 180.0 / math.pi for k, v in stats(vals).items()} for name, vals in by_name.items() if not name.endswith("_action")},
        "window_action": {name: stats(vals) for name, vals in by_name.items() if name.endswith("_action")},
        "clean_start_body_offsets_m": {
            "AL2_y_minus_trunk_y": al2[1] - trunk[1],
            "AR2_outward_y": trunk[1] - ar2[1],
            "AL2_z_minus_trunk_z": al2[2] - trunk[2],
            "AR2_z_minus_trunk_z": ar2[2] - trunk[2],
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
