import os
import sys

import isaacgym  # noqa: F401
import torch

sys.path.insert(0, os.getcwd())

from artifacts.autoresearch.eval_shoulder4_frozen import (  # noqa: E402
    actor_mean_scale_from_cfg,
    apply_command,
    clean_start,
    load_cfg,
    load_models,
    policy_step,
)
from envs import *  # noqa: F401,F403,E402


def main():
    task = "T1Shoulder4GaitPhaseLateralOutwardSignFix_from7000LegFrozen_train400"
    arm_ckpt = "logs/2026-05-10-12-29-22/nn/model_400.pth"
    leg_ckpt = "logs/2026-05-05-11-09-07/nn/model_4000.pth"
    cfg = load_cfg(task, 128, 20260510, True, 8.0)
    env = eval(task)(cfg)
    arm_model, leg_model = load_models(
        env,
        arm_ckpt,
        leg_ckpt,
        cfg["basic"]["rl_device"],
        float(cfg["algorithm"].get("logstd_init", -3.0)),
        actor_mean_scale_from_cfg(env, cfg),
        cfg["algorithm"].get("logstd_min"),
        cfg["algorithm"].get("logstd_max"),
    )

    arm_names = [env.dof_names[i] for i in env.arm_indices.tolist()]
    print("arm_dofs", arm_names, flush=True)
    commands = [
        ("stand", 0.0, 0.0, 0.0),
        ("left1", 0.0, 1.0, 0.0),
        ("right1", 0.0, -1.0, 0.0),
        ("diag_fl", 0.7071, 0.7071, 0.0),
        ("diag_fr", 0.7071, -0.7071, 0.0),
    ]
    for name, vx, vy, wz in commands:
        obs = clean_start(env)
        last_leg_action = torch.zeros(env.num_envs, len(env.leg_indices), dtype=torch.float, device=env.device)
        obs = apply_command(env, (vx, vy, wz), 1.8)
        q_samples = []
        action_samples = []
        for step in range(180):
            full_action, arm_action, leg_action = policy_step(env, arm_model, leg_model, obs, last_leg_action)
            obs, _, done, _ = env.step(full_action)
            last_leg_action[:] = leg_action
            last_leg_action[done] = 0.0
            if step >= 60:
                q = env.dof_pos[:, env.arm_indices] - env.default_dof_pos[:, env.arm_indices]
                q_samples.append(q.detach().cpu())
                action_samples.append(arm_action.detach().cpu())
        q_all = torch.cat(q_samples, dim=0)
        action_all = torch.cat(action_samples, dim=0)
        print(f"{name} cmd=({vx:.4f},{vy:.4f},{wz:.4f})", flush=True)
        for idx, dof_name in enumerate(arm_names):
            print(
                "  {} q_mean={:.5f} q_p05={:.5f} q_p95={:.5f} act_mean={:.5f}".format(
                    dof_name,
                    float(q_all[:, idx].mean()),
                    float(torch.quantile(q_all[:, idx], 0.05)),
                    float(torch.quantile(q_all[:, idx], 0.95)),
                    float(action_all[:, idx].mean()),
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
