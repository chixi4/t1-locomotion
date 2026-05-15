import os
import re
import sys

import isaacgym  # noqa: F401
from isaacgym import gymtorch
import torch

sys.path.insert(0, os.getcwd())

from artifacts.autoresearch.eval_shoulder4_frozen import clean_start, load_cfg  # noqa: E402
from envs import *  # noqa: F401,F403,E402


TASK = "T1Shoulder4GaitPhaseLateralOutwardSignFix_from7000LegFrozen_train400"


def refresh(env):
    env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(env.dof_state))
    env.gym.simulate(env.sim)
    if env.device == "cpu":
        env.gym.fetch_results(env.sim, True)
    env.gym.refresh_dof_state_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.base_pos[:] = env.root_states[:, 0:3]
    env.base_quat[:] = env.root_states[:, 3:7]
    env._refresh_feet_state()
    env._compute_observations()


def set_rolls(env, left_roll, right_roll):
    li = env.dof_names.index("Left_Shoulder_Roll")
    ri = env.dof_names.index("Right_Shoulder_Roll")
    env.dof_pos[:] = env.default_dof_pos
    env.dof_vel.zero_()
    env.dof_pos[:, li] = env.default_dof_pos[:, li] + left_roll
    env.dof_pos[:, ri] = env.default_dof_pos[:, ri] + right_roll
    refresh(env)


def body_y_report(env, label):
    body_names = env.gym.get_actor_rigid_body_names(env.envs[0], env.actor_handles[0])
    y = {}
    for i, name in enumerate(body_names):
        if re.match(r"A[LR]\d+$", name):
            y[name] = float(env.body_states[0, i, 1].detach().cpu())
    left_names = sorted([n for n in y if n.startswith("AL")], key=lambda n: int(n[2:]))
    right_names = sorted([n for n in y if n.startswith("AR")], key=lambda n: int(n[2:]))
    left_tip = left_names[-1] if left_names else None
    right_tip = right_names[-1] if right_names else None
    left_out = y[left_tip] - y[left_names[0]] if left_tip else 0.0
    right_out = y[right_names[0]] - y[right_tip] if right_tip else 0.0
    print(label, flush=True)
    print("  arm_y", " ".join(f"{k}={y[k]:+.4f}" for k in left_names + right_names), flush=True)
    print(f"  outward_proxy left_tip_minus_AL1={left_out:+.4f} right_AR1_minus_tip={right_out:+.4f}", flush=True)


def reward_report(env, label, left_roll, right_roll):
    set_rolls(env, left_roll, right_roll)
    env.commands[:, 0] = 0.0
    env.commands[:, 1] = 1.0
    env.commands[:, 2] = 0.0
    env._compute_observations()
    raw = env._reward_shoulder_lateral_roll_outward()
    scaled = raw * env.reward_scales["shoulder_lateral_roll_outward"]
    print(
        f"reward {label} q=({left_roll:+.3f},{right_roll:+.3f}) "
        f"raw_mean={float(raw.mean()):.6f} scaled_mean={float(scaled.mean()):.6f}",
        flush=True,
    )


def main():
    cfg = load_cfg(TASK, 1, 20260510, True, 8.0)
    env = eval(TASK)(cfg)
    clean_start(env)
    print("dof_names", env.dof_names, flush=True)
    print("body_names", env.gym.get_actor_rigid_body_names(env.envs[0], env.actor_handles[0]), flush=True)
    print("configured signs", cfg["rewards"]["shoulder_lateral_roll_left_sign"], cfg["rewards"]["shoulder_lateral_roll_right_sign"], flush=True)
    print("shoulder_roll scale/soft_limit", cfg["rewards"]["scales"]["shoulder_roll"], cfg["rewards"]["shoulder_roll_soft_limit"], flush=True)

    for label, left, right in [
        ("default", 0.0, 0.0),
        ("left +0.40 only", 0.40, 0.0),
        ("left -0.40 only", -0.40, 0.0),
        ("right +0.40 only", 0.0, 0.40),
        ("right -0.40 only", 0.0, -0.40),
        ("pair +left/-right", 0.40, -0.40),
        ("pair -left/+right", -0.40, 0.40),
    ]:
        set_rolls(env, left, right)
        body_y_report(env, label)

    for label, left, right in [
        ("yaml_target_base", 0.22, -0.22),
        ("observed_policy_direction", -0.22, 0.22),
        ("zero", 0.0, 0.0),
        ("big_yaml_target", 0.40, -0.40),
        ("big_observed_direction", -0.40, 0.40),
    ]:
        reward_report(env, label, left, right)


if __name__ == "__main__":
    main()
