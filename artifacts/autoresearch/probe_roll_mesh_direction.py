import os
import struct
import sys

import isaacgym  # noqa: F401
from isaacgym import gymtorch
import numpy as np
import torch

sys.path.insert(0, os.getcwd())

from artifacts.autoresearch.eval_shoulder4_frozen import clean_start, load_cfg  # noqa: E402
from envs import *  # noqa: F401,F403,E402


TASK = "T1Shoulder4GaitPhaseLateralOutwardSignFix_from7000LegFrozen_train400"


def read_stl_vertices(path):
    data = open(path, "rb").read()
    if len(data) >= 84:
        tri_count = struct.unpack("<I", data[80:84])[0]
        if 84 + 50 * tri_count == len(data):
            vertices = []
            off = 84
            for _ in range(tri_count):
                off += 12
                for _ in range(3):
                    vertices.append(struct.unpack("<fff", data[off : off + 12]))
                    off += 12
                off += 2
            return np.asarray(vertices, dtype=np.float64)
    vertices = []
    for line in data.decode("utf-8", errors="ignore").splitlines():
        line = line.strip()
        if line.startswith("vertex "):
            vertices.append(tuple(float(v) for v in line.split()[1:4]))
    return np.asarray(vertices, dtype=np.float64)


def quat_to_rot(q):
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def refresh(env):
    env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(env.dof_state))
    env.gym.simulate(env.sim)
    if env.device == "cpu":
        env.gym.fetch_results(env.sim, True)
    env.gym.refresh_dof_state_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)
    env.gym.refresh_actor_root_state_tensor(env.sim)


def set_rolls(env, left_roll, right_roll):
    li = env.dof_names.index("Left_Shoulder_Roll")
    ri = env.dof_names.index("Right_Shoulder_Roll")
    env.dof_pos[:] = env.default_dof_pos
    env.dof_vel.zero_()
    env.dof_pos[:, li] = env.default_dof_pos[:, li] + left_roll
    env.dof_pos[:, ri] = env.default_dof_pos[:, ri] + right_roll
    refresh(env)


def transformed_mesh_y(env, body_name, vertices):
    body_names = env.gym.get_actor_rigid_body_names(env.envs[0], env.actor_handles[0])
    idx = body_names.index(body_name)
    state = env.body_states[0, idx, :7].detach().cpu().numpy().astype(np.float64)
    pos = state[:3]
    rot = quat_to_rot(state[3:7])
    world = vertices @ rot.T + pos
    return {
        "mean": float(world[:, 1].mean()),
        "p05": float(np.quantile(world[:, 1], 0.05)),
        "p95": float(np.quantile(world[:, 1], 0.95)),
        "min": float(world[:, 1].min()),
        "max": float(world[:, 1].max()),
    }


def main():
    cfg = load_cfg(TASK, 1, 20260510, True, 8.0)
    env = eval(TASK)(cfg)
    clean_start(env)
    al2 = read_stl_vertices("resources/T1/meshes/AL2.STL")
    ar2 = read_stl_vertices("resources/T1/meshes/AR2.STL")
    print("mesh_bbox_AL2", al2.min(axis=0).round(5).tolist(), al2.max(axis=0).round(5).tolist(), flush=True)
    print("mesh_bbox_AR2", ar2.min(axis=0).round(5).tolist(), ar2.max(axis=0).round(5).tolist(), flush=True)
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
        left_y = transformed_mesh_y(env, "AL2", al2)
        right_y = transformed_mesh_y(env, "AR2", ar2)
        print(label, flush=True)
        print("  AL2_world_y", left_y, flush=True)
        print("  AR2_world_y", right_y, flush=True)


if __name__ == "__main__":
    main()
