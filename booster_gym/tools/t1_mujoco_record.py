import argparse
import importlib.util
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio
import mujoco
import numpy as np
import torch
import yaml

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from utils.model import ActorCritic


DEFAULT_STEPS = 1500
DEFAULT_FPS = 50
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--stage", type=str)
    parser.add_argument("--vx", default=0.0, type=float)
    parser.add_argument("--vy", default=0.0, type=float)
    parser.add_argument("--yaw", default=0.0, type=float)
    parser.add_argument("--steps", default=DEFAULT_STEPS, type=int)
    parser.add_argument("--fps", default=DEFAULT_FPS, type=int)
    parser.add_argument("--width", default=DEFAULT_WIDTH, type=int)
    parser.add_argument("--height", default=DEFAULT_HEIGHT, type=int)
    parser.add_argument("--camera", default="follow", choices=["follow", "orbit", "side", "front", "top", "wide_top"])
    parser.add_argument("--frames_dir", type=str)
    parser.add_argument("--out", type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_cfg(args.stage)
    policy = load_policy(cfg, args.checkpoint)
    sim = create_sim(cfg)
    path = args.out or default_video_path(args)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    record_policy(cfg, policy, sim, args, path)
    print(path)


def load_cfg(stage):
    with open(os.path.join("envs", "T1.yaml"), "r", encoding="utf-8") as file:
        cfg = yaml.load(file.read(), Loader=yaml.FullLoader)
    if stage:
        load_stage_module().apply_omni_stage(cfg, stage)
    return cfg


def load_stage_module():
    path = os.path.join(REPO_DIR, "envs", "t1_omni_stages.py")
    spec = importlib.util.spec_from_file_location("t1_omni_stages_for_mujoco", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_policy(cfg, checkpoint):
    policy = ActorCritic(cfg["env"]["num_actions"], cfg["env"]["num_observations"], cfg["env"]["num_privileged_obs"])
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    policy.load_state_dict(state["model"])
    policy.eval()
    return policy


def create_sim(cfg):
    model = mujoco.MjModel.from_xml_path(cfg["asset"]["mujoco_file"])
    model.opt.timestep = cfg["sim"]["dt"]
    data = mujoco.MjData(model)
    default_dof_pos, dof_stiffness, dof_damping = actuator_defaults(cfg, model)
    reset_pose(cfg, model, data, default_dof_pos)
    return {
        "model": model,
        "data": data,
        "default_dof_pos": default_dof_pos,
        "dof_stiffness": dof_stiffness,
        "dof_damping": dof_damping,
    }


def actuator_defaults(cfg, model):
    default_dof_pos = np.zeros(model.nu, dtype=np.float32)
    dof_stiffness = np.zeros(model.nu, dtype=np.float32)
    dof_damping = np.zeros(model.nu, dtype=np.float32)
    for index in range(model.nu):
        actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
        default_dof_pos[index] = named_value(cfg["init_state"]["default_joint_angles"], actuator_name)
        dof_stiffness[index] = named_value(cfg["control"]["stiffness"], actuator_name, required=True)
        dof_damping[index] = named_value(cfg["control"]["damping"], actuator_name, required=True)
    return default_dof_pos, dof_stiffness, dof_damping


def named_value(values, name, required=False):
    for key, value in values.items():
        if key in name:
            return value
    if required:
        raise ValueError(f"No configured value for {name}")
    return values["default"]


def reset_pose(cfg, model, data, default_dof_pos):
    data.qpos[:] = np.concatenate(
        [
            np.array(cfg["init_state"]["pos"], dtype=np.float32),
            np.array(cfg["init_state"]["rot"][3:4] + cfg["init_state"]["rot"][0:3], dtype=np.float32),
            default_dof_pos,
        ]
    )
    mujoco.mj_forward(model, data)


def record_policy(cfg, policy, sim, args, path):
    model = sim["model"]
    data = sim["data"]
    model.vis.global_.offwidth = args.width
    model.vis.global_.offheight = args.height
    actions = np.zeros(cfg["env"]["num_actions"], dtype=np.float32)
    targets = sim["default_dof_pos"].copy()
    gait_frequency = command_frequency(cfg, args)
    render_stride = max(1, int(round(1.0 / (args.fps * cfg["sim"]["dt"]))))
    total_frames = max(1, int(np.ceil(args.steps / render_stride)))
    frame_ids = screenshot_frame_ids(total_frames, args.frames_dir)
    with mujoco.Renderer(model, args.height, args.width) as renderer:
        with imageio.get_writer(path, fps=args.fps) as writer:
            frame_index = 0
            for step in range(args.steps):
                actions, targets = policy_step(cfg, policy, sim, args, actions, targets, gait_frequency, step)
                apply_pd_control(sim, targets)
                mujoco.mj_step(model, data)
                if step % render_stride == 0:
                    frame = render_frame(renderer, sim, args, step)
                    writer.append_data(frame)
                    write_screenshot(args.frames_dir, frame_ids, frame_index, frame)
                    frame_index += 1


def command_frequency(cfg, args):
    if abs(args.vx) + abs(args.vy) + abs(args.yaw) <= 1.0e-6:
        return 0.0
    return float(np.average(cfg["commands"]["gait_frequency"]))


def policy_step(cfg, policy, sim, args, actions, targets, gait_frequency, step):
    if step % cfg["control"]["decimation"] != 0:
        return actions, targets
    obs = build_obs(cfg, sim, args, actions, gait_frequency, step)
    with torch.no_grad():
        loc = policy.act(torch.tensor(obs).unsqueeze(0)).loc.squeeze(0).numpy()
    actions = np.clip(loc, -cfg["normalization"]["clip_actions"], cfg["normalization"]["clip_actions"])
    targets = sim["default_dof_pos"] + cfg["control"]["action_scale"] * actions
    return actions.astype(np.float32), targets.astype(np.float32)


def build_obs(cfg, sim, args, actions, gait_frequency, step):
    data = sim["data"]
    obs = np.zeros(cfg["env"]["num_observations"], dtype=np.float32)
    quat = data.sensor("orientation").data[[1, 2, 3, 0]].astype(np.float32)
    gait_process = np.fmod(step * cfg["sim"]["dt"] * gait_frequency, 1.0)
    obs[0:3] = quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0])) * cfg["normalization"]["gravity"]
    obs[3:6] = data.sensor("angular-velocity").data.astype(np.float32) * cfg["normalization"]["ang_vel"]
    obs[6:9] = np.array([args.vx, args.vy, args.yaw], dtype=np.float32)
    obs[6:8] *= cfg["normalization"]["lin_vel"]
    obs[8] *= cfg["normalization"]["ang_vel"]
    obs[9] = np.cos(2 * np.pi * gait_process) * (gait_frequency > 1.0e-8)
    obs[10] = np.sin(2 * np.pi * gait_process) * (gait_frequency > 1.0e-8)
    obs[11:23] = (data.qpos.astype(np.float32)[7:] - sim["default_dof_pos"]) * cfg["normalization"]["dof_pos"]
    obs[23:35] = data.qvel.astype(np.float32)[6:] * cfg["normalization"]["dof_vel"]
    obs[35:47] = actions
    return obs


def apply_pd_control(sim, targets):
    model = sim["model"]
    data = sim["data"]
    dof_pos = data.qpos.astype(np.float32)[7:]
    dof_vel = data.qvel.astype(np.float32)[6:]
    torque = sim["dof_stiffness"] * (targets - dof_pos) - sim["dof_damping"] * dof_vel
    data.ctrl[:] = np.clip(torque, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])


def render_frame(renderer, sim, args, step):
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance, camera.azimuth, camera.elevation = camera_pose(args.camera, step)
    camera.lookat[:] = sim["data"].qpos.astype(np.float32)[0:3]
    renderer.update_scene(sim["data"], camera=camera)
    return renderer.render()


def camera_pose(camera_name, step):
    if camera_name == "orbit":
        return 3.2, 135.0 + 0.12 * step, -18.0
    if camera_name == "side":
        return 3.0, 90.0, -12.0
    if camera_name == "front":
        return 3.0, 180.0, -12.0
    if camera_name == "top":
        return 5.2, 90.0, -88.0
    if camera_name == "wide_top":
        return 6.5, 135.0, -65.0
    return 3.0, 135.0, -18.0


def screenshot_frame_ids(total_frames, frames_dir):
    if not frames_dir:
        return set()
    os.makedirs(frames_dir, exist_ok=True)
    return {0, total_frames // 2, max(0, total_frames - 1)}


def write_screenshot(frames_dir, frame_ids, frame_index, frame):
    if frames_dir and frame_index in frame_ids:
        imageio.imwrite(os.path.join(frames_dir, f"frame_{frame_index:04d}.png"), frame)


def quat_rotate_inverse(q, v):
    q_w = q[-1]
    q_vec = q[:3]
    return v * (2.0 * q_w**2 - 1.0) - np.cross(q_vec, v) * (2.0 * q_w) + q_vec * (2.0 * np.dot(q_vec, v))


def default_video_path(args):
    stamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    raw_name = f"t1_mujoco_vx{args.vx}_vy{args.vy}_yaw{args.yaw}_{stamp}"
    return os.path.join("videos", raw_name.replace("-", "m").replace(".", "p") + ".mp4")


if __name__ == "__main__":
    main()
