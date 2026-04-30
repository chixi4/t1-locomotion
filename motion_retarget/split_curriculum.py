from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ANCHOR_BODY_NAME = "Trunk"
DEFAULT_MOTION_DIR = Path(__file__).resolve().parents[1] / "data" / "motions" / "T1"
MOTION_DIR = Path(os.environ.get("T1_MOTION_DIR", DEFAULT_MOTION_DIR))
SOURCE_MOTION = MOTION_DIR / "t1_walk_straight.npz"
OUTPUT_NAMES = {
    "stand": "stage_stand_hold.npz",
    "slow": "stage_walk_slow_fwd.npz",
    "medium": "stage_walk_mid_fwd.npz",
    "fast_forward": "stage_walk_fast_fwd_1p5.npz",
    "fast_backward": "stage_walk_fast_back_1p5.npz",
}
STAND_WINDOW_S = (9.2, 14.0)
SLOW_WINDOW_S = (45.9, 60.1)
MEDIUM_WINDOW_S = (87.22, 101.68)
FAST_FORWARD_TIME_SCALE = 1.65
MIN_FRAMES = 2


@dataclass(frozen=True)
class MotionClip:
    fps: float
    body_names: np.ndarray
    joint_names: np.ndarray
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    body_pos_w: np.ndarray
    body_quat_w: np.ndarray
    body_lin_vel_w: np.ndarray
    body_ang_vel_w: np.ndarray

    @property
    def frame_count(self) -> int:
        return int(self.joint_pos.shape[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build staged Booster T1 motion clips for curriculum training.")
    parser.add_argument("--source", type=Path, default=SOURCE_MOTION, help="Source npz motion.")
    parser.add_argument("--output_dir", type=Path, default=MOTION_DIR, help="Directory to save staged motion clips.")
    return parser.parse_args()


def load_motion(path: Path) -> MotionClip:
    data = np.load(path)
    return MotionClip(
        fps=float(np.asarray(data["fps"]).reshape(-1)[0]),
        body_names=data["body_names"],
        joint_names=data["joint_names"],
        joint_pos=data["joint_pos"].astype(np.float32),
        joint_vel=data["joint_vel"].astype(np.float32),
        body_pos_w=data["body_pos_w"].astype(np.float32),
        body_quat_w=data["body_quat_w"].astype(np.float32),
        body_lin_vel_w=data["body_lin_vel_w"].astype(np.float32),
        body_ang_vel_w=data["body_ang_vel_w"].astype(np.float32),
    )


def save_motion(path: Path, clip: MotionClip) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        fps=np.array([clip.fps], dtype=np.float32),
        body_names=clip.body_names,
        joint_names=clip.joint_names,
        joint_pos=clip.joint_pos,
        joint_vel=clip.joint_vel,
        body_pos_w=clip.body_pos_w,
        body_quat_w=clip.body_quat_w,
        body_lin_vel_w=clip.body_lin_vel_w,
        body_ang_vel_w=clip.body_ang_vel_w,
    )


def get_anchor_index(clip: MotionClip) -> int:
    body_names = clip.body_names.tolist()
    if ANCHOR_BODY_NAME not in body_names:
        raise ValueError(f"{ANCHOR_BODY_NAME} not found in motion body_names.")
    return int(body_names.index(ANCHOR_BODY_NAME))


def quat_multiply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.moveaxis(lhs, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(rhs, -1, 0)
    return np.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        axis=-1,
    )


def quat_normalize(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat, axis=-1, keepdims=True).clip(min=1.0e-8)
    return quat / norm


def quat_from_yaw(yaw: float) -> np.ndarray:
    half_yaw = 0.5 * yaw
    return np.array([np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)], dtype=np.float32)


def quat_to_yaw(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = np.moveaxis(quat, -1, 0)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return np.arctan2(siny_cosp, cosy_cosp)


def rotate_xy(array: np.ndarray, yaw: float) -> np.ndarray:
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    x = array[..., 0]
    y = array[..., 1]
    rotated = np.array(array, copy=True)
    rotated[..., 0] = cos_yaw * x - sin_yaw * y
    rotated[..., 1] = sin_yaw * x + cos_yaw * y
    return rotated


def slice_motion(clip: MotionClip, start_s: float, end_s: float) -> MotionClip:
    start = int(round(start_s * clip.fps))
    end = int(round(end_s * clip.fps))
    if end - start < MIN_FRAMES:
        raise ValueError(f"Invalid slice {start_s}-{end_s}s for clip with fps {clip.fps}.")
    return MotionClip(
        fps=clip.fps,
        body_names=clip.body_names,
        joint_names=clip.joint_names,
        joint_pos=clip.joint_pos[start:end].copy(),
        joint_vel=clip.joint_vel[start:end].copy(),
        body_pos_w=clip.body_pos_w[start:end].copy(),
        body_quat_w=clip.body_quat_w[start:end].copy(),
        body_lin_vel_w=clip.body_lin_vel_w[start:end].copy(),
        body_ang_vel_w=clip.body_ang_vel_w[start:end].copy(),
    )


def rebase_motion(clip: MotionClip) -> MotionClip:
    anchor_index = get_anchor_index(clip)
    yaw0 = float(quat_to_yaw(clip.body_quat_w[0, anchor_index]))
    yaw_inv = quat_from_yaw(-yaw0)
    yaw_stack = np.broadcast_to(yaw_inv, clip.body_quat_w.shape)

    body_pos = clip.body_pos_w.copy()
    xy_origin = body_pos[0, anchor_index, :2].copy()
    body_pos[..., :2] -= xy_origin
    body_pos = rotate_xy(body_pos, -yaw0)

    body_lin_vel = rotate_xy(clip.body_lin_vel_w, -yaw0)
    body_ang_vel = rotate_xy(clip.body_ang_vel_w, -yaw0)
    body_quat = quat_normalize(quat_multiply(yaw_stack, clip.body_quat_w))

    return MotionClip(
        fps=clip.fps,
        body_names=clip.body_names,
        joint_names=clip.joint_names,
        joint_pos=clip.joint_pos.copy(),
        joint_vel=clip.joint_vel.copy(),
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        body_lin_vel_w=body_lin_vel,
        body_ang_vel_w=body_ang_vel,
    )


def interpolate_array(array: np.ndarray, sample_points: np.ndarray) -> np.ndarray:
    frame_ids = np.arange(array.shape[0], dtype=np.float32)
    flat = array.reshape(array.shape[0], -1)
    interpolated = np.empty((sample_points.shape[0], flat.shape[1]), dtype=np.float32)
    for index in range(flat.shape[1]):
        interpolated[:, index] = np.interp(sample_points, frame_ids, flat[:, index])
    return interpolated.reshape((sample_points.shape[0],) + array.shape[1:])


def interpolate_quaternions(quat: np.ndarray, sample_points: np.ndarray) -> np.ndarray:
    base_ids = np.arange(quat.shape[0], dtype=np.float32)
    lower = np.floor(sample_points).astype(np.int32)
    upper = np.clip(lower + 1, 0, quat.shape[0] - 1)
    alpha = (sample_points - base_ids[lower]).astype(np.float32)[:, None, None]
    q0 = quat[lower]
    q1 = quat[upper]
    flip_mask = np.sum(q0 * q1, axis=-1, keepdims=True) < 0.0
    q1 = np.where(flip_mask, -q1, q1)
    blended = (1.0 - alpha) * q0 + alpha * q1
    return quat_normalize(blended.astype(np.float32))


def time_scale_motion(clip: MotionClip, scale: float) -> MotionClip:
    new_frames = max(MIN_FRAMES, int(round(clip.frame_count / scale)))
    sample_points = np.linspace(0.0, clip.frame_count - 1, new_frames, dtype=np.float32)
    return MotionClip(
        fps=clip.fps,
        body_names=clip.body_names,
        joint_names=clip.joint_names,
        joint_pos=interpolate_array(clip.joint_pos, sample_points),
        joint_vel=interpolate_array(clip.joint_vel, sample_points) * scale,
        body_pos_w=interpolate_array(clip.body_pos_w, sample_points),
        body_quat_w=interpolate_quaternions(clip.body_quat_w, sample_points),
        body_lin_vel_w=interpolate_array(clip.body_lin_vel_w, sample_points) * scale,
        body_ang_vel_w=interpolate_array(clip.body_ang_vel_w, sample_points) * scale,
    )


def reverse_motion(clip: MotionClip) -> MotionClip:
    return MotionClip(
        fps=clip.fps,
        body_names=clip.body_names,
        joint_names=clip.joint_names,
        joint_pos=clip.joint_pos[::-1].copy(),
        joint_vel=(-clip.joint_vel[::-1]).copy(),
        body_pos_w=clip.body_pos_w[::-1].copy(),
        body_quat_w=clip.body_quat_w[::-1].copy(),
        body_lin_vel_w=(-clip.body_lin_vel_w[::-1]).copy(),
        body_ang_vel_w=(-clip.body_ang_vel_w[::-1]).copy(),
    )


def freeze_motion(clip: MotionClip, duration_s: float) -> MotionClip:
    center_index = clip.frame_count // 2
    frame_count = max(MIN_FRAMES, int(round(duration_s * clip.fps)))
    joint_pos = np.repeat(clip.joint_pos[center_index:center_index + 1], frame_count, axis=0)
    body_pos_w = np.repeat(clip.body_pos_w[center_index:center_index + 1], frame_count, axis=0)
    body_quat_w = np.repeat(clip.body_quat_w[center_index:center_index + 1], frame_count, axis=0)
    joint_vel = np.zeros((frame_count,) + clip.joint_vel.shape[1:], dtype=np.float32)
    body_lin_vel_w = np.zeros((frame_count,) + clip.body_lin_vel_w.shape[1:], dtype=np.float32)
    body_ang_vel_w = np.zeros((frame_count,) + clip.body_ang_vel_w.shape[1:], dtype=np.float32)
    return MotionClip(clip.fps, clip.body_names, clip.joint_names, joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w)


def estimate_local_vx(clip: MotionClip) -> tuple[float, float]:
    anchor_index = get_anchor_index(clip)
    yaw = quat_to_yaw(clip.body_quat_w[:, anchor_index])
    body_vel = clip.body_lin_vel_w[:, anchor_index, :2]
    cos_yaw = np.cos(-yaw)
    sin_yaw = np.sin(-yaw)
    local_vx = cos_yaw * body_vel[:, 0] - sin_yaw * body_vel[:, 1]
    return float(local_vx.mean()), float(np.percentile(local_vx, 90))


def print_summary(name: str, clip: MotionClip) -> None:
    mean_vx, p90_vx = estimate_local_vx(clip)
    duration = clip.frame_count / clip.fps
    print(
        {
            "name": name,
            "frames": clip.frame_count,
            "seconds": round(duration, 2),
            "mean_local_vx": round(mean_vx, 3),
            "p90_local_vx": round(p90_vx, 3),
        }
    )


def main() -> None:
    args = parse_args()
    source = rebase_motion(load_motion(args.source))

    stand = freeze_motion(slice_motion(source, *STAND_WINDOW_S), duration_s=5.0)
    slow = slice_motion(source, *SLOW_WINDOW_S)
    medium = slice_motion(source, *MEDIUM_WINDOW_S)
    fast_forward = time_scale_motion(medium, FAST_FORWARD_TIME_SCALE)
    fast_backward = reverse_motion(fast_forward)

    clips = {
        "stand": stand,
        "slow": slow,
        "medium": medium,
        "fast_forward": fast_forward,
        "fast_backward": fast_backward,
    }
    for name, clip in clips.items():
        output_path = args.output_dir / OUTPUT_NAMES[name]
        save_motion(output_path, clip)
        print_summary(name, clip)
        print(f"[INFO] Saved {output_path}")


if __name__ == "__main__":
    main()
