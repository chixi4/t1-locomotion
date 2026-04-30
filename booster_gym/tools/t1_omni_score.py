import argparse
import glob
import json
import os
import sys
import time

import isaacgym
import imageio
import torch
import yaml

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from envs import *
from envs.t1_omni_stages import apply_omni_stage
from utils.model import ActorCritic


DEFAULT_SCORE_STEPS = 1500
DEFAULT_NUM_ENVS = 64
COMMAND_EPS = 1.0e-6
TRACKING_TOLERANCE = 0.12
YAW_TOLERANCE = 0.18
STANDING_DRIFT_TOLERANCE = 0.10
SYMMETRY_TOLERANCE = 0.10
STEP_LENGTH_TOLERANCE = 0.15
CLEARANCE_TOLERANCE = 0.15
SLIP_ASYMMETRY_TOLERANCE = 0.15
ACTION_ASYMMETRY_TOLERANCE = 0.10
VIDEO_FPS = 50


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="T1", type=str)
    parser.add_argument("--config", type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--stage", type=str)
    parser.add_argument("--num_envs", default=DEFAULT_NUM_ENVS, type=int)
    parser.add_argument("--steps", default=DEFAULT_SCORE_STEPS, type=int)
    parser.add_argument("--vx", default=0.0, type=float)
    parser.add_argument("--vy", default=0.0, type=float)
    parser.add_argument("--yaw", default=0.0, type=float)
    parser.add_argument("--gait_frequency", default=1.5, type=float)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--out", type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_cfg(args)
    checkpoint = resolve_checkpoint(args.checkpoint)
    env = eval(cfg["basic"]["task"])(cfg)
    model = load_model(cfg, checkpoint)
    obs, infos = env.reset()
    apply_fixed_command(env, args)
    counts = run_policy(env, model, obs, args)
    metrics = scalar_metrics(env.omni_metrics.step_metrics())
    report = build_report(args, checkpoint, metrics, counts)
    write_report(report, args.out)
    if args.video:
        write_video(env, report)
    print(json.dumps(report, indent=2, sort_keys=True))


def load_cfg(args):
    cfg_file = args.config or os.path.join("envs", f"{args.task}.yaml")
    with open(cfg_file, "r", encoding="utf-8") as file:
        cfg = yaml.load(file.read(), Loader=yaml.FullLoader)
    if args.stage:
        apply_omni_stage(cfg, args.stage)
    cfg["basic"]["task"] = args.task
    cfg["basic"]["checkpoint"] = args.checkpoint
    cfg["basic"]["headless"] = True
    cfg["env"]["num_envs"] = args.num_envs
    cfg["runner"]["use_wandb"] = False
    cfg["viewer"]["record_video"] = args.video
    return cfg


def resolve_checkpoint(checkpoint):
    if checkpoint not in {"-1", -1}:
        return checkpoint
    matches = sorted(glob.glob(os.path.join("logs", "**/*.pth"), recursive=True), key=os.path.getmtime)
    if not matches:
        raise FileNotFoundError("No checkpoint found under booster_gym/logs")
    return matches[-1]


def load_model(cfg, checkpoint):
    model = ActorCritic(cfg["env"]["num_actions"], cfg["env"]["num_observations"], cfg["env"]["num_privileged_obs"])
    model_dict = torch.load(checkpoint, map_location=cfg["basic"]["rl_device"], weights_only=True)
    model.load_state_dict(model_dict["model"])
    model.to(cfg["basic"]["rl_device"])
    model.eval()
    return model


def run_policy(env, model, obs, args):
    obs = obs.to(env.cfg["basic"]["rl_device"])
    counts = {"steps": 0, "resets": 0, "falls": 0, "timeouts": 0}
    for _ in range(args.steps):
        with torch.no_grad():
            actions = model.act(obs).loc
        obs, rew, done, infos = env.step(actions)
        obs = obs.to(env.cfg["basic"]["rl_device"])
        update_counts(counts, done, infos["time_outs"])
        apply_fixed_command(env, args)
    return counts


def apply_fixed_command(env, args):
    command = torch.tensor([args.vx, args.vy, args.yaw], dtype=torch.float, device=env.device)
    env.commands[:, :] = command.unsqueeze(0)
    moving = abs(args.vx) + abs(args.vy) + abs(args.yaw) > COMMAND_EPS
    env.gait_frequency[:] = args.gait_frequency if moving else 0.0
    env.cmd_resample_time[:] = env.episode_length_buf + args.steps + 1


def update_counts(counts, done, time_outs):
    done_cpu = done.detach().cpu()
    timeout_cpu = time_outs.detach().cpu()
    counts["steps"] += 1
    counts["resets"] += int(done_cpu.sum().item())
    counts["timeouts"] += int((done_cpu & timeout_cpu).sum().item())
    counts["falls"] += int((done_cpu & ~timeout_cpu).sum().item())


def scalar_metrics(metrics):
    return {key: float(value.detach().cpu().item()) if torch.is_tensor(value) else float(value) for key, value in metrics.items()}


def build_report(args, checkpoint, metrics, counts):
    return {
        "checkpoint": checkpoint,
        "command": {"vx": args.vx, "vy": args.vy, "yaw": args.yaw},
        "num_envs": args.num_envs,
        "steps": args.steps,
        "counts": counts,
        "metrics": metrics,
        "judgement": judge(args, metrics, counts),
    }


def judge(args, metrics, counts):
    tracking_ok = metrics["Metrics/command_error_x"] < TRACKING_TOLERANCE
    tracking_ok &= metrics["Metrics/command_error_y"] < TRACKING_TOLERANCE
    tracking_ok &= metrics["Metrics/command_error_yaw"] < YAW_TOLERANCE
    symmetry_ok = metrics["Metrics/stance_time_asymmetry"] < SYMMETRY_TOLERANCE
    symmetry_ok &= metrics["Metrics/step_length_asymmetry"] < STEP_LENGTH_TOLERANCE
    symmetry_ok &= metrics["Metrics/foot_clearance_asymmetry"] < CLEARANCE_TOLERANCE
    symmetry_ok &= metrics["Metrics/foot_slip_asymmetry"] < SLIP_ASYMMETRY_TOLERANCE
    symmetry_ok &= metrics["Metrics/leg_action_magnitude_asymmetry"] < ACTION_ASYMMETRY_TOLERANCE
    command_norm = abs(args.vx) + abs(args.vy) + abs(args.yaw)
    standing_ok = command_norm > COMMAND_EPS or metrics["Metrics/standing_policy_speed"] < STANDING_DRIFT_TOLERANCE
    stable_ok = counts["falls"] == 0
    return {
        "tracking_ok": bool(tracking_ok),
        "symmetry_ok": bool(symmetry_ok),
        "standing_ok": bool(standing_ok),
        "stable_ok": bool(stable_ok),
    }


def write_report(report, out_path):
    path = out_path or default_report_path(report)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, sort_keys=True)
    report["report_path"] = path


def default_report_path(report):
    stamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    command = report["command"]
    raw_name = f"score_vx{command['vx']}_vy{command['vy']}_yaw{command['yaw']}_{stamp}"
    safe_name = raw_name.replace("-", "m").replace(".", "p")
    return os.path.join("artifacts", "omni_scores", safe_name + ".json")


def write_video(env, report):
    os.makedirs("videos", exist_ok=True)
    command = report["command"]
    stamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    raw_name = f"t1_omni_vx{command['vx']}_vy{command['vy']}_yaw{command['yaw']}_{stamp}"
    path = os.path.join("videos", raw_name.replace("-", "m").replace(".", "p") + ".mp4")
    with imageio.get_writer(path, fps=VIDEO_FPS) as writer:
        for frame in env.camera_frames:
            writer.append_data(frame)
    report["video_path"] = path


if __name__ == "__main__":
    main()
