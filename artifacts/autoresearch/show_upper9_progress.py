#!/usr/bin/env python3
import argparse
import os
import re
import subprocess
import time
from pathlib import Path

import yaml
from tensorboard.backend.event_processing import event_accumulator


def find_run_dir(task):
    candidates = []
    for cfg_path in Path("logs").glob("*/config.yaml"):
        try:
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if cfg.get("basic", {}).get("task") == task:
            candidates.append(cfg_path.parent)
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]


def latest_scalars(run_dir):
    out = {}
    if run_dir is None:
        return out
    event_files = sorted((run_dir / "summaries").glob("events.out.tfevents.*"), key=lambda p: p.stat().st_mtime)
    if not event_files:
        return out
    try:
        ea = event_accumulator.EventAccumulator(str(event_files[-1]), size_guidance={"scalars": 0})
        ea.Reload()
        for tag in ea.Tags().get("scalars", []):
            values = ea.Scalars(tag)
            if values:
                out[tag] = values[-1]
    except Exception as exc:
        out["__error__"] = str(exc)
    return out


def parse_epoch(train_log, final_iter):
    if not train_log or not Path(train_log).exists():
        return None, []
    lines = Path(train_log).read_text(errors="ignore").splitlines()
    epoch_lines = [line for line in lines if line.startswith("epoch:")]
    epoch = None
    if epoch_lines:
        match = re.search(r"epoch:\s+(\d+)/(\d+)", epoch_lines[-1])
        if match:
            epoch = int(match.group(1))
            final_iter = int(match.group(2))
    return (epoch, final_iter), epoch_lines[-10:]


def gpu_line():
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).strip()
        return out
    except Exception:
        return "gpu unavailable"


def fmt_scalar(scalars, tag, precision=4):
    item = scalars.get(tag)
    if item is None:
        return "-"
    return f"{item.value:.{precision}g} @ {item.step}"


def bar(done, total, width=52):
    if not total:
        return "[" + "." * width + "]"
    frac = max(0.0, min(1.0, done / total))
    fill = int(round(frac * width))
    return "[" + "#" * fill + "." * (width - fill) + f"] {frac * 100:5.1f}%"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--final-iter", type=int, required=True)
    parser.add_argument("--train-log", required=True)
    parser.add_argument("--phase", default="train")
    args = parser.parse_args()

    run_dir = find_run_dir(args.task)
    scalars = latest_scalars(run_dir)
    parsed, recent = parse_epoch(args.train_log, args.final_iter)
    epoch = parsed[0] if parsed and parsed[0] is not None else 0
    total = parsed[1] if parsed and parsed[1] is not None else args.final_iter
    if epoch == 0:
        scalar_steps = [item.step for key, item in scalars.items() if not key.startswith("__")]
        if scalar_steps:
            epoch = max(scalar_steps) + 1

    os.system("clear")
    print("T1 Upper9 Camera-Stable Progress Dashboard")
    print("=" * 78)
    print(f"Run:   {args.label}")
    print(f"Task:  {args.task}")
    print("Goal:  official legs + zero-base shoulders + elbow/waist assist + stable camera")
    print(f"Phase: {args.phase}    {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"GPU:   {gpu_line()}")
    print(f"Dir:   {run_dir if run_dir else 'waiting for logs/...'}")
    print(f"Epoch: {epoch}/{total}  {bar(epoch, total)}")
    print("-" * 78)
    print(
        "reward {reward} | steps {steps} | value_loss {value_loss} | kl {kl} | lr {lr}".format(
            reward=fmt_scalar(scalars, "reward"),
            steps=fmt_scalar(scalars, "steps"),
            value_loss=fmt_scalar(scalars, "value_loss"),
            kl=fmt_scalar(scalars, "kl_mean"),
            lr=fmt_scalar(scalars, "lr", 6),
        )
    )
    print(
        "grid unlocked {unlocked} | tilt_rms {tilt} | arm_sat {sat}".format(
            unlocked=fmt_scalar(scalars, "curriculum/unlocked_cells"),
            tilt=fmt_scalar(scalars, "curriculum/active_tilt_rms_mean"),
            sat=fmt_scalar(scalars, "curriculum/active_arm_saturation_frac_mean"),
        )
    )
    print("-" * 78)
    for tag in [
        "episode/camera_stability",
        "episode/shoulder_static_posture",
        "episode/shoulder_kinematic_motion_target",
        "episode/elbow_kinematic_motion_target",
        "episode/waist_kinematic_motion_target",
        "episode/upper_body_static_posture",
        "episode/upper_body_smoothness",
    ]:
        print(f"{tag:42s} {fmt_scalar(scalars, tag)}")
    print("-" * 78)
    print("Recent progress:")
    for line in recent:
        print(line)
    if "__error__" in scalars:
        print("TensorBoard read:", scalars["__error__"])


if __name__ == "__main__":
    main()
