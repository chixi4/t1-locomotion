import glob
import json
import math
import os
import re

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = "/mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official"
RUN = "logs/2026-05-04-21-58-46"
LABEL = "T1CircleGridFace6Tight model_3000"
CKPT = f"{RUN}/nn/model_3000.pth"


def scalar_points(accumulator, tags, tag):
    if tag not in tags:
        return []
    return [{"step": int(event.step), "value": float(event.value)} for event in accumulator.Scalars(tag)]


def scalar_map(accumulator, tags, tag):
    return {point["step"]: point["value"] for point in scalar_points(accumulator, tags, tag)}


def metric_at(metrics, tag, step):
    values = metrics.get(tag, {})
    if step in values:
        return values[step]
    if step - 1 in values:
        return values[step - 1]
    candidates = [key for key in values if key <= step]
    return values[max(candidates)] if candidates else None


def checkpoint_step(path):
    return int(re.search(r"model_(\d+)\.pth", path).group(1))


def extract_metrics(accumulator, tags):
    wanted = ["value_loss", "steps", "reward", "lr", "kl_mean", "entropy"]
    metrics = {tag: scalar_points(accumulator, tags, tag) for tag in wanted}
    summary = {}
    for tag, points in metrics.items():
        if not points:
            continue
        values = [point["value"] for point in points]
        summary[tag] = {
            "last": values[-1],
            "min": min(values),
            "max": max(values),
            "min_last100": min(values[-100:]),
            "max_last100": max(values[-100:]),
        }
    output = {
        "run": LABEL,
        "checkpoint": CKPT,
        "sourceRun": RUN,
        "metrics": metrics,
        "summary": summary,
    }
    path = "artifacts/autoresearch/t1gridface6_model_3000_metrics.json"
    with open(path, "w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, separators=(",", ":"))
    return path, summary


def extract_curriculum(accumulator, tags):
    wanted_tags = [
        "value_loss",
        "kl_mean",
        "lr",
        "entropy",
        "reward",
        "steps",
        "curriculum/mean_lin_vel_level",
        "curriculum/mean_ang_vel_level",
        "curriculum/max_lin_vel_level",
        "curriculum/max_ang_vel_level",
    ]
    metrics = {tag: scalar_map(accumulator, tags, tag) for tag in wanted_tags}

    checkpoints = []
    for path in sorted(glob.glob(os.path.join(RUN, "nn", "model_*.pth")), key=checkpoint_step):
        step = checkpoint_step(path)
        model_dict = torch.load(path, map_location="cpu", weights_only=True)
        curriculum = model_dict.get("curriculum")
        if curriculum is None or curriculum.ndim != 3:
            continue
        curriculum = curriculum.detach().cpu()
        shape = list(curriculum.shape)
        lin_levels = (shape[0] - 1) // 2
        yaw_levels = (shape[2] - 1) // 2
        unlocked = curriculum > 0.5

        encoded = []
        rings = {str(i): 0 for i in range(lin_levels + 1)}
        yaw_abs = {str(i): 0 for i in range(yaw_levels + 1)}
        legal_count = 0
        for x in range(shape[0]):
            for y in range(shape[1]):
                lx = x - lin_levels
                ly = y - lin_levels
                if lx * lx + ly * ly > lin_levels * lin_levels:
                    continue
                for z in range(shape[2]):
                    legal_count += 1
                    if not bool(unlocked[x, y, z]):
                        continue
                    encoded.append((x * shape[1] + y) * shape[2] + z)
                    radius = int(round(math.sqrt(lx * lx + ly * ly)))
                    radius = max(0, min(lin_levels, radius))
                    rings[str(radius)] += 1
                    yaw_abs[str(abs(z - yaw_levels))] += 1

        encoded.sort()
        checkpoints.append(
            {
                "step": step,
                "shape": shape,
                "unlocked": encoded,
                "unlockedCount": len(encoded),
                "lin_levels": lin_levels,
                "yaw_levels": yaw_levels,
                "rings": rings,
                "yawAbs": yaw_abs,
                "metrics": {tag: metric_at(metrics, tag, step) for tag in wanted_tags},
            }
        )

    previous = set()
    max_new = {"step": None, "newCount": 0}
    for checkpoint in checkpoints:
        current = set(checkpoint["unlocked"])
        checkpoint["newUnlocked"] = sorted(current - previous)
        checkpoint["newCount"] = len(checkpoint["newUnlocked"])
        if checkpoint["newCount"] > max_new["newCount"]:
            max_new = {"step": checkpoint["step"], "newCount": checkpoint["newCount"]}
        previous = current

    total_legal = legal_count if checkpoints else 0
    output = {
        "run": LABEL,
        "sourceRun": RUN,
        "note": "Curriculum masks are reconstructed from saved checkpoints every 100 PPO iterations.",
        "grid": {
            "linLevels": 10,
            "yawLevels": 10,
            "shape": [21, 21, 21],
            "linearSpeedRange": [0, 1.0],
            "yawRangeApprox": [-1.0, 1.0],
            "unlockNeighbors": "face6",
            "totalLegal": total_legal,
        },
        "checkpoints": checkpoints,
        "summary": {
            "checkpointCount": len(checkpoints),
            "firstUnlocked": checkpoints[0]["unlockedCount"] if checkpoints else 0,
            "lastUnlocked": checkpoints[-1]["unlockedCount"] if checkpoints else 0,
            "coverage": checkpoints[-1]["unlockedCount"] / total_legal if checkpoints and total_legal else 0,
            "maxNew": max_new,
        },
    }
    path = "artifacts/autoresearch/t1gridface6_model3000_curriculum_unlock_timeline.json"
    with open(path, "w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, separators=(",", ":"))
    return path, output["summary"]


def main():
    os.chdir(ROOT)
    accumulator = EventAccumulator(os.path.join(RUN, "summaries"), size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = accumulator.Tags().get("scalars", [])
    metrics_path, metrics_summary = extract_metrics(accumulator, tags)
    curriculum_path, curriculum_summary = extract_curriculum(accumulator, tags)
    print(json.dumps({
        "metrics": metrics_path,
        "curriculum": curriculum_path,
        "metricsSummary": metrics_summary,
        "curriculumSummary": curriculum_summary,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
