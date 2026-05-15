import glob
import json
import math
import os
import re

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = "/mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official"
RUN = "logs/2026-05-04-12-54-43"


def scalar_map(accumulator, tags, tag):
    if tag not in tags:
        return {}
    return {int(event.step): float(event.value) for event in accumulator.Scalars(tag)}


def main():
    os.chdir(ROOT)
    ckpt_paths = sorted(
        glob.glob(os.path.join(RUN, "nn", "model_*.pth")),
        key=lambda path: int(re.search(r"model_(\d+)\.pth", path).group(1)),
    )

    checkpoints = []
    for path in ckpt_paths:
        step = int(re.search(r"model_(\d+)\.pth", path).group(1))
        model_dict = torch.load(path, map_location="cpu", weights_only=True)
        curriculum = model_dict.get("curriculum")
        if curriculum is None:
            continue

        curriculum = curriculum.detach().cpu()
        shape = list(curriculum.shape)
        unlocked = (curriculum > 0.5).nonzero(as_tuple=False)
        lin_levels = (shape[0] - 1) // 2
        yaw_levels = (shape[2] - 1) // 2

        encoded = []
        rings = {str(i): 0 for i in range(lin_levels + 1)}
        yaw_abs = {str(i): 0 for i in range(yaw_levels + 1)}

        for x, y, z in unlocked.tolist():
            lx = x - lin_levels
            ly = y - lin_levels
            wz = z - yaw_levels
            encoded.append((x * shape[1] + y) * shape[2] + z)

            radius = int(round(math.sqrt(lx * lx + ly * ly)))
            radius = max(0, min(lin_levels, radius))
            rings[str(radius)] += 1
            yaw_abs[str(abs(wz))] += 1

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
            }
        )

    previous = set()
    for checkpoint in checkpoints:
        current = set(checkpoint["unlocked"])
        checkpoint["newUnlocked"] = sorted(current - previous)
        checkpoint["newCount"] = len(checkpoint["newUnlocked"])
        previous = current

    accumulator = EventAccumulator(os.path.join(RUN, "summaries"), size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = accumulator.Tags().get("scalars", [])
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

    for checkpoint in checkpoints:
        step = checkpoint["step"]
        checkpoint["metrics"] = {tag: metrics[tag].get(step) for tag in wanted_tags}

    anomaly_window = []
    for step in range(1688, 1764):
        row = {"step": step}
        keep = False
        for tag in wanted_tags:
            value = metrics[tag].get(step)
            if value is not None:
                row[tag] = value
                keep = True
        if keep:
            anomaly_window.append(row)

    max_lin_changes = []
    last = None
    for event in accumulator.Scalars("curriculum/max_lin_vel_level"):
        value = round(float(event.value), 6)
        if last is None or abs(value - last) > 1e-6:
            max_lin_changes.append({"step": int(event.step), "value": float(event.value)})
            last = value

    output = {
        "run": "T1CircleGridCurriculum model_2000",
        "sourceRun": RUN,
        "note": "Curriculum masks are reconstructed from saved checkpoints every 100 PPO iterations; TensorBoard scalar events provide exact anomaly steps.",
        "grid": {
            "linLevels": 10,
            "yawLevels": 10,
            "shape": [21, 21, 21],
            "linearSpeedRange": [0, 1.0],
            "yawRangeApprox": [-1.0, 1.0],
        },
        "checkpoints": checkpoints,
        "anomalyWindow": anomaly_window,
        "maxLinLevelChanges": max_lin_changes,
        "spike": {
            "step": 1734,
            "value_loss": metrics["value_loss"].get(1734),
            "kl_mean": metrics["kl_mean"].get(1734),
            "lr": metrics["lr"].get(1734),
            "betweenCheckpoints": [1700, 1800],
        },
    }

    os.makedirs("artifacts/autoresearch", exist_ok=True)
    output_path = "artifacts/autoresearch/grid_curriculum_unlock_timeline.json"
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, separators=(",", ":"))

    summary = {
        "output": output_path,
        "checkpoints": len(checkpoints),
        "first": checkpoints[0]["unlockedCount"] if checkpoints else 0,
        "last": checkpoints[-1]["unlockedCount"] if checkpoints else 0,
        "spike": output["spike"],
        "counts": [(c["step"], c["unlockedCount"], c["newCount"]) for c in checkpoints],
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
