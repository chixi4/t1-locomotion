import json
import os

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = "/mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official"
RUN = "logs/2026-05-07-09-14-53"
LABEL = "T1Shoulder4 Pitch0.9/Roll0.8 model_1000"
CKPT = f"{RUN}/nn/model_1000.pth"


def scalar_points(accumulator, tags, tag):
    if tag not in tags:
        return []
    return [{"step": int(event.step), "value": float(event.value)} for event in accumulator.Scalars(tag)]


def main():
    os.chdir(ROOT)
    accumulator = EventAccumulator(os.path.join(RUN, "summaries"), size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = accumulator.Tags().get("scalars", [])
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
        "note": "Pitch actor_mean_scale=0.9, Roll actor_mean_scale=0.8; frozen model7000 legs, 32768 envs, trained 1000 iterations.",
        "metrics": metrics,
        "summary": summary,
    }
    path = "artifacts/autoresearch/shoulder4_pitch09roll08_model1000_metrics.json"
    with open(path, "w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, separators=(",", ":"))
    print(json.dumps({"metrics": path, "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
