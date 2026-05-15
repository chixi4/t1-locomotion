import json
import os

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = "/mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official"
RUN = "logs/2026-05-06-07-55-02"
LABEL = "T1Shoulder4SwayMin model_900"
CKPT = f"{RUN}/nn/model_900.pth"


def scalar_points(accumulator, tags, tag):
    if tag not in tags:
        return []
    return [{"step": int(event.step), "value": float(event.value)} for event in accumulator.Scalars(tag)]


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
        "note": "Shoulder-only frozen-leg run stopped at iteration 963; model_900 is the last saved pre-divergence checkpoint.",
        "metrics": metrics,
        "summary": summary,
    }
    path = "artifacts/autoresearch/shoulder4_model900_metrics.json"
    with open(path, "w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, separators=(",", ":"))
    return path, summary


def main():
    os.chdir(ROOT)
    accumulator = EventAccumulator(os.path.join(RUN, "summaries"), size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = accumulator.Tags().get("scalars", [])
    metrics_path, metrics_summary = extract_metrics(accumulator, tags)
    print(json.dumps({"metrics": metrics_path, "metricsSummary": metrics_summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
