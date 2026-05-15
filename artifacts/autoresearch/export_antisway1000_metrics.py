import json
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RUN_DIR = Path("logs/2026-05-07-13-32-44")
OUT = Path("artifacts/autoresearch/shoulder4_antisway_baseline_model1000_metrics.json")
TAGS = {
    "value_loss": ["value_loss"],
    "steps": ["Episode/steps", "steps"],
    "reward": ["Episode/reward", "reward"],
    "lr": ["lr"],
    "kl_mean": ["kl_mean"],
    "entropy": ["entropy"],
}


def pick_tag(scalars, candidates):
    for tag in candidates:
        if tag in scalars:
            return tag
    suffix = tuple(candidates)
    for tag in scalars:
        if tag.endswith(suffix):
            return tag
    lowered = {tag.lower(): tag for tag in scalars}
    for tag in candidates:
        if tag.lower() in lowered:
            return lowered[tag.lower()]
    return None


def summarize(points):
    vals = [p["value"] for p in points]
    tail = vals[-100:] if len(vals) >= 100 else vals
    return {
        "last": vals[-1],
        "min": min(vals),
        "max": max(vals),
        "min_last100": min(tail),
        "max_last100": max(tail),
    }


ea = EventAccumulator(str(RUN_DIR / "summaries"))
ea.Reload()
scalars = ea.Tags().get("scalars", [])
metrics = {}
summary = {}
used_tags = {}
for name, candidates in TAGS.items():
    tag = pick_tag(scalars, candidates)
    if not tag:
        metrics[name] = []
        continue
    points = [{"step": int(e.step), "value": float(e.value)} for e in ea.Scalars(tag)]
    metrics[name] = points
    if points:
        summary[name] = summarize(points)
    used_tags[name] = tag

payload = {
    "run": "T1Shoulder4 AntiSwayBaseline model_1000",
    "checkpoint": "logs/2026-05-07-13-32-44/nn/model_1000.pth",
    "sourceRun": "logs/2026-05-07-13-32-44",
    "note": "固定 model7000 腿部；Pitch actor_mean_scale=0.9、Roll actor_mean_scale=0.7；加入 fixed-arm sway baseline、arm_action_rate=-0.8、arm_action_high_freq=-0.8；32768 envs，训练 1000 iterations。",
    "metrics": metrics,
    "summary": summary,
    "usedTags": used_tags,
}
OUT.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
print(OUT)
print(json.dumps({"usedTags": used_tags, "summary": summary}, ensure_ascii=False, indent=2))
