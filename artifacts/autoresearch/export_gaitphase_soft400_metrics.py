import json
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RUN_DIR = Path("logs/2026-05-08-09-33-58")
OUT = Path("artifacts/autoresearch/shoulder4_gaitphase_soft400_metrics.json")
MAX_STEP = 400
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
    lowered = {tag.lower(): tag for tag in scalars}
    for tag in candidates:
        if tag.lower() in lowered:
            return lowered[tag.lower()]
    for tag in scalars:
        for c in candidates:
            if tag.endswith(c):
                return tag
    return None


def summarize(points):
    vals = [p["value"] for p in points]
    tail = vals[-100:] if len(vals) >= 100 else vals
    return {"last": vals[-1], "min": min(vals), "max": max(vals), "min_last100": min(tail), "max_last100": max(tail)}


ea = EventAccumulator(str(RUN_DIR / "summaries"))
ea.Reload()
scalars = ea.Tags().get("scalars", [])
metrics, summary, used = {}, {}, {}
for name, candidates in TAGS.items():
    tag = pick_tag(scalars, candidates)
    if not tag:
        metrics[name] = []
        continue
    points = [{"step": int(e.step), "value": float(e.value)} for e in ea.Scalars(tag) if int(e.step) <= MAX_STEP]
    metrics[name] = points
    if points:
        summary[name] = summarize(points)
    used[name] = tag

payload = {
    "run": "T1Shoulder4 GaitPhaseSoft model_400",
    "checkpoint": "logs/2026-05-08-09-33-58/nn/model_400.pth",
    "sourceRun": "logs/2026-05-08-09-33-58",
    "note": "冻结 model7000 腿部，只训练 4 个肩关节；新增轻量 gait phase shoulder pitch 先验；pair symmetry 降为辅助项；降低手臂幅度、软限位和探索；训练中断于 466，此网页按 model_400 截断训练曲线。",
    "metrics": metrics,
    "summary": summary,
    "usedTags": used,
}
OUT.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
print(OUT)
print(json.dumps({"usedTags": used, "summary": summary}, ensure_ascii=False, indent=2))
