import json
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RUN_DIR = Path("logs/2026-05-08-00-19-50")
OUT = Path("artifacts/autoresearch/shoulder4_softlimit_balanced500_metrics.json")
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
        for candidate in candidates:
            if tag.endswith(candidate):
                return tag
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
metrics, summary, used = {}, {}, {}

for name, candidates in TAGS.items():
    tag = pick_tag(scalars, candidates)
    if not tag:
        metrics[name] = []
        continue
    points = [{"step": int(e.step), "value": float(e.value)} for e in ea.Scalars(tag)]
    metrics[name] = points
    if points:
        summary[name] = summarize(points)
    used[name] = tag

payload = {
    "run": "T1Shoulder4 SoftLimitBalanced model_500",
    "checkpoint": "logs/2026-05-08-00-19-50/nn/model_500.pth",
    "sourceRun": "logs/2026-05-08-00-19-50",
    "note": "冻结 model7000 腿部，只训练 4 个肩关节；更大安全硬限位 + 舒适软限位；重新打开低速回中、左右对称、手臂速度/加速度/力矩/功率惩罚；reward 不做 only-positive clipping；logstd_init=-1.3；32768 envs，训练 500 iterations。",
    "metrics": metrics,
    "summary": summary,
    "usedTags": used,
}
OUT.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
print(OUT)
print(json.dumps({"usedTags": used, "summary": summary}, ensure_ascii=False, indent=2))
