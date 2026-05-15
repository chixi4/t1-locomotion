#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official
cat > artifacts/autoresearch/export_freearm_logstd17_model200_metrics.py <<'PY'
import json
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RUN_DIR = Path("logs/2026-05-07-18-57-33")
OUT = Path("artifacts/autoresearch/shoulder4_freearm_logstd17_model200_metrics.json")
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
    points = [{"step": int(e.step), "value": float(e.value)} for e in ea.Scalars(tag)]
    metrics[name] = points
    if points:
        summary[name] = summarize(points)
    used[name] = tag
payload = {
    "run": "T1Shoulder4 FreeArmNoPenalty Logstd17 model_200",
    "checkpoint": "logs/2026-05-07-18-57-33/nn/model_200.pth",
    "sourceRun": "logs/2026-05-07-18-57-33",
    "note": "诊断消融：冻结 model7000 腿部，只训练 4 个肩关节；手臂动作惩罚全部清零；anti_sway_vs_fixed_arm 保留；logstd_init=-1.7，logstd_max=-1.0；32768 envs，训练 200 iterations。",
    "metrics": metrics,
    "summary": summary,
    "usedTags": used,
}
OUT.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
print(OUT)
print(json.dumps({"usedTags": used, "summary": summary}, ensure_ascii=False, indent=2))
PY
export PATH="/opt/conda/bin:/usr/lib/wsl/lib:${PATH}"
/opt/conda/bin/python artifacts/autoresearch/export_freearm_logstd17_model200_metrics.py
