#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


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
        "count": len(vals),
    }


def apply_empty_policy(name, raw_points, policy):
    if policy == "raw" or name not in {"steps", "reward"}:
        return raw_points
    if policy == "drop_zero":
        return [p for p in raw_points if p["value"] != 0.0]
    if policy == "forward_fill":
        out = []
        last = None
        for point in raw_points:
            p = dict(point)
            if p["value"] == 0.0 and last is not None:
                p["value"] = last
                p["filled"] = True
            elif p["value"] != 0.0:
                last = p["value"]
            out.append(p)
        return out
    raise ValueError(f"unknown empty policy: {policy}")


def load_metrics(run_dir, max_step, empty_policy):
    ea = EventAccumulator(str(run_dir / "summaries"))
    ea.Reload()
    scalars = ea.Tags().get("scalars", [])
    metrics, summary, used = {}, {}, {}
    diagnostics = {}
    for name, candidates in TAGS.items():
        tag = pick_tag(scalars, candidates)
        if not tag:
            metrics[name] = []
            diagnostics[name] = {"tag": None, "raw_count": 0, "zero_count": 0, "kept_count": 0}
            continue
        raw = [
            {"step": int(e.step), "value": float(e.value)}
            for e in ea.Scalars(tag)
            if max_step is None or int(e.step) <= max_step
        ]
        points = apply_empty_policy(name, raw, empty_policy)
        metrics[name] = points
        if points:
            summary[name] = summarize(points)
        used[name] = tag
        diagnostics[name] = {
            "tag": tag,
            "raw_count": len(raw),
            "zero_count": sum(1 for p in raw if p["value"] == 0.0),
            "kept_count": len(points),
        }
    return metrics, summary, used, diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--note", default="")
    parser.add_argument("--max-step", type=int, default=None)
    parser.add_argument(
        "--empty-policy",
        choices=["raw", "drop_zero", "forward_fill"],
        default="drop_zero",
        help="How to handle legacy empty episode windows logged as exact 0 for steps/reward.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    metrics, summary, used, diagnostics = load_metrics(run_dir, args.max_step, args.empty_policy)
    payload = {
        "run": args.run_name,
        "checkpoint": args.checkpoint,
        "sourceRun": str(run_dir),
        "note": args.note,
        "emptyPolicy": args.empty_policy,
        "metrics": metrics,
        "summary": summary,
        "usedTags": used,
        "diagnostics": diagnostics,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(json.dumps({"out": str(out), "emptyPolicy": args.empty_policy, "diagnostics": diagnostics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
