#!/usr/bin/env python3
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


run_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "logs/2026-05-05-11-09-07")
summary_dir = run_dir / "summaries"
events = sorted(summary_dir.glob("events.out.tfevents.*"), key=lambda p: p.stat().st_mtime)
print(f"run={run_dir}")
print(f"event_files={len(events)}")
if not events:
    raise SystemExit(0)

acc = EventAccumulator(str(summary_dir), size_guidance={"scalars": 0})
acc.Reload()
tags = set(acc.Tags().get("scalars", []))
wanted = [
    "reward",
    "value_loss",
    "steps",
    "lr",
    "kl_mean",
    "entropy",
    "curriculum/max_lin_vel_level",
    "curriculum/max_ang_vel_level",
    "curriculum/mean_lin_vel_level",
    "curriculum/mean_ang_vel_level",
]
for tag in wanted:
    if tag not in tags:
        print(f"{tag}: MISSING")
        continue
    series = acc.Scalars(tag)
    if not series:
        print(f"{tag}: EMPTY")
        continue
    last = series[-1]
    window = [x.value for x in series[-50:]]
    print(
        f"{tag}: n={len(series)} step={last.step} last={last.value:.8g} "
        f"min50={min(window):.8g} max50={max(window):.8g}"
    )
