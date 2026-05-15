from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator


def main():
    root = Path("logs")
    runs = sorted(root.glob("2026-05-06-*/summaries"), key=lambda p: p.stat().st_mtime)
    if not runs:
        print("no runs")
        return
    run = runs[-1]
    print("run", run.parent)
    files = list(run.glob("events.out.tfevents.*"))
    print("event_files", len(files))
    if not files:
        return
    acc = event_accumulator.EventAccumulator(str(run), size_guidance={"scalars": 0})
    acc.Reload()
    tags = acc.Tags().get("scalars", [])
    want = [
        "value_loss",
        "actor_loss",
        "bound_loss",
        "reward",
        "steps",
        "lr",
        "kl_mean",
        "entropy",
        "curriculum/unlocked_cells",
        "curriculum/active_tilt_rms_mean",
        "curriculum/active_arm_saturation_frac_mean",
        "policy/mean_abs_max",
        "ppo/successful_updates",
        "ppo/nonfinite_updates",
    ]
    for tag in want:
        if tag not in tags:
            print(tag, "MISSING")
            continue
        vals = acc.Scalars(tag)
        if not vals:
            print(tag, "EMPTY")
            continue
        window = vals[-20:]
        avg = sum(v.value for v in window) / len(window)
        print(tag, "n=", len(vals), "last_step=", vals[-1].step, "last=", vals[-1].value, "avg20=", avg)


if __name__ == "__main__":
    main()
