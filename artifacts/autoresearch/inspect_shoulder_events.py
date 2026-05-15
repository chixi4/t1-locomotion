import glob
import sys

from tensorboard.backend.event_processing import event_accumulator


run = sys.argv[1] if len(sys.argv) > 1 else "logs/2026-05-06-07-55-02"
print("run", run)
print("event_files", glob.glob(run + "/summaries/events.out.tfevents*"))
ea = event_accumulator.EventAccumulator(run + "/summaries", size_guidance={"scalars": 0})
ea.Reload()
keys = ea.Tags().get("scalars", [])
want = [
    "value_loss",
    "actor_loss",
    "bound_loss",
    "entropy",
    "kl_mean",
    "lr",
    "reward",
    "steps",
    "curriculum/unlocked_cells",
    "curriculum/max_lin_vel_level",
    "curriculum/max_ang_vel_level",
]
for key in want:
    if key not in keys:
        print("\n" + key, "MISSING")
        continue
    vals = ea.Scalars(key)
    print("\n" + key, "count", len(vals))
    for event in vals[-12:]:
        print(event.step, float(event.value))
