import json
import re
from pathlib import Path


EVAL_DIR = Path(
    "/mnt/c/Users/Administrator/Documents/dev/official_baselines/"
    "booster_gym_official/artifacts/autoresearch/"
    "upper9_camera_stable_openleg18_legresidual_speedunlock300_fixed_eval"
)

CASE_NAMES = {"stand", "forward_18", "left_08", "right_08"}


def checkpoint_step(path):
    match = re.search(r"goal_(\d+)\.json$", path.name)
    if not match:
        raise ValueError(f"unexpected file name: {path.name}")
    return int(match.group(1))


def load_result(path):
    data = json.loads(path.read_text())
    results = data.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError(f"{path.name} has no result")
    return results[0]


def case_value(case, key):
    value = case.get(key)
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, int):
        return str(value)
    return "NA"


def print_case(step, case):
    return (
        f"{step:>4} {case['case']:<12} "
        f"reset={case_value(case, 'reset_events_per_env')} "
        f"lin={case_value(case, 'lin_error_mean')} "
        f"yaw={case_value(case, 'yaw_error_mean')} "
        f"camTilt={case_value(case, 'camera_tilt_p95')} "
        f"camAng={case_value(case, 'camera_ang_xy_rms')} "
        f"shoulder={case_value(case, 'stand_shoulder_abs_p95')} "
        f"pitch={case_value(case, 'pitch_abs_p95_moving')} "
        f"anti={case_value(case, 'pitch_lr_antisym_corr')} "
        f"sideL={case_value(case, 'side_left_out_p95_on_right')} "
        f"sideR={case_value(case, 'side_right_out_p95_on_left')} "
        f"resid={case_value(case, 'residual_abs_p95')}"
    )


def main():
    lines = []
    for path in sorted(EVAL_DIR.glob("goal_*.json"), key=checkpoint_step):
        step = checkpoint_step(path)
        result = load_result(path)
        cases = result.get("cases", [])
        selected = [case for case in cases if case.get("case") in CASE_NAMES]
        lines.append(f"checkpoint {step}")
        for case in selected:
            lines.append(print_case(step, case))
    output_path = EVAL_DIR / "case_summary.txt"
    output_path.write_text("\n".join(lines) + "\n")
    print(output_path)


if __name__ == "__main__":
    main()
