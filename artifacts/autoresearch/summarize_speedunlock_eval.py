import json
from datetime import datetime
from pathlib import Path


EVAL_DIR = Path(
    "/mnt/c/Users/Administrator/Documents/dev/official_baselines/"
    "booster_gym_official/artifacts/autoresearch/"
    "upper9_camera_stable_openleg18_legresidual_speedunlock300_fixed_eval"
)

REQUIRED_KEYS = [
    "checkpoint",
    "reset_events_per_env",
    "lin_error_mean",
    "yaw_error_mean",
    "camera_tilt_p95",
    "camera_ang_xy_rms",
    "stand_shoulder_abs_p95",
    "stand_elbow_abs_p95",
    "stand_waist_abs_p95",
    "pitch_abs_p95_moving",
    "pitch_lr_antisym_corr",
    "pitch_common_abs_p95",
    "side_left_out_p95_on_right",
    "side_right_out_p95_on_left",
    "elbow_abs_p95_moving",
    "waist_abs_p95_moving",
    "residual_abs_p95",
]


def load_row(path):
    data = json.loads(path.read_text())
    row = extract_result(data, path)
    missing = [key for key in REQUIRED_KEYS if key not in row]
    if missing:
        raise KeyError(f"{path.name} missing keys: {missing}")
    return {key: row[key] for key in REQUIRED_KEYS}


def extract_result(data, path):
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        if not data["results"]:
            raise ValueError(f"{path.name} has empty results")
        return data["results"][0]
    if isinstance(data, list) and data:
        return data[0]
    if isinstance(data, dict):
        return data
    raise TypeError(f"{path.name} has unsupported JSON structure")


def add_derived_metrics(row):
    enriched = dict(row)
    enriched["side_min"] = min(
        row["side_left_out_p95_on_right"],
        row["side_right_out_p95_on_left"],
    )
    enriched["score"] = score_row(enriched)
    enriched["passed_gate"] = passed_gate(enriched)
    return enriched


def score_row(row):
    forward_gap = max(0.0, row["lin_error_mean"] - 0.14)
    camera_gap = max(0.0, row["camera_tilt_p95"] - 0.095)
    shoulder_gap = max(0.0, row["stand_shoulder_abs_p95"] - 0.052)
    pitch_gap = max(0.0, 0.215 - row["pitch_abs_p95_moving"])
    side_gap = max(0.0, 0.20 - row["side_min"])
    common_gap = max(0.0, row["pitch_common_abs_p95"] - 0.078)
    return (
        70 * camera_gap
        + 26 * row["reset_events_per_env"]
        + 14 * forward_gap
        + 20 * shoulder_gap
        + 8 * pitch_gap
        + 12 * side_gap
        + 5 * common_gap
    )


def passed_gate(row):
    return (
        row["reset_events_per_env"] <= 0.035
        and row["lin_error_mean"] <= 0.145
        and row["camera_tilt_p95"] <= 0.105
        and row["camera_ang_xy_rms"] <= 0.47
        and row["stand_shoulder_abs_p95"] <= 0.055
        and row["pitch_abs_p95_moving"] >= 0.215
        and row["pitch_lr_antisym_corr"] >= 0.93
        and row["side_min"] >= 0.20
    )


def print_table(rows):
    header = "checkpoint reset lin yaw camTilt camAng shoulder pitch anti side residual score pass"
    print(header)
    for row in rows:
        print(
            f"{row['checkpoint']:>10} "
            f"{row['reset_events_per_env']:.4f} "
            f"{row['lin_error_mean']:.4f} "
            f"{row['yaw_error_mean']:.4f} "
            f"{row['camera_tilt_p95']:.4f} "
            f"{row['camera_ang_xy_rms']:.4f} "
            f"{row['stand_shoulder_abs_p95']:.4f} "
            f"{row['pitch_abs_p95_moving']:.4f} "
            f"{row['pitch_lr_antisym_corr']:.4f} "
            f"{row['side_min']:.4f} "
            f"{row['residual_abs_p95']:.4f} "
            f"{row['score']:.4f} "
            f"{row['passed_gate']}"
        )


def main():
    goal_paths = sorted(EVAL_DIR.glob("goal_*.json"))
    if not goal_paths:
        raise FileNotFoundError(f"no goal_*.json files in {EVAL_DIR}")

    rows = [add_derived_metrics(load_row(path)) for path in goal_paths]
    ranked = sorted(rows, key=lambda item: item["score"])
    summary = {
        "eval_dir": str(EVAL_DIR),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "rows": rows,
        "ranked": ranked,
        "best": ranked[0],
    }
    summary_path = EVAL_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print_table(ranked)
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
