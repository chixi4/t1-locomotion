#!/usr/bin/env bash
set -euo pipefail

cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official

export PATH=/opt/conda/bin:/usr/lib/wsl/lib:${PATH}
export LD_LIBRARY_PATH=/opt/conda/lib:/usr/lib/wsl/lib:/opt/isaacgym/python/isaacgym/_bindings/linux-x86_64:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/opt/isaacgym/python:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1

TASK="T1Upper9CameraStableOfficialOpenLeg18LegResidualSpeedUnlock_train300"
LABEL="upper9_camera_stable_openleg18_legresidual_speedunlock300"
RUN_DIR="${1:-}"

if [[ -z "${RUN_DIR}" ]]; then
  RUN_DIR="$(python3 - <<'PY'
from pathlib import Path

import yaml

task = "T1Upper9CameraStableOfficialOpenLeg18LegResidualSpeedUnlock_train300"
candidates = []
for cfg_path in Path("logs").glob("*/config.yaml"):
    try:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        continue
    if cfg.get("basic", {}).get("task") == task:
        candidates.append(cfg_path.parent)
if candidates:
    print(sorted(candidates, key=lambda path: path.stat().st_mtime)[-1])
PY
)"
fi

if [[ -z "${RUN_DIR}" || ! -d "${RUN_DIR}" ]]; then
  echo "No run dir found for ${TASK}" >&2
  exit 1
fi

OUT_DIR="artifacts/autoresearch/${LABEL}_fixed_eval"
mkdir -p "${OUT_DIR}"
echo "Evaluating ${RUN_DIR}"

while IFS= read -r checkpoint; do
  step="$(basename "${checkpoint}" .pth | sed 's/model_//')"
  out="${OUT_DIR}/goal_${step}.json"
  echo "==== fixed-command eval model_${step} ===="
  python3 -u artifacts/autoresearch/probe_residual_checkpoint.py \
    --task "${TASK}" \
    --checkpoints "${checkpoint}" \
    --num-envs 96 \
    --seconds 5 \
    --warmup-s 1.2 \
    --out "${out}"
done < <(find "${RUN_DIR}/nn" -maxdepth 1 -name 'model_*.pth' | sort -V)

python3 - "${OUT_DIR}" <<'PY'
import json
import sys
from pathlib import Path


def score(row):
    side = min(row["side_left_out_p95_on_right"], row["side_right_out_p95_on_left"])
    forward_gap = max(0.0, row["lin_error_mean"] - 0.14)
    camera_gap = max(0.0, row["camera_tilt_p95"] - 0.095)
    shoulder_gap = max(0.0, row["stand_shoulder_abs_p95"] - 0.052)
    return (
        70.0 * camera_gap
        + 26.0 * row["reset_events_per_env"]
        + 14.0 * forward_gap
        + 20.0 * shoulder_gap
        + 8.0 * max(0.0, 0.215 - row["pitch_abs_p95_moving"])
        + 12.0 * max(0.0, 0.20 - side)
        + 5.0 * max(0.0, row["pitch_common_abs_p95"] - 0.078)
    )


def passed(row):
    side = min(row["side_left_out_p95_on_right"], row["side_right_out_p95_on_left"])
    return (
        row["reset_events_per_env"] <= 0.035
        and row["lin_error_mean"] <= 0.145
        and row["camera_tilt_p95"] <= 0.105
        and row["camera_ang_xy_rms"] <= 0.47
        and row["stand_shoulder_abs_p95"] <= 0.055
        and row["pitch_abs_p95_moving"] >= 0.215
        and row["pitch_lr_antisym_corr"] >= 0.93
        and side >= 0.20
    )


out_dir = Path(sys.argv[1])
rows = []
for path in sorted(out_dir.glob("goal_*.json")):
    payload = json.loads(path.read_text(encoding="utf-8"))
    for row in payload.get("results", []):
        item = {"score": score(row), "passed": passed(row), **row}
        rows.append(item)
rows.sort(key=lambda item: item["score"])

print("checkpoint\tscore\tpass\treset\tlin_err\tyaw_err\tcam_p95\tcam_ang\tstand_sh\tpitch\tcorr\tside_min\tresid")
for row in rows:
    side = min(row["side_left_out_p95_on_right"], row["side_right_out_p95_on_left"])
    print(
        f"{Path(row['checkpoint']).name}\t{row['score']:.4f}\t{int(row['passed'])}\t"
        f"{row['reset_events_per_env']:.4f}\t{row['lin_error_mean']:.4f}\t"
        f"{row['yaw_error_mean']:.4f}\t{row['camera_tilt_p95']:.4f}\t"
        f"{row['camera_ang_xy_rms']:.4f}\t{row['stand_shoulder_abs_p95']:.4f}\t"
        f"{row['pitch_abs_p95_moving']:.4f}\t{row['pitch_lr_antisym_corr']:.4f}\t"
        f"{side:.4f}\t{row['residual_abs_p95']:.4f}"
    )

summary = {"best": rows[0] if rows else None, "rows": rows}
(out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
if rows:
    best = rows[0]
    print("BEST", Path(best["checkpoint"]).name, "pass", int(best["passed"]), "score", f"{best['score']:.4f}")
PY


