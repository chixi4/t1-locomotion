#!/usr/bin/env bash
set -euo pipefail

cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official

export PATH=/opt/conda/bin:/usr/lib/wsl/lib:${PATH}
export LD_LIBRARY_PATH=/opt/conda/lib:/usr/lib/wsl/lib:/opt/isaacgym/python/isaacgym/_bindings/linux-x86_64:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/opt/isaacgym/python:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1

TASK="T1Upper9CameraStableOfficialOpenLeg18LegResidualSpeedPolish_train200"
LABEL="upper9_camera_stable_openleg18_legresidual_speedpolish200"
RUN_DIR="${1:-}"

if [[ -z "${RUN_DIR}" ]]; then
  RUN_DIR="$(python3 -c "from pathlib import Path; import yaml; task='${TASK}'; c=[]; \
for p in Path('logs').glob('*/config.yaml'): \
    cfg=yaml.safe_load(open(p, encoding='utf-8')); \
    (cfg.get('basic', {}).get('task') == task) and c.append(p.parent); \
print(sorted(c, key=lambda x: x.stat().st_mtime)[-1] if c else '')")"
fi

if [[ -z "${RUN_DIR}" || ! -d "${RUN_DIR}" ]]; then
  echo "No run dir found for ${TASK}" >&2
  exit 1
fi

OUT_DIR="artifacts/autoresearch/${LABEL}_fixed_eval"
mkdir -p "${OUT_DIR}"
echo "Evaluating ${RUN_DIR}"

for step in 550 600 650 700; do
  checkpoint="${RUN_DIR}/nn/model_${step}.pth"
  if [[ ! -f "${checkpoint}" ]]; then
    continue
  fi
  out="${OUT_DIR}/goal_${step}.json"
  echo "==== fixed-command eval model_${step} ===="
  python3 -u artifacts/autoresearch/probe_residual_checkpoint.py \
    --task "${TASK}" \
    --checkpoints "${checkpoint}" \
    --num-envs 64 \
    --seconds 4 \
    --warmup-s 1.2 \
    --out "${out}"
done

python3 - "${OUT_DIR}" <<'PY'
import json
import math
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
rows = []
for path in sorted(out_dir.glob("goal_*.json")):
    payload = json.loads(path.read_text(encoding="utf-8"))
    for row in payload.get("results", []):
        ckpt = Path(row["checkpoint"]).name
        side = min(row["side_left_out_p95_on_right"], row["side_right_out_p95_on_left"])
        score = (
            130.0 * row["camera_tilt_p95"]
            + 35.0 * row["reset_events_per_env"]
            + 10.0 * row["lin_error_mean"]
            + 24.0 * max(0.0, row["stand_shoulder_abs_p95"] - 0.045)
            + 8.0 * max(0.0, 0.20 - row["pitch_abs_p95_moving"])
            + 10.0 * max(0.0, 0.19 - side)
            + 5.0 * max(0.0, row["pitch_common_abs_p95"] - 0.075)
        )
        passed = (
            row["reset_events_per_env"] <= 0.05
            and row["camera_tilt_p95"] <= 0.105
            and row["stand_shoulder_abs_p95"] <= 0.052
            and row["pitch_abs_p95_moving"] >= 0.20
            and row["pitch_lr_antisym_corr"] >= 0.92
            and side >= 0.19
            and row["lin_error_mean"] <= 0.20
        )
        rows.append((score, passed, ckpt, side, row))

rows.sort(key=lambda item: item[0])
summary = []
print("checkpoint\tscore\tpass\treset\tlin_err\tcam_p95\tcam_ang\tstand_sh\tpitch\tcorr\tside_min\tresid")
for score, passed, ckpt, side, row in rows:
    print(
        f"{ckpt}\t{score:.4f}\t{int(passed)}\t"
        f"{row['reset_events_per_env']:.4f}\t{row['lin_error_mean']:.4f}\t"
        f"{row['camera_tilt_p95']:.4f}\t{row['camera_ang_xy_rms']:.4f}\t"
        f"{row['stand_shoulder_abs_p95']:.4f}\t{row['pitch_abs_p95_moving']:.4f}\t"
        f"{row['pitch_lr_antisym_corr']:.4f}\t{side:.4f}\t{row['residual_abs_p95']:.4f}"
    )
    summary.append({"checkpoint": ckpt, "score": score, "passed": passed, "side_min": side, **row})

best = summary[0] if summary else None
(out_dir / "summary.json").write_text(json.dumps({"best": best, "rows": summary}, indent=2), encoding="utf-8")
if best:
    print("BEST", best["checkpoint"], "pass", int(best["passed"]), "score", f"{best['score']:.4f}")
PY
