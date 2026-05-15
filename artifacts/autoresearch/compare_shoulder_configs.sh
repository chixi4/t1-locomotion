#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official

echo "--- configs ---"
for f in \
  logs/2026-05-06-07-55-02/config.yaml \
  logs/2026-05-06-13-58-17/config.yaml \
  logs/2026-05-07-09-14-53/config.yaml \
  envs/T1Shoulder4SwayMin_from7000LegFrozen_satguard2000.yaml \
  envs/T1Shoulder4Pitch09Roll08_from7000LegFrozen_train1000.yaml
do
  echo "### $f"
  if [[ -f "$f" ]]; then
    python - <<PY
import yaml, json
path="$f"
cfg=yaml.load(open(path, encoding="utf-8"), Loader=yaml.FullLoader)
out={}
for key in ["basic","algorithm","runner","control","rewards","commands"]:
    if key in cfg:
        out[key]=cfg[key]
print(json.dumps(out, ensure_ascii=False, indent=2)[:12000])
PY
  else
    echo missing
  fi
done

echo "--- eval summaries ---"
for f in \
  artifacts/autoresearch/shoulder4_model900_eval/shoulder4_model900_summary.json \
  artifacts/autoresearch/shoulder4_model900_eval/shoulder4_satguard2000_model2000_summary.json \
  artifacts/autoresearch/shoulder4_model900_eval/shoulder4_pitch09roll08_model1000_random_eval.json \
  artifacts/autoresearch/shoulder4_model900_eval/shoulder4_pitch09roll08_model1000_fixed_eval.json
do
  echo "### $f"
  [[ -f "$f" ]] && python -m json.tool "$f" | sed -n '1,180p' || echo missing
done
