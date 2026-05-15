set -euo pipefail
cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official
stamp=$(date +%Y%m%d_%H%M%S)
backup=artifacts/autoresearch/backups/pitchroll09roll08_$stamp
mkdir -p "$backup/utils" "$backup/envs" "$backup/artifacts/autoresearch"
cp utils/model.py "$backup/utils/model.py"
cp utils/shoulder_runner.py "$backup/utils/shoulder_runner.py"
cp envs/__init__.py "$backup/envs/__init__.py"
cp artifacts/autoresearch/eval_shoulder4_frozen.py "$backup/artifacts/autoresearch/eval_shoulder4_frozen.py"
tar -C . -xzf artifacts/autoresearch/pitchroll_patch_upload.tgz
/opt/conda/bin/python -m py_compile utils/model.py utils/shoulder_runner.py artifacts/autoresearch/eval_shoulder4_frozen.py
printf 'backup=%s\n' "$backup"
grep -n 'T1Shoulder4Pitch09Roll08' envs/__init__.py
sed -n '32,55p' envs/T1Shoulder4Pitch09Roll08_from7000LegFrozen_train1000.yaml
