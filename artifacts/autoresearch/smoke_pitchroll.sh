set -euo pipefail
cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official
export PATH=/opt/conda/bin:$PATH
find . -name '._*' -delete
/opt/conda/bin/python -u train_shoulder4_frozen.py --task T1Shoulder4Pitch09Roll08_from7000LegFrozen_train1000 --num_envs 1024 --max_iterations 1 --headless True
