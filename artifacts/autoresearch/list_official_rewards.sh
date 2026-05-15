#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official

echo "--- reward configs ---"
find envs -maxdepth 1 -type f \( -name "*.yaml" -o -name "*.yml" \) -print | sort | while read -r f; do
  echo "### $f"
  python - "$f" <<'PY'
import sys, yaml, json
path=sys.argv[1]
cfg=yaml.load(open(path, encoding="utf-8"), Loader=yaml.FullLoader)
rewards=cfg.get("rewards")
if not rewards:
    print("no rewards")
    raise SystemExit
out={
    "basic_run_name": cfg.get("basic",{}).get("run_name"),
    "asset_file": cfg.get("asset",{}).get("file"),
    "num_actions": cfg.get("env",{}).get("num_actions"),
    "episode_length_s": rewards.get("episode_length_s"),
    "only_positive_rewards": rewards.get("only_positive_rewards"),
    "tracking_sigma": rewards.get("tracking_sigma"),
    "tracking_sigma_curriculum": rewards.get("tracking_sigma_curriculum"),
    "base_height_target": rewards.get("base_height_target"),
    "terminate_height": rewards.get("terminate_height"),
    "terminate_vel": rewards.get("terminate_vel"),
    "scales": rewards.get("scales", {}),
}
print(json.dumps(out, ensure_ascii=False, indent=2))
PY
done

echo "--- reward methods in envs/t1.py ---"
grep -n "def _reward_" envs/t1.py | sed -n '1,240p'

echo "--- reward prepare function refs ---"
grep -n "reward_scales\|_prepare_reward\|reward_functions\|episode_sums" envs/t1.py | sed -n '1,220p'
