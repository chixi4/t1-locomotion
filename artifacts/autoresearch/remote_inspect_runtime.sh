#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official
echo "PATH=$PATH"
which python || true
which python3 || true
which ninja || true
which /opt/conda/bin/python || true
/opt/conda/bin/python - <<'PY'
import shutil, sys
print(sys.executable)
print("ninja", shutil.which("ninja"))
try:
    import ninja
    print("ninja module", ninja.__file__)
except Exception as exc:
    print("ninja module error", repr(exc))
PY
ls -la /root/.cache/torch_extensions/py38_cu118 || true
find /root/.cache/torch_extensions/py38_cu118 -maxdepth 3 -type f -name 'gymtorch*.so' -print || true
ls artifacts/autoresearch/run_pitchroll1000_schtasks_cuda.sh artifacts/autoresearch/run_satguard2000_schtasks_cuda.sh 2>/dev/null || true
