#!/usr/bin/env bash
set -euo pipefail
/opt/conda/bin/python - <<'PY'
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    print("tensorboard yes")
except Exception as exc:
    print("tensorboard no", repr(exc))
try:
    import tensorflow as tf
    print("tensorflow yes", tf.__version__)
except Exception as exc:
    print("tensorflow no", repr(exc))
PY
