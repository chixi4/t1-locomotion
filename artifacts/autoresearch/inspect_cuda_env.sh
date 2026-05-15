#!/usr/bin/env bash
set -euo pipefail

echo "---env key---"
env | grep -E 'CUDA|LD_LIBRARY_PATH|ISAAC|CONDA|PATH|NVIDIA|WSL' | sort || true

echo "---ldconfig cuda/physx---"
ldconfig -p 2>/dev/null | grep -E 'libcuda|PhysX|physx' | head -40 || true

echo "---profile snippets---"
for f in /etc/profile ~/.bashrc ~/.profile ~/.bash_profile /etc/bash.bashrc; do
  echo "### $f"
  [[ -f "$f" ]] && grep -nE 'CUDA|LD_LIBRARY_PATH|ISAAC|conda|isaac|physx|gym' "$f" || true
done

echo "---isaac preload files---"
grep -RsnE 'failed to preload|LD_LIBRARY_PATH|libcuda|PhysX|preload' /opt/isaacgym/python/isaacgym 2>/dev/null | head -80 || true
