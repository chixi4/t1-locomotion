# AI Context

This repo is meant to be read as a research diary, not just code.

## Read First

1. `README.md`
2. `EXPERIMENT_INDEX.md`
3. `docs/research/2026-05-fullbody-camera-stable/README.md`
4. `docs/research/2026-05-fullbody-camera-stable/timeline.md`
5. `docs/research/2026-05-fullbody-camera-stable/experiment_table.tsv`
6. `docs/research/2026-05-fullbody-camera-stable/final_result.md`
7. `docs/research/2026-05-fullbody-camera-stable/reproduction.md`

## What This Repo Contains

- The original S0-S8 omni gait curriculum
- The later shoulder-only search
- The full-body / camera-stability residual search
- The web replay player and release packaging for human inspection
- Curated experiment records under `artifacts/autoresearch/`

## Current Best Working Route

- Shoulder-only winner: `T1Shoulder4 Official StrongZero VelOpp model_500`
- Full-body winner: `T1Upper9 CameraStable SpeedPush model_500`
- Useful follow-up probe: `SpeedUnlock` sequence, but it did not beat the speed-push result

## If You Continue Training

- Prefer fixed evaluation over training curves alone
- Keep a git checkpoint before each new experiment family
- Log the hypothesis and changed knobs before launching
- Add the result to the web replay player after evaluation

## What Not To Repeat

- Do not rely on chat history for experiment memory
- Do not judge action-shape goals from reward alone
- Do not let progress windows disappear during eval/pack steps

