# 2026-05 Full-Body Camera-Stable Search

This folder is the handoff package for the later T1 exploration after the leg curriculum was already working.

## Goal

Train a strategy that keeps the camera stable while preserving:

- near-zero shoulder base when standing
- forward/back anti-phase arm swing
- side-step contralateral arm lift
- low-amplitude elbow/waist assistance
- acceptable leg tracking after opening residual full-body control

## Current Winner

- Task: `T1Upper9CameraStableOfficialOpenLeg18LegResidualSpeedPush_train300`
- Selected checkpoint: `logs/2026-05-14-15-08-00/nn/model_500.pth`
- Selection reason: best combined camera stability, forward 1.8 m/s retention, and arm motion

## Read In Order

1. `timeline.md`
2. `experiment_table.tsv`
3. `metrics_definition.md`
4. `lessons.md`
5. `final_result.md`
6. `reproduction.md`

## Related Files

- `artifacts/autoresearch/upper9_camera_stable_openleg18_speedpush500_eval/summary.json`
- `artifacts/autoresearch/upper9_camera_stable_openleg18_legresidual_speedpush300_selected.json`
- `artifacts/autoresearch/upper9_camera_stable_openleg18_legresidual_speedunlock300_fixed_eval/summary.json`
- `artifacts/autoresearch/upper9_camera_stable_frozen500_eval/upper9_camera_stable_frozen500_summary.json`

