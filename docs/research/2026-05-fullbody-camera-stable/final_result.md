# Final Result

## Shoulder Winner

- Label: `T1Shoulder4 Official StrongZero VelOpp model_500`
- Goal: frozen-leg shoulder-only result with near-zero standing base, forward/back anti-phase pitch, and contralateral side lift
- Outcome: selected as the best shoulder-only result

## Full-Body Winner

- Task: `T1Upper9CameraStableOfficialOpenLeg18LegResidualSpeedPush_train300`
- Label: `T1Upper9CameraStableSpeedPush model_500`
- Checkpoint: `logs/2026-05-14-15-08-00/nn/model_500.pth`
- Selected because: best combined camera stability, speed retention, and visible arm coordination

## Key Fixed-Eval Numbers

- `reset_events_per_env`: `0.0391`
- `lin_error_mean`: `0.1555`
- `yaw_error_mean`: `0.1179`
- `camera_tilt_p95`: `0.0956`
- `camera_ang_xy_rms`: `0.4364`
- `stand_shoulder_abs_p95`: `0.0532`
- `pitch_abs_p95_moving`: `0.2133`
- `pitch_lr_antisym_corr`: `0.9301`
- `side_left_out_p95_on_right`: `0.1943`
- `side_right_out_p95_on_left`: `0.2021`
- `elbow_abs_p95_moving`: `0.1075`
- `waist_abs_p95_moving`: `0.1039`

## Why It Beat The Probe

- `SpeedUnlock` expanded the grid further, but resets and instability grew.
- `SpeedPush model_500` was the cleaner tradeoff, so it stayed the final winner.

## Artifact Pointers

- Selection JSON: `artifacts/autoresearch/upper9_camera_stable_openleg18_legresidual_speedpush300_selected.json`
- Confirmation eval: `artifacts/autoresearch/speedpush_model500_confirm128_cases.json`
- Summary: `artifacts/autoresearch/upper9_camera_stable_openleg18_speedpush500_eval/upper9_camera_stable_openleg18_speedpush500_summary.json`

