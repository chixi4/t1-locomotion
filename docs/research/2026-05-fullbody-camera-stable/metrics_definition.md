# Metrics

These are the metrics that mattered most in the later search.

| Metric | Meaning |
|---|---|
| `reset_events_per_env` | How often the policy fails or resets under the fixed test set |
| `lin_error_mean` | Average linear velocity tracking error |
| `yaw_error_mean` | Average yaw tracking error |
| `camera_tilt_p95` | 95th percentile camera tilt magnitude |
| `camera_ang_xy_rms` | RMS camera angular jitter in x/y |
| `stand_shoulder_abs_p95` | Shoulder angle magnitude while standing |
| `stand_elbow_abs_p95` | Elbow angle magnitude while standing |
| `stand_waist_abs_p95` | Waist angle magnitude while standing |
| `pitch_abs_p95_moving` | Moving forward/back shoulder pitch magnitude |
| `pitch_lr_antisym_corr` | How strongly left/right shoulder pitch behaves in anti-phase |
| `pitch_common_abs_p95` | Shared forward/back motion that both shoulders keep together |
| `side_left_out_p95_on_right` | Left-arm outward lift during rightward side stepping |
| `side_right_out_p95_on_left` | Right-arm outward lift during leftward side stepping |
| `elbow_abs_p95_moving` | Moving elbow activity |
| `waist_abs_p95_moving` | Moving waist activity |
| `residual_abs_p95` | Residual action magnitude on top of the frozen leg baseline |

## Reading Rule

For this project, curves alone are not enough. A run can have ugly reward curves and still be a good behavior result, or the reverse. Fixed-command evaluation and replay inspection decide the winner.

