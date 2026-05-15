# Release Assets Manifest

`release_assets/` 已在本地整理好，但被 `.gitignore` 排除，不会进入 git 历史。创建 GitHub Release 时上传这些文件即可。

## 当前主包

- `release_assets/T1Circle-All-Results-1000-2000-3000-7000-Player.zip`
- `release_assets/T1-Fullbody-Checkpoints.zip`

这个包现在包含：

- 早期 S0-S8 全向步态回放
- 肩部探索的关键节点
- `Official StrongZero VelOpp` 肩部赢家
- `Upper9 CameraStable SpeedPush` 全身赢家

## 回放包

- `release_assets/T1-S0-S8-Replay-Player.zip`

## Checkpoints

- `release_assets/checkpoints/s0_stand.pth`
- `release_assets/checkpoints/s1_forward_slow.pth`
- `release_assets/checkpoints/s2_forward_0.8.pth`
- `release_assets/checkpoints/s3_forward_backward.pth`
- `release_assets/checkpoints/s4_turn.pth`
- `release_assets/checkpoints/s5_strafe.pth`
- `release_assets/checkpoints/s6_arc.pth`
- `release_assets/checkpoints/s7_diagonal.pth`
- `release_assets/checkpoints/s8_omni_full.pth`
- `release_assets/checkpoints/s8_mirror100_final.pth`

## 额外的后期关键 checkpoint

- `release_assets/checkpoints/t1_leg_model_4000.pth`
- `release_assets/checkpoints/shoulder4_official_strongzero_velopp500_model_500.pth`
- `release_assets/checkpoints/upper9_camera_stable_upperwarmstart_model_500.pth`
- `release_assets/checkpoints/upper9_camera_stable_frozen500_model_500.pth`
- `release_assets/checkpoints/upper9_camera_stable_speedpush_model_500.pth`

## 来源

这些文件来自远端训练机的 `D:\dev\Booster-T1\booster_gym`：

- 最终回放包：`artifacts\packages\T1-S0-S8-Replay-Player.zip`
- 最终模型：`logs\2026-04-29-18-41-41_t1_omni_s8_mirrorloss_from400_6144_i100\nn\model_100.pth`
