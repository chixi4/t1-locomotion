# Release Assets Manifest

`release_assets/` 已在本地整理好，但被 `.gitignore` 排除，不会进入 git 历史。创建 GitHub Release 时上传这些文件即可。

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

## 来源

这些文件来自远端训练机的 `D:\dev\Booster-T1\booster_gym`：

- 最终回放包：`artifacts\packages\T1-S0-S8-Replay-Player.zip`
- 最终模型：`logs\2026-04-29-18-41-41_t1_omni_s8_mirrorloss_from400_6144_i100\nn\model_100.pth`

