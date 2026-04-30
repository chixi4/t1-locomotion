# final_mirror100 评估摘要

评分来源：远端 `booster_gym/artifacts/omni_scores/final_mirror100/`。

最终 checkpoint：

```text
logs/2026-04-29-18-41-41_t1_omni_s8_mirrorloss_from400_6144_i100/nn/model_100.pth
```

## 总结

13 个固定命令测试全部 `stable_ok=true`、`tracking_ok=true`、`falls=0`。这说明最终策略在站立、前进、后退、转向、横移、弧线、斜向和混合命令上都能稳定执行。

主要未完全解决的是对称性：只有 `arc_neg`、`diag_neg`、`diag_pos` 三个测试 `symmetry_ok=true`。其余测试大多因为 foot clearance asymmetry、foot slip asymmetry 或 leg action magnitude asymmetry 未过阈值。也就是说，最终成果已经是可展示的全向步态，但还不是完全“左右干净”的步态。

## 评分表

| 原始文件 | 整理后文件 | 命令 `(vx, vy, yaw)` | 稳定 | 跟踪 | 对称 | 摔倒 | 备注 |
|---|---|---:|---:|---:|---:|---:|---|
| `stand.json` | `stand.json` | `(0.0, 0.0, 0.0)` | yes | yes | no | 0 | 站立稳定，foot clearance asymmetry 偏大 |
| `forward_08.json` | `forward_0.8.json` | `(0.8, 0.0, 0.0)` | yes | yes | no | 0 | 前进速度跟踪好，仍有左右脚滑移/抬脚差异 |
| `back_03.json` | `backward_0.3.json` | `(-0.3, 0.0, 0.0)` | yes | yes | no | 0 | 远端没有 `backward_0.5`，保留真实 `-0.3 m/s` |
| `turn_pos.json` | `turn_left.json` | `(0.0, 0.0, 0.5)` | yes | yes | no | 0 | 原地正向 yaw 测试 |
| `turn_neg.json` | `turn_right.json` | `(0.0, 0.0, -0.5)` | yes | yes | no | 0 | 原地反向 yaw 测试 |
| `strafe_pos.json` | `strafe_right.json` | `(0.0, 0.3, 0.0)` | yes | yes | no | 0 | 横移正向测试 |
| `strafe_neg.json` | `strafe_left.json` | `(0.0, -0.3, 0.0)` | yes | yes | no | 0 | 横移反向测试 |
| `arc_pos.json` | `arc.json` | `(0.5, 0.0, 0.5)` | yes | yes | no | 0 | 前进加正向转弯 |
| `arc_neg.json` | `arc_reverse.json` | `(0.5, 0.0, -0.5)` | yes | yes | yes | 0 | 前进加反向转弯，对称性通过 |
| `diag_pos.json` | `diagonal.json` | `(0.5, 0.3, 0.0)` | yes | yes | yes | 0 | 斜向正向，对称性通过 |
| `diag_neg.json` | `diagonal_reverse.json` | `(0.5, -0.3, 0.0)` | yes | yes | yes | 0 | 斜向反向，对称性通过 |
| `omni_pos.json` | `omni_mix.json` | `(0.6, 0.3, 0.4)` | yes | yes | no | 0 | 全向混合正向 |
| `omni_neg.json` | `omni_mix_reverse.json` | `(-0.3, -0.3, -0.4)` | yes | yes | no | 0 | 全向混合反向 |

## 可读结论

- 展示层面：可以作为 S0-S8 全向步态成果展示，回放包已经覆盖各阶段。
- 训练层面：S8 full 后加入 `mirror_loss_coef=0.1` 的 100 iter 修正是最终使用版本。
- 继续优化方向：优先做 foot clearance / slip asymmetry 的奖励与课程再平衡，其次再做更强扰动、摩擦随机化和地形。

