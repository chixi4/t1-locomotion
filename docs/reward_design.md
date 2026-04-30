# 奖励设计

这篇先记录核心结构，详细推导后续再补。

## 核心思路

奖励设计围绕三件事：命令跟踪、稳定性、动作质量。

- 命令跟踪：`tracking_lin_vel_x`、`tracking_lin_vel_y`、`tracking_ang_vel`。
- 稳定性：`base_height`、`orientation`、`collision`、`lin_vel_z`、`ang_vel_xy` 等。
- 动作质量：`action_rate`、`torques`、`dof_acc`、`feet_slip`、腿部动作幅值对称性。

## 阶段调权

S1/S2 先把 `tracking_lin_vel_x` 拉高，解决“往前走”这件事。S4 转向阶段提高 `tracking_ang_vel`。S5/S7/S8 提高横向速度权重。后期 `feet_slip` 和 `leg_action_magnitude_symmetry` 权重变重，是为了减少滑脚和左右腿动作不均。

## 后续要补

- 每个 reward term 的实际公式。
- 哪些权重导致过度保守、滑脚或左右不均。
- `mirror_loss` 和 reward symmetry 的区别。

