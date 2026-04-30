# T1 Locomotion

这是一个 Booster T1 双足机器人的全向步态策略的探索回顾。目前的成果是基于 Booster Gym 的 S0 到 S8 课程强化学习：从站立、慢速前进、0.8 m/s 前进，逐步扩展到后退、原地转向、横移、弧线、斜向和全向混合命令。相关成果可以查看 GitHub Release 中的 `T1-S0-S8-Replay-Player.zip` 的网页展示包。

![T1 S0-S8 全向步态演示](media/demo.gif)

## 这是什么

这是一次 T1 步态训练项目的过程与成果的展示仓库：

- **`booster_gym/`**：最终跑通的 Booster Gym 修改版，包括 T1 环境、奖励、S0–S8 课程阶段、PPO runner、mirror loss 和评估/录制/回放工具。
- **`isaaclab_experiments/`**：4 月下旬在 IsaacLab 方向做过的配置、patch 和复盘。它们不是最终路线，但解释了为什么后来切回 Booster Gym。
- **`motion_retarget/`**：BVH/GMR 到 T1 动作数据的脚本链路，不包含 LAFAN1 原始或衍生数据。
- **`results/`**：最终 `final_mirror100` 的可追溯评分 JSON。

## 快速看效果

Release 附件中包含：

- `T1-S0-S8-Replay-Player.zip`——18 个回放视角 HTML（S0–S8 每阶段 orbit / multi_top 两种视角），内含 Three.js、STLLoader 和 T1 mesh
- `checkpoints/s8_mirror100_final.pth`——最终全向策略权重
- S0 到 S8 各阶段 checkpoint

解压 Replay Player 后，macOS 双击 `Start-macOS.command`，Windows 双击 `Start-Windows.bat`，即可在浏览器里看 3D WebGL 回放。

## 项目的探索历程

| 日期 | 主要内容 | 结果 |
|---|---|---|
| 2026-04-20 | 拉取 Booster SDK/deploy/assets、Robocup demo，整理 T1 上机与网络检查脚本 | 形成最早的机器人连接、配置和 demo 运行资料 |
| 2026-04-22 | 搭 IsaacLab Windows 环境、PyTorch/CUDA 检查，开始性能和视觉辅助实验 | IsaacLab 能跑，但 T1 任务没有稳定训练闭环 |
| 2026-04-23 | 引入 GMR/LAFAN1 动作重定向，尝试 human-ref、symclock、mirror、biomech curriculum | 得到动作数据工具链，但 IsaacLab 步态效果不稳定 |
| 2026-04-24–27 | 继续 IsaacLab：quiet upper、full symmetry、arm envelope、fullspeed momentum arm 等方向 | 学到约束设计经验，但未达到满意全向步态 |
| 2026-04-28 | 切回 Booster Gym，先跑 S0 站立，再扩展 S1/S2 前进速度 | S0/S1/S2 跑通，确认旧栈更快更可控 |
| 2026-04-29 | 连续完成 S3 到 S8，加入对称性、滑脚、mirror loss 修正 | 产出 `s8_mirror100_final.pth`、评分 JSON 和回放包 |

## 最终策略

目前的最优模型是：

```
logs/2026-04-29-18-41-41_t1_omni_s8_mirrorloss_from400_6144_i100/nn/model_100.pth
```

训练链路：

1. **`s0_stand`**：站住和轻微踏步，4096 envs。
2. **`s1_forward_slow`**：慢速前进（vx 0.25–0.45），开始强调前向速度跟踪。
3. **`s2_forward_05`** / **`s2_forward_08`**：把前进速度扩到 0.5 和 0.8 m/s。
4. **`s3_back_slow`** → **`s3_forward_backward`**：先练纯后退（vx -0.45– -0.15），再合并前进后退。
5. **`s4_turn`**：加入 yaw 原地转向（±0.8 rad/s）。
6. **`s5_strafe`**：加入左右横移（vy ±0.45）。
7. **`s6_arc`**：组合前进和转向，形成弧线行走。
8. **`s7_diagonal`**：组合前进和横移。
9. **`s8_omni`**：vx [-0.8, 1.0]、vy [-0.8, 0.8]、yaw [-1.0, 1.0] 的全向混合。
10. **`s8_mirror100_final`**：从 S8 full checkpoint 继续 100 iter，`mirror_loss_coef=0.1`，得到当前最好回放包。

关键文件：

- `booster_gym/envs/t1_omni_stages.py`：S0–S8 阶段定义、命令范围、奖励权重覆盖。
- `booster_gym/envs/T1.yaml`：基础配置（物理参数、PD 增益、DR 参数、PPO 超参）。
- `booster_gym/envs/t1.py`：观测构造（47 维）、reset、reward 和全向指标。
- `booster_gym/utils/runner.py`：PPO 训练、checkpoint、命令行参数、mirror loss。
- `booster_gym/utils/t1_symmetry.py`：T1 左右腿动作/观测镜像映射（12 维动作）。

## 方法概览

![T1 Locomotion 训练流程架构图](media/architecture.png)

### 课程学习 S0 到 S8

先学习站稳，再练习从低速到中速的行走，学会倒退，然后学习横移、转向和混合方向。

S8 之后还定义了 S9 系列（`s9_noise`、`s9_actuator`、`s9_friction`、`s9_push`、`s9_terrain`），用于在全向基础上叠加观测噪声、执行器扰动、摩擦随机化、外力推动和复杂地形等鲁棒性训练。

### 奖励设计

主线奖励围绕三个目标调权：命令跟踪、稳定性、动作质量。S1/S2 更重视 `tracking_lin_vel_x`，S4 提高 `tracking_ang_vel`，S5/S7/S8 提高 `tracking_lin_vel_y`，后期加重 `feet_slip` 和腿部动作幅值对称惩罚。

### Mirror Loss

`mirror_loss` 让"镜像后的观测输入"得到的动作，尽量等于"原动作的左右镜像"。这在每一步的策略网络层加上软约束，可以帮助减少左右腿动作的相对步态偏差。

## 结果

最终评分来自 `results/omni_scores/` 中的 JSON。所有固定命令测试均未摔倒，稳定性和命令跟踪全部通过。对称性指标未全部通过，残留问题在 foot clearance / slip asymmetry。

| 测试 | 命令 `(vx, vy, yaw)` | 稳定 | 跟踪 | 对称 | 摔倒 |
|---|---:|:---:|:---:|:---:|---:|
| stand | `(0.0, 0.0, 0.0)` | yes | yes | no | 0 |
| forward_0.8 | `(0.8, 0.0, 0.0)` | yes | yes | no | 0 |
| backward_0.3 | `(-0.3, 0.0, 0.0)` | yes | yes | no | 0 |
| turn_left | `(0.0, 0.0, 0.5)` | yes | yes | no | 0 |
| turn_right | `(0.0, 0.0, -0.5)` | yes | yes | no | 0 |
| strafe_right | `(0.0, 0.3, 0.0)` | yes | yes | no | 0 |
| strafe_left | `(0.0, -0.3, 0.0)` | yes | yes | no | 0 |
| arc | `(0.5, 0.0, 0.5)` | yes | yes | no | 0 |
| arc_reverse | `(0.5, 0.0, -0.5)` | yes | yes | yes | 0 |
| diagonal | `(0.5, 0.3, 0.0)` | yes | yes | yes | 0 |
| diagonal_reverse | `(0.5, -0.3, 0.0)` | yes | yes | yes | 0 |
| omni_mix | `(0.6, 0.3, 0.4)` | yes | yes | no | 0 |
| omni_mix_reverse | `(-0.3, -0.3, -0.4)` | yes | yes | no | 0 |

## 动作数据

仓库只放重定向脚本，因为协议原因，不包含 LAFAN1 原始 BVH 和由 LAFAN1 生成的 CSV/NPZ。需要时，可以按 `motion_retarget/README.md` 在本地下载 LAFAN1 再运行脚本生成 motion。

## Release 附件

`release_assets/` 被 `.gitignore` 排除，不进 git 历史。上传 GitHub Release 时可使用：

- `release_assets/T1-S0-S8-Replay-Player.zip`
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

## 致谢

- **Booster Robotics**：Booster Gym 框架、T1 机器人模型与相关 SDK/Assets
- **IsaacLab / Isaac Gym 生态**：提供早期探索路线和仿真训练基础
- **Ubisoft LAFAN1**：动作捕捉数据来源（数据未放入本仓库）
- **GMR**：BVH/人体动作到机器人动作的重定向流程参考

## 许可证

本仓库新增的整理、文档和自写脚本采用 [MIT License](LICENSE)。第三方代码和资产遵守各自许可证，详见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) 与 [`booster_gym/LICENSE`](booster_gym/LICENSE)。
