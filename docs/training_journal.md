# 训练日志

这份文件先保留简明时间线，后续可以扩展为完整训练日记。

## 时间线

- 2026-04-20：整理 Booster T1 连接、SDK、Assets、Robocup demo 和机器人检查脚本，确认项目基础环境。
- 2026-04-22：开始 IsaacLab Windows 环境与 CUDA/PyTorch 适配，做性能探针和 T1 任务注册准备。
- 2026-04-23：引入 GMR/LAFAN1 动作重定向，尝试 human-ref、symclock、mirror/recurrent 等方向。
- 2026-04-24 到 2026-04-27：继续 IsaacLab 奖励、对称性、quiet upper、full-sym、fullspeed momentum arm 等实验，但没有得到满意步态。
- 2026-04-28：切回 Booster Gym，跑通 S0 站立和 S1/S2 前进课程，确认旧栈训练效率更高。
- 2026-04-29：完成 S3-S8 全向课程，加入 symmetry/slip/mirror loss 修正，生成最终评分和 WebGL replay package。

## 最终 checkpoint 链

| 阶段 | 训练目录 | 用于下一阶段的 checkpoint |
|---|---|---|
| S0 | `2026-04-28-18-30-07_t1_omni_s0_stand_from500_i500` | `model_500.pth` |
| S1 | `2026-04-28-20-25-23_t1_omni_s1_forward_drive_v4_sym_12288_i500` | `model_500.pth` |
| S2 | `2026-04-28-23-23-48_t1_omni_s2_forward_08_survival_sym_12288_i300` | `model_300.pth` |
| S3 | `2026-04-29-12-54-49_t1_omni_s3_forward_backward_mix_12288_i500` | `model_500.pth` |
| S4 | `2026-04-29-13-44-13_t1_omni_s4_turn_sym_12288_i400` | `model_400.pth` |
| S5 | `2026-04-29-14-24-37_t1_omni_s5_strafe_sym_12288_i500` | `model_300.pth` |
| S6 | `2026-04-29-15-20-14_t1_omni_s6_arc_sym_12288_i400` | `model_200.pth` |
| S7 | `2026-04-29-16-01-27_t1_omni_s7_diagonal_sym_12288_i400` | `model_200.pth` |
| S8 full | `2026-04-29-16-43-50_t1_omni_s8_full_sym_12288_i600` | `model_400.pth` |
| S8 mirror final | `2026-04-29-18-41-41_t1_omni_s8_mirrorloss_from400_6144_i100` | `model_100.pth` |

## 后续要补

- 每天做了哪些关键决策。
- 每个失败 run 的原因和当时的判断。
- 为什么最后没有使用 18:53 的 polish run，而是把 mirror100 作为展示结果。

