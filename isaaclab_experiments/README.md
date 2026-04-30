# IsaacLab 探索记录

这一目录记录项目早期尝试过的 IsaacLab 路线。它不是最终成功路线，而是保留关键配置和 patch，说明为什么后来切回 Booster Gym。

## 目录

- `configs/`：T1 速度任务、human-ref、symclock、quiet-upper、full-sym、fullspeed momentum arm 等配置快照。
- `patches/t1_task_registration.patch`：T1 任务注册、T1 asset、对称性函数、播放/测试辅助脚本等改动。
- `patches/rsl_rl_windows_compat.patch`：Windows 下运行 IsaacLab/RSL-RL 时做过的兼容改动。
- `results/R1_to_R11_summary.md`：从 R1 到 R11 的路线复盘。

## 为什么没有把完整 IsaacLab 放进来

完整 IsaacLab 体积大、依赖复杂，而且这些实验不是最终可复现成果。这里保留的是“决策证据”：当时改过什么、尝试过什么、哪些方向没有达到预期。真正的最终训练链路在 `booster_gym/`。

## 这部分的价值

IsaacLab 路线虽然没有产出最终步态，但它帮助确认了几个事实：

- T1 的 asset、关节命名、镜像映射和 Windows 运行环境需要先打通，否则训练问题会被工程问题淹没。
- human-ref / symclock / full-sym 这些想法对步态形态有帮助，但在当时环境里调参成本过高。
- 最终回到 Booster Gym 后，课程学习和奖励权重可以更快闭环。

