# Mirror Loss

这篇先记录实现方式，详细实验对比后续再补。

## 实现位置

- `booster_gym/utils/t1_symmetry.py`：定义 T1 动作和观测的左右镜像。
- `booster_gym/utils/runner.py`：在 PPO loss 中加入 `_compute_mirror_loss`。

## 公式直觉

给定原始观测 `obs` 和策略动作均值 `action_loc`：

1. 把 `obs` 镜像成 `mirrored_obs`。
2. 用同一个策略计算 `mirrored_action_loc`。
3. 把原始 `action_loc` 镜像成 `mirrored_target`。
4. 让 `mirrored_action_loc` 接近 `mirrored_target`。

这等价于告诉策略：如果世界左右翻转，你的动作也应该左右翻转，而不是学出偏腿习惯。

## 最终设置

最终展示模型使用 `mirror_loss_coef=0.1`，从 S8 full 的 `model_400.pth` 继续训练 100 iter。

## 后续要补

- 不同 mirror loss 系数的对比。
- mirror loss 与 `leg_action_magnitude_symmetry` reward 的互补关系。
- 为什么它没有完全解决 foot clearance asymmetry。

