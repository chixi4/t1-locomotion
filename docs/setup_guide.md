# 环境搭建指南

这篇先保留最小复现线索，完整从零安装步骤后续再补。

## 最小要求

- Windows 或 Linux 训练机。
- NVIDIA GPU 和可用 CUDA。
- Python 3.8。
- Isaac Gym。
- PyTorch 与 CUDA 版本匹配。
- Booster Gym 依赖，见 `booster_gym/requirements.txt`。

## 基本步骤

1. 安装 Isaac Gym，并确认 sample 能运行。
2. 创建 Python 3.8 虚拟环境。
3. 安装 PyTorch、Isaac Gym Python 包和 `booster_gym/requirements.txt`。
4. 进入 `booster_gym/`，运行一个短 smoke test：

```bash
python train.py --task=T1 --curriculum_stage=s0_stand --headless=true --max_iterations=1 --num_envs=64
```

5. 能正常创建环境后，再按 README 的 S0-S8 顺序训练。

## 后续要补

- Windows/WSL/Linux 分平台安装步骤。
- CUDA、PyTorch、Isaac Gym 版本组合。
- 常见报错与排查。

