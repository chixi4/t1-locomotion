# 第三方来源与许可证说明

本仓库把最终成果、实验痕迹和复现实用脚本整理到一个可以上传 GitHub 的结构里。根目录 `LICENSE` 只覆盖我新增的整理、文档、配置和自写脚本；第三方项目继续遵守各自原许可证。

## Booster Gym

- 位置：`booster_gym/`
- 来源：Booster Robotics 官方 Booster Gym 框架及 T1 MuJoCo/URDF/mesh 资源
- 许可证：`booster_gym/LICENSE` 中声明为 Apache License 2.0，并注明其继承/参考了 IsaacGymEnvs、legged_gym、rsl_rl、humanoid-gym 等项目
- 本仓库处理方式：保留必要源码、T1 XML/URDF 与 mesh，用于复现 S0-S8 全向步态训练和回放

## Booster Assets / T1 STL

- 位置：`booster_gym/resources/T1/meshes/`
- 来源：Booster Robotics 公开资源
- 本仓库处理方式：作为机器人可视化与 MuJoCo 回放所需资源保留，并在 README 中注明来源

## IsaacLab

- 位置：`isaaclab_experiments/`
- 来源：本地 `third_party/IsaacLab-win` 上的探索性修改
- 许可证：IsaacLab 本体遵守其上游许可证；这里保留的是配置快照和 patch，便于理解曾经走过的路线
- 本仓库处理方式：不把完整 IsaacLab 依赖放入仓库，只保留 T1 任务配置、任务注册 patch 和 Windows/RSL-RL 兼容 patch

## GMR / Motion Retargeting

- 位置：`motion_retarget/`
- 来源：GMR 流程与本项目中针对 T1 的动作转换、镜像、课程拆分脚本
- 本仓库处理方式：只保留脚本，不保留受限动作数据

## LAFAN1

- 来源：Ubisoft La Forge Animation Dataset
- 许可证：CC BY-NC-ND 4.0
- 本仓库处理方式：不包含原始 BVH，也不包含由 LAFAN1 重定向得到的 CSV/NPZ。需要动作数据时，请按 `motion_retarget/README.md` 自行下载数据并在本地生成训练 motion。

