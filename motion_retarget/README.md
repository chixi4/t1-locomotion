# Motion Retarget

这一目录保留 BVH/GMR 到 T1 训练 motion 的工具链，但不放任何 LAFAN1 数据或 LAFAN1 衍生数据。

## 为什么不放数据

LAFAN1 原始 BVH 使用 CC BY-NC-ND 4.0。由 LAFAN1 重定向到 T1 得到的 CSV/NPZ 属于衍生作品，因此也不放进仓库。仓库只保存脚本，数据由使用者在本地自行下载和生成。

## 脚本

- `bvh_to_t1.py`：基于 GMR 的 BVH 到机器人动作重定向入口。
- `convert_27dof_to_23dof.py`：把 GMR/T1 中间结果转换到训练使用的 T1 23DOF 布局。
- `mirror_motion.py`：生成左右镜像 motion。
- `split_curriculum.py`：把长 motion 切分成 stand、walk、fast 等课程阶段。

## 建议流程

1. 下载 LAFAN1 到本地目录，例如 `data/lafan1/`。这个目录已被 `.gitignore` 排除。
2. 安装 GMR 所需依赖，并确认 T1 URDF/mesh 路径可被脚本找到。
3. 运行 `bvh_to_t1.py` 把 BVH 重定向到 T1 中间格式。
4. 如有 DOF 布局差异，运行 `convert_27dof_to_23dof.py`。
5. 运行 `mirror_motion.py` 生成左右镜像版本。
6. 运行 `split_curriculum.py` 按训练课程拆分 motion。
7. 只在本地使用生成的 CSV/NPZ，不提交到 git。

## 和最终成果的关系

最终 `T1-S0-S8-Replay-Player.zip` 使用的是 Booster Gym 课程强化学习产物，不依赖把 LAFAN1 数据放进仓库。动作重定向工具链保留下来，是为了记录早期 human-ref / imitation 尝试和后续扩展方向。

