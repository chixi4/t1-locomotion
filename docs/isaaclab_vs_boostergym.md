# IsaacLab vs Booster Gym

这篇先记录结论，详细对比后续再补。

## 简短结论

IsaacLab 路线更现代、模块化，也更适合长期工程化；但在这次 4 月下旬的目标里，最大需求是快速把 T1 的全向步态跑通。Booster Gym 的代码路径更短、T1 资源更直接、训练和回放工具更容易快速改，因此最后切回 Booster Gym。

## IsaacLab 学到的东西

- T1 asset、关节命名和镜像映射必须先固定。
- 对称性和上身安静约束有价值，但不能比速度和稳定性更早成为主目标。
- human-ref / symclock 是好方向，但数据链路和调参成本在当时偏高。

## Booster Gym 成功的原因

- T1 MuJoCo/XML/mesh 资源已经在官方框架里。
- 训练循环更容易直接插入 S0-S8 课程和 mirror loss。
- 评估、MuJoCo 录制、WebGL replay 可以很快闭环成最终展示包。

