# Level-C VHP KG-GCN Experiment v07

## 1. 定位

v07作为第五章VHP代表性算例的“清晰可观测典型故障”主实验候选。总体研究对象仍为燃煤机组汽轮机系统。

与早期版本的关系：
- v03：0.5--1.3 sigma，弱/中等故障鲁棒性；
- v05：1.0--2.0 sigma，中等故障；
- v06：1.5--2.5 sigma，发展期故障；
- **v07：2.0--3.0 sigma，清晰可观测典型故障主实验候选。**

所有版本保持相同真实健康残差背景、24个VHP DCS节点、日期split、故障空间签名和step/ramp/sigmoid时间轮廓，只改变故障严重度范围。

## 2. 数据协议

- Train / Val / Test：1712 / 556 / 560；
- 四类平衡：F0正常、F1 VHP通流侵蚀/SPE、F2一级抽汽逆止门后管路泄漏、F3 VHP汽封泄漏；
- 自然日互斥split，窗口禁止跨日/跨split；
- v07故障严重度：2.0--3.0个训练健康残差标准差；
- KG邻接继续使用v04冻结的A0本机有序传感器拓扑，不由故障签名反推。

## 3. 与杨昊东完全对齐的训练路线

本版恢复并验证杨昊东第四章的两阶段KG-GCN训练思路：

1. 将训练残差窗口视为无标签样本，不使用类别标签；
2. 随机遮蔽15%的残差位置；
3. 遮蔽窗口与归一化KG邻接矩阵共同输入GCN编码器；
4. 解码器仅在被遮蔽位置计算重构MSE；
5. 完成预训练后冻结GCN编码器参数；
6. 使用带标签故障窗口训练全连接分类层；
7. Softmax输出故障类别，交叉熵作为分类损失。

注意：预训练只使用train split，不接触validation/test，也不使用故障类别标签。

## 4. 当前关键结果：seed=123

| Metric | Result |
|---|---:|
| Accuracy | **0.980357** |
| Macro-Precision | **0.980599** |
| Macro-Recall | **0.980357** |
| Macro-F1 | **0.980322** |
| Validation Macro-F1 | 0.973066 |
| Mask ratio | 0.15 |
| Pretrain best epoch | 40 |
| Pretrain val masked MSE | 0.354710 |
| Classifier best epoch | 23 |

本结果已经达到并略超过杨昊东KG-GCN工程案例约97.59% Accuracy的量级，但目前仍需补足多随机种子、CNN/AE公平对比和混淆矩阵后才能作为最终论文结果。

## 5. 与scratch对比

同一v07、seed=123，未进行Mask预训练的Dual-path KG-GCN Macro-F1约0.9679；加入Mask预训练并冻结编码器后提升到0.9803，说明本版预训练目标已经由早期v04的负收益转为正收益。

## 6. 下一步

1. 补seed=101/202/303/404，计算5-seed均值与标准差；
2. 在同一v07数据上重新训练CNN、AE，确保最终结果方向为KG-GCN > CNN > AE；
3. 输出代表seed混淆矩阵及逐类Precision/Recall/F1；
4. 做scratch vs pretrain消融；
5. v03/v05/v06作为严重度鲁棒性补充实验保留，不覆盖或删除。
