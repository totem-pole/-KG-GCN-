# 第五章实验状态 v0.3（超时安全版）

## 1. 执行策略调整
前期超时的主要原因不是模型无法训练，而是把“数据构造、多模型、多随机种子、导出结果”放在一次执行中。后续固定采用分块执行：
1. 数据集单独生成并落盘；
2. 每个模型单独训练；
3. 先1个seed筛结构，再补多seed；
4. 每步结果即时保存CSV/NPZ；
5. masked pretraining仅在基础KG-GCN结构冻结后执行。

这样任一步超时都不会丢失前序结果。

## 2. 当前v03数据集已成功生成
`/mnt/data/levelC_experiment_v03/levelC_dataset_v03.npz`

- Train: 1712 windows
- Validation: 556 windows
- Test: 560 windows
- 节点数: 24个VHP最终状态测点
- 每个窗口长度: 60
- 类别: F0正常、F1通流面积增大/SPE、F2沉积/通流面积减小、F3汽封/泄漏类半物理场景
- 数据划分: 自然日互斥时间块，窗口不跨split

## 3. 已经恢复出的快速单seed结果（仅用于结构诊断，不能作为正式论文最终数字）
原始“GCN + 全局平均池化”表现较差：
- CNN: Accuracy 0.5036, Macro-F1 0.4886
- KG-GCN-1L: Accuracy 0.3464, Macro-F1 0.3033
- Identity-GCN: Accuracy 0.3054, Macro-F1 0.2841
- Corr-GCN: Accuracy 0.3054, Macro-F1 0.2768
- KG-GCN-2L: Accuracy 0.3071, Macro-F1 0.2573
- AE: Accuracy 0.2589, Macro-F1 0.1833

该结果表明“全局平均池化”会显著损失固定传感器图中的节点身份信息，不能直接作为最终KG-GCN结构。

## 4. Node-aware readout诊断
进一步保留节点身份后，结果明显改善：

### FlatResidual readout
- Identity: Accuracy 0.8536, Macro-F1 0.8544
- Corr: Accuracy 0.8732, Macro-F1 0.8740
- KG: Accuracy 0.8911, Macro-F1 0.8916

在相同FlatResidual读出结构下，当前单seed pilot呈现：
`KG > Corr > Identity`

这说明A0本机拓扑在保留节点身份的条件下可能具有正向价值，但该结论仍需多seed、同参数量和不同故障严重度重复验证。

### FlatNoResidual诊断
- Identity: Accuracy 0.9054, Macro-F1 0.9059
- Corr: Accuracy 0.7089, Macro-F1 0.7085
- KG: Accuracy 0.8393, Macro-F1 0.8400

Identity在该结构下异常强，说明半物理故障签名可能包含较明显的固定节点位置模式，因此不能只依据这一组结果声称KG优越。后续必须增加：
1. 多seed；
2. 未见严重度；
3. 故障时间轮廓变化；
4. Raw vs Residual；
5. 更严格的图读出结构。

## 5. 当前结构结论
1. 标准全局平均池化不适合当前固定24节点诊断任务，容易把故障发生位置平均掉；
2. 后续主结构优先采用“节点身份保留的图表示”，而不是简单Mean Pool；
3. KG效果只能在与Identity/Corr完全相同读出层下比较；
4. 当前v03结果只能称为pilot，不进入论文主结果表；
5. masked pretraining暂缓，等基础图结构稳定后再加。

## 6. 下一步分块执行顺序
### Block A：结构冻结
- 固定1层GCN + node-aware readout；
- 参数量对齐Identity/Corr/KG；
- 单seed确认训练稳定。

### Block B：三随机种子
分别运行seed 123/456/789，每个seed独立落盘。

### Block C：主消融
- Identity vs Corr vs KG
- Raw vs Residual

### Block D：鲁棒性
- light / medium / strong severity
- unseen severity
- step / ramp / sigmoid profile

### Block E：masked pretraining
在Block A-D稳定后比较scratch vs masked-pretrain。

## 7. 论文写作口径
正式论文第五章暂时只写实验协议、评价指标和实验设计，不回填当前单seed pilot数字。只有完成至少3个随机种子且结构冻结后，才进入主表。当前pilot可保留在研发记录中用于解释模型结构选择。
