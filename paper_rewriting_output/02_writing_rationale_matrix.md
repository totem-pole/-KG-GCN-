# 第二章 Writing Rationale Matrix

| Row ID | Manuscript Unit | Planned Function | Motivation Link | Evidence/Citation Anchor | Planned Text Move | Final Text Check |
|---|---|---|---|---|---|---|
| C2-00 | 第二章整体 | 将系统级健康建模与后续KG-GCN建立数据接口 | 故障诊断前必须先获得条件化健康基准 | 用户代码/冻结结果；PaperSpine贡献边界 | 系统对象先行，VHP只作完整示例，结尾落到残差 | PASS：正文不把研究对象缩成VHP |
| C2-01 | 2.1 | 定义汽轮机系统设备级建模对象 | 系统拓扑决定输入边界和状态输出 | 既有汽轮机资料；Dettori2017MonitoringModels；Zhang2018AllCondition | 从系统介质流向引出分设备建模 | PASS：系统级语义优先于VHP案例 |
| C2-02 | 2.2 | 数学化U+B->Y及滑动窗口 | 运行工况变化需从设备健康变化中分离 | metadata.json；Chen2023AdaptiveTransfer | 定义条件化健康模型而非无条件预测 | PASS：模型目标与后续残差逻辑一致 |
| C2-03 | 2.3 | 说明GRU及真实网络结构 | 时序热状态存在动态依赖 | train_compare_four_models.py；configuration.json；Cho2014GRU | 只讲与本文模型有关的门控机制和结构 | PASS：无教科书式冗长算法综述 |
| C2-04 | 2.4 | 展示VHP如何落地系统级范式 | 代表性算例提供可核验工程细节 | VHP冻结测点、drawio、metadata | 给出16输入、27候选输出、24最终诊断状态 | PASS：明确VHP是代表性示例 |
| C2-05 | 2.5 | 用最短充分证据证明GRU健康基准可靠 | 健康模型是KG-GCN残差源 | 24点四算法CSV、预测PDF、loss CSV | 宏R2+代表性压力/温度+收敛性，不堆满24张图 | PASS：每个图表对应一个结论 |
| C2-06 | 2.6 | 构建后续KG-GCN动态节点特征 | 把状态监测与故障诊断统一起来 | prediction NPZ；师兄KG-GCN残差输入路线 | 定义原始残差和健康标准化残差 | PASS：不把残差直接等同故障 |
