# 第三章 Evidence Bank — 汽轮机故障知识图谱

## Evidence policy
- **A**：直接系统拓扑/真实DCS资料，或原始同行评议论文直接支持的设备—故障—参数关系，可进入核心图谱。
- **B**：论文或工程资料提供间接支持、关系方向需进一步结合本机组资料确认，暂作为候选/辅助边。
- 不直接使用无来源LLM生成关系；LLM只负责候选抽取。
- 当前范围聚焦热力/通流故障，不纳入轴系振动、转子不平衡等机械故障。

## Core literature evidence

| Key | 研究对象 | 可进入图谱的主要证据 |
|---|---|---|
| ZhangHu2021FaultPropagationGNN | 汽轮机系统故障传播GNN | 设备沿物质/信息流连接；HP/IP/LP汽封缺陷→蒸汽泄漏→汽耗增加；LP轴封缺陷涉及空气漏入/真空问题 |
| ZhangHu2021ControlValveGraph | 汽轮机高压调节阀 | 控制阀决定蒸汽流量/功率；死区表现为指令变化而实际阀位不响应，可造成系统振荡 |
| Kubiak2002TurbineDegradation | 汽轮机通流退化现场案例 | 常见退化包括叶片粗糙度、SPE造成喷嘴面积增加、迷宫/径向汽封磨损、叶片沉积；关键压力可用于识别退化 |
| Kubiak2002ScaleDeposit | 汽轮机叶片沉积 | 沉积降低通流面积、出力和效率；级前后压力可用于估计堵塞/面积变化 |
| Xu2011HPIPLeakage | 600MW超临界汽轮机HP-IP泄漏 | N2中间汽封泄漏显著影响IP效率；热再温度与IP排汽温度是泄漏估计关键测量参数 |
| Li2016CondenserFouling | 600MW燃煤机组凝汽器 | 水侧污垢降低热有效性；污垢热阻可用于在线监测；异常趋势还可提示泄漏问题 |
| Medica2018CondenserCondition | 燃煤机组凝汽器 | 凝汽压力随负荷、循环水流量/入口温度和污垢变化；较高凝汽压力降低汽轮机可用焓降和效率/出力 |
| Barszcz2011FeedwaterHeater | 225MW燃煤机组给水加热器 | 模型诊断可跟踪小时/天尺度性能变化，包括管道裂纹导致的内部泄漏；换热功率等为关键指标 |
| Heo2012FWHLeakage | 闭式给水加热器内漏 | 管束、管板、隔板内漏影响回热循环效率和安全，可根据热力性能指标定位/估计泄漏 |
| Hogan2021KnowledgeGraphs | KG方法综述 | 支持KG的实体—关系—图模式、知识抽取与质量控制方法论 |
| Reimers2019SentenceBERT | 语义表示 | 支持利用语义向量和余弦相似度进行别名/同义实体对齐 |
| Han2022KnowledgeEnhancedGNN | 工业过程知识增强GNN | 支持将过程知识约束融入GNN以减少不符合领域知识的冗余关系 |

## Current computed v1 KG
- 107 unique semantic nodes
- 114 evidence triples
- Entity nodes: 23 device, 39 parameter, 24 anomaly, 15 fault, 6 action
- Main relations: 归属36，导致24，介质/能量连接15，发生于10，关联监测8，介质流向7，处置为6，影响4，表现为4
- File: `knowledge_graph/turbine_kg_evidence_triples_v1.csv`

## Representative evidence chains
1. HP级间汽封缺陷 → 蒸汽泄漏 → 汽耗率升高
2. LP轴端汽封缺陷 → 空气漏入凝汽器侧 → 真空下降 → 背压升高
3. HP-IP中间汽封泄漏 → IP效率下降；关联热再温度、IP排汽温度
4. 固体颗粒侵蚀 → 首级喷嘴面积增大 → 关键压力分布异常
5. 叶片表面沉积 → 通流面积减小 → 效率/出力下降
6. 凝汽器水侧污垢 → 换热能力下降 → 背压变化 → LP有效焓降下降
7. 给水加热器内部泄漏 → 回热性能下降 → 机组热效率下降
8. 高压调节阀死区 → 阀位对指令不敏感 → 蒸汽流量/压力振荡

## Gaps to fill later
- 将用户全系统DCS总点表完整映射到领域参数概念；当前只建立主要参数概念和VHP相关接口。
- 接入具体机组运行规程、缺陷记录、检修记录后扩充处置动作和本机组故障实体。
- 对B级边进行专家/现场资料二次确认。
- 在第四章根据实际进入KG-GCN的状态节点提取最终参数邻接子图。
