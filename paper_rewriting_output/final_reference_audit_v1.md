# 最终参考文献与引用审计 v1

> 目标：对当前 `main.tex` 实际引用的主稿进行定稿级参考文献核验。此文件记录“引用键存在性、核心元数据、正文可支持范围、待由最终编译器兜底的问题”。

## 1. 当前BibTeX库

主稿通过 `styles/yang_haodong_thesis.tex` 加载：

- `references/publication.bib`
- `references/methods.bib`
- `references/local_theses.bib`
- `references/kg_sources.bib`

参考文献样式：`gbt7714-numerical`。

## 2. 方法类关键引用

已确认以下key存在于 `methods.bib`：

- `Cho2014GRU`：GRU原始论文
- `Bai2018TCN`：TCN基线
- `Liu2024iTransformer`：iTransformer
- `Ba2016LayerNorm`：LayerNorm
- `Srivastava2014Dropout`：Dropout
- `Loshchilov2019AdamW`：AdamW
- `Kipf2017GCN`：GCN
- `Hou2022GraphMAE`：masked graph autoencoder背景

## 3. 汽轮机状态监测与故障诊断关键引用

已确认以下key存在于 `publication.bib`：

- `GonzalezSalazar2018Flexibility`
- `Yu2020HybridDigitalTwin`
- `Zhang2018AllCondition`
- `Banaszkiewicz2016ThermalStress`
- `Kubiak2002TurbineDegradation`
- `Jiang2014DataReconciliation`
- `Dettori2017MonitoringModels`
- `Zaleta2004ThermoCharacterization`
- `Madrigal2017BoilerTurbineFDI`
- `Cao2016LastStageNormalValue`
- `Li2021LSTMStageWarning`
- `Chen2023AdaptiveTransfer`
- `Ko2022DynamicThresholdAE`
- `Zhai2024CAEWarning`
- `Xu2025LSTMVAE`
- `Salahshoor2010SVMANFIS`
- `ZhangHu2021FaultPropagationGNN`
- `Jia2023TopologyGuidedGraph`
- `Han2022KnowledgeEnhancedGNN`

## 4. KG与故障机理关键引用

已确认以下key存在于 `kg_sources.bib`：

- `Hogan2021KnowledgeGraphs`
- `Reimers2019SentenceBERT`
- `ZhangHu2021ControlValveGraph`
- `Xu2011HPIPLeakage`
- `Kubiak2002ScaleDeposit`
- `Li2016CondenserFouling`
- `Medica2018CondenserCondition`
- `Barszcz2011FeedwaterHeater`
- `Heo2012FWHLeakage`
- `ASME2023TDP1`
- `NRC1997IN9150S1`
- `NRC1982IN8222`

其中正文证据边界已收紧：

1. `Kubiak2002ScaleDeposit`研究对象为地热蒸汽轮机，只作为跨型式蒸汽轮机沉积机理参考，不写成本机或燃煤机组现场直接证据；
2. `Xu2011HPIPLeakage`只支持HP--IP泄漏与IP效率、热再温度、IP排汽温度等关系，不外推成本文VHP具体DCS点固定方向；
3. 通用回热系统资料不用于证明本文机组未核实的具体“抽汽级次--单台加热器”关系；
4. LLM输出仅作为候选知识，最终KG关系需有机组资料或外部来源证据。

## 5. 本地学位论文引用

`local_theses.bib` 已确认：

- `LiPing2018SteamTurbine`
- `YangHaodong2026CFB`

杨昊东论文仅作为章节组织、信息展开和版式母版，不作为本文实验结果来源。

## 6. 第一章官方统计与政策来源

已通过官方网页核对正文所使用的三组公共统计/政策来源：

- `NEA2026PowerStatistics`：国家能源局2025年全国电力统计数据；
- `NBS2026EnergyTransition`：国家统计局“十四五”能源转型成就报告；
- `NDRC2025CoalPowerUpgrade`：发改能源〔2025〕363号《新一代煤电升级专项行动实施方案（2025--2027年）》。

注意：`NBS2026EnergyTransition` 当前BibTeX `note` 写为 `2026-06-04`，官方页面显示日期需在最终元数据轮再次核对；该1天级日期差异不影响正文统计值，但最终PDF前应统一。

## 7. 当前主稿核心cite-key存在性

### Ch1
已核对可见核心key：
`NEA2026PowerStatistics`、`NBS2026EnergyTransition`、`NDRC2025CoalPowerUpgrade`、`GonzalezSalazar2018Flexibility`、`Yu2020HybridDigitalTwin`、`Zhang2018AllCondition`、`Banaszkiewicz2016ThermalStress`、`Kubiak2002TurbineDegradation`、`Jiang2014DataReconciliation`、`Dettori2017MonitoringModels`、`Zaleta2004ThermoCharacterization` 等，均已在BibTeX库中定位。

### Ch2
已核对可见核心key：
`Cao2016LastStageNormalValue`、`Dettori2017MonitoringModels`、`Chen2023AdaptiveTransfer`、`Jiang2014DataReconciliation`、`Ko2022DynamicThresholdAE`、`Zhai2024CAEWarning`、`Xu2025LSTMVAE`、`LiPing2018SteamTurbine`、`Zhang2018AllCondition`、`Cho2014GRU`、`Bai2018TCN`、`Liu2024iTransformer`，均已定位。

### Ch3 v8
`Hogan2021KnowledgeGraphs`、`Kubiak2002TurbineDegradation`、`Kubiak2002ScaleDeposit`、`ZhangHu2021FaultPropagationGNN`、`Xu2011HPIPLeakage`、`Li2016CondenserFouling`、`Medica2018CondenserCondition`、`Heo2012FWHLeakage`、`Barszcz2011FeedwaterHeater`、`Reimers2019SentenceBERT`，均已定位。

### Ch4 v7
核心方法引用 `Kipf2017GCN` 已定位。

### Ch5 v8
主实验数据来自本仓库冻结CSV和实际训练输出，正文不以外部文献替代实验结果来源。

## 8. 最终编译器兜底检查

当前PR编译workflow会在完整BibTeX运行后自动检查：

- undefined citation；
- undefined reference；
- BibTeX database entry缺失。

若完整编译出现新的未定义key，以编译日志为最终判据并立即修复；源码层审计不替代编译器的一致性检查。

## 9. 当前结论

- 核心方法、汽轮机机理、KG及官方统计引用均已在四个BibTeX库中找到对应条目；
- 未发现当前Ch3/Ch4核心方法段落引用不存在的问题；
- 主要剩余风险为第一章长综述中较低频引用key及个别元数据细节，将由最终完整编译日志与第二轮元数据核验共同清零。
