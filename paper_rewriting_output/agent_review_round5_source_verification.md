# 多Agent审稿 Round 5：核心文献与知识关系真实性核验

## 审查目的
检查第三章核心BibTeX文献是否真实存在，以及正文中“文献→知识关系”的表述是否超出原始来源支持范围。外部检索只作为核验工具，最终知识入图仍按本文证据规则执行。

## 1. Zhang & Hu 2021 — 汽轮机故障传播GNN
文献：Yi-Jing Zhang, Li-Sheng Hu, *Fault Propagation Inference Based on a Graph Neural Network for Steam Turbine Systems*, Energies 14(2), 309, 2021, DOI 10.3390/en14020309。

核验结果：真实存在，BibTeX key `ZhangHu2021FaultPropagationGNN`可用。
直接支持：
- HP/IP/LP级间汽封和轴端汽封缺陷可导致相应蒸汽泄漏；
- LP轴端汽封的功能是防止大气空气漏入，空气漏入会造成真空损失；
- 汽封缺陷→蒸汽泄漏→汽耗增加的层次化故障传播关系；
- 依据汽轮机系统机理构建process graph，并用GNN进行故障传播推断。
结论：第三章当前对“汽封缺陷→蒸汽泄漏”“LP密封异常→空气漏入/真空恶化”的表述有直接来源支持。

## 2. Xu et al. 2011 — HP–IP泄漏
文献：Jian-qun Xu, Gang Li, Ling Li, Ke-yi Zhou, Yong-feng Shi, *Thermodynamic model of HP–IP leakage and IP turbine efficiency*, Applied Thermal Engineering 31(2–3), 311–318, 2011, DOI 10.1016/j.applthermaleng.2010.09.011。

核验结果：真实存在。仓库BibTeX当前作者字段使用“and others”，最终参考文献定稿时可补全Yong-feng Shi。
直接支持：
- HP–IP midspan packing (N2 packing)泄漏；
- hot reheat temperature和IP turbine exhaust temperature是影响N2泄漏计算精度的关键测量参数；
- N2泄漏是影响IP效率的重要因素。
结论：正文可表述“HP–IP泄漏与热再温度、IP排汽温度及IP效率具有诊断/计算关联”，但不应扩展为本文VHP具体测点的固定增减方向。

## 3. Kubiak et al. 2002 — 通流部件退化
文献：J. Kubiak, A. García-Gutiérrez, G. Urquiza, *The diagnosis of turbine component degradation—case histories*, Applied Thermal Engineering 22(17), 1955–1963, 2002, DOI 10.1016/S1359-4311(02)00118-7。

核验结果：真实存在，为蒸汽/燃气轮机运行诊断与检修验证案例。
直接支持：
- 常见蒸汽轮机退化包括叶片表面粗糙、SPE、汽封磨损和叶片沉积；
- SPE会增大首级喷嘴面积；
- 文中利用关键压力进行在线退化识别，并由检修测量验证。
结论：可用于支撑通流侵蚀/SPE及关键压力分布变化的通用机理。

## 4. Kubiak & Urquiza-Beltrán 2002 — 叶片沉积
文献：*Simulation of the effect of scale deposition on a geothermal turbine*, Geothermics 31(5), 545–562, DOI 10.1016/S0375-6505(02)00013-5。

核验结果：真实存在，但对象是地热蒸汽轮机。
直接支持：
- 叶片/喷嘴沉积会减小通流面积；
- 降低出力和效率；
- 影响首级前后蒸汽压力。
适用边界：只能作为“蒸汽轮机沉积机理”的跨型式参考，不应表述为本文燃煤机组的现场直接证据。正文如需使用，应明确其通用机理属性。

## 5. Li et al. 2016 — 凝汽器污垢
文献：Jianlan Li et al., *On-line fouling monitoring model of condenser in coal-fired power plants*, Applied Thermal Engineering 104, 628–635, DOI 10.1016/j.applthermaleng.2016.04.131。

核验结果：真实存在，且对象为燃煤机组凝汽器。
直接支持：水侧污垢降低凝汽器热有效性/换热性能，可通过在线污垢模型监测。
结论：第三章“凝汽器水侧污垢→换热能力下降”属于直接支持关系。

## 6. Medica-Viola et al. 2018 — 凝汽器状态监测
文献：*Numerical model for on-condition monitoring of condenser in coal-fired power plants*, International Journal of Heat and Mass Transfer 117, 912–923, DOI 10.1016/j.ijheatmasstransfer.2017.10.047。

核验结果：真实存在，煤电凝汽器实测验证。
直接支持：凝结压力显著影响电厂效率；循环水温度、流量和污垢等因素影响凝汽器性能/凝结压力。
结论：可用于支撑冷端边界—凝结压力—机组效率之间的通用关系。

## 7. Heo & Lee 2012 / Barszcz & Czop 2011 — 给水加热器内漏
两篇文献及DOI均核验真实：
- Heo & Lee, Expert Systems with Applications 39(5), 5078–5086, DOI 10.1016/j.eswa.2011.11.031；
- Barszcz & Czop, Applied Thermal Engineering 31(8–9), 1357–1367, DOI 10.1016/j.applthermaleng.2010.12.012。

直接支持：
- 闭式给水加热器内部泄漏及其位置/泄漏率诊断；
- 内漏会损害回热循环效率；
- 基于换热功率、传热系数等过程指标可进行模型诊断；Barszcz文中明确以裂纹管泄漏作为慢性变化示例。
结论：可用于支撑“加热器内部泄漏→回热性能/换热过程异常”，但具体本文机组某测点固定变化方向仍需机组侧或更直接证据。

## 8. Zhang & Hu 2021 — 调节阀关系图
文献：*Relationship Prediction Based on Graph Model for Steam Turbine Control Valve*, Actuators 10(5), 91, DOI 10.3390/act10050091。
核验结果：真实存在。
直接支持：高压调节阀dead zone、dead zone导致系统振荡、利用物理变量图和图卷积进行关系预测。

## Round 5结论
Blocking = 0。
Major = 1：`Kubiak2002ScaleDeposit`为地热蒸汽轮机证据，正文必须按跨型式通用机理引用，不能作为本文燃煤机组直接现场证据。
Minor = 1：`Xu2011HPIPLeakage` BibTeX作者建议补全Yong-feng Shi。

下一轮建议：对第三章每条A1型故障关系建立“正文句子—文献原始支持—适用范围”对照表，并据此继续收紧措辞。