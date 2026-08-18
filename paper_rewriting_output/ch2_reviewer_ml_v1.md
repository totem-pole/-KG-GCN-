# 第二章 Reviewer 2：机器学习与状态监测方法审稿

## Overall assessment
第二章v3已经补入RNN、GRU、序列输入、标准化和BPTT等内容，较早期版本完整得多，但方法层仍存在“公式有了，为什么这样设计还不够充分”的问题。对于一篇完整学位论文，GRU部分应形成“基础RNN局限→GRU机制→多变量多输出回归→正则化与优化→本文真实结构”的完整方法链。

## Major Concerns

### R2-M1 — Blocking: Yes
**Axis:** GRU方法理论完整性  
**Affected:** 2.3  
**Concern:** 当前GRU理论仍主要围绕门控公式展开，对普通RNN为何需要门控结构、长期依赖和梯度传播困难与汽轮机动态过程之间的联系阐释不足。  
**Required revision:** 在GRU公式前增加RNN长期依赖问题的解释：递归乘积可能导致梯度衰减/放大，难以稳定保留较长时间范围内的信息；随后说明GRU通过更新门和重置门形成更直接的信息通道。不要进行过度数学证明，但要给出与汽轮机快/慢变量共存的对应解释。  
**Resolution test:** 读者能回答“为什么不是普通RNN，而是GRU”。

### R2-M2 — Blocking: No, Major
**Axis:** 多变量多输出建模  
**Affected:** 2.3.3 / 2.4  
**Concern:** 当前已定义 $X\in R^{L\times m}$ 和多输出，但没有充分说明共享GRU表示为什么适用于压力、蒸汽温度、抽汽接口和金属温度等多类输出。  
**Required revision:** 增加共享动态特征与多任务回归思路：同一组入口边界与操作量驱动多个状态变量，GRU隐状态提取共享工况/动态特征，线性输出层分别映射至各状态维度。明确这不是将27个状态简单拼接，而是共享时序表示后联合回归。

### R2-M3 — Blocking: No, Major
**Axis:** LayerNorm/Dropout/AdamW的算法说明  
**Affected:** 2.3.4–2.4.3  
**Concern:** LayerNorm、Dropout、AdamW目前更多以实现参数出现，论文解释不足。  
**Required revision:** 采用简洁算法说明：LayerNorm在单样本特征维度上归一化，适合循环网络且训练/测试行为一致；Dropout通过训练期随机失活抑制特征共适应；AdamW将权重衰减与自适应梯度更新解耦。给出原始文献引用，但不需要推导完整优化器公式。  
**Suggested citations:** Ba et al., Layer Normalization (2016); Srivastava et al., JMLR 2014; Loshchilov & Hutter, ICLR 2019.

### R2-M4 — Blocking: No, Major
**Axis:** 模型参数设置解释  
**Affected:** 2.4.3 / 2.5.4  
**Concern:** hidden size=32、window=24、dropout=0.05目前像工程配置记录，缺少最小必要说明。  
**Required revision:** 不要虚构“经过大量网格搜索证明最优”。可以写成：采用小型单层GRU是为了在设备级多模型部署场景下兼顾表达能力和计算规模；24点窗口用于提供短期动态历史，最终选择依据验证集；具体超参数见表。若没有完整消融结果，不声称这些值具有全局最优性。

### R2-M5 — Blocking: Yes
**Axis:** 对比实验公平性  
**Affected:** 对比模型说明/结果讨论  
**Concern:** ANN仅使用当前时刻输入，而GRU/TCN/iTransformer使用24点历史，不能表述为“所有模型在完全相同输入条件下公平比较”。  
**Required revision:** 明确ANN是“静态基线”，用于量化显式时间历史的价值；TCN和iTransformer-small才是主要时序架构对照。结果讨论中避免把GRU对ANN的优势完全归因于网络门控能力。

### R2-M6 — Blocking: No, Major
**Axis:** 27输出→24评价点  
**Affected:** 2.4/2.5  
**Concern:** 目前27→24信息出现多次，容易让算法主线被数据冻结过程打断。  
**Required revision:** 算法结构处只报告真实Linear(27)；在VHP工程算例“输出变量定义”中集中解释删除CT132A/CT151A/CT152A后形成24点评价/诊断口径，后文不再反复强调。

## Minor Comments
- R2-m1：损失函数应明确是在标准化输出空间上计算。
- R2-m2：说明StandardScaler仅用训练数据拟合，已经做得好，保留。
- R2-m3：BPTT只需解释参数共享和梯度沿时间展开回传，无需写成深度学习教材。
- R2-m4：结果中不要使用“GRU全面优于iTransformer”，应保留17/24最优及困难温度点分析。
- R2-m5：如果后续有条件，可增加24×16 flatten-MLP作为更严格静态基线，但不作为当前第二章阻塞项。

## Recommendation
Major revision。方法理论补齐后，第二章可以从“工程实现描述”提升为完整的动态状态建模章节。