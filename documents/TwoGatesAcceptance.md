# 用 two-gates 定义浮点变换的可接受性

更新日期：2026-10-03。

状态：**已实现 specification、依赖报告、局部 Triton 原子关系、Python two-gates、GPU 实测与 CPU 重放，以及主目录和补充目录结果的 Lean 规则导出。** 两个目录共用局部 golden ULP 的平均偏差预算和最大绝对误差比；当前结果见各目录的 `report/summary.md`。

当前局部 ULP bias 与绝对误差比协议使用固定种子 20261003，在 H200 上对独立 replicate 采样，并经独立 CPU 回放。已发布报告绑定其源代码哈希和聚合协议；回放与导出拒绝哈希不匹配的报告。

交付目标为完整浮点支持与接受流程，原语架构和统一完成条件见 [FloatingPointPrimitives.md](./FloatingPointPrimitives.md)。加法重排等小例子用于核对语义，不构成缩减后的交付版本。

准入表只包含固定规模的局部表达式关系。Softmax、LayerNorm、SwiGLU、归约、scan 和 dot 等完整变换需由 Lean 推导，不能将整算法的测试通过变成原子假设。

候选规则与逐配置接受结果的记录方式见 [FloatingPointRewriteRules.md](./FloatingPointRewriteRules.md)。未运行两门的规则不得预填 PASS；严格不等反例与统计拒绝分别记录。

## 1. two-gates 的角色：为原子假设提供准入依据

对某条原子变换的参考片段 \(P\) 和候选片段 \(Q\)，先运行带配置和证据的**有向数值检查**：

\[
P \xrightarrow[C,E]{\mathrm{accept}} Q .
\]

它表达：在契约 \(C\) 指定的输入分布、浮点执行配置和接受策略下，证据 \(E\) 支持将这条原子关系加入假设表。Lean 随后在这些假设下推导实现等价性；正确性则独立地在实数语义中证明 implementation = 数学公式。

正确性证明与原子检验分工如下：

1. **独立的正确性证明**：在明确的实数输入域 \(\Omega\) 上，两个模型具有相同的数学结果。
2. **Bias gate**：检查候选相对参考是否出现达到报警条件的方向性偏差。
3. **Vars gate**：检查候选相对高精度 oracle 的误差是否相对参考异常放大。

两道数值门不能替代第一项。例如维度接线错误即使在某批探针上通过，也不能作为已证明的代数变换；证明暂未完成则记为未完成，不能据此声称两个程序不等价。对数学函数本身不同的近似替换，必须提供显式算法近似契约，不能直接归为代数等价。

`vars gate` 保留报告的名称，但其实际统计量是**峰值误差的放大率及其尾部 return level**，并非直接估计 \(\operatorname{Var}(Q-P)\)。两门使用不同参照系，不能将其公式直接称为同一个随机变量的严格偏差—方差恒等分解。

## 2. 契约与执行语义

每次判定至少固定以下内容；这些信息发生变化时，原结果不能自动沿用：

| 契约项 | 要记录的内容 |
|---|---|
| 实现身份 | 参考、候选、数学规格、checker 的版本或摘要，以及模型与被测实现的对应关系 |
| 执行配置 | 输入/输出 dtype、每步运算和累加精度、舍入模式、cast 位置、FMA/归约约定、特殊值策略；实际执行实验还需编译器、后端和设备信息 |
| 输入域与分布 | shape、尺度、探针族、固定权重及其身份；随机输入须落在代数定理的适用域内 |
| 分桶与尺度 | channel/head/expert 等桶定义；ULP 取值方向、尺度及跨输出聚合方式 |
| Oracle | 高精度实现、计算精度、输入提升方式及其数值可靠性依据 |
| 统计协议 | replicate 数与独立性假设、随机种子、阈值、尾部拟合与 bootstrap 配置、停止规则、多重比较处理 |
| 接受策略 | PASS/WARN/FAIL 的处理，特别是是否允许带 WARN 的替换 |

参考与候选消费同一份已构造的输入和权重。Oracle 也应从同一份输入提升精度，避免将输入量化差异混入 kernel 差异。fp64 是高精度参照的实现选择，不自动等于精确实数真值；抵消等情形需要检查其误差是否影响判决。

报告的 bias 动机使用训练累积视界，而 vars 实验使用一个 optimizer step 内的调用视界。本设计分别记录 \(T_{\mathrm{bias}}\) 与 \(T_{\mathrm{tail}}\)，并记录单位与采样单元；不能因两者都写成 \(T\) 而直接互换。固定权重下的结果只覆盖该权重配置。

当前 VeriTile 的普通浮点标签算术仍使用实数运算，抽象舍入主要位于 cast/store。本契约的实际浮点计算由 Python 调 Triton/GPU 完成，包括 bf16 输入配 fp32 累加等混合精度情况；不要求先实现完整 Lean IEEE 执行器。现有 `RoundingModel` 的幂等性等性质不足以唯一确定 bf16/fp32 的执行行为。

### 高斯探针与 shape 的处理

用户已确认：**规则准入只按用户指定的输入分布判断，不自动增加其他分布或附加通过条件。** 例如全部操作数独立服从 \(\mathcal N(1,\sigma^2)\)，就只检查这一配置；`sigma`、shape 和统计预算在运行前确定。“正态分布下接受”不表示对所有均值、尺度和协方差都成立。

对操作数角色 \(j\)，生成 \(Z_j\sim\mathcal N(\mu_j,\sigma_j^2)\)，再按输入 dtype 量化为 \(X_j\)。独立高斯探针明确各操作数独立，记录角色各自的均值与尺度；涉及相关输入时契约必须记录联合生成方式，不能自动沿用独立探针的结论。实际检查对象是量化后的分布。定义域限制、超出浮点有限范围的样本如何处理、是否截断或拒绝采样，也属于生成器定义，不允许静默丢弃异常样本。

不再默认运行标准高斯、角色不对称均值、角色不对称尺度等额外探针组。用户以后主动选择新的分布时，将它作为新的配置独立生成规则集，不能将不同均值或正负偏差的样本混在一起计算平均值。高斯输入也不意味着舍入误差服从高斯分布。

特别是加法重排：记 \(e(a,b,c)=Q(a,b,c)-P(a,b,c)\)。在两边使用相同加法精度、舍入规则且运算有限时，加法的交换性给出：

\[
P(c,b,a)=Q(a,b,c),\qquad e(c,b,a)=-e(a,b,c).
\]

若输入联合分布在交换 \(a,c\) 后不变，且期望存在，则 \(\mathbb E[e]=0\)。全部操作数使用同一个平移高斯也保留这个交换对称性。这是指定分布下的结果解释，不触发额外探针或新的拒绝条件；vars gate 仍按原协议检查误差放大。改变角色的均值或尺度属于用户显式选择的新配置，系统不自动进行这种改变。

Shape 按是否改变计算过程处理：

| 运算范围 | 处理方式 |
|---|---|
| 固定元数标量关系，如三个数的加法重排 | 检查一个三元组；\(R\) 个独立三元组是 \(R\) 个 replicate，该关系本身没有大张量 shape 参数 |
| 对每个元素独立应用同一关系 | 在证明逐元素执行和算术配置一致后复用标量分析；若门针对整张量峰值或多桶结果，仍需按元素数量重新处理统计量与比较次数 |
| Sum、Softmax、LayerNorm 等归约 | 保留归约长度和归约顺序；按长度分别报告两门结果 |
| Matmul、attention 等多轴计算 | 保留影响累加、归约或执行路径的维度；还需记录输出尺寸对峰值误差统计量的影响 |

对归约长度，建议使用几何增长的代表点，并覆盖实现中 tile/block 切换附近的合法尺寸，例如 \(B-1,B,B+1\)。这些是经验覆盖点，不构成区间内所有长度的证明；最大测试长度通过也不能自动推出更短长度通过。抽象接口可以接收任意合法 shape，数值接受证据仍绑定实际检查的配置。

加法结合律是语义回归案例之一：bf16/fp32 输入与明确的运算/累加精度，按用户配置的生成器产出两门结果。输入分布默认定义原子规则的测试环境，Lean 在生成的 `R` 下推导形式等价性，不额外要求证明中间变量的分布。只有当实验声称代表 kernel 内的实际操作数时，才需记录那个使用位置的联合采样依据；整 kernel 的统计结果也需独立实验。

### 已确认：允许按具体 shape 验证

用户已确认可以对具体 shape 给出数值接受结果，作为完整设计的配置范围。验证实例需固定 shape，但 Triton 源码仍可保留尺寸参数，无需把输入尺寸写死为字面量。需要区分：

- **输入张量 shape**：由调用方提供，例如向量长度 `N=4096`，以及对应 stride/layout。
- **kernel 内部 tile shape**：常通过 `BLOCK_SIZE: tl.constexpr` 等编译期参数确定；它与输入总长度不同。
- **启动及编译配置**：各实现的 grid、block 参数、`num_warps`、精度选项等。参考和候选可以采用不同配置，但两边都需记录。

仓库的 [vector_addition.py](../bench/tritonbench_g/vector_addition/vector_addition.py) 使用 `n_elements` 表示输入长度，以 `BLOCK_SIZE: tl.constexpr` 定义每个 program 处理的 tile，并由 wrapper 根据输入长度计算 grid。这与 [Triton 官方向量加法示例](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html) 的处理一致。当前 [SoftmaxTriton1.lean](../bench/tritonbench_g/softmax_triton1/SoftmaxTriton1.lean) 也将 `n_cols` 和 `BLOCK_SIZE` 分别保留为参数。

建议将数值检查绑定到“实现对 + 具体 shape/stride + dtype/运算精度 + 两边的执行配置 + 高斯探针配置”。同一份参数化源码可以产生多个检查实例；新 shape 尚未检查时标为未验证，不能自动继承旧实例的 PASS。已有参数化代数定理可按其前提实例化复用；数值证据的范围单独记录。使用 autotune 时，还需固定或记录本次实际选中的配置。

## 3. Bias gate：局部 ULP 单位的平均偏差预算

对每个元素使用同一 golden 尺度 \(u_{r,i}=\operatorname{ULP}_d(O(x_r)_i)\)，先归一化，再对本次 replicate 的全部同分布标量实例取一个均值。当前原子实验的各位置执行相同关系、使用相同分布，列位置没有独立 channel 语义。一个统计样本是整个 replicate 的均值，标准误中的 R 仍是 replicate 数：

\[
\Delta_r=\operatorname{mean}_i\frac{Q(x_r)_i-P(x_r)_i}{u_{r,i}},
\qquad SE=\frac{s}{\sqrt R},\qquad z=\frac{|\bar\Delta|}{SE}.
\]

当前 profile 固定 \(\tau=0.05\) local ULP、\(c=5\)，构造工程判定区间 \([\bar\Delta-cSE,\bar\Delta+cSE]\)。定义：

\[
B=|\bar\Delta|+cSE,\qquad
L=\max(0,|\bar\Delta|-cSE).
\]

- **PASS**：\(B\le\tau\)，即区间完全位于容差内。
- **FAIL**：\(L>\tau\)，即区间完全位于容差外；无效或非有限统计也 FAIL。
- **INCONCLUSIVE**：没有 FAIL，但区间跨过容差边界。数据不足以确认满足预算，不能准入。

该聚合只适用于当前同分布、同计算的标量原子实例；不同分布或语义的 channel/head/expert 需显式定义分组，不能直接合并。观测保存为 `[R, 1]`；bundle 和 checker 版本绑定这一聚合方式，旧的按列观测不能作为当前协议的正式证据。

\(z\) 继续报告，但不决定接受。极小的恒定偏差可以有无穷 z 而满足预算；反过来，均值接近零但标准误很大时，不能仅凭小 z 接受。零样本方差时按观测区间退化为一点计算；这不证明总体方差也为零。\(\tau\) 约束平均有符号偏差，不约束单个输出误差，也不替代 vars gate。

五个标准误是当前明确选择的工程协议，**不是已经校准的多桶、有限样本或自适应停止置信保证**。固定预算后用独立种子检验实现和结论；这也不自动证明协议的覆盖率。需要严格概率保证时，应另行固定检验族和采样规则并论证区间方法。不能逐条修改 tau 直到某个关系通过。

尺度按元素计算，不使用输出峰值给其他元素放宽容差。报告中的 B 是跨 replicate 均值的偏差上界；输入分布、精度、cast 位置和 oracle 仍属于不可省略的契约。阈值不是任何 bf16/fp32 算子的通用误差保证，也不自动约束误差在整个训练过程中的传播。

## 4. Vars gate：相对 oracle 的误差放大与尾部

对同一个 replicate，分别取参考与候选相对 oracle 的最大绝对误差：

\[
E^Q_r=\max_i|Q(x_r)_i-O(x_r)_i|,\qquad
E^P_r=\max_i|P(x_r)_i-O(x_r)_i| .
\]

由接受不等式 \(E^Q_r\le K E^P_r\) 得到无量纲误差比：

\[
\hat K_r=
\begin{cases}
0,& E^Q_r=0,\\
E^Q_r/E^P_r,& E^Q_r>0,\ E^P_r>0,\\
+\infty,& E^Q_r>0,\ E^P_r=0.
\end{cases}
\]

这里不除以逐元素 ULP，也没有加性容差。两侧的峰值可以来自不同元素。
该指标采用 [FlashAttention 的最大绝对误差比较](https://github.com/Dao-AILab/flash-attention/blob/main/tests/test_flash_attn.py)：FA 测试要求候选的峰值误差不超过基线的两倍；本项目将 U 的通过阈值设为 10。逐元素除以不同 ULP 后再取最大值会改变这个比较，因此只在 bias gate 使用局部 ULP。

Bias 衡量平均多少 ULP 的有符号偏移；vars 衡量相对基线的峰值误差放大，两者无需使用相同单位。当前协议的 bias gate 和下面的尾部外推都是额外要求，不等同于 FA 对已测样本直接应用的两倍判据。

报告用 POT/GPD 估计 \(\hat K\) 在视界 \(T_{\mathrm{tail}}\) 下的 return level，再构造上置信界估计 \(U\)。设阈值为 \(u\)、超阈概率为 \(\zeta\)、GPD 参数为 \((\xi,\sigma)\)，其尾部模型给出：

\[
r_T=u+\frac{\sigma}{\xi}\bigl[(T_{\mathrm{tail}}\zeta)^\xi-1\bigr],
\qquad
\xi=0:\quad r_T=u+\sigma\ln(T_{\mathrm{tail}}\zeta).
\]

该公式用于目标分位落在拟合尾部的情形。报告取 \(U=\hat r_T+3\,\widehat{\mathrm{se}}\)，标准误差由 bootstrap 估计。判决为：

| 条件 | Vars gate |
|---|---|
| \(U\le K_{\mathrm{warn}}\) | PASS |
| \(K_{\mathrm{warn}}<U\le K_{\mathrm{fail}}\) | WARN |
| \(U>K_{\mathrm{fail}}\)，或合法误差计算产生 \(\hat K_r=+\infty\) | FAIL |

当前配置使用 \(K_{\mathrm{warn}}=10\)、\(K_{\mathrm{fail}}=100\)：\(U\le10\) 通过，\(10<U\le100\) 警告，\(U>100\) 失败。当前协议在尾部不足时回退到经验最大值并显式标记；非有限最终统计量 FAIL，协议未完成不能准入。全零放大率等退化数据也走经验分支，不得伪造拟合或置信界。

矩和误差比统计在有效有限数值上定义。浮点执行层完整表示 NaN、无穷、signed-zero 和 subnormal；统计前按输出契约处理特殊值。在要求有限输出的域内，候选产生非有限输出触发 FAIL；非有限 reference/oracle 误差同样 FAIL；执行错误或输入定义域事件单独报告，不能静默过滤相关样本。

## 5. 哪些关系允许使用

原子执行成功、证据匹配当前契约且协议完整时，合并准入判决如下：

| Bias gate | Vars gate | 对变换的处理 |
|---|---|---|
| PASS | PASS | ACCEPT：满足本次统计接受协议 |
| PASS | WARN | 默认 pass_only 下 WARN_NOT_ACCEPTED；显式 allow_warn 可记 ACCEPT_WITH_WARNING |
| INCONCLUSIVE | PASS/WARN | INCONCLUSIVE：不能由 allow_warn 提升为接受 |
| 任意 | FAIL | REJECT：该契约下拒绝替换 |
| FAIL | 任意 | REJECT：该契约下拒绝替换 |
| 无 FAIL，但任一门无有效判定 | — | INCONCLUSIVE：尚无接受依据 |

当前 bias gate 使用 PASS/FAIL/INCONCLUSIVE；vars gate 保留 WARN。默认 pass_only 只接受双 PASS。显式 allow_warn 只放行 vars WARN，不能放行 bias INCONCLUSIVE。采用哪种策略属于契约内容，不能把 WARN 静默改成 PASS。原子准入不能替代两端各自的实数正确性证明。缺少实现对应或验证／重放证据时，可以报告实验数据，但不能把该条目当作已经验证的可用假设。

概念上，原子准入与实现等价性分成两层：

    Admitted(atom, evidence) = 验证/重放完成 ∧ 配置身份匹配 ∧ artifact存在 ∧ 双门满足策略
    assumptions ⊢ implementation₁ ≈ implementation₂ = 在已准入的原子假设下进行形式推导

[Spec.lean](../VeriTile/Spec.lean) 中 `AcceptedAtom` 表达第一层，`lhs ≡[R] rhs` 表达第二层（`VeriTile.Spec` scope，底层为 `Spec.FloatingPoint`）。实验配置与结果保存在 `R` 的规则表里，不进入公开规格的独立参数；`#print_spec` 展开这些记录及准入前提。Lean 不证明一次采样推出普遍的 IEEE 等式；我们明确把检验通过的关系作为优化理论的假设。数学正确性不再是浮点凭证结构中的一个字段，而是独立的实数规格。

### 结合律示例

\[
P_d(a,b,c)=\operatorname{fl}_d(\operatorname{fl}_d(a+b)+c),
\qquad
Q_d(a,b,c)=\operatorname{fl}_d(a+\operatorname{fl}_d(b+c)).
\]

去除舍入的实数模型可由结合律证明相等。具体浮点结果一般不同；例如 round-to-nearest, ties-to-even 下，fp32 的 \((a,b,c)=(2^{24},1,-2^{24})\) 给出 \(P=0,Q=1\)，bf16 的 \((2^8,1,-2^8)\) 同样如此。

随后按指定分布测试这对实现：双 PASS 可接受，有 FAIL 则拒绝，WARN 保留条件。某个配置接受不代表“浮点加法满足结合律”；上述单点反例也不能单独决定整个统计协议的判决。

原始数值检验通常**不对称**：vars gate 以参考误差作分母。该事实与形式理论必须区分：被准入的原子关系作为等价假设后，`Derivation` 可以对称、传递和放入共同语句前后文。这些是显式选择的证明规则，不宣称反向检验或组合后的整 kernel 检验已通过。假设仍需显示其配置和操作数分布适用范围；上下文推导本身不证明概率分布传播。

## 6. 接入 VeriTile 的边界

- 统一保留 `specification`：`Spec.Real` 用于数学正确性；`lhs ≡[R] rhs` 用于基于原子假设的实现等价性。
- `AtomicRule` / `RuleEntry` 绑定原子两侧片段、配置、实例身份和实验记录。`AcceptedAtom` 要求验证与重放前提、身份匹配、artifact 和门策略；当前没有跳过这些条件的准入构造器。
- `AcceptedAssumptions` 汇总可用的假设；`Derivation` 检查引用、组合、对称和共同语句上下文；`ProgramSyntax` 检查程序签名及语句序列。
- `#print_spec` 显示实现、声明的原子假设范围及逐项证据、模型前提和依赖公理。声明范围可能包含未使用条目，打印不关闭验证义务。
- [实验入口](../experiments/floating_point/README.md) 提供 GPU 原语对、源代码／PTX／后端身份、配置、种子、逐 replicate 统计量、拟合诊断和 CPU 导入重放。主目录的当前接受项导出为 Lean 规则数据，fp32 ADD-COMMUTE 已绑定到具体加法例子；JSON 导入不证明 `EvidenceValidated`，其他原子的语法绑定仍须逐项完成。
- 不把原子假设注册成具体 IEEE 函数的 Lean 等式或全局代数实例。已有实数正确性和抽象舍入证明不能冒充原子数值记录。
- 整 kernel 数值复查属于可额外报告的实验，不是这套形式等价性的定义。实际 GPU 结论仍需对应后端证据；软件 profile 结果只覆盖所选软件语义。

## 7. 报告中需修正或验证的保证

采用两门的职责、统计量与判决结构，不意味着报告中的强保证已成立。后续理论与实验工作至少需要处理：

1. **偏差预算与区间覆盖。** PASS 表示平均偏差的工程区间落在 ±tau 内，不表示严格无偏或逐元素小误差。tau 的应用合理性、区间覆盖率和所需样本量需要分别验证；均值接近零但区间宽时报告 INCONCLUSIVE。
2. **五西格玛与自适应停止。** 用样本标准差构造的统计量依赖分布和样本量；渐近正态近似不能直接给出有限样本的严格误报界。反复查看置信区间后停止，也不能自动沿用固定样本量的覆盖率。需固定协议或采用经过论证的序贯方法；置信序列是可研究的路径之一：[原始研究](https://arxiv.org/abs/2301.09573)。
3. **Return level 的含义。** 理想连续模型下，\(r_T\) 对应单次超越概率约 \(1/T\)，不是 \(T\) 次最大值的期望，也不是保证不被超越的最坏值。独立抽样时至少超越一次的概率为 \(1-(1-1/T)^T\)，随 \(T\) 增大趋于约 \(0.632\)。必须区分 return level 的估计置信度和未来运行的超越风险。[NIST 的 return value 定义](https://www.nist.gov/programs-projects/maps-non-hurricane-non-tornadic-extreme-wind-speeds-contiguous-united-states)。
4. **尾部模型和参数截断。** POT/GPD 拟合有尾部近似及采样假设，不能直接宣称有限样本“分布无关”。报告把正 \(\hat\xi\) 截到零，需要额外依据；当参考误差很小时，误差比可能出现长尾。即使极限尾部有界，也不足以保证有限阈值下的指数拟合保守。截断前后的诊断与覆盖率需要验证。
5. **从局部误差到训练危害。** \(T\mu\) 与 \(\sqrt T\sigma\) 的比较需要相关性和传播假设。一般训练扰动还经过随时间变化的 Jacobian；存在负曲率不能单独推出整个乘积具有正 Lyapunov 指数。forward 门的通过不能直接推出 backward、训练轨迹或训练质量保证。
6. **计量与复现。** ULP 在 binade 边界具有方向差异，例如 bf16 在 1 上方的相邻间隔为 \(2^{-7}\)，下方为 \(2^{-8}\)。必须固定 nextafter 方向及聚合规则。改变 bias 的逐元素尺度或 vars 的误差单位会改变所检验的量；只保存局部 ULP 峰值的观测无法恢复绝对误差峰值，必须重新采样，不能直接充当当前协议的结果。

### 当前验收协议

双门验收采用自适应停止规则：默认最少 4096 次、每批 512 次、预算上限 50000 次（完整末批可到 50176）；每批检查幅度置信带，达到最小预算后稳定或回退则停止。PWM 保留原始 shape 供诊断，计算 return level 时截断到非正；1000 次固定种子 bootstrap，alpha=1.35e-3。尾部不可拟合时按经验最大 K 判 PASS/WARN/FAIL，明确标记 `empirical_fallback`，不能把它称为尾部置信保证。

Bias 对每个 replicate 的全部同分布标量实例取一个均值，一维和二维布局使用相同聚合。默认 tau=0.05 local ULP，SE 倍数为 5；z 只作诊断，接受要求跨 replicate 均值的偏差上界不超过 tau。每个元素单独计算 ULP：取 golden 的绝对值，转到输出 dtype，再取向正无穷的 nextafter 间距；最大有限值取朝零间距，零和 subnormal 使用最小间距。golden 转换溢出产生无效尺度并使检查失败。仅 bias 先用该尺度归一化，再计算整次均值；vars 保存绝对 oracle 误差的峰值并计算 Ec/Er，不使用 ULP 或加性容差。不跨元素或 replicate 取最大 ULP 作为 bias 预算。CPU 回放检查 delta 的 `[R, 1]` 形状、实际采样数及所有自适应检查点。输入分布由 VeriTile 的 profile 指定，不引入新的正值或条件采样。具体参数与运行说明见实验目录，模型限制见第 7 节。

## 8. 下一步

- [x] 明确实数正确性与基于 two-gates 准入原子假设的浮点实现等价性。
- [x] 区分两门的参照系、三档判决、证据不足和有向关系的使用范围。
- [x] 记录高斯探针及 shape 处理建议，指出加法重排的交换对称盲点；具体配置尚未冻结。
- [x] 按完整交付目标补充原语设计，明确值表示、FMA、混合精度、归约、dot/MMA、特殊值和后端连接。
- [ ] 完成 bf16/fp32 支持范围内的原语、执行计划与契约；加法重排等案例提供逐位不同见证及对应代数证明。
- [x] 实现两门检查及 CPU 重放，核对统计实现、退化情形与停止规则。
- [ ] 校准阈值并验证功效、尾部模型与覆盖率。
- [x] 实现原子规则、准入条件、形式推导、两种 specification 和逐原子依赖报告；外部验证义务显式保留。
- [x] 将原子准入连接到 GPU 执行、证据重放与统计 checker。
- [ ] 继续完善其他原子的语法绑定、适用范围检查及执行层覆盖。
