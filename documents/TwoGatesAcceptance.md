# 用 two-gates 定义浮点变换的可接受性

更新日期：2026-10-01。

状态：**已实现 specification、依赖报告、26 条 Triton 候选对、Python two-gates 与结果重放；GPU 实测及 Lean 规则导入连接待完成。** 依据用户提供的 [tech-report-kernel-gates.md](/home/argustest/.codex/attachments/2fc372e4-f75b-4d94-8766-05f79473b646/tech-report-kernel-gates.md) 整理。该链接指向本次会话附件；附件报告中的实验结果尚未在 VeriTile 中复现，其理论保证也不作为已验证结论引用。

交付目标为完整浮点支持与接受流程，原语架构和统一完成条件见 [FloatingPointPrimitives.md](./FloatingPointPrimitives.md)。加法重排等小例子用于核对语义，不构成缩减后的交付版本。

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

## 3. Bias gate：候选相对参考的方向性偏差

按契约独立抽取 \(R\) 个输入 replicate \(x_r\)，固定共享权重。对桶 \(g\) 计算：

\[
\Delta_{r,g}
=\operatorname{mean}_{i\in g}\bigl(Q(x_r)_i-P(x_r)_i\bigr),
\qquad
z_g=\sqrt R\,\frac{\bar\Delta_g}{s_g},
\qquad
\hat\rho_g=\frac{|\bar\Delta_g|}{s_g}.
\]

统计单元是 replicate，不能将同一 replicate 内的输出元素计作独立样本。按照报告的判决规则，任一桶同时满足以下三项时，bias gate 返回 FAIL：

\[
|z_g|>z^\ast,\qquad
\hat\rho_g>s_{\min},\qquad
|\bar\Delta_g|>c\,u_g.
\]

若存在统计显著但未同时越过两个效应量地板的桶，且无 FAIL 桶，则返回 WARN；否则返回 PASS。报告使用 \(z^\ast=5\)、\(s_{\min}=10^{-2}\)、\(c=1\)，这里将它们作为待复现的起始配置，不能默认认为已针对所有 bf16/fp32 算子校准。

退化情形需要写入 checker 的版本化规则。建议采用以下计算约定：当 \(s_g=0,\bar\Delta_g=0\) 时，两个标准化统计量取零；当 \(s_g=0,\bar\Delta_g\ne0\) 时，其绝对值取无穷，仍由 ULP 地板区分 WARN/FAIL。这个约定只处理样本计算，不证明总体方差为零。ULP 尺度无法确定或样本量不足时返回 INCONCLUSIVE，不能让 NaN 比较意外变成 PASS。

探针分布和分桶决定检出能力。对称输入可能抵消乘性偏差；报告使用平移探针来暴露它。平移量与探针族需事先登记，选择多个探针族时一并计入比较范围。平移本身不能保证覆盖训练分布，也不能消除桶内所有抵消机制。

## 4. Vars gate：相对 oracle 的误差放大与尾部

设高精度 oracle 为 \(O\)。对相同的 replicate 计算：

\[
E^Q_r=\|Q(x_r)-O(x_r)\|_\infty,\qquad
E^P_r=\|P(x_r)-O(x_r)\|_\infty .
\]

由接受不等式 \(E^Q_r\le K E^P_r+\varepsilon_r\) 反解放大率：

\[
\hat K_r=
\begin{cases}
0,& E^Q_r\le\varepsilon_r,\\
(E^Q_r-\varepsilon_r)/E^P_r,
  & E^Q_r>\varepsilon_r,\ E^P_r>0,\\
+\infty,& E^Q_r>\varepsilon_r,\ E^P_r=0.
\end{cases}
\]

\(\varepsilon_r\) 是契约约定输出尺度上的 ULP 加性地板，位于原不等式右侧，不加进分母。先处理分支可避免 \(0/0\)；\(\hat K_r=0\) 只表示候选在 oracle 的地板误差内，不保证参考也在地板内。

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

报告的常见起始阈值是 \(K_{\mathrm{warn}}=2\)、\(K_{\mathrm{fail}}=10\)，需按算子和执行配置校准。尾部样本不足、拟合无效或协议未完成时返回 INCONCLUSIVE。全零放大率等无需拟合的退化数据需要单独记录判决分支，不得伪造拟合或置信界。

矩和误差比统计在有效有限数值上定义。浮点执行层完整表示 NaN、无穷、signed-zero 和 subnormal；统计前按输出契约处理特殊值。在要求有限输出的域内，候选产生非有限输出触发 FAIL；oracle 失败、执行错误或契约未规定的共同特殊值情形单独报告为无有效判定，不能纳入普通均值和比值计算，也不能静默过滤相关样本。

## 5. 哪些关系允许使用

原子执行成功、证据匹配当前契约且协议完整时，合并准入判决如下：

| Bias gate | Vars gate | 对变换的处理 |
|---|---|---|
| PASS | PASS | ACCEPT：满足本次统计接受协议 |
| WARN | PASS/WARN | ACCEPT_WITH_WARNING：保留报警原因，按契约的 WARN 策略决定是否允许替换 |
| PASS | WARN | ACCEPT_WITH_WARNING：同上 |
| 任意 | FAIL | REJECT：该契约下拒绝替换 |
| FAIL | 任意 | REJECT：该契约下拒绝替换 |
| 无 FAIL，但任一门无有效判定 | — | INCONCLUSIVE：尚无接受依据 |

报告中 WARN 有“记录但不拦截”的用途。框架保留这一选择，通过 `allowWarn` 等显式策略表达；也支持只自动采用双 PASS 的严格策略。采用哪种策略属于契约内容，不能把 WARN 静默改成 PASS。原子准入不能替代两端各自的实数正确性证明。缺少实现对应或验证／重放证据时，可以报告实验数据，但不能把该条目当作已经验证的可用假设。

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
- [实验入口](../experiments/floating_point/README.md) 已提供 GPU 原语对、源代码／PTX／后端身份、配置、种子、逐 replicate 统计量、拟合诊断和 CPU 导入重放。GPU 实测仍待用户运行；将重放后的准入表绑定到 Lean 原子关系的连接仍待实现，不能把 JSON 导入视为证明了 `EvidenceValidated`。
- 不把原子假设注册成具体 IEEE 函数的 Lean 等式或全局代数实例。已有实数正确性和历史抽象舍入证明不能冒充原子数值记录。
- 整 kernel 数值复查属于可额外报告的实验，不是这套形式等价性的定义。实际 GPU 结论仍需对应后端证据；软件 profile 结果只覆盖所选软件语义。

## 7. 报告中需修正或验证的保证

采用两门的职责、统计量与判决结构，不意味着报告中的强保证已成立。后续理论与实验工作至少需要处理：

1. **PASS 的含义与检出范围。** 不拒绝零均值假设不能证明无偏；报告的 FAIL 还同时要求超过 SNR 和 ULP 地板。因此仅满足 \(\rho_g\ge T_{\mathrm{bias}}^{-1/2}\) 并不保证检出，增加 \(R\) 也无法跨过固定效应量地板。需要针对完整报警区域给出功效目标与预算。[NIST 对检验与第二类错误的说明](https://www.itl.nist.gov/div898/handbook/prc/section1/prc13.htm)。
2. **五西格玛与自适应停止。** 用样本标准差构造的统计量依赖分布和样本量；渐近正态近似不能直接给出有限样本的严格误报界。反复查看置信区间后停止，也不能自动沿用固定样本量的覆盖率。需固定协议或采用经过论证的序贯方法；置信序列是可研究的路径之一：[原始研究](https://arxiv.org/abs/2301.09573)。
3. **Return level 的含义。** 理想连续模型下，\(r_T\) 对应单次超越概率约 \(1/T\)，不是 \(T\) 次最大值的期望，也不是保证不被超越的最坏值。独立抽样时至少超越一次的概率为 \(1-(1-1/T)^T\)，随 \(T\) 增大趋于约 \(0.632\)。必须区分 return level 的估计置信度和未来运行的超越风险。[NIST 的 return value 定义](https://www.nist.gov/programs-projects/maps-non-hurricane-non-tornadic-extreme-wind-speeds-contiguous-united-states)。
4. **尾部模型和参数截断。** POT/GPD 拟合有尾部近似及采样假设，不能直接宣称有限样本“分布无关”。报告把正 \(\hat\xi\) 截到零，需要额外依据；当参考误差很小时，误差比可能出现长尾。即使极限尾部有界，也不足以保证有限阈值下的指数拟合保守。截断前后的诊断与覆盖率需要验证。
5. **从局部误差到训练危害。** \(T\mu\) 与 \(\sqrt T\sigma\) 的比较需要相关性和传播假设。一般训练扰动还经过随时间变化的 Jacobian；存在负曲率不能单独推出整个乘积具有正 Lyapunov 指数。forward 门的通过不能直接推出 backward、训练轨迹或训练质量保证。
6. **计量与复现。** ULP 在 binade 边界具有方向差异，例如 bf16 在 1 上方的相邻间隔为 \(2^{-7}\)，下方为 \(2^{-8}\)。必须冻结 nextafter 方向及聚合规则。报告中的 29/30 负例检出、35/35 正例无 FAIL，须获得对应源码与原始结果后复现，再作为本项目证据。

### 当前 checker 的固定协议

`two-gates-fixed-pwm-v1` 采用固定 replicate 预算，不自适应提前接受；GPD 使用 PWM、条件尾部 bootstrap，不将正 shape 参数截为零。全零 K 明确走观察值退化分支，不伪造拟合；常数正尾部或任一无效拟合返回 INCONCLUSIVE。ULP 采用增大绝对值方向的格式间距；最大有限值使用该 binade 的间距。默认桶为输出列，向量归约输出使用一个桶。具体参数、范围、运行和回传命令见实验目录；这些选择不表示已校准或复现附件报告的保证。

## 8. 下一步

- [x] 明确实数正确性与基于 two-gates 准入原子假设的浮点实现等价性。
- [x] 区分两门的参照系、三档判决、证据不足和有向关系的使用范围。
- [x] 记录高斯探针及 shape 处理建议，指出加法重排的交换对称盲点；具体配置尚未冻结。
- [x] 按完整交付目标补充原语设计，明确值表示、FMA、混合精度、归约、dot/MMA、特殊值和后端连接。
- [ ] 完成 bf16/fp32 支持范围内的原语、执行计划与契约；加法重排等案例提供逐位不同见证及对应代数证明。
- [ ] 获取报告的 checker/实验代码，逐项核对统计实现、退化情形与停止规则；已有结果暂不视为 VeriTile 实验。
- [ ] 实现可重放的两门检查，校准阈值并报告功效与覆盖率限制。
- [x] 实现原子规则、准入条件、形式推导、两种 specification 和逐原子依赖报告；外部验证义务显式保留。
- [ ] 将原子准入连接到真实执行、证据重放与统计 checker；继续完善适用范围检查及执行层覆盖。
