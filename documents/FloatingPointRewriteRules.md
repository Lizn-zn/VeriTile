# 浮点变换规则与接受结果表

更新日期：2026-10-01。状态：**14 条原子候选关系的 42 个格式实例已有 GPU 报告，其中 30 个实例被准入。** 当前判决见[实测结果表](../experiments/floating_point/report/summary.md)；Lean 按用户确认信任该报告，冻结其 30 个接受实例。六个标量反例已在具体软件浮点模型下经 Lean 内核检查，结果见第 3 节。反例只否定通用严格相等，不等于 two-gates 拒绝。

这里的“等价性原语”对应可复用的变换规则，只包括固定规模的局部关系，例如结合律、分配律、FMA 与具体 cast 关系。归约重排和完整算法变换是待推导结论，不是准入原子。它与 [浮点执行原语](./FloatingPointPrimitives.md) 分开：执行原语定义程序怎样计算，本表记录参考计算如何变成候选计算，以及允许该变换的证据。

## 1. 表的单位与判决

**候选目录一行是一条有向规则；结果表一行是该规则在一个完整配置下的检查。** 同一条规则可以有多个结果，不能只按名字登记一次 PASS。

结果键至少包含：

    规则 ID + 参考/候选执行图身份
    + 输入/操作/累加/输出精度及 rounding/intrinsic 配置
    + 逐元素测试张量的 shape/stride 和 launch（归约、scan、dot 计划为 null）
    + 后端与双方编译/启动配置
    + 输入域/权重/高斯探针配置
    + two-gates 协议与检查器版本

分别保存严格等价证据和统计接受证据：

| 字段 | 可记录的值 | 含义 |
|---|---|---|
| 代数证明 | theorem + 前提 / 待证明 / 反例 | 理想算法层的性质 |
| 严格浮点关系 | theorem + 前提 / 未证明 / 数值反例 | 选定浮点语义下的严格关系 |
| Bias gate | NOT_RUN / PASS / WARN / FAIL / INCONCLUSIVE | 当前配置的一阶差异检查 |
| Vars gate | NOT_RUN / PASS / WARN / FAIL / INCONCLUSIVE | 当前配置的误差放大与尾部检查 |
| 接受决定 | NOT_EVALUATED / ACCEPT / ACCEPT_WITH_WARNING / REJECT / INCONCLUSIVE | 按契约合并证据与两门判决 |

严格浮点证据保留作内部分析和辅助引理；它不是 two-gates 的实验结果。加入当前原子假设表的数值条目仍须完成配置匹配的两门检查；未运行的 gate 保持 NOT_RUN。纯语法相同可用自反规则，不需要数值原子假设。

对外正确性使用实数语义 `Spec.Real`，实现等价性沿用 `lhs ≡[R] rhs`（`VeriTile.Spec` scope，底层为 `Spec.FloatingPoint`）。`R` 提供原子规则表及准入假设，实验配置和结果不作为公开规格的独立参数。two-gates 为表中的原子关系提供准入依据；被准入的关系作为假设，Lean 可以引用并组合成完整实现的形式等价证明。`#print_fp_assumptions` 只显示证明实际使用的原子名称。形式组合不等于整 kernel 已完成统计检查，也不构成 IEEE 位值相等；严格辅助证据不另设第三类公开规格。

统计接受路径按 [two-gates 协议](./TwoGatesAcceptance.md) 合并结果。待证明、待测试和证据不足分别记录，不能当作已证明错误；相反，某个输入上的数值不等也不能自动推导统计 FAIL。

## 2. 候选规则目录

以下公式是模式说明，正式检查必须实例化为完整的 typed Compute IR，明确每一步精度、转换和计算顺序。\(q_d\) 表示指定配置下转换到格式 \(d\)；数学规格会按其定义处理或投影这些转换。箭头标注数值检查的参考到候选方向；反向的实验结论需要另建记录。准入后的形式等价假设可以使用对称规则，但这不产生反向实验记录。

### 2.1 可研究严格证明的规则

这些条目的 two-gates 准入状态按实测格式实例记录；这不等于已有具体 IEEE 模型下的无条件严格证明。

| ID | 规则 | 模式 | 关键前提与当前证据 |
|---|---|---|---|
| ADD-COMMUTE | 加法交换 | \(a+b\to b+a\) | 相同操作格式与模式；需处理 NaN 选择、特殊值及观测标准；具体模型证明待完成 |
| MUL-COMMUTE | 乘法交换 | \(ab\to ba\) | 同上；不能把数值比较相等与 NaN 位模式相同混为一谈 |
| ROUND-IDEM | 同格式重复舍入消除 | \(q_d(q_d(x))\to q_d(x)\) | 已有抽象模型字段 round_idem；具体转换/特殊值/flush 规则仍需证明 |
| BF16-WIDEN-RETURN | 提升后转回 | bf16 → fp32 → bf16 | 有限 bf16 值、无改变值的 flush 等前提；具体模型证明待完成 |

### 2.2 需要按配置判定的数值变换

本节记录候选公式；各格式实例的数值判决以[当前结果表](../experiments/floating_point/report/summary.md)为准。数学定理不能代替 gate 结果，未通过的实例仍不可用于当前 FP 证明。

| ID | 规则 | 参考 → 候选 | 要绑定的条件或主要差异 |
|---|---|---|---|
| ADD-ASSOC | 加法结合重排 | \((a+b)+c\to a+(b+c)\) | 每次加法精度；不同角色的均值/尺度；第 3 节有严格不等反例 |
| MUL-ASSOC | 乘法结合重排 | \((ab)c\to a(bc)\) | 操作精度、尺度、overflow/underflow |
| MUL-DISTRIB | 乘法分配 | \(a(b+c)\to ab+ac\) | 舍入次数与 FMA 选择；第 3 节有严格不等反例 |
| FMA-CONTRACT | 乘加融合 | add(mul(a,b),c) → fma(a,b,c) | 分离乘加与一次舍入；第 3 节有严格不等反例 |
| CANCEL | 减加消除 | \((a-b)+b\to a\) | 抵消、运算格式、特殊值 |
| DIV-RCP | 除法改乘倒数 | `tl.div_rn(a,b) → a * tl.div_rn(1,b)` | 绑定实际 intrinsic 和额外乘法舍入；不能据此准入普通 Triton `/` 的替换 |
| SQRT-RSQRT | 倒平方根替换 | \(1/\sqrt{x}\to\operatorname{rsqrt}(x)\) | 理想规格要求 \(x>0\)；绑定函数实现及输入域 |
| CAST-MOVE | 输入量化后移 | \(q_d(q_d(a)+q_d(b))\to q_d(a+b)\) | 明确中间加法格式和提升路径；同一份原始输入 |
| CAST-REMOVE | 移除一个中间 cast | `q_compute(q_bf16(a+b) * c) → q_compute((a+b) * c)` | 两边使用相同输出 cast；只检查这个局部表达式，不量化任意 F/G |
| ACC-WIDEN | 三项和的局部精度提升 | `(a+b)+c` 的输入格式中间结果 → fp32 中间结果 | 两边使用相同输出 cast；不代表整个归约或 dot 的累加器可以直接替换 |

DIV-RCP、SQRT-RSQRT 等具有定义域前提的规则，需要明确从探针到合法操作数的生成方式，例如 kernel 内部生成正的平方和。条件化/变换后的分布必须登记，不能称作未经修改的标准高斯。所有规则都要处理特殊值，不得在查看差异后静默筛掉坏样本。

### 2.3 不进入原子假设表的内容

以下旧条目已从目录、GPU 执行入口和结果导入允许列表删除：

- 完整算法：`SOFTMAX-SHIFT`、`SOFTMAX-ONLINE`、`LAYERNORM-WELFORD`、`SWIGLU-FUSE`。
- 组合计算：`REDUCE-REORDER`、`REDUCE-SPLIT`、`SCAN-REORDER`、`DOT-LOWER`、`DOT-ACC-FUSE`、`GEMM-SPLIT-K`。
- 结构和内存性质：`LAYOUT-INVERSE`、`STORE-LOAD-FORWARD`。

前两组应由 Lean 根据已准入的局部关系和结构性推导证明；后一组需要索引、别名与内存条件的证明。整 kernel 数值实验即使通过，也不能将这些结论直接加入 `R`。例如 online softmax 若缺少某条指数关系，应明确该局部表达式并单独准入，不能改为假设整个 online 算法等价。此处并未声称现有 14 条关系已足以证明所有这些案例。

当前剩余案例的[原子关系缺口](./FPRemainingAdmissionGaps.md)已有 Lean 检查的标量反模型支持：现有准入关系不能直接补出初始化、普通除法或 exp/log 的关系。新增关系的定义域也须保留，例如不能将倒数消去无条件推广到零分母。

`shape = [4096, 4096]` 表示对原子表达式进行逐元素批量测试，不把该表达式升级为整张量归约假设。

已有证明只作为对应语义层的基础：

- [RoundingModel.lean](../VeriTile/Triton/Float/RoundingModel.lean) 给出抽象幂等性约束及 cast/store 引理。
- [FusedSwigluEquiv.lean](../bench/examples/FusedSwigluEquiv.lean) 有保留中间舍入的抽象模型结论，仍需检查独立 launch 与具体执行的连接。
- [FusedLayerNormEquiv.lean](../bench/examples/FusedLayerNormEquiv.lean) 有实数中间计算及输出抽象舍入的等价性结果。
- [OnlineSoftmaxCorrect.lean](../bench/examples/OnlineSoftmaxCorrect.lean) 有数学递推与 batch 输出的连接，完整 streaming 输出路径的范围仍按投稿计划核对。

## 3. 已计算的严格不等见证

以下输入均可在对应格式中精确表示。模式为 round-to-nearest, ties-to-even；每个普通加法/乘法都舍入到表中格式，FMA 行的候选只舍入一次。所有非零中间值都位于有限 normal 范围，零单独处理；这里不测试 flush、NaN 或原子调度。

| 规则 | 运算格式 | 输入 \((a,b,c)\) | 参考结果 | 候选结果 | 可以得出的结论 |
|---|---|---|---|---|---|
| ADD-ASSOC | bf16 | \((256,1,-256)\) | 0 | 1 | 不存在该模式下的无条件严格结合律 |
| ADD-ASSOC | fp32 | \((16777216,1,-16777216)\) | 0 | 1 | 同上 |
| MUL-DISTRIB | bf16 | \((3,256,1)\) | 768 | 772 | 不能作为无条件严格分配律 |
| MUL-DISTRIB | fp32 | \((3,16777216,1)\) | 50331648 | 50331652 | 同上 |
| FMA-CONTRACT | bf16 | \((129/128,127/128,-1)\) | 0 | \(-1/16384\) | 分离乘加与 FMA 可能不同 |
| FMA-CONTRACT | fp32 | \((8193/8192,8191/8192,-1)\) | 0 | \(-1/67108864\) | 同上 |

计算方法：使用 Python Fraction 保存精确有理数，对每个普通操作结果按格式舍入；bf16 的有效精度 \(p=8\)，fp32 为 \(p=24\)。对非零 normal 数，令 \(e=\lfloor\log_2|x|\rfloor\)，步长 \(h=2^{e-p+1}\)，将 \(|x|/h\) 舍入为最近整数、平局取偶数，再乘回 \(h\) 并恢复符号。FMA 在精确乘加后应用一次该过程。计算同时检查了输入可表示性及使用的 normal 范围。

这些结果最初由 Python Fraction 计算，现在也已由 [Counterexamples.lean](../VeriTile/Triton/Float/Counterexamples.lean) 中的六条定理复现：`bf16_add_assoc_witness`、`fp32_add_assoc_witness`、`bf16_distrib_witness`、`fp32_distrib_witness`、`bf16_fma_witness`、`fp32_fma_witness`（命名空间 `VeriTile.Triton.FP`）。证明使用 `decide +kernel`，检查的是具体软件 profile 下两侧各自的精确数值输出；没有引入外部数值计算公理。这不是 GPU 符合性实验。

**六条固定输入见证不属于随机采样的 two-gates 结果。** 标量实现的随机差分回归仅用于检查执行语义，不作为 gate 实验。

## 4. 按配置生成接受结果

正式结果表应保存下列字段，空结果使用 null/NOT_RUN，不填零：

机器可读目录位于 [rules.json](../experiments/floating_point/rules.json)。登记工具初始化的记录为 NOT_EVALUATED。[check_numerics.py](../scripts/check_numerics.py) 提供 Python 配置、14 条局部 Triton 实现对、GPU 配对采样和结果导入；导入端核对完整身份、PTX 与统计文件摘要并重新运行两门，生成逐配置数值准入表。哈希与统计重放不认证远端执行真实性，也不关闭 Lean 的外部验证义务。命令、精度支持矩阵和具体变换范围见 [实验目录说明](../experiments/floating_point/README.md)。

| 字段组 | 内容 |
|---|---|
| 规则与适用域 | rule ID、方向、数学前提、参考/候选图、证明引用 |
| 执行实例 | shape/stride、各步 dtype、累加计划、函数实现、编译/启动/设备身份 |
| 输入协议 | probe ID、每个角色的均值/尺度/联合关系、权重、量化/特殊值处理 |
| Bias 结果 | replicate 数、每桶均差/标准差/z/SNR/ULP、报警桶、门判决 |
| Vars 结果 | 逐 replicate oracle 误差和 K、return level/U、拟合诊断、门判决 |
| 审计证据 | checker/protocol 版本、原始结果位置、停止原因、时间与最终决定 |

当前结果由 [report_numerics.py](../scripts/report_numerics.py) 写入运行目录的
`summary.md`、`summary.csv` 和 `summary.json`，覆盖 14 条原子规则 × 3 种精度，
逐行报告最大绝对 z、U、U 的估计类型、两门状态和是否接受。缺失的统计量留空，
接受决定须经过 CPU 回放。每次刷新覆盖当前表，不追加历史表。

每个 probe 配置实际生成独立记录，不能把不同均值或尺度的样本混起来求偏差。规则汇总只覆盖明确列出的配置；未检查的 dtype、shape、计算树、探针或后端均保持未验证。

该表构成原子假设登记与证据查询接口。使用规则时须匹配配置与上下文前提；最终实现等价性由 Lean 在这些假设下推导。整 kernel 的数值复查可以另做，但不充当形式等价性的定义。
