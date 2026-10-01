# 浮点变换规则与接受结果表

更新日期：2026-10-01。状态：**26 条候选规则已有目录、Triton 实现对、Python 检查及重放入口；GPU two-gates 实测尚未运行。** 六个标量反例已在具体软件浮点模型下经 Lean 内核检查，结果见第 3 节。反例只否定通用严格相等，不等于 two-gates 拒绝。

这里的“等价性原语”对应可复用的变换规则，例如结合律、分配律、FMA 融合和归约重排。它与 [浮点执行原语](./FloatingPointPrimitives.md) 分开：执行原语定义程序怎样计算，本表记录参考计算如何变成候选计算，以及允许该变换的证据。

## 1. 表的单位与判决

**候选目录一行是一条有向规则；结果表一行是该规则在一个完整配置下的检查。** 同一条规则可以有多个结果，不能只按名字登记一次 PASS。

结果键至少包含：

    规则 ID + 参考/候选执行图身份
    + 输入/操作/累加/输出精度及 rounding/intrinsic 配置
    + shape/stride/归约或 dot 计划
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

对外正确性使用实数语义 `Spec.Real`，实现等价性沿用 `lhs ≡[R] rhs`（`VeriTile.Spec` scope，底层为 `Spec.FloatingPoint`）。`R` 提供原子规则表及准入假设，实验配置和结果不作为公开规格的独立参数。two-gates 为表中的原子关系提供准入依据；被准入的关系作为假设，Lean 可以引用并组合成完整实现的形式等价证明。`#print_spec` 显示原子表、配置、证据和前提。形式组合不等于整 kernel 已完成统计检查，也不构成 IEEE 位值相等；严格辅助证据不另设第三类公开规格。

统计接受路径按 [two-gates 协议](./TwoGatesAcceptance.md) 合并结果。待证明、待测试和证据不足分别记录，不能当作已证明错误；相反，某个输入上的数值不等也不能自动推导统计 FAIL。

## 2. 候选规则目录

以下公式是模式说明，正式检查必须实例化为完整的 typed Compute IR，明确每一步精度、转换和计算顺序。\(q_d\) 表示指定配置下转换到格式 \(d\)；数学规格会按其定义处理或投影这些转换。箭头标注数值检查的参考到候选方向；反向的实验结论需要另建记录。准入后的形式等价假设可以使用对称规则，但这不产生反向实验记录。

### 2.1 可研究严格证明的规则

这些条目当前未被登记为具体浮点模型下的已接受规则。

| ID | 规则 | 模式 | 关键前提与当前证据 |
|---|---|---|---|
| ADD-COMMUTE | 加法交换 | \(a+b\to b+a\) | 相同操作格式与模式；需处理 NaN 选择、特殊值及观测标准；具体模型证明待完成 |
| MUL-COMMUTE | 乘法交换 | \(ab\to ba\) | 同上；不能把数值比较相等与 NaN 位模式相同混为一谈 |
| ROUND-IDEM | 同格式重复舍入消除 | \(q_d(q_d(x))\to q_d(x)\) | 已有抽象模型字段 round_idem；具体转换/特殊值/flush 规则仍需证明 |
| BF16-WIDEN-RETURN | 提升后转回 | bf16 → fp32 → bf16 | 有限 bf16 值、无改变值的 flush 等前提；具体模型证明待完成 |
| LAYOUT-INVERSE | 结构操作抵消 | inverse-remap(remap(x)) → x | 索引映射互逆、位模式保持、有效 shape；需相应结构证明 |
| STORE-LOAD-FORWARD | 中间存取消除 | 同格式 store 后 load → 原值 | 无干扰写、别名/同步条件成立、转换位置相同；需内存与位值证明 |

### 2.2 需要按配置判定的数值变换

本表所有数值接受结果均为 **NOT_EVALUATED**。即使已有数学定理，仍不能预填 gate 结果。

| ID | 规则 | 参考 → 候选 | 要绑定的条件或主要差异 |
|---|---|---|---|
| ADD-ASSOC | 加法结合重排 | \((a+b)+c\to a+(b+c)\) | 每次加法精度；不同角色的均值/尺度；第 3 节有严格不等反例 |
| MUL-ASSOC | 乘法结合重排 | \((ab)c\to a(bc)\) | 操作精度、尺度、overflow/underflow |
| MUL-DISTRIB | 乘法分配 | \(a(b+c)\to ab+ac\) | 舍入次数与 FMA 选择；第 3 节有严格不等反例 |
| FMA-CONTRACT | 乘加融合 | add(mul(a,b),c) → fma(a,b,c) | 分离乘加与一次舍入；第 3 节有严格不等反例 |
| CANCEL | 减加消除 | \((a-b)+b\to a\) | 抵消、运算格式、特殊值 |
| DIV-RCP | 除法改乘倒数 | \(a/b\to a\cdot\operatorname{rcp}(b)\) | 理想规格要求 \(b\ne0\)；rcp 实现与额外乘法舍入 |
| SQRT-RSQRT | 倒平方根替换 | \(1/\sqrt{x}\to\operatorname{rsqrt}(x)\) | 理想规格要求 \(x>0\)；绑定函数实现及输入域 |
| CAST-MOVE | 输入量化后移 | \(q_d(q_d(a)+q_d(b))\to q_d(a+b)\) | 明确中间加法格式和提升路径；同一份原始输入 |
| CAST-REMOVE | 移除中间量化 | \(q_d(F(q_d(G(x))))\to q_d(F(G(x)))\) | 两边最终输出格式相同；保留 F/G 的实际运算实现 |
| ACC-WIDEN | 提升计算/累加精度 | 低精度中间计算 → 高精度中间计算，输出格式相同 | 更高精度不自动意味着符合参考的 bias gate |
| REDUCE-REORDER | 改变归约树 | reduction-plan A → plan B | 相同参与元素、长度、初值与 mask；实际树必须确定 |
| REDUCE-SPLIT | 分块归约 | 单段归约 → 分块后合并 | 块大小、局部与合并精度、padding 与初始化 |
| SCAN-REORDER | 改变 scan 计划 | serial plan → parallel plan | 每个输出的前缀/后缀语义相同；记录完整依赖图 |
| DOT-LOWER | 改变 dot 实现 | 显式乘积归约 → 目标 dot/MMA | 输入计算模式、乘积/累加行为、tile 计划 |
| DOT-ACC-FUSE | 累加器融合 | \(C+\operatorname{dot}(A,B)\to\operatorname{dotAcc}(A,B,C)\) | 累加器进入位置与各步舍入，不能代数化掉 |
| GEMM-SPLIT-K | K 维分块合并 | 完整 K 归约 → split-K 合并 | shape、分块、合并精度及原子调度 |
| SOFTMAX-SHIFT | softmax 平移 | 直接 exp/归一化 → 减去最大值后计算 | 相同数学结果；overflow/underflow 与归约实现变化 |
| SOFTMAX-ONLINE | 在线 softmax | batch → streaming/online | 完整输出、序列长度、分块与累加精度 |
| LAYERNORM-WELFORD | 方差算法替换 | two-pass → Welford | 相同方差定义、正长度、epsilon、归约与 affine 计算 |
| SWIGLU-FUSE | kernel 融合 | 独立阶段 → 融合执行 | 保留中间 cast 的变体与删除 cast 的变体分别登记；launch 边界也需对应 |

DIV-RCP、SQRT-RSQRT 等具有定义域前提的规则，需要明确从探针到合法操作数的生成方式，例如 kernel 内部生成正的平方和。条件化/变换后的分布必须登记，不能称作未经修改的标准高斯。所有规则都要处理特殊值，不得在查看差异后静默筛掉坏样本。

已有证明只作为对应语义层的基础：

- [RoundingModel.lean](../VeriTile/Triton/Float/RoundingModel.lean) 给出抽象幂等性约束及 cast/store 引理。
- [FusedSwigluEquiv.lean](../bench/examples/FusedSwigluEquiv.lean) 有保留中间舍入的抽象模型结论，仍需检查独立 launch 与具体执行的连接。
- [FusedLayerNormEquiv.lean](../bench/examples/FusedLayerNormEquiv.lean) 有实数中间计算及输出抽象舍入的等价性结果。
- [OnlineSoftmax.lean](../bench/examples/OnlineSoftmax.lean) 有数学递推与 batch 输出的连接，完整 streaming 输出路径的范围仍按投稿计划核对。

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

**尚未开展规则接受的随机采样或运行 two-gates；六条记录的两门状态均为 NOT_RUN。** 标量实现的随机差分回归仅用于检查执行语义，不作为 gate 实验。

## 4. 按配置生成接受结果

正式结果表应保存下列字段，空结果使用 null/NOT_RUN，不填零：

机器可读目录位于 [rules.json](../experiments/floating_point/rules.json)。登记工具初始化的记录仍为 NOT_EVALUATED。[fp_experiment.py](../scripts/fp_experiment.py) 提供 Python 配置、26 条具体 Triton 实现对、GPU 配对采样和结果导入；导入端核对完整身份、PTX 与统计文件摘要并重新运行两门，生成逐配置数值准入表。哈希与统计重放不认证远端执行真实性，也不关闭 Lean 的外部验证义务。当前尚无实际 GPU 准入结果；命令、精度支持矩阵和具体变换范围见 [实验目录说明](../experiments/floating_point/README.md)。

| 字段组 | 内容 |
|---|---|
| 规则与适用域 | rule ID、方向、数学前提、参考/候选图、证明引用 |
| 执行实例 | shape/stride、各步 dtype、累加计划、函数实现、编译/启动/设备身份 |
| 输入协议 | probe ID、每个角色的均值/尺度/联合关系、权重、量化/特殊值处理 |
| Bias 结果 | replicate 数、每桶均差/标准差/z/SNR/ULP、报警桶、门判决 |
| Vars 结果 | 逐 replicate oracle 误差和 K、return level/U、拟合诊断、门判决 |
| 审计证据 | checker/protocol 版本、原始结果位置、停止原因、时间与最终决定 |

结果展示形式如下；这些行说明记录粒度，**没有声称完成实验**。bf16-in/fp32-op 的实例明确输入转换和输出格式。

| 规则 | 输入 / 运算 / 输出精度 | Shape 与探针 | Bias | Vars | 接受决定 |
|---|---|---|---|---|---|
| ADD-ASSOC | bf16 / bf16 / bf16 | 每操作数标量；单独登记的高斯配置 | NOT_RUN | NOT_RUN | NOT_EVALUATED |
| ADD-ASSOC | bf16 / fp32 / bf16 | 同上；最终转回 bf16 | NOT_RUN | NOT_RUN | NOT_EVALUATED |
| ADD-ASSOC | fp32 / fp32 / fp32 | 每操作数标量；单独登记的高斯配置 | NOT_RUN | NOT_RUN | NOT_EVALUATED |
| REDUCE-REORDER | 按节点记录 | 某个确定长度与两棵确定的树 | NOT_RUN | NOT_RUN | NOT_EVALUATED |
| DOT-ACC-FUSE | 按 dot plan 记录 | 确定 M/N/K、tile 及输入模式 | NOT_RUN | NOT_RUN | NOT_EVALUATED |

每个 probe 配置实际生成独立记录，不能把不同均值或尺度的样本混起来求偏差。规则汇总只覆盖明确列出的配置；未检查的 dtype、shape、计算树、探针或后端均保持未验证。

该表构成原子假设登记与证据查询接口。使用规则时须匹配配置与上下文前提；最终实现等价性由 Lean 在这些假设下推导。整 kernel 的数值复查可以另做，但不充当形式等价性的定义。
