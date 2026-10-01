# 浮点运算原语：完整设计

更新日期：2026-10-01。状态：**已实现 Python 配置、14 条 Triton 原子关系、two-gates 和 CPU 结果重放；GPU 实测与 Lean 规则绑定仍待完成，证明将复用现有 agent/comparator**。

用户要求交付完整的“配置 → 检查原语 → 生成规则集 → 自动证明”流程，不以标量演示代替完整流程。已确认的实现路线减少了对自研执行器的要求：实际浮点运算交给 Triton/GPU，two-gates 在 Python 中运行，Lean 检查规则假设下的推导。[TwoGatesAcceptance.md](./TwoGatesAcceptance.md) 定义变换接受协议。

[FloatingPointRewriteRules.md](./FloatingPointRewriteRules.md) 登记候选变换规则、严格不等见证以及按配置生成的两门结果。执行原语与规则登记表共同服务完整验证流程。

## 1. 完整目标与架构

准入目录只记录 bf16、fp32 及混合精度下固定规模的局部表达式关系，包括算术、转换、FMA 与三项和的局部精度提升。当前保留 14 条；每条在指定 shape/config 下由 Triton/GPU 检查。不支持的实例明确报告。

归约、扫描、dot、Softmax、LayerNorm 和 SwiGLU 等完整变换是 Lean 的待推导结论，不作为 two-gates 准入原子。Layout 与 store/load 结构性质依靠索引、别名和内存证明。完整交付指从基本关系推导这些优化，不是将完整优化逐个登记成假设。

### 已确认的实现边界

| 部分 | 本轮处理 |
|---|---|
| 实数正确性、索引、内存与结构证明 | 复用现有 Lean 语义与证明 |
| 原语实际浮点执行 | Python 调用 Triton，在 GPU 上运行双方原语实现 |
| two-gates 与结果管理 | Python 负责采样、oracle、统计判定和记录；只用用户指定的分布准入 |
| 规则集与等价性 | 导入可用原子关系作为 `R`，Lean 检查引用、条件与组合；公开记号仍为 `lhs ≡[R] rhs` |
| 自动证明 | 复用 `scripts/prove.sh` 的证明 agent，并由 Lean/comparator 检查；不另建独立证明搜索器 |
| 完整 Lean IEEE 执行器、位值内存和 GPU 指令语义证明 | 不作为这条交付路线的前置条件；已有标量模块保留作参考和反例工具 |

后文第 2–6 节保留数值语义要求和可选软件参考设计。它们不要求新增一套完整 Lean 浮点执行器；涉及被测原语的精度、cast、执行顺序等信息仍须保留。当前交付范围以本节及第 10–11 节为准。下文的归约、DotAcc 等执行节点不因此成为准入规则；执行操作与可假设的局部关系是不同层次。

保留三个独立但连接的对象：

| 对象 | 职责 | 结论范围 |
|---|---|---|
| 数学规格与算法语义 | 表达理想计算，证明代数变换、索引与内存条件 | 理想模型中的精确性质 |
| Triton/GPU 原语执行 | 记录双方实现、格式、计算顺序及编译/设备配置，并取得实际输出 | 被测后端与固定配置下的输出 |
| Two-gates 接受关系 | 比较参考与候选的执行结果 | 指定探针、shape/config 与统计协议下的接受决定 |

浮点原语本身按其定义执行，不能因为某个输入分布上的实验通过，就修改原语定义使结合律成立。高斯探针、shape 的验收实例和统计阈值属于接受契约，不进入加法原语的数学定义。

用户入口统一为 Python 中的验证配置，主要指定 shape 和输入生成方式；dtype、累加精度可显式指定或从目标程序解析。用户不手动给每条 Lean 规格传实验参数，也不手动填写规则的 PASS。

    Python 验证配置 + 预定义候选规则库 + 目标实现对
                         |
               实例化规则、解析数值执行配置
                         |
             配对采样并逐条执行 two-gates
                         |
                  生成可用原子规则集 R
                         |
            现有证明 agent + Lean/comparator
                         |
                   lhs ≡[R] rhs

数学正确性保留独立路径：目标实现的数学投影 → Lean 证明 implementation = 数学公式，不依赖数值门的通过情况。

执行顺序、精度选择和来源映射必须在擦除 dtype 或代数化之前保留。具体的源码到 IR、IR 到目标执行连接分别记录证明或信任依据。

## 2. 值表示：可选软件参考与特殊值

已有 Lean 软件参考使用按格式索引的原始位模式作为浮点载体。它用于参考计算和反例，不替代选定的 Triton/GPU 测试：

    FPValue format := BitVec format.bitWidth

bf16 描述包含 16 位、8 位指数和 7 位显式 fraction；fp32 包含 32 位、8 位指数和 23 位显式 fraction。格式描述还需给出 bias、subnormal、特殊编码等规则，不能仅靠总位宽推断行为。数值分类为 zero、subnormal、normal、infinity、NaN，保留符号与 NaN payload/quiet bit；具体运算怎样传播或规范化 NaN，由后端配置约束。

同时提供：

- 精确 decode：有限值解码为有理数，并保留原始位模式；到实数的映射只在有意义的域上使用。
- 有规范的 encode/round：从精确中间结果舍入到目标格式，处理 tie、overflow、underflow 和有符号零。
- 数值比较、位相等与分类三个不同接口。例如 \(+0,-0\) 位模式不同，数值比较可相等；NaN 的数值比较不能使用 Lean 值相等代替。

未定义 lane、未初始化数据、越界访问或执行失败属于执行状态，不借用 NaN 或无穷表示。当前 `WithBot ℝ` 同时承担的数学 bottom/负无穷等角色不能直接迁移为具体浮点值。掩码掉的读取应按原语语义产生 other 或未定义值，不能一律当作零。

## 3. 标量原语及签名

下面是接口设计，名称不表示仓库已存在对应定义。原语的输入、输出格式由类型携带；不合法组合在 lowering 时拒绝。

| 原语族 | 必须保留的参数 | 执行规则 |
|---|---|---|
| const / fromBits | 原始字面量来源、目标格式、字面量转换路径 | 确定一次转换结果，不能使用任意实数常量冒充浮点常量 |
| convert | 源/目标格式、舍入模式、目标适用的特殊值策略 | 数值转换；与 bitcast 分离 |
| bitcast | 源/目标位宽及其相等证据 | 保持位模式，不进行数值舍入 |
| add / sub / mul | 操作格式、舍入与 subnormal 策略 | 在该操作的精度下计算并舍入一次 |
| divRN / sqrtRN | 操作格式与明确的舍入语义 | 对精确商/平方根正确舍入，特殊值另有规则 |
| fma | 输入类型、结果格式、舍入模式 | 乘加整体只进行一次规定的舍入 |
| neg / abs / copySign | 格式与特殊值处理 | 按位/分类规则执行，不凭空增加算术舍入 |
| compare / isNaN / isInf / isFinite | 比较谓词、NaN 规则 | 产生布尔值；不沿用实数全序 |
| min / max / arg selection | NaN 传播策略、signed-zero 与相等时的选择规则 | 明确选择哪一项 |
| floor / ceil / trunc / round-to-integral | 格式、舍入或取整规则 | 区分浮点结果与整数转换 |
| float↔integer | 整数位宽/符号、舍入、越界处理 | 保留转换域；未定义情形不能用任意数值默认值掩盖 |
| nextafter / ULP | 格式、方向、特殊值策略 | 用于量化诊断与 gate 地板，不能用宿主 fp64 的相邻值代替 bf16/fp32 的相邻值 |
| backend intrinsic | 函数身份、输入/输出格式、实现与误差契约 | 用于快速除法、rcp/rsqrt、超越函数及机器相关指令 |

有限输入下，标准加法的数值关系可写为：

\[
\operatorname{add}_{f,m}(x,y)
=\operatorname{round}_{f,m}(\operatorname{decode}(x)+\operatorname{decode}(y)).
\]

这只是有限值分支的说明，完整定义还包括 NaN、无穷、有符号零以及目标允许的输入/输出 subnormal 处理。

原语不引入独立高斯噪声，也不假设舍入误差零均值或相互独立。同一确定性配置下，相同位模式输入产生相同结果；随机性由探针生成、实际随机舍入或执行调度的单独模型表达。实现路线中这些原语由 Triton/GPU 执行，Lean 侧只须保留规则匹配所需的操作身份和数值参数。

### 3.1 FMA 必须独立

\[
\operatorname{fma}(a,b,c)=\operatorname{round}(ab+c)
\]

与

\[
\operatorname{add}(\operatorname{mul}(a,b),c)
=\operatorname{round}(\operatorname{round}(ab)+c)
\]

属于不同执行图。源码中相邻的乘法和加法是否被编译器合并，需要从固定编译配置及 lowering 结果确认。不能只看 Python 括号推断硬件实际舍入点。NVIDIA 的说明分别讨论了 FMA、计算顺序与编译选项对结果的影响：[浮点执行说明](https://docs.nvidia.com/cuda/floating-point/index.html)。

### 3.2 精度属于每个操作

输入存储格式、操作格式、乘积/累加方式和输出存储格式分别表达。对于 bf16 输入、fp32 运算、bf16 输出，规范化执行图示意为：

    a32 = convert bf16 → fp32 a
    b32 = convert bf16 → fp32 b
    c32 = convert bf16 → fp32 c
    t32 = add fp32 a32 b32
    y32 = add fp32 t32 c32
    y16 = convert fp32 → bf16 y32
    storeBits bf16 y16

逐操作 bf16 加法则会在每次加法后得到 bf16 值，两者不能共用一个只写“输入 bf16”的语义配置。所谓累加精度，具体落实为循环携带值/归约节点的格式与运算，而非给整个 kernel 放一个不约束内部节点的字符串。

若某个目标用 fp32 运算再下转换来实现源级 bf16 操作，应显式记录这条路径；不能未经论证就将多步实现替换成理想 bf16 原语，忽略双重舍入或 subnormal 处理的差异。

前端需按固定 Triton 版本解析类型提升、标量常量与显式 cast。Triton 对不同 dtype、标量参与运算和除法有不同的提升规则；不能统一按“选最宽类型”处理：[类型提升规则](https://triton-lang.org/main/python-api/triton-semantics.html)。下转换的 rounding 参数及 bitcast 也应保留：[cast 接口](https://triton-lang.org/main/python-api/generated/triton.language.cast.html)。

## 4. Tensor、归约与矩阵乘法

逐元素计算使用标量原语加显式广播/索引映射；reshape、transpose、broadcast、join/split、gather 等结构操作保持被选值的位模式。索引无效与 shape 不匹配是独立义务，不默认填零。

### 4.1 ReductionPlan 与 ScanPlan

浮点 sum 以有序计算树或 DAG 表示，至少保留：

- 输入 shape、归约轴、索引映射及输入转换。
- 初始化值、padding/mask 的值和参与规则。
- 每个组合节点的原语、操作格式与舍入。
- 局部 tile、跨 tile 的组合顺序，以及循环携带的累加器。

定义 well-formed 条件，保证需参与的输入被正确引用，初始化和 padding 与选定执行相符。数学规格仍可写 \(\sum\)，具体执行不能使用依赖结合律的无序 `Finset.sum` 代替。即使 dtype 和 shape 相同，不同计算树也可能给出不同结果。

Scan 为各输出保留依赖图和方向；不能将并行 scan 的计算顺序默认换成串行 fold。Min/max/argmin/argmax/sort 同样携带 NaN、signed-zero 与 tie 规则。

归约顺序若未从目标实现解析出来，可以使用明确给出的软件计划验证该计划，或用受约束的目标算子契约；不能声称某个随意选择的树就是实际 GPU 的执行树。

### 4.2 DotAcc 与 MMA

定义带累加器的独立原语：

    dotAcc(plan, A, B, C) → D

累加器 C 的参与位置、每个 tile 的结果如何合并都在 plan 中表达。当前 `Op.dot` 把三参数调用解释为 `acc + dot(a,b)` 的做法只能保留在数学层，不能用作浮点执行定义。

DotPlan 至少包括：

| 参数 | 作用 |
|---|---|
| A/B 存储及实际参与乘法的格式 | 表达输入转换、TF32 等计算模式 |
| 乘积与累加规则 | 区分独立 mul+add、FMA、分组 MMA 和内部累加行为 |
| 累加器初值及格式 | 保留 C 的值、进入时机与更新精度 |
| shape、tile 与 split/merge 计划 | 记录 K 分块、split-K、跨 program 合并 |
| 输出转换 | 明确寄存器结果何时转换到存储格式 |
| 目标指令/后端身份 | 绑定具体实现及其约束 |

对 scalar FMA 实现，用明确的原语图解释。对 Tensor Core/MMA，采用目标指令级原语及其契约；没有依据时不能假定内部行为等于某个逐项 fp32 FMA 循环。

Triton 的 `dot` 区分 accumulator 与 `input_precision` 等配置，fp32 输入也可能采用降低输入有效精度的计算模式。因此 input dtype + accumulator dtype 不能唯一确定 dot 行为：[dot 接口](https://triton-lang.org/main/python-api/generated/triton.language.dot.html)。配置应记录解析后的实际模式，不依赖随版本变化的默认值。

## 5. 数学函数与后端语义

exp/log/exp2/log2/rsqrt/tanh/sin/cos/tan/atan/sinh/cosh/erf/pow，以及由它们构造的 sigmoid 等，必须绑定实现。

- 数学规格使用理想函数，并写清定义域。
- 正确舍入的参考实现必须实际满足对应规格。
- 后端近似实现绑定函数身份、目标、输入域、特殊值及适用的误差性质；不能直接声明它等于理想函数最后舍入一次。
- sigmoid、GELU 等复合函数采用明确的分解图或命名后端实现，保留每一步精度。

后端接口可参数化函数：

    evalIntrinsic : IntrinsicId → TypedInputs → TypedOutput
    satisfiesContract : 所选实现满足其声明契约的证据或显式外部义务

参数化函数可用于证明，但不能完全无约束地补成某个函数后，便声称覆盖真实 GPU。每个被执行实例必须能解析到可执行实现；每项硬件对应主张另有依据。允许契约描述多个合法结果时，语义以关系/结果集合表达，不能任选一个结果代表全部行为。

标准正确舍入操作与快速数学操作使用不同标识。例如 Triton 提供明确的 `div_rn` 接口：[精确除法](https://triton-lang.org/main/python-api/generated/triton.language.div_rn.html)。通过类型或能力约束避免给不支持的指令任意组合 rounding/flush 选项。

可选软件参考可以使用位运算、整数/有理数算法或外部实现，例如 [Berkeley SoftFloat](https://www.jhauser.us/arithmetic/SoftFloat.html)。主线不要求完成这种参考执行器；two-gates 的高精度 oracle 单独定义和核对。不能把 fp64 算完再转换默认当作 GPU 每一步浮点执行的替代。

## 6. 内存、控制流与非确定性

同格式 load/store 保持位模式。若 source value 与 buffer 格式不同，lowering 插入显式 convert，再写位模式；已经转换到目标格式的值不会在普通 store 上无故多舍入一次。原始 NaN payload 的搬运与执行算术后的 NaN 处理分别定义。

相关规则涉及以下行为时，须保留测试与证明所需的条件；不要求为此新增完整的 Lean 位值内存/控制流执行器：

- masked load 的 other/undefined，masked store 的不写入，以及真实负无穷填充值。
- 比较、select、条件分支、循环；保留浮点比较结果对执行路径的影响。
- `where` 的操作数求值和真正分支的区别，不能因未选某个值就默认相关内存访问未发生。
- atomic add 的读—计算—写和返回旧值；CAS 的位比较与普通浮点数值比较分别定义。
- 浮点原子归约的合法调度序列，与现有并发/内存模型的连接。

非确定执行需显式给出量化范围：对所有合法调度的证明，或对固定运行协议采样的接受结果。Two-gates 测到某批原子调度，不能推出全部调度都可接受。Race、越界和未定义行为的义务独立检查，不能用数值统计通过来关闭。

## 7. Shape 与实例身份

源码保持参数化，验证实例固定合法 shape/stride、dtype 和双方的编译/启动配置。标量原语不依赖全局张量尺寸；具体的归约树、dot 计划、索引和 mask 会依赖实例配置。

每个实例记录：

    source/model/IR identity
    input/output shape and strides
    per-operation precision and intrinsic profiles
    each side's launch, compilation and autotune choices
    backend/compiler/device identity when targeting actual execution

编译器的 fusion、fast-math、TF32 模式等必须进入身份。替换某个 kernel 后原证据不能继续指向旧的 IR 或二进制。仅固定 shape 仍不足以固定执行语义。

### 7.1 用户流程：配置 → 检查原语 → 生成规则集 → 自动证明

这是完整交付采用的入口设计，**不是已经接通的运行 API**。用户在 Python 中定义一个验证配置。例如，下列配置取每个元素独立服从 \(\mathcal N(1,\sigma^2)\)，`sigma=0.5` 仅为可运行的配置示例值：

```python
sigma = 0.5
profile = {
    "shape": (4096, 4096),
    "input_distribution": {
        "family": "normal", "mean": 1.0, "std": sigma,
        "elements": "independent", "operands": "independent",
    },
    "dtype": "bf16",
    "acc_dtype": "fp32",
}
```

`std` 是标准差，方差为 `sigma ** 2`。允许用户提供自定义的联合采样函数，例如相关操作数、固定权重或不同角色的分布；系统记录其配置和实现身份。实际测试输入按声明的 dtype 量化。固定权重与每个 replicate 重抽的输入分开；不能每调用一次采样函数就改变已声明固定的权重。原语的参考、候选和 oracle 接收同一份已生成输入。

系统负责以下步骤，用户入口不增加逐原语的 `experiment`、`evidence` 参数：

1. **实例化预定义规则。** 候选目录固定，shape/精度/采样配置变化时重新实例化。逐元素加法可把 `(4096, 4096)` 作为三个操作数各自的 shape。该尺寸只定义局部表达式的测试批次。归约树、dot 执行计划和循环结构属于后续 Lean 推导，不作为整体关系送入原子准入表。
2. **逐条运行 two-gates。** 同一 replicate 内配对执行两侧及 oracle，再按固定协议给出结果。一整个张量采样是一个 replicate，不能把 `4096 × 4096` 个元素当成同样数量的独立 replicate。协议预算、分桶和阈值由选定的协议配置提供，不能看过结果再调到通过。用户已确认只按指定分布准入；系统不追加其他探针分布，也不要求它们通过。同分布导致的偏差抵消作为结果解释，不能自行变成新的拒绝条件。
3. **自动生成规则集 `R`。** 按选定策略收集接受的条目；失败、无结论、未运行及不支持的条目保留状态但不成为可用假设。规则更新改变的是可用原子关系集合，不修改浮点加法等执行原语的定义。
4. **复用现有 agent 自动尝试证明。** 先固定准入规则表和目标规格，再调用 `scripts/prove.sh`。Agent 使用已接受的规则模板，处理 shape、各步精度和规则附带条件，Lean/comparator 检查推导及目标未被改写。公开结论仍是 `lhs ≡[R] rhs`，`#print_spec` 展开原语及假设。Agent 不负责修改准入判决、阈值或可信规则文件；找不到推导时报告未解决目标，不自动补一个假设。
5. **配置变化后重新检查。** 改变 shape、分布、精度或执行后端，生成新的规则集并重跑依赖它的证明。旧结果保留原配置；可以复用配置身份完全匹配的检查，但不把旧 PASS 覆盖到新配置。规则通过数量不保证随 shape 或 `sigma` 单调变化。

这里的输入分布默认指**原子规则的测试操作数分布**。由此得到 `R` 下的形式等价性，不要求证明程序中间值也独立同分布。若某条检查明确要代表实际 kernel 内部的使用位置，则由输入生成器执行到该位置，保留中间操作数的联合关系并另建规则实例；不能把它们重抽成独立高斯。整 kernel 的统计接受结果是可选的额外实验，不从规则组合自动推出。

当前仓库已具备 Python 配置、14 条局部 Triton 实现对、two-gates runner、数值准入表导入、具体标量参考、条件等价推导、假设打印和通用 agent/comparator 入口。GPU 实测由用户在独立机器运行并回传；命令见[实验目录](../experiments/floating_point/README.md)。当前生成的是 JSON 准入表，尚未构造 Lean `Rules` 或关闭 `EvidenceValidated`；仍需绑定可实例化的 Lean 规则模板并连接已有证明入口。完整流程不以脚本编译通过代替验收。

## 8. 正确性、等价性与原子假设

用户确认的划分是：**正确性一律在实数语义下证明 implementation = 数学公式；实现等价性在浮点优化理论下，由原子假设推导。** two-gates 检查的是结合重排、交换、FMA 融合等原子关系，通过后才允许将它登记为可用假设。

| 对外接口 | 含义 |
|---|---|
| `Spec.Real claim` | 实数意义上的实现正确性 |
| `lhs ≡[R] rhs` | 在 `R` 提供的原子假设下，两段实现的形式等价性 |

[Spec.lean](../VeriTile/Spec.lean) 已实现 `AtomicRule`、`RuleEntry`、`AcceptedAtom`、`AcceptedAssumptions` 和 `Derivation`。每条原子记录须有配置匹配、有效 artifact、`EvidenceValidated` 验证／重放义务及符合策略的双门结果；仅填 PASS 不能成为可用假设。原来的整 kernel 凭证组装接口已删除。

`Derivation` 支持自反、对称、传递及共同语句前后文；`FloatingPoint` 还检查程序签名保持一致。当前 ComputeKernel 视图保留输入／输出元数据和真实 ComputeStmt 序列。它实现的是假设生成的形式理论，不把统计检验结果解释成对称、传递的概率保证，也不把该关系转换成具体 IEEE 位值的 Lean `=`。

`#print_spec` 打印两段实现、原子假设表及逐项配置／证据、声明前提、传递依赖的原语／规则和公理；`full` 补充完整依赖。报告与声明假设集合均是保守范围，不声称是最小使用集合。严格浮点事实保留为内部辅助引理；不能向具体浮点函数注入与已知反例矛盾的结合律公理。

[TritonBench 示例](../bench/examples/TritonBenchVectorAdditionFP.lean) 复用原始 vector_addition，构造只交换 `x + y` 操作数的变体。整个 kernel 的等价证明引用一条 `ADD-COMMUTE` 原子假设，并在 Lean 中验证前后 load/store 语法保持不变。原来的实数正确性证明独立保留。公开规格只写 `(R : Rules) : originalKernel ≡[R] optimizedKernel`，实验配置与结果归入 `R` 的规则表，由 `#print_spec` 展开。该规格以 `R` 中的原子准入假设为条件；未运行 two-gates，不提供已验证的 `R`。

数值执行层和 checker 仍负责真实浮点计算、配对采样、oracle、非有限值及统计协议。规则配置保留 dtype、累加精度、shape、执行顺序、后端及原子测试操作数分布。原始输入为高斯不保证中间值高斯；按第 7.1 节的默认流程，Lean 检查 `R` 下的形式推导，不证明分布传播。声称某个原子实验代表实际内部使用位置时，需要对应的操作数来源和联合采样依据。

完整 kernel 的数值复查可以作为额外实验，但不再是定义形式等价性的必备步骤。对称和组合后的形式等价定理，也不自动成为新的整 kernel 两门实验记录。

## 9. 与当前代码的连接

沿现有算法/compute 分层保留实数证明，只补浮点规则使用所需的表示和连接，不将旧的实数 carrier 替换成完整位值执行器：

| 当前位置 | 已核对的缺口 | 完整设计中的处理 |
|---|---|---|
| [Core/Types.lean](../VeriTile/Triton/Core/Types.lean) 与 [Semantics/Scalar.lean](../VeriTile/Triton/Semantics/Scalar.lean) | 浮点标签的数学 carrier 为 WithBot ℝ | 保留实数正确性路径；实际数值计算放在 Triton 原语实现 |
| [Core/Ast.lean](../VeriTile/Triton/Core/Ast.lean) | 当前规则只匹配固定语句片段 | 增加带变量和条件的规则实例化，保留操作顺序与各步精度；不新增通用位值解释器 |
| [DSL/Expansion/Main.lean](../VeriTile/Triton/DSL/Expansion/Main.lean) 的 dot 展开 | acc 被代数化为额外加法，部分精度参数被擦除 | 浮点证明表示保留 DotAcc/FMA、精度模式及必要执行配置，防止未检查的数值差异变成语法相同 |
| [Semantics/TileOps.lean](../VeriTile/Triton/Semantics/TileOps.lean) | reduceSum/dot 使用实数求和 | 数学模型保留；归约/矩阵变换需展开或建立明确的执行连接后由局部原子关系推导，不直接统计准入整个算子 |
| [Float/EvalOpR.lean](../VeriTile/Triton/Float/EvalOpR.lean) 与 [Float/StepR.lean](../VeriTile/Triton/Float/StepR.lean) | 现有抽象 cast/store 模型 | 保留历史语义，不将其当成 GPU 执行器或新的准入结果 |
| [Spec.lean](../VeriTile/Spec.lean) | `EvidenceValidated` 仍是未接通的验证前提 | 将已校验的 Python 规则表接入模型假设；Lean 检查形式推导，不承担统计检验的形式化证明 |
| [prove.py](../scripts/prove.py) | 已有 agent、可信输入快照与 comparator 检查 | 在调用前固定规则集和目标；复用入口，不允许证明 agent 修改准入集合 |

测试绑定的是实际 Triton 原语及其配置，不能以 Lean 的数学 evaluator 替代其数值输出。数学投影也可以失败：runtime bitcast、NaN 依赖分支等需要相应规格与前提，不能假设删掉 rounding 就得到合法代数证明。旧等价证明中的实数代数步骤不能直接作为新浮点规则，需要迁移为引用已准入的原子关系。

抽象舍入模型与具体 FP 模型之间只建立已证明、范围明确的桥接；不能宣称包含特殊值的全部 IEEE 行为都是当前 `R : ℝ → ℝ` 模型的实例。

## 10. 完整交付的完成条件

以下条件验收整条流程；不再要求先完成通用 Lean IEEE 执行器或独立自动证明器：

- [ ] Python 配置接受 shape、用户输入生成器、dtype 与累加精度，并解析双方原语的实际执行配置。
- [ ] 候选规则目录具有可执行的 Triton 原语对及 oracle，只覆盖 bf16/fp32 与混合精度下固定规模的局部表达式；复合算法、归约/scan/dot 变换必须由 Lean 推导。
- [ ] Two-gates 在 GPU 原语输出上工作；配对采样、分桶、预算、退化情形、非有限结果和尾部拟合失败有明确处理；只按指定分布准入。
- [ ] 通过检查的规则自动进入配置专属的 `R`；旧 PASS 不跨配置复用，失败/无结论/未运行记录不能混入可用集合。
- [ ] 规则可带变量实例化；FP 证明路径保留 shape、dtype、cast、FMA、DotAcc 与必要执行配置，不利用擦除后的语法绕过准入。
- [ ] 调用前固定规则文件与目标规格，复用现有 agent 和 comparator；成功须通过 Lean 检查，失败保留未解决目标，不能改题或新增假设来制造成功。
- [ ] 公开规格沿用 `lhs ≡[R] rhs`，`#print_spec` 展示原语及假设；实数正确性独立保留。
- [ ] 代表性 TritonBench 例子完成端到端运行，配置变更能够触发检查和证明重跑，产出可复现结果与成本评估。

已有位值模块和反例保留作辅助工具。检查仍须区分 FMA 与分离乘加、中途 bf16 舍入与 fp32 累加、不同归约/矩阵配置，以及编译配置变更；这些对应由 GPU 测试及规则表示共同承担，不需要先证明整个 GPU 的指令语义。

## 11. 实施计划与依赖（2026-10-01 开始）

以下是同一完整交付的依赖顺序，不是逐步缩减范围的不同产品版本。第 10 节是整体完成条件；某个阶段完成不能替代整体验收。

| 顺序 | 工作与产物 | 验收与依赖 |
|---|---|---|
| 1 | Python 配置入口、候选规则目录、Triton 原语对及 oracle | 14 条原子目录与配置身份可复用；实际运算由 GPU 执行，形状和各步精度必须匹配 |
| 2 | 指定输入分布、配对采样、bias/vars gates、结果记录 | 不追加其他探针；处理退化数据、失败和无结论；实际运行需要可用 GPU。依赖 1 |
| 3 | 规则集生成、可实例化原子关系和 Lean 接口简化 | 解开占位验证前提与规则导入的阻塞；只开放准入条目，保留数值操作身份。可与 1–2 的独立部分推进 |
| 4 | 接入现有证明 agent 与 comparator | 先冻结规则集和目标，再证明；拒绝修改目标或可信规则文件；打印依赖。依赖 3 |
| 5 | 全流程回归、规则覆盖、TritonBench 例子和论文实验 | 配置变更重跑；记录成功、拒绝、无结论、未支持与未证明目标；核对第 10 节。依赖前述阶段 |

GPU 主线已写好 [profile.py](../experiments/floating_point/profile.py)、[Triton 候选对](../experiments/floating_point/triton_rules.py)、[运行／重放入口](../scripts/fp_experiment.py) 和 [checker](../scripts/fp_two_gates.py)。默认配置为 4096×4096、独立 Normal(1, 1²)、三组 bf16/fp32 精度；均值、sigma、预算和规则选择可在运行前修改。当前执行器仅检查局部原子关系；`ACC-WIDEN` 只比较三项和的输入格式与 fp32 中间结果。自定义联合采样器与更多局部关系仍需单独实现。编译检查覆盖 masked 32×33 和 4096×4096，每个尺寸有 82 个支持的原子编译实例；实际 GPU 统计结果仍为 NOT_RUN。

可复用的其他基础如下，前三项是可选软件参考，不再是 GPU 主线的依赖阶段：

- [BitValue.lean](../VeriTile/Triton/Float/BitValue.lean)：按格式索引的位值、全部类别解码、四种舍入模式、转换以及分别配置输入/输出 flush。
- [ScalarOps.lean](../VeriTile/Triton/Float/ScalarOps.lean)：逐操作舍入的 add/sub/mul/div、单次舍入 FMA，以及数值相等和有序比较。运算精度由参数的格式确定，混合格式须显式转换。
- [Counterexamples.lean](../VeriTile/Triton/Float/Counterexamples.lean)：六条具体反例定理，通过 Lean 内核计算检查，覆盖 bf16/fp32 的结合重排、分配和 FMA。
- [规则目录](../experiments/floating_point/rules.json) 与 [登记工具](../scripts/fp_rule_registry.py)：14 个原子 ID、完整身份维度及初始未运行记录。目录中的严格候选仍需逐条证明，不能按类别自动接受。

软件标量 profile 的明确选择是：算术/转换产生 canonical quiet NaN，保留带符号零，分别配置输入与输出 subnormal flush，不观测异常 flags。它定义一套可执行值语义，尚不声称符合任何 GPU 指令。原始位值及符号操作保留 NaN payload；算术 NaN 策略与硬件的连接需在后端阶段验证。

当前验证包括全部 65,536 个 bf16 编码的提升/转回检查（NaN 按所定义策略 canonicalize）、边界与特殊值、四种舍入模式的有理数区间参考差分、fp32 与 NumPy 的宿主差分、以及六个文档反例的回归。测试命令见 [实验目录说明](../experiments/floating_point/README.md)。这些测试与六条反例证明不等于全部原语的通用舍入正确性证明，也不认证实际 Triton/GPU 的指令数值行为。

普通乘加与 FMA 保留为不同操作：后者对精确乘积加上第三项后只舍入一次。这一执行差异也是后端检查必须保留的条件（参见 [NVIDIA 浮点说明](https://docs.nvidia.com/cuda/floating-point/index.html#the-fused-multiply-add-fma)）。
