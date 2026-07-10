# 语义 Caveats

**中文** | [English](SemanticCaveats.md)

本文记录 VeriTile 相比真实 Triton/CUDA 有意简化或不完全忠实的语义点。这些不是
Lean soundness 问题,而是解释 theorem statement 时必须带着看的边界。除非 theorem
陈述或 `GapPolicy` 明确覆盖硬件 gap,否则已证明 theorem 应理解为 VeriTile 数学
算法语义下的 theorem。

## 高风险边界

| 边界 | 当前模型 | 风险 | 必要 discipline |
| --- | --- | --- | --- |
| IEEE 浮点 | 浮点 carrier 是 `WithBot ℝ`;dtype tag 在算法证明层擦到数学 Real 语义。 | 没有证明 NaN、signed zero、rounding、overflow、underflow、denormal、exception flag、fast-math rewrite、硬件 dot precision。 | 对数学算子写明 domain/range 前提;bit-level claim 走 `GapPolicy` / 外部检查。 |
| partial 数学函数 | `tl.log`、`tl.sqrt`、`tl.rsqrt`、`tl.extra.cuda.libdevice.pow` 在有限 Real 输入上使用 Mathlib total function。 | 真实硬件上会产生 NaN 或 inf 的非法输入,在模型里可能变成普通数学值。 | 声称 Triton/CUDA fidelity 时,给出 `x > 0`、`x ≥ 0`、分母非零、`pow` base 为正等前提。 |
| fixed-width integer | `tl.int*` 映射到数学 `Int`;`tl.uint*` 映射到 `Nat`。 | 没有 width、overflow、wraparound、saturation、sign-extension、signed fixed-width bitwise 语义。 | quantization / int kernel 需要 `tritonInt8CastInRange` 这类 range predicate;不要从 `.int` proof 推出硬件宽度行为。 |
| 地址 | pointer 是 `RegionName × Nat`;block-pointer offset、stride、base offset 都是 `Nat`。`BlockPtr.AdvanceNonnegative` 记录 signed `tl.advance` delta 的 theorem-side 无 underflow 义务。 | 执行语义里的负向 pointer arithmetic 和 underflow 会被截断或无法表达;模型是 cell offset,不是 byte address。 | 对 `tl.advance`、pointer subtraction、block-pointer rewind 写明 `BlockPtr.AdvanceNonnegative` 和 bounds obligation。 |
| total memory read | `readMem` / `readMemAs` 在 dtype mismatch 或非 Real cell 上返回 0;`readMemValue` mismatch 时返回 dtype default。 | malformed load 可能被静默 totalize,而不体现失败或硬件 undefined 行为。 | 优先使用 typed-load 假设、`RegionTyping` / `Kernel.checkStrict`、显式 `TensorView.loaded` 假设。 |
| store `⊥` | 浮点 store 会把 `⊥` 通过 finite fallback 降成 `0`。 | 本应 impossible、undefined、NaN 或 `-inf` 的路径可能变成普通 zero write。 | headline proof 应证明被 store 的浮点值是 finite,或者显式把 fallback 写成模型假设。 |
| atomic / concurrency | atomic 支持限于窄的 algorithm-level marker 和显式 trace / linearization relation。 | 没有完整 scheduler、memory ordering、warp/block interleaving、barrier、shared memory、async 或 TMA 语义。CAS 比较的是数学 `MemCell` 相等,不是硬件 bit pattern。 | atomic theorem 应理解为抽象 RMW / linearization theorem,不是 CUDA memory-model theorem。 |
| block-pointer metadata | runtime block-pointer 地址计算仍通过 list `getD` default 保持 total;可选 checker 会拒绝 rank 不一致的 metadata 和静态可见的 `tl.advance` underflow,并通过统一的 `BlockPtrSummary` 在简单 block-pointer expression / register 间传播 region、可选 parent rank、可选静态 offsets。theorem-side predicate 是 `BlockPtr.WellFormed`、`BlockPtr.CheckedAxesValid`、`BlockPtr.AdvanceNonnegative`。 | 动态 block pointer 仍可能需要证明假设;未经过 checker 的 malformed metadata 在模型里可能比真实 Triton IR 更 defined。 | 尽量使用 `Kernel.checkStrict`;block-pointer consumer 的 theorem statement 里携带 `BlockPtr.WellFormed` / `BlockPtr.CheckedAxesValid` / `BlockPtr.AdvanceNonnegative`。 |

局部 bridge theorem 已把 checker success 接到这些 predicate:
`checkBlockPtrMetadata_ok`、`checkBoundaryAxes_ok`、
`checkStaticAdvanceNonnegative_ok`。
summary 层也有对应的局部 bridge:
`BlockPtrSummary.ofStaticChecked_ok`、
`BlockPtrSummary.ofDynamicOffsetsChecked_ok`、
`BlockPtrSummary.checkedAdvance_ok`、
`BlockPtrSummary.checkBoundary_ok`。
后续新增局部 checker 义务时沿用同一分层:一个共享 executable Bool helper,一个面向
theorem statement 的 Prop contract,以及一条从 checker success 到 Prop contract 的
`_ok` theorem。

## Review Checklist

把一个 theorem 称为 "faithful to Triton" 而不是 "VeriTile Real 语义下正确"
之前,至少检查:

- 每个 `log`、`sqrt`、`rsqrt`、division、`pow` 使用都有对应 domain 前提。
- 每个 fixed-width integer 或 quantization claim 都有 range/width 前提,否则只停留在数学 `Int` / `Nat` 层。
- 每个可能后退的 pointer 或 block-pointer offset 都有非负 / in-bounds obligation;
  signed block-pointer advance 优先写成 `BlockPtr.AdvanceNonnegative`。
- 证明没有依赖 dtype-mismatch read 返回 0。
- 证明没有依赖 store `⊥` 为 0,除非这正是 statement 里的模型 contract。
- atomic / whole-grid theorem 写清楚自己用的是 disjoint-frame merge、atomic-add sum merge,还是显式 RMW linearization。

## 什么时候应该强化模型

当某个 consumer 的 claim 不能诚实地表达为 Real-semantics theorem 加外部 gap
contract 时,就该考虑强化内部语义。典型情况:

- IEEE 或 mixed-precision theorem 的结论依赖 rounding magnitude;
- quantization theorem 的结论依赖 fixed-width wraparound 或 saturation;
- pointer-heavy theorem 的正确性依赖 byte addressing 或负 offset;
- concurrent kernel 的正确性依赖 memory ordering、barrier、async/TMA completion 或 scheduler interleaving。
