# 支持的 Triton 子集

本文档记录 VeriTile 当前嵌入到 Lean 中的 Triton 子集, 以及暂未支持的部分.

## 当前支持

- `tl.load($(REGION) + offs)` 与 `tl.store($(REGION) + offs, value)`,
  支持 gather/scatter tile offset —— surface syntax 模仿 Triton 的
  pointer-plus-offset 形式. 单 cell 糖 `tl.load($(REGION))` /
  `tl.store($(REGION), value)` 展开成 offset `0`.
  注: `$(REGION)` 是 Lean `RegionName` 项,**不是** CUDA 指针;
  详见 [RP1](../Notes/research_problem_pointer_vs_named_region.md).
- `tl.arange(N)` 与 `tl.arange(start, end)`
- `tl.broadcast`, `tl.full`
- 标量 `ℝ` 常量(数字字面量)与 `Nat` 地址常量(`$(...)` 反引用);
  `ℝ` / `Nat` 双通道见 [RP2](../Notes/research_problem_address_typing.md).
- `tl.exp`, `tl.log`, `tl.maximum`
- 算术: `+`, `-`, `*`, `/`(按载体类型 dispatch —— 仅允许
  `ℝ × ℝ` 与 `Nat × Nat`,混型表达式属于 `evalOp` 错误)
- reduction: `tl.max`, `tl.sum`(仅作用于 `ℝ` tile)
- `tl.program_id`, `tl.constexpr`
- 赋值与 `tl.store` 语句

## 已知限制

- `forLoop` 已在 AST 中, 但尚未实现操作语义.
- `tl.dot`, atomic, async copy, masking, 多 block 执行属于后续工作.
- 浮点算术目前建模为 `ℝ`; IEEE-754 保真不在当前证明模型范围内.
- 内存 region(`$(REGION)`)是静态命名的,**不是 first-class 指针** ——
  `if cond then xReg else yReg` 这种风格无法表达
  ([RP1](../Notes/research_problem_pointer_vs_named_region.md)).
