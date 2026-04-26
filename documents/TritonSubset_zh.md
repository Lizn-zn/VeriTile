# 支持的 Triton 子集

本文档记录 VeriTile 当前嵌入到 Lean 中的 Triton 子集, 以及暂未支持的部分.

## 当前支持

- `tl.load`, `tl.store`, 支持 gather/scatter tile offset
- `tl.arange`, `tl.broadcast`, `tl.full`
- 标量和张量常量
- `tl.exp`, `tl.log`, `tl.maximum`
- 算术: `+`, `-`, `*`, `/`
- reduction: `tl.max`, `tl.sum`
- `tl.program_id`, `tl.constexpr`
- 赋值和 store 语句

## 已知限制

- `forLoop` 已在 AST 中, 但尚未实现操作语义.
- `tl.dot`, atomic, async copy, masking, 多 block 执行属于后续工作.
- 浮点算术目前建模为 `Real`; IEEE-754 保真不在当前证明模型范围内.
