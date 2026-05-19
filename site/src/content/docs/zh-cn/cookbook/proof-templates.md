---
title: 证明模板
description: 标准 1D scatter、双通道写、单步循环、helper 地图、什么时候用什么。
---

到现在,bench 语料的证明模式稳定在几个模板上。本页把每个 pattern 映射到
`Semantics/Scalar.lean`、`Semantics/State.lean`、
`VeriTile/Triton/LoopInvariant.lean` 里的 helper,并给出可直接抄的骨架。

:::caution[时效性]
下文列出的 helper 名最近一次盘点是 2026-05-17。**lemma 名是稳定的,但
issue 编号和进度列表不是** —— 本页讲 pattern,bench 覆盖现状看
[项目状态](/VeriTile/zh-cn/status/)。
:::

## 标准 1D scatter 证明

最常见的形状:kernel 算出每 lane 值并通过单射 offset 函数 store。Helper:
`BlockState.scatter_readback_prop_masked_nd`。

```lean
intro i
simp [exec, KERNEL_NAME, stepStmts, stepStmt, evalOp, Option.bind, ...] at hExec
rw [← hExec]
simp only [outOffsetDef]
rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
      (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
by_cases h : i.val < N
· simp [BlockState.pid_eq, specDef, inOffsetDef, h]
· simp [BlockState.pid_eq, h]
```

Injectivity witness 从下表挑:

| Offset 形状 | Helper |
|---|---|
| `fun idx : TileIndex [BLOCK] => base + idx.1.val` | `tileIndex1d_base_offset_injective` |
| `+ idx.1.val * stride`(需 `stride ≠ 0`) | `tileIndex1d_base_strided_offset_injective` |
| 裸 `fun idx => idx.1.val` | `tileIndex1d_offset_injective` |
| 2D 行主:`+ idx.1.val * Nstride + idx.2.1.val`(需 `N ≤ Nstride`) | `tileIndex2d_base_row_major_injective` |
| 2D 完全 strided:`+ idx.1.val * Mstride + idx.2.1.val * Nstride` | `tileIndex2d_base_strided_injective` |
| 2D 非内侧 softmax 风格 | `nonInnerOffset_injective` |

### Explicit-trace 变体

当 `simp` 留下一个未化简的 `foldl`,简单形式 match 不上时,在
`scatter_readback_*` 的 `s :=` 参数里**显式**构造 `setReg`-trace 状态。
`fused_rotary_embedding` 的 Q first-half 证明或 `decoding_*_first_half_correct`
是参考实现。

## 双通道写(real + nat/int 索引)

Kernel 同时写一个 `.real` 值和一个 `.nat` / `.int` 索引到两个不同区域
(典型是 argmax 的 `kernel_1`)。

Pattern:

1. 加 region 不重合假设:
   ```lean
   hRegions : value_region ≠ (Region.cast index_region : RegionName)
   ```
2. **值通道**:`cases hExec` 后,把索引写 strip 掉:
   ```lean
   rw [BlockState.writeMemTyped_nat_readMem_of_ne _ _ _ _ _ _
         (by intro ⟨h1, _⟩; exact hRegions h1)]
   ```
   然后 `simp; congr`。
3. **索引通道**:平凡:
   ```lean
   simp [BlockState.writeMemTyped_nat_readMemValue_nat]
   ```

`Semantics/State.lean` 里的 bridge lemma
`BlockState.writeMemTyped_int_readMem_of_ne` 和
`writeMemTyped_nat_readMem_of_ne` 让这套干净 —— typed nat/int 写不打扰
不重合地址的 real-channel `readMem`。

## 循环不变量证明

Kernel body 被 `for` / `tl.for` 包住。用
[`forLoop_inv`](https://github.com/Lizn-zn/VeriTile/blob/main/VeriTile/Triton/LoopInvariant.lean)
或它的兄弟。

### DSL → AST → helper 映射

| DSL 形式 | AST 形式 | Helper |
|---|---|---|
| `tl.for i in $(n) { ... }` | `Stmt.forLoop idx n body` | `forLoop_inv` |
| `for i in range($(s), $(t), $(step)) { ... }` | `Stmt.forRange idx s t step body` | `forRange_inv` |
| `for i in range(expr)` / `range(e1, e2)` | `Stmt.forRangeDyn ...` | `evalOp` 化简到静态 stop,再 `forRange_inv` |

### Loop readout corollaries

证明只需要在 loop 结束时读回某个 register:

- `forLoop_readout_scalar` / `forLoop_readout_tile`
- `forRange_readout_scalar` / `forRange_readout_tile`

后置条件形如"`n` 处那个 register 持有值 `v`"时,用 readout 而不是完整
`_inv` —— 管线更短。

### Aux"起点参数化"形式

`forLoopAux_inv`、`forRangeAux_inv` —— 起点是参数而不是 `0`。kernel
循环不从 0 开始、或归纳证明需要处理子区间时用。

### Spec 文档

详细语义在
[`documents/ForLoopInvDesign.md`](https://github.com/Lizn-zn/VeriTile/blob/main/documents/ForLoopInvDesign.md)
§4.1 / §4.2 / §4.3。bench 文件
[`DiagSsmTriton`](https://github.com/Lizn-zn/VeriTile/tree/main/bench/tritonbench_g/diag_ssm_triton)、
[`MeanReduction`](https://github.com/Lizn-zn/VeriTile/tree/main/bench/tritonbench_g/mean_reduction)、
[`EmbeddingTritonKernel`](https://github.com/Lizn-zn/VeriTile/tree/main/bench/tritonbench_g/embedding_triton_kernel)
在生产中用这套 API,是好参考。

## 单次迭代循环

Python 测试只跑一个 step 对齐的 chunk(`start < stop ≤ start + step`)
时,用 `LoopInvariant.lean` 里的 `forRange_single_step` /
`forRangeDyn_single_step`。它们把循环坍缩成一次 body 执行,不需要归纳的
`P`。

适用:

- `var_len_copy`,length ≤ `BLOCK_SIZE` 时。
- 单 block 的 rmsnorm / layernorm。
- 维度特化的 argmax,N ≤ `BLOCK_N` 时。

```text
forRangeDyn_single_step
  (hStart : evalOp startOp s_init = some (Tile.scalar start))
  (hStop  : evalOp stopOp  s_init = some (Tile.scalar stop))
  (hStepOp: evalOp stepOp  s_init = some (Tile.scalar step))
  (hstep : step ≠ 0) (hlt : start < stop) (hle : stop ≤ start + step)
  (hBody : stepStmts body (s_init.setReg idx .nat [] (Tile.scalar start)) = some s_body) :
  stepStmt (.forRangeDyn idx startOp stopOp stepOp body) s_init = some s_body
```

具体应用还得证 `hBody`(body 对寄存器/内存的影响) —— helper 只省了循环
unfold 那一段。

## 跨区域多 store strip

两个 store 写**不同**区域,被合在一个 trace 里,想通过另一区域 readback:
`BlockState.foldl_writeMem_const_region_prop_masked_readMem_other`
(`Semantics/State.lean`)。

```lean
rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
      (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
rw [BlockState.foldl_writeMem_const_region_prop_masked_readMem_other
      <other_region> _ _ _ _ _ _ _ <h_ne>]
by_cases hi : <mask cond>
· simp [hi, <spec>, <offset>]
· simp [hi]
```

`_` 让 Lean 从外层 foldl 反推 `offsetFn` / `valueFn` / `P` / `l` / `s` /
`off`。不重合假设 `<h_ne>`(通常是定理签名里 `hRegions : R ≠ other_region`)
是关键前提。`adam_update_triton/AdamUpdateTriton.lean` 是参考。

## Bridge lemma:`tl.maximum` / `tl.where(>)`

`tl.maximum` 和带 Bool 条件的 `tl.where(cond, a, b)` 需要 Bool↔Prop 桥接。
`Semantics/Scalar.lean` 里的标准 helper:

- `ComparableDType.{gt,lt,ge,le,eq,ne}_eq_true` —— Bool↔Prop 桥接。
- `ComparableDType.real_gt_some_some_eq_true_iff`(kldiv_ops) —— 当
  classical `Decidable` 会被烤进去时,保持 Bool-form decode 可化简。

下一页 [`forLoop_inv` 陷阱](/VeriTile/zh-cn/cookbook/forloop-pitfalls/) 收集应用
这些模板时反复栽跟头的 tactical 坑。
