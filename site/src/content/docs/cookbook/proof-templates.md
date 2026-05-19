---
title: Proof templates
description: Standard 1D scatter, dual-channel store, single-step loops, the helper map, and when to reach for what.
---

By now the proof patterns for the bench corpus have stabilized into a
handful of templates. This page maps each pattern to the helpers in
`Semantics/Scalar.lean`, `Semantics/State.lean`, and
`VeriTile/Triton/LoopInvariant.lean`, and gives a skeleton you can copy.

:::caution[Currency]
The named helpers below were last surveyed 2026-05-17. Specific lemma
names are stable but **issue numbers and progress lists are not** —
this page is about the patterns, not the bench-wide closure state. See
[Project status](/VeriTile/status/) for current coverage.
:::

## Standard 1D scatter proof

The most common shape: kernel computes a per-lane value and stores it
through an injective offset function. Helper:
`BlockState.scatter_readback_prop_masked_nd`.

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

The injectivity-witness comes from one of the standard injection helpers:

| Offset shape | Helper |
|---|---|
| `fun idx : TileIndex [BLOCK] => base + idx.1.val` | `tileIndex1d_base_offset_injective` |
| `+ idx.1.val * stride` (needs `stride ≠ 0`) | `tileIndex1d_base_strided_offset_injective` |
| Bare `fun idx => idx.1.val` | `tileIndex1d_offset_injective` |
| 2D row-major: `+ idx.1.val * Nstride + idx.2.1.val` (needs `N ≤ Nstride`) | `tileIndex2d_base_row_major_injective` |
| 2D fully strided: `+ idx.1.val * Mstride + idx.2.1.val * Nstride` | `tileIndex2d_base_strided_injective` |
| 2D non-inner softmax-style | `nonInnerOffset_injective` |

### Explicit-trace variant

When `simp` leaves an unsimplified `foldl` and the simple form doesn't
match, build the explicit `setReg`-trace state in `scatter_readback_*`'s
`s :=` argument. Look at `fused_rotary_embedding`'s Q first-half proof or
`decoding_*_first_half_correct` for working examples.

## Dual-channel store (real + nat/int index)

For kernels writing both a `.real` value and a `.nat` / `.int` index to
two different regions (the canonical case is argmax's `kernel_1`).

The pattern:

1. Add a regions-distinct hypothesis:
   ```lean
   hRegions : value_region ≠ (Region.cast index_region : RegionName)
   ```
2. **Value channel** proof — after `cases hExec`, strip the index write:
   ```lean
   rw [BlockState.writeMemTyped_nat_readMem_of_ne _ _ _ _ _ _
         (by intro ⟨h1, _⟩; exact hRegions h1)]
   ```
   then `simp; congr`.
3. **Index channel** proof — trivial:
   ```lean
   simp [BlockState.writeMemTyped_nat_readMemValue_nat]
   ```

The bridge lemmas
`BlockState.writeMemTyped_int_readMem_of_ne` and
`writeMemTyped_nat_readMem_of_ne` in `Semantics/State.lean` are what make
this clean — a typed nat/int write doesn't disturb the real-channel
`readMem` at disjoint addresses.

## Loop-invariant proofs

The kernel's body is wrapped in a `for` / `tl.for`. Use
[`forLoop_inv`](https://github.com/Lizn-zn/VeriTile/blob/main/VeriTile/Triton/LoopInvariant.lean)
or its siblings.

### DSL → AST → helper map

| DSL form | AST form | Helper |
|---|---|---|
| `tl.for i in $(n) { ... }` | `Stmt.forLoop idx n body` | `forLoop_inv` |
| `for i in range($(s), $(t), $(step)) { ... }` | `Stmt.forRange idx s t step body` | `forRange_inv` |
| `for i in range(expr)` / `range(e1, e2)` | `Stmt.forRangeDyn ...` | `evalOp`-reduce to static stop, then `forRange_inv` |

### Loop readout corollaries

When the proof only needs to read a register back at loop completion:

- `forLoop_readout_scalar` / `forLoop_readout_tile`
- `forRange_readout_scalar` / `forRange_readout_tile`

Use these instead of the full `_inv` form when the postcondition is "the
register at index `n` holds value `v`". Less plumbing.

### Auxiliary "start-quantified" forms

`forLoopAux_inv`, `forRangeAux_inv` — the start index is a parameter
instead of `0`. Use when the kernel's loop doesn't start at 0 or when an
inductive proof needs to handle a sub-range.

### Spec document

Detailed semantics in
[`documents/ForLoopInvDesign.md`](https://github.com/Lizn-zn/VeriTile/blob/main/documents/ForLoopInvDesign.md)
§4.1 / §4.2 / §4.3. The bench files
[`DiagSsmTriton`](https://github.com/Lizn-zn/VeriTile/tree/main/bench/tritonbench_g/diag_ssm_triton),
[`MeanReduction`](https://github.com/Lizn-zn/VeriTile/tree/main/bench/tritonbench_g/mean_reduction),
and
[`EmbeddingTritonKernel`](https://github.com/Lizn-zn/VeriTile/tree/main/bench/tritonbench_g/embedding_triton_kernel)
all use the API in production and are good worked references.

## Single-iteration loops

When the Python test only drives a single step-aligned chunk
(`start < stop ≤ start + step`), use
`forRange_single_step` / `forRangeDyn_single_step` from
`LoopInvariant.lean`. These collapse the loop to a single body execution
without needing an inductive `P`.

Applies to:

- `var_len_copy` when length ≤ `BLOCK_SIZE`.
- Single-block rmsnorm / layernorm.
- Dim-specific argmax with N ≤ `BLOCK_N`.

```text
forRangeDyn_single_step
  (hStart : evalOp startOp s_init = some (Tile.scalar start))
  (hStop  : evalOp stopOp  s_init = some (Tile.scalar stop))
  (hStepOp: evalOp stepOp  s_init = some (Tile.scalar step))
  (hstep : step ≠ 0) (hlt : start < stop) (hle : stop ≤ start + step)
  (hBody : stepStmts body (s_init.setReg idx .nat [] (Tile.scalar start)) = some s_body) :
  stepStmt (.forRangeDyn idx startOp stopOp stepOp body) s_init = some s_body
```

Per-kernel application still requires proving `hBody` (the body's effect
on registers and memory) — the helper just eliminates the for-loop
unfolding plumbing.

## Multi-store cross-region strip

When two stores to **different regions** are folded into one trace and you
want to read back through the other:
`BlockState.foldl_writeMem_const_region_prop_masked_readMem_other`
(`Semantics/State.lean`).

```lean
rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
      (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
rw [BlockState.foldl_writeMem_const_region_prop_masked_readMem_other
      <other_region> _ _ _ _ _ _ _ <h_ne>]
by_cases hi : <mask cond>
· simp [hi, <spec>, <offset>]
· simp [hi]
```

The `_`s let Lean infer `offsetFn` / `valueFn` / `P` / `l` / `s` / `off`
from the outer foldl. The disjointness hypothesis `<h_ne>` (typically
`R ≠ other_region` passed as `hRegions` in the theorem signature) is the
key prerequisite. `adam_update_triton/AdamUpdateTriton.lean` is the
worked reference.

## Bridge lemmas: `tl.maximum`, `tl.where(>)`

`tl.maximum` and `tl.where(cond, a, b)` with a Bool condition need Bool↔Prop
plumbing. Standard helpers in `Semantics/Scalar.lean`:

- `ComparableDType.{gt,lt,ge,le,eq,ne}_eq_true` — Bool↔Prop bridge.
- `ComparableDType.real_gt_some_some_eq_true_iff` (kldiv_ops) — keeps
  Bool-form decode tractable when classical `Decidable` would otherwise
  bake in.

The next page, [`forLoop_inv` pitfalls](/VeriTile/cookbook/forloop-pitfalls/),
collects the tactical traps that recur when applying these templates.
