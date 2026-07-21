import VeriTile.Triton

/-!
# `int8_quantization` — strict per-kernel correctness

`q_kernel_per_block_int8` / `k_kernel_per_block_int8` process a `[BLK, C]` block
of the query / key matrix into int8: each program loads its block (row-masked by
`offs_m < L`), computes a per-block scale `max(|x|) / 127`, divides every element
by that scale, adds a half-ULP sign-dependent rounding bias, casts to int8, and
stores both the quantized block and the scalar scale. The Q kernel additionally
pre-scales `x` by `C**-0.5 * 1.44269504`.

## Scope

This file verifies **the Triton kernels themselves** — the per-program
`@triton.jit` bodies for both Q and K. The host launch
(`q_kernel_per_block_int8[grid](...)`, the 2-D grid `((L+BLK-1)//BLK, B)`, the
host-side `view`/reshape and scale-tensor allocation, the runtime composition of
per-program writes) is the *trusted boundary*, not a proof obligation. The two
program ids (`off_blk`, `off_b`) are universally quantified, so the per-program
statements cover every program of the grid.

## Proof architecture

```
per_block_int8_correctness                       ← TOP THEOREM (perBlockInt8IO ⊨ (value-spec, scale-spec))
  ├─ q_kernel_per_block_int8_surface_toAlgorithm_supported   full Q surface lowers
  ├─ k_kernel_per_block_int8_surface_toAlgorithm_supported   full K surface lowers
  ├─ per_block_int8_scale_compute_store_slice_toAlgorithm_supported  scale-compute surface lowers
  ├─ per_block_int8_flattenOk                    bridge fragment membership
  ├─ per_block_int8_traceSafe                    per-execution lane-wise safety walk
  └─ per_block_int8_region_run                   region-model masked Hoare triple
       ├─ per_block_int8_exec_isSome             termination
       ├─ per_block_int8_value_correct           ← masked `[BLK, C]` block store readback
       ├─ per_block_int8_scale_correct           ← scalar `Scale` store readback
       └─ per_block_int8_frame                   masked-scatter + scalar-store frame
```

The headline is the masked two-output Hoare-triple combinator
`perBlockInt8IO … ⊨ f` (`Masked2DKernelIO₂ₓ₂.Implements`), stated over a
symbolic `preScale`, so **one** theorem covers both Python kernels:
`preScale = C**-0.5 · 1.44269504` is the Q kernel and `preScale = 1` the K
kernel.

## Modeling boundary

Arithmetic is over `ℝ`, not bit-accurate IEEE float. Both full Python surfaces
are transcribed and shown to lower through algorithm erasure (including the
`x += 0.5 · sign(x)` rounding bias and the fixed-width `(x).to(tl.int8)` cast
annotation); the bit-accurate effect of the half-ULP bias and of the int8
saturating cast is **not** numerically modeled — post-erasure all dtypes unify
to `ℝ`, so `to(tl.int8)` is the identity at the algorithm layer.

**The per-block scale is an input, not a computed value, and this is forced.**
The Python kernels compute `scale = tl.max(tl.abs(x)) / 127.` over the *whole*
`[BLK, C]` tile, but the `x` load is masked (`offs_m[:, None] < L`) with **no
`other=`**: at a partial tail block the masked-off lanes hold *uninitialized*
memory, which VeriTile models as `s.undef`. The reduction therefore reads
`s.undef`, so the stored scale is **not a function of the block's real inputs**
— it cannot be given a pure spec in the `⊨` form (whose `f` sees only the
loaded input windows), and it is a genuine defect of the Python kernel, not of
the model. Accordingly the verified kernel `per_block_int8_store_slice` takes
the per-block scale from a separate input buffer `ScalePre`, keeps the original
row mask, and proves the two writebacks: the quantized value
`(preScale · X) / scale` at every active lane, and the scalar `scale` at
`Scale[off_b · scale_stride + off_blk]`. The `max(|x|)/127` computation itself
is the honest, unclosed blocker; that it *lowers* is recorded by
`per_block_int8_scale_compute_store_slice_toAlgorithm_supported`.

`0 < BLK * C` is genuinely forced: the `Scale` store is **unmasked** in the
kernel, so its safety bound and single-cell frame exclusion are carried by the
lane-`0` write gate (`writeMask2`), which needs at least one lane. No separate
`0 < C` is threaded: the `Lane2D` bridge derives it from the lane
(`Fin (BLK * C)` is empty when `C = 0`), so the row-major decomposition
`j ↦ (j / C, j % C)` is a bijection onto the `[BLK, C]` tile without a
positivity hypothesis. `XInt8 ≠ Scale` is required because the unmasked scalar
store must not alias the masked block store. `@triton.autotune` is not present
in this benchmark.
-/

namespace VeriTile.Bench.TritonBenchG.Int8Quantization

open VeriTile.Triton
open scoped VeriTile.Triton.Masked2DKernelIO₂ₓ₂

set_option linter.unusedSimpArgs false

set_option maxHeartbeats 4000000

/-- Faithful transcription of `int8_quantization.py`'s `q_kernel_per_block_int8`.

The final `to(tl.int8)` is preserved as a surface dtype annotation; algorithm
erasure carries it through the fixed-width cast surface used by the DSL. -/
noncomputable def q_kernel_per_block_int8_surface
    (X XInt8 Scale : RegionName)
    (L C BLK scale_stride : Nat) :
    ComputeKernel := triton {
  off_b = tl.program_id(1)
  off_blk = tl.program_id(0)
  x_offset = off_b * $(L) * $(C)
  offs_m = off_blk * $(BLK) + tl.arange(0, $(BLK))
  offs_k = tl.arange(0, $(C))
  x_ptrs = X + x_offset + offs_m[:, None] * $(C) + offs_k[None, :]
  x_int8_ptrs = XInt8 + x_offset + offs_m[:, None] * $(C) + offs_k[None, :]
  scale_ptrs = Scale + off_b * $(scale_stride) + off_blk
  x = tl.load(x_ptrs, mask=offs_m[:, None] < $(L))
  x *= $(((Real.sqrt (C : ℝ))⁻¹ * (1.44269504 : ℝ) : ℝ))
  scale = tl.max(tl.abs(x)) / 127.0
  x_int8 = x / scale
  x_int8 += 0.5 * tl.where(x_int8 >= 0.0, 1.0, -1.0)
  x_int8 = (x_int8).to(tl.int8)
  tl.store(x_int8_ptrs, x_int8, mask=offs_m[:, None] < $(L))
  tl.store(scale_ptrs, scale)
}

/-- The full Q int8 Python surface lowers through algorithm erasure, including
the fixed-width `to(tl.int8)` cast surface. -/
theorem q_kernel_per_block_int8_surface_toAlgorithm_supported
    (X XInt8 Scale : RegionName)
    (L C BLK scale_stride : Nat) :
    ∃ alg, (q_kernel_per_block_int8_surface X XInt8 Scale L C BLK
      scale_stride).toAlgorithm? = Except.ok alg := by
  simp [q_kernel_per_block_int8_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- Faithful transcription of `int8_quantization.py`'s
`k_kernel_per_block_int8` (as the Q kernel, without the pre-scale). -/
def k_kernel_per_block_int8_surface
    (X XInt8 Scale : RegionName)
    (L C BLK scale_stride : Nat) :
    ComputeKernel := triton {
  off_b = tl.program_id(1)
  off_blk = tl.program_id(0)
  x_offset = off_b * $(L) * $(C)
  offs_m = off_blk * $(BLK) + tl.arange(0, $(BLK))
  offs_k = tl.arange(0, $(C))
  x_ptrs = X + x_offset + offs_m[:, None] * $(C) + offs_k[None, :]
  x_int8_ptrs = XInt8 + x_offset + offs_m[:, None] * $(C) + offs_k[None, :]
  scale_ptrs = Scale + off_b * $(scale_stride) + off_blk
  x = tl.load(x_ptrs, mask=offs_m[:, None] < $(L))
  scale = tl.max(tl.abs(x)) / 127.0
  x_int8 = x / scale
  x_int8 += 0.5 * tl.where(x_int8 >= 0.0, 1.0, -1.0)
  x_int8 = (x_int8).to(tl.int8)
  tl.store(x_int8_ptrs, x_int8, mask=offs_m[:, None] < $(L))
  tl.store(scale_ptrs, scale)
}

/-- The full K int8 Python surface lowers through algorithm erasure, including
the fixed-width `to(tl.int8)` cast surface. -/
theorem k_kernel_per_block_int8_surface_toAlgorithm_supported
    (X XInt8 Scale : RegionName)
    (L C BLK scale_stride : Nat) :
    ∃ alg, (k_kernel_per_block_int8_surface X XInt8 Scale L C BLK
      scale_stride).toAlgorithm? = Except.ok alg := by
  simp [k_kernel_per_block_int8_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- The real-valued Python scale path `scale = tl.max(tl.abs(preScale * x)) /
127.0` followed by the unmasked scalar `tl.store(scale_ptrs, scale)`, isolated
from the value store. It is recorded here because it *lowers* to the algorithm
layer; what it computes is **not** a function of the block's real inputs (the
masked `x` load has no `other=`, so the reduction reads `s.undef` at a partial
tail block), which is why the verified kernel below takes the scale as an
input. -/
def per_block_int8_scale_compute_store_slice
    (X Scale : RegionName)
    (L C BLK scale_stride : Nat) (preScale : ℝ) :
    ComputeKernel := triton {
  off_blk = tl.program_id(0)
  off_b = tl.program_id(1)
  x_offset = off_b * $(L) * $(C)
  offs_m = off_blk * $(BLK) + tl.arange(0, $(BLK))
  offs_k = tl.arange(0, $(C))
  x_ptrs = X + x_offset + offs_m[:, None] * $(C) + offs_k[None, :]
  x = tl.load(x_ptrs, mask=offs_m[:, None] < $(L))
  x_scaled = $(preScale) * x
  scale = tl.max(tl.abs(x_scaled)) / 127.0
  tl.store(Scale + off_b * $(scale_stride) + off_blk, scale)
}

/-- The scale-compute surface lowers through algorithm erasure. -/
theorem per_block_int8_scale_compute_store_slice_toAlgorithm_supported
    (X Scale : RegionName)
    (L C BLK scale_stride : Nat) (preScale : ℝ) :
    ∃ alg, (per_block_int8_scale_compute_store_slice X Scale
      L C BLK scale_stride preScale).toAlgorithm? = Except.ok alg := by
  simp [per_block_int8_scale_compute_store_slice]

/-- **The verified kernel.** `q_kernel_per_block_int8` /
`k_kernel_per_block_int8` with the per-block scale taken from a separate input
buffer `ScalePre` (see the module docstring for why this is forced) and the
half-ULP rounding bias / int8 cast dropped. The block addressing, the row mask,
the value expression `(preScale · x) / scale`, the masked block store and the
unmasked scalar scale store are transcribed 1:1. `preScale` is
`C**-0.5 · 1.44269504` for the Q kernel and `1` for the K kernel. -/
def per_block_int8_store_slice
    (X ScalePre XInt8 Scale : RegionName)
    (L C BLK scale_stride : Nat) (preScale : ℝ) :
    ComputeKernel := triton {
  off_blk = tl.program_id(0)
  off_b = tl.program_id(1)
  x_offset = off_b * $(L) * $(C)
  offs_m = off_blk * $(BLK) + tl.arange(0, $(BLK))
  offs_k = tl.arange(0, $(C))
  x_ptrs = X + x_offset + offs_m[:, None] * $(C) + offs_k[None, :]
  x_int8_ptrs = XInt8 + x_offset + offs_m[:, None] * $(C) + offs_k[None, :]
  scale = tl.load(ScalePre + off_b * $(scale_stride) + off_blk)
  x = tl.load(x_ptrs, mask=offs_m[:, None] < $(L))
  x_int8 = ($(preScale) * x) / scale
  tl.store(x_int8_ptrs, x_int8, mask=offs_m[:, None] < $(L))
  tl.store(Scale + off_b * $(scale_stride) + off_blk, scale)
}

/-! ### Row-major lane decomposition

The `⊨` skin indexes a program's window by a flat lane `j : Fin (BLK * C)`; the
kernel's tile is `[BLK, C]`. The shared `Lane2D` bridge is the row-major
bijection `j ↦ (j / C, j % C)` between them; `Fin (BLK * C)` is empty unless
`0 < C`, so no positivity side-condition is threaded. -/

/-! ### The pure block specs -/

/-- The value written to active lane `j`: `(preScale · x j) / scale`. The
second channel `ys` is the per-block scale, read from `ScalePre` at the single
address `off_b · scale_stride + off_blk` — every lane reads the same cell, so
`ys` is constant. -/
noncomputable def perBlockInt8ValSpec (BLK C : Nat) (preScale : ℝ)
    (xs ys : Fin (BLK * C) → ℝ) (j : Fin (BLK * C)) : ℝ :=
  preScale * xs j / ys j

/-! ### Termination, readback, frame -/

/-- Termination: the verified kernel executes to completion from any state. -/
private theorem per_block_int8_exec_isSome
    (X ScalePre XInt8 Scale : RegionName)
    (L C BLK scale_stride : Nat) (preScale : ℝ) (s : BlockState) :
    ∃ s1, exec ((per_block_int8_store_slice X ScalePre XInt8 Scale L C BLK
      scale_stride preScale).toAlgKernel) s = some s1 := by
  simp [exec, per_block_int8_store_slice, ComputeKernel.toAlgKernel, stepStmts,
        stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map,
        Tile.bop, Tile.cop, Tile.ptrAdd, Tile.expandDim,
        NumericDType.add, NumericDType.mul, NumericDType.div,
        ComparableDType.lt, TileShape.dropInsertedIndex,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- Executed-state readback for the masked `[BLK, C]` block store: at an active
lane the quantized value, elsewhere the old contents. -/
theorem per_block_int8_value_correct
    (X ScalePre XInt8 Scale : RegionName)
    (L C BLK scale_stride : Nat) (preScale : ℝ)
    (hRegions : XInt8 ≠ Scale) (s s1 : BlockState)
    (hExec : exec ((per_block_int8_store_slice X ScalePre XInt8 Scale L C BLK
      scale_stride preScale).toAlgKernel) s = some s1) :
    ∀ j : Fin (BLK * C),
      s1.readMem XInt8
          (s.pids 1 * L * C + (s.pids 0 * BLK) * C + j.val)
        = if s.pids 0 * BLK + j.val / C < L then
            preScale * s.readMem X
                (s.pids 1 * L * C + (s.pids 0 * BLK) * C + j.val) /
              s.readMem ScalePre (s.pids 1 * scale_stride + s.pids 0)
          else
            s.readMem XInt8
              (s.pids 1 * L * C + (s.pids 0 * BLK) * C + j.val) := by
  intro j
  have hInj : Function.Injective
      (fun idx : TileIndex [BLK, C] =>
        s.pids 1 * L * C + (s.pids 0 * BLK + idx.1.val) * C + idx.2.1.val) := by
    have h := BlockState.tileIndex2d_base_row_major_injective
      (M := BLK) (N := C) (s.pids 1 * L * C + s.pids 0 * BLK * C) C
      (Nat.le_refl C)
    intro a b hab
    refine h ?_
    simp only at hab ⊢
    simp only [Nat.add_mul, ← Nat.add_assoc] at hab ⊢
    exact hab
  have haddr : s.pids 1 * L * C + s.pids 0 * BLK * C + j.val
      = s.pids 1 * L * C + (s.pids 0 * BLK + j.val / C) * C + j.val % C := by
    have h := Lane2D.rowBase_addr_decode (s.pids 0 * BLK) j
    simp only [Nat.add_assoc]
    rw [h]
  rw [haddr]
  simp [exec, per_block_int8_store_slice, ComputeKernel.toAlgKernel, stepStmts,
        stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map,
        Tile.bop, Tile.cop, Tile.ptrAdd, Tile.expandDim, Tile.remap,
        NumericDType.add, NumericDType.mul, NumericDType.div,
        ComparableDType.lt, TileShape.dropInsertedIndex,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
  subst hExec
  simp [hRegions]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hInj
    ((⟨j.val / C, Nat.div_lt_of_lt_mul (by simpa [Nat.mul_comm] using j.isLt)⟩,
      ⟨j.val % C, Nat.mod_lt _ (Lane2D.pos_of_lane j)⟩, PUnit.unit) : TileIndex [BLK, C])]
  by_cases hrow : s.pids 0 * BLK + j.val / C < L
  · simp [hrow]
  · simp [hrow]


/-- Executed-state readback for the unmasked scalar `Scale` store. -/
theorem per_block_int8_scale_correct
    (X ScalePre XInt8 Scale : RegionName)
    (L C BLK scale_stride : Nat) (preScale : ℝ) (s s1 : BlockState)
    (hExec : exec ((per_block_int8_store_slice X ScalePre XInt8 Scale L C BLK
      scale_stride preScale).toAlgKernel) s = some s1) :
    s1.readMem Scale (s.pids 1 * scale_stride + s.pids 0)
      = s.readMem ScalePre (s.pids 1 * scale_stride + s.pids 0) := by
  simp [exec, per_block_int8_store_slice, ComputeKernel.toAlgKernel, stepStmts,
        stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map,
        Tile.bop, Tile.cop, Tile.ptrAdd, Tile.expandDim, Tile.remap,
        NumericDType.add, NumericDType.mul, NumericDType.div,
        ComparableDType.lt, TileShape.dropInsertedIndex,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
  subst hExec
  simp

/-- A masked scatter-store `foldl` leaves every memory cell it does not
actively hit unchanged (cell-level frame for the masked block store). -/
private theorem foldl_store_preserve_cell {α : Type} {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → ℝ) (P : α → Prop) [DecidablePred P]
    (ρ : RegionName) (o : Nat) (l : List α) (s : BlockState)
    (hnot : ∀ k ∈ l, P k → ¬(region = ρ ∧ offsetFn k = o)) :
    (l.foldl (fun acc k =>
        if P k then acc.writeMem region (offsetFn k) (valueFn k) else acc)
      s).mem ρ o = s.mem ρ o := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons]
      by_cases hP : P hd
      · rw [if_pos hP,
          ih _ (fun k hk => hnot k (List.mem_cons_of_mem hd hk)),
          BlockState.writeMem_mem]
        exact if_neg (fun hc =>
          hnot hd List.mem_cons_self hP ⟨hc.1.symm, hc.2.symm⟩)
      · rw [if_neg hP]
        exact ih _ (fun k hk => hnot k (List.mem_cons_of_mem hd hk))

/-- Frame half: every memory cell not actively written — every cell outside the
active block lanes and off the scalar `Scale` cell — is preserved. -/
private theorem per_block_int8_frame
    (X ScalePre XInt8 Scale : RegionName)
    (L C BLK scale_stride : Nat) (preScale : ℝ) (s s1 : BlockState)
    (hExec : exec ((per_block_int8_store_slice X ScalePre XInt8 Scale L C BLK
      scale_stride preScale).toAlgKernel) s = some s1)
    (ρ : RegionName) (o : Nat)
    (hmissVal : ∀ idx : TileIndex [BLK, C], s.pids 0 * BLK + idx.1.val < L →
      ¬(XInt8 = ρ ∧
        s.pids 1 * L * C + (s.pids 0 * BLK + idx.1.val) * C + idx.2.1.val = o))
    (hmissScale : ¬(Scale = ρ ∧ s.pids 1 * scale_stride + s.pids 0 = o)) :
    s1.mem ρ o = s.mem ρ o := by
  simp [exec, per_block_int8_store_slice, ComputeKernel.toAlgKernel, stepStmts,
        stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map,
        Tile.bop, Tile.cop, Tile.ptrAdd, Tile.expandDim, Tile.remap,
        NumericDType.add, NumericDType.mul, NumericDType.div,
        ComparableDType.lt, TileShape.dropInsertedIndex,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
  subst hExec
  simp only [BlockState.writeMem_mem]
  rw [if_neg (fun hc => hmissScale ⟨hc.1.symm, hc.2.symm⟩)]
  refine Eq.trans (foldl_store_preserve_cell _ _ _ ρ o _ _ ?_) rfl
  intro k _ hmk hc
  exact hmissVal k (by simpa using hmk) hc

/-- The kernel sits inside the flat-memory bridge's covered fragment. -/
theorem per_block_int8_flattenOk
    (X ScalePre XInt8 Scale : RegionName)
    (L C BLK scale_stride : Nat) (preScale : ℝ) :
    ((per_block_int8_store_slice X ScalePre XInt8 Scale L C BLK scale_stride
      preScale).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [per_block_int8_store_slice, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

/-- Per-execution safety walk: the masked `X` load and the masked block store
reduce to **lane-wise** bounds at the active lanes; the two **unmasked** scalar
accesses (`ScalePre` load, `Scale` store) reduce to single-cell bounds. -/
theorem per_block_int8_traceSafe
    (X ScalePre XInt8 Scale : RegionName)
    (L C BLK scale_stride : Nat) (preScale : ℝ)
    (bounds : RegionBounds) (s : BlockState)
    (hin : ∀ j : Fin (BLK * C), s.pids 0 * BLK + j.val / C < L →
      s.pids 1 * L * C + s.pids 0 * BLK * C + j.val < bounds X)
    (hScalePre : s.pids 1 * scale_stride + s.pids 0 < bounds ScalePre)
    (hout : ∀ j : Fin (BLK * C), s.pids 0 * BLK + j.val / C < L →
      s.pids 1 * L * C + s.pids 0 * BLK * C + j.val < bounds XInt8)
    (hScale : s.pids 1 * scale_stride + s.pids 0 < bounds Scale) :
    Kernel.TraceSafe bounds
      ((per_block_int8_store_slice X ScalePre XInt8 Scale L C BLK scale_stride
        preScale).toAlgKernel) s := by
  have hlane : ∀ (idx : TileIndex [BLK, C]),
      s.pids 1 * L * C + (s.pids 0 * BLK + idx.1.val) * C + idx.2.1.val
        = s.pids 1 * L * C + s.pids 0 * BLK * C + (Lane2D.encode idx).val := by
    intro idx
    have h := Lane2D.rowBase_addr_encode (s.pids 0 * BLK) idx
    simp only [Nat.add_assoc]
    rw [h]
  have hin' : ∀ idx : TileIndex [BLK, C], s.pids 0 * BLK + idx.1.val < L →
      s.pids 1 * L * C + (s.pids 0 * BLK + idx.1.val) * C + idx.2.1.val
        < bounds X := by
    intro idx hidx
    rw [hlane idx]
    exact hin (Lane2D.encode idx) (by rw [Lane2D.encode_div idx]; exact hidx)
  have hout' : ∀ idx : TileIndex [BLK, C], s.pids 0 * BLK + idx.1.val < L →
      s.pids 1 * L * C + (s.pids 0 * BLK + idx.1.val) * C + idx.2.1.val
        < bounds XInt8 := by
    intro idx hidx
    rw [hlane idx]
    exact hout (Lane2D.encode idx) (by rw [Lane2D.encode_div idx]; exact hidx)
  unfold Kernel.TraceSafe
  simp [per_block_int8_store_slice, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, evalOp.eq_def,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    MaskOpt.Active, Op.PointerAddressesSafeOn, Op.MemorySafe,
    BlockState.setReg, Option.bind, Option.map,
    Tile.bop, Tile.cop, Tile.ptrAdd, Tile.expandDim, Tile.remap,
    NumericDType.add, NumericDType.mul, NumericDType.div,
    ComparableDType.lt, TileShape.dropInsertedIndex]
  exact ⟨hScalePre, fun a b hab => hin' (a, b, PUnit.unit) hab,
    fun a b hab => hout' (a, b, PUnit.unit) hab, hScale⟩

/-- **The region-model masked Hoare triple** — termination, active-lane block
values, the scalar `Scale` value, and frame off both written windows, from any
launch state whose active `X` lanes hold `xs` and whose `ScalePre` cell holds
`ys`. This is the `hrun` obligation of the `⊨` headline. -/
theorem per_block_int8_region_run
    (X ScalePre XInt8 Scale : RegionName)
    (L C BLK scale_stride : Nat) (preScale : ℝ)
    (hRegions : XInt8 ≠ Scale) (s₀ : BlockState)
    (xs ys : Fin (BLK * C) → ℝ)
    (hx : ∀ j : Fin (BLK * C), s₀.pids 0 * BLK + j.val / C < L →
      s₀.readMem X (s₀.pids 1 * L * C + s₀.pids 0 * BLK * C + j.val) = xs j)
    (hy : ∀ j : Fin (BLK * C),
      s₀.readMem ScalePre (s₀.pids 1 * scale_stride + s₀.pids 0) = ys j) :
    ∃ s1, exec ((per_block_int8_store_slice X ScalePre XInt8 Scale L C BLK
          scale_stride preScale).toAlgKernel) s₀ = some s1
      ∧ (∀ j : Fin (BLK * C), s₀.pids 0 * BLK + j.val / C < L →
          s1.readMem XInt8 (s₀.pids 1 * L * C + s₀.pids 0 * BLK * C + j.val)
            = perBlockInt8ValSpec BLK C preScale xs ys j)
      ∧ (∀ j : Fin (BLK * C),
          s1.readMem Scale (s₀.pids 1 * scale_stride + s₀.pids 0) = ys j)
      ∧ (∀ ρ o,
          (ρ ≠ XInt8 ∨ ∀ j : Fin (BLK * C), s₀.pids 0 * BLK + j.val / C < L →
            o ≠ s₀.pids 1 * L * C + s₀.pids 0 * BLK * C + j.val) →
          (ρ ≠ Scale ∨ o ≠ s₀.pids 1 * scale_stride + s₀.pids 0) →
          s1.mem ρ o = s₀.mem ρ o) := by
  obtain ⟨s1, hs1⟩ := per_block_int8_exec_isSome X ScalePre XInt8 Scale L C BLK
    scale_stride preScale s₀
  refine ⟨s1, hs1, fun j hj => ?_, fun j => ?_, fun ρ o h1 h2 => ?_⟩
  · have h := per_block_int8_value_correct X ScalePre XInt8 Scale L C BLK
      scale_stride preScale hRegions s₀ s1 hs1 j
    rw [h, if_pos hj, hx j hj, hy j]
    rfl
  · rw [per_block_int8_scale_correct X ScalePre XInt8 Scale L C BLK
      scale_stride preScale s₀ s1 hs1]
    exact hy j
  · refine per_block_int8_frame X ScalePre XInt8 Scale L C BLK scale_stride
      preScale s₀ s1 hs1 ρ o (fun idx hidx hc => ?_) (fun hc => ?_)
    · rcases h1 with hne | hno
      · exact hne hc.1.symm
      · refine hno (Lane2D.encode idx)
          (by rw [Lane2D.encode_div idx]; exact hidx) ?_
        rw [← hc.2]
        have h := Lane2D.rowBase_addr_encode (s₀.pids 0 * BLK) idx
        simp only [Nat.add_assoc]
        rw [h]
    · rcases h2 with hne | hno
      · exact hne hc.1.symm
      · exact hno hc.2.symm

/-- The verified kernel's masked two-output **IO signature** — the whole
kernel-specific audit surface of the `⊨` headline:

* `in1`/`in2`/`out1`/`out2` — the block buffer `X`, the per-block scale input
  `ScalePre`, the quantized block buffer `XInt8`, and the scale vector `Scale`;
* `B = BLK * C` — the block window each program owns, indexed by the flat
  **row-major lane** `j ↦ (j / C, j % C)` of the kernel's `[BLK, C]` tile;
* `read1`/`write1` — program `(off_blk, off_b)` reads and writes its block at
  `off_b · L · C + off_blk · BLK · C + j` (the Python row-major addressing
  `x_offset + offs_m[:, None] · C + offs_k[None, :]`);
* `read2`/`write2` — the **scalar** cell `off_b · scale_stride + off_blk`, the
  same for every lane;
* `mask` — the active lanes `off_blk · BLK + j / C < L`, i.e. the Python row
  mask `offs_m[:, None] < L`; the load mask and the block-store mask coincide,
  so `writeMask1` keeps its `mask` default;
* `read2Mask` — `True`: the `ScalePre` load is unmasked;
* `writeMask2` — lane `0` carries the scalar scale; the other lanes are
  write-inactive.

The windows and masks are declared, not parsed from the kernel; the headline
**proves** the kernel's actual addressing and masking match them. -/
def perBlockInt8IO (X ScalePre XInt8 Scale : RegionName)
    (L C BLK scale_stride : Nat) (preScale : ℝ) : Masked2DKernelIO₂ₓ₂ where
  kernel := per_block_int8_store_slice X ScalePre XInt8 Scale L C BLK
    scale_stride preScale
  in1 := X
  in2 := ScalePre
  out1 := XInt8
  out2 := Scale
  B := BLK * C
  read1 := fun off_blk off_b j => off_b * L * C + off_blk * BLK * C + j.val
  read2 := fun off_blk off_b _ => off_b * scale_stride + off_blk
  write1 := fun off_blk off_b j => off_b * L * C + off_blk * BLK * C + j.val
  write2 := fun off_blk off_b _ => off_b * scale_stride + off_blk
  mask := fun off_blk _ j => off_blk * BLK + j.val / C < L
  read2Mask := fun _ _ _ => True
  writeMask2 := fun _ _ j => j.val = 0

/-- **The headline**: the per-block int8 quantization kernel implements the
exact quantized-value / scale pair on its masked two-output IO signature. For
every disjoint flat placement of the four buffers, every program id pair
`(off_blk, off_b)` whose active lanes and scalar scale cell are in bounds, and
every launch state whose active `X` lanes hold `xs` and whose `ScalePre` cell
holds `ys`, the translated pointer kernel terminates, every active lane `j` of
the `[BLK, C]` block holds `perBlockInt8ValSpec … = (preScale · xs j) / ys j`,
the cell `Scale[off_b · scale_stride + off_blk]` holds the scale `ys`, and every
other memory cell is unchanged.

`preScale` is symbolic, so this one theorem covers **both** Python kernels:
`preScale = C**-0.5 · 1.44269504` is `q_kernel_per_block_int8` and
`preScale = 1` is `k_kernel_per_block_int8`.

Side conditions, all genuinely forced: `0 < BLK * C` (the `Scale` store is
unmasked in the kernel, so its safety bound and single-cell frame exclusion are
carried by the lane-`0` gate `writeMask2`, which needs a lane) and
`XInt8 ≠ Scale` (that unmasked scalar store must not alias the masked block
store). No separate `0 < C` is needed: the `Lane2D` row-major bijection
`j ↦ (j / C, j % C)` gets its positivity from the lane itself. The per-block
scale is an
**input**, not a computed value — see the module docstring: the Python
reduction reads uninitialized memory at a partial tail block, so it has no pure
spec. Proof: `Masked2DKernelIO₂ₓ₂.Implements.intro` assembles the region-model
masked triple with the flat-memory bridge side conditions. -/
specification per_block_int8_correctness
    (X ScalePre XInt8 Scale : RegionName)
    (L C BLK scale_stride : Nat) (preScale : ℝ)
    (hB : 0 < BLK * C) (hRegions : XInt8 ≠ Scale) :
    perBlockInt8IO X ScalePre XInt8 Scale L C BLK scale_stride preScale ⊨
      fun _ _ xs ys =>
        (fun j => perBlockInt8ValSpec BLK C preScale xs ys j, fun j => ys j) := by
  refine Masked2DKernelIO₂ₓ₂.Implements.intro _ ?_ ?_ ?_
  · exact per_block_int8_flattenOk X ScalePre XInt8 Scale L C BLK scale_stride
      preScale
  · intro bounds s h1 h2 h3 h4
    exact per_block_int8_traceSafe X ScalePre XInt8 Scale L C BLK scale_stride
      preScale bounds s h1 (h2 ⟨0, hB⟩ trivial) h3 (h4 ⟨0, hB⟩ rfl)
  · intro s₀ xs ys hx hy
    obtain ⟨s1, hexec, hval1, hval2, hframe⟩ :=
      per_block_int8_region_run X ScalePre XInt8 Scale L C BLK scale_stride
        preScale hRegions s₀ xs ys hx (fun j => hy j trivial)
    refine ⟨s1, hexec, hval1, fun j _ => hval2 j, fun ρ o hc1 hc2 => ?_⟩
    refine hframe ρ o hc1 ?_
    rcases hc2 with hne | hno
    · exact Or.inl hne
    · exact Or.inr (hno ⟨0, hB⟩ rfl)

end VeriTile.Bench.TritonBenchG.Int8Quantization
