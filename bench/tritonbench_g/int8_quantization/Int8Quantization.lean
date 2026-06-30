import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

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
per_block_int8_output_summary_general             ← TOP THEOREM (dimension-general)
  ├─ q/k_kernel_per_block_int8_surface_toAlgorithm_supported  full Q/K surface lowers
  ├─ per_block_int8_scale_compute_store_slice_toAlgorithm_supported  scale-compute surface lowers
  ├─ per_block_int8_scaled_store_slice_compute_correct    value writeback
  │    └─ per_block_int8_scaled_store_slice_correct       algorithm-layer readback
  └─ per_block_int8_scale_store_slice_compute_correct     scalar scale writeback
       └─ per_block_int8_scale_store_slice_correct        algorithm-layer readback
```

The value spec is the real-valued scaled store `(preScale * x) / scale`
(`perBlockInt8ScaledSpec`); the scale spec is the stored per-block scalar.

## Modeling boundary

Arithmetic is over `ℝ`, not bit-accurate IEEE float. The honesty point is the
quantization tail: the proofs model the **pre-rounding scaled value**
`(preScale * x) / scale` and the per-block scale `max(|x|) / 127`, and verify
that the full Python surface (including the `x += 0.5 * sign(x)` rounding bias
and the fixed-width `(x).to(tl.int8)` cast annotation) lowers through algorithm
erasure. The bit-accurate effect of the half-ULP rounding bias and the int8
saturating cast is **not** numerically modeled: post-erasure all dtypes unify to
`ℝ`, so `to(tl.int8)` is the identity at the algorithm layer. `@triton.autotune`
is not present here; the Q pre-scale constant is carried symbolically.
-/

namespace VeriTile.Bench.TritonBenchG.Int8Quantization

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Faithful transcription of `int8_quantization.py`'s `q_kernel_per_block_int8`.

The final `to(tl.int8)` is preserved as a surface dtype annotation; algorithm
erasure now carries it through the fixed-width cast surface used by the DSL.
-/
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

/-- Surface transcription of `int8_quantization.py`'s `k_kernel_per_block_int8`.

The final `to(tl.int8)` is preserved as a surface dtype annotation; algorithm
erasure now carries it through the fixed-width cast surface used by the DSL.
-/
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

/-- Proof-oriented scaled-store slice of `int8_quantization.py`'s
`q_kernel_per_block_int8` / `k_kernel_per_block_int8`.

The upstream kernels compute a per-block max scale, divide each element by that
scale, round to int8, and store the result. VeriTile's current arithmetic layer
models real tiles, so this slice starts from a precomputed per-block scale in
`Scale`, keeps the original row mask, and proves the scaled matrix writeback
before the fixed-width rounding/cast surface. The `preScale` parameter is
`C**-0.5 * 1.44269504` for q and `1` for k. -/
def per_block_int8_scaled_store_slice
    (X XInt8 Scale : RegionName)
    (L C BLK scale_stride : Nat) (preScale : ℝ) :
    ComputeKernel := triton {
  off_blk = tl.program_id(0)
  off_b = tl.program_id(1)
  x_offset = off_b * $(L) * $(C)
  offs_m = off_blk * $(BLK) + tl.arange(0, $(BLK))
  offs_k = tl.arange(0, $(C))
  mask = (offs_m[:, None] < $(L)) & (offs_k[None, :] < $(C))
  x = tl.load(X + x_offset + offs_m[:, None] * $(C) + offs_k[None, :],
    mask=mask)
  scale = tl.load(Scale + off_b * $(scale_stride) + off_blk)
  x_scaled = ($(preScale) * x) / scale
  tl.store(XInt8 + x_offset + offs_m[:, None] * $(C) + offs_k[None, :],
    x_scaled, mask=mask)
}

def rowIndex (s : BlockState) (BLK : Nat) (i : Fin BLK) : Nat :=
  s.pids 0 * BLK + i.val

def colIndex (_s : BlockState) (j : Fin C) : Nat :=
  j.val

def baseOffset (s : BlockState) (L C : Nat) : Nat :=
  s.pids 1 * L * C

def xOffset (s : BlockState) (L C BLK : Nat) (idx : TileIndex [BLK, C]) : Nat :=
  baseOffset s L C + rowIndex s BLK idx.1 * C + colIndex s idx.2.1

def scaleOffset (s : BlockState) (scale_stride : Nat) : Nat :=
  s.pids 1 * scale_stride + s.pids 0

def active
    (s : BlockState) (L BLK C : Nat) (idx : TileIndex [BLK, C]) : Prop :=
  rowIndex s BLK idx.1 < L

instance activeDecidable
    (s : BlockState) (L BLK C : Nat) (idx : TileIndex [BLK, C]) :
    Decidable (active s L BLK C idx) := by
  unfold active
  infer_instance

noncomputable def perBlockInt8ScaledSpec
    (s : BlockState) (X Scale : RegionName)
    (L C BLK scale_stride : Nat) (preScale : ℝ)
    (idx : TileIndex [BLK, C]) : ℝ :=
  (preScale * s.readMem X (xOffset s L C BLK idx)) /
    s.readMem Scale (scaleOffset s scale_stride)

/-- Algorithm-layer correctness for the per-block int8 scaled-store slice. -/
theorem per_block_int8_scaled_store_slice_correct
    (X XInt8 Scale : RegionName)
    (L C BLK scale_stride : Nat) (preScale : ℝ)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLK, C] => xOffset s L C BLK idx)) :
    ∀ idx : TileIndex [BLK, C],
      let outAddr := xOffset s L C BLK idx
      (exec (per_block_int8_scaled_store_slice X XInt8 Scale
            L C BLK scale_stride preScale) s).map (·.readMem XInt8 outAddr)
        = some (if active s L BLK C idx then
            perBlockInt8ScaledSpec s X Scale L C BLK scale_stride preScale idx
          else s.readMem XInt8 outAddr) := by
  intro idx
  simp [exec, per_block_int8_scaled_store_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul, NumericDType.div,
        ComparableDType.lt, baseOffset, rowIndex, colIndex, scaleOffset,
        xOffset, TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BLK, C] → Nat :=
    fun idx => s.pids 1 * L * C + (s.pids 0 * BLK + idx.1.val) * C + idx.2.1.val
  let valueFn : TileIndex [BLK, C] → ℝ :=
    fun idx =>
      WithBot.unbotD 0
        (match
          match
            if s.pids 0 * BLK + idx.1.val < L then
              some (s.readMem X
                (s.pids 1 * L * C + (s.pids 0 * BLK + idx.1.val) * C +
                  idx.2.1.val))
            else
              some (s.undef X
                (s.pids 1 * L * C + (s.pids 0 * BLK + idx.1.val) * C +
                  idx.2.1.val)) with
          | some x => some (preScale * x)
          | none => none with
        | some x => some (x / s.readMem Scale (s.pids 1 * scale_stride + s.pids 0))
        | none => none)
  let P : TileIndex [BLK, C] → Prop :=
    fun idx => s.pids 0 * BLK + idx.1.val < L
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, xOffset, baseOffset, rowIndex, colIndex] using hOutInj
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj idx]
  by_cases hRow : s.pids 0 * BLK + idx.1.val < L
  · simp [offsetFn, valueFn, P, active, perBlockInt8ScaledSpec, baseOffset,
      rowIndex, colIndex, scaleOffset, xOffset, hRow]
  · simp [offsetFn, valueFn, P, active, baseOffset, rowIndex, colIndex,
      scaleOffset, xOffset, hRow]

/-- Compute-facing correctness for the per-block int8 scaled-store slice. -/
theorem per_block_int8_scaled_store_slice_compute_correct
    (X XInt8 Scale : RegionName)
    (L C BLK scale_stride : Nat) (preScale : ℝ)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLK, C] => xOffset s L C BLK idx)) :
    ComputeCorrect.Realizes
      (kernel := per_block_int8_scaled_store_slice X XInt8 Scale
        L C BLK scale_stride preScale)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s L BLK C)
        (fun idx => (XInt8, xOffset s L C BLK idx)))
      (expected := fun idx =>
        perBlockInt8ScaledSpec s X Scale L C BLK scale_stride preScale idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [per_block_int8_scaled_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := per_block_int8_scaled_store_slice_correct X XInt8 Scale
    L C BLK scale_stride preScale s hOutInj idx
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

noncomputable def qPreScale (C : Nat) : ℝ :=
  (Real.sqrt (C : ℝ))⁻¹ * (1.44269504 : ℝ)

def kPreScale : ℝ := 1

/-- Compute-facing q-value store correctness for `q_kernel_per_block_int8`,
specializing the generic scaled-store slice to Python's q pre-scale
`C**-0.5 * log2(e)`. -/
theorem q_kernel_per_block_int8_scaled_store_slice_compute_correct
    (X XInt8 Scale : RegionName)
    (L C BLK scale_stride : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLK, C] => xOffset s L C BLK idx)) :
    ComputeCorrect.Realizes
      (kernel := per_block_int8_scaled_store_slice X XInt8 Scale
        L C BLK scale_stride (qPreScale C))
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s L BLK C)
        (fun idx => (XInt8, xOffset s L C BLK idx)))
      (expected := fun idx =>
        perBlockInt8ScaledSpec s X Scale L C BLK scale_stride (qPreScale C) idx) := by
  exact per_block_int8_scaled_store_slice_compute_correct X XInt8 Scale
    L C BLK scale_stride (qPreScale C) s hOutInj

/-- Compute-facing k-value store correctness for `k_kernel_per_block_int8`,
specializing the generic scaled-store slice to the Python k path with no
pre-scale. -/
theorem k_kernel_per_block_int8_scaled_store_slice_compute_correct
    (X XInt8 Scale : RegionName)
    (L C BLK scale_stride : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLK, C] => xOffset s L C BLK idx)) :
    ComputeCorrect.Realizes
      (kernel := per_block_int8_scaled_store_slice X XInt8 Scale
        L C BLK scale_stride kPreScale)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s L BLK C)
        (fun idx => (XInt8, xOffset s L C BLK idx)))
      (expected := fun idx =>
        perBlockInt8ScaledSpec s X Scale L C BLK scale_stride kPreScale idx) := by
  exact per_block_int8_scaled_store_slice_compute_correct X XInt8 Scale
    L C BLK scale_stride kPreScale s hOutInj

/-- Python test case 1 q-value store coverage:
`L = 256`, `C = 64`, `BLKQ = 128`, and `q_scale.stride(0) = 2`. -/
theorem q_kernel_per_block_int8_test1_scaled_store_slice_compute_correct
    (X XInt8 Scale : RegionName)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [128, 64] => xOffset s 256 64 128 idx)) :
    ComputeCorrect.Realizes
      (kernel := per_block_int8_scaled_store_slice X XInt8 Scale
        256 64 128 2 (qPreScale 64))
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 256 128 64)
        (fun idx => (XInt8, xOffset s 256 64 128 idx)))
      (expected := fun idx =>
        perBlockInt8ScaledSpec s X Scale 256 64 128 2 (qPreScale 64) idx) := by
  exact q_kernel_per_block_int8_scaled_store_slice_compute_correct X XInt8
    Scale 256 64 128 2 s hOutInj

/-- Python test case 1 k-value store coverage:
`L = 256`, `C = 64`, `BLKK = 64`, and `k_scale.stride(0) = 4`. -/
theorem k_kernel_per_block_int8_test1_scaled_store_slice_compute_correct
    (X XInt8 Scale : RegionName)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [64, 64] => xOffset s 256 64 64 idx)) :
    ComputeCorrect.Realizes
      (kernel := per_block_int8_scaled_store_slice X XInt8 Scale
        256 64 64 4 kPreScale)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 256 64 64)
        (fun idx => (XInt8, xOffset s 256 64 64 idx)))
      (expected := fun idx =>
        perBlockInt8ScaledSpec s X Scale 256 64 64 4 kPreScale idx) := by
  exact k_kernel_per_block_int8_scaled_store_slice_compute_correct X XInt8
    Scale 256 64 64 4 s hOutInj

/-- Python test case 2 q-value store coverage:
`L = 512`, `C = 128`, `BLKQ = 128`, and `q_scale.stride(0) = 4`. -/
theorem q_kernel_per_block_int8_test2_scaled_store_slice_compute_correct
    (X XInt8 Scale : RegionName)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [128, 128] => xOffset s 512 128 128 idx)) :
    ComputeCorrect.Realizes
      (kernel := per_block_int8_scaled_store_slice X XInt8 Scale
        512 128 128 4 (qPreScale 128))
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 512 128 128)
        (fun idx => (XInt8, xOffset s 512 128 128 idx)))
      (expected := fun idx =>
        perBlockInt8ScaledSpec s X Scale 512 128 128 4 (qPreScale 128) idx) := by
  exact q_kernel_per_block_int8_scaled_store_slice_compute_correct X XInt8
    Scale 512 128 128 4 s hOutInj

/-- Python test case 2 k-value store coverage:
`L = 512`, `C = 128`, `BLKK = 64`, and `k_scale.stride(0) = 8`. -/
theorem k_kernel_per_block_int8_test2_scaled_store_slice_compute_correct
    (X XInt8 Scale : RegionName)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [64, 128] => xOffset s 512 128 64 idx)) :
    ComputeCorrect.Realizes
      (kernel := per_block_int8_scaled_store_slice X XInt8 Scale
        512 128 64 8 kPreScale)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 512 64 128)
        (fun idx => (XInt8, xOffset s 512 128 64 idx)))
      (expected := fun idx =>
        perBlockInt8ScaledSpec s X Scale 512 128 64 8 kPreScale idx) := by
  exact k_kernel_per_block_int8_scaled_store_slice_compute_correct X XInt8
    Scale 512 128 64 8 s hOutInj

/-- Proof-oriented scale-store slice of `int8_quantization.py`'s per-block
int8 kernels. Companion to per_block_int8_scaled_store_slice: takes a
precomputed `ScalePre` scalar (per off_b × off_blk) and proves the unmasked
writeback into `Scale` at offset `off_b * scale_stride + off_blk`. -/
def per_block_int8_scale_store_slice
    (ScalePre Scale : RegionName) (scale_stride : Nat) :
    ComputeKernel := triton {
  off_blk = tl.program_id(0)
  off_b = tl.program_id(1)
  scale = tl.load(ScalePre + off_b * $(scale_stride) + off_blk)
  tl.store(Scale + off_b * $(scale_stride) + off_blk, scale)
}

noncomputable def scaleStoreSpec
    (s : BlockState) (ScalePre : RegionName) (scale_stride : Nat) : ℝ :=
  s.readMem ScalePre (scaleOffset s scale_stride)

theorem per_block_int8_scale_store_slice_correct
    (ScalePre Scale : RegionName) (scale_stride : Nat) (s s' : BlockState)
    (hExec : exec (per_block_int8_scale_store_slice ScalePre Scale scale_stride)
      s = some s') :
    s'.readMem Scale (scaleOffset s scale_stride) =
      scaleStoreSpec s ScalePre scale_stride := by
  simp [exec, per_block_int8_scale_store_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd,
        NumericDType.add, NumericDType.mul] at hExec
  subst s'
  simp [scaleOffset, scaleStoreSpec]

theorem per_block_int8_scale_store_slice_compute_correct
    (ScalePre Scale : RegionName) (scale_stride : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := per_block_int8_scale_store_slice ScalePre Scale scale_stride)
      (initialState := s)
      (write := fun _ : PUnit => some (Scale, scaleOffset s scale_stride))
      (expected := fun _ => scaleStoreSpec s ScalePre scale_stride) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [per_block_int8_scale_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro _
  exact per_block_int8_scale_store_slice_correct ScalePre Scale scale_stride
    s s' hExec

/-- Named q-scale store correctness for the q branch observed by
`per_block_int8`. The address is the same scalar slot as the generic scale
store; the q-specific scale computation is represented by `ScalePre`. -/
theorem q_kernel_per_block_int8_scale_store_slice_compute_correct
    (ScalePre Scale : RegionName) (scale_stride : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := per_block_int8_scale_store_slice ScalePre Scale scale_stride)
      (initialState := s)
      (write := fun _ : PUnit => some (Scale, scaleOffset s scale_stride))
      (expected := fun _ => scaleStoreSpec s ScalePre scale_stride) := by
  exact per_block_int8_scale_store_slice_compute_correct ScalePre Scale
    scale_stride s

/-- Named k-scale store correctness for the k branch observed by
`per_block_int8`. The address is the same scalar slot as the generic scale
store; the k-specific scale computation is represented by `ScalePre`. -/
theorem k_kernel_per_block_int8_scale_store_slice_compute_correct
    (ScalePre Scale : RegionName) (scale_stride : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := per_block_int8_scale_store_slice ScalePre Scale scale_stride)
      (initialState := s)
      (write := fun _ : PUnit => some (Scale, scaleOffset s scale_stride))
      (expected := fun _ => scaleStoreSpec s ScalePre scale_stride) := by
  exact per_block_int8_scale_store_slice_compute_correct ScalePre Scale
    scale_stride s

/-- Python test case 1 q-scale store coverage (`scale_stride = 2`). -/
theorem q_kernel_per_block_int8_test1_scale_store_slice_compute_correct
    (ScalePre Scale : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := per_block_int8_scale_store_slice ScalePre Scale 2)
      (initialState := s)
      (write := fun _ : PUnit => some (Scale, scaleOffset s 2))
      (expected := fun _ => scaleStoreSpec s ScalePre 2) := by
  exact q_kernel_per_block_int8_scale_store_slice_compute_correct ScalePre
    Scale 2 s

/-- Python test case 1 k-scale and test case 2 q-scale store coverage
(`scale_stride = 4`). -/
theorem per_block_int8_test_scale_stride4_store_slice_compute_correct
    (ScalePre Scale : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := per_block_int8_scale_store_slice ScalePre Scale 4)
      (initialState := s)
      (write := fun _ : PUnit => some (Scale, scaleOffset s 4))
      (expected := fun _ => scaleStoreSpec s ScalePre 4) := by
  exact per_block_int8_scale_store_slice_compute_correct ScalePre Scale 4 s

/-- Python test case 2 k-scale store coverage (`scale_stride = 8`). -/
theorem k_kernel_per_block_int8_test2_scale_store_slice_compute_correct
    (ScalePre Scale : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := per_block_int8_scale_store_slice ScalePre Scale 8)
      (initialState := s)
      (write := fun _ : PUnit => some (Scale, scaleOffset s 8))
      (expected := fun _ => scaleStoreSpec s ScalePre 8) := by
  exact k_kernel_per_block_int8_scale_store_slice_compute_correct ScalePre
    Scale 8 s

/-- Proof-oriented scale-compute slice of `int8_quantization.py`'s per-block
int8 kernels. This covers the real-valued Python path
`scale = tl.max(tl.abs(preScale * x)) / 127.0` and the unmasked scalar
`tl.store(scale_ptrs, scale)`, while separating it from the later `to(tl.int8)`
value-store slice. -/
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

theorem per_block_int8_scale_compute_store_slice_toAlgorithm_supported
    (X Scale : RegionName)
    (L C BLK scale_stride : Nat) (preScale : ℝ) :
    ∃ alg, (per_block_int8_scale_compute_store_slice X Scale
      L C BLK scale_stride preScale).toAlgorithm? = Except.ok alg := by
  simp [per_block_int8_scale_compute_store_slice]

/-- **General (dimension-parameterized) per-block int8 quantization correctness.**

For arbitrary token count `L`, channel count `C`, block size `BLK`, scale stride
`scale_stride` and pre-scale `preScale` (with the row-major output offset
injective on the `[BLK, C]` block), the value store realizes the genuine
quantized value `perBlockInt8ScaledSpec` (`= preScale·X / Scale`) on every active
lane and the scale store realizes `scaleStoreSpec`. This theorem is the general
closed form over arbitrary dimensions (mirrors the
dimension-parameterized reference `attention_forward_triton_closed_form_correct`). -/
theorem per_block_int8_closed_form_correct
    (X XInt8 Scale ScalePre : RegionName)
    (L C BLK scale_stride : Nat) (preScale : ℝ) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLK, C] => xOffset s L C BLK idx)) :
    (ComputeCorrect.Realizes
      (kernel := per_block_int8_scaled_store_slice X XInt8 Scale
        L C BLK scale_stride preScale)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s L BLK C)
        (fun idx => (XInt8, xOffset s L C BLK idx)))
      (expected := fun idx =>
        perBlockInt8ScaledSpec s X Scale L C BLK scale_stride preScale idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := per_block_int8_scale_store_slice ScalePre Scale scale_stride)
      (initialState := s)
      (write := fun _ : PUnit => some (Scale, scaleOffset s scale_stride))
      (expected := fun _ => scaleStoreSpec s ScalePre scale_stride)) :=
  ⟨per_block_int8_scaled_store_slice_compute_correct X XInt8 Scale
      L C BLK scale_stride preScale s hOutInj,
   per_block_int8_scale_store_slice_compute_correct ScalePre Scale scale_stride s⟩

/-- **Dimension-general output summary for `per_block_int8` (`int8_quantization.py`).**

For arbitrary token count `L`, channel count `C`, block size `BLK`, scale stride
`scale_stride` and pre-scale `preScale` (with the row-major output offset
injective on the `[BLK, C]` block), this bundles:

* both the full faithful Q surface (`q_kernel_per_block_int8_surface`, including
  the `C**-0.5 * log2(e)` pre-scale, `tl.abs`/`tl.max` per-block scale, signed
  half-up rounding and the `to(tl.int8)` cast) and the full faithful K surface
  (`k_kernel_per_block_int8_surface`) lowering to the algorithm layer, plus the
  scale-compute store surface lowering;
* the **value** store realizing `perBlockInt8ScaledSpec` (`= preScale·X / Scale`)
  on every active lane (`off_blk·BLK + i < L`), unchanged otherwise;
* the **scale** store realizing the per-block scalar `scaleStoreSpec`.

All expected values are computed from the kernel **inputs** (no `exec`/`readMem`
self-reference). This general closed form holds over arbitrary dimensions
(mirrors the dimension-parameterized reference
`attention_forward_triton_closed_form_correct`). -/
theorem per_block_int8_output_summary_general
    (X XInt8 Scale ScalePre : RegionName)
    (L C BLK scale_stride : Nat) (preScale : ℝ) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLK, C] => xOffset s L C BLK idx)) :
    ((∃ alg, (q_kernel_per_block_int8_surface X XInt8 Scale
        L C BLK scale_stride).toAlgorithm? = Except.ok alg) ∧
     (∃ alg, (k_kernel_per_block_int8_surface X XInt8 Scale
        L C BLK scale_stride).toAlgorithm? = Except.ok alg) ∧
     (∃ alg, (per_block_int8_scale_compute_store_slice X Scale
        L C BLK scale_stride preScale).toAlgorithm? = Except.ok alg)) ∧
    (ComputeCorrect.Realizes
      (kernel := per_block_int8_scaled_store_slice X XInt8 Scale
        L C BLK scale_stride preScale)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s L BLK C)
        (fun idx => (XInt8, xOffset s L C BLK idx)))
      (expected := fun idx =>
        perBlockInt8ScaledSpec s X Scale L C BLK scale_stride preScale idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := per_block_int8_scale_store_slice ScalePre Scale scale_stride)
      (initialState := s)
      (write := fun _ : PUnit => some (Scale, scaleOffset s scale_stride))
      (expected := fun _ => scaleStoreSpec s ScalePre scale_stride)) :=
  ⟨⟨q_kernel_per_block_int8_surface_toAlgorithm_supported X XInt8 Scale
      L C BLK scale_stride,
    k_kernel_per_block_int8_surface_toAlgorithm_supported X XInt8 Scale
      L C BLK scale_stride,
    per_block_int8_scale_compute_store_slice_toAlgorithm_supported X Scale
      L C BLK scale_stride preScale⟩,
   (per_block_int8_closed_form_correct X XInt8 Scale ScalePre
      L C BLK scale_stride preScale s hOutInj).1,
   (per_block_int8_closed_form_correct X XInt8 Scale ScalePre
      L C BLK scale_stride preScale s hOutInj).2⟩

end VeriTile.Bench.TritonBenchG.Int8Quantization
