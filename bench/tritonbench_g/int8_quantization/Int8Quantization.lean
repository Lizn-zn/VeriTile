import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.Int8Quantization

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Surface transcription of `int8_quantization.py`'s `q_kernel_per_block_int8`.

The `preScale` parameter represents Python's `C**-0.5 * 1.44269504` factor.
The final `to(tl.int8)` is preserved as a surface dtype annotation while the
algorithm carrier records the rounded real-valued expression. -/
def q_kernel_per_block_int8_surface
    (X XInt8 Scale : RegionName)
    (L C BLK scale_stride : Nat) (preScale : ℝ) :
    ComputeKernel := triton {
  off_b = tl.program_id(axis=1)
  off_blk = tl.program_id(axis=0)
  x_offset = off_b * $(L) * $(C)
  offs_m = off_blk * $(BLK) + tl.arange(0, $(BLK))
  offs_k = tl.arange(0, $(C))
  x_ptrs = X + x_offset + offs_m[:, None] * $(C) + offs_k[None, :]
  x_int8_ptrs = XInt8 + x_offset + offs_m[:, None] * $(C) + offs_k[None, :]
  scale_ptrs = Scale + off_b * $(scale_stride) + off_blk
  x = tl.load(x_ptrs, mask=offs_m[:, None] < $(L))
  x *= $(preScale)
  scale = tl.max(tl.abs(x)) / 127.0
  x_int8 = x / scale
  x_int8 += 0.5 * tl.where(x_int8 >= 0.0, 1.0, -1.0)
  x_int8 = (x_int8).to(tl.int8)
  tl.store(x_int8_ptrs, x_int8, mask=offs_m[:, None] < $(L))
  tl.store(scale_ptrs, scale)
}

/-- Surface transcription of `int8_quantization.py`'s `k_kernel_per_block_int8`.

The final `to(tl.int8)` is preserved as a surface dtype annotation while the
algorithm carrier records the rounded real-valued expression. -/
def k_kernel_per_block_int8_surface
    (X XInt8 Scale : RegionName)
    (L C BLK scale_stride : Nat) :
    ComputeKernel := triton {
  off_b = tl.program_id(axis=1)
  off_blk = tl.program_id(axis=0)
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

/-- Proof-oriented scaled-store slice of `int8_quantization.py`'s
`q_kernel_per_block_int8` / `k_kernel_per_block_int8`.

The upstream kernels compute a per-block max scale, divide each element by that
scale, round to int8, and store the result. VeriTile's current arithmetic layer
models real tiles, so this slice starts from a precomputed per-block scale in
`Scale`, keeps the original row mask, and proves the scaled matrix writeback
before the backend-specific rounding/cast step. The `preScale` parameter is
`C**-0.5 * 1.44269504` for q and `1` for k. -/
def per_block_int8_scaled_store_slice
    (X XInt8 Scale : RegionName)
    (L C BLK scale_stride : Nat) (preScale : ℝ) :
    ComputeKernel := triton {
  off_blk = tl.program_id(axis=0)
  off_b = tl.program_id(axis=1)
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
  simp [exec, per_block_int8_scaled_store_slice, stepStmts, stepStmt, evalOp,
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

end VeriTile.Bench.TritonBenchG.Int8Quantization
