import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.Int8Quantization

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Faithful transcription of `int8_quantization.py`'s `q_kernel_per_block_int8`.

The final `to(tl.int8)` is preserved as a surface dtype annotation while the
algorithm carrier records the rounded real-valued expression. -/
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

/-- Surface transcription of `int8_quantization.py`'s `k_kernel_per_block_int8`.

The final `to(tl.int8)` is preserved as a surface dtype annotation while the
algorithm carrier records the rounded real-valued expression. -/
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
  simp [exec, per_block_int8_scale_store_slice, stepStmts, stepStmt, evalOp,
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

/-! ## Full-kernel scale-store correctness

Although the int8 value store in `q_kernel_per_block_int8_surface` /
`k_kernel_per_block_int8_surface` is blocked by the `to(tl.int8)` cast (the
real-channel algorithm carrier records the pre-cast real value), the scale
store is written from `tl.max(tl.abs(x_scaled)) / 127.0` -- a purely
real-valued computation that does NOT depend on the int8 cast. The scale store
is unmasked and lands at a different region from the value store, so the
value-store foldl can be stripped via region-disjointness. -/

/-- The pre-`abs` tile fed into `tl.max` in the surface kernel: a `[BLK, C]`
real tile with active lanes holding `preScale * s.readMem X (xOffset ...)` and
masked-out lanes holding `preScale * s.undef X (...)`. -/
noncomputable def perBlockInt8ScaleInputTile
    (s : BlockState) (X : RegionName) (L C BLK : Nat) (preScale : ℝ) :
    Tile .real [BLK, C] :=
  { data := fun idx =>
      if s.pids 0 * BLK + idx.1.val < L then
        some (preScale * s.readMem X (xOffset s L C BLK idx))
      else
        some (preScale * s.undef X (xOffset s L C BLK idx)) }

/-- Absolute-value tile produced by the surface kernel's `tl.abs(x)`. Triton
lowers `tl.abs` to `select(x < 0, -x, x)` over the masked input tile. -/
noncomputable def perBlockInt8ScaleAbsTile
    (s : BlockState) (X : RegionName) (L C BLK : Nat) (preScale : ℝ) :
    Tile .real [BLK, C] :=
  let x := perBlockInt8ScaleInputTile s X L C BLK preScale
  Tile.select
    { data := fun i =>
        ComparableDType.real.lt (x.data i) (some 0) }
    { data := fun i => NumericDType.real.sub (some 0) (x.data i) }
    x

/-- Scale spec for the per-block int8 surface kernel:
`tl.max(tl.abs(preScale * x)) / 127.0`, reduced over both axes of the loaded
`[BLK, C]` tile. -/
noncomputable def perBlockInt8ScaleSpec
    (s : BlockState) (X : RegionName) (L C BLK : Nat) (preScale : ℝ) : ℝ :=
  match Tile.reduceMaxDrop (shape := [BLK, C]) ⟨1, by simp⟩
      (perBlockInt8ScaleAbsTile s X L C BLK preScale) with
  | some t1 =>
      match Tile.reduceMaxDrop (shape := [BLK]) ⟨0, by simp⟩ t1 with
      | some t0 =>
          WithBot.unbotD 0
            (match t0.data PUnit.unit with
             | some x => some (x / 127.0)
             | none => none)
      | none => 0
  | none => 0

/-- Algorithm-layer correctness for the scale store of
`q_kernel_per_block_int8_surface`. The `preScale` factor is
`(Real.sqrt C)⁻¹ * 1.44269504`. The cast-blocked int8 value store is
stripped via region-disjointness (`Scale ≠ XInt8`).

The `0 < BLK` and `0 < C` hypotheses are required for `tl.max` over the
`[BLK, C]` tile to be defined; the upstream Python launches with positive
`BLK` and `C`. -/
theorem q_kernel_per_block_int8_surface_scale_correct
    (X XInt8 Scale : RegionName)
    (L C BLK scale_stride : Nat)
    (s s' : BlockState)
    (hRegions : Scale ≠ XInt8)
    (hBLK : 0 < BLK) (hC : 0 < C)
    (hExec : exec (q_kernel_per_block_int8_surface X XInt8 Scale
        L C BLK scale_stride) s = some s') :
    s'.readMem Scale (scaleOffset s scale_stride) =
      perBlockInt8ScaleSpec s X L C BLK
        ((Real.sqrt (C : ℝ))⁻¹ * (1.44269504 : ℝ)) := by
  simp [exec, q_kernel_per_block_int8_surface, stepStmts, stepStmt, evalOp,
        Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul, NumericDType.div,
        ComparableDType.lt, ComparableDType.ge, scaleOffset, xOffset,
        baseOffset, rowIndex, colIndex,
        Tile.reduceMaxDrop, TileShape.axisDim, TileShape.eraseAxis,
        TileShape.insertAxisIndex, hBLK, hC] at hExec
  subst hExec
  simp [BlockState.writeMem_readMem, perBlockInt8ScaleSpec,
        perBlockInt8ScaleAbsTile, perBlockInt8ScaleInputTile,
        xOffset, baseOffset, rowIndex, colIndex, scaleOffset,
        Tile.reduceMaxDrop, TileShape.axisDim, TileShape.eraseAxis,
        TileShape.insertAxisIndex, hBLK, hC]

/-- Compute-facing scale-store correctness for `q_kernel_per_block_int8_surface`. -/
theorem q_kernel_per_block_int8_surface_scale_compute_correct
    (X XInt8 Scale : RegionName)
    (L C BLK scale_stride : Nat)
    (s : BlockState)
    (hRegions : Scale ≠ XInt8)
    (hBLK : 0 < BLK) (hC : 0 < C) :
    ComputeCorrect.Realizes
      (kernel := q_kernel_per_block_int8_surface X XInt8 Scale
        L C BLK scale_stride)
      (initialState := s)
      (write := fun _ : PUnit => some (Scale, scaleOffset s scale_stride))
      (expected := fun _ =>
        perBlockInt8ScaleSpec s X L C BLK
          ((Real.sqrt (C : ℝ))⁻¹ * (1.44269504 : ℝ))) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [q_kernel_per_block_int8_surface]
  intro s0 s' hExec hs0
  subst s0
  intro _
  exact q_kernel_per_block_int8_surface_scale_correct X XInt8 Scale
    L C BLK scale_stride s s' hRegions hBLK hC hExec

/-- Algorithm-layer correctness for the scale store of
`k_kernel_per_block_int8_surface`. The k variant has `preScale = 1`. -/
theorem k_kernel_per_block_int8_surface_scale_correct
    (X XInt8 Scale : RegionName)
    (L C BLK scale_stride : Nat)
    (s s' : BlockState)
    (hRegions : Scale ≠ XInt8)
    (hExec : exec (k_kernel_per_block_int8_surface X XInt8 Scale
        L C BLK scale_stride) s = some s') :
    s'.readMem Scale (scaleOffset s scale_stride) =
      perBlockInt8ScaleSpec s X L C BLK 1 := by
  simp [exec, k_kernel_per_block_int8_surface, stepStmts, stepStmt, evalOp,
        Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul, NumericDType.div,
        ComparableDType.lt, ComparableDType.ge, scaleOffset, xOffset,
        baseOffset, rowIndex, colIndex] at hExec
  subst hExec
  simp [BlockState.writeMem_readMem, perBlockInt8ScaleSpec,
        perBlockInt8ScaleAbsTile, perBlockInt8ScaleInputTile,
        xOffset, baseOffset, rowIndex, colIndex, scaleOffset,
        Tile.reduceMaxDrop, TileShape.axisDim, TileShape.eraseAxis,
        TileShape.insertAxisIndex]

/-- Compute-facing scale-store correctness for `k_kernel_per_block_int8_surface`. -/
theorem k_kernel_per_block_int8_surface_scale_compute_correct
    (X XInt8 Scale : RegionName)
    (L C BLK scale_stride : Nat)
    (s : BlockState)
    (hRegions : Scale ≠ XInt8) :
    ComputeCorrect.Realizes
      (kernel := k_kernel_per_block_int8_surface X XInt8 Scale
        L C BLK scale_stride)
      (initialState := s)
      (write := fun _ : PUnit => some (Scale, scaleOffset s scale_stride))
      (expected := fun _ => perBlockInt8ScaleSpec s X L C BLK 1) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [k_kernel_per_block_int8_surface]
  intro s0 s' hExec hs0
  subst s0
  intro _
  exact k_kernel_per_block_int8_surface_scale_correct X XInt8 Scale
    L C BLK scale_stride s s' hRegions hExec

>end VeriTile.Bench.TritonBenchG.Int8Quantization
