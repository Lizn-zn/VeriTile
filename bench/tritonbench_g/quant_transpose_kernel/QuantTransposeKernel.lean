import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.QuantTransposeKernel

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Proof-oriented scaled-store tile slice of `quant_transpose_kernel.py`'s
`_quantize_global_transpose`.

The full Triton kernel uses a one-dimensional grouped program-id schedule to
derive `pid_m` and `pid_n`. This slice starts after that scheduling choice, uses
program axes 0/1 for the tile coordinates, loads the `BLOCK_M × BLOCK_N` tile
from `A`, applies the global `absmax_inv` scale, and proves the masked writeback
into `B`. CUDA `llrint` and int8 casting are outside VeriTile's current real-tile
arithmetic layer, matching the other quantization ports. -/
def quantize_global_transpose_scaled_store_slice
    (A AbsmaxInv B : RegionName)
    (stride_am stride_an stride_bn stride_bm M N BLOCK_M BLOCK_N : Nat)
    (scale127 : ℝ) :
    ComputeKernel := triton {
  pid_m = tl.program_id(axis=0)
  pid_n = tl.program_id(axis=1)
  rm = pid_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  rn = pid_n * $(BLOCK_N) + tl.arange(0, $(BLOCK_N))
  mask = (rm[:, None] < $(M)) & (rn[None, :] < $(N))
  a = tl.load(A + rm[:, None] * $(stride_am) + rn[None, :] * $(stride_an),
    mask=mask)
  absmax_inv = tl.load(AbsmaxInv)
  output = $(scale127) * (a * absmax_inv)
  tl.store(B + rm[:, None] * $(stride_bm) + rn[None, :] * $(stride_bn),
    output, mask=mask)
}

def rowIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val

def colIndex (s : BlockState) (BLOCK_N : Nat) (j : Fin BLOCK_N) : Nat :=
  s.pids 1 * BLOCK_N + j.val

def active
    (s : BlockState) (M N BLOCK_M BLOCK_N : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_N]) : Prop :=
  rowIndex s BLOCK_M idx.1 < M ∧ colIndex s BLOCK_N idx.2.1 < N

instance activeDecidable
    (s : BlockState) (M N BLOCK_M BLOCK_N : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_N]) :
    Decidable (active s M N BLOCK_M BLOCK_N idx) := by
  unfold active
  infer_instance

def aOffset
    (s : BlockState) (stride_am stride_an BLOCK_M BLOCK_N : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_N]) : Nat :=
  rowIndex s BLOCK_M idx.1 * stride_am + colIndex s BLOCK_N idx.2.1 * stride_an

def bOffset
    (s : BlockState) (stride_bm stride_bn BLOCK_M BLOCK_N : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_N]) : Nat :=
  rowIndex s BLOCK_M idx.1 * stride_bm + colIndex s BLOCK_N idx.2.1 * stride_bn

noncomputable def quantTransposeScaledSpec
    (s : BlockState) (A AbsmaxInv : RegionName)
    (stride_am stride_an BLOCK_M BLOCK_N : Nat) (scale127 : ℝ)
    (idx : TileIndex [BLOCK_M, BLOCK_N]) : ℝ :=
  scale127 *
    (s.readMem A (aOffset s stride_am stride_an BLOCK_M BLOCK_N idx) *
      s.readMem AbsmaxInv 0)

/-- Algorithm-layer correctness for the quantize-global-transpose store slice. -/
theorem quantize_global_transpose_scaled_store_slice_correct
    (A AbsmaxInv B : RegionName)
    (stride_am stride_an stride_bn stride_bm M N BLOCK_M BLOCK_N : Nat)
    (scale127 : ℝ)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
        bOffset s stride_bm stride_bn BLOCK_M BLOCK_N idx)) :
    ∀ idx : TileIndex [BLOCK_M, BLOCK_N],
      let outAddr := bOffset s stride_bm stride_bn BLOCK_M BLOCK_N idx
      (exec (quantize_global_transpose_scaled_store_slice A AbsmaxInv B
            stride_am stride_an stride_bn stride_bm M N BLOCK_M BLOCK_N scale127)
          s).map (·.readMem B outAddr)
        = some (if active s M N BLOCK_M BLOCK_N idx then
            quantTransposeScaledSpec s A AbsmaxInv stride_am stride_an
              BLOCK_M BLOCK_N scale127 idx
          else s.readMem B outAddr) := by
  intro idx
  simp [exec, quantize_global_transpose_scaled_store_slice, stepStmts, stepStmt,
        evalOp, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul, ComparableDType.lt,
        rowIndex, colIndex, aOffset, bOffset, TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BLOCK_M, BLOCK_N] → Nat :=
    fun idx =>
      (s.pids 0 * BLOCK_M + idx.1.val) * stride_bm +
        (s.pids 1 * BLOCK_N + idx.2.1.val) * stride_bn
  let valueFn : TileIndex [BLOCK_M, BLOCK_N] → ℝ :=
    fun idx =>
      WithBot.unbotD 0
        (match
          match
            if s.pids 0 * BLOCK_M + idx.1.val < M ∧
                s.pids 1 * BLOCK_N + idx.2.1.val < N then
              some (s.readMem A
                ((s.pids 0 * BLOCK_M + idx.1.val) * stride_am +
                  (s.pids 1 * BLOCK_N + idx.2.1.val) * stride_an))
            else
              some (s.undef A
                ((s.pids 0 * BLOCK_M + idx.1.val) * stride_am +
                  (s.pids 1 * BLOCK_N + idx.2.1.val) * stride_an)) with
          | some x => some (x * s.readMem AbsmaxInv 0)
          | none => none with
        | some x => some (scale127 * x)
        | none => none)
  let P : TileIndex [BLOCK_M, BLOCK_N] → Prop :=
    fun idx =>
      s.pids 0 * BLOCK_M + idx.1.val < M ∧
        s.pids 1 * BLOCK_N + idx.2.1.val < N
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, bOffset, rowIndex, colIndex] using hOutInj
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj idx]
  by_cases hActive :
      s.pids 0 * BLOCK_M + idx.1.val < M ∧
        s.pids 1 * BLOCK_N + idx.2.1.val < N
  · simp [offsetFn, valueFn, P, active, quantTransposeScaledSpec, rowIndex,
      colIndex, aOffset, bOffset, hActive]
  · simp [offsetFn, valueFn, P, active, rowIndex, colIndex, aOffset, bOffset,
      hActive]

/-- Compute-facing correctness for the quantize-global-transpose store slice. -/
theorem quantize_global_transpose_scaled_store_slice_compute_correct
    (A AbsmaxInv B : RegionName)
    (stride_am stride_an stride_bn stride_bm M N BLOCK_M BLOCK_N : Nat)
    (scale127 : ℝ)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
        bOffset s stride_bm stride_bn BLOCK_M BLOCK_N idx)) :
    ComputeCorrect.Realizes
      (kernel := quantize_global_transpose_scaled_store_slice A AbsmaxInv B
        stride_am stride_an stride_bn stride_bm M N BLOCK_M BLOCK_N scale127)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s M N BLOCK_M BLOCK_N)
        (fun idx => (B, bOffset s stride_bm stride_bn BLOCK_M BLOCK_N idx)))
      (expected := fun idx =>
        quantTransposeScaledSpec s A AbsmaxInv stride_am stride_an
          BLOCK_M BLOCK_N scale127 idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [quantize_global_transpose_scaled_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := quantize_global_transpose_scaled_store_slice_correct A AbsmaxInv B
    stride_am stride_an stride_bn stride_bm M N BLOCK_M BLOCK_N scale127
    s hOutInj idx
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

end VeriTile.Bench.TritonBenchG.QuantTransposeKernel
