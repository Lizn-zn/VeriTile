import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.TritonArgmax

open VeriTile.Triton

set_option maxHeartbeats 5000000

/-- Faithful transcription of `triton_argmax.py`'s `argmax_kernel_2`.

DSL-gap interim: `tl.load(mid_index_ptrs)` in `.py` returns int64 (the
launch site supplies a `torch.int64` buffer for `mid_index`); VeriTile's
region model does not carry element dtypes, and the macro's `.nat`
inference only fires for offset-position loads. `out_val` here flows
straight into `tl.store(out, out_val)` (store-value position), so the
DSL has no signal to default the load to `.nat`. Until typed regions
land (issue #115), the explicit `dtype=tl.uint64` is required to keep
the algorithm-layer spec on the `.nat` memory channel. -/
def argmax_kernel_2
    (mid_value mid_index out : RegionName)
    (mid_size BLOCK_MID : Nat) :
    ComputeKernel := triton {
  offset = tl.arange(0, $(BLOCK_MID))
  mid_ptrs = mid_value + offset
  mask = offset < $(mid_size)
  mid_val = tl.load(mid_ptrs, mask=mask, other=-float("inf"))
  index_val = tl.argmax(mid_val, axis=0)
  mid_index_ptrs = mid_index + index_val
  out_val = tl.load(mid_index_ptrs, dtype=tl.uint64)
  tl.store(out, out_val)
}

noncomputable def argmaxKernel2InputTile
    (s : BlockState) (mid_value : RegionName) (mid_size BLOCK_MID : Nat) :
    Tile .real [BLOCK_MID] :=
  { data := fun idx =>
      if idx.1.val < mid_size then some (s.readMem mid_value idx.1.val) else none }

noncomputable def argmaxKernel2IndexOffset
    (s : BlockState) (mid_value : RegionName) (mid_size BLOCK_MID : Nat) : Nat :=
  (Tile.argMaxDrop (shape := [BLOCK_MID]) ⟨0, by simp⟩
    (argmaxKernel2InputTile s mid_value mid_size BLOCK_MID)).data PUnit.unit

noncomputable def argmaxKernel2Spec
    (s : BlockState) (mid_value mid_index : RegionName) (mid_size BLOCK_MID : Nat) : Nat :=
  s.readMemValue .nat mid_index
    (argmaxKernel2IndexOffset s mid_value mid_size BLOCK_MID)

/-- Compute-facing correctness for `argmax_kernel_2`. -/
theorem argmax_kernel_2_compute_correct
    (mid_value mid_index out : RegionName)
    (mid_size BLOCK_MID : Nat)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := argmax_kernel_2 mid_value mid_index out mid_size BLOCK_MID)
      (initialState := s)
      (write := fun _ : PUnit => some (out, 0))
      (expected := fun _ => argmaxKernel2Spec s mid_value mid_index mid_size BLOCK_MID) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [argmax_kernel_2]
  intro s0 s' hExec hs0
  subst s0
  intro _
  by_cases hB : 0 < BLOCK_MID
  · simp [exec, argmax_kernel_2, stepStmts, stepStmt, evalOp, Option.bind,
          Tile.cop, Tile.ptrAdd, Tile.argMaxDrop, Tile.argBestDrop,
          TileShape.axisDim, TileShape.eraseAxis, ComparableDType.lt,
          argmaxKernel2Spec, argmaxKernel2IndexOffset,
          argmaxKernel2InputTile, hB] at hExec ⊢
    cases hExec
    simp [BlockState.writeMemTyped_nat_readMemValue_nat]
  · simp [exec, argmax_kernel_2, stepStmts, stepStmt, evalOp, Option.bind,
          Tile.cop, Tile.ptrAdd, Tile.argMaxDrop, Tile.argBestDrop,
          TileShape.axisDim, TileShape.eraseAxis, ComparableDType.lt,
          argmaxKernel2Spec, argmaxKernel2IndexOffset,
          hB] at hExec ⊢
    cases hExec
    simp [BlockState.writeMemTyped_nat_readMemValue_nat]

end VeriTile.Bench.TritonBenchG.TritonArgmax
