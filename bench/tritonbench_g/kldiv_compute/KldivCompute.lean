import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.Math.Loss
import VeriTile.Triton.DSL
import VeriTile.Examples.Common

namespace VeriTile.Bench.TritonBenchG.KldivCompute

open VeriTile.Triton VeriTile.Examples
open VeriTile.Triton.TiledLoss

/-- Faithful 1:1 transcription of `kldiv_compute.py`'s
`kldivergence_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` → Lean `Nat` parameter. -/
def kldivergence_kernel
    (x_ptr y_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  x = tl.load(x_ptr + offsets, mask=mask)
  y = tl.load(y_ptr + offsets, mask=mask)
  output = x * tl.log(x / y)
  tl.store(output_ptr + offsets, output, mask=mask)
}

/-- Algorithm-layer correctness for `kldivergence_kernel`.

For inactive lanes the masked store does not write, so the output cell is
preserved from the initial state. -/
theorem kldivergence_kernel_correct
    (x_ptr y_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) (_hBlockSize : 0 < BLOCK_SIZE)
    (s : BlockState) (xs ys : Fin BLOCK_SIZE → ℝ)
    (h_x : InputLoadedAt s x_ptr BLOCK_SIZE xs)
    (h_y : InputLoadedAt s y_ptr BLOCK_SIZE ys) :
    ∀ i : Fin BLOCK_SIZE,
      let addr := s.pid * BLOCK_SIZE + i.val
      observeAt (exec (kldivergence_kernel x_ptr y_ptr output_ptr n_elements BLOCK_SIZE) s)
          output_ptr BLOCK_SIZE s.pid i
        = some (if addr < n_elements then klDivSpec (xs i) (ys i)
                else s.readMem output_ptr addr) := by
  intro i
  have h_inj := injective_offset_singleton (n := BLOCK_SIZE) (s.pid * BLOCK_SIZE)
  simp [observeAt, exec, kldivergence_kernel, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Tile.bop, Tile.uop, Tile.cop,
        NumericDType.add, NumericDType.mul, NumericDType.div,
        ComparableDType.lt, klDivSpec]
  unfold InputLoadedAt at h_x h_y
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj (i, PUnit.unit)]
  by_cases hi : s.pid * BLOCK_SIZE + i.val < n_elements
  · simp [hi, h_x, h_y]
  · simp [hi]

/-- Compute-facing correctness for `kldivergence_kernel`. -/
theorem kldivergence_kernel_compute_correct
    (x_ptr y_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) (hBlockSize : 0 < BLOCK_SIZE)
    (s : BlockState) (xs ys : Fin BLOCK_SIZE → ℝ)
    (h_x : InputLoadedAt s x_ptr BLOCK_SIZE xs)
    (h_y : InputLoadedAt s y_ptr BLOCK_SIZE ys) :
    ComputeCorrect.Realizes
      (kernel := kldivergence_kernel x_ptr y_ptr output_ptr n_elements BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin BLOCK_SIZE => s.pid * BLOCK_SIZE + i.val < n_elements)
          (fun i => (output_ptr, s.pid * BLOCK_SIZE + i.val)))
      (expected := fun i => klDivSpec (xs i) (ys i)) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel rfl
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have hi := kldivergence_kernel_correct x_ptr y_ptr output_ptr n_elements
    BLOCK_SIZE hBlockSize s xs ys h_x h_y i
  rw [hExec] at hi
  simp [observeAt, hActive] at hi
  exact hi

end VeriTile.Bench.TritonBenchG.KldivCompute
