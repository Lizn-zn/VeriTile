import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Examples.Common

namespace VeriTile.Bench.TritonBenchG.SinKernel

open VeriTile.Triton VeriTile.Examples

/-- Faithful 1:1 transcription of `sin_kernel.py`'s `kernel_function`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` → Lean `Nat` parameter. -/
def kernel_function
    (x_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  x = tl.load(x_ptr + offsets, mask=mask)
  output = tl.math.sin(x)
  tl.store(output_ptr + offsets, output, mask=mask)
}

/-- Algorithm-layer correctness for `kernel_function`. -/
theorem kernel_function_correct
    (x_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) (_hBlockSize : 0 < BLOCK_SIZE)
    (s : BlockState) (xs : Fin BLOCK_SIZE → ℝ)
    (h_x : InputLoadedAt s x_ptr BLOCK_SIZE xs) :
    ∀ i : Fin BLOCK_SIZE,
      let addr := s.pid * BLOCK_SIZE + i.val
      observeAt (exec (kernel_function x_ptr output_ptr n_elements BLOCK_SIZE) s)
          output_ptr BLOCK_SIZE s.pid i
        = some (if addr < n_elements then Real.sin (xs i)
                else s.readMem output_ptr addr) := by
  intro i
  have h_inj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE] => s.pid * BLOCK_SIZE + idx.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp [observeAt, exec, kernel_function, stepStmts, stepStmt, evalOp,
        Tile.bop, Tile.uop, Tile.cop, NumericDType.add, NumericDType.mul,
        ComparableDType.lt, WithBot.realSin]
  unfold InputLoadedAt at h_x
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj (i, PUnit.unit)]
  by_cases hi : s.pid * BLOCK_SIZE + i.val < n_elements
  · simp [hi, h_x]
  · simp [hi]

/-- Compute-facing correctness for `kernel_function`. -/
theorem kernel_function_compute_correct
    (x_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) (hBlockSize : 0 < BLOCK_SIZE)
    (s : BlockState) (xs : Fin BLOCK_SIZE → ℝ)
    (h_x : InputLoadedAt s x_ptr BLOCK_SIZE xs) :
    ComputeKernel.ComputeCorrect
      (kernel_function x_ptr output_ptr n_elements BLOCK_SIZE)
      (fun s0 s' =>
        s0 = s →
        ∀ i : Fin BLOCK_SIZE,
          let addr := s.pid * BLOCK_SIZE + i.val
          observeAt (some s') output_ptr BLOCK_SIZE s.pid i
            = some (if addr < n_elements then Real.sin (xs i)
                    else s.readMem output_ptr addr)) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel rfl
  intro s0 s' hExec hs0
  subst s0
  intro i
  have hi := kernel_function_correct x_ptr output_ptr n_elements BLOCK_SIZE
    hBlockSize s xs h_x i
  rw [hExec] at hi
  simpa using hi

end VeriTile.Bench.TritonBenchG.SinKernel
