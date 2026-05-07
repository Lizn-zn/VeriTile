import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Examples.Common

namespace VeriTile.Bench.TritonBenchG.TritonMul2

open VeriTile.Triton VeriTile.Examples

/-- Faithful 1:1 transcription of `triton_mul2.py`'s `mul2_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` → Lean `Nat` parameter. -/
def mul2_kernel
    (in_ptr0 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  x = tl.load(in_ptr0 + offsets, mask=mask)
  output = 2 * x
  tl.store(out_ptr + offsets, output, mask=mask)
}

/-- Faithful 1:1 transcription of `triton_mul2.py`'s `mul2_inplace_kernel`.

Same allowed mechanical Lean-syntax-only changes as `mul2_kernel`. -/
def mul2_inplace_kernel
    (ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  x = tl.load(ptr + offsets, mask=mask)
  output = 2 * x
  tl.store(ptr + offsets, output, mask=mask)
}

/-- Algorithm-layer correctness for `mul2_kernel`. -/
theorem mul2_kernel_correct
    (in_ptr0 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) (_hBlockSize : 0 < BLOCK_SIZE)
    (s : BlockState) (xs : Fin BLOCK_SIZE → ℝ)
    (h_x : InputLoadedAt s in_ptr0 BLOCK_SIZE xs) :
    ∀ i : Fin BLOCK_SIZE,
      let addr := s.pid * BLOCK_SIZE + i.val
      observeAt (exec (mul2_kernel in_ptr0 out_ptr n_elements BLOCK_SIZE) s)
          out_ptr BLOCK_SIZE s.pid i
        = some (if addr < n_elements then 2 * xs i else s.readMem out_ptr addr) := by
  intro i
  have h_inj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE] => s.pid * BLOCK_SIZE + idx.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp [observeAt, exec, mul2_kernel, stepStmts, stepStmt, evalOp,
        Tile.bop, Tile.cop, NumericDType.add, NumericDType.mul,
        ComparableDType.lt]
  unfold InputLoadedAt at h_x
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj (i, PUnit.unit)]
  by_cases hi : s.pid * BLOCK_SIZE + i.val < n_elements
  · simp [hi, h_x]
  · simp [hi]

/-- Compute-facing correctness for `mul2_kernel`. -/
theorem mul2_kernel_compute_correct
    (in_ptr0 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) (hBlockSize : 0 < BLOCK_SIZE)
    (s : BlockState) (xs : Fin BLOCK_SIZE → ℝ)
    (h_x : InputLoadedAt s in_ptr0 BLOCK_SIZE xs) :
    ComputeKernel.ComputeCorrect
      (mul2_kernel in_ptr0 out_ptr n_elements BLOCK_SIZE)
      (fun s0 s' =>
        s0 = s →
        ∀ i : Fin BLOCK_SIZE,
          let addr := s.pid * BLOCK_SIZE + i.val
          observeAt (some s') out_ptr BLOCK_SIZE s.pid i
            = some (if addr < n_elements then 2 * xs i else s.readMem out_ptr addr)) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel rfl
  intro s0 s' hExec hs0
  subst s0
  intro i
  have hi := mul2_kernel_correct in_ptr0 out_ptr n_elements BLOCK_SIZE
    hBlockSize s xs h_x i
  rw [hExec] at hi
  simpa using hi

/-- Algorithm-layer correctness for `mul2_inplace_kernel`. -/
theorem mul2_inplace_kernel_correct
    (ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) (_hBlockSize : 0 < BLOCK_SIZE)
    (s : BlockState) (xs : Fin BLOCK_SIZE → ℝ)
    (h_x : InputLoadedAt s ptr BLOCK_SIZE xs) :
    ∀ i : Fin BLOCK_SIZE,
      let addr := s.pid * BLOCK_SIZE + i.val
      observeAt (exec (mul2_inplace_kernel ptr n_elements BLOCK_SIZE) s)
          ptr BLOCK_SIZE s.pid i
        = some (if addr < n_elements then 2 * xs i else s.readMem ptr addr) := by
  intro i
  have h_inj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE] => s.pid * BLOCK_SIZE + idx.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp [observeAt, exec, mul2_inplace_kernel, stepStmts, stepStmt, evalOp,
        Tile.bop, Tile.cop, NumericDType.add, NumericDType.mul,
        ComparableDType.lt]
  unfold InputLoadedAt at h_x
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj (i, PUnit.unit)]
  by_cases hi : s.pid * BLOCK_SIZE + i.val < n_elements
  · simp [hi, h_x]
  · simp [hi]

/-- Compute-facing correctness for `mul2_inplace_kernel`. -/
theorem mul2_inplace_kernel_compute_correct
    (ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) (hBlockSize : 0 < BLOCK_SIZE)
    (s : BlockState) (xs : Fin BLOCK_SIZE → ℝ)
    (h_x : InputLoadedAt s ptr BLOCK_SIZE xs) :
    ComputeKernel.ComputeCorrect
      (mul2_inplace_kernel ptr n_elements BLOCK_SIZE)
      (fun s0 s' =>
        s0 = s →
        ∀ i : Fin BLOCK_SIZE,
          let addr := s.pid * BLOCK_SIZE + i.val
          observeAt (some s') ptr BLOCK_SIZE s.pid i
            = some (if addr < n_elements then 2 * xs i else s.readMem ptr addr)) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel rfl
  intro s0 s' hExec hs0
  subst s0
  intro i
  have hi := mul2_inplace_kernel_correct ptr n_elements BLOCK_SIZE
    hBlockSize s xs h_x i
  rw [hExec] at hi
  simpa using hi

end VeriTile.Bench.TritonBenchG.TritonMul2
