import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Examples.Common

namespace VeriTile.Bench.TritonBenchG.MulExponentCompensator

open VeriTile.Triton VeriTile.Examples

/-- The constexpr multiplier from `mul_exponent_compensator.py`. -/
noncomputable def exponentCompensator : ℝ :=
  (2 : ℝ) ^ (127 - 15)

/-- Faithful 1:1 transcription of `mul_exponent_compensator.py`'s
`mul_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python local constexpr `2.0 ** (127 - 15)` is represented by
  `exponentCompensator` and injected with `$(_)`.
- Python `BLOCK_SIZE: tl.constexpr` → Lean `Nat` parameter. -/
noncomputable def mul_kernel
    (src dst : RegionName)
    (BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  idxs = tl.program_id(0) * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
  x = tl.load(src + idxs)
  y = x * $(exponentCompensator)
  tl.store(dst + idxs, y)
}

/-- Algorithm-layer correctness for `mul_kernel`. -/
theorem mul_kernel_correct
    (src dst : RegionName)
    (BLOCK_SIZE : Nat) (_hBlockSize : 0 < BLOCK_SIZE)
    (s : BlockState) (xs : Fin BLOCK_SIZE → ℝ)
    (h_x : InputLoadedAt s src BLOCK_SIZE xs) :
    ∀ i : Fin BLOCK_SIZE,
      observeAt (exec (mul_kernel src dst BLOCK_SIZE) s)
          dst BLOCK_SIZE s.pid i
        = some (xs i * exponentCompensator) := by
  intro i
  have h_inj := injective_offset_singleton (n := BLOCK_SIZE) (s.pid * BLOCK_SIZE)
  simp [observeAt, exec, mul_kernel, stepStmts, stepStmt, evalOp,
        Tile.bop, NumericDType.add, NumericDType.mul,
        exponentCompensator]
  unfold InputLoadedAt at h_x
  rw [BlockState.scatter_readback_nd _ _ _ h_inj (i, PUnit.unit)]
  simp [h_x]

/-- Compute-facing correctness for `mul_kernel`. -/
theorem mul_kernel_compute_correct
    (src dst : RegionName)
    (BLOCK_SIZE : Nat) (hBlockSize : 0 < BLOCK_SIZE)
    (s : BlockState) (xs : Fin BLOCK_SIZE → ℝ)
    (h_x : InputLoadedAt s src BLOCK_SIZE xs) :
    ComputeCorrect.Realizes
      (kernel := mul_kernel src dst BLOCK_SIZE)
      (initialState := s)
      (write := fun i : Fin BLOCK_SIZE =>
          some (dst, s.pid * BLOCK_SIZE + i.val))
      (expected := fun i => xs i * exponentCompensator) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel rfl
  intro s0 s' hExec hs0
  subst s0
  intro i
  have hi := mul_kernel_correct src dst BLOCK_SIZE hBlockSize s xs h_x i
  rw [hExec] at hi
  simpa [observeAt] using hi

end VeriTile.Bench.TritonBenchG.MulExponentCompensator
