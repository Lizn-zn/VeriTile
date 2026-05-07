import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Examples.Common

namespace VeriTile.Bench.TritonBenchG.ReluTritonKernel

open VeriTile.Triton VeriTile.Examples

/-- Faithful 1:1 transcription of `relu_triton_kernel.py`'s `relu_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `N: tl.constexpr` / `block_size: tl.constexpr` → Lean `Nat`.
- Python `if cond: body` → `if cond { body }`, the DSL-side gate (block syntax
  uses braces because Lean is whitespace-insensitive; the `if` keyword itself
  is shared). -/
def relu_kernel
    (x_ptr out_ptr : RegionName)
    (N block_size : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  block_start = pid * $(block_size)
  offsets = block_start + tl.arange(0, $(block_size))
  mask = offsets < $(N)
  x = tl.load(x_ptr + offsets, mask=mask)
  result = tl.where(x >= 0, x, 0.0)
  if pid == 0 {
    tl.store(out_ptr + offsets, result, mask=mask)
  }
}

/-- Algorithm-layer branch form of ReLU, matching `tl.where(x >= 0, x, 0)`. -/
noncomputable def reluSpec (x : ℝ) : ℝ := by
  classical
  exact WithBot.unbotD 0
    (if ComparableDType.real.ge (some x) (some 0) then
      (some x : WithBot ℝ)
    else (some 0.0 : WithBot ℝ))

/-- Algorithm-layer correctness for `relu_kernel`.

Only `pid = 0` writes. In that branch, active lanes write the branch-form ReLU
and inactive tail lanes are preserved; nonzero pids preserve the observed
output cells. -/
theorem relu_kernel_correct
    (x_ptr out_ptr : RegionName)
    (N block_size : Nat) (_hBlockSize : 0 < block_size)
    (s : BlockState) (xs : Fin block_size → ℝ)
    (h_x : InputLoadedAt s x_ptr block_size xs) :
    ∀ i : Fin block_size,
      let addr := s.pid * block_size + i.val
      observeAt (exec (relu_kernel x_ptr out_ptr N block_size) s)
          out_ptr block_size s.pid i
        = some (if s.pid = 0 ∧ addr < N then reluSpec (xs i)
                else s.readMem out_ptr addr) := by
  intro i
  by_cases hpid : s.pid = 0
  · have h_inj : Function.Injective
        (fun idx : TileIndex [block_size] => idx.1.val) := by
      rintro ⟨a, _⟩ ⟨b, _⟩ hab
      obtain rfl : a = b := Fin.ext hab
      rfl
    simp [observeAt, exec, relu_kernel, stepStmts, stepStmt, evalOp,
          Tile.bop, Tile.cop, Tile.select, NumericDType.add, NumericDType.mul,
          ComparableDType.lt, ComparableDType.eq, ComparableDType.ge, hpid]
    unfold InputLoadedAt at h_x
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj (i, PUnit.unit)]
    have hx0 : s.readMem x_ptr i.val = xs i := by
      simpa [hpid] using h_x i
    by_cases hi : i.val < N
    · simp [hi, hx0, reluSpec, ComparableDType.ge]
    · simp [hi]
  · simp [observeAt, exec, relu_kernel, stepStmts, stepStmt, evalOp,
          Tile.bop, Tile.cop, Tile.select, NumericDType.add, NumericDType.mul,
          ComparableDType.lt, ComparableDType.eq, ComparableDType.ge, hpid]

/-- Compute-facing correctness for `relu_kernel`. -/
theorem relu_kernel_compute_correct
    (x_ptr out_ptr : RegionName)
    (N block_size : Nat) (hBlockSize : 0 < block_size)
    (s : BlockState) (xs : Fin block_size → ℝ)
    (h_x : InputLoadedAt s x_ptr block_size xs) :
    ComputeCorrect.General
      (relu_kernel x_ptr out_ptr N block_size)
      (fun s0 s' =>
        s0 = s →
        ∀ i : Fin block_size,
          let addr := s.pid * block_size + i.val
          observeAt (some s') out_ptr block_size s.pid i
            = some (if s.pid = 0 ∧ addr < N then reluSpec (xs i)
                    else s.readMem out_ptr addr)) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel rfl
  intro s0 s' hExec hs0
  subst s0
  intro i
  have hi := relu_kernel_correct x_ptr out_ptr N block_size hBlockSize s xs h_x i
  rw [hExec] at hi
  simpa using hi

end VeriTile.Bench.TritonBenchG.ReluTritonKernel
