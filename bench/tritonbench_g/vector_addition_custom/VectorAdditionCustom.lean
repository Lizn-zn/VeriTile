import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Examples.Common

namespace VeriTile.Bench.TritonBenchG.VectorAdditionCustom

open VeriTile.Triton VeriTile.Examples

/-- Faithful 1:1 transcription of `vector_addition_custom.py`'s `_add_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK: tl.constexpr` → Lean `Nat` parameter. -/
def _add_kernel
    (A B C : RegionName)
    (size BLOCK : Nat) :
    ComputeKernel := triton {
  prog_id = tl.program_id(0)
  offs = prog_id * $(BLOCK) + tl.arange(0, $(BLOCK))
  a = tl.load(A + offs, mask=offs < $(size))
  b = tl.load(B + offs, mask=offs < $(size))
  tl.store(C + offs, a + b, mask=offs < $(size))
}

/-- Algorithm-layer correctness for `_add_kernel`.

Each active lane writes `A + B`; inactive tail lanes are preserved. -/
theorem add_kernel_correct
    (A B C : RegionName)
    (size BLOCK : Nat) (_hBlock : 0 < BLOCK)
    (s : BlockState) (as bs : Fin BLOCK → ℝ)
    (h_a : InputLoadedAt s A BLOCK as)
    (h_b : InputLoadedAt s B BLOCK bs) :
    ∀ i : Fin BLOCK,
      let addr := s.pid * BLOCK + i.val
      observeAt (exec (_add_kernel A B C size BLOCK) s) C BLOCK s.pid i
        = some (if addr < size then as i + bs i else s.readMem C addr) := by
  intro i
  have h_inj : Function.Injective
      (fun idx : TileIndex [BLOCK] => s.pid * BLOCK + idx.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp [observeAt, exec, _add_kernel, stepStmts, stepStmt, evalOp,
        Tile.bop, Tile.cop, NumericDType.add, NumericDType.mul,
        ComparableDType.lt]
  unfold InputLoadedAt at h_a h_b
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj (i, PUnit.unit)]
  by_cases hi : s.pid * BLOCK + i.val < size
  · simp [hi, h_a, h_b]
  · simp [hi]

/-- Compute-facing correctness for `_add_kernel`. -/
theorem add_kernel_compute_correct
    (A B C : RegionName)
    (size BLOCK : Nat) (hBlock : 0 < BLOCK)
    (s : BlockState) (as bs : Fin BLOCK → ℝ)
    (h_a : InputLoadedAt s A BLOCK as)
    (h_b : InputLoadedAt s B BLOCK bs) :
    ComputeCorrect.General
      (_add_kernel A B C size BLOCK)
      (fun s0 s' =>
        s0 = s →
        ∀ i : Fin BLOCK,
          let addr := s.pid * BLOCK + i.val
          observeAt (some s') C BLOCK s.pid i
            = some (if addr < size then as i + bs i else s.readMem C addr)) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel rfl
  intro s0 s' hExec hs0
  subst s0
  intro i
  have hi := add_kernel_correct A B C size BLOCK hBlock s as bs h_a h_b i
  rw [hExec] at hi
  simpa using hi

end VeriTile.Bench.TritonBenchG.VectorAdditionCustom
