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
  simp [observeAt, exec, _add_kernel, stepStmts, stepStmt, evalOp,
        Tile.bop, Tile.cop, NumericDType.add, NumericDType.mul,
        ComparableDType.lt]
  unfold InputLoadedAt at h_a h_b
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
        (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
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
    ComputeCorrect.Realizes
      (kernel := _add_kernel A B C size BLOCK)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin BLOCK => s.pid * BLOCK + i.val < size)
          (fun i => (C, s.pid * BLOCK + i.val)))
      (expected := fun i => as i + bs i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel rfl
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have hi := add_kernel_correct A B C size BLOCK hBlock s as bs h_a h_b i
  rw [hExec] at hi
  simp [observeAt, hActive] at hi
  exact hi

end VeriTile.Bench.TritonBenchG.VectorAdditionCustom
