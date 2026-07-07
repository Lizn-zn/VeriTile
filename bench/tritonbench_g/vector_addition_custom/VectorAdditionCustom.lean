import VeriTile.Core
import VeriTile.Semantics
import VeriTile.Float
import VeriTile.Frontend.Triton.DSL
import VeriTile.Examples.Common

/-!
# `vector_addition_custom` — strict per-kernel correctness

`_add_kernel` is an elementwise add: program `prog_id` loads block
`[prog_id·BLOCK, (prog_id+1)·BLOCK)` of inputs `A` and `B`, adds them lane-wise,
and stores to `C`, masked by `offs < size`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`_add_kernel[grid](...)`, the grid size
`cdiv(size, BLOCK)`, the host-side `BLOCK` choice, and how the runtime composes
per-program writes into one buffer) is the *trusted boundary*, not a proof
obligation here. Because `prog_id` is universally quantified (as `s.pid`), the
per-program statement covers every program of the grid.

## Proof architecture

```
add_kernel_output_summary                     ← TOP THEOREM
  ├─ (toAlgorithm? = Except.ok _)             surface lowers to the algorithm layer
  └─ add_kernel_compute_correct               ← ComputeCorrect over the masked store
       └─ add_kernel_correct                  ← algorithm-layer readback per lane
```

The spec is plain elementwise `as i + bs i` — no optimizer/reduction oracle
applies. Inputs are presented via `InputLoadedAt` (the values each lane loads).

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float). No output/input
disjointness is assumed: both inputs are read into
registers before the scatter, so the result is correct even if `C` aliases `A`
or `B`.
-/

namespace VeriTile.Bench.TritonBenchG.VectorAdditionCustom

open VeriTile VeriTile.Examples

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
  simp [observeAt, exec, _add_kernel, stepStmts, stepStmt, evalOp, evalOp.eq_def,
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

/-- Per-kernel output summary for `_add_kernel`: the DSL surface lowers to the
algorithm layer, and the masked store to `C` is compute-correct — every active
lane holds `as i + bs i`, out-of-bounds lanes are preserved. -/
theorem add_kernel_output_summary
    (A B C : RegionName)
    (size BLOCK : Nat) (hBlock : 0 < BLOCK)
    (s : BlockState) (as bs : Fin BLOCK → ℝ)
    (h_a : InputLoadedAt s A BLOCK as)
    (h_b : InputLoadedAt s B BLOCK bs) :
    (∃ alg, (_add_kernel A B C size BLOCK).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := _add_kernel A B C size BLOCK)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin BLOCK => s.pid * BLOCK + i.val < size)
          (fun i => (C, s.pid * BLOCK + i.val)))
      (expected := fun i => as i + bs i) := by
  refine ⟨⟨_, rfl⟩, ?_⟩
  exact add_kernel_compute_correct A B C size BLOCK hBlock s as bs h_a h_b

end VeriTile.Bench.TritonBenchG.VectorAdditionCustom
