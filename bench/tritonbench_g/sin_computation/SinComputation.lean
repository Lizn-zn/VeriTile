import VeriTile.Triton
import VeriTile.Examples.Common

/-!
# `sin_computation` — strict per-kernel correctness

`sin_kernel` is an elementwise sine: program `pid` loads block
`[pid·BLOCK_SIZE, (pid+1)·BLOCK_SIZE)` of `in_ptr0`, applies `tl.sin` lane-wise,
and stores to `out_ptr`, masked by `offsets < n_elements`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`sin_kernel[(n_elements,)](...)`, the grid size, and how
the runtime composes per-program writes into one buffer) is the *trusted
boundary*, not a proof obligation here. Because `pid` is universally quantified,
the per-program statement covers every program of the grid.

## Proof architecture

```
sin_kernel_output_summary                     ← TOP THEOREM
  ├─ (toAlgorithm? = Except.ok _)             surface lowers to the algorithm layer
  └─ sin_kernel_compute_correct               ← ComputeCorrect over the masked store
       └─ sin_kernel_correct                  ← algorithm-layer readback per lane
```

The spec is plain elementwise `Real.sin (xs i)`. Inputs are presented via
`InputLoadedAt` (the values each lane loads).

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float), and `tl.sin` is modeled by
the exact `Real.sin`. No output/input
disjointness is assumed: the input is read into registers before the scatter, so
the result is correct even if `out_ptr` aliases `in_ptr0`.
-/

namespace VeriTile.Bench.TritonBenchG.SinComputation

open VeriTile.Triton VeriTile.Examples

/-- Faithful 1:1 transcription of `sin_computation.py`'s `sin_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` → Lean `Nat` parameter. -/
def sin_kernel
    (in_ptr0 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  x = tl.load(in_ptr0 + offsets, mask=mask)
  output = tl.sin(x)
  tl.store(out_ptr + offsets, output, mask=mask)
}

/-- Algorithm-layer correctness for `sin_kernel`. -/
theorem sin_kernel_correct
    (in_ptr0 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) (_hBlockSize : 0 < BLOCK_SIZE)
    (s : BlockState) (xs : Fin BLOCK_SIZE → ℝ)
    (h_x : InputLoadedAt s in_ptr0 BLOCK_SIZE xs) :
    ∀ i : Fin BLOCK_SIZE,
      let addr := s.pid * BLOCK_SIZE + i.val
      observeAt (exec (sin_kernel in_ptr0 out_ptr n_elements BLOCK_SIZE) s)
          out_ptr BLOCK_SIZE s.pid i
        = some (if addr < n_elements then Real.sin (xs i)
                else s.readMem out_ptr addr) := by
  intro i
  simp [observeAt, exec, sin_kernel, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Tile.bop, Tile.uop, Tile.cop, NumericDType.add, NumericDType.mul,
        ComparableDType.lt, WithBot.realSin]
  unfold InputLoadedAt at h_x
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
        (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
  by_cases hi : s.pid * BLOCK_SIZE + i.val < n_elements
  · simp [hi, h_x]
  · simp [hi]

/-- Compute-facing correctness for `sin_kernel`. -/
theorem sin_kernel_compute_correct
    (in_ptr0 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) (hBlockSize : 0 < BLOCK_SIZE)
    (s : BlockState) (xs : Fin BLOCK_SIZE → ℝ)
    (h_x : InputLoadedAt s in_ptr0 BLOCK_SIZE xs) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := sin_kernel in_ptr0 out_ptr n_elements BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin BLOCK_SIZE => s.pid * BLOCK_SIZE + i.val < n_elements)
          (fun i => (out_ptr, s.pid * BLOCK_SIZE + i.val)))
      (expected := fun i => Real.sin (xs i)) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel rfl
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have hi := sin_kernel_correct in_ptr0 out_ptr n_elements BLOCK_SIZE
    hBlockSize s xs h_x i
  rw [hExec] at hi
  simp [observeAt, hActive] at hi
  exact hi

/-- Per-kernel output summary for `sin_kernel`: the DSL surface lowers to the
algorithm layer, and the masked store to `out_ptr` is compute-correct — every
active lane holds `Real.sin (xs i)`, out-of-bounds lanes are preserved. -/
theorem sin_kernel_output_summary
    (in_ptr0 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) (hBlockSize : 0 < BLOCK_SIZE)
    (s : BlockState) (xs : Fin BLOCK_SIZE → ℝ)
    (h_x : InputLoadedAt s in_ptr0 BLOCK_SIZE xs) :
    (∃ alg, (sin_kernel in_ptr0 out_ptr n_elements BLOCK_SIZE).toAlgorithm? =
        Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := sin_kernel in_ptr0 out_ptr n_elements BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin BLOCK_SIZE => s.pid * BLOCK_SIZE + i.val < n_elements)
          (fun i => (out_ptr, s.pid * BLOCK_SIZE + i.val)))
      (expected := fun i => Real.sin (xs i)) := by
  refine ⟨⟨_, rfl⟩, ?_⟩
  exact sin_kernel_compute_correct in_ptr0 out_ptr n_elements BLOCK_SIZE
    hBlockSize s xs h_x

end VeriTile.Bench.TritonBenchG.SinComputation
