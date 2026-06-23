import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Examples.Common

/-!
# `triton_mul2` — strict per-kernel correctness

The Python module defines two `@triton.jit` programs that both double their
input: `mul2_kernel` (out-of-place, `output = 2 * x` stored to `out_ptr`) and
`mul2_inplace_kernel` (in-place, doubling the elements of `ptr`). Each program
`pid` handles block `[pid·BLOCK_SIZE, (pid+1)·BLOCK_SIZE)`, masked by
`offsets < n_elements`.

## Scope

This file verifies **the Triton kernels themselves** — the per-program
`@triton.jit` bodies. The host launches (`mul2_kernel[grid](...)`,
`mul2_inplace_kernel[grid](...)`, the grid size `cdiv(n_elements, BLOCK_SIZE)`,
scheduling, and how the runtime composes per-program writes into one buffer) are
the *trusted boundary*, not proof obligations here. Because `pid` is universally
quantified, each per-program statement covers every program of the grid.

## Proof architecture

```
mul2_kernel_output_summary                    ← TOP THEOREM (covers both kernels)
  ├─ (mul2_kernel.toAlgorithm? = Except.ok _)         out-of-place surface lowers
  ├─ mul2_kernel_compute_correct                      ← ComputeCorrect, out-of-place store
  │    └─ mul2_kernel_correct                         ← algorithm-layer readback per lane
  ├─ (mul2_inplace_kernel.toAlgorithm? = Except.ok _) in-place surface lowers
  └─ mul2_inplace_kernel_compute_correct              ← ComputeCorrect, in-place store
       └─ mul2_inplace_kernel_correct                 ← algorithm-layer readback per lane
```

The spec is plain elementwise `2 * xs i` — no optimizer/reduction oracle
applies. Inputs are presented via `InputLoadedAt` (the values each lane loads).

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float). No output/input
disjointness is assumed: each kernel reads its input
into registers before the scatter, so `mul2_kernel` is correct even if `out_ptr`
aliases `in_ptr0`, and `mul2_inplace_kernel` is correct writing back in place.
-/

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
  simp [observeAt, exec, mul2_kernel, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Tile.bop, Tile.cop, NumericDType.add, NumericDType.mul,
        ComparableDType.lt]
  unfold InputLoadedAt at h_x
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
        (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
  by_cases hi : s.pid * BLOCK_SIZE + i.val < n_elements
  · simp [hi, h_x]
  · simp [hi]

/-- Compute-facing correctness for `mul2_kernel`. -/
theorem mul2_kernel_compute_correct
    (in_ptr0 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) (hBlockSize : 0 < BLOCK_SIZE)
    (s : BlockState) (xs : Fin BLOCK_SIZE → ℝ)
    (h_x : InputLoadedAt s in_ptr0 BLOCK_SIZE xs) :
    ComputeCorrect.Realizes
      (kernel := mul2_kernel in_ptr0 out_ptr n_elements BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin BLOCK_SIZE => s.pid * BLOCK_SIZE + i.val < n_elements)
          (fun i => (out_ptr, s.pid * BLOCK_SIZE + i.val)))
      (expected := fun i => 2 * xs i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel rfl
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have hi := mul2_kernel_correct in_ptr0 out_ptr n_elements BLOCK_SIZE
    hBlockSize s xs h_x i
  rw [hExec] at hi
  simp [observeAt, hActive] at hi
  exact hi

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
  simp [observeAt, exec, mul2_inplace_kernel, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Tile.bop, Tile.cop, NumericDType.add, NumericDType.mul,
        ComparableDType.lt]
  unfold InputLoadedAt at h_x
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
        (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
  by_cases hi : s.pid * BLOCK_SIZE + i.val < n_elements
  · simp [hi, h_x]
  · simp [hi]

/-- Compute-facing correctness for `mul2_inplace_kernel`. -/
theorem mul2_inplace_kernel_compute_correct
    (ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) (hBlockSize : 0 < BLOCK_SIZE)
    (s : BlockState) (xs : Fin BLOCK_SIZE → ℝ)
    (h_x : InputLoadedAt s ptr BLOCK_SIZE xs) :
    ComputeCorrect.Realizes
      (kernel := mul2_inplace_kernel ptr n_elements BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin BLOCK_SIZE => s.pid * BLOCK_SIZE + i.val < n_elements)
          (fun i => (ptr, s.pid * BLOCK_SIZE + i.val)))
      (expected := fun i => 2 * xs i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel rfl
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have hi := mul2_inplace_kernel_correct ptr n_elements BLOCK_SIZE
    hBlockSize s xs h_x i
  rw [hExec] at hi
  simp [observeAt, hActive] at hi
  exact hi

/-- Per-kernel output summary for `triton_mul2`'s two programs: both DSL surfaces
lower to the algorithm layer, and both masked stores are compute-correct — every
active lane holds `2 * xs i`, out-of-bounds lanes are preserved. The first
conjunct covers the out-of-place `mul2_kernel` store to `out_ptr`; the second
covers the in-place `mul2_inplace_kernel` store to `ptr`. -/
theorem mul2_kernel_output_summary
    (in_ptr0 out_ptr ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) (hBlockSize : 0 < BLOCK_SIZE)
    (s : BlockState) (xs xsInplace : Fin BLOCK_SIZE → ℝ)
    (h_x : InputLoadedAt s in_ptr0 BLOCK_SIZE xs)
    (h_xInplace : InputLoadedAt s ptr BLOCK_SIZE xsInplace) :
    ((∃ alg, (mul2_kernel in_ptr0 out_ptr n_elements BLOCK_SIZE).toAlgorithm? =
        Except.ok alg) ∧
      ComputeCorrect.Realizes
        (kernel := mul2_kernel in_ptr0 out_ptr n_elements BLOCK_SIZE)
        (initialState := s)
        (write := ComputeCorrect.WriteMap.writeIf
            (fun i : Fin BLOCK_SIZE => s.pid * BLOCK_SIZE + i.val < n_elements)
            (fun i => (out_ptr, s.pid * BLOCK_SIZE + i.val)))
        (expected := fun i => 2 * xs i)) ∧
    ((∃ alg, (mul2_inplace_kernel ptr n_elements BLOCK_SIZE).toAlgorithm? =
        Except.ok alg) ∧
      ComputeCorrect.Realizes
        (kernel := mul2_inplace_kernel ptr n_elements BLOCK_SIZE)
        (initialState := s)
        (write := ComputeCorrect.WriteMap.writeIf
            (fun i : Fin BLOCK_SIZE => s.pid * BLOCK_SIZE + i.val < n_elements)
            (fun i => (ptr, s.pid * BLOCK_SIZE + i.val)))
        (expected := fun i => 2 * xsInplace i)) := by
  refine ⟨⟨⟨_, rfl⟩, ?_⟩, ⟨⟨_, rfl⟩, ?_⟩⟩
  · exact mul2_kernel_compute_correct in_ptr0 out_ptr n_elements BLOCK_SIZE
      hBlockSize s xs h_x
  · exact mul2_inplace_kernel_compute_correct ptr n_elements BLOCK_SIZE
      hBlockSize s xsInplace h_xInplace

end VeriTile.Bench.TritonBenchG.TritonMul2
