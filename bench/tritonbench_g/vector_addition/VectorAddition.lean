import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Examples.Common

/-!
# `vector_addition` — strict per-kernel correctness

`add_kernel` is the canonical elementwise add: program `pid` loads block
`[pid·BLOCK_SIZE, (pid+1)·BLOCK_SIZE)` of two inputs, adds them lane-wise, and
stores to `output_ptr`, masked by `offsets < n_elements`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`add_kernel[grid](...)`, the grid size
`cdiv(n_elements, BLOCK_SIZE)`, and how the runtime composes per-program writes
into one buffer) is the *trusted boundary*, not a proof obligation here. Because
`pid` is universally quantified, the per-program statement covers every program
of the grid.

## Proof architecture

```
add_kernel_output_summary                     ← TOP THEOREM
  ├─ (toAlgorithm? = Except.ok _)             surface lowers to the algorithm layer
  └─ add_kernel_compute_correct               ← ComputeCorrect over the masked store
       └─ add_kernel_correct                  ← algorithm-layer readback per lane
```

The spec is plain elementwise `xs i + ys i` — no optimizer/reduction oracle
applies. Inputs are presented via `InputLoadedAt` (the values each lane loads).

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float). No output/input
disjointness is assumed: both inputs are read into
registers before the scatter, so the result is correct even if `output_ptr`
aliases an input.
-/

namespace VeriTile.Bench.TritonBenchG.VectorAddition

open VeriTile.Triton VeriTile.Examples

/-- Faithful 1:1 transcription of `vector_addition.py`'s `add_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` → Lean `Nat` parameter. -/
def add_kernel
    (x_ptr y_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  x = tl.load(x_ptr + offsets, mask=mask)
  y = tl.load(y_ptr + offsets, mask=mask)
  output = x + y
  tl.store(output_ptr + offsets, output, mask=mask)
}

/-- Algorithm-layer correctness for `add_kernel`.

Each active lane writes `x + y`; inactive tail lanes are preserved. -/
theorem add_kernel_correct
    (x_ptr y_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) (_hBlockSize : 0 < BLOCK_SIZE)
    (s : BlockState) (xs ys : Fin BLOCK_SIZE → ℝ)
    (h_x : InputLoadedAt s x_ptr BLOCK_SIZE xs)
    (h_y : InputLoadedAt s y_ptr BLOCK_SIZE ys) :
    ∀ i : Fin BLOCK_SIZE,
      let addr := s.pid * BLOCK_SIZE + i.val
      observeAt (exec (add_kernel x_ptr y_ptr output_ptr n_elements BLOCK_SIZE) s)
          output_ptr BLOCK_SIZE s.pid i
        = some (if addr < n_elements then xs i + ys i
                else s.readMem output_ptr addr) := by
  intro i
  simp [observeAt, exec, add_kernel, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Tile.bop, Tile.cop, NumericDType.add, NumericDType.mul,
        ComparableDType.lt]
  unfold InputLoadedAt at h_x h_y
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
        (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
  by_cases hi : s.pid * BLOCK_SIZE + i.val < n_elements
  · simp [hi, h_x, h_y]
  · simp [hi]

/-- Compute-facing correctness for `add_kernel`. -/
theorem add_kernel_compute_correct
    (x_ptr y_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) (hBlockSize : 0 < BLOCK_SIZE)
    (s : BlockState) (xs ys : Fin BLOCK_SIZE → ℝ)
    (h_x : InputLoadedAt s x_ptr BLOCK_SIZE xs)
    (h_y : InputLoadedAt s y_ptr BLOCK_SIZE ys) :
    ComputeCorrect.Realizes
      (kernel := add_kernel x_ptr y_ptr output_ptr n_elements BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin BLOCK_SIZE => s.pid * BLOCK_SIZE + i.val < n_elements)
          (fun i => (output_ptr, s.pid * BLOCK_SIZE + i.val)))
      (expected := fun i => xs i + ys i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel rfl
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have hi := add_kernel_correct x_ptr y_ptr output_ptr n_elements BLOCK_SIZE
    hBlockSize s xs ys h_x h_y i
  rw [hExec] at hi
  simp [observeAt, hActive] at hi
  exact hi

/-- Per-kernel output summary for `add_kernel`: the DSL surface lowers to the
algorithm layer, and the masked store to `output_ptr` is compute-correct — every
active lane holds `xs i + ys i`, out-of-bounds lanes are preserved. -/
theorem add_kernel_output_summary
    (x_ptr y_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) (hBlockSize : 0 < BLOCK_SIZE)
    (s : BlockState) (xs ys : Fin BLOCK_SIZE → ℝ)
    (h_x : InputLoadedAt s x_ptr BLOCK_SIZE xs)
    (h_y : InputLoadedAt s y_ptr BLOCK_SIZE ys) :
    (∃ alg, (add_kernel x_ptr y_ptr output_ptr n_elements BLOCK_SIZE).toAlgorithm? =
        Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := add_kernel x_ptr y_ptr output_ptr n_elements BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin BLOCK_SIZE => s.pid * BLOCK_SIZE + i.val < n_elements)
          (fun i => (output_ptr, s.pid * BLOCK_SIZE + i.val)))
      (expected := fun i => xs i + ys i) := by
  refine ⟨⟨_, rfl⟩, ?_⟩
  exact add_kernel_compute_correct x_ptr y_ptr output_ptr n_elements BLOCK_SIZE
    hBlockSize s xs ys h_x h_y

end VeriTile.Bench.TritonBenchG.VectorAddition
