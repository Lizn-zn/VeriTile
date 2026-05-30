import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Examples.Common

/-!
# `add_value` — strict per-kernel correctness

`puzzle1_kernel` is an elementwise add-a-scalar: program `pid` loads block
`[pid·BLOCK_SIZE, (pid+1)·BLOCK_SIZE)` of one input, adds the host-supplied
scalar `value` lane-wise, and stores to `output_ptr`, masked by `offsets < N`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`puzzle1_kernel[grid](...)`, the grid size
`cdiv(N, BLOCK_SIZE)`, scheduling, and how the runtime composes per-program
writes into one buffer) is the *trusted boundary*, not a proof obligation here.
Because `pid` is universally quantified, the per-program statement covers every
program of the grid.

## Proof architecture

```
puzzle1_kernel_output_summary                 ← TOP THEOREM
  ├─ (toAlgorithm? = Except.ok _)             surface lowers to the algorithm layer
  └─ puzzle1_kernel_compute_correct           ← ComputeCorrect over the masked store
       └─ puzzle1_kernel_correct              ← algorithm-layer readback per lane
```

The spec is plain elementwise `xs i + value` — no optimizer/reduction oracle
applies. Inputs are presented via `InputLoadedAt` (the values each lane loads).

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); `@triton.autotune` is not
modeled. No output/input disjointness is assumed: the input is read into
registers before the scatter, so the result is correct even if `output_ptr`
aliases `x_ptr`.
-/

namespace VeriTile.Bench.TritonBenchG.AddValue

open VeriTile.Triton VeriTile.Examples

/-- Faithful 1:1 transcription of `add_value.py`'s `puzzle1_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` → Lean `Nat` parameter.
- `value` (Lean `ℝ` parameter) injected via `$(...)`. -/
def puzzle1_kernel
    (x_ptr output_ptr : RegionName)
    (N BLOCK_SIZE : Nat) (value : ℝ) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(N)
  x = tl.load(x_ptr + offsets, mask=mask)
  output = x + $(value)
  tl.store(output_ptr + offsets, output, mask=mask)
}

/-! ## Correctness -/

/-- Algorithm-layer correctness for `puzzle1_kernel`.

For each lane `i ∈ Fin BLOCK_SIZE`:
* In-bounds (`pid * BLOCK_SIZE + i < N`): the output region holds `xs i + value`.
* Out-of-bounds: the value at `pid * BLOCK_SIZE + i` is preserved from the
  initial state (mask=false → no store). -/
theorem puzzle1_kernel_correct
    (x_ptr output_ptr : RegionName)
    (N BLOCK_SIZE : Nat) (value : ℝ) (_hBlockSize : 0 < BLOCK_SIZE)
    (s : BlockState) (xs : Fin BLOCK_SIZE → ℝ)
    (h_x : InputLoadedAt s x_ptr BLOCK_SIZE xs) :
    ∀ i : Fin BLOCK_SIZE,
      let addr := s.pid * BLOCK_SIZE + i.val
      observeAt (exec (puzzle1_kernel x_ptr output_ptr N BLOCK_SIZE value) s)
                output_ptr BLOCK_SIZE s.pid i
        = some (if addr < N then xs i + value
                else s.readMem output_ptr addr) := by
  intro i
  have h_inj := injective_offset_singleton (n := BLOCK_SIZE) (s.pid * BLOCK_SIZE)
  simp [observeAt, exec, puzzle1_kernel, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Tile.bop, Tile.cop, NumericDType.add, NumericDType.mul,
        ComparableDType.lt]
  unfold InputLoadedAt at h_x
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj (i, PUnit.unit)]
  by_cases hi : s.pid * BLOCK_SIZE + i.val < N
  · simp [hi, h_x]
  · simp [hi]

/-- **Compute-facing `ComputeCorrect` for `puzzle1_kernel`.**

For any region names, any block-size > 0, any pre-loaded input tile `xs`, the
kernel writes `xs i + value` at every in-bounds lane and preserves out-of-bounds
lanes verbatim. -/
theorem puzzle1_kernel_compute_correct
    (x_ptr output_ptr : RegionName)
    (N BLOCK_SIZE : Nat) (value : ℝ) (hBlockSize : 0 < BLOCK_SIZE)
    (s : BlockState) (xs : Fin BLOCK_SIZE → ℝ)
    (h_x : InputLoadedAt s x_ptr BLOCK_SIZE xs) :
    ComputeCorrect.Realizes
      (kernel := puzzle1_kernel x_ptr output_ptr N BLOCK_SIZE value)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin BLOCK_SIZE => s.pid * BLOCK_SIZE + i.val < N)
          (fun i => (output_ptr, s.pid * BLOCK_SIZE + i.val)))
      (expected := fun i => xs i + value) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel rfl
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have hi := puzzle1_kernel_correct x_ptr output_ptr N BLOCK_SIZE value
    hBlockSize s xs h_x i
  rw [hExec] at hi
  simp [observeAt, hActive] at hi
  exact hi

/-- Per-kernel output summary for `puzzle1_kernel`: the DSL surface lowers to the
algorithm layer, and the masked store to `output_ptr` is compute-correct — every
active lane holds `xs i + value`, out-of-bounds lanes are preserved. -/
theorem puzzle1_kernel_output_summary
    (x_ptr output_ptr : RegionName)
    (N BLOCK_SIZE : Nat) (value : ℝ) (hBlockSize : 0 < BLOCK_SIZE)
    (s : BlockState) (xs : Fin BLOCK_SIZE → ℝ)
    (h_x : InputLoadedAt s x_ptr BLOCK_SIZE xs) :
    (∃ alg, (puzzle1_kernel x_ptr output_ptr N BLOCK_SIZE value).toAlgorithm? =
        Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := puzzle1_kernel x_ptr output_ptr N BLOCK_SIZE value)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin BLOCK_SIZE => s.pid * BLOCK_SIZE + i.val < N)
          (fun i => (output_ptr, s.pid * BLOCK_SIZE + i.val)))
      (expected := fun i => xs i + value) := by
  refine ⟨⟨_, rfl⟩, ?_⟩
  exact puzzle1_kernel_compute_correct x_ptr output_ptr N BLOCK_SIZE value
    hBlockSize s xs h_x

end VeriTile.Bench.TritonBenchG.AddValue
