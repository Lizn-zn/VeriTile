import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Examples.Common

/-!
# `add_example` — strict per-kernel correctness

`add_kernel` is the canonical elementwise add: program `pid` loads block
`[pid·BLOCK_SIZE, (pid+1)·BLOCK_SIZE)` of two inputs, adds them lane-wise, and
stores to `out_ptr`, masked by `offsets < n_elements`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`add_kernel[(num_blocks,)](...)`, the grid size
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

Arithmetic is over `ℝ` (not bit-accurate IEEE float); `@triton.autotune` is not
modeled. No output/input disjointness is assumed: both inputs are read into
registers before the scatter, so the result is correct even if `out_ptr`
aliases an input.
-/

namespace VeriTile.Bench.TritonBenchG.AddExample

open VeriTile.Triton VeriTile.Examples

/-- Faithful 1:1 transcription of `add_example.py`'s `add_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` annotation → Lean `Nat` parameter
  (the `tl.constexpr` is implicit in Lean params).

Everything else is verbatim from the upstream kernel. -/
def add_kernel
    (in_ptr0 in_ptr1 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  x = tl.load(in_ptr0 + offsets, mask=mask)
  y = tl.load(in_ptr1 + offsets, mask=mask)
  output = x + y
  tl.store(out_ptr + offsets, output, mask=mask)
}

/-! ## Correctness -/

/-- Algorithm-layer correctness for `add_kernel`.

For each lane `i ∈ Fin BLOCK_SIZE`:
* In-bounds (`pid * BLOCK_SIZE + i < n_elements`): the output region holds
  `xs i + ys i`.
* Out-of-bounds: the value at `pid * BLOCK_SIZE + i` is preserved from the
  initial state (mask=false → no store).

No region-disjointness hypothesis: the kernel reads both inputs into local
registers BEFORE the scatter to `out_ptr`, so the result is correct even when
`out_ptr` aliases `in_ptr0` or `in_ptr1`. -/
theorem add_kernel_correct
    (in_ptr0 in_ptr1 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) (_hBlockSize : 0 < BLOCK_SIZE)
    (s : BlockState) (xs ys : Fin BLOCK_SIZE → ℝ)
    (h_x : InputLoadedAt s in_ptr0 BLOCK_SIZE xs)
    (h_y : InputLoadedAt s in_ptr1 BLOCK_SIZE ys) :
    ∀ i : Fin BLOCK_SIZE,
      let addr := s.pid * BLOCK_SIZE + i.val
      observeAt (exec (add_kernel in_ptr0 in_ptr1 out_ptr n_elements BLOCK_SIZE) s)
                out_ptr BLOCK_SIZE s.pid i
        = some (if addr < n_elements then xs i + ys i
                else s.readMem out_ptr addr) := by
  intro i
  have h_inj := injective_offset_singleton (n := BLOCK_SIZE) (s.pid * BLOCK_SIZE)
  simp [observeAt, exec, add_kernel, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Tile.bop, Tile.cop, NumericDType.add, NumericDType.mul,
        ComparableDType.lt]
  unfold InputLoadedAt at h_x h_y
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj (i, PUnit.unit)]
  by_cases hi : s.pid * BLOCK_SIZE + i.val < n_elements
  · simp [hi, h_x, h_y]
  · simp [hi]

/-- **Compute-facing `ComputeCorrect` for `add_kernel`.**

For any region names, any block-size > 0, any pre-loaded input tiles `xs`/`ys`,
the kernel writes `xs i + ys i` at every in-bounds lane and preserves
out-of-bounds lanes verbatim. -/
theorem add_kernel_compute_correct
    (in_ptr0 in_ptr1 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) (hBlockSize : 0 < BLOCK_SIZE)
    (s : BlockState) (xs ys : Fin BLOCK_SIZE → ℝ)
    (h_x : InputLoadedAt s in_ptr0 BLOCK_SIZE xs)
    (h_y : InputLoadedAt s in_ptr1 BLOCK_SIZE ys) :
    ComputeCorrect.Realizes
      (kernel := add_kernel in_ptr0 in_ptr1 out_ptr n_elements BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin BLOCK_SIZE => s.pid * BLOCK_SIZE + i.val < n_elements)
          (fun i => (out_ptr, s.pid * BLOCK_SIZE + i.val)))
      (expected := fun i => xs i + ys i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel rfl
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have hi := add_kernel_correct in_ptr0 in_ptr1 out_ptr n_elements BLOCK_SIZE
    hBlockSize s xs ys h_x h_y i
  rw [hExec] at hi
  simp [observeAt, hActive] at hi
  exact hi

/-- Per-kernel output summary for `add_kernel`: the DSL surface lowers to the
algorithm layer, and the masked store to `out_ptr` is compute-correct — every
active lane holds `xs i + ys i`, out-of-bounds lanes are preserved. -/
theorem add_kernel_output_summary
    (in_ptr0 in_ptr1 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) (hBlockSize : 0 < BLOCK_SIZE)
    (s : BlockState) (xs ys : Fin BLOCK_SIZE → ℝ)
    (h_x : InputLoadedAt s in_ptr0 BLOCK_SIZE xs)
    (h_y : InputLoadedAt s in_ptr1 BLOCK_SIZE ys) :
    (∃ alg, (add_kernel in_ptr0 in_ptr1 out_ptr n_elements BLOCK_SIZE).toAlgorithm? =
        Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := add_kernel in_ptr0 in_ptr1 out_ptr n_elements BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin BLOCK_SIZE => s.pid * BLOCK_SIZE + i.val < n_elements)
          (fun i => (out_ptr, s.pid * BLOCK_SIZE + i.val)))
      (expected := fun i => xs i + ys i) := by
  refine ⟨⟨_, rfl⟩, ?_⟩
  exact add_kernel_compute_correct in_ptr0 in_ptr1 out_ptr n_elements BLOCK_SIZE
    hBlockSize s xs ys h_x h_y

end VeriTile.Bench.TritonBenchG.AddExample
