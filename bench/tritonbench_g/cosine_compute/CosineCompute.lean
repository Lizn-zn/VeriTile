import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Examples.Common

/-!
# `cosine_compute` — strict per-kernel correctness

`cos_func` is an elementwise cosine: program `pid` loads block
`[pid·BLOCK_SIZE, (pid+1)·BLOCK_SIZE)` of input `a`, casts each lane to float32,
applies `tl.cos`, and stores `cos(x)` to output `b`, masked by
`offset < n_elements`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`cos_func[(grid_size,1,1)](...)`, the grid size
`cdiv(n_elements, block_size)`, the host-side `block_size` choice, and how the
runtime composes per-program writes into one buffer) is the *trusted boundary*,
not a proof obligation here. Because `pid` is universally quantified, the
per-program statement covers every program of the grid.

## Proof architecture

```
cos_func_output_summary                       ← TOP THEOREM
  ├─ (toAlgorithm? = Except.ok _)             surface lowers to the algorithm layer
  └─ cos_func_compute_correct                 ← ComputeCorrect over the masked store
       └─ cos_func_correct                    ← algorithm-layer readback per lane
```

The spec is plain elementwise `Real.cos (xs i)` — no optimizer/reduction oracle
applies. Inputs are presented via `InputLoadedAt` (the values each lane loads).

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float). The `(a_value).to(tl.float32)` cast reduces to the identity at the
algorithm layer (post-erasure all dtypes unify to `ℝ`). No output/input
disjointness is assumed: the input is read into registers before the scatter, so
the result is correct even if `b` aliases `a`.
-/

namespace VeriTile.Bench.TritonBenchG.CosineCompute

open VeriTile.Triton VeriTile.Examples

/-- Faithful 1:1 transcription of `cosine_compute.py`'s `cos_func`.

Allowed mechanical Lean-syntax-only changes:
-/
def cos_func
    (a b : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  offset = tl.program_id(0) * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
  mask = offset < $(n_elements)
  a_value = tl.load(a + offset, mask=mask)
  b_value = tl.cos((a_value).to(tl.float32))
  tl.store(b + offset, b_value, mask=mask)
}

/-! ## Correctness -/

/-- Algorithm-layer correctness for `cos_func`.

For each lane `i ∈ Fin BLOCK_SIZE`:
* In-bounds (`pid * BLOCK_SIZE + i < n_elements`): the output region holds
  `Real.cos (xs i)`.
* Out-of-bounds: the value at `pid * BLOCK_SIZE + i` is preserved from the
  initial state (mask=false → no store).

The `(a_value).to(tl.float32)` cast at the compute layer reduces to
`Op.castFloat`, which is the identity at the algorithm layer (post-erasure
all dtypes unify to `ℝ`). -/
theorem cos_func_correct
    (a b : RegionName)
    (n_elements BLOCK_SIZE : Nat) (_hBlockSize : 0 < BLOCK_SIZE)
    (s : BlockState) (xs : Fin BLOCK_SIZE → ℝ)
    (h_x : InputLoadedAt s a BLOCK_SIZE xs) :
    ∀ i : Fin BLOCK_SIZE,
      let addr := s.pid * BLOCK_SIZE + i.val
      observeAt (exec (cos_func a b n_elements BLOCK_SIZE) s)
                b BLOCK_SIZE s.pid i
        = some (if addr < n_elements then Real.cos (xs i)
                else s.readMem b addr) := by
  intro i
  have h_inj := injective_offset_singleton (n := BLOCK_SIZE) (s.pid * BLOCK_SIZE)
  simp [observeAt, exec, cos_func, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Tile.bop, Tile.uop, Tile.cop, NumericDType.add, NumericDType.mul,
        ComparableDType.lt, WithBot.realCos,
        FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot]
  unfold InputLoadedAt at h_x
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj (i, PUnit.unit)]
  by_cases hi : s.pid * BLOCK_SIZE + i.val < n_elements
  · simp [hi, h_x]
  · simp [hi]

/-- **Compute-facing `ComputeCorrect` for `cos_func`.**

For any region names, any block-size > 0, any pre-loaded input tile `xs`, the
kernel writes `Real.cos (xs i)` at every in-bounds lane and preserves
out-of-bounds lanes verbatim. -/
theorem cos_func_compute_correct
    (a b : RegionName)
    (n_elements BLOCK_SIZE : Nat) (hBlockSize : 0 < BLOCK_SIZE)
    (s : BlockState) (xs : Fin BLOCK_SIZE → ℝ)
    (h_x : InputLoadedAt s a BLOCK_SIZE xs) :
    ComputeCorrect.Realizes
      (kernel := cos_func a b n_elements BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin BLOCK_SIZE => s.pid * BLOCK_SIZE + i.val < n_elements)
          (fun i => (b, s.pid * BLOCK_SIZE + i.val)))
      (expected := fun i => Real.cos (xs i)) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel rfl
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have hi := cos_func_correct a b n_elements BLOCK_SIZE
    hBlockSize s xs h_x i
  rw [hExec] at hi
  simp [observeAt, hActive] at hi
  exact hi

/-- Per-kernel output summary for `cos_func`: the DSL surface lowers to the
algorithm layer, and the masked store to `b` is compute-correct — every active
lane holds `Real.cos (xs i)`, out-of-bounds lanes are preserved. -/
theorem cos_func_output_summary
    (a b : RegionName)
    (n_elements BLOCK_SIZE : Nat) (hBlockSize : 0 < BLOCK_SIZE)
    (s : BlockState) (xs : Fin BLOCK_SIZE → ℝ)
    (h_x : InputLoadedAt s a BLOCK_SIZE xs) :
    (∃ alg, (cos_func a b n_elements BLOCK_SIZE).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := cos_func a b n_elements BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin BLOCK_SIZE => s.pid * BLOCK_SIZE + i.val < n_elements)
          (fun i => (b, s.pid * BLOCK_SIZE + i.val)))
      (expected := fun i => Real.cos (xs i)) := by
  refine ⟨⟨_, rfl⟩, ?_⟩
  exact cos_func_compute_correct a b n_elements BLOCK_SIZE hBlockSize s xs h_x

end VeriTile.Bench.TritonBenchG.CosineCompute
