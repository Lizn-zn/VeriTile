import VeriTile.Triton

/-!
# `dropout_triton` — strict per-kernel correctness

`_dropout` is inverted dropout with a *given* keep-mask: program `pid` loads
block `[pid·BLOCK_SIZE, (pid+1)·BLOCK_SIZE)` of the input `x_ptr` and the keep
buffer `x_keep_ptr`, computes `tl.where(x_keep, x / (1 - p), 0.0)` lane-wise,
and stores to `output_ptr`, masked by `offsets < n_elements`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`_dropout[grid](...)`, the grid size
`cdiv(n_elements, BLOCK_SIZE)`, and how the runtime composes per-program writes
into one buffer) is the *trusted boundary*, not a proof obligation here. Because
`pid` is universally quantified, the per-program statement covers every program
of the grid.

## Proof architecture

```
dropout_kernel_output_summary                 ← TOP THEOREM
  ├─ (toAlgorithm? = Except.ok _)             surface lowers to the algorithm layer
  └─ dropout_kernel_compute_correct           ← ComputeCorrect over the masked store
       └─ dropout_kernel_correct              ← algorithm-layer readback per lane
```

The spec `dropoutSpec` reads the keep-mask and input *from memory* at this
lane's offset and returns `x / (1 - p)` when the mask is set, else `0`.

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); `@triton.autotune` is not
modeled. The RNG is **not** modeled here: the upstream `_dropout` takes the
keep-mask `x_keep_ptr` as an explicit input buffer (the random keep/drop
decision is made by the caller), so the per-lane keep bit is read from memory
via `s.readMemValue .bool x_keep_ptr ...` and treated as given — the existing
theorems assume nothing about how it was produced. `x_keep_ptr` is a typed
boolean region. No output/input disjointness is assumed: the inputs are read
into registers before the scatter, so the result is correct even if
`output_ptr` aliases `x_ptr` or `x_keep_ptr`.
-/

namespace VeriTile.Bench.TritonBenchG.DropoutTriton

open VeriTile.Triton

/-- Faithful transcription of `dropout_triton.py`'s `_dropout`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` → Lean `Nat` parameter.
- `x_keep_ptr` is a typed Lean boolean region so its `tl.load` call does not
  need an extra `dtype=` kwarg. -/
def dropout_kernel
    (x_ptr : RegionName) (x_keep_ptr : Region .bool) (output_ptr : RegionName)
    (n_elements : Nat) (p : ℝ) (BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  x = tl.load(x_ptr + offsets, mask=mask)
  x_keep = tl.load(x_keep_ptr + offsets, mask=mask)
  output = tl.where(x_keep, x / (1 - $(p)), 0.0)
  tl.store(output_ptr + offsets, output, mask=mask)
}

def dropoutOffset (s : BlockState) (BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pid * BLOCK_SIZE + i.val

noncomputable def dropoutSpec
    (s : BlockState) (x_ptr x_keep_ptr : RegionName)
    (p : ℝ) (BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) : ℝ :=
  if s.readMemValue .bool x_keep_ptr (dropoutOffset s BLOCK_SIZE i) then
    s.readMem x_ptr (dropoutOffset s BLOCK_SIZE i) / (1 - p)
  else
    0.0

/-- Algorithm-layer correctness for `_dropout`.

Active lanes write the scaled-or-zero dropout result; inactive tail lanes are
preserved. -/
theorem dropout_kernel_correct
    (x_ptr x_keep_ptr output_ptr : RegionName)
    (n_elements : Nat) (p : ℝ) (BLOCK_SIZE : Nat)
    (s s' : BlockState)
    (hExec : exec (dropout_kernel x_ptr x_keep_ptr output_ptr
          n_elements p BLOCK_SIZE) s = some s') :
    ∀ i : Fin BLOCK_SIZE,
      let outAddr := dropoutOffset s BLOCK_SIZE i
      s'.readMem output_ptr outAddr =
        if outAddr < n_elements then
          dropoutSpec s x_ptr x_keep_ptr p BLOCK_SIZE i
        else s.readMem output_ptr outAddr := by
  intro i
  simp [exec, dropout_kernel, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Tile.bop, Tile.cop, Tile.select,
        NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
        ComparableDType.lt] at hExec
  subst s'
  simp only [dropoutOffset]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
        (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
  by_cases hBounds : s.pid * BLOCK_SIZE + i.val < n_elements
  · simp [hBounds, dropoutSpec, dropoutOffset]
    cases hKeep : s.readMemValue .bool x_keep_ptr (s.pid * BLOCK_SIZE + i.val) <;>
      simp
  · simp [hBounds]

/-- Compute-facing correctness for `_dropout`. -/
theorem dropout_kernel_compute_correct
    (x_ptr x_keep_ptr output_ptr : RegionName)
    (n_elements : Nat) (p : ℝ) (BLOCK_SIZE : Nat)
    (s : BlockState) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := dropout_kernel x_ptr x_keep_ptr output_ptr
        n_elements p BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => dropoutOffset s BLOCK_SIZE i < n_elements)
        (fun i => (output_ptr, dropoutOffset s BLOCK_SIZE i)))
      (expected := fun i => dropoutSpec s x_ptr x_keep_ptr p BLOCK_SIZE i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [dropout_kernel]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := dropout_kernel_correct x_ptr x_keep_ptr output_ptr
    n_elements p BLOCK_SIZE s s' hExec i
  simpa [hActive] using h

/-- Per-kernel output summary for `_dropout`: the DSL surface lowers to the
algorithm layer, and the masked store to `output_ptr` is compute-correct — every
active lane holds `dropoutSpec` (the keep-gated scaled input), out-of-bounds
lanes are preserved. -/
theorem dropout_kernel_output_summary
    (x_ptr x_keep_ptr output_ptr : RegionName)
    (n_elements : Nat) (p : ℝ) (BLOCK_SIZE : Nat)
    (s : BlockState) :
    (∃ alg, (dropout_kernel x_ptr x_keep_ptr output_ptr
        n_elements p BLOCK_SIZE).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := dropout_kernel x_ptr x_keep_ptr output_ptr
        n_elements p BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => dropoutOffset s BLOCK_SIZE i < n_elements)
        (fun i => (output_ptr, dropoutOffset s BLOCK_SIZE i)))
      (expected := fun i => dropoutSpec s x_ptr x_keep_ptr p BLOCK_SIZE i) := by
  refine ⟨?_, ?_⟩
  · simp [dropout_kernel, ComputeExpr.toAlgorithm?]
  · exact dropout_kernel_compute_correct x_ptr x_keep_ptr output_ptr
      n_elements p BLOCK_SIZE s

end VeriTile.Bench.TritonBenchG.DropoutTriton
