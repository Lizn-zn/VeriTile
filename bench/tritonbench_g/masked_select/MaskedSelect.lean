import VeriTile.Triton

/-!
# `masked_select` — strict per-kernel correctness

`masked_select_kernel` is a compaction scatter: program `pid` loads block
`[pid·BLOCK_SIZE, (pid+1)·BLOCK_SIZE)` of the input, the boolean select mask, and
a precomputed exclusive-prefix-sum of the mask; each selected lane scatters its
input value to `out_ptr[prefix_sum − 1]`, i.e. its compacted output slot.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`masked_select_kernel[grid](...)`, the grid size
`cdiv(n_elements, BLOCK_SIZE)`, the host-side `prefix_sum`/output allocation, and
how the runtime composes per-program writes) is the *trusted boundary*, not a
proof obligation here. Because `pid` is universally quantified, the per-program
statement covers every program of the grid.

## Proof architecture

```
masked_select_kernel_compute_correct          ← ComputeCorrect over the masked scatter
  └─ masked_select_kernel_correct_of_exec      executed-state readback per active lane
       └─ masked_select_kernel_correct         ← algorithm-layer readback per active lane
```

The spec is the value-preserving gather/scatter `out[prefix_sum i − 1] = inp i`
on active lanes — no optimizer/reduction oracle applies.

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); `@triton.autotune` is not
modeled. The destination offset is data-dependent (read from `prefix_sum_ptr`),
so a per-lane injectivity hypothesis `hOutInj` on the store offsets is required
as a side condition — it captures the no-duplicate-destination property that the
host's prefix-sum guarantees. The `.to(tl.int1)` mask cast and dtype casts
reduce to identity at the algorithm layer. The statement is scoped to *active*
lanes (in-bounds and selected); out-of-bounds or unselected lanes are not part
of the realized write map.
-/

namespace VeriTile.Bench.TritonBenchG.MaskedSelect

open VeriTile.Triton

/-- Faithful transcription of `masked_select.py`'s `masked_select_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` -> Lean `Nat` parameter.
- `select_mask_ptr` and `prefix_sum_ptr` are typed Lean regions so their
  `tl.load` calls do not need extra `dtype=` kwargs. -/
def masked_select_kernel
    (inp_ptr : RegionName) (select_mask_ptr : Region .bool)
    (prefix_sum_ptr : Region .nat) (out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  offsets = pid * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  inp = tl.load(inp_ptr + offsets, mask=mask, other=0.0)
  select_mask = tl.load(select_mask_ptr + offsets,
    mask=mask, other=0.0).to(tl.int1)
  out_offset = tl.load(prefix_sum_ptr + offsets,
    mask=mask, other=0.0) - $(1)
  tl.store(out_ptr + out_offset, inp, mask=select_mask and mask)
}

def maskedSelectOffset (s : BlockState) (BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pid * BLOCK_SIZE + i.val

def maskedSelectStoreOffset
    (s : BlockState) (prefix_sum_ptr : RegionName) (n_elements BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : Nat :=
  (if maskedSelectOffset s BLOCK_SIZE i < n_elements then
      s.readMemValue .nat prefix_sum_ptr (maskedSelectOffset s BLOCK_SIZE i)
    else
      0) - 1

def active
    (s : BlockState) (select_mask_ptr : RegionName) (n_elements BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : Prop :=
  maskedSelectOffset s BLOCK_SIZE i < n_elements ∧
    s.readMemValue .bool select_mask_ptr (maskedSelectOffset s BLOCK_SIZE i) = Bool.true

instance activeDecidable
    (s : BlockState) (select_mask_ptr : RegionName) (n_elements BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) :
    Decidable (active s select_mask_ptr n_elements BLOCK_SIZE i) := by
  unfold active
  infer_instance

/-- Algorithm-layer cellwise correctness for active masked-select lanes. -/
theorem masked_select_kernel_correct
    (inp_ptr select_mask_ptr prefix_sum_ptr out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE =>
        maskedSelectStoreOffset s prefix_sum_ptr n_elements BLOCK_SIZE i)) :
    ∀ i : Fin BLOCK_SIZE,
      active s select_mask_ptr n_elements BLOCK_SIZE i →
      (exec (masked_select_kernel inp_ptr select_mask_ptr prefix_sum_ptr out_ptr
          n_elements BLOCK_SIZE) s).map
          (fun s' => s'.readMem out_ptr
            (maskedSelectStoreOffset s prefix_sum_ptr n_elements BLOCK_SIZE i))
        = some (s.readMem inp_ptr (maskedSelectOffset s BLOCK_SIZE i)) := by
  intro i hActive
  simp [exec, masked_select_kernel, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Tile.bop, Tile.cop, tile_elementwise,
        NumericDType.add, NumericDType.mul, NumericDType.sub, ComparableDType.lt,
        BlockState.readMemValue, Option.bind, Option.map]
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE] =>
        (if s.pid * BLOCK_SIZE + idx.1.val < n_elements then
            (match s.readMemTyped TileDType.nat prefix_sum_ptr
                (s.pid * BLOCK_SIZE + idx.1.val) with
              | some value => value
              | none => BlockState.defaultCarrier TileDType.nat)
          else
            0) - 1) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ h
    have hab : a = b := hOutInj (by
      simpa [maskedSelectStoreOffset, maskedSelectOffset, BlockState.pid_eq,
        BlockState.readMemValue] using h)
    subst b
    rfl
  simp [maskedSelectOffset, maskedSelectStoreOffset, BlockState.readMemValue]
  rcases hActive with ⟨hBounds, hMask⟩
  have hBoundsRaw : s.pid * BLOCK_SIZE + i.val < n_elements := by
    simpa [maskedSelectOffset] using hBounds
  have hMaskRaw :
      s.readMemValue .bool select_mask_ptr (s.pid * BLOCK_SIZE + i.val) = Bool.true := by
    simpa [maskedSelectOffset] using hMask
  have hMaskMatch :
      (match s.readMemTyped TileDType.bool select_mask_ptr
          (s.pid * BLOCK_SIZE + i.val) with
        | some value => value
        | none => BlockState.defaultCarrier TileDType.bool) = Bool.true := by
    simpa [BlockState.readMemValue] using hMaskRaw
  simpa [hBoundsRaw, hMaskRaw, hMaskMatch] using
    (BlockState.scatter_readback_prop_masked_nd
      (region := out_ptr)
      (shape := [BLOCK_SIZE])
      (offsetFn := fun idx : TileIndex [BLOCK_SIZE] =>
        (if s.pid * BLOCK_SIZE + idx.1.val < n_elements then
            (match s.readMemTyped TileDType.nat prefix_sum_ptr
                (s.pid * BLOCK_SIZE + idx.1.val) with
              | some value => value
              | none => BlockState.defaultCarrier TileDType.nat)
          else
            0) - 1)
      (valueFn := fun idx : TileIndex [BLOCK_SIZE] =>
        WithBot.unbotD 0
          (if s.pid * BLOCK_SIZE + idx.1.val < n_elements then
            some (s.readMem inp_ptr (s.pid * BLOCK_SIZE + idx.1.val))
          else
            some 0.0))
      (P := fun idx : TileIndex [BLOCK_SIZE] =>
        (s.pid * BLOCK_SIZE + idx.1.val < n_elements ∧
            (match s.readMemTyped TileDType.bool select_mask_ptr
                (s.pid * BLOCK_SIZE + idx.1.val) with
              | some value => value
              | none => BlockState.defaultCarrier TileDType.bool) = Bool.true) ∧
          s.pid * BLOCK_SIZE + idx.1.val < n_elements)
      _ hRawInj (i, PUnit.unit))

/-- Executed-state form of `masked_select_kernel_correct`. -/
theorem masked_select_kernel_correct_of_exec
    (inp_ptr select_mask_ptr prefix_sum_ptr out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE =>
        maskedSelectStoreOffset s prefix_sum_ptr n_elements BLOCK_SIZE i))
    (s' : BlockState)
    (hExec : exec (masked_select_kernel inp_ptr select_mask_ptr prefix_sum_ptr out_ptr
        n_elements BLOCK_SIZE) s = some s') :
    ∀ i : Fin BLOCK_SIZE,
      active s select_mask_ptr n_elements BLOCK_SIZE i →
      s'.readMem out_ptr
          (maskedSelectStoreOffset s prefix_sum_ptr n_elements BLOCK_SIZE i)
        = s.readMem inp_ptr (maskedSelectOffset s BLOCK_SIZE i) := by
  intro i hActive
  have h := masked_select_kernel_correct inp_ptr select_mask_ptr prefix_sum_ptr
    out_ptr n_elements BLOCK_SIZE s hOutInj i hActive
  rw [hExec] at h
  simpa using h

/-- Compute-facing correctness for active masked-select lanes. -/
theorem masked_select_kernel_compute_correct
    (inp_ptr select_mask_ptr prefix_sum_ptr out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE =>
        maskedSelectStoreOffset s prefix_sum_ptr n_elements BLOCK_SIZE i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := masked_select_kernel inp_ptr select_mask_ptr prefix_sum_ptr out_ptr
        n_elements BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s select_mask_ptr n_elements BLOCK_SIZE)
        (fun i => (out_ptr,
          maskedSelectStoreOffset s prefix_sum_ptr n_elements BLOCK_SIZE i)))
      (expected := fun i => s.readMem inp_ptr (maskedSelectOffset s BLOCK_SIZE i)) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [masked_select_kernel]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  exact masked_select_kernel_correct_of_exec inp_ptr select_mask_ptr prefix_sum_ptr
    out_ptr n_elements BLOCK_SIZE s hOutInj s' hExec i hActive

/-- Per-kernel output summary for `masked_select_kernel`: the DSL surface lowers
to the algorithm layer, and the data-dependent compaction scatter to `out_ptr` is
compute-correct — under the no-duplicate-destination hypothesis `hOutInj`, every
active (in-bounds and selected) lane scatters its input value to its compacted
slot. -/
theorem masked_select_kernel_output_summary
    (inp_ptr select_mask_ptr prefix_sum_ptr out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE =>
        maskedSelectStoreOffset s prefix_sum_ptr n_elements BLOCK_SIZE i)) :
    (∃ alg, (masked_select_kernel inp_ptr select_mask_ptr prefix_sum_ptr out_ptr
        n_elements BLOCK_SIZE).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := masked_select_kernel inp_ptr select_mask_ptr prefix_sum_ptr out_ptr
        n_elements BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s select_mask_ptr n_elements BLOCK_SIZE)
        (fun i => (out_ptr,
          maskedSelectStoreOffset s prefix_sum_ptr n_elements BLOCK_SIZE i)))
      (expected := fun i => s.readMem inp_ptr (maskedSelectOffset s BLOCK_SIZE i)) := by
  refine ⟨?_, ?_⟩
  · simp [masked_select_kernel, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  · exact masked_select_kernel_compute_correct inp_ptr select_mask_ptr prefix_sum_ptr
      out_ptr n_elements BLOCK_SIZE s hOutInj

end VeriTile.Bench.TritonBenchG.MaskedSelect
