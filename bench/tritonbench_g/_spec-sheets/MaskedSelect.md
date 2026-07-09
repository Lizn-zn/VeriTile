# Spec sheet — `bench/tritonbench_g/masked_select/MaskedSelect.lean`

**Python source:** `bench/tritonbench_g/masked_select/masked_select.py`

## Public theorem: `masked_select_kernel_output_summary`

<details><summary>docstring</summary>

```
/-- Per-kernel output summary for `masked_select_kernel`: the DSL surface lowers
to the algorithm layer, and the data-dependent compaction scatter to `out_ptr` is
compute-correct — under the no-duplicate-destination hypothesis `hOutInj`, every
active (in-bounds and selected) lane scatters its input value to its compacted
slot. -/
```
</details>

**Statement:**
```lean
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
      (expected := fun i => s.readMem inp_ptr (maskedSelectOffset s BLOCK_SIZE i))
```

**Assumptions / layout contracts:**
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE =>
        maskedSelectStoreOffset s prefix_sum_ptr n_elements BLOCK_SIZE i)`

**Closed-form spec defs (transitive):** `maskedSelectStoreOffset`, `masked_select_kernel`, `active`, `maskedSelectOffset`

<details><summary><code>maskedSelectStoreOffset</code></summary>

```lean
def maskedSelectStoreOffset
    (s : BlockState) (prefix_sum_ptr : RegionName) (n_elements BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : Nat :=
  (if maskedSelectOffset s BLOCK_SIZE i < n_elements then
      s.readMemValue .nat prefix_sum_ptr (maskedSelectOffset s BLOCK_SIZE i)
    else
      0) - 1
```
</details>

<details><summary><code>masked_select_kernel</code></summary>

```
/-- Faithful transcription of `masked_select.py`'s `masked_select_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` -> Lean `Nat` parameter.
- `select_mask_ptr` and `prefix_sum_ptr` are typed Lean regions so their
  `tl.load` calls do not need extra `dtype=` kwargs. -/
```
```lean
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
```
</details>

<details><summary><code>active</code></summary>

```lean
def active
    (s : BlockState) (select_mask_ptr : RegionName) (n_elements BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : Prop :=
  maskedSelectOffset s BLOCK_SIZE i < n_elements ∧
    s.readMemValue .bool select_mask_ptr (maskedSelectOffset s BLOCK_SIZE i) = Bool.true
```
</details>

<details><summary><code>maskedSelectOffset</code></summary>

```lean
def maskedSelectOffset (s : BlockState) (BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pid * BLOCK_SIZE + i.val
```
</details>

## Also present (pinned special-case summaries)
- `masked_select_kernel_compute_correct`
