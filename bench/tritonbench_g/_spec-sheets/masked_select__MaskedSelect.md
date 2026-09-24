# Spec sheet — `bench/tritonbench_g/masked_select/MaskedSelect.lean`

**Python source:** `bench/tritonbench_g/masked_select/masked_select.py`

## Public theorem: `masked_select_kernel_correctness`

<details><summary>docstring</summary>

```
/-- **The headline**: `masked_select_kernel` implements the value-preserving
compaction scatter on its scatter IO signature — for every disjoint flat
placement of the four buffers, every program id whose static-mask lanes are in
bounds (the write bound at the scatter cells `ids j − 1`), and every launch
state whose windows hold the data tile `xs`, the select tile `bs` (`.bool`
channel), and the prefix-sum tile `ids` (`.nat` channel) at the static-mask
lanes, the translated pointer kernel terminates; under the per-context
`WriteInj` antecedent (no two write-active lanes share a destination — the
host prefix-sum's no-duplicate-destination guarantee, the successor of the old
summary's `hOutInj` side condition, now demanded only at the write-active
lanes where it is actually true of a genuine prefix sum), every write-active
lane `j` (in-bounds and `bs j = Bool.true`, the kernel's `select_mask and mask`)
put `xs j` at its compacted slot `ids j − 1` of `out_ptr`; and every memory
cell off the raw scatter cells is unchanged — unconditionally. Proof:
`BoolScatterMasked2DKernelIO₁.Implements.intro` assembles the region-model
scatter triple with the flat-memory bridge side conditions. -/
```
</details>

**Statement:**
```lean
specification masked_select_kernel_correctness
    (inp_ptr select_mask_ptr prefix_sum_ptr out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    maskedSelectIO inp_ptr select_mask_ptr prefix_sum_ptr out_ptr n_elements
      BLOCK_SIZE ⊨ fun _ _ _ _ xs j => xs j
```

**Closed-form spec defs (transitive):** `maskedSelectIO`, `masked_select_kernel`

<details><summary><code>maskedSelectIO</code></summary>

```
/-- `masked_select_kernel`'s compaction-scatter **IO signature** — the whole
kernel-specific audit surface of the `⊨` headline:

* `inp`/`mbuf`/`idxbuf`/`out` — which buffer is which argument (data
  `inp_ptr`, `.bool` select gate `select_mask_ptr`, `.nat` prefix-sum index
  channel `prefix_sum_ptr`, scatter target `out_ptr`);
* `B = BLOCK_SIZE`, all three read windows at lane `j` =
  `pid₀ * BLOCK_SIZE + j` (the host-side 1-D `cdiv(n_elements, BLOCK_SIZE)`
  launch convention; the family's second program id is ignored);
* `mask` — the **static** bounds lanes `pid₀ * BLOCK_SIZE + j < n_elements`
  (the trace-safety/input superset);
* `writeMask` — the **data-dependent** store gate
  `pid₀ * BLOCK_SIZE + j < n_elements ∧ bs j = Bool.true`: exactly the kernel's
  `select_mask and mask`;
* `write` — the **data-dependent** scatter destination `ids j − 1`: the loaded
  `prefix_sum` value minus one, i.e. lane `j`'s compacted output slot.

The windows and masks are declared, not parsed from the kernel; the headline
**proves** the kernel's actual addressing, masking, and prefix-sum indexing
match them. Buffer sizes are not signature content: the headline quantifies
over every allocation whose extents cover the static-mask lanes and the
write-active scatter cells. -/
```
```lean
def maskedSelectIO (inp_ptr select_mask_ptr prefix_sum_ptr out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) : BoolScatterMasked2DKernelIO₁ where
  kernel := masked_select_kernel inp_ptr select_mask_ptr prefix_sum_ptr out_ptr
    n_elements BLOCK_SIZE
  inp := inp_ptr
  mbuf := select_mask_ptr
  idxbuf := prefix_sum_ptr
  out := out_ptr
  B := BLOCK_SIZE
  read := fun pid₀ _ j => pid₀ * BLOCK_SIZE + j.val
  readm := fun pid₀ _ j => pid₀ * BLOCK_SIZE + j.val
  readx := fun pid₀ _ j => pid₀ * BLOCK_SIZE + j.val
  write := fun _ _ ids j => ids j - 1
  mask := fun pid₀ _ j => pid₀ * BLOCK_SIZE + j.val < n_elements
  writeMask := fun pid₀ _ bs _ j =>
    pid₀ * BLOCK_SIZE + j.val < n_elements ∧ bs j = Bool.true
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
