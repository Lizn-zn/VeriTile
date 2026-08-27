# Spec sheet — `bench/tritonbench_g/destindex_copy_kv1/DestindexCopyKv1.lean`

**Python source:** `bench/tritonbench_g/destindex_copy_kv1/destindex_copy_kv1.py`

## Public theorem: `fwd_kernel_destindex_copy_kv_correctness`

<details><summary>docstring</summary>

```
/-- **The headline**: the dest-indexed KV scatter implements the pure copy `xs`
on its metadata IO signature — for every disjoint flat placement of the
buffers, every program id whose declared cells/lanes are in bounds, and every
launch state pinning the loaded destination row to `m₁` and the active source
lanes to `xs`, the translated pointer kernel terminates, every active lane
(`head < head_num ∧ dim < head_dim`) of the **`m₁`-indexed** output row holds
`xs j`, and every other memory cell is unchanged.

Side condition `hInj` (required for truth, not convenience): the tile part of
the output address map is injective, i.e. no two cells of one program's tile
scatter to the same cell. Without it the masked scatter is last-writer-wins
and the per-lane readback is false. It does not mention `Dest_loc`: the loaded
row only shifts every address by the constant `m₁ · stride_o_bs`.

Proof: `MetaMasked2DKernelIO₁.Implements.intro` assembles the region-model
metadata triple with the flat-memory bridge side conditions. -/
```
</details>

**Statement:**
```lean
specification fwd_kernel_destindex_copy_kv_correctness
    (K : RegionName) (Dest_loc : Region .nat) (Out : RegionName)
    (stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
      head_num head_dim BLOCK_DMODEL BLOCK_HEAD : Nat)
    (hInj : Function.Injective
      (fun idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL] =>
        stride_o_h * headIndex idx + stride_o_d * dimIndex idx)) :
    destindexCopyKvIO K Dest_loc Out
        stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
        head_num head_dim BLOCK_DMODEL BLOCK_HEAD
      ⊨ fun _ _ _ _ xs => xs
```

**Assumptions / layout contracts:**
- `hInj : Function.Injective
      (fun idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL] =>
        stride_o_h * headIndex idx + stride_o_d * dimIndex idx)`

**Closed-form spec defs (transitive):** `headIndex`, `dimIndex`, `destindexCopyKvIO`, `fwd_kernel_destindex_copy_kv`

<details><summary><code>headIndex</code></summary>

```lean
def headIndex (idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL]) : Nat :=
  idx.1.val
```
</details>

<details><summary><code>dimIndex</code></summary>

```lean
def dimIndex (idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL]) : Nat :=
  idx.2.1.val
```
</details>

<details><summary><code>destindexCopyKvIO</code></summary>

```
/-- `_fwd_kernel_destindex_copy_kv`'s metadata-genre **IO signature** — the
whole kernel-specific audit surface of the `⊨` headline
(`MetaMasked2DKernelIO₁`, the genre where a value *loaded from memory* drives
the windows):

* `mbuf1 = mbuf2 = Dest_loc`, `mwin1 = mwin2 = pid₀` — the `.nat` metadata
  slot: program `cur_index = pid₀` reads cell `pid₀` of `Dest_loc`, yielding
  the destination row `m₁`. This kernel has **one** slot; the genre carries
  two, so the second is wired to the same cell (hence `m₂ = m₁`) and is
  ignored by every window;
* `inp = K`, `out = Out`;
* `B = BLOCK_HEAD * BLOCK_DMODEL` — the flattened `[head, dim]` tile: lane `j`
  is cell `(j / BLOCK_DMODEL, j % BLOCK_DMODEL)`;
* `read` — lane `j` reads `pid₀·stride_k_bs + stride_k_h·head + stride_k_d·dim`
  (the program's own source row);
* `write` — lane `j` writes `m₁·stride_o_bs + stride_o_h·head +
  stride_o_d·dim`: the address **eats the loaded slot value**, which is the
  whole point of the scatter;
* `mask` (and the defaulted `writeMask`) — the active lanes
  `head < head_num ∧ dim < head_dim`.

The slot cell, windows, and masks are declared, not parsed from the kernel;
the headline **proves** the kernel's actual slot load, addressing and masking
match them. Buffer sizes are not signature content: the headline quantifies
over every allocation whose extents cover the declared cells. -/
```
```lean
def destindexCopyKvIO
    (K : RegionName) (Dest_loc : Region .nat) (Out : RegionName)
    (stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
      head_num head_dim BLOCK_DMODEL BLOCK_HEAD : Nat) :
    MetaMasked2DKernelIO₁ where
  kernel := fwd_kernel_destindex_copy_kv K Dest_loc Out
    stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
    head_num head_dim BLOCK_DMODEL BLOCK_HEAD
  mbuf1 := Dest_loc
  mbuf2 := Dest_loc
  inp := K
  out := Out
  B := BLOCK_HEAD * BLOCK_DMODEL
  mwin1 := fun pid₀ _ => pid₀
  mwin2 := fun pid₀ _ => pid₀
  read := fun pid₀ _ _ _ j =>
    pid₀ * stride_k_bs + stride_k_h * (j.val / BLOCK_DMODEL) +
      stride_k_d * (j.val % BLOCK_DMODEL)
  write := fun _ _ m₁ _ j =>
    m₁ * stride_o_bs + stride_o_h * (j.val / BLOCK_DMODEL) +
      stride_o_d * (j.val % BLOCK_DMODEL)
  mask := fun _ _ _ _ j =>
    j.val / BLOCK_DMODEL < head_num ∧ j.val % BLOCK_DMODEL < head_dim
```
</details>

<details><summary><code>fwd_kernel_destindex_copy_kv</code></summary>

```
/-- Faithful transcription of `destindex_copy_kv1.py`'s
`_fwd_kernel_destindex_copy_kv`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_DMODEL: tl.constexpr` / `BLOCK_HEAD: tl.constexpr` -> Lean
  `Nat` parameters. -/
```
```lean
def fwd_kernel_destindex_copy_kv
    (K : RegionName) (Dest_loc : Region .nat) (Out : RegionName)
    (stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
      head_num head_dim BLOCK_DMODEL BLOCK_HEAD : Nat) :
    ComputeKernel := triton {
  cur_index = tl.program_id(0)
  offs_h = tl.arange(0, $(BLOCK_HEAD))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  dest_index = tl.load(Dest_loc + cur_index)
  k_ptrs = K + cur_index * $(stride_k_bs) +
    $(stride_k_h) * offs_h[:, None] + $(stride_k_d) * offs_d[None, :]
  o_ptrs = Out + dest_index * $(stride_o_bs) +
    $(stride_o_h) * offs_h[:, None] + $(stride_o_d) * offs_d[None, :]
  k = tl.load(k_ptrs,
    mask=(offs_h[:, None] < $(head_num)) & (offs_d[None, :] < $(head_dim)),
    other=0.0)
  tl.store(o_ptrs, k,
    mask=(offs_h[:, None] < $(head_num)) & (offs_d[None, :] < $(head_dim)))
}
```
</details>
