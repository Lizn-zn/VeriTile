# Spec sheet — `bench/tritonbench_g/kcache_copy_triton/KcacheCopyTriton.lean`

**Python source:** `bench/tritonbench_g/kcache_copy_triton/kcache_copy_triton.py`

## Public theorem: `copy_to_kcache_seqlen_n1_correctness`

<details><summary>docstring</summary>

```
/-- **The headline**: the `n_tokens = 1` decode path of
`_copy_to_kcache_seqlen_n_kernel` implements the pure paged-cache copy on its
chained-metadata IO signature — for every disjoint flat placement of the
buffers, every program `(cur_token_idx, cur_kv_head_idx)` and split partition
`SPLIT_X` whose declared cells/lanes are in bounds, and every launch state
pinning the **raw** sequence length `m₁` at `seq_lengths[pid₀]`, the **raw**
block id `m₂` at the chained block-table cell
`BLOCK_TABLES[pid₀·bts + ((m₁−1)/block_size)·btb]`, and the K tile `xs`, the
translated pointer kernel terminates and writes

* `KCache[m₂·kcb + pid₁·kch + SPLIT_X·kcsplit_x + ((m₁−1)%block_size)·kcs + j]
  = xs j` — lane `j` of the K tile, verbatim, at the `m₂`-selected cache
  window (the value legs' per-context `WriteInj` no-aliasing antecedents are
  supplied by the skin);

the second data channel is off (`False` gates: its value leg is vacuous), and
every other memory cell is unchanged. Proof:
`ChainMetaMasked2DKernelIO₂ₓ₂.Implements.intro` assembles the region-model
chained triple with the flat-memory bridge side conditions. -/
```
</details>

**Statement:**
```lean
specification copy_to_kcache_seqlen_n1_correctness
    (K KCache BLOCK_TABLES seq_lengths : RegionName)
    (SPLIT_X stride_kt stride_kh stride_kd stride_kcb stride_kch
      stride_kcsplit_x stride_kcs stride_bts stride_btb block_size
      KCACHE_X : Nat) :
    kcacheCopyN1IO K KCache BLOCK_TABLES seq_lengths SPLIT_X stride_kt
        stride_kh stride_kd stride_kcb stride_kch stride_kcsplit_x stride_kcs
        stride_bts stride_btb block_size KCACHE_X ⊨
      fun _ _ _ _ xs _ => (xs, xs)
```

**Closed-form spec defs (transitive):** `kcacheCopyN1IO`, `copy_to_kcache_seqlen_n1_surface`

<details><summary><code>kcacheCopyN1IO</code></summary>

```
/-- `copy_to_kcache_seqlen_n1_surface`'s chained-metadata **IO signature** —
the whole kernel-specific audit surface of the `⊨` headline:

* `mbuf1` — the `seq_lengths` slot: program `(cur_token_idx, cur_kv_head_idx)`
  loads the **raw** sequence length `m₁ = seq_lengths[pid₀]` (`mwin1`; the
  decode path's `− 1` lives in the windows, Nat-truncated exactly as in the
  kernel);
* `mbuf2` — the `BLOCK_TABLES` slot, **chained**: its cell
  `pid₀·bts + ((m₁−1)/block_size)·btb` (`mwin2`) eats the first slot's loaded
  value, pinning the raw block id `m₂`;
* `in1 → out1` — the K tile at
  `pid₀·kt + pid₁·kh + (SPLIT_X·KCACHE_X + j)·kd` (`read1`), copied to the
  `m₂`-selected cache window
  `m₂·kcb + pid₁·kch + SPLIT_X·kcsplit_x + ((m₁−1)%block_size)·kcs + j`
  (`write1`), both unmasked (`mask1`/`writeMask1 := True` — the kernel's load
  and store carry no mask);
* `in2 → out2` — the family's second data channel, instantiated **off** for
  this K-only sibling: duplicate-region wiring `in2 := K`, `out2 := KCache`
  with `mask2`/`writeMask2 := False`, so its legs are vacuous.

The windows and masks are declared, not parsed from the kernel; the headline
**proves** the kernel's actual addressing and chaining match them. -/
```
```lean
def kcacheCopyN1IO (K KCache BLOCK_TABLES seq_lengths : RegionName)
    (SPLIT_X stride_kt stride_kh stride_kd stride_kcb stride_kch
      stride_kcsplit_x stride_kcs stride_bts stride_btb block_size
      KCACHE_X : Nat) : ChainMetaMasked2DKernelIO₂ₓ₂ where
  kernel := copy_to_kcache_seqlen_n1_surface K KCache BLOCK_TABLES seq_lengths
    SPLIT_X stride_kt stride_kh stride_kd stride_kcb stride_kch
    stride_kcsplit_x stride_kcs 0 stride_bts stride_btb block_size 1 0 KCACHE_X
  mbuf1 := seq_lengths
  mbuf2 := BLOCK_TABLES
  in1 := K
  in2 := K
  out1 := KCache
  out2 := KCache
  B := KCACHE_X
  mwin1 := fun pid₀ _ => pid₀
  mwin2 := fun pid₀ _ m₁ =>
    pid₀ * stride_bts + ((m₁ - 1) / block_size) * stride_btb
  read1 := fun pid₀ pid₁ _ _ j =>
    pid₀ * stride_kt + pid₁ * stride_kh +
      (SPLIT_X * KCACHE_X + j.val) * stride_kd
  read2 := fun pid₀ pid₁ _ _ j =>
    pid₀ * stride_kt + pid₁ * stride_kh +
      (SPLIT_X * KCACHE_X + j.val) * stride_kd
  mask1 := fun _ _ _ _ _ => True
  mask2 := fun _ _ _ _ _ => False
  write1 := fun _ pid₁ m₁ m₂ j =>
    m₂ * stride_kcb + pid₁ * stride_kch + SPLIT_X * stride_kcsplit_x +
      ((m₁ - 1) % block_size) * stride_kcs + j.val
  write2 := fun _ pid₁ m₁ m₂ j =>
    m₂ * stride_kcb + pid₁ * stride_kch + SPLIT_X * stride_kcsplit_x +
      ((m₁ - 1) % block_size) * stride_kcs + j.val
  writeMask1 := fun _ _ _ _ _ => True
  writeMask2 := fun _ _ _ _ _ => False
```
</details>

<details><summary><code>copy_to_kcache_seqlen_n1_surface</code></summary>

```
/-- Surface transcription of `kcache_copy_triton.py`'s
`_copy_to_kcache_seqlen_n_kernel` for the `n_tokens = 1` decode path.

For `n_tokens = 1`, Python treats `cur_token_idx` as the sequence id and uses
`seq_lengths[cur_seq_idx] - 1` as the position being copied. This surface keeps
that block-table lookup, offset-within-block computation, split-x K load, and
K-cache store. The `n_tokens > 1` path needs negative `cur_token_shift`
arithmetic before the copy and remains outside the current Nat-only pointer
surface. Python's `split_x_idx = tl.program_id(2)` axis is pinned as the host
parameter `SPLIT_X` (the `kv_cache_copy` sibling's convention): the chained
two-pid `⊨` family below has no third program id, and the universally
quantified `SPLIT_X` covers every program of the third grid dimension.
Python's unused `stride_kcx` and `HEAD_DIM` arguments are retained as ignored
parameters; `n_tokens` is fixed to the documented decode value `1` in the
proofs. -/
```
```lean
def copy_to_kcache_seqlen_n1_surface
    (K KCache : RegionName) (BLOCK_TABLES seq_lengths : Region .nat)
    (SPLIT_X stride_kt stride_kh stride_kd stride_kcb stride_kch
      stride_kcsplit_x stride_kcs _stride_kcx stride_bts stride_btb block_size
      _n_tokens _HEAD_DIM KCACHE_X : Nat) :
    ComputeKernel := triton {
  cur_token_idx = tl.program_id(0)
  cur_seq_idx = cur_token_idx
  cur_kv_head_idx = tl.program_id(1)
  past_kv_seq_len = tl.load(seq_lengths + cur_seq_idx) - $(1)
  last_bt_block_idx = past_kv_seq_len // $(block_size)
  block_id = tl.load(BLOCK_TABLES + cur_seq_idx * $(stride_bts) +
    last_bt_block_idx * $(stride_btb))
  offset_last_block = past_kv_seq_len % $(block_size)
  offsets_dmodel = $(SPLIT_X) * $(KCACHE_X) + tl.arange(0, $(KCACHE_X))
  k = tl.load(K + cur_token_idx * $(stride_kt) +
    cur_kv_head_idx * $(stride_kh) + offsets_dmodel * $(stride_kd))
  tl.store(KCache + block_id * $(stride_kcb) +
    cur_kv_head_idx * $(stride_kch) +
    $(SPLIT_X) * $(stride_kcsplit_x) +
    offset_last_block * $(stride_kcs) + tl.arange(0, $(KCACHE_X)), k)
}
```
</details>

## Also present (pinned special-case summaries)
- `copy_to_kcache_seqlen_n1_surface_compute_correct`
- `copy_to_kcache_seqlen_n1_old_layout_block_compute_correct`
- `copy_to_kcache_seqlen_n1_new_layout_xblock_compute_correct`
- `copy_to_kcache_split_x_block_compute_correct`
