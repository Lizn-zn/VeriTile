# Spec sheet — `bench/tritonbench_g/kv_cache_copy/KvCacheCopy.lean`

**Python source:** `bench/tritonbench_g/kv_cache_copy/kv_cache_copy.py`

## Public theorem: `kv_cache_copy_correctness`

<details><summary>docstring</summary>

```
/-- **The headline**: the per-split KV-cache copy surface implements the pure
masked copy `(xs, ys)` on its chained-metadata IO signature — for every
disjoint flat placement of the six buffers, every program
`(cur_seq_idx, cur_kv_head_idx)` whose declared cells/lanes are in bounds,
and every launch state pinning the **raw** loaded scalars — `m₁` at the
`context_lengths` slot cell and `m₂` at the **`m₁`-dependent** block-table
cell (the chain) — and the active K/V windows to `xs`/`ys`, the translated
pointer kernel terminates, every active (`SPLIT_X·KCACHE_X + j < HEAD_DIM`)
K-cache window lane holds `xs j` and V-cache window lane `ys j` — each
readback gated by its per-pinned-context `WriteInj` no-aliasing side
condition on the block-id-dependent cache window — and every other memory
cell is unchanged. All `- 1` / `/` / `%` decode arithmetic is Nat-truncated,
exactly as the DSL executes it, and lives in the declared windows, not in the
pins. Side condition: `KCache ≠ VCache` (the V store must not clobber the K
output). Instantiating
`SPLIT_X` at every `split_x < HEAD_DIM / KCACHE_X` covers the full static
loop; the legacy cache layout is `SPLIT_X = 0`, `KCACHE_X = HEAD_DIM`,
`stride_kcsplit_x = 0`. Proof:
`ChainMetaMasked2DKernelIO₂ₓ₂.Implements.intro` assembles the region-model
chained triple with the flat-memory bridge side conditions. -/
```
</details>

**Statement:**
```lean
specification kv_cache_copy_correctness
    (K V KCache VCache BLOCK_TABLES context_lengths : RegionName)
    (SPLIT_X stride_kt stride_kh stride_kd
      stride_vt stride_vh stride_vd
      stride_kcb stride_kch stride_kcsplit_x stride_kcs
      stride_vcb stride_vch stride_vcs stride_vcd
      stride_bts stride_btb block_size HEAD_DIM KCACHE_X : Nat)
    (hKV : KCache ≠ VCache) :
    kvCacheCopyIO K V KCache VCache BLOCK_TABLES context_lengths
        SPLIT_X stride_kt stride_kh stride_kd stride_vt stride_vh stride_vd
      stride_kcb stride_kch stride_kcsplit_x stride_kcs stride_vcb stride_vch
      stride_vcs stride_vcd stride_bts stride_btb block_size HEAD_DIM KCACHE_X
      ⊨ fun _ _ _ _ xs ys => (xs, ys)
```

**Assumptions / layout contracts:**
- `hKV : KCache ≠ VCache`

**Closed-form spec defs (transitive):** `kvCacheCopyIO`, `copy_to_kvcache_seqlen1_xblock`

<details><summary><code>kvCacheCopyIO</code></summary>

```
/-- `copy_to_kvcache_seqlen1_xblock`'s chained-metadata **IO signature** — the
whole kernel-specific audit surface of the `⊨` headline
(`ChainMetaMasked2DKernelIO₂ₓ₂`, the chained-metadata genre this kernel was
designed from):

* `mbuf1 = context_lengths` — slot 1: program `(cur_seq_idx, cur_kv_head_idx)
  = (pid₀, pid₁)` reads cell `pid₀` (`mwin1`), yielding the **raw** context
  length `m₁`;
* `mbuf2 = BLOCK_TABLES` — slot 2, **chained**: its cell
  `pid₀·stride_bts + ((m₁ − 1) / block_size)·stride_btb` (`mwin2`) eats
  slot 1's loaded value (`- 1` is the DSL's Nat-truncated subtraction),
  yielding the raw block id `m₂ = block_id`;
* `in1 = K → out1 = KCache`, `in2 = V → out2 = VCache` — the two data
  channels, `B = KCACHE_X` lanes at x-partition `SPLIT_X`;
* `read1`/`read2` — lane `j` reads
  `pid₀·stride_*t + pid₁·stride_*h + (SPLIT_X·KCACHE_X + j)·stride_*d`;
* `write1` — the K-cache cell `m₂·stride_kcb + pid₁·stride_kch +
  SPLIT_X·stride_kcsplit_x + ((m₁ − 1) % block_size)·stride_kcs + j`;
* `write2` — the V-cache cell `m₂·stride_vcb + pid₁·stride_vch +
  ((m₁ − 1) % block_size)·stride_vcs + (SPLIT_X·KCACHE_X + j)·stride_vcd`;
* `mask1 = mask2` — the active lanes `SPLIT_X·KCACHE_X + j < HEAD_DIM`
  (`writeMask`s default to the masks).

The slot cells, windows, and masks are declared, not parsed from the kernel;
the headline **proves** the kernel's actual chained slot loads, addressing,
and masking match them. Buffer sizes are not signature content: the headline
quantifies over every allocation whose extents cover the declared cells. -/
```
```lean
def kvCacheCopyIO
    (K V KCache VCache BLOCK_TABLES context_lengths : RegionName)
    (SPLIT_X stride_kt stride_kh stride_kd
      stride_vt stride_vh stride_vd
      stride_kcb stride_kch stride_kcsplit_x stride_kcs
      stride_vcb stride_vch stride_vcs stride_vcd
      stride_bts stride_btb block_size HEAD_DIM KCACHE_X : Nat) : ChainMetaMasked2DKernelIO₂ₓ₂ where
  kernel := copy_to_kvcache_seqlen1_xblock K V KCache VCache BLOCK_TABLES
    context_lengths SPLIT_X stride_kt stride_kh stride_kd stride_vt stride_vh stride_vd
      stride_kcb stride_kch stride_kcsplit_x stride_kcs stride_vcb stride_vch
      stride_vcs stride_vcd stride_bts stride_btb block_size HEAD_DIM KCACHE_X
  mbuf1 := context_lengths
  mbuf2 := BLOCK_TABLES
  in1 := K
  in2 := V
  out1 := KCache
  out2 := VCache
  B := KCACHE_X
  mwin1 := fun pid₀ _ => pid₀
  mwin2 := fun pid₀ _ m₁ =>
    pid₀ * stride_bts + ((m₁ - 1) / block_size) * stride_btb
  read1 := fun pid₀ pid₁ _ _ j =>
    pid₀ * stride_kt + pid₁ * stride_kh +
      (SPLIT_X * KCACHE_X + j.val) * stride_kd
  read2 := fun pid₀ pid₁ _ _ j =>
    pid₀ * stride_vt + pid₁ * stride_vh +
      (SPLIT_X * KCACHE_X + j.val) * stride_vd
  mask1 := fun _ _ _ _ j => SPLIT_X * KCACHE_X + j.val < HEAD_DIM
  mask2 := fun _ _ _ _ j => SPLIT_X * KCACHE_X + j.val < HEAD_DIM
  write1 := fun _ pid₁ m₁ m₂ j =>
    m₂ * stride_kcb + pid₁ * stride_kch + SPLIT_X * stride_kcsplit_x +
      ((m₁ - 1) % block_size) * stride_kcs + j.val
  write2 := fun _ pid₁ m₁ m₂ j =>
    m₂ * stride_vcb + pid₁ * stride_vch +
      ((m₁ - 1) % block_size) * stride_vcs +
      (SPLIT_X * KCACHE_X + j.val) * stride_vcd
```
</details>

<details><summary><code>copy_to_kvcache_seqlen1_xblock</code></summary>

```
/-- **The headline surface**: the decode prologue plus ONE `split_x` iteration
of `_copy_to_kvcache_seqlen1_kernel`'s static loop, with BOTH cache stores in
a single program — `context_lengths[cur_seq_idx] - 1`, block division/modulo,
the chained block-table lookup, the masked K load / K-cache store and the
masked V load / V-cache store for the `SPLIT_X` x-partition. Instantiating
`SPLIT_X` at every `split_x < HEAD_DIM / KCACHE_X` covers the whole loop; the
legacy cache layout is the instance `SPLIT_X = 0`, `KCACHE_X = HEAD_DIM`,
`stride_kcsplit_x = 0`. -/
```
```lean
def copy_to_kvcache_seqlen1_xblock
    (K V KCache VCache : RegionName)
    (BLOCK_TABLES context_lengths : Region .nat)
    (SPLIT_X stride_kt stride_kh stride_kd
      stride_vt stride_vh stride_vd
      stride_kcb stride_kch stride_kcsplit_x stride_kcs
      stride_vcb stride_vch stride_vcs stride_vcd
      stride_bts stride_btb block_size HEAD_DIM KCACHE_X : Nat) :
    ComputeKernel := triton {
  cur_seq_idx = tl.program_id(0)
  cur_kv_head_idx = tl.program_id(1)
  past_kv_seq_len = tl.load(context_lengths + cur_seq_idx) - $(1)
  last_bt_block_idx = past_kv_seq_len // $(block_size)
  block_id = tl.load(BLOCK_TABLES + cur_seq_idx * $(stride_bts) +
    last_bt_block_idx * $(stride_btb))
  offsets_in_last_block = past_kv_seq_len % $(block_size)
  range_x = tl.arange(0, $(KCACHE_X))
  offsets_dmodel_x_partition = $(SPLIT_X) * $(KCACHE_X) + range_x
  k = tl.load(K + cur_seq_idx * $(stride_kt) +
      cur_kv_head_idx * $(stride_kh) +
      offsets_dmodel_x_partition * $(stride_kd),
    mask=offsets_dmodel_x_partition < $(HEAD_DIM), other=0.0)
  v = tl.load(V + cur_seq_idx * $(stride_vt) +
      cur_kv_head_idx * $(stride_vh) +
      offsets_dmodel_x_partition * $(stride_vd),
    mask=offsets_dmodel_x_partition < $(HEAD_DIM), other=0.0)
  tl.store(KCache + block_id * $(stride_kcb) +
      cur_kv_head_idx * $(stride_kch) +
      $(SPLIT_X) * $(stride_kcsplit_x) +
      offsets_in_last_block * $(stride_kcs) + range_x,
    k, mask=offsets_dmodel_x_partition < $(HEAD_DIM))
  tl.store(VCache + block_id * $(stride_vcb) +
      cur_kv_head_idx * $(stride_vch) +
      offsets_in_last_block * $(stride_vcs) +
      offsets_dmodel_x_partition * $(stride_vcd),
    v, mask=offsets_dmodel_x_partition < $(HEAD_DIM))
}
```
</details>
