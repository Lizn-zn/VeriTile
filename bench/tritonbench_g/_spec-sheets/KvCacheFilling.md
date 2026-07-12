# Spec sheet — `bench/tritonbench_g/kv_cache_filling/KvCacheFilling.lean`

**Python source:** `bench/tritonbench_g/kv_cache_filling/kv_cache_filling.py`

## Public theorem: `fill_kv_cache_output_summary_general`

<details><summary>docstring</summary>

```
/-- **Dimension-general** non-quantized cache-fill summary.

Every shape / stride is a universally quantified `Nat`, with the per-program
K-cache and V-cache offset-injectivity facts taken as hypotheses. The surface
conjunct lowers the faithful `_fill_kv_cache_kernel` body, and the output
conjuncts expose the masked K/V cache tile writebacks at the selected
source/cache block positions. The Python test shapes (`num_heads = 4`,
`head_dim = head_dim_v = 16`, `BLOCK = 8`) are just an instance of this
theorem. -/
```
</details>

**Statement:**
```lean
specification fill_kv_cache_output_summary_general
    (KStates VStates KCaches VCaches QStartLoc QSeqLens KVSeqLens
      BlockOffsets : RegionName)
    (SIDX BIDX KV_BLOCK_IDX
      num_heads head_dim head_dim_v stride_kss stride_ksh stride_ksd
      stride_vss stride_vsh stride_vsd stride_kcn stride_kcb stride_kch
      stride_kcd stride_vcn stride_vcb stride_vch stride_vcd stride_boff
      BLOCK BLOCK_D BLOCK_DV BLOCK_H : Nat)
    (s : BlockState)
    (hKInj : Function.Injective
      (fun idx : TileIndex [BLOCK_H, BLOCK_D] =>
        kCacheOffset s BlockOffsets BIDX KV_BLOCK_IDX stride_kcn stride_kcb
          stride_kch stride_kcd stride_boff idx))
    (hVInj : Function.Injective
      (fun idx : TileIndex [BLOCK_H, BLOCK_DV] =>
        vCacheOffset s BlockOffsets BIDX KV_BLOCK_IDX stride_vcn stride_vcb
          stride_vch stride_vcd stride_boff idx)) :
    (∃ alg, (fill_kv_cache_kernel_surface KStates VStates KCaches VCaches
      QStartLoc QSeqLens KVSeqLens BlockOffsets num_heads head_dim head_dim_v
      stride_kss stride_ksh stride_ksd stride_vss stride_vsh stride_vsd
      stride_kcn stride_kcb stride_kch stride_kcd stride_vcn stride_vcb
      stride_vch stride_vcd stride_boff BLOCK BLOCK_D BLOCK_DV
      BLOCK_H).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := fill_k_cache_tile KStates KCaches BlockOffsets
        SIDX BIDX KV_BLOCK_IDX stride_kss stride_ksh stride_ksd
        stride_kcn stride_kcb stride_kch stride_kcd stride_boff
        num_heads head_dim BLOCK_H BLOCK_D)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s num_heads head_dim BLOCK_H BLOCK_D)
        (fun idx => (KCaches,
          kCacheOffset s BlockOffsets BIDX KV_BLOCK_IDX stride_kcn stride_kcb
            stride_kch stride_kcd stride_boff idx)))
      (expected := fun idx =>
        s.readMem KStates
          (kSourceOffset s SIDX stride_kss stride_ksh stride_ksd idx))) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := fill_v_cache_tile VStates VCaches BlockOffsets
        SIDX BIDX KV_BLOCK_IDX stride_vss stride_vsh stride_vsd
        stride_vcn stride_vcb stride_vch stride_vcd stride_boff
        num_heads head_dim_v BLOCK_H BLOCK_DV)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s num_heads head_dim_v BLOCK_H BLOCK_DV)
        (fun idx => (VCaches,
          vCacheOffset s BlockOffsets BIDX KV_BLOCK_IDX stride_vcn stride_vcb
            stride_vch stride_vcd stride_boff idx)))
      (expected := fun idx =>
        s.readMem VStates
          (vSourceOffset s SIDX stride_vss stride_vsh stride_vsd idx)))
```

**Assumptions / layout contracts:**
- `hKInj : Function.Injective
      (fun idx : TileIndex [BLOCK_H, BLOCK_D] =>
        kCacheOffset s BlockOffsets BIDX KV_BLOCK_IDX stride_kcn stride_kcb
          stride_kch stride_kcd stride_boff idx)`
- `hVInj : Function.Injective
      (fun idx : TileIndex [BLOCK_H, BLOCK_DV] =>
        vCacheOffset s BlockOffsets BIDX KV_BLOCK_IDX stride_vcn stride_vcb
          stride_vch stride_vcd stride_boff idx)`

**Closed-form spec defs (transitive):** `kCacheOffset`, `vCacheOffset`, `fill_kv_cache_kernel_surface`, `fill_k_cache_tile`, `active`, `kSourceOffset`, `fill_v_cache_tile`, `vSourceOffset`, `blockOff`, `headIndex`, `dimIndex`

<details><summary><code>kCacheOffset</code></summary>

```lean
def kCacheOffset
    (s : BlockState) (BlockOffsets : RegionName)
    (BIDX KV_BLOCK_IDX stride_kcn stride_kcb stride_kch stride_kcd stride_boff : Nat)
    (idx : TileIndex [BLOCK_H, BLOCK_D]) : Nat :=
  blockOff s BlockOffsets KV_BLOCK_IDX stride_boff * stride_kcn +
    BIDX * stride_kcb + headIndex s idx.1 * stride_kch +
    dimIndex s idx.2.1 * stride_kcd
```
</details>

<details><summary><code>vCacheOffset</code></summary>

```lean
def vCacheOffset
    (s : BlockState) (BlockOffsets : RegionName)
    (BIDX KV_BLOCK_IDX stride_vcn stride_vcb stride_vch stride_vcd stride_boff : Nat)
    (idx : TileIndex [BLOCK_H, BLOCK_DV]) : Nat :=
  blockOff s BlockOffsets KV_BLOCK_IDX stride_boff * stride_vcn +
    BIDX * stride_vcb + headIndex s idx.1 * stride_vch +
    dimIndex s idx.2.1 * stride_vcd
```
</details>

<details><summary><code>fill_kv_cache_kernel_surface</code></summary>

```
/-- Lean port of `kv_cache_filling.py`'s `_fill_kv_cache_kernel`.

This covers the Python wrapper's `quant_policy = 0` path. Correctness below is
still proved on smaller K/V tile lemmas, while this definition records the
complete non-quant kernel body used by the unquantized path. -/
```
```lean
def fill_kv_cache_kernel_surface
    (KStates VStates KCaches VCaches : RegionName)
    (QStartLoc QSeqLens KVSeqLens BlockOffsets : Region .nat)
    (num_heads head_dim head_dim_v stride_kss stride_ksh stride_ksd
      stride_vss stride_vsh stride_vsd stride_kcn stride_kcb stride_kch
      stride_kcd stride_vcn stride_vcb stride_vch stride_vcd stride_boff
      BLOCK BLOCK_D BLOCK_DV BLOCK_H : Nat) :
    ComputeKernel := triton {
  batch_id = tl.program_id(0)
  block_id = tl.program_id(1)
  h_off = tl.arange(0, $(BLOCK_H))
  d_off = tl.arange(0, $(BLOCK_D))
  q_startloc = tl.load(QStartLoc + batch_id)
  q_seqlen = tl.load(QSeqLens + batch_id)
  kv_seqlen = tl.load(KVSeqLens + batch_id)
  history_seqlen = kv_seqlen - q_seqlen
  block0_first_tokenloc = history_seqlen % $(BLOCK)
  state_token_offset = tl.maximum(block_id * $(BLOCK) - block0_first_tokenloc, $(0))
  kv_block_id = _div_up(history_seqlen + $(1), $(BLOCK)) - $(1) + block_id
  kv_block_id = min(kv_block_id, $((stride_boff - 1 : Nat)))
  block_off = tl.load(BlockOffsets +
    batch_id * $(stride_boff) + kv_block_id)
  cur_startloc = q_startloc + state_token_offset
  ks_ptr = KStates + cur_startloc * $(stride_kss)
  vs_ptr = VStates + cur_startloc * $(stride_vss)
  kc_ptr = KCaches + block_off * $(stride_kcn)
  vc_ptr = VCaches + block_off * $(stride_vcn)
  c_first_tokenloc = block0_first_tokenloc
  if block_id != $(0) {
    c_first_tokenloc *= $(0)
  }
  c_last_tokenloc = tl.minimum($(BLOCK),
    q_seqlen + block0_first_tokenloc - block_id * $(BLOCK))
  for bidx in range(c_first_tokenloc, c_last_tokenloc) {
    sidx = bidx - c_first_tokenloc
    mask = (h_off[:, None] < $(num_heads)) & (d_off[None, :] < $(head_dim))
    k = tl.load(ks_ptr + sidx * $(stride_kss) + h_off[:, None] * $(stride_ksh) +
        d_off[None, :] * $(stride_ksd), mask=mask)
    tl.store(kc_ptr + bidx * $(stride_kcb) + h_off[:, None] * $(stride_kch) +
        d_off[None, :] * $(stride_kcd), k, mask=mask)
    if $((BLOCK_DV > 0 : Bool)) {
      dv_off = tl.arange(0, $(BLOCK_DV))
      maskv = (h_off[:, None] < $(num_heads)) & (dv_off[None, :] < $(head_dim_v))
      v = tl.load(vs_ptr + sidx * $(stride_vss) +
          h_off[:, None] * $(stride_vsh) + dv_off[None, :] * $(stride_vsd),
        mask=maskv)
      tl.store(vc_ptr + bidx * $(stride_vcb) + h_off[:, None] * $(stride_vch) +
          dv_off[None, :] * $(stride_vcd), v, mask=maskv)
    }
  }
}
```
</details>

<details><summary><code>fill_k_cache_tile</code></summary>

```
/-- Surface transcription and proof-oriented K-cache fill slice of `kv_cache_filling.py`'s
`_fill_kv_cache_kernel`.

The full kernel computes sequence/block positions and loops over cache block
slots. This slice starts after that arithmetic has selected `SIDX`, `BIDX`, and
`KV_BLOCK_IDX`: load a block offset from `BlockOffsets`, load a
`BLOCK_H × BLOCK_D` K tile from `KStates`, and store it into `KCaches` under the
original `num_heads/head_dim` mask.

This tile is the `quant_policy = 0` value path. The quantized value and
metadata writebacks, plus the `_quant_int8`/`_quant_int4` helper surfaces for
rounding, scale/zero metadata, and int4 packing, are represented separately
below so each Python-tested branch has an explicit DSL artifact. -/
```
```lean
def fill_k_cache_tile
    (KStates KCaches : RegionName) (BlockOffsets : Region .nat)
    (SIDX BIDX KV_BLOCK_IDX
      stride_kss stride_ksh stride_ksd
      stride_kcn stride_kcb stride_kch stride_kcd
      stride_boff num_heads head_dim BLOCK_H BLOCK_D : Nat) :
    ComputeKernel := triton {
  batch_id = tl.program_id(0)
  h_off = tl.arange(0, $(BLOCK_H))
  d_off = tl.arange(0, $(BLOCK_D))
  block_off = tl.load(BlockOffsets + batch_id * $(stride_boff) + $(KV_BLOCK_IDX))
  mask = (h_off[:, None] < $(num_heads)) & (d_off[None, :] < $(head_dim))
  k = tl.load(KStates + $(SIDX) * $(stride_kss) +
      h_off[:, None] * $(stride_ksh) + d_off[None, :] * $(stride_ksd),
    mask=mask, other=0.0)
  tl.store(KCaches + block_off * $(stride_kcn) + $(BIDX) * $(stride_kcb) +
      h_off[:, None] * $(stride_kch) + d_off[None, :] * $(stride_kcd),
    k, mask=mask)
}
```
</details>

<details><summary><code>active</code></summary>

```lean
def active
    (s : BlockState) (num_heads head_dim BLOCK_H BLOCK_D : Nat)
    (idx : TileIndex [BLOCK_H, BLOCK_D]) : Prop :=
  headIndex s idx.1 < num_heads ∧ dimIndex s idx.2.1 < head_dim
```
</details>

<details><summary><code>kSourceOffset</code></summary>

```lean
def kSourceOffset
    (s : BlockState) (SIDX stride_kss stride_ksh stride_ksd : Nat)
    (idx : TileIndex [BLOCK_H, BLOCK_D]) : Nat :=
  SIDX * stride_kss + headIndex s idx.1 * stride_ksh +
    dimIndex s idx.2.1 * stride_ksd
```
</details>

<details><summary><code>fill_v_cache_tile</code></summary>

```
/-- Surface transcription of the V-cache fill side of `kv_cache_filling.py`'s
`_fill_kv_cache_kernel`.

This is the `BLOCK_DV > 0` branch paired with `fill_k_cache_tile`: after the
sequence/block arithmetic has selected `SIDX`, `BIDX`, and `KV_BLOCK_IDX`, it
loads one `BLOCK_H × BLOCK_DV` tile from `VStates` and stores it into
`VCaches` under the original `num_heads/head_dim_v` mask. -/
```
```lean
def fill_v_cache_tile
    (VStates VCaches : RegionName) (BlockOffsets : Region .nat)
    (SIDX BIDX KV_BLOCK_IDX
      stride_vss stride_vsh stride_vsd
      stride_vcn stride_vcb stride_vch stride_vcd
      stride_boff num_heads head_dim_v BLOCK_H BLOCK_DV : Nat) :
    ComputeKernel := triton {
  batch_id = tl.program_id(0)
  h_off = tl.arange(0, $(BLOCK_H))
  dv_off = tl.arange(0, $(BLOCK_DV))
  block_off = tl.load(BlockOffsets + batch_id * $(stride_boff) + $(KV_BLOCK_IDX))
  maskv = (h_off[:, None] < $(num_heads)) & (dv_off[None, :] < $(head_dim_v))
  v = tl.load(VStates + $(SIDX) * $(stride_vss) +
      h_off[:, None] * $(stride_vsh) + dv_off[None, :] * $(stride_vsd),
    mask=maskv, other=0.0)
  tl.store(VCaches + block_off * $(stride_vcn) + $(BIDX) * $(stride_vcb) +
      h_off[:, None] * $(stride_vch) + dv_off[None, :] * $(stride_vcd),
    v, mask=maskv)
}
```
</details>

<details><summary><code>vSourceOffset</code></summary>

```
/-! ## V-cache tile fill correctness -/
```
```lean
def vSourceOffset
    (s : BlockState) (SIDX stride_vss stride_vsh stride_vsd : Nat)
    (idx : TileIndex [BLOCK_H, BLOCK_DV]) : Nat :=
  SIDX * stride_vss + headIndex s idx.1 * stride_vsh +
    dimIndex s idx.2.1 * stride_vsd
```
</details>

<details><summary><code>blockOff</code></summary>

```lean
def blockOff (s : BlockState) (BlockOffsets : RegionName)
    (KV_BLOCK_IDX stride_boff : Nat) : Nat :=
  s.readMemValue .nat BlockOffsets (s.pids 0 * stride_boff + KV_BLOCK_IDX)
```
</details>

<details><summary><code>headIndex</code></summary>

```lean
def headIndex (_s : BlockState) (i : Fin BLOCK_H) : Nat :=
  i.val
```
</details>

<details><summary><code>dimIndex</code></summary>

```lean
def dimIndex (_s : BlockState) (j : Fin BLOCK_D) : Nat :=
  j.val
```
</details>

## Also present (pinned special-case summaries)
- `fill_k_cache_tile_compute_correct`
- `fill_v_cache_tile_compute_correct`
- `fill_quant_k_cache_value_store_slice_compute_correct`
- `fill_quant_v_cache_value_store_slice_compute_correct`
- `fill_quant_meta_store_slice_compute_correct`
- `fill_quant_k_scale_store_slice_compute_correct`
- `fill_quant_k_zero_store_slice_compute_correct`
- `fill_quant_v_scale_store_slice_compute_correct`
- `fill_quant_v_zero_store_slice_compute_correct`
