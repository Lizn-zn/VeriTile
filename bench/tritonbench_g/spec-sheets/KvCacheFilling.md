# Spec sheet — `bench/tritonbench_g/kv_cache_filling/KvCacheFilling.lean`

**Python source:** `bench/tritonbench_g/kv_cache_filling/kv_cache_filling.py`

## Public theorem: `fill_kv_cache_python_test_layout_output_summary`

<details><summary>docstring</summary>

```
/-- Public Python non-quantized cache-fill summary for the checked layout.

The surface conjunct pins the faithful `_fill_kv_cache_kernel` launch for
`num_heads = 4`, `head_dim = head_dim_v = 16`, `BLOCK = 8`, and contiguous
K/V state/cache strides used by the benchmark. The output conjunct exposes the
checked K-cache and V-cache tile writebacks for the selected source/cache
block positions. -/
```
</details>

**Statement:**
```lean
theorem fill_kv_cache_python_test_layout_output_summary
    (KStates VStates KCaches VCaches QStartLoc QSeqLens KVSeqLens
      BlockOffsets : RegionName)
    (SIDX BIDX KV_BLOCK_IDX : Nat) (s : BlockState) :
    (∃ alg, (fill_kv_cache_kernel_surface KStates VStates KCaches VCaches
      QStartLoc QSeqLens KVSeqLens BlockOffsets
      4 16 16 64 16 1 64 16 1 512 64 16 1 512 64 16 1 5 4 16 16 4
      ).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := fill_k_cache_tile KStates KCaches BlockOffsets
        SIDX BIDX KV_BLOCK_IDX 64 16 1 512 64 16 1 5 4 16 4 16)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 4 16 4 16)
        (fun idx => (KCaches,
          kCacheOffset s BlockOffsets BIDX KV_BLOCK_IDX 512 64 16 1 5 idx)))
      (expected := fun idx =>
        s.readMem KStates (kSourceOffset s SIDX 64 16 1 idx))) ∧
    (ComputeCorrect.Realizes
      (kernel := fill_v_cache_tile VStates VCaches BlockOffsets
        SIDX BIDX KV_BLOCK_IDX 64 16 1 512 64 16 1 5 4 16 4 16)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 4 16 4 16)
        (fun idx => (VCaches,
          vCacheOffset s BlockOffsets BIDX KV_BLOCK_IDX 512 64 16 1 5 idx)))
      (expected := fun idx =>
        s.readMem VStates (vSourceOffset s SIDX 64 16 1 idx)))
```

**Assumptions / layout contracts:**
- `kernel : = fill_k_cache_tile KStates KCaches BlockOffsets
        SIDX BIDX KV_BLOCK_IDX 64 16 1 512 64 16 1 5 4 16 4 16`
- `initialState : = s`
- `expected : = fun idx =>
        s.readMem KStates (kSourceOffset s SIDX 64 16 1 idx)`
- `kernel : = fill_v_cache_tile VStates VCaches BlockOffsets
        SIDX BIDX KV_BLOCK_IDX 64 16 1 512 64 16 1 5 4 16 4 16`
- `initialState : = s`
- `expected : = fun idx =>
        s.readMem VStates (vSourceOffset s SIDX 64 16 1 idx)`

**Closed-form spec defs (transitive):** `fill_kv_cache_kernel_surface`, `fill_k_cache_tile`, `active`, `kCacheOffset`, `kSourceOffset`, `fill_v_cache_tile`, `vCacheOffset`, `vSourceOffset`, `headIndex`, `dimIndex`, `blockOff`

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

<details><summary><code>blockOff</code></summary>

```lean
def blockOff (s : BlockState) (BlockOffsets : RegionName)
    (KV_BLOCK_IDX stride_boff : Nat) : Nat :=
  s.readMemValue .nat BlockOffsets (s.pids 0 * stride_boff + KV_BLOCK_IDX)
```
</details>

## Public theorem: `fill_quant_int8_kv_cache_python_test_layout_summary`

<details><summary>docstring</summary>

```
/-- Python `quant_policy = 8` summary for the checked cache-fill layout.

This combines the `_quant_int8` helper surface, including min/max scale and
zero computation plus the uint8 cast, with the checked K/V cache and metadata
writeback obligations at the observed `H = 4`, `D = 16` test shape. -/
```
</details>

**Statement:**
```lean
theorem fill_quant_int8_kv_cache_python_test_layout_summary
    (KStates VStates QKPre QVPre KScalePre KZeroPre VScalePre VZeroPre KCaches
      VCaches KScalesZeros VScalesZeros BlockOffsets : RegionName)
    (SIDX BIDX KV_BLOCK_IDX : Nat) (s : BlockState) :
    (∃ alg, (quant_int8_compute_store_slice KStates QKPre KScalePre KZeroPre
      16 1 16 1 4 16 4 16).toAlgorithm? = Except.ok alg) ∧
    (∃ alg, (quant_int8_compute_store_slice VStates QVPre VScalePre VZeroPre
      16 1 16 1 4 16 4 16).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := fill_k_cache_tile QKPre KCaches BlockOffsets
        SIDX BIDX KV_BLOCK_IDX 64 16 1 512 64 16 1 5 4 16 4 16)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 4 16 4 16)
        (fun idx => (KCaches,
          kCacheOffset s BlockOffsets BIDX KV_BLOCK_IDX 512 64 16 1 5 idx)))
      (expected := fun idx =>
        s.readMem QKPre (kSourceOffset s SIDX 64 16 1 idx))) ∧
    (ComputeCorrect.Realizes
      (kernel := fill_v_cache_tile QVPre VCaches BlockOffsets
        SIDX BIDX KV_BLOCK_IDX 64 16 1 512 64 16 1 5 4 16 4 16)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 4 16 4 16)
        (fun idx => (VCaches,
          vCacheOffset s BlockOffsets BIDX KV_BLOCK_IDX 512 64 16 1 5 idx)))
      (expected := fun idx =>
        s.readMem QVPre (vSourceOffset s SIDX 64 16 1 idx))) ∧
    (ComputeCorrect.Realizes
      (kernel := fill_quant_meta_store_slice KScalePre KScalesZeros BlockOffsets
        BIDX KV_BLOCK_IDX 0 64 8 2 1 5 4 4)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 4 => metaActive s 4 4 i)
        (fun i : Fin 4 => (KScalesZeros,
          metaOffset s BlockOffsets BIDX KV_BLOCK_IDX 0 64 8 2 1 5 i)))
      (expected := fun i : Fin 4 => metaStoreSpec s KScalePre i)) ∧
    (ComputeCorrect.Realizes
      (kernel := fill_quant_meta_store_slice KZeroPre KScalesZeros BlockOffsets
        BIDX KV_BLOCK_IDX 1 64 8 2 1 5 4 4)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 4 => metaActive s 4 4 i)
        (fun i : Fin 4 => (KScalesZeros,
          metaOffset s BlockOffsets BIDX KV_BLOCK_IDX 1 64 8 2 1 5 i)))
      (expected := fun i : Fin 4 => metaStoreSpec s KZeroPre i)) ∧
    (ComputeCorrect.Realizes
      (kernel := fill_quant_meta_store_slice VScalePre VScalesZeros BlockOffsets
        BIDX KV_BLOCK_IDX 0 64 8 2 1 5 4 4)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 4 => metaActive s 4 4 i)
        (fun i : Fin 4 => (VScalesZeros,
          metaOffset s BlockOffsets BIDX KV_BLOCK_IDX 0 64 8 2 1 5 i)))
      (expected := fun i : Fin 4 => metaStoreSpec s VScalePre i)) ∧
    (ComputeCorrect.Realizes
      (kernel := fill_quant_meta_store_slice VZeroPre VScalesZeros BlockOffsets
        BIDX KV_BLOCK_IDX 1 64 8 2 1 5 4 4)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 4 => metaActive s 4 4 i)
        (fun i : Fin 4 => (VScalesZeros,
          metaOffset s BlockOffsets BIDX KV_BLOCK_IDX 1 64 8 2 1 5 i)))
      (expected := fun i : Fin 4 => metaStoreSpec s VZeroPre i))
```

**Assumptions / layout contracts:**
- `kernel : = fill_k_cache_tile QKPre KCaches BlockOffsets
        SIDX BIDX KV_BLOCK_IDX 64 16 1 512 64 16 1 5 4 16 4 16`
- `initialState : = s`
- `expected : = fun idx =>
        s.readMem QKPre (kSourceOffset s SIDX 64 16 1 idx)`
- `kernel : = fill_v_cache_tile QVPre VCaches BlockOffsets
        SIDX BIDX KV_BLOCK_IDX 64 16 1 512 64 16 1 5 4 16 4 16`
- `initialState : = s`
- `expected : = fun idx =>
        s.readMem QVPre (vSourceOffset s SIDX 64 16 1 idx)`
- `kernel : = fill_quant_meta_store_slice KScalePre KScalesZeros BlockOffsets
        BIDX KV_BLOCK_IDX 0 64 8 2 1 5 4 4`
- `initialState : = s`
- `fun i : Fin 4 => metaActive s 4 4 i`
- `fun i : Fin 4 => (KScalesZeros,
          metaOffset s BlockOffsets BIDX KV_BLOCK_IDX 0 64 8 2 1 5 i)`
- `expected : = fun i : Fin 4 => metaStoreSpec s KScalePre i`
- `kernel : = fill_quant_meta_store_slice KZeroPre KScalesZeros BlockOffsets
        BIDX KV_BLOCK_IDX 1 64 8 2 1 5 4 4`
- `initialState : = s`
- `fun i : Fin 4 => metaActive s 4 4 i`
- `fun i : Fin 4 => (KScalesZeros,
          metaOffset s BlockOffsets BIDX KV_BLOCK_IDX 1 64 8 2 1 5 i)`
- `expected : = fun i : Fin 4 => metaStoreSpec s KZeroPre i`
- `kernel : = fill_quant_meta_store_slice VScalePre VScalesZeros BlockOffsets
        BIDX KV_BLOCK_IDX 0 64 8 2 1 5 4 4`
- `initialState : = s`
- `fun i : Fin 4 => metaActive s 4 4 i`
- `fun i : Fin 4 => (VScalesZeros,
          metaOffset s BlockOffsets BIDX KV_BLOCK_IDX 0 64 8 2 1 5 i)`
- `expected : = fun i : Fin 4 => metaStoreSpec s VScalePre i`
- `kernel : = fill_quant_meta_store_slice VZeroPre VScalesZeros BlockOffsets
        BIDX KV_BLOCK_IDX 1 64 8 2 1 5 4 4`
- `initialState : = s`
- `fun i : Fin 4 => metaActive s 4 4 i`
- `fun i : Fin 4 => (VScalesZeros,
          metaOffset s BlockOffsets BIDX KV_BLOCK_IDX 1 64 8 2 1 5 i)`
- `expected : = fun i : Fin 4 => metaStoreSpec s VZeroPre i`

**Closed-form spec defs (transitive):** `quant_int8_compute_store_slice`, `fill_k_cache_tile`, `active`, `kCacheOffset`, `kSourceOffset`, `fill_v_cache_tile`, `vCacheOffset`, `vSourceOffset`, `fill_quant_meta_store_slice`, `metaActive`, `metaOffset`, `metaStoreSpec`, `headIndex`, `dimIndex`, `blockOff`

<details><summary><code>quant_int8_compute_store_slice</code></summary>

```
/-- Proof-oriented int8 quantization compute slice for `_quant_int8`.

The Python helper computes per-head `val_min`, `val_max`, `scales`, `zeros`,
then forms `val / scales[:, None] + zeros[:, None] + 0.5` before the
`tl.uint8` cast. This slice keeps that operator chain explicit and writes the
uint8-cast tile plus scale/zero vectors to proof regions. -/
```
```lean
def quant_int8_compute_store_slice
    (Val QVal ScaleOut ZeroOut : RegionName)
    (stride_vh stride_vd stride_qh stride_qd
      num_heads head_dim BLOCK_H BLOCK_D : Nat) :
    ComputeKernel := triton {
  h_off = tl.arange(0, $(BLOCK_H))
  d_off = tl.arange(0, $(BLOCK_D))
  mask = (h_off[:, None] < $(num_heads)) & (d_off[None, :] < $(head_dim))
  val = tl.load(Val + h_off[:, None] * $(stride_vh) +
      d_off[None, :] * $(stride_vd), mask=mask, other=0.0).to(tl.float32)
  val_min = -tl.max(-val, 1)
  val_max = tl.max(val, 1)
  scales = (val_max - val_min) / 255.0
  zeros = -val_min / scales
  q_val = (val / scales[:, None] + zeros[:, None] + 0.5).to(tl.uint8)
  tl.store(QVal + h_off[:, None] * $(stride_qh) + d_off[None, :] * $(stride_qd),
    q_val, mask=mask)
  tl.store(ScaleOut + h_off, scales, mask=h_off < $(num_heads))
  tl.store(ZeroOut + h_off, zeros, mask=h_off < $(num_heads))
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

<details><summary><code>fill_quant_meta_store_slice</code></summary>

```
/-- Proof-oriented metadata store slice for `_fill_kv_cache_quant_kernel`.

The Python quantized path writes per-head scale values at `szd = 0` and zero
points at `szd = 1` for both K and V metadata regions. This generic slice fixes
one slot (`SZD = 0` or `SZD = 1`) and proves the masked writeback from a
precomputed per-head metadata vector. -/
```
```lean
def fill_quant_meta_store_slice
    (MetaPre MetaOut : RegionName) (BlockOffsets : Region .nat)
    (BIDX KV_BLOCK_IDX SZD
      stride_mn stride_mb stride_mh stride_md stride_boff
      num_heads BLOCK_H : Nat) :
    ComputeKernel := triton {
  batch_id = tl.program_id(0)
  h_off = tl.arange(0, $(BLOCK_H))
  block_off = tl.load(BlockOffsets + batch_id * $(stride_boff) + $(KV_BLOCK_IDX))
  mask = h_off < $(num_heads)
  meta_val = tl.load(MetaPre + h_off, mask=mask, other=0.0)
  tl.store(MetaOut + block_off * $(stride_mn) + $(BIDX) * $(stride_mb) +
      h_off * $(stride_mh) + $(SZD) * $(stride_md), meta_val, mask=mask)
}
```
</details>

<details><summary><code>metaActive</code></summary>

```lean
def metaActive (_s : BlockState) (num_heads BLOCK_H : Nat) (i : Fin BLOCK_H) : Prop :=
  i.val < num_heads
```
</details>

<details><summary><code>metaOffset</code></summary>

```lean
def metaOffset
    (s : BlockState) (BlockOffsets : RegionName)
    (BIDX KV_BLOCK_IDX SZD stride_mn stride_mb stride_mh stride_md stride_boff : Nat)
    (i : Fin BLOCK_H) : Nat :=
  blockOff s BlockOffsets KV_BLOCK_IDX stride_boff * stride_mn +
    BIDX * stride_mb + i.val * stride_mh + SZD * stride_md
```
</details>

<details><summary><code>metaStoreSpec</code></summary>

```lean
noncomputable def metaStoreSpec
    (s : BlockState) (MetaPre : RegionName) (i : Fin BLOCK_H) : ℝ :=
  s.readMem MetaPre i.val
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

<details><summary><code>blockOff</code></summary>

```lean
def blockOff (s : BlockState) (BlockOffsets : RegionName)
    (KV_BLOCK_IDX stride_boff : Nat) : Nat :=
  s.readMemValue .nat BlockOffsets (s.pids 0 * stride_boff + KV_BLOCK_IDX)
```
</details>

## Public theorem: `fill_quant_int4_kv_cache_python_test_layout_summary`

<details><summary>docstring</summary>

```
/-- Python `quant_policy = 4` summary for the checked cache-fill layout.

The int4 helper proof keeps both half-lane uint8 casts and the
`q_val1 + q_val2 * 16` packing formula in the lowering surface, then reuses
the same checked K/V cache and metadata writeback obligations. -/
```
</details>

**Statement:**
```lean
theorem fill_quant_int4_kv_cache_python_test_layout_summary
    (KStatesLo KStatesHi VStatesLo VStatesHi QKPre QVPre KScalePre KZeroPre
      VScalePre VZeroPre KCaches VCaches KScalesZeros VScalesZeros
      BlockOffsets : RegionName)
    (SIDX BIDX KV_BLOCK_IDX : Nat) (s : BlockState) :
    (∃ alg, (quant_int4_compute_store_slice KStatesLo KStatesHi QKPre
      KScalePre KZeroPre 16 1 16 1 8 1 4 8 4 8).toAlgorithm? =
        Except.ok alg) ∧
    (∃ alg, (quant_int4_compute_store_slice VStatesLo VStatesHi QVPre
      VScalePre VZeroPre 16 1 16 1 8 1 4 8 4 8).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := fill_k_cache_tile QKPre KCaches BlockOffsets
        SIDX BIDX KV_BLOCK_IDX 64 16 1 512 64 16 1 5 4 16 4 16)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 4 16 4 16)
        (fun idx => (KCaches,
          kCacheOffset s BlockOffsets BIDX KV_BLOCK_IDX 512 64 16 1 5 idx)))
      (expected := fun idx =>
        s.readMem QKPre (kSourceOffset s SIDX 64 16 1 idx))) ∧
    (ComputeCorrect.Realizes
      (kernel := fill_v_cache_tile QVPre VCaches BlockOffsets
        SIDX BIDX KV_BLOCK_IDX 64 16 1 512 64 16 1 5 4 16 4 16)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 4 16 4 16)
        (fun idx => (VCaches,
          vCacheOffset s BlockOffsets BIDX KV_BLOCK_IDX 512 64 16 1 5 idx)))
      (expected := fun idx =>
        s.readMem QVPre (vSourceOffset s SIDX 64 16 1 idx))) ∧
    (ComputeCorrect.Realizes
      (kernel := fill_quant_meta_store_slice KScalePre KScalesZeros BlockOffsets
        BIDX KV_BLOCK_IDX 0 64 8 2 1 5 4 4)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 4 => metaActive s 4 4 i)
        (fun i : Fin 4 => (KScalesZeros,
          metaOffset s BlockOffsets BIDX KV_BLOCK_IDX 0 64 8 2 1 5 i)))
      (expected := fun i : Fin 4 => metaStoreSpec s KScalePre i)) ∧
    (ComputeCorrect.Realizes
      (kernel := fill_quant_meta_store_slice KZeroPre KScalesZeros BlockOffsets
        BIDX KV_BLOCK_IDX 1 64 8 2 1 5 4 4)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 4 => metaActive s 4 4 i)
        (fun i : Fin 4 => (KScalesZeros,
          metaOffset s BlockOffsets BIDX KV_BLOCK_IDX 1 64 8 2 1 5 i)))
      (expected := fun i : Fin 4 => metaStoreSpec s KZeroPre i)) ∧
    (ComputeCorrect.Realizes
      (kernel := fill_quant_meta_store_slice VScalePre VScalesZeros BlockOffsets
        BIDX KV_BLOCK_IDX 0 64 8 2 1 5 4 4)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 4 => metaActive s 4 4 i)
        (fun i : Fin 4 => (VScalesZeros,
          metaOffset s BlockOffsets BIDX KV_BLOCK_IDX 0 64 8 2 1 5 i)))
      (expected := fun i : Fin 4 => metaStoreSpec s VScalePre i)) ∧
    (ComputeCorrect.Realizes
      (kernel := fill_quant_meta_store_slice VZeroPre VScalesZeros BlockOffsets
        BIDX KV_BLOCK_IDX 1 64 8 2 1 5 4 4)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 4 => metaActive s 4 4 i)
        (fun i : Fin 4 => (VScalesZeros,
          metaOffset s BlockOffsets BIDX KV_BLOCK_IDX 1 64 8 2 1 5 i)))
      (expected := fun i : Fin 4 => metaStoreSpec s VZeroPre i))
```

**Assumptions / layout contracts:**
- `kernel : = fill_k_cache_tile QKPre KCaches BlockOffsets
        SIDX BIDX KV_BLOCK_IDX 64 16 1 512 64 16 1 5 4 16 4 16`
- `initialState : = s`
- `expected : = fun idx =>
        s.readMem QKPre (kSourceOffset s SIDX 64 16 1 idx)`
- `kernel : = fill_v_cache_tile QVPre VCaches BlockOffsets
        SIDX BIDX KV_BLOCK_IDX 64 16 1 512 64 16 1 5 4 16 4 16`
- `initialState : = s`
- `expected : = fun idx =>
        s.readMem QVPre (vSourceOffset s SIDX 64 16 1 idx)`
- `kernel : = fill_quant_meta_store_slice KScalePre KScalesZeros BlockOffsets
        BIDX KV_BLOCK_IDX 0 64 8 2 1 5 4 4`
- `initialState : = s`
- `fun i : Fin 4 => metaActive s 4 4 i`
- `fun i : Fin 4 => (KScalesZeros,
          metaOffset s BlockOffsets BIDX KV_BLOCK_IDX 0 64 8 2 1 5 i)`
- `expected : = fun i : Fin 4 => metaStoreSpec s KScalePre i`
- `kernel : = fill_quant_meta_store_slice KZeroPre KScalesZeros BlockOffsets
        BIDX KV_BLOCK_IDX 1 64 8 2 1 5 4 4`
- `initialState : = s`
- `fun i : Fin 4 => metaActive s 4 4 i`
- `fun i : Fin 4 => (KScalesZeros,
          metaOffset s BlockOffsets BIDX KV_BLOCK_IDX 1 64 8 2 1 5 i)`
- `expected : = fun i : Fin 4 => metaStoreSpec s KZeroPre i`
- `kernel : = fill_quant_meta_store_slice VScalePre VScalesZeros BlockOffsets
        BIDX KV_BLOCK_IDX 0 64 8 2 1 5 4 4`
- `initialState : = s`
- `fun i : Fin 4 => metaActive s 4 4 i`
- `fun i : Fin 4 => (VScalesZeros,
          metaOffset s BlockOffsets BIDX KV_BLOCK_IDX 0 64 8 2 1 5 i)`
- `expected : = fun i : Fin 4 => metaStoreSpec s VScalePre i`
- `kernel : = fill_quant_meta_store_slice VZeroPre VScalesZeros BlockOffsets
        BIDX KV_BLOCK_IDX 1 64 8 2 1 5 4 4`
- `initialState : = s`
- `fun i : Fin 4 => metaActive s 4 4 i`
- `fun i : Fin 4 => (VScalesZeros,
          metaOffset s BlockOffsets BIDX KV_BLOCK_IDX 1 64 8 2 1 5 i)`
- `expected : = fun i : Fin 4 => metaStoreSpec s VZeroPre i`

**Closed-form spec defs (transitive):** `quant_int4_compute_store_slice`, `fill_k_cache_tile`, `active`, `kCacheOffset`, `kSourceOffset`, `fill_v_cache_tile`, `vCacheOffset`, `vSourceOffset`, `fill_quant_meta_store_slice`, `metaActive`, `metaOffset`, `metaStoreSpec`, `headIndex`, `dimIndex`, `blockOff`

<details><summary><code>quant_int4_compute_store_slice</code></summary>

```
/-- Proof-oriented int4 quantization compute slice for `_quant_int4`.

The Python helper casts both halves to float32, computes min/max across the
paired halves, forms scale/zero vectors, rounds both halves with `+ 0.5`, and
packs them as `q_val1 + q_val2 * 16` after uint8 casts. This slice preserves
that full helper chain. -/
```
```lean
def quant_int4_compute_store_slice
    (Val1 Val2 QVal ScaleOut ZeroOut : RegionName)
    (stride_v1h stride_v1d stride_v2h stride_v2d stride_qh stride_qd
      num_heads packed_head_dim BLOCK_H BLOCK_D : Nat) :
    ComputeKernel := triton {
  h_off = tl.arange(0, $(BLOCK_H))
  d_off = tl.arange(0, $(BLOCK_D))
  mask = (h_off[:, None] < $(num_heads)) &
    (d_off[None, :] < $(packed_head_dim))
  val1 = tl.load(Val1 + h_off[:, None] * $(stride_v1h) +
      d_off[None, :] * $(stride_v1d), mask=mask, other=0.0).to(tl.float32)
  val2 = tl.load(Val2 + h_off[:, None] * $(stride_v2h) +
      d_off[None, :] * $(stride_v2d), mask=mask, other=0.0).to(tl.float32)
  val_min = -tl.max(-tl.minimum(val1, val2), 1)
  val_max = tl.max(tl.maximum(val1, val2), 1)
  scales = (val_max - val_min) / 15.0
  zeros = -val_min / scales
  q_val1 = (val1 / scales[:, None] + zeros[:, None] + 0.5).to(tl.uint8)
  q_val2 = (val2 / scales[:, None] + zeros[:, None] + 0.5).to(tl.uint8)
  q_val = q_val1 + q_val2 * 16
  tl.store(QVal + h_off[:, None] * $(stride_qh) + d_off[None, :] * $(stride_qd),
    q_val, mask=mask)
  tl.store(ScaleOut + h_off, scales, mask=h_off < $(num_heads))
  tl.store(ZeroOut + h_off, zeros, mask=h_off < $(num_heads))
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

<details><summary><code>fill_quant_meta_store_slice</code></summary>

```
/-- Proof-oriented metadata store slice for `_fill_kv_cache_quant_kernel`.

The Python quantized path writes per-head scale values at `szd = 0` and zero
points at `szd = 1` for both K and V metadata regions. This generic slice fixes
one slot (`SZD = 0` or `SZD = 1`) and proves the masked writeback from a
precomputed per-head metadata vector. -/
```
```lean
def fill_quant_meta_store_slice
    (MetaPre MetaOut : RegionName) (BlockOffsets : Region .nat)
    (BIDX KV_BLOCK_IDX SZD
      stride_mn stride_mb stride_mh stride_md stride_boff
      num_heads BLOCK_H : Nat) :
    ComputeKernel := triton {
  batch_id = tl.program_id(0)
  h_off = tl.arange(0, $(BLOCK_H))
  block_off = tl.load(BlockOffsets + batch_id * $(stride_boff) + $(KV_BLOCK_IDX))
  mask = h_off < $(num_heads)
  meta_val = tl.load(MetaPre + h_off, mask=mask, other=0.0)
  tl.store(MetaOut + block_off * $(stride_mn) + $(BIDX) * $(stride_mb) +
      h_off * $(stride_mh) + $(SZD) * $(stride_md), meta_val, mask=mask)
}
```
</details>

<details><summary><code>metaActive</code></summary>

```lean
def metaActive (_s : BlockState) (num_heads BLOCK_H : Nat) (i : Fin BLOCK_H) : Prop :=
  i.val < num_heads
```
</details>

<details><summary><code>metaOffset</code></summary>

```lean
def metaOffset
    (s : BlockState) (BlockOffsets : RegionName)
    (BIDX KV_BLOCK_IDX SZD stride_mn stride_mb stride_mh stride_md stride_boff : Nat)
    (i : Fin BLOCK_H) : Nat :=
  blockOff s BlockOffsets KV_BLOCK_IDX stride_boff * stride_mn +
    BIDX * stride_mb + i.val * stride_mh + SZD * stride_md
```
</details>

<details><summary><code>metaStoreSpec</code></summary>

```lean
noncomputable def metaStoreSpec
    (s : BlockState) (MetaPre : RegionName) (i : Fin BLOCK_H) : ℝ :=
  s.readMem MetaPre i.val
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

<details><summary><code>blockOff</code></summary>

```lean
def blockOff (s : BlockState) (BlockOffsets : RegionName)
    (KV_BLOCK_IDX stride_boff : Nat) : Nat :=
  s.readMemValue .nat BlockOffsets (s.pids 0 * stride_boff + KV_BLOCK_IDX)
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
- `fill_k_cache_tile_test_h4_d16_compute_correct`
- `fill_v_cache_tile_test_h4_dv16_compute_correct`
- `fill_quant_k_cache_value_store_test_h4_d16_compute_correct`
- `fill_quant_v_cache_value_store_test_h4_dv16_compute_correct`
- `fill_quant_k_scale_store_test_h4_compute_correct`
- `fill_quant_k_zero_store_test_h4_compute_correct`
- `fill_quant_v_scale_store_test_h4_compute_correct`
- `fill_quant_v_zero_store_test_h4_compute_correct`
- `fill_k_cache_tile_python_test_layout_compute_correct`
- `fill_v_cache_tile_python_test_layout_compute_correct`
- `fill_quant_k_cache_value_store_python_test_layout_compute_correct`
- `fill_quant_v_cache_value_store_python_test_layout_compute_correct`
- `fill_quant_k_scale_store_python_test_layout_compute_correct`
- `fill_quant_k_zero_store_python_test_layout_compute_correct`
- `fill_quant_v_scale_store_python_test_layout_compute_correct`
- `fill_quant_v_zero_store_python_test_layout_compute_correct`
- `fill_kv_cache_python_test_layout_all_outputs_compute_correct`
- `fill_quant_kv_cache_python_test_layout_all_outputs_compute_correct`
