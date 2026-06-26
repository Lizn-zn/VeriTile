# Spec sheet — `bench/tritonbench_g/kv_cache_copy/KvCacheCopy.lean`

**Python source:** `bench/tritonbench_g/kv_cache_copy/kv_cache_copy.py`

## Public theorem: `kv_cache_copy_seqlen1_output_summary_general`

<details><summary>docstring</summary>

```
/-- **Dimension-general** old-layout (`seqlen1`, single x-block) summary.

Symbolic-dimension version of `kv_cache_copy_python_case1_all_outputs_summary`:
the surface lowers, the legacy K-cache writeback covers the whole head in one
x-block, and the V-cache writeback realizes its strides. Offset-injectivity for
the K and V cache stores is supplied as hypotheses. -/
```
</details>

**Statement:**
```lean
theorem kv_cache_copy_seqlen1_output_summary_general
    (K V KCache VCache : RegionName)
    (BLOCK_TABLES context_lengths : Region .nat)
    (stride_kt stride_kh stride_kd
      stride_vt stride_vh stride_vd
      stride_kcb stride_kch stride_kcs _stride_kcd
      stride_vcb stride_vch stride_vcs stride_vcd
      stride_bts stride_btb block_size HEAD_DIM : Nat)
    (s : BlockState)
    (hKInj : Function.Injective
      (fun i : Fin HEAD_DIM =>
        seqlen1KCacheOffset s BLOCK_TABLES context_lengths 0 stride_kcb
          stride_kch 0 stride_kcs stride_bts stride_btb block_size HEAD_DIM i))
    (hVInj : Function.Injective
      (fun i : Fin HEAD_DIM =>
        seqlen1VCacheOffset s BLOCK_TABLES context_lengths stride_vcb stride_vch
          stride_vcs stride_vcd stride_bts stride_btb block_size i)) :
    (∃ alg, (copy_to_kvcache_seqlen1_kernel K V KCache VCache BLOCK_TABLES
      context_lengths stride_kt stride_kh stride_kd stride_vt stride_vh stride_vd
      stride_kcb stride_kch 0 stride_kcs _stride_kcd stride_vcb stride_vch
      stride_vcs stride_vcd stride_bts stride_btb block_size HEAD_DIM
      HEAD_DIM).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := copy_to_kcache_seqlen1_xblock K KCache BLOCK_TABLES
        context_lengths 0 stride_kt stride_kh stride_kd stride_kcb stride_kch
        0 stride_kcs stride_bts stride_btb block_size HEAD_DIM HEAD_DIM)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin HEAD_DIM => kActive 0 HEAD_DIM HEAD_DIM i)
        (fun i => (KCache,
          seqlen1KCacheOffset s BLOCK_TABLES context_lengths 0 stride_kcb
            stride_kch 0 stride_kcs stride_bts stride_btb block_size HEAD_DIM i)))
      (expected := fun i =>
        s.readMem K (kSourceOffset s 0 stride_kt stride_kh stride_kd
          HEAD_DIM i))) ∧
    (ComputeCorrect.Realizes
      (kernel := copy_to_vcache_seqlen1_dblock V VCache BLOCK_TABLES
        context_lengths stride_vt stride_vh stride_vd stride_vcb stride_vch
        stride_vcs stride_vcd stride_bts stride_btb block_size HEAD_DIM HEAD_DIM)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin HEAD_DIM => active HEAD_DIM i)
        (fun i => (VCache,
          seqlen1VCacheOffset s BLOCK_TABLES context_lengths stride_vcb
            stride_vch stride_vcs stride_vcd stride_bts stride_btb block_size i)))
      (expected := fun i =>
        s.readMem V (vSourceOffset s stride_vt stride_vh stride_vd i)))
```

**Assumptions / layout contracts:**
- `hKInj : Function.Injective
      (fun i : Fin HEAD_DIM =>
        seqlen1KCacheOffset s BLOCK_TABLES context_lengths 0 stride_kcb
          stride_kch 0 stride_kcs stride_bts stride_btb block_size HEAD_DIM i)`
- `hVInj : Function.Injective
      (fun i : Fin HEAD_DIM =>
        seqlen1VCacheOffset s BLOCK_TABLES context_lengths stride_vcb stride_vch
          stride_vcs stride_vcd stride_bts stride_btb block_size i)`
- `fun i : Fin HEAD_DIM => kActive 0 HEAD_DIM HEAD_DIM i`
- `fun i : Fin HEAD_DIM => active HEAD_DIM i`

**Closed-form spec defs (transitive):** `seqlen1KCacheOffset`, `seqlen1VCacheOffset`, `copy_to_kvcache_seqlen1_kernel`, `copy_to_kcache_seqlen1_xblock`, `kActive`, `kSourceOffset`, `copy_to_vcache_seqlen1_dblock`, `active`, `vSourceOffset`, `seqlen1BlockId`, `seqlen1OffsetLastBlock`, `dimIndex`, `kDimIndex`, `seqlen1LastBlockIdx`, `seqlen1PastKvSeqLen`

<details><summary><code>seqlen1KCacheOffset</code></summary>

```lean
def seqlen1KCacheOffset
    (s : BlockState) (BLOCK_TABLES context_lengths : RegionName)
    (SPLIT_X stride_kcb stride_kch stride_kcsplit_x stride_kcs stride_bts
      stride_btb block_size KCACHE_X : Nat)
    (i : Fin KCACHE_X) : Nat :=
  seqlen1BlockId s BLOCK_TABLES context_lengths stride_bts stride_btb block_size *
      stride_kcb +
    s.pids 1 * stride_kch + SPLIT_X * stride_kcsplit_x +
    seqlen1OffsetLastBlock s context_lengths block_size * stride_kcs + i.val
```
</details>

<details><summary><code>seqlen1VCacheOffset</code></summary>

```lean
def seqlen1VCacheOffset
    (s : BlockState) (BLOCK_TABLES context_lengths : RegionName)
    (stride_vcb stride_vch stride_vcs stride_vcd stride_bts stride_btb
      block_size : Nat)
    (i : Fin BLOCK_D) : Nat :=
  seqlen1BlockId s BLOCK_TABLES context_lengths stride_bts stride_btb block_size *
      stride_vcb +
    s.pids 1 * stride_vch +
    seqlen1OffsetLastBlock s context_lengths block_size * stride_vcs +
    dimIndex i * stride_vcd
```
</details>

<details><summary><code>copy_to_kvcache_seqlen1_kernel</code></summary>

```
/-- Faithful transcription of `kv_cache_copy.py`'s
`_copy_to_kvcache_seqlen1_kernel`. -/
```
```lean
def copy_to_kvcache_seqlen1_kernel
    (K V KCache VCache : RegionName)
    (BLOCK_TABLES context_lengths : Region .nat)
    (stride_kt stride_kh stride_kd
      stride_vt stride_vh stride_vd
      stride_kcb stride_kch stride_kcsplit_x stride_kcs _stride_kcd
      stride_vcb stride_vch stride_vcs stride_vcd
      stride_bts stride_btb block_size HEAD_DIM KCACHE_X : Nat) :
    ComputeKernel := triton {
  cur_seq_idx = tl.program_id(0)
  cur_kv_head_idx = tl.program_id(1)
  past_kv_seq_len = tl.load(context_lengths + cur_seq_idx) - $(1)
  last_bt_block_idx = past_kv_seq_len // $(block_size)
  block_table_ptr = BLOCK_TABLES + cur_seq_idx * $(stride_bts)
  block_id = tl.load(block_table_ptr + last_bt_block_idx * $(stride_btb))
  offsets_in_last_block = past_kv_seq_len % $(block_size)
  range_x = tl.arange(0, $(KCACHE_X))
  offsets_dmodel_x_partition = tl.arange(0, $(KCACHE_X))
  for split_x in tl.static_range($((HEAD_DIM / KCACHE_X : Nat))) {
    offsets_dmodel_x_partition =
      tl.arange(split_x * $(KCACHE_X), (split_x + $(1)) * $(KCACHE_X))
    offsets_k = cur_seq_idx * $(stride_kt) + cur_kv_head_idx * $(stride_kh) +
      offsets_dmodel_x_partition * $(stride_kd)
    k = tl.load(K + offsets_k)
    offsets_v = cur_seq_idx * $(stride_vt) + cur_kv_head_idx * $(stride_vh) +
      offsets_dmodel_x_partition * $(stride_vd)
    v = tl.load(V + offsets_v)
    offsets_kcache = block_id * $(stride_kcb) +
      cur_kv_head_idx * $(stride_kch) +
      split_x * $(stride_kcsplit_x) +
      offsets_in_last_block * $(stride_kcs) +
      range_x
    tl.store(KCache + offsets_kcache, k)
    offsets_vcache = block_id * $(stride_vcb) +
      cur_kv_head_idx * $(stride_vch) +
      offsets_in_last_block * $(stride_vcs) +
      offsets_dmodel_x_partition * $(stride_vcd)
    tl.store(VCache + offsets_vcache, v)
  }
}
```
</details>

<details><summary><code>copy_to_kcache_seqlen1_xblock</code></summary>

```
/-- Surface transcription of the K-cache store in `kv_cache_copy.py`'s
`_copy_to_kvcache_seqlen1_kernel`.

Unlike `copy_to_kcache_one_xblock`, this keeps the decode-path arithmetic from
the Python kernel: `context_lengths[cur_seq_idx] - 1`, block-table lookup,
last-block offset, one K load, and the K-cache store for one `split_x`
partition. -/
```
```lean
def copy_to_kcache_seqlen1_xblock
    (K KCache : RegionName) (BLOCK_TABLES context_lengths : Region .nat)
    (SPLIT_X stride_kt stride_kh stride_kd
      stride_kcb stride_kch stride_kcsplit_x stride_kcs
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
  tl.store(KCache + block_id * $(stride_kcb) +
      cur_kv_head_idx * $(stride_kch) +
      $(SPLIT_X) * $(stride_kcsplit_x) +
      offsets_in_last_block * $(stride_kcs) + range_x,
    k, mask=offsets_dmodel_x_partition < $(HEAD_DIM))
}
```
</details>

<details><summary><code>kActive</code></summary>

```lean
def kActive (SPLIT_X HEAD_DIM KCACHE_X : Nat) (i : Fin KCACHE_X) : Prop :=
  kDimIndex SPLIT_X KCACHE_X i < HEAD_DIM
```
</details>

<details><summary><code>kSourceOffset</code></summary>

```lean
def kSourceOffset
    (s : BlockState) (SPLIT_X stride_kt stride_kh stride_kd KCACHE_X : Nat)
    (i : Fin KCACHE_X) : Nat :=
  s.pids 0 * stride_kt + s.pids 1 * stride_kh +
    kDimIndex SPLIT_X KCACHE_X i * stride_kd
```
</details>

<details><summary><code>copy_to_vcache_seqlen1_dblock</code></summary>

```
/-- Surface transcription of the V-cache store in `kv_cache_copy.py`'s
`_copy_to_kvcache_seqlen1_kernel`.

This keeps the Python decode path's `context_lengths[cur_seq_idx] - 1`, block
division/modulo, block-table lookup, V load, and V-cache store for one
dimension block. -/
```
```lean
def copy_to_vcache_seqlen1_dblock
    (V VCache : RegionName) (BLOCK_TABLES context_lengths : Region .nat)
    (stride_vt stride_vh stride_vd stride_vcb stride_vch stride_vcs stride_vcd
      stride_bts stride_btb block_size HEAD_DIM BLOCK_D : Nat) :
    ComputeKernel := triton {
  cur_seq_idx = tl.program_id(0)
  cur_kv_head_idx = tl.program_id(1)
  d = tl.arange(0, $(BLOCK_D))
  past_kv_seq_len = tl.load(context_lengths + cur_seq_idx) - $(1)
  last_bt_block_idx = past_kv_seq_len // $(block_size)
  block_id = tl.load(BLOCK_TABLES + cur_seq_idx * $(stride_bts) +
    last_bt_block_idx * $(stride_btb))
  offset_last_block = past_kv_seq_len % $(block_size)
  v = tl.load(V + cur_seq_idx * $(stride_vt) +
      cur_kv_head_idx * $(stride_vh) + d * $(stride_vd),
    mask=d < $(HEAD_DIM), other=0.0)
  tl.store(VCache + block_id * $(stride_vcb) +
      cur_kv_head_idx * $(stride_vch) + offset_last_block * $(stride_vcs) +
      d * $(stride_vcd),
    v, mask=d < $(HEAD_DIM))
}
```
</details>

<details><summary><code>active</code></summary>

```lean
def active (HEAD_DIM : Nat) (i : Fin BLOCK_D) : Prop :=
  dimIndex i < HEAD_DIM
```
</details>

<details><summary><code>vSourceOffset</code></summary>

```lean
def vSourceOffset
    (s : BlockState) (stride_vt stride_vh stride_vd : Nat)
    (i : Fin BLOCK_D) : Nat :=
  s.pids 0 * stride_vt + s.pids 1 * stride_vh + dimIndex i * stride_vd
```
</details>

<details><summary><code>seqlen1BlockId</code></summary>

```lean
def seqlen1BlockId (s : BlockState) (BLOCK_TABLES context_lengths : RegionName)
    (stride_bts stride_btb block_size : Nat) : Nat :=
  s.readMemValue .nat BLOCK_TABLES
    (s.pids 0 * stride_bts +
      seqlen1LastBlockIdx s context_lengths block_size * stride_btb)
```
</details>

<details><summary><code>seqlen1OffsetLastBlock</code></summary>

```lean
def seqlen1OffsetLastBlock (s : BlockState) (context_lengths : RegionName)
    (block_size : Nat) : Nat :=
  seqlen1PastKvSeqLen s context_lengths % block_size
```
</details>

<details><summary><code>dimIndex</code></summary>

```lean
def dimIndex (i : Fin BLOCK_D) : Nat :=
  i.val
```
</details>

<details><summary><code>kDimIndex</code></summary>

```lean
def kDimIndex (SPLIT_X KCACHE_X : Nat) (i : Fin KCACHE_X) : Nat :=
  SPLIT_X * KCACHE_X + i.val
```
</details>

<details><summary><code>seqlen1LastBlockIdx</code></summary>

```lean
def seqlen1LastBlockIdx (s : BlockState) (context_lengths : RegionName)
    (block_size : Nat) : Nat :=
  seqlen1PastKvSeqLen s context_lengths / block_size
```
</details>

<details><summary><code>seqlen1PastKvSeqLen</code></summary>

```lean
def seqlen1PastKvSeqLen (s : BlockState) (context_lengths : RegionName) : Nat :=
  s.readMemValue .nat context_lengths (s.pids 0) - 1
```
</details>

## Public theorem: `kv_cache_copy_split_x_output_summary_general`

<details><summary>docstring</summary>

```
/-- **Dimension-general** split-x (new layout) summary.

Symbolic-dimension version of `kv_cache_copy_python_case2_all_outputs_summary`:
the surface lowers, every `split_x` partition's K-cache writeback realizes its
x-block, and the V-cache writeback realizes its strides. Per-`split_x`
K-cache offset-injectivity and the V-cache offset-injectivity are hypotheses. -/
```
</details>

**Statement:**
```lean
theorem kv_cache_copy_split_x_output_summary_general
    (K V KCache VCache : RegionName)
    (BLOCK_TABLES context_lengths : Region .nat)
    (stride_kt stride_kh stride_kd
      stride_vt stride_vh stride_vd
      stride_kcb stride_kch stride_kcsplit_x stride_kcs _stride_kcd
      stride_vcb stride_vch stride_vcs stride_vcd
      stride_bts stride_btb block_size HEAD_DIM KCACHE_X NUM_SPLITS : Nat)
    (s : BlockState)
    (hKInj : ∀ split_x : Fin NUM_SPLITS, Function.Injective
      (fun i : Fin KCACHE_X =>
        seqlen1KCacheOffset s BLOCK_TABLES context_lengths split_x.val stride_kcb
          stride_kch stride_kcsplit_x stride_kcs stride_bts stride_btb block_size
          KCACHE_X i))
    (hVInj : Function.Injective
      (fun i : Fin HEAD_DIM =>
        seqlen1VCacheOffset s BLOCK_TABLES context_lengths stride_vcb stride_vch
          stride_vcs stride_vcd stride_bts stride_btb block_size i)) :
    (∃ alg, (copy_to_kvcache_seqlen1_kernel K V KCache VCache BLOCK_TABLES
      context_lengths stride_kt stride_kh stride_kd stride_vt stride_vh stride_vd
      stride_kcb stride_kch stride_kcsplit_x stride_kcs _stride_kcd stride_vcb
      stride_vch stride_vcs stride_vcd stride_bts stride_btb block_size HEAD_DIM
      KCACHE_X).toAlgorithm? = Except.ok alg) ∧
    (∀ split_x : Fin NUM_SPLITS,
      ComputeCorrect.Realizes
        (kernel := copy_to_kcache_seqlen1_xblock K KCache BLOCK_TABLES
          context_lengths split_x.val stride_kt stride_kh stride_kd stride_kcb
          stride_kch stride_kcsplit_x stride_kcs stride_bts stride_btb block_size
          HEAD_DIM KCACHE_X)
        (initialState := s)
        (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin KCACHE_X => kActive split_x.val HEAD_DIM KCACHE_X i)
          (fun i => (KCache,
            seqlen1KCacheOffset s BLOCK_TABLES context_lengths split_x.val
              stride_kcb stride_kch stride_kcsplit_x stride_kcs stride_bts
              stride_btb block_size KCACHE_X i)))
        (expected := fun i =>
          s.readMem K (kSourceOffset s split_x.val stride_kt stride_kh stride_kd
            KCACHE_X i))) ∧
    (ComputeCorrect.Realizes
      (kernel := copy_to_vcache_seqlen1_dblock V VCache BLOCK_TABLES
        context_lengths stride_vt stride_vh stride_vd stride_vcb stride_vch
        stride_vcs stride_vcd stride_bts stride_btb block_size HEAD_DIM HEAD_DIM)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin HEAD_DIM => active HEAD_DIM i)
        (fun i => (VCache,
          seqlen1VCacheOffset s BLOCK_TABLES context_lengths stride_vcb
            stride_vch stride_vcs stride_vcd stride_bts stride_btb block_size i)))
      (expected := fun i =>
        s.readMem V (vSourceOffset s stride_vt stride_vh stride_vd i)))
```

**Assumptions / layout contracts:**
- `hKInj : ∀ split_x : Fin NUM_SPLITS, Function.Injective
      (fun i : Fin KCACHE_X =>
        seqlen1KCacheOffset s BLOCK_TABLES context_lengths split_x.val stride_kcb
          stride_kch stride_kcsplit_x stride_kcs stride_bts stride_btb block_size
          KCACHE_X i)`
- `hVInj : Function.Injective
      (fun i : Fin HEAD_DIM =>
        seqlen1VCacheOffset s BLOCK_TABLES context_lengths stride_vcb stride_vch
          stride_vcs stride_vcd stride_bts stride_btb block_size i)`
- `fun i : Fin KCACHE_X => kActive split_x.val HEAD_DIM KCACHE_X i`
- `fun i : Fin HEAD_DIM => active HEAD_DIM i`

**Closed-form spec defs (transitive):** `seqlen1KCacheOffset`, `seqlen1VCacheOffset`, `copy_to_kvcache_seqlen1_kernel`, `copy_to_kcache_seqlen1_xblock`, `kActive`, `kSourceOffset`, `copy_to_vcache_seqlen1_dblock`, `active`, `vSourceOffset`, `seqlen1BlockId`, `seqlen1OffsetLastBlock`, `dimIndex`, `kDimIndex`, `seqlen1LastBlockIdx`, `seqlen1PastKvSeqLen`

<details><summary><code>seqlen1KCacheOffset</code></summary>

```lean
def seqlen1KCacheOffset
    (s : BlockState) (BLOCK_TABLES context_lengths : RegionName)
    (SPLIT_X stride_kcb stride_kch stride_kcsplit_x stride_kcs stride_bts
      stride_btb block_size KCACHE_X : Nat)
    (i : Fin KCACHE_X) : Nat :=
  seqlen1BlockId s BLOCK_TABLES context_lengths stride_bts stride_btb block_size *
      stride_kcb +
    s.pids 1 * stride_kch + SPLIT_X * stride_kcsplit_x +
    seqlen1OffsetLastBlock s context_lengths block_size * stride_kcs + i.val
```
</details>

<details><summary><code>seqlen1VCacheOffset</code></summary>

```lean
def seqlen1VCacheOffset
    (s : BlockState) (BLOCK_TABLES context_lengths : RegionName)
    (stride_vcb stride_vch stride_vcs stride_vcd stride_bts stride_btb
      block_size : Nat)
    (i : Fin BLOCK_D) : Nat :=
  seqlen1BlockId s BLOCK_TABLES context_lengths stride_bts stride_btb block_size *
      stride_vcb +
    s.pids 1 * stride_vch +
    seqlen1OffsetLastBlock s context_lengths block_size * stride_vcs +
    dimIndex i * stride_vcd
```
</details>

<details><summary><code>copy_to_kvcache_seqlen1_kernel</code></summary>

```
/-- Faithful transcription of `kv_cache_copy.py`'s
`_copy_to_kvcache_seqlen1_kernel`. -/
```
```lean
def copy_to_kvcache_seqlen1_kernel
    (K V KCache VCache : RegionName)
    (BLOCK_TABLES context_lengths : Region .nat)
    (stride_kt stride_kh stride_kd
      stride_vt stride_vh stride_vd
      stride_kcb stride_kch stride_kcsplit_x stride_kcs _stride_kcd
      stride_vcb stride_vch stride_vcs stride_vcd
      stride_bts stride_btb block_size HEAD_DIM KCACHE_X : Nat) :
    ComputeKernel := triton {
  cur_seq_idx = tl.program_id(0)
  cur_kv_head_idx = tl.program_id(1)
  past_kv_seq_len = tl.load(context_lengths + cur_seq_idx) - $(1)
  last_bt_block_idx = past_kv_seq_len // $(block_size)
  block_table_ptr = BLOCK_TABLES + cur_seq_idx * $(stride_bts)
  block_id = tl.load(block_table_ptr + last_bt_block_idx * $(stride_btb))
  offsets_in_last_block = past_kv_seq_len % $(block_size)
  range_x = tl.arange(0, $(KCACHE_X))
  offsets_dmodel_x_partition = tl.arange(0, $(KCACHE_X))
  for split_x in tl.static_range($((HEAD_DIM / KCACHE_X : Nat))) {
    offsets_dmodel_x_partition =
      tl.arange(split_x * $(KCACHE_X), (split_x + $(1)) * $(KCACHE_X))
    offsets_k = cur_seq_idx * $(stride_kt) + cur_kv_head_idx * $(stride_kh) +
      offsets_dmodel_x_partition * $(stride_kd)
    k = tl.load(K + offsets_k)
    offsets_v = cur_seq_idx * $(stride_vt) + cur_kv_head_idx * $(stride_vh) +
      offsets_dmodel_x_partition * $(stride_vd)
    v = tl.load(V + offsets_v)
    offsets_kcache = block_id * $(stride_kcb) +
      cur_kv_head_idx * $(stride_kch) +
      split_x * $(stride_kcsplit_x) +
      offsets_in_last_block * $(stride_kcs) +
      range_x
    tl.store(KCache + offsets_kcache, k)
    offsets_vcache = block_id * $(stride_vcb) +
      cur_kv_head_idx * $(stride_vch) +
      offsets_in_last_block * $(stride_vcs) +
      offsets_dmodel_x_partition * $(stride_vcd)
    tl.store(VCache + offsets_vcache, v)
  }
}
```
</details>

<details><summary><code>copy_to_kcache_seqlen1_xblock</code></summary>

```
/-- Surface transcription of the K-cache store in `kv_cache_copy.py`'s
`_copy_to_kvcache_seqlen1_kernel`.

Unlike `copy_to_kcache_one_xblock`, this keeps the decode-path arithmetic from
the Python kernel: `context_lengths[cur_seq_idx] - 1`, block-table lookup,
last-block offset, one K load, and the K-cache store for one `split_x`
partition. -/
```
```lean
def copy_to_kcache_seqlen1_xblock
    (K KCache : RegionName) (BLOCK_TABLES context_lengths : Region .nat)
    (SPLIT_X stride_kt stride_kh stride_kd
      stride_kcb stride_kch stride_kcsplit_x stride_kcs
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
  tl.store(KCache + block_id * $(stride_kcb) +
      cur_kv_head_idx * $(stride_kch) +
      $(SPLIT_X) * $(stride_kcsplit_x) +
      offsets_in_last_block * $(stride_kcs) + range_x,
    k, mask=offsets_dmodel_x_partition < $(HEAD_DIM))
}
```
</details>

<details><summary><code>kActive</code></summary>

```lean
def kActive (SPLIT_X HEAD_DIM KCACHE_X : Nat) (i : Fin KCACHE_X) : Prop :=
  kDimIndex SPLIT_X KCACHE_X i < HEAD_DIM
```
</details>

<details><summary><code>kSourceOffset</code></summary>

```lean
def kSourceOffset
    (s : BlockState) (SPLIT_X stride_kt stride_kh stride_kd KCACHE_X : Nat)
    (i : Fin KCACHE_X) : Nat :=
  s.pids 0 * stride_kt + s.pids 1 * stride_kh +
    kDimIndex SPLIT_X KCACHE_X i * stride_kd
```
</details>

<details><summary><code>copy_to_vcache_seqlen1_dblock</code></summary>

```
/-- Surface transcription of the V-cache store in `kv_cache_copy.py`'s
`_copy_to_kvcache_seqlen1_kernel`.

This keeps the Python decode path's `context_lengths[cur_seq_idx] - 1`, block
division/modulo, block-table lookup, V load, and V-cache store for one
dimension block. -/
```
```lean
def copy_to_vcache_seqlen1_dblock
    (V VCache : RegionName) (BLOCK_TABLES context_lengths : Region .nat)
    (stride_vt stride_vh stride_vd stride_vcb stride_vch stride_vcs stride_vcd
      stride_bts stride_btb block_size HEAD_DIM BLOCK_D : Nat) :
    ComputeKernel := triton {
  cur_seq_idx = tl.program_id(0)
  cur_kv_head_idx = tl.program_id(1)
  d = tl.arange(0, $(BLOCK_D))
  past_kv_seq_len = tl.load(context_lengths + cur_seq_idx) - $(1)
  last_bt_block_idx = past_kv_seq_len // $(block_size)
  block_id = tl.load(BLOCK_TABLES + cur_seq_idx * $(stride_bts) +
    last_bt_block_idx * $(stride_btb))
  offset_last_block = past_kv_seq_len % $(block_size)
  v = tl.load(V + cur_seq_idx * $(stride_vt) +
      cur_kv_head_idx * $(stride_vh) + d * $(stride_vd),
    mask=d < $(HEAD_DIM), other=0.0)
  tl.store(VCache + block_id * $(stride_vcb) +
      cur_kv_head_idx * $(stride_vch) + offset_last_block * $(stride_vcs) +
      d * $(stride_vcd),
    v, mask=d < $(HEAD_DIM))
}
```
</details>

<details><summary><code>active</code></summary>

```lean
def active (HEAD_DIM : Nat) (i : Fin BLOCK_D) : Prop :=
  dimIndex i < HEAD_DIM
```
</details>

<details><summary><code>vSourceOffset</code></summary>

```lean
def vSourceOffset
    (s : BlockState) (stride_vt stride_vh stride_vd : Nat)
    (i : Fin BLOCK_D) : Nat :=
  s.pids 0 * stride_vt + s.pids 1 * stride_vh + dimIndex i * stride_vd
```
</details>

<details><summary><code>seqlen1BlockId</code></summary>

```lean
def seqlen1BlockId (s : BlockState) (BLOCK_TABLES context_lengths : RegionName)
    (stride_bts stride_btb block_size : Nat) : Nat :=
  s.readMemValue .nat BLOCK_TABLES
    (s.pids 0 * stride_bts +
      seqlen1LastBlockIdx s context_lengths block_size * stride_btb)
```
</details>

<details><summary><code>seqlen1OffsetLastBlock</code></summary>

```lean
def seqlen1OffsetLastBlock (s : BlockState) (context_lengths : RegionName)
    (block_size : Nat) : Nat :=
  seqlen1PastKvSeqLen s context_lengths % block_size
```
</details>

<details><summary><code>dimIndex</code></summary>

```lean
def dimIndex (i : Fin BLOCK_D) : Nat :=
  i.val
```
</details>

<details><summary><code>kDimIndex</code></summary>

```lean
def kDimIndex (SPLIT_X KCACHE_X : Nat) (i : Fin KCACHE_X) : Nat :=
  SPLIT_X * KCACHE_X + i.val
```
</details>

<details><summary><code>seqlen1LastBlockIdx</code></summary>

```lean
def seqlen1LastBlockIdx (s : BlockState) (context_lengths : RegionName)
    (block_size : Nat) : Nat :=
  seqlen1PastKvSeqLen s context_lengths / block_size
```
</details>

<details><summary><code>seqlen1PastKvSeqLen</code></summary>

```lean
def seqlen1PastKvSeqLen (s : BlockState) (context_lengths : RegionName) : Nat :=
  s.readMemValue .nat context_lengths (s.pids 0) - 1
```
</details>

## Also present (pinned special-case summaries)
- `copy_to_kcache_one_xblock_compute_correct`
- `copy_to_kcache_seqlen1_xblock_compute_correct`
- `copy_to_kcache_seqlen1_old_layout_block_compute_correct`
- `copy_to_kcache_seqlen1_new_layout_xblock_compute_correct`
- `copy_to_kcache_old_layout_block_compute_correct`
- `copy_to_kcache_new_layout_xblock_compute_correct`
- `copy_to_vcache_seqlen1_dblock_compute_correct`
- `copy_to_vcache_one_dblock_compute_correct`
- `kv_cache_copy_python_case1_all_outputs_summary`
- `kv_cache_copy_python_case2_all_outputs_summary`
