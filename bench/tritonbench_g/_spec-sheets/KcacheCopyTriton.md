# Spec sheet — `bench/tritonbench_g/kcache_copy_triton/KcacheCopyTriton.lean`

**Python source:** `bench/tritonbench_g/kcache_copy_triton/kcache_copy_triton.py`

## Public theorem: `copy_to_kcache_seqlen_n1_surface_output_summary`

<details><summary>docstring</summary>

```
/-- Per-kernel output summary for the `n_tokens = 1` decode-path K-cache copy:
the DSL surface lowers to the algorithm layer, and the paged split-x scatter to
`KCache` is compute-correct — every cell holds the matching cell of `K` at the
block-table / seq-length-selected cache slot. -/
```
</details>

**Statement:**
```lean
specification copy_to_kcache_seqlen_n1_surface_output_summary
    (K KCache BLOCK_TABLES seq_lengths : RegionName)
    (stride_kt stride_kh stride_kd stride_kcb stride_kch stride_kcsplit_x
      stride_kcs stride_bts stride_btb block_size KCACHE_X : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin KCACHE_X =>
        n1KCacheOffset s BLOCK_TABLES seq_lengths stride_kcb stride_kch
          stride_kcsplit_x stride_kcs stride_bts stride_btb block_size i)) :
    (∃ alg,
      (copy_to_kcache_seqlen_n1_surface K KCache BLOCK_TABLES seq_lengths
        stride_kt stride_kh stride_kd stride_kcb stride_kch stride_kcsplit_x
        stride_kcs 0 stride_bts stride_btb block_size 1 0
        KCACHE_X).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := copy_to_kcache_seqlen_n1_surface K KCache BLOCK_TABLES
        seq_lengths stride_kt stride_kh stride_kd stride_kcb stride_kch
        stride_kcsplit_x stride_kcs 0 stride_bts stride_btb block_size 1 0
        KCACHE_X)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _i : Fin KCACHE_X => True)
        (fun i => (KCache,
          n1KCacheOffset s BLOCK_TABLES seq_lengths stride_kcb stride_kch
            stride_kcsplit_x stride_kcs stride_bts stride_btb block_size i)))
      (expected := fun i =>
        s.readMem K (kSourceOffset s stride_kt stride_kh stride_kd KCACHE_X i))
```

**Assumptions / layout contracts:**
- `hOutInj : Function.Injective
      (fun i : Fin KCACHE_X =>
        n1KCacheOffset s BLOCK_TABLES seq_lengths stride_kcb stride_kch
          stride_kcsplit_x stride_kcs stride_bts stride_btb block_size i)`
- `fun _i : Fin KCACHE_X => True`

**Closed-form spec defs (transitive):** `n1KCacheOffset`, `copy_to_kcache_seqlen_n1_surface`, `kSourceOffset`, `n1BlockId`, `n1OffsetLastBlock`, `dimIndex`, `n1LastBlockIdx`, `n1PastKvSeqLen`

<details><summary><code>n1KCacheOffset</code></summary>

```lean
def n1KCacheOffset
    (s : BlockState) (BLOCK_TABLES seq_lengths : RegionName)
    (stride_kcb stride_kch stride_kcsplit_x stride_kcs
      stride_bts stride_btb block_size : Nat)
    (i : Fin KCACHE_X) : Nat :=
  n1BlockId s BLOCK_TABLES seq_lengths stride_bts stride_btb block_size * stride_kcb +
    s.pids 1 * stride_kch + s.pids 2 * stride_kcsplit_x +
    n1OffsetLastBlock s seq_lengths block_size * stride_kcs + dimIndex i
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
surface. Python's unused `stride_kcx` and `HEAD_DIM` arguments are retained as
ignored parameters; `n_tokens` is fixed to the documented decode value `1` in
the proofs. -/
```
```lean
def copy_to_kcache_seqlen_n1_surface
    (K KCache : RegionName) (BLOCK_TABLES seq_lengths : Region .nat)
    (stride_kt stride_kh stride_kd stride_kcb stride_kch stride_kcsplit_x
      stride_kcs _stride_kcx stride_bts stride_btb block_size _n_tokens
      _HEAD_DIM KCACHE_X : Nat) :
    ComputeKernel := triton {
  cur_token_idx = tl.program_id(0)
  cur_seq_idx = cur_token_idx
  cur_kv_head_idx = tl.program_id(1)
  split_x_idx = tl.program_id(2)
  past_kv_seq_len = tl.load(seq_lengths + cur_seq_idx) - $(1)
  last_bt_block_idx = past_kv_seq_len // $(block_size)
  block_id = tl.load(BLOCK_TABLES + cur_seq_idx * $(stride_bts) +
    last_bt_block_idx * $(stride_btb))
  offset_last_block = past_kv_seq_len % $(block_size)
  offsets_dmodel = split_x_idx * $(KCACHE_X) + tl.arange(0, $(KCACHE_X))
  k = tl.load(K + cur_token_idx * $(stride_kt) +
    cur_kv_head_idx * $(stride_kh) + offsets_dmodel * $(stride_kd))
  tl.store(KCache + block_id * $(stride_kcb) +
    cur_kv_head_idx * $(stride_kch) +
    split_x_idx * $(stride_kcsplit_x) +
    offset_last_block * $(stride_kcs) + tl.arange(0, $(KCACHE_X)), k)
}
```
</details>

<details><summary><code>kSourceOffset</code></summary>

```lean
def kSourceOffset
    (s : BlockState) (stride_kt stride_kh stride_kd KCACHE_X : Nat)
    (i : Fin KCACHE_X) : Nat :=
  s.pids 0 * stride_kt + s.pids 1 * stride_kh +
    (s.pids 2 * KCACHE_X + dimIndex i) * stride_kd
```
</details>

<details><summary><code>n1BlockId</code></summary>

```lean
def n1BlockId (s : BlockState) (BLOCK_TABLES seq_lengths : RegionName)
    (stride_bts stride_btb block_size : Nat) : Nat :=
  s.readMemValue .nat BLOCK_TABLES
    (s.pids 0 * stride_bts + n1LastBlockIdx s seq_lengths block_size * stride_btb)
```
</details>

<details><summary><code>n1OffsetLastBlock</code></summary>

```lean
def n1OffsetLastBlock (s : BlockState) (seq_lengths : RegionName)
    (block_size : Nat) : Nat :=
  n1PastKvSeqLen s seq_lengths % block_size
```
</details>

<details><summary><code>dimIndex</code></summary>

```lean
def dimIndex (i : Fin KCACHE_X) : Nat :=
  i.val
```
</details>

<details><summary><code>n1LastBlockIdx</code></summary>

```lean
def n1LastBlockIdx (s : BlockState) (seq_lengths : RegionName)
    (block_size : Nat) : Nat :=
  n1PastKvSeqLen s seq_lengths / block_size
```
</details>

<details><summary><code>n1PastKvSeqLen</code></summary>

```lean
def n1PastKvSeqLen (s : BlockState) (seq_lengths : RegionName) : Nat :=
  s.readMemValue .nat seq_lengths (s.pids 0) - 1
```
</details>

## Also present (pinned special-case summaries)
- `copy_to_kcache_seqlen_n1_surface_compute_correct`
- `copy_to_kcache_seqlen_n1_old_layout_block_compute_correct`
- `copy_to_kcache_seqlen_n1_new_layout_xblock_compute_correct`
- `copy_to_kcache_split_x_block_compute_correct`
