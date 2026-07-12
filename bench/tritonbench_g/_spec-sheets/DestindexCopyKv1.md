# Spec sheet — `bench/tritonbench_g/destindex_copy_kv1/DestindexCopyKv1.lean`

**Python source:** `bench/tritonbench_g/destindex_copy_kv1/destindex_copy_kv1.py`

## Public theorem: `fwd_kernel_destindex_copy_kv_output_summary`

<details><summary>docstring</summary>

```
/-- Per-kernel output summary for `_fwd_kernel_destindex_copy_kv`: the DSL
surface lowers to the algorithm layer, and the dest-indexed masked scatter to
`Out` is compute-correct — every active `[head, dim]` cell holds the matching
cell of `K`, inactive cells are preserved. -/
```
</details>

**Statement:**
```lean
specification fwd_kernel_destindex_copy_kv_output_summary
    (K : RegionName) (Dest_loc : Region .nat) (Out : RegionName)
    (stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
      head_num head_dim BLOCK_DMODEL BLOCK_HEAD : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL] =>
        outAddr s Dest_loc stride_o_bs stride_o_h stride_o_d idx)) :
    (∃ alg, (fwd_kernel_destindex_copy_kv K Dest_loc Out
        stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
        head_num head_dim BLOCK_DMODEL BLOCK_HEAD).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := fwd_kernel_destindex_copy_kv K Dest_loc Out
        stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
        head_num head_dim BLOCK_DMODEL BLOCK_HEAD)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL] => active head_num head_dim idx)
        (fun idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL] =>
          (Out, outAddr s Dest_loc stride_o_bs stride_o_h stride_o_d idx)))
      (expected := fun idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL] =>
        s.readMem K (sourceAddr s stride_k_bs stride_k_h stride_k_d idx))
```

**Assumptions / layout contracts:**
- `hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL] =>
        outAddr s Dest_loc stride_o_bs stride_o_h stride_o_d idx)`
- `fun idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL] => active head_num head_dim idx`
- `fun idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL] =>
          (Out, outAddr s Dest_loc stride_o_bs stride_o_h stride_o_d idx)`

**Closed-form spec defs (transitive):** `outAddr`, `fwd_kernel_destindex_copy_kv`, `active`, `sourceAddr`, `destBase`, `headIndex`, `dimIndex`

<details><summary><code>outAddr</code></summary>

```lean
def outAddr
    (s : BlockState) (Dest_loc : RegionName)
    (stride_o_bs stride_o_h stride_o_d : Nat)
    (idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL]) : Nat :=
  destBase s Dest_loc * stride_o_bs + stride_o_h * headIndex idx + stride_o_d * dimIndex idx
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

<details><summary><code>active</code></summary>

```lean
def active (head_num head_dim : Nat) (idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL]) : Prop :=
  headIndex idx < head_num ∧ dimIndex idx < head_dim
```
</details>

<details><summary><code>sourceAddr</code></summary>

```lean
def sourceAddr
    (s : BlockState) (stride_k_bs stride_k_h stride_k_d : Nat)
    (idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL]) : Nat :=
  s.pid * stride_k_bs + stride_k_h * headIndex idx + stride_k_d * dimIndex idx
```
</details>

<details><summary><code>destBase</code></summary>

```lean
def destBase (s : BlockState) (Dest_loc : RegionName) : Nat :=
  s.readMemValue .nat Dest_loc s.pid
```
</details>

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

## Also present (pinned special-case summaries)
- `fwd_kernel_destindex_copy_kv_compute_correct`
