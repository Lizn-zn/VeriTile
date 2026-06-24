# Spec sheet — `bench/tritonbench_g/chunked_cumsum_fwd/ChunkedCumsumFwd.lean`

**Python source:** `bench/tritonbench_g/chunked_cumsum_fwd/chunked_cumsum_fwd.py`

## Public theorem: `chunked_cumsum_fwd_summary_general`

<details><summary>docstring</summary>

```
/-- **Dimension-general genuine correctness summary for chunked cumsum forward.**
The full surface lowers to the algorithm layer for *arbitrary* dimensions,
strides, and the `HAS_DT_BIAS` / `DT_SOFTPLUS` flags, and — given output-offset
injectivity — every output slice realizes its genuine specification, with the
`dA_cumsum` compute slice realizing the standalone closed form `dAClosed`. This
removes the test-shape pin: the statement is universally quantified over all
`Nat` dimension/stride parameters (cf. the symbolic sibling
`chunk_cumsum_scalar_output_summary_general`, generalized over `T`, `BT`). -/
```
</details>

**Statement:**
```lean
theorem chunked_cumsum_fwd_summary_general
    (dt_ptr A_ptr dt_bias_ptr dt_out_ptr dA_cumsum_ptr
      DtPrepared DtOut DAcs A DACumsum : RegionName)
    (batch seqlen nheads chunk_size : Nat)
    (dt_min dt_max : ℝ)
    (stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head
      stride_dt_bias_head
      stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize
      stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize : Nat)
    (DT_SOFTPLUS HAS_DT_BIAS : Bool)
    (BLOCK_SIZE_H BLOCK_SIZE_CHUNK : Nat)
    (s : BlockState)
    (hDtOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE_H, BLOCK_SIZE_CHUNK] =>
        dtOutOffset s stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head
          stride_dt_out_csize BLOCK_SIZE_H idx))
    (hDACsInj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE_H, BLOCK_SIZE_CHUNK] =>
        dACsOutOffset s stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head
          stride_dA_cs_csize BLOCK_SIZE_H idx)) :
    (∃ alg, (chunked_cumsum_fwd_surface dt_ptr A_ptr dt_bias_ptr dt_out_ptr
      dA_cumsum_ptr batch seqlen nheads chunk_size dt_min dt_max stride_dt_batch
      stride_dt_seqlen stride_dt_head stride_A_head stride_dt_bias_head
      stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize
      stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize
      DT_SOFTPLUS HAS_DT_BIAS BLOCK_SIZE_H BLOCK_SIZE_CHUNK).toAlgorithm?
        = Except.ok alg) ∧
    ((ComputeCorrect.Realizes
      (kernel := chunked_cumsum_dt_out_store_slice DtPrepared DtOut
        stride_dt_batch stride_dt_seqlen stride_dt_head stride_dt_out_batch
        stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize nheads
        chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK)
        (fun idx => (DtOut,
          dtOutOffset s stride_dt_out_batch stride_dt_out_chunk
            stride_dt_out_head stride_dt_out_csize BLOCK_SIZE_H idx)))
      (expected := fun idx =>
        s.readMem DtPrepared
          (dtPreparedOffset s stride_dt_batch stride_dt_seqlen stride_dt_head
            chunk_size BLOCK_SIZE_H idx))) ∧
    (ComputeCorrect.Realizes
      (kernel := chunked_cumsum_dA_cs_store_slice DAcs DACumsum
        stride_dt_batch stride_dt_seqlen stride_dt_head stride_dA_cs_batch
        stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize nheads
        chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK)
        (fun idx => (DACumsum,
          dACsOutOffset s stride_dA_cs_batch stride_dA_cs_chunk
            stride_dA_cs_head stride_dA_cs_csize BLOCK_SIZE_H idx)))
      (expected := fun idx =>
        s.readMem DAcs
          (dtPreparedOffset s stride_dt_batch stride_dt_seqlen stride_dt_head
            chunk_size BLOCK_SIZE_H idx))) ∧
    (ComputeCorrect.Realizes
      (kernel := chunked_cumsum_dA_cs_compute_slice DtPrepared A DACumsum
        stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head
        stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head
        stride_dA_cs_csize nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK)
        (fun idx => (DACumsum,
          dACsOutOffset s stride_dA_cs_batch stride_dA_cs_chunk
            stride_dA_cs_head stride_dA_cs_csize BLOCK_SIZE_H idx)))
      (expected := fun idx =>
        dAClosed s DtPrepared A stride_dt_batch stride_dt_seqlen
          stride_dt_head stride_A_head nheads chunk_size BLOCK_SIZE_H
          BLOCK_SIZE_CHUNK idx)))
```

**Assumptions / layout contracts:**
- `hDtOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE_H, BLOCK_SIZE_CHUNK] =>
        dtOutOffset s stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head
          stride_dt_out_csize BLOCK_SIZE_H idx)`
- `hDACsInj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE_H, BLOCK_SIZE_CHUNK] =>
        dACsOutOffset s stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head
          stride_dA_cs_csize BLOCK_SIZE_H idx)`

**Closed-form spec defs (transitive):** `dtOutOffset`, `dACsOutOffset`, `chunked_cumsum_fwd_surface`, `chunked_cumsum_dt_out_store_slice`, `active`, `dtPreparedOffset`, `chunked_cumsum_dA_cs_store_slice`, `chunked_cumsum_dA_cs_compute_slice`, `dAClosed`, `headIndex`, `chunkIndex`

<details><summary><code>dtOutOffset</code></summary>

```lean
def dtOutOffset
    (s : BlockState)
    (stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize
      BLOCK_SIZE_H : Nat)
    (idx : TileIndex [BLOCK_SIZE_H, BLOCK_SIZE_CHUNK]) : Nat :=
  s.pids 0 * stride_dt_out_batch + s.pids 1 * stride_dt_out_chunk +
    headIndex s BLOCK_SIZE_H idx.1 * stride_dt_out_head +
    chunkIndex s idx.2.1 * stride_dt_out_csize
```
</details>

<details><summary><code>dACsOutOffset</code></summary>

```lean
def dACsOutOffset
    (s : BlockState)
    (stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize
      BLOCK_SIZE_H : Nat)
    (idx : TileIndex [BLOCK_SIZE_H, BLOCK_SIZE_CHUNK]) : Nat :=
  s.pids 0 * stride_dA_cs_batch + s.pids 1 * stride_dA_cs_chunk +
    headIndex s BLOCK_SIZE_H idx.1 * stride_dA_cs_head +
    chunkIndex s idx.2.1 * stride_dA_cs_csize
```
</details>

<details><summary><code>chunked_cumsum_fwd_surface</code></summary>

```
/-- Faithful transcription of `chunked_cumsum_fwd.py`'s
`_chunk_cumsum_fwd_kernel`.

This preserves the optional `dt_bias` and softplus paths, the clamp via
`tl.minimum(tl.maximum(...))`, the masked `dt_out` store, and the `dA_cumsum`
store computed with `tl.cumsum` along the chunk axis. -/
```
```lean
def chunked_cumsum_fwd_surface
    (dt_ptr A_ptr dt_bias_ptr dt_out_ptr dA_cumsum_ptr : RegionName)
    (batch seqlen nheads chunk_size : Nat)
    (dt_min dt_max : ℝ)
    (stride_dt_batch stride_dt_seqlen stride_dt_head
      stride_A_head
      stride_dt_bias_head
      stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize
      stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize : Nat)
    (DT_SOFTPLUS HAS_DT_BIAS : Bool)
    (BLOCK_SIZE_H BLOCK_SIZE_CHUNK : Nat) :
    ComputeKernel := triton {
  pid_b = tl.program_id(axis=0)
  pid_c = tl.program_id(axis=1)
  pid_h = tl.program_id(axis=2)
  dt_ptr += pid_b * $(stride_dt_batch) +
    pid_c * $(chunk_size) * $(stride_dt_seqlen)
  dt_out_ptr += pid_b * $(stride_dt_out_batch) +
    pid_c * $(stride_dt_out_chunk)
  dA_cumsum_ptr += pid_b * $(stride_dA_cs_batch) +
    pid_c * $(stride_dA_cs_chunk)
  offs_h = pid_h * $(BLOCK_SIZE_H) + tl.arange(0, $(BLOCK_SIZE_H))
  offs_c = tl.arange(0, $(BLOCK_SIZE_CHUNK))
  dt_ptrs = dt_ptr + offs_h[:, None] * $(stride_dt_head) +
    offs_c[None, :] * $(stride_dt_seqlen)
  A_ptrs = A_ptr + offs_h * $(stride_A_head)
  dt_out_ptrs = dt_out_ptr + offs_h[:, None] * $(stride_dt_out_head) +
    offs_c[None, :] * $(stride_dt_out_csize)
  dA_cs_ptrs = dA_cumsum_ptr + offs_h[:, None] * $(stride_dA_cs_head) +
    offs_c[None, :] * $(stride_dA_cs_csize)
  chunk_size_limit = min($(chunk_size), $(seqlen) - pid_c * $(chunk_size))
  dt = tl.load(dt_ptrs, mask=(offs_h[:, None] < $(nheads)) &
    (offs_c[None, :] < chunk_size_limit), other=0.0).to(tl.float32)
  if HAS_DT_BIAS {
    dt_bias = tl.load(dt_bias_ptr + offs_h * $(stride_dt_bias_head),
      mask=offs_h < $(nheads), other=0.0).to(tl.float32)
    dt += dt_bias[:, None]
  }
  if DT_SOFTPLUS {
    dt = tl.where(dt <= 20.0, tl.log(1.0 + tl.exp(dt)), dt)
  }
  dt = tl.minimum(tl.maximum(dt, $(dt_min)), $(dt_max))
  dt = tl.where((offs_h[:, None] < $(nheads)) & (offs_c[None, :] < chunk_size_limit),
    dt, 0.0)
  tl.store(dt_out_ptrs, dt, mask=(offs_h[:, None] < $(nheads)) &
    (offs_c[None, :] < $(chunk_size)))
  A = tl.load(A_ptrs, mask=offs_h < $(nheads), other=0.0).to(tl.float32)
  dA = dt * A[:, None]
  dA_cs = tl.cumsum(dA, axis=1)
  tl.store(dA_cs_ptrs, dA_cs, mask=(offs_h[:, None] < $(nheads)) &
    (offs_c[None, :] < $(chunk_size)))
}
```
</details>

<details><summary><code>chunked_cumsum_dt_out_store_slice</code></summary>

```
/-- Proof-oriented `dt_out` writeback slice of `chunked_cumsum_fwd.py`'s
`_chunk_cumsum_fwd_kernel`.

The full kernel loads `dt`, optionally adds bias/softplus, clamps it, then
stores `dt_out` and computes/stores `dA_cumsum`. This slice starts from a
preprocessed `DtPrepared` tile and proves the masked `dt_out` writeback using
the original head/chunk indexing and output strides. -/
```
```lean
def chunked_cumsum_dt_out_store_slice
    (DtPrepared DtOut : RegionName)
    (stride_dt_batch stride_dt_seqlen stride_dt_head
      stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize
      nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK : Nat) :
    ComputeKernel := triton {
  pid_b = tl.program_id(axis=0)
  pid_c = tl.program_id(axis=1)
  pid_h = tl.program_id(axis=2)
  offs_h = pid_h * $(BLOCK_SIZE_H) + tl.arange(0, $(BLOCK_SIZE_H))
  offs_c = tl.arange(0, $(BLOCK_SIZE_CHUNK))
  mask = (offs_h[:, None] < $(nheads)) & (offs_c[None, :] < $(chunk_size))
  dt = tl.load(DtPrepared + pid_b * $(stride_dt_batch) +
      (pid_c * $(chunk_size) + offs_c[None, :]) * $(stride_dt_seqlen) +
      offs_h[:, None] * $(stride_dt_head),
    mask=mask, other=0.0)
  tl.store(DtOut + pid_b * $(stride_dt_out_batch) + pid_c * $(stride_dt_out_chunk) +
      offs_h[:, None] * $(stride_dt_out_head) + offs_c[None, :] * $(stride_dt_out_csize),
    dt, mask=mask)
}
```
</details>

<details><summary><code>active</code></summary>

```lean
def active
    (s : BlockState) (nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK : Nat)
    (idx : TileIndex [BLOCK_SIZE_H, BLOCK_SIZE_CHUNK]) : Prop :=
  headIndex s BLOCK_SIZE_H idx.1 < nheads ∧ chunkIndex s idx.2.1 < chunk_size
```
</details>

<details><summary><code>dtPreparedOffset</code></summary>

```lean
def dtPreparedOffset
    (s : BlockState)
    (stride_dt_batch stride_dt_seqlen stride_dt_head chunk_size BLOCK_SIZE_H : Nat)
    (idx : TileIndex [BLOCK_SIZE_H, BLOCK_SIZE_CHUNK]) : Nat :=
  s.pids 0 * stride_dt_batch +
    (s.pids 1 * chunk_size + chunkIndex s idx.2.1) * stride_dt_seqlen +
    headIndex s BLOCK_SIZE_H idx.1 * stride_dt_head
```
</details>

<details><summary><code>chunked_cumsum_dA_cs_store_slice</code></summary>

```
/-- Proof-oriented `dA_cumsum` writeback slice of `chunked_cumsum_fwd.py`'s
`_chunk_cumsum_fwd_kernel`.

The full kernel produces a per-head running cumsum tile `dA_cs = cumsum(dt * A,
axis=1)`. This slice starts from a precomputed `DAcs` tile and proves the
masked writeback into `dA_cumsum_ptr` using the original head/chunk indexing
and output strides. -/
```
```lean
def chunked_cumsum_dA_cs_store_slice
    (DAcs DACumsum : RegionName)
    (stride_dt_batch stride_dt_seqlen stride_dt_head
      stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize
      nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK : Nat) :
    ComputeKernel := triton {
  pid_b = tl.program_id(axis=0)
  pid_c = tl.program_id(axis=1)
  pid_h = tl.program_id(axis=2)
  offs_h = pid_h * $(BLOCK_SIZE_H) + tl.arange(0, $(BLOCK_SIZE_H))
  offs_c = tl.arange(0, $(BLOCK_SIZE_CHUNK))
  mask = (offs_h[:, None] < $(nheads)) & (offs_c[None, :] < $(chunk_size))
  dA_cs = tl.load(DAcs + pid_b * $(stride_dt_batch) +
      (pid_c * $(chunk_size) + offs_c[None, :]) * $(stride_dt_seqlen) +
      offs_h[:, None] * $(stride_dt_head),
    mask=mask, other=0.0)
  tl.store(DACumsum + pid_b * $(stride_dA_cs_batch) + pid_c * $(stride_dA_cs_chunk) +
      offs_h[:, None] * $(stride_dA_cs_head) + offs_c[None, :] * $(stride_dA_cs_csize),
    dA_cs, mask=mask)
}
```
</details>

<details><summary><code>chunked_cumsum_dA_cs_compute_slice</code></summary>

```
/-! ## Computed `dA_cumsum` slice

The store slice above assumes the `dA_cs` tile has already been prepared. The
next slice covers the Python computation immediately before that store:
`A = load(A_ptrs)`, `dA = dt * A[:, None]`, and
`dA_cs = tl.cumsum(dA, axis=1)`. -/
```
```lean
def chunked_cumsum_dA_cs_compute_slice
    (DtPrepared A DACumsum : RegionName)
    (stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head
      stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize
      nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK : Nat) :
    ComputeKernel := triton {
  pid_b = tl.program_id(axis=0)
  pid_c = tl.program_id(axis=1)
  pid_h = tl.program_id(axis=2)
  offs_h = pid_h * $(BLOCK_SIZE_H) + tl.arange(0, $(BLOCK_SIZE_H))
  offs_c = tl.arange(0, $(BLOCK_SIZE_CHUNK))
  mask = (offs_h[:, None] < $(nheads)) & (offs_c[None, :] < $(chunk_size))
  dt = tl.load(DtPrepared + pid_b * $(stride_dt_batch) +
      (pid_c * $(chunk_size) + offs_c[None, :]) * $(stride_dt_seqlen) +
      offs_h[:, None] * $(stride_dt_head),
    mask=mask, other=0.0)
  a = tl.load(A + offs_h * $(stride_A_head),
    mask=offs_h < $(nheads), other=0.0)
  dA = dt * a[:, None]
  dA_cs = tl.cumsum(dA, axis=1)
  tl.store(DACumsum + pid_b * $(stride_dA_cs_batch) + pid_c * $(stride_dA_cs_chunk) +
      offs_h[:, None] * $(stride_dA_cs_head) + offs_c[None, :] * $(stride_dA_cs_csize),
    dA_cs, mask=mask)
}
```
</details>

<details><summary><code>dAClosed</code></summary>

```
/-! ## Genuine within-chunk closed form for `dA_cumsum`

`dAClosed` is the genuine mathematical specification of the `dA_cumsum` output:
a `Finset.sum` over all *active* chunk positions `k ≤ idx.chunk` (with the head
in range and `k < chunk_size`) of the prepared `dt` value at `(head, k)` times
the per-head scaling `A[head]`. It is **not** a read-back of the kernel's own
output, so realizing it is a true correctness statement. There is no cross-chunk
carry: each chunk is summed independently along its own axis. -/
```
```lean
noncomputable def dAClosed
    (s : BlockState) (DtPrepared A : RegionName)
    (stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head
      nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK : Nat)
    (idx : TileIndex [BLOCK_SIZE_H, BLOCK_SIZE_CHUNK]) : ℝ :=
  ∑ k ∈ (Finset.univ.filter
      (fun k : Fin BLOCK_SIZE_CHUNK =>
        k.val ≤ idx.2.1.val ∧
          (headIndex s BLOCK_SIZE_H idx.1 < nheads ∧ k.val < chunk_size))),
    s.readMem DtPrepared
        (s.pids 0 * stride_dt_batch +
          (s.pids 1 * chunk_size + k.val) * stride_dt_seqlen +
          headIndex s BLOCK_SIZE_H idx.1 * stride_dt_head)
      * s.readMem A (headIndex s BLOCK_SIZE_H idx.1 * stride_A_head)
```
</details>

<details><summary><code>headIndex</code></summary>

```lean
def headIndex (s : BlockState) (BLOCK_SIZE_H : Nat) (i : Fin BLOCK_SIZE_H) : Nat :=
  s.pids 2 * BLOCK_SIZE_H + i.val
```
</details>

<details><summary><code>chunkIndex</code></summary>

```lean
def chunkIndex (_s : BlockState) (j : Fin BLOCK_SIZE_CHUNK) : Nat :=
  j.val
```
</details>

## Also present (pinned special-case summaries)
- `chunked_cumsum_dt_out_store_slice_compute_correct`
- `chunked_cumsum_dA_cs_store_slice_compute_correct`
- `chunked_cumsum_dA_cs_compute_slice_compute_correct`
- `chunked_cumsum_fwd_all_outputs_compute_correct_general`
