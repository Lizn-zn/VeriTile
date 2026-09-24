# Spec sheet — `bench/tritonbench_g/chunked_cumsum_fwd/ChunkedCumsumFwd.lean`

**Python source:** `bench/tritonbench_g/chunked_cumsum_fwd/chunked_cumsum_fwd.py`

## Public theorem: `chunked_cumsum_fwd_correctness`

<details><summary>docstring</summary>

```
/-- **The three-pid `⊨` headline for chunked cumsum forward.** On its masked
two-input / two-output IO signature over the 3-D launch grid, the fused kernel
implements `dt_out[j] = xs j` (the prepared `dt`) and `dA_cumsum[j] =
dACumsumLaneClosed …` — the standalone within-chunk prefix `Finset.sum`
`Σ_{k ≤ col, k < chunk_size} xs[(row, k)] · ys[j]` — **never** a read-back of the
kernel's own output. Fully general over all dimensions and strides and the two
program ids the spec sees (the third enters through the windows/masks). Side
conditions, all truth-forced: `DtOut ≠ DACumsum` (the two masked block stores
must not alias, so each output's readback sees through the other), and per-pid
output-offset injectivity (arbitrary strides may otherwise collide two lanes on
one cell — the same host-side address-layout condition carried by the slice
decomposition). No `0 < _` positivity is needed: the within-chunk cumsum is a
total `Finset.sum` and the `Fin (BLOCK_SIZE_H · BLOCK_SIZE_CHUNK)` lanes supply
their own `0 < BLOCK_SIZE_CHUNK` for the row-major decode. -/
```
</details>

**Statement:**
```lean
specification chunked_cumsum_fwd_correctness
    (DtPrepared A DtOut DACumsum : RegionName)
    (stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head
      stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize
      stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize
      nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK : Nat)
    (hDtNe : DtOut ≠ DACumsum) (hCK : 0 < BLOCK_SIZE_CHUNK)
    (hDtOutInj : ∀ p₀ p₁ p₂ : Nat, Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE_H, BLOCK_SIZE_CHUNK] =>
        p₀ * stride_dt_out_batch + p₁ * stride_dt_out_chunk +
          (p₂ * BLOCK_SIZE_H + idx.1.val) * stride_dt_out_head +
          idx.2.1.val * stride_dt_out_csize))
    (hDACsInj : ∀ p₀ p₁ p₂ : Nat, Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE_H, BLOCK_SIZE_CHUNK] =>
        p₀ * stride_dA_cs_batch + p₁ * stride_dA_cs_chunk +
          (p₂ * BLOCK_SIZE_H + idx.1.val) * stride_dA_cs_head +
          idx.2.1.val * stride_dA_cs_csize)) :
    chunkedCumsumFwdIO DtPrepared A DtOut DACumsum
        stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head
        stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize
        stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize
        nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK ⊨
      fun _ _ xs ys =>
        (fun j => xs j,
         fun j => dACumsumLaneClosed chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK xs ys j)
```

**Assumptions / layout contracts:**
- `hDtNe : DtOut ≠ DACumsum`
- `hCK : 0 < BLOCK_SIZE_CHUNK`

**Closed-form spec defs (transitive):** `chunkedCumsumFwdIO`, `dACumsumLaneClosed`, `chunked_cumsum_fused_slice`

<details><summary><code>chunkedCumsumFwdIO</code></summary>

```
/-- The fused chunked-cumsum kernel's masked two-input / two-output **IO
signature** over the **three-program-id** grid `(pid_b, pid_c, pid_h)` — the
first consumer of `Masked3DKernelIO₂ₓ₂`:

* `in1`/`in2`/`out1`/`out2` — the prepared `dt` tile `DtPrepared`, the per-head
  scale `A`, the `dt_out` buffer, and the `dA_cumsum` buffer;
* `B = BLOCK_SIZE_H · BLOCK_SIZE_CHUNK` — the `[head, chunk]` tile each program
  owns, indexed by the flat **row-major lane** `j ↦ (j / BLOCK_SIZE_CHUNK,
  j % BLOCK_SIZE_CHUNK)`;
* `read1`/`write1`/`write2` — the general-stride `(pid_b, pid_c, pid_h)`
  addresses: `pid_h` enters the head coordinate `pid_h · BLOCK_SIZE_H + row`,
  `pid_b`/`pid_c` the batch/chunk offsets — genuinely using the third program
  id;
* `read2` — the per-head scale address `(pid_h · BLOCK_SIZE_H + row) ·
  stride_A_head`, constant across a head's columns;
* `mask` — the active lanes `pid_h · BLOCK_SIZE_H + row < nheads ∧ col <
  chunk_size`; `read2Mask` is the head-only gate. -/
```
```lean
def chunkedCumsumFwdIO (DtPrepared A DtOut DACumsum : RegionName)
    (stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head
      stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize
      stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize
      nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK : Nat) :
    Masked3DKernelIO₂ₓ₂ where
  kernel := chunked_cumsum_fused_slice DtPrepared A DtOut DACumsum
    stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head
    stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize
    stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize
    nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK
  in1 := DtPrepared
  in2 := A
  out1 := DtOut
  out2 := DACumsum
  B := BLOCK_SIZE_H * BLOCK_SIZE_CHUNK
  read1 := fun p₀ p₁ p₂ j =>
    p₀ * stride_dt_batch +
      (p₁ * chunk_size + j.val % BLOCK_SIZE_CHUNK) * stride_dt_seqlen +
      (p₂ * BLOCK_SIZE_H + j.val / BLOCK_SIZE_CHUNK) * stride_dt_head
  read2 := fun _ _ p₂ j =>
    (p₂ * BLOCK_SIZE_H + j.val / BLOCK_SIZE_CHUNK) * stride_A_head
  write1 := fun p₀ p₁ p₂ j =>
    p₀ * stride_dt_out_batch + p₁ * stride_dt_out_chunk +
      (p₂ * BLOCK_SIZE_H + j.val / BLOCK_SIZE_CHUNK) * stride_dt_out_head +
      j.val % BLOCK_SIZE_CHUNK * stride_dt_out_csize
  write2 := fun p₀ p₁ p₂ j =>
    p₀ * stride_dA_cs_batch + p₁ * stride_dA_cs_chunk +
      (p₂ * BLOCK_SIZE_H + j.val / BLOCK_SIZE_CHUNK) * stride_dA_cs_head +
      j.val % BLOCK_SIZE_CHUNK * stride_dA_cs_csize
  mask := fun _ _ p₂ j =>
    p₂ * BLOCK_SIZE_H + j.val / BLOCK_SIZE_CHUNK < nheads ∧
      j.val % BLOCK_SIZE_CHUNK < chunk_size
  read2Mask := fun _ _ p₂ j => p₂ * BLOCK_SIZE_H + j.val / BLOCK_SIZE_CHUNK < nheads
```
</details>

<details><summary><code>dACumsumLaneClosed</code></summary>

```
/-- The genuine `dA_cumsum` closed form as a **pure function of the pinned
inputs**, indexed by the flat row-major lane `j` (`Fin (BLOCK_SIZE_H ·
BLOCK_SIZE_CHUNK)`). At lane `j` (row `j / C`, column `j % C`) it is the
within-chunk prefix sum `Σ_{k ≤ col, k < chunk_size} xs[(row, k)] · ys[j]` — the
`⊨`-form of `dAClosed`, reading only the pinned `DtPrepared`/`A` tiles. -/
```
```lean
noncomputable def dACumsumLaneClosed (chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK : Nat)
    (xs ys : Fin (BLOCK_SIZE_H * BLOCK_SIZE_CHUNK) → ℝ)
    (j : Fin (BLOCK_SIZE_H * BLOCK_SIZE_CHUNK)) : ℝ :=
  ∑ k ∈ (Finset.univ.filter
      (fun k : Fin BLOCK_SIZE_CHUNK =>
        k.val ≤ j.val % BLOCK_SIZE_CHUNK ∧ k.val < chunk_size)),
    xs (Lane2D.encode ((Lane2D.decode j).1, k, PUnit.unit)) * ys j
```
</details>

<details><summary><code>chunked_cumsum_fused_slice</code></summary>

```
/-- The fused proof-oriented slice: loads prepared `dt` and per-head `A`, stores
`dt_out`, then stores `dA_cumsum = cumsum(dt · A, axis=1)`. The two loads
precede both stores, so each load reads the launch memory directly. -/
```
```lean
def chunked_cumsum_fused_slice
    (DtPrepared A DtOut DACumsum : RegionName)
    (stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head
      stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize
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
  tl.store(DtOut + pid_b * $(stride_dt_out_batch) + pid_c * $(stride_dt_out_chunk) +
      offs_h[:, None] * $(stride_dt_out_head) + offs_c[None, :] * $(stride_dt_out_csize),
    dt, mask=mask)
  dA = dt * a[:, None]
  dA_cs = tl.cumsum(dA, axis=1)
  tl.store(DACumsum + pid_b * $(stride_dA_cs_batch) + pid_c * $(stride_dA_cs_chunk) +
      offs_h[:, None] * $(stride_dA_cs_head) + offs_c[None, :] * $(stride_dA_cs_csize),
    dA_cs, mask=mask)
}
```
</details>

## Public theorem: `chunked_cumsum_fwd_summary_general`

<details><summary>docstring</summary>

```
/-- **Dimension-general genuine correctness summary for chunked cumsum forward.**
The full Python surface (with the `HAS_DT_BIAS` / `DT_SOFTPLUS` flags) lowers to
the algorithm layer for arbitrary dimensions and strides, and — given the
truth-forced side conditions — the fused verified kernel satisfies the genuine
three-pid `⊨` contract (`dt_out` = prepared `dt`, `dA_cumsum` = the standalone
within-chunk prefix `Finset.sum`). -/
```
</details>

**Statement:**
```lean
specification chunked_cumsum_fwd_summary_general
    (dt_ptr A_ptr dt_bias_ptr dt_out_ptr dA_cumsum_ptr
      DtPrepared A DtOut DACumsum : RegionName)
    (batch seqlen nheads chunk_size : Nat)
    (dt_min dt_max : ℝ)
    (stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head
      stride_dt_bias_head
      stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize
      stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize : Nat)
    (DT_SOFTPLUS HAS_DT_BIAS : Bool)
    (BLOCK_SIZE_H BLOCK_SIZE_CHUNK : Nat)
    (hDtNe : DtOut ≠ DACumsum) (hCK : 0 < BLOCK_SIZE_CHUNK)
    (hDtOutInj : ∀ p₀ p₁ p₂ : Nat, Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE_H, BLOCK_SIZE_CHUNK] =>
        p₀ * stride_dt_out_batch + p₁ * stride_dt_out_chunk +
          (p₂ * BLOCK_SIZE_H + idx.1.val) * stride_dt_out_head +
          idx.2.1.val * stride_dt_out_csize))
    (hDACsInj : ∀ p₀ p₁ p₂ : Nat, Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE_H, BLOCK_SIZE_CHUNK] =>
        p₀ * stride_dA_cs_batch + p₁ * stride_dA_cs_chunk +
          (p₂ * BLOCK_SIZE_H + idx.1.val) * stride_dA_cs_head +
          idx.2.1.val * stride_dA_cs_csize)) :
    (∃ alg, (chunked_cumsum_fwd_surface dt_ptr A_ptr dt_bias_ptr dt_out_ptr
      dA_cumsum_ptr batch seqlen nheads chunk_size dt_min dt_max stride_dt_batch
      stride_dt_seqlen stride_dt_head stride_A_head stride_dt_bias_head
      stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize
      stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize
      DT_SOFTPLUS HAS_DT_BIAS BLOCK_SIZE_H BLOCK_SIZE_CHUNK).toAlgorithm?
        = Except.ok alg) ∧
    (chunkedCumsumFwdIO DtPrepared A DtOut DACumsum
        stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head
        stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize
        stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize
        nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK ⊨
      fun _ _ xs ys =>
        (fun j => xs j,
         fun j => dACumsumLaneClosed chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK xs ys j))
```

**Assumptions / layout contracts:**
- `hDtNe : DtOut ≠ DACumsum`
- `hCK : 0 < BLOCK_SIZE_CHUNK`

**Closed-form spec defs (transitive):** `chunked_cumsum_fwd_surface`, `chunkedCumsumFwdIO`, `dACumsumLaneClosed`, `chunked_cumsum_fused_slice`

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

<details><summary><code>chunkedCumsumFwdIO</code></summary>

```
/-- The fused chunked-cumsum kernel's masked two-input / two-output **IO
signature** over the **three-program-id** grid `(pid_b, pid_c, pid_h)` — the
first consumer of `Masked3DKernelIO₂ₓ₂`:

* `in1`/`in2`/`out1`/`out2` — the prepared `dt` tile `DtPrepared`, the per-head
  scale `A`, the `dt_out` buffer, and the `dA_cumsum` buffer;
* `B = BLOCK_SIZE_H · BLOCK_SIZE_CHUNK` — the `[head, chunk]` tile each program
  owns, indexed by the flat **row-major lane** `j ↦ (j / BLOCK_SIZE_CHUNK,
  j % BLOCK_SIZE_CHUNK)`;
* `read1`/`write1`/`write2` — the general-stride `(pid_b, pid_c, pid_h)`
  addresses: `pid_h` enters the head coordinate `pid_h · BLOCK_SIZE_H + row`,
  `pid_b`/`pid_c` the batch/chunk offsets — genuinely using the third program
  id;
* `read2` — the per-head scale address `(pid_h · BLOCK_SIZE_H + row) ·
  stride_A_head`, constant across a head's columns;
* `mask` — the active lanes `pid_h · BLOCK_SIZE_H + row < nheads ∧ col <
  chunk_size`; `read2Mask` is the head-only gate. -/
```
```lean
def chunkedCumsumFwdIO (DtPrepared A DtOut DACumsum : RegionName)
    (stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head
      stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize
      stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize
      nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK : Nat) :
    Masked3DKernelIO₂ₓ₂ where
  kernel := chunked_cumsum_fused_slice DtPrepared A DtOut DACumsum
    stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head
    stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize
    stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize
    nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK
  in1 := DtPrepared
  in2 := A
  out1 := DtOut
  out2 := DACumsum
  B := BLOCK_SIZE_H * BLOCK_SIZE_CHUNK
  read1 := fun p₀ p₁ p₂ j =>
    p₀ * stride_dt_batch +
      (p₁ * chunk_size + j.val % BLOCK_SIZE_CHUNK) * stride_dt_seqlen +
      (p₂ * BLOCK_SIZE_H + j.val / BLOCK_SIZE_CHUNK) * stride_dt_head
  read2 := fun _ _ p₂ j =>
    (p₂ * BLOCK_SIZE_H + j.val / BLOCK_SIZE_CHUNK) * stride_A_head
  write1 := fun p₀ p₁ p₂ j =>
    p₀ * stride_dt_out_batch + p₁ * stride_dt_out_chunk +
      (p₂ * BLOCK_SIZE_H + j.val / BLOCK_SIZE_CHUNK) * stride_dt_out_head +
      j.val % BLOCK_SIZE_CHUNK * stride_dt_out_csize
  write2 := fun p₀ p₁ p₂ j =>
    p₀ * stride_dA_cs_batch + p₁ * stride_dA_cs_chunk +
      (p₂ * BLOCK_SIZE_H + j.val / BLOCK_SIZE_CHUNK) * stride_dA_cs_head +
      j.val % BLOCK_SIZE_CHUNK * stride_dA_cs_csize
  mask := fun _ _ p₂ j =>
    p₂ * BLOCK_SIZE_H + j.val / BLOCK_SIZE_CHUNK < nheads ∧
      j.val % BLOCK_SIZE_CHUNK < chunk_size
  read2Mask := fun _ _ p₂ j => p₂ * BLOCK_SIZE_H + j.val / BLOCK_SIZE_CHUNK < nheads
```
</details>

<details><summary><code>dACumsumLaneClosed</code></summary>

```
/-- The genuine `dA_cumsum` closed form as a **pure function of the pinned
inputs**, indexed by the flat row-major lane `j` (`Fin (BLOCK_SIZE_H ·
BLOCK_SIZE_CHUNK)`). At lane `j` (row `j / C`, column `j % C`) it is the
within-chunk prefix sum `Σ_{k ≤ col, k < chunk_size} xs[(row, k)] · ys[j]` — the
`⊨`-form of `dAClosed`, reading only the pinned `DtPrepared`/`A` tiles. -/
```
```lean
noncomputable def dACumsumLaneClosed (chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK : Nat)
    (xs ys : Fin (BLOCK_SIZE_H * BLOCK_SIZE_CHUNK) → ℝ)
    (j : Fin (BLOCK_SIZE_H * BLOCK_SIZE_CHUNK)) : ℝ :=
  ∑ k ∈ (Finset.univ.filter
      (fun k : Fin BLOCK_SIZE_CHUNK =>
        k.val ≤ j.val % BLOCK_SIZE_CHUNK ∧ k.val < chunk_size)),
    xs (Lane2D.encode ((Lane2D.decode j).1, k, PUnit.unit)) * ys j
```
</details>

<details><summary><code>chunked_cumsum_fused_slice</code></summary>

```
/-- The fused proof-oriented slice: loads prepared `dt` and per-head `A`, stores
`dt_out`, then stores `dA_cumsum = cumsum(dt · A, axis=1)`. The two loads
precede both stores, so each load reads the launch memory directly. -/
```
```lean
def chunked_cumsum_fused_slice
    (DtPrepared A DtOut DACumsum : RegionName)
    (stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head
      stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize
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
  tl.store(DtOut + pid_b * $(stride_dt_out_batch) + pid_c * $(stride_dt_out_chunk) +
      offs_h[:, None] * $(stride_dt_out_head) + offs_c[None, :] * $(stride_dt_out_csize),
    dt, mask=mask)
  dA = dt * a[:, None]
  dA_cs = tl.cumsum(dA, axis=1)
  tl.store(DACumsum + pid_b * $(stride_dA_cs_batch) + pid_c * $(stride_dA_cs_chunk) +
      offs_h[:, None] * $(stride_dA_cs_head) + offs_c[None, :] * $(stride_dA_cs_csize),
    dA_cs, mask=mask)
}
```
</details>

## Also present (pinned special-case summaries)
- `chunked_cumsum_dt_out_store_slice_compute_correct`
- `chunked_cumsum_dA_cs_store_slice_compute_correct`
- `chunked_cumsum_dA_cs_compute_slice_compute_correct`
- `chunked_cumsum_fwd_all_outputs_compute_correct_general`
