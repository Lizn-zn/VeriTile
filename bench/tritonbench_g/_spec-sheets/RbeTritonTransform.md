# Spec sheet — `bench/tritonbench_g/rbe_triton_transform/RbeTritonTransform.lean`

**Python source:** `bench/tritonbench_g/rbe_triton_transform/rbe_triton_transform.py`

## Public theorem: `rbe_triton_transform_output_summary_general`

<details><summary>docstring</summary>

```
/-- **Dimension-general** correctness summary for `rbe_triton_transform.py`'s
`rbe_triton`, against the **genuine rotary closed form** — a pure function of
INPUT memory and the exact `Real.cos` / `Real.sin` / `Real.rpow`, never a
read-back of the kernel's own output — for arbitrary `M`, `K`, all six
strides, `start_token_position`, `THETA`, `DIM`, `BLOCK_SIZE_M`,
`BLOCK_SIZE_K`, and arbitrary program ids (`s.pids 0` = batch,
`s.pids 1` = the fused `(m, k)` CTA id decomposed in-kernel). It packages:

* the full faithful surface (both `tl.debug_barrier()` calls, both masked
  loads/stores, the inlined `get_freq_multi_tokens`) lowers to the algorithm
  layer;
* the even-offset (`out_real`) store: every `out_real_mask`-active lane
  `(i, j)` holds `x_real·cos(freq) − x_imag·sin(freq)` where
  `freq = (start_token_position + offs_m[i]) / THETA^((offs_n[j] % DIM)/DIM)`
  (`freqSpec_eq_token`) and `x_imag` is the masked imaginary load (`0` on the
  `1 + offs_n[j] ≥ K` boundary lane);
* the odd-offset (`out_imag`) store: every `out_imag_mask`-active lane holds
  `x_real·sin(freq) + x_imag·cos(freq)`.

Honest side conditions: the even-offset output footprint is injective
(`hOutInj`) and even-offset cells never collide with odd-offset (`+ 1`) cells
(`hRI`) — both hold for the wrapper's contiguous row-major layout. -/
```
</details>

**Statement:**
```lean
theorem rbe_triton_transform_output_summary_general
    (x_ptr out_ptr : RegionName)
    (M K stride_x_batch stride_x_m stride_x_n
      stride_out_batch stride_out_m stride_out_n
      start_token_position : Nat)
    (THETA : ℝ) (DIM BLOCK_SIZE_M BLOCK_SIZE_K : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_K / 2] =>
        outOff s K stride_out_batch stride_out_m stride_out_n BLOCK_SIZE_M
          BLOCK_SIZE_K idx))
    (hRI : ∀ idx k : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_K / 2],
      outOff s K stride_out_batch stride_out_m stride_out_n BLOCK_SIZE_M
          BLOCK_SIZE_K idx
        ≠ outOff s K stride_out_batch stride_out_m stride_out_n BLOCK_SIZE_M
            BLOCK_SIZE_K k + 1) :
    -- (1) the full faithful surface lowers to the algorithm layer
    (∃ alg, (rbe_triton_surface x_ptr out_ptr M K stride_x_batch stride_x_m
      stride_x_n stride_out_batch stride_out_m stride_out_n
      start_token_position THETA DIM BLOCK_SIZE_M BLOCK_SIZE_K).toAlgorithm?
        = Except.ok alg) ∧
    -- (2) even offsets: genuine `x_real·cos − x_imag·sin`
    ComputeCorrect.Realizes_without_Rounding
      (kernel
```

**Assumptions / layout contracts:**
- `hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_K / 2] =>
        outOff s K stride_out_batch stride_out_m stride_out_n BLOCK_SIZE_M
          BLOCK_SIZE_K idx)`
- `hRI : ∀ idx k : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_K / 2],
      outOff s K stride_out_batch stride_out_m stride_out_n BLOCK_SIZE_M
          BLOCK_SIZE_K idx
        ≠ outOff s K stride_out_batch stride_out_m stride_out_n BLOCK_SIZE_M
            BLOCK_SIZE_K k + 1`

**Closed-form spec defs (transitive):** `outOff`, `rbe_triton_surface`, `rowIdx`, `colIdx`, `pidM`, `pidN`, `kCdiv`

<details><summary><code>outOff</code></summary>

```
/-- Output address of the real part written by lane `(i, j)` (the imaginary
part is stored at `+ 1`). -/
```
```lean
def outOff (s : BlockState)
    (K stride_out_batch stride_out_m stride_out_n
      BLOCK_SIZE_M BLOCK_SIZE_K : Nat)
    (idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_K / 2]) : Nat :=
  s.pids 0 * stride_out_batch +
    stride_out_m * rowIdx s K BLOCK_SIZE_M BLOCK_SIZE_K idx.1 +
    stride_out_n * colIdx s K BLOCK_SIZE_K idx.2.1
```
</details>

<details><summary><code>rbe_triton_surface</code></summary>

```
/-- Faithful transcription of `rbe_triton_transform.py`'s `rbe_triton`, with
`get_freq_multi_tokens` inlined at its single call site (see the
Translation-surface blocker preamble for the inlining conventions). -/
```
```lean
def rbe_triton_surface
    (x_ptr out_ptr : RegionName)
    (M K stride_x_batch stride_x_m stride_x_n
      stride_out_batch stride_out_m stride_out_n
      start_token_position : Nat)
    (THETA : ℝ) (DIM BLOCK_SIZE_M BLOCK_SIZE_K : Nat) :
    ComputeKernel := triton {
  pid_batch = tl.program_id(axis=0)
  pid = tl.program_id(axis=1)
  pid_m = pid // tl.cdiv($(K), $(BLOCK_SIZE_K))
  pid_n = pid % tl.cdiv($(K), $(BLOCK_SIZE_K))

  offs_m = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_n = pid_n * $(BLOCK_SIZE_K) + tl.arange(0, $(BLOCK_SIZE_K) // $(2)) * $(2)
  x_ptrs = x_ptr + (pid_batch * $(stride_x_batch) + $(stride_x_m) * offs_m[:, None] +
    $(stride_x_n) * offs_n[None, :])
  x_real_mask = (offs_m[:, None] < $(M)) & (offs_n[None, :] < $(K))
  real = tl.load(x_ptrs, mask=x_real_mask, other=0.0)
  x_imag_mask = (offs_m[:, None] < $(M)) & ($((1 : Nat)) + offs_n[None, :] < $(K))
  imag = tl.load(x_ptrs + $(1), mask=x_imag_mask, other=0.0)
  tl.debug_barrier()
  start_block = $((start_token_position : Nat)) + pid_m * $(BLOCK_SIZE_M)
  freqs = offs_n % $(DIM)
  freqs_f = tl.toReal(freqs) / $(DIM)
  freqs_p = tl.extra.cuda.libdevice.pow($((THETA : ℝ)), freqs_f)
  tks = tl.arange(0, $(BLOCK_SIZE_M)) + start_block
  tks_f = tl.toReal(tks)
  freqs_mn = tks_f[:, None] / freqs_p[None, :]
  cos = tl.cos(freqs_mn)
  sin = tl.sin(freqs_mn)

  out_real = real * cos - imag * sin
  out_imag = real * sin + imag * cos
  tl.debug_barrier()
  out_ptrs = out_ptr + (pid_batch * $(stride_out_batch) + $(stride_out_m) * offs_m[:, None] +
    $(stride_out_n) * offs_n[None, :])
  out_real_mask = (offs_m[:, None] < $(M)) & (offs_n[None, :] < $(K))
  tl.store(out_ptrs, out_real, mask=out_real_mask)
  out_imag_mask = (offs_m[:, None] < $(M)) & ($((1 : Nat)) + offs_n[None, :] < $(K))
  tl.store(out_ptrs + $(1), out_imag, mask=out_imag_mask)
}
```
</details>

<details><summary><code>rowIdx</code></summary>

```
/-- Global row `offs_m[i] = pid_m·BLOCK_SIZE_M + i` covered by tile lane `i`. -/
```
```lean
def rowIdx (s : BlockState) (K BLOCK_SIZE_M BLOCK_SIZE_K : Nat)
    (i : Fin BLOCK_SIZE_M) : Nat :=
  pidM s K BLOCK_SIZE_K * BLOCK_SIZE_M + i.val
```
</details>

<details><summary><code>colIdx</code></summary>

```
/-- Global (even) column `offs_n[j] = pid_n·BLOCK_SIZE_K + 2j` covered by tile
lane `j`. -/
```
```lean
def colIdx (s : BlockState) (K BLOCK_SIZE_K : Nat)
    (j : Fin (BLOCK_SIZE_K / 2)) : Nat :=
  pidN s K BLOCK_SIZE_K * BLOCK_SIZE_K + j.val * 2
```
</details>

<details><summary><code>pidM</code></summary>

```
/-- `pid_m = pid // tl.cdiv(K, BLOCK_SIZE_K)` of this program. -/
```
```lean
def pidM (s : BlockState) (K BLOCK_SIZE_K : Nat) : Nat :=
  s.pids 1 / kCdiv K BLOCK_SIZE_K
```
</details>

<details><summary><code>pidN</code></summary>

```
/-- `pid_n = pid % tl.cdiv(K, BLOCK_SIZE_K)` of this program. -/
```
```lean
def pidN (s : BlockState) (K BLOCK_SIZE_K : Nat) : Nat :=
  s.pids 1 % kCdiv K BLOCK_SIZE_K
```
</details>

<details><summary><code>kCdiv</code></summary>

```
/-- `tl.cdiv(K, BLOCK_SIZE_K)` at the algorithm layer. -/
```
```lean
def kCdiv (K BLOCK_SIZE_K : Nat) : Nat := (K + BLOCK_SIZE_K - 1) / BLOCK_SIZE_K
```
</details>

## Also present (pinned special-case summaries)
- `rbe_real_compute_correct`
- `rbe_imag_compute_correct`
