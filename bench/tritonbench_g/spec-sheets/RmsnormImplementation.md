# Spec sheet — `bench/tritonbench_g/rmsnorm_implementation/RmsnormImplementation.lean`

**Python source:** `bench/tritonbench_g/rmsnorm_implementation/rmsnorm_implementation.py`

## Public theorem: `rmsnorm_implementation_output_summary`

<details><summary>docstring</summary>

```
/-- Per-kernel output summary for `rmsnorm_triton`: the DSL surface lowers to the
algorithm layer, and the masked store to `out_ptr` is compute-correct for
arbitrary `N_SIZE` — every output column holds the full-`N` RMSNorm spec
`rmsnormWeightedYFullNSpec`. Built on the multi-block `*_compute_fullN_correct`
result; requires `0 < BLOCK_N_SIZE`, `0 < stride_out_k`, and output/input
disjointness. -/
```
</details>

**Statement:**
```lean
theorem rmsnorm_implementation_output_summary
    (x_ptr rms_w_ptr out_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE : Nat)
    (eps : ℝ) (s : BlockState)
    (hBlockPos : 0 < BLOCK_N_SIZE)
    (hStrideOutKPos : 0 < stride_out_k)
    (hXOutNe : x_ptr ≠ out_ptr)
    (hWOutNe : rms_w_ptr ≠ out_ptr) :
    (∃ alg, (rmsnorm_implementation x_ptr rms_w_ptr out_ptr
        stride_x_batch stride_x_m stride_x_k stride_rms_w
        stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE
        eps).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := rmsnorm_implementation x_ptr rms_w_ptr out_ptr
        stride_x_batch stride_x_m stride_x_k stride_rms_w
        stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE eps)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin N_SIZE => True)
        (fun i => (out_ptr,
          outColOffset s stride_out_batch stride_out_m stride_out_k i.val)))
      (expected := fun i =>
        rmsnormWeightedYFullNSpec s x_ptr rms_w_ptr stride_x_batch stride_x_m
          stride_x_k stride_rms_w N_SIZE BLOCK_N_SIZE eps i)
```

**Assumptions / layout contracts:**
- `hBlockPos : 0 < BLOCK_N_SIZE`
- `hStrideOutKPos : 0 < stride_out_k`
- `hXOutNe : x_ptr ≠ out_ptr`
- `hWOutNe : rms_w_ptr ≠ out_ptr`
- `fun _ : Fin N_SIZE => True`

**Closed-form spec defs (transitive):** `rmsnorm_implementation`, `outColOffset`, `rmsnormWeightedYFullNSpec`, `rmsnormYFullNSpec`, `rmsInvVarFullN`, `rmsVarFullNCarrier`, `xColOffset`

<details><summary><code>rmsnorm_implementation</code></summary>

```
/-- Faithful transcription of `rmsnorm_implementation.py`'s `rmsnorm_triton`.

Allowed mechanical Lean-syntax-only changes:
- Python `N_SIZE: tl.constexpr` / `eps: tl.constexpr` / `BLOCK_N_SIZE: tl.constexpr`
  -> Lean `Nat` / `ℝ` parameters. -/
```
```lean
def rmsnorm_implementation
    (x_ptr rms_w_ptr out_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k : Nat)
    (N_SIZE BLOCK_N_SIZE : Nat) (eps : ℝ) :
    ComputeKernel := triton {
  pid_batch = tl.program_id(0)
  pid_m = tl.program_id(1)
  offset_m = pid_batch * $(stride_x_batch) + pid_m * $(stride_x_m)
  block_n_size = tl.arange(0, $(BLOCK_N_SIZE))
  var = tl.zeros([$(BLOCK_N_SIZE)], tl.float32)
  for block_n_strart_ptr in range(0, $(N_SIZE), $(BLOCK_N_SIZE)) {
    offset_n = block_n_strart_ptr + block_n_size
    x_ptr_mask = offset_n < $(N_SIZE)
    x = tl.load(x_ptr + offset_m + offset_n * $(stride_x_k), mask=x_ptr_mask, other=0.0)
    xf = (x).to(tl.float32)
    var += xf * xf
  }
  var = tl.sum(var, axis=0) / $(N_SIZE)
  std = tl.sqrt(var + $(eps))
  for block_n_strart_ptr in range(0, $(N_SIZE), $(BLOCK_N_SIZE)) {
    offset_n = block_n_strart_ptr + block_n_size
    x_ptr_mask = offset_n < $(N_SIZE)
    rms_w_offset = tl.load(rms_w_ptr + offset_n * $(stride_rms_w), mask=x_ptr_mask)
    x = tl.load(x_ptr + offset_m + offset_n * $(stride_x_k), mask=x_ptr_mask, other=0.0)
    x_new = x / std
    out = x_new * rms_w_offset
    out_offset = pid_batch * $(stride_out_batch) + pid_m * $(stride_out_m) +
      offset_n * $(stride_out_k)
    tl.store(out_ptr + out_offset, out, mask=x_ptr_mask)
  }
}
```
</details>

<details><summary><code>outColOffset</code></summary>

```lean
def outColOffset
    (s : BlockState) (stride_out_batch stride_out_m stride_out_k col : Nat) : Nat :=
  s.pids 0 * stride_out_batch + s.pids 1 * stride_out_m + col * stride_out_k
```
</details>

<details><summary><code>rmsnormWeightedYFullNSpec</code></summary>

```
/-- Full-N RMSNorm output including the learned RMS weight. This is the
Python-observable value written by `rmsnorm_triton` for column `i`. -/
```
```lean
noncomputable def rmsnormWeightedYFullNSpec
    (s : BlockState) (x_ptr rms_w_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k stride_rms_w N_SIZE BLOCK_N_SIZE : Nat)
    (eps : ℝ) (i : Fin N_SIZE) : ℝ :=
  rmsnormYFullNSpec s x_ptr stride_x_batch stride_x_m stride_x_k
      N_SIZE BLOCK_N_SIZE eps i *
    s.readMem rms_w_ptr (i.val * stride_rms_w)
```
</details>

<details><summary><code>rmsnormYFullNSpec</code></summary>

```
/-- Multi-block full-N output spec: `x[i] * rmsInvVarFullN` for each
`i < N_SIZE`, expressed against the algebraic ground truth. -/
```
```lean
noncomputable def rmsnormYFullNSpec
    (s : BlockState) (x_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k N_SIZE BLOCK_N_SIZE : Nat)
    (eps : ℝ) (i : Fin N_SIZE) : ℝ :=
  s.readMem x_ptr
      (s.pids 0 * stride_x_batch + s.pids 1 * stride_x_m + i.val * stride_x_k) *
    rmsInvVarFullN s x_ptr stride_x_batch stride_x_m stride_x_k
      N_SIZE BLOCK_N_SIZE eps
```
</details>

<details><summary><code>rmsInvVarFullN</code></summary>

```
/-- Multi-block full-N reciprocal-standard-deviation:
`1 / sqrt(Σ x_j² / N_SIZE + eps)`. -/
```
```lean
noncomputable def rmsInvVarFullN
    (s : BlockState) (x_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k N_SIZE BLOCK_N_SIZE : Nat)
    (eps : ℝ) : ℝ :=
  1 / Real.sqrt
    (rmsVarFullNCarrier s x_ptr stride_x_batch stride_x_m stride_x_k
      N_SIZE BLOCK_N_SIZE / (N_SIZE : ℝ) + eps)
```
</details>

<details><summary><code>rmsVarFullNCarrier</code></summary>

```
/-- Multi-block full-N variance carrier: the algebraic ground truth for
`Σ_{j < N_SIZE} (x[j])²`, independent of any block decomposition. -/
```
```lean
noncomputable def rmsVarFullNCarrier
    (s : BlockState) (x_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k N_SIZE _BLOCK_N_SIZE : Nat) : ℝ :=
  ∑ j : Fin N_SIZE,
    (s.readMem x_ptr
        (xColOffset s stride_x_batch stride_x_m stride_x_k j.val))^2
```
</details>

<details><summary><code>xColOffset</code></summary>

```
/-- Algebraic offset into `x_ptr` for an arbitrary column index `col : Nat`:
row `(batch, m) = (pids 0, pids 1)` at strides `(stride_x_batch, stride_x_m)`,
column `col` at stride `stride_x_k`. -/
```
```lean
def xColOffset
    (s : BlockState) (stride_x_batch stride_x_m stride_x_k col : Nat) : Nat :=
  s.pids 0 * stride_x_batch + s.pids 1 * stride_x_m + col * stride_x_k
```
</details>

## Also present (pinned special-case summaries)
- `rmsnorm_implementation_compute_correct`
