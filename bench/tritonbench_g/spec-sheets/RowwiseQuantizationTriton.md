# Spec sheet — `bench/tritonbench_g/rowwise_quantization_triton/RowwiseQuantizationTriton.lean`

**Python source:** `bench/tritonbench_g/rowwise_quantization_triton/rowwise_quantization_triton.py`

## Public theorem: `quantize_rowwise_python_case1_blocked_output_summary`

<details><summary>docstring</summary>

```
/-- Public Python case 1 summary for rowwise quantization.

The faithful surface is intentionally recorded as blocked at algorithm erasure
by CUDA `llrint`. The accompanying proof slices cover the real-valued scaled
output before backend rounding/cast and the per-row `output_maxs` writeback. -/
```
</details>

**Statement:**
```lean
theorem quantize_rowwise_python_case1_blocked_output_summary
    (x_ptr output_ptr MaxVals output_maxs : RegionName) (s : BlockState) :
    (∃ err,
      (quantize_rowwise_real_surface x_ptr output_ptr output_maxs
        6 3 4).toAlgorithm? = Except.error err) ∧
    ((ComputeCorrect.Realizes
      (kernel := quantize_rowwise_scaled_store_slice x_ptr output_ptr MaxVals
        6 3 4 127.0)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin 4 => i.val < 3)
          (fun i => (output_ptr, offset s 3 i)))
      (expected := fun i =>
        quantizeRowwiseScaledSpec s x_ptr MaxVals 3 127.0 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := quantize_rowwise_max_store_slice MaxVals output_maxs)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.scalar output_maxs (maxOffset s))
      (expected := fun _ : PUnit => quantizeRowwiseMaxSpec s MaxVals)))
```

**Assumptions / layout contracts:**
- `fun i : Fin 4 => i.val < 3`

**Closed-form spec defs (transitive):** `quantize_rowwise_real_surface`, `quantize_rowwise_scaled_store_slice`, `offset`, `quantizeRowwiseScaledSpec`, `quantize_rowwise_max_store_slice`, `maxOffset`, `quantizeRowwiseMaxSpec`

<details><summary><code>quantize_rowwise_real_surface</code></summary>

```
/-- Real-valued surface of `rowwise_quantization_triton.py`'s
`_quantize_rowwise`.

This preserves row addressing, `tl.abs`, masked max reduction, the `output_maxs`
store, CUDA `llrint` surface operation, and the scaled output expression. The
algorithm carrier records the pre-cast real value. -/
```
```lean
def quantize_rowwise_real_surface
    (x_ptr output_ptr output_maxs : RegionName)
    (_n_elements BLOCK_SIZE P2 : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  arange = tl.arange(0, $(P2))
  offsets = block_start + arange
  row_mask = arange < $(BLOCK_SIZE)
  x = tl.load(x_ptr + offsets, mask=row_mask)
  abs_x = tl.abs(x)
  max_val = tl.max(tl.where(row_mask, abs_x, 0.0), axis=0)
  output = tl.extra.cuda.libdevice.llrint(127.0 * (x / max_val))
  tl.store(output_ptr + offsets, output, mask=row_mask)
  tl.store(output_maxs + pid, max_val)
}
```
</details>

<details><summary><code>quantize_rowwise_scaled_store_slice</code></summary>

```
/-- Proof-oriented scaled-output store slice of `rowwise_quantization_triton.py`'s
`_quantize_rowwise`.

The upstream kernel computes `max_val = max(abs(x))`, stores it in
`output_maxs`, rounds `127.0 * (x / max_val)` with CUDA `llrint`, and stores an
int8 row. VeriTile's current arithmetic layer models real tiles, so this slice
starts from a precomputed per-row maximum `MaxVals` and proves the masked
scaled output writeback before the backend-specific rounding/cast step. -/
```
```lean
def quantize_rowwise_scaled_store_slice
    (x_ptr output_ptr MaxVals : RegionName)
    (_n_elements BLOCK_SIZE P2 : Nat) (scale127 : ℝ) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  arange = tl.arange(0, $(P2))
  offsets = block_start + arange
  row_mask = arange < $(BLOCK_SIZE)
  x = tl.load(x_ptr + offsets, mask=row_mask)
  max_val = tl.load(MaxVals + pid)
  output = $(scale127) * (x / max_val)
  tl.store(output_ptr + offsets, output, mask=row_mask)
}
```
</details>

<details><summary><code>offset</code></summary>

```lean
def offset (s : BlockState) (BLOCK_SIZE : Nat) (i : Fin P2) : Nat :=
  s.pid * BLOCK_SIZE + i.val
```
</details>

<details><summary><code>quantizeRowwiseScaledSpec</code></summary>

```lean
noncomputable def quantizeRowwiseScaledSpec
    (s : BlockState) (x_ptr MaxVals : RegionName)
    (BLOCK_SIZE : Nat) (scale127 : ℝ) (i : Fin P2) : ℝ :=
  scale127 * (s.readMem x_ptr (offset s BLOCK_SIZE i) /
    s.readMem MaxVals s.pid)
```
</details>

<details><summary><code>quantize_rowwise_max_store_slice</code></summary>

```
/-- Store slice for the per-row maxima produced by `_quantize_rowwise`.

The full kernel computes `max(abs(x))`; this proof slice starts from a
precomputed `MaxVals` scalar per row and proves the Python `output_maxs + pid`
writeback. -/
```
```lean
def quantize_rowwise_max_store_slice
    (MaxVals output_maxs : RegionName) : ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  max_val = tl.load(MaxVals + pid)
  tl.store(output_maxs + pid, max_val)
}
```
</details>

<details><summary><code>maxOffset</code></summary>

```lean
def maxOffset (s : BlockState) : Nat :=
  s.pid
```
</details>

<details><summary><code>quantizeRowwiseMaxSpec</code></summary>

```lean
noncomputable def quantizeRowwiseMaxSpec (s : BlockState) (MaxVals : RegionName) : ℝ :=
  s.readMem MaxVals (maxOffset s)
```
</details>

## Public theorem: `quantize_rowwise_python_case3_blocked_output_summary`

<details><summary>docstring</summary>

```
/-- Public Python case 3 summary for rowwise quantization.

As in case 1, this keeps the faithful `llrint` blocker explicit while exposing
the checked real-valued scaled output and `output_maxs` proof slices. -/
```
</details>

**Statement:**
```lean
theorem quantize_rowwise_python_case3_blocked_output_summary
    (x_ptr output_ptr MaxVals output_maxs : RegionName) (s : BlockState) :
    (∃ err,
      (quantize_rowwise_real_surface x_ptr output_ptr output_maxs
        10 5 8).toAlgorithm? = Except.error err) ∧
    ((ComputeCorrect.Realizes
      (kernel := quantize_rowwise_scaled_store_slice x_ptr output_ptr MaxVals
        10 5 8 127.0)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin 8 => i.val < 5)
          (fun i => (output_ptr, offset s 5 i)))
      (expected := fun i =>
        quantizeRowwiseScaledSpec s x_ptr MaxVals 5 127.0 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := quantize_rowwise_max_store_slice MaxVals output_maxs)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.scalar output_maxs (maxOffset s))
      (expected := fun _ : PUnit => quantizeRowwiseMaxSpec s MaxVals)))
```

**Assumptions / layout contracts:**
- `fun i : Fin 8 => i.val < 5`

**Closed-form spec defs (transitive):** `quantize_rowwise_real_surface`, `quantize_rowwise_scaled_store_slice`, `offset`, `quantizeRowwiseScaledSpec`, `quantize_rowwise_max_store_slice`, `maxOffset`, `quantizeRowwiseMaxSpec`

<details><summary><code>quantize_rowwise_real_surface</code></summary>

```
/-- Real-valued surface of `rowwise_quantization_triton.py`'s
`_quantize_rowwise`.

This preserves row addressing, `tl.abs`, masked max reduction, the `output_maxs`
store, CUDA `llrint` surface operation, and the scaled output expression. The
algorithm carrier records the pre-cast real value. -/
```
```lean
def quantize_rowwise_real_surface
    (x_ptr output_ptr output_maxs : RegionName)
    (_n_elements BLOCK_SIZE P2 : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  arange = tl.arange(0, $(P2))
  offsets = block_start + arange
  row_mask = arange < $(BLOCK_SIZE)
  x = tl.load(x_ptr + offsets, mask=row_mask)
  abs_x = tl.abs(x)
  max_val = tl.max(tl.where(row_mask, abs_x, 0.0), axis=0)
  output = tl.extra.cuda.libdevice.llrint(127.0 * (x / max_val))
  tl.store(output_ptr + offsets, output, mask=row_mask)
  tl.store(output_maxs + pid, max_val)
}
```
</details>

<details><summary><code>quantize_rowwise_scaled_store_slice</code></summary>

```
/-- Proof-oriented scaled-output store slice of `rowwise_quantization_triton.py`'s
`_quantize_rowwise`.

The upstream kernel computes `max_val = max(abs(x))`, stores it in
`output_maxs`, rounds `127.0 * (x / max_val)` with CUDA `llrint`, and stores an
int8 row. VeriTile's current arithmetic layer models real tiles, so this slice
starts from a precomputed per-row maximum `MaxVals` and proves the masked
scaled output writeback before the backend-specific rounding/cast step. -/
```
```lean
def quantize_rowwise_scaled_store_slice
    (x_ptr output_ptr MaxVals : RegionName)
    (_n_elements BLOCK_SIZE P2 : Nat) (scale127 : ℝ) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  arange = tl.arange(0, $(P2))
  offsets = block_start + arange
  row_mask = arange < $(BLOCK_SIZE)
  x = tl.load(x_ptr + offsets, mask=row_mask)
  max_val = tl.load(MaxVals + pid)
  output = $(scale127) * (x / max_val)
  tl.store(output_ptr + offsets, output, mask=row_mask)
}
```
</details>

<details><summary><code>offset</code></summary>

```lean
def offset (s : BlockState) (BLOCK_SIZE : Nat) (i : Fin P2) : Nat :=
  s.pid * BLOCK_SIZE + i.val
```
</details>

<details><summary><code>quantizeRowwiseScaledSpec</code></summary>

```lean
noncomputable def quantizeRowwiseScaledSpec
    (s : BlockState) (x_ptr MaxVals : RegionName)
    (BLOCK_SIZE : Nat) (scale127 : ℝ) (i : Fin P2) : ℝ :=
  scale127 * (s.readMem x_ptr (offset s BLOCK_SIZE i) /
    s.readMem MaxVals s.pid)
```
</details>

<details><summary><code>quantize_rowwise_max_store_slice</code></summary>

```
/-- Store slice for the per-row maxima produced by `_quantize_rowwise`.

The full kernel computes `max(abs(x))`; this proof slice starts from a
precomputed `MaxVals` scalar per row and proves the Python `output_maxs + pid`
writeback. -/
```
```lean
def quantize_rowwise_max_store_slice
    (MaxVals output_maxs : RegionName) : ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  max_val = tl.load(MaxVals + pid)
  tl.store(output_maxs + pid, max_val)
}
```
</details>

<details><summary><code>maxOffset</code></summary>

```lean
def maxOffset (s : BlockState) : Nat :=
  s.pid
```
</details>

<details><summary><code>quantizeRowwiseMaxSpec</code></summary>

```lean
noncomputable def quantizeRowwiseMaxSpec (s : BlockState) (MaxVals : RegionName) : ℝ :=
  s.readMem MaxVals (maxOffset s)
```
</details>

## Also present (pinned special-case summaries)
- `quantize_rowwise_scaled_store_slice_compute_correct`
- `quantize_rowwise_max_store_slice_compute_correct`
- `quantize_rowwise_python_case1_scaled_output_compute_correct`
- `quantize_rowwise_python_case3_scaled_output_compute_correct`
- `quantize_rowwise_python_output_maxs_compute_correct`
- `quantize_rowwise_python_case1_all_outputs_compute_correct`
- `quantize_rowwise_python_case3_all_outputs_compute_correct`
