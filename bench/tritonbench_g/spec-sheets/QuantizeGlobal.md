# Spec sheet — `bench/tritonbench_g/quantize_global/QuantizeGlobal.lean`

**Python source:** `bench/tritonbench_g/quantize_global/quantize_global.py`

## Public theorem: `quantize_global_python_n1024_bs1024_blocked_output_summary`

<details><summary>docstring</summary>

```
/-- Public Python summary for `quantize_global`, `n_elements = 1024`,
`BLOCK_SIZE = 1024`.

The faithful Python surface contains CUDA `llrint`, so algorithm projection is
recorded as blocked; the checked store slice characterizes the real-valued
pre-rounding quantity `127.0 * (x * absmax_inv)`. Python's returned `absmax` is
computed by PyTorch before launching the Triton kernel and is outside this
kernel proof. -/
```
</details>

**Statement:**
```lean
theorem quantize_global_python_n1024_bs1024_blocked_output_summary
    (x_ptr absmax_inv_ptr output_ptr : RegionName) (s : BlockState) :
    (∃ err, (quantize_global_surface x_ptr absmax_inv_ptr output_ptr
      1024 1024).toAlgorithm? = Except.error err) ∧
    ComputeCorrect.Realizes
      (kernel := quantize_global_scaled_store_slice x_ptr absmax_inv_ptr output_ptr
        1024 1024 127.0)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin 1024 => offset s 1024 i < 1024)
          (fun i => (output_ptr, offset s 1024 i)))
      (expected := fun i =>
        quantizeGlobalScaledSpec s x_ptr absmax_inv_ptr 1024 127.0 i)
```

**Assumptions / layout contracts:**
- `kernel : = quantize_global_scaled_store_slice x_ptr absmax_inv_ptr output_ptr
        1024 1024 127.0`
- `initialState : = s`
- `fun i : Fin 1024 => offset s 1024 i < 1024`
- `expected : = fun i =>
        quantizeGlobalScaledSpec s x_ptr absmax_inv_ptr 1024 127.0 i`

**Closed-form spec defs (transitive):** `quantize_global_surface`, `quantize_global_scaled_store_slice`, `offset`, `quantizeGlobalScaledSpec`

<details><summary><code>quantize_global_surface</code></summary>

```
/-- Faithful transcription of `quantize_global.py`'s `_quantize_global`.

The CUDA `llrint` operation is preserved as a surface operation; the algorithm
carrier records the pre-cast real value. -/
```
```lean
def quantize_global_surface
    (x_ptr absmax_inv_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  x = tl.load(x_ptr + offsets, mask=mask)
  absmax_inv = tl.load(absmax_inv_ptr)
  output = tl.extra.cuda.libdevice.llrint(127.0 * (x * absmax_inv))
  tl.store(output_ptr + offsets, output, mask=mask)
}
```
</details>

<details><summary><code>quantize_global_scaled_store_slice</code></summary>

```
/-- Proof-oriented arithmetic store slice of `quantize_global.py`'s
`_quantize_global`.

This slice proves the masked vector addressing and scaled value before the
backend-specific rounding/cast step. -/
```
```lean
def quantize_global_scaled_store_slice
    (x_ptr absmax_inv_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) (scale127 : ℝ) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  x = tl.load(x_ptr + offsets, mask=mask)
  absmax_inv = tl.load(absmax_inv_ptr)
  output = $(scale127) * (x * absmax_inv)
  tl.store(output_ptr + offsets, output, mask=mask)
}
```
</details>

<details><summary><code>offset</code></summary>

```lean
def offset (s : BlockState) (BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pid * BLOCK_SIZE + i.val
```
</details>

<details><summary><code>quantizeGlobalScaledSpec</code></summary>

```lean
noncomputable def quantizeGlobalScaledSpec
    (s : BlockState) (x_ptr absmax_inv_ptr : RegionName)
    (BLOCK_SIZE : Nat) (scale127 : ℝ) (i : Fin BLOCK_SIZE) : ℝ :=
  scale127 * (s.readMem x_ptr (offset s BLOCK_SIZE i) *
    s.readMem absmax_inv_ptr 0)
```
</details>

## Public theorem: `quantize_global_python_n2048_bs1024_blocked_output_summary`

<details><summary>docstring</summary>

```
/-- Public Python summary for `quantize_global`, `n_elements = 2048`,
`BLOCK_SIZE = 1024`. -/
```
</details>

**Statement:**
```lean
theorem quantize_global_python_n2048_bs1024_blocked_output_summary
    (x_ptr absmax_inv_ptr output_ptr : RegionName) (s : BlockState) :
    (∃ err, (quantize_global_surface x_ptr absmax_inv_ptr output_ptr
      2048 1024).toAlgorithm? = Except.error err) ∧
    ComputeCorrect.Realizes
      (kernel := quantize_global_scaled_store_slice x_ptr absmax_inv_ptr output_ptr
        2048 1024 127.0)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin 1024 => offset s 1024 i < 2048)
          (fun i => (output_ptr, offset s 1024 i)))
      (expected := fun i =>
        quantizeGlobalScaledSpec s x_ptr absmax_inv_ptr 1024 127.0 i)
```

**Assumptions / layout contracts:**
- `kernel : = quantize_global_scaled_store_slice x_ptr absmax_inv_ptr output_ptr
        2048 1024 127.0`
- `initialState : = s`
- `fun i : Fin 1024 => offset s 1024 i < 2048`
- `expected : = fun i =>
        quantizeGlobalScaledSpec s x_ptr absmax_inv_ptr 1024 127.0 i`

**Closed-form spec defs (transitive):** `quantize_global_surface`, `quantize_global_scaled_store_slice`, `offset`, `quantizeGlobalScaledSpec`

<details><summary><code>quantize_global_surface</code></summary>

```
/-- Faithful transcription of `quantize_global.py`'s `_quantize_global`.

The CUDA `llrint` operation is preserved as a surface operation; the algorithm
carrier records the pre-cast real value. -/
```
```lean
def quantize_global_surface
    (x_ptr absmax_inv_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  x = tl.load(x_ptr + offsets, mask=mask)
  absmax_inv = tl.load(absmax_inv_ptr)
  output = tl.extra.cuda.libdevice.llrint(127.0 * (x * absmax_inv))
  tl.store(output_ptr + offsets, output, mask=mask)
}
```
</details>

<details><summary><code>quantize_global_scaled_store_slice</code></summary>

```
/-- Proof-oriented arithmetic store slice of `quantize_global.py`'s
`_quantize_global`.

This slice proves the masked vector addressing and scaled value before the
backend-specific rounding/cast step. -/
```
```lean
def quantize_global_scaled_store_slice
    (x_ptr absmax_inv_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) (scale127 : ℝ) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  x = tl.load(x_ptr + offsets, mask=mask)
  absmax_inv = tl.load(absmax_inv_ptr)
  output = $(scale127) * (x * absmax_inv)
  tl.store(output_ptr + offsets, output, mask=mask)
}
```
</details>

<details><summary><code>offset</code></summary>

```lean
def offset (s : BlockState) (BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pid * BLOCK_SIZE + i.val
```
</details>

<details><summary><code>quantizeGlobalScaledSpec</code></summary>

```lean
noncomputable def quantizeGlobalScaledSpec
    (s : BlockState) (x_ptr absmax_inv_ptr : RegionName)
    (BLOCK_SIZE : Nat) (scale127 : ℝ) (i : Fin BLOCK_SIZE) : ℝ :=
  scale127 * (s.readMem x_ptr (offset s BLOCK_SIZE i) *
    s.readMem absmax_inv_ptr 0)
```
</details>

## Public theorem: `quantize_global_python_n2048_bs2048_blocked_output_summary`

<details><summary>docstring</summary>

```
/-- Public Python summary for `quantize_global`, `n_elements = 2048`,
`BLOCK_SIZE = 2048`. -/
```
</details>

**Statement:**
```lean
theorem quantize_global_python_n2048_bs2048_blocked_output_summary
    (x_ptr absmax_inv_ptr output_ptr : RegionName) (s : BlockState) :
    (∃ err, (quantize_global_surface x_ptr absmax_inv_ptr output_ptr
      2048 2048).toAlgorithm? = Except.error err) ∧
    ComputeCorrect.Realizes
      (kernel := quantize_global_scaled_store_slice x_ptr absmax_inv_ptr output_ptr
        2048 2048 127.0)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin 2048 => offset s 2048 i < 2048)
          (fun i => (output_ptr, offset s 2048 i)))
      (expected := fun i =>
        quantizeGlobalScaledSpec s x_ptr absmax_inv_ptr 2048 127.0 i)
```

**Assumptions / layout contracts:**
- `kernel : = quantize_global_scaled_store_slice x_ptr absmax_inv_ptr output_ptr
        2048 2048 127.0`
- `initialState : = s`
- `fun i : Fin 2048 => offset s 2048 i < 2048`
- `expected : = fun i =>
        quantizeGlobalScaledSpec s x_ptr absmax_inv_ptr 2048 127.0 i`

**Closed-form spec defs (transitive):** `quantize_global_surface`, `quantize_global_scaled_store_slice`, `offset`, `quantizeGlobalScaledSpec`

<details><summary><code>quantize_global_surface</code></summary>

```
/-- Faithful transcription of `quantize_global.py`'s `_quantize_global`.

The CUDA `llrint` operation is preserved as a surface operation; the algorithm
carrier records the pre-cast real value. -/
```
```lean
def quantize_global_surface
    (x_ptr absmax_inv_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  x = tl.load(x_ptr + offsets, mask=mask)
  absmax_inv = tl.load(absmax_inv_ptr)
  output = tl.extra.cuda.libdevice.llrint(127.0 * (x * absmax_inv))
  tl.store(output_ptr + offsets, output, mask=mask)
}
```
</details>

<details><summary><code>quantize_global_scaled_store_slice</code></summary>

```
/-- Proof-oriented arithmetic store slice of `quantize_global.py`'s
`_quantize_global`.

This slice proves the masked vector addressing and scaled value before the
backend-specific rounding/cast step. -/
```
```lean
def quantize_global_scaled_store_slice
    (x_ptr absmax_inv_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) (scale127 : ℝ) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  x = tl.load(x_ptr + offsets, mask=mask)
  absmax_inv = tl.load(absmax_inv_ptr)
  output = $(scale127) * (x * absmax_inv)
  tl.store(output_ptr + offsets, output, mask=mask)
}
```
</details>

<details><summary><code>offset</code></summary>

```lean
def offset (s : BlockState) (BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pid * BLOCK_SIZE + i.val
```
</details>

<details><summary><code>quantizeGlobalScaledSpec</code></summary>

```lean
noncomputable def quantizeGlobalScaledSpec
    (s : BlockState) (x_ptr absmax_inv_ptr : RegionName)
    (BLOCK_SIZE : Nat) (scale127 : ℝ) (i : Fin BLOCK_SIZE) : ℝ :=
  scale127 * (s.readMem x_ptr (offset s BLOCK_SIZE i) *
    s.readMem absmax_inv_ptr 0)
```
</details>

## Public theorem: `quantize_global_python_n3072_bs1024_blocked_output_summary`

<details><summary>docstring</summary>

```
/-- Public Python summary for `quantize_global`, `n_elements = 3072`,
`BLOCK_SIZE = 1024`. -/
```
</details>

**Statement:**
```lean
theorem quantize_global_python_n3072_bs1024_blocked_output_summary
    (x_ptr absmax_inv_ptr output_ptr : RegionName) (s : BlockState) :
    (∃ err, (quantize_global_surface x_ptr absmax_inv_ptr output_ptr
      3072 1024).toAlgorithm? = Except.error err) ∧
    ComputeCorrect.Realizes
      (kernel := quantize_global_scaled_store_slice x_ptr absmax_inv_ptr output_ptr
        3072 1024 127.0)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin 1024 => offset s 1024 i < 3072)
          (fun i => (output_ptr, offset s 1024 i)))
      (expected := fun i =>
        quantizeGlobalScaledSpec s x_ptr absmax_inv_ptr 1024 127.0 i)
```

**Assumptions / layout contracts:**
- `kernel : = quantize_global_scaled_store_slice x_ptr absmax_inv_ptr output_ptr
        3072 1024 127.0`
- `initialState : = s`
- `fun i : Fin 1024 => offset s 1024 i < 3072`
- `expected : = fun i =>
        quantizeGlobalScaledSpec s x_ptr absmax_inv_ptr 1024 127.0 i`

**Closed-form spec defs (transitive):** `quantize_global_surface`, `quantize_global_scaled_store_slice`, `offset`, `quantizeGlobalScaledSpec`

<details><summary><code>quantize_global_surface</code></summary>

```
/-- Faithful transcription of `quantize_global.py`'s `_quantize_global`.

The CUDA `llrint` operation is preserved as a surface operation; the algorithm
carrier records the pre-cast real value. -/
```
```lean
def quantize_global_surface
    (x_ptr absmax_inv_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  x = tl.load(x_ptr + offsets, mask=mask)
  absmax_inv = tl.load(absmax_inv_ptr)
  output = tl.extra.cuda.libdevice.llrint(127.0 * (x * absmax_inv))
  tl.store(output_ptr + offsets, output, mask=mask)
}
```
</details>

<details><summary><code>quantize_global_scaled_store_slice</code></summary>

```
/-- Proof-oriented arithmetic store slice of `quantize_global.py`'s
`_quantize_global`.

This slice proves the masked vector addressing and scaled value before the
backend-specific rounding/cast step. -/
```
```lean
def quantize_global_scaled_store_slice
    (x_ptr absmax_inv_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) (scale127 : ℝ) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  x = tl.load(x_ptr + offsets, mask=mask)
  absmax_inv = tl.load(absmax_inv_ptr)
  output = $(scale127) * (x * absmax_inv)
  tl.store(output_ptr + offsets, output, mask=mask)
}
```
</details>

<details><summary><code>offset</code></summary>

```lean
def offset (s : BlockState) (BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pid * BLOCK_SIZE + i.val
```
</details>

<details><summary><code>quantizeGlobalScaledSpec</code></summary>

```lean
noncomputable def quantizeGlobalScaledSpec
    (s : BlockState) (x_ptr absmax_inv_ptr : RegionName)
    (BLOCK_SIZE : Nat) (scale127 : ℝ) (i : Fin BLOCK_SIZE) : ℝ :=
  scale127 * (s.readMem x_ptr (offset s BLOCK_SIZE i) *
    s.readMem absmax_inv_ptr 0)
```
</details>

## Public theorem: `quantize_global_python_n3072_bs2048_blocked_output_summary`

<details><summary>docstring</summary>

```
/-- Public Python summary for `quantize_global`, `n_elements = 3072`,
`BLOCK_SIZE = 2048`. -/
```
</details>

**Statement:**
```lean
theorem quantize_global_python_n3072_bs2048_blocked_output_summary
    (x_ptr absmax_inv_ptr output_ptr : RegionName) (s : BlockState) :
    (∃ err, (quantize_global_surface x_ptr absmax_inv_ptr output_ptr
      3072 2048).toAlgorithm? = Except.error err) ∧
    ComputeCorrect.Realizes
      (kernel := quantize_global_scaled_store_slice x_ptr absmax_inv_ptr output_ptr
        3072 2048 127.0)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin 2048 => offset s 2048 i < 3072)
          (fun i => (output_ptr, offset s 2048 i)))
      (expected := fun i =>
        quantizeGlobalScaledSpec s x_ptr absmax_inv_ptr 2048 127.0 i)
```

**Assumptions / layout contracts:**
- `kernel : = quantize_global_scaled_store_slice x_ptr absmax_inv_ptr output_ptr
        3072 2048 127.0`
- `initialState : = s`
- `fun i : Fin 2048 => offset s 2048 i < 3072`
- `expected : = fun i =>
        quantizeGlobalScaledSpec s x_ptr absmax_inv_ptr 2048 127.0 i`

**Closed-form spec defs (transitive):** `quantize_global_surface`, `quantize_global_scaled_store_slice`, `offset`, `quantizeGlobalScaledSpec`

<details><summary><code>quantize_global_surface</code></summary>

```
/-- Faithful transcription of `quantize_global.py`'s `_quantize_global`.

The CUDA `llrint` operation is preserved as a surface operation; the algorithm
carrier records the pre-cast real value. -/
```
```lean
def quantize_global_surface
    (x_ptr absmax_inv_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  x = tl.load(x_ptr + offsets, mask=mask)
  absmax_inv = tl.load(absmax_inv_ptr)
  output = tl.extra.cuda.libdevice.llrint(127.0 * (x * absmax_inv))
  tl.store(output_ptr + offsets, output, mask=mask)
}
```
</details>

<details><summary><code>quantize_global_scaled_store_slice</code></summary>

```
/-- Proof-oriented arithmetic store slice of `quantize_global.py`'s
`_quantize_global`.

This slice proves the masked vector addressing and scaled value before the
backend-specific rounding/cast step. -/
```
```lean
def quantize_global_scaled_store_slice
    (x_ptr absmax_inv_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) (scale127 : ℝ) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  x = tl.load(x_ptr + offsets, mask=mask)
  absmax_inv = tl.load(absmax_inv_ptr)
  output = $(scale127) * (x * absmax_inv)
  tl.store(output_ptr + offsets, output, mask=mask)
}
```
</details>

<details><summary><code>offset</code></summary>

```lean
def offset (s : BlockState) (BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pid * BLOCK_SIZE + i.val
```
</details>

<details><summary><code>quantizeGlobalScaledSpec</code></summary>

```lean
noncomputable def quantizeGlobalScaledSpec
    (s : BlockState) (x_ptr absmax_inv_ptr : RegionName)
    (BLOCK_SIZE : Nat) (scale127 : ℝ) (i : Fin BLOCK_SIZE) : ℝ :=
  scale127 * (s.readMem x_ptr (offset s BLOCK_SIZE i) *
    s.readMem absmax_inv_ptr 0)
```
</details>

## Public theorem: `quantize_global_python_n4096_bs1024_blocked_output_summary`

<details><summary>docstring</summary>

```
/-- Public Python summary for `quantize_global`, `n_elements = 4096`,
`BLOCK_SIZE = 1024`. -/
```
</details>

**Statement:**
```lean
theorem quantize_global_python_n4096_bs1024_blocked_output_summary
    (x_ptr absmax_inv_ptr output_ptr : RegionName) (s : BlockState) :
    (∃ err, (quantize_global_surface x_ptr absmax_inv_ptr output_ptr
      4096 1024).toAlgorithm? = Except.error err) ∧
    ComputeCorrect.Realizes
      (kernel := quantize_global_scaled_store_slice x_ptr absmax_inv_ptr output_ptr
        4096 1024 127.0)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin 1024 => offset s 1024 i < 4096)
          (fun i => (output_ptr, offset s 1024 i)))
      (expected := fun i =>
        quantizeGlobalScaledSpec s x_ptr absmax_inv_ptr 1024 127.0 i)
```

**Assumptions / layout contracts:**
- `kernel : = quantize_global_scaled_store_slice x_ptr absmax_inv_ptr output_ptr
        4096 1024 127.0`
- `initialState : = s`
- `fun i : Fin 1024 => offset s 1024 i < 4096`
- `expected : = fun i =>
        quantizeGlobalScaledSpec s x_ptr absmax_inv_ptr 1024 127.0 i`

**Closed-form spec defs (transitive):** `quantize_global_surface`, `quantize_global_scaled_store_slice`, `offset`, `quantizeGlobalScaledSpec`

<details><summary><code>quantize_global_surface</code></summary>

```
/-- Faithful transcription of `quantize_global.py`'s `_quantize_global`.

The CUDA `llrint` operation is preserved as a surface operation; the algorithm
carrier records the pre-cast real value. -/
```
```lean
def quantize_global_surface
    (x_ptr absmax_inv_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  x = tl.load(x_ptr + offsets, mask=mask)
  absmax_inv = tl.load(absmax_inv_ptr)
  output = tl.extra.cuda.libdevice.llrint(127.0 * (x * absmax_inv))
  tl.store(output_ptr + offsets, output, mask=mask)
}
```
</details>

<details><summary><code>quantize_global_scaled_store_slice</code></summary>

```
/-- Proof-oriented arithmetic store slice of `quantize_global.py`'s
`_quantize_global`.

This slice proves the masked vector addressing and scaled value before the
backend-specific rounding/cast step. -/
```
```lean
def quantize_global_scaled_store_slice
    (x_ptr absmax_inv_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) (scale127 : ℝ) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  x = tl.load(x_ptr + offsets, mask=mask)
  absmax_inv = tl.load(absmax_inv_ptr)
  output = $(scale127) * (x * absmax_inv)
  tl.store(output_ptr + offsets, output, mask=mask)
}
```
</details>

<details><summary><code>offset</code></summary>

```lean
def offset (s : BlockState) (BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pid * BLOCK_SIZE + i.val
```
</details>

<details><summary><code>quantizeGlobalScaledSpec</code></summary>

```lean
noncomputable def quantizeGlobalScaledSpec
    (s : BlockState) (x_ptr absmax_inv_ptr : RegionName)
    (BLOCK_SIZE : Nat) (scale127 : ℝ) (i : Fin BLOCK_SIZE) : ℝ :=
  scale127 * (s.readMem x_ptr (offset s BLOCK_SIZE i) *
    s.readMem absmax_inv_ptr 0)
```
</details>

## Public theorem: `quantize_global_python_n4096_bs2048_blocked_output_summary`

<details><summary>docstring</summary>

```
/-- Public Python summary for `quantize_global`, `n_elements = 4096`,
`BLOCK_SIZE = 2048`. -/
```
</details>

**Statement:**
```lean
theorem quantize_global_python_n4096_bs2048_blocked_output_summary
    (x_ptr absmax_inv_ptr output_ptr : RegionName) (s : BlockState) :
    (∃ err, (quantize_global_surface x_ptr absmax_inv_ptr output_ptr
      4096 2048).toAlgorithm? = Except.error err) ∧
    ComputeCorrect.Realizes
      (kernel := quantize_global_scaled_store_slice x_ptr absmax_inv_ptr output_ptr
        4096 2048 127.0)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin 2048 => offset s 2048 i < 4096)
          (fun i => (output_ptr, offset s 2048 i)))
      (expected := fun i =>
        quantizeGlobalScaledSpec s x_ptr absmax_inv_ptr 2048 127.0 i)
```

**Assumptions / layout contracts:**
- `kernel : = quantize_global_scaled_store_slice x_ptr absmax_inv_ptr output_ptr
        4096 2048 127.0`
- `initialState : = s`
- `fun i : Fin 2048 => offset s 2048 i < 4096`
- `expected : = fun i =>
        quantizeGlobalScaledSpec s x_ptr absmax_inv_ptr 2048 127.0 i`

**Closed-form spec defs (transitive):** `quantize_global_surface`, `quantize_global_scaled_store_slice`, `offset`, `quantizeGlobalScaledSpec`

<details><summary><code>quantize_global_surface</code></summary>

```
/-- Faithful transcription of `quantize_global.py`'s `_quantize_global`.

The CUDA `llrint` operation is preserved as a surface operation; the algorithm
carrier records the pre-cast real value. -/
```
```lean
def quantize_global_surface
    (x_ptr absmax_inv_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  x = tl.load(x_ptr + offsets, mask=mask)
  absmax_inv = tl.load(absmax_inv_ptr)
  output = tl.extra.cuda.libdevice.llrint(127.0 * (x * absmax_inv))
  tl.store(output_ptr + offsets, output, mask=mask)
}
```
</details>

<details><summary><code>quantize_global_scaled_store_slice</code></summary>

```
/-- Proof-oriented arithmetic store slice of `quantize_global.py`'s
`_quantize_global`.

This slice proves the masked vector addressing and scaled value before the
backend-specific rounding/cast step. -/
```
```lean
def quantize_global_scaled_store_slice
    (x_ptr absmax_inv_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) (scale127 : ℝ) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  x = tl.load(x_ptr + offsets, mask=mask)
  absmax_inv = tl.load(absmax_inv_ptr)
  output = $(scale127) * (x * absmax_inv)
  tl.store(output_ptr + offsets, output, mask=mask)
}
```
</details>

<details><summary><code>offset</code></summary>

```lean
def offset (s : BlockState) (BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pid * BLOCK_SIZE + i.val
```
</details>

<details><summary><code>quantizeGlobalScaledSpec</code></summary>

```lean
noncomputable def quantizeGlobalScaledSpec
    (s : BlockState) (x_ptr absmax_inv_ptr : RegionName)
    (BLOCK_SIZE : Nat) (scale127 : ℝ) (i : Fin BLOCK_SIZE) : ℝ :=
  scale127 * (s.readMem x_ptr (offset s BLOCK_SIZE i) *
    s.readMem absmax_inv_ptr 0)
```
</details>

## Also present (pinned special-case summaries)
- `quantize_global_scaled_store_slice_compute_correct`
- `quantize_global_python_n1024_bs1024_compute_correct`
- `quantize_global_python_n2048_bs1024_compute_correct`
- `quantize_global_python_n2048_bs2048_compute_correct`
- `quantize_global_python_n3072_bs1024_compute_correct`
- `quantize_global_python_n3072_bs2048_compute_correct`
- `quantize_global_python_n4096_bs1024_compute_correct`
- `quantize_global_python_n4096_bs2048_compute_correct`
