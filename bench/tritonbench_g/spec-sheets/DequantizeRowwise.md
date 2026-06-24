# Spec sheet — `bench/tritonbench_g/dequantize_rowwise/DequantizeRowwise.lean`

**Python source:** `bench/tritonbench_g/dequantize_rowwise/dequantize_rowwise.py`

## Public theorem: `dequantize_rowwise_python_case1_output_summary`

<details><summary>docstring</summary>

```
/-- Public Python case-1 summary for `dequantize_rowwise`.

The bundled case has shape `(2, 4)`, so each program row writes the four active
lanes of one output row. -/
```
</details>

**Statement:**
```lean
theorem dequantize_rowwise_python_case1_output_summary
    (x_ptr state_x output_ptr : RegionName) (s : BlockState) :
    (∃ alg, (dequantize_rowwise_kernel x_ptr state_x output_ptr (1.0 / 127)
      8 4 4).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := dequantize_rowwise_kernel x_ptr state_x output_ptr (1.0 / 127)
        8 4 4)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin 4 => i.val < 4)
          (fun i => (output_ptr, s.pid * 4 + i.val)))
      (expected := fun i => dequantizeRowwiseSpec s x_ptr state_x 4 (1.0 / 127) i)
```

**Assumptions / layout contracts:**
- `fun i : Fin 4 => i.val < 4`

**Closed-form spec defs (transitive):** `dequantize_rowwise_kernel`, `dequantizeRowwiseSpec`

<details><summary><code>dequantize_rowwise_kernel</code></summary>

```
/-- Faithful 1:1 transcription of `dequantize_rowwise.py`'s `_dequantize_rowwise`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` / `P2: tl.constexpr` → Lean `Nat`
  parameters.
- `n_elements` is kept as `_n_elements`: the upstream Triton kernel accepts it
  but does not use it in the body. -/
```
```lean
def dequantize_rowwise_kernel
    (x_ptr state_x output_ptr : RegionName)
    (inv_127 : ℝ) (_n_elements BLOCK_SIZE P2 : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  arange = tl.arange(0, $(P2))
  offsets = block_start + arange
  row_mask = arange < $(BLOCK_SIZE)
  x = tl.load(x_ptr + offsets, mask=row_mask)
  max_val = tl.load(state_x + pid)
  output = max_val * x * $(inv_127)
  tl.store(output_ptr + offsets, output, mask=row_mask)
}
```
</details>

<details><summary><code>dequantizeRowwiseSpec</code></summary>

```
/-- Exact dequantized value written at active lane `i`. -/
```
```lean
noncomputable def dequantizeRowwiseSpec
    (s : BlockState) (x_ptr state_x : RegionName)
    (BLOCK_SIZE : Nat) (inv_127 : ℝ) (i : Fin P2) : ℝ :=
  s.readMem state_x s.pid *
    s.readMem x_ptr (s.pid * BLOCK_SIZE + i.val) * inv_127
```
</details>

## Public theorem: `dequantize_rowwise_python_case2_output_summary`

<details><summary>docstring</summary>

```
/-- Public Python case-2 summary for `dequantize_rowwise`, shape `(10, 16)`. -/
```
</details>

**Statement:**
```lean
theorem dequantize_rowwise_python_case2_output_summary
    (x_ptr state_x output_ptr : RegionName) (s : BlockState) :
    (∃ alg, (dequantize_rowwise_kernel x_ptr state_x output_ptr (1.0 / 127)
      160 16 16).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := dequantize_rowwise_kernel x_ptr state_x output_ptr (1.0 / 127)
        160 16 16)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin 16 => i.val < 16)
          (fun i => (output_ptr, s.pid * 16 + i.val)))
      (expected := fun i => dequantizeRowwiseSpec s x_ptr state_x 16 (1.0 / 127) i)
```

**Assumptions / layout contracts:**
- `fun i : Fin 16 => i.val < 16`

**Closed-form spec defs (transitive):** `dequantize_rowwise_kernel`, `dequantizeRowwiseSpec`

<details><summary><code>dequantize_rowwise_kernel</code></summary>

```
/-- Faithful 1:1 transcription of `dequantize_rowwise.py`'s `_dequantize_rowwise`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` / `P2: tl.constexpr` → Lean `Nat`
  parameters.
- `n_elements` is kept as `_n_elements`: the upstream Triton kernel accepts it
  but does not use it in the body. -/
```
```lean
def dequantize_rowwise_kernel
    (x_ptr state_x output_ptr : RegionName)
    (inv_127 : ℝ) (_n_elements BLOCK_SIZE P2 : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  arange = tl.arange(0, $(P2))
  offsets = block_start + arange
  row_mask = arange < $(BLOCK_SIZE)
  x = tl.load(x_ptr + offsets, mask=row_mask)
  max_val = tl.load(state_x + pid)
  output = max_val * x * $(inv_127)
  tl.store(output_ptr + offsets, output, mask=row_mask)
}
```
</details>

<details><summary><code>dequantizeRowwiseSpec</code></summary>

```
/-- Exact dequantized value written at active lane `i`. -/
```
```lean
noncomputable def dequantizeRowwiseSpec
    (s : BlockState) (x_ptr state_x : RegionName)
    (BLOCK_SIZE : Nat) (inv_127 : ℝ) (i : Fin P2) : ℝ :=
  s.readMem state_x s.pid *
    s.readMem x_ptr (s.pid * BLOCK_SIZE + i.val) * inv_127
```
</details>

## Public theorem: `dequantize_rowwise_python_case3_output_summary`

<details><summary>docstring</summary>

```
/-- Public Python case-3 summary for `dequantize_rowwise`, shape `(5, 8)`. -/
```
</details>

**Statement:**
```lean
theorem dequantize_rowwise_python_case3_output_summary
    (x_ptr state_x output_ptr : RegionName) (s : BlockState) :
    (∃ alg, (dequantize_rowwise_kernel x_ptr state_x output_ptr (1.0 / 127)
      40 8 8).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := dequantize_rowwise_kernel x_ptr state_x output_ptr (1.0 / 127)
        40 8 8)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin 8 => i.val < 8)
          (fun i => (output_ptr, s.pid * 8 + i.val)))
      (expected := fun i => dequantizeRowwiseSpec s x_ptr state_x 8 (1.0 / 127) i)
```

**Assumptions / layout contracts:**
- `fun i : Fin 8 => i.val < 8`

**Closed-form spec defs (transitive):** `dequantize_rowwise_kernel`, `dequantizeRowwiseSpec`

<details><summary><code>dequantize_rowwise_kernel</code></summary>

```
/-- Faithful 1:1 transcription of `dequantize_rowwise.py`'s `_dequantize_rowwise`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` / `P2: tl.constexpr` → Lean `Nat`
  parameters.
- `n_elements` is kept as `_n_elements`: the upstream Triton kernel accepts it
  but does not use it in the body. -/
```
```lean
def dequantize_rowwise_kernel
    (x_ptr state_x output_ptr : RegionName)
    (inv_127 : ℝ) (_n_elements BLOCK_SIZE P2 : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  arange = tl.arange(0, $(P2))
  offsets = block_start + arange
  row_mask = arange < $(BLOCK_SIZE)
  x = tl.load(x_ptr + offsets, mask=row_mask)
  max_val = tl.load(state_x + pid)
  output = max_val * x * $(inv_127)
  tl.store(output_ptr + offsets, output, mask=row_mask)
}
```
</details>

<details><summary><code>dequantizeRowwiseSpec</code></summary>

```
/-- Exact dequantized value written at active lane `i`. -/
```
```lean
noncomputable def dequantizeRowwiseSpec
    (s : BlockState) (x_ptr state_x : RegionName)
    (BLOCK_SIZE : Nat) (inv_127 : ℝ) (i : Fin P2) : ℝ :=
  s.readMem state_x s.pid *
    s.readMem x_ptr (s.pid * BLOCK_SIZE + i.val) * inv_127
```
</details>

## Public theorem: `dequantize_rowwise_python_case4_output_summary`

<details><summary>docstring</summary>

```
/-- Public Python case-4 summary for `dequantize_rowwise`, shape `(3, 32)`. -/
```
</details>

**Statement:**
```lean
theorem dequantize_rowwise_python_case4_output_summary
    (x_ptr state_x output_ptr : RegionName) (s : BlockState) :
    (∃ alg, (dequantize_rowwise_kernel x_ptr state_x output_ptr (1.0 / 127)
      96 32 32).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := dequantize_rowwise_kernel x_ptr state_x output_ptr (1.0 / 127)
        96 32 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin 32 => i.val < 32)
          (fun i => (output_ptr, s.pid * 32 + i.val)))
      (expected := fun i => dequantizeRowwiseSpec s x_ptr state_x 32 (1.0 / 127) i)
```

**Assumptions / layout contracts:**
- `fun i : Fin 32 => i.val < 32`

**Closed-form spec defs (transitive):** `dequantize_rowwise_kernel`, `dequantizeRowwiseSpec`

<details><summary><code>dequantize_rowwise_kernel</code></summary>

```
/-- Faithful 1:1 transcription of `dequantize_rowwise.py`'s `_dequantize_rowwise`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` / `P2: tl.constexpr` → Lean `Nat`
  parameters.
- `n_elements` is kept as `_n_elements`: the upstream Triton kernel accepts it
  but does not use it in the body. -/
```
```lean
def dequantize_rowwise_kernel
    (x_ptr state_x output_ptr : RegionName)
    (inv_127 : ℝ) (_n_elements BLOCK_SIZE P2 : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  arange = tl.arange(0, $(P2))
  offsets = block_start + arange
  row_mask = arange < $(BLOCK_SIZE)
  x = tl.load(x_ptr + offsets, mask=row_mask)
  max_val = tl.load(state_x + pid)
  output = max_val * x * $(inv_127)
  tl.store(output_ptr + offsets, output, mask=row_mask)
}
```
</details>

<details><summary><code>dequantizeRowwiseSpec</code></summary>

```
/-- Exact dequantized value written at active lane `i`. -/
```
```lean
noncomputable def dequantizeRowwiseSpec
    (s : BlockState) (x_ptr state_x : RegionName)
    (BLOCK_SIZE : Nat) (inv_127 : ℝ) (i : Fin P2) : ℝ :=
  s.readMem state_x s.pid *
    s.readMem x_ptr (s.pid * BLOCK_SIZE + i.val) * inv_127
```
</details>

## Also present (pinned special-case summaries)
- `dequantize_rowwise_kernel_compute_correct`
- `dequantize_rowwise_python_case1_compute_correct`
- `dequantize_rowwise_python_case2_compute_correct`
- `dequantize_rowwise_python_case3_compute_correct`
- `dequantize_rowwise_python_case4_compute_correct`
