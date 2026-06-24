# Spec sheet — `bench/tritonbench_g/quant_transpose_kernel/QuantTransposeKernel.lean`

**Python source:** `bench/tritonbench_g/quant_transpose_kernel/quant_transpose_kernel.py`

## Public theorem: `quantize_global_transpose_python_case1_blocked_output_summary`

<details><summary>docstring</summary>

```
/-- Public Python case-1 summary for `_quantize_global_transpose`.

The faithful surface uses CUDA `llrint`, so the summary records algorithm
projection as blocked and pairs it with the checked pre-rounding scaled-store
slice. Python's returned `absmax` is computed before the Triton kernel and is
outside this store proof. -/
```
</details>

**Statement:**
```lean
theorem quantize_global_transpose_python_case1_blocked_output_summary
    (A AbsmaxInv B : RegionName) (s : BlockState) :
    (∃ err, (quantize_global_transpose_real_surface A AbsmaxInv B
      256 1 128 1 128 256 128 128 8).toAlgorithm? = Except.error err) ∧
    ComputeCorrect.Realizes
      (kernel := quantize_global_transpose_scaled_store_slice A AbsmaxInv B
        256 1 128 1 128 256 128 128 127.0)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 128 256 128 128)
        (fun idx => (B, bOffset s 1 128 128 128 idx)))
      (expected := fun idx =>
        quantTransposeScaledSpec s A AbsmaxInv 256 1 128 128 127.0 idx)
```

**Assumptions / layout contracts:**
- `kernel : = quantize_global_transpose_scaled_store_slice A AbsmaxInv B
        256 1 128 1 128 256 128 128 127.0`
- `initialState : = s`
- `expected : = fun idx =>
        quantTransposeScaledSpec s A AbsmaxInv 256 1 128 128 127.0 idx`

**Closed-form spec defs (transitive):** `quantize_global_transpose_real_surface`, `quantize_global_transpose_scaled_store_slice`, `active`, `bOffset`, `quantTransposeScaledSpec`, `rowIndex`, `colIndex`, `aOffset`

<details><summary><code>quantize_global_transpose_real_surface</code></summary>

```
/-- Real-valued surface of `quant_transpose_kernel.py`'s
`_quantize_global_transpose`.

This preserves the grouped one-dimensional program-id schedule, masked load,
global scale, CUDA `llrint` surface operation, transposed store addressing, and
masked writeback. The algorithm carrier records the pre-cast real value. -/
```
```lean
def quantize_global_transpose_real_surface
    (A AbsmaxInv B : RegionName)
    (stride_am stride_an stride_bn stride_bm M N BLOCK_M BLOCK_N GROUP_M : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  grid_m = ($((M : Nat)) + $((BLOCK_M : Nat)) - $((1 : Nat))) // $((BLOCK_M : Nat))
  grid_n = ($((N : Nat)) + $((BLOCK_N : Nat)) - $((1 : Nat))) // $((BLOCK_N : Nat))
  width = $(GROUP_M) * grid_n
  group_id = pid // width
  group_size = min(grid_m - group_id * $(GROUP_M), $(GROUP_M))
  pid_m = group_id * $(GROUP_M) + (pid % group_size)
  pid_n = (pid % width) // group_size
  rm = pid_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  rn = pid_n * $(BLOCK_N) + tl.arange(0, $(BLOCK_N))
  A = A + (rm[:, None] * $(stride_am) + rn[None, :] * $(stride_an))
  mask = (rm < $(M))[:, None] & (rn < $(N))[None, :]
  a = tl.load(A, mask=mask)
  absmax_inv = tl.load(AbsmaxInv)
  rm = pid_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  rn = pid_n * $(BLOCK_N) + tl.arange(0, $(BLOCK_N))
  B = B + (rm[:, None] * $(stride_bm) + rn[None, :] * $(stride_bn))
  mask = (rm < $(M))[:, None] & (rn < $(N))[None, :]
  output = tl.extra.cuda.libdevice.llrint(127.0 * (a * absmax_inv))
  tl.store(B, output, mask=mask)
}
```
</details>

<details><summary><code>quantize_global_transpose_scaled_store_slice</code></summary>

```
/-- Proof-oriented scaled-store tile slice of `quant_transpose_kernel.py`'s
`_quantize_global_transpose`.

The full Triton kernel uses a one-dimensional grouped program-id schedule to
derive `pid_m` and `pid_n`. This slice starts after that scheduling choice, uses
program axes 0/1 for the tile coordinates, loads the `BLOCK_M × BLOCK_N` tile
from `A`, applies the global `absmax_inv` scale, and proves the masked writeback
into `B`. CUDA `llrint` and int8 casting are outside VeriTile's current real-tile
arithmetic layer, matching the other quantization ports. -/
```
```lean
def quantize_global_transpose_scaled_store_slice
    (A AbsmaxInv B : RegionName)
    (stride_am stride_an stride_bn stride_bm M N BLOCK_M BLOCK_N : Nat)
    (scale127 : ℝ) :
    ComputeKernel := triton {
  pid_m = tl.program_id(0)
  pid_n = tl.program_id(1)
  rm = pid_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  rn = pid_n * $(BLOCK_N) + tl.arange(0, $(BLOCK_N))
  mask = (rm[:, None] < $(M)) & (rn[None, :] < $(N))
  a = tl.load(A + rm[:, None] * $(stride_am) + rn[None, :] * $(stride_an),
    mask=mask)
  absmax_inv = tl.load(AbsmaxInv)
  output = $(scale127) * (a * absmax_inv)
  tl.store(B + rm[:, None] * $(stride_bm) + rn[None, :] * $(stride_bn),
    output, mask=mask)
}
```
</details>

<details><summary><code>active</code></summary>

```lean
def active
    (s : BlockState) (M N BLOCK_M BLOCK_N : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_N]) : Prop :=
  rowIndex s BLOCK_M idx.1 < M ∧ colIndex s BLOCK_N idx.2.1 < N
```
</details>

<details><summary><code>bOffset</code></summary>

```lean
def bOffset
    (s : BlockState) (stride_bm stride_bn BLOCK_M BLOCK_N : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_N]) : Nat :=
  rowIndex s BLOCK_M idx.1 * stride_bm + colIndex s BLOCK_N idx.2.1 * stride_bn
```
</details>

<details><summary><code>quantTransposeScaledSpec</code></summary>

```lean
noncomputable def quantTransposeScaledSpec
    (s : BlockState) (A AbsmaxInv : RegionName)
    (stride_am stride_an BLOCK_M BLOCK_N : Nat) (scale127 : ℝ)
    (idx : TileIndex [BLOCK_M, BLOCK_N]) : ℝ :=
  scale127 *
    (s.readMem A (aOffset s stride_am stride_an BLOCK_M BLOCK_N idx) *
      s.readMem AbsmaxInv 0)
```
</details>

<details><summary><code>rowIndex</code></summary>

```lean
def rowIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val
```
</details>

<details><summary><code>colIndex</code></summary>

```lean
def colIndex (s : BlockState) (BLOCK_N : Nat) (j : Fin BLOCK_N) : Nat :=
  s.pids 1 * BLOCK_N + j.val
```
</details>

<details><summary><code>aOffset</code></summary>

```lean
def aOffset
    (s : BlockState) (stride_am stride_an BLOCK_M BLOCK_N : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_N]) : Nat :=
  rowIndex s BLOCK_M idx.1 * stride_am + colIndex s BLOCK_N idx.2.1 * stride_an
```
</details>

## Public theorem: `quantize_global_transpose_python_case2_blocked_output_summary`

<details><summary>docstring</summary>

```
/-- Public Python case-2 summary for `_quantize_global_transpose`. -/
```
</details>

**Statement:**
```lean
theorem quantize_global_transpose_python_case2_blocked_output_summary
    (A AbsmaxInv B : RegionName) (s : BlockState) :
    (∃ err, (quantize_global_transpose_real_surface A AbsmaxInv B
      128 1 256 1 256 128 128 128 8).toAlgorithm? = Except.error err) ∧
    ComputeCorrect.Realizes
      (kernel := quantize_global_transpose_scaled_store_slice A AbsmaxInv B
        128 1 256 1 256 128 128 128 127.0)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 256 128 128 128)
        (fun idx => (B, bOffset s 1 256 128 128 idx)))
      (expected := fun idx =>
        quantTransposeScaledSpec s A AbsmaxInv 128 1 128 128 127.0 idx)
```

**Assumptions / layout contracts:**
- `kernel : = quantize_global_transpose_scaled_store_slice A AbsmaxInv B
        128 1 256 1 256 128 128 128 127.0`
- `initialState : = s`
- `expected : = fun idx =>
        quantTransposeScaledSpec s A AbsmaxInv 128 1 128 128 127.0 idx`

**Closed-form spec defs (transitive):** `quantize_global_transpose_real_surface`, `quantize_global_transpose_scaled_store_slice`, `active`, `bOffset`, `quantTransposeScaledSpec`, `rowIndex`, `colIndex`, `aOffset`

<details><summary><code>quantize_global_transpose_real_surface</code></summary>

```
/-- Real-valued surface of `quant_transpose_kernel.py`'s
`_quantize_global_transpose`.

This preserves the grouped one-dimensional program-id schedule, masked load,
global scale, CUDA `llrint` surface operation, transposed store addressing, and
masked writeback. The algorithm carrier records the pre-cast real value. -/
```
```lean
def quantize_global_transpose_real_surface
    (A AbsmaxInv B : RegionName)
    (stride_am stride_an stride_bn stride_bm M N BLOCK_M BLOCK_N GROUP_M : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  grid_m = ($((M : Nat)) + $((BLOCK_M : Nat)) - $((1 : Nat))) // $((BLOCK_M : Nat))
  grid_n = ($((N : Nat)) + $((BLOCK_N : Nat)) - $((1 : Nat))) // $((BLOCK_N : Nat))
  width = $(GROUP_M) * grid_n
  group_id = pid // width
  group_size = min(grid_m - group_id * $(GROUP_M), $(GROUP_M))
  pid_m = group_id * $(GROUP_M) + (pid % group_size)
  pid_n = (pid % width) // group_size
  rm = pid_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  rn = pid_n * $(BLOCK_N) + tl.arange(0, $(BLOCK_N))
  A = A + (rm[:, None] * $(stride_am) + rn[None, :] * $(stride_an))
  mask = (rm < $(M))[:, None] & (rn < $(N))[None, :]
  a = tl.load(A, mask=mask)
  absmax_inv = tl.load(AbsmaxInv)
  rm = pid_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  rn = pid_n * $(BLOCK_N) + tl.arange(0, $(BLOCK_N))
  B = B + (rm[:, None] * $(stride_bm) + rn[None, :] * $(stride_bn))
  mask = (rm < $(M))[:, None] & (rn < $(N))[None, :]
  output = tl.extra.cuda.libdevice.llrint(127.0 * (a * absmax_inv))
  tl.store(B, output, mask=mask)
}
```
</details>

<details><summary><code>quantize_global_transpose_scaled_store_slice</code></summary>

```
/-- Proof-oriented scaled-store tile slice of `quant_transpose_kernel.py`'s
`_quantize_global_transpose`.

The full Triton kernel uses a one-dimensional grouped program-id schedule to
derive `pid_m` and `pid_n`. This slice starts after that scheduling choice, uses
program axes 0/1 for the tile coordinates, loads the `BLOCK_M × BLOCK_N` tile
from `A`, applies the global `absmax_inv` scale, and proves the masked writeback
into `B`. CUDA `llrint` and int8 casting are outside VeriTile's current real-tile
arithmetic layer, matching the other quantization ports. -/
```
```lean
def quantize_global_transpose_scaled_store_slice
    (A AbsmaxInv B : RegionName)
    (stride_am stride_an stride_bn stride_bm M N BLOCK_M BLOCK_N : Nat)
    (scale127 : ℝ) :
    ComputeKernel := triton {
  pid_m = tl.program_id(0)
  pid_n = tl.program_id(1)
  rm = pid_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  rn = pid_n * $(BLOCK_N) + tl.arange(0, $(BLOCK_N))
  mask = (rm[:, None] < $(M)) & (rn[None, :] < $(N))
  a = tl.load(A + rm[:, None] * $(stride_am) + rn[None, :] * $(stride_an),
    mask=mask)
  absmax_inv = tl.load(AbsmaxInv)
  output = $(scale127) * (a * absmax_inv)
  tl.store(B + rm[:, None] * $(stride_bm) + rn[None, :] * $(stride_bn),
    output, mask=mask)
}
```
</details>

<details><summary><code>active</code></summary>

```lean
def active
    (s : BlockState) (M N BLOCK_M BLOCK_N : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_N]) : Prop :=
  rowIndex s BLOCK_M idx.1 < M ∧ colIndex s BLOCK_N idx.2.1 < N
```
</details>

<details><summary><code>bOffset</code></summary>

```lean
def bOffset
    (s : BlockState) (stride_bm stride_bn BLOCK_M BLOCK_N : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_N]) : Nat :=
  rowIndex s BLOCK_M idx.1 * stride_bm + colIndex s BLOCK_N idx.2.1 * stride_bn
```
</details>

<details><summary><code>quantTransposeScaledSpec</code></summary>

```lean
noncomputable def quantTransposeScaledSpec
    (s : BlockState) (A AbsmaxInv : RegionName)
    (stride_am stride_an BLOCK_M BLOCK_N : Nat) (scale127 : ℝ)
    (idx : TileIndex [BLOCK_M, BLOCK_N]) : ℝ :=
  scale127 *
    (s.readMem A (aOffset s stride_am stride_an BLOCK_M BLOCK_N idx) *
      s.readMem AbsmaxInv 0)
```
</details>

<details><summary><code>rowIndex</code></summary>

```lean
def rowIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val
```
</details>

<details><summary><code>colIndex</code></summary>

```lean
def colIndex (s : BlockState) (BLOCK_N : Nat) (j : Fin BLOCK_N) : Nat :=
  s.pids 1 * BLOCK_N + j.val
```
</details>

<details><summary><code>aOffset</code></summary>

```lean
def aOffset
    (s : BlockState) (stride_am stride_an BLOCK_M BLOCK_N : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_N]) : Nat :=
  rowIndex s BLOCK_M idx.1 * stride_am + colIndex s BLOCK_N idx.2.1 * stride_an
```
</details>

## Public theorem: `quantize_global_transpose_python_case3_blocked_output_summary`

<details><summary>docstring</summary>

```
/-- Public Python case-3 summary for `_quantize_global_transpose`. -/
```
</details>

**Statement:**
```lean
theorem quantize_global_transpose_python_case3_blocked_output_summary
    (A AbsmaxInv B : RegionName) (s : BlockState) :
    (∃ err, (quantize_global_transpose_real_surface A AbsmaxInv B
      256 1 512 1 512 256 128 128 8).toAlgorithm? = Except.error err) ∧
    ComputeCorrect.Realizes
      (kernel := quantize_global_transpose_scaled_store_slice A AbsmaxInv B
        256 1 512 1 512 256 128 128 127.0)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 512 256 128 128)
        (fun idx => (B, bOffset s 1 512 128 128 idx)))
      (expected := fun idx =>
        quantTransposeScaledSpec s A AbsmaxInv 256 1 128 128 127.0 idx)
```

**Assumptions / layout contracts:**
- `kernel : = quantize_global_transpose_scaled_store_slice A AbsmaxInv B
        256 1 512 1 512 256 128 128 127.0`
- `initialState : = s`
- `expected : = fun idx =>
        quantTransposeScaledSpec s A AbsmaxInv 256 1 128 128 127.0 idx`

**Closed-form spec defs (transitive):** `quantize_global_transpose_real_surface`, `quantize_global_transpose_scaled_store_slice`, `active`, `bOffset`, `quantTransposeScaledSpec`, `rowIndex`, `colIndex`, `aOffset`

<details><summary><code>quantize_global_transpose_real_surface</code></summary>

```
/-- Real-valued surface of `quant_transpose_kernel.py`'s
`_quantize_global_transpose`.

This preserves the grouped one-dimensional program-id schedule, masked load,
global scale, CUDA `llrint` surface operation, transposed store addressing, and
masked writeback. The algorithm carrier records the pre-cast real value. -/
```
```lean
def quantize_global_transpose_real_surface
    (A AbsmaxInv B : RegionName)
    (stride_am stride_an stride_bn stride_bm M N BLOCK_M BLOCK_N GROUP_M : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  grid_m = ($((M : Nat)) + $((BLOCK_M : Nat)) - $((1 : Nat))) // $((BLOCK_M : Nat))
  grid_n = ($((N : Nat)) + $((BLOCK_N : Nat)) - $((1 : Nat))) // $((BLOCK_N : Nat))
  width = $(GROUP_M) * grid_n
  group_id = pid // width
  group_size = min(grid_m - group_id * $(GROUP_M), $(GROUP_M))
  pid_m = group_id * $(GROUP_M) + (pid % group_size)
  pid_n = (pid % width) // group_size
  rm = pid_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  rn = pid_n * $(BLOCK_N) + tl.arange(0, $(BLOCK_N))
  A = A + (rm[:, None] * $(stride_am) + rn[None, :] * $(stride_an))
  mask = (rm < $(M))[:, None] & (rn < $(N))[None, :]
  a = tl.load(A, mask=mask)
  absmax_inv = tl.load(AbsmaxInv)
  rm = pid_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  rn = pid_n * $(BLOCK_N) + tl.arange(0, $(BLOCK_N))
  B = B + (rm[:, None] * $(stride_bm) + rn[None, :] * $(stride_bn))
  mask = (rm < $(M))[:, None] & (rn < $(N))[None, :]
  output = tl.extra.cuda.libdevice.llrint(127.0 * (a * absmax_inv))
  tl.store(B, output, mask=mask)
}
```
</details>

<details><summary><code>quantize_global_transpose_scaled_store_slice</code></summary>

```
/-- Proof-oriented scaled-store tile slice of `quant_transpose_kernel.py`'s
`_quantize_global_transpose`.

The full Triton kernel uses a one-dimensional grouped program-id schedule to
derive `pid_m` and `pid_n`. This slice starts after that scheduling choice, uses
program axes 0/1 for the tile coordinates, loads the `BLOCK_M × BLOCK_N` tile
from `A`, applies the global `absmax_inv` scale, and proves the masked writeback
into `B`. CUDA `llrint` and int8 casting are outside VeriTile's current real-tile
arithmetic layer, matching the other quantization ports. -/
```
```lean
def quantize_global_transpose_scaled_store_slice
    (A AbsmaxInv B : RegionName)
    (stride_am stride_an stride_bn stride_bm M N BLOCK_M BLOCK_N : Nat)
    (scale127 : ℝ) :
    ComputeKernel := triton {
  pid_m = tl.program_id(0)
  pid_n = tl.program_id(1)
  rm = pid_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  rn = pid_n * $(BLOCK_N) + tl.arange(0, $(BLOCK_N))
  mask = (rm[:, None] < $(M)) & (rn[None, :] < $(N))
  a = tl.load(A + rm[:, None] * $(stride_am) + rn[None, :] * $(stride_an),
    mask=mask)
  absmax_inv = tl.load(AbsmaxInv)
  output = $(scale127) * (a * absmax_inv)
  tl.store(B + rm[:, None] * $(stride_bm) + rn[None, :] * $(stride_bn),
    output, mask=mask)
}
```
</details>

<details><summary><code>active</code></summary>

```lean
def active
    (s : BlockState) (M N BLOCK_M BLOCK_N : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_N]) : Prop :=
  rowIndex s BLOCK_M idx.1 < M ∧ colIndex s BLOCK_N idx.2.1 < N
```
</details>

<details><summary><code>bOffset</code></summary>

```lean
def bOffset
    (s : BlockState) (stride_bm stride_bn BLOCK_M BLOCK_N : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_N]) : Nat :=
  rowIndex s BLOCK_M idx.1 * stride_bm + colIndex s BLOCK_N idx.2.1 * stride_bn
```
</details>

<details><summary><code>quantTransposeScaledSpec</code></summary>

```lean
noncomputable def quantTransposeScaledSpec
    (s : BlockState) (A AbsmaxInv : RegionName)
    (stride_am stride_an BLOCK_M BLOCK_N : Nat) (scale127 : ℝ)
    (idx : TileIndex [BLOCK_M, BLOCK_N]) : ℝ :=
  scale127 *
    (s.readMem A (aOffset s stride_am stride_an BLOCK_M BLOCK_N idx) *
      s.readMem AbsmaxInv 0)
```
</details>

<details><summary><code>rowIndex</code></summary>

```lean
def rowIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val
```
</details>

<details><summary><code>colIndex</code></summary>

```lean
def colIndex (s : BlockState) (BLOCK_N : Nat) (j : Fin BLOCK_N) : Nat :=
  s.pids 1 * BLOCK_N + j.val
```
</details>

<details><summary><code>aOffset</code></summary>

```lean
def aOffset
    (s : BlockState) (stride_am stride_an BLOCK_M BLOCK_N : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_N]) : Nat :=
  rowIndex s BLOCK_M idx.1 * stride_am + colIndex s BLOCK_N idx.2.1 * stride_an
```
</details>

## Public theorem: `quantize_global_transpose_python_case4_blocked_output_summary`

<details><summary>docstring</summary>

```
/-- Public Python case-4 summary for `_quantize_global_transpose`. -/
```
</details>

**Statement:**
```lean
theorem quantize_global_transpose_python_case4_blocked_output_summary
    (A AbsmaxInv B : RegionName) (s : BlockState) :
    (∃ err, (quantize_global_transpose_real_surface A AbsmaxInv B
      512 1 256 1 256 512 128 128 8).toAlgorithm? = Except.error err) ∧
    ComputeCorrect.Realizes
      (kernel := quantize_global_transpose_scaled_store_slice A AbsmaxInv B
        512 1 256 1 256 512 128 128 127.0)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 256 512 128 128)
        (fun idx => (B, bOffset s 1 256 128 128 idx)))
      (expected := fun idx =>
        quantTransposeScaledSpec s A AbsmaxInv 512 1 128 128 127.0 idx)
```

**Assumptions / layout contracts:**
- `kernel : = quantize_global_transpose_scaled_store_slice A AbsmaxInv B
        512 1 256 1 256 512 128 128 127.0`
- `initialState : = s`
- `expected : = fun idx =>
        quantTransposeScaledSpec s A AbsmaxInv 512 1 128 128 127.0 idx`

**Closed-form spec defs (transitive):** `quantize_global_transpose_real_surface`, `quantize_global_transpose_scaled_store_slice`, `active`, `bOffset`, `quantTransposeScaledSpec`, `rowIndex`, `colIndex`, `aOffset`

<details><summary><code>quantize_global_transpose_real_surface</code></summary>

```
/-- Real-valued surface of `quant_transpose_kernel.py`'s
`_quantize_global_transpose`.

This preserves the grouped one-dimensional program-id schedule, masked load,
global scale, CUDA `llrint` surface operation, transposed store addressing, and
masked writeback. The algorithm carrier records the pre-cast real value. -/
```
```lean
def quantize_global_transpose_real_surface
    (A AbsmaxInv B : RegionName)
    (stride_am stride_an stride_bn stride_bm M N BLOCK_M BLOCK_N GROUP_M : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  grid_m = ($((M : Nat)) + $((BLOCK_M : Nat)) - $((1 : Nat))) // $((BLOCK_M : Nat))
  grid_n = ($((N : Nat)) + $((BLOCK_N : Nat)) - $((1 : Nat))) // $((BLOCK_N : Nat))
  width = $(GROUP_M) * grid_n
  group_id = pid // width
  group_size = min(grid_m - group_id * $(GROUP_M), $(GROUP_M))
  pid_m = group_id * $(GROUP_M) + (pid % group_size)
  pid_n = (pid % width) // group_size
  rm = pid_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  rn = pid_n * $(BLOCK_N) + tl.arange(0, $(BLOCK_N))
  A = A + (rm[:, None] * $(stride_am) + rn[None, :] * $(stride_an))
  mask = (rm < $(M))[:, None] & (rn < $(N))[None, :]
  a = tl.load(A, mask=mask)
  absmax_inv = tl.load(AbsmaxInv)
  rm = pid_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  rn = pid_n * $(BLOCK_N) + tl.arange(0, $(BLOCK_N))
  B = B + (rm[:, None] * $(stride_bm) + rn[None, :] * $(stride_bn))
  mask = (rm < $(M))[:, None] & (rn < $(N))[None, :]
  output = tl.extra.cuda.libdevice.llrint(127.0 * (a * absmax_inv))
  tl.store(B, output, mask=mask)
}
```
</details>

<details><summary><code>quantize_global_transpose_scaled_store_slice</code></summary>

```
/-- Proof-oriented scaled-store tile slice of `quant_transpose_kernel.py`'s
`_quantize_global_transpose`.

The full Triton kernel uses a one-dimensional grouped program-id schedule to
derive `pid_m` and `pid_n`. This slice starts after that scheduling choice, uses
program axes 0/1 for the tile coordinates, loads the `BLOCK_M × BLOCK_N` tile
from `A`, applies the global `absmax_inv` scale, and proves the masked writeback
into `B`. CUDA `llrint` and int8 casting are outside VeriTile's current real-tile
arithmetic layer, matching the other quantization ports. -/
```
```lean
def quantize_global_transpose_scaled_store_slice
    (A AbsmaxInv B : RegionName)
    (stride_am stride_an stride_bn stride_bm M N BLOCK_M BLOCK_N : Nat)
    (scale127 : ℝ) :
    ComputeKernel := triton {
  pid_m = tl.program_id(0)
  pid_n = tl.program_id(1)
  rm = pid_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  rn = pid_n * $(BLOCK_N) + tl.arange(0, $(BLOCK_N))
  mask = (rm[:, None] < $(M)) & (rn[None, :] < $(N))
  a = tl.load(A + rm[:, None] * $(stride_am) + rn[None, :] * $(stride_an),
    mask=mask)
  absmax_inv = tl.load(AbsmaxInv)
  output = $(scale127) * (a * absmax_inv)
  tl.store(B + rm[:, None] * $(stride_bm) + rn[None, :] * $(stride_bn),
    output, mask=mask)
}
```
</details>

<details><summary><code>active</code></summary>

```lean
def active
    (s : BlockState) (M N BLOCK_M BLOCK_N : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_N]) : Prop :=
  rowIndex s BLOCK_M idx.1 < M ∧ colIndex s BLOCK_N idx.2.1 < N
```
</details>

<details><summary><code>bOffset</code></summary>

```lean
def bOffset
    (s : BlockState) (stride_bm stride_bn BLOCK_M BLOCK_N : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_N]) : Nat :=
  rowIndex s BLOCK_M idx.1 * stride_bm + colIndex s BLOCK_N idx.2.1 * stride_bn
```
</details>

<details><summary><code>quantTransposeScaledSpec</code></summary>

```lean
noncomputable def quantTransposeScaledSpec
    (s : BlockState) (A AbsmaxInv : RegionName)
    (stride_am stride_an BLOCK_M BLOCK_N : Nat) (scale127 : ℝ)
    (idx : TileIndex [BLOCK_M, BLOCK_N]) : ℝ :=
  scale127 *
    (s.readMem A (aOffset s stride_am stride_an BLOCK_M BLOCK_N idx) *
      s.readMem AbsmaxInv 0)
```
</details>

<details><summary><code>rowIndex</code></summary>

```lean
def rowIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val
```
</details>

<details><summary><code>colIndex</code></summary>

```lean
def colIndex (s : BlockState) (BLOCK_N : Nat) (j : Fin BLOCK_N) : Nat :=
  s.pids 1 * BLOCK_N + j.val
```
</details>

<details><summary><code>aOffset</code></summary>

```lean
def aOffset
    (s : BlockState) (stride_am stride_an BLOCK_M BLOCK_N : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_N]) : Nat :=
  rowIndex s BLOCK_M idx.1 * stride_am + colIndex s BLOCK_N idx.2.1 * stride_an
```
</details>

## Also present (pinned special-case summaries)
- `quantize_global_transpose_scaled_store_slice_compute_correct`
- `quantize_global_transpose_python_case1_compute_correct`
- `quantize_global_transpose_python_case2_compute_correct`
- `quantize_global_transpose_python_case3_compute_correct`
- `quantize_global_transpose_python_case4_compute_correct`
