# Spec sheet — `bench/tritonbench_g/dequantize_matmul/DequantizeMatmul.lean`

**Python source:** `bench/tritonbench_g/dequantize_matmul/dequantize_matmul.py`

## Public theorem: `dequantize_matmul_python_128x128_output_summary`

<details><summary>docstring</summary>

```
/-- Public Python config summary for `dequantize_matmul.py`,
`BLOCK_SIZE_N=128`, `BLOCK_SIZE_K=128`.

This summarizes the Triton dequantization store into `fp_b`. -/
```
</details>

**Statement:**
```lean
theorem dequantize_matmul_python_128x128_output_summary
    (b_ptr b_scale_ptr fpb_ptr : RegionName) (s : BlockState) :
    (∃ alg, (dequantize_kernel b_ptr b_scale_ptr fpb_ptr
      128 256 256 1 256 1 128 128).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := dequantize_kernel b_ptr b_scale_ptr fpb_ptr
        128 256 256 1 256 1 128 128)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (dequantizeActive s 128 256 128 128)
          (fun idx => (fpb_ptr, fpbOffset s 256 1 128 128 idx)))
      (expected := fun idx =>
        dequantizeSpec s b_ptr b_scale_ptr 256 1 128 128 idx)
```

**Closed-form spec defs (transitive):** `dequantize_kernel`, `dequantizeActive`, `fpbOffset`, `dequantizeSpec`, `bOffset`, `nOffset`

<details><summary><code>dequantize_kernel</code></summary>

```
/-- Faithful 1:1 transcription of `dequantize_matmul.py`'s `dequantize_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE_N: tl.constexpr` / `BLOCK_SIZE_K: tl.constexpr` → Lean
  `Nat` parameters. -/
```
```lean
def dequantize_kernel
    (b_ptr b_scale_ptr fpb_ptr : RegionName)
    (K N stride_bk stride_bn stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K : Nat) :
    ComputeKernel := triton {
  k_block_idx = tl.program_id(axis=0)
  n_block_idx = tl.program_id(axis=1)
  offs_k = tl.arange(0, $(BLOCK_SIZE_K))
  offs_n = tl.arange(0, $(BLOCK_SIZE_N))
  b_offs = (k_block_idx * $(BLOCK_SIZE_K) + offs_k[:, None]) * $(stride_bk) +
    (n_block_idx * $(BLOCK_SIZE_N) + offs_n[None, :]) * $(stride_bn)
  fpb_offs = (k_block_idx * $(BLOCK_SIZE_K) + offs_k[:, None]) * $(stride_fpbk) +
    (n_block_idx * $(BLOCK_SIZE_N) + offs_n[None, :]) * $(stride_fpbn)
  bs_offs = n_block_idx * $(BLOCK_SIZE_N) + offs_n[None, :]
  n_mask = n_block_idx * $(BLOCK_SIZE_N) + offs_n[None, :] < $(N)
  mask = (k_block_idx * $(BLOCK_SIZE_K) + offs_k[:, None] < $(K)) & n_mask
  int_b = tl.load(b_ptr + b_offs, mask=mask, other=0.0)
  scale_b = tl.load(b_scale_ptr + bs_offs, mask=n_mask, other=0.0)
  tl.store(fpb_ptr + fpb_offs, int_b * scale_b, mask=mask)
}
```
</details>

<details><summary><code>dequantizeActive</code></summary>

```lean
def dequantizeActive (s : BlockState) (K N BLOCK_SIZE_N BLOCK_SIZE_K : Nat)
    (idx : TileIndex [BLOCK_SIZE_K, BLOCK_SIZE_N]) : Prop :=
  s.pids 0 * BLOCK_SIZE_K + idx.1.val < K ∧
    s.pids 1 * BLOCK_SIZE_N + idx.2.1.val < N
```
</details>

<details><summary><code>fpbOffset</code></summary>

```lean
def fpbOffset (s : BlockState) (stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K : Nat)
    (idx : TileIndex [BLOCK_SIZE_K, BLOCK_SIZE_N]) : Nat :=
  (s.pids 0 * BLOCK_SIZE_K + idx.1.val) * stride_fpbk +
    (s.pids 1 * BLOCK_SIZE_N + idx.2.1.val) * stride_fpbn
```
</details>

<details><summary><code>dequantizeSpec</code></summary>

```
/-- Exact dequantized value written at active tile lane `idx`. -/
```
```lean
noncomputable def dequantizeSpec
    (s : BlockState) (b_ptr b_scale_ptr : RegionName)
    (stride_bk stride_bn BLOCK_SIZE_N BLOCK_SIZE_K : Nat)
    (idx : TileIndex [BLOCK_SIZE_K, BLOCK_SIZE_N]) : ℝ :=
  s.readMem b_ptr (bOffset s stride_bk stride_bn BLOCK_SIZE_N BLOCK_SIZE_K idx) *
    s.readMem b_scale_ptr (nOffset s BLOCK_SIZE_N idx.2.1)
```
</details>

<details><summary><code>bOffset</code></summary>

```lean
def bOffset (s : BlockState) (stride_bk stride_bn BLOCK_SIZE_N BLOCK_SIZE_K : Nat)
    (idx : TileIndex [BLOCK_SIZE_K, BLOCK_SIZE_N]) : Nat :=
  (s.pids 0 * BLOCK_SIZE_K + idx.1.val) * stride_bk +
    (s.pids 1 * BLOCK_SIZE_N + idx.2.1.val) * stride_bn
```
</details>

<details><summary><code>nOffset</code></summary>

```lean
def nOffset (s : BlockState) (BLOCK_SIZE_N : Nat) (j : Fin BLOCK_SIZE_N) : Nat :=
  s.pids 1 * BLOCK_SIZE_N + j.val
```
</details>

## Public theorem: `dequantize_matmul_python_64x256_output_summary`

<details><summary>docstring</summary>

```
/-- Public Python config summary for `dequantize_matmul.py`,
`BLOCK_SIZE_N=64`, `BLOCK_SIZE_K=256`. -/
```
</details>

**Statement:**
```lean
theorem dequantize_matmul_python_64x256_output_summary
    (b_ptr b_scale_ptr fpb_ptr : RegionName) (s : BlockState) :
    (∃ alg, (dequantize_kernel b_ptr b_scale_ptr fpb_ptr
      128 256 256 1 256 1 64 256).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := dequantize_kernel b_ptr b_scale_ptr fpb_ptr
        128 256 256 1 256 1 64 256)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (dequantizeActive s 128 256 64 256)
          (fun idx => (fpb_ptr, fpbOffset s 256 1 64 256 idx)))
      (expected := fun idx =>
        dequantizeSpec s b_ptr b_scale_ptr 256 1 64 256 idx)
```

**Closed-form spec defs (transitive):** `dequantize_kernel`, `dequantizeActive`, `fpbOffset`, `dequantizeSpec`, `bOffset`, `nOffset`

<details><summary><code>dequantize_kernel</code></summary>

```
/-- Faithful 1:1 transcription of `dequantize_matmul.py`'s `dequantize_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE_N: tl.constexpr` / `BLOCK_SIZE_K: tl.constexpr` → Lean
  `Nat` parameters. -/
```
```lean
def dequantize_kernel
    (b_ptr b_scale_ptr fpb_ptr : RegionName)
    (K N stride_bk stride_bn stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K : Nat) :
    ComputeKernel := triton {
  k_block_idx = tl.program_id(axis=0)
  n_block_idx = tl.program_id(axis=1)
  offs_k = tl.arange(0, $(BLOCK_SIZE_K))
  offs_n = tl.arange(0, $(BLOCK_SIZE_N))
  b_offs = (k_block_idx * $(BLOCK_SIZE_K) + offs_k[:, None]) * $(stride_bk) +
    (n_block_idx * $(BLOCK_SIZE_N) + offs_n[None, :]) * $(stride_bn)
  fpb_offs = (k_block_idx * $(BLOCK_SIZE_K) + offs_k[:, None]) * $(stride_fpbk) +
    (n_block_idx * $(BLOCK_SIZE_N) + offs_n[None, :]) * $(stride_fpbn)
  bs_offs = n_block_idx * $(BLOCK_SIZE_N) + offs_n[None, :]
  n_mask = n_block_idx * $(BLOCK_SIZE_N) + offs_n[None, :] < $(N)
  mask = (k_block_idx * $(BLOCK_SIZE_K) + offs_k[:, None] < $(K)) & n_mask
  int_b = tl.load(b_ptr + b_offs, mask=mask, other=0.0)
  scale_b = tl.load(b_scale_ptr + bs_offs, mask=n_mask, other=0.0)
  tl.store(fpb_ptr + fpb_offs, int_b * scale_b, mask=mask)
}
```
</details>

<details><summary><code>dequantizeActive</code></summary>

```lean
def dequantizeActive (s : BlockState) (K N BLOCK_SIZE_N BLOCK_SIZE_K : Nat)
    (idx : TileIndex [BLOCK_SIZE_K, BLOCK_SIZE_N]) : Prop :=
  s.pids 0 * BLOCK_SIZE_K + idx.1.val < K ∧
    s.pids 1 * BLOCK_SIZE_N + idx.2.1.val < N
```
</details>

<details><summary><code>fpbOffset</code></summary>

```lean
def fpbOffset (s : BlockState) (stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K : Nat)
    (idx : TileIndex [BLOCK_SIZE_K, BLOCK_SIZE_N]) : Nat :=
  (s.pids 0 * BLOCK_SIZE_K + idx.1.val) * stride_fpbk +
    (s.pids 1 * BLOCK_SIZE_N + idx.2.1.val) * stride_fpbn
```
</details>

<details><summary><code>dequantizeSpec</code></summary>

```
/-- Exact dequantized value written at active tile lane `idx`. -/
```
```lean
noncomputable def dequantizeSpec
    (s : BlockState) (b_ptr b_scale_ptr : RegionName)
    (stride_bk stride_bn BLOCK_SIZE_N BLOCK_SIZE_K : Nat)
    (idx : TileIndex [BLOCK_SIZE_K, BLOCK_SIZE_N]) : ℝ :=
  s.readMem b_ptr (bOffset s stride_bk stride_bn BLOCK_SIZE_N BLOCK_SIZE_K idx) *
    s.readMem b_scale_ptr (nOffset s BLOCK_SIZE_N idx.2.1)
```
</details>

<details><summary><code>bOffset</code></summary>

```lean
def bOffset (s : BlockState) (stride_bk stride_bn BLOCK_SIZE_N BLOCK_SIZE_K : Nat)
    (idx : TileIndex [BLOCK_SIZE_K, BLOCK_SIZE_N]) : Nat :=
  (s.pids 0 * BLOCK_SIZE_K + idx.1.val) * stride_bk +
    (s.pids 1 * BLOCK_SIZE_N + idx.2.1.val) * stride_bn
```
</details>

<details><summary><code>nOffset</code></summary>

```lean
def nOffset (s : BlockState) (BLOCK_SIZE_N : Nat) (j : Fin BLOCK_SIZE_N) : Nat :=
  s.pids 1 * BLOCK_SIZE_N + j.val
```
</details>

## Public theorem: `dequantize_matmul_python_32x256_output_summary`

<details><summary>docstring</summary>

```
/-- Public Python config summary for `dequantize_matmul.py`,
`BLOCK_SIZE_N=32`, `BLOCK_SIZE_K=256`. -/
```
</details>

**Statement:**
```lean
theorem dequantize_matmul_python_32x256_output_summary
    (b_ptr b_scale_ptr fpb_ptr : RegionName) (s : BlockState) :
    (∃ alg, (dequantize_kernel b_ptr b_scale_ptr fpb_ptr
      128 256 256 1 256 1 32 256).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := dequantize_kernel b_ptr b_scale_ptr fpb_ptr
        128 256 256 1 256 1 32 256)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (dequantizeActive s 128 256 32 256)
          (fun idx => (fpb_ptr, fpbOffset s 256 1 32 256 idx)))
      (expected := fun idx =>
        dequantizeSpec s b_ptr b_scale_ptr 256 1 32 256 idx)
```

**Closed-form spec defs (transitive):** `dequantize_kernel`, `dequantizeActive`, `fpbOffset`, `dequantizeSpec`, `bOffset`, `nOffset`

<details><summary><code>dequantize_kernel</code></summary>

```
/-- Faithful 1:1 transcription of `dequantize_matmul.py`'s `dequantize_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE_N: tl.constexpr` / `BLOCK_SIZE_K: tl.constexpr` → Lean
  `Nat` parameters. -/
```
```lean
def dequantize_kernel
    (b_ptr b_scale_ptr fpb_ptr : RegionName)
    (K N stride_bk stride_bn stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K : Nat) :
    ComputeKernel := triton {
  k_block_idx = tl.program_id(axis=0)
  n_block_idx = tl.program_id(axis=1)
  offs_k = tl.arange(0, $(BLOCK_SIZE_K))
  offs_n = tl.arange(0, $(BLOCK_SIZE_N))
  b_offs = (k_block_idx * $(BLOCK_SIZE_K) + offs_k[:, None]) * $(stride_bk) +
    (n_block_idx * $(BLOCK_SIZE_N) + offs_n[None, :]) * $(stride_bn)
  fpb_offs = (k_block_idx * $(BLOCK_SIZE_K) + offs_k[:, None]) * $(stride_fpbk) +
    (n_block_idx * $(BLOCK_SIZE_N) + offs_n[None, :]) * $(stride_fpbn)
  bs_offs = n_block_idx * $(BLOCK_SIZE_N) + offs_n[None, :]
  n_mask = n_block_idx * $(BLOCK_SIZE_N) + offs_n[None, :] < $(N)
  mask = (k_block_idx * $(BLOCK_SIZE_K) + offs_k[:, None] < $(K)) & n_mask
  int_b = tl.load(b_ptr + b_offs, mask=mask, other=0.0)
  scale_b = tl.load(b_scale_ptr + bs_offs, mask=n_mask, other=0.0)
  tl.store(fpb_ptr + fpb_offs, int_b * scale_b, mask=mask)
}
```
</details>

<details><summary><code>dequantizeActive</code></summary>

```lean
def dequantizeActive (s : BlockState) (K N BLOCK_SIZE_N BLOCK_SIZE_K : Nat)
    (idx : TileIndex [BLOCK_SIZE_K, BLOCK_SIZE_N]) : Prop :=
  s.pids 0 * BLOCK_SIZE_K + idx.1.val < K ∧
    s.pids 1 * BLOCK_SIZE_N + idx.2.1.val < N
```
</details>

<details><summary><code>fpbOffset</code></summary>

```lean
def fpbOffset (s : BlockState) (stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K : Nat)
    (idx : TileIndex [BLOCK_SIZE_K, BLOCK_SIZE_N]) : Nat :=
  (s.pids 0 * BLOCK_SIZE_K + idx.1.val) * stride_fpbk +
    (s.pids 1 * BLOCK_SIZE_N + idx.2.1.val) * stride_fpbn
```
</details>

<details><summary><code>dequantizeSpec</code></summary>

```
/-- Exact dequantized value written at active tile lane `idx`. -/
```
```lean
noncomputable def dequantizeSpec
    (s : BlockState) (b_ptr b_scale_ptr : RegionName)
    (stride_bk stride_bn BLOCK_SIZE_N BLOCK_SIZE_K : Nat)
    (idx : TileIndex [BLOCK_SIZE_K, BLOCK_SIZE_N]) : ℝ :=
  s.readMem b_ptr (bOffset s stride_bk stride_bn BLOCK_SIZE_N BLOCK_SIZE_K idx) *
    s.readMem b_scale_ptr (nOffset s BLOCK_SIZE_N idx.2.1)
```
</details>

<details><summary><code>bOffset</code></summary>

```lean
def bOffset (s : BlockState) (stride_bk stride_bn BLOCK_SIZE_N BLOCK_SIZE_K : Nat)
    (idx : TileIndex [BLOCK_SIZE_K, BLOCK_SIZE_N]) : Nat :=
  (s.pids 0 * BLOCK_SIZE_K + idx.1.val) * stride_bk +
    (s.pids 1 * BLOCK_SIZE_N + idx.2.1.val) * stride_bn
```
</details>

<details><summary><code>nOffset</code></summary>

```lean
def nOffset (s : BlockState) (BLOCK_SIZE_N : Nat) (j : Fin BLOCK_SIZE_N) : Nat :=
  s.pids 1 * BLOCK_SIZE_N + j.val
```
</details>

## Public theorem: `dequantize_matmul_python_256x64_output_summary`

<details><summary>docstring</summary>

```
/-- Public Python config summary for `dequantize_matmul.py`,
`BLOCK_SIZE_N=256`, `BLOCK_SIZE_K=64`. -/
```
</details>

**Statement:**
```lean
theorem dequantize_matmul_python_256x64_output_summary
    (b_ptr b_scale_ptr fpb_ptr : RegionName) (s : BlockState) :
    (∃ alg, (dequantize_kernel b_ptr b_scale_ptr fpb_ptr
      128 256 256 1 256 1 256 64).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := dequantize_kernel b_ptr b_scale_ptr fpb_ptr
        128 256 256 1 256 1 256 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (dequantizeActive s 128 256 256 64)
          (fun idx => (fpb_ptr, fpbOffset s 256 1 256 64 idx)))
      (expected := fun idx =>
        dequantizeSpec s b_ptr b_scale_ptr 256 1 256 64 idx)
```

**Closed-form spec defs (transitive):** `dequantize_kernel`, `dequantizeActive`, `fpbOffset`, `dequantizeSpec`, `bOffset`, `nOffset`

<details><summary><code>dequantize_kernel</code></summary>

```
/-- Faithful 1:1 transcription of `dequantize_matmul.py`'s `dequantize_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE_N: tl.constexpr` / `BLOCK_SIZE_K: tl.constexpr` → Lean
  `Nat` parameters. -/
```
```lean
def dequantize_kernel
    (b_ptr b_scale_ptr fpb_ptr : RegionName)
    (K N stride_bk stride_bn stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K : Nat) :
    ComputeKernel := triton {
  k_block_idx = tl.program_id(axis=0)
  n_block_idx = tl.program_id(axis=1)
  offs_k = tl.arange(0, $(BLOCK_SIZE_K))
  offs_n = tl.arange(0, $(BLOCK_SIZE_N))
  b_offs = (k_block_idx * $(BLOCK_SIZE_K) + offs_k[:, None]) * $(stride_bk) +
    (n_block_idx * $(BLOCK_SIZE_N) + offs_n[None, :]) * $(stride_bn)
  fpb_offs = (k_block_idx * $(BLOCK_SIZE_K) + offs_k[:, None]) * $(stride_fpbk) +
    (n_block_idx * $(BLOCK_SIZE_N) + offs_n[None, :]) * $(stride_fpbn)
  bs_offs = n_block_idx * $(BLOCK_SIZE_N) + offs_n[None, :]
  n_mask = n_block_idx * $(BLOCK_SIZE_N) + offs_n[None, :] < $(N)
  mask = (k_block_idx * $(BLOCK_SIZE_K) + offs_k[:, None] < $(K)) & n_mask
  int_b = tl.load(b_ptr + b_offs, mask=mask, other=0.0)
  scale_b = tl.load(b_scale_ptr + bs_offs, mask=n_mask, other=0.0)
  tl.store(fpb_ptr + fpb_offs, int_b * scale_b, mask=mask)
}
```
</details>

<details><summary><code>dequantizeActive</code></summary>

```lean
def dequantizeActive (s : BlockState) (K N BLOCK_SIZE_N BLOCK_SIZE_K : Nat)
    (idx : TileIndex [BLOCK_SIZE_K, BLOCK_SIZE_N]) : Prop :=
  s.pids 0 * BLOCK_SIZE_K + idx.1.val < K ∧
    s.pids 1 * BLOCK_SIZE_N + idx.2.1.val < N
```
</details>

<details><summary><code>fpbOffset</code></summary>

```lean
def fpbOffset (s : BlockState) (stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K : Nat)
    (idx : TileIndex [BLOCK_SIZE_K, BLOCK_SIZE_N]) : Nat :=
  (s.pids 0 * BLOCK_SIZE_K + idx.1.val) * stride_fpbk +
    (s.pids 1 * BLOCK_SIZE_N + idx.2.1.val) * stride_fpbn
```
</details>

<details><summary><code>dequantizeSpec</code></summary>

```
/-- Exact dequantized value written at active tile lane `idx`. -/
```
```lean
noncomputable def dequantizeSpec
    (s : BlockState) (b_ptr b_scale_ptr : RegionName)
    (stride_bk stride_bn BLOCK_SIZE_N BLOCK_SIZE_K : Nat)
    (idx : TileIndex [BLOCK_SIZE_K, BLOCK_SIZE_N]) : ℝ :=
  s.readMem b_ptr (bOffset s stride_bk stride_bn BLOCK_SIZE_N BLOCK_SIZE_K idx) *
    s.readMem b_scale_ptr (nOffset s BLOCK_SIZE_N idx.2.1)
```
</details>

<details><summary><code>bOffset</code></summary>

```lean
def bOffset (s : BlockState) (stride_bk stride_bn BLOCK_SIZE_N BLOCK_SIZE_K : Nat)
    (idx : TileIndex [BLOCK_SIZE_K, BLOCK_SIZE_N]) : Nat :=
  (s.pids 0 * BLOCK_SIZE_K + idx.1.val) * stride_bk +
    (s.pids 1 * BLOCK_SIZE_N + idx.2.1.val) * stride_bn
```
</details>

<details><summary><code>nOffset</code></summary>

```lean
def nOffset (s : BlockState) (BLOCK_SIZE_N : Nat) (j : Fin BLOCK_SIZE_N) : Nat :=
  s.pids 1 * BLOCK_SIZE_N + j.val
```
</details>

## Also present (pinned special-case summaries)
- `dequantize_kernel_compute_correct`
- `dequantize_matmul_python_128x128_compute_correct`
- `dequantize_matmul_python_64x256_compute_correct`
- `dequantize_matmul_python_32x256_compute_correct`
- `dequantize_matmul_python_256x64_compute_correct`
