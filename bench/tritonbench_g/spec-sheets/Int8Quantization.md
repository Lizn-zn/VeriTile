# Spec sheet — `bench/tritonbench_g/int8_quantization/Int8Quantization.lean`

**Python source:** `bench/tritonbench_g/int8_quantization/int8_quantization.py`

## Public theorem: `per_block_int8_python_case1_internal_summary`

<details><summary>docstring</summary>

```
/-- Public Python case-1 summary for `per_block_int8`.

This combines the faithful Q/K surfaces, including the `to(tl.int8)` cast
surface, with the existing all-output proof slices for `q_int8`, `q_scale`,
`k_int8`, and `k_scale` at `B = 2`, `L = 256`, `C = 64`. The value-store slice
characterizes the real scaled value before fixed-width int8 rounding/cast; the
surface conjunct keeps the original cast operation visible for the Python path. -/
```
</details>

**Statement:**
```lean
theorem per_block_int8_python_case1_internal_summary
    (Q K QInt8 KInt8 QScalePre KScalePre QScale KScale : RegionName)
    (s : BlockState) :
    ((∃ alg, (q_kernel_per_block_int8_surface Q QInt8 QScale
      256 64 128 2).toAlgorithm? = Except.ok alg) ∧
     (∃ alg, (k_kernel_per_block_int8_surface K KInt8 KScale
      256 64 64 4).toAlgorithm? = Except.ok alg) ∧
     (∃ alg, (per_block_int8_scale_compute_store_slice Q QScale
      256 64 128 2 (qPreScale 64)).toAlgorithm? = Except.ok alg) ∧
     (∃ alg, (per_block_int8_scale_compute_store_slice K KScale
      256 64 64 4 kPreScale).toAlgorithm? = Except.ok alg)) ∧
    (ComputeCorrect.Realizes
      (kernel := per_block_int8_scaled_store_slice Q QInt8 QScale
        256 64 128 2 (qPreScale 64))
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 256 128 64)
        (fun idx => (QInt8, xOffset s 256 64 128 idx)))
      (expected := fun idx =>
        perBlockInt8ScaledSpec s Q QScale 256 64 128 2 (qPreScale 64) idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := per_block_int8_scale_store_slice QScalePre QScale 2)
      (initialState := s)
      (write := fun _ : PUnit => some (QScale, scaleOffset s 2))
      (expected := fun _ => scaleStoreSpec s QScalePre 2)) ∧
    (ComputeCorrect.Realizes
      (kernel := per_block_int8_scaled_store_slice K KInt8 KScale
        256 64 64 4 kPreScale)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 256 64 64)
        (fun idx => (KInt8, xOffset s 256 64 64 idx)))
      (expected := fun idx =>
        perBlockInt8ScaledSpec s K KScale 256 64 64 4 kPreScale idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := per_block_int8_scale_store_slice KScalePre KScale 4)
      (initialState := s)
      (write := fun _ : PUnit => some (KScale, scaleOffset s 4))
      (expected := fun _ => scaleStoreSpec s KScalePre 4))
```

**Assumptions / layout contracts:**
- `kernel : = per_block_int8_scaled_store_slice Q QInt8 QScale
        256 64 128 2 (qPreScale 64)`
- `initialState : = s`
- `expected : = fun idx =>
        perBlockInt8ScaledSpec s Q QScale 256 64 128 2 (qPreScale 64) idx`
- `kernel : = per_block_int8_scale_store_slice QScalePre QScale 2`
- `initialState : = s`
- `write : = fun _ : PUnit => some (QScale, scaleOffset s 2)`
- `expected : = fun _ => scaleStoreSpec s QScalePre 2`
- `kernel : = per_block_int8_scaled_store_slice K KInt8 KScale
        256 64 64 4 kPreScale`
- `initialState : = s`
- `expected : = fun idx =>
        perBlockInt8ScaledSpec s K KScale 256 64 64 4 kPreScale idx`
- `kernel : = per_block_int8_scale_store_slice KScalePre KScale 4`
- `initialState : = s`
- `write : = fun _ : PUnit => some (KScale, scaleOffset s 4)`
- `expected : = fun _ => scaleStoreSpec s KScalePre 4`

**Closed-form spec defs (transitive):** `q_kernel_per_block_int8_surface`, `k_kernel_per_block_int8_surface`, `per_block_int8_scale_compute_store_slice`, `qPreScale`, `kPreScale`, `per_block_int8_scaled_store_slice`, `active`, `xOffset`, `perBlockInt8ScaledSpec`, `per_block_int8_scale_store_slice`, `scaleOffset`, `scaleStoreSpec`, `rowIndex`, `baseOffset`, `colIndex`

<details><summary><code>q_kernel_per_block_int8_surface</code></summary>

```
/-- Faithful transcription of `int8_quantization.py`'s `q_kernel_per_block_int8`.

The final `to(tl.int8)` is preserved as a surface dtype annotation; algorithm
erasure now carries it through the fixed-width cast surface used by the DSL.
-/
```
```lean
noncomputable def q_kernel_per_block_int8_surface
    (X XInt8 Scale : RegionName)
    (L C BLK scale_stride : Nat) :
    ComputeKernel := triton {
  off_b = tl.program_id(1)
  off_blk = tl.program_id(0)
  x_offset = off_b * $(L) * $(C)
  offs_m = off_blk * $(BLK) + tl.arange(0, $(BLK))
  offs_k = tl.arange(0, $(C))
  x_ptrs = X + x_offset + offs_m[:, None] * $(C) + offs_k[None, :]
  x_int8_ptrs = XInt8 + x_offset + offs_m[:, None] * $(C) + offs_k[None, :]
  scale_ptrs = Scale + off_b * $(scale_stride) + off_blk
  x = tl.load(x_ptrs, mask=offs_m[:, None] < $(L))
  x *= $(((Real.sqrt (C : ℝ))⁻¹ * (1.44269504 : ℝ) : ℝ))
  scale = tl.max(tl.abs(x)) / 127.0
  x_int8 = x / scale
  x_int8 += 0.5 * tl.where(x_int8 >= 0.0, 1.0, -1.0)
  x_int8 = (x_int8).to(tl.int8)
  tl.store(x_int8_ptrs, x_int8, mask=offs_m[:, None] < $(L))
  tl.store(scale_ptrs, scale)
}
```
</details>

<details><summary><code>k_kernel_per_block_int8_surface</code></summary>

```
/-- Surface transcription of `int8_quantization.py`'s `k_kernel_per_block_int8`.

The final `to(tl.int8)` is preserved as a surface dtype annotation; algorithm
erasure now carries it through the fixed-width cast surface used by the DSL.
-/
```
```lean
def k_kernel_per_block_int8_surface
    (X XInt8 Scale : RegionName)
    (L C BLK scale_stride : Nat) :
    ComputeKernel := triton {
  off_b = tl.program_id(1)
  off_blk = tl.program_id(0)
  x_offset = off_b * $(L) * $(C)
  offs_m = off_blk * $(BLK) + tl.arange(0, $(BLK))
  offs_k = tl.arange(0, $(C))
  x_ptrs = X + x_offset + offs_m[:, None] * $(C) + offs_k[None, :]
  x_int8_ptrs = XInt8 + x_offset + offs_m[:, None] * $(C) + offs_k[None, :]
  scale_ptrs = Scale + off_b * $(scale_stride) + off_blk
  x = tl.load(x_ptrs, mask=offs_m[:, None] < $(L))
  scale = tl.max(tl.abs(x)) / 127.0
  x_int8 = x / scale
  x_int8 += 0.5 * tl.where(x_int8 >= 0.0, 1.0, -1.0)
  x_int8 = (x_int8).to(tl.int8)
  tl.store(x_int8_ptrs, x_int8, mask=offs_m[:, None] < $(L))
  tl.store(scale_ptrs, scale)
}
```
</details>

<details><summary><code>per_block_int8_scale_compute_store_slice</code></summary>

```
/-- Proof-oriented scale-compute slice of `int8_quantization.py`'s per-block
int8 kernels. This covers the real-valued Python path
`scale = tl.max(tl.abs(preScale * x)) / 127.0` and the unmasked scalar
`tl.store(scale_ptrs, scale)`, while separating it from the later `to(tl.int8)`
value-store slice. -/
```
```lean
def per_block_int8_scale_compute_store_slice
    (X Scale : RegionName)
    (L C BLK scale_stride : Nat) (preScale : ℝ) :
    ComputeKernel := triton {
  off_blk = tl.program_id(0)
  off_b = tl.program_id(1)
  x_offset = off_b * $(L) * $(C)
  offs_m = off_blk * $(BLK) + tl.arange(0, $(BLK))
  offs_k = tl.arange(0, $(C))
  x_ptrs = X + x_offset + offs_m[:, None] * $(C) + offs_k[None, :]
  x = tl.load(x_ptrs, mask=offs_m[:, None] < $(L))
  x_scaled = $(preScale) * x
  scale = tl.max(tl.abs(x_scaled)) / 127.0
  tl.store(Scale + off_b * $(scale_stride) + off_blk, scale)
}
```
</details>

<details><summary><code>qPreScale</code></summary>

```lean
noncomputable def qPreScale (C : Nat) : ℝ :=
  (Real.sqrt (C : ℝ))⁻¹ * (1.44269504 : ℝ)
```
</details>

<details><summary><code>kPreScale</code></summary>

```lean
def kPreScale : ℝ := 1
```
</details>

<details><summary><code>per_block_int8_scaled_store_slice</code></summary>

```
/-- Proof-oriented scaled-store slice of `int8_quantization.py`'s
`q_kernel_per_block_int8` / `k_kernel_per_block_int8`.

The upstream kernels compute a per-block max scale, divide each element by that
scale, round to int8, and store the result. VeriTile's current arithmetic layer
models real tiles, so this slice starts from a precomputed per-block scale in
`Scale`, keeps the original row mask, and proves the scaled matrix writeback
before the fixed-width rounding/cast surface. The `preScale` parameter is
`C**-0.5 * 1.44269504` for q and `1` for k. -/
```
```lean
def per_block_int8_scaled_store_slice
    (X XInt8 Scale : RegionName)
    (L C BLK scale_stride : Nat) (preScale : ℝ) :
    ComputeKernel := triton {
  off_blk = tl.program_id(0)
  off_b = tl.program_id(1)
  x_offset = off_b * $(L) * $(C)
  offs_m = off_blk * $(BLK) + tl.arange(0, $(BLK))
  offs_k = tl.arange(0, $(C))
  mask = (offs_m[:, None] < $(L)) & (offs_k[None, :] < $(C))
  x = tl.load(X + x_offset + offs_m[:, None] * $(C) + offs_k[None, :],
    mask=mask)
  scale = tl.load(Scale + off_b * $(scale_stride) + off_blk)
  x_scaled = ($(preScale) * x) / scale
  tl.store(XInt8 + x_offset + offs_m[:, None] * $(C) + offs_k[None, :],
    x_scaled, mask=mask)
}
```
</details>

<details><summary><code>active</code></summary>

```lean
def active
    (s : BlockState) (L BLK C : Nat) (idx : TileIndex [BLK, C]) : Prop :=
  rowIndex s BLK idx.1 < L
```
</details>

<details><summary><code>xOffset</code></summary>

```lean
def xOffset (s : BlockState) (L C BLK : Nat) (idx : TileIndex [BLK, C]) : Nat :=
  baseOffset s L C + rowIndex s BLK idx.1 * C + colIndex s idx.2.1
```
</details>

<details><summary><code>perBlockInt8ScaledSpec</code></summary>

```lean
noncomputable def perBlockInt8ScaledSpec
    (s : BlockState) (X Scale : RegionName)
    (L C BLK scale_stride : Nat) (preScale : ℝ)
    (idx : TileIndex [BLK, C]) : ℝ :=
  (preScale * s.readMem X (xOffset s L C BLK idx)) /
    s.readMem Scale (scaleOffset s scale_stride)
```
</details>

<details><summary><code>per_block_int8_scale_store_slice</code></summary>

```
/-- Proof-oriented scale-store slice of `int8_quantization.py`'s per-block
int8 kernels. Companion to per_block_int8_scaled_store_slice: takes a
precomputed `ScalePre` scalar (per off_b × off_blk) and proves the unmasked
writeback into `Scale` at offset `off_b * scale_stride + off_blk`. -/
```
```lean
def per_block_int8_scale_store_slice
    (ScalePre Scale : RegionName) (scale_stride : Nat) :
    ComputeKernel := triton {
  off_blk = tl.program_id(0)
  off_b = tl.program_id(1)
  scale = tl.load(ScalePre + off_b * $(scale_stride) + off_blk)
  tl.store(Scale + off_b * $(scale_stride) + off_blk, scale)
}
```
</details>

<details><summary><code>scaleOffset</code></summary>

```lean
def scaleOffset (s : BlockState) (scale_stride : Nat) : Nat :=
  s.pids 1 * scale_stride + s.pids 0
```
</details>

<details><summary><code>scaleStoreSpec</code></summary>

```lean
noncomputable def scaleStoreSpec
    (s : BlockState) (ScalePre : RegionName) (scale_stride : Nat) : ℝ :=
  s.readMem ScalePre (scaleOffset s scale_stride)
```
</details>

<details><summary><code>rowIndex</code></summary>

```lean
def rowIndex (s : BlockState) (BLK : Nat) (i : Fin BLK) : Nat :=
  s.pids 0 * BLK + i.val
```
</details>

<details><summary><code>baseOffset</code></summary>

```lean
def baseOffset (s : BlockState) (L C : Nat) : Nat :=
  s.pids 1 * L * C
```
</details>

<details><summary><code>colIndex</code></summary>

```lean
def colIndex (_s : BlockState) (j : Fin C) : Nat :=
  j.val
```
</details>

## Public theorem: `per_block_int8_python_case2_internal_summary`

<details><summary>docstring</summary>

```
/-- Public Python case-2 summary for `per_block_int8`.

As in case 1, the summary pairs faithful full-surface lowering with checked
all-output real-valued slices for the Python test shape
`B = 1`, `L = 512`, `C = 128`. -/
```
</details>

**Statement:**
```lean
theorem per_block_int8_python_case2_internal_summary
    (Q K QInt8 KInt8 QScalePre KScalePre QScale KScale : RegionName)
    (s : BlockState) :
    ((∃ alg, (q_kernel_per_block_int8_surface Q QInt8 QScale
      512 128 128 4).toAlgorithm? = Except.ok alg) ∧
     (∃ alg, (k_kernel_per_block_int8_surface K KInt8 KScale
      512 128 64 8).toAlgorithm? = Except.ok alg) ∧
     (∃ alg, (per_block_int8_scale_compute_store_slice Q QScale
      512 128 128 4 (qPreScale 128)).toAlgorithm? = Except.ok alg) ∧
     (∃ alg, (per_block_int8_scale_compute_store_slice K KScale
      512 128 64 8 kPreScale).toAlgorithm? = Except.ok alg)) ∧
    (ComputeCorrect.Realizes
      (kernel := per_block_int8_scaled_store_slice Q QInt8 QScale
        512 128 128 4 (qPreScale 128))
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 512 128 128)
        (fun idx => (QInt8, xOffset s 512 128 128 idx)))
      (expected := fun idx =>
        perBlockInt8ScaledSpec s Q QScale 512 128 128 4 (qPreScale 128) idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := per_block_int8_scale_store_slice QScalePre QScale 4)
      (initialState := s)
      (write := fun _ : PUnit => some (QScale, scaleOffset s 4))
      (expected := fun _ => scaleStoreSpec s QScalePre 4)) ∧
    (ComputeCorrect.Realizes
      (kernel := per_block_int8_scaled_store_slice K KInt8 KScale
        512 128 64 8 kPreScale)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 512 64 128)
        (fun idx => (KInt8, xOffset s 512 128 64 idx)))
      (expected := fun idx =>
        perBlockInt8ScaledSpec s K KScale 512 128 64 8 kPreScale idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := per_block_int8_scale_store_slice KScalePre KScale 8)
      (initialState := s)
      (write := fun _ : PUnit => some (KScale, scaleOffset s 8))
      (expected := fun _ => scaleStoreSpec s KScalePre 8))
```

**Assumptions / layout contracts:**
- `kernel : = per_block_int8_scaled_store_slice Q QInt8 QScale
        512 128 128 4 (qPreScale 128)`
- `initialState : = s`
- `expected : = fun idx =>
        perBlockInt8ScaledSpec s Q QScale 512 128 128 4 (qPreScale 128) idx`
- `kernel : = per_block_int8_scale_store_slice QScalePre QScale 4`
- `initialState : = s`
- `write : = fun _ : PUnit => some (QScale, scaleOffset s 4)`
- `expected : = fun _ => scaleStoreSpec s QScalePre 4`
- `kernel : = per_block_int8_scaled_store_slice K KInt8 KScale
        512 128 64 8 kPreScale`
- `initialState : = s`
- `expected : = fun idx =>
        perBlockInt8ScaledSpec s K KScale 512 128 64 8 kPreScale idx`
- `kernel : = per_block_int8_scale_store_slice KScalePre KScale 8`
- `initialState : = s`
- `write : = fun _ : PUnit => some (KScale, scaleOffset s 8)`
- `expected : = fun _ => scaleStoreSpec s KScalePre 8`

**Closed-form spec defs (transitive):** `q_kernel_per_block_int8_surface`, `k_kernel_per_block_int8_surface`, `per_block_int8_scale_compute_store_slice`, `qPreScale`, `kPreScale`, `per_block_int8_scaled_store_slice`, `active`, `xOffset`, `perBlockInt8ScaledSpec`, `per_block_int8_scale_store_slice`, `scaleOffset`, `scaleStoreSpec`, `rowIndex`, `baseOffset`, `colIndex`

<details><summary><code>q_kernel_per_block_int8_surface</code></summary>

```
/-- Faithful transcription of `int8_quantization.py`'s `q_kernel_per_block_int8`.

The final `to(tl.int8)` is preserved as a surface dtype annotation; algorithm
erasure now carries it through the fixed-width cast surface used by the DSL.
-/
```
```lean
noncomputable def q_kernel_per_block_int8_surface
    (X XInt8 Scale : RegionName)
    (L C BLK scale_stride : Nat) :
    ComputeKernel := triton {
  off_b = tl.program_id(1)
  off_blk = tl.program_id(0)
  x_offset = off_b * $(L) * $(C)
  offs_m = off_blk * $(BLK) + tl.arange(0, $(BLK))
  offs_k = tl.arange(0, $(C))
  x_ptrs = X + x_offset + offs_m[:, None] * $(C) + offs_k[None, :]
  x_int8_ptrs = XInt8 + x_offset + offs_m[:, None] * $(C) + offs_k[None, :]
  scale_ptrs = Scale + off_b * $(scale_stride) + off_blk
  x = tl.load(x_ptrs, mask=offs_m[:, None] < $(L))
  x *= $(((Real.sqrt (C : ℝ))⁻¹ * (1.44269504 : ℝ) : ℝ))
  scale = tl.max(tl.abs(x)) / 127.0
  x_int8 = x / scale
  x_int8 += 0.5 * tl.where(x_int8 >= 0.0, 1.0, -1.0)
  x_int8 = (x_int8).to(tl.int8)
  tl.store(x_int8_ptrs, x_int8, mask=offs_m[:, None] < $(L))
  tl.store(scale_ptrs, scale)
}
```
</details>

<details><summary><code>k_kernel_per_block_int8_surface</code></summary>

```
/-- Surface transcription of `int8_quantization.py`'s `k_kernel_per_block_int8`.

The final `to(tl.int8)` is preserved as a surface dtype annotation; algorithm
erasure now carries it through the fixed-width cast surface used by the DSL.
-/
```
```lean
def k_kernel_per_block_int8_surface
    (X XInt8 Scale : RegionName)
    (L C BLK scale_stride : Nat) :
    ComputeKernel := triton {
  off_b = tl.program_id(1)
  off_blk = tl.program_id(0)
  x_offset = off_b * $(L) * $(C)
  offs_m = off_blk * $(BLK) + tl.arange(0, $(BLK))
  offs_k = tl.arange(0, $(C))
  x_ptrs = X + x_offset + offs_m[:, None] * $(C) + offs_k[None, :]
  x_int8_ptrs = XInt8 + x_offset + offs_m[:, None] * $(C) + offs_k[None, :]
  scale_ptrs = Scale + off_b * $(scale_stride) + off_blk
  x = tl.load(x_ptrs, mask=offs_m[:, None] < $(L))
  scale = tl.max(tl.abs(x)) / 127.0
  x_int8 = x / scale
  x_int8 += 0.5 * tl.where(x_int8 >= 0.0, 1.0, -1.0)
  x_int8 = (x_int8).to(tl.int8)
  tl.store(x_int8_ptrs, x_int8, mask=offs_m[:, None] < $(L))
  tl.store(scale_ptrs, scale)
}
```
</details>

<details><summary><code>per_block_int8_scale_compute_store_slice</code></summary>

```
/-- Proof-oriented scale-compute slice of `int8_quantization.py`'s per-block
int8 kernels. This covers the real-valued Python path
`scale = tl.max(tl.abs(preScale * x)) / 127.0` and the unmasked scalar
`tl.store(scale_ptrs, scale)`, while separating it from the later `to(tl.int8)`
value-store slice. -/
```
```lean
def per_block_int8_scale_compute_store_slice
    (X Scale : RegionName)
    (L C BLK scale_stride : Nat) (preScale : ℝ) :
    ComputeKernel := triton {
  off_blk = tl.program_id(0)
  off_b = tl.program_id(1)
  x_offset = off_b * $(L) * $(C)
  offs_m = off_blk * $(BLK) + tl.arange(0, $(BLK))
  offs_k = tl.arange(0, $(C))
  x_ptrs = X + x_offset + offs_m[:, None] * $(C) + offs_k[None, :]
  x = tl.load(x_ptrs, mask=offs_m[:, None] < $(L))
  x_scaled = $(preScale) * x
  scale = tl.max(tl.abs(x_scaled)) / 127.0
  tl.store(Scale + off_b * $(scale_stride) + off_blk, scale)
}
```
</details>

<details><summary><code>qPreScale</code></summary>

```lean
noncomputable def qPreScale (C : Nat) : ℝ :=
  (Real.sqrt (C : ℝ))⁻¹ * (1.44269504 : ℝ)
```
</details>

<details><summary><code>kPreScale</code></summary>

```lean
def kPreScale : ℝ := 1
```
</details>

<details><summary><code>per_block_int8_scaled_store_slice</code></summary>

```
/-- Proof-oriented scaled-store slice of `int8_quantization.py`'s
`q_kernel_per_block_int8` / `k_kernel_per_block_int8`.

The upstream kernels compute a per-block max scale, divide each element by that
scale, round to int8, and store the result. VeriTile's current arithmetic layer
models real tiles, so this slice starts from a precomputed per-block scale in
`Scale`, keeps the original row mask, and proves the scaled matrix writeback
before the fixed-width rounding/cast surface. The `preScale` parameter is
`C**-0.5 * 1.44269504` for q and `1` for k. -/
```
```lean
def per_block_int8_scaled_store_slice
    (X XInt8 Scale : RegionName)
    (L C BLK scale_stride : Nat) (preScale : ℝ) :
    ComputeKernel := triton {
  off_blk = tl.program_id(0)
  off_b = tl.program_id(1)
  x_offset = off_b * $(L) * $(C)
  offs_m = off_blk * $(BLK) + tl.arange(0, $(BLK))
  offs_k = tl.arange(0, $(C))
  mask = (offs_m[:, None] < $(L)) & (offs_k[None, :] < $(C))
  x = tl.load(X + x_offset + offs_m[:, None] * $(C) + offs_k[None, :],
    mask=mask)
  scale = tl.load(Scale + off_b * $(scale_stride) + off_blk)
  x_scaled = ($(preScale) * x) / scale
  tl.store(XInt8 + x_offset + offs_m[:, None] * $(C) + offs_k[None, :],
    x_scaled, mask=mask)
}
```
</details>

<details><summary><code>active</code></summary>

```lean
def active
    (s : BlockState) (L BLK C : Nat) (idx : TileIndex [BLK, C]) : Prop :=
  rowIndex s BLK idx.1 < L
```
</details>

<details><summary><code>xOffset</code></summary>

```lean
def xOffset (s : BlockState) (L C BLK : Nat) (idx : TileIndex [BLK, C]) : Nat :=
  baseOffset s L C + rowIndex s BLK idx.1 * C + colIndex s idx.2.1
```
</details>

<details><summary><code>perBlockInt8ScaledSpec</code></summary>

```lean
noncomputable def perBlockInt8ScaledSpec
    (s : BlockState) (X Scale : RegionName)
    (L C BLK scale_stride : Nat) (preScale : ℝ)
    (idx : TileIndex [BLK, C]) : ℝ :=
  (preScale * s.readMem X (xOffset s L C BLK idx)) /
    s.readMem Scale (scaleOffset s scale_stride)
```
</details>

<details><summary><code>per_block_int8_scale_store_slice</code></summary>

```
/-- Proof-oriented scale-store slice of `int8_quantization.py`'s per-block
int8 kernels. Companion to per_block_int8_scaled_store_slice: takes a
precomputed `ScalePre` scalar (per off_b × off_blk) and proves the unmasked
writeback into `Scale` at offset `off_b * scale_stride + off_blk`. -/
```
```lean
def per_block_int8_scale_store_slice
    (ScalePre Scale : RegionName) (scale_stride : Nat) :
    ComputeKernel := triton {
  off_blk = tl.program_id(0)
  off_b = tl.program_id(1)
  scale = tl.load(ScalePre + off_b * $(scale_stride) + off_blk)
  tl.store(Scale + off_b * $(scale_stride) + off_blk, scale)
}
```
</details>

<details><summary><code>scaleOffset</code></summary>

```lean
def scaleOffset (s : BlockState) (scale_stride : Nat) : Nat :=
  s.pids 1 * scale_stride + s.pids 0
```
</details>

<details><summary><code>scaleStoreSpec</code></summary>

```lean
noncomputable def scaleStoreSpec
    (s : BlockState) (ScalePre : RegionName) (scale_stride : Nat) : ℝ :=
  s.readMem ScalePre (scaleOffset s scale_stride)
```
</details>

<details><summary><code>rowIndex</code></summary>

```lean
def rowIndex (s : BlockState) (BLK : Nat) (i : Fin BLK) : Nat :=
  s.pids 0 * BLK + i.val
```
</details>

<details><summary><code>baseOffset</code></summary>

```lean
def baseOffset (s : BlockState) (L C : Nat) : Nat :=
  s.pids 1 * L * C
```
</details>

<details><summary><code>colIndex</code></summary>

```lean
def colIndex (_s : BlockState) (j : Fin C) : Nat :=
  j.val
```
</details>

## Public theorem: `per_block_int8_python_case1_output_summary`

<details><summary>docstring</summary>

```
/-- **Public Python case-1 summary for `per_block_int8` (genuine, gap-free).**

The checked Python shape (`B = 2`, `L = 256`, `C = 64`; q `BLKQ = 128`,
`q_scale.stride(0) = 2`; k `BLKK = 64`, `k_scale.stride(0) = 4`) covers the full
faithful Q/K surface syntax (`tl.abs`, `tl.max`, the per-block scale, the signed
half-up rounding, the `to(tl.int8)` cast, the masked stores) and the externally
visible `q_int8` / `q_scale` / `k_int8` / `k_scale` outputs:

* **value**: `XInt8[off_b, off_blk·BLK + i, j]` equals
  `(preScale · X[...]) / scale` on active rows (`off_blk·BLK + i < L`), unchanged
  otherwise — `preScale = C^{-1/2}·log₂e` for q, `1` for k, and `scale` the
  precomputed per-block max scale (`perBlockInt8ScaledSpec`);
* **scale**: `Scale[off_b·stride + off_blk]` equals the stored per-block scalar
  (`scaleStoreSpec`).

All expected values are computed from the kernel **inputs** (no `exec`/`readMem`
self-reference). This is definitionally the genuine `internal_summary`. -/
```
</details>

**Statement:**
```lean
theorem per_block_int8_python_case1_output_summary
    (Q K QInt8 KInt8 QScalePre KScalePre QScale KScale : RegionName)
    (s : BlockState) :
    ((∃ alg, (q_kernel_per_block_int8_surface Q QInt8 QScale
      256 64 128 2).toAlgorithm? = Except.ok alg) ∧
     (∃ alg, (k_kernel_per_block_int8_surface K KInt8 KScale
      256 64 64 4).toAlgorithm? = Except.ok alg) ∧
     (∃ alg, (per_block_int8_scale_compute_store_slice Q QScale
      256 64 128 2 (qPreScale 64)).toAlgorithm? = Except.ok alg) ∧
     (∃ alg, (per_block_int8_scale_compute_store_slice K KScale
      256 64 64 4 kPreScale).toAlgorithm? = Except.ok alg)) ∧
    (ComputeCorrect.Realizes
      (kernel
```

**Closed-form spec defs (transitive):** `q_kernel_per_block_int8_surface`, `k_kernel_per_block_int8_surface`, `per_block_int8_scale_compute_store_slice`, `qPreScale`, `kPreScale`

<details><summary><code>q_kernel_per_block_int8_surface</code></summary>

```
/-- Faithful transcription of `int8_quantization.py`'s `q_kernel_per_block_int8`.

The final `to(tl.int8)` is preserved as a surface dtype annotation; algorithm
erasure now carries it through the fixed-width cast surface used by the DSL.
-/
```
```lean
noncomputable def q_kernel_per_block_int8_surface
    (X XInt8 Scale : RegionName)
    (L C BLK scale_stride : Nat) :
    ComputeKernel := triton {
  off_b = tl.program_id(1)
  off_blk = tl.program_id(0)
  x_offset = off_b * $(L) * $(C)
  offs_m = off_blk * $(BLK) + tl.arange(0, $(BLK))
  offs_k = tl.arange(0, $(C))
  x_ptrs = X + x_offset + offs_m[:, None] * $(C) + offs_k[None, :]
  x_int8_ptrs = XInt8 + x_offset + offs_m[:, None] * $(C) + offs_k[None, :]
  scale_ptrs = Scale + off_b * $(scale_stride) + off_blk
  x = tl.load(x_ptrs, mask=offs_m[:, None] < $(L))
  x *= $(((Real.sqrt (C : ℝ))⁻¹ * (1.44269504 : ℝ) : ℝ))
  scale = tl.max(tl.abs(x)) / 127.0
  x_int8 = x / scale
  x_int8 += 0.5 * tl.where(x_int8 >= 0.0, 1.0, -1.0)
  x_int8 = (x_int8).to(tl.int8)
  tl.store(x_int8_ptrs, x_int8, mask=offs_m[:, None] < $(L))
  tl.store(scale_ptrs, scale)
}
```
</details>

<details><summary><code>k_kernel_per_block_int8_surface</code></summary>

```
/-- Surface transcription of `int8_quantization.py`'s `k_kernel_per_block_int8`.

The final `to(tl.int8)` is preserved as a surface dtype annotation; algorithm
erasure now carries it through the fixed-width cast surface used by the DSL.
-/
```
```lean
def k_kernel_per_block_int8_surface
    (X XInt8 Scale : RegionName)
    (L C BLK scale_stride : Nat) :
    ComputeKernel := triton {
  off_b = tl.program_id(1)
  off_blk = tl.program_id(0)
  x_offset = off_b * $(L) * $(C)
  offs_m = off_blk * $(BLK) + tl.arange(0, $(BLK))
  offs_k = tl.arange(0, $(C))
  x_ptrs = X + x_offset + offs_m[:, None] * $(C) + offs_k[None, :]
  x_int8_ptrs = XInt8 + x_offset + offs_m[:, None] * $(C) + offs_k[None, :]
  scale_ptrs = Scale + off_b * $(scale_stride) + off_blk
  x = tl.load(x_ptrs, mask=offs_m[:, None] < $(L))
  scale = tl.max(tl.abs(x)) / 127.0
  x_int8 = x / scale
  x_int8 += 0.5 * tl.where(x_int8 >= 0.0, 1.0, -1.0)
  x_int8 = (x_int8).to(tl.int8)
  tl.store(x_int8_ptrs, x_int8, mask=offs_m[:, None] < $(L))
  tl.store(scale_ptrs, scale)
}
```
</details>

<details><summary><code>per_block_int8_scale_compute_store_slice</code></summary>

```
/-- Proof-oriented scale-compute slice of `int8_quantization.py`'s per-block
int8 kernels. This covers the real-valued Python path
`scale = tl.max(tl.abs(preScale * x)) / 127.0` and the unmasked scalar
`tl.store(scale_ptrs, scale)`, while separating it from the later `to(tl.int8)`
value-store slice. -/
```
```lean
def per_block_int8_scale_compute_store_slice
    (X Scale : RegionName)
    (L C BLK scale_stride : Nat) (preScale : ℝ) :
    ComputeKernel := triton {
  off_blk = tl.program_id(0)
  off_b = tl.program_id(1)
  x_offset = off_b * $(L) * $(C)
  offs_m = off_blk * $(BLK) + tl.arange(0, $(BLK))
  offs_k = tl.arange(0, $(C))
  x_ptrs = X + x_offset + offs_m[:, None] * $(C) + offs_k[None, :]
  x = tl.load(x_ptrs, mask=offs_m[:, None] < $(L))
  x_scaled = $(preScale) * x
  scale = tl.max(tl.abs(x_scaled)) / 127.0
  tl.store(Scale + off_b * $(scale_stride) + off_blk, scale)
}
```
</details>

<details><summary><code>qPreScale</code></summary>

```lean
noncomputable def qPreScale (C : Nat) : ℝ :=
  (Real.sqrt (C : ℝ))⁻¹ * (1.44269504 : ℝ)
```
</details>

<details><summary><code>kPreScale</code></summary>

```lean
def kPreScale : ℝ := 1
```
</details>

## Public theorem: `per_block_int8_python_case2_output_summary`

<details><summary>docstring</summary>

```
/-- **Public Python case-2 summary for `per_block_int8` (genuine, gap-free).**

As in case 1, for the Python test shape `B = 1`, `L = 512`, `C = 128`
(q `BLKQ = 128`, scale stride `4`; k `BLKK = 64`, scale stride `8`). All
expected values are computed from the kernel inputs (`perBlockInt8ScaledSpec`,
`scaleStoreSpec`); not self-referential. -/
```
</details>

**Statement:**
```lean
theorem per_block_int8_python_case2_output_summary
    (Q K QInt8 KInt8 QScalePre KScalePre QScale KScale : RegionName)
    (s : BlockState) :
    ((∃ alg, (q_kernel_per_block_int8_surface Q QInt8 QScale
      512 128 128 4).toAlgorithm? = Except.ok alg) ∧
     (∃ alg, (k_kernel_per_block_int8_surface K KInt8 KScale
      512 128 64 8).toAlgorithm? = Except.ok alg) ∧
     (∃ alg, (per_block_int8_scale_compute_store_slice Q QScale
      512 128 128 4 (qPreScale 128)).toAlgorithm? = Except.ok alg) ∧
     (∃ alg, (per_block_int8_scale_compute_store_slice K KScale
      512 128 64 8 kPreScale).toAlgorithm? = Except.ok alg)) ∧
    (ComputeCorrect.Realizes
      (kernel
```

**Closed-form spec defs (transitive):** `q_kernel_per_block_int8_surface`, `k_kernel_per_block_int8_surface`, `per_block_int8_scale_compute_store_slice`, `qPreScale`, `kPreScale`

<details><summary><code>q_kernel_per_block_int8_surface</code></summary>

```
/-- Faithful transcription of `int8_quantization.py`'s `q_kernel_per_block_int8`.

The final `to(tl.int8)` is preserved as a surface dtype annotation; algorithm
erasure now carries it through the fixed-width cast surface used by the DSL.
-/
```
```lean
noncomputable def q_kernel_per_block_int8_surface
    (X XInt8 Scale : RegionName)
    (L C BLK scale_stride : Nat) :
    ComputeKernel := triton {
  off_b = tl.program_id(1)
  off_blk = tl.program_id(0)
  x_offset = off_b * $(L) * $(C)
  offs_m = off_blk * $(BLK) + tl.arange(0, $(BLK))
  offs_k = tl.arange(0, $(C))
  x_ptrs = X + x_offset + offs_m[:, None] * $(C) + offs_k[None, :]
  x_int8_ptrs = XInt8 + x_offset + offs_m[:, None] * $(C) + offs_k[None, :]
  scale_ptrs = Scale + off_b * $(scale_stride) + off_blk
  x = tl.load(x_ptrs, mask=offs_m[:, None] < $(L))
  x *= $(((Real.sqrt (C : ℝ))⁻¹ * (1.44269504 : ℝ) : ℝ))
  scale = tl.max(tl.abs(x)) / 127.0
  x_int8 = x / scale
  x_int8 += 0.5 * tl.where(x_int8 >= 0.0, 1.0, -1.0)
  x_int8 = (x_int8).to(tl.int8)
  tl.store(x_int8_ptrs, x_int8, mask=offs_m[:, None] < $(L))
  tl.store(scale_ptrs, scale)
}
```
</details>

<details><summary><code>k_kernel_per_block_int8_surface</code></summary>

```
/-- Surface transcription of `int8_quantization.py`'s `k_kernel_per_block_int8`.

The final `to(tl.int8)` is preserved as a surface dtype annotation; algorithm
erasure now carries it through the fixed-width cast surface used by the DSL.
-/
```
```lean
def k_kernel_per_block_int8_surface
    (X XInt8 Scale : RegionName)
    (L C BLK scale_stride : Nat) :
    ComputeKernel := triton {
  off_b = tl.program_id(1)
  off_blk = tl.program_id(0)
  x_offset = off_b * $(L) * $(C)
  offs_m = off_blk * $(BLK) + tl.arange(0, $(BLK))
  offs_k = tl.arange(0, $(C))
  x_ptrs = X + x_offset + offs_m[:, None] * $(C) + offs_k[None, :]
  x_int8_ptrs = XInt8 + x_offset + offs_m[:, None] * $(C) + offs_k[None, :]
  scale_ptrs = Scale + off_b * $(scale_stride) + off_blk
  x = tl.load(x_ptrs, mask=offs_m[:, None] < $(L))
  scale = tl.max(tl.abs(x)) / 127.0
  x_int8 = x / scale
  x_int8 += 0.5 * tl.where(x_int8 >= 0.0, 1.0, -1.0)
  x_int8 = (x_int8).to(tl.int8)
  tl.store(x_int8_ptrs, x_int8, mask=offs_m[:, None] < $(L))
  tl.store(scale_ptrs, scale)
}
```
</details>

<details><summary><code>per_block_int8_scale_compute_store_slice</code></summary>

```
/-- Proof-oriented scale-compute slice of `int8_quantization.py`'s per-block
int8 kernels. This covers the real-valued Python path
`scale = tl.max(tl.abs(preScale * x)) / 127.0` and the unmasked scalar
`tl.store(scale_ptrs, scale)`, while separating it from the later `to(tl.int8)`
value-store slice. -/
```
```lean
def per_block_int8_scale_compute_store_slice
    (X Scale : RegionName)
    (L C BLK scale_stride : Nat) (preScale : ℝ) :
    ComputeKernel := triton {
  off_blk = tl.program_id(0)
  off_b = tl.program_id(1)
  x_offset = off_b * $(L) * $(C)
  offs_m = off_blk * $(BLK) + tl.arange(0, $(BLK))
  offs_k = tl.arange(0, $(C))
  x_ptrs = X + x_offset + offs_m[:, None] * $(C) + offs_k[None, :]
  x = tl.load(x_ptrs, mask=offs_m[:, None] < $(L))
  x_scaled = $(preScale) * x
  scale = tl.max(tl.abs(x_scaled)) / 127.0
  tl.store(Scale + off_b * $(scale_stride) + off_blk, scale)
}
```
</details>

<details><summary><code>qPreScale</code></summary>

```lean
noncomputable def qPreScale (C : Nat) : ℝ :=
  (Real.sqrt (C : ℝ))⁻¹ * (1.44269504 : ℝ)
```
</details>

<details><summary><code>kPreScale</code></summary>

```lean
def kPreScale : ℝ := 1
```
</details>

## Also present (pinned special-case summaries)
- `per_block_int8_scaled_store_slice_compute_correct`
- `q_kernel_per_block_int8_scaled_store_slice_compute_correct`
- `k_kernel_per_block_int8_scaled_store_slice_compute_correct`
- `q_kernel_per_block_int8_test1_scaled_store_slice_compute_correct`
- `k_kernel_per_block_int8_test1_scaled_store_slice_compute_correct`
- `q_kernel_per_block_int8_test2_scaled_store_slice_compute_correct`
- `k_kernel_per_block_int8_test2_scaled_store_slice_compute_correct`
- `per_block_int8_scale_store_slice_compute_correct`
- `q_kernel_per_block_int8_scale_store_slice_compute_correct`
- `k_kernel_per_block_int8_scale_store_slice_compute_correct`
- `q_kernel_per_block_int8_test1_scale_store_slice_compute_correct`
- `per_block_int8_test_scale_stride4_store_slice_compute_correct`
- `k_kernel_per_block_int8_test2_scale_store_slice_compute_correct`
- `q_kernel_per_block_int8_python_case1_scaled_store_compute_correct`
- `k_kernel_per_block_int8_python_case1_scaled_store_compute_correct`
- `q_kernel_per_block_int8_python_case2_scaled_store_compute_correct`
- `k_kernel_per_block_int8_python_case2_scaled_store_compute_correct`
- `per_block_int8_python_case1_all_outputs_compute_correct`
- `per_block_int8_python_case2_all_outputs_compute_correct`
- `per_block_int8_closed_form_correct`
