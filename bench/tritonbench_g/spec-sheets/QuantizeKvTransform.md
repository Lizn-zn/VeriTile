# Spec sheet — `bench/tritonbench_g/quantize_kv_transform/QuantizeKvTransform.lean`

**Python source:** `bench/tritonbench_g/quantize_kv_transform/quantize_kv_transform.py`

## Public theorem: `destindex_copy_quantize_kv_transform_python_h12_d96_summary`

<details><summary>docstring</summary>

```
/-- Public Python `H = 12, D = 96` summary: the checked shape covers the full
surface syntax and the two externally visible outputs (values and scales). -/
```
</details>

**Statement:**
```lean
theorem destindex_copy_quantize_kv_transform_python_h12_d96_summary
    (K Scale DestLoc Out OutScale : RegionName) (s : BlockState) :
    (∃ alg,
      (destindex_copy_quantize_kv_transform_real_surface K DestLoc Out OutScale
        1152 96 1 1152 96 1 12 1 1 12 96 128 16).toAlgorithm? =
          Except.ok alg) ∧
    ((ComputeCorrect.Realizes
      (kernel := destindex_copy_quantize_kv_transform_value_store_slice K
        DestLoc Out OutScale 1152 96 1 1152 96 1 12 1 12 96 16 128)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 12 96 16 128)
        (fun idx => (Out, outOffset s DestLoc 1152 96 1 idx)))
      (expected := fun idx =>
        quantizeKvTransformValueSpec s K DestLoc OutScale 1152 96 1 12 1 idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := destindex_copy_quantize_kv_transform_scale_store_slice Scale
        DestLoc OutScale 12 12 1 12 16)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (scaleActive 12 16)
        (fun i => (OutScale, scaleOutOffset s DestLoc 12 1 i)))
      (expected := fun i =>
        quantizeKvTransformScaleSpec s Scale 12 i)))
```

**Assumptions / layout contracts:**
- `kernel : = destindex_copy_quantize_kv_transform_value_store_slice K
        DestLoc Out OutScale 1152 96 1 1152 96 1 12 1 12 96 16 128`
- `initialState : = s`
- `expected : = fun idx =>
        quantizeKvTransformValueSpec s K DestLoc OutScale 1152 96 1 12 1 idx`
- `kernel : = destindex_copy_quantize_kv_transform_scale_store_slice Scale
        DestLoc OutScale 12 12 1 12 16`
- `initialState : = s`
- `expected : = fun i =>
        quantizeKvTransformScaleSpec s Scale 12 i`

**Closed-form spec defs (transitive):** `destindex_copy_quantize_kv_transform_real_surface`, `destindex_copy_quantize_kv_transform_value_store_slice`, `active`, `outOffset`, `quantizeKvTransformValueSpec`, `destindex_copy_quantize_kv_transform_scale_store_slice`, `scaleActive`, `scaleOutOffset`, `quantizeKvTransformScaleSpec`, `headIndex`, `dimIndex`, `destIndex`, `kOffset`, `scaleOffset`, `scaleSourceOffset`

<details><summary><code>destindex_copy_quantize_kv_transform_real_surface</code></summary>

```
/-- Real-valued surface of `quantize_kv_transform.py`'s
`_fwd_kernel_destindex_copy_quantize_kv`.

This preserves destination-indexed addressing, the `head_num/head_dim` mask,
`tl.abs`, per-head scale computation, value writeback, and scale writeback. The
Python kernel casts the scale to `OutScale.dtype.element_ty`; that cast is
represented explicitly. The final quotient cast to int8 is preserved as a
surface dtype annotation and lowers through the DSL's fixed-width cast
surface. -/
```
```lean
def destindex_copy_quantize_kv_transform_real_surface
    (K : RegionName) (DestLoc : Region .nat) (Out OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_d
      stride_o_bs stride_o_h stride_o_d
      stride_os_bs stride_os_h _stride_os_d
      head_num head_dim BLOCK_DMODEL BLOCK_HEAD : Nat) :
    ComputeKernel := triton {
  cur_index = tl.program_id(0)
  offs_h = tl.arange(0, $(BLOCK_HEAD))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  dest_index = tl.load(DestLoc + cur_index)
  src_data = tl.load(K + cur_index * $(stride_k_bs) +
      offs_h[:, None] * $(stride_k_h) + $(stride_k_d) * offs_d[None, :],
    mask=(offs_h[:, None] < $(head_num)) & (offs_d[None, :] < $(head_dim)),
    other=0.0)
  abs_data = tl.abs(src_data)
  data_scale = ((tl.max(abs_data, axis=1) / 127.0).to(OutScale.dtype.element_ty))[:, None]
  q_src_data = (src_data / data_scale).to(tl.int8)
  o_ptrs = Out + dest_index * $(stride_o_bs) +
    $(stride_o_h) * offs_h[:, None] + $(stride_o_d) * offs_d[None, :]
  os_ptrs = OutScale + dest_index * $(stride_os_bs) + $(stride_os_h) * offs_h[:, None]
  tl.store(o_ptrs, q_src_data,
    mask=(offs_h[:, None] < $(head_num)) & (offs_d[None, :] < $(head_dim)))
  tl.store(os_ptrs, data_scale, mask=offs_h[:, None] < $(head_num))
}
```
</details>

<details><summary><code>destindex_copy_quantize_kv_transform_value_store_slice</code></summary>

```
/-- Proof-oriented q-value writeback slice of `quantize_kv_transform.py`'s
`_fwd_kernel_destindex_copy_quantize_kv`.

The full kernel computes a per-head scale from `max(abs(src_data))`, casts the
quotient to int8, and stores both values and scales. This slice starts from a
precomputed per-head scale in `OutScale`, preserves the source
`head_num/head_dim` mask, and proves the destination-indexed value writeback in
VeriTile's real-tile arithmetic layer. -/
```
```lean
def destindex_copy_quantize_kv_transform_value_store_slice
    (K : RegionName) (DestLoc : Region .nat) (Out OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_d
      stride_o_bs stride_o_h stride_o_d
      stride_os_bs stride_os_h
      head_num head_dim BLOCK_HEAD BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  cur_index = tl.program_id(0)
  offs_h = tl.arange(0, $(BLOCK_HEAD))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  dest_index = tl.load(DestLoc + cur_index)
  mask = (offs_h[:, None] < $(head_num)) & (offs_d[None, :] < $(head_dim))
  src_data = tl.load(K + cur_index * $(stride_k_bs) +
      offs_h[:, None] * $(stride_k_h) + $(stride_k_d) * offs_d[None, :],
    mask=mask, other=0.0)
  data_scale = tl.load(OutScale + dest_index * $(stride_os_bs) +
      $(stride_os_h) * offs_h,
    mask=offs_h < $(head_num), other=1.0)
  q_src_data = src_data / data_scale[:, None]
  tl.store(Out + dest_index * $(stride_o_bs) +
      $(stride_o_h) * offs_h[:, None] + $(stride_o_d) * offs_d[None, :],
    q_src_data, mask=mask)
}
```
</details>

<details><summary><code>active</code></summary>

```lean
def active
    (s : BlockState) (head_num head_dim BLOCK_HEAD BLOCK_DMODEL : Nat)
    (idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL]) : Prop :=
  headIndex s idx.1 < head_num ∧ dimIndex s idx.2.1 < head_dim
```
</details>

<details><summary><code>outOffset</code></summary>

```lean
def outOffset
    (s : BlockState) (DestLoc : RegionName)
    (stride_o_bs stride_o_h stride_o_d : Nat)
    (idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL]) : Nat :=
  destIndex s DestLoc * stride_o_bs + stride_o_h * headIndex s idx.1 +
    stride_o_d * dimIndex s idx.2.1
```
</details>

<details><summary><code>quantizeKvTransformValueSpec</code></summary>

```lean
noncomputable def quantizeKvTransformValueSpec
    (s : BlockState) (K DestLoc OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_d stride_os_bs stride_os_h : Nat)
    (idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL]) : ℝ :=
  s.readMem K (kOffset s stride_k_bs stride_k_h stride_k_d idx) /
    s.readMem OutScale (scaleOffset s DestLoc stride_os_bs stride_os_h idx)
```
</details>

<details><summary><code>destindex_copy_quantize_kv_transform_scale_store_slice</code></summary>

```
/-- Proof-oriented scale-writeback slice of `quantize_kv_transform.py`'s
`_fwd_kernel_destindex_copy_quantize_kv`. Companion to the value-store slice:
covers the 1D `Out_scale` writeback from a precomputed `Scale` tile under the
destination-indexed addressing. -/
```
```lean
def destindex_copy_quantize_kv_transform_scale_store_slice
    (Scale : RegionName) (DestLoc : Region .nat) (OutScale : RegionName)
    (stride_s_bs stride_os_bs stride_os_h head_num BLOCK_HEAD : Nat) :
    ComputeKernel := triton {
  cur_index = tl.program_id(0)
  offs_h = tl.arange(0, $(BLOCK_HEAD))
  dest_index = tl.load(DestLoc + cur_index)
  mask = offs_h < $(head_num)
  data_scale = tl.load(Scale + cur_index * $(stride_s_bs) + offs_h,
    mask=mask, other=0.0)
  tl.store(OutScale + dest_index * $(stride_os_bs) +
      $(stride_os_h) * offs_h,
    data_scale, mask=mask)
}
```
</details>

<details><summary><code>scaleActive</code></summary>

```lean
def scaleActive (head_num BLOCK_HEAD : Nat) (i : Fin BLOCK_HEAD) : Prop :=
  i.val < head_num
```
</details>

<details><summary><code>scaleOutOffset</code></summary>

```lean
def scaleOutOffset
    (s : BlockState) (DestLoc : RegionName)
    (stride_os_bs stride_os_h : Nat) (i : Fin BLOCK_HEAD) : Nat :=
  destIndex s DestLoc * stride_os_bs + stride_os_h * i.val
```
</details>

<details><summary><code>quantizeKvTransformScaleSpec</code></summary>

```lean
noncomputable def quantizeKvTransformScaleSpec
    (s : BlockState) (Scale : RegionName) (stride_s_bs : Nat)
    (i : Fin BLOCK_HEAD) : ℝ :=
  s.readMem Scale (scaleSourceOffset s stride_s_bs i)
```
</details>

<details><summary><code>headIndex</code></summary>

```lean
def headIndex (_s : BlockState) (i : Fin BLOCK_HEAD) : Nat :=
  i.val
```
</details>

<details><summary><code>dimIndex</code></summary>

```lean
def dimIndex (_s : BlockState) (j : Fin BLOCK_DMODEL) : Nat :=
  j.val
```
</details>

<details><summary><code>destIndex</code></summary>

```lean
def destIndex (s : BlockState) (DestLoc : RegionName) : Nat :=
  s.readMemValue .nat DestLoc (s.pids 0)
```
</details>

<details><summary><code>kOffset</code></summary>

```lean
def kOffset
    (s : BlockState) (stride_k_bs stride_k_h stride_k_d : Nat)
    (idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL]) : Nat :=
  s.pids 0 * stride_k_bs + headIndex s idx.1 * stride_k_h +
    stride_k_d * dimIndex s idx.2.1
```
</details>

<details><summary><code>scaleOffset</code></summary>

```lean
def scaleOffset
    (s : BlockState) (DestLoc : RegionName)
    (stride_os_bs stride_os_h : Nat)
    (idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL]) : Nat :=
  destIndex s DestLoc * stride_os_bs + stride_os_h * headIndex s idx.1
```
</details>

<details><summary><code>scaleSourceOffset</code></summary>

```lean
def scaleSourceOffset (s : BlockState) (stride_s_bs : Nat)
    (i : Fin BLOCK_HEAD) : Nat :=
  s.pids 0 * stride_s_bs + i.val
```
</details>

## Public theorem: `destindex_copy_quantize_kv_transform_python_h8_d64_summary`

<details><summary>docstring</summary>

```
/-- Public Python `H = 8, D = 64` summary: the checked shape covers the full
surface syntax and the two externally visible outputs (values and scales). -/
```
</details>

**Statement:**
```lean
theorem destindex_copy_quantize_kv_transform_python_h8_d64_summary
    (K Scale DestLoc Out OutScale : RegionName) (s : BlockState) :
    (∃ alg,
      (destindex_copy_quantize_kv_transform_real_surface K DestLoc Out OutScale
        512 64 1 512 64 1 8 1 1 8 64 64 8).toAlgorithm? =
          Except.ok alg) ∧
    ((ComputeCorrect.Realizes
      (kernel := destindex_copy_quantize_kv_transform_value_store_slice K
        DestLoc Out OutScale 512 64 1 512 64 1 8 1 8 64 8 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 8 64 8 64)
        (fun idx => (Out, outOffset s DestLoc 512 64 1 idx)))
      (expected := fun idx =>
        quantizeKvTransformValueSpec s K DestLoc OutScale 512 64 1 8 1 idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := destindex_copy_quantize_kv_transform_scale_store_slice Scale
        DestLoc OutScale 8 8 1 8 8)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (scaleActive 8 8)
        (fun i => (OutScale, scaleOutOffset s DestLoc 8 1 i)))
      (expected := fun i =>
        quantizeKvTransformScaleSpec s Scale 8 i)))
```

**Assumptions / layout contracts:**
- `kernel : = destindex_copy_quantize_kv_transform_value_store_slice K
        DestLoc Out OutScale 512 64 1 512 64 1 8 1 8 64 8 64`
- `initialState : = s`
- `expected : = fun idx =>
        quantizeKvTransformValueSpec s K DestLoc OutScale 512 64 1 8 1 idx`
- `kernel : = destindex_copy_quantize_kv_transform_scale_store_slice Scale
        DestLoc OutScale 8 8 1 8 8`
- `initialState : = s`
- `expected : = fun i =>
        quantizeKvTransformScaleSpec s Scale 8 i`

**Closed-form spec defs (transitive):** `destindex_copy_quantize_kv_transform_real_surface`, `destindex_copy_quantize_kv_transform_value_store_slice`, `active`, `outOffset`, `quantizeKvTransformValueSpec`, `destindex_copy_quantize_kv_transform_scale_store_slice`, `scaleActive`, `scaleOutOffset`, `quantizeKvTransformScaleSpec`, `headIndex`, `dimIndex`, `destIndex`, `kOffset`, `scaleOffset`, `scaleSourceOffset`

<details><summary><code>destindex_copy_quantize_kv_transform_real_surface</code></summary>

```
/-- Real-valued surface of `quantize_kv_transform.py`'s
`_fwd_kernel_destindex_copy_quantize_kv`.

This preserves destination-indexed addressing, the `head_num/head_dim` mask,
`tl.abs`, per-head scale computation, value writeback, and scale writeback. The
Python kernel casts the scale to `OutScale.dtype.element_ty`; that cast is
represented explicitly. The final quotient cast to int8 is preserved as a
surface dtype annotation and lowers through the DSL's fixed-width cast
surface. -/
```
```lean
def destindex_copy_quantize_kv_transform_real_surface
    (K : RegionName) (DestLoc : Region .nat) (Out OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_d
      stride_o_bs stride_o_h stride_o_d
      stride_os_bs stride_os_h _stride_os_d
      head_num head_dim BLOCK_DMODEL BLOCK_HEAD : Nat) :
    ComputeKernel := triton {
  cur_index = tl.program_id(0)
  offs_h = tl.arange(0, $(BLOCK_HEAD))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  dest_index = tl.load(DestLoc + cur_index)
  src_data = tl.load(K + cur_index * $(stride_k_bs) +
      offs_h[:, None] * $(stride_k_h) + $(stride_k_d) * offs_d[None, :],
    mask=(offs_h[:, None] < $(head_num)) & (offs_d[None, :] < $(head_dim)),
    other=0.0)
  abs_data = tl.abs(src_data)
  data_scale = ((tl.max(abs_data, axis=1) / 127.0).to(OutScale.dtype.element_ty))[:, None]
  q_src_data = (src_data / data_scale).to(tl.int8)
  o_ptrs = Out + dest_index * $(stride_o_bs) +
    $(stride_o_h) * offs_h[:, None] + $(stride_o_d) * offs_d[None, :]
  os_ptrs = OutScale + dest_index * $(stride_os_bs) + $(stride_os_h) * offs_h[:, None]
  tl.store(o_ptrs, q_src_data,
    mask=(offs_h[:, None] < $(head_num)) & (offs_d[None, :] < $(head_dim)))
  tl.store(os_ptrs, data_scale, mask=offs_h[:, None] < $(head_num))
}
```
</details>

<details><summary><code>destindex_copy_quantize_kv_transform_value_store_slice</code></summary>

```
/-- Proof-oriented q-value writeback slice of `quantize_kv_transform.py`'s
`_fwd_kernel_destindex_copy_quantize_kv`.

The full kernel computes a per-head scale from `max(abs(src_data))`, casts the
quotient to int8, and stores both values and scales. This slice starts from a
precomputed per-head scale in `OutScale`, preserves the source
`head_num/head_dim` mask, and proves the destination-indexed value writeback in
VeriTile's real-tile arithmetic layer. -/
```
```lean
def destindex_copy_quantize_kv_transform_value_store_slice
    (K : RegionName) (DestLoc : Region .nat) (Out OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_d
      stride_o_bs stride_o_h stride_o_d
      stride_os_bs stride_os_h
      head_num head_dim BLOCK_HEAD BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  cur_index = tl.program_id(0)
  offs_h = tl.arange(0, $(BLOCK_HEAD))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  dest_index = tl.load(DestLoc + cur_index)
  mask = (offs_h[:, None] < $(head_num)) & (offs_d[None, :] < $(head_dim))
  src_data = tl.load(K + cur_index * $(stride_k_bs) +
      offs_h[:, None] * $(stride_k_h) + $(stride_k_d) * offs_d[None, :],
    mask=mask, other=0.0)
  data_scale = tl.load(OutScale + dest_index * $(stride_os_bs) +
      $(stride_os_h) * offs_h,
    mask=offs_h < $(head_num), other=1.0)
  q_src_data = src_data / data_scale[:, None]
  tl.store(Out + dest_index * $(stride_o_bs) +
      $(stride_o_h) * offs_h[:, None] + $(stride_o_d) * offs_d[None, :],
    q_src_data, mask=mask)
}
```
</details>

<details><summary><code>active</code></summary>

```lean
def active
    (s : BlockState) (head_num head_dim BLOCK_HEAD BLOCK_DMODEL : Nat)
    (idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL]) : Prop :=
  headIndex s idx.1 < head_num ∧ dimIndex s idx.2.1 < head_dim
```
</details>

<details><summary><code>outOffset</code></summary>

```lean
def outOffset
    (s : BlockState) (DestLoc : RegionName)
    (stride_o_bs stride_o_h stride_o_d : Nat)
    (idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL]) : Nat :=
  destIndex s DestLoc * stride_o_bs + stride_o_h * headIndex s idx.1 +
    stride_o_d * dimIndex s idx.2.1
```
</details>

<details><summary><code>quantizeKvTransformValueSpec</code></summary>

```lean
noncomputable def quantizeKvTransformValueSpec
    (s : BlockState) (K DestLoc OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_d stride_os_bs stride_os_h : Nat)
    (idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL]) : ℝ :=
  s.readMem K (kOffset s stride_k_bs stride_k_h stride_k_d idx) /
    s.readMem OutScale (scaleOffset s DestLoc stride_os_bs stride_os_h idx)
```
</details>

<details><summary><code>destindex_copy_quantize_kv_transform_scale_store_slice</code></summary>

```
/-- Proof-oriented scale-writeback slice of `quantize_kv_transform.py`'s
`_fwd_kernel_destindex_copy_quantize_kv`. Companion to the value-store slice:
covers the 1D `Out_scale` writeback from a precomputed `Scale` tile under the
destination-indexed addressing. -/
```
```lean
def destindex_copy_quantize_kv_transform_scale_store_slice
    (Scale : RegionName) (DestLoc : Region .nat) (OutScale : RegionName)
    (stride_s_bs stride_os_bs stride_os_h head_num BLOCK_HEAD : Nat) :
    ComputeKernel := triton {
  cur_index = tl.program_id(0)
  offs_h = tl.arange(0, $(BLOCK_HEAD))
  dest_index = tl.load(DestLoc + cur_index)
  mask = offs_h < $(head_num)
  data_scale = tl.load(Scale + cur_index * $(stride_s_bs) + offs_h,
    mask=mask, other=0.0)
  tl.store(OutScale + dest_index * $(stride_os_bs) +
      $(stride_os_h) * offs_h,
    data_scale, mask=mask)
}
```
</details>

<details><summary><code>scaleActive</code></summary>

```lean
def scaleActive (head_num BLOCK_HEAD : Nat) (i : Fin BLOCK_HEAD) : Prop :=
  i.val < head_num
```
</details>

<details><summary><code>scaleOutOffset</code></summary>

```lean
def scaleOutOffset
    (s : BlockState) (DestLoc : RegionName)
    (stride_os_bs stride_os_h : Nat) (i : Fin BLOCK_HEAD) : Nat :=
  destIndex s DestLoc * stride_os_bs + stride_os_h * i.val
```
</details>

<details><summary><code>quantizeKvTransformScaleSpec</code></summary>

```lean
noncomputable def quantizeKvTransformScaleSpec
    (s : BlockState) (Scale : RegionName) (stride_s_bs : Nat)
    (i : Fin BLOCK_HEAD) : ℝ :=
  s.readMem Scale (scaleSourceOffset s stride_s_bs i)
```
</details>

<details><summary><code>headIndex</code></summary>

```lean
def headIndex (_s : BlockState) (i : Fin BLOCK_HEAD) : Nat :=
  i.val
```
</details>

<details><summary><code>dimIndex</code></summary>

```lean
def dimIndex (_s : BlockState) (j : Fin BLOCK_DMODEL) : Nat :=
  j.val
```
</details>

<details><summary><code>destIndex</code></summary>

```lean
def destIndex (s : BlockState) (DestLoc : RegionName) : Nat :=
  s.readMemValue .nat DestLoc (s.pids 0)
```
</details>

<details><summary><code>kOffset</code></summary>

```lean
def kOffset
    (s : BlockState) (stride_k_bs stride_k_h stride_k_d : Nat)
    (idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL]) : Nat :=
  s.pids 0 * stride_k_bs + headIndex s idx.1 * stride_k_h +
    stride_k_d * dimIndex s idx.2.1
```
</details>

<details><summary><code>scaleOffset</code></summary>

```lean
def scaleOffset
    (s : BlockState) (DestLoc : RegionName)
    (stride_os_bs stride_os_h : Nat)
    (idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL]) : Nat :=
  destIndex s DestLoc * stride_os_bs + stride_os_h * headIndex s idx.1
```
</details>

<details><summary><code>scaleSourceOffset</code></summary>

```lean
def scaleSourceOffset (s : BlockState) (stride_s_bs : Nat)
    (i : Fin BLOCK_HEAD) : Nat :=
  s.pids 0 * stride_s_bs + i.val
```
</details>

## Public theorem: `destindex_copy_quantize_kv_transform_python_h1_d1_summary`

<details><summary>docstring</summary>

```
/-- Public Python `H = 1, D = 1` summary: the checked shape covers the full
surface syntax and the two externally visible outputs (values and scales). -/
```
</details>

**Statement:**
```lean
theorem destindex_copy_quantize_kv_transform_python_h1_d1_summary
    (K Scale DestLoc Out OutScale : RegionName) (s : BlockState) :
    (∃ alg,
      (destindex_copy_quantize_kv_transform_real_surface K DestLoc Out OutScale
        1 1 1 1 1 1 1 1 1 1 1 1 1).toAlgorithm? =
          Except.ok alg) ∧
    ((ComputeCorrect.Realizes
      (kernel := destindex_copy_quantize_kv_transform_value_store_slice K
        DestLoc Out OutScale 1 1 1 1 1 1 1 1 1 1 1 1)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 1 1 1 1)
        (fun idx => (Out, outOffset s DestLoc 1 1 1 idx)))
      (expected := fun idx =>
        quantizeKvTransformValueSpec s K DestLoc OutScale 1 1 1 1 1 idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := destindex_copy_quantize_kv_transform_scale_store_slice Scale
        DestLoc OutScale 1 1 1 1 1)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (scaleActive 1 1)
        (fun i => (OutScale, scaleOutOffset s DestLoc 1 1 i)))
      (expected := fun i =>
        quantizeKvTransformScaleSpec s Scale 1 i)))
```

**Assumptions / layout contracts:**
- `kernel : = destindex_copy_quantize_kv_transform_value_store_slice K
        DestLoc Out OutScale 1 1 1 1 1 1 1 1 1 1 1 1`
- `initialState : = s`
- `expected : = fun idx =>
        quantizeKvTransformValueSpec s K DestLoc OutScale 1 1 1 1 1 idx`
- `kernel : = destindex_copy_quantize_kv_transform_scale_store_slice Scale
        DestLoc OutScale 1 1 1 1 1`
- `initialState : = s`
- `expected : = fun i =>
        quantizeKvTransformScaleSpec s Scale 1 i`

**Closed-form spec defs (transitive):** `destindex_copy_quantize_kv_transform_real_surface`, `destindex_copy_quantize_kv_transform_value_store_slice`, `active`, `outOffset`, `quantizeKvTransformValueSpec`, `destindex_copy_quantize_kv_transform_scale_store_slice`, `scaleActive`, `scaleOutOffset`, `quantizeKvTransformScaleSpec`, `headIndex`, `dimIndex`, `destIndex`, `kOffset`, `scaleOffset`, `scaleSourceOffset`

<details><summary><code>destindex_copy_quantize_kv_transform_real_surface</code></summary>

```
/-- Real-valued surface of `quantize_kv_transform.py`'s
`_fwd_kernel_destindex_copy_quantize_kv`.

This preserves destination-indexed addressing, the `head_num/head_dim` mask,
`tl.abs`, per-head scale computation, value writeback, and scale writeback. The
Python kernel casts the scale to `OutScale.dtype.element_ty`; that cast is
represented explicitly. The final quotient cast to int8 is preserved as a
surface dtype annotation and lowers through the DSL's fixed-width cast
surface. -/
```
```lean
def destindex_copy_quantize_kv_transform_real_surface
    (K : RegionName) (DestLoc : Region .nat) (Out OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_d
      stride_o_bs stride_o_h stride_o_d
      stride_os_bs stride_os_h _stride_os_d
      head_num head_dim BLOCK_DMODEL BLOCK_HEAD : Nat) :
    ComputeKernel := triton {
  cur_index = tl.program_id(0)
  offs_h = tl.arange(0, $(BLOCK_HEAD))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  dest_index = tl.load(DestLoc + cur_index)
  src_data = tl.load(K + cur_index * $(stride_k_bs) +
      offs_h[:, None] * $(stride_k_h) + $(stride_k_d) * offs_d[None, :],
    mask=(offs_h[:, None] < $(head_num)) & (offs_d[None, :] < $(head_dim)),
    other=0.0)
  abs_data = tl.abs(src_data)
  data_scale = ((tl.max(abs_data, axis=1) / 127.0).to(OutScale.dtype.element_ty))[:, None]
  q_src_data = (src_data / data_scale).to(tl.int8)
  o_ptrs = Out + dest_index * $(stride_o_bs) +
    $(stride_o_h) * offs_h[:, None] + $(stride_o_d) * offs_d[None, :]
  os_ptrs = OutScale + dest_index * $(stride_os_bs) + $(stride_os_h) * offs_h[:, None]
  tl.store(o_ptrs, q_src_data,
    mask=(offs_h[:, None] < $(head_num)) & (offs_d[None, :] < $(head_dim)))
  tl.store(os_ptrs, data_scale, mask=offs_h[:, None] < $(head_num))
}
```
</details>

<details><summary><code>destindex_copy_quantize_kv_transform_value_store_slice</code></summary>

```
/-- Proof-oriented q-value writeback slice of `quantize_kv_transform.py`'s
`_fwd_kernel_destindex_copy_quantize_kv`.

The full kernel computes a per-head scale from `max(abs(src_data))`, casts the
quotient to int8, and stores both values and scales. This slice starts from a
precomputed per-head scale in `OutScale`, preserves the source
`head_num/head_dim` mask, and proves the destination-indexed value writeback in
VeriTile's real-tile arithmetic layer. -/
```
```lean
def destindex_copy_quantize_kv_transform_value_store_slice
    (K : RegionName) (DestLoc : Region .nat) (Out OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_d
      stride_o_bs stride_o_h stride_o_d
      stride_os_bs stride_os_h
      head_num head_dim BLOCK_HEAD BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  cur_index = tl.program_id(0)
  offs_h = tl.arange(0, $(BLOCK_HEAD))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  dest_index = tl.load(DestLoc + cur_index)
  mask = (offs_h[:, None] < $(head_num)) & (offs_d[None, :] < $(head_dim))
  src_data = tl.load(K + cur_index * $(stride_k_bs) +
      offs_h[:, None] * $(stride_k_h) + $(stride_k_d) * offs_d[None, :],
    mask=mask, other=0.0)
  data_scale = tl.load(OutScale + dest_index * $(stride_os_bs) +
      $(stride_os_h) * offs_h,
    mask=offs_h < $(head_num), other=1.0)
  q_src_data = src_data / data_scale[:, None]
  tl.store(Out + dest_index * $(stride_o_bs) +
      $(stride_o_h) * offs_h[:, None] + $(stride_o_d) * offs_d[None, :],
    q_src_data, mask=mask)
}
```
</details>

<details><summary><code>active</code></summary>

```lean
def active
    (s : BlockState) (head_num head_dim BLOCK_HEAD BLOCK_DMODEL : Nat)
    (idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL]) : Prop :=
  headIndex s idx.1 < head_num ∧ dimIndex s idx.2.1 < head_dim
```
</details>

<details><summary><code>outOffset</code></summary>

```lean
def outOffset
    (s : BlockState) (DestLoc : RegionName)
    (stride_o_bs stride_o_h stride_o_d : Nat)
    (idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL]) : Nat :=
  destIndex s DestLoc * stride_o_bs + stride_o_h * headIndex s idx.1 +
    stride_o_d * dimIndex s idx.2.1
```
</details>

<details><summary><code>quantizeKvTransformValueSpec</code></summary>

```lean
noncomputable def quantizeKvTransformValueSpec
    (s : BlockState) (K DestLoc OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_d stride_os_bs stride_os_h : Nat)
    (idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL]) : ℝ :=
  s.readMem K (kOffset s stride_k_bs stride_k_h stride_k_d idx) /
    s.readMem OutScale (scaleOffset s DestLoc stride_os_bs stride_os_h idx)
```
</details>

<details><summary><code>destindex_copy_quantize_kv_transform_scale_store_slice</code></summary>

```
/-- Proof-oriented scale-writeback slice of `quantize_kv_transform.py`'s
`_fwd_kernel_destindex_copy_quantize_kv`. Companion to the value-store slice:
covers the 1D `Out_scale` writeback from a precomputed `Scale` tile under the
destination-indexed addressing. -/
```
```lean
def destindex_copy_quantize_kv_transform_scale_store_slice
    (Scale : RegionName) (DestLoc : Region .nat) (OutScale : RegionName)
    (stride_s_bs stride_os_bs stride_os_h head_num BLOCK_HEAD : Nat) :
    ComputeKernel := triton {
  cur_index = tl.program_id(0)
  offs_h = tl.arange(0, $(BLOCK_HEAD))
  dest_index = tl.load(DestLoc + cur_index)
  mask = offs_h < $(head_num)
  data_scale = tl.load(Scale + cur_index * $(stride_s_bs) + offs_h,
    mask=mask, other=0.0)
  tl.store(OutScale + dest_index * $(stride_os_bs) +
      $(stride_os_h) * offs_h,
    data_scale, mask=mask)
}
```
</details>

<details><summary><code>scaleActive</code></summary>

```lean
def scaleActive (head_num BLOCK_HEAD : Nat) (i : Fin BLOCK_HEAD) : Prop :=
  i.val < head_num
```
</details>

<details><summary><code>scaleOutOffset</code></summary>

```lean
def scaleOutOffset
    (s : BlockState) (DestLoc : RegionName)
    (stride_os_bs stride_os_h : Nat) (i : Fin BLOCK_HEAD) : Nat :=
  destIndex s DestLoc * stride_os_bs + stride_os_h * i.val
```
</details>

<details><summary><code>quantizeKvTransformScaleSpec</code></summary>

```lean
noncomputable def quantizeKvTransformScaleSpec
    (s : BlockState) (Scale : RegionName) (stride_s_bs : Nat)
    (i : Fin BLOCK_HEAD) : ℝ :=
  s.readMem Scale (scaleSourceOffset s stride_s_bs i)
```
</details>

<details><summary><code>headIndex</code></summary>

```lean
def headIndex (_s : BlockState) (i : Fin BLOCK_HEAD) : Nat :=
  i.val
```
</details>

<details><summary><code>dimIndex</code></summary>

```lean
def dimIndex (_s : BlockState) (j : Fin BLOCK_DMODEL) : Nat :=
  j.val
```
</details>

<details><summary><code>destIndex</code></summary>

```lean
def destIndex (s : BlockState) (DestLoc : RegionName) : Nat :=
  s.readMemValue .nat DestLoc (s.pids 0)
```
</details>

<details><summary><code>kOffset</code></summary>

```lean
def kOffset
    (s : BlockState) (stride_k_bs stride_k_h stride_k_d : Nat)
    (idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL]) : Nat :=
  s.pids 0 * stride_k_bs + headIndex s idx.1 * stride_k_h +
    stride_k_d * dimIndex s idx.2.1
```
</details>

<details><summary><code>scaleOffset</code></summary>

```lean
def scaleOffset
    (s : BlockState) (DestLoc : RegionName)
    (stride_os_bs stride_os_h : Nat)
    (idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL]) : Nat :=
  destIndex s DestLoc * stride_os_bs + stride_os_h * headIndex s idx.1
```
</details>

<details><summary><code>scaleSourceOffset</code></summary>

```lean
def scaleSourceOffset (s : BlockState) (stride_s_bs : Nat)
    (i : Fin BLOCK_HEAD) : Nat :=
  s.pids 0 * stride_s_bs + i.val
```
</details>

## Public theorem: `destindex_copy_quantize_kv_transform_python_h12_d96_output_summary`

<details><summary>docstring</summary>

```
/-- **Public Python `H = 12, D = 96` summary (genuine, gap-free).** The checked
Python shape (`H = 12`, `D = 96`, `BLOCK_HEAD = 16`, `BLOCK_DMODEL = 128`, K/Out
strides `(1152, 96, 1)`, `Out_scale` strides `(12, 1, 1)`) covers the full
faithful surface syntax (the combined head-and-dim mask, `tl.abs`,
`tl.max(..., axis=1)`, the `element_ty` scale cast, the int8 quotient cast, the
destination-indexed masked stores) and the two externally visible outputs:

* **value**: `Out[dest_index, h, d]` (read back via `readMemValue .int`) equals
  `(K[cur_index, h, d] / (max(|K[cur_index,h,·<head_dim]|)/127)).to(int8)` on
  active lanes (`h < head_num ∧ d < head_dim`);
* **scale**: `Out_scale[dest_index, h]` (read back via `readMem`) equals the real
  value of `max(|K[cur_index,h,·<head_dim]|)/127` on active heads, unchanged
  otherwise.

Both expected values are computed from the kernel **inputs**, so this summary is
not self-referential. The value output uses the active-lane variant because the
`BLOCK_DMODEL = 128 > head_dim = 96` padding makes the full tile offset map
non-injective. `hOut : Out ≠ OutScale` is the region-distinctness no-aliasing
hypothesis. -/
```
</details>

**Statement:**
```lean
theorem destindex_copy_quantize_kv_transform_python_h12_d96_output_summary
    (K DestLoc Out OutScale : RegionName) (s : BlockState) (hOut : Out ≠ OutScale) :
    (∃ alg,
      (destindex_copy_quantize_kv_transform_real_surface K DestLoc Out OutScale
        1152 96 1 1152 96 1 12 1 1 12 96 128 16).toAlgorithm? =
          Except.ok alg) ∧
    (∀ idx : TileIndex [16, 128],
      active s 12 96 16 128 idx →
      (exec (destindex_copy_quantize_kv_transform_real_surface K DestLoc Out OutScale
            1152 96 1 1152 96 1 12 1 1 12 96 128 16) s).map
          (·.readMemValue .int Out (outOffset s DestLoc 1152 96 1 idx))
        = some (quantizeKvTransformSurfaceIntValue s K 1152 96 1 12 96 128 (by norm_num) idx)) ∧
    (∀ i : Fin 16,
      (exec (destindex_copy_quantize_kv_transform_real_surface K DestLoc Out OutScale
            1152 96 1 1152 96 1 12 1 1 12 96 128 16) s).map
          (·.readMem OutScale (scaleOutOffset s DestLoc 12 1 i))
        = some (if scaleActive 12 16 i then
            quantizeKvTransformScaleCell s K 1152 96 1 12 96 128 (by norm_num) i.val
          else s.readMem OutScale (scaleOutOffset s DestLoc 12 1 i)))
```

**Assumptions / layout contracts:**
- `hOut : Out ≠ OutScale`

**Closed-form spec defs (transitive):** `destindex_copy_quantize_kv_transform_real_surface`, `active`, `outOffset`, `quantizeKvTransformSurfaceIntValue`, `scaleOutOffset`, `scaleActive`, `quantizeKvTransformScaleCell`, `headIndex`, `dimIndex`, `destIndex`, `maskedSrc`, `quantizeKvTransformScaleValue`

<details><summary><code>destindex_copy_quantize_kv_transform_real_surface</code></summary>

```
/-- Real-valued surface of `quantize_kv_transform.py`'s
`_fwd_kernel_destindex_copy_quantize_kv`.

This preserves destination-indexed addressing, the `head_num/head_dim` mask,
`tl.abs`, per-head scale computation, value writeback, and scale writeback. The
Python kernel casts the scale to `OutScale.dtype.element_ty`; that cast is
represented explicitly. The final quotient cast to int8 is preserved as a
surface dtype annotation and lowers through the DSL's fixed-width cast
surface. -/
```
```lean
def destindex_copy_quantize_kv_transform_real_surface
    (K : RegionName) (DestLoc : Region .nat) (Out OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_d
      stride_o_bs stride_o_h stride_o_d
      stride_os_bs stride_os_h _stride_os_d
      head_num head_dim BLOCK_DMODEL BLOCK_HEAD : Nat) :
    ComputeKernel := triton {
  cur_index = tl.program_id(0)
  offs_h = tl.arange(0, $(BLOCK_HEAD))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  dest_index = tl.load(DestLoc + cur_index)
  src_data = tl.load(K + cur_index * $(stride_k_bs) +
      offs_h[:, None] * $(stride_k_h) + $(stride_k_d) * offs_d[None, :],
    mask=(offs_h[:, None] < $(head_num)) & (offs_d[None, :] < $(head_dim)),
    other=0.0)
  abs_data = tl.abs(src_data)
  data_scale = ((tl.max(abs_data, axis=1) / 127.0).to(OutScale.dtype.element_ty))[:, None]
  q_src_data = (src_data / data_scale).to(tl.int8)
  o_ptrs = Out + dest_index * $(stride_o_bs) +
    $(stride_o_h) * offs_h[:, None] + $(stride_o_d) * offs_d[None, :]
  os_ptrs = OutScale + dest_index * $(stride_os_bs) + $(stride_os_h) * offs_h[:, None]
  tl.store(o_ptrs, q_src_data,
    mask=(offs_h[:, None] < $(head_num)) & (offs_d[None, :] < $(head_dim)))
  tl.store(os_ptrs, data_scale, mask=offs_h[:, None] < $(head_num))
}
```
</details>

<details><summary><code>active</code></summary>

```lean
def active
    (s : BlockState) (head_num head_dim BLOCK_HEAD BLOCK_DMODEL : Nat)
    (idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL]) : Prop :=
  headIndex s idx.1 < head_num ∧ dimIndex s idx.2.1 < head_dim
```
</details>

<details><summary><code>outOffset</code></summary>

```lean
def outOffset
    (s : BlockState) (DestLoc : RegionName)
    (stride_o_bs stride_o_h stride_o_d : Nat)
    (idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL]) : Nat :=
  destIndex s DestLoc * stride_o_bs + stride_o_h * headIndex s idx.1 +
    stride_o_d * dimIndex s idx.2.1
```
</details>

<details><summary><code>quantizeKvTransformSurfaceIntValue</code></summary>

```
/-- Genuine quantized value spec (`(src / data_scale).to(int8)`): the int8 cast
of the masked source lane divided by the per-head scale. -/
```
```lean
noncomputable def quantizeKvTransformSurfaceIntValue
    (s : BlockState) (K : RegionName)
    (stride_k_bs stride_k_h stride_k_d head_num head_dim BLOCK_DMODEL : Nat)
    (hD : 0 < BLOCK_DMODEL) (idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL]) : Int :=
  WithBot.realToInt8
    (FloatDType.real.cast FloatDType.real
      (Option.map₂ (fun x1 x2 => x1 / x2)
        (maskedSrc s K stride_k_bs stride_k_h stride_k_d head_num head_dim
          idx.1.val idx.2.1.val)
        (quantizeKvTransformScaleValue s K stride_k_bs stride_k_h stride_k_d
          head_num head_dim BLOCK_DMODEL hD idx.1.val)))
```
</details>

<details><summary><code>scaleOutOffset</code></summary>

```lean
def scaleOutOffset
    (s : BlockState) (DestLoc : RegionName)
    (stride_os_bs stride_os_h : Nat) (i : Fin BLOCK_HEAD) : Nat :=
  destIndex s DestLoc * stride_os_bs + stride_os_h * i.val
```
</details>

<details><summary><code>scaleActive</code></summary>

```lean
def scaleActive (head_num BLOCK_HEAD : Nat) (i : Fin BLOCK_HEAD) : Prop :=
  i.val < head_num
```
</details>

<details><summary><code>quantizeKvTransformScaleCell</code></summary>

```
/-- Genuine scale store cell (`data_scale`): the real value the kernel writes to
`Out_scale`, observed through `readMem`. -/
```
```lean
noncomputable def quantizeKvTransformScaleCell
    (s : BlockState) (K : RegionName)
    (stride_k_bs stride_k_h stride_k_d head_num head_dim BLOCK_DMODEL : Nat)
    (hD : 0 < BLOCK_DMODEL) (h : Nat) : ℝ :=
  WithBot.unbotD 0
    (quantizeKvTransformScaleValue s K stride_k_bs stride_k_h stride_k_d
      head_num head_dim BLOCK_DMODEL hD h)
```
</details>

<details><summary><code>headIndex</code></summary>

```lean
def headIndex (_s : BlockState) (i : Fin BLOCK_HEAD) : Nat :=
  i.val
```
</details>

<details><summary><code>dimIndex</code></summary>

```lean
def dimIndex (_s : BlockState) (j : Fin BLOCK_DMODEL) : Nat :=
  j.val
```
</details>

<details><summary><code>destIndex</code></summary>

```lean
def destIndex (s : BlockState) (DestLoc : RegionName) : Nat :=
  s.readMemValue .nat DestLoc (s.pids 0)
```
</details>

<details><summary><code>maskedSrc</code></summary>

```
/-- The masked source lane value (`tl.load(..., other=0.0)`): the head- and
dim-masked `K[cur_index, h, d]` as a `WithBot ℝ`. -/
```
```lean
noncomputable def maskedSrc
    (s : BlockState) (K : RegionName)
    (stride_k_bs stride_k_h stride_k_d head_num head_dim : Nat)
    (h d : Nat) : WithBot ℝ :=
  if h < head_num ∧ d < head_dim then
    some (s.readMem K (s.pids 0 * stride_k_bs + h * stride_k_h + stride_k_d * d))
  else some (0.0 : ℝ)
```
</details>

<details><summary><code>quantizeKvTransformScaleValue</code></summary>

```
/-- The per-head `data_scale` *value* (`max(|src|, axis=1) / 127`): the row
reduce-max of `|maskedSrc|` divided by `127`, as a real value. The
`.to(OutScale.dtype.element_ty)` cast in the surface is the identity over `ℝ`. -/
```
```lean
noncomputable def quantizeKvTransformScaleValue
    (s : BlockState) (K : RegionName)
    (stride_k_bs stride_k_h stride_k_d head_num head_dim BLOCK_DMODEL : Nat)
    (hD : 0 < BLOCK_DMODEL) (h : Nat) : WithBot ℝ :=
  Option.map (· / 127.0)
    ((Finset.univ.sup'
        (⟨⟨0, hD⟩, Finset.mem_univ _⟩ : (Finset.univ : Finset (Fin BLOCK_DMODEL)).Nonempty)
        (fun x : Fin BLOCK_DMODEL =>
          if maskedSrc s K stride_k_bs stride_k_h stride_k_d head_num head_dim h x.val
              < (some 0 : WithBot ℝ) then
            NumericDType.real.sub (some 0)
              (maskedSrc s K stride_k_bs stride_k_h stride_k_d head_num head_dim h x.val)
          else maskedSrc s K stride_k_bs stride_k_h stride_k_d head_num head_dim h x.val) :
      WithBot ℝ))
```
</details>

## Also present (pinned special-case summaries)
- `destindex_copy_quantize_kv_transform_value_store_slice_compute_correct`
- `destindex_copy_quantize_kv_transform_value_store_slice_active_compute_correct`
- `destindex_copy_quantize_kv_transform_scale_store_slice_compute_correct`
- `destindex_copy_quantize_kv_transform_real_surface_value_output_compute_correct`
- `destindex_copy_quantize_kv_transform_real_surface_active_value_output_compute_correct`
- `destindex_copy_quantize_kv_transform_real_surface_scale_output_compute_correct`
- `destindex_copy_quantize_kv_transform_python_h12_d96_value_store_compute_correct`
- `destindex_copy_quantize_kv_transform_python_h8_d64_value_store_compute_correct`
- `destindex_copy_quantize_kv_transform_python_h1_d1_value_store_compute_correct`
- `destindex_copy_quantize_kv_transform_python_h12_scale_store_compute_correct`
- `destindex_copy_quantize_kv_transform_python_h8_scale_store_compute_correct`
- `destindex_copy_quantize_kv_transform_python_h1_scale_store_compute_correct`
- `destindex_copy_quantize_kv_transform_python_h12_d96_all_outputs_compute_correct`
- `destindex_copy_quantize_kv_transform_python_h8_d64_all_outputs_compute_correct`
- `destindex_copy_quantize_kv_transform_python_h1_d1_all_outputs_compute_correct`
