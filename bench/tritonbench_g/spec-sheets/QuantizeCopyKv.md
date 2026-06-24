# Spec sheet — `bench/tritonbench_g/quantize_copy_kv/QuantizeCopyKv.lean`

**Python source:** `bench/tritonbench_g/quantize_copy_kv/quantize_copy_kv.py`

## Public theorem: `destindex_copy_quantize_kv_python_d64_summary`

<details><summary>docstring</summary>

```
/-- Public Python `D = 64` summary: the checked Python shape covers the full
surface syntax and the two externally visible outputs (values and scales). -/
```
</details>

**Statement:**
```lean
theorem destindex_copy_quantize_kv_python_d64_summary
    (K Scale DestLoc Out OutScale : RegionName) (s : BlockState) :
    (∃ alg,
      (destindex_copy_quantize_kv_real_surface K DestLoc Out OutScale
        512 64 1 512 64 1 8 1 1 8 64 8).toAlgorithm? =
          Except.ok alg) ∧
    ((ComputeCorrect.Realizes
      (kernel := destindex_copy_quantize_kv_value_store_slice K DestLoc Out
        OutScale 512 64 1 512 64 1 8 1 8 8 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 8 8 64)
        (fun idx => (Out, outOffset s DestLoc 512 64 1 idx)))
      (expected := fun idx =>
        quantizeCopyKvValueSpec s K DestLoc OutScale 512 64 1 8 1 idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := destindex_copy_quantize_kv_scale_store_slice Scale DestLoc
        OutScale 8 1 8 1 8 8)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (scaleActive 8 8)
        (fun i => (OutScale, scaleOutOffset1 s DestLoc 8 1 i)))
      (expected := fun i => quantizeCopyKvScaleSpec s Scale 8 1 i)))
```

**Closed-form spec defs (transitive):** `destindex_copy_quantize_kv_real_surface`, `destindex_copy_quantize_kv_value_store_slice`, `active`, `outOffset`, `quantizeCopyKvValueSpec`, `destindex_copy_quantize_kv_scale_store_slice`, `scaleActive`, `scaleOutOffset1`, `quantizeCopyKvScaleSpec`, `headIndex`, `destIndex`, `dimIndex`, `kOffset`, `scaleOffset`, `scaleSourceOffset`

<details><summary><code>destindex_copy_quantize_kv_real_surface</code></summary>

```
/-- Real-valued surface of `quantize_copy_kv.py`'s
`_fwd_kernel_destindex_copy_quantize_kv`.

This preserves destination-indexed addressing, `tl.abs`, the per-head
`tl.max(..., axis=1)` scale computation, value writeback, and scale writeback.
The Python kernel casts the scale to fp16 before broadcasting it and casts the
quotient to int8; both casts are preserved as surface dtype annotations and
lower through the DSL's fixed-width cast surfaces. -/
```
```lean
def destindex_copy_quantize_kv_real_surface
    (K : RegionName) (DestLoc : Region .nat) (Out OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_d
      stride_o_bs stride_o_h stride_o_d
      stride_os_bs stride_os_h _stride_os_d
      head_num BLOCK_DMODEL BLOCK_HEAD : Nat) :
    ComputeKernel := triton {
  cur_index = tl.program_id(0)
  offs_h = tl.arange(0, $(BLOCK_HEAD))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  dest_index = tl.load(DestLoc + cur_index)
  src_data = tl.load(K + cur_index * $(stride_k_bs) +
      offs_h[:, None] * $(stride_k_h) + $(stride_k_d) * offs_d[None, :],
    mask=offs_h[:, None] < $(head_num), other=0.0)
  abs_data = tl.abs(src_data)
  data_scale = ((tl.max(abs_data, axis=1) / 127.0).to(tl.float16))[:, None]
  q_src_data = (src_data / data_scale).to(tl.int8)
  o_ptrs = Out + dest_index * $(stride_o_bs) +
    $(stride_o_h) * offs_h[:, None] + $(stride_o_d) * offs_d[None, :]
  os_ptrs = OutScale + dest_index * $(stride_os_bs) + $(stride_os_h) * offs_h[:, None]
  tl.store(o_ptrs, q_src_data, mask=offs_h[:, None] < $(head_num))
  tl.store(os_ptrs, data_scale, mask=offs_h[:, None] < $(head_num))
}
```
</details>

<details><summary><code>destindex_copy_quantize_kv_value_store_slice</code></summary>

```
/-- Proof-oriented q-value writeback slice of `quantize_copy_kv.py`'s
`_fwd_kernel_destindex_copy_quantize_kv`.

The full kernel computes a per-head `data_scale` with `max(abs(src_data))`,
casts the quotient to int8, and stores both the quantized values and the scale.
VeriTile's current arithmetic layer models real tiles, so this slice starts from
a precomputed per-head scale in `OutScale` and proves the masked destination
indexed value writeback before the fixed-width int8 cast surface. -/
```
```lean
def destindex_copy_quantize_kv_value_store_slice
    (K : RegionName) (DestLoc : Region .nat) (Out OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_d
      stride_o_bs stride_o_h stride_o_d
      stride_os_bs stride_os_h
      head_num BLOCK_HEAD BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  cur_index = tl.program_id(0)
  offs_h = tl.arange(0, $(BLOCK_HEAD))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  dest_index = tl.load(DestLoc + cur_index)
  mask = (offs_h[:, None] < $(head_num)) & (offs_d[None, :] < $(BLOCK_DMODEL))
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
    (s : BlockState) (head_num BLOCK_HEAD BLOCK_DMODEL : Nat)
    (idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL]) : Prop :=
  headIndex s idx.1 < head_num
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

<details><summary><code>quantizeCopyKvValueSpec</code></summary>

```lean
noncomputable def quantizeCopyKvValueSpec
    (s : BlockState) (K DestLoc OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_d stride_os_bs stride_os_h : Nat)
    (idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL]) : ℝ :=
  s.readMem K (kOffset s stride_k_bs stride_k_h stride_k_d idx) /
    s.readMem OutScale (scaleOffset s DestLoc stride_os_bs stride_os_h idx)
```
</details>

<details><summary><code>destindex_copy_quantize_kv_scale_store_slice</code></summary>

```
/-- Proof-oriented scale writeback slice of `quantize_copy_kv.py`'s
`_fwd_kernel_destindex_copy_quantize_kv`.

The full kernel computes `data_scale = max(abs(src_data), axis=1) / 127` and
stores it to `Out_scale` with destination-indexed addressing. This slice starts
from a precomputed per-head `Scale` region and proves the observed scale store
surface independently of the integer value-store cast surface. -/
```
```lean
def destindex_copy_quantize_kv_scale_store_slice
    (Scale : RegionName) (DestLoc : Region .nat) (OutScale : RegionName)
    (stride_s_bs stride_s_h stride_os_bs stride_os_h
      head_num BLOCK_HEAD : Nat) :
    ComputeKernel := triton {
  cur_index = tl.program_id(0)
  offs_h = tl.arange(0, $(BLOCK_HEAD))
  dest_index = tl.load(DestLoc + cur_index)
  mask = offs_h < $(head_num)
  data_scale = tl.load(Scale + cur_index * $(stride_s_bs) +
      offs_h * $(stride_s_h), mask=mask, other=0.0)
  tl.store(OutScale + dest_index * $(stride_os_bs) +
      offs_h * $(stride_os_h), data_scale, mask=mask)
}
```
</details>

<details><summary><code>scaleActive</code></summary>

```lean
def scaleActive (head_num BLOCK_HEAD : Nat) (i : Fin BLOCK_HEAD) : Prop :=
  i.val < head_num
```
</details>

<details><summary><code>scaleOutOffset1</code></summary>

```lean
def scaleOutOffset1
    (s : BlockState) (DestLoc : RegionName)
    (stride_os_bs stride_os_h : Nat) (i : Fin BLOCK_HEAD) : Nat :=
  destIndex s DestLoc * stride_os_bs + i.val * stride_os_h
```
</details>

<details><summary><code>quantizeCopyKvScaleSpec</code></summary>

```lean
noncomputable def quantizeCopyKvScaleSpec
    (s : BlockState) (Scale : RegionName)
    (stride_s_bs stride_s_h : Nat) (i : Fin BLOCK_HEAD) : ℝ :=
  s.readMem Scale (scaleSourceOffset s stride_s_bs stride_s_h i)
```
</details>

<details><summary><code>headIndex</code></summary>

```lean
def headIndex (_s : BlockState) (i : Fin BLOCK_HEAD) : Nat :=
  i.val
```
</details>

<details><summary><code>destIndex</code></summary>

```lean
def destIndex (s : BlockState) (DestLoc : RegionName) : Nat :=
  s.readMemValue .nat DestLoc (s.pids 0)
```
</details>

<details><summary><code>dimIndex</code></summary>

```lean
def dimIndex (_s : BlockState) (j : Fin BLOCK_DMODEL) : Nat :=
  j.val
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
def scaleSourceOffset (s : BlockState) (stride_s_bs stride_s_h : Nat)
    (i : Fin BLOCK_HEAD) : Nat :=
  s.pids 0 * stride_s_bs + i.val * stride_s_h
```
</details>

## Public theorem: `destindex_copy_quantize_kv_python_d256_summary`

<details><summary>docstring</summary>

```
/-- Public Python `D = 256` summary: the checked Python shape covers the full
surface syntax and the two externally visible outputs (values and scales). -/
```
</details>

**Statement:**
```lean
theorem destindex_copy_quantize_kv_python_d256_summary
    (K Scale DestLoc Out OutScale : RegionName) (s : BlockState) :
    (∃ alg,
      (destindex_copy_quantize_kv_real_surface K DestLoc Out OutScale
        2048 256 1 2048 256 1 8 1 1 8 256 8).toAlgorithm? =
          Except.ok alg) ∧
    ((ComputeCorrect.Realizes
      (kernel := destindex_copy_quantize_kv_value_store_slice K DestLoc Out
        OutScale 2048 256 1 2048 256 1 8 1 8 8 256)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 8 8 256)
        (fun idx => (Out, outOffset s DestLoc 2048 256 1 idx)))
      (expected := fun idx =>
        quantizeCopyKvValueSpec s K DestLoc OutScale 2048 256 1 8 1 idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := destindex_copy_quantize_kv_scale_store_slice Scale DestLoc
        OutScale 8 1 8 1 8 8)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (scaleActive 8 8)
        (fun i => (OutScale, scaleOutOffset1 s DestLoc 8 1 i)))
      (expected := fun i => quantizeCopyKvScaleSpec s Scale 8 1 i)))
```

**Closed-form spec defs (transitive):** `destindex_copy_quantize_kv_real_surface`, `destindex_copy_quantize_kv_value_store_slice`, `active`, `outOffset`, `quantizeCopyKvValueSpec`, `destindex_copy_quantize_kv_scale_store_slice`, `scaleActive`, `scaleOutOffset1`, `quantizeCopyKvScaleSpec`, `headIndex`, `destIndex`, `dimIndex`, `kOffset`, `scaleOffset`, `scaleSourceOffset`

<details><summary><code>destindex_copy_quantize_kv_real_surface</code></summary>

```
/-- Real-valued surface of `quantize_copy_kv.py`'s
`_fwd_kernel_destindex_copy_quantize_kv`.

This preserves destination-indexed addressing, `tl.abs`, the per-head
`tl.max(..., axis=1)` scale computation, value writeback, and scale writeback.
The Python kernel casts the scale to fp16 before broadcasting it and casts the
quotient to int8; both casts are preserved as surface dtype annotations and
lower through the DSL's fixed-width cast surfaces. -/
```
```lean
def destindex_copy_quantize_kv_real_surface
    (K : RegionName) (DestLoc : Region .nat) (Out OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_d
      stride_o_bs stride_o_h stride_o_d
      stride_os_bs stride_os_h _stride_os_d
      head_num BLOCK_DMODEL BLOCK_HEAD : Nat) :
    ComputeKernel := triton {
  cur_index = tl.program_id(0)
  offs_h = tl.arange(0, $(BLOCK_HEAD))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  dest_index = tl.load(DestLoc + cur_index)
  src_data = tl.load(K + cur_index * $(stride_k_bs) +
      offs_h[:, None] * $(stride_k_h) + $(stride_k_d) * offs_d[None, :],
    mask=offs_h[:, None] < $(head_num), other=0.0)
  abs_data = tl.abs(src_data)
  data_scale = ((tl.max(abs_data, axis=1) / 127.0).to(tl.float16))[:, None]
  q_src_data = (src_data / data_scale).to(tl.int8)
  o_ptrs = Out + dest_index * $(stride_o_bs) +
    $(stride_o_h) * offs_h[:, None] + $(stride_o_d) * offs_d[None, :]
  os_ptrs = OutScale + dest_index * $(stride_os_bs) + $(stride_os_h) * offs_h[:, None]
  tl.store(o_ptrs, q_src_data, mask=offs_h[:, None] < $(head_num))
  tl.store(os_ptrs, data_scale, mask=offs_h[:, None] < $(head_num))
}
```
</details>

<details><summary><code>destindex_copy_quantize_kv_value_store_slice</code></summary>

```
/-- Proof-oriented q-value writeback slice of `quantize_copy_kv.py`'s
`_fwd_kernel_destindex_copy_quantize_kv`.

The full kernel computes a per-head `data_scale` with `max(abs(src_data))`,
casts the quotient to int8, and stores both the quantized values and the scale.
VeriTile's current arithmetic layer models real tiles, so this slice starts from
a precomputed per-head scale in `OutScale` and proves the masked destination
indexed value writeback before the fixed-width int8 cast surface. -/
```
```lean
def destindex_copy_quantize_kv_value_store_slice
    (K : RegionName) (DestLoc : Region .nat) (Out OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_d
      stride_o_bs stride_o_h stride_o_d
      stride_os_bs stride_os_h
      head_num BLOCK_HEAD BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  cur_index = tl.program_id(0)
  offs_h = tl.arange(0, $(BLOCK_HEAD))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  dest_index = tl.load(DestLoc + cur_index)
  mask = (offs_h[:, None] < $(head_num)) & (offs_d[None, :] < $(BLOCK_DMODEL))
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
    (s : BlockState) (head_num BLOCK_HEAD BLOCK_DMODEL : Nat)
    (idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL]) : Prop :=
  headIndex s idx.1 < head_num
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

<details><summary><code>quantizeCopyKvValueSpec</code></summary>

```lean
noncomputable def quantizeCopyKvValueSpec
    (s : BlockState) (K DestLoc OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_d stride_os_bs stride_os_h : Nat)
    (idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL]) : ℝ :=
  s.readMem K (kOffset s stride_k_bs stride_k_h stride_k_d idx) /
    s.readMem OutScale (scaleOffset s DestLoc stride_os_bs stride_os_h idx)
```
</details>

<details><summary><code>destindex_copy_quantize_kv_scale_store_slice</code></summary>

```
/-- Proof-oriented scale writeback slice of `quantize_copy_kv.py`'s
`_fwd_kernel_destindex_copy_quantize_kv`.

The full kernel computes `data_scale = max(abs(src_data), axis=1) / 127` and
stores it to `Out_scale` with destination-indexed addressing. This slice starts
from a precomputed per-head `Scale` region and proves the observed scale store
surface independently of the integer value-store cast surface. -/
```
```lean
def destindex_copy_quantize_kv_scale_store_slice
    (Scale : RegionName) (DestLoc : Region .nat) (OutScale : RegionName)
    (stride_s_bs stride_s_h stride_os_bs stride_os_h
      head_num BLOCK_HEAD : Nat) :
    ComputeKernel := triton {
  cur_index = tl.program_id(0)
  offs_h = tl.arange(0, $(BLOCK_HEAD))
  dest_index = tl.load(DestLoc + cur_index)
  mask = offs_h < $(head_num)
  data_scale = tl.load(Scale + cur_index * $(stride_s_bs) +
      offs_h * $(stride_s_h), mask=mask, other=0.0)
  tl.store(OutScale + dest_index * $(stride_os_bs) +
      offs_h * $(stride_os_h), data_scale, mask=mask)
}
```
</details>

<details><summary><code>scaleActive</code></summary>

```lean
def scaleActive (head_num BLOCK_HEAD : Nat) (i : Fin BLOCK_HEAD) : Prop :=
  i.val < head_num
```
</details>

<details><summary><code>scaleOutOffset1</code></summary>

```lean
def scaleOutOffset1
    (s : BlockState) (DestLoc : RegionName)
    (stride_os_bs stride_os_h : Nat) (i : Fin BLOCK_HEAD) : Nat :=
  destIndex s DestLoc * stride_os_bs + i.val * stride_os_h
```
</details>

<details><summary><code>quantizeCopyKvScaleSpec</code></summary>

```lean
noncomputable def quantizeCopyKvScaleSpec
    (s : BlockState) (Scale : RegionName)
    (stride_s_bs stride_s_h : Nat) (i : Fin BLOCK_HEAD) : ℝ :=
  s.readMem Scale (scaleSourceOffset s stride_s_bs stride_s_h i)
```
</details>

<details><summary><code>headIndex</code></summary>

```lean
def headIndex (_s : BlockState) (i : Fin BLOCK_HEAD) : Nat :=
  i.val
```
</details>

<details><summary><code>destIndex</code></summary>

```lean
def destIndex (s : BlockState) (DestLoc : RegionName) : Nat :=
  s.readMemValue .nat DestLoc (s.pids 0)
```
</details>

<details><summary><code>dimIndex</code></summary>

```lean
def dimIndex (_s : BlockState) (j : Fin BLOCK_DMODEL) : Nat :=
  j.val
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
def scaleSourceOffset (s : BlockState) (stride_s_bs stride_s_h : Nat)
    (i : Fin BLOCK_HEAD) : Nat :=
  s.pids 0 * stride_s_bs + i.val * stride_s_h
```
</details>

## Public theorem: `destindex_copy_quantize_kv_python_d64_output_summary`

<details><summary>docstring</summary>

```
/-- **Public Python `D = 64` summary (genuine, gap-free).** The checked Python
shape (`H = 8`, `D = 64`, `BLOCK_HEAD = 8`, K/Out strides `(512, 64, 1)`,
`Out_scale` strides `(8, 1, 1)`) covers the full faithful surface syntax
(`tl.abs`, `tl.max(..., axis=1)`, the fp16 scale cast, the int8 quotient cast,
the destination-indexed masked stores) and the two externally visible outputs:

* **value**: `Out[dest_index, h, d]` (read back via `readMemValue .int`) equals
  `(K[cur_index, h, d] / fp16(max(|K[cur_index,h,·]|)/127)).to(int8)` on active
  heads (`h < head_num`), and is unchanged otherwise;
* **scale**: `Out_scale[dest_index, h]` (read back via `readMemValue .fp16`)
  equals the stored fp16 cell of `max(|K[cur_index,h,·]|)/127` on active heads,
  unchanged otherwise.

Both expected values are computed from the kernel **inputs**, so this summary is
not self-referential. `hOut : Out ≠ OutScale` is the region-distinctness
no-aliasing hypothesis. -/
```
</details>

**Statement:**
```lean
theorem destindex_copy_quantize_kv_python_d64_output_summary
    (K DestLoc Out OutScale : RegionName) (s : BlockState) (hOut : Out ≠ OutScale) :
    (∃ alg,
      (destindex_copy_quantize_kv_real_surface K DestLoc Out OutScale
        512 64 1 512 64 1 8 1 1 8 64 8).toAlgorithm? =
          Except.ok alg) ∧
    (∀ idx : TileIndex [8, 64],
      (exec (destindex_copy_quantize_kv_real_surface K DestLoc Out OutScale
            512 64 1 512 64 1 8 1 1 8 64 8) s).map
          (·.readMemValue .int Out (outOffset s DestLoc 512 64 1 idx))
        = some (if active s 8 8 64 idx then
            quantizeCopyKvSurfaceIntValue s K 512 64 1 8 64 (by norm_num) idx
          else s.readMemValue .int Out (outOffset s DestLoc 512 64 1 idx))) ∧
    (∀ i : Fin 8,
      (exec (destindex_copy_quantize_kv_real_surface K DestLoc Out OutScale
            512 64 1 512 64 1 8 1 1 8 64 8) s).map
          (·.readMemValue .fp16 OutScale (scaleOutOffset1 s DestLoc 8 1 i))
        = some (if scaleActive 8 8 i then
            quantizeCopyKvScaleCell s K 512 64 1 8 64 (by norm_num) i.val
          else s.readMemValue .fp16 OutScale (scaleOutOffset1 s DestLoc 8 1 i)))
```

**Assumptions / layout contracts:**
- `hOut : Out ≠ OutScale`

**Closed-form spec defs (transitive):** `destindex_copy_quantize_kv_real_surface`, `outOffset`, `active`, `quantizeCopyKvSurfaceIntValue`, `scaleOutOffset1`, `scaleActive`, `quantizeCopyKvScaleCell`, `destIndex`, `headIndex`, `dimIndex`, `maskedSrc`, `quantizeCopyKvScaleValue`

<details><summary><code>destindex_copy_quantize_kv_real_surface</code></summary>

```
/-- Real-valued surface of `quantize_copy_kv.py`'s
`_fwd_kernel_destindex_copy_quantize_kv`.

This preserves destination-indexed addressing, `tl.abs`, the per-head
`tl.max(..., axis=1)` scale computation, value writeback, and scale writeback.
The Python kernel casts the scale to fp16 before broadcasting it and casts the
quotient to int8; both casts are preserved as surface dtype annotations and
lower through the DSL's fixed-width cast surfaces. -/
```
```lean
def destindex_copy_quantize_kv_real_surface
    (K : RegionName) (DestLoc : Region .nat) (Out OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_d
      stride_o_bs stride_o_h stride_o_d
      stride_os_bs stride_os_h _stride_os_d
      head_num BLOCK_DMODEL BLOCK_HEAD : Nat) :
    ComputeKernel := triton {
  cur_index = tl.program_id(0)
  offs_h = tl.arange(0, $(BLOCK_HEAD))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  dest_index = tl.load(DestLoc + cur_index)
  src_data = tl.load(K + cur_index * $(stride_k_bs) +
      offs_h[:, None] * $(stride_k_h) + $(stride_k_d) * offs_d[None, :],
    mask=offs_h[:, None] < $(head_num), other=0.0)
  abs_data = tl.abs(src_data)
  data_scale = ((tl.max(abs_data, axis=1) / 127.0).to(tl.float16))[:, None]
  q_src_data = (src_data / data_scale).to(tl.int8)
  o_ptrs = Out + dest_index * $(stride_o_bs) +
    $(stride_o_h) * offs_h[:, None] + $(stride_o_d) * offs_d[None, :]
  os_ptrs = OutScale + dest_index * $(stride_os_bs) + $(stride_os_h) * offs_h[:, None]
  tl.store(o_ptrs, q_src_data, mask=offs_h[:, None] < $(head_num))
  tl.store(os_ptrs, data_scale, mask=offs_h[:, None] < $(head_num))
}
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

<details><summary><code>active</code></summary>

```lean
def active
    (s : BlockState) (head_num BLOCK_HEAD BLOCK_DMODEL : Nat)
    (idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL]) : Prop :=
  headIndex s idx.1 < head_num
```
</details>

<details><summary><code>quantizeCopyKvSurfaceIntValue</code></summary>

```
/-- Genuine quantized value spec (`(src / data_scale).to(int8)`): the int8 cast
of the masked source lane divided by the fp16-cast per-head scale. -/
```
```lean
noncomputable def quantizeCopyKvSurfaceIntValue
    (s : BlockState) (K : RegionName)
    (stride_k_bs stride_k_h stride_k_d head_num BLOCK_DMODEL : Nat)
    (hD : 0 < BLOCK_DMODEL) (idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL]) : Int :=
  WithBot.realToInt8
    (FloatDType.real.cast FloatDType.real
      (Option.map₂ (fun x1 x2 => x1 / x2)
        (maskedSrc s K stride_k_bs stride_k_h stride_k_d head_num idx.1.val idx.2.1.val)
        (FloatDType.fp16.cast FloatDType.real
          (quantizeCopyKvScaleValue s K stride_k_bs stride_k_h stride_k_d head_num
            BLOCK_DMODEL hD idx.1.val))))
```
</details>

<details><summary><code>scaleOutOffset1</code></summary>

```lean
def scaleOutOffset1
    (s : BlockState) (DestLoc : RegionName)
    (stride_os_bs stride_os_h : Nat) (i : Fin BLOCK_HEAD) : Nat :=
  destIndex s DestLoc * stride_os_bs + i.val * stride_os_h
```
</details>

<details><summary><code>scaleActive</code></summary>

```lean
def scaleActive (head_num BLOCK_HEAD : Nat) (i : Fin BLOCK_HEAD) : Prop :=
  i.val < head_num
```
</details>

<details><summary><code>quantizeCopyKvScaleCell</code></summary>

```
/-- Genuine fp16 scale store cell (`data_scale` after the fp16 store rounding):
the value the kernel writes to `Out_scale`, observed through `readMemValue
.fp16`. -/
```
```lean
noncomputable def quantizeCopyKvScaleCell
    (s : BlockState) (K : RegionName)
    (stride_k_bs stride_k_h stride_k_d head_num BLOCK_DMODEL : Nat)
    (hD : 0 < BLOCK_DMODEL) (h : Nat) : TileCarrier .fp16 :=
  FloatDType.fp16.ofReal (FloatDType.fp16.storeValue
    (quantizeCopyKvScaleValue s K stride_k_bs stride_k_h stride_k_d head_num
      BLOCK_DMODEL hD h))
```
</details>

<details><summary><code>destIndex</code></summary>

```lean
def destIndex (s : BlockState) (DestLoc : RegionName) : Nat :=
  s.readMemValue .nat DestLoc (s.pids 0)
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

<details><summary><code>maskedSrc</code></summary>

```
/-- The masked source lane value (`tl.load(..., other=0.0)`): the head-masked
`K[cur_index, h, d]` as a `WithBot ℝ`. -/
```
```lean
noncomputable def maskedSrc
    (s : BlockState) (K : RegionName) (stride_k_bs stride_k_h stride_k_d head_num : Nat)
    (h d : Nat) : WithBot ℝ :=
  if h < head_num then
    some (s.readMem K (s.pids 0 * stride_k_bs + h * stride_k_h + stride_k_d * d))
  else some (0.0 : ℝ)
```
</details>

<details><summary><code>quantizeCopyKvScaleValue</code></summary>

```
/-- The per-head `data_scale` *value* (`(max(|src|, axis=1) / 127).to(fp16)`):
the fp16 cast of the row reduce-max of `|maskedSrc|` divided by `127`. This is
the value computed by the kernel before the fp16 store rounding. -/
```
```lean
noncomputable def quantizeCopyKvScaleValue
    (s : BlockState) (K : RegionName)
    (stride_k_bs stride_k_h stride_k_d head_num BLOCK_DMODEL : Nat)
    (hD : 0 < BLOCK_DMODEL) (h : Nat) : TileCarrier .fp16 :=
  FloatDType.real.cast FloatDType.fp16
    (Option.map (· / 127.0)
      ((Finset.univ.sup'
          (⟨⟨0, hD⟩, Finset.mem_univ _⟩ : (Finset.univ : Finset (Fin BLOCK_DMODEL)).Nonempty)
          (fun x : Fin BLOCK_DMODEL =>
            if maskedSrc s K stride_k_bs stride_k_h stride_k_d head_num h x.val
                < (some 0 : WithBot ℝ) then
              NumericDType.real.sub (some 0)
                (maskedSrc s K stride_k_bs stride_k_h stride_k_d head_num h x.val)
            else maskedSrc s K stride_k_bs stride_k_h stride_k_d head_num h x.val) :
        WithBot ℝ)))
```
</details>

## Also present (pinned special-case summaries)
- `destindex_copy_quantize_kv_value_store_slice_compute_correct`
- `destindex_copy_quantize_kv_scale_store_slice_compute_correct`
- `destindex_copy_quantize_kv_real_surface_value_output_compute_correct`
- `destindex_copy_quantize_kv_real_surface_scale_output_compute_correct`
- `destindex_copy_quantize_kv_python_d64_value_store_compute_correct`
- `destindex_copy_quantize_kv_python_d256_value_store_compute_correct`
- `destindex_copy_quantize_kv_python_scale_store_compute_correct`
- `destindex_copy_quantize_kv_python_d64_all_outputs_compute_correct`
- `destindex_copy_quantize_kv_python_d256_all_outputs_compute_correct`
