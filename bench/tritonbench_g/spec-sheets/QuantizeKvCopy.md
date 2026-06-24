# Spec sheet — `bench/tritonbench_g/quantize_kv_copy/QuantizeKvCopy.lean`

**Python source:** `bench/tritonbench_g/quantize_kv_copy/quantize_kv_copy.py`

## Public theorem: `destindex_copy_quantize_kv_group_python_output_summary`

<details><summary>docstring</summary>

```
/-- **Public Python grouped summary (genuine, gap-free).** The checked Python
shape (`head_num = 4`, `head_dim = 16`, `quant_group_dim = 8`, hence
`group_size = 2`, `BLOCK_GROUP_NUM = 2`, `BLOCK_GROUP_DIM = 8`; `K/Out` grouped
strides `(64, 16, 8, 1)`, `Out_scale` strides `(8, 2, 1)`) covers the full
faithful surface syntax (`tl.abs`, `tl.max(..., axis=1)`, the output-dtype scale
cast, the int8 quotient cast, the destination-indexed grouped masked stores) and
the two externally visible outputs:

* **value**: `Out[dest_index, cur_head, g, d]` (read back via `readMemValue
  .int`) equals `(K[cur_index, cur_head, g, d] /
  (max(|K[cur_index,cur_head,g,·]|)/127)).to(int8)` on active groups
  (`g < group_size`), and is unchanged otherwise;
* **scale**: `Out_scale[dest_index, cur_head, g]` (read back via `readMem`)
  equals the stored real cell of `max(|K[cur_index,cur_head,g,·]|)/127` on active
  groups, unchanged otherwise.

Both expected values are computed from the kernel **inputs**, so this summary is
not self-referential. `hOut : Out ≠ OutScale` is the region-distinctness
no-aliasing hypothesis. -/
```
</details>

**Statement:**
```lean
theorem destindex_copy_quantize_kv_group_python_output_summary
    (K DestLoc Out OutScale : RegionName) (s : BlockState) (hOut : Out ≠ OutScale) :
    (∃ alg,
      (destindex_copy_quantize_kv_group_real_surface K DestLoc Out OutScale
        64 16 8 1 64 16 8 1 8 2 1 2 2 8).toAlgorithm? =
          Except.ok alg) ∧
    (∀ idx : TileIndex [2, 8],
      (exec (destindex_copy_quantize_kv_group_real_surface K DestLoc Out OutScale
            64 16 8 1 64 16 8 1 8 2 1 2 2 8) s).map
          (·.readMemValue .int Out (outOffset s DestLoc 64 16 8 1 idx))
        = some (if active s 2 2 8 idx then
            quantizeKvCopyGroupSurfaceIntValue s K 64 16 8 2 8 (by norm_num) idx
          else s.readMemValue .int Out (outOffset s DestLoc 64 16 8 1 idx))) ∧
    (∀ i : Fin 2,
      (exec (destindex_copy_quantize_kv_group_real_surface K DestLoc Out OutScale
            64 16 8 1 64 16 8 1 8 2 1 2 2 8) s).map
          (·.readMem OutScale (scaleOutOffset s DestLoc 8 2 i))
        = some (if scaleActive 2 2 i then
            quantizeKvCopyGroupScaleCell s K 64 16 8 2 8 (by norm_num) i.val
          else s.readMem OutScale (scaleOutOffset s DestLoc 8 2 i)))
```

**Assumptions / layout contracts:**
- `hOut : Out ≠ OutScale`

**Closed-form spec defs (transitive):** `destindex_copy_quantize_kv_group_real_surface`, `outOffset`, `active`, `quantizeKvCopyGroupSurfaceIntValue`, `scaleOutOffset`, `scaleActive`, `quantizeKvCopyGroupScaleCell`, `destIndex`, `groupIndex`, `dimIndex`, `maskedSrc`, `quantizeKvCopyGroupScaleValue`

<details><summary><code>destindex_copy_quantize_kv_group_real_surface</code></summary>

```
/-- Real-valued surface of `quantize_kv_copy.py`'s grouped
`_fwd_kernel_destindex_copy_quantize_kv`.

This preserves destination-indexed grouped addressing, `tl.abs`, per-group
scale computation, value writeback, and scale writeback. The Python kernel casts
the scale to `OutScale.dtype.element_ty`; that cast is represented explicitly.
The final quotient cast to int8 is preserved as a surface dtype annotation and
lowers through the DSL's fixed-width cast surface. -/
```
```lean
def destindex_copy_quantize_kv_group_real_surface
    (K : RegionName) (DestLoc : Region .nat) (Out OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_g _stride_k_d
      stride_o_bs stride_o_h stride_o_g _stride_o_d
      stride_os_bs stride_os_h _stride_os_g
      group_size BLOCK_GROUP_NUM BLOCK_GROUP_DIM : Nat) :
    ComputeKernel := triton {
  cur_index = tl.program_id(0)
  cur_head = tl.program_id(1)
  offs_g = tl.arange(0, $(BLOCK_GROUP_NUM))
  offs_d = tl.arange(0, $(BLOCK_GROUP_DIM))
  dest_index = tl.load(DestLoc + cur_index)
  src_data = tl.load(K + cur_index * $(stride_k_bs) + cur_head * $(stride_k_h) +
      offs_g[:, None] * $(stride_k_g) + offs_d[None, :],
    mask=offs_g[:, None] < $(group_size), other=0.0)
  abs_data = tl.abs(src_data)
  data_scale = (tl.max(abs_data, axis=1) / 127.0).to(OutScale.dtype.element_ty)
  q_src_data = (src_data / data_scale[:, None]).to(tl.int8)
  o_ptrs = Out + dest_index * $(stride_o_bs) + cur_head * $(stride_o_h) +
    offs_g[:, None] * $(stride_o_g) + offs_d[None, :]
  os_ptrs = OutScale + dest_index * $(stride_os_bs) + cur_head * $(stride_os_h) +
    offs_g
  tl.store(o_ptrs, q_src_data, mask=offs_g[:, None] < $(group_size))
  tl.store(os_ptrs, data_scale, mask=offs_g < $(group_size))
}
```
</details>

<details><summary><code>outOffset</code></summary>

```lean
def outOffset
    (s : BlockState) (DestLoc : RegionName)
    (stride_o_bs stride_o_h stride_o_g _stride_o_d : Nat)
    (idx : TileIndex [BLOCK_GROUP_NUM, BLOCK_GROUP_DIM]) : Nat :=
  destIndex s DestLoc * stride_o_bs + s.pids 1 * stride_o_h +
    groupIndex s idx.1 * stride_o_g + dimIndex s idx.2.1
```
</details>

<details><summary><code>active</code></summary>

```lean
def active
    (s : BlockState) (group_size BLOCK_GROUP_NUM BLOCK_GROUP_DIM : Nat)
    (idx : TileIndex [BLOCK_GROUP_NUM, BLOCK_GROUP_DIM]) : Prop :=
  groupIndex s idx.1 < group_size
```
</details>

<details><summary><code>quantizeKvCopyGroupSurfaceIntValue</code></summary>

```
/-- Genuine quantized value spec (`(src / data_scale).to(int8)`): the int8 cast
of the masked source lane divided by the per-group scale. -/
```
```lean
noncomputable def quantizeKvCopyGroupSurfaceIntValue
    (s : BlockState) (K : RegionName)
    (stride_k_bs stride_k_h stride_k_g group_size BLOCK_GROUP_DIM : Nat)
    (hD : 0 < BLOCK_GROUP_DIM) (idx : TileIndex [BLOCK_GROUP_NUM, BLOCK_GROUP_DIM]) : Int :=
  WithBot.realToInt8
    (FloatDType.real.cast FloatDType.real
      (Option.map₂ (fun x1 x2 => x1 / x2)
        (maskedSrc s K stride_k_bs stride_k_h stride_k_g group_size idx.1.val idx.2.1.val)
        (quantizeKvCopyGroupScaleValue s K stride_k_bs stride_k_h stride_k_g
          group_size BLOCK_GROUP_DIM hD idx.1.val)))
```
</details>

<details><summary><code>scaleOutOffset</code></summary>

```lean
def scaleOutOffset
    (s : BlockState) (DestLoc : RegionName)
    (stride_os_bs stride_os_h : Nat) (i : Fin BLOCK_GROUP_NUM) : Nat :=
  destIndex s DestLoc * stride_os_bs + s.pids 1 * stride_os_h + i.val
```
</details>

<details><summary><code>scaleActive</code></summary>

```lean
def scaleActive (group_size BLOCK_GROUP_NUM : Nat) (i : Fin BLOCK_GROUP_NUM) :
    Prop := i.val < group_size
```
</details>

<details><summary><code>quantizeKvCopyGroupScaleCell</code></summary>

```
/-- Genuine scale store cell (`data_scale`): the real value the kernel writes to
`Out_scale`, observed through `readMem`. -/
```
```lean
noncomputable def quantizeKvCopyGroupScaleCell
    (s : BlockState) (K : RegionName)
    (stride_k_bs stride_k_h stride_k_g group_size BLOCK_GROUP_DIM : Nat)
    (hD : 0 < BLOCK_GROUP_DIM) (g : Nat) : ℝ :=
  WithBot.unbotD 0
    (quantizeKvCopyGroupScaleValue s K stride_k_bs stride_k_h stride_k_g
      group_size BLOCK_GROUP_DIM hD g)
```
</details>

<details><summary><code>destIndex</code></summary>

```lean
def destIndex (s : BlockState) (DestLoc : RegionName) : Nat :=
  s.readMemValue .nat DestLoc (s.pids 0)
```
</details>

<details><summary><code>groupIndex</code></summary>

```lean
def groupIndex (_s : BlockState) (i : Fin BLOCK_GROUP_NUM) : Nat :=
  i.val
```
</details>

<details><summary><code>dimIndex</code></summary>

```lean
def dimIndex (_s : BlockState) (j : Fin BLOCK_GROUP_DIM) : Nat :=
  j.val
```
</details>

<details><summary><code>maskedSrc</code></summary>

```
/-- The masked source lane value (`tl.load(..., other=0.0)`): the group-masked
`K[cur_index, cur_head, g, d]` as a `WithBot ℝ`. -/
```
```lean
noncomputable def maskedSrc
    (s : BlockState) (K : RegionName)
    (stride_k_bs stride_k_h stride_k_g group_size : Nat)
    (g d : Nat) : WithBot ℝ :=
  if g < group_size then
    some (s.readMem K
      (s.pids 0 * stride_k_bs + s.pids 1 * stride_k_h + g * stride_k_g + d))
  else some (0.0 : ℝ)
```
</details>

<details><summary><code>quantizeKvCopyGroupScaleValue</code></summary>

```
/-- The per-group `data_scale` *value* (`max(|src|, axis=1) / 127`): the row
reduce-max of `|maskedSrc|` divided by `127`, as a `WithBot ℝ`. This is the value
the kernel both divides by (for the int8 value) and stores to `Out_scale`. -/
```
```lean
noncomputable def quantizeKvCopyGroupScaleValue
    (s : BlockState) (K : RegionName)
    (stride_k_bs stride_k_h stride_k_g group_size BLOCK_GROUP_DIM : Nat)
    (hD : 0 < BLOCK_GROUP_DIM) (g : Nat) : WithBot ℝ :=
  Option.map (· / 127.0)
    ((Finset.univ.sup'
        (⟨⟨0, hD⟩, Finset.mem_univ _⟩ : (Finset.univ : Finset (Fin BLOCK_GROUP_DIM)).Nonempty)
        (fun x : Fin BLOCK_GROUP_DIM =>
          if maskedSrc s K stride_k_bs stride_k_h stride_k_g group_size g x.val
              < (some 0 : WithBot ℝ) then
            NumericDType.real.sub (some 0)
              (maskedSrc s K stride_k_bs stride_k_h stride_k_g group_size g x.val)
          else maskedSrc s K stride_k_bs stride_k_h stride_k_g group_size g x.val) :
      WithBot ℝ))
```
</details>

## Also present (pinned special-case summaries)
- `destindex_copy_quantize_kv_group_value_store_slice_compute_correct`
- `destindex_copy_quantize_kv_group_scale_store_slice_compute_correct`
- `destindex_copy_quantize_kv_group_real_surface_value_output_compute_correct`
- `destindex_copy_quantize_kv_group_real_surface_scale_output_compute_correct`
