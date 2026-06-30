# Spec sheet — `bench/tritonbench_g/quantize_kv_copy/QuantizeKvCopy.lean`

**Python source:** `bench/tritonbench_g/quantize_kv_copy/quantize_kv_copy.py`

## Public theorem: `destindex_copy_quantize_kv_group_output_summary_general`

<details><summary>docstring</summary>

```
/-- **Dimension-general grouped summary (genuine, gap-free).** The symbolic
generalization of the former pinned Python-shape grouped summary: all
strides, `group_size`, `BLOCK_GROUP_NUM`, and `BLOCK_GROUP_DIM` are arbitrary
`Nat` parameters. Bundles the three already-general building blocks
(`..._toAlgorithm_supported`, `..._value_output_compute_correct`,
`..._scale_output_compute_correct`) under honest hypotheses: `hD : 0 <
BLOCK_GROUP_DIM` (nonempty reduce axis), `hOut : Out ≠ OutScale` (no aliasing),
and the value/scale destination-offset injectivity (no-collision) hypotheses the
genuine readbacks need. Both expected values are computed from the kernel
**inputs**, so this summary is not self-referential. -/
```
</details>

**Statement:**
```lean
theorem destindex_copy_quantize_kv_group_output_summary_general
    (K DestLoc Out OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_g stride_k_d
      stride_o_bs stride_o_h stride_o_g stride_o_d
      stride_os_bs stride_os_h stride_os_g
      group_size BLOCK_GROUP_NUM BLOCK_GROUP_DIM : Nat)
    (s : BlockState) (hD : 0 < BLOCK_GROUP_DIM) (hOut : Out ≠ OutScale)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_GROUP_NUM, BLOCK_GROUP_DIM] =>
        outOffset s DestLoc stride_o_bs stride_o_h stride_o_g stride_o_d idx))
    (hScaleInj : Function.Injective
      (fun i : Fin BLOCK_GROUP_NUM =>
        scaleOutOffset s DestLoc stride_os_bs stride_os_h i)) :
    (∃ alg,
      (destindex_copy_quantize_kv_group_real_surface K DestLoc Out OutScale
        stride_k_bs stride_k_h stride_k_g stride_k_d stride_o_bs stride_o_h
        stride_o_g stride_o_d stride_os_bs stride_os_h stride_os_g group_size
        BLOCK_GROUP_NUM BLOCK_GROUP_DIM).toAlgorithm? =
          Except.ok alg) ∧
    (∀ idx : TileIndex [BLOCK_GROUP_NUM, BLOCK_GROUP_DIM],
      let outAddr
```

**Assumptions / layout contracts:**
- `hD : 0 < BLOCK_GROUP_DIM`
- `hOut : Out ≠ OutScale`
- `hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_GROUP_NUM, BLOCK_GROUP_DIM] =>
        outOffset s DestLoc stride_o_bs stride_o_h stride_o_g stride_o_d idx)`
- `hScaleInj : Function.Injective
      (fun i : Fin BLOCK_GROUP_NUM =>
        scaleOutOffset s DestLoc stride_os_bs stride_os_h i)`

**Closed-form spec defs (transitive):** `outOffset`, `scaleOutOffset`, `destindex_copy_quantize_kv_group_real_surface`, `destIndex`, `groupIndex`, `dimIndex`

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

<details><summary><code>scaleOutOffset</code></summary>

```lean
def scaleOutOffset
    (s : BlockState) (DestLoc : RegionName)
    (stride_os_bs stride_os_h : Nat) (i : Fin BLOCK_GROUP_NUM) : Nat :=
  destIndex s DestLoc * stride_os_bs + s.pids 1 * stride_os_h + i.val
```
</details>

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

## Also present (pinned special-case summaries)
- `destindex_copy_quantize_kv_group_value_store_slice_compute_correct`
- `destindex_copy_quantize_kv_group_scale_store_slice_compute_correct`
- `destindex_copy_quantize_kv_group_real_surface_value_output_compute_correct`
- `destindex_copy_quantize_kv_group_real_surface_scale_output_compute_correct`
