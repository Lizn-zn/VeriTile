# Spec sheet — `bench/tritonbench_g/quantize_kv_transform/QuantizeKvTransform.lean`

**Python source:** `bench/tritonbench_g/quantize_kv_transform/quantize_kv_transform.py`

## Public theorem: `destindex_copy_quantize_kv_transform_output_summary_general`

<details><summary>docstring</summary>

```
/-- **Dimension-general output summary (`ComputeCorrect.Realizes_without_Rounding` form).** For
arbitrary strides / `head_num` / `head_dim` / `BLOCK_DMODEL` / `BLOCK_HEAD` (and
any program ids in `s`), the destindex quantize-KV-transform surface lowers
(`∃ alg, … = Except.ok alg`), and its two stored outputs each `Realizes_without_Rounding` a
genuine input-memory closed form:

* the int8 value store to `Out` realizes the genuine per-cell int value
  `quantizeKvTransformSurfaceIntValue` on active lanes (`active`), and preserves
  the prior `Out` contents on inactive lanes — read back via `readMemValue .int`;
* the real per-row scale store to `OutScale` realizes the genuine
  `quantizeKvTransformScaleCell` on active rows (`scaleActive`), and preserves the
  prior `OutScale` contents on inactive rows — read back via `readMem`.

Both write maps are total (`fun _ => some …`); the active/inactive split is
carried inside `expected`. Honest side conditions: offset injectivity for the
value tile (`hValInj`) and the per-head scale (`hScaleInj`), and region
distinctness `hOut`. The general version assumes full tile injectivity, which
holds whenever `BLOCK_DMODEL = head_dim` and `BLOCK_HEAD = head_num`. -/
```
</details>

**Statement:**
```lean
specification destindex_copy_quantize_kv_transform_output_summary_general
    (K DestLoc Out OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
      stride_os_bs stride_os_h stride_os_d head_num head_dim BLOCK_DMODEL BLOCK_HEAD : Nat)
    (s : BlockState) (hD : 0 < BLOCK_DMODEL) (hOut : Out ≠ OutScale)
    (hValInj : Function.Injective
      (fun idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL] =>
        outOffset s DestLoc stride_o_bs stride_o_h stride_o_d idx))
    (hScaleInj : Function.Injective
      (fun i : Fin BLOCK_HEAD => scaleOutOffset s DestLoc stride_os_bs stride_os_h i)) :
    (∃ alg, (destindex_copy_quantize_kv_transform_real_surface K DestLoc Out OutScale
        stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
        stride_os_bs stride_os_h stride_os_d head_num head_dim BLOCK_DMODEL BLOCK_HEAD).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := destindex_copy_quantize_kv_transform_real_surface K DestLoc Out OutScale
        stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
        stride_os_bs stride_os_h stride_os_d head_num head_dim BLOCK_DMODEL BLOCK_HEAD)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL] =>
        some (Out, outOffset s DestLoc stride_o_bs stride_o_h stride_o_d idx))
      (expected := fun idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL] =>
        (if active s head_num head_dim BLOCK_HEAD BLOCK_DMODEL idx then
            quantizeKvTransformSurfaceIntValue s K stride_k_bs stride_k_h stride_k_d head_num head_dim BLOCK_DMODEL hD idx
          else s.readMemValue .int Out (outOffset s DestLoc stride_o_bs stride_o_h stride_o_d idx))) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := destindex_copy_quantize_kv_transform_real_surface K DestLoc Out OutScale
        stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
        stride_os_bs stride_os_h stride_os_d head_num head_dim BLOCK_DMODEL BLOCK_HEAD)
      (initialState := s)
      (write := fun i : Fin BLOCK_HEAD =>
        some (OutScale, scaleOutOffset s DestLoc stride_os_bs stride_os_h i))
      (expected := fun i : Fin BLOCK_HEAD =>
        (if scaleActive head_num BLOCK_HEAD i then
            quantizeKvTransformScaleCell s K stride_k_bs stride_k_h stride_k_d head_num head_dim BLOCK_DMODEL hD i.val
          else s.readMem OutScale (scaleOutOffset s DestLoc stride_os_bs stride_os_h i)))
```

**Assumptions / layout contracts:**
- `hD : 0 < BLOCK_DMODEL`
- `hOut : Out ≠ OutScale`
- `hValInj : Function.Injective
      (fun idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL] =>
        outOffset s DestLoc stride_o_bs stride_o_h stride_o_d idx)`
- `hScaleInj : Function.Injective
      (fun i : Fin BLOCK_HEAD => scaleOutOffset s DestLoc stride_os_bs stride_os_h i)`

**Closed-form spec defs (transitive):** `outOffset`, `scaleOutOffset`, `destindex_copy_quantize_kv_transform_real_surface`, `active`, `quantizeKvTransformSurfaceIntValue`, `scaleActive`, `quantizeKvTransformScaleCell`, `destIndex`, `headIndex`, `dimIndex`, `maskedSrc`, `quantizeKvTransformScaleValue`

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

<details><summary><code>scaleOutOffset</code></summary>

```lean
def scaleOutOffset
    (s : BlockState) (DestLoc : RegionName)
    (stride_os_bs stride_os_h : Nat) (i : Fin BLOCK_HEAD) : Nat :=
  destIndex s DestLoc * stride_os_bs + stride_os_h * i.val
```
</details>

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
- `destindex_copy_quantize_kv_transform_real_surface_value_output_compute_correct`
- `destindex_copy_quantize_kv_transform_real_surface_scale_output_compute_correct`
