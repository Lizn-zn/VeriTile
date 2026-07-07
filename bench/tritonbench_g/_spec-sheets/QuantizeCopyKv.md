# Spec sheet — `bench/tritonbench_g/quantize_copy_kv/QuantizeCopyKv.lean`

**Python source:** `bench/tritonbench_g/quantize_copy_kv/quantize_copy_kv.py`

## Public theorem: `destindex_copy_quantize_kv_output_summary_general`

<details><summary>docstring</summary>

```
/-- **Dimension-general output summary.** For arbitrary strides / `head_num` /
`BLOCK_DMODEL` / `BLOCK_HEAD` (and any program ids in `s`), the destindex
quantize-KV-copy surface lowers, writes the genuine per-cell int value
`quantizeCopyKvSurfaceIntValue` to `Out` (masked by `active`), and the genuine
fp16 per-row scale `quantizeCopyKvScaleCell` to `OutScale` (masked by
`scaleActive`) — under honest offset-injectivity side conditions.

The three headline conjuncts are: (1) the surface lowers through algorithm
erasure; (2) the int8 value output stated as `ComputeCorrect.Realizes` over a
total per-lane write map to `Out` (the `expected` carries the inactive-lane
`readMemValue .int Out` guarantee, so the map is total and the mask lives in
`expected`); (3) the fp16 per-row scale output.

**Honest carrier note for conjunct (3).** Unlike conjunct (2), the fp16 scale
output is NOT phrased as `ComputeCorrect.Realizes`: it stays in the raw
`(exec …).map (·.readMemValue .fp16 OutScale …) = some (if …)` form. This is a
framework *carrier* limitation, not a proof gap. `ComputeCorrect.Realizes`
requires an `OutputReadable` instance for the readback type, and the only
instances in `VeriTile.Triton.Correctness` are for `MemCell`, `ℝ`, `Nat`,
and `Int`. The scale reads back at `TileCarrier .fp16` (the *decoded* fp16
value), which has no `OutputReadable` carrier, so it cannot be wrapped in
`Realizes`. The conjunct is nonetheless genuine and non-self-referential: it
reads INPUT memory and `quantizeCopyKvScaleCell` is computed from the kernel
inputs (see `destindex_copy_quantize_kv_real_surface_scale_output_compute_correct`).
This is the honest-blocker outcome of `MAIN_THEOREM_CONVENTIONS.md` §6.

Concrete literal-dimension instantiations of this general summary are not kept as
separate declarations. -/
```
</details>

**Statement:**
```lean
theorem destindex_copy_quantize_kv_output_summary_general
    (K DestLoc Out OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
      stride_os_bs stride_os_h stride_os_d head_num BLOCK_DMODEL BLOCK_HEAD : Nat)
    (s : BlockState) (hD : 0 < BLOCK_DMODEL) (hOut : Out ≠ OutScale)
    (hValInj : Function.Injective
      (fun idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL] =>
        outOffset s DestLoc stride_o_bs stride_o_h stride_o_d idx))
    (hScaleInj : Function.Injective
      (fun i : Fin BLOCK_HEAD => scaleOutOffset1 s DestLoc stride_os_bs stride_os_h i)) :
    (∃ alg, (destindex_copy_quantize_kv_real_surface K DestLoc Out OutScale
        stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
        stride_os_bs stride_os_h stride_os_d head_num BLOCK_DMODEL BLOCK_HEAD).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := destindex_copy_quantize_kv_real_surface K DestLoc Out OutScale
        stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
        stride_os_bs stride_os_h stride_os_d head_num BLOCK_DMODEL BLOCK_HEAD)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL] =>
        some (Out, outOffset s DestLoc stride_o_bs stride_o_h stride_o_d idx))
      (expected := fun idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL] =>
        (if active s head_num BLOCK_HEAD BLOCK_DMODEL idx then
            quantizeCopyKvSurfaceIntValue s K stride_k_bs stride_k_h stride_k_d head_num BLOCK_DMODEL hD idx
          else s.readMemValue .int Out (outOffset s DestLoc stride_o_bs stride_o_h stride_o_d idx) : Int))) ∧
    -- Conjunct (3): raw `.map … = some (if …)` form, NOT `Realizes` — see the
    -- honest carrier note in the docstring. `TileCarrier .fp16` (the decoded fp16
    -- value read back here) has no `OutputReadable` instance in the framework, so
    -- this genuine, non-self-referential output cannot be wrapped in `Realizes`.
    (∀ i : Fin BLOCK_HEAD,
      (exec (destindex_copy_quantize_kv_real_surface K DestLoc Out OutScale
            stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
            stride_os_bs stride_os_h stride_os_d head_num BLOCK_DMODEL BLOCK_HEAD) s).map
          (·.readMemValue .fp16 OutScale (scaleOutOffset1 s DestLoc stride_os_bs stride_os_h i))
        = some (if scaleActive head_num BLOCK_HEAD i then
            quantizeCopyKvScaleCell s K stride_k_bs stride_k_h stride_k_d head_num BLOCK_DMODEL hD i.val
          else s.readMemValue .fp16 OutScale (scaleOutOffset1 s DestLoc stride_os_bs stride_os_h i)))
```

**Assumptions / layout contracts:**
- `hD : 0 < BLOCK_DMODEL`
- `hOut : Out ≠ OutScale`
- `hValInj : Function.Injective
      (fun idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL] =>
        outOffset s DestLoc stride_o_bs stride_o_h stride_o_d idx)`
- `hScaleInj : Function.Injective
      (fun i : Fin BLOCK_HEAD => scaleOutOffset1 s DestLoc stride_os_bs stride_os_h i)`

**Closed-form spec defs (transitive):** `outOffset`, `scaleOutOffset1`, `destindex_copy_quantize_kv_real_surface`, `active`, `quantizeCopyKvSurfaceIntValue`, `scaleActive`, `quantizeCopyKvScaleCell`, `destIndex`, `headIndex`, `dimIndex`, `maskedSrc`, `quantizeCopyKvScaleValue`

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

<details><summary><code>scaleOutOffset1</code></summary>

```lean
def scaleOutOffset1
    (s : BlockState) (DestLoc : RegionName)
    (stride_os_bs stride_os_h : Nat) (i : Fin BLOCK_HEAD) : Nat :=
  destIndex s DestLoc * stride_os_bs + i.val * stride_os_h
```
</details>

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
- `destindex_copy_quantize_kv_real_surface_value_output_compute_correct`
- `destindex_copy_quantize_kv_real_surface_scale_output_compute_correct`
