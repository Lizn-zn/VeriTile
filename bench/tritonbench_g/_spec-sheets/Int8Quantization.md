# Spec sheet — `bench/tritonbench_g/int8_quantization/Int8Quantization.lean`

**Python source:** `bench/tritonbench_g/int8_quantization/int8_quantization.py`

## Public theorem: `per_block_int8_output_summary_general`

<details><summary>docstring</summary>

```
/-- **Dimension-general output summary for `per_block_int8` (`int8_quantization.py`).**

For arbitrary token count `L`, channel count `C`, block size `BLK`, scale stride
`scale_stride` and pre-scale `preScale` (with the row-major output offset
injective on the `[BLK, C]` block), this bundles:

* both the full faithful Q surface (`q_kernel_per_block_int8_surface`, including
  the `C**-0.5 * log2(e)` pre-scale, `tl.abs`/`tl.max` per-block scale, signed
  half-up rounding and the `to(tl.int8)` cast) and the full faithful K surface
  (`k_kernel_per_block_int8_surface`) lowering to the algorithm layer, plus the
  scale-compute store surface lowering;
* the **value** store realizing `perBlockInt8ScaledSpec` (`= preScale·X / Scale`)
  on every active lane (`off_blk·BLK + i < L`), unchanged otherwise;
* the **scale** store realizing the per-block scalar `scaleStoreSpec`.

All expected values are computed from the kernel **inputs** (no `exec`/`readMem`
self-reference). This general closed form holds over arbitrary dimensions
(mirrors the dimension-parameterized reference
`attention_forward_triton_closed_form_correct`). -/
```
</details>

**Statement:**
```lean
theorem per_block_int8_output_summary_general
    (X XInt8 Scale ScalePre : RegionName)
    (L C BLK scale_stride : Nat) (preScale : ℝ) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLK, C] => xOffset s L C BLK idx)) :
    ((∃ alg, (q_kernel_per_block_int8_surface X XInt8 Scale
        L C BLK scale_stride).toAlgorithm? = Except.ok alg) ∧
     (∃ alg, (k_kernel_per_block_int8_surface X XInt8 Scale
        L C BLK scale_stride).toAlgorithm? = Except.ok alg) ∧
     (∃ alg, (per_block_int8_scale_compute_store_slice X Scale
        L C BLK scale_stride preScale).toAlgorithm? = Except.ok alg)) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel
```

**Assumptions / layout contracts:**
- `hOutInj : Function.Injective
      (fun idx : TileIndex [BLK, C] => xOffset s L C BLK idx)`

**Closed-form spec defs (transitive):** `xOffset`, `q_kernel_per_block_int8_surface`, `k_kernel_per_block_int8_surface`, `per_block_int8_scale_compute_store_slice`, `baseOffset`, `rowIndex`, `colIndex`

<details><summary><code>xOffset</code></summary>

```lean
def xOffset (s : BlockState) (L C BLK : Nat) (idx : TileIndex [BLK, C]) : Nat :=
  baseOffset s L C + rowIndex s BLK idx.1 * C + colIndex s idx.2.1
```
</details>

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

<details><summary><code>baseOffset</code></summary>

```lean
def baseOffset (s : BlockState) (L C : Nat) : Nat :=
  s.pids 1 * L * C
```
</details>

<details><summary><code>rowIndex</code></summary>

```lean
def rowIndex (s : BlockState) (BLK : Nat) (i : Fin BLK) : Nat :=
  s.pids 0 * BLK + i.val
```
</details>

<details><summary><code>colIndex</code></summary>

```lean
def colIndex (_s : BlockState) (j : Fin C) : Nat :=
  j.val
```
</details>

## Also present (pinned special-case summaries)
- `per_block_int8_scaled_store_slice_compute_correct`
- `per_block_int8_scale_store_slice_compute_correct`
- `per_block_int8_closed_form_correct`
