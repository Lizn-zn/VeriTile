# Spec sheet — `bench/tritonbench_g/f8_conversion_utils/F8ConversionUtils.lean`

**Python source:** `bench/tritonbench_g/f8_conversion_utils/f8_conversion_utils.py`

## Public theorem: `f16_to_f8_io_correctness`

<details><summary>docstring</summary>

```
/-- **The fp8 headline**: for every rounding model `R`, `kernel_f16_to_f8`
implements the masked identity copy on its IO signature at the `.f8e5`
output grid — every active output lane reads back as an `.f8e5`-typed cell
holding `R.round .f8e5 (xs j)`: the input value quantized **once** onto the
fp8 (e5m2) grid. The corpus's first fp8 boundary quantization. -/
```
</details>

**Statement:**
```lean
specification f16_to_f8_io_correctness (R : RoundingModel)
    (Y X : RegionName) (N BLOCK_SIZE : Nat) :
    f16ToF8IO Y X N BLOCK_SIZE ⊨[R, .f8e5] fun xs i => xs i
```

**Closed-form spec defs (transitive):** `f16ToF8IO`, `kernel_f16_to_f8`

<details><summary><code>f16ToF8IO</code></summary>

```
/-- `kernel_f16_to_f8`'s masked IO signature: 1-D window
`[pid·B, (pid+1)·B)` on both buffers, active lanes `pid·B + j < N` (the
kernel's shared load/store mask). -/
```
```lean
def f16ToF8IO (Y X : RegionName) (N BLOCK_SIZE : Nat) : MaskedKernelIO₁ where
  kernel := kernel_f16_to_f8 Y X N BLOCK_SIZE
  inp := X
  out := Y
  B := BLOCK_SIZE
  read := fun pid => pid * BLOCK_SIZE
  write := fun pid => pid * BLOCK_SIZE
  mask := fun pid j => pid * BLOCK_SIZE + j.val < N
```
</details>

<details><summary><code>kernel_f16_to_f8</code></summary>

```
/-- Faithful transcription of `f8_conversion_utils.py`'s `kernel_f16_to_f8`
(fp16 → fp8 masked copy). The store's implicit fp8 cast is spelled
`(x).to(tl.float8e5)` — see the preamble blocker note. -/
```
```lean
def kernel_f16_to_f8 (Y X : RegionName) (N BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  offs = pid * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
  mask = offs < $(N)
  x = tl.load(X + offs, mask=mask)
  tl.store(Y + offs, (x).to(tl.float8e5), mask=mask)
}
```
</details>

## Public theorem: `f8_to_f16_io_correctness`

<details><summary>docstring</summary>

```
/-- **The headline**: for every rounding model `R`, `kernel_f8_to_f16`
implements the masked identity copy on its IO signature at the `.fp16`
output grid — every active output lane reads back as an `.fp16`-typed cell
holding `R.round .fp16 (xs j)` (the upstream duplicate store is idempotent:
the second scatter rewrites the same rounded cells). -/
```
</details>

**Statement:**
```lean
specification f8_to_f16_io_correctness (R : RoundingModel)
    (Y X : RegionName) (N BLOCK_SIZE : Nat) :
    f8ToF16IO Y X N BLOCK_SIZE ⊨[R, .fp16] fun xs i => xs i
```

**Closed-form spec defs (transitive):** `f8ToF16IO`, `kernel_f8_to_f16`

<details><summary><code>f8ToF16IO</code></summary>

```
/-- `kernel_f8_to_f16`'s masked IO signature (same window/mask shape as the
reverse direction). -/
```
```lean
def f8ToF16IO (Y X : RegionName) (N BLOCK_SIZE : Nat) : MaskedKernelIO₁ where
  kernel := kernel_f8_to_f16 Y X N BLOCK_SIZE
  inp := X
  out := Y
  B := BLOCK_SIZE
  read := fun pid => pid * BLOCK_SIZE
  write := fun pid => pid * BLOCK_SIZE
  mask := fun pid j => pid * BLOCK_SIZE + j.val < N
```
</details>

<details><summary><code>kernel_f8_to_f16</code></summary>

```
/-- Faithful transcription of `f8_conversion_utils.py`'s `kernel_f8_to_f16`
(fp8 → fp16 masked copy, double store). The store's implicit fp16 cast is
spelled `(x).to(tl.float16)` — see the preamble blocker note. -/
```
```lean
def kernel_f8_to_f16 (Y X : RegionName) (N BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  offs = pid * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
  mask = offs < $(N)
  x = tl.load(X + offs, mask=mask)
  tl.store(Y + offs, (x).to(tl.float16), mask=mask)
  tl.store(Y + offs, (x).to(tl.float16), mask=mask)
}
```
</details>
