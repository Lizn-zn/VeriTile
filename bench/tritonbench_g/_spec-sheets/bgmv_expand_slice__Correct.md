# Spec sheet — `bench/tritonbench_g/bgmv_expand_slice/Correct.lean`

**Python source:** `bench/tritonbench_g/bgmv_expand_slice/bgmv_expand_slice.py`

## Public theorem: `bgmv_expand_slice_correctness`

**Statement:**
```lean
specification bgmv_expand_slice_correctness (p : Parameters)
    (input : TypedSource.InputStorage) (weight : TypedSource.WeightStorage) (out_dtype : FloatDType)
    (cast even add : Bool) (hBN : 0 < p.BN) (hstride : 0 < p.outputStride)
    (hK : p.K ≤ p.BK) (hsplits : 0 < p.splits) (hw : p.weights ≠ p.output) :
    Spec.Real (originalIO p input weight out_dtype cast even add ⊨ expected p hK add)
```

**Assumptions / layout contracts:**
- `hBN : 0 < p.BN`
- `hstride : 0 < p.outputStride`
- `hK : p.K ≤ p.BK`
- `hsplits : 0 < p.splits`
- `hw : p.weights ≠ p.output`

**Closed-form spec defs (transitive):** `Parameters`, `originalIO`, `expected`, `original`, `specialize`

<details><summary><code>Parameters</code></summary>

```lean
abbrev Parameters := RealSetupSource.Parameters
```
</details>

<details><summary><code>originalIO</code></summary>

```
/-- Input has BK lanes, active when EVEN_K or k<K, even for an empty output
window. Weights cover every split row and K rank entries; initial output is
pinned at the write window only when ADD_INPUTS. Allocation roles are vector,
weights, output and signed metadata: the read/write output is allocated once.
`length = (N + SPLIT_N - 1) / SPLIT_N`. The split term in the output address
is NOT multiplied by outputStride, matching the source. -/
```
```lean
def originalIO : RealIO.IO p where
  kernel := (TypedSource.original input weight cast out_dtype p.input p.weights p.output p.N p.K p.indices
    p.inputBatchStride p.inputStride p.slotStride p.rowStride p.colStride p.outputBatchStride
    p.outputStride p.slice p.BN p.BK p.splits even add).eraseDType
  projection := RealIO.projection p input weight out_dtype cast Bool.false even add
  even := even
  add := add
  readIndex := fun batch => batch
  readInput := fun batch k => batch*p.inputBatchStride+k.val*p.inputStride
  readWeight := fun split slot g k => p.slotStride*slot.toNat+split*p.length*p.rowStride+g.val*p.rowStride+k.val*p.colStride
  write := fun split batch g => batch*p.outputBatchStride+split*p.length+p.slice*p.outputStride+g.val*p.outputStride
```
</details>

<details><summary><code>expected</code></summary>

```
/-- Independent mathematical sum. All three fields of xs are INITIAL input
values, including initialOutput; no computed output is assumed. -/
```
```lean
def expected (p : Parameters) (hK : p.K ≤ p.BK) (add : Bool) (xs : Inputs p) (g : Fin p.length) : ℝ :=
  let v := ∑ k : Fin p.K, xs.vector ⟨k.val, lt_of_lt_of_le k.isLt hK⟩ * xs.weights g k
  if add then v+xs.initialOutput g else v
```
</details>

<details><summary><code>original</code></summary>

```lean
def original (input : InputStorage) (weight : WeightStorage) (CAST_TYPE : Bool) : KernelSource :=
  specialize input weight CAST_TYPE Bool.false
```
</details>

<details><summary><code>specialize</code></summary>

```
/-- Specialize only compile-time precision/cast choices. All shape, stride,
program-ID, sentinel, masking and accumulation behavior remains in the source. -/
```
```lean
def specialize (input : InputStorage) (weight : WeightStorage)
    (CAST_TYPE swapped : Bool) : KernelSource :=
  match input, weight, CAST_TYPE, swapped with
  | .fp16, .fp16, Bool.false, Bool.false => bgmvSource%[tl.float16, tl.float16, keep, original]
  | .fp16, .fp16, Bool.false, Bool.true => bgmvSource%[tl.float16, tl.float16, keep, swapped]
  | .fp16, .fp16, Bool.true, Bool.false => bgmvSource%[tl.float16, tl.float16, cast, original]
  | .fp16, .fp16, Bool.true, Bool.true => bgmvSource%[tl.float16, tl.float16, cast, swapped]
  | .fp16, .bf16, Bool.false, Bool.false => bgmvSource%[tl.float16, tl.bfloat16, keep, original]
  | .fp16, .bf16, Bool.false, Bool.true => bgmvSource%[tl.float16, tl.bfloat16, keep, swapped]
  | .fp16, .bf16, Bool.true, Bool.false => bgmvSource%[tl.float16, tl.bfloat16, cast, original]
  | .fp16, .bf16, Bool.true, Bool.true => bgmvSource%[tl.float16, tl.bfloat16, cast, swapped]
  | .bf16, .fp16, Bool.false, Bool.false => bgmvSource%[tl.bfloat16, tl.float16, keep, original]
  | .bf16, .fp16, Bool.false, Bool.true => bgmvSource%[tl.bfloat16, tl.float16, keep, swapped]
  | .bf16, .fp16, Bool.true, Bool.false => bgmvSource%[tl.bfloat16, tl.float16, cast, original]
  | .bf16, .fp16, Bool.true, Bool.true => bgmvSource%[tl.bfloat16, tl.float16, cast, swapped]
  | .bf16, .bf16, Bool.false, Bool.false => bgmvSource%[tl.bfloat16, tl.bfloat16, keep, original]
  | .bf16, .bf16, Bool.false, Bool.true => bgmvSource%[tl.bfloat16, tl.bfloat16, keep, swapped]
  | .bf16, .bf16, Bool.true, Bool.false => bgmvSource%[tl.bfloat16, tl.bfloat16, cast, original]
  | .bf16, .bf16, Bool.true, Bool.true => bgmvSource%[tl.bfloat16, tl.bfloat16, cast, swapped]
  | .fp32, .fp16, Bool.false, Bool.false => bgmvSource%[tl.float32, tl.float16, keep, original]
  | .fp32, .fp16, Bool.false, Bool.true => bgmvSource%[tl.float32, tl.float16, keep, swapped]
  | .fp32, .fp16, Bool.true, Bool.false => bgmvSource%[tl.float32, tl.float16, cast, original]
  | .fp32, .fp16, Bool.true, Bool.true => bgmvSource%[tl.float32, tl.float16, cast, swapped]
  | .fp32, .bf16, Bool.false, Bool.false => bgmvSource%[tl.float32, tl.bfloat16, keep, original]
  | .fp32, .bf16, Bool.false, Bool.true => bgmvSource%[tl.float32, tl.bfloat16, keep, swapped]
  | .fp32, .bf16, Bool.true, Bool.false => bgmvSource%[tl.float32, tl.bfloat16, cast, original]
  | .fp32, .bf16, Bool.true, Bool.true => bgmvSource%[tl.float32, tl.bfloat16, cast, swapped]
```
</details>

## Public theorem: `bgmv_output_first_correctness`

**Statement:**
```lean
specification bgmv_output_first_correctness (p : Parameters)
    (input : TypedSource.InputStorage) (weight : TypedSource.WeightStorage) (out_dtype : FloatDType)
    (cast even add : Bool) (hBN : 0 < p.BN) (hstride : 0 < p.outputStride)
    (hK : p.K ≤ p.BK) (hsplits : 0 < p.splits) (hw : p.weights ≠ p.output) :
    Spec.Real (outputFirstIO p input weight out_dtype cast even add ⊨ expected p hK add)
```

**Assumptions / layout contracts:**
- `hBN : 0 < p.BN`
- `hstride : 0 < p.outputStride`
- `hK : p.K ≤ p.BK`
- `hsplits : 0 < p.splits`
- `hw : p.weights ≠ p.output`

**Closed-form spec defs (transitive):** `Parameters`, `outputFirstIO`, `expected`, `originalIO`, `outputFirst`, `original`, `specialize`

<details><summary><code>Parameters</code></summary>

```lean
abbrev Parameters := RealSetupSource.Parameters
```
</details>

<details><summary><code>outputFirstIO</code></summary>

```lean
def outputFirstIO : RealIO.IO p :=
  { originalIO p input weight out_dtype cast even add with
    kernel := (TypedSource.outputFirst input weight cast out_dtype p.input p.weights p.output p.N p.K p.indices
      p.inputBatchStride p.inputStride p.slotStride p.rowStride p.colStride p.outputBatchStride
      p.outputStride p.slice p.BN p.BK p.splits even add).eraseDType
    projection := RealIO.projection p input weight out_dtype cast Bool.true even add }
```
</details>

<details><summary><code>expected</code></summary>

```
/-- Independent mathematical sum. All three fields of xs are INITIAL input
values, including initialOutput; no computed output is assumed. -/
```
```lean
def expected (p : Parameters) (hK : p.K ≤ p.BK) (add : Bool) (xs : Inputs p) (g : Fin p.length) : ℝ :=
  let v := ∑ k : Fin p.K, xs.vector ⟨k.val, lt_of_lt_of_le k.isLt hK⟩ * xs.weights g k
  if add then v+xs.initialOutput g else v
```
</details>

<details><summary><code>originalIO</code></summary>

```
/-- Input has BK lanes, active when EVEN_K or k<K, even for an empty output
window. Weights cover every split row and K rank entries; initial output is
pinned at the write window only when ADD_INPUTS. Allocation roles are vector,
weights, output and signed metadata: the read/write output is allocated once.
`length = (N + SPLIT_N - 1) / SPLIT_N`. The split term in the output address
is NOT multiplied by outputStride, matching the source. -/
```
```lean
def originalIO : RealIO.IO p where
  kernel := (TypedSource.original input weight cast out_dtype p.input p.weights p.output p.N p.K p.indices
    p.inputBatchStride p.inputStride p.slotStride p.rowStride p.colStride p.outputBatchStride
    p.outputStride p.slice p.BN p.BK p.splits even add).eraseDType
  projection := RealIO.projection p input weight out_dtype cast Bool.false even add
  even := even
  add := add
  readIndex := fun batch => batch
  readInput := fun batch k => batch*p.inputBatchStride+k.val*p.inputStride
  readWeight := fun split slot g k => p.slotStride*slot.toNat+split*p.length*p.rowStride+g.val*p.rowStride+k.val*p.colStride
  write := fun split batch g => batch*p.outputBatchStride+split*p.length+p.slice*p.outputStride+g.val*p.outputStride
```
</details>

<details><summary><code>outputFirst</code></summary>

```lean
def outputFirst (input : InputStorage) (weight : WeightStorage) (CAST_TYPE : Bool) : KernelSource :=
  specialize input weight CAST_TYPE Bool.true
```
</details>

<details><summary><code>original</code></summary>

```lean
def original (input : InputStorage) (weight : WeightStorage) (CAST_TYPE : Bool) : KernelSource :=
  specialize input weight CAST_TYPE Bool.false
```
</details>

<details><summary><code>specialize</code></summary>

```
/-- Specialize only compile-time precision/cast choices. All shape, stride,
program-ID, sentinel, masking and accumulation behavior remains in the source. -/
```
```lean
def specialize (input : InputStorage) (weight : WeightStorage)
    (CAST_TYPE swapped : Bool) : KernelSource :=
  match input, weight, CAST_TYPE, swapped with
  | .fp16, .fp16, Bool.false, Bool.false => bgmvSource%[tl.float16, tl.float16, keep, original]
  | .fp16, .fp16, Bool.false, Bool.true => bgmvSource%[tl.float16, tl.float16, keep, swapped]
  | .fp16, .fp16, Bool.true, Bool.false => bgmvSource%[tl.float16, tl.float16, cast, original]
  | .fp16, .fp16, Bool.true, Bool.true => bgmvSource%[tl.float16, tl.float16, cast, swapped]
  | .fp16, .bf16, Bool.false, Bool.false => bgmvSource%[tl.float16, tl.bfloat16, keep, original]
  | .fp16, .bf16, Bool.false, Bool.true => bgmvSource%[tl.float16, tl.bfloat16, keep, swapped]
  | .fp16, .bf16, Bool.true, Bool.false => bgmvSource%[tl.float16, tl.bfloat16, cast, original]
  | .fp16, .bf16, Bool.true, Bool.true => bgmvSource%[tl.float16, tl.bfloat16, cast, swapped]
  | .bf16, .fp16, Bool.false, Bool.false => bgmvSource%[tl.bfloat16, tl.float16, keep, original]
  | .bf16, .fp16, Bool.false, Bool.true => bgmvSource%[tl.bfloat16, tl.float16, keep, swapped]
  | .bf16, .fp16, Bool.true, Bool.false => bgmvSource%[tl.bfloat16, tl.float16, cast, original]
  | .bf16, .fp16, Bool.true, Bool.true => bgmvSource%[tl.bfloat16, tl.float16, cast, swapped]
  | .bf16, .bf16, Bool.false, Bool.false => bgmvSource%[tl.bfloat16, tl.bfloat16, keep, original]
  | .bf16, .bf16, Bool.false, Bool.true => bgmvSource%[tl.bfloat16, tl.bfloat16, keep, swapped]
  | .bf16, .bf16, Bool.true, Bool.false => bgmvSource%[tl.bfloat16, tl.bfloat16, cast, original]
  | .bf16, .bf16, Bool.true, Bool.true => bgmvSource%[tl.bfloat16, tl.bfloat16, cast, swapped]
  | .fp32, .fp16, Bool.false, Bool.false => bgmvSource%[tl.float32, tl.float16, keep, original]
  | .fp32, .fp16, Bool.false, Bool.true => bgmvSource%[tl.float32, tl.float16, keep, swapped]
  | .fp32, .fp16, Bool.true, Bool.false => bgmvSource%[tl.float32, tl.float16, cast, original]
  | .fp32, .fp16, Bool.true, Bool.true => bgmvSource%[tl.float32, tl.float16, cast, swapped]
  | .fp32, .bf16, Bool.false, Bool.false => bgmvSource%[tl.float32, tl.bfloat16, keep, original]
  | .fp32, .bf16, Bool.false, Bool.true => bgmvSource%[tl.float32, tl.bfloat16, keep, swapped]
  | .fp32, .bf16, Bool.true, Bool.false => bgmvSource%[tl.float32, tl.bfloat16, cast, original]
  | .fp32, .bf16, Bool.true, Bool.true => bgmvSource%[tl.float32, tl.bfloat16, cast, swapped]
```
</details>
