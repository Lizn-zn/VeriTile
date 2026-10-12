# Spec sheet — `bench/tritonbench_g/dequantize_matmul/Correct.lean`

**Python source:** `bench/tritonbench_g/dequantize_matmul/dequantize_matmul.py`

## Public theorem: `original_correct`

**Statement:**
```lean
specification original_correct (output : Storage) (weights scales out : RegionName)
    (K N weightRow weightCol outputRow outputCol BN BK : Nat)
    (positiveKBlock : 0<BK) (distinctOutputs : outputInjective outputRow outputCol BN BK) :
    Spec.Real (originalIO output weights scales out K N weightRow weightCol outputRow outputCol BN BK
      ⊨ᵢ fun _ _ xs ys j => (xs j : ℝ)*ys j)
```

**Assumptions / layout contracts:**
- `positiveKBlock : 0<BK`

**Closed-form spec defs (transitive):** `outputInjective`, `originalIO`, `kernel`

<details><summary><code>outputInjective</code></summary>

```
/-- Distinct tile lanes must target distinct output cells. -/
```
```lean
def outputInjective (outputRow outputCol BN BK : Nat) : Prop :=
  ∀ pk pn : Nat, Function.Injective fun j : Fin (BK*BN) =>
    (pk*BK+j.val/BN)*outputRow+(pn*BN+j.val%BN)*outputCol
```
</details>

<details><summary><code>originalIO</code></summary>

```
/-- The scale mask is column-only: the source reads these cells even if no
weight or output lane in a K row is active. -/
```
```lean
def originalIO (output : Storage) (weights scales out : RegionName)
    (K N weightRow weightCol outputRow outputCol BN BK : Nat) : Masked2DKernelIO₂ where
  kernel := (kernel output Bool.false weights scales out K N weightRow weightCol outputRow outputCol BN BK).eraseDType
  projection := by
    simp only [ComputeKernel.eraseDType,Projection.source_projects,
      ComputeKernel.toAlgorithm?_fromKernelBody,ComputeKernel.toAlgKernel_fromKernelBody]
  in1 := weights
  in2 := scales
  out := out
  B := BK*BN
  read1 := fun pk pn j => (pk*BK+j.val/BN)*weightRow+(pn*BN+j.val%BN)*weightCol
  read2 := fun _ pn j => pn*BN+j.val%BN
  write := fun pk pn j => (pk*BK+j.val/BN)*outputRow+(pn*BN+j.val%BN)*outputCol
  mask := fun pk pn j => pk*BK+j.val/BN<K ∧ pn*BN+j.val%BN<N
  read2Mask := fun _ pn j => pn*BN+j.val%BN<N
```
</details>

<details><summary><code>kernel</code></summary>

```lean
def kernel (output : Storage) (swapped : Bool) := match output with
  | .fp16 => if swapped then dequantizeMatmulSource%[tl.float16,indexCommuted] else dequantizeMatmulSource%[tl.float16,original]
  | .bf16 => if swapped then dequantizeMatmulSource%[tl.bfloat16,indexCommuted] else dequantizeMatmulSource%[tl.bfloat16,original]
  | .fp32 => if swapped then dequantizeMatmulSource%[tl.float32,indexCommuted] else dequantizeMatmulSource%[tl.float32,original]
```
</details>

## Public theorem: `index_commuted_correct`

**Statement:**
```lean
specification index_commuted_correct (output : Storage) (weights scales out : RegionName)
    (K N weightRow weightCol outputRow outputCol BN BK : Nat)
    (positiveKBlock : 0<BK) (distinctOutputs : outputInjective outputRow outputCol BN BK) :
    Spec.Real (indexCommutedIO output weights scales out K N weightRow weightCol outputRow outputCol BN BK
      ⊨ᵢ fun _ _ xs ys j => (xs j : ℝ)*ys j)
```

**Assumptions / layout contracts:**
- `positiveKBlock : 0<BK`

**Closed-form spec defs (transitive):** `outputInjective`, `indexCommutedIO`, `originalIO`, `kernel`

<details><summary><code>outputInjective</code></summary>

```
/-- Distinct tile lanes must target distinct output cells. -/
```
```lean
def outputInjective (outputRow outputCol BN BK : Nat) : Prop :=
  ∀ pk pn : Nat, Function.Injective fun j : Fin (BK*BN) =>
    (pk*BK+j.val/BN)*outputRow+(pn*BN+j.val%BN)*outputCol
```
</details>

<details><summary><code>indexCommutedIO</code></summary>

```lean
def indexCommutedIO (output : Storage) (weights scales out : RegionName)
    (K N weightRow weightCol outputRow outputCol BN BK : Nat) : Masked2DKernelIO₂ :=
  { originalIO output weights scales out K N weightRow weightCol outputRow outputCol BN BK with
    kernel := (kernel output Bool.true weights scales out K N weightRow weightCol outputRow outputCol BN BK).eraseDType
    projection := by
      simp only [ComputeKernel.eraseDType,Projection.source_projects,
        ComputeKernel.toAlgorithm?_fromKernelBody,ComputeKernel.toAlgKernel_fromKernelBody] }
```
</details>

<details><summary><code>originalIO</code></summary>

```
/-- The scale mask is column-only: the source reads these cells even if no
weight or output lane in a K row is active. -/
```
```lean
def originalIO (output : Storage) (weights scales out : RegionName)
    (K N weightRow weightCol outputRow outputCol BN BK : Nat) : Masked2DKernelIO₂ where
  kernel := (kernel output Bool.false weights scales out K N weightRow weightCol outputRow outputCol BN BK).eraseDType
  projection := by
    simp only [ComputeKernel.eraseDType,Projection.source_projects,
      ComputeKernel.toAlgorithm?_fromKernelBody,ComputeKernel.toAlgKernel_fromKernelBody]
  in1 := weights
  in2 := scales
  out := out
  B := BK*BN
  read1 := fun pk pn j => (pk*BK+j.val/BN)*weightRow+(pn*BN+j.val%BN)*weightCol
  read2 := fun _ pn j => pn*BN+j.val%BN
  write := fun pk pn j => (pk*BK+j.val/BN)*outputRow+(pn*BN+j.val%BN)*outputCol
  mask := fun pk pn j => pk*BK+j.val/BN<K ∧ pn*BN+j.val%BN<N
  read2Mask := fun _ pn j => pn*BN+j.val%BN<N
```
</details>

<details><summary><code>kernel</code></summary>

```lean
def kernel (output : Storage) (swapped : Bool) := match output with
  | .fp16 => if swapped then dequantizeMatmulSource%[tl.float16,indexCommuted] else dequantizeMatmulSource%[tl.float16,original]
  | .bf16 => if swapped then dequantizeMatmulSource%[tl.bfloat16,indexCommuted] else dequantizeMatmulSource%[tl.bfloat16,original]
  | .fp32 => if swapped then dequantizeMatmulSource%[tl.float32,indexCommuted] else dequantizeMatmulSource%[tl.float32,original]
```
</details>
