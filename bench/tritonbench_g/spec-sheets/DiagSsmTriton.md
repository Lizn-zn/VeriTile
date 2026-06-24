# Spec sheet — `bench/tritonbench_g/diag_ssm_triton/DiagSsmTriton.lean`

**Python source:** `bench/tritonbench_g/diag_ssm_triton/diag_ssm_triton.py`

## Public theorem: `diag_ssm_forward_kernel_output_summary`

<details><summary>docstring</summary>

```
/-- Per-kernel output summary for `diag_ssm_forward_kernel`: the DSL surface
lowers to the algorithm layer, and the time-step stores to `y_ptr` are
compute-correct — after the `0..length` recurrent scan every active output
offset holds the diagonal-SSM spec value `diagSsmForwardSpecAt`, and inactive
lanes are preserved. Mirrors `add_kernel_output_summary`. -/
```
</details>

**Statement:**
```lean
theorem diag_ssm_forward_kernel_output_summary
    (s_ptr x_ptr lambda_ptr y_ptr : RegionName)
    (length batch_size dim BLOCK_SIZE : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [length, BLOCK_SIZE] =>
        diagSsmForwardOutOffset s batch_size dim BLOCK_SIZE idx))
    (hXOutNe : x_ptr ≠ y_ptr) :
    (∃ alg, (diag_ssm_forward_kernel s_ptr x_ptr lambda_ptr y_ptr
        length batch_size dim BLOCK_SIZE).toAlgorithm? = Except.ok alg) ∧
    diag_ssm_forward_kernel_correct_target s_ptr x_ptr lambda_ptr y_ptr
      length batch_size dim BLOCK_SIZE s
```

**Assumptions / layout contracts:**
- `hOutInj : Function.Injective
      (fun idx : TileIndex [length, BLOCK_SIZE] =>
        diagSsmForwardOutOffset s batch_size dim BLOCK_SIZE idx)`
- `hXOutNe : x_ptr ≠ y_ptr`

**Closed-form spec defs (transitive):** `diagSsmForwardOutOffset`, `diag_ssm_forward_kernel`, `diag_ssm_forward_kernel_correct_target`, `timeOffset`, `diagSsmForwardActive`, `diagSsmForwardSpecAt`, `colOffset`, `active`, `diagSsmForwardSpec`, `diagSsmStateAfter`

<details><summary><code>diagSsmForwardOutOffset</code></summary>

```lean
def diagSsmForwardOutOffset
    {length : Nat}
    (st : BlockState) (batch_size dim BLOCK_SIZE : Nat)
    (idx : TileIndex [length, BLOCK_SIZE]) : Nat :=
  timeOffset st batch_size dim BLOCK_SIZE idx.1.val idx.2.1
```
</details>

<details><summary><code>diag_ssm_forward_kernel</code></summary>

```
/-- Faithful transcription of `diag_ssm_triton.py`'s
`diag_ssm_forward_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` → Lean `Nat` parameter.
- Python `length`, `batch_size`, `dim` → Lean `Nat` parameters.

The proof below connects the recurrence invariant across `tl.for t in length`
to `ComputeCorrect.Realizes` under the stated no-collision/no-alias
hypotheses. -/
```
```lean
def diag_ssm_forward_kernel
    (s_ptr x_ptr lambda_ptr y_ptr : RegionName)
    (length batch_size dim BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  col_idx = tl.program_id(0) * $(BLOCK_SIZE)
  col_offsets = col_idx + tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(batch_size * dim)
  s = tl.load(s_ptr + col_offsets, mask=mask, other=0)
  Lambda = tl.load(lambda_ptr + col_offsets % $(dim), mask=mask, other=0)
  tl.for t in $(length) {
    offsets = t * $(batch_size * dim) + col_offsets
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    s = s * Lambda + x
    tl.store(y_ptr + offsets, s, mask=mask)
  }
}
```
</details>

<details><summary><code>diag_ssm_forward_kernel_correct_target</code></summary>

```lean
def diag_ssm_forward_kernel_correct_target
    (s_ptr x_ptr lambda_ptr y_ptr : RegionName)
    (length batch_size dim BLOCK_SIZE : Nat) (s : BlockState) : Prop :=
  ComputeCorrect.Realizes
    (kernel := diag_ssm_forward_kernel s_ptr x_ptr lambda_ptr y_ptr
      length batch_size dim BLOCK_SIZE)
    (initialState := s)
    (write := ComputeCorrect.WriteMap.writeIf
      (fun idx : TileIndex [length, BLOCK_SIZE] =>
        diagSsmForwardActive s batch_size dim BLOCK_SIZE idx)
      (fun idx : TileIndex [length, BLOCK_SIZE] =>
        (y_ptr, diagSsmForwardOutOffset s batch_size dim BLOCK_SIZE idx)))
    (expected := fun idx : TileIndex [length, BLOCK_SIZE] =>
      diagSsmForwardSpecAt s s_ptr x_ptr lambda_ptr batch_size dim BLOCK_SIZE idx)
```
</details>

<details><summary><code>timeOffset</code></summary>

```lean
def timeOffset
    (st : BlockState) (batch_size dim BLOCK_SIZE t : Nat)
    (i : Fin BLOCK_SIZE) : Nat :=
  t * (batch_size * dim) + colOffset st BLOCK_SIZE i
```
</details>

<details><summary><code>diagSsmForwardActive</code></summary>

```lean
def diagSsmForwardActive
    {length : Nat}
    (st : BlockState) (batch_size dim BLOCK_SIZE : Nat)
    (idx : TileIndex [length, BLOCK_SIZE]) : Prop :=
  active st batch_size dim BLOCK_SIZE idx.2.1
```
</details>

<details><summary><code>diagSsmForwardSpecAt</code></summary>

```lean
noncomputable def diagSsmForwardSpecAt
    {length : Nat}
    (st : BlockState) (s_ptr x_ptr lambda_ptr : RegionName)
    (batch_size dim BLOCK_SIZE : Nat)
    (idx : TileIndex [length, BLOCK_SIZE]) : ℝ :=
  diagSsmForwardSpec st s_ptr x_ptr lambda_ptr batch_size dim BLOCK_SIZE
    idx.1.val idx.2.1
```
</details>

<details><summary><code>colOffset</code></summary>

```
/-- Faithful transcription of `diag_ssm_triton.py`'s
`diag_ssm_backward_kernel_complex`. -/
def diag_ssm_backward_kernel_complex
    (s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr :
      RegionName)
    (length batch_size dim BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  col_idx = tl.program_id(0) * $(BLOCK_SIZE)
  col_offsets = col_idx + tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(batch_size * dim)
  lambda_real = tl.load(lambda_ptr + (col_offsets % $(dim)) * $(2),
    mask=mask, other=0)
  lambda_imag = tl.load(lambda_ptr + (col_offsets % $(dim)) * $(2) + $(1),
    mask=mask, other=0)
  grad_s_real = tl.zeros_like(lambda_real)
  grad_s_imag = tl.zeros_like(lambda_imag)
  grad_lambda_real = tl.zeros_like(lambda_real)
  grad_lambda_imag = tl.zeros_like(lambda_imag)
  for i in range(0, $(length), $(1)) {
    t = $(length) - $(1) - i
    offsets = (t * $(batch_size * dim) + col_offsets) * $(2)
    grad_y_real = tl.load(grad_y_ptr + offsets, mask=mask, other=0)
    grad_y_imag = -tl.load(grad_y_ptr + offsets + $(1), mask=mask, other=0)
    if t > 0 {
      s_real = tl.load(y_ptr + (offsets - $(2 * batch_size * dim)),
        mask=mask, other=0)
      s_imag = tl.load(y_ptr + (offsets - $(2 * batch_size * dim) + $(1)),
        mask=mask, other=0)
    } else {
      s_real = tl.load(s_ptr + $(2) * col_offsets, mask=mask, other=0)
      s_imag = tl.load(s_ptr + $(2) * col_offsets + $(1), mask=mask, other=0)
    }
    grad_s_real = grad_y_real + grad_s_real
    grad_s_imag = grad_y_imag + grad_s_imag
    grad_x_real = grad_s_real
    grad_x_imag = grad_s_imag
    grad_lambda_real += grad_s_real * s_real - grad_s_imag * s_imag
    grad_lambda_imag += grad_s_real * s_imag + grad_s_imag * s_real
    grad_s_real = grad_x_real * lambda_real - grad_x_imag * lambda_imag
    grad_s_imag = grad_x_real * lambda_imag + grad_x_imag * lambda_real
    tl.store(grad_x_ptr + offsets, grad_x_real, mask=mask)
    tl.store(grad_x_ptr + offsets + $(1), -grad_x_imag, mask=mask)
  }
  tl.store(grad_s_ptr + col_offsets * $(2), grad_s_real, mask=mask)
  tl.store(grad_s_ptr + col_offsets * $(2) + $(1), -grad_s_imag, mask=mask)
  tl.store(grad_lambda_ptr + col_offsets * $(2), grad_lambda_real, mask=mask)
  tl.store(grad_lambda_ptr + col_offsets * $(2) + $(1), -grad_lambda_imag,
    mask=mask)
}

/-
Complex correctness blocker (#119): the complex forward/backward surfaces above
are faithful transcriptions and pass the port checker, but this file only proves
`ComputeCorrect.Realizes` for the real-valued kernels. The complex kernels encode
complex tensors as paired real memory lanes, including conjugated gradient signs;
their readback theorem should use a paired-lane output map/spec rather than the
single-real-lane recurrence used below.
-/
```
```lean
def colOffset (st : BlockState) (BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  st.pids 0 * BLOCK_SIZE + i.val
```
</details>

<details><summary><code>active</code></summary>

```lean
def active (st : BlockState) (batch_size dim BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : Prop :=
  colOffset st BLOCK_SIZE i < batch_size * dim
```
</details>

<details><summary><code>diagSsmForwardSpec</code></summary>

```lean
noncomputable def diagSsmForwardSpec
    (st : BlockState) (s_ptr x_ptr lambda_ptr : RegionName)
    (batch_size dim BLOCK_SIZE : Nat) (t : Nat) (i : Fin BLOCK_SIZE) : ℝ :=
  diagSsmStateAfter st s_ptr x_ptr lambda_ptr batch_size dim BLOCK_SIZE i (t + 1)
```
</details>

<details><summary><code>diagSsmStateAfter</code></summary>

```lean
noncomputable def diagSsmStateAfter
    (st : BlockState) (s_ptr x_ptr lambda_ptr : RegionName)
    (batch_size dim BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) : Nat → ℝ
  | 0 => st.readMem s_ptr (colOffset st BLOCK_SIZE i)
  | t + 1 =>
      diagSsmStateAfter st s_ptr x_ptr lambda_ptr batch_size dim BLOCK_SIZE i t *
          st.readMem lambda_ptr (IntegralDType.nat.mod (colOffset st BLOCK_SIZE i) dim) +
        st.readMem x_ptr (timeOffset st batch_size dim BLOCK_SIZE t i)
```
</details>

## Also present (pinned special-case summaries)
- `diag_ssm_backward_kernel_compute_correct`
- `diag_ssm_forward_kernel_compute_correct`
