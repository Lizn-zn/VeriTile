# Spec sheet — `bench/tritonbench_g/kldiv_triton/KldivTriton.lean`

**Python source:** `bench/tritonbench_g/kldiv_triton/kldiv_triton.py`

## Public theorem: `kldiv_backward_default_compute_correct`

<details><summary>docstring</summary>

```
/-- Compute-facing correctness for the default backward kernel under the
Python-tested single-chunk regime `0 < n_cols ≤ BLOCK_SIZE`. -/
```
</details>

**Statement:**
```lean
specification kldiv_backward_default_compute_correct
    (input_ptr target_ptr : RegionName)
    (input_stride target_stride n_cols BLOCK_SIZE : Nat)
    (s : BlockState)
    (hBS : 0 < BLOCK_SIZE)
    (hLen : n_cols ≤ BLOCK_SIZE)
    (hLenPos : 0 < n_cols)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => linearOffset s input_stride i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := kldiv_backward_default input_ptr target_ptr
        input_stride target_stride n_cols BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < n_cols)
        (fun i => (input_ptr, linearOffset s input_stride i)))
      (expected := fun i => defaultSpec s target_ptr target_stride i)
```

**Assumptions / layout contracts:**
- `hBS : 0 < BLOCK_SIZE`
- `hLen : n_cols ≤ BLOCK_SIZE`
- `hLenPos : 0 < n_cols`
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => linearOffset s input_stride i)`
- `fun i : Fin BLOCK_SIZE => i.val < n_cols`

**Closed-form spec defs (transitive):** `kldiv_backward_default`, `defaultSpec`, `inOffset`

<details><summary><code>kldiv_backward_default</code></summary>

```
/-- Faithful transcription of `kldiv_triton.py`'s `_kldiv_kernel_backward`
for the `log_target = False` constexpr branch.

Includes the Python `for i in range(0, n_cols, BLOCK_SIZE)` loop. Proofs
target the Python-tested single-chunk regime `0 < n_cols ≤ BLOCK_SIZE`.

Allowed mechanical Lean-syntax-only changes:
- Python `log_target: tl.constexpr` → separate kernel defs per branch. -/
```
```lean
def kldiv_backward_default
    (input_ptr target_ptr : RegionName)
    (input_stride target_stride n_cols BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0).to(tl.int64)
  input_ptr += pid * $(input_stride)
  target_ptr += pid * $(target_stride)
  base_offsets = tl.arange(0, $(BLOCK_SIZE))
  for i in range($(0), $(n_cols), $(BLOCK_SIZE)) {
    offsets = i + base_offsets
    mask = offsets < $(n_cols)
    target = tl.load(target_ptr + offsets, mask=mask, other=0.0)
    res = target * -1
    tl.store(input_ptr + offsets, res, mask=mask)
  }
}
```
</details>

<details><summary><code>defaultSpec</code></summary>

```lean
noncomputable def defaultSpec
    (s : BlockState) (target_ptr : RegionName) (target_stride : Nat)
    (i : Fin BLOCK_SIZE) : ℝ :=
  s.readMem target_ptr (inOffset s target_stride i) * (0.0 - 1)
```
</details>

<details><summary><code>inOffset</code></summary>

```lean
def inOffset (s : BlockState) (target_stride : Nat) (i : Fin BLOCK_SIZE) :
    Nat :=
  s.pid * target_stride + i.val
```
</details>

## Public theorem: `kldiv_backward_log_target_compute_correct`

<details><summary>docstring</summary>

```
/-- Compute-facing correctness for the log-target backward kernel under the
Python-tested single-chunk regime `0 < n_cols ≤ BLOCK_SIZE`. -/
```
</details>

**Statement:**
```lean
specification kldiv_backward_log_target_compute_correct
    (input_ptr target_ptr : RegionName)
    (input_stride target_stride n_cols BLOCK_SIZE : Nat)
    (s : BlockState)
    (hBS : 0 < BLOCK_SIZE)
    (hLen : n_cols ≤ BLOCK_SIZE)
    (hLenPos : 0 < n_cols)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => linearOffset s input_stride i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := kldiv_backward_log_target input_ptr target_ptr
        input_stride target_stride n_cols BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < n_cols)
        (fun i => (input_ptr, linearOffset s input_stride i)))
      (expected := fun i => logTargetSpec s target_ptr target_stride i)
```

**Assumptions / layout contracts:**
- `hBS : 0 < BLOCK_SIZE`
- `hLen : n_cols ≤ BLOCK_SIZE`
- `hLenPos : 0 < n_cols`
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => linearOffset s input_stride i)`
- `fun i : Fin BLOCK_SIZE => i.val < n_cols`

**Closed-form spec defs (transitive):** `kldiv_backward_log_target`, `logTargetSpec`, `inOffset`

<details><summary><code>kldiv_backward_log_target</code></summary>

```
/-- Faithful transcription of `_kldiv_kernel_backward` for the
`log_target = True` constexpr branch. Includes the Python
`for i in range(0, n_cols, BLOCK_SIZE)` loop; proofs target the
Python-tested single-chunk regime `0 < n_cols ≤ BLOCK_SIZE`. -/
```
```lean
def kldiv_backward_log_target
    (input_ptr target_ptr : RegionName)
    (input_stride target_stride n_cols BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0).to(tl.int64)
  input_ptr += pid * $(input_stride)
  target_ptr += pid * $(target_stride)
  base_offsets = tl.arange(0, $(BLOCK_SIZE))
  for i in range($(0), $(n_cols), $(BLOCK_SIZE)) {
    offsets = i + base_offsets
    mask = offsets < $(n_cols)
    target = tl.load(target_ptr + offsets, mask=mask, other=0.0)
    res = -tl.exp(target)
    tl.store(input_ptr + offsets, res, mask=mask)
  }
}
```
</details>

<details><summary><code>logTargetSpec</code></summary>

```lean
noncomputable def logTargetSpec
    (s : BlockState) (target_ptr : RegionName) (target_stride : Nat)
    (i : Fin BLOCK_SIZE) : ℝ :=
  0.0 - Real.exp (s.readMem target_ptr (inOffset s target_stride i))
```
</details>

<details><summary><code>inOffset</code></summary>

```lean
def inOffset (s : BlockState) (target_stride : Nat) (i : Fin BLOCK_SIZE) :
    Nat :=
  s.pid * target_stride + i.val
```
</details>

## Public theorem: `kldiv_forward_default_none_compute_correct`

<details><summary>docstring</summary>

```
/-- Compute-facing correctness for the forward `log_target=False`,
`reduction=0` kernel under the Python-tested single-chunk regime
`0 < n_cols ≤ BLOCK_SIZE`. -/
```
</details>

**Statement:**
```lean
specification kldiv_forward_default_none_compute_correct
    (y_ptr gt_ptr loss_ptr : RegionName)
    (y_stride gt_stride loss_stride n_cols BLOCK_SIZE : Nat)
    (s : BlockState)
    (hBS : 0 < BLOCK_SIZE)
    (hLen : n_cols ≤ BLOCK_SIZE)
    (hLenPos : 0 < n_cols)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => linearOffset s loss_stride i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := kldiv_forward_default_none y_ptr gt_ptr loss_ptr
        y_stride gt_stride loss_stride n_cols BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < n_cols)
        (fun i => (loss_ptr, linearOffset s loss_stride i)))
      (expected := fun i =>
        forwardDefaultSpec s y_ptr gt_ptr y_stride gt_stride i)
```

**Assumptions / layout contracts:**
- `hBS : 0 < BLOCK_SIZE`
- `hLen : n_cols ≤ BLOCK_SIZE`
- `hLenPos : 0 < n_cols`
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => linearOffset s loss_stride i)`
- `fun i : Fin BLOCK_SIZE => i.val < n_cols`

**Closed-form spec defs (transitive):** `kldiv_forward_default_none`, `forwardDefaultSpec`, `inOffset`

<details><summary><code>kldiv_forward_default_none</code></summary>

```
/-- Faithful transcription of `kldiv_triton.py`'s `_kldiv_kernel_forward`
for the `log_target = False`, `reduction = 0` (None) constexpr branch.

Includes the Python `for i in range(0, n_cols, BLOCK_SIZE)` loop; proofs
target the Python-tested single-chunk regime `0 < n_cols ≤ BLOCK_SIZE`.
Mirrors the Python `loss = y_true * (tl.log(y_true) - y)` body; no `eps`
clamp (unlike the kldiv_ops port). -/
```
```lean
def kldiv_forward_default_none
    (y_ptr gt_ptr loss_ptr : RegionName)
    (y_stride gt_stride loss_stride n_cols BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0).to(tl.int64)
  y_ptr += pid * $(y_stride)
  gt_ptr += pid * $(gt_stride)
  loss_ptr += pid * $(loss_stride)
  base_offsets = tl.arange(0, $(BLOCK_SIZE))
  for i in range($(0), $(n_cols), $(BLOCK_SIZE)) {
    offsets = i + base_offsets
    mask = offsets < $(n_cols)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    y_true = tl.load(gt_ptr + offsets, mask=mask, other=0.0)
    loss = y_true * (tl.log(y_true) - y)
    tl.store(loss_ptr + offsets, loss, mask=mask)
  }
}
```
</details>

<details><summary><code>forwardDefaultSpec</code></summary>

```
/-- Forward KL-divergence per-element value (`log_target = False`)
for the `reduction = 0` (None) elementwise-store path. -/
```
```lean
noncomputable def forwardDefaultSpec
    (s : BlockState) (y_ptr gt_ptr : RegionName)
    (y_stride gt_stride : Nat) (i : Fin BLOCK_SIZE) : ℝ :=
  let y := s.readMem y_ptr (inOffset s y_stride i)
  let y_true := s.readMem gt_ptr (inOffset s gt_stride i)
  y_true * (Real.log y_true - y)
```
</details>

<details><summary><code>inOffset</code></summary>

```lean
def inOffset (s : BlockState) (target_stride : Nat) (i : Fin BLOCK_SIZE) :
    Nat :=
  s.pid * target_stride + i.val
```
</details>

## Public theorem: `kldiv_forward_log_target_none_compute_correct`

<details><summary>docstring</summary>

```
/-- Compute-facing correctness for the forward log-target kernel under the
Python-tested single-chunk regime `0 < n_cols ≤ BLOCK_SIZE`. -/
```
</details>

**Statement:**
```lean
specification kldiv_forward_log_target_none_compute_correct
    (y_ptr gt_ptr loss_ptr : RegionName)
    (y_stride gt_stride loss_stride n_cols BLOCK_SIZE : Nat)
    (s : BlockState)
    (hBS : 0 < BLOCK_SIZE)
    (hLen : n_cols ≤ BLOCK_SIZE)
    (hLenPos : 0 < n_cols)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => linearOffset s loss_stride i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := kldiv_forward_log_target_none y_ptr gt_ptr loss_ptr
        y_stride gt_stride loss_stride n_cols BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < n_cols)
        (fun i => (loss_ptr, linearOffset s loss_stride i)))
      (expected := fun i =>
        forwardLogTargetSpec s y_ptr gt_ptr y_stride gt_stride i)
```

**Assumptions / layout contracts:**
- `hBS : 0 < BLOCK_SIZE`
- `hLen : n_cols ≤ BLOCK_SIZE`
- `hLenPos : 0 < n_cols`
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => linearOffset s loss_stride i)`
- `fun i : Fin BLOCK_SIZE => i.val < n_cols`

**Closed-form spec defs (transitive):** `kldiv_forward_log_target_none`, `forwardLogTargetSpec`, `inOffset`

<details><summary><code>kldiv_forward_log_target_none</code></summary>

```
/-- Faithful transcription of `kldiv_triton.py`'s `_kldiv_kernel_forward`
for the `log_target = True`, `reduction = 0` (None) constexpr branch.

Includes the Python `for i in range(0, n_cols, BLOCK_SIZE)` loop with the
elementwise-store path; proofs target the Python-tested single-chunk regime
`0 < n_cols ≤ BLOCK_SIZE`. -/
```
```lean
def kldiv_forward_log_target_none
    (y_ptr gt_ptr loss_ptr : RegionName)
    (y_stride gt_stride loss_stride n_cols BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0).to(tl.int64)
  y_ptr += pid * $(y_stride)
  gt_ptr += pid * $(gt_stride)
  loss_ptr += pid * $(loss_stride)
  base_offsets = tl.arange(0, $(BLOCK_SIZE))
  for i in range($(0), $(n_cols), $(BLOCK_SIZE)) {
    offsets = i + base_offsets
    mask = offsets < $(n_cols)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    y_true = tl.load(gt_ptr + offsets, mask=mask, other=0.0)
    loss = tl.exp(y_true) * (y_true - y)
    tl.store(loss_ptr + offsets, loss, mask=mask)
  }
}
```
</details>

<details><summary><code>forwardLogTargetSpec</code></summary>

```
/-- Forward KL-divergence per-element value (`log_target = True`)
for the `reduction = 0` (None) elementwise-store path. -/
```
```lean
noncomputable def forwardLogTargetSpec
    (s : BlockState) (y_ptr gt_ptr : RegionName)
    (y_stride gt_stride : Nat) (i : Fin BLOCK_SIZE) : ℝ :=
  let y := s.readMem y_ptr (inOffset s y_stride i)
  let y_true := s.readMem gt_ptr (inOffset s gt_stride i)
  Real.exp y_true * (y_true - y)
```
</details>

<details><summary><code>inOffset</code></summary>

```lean
def inOffset (s : BlockState) (target_stride : Nat) (i : Fin BLOCK_SIZE) :
    Nat :=
  s.pid * target_stride + i.val
```
</details>
