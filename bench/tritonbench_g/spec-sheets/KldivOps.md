# Spec sheet — `bench/tritonbench_g/kldiv_ops/KldivOps.lean`

**Python source:** `bench/tritonbench_g/kldiv_ops/kldiv_ops.py`

## Public theorem: `kldiv_backward_default_compute_correct`

<details><summary>docstring</summary>

```
/-- Compute-facing correctness for the default backward one-block slice. -/
```
</details>

**Statement:**
```lean
theorem kldiv_backward_default_compute_correct
    (target_ptr new_grads_ptr : RegionName)
    (target_stride new_grads_stride n_cols BLOCK_SIZE : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => linearOffset s new_grads_stride i)) :
    ComputeCorrect.Realizes
      (kernel := kldiv_backward_default target_ptr new_grads_ptr
        target_stride new_grads_stride n_cols BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < n_cols)
        (fun i => (new_grads_ptr, linearOffset s new_grads_stride i)))
      (expected := fun i => defaultSpec s target_ptr target_stride i)
```

**Assumptions / layout contracts:**
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => linearOffset s new_grads_stride i)`
- `kernel : = kldiv_backward_default target_ptr new_grads_ptr
        target_stride new_grads_stride n_cols BLOCK_SIZE`
- `initialState : = s`
- `fun i : Fin BLOCK_SIZE => i.val < n_cols`
- `expected : = fun i => defaultSpec s target_ptr target_stride i`

**Closed-form spec defs (transitive):** `kldiv_backward_default`, `defaultSpec`, `inOffset`

<details><summary><code>kldiv_backward_default</code></summary>

```
/-- Documented one-block slice of `kldiv_ops.py`'s `_kldiv_kernel_backward`
for the `log_target = False` constexpr branch.

This models one `BLOCK_SIZE` iteration of Python's `for i in range(0, n_cols,
BLOCK_SIZE)` loop after the row pointers have been advanced.

Allowed mechanical Lean-syntax-only changes:
- Python `log_target: tl.constexpr` → separate kernel defs per branch. -/
```
```lean
def kldiv_backward_default
    (target_ptr new_grads_ptr : RegionName)
    (target_stride new_grads_stride n_cols BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0).to(tl.int64)
  target_ptr += pid * $(target_stride)
  new_grads_ptr += pid * $(new_grads_stride)
  offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_cols)
  target = tl.load(target_ptr + offsets, mask=mask, other=0.0)
  res = target * -1
  tl.store(new_grads_ptr + offsets, res, mask=mask)
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
/-- Compute-facing correctness for the log-target backward one-block slice. -/
```
</details>

**Statement:**
```lean
theorem kldiv_backward_log_target_compute_correct
    (target_ptr new_grads_ptr : RegionName)
    (target_stride new_grads_stride n_cols BLOCK_SIZE : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => linearOffset s new_grads_stride i)) :
    ComputeCorrect.Realizes
      (kernel := kldiv_backward_log_target target_ptr new_grads_ptr
        target_stride new_grads_stride n_cols BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < n_cols)
        (fun i => (new_grads_ptr, linearOffset s new_grads_stride i)))
      (expected := fun i => logTargetSpec s target_ptr target_stride i)
```

**Assumptions / layout contracts:**
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => linearOffset s new_grads_stride i)`
- `kernel : = kldiv_backward_log_target target_ptr new_grads_ptr
        target_stride new_grads_stride n_cols BLOCK_SIZE`
- `initialState : = s`
- `fun i : Fin BLOCK_SIZE => i.val < n_cols`
- `expected : = fun i => logTargetSpec s target_ptr target_stride i`

**Closed-form spec defs (transitive):** `kldiv_backward_log_target`, `logTargetSpec`, `inOffset`

<details><summary><code>kldiv_backward_log_target</code></summary>

```
/-- Documented one-block slice of `_kldiv_kernel_backward` for the
`log_target = True` constexpr branch. -/
```
```lean
def kldiv_backward_log_target
    (target_ptr new_grads_ptr : RegionName)
    (target_stride new_grads_stride n_cols BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0).to(tl.int64)
  target_ptr += pid * $(target_stride)
  new_grads_ptr += pid * $(new_grads_stride)
  offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_cols)
  target = tl.load(target_ptr + offsets, mask=mask, other=0.0)
  res = -tl.exp(target)
  tl.store(new_grads_ptr + offsets, res, mask=mask)
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
`reduction=0` one-block slice. -/
```
</details>

**Statement:**
```lean
theorem kldiv_forward_default_none_compute_correct
    (y_ptr gt_ptr loss_ptr : RegionName)
    (y_stride gt_stride loss_stride n_cols BLOCK_SIZE : Nat) (eps : ℝ)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => linearOffset s loss_stride i)) :
    ComputeCorrect.Realizes
      (kernel := kldiv_forward_default_none y_ptr gt_ptr loss_ptr
        y_stride gt_stride loss_stride n_cols BLOCK_SIZE eps)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < n_cols)
        (fun i => (loss_ptr, linearOffset s loss_stride i)))
      (expected := fun i =>
        forwardDefaultSpec s y_ptr gt_ptr y_stride gt_stride eps i)
```

**Assumptions / layout contracts:**
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => linearOffset s loss_stride i)`
- `kernel : = kldiv_forward_default_none y_ptr gt_ptr loss_ptr
        y_stride gt_stride loss_stride n_cols BLOCK_SIZE eps`
- `initialState : = s`
- `fun i : Fin BLOCK_SIZE => i.val < n_cols`
- `expected : = fun i =>
        forwardDefaultSpec s y_ptr gt_ptr y_stride gt_stride eps i`

**Closed-form spec defs (transitive):** `kldiv_forward_default_none`, `forwardDefaultSpec`, `inOffset`

<details><summary><code>kldiv_forward_default_none</code></summary>

```
/-- Documented one-block slice of `kldiv_ops.py`'s `_kldiv_kernel_forward`
for the `log_target = False`, `reduction = 0` (None) constexpr branch.

This models one `BLOCK_SIZE` iteration of Python's
`for i in range(0, n_cols, BLOCK_SIZE)` loop after the row pointers have been
advanced, taking the elementwise-store path of `reduction == 0`. -/
```
```lean
def kldiv_forward_default_none
    (y_ptr gt_ptr loss_ptr : RegionName)
    (y_stride gt_stride loss_stride n_cols BLOCK_SIZE : Nat) (eps : ℝ) :
    ComputeKernel := triton {
  pid = tl.program_id(0).to(tl.int64)
  y_ptr += pid * $(y_stride)
  gt_ptr += pid * $(gt_stride)
  loss_ptr += pid * $(loss_stride)
  offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_cols)
  y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
  y_true = tl.load(gt_ptr + offsets, mask=mask, other=0.0)
  loss = y_true * (tl.log(tl.maximum(y_true, $(eps))) - y)
  tl.store(loss_ptr + offsets, loss, mask=mask)
}
```
</details>

<details><summary><code>forwardDefaultSpec</code></summary>

```
/-- Forward KL-divergence per-element value (default, `log_target = False`)
for the `reduction = 0` (None) elementwise-store path. -/
```
```lean
noncomputable def forwardDefaultSpec
    (s : BlockState) (y_ptr gt_ptr : RegionName)
    (y_stride gt_stride : Nat) (eps : ℝ) (i : Fin BLOCK_SIZE) : ℝ :=
  let y := s.readMem y_ptr (inOffset s y_stride i)
  let y_true := s.readMem gt_ptr (inOffset s gt_stride i)
  y_true * (Real.log (max y_true eps) - y)
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
/-- Compute-facing correctness for the forward log-target one-block slice. -/
```
</details>

**Statement:**
```lean
theorem kldiv_forward_log_target_none_compute_correct
    (y_ptr gt_ptr loss_ptr : RegionName)
    (y_stride gt_stride loss_stride n_cols BLOCK_SIZE : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => linearOffset s loss_stride i)) :
    ComputeCorrect.Realizes
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
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => linearOffset s loss_stride i)`
- `kernel : = kldiv_forward_log_target_none y_ptr gt_ptr loss_ptr
        y_stride gt_stride loss_stride n_cols BLOCK_SIZE`
- `initialState : = s`
- `fun i : Fin BLOCK_SIZE => i.val < n_cols`
- `expected : = fun i =>
        forwardLogTargetSpec s y_ptr gt_ptr y_stride gt_stride i`

**Closed-form spec defs (transitive):** `kldiv_forward_log_target_none`, `forwardLogTargetSpec`, `inOffset`

<details><summary><code>kldiv_forward_log_target_none</code></summary>

```
/-- Documented one-block slice of `_kldiv_kernel_forward` for the
`log_target = True`, `reduction = 0` (None) constexpr branch. -/
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
  offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_cols)
  y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
  y_true = tl.load(gt_ptr + offsets, mask=mask, other=0.0)
  loss = tl.exp(y_true) * (y_true - y)
  tl.store(loss_ptr + offsets, loss, mask=mask)
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
