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

## Public theorem: `kldiv_forward_default_none_io_correctness`

<details><summary>docstring</summary>

```
/-- **The `⊨[R]` streaming headline (wave-5 S3 per-step emit genre).** For
every rounding model `R`, the faithful `kldiv_forward_default_none` surface
(the `log_target = False`, `reduction = 0` (None) forward branch) implements,
on its `StreamEmitMasked2DKernelIO₂` signature, the **ideal ℝ elementwise
forward KL divergence** over the streamed tiles: emitted window `(t, j)` holds
`y_true[t,j] · (log y_true[t,j] − y[t,j])`, with `y` the log-space prediction
stream and `y_true` the ground-truth stream. The spec `f` is exact real
arithmetic — this port has **no `eps` clamp** (unlike the `kldiv_ops` port).

Rounding story: the kernel has **zero rounding events**. Both `tl.load`s and
the in-loop `tl.store` are at `.real` (the lowered store is
`Stmt.store TileDType.real … (MaskOpt.mask …)` — no cast on the stored value),
and the kernel contains no `castFloat` at all; the only casts are the
`int`-flavoured `pid = tl.program_id(0).to(tl.int64)` pointer arithmetic, which
is exact. So the skin's boundary quantization degenerates: at the declared grid
`outDType := .real` the readback contract's `R.round .real` is the identity by
the model's defining `round_real`, and the `.real` in-loop stores are exact
under `execR R`. The ∀-`R` face therefore holds via the `RoundingModel` `.real`
identity fields, **not** as a `.triv` special case.

Layer map: the prologue and the loop body are cast-free
(`klPrefix_castFree` / `klLossBody_castFree`), so under `execR R` they collapse
verbatim onto the exact stepper; the loop is driven by the `forRange`
invariant `klInv` (`kl_preLoop` + `klInv_step` + `kl_runR`), the safety walk by
`Stmt.forRangeTraceSafeR_inv` over the same invariant (`kl_lossBodySafeR` +
`kldiv_traceSafeR`), and the stream indices are matched to the memory-level
closed form by `klStreamSpec_eq_klCell`.

**Scope: this face is dimension-general — the full multi-chunk stream.** `T`
is `⌈n_cols / BLOCK_SIZE⌉`, so the loop is proved for an arbitrary number of
iterations; there is **no** `n_cols ≤ BLOCK_SIZE` regime hypothesis and no
`0 < n_cols` hypothesis. (The retained exact surfaces below/above are the
narrower single-chunk statements; see "Relation to the exact surfaces".)

Every hypothesis is truth-forced:

* `hBS : 0 < BLOCK_SIZE` — the loop steps by `BLOCK_SIZE`
  (`range(0, n_cols, BLOCK_SIZE)`); at `BLOCK_SIZE = 0` with `n_cols > 0` the
  `forRange` fold does not advance, `execR` cannot terminate the way the
  invariant requires, and the step index `i / BLOCK_SIZE` is meaningless. It
  holds for every real launch (the host picks
  `BLOCK_SIZE = min(16384, next_power_of_2(S))`).
* `hly : loss_ptr ≠ y_ptr`, `hlg : loss_ptr ≠ gt_ptr` — the store sits
  **inside** the loop, so iteration `t + 1` loads *after* iteration `t` has
  already stored. If the loss buffer aliased either input buffer, later chunks
  would re-read overwritten values and the closed form would be false. (In the
  single-chunk regime these are vacuous; they are exactly the price of the
  general multi-chunk face.) The host allocates `output_tensor` fresh, so both
  hold for every real launch.

Relation to the exact surfaces: the file's exact headline
`kldiv_forward_default_none_compute_correct`
(`Realizes_without_Rounding`, scoped to the Python-tested single-chunk regime
`0 < n_cols ≤ BLOCK_SIZE`, with the offset-injectivity side condition) is
retained unchanged; this `⊨[R]` face restates the same KL content on the
streaming emit skin, for every `R` at once and for an arbitrary number of
chunks (at the `.real` grid the two faces carry the same exact cell on the
single-chunk overlap). Both faces are kept per the rounding-as-default
doctrine. The `log_target = True` branch and the `reduction ≠ 0` reduction
modes keep their existing (exact, partial) treatment — the reduction-mode
surface `kldiv_forward_surface` is still only proved to lower. -/
```
</details>

**Statement:**
```lean
specification kldiv_forward_default_none_io_correctness (R : RoundingModel)
    (y_ptr gt_ptr loss_ptr : RegionName)
    (y_stride gt_stride loss_stride n_cols BLOCK_SIZE : Nat)
    (hBS : 0 < BLOCK_SIZE)
    (hly : loss_ptr ≠ y_ptr) (hlg : loss_ptr ≠ gt_ptr) :
    kldivForwardDefaultNoneIO y_ptr gt_ptr loss_ptr y_stride gt_stride loss_stride
        n_cols BLOCK_SIZE ⊨[R]
      fun _ _ xs ys t j => klStreamSpec n_cols BLOCK_SIZE xs ys t j
```

**Assumptions / layout contracts:**
- `hBS : 0 < BLOCK_SIZE`
- `hly : loss_ptr ≠ y_ptr`
- `hlg : loss_ptr ≠ gt_ptr`

**Closed-form spec defs (transitive):** `kldivForwardDefaultNoneIO`, `klStreamSpec`, `kldiv_forward_default_none`, `klNumSteps`

<details><summary><code>kldivForwardDefaultNoneIO</code></summary>

```
/-- **Streaming IO signature** of `kldiv_forward_default_none` on the
two-stream per-step emit skin (S3: in-loop store). Step `t` (at `i =
t·BLOCK_SIZE`) reads the `BLOCK_SIZE`-lane `y` tile (`read1`) and the `gt`
tile (`read2`), and stores the `BLOCK_SIZE`-lane loss window (`write`) at the
**`.real`** grid (`outDType` default — the kernel's `tl.store(loss_ptr +
offsets, loss, mask=mask)` lowers to `Stmt.store TileDType.real` with no cast
on the value, so the per-step stores carry no quantization event). The windows
transcribe the kernel's pointer arithmetic verbatim (`offsets = i +
tl.arange(0, BLOCK_SIZE)` on top of `ptr += pid * stride`):

* `read1` step `t`, lane `j`: `pid₀·y_stride + (t·BLOCK_SIZE + j)`;
* `read2` step `t`, lane `j`: `pid₀·gt_stride + (t·BLOCK_SIZE + j)`;
* `write` step `t`, lane `j`: `pid₀·loss_stride + (t·BLOCK_SIZE + j)`.

All three masks are the kernel's single `mask`: `t·BLOCK_SIZE + j < n_cols`.
The second grid axis `pid₁` is unused (the launch grid is 1-D `(B,)`). -/
```
```lean
def kldivForwardDefaultNoneIO (y_ptr gt_ptr loss_ptr : RegionName)
    (y_stride gt_stride loss_stride n_cols BLOCK_SIZE : Nat) :
    StreamEmitMasked2DKernelIO₂ where
  kernel := kldiv_forward_default_none y_ptr gt_ptr loss_ptr y_stride gt_stride
    loss_stride n_cols BLOCK_SIZE
  inp1 := y_ptr
  inp2 := gt_ptr
  out := loss_ptr
  T := klNumSteps n_cols BLOCK_SIZE
  B1 := BLOCK_SIZE
  B2 := BLOCK_SIZE
  C := BLOCK_SIZE
  read1 := fun p₀ _ t j => p₀ * y_stride + (t.val * BLOCK_SIZE + j.val)
  read2 := fun p₀ _ t j => p₀ * gt_stride + (t.val * BLOCK_SIZE + j.val)
  write := fun p₀ _ t j => p₀ * loss_stride + (t.val * BLOCK_SIZE + j.val)
  mask1 := fun _ _ t j => t.val * BLOCK_SIZE + j.val < n_cols
  mask2 := fun _ _ t j => t.val * BLOCK_SIZE + j.val < n_cols
  writeMask := fun _ _ t j => t.val * BLOCK_SIZE + j.val < n_cols
```
</details>

<details><summary><code>klStreamSpec</code></summary>

```
/-- The stream-level forward KL spec (the emit genre's *elementwise* shape:
output window `(t, j)` depends only on the step-`t` input tiles):
`ys t j · (log (ys t j) − xs t j)` — exact real arithmetic, the
stream-index spelling of the file's `forwardDefaultSpec`. -/
```
```lean
noncomputable def klStreamSpec (n_cols BLOCK_SIZE : Nat)
    (xs ys : Fin (klNumSteps n_cols BLOCK_SIZE) → Fin BLOCK_SIZE → ℝ)
    (t : Fin (klNumSteps n_cols BLOCK_SIZE)) (j : Fin BLOCK_SIZE) : ℝ :=
  ys t j * (Real.log (ys t j) - xs t j)
```
</details>

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

<details><summary><code>klNumSteps</code></summary>

```
/-- Trip count of `for i in range(0, n_cols, BLOCK_SIZE)`:
`⌈n_cols / BLOCK_SIZE⌉`. -/
```
```lean
def klNumSteps (n_cols BLOCK_SIZE : Nat) : Nat := (n_cols + BLOCK_SIZE - 1) / BLOCK_SIZE
```
</details>
