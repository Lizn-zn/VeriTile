# Spec sheet — `bench/tritonbench_g/cross_entropy1/Correct.lean`

**Python source:** `bench/tritonbench_g/cross_entropy1/cross_entropy1.py`

## Public theorem: `forward_original_correct`

<details><summary>docstring</summary>

```
/-- Positive B supports max reduction. Separate LSE/loss outputs preserve both
results; LSE must not overwrite the logits read by the later gather. Blocks
past the row end have zero readback in the framework's real sentinel model. -/
```
</details>

**Statement:**
```lean
specification forward_original_correct (storage : Storage)
    (loss_ptr lse_ptr logits_ptr : RegionName) (labels_ptr : Region .int)
    (smoothing lse_square_scale : ℝ) (ignored_index : Int)
    (total_classes : Nat) (class_start_idx : Int)
    (n_cols n_rows logits_row_stride BLOCK_SIZE : Nat)
    (HAS_SMOOTHING SPLIT : Bool)
    (hB : 0 < BLOCK_SIZE) (hne : lse_ptr ≠ loss_ptr) (hLL : lse_ptr ≠ logits_ptr) :
    Spec.Real (forwardOriginalIO storage loss_ptr lse_ptr logits_ptr labels_ptr smoothing lse_square_scale ignored_index total_classes class_start_idx n_cols n_rows logits_row_stride BLOCK_SIZE HAS_SMOOTHING SPLIT ⊨
      fun _ pid₁ lab xs g => if pid₁*BLOCK_SIZE < n_cols then
        (loss n_cols total_classes BLOCK_SIZE smoothing lse_square_scale ignored_index class_start_idx
          HAS_SMOOTHING SPLIT pid₁ lab xs g,blockLSE n_cols BLOCK_SIZE pid₁ xs)
      else (0,0))
```

**Assumptions / layout contracts:**
- `hB : 0 < BLOCK_SIZE`
- `hne : lse_ptr ≠ loss_ptr`
- `hLL : lse_ptr ≠ logits_ptr`

**Closed-form spec defs (transitive):** `forwardOriginalIO`, `loss`, `blockLSE`, `forward`, `blockSum`

<details><summary><code>forwardOriginalIO</code></summary>

```
/-- The label slot, active block logits, conditionally gathered target logit
and both scalar outputs are visible in this contract. Bounds apply only at
these actual accesses. The LSE store precedes the target-logit gather. -/
```
```lean
def forwardOriginalIO (storage : Storage)
    (loss_ptr lse_ptr logits_ptr : RegionName) (labels_ptr : Region .int)
    (smoothing lse_square_scale : ℝ) (ignored_index : Int)
    (total_classes : Nat) (class_start_idx : Int)
    (n_cols n_rows logits_row_stride BLOCK_SIZE : Nat)
    (HAS_SMOOTHING SPLIT : Bool) : MetaGatherMasked2DKernelIO₂ₓ₂ where
  kernel := (forward storage Bool.false loss_ptr lse_ptr logits_ptr labels_ptr
    smoothing lse_square_scale ignored_index total_classes class_start_idx
    n_cols n_rows logits_row_stride BLOCK_SIZE HAS_SMOOTHING SPLIT).eraseDType
  projection := ForwardProjection.projection storage Bool.false loss_ptr lse_ptr logits_ptr labels_ptr
    smoothing lse_square_scale ignored_index total_classes class_start_idx
    n_cols n_rows logits_row_stride BLOCK_SIZE HAS_SMOOTHING SPLIT
  mbufL := Region.cast labels_ptr
  inp := logits_ptr
  out1 := loss_ptr
  out2 := lse_ptr
  B := BLOCK_SIZE
  mwinL := fun pid₀ _ => pid₀
  read := fun pid₀ pid₁ _ j =>
    pid₀ * logits_row_stride + (pid₁ * BLOCK_SIZE + j.val)
  mask := fun _ pid₁ _ j => pid₁ * BLOCK_SIZE + j.val < n_cols
  gwin := fun pid₀ _ lab =>
    pid₀ * logits_row_stride + (lab - class_start_idx).toNat
  gmask := fun _ pid₁ lab =>
    lab ≠ ignored_index ∧
    lab - class_start_idx ≥ (↑(pid₁ * BLOCK_SIZE) : Int) ∧
    lab - class_start_idx < (↑(min n_cols ((pid₁ + 1) * BLOCK_SIZE)) : Int)
  write1 := fun pid₀ pid₁ _ => pid₁ * n_rows + pid₀
  write2 := fun pid₀ pid₁ _ => pid₁ * n_rows + pid₀
```
</details>

<details><summary><code>loss</code></summary>

```lean
noncomputable def loss (n_cols total_classes B : Nat)
    (smoothing lse_square_scale : ℝ) (ignored_index class_start_idx : Int)
    (HAS_SMOOTHING SPLIT : Bool)
    (pid₁ : Nat) (lab : Int) (xs : Fin B → ℝ) (g : ℝ) : ℝ :=
  if lab = ignored_index then 0 else
    let lblShift : Int := lab - class_start_idx
    let lse : ℝ := blockLSE n_cols B pid₁ xs
    let lseTerm : ℝ := if SPLIT then 0 else lse
    let sq : ℝ := if SPLIT then 0 else lse_square_scale * lse * lse
    let core : ℝ :=
      if (lblShift ≥ (pid₁ * B : Nat)) ∧
         (lblShift < (min n_cols ((pid₁ + 1) * B) : Nat)) then
        if HAS_SMOOTHING then
          lseTerm - smoothing * blockSum n_cols B pid₁ xs / total_classes
            - (1 - smoothing) * g
        else
          lseTerm - g
      else
        if HAS_SMOOTHING then
          smoothing * (lseTerm - blockSum n_cols B pid₁ xs / total_classes)
        else 0
    core + sq
```
</details>

<details><summary><code>blockLSE</code></summary>

```
/-- Independent mathematical functions of the initial block and target logit. -/
```
```lean
noncomputable def blockLSE (n_cols B pid₁ : Nat) (xs : Fin B → ℝ) : ℝ :=
  Real.log (∑ i ∈ Finset.univ.filter (fun i : Fin B => pid₁ * B + i.val < n_cols),
    Real.exp (xs i))
```
</details>

<details><summary><code>forward</code></summary>

```lean
def forward (storage : Storage) (swapped : Bool) := match storage with
  | .fp16 => if swapped then crossEntropyFwdSource%[tl.float16,indexCommuted] else crossEntropyFwdSource%[tl.float16,original]
  | .bf16 => if swapped then crossEntropyFwdSource%[tl.bfloat16,indexCommuted] else crossEntropyFwdSource%[tl.bfloat16,original]
  | .fp32 => if swapped then crossEntropyFwdSource%[tl.float32,indexCommuted] else crossEntropyFwdSource%[tl.float32,original]

/-! Source: cross_entropy1.py, cross_entropy_bwd_kernel. -/
syntax "crossEntropyBwdSource%" "[" tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(crossEntropyBwdSource%[$storage:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `indexCommuted do
      Macro.throwError "expected original or indexCommuted"
    let colIndex : TSyntax `tritonStmt ←
      if version.getId == `indexCommuted then
        `(tritonStmt| col_offsets = $(BLOCK_SIZE) * col_block_idx + tl.arange(0, $(BLOCK_SIZE)))
      else `(tritonStmt| col_offsets = col_block_idx * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE)))
    `(fun (dlogits_ptr dloss_ptr logits_ptr lse_ptr : RegionName) (labels_ptr : Region .int)
        (smoothing lse_square_scale : ℝ) (ignored_index : Int)
        (total_classes : Nat) (class_start_idx : Int)
        (n_cols logits_row_stride dlogits_row_stride dloss_row_stride BLOCK_SIZE : Nat)
        (HAS_SMOOTHING : Bool) => ComputeKernel.mk
      [dloss_ptr,logits_ptr,lse_ptr,Region.cast labels_ptr] [dlogits_ptr] (triton {
  tl.region logits_ptr = $storage, dlogits_ptr = $storage,
    dloss_ptr = tl.float32, lse_ptr = tl.float32
  row_idx = tl.program_id(0)
  col_block_idx = tl.program_id(1)
  logits_ptr = logits_ptr + row_idx * ($(logits_row_stride)).to(tl.int64)
  dlogits_ptr = dlogits_ptr + row_idx * ($(dlogits_row_stride)).to(tl.int64)
  $colIndex
  label_idx = tl.load(labels_ptr + row_idx)
  if label_idx != $((ignored_index : Int)) {
    dloss = tl.load(dloss_ptr + row_idx * $(dloss_row_stride))
  } else {
    dloss = 0.0
  }
  logits = tl.load(logits_ptr + col_offsets,
    mask=col_offsets < $(n_cols), other=-float("inf")).to(tl.float32)
  lse = tl.load(lse_ptr + row_idx)
  probs = tl.exp(logits - lse)
  probs += 2.0 * $((lse_square_scale : ℝ)) * lse * probs
  label_idx -= $((class_start_idx : Int))
  if HAS_SMOOTHING {
    smooth_negative = $((smoothing : ℝ)) / $(total_classes)
    probs = tl.where(col_offsets == label_idx, probs - (1 - $((smoothing : ℝ))), probs) - smooth_negative
  } else {
    probs = tl.where(col_offsets == label_idx, probs - 1.0, probs)
  }
  tl.store(dlogits_ptr + col_offsets, (dloss * probs).to($storage:tritonDType), mask=col_offsets < $(n_cols))
}).surfaceBody)
```
</details>

<details><summary><code>blockSum</code></summary>

```lean
noncomputable def blockSum (n_cols B pid₁ : Nat) (xs : Fin B → ℝ) : ℝ :=
  ∑ i : Fin B, if pid₁ * B + i.val < n_cols then xs i else 0
```
</details>

## Public theorem: `forward_index_commuted_correct`

<details><summary>docstring</summary>

```
/-- Positive B supports max reduction. Separate LSE/loss outputs preserve both
results; LSE must not overwrite the logits read by the later gather. Blocks
past the row end have zero readback in the framework's real sentinel model. -/
```
</details>

**Statement:**
```lean
specification forward_index_commuted_correct (storage : Storage)
    (loss_ptr lse_ptr logits_ptr : RegionName) (labels_ptr : Region .int)
    (smoothing lse_square_scale : ℝ) (ignored_index : Int)
    (total_classes : Nat) (class_start_idx : Int)
    (n_cols n_rows logits_row_stride BLOCK_SIZE : Nat)
    (HAS_SMOOTHING SPLIT : Bool)
    (hB : 0 < BLOCK_SIZE) (hne : lse_ptr ≠ loss_ptr) (hLL : lse_ptr ≠ logits_ptr) :
    Spec.Real (forwardIndexCommutedIO storage loss_ptr lse_ptr logits_ptr labels_ptr smoothing lse_square_scale ignored_index total_classes class_start_idx n_cols n_rows logits_row_stride BLOCK_SIZE HAS_SMOOTHING SPLIT ⊨
      fun _ pid₁ lab xs g => if pid₁*BLOCK_SIZE < n_cols then
        (loss n_cols total_classes BLOCK_SIZE smoothing lse_square_scale ignored_index class_start_idx
          HAS_SMOOTHING SPLIT pid₁ lab xs g,blockLSE n_cols BLOCK_SIZE pid₁ xs)
      else (0,0))
```

**Assumptions / layout contracts:**
- `hB : 0 < BLOCK_SIZE`
- `hne : lse_ptr ≠ loss_ptr`
- `hLL : lse_ptr ≠ logits_ptr`

**Closed-form spec defs (transitive):** `forwardIndexCommutedIO`, `loss`, `blockLSE`, `forwardOriginalIO`, `forward`, `blockSum`

<details><summary><code>forwardIndexCommutedIO</code></summary>

```lean
def forwardIndexCommutedIO (storage : Storage)
    (loss_ptr lse_ptr logits_ptr : RegionName) (labels_ptr : Region .int)
    (smoothing lse_square_scale : ℝ) (ignored_index : Int)
    (total_classes : Nat) (class_start_idx : Int)
    (n_cols n_rows logits_row_stride BLOCK_SIZE : Nat)
    (HAS_SMOOTHING SPLIT : Bool) : MetaGatherMasked2DKernelIO₂ₓ₂ :=
  { forwardOriginalIO storage loss_ptr lse_ptr logits_ptr labels_ptr smoothing lse_square_scale ignored_index total_classes class_start_idx n_cols n_rows logits_row_stride BLOCK_SIZE HAS_SMOOTHING SPLIT with
    kernel := (forward storage Bool.true loss_ptr lse_ptr logits_ptr labels_ptr smoothing lse_square_scale ignored_index total_classes class_start_idx n_cols n_rows logits_row_stride BLOCK_SIZE HAS_SMOOTHING SPLIT).eraseDType
    projection := ForwardProjection.projection storage Bool.true loss_ptr lse_ptr logits_ptr labels_ptr smoothing lse_square_scale ignored_index total_classes class_start_idx n_cols n_rows logits_row_stride BLOCK_SIZE HAS_SMOOTHING SPLIT }
```
</details>

<details><summary><code>loss</code></summary>

```lean
noncomputable def loss (n_cols total_classes B : Nat)
    (smoothing lse_square_scale : ℝ) (ignored_index class_start_idx : Int)
    (HAS_SMOOTHING SPLIT : Bool)
    (pid₁ : Nat) (lab : Int) (xs : Fin B → ℝ) (g : ℝ) : ℝ :=
  if lab = ignored_index then 0 else
    let lblShift : Int := lab - class_start_idx
    let lse : ℝ := blockLSE n_cols B pid₁ xs
    let lseTerm : ℝ := if SPLIT then 0 else lse
    let sq : ℝ := if SPLIT then 0 else lse_square_scale * lse * lse
    let core : ℝ :=
      if (lblShift ≥ (pid₁ * B : Nat)) ∧
         (lblShift < (min n_cols ((pid₁ + 1) * B) : Nat)) then
        if HAS_SMOOTHING then
          lseTerm - smoothing * blockSum n_cols B pid₁ xs / total_classes
            - (1 - smoothing) * g
        else
          lseTerm - g
      else
        if HAS_SMOOTHING then
          smoothing * (lseTerm - blockSum n_cols B pid₁ xs / total_classes)
        else 0
    core + sq
```
</details>

<details><summary><code>blockLSE</code></summary>

```
/-- Independent mathematical functions of the initial block and target logit. -/
```
```lean
noncomputable def blockLSE (n_cols B pid₁ : Nat) (xs : Fin B → ℝ) : ℝ :=
  Real.log (∑ i ∈ Finset.univ.filter (fun i : Fin B => pid₁ * B + i.val < n_cols),
    Real.exp (xs i))
```
</details>

<details><summary><code>forwardOriginalIO</code></summary>

```
/-- The label slot, active block logits, conditionally gathered target logit
and both scalar outputs are visible in this contract. Bounds apply only at
these actual accesses. The LSE store precedes the target-logit gather. -/
```
```lean
def forwardOriginalIO (storage : Storage)
    (loss_ptr lse_ptr logits_ptr : RegionName) (labels_ptr : Region .int)
    (smoothing lse_square_scale : ℝ) (ignored_index : Int)
    (total_classes : Nat) (class_start_idx : Int)
    (n_cols n_rows logits_row_stride BLOCK_SIZE : Nat)
    (HAS_SMOOTHING SPLIT : Bool) : MetaGatherMasked2DKernelIO₂ₓ₂ where
  kernel := (forward storage Bool.false loss_ptr lse_ptr logits_ptr labels_ptr
    smoothing lse_square_scale ignored_index total_classes class_start_idx
    n_cols n_rows logits_row_stride BLOCK_SIZE HAS_SMOOTHING SPLIT).eraseDType
  projection := ForwardProjection.projection storage Bool.false loss_ptr lse_ptr logits_ptr labels_ptr
    smoothing lse_square_scale ignored_index total_classes class_start_idx
    n_cols n_rows logits_row_stride BLOCK_SIZE HAS_SMOOTHING SPLIT
  mbufL := Region.cast labels_ptr
  inp := logits_ptr
  out1 := loss_ptr
  out2 := lse_ptr
  B := BLOCK_SIZE
  mwinL := fun pid₀ _ => pid₀
  read := fun pid₀ pid₁ _ j =>
    pid₀ * logits_row_stride + (pid₁ * BLOCK_SIZE + j.val)
  mask := fun _ pid₁ _ j => pid₁ * BLOCK_SIZE + j.val < n_cols
  gwin := fun pid₀ _ lab =>
    pid₀ * logits_row_stride + (lab - class_start_idx).toNat
  gmask := fun _ pid₁ lab =>
    lab ≠ ignored_index ∧
    lab - class_start_idx ≥ (↑(pid₁ * BLOCK_SIZE) : Int) ∧
    lab - class_start_idx < (↑(min n_cols ((pid₁ + 1) * BLOCK_SIZE)) : Int)
  write1 := fun pid₀ pid₁ _ => pid₁ * n_rows + pid₀
  write2 := fun pid₀ pid₁ _ => pid₁ * n_rows + pid₀
```
</details>

<details><summary><code>forward</code></summary>

```lean
def forward (storage : Storage) (swapped : Bool) := match storage with
  | .fp16 => if swapped then crossEntropyFwdSource%[tl.float16,indexCommuted] else crossEntropyFwdSource%[tl.float16,original]
  | .bf16 => if swapped then crossEntropyFwdSource%[tl.bfloat16,indexCommuted] else crossEntropyFwdSource%[tl.bfloat16,original]
  | .fp32 => if swapped then crossEntropyFwdSource%[tl.float32,indexCommuted] else crossEntropyFwdSource%[tl.float32,original]

/-! Source: cross_entropy1.py, cross_entropy_bwd_kernel. -/
syntax "crossEntropyBwdSource%" "[" tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(crossEntropyBwdSource%[$storage:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `indexCommuted do
      Macro.throwError "expected original or indexCommuted"
    let colIndex : TSyntax `tritonStmt ←
      if version.getId == `indexCommuted then
        `(tritonStmt| col_offsets = $(BLOCK_SIZE) * col_block_idx + tl.arange(0, $(BLOCK_SIZE)))
      else `(tritonStmt| col_offsets = col_block_idx * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE)))
    `(fun (dlogits_ptr dloss_ptr logits_ptr lse_ptr : RegionName) (labels_ptr : Region .int)
        (smoothing lse_square_scale : ℝ) (ignored_index : Int)
        (total_classes : Nat) (class_start_idx : Int)
        (n_cols logits_row_stride dlogits_row_stride dloss_row_stride BLOCK_SIZE : Nat)
        (HAS_SMOOTHING : Bool) => ComputeKernel.mk
      [dloss_ptr,logits_ptr,lse_ptr,Region.cast labels_ptr] [dlogits_ptr] (triton {
  tl.region logits_ptr = $storage, dlogits_ptr = $storage,
    dloss_ptr = tl.float32, lse_ptr = tl.float32
  row_idx = tl.program_id(0)
  col_block_idx = tl.program_id(1)
  logits_ptr = logits_ptr + row_idx * ($(logits_row_stride)).to(tl.int64)
  dlogits_ptr = dlogits_ptr + row_idx * ($(dlogits_row_stride)).to(tl.int64)
  $colIndex
  label_idx = tl.load(labels_ptr + row_idx)
  if label_idx != $((ignored_index : Int)) {
    dloss = tl.load(dloss_ptr + row_idx * $(dloss_row_stride))
  } else {
    dloss = 0.0
  }
  logits = tl.load(logits_ptr + col_offsets,
    mask=col_offsets < $(n_cols), other=-float("inf")).to(tl.float32)
  lse = tl.load(lse_ptr + row_idx)
  probs = tl.exp(logits - lse)
  probs += 2.0 * $((lse_square_scale : ℝ)) * lse * probs
  label_idx -= $((class_start_idx : Int))
  if HAS_SMOOTHING {
    smooth_negative = $((smoothing : ℝ)) / $(total_classes)
    probs = tl.where(col_offsets == label_idx, probs - (1 - $((smoothing : ℝ))), probs) - smooth_negative
  } else {
    probs = tl.where(col_offsets == label_idx, probs - 1.0, probs)
  }
  tl.store(dlogits_ptr + col_offsets, (dloss * probs).to($storage:tritonDType), mask=col_offsets < $(n_cols))
}).surfaceBody)
```
</details>

<details><summary><code>blockSum</code></summary>

```lean
noncomputable def blockSum (n_cols B pid₁ : Nat) (xs : Fin B → ℝ) : ℝ :=
  ∑ i : Fin B, if pid₁ * B + i.val < n_cols then xs i else 0
```
</details>

## Public theorem: `backward_original_correct`

**Statement:**
```lean
specification backward_original_correct (p : Layout) (r : Regions) (storage : Storage) :
    Spec.Real (backwardOriginalIO p r storage ⊨ gradient p)
```

**Closed-form spec defs (transitive):** `backwardOriginalIO`, `gradient`

<details><summary><code>backwardOriginalIO</code></summary>

```
/-- Every actual scalar/active-lane access is bounded. An ignored label skips
only dloss; labels, LSE and active logits are still read. Inputs may alias the
output because all reads precede the sole store. The column offsets are
injective, so no extra output-distinctness assumption is needed. -/
```
```lean
def backwardOriginalIO (p : Layout) (r : Regions) (storage : Storage) : BackwardIO.IO p where
  kernel := (source p r storage Bool.false).eraseDType
  projection := BackwardProgram.projection p r storage Bool.false
  read := fun u => {
    logits := u.readMem r.logits, lse := u.readMem r.lse,
    dloss := u.readMem r.dloss, label := u.readMemValue .int r.label }
  bounds := BackwardSafety.Bounds p r
  out := r.gradient
  write := fun l i => l.row*p.gradientStride+(l.block*p.B+i.val)
  mask := fun l i => l.block*p.B+i.val < p.nCols
```
</details>

<details><summary><code>gradient</code></summary>

```
/-- Independent formula using the backward kernel's legitimate initial inputs;
LSE is supplied by the caller and probabilities are computed by this kernel. -/
```
```lean
def gradient (p : Layout) (l : Launch) (a : Inputs) (i : Fin p.B) : ℝ :=
  let col := l.block*p.B+i.val
  let prob := Real.exp (a.logits (l.row*p.logitsStride+col)-a.lse l.row)
  let adj := prob+2*p.squareScale*a.lse l.row*prob
  let upstream := if a.label l.row=p.ignored then 0 else a.dloss (l.row*p.dlossStride)
  upstream *
    (if p.hasSmoothing then
      (adj - if (col : Int)=a.label l.row-p.classStart then 1-p.smoothing else 0) -
        p.smoothing / p.totalClasses
     else adj - if (col : Int)=a.label l.row-p.classStart then 1 else 0)
```
</details>

## Public theorem: `backward_index_commuted_correct`

**Statement:**
```lean
specification backward_index_commuted_correct (p : Layout) (r : Regions) (storage : Storage) :
    Spec.Real (backwardIndexCommutedIO p r storage ⊨ gradient p)
```

**Closed-form spec defs (transitive):** `backwardIndexCommutedIO`, `gradient`, `backwardOriginalIO`

<details><summary><code>backwardIndexCommutedIO</code></summary>

```lean
def backwardIndexCommutedIO (p : Layout) (r : Regions) (storage : Storage) : BackwardIO.IO p :=
  { backwardOriginalIO p r storage with
    kernel := (source p r storage Bool.true).eraseDType
    projection := BackwardProgram.projection p r storage Bool.true }
```
</details>

<details><summary><code>gradient</code></summary>

```
/-- Independent formula using the backward kernel's legitimate initial inputs;
LSE is supplied by the caller and probabilities are computed by this kernel. -/
```
```lean
def gradient (p : Layout) (l : Launch) (a : Inputs) (i : Fin p.B) : ℝ :=
  let col := l.block*p.B+i.val
  let prob := Real.exp (a.logits (l.row*p.logitsStride+col)-a.lse l.row)
  let adj := prob+2*p.squareScale*a.lse l.row*prob
  let upstream := if a.label l.row=p.ignored then 0 else a.dloss (l.row*p.dlossStride)
  upstream *
    (if p.hasSmoothing then
      (adj - if (col : Int)=a.label l.row-p.classStart then 1-p.smoothing else 0) -
        p.smoothing / p.totalClasses
     else adj - if (col : Int)=a.label l.row-p.classStart then 1 else 0)
```
</details>

<details><summary><code>backwardOriginalIO</code></summary>

```
/-- Every actual scalar/active-lane access is bounded. An ignored label skips
only dloss; labels, LSE and active logits are still read. Inputs may alias the
output because all reads precede the sole store. The column offsets are
injective, so no extra output-distinctness assumption is needed. -/
```
```lean
def backwardOriginalIO (p : Layout) (r : Regions) (storage : Storage) : BackwardIO.IO p where
  kernel := (source p r storage Bool.false).eraseDType
  projection := BackwardProgram.projection p r storage Bool.false
  read := fun u => {
    logits := u.readMem r.logits, lse := u.readMem r.lse,
    dloss := u.readMem r.dloss, label := u.readMemValue .int r.label }
  bounds := BackwardSafety.Bounds p r
  out := r.gradient
  write := fun l i => l.row*p.gradientStride+(l.block*p.B+i.val)
  mask := fun l i => l.block*p.B+i.val < p.nCols
```
</details>
