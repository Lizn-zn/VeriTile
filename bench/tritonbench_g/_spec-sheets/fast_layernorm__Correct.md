# Spec sheet — `bench/tritonbench_g/fast_layernorm/Correct.lean`

**Python source:** `bench/tritonbench_g/fast_layernorm/fast_layernorm.py`

## Public theorem: `forward_correctness`

**Statement:**
```lean
specification forward_correctness (input weight bias : Storage) (Y X W b r mu : RegionName)
    (YR XR N B : Nat) (eps : ℝ) :
    Spec.Real (forward_correct_target
      (forward input weight bias Bool.false Y X W b r mu YR XR N B eps).eraseDType.toAlgKernel
      Y X W b r mu YR XR N B eps)
```

**Closed-form spec defs (transitive):** `forward_correct_target`, `forward`, `variance`, `output`, `invStd`, `mean`, `forwardOutside`, `padded`

<details><summary><code>forward_correct_target</code></summary>

```lean
def forward_correct_target (kernel : Kernel) (Y X W b r mu : RegionName) (YR XR N B : Nat) (eps : ℝ) : Prop :=
  ∀ A : FlatAlloc, A.Disjoint → (∀ q,q∉A.regions → A.extent q=0) →
  ∀ u : BlockState, u.undef=(fun _ _ => 0) → ForwardBounds Y X W b r mu YR XR N B u A.extent →
  0<N → N≤B → Y≠r → Y≠mu → r≠mu →
  0<variance N (fun j : Fin B => u.readMem X (u.pids 0*XR+j.val))+eps →
    ∃ v,exec (A.flattenKernel kernel) (A.flattenState u)=some v ∧
      (∀ j : Fin B,j.val<N → v.readMem A.flat (A.addr Y (u.pids 0*YR+j.val))=
        output N eps (fun j : Fin B => u.readMem X (u.pids 0*XR+j.val))
          (fun j : Fin B => u.readMem W j.val) (fun j : Fin B => u.readMem b j.val) j) ∧
      v.readMem A.flat (A.addr r (u.pids 0))=
        invStd N eps (fun j : Fin B => u.readMem X (u.pids 0*XR+j.val)) ∧
      v.readMem A.flat (A.addr mu (u.pids 0))=
        mean N (fun j : Fin B => u.readMem X (u.pids 0*XR+j.val)) ∧
      (∀ q o,forwardOutside A u Y r mu YR N B q o → v.mem q o=(A.flattenState u).mem q o)
```
</details>

<details><summary><code>forward</code></summary>

```lean
def forward (input weight bias : Storage) (swapped : Bool) := match input,weight,bias with
  | .fp16,.fp16,.fp16 => if swapped then fastLayernormForward%[tl.float16,tl.float16,tl.float16,indexCommuted] else fastLayernormForward%[tl.float16,tl.float16,tl.float16,original]
  | .fp16,.fp16,.bf16 => if swapped then fastLayernormForward%[tl.float16,tl.float16,tl.bfloat16,indexCommuted] else fastLayernormForward%[tl.float16,tl.float16,tl.bfloat16,original]
  | .fp16,.fp16,.fp32 => if swapped then fastLayernormForward%[tl.float16,tl.float16,tl.float32,indexCommuted] else fastLayernormForward%[tl.float16,tl.float16,tl.float32,original]
  | .fp16,.bf16,.fp16 => if swapped then fastLayernormForward%[tl.float16,tl.bfloat16,tl.float16,indexCommuted] else fastLayernormForward%[tl.float16,tl.bfloat16,tl.float16,original]
  | .fp16,.bf16,.bf16 => if swapped then fastLayernormForward%[tl.float16,tl.bfloat16,tl.bfloat16,indexCommuted] else fastLayernormForward%[tl.float16,tl.bfloat16,tl.bfloat16,original]
  | .fp16,.bf16,.fp32 => if swapped then fastLayernormForward%[tl.float16,tl.bfloat16,tl.float32,indexCommuted] else fastLayernormForward%[tl.float16,tl.bfloat16,tl.float32,original]
  | .fp16,.fp32,.fp16 => if swapped then fastLayernormForward%[tl.float16,tl.float32,tl.float16,indexCommuted] else fastLayernormForward%[tl.float16,tl.float32,tl.float16,original]
  | .fp16,.fp32,.bf16 => if swapped then fastLayernormForward%[tl.float16,tl.float32,tl.bfloat16,indexCommuted] else fastLayernormForward%[tl.float16,tl.float32,tl.bfloat16,original]
  | .fp16,.fp32,.fp32 => if swapped then fastLayernormForward%[tl.float16,tl.float32,tl.float32,indexCommuted] else fastLayernormForward%[tl.float16,tl.float32,tl.float32,original]
  | .bf16,.fp16,.fp16 => if swapped then fastLayernormForward%[tl.bfloat16,tl.float16,tl.float16,indexCommuted] else fastLayernormForward%[tl.bfloat16,tl.float16,tl.float16,original]
  | .bf16,.fp16,.bf16 => if swapped then fastLayernormForward%[tl.bfloat16,tl.float16,tl.bfloat16,indexCommuted] else fastLayernormForward%[tl.bfloat16,tl.float16,tl.bfloat16,original]
  | .bf16,.fp16,.fp32 => if swapped then fastLayernormForward%[tl.bfloat16,tl.float16,tl.float32,indexCommuted] else fastLayernormForward%[tl.bfloat16,tl.float16,tl.float32,original]
  | .bf16,.bf16,.fp16 => if swapped then fastLayernormForward%[tl.bfloat16,tl.bfloat16,tl.float16,indexCommuted] else fastLayernormForward%[tl.bfloat16,tl.bfloat16,tl.float16,original]
  | .bf16,.bf16,.bf16 => if swapped then fastLayernormForward%[tl.bfloat16,tl.bfloat16,tl.bfloat16,indexCommuted] else fastLayernormForward%[tl.bfloat16,tl.bfloat16,tl.bfloat16,original]
  | .bf16,.bf16,.fp32 => if swapped then fastLayernormForward%[tl.bfloat16,tl.bfloat16,tl.float32,indexCommuted] else fastLayernormForward%[tl.bfloat16,tl.bfloat16,tl.float32,original]
  | .bf16,.fp32,.fp16 => if swapped then fastLayernormForward%[tl.bfloat16,tl.float32,tl.float16,indexCommuted] else fastLayernormForward%[tl.bfloat16,tl.float32,tl.float16,original]
  | .bf16,.fp32,.bf16 => if swapped then fastLayernormForward%[tl.bfloat16,tl.float32,tl.bfloat16,indexCommuted] else fastLayernormForward%[tl.bfloat16,tl.float32,tl.bfloat16,original]
  | .bf16,.fp32,.fp32 => if swapped then fastLayernormForward%[tl.bfloat16,tl.float32,tl.float32,indexCommuted] else fastLayernormForward%[tl.bfloat16,tl.float32,tl.float32,original]
  | .fp32,.fp16,.fp16 => if swapped then fastLayernormForward%[tl.float32,tl.float16,tl.float16,indexCommuted] else fastLayernormForward%[tl.float32,tl.float16,tl.float16,original]
  | .fp32,.fp16,.bf16 => if swapped then fastLayernormForward%[tl.float32,tl.float16,tl.bfloat16,indexCommuted] else fastLayernormForward%[tl.float32,tl.float16,tl.bfloat16,original]
  | .fp32,.fp16,.fp32 => if swapped then fastLayernormForward%[tl.float32,tl.float16,tl.float32,indexCommuted] else fastLayernormForward%[tl.float32,tl.float16,tl.float32,original]
  | .fp32,.bf16,.fp16 => if swapped then fastLayernormForward%[tl.float32,tl.bfloat16,tl.float16,indexCommuted] else fastLayernormForward%[tl.float32,tl.bfloat16,tl.float16,original]
  | .fp32,.bf16,.bf16 => if swapped then fastLayernormForward%[tl.float32,tl.bfloat16,tl.bfloat16,indexCommuted] else fastLayernormForward%[tl.float32,tl.bfloat16,tl.bfloat16,original]
  | .fp32,.bf16,.fp32 => if swapped then fastLayernormForward%[tl.float32,tl.bfloat16,tl.float32,indexCommuted] else fastLayernormForward%[tl.float32,tl.bfloat16,tl.float32,original]
  | .fp32,.fp32,.fp16 => if swapped then fastLayernormForward%[tl.float32,tl.float32,tl.float16,indexCommuted] else fastLayernormForward%[tl.float32,tl.float32,tl.float16,original]
  | .fp32,.fp32,.bf16 => if swapped then fastLayernormForward%[tl.float32,tl.float32,tl.bfloat16,indexCommuted] else fastLayernormForward%[tl.float32,tl.float32,tl.bfloat16,original]
  | .fp32,.fp32,.fp32 => if swapped then fastLayernormForward%[tl.float32,tl.float32,tl.float32,indexCommuted] else fastLayernormForward%[tl.float32,tl.float32,tl.float32,original]

syntax "fastLayernormBackward%" "[" tritonDType "," tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(fastLayernormBackward%[$inputTy:tritonDType,$weightTy:tritonDType,$biasTy:tritonDType,$version:ident]) => do
    unless version.getId == `original || version.getId == `indexCommuted do
      Macro.throwError "expected original or indexCommuted"
    let rowIndex : TSyntax `tritonStmt ←
      if version.getId == `indexCommuted then `(tritonStmt| X += $(X_row_stride) * row_idx)
      else `(tritonStmt| X += row_idx * $(X_row_stride))
    `(fun (dY X W b r mu : RegionName) (dY_row_stride X_row_stride n_cols BLOCK_SIZE : Nat) (eps : ℝ) =>
      ComputeKernel.mk [dY,X,W,b,r,mu] [dY] (triton {
  tl.region dY = $inputTy, X = $inputTy, W = $weightTy, b = $biasTy, r = tl.float32, mu = tl.float32
  row_idx = tl.program_id(0)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(n_cols)
  dY += row_idx * $(dY_row_stride)
  $rowIndex
  r += row_idx
  mu += row_idx
  dY_row = tl.load(dY + col_offsets, mask=mask, other=0).to(tl.float32)
  X_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)
  W_row = tl.load(W + col_offsets, mask=mask, other=0).to(tl.float32)
  b_row = tl.load(b + col_offsets, mask=mask, other=0).to(tl.float32)
  inv_var = tl.load(r).to(tl.float32)
  mean = tl.load(mu).to(tl.float32)
  normed = (X_row - mean) * inv_var
  dY_W = dY_row * W_row
  dX_row = dY_W - tl.sum(dY_W, axis=0) / $(n_cols) - normed * tl.sum(dY_W * normed, axis=0) / $(n_cols)
  dX_row = dX_row * inv_var
  tl.store(dY + col_offsets, dX_row.to($inputTy:tritonDType), mask=mask)
}).surfaceBody)
```
</details>

<details><summary><code>variance</code></summary>

```
/-- Every lane is centered, including the initially zero-padded lanes. -/
```
```lean
def variance {B : Nat} (N : Nat) (xs : Fin B → ℝ) :=
  (∑ j, (padded N xs j-mean N xs)^2)/(N : ℝ)
```
</details>

<details><summary><code>output</code></summary>

```lean
def output {B : Nat} (N : Nat) (eps : ℝ) (xs ws bs : Fin B → ℝ) (j : Fin B) :=
  ((xs j-mean N xs)*invStd N eps xs)*ws j+bs j
```
</details>

<details><summary><code>invStd</code></summary>

```lean
def invStd {B : Nat} (N : Nat) (eps : ℝ) (xs : Fin B → ℝ) :=
  1/Real.sqrt (variance N xs+eps)
```
</details>

<details><summary><code>mean</code></summary>

```lean
def mean {B : Nat} (N : Nat) (xs : Fin B → ℝ) := (∑ j, padded N xs j)/(N : ℝ)
```
</details>

<details><summary><code>forwardOutside</code></summary>

```lean
def forwardOutside (A : FlatAlloc) (u : BlockState) (Y r mu : RegionName) (YR N B : Nat)
    (q : RegionName) (o : Nat) : Prop :=
  q≠A.flat ∨ ((∀ j : Fin B,j.val<N → o≠A.addr Y (u.pids 0*YR+j.val)) ∧
    o≠A.addr r (u.pids 0) ∧ o≠A.addr mu (u.pids 0))
```
</details>

<details><summary><code>padded</code></summary>

```lean
def padded {B : Nat} (N : Nat) (xs : Fin B → ℝ) (j : Fin B) :=
  if j.val<N then xs j else 0
```
</details>

## Public theorem: `forward_index_commuted_correctness`

**Statement:**
```lean
specification forward_index_commuted_correctness (input weight bias : Storage) (Y X W b r mu : RegionName)
    (YR XR N B : Nat) (eps : ℝ) :
    Spec.Real (forward_correct_target
      (forward input weight bias Bool.true Y X W b r mu YR XR N B eps).eraseDType.toAlgKernel
      Y X W b r mu YR XR N B eps)
```

**Closed-form spec defs (transitive):** `forward_correct_target`, `forward`, `variance`, `output`, `invStd`, `mean`, `forwardOutside`, `padded`

<details><summary><code>forward_correct_target</code></summary>

```lean
def forward_correct_target (kernel : Kernel) (Y X W b r mu : RegionName) (YR XR N B : Nat) (eps : ℝ) : Prop :=
  ∀ A : FlatAlloc, A.Disjoint → (∀ q,q∉A.regions → A.extent q=0) →
  ∀ u : BlockState, u.undef=(fun _ _ => 0) → ForwardBounds Y X W b r mu YR XR N B u A.extent →
  0<N → N≤B → Y≠r → Y≠mu → r≠mu →
  0<variance N (fun j : Fin B => u.readMem X (u.pids 0*XR+j.val))+eps →
    ∃ v,exec (A.flattenKernel kernel) (A.flattenState u)=some v ∧
      (∀ j : Fin B,j.val<N → v.readMem A.flat (A.addr Y (u.pids 0*YR+j.val))=
        output N eps (fun j : Fin B => u.readMem X (u.pids 0*XR+j.val))
          (fun j : Fin B => u.readMem W j.val) (fun j : Fin B => u.readMem b j.val) j) ∧
      v.readMem A.flat (A.addr r (u.pids 0))=
        invStd N eps (fun j : Fin B => u.readMem X (u.pids 0*XR+j.val)) ∧
      v.readMem A.flat (A.addr mu (u.pids 0))=
        mean N (fun j : Fin B => u.readMem X (u.pids 0*XR+j.val)) ∧
      (∀ q o,forwardOutside A u Y r mu YR N B q o → v.mem q o=(A.flattenState u).mem q o)
```
</details>

<details><summary><code>forward</code></summary>

```lean
def forward (input weight bias : Storage) (swapped : Bool) := match input,weight,bias with
  | .fp16,.fp16,.fp16 => if swapped then fastLayernormForward%[tl.float16,tl.float16,tl.float16,indexCommuted] else fastLayernormForward%[tl.float16,tl.float16,tl.float16,original]
  | .fp16,.fp16,.bf16 => if swapped then fastLayernormForward%[tl.float16,tl.float16,tl.bfloat16,indexCommuted] else fastLayernormForward%[tl.float16,tl.float16,tl.bfloat16,original]
  | .fp16,.fp16,.fp32 => if swapped then fastLayernormForward%[tl.float16,tl.float16,tl.float32,indexCommuted] else fastLayernormForward%[tl.float16,tl.float16,tl.float32,original]
  | .fp16,.bf16,.fp16 => if swapped then fastLayernormForward%[tl.float16,tl.bfloat16,tl.float16,indexCommuted] else fastLayernormForward%[tl.float16,tl.bfloat16,tl.float16,original]
  | .fp16,.bf16,.bf16 => if swapped then fastLayernormForward%[tl.float16,tl.bfloat16,tl.bfloat16,indexCommuted] else fastLayernormForward%[tl.float16,tl.bfloat16,tl.bfloat16,original]
  | .fp16,.bf16,.fp32 => if swapped then fastLayernormForward%[tl.float16,tl.bfloat16,tl.float32,indexCommuted] else fastLayernormForward%[tl.float16,tl.bfloat16,tl.float32,original]
  | .fp16,.fp32,.fp16 => if swapped then fastLayernormForward%[tl.float16,tl.float32,tl.float16,indexCommuted] else fastLayernormForward%[tl.float16,tl.float32,tl.float16,original]
  | .fp16,.fp32,.bf16 => if swapped then fastLayernormForward%[tl.float16,tl.float32,tl.bfloat16,indexCommuted] else fastLayernormForward%[tl.float16,tl.float32,tl.bfloat16,original]
  | .fp16,.fp32,.fp32 => if swapped then fastLayernormForward%[tl.float16,tl.float32,tl.float32,indexCommuted] else fastLayernormForward%[tl.float16,tl.float32,tl.float32,original]
  | .bf16,.fp16,.fp16 => if swapped then fastLayernormForward%[tl.bfloat16,tl.float16,tl.float16,indexCommuted] else fastLayernormForward%[tl.bfloat16,tl.float16,tl.float16,original]
  | .bf16,.fp16,.bf16 => if swapped then fastLayernormForward%[tl.bfloat16,tl.float16,tl.bfloat16,indexCommuted] else fastLayernormForward%[tl.bfloat16,tl.float16,tl.bfloat16,original]
  | .bf16,.fp16,.fp32 => if swapped then fastLayernormForward%[tl.bfloat16,tl.float16,tl.float32,indexCommuted] else fastLayernormForward%[tl.bfloat16,tl.float16,tl.float32,original]
  | .bf16,.bf16,.fp16 => if swapped then fastLayernormForward%[tl.bfloat16,tl.bfloat16,tl.float16,indexCommuted] else fastLayernormForward%[tl.bfloat16,tl.bfloat16,tl.float16,original]
  | .bf16,.bf16,.bf16 => if swapped then fastLayernormForward%[tl.bfloat16,tl.bfloat16,tl.bfloat16,indexCommuted] else fastLayernormForward%[tl.bfloat16,tl.bfloat16,tl.bfloat16,original]
  | .bf16,.bf16,.fp32 => if swapped then fastLayernormForward%[tl.bfloat16,tl.bfloat16,tl.float32,indexCommuted] else fastLayernormForward%[tl.bfloat16,tl.bfloat16,tl.float32,original]
  | .bf16,.fp32,.fp16 => if swapped then fastLayernormForward%[tl.bfloat16,tl.float32,tl.float16,indexCommuted] else fastLayernormForward%[tl.bfloat16,tl.float32,tl.float16,original]
  | .bf16,.fp32,.bf16 => if swapped then fastLayernormForward%[tl.bfloat16,tl.float32,tl.bfloat16,indexCommuted] else fastLayernormForward%[tl.bfloat16,tl.float32,tl.bfloat16,original]
  | .bf16,.fp32,.fp32 => if swapped then fastLayernormForward%[tl.bfloat16,tl.float32,tl.float32,indexCommuted] else fastLayernormForward%[tl.bfloat16,tl.float32,tl.float32,original]
  | .fp32,.fp16,.fp16 => if swapped then fastLayernormForward%[tl.float32,tl.float16,tl.float16,indexCommuted] else fastLayernormForward%[tl.float32,tl.float16,tl.float16,original]
  | .fp32,.fp16,.bf16 => if swapped then fastLayernormForward%[tl.float32,tl.float16,tl.bfloat16,indexCommuted] else fastLayernormForward%[tl.float32,tl.float16,tl.bfloat16,original]
  | .fp32,.fp16,.fp32 => if swapped then fastLayernormForward%[tl.float32,tl.float16,tl.float32,indexCommuted] else fastLayernormForward%[tl.float32,tl.float16,tl.float32,original]
  | .fp32,.bf16,.fp16 => if swapped then fastLayernormForward%[tl.float32,tl.bfloat16,tl.float16,indexCommuted] else fastLayernormForward%[tl.float32,tl.bfloat16,tl.float16,original]
  | .fp32,.bf16,.bf16 => if swapped then fastLayernormForward%[tl.float32,tl.bfloat16,tl.bfloat16,indexCommuted] else fastLayernormForward%[tl.float32,tl.bfloat16,tl.bfloat16,original]
  | .fp32,.bf16,.fp32 => if swapped then fastLayernormForward%[tl.float32,tl.bfloat16,tl.float32,indexCommuted] else fastLayernormForward%[tl.float32,tl.bfloat16,tl.float32,original]
  | .fp32,.fp32,.fp16 => if swapped then fastLayernormForward%[tl.float32,tl.float32,tl.float16,indexCommuted] else fastLayernormForward%[tl.float32,tl.float32,tl.float16,original]
  | .fp32,.fp32,.bf16 => if swapped then fastLayernormForward%[tl.float32,tl.float32,tl.bfloat16,indexCommuted] else fastLayernormForward%[tl.float32,tl.float32,tl.bfloat16,original]
  | .fp32,.fp32,.fp32 => if swapped then fastLayernormForward%[tl.float32,tl.float32,tl.float32,indexCommuted] else fastLayernormForward%[tl.float32,tl.float32,tl.float32,original]

syntax "fastLayernormBackward%" "[" tritonDType "," tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(fastLayernormBackward%[$inputTy:tritonDType,$weightTy:tritonDType,$biasTy:tritonDType,$version:ident]) => do
    unless version.getId == `original || version.getId == `indexCommuted do
      Macro.throwError "expected original or indexCommuted"
    let rowIndex : TSyntax `tritonStmt ←
      if version.getId == `indexCommuted then `(tritonStmt| X += $(X_row_stride) * row_idx)
      else `(tritonStmt| X += row_idx * $(X_row_stride))
    `(fun (dY X W b r mu : RegionName) (dY_row_stride X_row_stride n_cols BLOCK_SIZE : Nat) (eps : ℝ) =>
      ComputeKernel.mk [dY,X,W,b,r,mu] [dY] (triton {
  tl.region dY = $inputTy, X = $inputTy, W = $weightTy, b = $biasTy, r = tl.float32, mu = tl.float32
  row_idx = tl.program_id(0)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(n_cols)
  dY += row_idx * $(dY_row_stride)
  $rowIndex
  r += row_idx
  mu += row_idx
  dY_row = tl.load(dY + col_offsets, mask=mask, other=0).to(tl.float32)
  X_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)
  W_row = tl.load(W + col_offsets, mask=mask, other=0).to(tl.float32)
  b_row = tl.load(b + col_offsets, mask=mask, other=0).to(tl.float32)
  inv_var = tl.load(r).to(tl.float32)
  mean = tl.load(mu).to(tl.float32)
  normed = (X_row - mean) * inv_var
  dY_W = dY_row * W_row
  dX_row = dY_W - tl.sum(dY_W, axis=0) / $(n_cols) - normed * tl.sum(dY_W * normed, axis=0) / $(n_cols)
  dX_row = dX_row * inv_var
  tl.store(dY + col_offsets, dX_row.to($inputTy:tritonDType), mask=mask)
}).surfaceBody)
```
</details>

<details><summary><code>variance</code></summary>

```
/-- Every lane is centered, including the initially zero-padded lanes. -/
```
```lean
def variance {B : Nat} (N : Nat) (xs : Fin B → ℝ) :=
  (∑ j, (padded N xs j-mean N xs)^2)/(N : ℝ)
```
</details>

<details><summary><code>output</code></summary>

```lean
def output {B : Nat} (N : Nat) (eps : ℝ) (xs ws bs : Fin B → ℝ) (j : Fin B) :=
  ((xs j-mean N xs)*invStd N eps xs)*ws j+bs j
```
</details>

<details><summary><code>invStd</code></summary>

```lean
def invStd {B : Nat} (N : Nat) (eps : ℝ) (xs : Fin B → ℝ) :=
  1/Real.sqrt (variance N xs+eps)
```
</details>

<details><summary><code>mean</code></summary>

```lean
def mean {B : Nat} (N : Nat) (xs : Fin B → ℝ) := (∑ j, padded N xs j)/(N : ℝ)
```
</details>

<details><summary><code>forwardOutside</code></summary>

```lean
def forwardOutside (A : FlatAlloc) (u : BlockState) (Y r mu : RegionName) (YR N B : Nat)
    (q : RegionName) (o : Nat) : Prop :=
  q≠A.flat ∨ ((∀ j : Fin B,j.val<N → o≠A.addr Y (u.pids 0*YR+j.val)) ∧
    o≠A.addr r (u.pids 0) ∧ o≠A.addr mu (u.pids 0))
```
</details>

<details><summary><code>padded</code></summary>

```lean
def padded {B : Nat} (N : Nat) (xs : Fin B → ℝ) (j : Fin B) :=
  if j.val<N then xs j else 0
```
</details>

## Public theorem: `backward_correctness`

**Statement:**
```lean
specification backward_correctness (input weight bias : Storage) (dY X W b r mu : RegionName)
    (YR XR N B : Nat) (eps : ℝ) :
    Spec.Real (backward_correct_target
      (backward input weight bias Bool.false dY X W b r mu YR XR N B eps).eraseDType.toAlgKernel
      dY X W b r mu YR XR N B)
```

**Closed-form spec defs (transitive):** `backward_correct_target`, `backward`, `gradient`, `backwardOutside`, `padded`, `output`, `mean`, `invStd`, `variance`

<details><summary><code>backward_correct_target</code></summary>

```lean
def backward_correct_target (kernel : Kernel) (dY X W b r mu : RegionName) (YR XR N B : Nat) : Prop :=
  ∀ A : FlatAlloc, A.Disjoint → (∀ q,q∉A.regions → A.extent q=0) →
  ∀ u : BlockState, u.undef=(fun _ _ => 0) → BackwardBounds dY X W b r mu YR XR N B u A.extent →
  0<N → N≤B →
    ∃ v,exec (A.flattenKernel kernel) (A.flattenState u)=some v ∧
      (∀ j : Fin B,j.val<N → v.readMem A.flat (A.addr dY (u.pids 0*YR+j.val))=
        gradient N (u.readMem r (u.pids 0)) (u.readMem mu (u.pids 0))
          (fun j : Fin B => u.readMem X (u.pids 0*XR+j.val))
          (fun j : Fin B => u.readMem dY (u.pids 0*YR+j.val))
          (fun j : Fin B => u.readMem W j.val) j) ∧
      (∀ q o,backwardOutside A u dY YR N B q o → v.mem q o=(A.flattenState u).mem q o)
```
</details>

<details><summary><code>backward</code></summary>

```lean
def backward (input weight bias : Storage) (swapped : Bool) := match input,weight,bias with
  | .fp16,.fp16,.fp16 => if swapped then fastLayernormBackward%[tl.float16,tl.float16,tl.float16,indexCommuted] else fastLayernormBackward%[tl.float16,tl.float16,tl.float16,original]
  | .fp16,.fp16,.bf16 => if swapped then fastLayernormBackward%[tl.float16,tl.float16,tl.bfloat16,indexCommuted] else fastLayernormBackward%[tl.float16,tl.float16,tl.bfloat16,original]
  | .fp16,.fp16,.fp32 => if swapped then fastLayernormBackward%[tl.float16,tl.float16,tl.float32,indexCommuted] else fastLayernormBackward%[tl.float16,tl.float16,tl.float32,original]
  | .fp16,.bf16,.fp16 => if swapped then fastLayernormBackward%[tl.float16,tl.bfloat16,tl.float16,indexCommuted] else fastLayernormBackward%[tl.float16,tl.bfloat16,tl.float16,original]
  | .fp16,.bf16,.bf16 => if swapped then fastLayernormBackward%[tl.float16,tl.bfloat16,tl.bfloat16,indexCommuted] else fastLayernormBackward%[tl.float16,tl.bfloat16,tl.bfloat16,original]
  | .fp16,.bf16,.fp32 => if swapped then fastLayernormBackward%[tl.float16,tl.bfloat16,tl.float32,indexCommuted] else fastLayernormBackward%[tl.float16,tl.bfloat16,tl.float32,original]
  | .fp16,.fp32,.fp16 => if swapped then fastLayernormBackward%[tl.float16,tl.float32,tl.float16,indexCommuted] else fastLayernormBackward%[tl.float16,tl.float32,tl.float16,original]
  | .fp16,.fp32,.bf16 => if swapped then fastLayernormBackward%[tl.float16,tl.float32,tl.bfloat16,indexCommuted] else fastLayernormBackward%[tl.float16,tl.float32,tl.bfloat16,original]
  | .fp16,.fp32,.fp32 => if swapped then fastLayernormBackward%[tl.float16,tl.float32,tl.float32,indexCommuted] else fastLayernormBackward%[tl.float16,tl.float32,tl.float32,original]
  | .bf16,.fp16,.fp16 => if swapped then fastLayernormBackward%[tl.bfloat16,tl.float16,tl.float16,indexCommuted] else fastLayernormBackward%[tl.bfloat16,tl.float16,tl.float16,original]
  | .bf16,.fp16,.bf16 => if swapped then fastLayernormBackward%[tl.bfloat16,tl.float16,tl.bfloat16,indexCommuted] else fastLayernormBackward%[tl.bfloat16,tl.float16,tl.bfloat16,original]
  | .bf16,.fp16,.fp32 => if swapped then fastLayernormBackward%[tl.bfloat16,tl.float16,tl.float32,indexCommuted] else fastLayernormBackward%[tl.bfloat16,tl.float16,tl.float32,original]
  | .bf16,.bf16,.fp16 => if swapped then fastLayernormBackward%[tl.bfloat16,tl.bfloat16,tl.float16,indexCommuted] else fastLayernormBackward%[tl.bfloat16,tl.bfloat16,tl.float16,original]
  | .bf16,.bf16,.bf16 => if swapped then fastLayernormBackward%[tl.bfloat16,tl.bfloat16,tl.bfloat16,indexCommuted] else fastLayernormBackward%[tl.bfloat16,tl.bfloat16,tl.bfloat16,original]
  | .bf16,.bf16,.fp32 => if swapped then fastLayernormBackward%[tl.bfloat16,tl.bfloat16,tl.float32,indexCommuted] else fastLayernormBackward%[tl.bfloat16,tl.bfloat16,tl.float32,original]
  | .bf16,.fp32,.fp16 => if swapped then fastLayernormBackward%[tl.bfloat16,tl.float32,tl.float16,indexCommuted] else fastLayernormBackward%[tl.bfloat16,tl.float32,tl.float16,original]
  | .bf16,.fp32,.bf16 => if swapped then fastLayernormBackward%[tl.bfloat16,tl.float32,tl.bfloat16,indexCommuted] else fastLayernormBackward%[tl.bfloat16,tl.float32,tl.bfloat16,original]
  | .bf16,.fp32,.fp32 => if swapped then fastLayernormBackward%[tl.bfloat16,tl.float32,tl.float32,indexCommuted] else fastLayernormBackward%[tl.bfloat16,tl.float32,tl.float32,original]
  | .fp32,.fp16,.fp16 => if swapped then fastLayernormBackward%[tl.float32,tl.float16,tl.float16,indexCommuted] else fastLayernormBackward%[tl.float32,tl.float16,tl.float16,original]
  | .fp32,.fp16,.bf16 => if swapped then fastLayernormBackward%[tl.float32,tl.float16,tl.bfloat16,indexCommuted] else fastLayernormBackward%[tl.float32,tl.float16,tl.bfloat16,original]
  | .fp32,.fp16,.fp32 => if swapped then fastLayernormBackward%[tl.float32,tl.float16,tl.float32,indexCommuted] else fastLayernormBackward%[tl.float32,tl.float16,tl.float32,original]
  | .fp32,.bf16,.fp16 => if swapped then fastLayernormBackward%[tl.float32,tl.bfloat16,tl.float16,indexCommuted] else fastLayernormBackward%[tl.float32,tl.bfloat16,tl.float16,original]
  | .fp32,.bf16,.bf16 => if swapped then fastLayernormBackward%[tl.float32,tl.bfloat16,tl.bfloat16,indexCommuted] else fastLayernormBackward%[tl.float32,tl.bfloat16,tl.bfloat16,original]
  | .fp32,.bf16,.fp32 => if swapped then fastLayernormBackward%[tl.float32,tl.bfloat16,tl.float32,indexCommuted] else fastLayernormBackward%[tl.float32,tl.bfloat16,tl.float32,original]
  | .fp32,.fp32,.fp16 => if swapped then fastLayernormBackward%[tl.float32,tl.float32,tl.float16,indexCommuted] else fastLayernormBackward%[tl.float32,tl.float32,tl.float16,original]
  | .fp32,.fp32,.bf16 => if swapped then fastLayernormBackward%[tl.float32,tl.float32,tl.bfloat16,indexCommuted] else fastLayernormBackward%[tl.float32,tl.float32,tl.bfloat16,original]
  | .fp32,.fp32,.fp32 => if swapped then fastLayernormBackward%[tl.float32,tl.float32,tl.float32,indexCommuted] else fastLayernormBackward%[tl.float32,tl.float32,tl.float32,original]
```
</details>

<details><summary><code>gradient</code></summary>

```
/-- The backward body reads these statistics from the caller. Bias is loaded
but is not an operand of the returned gradient. -/
```
```lean
def gradient {B : Nat} (N : Nat) (inv avg : ℝ) (xs dys ws : Fin B → ℝ) (j : Fin B) :=
  let q := fun k => padded N dys k*padded N ws k
  let normed := fun k => (padded N xs k-avg)*inv
  (q j-(∑ k, q k)/(N : ℝ)-normed j*(∑ k, q k*normed k)/(N : ℝ))*inv


structure ForwardBounds (Y X W b r mu : RegionName) (YR XR N B : Nat)
    (u : BlockState) (bounds : RegionBounds) : Prop where
  input : ∀ j : Fin B, j.val<N → u.pids 0*XR+j.val<bounds X
  weight : ∀ j : Fin B, j.val<N → j.val<bounds W
  bias : ∀ j : Fin B, j.val<N → j.val<bounds b
  output : ∀ j : Fin B, j.val<N → u.pids 0*YR+j.val<bounds Y
  inv : u.pids 0<bounds r
  mean : u.pids 0<bounds mu
```
</details>

<details><summary><code>backwardOutside</code></summary>

```lean
def backwardOutside (A : FlatAlloc) (u : BlockState) (dY : RegionName) (YR N B : Nat)
    (q : RegionName) (o : Nat) : Prop :=
  q≠A.flat ∨ ∀ j : Fin B,j.val<N → o≠A.addr dY (u.pids 0*YR+j.val)
```
</details>

<details><summary><code>padded</code></summary>

```lean
def padded {B : Nat} (N : Nat) (xs : Fin B → ℝ) (j : Fin B) :=
  if j.val<N then xs j else 0
```
</details>

<details><summary><code>output</code></summary>

```lean
def output {B : Nat} (N : Nat) (eps : ℝ) (xs ws bs : Fin B → ℝ) (j : Fin B) :=
  ((xs j-mean N xs)*invStd N eps xs)*ws j+bs j
```
</details>

<details><summary><code>mean</code></summary>

```lean
def mean {B : Nat} (N : Nat) (xs : Fin B → ℝ) := (∑ j, padded N xs j)/(N : ℝ)
```
</details>

<details><summary><code>invStd</code></summary>

```lean
def invStd {B : Nat} (N : Nat) (eps : ℝ) (xs : Fin B → ℝ) :=
  1/Real.sqrt (variance N xs+eps)
```
</details>

<details><summary><code>variance</code></summary>

```
/-- Every lane is centered, including the initially zero-padded lanes. -/
```
```lean
def variance {B : Nat} (N : Nat) (xs : Fin B → ℝ) :=
  (∑ j, (padded N xs j-mean N xs)^2)/(N : ℝ)
```
</details>

## Public theorem: `backward_index_commuted_correctness`

**Statement:**
```lean
specification backward_index_commuted_correctness (input weight bias : Storage) (dY X W b r mu : RegionName)
    (YR XR N B : Nat) (eps : ℝ) :
    Spec.Real (backward_correct_target
      (backward input weight bias Bool.true dY X W b r mu YR XR N B eps).eraseDType.toAlgKernel
      dY X W b r mu YR XR N B)
```

**Closed-form spec defs (transitive):** `backward_correct_target`, `backward`, `gradient`, `backwardOutside`, `padded`, `output`, `mean`, `invStd`, `variance`

<details><summary><code>backward_correct_target</code></summary>

```lean
def backward_correct_target (kernel : Kernel) (dY X W b r mu : RegionName) (YR XR N B : Nat) : Prop :=
  ∀ A : FlatAlloc, A.Disjoint → (∀ q,q∉A.regions → A.extent q=0) →
  ∀ u : BlockState, u.undef=(fun _ _ => 0) → BackwardBounds dY X W b r mu YR XR N B u A.extent →
  0<N → N≤B →
    ∃ v,exec (A.flattenKernel kernel) (A.flattenState u)=some v ∧
      (∀ j : Fin B,j.val<N → v.readMem A.flat (A.addr dY (u.pids 0*YR+j.val))=
        gradient N (u.readMem r (u.pids 0)) (u.readMem mu (u.pids 0))
          (fun j : Fin B => u.readMem X (u.pids 0*XR+j.val))
          (fun j : Fin B => u.readMem dY (u.pids 0*YR+j.val))
          (fun j : Fin B => u.readMem W j.val) j) ∧
      (∀ q o,backwardOutside A u dY YR N B q o → v.mem q o=(A.flattenState u).mem q o)
```
</details>

<details><summary><code>backward</code></summary>

```lean
def backward (input weight bias : Storage) (swapped : Bool) := match input,weight,bias with
  | .fp16,.fp16,.fp16 => if swapped then fastLayernormBackward%[tl.float16,tl.float16,tl.float16,indexCommuted] else fastLayernormBackward%[tl.float16,tl.float16,tl.float16,original]
  | .fp16,.fp16,.bf16 => if swapped then fastLayernormBackward%[tl.float16,tl.float16,tl.bfloat16,indexCommuted] else fastLayernormBackward%[tl.float16,tl.float16,tl.bfloat16,original]
  | .fp16,.fp16,.fp32 => if swapped then fastLayernormBackward%[tl.float16,tl.float16,tl.float32,indexCommuted] else fastLayernormBackward%[tl.float16,tl.float16,tl.float32,original]
  | .fp16,.bf16,.fp16 => if swapped then fastLayernormBackward%[tl.float16,tl.bfloat16,tl.float16,indexCommuted] else fastLayernormBackward%[tl.float16,tl.bfloat16,tl.float16,original]
  | .fp16,.bf16,.bf16 => if swapped then fastLayernormBackward%[tl.float16,tl.bfloat16,tl.bfloat16,indexCommuted] else fastLayernormBackward%[tl.float16,tl.bfloat16,tl.bfloat16,original]
  | .fp16,.bf16,.fp32 => if swapped then fastLayernormBackward%[tl.float16,tl.bfloat16,tl.float32,indexCommuted] else fastLayernormBackward%[tl.float16,tl.bfloat16,tl.float32,original]
  | .fp16,.fp32,.fp16 => if swapped then fastLayernormBackward%[tl.float16,tl.float32,tl.float16,indexCommuted] else fastLayernormBackward%[tl.float16,tl.float32,tl.float16,original]
  | .fp16,.fp32,.bf16 => if swapped then fastLayernormBackward%[tl.float16,tl.float32,tl.bfloat16,indexCommuted] else fastLayernormBackward%[tl.float16,tl.float32,tl.bfloat16,original]
  | .fp16,.fp32,.fp32 => if swapped then fastLayernormBackward%[tl.float16,tl.float32,tl.float32,indexCommuted] else fastLayernormBackward%[tl.float16,tl.float32,tl.float32,original]
  | .bf16,.fp16,.fp16 => if swapped then fastLayernormBackward%[tl.bfloat16,tl.float16,tl.float16,indexCommuted] else fastLayernormBackward%[tl.bfloat16,tl.float16,tl.float16,original]
  | .bf16,.fp16,.bf16 => if swapped then fastLayernormBackward%[tl.bfloat16,tl.float16,tl.bfloat16,indexCommuted] else fastLayernormBackward%[tl.bfloat16,tl.float16,tl.bfloat16,original]
  | .bf16,.fp16,.fp32 => if swapped then fastLayernormBackward%[tl.bfloat16,tl.float16,tl.float32,indexCommuted] else fastLayernormBackward%[tl.bfloat16,tl.float16,tl.float32,original]
  | .bf16,.bf16,.fp16 => if swapped then fastLayernormBackward%[tl.bfloat16,tl.bfloat16,tl.float16,indexCommuted] else fastLayernormBackward%[tl.bfloat16,tl.bfloat16,tl.float16,original]
  | .bf16,.bf16,.bf16 => if swapped then fastLayernormBackward%[tl.bfloat16,tl.bfloat16,tl.bfloat16,indexCommuted] else fastLayernormBackward%[tl.bfloat16,tl.bfloat16,tl.bfloat16,original]
  | .bf16,.bf16,.fp32 => if swapped then fastLayernormBackward%[tl.bfloat16,tl.bfloat16,tl.float32,indexCommuted] else fastLayernormBackward%[tl.bfloat16,tl.bfloat16,tl.float32,original]
  | .bf16,.fp32,.fp16 => if swapped then fastLayernormBackward%[tl.bfloat16,tl.float32,tl.float16,indexCommuted] else fastLayernormBackward%[tl.bfloat16,tl.float32,tl.float16,original]
  | .bf16,.fp32,.bf16 => if swapped then fastLayernormBackward%[tl.bfloat16,tl.float32,tl.bfloat16,indexCommuted] else fastLayernormBackward%[tl.bfloat16,tl.float32,tl.bfloat16,original]
  | .bf16,.fp32,.fp32 => if swapped then fastLayernormBackward%[tl.bfloat16,tl.float32,tl.float32,indexCommuted] else fastLayernormBackward%[tl.bfloat16,tl.float32,tl.float32,original]
  | .fp32,.fp16,.fp16 => if swapped then fastLayernormBackward%[tl.float32,tl.float16,tl.float16,indexCommuted] else fastLayernormBackward%[tl.float32,tl.float16,tl.float16,original]
  | .fp32,.fp16,.bf16 => if swapped then fastLayernormBackward%[tl.float32,tl.float16,tl.bfloat16,indexCommuted] else fastLayernormBackward%[tl.float32,tl.float16,tl.bfloat16,original]
  | .fp32,.fp16,.fp32 => if swapped then fastLayernormBackward%[tl.float32,tl.float16,tl.float32,indexCommuted] else fastLayernormBackward%[tl.float32,tl.float16,tl.float32,original]
  | .fp32,.bf16,.fp16 => if swapped then fastLayernormBackward%[tl.float32,tl.bfloat16,tl.float16,indexCommuted] else fastLayernormBackward%[tl.float32,tl.bfloat16,tl.float16,original]
  | .fp32,.bf16,.bf16 => if swapped then fastLayernormBackward%[tl.float32,tl.bfloat16,tl.bfloat16,indexCommuted] else fastLayernormBackward%[tl.float32,tl.bfloat16,tl.bfloat16,original]
  | .fp32,.bf16,.fp32 => if swapped then fastLayernormBackward%[tl.float32,tl.bfloat16,tl.float32,indexCommuted] else fastLayernormBackward%[tl.float32,tl.bfloat16,tl.float32,original]
  | .fp32,.fp32,.fp16 => if swapped then fastLayernormBackward%[tl.float32,tl.float32,tl.float16,indexCommuted] else fastLayernormBackward%[tl.float32,tl.float32,tl.float16,original]
  | .fp32,.fp32,.bf16 => if swapped then fastLayernormBackward%[tl.float32,tl.float32,tl.bfloat16,indexCommuted] else fastLayernormBackward%[tl.float32,tl.float32,tl.bfloat16,original]
  | .fp32,.fp32,.fp32 => if swapped then fastLayernormBackward%[tl.float32,tl.float32,tl.float32,indexCommuted] else fastLayernormBackward%[tl.float32,tl.float32,tl.float32,original]
```
</details>

<details><summary><code>gradient</code></summary>

```
/-- The backward body reads these statistics from the caller. Bias is loaded
but is not an operand of the returned gradient. -/
```
```lean
def gradient {B : Nat} (N : Nat) (inv avg : ℝ) (xs dys ws : Fin B → ℝ) (j : Fin B) :=
  let q := fun k => padded N dys k*padded N ws k
  let normed := fun k => (padded N xs k-avg)*inv
  (q j-(∑ k, q k)/(N : ℝ)-normed j*(∑ k, q k*normed k)/(N : ℝ))*inv


structure ForwardBounds (Y X W b r mu : RegionName) (YR XR N B : Nat)
    (u : BlockState) (bounds : RegionBounds) : Prop where
  input : ∀ j : Fin B, j.val<N → u.pids 0*XR+j.val<bounds X
  weight : ∀ j : Fin B, j.val<N → j.val<bounds W
  bias : ∀ j : Fin B, j.val<N → j.val<bounds b
  output : ∀ j : Fin B, j.val<N → u.pids 0*YR+j.val<bounds Y
  inv : u.pids 0<bounds r
  mean : u.pids 0<bounds mu
```
</details>

<details><summary><code>backwardOutside</code></summary>

```lean
def backwardOutside (A : FlatAlloc) (u : BlockState) (dY : RegionName) (YR N B : Nat)
    (q : RegionName) (o : Nat) : Prop :=
  q≠A.flat ∨ ∀ j : Fin B,j.val<N → o≠A.addr dY (u.pids 0*YR+j.val)
```
</details>

<details><summary><code>padded</code></summary>

```lean
def padded {B : Nat} (N : Nat) (xs : Fin B → ℝ) (j : Fin B) :=
  if j.val<N then xs j else 0
```
</details>

<details><summary><code>output</code></summary>

```lean
def output {B : Nat} (N : Nat) (eps : ℝ) (xs ws bs : Fin B → ℝ) (j : Fin B) :=
  ((xs j-mean N xs)*invStd N eps xs)*ws j+bs j
```
</details>

<details><summary><code>mean</code></summary>

```lean
def mean {B : Nat} (N : Nat) (xs : Fin B → ℝ) := (∑ j, padded N xs j)/(N : ℝ)
```
</details>

<details><summary><code>invStd</code></summary>

```lean
def invStd {B : Nat} (N : Nat) (eps : ℝ) (xs : Fin B → ℝ) :=
  1/Real.sqrt (variance N xs+eps)
```
</details>

<details><summary><code>variance</code></summary>

```
/-- Every lane is centered, including the initially zero-padded lanes. -/
```
```lean
def variance {B : Nat} (N : Nat) (xs : Fin B → ℝ) :=
  (∑ j, (padded N xs j-mean N xs)^2)/(N : ℝ)
```
</details>
