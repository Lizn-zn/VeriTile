# Spec sheet — `bench/tritonbench_g/fast_rms_layernorm/Correct.lean`

**Python source:** `bench/tritonbench_g/fast_rms_layernorm/fast_rms_layernorm.py`

## Public theorem: `forward_correctness`

**Statement:**
```lean
specification forward_correctness (input weight : Storage) (Y X W r : RegionName)
    (YR XR WR RR N B : Nat) (eps : ℝ) :
    Spec.Real (forward_correct_target (forward input weight Bool.false Y X W r YR XR WR RR N B eps).eraseDType.toAlgKernel
      Bool.false Y X W r YR XR WR RR N B eps)
```

**Closed-form spec defs (transitive):** `forward_correct_target`, `forward`, `meanSquare`, `output`, `weightOffset`, `invStd`, `forwardOutside`, `squareSum`, `gemma`, `padded`

<details><summary><code>forward_correct_target</code></summary>

```lean
def forward_correct_target (kernel : Kernel) (isGemma : Bool) (Y X W r : RegionName)
    (YR XR WR RR N B : Nat) (eps : ℝ) : Prop :=
  ∀ A : FlatAlloc,A.Disjoint → (∀ q,q∉A.regions → A.extent q=0) →
  ∀ u : BlockState,u.undef=(fun _ _ => 0) → ForwardBounds isGemma Y X W r YR XR WR RR N B u A.extent →
  0<N → N≤B → Y≠r →
  0<meanSquare N (fun j : Fin B => u.readMem X (u.pids 0*XR+j.val))+eps →
    ∃ v,exec (A.flattenKernel kernel) (A.flattenState u)=some v ∧
      (∀ j : Fin B,j.val<N → v.readMem A.flat (A.addr Y (u.pids 0*YR+j.val))=
        output isGemma N eps (fun k : Fin B => u.readMem X (u.pids 0*XR+k.val))
          (fun k : Fin B => u.readMem W (weightOffset isGemma WR k.val)) j) ∧
      v.readMem A.flat (A.addr r (u.pids 0*RR))=
        invStd N eps (fun j : Fin B => u.readMem X (u.pids 0*XR+j.val)) ∧
      (∀ q o,forwardOutside A u Y r YR RR N B q o → v.mem q o=(A.flattenState u).mem q o)
```
</details>

<details><summary><code>forward</code></summary>

```lean
def forward (input weight : Storage) (swapped : Bool) := match input,weight with
  | .fp16,.fp16 => if swapped then fastRmsForward%[tl.float16,tl.float16,indexCommuted] else fastRmsForward%[tl.float16,tl.float16,original]
  | .fp16,.bf16 => if swapped then fastRmsForward%[tl.float16,tl.bfloat16,indexCommuted] else fastRmsForward%[tl.float16,tl.bfloat16,original]
  | .fp16,.fp32 => if swapped then fastRmsForward%[tl.float16,tl.float32,indexCommuted] else fastRmsForward%[tl.float16,tl.float32,original]
  | .bf16,.fp16 => if swapped then fastRmsForward%[tl.bfloat16,tl.float16,indexCommuted] else fastRmsForward%[tl.bfloat16,tl.float16,original]
  | .bf16,.bf16 => if swapped then fastRmsForward%[tl.bfloat16,tl.bfloat16,indexCommuted] else fastRmsForward%[tl.bfloat16,tl.bfloat16,original]
  | .bf16,.fp32 => if swapped then fastRmsForward%[tl.bfloat16,tl.float32,indexCommuted] else fastRmsForward%[tl.bfloat16,tl.float32,original]
  | .fp32,.fp16 => if swapped then fastRmsForward%[tl.float32,tl.float16,indexCommuted] else fastRmsForward%[tl.float32,tl.float16,original]
  | .fp32,.bf16 => if swapped then fastRmsForward%[tl.float32,tl.bfloat16,indexCommuted] else fastRmsForward%[tl.float32,tl.bfloat16,original]
  | .fp32,.fp32 => if swapped then fastRmsForward%[tl.float32,tl.float32,indexCommuted] else fastRmsForward%[tl.float32,tl.float32,original]

syntax "fastRmsGemma%" "[" tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(fastRmsGemma%[$inputTy:tritonDType,$weightTy:tritonDType,$version:ident]) => do
    unless version.getId == `original || version.getId == `indexCommuted do
      Macro.throwError "expected original or indexCommuted"
    let rowIndex : TSyntax `tritonStmt ←
      if version.getId == `indexCommuted then `(tritonStmt| X += $(X_row_stride) * row_idx)
      else `(tritonStmt| X += row_idx * $(X_row_stride))
    `(fun (Y X W r : RegionName) (Y_row_stride X_row_stride W_row_stride r_row_stride n_cols BLOCK_SIZE : Nat) (eps : ℝ) =>
      ComputeKernel.mk [X,W] [Y,r] (triton {
  tl.region Y = $inputTy, X = $inputTy, W = $weightTy, r = tl.float32
  row_idx = tl.program_id(0)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(n_cols)
  Y += row_idx * $(Y_row_stride)
  $rowIndex
  r += row_idx * $(r_row_stride)
  X_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)
  W_row = tl.load(W + col_offsets, mask=mask, other=0).to(tl.float32)
  row_var = tl.sum(X_row * X_row, axis=0) / $(n_cols)
  inv_var = tl.math.rsqrt(row_var + $((eps : ℝ)))
  tl.store(r, inv_var.to(tl.float32))
  normed = X_row * inv_var
  output = normed * (W_row + 1.0)
  tl.store(Y + col_offsets, output.to($inputTy:tritonDType), mask=mask)
}).surfaceBody)
```
</details>

<details><summary><code>meanSquare</code></summary>

```lean
def meanSquare {B : Nat} (N : Nat) (xs : Fin B → ℝ) := squareSum N xs/(N : ℝ)
```
</details>

<details><summary><code>output</code></summary>

```lean
def output {B : Nat} (gemma : Bool) (N : Nat) (eps : ℝ) (xs ws : Fin B → ℝ) (j : Fin B) :=
  (xs j*invStd N eps xs)*(if gemma then ws j+1 else ws j)
```
</details>

<details><summary><code>weightOffset</code></summary>

```lean
def weightOffset (isGemma : Bool) (WR j : Nat) := if isGemma then j else j*WR

structure ForwardBounds (isGemma : Bool) (Y X W r : RegionName) (YR XR WR RR N B : Nat)
    (u : BlockState) (bounds : RegionBounds) : Prop where
  input : ∀ j : Fin B,j.val<N → u.pids 0*XR+j.val<bounds X
  weight : ∀ j : Fin B,j.val<N → weightOffset isGemma WR j.val<bounds W
  output : ∀ j : Fin B,j.val<N → u.pids 0*YR+j.val<bounds Y
  inv : u.pids 0*RR<bounds r
```
</details>

<details><summary><code>invStd</code></summary>

```lean
def invStd {B : Nat} (N : Nat) (eps : ℝ) (xs : Fin B → ℝ) :=
  1/Real.sqrt (meanSquare N xs+eps)
```
</details>

<details><summary><code>forwardOutside</code></summary>

```lean
def forwardOutside (A : FlatAlloc) (u : BlockState) (Y r : RegionName) (YR RR N B : Nat)
    (q : RegionName) (o : Nat) : Prop :=
  q≠A.flat ∨ ((∀ j : Fin B,j.val<N → o≠A.addr Y (u.pids 0*YR+j.val)) ∧
    o≠A.addr r (u.pids 0*RR))
```
</details>

<details><summary><code>squareSum</code></summary>

```lean
def squareSum {B : Nat} (N : Nat) (xs : Fin B → ℝ) := ∑ j, (padded N xs j)^2
```
</details>

<details><summary><code>gemma</code></summary>

```lean
def gemma (input weight : Storage) (swapped : Bool) := match input,weight with
  | .fp16,.fp16 => if swapped then fastRmsGemma%[tl.float16,tl.float16,indexCommuted] else fastRmsGemma%[tl.float16,tl.float16,original]
  | .fp16,.bf16 => if swapped then fastRmsGemma%[tl.float16,tl.bfloat16,indexCommuted] else fastRmsGemma%[tl.float16,tl.bfloat16,original]
  | .fp16,.fp32 => if swapped then fastRmsGemma%[tl.float16,tl.float32,indexCommuted] else fastRmsGemma%[tl.float16,tl.float32,original]
  | .bf16,.fp16 => if swapped then fastRmsGemma%[tl.bfloat16,tl.float16,indexCommuted] else fastRmsGemma%[tl.bfloat16,tl.float16,original]
  | .bf16,.bf16 => if swapped then fastRmsGemma%[tl.bfloat16,tl.bfloat16,indexCommuted] else fastRmsGemma%[tl.bfloat16,tl.bfloat16,original]
  | .bf16,.fp32 => if swapped then fastRmsGemma%[tl.bfloat16,tl.float32,indexCommuted] else fastRmsGemma%[tl.bfloat16,tl.float32,original]
  | .fp32,.fp16 => if swapped then fastRmsGemma%[tl.float32,tl.float16,indexCommuted] else fastRmsGemma%[tl.float32,tl.float16,original]
  | .fp32,.bf16 => if swapped then fastRmsGemma%[tl.float32,tl.bfloat16,indexCommuted] else fastRmsGemma%[tl.float32,tl.bfloat16,original]
  | .fp32,.fp32 => if swapped then fastRmsGemma%[tl.float32,tl.float32,indexCommuted] else fastRmsGemma%[tl.float32,tl.float32,original]

syntax "fastRmsBackward%" "[" tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(fastRmsBackward%[$inputTy:tritonDType,$weightTy:tritonDType,$version:ident]) => do
    unless version.getId == `original || version.getId == `indexCommuted do
      Macro.throwError "expected original or indexCommuted"
    let rowIndex : TSyntax `tritonStmt ←
      if version.getId == `indexCommuted then `(tritonStmt| X += $(X_row_stride) * row_idx)
      else `(tritonStmt| X += row_idx * $(X_row_stride))
    `(fun (dY X W r dW : RegionName) (dY_row_stride X_row_stride W_row_stride r_row_stride dW_row_stride n_cols BLOCK_SIZE : Nat) (eps : ℝ) (GEMMA : Bool) =>
      ComputeKernel.mk [dY,X,W,r] [dY] (triton {
  tl.region dY = $inputTy, X = $inputTy, W = $weightTy, r = tl.float32, dW = $inputTy
  row_idx = tl.program_id(0)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(n_cols)
  dY += row_idx * $(dY_row_stride)
  $rowIndex
  r += row_idx * $(r_row_stride)
  dY_row = tl.load(dY + col_offsets, mask=mask, other=0).to(tl.float32)
  X_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)
  W_row = tl.load(W + col_offsets, mask=mask, other=0).to(tl.float32)
  inv_var = tl.load(r).to(tl.float32)
  normed = X_row * inv_var
  if $((GEMMA : Bool)) {
    dY_W = dY_row * (W_row + 1.0)
  } else {
    dY_W = dY_row * W_row
  }
  rowsum_dY_normed = tl.sum(dY_W * normed, axis=0)
  output = inv_var / $(n_cols) * ($(n_cols) * dY_W - normed * rowsum_dY_normed)
  tl.store(dY + col_offsets, output.to($inputTy:tritonDType), mask=mask)
}).surfaceBody)
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
specification forward_index_commuted_correctness (input weight : Storage) (Y X W r : RegionName)
    (YR XR WR RR N B : Nat) (eps : ℝ) :
    Spec.Real (forward_correct_target (forward input weight Bool.true Y X W r YR XR WR RR N B eps).eraseDType.toAlgKernel
      Bool.false Y X W r YR XR WR RR N B eps)
```

**Closed-form spec defs (transitive):** `forward_correct_target`, `forward`, `meanSquare`, `output`, `weightOffset`, `invStd`, `forwardOutside`, `squareSum`, `gemma`, `padded`

<details><summary><code>forward_correct_target</code></summary>

```lean
def forward_correct_target (kernel : Kernel) (isGemma : Bool) (Y X W r : RegionName)
    (YR XR WR RR N B : Nat) (eps : ℝ) : Prop :=
  ∀ A : FlatAlloc,A.Disjoint → (∀ q,q∉A.regions → A.extent q=0) →
  ∀ u : BlockState,u.undef=(fun _ _ => 0) → ForwardBounds isGemma Y X W r YR XR WR RR N B u A.extent →
  0<N → N≤B → Y≠r →
  0<meanSquare N (fun j : Fin B => u.readMem X (u.pids 0*XR+j.val))+eps →
    ∃ v,exec (A.flattenKernel kernel) (A.flattenState u)=some v ∧
      (∀ j : Fin B,j.val<N → v.readMem A.flat (A.addr Y (u.pids 0*YR+j.val))=
        output isGemma N eps (fun k : Fin B => u.readMem X (u.pids 0*XR+k.val))
          (fun k : Fin B => u.readMem W (weightOffset isGemma WR k.val)) j) ∧
      v.readMem A.flat (A.addr r (u.pids 0*RR))=
        invStd N eps (fun j : Fin B => u.readMem X (u.pids 0*XR+j.val)) ∧
      (∀ q o,forwardOutside A u Y r YR RR N B q o → v.mem q o=(A.flattenState u).mem q o)
```
</details>

<details><summary><code>forward</code></summary>

```lean
def forward (input weight : Storage) (swapped : Bool) := match input,weight with
  | .fp16,.fp16 => if swapped then fastRmsForward%[tl.float16,tl.float16,indexCommuted] else fastRmsForward%[tl.float16,tl.float16,original]
  | .fp16,.bf16 => if swapped then fastRmsForward%[tl.float16,tl.bfloat16,indexCommuted] else fastRmsForward%[tl.float16,tl.bfloat16,original]
  | .fp16,.fp32 => if swapped then fastRmsForward%[tl.float16,tl.float32,indexCommuted] else fastRmsForward%[tl.float16,tl.float32,original]
  | .bf16,.fp16 => if swapped then fastRmsForward%[tl.bfloat16,tl.float16,indexCommuted] else fastRmsForward%[tl.bfloat16,tl.float16,original]
  | .bf16,.bf16 => if swapped then fastRmsForward%[tl.bfloat16,tl.bfloat16,indexCommuted] else fastRmsForward%[tl.bfloat16,tl.bfloat16,original]
  | .bf16,.fp32 => if swapped then fastRmsForward%[tl.bfloat16,tl.float32,indexCommuted] else fastRmsForward%[tl.bfloat16,tl.float32,original]
  | .fp32,.fp16 => if swapped then fastRmsForward%[tl.float32,tl.float16,indexCommuted] else fastRmsForward%[tl.float32,tl.float16,original]
  | .fp32,.bf16 => if swapped then fastRmsForward%[tl.float32,tl.bfloat16,indexCommuted] else fastRmsForward%[tl.float32,tl.bfloat16,original]
  | .fp32,.fp32 => if swapped then fastRmsForward%[tl.float32,tl.float32,indexCommuted] else fastRmsForward%[tl.float32,tl.float32,original]

syntax "fastRmsGemma%" "[" tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(fastRmsGemma%[$inputTy:tritonDType,$weightTy:tritonDType,$version:ident]) => do
    unless version.getId == `original || version.getId == `indexCommuted do
      Macro.throwError "expected original or indexCommuted"
    let rowIndex : TSyntax `tritonStmt ←
      if version.getId == `indexCommuted then `(tritonStmt| X += $(X_row_stride) * row_idx)
      else `(tritonStmt| X += row_idx * $(X_row_stride))
    `(fun (Y X W r : RegionName) (Y_row_stride X_row_stride W_row_stride r_row_stride n_cols BLOCK_SIZE : Nat) (eps : ℝ) =>
      ComputeKernel.mk [X,W] [Y,r] (triton {
  tl.region Y = $inputTy, X = $inputTy, W = $weightTy, r = tl.float32
  row_idx = tl.program_id(0)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(n_cols)
  Y += row_idx * $(Y_row_stride)
  $rowIndex
  r += row_idx * $(r_row_stride)
  X_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)
  W_row = tl.load(W + col_offsets, mask=mask, other=0).to(tl.float32)
  row_var = tl.sum(X_row * X_row, axis=0) / $(n_cols)
  inv_var = tl.math.rsqrt(row_var + $((eps : ℝ)))
  tl.store(r, inv_var.to(tl.float32))
  normed = X_row * inv_var
  output = normed * (W_row + 1.0)
  tl.store(Y + col_offsets, output.to($inputTy:tritonDType), mask=mask)
}).surfaceBody)
```
</details>

<details><summary><code>meanSquare</code></summary>

```lean
def meanSquare {B : Nat} (N : Nat) (xs : Fin B → ℝ) := squareSum N xs/(N : ℝ)
```
</details>

<details><summary><code>output</code></summary>

```lean
def output {B : Nat} (gemma : Bool) (N : Nat) (eps : ℝ) (xs ws : Fin B → ℝ) (j : Fin B) :=
  (xs j*invStd N eps xs)*(if gemma then ws j+1 else ws j)
```
</details>

<details><summary><code>weightOffset</code></summary>

```lean
def weightOffset (isGemma : Bool) (WR j : Nat) := if isGemma then j else j*WR

structure ForwardBounds (isGemma : Bool) (Y X W r : RegionName) (YR XR WR RR N B : Nat)
    (u : BlockState) (bounds : RegionBounds) : Prop where
  input : ∀ j : Fin B,j.val<N → u.pids 0*XR+j.val<bounds X
  weight : ∀ j : Fin B,j.val<N → weightOffset isGemma WR j.val<bounds W
  output : ∀ j : Fin B,j.val<N → u.pids 0*YR+j.val<bounds Y
  inv : u.pids 0*RR<bounds r
```
</details>

<details><summary><code>invStd</code></summary>

```lean
def invStd {B : Nat} (N : Nat) (eps : ℝ) (xs : Fin B → ℝ) :=
  1/Real.sqrt (meanSquare N xs+eps)
```
</details>

<details><summary><code>forwardOutside</code></summary>

```lean
def forwardOutside (A : FlatAlloc) (u : BlockState) (Y r : RegionName) (YR RR N B : Nat)
    (q : RegionName) (o : Nat) : Prop :=
  q≠A.flat ∨ ((∀ j : Fin B,j.val<N → o≠A.addr Y (u.pids 0*YR+j.val)) ∧
    o≠A.addr r (u.pids 0*RR))
```
</details>

<details><summary><code>squareSum</code></summary>

```lean
def squareSum {B : Nat} (N : Nat) (xs : Fin B → ℝ) := ∑ j, (padded N xs j)^2
```
</details>

<details><summary><code>gemma</code></summary>

```lean
def gemma (input weight : Storage) (swapped : Bool) := match input,weight with
  | .fp16,.fp16 => if swapped then fastRmsGemma%[tl.float16,tl.float16,indexCommuted] else fastRmsGemma%[tl.float16,tl.float16,original]
  | .fp16,.bf16 => if swapped then fastRmsGemma%[tl.float16,tl.bfloat16,indexCommuted] else fastRmsGemma%[tl.float16,tl.bfloat16,original]
  | .fp16,.fp32 => if swapped then fastRmsGemma%[tl.float16,tl.float32,indexCommuted] else fastRmsGemma%[tl.float16,tl.float32,original]
  | .bf16,.fp16 => if swapped then fastRmsGemma%[tl.bfloat16,tl.float16,indexCommuted] else fastRmsGemma%[tl.bfloat16,tl.float16,original]
  | .bf16,.bf16 => if swapped then fastRmsGemma%[tl.bfloat16,tl.bfloat16,indexCommuted] else fastRmsGemma%[tl.bfloat16,tl.bfloat16,original]
  | .bf16,.fp32 => if swapped then fastRmsGemma%[tl.bfloat16,tl.float32,indexCommuted] else fastRmsGemma%[tl.bfloat16,tl.float32,original]
  | .fp32,.fp16 => if swapped then fastRmsGemma%[tl.float32,tl.float16,indexCommuted] else fastRmsGemma%[tl.float32,tl.float16,original]
  | .fp32,.bf16 => if swapped then fastRmsGemma%[tl.float32,tl.bfloat16,indexCommuted] else fastRmsGemma%[tl.float32,tl.bfloat16,original]
  | .fp32,.fp32 => if swapped then fastRmsGemma%[tl.float32,tl.float32,indexCommuted] else fastRmsGemma%[tl.float32,tl.float32,original]

syntax "fastRmsBackward%" "[" tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(fastRmsBackward%[$inputTy:tritonDType,$weightTy:tritonDType,$version:ident]) => do
    unless version.getId == `original || version.getId == `indexCommuted do
      Macro.throwError "expected original or indexCommuted"
    let rowIndex : TSyntax `tritonStmt ←
      if version.getId == `indexCommuted then `(tritonStmt| X += $(X_row_stride) * row_idx)
      else `(tritonStmt| X += row_idx * $(X_row_stride))
    `(fun (dY X W r dW : RegionName) (dY_row_stride X_row_stride W_row_stride r_row_stride dW_row_stride n_cols BLOCK_SIZE : Nat) (eps : ℝ) (GEMMA : Bool) =>
      ComputeKernel.mk [dY,X,W,r] [dY] (triton {
  tl.region dY = $inputTy, X = $inputTy, W = $weightTy, r = tl.float32, dW = $inputTy
  row_idx = tl.program_id(0)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(n_cols)
  dY += row_idx * $(dY_row_stride)
  $rowIndex
  r += row_idx * $(r_row_stride)
  dY_row = tl.load(dY + col_offsets, mask=mask, other=0).to(tl.float32)
  X_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)
  W_row = tl.load(W + col_offsets, mask=mask, other=0).to(tl.float32)
  inv_var = tl.load(r).to(tl.float32)
  normed = X_row * inv_var
  if $((GEMMA : Bool)) {
    dY_W = dY_row * (W_row + 1.0)
  } else {
    dY_W = dY_row * W_row
  }
  rowsum_dY_normed = tl.sum(dY_W * normed, axis=0)
  output = inv_var / $(n_cols) * ($(n_cols) * dY_W - normed * rowsum_dY_normed)
  tl.store(dY + col_offsets, output.to($inputTy:tritonDType), mask=mask)
}).surfaceBody)
```
</details>

<details><summary><code>padded</code></summary>

```lean
def padded {B : Nat} (N : Nat) (xs : Fin B → ℝ) (j : Fin B) :=
  if j.val<N then xs j else 0
```
</details>

## Public theorem: `gemma_correctness`

**Statement:**
```lean
specification gemma_correctness (input weight : Storage) (Y X W r : RegionName)
    (YR XR WR RR N B : Nat) (eps : ℝ) :
    Spec.Real (forward_correct_target (gemma input weight Bool.false Y X W r YR XR WR RR N B eps).eraseDType.toAlgKernel
      Bool.true Y X W r YR XR WR RR N B eps)
```

**Closed-form spec defs (transitive):** `forward_correct_target`, `gemma`, `meanSquare`, `output`, `weightOffset`, `invStd`, `forwardOutside`, `squareSum`, `padded`

<details><summary><code>forward_correct_target</code></summary>

```lean
def forward_correct_target (kernel : Kernel) (isGemma : Bool) (Y X W r : RegionName)
    (YR XR WR RR N B : Nat) (eps : ℝ) : Prop :=
  ∀ A : FlatAlloc,A.Disjoint → (∀ q,q∉A.regions → A.extent q=0) →
  ∀ u : BlockState,u.undef=(fun _ _ => 0) → ForwardBounds isGemma Y X W r YR XR WR RR N B u A.extent →
  0<N → N≤B → Y≠r →
  0<meanSquare N (fun j : Fin B => u.readMem X (u.pids 0*XR+j.val))+eps →
    ∃ v,exec (A.flattenKernel kernel) (A.flattenState u)=some v ∧
      (∀ j : Fin B,j.val<N → v.readMem A.flat (A.addr Y (u.pids 0*YR+j.val))=
        output isGemma N eps (fun k : Fin B => u.readMem X (u.pids 0*XR+k.val))
          (fun k : Fin B => u.readMem W (weightOffset isGemma WR k.val)) j) ∧
      v.readMem A.flat (A.addr r (u.pids 0*RR))=
        invStd N eps (fun j : Fin B => u.readMem X (u.pids 0*XR+j.val)) ∧
      (∀ q o,forwardOutside A u Y r YR RR N B q o → v.mem q o=(A.flattenState u).mem q o)
```
</details>

<details><summary><code>gemma</code></summary>

```lean
def gemma (input weight : Storage) (swapped : Bool) := match input,weight with
  | .fp16,.fp16 => if swapped then fastRmsGemma%[tl.float16,tl.float16,indexCommuted] else fastRmsGemma%[tl.float16,tl.float16,original]
  | .fp16,.bf16 => if swapped then fastRmsGemma%[tl.float16,tl.bfloat16,indexCommuted] else fastRmsGemma%[tl.float16,tl.bfloat16,original]
  | .fp16,.fp32 => if swapped then fastRmsGemma%[tl.float16,tl.float32,indexCommuted] else fastRmsGemma%[tl.float16,tl.float32,original]
  | .bf16,.fp16 => if swapped then fastRmsGemma%[tl.bfloat16,tl.float16,indexCommuted] else fastRmsGemma%[tl.bfloat16,tl.float16,original]
  | .bf16,.bf16 => if swapped then fastRmsGemma%[tl.bfloat16,tl.bfloat16,indexCommuted] else fastRmsGemma%[tl.bfloat16,tl.bfloat16,original]
  | .bf16,.fp32 => if swapped then fastRmsGemma%[tl.bfloat16,tl.float32,indexCommuted] else fastRmsGemma%[tl.bfloat16,tl.float32,original]
  | .fp32,.fp16 => if swapped then fastRmsGemma%[tl.float32,tl.float16,indexCommuted] else fastRmsGemma%[tl.float32,tl.float16,original]
  | .fp32,.bf16 => if swapped then fastRmsGemma%[tl.float32,tl.bfloat16,indexCommuted] else fastRmsGemma%[tl.float32,tl.bfloat16,original]
  | .fp32,.fp32 => if swapped then fastRmsGemma%[tl.float32,tl.float32,indexCommuted] else fastRmsGemma%[tl.float32,tl.float32,original]

syntax "fastRmsBackward%" "[" tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(fastRmsBackward%[$inputTy:tritonDType,$weightTy:tritonDType,$version:ident]) => do
    unless version.getId == `original || version.getId == `indexCommuted do
      Macro.throwError "expected original or indexCommuted"
    let rowIndex : TSyntax `tritonStmt ←
      if version.getId == `indexCommuted then `(tritonStmt| X += $(X_row_stride) * row_idx)
      else `(tritonStmt| X += row_idx * $(X_row_stride))
    `(fun (dY X W r dW : RegionName) (dY_row_stride X_row_stride W_row_stride r_row_stride dW_row_stride n_cols BLOCK_SIZE : Nat) (eps : ℝ) (GEMMA : Bool) =>
      ComputeKernel.mk [dY,X,W,r] [dY] (triton {
  tl.region dY = $inputTy, X = $inputTy, W = $weightTy, r = tl.float32, dW = $inputTy
  row_idx = tl.program_id(0)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(n_cols)
  dY += row_idx * $(dY_row_stride)
  $rowIndex
  r += row_idx * $(r_row_stride)
  dY_row = tl.load(dY + col_offsets, mask=mask, other=0).to(tl.float32)
  X_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)
  W_row = tl.load(W + col_offsets, mask=mask, other=0).to(tl.float32)
  inv_var = tl.load(r).to(tl.float32)
  normed = X_row * inv_var
  if $((GEMMA : Bool)) {
    dY_W = dY_row * (W_row + 1.0)
  } else {
    dY_W = dY_row * W_row
  }
  rowsum_dY_normed = tl.sum(dY_W * normed, axis=0)
  output = inv_var / $(n_cols) * ($(n_cols) * dY_W - normed * rowsum_dY_normed)
  tl.store(dY + col_offsets, output.to($inputTy:tritonDType), mask=mask)
}).surfaceBody)
```
</details>

<details><summary><code>meanSquare</code></summary>

```lean
def meanSquare {B : Nat} (N : Nat) (xs : Fin B → ℝ) := squareSum N xs/(N : ℝ)
```
</details>

<details><summary><code>output</code></summary>

```lean
def output {B : Nat} (gemma : Bool) (N : Nat) (eps : ℝ) (xs ws : Fin B → ℝ) (j : Fin B) :=
  (xs j*invStd N eps xs)*(if gemma then ws j+1 else ws j)
```
</details>

<details><summary><code>weightOffset</code></summary>

```lean
def weightOffset (isGemma : Bool) (WR j : Nat) := if isGemma then j else j*WR

structure ForwardBounds (isGemma : Bool) (Y X W r : RegionName) (YR XR WR RR N B : Nat)
    (u : BlockState) (bounds : RegionBounds) : Prop where
  input : ∀ j : Fin B,j.val<N → u.pids 0*XR+j.val<bounds X
  weight : ∀ j : Fin B,j.val<N → weightOffset isGemma WR j.val<bounds W
  output : ∀ j : Fin B,j.val<N → u.pids 0*YR+j.val<bounds Y
  inv : u.pids 0*RR<bounds r
```
</details>

<details><summary><code>invStd</code></summary>

```lean
def invStd {B : Nat} (N : Nat) (eps : ℝ) (xs : Fin B → ℝ) :=
  1/Real.sqrt (meanSquare N xs+eps)
```
</details>

<details><summary><code>forwardOutside</code></summary>

```lean
def forwardOutside (A : FlatAlloc) (u : BlockState) (Y r : RegionName) (YR RR N B : Nat)
    (q : RegionName) (o : Nat) : Prop :=
  q≠A.flat ∨ ((∀ j : Fin B,j.val<N → o≠A.addr Y (u.pids 0*YR+j.val)) ∧
    o≠A.addr r (u.pids 0*RR))
```
</details>

<details><summary><code>squareSum</code></summary>

```lean
def squareSum {B : Nat} (N : Nat) (xs : Fin B → ℝ) := ∑ j, (padded N xs j)^2
```
</details>

<details><summary><code>padded</code></summary>

```lean
def padded {B : Nat} (N : Nat) (xs : Fin B → ℝ) (j : Fin B) :=
  if j.val<N then xs j else 0
```
</details>

## Public theorem: `gemma_index_commuted_correctness`

**Statement:**
```lean
specification gemma_index_commuted_correctness (input weight : Storage) (Y X W r : RegionName)
    (YR XR WR RR N B : Nat) (eps : ℝ) :
    Spec.Real (forward_correct_target (gemma input weight Bool.true Y X W r YR XR WR RR N B eps).eraseDType.toAlgKernel
      Bool.true Y X W r YR XR WR RR N B eps)
```

**Closed-form spec defs (transitive):** `forward_correct_target`, `gemma`, `meanSquare`, `output`, `weightOffset`, `invStd`, `forwardOutside`, `squareSum`, `padded`

<details><summary><code>forward_correct_target</code></summary>

```lean
def forward_correct_target (kernel : Kernel) (isGemma : Bool) (Y X W r : RegionName)
    (YR XR WR RR N B : Nat) (eps : ℝ) : Prop :=
  ∀ A : FlatAlloc,A.Disjoint → (∀ q,q∉A.regions → A.extent q=0) →
  ∀ u : BlockState,u.undef=(fun _ _ => 0) → ForwardBounds isGemma Y X W r YR XR WR RR N B u A.extent →
  0<N → N≤B → Y≠r →
  0<meanSquare N (fun j : Fin B => u.readMem X (u.pids 0*XR+j.val))+eps →
    ∃ v,exec (A.flattenKernel kernel) (A.flattenState u)=some v ∧
      (∀ j : Fin B,j.val<N → v.readMem A.flat (A.addr Y (u.pids 0*YR+j.val))=
        output isGemma N eps (fun k : Fin B => u.readMem X (u.pids 0*XR+k.val))
          (fun k : Fin B => u.readMem W (weightOffset isGemma WR k.val)) j) ∧
      v.readMem A.flat (A.addr r (u.pids 0*RR))=
        invStd N eps (fun j : Fin B => u.readMem X (u.pids 0*XR+j.val)) ∧
      (∀ q o,forwardOutside A u Y r YR RR N B q o → v.mem q o=(A.flattenState u).mem q o)
```
</details>

<details><summary><code>gemma</code></summary>

```lean
def gemma (input weight : Storage) (swapped : Bool) := match input,weight with
  | .fp16,.fp16 => if swapped then fastRmsGemma%[tl.float16,tl.float16,indexCommuted] else fastRmsGemma%[tl.float16,tl.float16,original]
  | .fp16,.bf16 => if swapped then fastRmsGemma%[tl.float16,tl.bfloat16,indexCommuted] else fastRmsGemma%[tl.float16,tl.bfloat16,original]
  | .fp16,.fp32 => if swapped then fastRmsGemma%[tl.float16,tl.float32,indexCommuted] else fastRmsGemma%[tl.float16,tl.float32,original]
  | .bf16,.fp16 => if swapped then fastRmsGemma%[tl.bfloat16,tl.float16,indexCommuted] else fastRmsGemma%[tl.bfloat16,tl.float16,original]
  | .bf16,.bf16 => if swapped then fastRmsGemma%[tl.bfloat16,tl.bfloat16,indexCommuted] else fastRmsGemma%[tl.bfloat16,tl.bfloat16,original]
  | .bf16,.fp32 => if swapped then fastRmsGemma%[tl.bfloat16,tl.float32,indexCommuted] else fastRmsGemma%[tl.bfloat16,tl.float32,original]
  | .fp32,.fp16 => if swapped then fastRmsGemma%[tl.float32,tl.float16,indexCommuted] else fastRmsGemma%[tl.float32,tl.float16,original]
  | .fp32,.bf16 => if swapped then fastRmsGemma%[tl.float32,tl.bfloat16,indexCommuted] else fastRmsGemma%[tl.float32,tl.bfloat16,original]
  | .fp32,.fp32 => if swapped then fastRmsGemma%[tl.float32,tl.float32,indexCommuted] else fastRmsGemma%[tl.float32,tl.float32,original]

syntax "fastRmsBackward%" "[" tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(fastRmsBackward%[$inputTy:tritonDType,$weightTy:tritonDType,$version:ident]) => do
    unless version.getId == `original || version.getId == `indexCommuted do
      Macro.throwError "expected original or indexCommuted"
    let rowIndex : TSyntax `tritonStmt ←
      if version.getId == `indexCommuted then `(tritonStmt| X += $(X_row_stride) * row_idx)
      else `(tritonStmt| X += row_idx * $(X_row_stride))
    `(fun (dY X W r dW : RegionName) (dY_row_stride X_row_stride W_row_stride r_row_stride dW_row_stride n_cols BLOCK_SIZE : Nat) (eps : ℝ) (GEMMA : Bool) =>
      ComputeKernel.mk [dY,X,W,r] [dY] (triton {
  tl.region dY = $inputTy, X = $inputTy, W = $weightTy, r = tl.float32, dW = $inputTy
  row_idx = tl.program_id(0)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(n_cols)
  dY += row_idx * $(dY_row_stride)
  $rowIndex
  r += row_idx * $(r_row_stride)
  dY_row = tl.load(dY + col_offsets, mask=mask, other=0).to(tl.float32)
  X_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)
  W_row = tl.load(W + col_offsets, mask=mask, other=0).to(tl.float32)
  inv_var = tl.load(r).to(tl.float32)
  normed = X_row * inv_var
  if $((GEMMA : Bool)) {
    dY_W = dY_row * (W_row + 1.0)
  } else {
    dY_W = dY_row * W_row
  }
  rowsum_dY_normed = tl.sum(dY_W * normed, axis=0)
  output = inv_var / $(n_cols) * ($(n_cols) * dY_W - normed * rowsum_dY_normed)
  tl.store(dY + col_offsets, output.to($inputTy:tritonDType), mask=mask)
}).surfaceBody)
```
</details>

<details><summary><code>meanSquare</code></summary>

```lean
def meanSquare {B : Nat} (N : Nat) (xs : Fin B → ℝ) := squareSum N xs/(N : ℝ)
```
</details>

<details><summary><code>output</code></summary>

```lean
def output {B : Nat} (gemma : Bool) (N : Nat) (eps : ℝ) (xs ws : Fin B → ℝ) (j : Fin B) :=
  (xs j*invStd N eps xs)*(if gemma then ws j+1 else ws j)
```
</details>

<details><summary><code>weightOffset</code></summary>

```lean
def weightOffset (isGemma : Bool) (WR j : Nat) := if isGemma then j else j*WR

structure ForwardBounds (isGemma : Bool) (Y X W r : RegionName) (YR XR WR RR N B : Nat)
    (u : BlockState) (bounds : RegionBounds) : Prop where
  input : ∀ j : Fin B,j.val<N → u.pids 0*XR+j.val<bounds X
  weight : ∀ j : Fin B,j.val<N → weightOffset isGemma WR j.val<bounds W
  output : ∀ j : Fin B,j.val<N → u.pids 0*YR+j.val<bounds Y
  inv : u.pids 0*RR<bounds r
```
</details>

<details><summary><code>invStd</code></summary>

```lean
def invStd {B : Nat} (N : Nat) (eps : ℝ) (xs : Fin B → ℝ) :=
  1/Real.sqrt (meanSquare N xs+eps)
```
</details>

<details><summary><code>forwardOutside</code></summary>

```lean
def forwardOutside (A : FlatAlloc) (u : BlockState) (Y r : RegionName) (YR RR N B : Nat)
    (q : RegionName) (o : Nat) : Prop :=
  q≠A.flat ∨ ((∀ j : Fin B,j.val<N → o≠A.addr Y (u.pids 0*YR+j.val)) ∧
    o≠A.addr r (u.pids 0*RR))
```
</details>

<details><summary><code>squareSum</code></summary>

```lean
def squareSum {B : Nat} (N : Nat) (xs : Fin B → ℝ) := ∑ j, (padded N xs j)^2
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
specification backward_correctness (input weight : Storage) (isGemma : Bool) (Y X W r dW : RegionName)
    (YR XR WR RR DWR N B : Nat) (eps : ℝ) :
    Spec.Real (backward_correct_target (backward input weight Bool.false Y X W r dW YR XR WR RR DWR N B eps isGemma).eraseDType.toAlgKernel
      isGemma Y X W r YR XR RR N B)
```

**Closed-form spec defs (transitive):** `backward_correct_target`, `backward`, `gradient`, `backwardOutside`, `padded`, `gemma`, `output`, `invStd`, `meanSquare`, `squareSum`

<details><summary><code>backward_correct_target</code></summary>

```lean
def backward_correct_target (kernel : Kernel) (isGemma : Bool) (Y X W r : RegionName) (YR XR RR N B : Nat) : Prop :=
  ∀ A : FlatAlloc,A.Disjoint → (∀ q,q∉A.regions → A.extent q=0) →
  ∀ u : BlockState,u.undef=(fun _ _ => 0) → BackwardBounds Y X W r YR XR RR N B u A.extent →
  0<N → N≤B →
    ∃ v,exec (A.flattenKernel kernel) (A.flattenState u)=some v ∧
      (∀ j : Fin B,j.val<N → v.readMem A.flat (A.addr Y (u.pids 0*YR+j.val))=
        gradient isGemma N (u.readMem r (u.pids 0*RR))
          (fun k : Fin B => u.readMem X (u.pids 0*XR+k.val))
          (fun k : Fin B => u.readMem Y (u.pids 0*YR+k.val))
          (fun k : Fin B => u.readMem W k.val) j) ∧
      (∀ q o,backwardOutside A u Y YR N B q o → v.mem q o=(A.flattenState u).mem q o)
```
</details>

<details><summary><code>backward</code></summary>

```lean
def backward (input weight : Storage) (swapped : Bool) := match input,weight with
  | .fp16,.fp16 => if swapped then fastRmsBackward%[tl.float16,tl.float16,indexCommuted] else fastRmsBackward%[tl.float16,tl.float16,original]
  | .fp16,.bf16 => if swapped then fastRmsBackward%[tl.float16,tl.bfloat16,indexCommuted] else fastRmsBackward%[tl.float16,tl.bfloat16,original]
  | .fp16,.fp32 => if swapped then fastRmsBackward%[tl.float16,tl.float32,indexCommuted] else fastRmsBackward%[tl.float16,tl.float32,original]
  | .bf16,.fp16 => if swapped then fastRmsBackward%[tl.bfloat16,tl.float16,indexCommuted] else fastRmsBackward%[tl.bfloat16,tl.float16,original]
  | .bf16,.bf16 => if swapped then fastRmsBackward%[tl.bfloat16,tl.bfloat16,indexCommuted] else fastRmsBackward%[tl.bfloat16,tl.bfloat16,original]
  | .bf16,.fp32 => if swapped then fastRmsBackward%[tl.bfloat16,tl.float32,indexCommuted] else fastRmsBackward%[tl.bfloat16,tl.float32,original]
  | .fp32,.fp16 => if swapped then fastRmsBackward%[tl.float32,tl.float16,indexCommuted] else fastRmsBackward%[tl.float32,tl.float16,original]
  | .fp32,.bf16 => if swapped then fastRmsBackward%[tl.float32,tl.bfloat16,indexCommuted] else fastRmsBackward%[tl.float32,tl.bfloat16,original]
  | .fp32,.fp32 => if swapped then fastRmsBackward%[tl.float32,tl.float32,indexCommuted] else fastRmsBackward%[tl.float32,tl.float32,original]
```
</details>

<details><summary><code>gradient</code></summary>

```
/-- Backward consumes caller statistics, and always reads weights contiguously. -/
```
```lean
def gradient {B : Nat} (gemma : Bool) (N : Nat) (inv : ℝ) (xs dys ws : Fin B → ℝ) (j : Fin B) :=
  let q := fun k => padded N dys k*(if gemma then padded N ws k+1 else padded N ws k)
  let z := fun k => padded N xs k*inv
  inv/(N : ℝ)*((N : ℝ)*q j-z j*(∑ k,q k*z k))
```
</details>

<details><summary><code>backwardOutside</code></summary>

```lean
def backwardOutside (A : FlatAlloc) (u : BlockState) (Y : RegionName) (YR N B : Nat)
    (q : RegionName) (o : Nat) : Prop :=
  q≠A.flat ∨ ∀ j : Fin B,j.val<N → o≠A.addr Y (u.pids 0*YR+j.val)
```
</details>

<details><summary><code>padded</code></summary>

```lean
def padded {B : Nat} (N : Nat) (xs : Fin B → ℝ) (j : Fin B) :=
  if j.val<N then xs j else 0
```
</details>

<details><summary><code>gemma</code></summary>

```lean
def gemma (input weight : Storage) (swapped : Bool) := match input,weight with
  | .fp16,.fp16 => if swapped then fastRmsGemma%[tl.float16,tl.float16,indexCommuted] else fastRmsGemma%[tl.float16,tl.float16,original]
  | .fp16,.bf16 => if swapped then fastRmsGemma%[tl.float16,tl.bfloat16,indexCommuted] else fastRmsGemma%[tl.float16,tl.bfloat16,original]
  | .fp16,.fp32 => if swapped then fastRmsGemma%[tl.float16,tl.float32,indexCommuted] else fastRmsGemma%[tl.float16,tl.float32,original]
  | .bf16,.fp16 => if swapped then fastRmsGemma%[tl.bfloat16,tl.float16,indexCommuted] else fastRmsGemma%[tl.bfloat16,tl.float16,original]
  | .bf16,.bf16 => if swapped then fastRmsGemma%[tl.bfloat16,tl.bfloat16,indexCommuted] else fastRmsGemma%[tl.bfloat16,tl.bfloat16,original]
  | .bf16,.fp32 => if swapped then fastRmsGemma%[tl.bfloat16,tl.float32,indexCommuted] else fastRmsGemma%[tl.bfloat16,tl.float32,original]
  | .fp32,.fp16 => if swapped then fastRmsGemma%[tl.float32,tl.float16,indexCommuted] else fastRmsGemma%[tl.float32,tl.float16,original]
  | .fp32,.bf16 => if swapped then fastRmsGemma%[tl.float32,tl.bfloat16,indexCommuted] else fastRmsGemma%[tl.float32,tl.bfloat16,original]
  | .fp32,.fp32 => if swapped then fastRmsGemma%[tl.float32,tl.float32,indexCommuted] else fastRmsGemma%[tl.float32,tl.float32,original]

syntax "fastRmsBackward%" "[" tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(fastRmsBackward%[$inputTy:tritonDType,$weightTy:tritonDType,$version:ident]) => do
    unless version.getId == `original || version.getId == `indexCommuted do
      Macro.throwError "expected original or indexCommuted"
    let rowIndex : TSyntax `tritonStmt ←
      if version.getId == `indexCommuted then `(tritonStmt| X += $(X_row_stride) * row_idx)
      else `(tritonStmt| X += row_idx * $(X_row_stride))
    `(fun (dY X W r dW : RegionName) (dY_row_stride X_row_stride W_row_stride r_row_stride dW_row_stride n_cols BLOCK_SIZE : Nat) (eps : ℝ) (GEMMA : Bool) =>
      ComputeKernel.mk [dY,X,W,r] [dY] (triton {
  tl.region dY = $inputTy, X = $inputTy, W = $weightTy, r = tl.float32, dW = $inputTy
  row_idx = tl.program_id(0)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(n_cols)
  dY += row_idx * $(dY_row_stride)
  $rowIndex
  r += row_idx * $(r_row_stride)
  dY_row = tl.load(dY + col_offsets, mask=mask, other=0).to(tl.float32)
  X_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)
  W_row = tl.load(W + col_offsets, mask=mask, other=0).to(tl.float32)
  inv_var = tl.load(r).to(tl.float32)
  normed = X_row * inv_var
  if $((GEMMA : Bool)) {
    dY_W = dY_row * (W_row + 1.0)
  } else {
    dY_W = dY_row * W_row
  }
  rowsum_dY_normed = tl.sum(dY_W * normed, axis=0)
  output = inv_var / $(n_cols) * ($(n_cols) * dY_W - normed * rowsum_dY_normed)
  tl.store(dY + col_offsets, output.to($inputTy:tritonDType), mask=mask)
}).surfaceBody)
```
</details>

<details><summary><code>output</code></summary>

```lean
def output {B : Nat} (gemma : Bool) (N : Nat) (eps : ℝ) (xs ws : Fin B → ℝ) (j : Fin B) :=
  (xs j*invStd N eps xs)*(if gemma then ws j+1 else ws j)
```
</details>

<details><summary><code>invStd</code></summary>

```lean
def invStd {B : Nat} (N : Nat) (eps : ℝ) (xs : Fin B → ℝ) :=
  1/Real.sqrt (meanSquare N xs+eps)
```
</details>

<details><summary><code>meanSquare</code></summary>

```lean
def meanSquare {B : Nat} (N : Nat) (xs : Fin B → ℝ) := squareSum N xs/(N : ℝ)
```
</details>

<details><summary><code>squareSum</code></summary>

```lean
def squareSum {B : Nat} (N : Nat) (xs : Fin B → ℝ) := ∑ j, (padded N xs j)^2
```
</details>

## Public theorem: `backward_index_commuted_correctness`

**Statement:**
```lean
specification backward_index_commuted_correctness (input weight : Storage) (isGemma : Bool) (Y X W r dW : RegionName)
    (YR XR WR RR DWR N B : Nat) (eps : ℝ) :
    Spec.Real (backward_correct_target (backward input weight Bool.true Y X W r dW YR XR WR RR DWR N B eps isGemma).eraseDType.toAlgKernel
      isGemma Y X W r YR XR RR N B)
```

**Closed-form spec defs (transitive):** `backward_correct_target`, `backward`, `gradient`, `backwardOutside`, `padded`, `gemma`, `output`, `invStd`, `meanSquare`, `squareSum`

<details><summary><code>backward_correct_target</code></summary>

```lean
def backward_correct_target (kernel : Kernel) (isGemma : Bool) (Y X W r : RegionName) (YR XR RR N B : Nat) : Prop :=
  ∀ A : FlatAlloc,A.Disjoint → (∀ q,q∉A.regions → A.extent q=0) →
  ∀ u : BlockState,u.undef=(fun _ _ => 0) → BackwardBounds Y X W r YR XR RR N B u A.extent →
  0<N → N≤B →
    ∃ v,exec (A.flattenKernel kernel) (A.flattenState u)=some v ∧
      (∀ j : Fin B,j.val<N → v.readMem A.flat (A.addr Y (u.pids 0*YR+j.val))=
        gradient isGemma N (u.readMem r (u.pids 0*RR))
          (fun k : Fin B => u.readMem X (u.pids 0*XR+k.val))
          (fun k : Fin B => u.readMem Y (u.pids 0*YR+k.val))
          (fun k : Fin B => u.readMem W k.val) j) ∧
      (∀ q o,backwardOutside A u Y YR N B q o → v.mem q o=(A.flattenState u).mem q o)
```
</details>

<details><summary><code>backward</code></summary>

```lean
def backward (input weight : Storage) (swapped : Bool) := match input,weight with
  | .fp16,.fp16 => if swapped then fastRmsBackward%[tl.float16,tl.float16,indexCommuted] else fastRmsBackward%[tl.float16,tl.float16,original]
  | .fp16,.bf16 => if swapped then fastRmsBackward%[tl.float16,tl.bfloat16,indexCommuted] else fastRmsBackward%[tl.float16,tl.bfloat16,original]
  | .fp16,.fp32 => if swapped then fastRmsBackward%[tl.float16,tl.float32,indexCommuted] else fastRmsBackward%[tl.float16,tl.float32,original]
  | .bf16,.fp16 => if swapped then fastRmsBackward%[tl.bfloat16,tl.float16,indexCommuted] else fastRmsBackward%[tl.bfloat16,tl.float16,original]
  | .bf16,.bf16 => if swapped then fastRmsBackward%[tl.bfloat16,tl.bfloat16,indexCommuted] else fastRmsBackward%[tl.bfloat16,tl.bfloat16,original]
  | .bf16,.fp32 => if swapped then fastRmsBackward%[tl.bfloat16,tl.float32,indexCommuted] else fastRmsBackward%[tl.bfloat16,tl.float32,original]
  | .fp32,.fp16 => if swapped then fastRmsBackward%[tl.float32,tl.float16,indexCommuted] else fastRmsBackward%[tl.float32,tl.float16,original]
  | .fp32,.bf16 => if swapped then fastRmsBackward%[tl.float32,tl.bfloat16,indexCommuted] else fastRmsBackward%[tl.float32,tl.bfloat16,original]
  | .fp32,.fp32 => if swapped then fastRmsBackward%[tl.float32,tl.float32,indexCommuted] else fastRmsBackward%[tl.float32,tl.float32,original]
```
</details>

<details><summary><code>gradient</code></summary>

```
/-- Backward consumes caller statistics, and always reads weights contiguously. -/
```
```lean
def gradient {B : Nat} (gemma : Bool) (N : Nat) (inv : ℝ) (xs dys ws : Fin B → ℝ) (j : Fin B) :=
  let q := fun k => padded N dys k*(if gemma then padded N ws k+1 else padded N ws k)
  let z := fun k => padded N xs k*inv
  inv/(N : ℝ)*((N : ℝ)*q j-z j*(∑ k,q k*z k))
```
</details>

<details><summary><code>backwardOutside</code></summary>

```lean
def backwardOutside (A : FlatAlloc) (u : BlockState) (Y : RegionName) (YR N B : Nat)
    (q : RegionName) (o : Nat) : Prop :=
  q≠A.flat ∨ ∀ j : Fin B,j.val<N → o≠A.addr Y (u.pids 0*YR+j.val)
```
</details>

<details><summary><code>padded</code></summary>

```lean
def padded {B : Nat} (N : Nat) (xs : Fin B → ℝ) (j : Fin B) :=
  if j.val<N then xs j else 0
```
</details>

<details><summary><code>gemma</code></summary>

```lean
def gemma (input weight : Storage) (swapped : Bool) := match input,weight with
  | .fp16,.fp16 => if swapped then fastRmsGemma%[tl.float16,tl.float16,indexCommuted] else fastRmsGemma%[tl.float16,tl.float16,original]
  | .fp16,.bf16 => if swapped then fastRmsGemma%[tl.float16,tl.bfloat16,indexCommuted] else fastRmsGemma%[tl.float16,tl.bfloat16,original]
  | .fp16,.fp32 => if swapped then fastRmsGemma%[tl.float16,tl.float32,indexCommuted] else fastRmsGemma%[tl.float16,tl.float32,original]
  | .bf16,.fp16 => if swapped then fastRmsGemma%[tl.bfloat16,tl.float16,indexCommuted] else fastRmsGemma%[tl.bfloat16,tl.float16,original]
  | .bf16,.bf16 => if swapped then fastRmsGemma%[tl.bfloat16,tl.bfloat16,indexCommuted] else fastRmsGemma%[tl.bfloat16,tl.bfloat16,original]
  | .bf16,.fp32 => if swapped then fastRmsGemma%[tl.bfloat16,tl.float32,indexCommuted] else fastRmsGemma%[tl.bfloat16,tl.float32,original]
  | .fp32,.fp16 => if swapped then fastRmsGemma%[tl.float32,tl.float16,indexCommuted] else fastRmsGemma%[tl.float32,tl.float16,original]
  | .fp32,.bf16 => if swapped then fastRmsGemma%[tl.float32,tl.bfloat16,indexCommuted] else fastRmsGemma%[tl.float32,tl.bfloat16,original]
  | .fp32,.fp32 => if swapped then fastRmsGemma%[tl.float32,tl.float32,indexCommuted] else fastRmsGemma%[tl.float32,tl.float32,original]

syntax "fastRmsBackward%" "[" tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(fastRmsBackward%[$inputTy:tritonDType,$weightTy:tritonDType,$version:ident]) => do
    unless version.getId == `original || version.getId == `indexCommuted do
      Macro.throwError "expected original or indexCommuted"
    let rowIndex : TSyntax `tritonStmt ←
      if version.getId == `indexCommuted then `(tritonStmt| X += $(X_row_stride) * row_idx)
      else `(tritonStmt| X += row_idx * $(X_row_stride))
    `(fun (dY X W r dW : RegionName) (dY_row_stride X_row_stride W_row_stride r_row_stride dW_row_stride n_cols BLOCK_SIZE : Nat) (eps : ℝ) (GEMMA : Bool) =>
      ComputeKernel.mk [dY,X,W,r] [dY] (triton {
  tl.region dY = $inputTy, X = $inputTy, W = $weightTy, r = tl.float32, dW = $inputTy
  row_idx = tl.program_id(0)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(n_cols)
  dY += row_idx * $(dY_row_stride)
  $rowIndex
  r += row_idx * $(r_row_stride)
  dY_row = tl.load(dY + col_offsets, mask=mask, other=0).to(tl.float32)
  X_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)
  W_row = tl.load(W + col_offsets, mask=mask, other=0).to(tl.float32)
  inv_var = tl.load(r).to(tl.float32)
  normed = X_row * inv_var
  if $((GEMMA : Bool)) {
    dY_W = dY_row * (W_row + 1.0)
  } else {
    dY_W = dY_row * W_row
  }
  rowsum_dY_normed = tl.sum(dY_W * normed, axis=0)
  output = inv_var / $(n_cols) * ($(n_cols) * dY_W - normed * rowsum_dY_normed)
  tl.store(dY + col_offsets, output.to($inputTy:tritonDType), mask=mask)
}).surfaceBody)
```
</details>

<details><summary><code>output</code></summary>

```lean
def output {B : Nat} (gemma : Bool) (N : Nat) (eps : ℝ) (xs ws : Fin B → ℝ) (j : Fin B) :=
  (xs j*invStd N eps xs)*(if gemma then ws j+1 else ws j)
```
</details>

<details><summary><code>invStd</code></summary>

```lean
def invStd {B : Nat} (N : Nat) (eps : ℝ) (xs : Fin B → ℝ) :=
  1/Real.sqrt (meanSquare N xs+eps)
```
</details>

<details><summary><code>meanSquare</code></summary>

```lean
def meanSquare {B : Nat} (N : Nat) (xs : Fin B → ℝ) := squareSum N xs/(N : ℝ)
```
</details>

<details><summary><code>squareSum</code></summary>

```lean
def squareSum {B : Nat} (N : Nat) (xs : Fin B → ℝ) := ∑ j, (padded N xs j)^2
```
</details>
