# Spec sheet — `bench/tritonbench_g/fast_ce_loss/Correct.lean`

**Python source:** `bench/tritonbench_g/fast_ce_loss/fast_ce_loss.py`

## Public theorem: `forward_correctness`

**Statement:**
```lean
specification forward_correctness (storage : Storage) (cap scale : Bool)
    (X Y Z : RegionName) (L : Region .int) (V R B : Nat) (SC SL : ℝ) :
    Spec.Real (forward_correct_target
      (forward storage cap scale Bool.false X Y Z L V R B SC SL).eraseDType.toAlgKernel
      cap scale X Y Z L V R B SC SL)
```

**Closed-form spec defs (transitive):** `forward_correct_target`, `forward`, `lse`, `loss`, `forwardOutside`, `transform`

<details><summary><code>forward_correct_target</code></summary>

```lean
def forward_correct_target (kernel : Kernel) (cap scale : Bool) (X Y Z : RegionName) (L : Region .int)
    (V R B : Nat) (SC SL : ℝ) : Prop :=
  (cap=Bool.true → 0<SC) → (scale=Bool.true → 0<SL) →
  ∀ A : FlatAlloc, A.Disjoint → (∀ r, r ∉ A.regions → A.extent r=0) →
  ∀ u : BlockState, u.undef=(fun _ _ => 0) → ForwardBounds X Y Z L V R B u A.extent →
  0<V → V≤B → Y≠Z →
    ∃ v, exec (A.flattenKernel kernel) (A.flattenState u)=some v ∧
      v.readMem A.flat (A.addr Z (u.pids 0)) = lse cap scale SC SL V 0
        (fun j : Fin B => u.readMem X (u.pids 0*R+j.val)) ∧
      v.readMem A.flat (A.addr Y (u.pids 0)) = loss cap scale SC SL V
        (u.readMemValue .int (Region.cast L) (u.pids 0))
        (fun j : Fin B => u.readMem X (u.pids 0*R+j.val))
        (u.readMem X (u.pids 0*R+(u.readMemValue .int (Region.cast L) (u.pids 0)).toNat)) ∧
      (∀ r o, forwardOutside A u Y Z r o → v.mem r o=(A.flattenState u).mem r o)
```
</details>

<details><summary><code>forward</code></summary>

```lean
def forward (storage : Storage) (cap scale swapped : Bool) := match storage,cap,scale with
  | .fp16,Bool.false,Bool.false => if swapped then fastCeForward%[tl.float16,noCap,noScale,indexCommuted] else fastCeForward%[tl.float16,noCap,noScale,original]
  | .fp16,Bool.false,Bool.true => if swapped then fastCeForward%[tl.float16,noCap,scale,indexCommuted] else fastCeForward%[tl.float16,noCap,scale,original]
  | .fp16,Bool.true,Bool.false => if swapped then fastCeForward%[tl.float16,cap,noScale,indexCommuted] else fastCeForward%[tl.float16,cap,noScale,original]
  | .fp16,Bool.true,Bool.true => if swapped then fastCeForward%[tl.float16,cap,scale,indexCommuted] else fastCeForward%[tl.float16,cap,scale,original]
  | .bf16,Bool.false,Bool.false => if swapped then fastCeForward%[tl.bfloat16,noCap,noScale,indexCommuted] else fastCeForward%[tl.bfloat16,noCap,noScale,original]
  | .bf16,Bool.false,Bool.true => if swapped then fastCeForward%[tl.bfloat16,noCap,scale,indexCommuted] else fastCeForward%[tl.bfloat16,noCap,scale,original]
  | .bf16,Bool.true,Bool.false => if swapped then fastCeForward%[tl.bfloat16,cap,noScale,indexCommuted] else fastCeForward%[tl.bfloat16,cap,noScale,original]
  | .bf16,Bool.true,Bool.true => if swapped then fastCeForward%[tl.bfloat16,cap,scale,indexCommuted] else fastCeForward%[tl.bfloat16,cap,scale,original]
  | .fp32,Bool.false,Bool.false => if swapped then fastCeForward%[tl.float32,noCap,noScale,indexCommuted] else fastCeForward%[tl.float32,noCap,noScale,original]
  | .fp32,Bool.false,Bool.true => if swapped then fastCeForward%[tl.float32,noCap,scale,indexCommuted] else fastCeForward%[tl.float32,noCap,scale,original]
  | .fp32,Bool.true,Bool.false => if swapped then fastCeForward%[tl.float32,cap,noScale,indexCommuted] else fastCeForward%[tl.float32,cap,noScale,original]
  | .fp32,Bool.true,Bool.true => if swapped then fastCeForward%[tl.float32,cap,scale,indexCommuted] else fastCeForward%[tl.float32,cap,scale,original]

syntax "fastCeChunked%" "[" tritonDType "," ident "," ident "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(fastCeChunked%[$storage:tritonDType,$capFlag:ident,$scaleFlag:ident,$version:ident]) => do
    unless version.getId == `original || version.getId == `indexCommuted do
      Macro.throwError "expected original or indexCommuted"
    let rowIndex : TSyntax `tritonStmt ←
      if version.getId == `indexCommuted then `(tritonStmt| logits_ptr += ($(logits_row_stride)).to(tl.int64) * row_idx)
      else `(tritonStmt| logits_ptr += row_idx * ($(logits_row_stride)).to(tl.int64))
    `(fun (logits_ptr loss_ptr logsumexp_ptr : RegionName) (labels_ptr : Region .int)
        (VOCAB_SIZE N_CHUNKS logits_row_stride BLOCK_SIZE : Nat) (SOFTCAP LOGIT_SCALE : ℝ)
        => ComputeKernel.mk
      [logits_ptr,Region.cast labels_ptr] [loss_ptr,logsumexp_ptr] (fastCeStage%[$capFlag,$scaleFlag] triton {
  tl.region labels_ptr = tl.int32, logits_ptr = $storage, loss_ptr = tl.float32, logsumexp_ptr = tl.float32
  row_idx = tl.program_id(0)
  chunk_idx = tl.program_id(1)
  $rowIndex
  loss_ptr += row_idx
  logsumexp_ptr += row_idx * $(N_CHUNKS) + chunk_idx
  labels_ptr += row_idx
  col_offsets = chunk_idx * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(VOCAB_SIZE)
  label_idx = tl.load(labels_ptr).to(tl.int32)
  logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))
  if DO_LOGIT_SCALING {
    logits = $((LOGIT_SCALE : ℝ)) * logits
  }
  if DO_SOFTCAPPING {
    logits = $((SOFTCAP : ℝ)) * triton_tanh(logits / $((SOFTCAP : ℝ)))
  }
  logits = logits.to(tl.float32)
  c = tl.max(logits, 0)
  logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), 0))
  if chunk_idx == $(0) {
    if label_idx != $((-100 : Int)) {
      x = tl.load(logits_ptr + label_idx).to(tl.float32)
      if DO_LOGIT_SCALING {
        x = $((LOGIT_SCALE : ℝ)) * x
      }
      if DO_SOFTCAPPING {
        x = $((SOFTCAP : ℝ)) * triton_tanh(x / $((SOFTCAP : ℝ)))
      }
      loss = -1.0 * x.to(tl.float32)
    } else {
      loss = 0.0
    }
    tl.store(loss_ptr, loss.to(tl.float32))
    tl.store(logsumexp_ptr, logsumexp.to(tl.float32))
  }
}).surfaceBody)
```
</details>

<details><summary><code>lse</code></summary>

```
/-- A padded softcap lane contributes exp(-SC); an uncapped masked lane contributes zero. -/
```
```lean
def lse {B : Nat} (cap scale : Bool) (SC SL : ℝ) (V start : Nat) (xs : Fin B → ℝ) : ℝ :=
  Real.log (∑ j, if start+j.val<V then Real.exp (transform cap scale SC SL (xs j))
    else if cap then Real.exp (-SC) else 0)
```
</details>

<details><summary><code>loss</code></summary>

```lean
def loss {B : Nat} (cap scale : Bool) (SC SL : ℝ) (V : Nat)
    (lab : Int) (xs : Fin B → ℝ) (gather : ℝ) : ℝ :=
  if lab = -100 then 0 else lse cap scale SC SL V 0 xs-transform cap scale SC SL gather
```
</details>

<details><summary><code>forwardOutside</code></summary>

```lean
def forwardOutside (A : FlatAlloc) (u : BlockState) (Y Z : RegionName)
    (r : RegionName) (o : Nat) : Prop :=
  r≠A.flat ∨ (o≠A.addr Y (u.pids 0) ∧ o≠A.addr Z (u.pids 0))
```
</details>

<details><summary><code>transform</code></summary>

```
/-! Pure formulas use initial values, independently of execution. `lse`
includes capped padded lanes; `partialLoss` is only the chunk-zero partial
term. `gradient` uses the caller-supplied LSE, without claiming a host
forward/backward composition or a calculus theorem. -/
```
```lean
def transform (cap scale : Bool) (SC SL x : ℝ) : ℝ :=
  let x := if scale then SL*x else x
  if cap then SC*Real.tanh (x/SC) else x
```
</details>

## Public theorem: `forward_index_commuted_correctness`

**Statement:**
```lean
specification forward_index_commuted_correctness (storage : Storage) (cap scale : Bool)
    (X Y Z : RegionName) (L : Region .int) (V R B : Nat) (SC SL : ℝ) :
    Spec.Real (forward_correct_target
      (forward storage cap scale Bool.true X Y Z L V R B SC SL).eraseDType.toAlgKernel
      cap scale X Y Z L V R B SC SL)
```

**Closed-form spec defs (transitive):** `forward_correct_target`, `forward`, `lse`, `loss`, `forwardOutside`, `transform`

<details><summary><code>forward_correct_target</code></summary>

```lean
def forward_correct_target (kernel : Kernel) (cap scale : Bool) (X Y Z : RegionName) (L : Region .int)
    (V R B : Nat) (SC SL : ℝ) : Prop :=
  (cap=Bool.true → 0<SC) → (scale=Bool.true → 0<SL) →
  ∀ A : FlatAlloc, A.Disjoint → (∀ r, r ∉ A.regions → A.extent r=0) →
  ∀ u : BlockState, u.undef=(fun _ _ => 0) → ForwardBounds X Y Z L V R B u A.extent →
  0<V → V≤B → Y≠Z →
    ∃ v, exec (A.flattenKernel kernel) (A.flattenState u)=some v ∧
      v.readMem A.flat (A.addr Z (u.pids 0)) = lse cap scale SC SL V 0
        (fun j : Fin B => u.readMem X (u.pids 0*R+j.val)) ∧
      v.readMem A.flat (A.addr Y (u.pids 0)) = loss cap scale SC SL V
        (u.readMemValue .int (Region.cast L) (u.pids 0))
        (fun j : Fin B => u.readMem X (u.pids 0*R+j.val))
        (u.readMem X (u.pids 0*R+(u.readMemValue .int (Region.cast L) (u.pids 0)).toNat)) ∧
      (∀ r o, forwardOutside A u Y Z r o → v.mem r o=(A.flattenState u).mem r o)
```
</details>

<details><summary><code>forward</code></summary>

```lean
def forward (storage : Storage) (cap scale swapped : Bool) := match storage,cap,scale with
  | .fp16,Bool.false,Bool.false => if swapped then fastCeForward%[tl.float16,noCap,noScale,indexCommuted] else fastCeForward%[tl.float16,noCap,noScale,original]
  | .fp16,Bool.false,Bool.true => if swapped then fastCeForward%[tl.float16,noCap,scale,indexCommuted] else fastCeForward%[tl.float16,noCap,scale,original]
  | .fp16,Bool.true,Bool.false => if swapped then fastCeForward%[tl.float16,cap,noScale,indexCommuted] else fastCeForward%[tl.float16,cap,noScale,original]
  | .fp16,Bool.true,Bool.true => if swapped then fastCeForward%[tl.float16,cap,scale,indexCommuted] else fastCeForward%[tl.float16,cap,scale,original]
  | .bf16,Bool.false,Bool.false => if swapped then fastCeForward%[tl.bfloat16,noCap,noScale,indexCommuted] else fastCeForward%[tl.bfloat16,noCap,noScale,original]
  | .bf16,Bool.false,Bool.true => if swapped then fastCeForward%[tl.bfloat16,noCap,scale,indexCommuted] else fastCeForward%[tl.bfloat16,noCap,scale,original]
  | .bf16,Bool.true,Bool.false => if swapped then fastCeForward%[tl.bfloat16,cap,noScale,indexCommuted] else fastCeForward%[tl.bfloat16,cap,noScale,original]
  | .bf16,Bool.true,Bool.true => if swapped then fastCeForward%[tl.bfloat16,cap,scale,indexCommuted] else fastCeForward%[tl.bfloat16,cap,scale,original]
  | .fp32,Bool.false,Bool.false => if swapped then fastCeForward%[tl.float32,noCap,noScale,indexCommuted] else fastCeForward%[tl.float32,noCap,noScale,original]
  | .fp32,Bool.false,Bool.true => if swapped then fastCeForward%[tl.float32,noCap,scale,indexCommuted] else fastCeForward%[tl.float32,noCap,scale,original]
  | .fp32,Bool.true,Bool.false => if swapped then fastCeForward%[tl.float32,cap,noScale,indexCommuted] else fastCeForward%[tl.float32,cap,noScale,original]
  | .fp32,Bool.true,Bool.true => if swapped then fastCeForward%[tl.float32,cap,scale,indexCommuted] else fastCeForward%[tl.float32,cap,scale,original]

syntax "fastCeChunked%" "[" tritonDType "," ident "," ident "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(fastCeChunked%[$storage:tritonDType,$capFlag:ident,$scaleFlag:ident,$version:ident]) => do
    unless version.getId == `original || version.getId == `indexCommuted do
      Macro.throwError "expected original or indexCommuted"
    let rowIndex : TSyntax `tritonStmt ←
      if version.getId == `indexCommuted then `(tritonStmt| logits_ptr += ($(logits_row_stride)).to(tl.int64) * row_idx)
      else `(tritonStmt| logits_ptr += row_idx * ($(logits_row_stride)).to(tl.int64))
    `(fun (logits_ptr loss_ptr logsumexp_ptr : RegionName) (labels_ptr : Region .int)
        (VOCAB_SIZE N_CHUNKS logits_row_stride BLOCK_SIZE : Nat) (SOFTCAP LOGIT_SCALE : ℝ)
        => ComputeKernel.mk
      [logits_ptr,Region.cast labels_ptr] [loss_ptr,logsumexp_ptr] (fastCeStage%[$capFlag,$scaleFlag] triton {
  tl.region labels_ptr = tl.int32, logits_ptr = $storage, loss_ptr = tl.float32, logsumexp_ptr = tl.float32
  row_idx = tl.program_id(0)
  chunk_idx = tl.program_id(1)
  $rowIndex
  loss_ptr += row_idx
  logsumexp_ptr += row_idx * $(N_CHUNKS) + chunk_idx
  labels_ptr += row_idx
  col_offsets = chunk_idx * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(VOCAB_SIZE)
  label_idx = tl.load(labels_ptr).to(tl.int32)
  logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))
  if DO_LOGIT_SCALING {
    logits = $((LOGIT_SCALE : ℝ)) * logits
  }
  if DO_SOFTCAPPING {
    logits = $((SOFTCAP : ℝ)) * triton_tanh(logits / $((SOFTCAP : ℝ)))
  }
  logits = logits.to(tl.float32)
  c = tl.max(logits, 0)
  logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), 0))
  if chunk_idx == $(0) {
    if label_idx != $((-100 : Int)) {
      x = tl.load(logits_ptr + label_idx).to(tl.float32)
      if DO_LOGIT_SCALING {
        x = $((LOGIT_SCALE : ℝ)) * x
      }
      if DO_SOFTCAPPING {
        x = $((SOFTCAP : ℝ)) * triton_tanh(x / $((SOFTCAP : ℝ)))
      }
      loss = -1.0 * x.to(tl.float32)
    } else {
      loss = 0.0
    }
    tl.store(loss_ptr, loss.to(tl.float32))
    tl.store(logsumexp_ptr, logsumexp.to(tl.float32))
  }
}).surfaceBody)
```
</details>

<details><summary><code>lse</code></summary>

```
/-- A padded softcap lane contributes exp(-SC); an uncapped masked lane contributes zero. -/
```
```lean
def lse {B : Nat} (cap scale : Bool) (SC SL : ℝ) (V start : Nat) (xs : Fin B → ℝ) : ℝ :=
  Real.log (∑ j, if start+j.val<V then Real.exp (transform cap scale SC SL (xs j))
    else if cap then Real.exp (-SC) else 0)
```
</details>

<details><summary><code>loss</code></summary>

```lean
def loss {B : Nat} (cap scale : Bool) (SC SL : ℝ) (V : Nat)
    (lab : Int) (xs : Fin B → ℝ) (gather : ℝ) : ℝ :=
  if lab = -100 then 0 else lse cap scale SC SL V 0 xs-transform cap scale SC SL gather
```
</details>

<details><summary><code>forwardOutside</code></summary>

```lean
def forwardOutside (A : FlatAlloc) (u : BlockState) (Y Z : RegionName)
    (r : RegionName) (o : Nat) : Prop :=
  r≠A.flat ∨ (o≠A.addr Y (u.pids 0) ∧ o≠A.addr Z (u.pids 0))
```
</details>

<details><summary><code>transform</code></summary>

```
/-! Pure formulas use initial values, independently of execution. `lse`
includes capped padded lanes; `partialLoss` is only the chunk-zero partial
term. `gradient` uses the caller-supplied LSE, without claiming a host
forward/backward composition or a calculus theorem. -/
```
```lean
def transform (cap scale : Bool) (SC SL x : ℝ) : ℝ :=
  let x := if scale then SL*x else x
  if cap then SC*Real.tanh (x/SC) else x
```
</details>

## Public theorem: `chunked_correctness`

**Statement:**
```lean
specification chunked_correctness (storage : Storage) (cap scale : Bool)
    (X Y Z : RegionName) (L : Region .int) (V C R B : Nat) (SC SL : ℝ) :
    Spec.Real (chunked_correct_target
      (chunked storage cap scale Bool.false X Y Z L V C R B SC SL).eraseDType.toAlgKernel
      cap scale X Y Z L V C R B SC SL)
```

**Closed-form spec defs (transitive):** `chunked_correct_target`, `chunked`, `chunkedOutputs`, `lse`, `partialLoss`, `chunkedOutside`, `transform`, `loss`

<details><summary><code>chunked_correct_target</code></summary>

```
/-- Conditional source behavior: only chunk zero writes its partial loss and LSE. -/
```
```lean
def chunked_correct_target (kernel : Kernel) (cap scale : Bool) (X Y Z : RegionName) (L : Region .int)
    (V C R B : Nat) (SC SL : ℝ) : Prop :=
  (cap=Bool.true → 0<SC) → (scale=Bool.true → 0<SL) →
  ∀ A : FlatAlloc, A.Disjoint → (∀ r, r ∉ A.regions → A.extent r=0) →
  ∀ u : BlockState, u.undef=(fun _ _ => 0) → ChunkedBounds X Y Z L V C R B u A.extent →
  0<V → 0<B → Y≠Z →
    ∃ v, exec (A.flattenKernel kernel) (A.flattenState u)=some v ∧
      chunkedOutputs A u v cap scale X Y Z L V C R B SC SL
```
</details>

<details><summary><code>chunked</code></summary>

```lean
def chunked (storage : Storage) (cap scale swapped : Bool) := match storage,cap,scale with
  | .fp16,Bool.false,Bool.false => if swapped then fastCeChunked%[tl.float16,noCap,noScale,indexCommuted] else fastCeChunked%[tl.float16,noCap,noScale,original]
  | .fp16,Bool.false,Bool.true => if swapped then fastCeChunked%[tl.float16,noCap,scale,indexCommuted] else fastCeChunked%[tl.float16,noCap,scale,original]
  | .fp16,Bool.true,Bool.false => if swapped then fastCeChunked%[tl.float16,cap,noScale,indexCommuted] else fastCeChunked%[tl.float16,cap,noScale,original]
  | .fp16,Bool.true,Bool.true => if swapped then fastCeChunked%[tl.float16,cap,scale,indexCommuted] else fastCeChunked%[tl.float16,cap,scale,original]
  | .bf16,Bool.false,Bool.false => if swapped then fastCeChunked%[tl.bfloat16,noCap,noScale,indexCommuted] else fastCeChunked%[tl.bfloat16,noCap,noScale,original]
  | .bf16,Bool.false,Bool.true => if swapped then fastCeChunked%[tl.bfloat16,noCap,scale,indexCommuted] else fastCeChunked%[tl.bfloat16,noCap,scale,original]
  | .bf16,Bool.true,Bool.false => if swapped then fastCeChunked%[tl.bfloat16,cap,noScale,indexCommuted] else fastCeChunked%[tl.bfloat16,cap,noScale,original]
  | .bf16,Bool.true,Bool.true => if swapped then fastCeChunked%[tl.bfloat16,cap,scale,indexCommuted] else fastCeChunked%[tl.bfloat16,cap,scale,original]
  | .fp32,Bool.false,Bool.false => if swapped then fastCeChunked%[tl.float32,noCap,noScale,indexCommuted] else fastCeChunked%[tl.float32,noCap,noScale,original]
  | .fp32,Bool.false,Bool.true => if swapped then fastCeChunked%[tl.float32,noCap,scale,indexCommuted] else fastCeChunked%[tl.float32,noCap,scale,original]
  | .fp32,Bool.true,Bool.false => if swapped then fastCeChunked%[tl.float32,cap,noScale,indexCommuted] else fastCeChunked%[tl.float32,cap,noScale,original]
  | .fp32,Bool.true,Bool.true => if swapped then fastCeChunked%[tl.float32,cap,scale,indexCommuted] else fastCeChunked%[tl.float32,cap,scale,original]

syntax "fastCeBackward%" "[" tritonDType "," ident "," ident "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(fastCeBackward%[$storage:tritonDType,$capFlag:ident,$scaleFlag:ident,$version:ident]) => do
    unless version.getId == `original || version.getId == `indexCommuted do
      Macro.throwError "expected original or indexCommuted"
    let rowIndex : TSyntax `tritonStmt ←
      if version.getId == `indexCommuted then `(tritonStmt| logits_ptr += ($(logits_row_stride)).to(tl.int64) * row_idx)
      else `(tritonStmt| logits_ptr += row_idx * ($(logits_row_stride)).to(tl.int64))
    `(fun (logits_ptr dloss_ptr logsumexp_ptr : RegionName) (labels_ptr : Region .int)
        (VOCAB_SIZE logits_row_stride dloss_row_stride BLOCK_SIZE : Nat) (SOFTCAP LOGIT_SCALE : ℝ)
        => ComputeKernel.mk
      [logits_ptr,dloss_ptr,logsumexp_ptr,Region.cast labels_ptr] [logits_ptr] (fastCeStage%[$capFlag,$scaleFlag] triton {
  tl.region labels_ptr = tl.int32, logits_ptr = $storage, dloss_ptr = tl.float32, logsumexp_ptr = tl.float32
  row_idx = tl.program_id(0)
  block_idx = tl.program_id(1)
  $rowIndex
  dloss_ptr += row_idx * $(dloss_row_stride)
  col_offsets = block_idx * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(VOCAB_SIZE)
  label_idx = tl.load(labels_ptr + row_idx).to(tl.int32)
  if label_idx != $((-100 : Int)) {
    dloss = tl.load(dloss_ptr)
  }
  else {
    dloss = 0.0
  }
  x = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))
  if DO_LOGIT_SCALING {
    x = x * $((LOGIT_SCALE : ℝ))
  }
  if DO_SOFTCAPPING {
    «partial» = triton_tanh(x / $((SOFTCAP : ℝ)))
    x = $((SOFTCAP : ℝ)) * «partial»
  }
  logsumexp = tl.load(logsumexp_ptr + row_idx)
  y = tl.exp(x.to(tl.float32) - logsumexp)
  y = tl.where(col_offsets == label_idx, y - 1.0, y)
  if DO_LOGIT_SCALING {
    y = y * $((LOGIT_SCALE : ℝ))
  }
  if DO_SOFTCAPPING {
    y = y * (1.0 - «partial» * «partial»)
  }
  tl.store(logits_ptr + col_offsets, (dloss * y).to($storage:tritonDType), mask=mask)
}).surfaceBody)
```
</details>

<details><summary><code>chunkedOutputs</code></summary>

```lean
def chunkedOutputs (A : FlatAlloc) (u v : BlockState) (cap scale : Bool)
    (X Y Z : RegionName) (L : Region .int) (V C R B : Nat) (SC SL : ℝ) : Prop :=
  if u.pids 1=0 then
    v.readMem A.flat (A.addr Z (u.pids 0*C)) = lse cap scale SC SL V 0
      (fun j : Fin B => u.readMem X (u.pids 0*R+j.val)) ∧
    v.readMem A.flat (A.addr Y (u.pids 0)) = partialLoss cap scale SC SL
      (u.readMemValue .int (Region.cast L) (u.pids 0))
      (u.readMem X (u.pids 0*R+(u.readMemValue .int (Region.cast L) (u.pids 0)).toNat)) ∧
    (∀ r o, chunkedOutside A u Y Z C r o → v.mem r o=(A.flattenState u).mem r o)
  else v.mem=(A.flattenState u).mem
```
</details>

<details><summary><code>lse</code></summary>

```
/-- A padded softcap lane contributes exp(-SC); an uncapped masked lane contributes zero. -/
```
```lean
def lse {B : Nat} (cap scale : Bool) (SC SL : ℝ) (V start : Nat) (xs : Fin B → ℝ) : ℝ :=
  Real.log (∑ j, if start+j.val<V then Real.exp (transform cap scale SC SL (xs j))
    else if cap then Real.exp (-SC) else 0)
```
</details>

<details><summary><code>partialLoss</code></summary>

```lean
def partialLoss (cap scale : Bool) (SC SL : ℝ) (lab : Int) (gather : ℝ) : ℝ :=
  if lab = -100 then 0 else -transform cap scale SC SL gather

structure ForwardBounds (X Y Z : RegionName) (L : Region .int) (V R B : Nat)
    (u : BlockState) (bounds : RegionBounds) : Prop where
  label : u.pids 0 < bounds (Region.cast L)
  labelTyped : ∃ lab : Int, u.mem (Region.cast L) (u.pids 0)=MemCell.of .int lab
  labelValid : let lab := u.readMemValue .int (Region.cast L) (u.pids 0)
    lab = -100 ∨ (0≤lab ∧ lab < (V : Int))
  loss : u.pids 0 < bounds Y
  lse : u.pids 0 < bounds Z
  gather : u.readMemValue .int (Region.cast L) (u.pids 0) ≠ -100 →
    u.pids 0*R+(u.readMemValue .int (Region.cast L) (u.pids 0)).toNat < bounds X
  logits : ∀ j : Fin B, j.val<V → u.pids 0*R+j.val < bounds X
```
</details>

<details><summary><code>chunkedOutside</code></summary>

```lean
def chunkedOutside (A : FlatAlloc) (u : BlockState) (Y Z : RegionName) (C : Nat)
    (r : RegionName) (o : Nat) : Prop :=
  r≠A.flat ∨ (o≠A.addr Y (u.pids 0) ∧ o≠A.addr Z (u.pids 0*C))
```
</details>

<details><summary><code>transform</code></summary>

```
/-! Pure formulas use initial values, independently of execution. `lse`
includes capped padded lanes; `partialLoss` is only the chunk-zero partial
term. `gradient` uses the caller-supplied LSE, without claiming a host
forward/backward composition or a calculus theorem. -/
```
```lean
def transform (cap scale : Bool) (SC SL x : ℝ) : ℝ :=
  let x := if scale then SL*x else x
  if cap then SC*Real.tanh (x/SC) else x
```
</details>

<details><summary><code>loss</code></summary>

```lean
def loss {B : Nat} (cap scale : Bool) (SC SL : ℝ) (V : Nat)
    (lab : Int) (xs : Fin B → ℝ) (gather : ℝ) : ℝ :=
  if lab = -100 then 0 else lse cap scale SC SL V 0 xs-transform cap scale SC SL gather
```
</details>

## Public theorem: `chunked_index_commuted_correctness`

**Statement:**
```lean
specification chunked_index_commuted_correctness (storage : Storage) (cap scale : Bool)
    (X Y Z : RegionName) (L : Region .int) (V C R B : Nat) (SC SL : ℝ) :
    Spec.Real (chunked_correct_target
      (chunked storage cap scale Bool.true X Y Z L V C R B SC SL).eraseDType.toAlgKernel
      cap scale X Y Z L V C R B SC SL)
```

**Closed-form spec defs (transitive):** `chunked_correct_target`, `chunked`, `chunkedOutputs`, `lse`, `partialLoss`, `chunkedOutside`, `transform`, `loss`

<details><summary><code>chunked_correct_target</code></summary>

```
/-- Conditional source behavior: only chunk zero writes its partial loss and LSE. -/
```
```lean
def chunked_correct_target (kernel : Kernel) (cap scale : Bool) (X Y Z : RegionName) (L : Region .int)
    (V C R B : Nat) (SC SL : ℝ) : Prop :=
  (cap=Bool.true → 0<SC) → (scale=Bool.true → 0<SL) →
  ∀ A : FlatAlloc, A.Disjoint → (∀ r, r ∉ A.regions → A.extent r=0) →
  ∀ u : BlockState, u.undef=(fun _ _ => 0) → ChunkedBounds X Y Z L V C R B u A.extent →
  0<V → 0<B → Y≠Z →
    ∃ v, exec (A.flattenKernel kernel) (A.flattenState u)=some v ∧
      chunkedOutputs A u v cap scale X Y Z L V C R B SC SL
```
</details>

<details><summary><code>chunked</code></summary>

```lean
def chunked (storage : Storage) (cap scale swapped : Bool) := match storage,cap,scale with
  | .fp16,Bool.false,Bool.false => if swapped then fastCeChunked%[tl.float16,noCap,noScale,indexCommuted] else fastCeChunked%[tl.float16,noCap,noScale,original]
  | .fp16,Bool.false,Bool.true => if swapped then fastCeChunked%[tl.float16,noCap,scale,indexCommuted] else fastCeChunked%[tl.float16,noCap,scale,original]
  | .fp16,Bool.true,Bool.false => if swapped then fastCeChunked%[tl.float16,cap,noScale,indexCommuted] else fastCeChunked%[tl.float16,cap,noScale,original]
  | .fp16,Bool.true,Bool.true => if swapped then fastCeChunked%[tl.float16,cap,scale,indexCommuted] else fastCeChunked%[tl.float16,cap,scale,original]
  | .bf16,Bool.false,Bool.false => if swapped then fastCeChunked%[tl.bfloat16,noCap,noScale,indexCommuted] else fastCeChunked%[tl.bfloat16,noCap,noScale,original]
  | .bf16,Bool.false,Bool.true => if swapped then fastCeChunked%[tl.bfloat16,noCap,scale,indexCommuted] else fastCeChunked%[tl.bfloat16,noCap,scale,original]
  | .bf16,Bool.true,Bool.false => if swapped then fastCeChunked%[tl.bfloat16,cap,noScale,indexCommuted] else fastCeChunked%[tl.bfloat16,cap,noScale,original]
  | .bf16,Bool.true,Bool.true => if swapped then fastCeChunked%[tl.bfloat16,cap,scale,indexCommuted] else fastCeChunked%[tl.bfloat16,cap,scale,original]
  | .fp32,Bool.false,Bool.false => if swapped then fastCeChunked%[tl.float32,noCap,noScale,indexCommuted] else fastCeChunked%[tl.float32,noCap,noScale,original]
  | .fp32,Bool.false,Bool.true => if swapped then fastCeChunked%[tl.float32,noCap,scale,indexCommuted] else fastCeChunked%[tl.float32,noCap,scale,original]
  | .fp32,Bool.true,Bool.false => if swapped then fastCeChunked%[tl.float32,cap,noScale,indexCommuted] else fastCeChunked%[tl.float32,cap,noScale,original]
  | .fp32,Bool.true,Bool.true => if swapped then fastCeChunked%[tl.float32,cap,scale,indexCommuted] else fastCeChunked%[tl.float32,cap,scale,original]

syntax "fastCeBackward%" "[" tritonDType "," ident "," ident "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(fastCeBackward%[$storage:tritonDType,$capFlag:ident,$scaleFlag:ident,$version:ident]) => do
    unless version.getId == `original || version.getId == `indexCommuted do
      Macro.throwError "expected original or indexCommuted"
    let rowIndex : TSyntax `tritonStmt ←
      if version.getId == `indexCommuted then `(tritonStmt| logits_ptr += ($(logits_row_stride)).to(tl.int64) * row_idx)
      else `(tritonStmt| logits_ptr += row_idx * ($(logits_row_stride)).to(tl.int64))
    `(fun (logits_ptr dloss_ptr logsumexp_ptr : RegionName) (labels_ptr : Region .int)
        (VOCAB_SIZE logits_row_stride dloss_row_stride BLOCK_SIZE : Nat) (SOFTCAP LOGIT_SCALE : ℝ)
        => ComputeKernel.mk
      [logits_ptr,dloss_ptr,logsumexp_ptr,Region.cast labels_ptr] [logits_ptr] (fastCeStage%[$capFlag,$scaleFlag] triton {
  tl.region labels_ptr = tl.int32, logits_ptr = $storage, dloss_ptr = tl.float32, logsumexp_ptr = tl.float32
  row_idx = tl.program_id(0)
  block_idx = tl.program_id(1)
  $rowIndex
  dloss_ptr += row_idx * $(dloss_row_stride)
  col_offsets = block_idx * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(VOCAB_SIZE)
  label_idx = tl.load(labels_ptr + row_idx).to(tl.int32)
  if label_idx != $((-100 : Int)) {
    dloss = tl.load(dloss_ptr)
  }
  else {
    dloss = 0.0
  }
  x = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))
  if DO_LOGIT_SCALING {
    x = x * $((LOGIT_SCALE : ℝ))
  }
  if DO_SOFTCAPPING {
    «partial» = triton_tanh(x / $((SOFTCAP : ℝ)))
    x = $((SOFTCAP : ℝ)) * «partial»
  }
  logsumexp = tl.load(logsumexp_ptr + row_idx)
  y = tl.exp(x.to(tl.float32) - logsumexp)
  y = tl.where(col_offsets == label_idx, y - 1.0, y)
  if DO_LOGIT_SCALING {
    y = y * $((LOGIT_SCALE : ℝ))
  }
  if DO_SOFTCAPPING {
    y = y * (1.0 - «partial» * «partial»)
  }
  tl.store(logits_ptr + col_offsets, (dloss * y).to($storage:tritonDType), mask=mask)
}).surfaceBody)
```
</details>

<details><summary><code>chunkedOutputs</code></summary>

```lean
def chunkedOutputs (A : FlatAlloc) (u v : BlockState) (cap scale : Bool)
    (X Y Z : RegionName) (L : Region .int) (V C R B : Nat) (SC SL : ℝ) : Prop :=
  if u.pids 1=0 then
    v.readMem A.flat (A.addr Z (u.pids 0*C)) = lse cap scale SC SL V 0
      (fun j : Fin B => u.readMem X (u.pids 0*R+j.val)) ∧
    v.readMem A.flat (A.addr Y (u.pids 0)) = partialLoss cap scale SC SL
      (u.readMemValue .int (Region.cast L) (u.pids 0))
      (u.readMem X (u.pids 0*R+(u.readMemValue .int (Region.cast L) (u.pids 0)).toNat)) ∧
    (∀ r o, chunkedOutside A u Y Z C r o → v.mem r o=(A.flattenState u).mem r o)
  else v.mem=(A.flattenState u).mem
```
</details>

<details><summary><code>lse</code></summary>

```
/-- A padded softcap lane contributes exp(-SC); an uncapped masked lane contributes zero. -/
```
```lean
def lse {B : Nat} (cap scale : Bool) (SC SL : ℝ) (V start : Nat) (xs : Fin B → ℝ) : ℝ :=
  Real.log (∑ j, if start+j.val<V then Real.exp (transform cap scale SC SL (xs j))
    else if cap then Real.exp (-SC) else 0)
```
</details>

<details><summary><code>partialLoss</code></summary>

```lean
def partialLoss (cap scale : Bool) (SC SL : ℝ) (lab : Int) (gather : ℝ) : ℝ :=
  if lab = -100 then 0 else -transform cap scale SC SL gather

structure ForwardBounds (X Y Z : RegionName) (L : Region .int) (V R B : Nat)
    (u : BlockState) (bounds : RegionBounds) : Prop where
  label : u.pids 0 < bounds (Region.cast L)
  labelTyped : ∃ lab : Int, u.mem (Region.cast L) (u.pids 0)=MemCell.of .int lab
  labelValid : let lab := u.readMemValue .int (Region.cast L) (u.pids 0)
    lab = -100 ∨ (0≤lab ∧ lab < (V : Int))
  loss : u.pids 0 < bounds Y
  lse : u.pids 0 < bounds Z
  gather : u.readMemValue .int (Region.cast L) (u.pids 0) ≠ -100 →
    u.pids 0*R+(u.readMemValue .int (Region.cast L) (u.pids 0)).toNat < bounds X
  logits : ∀ j : Fin B, j.val<V → u.pids 0*R+j.val < bounds X
```
</details>

<details><summary><code>chunkedOutside</code></summary>

```lean
def chunkedOutside (A : FlatAlloc) (u : BlockState) (Y Z : RegionName) (C : Nat)
    (r : RegionName) (o : Nat) : Prop :=
  r≠A.flat ∨ (o≠A.addr Y (u.pids 0) ∧ o≠A.addr Z (u.pids 0*C))
```
</details>

<details><summary><code>transform</code></summary>

```
/-! Pure formulas use initial values, independently of execution. `lse`
includes capped padded lanes; `partialLoss` is only the chunk-zero partial
term. `gradient` uses the caller-supplied LSE, without claiming a host
forward/backward composition or a calculus theorem. -/
```
```lean
def transform (cap scale : Bool) (SC SL x : ℝ) : ℝ :=
  let x := if scale then SL*x else x
  if cap then SC*Real.tanh (x/SC) else x
```
</details>

<details><summary><code>loss</code></summary>

```lean
def loss {B : Nat} (cap scale : Bool) (SC SL : ℝ) (V : Nat)
    (lab : Int) (xs : Fin B → ℝ) (gather : ℝ) : ℝ :=
  if lab = -100 then 0 else lse cap scale SC SL V 0 xs-transform cap scale SC SL gather
```
</details>

## Public theorem: `backward_correctness`

**Statement:**
```lean
specification backward_correctness (storage : Storage) (cap scale : Bool)
    (X D Z : RegionName) (L : Region .int) (V R DR B : Nat) (SC SL : ℝ) :
    Spec.Real (backward_correct_target
      (backward storage cap scale Bool.false X D Z L V R DR B SC SL).eraseDType.toAlgKernel
      cap scale X D Z L V R DR B SC SL)
```

**Closed-form spec defs (transitive):** `backward_correct_target`, `backward`, `col`, `address`, `gradient`, `backwardOutside`, `lse`, `transform`

<details><summary><code>backward_correct_target</code></summary>

```lean
def backward_correct_target (kernel : Kernel) (cap scale : Bool) (X D Z : RegionName) (L : Region .int)
    (V R DR B : Nat) (SC SL : ℝ) : Prop :=
  (cap=Bool.true → SC≠0) →
  ∀ A : FlatAlloc, A.Disjoint → (∀ r, r ∉ A.regions → A.extent r=0) →
  ∀ u : BlockState, u.undef=(fun _ _ => 0) → BackwardBounds X D Z L V R DR B u A.extent →
    ∃ v, exec (A.flattenKernel kernel) (A.flattenState u)=some v ∧
      (∀ j : Fin B, col u B j<V → v.readMem A.flat (A.addr X (address u R B j)) =
        gradient cap scale SC SL (u.readMemValue .int (Region.cast L) (u.pids 0))
          (col u B j) (u.readMem X (address u R B j))
          (u.readMem Z (u.pids 0)) (u.readMem D (u.pids 0*DR))) ∧
      (∀ r o, backwardOutside A u X V R B r o → v.mem r o=(A.flattenState u).mem r o)
```
</details>

<details><summary><code>backward</code></summary>

```lean
def backward (storage : Storage) (cap scale swapped : Bool) := match storage,cap,scale with
  | .fp16,Bool.false,Bool.false => if swapped then fastCeBackward%[tl.float16,noCap,noScale,indexCommuted] else fastCeBackward%[tl.float16,noCap,noScale,original]
  | .fp16,Bool.false,Bool.true => if swapped then fastCeBackward%[tl.float16,noCap,scale,indexCommuted] else fastCeBackward%[tl.float16,noCap,scale,original]
  | .fp16,Bool.true,Bool.false => if swapped then fastCeBackward%[tl.float16,cap,noScale,indexCommuted] else fastCeBackward%[tl.float16,cap,noScale,original]
  | .fp16,Bool.true,Bool.true => if swapped then fastCeBackward%[tl.float16,cap,scale,indexCommuted] else fastCeBackward%[tl.float16,cap,scale,original]
  | .bf16,Bool.false,Bool.false => if swapped then fastCeBackward%[tl.bfloat16,noCap,noScale,indexCommuted] else fastCeBackward%[tl.bfloat16,noCap,noScale,original]
  | .bf16,Bool.false,Bool.true => if swapped then fastCeBackward%[tl.bfloat16,noCap,scale,indexCommuted] else fastCeBackward%[tl.bfloat16,noCap,scale,original]
  | .bf16,Bool.true,Bool.false => if swapped then fastCeBackward%[tl.bfloat16,cap,noScale,indexCommuted] else fastCeBackward%[tl.bfloat16,cap,noScale,original]
  | .bf16,Bool.true,Bool.true => if swapped then fastCeBackward%[tl.bfloat16,cap,scale,indexCommuted] else fastCeBackward%[tl.bfloat16,cap,scale,original]
  | .fp32,Bool.false,Bool.false => if swapped then fastCeBackward%[tl.float32,noCap,noScale,indexCommuted] else fastCeBackward%[tl.float32,noCap,noScale,original]
  | .fp32,Bool.false,Bool.true => if swapped then fastCeBackward%[tl.float32,noCap,scale,indexCommuted] else fastCeBackward%[tl.float32,noCap,scale,original]
  | .fp32,Bool.true,Bool.false => if swapped then fastCeBackward%[tl.float32,cap,noScale,indexCommuted] else fastCeBackward%[tl.float32,cap,noScale,original]
  | .fp32,Bool.true,Bool.true => if swapped then fastCeBackward%[tl.float32,cap,scale,indexCommuted] else fastCeBackward%[tl.float32,cap,scale,original]
```
</details>

<details><summary><code>col</code></summary>

```lean
def col (u : BlockState) (B : Nat) (j : Fin B) := u.pids 1*B+j.val
```
</details>

<details><summary><code>address</code></summary>

```lean
def address (u : BlockState) (R B : Nat) (j : Fin B) := u.pids 0*R+col u B j
```
</details>

<details><summary><code>gradient</code></summary>

```lean
def gradient (cap scale : Bool) (SC SL : ℝ) (lab : Int) (col : Nat)
    (raw lse upstream : ℝ) : ℝ :=
  let scaled := if scale then raw * SL else raw
  let t := Real.tanh (scaled / SC)
  let x := if cap then SC * t else scaled
  let p := Real.exp (x - lse)
  let y := if (col : Int) = lab then p - 1 else p
  let y := if scale then y * SL else y
  let y := if cap then y * (1 - t*t) else y
  (if lab = -100 then 0 else upstream) * y
```
</details>

<details><summary><code>backwardOutside</code></summary>

```lean
def backwardOutside (A : FlatAlloc) (u : BlockState) (X : RegionName) (V R B : Nat)
    (r : RegionName) (o : Nat) : Prop :=
  r≠A.flat ∨ ∀ j : Fin B, col u B j<V → o≠A.addr X (address u R B j)
```
</details>

<details><summary><code>lse</code></summary>

```
/-- A padded softcap lane contributes exp(-SC); an uncapped masked lane contributes zero. -/
```
```lean
def lse {B : Nat} (cap scale : Bool) (SC SL : ℝ) (V start : Nat) (xs : Fin B → ℝ) : ℝ :=
  Real.log (∑ j, if start+j.val<V then Real.exp (transform cap scale SC SL (xs j))
    else if cap then Real.exp (-SC) else 0)
```
</details>

<details><summary><code>transform</code></summary>

```
/-! Pure formulas use initial values, independently of execution. `lse`
includes capped padded lanes; `partialLoss` is only the chunk-zero partial
term. `gradient` uses the caller-supplied LSE, without claiming a host
forward/backward composition or a calculus theorem. -/
```
```lean
def transform (cap scale : Bool) (SC SL x : ℝ) : ℝ :=
  let x := if scale then SL*x else x
  if cap then SC*Real.tanh (x/SC) else x
```
</details>

## Public theorem: `backward_index_commuted_correctness`

**Statement:**
```lean
specification backward_index_commuted_correctness (storage : Storage) (cap scale : Bool)
    (X D Z : RegionName) (L : Region .int) (V R DR B : Nat) (SC SL : ℝ) :
    Spec.Real (backward_correct_target
      (backward storage cap scale Bool.true X D Z L V R DR B SC SL).eraseDType.toAlgKernel
      cap scale X D Z L V R DR B SC SL)
```

**Closed-form spec defs (transitive):** `backward_correct_target`, `backward`, `col`, `address`, `gradient`, `backwardOutside`, `lse`, `transform`

<details><summary><code>backward_correct_target</code></summary>

```lean
def backward_correct_target (kernel : Kernel) (cap scale : Bool) (X D Z : RegionName) (L : Region .int)
    (V R DR B : Nat) (SC SL : ℝ) : Prop :=
  (cap=Bool.true → SC≠0) →
  ∀ A : FlatAlloc, A.Disjoint → (∀ r, r ∉ A.regions → A.extent r=0) →
  ∀ u : BlockState, u.undef=(fun _ _ => 0) → BackwardBounds X D Z L V R DR B u A.extent →
    ∃ v, exec (A.flattenKernel kernel) (A.flattenState u)=some v ∧
      (∀ j : Fin B, col u B j<V → v.readMem A.flat (A.addr X (address u R B j)) =
        gradient cap scale SC SL (u.readMemValue .int (Region.cast L) (u.pids 0))
          (col u B j) (u.readMem X (address u R B j))
          (u.readMem Z (u.pids 0)) (u.readMem D (u.pids 0*DR))) ∧
      (∀ r o, backwardOutside A u X V R B r o → v.mem r o=(A.flattenState u).mem r o)
```
</details>

<details><summary><code>backward</code></summary>

```lean
def backward (storage : Storage) (cap scale swapped : Bool) := match storage,cap,scale with
  | .fp16,Bool.false,Bool.false => if swapped then fastCeBackward%[tl.float16,noCap,noScale,indexCommuted] else fastCeBackward%[tl.float16,noCap,noScale,original]
  | .fp16,Bool.false,Bool.true => if swapped then fastCeBackward%[tl.float16,noCap,scale,indexCommuted] else fastCeBackward%[tl.float16,noCap,scale,original]
  | .fp16,Bool.true,Bool.false => if swapped then fastCeBackward%[tl.float16,cap,noScale,indexCommuted] else fastCeBackward%[tl.float16,cap,noScale,original]
  | .fp16,Bool.true,Bool.true => if swapped then fastCeBackward%[tl.float16,cap,scale,indexCommuted] else fastCeBackward%[tl.float16,cap,scale,original]
  | .bf16,Bool.false,Bool.false => if swapped then fastCeBackward%[tl.bfloat16,noCap,noScale,indexCommuted] else fastCeBackward%[tl.bfloat16,noCap,noScale,original]
  | .bf16,Bool.false,Bool.true => if swapped then fastCeBackward%[tl.bfloat16,noCap,scale,indexCommuted] else fastCeBackward%[tl.bfloat16,noCap,scale,original]
  | .bf16,Bool.true,Bool.false => if swapped then fastCeBackward%[tl.bfloat16,cap,noScale,indexCommuted] else fastCeBackward%[tl.bfloat16,cap,noScale,original]
  | .bf16,Bool.true,Bool.true => if swapped then fastCeBackward%[tl.bfloat16,cap,scale,indexCommuted] else fastCeBackward%[tl.bfloat16,cap,scale,original]
  | .fp32,Bool.false,Bool.false => if swapped then fastCeBackward%[tl.float32,noCap,noScale,indexCommuted] else fastCeBackward%[tl.float32,noCap,noScale,original]
  | .fp32,Bool.false,Bool.true => if swapped then fastCeBackward%[tl.float32,noCap,scale,indexCommuted] else fastCeBackward%[tl.float32,noCap,scale,original]
  | .fp32,Bool.true,Bool.false => if swapped then fastCeBackward%[tl.float32,cap,noScale,indexCommuted] else fastCeBackward%[tl.float32,cap,noScale,original]
  | .fp32,Bool.true,Bool.true => if swapped then fastCeBackward%[tl.float32,cap,scale,indexCommuted] else fastCeBackward%[tl.float32,cap,scale,original]
```
</details>

<details><summary><code>col</code></summary>

```lean
def col (u : BlockState) (B : Nat) (j : Fin B) := u.pids 1*B+j.val
```
</details>

<details><summary><code>address</code></summary>

```lean
def address (u : BlockState) (R B : Nat) (j : Fin B) := u.pids 0*R+col u B j
```
</details>

<details><summary><code>gradient</code></summary>

```lean
def gradient (cap scale : Bool) (SC SL : ℝ) (lab : Int) (col : Nat)
    (raw lse upstream : ℝ) : ℝ :=
  let scaled := if scale then raw * SL else raw
  let t := Real.tanh (scaled / SC)
  let x := if cap then SC * t else scaled
  let p := Real.exp (x - lse)
  let y := if (col : Int) = lab then p - 1 else p
  let y := if scale then y * SL else y
  let y := if cap then y * (1 - t*t) else y
  (if lab = -100 then 0 else upstream) * y
```
</details>

<details><summary><code>backwardOutside</code></summary>

```lean
def backwardOutside (A : FlatAlloc) (u : BlockState) (X : RegionName) (V R B : Nat)
    (r : RegionName) (o : Nat) : Prop :=
  r≠A.flat ∨ ∀ j : Fin B, col u B j<V → o≠A.addr X (address u R B j)
```
</details>

<details><summary><code>lse</code></summary>

```
/-- A padded softcap lane contributes exp(-SC); an uncapped masked lane contributes zero. -/
```
```lean
def lse {B : Nat} (cap scale : Bool) (SC SL : ℝ) (V start : Nat) (xs : Fin B → ℝ) : ℝ :=
  Real.log (∑ j, if start+j.val<V then Real.exp (transform cap scale SC SL (xs j))
    else if cap then Real.exp (-SC) else 0)
```
</details>

<details><summary><code>transform</code></summary>

```
/-! Pure formulas use initial values, independently of execution. `lse`
includes capped padded lanes; `partialLoss` is only the chunk-zero partial
term. `gradient` uses the caller-supplied LSE, without claiming a host
forward/backward composition or a calculus theorem. -/
```
```lean
def transform (cap scale : Bool) (SC SL x : ℝ) : ℝ :=
  let x := if scale then SL*x else x
  if cap then SC*Real.tanh (x/SC) else x
```
</details>
