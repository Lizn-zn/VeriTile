# Spec sheet — `bench/tritonbench_g/fast_layernorm/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/fast_layernorm/fast_layernorm.py`

## Public theorem: `forward_equiv`

**Statement:**
```lean
specification forward_equiv (input weight bias : Storage)
    (Y X W b r mu : RegionName) (YR XR N B : Nat) (eps : ℝ) :
    forwardProgram input weight bias Bool.false Y X W b r mu YR XR N B eps ≡[assumptions]
    forwardProgram input weight bias Bool.true Y X W b r mu YR XR N B eps
```

**Closed-form spec defs (transitive):** `forwardProgram`, `assumptions`, `forward`

<details><summary><code>forwardProgram</code></summary>

```lean
def forwardProgram (input weight bias : Storage) (swapped : Bool)
    (Y X W b r mu : RegionName) (YR XR N B : Nat) (eps : ℝ) : Program where
  kernel := forward input weight bias swapped Y X W b r mu YR XR N B eps
  domain := []
```
</details>

<details><summary><code>assumptions</code></summary>

```lean
def assumptions : Spec.Assumptions FP.GuardedFragment := []
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

## Public theorem: `backward_equiv`

**Statement:**
```lean
specification backward_equiv (input weight bias : Storage)
    (Y X W b r mu : RegionName) (YR XR N B : Nat) (eps : ℝ) :
    backwardProgram input weight bias Bool.false Y X W b r mu YR XR N B eps ≡[assumptions]
    backwardProgram input weight bias Bool.true Y X W b r mu YR XR N B eps
```

**Closed-form spec defs (transitive):** `backwardProgram`, `assumptions`, `backward`

<details><summary><code>backwardProgram</code></summary>

```lean
def backwardProgram (input weight bias : Storage) (swapped : Bool)
    (Y X W b r mu : RegionName) (YR XR N B : Nat) (eps : ℝ) : Program where
  kernel := backward input weight bias swapped Y X W b r mu YR XR N B eps
  domain := []
```
</details>

<details><summary><code>assumptions</code></summary>

```lean
def assumptions : Spec.Assumptions FP.GuardedFragment := []
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
