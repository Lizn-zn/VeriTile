# Spec sheet — `bench/tritonbench_g/chunk_linear_attn/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/chunk_linear_attn/chunk_linear_attn.py`

## Public theorem: `forwardState_equiv`

**Statement:**
```lean
specification forwardState_equiv (R : FP.ScalarArithmetic.Rules) :
    forwardState dotStorage dtype out_dtype k v h h0 ht
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t T K V BT BK BV NT USE_INITIAL_STATE STORE_FINAL_STATE ≡[R] forwardStateDotLeft dotStorage dtype out_dtype k v h h0 ht
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t T K V BT BK BV NT USE_INITIAL_STATE STORE_FINAL_STATE
```

**Closed-form spec defs (transitive):** `forwardState`, `forwardStateDotLeft`, `forwardH`, `forwardHDotLeft`, `forwardHSource`

<details><summary><code>forwardState</code></summary>

```lean
def forwardState : Program where
  kernel := ChunkLinearAttn.forwardH dotStorage dtype out_dtype k v h h0 ht
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t T K V BT BK BV NT USE_INITIAL_STATE STORE_FINAL_STATE
  domain := FPProof.forwardHRequirements (ChunkLinearAttn.forwardH dotStorage dtype out_dtype k v h h0 ht
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t T K V BT BK BV NT USE_INITIAL_STATE STORE_FINAL_STATE) dotStorage BK BV BT
```
</details>

<details><summary><code>forwardStateDotLeft</code></summary>

```lean
def forwardStateDotLeft : Program :=
  { forwardState dotStorage dtype out_dtype k v h h0 ht
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t T K V BT BK BV NT USE_INITIAL_STATE STORE_FINAL_STATE with
    kernel := ChunkLinearAttn.forwardHDotLeft dotStorage dtype out_dtype k v h h0 ht
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t T K V BT BK BV NT USE_INITIAL_STATE STORE_FINAL_STATE }
```
</details>

<details><summary><code>forwardH</code></summary>

```lean
def forwardH (dotStorage : Storage) := forwardHSource dotStorage Bool.false
```
</details>

<details><summary><code>forwardHDotLeft</code></summary>

```lean
def forwardHDotLeft (dotStorage : Storage) := forwardHSource dotStorage Bool.true

/-- Source: chunk_linear_attn_fwd_kernel_o. -/
syntax "linearForwardOSource%" "[" tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(linearForwardOSource%[$dotTy:tritonDType, $vTy:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `scaleFirst do
      Macro.throwError "expected original or scaleFirst"
    let update : TSyntax `tritonStmt ←
      if version.getId == `scaleFirst then `(tritonStmt| b_o = $((scale : ℝ)) * (b_o + tl.dot((b_s).to($vTy:tritonDType), b_v, allow_tf32=False)))
      else `(tritonStmt| b_o = (b_o + tl.dot((b_s).to($vTy:tritonDType), b_v, allow_tf32=False)) * $((scale : ℝ)))
    `(fun (out_dtype : FloatDType) (q k v h o : RegionName)
        (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat) (scale : ℝ) (T K V BT BK BV : Nat) =>
      ComputeKernel.mk [q,k,v,h] [o] (triton {
        tl.region q = $dotTy, k = $dotTy, h = $dotTy, v = $vTy, o = OUT_DTYPE
        i_v = tl.program_id(0)
        i_t = tl.program_id(1)
        i_bh = tl.program_id(2)
        o_i = tl.arange(0, $(BT))
        m_s = o_i[:, None] >= o_i[None, :]
        b_o = tl.zeros([$(BT), $(BV)], dtype=tl.float32)
        b_s = tl.zeros([$(BT), $(BT)], dtype=tl.float32)
        for i_k in range(tl.cdiv($(K), $(BK))) {
          p_q = tl.make_block_ptr(base=q + (i_bh * $(s_qk_h)), shape=($(T), $(K)), strides=($(s_qk_t), $(s_qk_d)), offsets=((i_t * $(BT)), (i_k * $(BK))), block_shape=($(BT), $(BK)), order=(1, 0))
          p_k = tl.make_block_ptr(base=k + (i_bh * $(s_qk_h)), shape=($(K), $(T)), strides=($(s_qk_d), $(s_qk_t)), offsets=((i_k * $(BK)), (i_t * $(BT))), block_shape=($(BK), $(BT)), order=(0, 1))
          p_h = tl.make_block_ptr(base=h + (i_bh * $(s_h_h)) + ((i_t * $(K)) * $(V)), shape=($(K), $(V)), strides=($(s_h_t), 1), offsets=((i_k * $(BK)), (i_v * $(BV))), block_shape=($(BK), $(BV)), order=(1, 0))
          b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat))
          b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
          b_h = tl.load(p_h, boundary_check=([0, 1] : List Nat))
          b_o += tl.dot(b_q, b_h, allow_tf32=False)
          b_s += tl.dot(b_q, b_k, allow_tf32=False)
        }
        b_s = tl.where(m_s, b_s, 0)
        p_v = tl.make_block_ptr(base=v + (i_bh * $(s_vo_h)), shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)), offsets=((i_t * $(BT)), (i_v * $(BV))), block_shape=($(BT), $(BV)), order=(1, 0))
        p_o = tl.make_block_ptr(base=o + (i_bh * $(s_vo_h)), shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)), offsets=((i_t * $(BT)), (i_v * $(BV))), block_shape=($(BT), $(BV)), order=(1, 0))
        b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
        $update
        tl.store(p_o, (b_o).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
      }).surfaceBody)
```
</details>

<details><summary><code>forwardHSource</code></summary>

```lean
def forwardHSource (dotStorage : Storage) (swapped : Bool) :=
  match dotStorage with
  | .fp32 =>
      if swapped then linearForwardHSource%[tl.float32, dotLeft]
      else linearForwardHSource%[tl.float32, original]
  | .fp16 =>
      if swapped then linearForwardHSource%[tl.float16, dotLeft]
      else linearForwardHSource%[tl.float16, original]
  | .bf16 =>
      if swapped then linearForwardHSource%[tl.bfloat16, dotLeft]
      else linearForwardHSource%[tl.bfloat16, original]
```
</details>

## Public theorem: `forwardOutput_equiv`

**Statement:**
```lean
specification forwardOutput_equiv (R : FP.ScalarArithmetic.Rules) :
    forwardOutput dotStorage valueStorage out_dtype q k v h o
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t scale T K V BT BK BV  ≡[R] forwardOutputScaleFirst dotStorage valueStorage out_dtype q k v h o
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t scale T K V BT BK BV
```

**Closed-form spec defs (transitive):** `forwardOutput`, `forwardOutputScaleFirst`, `forwardO`, `forwardOScaleFirst`, `forwardOSource`

<details><summary><code>forwardOutput</code></summary>

```lean
def forwardOutput : Program where
  kernel := ChunkLinearAttn.forwardO dotStorage valueStorage out_dtype q k v h o
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t scale T K V BT BK BV
  domain := FPProof.forwardORequirements (ChunkLinearAttn.forwardO dotStorage valueStorage out_dtype q k v h o
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t scale T K V BT BK BV ) valueStorage BT BV scale
```
</details>

<details><summary><code>forwardOutputScaleFirst</code></summary>

```lean
def forwardOutputScaleFirst : Program :=
  { forwardOutput dotStorage valueStorage out_dtype q k v h o
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t scale T K V BT BK BV  with
    kernel := ChunkLinearAttn.forwardOScaleFirst dotStorage valueStorage out_dtype q k v h o
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t scale T K V BT BK BV  }
```
</details>

<details><summary><code>forwardO</code></summary>

```lean
def forwardO (dotStorage : Storage) (valueStorage : Storage) := forwardOSource dotStorage valueStorage Bool.false
```
</details>

<details><summary><code>forwardOScaleFirst</code></summary>

```lean
def forwardOScaleFirst (dotStorage : Storage) (valueStorage : Storage) := forwardOSource dotStorage valueStorage Bool.true

/-- Source: chunk_linear_attn_bwd_kernel_dh. -/
syntax "linearBackwardHSource%" "[" tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(linearBackwardHSource%[$dotTy:tritonDType, $vTy:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `dotLeft do
      Macro.throwError "expected original or dotLeft"
    let update : TSyntax `tritonStmt ←
      if version.getId == `dotLeft then `(tritonStmt| b_dh = tl.dot(b_q, (b_do).to($dotTy:tritonDType), allow_tf32=False) + b_dh)
      else `(tritonStmt| b_dh += tl.dot(b_q, (b_do).to($dotTy:tritonDType), allow_tf32=False))
    `(fun (out_dtype : FloatDType) (q do_ dh : RegionName)
        (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat) (scale : ℝ) (T K V BT BK BV NT : Nat) =>
      ComputeKernel.mk [q,do_] [dh] (triton {
        tl.region q = $dotTy, do_ = $vTy, dh = OUT_DTYPE
        i_k = tl.program_id(0)
        i_v = tl.program_id(1)
        i_bh = tl.program_id(2)
        b_dh = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
        for j in range($(0), $(NT), $(1)) {
          i_t = $(NT) - $(1) - j
          p_q = tl.make_block_ptr(base=q + (i_bh * $(s_qk_h)), shape=($(K), $(T)), strides=($(s_qk_d), $(s_qk_t)), offsets=((i_k * $(BK)), (i_t * $(BT))), block_shape=($(BK), $(BT)), order=(0, 1))
          p_do = tl.make_block_ptr(base=do_ + (i_bh * $(s_vo_h)), shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)), offsets=((i_t * $(BT)), (i_v * $(BV))), block_shape=($(BT), $(BV)), order=(1, 0))
          p_dh = tl.make_block_ptr(base=(dh + (i_bh * $(s_h_h))) + ((i_t * $(K)) * $(V)), shape=($(K), $(V)), strides=($(s_h_t), 1), offsets=((i_k * $(BK)), (i_v * $(BV))), block_shape=($(BK), $(BV)), order=(1, 0))
          tl.store(p_dh, (b_dh).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
          b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat))
          b_q = ((b_q * $((scale : ℝ)))).to($dotTy:tritonDType)
          b_do = tl.load(p_do, boundary_check=([0, 1] : List Nat))
          $update
        }
      }).surfaceBody)
```
</details>

<details><summary><code>forwardOSource</code></summary>

```lean
def forwardOSource (dotStorage : Storage) (valueStorage : Storage) (swapped : Bool) :=
  match dotStorage,valueStorage with
  | .fp32, .fp32 =>
      if swapped then linearForwardOSource%[tl.float32, tl.float32, scaleFirst]
      else linearForwardOSource%[tl.float32, tl.float32, original]
  | .fp32, .fp16 =>
      if swapped then linearForwardOSource%[tl.float32, tl.float16, scaleFirst]
      else linearForwardOSource%[tl.float32, tl.float16, original]
  | .fp32, .bf16 =>
      if swapped then linearForwardOSource%[tl.float32, tl.bfloat16, scaleFirst]
      else linearForwardOSource%[tl.float32, tl.bfloat16, original]
  | .fp16, .fp32 =>
      if swapped then linearForwardOSource%[tl.float16, tl.float32, scaleFirst]
      else linearForwardOSource%[tl.float16, tl.float32, original]
  | .fp16, .fp16 =>
      if swapped then linearForwardOSource%[tl.float16, tl.float16, scaleFirst]
      else linearForwardOSource%[tl.float16, tl.float16, original]
  | .fp16, .bf16 =>
      if swapped then linearForwardOSource%[tl.float16, tl.bfloat16, scaleFirst]
      else linearForwardOSource%[tl.float16, tl.bfloat16, original]
  | .bf16, .fp32 =>
      if swapped then linearForwardOSource%[tl.bfloat16, tl.float32, scaleFirst]
      else linearForwardOSource%[tl.bfloat16, tl.float32, original]
  | .bf16, .fp16 =>
      if swapped then linearForwardOSource%[tl.bfloat16, tl.float16, scaleFirst]
      else linearForwardOSource%[tl.bfloat16, tl.float16, original]
  | .bf16, .bf16 =>
      if swapped then linearForwardOSource%[tl.bfloat16, tl.bfloat16, scaleFirst]
      else linearForwardOSource%[tl.bfloat16, tl.bfloat16, original]
```
</details>

## Public theorem: `backwardState_equiv`

**Statement:**
```lean
specification backwardState_equiv (R : FP.ScalarArithmetic.Rules) :
    backwardState dotStorage valueStorage out_dtype q do_ dh
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t scale T K V BT BK BV NT  ≡[R] backwardStateDotLeft dotStorage valueStorage out_dtype q do_ dh
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t scale T K V BT BK BV NT
```

**Closed-form spec defs (transitive):** `backwardState`, `backwardStateDotLeft`, `backwardH`, `backwardHDotLeft`, `backwardHSource`

<details><summary><code>backwardState</code></summary>

```lean
def backwardState : Program where
  kernel := ChunkLinearAttn.backwardH dotStorage valueStorage out_dtype q do_ dh
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t scale T K V BT BK BV NT
  domain := FPProof.backwardHRequirements (ChunkLinearAttn.backwardH dotStorage valueStorage out_dtype q do_ dh
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t scale T K V BT BK BV NT ) dotStorage valueStorage BK BV BT
```
</details>

<details><summary><code>backwardStateDotLeft</code></summary>

```lean
def backwardStateDotLeft : Program :=
  { backwardState dotStorage valueStorage out_dtype q do_ dh
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t scale T K V BT BK BV NT  with
    kernel := ChunkLinearAttn.backwardHDotLeft dotStorage valueStorage out_dtype q do_ dh
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t scale T K V BT BK BV NT  }
```
</details>

<details><summary><code>backwardH</code></summary>

```lean
def backwardH (dotStorage : Storage) (valueStorage : Storage) := backwardHSource dotStorage valueStorage Bool.false
```
</details>

<details><summary><code>backwardHDotLeft</code></summary>

```lean
def backwardHDotLeft (dotStorage : Storage) (valueStorage : Storage) := backwardHSource dotStorage valueStorage Bool.true

/-- Source: chunk_linear_attn_bwd_kernel_dqkv. -/
syntax "linearBackwardQKVSource%" "[" tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(linearBackwardQKVSource%[$dotTy:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `dotLeft do
      Macro.throwError "expected original or dotLeft"
    let update : TSyntax `tritonStmt ←
      if version.getId == `dotLeft then `(tritonStmt| b_dq = tl.dot(b_ds, b_k, allow_tf32=False) + b_dq)
      else `(tritonStmt| b_dq += tl.dot(b_ds, b_k, allow_tf32=False))
    `(fun  (q k v h do_ dh dq dk dv : RegionName)
        (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat) (scale : ℝ) (T K V BT BK BV NT : Nat) =>
      ComputeKernel.mk [q,k,v,h,do_,dh] [dq,dk,dv] (triton {
        tl.region q = $dotTy, k = $dotTy, v = $dotTy, h = $dotTy, do_ = $dotTy, dh = $dotTy, dq = $dotTy, dk = $dotTy, dv = $dotTy
        i_k = tl.program_id(0)
        i_t = tl.program_id(1)
        i_bh = tl.program_id(2)
        n_bh = tl.num_programs(2)
        o_i = tl.arange(0, $(BT))
        p_q = tl.make_block_ptr(base=q + (i_bh * $(s_qk_h)), shape=($(K), $(T)), strides=($(s_qk_d), $(s_qk_t)), offsets=((i_k * $(BK)), (i_t * $(BT))), block_shape=($(BK), $(BT)), order=(0, 1))
        p_k = tl.make_block_ptr(base=k + (i_bh * $(s_qk_h)), shape=($(T), $(K)), strides=($(s_qk_t), $(s_qk_d)), offsets=((i_t * $(BT)), (i_k * $(BK))), block_shape=($(BT), $(BK)), order=(1, 0))
        b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat))
        b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
        b_s = (tl.dot(b_k, b_q, allow_tf32=False) * $((scale : ℝ)))
        b_s = tl.where(o_i[:, None] <= o_i[None, :], b_s, 0)
        b_dq = tl.zeros([$(BT), $(BK)], dtype=tl.float32)
        b_dk = tl.zeros([$(BT), $(BK)], dtype=tl.float32)
        b_ds = tl.zeros([$(BT), $(BT)], dtype=tl.float32)
        for i_v in range(tl.cdiv($(V), $(BV))) {
          p_v = tl.make_block_ptr(base=v + (i_bh * $(s_vo_h)), shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)), offsets=((i_t * $(BT)), (i_v * $(BV))), block_shape=($(BT), $(BV)), order=(1, 0))
          p_h = tl.make_block_ptr(base=h + (i_bh * $(s_h_h)), shape=($(V), $(NT * K)), strides=(1, $(s_h_t)), offsets=((i_v * $(BV)), ((i_t * $(K)) + (i_k * $(BK)))), block_shape=($(BV), $(BK)), order=(0, 1))
          p_do = tl.make_block_ptr(base=do_ + (i_bh * $(s_vo_h)), shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)), offsets=((i_t * $(BT)), (i_v * $(BV))), block_shape=($(BT), $(BV)), order=(1, 0))
          p_dh = tl.make_block_ptr(base=dh + (i_bh * $(s_h_h)), shape=($(NT * K), $(V)), strides=($(s_h_t), 1), offsets=(((i_t * $(K)) + (i_k * $(BK))), (i_v * $(BV))), block_shape=($(BK), $(BV)), order=(1, 0))
          p_dv = tl.make_block_ptr(base=dv + (((i_k * n_bh) + i_bh) * $(s_vo_h)), shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)), offsets=((i_t * $(BT)), (i_v * $(BV))), block_shape=($(BT), $(BV)), order=(1, 0))
          b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
          b_do = tl.load(p_do, boundary_check=([0, 1] : List Nat))
          b_h = tl.load(p_h, boundary_check=([0, 1] : List Nat))
          b_dh = tl.load(p_dh, boundary_check=([0, 1] : List Nat))
          b_ds += tl.dot(b_do, tl.trans(b_v), allow_tf32=False)
          b_dq += (tl.dot(b_do, b_h, allow_tf32=False) * $((scale : ℝ)))
          b_dk += tl.dot(b_v, tl.trans(b_dh), allow_tf32=False)
          b_dv = (tl.dot(b_k, b_dh, allow_tf32=False) + tl.dot((b_s).to($dotTy:tritonDType), b_do, allow_tf32=False))
          tl.store(p_dv, (b_dv).to($dotTy:tritonDType), boundary_check=([0, 1] : List Nat))
        }
        b_ds = (tl.where(o_i[:, None] >= o_i[None, :], (b_ds * $((scale : ℝ))), 0)).to($dotTy:tritonDType)
        $update
        b_dk += tl.trans(tl.dot(b_q, b_ds, allow_tf32=False))
        p_dq = tl.make_block_ptr(base=dq + (i_bh * $(s_qk_h)), shape=($(T), $(K)), strides=($(s_qk_t), $(s_qk_d)), offsets=((i_t * $(BT)), (i_k * $(BK))), block_shape=($(BT), $(BK)), order=(1, 0))
        p_dk = tl.make_block_ptr(base=dk + (i_bh * $(s_qk_h)), shape=($(T), $(K)), strides=($(s_qk_t), $(s_qk_d)), offsets=((i_t * $(BT)), (i_k * $(BK))), block_shape=($(BT), $(BK)), order=(1, 0))
        tl.store(p_dq, (b_dq).to($dotTy:tritonDType), boundary_check=([0, 1] : List Nat))
        tl.store(p_dk, (b_dk).to($dotTy:tritonDType), boundary_check=([0, 1] : List Nat))
      }).surfaceBody)
```
</details>

<details><summary><code>backwardHSource</code></summary>

```lean
def backwardHSource (dotStorage : Storage) (valueStorage : Storage) (swapped : Bool) :=
  match dotStorage,valueStorage with
  | .fp32, .fp32 =>
      if swapped then linearBackwardHSource%[tl.float32, tl.float32, dotLeft]
      else linearBackwardHSource%[tl.float32, tl.float32, original]
  | .fp32, .fp16 =>
      if swapped then linearBackwardHSource%[tl.float32, tl.float16, dotLeft]
      else linearBackwardHSource%[tl.float32, tl.float16, original]
  | .fp32, .bf16 =>
      if swapped then linearBackwardHSource%[tl.float32, tl.bfloat16, dotLeft]
      else linearBackwardHSource%[tl.float32, tl.bfloat16, original]
  | .fp16, .fp32 =>
      if swapped then linearBackwardHSource%[tl.float16, tl.float32, dotLeft]
      else linearBackwardHSource%[tl.float16, tl.float32, original]
  | .fp16, .fp16 =>
      if swapped then linearBackwardHSource%[tl.float16, tl.float16, dotLeft]
      else linearBackwardHSource%[tl.float16, tl.float16, original]
  | .fp16, .bf16 =>
      if swapped then linearBackwardHSource%[tl.float16, tl.bfloat16, dotLeft]
      else linearBackwardHSource%[tl.float16, tl.bfloat16, original]
  | .bf16, .fp32 =>
      if swapped then linearBackwardHSource%[tl.bfloat16, tl.float32, dotLeft]
      else linearBackwardHSource%[tl.bfloat16, tl.float32, original]
  | .bf16, .fp16 =>
      if swapped then linearBackwardHSource%[tl.bfloat16, tl.float16, dotLeft]
      else linearBackwardHSource%[tl.bfloat16, tl.float16, original]
  | .bf16, .bf16 =>
      if swapped then linearBackwardHSource%[tl.bfloat16, tl.bfloat16, dotLeft]
      else linearBackwardHSource%[tl.bfloat16, tl.bfloat16, original]
```
</details>

## Public theorem: `backwardGradients_equiv`

**Statement:**
```lean
specification backwardGradients_equiv (R : FP.ScalarArithmetic.Rules) :
    backwardGradients dotStorage  q k v h do_ dh dq dk dv
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t scale T K V BT BK BV NT  ≡[R] backwardGradientsDotLeft dotStorage  q k v h do_ dh dq dk dv
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t scale T K V BT BK BV NT
```

**Closed-form spec defs (transitive):** `backwardGradients`, `backwardGradientsDotLeft`, `backwardQKV`, `backwardQKVDotLeft`, `backwardQKVSource`

<details><summary><code>backwardGradients</code></summary>

```lean
def backwardGradients : Program where
  kernel := ChunkLinearAttn.backwardQKV dotStorage  q k v h do_ dh dq dk dv
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t scale T K V BT BK BV NT
  domain := FPProof.backwardQKVRequirements (ChunkLinearAttn.backwardQKV dotStorage  q k v h do_ dh dq dk dv
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t scale T K V BT BK BV NT ) dotStorage BT BK
```
</details>

<details><summary><code>backwardGradientsDotLeft</code></summary>

```lean
def backwardGradientsDotLeft : Program :=
  { backwardGradients dotStorage  q k v h do_ dh dq dk dv
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t scale T K V BT BK BV NT  with
    kernel := ChunkLinearAttn.backwardQKVDotLeft dotStorage  q k v h do_ dh dq dk dv
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t scale T K V BT BK BV NT  }
```
</details>

<details><summary><code>backwardQKV</code></summary>

```lean
def backwardQKV (dotStorage : Storage) := backwardQKVSource dotStorage Bool.false
```
</details>

<details><summary><code>backwardQKVDotLeft</code></summary>

```lean
def backwardQKVDotLeft (dotStorage : Storage) := backwardQKVSource dotStorage Bool.true

end VeriTile.Bench.TritonBenchG.ChunkLinearAttn

#guard_msgs (drop info) in
#auditModuleAxioms
```
</details>

<details><summary><code>backwardQKVSource</code></summary>

```lean
def backwardQKVSource (dotStorage : Storage) (swapped : Bool) :=
  match dotStorage with
  | .fp32 =>
      if swapped then linearBackwardQKVSource%[tl.float32, dotLeft]
      else linearBackwardQKVSource%[tl.float32, original]
  | .fp16 =>
      if swapped then linearBackwardQKVSource%[tl.float16, dotLeft]
      else linearBackwardQKVSource%[tl.float16, original]
  | .bf16 =>
      if swapped then linearBackwardQKVSource%[tl.bfloat16, dotLeft]
      else linearBackwardQKVSource%[tl.bfloat16, original]
```
</details>
