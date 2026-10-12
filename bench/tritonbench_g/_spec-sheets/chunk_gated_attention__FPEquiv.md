# Spec sheet — `bench/tritonbench_g/chunk_gated_attention/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/chunk_gated_attention/chunk_gated_attention.py`

## Public theorem: `cumulative_equiv`

**Statement:**
```lean
specification cumulative_equiv :
    cumulative dtype out_dtype s o s_s_h s_s_t s_s_d T S BT BS ≡[([] : Spec.Assumptions FP.GuardedFragment)]
    cumulativeIndexFirst dtype out_dtype s o s_s_h s_s_t s_s_d T S BT BS
```

**Closed-form spec defs (transitive):** `cumulative`, `cumulativeIndexFirst`, `cumOriginal`, `cumIndexFirst`

<details><summary><code>cumulative</code></summary>

```lean
def cumulative : Program where
  kernel := TypedSource.cumOriginal dtype out_dtype s o s_s_h s_s_t s_s_d T S BT BS
  domain := []
```
</details>

<details><summary><code>cumulativeIndexFirst</code></summary>

```lean
def cumulativeIndexFirst : Program :=
  { cumulative dtype out_dtype s o s_s_h s_s_t s_s_d T S BT BS with
    kernel := TypedSource.cumIndexFirst dtype out_dtype s o s_s_h s_s_t s_s_d T S BT BS }
```
</details>

<details><summary><code>cumOriginal</code></summary>

```lean
def cumOriginal := gatedCumSource% original
```
</details>

<details><summary><code>cumIndexFirst</code></summary>

```lean
def cumIndexFirst := gatedCumSource% indexFirst

/-- Source: chunk_gated_attention.py, chunk_gated_abc_fwd_kernel_h.
K/V share dot-input storage. Gate/history/initial/final storage are explicit.
The candidate commutes only the post-gate fp32 carry/dot addition. In
particular, the optional initial-state addition and intrinsic tl.exp remain.
Both endpoint pointers use the full chunk-end index in a flattened parent;
checked-off loads have unspecified values, including partial final chunks. -/
syntax "gatedHSource%" "[" tritonDType "," tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(gatedHSource%[$dotTy:tritonDType, $gateTy:tritonDType, $historyTy:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `sumLeft do
      Macro.throwError "expected original or sumLeft"
    let update : TSyntax `tritonStmt ←
      if version.getId == `sumLeft then `(tritonStmt| b_h = tl.dot(b_k, b_v, allow_tf32=false) + b_h)
      else `(tritonStmt| b_h += tl.dot(b_k, b_v, allow_tf32=false))
    `(fun (dtype out_dtype : FloatDType) (k v g h h0 ht : RegionName)
        (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d
          T K V BT BK BV NT : Nat) (GATEK USE_INITIAL_STATE STORE_FINAL_STATE : Bool) =>
      ComputeKernel.mk [k, v, g, h0] [h, ht] (triton {
        tl.region k = $dotTy, v = $dotTy, g = $gateTy, h = $historyTy,
          h0 = DTYPE, ht = OUT_DTYPE
        i_v = tl.program_id(0)
        i_k = tl.program_id(1)
        i_bh = tl.program_id(2)
        b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
        if USE_INITIAL_STATE {
          p_h = tl.make_block_ptr(base=h0 + i_bh * $(K) * $(V),
            shape=($(K), $(V)), strides=($(V), $(1)),
            offsets=(i_k * $(BK), i_v * $(BV)), block_shape=($(BK), $(BV)), order=(1, 0))
          b_h += tl.load(p_h, boundary_check=([0, 1] : List Nat)).to(tl.float32)
        }
        for i_t in range($(0), $(NT), $(1)) {
          p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h),
            shape=($(K), $(T)), strides=($(s_k_d), $(s_k_t)),
            offsets=(i_k * $(BK), i_t * $(BT)), block_shape=($(BK), $(BT)), order=(0, 1))
          p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h),
            shape=($(T), $(V)), strides=($(s_v_t), $(s_v_d)),
            offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
          p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V),
            shape=($(K), $(V)), strides=($(s_h_t), $(s_h_d)),
            offsets=(i_k * $(BK), i_v * $(BV)), block_shape=($(BK), $(BV)), order=(1, 0))
          tl.store(p_h, (b_h).to($historyTy:tritonDType), boundary_check=([0, 1] : List Nat))
          b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
          b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
          if GATEK {
            p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h),
              shape=($(K), $(T)), strides=($(s_k_d), $(s_k_t)),
              offsets=(i_k * $(BK), i_t * $(BT)), block_shape=($(BK), $(BT)), order=(0, 1))
            p_gn = tl.make_block_ptr(base=g + i_bh * $(s_k_h),
              shape=($(T * K)), strides=($(s_k_d)),
              offsets=((i_t * $(BT) + $(BT) - $(1)) * $(K) + i_k * $(BK)),
              block_shape=($(BK)), order=(0))
            b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
            b_h = b_h * tl.exp(b_gn)[:, None]
            b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
            b_k = (b_k * tl.exp(b_gn[:, None] - b_g)).to($dotTy:tritonDType)
          } else {
            p_g = tl.make_block_ptr(base=g + i_bh * $(s_v_h),
              shape=($(T), $(V)), strides=($(s_v_t), $(s_v_d)),
              offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
            p_gn = tl.make_block_ptr(base=g + i_bh * $(s_v_h),
              shape=($(T * V)), strides=($(s_v_d)),
              offsets=((i_t * $(BT) + $(BT) - $(1)) * $(V) + i_v * $(BV)),
              block_shape=($(BV)), order=(0))
            b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
            b_h = b_h * tl.exp(b_gn)[None, :]
            b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
            b_v = (b_v * tl.exp(b_gn[None, :] - b_g)).to($dotTy:tritonDType)
          }
          $update
        }
        if STORE_FINAL_STATE {
          p_h = tl.make_block_ptr(base=ht + i_bh * $(K) * $(V),
            shape=($(K), $(V)), strides=($(V), $(1)),
            offsets=(i_k * $(BK), i_v * $(BV)), block_shape=($(BK), $(BV)), order=(1, 0))
          tl.store(p_h, (b_h).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
        }
      }).surfaceBody)

inductive Storage where
  | fp32 | fp16 | bf16
  deriving DecidableEq
```
</details>

## Public theorem: `recurrent_equiv`

**Statement:**
```lean
specification recurrent_equiv (R : FP.ScalarArithmetic.Rules) :
    recurrent dotStorage gateStorage historyStorage dtype out_dtype k v g h h0 ht
      s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d T K V BT BK BV NT gateK hasSeed storeFinal ≡[R]
    recurrentSumLeft dotStorage gateStorage historyStorage dtype out_dtype k v g h h0 ht
      s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d T K V BT BK BV NT gateK hasSeed storeFinal
```

**Closed-form spec defs (transitive):** `recurrent`, `recurrentSumLeft`, `hOriginal`, `hSumLeft`, `hSource`

<details><summary><code>recurrent</code></summary>

```
/-- Every reached iteration executes the actual history/K/V/gate prefix before
checking the fp32 carry and TF32-disabled dot result. Intrinsic tl.exp remains
unchanged, so no exponential rewrite atom is used. -/
```
```lean
def recurrent : Program where
  kernel := TypedSource.hOriginal dotStorage gateStorage historyStorage dtype out_dtype k v g h h0 ht
    s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d T K V BT BK BV NT gateK hasSeed storeFinal
  domain := FPEquivProofs.requirements
    (TypedSource.hOriginal dotStorage gateStorage historyStorage dtype out_dtype k v g h h0 ht
      s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d T K V BT BK BV NT gateK hasSeed storeFinal)
    dotStorage BK BT BV
```
</details>

<details><summary><code>recurrentSumLeft</code></summary>

```lean
def recurrentSumLeft : Program :=
  { recurrent dotStorage gateStorage historyStorage dtype out_dtype k v g h h0 ht
      s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d T K V BT BK BV NT gateK hasSeed storeFinal with
    kernel := TypedSource.hSumLeft dotStorage gateStorage historyStorage dtype out_dtype k v g h h0 ht
      s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d T K V BT BK BV NT gateK hasSeed storeFinal }
```
</details>

<details><summary><code>hOriginal</code></summary>

```lean
def hOriginal (dotStorage gateStorage historyStorage : Storage) :=
  hSource dotStorage gateStorage historyStorage Bool.false
```
</details>

<details><summary><code>hSumLeft</code></summary>

```lean
def hSumLeft (dotStorage gateStorage historyStorage : Storage) :=
  hSource dotStorage gateStorage historyStorage Bool.true
```
</details>

<details><summary><code>hSource</code></summary>

```lean
def hSource (dotStorage gateStorage historyStorage : Storage) (swapped : Bool) :=
  match dotStorage, gateStorage, historyStorage with
  | .fp32, .fp32, .fp32 =>
      if swapped then gatedHSource%[tl.float32, tl.float32, tl.float32, sumLeft]
      else gatedHSource%[tl.float32, tl.float32, tl.float32, original]
  | .fp32, .fp32, .fp16 =>
      if swapped then gatedHSource%[tl.float32, tl.float32, tl.float16, sumLeft]
      else gatedHSource%[tl.float32, tl.float32, tl.float16, original]
  | .fp32, .fp32, .bf16 =>
      if swapped then gatedHSource%[tl.float32, tl.float32, tl.bfloat16, sumLeft]
      else gatedHSource%[tl.float32, tl.float32, tl.bfloat16, original]
  | .fp32, .fp16, .fp32 =>
      if swapped then gatedHSource%[tl.float32, tl.float16, tl.float32, sumLeft]
      else gatedHSource%[tl.float32, tl.float16, tl.float32, original]
  | .fp32, .fp16, .fp16 =>
      if swapped then gatedHSource%[tl.float32, tl.float16, tl.float16, sumLeft]
      else gatedHSource%[tl.float32, tl.float16, tl.float16, original]
  | .fp32, .fp16, .bf16 =>
      if swapped then gatedHSource%[tl.float32, tl.float16, tl.bfloat16, sumLeft]
      else gatedHSource%[tl.float32, tl.float16, tl.bfloat16, original]
  | .fp32, .bf16, .fp32 =>
      if swapped then gatedHSource%[tl.float32, tl.bfloat16, tl.float32, sumLeft]
      else gatedHSource%[tl.float32, tl.bfloat16, tl.float32, original]
  | .fp32, .bf16, .fp16 =>
      if swapped then gatedHSource%[tl.float32, tl.bfloat16, tl.float16, sumLeft]
      else gatedHSource%[tl.float32, tl.bfloat16, tl.float16, original]
  | .fp32, .bf16, .bf16 =>
      if swapped then gatedHSource%[tl.float32, tl.bfloat16, tl.bfloat16, sumLeft]
      else gatedHSource%[tl.float32, tl.bfloat16, tl.bfloat16, original]
  | .fp16, .fp32, .fp32 =>
      if swapped then gatedHSource%[tl.float16, tl.float32, tl.float32, sumLeft]
      else gatedHSource%[tl.float16, tl.float32, tl.float32, original]
  | .fp16, .fp32, .fp16 =>
      if swapped then gatedHSource%[tl.float16, tl.float32, tl.float16, sumLeft]
      else gatedHSource%[tl.float16, tl.float32, tl.float16, original]
  | .fp16, .fp32, .bf16 =>
      if swapped then gatedHSource%[tl.float16, tl.float32, tl.bfloat16, sumLeft]
      else gatedHSource%[tl.float16, tl.float32, tl.bfloat16, original]
  | .fp16, .fp16, .fp32 =>
      if swapped then gatedHSource%[tl.float16, tl.float16, tl.float32, sumLeft]
      else gatedHSource%[tl.float16, tl.float16, tl.float32, original]
  | .fp16, .fp16, .fp16 =>
      if swapped then gatedHSource%[tl.float16, tl.float16, tl.float16, sumLeft]
      else gatedHSource%[tl.float16, tl.float16, tl.float16, original]
  | .fp16, .fp16, .bf16 =>
      if swapped then gatedHSource%[tl.float16, tl.float16, tl.bfloat16, sumLeft]
      else gatedHSource%[tl.float16, tl.float16, tl.bfloat16, original]
  | .fp16, .bf16, .fp32 =>
      if swapped then gatedHSource%[tl.float16, tl.bfloat16, tl.float32, sumLeft]
      else gatedHSource%[tl.float16, tl.bfloat16, tl.float32, original]
  | .fp16, .bf16, .fp16 =>
      if swapped then gatedHSource%[tl.float16, tl.bfloat16, tl.float16, sumLeft]
      else gatedHSource%[tl.float16, tl.bfloat16, tl.float16, original]
  | .fp16, .bf16, .bf16 =>
      if swapped then gatedHSource%[tl.float16, tl.bfloat16, tl.bfloat16, sumLeft]
      else gatedHSource%[tl.float16, tl.bfloat16, tl.bfloat16, original]
  | .bf16, .fp32, .fp32 =>
      if swapped then gatedHSource%[tl.bfloat16, tl.float32, tl.float32, sumLeft]
      else gatedHSource%[tl.bfloat16, tl.float32, tl.float32, original]
  | .bf16, .fp32, .fp16 =>
      if swapped then gatedHSource%[tl.bfloat16, tl.float32, tl.float16, sumLeft]
      else gatedHSource%[tl.bfloat16, tl.float32, tl.float16, original]
  | .bf16, .fp32, .bf16 =>
      if swapped then gatedHSource%[tl.bfloat16, tl.float32, tl.bfloat16, sumLeft]
      else gatedHSource%[tl.bfloat16, tl.float32, tl.bfloat16, original]
  | .bf16, .fp16, .fp32 =>
      if swapped then gatedHSource%[tl.bfloat16, tl.float16, tl.float32, sumLeft]
      else gatedHSource%[tl.bfloat16, tl.float16, tl.float32, original]
  | .bf16, .fp16, .fp16 =>
      if swapped then gatedHSource%[tl.bfloat16, tl.float16, tl.float16, sumLeft]
      else gatedHSource%[tl.bfloat16, tl.float16, tl.float16, original]
  | .bf16, .fp16, .bf16 =>
      if swapped then gatedHSource%[tl.bfloat16, tl.float16, tl.bfloat16, sumLeft]
      else gatedHSource%[tl.bfloat16, tl.float16, tl.bfloat16, original]
  | .bf16, .bf16, .fp32 =>
      if swapped then gatedHSource%[tl.bfloat16, tl.bfloat16, tl.float32, sumLeft]
      else gatedHSource%[tl.bfloat16, tl.bfloat16, tl.float32, original]
  | .bf16, .bf16, .fp16 =>
      if swapped then gatedHSource%[tl.bfloat16, tl.bfloat16, tl.float16, sumLeft]
      else gatedHSource%[tl.bfloat16, tl.bfloat16, tl.float16, original]
  | .bf16, .bf16, .bf16 =>
      if swapped then gatedHSource%[tl.bfloat16, tl.bfloat16, tl.bfloat16, sumLeft]
      else gatedHSource%[tl.bfloat16, tl.bfloat16, tl.bfloat16, original]
```
</details>
