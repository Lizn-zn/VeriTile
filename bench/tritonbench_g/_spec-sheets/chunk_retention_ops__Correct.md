# Spec sheet — `bench/tritonbench_g/chunk_retention_ops/Correct.lean`

**Python source:** `bench/tritonbench_g/chunk_retention_ops/chunk_retention_ops.py`

## Public theorem: `forwardState_correct`

**Statement:**
```lean
specification forwardState_correct (p : Layout) (r : Regions) (storage : Storage) (seedType finalType : FloatDType)
    (hasSeed storeFinal : Bool) (hkey : r.key ≠ r.history) (hvalue : r.value ≠ r.history)
    (hfinal : storeFinal = Bool.true → r.history ≠ r.final) :
    forwardState p r storage seedType finalType hasSeed storeFinal ⊨ forwardStateFormula p hasSeed
```

**Assumptions / layout contracts:**
- `hkey : r.key ≠ r.history`
- `hvalue : r.value ≠ r.history`
- `hfinal : storeFinal = Bool.true → r.history ≠ r.final`

**Closed-form spec defs (transitive):** `Storage`, `forwardState`, `forwardStateFormula`, `forwardH`, `forwardStateBefore`, `forwardHSource`

<details><summary><code>Storage</code></summary>

```lean
abbrev Storage := VeriTile.Bench.TritonBenchG.ChunkRetention.Storage

/-- Source: chunk_retention_fwd_kernel_h. -/
syntax "retentionOpsForwardHSource%" "[" tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(retentionOpsForwardHSource%[$dotTy:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `dotLeft do
      Macro.throwError "expected original or dotLeft"
    let update : TSyntax `tritonStmt ←
      if version.getId == `dotLeft then `(tritonStmt| b_h = tl.dot(b_k, (b_v * d_i[:, None]).to($dotTy:tritonDType), allow_tf32=False) + d_b * b_h)
      else `(tritonStmt| b_h = d_b * b_h + tl.dot(b_k, (b_v * d_i[:, None]).to($dotTy:tritonDType), allow_tf32=False))
    `(fun (dtype out_dtype : FloatDType) (k v h h0 ht : RegionName)
        (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat) (H T K V BT BK BV NT : Nat) (USE_INITIAL_STATE STORE_FINAL_STATE : Bool) =>
      ComputeKernel.mk [k,v,h0] [h,ht] (triton {
        tl.region k = $dotTy, v = $dotTy, h = $dotTy, h0 = DTYPE, ht = OUT_DTYPE
        i_k = tl.program_id(0)
        i_v = tl.program_id(1)
        i_bh = tl.program_id(2)
        i_h = i_bh % $(H)
        b_b = tl.math.log2(1.0 - tl.math.exp2($((-5 : ℝ)) - i_h * 1.0))
        o_i = (tl.arange(0, $(BT))).to(tl.int32)
        d_b = tl.math.exp2($(BT) * b_b)
        d_i = tl.math.exp2(($(BT) - o_i - $(1)) * b_b)
        b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
        if USE_INITIAL_STATE {
          p_h0 = tl.make_block_ptr(base=h0 + ((i_bh * $(K)) * $(V)), shape=($(K), $(V)), strides=($(V), 1), offsets=((i_k * $(BK)), (i_v * $(BV))), block_shape=($(BK), $(BV)), order=(1, 0))
          b_h = (tl.load(p_h0, boundary_check=([0, 1] : List Nat))).to(tl.float32)
        }
        for i_t in range($(NT)) {
          p_k = tl.make_block_ptr(base=k + (i_bh * $(s_qk_h)), shape=($(K), $(T)), strides=($(s_qk_d), $(s_qk_t)), offsets=((i_k * $(BK)), (i_t * $(BT))), block_shape=($(BK), $(BT)), order=(0, 1))
          p_v = tl.make_block_ptr(base=v + (i_bh * $(s_vo_h)), shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)), offsets=((i_t * $(BT)), (i_v * $(BV))), block_shape=($(BT), $(BV)), order=(1, 0))
          p_h = tl.make_block_ptr(base=h + (i_bh * $(s_h_h)) + ((i_t * $(K)) * $(V)), shape=($(K), $(V)), strides=($(s_h_t), 1), offsets=((i_k * $(BK)), (i_v * $(BV))), block_shape=($(BK), $(BV)), order=(1, 0))
          tl.store(p_h, (b_h).to($dotTy:tritonDType), boundary_check=([0, 1] : List Nat))
          b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
          b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
          if i_t == $(NT) - $(1) and $(T) % $(BT) != $(0) {
            d_b = tl.math.exp2(($(T) % $(BT)) * b_b)
            d_i = tl.math.exp2((($(T) % $(BT)) - o_i - $(1)) * b_b)
          }
          $update
        }
        if STORE_FINAL_STATE {
          p_ht = tl.make_block_ptr(base=ht + ((i_bh * $(K)) * $(V)), shape=($(K), $(V)), strides=($(V), 1), offsets=((i_k * $(BK)), (i_v * $(BV))), block_shape=($(BK), $(BV)), order=(1, 0))
          tl.store(p_ht, (b_h).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
        }
      }).surfaceBody)
```
</details>

<details><summary><code>forwardState</code></summary>

```
/-- Active history stores must be injective across chunks and lanes; the
optional final-state store must also be injective. Bounds include the optional
seed/final accesses and every K/V/history access in the full NT loop. -/
```
```lean
def forwardState (p : Layout) (r : Regions) (storage : Storage) (seedType finalType : FloatDType)
    (hasSeed storeFinal : Bool) : RealHIO.IO p where
  kernel := (VeriTile.Bench.TritonBenchG.ChunkRetentionOps.forwardH storage seedType finalType r.key r.value r.history r.seed r.final
    p.keyHead p.keyTime p.keyFeature p.valueHead p.valueTime p.valueFeature p.historyHead p.historyKey
    p.H p.T p.K p.V p.BT p.BK p.BV p.NT hasSeed storeFinal).eraseDType
  projection := RealHIO.projection p r storage seedType finalType hasSeed storeFinal Bool.false
  read := fun s => ⟨s.readMem r.key,s.readMem r.value,s.readMem r.seed⟩
  valid := fun s => RealHLoop.HistoryNoCollision p (launch s) ∧
    (storeFinal = Bool.true → RealHStore.NoCollision p (launch s) (RealHFinal.base p (launch s)) p.V 1)
  bounds := fun s => RealHSafety.Bounds p (launch s) r hasSeed storeFinal
  history := r.history
  final := r.final
  storeFinal := storeFinal
  historyWindow := fun l c e j => historyAddress p l c (keyIndex p l e.val) (valueIndex p l j.val)
  finalWindow := fun l e j => stateAddress p l (keyIndex p l e.val) (valueIndex p l j.val)
  stateMask := fun l e j => keyIndex p l e.val < p.K ∧ valueIndex p l j.val < p.V
```
</details>

<details><summary><code>forwardStateFormula</code></summary>

```lean
def forwardStateFormula (p : Layout) (hasSeed : Bool) (l : Launch) (a : Inputs) : RealHIO.Outputs p where
  history := fun c e j => forwardStateBefore p hasSeed l a c (keyIndex p l e.val) (valueIndex p l j.val)
  final := fun e j => forwardStateBefore p hasSeed l a p.NT (keyIndex p l e.val) (valueIndex p l j.val)
```
</details>

<details><summary><code>forwardH</code></summary>

```lean
def forwardH (dotStorage : Storage) := forwardHSource dotStorage Bool.false
```
</details>

<details><summary><code>forwardStateBefore</code></summary>

```
/-- Mathematical decayed recurrence before chunk c. Each history tile is
stored before its chunk update. The last ragged chunk uses T % BT for both
its carry decay and signed row decay; the source accessors zero-extend arrays. -/
```
```lean
def forwardStateBefore (p : Layout) (hasSeed : Bool) (l : Launch) (a : Inputs) : Nat → Nat → Nat → ℝ
  | 0, k, v => seed p l a hasSeed k v
  | c+1, k, v => Decay.factor (base p l) (length p c) * forwardStateBefore p hasSeed l a c k v +
      ∑ lane : Fin p.BT, keyValue p l a k (c*p.BT+lane.val) *
        (valueValue p l a (c*p.BT+lane.val) v * Decay.key (base p l) (length p c) lane.val)
```
</details>

<details><summary><code>forwardHSource</code></summary>

```lean
def forwardHSource (dotStorage : Storage) (swapped : Bool) :=
  match dotStorage with
  | .fp32 =>
      if swapped then retentionOpsForwardHSource%[tl.float32, dotLeft]
      else retentionOpsForwardHSource%[tl.float32, original]
  | .fp16 =>
      if swapped then retentionOpsForwardHSource%[tl.float16, dotLeft]
      else retentionOpsForwardHSource%[tl.float16, original]
  | .bf16 =>
      if swapped then retentionOpsForwardHSource%[tl.bfloat16, dotLeft]
      else retentionOpsForwardHSource%[tl.bfloat16, original]
```
</details>

## Public theorem: `forwardStateDotLeft_correct`

**Statement:**
```lean
specification forwardStateDotLeft_correct (p : Layout) (r : Regions) (storage : Storage) (seedType finalType : FloatDType)
    (hasSeed storeFinal : Bool) (hkey : r.key ≠ r.history) (hvalue : r.value ≠ r.history)
    (hfinal : storeFinal = Bool.true → r.history ≠ r.final) :
    forwardStateDotLeft p r storage seedType finalType hasSeed storeFinal ⊨ forwardStateFormula p hasSeed
```

**Assumptions / layout contracts:**
- `hkey : r.key ≠ r.history`
- `hvalue : r.value ≠ r.history`
- `hfinal : storeFinal = Bool.true → r.history ≠ r.final`

**Closed-form spec defs (transitive):** `Storage`, `forwardStateDotLeft`, `forwardStateFormula`, `forwardState`, `forwardHDotLeft`, `forwardStateBefore`, `forwardH`, `forwardHSource`

<details><summary><code>Storage</code></summary>

```lean
abbrev Storage := VeriTile.Bench.TritonBenchG.ChunkRetention.Storage

/-- Source: chunk_retention_fwd_kernel_h. -/
syntax "retentionOpsForwardHSource%" "[" tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(retentionOpsForwardHSource%[$dotTy:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `dotLeft do
      Macro.throwError "expected original or dotLeft"
    let update : TSyntax `tritonStmt ←
      if version.getId == `dotLeft then `(tritonStmt| b_h = tl.dot(b_k, (b_v * d_i[:, None]).to($dotTy:tritonDType), allow_tf32=False) + d_b * b_h)
      else `(tritonStmt| b_h = d_b * b_h + tl.dot(b_k, (b_v * d_i[:, None]).to($dotTy:tritonDType), allow_tf32=False))
    `(fun (dtype out_dtype : FloatDType) (k v h h0 ht : RegionName)
        (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat) (H T K V BT BK BV NT : Nat) (USE_INITIAL_STATE STORE_FINAL_STATE : Bool) =>
      ComputeKernel.mk [k,v,h0] [h,ht] (triton {
        tl.region k = $dotTy, v = $dotTy, h = $dotTy, h0 = DTYPE, ht = OUT_DTYPE
        i_k = tl.program_id(0)
        i_v = tl.program_id(1)
        i_bh = tl.program_id(2)
        i_h = i_bh % $(H)
        b_b = tl.math.log2(1.0 - tl.math.exp2($((-5 : ℝ)) - i_h * 1.0))
        o_i = (tl.arange(0, $(BT))).to(tl.int32)
        d_b = tl.math.exp2($(BT) * b_b)
        d_i = tl.math.exp2(($(BT) - o_i - $(1)) * b_b)
        b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
        if USE_INITIAL_STATE {
          p_h0 = tl.make_block_ptr(base=h0 + ((i_bh * $(K)) * $(V)), shape=($(K), $(V)), strides=($(V), 1), offsets=((i_k * $(BK)), (i_v * $(BV))), block_shape=($(BK), $(BV)), order=(1, 0))
          b_h = (tl.load(p_h0, boundary_check=([0, 1] : List Nat))).to(tl.float32)
        }
        for i_t in range($(NT)) {
          p_k = tl.make_block_ptr(base=k + (i_bh * $(s_qk_h)), shape=($(K), $(T)), strides=($(s_qk_d), $(s_qk_t)), offsets=((i_k * $(BK)), (i_t * $(BT))), block_shape=($(BK), $(BT)), order=(0, 1))
          p_v = tl.make_block_ptr(base=v + (i_bh * $(s_vo_h)), shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)), offsets=((i_t * $(BT)), (i_v * $(BV))), block_shape=($(BT), $(BV)), order=(1, 0))
          p_h = tl.make_block_ptr(base=h + (i_bh * $(s_h_h)) + ((i_t * $(K)) * $(V)), shape=($(K), $(V)), strides=($(s_h_t), 1), offsets=((i_k * $(BK)), (i_v * $(BV))), block_shape=($(BK), $(BV)), order=(1, 0))
          tl.store(p_h, (b_h).to($dotTy:tritonDType), boundary_check=([0, 1] : List Nat))
          b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
          b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
          if i_t == $(NT) - $(1) and $(T) % $(BT) != $(0) {
            d_b = tl.math.exp2(($(T) % $(BT)) * b_b)
            d_i = tl.math.exp2((($(T) % $(BT)) - o_i - $(1)) * b_b)
          }
          $update
        }
        if STORE_FINAL_STATE {
          p_ht = tl.make_block_ptr(base=ht + ((i_bh * $(K)) * $(V)), shape=($(K), $(V)), strides=($(V), 1), offsets=((i_k * $(BK)), (i_v * $(BV))), block_shape=($(BK), $(BV)), order=(1, 0))
          tl.store(p_ht, (b_h).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
        }
      }).surfaceBody)
```
</details>

<details><summary><code>forwardStateDotLeft</code></summary>

```lean
def forwardStateDotLeft (p : Layout) (r : Regions) (storage : Storage) (seedType finalType : FloatDType)
    (hasSeed storeFinal : Bool) : RealHIO.IO p :=
  { forwardState p r storage seedType finalType hasSeed storeFinal with
    kernel := (VeriTile.Bench.TritonBenchG.ChunkRetentionOps.forwardHDotLeft storage seedType finalType r.key r.value r.history r.seed r.final
      p.keyHead p.keyTime p.keyFeature p.valueHead p.valueTime p.valueFeature p.historyHead p.historyKey
      p.H p.T p.K p.V p.BT p.BK p.BV p.NT hasSeed storeFinal).eraseDType
    projection := RealHIO.projection p r storage seedType finalType hasSeed storeFinal Bool.true }
```
</details>

<details><summary><code>forwardStateFormula</code></summary>

```lean
def forwardStateFormula (p : Layout) (hasSeed : Bool) (l : Launch) (a : Inputs) : RealHIO.Outputs p where
  history := fun c e j => forwardStateBefore p hasSeed l a c (keyIndex p l e.val) (valueIndex p l j.val)
  final := fun e j => forwardStateBefore p hasSeed l a p.NT (keyIndex p l e.val) (valueIndex p l j.val)
```
</details>

<details><summary><code>forwardState</code></summary>

```
/-- Active history stores must be injective across chunks and lanes; the
optional final-state store must also be injective. Bounds include the optional
seed/final accesses and every K/V/history access in the full NT loop. -/
```
```lean
def forwardState (p : Layout) (r : Regions) (storage : Storage) (seedType finalType : FloatDType)
    (hasSeed storeFinal : Bool) : RealHIO.IO p where
  kernel := (VeriTile.Bench.TritonBenchG.ChunkRetentionOps.forwardH storage seedType finalType r.key r.value r.history r.seed r.final
    p.keyHead p.keyTime p.keyFeature p.valueHead p.valueTime p.valueFeature p.historyHead p.historyKey
    p.H p.T p.K p.V p.BT p.BK p.BV p.NT hasSeed storeFinal).eraseDType
  projection := RealHIO.projection p r storage seedType finalType hasSeed storeFinal Bool.false
  read := fun s => ⟨s.readMem r.key,s.readMem r.value,s.readMem r.seed⟩
  valid := fun s => RealHLoop.HistoryNoCollision p (launch s) ∧
    (storeFinal = Bool.true → RealHStore.NoCollision p (launch s) (RealHFinal.base p (launch s)) p.V 1)
  bounds := fun s => RealHSafety.Bounds p (launch s) r hasSeed storeFinal
  history := r.history
  final := r.final
  storeFinal := storeFinal
  historyWindow := fun l c e j => historyAddress p l c (keyIndex p l e.val) (valueIndex p l j.val)
  finalWindow := fun l e j => stateAddress p l (keyIndex p l e.val) (valueIndex p l j.val)
  stateMask := fun l e j => keyIndex p l e.val < p.K ∧ valueIndex p l j.val < p.V
```
</details>

<details><summary><code>forwardHDotLeft</code></summary>

```lean
def forwardHDotLeft (dotStorage : Storage) := forwardHSource dotStorage Bool.true

/-- Source: chunk_retention_fwd_kernel_o. -/
syntax "retentionOpsForwardOSource%" "[" tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(retentionOpsForwardOSource%[$dotTy:tritonDType, $vTy:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `scaleFirst do
      Macro.throwError "expected original or scaleFirst"
    let update : TSyntax `tritonStmt ←
      if version.getId == `scaleFirst then `(tritonStmt| b_o = $((scale : ℝ)) * (b_o + tl.dot((b_s).to($vTy:tritonDType), b_v, allow_tf32=False)))
      else `(tritonStmt| b_o = (b_o + tl.dot((b_s).to($vTy:tritonDType), b_v, allow_tf32=False)) * $((scale : ℝ)))
    `(fun (out_dtype : FloatDType) (q k v h o : RegionName)
        (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat) (scale : ℝ) (H T K V BT BK BV : Nat) =>
      ComputeKernel.mk [q,k,v,h] [o] (triton {
        tl.region q = $dotTy, k = $dotTy, h = $dotTy, v = $vTy, o = OUT_DTYPE
        i_v = tl.program_id(0)
        i_t = tl.program_id(1)
        i_bh = tl.program_id(2)
        i_h = i_bh % $(H)
        b_b = tl.math.log2(1.0 - tl.math.exp2($((-5 : ℝ)) - i_h * 1.0))
        o_i = (tl.arange(0, $(BT))).to(tl.int32)
        d_i = tl.math.exp2((o_i + $(1)) * b_b)
        m_s = o_i[:, None] >= o_i[None, :]
        d_s = tl.where(m_s, tl.math.exp2((o_i[:, None] - o_i[None, :]) * b_b), 0)
        b_o = tl.zeros([$(BT), $(BV)], dtype=tl.float32)
        b_s = tl.zeros([$(BT), $(BT)], dtype=tl.float32)
        for i_k in range(tl.cdiv($(K), $(BK))) {
          p_q = tl.make_block_ptr(base=q + (i_bh * $(s_qk_h)), shape=($(T), $(K)), strides=($(s_qk_t), $(s_qk_d)), offsets=((i_t * $(BT)), (i_k * $(BK))), block_shape=($(BT), $(BK)), order=(1, 0))
          p_k = tl.make_block_ptr(base=k + (i_bh * $(s_qk_h)), shape=($(K), $(T)), strides=($(s_qk_d), $(s_qk_t)), offsets=((i_k * $(BK)), (i_t * $(BT))), block_shape=($(BK), $(BT)), order=(0, 1))
          p_h = tl.make_block_ptr(base=h + (i_bh * $(s_h_h)) + ((i_t * $(K)) * $(V)), shape=($(K), $(V)), strides=($(s_h_t), 1), offsets=((i_k * $(BK)), (i_v * $(BV))), block_shape=($(BK), $(BV)), order=(1, 0))
          b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat))
          b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
          b_h = tl.load(p_h, boundary_check=([0, 1] : List Nat))
          b_o += tl.dot((b_q * d_i[:, None]).to($dotTy:tritonDType), b_h, allow_tf32=False)
          b_s += tl.dot(b_q, b_k, allow_tf32=False)
        }
        b_s *= d_s
        p_v = tl.make_block_ptr(base=v + (i_bh * $(s_vo_h)), shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)), offsets=((i_t * $(BT)), (i_v * $(BV))), block_shape=($(BT), $(BV)), order=(1, 0))
        b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
        $update
        p_o = tl.make_block_ptr(base=o + (i_bh * $(s_vo_h)), shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)), offsets=((i_t * $(BT)), (i_v * $(BV))), block_shape=($(BT), $(BV)), order=(1, 0))
        tl.store(p_o, (b_o).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
      }).surfaceBody)
```
</details>

<details><summary><code>forwardStateBefore</code></summary>

```
/-- Mathematical decayed recurrence before chunk c. Each history tile is
stored before its chunk update. The last ragged chunk uses T % BT for both
its carry decay and signed row decay; the source accessors zero-extend arrays. -/
```
```lean
def forwardStateBefore (p : Layout) (hasSeed : Bool) (l : Launch) (a : Inputs) : Nat → Nat → Nat → ℝ
  | 0, k, v => seed p l a hasSeed k v
  | c+1, k, v => Decay.factor (base p l) (length p c) * forwardStateBefore p hasSeed l a c k v +
      ∑ lane : Fin p.BT, keyValue p l a k (c*p.BT+lane.val) *
        (valueValue p l a (c*p.BT+lane.val) v * Decay.key (base p l) (length p c) lane.val)
```
</details>

<details><summary><code>forwardH</code></summary>

```lean
def forwardH (dotStorage : Storage) := forwardHSource dotStorage Bool.false
```
</details>

<details><summary><code>forwardHSource</code></summary>

```lean
def forwardHSource (dotStorage : Storage) (swapped : Bool) :=
  match dotStorage with
  | .fp32 =>
      if swapped then retentionOpsForwardHSource%[tl.float32, dotLeft]
      else retentionOpsForwardHSource%[tl.float32, original]
  | .fp16 =>
      if swapped then retentionOpsForwardHSource%[tl.float16, dotLeft]
      else retentionOpsForwardHSource%[tl.float16, original]
  | .bf16 =>
      if swapped then retentionOpsForwardHSource%[tl.bfloat16, dotLeft]
      else retentionOpsForwardHSource%[tl.bfloat16, original]
```
</details>

## Public theorem: `forwardOutput_correct`

**Statement:**
```lean
specification forwardOutput_correct (p : Layout) (r : Regions) (storage valueType : Storage) (out : FloatDType) (scale : ℝ) :
    forwardOutput p r storage valueType out scale ⊨ forwardOutputFormula p scale
```

**Closed-form spec defs (transitive):** `Storage`, `forwardOutput`, `forwardOutputFormula`, `forwardO`, `forwardOSource`

<details><summary><code>Storage</code></summary>

```lean
abbrev Storage := VeriTile.Bench.TritonBenchG.ChunkRetention.Storage

/-- Source: chunk_retention_fwd_kernel_h. -/
syntax "retentionOpsForwardHSource%" "[" tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(retentionOpsForwardHSource%[$dotTy:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `dotLeft do
      Macro.throwError "expected original or dotLeft"
    let update : TSyntax `tritonStmt ←
      if version.getId == `dotLeft then `(tritonStmt| b_h = tl.dot(b_k, (b_v * d_i[:, None]).to($dotTy:tritonDType), allow_tf32=False) + d_b * b_h)
      else `(tritonStmt| b_h = d_b * b_h + tl.dot(b_k, (b_v * d_i[:, None]).to($dotTy:tritonDType), allow_tf32=False))
    `(fun (dtype out_dtype : FloatDType) (k v h h0 ht : RegionName)
        (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat) (H T K V BT BK BV NT : Nat) (USE_INITIAL_STATE STORE_FINAL_STATE : Bool) =>
      ComputeKernel.mk [k,v,h0] [h,ht] (triton {
        tl.region k = $dotTy, v = $dotTy, h = $dotTy, h0 = DTYPE, ht = OUT_DTYPE
        i_k = tl.program_id(0)
        i_v = tl.program_id(1)
        i_bh = tl.program_id(2)
        i_h = i_bh % $(H)
        b_b = tl.math.log2(1.0 - tl.math.exp2($((-5 : ℝ)) - i_h * 1.0))
        o_i = (tl.arange(0, $(BT))).to(tl.int32)
        d_b = tl.math.exp2($(BT) * b_b)
        d_i = tl.math.exp2(($(BT) - o_i - $(1)) * b_b)
        b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
        if USE_INITIAL_STATE {
          p_h0 = tl.make_block_ptr(base=h0 + ((i_bh * $(K)) * $(V)), shape=($(K), $(V)), strides=($(V), 1), offsets=((i_k * $(BK)), (i_v * $(BV))), block_shape=($(BK), $(BV)), order=(1, 0))
          b_h = (tl.load(p_h0, boundary_check=([0, 1] : List Nat))).to(tl.float32)
        }
        for i_t in range($(NT)) {
          p_k = tl.make_block_ptr(base=k + (i_bh * $(s_qk_h)), shape=($(K), $(T)), strides=($(s_qk_d), $(s_qk_t)), offsets=((i_k * $(BK)), (i_t * $(BT))), block_shape=($(BK), $(BT)), order=(0, 1))
          p_v = tl.make_block_ptr(base=v + (i_bh * $(s_vo_h)), shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)), offsets=((i_t * $(BT)), (i_v * $(BV))), block_shape=($(BT), $(BV)), order=(1, 0))
          p_h = tl.make_block_ptr(base=h + (i_bh * $(s_h_h)) + ((i_t * $(K)) * $(V)), shape=($(K), $(V)), strides=($(s_h_t), 1), offsets=((i_k * $(BK)), (i_v * $(BV))), block_shape=($(BK), $(BV)), order=(1, 0))
          tl.store(p_h, (b_h).to($dotTy:tritonDType), boundary_check=([0, 1] : List Nat))
          b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
          b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
          if i_t == $(NT) - $(1) and $(T) % $(BT) != $(0) {
            d_b = tl.math.exp2(($(T) % $(BT)) * b_b)
            d_i = tl.math.exp2((($(T) % $(BT)) - o_i - $(1)) * b_b)
          }
          $update
        }
        if STORE_FINAL_STATE {
          p_ht = tl.make_block_ptr(base=ht + ((i_bh * $(K)) * $(V)), shape=($(K), $(V)), strides=($(V), 1), offsets=((i_k * $(BK)), (i_v * $(BV))), block_shape=($(BK), $(BV)), order=(1, 0))
          tl.store(p_ht, (b_h).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
        }
      }).surfaceBody)
```
</details>

<details><summary><code>forwardOutput</code></summary>

```
/-- Bounds cover every Q/K/H tile, the V tile and output. Output addresses
must be injective among active lanes. Inputs may alias output: all loads
precede its sole store. Both feature strides stay independent. -/
```
```lean
def forwardOutput (p : Layout) (r : Regions) (storage valueType : Storage) (out : FloatDType) (scale : ℝ) :
    IO Launch Inputs (TileIndex [p.BT,p.BV]) where
  kernel := (VeriTile.Bench.TritonBenchG.ChunkRetentionOps.forwardO storage valueType out r.query r.key r.value r.history r.output
    p.keyHead p.keyTime p.keyDimension p.valueHead p.valueTime p.valueDimension p.historyHead p.historyKey
    scale p.H p.T p.K p.V p.BT p.BK p.BV).eraseDType
  projection := OutputIO.projection p r storage valueType out scale Bool.false
  grid := fun s => ⟨s.pids 0,s.pids 1,s.pids 2⟩
  read := fun s => ⟨s.readMem r.query,s.readMem r.key,s.readMem r.value,s.readMem r.history⟩
  valid := fun s => NoCollision p (launch s)
  bounds := fun s => OutputSafe.Bounds p (launch s) r
  out := r.output
  write := fun l i => l.head*p.valueHead+(l.chunk*p.BT+i.1.val)*p.valueTime+(l.value*p.BV+i.2.1.val)*p.valueDimension
  mask := fun l i => l.chunk*p.BT+i.1.val < p.T ∧ l.value*p.BV+i.2.1.val < p.V
```
</details>

<details><summary><code>forwardOutputFormula</code></summary>

```
/-- Row-decayed Q·H plus causal, distance-decayed Q·K·V, then scale.
The head base is log2(1 - 2^(-5 - head % H)). All sums read initial arrays. -/
```
```lean
def forwardOutputFormula (p : Layout) (scale : ℝ) (l : Launch) (a : Inputs) (i : TileIndex [p.BT,p.BV]) : ℝ :=
  ((∑ kb ∈ Finset.range (numBlocks p), ∑ e : Fin p.BK,
      (query p l a i.1.val (kb*p.BK+e.val) * Decay.query (base p l) i.1.val) *
        history p l a (kb*p.BK+e.val) i.2.1.val) +
    ∑ j : Fin p.BT,
      ((∑ kb ∈ Finset.range (numBlocks p), ∑ e : Fin p.BK,
        query p l a i.1.val (kb*p.BK+e.val)*key p l a (kb*p.BK+e.val) j.val) *
        Decay.score (base p l) i.1.val j.val) * value p l a j.val i.2.1.val) * scale
```
</details>

<details><summary><code>forwardO</code></summary>

```lean
def forwardO (dotStorage : Storage) (valueStorage : Storage) := forwardOSource dotStorage valueStorage Bool.false
```
</details>

<details><summary><code>forwardOSource</code></summary>

```lean
def forwardOSource (dotStorage : Storage) (valueStorage : Storage) (swapped : Bool) :=
  match dotStorage,valueStorage with
  | .fp32, .fp32 =>
      if swapped then retentionOpsForwardOSource%[tl.float32, tl.float32, scaleFirst]
      else retentionOpsForwardOSource%[tl.float32, tl.float32, original]
  | .fp32, .fp16 =>
      if swapped then retentionOpsForwardOSource%[tl.float32, tl.float16, scaleFirst]
      else retentionOpsForwardOSource%[tl.float32, tl.float16, original]
  | .fp32, .bf16 =>
      if swapped then retentionOpsForwardOSource%[tl.float32, tl.bfloat16, scaleFirst]
      else retentionOpsForwardOSource%[tl.float32, tl.bfloat16, original]
  | .fp16, .fp32 =>
      if swapped then retentionOpsForwardOSource%[tl.float16, tl.float32, scaleFirst]
      else retentionOpsForwardOSource%[tl.float16, tl.float32, original]
  | .fp16, .fp16 =>
      if swapped then retentionOpsForwardOSource%[tl.float16, tl.float16, scaleFirst]
      else retentionOpsForwardOSource%[tl.float16, tl.float16, original]
  | .fp16, .bf16 =>
      if swapped then retentionOpsForwardOSource%[tl.float16, tl.bfloat16, scaleFirst]
      else retentionOpsForwardOSource%[tl.float16, tl.bfloat16, original]
  | .bf16, .fp32 =>
      if swapped then retentionOpsForwardOSource%[tl.bfloat16, tl.float32, scaleFirst]
      else retentionOpsForwardOSource%[tl.bfloat16, tl.float32, original]
  | .bf16, .fp16 =>
      if swapped then retentionOpsForwardOSource%[tl.bfloat16, tl.float16, scaleFirst]
      else retentionOpsForwardOSource%[tl.bfloat16, tl.float16, original]
  | .bf16, .bf16 =>
      if swapped then retentionOpsForwardOSource%[tl.bfloat16, tl.bfloat16, scaleFirst]
      else retentionOpsForwardOSource%[tl.bfloat16, tl.bfloat16, original]
```
</details>

## Public theorem: `forwardOutputScaleFirst_correct`

**Statement:**
```lean
specification forwardOutputScaleFirst_correct (p : Layout) (r : Regions) (storage valueType : Storage) (out : FloatDType) (scale : ℝ) :
    forwardOutputScaleFirst p r storage valueType out scale ⊨ forwardOutputFormula p scale
```

**Closed-form spec defs (transitive):** `Storage`, `forwardOutputScaleFirst`, `forwardOutputFormula`, `forwardOutput`, `forwardOScaleFirst`, `forwardO`, `forwardOSource`

<details><summary><code>Storage</code></summary>

```lean
abbrev Storage := VeriTile.Bench.TritonBenchG.ChunkRetention.Storage

/-- Source: chunk_retention_fwd_kernel_h. -/
syntax "retentionOpsForwardHSource%" "[" tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(retentionOpsForwardHSource%[$dotTy:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `dotLeft do
      Macro.throwError "expected original or dotLeft"
    let update : TSyntax `tritonStmt ←
      if version.getId == `dotLeft then `(tritonStmt| b_h = tl.dot(b_k, (b_v * d_i[:, None]).to($dotTy:tritonDType), allow_tf32=False) + d_b * b_h)
      else `(tritonStmt| b_h = d_b * b_h + tl.dot(b_k, (b_v * d_i[:, None]).to($dotTy:tritonDType), allow_tf32=False))
    `(fun (dtype out_dtype : FloatDType) (k v h h0 ht : RegionName)
        (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat) (H T K V BT BK BV NT : Nat) (USE_INITIAL_STATE STORE_FINAL_STATE : Bool) =>
      ComputeKernel.mk [k,v,h0] [h,ht] (triton {
        tl.region k = $dotTy, v = $dotTy, h = $dotTy, h0 = DTYPE, ht = OUT_DTYPE
        i_k = tl.program_id(0)
        i_v = tl.program_id(1)
        i_bh = tl.program_id(2)
        i_h = i_bh % $(H)
        b_b = tl.math.log2(1.0 - tl.math.exp2($((-5 : ℝ)) - i_h * 1.0))
        o_i = (tl.arange(0, $(BT))).to(tl.int32)
        d_b = tl.math.exp2($(BT) * b_b)
        d_i = tl.math.exp2(($(BT) - o_i - $(1)) * b_b)
        b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
        if USE_INITIAL_STATE {
          p_h0 = tl.make_block_ptr(base=h0 + ((i_bh * $(K)) * $(V)), shape=($(K), $(V)), strides=($(V), 1), offsets=((i_k * $(BK)), (i_v * $(BV))), block_shape=($(BK), $(BV)), order=(1, 0))
          b_h = (tl.load(p_h0, boundary_check=([0, 1] : List Nat))).to(tl.float32)
        }
        for i_t in range($(NT)) {
          p_k = tl.make_block_ptr(base=k + (i_bh * $(s_qk_h)), shape=($(K), $(T)), strides=($(s_qk_d), $(s_qk_t)), offsets=((i_k * $(BK)), (i_t * $(BT))), block_shape=($(BK), $(BT)), order=(0, 1))
          p_v = tl.make_block_ptr(base=v + (i_bh * $(s_vo_h)), shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)), offsets=((i_t * $(BT)), (i_v * $(BV))), block_shape=($(BT), $(BV)), order=(1, 0))
          p_h = tl.make_block_ptr(base=h + (i_bh * $(s_h_h)) + ((i_t * $(K)) * $(V)), shape=($(K), $(V)), strides=($(s_h_t), 1), offsets=((i_k * $(BK)), (i_v * $(BV))), block_shape=($(BK), $(BV)), order=(1, 0))
          tl.store(p_h, (b_h).to($dotTy:tritonDType), boundary_check=([0, 1] : List Nat))
          b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
          b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
          if i_t == $(NT) - $(1) and $(T) % $(BT) != $(0) {
            d_b = tl.math.exp2(($(T) % $(BT)) * b_b)
            d_i = tl.math.exp2((($(T) % $(BT)) - o_i - $(1)) * b_b)
          }
          $update
        }
        if STORE_FINAL_STATE {
          p_ht = tl.make_block_ptr(base=ht + ((i_bh * $(K)) * $(V)), shape=($(K), $(V)), strides=($(V), 1), offsets=((i_k * $(BK)), (i_v * $(BV))), block_shape=($(BK), $(BV)), order=(1, 0))
          tl.store(p_ht, (b_h).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
        }
      }).surfaceBody)
```
</details>

<details><summary><code>forwardOutputScaleFirst</code></summary>

```lean
def forwardOutputScaleFirst (p : Layout) (r : Regions) (storage valueType : Storage) (out : FloatDType) (scale : ℝ) :
    IO Launch Inputs (TileIndex [p.BT,p.BV]) :=
  { forwardOutput p r storage valueType out scale with
    kernel := (VeriTile.Bench.TritonBenchG.ChunkRetentionOps.forwardOScaleFirst storage valueType out r.query r.key r.value r.history r.output
      p.keyHead p.keyTime p.keyDimension p.valueHead p.valueTime p.valueDimension p.historyHead p.historyKey
      scale p.H p.T p.K p.V p.BT p.BK p.BV).eraseDType
    projection := OutputIO.projection p r storage valueType out scale Bool.true }
```
</details>

<details><summary><code>forwardOutputFormula</code></summary>

```
/-- Row-decayed Q·H plus causal, distance-decayed Q·K·V, then scale.
The head base is log2(1 - 2^(-5 - head % H)). All sums read initial arrays. -/
```
```lean
def forwardOutputFormula (p : Layout) (scale : ℝ) (l : Launch) (a : Inputs) (i : TileIndex [p.BT,p.BV]) : ℝ :=
  ((∑ kb ∈ Finset.range (numBlocks p), ∑ e : Fin p.BK,
      (query p l a i.1.val (kb*p.BK+e.val) * Decay.query (base p l) i.1.val) *
        history p l a (kb*p.BK+e.val) i.2.1.val) +
    ∑ j : Fin p.BT,
      ((∑ kb ∈ Finset.range (numBlocks p), ∑ e : Fin p.BK,
        query p l a i.1.val (kb*p.BK+e.val)*key p l a (kb*p.BK+e.val) j.val) *
        Decay.score (base p l) i.1.val j.val) * value p l a j.val i.2.1.val) * scale
```
</details>

<details><summary><code>forwardOutput</code></summary>

```
/-- Bounds cover every Q/K/H tile, the V tile and output. Output addresses
must be injective among active lanes. Inputs may alias output: all loads
precede its sole store. Both feature strides stay independent. -/
```
```lean
def forwardOutput (p : Layout) (r : Regions) (storage valueType : Storage) (out : FloatDType) (scale : ℝ) :
    IO Launch Inputs (TileIndex [p.BT,p.BV]) where
  kernel := (VeriTile.Bench.TritonBenchG.ChunkRetentionOps.forwardO storage valueType out r.query r.key r.value r.history r.output
    p.keyHead p.keyTime p.keyDimension p.valueHead p.valueTime p.valueDimension p.historyHead p.historyKey
    scale p.H p.T p.K p.V p.BT p.BK p.BV).eraseDType
  projection := OutputIO.projection p r storage valueType out scale Bool.false
  grid := fun s => ⟨s.pids 0,s.pids 1,s.pids 2⟩
  read := fun s => ⟨s.readMem r.query,s.readMem r.key,s.readMem r.value,s.readMem r.history⟩
  valid := fun s => NoCollision p (launch s)
  bounds := fun s => OutputSafe.Bounds p (launch s) r
  out := r.output
  write := fun l i => l.head*p.valueHead+(l.chunk*p.BT+i.1.val)*p.valueTime+(l.value*p.BV+i.2.1.val)*p.valueDimension
  mask := fun l i => l.chunk*p.BT+i.1.val < p.T ∧ l.value*p.BV+i.2.1.val < p.V
```
</details>

<details><summary><code>forwardOScaleFirst</code></summary>

```lean
def forwardOScaleFirst (dotStorage : Storage) (valueStorage : Storage) := forwardOSource dotStorage valueStorage Bool.true

/-- Source: chunk_retention_bwd_kernel_dh. -/
syntax "retentionOpsBackwardHSource%" "[" tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(retentionOpsBackwardHSource%[$dotTy:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `dotLeft do
      Macro.throwError "expected original or dotLeft"
    let update : TSyntax `tritonStmt ←
      if version.getId == `dotLeft then `(tritonStmt| b_dh = tl.dot(b_q, (b_do * d_i[:, None]).to($dotTy:tritonDType), allow_tf32=False) + d_b * b_dh)
      else `(tritonStmt| b_dh = d_b * b_dh + tl.dot(b_q, (b_do * d_i[:, None]).to($dotTy:tritonDType), allow_tf32=False))
    `(fun (out_dtype : FloatDType) (q do_ dh : RegionName)
        (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat) (scale : ℝ) (H T K V BT BK BV NT : Nat) =>
      ComputeKernel.mk [q,do_] [dh] (triton {
        tl.region q = $dotTy, do_ = $dotTy, dh = OUT_DTYPE
        i_k = tl.program_id(0)
        i_v = tl.program_id(1)
        i_bh = tl.program_id(2)
        i_h = i_bh % $(H)
        b_b = tl.math.log2(1.0 - tl.math.exp2($((-5 : ℝ)) - i_h * 1.0))
        o_i = (tl.arange(0, $(BT))).to(tl.int32)
        d_b = tl.math.exp2($(BT) * b_b)
        d_i = tl.math.exp2((o_i + $(1)) * b_b)
        b_dh = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
        for j in range($(0), $(NT), $(1)) {
          i_t = $(NT) - $(1) - j
          p_q = tl.make_block_ptr(base=q + (i_bh * $(s_qk_h)), shape=($(K), $(T)), strides=($(s_qk_d), $(s_qk_t)), offsets=((i_k * $(BK)), (i_t * $(BT))), block_shape=($(BK), $(BT)), order=(0, 1))
          p_do = tl.make_block_ptr(base=do_ + (i_bh * $(s_vo_h)), shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)), offsets=((i_t * $(BT)), (i_v * $(BV))), block_shape=($(BT), $(BV)), order=(1, 0))
          p_dh = tl.make_block_ptr(base=dh + (i_bh * $(s_h_h)) + ((i_t * $(K)) * $(V)), shape=($(K), $(V)), strides=($(s_h_t), 1), offsets=((i_k * $(BK)), (i_v * $(BV))), block_shape=($(BK), $(BV)), order=(1, 0))
          tl.store(p_dh, (b_dh).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
          b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat))
          b_q = (b_q * $((scale : ℝ))).to($dotTy:tritonDType)
          b_do = tl.load(p_do, boundary_check=([0, 1] : List Nat))
          $update
        }
      }).surfaceBody)
```
</details>

<details><summary><code>forwardO</code></summary>

```lean
def forwardO (dotStorage : Storage) (valueStorage : Storage) := forwardOSource dotStorage valueStorage Bool.false
```
</details>

<details><summary><code>forwardOSource</code></summary>

```lean
def forwardOSource (dotStorage : Storage) (valueStorage : Storage) (swapped : Bool) :=
  match dotStorage,valueStorage with
  | .fp32, .fp32 =>
      if swapped then retentionOpsForwardOSource%[tl.float32, tl.float32, scaleFirst]
      else retentionOpsForwardOSource%[tl.float32, tl.float32, original]
  | .fp32, .fp16 =>
      if swapped then retentionOpsForwardOSource%[tl.float32, tl.float16, scaleFirst]
      else retentionOpsForwardOSource%[tl.float32, tl.float16, original]
  | .fp32, .bf16 =>
      if swapped then retentionOpsForwardOSource%[tl.float32, tl.bfloat16, scaleFirst]
      else retentionOpsForwardOSource%[tl.float32, tl.bfloat16, original]
  | .fp16, .fp32 =>
      if swapped then retentionOpsForwardOSource%[tl.float16, tl.float32, scaleFirst]
      else retentionOpsForwardOSource%[tl.float16, tl.float32, original]
  | .fp16, .fp16 =>
      if swapped then retentionOpsForwardOSource%[tl.float16, tl.float16, scaleFirst]
      else retentionOpsForwardOSource%[tl.float16, tl.float16, original]
  | .fp16, .bf16 =>
      if swapped then retentionOpsForwardOSource%[tl.float16, tl.bfloat16, scaleFirst]
      else retentionOpsForwardOSource%[tl.float16, tl.bfloat16, original]
  | .bf16, .fp32 =>
      if swapped then retentionOpsForwardOSource%[tl.bfloat16, tl.float32, scaleFirst]
      else retentionOpsForwardOSource%[tl.bfloat16, tl.float32, original]
  | .bf16, .fp16 =>
      if swapped then retentionOpsForwardOSource%[tl.bfloat16, tl.float16, scaleFirst]
      else retentionOpsForwardOSource%[tl.bfloat16, tl.float16, original]
  | .bf16, .bf16 =>
      if swapped then retentionOpsForwardOSource%[tl.bfloat16, tl.bfloat16, scaleFirst]
      else retentionOpsForwardOSource%[tl.bfloat16, tl.bfloat16, original]
```
</details>

## Public theorem: `backwardState_correct`

**Statement:**
```lean
specification backwardState_correct (p : Layout) (r : Regions) (storage : Storage) (out : FloatDType) (scale : ℝ)
    (hq : r.query ≠ r.history) (hd : r.incoming ≠ r.history) :
    backwardState p r storage out scale ⊨ backwardStateFormula p scale
```

**Assumptions / layout contracts:**
- `hq : r.query ≠ r.history`
- `hd : r.incoming ≠ r.history`

**Closed-form spec defs (transitive):** `Storage`, `backwardState`, `backwardStateFormula`, `backwardH`, `backwardStateAfter`, `backwardHSource`

<details><summary><code>Storage</code></summary>

```lean
abbrev Storage := VeriTile.Bench.TritonBenchG.ChunkRetention.Storage

/-- Source: chunk_retention_fwd_kernel_h. -/
syntax "retentionOpsForwardHSource%" "[" tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(retentionOpsForwardHSource%[$dotTy:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `dotLeft do
      Macro.throwError "expected original or dotLeft"
    let update : TSyntax `tritonStmt ←
      if version.getId == `dotLeft then `(tritonStmt| b_h = tl.dot(b_k, (b_v * d_i[:, None]).to($dotTy:tritonDType), allow_tf32=False) + d_b * b_h)
      else `(tritonStmt| b_h = d_b * b_h + tl.dot(b_k, (b_v * d_i[:, None]).to($dotTy:tritonDType), allow_tf32=False))
    `(fun (dtype out_dtype : FloatDType) (k v h h0 ht : RegionName)
        (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat) (H T K V BT BK BV NT : Nat) (USE_INITIAL_STATE STORE_FINAL_STATE : Bool) =>
      ComputeKernel.mk [k,v,h0] [h,ht] (triton {
        tl.region k = $dotTy, v = $dotTy, h = $dotTy, h0 = DTYPE, ht = OUT_DTYPE
        i_k = tl.program_id(0)
        i_v = tl.program_id(1)
        i_bh = tl.program_id(2)
        i_h = i_bh % $(H)
        b_b = tl.math.log2(1.0 - tl.math.exp2($((-5 : ℝ)) - i_h * 1.0))
        o_i = (tl.arange(0, $(BT))).to(tl.int32)
        d_b = tl.math.exp2($(BT) * b_b)
        d_i = tl.math.exp2(($(BT) - o_i - $(1)) * b_b)
        b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
        if USE_INITIAL_STATE {
          p_h0 = tl.make_block_ptr(base=h0 + ((i_bh * $(K)) * $(V)), shape=($(K), $(V)), strides=($(V), 1), offsets=((i_k * $(BK)), (i_v * $(BV))), block_shape=($(BK), $(BV)), order=(1, 0))
          b_h = (tl.load(p_h0, boundary_check=([0, 1] : List Nat))).to(tl.float32)
        }
        for i_t in range($(NT)) {
          p_k = tl.make_block_ptr(base=k + (i_bh * $(s_qk_h)), shape=($(K), $(T)), strides=($(s_qk_d), $(s_qk_t)), offsets=((i_k * $(BK)), (i_t * $(BT))), block_shape=($(BK), $(BT)), order=(0, 1))
          p_v = tl.make_block_ptr(base=v + (i_bh * $(s_vo_h)), shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)), offsets=((i_t * $(BT)), (i_v * $(BV))), block_shape=($(BT), $(BV)), order=(1, 0))
          p_h = tl.make_block_ptr(base=h + (i_bh * $(s_h_h)) + ((i_t * $(K)) * $(V)), shape=($(K), $(V)), strides=($(s_h_t), 1), offsets=((i_k * $(BK)), (i_v * $(BV))), block_shape=($(BK), $(BV)), order=(1, 0))
          tl.store(p_h, (b_h).to($dotTy:tritonDType), boundary_check=([0, 1] : List Nat))
          b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
          b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
          if i_t == $(NT) - $(1) and $(T) % $(BT) != $(0) {
            d_b = tl.math.exp2(($(T) % $(BT)) * b_b)
            d_i = tl.math.exp2((($(T) % $(BT)) - o_i - $(1)) * b_b)
          }
          $update
        }
        if STORE_FINAL_STATE {
          p_ht = tl.make_block_ptr(base=ht + ((i_bh * $(K)) * $(V)), shape=($(K), $(V)), strides=($(V), 1), offsets=((i_k * $(BK)), (i_v * $(BV))), block_shape=($(BK), $(BV)), order=(1, 0))
          tl.store(p_ht, (b_h).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
        }
      }).surfaceBody)
```
</details>

<details><summary><code>backwardState</code></summary>

```
/-- Every history store must be injective across chunks and lanes. Bounds
cover the actual Q/dO loads and history stores. The reverse loop may be empty. -/
```
```lean
def backwardState (p : Layout) (r : Regions) (storage : Storage) (out : FloatDType) (scale : ℝ) : RealDhIO.IO p where
  kernel := (VeriTile.Bench.TritonBenchG.ChunkRetentionOps.backwardH storage out r.query r.incoming r.history
    p.keyHead p.keyTime p.keyFeature p.valueHead p.valueTime p.valueFeature p.historyHead p.historyKey
    scale p.H p.T p.K p.V p.BT p.BK p.BV p.NT).eraseDType
  projection := RealDhIO.projection p r storage out scale Bool.false
  read := fun s => ⟨s.readMem r.query,s.readMem r.incoming⟩
  valid := fun s => RealDhLoop.HistoryNoCollision p (launch s)
  bounds := fun s => RealDhSafety.Bounds p (launch s) r
  output := r.history
  window := fun l c e j => historyAddress p l c (keyIndex p l e.val) (valueIndex p l j.val)
  mask := fun l e j => keyIndex p l e.val < p.K ∧ valueIndex p l j.val < p.V
```
</details>

<details><summary><code>backwardStateFormula</code></summary>

```lean
def backwardStateFormula (p : Layout) (scale : ℝ) (l : Launch) (a : Inputs) : RealDhIO.Values p :=
  fun c e j => backwardStateAfter p scale l a (p.NT-1-c) (keyIndex p l e.val) (valueIndex p l j.val)
```
</details>

<details><summary><code>backwardH</code></summary>

```lean
def backwardH (dotStorage : Storage) := backwardHSource dotStorage Bool.false
```
</details>

<details><summary><code>backwardStateAfter</code></summary>

```
/-- Carry after n reverse-time updates of initial Q/dO. A history tile is
stored before its update; unlike forward state, all backward decay lengths
remain BT, including the partial last chunk. -/
```
```lean
def backwardStateAfter (p : Layout) (scale : ℝ) (l : Launch) (a : Inputs) : Nat → Nat → Nat → ℝ
  | 0, _, _ => 0
  | n+1, k, v => Decay.factor (RealHMath.base p l) p.BT * backwardStateAfter p scale l a n k v +
      ∑ t : Fin p.BT,
        (query p l a k (timeIndex p (p.NT-1-n) t.val) * scale) *
          (incoming p l a (timeIndex p (p.NT-1-n) t.val) v * Decay.query (RealHMath.base p l) t.val)
```
</details>

<details><summary><code>backwardHSource</code></summary>

```lean
def backwardHSource (dotStorage : Storage) (swapped : Bool) :=
  match dotStorage with
  | .fp32 => if swapped then retentionOpsBackwardHSource%[tl.float32, dotLeft] else retentionOpsBackwardHSource%[tl.float32, original]
  | .fp16 => if swapped then retentionOpsBackwardHSource%[tl.float16, dotLeft] else retentionOpsBackwardHSource%[tl.float16, original]
  | .bf16 => if swapped then retentionOpsBackwardHSource%[tl.bfloat16, dotLeft] else retentionOpsBackwardHSource%[tl.bfloat16, original]
```
</details>

## Public theorem: `backwardStateDotLeft_correct`

**Statement:**
```lean
specification backwardStateDotLeft_correct (p : Layout) (r : Regions) (storage : Storage) (out : FloatDType) (scale : ℝ)
    (hq : r.query ≠ r.history) (hd : r.incoming ≠ r.history) :
    backwardStateDotLeft p r storage out scale ⊨ backwardStateFormula p scale
```

**Assumptions / layout contracts:**
- `hq : r.query ≠ r.history`
- `hd : r.incoming ≠ r.history`

**Closed-form spec defs (transitive):** `Storage`, `backwardStateDotLeft`, `backwardStateFormula`, `backwardState`, `backwardHDotLeft`, `backwardStateAfter`, `backwardH`, `backwardHSource`

<details><summary><code>Storage</code></summary>

```lean
abbrev Storage := VeriTile.Bench.TritonBenchG.ChunkRetention.Storage

/-- Source: chunk_retention_fwd_kernel_h. -/
syntax "retentionOpsForwardHSource%" "[" tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(retentionOpsForwardHSource%[$dotTy:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `dotLeft do
      Macro.throwError "expected original or dotLeft"
    let update : TSyntax `tritonStmt ←
      if version.getId == `dotLeft then `(tritonStmt| b_h = tl.dot(b_k, (b_v * d_i[:, None]).to($dotTy:tritonDType), allow_tf32=False) + d_b * b_h)
      else `(tritonStmt| b_h = d_b * b_h + tl.dot(b_k, (b_v * d_i[:, None]).to($dotTy:tritonDType), allow_tf32=False))
    `(fun (dtype out_dtype : FloatDType) (k v h h0 ht : RegionName)
        (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat) (H T K V BT BK BV NT : Nat) (USE_INITIAL_STATE STORE_FINAL_STATE : Bool) =>
      ComputeKernel.mk [k,v,h0] [h,ht] (triton {
        tl.region k = $dotTy, v = $dotTy, h = $dotTy, h0 = DTYPE, ht = OUT_DTYPE
        i_k = tl.program_id(0)
        i_v = tl.program_id(1)
        i_bh = tl.program_id(2)
        i_h = i_bh % $(H)
        b_b = tl.math.log2(1.0 - tl.math.exp2($((-5 : ℝ)) - i_h * 1.0))
        o_i = (tl.arange(0, $(BT))).to(tl.int32)
        d_b = tl.math.exp2($(BT) * b_b)
        d_i = tl.math.exp2(($(BT) - o_i - $(1)) * b_b)
        b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
        if USE_INITIAL_STATE {
          p_h0 = tl.make_block_ptr(base=h0 + ((i_bh * $(K)) * $(V)), shape=($(K), $(V)), strides=($(V), 1), offsets=((i_k * $(BK)), (i_v * $(BV))), block_shape=($(BK), $(BV)), order=(1, 0))
          b_h = (tl.load(p_h0, boundary_check=([0, 1] : List Nat))).to(tl.float32)
        }
        for i_t in range($(NT)) {
          p_k = tl.make_block_ptr(base=k + (i_bh * $(s_qk_h)), shape=($(K), $(T)), strides=($(s_qk_d), $(s_qk_t)), offsets=((i_k * $(BK)), (i_t * $(BT))), block_shape=($(BK), $(BT)), order=(0, 1))
          p_v = tl.make_block_ptr(base=v + (i_bh * $(s_vo_h)), shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)), offsets=((i_t * $(BT)), (i_v * $(BV))), block_shape=($(BT), $(BV)), order=(1, 0))
          p_h = tl.make_block_ptr(base=h + (i_bh * $(s_h_h)) + ((i_t * $(K)) * $(V)), shape=($(K), $(V)), strides=($(s_h_t), 1), offsets=((i_k * $(BK)), (i_v * $(BV))), block_shape=($(BK), $(BV)), order=(1, 0))
          tl.store(p_h, (b_h).to($dotTy:tritonDType), boundary_check=([0, 1] : List Nat))
          b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
          b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
          if i_t == $(NT) - $(1) and $(T) % $(BT) != $(0) {
            d_b = tl.math.exp2(($(T) % $(BT)) * b_b)
            d_i = tl.math.exp2((($(T) % $(BT)) - o_i - $(1)) * b_b)
          }
          $update
        }
        if STORE_FINAL_STATE {
          p_ht = tl.make_block_ptr(base=ht + ((i_bh * $(K)) * $(V)), shape=($(K), $(V)), strides=($(V), 1), offsets=((i_k * $(BK)), (i_v * $(BV))), block_shape=($(BK), $(BV)), order=(1, 0))
          tl.store(p_ht, (b_h).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
        }
      }).surfaceBody)
```
</details>

<details><summary><code>backwardStateDotLeft</code></summary>

```lean
def backwardStateDotLeft (p : Layout) (r : Regions) (storage : Storage) (out : FloatDType) (scale : ℝ) : RealDhIO.IO p :=
  { backwardState p r storage out scale with
    kernel := (VeriTile.Bench.TritonBenchG.ChunkRetentionOps.backwardHDotLeft storage out r.query r.incoming r.history
      p.keyHead p.keyTime p.keyFeature p.valueHead p.valueTime p.valueFeature p.historyHead p.historyKey
      scale p.H p.T p.K p.V p.BT p.BK p.BV p.NT).eraseDType
    projection := RealDhIO.projection p r storage out scale Bool.true }
```
</details>

<details><summary><code>backwardStateFormula</code></summary>

```lean
def backwardStateFormula (p : Layout) (scale : ℝ) (l : Launch) (a : Inputs) : RealDhIO.Values p :=
  fun c e j => backwardStateAfter p scale l a (p.NT-1-c) (keyIndex p l e.val) (valueIndex p l j.val)
```
</details>

<details><summary><code>backwardState</code></summary>

```
/-- Every history store must be injective across chunks and lanes. Bounds
cover the actual Q/dO loads and history stores. The reverse loop may be empty. -/
```
```lean
def backwardState (p : Layout) (r : Regions) (storage : Storage) (out : FloatDType) (scale : ℝ) : RealDhIO.IO p where
  kernel := (VeriTile.Bench.TritonBenchG.ChunkRetentionOps.backwardH storage out r.query r.incoming r.history
    p.keyHead p.keyTime p.keyFeature p.valueHead p.valueTime p.valueFeature p.historyHead p.historyKey
    scale p.H p.T p.K p.V p.BT p.BK p.BV p.NT).eraseDType
  projection := RealDhIO.projection p r storage out scale Bool.false
  read := fun s => ⟨s.readMem r.query,s.readMem r.incoming⟩
  valid := fun s => RealDhLoop.HistoryNoCollision p (launch s)
  bounds := fun s => RealDhSafety.Bounds p (launch s) r
  output := r.history
  window := fun l c e j => historyAddress p l c (keyIndex p l e.val) (valueIndex p l j.val)
  mask := fun l e j => keyIndex p l e.val < p.K ∧ valueIndex p l j.val < p.V
```
</details>

<details><summary><code>backwardHDotLeft</code></summary>

```lean
def backwardHDotLeft (dotStorage : Storage) := backwardHSource dotStorage Bool.true

/-- Source: chunk_retention_bwd_kernel_dqkv. -/
syntax "retentionOpsBackwardQKVSource%" "[" tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(retentionOpsBackwardQKVSource%[$dotTy:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `dotLeft do
      Macro.throwError "expected original or dotLeft"
    let update : TSyntax `tritonStmt ←
      if version.getId == `dotLeft then `(tritonStmt| b_dq = tl.dot(b_ds, b_k, allow_tf32=False) + b_dq * d_q[:, None])
      else `(tritonStmt| b_dq = b_dq * d_q[:, None] + tl.dot(b_ds, b_k, allow_tf32=False))
    `(fun  (q k v h do_ dh dq dk dv : RegionName)
        (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat) (scale : ℝ) (H T K V BT BK BV NT : Nat) =>
      ComputeKernel.mk [q,k,v,h,do_,dh] [dq,dk,dv] (triton {
        tl.region q = $dotTy, k = $dotTy, v = $dotTy, h = $dotTy, do_ = $dotTy, dh = $dotTy, dq = $dotTy, dk = $dotTy, dv = $dotTy
        i_k = tl.program_id(0)
        i_t = tl.program_id(1)
        i_bh = tl.program_id(2)
        i_h = i_bh % $(H)
        n_bh = tl.num_programs(2)
        b_b = tl.math.log2(1.0 - tl.math.exp2($((-5 : ℝ)) - i_h * 1.0))
        o_i = (tl.arange(0, $(BT))).to(tl.int32)
        d_q = tl.math.exp2((o_i + $(1)) * b_b)
        d_k = tl.math.exp2(($(BT) - o_i - $(1)) * b_b)
        d_q = (d_q * $((scale : ℝ))).to(tl.float32)
        m_s = o_i[:, None] >= o_i[None, :]
        d_s = tl.where(m_s, tl.math.exp2((o_i[:, None] - o_i[None, :]) * b_b), 0) * $((scale : ℝ))
        p_q = tl.make_block_ptr(base=q + (i_bh * $(s_qk_h)), shape=($(K), $(T)), strides=($(s_qk_d), $(s_qk_t)), offsets=((i_k * $(BK)), (i_t * $(BT))), block_shape=($(BK), $(BT)), order=(0, 1))
        p_k = tl.make_block_ptr(base=k + (i_bh * $(s_qk_h)), shape=($(T), $(K)), strides=($(s_qk_t), $(s_qk_d)), offsets=((i_t * $(BT)), (i_k * $(BK))), block_shape=($(BT), $(BK)), order=(1, 0))
        b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat))
        b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
        b_s = (tl.dot(b_k, b_q, allow_tf32=False) * tl.trans(d_s))
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
          b_dq += tl.dot(b_do, b_h, allow_tf32=False)
          b_dk += tl.dot(b_v, tl.trans(b_dh), allow_tf32=False)
          b_dv = (tl.dot(b_k, b_dh, allow_tf32=False) * d_k[:, None] + tl.dot((b_s).to($dotTy:tritonDType), b_do, allow_tf32=False))
          tl.store(p_dv, (b_dv).to($dotTy:tritonDType), boundary_check=([0, 1] : List Nat))
        }
        b_ds = (b_ds * d_s).to($dotTy:tritonDType)
        $update
        b_dk = b_dk * d_k[:, None] + tl.trans(tl.dot(b_q, b_ds, allow_tf32=False))
        p_dq = tl.make_block_ptr(base=dq + (i_bh * $(s_qk_h)), shape=($(T), $(K)), strides=($(s_qk_t), $(s_qk_d)), offsets=((i_t * $(BT)), (i_k * $(BK))), block_shape=($(BT), $(BK)), order=(1, 0))
        p_dk = tl.make_block_ptr(base=dk + (i_bh * $(s_qk_h)), shape=($(T), $(K)), strides=($(s_qk_t), $(s_qk_d)), offsets=((i_t * $(BT)), (i_k * $(BK))), block_shape=($(BT), $(BK)), order=(1, 0))
        tl.store(p_dq, (b_dq).to($dotTy:tritonDType), boundary_check=([0, 1] : List Nat))
        tl.store(p_dk, (b_dk).to($dotTy:tritonDType), boundary_check=([0, 1] : List Nat))
      }).surfaceBody)
```
</details>

<details><summary><code>backwardStateAfter</code></summary>

```
/-- Carry after n reverse-time updates of initial Q/dO. A history tile is
stored before its update; unlike forward state, all backward decay lengths
remain BT, including the partial last chunk. -/
```
```lean
def backwardStateAfter (p : Layout) (scale : ℝ) (l : Launch) (a : Inputs) : Nat → Nat → Nat → ℝ
  | 0, _, _ => 0
  | n+1, k, v => Decay.factor (RealHMath.base p l) p.BT * backwardStateAfter p scale l a n k v +
      ∑ t : Fin p.BT,
        (query p l a k (timeIndex p (p.NT-1-n) t.val) * scale) *
          (incoming p l a (timeIndex p (p.NT-1-n) t.val) v * Decay.query (RealHMath.base p l) t.val)
```
</details>

<details><summary><code>backwardH</code></summary>

```lean
def backwardH (dotStorage : Storage) := backwardHSource dotStorage Bool.false
```
</details>

<details><summary><code>backwardHSource</code></summary>

```lean
def backwardHSource (dotStorage : Storage) (swapped : Bool) :=
  match dotStorage with
  | .fp32 => if swapped then retentionOpsBackwardHSource%[tl.float32, dotLeft] else retentionOpsBackwardHSource%[tl.float32, original]
  | .fp16 => if swapped then retentionOpsBackwardHSource%[tl.float16, dotLeft] else retentionOpsBackwardHSource%[tl.float16, original]
  | .bf16 => if swapped then retentionOpsBackwardHSource%[tl.bfloat16, dotLeft] else retentionOpsBackwardHSource%[tl.bfloat16, original]
```
</details>

## Public theorem: `backwardGradients_correct`

<details><summary>docstring</summary>

```
/-- The four repeatedly loaded inputs are separate from dV; the three
outputs are pairwise distinct. Q/K can alias dV because they are loaded first. -/
```
</details>

**Statement:**
```lean
specification backwardGradients_correct (p : Layout) (r : Regions) (storage : Storage) (scale : ℝ)
    (hv : r.value ≠ r.dv) (ho : r.incoming ≠ r.dv) (hh : r.history ≠ r.dv) (hdh : r.historyGradient ≠ r.dv)
    (hqk : r.dq ≠ r.dk) (hqv : r.dq ≠ r.dv) (hkv : r.dk ≠ r.dv) :
    backwardGradients p r storage scale ⊨ backwardGradientsFormula p scale
```

**Assumptions / layout contracts:**
- `hv : r.value ≠ r.dv`
- `ho : r.incoming ≠ r.dv`
- `hh : r.history ≠ r.dv`
- `hdh : r.historyGradient ≠ r.dv`
- `hqk : r.dq ≠ r.dk`
- `hqv : r.dq ≠ r.dv`
- `hkv : r.dk ≠ r.dv`

**Closed-form spec defs (transitive):** `Storage`, `backwardGradients`, `backwardGradientsFormula`, `backwardQKV`, `backwardGradientsScoreGradient`, `backwardGradientsScore`, `backwardQKVSource`

<details><summary><code>Storage</code></summary>

```lean
abbrev Storage := VeriTile.Bench.TritonBenchG.ChunkRetention.Storage

/-- Source: chunk_retention_fwd_kernel_h. -/
syntax "retentionOpsForwardHSource%" "[" tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(retentionOpsForwardHSource%[$dotTy:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `dotLeft do
      Macro.throwError "expected original or dotLeft"
    let update : TSyntax `tritonStmt ←
      if version.getId == `dotLeft then `(tritonStmt| b_h = tl.dot(b_k, (b_v * d_i[:, None]).to($dotTy:tritonDType), allow_tf32=False) + d_b * b_h)
      else `(tritonStmt| b_h = d_b * b_h + tl.dot(b_k, (b_v * d_i[:, None]).to($dotTy:tritonDType), allow_tf32=False))
    `(fun (dtype out_dtype : FloatDType) (k v h h0 ht : RegionName)
        (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat) (H T K V BT BK BV NT : Nat) (USE_INITIAL_STATE STORE_FINAL_STATE : Bool) =>
      ComputeKernel.mk [k,v,h0] [h,ht] (triton {
        tl.region k = $dotTy, v = $dotTy, h = $dotTy, h0 = DTYPE, ht = OUT_DTYPE
        i_k = tl.program_id(0)
        i_v = tl.program_id(1)
        i_bh = tl.program_id(2)
        i_h = i_bh % $(H)
        b_b = tl.math.log2(1.0 - tl.math.exp2($((-5 : ℝ)) - i_h * 1.0))
        o_i = (tl.arange(0, $(BT))).to(tl.int32)
        d_b = tl.math.exp2($(BT) * b_b)
        d_i = tl.math.exp2(($(BT) - o_i - $(1)) * b_b)
        b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
        if USE_INITIAL_STATE {
          p_h0 = tl.make_block_ptr(base=h0 + ((i_bh * $(K)) * $(V)), shape=($(K), $(V)), strides=($(V), 1), offsets=((i_k * $(BK)), (i_v * $(BV))), block_shape=($(BK), $(BV)), order=(1, 0))
          b_h = (tl.load(p_h0, boundary_check=([0, 1] : List Nat))).to(tl.float32)
        }
        for i_t in range($(NT)) {
          p_k = tl.make_block_ptr(base=k + (i_bh * $(s_qk_h)), shape=($(K), $(T)), strides=($(s_qk_d), $(s_qk_t)), offsets=((i_k * $(BK)), (i_t * $(BT))), block_shape=($(BK), $(BT)), order=(0, 1))
          p_v = tl.make_block_ptr(base=v + (i_bh * $(s_vo_h)), shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)), offsets=((i_t * $(BT)), (i_v * $(BV))), block_shape=($(BT), $(BV)), order=(1, 0))
          p_h = tl.make_block_ptr(base=h + (i_bh * $(s_h_h)) + ((i_t * $(K)) * $(V)), shape=($(K), $(V)), strides=($(s_h_t), 1), offsets=((i_k * $(BK)), (i_v * $(BV))), block_shape=($(BK), $(BV)), order=(1, 0))
          tl.store(p_h, (b_h).to($dotTy:tritonDType), boundary_check=([0, 1] : List Nat))
          b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
          b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
          if i_t == $(NT) - $(1) and $(T) % $(BT) != $(0) {
            d_b = tl.math.exp2(($(T) % $(BT)) * b_b)
            d_i = tl.math.exp2((($(T) % $(BT)) - o_i - $(1)) * b_b)
          }
          $update
        }
        if STORE_FINAL_STATE {
          p_ht = tl.make_block_ptr(base=ht + ((i_bh * $(K)) * $(V)), shape=($(K), $(V)), strides=($(V), 1), offsets=((i_k * $(BK)), (i_v * $(BV))), block_shape=($(BK), $(BV)), order=(1, 0))
          tl.store(p_ht, (b_h).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
        }
      }).surfaceBody)
```
</details>

<details><summary><code>backwardGradients</code></summary>

```
/-- dV writes must be injective across all value blocks; dQ/dK writes must
be injective in their active tile. Bounds cover every input/output access.
The dV base includes the runtime head count as in the source. -/
```
```lean
def backwardGradients (p : Layout) (r : Regions) (storage : Storage) (scale : ℝ) : RealQKVIO.IO p where
  kernel := (VeriTile.Bench.TritonBenchG.ChunkRetentionOps.backwardQKV storage r.query r.key r.value r.history r.incoming r.historyGradient r.dq r.dk r.dv
    p.keyHead p.keyTime p.keyFeature p.valueHead p.valueTime p.valueFeature p.historyHead p.historyKey
    scale p.H p.T p.K p.V p.BT p.BK p.BV p.NT).eraseDType
  projection := RealQKVIO.projection p r storage scale Bool.false
  read := fun s => ⟨s.readMem r.query,s.readMem r.key,s.readMem r.value,s.readMem r.history,
    s.readMem r.incoming,s.readMem r.historyGradient⟩
  valid := fun s => RealQKVLoop.NoCollision p (launch s) r ∧ (RealQKVFinal.keyWindow p (launch s) r.dq).NoCollision
  bounds := fun s => RealQKVSafety.Bounds p (launch s) r
  dq := r.dq
  dk := r.dk
  dv := r.dv
  keyWindow := fun l i e => l.head*p.keyHead+(l.chunk*p.BT+i.val)*p.keyTime+(l.key*p.BK+e.val)*p.keyFeature
  valueWindow := fun l n i j => (l.key*l.heads+l.head)*p.valueHead+(l.chunk*p.BT+i.val)*p.valueTime+(n*p.BV+j.val)*p.valueFeature
  keyMask := fun l i e => l.chunk*p.BT+i.val < p.T ∧ l.key*p.BK+e.val < p.K
  valueMask := fun l n i j => l.chunk*p.BT+i.val < p.T ∧ n*p.BV+j.val < p.V
```
</details>

<details><summary><code>backwardGradientsFormula</code></summary>

```
/-- Independent initial-array contractions for dQ, dK and dV. Accessors
retain the source strides and distinct K versus NT*K history boundary checks. -/
```
```lean
def backwardGradientsFormula (p : Layout) (scale : ℝ) (l : Launch) (a : Inputs) : RealQKVIO.Values p where
  query := fun i e =>
    (∑ c ∈ Finset.range (numBlocks p),
      (∑ v : Fin p.BV, incoming p l a c i.val v.val*history p l a c v.val e.val)) *
      (Decay.query (base p l) i.val * scale) +
      ∑ j : Fin p.BT, backwardGradientsScoreGradient p scale l a i.val j.val*key p l a j.val e.val
  key := fun i e =>
    (∑ c ∈ Finset.range (numBlocks p), ∑ v : Fin p.BV,
      value p l a c i.val v.val*historyGradient p l a c e.val v.val) * Decay.key (base p l) p.BT i.val +
      ∑ j : Fin p.BT, query p l a e.val j.val*backwardGradientsScoreGradient p scale l a j.val i.val
  value := fun c i j =>
    (∑ e : Fin p.BK, key p l a i.val e.val*historyGradient p l a c e.val j.val) * Decay.key (base p l) p.BT i.val +
      ∑ t : Fin p.BT, backwardGradientsScore p scale l a i.val t.val*incoming p l a c t.val j.val
```
</details>

<details><summary><code>backwardQKV</code></summary>

```lean
def backwardQKV (dotStorage : Storage) := backwardQKVSource dotStorage Bool.false
```
</details>

<details><summary><code>backwardGradientsScoreGradient</code></summary>

```
/-- Lower causal dO·Vᵀ score gradient, summing every value block. -/
```
```lean
def backwardGradientsScoreGradient (p : Layout) (scale : ℝ) (l : Launch) (a : Inputs) (i j : Nat) : ℝ :=
  (∑ c ∈ Finset.range (numBlocks p), ∑ v : Fin p.BV,
    incoming p l a c i v.val*value p l a c j v.val) * (Decay.score (base p l) i j * scale)
```
</details>

<details><summary><code>backwardGradientsScore</code></summary>

```
/-- Upper causal K·Q score. -/
```
```lean
def backwardGradientsScore (p : Layout) (scale : ℝ) (l : Launch) (a : Inputs) (i j : Nat) : ℝ :=
  (∑ e : Fin p.BK, key p l a i e.val*query p l a e.val j) * (Decay.score (base p l) j i * scale)
```
</details>

<details><summary><code>backwardQKVSource</code></summary>

```lean
def backwardQKVSource (dotStorage : Storage) (swapped : Bool) :=
  match dotStorage with
  | .fp32 =>
      if swapped then retentionOpsBackwardQKVSource%[tl.float32, dotLeft]
      else retentionOpsBackwardQKVSource%[tl.float32, original]
  | .fp16 =>
      if swapped then retentionOpsBackwardQKVSource%[tl.float16, dotLeft]
      else retentionOpsBackwardQKVSource%[tl.float16, original]
  | .bf16 =>
      if swapped then retentionOpsBackwardQKVSource%[tl.bfloat16, dotLeft]
      else retentionOpsBackwardQKVSource%[tl.bfloat16, original]
```
</details>

## Public theorem: `backwardGradientsDotLeft_correct`

**Statement:**
```lean
specification backwardGradientsDotLeft_correct (p : Layout) (r : Regions) (storage : Storage) (scale : ℝ)
    (hv : r.value ≠ r.dv) (ho : r.incoming ≠ r.dv) (hh : r.history ≠ r.dv) (hdh : r.historyGradient ≠ r.dv)
    (hqk : r.dq ≠ r.dk) (hqv : r.dq ≠ r.dv) (hkv : r.dk ≠ r.dv) :
    backwardGradientsDotLeft p r storage scale ⊨ backwardGradientsFormula p scale
```

**Assumptions / layout contracts:**
- `hv : r.value ≠ r.dv`
- `ho : r.incoming ≠ r.dv`
- `hh : r.history ≠ r.dv`
- `hdh : r.historyGradient ≠ r.dv`
- `hqk : r.dq ≠ r.dk`
- `hqv : r.dq ≠ r.dv`
- `hkv : r.dk ≠ r.dv`

**Closed-form spec defs (transitive):** `Storage`, `backwardGradientsDotLeft`, `backwardGradientsFormula`, `backwardGradients`, `backwardQKVDotLeft`, `backwardGradientsScoreGradient`, `backwardGradientsScore`, `backwardQKV`, `backwardQKVSource`

<details><summary><code>Storage</code></summary>

```lean
abbrev Storage := VeriTile.Bench.TritonBenchG.ChunkRetention.Storage

/-- Source: chunk_retention_fwd_kernel_h. -/
syntax "retentionOpsForwardHSource%" "[" tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(retentionOpsForwardHSource%[$dotTy:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `dotLeft do
      Macro.throwError "expected original or dotLeft"
    let update : TSyntax `tritonStmt ←
      if version.getId == `dotLeft then `(tritonStmt| b_h = tl.dot(b_k, (b_v * d_i[:, None]).to($dotTy:tritonDType), allow_tf32=False) + d_b * b_h)
      else `(tritonStmt| b_h = d_b * b_h + tl.dot(b_k, (b_v * d_i[:, None]).to($dotTy:tritonDType), allow_tf32=False))
    `(fun (dtype out_dtype : FloatDType) (k v h h0 ht : RegionName)
        (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat) (H T K V BT BK BV NT : Nat) (USE_INITIAL_STATE STORE_FINAL_STATE : Bool) =>
      ComputeKernel.mk [k,v,h0] [h,ht] (triton {
        tl.region k = $dotTy, v = $dotTy, h = $dotTy, h0 = DTYPE, ht = OUT_DTYPE
        i_k = tl.program_id(0)
        i_v = tl.program_id(1)
        i_bh = tl.program_id(2)
        i_h = i_bh % $(H)
        b_b = tl.math.log2(1.0 - tl.math.exp2($((-5 : ℝ)) - i_h * 1.0))
        o_i = (tl.arange(0, $(BT))).to(tl.int32)
        d_b = tl.math.exp2($(BT) * b_b)
        d_i = tl.math.exp2(($(BT) - o_i - $(1)) * b_b)
        b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
        if USE_INITIAL_STATE {
          p_h0 = tl.make_block_ptr(base=h0 + ((i_bh * $(K)) * $(V)), shape=($(K), $(V)), strides=($(V), 1), offsets=((i_k * $(BK)), (i_v * $(BV))), block_shape=($(BK), $(BV)), order=(1, 0))
          b_h = (tl.load(p_h0, boundary_check=([0, 1] : List Nat))).to(tl.float32)
        }
        for i_t in range($(NT)) {
          p_k = tl.make_block_ptr(base=k + (i_bh * $(s_qk_h)), shape=($(K), $(T)), strides=($(s_qk_d), $(s_qk_t)), offsets=((i_k * $(BK)), (i_t * $(BT))), block_shape=($(BK), $(BT)), order=(0, 1))
          p_v = tl.make_block_ptr(base=v + (i_bh * $(s_vo_h)), shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)), offsets=((i_t * $(BT)), (i_v * $(BV))), block_shape=($(BT), $(BV)), order=(1, 0))
          p_h = tl.make_block_ptr(base=h + (i_bh * $(s_h_h)) + ((i_t * $(K)) * $(V)), shape=($(K), $(V)), strides=($(s_h_t), 1), offsets=((i_k * $(BK)), (i_v * $(BV))), block_shape=($(BK), $(BV)), order=(1, 0))
          tl.store(p_h, (b_h).to($dotTy:tritonDType), boundary_check=([0, 1] : List Nat))
          b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
          b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
          if i_t == $(NT) - $(1) and $(T) % $(BT) != $(0) {
            d_b = tl.math.exp2(($(T) % $(BT)) * b_b)
            d_i = tl.math.exp2((($(T) % $(BT)) - o_i - $(1)) * b_b)
          }
          $update
        }
        if STORE_FINAL_STATE {
          p_ht = tl.make_block_ptr(base=ht + ((i_bh * $(K)) * $(V)), shape=($(K), $(V)), strides=($(V), 1), offsets=((i_k * $(BK)), (i_v * $(BV))), block_shape=($(BK), $(BV)), order=(1, 0))
          tl.store(p_ht, (b_h).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
        }
      }).surfaceBody)
```
</details>

<details><summary><code>backwardGradientsDotLeft</code></summary>

```lean
def backwardGradientsDotLeft (p : Layout) (r : Regions) (storage : Storage) (scale : ℝ) : RealQKVIO.IO p :=
  { backwardGradients p r storage scale with
    kernel := (VeriTile.Bench.TritonBenchG.ChunkRetentionOps.backwardQKVDotLeft storage r.query r.key r.value r.history r.incoming r.historyGradient r.dq r.dk r.dv
      p.keyHead p.keyTime p.keyFeature p.valueHead p.valueTime p.valueFeature p.historyHead p.historyKey
      scale p.H p.T p.K p.V p.BT p.BK p.BV p.NT).eraseDType
    projection := RealQKVIO.projection p r storage scale Bool.true }
```
</details>

<details><summary><code>backwardGradientsFormula</code></summary>

```
/-- Independent initial-array contractions for dQ, dK and dV. Accessors
retain the source strides and distinct K versus NT*K history boundary checks. -/
```
```lean
def backwardGradientsFormula (p : Layout) (scale : ℝ) (l : Launch) (a : Inputs) : RealQKVIO.Values p where
  query := fun i e =>
    (∑ c ∈ Finset.range (numBlocks p),
      (∑ v : Fin p.BV, incoming p l a c i.val v.val*history p l a c v.val e.val)) *
      (Decay.query (base p l) i.val * scale) +
      ∑ j : Fin p.BT, backwardGradientsScoreGradient p scale l a i.val j.val*key p l a j.val e.val
  key := fun i e =>
    (∑ c ∈ Finset.range (numBlocks p), ∑ v : Fin p.BV,
      value p l a c i.val v.val*historyGradient p l a c e.val v.val) * Decay.key (base p l) p.BT i.val +
      ∑ j : Fin p.BT, query p l a e.val j.val*backwardGradientsScoreGradient p scale l a j.val i.val
  value := fun c i j =>
    (∑ e : Fin p.BK, key p l a i.val e.val*historyGradient p l a c e.val j.val) * Decay.key (base p l) p.BT i.val +
      ∑ t : Fin p.BT, backwardGradientsScore p scale l a i.val t.val*incoming p l a c t.val j.val
```
</details>

<details><summary><code>backwardGradients</code></summary>

```
/-- dV writes must be injective across all value blocks; dQ/dK writes must
be injective in their active tile. Bounds cover every input/output access.
The dV base includes the runtime head count as in the source. -/
```
```lean
def backwardGradients (p : Layout) (r : Regions) (storage : Storage) (scale : ℝ) : RealQKVIO.IO p where
  kernel := (VeriTile.Bench.TritonBenchG.ChunkRetentionOps.backwardQKV storage r.query r.key r.value r.history r.incoming r.historyGradient r.dq r.dk r.dv
    p.keyHead p.keyTime p.keyFeature p.valueHead p.valueTime p.valueFeature p.historyHead p.historyKey
    scale p.H p.T p.K p.V p.BT p.BK p.BV p.NT).eraseDType
  projection := RealQKVIO.projection p r storage scale Bool.false
  read := fun s => ⟨s.readMem r.query,s.readMem r.key,s.readMem r.value,s.readMem r.history,
    s.readMem r.incoming,s.readMem r.historyGradient⟩
  valid := fun s => RealQKVLoop.NoCollision p (launch s) r ∧ (RealQKVFinal.keyWindow p (launch s) r.dq).NoCollision
  bounds := fun s => RealQKVSafety.Bounds p (launch s) r
  dq := r.dq
  dk := r.dk
  dv := r.dv
  keyWindow := fun l i e => l.head*p.keyHead+(l.chunk*p.BT+i.val)*p.keyTime+(l.key*p.BK+e.val)*p.keyFeature
  valueWindow := fun l n i j => (l.key*l.heads+l.head)*p.valueHead+(l.chunk*p.BT+i.val)*p.valueTime+(n*p.BV+j.val)*p.valueFeature
  keyMask := fun l i e => l.chunk*p.BT+i.val < p.T ∧ l.key*p.BK+e.val < p.K
  valueMask := fun l n i j => l.chunk*p.BT+i.val < p.T ∧ n*p.BV+j.val < p.V
```
</details>

<details><summary><code>backwardQKVDotLeft</code></summary>

```lean
def backwardQKVDotLeft (dotStorage : Storage) := backwardQKVSource dotStorage Bool.true

end VeriTile.Bench.TritonBenchG.ChunkRetentionOps

#guard_msgs (drop info) in
#auditModuleAxioms
```
</details>

<details><summary><code>backwardGradientsScoreGradient</code></summary>

```
/-- Lower causal dO·Vᵀ score gradient, summing every value block. -/
```
```lean
def backwardGradientsScoreGradient (p : Layout) (scale : ℝ) (l : Launch) (a : Inputs) (i j : Nat) : ℝ :=
  (∑ c ∈ Finset.range (numBlocks p), ∑ v : Fin p.BV,
    incoming p l a c i v.val*value p l a c j v.val) * (Decay.score (base p l) i j * scale)
```
</details>

<details><summary><code>backwardGradientsScore</code></summary>

```
/-- Upper causal K·Q score. -/
```
```lean
def backwardGradientsScore (p : Layout) (scale : ℝ) (l : Launch) (a : Inputs) (i j : Nat) : ℝ :=
  (∑ e : Fin p.BK, key p l a i e.val*query p l a e.val j) * (Decay.score (base p l) j i * scale)
```
</details>

<details><summary><code>backwardQKV</code></summary>

```lean
def backwardQKV (dotStorage : Storage) := backwardQKVSource dotStorage Bool.false
```
</details>

<details><summary><code>backwardQKVSource</code></summary>

```lean
def backwardQKVSource (dotStorage : Storage) (swapped : Bool) :=
  match dotStorage with
  | .fp32 =>
      if swapped then retentionOpsBackwardQKVSource%[tl.float32, dotLeft]
      else retentionOpsBackwardQKVSource%[tl.float32, original]
  | .fp16 =>
      if swapped then retentionOpsBackwardQKVSource%[tl.float16, dotLeft]
      else retentionOpsBackwardQKVSource%[tl.float16, original]
  | .bf16 =>
      if swapped then retentionOpsBackwardQKVSource%[tl.bfloat16, dotLeft]
      else retentionOpsBackwardQKVSource%[tl.bfloat16, original]
```
</details>
