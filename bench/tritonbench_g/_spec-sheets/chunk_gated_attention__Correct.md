# Spec sheet — `bench/tritonbench_g/chunk_gated_attention/Correct.lean`

**Python source:** `bench/tritonbench_g/chunk_gated_attention/chunk_gated_attention.py`

## Public theorem: `cumulative_correctness`

**Statement:**
```lean
specification cumulative_correctness (p : CumLayout) (s o : RegionName) (dtype out_dtype : FloatDType)
    (hinj : cumulativeNoOverlap p) :
    Spec.Real (cumulativeIO p s o dtype out_dtype ⊨ cumulativeExpected p)
```

**Assumptions / layout contracts:**
- `hinj : cumulativeNoOverlap p`

**Closed-form spec defs (transitive):** `CumLayout`, `cumulativeNoOverlap`, `cumulativeIO`, `cumulativeExpected`, `cumOriginal`

<details><summary><code>CumLayout</code></summary>

```lean
abbrev CumLayout := RealCumProgram.Layout
```
</details>

<details><summary><code>cumulativeNoOverlap</code></summary>

```
/-- Only active output cells of the same program must have distinct addresses.
The IO relation also requires every active input/output address to be allocated. -/
```
```lean
def cumulativeNoOverlap (p : CumLayout) : Prop :=
  ∀ feature c head, RealCumExecution.NoCollision p ⟨feature,head⟩ c
```
</details>

<details><summary><code>cumulativeIO</code></summary>

```
/-- Feature-block, chunk and head are program axes 0/1/2. Checked-off time and
feature lanes are excluded from the BT×BS output window. -/
```
```lean
def cumulativeIO (p : CumLayout) (s o : RegionName) (dtype out_dtype : FloatDType) : Masked3DTileKernelIO₁ where
  kernel := (TypedSource.cumOriginal dtype out_dtype s o p.headStride p.timeStride p.channelStride
    p.T p.S p.BT p.BS).eraseDType
  projection := (RealCumIO.io Bool.false p s o dtype out_dtype).projection
  inp := s
  out := o
  shape := [p.BT,p.BS]
  read := fun feature c head i =>
    head*p.headStride+(c*p.BT+i.1.val)*p.timeStride+(feature*p.BS+i.2.1.val)*p.channelStride
  write := fun feature c head i =>
    head*p.headStride+(c*p.BT+i.1.val)*p.timeStride+(feature*p.BS+i.2.1.val)*p.channelStride
  mask := fun feature c _ i => c*p.BT+i.1.val < p.T ∧ feature*p.BS+i.2.1.val < p.S
```
</details>

<details><summary><code>cumulativeExpected</code></summary>

```
/-- Prefix sum of initial inputs within one chunk, independently per feature. -/
```
```lean
def cumulativeExpected (p : CumLayout) (_feature _chunk : Nat) (xs : TileIndex [p.BT,p.BS] → ℝ)
    (i : TileIndex [p.BT,p.BS]) : ℝ :=
  ∑ k : Fin p.BT, if k.val ≤ i.1.val then xs (k,i.2.1,PUnit.unit) else 0
```
</details>

<details><summary><code>cumOriginal</code></summary>

```lean
def cumOriginal := gatedCumSource% original
```
</details>

## Public theorem: `cumulative_index_first_correctness`

**Statement:**
```lean
specification cumulative_index_first_correctness (p : CumLayout) (s o : RegionName) (dtype out_dtype : FloatDType)
    (hinj : cumulativeNoOverlap p) :
    Spec.Real (cumulativeIndexFirstIO p s o dtype out_dtype ⊨ cumulativeExpected p)
```

**Assumptions / layout contracts:**
- `hinj : cumulativeNoOverlap p`

**Closed-form spec defs (transitive):** `CumLayout`, `cumulativeNoOverlap`, `cumulativeIndexFirstIO`, `cumulativeExpected`, `cumulativeIO`, `cumIndexFirst`, `cumOriginal`

<details><summary><code>CumLayout</code></summary>

```lean
abbrev CumLayout := RealCumProgram.Layout
```
</details>

<details><summary><code>cumulativeNoOverlap</code></summary>

```
/-- Only active output cells of the same program must have distinct addresses.
The IO relation also requires every active input/output address to be allocated. -/
```
```lean
def cumulativeNoOverlap (p : CumLayout) : Prop :=
  ∀ feature c head, RealCumExecution.NoCollision p ⟨feature,head⟩ c
```
</details>

<details><summary><code>cumulativeIndexFirstIO</code></summary>

```lean
def cumulativeIndexFirstIO (p : CumLayout) (s o : RegionName) (dtype out_dtype : FloatDType) : Masked3DTileKernelIO₁ :=
  { cumulativeIO p s o dtype out_dtype with
    kernel := (TypedSource.cumIndexFirst dtype out_dtype s o p.headStride p.timeStride p.channelStride
      p.T p.S p.BT p.BS).eraseDType
    projection := (RealCumIO.io Bool.true p s o dtype out_dtype).projection }
```
</details>

<details><summary><code>cumulativeExpected</code></summary>

```
/-- Prefix sum of initial inputs within one chunk, independently per feature. -/
```
```lean
def cumulativeExpected (p : CumLayout) (_feature _chunk : Nat) (xs : TileIndex [p.BT,p.BS] → ℝ)
    (i : TileIndex [p.BT,p.BS]) : ℝ :=
  ∑ k : Fin p.BT, if k.val ≤ i.1.val then xs (k,i.2.1,PUnit.unit) else 0
```
</details>

<details><summary><code>cumulativeIO</code></summary>

```
/-- Feature-block, chunk and head are program axes 0/1/2. Checked-off time and
feature lanes are excluded from the BT×BS output window. -/
```
```lean
def cumulativeIO (p : CumLayout) (s o : RegionName) (dtype out_dtype : FloatDType) : Masked3DTileKernelIO₁ where
  kernel := (TypedSource.cumOriginal dtype out_dtype s o p.headStride p.timeStride p.channelStride
    p.T p.S p.BT p.BS).eraseDType
  projection := (RealCumIO.io Bool.false p s o dtype out_dtype).projection
  inp := s
  out := o
  shape := [p.BT,p.BS]
  read := fun feature c head i =>
    head*p.headStride+(c*p.BT+i.1.val)*p.timeStride+(feature*p.BS+i.2.1.val)*p.channelStride
  write := fun feature c head i =>
    head*p.headStride+(c*p.BT+i.1.val)*p.timeStride+(feature*p.BS+i.2.1.val)*p.channelStride
  mask := fun feature c _ i => c*p.BT+i.1.val < p.T ∧ feature*p.BS+i.2.1.val < p.S
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

<details><summary><code>cumOriginal</code></summary>

```lean
def cumOriginal := gatedCumSource% original
```
</details>

## Public theorem: `recurrent_correctness`

<details><summary>docstring</summary>

```
/-- The history store precedes K/V/gate loads, so those inputs cannot alias it.
The final store preserves history. Seed may alias outputs: it is read first. -/
```
</details>

**Statement:**
```lean
specification recurrent_correctness (p : HLayout) (r : Regions) (dot gate history : TypedSource.Storage)
    (seed final : FloatDType) (gateK hasSeed storeFinal : Bool)
    (hs : RealHStep.Separate r) (hfsep : storeFinal = Bool.true → r.history ≠ r.final) :
    Spec.Real (recurrentIO p r dot gate history seed final gateK hasSeed storeFinal ⊨ recurrentExpected p gateK hasSeed)
```

**Assumptions / layout contracts:**
- `hs : RealHStep.Separate r`
- `hfsep : storeFinal = Bool.true → r.history ≠ r.final`

**Closed-form spec defs (transitive):** `HLayout`, `recurrentIO`, `recurrentExpected`, `hOriginal`, `hSource`

<details><summary><code>HLayout</code></summary>

```lean
abbrev HLayout := RealHMath.Layout
```
</details>

<details><summary><code>recurrentIO</code></summary>

```
/-- History observes H before each update; optional final output observes H[NT].
K/V/gate/seed arrays are read from initial memory. Access bounds cover the actual
selected gate tile and flattened full-chunk endpoint, plus enabled state loads
and stores. `valid` prevents active history and final-address collisions.
IO.Implements explicitly requires zero unspecified real values. -/
```
```lean
def recurrentIO (p : HLayout) (r : Regions) (dot gate history : TypedSource.Storage)
    (seed final : FloatDType) (gateK hasSeed storeFinal : Bool) : RealHIO.IO p where
  kernel := (TypedSource.hOriginal dot gate history seed final r.key r.value r.gate r.history r.seed r.final
    p.keyHead p.keyTime p.keyFeature p.valueHead p.valueTime p.valueFeature
    p.historyHead p.historyKey p.historyValue p.T p.K p.V p.BT p.BK p.BV p.NT gateK hasSeed storeFinal).eraseDType
  projection := RealHIO.projection p r dot gate history seed final gateK hasSeed storeFinal Bool.false
  read := fun s => ⟨s.readMem r.key,s.readMem r.value,s.readMem r.gate,s.readMem r.seed⟩
  valid := fun s => RealHLoop.HistoryNoCollision p (launch s) ∧
    (storeFinal = Bool.true → RealHStore.NoCollision p (launch s) (RealHFinal.base p (launch s)) p.V 1)
  bounds := fun s => RealHSafety.Bounds p (launch s) r gateK hasSeed storeFinal
  history := r.history
  final := r.final
  storeFinal := storeFinal
  historyWindow := fun l c e j =>
    l.head*p.historyHead+c*p.K*p.V+(l.key*p.BK+e.val)*p.historyKey+(l.value*p.BV+j.val)*p.historyValue
  finalWindow := fun l e j => l.head*p.K*p.V+(l.key*p.BK+e.val)*p.V+(l.value*p.BV+j.val)
  stateMask := fun l e j => l.key*p.BK+e.val < p.K ∧ l.value*p.BV+j.val < p.V
```
</details>

<details><summary><code>recurrentExpected</code></summary>

```
/-- G[c]=exp(full-chunk endpoint), and P[c] is the BT-term gated K/V contraction.
`gateFactor` and `contribution` independently index the initial arrays, with
zero extension outside each source pointer's checked parent. GATEK selects
whether K or V carries the exp(endpoint-gate) weight. The closed form is
H[c]=seed*product(G[0:c])+sum(P[t]*product(G[t+1:c])). -/
```
```lean
def recurrentExpected (p : HLayout) (gateK hasSeed : Bool) (l : Launch) (a : Inputs) : RealHIO.Outputs p :=
  let state := fun c k v =>
    RealHMath.seed p l a hasSeed k v * (∏ t ∈ Finset.range c, gateFactor p l a gateK t k v) +
      ∑ t ∈ Finset.range c, contribution p l a gateK t k v *
        ∏ u ∈ Finset.Ico (t+1) c, gateFactor p l a gateK u k v
  { history := fun c e j => state c (keyIndex p l e.val) (valueIndex p l j.val)
    final := fun e j => state p.NT (keyIndex p l e.val) (valueIndex p l j.val) }
```
</details>

<details><summary><code>hOriginal</code></summary>

```lean
def hOriginal (dotStorage gateStorage historyStorage : Storage) :=
  hSource dotStorage gateStorage historyStorage Bool.false
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

## Public theorem: `recurrent_sum_left_correctness`

**Statement:**
```lean
specification recurrent_sum_left_correctness (p : HLayout) (r : Regions) (dot gate history : TypedSource.Storage)
    (seed final : FloatDType) (gateK hasSeed storeFinal : Bool)
    (hs : RealHStep.Separate r) (hfsep : storeFinal = Bool.true → r.history ≠ r.final) :
    Spec.Real (recurrentSumLeftIO p r dot gate history seed final gateK hasSeed storeFinal ⊨ recurrentExpected p gateK hasSeed)
```

**Assumptions / layout contracts:**
- `hs : RealHStep.Separate r`
- `hfsep : storeFinal = Bool.true → r.history ≠ r.final`

**Closed-form spec defs (transitive):** `HLayout`, `recurrentSumLeftIO`, `recurrentExpected`, `recurrentIO`, `hSumLeft`, `hOriginal`, `hSource`

<details><summary><code>HLayout</code></summary>

```lean
abbrev HLayout := RealHMath.Layout
```
</details>

<details><summary><code>recurrentSumLeftIO</code></summary>

```lean
def recurrentSumLeftIO (p : HLayout) (r : Regions) (dot gate history : TypedSource.Storage)
    (seed final : FloatDType) (gateK hasSeed storeFinal : Bool) : RealHIO.IO p :=
  { recurrentIO p r dot gate history seed final gateK hasSeed storeFinal with
    kernel := (TypedSource.hSumLeft dot gate history seed final r.key r.value r.gate r.history r.seed r.final
      p.keyHead p.keyTime p.keyFeature p.valueHead p.valueTime p.valueFeature
      p.historyHead p.historyKey p.historyValue p.T p.K p.V p.BT p.BK p.BV p.NT gateK hasSeed storeFinal).eraseDType
    projection := RealHIO.projection p r dot gate history seed final gateK hasSeed storeFinal Bool.true }
```
</details>

<details><summary><code>recurrentExpected</code></summary>

```
/-- G[c]=exp(full-chunk endpoint), and P[c] is the BT-term gated K/V contraction.
`gateFactor` and `contribution` independently index the initial arrays, with
zero extension outside each source pointer's checked parent. GATEK selects
whether K or V carries the exp(endpoint-gate) weight. The closed form is
H[c]=seed*product(G[0:c])+sum(P[t]*product(G[t+1:c])). -/
```
```lean
def recurrentExpected (p : HLayout) (gateK hasSeed : Bool) (l : Launch) (a : Inputs) : RealHIO.Outputs p :=
  let state := fun c k v =>
    RealHMath.seed p l a hasSeed k v * (∏ t ∈ Finset.range c, gateFactor p l a gateK t k v) +
      ∑ t ∈ Finset.range c, contribution p l a gateK t k v *
        ∏ u ∈ Finset.Ico (t+1) c, gateFactor p l a gateK u k v
  { history := fun c e j => state c (keyIndex p l e.val) (valueIndex p l j.val)
    final := fun e j => state p.NT (keyIndex p l e.val) (valueIndex p l j.val) }
```
</details>

<details><summary><code>recurrentIO</code></summary>

```
/-- History observes H before each update; optional final output observes H[NT].
K/V/gate/seed arrays are read from initial memory. Access bounds cover the actual
selected gate tile and flattened full-chunk endpoint, plus enabled state loads
and stores. `valid` prevents active history and final-address collisions.
IO.Implements explicitly requires zero unspecified real values. -/
```
```lean
def recurrentIO (p : HLayout) (r : Regions) (dot gate history : TypedSource.Storage)
    (seed final : FloatDType) (gateK hasSeed storeFinal : Bool) : RealHIO.IO p where
  kernel := (TypedSource.hOriginal dot gate history seed final r.key r.value r.gate r.history r.seed r.final
    p.keyHead p.keyTime p.keyFeature p.valueHead p.valueTime p.valueFeature
    p.historyHead p.historyKey p.historyValue p.T p.K p.V p.BT p.BK p.BV p.NT gateK hasSeed storeFinal).eraseDType
  projection := RealHIO.projection p r dot gate history seed final gateK hasSeed storeFinal Bool.false
  read := fun s => ⟨s.readMem r.key,s.readMem r.value,s.readMem r.gate,s.readMem r.seed⟩
  valid := fun s => RealHLoop.HistoryNoCollision p (launch s) ∧
    (storeFinal = Bool.true → RealHStore.NoCollision p (launch s) (RealHFinal.base p (launch s)) p.V 1)
  bounds := fun s => RealHSafety.Bounds p (launch s) r gateK hasSeed storeFinal
  history := r.history
  final := r.final
  storeFinal := storeFinal
  historyWindow := fun l c e j =>
    l.head*p.historyHead+c*p.K*p.V+(l.key*p.BK+e.val)*p.historyKey+(l.value*p.BV+j.val)*p.historyValue
  finalWindow := fun l e j => l.head*p.K*p.V+(l.key*p.BK+e.val)*p.V+(l.value*p.BV+j.val)
  stateMask := fun l e j => l.key*p.BK+e.val < p.K ∧ l.value*p.BV+j.val < p.V
```
</details>

<details><summary><code>hSumLeft</code></summary>

```lean
def hSumLeft (dotStorage gateStorage historyStorage : Storage) :=
  hSource dotStorage gateStorage historyStorage Bool.true
```
</details>

<details><summary><code>hOriginal</code></summary>

```lean
def hOriginal (dotStorage gateStorage historyStorage : Storage) :=
  hSource dotStorage gateStorage historyStorage Bool.false
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
