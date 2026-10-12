# Spec sheet — `bench/tritonbench_g/attention_fwd_triton3/Correct.lean`

**Python source:** `bench/tritonbench_g/attention_fwd_triton3/attention_fwd_triton3.py`

## Public theorem: `attention_correctness`

<details><summary>docstring</summary>

```
/-- Unchecked KV loads require whole key blocks. Positive inner stride and
sufficient row stride prevent output overlap; M/L/Out are distinct regions.
Active address bounds are supplied by the displayed IO windows. -/
```
</details>

**Statement:**
```lean
specification attention_correctness (BM BN BD Z finish : Nat) (p : Parameters)
    (hN : 0 < BN) (hflag : p.evenN ≠ 0 → p.layout.kvContext % BN = 0)
    (hd : 0 < p.layout.oDim) (hr : BD * p.layout.oDim ≤ p.layout.oRow)
    (hm : p.layout.M ≠ p.layout.Out) (hlm : p.layout.L ≠ p.layout.M) (hlo : p.layout.L ≠ p.layout.Out) :
    Spec.Real (originalIO BM BN BD Z finish p ⊨ fun start _ xs => expected BM BN BD finish start p xs)
```

**Assumptions / layout contracts:**
- `hN : 0 < BN`
- `hflag : p.evenN ≠ 0 → p.layout.kvContext % BN = 0`
- `hd : 0 < p.layout.oDim`
- `hr : BD * p.layout.oDim ≤ p.layout.oRow`
- `hm : p.layout.M ≠ p.layout.Out`
- `hlm : p.layout.L ≠ p.layout.M`
- `hlo : p.layout.L ≠ p.layout.Out`

**Closed-form spec defs (transitive):** `Parameters`, `originalIO`, `expected`, `original`, `carry`, `row`, `initial`, `summarize`, `score`

<details><summary><code>Parameters</code></summary>

```
/-- Source parameters: query/KV head counts; independent batch/head and
row/inner strides; Q/K/V/M/L/Out regions and context lengths; scale; INIT,
even-row flags and signed window selection. Every use is visible below. -/
```
```lean
abbrev Parameters := RealPrologue.Parameters
```
</details>

<details><summary><code>originalIO</code></summary>

```
/-- Q is boundary checked unless IS_EVEN_M is set. K/V cover the global
context. Old carry values are read only for INIT=0. M/Out are always written;
L is written only for END=0. The same six allocations supply in-place carry. -/
```
```lean
def originalIO (BM BN BD Z finish : Nat) (p : Parameters) : RealIO.IO BM BD p.layout.kvContext where
  kernel := (TypedSource.original p.layout.Q p.layout.K p.layout.V p.layout.M p.layout.Out p.layout.L p.scale
    p.strides.qBatch p.strides.qHead p.layout.qRow p.layout.qDim
    p.strides.kBatch p.strides.kHead p.layout.kRow p.layout.kDim
    p.strides.vBatch p.strides.vHead p.layout.vRow p.layout.vDim
    p.strides.oBatch p.strides.oHead p.layout.oRow p.layout.oDim
    Z p.heads p.kvHeads p.layout.queryContext p.layout.roundedContext p.layout.kvContext
    p.evenM p.evenN BM BD BN finish p.init p.window p.complement p.offset p.size).eraseDType
  projection := RealSource.projection BM BN BD Z finish p Bool.false
  Q := p.layout.Q
  K := p.layout.K
  V := p.layout.V
  M := p.layout.M
  L := p.layout.L
  Out := p.layout.Out
  queryWindow := fun p₀ p₁ _ i =>
    p₁ / p.heads * p.strides.qBatch + p₁ % p.heads * p.strides.qHead +
      (p₀ * BM + i.1.val) * p.layout.qRow + i.2.1.val * p.layout.qDim
  keyWindow := fun _ p₁ _ i =>
    p₁ / p.heads * p.strides.kBatch + ((p₁ % p.heads) / (p.heads / p.kvHeads)) * p.strides.kHead +
      i.2.1.val * p.layout.kDim + i.1.val * p.layout.kRow
  valueWindow := fun _ p₁ _ i =>
    p₁ / p.heads * p.strides.vBatch + ((p₁ % p.heads) / (p.heads / p.kvHeads)) * p.strides.vHead +
      i.1.val * p.layout.vRow + i.2.1.val * p.layout.vDim
  centerWindow := fun p₀ p₁ _ i => p₁ * p.layout.roundedContext + p₀ * BM + i.1.val
  denominatorWindow := fun p₀ p₁ _ i => p₁ * p.layout.roundedContext + p₀ * BM + i.1.val
  outputWindow := fun p₀ p₁ _ i =>
    p₁ / p.heads * p.strides.oBatch + p₁ % p.heads * p.strides.oHead +
      (p₀ * BM + i.1.val) * p.layout.oRow + i.2.1.val * p.layout.oDim
  queryMask := fun p₀ _ _ i => p.evenM ≠ 0 ∨ p₀ * BM + i.1.val < p.layout.queryContext
  readCarry := p.init = 0
  writeDenominator := finish = 0
```
</details>

<details><summary><code>expected</code></summary>

```
/-- END stores M+log2(L) and Out/L, preserving L memory. Continuation stores
all three carry values. Real division/log are total; bottom stored M reads as
zero under the framework's finite real-memory view. -/
```
```lean
def expected (BM BN BD finish start : Nat) (p : Parameters)
    (xs : Inputs BM BD p.layout.kvContext) : RealIO.Outputs BM BD :=
  let c := carry BM BN BD start p xs
  { center := fun r => (c.center r).elim 0 (fun m =>
      if decide (finish ≠ 0) then m + Real.log (c.denominator r) / Real.log 2 else m),
    denominator := c.denominator,
    output := fun i => if finish ≠ 0 then c.numerator i / c.denominator i.1 else c.numerator i }
```
</details>

<details><summary><code>original</code></summary>

```lean
def original
    (Q K V M Out L : RegionName) (sm_scale : ℝ)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      _Z H H_KV N_CTX ROUND_CTX NKV_CTX
      IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N END INIT
      SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW : Nat)
    (sliding_window_offset sliding_window_size : Int) :
    ComputeKernel := triton {
  tl.region Q = tl.float16, K = tl.float16, V = tl.float16, Out = tl.float16, M = tl.float32, L = tl.float32
  start_m = tl.program_id(0)
  off_hz = tl.program_id(1)
  off_z = off_hz // $(H)
  off_h = off_hz % $(H)
  off_hkv = off_h // ($(H) // $(H_KV))
  q_offset = off_z.to(tl.int64) * $(stride_qz) + off_h.to(tl.int64) * $(stride_qh)
  k_offset = off_z.to(tl.int64) * $(stride_kz) + off_hkv.to(tl.int64) * $(stride_kh)
  v_offset = off_z.to(tl.int64) * $(stride_vz) + off_hkv.to(tl.int64) * $(stride_vh)
  o_offset = off_z.to(tl.int64) * $(stride_oz) + off_h.to(tl.int64) * $(stride_oh)

  Q_block_ptr = tl.make_block_ptr(base=Q + q_offset, shape=($(N_CTX), $(BLOCK_DMODEL)),
    strides=($(stride_qm), $(stride_qk)), offsets=(start_m * $(BLOCK_M), 0),
    block_shape=($(BLOCK_M), $(BLOCK_DMODEL)), order=(1, 0))
  V_block_ptr = tl.make_block_ptr(base=V + v_offset, shape=($(NKV_CTX), $(BLOCK_DMODEL)),
    strides=($(stride_vk), $(stride_vn)), offsets=(0, 0),
    block_shape=($(BLOCK_N), $(BLOCK_DMODEL)), order=(1, 0))
  K_block_ptr = tl.make_block_ptr(base=K + k_offset, shape=($(BLOCK_DMODEL), $(NKV_CTX)),
    strides=($(stride_kk), $(stride_kn)), offsets=(0, 0),
    block_shape=($(BLOCK_DMODEL), $(BLOCK_N)), order=(0, 1))
  O_block_ptr = tl.make_block_ptr(base=Out + o_offset, shape=($(ROUND_CTX), $(BLOCK_DMODEL)),
    strides=($(stride_om), $(stride_on)), offsets=(start_m * $(BLOCK_M), 0),
    block_shape=($(BLOCK_M), $(BLOCK_DMODEL)), order=(1, 0))
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  m_ptrs = M + off_hz * $(ROUND_CTX) + offs_m
  l_ptrs = L + off_hz * $(ROUND_CTX) + offs_m
  if $(INIT) != $(0) {
    m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) + 1.0
    acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)
  } else {
    m_i = tl.load(m_ptrs).to(tl.float32)
    l_i = tl.load(l_ptrs).to(tl.float32)
    acc = tl.load(O_block_ptr).to(tl.float32)
  }
  qk_scale = $((sm_scale : ℝ))
  qk_scale *= 1.4426950408889634
  if $(IS_EVEN_M) != $(0) {
    q = tl.load(Q_block_ptr)
  } else {
    q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
  }
  if ($(SLIDING_WINDOW) != $(0)) & ($(COMPLEMENT_SLIDING_WINDOW) == $(0)) {
    if $(COMPLEMENT_SLIDING_WINDOW) != $(0) {
      lo = $((0 : Int))
      hi = (((start_m + 1) * $(BLOCK_M) + $((sliding_window_offset : Int))
        - $((sliding_window_size : Int)) + $(BLOCK_N) - 1) // $(BLOCK_N)) * $(BLOCK_N)
    } else {
      lo = ((start_m * $(BLOCK_M) + $((sliding_window_offset : Int))
        - $((sliding_window_size : Int)) + 1) // $(BLOCK_N)) * $(BLOCK_N)
      hi = ((((start_m + 1) * $(BLOCK_M) - 1) + $((sliding_window_offset : Int))
        + $(BLOCK_N)) // $(BLOCK_N)) * $(BLOCK_N)
      if lo < $((0 : Int)) {
        lo = $((0 : Int))
      }
      if hi > $(NKV_CTX) {
        hi = $(((NKV_CTX : Int) : Int))
      }
      lo = tl.multiple_of(lo, $(BLOCK_N))
      K_block_ptr = tl.advance(K_block_ptr, (0, lo))
      V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    }
  } else {
    lo = $((0 : Int))
    hi = $(((NKV_CTX : Int) : Int))
  }
  for start_n in range(lo, hi, $(BLOCK_N)) {
    start_n = tl.multiple_of(start_n, $(BLOCK_N))
    if $(IS_EVEN_N) != $(0) {
      k = tl.load(K_block_ptr)
    } else {
      k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
    }
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    qk += tl.dot(q, k)
    qk = qk * qk_scale
    if $(SLIDING_WINDOW) != $(0) {
      dist = tl.arange(0, $(BLOCK_M))[:, None] - tl.arange(0, $(BLOCK_N))[None, :]
        + start_m * $(BLOCK_M) - start_n + $((sliding_window_offset : Int))
      if $(COMPLEMENT_SLIDING_WINDOW) != $(0) {
        mask = dist >= $((sliding_window_size : Int))
      } else {
        mask = (dist >= $(0)) & (dist < $((sliding_window_size : Int)))
      }
      qk = tl.where(mask, qk, float("-inf"))
    }
    if $(IS_EVEN_N) == $(0) {
      qk = tl.where(((tl.arange(0, $(BLOCK_N)) + start_n) < $(NKV_CTX))[None, :], qk, float("-inf"))
    }
    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    qk = qk - m_ij[:, None]
    p = tl.math.exp2(qk)
    if $(SLIDING_WINDOW) != $(0) {
      p = tl.where(mask, p, 0.0)
    }
    if $(IS_EVEN_N) == $(0) {
      p = tl.where(((tl.arange(0, $(BLOCK_N)) + start_n) < $(NKV_CTX))[None, :], p, 0.0)
    }
    l_ij = tl.sum(p, 1)
    tmp = m_i - m_ij
    alpha_mask = tmp != tmp
    alpha = tl.math.exp2(tmp)
    alpha = tl.where(alpha_mask, 1.0, alpha)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]
    if $(IS_EVEN_N) != $(0) {
      v = tl.load(V_block_ptr)
    } else {
      v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
    }
    acc += tl.dot((p).to(tl.float16), v)
    m_i = m_ij
    V_block_ptr = tl.advance(V_block_ptr, [$(BLOCK_N), $(0)])
    K_block_ptr = tl.advance(K_block_ptr, [$(0), $(BLOCK_N)])
  }
  if $(END) != $(0) {
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
  } else {
    tl.store(l_ptrs, l_i)
  }
  tl.store(m_ptrs, m_i)
  tl.store(O_block_ptr, (acc).to(tl.float16))
}
```
</details>

<details><summary><code>carry</code></summary>

```lean
def carry (BM BN BD start : Nat) (p : Parameters) (xs : Inputs BM BD p.layout.kvContext) : RealLoopFormula.Carry BM BD where
  center := fun r => (row BM BN BD start p xs r 0 (fun _ => 0)).center
  denominator := fun r => (row BM BN BD start p xs r 0 (fun _ => 0)).denominator
  numerator := fun i => (row BM BN BD start p xs i.1
    ((initial BM BD p xs).numerator i) (fun k => xs.value k i.2.1)).numerator
```
</details>

<details><summary><code>row</code></summary>

```
/-- Regular windows keep 0 ≤ query-key+offset < size; the complement arm
keeps query-key+offset ≥ size. No loop iterations preserve the initial state,
which differs from executing a block whose row has no kept keys. -/
```
```lean
def row (BM BN BD start : Nat) (p : Parameters) (xs : Inputs BM BD p.layout.kvContext)
    (r : Fin BM) (initialValue : ℝ) (v : Fin p.layout.kvContext → ℝ) : RealFormula.RowState :=
  let seed := RealLoopFormula.rowSeed (initial BM BD p xs) r initialValue
  if (RealKeys.lower start BM BN p.window p.complement p.offset p.size : Int) <
      RealWindow.upper start BM BN p.layout.kvContext p.window p.complement p.offset then
    summarize seed
      (((List.finRange p.layout.kvContext).filter
        (fun k => RealKeys.keep start BM r.val p.window p.complement p.offset p.size k.val)).map
        (fun k => (score BM BD start p xs r k, v k)))
  else seed
```
</details>

<details><summary><code>initial</code></summary>

```lean
def initial (BM BD : Nat) (p : Parameters) (xs : Inputs BM BD p.layout.kvContext) : RealLoopFormula.Carry BM BD where
  center := fun r => if p.init ≠ 0 then ⊥ else some (xs.center r)
  denominator := fun r => if p.init ≠ 0 then 1 else xs.denominator r
  numerator := fun i => if p.init ≠ 0 then 0 else xs.output i.1 i.2.1
```
</details>

<details><summary><code>summarize</code></summary>

```
/-- For selected (score,value) pairs, m is their maximum together with the
initial center. The denominator and numerator are rescaled initial carry plus
Σ 2^(score-m) and Σ 2^(score-m)*value. Bottom centers have weight zero. -/
```
```lean
def summarize (seed : RealFormula.RowState) (pairs : List (ℝ × ℝ)) : RealFormula.RowState :=
  let m := seed.center ⊔ RealFormula.maxScore pairs
  { center := m,
    denominator := seed.denominator * RealFormula.weight seed.center m + RealFormula.sumAt pairs m (fun _ => 1),
    numerator := seed.numerator * RealFormula.weight seed.center m + RealFormula.sumAt pairs m Prod.snd }
```
</details>

<details><summary><code>score</code></summary>

```
/-- The exact decimal is the source's log2 scale conversion, retained in the
real projection. Inactive checked Q rows contribute zero. -/
```
```lean
def score (BM BD start : Nat) (p : Parameters) (xs : Inputs BM BD p.layout.kvContext)
    (r : Fin BM) (k : Fin p.layout.kvContext) : ℝ :=
  (∑ d : Fin BD,
    (if p.evenM ≠ 0 ∨ start * BM + r.val < p.layout.queryContext then xs.query r d else 0) *
      xs.key k d) * (p.scale * 1.4426950408889634)
```
</details>

## Public theorem: `attention_reciprocal_correctness`

**Statement:**
```lean
specification attention_reciprocal_correctness (BM BN BD Z finish : Nat) (p : Parameters)
    (hN : 0 < BN) (hflag : p.evenN ≠ 0 → p.layout.kvContext % BN = 0)
    (hd : 0 < p.layout.oDim) (hr : BD * p.layout.oDim ≤ p.layout.oRow)
    (hm : p.layout.M ≠ p.layout.Out) (hlm : p.layout.L ≠ p.layout.M) (hlo : p.layout.L ≠ p.layout.Out) :
    Spec.Real (reciprocalIO BM BN BD Z finish p ⊨ fun start _ xs => expected BM BN BD finish start p xs)
```

**Assumptions / layout contracts:**
- `hN : 0 < BN`
- `hflag : p.evenN ≠ 0 → p.layout.kvContext % BN = 0`
- `hd : 0 < p.layout.oDim`
- `hr : BD * p.layout.oDim ≤ p.layout.oRow`
- `hm : p.layout.M ≠ p.layout.Out`
- `hlm : p.layout.L ≠ p.layout.M`
- `hlo : p.layout.L ≠ p.layout.Out`

**Closed-form spec defs (transitive):** `Parameters`, `reciprocalIO`, `expected`, `originalIO`, `reciprocal`, `carry`, `original`, `row`, `initial`, `summarize`, `score`

<details><summary><code>Parameters</code></summary>

```
/-- Source parameters: query/KV head counts; independent batch/head and
row/inner strides; Q/K/V/M/L/Out regions and context lengths; scale; INIT,
even-row flags and signed window selection. Every use is visible below. -/
```
```lean
abbrev Parameters := RealPrologue.Parameters
```
</details>

<details><summary><code>reciprocalIO</code></summary>

```
/-- The reciprocal candidate has exactly the same windows and masks. -/
```
```lean
def reciprocalIO (BM BN BD Z finish : Nat) (p : Parameters) : RealIO.IO BM BD p.layout.kvContext :=
  { originalIO BM BN BD Z finish p with
    kernel := (TypedSource.reciprocal p.layout.Q p.layout.K p.layout.V p.layout.M p.layout.Out p.layout.L p.scale
    p.strides.qBatch p.strides.qHead p.layout.qRow p.layout.qDim
    p.strides.kBatch p.strides.kHead p.layout.kRow p.layout.kDim
    p.strides.vBatch p.strides.vHead p.layout.vRow p.layout.vDim
    p.strides.oBatch p.strides.oHead p.layout.oRow p.layout.oDim
    Z p.heads p.kvHeads p.layout.queryContext p.layout.roundedContext p.layout.kvContext
    p.evenM p.evenN BM BD BN finish p.init p.window p.complement p.offset p.size).eraseDType
    projection := RealSource.projection BM BN BD Z finish p Bool.true }
```
</details>

<details><summary><code>expected</code></summary>

```
/-- END stores M+log2(L) and Out/L, preserving L memory. Continuation stores
all three carry values. Real division/log are total; bottom stored M reads as
zero under the framework's finite real-memory view. -/
```
```lean
def expected (BM BN BD finish start : Nat) (p : Parameters)
    (xs : Inputs BM BD p.layout.kvContext) : RealIO.Outputs BM BD :=
  let c := carry BM BN BD start p xs
  { center := fun r => (c.center r).elim 0 (fun m =>
      if decide (finish ≠ 0) then m + Real.log (c.denominator r) / Real.log 2 else m),
    denominator := c.denominator,
    output := fun i => if finish ≠ 0 then c.numerator i / c.denominator i.1 else c.numerator i }
```
</details>

<details><summary><code>originalIO</code></summary>

```
/-- Q is boundary checked unless IS_EVEN_M is set. K/V cover the global
context. Old carry values are read only for INIT=0. M/Out are always written;
L is written only for END=0. The same six allocations supply in-place carry. -/
```
```lean
def originalIO (BM BN BD Z finish : Nat) (p : Parameters) : RealIO.IO BM BD p.layout.kvContext where
  kernel := (TypedSource.original p.layout.Q p.layout.K p.layout.V p.layout.M p.layout.Out p.layout.L p.scale
    p.strides.qBatch p.strides.qHead p.layout.qRow p.layout.qDim
    p.strides.kBatch p.strides.kHead p.layout.kRow p.layout.kDim
    p.strides.vBatch p.strides.vHead p.layout.vRow p.layout.vDim
    p.strides.oBatch p.strides.oHead p.layout.oRow p.layout.oDim
    Z p.heads p.kvHeads p.layout.queryContext p.layout.roundedContext p.layout.kvContext
    p.evenM p.evenN BM BD BN finish p.init p.window p.complement p.offset p.size).eraseDType
  projection := RealSource.projection BM BN BD Z finish p Bool.false
  Q := p.layout.Q
  K := p.layout.K
  V := p.layout.V
  M := p.layout.M
  L := p.layout.L
  Out := p.layout.Out
  queryWindow := fun p₀ p₁ _ i =>
    p₁ / p.heads * p.strides.qBatch + p₁ % p.heads * p.strides.qHead +
      (p₀ * BM + i.1.val) * p.layout.qRow + i.2.1.val * p.layout.qDim
  keyWindow := fun _ p₁ _ i =>
    p₁ / p.heads * p.strides.kBatch + ((p₁ % p.heads) / (p.heads / p.kvHeads)) * p.strides.kHead +
      i.2.1.val * p.layout.kDim + i.1.val * p.layout.kRow
  valueWindow := fun _ p₁ _ i =>
    p₁ / p.heads * p.strides.vBatch + ((p₁ % p.heads) / (p.heads / p.kvHeads)) * p.strides.vHead +
      i.1.val * p.layout.vRow + i.2.1.val * p.layout.vDim
  centerWindow := fun p₀ p₁ _ i => p₁ * p.layout.roundedContext + p₀ * BM + i.1.val
  denominatorWindow := fun p₀ p₁ _ i => p₁ * p.layout.roundedContext + p₀ * BM + i.1.val
  outputWindow := fun p₀ p₁ _ i =>
    p₁ / p.heads * p.strides.oBatch + p₁ % p.heads * p.strides.oHead +
      (p₀ * BM + i.1.val) * p.layout.oRow + i.2.1.val * p.layout.oDim
  queryMask := fun p₀ _ _ i => p.evenM ≠ 0 ∨ p₀ * BM + i.1.val < p.layout.queryContext
  readCarry := p.init = 0
  writeDenominator := finish = 0
```
</details>

<details><summary><code>reciprocal</code></summary>

```
/-- Reciprocal normalization, preserving the full original context. -/
```
```lean
def reciprocal
    (Q K V M Out L : RegionName) (sm_scale : ℝ)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      _Z H H_KV N_CTX ROUND_CTX NKV_CTX
      IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N END INIT
      SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW : Nat)
    (sliding_window_offset sliding_window_size : Int) :
    ComputeKernel := triton {
  tl.region Q = tl.float16, K = tl.float16, V = tl.float16, Out = tl.float16, M = tl.float32, L = tl.float32
  start_m = tl.program_id(0)
  off_hz = tl.program_id(1)
  off_z = off_hz // $(H)
  off_h = off_hz % $(H)
  off_hkv = off_h // ($(H) // $(H_KV))
  q_offset = off_z.to(tl.int64) * $(stride_qz) + off_h.to(tl.int64) * $(stride_qh)
  k_offset = off_z.to(tl.int64) * $(stride_kz) + off_hkv.to(tl.int64) * $(stride_kh)
  v_offset = off_z.to(tl.int64) * $(stride_vz) + off_hkv.to(tl.int64) * $(stride_vh)
  o_offset = off_z.to(tl.int64) * $(stride_oz) + off_h.to(tl.int64) * $(stride_oh)

  Q_block_ptr = tl.make_block_ptr(base=Q + q_offset, shape=($(N_CTX), $(BLOCK_DMODEL)),
    strides=($(stride_qm), $(stride_qk)), offsets=(start_m * $(BLOCK_M), 0),
    block_shape=($(BLOCK_M), $(BLOCK_DMODEL)), order=(1, 0))
  V_block_ptr = tl.make_block_ptr(base=V + v_offset, shape=($(NKV_CTX), $(BLOCK_DMODEL)),
    strides=($(stride_vk), $(stride_vn)), offsets=(0, 0),
    block_shape=($(BLOCK_N), $(BLOCK_DMODEL)), order=(1, 0))
  K_block_ptr = tl.make_block_ptr(base=K + k_offset, shape=($(BLOCK_DMODEL), $(NKV_CTX)),
    strides=($(stride_kk), $(stride_kn)), offsets=(0, 0),
    block_shape=($(BLOCK_DMODEL), $(BLOCK_N)), order=(0, 1))
  O_block_ptr = tl.make_block_ptr(base=Out + o_offset, shape=($(ROUND_CTX), $(BLOCK_DMODEL)),
    strides=($(stride_om), $(stride_on)), offsets=(start_m * $(BLOCK_M), 0),
    block_shape=($(BLOCK_M), $(BLOCK_DMODEL)), order=(1, 0))
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  m_ptrs = M + off_hz * $(ROUND_CTX) + offs_m
  l_ptrs = L + off_hz * $(ROUND_CTX) + offs_m
  if $(INIT) != $(0) {
    m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) + 1.0
    acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)
  } else {
    m_i = tl.load(m_ptrs).to(tl.float32)
    l_i = tl.load(l_ptrs).to(tl.float32)
    acc = tl.load(O_block_ptr).to(tl.float32)
  }
  qk_scale = $((sm_scale : ℝ))
  qk_scale *= 1.4426950408889634
  if $(IS_EVEN_M) != $(0) {
    q = tl.load(Q_block_ptr)
  } else {
    q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
  }
  if ($(SLIDING_WINDOW) != $(0)) & ($(COMPLEMENT_SLIDING_WINDOW) == $(0)) {
    if $(COMPLEMENT_SLIDING_WINDOW) != $(0) {
      lo = $((0 : Int))
      hi = (((start_m + 1) * $(BLOCK_M) + $((sliding_window_offset : Int))
        - $((sliding_window_size : Int)) + $(BLOCK_N) - 1) // $(BLOCK_N)) * $(BLOCK_N)
    } else {
      lo = ((start_m * $(BLOCK_M) + $((sliding_window_offset : Int))
        - $((sliding_window_size : Int)) + 1) // $(BLOCK_N)) * $(BLOCK_N)
      hi = ((((start_m + 1) * $(BLOCK_M) - 1) + $((sliding_window_offset : Int))
        + $(BLOCK_N)) // $(BLOCK_N)) * $(BLOCK_N)
      if lo < $((0 : Int)) {
        lo = $((0 : Int))
      }
      if hi > $(NKV_CTX) {
        hi = $(((NKV_CTX : Int) : Int))
      }
      lo = tl.multiple_of(lo, $(BLOCK_N))
      K_block_ptr = tl.advance(K_block_ptr, (0, lo))
      V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    }
  } else {
    lo = $((0 : Int))
    hi = $(((NKV_CTX : Int) : Int))
  }
  for start_n in range(lo, hi, $(BLOCK_N)) {
    start_n = tl.multiple_of(start_n, $(BLOCK_N))
    if $(IS_EVEN_N) != $(0) {
      k = tl.load(K_block_ptr)
    } else {
      k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
    }
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    qk += tl.dot(q, k)
    qk = qk * qk_scale
    if $(SLIDING_WINDOW) != $(0) {
      dist = tl.arange(0, $(BLOCK_M))[:, None] - tl.arange(0, $(BLOCK_N))[None, :]
        + start_m * $(BLOCK_M) - start_n + $((sliding_window_offset : Int))
      if $(COMPLEMENT_SLIDING_WINDOW) != $(0) {
        mask = dist >= $((sliding_window_size : Int))
      } else {
        mask = (dist >= $(0)) & (dist < $((sliding_window_size : Int)))
      }
      qk = tl.where(mask, qk, float("-inf"))
    }
    if $(IS_EVEN_N) == $(0) {
      qk = tl.where(((tl.arange(0, $(BLOCK_N)) + start_n) < $(NKV_CTX))[None, :], qk, float("-inf"))
    }
    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    qk = qk - m_ij[:, None]
    p = tl.math.exp2(qk)
    if $(SLIDING_WINDOW) != $(0) {
      p = tl.where(mask, p, 0.0)
    }
    if $(IS_EVEN_N) == $(0) {
      p = tl.where(((tl.arange(0, $(BLOCK_N)) + start_n) < $(NKV_CTX))[None, :], p, 0.0)
    }
    l_ij = tl.sum(p, 1)
    tmp = m_i - m_ij
    alpha_mask = tmp != tmp
    alpha = tl.math.exp2(tmp)
    alpha = tl.where(alpha_mask, 1.0, alpha)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]
    if $(IS_EVEN_N) != $(0) {
      v = tl.load(V_block_ptr)
    } else {
      v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
    }
    acc += tl.dot((p).to(tl.float16), v)
    m_i = m_ij
    V_block_ptr = tl.advance(V_block_ptr, [$(BLOCK_N), $(0)])
    K_block_ptr = tl.advance(K_block_ptr, [$(0), $(BLOCK_N)])
  }
  if $(END) != $(0) {
    m_i += tl.math.log2(l_i)
    acc = acc * (1.0 / l_i[:, None])
  } else {
    tl.store(l_ptrs, l_i)
  }
  tl.store(m_ptrs, m_i)
  tl.store(O_block_ptr, (acc).to(tl.float16))
}
```
</details>

<details><summary><code>carry</code></summary>

```lean
def carry (BM BN BD start : Nat) (p : Parameters) (xs : Inputs BM BD p.layout.kvContext) : RealLoopFormula.Carry BM BD where
  center := fun r => (row BM BN BD start p xs r 0 (fun _ => 0)).center
  denominator := fun r => (row BM BN BD start p xs r 0 (fun _ => 0)).denominator
  numerator := fun i => (row BM BN BD start p xs i.1
    ((initial BM BD p xs).numerator i) (fun k => xs.value k i.2.1)).numerator
```
</details>

<details><summary><code>original</code></summary>

```lean
def original
    (Q K V M Out L : RegionName) (sm_scale : ℝ)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      _Z H H_KV N_CTX ROUND_CTX NKV_CTX
      IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N END INIT
      SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW : Nat)
    (sliding_window_offset sliding_window_size : Int) :
    ComputeKernel := triton {
  tl.region Q = tl.float16, K = tl.float16, V = tl.float16, Out = tl.float16, M = tl.float32, L = tl.float32
  start_m = tl.program_id(0)
  off_hz = tl.program_id(1)
  off_z = off_hz // $(H)
  off_h = off_hz % $(H)
  off_hkv = off_h // ($(H) // $(H_KV))
  q_offset = off_z.to(tl.int64) * $(stride_qz) + off_h.to(tl.int64) * $(stride_qh)
  k_offset = off_z.to(tl.int64) * $(stride_kz) + off_hkv.to(tl.int64) * $(stride_kh)
  v_offset = off_z.to(tl.int64) * $(stride_vz) + off_hkv.to(tl.int64) * $(stride_vh)
  o_offset = off_z.to(tl.int64) * $(stride_oz) + off_h.to(tl.int64) * $(stride_oh)

  Q_block_ptr = tl.make_block_ptr(base=Q + q_offset, shape=($(N_CTX), $(BLOCK_DMODEL)),
    strides=($(stride_qm), $(stride_qk)), offsets=(start_m * $(BLOCK_M), 0),
    block_shape=($(BLOCK_M), $(BLOCK_DMODEL)), order=(1, 0))
  V_block_ptr = tl.make_block_ptr(base=V + v_offset, shape=($(NKV_CTX), $(BLOCK_DMODEL)),
    strides=($(stride_vk), $(stride_vn)), offsets=(0, 0),
    block_shape=($(BLOCK_N), $(BLOCK_DMODEL)), order=(1, 0))
  K_block_ptr = tl.make_block_ptr(base=K + k_offset, shape=($(BLOCK_DMODEL), $(NKV_CTX)),
    strides=($(stride_kk), $(stride_kn)), offsets=(0, 0),
    block_shape=($(BLOCK_DMODEL), $(BLOCK_N)), order=(0, 1))
  O_block_ptr = tl.make_block_ptr(base=Out + o_offset, shape=($(ROUND_CTX), $(BLOCK_DMODEL)),
    strides=($(stride_om), $(stride_on)), offsets=(start_m * $(BLOCK_M), 0),
    block_shape=($(BLOCK_M), $(BLOCK_DMODEL)), order=(1, 0))
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  m_ptrs = M + off_hz * $(ROUND_CTX) + offs_m
  l_ptrs = L + off_hz * $(ROUND_CTX) + offs_m
  if $(INIT) != $(0) {
    m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) + 1.0
    acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)
  } else {
    m_i = tl.load(m_ptrs).to(tl.float32)
    l_i = tl.load(l_ptrs).to(tl.float32)
    acc = tl.load(O_block_ptr).to(tl.float32)
  }
  qk_scale = $((sm_scale : ℝ))
  qk_scale *= 1.4426950408889634
  if $(IS_EVEN_M) != $(0) {
    q = tl.load(Q_block_ptr)
  } else {
    q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
  }
  if ($(SLIDING_WINDOW) != $(0)) & ($(COMPLEMENT_SLIDING_WINDOW) == $(0)) {
    if $(COMPLEMENT_SLIDING_WINDOW) != $(0) {
      lo = $((0 : Int))
      hi = (((start_m + 1) * $(BLOCK_M) + $((sliding_window_offset : Int))
        - $((sliding_window_size : Int)) + $(BLOCK_N) - 1) // $(BLOCK_N)) * $(BLOCK_N)
    } else {
      lo = ((start_m * $(BLOCK_M) + $((sliding_window_offset : Int))
        - $((sliding_window_size : Int)) + 1) // $(BLOCK_N)) * $(BLOCK_N)
      hi = ((((start_m + 1) * $(BLOCK_M) - 1) + $((sliding_window_offset : Int))
        + $(BLOCK_N)) // $(BLOCK_N)) * $(BLOCK_N)
      if lo < $((0 : Int)) {
        lo = $((0 : Int))
      }
      if hi > $(NKV_CTX) {
        hi = $(((NKV_CTX : Int) : Int))
      }
      lo = tl.multiple_of(lo, $(BLOCK_N))
      K_block_ptr = tl.advance(K_block_ptr, (0, lo))
      V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    }
  } else {
    lo = $((0 : Int))
    hi = $(((NKV_CTX : Int) : Int))
  }
  for start_n in range(lo, hi, $(BLOCK_N)) {
    start_n = tl.multiple_of(start_n, $(BLOCK_N))
    if $(IS_EVEN_N) != $(0) {
      k = tl.load(K_block_ptr)
    } else {
      k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
    }
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    qk += tl.dot(q, k)
    qk = qk * qk_scale
    if $(SLIDING_WINDOW) != $(0) {
      dist = tl.arange(0, $(BLOCK_M))[:, None] - tl.arange(0, $(BLOCK_N))[None, :]
        + start_m * $(BLOCK_M) - start_n + $((sliding_window_offset : Int))
      if $(COMPLEMENT_SLIDING_WINDOW) != $(0) {
        mask = dist >= $((sliding_window_size : Int))
      } else {
        mask = (dist >= $(0)) & (dist < $((sliding_window_size : Int)))
      }
      qk = tl.where(mask, qk, float("-inf"))
    }
    if $(IS_EVEN_N) == $(0) {
      qk = tl.where(((tl.arange(0, $(BLOCK_N)) + start_n) < $(NKV_CTX))[None, :], qk, float("-inf"))
    }
    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    qk = qk - m_ij[:, None]
    p = tl.math.exp2(qk)
    if $(SLIDING_WINDOW) != $(0) {
      p = tl.where(mask, p, 0.0)
    }
    if $(IS_EVEN_N) == $(0) {
      p = tl.where(((tl.arange(0, $(BLOCK_N)) + start_n) < $(NKV_CTX))[None, :], p, 0.0)
    }
    l_ij = tl.sum(p, 1)
    tmp = m_i - m_ij
    alpha_mask = tmp != tmp
    alpha = tl.math.exp2(tmp)
    alpha = tl.where(alpha_mask, 1.0, alpha)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]
    if $(IS_EVEN_N) != $(0) {
      v = tl.load(V_block_ptr)
    } else {
      v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
    }
    acc += tl.dot((p).to(tl.float16), v)
    m_i = m_ij
    V_block_ptr = tl.advance(V_block_ptr, [$(BLOCK_N), $(0)])
    K_block_ptr = tl.advance(K_block_ptr, [$(0), $(BLOCK_N)])
  }
  if $(END) != $(0) {
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
  } else {
    tl.store(l_ptrs, l_i)
  }
  tl.store(m_ptrs, m_i)
  tl.store(O_block_ptr, (acc).to(tl.float16))
}
```
</details>

<details><summary><code>row</code></summary>

```
/-- Regular windows keep 0 ≤ query-key+offset < size; the complement arm
keeps query-key+offset ≥ size. No loop iterations preserve the initial state,
which differs from executing a block whose row has no kept keys. -/
```
```lean
def row (BM BN BD start : Nat) (p : Parameters) (xs : Inputs BM BD p.layout.kvContext)
    (r : Fin BM) (initialValue : ℝ) (v : Fin p.layout.kvContext → ℝ) : RealFormula.RowState :=
  let seed := RealLoopFormula.rowSeed (initial BM BD p xs) r initialValue
  if (RealKeys.lower start BM BN p.window p.complement p.offset p.size : Int) <
      RealWindow.upper start BM BN p.layout.kvContext p.window p.complement p.offset then
    summarize seed
      (((List.finRange p.layout.kvContext).filter
        (fun k => RealKeys.keep start BM r.val p.window p.complement p.offset p.size k.val)).map
        (fun k => (score BM BD start p xs r k, v k)))
  else seed
```
</details>

<details><summary><code>initial</code></summary>

```lean
def initial (BM BD : Nat) (p : Parameters) (xs : Inputs BM BD p.layout.kvContext) : RealLoopFormula.Carry BM BD where
  center := fun r => if p.init ≠ 0 then ⊥ else some (xs.center r)
  denominator := fun r => if p.init ≠ 0 then 1 else xs.denominator r
  numerator := fun i => if p.init ≠ 0 then 0 else xs.output i.1 i.2.1
```
</details>

<details><summary><code>summarize</code></summary>

```
/-- For selected (score,value) pairs, m is their maximum together with the
initial center. The denominator and numerator are rescaled initial carry plus
Σ 2^(score-m) and Σ 2^(score-m)*value. Bottom centers have weight zero. -/
```
```lean
def summarize (seed : RealFormula.RowState) (pairs : List (ℝ × ℝ)) : RealFormula.RowState :=
  let m := seed.center ⊔ RealFormula.maxScore pairs
  { center := m,
    denominator := seed.denominator * RealFormula.weight seed.center m + RealFormula.sumAt pairs m (fun _ => 1),
    numerator := seed.numerator * RealFormula.weight seed.center m + RealFormula.sumAt pairs m Prod.snd }
```
</details>

<details><summary><code>score</code></summary>

```
/-- The exact decimal is the source's log2 scale conversion, retained in the
real projection. Inactive checked Q rows contribute zero. -/
```
```lean
def score (BM BD start : Nat) (p : Parameters) (xs : Inputs BM BD p.layout.kvContext)
    (r : Fin BM) (k : Fin p.layout.kvContext) : ℝ :=
  (∑ d : Fin BD,
    (if p.evenM ≠ 0 ∨ start * BM + r.val < p.layout.queryContext then xs.query r d else 0) *
      xs.key k d) * (p.scale * 1.4426950408889634)
```
</details>
