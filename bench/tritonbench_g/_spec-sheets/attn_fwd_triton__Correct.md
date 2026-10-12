# Spec sheet — `bench/tritonbench_g/attn_fwd_triton/Correct.lean`

**Python source:** `bench/tritonbench_g/attn_fwd_triton/attn_fwd_triton.py`

## Public theorem: `attention_correctness`

**Statement:**
```lean
specification attention_correctness (BM BN BD : Nat) (p : Parameters) (hBN : 0 < BN) :
    Spec.Real (originalIO BM BN BD p ⊨ fun startM _ xs => expected BM BN BD startM p xs)
```

**Assumptions / layout contracts:**
- `hBN : 0 < BN`

**Closed-form spec defs (transitive):** `Parameters`, `originalIO`, `expected`, `original`, `addresses`, `pairs`, `visits`, `score`, `rangeVisits`

<details><summary><code>Parameters</code></summary>

```
/-- Named Q/K/V/QS/KS/Out buffers, all source strides, batch/head/context
sizes and the legal outer stage. Source-unused strides remain present. -/
```
```lean
abbrev Parameters := RealProofs.Parameters
```
</details>

<details><summary><code>originalIO</code></summary>

```lean
def originalIO (BM BN BD : Nat) (p : Parameters) : AttnFwdCausal.RealIO.IO BM BN BD where
  kernel := (TypedSource.original p.Q p.K p.V p.QS p.KS p.Out
    p.queryBatch p.queryHead p.queryRow p.queryDim p.keyBatch p.keyHead p.keyRow p.keyDim
    p.valueBatch p.valueHead p.valueRow p.valueDim p.outputBatch p.outputHead p.outputRow p.outputDim
    p.batches p.heads p.context p.headDim BM BN BD p.activeHead p.stage p.validStage).eraseDType
  projection := RealProofs.projection BM BN BD p Bool.false
  configuration := addresses BM BN p
  out := p.Out
  stage := p.stage
```
</details>

<details><summary><code>expected</code></summary>

```
/-- Independent normalized base-2 weighted sum of the visited input values. -/
```
```lean
def expected (BM BN BD startM : Nat) (p : Parameters) (xs : Inputs BM BN BD)
    (i : TileIndex [BM,BD]) : ℝ :=
  let values := pairs BM BN BD startM p xs i.1 i.2.1
  (values.map (fun sv => pow2 sv.1 * sv.2)).sum / (values.map (fun sv => pow2 sv.1)).sum
```
</details>

<details><summary><code>original</code></summary>

```lean
def original
    (Q K V Q_scale K_scale Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk
      _stride_kz _stride_kh stride_kn _stride_kk
      _stride_vz _stride_vh _stride_vk _stride_vn
      _stride_oz _stride_oh _stride_om _stride_on
      _Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE : Nat)
    (_hStage : STAGE = 2 ∨ STAGE = 3) : ComputeKernel :=
  let FIRST_DIAGONAL := decide (STAGE = 2)
  .mk [Q,K,V,Q_scale,K_scale] [Out] (triton {
  tl.region Q = tl.bfloat16, K = tl.bfloat16, V = tl.bfloat16,
    Q_scale = tl.float32, K_scale = tl.float32, Out = tl.bfloat16
  start_m = tl.program_id(0)
  off_hz = tl.program_id(1)

  off_z = off_hz // $(H)
  off_h = off_hz % $(H)
  qvk_offset = (off_z).to(tl.int64) * $(stride_qz) + (off_h).to(tl.int64) * $(stride_qh)
  vk_offset = qvk_offset // $(stride_qm)
  q_scale_offset = off_hz * tl.cdiv($(N_CTX), $(BLOCK_M))
  k_scale_offset = off_hz * tl.cdiv($(N_CTX), $(BLOCK_N))

  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_n = tl.arange(0, $(BLOCK_N))
  offs_k = tl.arange(0, $(BLOCK_DMODEL))
  Q_ptrs = Q + qvk_offset + offs_m[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk)
  Q_scale_ptr = Q_scale + q_scale_offset + start_m
  K_ptrs = K + qvk_offset + offs_k[:, None] + offs_n[None, :] * $(stride_kn)
  K_scale_ptr = K_scale + k_scale_offset
  V_ptrs = V + qvk_offset + offs_n[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk)
  O_block_ptr = Out + qvk_offset + offs_m[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk)
  m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) + 1.0
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)
  q = tl.load(Q_ptrs,
    mask=(offs_m[:, None] < $(N_CTX)) & (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[None, :])
  q_scale = tl.load(Q_scale_ptr)
  -- Local helper pointers: the first call does not mutate the caller's bases.
  first_lo = 0
  first_hi = start_m * $(BLOCK_M)
  K_ptrs_first = K_ptrs
  K_scale_ptr_first = K_scale_ptr
  V_ptrs_first = V_ptrs
  if FIRST_DIAGONAL {
    first_lo = start_m * $(BLOCK_M)
    first_hi = (start_m + $(1)) * $(BLOCK_M)
    first_lo = tl.multiple_of(first_lo, $(BLOCK_M))
    K_scale_ptr_first += first_lo // $(BLOCK_N)
    K_ptrs_first += $(HEAD_DIM) * first_lo
    V_ptrs_first += $(HEAD_DIM) * first_lo
  }
  for start_n in range(first_lo, first_hi, $(BLOCK_N)) {
    start_n = tl.multiple_of(start_n, $(BLOCK_N))
    k_mask = (offs_n[None, :] < ($(N_CTX) - start_n)) &
      (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[:, None]
    k = tl.load(K_ptrs_first, mask=k_mask)
    k_scale = tl.load(K_scale_ptr_first)
    qk = (tl.dot(q, k)).to(tl.float32) * q_scale * k_scale
    if FIRST_DIAGONAL {
      mask_first = offs_m[:, None] >= (start_n + offs_n[None, :])
      qk = qk + tl.where(mask_first, 0, -1.0e6)
      m_ij = tl.maximum(m_i, tl.max(qk, 1))
      qk -= m_ij[:, None]
    } else {
      m_ij = tl.maximum(m_i, tl.max(qk, 1))
      qk = qk - m_ij[:, None]
    }
    p = tl.math.exp2(qk)
    l_ij = tl.sum(p, 1)
    alpha = tl.math.exp2(m_i - m_ij)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]
    v = tl.load(V_ptrs_first,
      mask=(offs_n[:, None] < ($(N_CTX) - start_n)) &
        (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[None, :])
    p = (p).to(tl.float16)
    acc += tl.dot(p, (v).to(tl.float16), out_dtype=tl.float16)
    m_i = m_ij
    K_ptrs_first += $(BLOCK_N) * $(HEAD_DIM)
    K_scale_ptr_first += $(1)
    V_ptrs_first += $(BLOCK_N) * $(HEAD_DIM)
  }
  -- The second helper starts from the caller's original pointers and advances
  -- to the diagonal block, exactly as its STAGE=2 prefix does.
  second_lo = start_m * $(BLOCK_M)
  second_hi = (start_m + $(1)) * $(BLOCK_M)
  second_lo = tl.multiple_of(second_lo, $(BLOCK_M))
  K_scale_ptr_second = K_scale_ptr + second_lo // $(BLOCK_N)
  K_ptrs_second = K_ptrs + $(HEAD_DIM) * second_lo
  V_ptrs_second = V_ptrs + $(HEAD_DIM) * second_lo
  for start_n in range(second_lo, second_hi, $(BLOCK_N)) {
    start_n = tl.multiple_of(start_n, $(BLOCK_N))
    k_mask = (offs_n[None, :] < ($(N_CTX) - start_n)) &
      (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[:, None]
    k = tl.load(K_ptrs_second, mask=k_mask)
    k_scale = tl.load(K_scale_ptr_second)
    qk = (tl.dot(q, k)).to(tl.float32) * q_scale * k_scale
    mask = offs_m[:, None] >= (start_n + offs_n[None, :])
    qk = qk + tl.where(mask, 0, -1.0e6)
    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    qk -= m_ij[:, None]
    p = tl.math.exp2(qk)
    l_ij = tl.sum(p, 1)
    alpha = tl.math.exp2(m_i - m_ij)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]
    v = tl.load(V_ptrs_second,
      mask=(offs_n[:, None] < ($(N_CTX) - start_n)) &
        (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[None, :])
    p = (p).to(tl.float16)
    acc += tl.dot(p, (v).to(tl.float16), out_dtype=tl.float16)
    m_i = m_ij
    K_ptrs_second += $(BLOCK_N) * $(HEAD_DIM)
    K_scale_ptr_second += $(1)
    V_ptrs_second += $(BLOCK_N) * $(HEAD_DIM)
  }
  acc = acc / l_i[:, None]
  -- Out.type.element_ty is bf16 in the Python wrapper.
  tl.store(O_block_ptr, (acc).to(tl.bfloat16),
    mask=(offs_m[:, None] < $(N_CTX)) & (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[None, :])
}).surfaceBody
```
</details>

<details><summary><code>addresses</code></summary>

```
/-- The contract uses these exact source addresses for tile origin `start`:
Q/Out = base+(startM*BM+row)*queryRow+d*queryDim;
K = base+d+j*keyRow+headDim*start;
V = base+j*queryRow+d*queryDim+headDim*start.
Q/Out require row<N and d<activeHead; K/V require start+j<N and d<activeHead.
QS[queryScaleBase+startM] and KS[keyScaleBase+start/BN] are unmasked reads,
including key scales for fully padded tiles. Only actual visits are pinned.
There is no scratch allocation. -/
```
```lean
def addresses (BM BN : Nat) (p : Parameters) (startM headBatch : Nat) : AttnFwdCausal.RealLoads.Config where
  Q := p.Q
  K := p.K
  V := p.V
  QS := p.QS
  KS := p.KS
  base := headBatch/p.heads*p.queryBatch+headBatch%p.heads*p.queryHead
  queryScaleBase := headBatch*((p.context+BM-1)/BM)
  keyScaleBase := headBatch*((p.context+BN-1)/BN)
  strideM := p.queryRow
  strideD := p.queryDim
  strideKN := p.keyRow
  headDim := p.headDim
  context := p.context
  activeHead := p.activeHead
  startM := startM
```
</details>

<details><summary><code>pairs</code></summary>

```lean
def pairs (BM BN BD startM : Nat) (p : Parameters) (xs : Inputs BM BN BD)
    (row : Fin BM) (d : Fin BD) : List (ℝ × ℝ) :=
  ((visits BM BN startM p).map (fun v => (List.finRange BN).map (fun key =>
    (score BM BN BD startM p xs v row key +
      (if v.diagonal = Bool.true ∧ startM*BM+row.val < v.start+key.val then -1000000 else 0),
     if v.start+key.val < p.context ∧ d.val < p.activeHead then xs.value v.start key d else 0)))).flatten
```
</details>

<details><summary><code>visits</code></summary>

```
/-- Retain repeated visits and partial final tiles exactly as in the source. -/
```
```lean
def visits (BM BN startM : Nat) (p : Parameters) : List Visit :=
  (if p.stage = 2 then rangeVisits (startM*BM) ((startM+1)*BM) BN Bool.true
    else rangeVisits 0 (startM*BM) BN Bool.false) ++
    rangeVisits (startM*BM) ((startM+1)*BM) BN Bool.true
```
</details>

<details><summary><code>score</code></summary>

```lean
def score (BM BN BD startM : Nat) (p : Parameters) (xs : Inputs BM BN BD)
    (v : Visit) (row : Fin BM) (key : Fin BN) : ℝ :=
  (∑ d : Fin BD,
    (if startM*BM+row.val < p.context ∧ d.val < p.activeHead then xs.query row d else 0) *
    (if v.start+key.val < p.context ∧ d.val < p.activeHead then xs.key v.start key d else 0)) *
      xs.queryScale * xs.keyScale (v.start/BN)
```
</details>

<details><summary><code>rangeVisits</code></summary>

```lean
def rangeVisits (lower upper BN : Nat) (diagonal : Bool) : List Visit :=
  (List.range ((upper-lower+BN-1)/BN)).map (fun j => ⟨lower+j*BN,diagonal⟩)
```
</details>

## Public theorem: `attention_reciprocal_correctness`

**Statement:**
```lean
specification attention_reciprocal_correctness (BM BN BD : Nat) (p : Parameters) (hBN : 0 < BN) :
    Spec.Real (reciprocalIO BM BN BD p ⊨ fun startM _ xs => expected BM BN BD startM p xs)
```

**Assumptions / layout contracts:**
- `hBN : 0 < BN`

**Closed-form spec defs (transitive):** `Parameters`, `reciprocalIO`, `expected`, `originalIO`, `reciprocal`, `pairs`, `original`, `addresses`, `visits`, `score`, `rangeVisits`

<details><summary><code>Parameters</code></summary>

```
/-- Named Q/K/V/QS/KS/Out buffers, all source strides, batch/head/context
sizes and the legal outer stage. Source-unused strides remain present. -/
```
```lean
abbrev Parameters := RealProofs.Parameters
```
</details>

<details><summary><code>reciprocalIO</code></summary>

```lean
def reciprocalIO (BM BN BD : Nat) (p : Parameters) : AttnFwdCausal.RealIO.IO BM BN BD :=
  { originalIO BM BN BD p with
    kernel := (TypedSource.reciprocal p.Q p.K p.V p.QS p.KS p.Out
      p.queryBatch p.queryHead p.queryRow p.queryDim p.keyBatch p.keyHead p.keyRow p.keyDim
      p.valueBatch p.valueHead p.valueRow p.valueDim p.outputBatch p.outputHead p.outputRow p.outputDim
      p.batches p.heads p.context p.headDim BM BN BD p.activeHead p.stage p.validStage).eraseDType
    projection := RealProofs.projection BM BN BD p Bool.true }
```
</details>

<details><summary><code>expected</code></summary>

```
/-- Independent normalized base-2 weighted sum of the visited input values. -/
```
```lean
def expected (BM BN BD startM : Nat) (p : Parameters) (xs : Inputs BM BN BD)
    (i : TileIndex [BM,BD]) : ℝ :=
  let values := pairs BM BN BD startM p xs i.1 i.2.1
  (values.map (fun sv => pow2 sv.1 * sv.2)).sum / (values.map (fun sv => pow2 sv.1)).sum
```
</details>

<details><summary><code>originalIO</code></summary>

```lean
def originalIO (BM BN BD : Nat) (p : Parameters) : AttnFwdCausal.RealIO.IO BM BN BD where
  kernel := (TypedSource.original p.Q p.K p.V p.QS p.KS p.Out
    p.queryBatch p.queryHead p.queryRow p.queryDim p.keyBatch p.keyHead p.keyRow p.keyDim
    p.valueBatch p.valueHead p.valueRow p.valueDim p.outputBatch p.outputHead p.outputRow p.outputDim
    p.batches p.heads p.context p.headDim BM BN BD p.activeHead p.stage p.validStage).eraseDType
  projection := RealProofs.projection BM BN BD p Bool.false
  configuration := addresses BM BN p
  out := p.Out
  stage := p.stage
```
</details>

<details><summary><code>reciprocal</code></summary>

```
/-- Candidate changes only final row division to reciprocal multiplication. -/
```
```lean
def reciprocal
    (Q K V Q_scale K_scale Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk
      _stride_kz _stride_kh stride_kn _stride_kk
      _stride_vz _stride_vh _stride_vk _stride_vn
      _stride_oz _stride_oh _stride_om _stride_on
      _Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE : Nat)
    (_hStage : STAGE = 2 ∨ STAGE = 3) : ComputeKernel :=
  let FIRST_DIAGONAL := decide (STAGE = 2)
  .mk [Q,K,V,Q_scale,K_scale] [Out] (triton {
  tl.region Q = tl.bfloat16, K = tl.bfloat16, V = tl.bfloat16,
    Q_scale = tl.float32, K_scale = tl.float32, Out = tl.bfloat16
  start_m = tl.program_id(0)
  off_hz = tl.program_id(1)

  off_z = off_hz // $(H)
  off_h = off_hz % $(H)
  qvk_offset = (off_z).to(tl.int64) * $(stride_qz) + (off_h).to(tl.int64) * $(stride_qh)
  vk_offset = qvk_offset // $(stride_qm)
  q_scale_offset = off_hz * tl.cdiv($(N_CTX), $(BLOCK_M))
  k_scale_offset = off_hz * tl.cdiv($(N_CTX), $(BLOCK_N))

  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_n = tl.arange(0, $(BLOCK_N))
  offs_k = tl.arange(0, $(BLOCK_DMODEL))
  Q_ptrs = Q + qvk_offset + offs_m[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk)
  Q_scale_ptr = Q_scale + q_scale_offset + start_m
  K_ptrs = K + qvk_offset + offs_k[:, None] + offs_n[None, :] * $(stride_kn)
  K_scale_ptr = K_scale + k_scale_offset
  V_ptrs = V + qvk_offset + offs_n[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk)
  O_block_ptr = Out + qvk_offset + offs_m[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk)
  m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) + 1.0
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)
  q = tl.load(Q_ptrs,
    mask=(offs_m[:, None] < $(N_CTX)) & (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[None, :])
  q_scale = tl.load(Q_scale_ptr)
  -- Local helper pointers: the first call does not mutate the caller's bases.
  first_lo = 0
  first_hi = start_m * $(BLOCK_M)
  K_ptrs_first = K_ptrs
  K_scale_ptr_first = K_scale_ptr
  V_ptrs_first = V_ptrs
  if FIRST_DIAGONAL {
    first_lo = start_m * $(BLOCK_M)
    first_hi = (start_m + $(1)) * $(BLOCK_M)
    first_lo = tl.multiple_of(first_lo, $(BLOCK_M))
    K_scale_ptr_first += first_lo // $(BLOCK_N)
    K_ptrs_first += $(HEAD_DIM) * first_lo
    V_ptrs_first += $(HEAD_DIM) * first_lo
  }
  for start_n in range(first_lo, first_hi, $(BLOCK_N)) {
    start_n = tl.multiple_of(start_n, $(BLOCK_N))
    k_mask = (offs_n[None, :] < ($(N_CTX) - start_n)) &
      (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[:, None]
    k = tl.load(K_ptrs_first, mask=k_mask)
    k_scale = tl.load(K_scale_ptr_first)
    qk = (tl.dot(q, k)).to(tl.float32) * q_scale * k_scale
    if FIRST_DIAGONAL {
      mask_first = offs_m[:, None] >= (start_n + offs_n[None, :])
      qk = qk + tl.where(mask_first, 0, -1.0e6)
      m_ij = tl.maximum(m_i, tl.max(qk, 1))
      qk -= m_ij[:, None]
    } else {
      m_ij = tl.maximum(m_i, tl.max(qk, 1))
      qk = qk - m_ij[:, None]
    }
    p = tl.math.exp2(qk)
    l_ij = tl.sum(p, 1)
    alpha = tl.math.exp2(m_i - m_ij)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]
    v = tl.load(V_ptrs_first,
      mask=(offs_n[:, None] < ($(N_CTX) - start_n)) &
        (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[None, :])
    p = (p).to(tl.float16)
    acc += tl.dot(p, (v).to(tl.float16), out_dtype=tl.float16)
    m_i = m_ij
    K_ptrs_first += $(BLOCK_N) * $(HEAD_DIM)
    K_scale_ptr_first += $(1)
    V_ptrs_first += $(BLOCK_N) * $(HEAD_DIM)
  }
  -- The second helper starts from the caller's original pointers and advances
  -- to the diagonal block, exactly as its STAGE=2 prefix does.
  second_lo = start_m * $(BLOCK_M)
  second_hi = (start_m + $(1)) * $(BLOCK_M)
  second_lo = tl.multiple_of(second_lo, $(BLOCK_M))
  K_scale_ptr_second = K_scale_ptr + second_lo // $(BLOCK_N)
  K_ptrs_second = K_ptrs + $(HEAD_DIM) * second_lo
  V_ptrs_second = V_ptrs + $(HEAD_DIM) * second_lo
  for start_n in range(second_lo, second_hi, $(BLOCK_N)) {
    start_n = tl.multiple_of(start_n, $(BLOCK_N))
    k_mask = (offs_n[None, :] < ($(N_CTX) - start_n)) &
      (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[:, None]
    k = tl.load(K_ptrs_second, mask=k_mask)
    k_scale = tl.load(K_scale_ptr_second)
    qk = (tl.dot(q, k)).to(tl.float32) * q_scale * k_scale
    mask = offs_m[:, None] >= (start_n + offs_n[None, :])
    qk = qk + tl.where(mask, 0, -1.0e6)
    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    qk -= m_ij[:, None]
    p = tl.math.exp2(qk)
    l_ij = tl.sum(p, 1)
    alpha = tl.math.exp2(m_i - m_ij)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]
    v = tl.load(V_ptrs_second,
      mask=(offs_n[:, None] < ($(N_CTX) - start_n)) &
        (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[None, :])
    p = (p).to(tl.float16)
    acc += tl.dot(p, (v).to(tl.float16), out_dtype=tl.float16)
    m_i = m_ij
    K_ptrs_second += $(BLOCK_N) * $(HEAD_DIM)
    K_scale_ptr_second += $(1)
    V_ptrs_second += $(BLOCK_N) * $(HEAD_DIM)
  }
  acc = acc * (1.0 / l_i[:, None])
  -- Out.type.element_ty is bf16 in the Python wrapper.
  tl.store(O_block_ptr, (acc).to(tl.bfloat16),
    mask=(offs_m[:, None] < $(N_CTX)) & (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[None, :])
}).surfaceBody
```
</details>

<details><summary><code>pairs</code></summary>

```lean
def pairs (BM BN BD startM : Nat) (p : Parameters) (xs : Inputs BM BN BD)
    (row : Fin BM) (d : Fin BD) : List (ℝ × ℝ) :=
  ((visits BM BN startM p).map (fun v => (List.finRange BN).map (fun key =>
    (score BM BN BD startM p xs v row key +
      (if v.diagonal = Bool.true ∧ startM*BM+row.val < v.start+key.val then -1000000 else 0),
     if v.start+key.val < p.context ∧ d.val < p.activeHead then xs.value v.start key d else 0)))).flatten
```
</details>

<details><summary><code>original</code></summary>

```lean
def original
    (Q K V Q_scale K_scale Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk
      _stride_kz _stride_kh stride_kn _stride_kk
      _stride_vz _stride_vh _stride_vk _stride_vn
      _stride_oz _stride_oh _stride_om _stride_on
      _Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE : Nat)
    (_hStage : STAGE = 2 ∨ STAGE = 3) : ComputeKernel :=
  let FIRST_DIAGONAL := decide (STAGE = 2)
  .mk [Q,K,V,Q_scale,K_scale] [Out] (triton {
  tl.region Q = tl.bfloat16, K = tl.bfloat16, V = tl.bfloat16,
    Q_scale = tl.float32, K_scale = tl.float32, Out = tl.bfloat16
  start_m = tl.program_id(0)
  off_hz = tl.program_id(1)

  off_z = off_hz // $(H)
  off_h = off_hz % $(H)
  qvk_offset = (off_z).to(tl.int64) * $(stride_qz) + (off_h).to(tl.int64) * $(stride_qh)
  vk_offset = qvk_offset // $(stride_qm)
  q_scale_offset = off_hz * tl.cdiv($(N_CTX), $(BLOCK_M))
  k_scale_offset = off_hz * tl.cdiv($(N_CTX), $(BLOCK_N))

  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_n = tl.arange(0, $(BLOCK_N))
  offs_k = tl.arange(0, $(BLOCK_DMODEL))
  Q_ptrs = Q + qvk_offset + offs_m[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk)
  Q_scale_ptr = Q_scale + q_scale_offset + start_m
  K_ptrs = K + qvk_offset + offs_k[:, None] + offs_n[None, :] * $(stride_kn)
  K_scale_ptr = K_scale + k_scale_offset
  V_ptrs = V + qvk_offset + offs_n[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk)
  O_block_ptr = Out + qvk_offset + offs_m[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk)
  m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) + 1.0
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)
  q = tl.load(Q_ptrs,
    mask=(offs_m[:, None] < $(N_CTX)) & (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[None, :])
  q_scale = tl.load(Q_scale_ptr)
  -- Local helper pointers: the first call does not mutate the caller's bases.
  first_lo = 0
  first_hi = start_m * $(BLOCK_M)
  K_ptrs_first = K_ptrs
  K_scale_ptr_first = K_scale_ptr
  V_ptrs_first = V_ptrs
  if FIRST_DIAGONAL {
    first_lo = start_m * $(BLOCK_M)
    first_hi = (start_m + $(1)) * $(BLOCK_M)
    first_lo = tl.multiple_of(first_lo, $(BLOCK_M))
    K_scale_ptr_first += first_lo // $(BLOCK_N)
    K_ptrs_first += $(HEAD_DIM) * first_lo
    V_ptrs_first += $(HEAD_DIM) * first_lo
  }
  for start_n in range(first_lo, first_hi, $(BLOCK_N)) {
    start_n = tl.multiple_of(start_n, $(BLOCK_N))
    k_mask = (offs_n[None, :] < ($(N_CTX) - start_n)) &
      (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[:, None]
    k = tl.load(K_ptrs_first, mask=k_mask)
    k_scale = tl.load(K_scale_ptr_first)
    qk = (tl.dot(q, k)).to(tl.float32) * q_scale * k_scale
    if FIRST_DIAGONAL {
      mask_first = offs_m[:, None] >= (start_n + offs_n[None, :])
      qk = qk + tl.where(mask_first, 0, -1.0e6)
      m_ij = tl.maximum(m_i, tl.max(qk, 1))
      qk -= m_ij[:, None]
    } else {
      m_ij = tl.maximum(m_i, tl.max(qk, 1))
      qk = qk - m_ij[:, None]
    }
    p = tl.math.exp2(qk)
    l_ij = tl.sum(p, 1)
    alpha = tl.math.exp2(m_i - m_ij)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]
    v = tl.load(V_ptrs_first,
      mask=(offs_n[:, None] < ($(N_CTX) - start_n)) &
        (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[None, :])
    p = (p).to(tl.float16)
    acc += tl.dot(p, (v).to(tl.float16), out_dtype=tl.float16)
    m_i = m_ij
    K_ptrs_first += $(BLOCK_N) * $(HEAD_DIM)
    K_scale_ptr_first += $(1)
    V_ptrs_first += $(BLOCK_N) * $(HEAD_DIM)
  }
  -- The second helper starts from the caller's original pointers and advances
  -- to the diagonal block, exactly as its STAGE=2 prefix does.
  second_lo = start_m * $(BLOCK_M)
  second_hi = (start_m + $(1)) * $(BLOCK_M)
  second_lo = tl.multiple_of(second_lo, $(BLOCK_M))
  K_scale_ptr_second = K_scale_ptr + second_lo // $(BLOCK_N)
  K_ptrs_second = K_ptrs + $(HEAD_DIM) * second_lo
  V_ptrs_second = V_ptrs + $(HEAD_DIM) * second_lo
  for start_n in range(second_lo, second_hi, $(BLOCK_N)) {
    start_n = tl.multiple_of(start_n, $(BLOCK_N))
    k_mask = (offs_n[None, :] < ($(N_CTX) - start_n)) &
      (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[:, None]
    k = tl.load(K_ptrs_second, mask=k_mask)
    k_scale = tl.load(K_scale_ptr_second)
    qk = (tl.dot(q, k)).to(tl.float32) * q_scale * k_scale
    mask = offs_m[:, None] >= (start_n + offs_n[None, :])
    qk = qk + tl.where(mask, 0, -1.0e6)
    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    qk -= m_ij[:, None]
    p = tl.math.exp2(qk)
    l_ij = tl.sum(p, 1)
    alpha = tl.math.exp2(m_i - m_ij)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]
    v = tl.load(V_ptrs_second,
      mask=(offs_n[:, None] < ($(N_CTX) - start_n)) &
        (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[None, :])
    p = (p).to(tl.float16)
    acc += tl.dot(p, (v).to(tl.float16), out_dtype=tl.float16)
    m_i = m_ij
    K_ptrs_second += $(BLOCK_N) * $(HEAD_DIM)
    K_scale_ptr_second += $(1)
    V_ptrs_second += $(BLOCK_N) * $(HEAD_DIM)
  }
  acc = acc / l_i[:, None]
  -- Out.type.element_ty is bf16 in the Python wrapper.
  tl.store(O_block_ptr, (acc).to(tl.bfloat16),
    mask=(offs_m[:, None] < $(N_CTX)) & (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[None, :])
}).surfaceBody
```
</details>

<details><summary><code>addresses</code></summary>

```
/-- The contract uses these exact source addresses for tile origin `start`:
Q/Out = base+(startM*BM+row)*queryRow+d*queryDim;
K = base+d+j*keyRow+headDim*start;
V = base+j*queryRow+d*queryDim+headDim*start.
Q/Out require row<N and d<activeHead; K/V require start+j<N and d<activeHead.
QS[queryScaleBase+startM] and KS[keyScaleBase+start/BN] are unmasked reads,
including key scales for fully padded tiles. Only actual visits are pinned.
There is no scratch allocation. -/
```
```lean
def addresses (BM BN : Nat) (p : Parameters) (startM headBatch : Nat) : AttnFwdCausal.RealLoads.Config where
  Q := p.Q
  K := p.K
  V := p.V
  QS := p.QS
  KS := p.KS
  base := headBatch/p.heads*p.queryBatch+headBatch%p.heads*p.queryHead
  queryScaleBase := headBatch*((p.context+BM-1)/BM)
  keyScaleBase := headBatch*((p.context+BN-1)/BN)
  strideM := p.queryRow
  strideD := p.queryDim
  strideKN := p.keyRow
  headDim := p.headDim
  context := p.context
  activeHead := p.activeHead
  startM := startM
```
</details>

<details><summary><code>visits</code></summary>

```
/-- Retain repeated visits and partial final tiles exactly as in the source. -/
```
```lean
def visits (BM BN startM : Nat) (p : Parameters) : List Visit :=
  (if p.stage = 2 then rangeVisits (startM*BM) ((startM+1)*BM) BN Bool.true
    else rangeVisits 0 (startM*BM) BN Bool.false) ++
    rangeVisits (startM*BM) ((startM+1)*BM) BN Bool.true
```
</details>

<details><summary><code>score</code></summary>

```lean
def score (BM BN BD startM : Nat) (p : Parameters) (xs : Inputs BM BN BD)
    (v : Visit) (row : Fin BM) (key : Fin BN) : ℝ :=
  (∑ d : Fin BD,
    (if startM*BM+row.val < p.context ∧ d.val < p.activeHead then xs.query row d else 0) *
    (if v.start+key.val < p.context ∧ d.val < p.activeHead then xs.key v.start key d else 0)) *
      xs.queryScale * xs.keyScale (v.start/BN)
```
</details>

<details><summary><code>rangeVisits</code></summary>

```lean
def rangeVisits (lower upper BN : Nat) (diagonal : Bool) : List Visit :=
  (List.range ((upper-lower+BN-1)/BN)).map (fun j => ⟨lower+j*BN,diagonal⟩)
```
</details>
