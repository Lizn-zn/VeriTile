# Spec sheet — `bench/tritonbench_g/attention_score/Correct.lean`

**Python source:** `bench/tritonbench_g/attention_score/attention_score.py`

## Public theorem: `score_correctness`

**Statement:**
```lean
specification score_correctness (B D : Nat) (p : Parameters) (hB : 0 < B) :
    Spec.Real (originalIO B D p ⊨ fun startN _ xs => expected B D startN p xs)
```

**Assumptions / layout contracts:**
- `hB : 0 < B`

**Closed-form spec defs (transitive):** `Parameters`, `originalIO`, `expected`, `original`, `blocks`, `weight`, `windowAllows`, `score`

<details><summary><code>Parameters</code></summary>

```
/-- Q/K/M/Out regions; independent batch, head, row and inner strides; head
counts and context sizes; scale; signed offset/size; window and even flags.
The source's unused output inner stride and batch count remain parameters. -/
```
```lean
abbrev Parameters := RealSource.Parameters
```
</details>

<details><summary><code>originalIO</code></summary>

```
/-- Checked Q/K loads require only their active addresses; unchecked loads
require whole windows. M is always read for every visited row, including
padding. K is read even if ROUND_CTX=0. Only active output keys are written.
The shared flat IO contract checks physical allocation and disjoint regions. -/
```
```lean
def originalIO (B D : Nat) (p : Parameters) : RealIO.IO (rows B p) B D where
  kernel := (TypedSource.original p.Q p.K p.M p.Out
    p.queryBatch p.queryHead p.queryRow p.queryDim p.keyBatch p.keyHead p.keyRow p.keyDim
    p.outputBatch p.outputHead p.outputInner p.batches p.heads p.kvHeads p.queryContext p.roundedContext
    p.keyContext B D B p.offset p.size p.scale p.window p.complement p.evenM p.evenN rfl).eraseDType
  projection := RealSource.projection B D p Bool.false
  Q := p.Q
  K := p.K
  M := p.M
  Out := p.Out
  queryWindow := fun _ headBatch _ i =>
    headBatch / p.heads * p.queryBatch + headBatch % p.heads * p.queryHead +
      i.1.val * p.queryRow + i.2.1.val * p.queryDim
  keyWindow := fun startN headBatch _ i =>
    headBatch / p.heads * p.keyBatch + (headBatch % p.heads) / (p.heads / p.kvHeads) * p.keyHead +
      i.1.val * p.keyDim + (startN * B + i.2.1.val) * p.keyRow
  normalizerWindow := fun _ headBatch _ i => headBatch * p.roundedContext + i.1.val
  outputWindow := fun startN headBatch _ i =>
    headBatch / p.heads * p.outputBatch + headBatch % p.heads * p.outputHead + startN * B + i.val
  queryMask := fun _ _ _ i => p.evenM = Bool.true ∨ i.1.val < p.queryContext
  keyMask := fun startN _ _ i => p.evenN = Bool.true ∨ startN * B + i.2.1.val < p.keyContext
  outputMask := fun startN _ _ i => startN * B + i.val < p.keyContext
```
</details>

<details><summary><code>expected</code></summary>

```lean
def expected (B D startN : Nat) (p : Parameters) (xs : Inputs (rows B p) B D) (key : Fin B) : ℝ :=
  ∑ block : Fin (blocks B p), ∑ row : Fin B,
    weight B D startN p xs (RealInputs.rowIndex block row) key
```
</details>

<details><summary><code>original</code></summary>

```lean
def original
    (Q K M Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_oz stride_oh _stride_on
      _Z H H_KV N_CTX ROUND_CTX NKV_CTX
      BLOCK_M BLOCK_DMODEL BLOCK_N : Nat)
    (sliding_window_offset sliding_window_size : Int) (sm_scale : ℝ)
    (SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW IS_EVEN_M IS_EVEN_N : Bool)
    (_hBlockMN : BLOCK_M = BLOCK_N) :
    ComputeKernel := .mk [Q, K, M] [Out] (triton {
  tl.region Q = tl.float16, K = tl.float16, M = tl.float32, Out = tl.float16
  start_n = tl.program_id(0)
  off_hz = tl.program_id(1)
  off_z = off_hz // $(H)
  off_h = off_hz % $(H)
  off_hkv = off_h // ($(H) // $(H_KV))
  q_offset = (off_z).to(tl.int64) * $(stride_qz) + (off_h).to(tl.int64) * $(stride_qh)
  k_offset = (off_z).to(tl.int64) * $(stride_kz) + (off_hkv).to(tl.int64) * $(stride_kh)
  m_ptrs = M + off_hz * $(ROUND_CTX) + tl.arange(0, $(BLOCK_M))
  o = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  Q_block_ptr = tl.make_block_ptr(base=Q + q_offset,
    shape=($(N_CTX), $(BLOCK_DMODEL)),
    strides=($(stride_qm), $(stride_qk)),
    offsets=(0, 0),
    block_shape=($(BLOCK_M), $(BLOCK_DMODEL)),
    order=(1, 0))
  K_block_ptr = tl.make_block_ptr(base=K + k_offset,
    shape=($(BLOCK_DMODEL), $(NKV_CTX)),
    strides=($(stride_kk), $(stride_kn)),
    offsets=(0, start_n * $(BLOCK_N)),
    block_shape=($(BLOCK_DMODEL), $(BLOCK_N)),
    order=(0, 1))
  if IS_EVEN_N {
    k = tl.load(K_block_ptr)
  } else {
    k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
  }
  lo = 0
  hi = $(ROUND_CTX)
  qk_scale = $((sm_scale : ℝ))
  qk_scale *= 1.4426950408889634
  for start_m in range(lo, hi, $(BLOCK_M)) {
    start_m = tl.multiple_of(start_m, $(BLOCK_M))
    if IS_EVEN_M {
      q = tl.load(Q_block_ptr)
    } else {
      q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    }
    m = tl.load(m_ptrs)
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    qk += tl.dot(q, k)
    qk = qk * qk_scale
    if SLIDING_WINDOW {
      dist = tl.arange(0, $(BLOCK_M))[:, None] -
        tl.arange(0, $(BLOCK_N))[None, :] + start_m -
        start_n * $(BLOCK_N) + $((sliding_window_offset : Int))
      if COMPLEMENT_SLIDING_WINDOW {
        mask = dist >= $((sliding_window_size : Int))
      } else {
        mask = (dist >= $(0)) & (dist < $((sliding_window_size : Int)))
      }
    }
    qk = qk - m[:, None]
    p = tl.math.exp2(qk)
    if SLIDING_WINDOW {
      p = tl.where(mask, p, 0)
    }
    if not IS_EVEN_N {
      p = tl.where(((tl.arange(0, $(BLOCK_M)) + start_m) < $(N_CTX))[:, None],
        p, 0)
    }
    o += tl.sum(p, axis=0)
    Q_block_ptr = tl.advance(Q_block_ptr, offsets=($(BLOCK_M), 0))
    m_ptrs = m_ptrs + $(BLOCK_M)
  }
  o_offset = (off_z).to(tl.int64) * $(stride_oz) + (off_h).to(tl.int64) * $(stride_oh)
  o_range = tl.arange(0, $(BLOCK_N)) + start_n * $(BLOCK_N)
  o_ptrs = Out + o_offset + o_range
  tl.store(o_ptrs, (o).to(tl.float16),
    mask=o_range < $(NKV_CTX))
}).surfaceBody
```
</details>

<details><summary><code>blocks</code></summary>

```lean
def blocks (B : Nat) (p : Parameters) : Nat := (p.roundedContext + B - 1) / B
```
</details>

<details><summary><code>weight</code></summary>

```
/-- The source selects query rows using evenN, independently of the Q-load
flag evenM. Keep this distinction, including padded rows when evenN=true. -/
```
```lean
def weight (B D startN : Nat) (p : Parameters) (xs : Inputs (rows B p) B D)
    (row : Fin (rows B p)) (key : Fin B) : ℝ :=
  if windowAllows p row.val (startN*B+key.val) ∧
      (p.evenN = Bool.true ∨ row.val < p.queryContext) then
    pow2 (score B D startN p xs row key - xs.normalizer row)
  else 0
```
</details>

<details><summary><code>windowAllows</code></summary>

```
/-- The regular window keeps 0 ≤ query-key+offset < size. The complement
arm keeps distance ≥ size. With windows disabled all distances are allowed. -/
```
```lean
def windowAllows (p : Parameters) (row key : Nat) : Prop :=
  p.window = Bool.false ∨ if p.complement then p.size ≤ (row : Int) - key + p.offset
    else 0 ≤ (row : Int) - key + p.offset ∧ (row : Int) - key + p.offset < p.size

instance (p : Parameters) (row key : Nat) : Decidable (windowAllows p row key) := by
  unfold windowAllows
  infer_instance
```
</details>

<details><summary><code>score</code></summary>

```
/-- The exact decimal is the source's conversion of scale to base 2.
Boundary-checked inactive Q/K lanes contribute zero to the dot product. -/
```
```lean
def score (B D startN : Nat) (p : Parameters) (xs : Inputs (rows B p) B D)
    (row : Fin (rows B p)) (key : Fin B) : ℝ :=
  (∑ d : Fin D,
    (if p.evenM = Bool.true ∨ row.val < p.queryContext then xs.query row d else 0) *
    (if p.evenN = Bool.true ∨ startN*B+key.val < p.keyContext then xs.key key d else 0)) *
      (p.scale * 1.4426950408889634)
```
</details>

## Public theorem: `score_scalar_left_correctness`

**Statement:**
```lean
specification score_scalar_left_correctness (B D : Nat) (p : Parameters) (hB : 0 < B) :
    Spec.Real (scalarLeftIO B D p ⊨ fun startN _ xs => expected B D startN p xs)
```

**Assumptions / layout contracts:**
- `hB : 0 < B`

**Closed-form spec defs (transitive):** `Parameters`, `scalarLeftIO`, `expected`, `originalIO`, `scalarLeft`, `blocks`, `weight`, `original`, `windowAllows`, `score`

<details><summary><code>Parameters</code></summary>

```
/-- Q/K/M/Out regions; independent batch, head, row and inner strides; head
counts and context sizes; scale; signed offset/size; window and even flags.
The source's unused output inner stride and batch count remain parameters. -/
```
```lean
abbrev Parameters := RealSource.Parameters
```
</details>

<details><summary><code>scalarLeftIO</code></summary>

```lean
def scalarLeftIO (B D : Nat) (p : Parameters) : RealIO.IO (rows B p) B D :=
  { originalIO B D p with
    kernel := (TypedSource.scalarLeft p.Q p.K p.M p.Out
      p.queryBatch p.queryHead p.queryRow p.queryDim p.keyBatch p.keyHead p.keyRow p.keyDim
      p.outputBatch p.outputHead p.outputInner p.batches p.heads p.kvHeads p.queryContext p.roundedContext
      p.keyContext B D B p.offset p.size p.scale p.window p.complement p.evenM p.evenN rfl).eraseDType
    projection := RealSource.projection B D p Bool.true }
```
</details>

<details><summary><code>expected</code></summary>

```lean
def expected (B D startN : Nat) (p : Parameters) (xs : Inputs (rows B p) B D) (key : Fin B) : ℝ :=
  ∑ block : Fin (blocks B p), ∑ row : Fin B,
    weight B D startN p xs (RealInputs.rowIndex block row) key
```
</details>

<details><summary><code>originalIO</code></summary>

```
/-- Checked Q/K loads require only their active addresses; unchecked loads
require whole windows. M is always read for every visited row, including
padding. K is read even if ROUND_CTX=0. Only active output keys are written.
The shared flat IO contract checks physical allocation and disjoint regions. -/
```
```lean
def originalIO (B D : Nat) (p : Parameters) : RealIO.IO (rows B p) B D where
  kernel := (TypedSource.original p.Q p.K p.M p.Out
    p.queryBatch p.queryHead p.queryRow p.queryDim p.keyBatch p.keyHead p.keyRow p.keyDim
    p.outputBatch p.outputHead p.outputInner p.batches p.heads p.kvHeads p.queryContext p.roundedContext
    p.keyContext B D B p.offset p.size p.scale p.window p.complement p.evenM p.evenN rfl).eraseDType
  projection := RealSource.projection B D p Bool.false
  Q := p.Q
  K := p.K
  M := p.M
  Out := p.Out
  queryWindow := fun _ headBatch _ i =>
    headBatch / p.heads * p.queryBatch + headBatch % p.heads * p.queryHead +
      i.1.val * p.queryRow + i.2.1.val * p.queryDim
  keyWindow := fun startN headBatch _ i =>
    headBatch / p.heads * p.keyBatch + (headBatch % p.heads) / (p.heads / p.kvHeads) * p.keyHead +
      i.1.val * p.keyDim + (startN * B + i.2.1.val) * p.keyRow
  normalizerWindow := fun _ headBatch _ i => headBatch * p.roundedContext + i.1.val
  outputWindow := fun startN headBatch _ i =>
    headBatch / p.heads * p.outputBatch + headBatch % p.heads * p.outputHead + startN * B + i.val
  queryMask := fun _ _ _ i => p.evenM = Bool.true ∨ i.1.val < p.queryContext
  keyMask := fun startN _ _ i => p.evenN = Bool.true ∨ startN * B + i.2.1.val < p.keyContext
  outputMask := fun startN _ _ i => startN * B + i.val < p.keyContext
```
</details>

<details><summary><code>scalarLeft</code></summary>

```
/-- Scalar-left score scaling in the same complete source context. -/
```
```lean
def scalarLeft
    (Q K M Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_oz stride_oh _stride_on
      _Z H H_KV N_CTX ROUND_CTX NKV_CTX
      BLOCK_M BLOCK_DMODEL BLOCK_N : Nat)
    (sliding_window_offset sliding_window_size : Int) (sm_scale : ℝ)
    (SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW IS_EVEN_M IS_EVEN_N : Bool)
    (_hBlockMN : BLOCK_M = BLOCK_N) :
    ComputeKernel := .mk [Q, K, M] [Out] (triton {
  tl.region Q = tl.float16, K = tl.float16, M = tl.float32, Out = tl.float16
  start_n = tl.program_id(0)
  off_hz = tl.program_id(1)
  off_z = off_hz // $(H)
  off_h = off_hz % $(H)
  off_hkv = off_h // ($(H) // $(H_KV))
  q_offset = (off_z).to(tl.int64) * $(stride_qz) + (off_h).to(tl.int64) * $(stride_qh)
  k_offset = (off_z).to(tl.int64) * $(stride_kz) + (off_hkv).to(tl.int64) * $(stride_kh)
  m_ptrs = M + off_hz * $(ROUND_CTX) + tl.arange(0, $(BLOCK_M))
  o = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  Q_block_ptr = tl.make_block_ptr(base=Q + q_offset,
    shape=($(N_CTX), $(BLOCK_DMODEL)),
    strides=($(stride_qm), $(stride_qk)),
    offsets=(0, 0),
    block_shape=($(BLOCK_M), $(BLOCK_DMODEL)),
    order=(1, 0))
  K_block_ptr = tl.make_block_ptr(base=K + k_offset,
    shape=($(BLOCK_DMODEL), $(NKV_CTX)),
    strides=($(stride_kk), $(stride_kn)),
    offsets=(0, start_n * $(BLOCK_N)),
    block_shape=($(BLOCK_DMODEL), $(BLOCK_N)),
    order=(0, 1))
  if IS_EVEN_N {
    k = tl.load(K_block_ptr)
  } else {
    k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
  }
  lo = 0
  hi = $(ROUND_CTX)
  qk_scale = $((sm_scale : ℝ))
  qk_scale *= 1.4426950408889634
  for start_m in range(lo, hi, $(BLOCK_M)) {
    start_m = tl.multiple_of(start_m, $(BLOCK_M))
    if IS_EVEN_M {
      q = tl.load(Q_block_ptr)
    } else {
      q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    }
    m = tl.load(m_ptrs)
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    qk += tl.dot(q, k)
    qk = qk_scale * qk
    if SLIDING_WINDOW {
      dist = tl.arange(0, $(BLOCK_M))[:, None] -
        tl.arange(0, $(BLOCK_N))[None, :] + start_m -
        start_n * $(BLOCK_N) + $((sliding_window_offset : Int))
      if COMPLEMENT_SLIDING_WINDOW {
        mask = dist >= $((sliding_window_size : Int))
      } else {
        mask = (dist >= $(0)) & (dist < $((sliding_window_size : Int)))
      }
    }
    qk = qk - m[:, None]
    p = tl.math.exp2(qk)
    if SLIDING_WINDOW {
      p = tl.where(mask, p, 0)
    }
    if not IS_EVEN_N {
      p = tl.where(((tl.arange(0, $(BLOCK_M)) + start_m) < $(N_CTX))[:, None],
        p, 0)
    }
    o += tl.sum(p, axis=0)
    Q_block_ptr = tl.advance(Q_block_ptr, offsets=($(BLOCK_M), 0))
    m_ptrs = m_ptrs + $(BLOCK_M)
  }
  o_offset = (off_z).to(tl.int64) * $(stride_oz) + (off_h).to(tl.int64) * $(stride_oh)
  o_range = tl.arange(0, $(BLOCK_N)) + start_n * $(BLOCK_N)
  o_ptrs = Out + o_offset + o_range
  tl.store(o_ptrs, (o).to(tl.float16),
    mask=o_range < $(NKV_CTX))
}).surfaceBody
```
</details>

<details><summary><code>blocks</code></summary>

```lean
def blocks (B : Nat) (p : Parameters) : Nat := (p.roundedContext + B - 1) / B
```
</details>

<details><summary><code>weight</code></summary>

```
/-- The source selects query rows using evenN, independently of the Q-load
flag evenM. Keep this distinction, including padded rows when evenN=true. -/
```
```lean
def weight (B D startN : Nat) (p : Parameters) (xs : Inputs (rows B p) B D)
    (row : Fin (rows B p)) (key : Fin B) : ℝ :=
  if windowAllows p row.val (startN*B+key.val) ∧
      (p.evenN = Bool.true ∨ row.val < p.queryContext) then
    pow2 (score B D startN p xs row key - xs.normalizer row)
  else 0
```
</details>

<details><summary><code>original</code></summary>

```lean
def original
    (Q K M Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_oz stride_oh _stride_on
      _Z H H_KV N_CTX ROUND_CTX NKV_CTX
      BLOCK_M BLOCK_DMODEL BLOCK_N : Nat)
    (sliding_window_offset sliding_window_size : Int) (sm_scale : ℝ)
    (SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW IS_EVEN_M IS_EVEN_N : Bool)
    (_hBlockMN : BLOCK_M = BLOCK_N) :
    ComputeKernel := .mk [Q, K, M] [Out] (triton {
  tl.region Q = tl.float16, K = tl.float16, M = tl.float32, Out = tl.float16
  start_n = tl.program_id(0)
  off_hz = tl.program_id(1)
  off_z = off_hz // $(H)
  off_h = off_hz % $(H)
  off_hkv = off_h // ($(H) // $(H_KV))
  q_offset = (off_z).to(tl.int64) * $(stride_qz) + (off_h).to(tl.int64) * $(stride_qh)
  k_offset = (off_z).to(tl.int64) * $(stride_kz) + (off_hkv).to(tl.int64) * $(stride_kh)
  m_ptrs = M + off_hz * $(ROUND_CTX) + tl.arange(0, $(BLOCK_M))
  o = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  Q_block_ptr = tl.make_block_ptr(base=Q + q_offset,
    shape=($(N_CTX), $(BLOCK_DMODEL)),
    strides=($(stride_qm), $(stride_qk)),
    offsets=(0, 0),
    block_shape=($(BLOCK_M), $(BLOCK_DMODEL)),
    order=(1, 0))
  K_block_ptr = tl.make_block_ptr(base=K + k_offset,
    shape=($(BLOCK_DMODEL), $(NKV_CTX)),
    strides=($(stride_kk), $(stride_kn)),
    offsets=(0, start_n * $(BLOCK_N)),
    block_shape=($(BLOCK_DMODEL), $(BLOCK_N)),
    order=(0, 1))
  if IS_EVEN_N {
    k = tl.load(K_block_ptr)
  } else {
    k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
  }
  lo = 0
  hi = $(ROUND_CTX)
  qk_scale = $((sm_scale : ℝ))
  qk_scale *= 1.4426950408889634
  for start_m in range(lo, hi, $(BLOCK_M)) {
    start_m = tl.multiple_of(start_m, $(BLOCK_M))
    if IS_EVEN_M {
      q = tl.load(Q_block_ptr)
    } else {
      q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    }
    m = tl.load(m_ptrs)
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    qk += tl.dot(q, k)
    qk = qk * qk_scale
    if SLIDING_WINDOW {
      dist = tl.arange(0, $(BLOCK_M))[:, None] -
        tl.arange(0, $(BLOCK_N))[None, :] + start_m -
        start_n * $(BLOCK_N) + $((sliding_window_offset : Int))
      if COMPLEMENT_SLIDING_WINDOW {
        mask = dist >= $((sliding_window_size : Int))
      } else {
        mask = (dist >= $(0)) & (dist < $((sliding_window_size : Int)))
      }
    }
    qk = qk - m[:, None]
    p = tl.math.exp2(qk)
    if SLIDING_WINDOW {
      p = tl.where(mask, p, 0)
    }
    if not IS_EVEN_N {
      p = tl.where(((tl.arange(0, $(BLOCK_M)) + start_m) < $(N_CTX))[:, None],
        p, 0)
    }
    o += tl.sum(p, axis=0)
    Q_block_ptr = tl.advance(Q_block_ptr, offsets=($(BLOCK_M), 0))
    m_ptrs = m_ptrs + $(BLOCK_M)
  }
  o_offset = (off_z).to(tl.int64) * $(stride_oz) + (off_h).to(tl.int64) * $(stride_oh)
  o_range = tl.arange(0, $(BLOCK_N)) + start_n * $(BLOCK_N)
  o_ptrs = Out + o_offset + o_range
  tl.store(o_ptrs, (o).to(tl.float16),
    mask=o_range < $(NKV_CTX))
}).surfaceBody
```
</details>

<details><summary><code>windowAllows</code></summary>

```
/-- The regular window keeps 0 ≤ query-key+offset < size. The complement
arm keeps distance ≥ size. With windows disabled all distances are allowed. -/
```
```lean
def windowAllows (p : Parameters) (row key : Nat) : Prop :=
  p.window = Bool.false ∨ if p.complement then p.size ≤ (row : Int) - key + p.offset
    else 0 ≤ (row : Int) - key + p.offset ∧ (row : Int) - key + p.offset < p.size

instance (p : Parameters) (row key : Nat) : Decidable (windowAllows p row key) := by
  unfold windowAllows
  infer_instance
```
</details>

<details><summary><code>score</code></summary>

```
/-- The exact decimal is the source's conversion of scale to base 2.
Boundary-checked inactive Q/K lanes contribute zero to the dot product. -/
```
```lean
def score (B D startN : Nat) (p : Parameters) (xs : Inputs (rows B p) B D)
    (row : Fin (rows B p)) (key : Fin B) : ℝ :=
  (∑ d : Fin D,
    (if p.evenM = Bool.true ∨ row.val < p.queryContext then xs.query row d else 0) *
    (if p.evenN = Bool.true ∨ startN*B+key.val < p.keyContext then xs.key key d else 0)) *
      (p.scale * 1.4426950408889634)
```
</details>
