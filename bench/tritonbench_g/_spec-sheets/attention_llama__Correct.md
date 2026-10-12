# Spec sheet — `bench/tritonbench_g/attention_llama/Correct.lean`

**Python source:** `bench/tritonbench_g/attention_llama/attention_llama.py`

## Public theorem: `attention_correctness`

<details><summary>docstring</summary>

```
/-- Each active output equals the formula for the initial input arrays; all
other cells remain unchanged. Injectivity prevents overlapping output lanes. -/
```
</details>

**Statement:**
```lean
specification attention_correctness (hBM : 0 < BM) (hBN : 0 < BN) (hT : 0 < T)
    (hinj : Function.Injective (fun i : TileIndex [BM, BD] =>
      i.1.val * som + i.2.1.val * son)) :
    Spec.Real (io Q K V Out scale
      sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD T candidate causal ⊨
      fun p _ _ xs ys zs => expected T BM BN BD H SP hBN hT causal scale p xs ys zs)
```

**Assumptions / layout contracts:**
- `hBM : 0 < BM`
- `hBN : 0 < BN`
- `hT : 0 < T`
- `hinj : Function.Injective (fun i : TileIndex [BM, BD] =>
      i.1.val * som + i.2.1.val * son)`

**Closed-form spec defs (transitive):** `io`, `expected`, `carryFirst`, `original`, `decode`, `score`, `stream`, `query`

<details><summary><code>io</code></summary>

```
/-- Q is static across steps; K/V advance by BN rows. All four buffers keep
their independent batch/head/row/column strides. Query and output masks use H;
the whole-block launch precondition makes all streamed K/V lanes active. -/
```
```lean
def io (Q K V Out : RegionName) (scale : ℝ)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD T : Nat) (candidate causal : Bool) : StreamMasked3DKernelIO₃ where
  kernel := (if candidate then
    TypedSource.carryFirst Q K V Out scale
      sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD causal
  else
    TypedSource.original Q K V Out scale
      sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD causal).eraseDType
  projection := by
    cases candidate
    · exact RealSource.original_projection_supported ..
    · exact RealSource.candidate_projection_supported ..
  inp1 := Q
  inp2 := K
  inp3 := V
  out := Out
  T := T
  B1 := BM * BD
  B2 := BN * BD
  B3 := BN * BD
  C := BM * BD
  outDType := .real
  pre := fun p _ _ => if causal then (p + 1) * BN + SP = BN * T ∧ BN * T ≤ N ∧ SP ≤ p * BM
    else N = BN * T
  read1 := fun p h _ _ j => h / NH * sqz + h % NH * sqh +
    (p * BM + j.val / BD) * sqm + j.val % BD * sqk
  read2 := fun _ h _ t j => h / NH * skz + h % NH * skh +
    (t.val * BN + j.val / BD) * skn + j.val % BD * skk
  read3 := fun _ h _ t j => h / NH * svz + h % NH * svh +
    (t.val * BN + j.val / BD) * svk + j.val % BD * svn
  write := fun p h _ j => h / NH * soz + h % NH * soh +
    (p * BM + j.val / BD) * som + j.val % BD * son
  mask1 := fun p _ _ _ j => p * BM + j.val / BD < H
  mask2 := fun _ _ _ _ _ => True
  mask3 := fun _ _ _ _ _ => True
  writeMask := fun p _ _ j => p * BM + j.val / BD < H
```
</details>

<details><summary><code>expected</code></summary>

```
/-- Independent Σ exp(score)*V / Σ exp(score). Under the causal launch
precondition, k ≤ p*BM-SP+r is exactly k+SP ≤ p*BM+r. This source shrinks
visibility by SP; it does not add a prefix cache to the visible keys. -/
```
```lean
noncomputable def expected (T BM BN BD H SP : Nat) (hBN : 0 < BN) (hT : 0 < T)
    (causal : Bool) (scale : ℝ) (p : Nat)
    (xs : Fin T → Fin (BM * BD) → ℝ)
    (ys zs : Fin T → Fin (BN * BD) → ℝ) (j : Fin (BM * BD)) : ℝ :=
  let i := Lane2D.decode j
  let s := score T BM BN BD H hBN hT scale p xs ys i.1
  if causal then
    let weight := fun k : Fin (BN * T) =>
      if k.val ≤ p * BM - SP + i.1.val then Real.exp (s k) else 0
    (Finset.univ.sum fun k => weight k * stream T BN BD hBN zs (k, i.2.1, PUnit.unit)) /
      (Finset.univ.sum fun k => weight k)
  else
    (Finset.univ.sum fun k => Real.exp (s k) * stream T BN BD hBN zs (k, i.2.1, PUnit.unit)) /
      (Finset.univ.sum fun k => Real.exp (s k))
```
</details>

<details><summary><code>carryFirst</code></summary>

```lean
def carryFirst
    (Q K V Out : RegionName) (sm_scale : ℝ)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      N_HEAD H N_CTX start_position
      BLOCK_M BLOCK_N BLOCK_DMODEL : Nat) (IS_CAUSAL : Bool) :
    ComputeKernel := .mk [Q, K, V] [Out] (triton {
  tl.region Q = tl.float16, K = tl.float16, V = tl.float16, Out = tl.float16
  start_m = tl.program_id(0)

  head_idx = tl.program_id(1)
  batch_id = head_idx // $(N_HEAD)
  off_hz = head_idx % $(N_HEAD)

  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_n = tl.arange(0, $(BLOCK_N))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  off_q = batch_id * $(stride_qz) + off_hz * $(stride_qh) + offs_m[:, None] * $(stride_qm) + offs_d[None, :] * $(stride_qk)
  off_k = batch_id * $(stride_kz) + off_hz * $(stride_kh) + offs_n[:, None] * $(stride_kn) + offs_d[None, :] * $(stride_kk)
  off_v = batch_id * $(stride_vz) + off_hz * $(stride_vh) + offs_n[:, None] * $(stride_vk) + offs_d[None, :] * $(stride_vn)
  q_ptrs = Q + off_q
  k_ptrs = K + off_k
  v_ptrs = V + off_v
  m_prev = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_prev = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)
  q = tl.load(q_ptrs, offs_m[:, None] < $(H), other=0.0)
  block_n_end = $(N_CTX)
  if IS_CAUSAL {
    block_n_end = (start_m + $(1)) * $(BLOCK_N) + $(start_position)
  }
  for start_n in range($(0), block_n_end, $(BLOCK_N)) {
    block_n_offs = start_n + offs_n
    k = tl.load(k_ptrs, block_n_offs[:, None] < $(N_CTX), other=0.0)
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    qk += tl.dot(q, tl.trans(k))
    qk = tl.where(offs_n[None, :] < $(N_CTX), qk, float("-inf"))
    qk *= $((sm_scale : ℝ))
    if IS_CAUSAL {
      qk = tl.where(offs_m[:, None] >= (block_n_offs[None, :] + $(start_position)), qk, float("-inf"))
    }
    m_curr = tl.maximum(tl.max(qk, 1), m_prev)
    l_prev *= tl.exp(m_prev - m_curr)
    p = tl.exp(qk - m_curr[:, None])
    l_curr = l_prev + tl.sum(p, 1)
    l_rcp = 1.0 / l_curr
    p *= l_rcp[:, None]
    acc *= (l_prev * l_rcp)[:, None]
    p = (p).to(tl.float16)
    v = tl.load(v_ptrs, block_n_offs[:, None] < $(N_CTX), other=0.0)
    acc += tl.dot(p, v)
    l_prev = l_curr
    m_prev = m_curr
    k_ptrs += $(BLOCK_N) * $(stride_kn)
    v_ptrs += $(BLOCK_N) * $(stride_vk)
  }
  start_m = tl.program_id(0)
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))

  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  off_o = batch_id * $(stride_oz) + off_hz * $(stride_oh) + offs_m[:, None] * $(stride_om) + offs_d[None, :] * $(stride_on)
  out_ptrs = Out + off_o
  tl.store(out_ptrs, (acc).to(tl.float16), offs_m[:, None] < $(H))
}).surfaceBody
```
</details>

<details><summary><code>original</code></summary>

```lean
def original
    (Q K V Out : RegionName) (sm_scale : ℝ)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      N_HEAD H N_CTX start_position
      BLOCK_M BLOCK_N BLOCK_DMODEL : Nat) (IS_CAUSAL : Bool) :
    ComputeKernel := .mk [Q, K, V] [Out] (triton {
  tl.region Q = tl.float16, K = tl.float16, V = tl.float16, Out = tl.float16
  start_m = tl.program_id(0)

  head_idx = tl.program_id(1)
  batch_id = head_idx // $(N_HEAD)
  off_hz = head_idx % $(N_HEAD)

  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_n = tl.arange(0, $(BLOCK_N))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  off_q = batch_id * $(stride_qz) + off_hz * $(stride_qh) + offs_m[:, None] * $(stride_qm) + offs_d[None, :] * $(stride_qk)
  off_k = batch_id * $(stride_kz) + off_hz * $(stride_kh) + offs_n[:, None] * $(stride_kn) + offs_d[None, :] * $(stride_kk)
  off_v = batch_id * $(stride_vz) + off_hz * $(stride_vh) + offs_n[:, None] * $(stride_vk) + offs_d[None, :] * $(stride_vn)
  q_ptrs = Q + off_q
  k_ptrs = K + off_k
  v_ptrs = V + off_v
  m_prev = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_prev = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)
  q = tl.load(q_ptrs, offs_m[:, None] < $(H), other=0.0)
  block_n_end = $(N_CTX)
  if IS_CAUSAL {
    block_n_end = (start_m + $(1)) * $(BLOCK_N) + $(start_position)
  }
  for start_n in range($(0), block_n_end, $(BLOCK_N)) {
    block_n_offs = start_n + offs_n
    k = tl.load(k_ptrs, block_n_offs[:, None] < $(N_CTX), other=0.0)
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    qk += tl.dot(q, tl.trans(k))
    qk = tl.where(offs_n[None, :] < $(N_CTX), qk, float("-inf"))
    qk *= $((sm_scale : ℝ))
    if IS_CAUSAL {
      qk = tl.where(offs_m[:, None] >= (block_n_offs[None, :] + $(start_position)), qk, float("-inf"))
    }
    m_curr = tl.maximum(tl.max(qk, 1), m_prev)
    l_prev *= tl.exp(m_prev - m_curr)
    p = tl.exp(qk - m_curr[:, None])
    l_curr = tl.sum(p, 1) + l_prev
    l_rcp = 1.0 / l_curr
    p *= l_rcp[:, None]
    acc *= (l_prev * l_rcp)[:, None]
    p = (p).to(tl.float16)
    v = tl.load(v_ptrs, block_n_offs[:, None] < $(N_CTX), other=0.0)
    acc += tl.dot(p, v)
    l_prev = l_curr
    m_prev = m_curr
    k_ptrs += $(BLOCK_N) * $(stride_kn)
    v_ptrs += $(BLOCK_N) * $(stride_vk)
  }
  start_m = tl.program_id(0)
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))

  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  off_o = batch_id * $(stride_oz) + off_hz * $(stride_oh) + offs_m[:, None] * $(stride_om) + offs_d[None, :] * $(stride_on)
  out_ptrs = Out + off_o
  tl.store(out_ptrs, (acc).to(tl.float16), offs_m[:, None] < $(H))
}).surfaceBody
```
</details>

<details><summary><code>decode</code></summary>

```lean
noncomputable def decode (b : Byte) : ℝ :=
  (((Float8E5Bits.ofInt b.val).decodeRat.getD 0 : ℚ) : ℝ)

section
variable (Q K V Out : RegionName) (scale : ℝ)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD T : Nat) (candidate causal : Bool)
```
</details>

<details><summary><code>score</code></summary>

```lean
noncomputable def score (T BM BN BD H : Nat) (hBN : 0 < BN) (hT : 0 < T)
    (scale : ℝ) (p : Nat) (xs : Fin T → Fin (BM * BD) → ℝ)
    (ys : Fin T → Fin (BN * BD) → ℝ) (r : Fin BM) (k : Fin (BN * T)) : ℝ :=
  scale * Finset.univ.sum (fun d : Fin BD =>
    query T BM BD p H hT xs (r, d, PUnit.unit) * stream T BN BD hBN ys (k, d, PUnit.unit))
```
</details>

<details><summary><code>stream</code></summary>

```lean
noncomputable abbrev stream := RealIO.stream
```
</details>

<details><summary><code>query</code></summary>

```
/-- Array indexing only: query reads the first stream slice, zeroing masked
rows; stream reconstructs global key k from step k/BN and lane k%BN. -/
```
```lean
noncomputable abbrev query := RealIO.query
```
</details>

## Public theorem: `attention_fp8_correctness`

<details><summary>docstring</summary>

```
/-- The same formula over the original integer K/V buffers, with exact E5M2
decoding. Q/Out are distinct from K/V so real cells cannot replace byte cells. -/
```
</details>

**Statement:**
```lean
specification attention_fp8_correctness (hBM : 0 < BM) (hBN : 0 < BN) (hT : 0 < T)
    (hQK : Q ≠ K) (hQV : Q ≠ V) (hOK : Out ≠ K) (hOV : Out ≠ V)
    (hinj : Function.Injective (fun i : TileIndex [BM, BD] =>
      i.1.val * som + i.2.1.val * son)) :
    Spec.Real ((byteIO Q K V Out scale
      sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD T candidate causal).ImplementsCells
      MemCell.real byteCell byteCell
      (fun p _ _ xs ys zs => expected T BM BN BD H SP hBN hT causal scale p xs
        (fun t i => decode (ys t i)) (fun t i => decode (zs t i))))
```

**Assumptions / layout contracts:**
- `hBM : 0 < BM`
- `hBN : 0 < BN`
- `hT : 0 < T`
- `hQK : Q ≠ K`
- `hQV : Q ≠ V`
- `hOK : Out ≠ K`
- `hOV : Out ≠ V`
- `hinj : Function.Injective (fun i : TileIndex [BM, BD] =>
      i.1.val * som + i.2.1.val * son)`

**Closed-form spec defs (transitive):** `byteIO`, `byteCell`, `expected`, `decode`, `io`, `carryFirstFP8`, `originalFP8`, `score`, `stream`, `carryFirst`, `original`, `query`

<details><summary><code>byteIO</code></summary>

```
/-- The FP8 sources use the same addresses and masks, loading signed integer
K/V cells before bit reinterpretation and fp16 conversion inside the kernel. -/
```
```lean
def byteIO (Q K V Out : RegionName) (scale : ℝ)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD T : Nat) (candidate causal : Bool) : StreamMasked3DKernelIO₃ :=
  { io Q K V Out scale
      sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD T candidate causal with
    kernel := (if candidate then
      TypedSource.carryFirstFP8 Q K V Out scale
      sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD causal
    else
      TypedSource.originalFP8 Q K V Out scale
      sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD causal).eraseDType
    projection := FP8RealSource.projection_supported .. }
```
</details>

<details><summary><code>byteCell</code></summary>

```lean
def byteCell (b : Byte) : MemCell := .of .int b.val
```
</details>

<details><summary><code>expected</code></summary>

```
/-- Independent Σ exp(score)*V / Σ exp(score). Under the causal launch
precondition, k ≤ p*BM-SP+r is exactly k+SP ≤ p*BM+r. This source shrinks
visibility by SP; it does not add a prefix cache to the visible keys. -/
```
```lean
noncomputable def expected (T BM BN BD H SP : Nat) (hBN : 0 < BN) (hT : 0 < T)
    (causal : Bool) (scale : ℝ) (p : Nat)
    (xs : Fin T → Fin (BM * BD) → ℝ)
    (ys zs : Fin T → Fin (BN * BD) → ℝ) (j : Fin (BM * BD)) : ℝ :=
  let i := Lane2D.decode j
  let s := score T BM BN BD H hBN hT scale p xs ys i.1
  if causal then
    let weight := fun k : Fin (BN * T) =>
      if k.val ≤ p * BM - SP + i.1.val then Real.exp (s k) else 0
    (Finset.univ.sum fun k => weight k * stream T BN BD hBN zs (k, i.2.1, PUnit.unit)) /
      (Finset.univ.sum fun k => weight k)
  else
    (Finset.univ.sum fun k => Real.exp (s k) * stream T BN BD hBN zs (k, i.2.1, PUnit.unit)) /
      (Finset.univ.sum fun k => Real.exp (s k))
```
</details>

<details><summary><code>decode</code></summary>

```lean
noncomputable def decode (b : Byte) : ℝ :=
  (((Float8E5Bits.ofInt b.val).decodeRat.getD 0 : ℚ) : ℝ)

section
variable (Q K V Out : RegionName) (scale : ℝ)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD T : Nat) (candidate causal : Bool)
```
</details>

<details><summary><code>io</code></summary>

```
/-- Q is static across steps; K/V advance by BN rows. All four buffers keep
their independent batch/head/row/column strides. Query and output masks use H;
the whole-block launch precondition makes all streamed K/V lanes active. -/
```
```lean
def io (Q K V Out : RegionName) (scale : ℝ)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD T : Nat) (candidate causal : Bool) : StreamMasked3DKernelIO₃ where
  kernel := (if candidate then
    TypedSource.carryFirst Q K V Out scale
      sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD causal
  else
    TypedSource.original Q K V Out scale
      sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD causal).eraseDType
  projection := by
    cases candidate
    · exact RealSource.original_projection_supported ..
    · exact RealSource.candidate_projection_supported ..
  inp1 := Q
  inp2 := K
  inp3 := V
  out := Out
  T := T
  B1 := BM * BD
  B2 := BN * BD
  B3 := BN * BD
  C := BM * BD
  outDType := .real
  pre := fun p _ _ => if causal then (p + 1) * BN + SP = BN * T ∧ BN * T ≤ N ∧ SP ≤ p * BM
    else N = BN * T
  read1 := fun p h _ _ j => h / NH * sqz + h % NH * sqh +
    (p * BM + j.val / BD) * sqm + j.val % BD * sqk
  read2 := fun _ h _ t j => h / NH * skz + h % NH * skh +
    (t.val * BN + j.val / BD) * skn + j.val % BD * skk
  read3 := fun _ h _ t j => h / NH * svz + h % NH * svh +
    (t.val * BN + j.val / BD) * svk + j.val % BD * svn
  write := fun p h _ j => h / NH * soz + h % NH * soh +
    (p * BM + j.val / BD) * som + j.val % BD * son
  mask1 := fun p _ _ _ j => p * BM + j.val / BD < H
  mask2 := fun _ _ _ _ _ => True
  mask3 := fun _ _ _ _ _ => True
  writeMask := fun p _ _ j => p * BM + j.val / BD < H
```
</details>

<details><summary><code>carryFirstFP8</code></summary>

```lean
def carryFirstFP8
    (Q K V Out : RegionName) (sm_scale : ℝ)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      N_HEAD H N_CTX start_position
      BLOCK_M BLOCK_N BLOCK_DMODEL : Nat) (IS_CAUSAL : Bool) :
    ComputeKernel := .mk [Q, K, V] [Out] (triton {
  tl.region Q = tl.float16, K = tl.int8, V = tl.int8, Out = tl.float16
  start_m = tl.program_id(0)

  head_idx = tl.program_id(1)
  batch_id = head_idx // $(N_HEAD)
  off_hz = head_idx % $(N_HEAD)

  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_n = tl.arange(0, $(BLOCK_N))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  off_q = batch_id * $(stride_qz) + off_hz * $(stride_qh) + offs_m[:, None] * $(stride_qm) + offs_d[None, :] * $(stride_qk)
  off_k = batch_id * $(stride_kz) + off_hz * $(stride_kh) + offs_n[:, None] * $(stride_kn) + offs_d[None, :] * $(stride_kk)
  off_v = batch_id * $(stride_vz) + off_hz * $(stride_vh) + offs_n[:, None] * $(stride_vk) + offs_d[None, :] * $(stride_vn)
  q_ptrs = Q + off_q
  k_ptrs = K + off_k
  v_ptrs = V + off_v
  m_prev = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_prev = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)
  q = tl.load(q_ptrs, offs_m[:, None] < $(H), other=0.0)
  block_n_end = $(N_CTX)
  if IS_CAUSAL {
    block_n_end = (start_m + $(1)) * $(BLOCK_N) + $(start_position)
  }
  for start_n in range($(0), block_n_end, $(BLOCK_N)) {
    block_n_offs = start_n + offs_n
    k = tl.load(k_ptrs, block_n_offs[:, None] < $(N_CTX), other=0.0)
    k = k.to(tl.float8e5, bitcast=True)
    k = k.to(tl.float16)
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    qk += tl.dot(q, tl.trans(k))
    qk = tl.where(offs_n[None, :] < $(N_CTX), qk, float("-inf"))
    qk *= $((sm_scale : ℝ))
    if IS_CAUSAL {
      qk = tl.where(offs_m[:, None] >= (block_n_offs[None, :] + $(start_position)), qk, float("-inf"))
    }
    m_curr = tl.maximum(tl.max(qk, 1), m_prev)
    l_prev *= tl.exp(m_prev - m_curr)
    p = tl.exp(qk - m_curr[:, None])
    l_curr = l_prev + tl.sum(p, 1)
    l_rcp = 1.0 / l_curr
    p *= l_rcp[:, None]
    acc *= (l_prev * l_rcp)[:, None]
    p = (p).to(tl.float16)
    v = tl.load(v_ptrs, block_n_offs[:, None] < $(N_CTX), other=0.0)
    v = v.to(tl.float8e5, bitcast=True)
    v = v.to(tl.float16)
    acc += tl.dot(p, v)
    l_prev = l_curr
    m_prev = m_curr
    k_ptrs += $(BLOCK_N) * $(stride_kn)
    v_ptrs += $(BLOCK_N) * $(stride_vk)
  }
  start_m = tl.program_id(0)
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))

  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  off_o = batch_id * $(stride_oz) + off_hz * $(stride_oh) + offs_m[:, None] * $(stride_om) + offs_d[None, :] * $(stride_on)
  out_ptrs = Out + off_o
  tl.store(out_ptrs, (acc).to(tl.float16), offs_m[:, None] < $(H))
}).surfaceBody
```
</details>

<details><summary><code>originalFP8</code></summary>

```
/-- USE_FP8=True specialization. K and V store signed int8 payloads; each
load is followed by E5M2 bit reinterpretation and a separate fp16 conversion.
Both causal conditionals and every other source statement are retained. -/
```
```lean
def originalFP8
    (Q K V Out : RegionName) (sm_scale : ℝ)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      N_HEAD H N_CTX start_position
      BLOCK_M BLOCK_N BLOCK_DMODEL : Nat) (IS_CAUSAL : Bool) :
    ComputeKernel := .mk [Q, K, V] [Out] (triton {
  tl.region Q = tl.float16, K = tl.int8, V = tl.int8, Out = tl.float16
  start_m = tl.program_id(0)

  head_idx = tl.program_id(1)
  batch_id = head_idx // $(N_HEAD)
  off_hz = head_idx % $(N_HEAD)

  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_n = tl.arange(0, $(BLOCK_N))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  off_q = batch_id * $(stride_qz) + off_hz * $(stride_qh) + offs_m[:, None] * $(stride_qm) + offs_d[None, :] * $(stride_qk)
  off_k = batch_id * $(stride_kz) + off_hz * $(stride_kh) + offs_n[:, None] * $(stride_kn) + offs_d[None, :] * $(stride_kk)
  off_v = batch_id * $(stride_vz) + off_hz * $(stride_vh) + offs_n[:, None] * $(stride_vk) + offs_d[None, :] * $(stride_vn)
  q_ptrs = Q + off_q
  k_ptrs = K + off_k
  v_ptrs = V + off_v
  m_prev = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_prev = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)
  q = tl.load(q_ptrs, offs_m[:, None] < $(H), other=0.0)
  block_n_end = $(N_CTX)
  if IS_CAUSAL {
    block_n_end = (start_m + $(1)) * $(BLOCK_N) + $(start_position)
  }
  for start_n in range($(0), block_n_end, $(BLOCK_N)) {
    block_n_offs = start_n + offs_n
    k = tl.load(k_ptrs, block_n_offs[:, None] < $(N_CTX), other=0.0)
    k = k.to(tl.float8e5, bitcast=True)
    k = k.to(tl.float16)
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    qk += tl.dot(q, tl.trans(k))
    qk = tl.where(offs_n[None, :] < $(N_CTX), qk, float("-inf"))
    qk *= $((sm_scale : ℝ))
    if IS_CAUSAL {
      qk = tl.where(offs_m[:, None] >= (block_n_offs[None, :] + $(start_position)), qk, float("-inf"))
    }
    m_curr = tl.maximum(tl.max(qk, 1), m_prev)
    l_prev *= tl.exp(m_prev - m_curr)
    p = tl.exp(qk - m_curr[:, None])
    l_curr = tl.sum(p, 1) + l_prev
    l_rcp = 1.0 / l_curr
    p *= l_rcp[:, None]
    acc *= (l_prev * l_rcp)[:, None]
    p = (p).to(tl.float16)
    v = tl.load(v_ptrs, block_n_offs[:, None] < $(N_CTX), other=0.0)
    v = v.to(tl.float8e5, bitcast=True)
    v = v.to(tl.float16)
    acc += tl.dot(p, v)
    l_prev = l_curr
    m_prev = m_curr
    k_ptrs += $(BLOCK_N) * $(stride_kn)
    v_ptrs += $(BLOCK_N) * $(stride_vk)
  }
  start_m = tl.program_id(0)
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))

  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  off_o = batch_id * $(stride_oz) + off_hz * $(stride_oh) + offs_m[:, None] * $(stride_om) + offs_d[None, :] * $(stride_on)
  out_ptrs = Out + off_o
  tl.store(out_ptrs, (acc).to(tl.float16), offs_m[:, None] < $(H))
}).surfaceBody
```
</details>

<details><summary><code>score</code></summary>

```lean
noncomputable def score (T BM BN BD H : Nat) (hBN : 0 < BN) (hT : 0 < T)
    (scale : ℝ) (p : Nat) (xs : Fin T → Fin (BM * BD) → ℝ)
    (ys : Fin T → Fin (BN * BD) → ℝ) (r : Fin BM) (k : Fin (BN * T)) : ℝ :=
  scale * Finset.univ.sum (fun d : Fin BD =>
    query T BM BD p H hT xs (r, d, PUnit.unit) * stream T BN BD hBN ys (k, d, PUnit.unit))
```
</details>

<details><summary><code>stream</code></summary>

```lean
noncomputable abbrev stream := RealIO.stream
```
</details>

<details><summary><code>carryFirst</code></summary>

```lean
def carryFirst
    (Q K V Out : RegionName) (sm_scale : ℝ)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      N_HEAD H N_CTX start_position
      BLOCK_M BLOCK_N BLOCK_DMODEL : Nat) (IS_CAUSAL : Bool) :
    ComputeKernel := .mk [Q, K, V] [Out] (triton {
  tl.region Q = tl.float16, K = tl.float16, V = tl.float16, Out = tl.float16
  start_m = tl.program_id(0)

  head_idx = tl.program_id(1)
  batch_id = head_idx // $(N_HEAD)
  off_hz = head_idx % $(N_HEAD)

  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_n = tl.arange(0, $(BLOCK_N))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  off_q = batch_id * $(stride_qz) + off_hz * $(stride_qh) + offs_m[:, None] * $(stride_qm) + offs_d[None, :] * $(stride_qk)
  off_k = batch_id * $(stride_kz) + off_hz * $(stride_kh) + offs_n[:, None] * $(stride_kn) + offs_d[None, :] * $(stride_kk)
  off_v = batch_id * $(stride_vz) + off_hz * $(stride_vh) + offs_n[:, None] * $(stride_vk) + offs_d[None, :] * $(stride_vn)
  q_ptrs = Q + off_q
  k_ptrs = K + off_k
  v_ptrs = V + off_v
  m_prev = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_prev = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)
  q = tl.load(q_ptrs, offs_m[:, None] < $(H), other=0.0)
  block_n_end = $(N_CTX)
  if IS_CAUSAL {
    block_n_end = (start_m + $(1)) * $(BLOCK_N) + $(start_position)
  }
  for start_n in range($(0), block_n_end, $(BLOCK_N)) {
    block_n_offs = start_n + offs_n
    k = tl.load(k_ptrs, block_n_offs[:, None] < $(N_CTX), other=0.0)
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    qk += tl.dot(q, tl.trans(k))
    qk = tl.where(offs_n[None, :] < $(N_CTX), qk, float("-inf"))
    qk *= $((sm_scale : ℝ))
    if IS_CAUSAL {
      qk = tl.where(offs_m[:, None] >= (block_n_offs[None, :] + $(start_position)), qk, float("-inf"))
    }
    m_curr = tl.maximum(tl.max(qk, 1), m_prev)
    l_prev *= tl.exp(m_prev - m_curr)
    p = tl.exp(qk - m_curr[:, None])
    l_curr = l_prev + tl.sum(p, 1)
    l_rcp = 1.0 / l_curr
    p *= l_rcp[:, None]
    acc *= (l_prev * l_rcp)[:, None]
    p = (p).to(tl.float16)
    v = tl.load(v_ptrs, block_n_offs[:, None] < $(N_CTX), other=0.0)
    acc += tl.dot(p, v)
    l_prev = l_curr
    m_prev = m_curr
    k_ptrs += $(BLOCK_N) * $(stride_kn)
    v_ptrs += $(BLOCK_N) * $(stride_vk)
  }
  start_m = tl.program_id(0)
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))

  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  off_o = batch_id * $(stride_oz) + off_hz * $(stride_oh) + offs_m[:, None] * $(stride_om) + offs_d[None, :] * $(stride_on)
  out_ptrs = Out + off_o
  tl.store(out_ptrs, (acc).to(tl.float16), offs_m[:, None] < $(H))
}).surfaceBody
```
</details>

<details><summary><code>original</code></summary>

```lean
def original
    (Q K V Out : RegionName) (sm_scale : ℝ)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      N_HEAD H N_CTX start_position
      BLOCK_M BLOCK_N BLOCK_DMODEL : Nat) (IS_CAUSAL : Bool) :
    ComputeKernel := .mk [Q, K, V] [Out] (triton {
  tl.region Q = tl.float16, K = tl.float16, V = tl.float16, Out = tl.float16
  start_m = tl.program_id(0)

  head_idx = tl.program_id(1)
  batch_id = head_idx // $(N_HEAD)
  off_hz = head_idx % $(N_HEAD)

  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_n = tl.arange(0, $(BLOCK_N))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  off_q = batch_id * $(stride_qz) + off_hz * $(stride_qh) + offs_m[:, None] * $(stride_qm) + offs_d[None, :] * $(stride_qk)
  off_k = batch_id * $(stride_kz) + off_hz * $(stride_kh) + offs_n[:, None] * $(stride_kn) + offs_d[None, :] * $(stride_kk)
  off_v = batch_id * $(stride_vz) + off_hz * $(stride_vh) + offs_n[:, None] * $(stride_vk) + offs_d[None, :] * $(stride_vn)
  q_ptrs = Q + off_q
  k_ptrs = K + off_k
  v_ptrs = V + off_v
  m_prev = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_prev = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)
  q = tl.load(q_ptrs, offs_m[:, None] < $(H), other=0.0)
  block_n_end = $(N_CTX)
  if IS_CAUSAL {
    block_n_end = (start_m + $(1)) * $(BLOCK_N) + $(start_position)
  }
  for start_n in range($(0), block_n_end, $(BLOCK_N)) {
    block_n_offs = start_n + offs_n
    k = tl.load(k_ptrs, block_n_offs[:, None] < $(N_CTX), other=0.0)
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    qk += tl.dot(q, tl.trans(k))
    qk = tl.where(offs_n[None, :] < $(N_CTX), qk, float("-inf"))
    qk *= $((sm_scale : ℝ))
    if IS_CAUSAL {
      qk = tl.where(offs_m[:, None] >= (block_n_offs[None, :] + $(start_position)), qk, float("-inf"))
    }
    m_curr = tl.maximum(tl.max(qk, 1), m_prev)
    l_prev *= tl.exp(m_prev - m_curr)
    p = tl.exp(qk - m_curr[:, None])
    l_curr = tl.sum(p, 1) + l_prev
    l_rcp = 1.0 / l_curr
    p *= l_rcp[:, None]
    acc *= (l_prev * l_rcp)[:, None]
    p = (p).to(tl.float16)
    v = tl.load(v_ptrs, block_n_offs[:, None] < $(N_CTX), other=0.0)
    acc += tl.dot(p, v)
    l_prev = l_curr
    m_prev = m_curr
    k_ptrs += $(BLOCK_N) * $(stride_kn)
    v_ptrs += $(BLOCK_N) * $(stride_vk)
  }
  start_m = tl.program_id(0)
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))

  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  off_o = batch_id * $(stride_oz) + off_hz * $(stride_oh) + offs_m[:, None] * $(stride_om) + offs_d[None, :] * $(stride_on)
  out_ptrs = Out + off_o
  tl.store(out_ptrs, (acc).to(tl.float16), offs_m[:, None] < $(H))
}).surfaceBody
```
</details>

<details><summary><code>query</code></summary>

```
/-- Array indexing only: query reads the first stream slice, zeroing masked
rows; stream reconstructs global key k from step k/BN and lane k%BN. -/
```
```lean
noncomputable abbrev query := RealIO.query
```
</details>
