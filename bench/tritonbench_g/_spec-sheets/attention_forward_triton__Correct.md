# Spec sheet — `bench/tritonbench_g/attention_forward_triton/Correct.lean`

**Python source:** `bench/tritonbench_g/attention_forward_triton/attention_forward_triton.py`

## Public theorem: `attention_correctness`

<details><summary>docstring</summary>

```
/-- Both positive key-block dimensions are needed for the nonempty reduction.
Active head lanes fit the tile and distinct output rows do not overlap. -/
```
</details>

**Statement:**
```lean
specification attention_correctness
    (Q K V QScale KScale Out : RegionName)
    (stride_qz stride_qh Z H HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE
      numKVBlocks : Nat)
    (hBN : 0 < BLOCK_N) (hnum : 0 < numKVBlocks)
    (hActiveLe : HEAD_ACTIVE ≤ BLOCK_DMODEL) (hHD : HEAD_ACTIVE ≤ HEAD_DIM) :
    Spec.Real (originalIO Q K V QScale KScale Out stride_qz stride_qh Z H HEAD_DIM
      BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE numKVBlocks ⊨
      fun p₀ _ _ xs ys zs ws vs j =>
        expected p₀ BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks xs ys zs ws vs j)
```

**Assumptions / layout contracts:**
- `hBN : 0 < BLOCK_N`
- `hnum : 0 < numKVBlocks`
- `hActiveLe : HEAD_ACTIVE ≤ BLOCK_DMODEL`
- `hHD : HEAD_ACTIVE ≤ HEAD_DIM`

**Closed-form spec defs (transitive):** `originalIO`, `expected`, `original`

<details><summary><code>originalIO</code></summary>

```
/-- Five input streams: static Q and Q_scale, and streamed K, V and K_scale.
The flat allocation must cover each active read/write address. Q and output
have the same row/head mask; the two scale loads are unmasked. -/
```
```lean
def originalIO (Q K V QScale KScale Out : RegionName)
    (stride_qz stride_qh Z H HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE
      numKVBlocks : Nat) : StreamMasked3DKernelIO₅ where
  kernel := (TypedSource.original Q K V QScale KScale Out
    stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
    stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
    Z H (BLOCK_N * numKVBlocks) HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE).eraseDType
  projection := RealNormalization.original_projection_supported ..
  inp1 := Q
  inp2 := K
  inp3 := V
  inp4 := QScale
  inp5 := KScale
  out := Out
  T := numKVBlocks
  B1 := BLOCK_M * BLOCK_DMODEL
  B2 := BLOCK_DMODEL * BLOCK_N
  B3 := BLOCK_N * BLOCK_DMODEL
  B4 := 1
  B5 := 1
  C := BLOCK_M * BLOCK_DMODEL
  outDType := .real
  read1 := fun p₀ p₁ _ _ j =>
    p₁ / H * stride_qz + p₁ % H * stride_qh
      + (p₀ * BLOCK_M + j.val / BLOCK_DMODEL) * HEAD_DIM + j.val % BLOCK_DMODEL
  read2 := fun _ p₁ _ t j =>
    p₁ / H * stride_qz + p₁ % H * stride_qh
      + j.val / BLOCK_N + (t.val * BLOCK_N + j.val % BLOCK_N) * HEAD_DIM
  read3 := fun _ p₁ _ t j =>
    p₁ / H * stride_qz + p₁ % H * stride_qh
      + (t.val * BLOCK_N + j.val / BLOCK_DMODEL) * HEAD_DIM + j.val % BLOCK_DMODEL
  read4 := fun p₀ p₁ _ _ _ =>
    p₁ * ((BLOCK_N * numKVBlocks + BLOCK_M - 1) / BLOCK_M) + p₀
  read5 := fun _ p₁ _ t _ =>
    p₁ * ((BLOCK_N * numKVBlocks + BLOCK_N - 1) / BLOCK_N) + t.val
  write := fun p₀ p₁ _ j =>
    p₁ / H * stride_qz + p₁ % H * stride_qh
      + (p₀ * BLOCK_M + j.val / BLOCK_DMODEL) * HEAD_DIM + j.val % BLOCK_DMODEL
  mask1 := fun p₀ _ _ _ j =>
    p₀ * BLOCK_M + j.val / BLOCK_DMODEL < BLOCK_N * numKVBlocks
      ∧ j.val % BLOCK_DMODEL < HEAD_ACTIVE
  mask2 := fun _ _ _ t j =>
    j.val % BLOCK_N < BLOCK_N * numKVBlocks - t.val * BLOCK_N ∧ j.val / BLOCK_N < HEAD_ACTIVE
  mask3 := fun _ _ _ t j =>
    j.val / BLOCK_DMODEL < BLOCK_N * numKVBlocks - t.val * BLOCK_N
      ∧ j.val % BLOCK_DMODEL < HEAD_ACTIVE
  mask4 := fun _ _ _ _ _ => True
  mask5 := fun _ _ _ _ _ => True
  writeMask := fun p₀ _ _ j =>
    p₀ * BLOCK_M + j.val / BLOCK_DMODEL < BLOCK_N * numKVBlocks
      ∧ j.val % BLOCK_DMODEL < HEAD_ACTIVE
```
</details>

<details><summary><code>expected</code></summary>

```
/-- Reconstruct the mathematical tiles from the five pinned input streams.
Q uses the repeated stream's first slice; K/V and key scales use the block
containing each global key. The helpers only index inputs, never execute code. -/
```
```lean
noncomputable def expected (p₀ BM BN BD HA T : Nat)
    (xs : Fin T → Fin (BM * BD) → ℝ)
    (ys : Fin T → Fin (BD * BN) → ℝ)
    (zs : Fin T → Fin (BN * BD) → ℝ)
    (ws vs : Fin T → Fin 1 → ℝ) (j : Fin (BM * BD)) : ℝ :=
  if h : j.val % BD < HA then
    let Q := attnIOqTm p₀ (BN * T) BM BD HA T xs
    let K := attnIOkTm BN BD HA T ys
    let V := attnIOvTm BN BD HA T zs
    let scale := attnIOkeyScale BN T ws vs
    let i := (Lane2D.decode j).1
    let d : Fin HA := ⟨j.val % BD, h⟩
    let weight := fun k : Fin (BN * T) => Real.exp (Real.log 2 *
      (scale k * ∑ e : Fin HA, Q (i, e, PUnit.unit) * K (k, e, PUnit.unit)))
    (∑ k : Fin (BN * T), weight k * V (k, d, PUnit.unit)) /
      ∑ k : Fin (BN * T), weight k
  else 0
```
</details>

<details><summary><code>original</code></summary>

```lean
def original
    (q k v qScale kScale out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk
      _stride_kz _stride_kh stride_kn _stride_kk
      _stride_vz _stride_vh _stride_vk _stride_vn
      _stride_oz _stride_oh _stride_om _stride_on
      _Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE _STAGE : Nat) :
    ComputeKernel :=
  let Q : Region .fp16 := q.cast
  let K : Region .fp16 := k.cast
  let V : Region .fp16 := v.cast
  let Q_scale : Region .fp32 := qScale.cast
  let K_scale : Region .fp32 := kScale.cast
  let Out : Region .fp16 := out.cast
  triton {
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
  for start_n in range(0, $(N_CTX), $(BLOCK_N)) {
    start_n = tl.multiple_of(start_n, $(BLOCK_N))
    k_mask = (offs_n[None, :] < ($(N_CTX) - start_n)) &
      (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[:, None]
    k = tl.load(K_ptrs, mask=k_mask)
    k_scale = tl.load(K_scale_ptr)
    qk = (tl.dot(q, k)).to(tl.float32) * q_scale * k_scale
    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    qk = qk - m_ij[:, None]
    p = tl.math.exp2(qk)
    l_ij = tl.sum(p, 1)
    alpha = tl.math.exp2(m_i - m_ij)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]
    v = tl.load(V_ptrs,
      mask=(offs_n[:, None] < ($(N_CTX) - start_n)) &
        (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[None, :])
    p = (p).to(tl.float16)
    acc += tl.dot(p, v, out_dtype=tl.float16)
    m_i = m_ij
    K_ptrs += $(BLOCK_N) * $(HEAD_DIM)
    K_scale_ptr += $(1)
    V_ptrs += $(BLOCK_N) * $(HEAD_DIM)
  }
  acc = acc / l_i[:, None]
  -- Out.type.element_ty is fp16 in the Python wrapper.
  tl.store(O_block_ptr, (acc).to(tl.float16),
    mask=(offs_m[:, None] < $(N_CTX)) & (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[None, :])
}
```
</details>

## Public theorem: `attention_reciprocal_correctness`

<details><summary>docstring</summary>

```
/-- The complete candidate satisfies the same independent attention formula. -/
```
</details>

**Statement:**
```lean
specification attention_reciprocal_correctness
    (Q K V QScale KScale Out : RegionName)
    (stride_qz stride_qh Z H HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE
      numKVBlocks : Nat)
    (hBN : 0 < BLOCK_N) (hnum : 0 < numKVBlocks)
    (hActiveLe : HEAD_ACTIVE ≤ BLOCK_DMODEL) (hHD : HEAD_ACTIVE ≤ HEAD_DIM) :
    Spec.Real (reciprocalIO Q K V QScale KScale Out stride_qz stride_qh Z H HEAD_DIM
      BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE numKVBlocks ⊨
      fun p₀ _ _ xs ys zs ws vs j =>
        expected p₀ BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks xs ys zs ws vs j)
```

**Assumptions / layout contracts:**
- `hBN : 0 < BLOCK_N`
- `hnum : 0 < numKVBlocks`
- `hActiveLe : HEAD_ACTIVE ≤ BLOCK_DMODEL`
- `hHD : HEAD_ACTIVE ≤ HEAD_DIM`

**Closed-form spec defs (transitive):** `reciprocalIO`, `expected`, `originalIO`, `reciprocal`, `original`

<details><summary><code>reciprocalIO</code></summary>

```
/-- The reciprocal candidate retains all six windows and masks. -/
```
```lean
def reciprocalIO (Q K V QScale KScale Out : RegionName)
    (stride_qz stride_qh Z H HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE
      numKVBlocks : Nat) : StreamMasked3DKernelIO₅ :=
  { originalIO Q K V QScale KScale Out stride_qz stride_qh Z H HEAD_DIM BLOCK_M BLOCK_N
      BLOCK_DMODEL HEAD_ACTIVE STAGE numKVBlocks with
    kernel := (TypedSource.reciprocal Q K V QScale KScale Out
      stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
      stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
      Z H (BLOCK_N * numKVBlocks) HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE).eraseDType
    projection := RealNormalization.reciprocal_projection_supported .. }
```
</details>

<details><summary><code>expected</code></summary>

```
/-- Reconstruct the mathematical tiles from the five pinned input streams.
Q uses the repeated stream's first slice; K/V and key scales use the block
containing each global key. The helpers only index inputs, never execute code. -/
```
```lean
noncomputable def expected (p₀ BM BN BD HA T : Nat)
    (xs : Fin T → Fin (BM * BD) → ℝ)
    (ys : Fin T → Fin (BD * BN) → ℝ)
    (zs : Fin T → Fin (BN * BD) → ℝ)
    (ws vs : Fin T → Fin 1 → ℝ) (j : Fin (BM * BD)) : ℝ :=
  if h : j.val % BD < HA then
    let Q := attnIOqTm p₀ (BN * T) BM BD HA T xs
    let K := attnIOkTm BN BD HA T ys
    let V := attnIOvTm BN BD HA T zs
    let scale := attnIOkeyScale BN T ws vs
    let i := (Lane2D.decode j).1
    let d : Fin HA := ⟨j.val % BD, h⟩
    let weight := fun k : Fin (BN * T) => Real.exp (Real.log 2 *
      (scale k * ∑ e : Fin HA, Q (i, e, PUnit.unit) * K (k, e, PUnit.unit)))
    (∑ k : Fin (BN * T), weight k * V (k, d, PUnit.unit)) /
      ∑ k : Fin (BN * T), weight k
  else 0
```
</details>

<details><summary><code>originalIO</code></summary>

```
/-- Five input streams: static Q and Q_scale, and streamed K, V and K_scale.
The flat allocation must cover each active read/write address. Q and output
have the same row/head mask; the two scale loads are unmasked. -/
```
```lean
def originalIO (Q K V QScale KScale Out : RegionName)
    (stride_qz stride_qh Z H HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE
      numKVBlocks : Nat) : StreamMasked3DKernelIO₅ where
  kernel := (TypedSource.original Q K V QScale KScale Out
    stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
    stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
    Z H (BLOCK_N * numKVBlocks) HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE).eraseDType
  projection := RealNormalization.original_projection_supported ..
  inp1 := Q
  inp2 := K
  inp3 := V
  inp4 := QScale
  inp5 := KScale
  out := Out
  T := numKVBlocks
  B1 := BLOCK_M * BLOCK_DMODEL
  B2 := BLOCK_DMODEL * BLOCK_N
  B3 := BLOCK_N * BLOCK_DMODEL
  B4 := 1
  B5 := 1
  C := BLOCK_M * BLOCK_DMODEL
  outDType := .real
  read1 := fun p₀ p₁ _ _ j =>
    p₁ / H * stride_qz + p₁ % H * stride_qh
      + (p₀ * BLOCK_M + j.val / BLOCK_DMODEL) * HEAD_DIM + j.val % BLOCK_DMODEL
  read2 := fun _ p₁ _ t j =>
    p₁ / H * stride_qz + p₁ % H * stride_qh
      + j.val / BLOCK_N + (t.val * BLOCK_N + j.val % BLOCK_N) * HEAD_DIM
  read3 := fun _ p₁ _ t j =>
    p₁ / H * stride_qz + p₁ % H * stride_qh
      + (t.val * BLOCK_N + j.val / BLOCK_DMODEL) * HEAD_DIM + j.val % BLOCK_DMODEL
  read4 := fun p₀ p₁ _ _ _ =>
    p₁ * ((BLOCK_N * numKVBlocks + BLOCK_M - 1) / BLOCK_M) + p₀
  read5 := fun _ p₁ _ t _ =>
    p₁ * ((BLOCK_N * numKVBlocks + BLOCK_N - 1) / BLOCK_N) + t.val
  write := fun p₀ p₁ _ j =>
    p₁ / H * stride_qz + p₁ % H * stride_qh
      + (p₀ * BLOCK_M + j.val / BLOCK_DMODEL) * HEAD_DIM + j.val % BLOCK_DMODEL
  mask1 := fun p₀ _ _ _ j =>
    p₀ * BLOCK_M + j.val / BLOCK_DMODEL < BLOCK_N * numKVBlocks
      ∧ j.val % BLOCK_DMODEL < HEAD_ACTIVE
  mask2 := fun _ _ _ t j =>
    j.val % BLOCK_N < BLOCK_N * numKVBlocks - t.val * BLOCK_N ∧ j.val / BLOCK_N < HEAD_ACTIVE
  mask3 := fun _ _ _ t j =>
    j.val / BLOCK_DMODEL < BLOCK_N * numKVBlocks - t.val * BLOCK_N
      ∧ j.val % BLOCK_DMODEL < HEAD_ACTIVE
  mask4 := fun _ _ _ _ _ => True
  mask5 := fun _ _ _ _ _ => True
  writeMask := fun p₀ _ _ j =>
    p₀ * BLOCK_M + j.val / BLOCK_DMODEL < BLOCK_N * numKVBlocks
      ∧ j.val % BLOCK_DMODEL < HEAD_ACTIVE
```
</details>

<details><summary><code>reciprocal</code></summary>

```
/-- Candidate using the admitted scalar div_mul_rcp relation at normalization.
Its finite/nonzero intermediate-value guards must be discharged in the FP proof. -/
```
```lean
def reciprocal
    (q k v qScale kScale out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk
      _stride_kz _stride_kh stride_kn _stride_kk
      _stride_vz _stride_vh _stride_vk _stride_vn
      _stride_oz _stride_oh _stride_om _stride_on
      _Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE _STAGE : Nat) :
    ComputeKernel :=
  let Q : Region .fp16 := q.cast
  let K : Region .fp16 := k.cast
  let V : Region .fp16 := v.cast
  let Q_scale : Region .fp32 := qScale.cast
  let K_scale : Region .fp32 := kScale.cast
  let Out : Region .fp16 := out.cast
  triton {
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
  for start_n in range(0, $(N_CTX), $(BLOCK_N)) {
    start_n = tl.multiple_of(start_n, $(BLOCK_N))
    k_mask = (offs_n[None, :] < ($(N_CTX) - start_n)) &
      (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[:, None]
    k = tl.load(K_ptrs, mask=k_mask)
    k_scale = tl.load(K_scale_ptr)
    qk = (tl.dot(q, k)).to(tl.float32) * q_scale * k_scale
    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    qk = qk - m_ij[:, None]
    p = tl.math.exp2(qk)
    l_ij = tl.sum(p, 1)
    alpha = tl.math.exp2(m_i - m_ij)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]
    v = tl.load(V_ptrs,
      mask=(offs_n[:, None] < ($(N_CTX) - start_n)) &
        (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[None, :])
    p = (p).to(tl.float16)
    acc += tl.dot(p, v, out_dtype=tl.float16)
    m_i = m_ij
    K_ptrs += $(BLOCK_N) * $(HEAD_DIM)
    K_scale_ptr += $(1)
    V_ptrs += $(BLOCK_N) * $(HEAD_DIM)
  }
  acc = acc * (1.0 / l_i[:, None])
  -- Out.type.element_ty is fp16 in the Python wrapper.
  tl.store(O_block_ptr, (acc).to(tl.float16),
    mask=(offs_m[:, None] < $(N_CTX)) & (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[None, :])
}
```
</details>

<details><summary><code>original</code></summary>

```lean
def original
    (q k v qScale kScale out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk
      _stride_kz _stride_kh stride_kn _stride_kk
      _stride_vz _stride_vh _stride_vk _stride_vn
      _stride_oz _stride_oh _stride_om _stride_on
      _Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE _STAGE : Nat) :
    ComputeKernel :=
  let Q : Region .fp16 := q.cast
  let K : Region .fp16 := k.cast
  let V : Region .fp16 := v.cast
  let Q_scale : Region .fp32 := qScale.cast
  let K_scale : Region .fp32 := kScale.cast
  let Out : Region .fp16 := out.cast
  triton {
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
  for start_n in range(0, $(N_CTX), $(BLOCK_N)) {
    start_n = tl.multiple_of(start_n, $(BLOCK_N))
    k_mask = (offs_n[None, :] < ($(N_CTX) - start_n)) &
      (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[:, None]
    k = tl.load(K_ptrs, mask=k_mask)
    k_scale = tl.load(K_scale_ptr)
    qk = (tl.dot(q, k)).to(tl.float32) * q_scale * k_scale
    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    qk = qk - m_ij[:, None]
    p = tl.math.exp2(qk)
    l_ij = tl.sum(p, 1)
    alpha = tl.math.exp2(m_i - m_ij)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]
    v = tl.load(V_ptrs,
      mask=(offs_n[:, None] < ($(N_CTX) - start_n)) &
        (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[None, :])
    p = (p).to(tl.float16)
    acc += tl.dot(p, v, out_dtype=tl.float16)
    m_i = m_ij
    K_ptrs += $(BLOCK_N) * $(HEAD_DIM)
    K_scale_ptr += $(1)
    V_ptrs += $(BLOCK_N) * $(HEAD_DIM)
  }
  acc = acc / l_i[:, None]
  -- Out.type.element_ty is fp16 in the Python wrapper.
  tl.store(O_block_ptr, (acc).to(tl.float16),
    mask=(offs_m[:, None] < $(N_CTX)) & (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[None, :])
}
```
</details>
