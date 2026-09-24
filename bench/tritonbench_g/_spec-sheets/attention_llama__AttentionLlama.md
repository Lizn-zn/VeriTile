# Spec sheet — `bench/tritonbench_g/attention_llama/AttentionLlama.lean`

**Python source:** `bench/tritonbench_g/attention_llama/attention_llama.py`

## Public theorem: `attention_llama_fwd_closed_form_correct`

<details><summary>docstring</summary>

```
/-- **★ MAIN (non-causal).** Every ACTIVE `Out` cell (query row `< H`, the
kernel's own masked store) holds the FULL natural-exp softmax attention
over the `N_CTX` keys at scale `sm_scale`, read from input memory through
the kernel's own offset expressions: the H-guarded Q tile (`alQTileG` —
the guard is the kernel's own `offs_m < H` load mask, true on every
active row) against the K/V tiles at the batch/head base. Side
conditions: `N_CTX = BLOCK_N · numKVBlocks` (the ghost-lane divisibility;
see the header), `0 < numKVBlocks`, output-offset injectivity, and the
clean-undef carrier (which also pins the k/v masked loads' dead lanes to
the Python's positional `other=0.`). -/
```
</details>

**Statement:**
```lean
specification attention_llama_fwd_closed_form_correct
    (Q K V Out : RegionName) (s : BlockState) (sm_scale : ℝ)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      N_HEAD H N_CTX start_position BLOCK_M BLOCK_N BLOCK_DMODEL numKVBlocks : Nat)
    (hBM : 0 < BLOCK_M) (hBN : 0 < BLOCK_N)
    (hSEQ : N_CTX = BLOCK_N * numKVBlocks) (hnum : 0 < numKVBlocks)
    (houtinj : Function.Injective (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        (s.pids 0 * BLOCK_M + idx.1.val) * stride_om + idx.2.1.val * stride_on))
    (hundef : ∀ rg o, s.undef rg o = 0) :
    (∃ alg, (attention_llama_fwd_surface Q K V Out sm_scale
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      N_HEAD H N_CTX start_position BLOCK_M BLOCK_N BLOCK_DMODEL).toAlgorithm?
        = Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := attention_llama_fwd_surface Q K V Out sm_scale
        stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
        stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
        N_HEAD H N_CTX start_position BLOCK_M BLOCK_N BLOCK_DMODEL)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => alActive s H BLOCK_M idx)
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          (Out, alOutOffset s N_HEAD stride_oz stride_oh stride_om stride_on BLOCK_M idx)))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        attentionReal
          (alQTileG s Q N_HEAD stride_qz stride_qh stride_qm stride_qk H BLOCK_M BLOCK_DMODEL)
          (alKTileG s K N_HEAD stride_kz stride_kh stride_kn stride_kk N_CTX BLOCK_DMODEL)
          (alVTileG s V N_HEAD stride_vz stride_vh stride_vk stride_vn N_CTX BLOCK_DMODEL)
          sm_scale idx)
```

**Assumptions / layout contracts:**
- `hBM : 0 < BLOCK_M`
- `hBN : 0 < BLOCK_N`
- `hSEQ : N_CTX = BLOCK_N * numKVBlocks`
- `hnum : 0 < numKVBlocks`
- `hundef : ∀ rg o, s.undef rg o = 0`

**Closed-form spec defs (transitive):** `attention_llama_fwd_surface`, `alActive`, `alOutOffset`, `alQTileG`, `alKTileG`, `alVTileG`, `alRow`, `alBase`

<details><summary><code>attention_llama_fwd_surface</code></summary>

```
/-- DSL port of `attention_llama.py`'s `_fwd_kernel`, `IS_CAUSAL = False`
arm (`USE_FP8 = False`; see the header blocker for the dropped fp8 arm).
`block_n_end = N_CTX` is kept faithfully as a runtime scalar register, so
the streaming loop is a `forRangeDyn` over it. `start_position` is passed
but (faithfully) unused in this arm. -/
```
```lean
def attention_llama_fwd_surface
    (Q K V Out : RegionName) (sm_scale : ℝ)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      N_HEAD H N_CTX _start_position
      BLOCK_M BLOCK_N BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
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
  for start_n in range($(0), block_n_end, $(BLOCK_N)) {
    block_n_offs = start_n + offs_n
    k = tl.load(k_ptrs, block_n_offs[:, None] < $(N_CTX))
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    qk += tl.dot(q, tl.trans(k))
    qk = tl.where(offs_n[None, :] < $(N_CTX), qk, float("-inf"))
    qk *= $((sm_scale : ℝ))
    m_curr = tl.maximum(tl.max(qk, 1), m_prev)
    l_prev *= tl.exp(m_prev - m_curr)
    p = tl.exp(qk - m_curr[:, None])
    l_curr = tl.sum(p, 1) + l_prev
    l_rcp = 1.0 / l_curr
    p *= l_rcp[:, None]
    acc *= (l_prev * l_rcp)[:, None]
    p = (p).to(Q.dtype.element_ty)
    v = tl.load(v_ptrs, block_n_offs[:, None] < $(N_CTX))
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
  tl.store(out_ptrs, acc, offs_m[:, None] < $(H))
}
```
</details>

<details><summary><code>alActive</code></summary>

```
/-- Active (stored) output lanes: the kernel's `offs_m < H` store mask
(`H` = the launcher's `m_size`, the QUERY length). -/
```
```lean
def alActive {BD : Nat} (s : BlockState) (H BM : Nat)
    (idx : TileIndex [BM, BD]) : Prop :=
  alRow s BM idx.1 < H

instance {BD : Nat} (s : BlockState) (H BM : Nat) :
    DecidablePred (alActive (BD := BD) s H BM) := fun idx =>
  inferInstanceAs (Decidable (alRow s BM idx.1 < H))
```
</details>

<details><summary><code>alOutOffset</code></summary>

```
/-- Flat `Out` offset of output lane `idx` (the kernel's own
`off_o = batch_id·stride_oz + off_hz·stride_oh + row·stride_om + col·stride_on`). -/
```
```lean
def alOutOffset (s : BlockState) (N_HEAD soz soh som son BM : Nat)
    {BD : Nat} (idx : TileIndex [BM, BD]) : Nat :=
  alBase s N_HEAD soz soh + alRow s BM idx.1 * som + idx.2.1.val * son
```
</details>

<details><summary><code>alQTileG</code></summary>

```
/-- The loaded (H-guarded) Q tile: the kernel's masked
`q = tl.load(q_ptrs, offs_m[:,None] < H, other=0.0)`. -/
```
```lean
noncomputable def alQTileG (s : BlockState) (Q : RegionName)
    (N_HEAD sqz sqh sqm sqk H BM BD : Nat) :
    TileIndex [BM, BD] → ℝ :=
  fun idx =>
    if alRow s BM idx.1 < H then
      s.readMem Q (alBase s N_HEAD sqz sqh + alRow s BM idx.1 * sqm + idx.2.1.val * sqk)
    else 0
```
</details>

<details><summary><code>alKTileG</code></summary>

```
/-- The K tile over the streamed key span `SEQ` (every loaded lane is
in-range under the headline's side conditions, so no guard). -/
```
```lean
noncomputable def alKTileG (s : BlockState) (K : RegionName)
    (N_HEAD skz skh skn skk SEQ BD : Nat) :
    TileIndex [SEQ, BD] → ℝ :=
  fun idx =>
    s.readMem K (alBase s N_HEAD skz skh + idx.1.val * skn + idx.2.1.val * skk)
```
</details>

<details><summary><code>alVTileG</code></summary>

```
/-- The V tile over the streamed key span `SEQ`. -/
```
```lean
noncomputable def alVTileG (s : BlockState) (V : RegionName)
    (N_HEAD svz svh svk svn SEQ BD : Nat) :
    TileIndex [SEQ, BD] → ℝ :=
  fun idx =>
    s.readMem V (alBase s N_HEAD svz svh + idx.1.val * svk + idx.2.1.val * svn)
```
</details>

<details><summary><code>alRow</code></summary>

```
/-- Global query row of local row `i`. -/
```
```lean
def alRow (s : BlockState) (BM : Nat) (i : Fin BM) : Nat :=
  s.pids 0 * BM + i.val
```
</details>

<details><summary><code>alBase</code></summary>

```
/-- The program's batch/head base offset for a region with batch stride
`sz` and head stride `sh`: `batch_id·sz + off_hz·sh` with
`batch_id = pids 1 / N_HEAD`, `off_hz = pids 1 % N_HEAD`. -/
```
```lean
def alBase (s : BlockState) (N_HEAD sz sh : Nat) : Nat :=
  (s.pids 1 / N_HEAD) * sz + (s.pids 1 % N_HEAD) * sh
```
</details>

## Public theorem: `attention_llama_fwd_causal_closed_form_correct`

<details><summary>docstring</summary>

```
/-- **★ MAIN (causal).** Every ACTIVE `Out` cell (query row `< H`) holds
the natural-exp CAUSAL softmax attention over the streamed key span
`span = (pid₀+1)·BLOCK_N + start_position = BLOCK_N·numCausalBlocks`
(`hspanEq` — equivalent to `BLOCK_N ∣ start_position` at this pid): the
visible keys of query row `row = pid₀·BLOCK_M + i` are exactly
`{j < span : j ≤ (pid₀·BLOCK_M − start_position) + i}`, i.e.
`j + start_position ≤ row` under `hsp : start_position ≤ pid₀·BLOCK_M` —
the kernel's own `offs_m ≥ block_n_offs + start_position` predicate
(`start_position` SHRINKS visibility; see the header quirk note). Side
conditions: `hspanEq`, `span ≤ N_CTX` (`hspanle`, which makes every
loaded key lane in-range — the causal arm tolerates arbitrary
`N_CTX ≥ span`), `hsp`, output-offset injectivity, and the clean-undef
carrier. -/
```
</details>

**Statement:**
```lean
specification attention_llama_fwd_causal_closed_form_correct
    (Q K V Out : RegionName) (s : BlockState) (sm_scale : ℝ)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      N_HEAD H N_CTX start_position BLOCK_M BLOCK_N BLOCK_DMODEL numCausalBlocks : Nat)
    (hBM : 0 < BLOCK_M) (hBN : 0 < BLOCK_N)
    (hspanEq : (s.pids 0 + 1) * BLOCK_N + start_position = BLOCK_N * numCausalBlocks)
    (hspanle : BLOCK_N * numCausalBlocks ≤ N_CTX)
    (hsp : start_position ≤ s.pids 0 * BLOCK_M)
    (houtinj : Function.Injective (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        (s.pids 0 * BLOCK_M + idx.1.val) * stride_om + idx.2.1.val * stride_on))
    (hundef : ∀ rg o, s.undef rg o = 0) :
    (∃ alg, (attention_llama_fwd_causal_surface Q K V Out sm_scale
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      N_HEAD H N_CTX start_position BLOCK_M BLOCK_N BLOCK_DMODEL).toAlgorithm?
        = Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := attention_llama_fwd_causal_surface Q K V Out sm_scale
        stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
        stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
        N_HEAD H N_CTX start_position BLOCK_M BLOCK_N BLOCK_DMODEL)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => alActive s H BLOCK_M idx)
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          (Out, alOutOffset s N_HEAD stride_oz stride_oh stride_om stride_on BLOCK_M idx)))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        attentionRealCausalBlock (s.pids 0 * BLOCK_M - start_position)
          (alQTileG s Q N_HEAD stride_qz stride_qh stride_qm stride_qk H BLOCK_M BLOCK_DMODEL)
          (alKTileG s K N_HEAD stride_kz stride_kh stride_kn stride_kk
            (BLOCK_N * numCausalBlocks) BLOCK_DMODEL)
          (alVTileG s V N_HEAD stride_vz stride_vh stride_vk stride_vn
            (BLOCK_N * numCausalBlocks) BLOCK_DMODEL)
          sm_scale idx)
```

**Assumptions / layout contracts:**
- `hBM : 0 < BLOCK_M`
- `hBN : 0 < BLOCK_N`
- `hspanEq : (s.pids 0 + 1) * BLOCK_N + start_position = BLOCK_N * numCausalBlocks`
- `hspanle : BLOCK_N * numCausalBlocks ≤ N_CTX`
- `hsp : start_position ≤ s.pids 0 * BLOCK_M`
- `hundef : ∀ rg o, s.undef rg o = 0`

**Closed-form spec defs (transitive):** `attention_llama_fwd_causal_surface`, `alActive`, `alOutOffset`, `alQTileG`, `alKTileG`, `alVTileG`, `alRow`, `alBase`

<details><summary><code>attention_llama_fwd_causal_surface</code></summary>

```
/-- DSL port of `attention_llama.py`'s `_fwd_kernel`, `IS_CAUSAL = True`
arm (`USE_FP8 = False`). Keeps BOTH `block_n_end` assignments and the
causal `tl.where` statement faithfully; everything else is identical to
the non-causal twin. -/
```
```lean
def attention_llama_fwd_causal_surface
    (Q K V Out : RegionName) (sm_scale : ℝ)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      N_HEAD H N_CTX start_position
      BLOCK_M BLOCK_N BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
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
  block_n_end = (start_m + $(1)) * $(BLOCK_N) + $(start_position)
  for start_n in range($(0), block_n_end, $(BLOCK_N)) {
    block_n_offs = start_n + offs_n
    k = tl.load(k_ptrs, block_n_offs[:, None] < $(N_CTX))
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    qk += tl.dot(q, tl.trans(k))
    qk = tl.where(offs_n[None, :] < $(N_CTX), qk, float("-inf"))
    qk *= $((sm_scale : ℝ))
    qk = tl.where(offs_m[:, None] >= (block_n_offs[None, :] + $(start_position)), qk, float("-inf"))
    m_curr = tl.maximum(tl.max(qk, 1), m_prev)
    l_prev *= tl.exp(m_prev - m_curr)
    p = tl.exp(qk - m_curr[:, None])
    l_curr = tl.sum(p, 1) + l_prev
    l_rcp = 1.0 / l_curr
    p *= l_rcp[:, None]
    acc *= (l_prev * l_rcp)[:, None]
    p = (p).to(Q.dtype.element_ty)
    v = tl.load(v_ptrs, block_n_offs[:, None] < $(N_CTX))
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
  tl.store(out_ptrs, acc, offs_m[:, None] < $(H))
}
```
</details>

<details><summary><code>alActive</code></summary>

```
/-- Active (stored) output lanes: the kernel's `offs_m < H` store mask
(`H` = the launcher's `m_size`, the QUERY length). -/
```
```lean
def alActive {BD : Nat} (s : BlockState) (H BM : Nat)
    (idx : TileIndex [BM, BD]) : Prop :=
  alRow s BM idx.1 < H

instance {BD : Nat} (s : BlockState) (H BM : Nat) :
    DecidablePred (alActive (BD := BD) s H BM) := fun idx =>
  inferInstanceAs (Decidable (alRow s BM idx.1 < H))
```
</details>

<details><summary><code>alOutOffset</code></summary>

```
/-- Flat `Out` offset of output lane `idx` (the kernel's own
`off_o = batch_id·stride_oz + off_hz·stride_oh + row·stride_om + col·stride_on`). -/
```
```lean
def alOutOffset (s : BlockState) (N_HEAD soz soh som son BM : Nat)
    {BD : Nat} (idx : TileIndex [BM, BD]) : Nat :=
  alBase s N_HEAD soz soh + alRow s BM idx.1 * som + idx.2.1.val * son
```
</details>

<details><summary><code>alQTileG</code></summary>

```
/-- The loaded (H-guarded) Q tile: the kernel's masked
`q = tl.load(q_ptrs, offs_m[:,None] < H, other=0.0)`. -/
```
```lean
noncomputable def alQTileG (s : BlockState) (Q : RegionName)
    (N_HEAD sqz sqh sqm sqk H BM BD : Nat) :
    TileIndex [BM, BD] → ℝ :=
  fun idx =>
    if alRow s BM idx.1 < H then
      s.readMem Q (alBase s N_HEAD sqz sqh + alRow s BM idx.1 * sqm + idx.2.1.val * sqk)
    else 0
```
</details>

<details><summary><code>alKTileG</code></summary>

```
/-- The K tile over the streamed key span `SEQ` (every loaded lane is
in-range under the headline's side conditions, so no guard). -/
```
```lean
noncomputable def alKTileG (s : BlockState) (K : RegionName)
    (N_HEAD skz skh skn skk SEQ BD : Nat) :
    TileIndex [SEQ, BD] → ℝ :=
  fun idx =>
    s.readMem K (alBase s N_HEAD skz skh + idx.1.val * skn + idx.2.1.val * skk)
```
</details>

<details><summary><code>alVTileG</code></summary>

```
/-- The V tile over the streamed key span `SEQ`. -/
```
```lean
noncomputable def alVTileG (s : BlockState) (V : RegionName)
    (N_HEAD svz svh svk svn SEQ BD : Nat) :
    TileIndex [SEQ, BD] → ℝ :=
  fun idx =>
    s.readMem V (alBase s N_HEAD svz svh + idx.1.val * svk + idx.2.1.val * svn)
```
</details>

<details><summary><code>alRow</code></summary>

```
/-- Global query row of local row `i`. -/
```
```lean
def alRow (s : BlockState) (BM : Nat) (i : Fin BM) : Nat :=
  s.pids 0 * BM + i.val
```
</details>

<details><summary><code>alBase</code></summary>

```
/-- The program's batch/head base offset for a region with batch stride
`sz` and head stride `sh`: `batch_id·sz + off_hz·sh` with
`batch_id = pids 1 / N_HEAD`, `off_hz = pids 1 % N_HEAD`. -/
```
```lean
def alBase (s : BlockState) (N_HEAD sz sh : Nat) : Nat :=
  (s.pids 1 / N_HEAD) * sz + (s.pids 1 % N_HEAD) * sh
```
</details>
