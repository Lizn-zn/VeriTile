# Spec sheet — `bench/tritonbench_g/attention_fwd_triton3/AttentionFwdTriton3.lean`

**Python source:** `bench/tritonbench_g/attention_fwd_triton3/attention_fwd_triton3.py`

## Public theorem: `attention_fwd_triton3_python_case1_output_summary_general`

<details><summary>docstring</summary>

```
/-- **Case 1 general genuine output summary.** -/
```
</details>

**Statement:**
```lean
specification attention_fwd_triton3_python_case1_output_summary_general
    (Q K V M Out L : RegionName) (sm_scale : ℝ)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      Z H H_KV N_CTX ROUND_CTX NKV_CTX off size BM ND BN : Nat) (s : BlockState)
    (hND : 0 < ND) (hBM : 0 < BM) (hBN : 0 < BN) (hNC : 0 < NKV_CTX) (hBNdvd : BN ∣ NKV_CTX)
    (hH : 0 < H) (hHKV : H_KV = H)
    (hskz : skz = sqz) (hskh : skh = sqh) (hsvz : svz = sqz) (hsvh : svh = sqh)
    (hsoz : soz = sqz) (hsoh : soh = sqh)
    (hMO : M ≠ Out) (hundef : ∀ rg o, s.undef rg o = 0)
    (hinjO : Function.Injective
      (fun idx : TileIndex [BM, ND] => (s.pids 1 / H * sqz + s.pids 1 % H * sqh) + (s.pids 0 * BM + idx.1.val) * som + idx.2.1.val * son))
    (hinjM : Function.Injective
      (fun r : TileIndex [BM] => s.pids 1 * ROUND_CTX + (s.pids 0 * BM + r.1.val))) :
    (∃ alg, (attention_fwd_triton3_surface Q K V M Out L sm_scale
      sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      Z H H_KV N_CTX ROUND_CTX NKV_CTX off size 1 1 BM ND BN 1 1 1 0).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := attention_fwd_triton3_surface Q K V M Out L sm_scale
        sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
        Z H H_KV N_CTX ROUND_CTX NKV_CTX off size 1 1 BM ND BN 1 1 1 0)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BM, ND] => active s N_CTX ND BM idx)
        (fun idx : TileIndex [BM, ND] => (Out, outOffset s H sqz sqh som son BM idx)))
      (expected := fun idx : TileIndex [BM, ND] =>
        attentionFwdTriton3Case1OutSpecG s Q K V (s.pids 1 / H * sqz + s.pids 1 % H * sqh) BM ND NKV_CTX sqm sqk skn skk svk svn (sm_scale * 1.4426950408889634) BN off size idx) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := attention_fwd_triton3_surface Q K V M Out L sm_scale
        sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
        Z H H_KV N_CTX ROUND_CTX NKV_CTX off size 1 1 BM ND BN 1 1 1 0)
      (initialState := s)
      (write := fun i : Fin BM => some (M, lRowOffset s (s.pids 1) ROUND_CTX BM i))
      (expected := fun i : Fin BM =>
        attentionFwdTriton3KMSpecG s Q K V (s.pids 1 / H * sqz + s.pids 1 % H * sqh) BM ND NKV_CTX sqm sqk skn skk svk svn (sm_scale * 1.4426950408889634) (fun i j => natSlidingWindowKeepG (s.pids 0) BM BN off size i j) i hND)
```

**Assumptions / layout contracts:**
- `hND : 0 < ND`
- `hBM : 0 < BM`
- `hBN : 0 < BN`
- `hNC : 0 < NKV_CTX`
- `hBNdvd : BN ∣ NKV_CTX`
- `hH : 0 < H`
- `hHKV : H_KV = H`
- `hskz : skz = sqz`
- `hskh : skh = sqh`
- `hsvz : svz = sqz`
- `hsvh : svh = sqh`
- `hsoz : soz = sqz`
- `hsoh : soh = sqh`
- `hMO : M ≠ Out`
- `hundef : ∀ rg o, s.undef rg o = 0`
- `fun idx : TileIndex [BM, ND] => (s.pids 1 / H * sqz + s.pids 1 % H * sqh) + (s.pids 0 * BM + idx.1.val) * som + idx.2.1.val * son`
- `fun r : TileIndex [BM] => s.pids 1 * ROUND_CTX + (s.pids 0 * BM + r.1.val)`
- `fun idx : TileIndex [BM, ND] => active s N_CTX ND BM idx`
- `fun idx : TileIndex [BM, ND] => (Out, outOffset s H sqz sqh som son BM idx)`

**Closed-form spec defs (transitive):** `attention_fwd_triton3_surface`, `active`, `outOffset`, `attentionFwdTriton3Case1OutSpecG`, `lRowOffset`, `attentionFwdTriton3KMSpecG`, `natSlidingWindowKeepG`, `mIndex`, `kIndex`, `offZ`, `offH`, `qTile3G`, `kTile3G`, `vTile3G`, `keyScale3G`, `aft3RunningMaxG`, `aft3StateBotKG`, `natDist3G`, `aft3KeysUptoG`, `aft3StateBotG`, `aft3OsStepBot`

<details><summary><code>attention_fwd_triton3_surface</code></summary>

```
/-- Full Lean port of `attention_fwd_triton3.py`'s `_attn_fwd`. -/
```
```lean
def attention_fwd_triton3_surface
    (Q K V M Out L : RegionName) (sm_scale : ℝ)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      _Z H H_KV N_CTX ROUND_CTX NKV_CTX
      _sliding_window_offset _sliding_window_size
      IS_EVEN_M _IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N END INIT
      _SLIDING_WINDOW _COMPLEMENT_SLIDING_WINDOW : Nat) :
    ComputeKernel := triton {
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
  qk_scale = $(sm_scale) * 1.0
  qk_scale *= 1.4426950408889634
  if $(IS_EVEN_M) != $(0) {
    q = tl.load(Q_block_ptr)
  } else {
    q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
  }
  for start_n in range($(0), $(NKV_CTX), $(BLOCK_N)) {
    k = tl.load(K_block_ptr)
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    qk += tl.dot(q, k)
    qk = qk * qk_scale
    if $(_SLIDING_WINDOW) != $(0) {
      dist = tl.arange(0, $(BLOCK_M))[:, None] - tl.arange(0, $(BLOCK_N))[None, :]
        + start_m * $(BLOCK_M) - start_n + $(_sliding_window_offset)
      if $(_COMPLEMENT_SLIDING_WINDOW) != $(0) {
        mask = dist >= $(_sliding_window_size)
      } else {
        mask = (dist >= $(0)) & (dist < $(_sliding_window_size))
      }
      qk = tl.where(mask, qk, float("-inf"))
    }
    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    qk = qk - m_ij[:, None]
    p = tl.math.exp2(qk)
    if $(_SLIDING_WINDOW) != $(0) {
      p = tl.where(mask, p, 0.0)
    }
    l_ij = tl.sum(p, 1)
    tmp = m_i - m_ij
    alpha = tl.math.exp2(tmp)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]
    v = tl.load(V_block_ptr)
    acc += tl.dot(p, v)
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
  tl.store(O_block_ptr, (acc).to(Out.type.element_ty))
}
```
</details>

<details><summary><code>active</code></summary>

```lean
def active
    (s : BlockState) (N_CTX HEAD_ACTIVE BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Prop :=
  mIndex s BLOCK_M idx.1 < N_CTX ∧ kIndex idx < HEAD_ACTIVE
```
</details>

<details><summary><code>outOffset</code></summary>

```lean
def outOffset
    (s : BlockState)
    (H stride_qz stride_qh stride_qm stride_qk BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  offZ s H * stride_qz + offH s H * stride_qh +
    mIndex s BLOCK_M idx.1 * stride_qm + kIndex idx * stride_qk
```
</details>

<details><summary><code>attentionFwdTriton3Case1OutSpecG</code></summary>

```
/-- General genuine closed form, case 1 (sliding window). -/
```
```lean
noncomputable def attentionFwdTriton3Case1OutSpecG
    (s : BlockState) (Q K V : RegionName)
    (base BM ND NC sqm sqk skn skk svk svn : Nat) (sc : ℝ) (BN off size : Nat)
    (idx : TileIndex [BM, ND]) : ℝ :=
  attentionRealBase2PerKeyScalePred (qTile3G s Q base BM ND sqm sqk)
    (kTile3G s K base NC ND skn skk) (vTile3G s V base NC ND svk svn)
    (keyScale3G sc NC) (fun i j => natSlidingWindowKeepG (s.pids 0) BM BN off size i j) idx
```
</details>

<details><summary><code>lRowOffset</code></summary>

```
/-- Proof-oriented L (log-sum-exp) row store slice of `attention_fwd_triton3.py`.
Takes a precomputed `LPre` vector and proves the row writeback into `L` at
offset `off_hz * ROUND_CTX + offs_m`. -/
```
```lean
def lRowOffset (s : BlockState) (off_hz ROUND_CTX BLOCK_M : Nat)
    (i : Fin BLOCK_M) : Nat :=
  off_hz * ROUND_CTX + (s.pids 0 * BLOCK_M + i.val)
```
</details>

<details><summary><code>attentionFwdTriton3KMSpecG</code></summary>

```
/-- General genuine `M`-row spec (cases 1/2): raw `(M ⊔ … + log2 l).unbotD`. -/
```
```lean
noncomputable def attentionFwdTriton3KMSpecG
    (s : BlockState) (Q K V : RegionName)
    (base BM ND NC sqm sqk skn skk svk svn : Nat) (sc : ℝ)
    (keep : Fin BM → Fin NC → Prop) [∀ i j, Decidable (keep i j)] (i : Fin BM) (hND : 0 < ND) : ℝ :=
  (WithBot.realAdd
      (aft3RunningMaxG (qTile3G s Q base BM ND sqm sqk) (kTile3G s K base NC ND skn skk)
        (vTile3G s V base NC ND svk svn) (keyScale3G sc NC) keep NC i ⟨0, hND⟩)
      (WithBot.realLog2 (((aft3StateBotKG (qTile3G s Q base BM ND sqm sqk) (kTile3G s K base NC ND skn skk)
        (vTile3G s V base NC ND svk svn) (keyScale3G sc NC) keep NC i ⟨0, hND⟩).2.1 : ℝ) : WithBot ℝ))).unbotD 0
```
</details>

<details><summary><code>natSlidingWindowKeepG</code></summary>

```
/-- General case-1 keep predicate: `dist < size`. -/
```
```lean
def natSlidingWindowKeepG (SM BM BN off size : Nat) {NC : Nat}
    (i : Fin BM) (j : Fin NC) : Prop :=
  natDist3G SM BM BN off i j < size
```
</details>

<details><summary><code>mIndex</code></summary>

```lean
def mIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val
```
</details>

<details><summary><code>kIndex</code></summary>

```lean
def kIndex (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  idx.2.1.val
```
</details>

<details><summary><code>offZ</code></summary>

```
/-- Surface transcription/proof-oriented final output-store slice of `attention_fwd_triton3.py`'s
`_attn_fwd`.

The full kernel runs separate streaming attention stages, including the causal
stage when requested. This slice starts after those stages have produced a
precomputed normalized `Acc` tile and proves the final masked writeback into
`Out`, preserving the source store address and mask
`(offs_m < N_CTX) & (offs_k < HEAD_ACTIVE)`. The inner `tl.float32` accumulator is
outside this slice. -/
```
```lean
def offZ (s : BlockState) (H : Nat) : Nat :=
  s.pids 1 / H
```
</details>

<details><summary><code>offH</code></summary>

```lean
def offH (s : BlockState) (H : Nat) : Nat :=
  s.pids 1 % H
```
</details>

<details><summary><code>qTile3G</code></summary>

```
/-- General query tile: query row `i`, head lane `e`, at
`base + (pid0·BM + i)·sqm + e·sqk`. -/
```
```lean
noncomputable def qTile3G (s : BlockState) (Q : RegionName)
    (base BM ND sqm sqk : Nat) : TileIndex [BM, ND] → ℝ :=
  fun (i, e, _) => s.readMem Q (base + (s.pids 0 * BM + i.val) * sqm + e.val * sqk)
```
</details>

<details><summary><code>kTile3G</code></summary>

```
/-- General key tile: key `j` (global), head lane `e`, at `base + j·skn + e·skk`. -/
```
```lean
noncomputable def kTile3G (s : BlockState) (K : RegionName)
    (base NC ND skn skk : Nat) : TileIndex [NC, ND] → ℝ :=
  fun (j, e, _) => s.readMem K (base + j.val * skn + e.val * skk)
```
</details>

<details><summary><code>vTile3G</code></summary>

```
/-- General value tile: key `j` (global), head lane `d`, at `base + j·svk + d·svn`. -/
```
```lean
noncomputable def vTile3G (s : BlockState) (V : RegionName)
    (base NC ND svk svn : Nat) : TileIndex [NC, ND] → ℝ :=
  fun (j, d, _) => s.readMem V (base + j.val * svk + d.val * svn)
```
</details>

<details><summary><code>keyScale3G</code></summary>

```
/-- General per-key uniform score scale (= `sm_scale · log2e`). -/
```
```lean
noncomputable def keyScale3G (sc : ℝ) (NC : Nat) : Fin NC → ℝ := fun _ => sc
```
</details>

<details><summary><code>aft3RunningMaxG</code></summary>

```
/-- General ⊥-seeded running max over the windowed prefix. -/
```
```lean
noncomputable def aft3RunningMaxG {BM ND NC : Nat}
    (qT : TileIndex [BM, ND] → ℝ) (kT vT : TileIndex [NC, ND] → ℝ)
    (keyScale : Fin NC → ℝ) (keep : Fin BM → Fin NC → Prop)
    [∀ i j, Decidable (keep i j)] (hi : Nat) (i : Fin BM) (d : Fin ND) :
    WithBot ℝ :=
  ((aft3KeysUptoG qT kT vT keyScale keep hi i d).map
    (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥
```
</details>

<details><summary><code>aft3StateBotKG</code></summary>

```
/-- General faithful kernel ⊥-carry state (seed-1 at window 0, seed-0 ⊥-state after). -/
```
```lean
noncomputable def aft3StateBotKG {BM ND NC : Nat}
    (qT : TileIndex [BM, ND] → ℝ) (kT vT : TileIndex [NC, ND] → ℝ)
    (keyScale : Fin NC → ℝ) (keep : Fin BM → Fin NC → Prop)
    [∀ i j, Decidable (keep i j)] (hi : Nat) (i : Fin BM) (d : Fin ND) :
    WithBot ℝ × ℝ × ℝ :=
  if hi = 0 then (⊥, 1, 0)
  else aft3StateBotG qT kT vT keyScale keep hi i d
```
</details>

<details><summary><code>natDist3G</code></summary>

```
/-- General faithful nat-truncated sliding-window distance, block-local key
`jL = j mod BN`, block start `start_n = (j / BN)·BN`:
`dist = (i − jL : ℕ) + SM·BM − start_n + offset`. -/
```
```lean
def natDist3G (SM BM BN off : Nat) {NC : Nat} (i : Fin BM) (j : Fin NC) : Nat :=
  (i.val - j.val % BN) + SM * BM - (j.val / BN) * BN + off
```
</details>

<details><summary><code>aft3KeysUptoG</code></summary>

```
/-- General windowed prefix key list `[0, hi)`. -/
```
```lean
noncomputable def aft3KeysUptoG {BM ND NC : Nat}
    (qT : TileIndex [BM, ND] → ℝ) (kT vT : TileIndex [NC, ND] → ℝ)
    (keyScale : Fin NC → ℝ) (keep : Fin BM → Fin NC → Prop)
    [∀ i j, Decidable (keep i j)] (hi : Nat) (i : Fin BM) (d : Fin ND) :
    List (ℝ × ℝ) :=
  (List.finRange NC).filterMap (fun j : Fin NC =>
    if j.val < hi ∧ keep i j then
      some (keyScale j * Finset.univ.sum (fun e : Fin ND =>
              qT (i, e, PUnit.unit) * kT (j, e, PUnit.unit)),
            vT (j, d, PUnit.unit))
    else none)
```
</details>

<details><summary><code>aft3StateBotG</code></summary>

```
/-- General ⊥-seeded running `(max, denom, acc)` after the windowed prefix `[0, hi)`. -/
```
```lean
noncomputable def aft3StateBotG {BM ND NC : Nat}
    (qT : TileIndex [BM, ND] → ℝ) (kT vT : TileIndex [NC, ND] → ℝ)
    (keyScale : Fin NC → ℝ) (keep : Fin BM → Fin NC → Prop)
    [∀ i j, Decidable (keep i j)] (hi : Nat) (i : Fin BM) (d : Fin ND) :
    WithBot ℝ × ℝ × ℝ :=
  (aft3KeysUptoG qT kT vT keyScale keep hi i d).foldl aft3OsStepBot (⊥, 0, 0)
```
</details>

<details><summary><code>aft3OsStepBot</code></summary>

```
/-- **Body split (case 1).** The lowered algorithm body of the case-1 surface is
exactly `aft3PreLoopG ++ Stmt.forRange "start_n" 0 128 64 aft3LoopBodyG ::
aft3PostLoopG`. -/
```
```lean
noncomputable def aft3OsStepBot (st : WithBot ℝ × ℝ × ℝ) (sv : ℝ × ℝ) : WithBot ℝ × ℝ × ℝ :=
  let m := st.1; let l := st.2.1; let acc := st.2.2
  let s := sv.1; let v := sv.2
  let m' := m ⊔ ((s : ℝ) : WithBot ℝ)
  let α := (WithBot.realExp2 (WithBot.realSub m m')).unbotD 0
  let p := pow2 (s - m'.unbotD 0)
  (m', l * α + p, acc * α + p * v)
```
</details>

## Public theorem: `attention_fwd_triton3_python_case2_output_summary_general`

<details><summary>docstring</summary>

```
/-- **Case 2 general genuine output summary.** -/
```
</details>

**Statement:**
```lean
specification attention_fwd_triton3_python_case2_output_summary_general
    (Q K V M Out L : RegionName) (sm_scale : ℝ)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      Z H H_KV N_CTX ROUND_CTX NKV_CTX off size BM ND BN : Nat) (s : BlockState)
    (hND : 0 < ND) (hBM : 0 < BM) (hBN : 0 < BN) (hNC : 0 < NKV_CTX) (hBNdvd : BN ∣ NKV_CTX)
    (hH : 0 < H) (hHKV : H_KV = H)
    (hskz : skz = sqz) (hskh : skh = sqh) (hsvz : svz = sqz) (hsvh : svh = sqh)
    (hsoz : soz = sqz) (hsoh : soh = sqh)
    (hMO : M ≠ Out) (hundef : ∀ rg o, s.undef rg o = 0)
    (hinjO : Function.Injective
      (fun idx : TileIndex [BM, ND] => (s.pids 1 / H * sqz + s.pids 1 % H * sqh) + (s.pids 0 * BM + idx.1.val) * som + idx.2.1.val * son))
    (hinjM : Function.Injective
      (fun r : TileIndex [BM] => s.pids 1 * ROUND_CTX + (s.pids 0 * BM + r.1.val))) :
    (∃ alg, (attention_fwd_triton3_surface Q K V M Out L sm_scale
      sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      Z H H_KV N_CTX ROUND_CTX NKV_CTX off size 1 1 BM ND BN 1 1 1 1).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := attention_fwd_triton3_surface Q K V M Out L sm_scale
        sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
        Z H H_KV N_CTX ROUND_CTX NKV_CTX off size 1 1 BM ND BN 1 1 1 1)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BM, ND] => active s N_CTX ND BM idx)
        (fun idx : TileIndex [BM, ND] => (Out, outOffset s H sqz sqh som son BM idx)))
      (expected := fun idx : TileIndex [BM, ND] =>
        attentionFwdTriton3Case2OutSpecG s Q K V (s.pids 1 / H * sqz + s.pids 1 % H * sqh) BM ND NKV_CTX sqm sqk skn skk svk svn (sm_scale * 1.4426950408889634) BN off size idx) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := attention_fwd_triton3_surface Q K V M Out L sm_scale
        sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
        Z H H_KV N_CTX ROUND_CTX NKV_CTX off size 1 1 BM ND BN 1 1 1 1)
      (initialState := s)
      (write := fun i : Fin BM => some (M, lRowOffset s (s.pids 1) ROUND_CTX BM i))
      (expected := fun i : Fin BM =>
        attentionFwdTriton3KMSpecG s Q K V (s.pids 1 / H * sqz + s.pids 1 % H * sqh) BM ND NKV_CTX sqm sqk skn skk svk svn (sm_scale * 1.4426950408889634) (fun i j => natComplementSlidingWindowKeepG (s.pids 0) BM BN off size i j) i hND)
```

**Assumptions / layout contracts:**
- `hND : 0 < ND`
- `hBM : 0 < BM`
- `hBN : 0 < BN`
- `hNC : 0 < NKV_CTX`
- `hBNdvd : BN ∣ NKV_CTX`
- `hH : 0 < H`
- `hHKV : H_KV = H`
- `hskz : skz = sqz`
- `hskh : skh = sqh`
- `hsvz : svz = sqz`
- `hsvh : svh = sqh`
- `hsoz : soz = sqz`
- `hsoh : soh = sqh`
- `hMO : M ≠ Out`
- `hundef : ∀ rg o, s.undef rg o = 0`
- `fun idx : TileIndex [BM, ND] => (s.pids 1 / H * sqz + s.pids 1 % H * sqh) + (s.pids 0 * BM + idx.1.val) * som + idx.2.1.val * son`
- `fun r : TileIndex [BM] => s.pids 1 * ROUND_CTX + (s.pids 0 * BM + r.1.val)`
- `fun idx : TileIndex [BM, ND] => active s N_CTX ND BM idx`
- `fun idx : TileIndex [BM, ND] => (Out, outOffset s H sqz sqh som son BM idx)`

**Closed-form spec defs (transitive):** `attention_fwd_triton3_surface`, `active`, `outOffset`, `attentionFwdTriton3Case2OutSpecG`, `lRowOffset`, `attentionFwdTriton3KMSpecG`, `natComplementSlidingWindowKeepG`, `mIndex`, `kIndex`, `offZ`, `offH`, `qTile3G`, `kTile3G`, `vTile3G`, `keyScale3G`, `aft3RunningMaxG`, `aft3StateBotKG`, `natDist3G`, `aft3KeysUptoG`, `aft3StateBotG`, `aft3OsStepBot`

<details><summary><code>attention_fwd_triton3_surface</code></summary>

```
/-- Full Lean port of `attention_fwd_triton3.py`'s `_attn_fwd`. -/
```
```lean
def attention_fwd_triton3_surface
    (Q K V M Out L : RegionName) (sm_scale : ℝ)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      _Z H H_KV N_CTX ROUND_CTX NKV_CTX
      _sliding_window_offset _sliding_window_size
      IS_EVEN_M _IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N END INIT
      _SLIDING_WINDOW _COMPLEMENT_SLIDING_WINDOW : Nat) :
    ComputeKernel := triton {
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
  qk_scale = $(sm_scale) * 1.0
  qk_scale *= 1.4426950408889634
  if $(IS_EVEN_M) != $(0) {
    q = tl.load(Q_block_ptr)
  } else {
    q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
  }
  for start_n in range($(0), $(NKV_CTX), $(BLOCK_N)) {
    k = tl.load(K_block_ptr)
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    qk += tl.dot(q, k)
    qk = qk * qk_scale
    if $(_SLIDING_WINDOW) != $(0) {
      dist = tl.arange(0, $(BLOCK_M))[:, None] - tl.arange(0, $(BLOCK_N))[None, :]
        + start_m * $(BLOCK_M) - start_n + $(_sliding_window_offset)
      if $(_COMPLEMENT_SLIDING_WINDOW) != $(0) {
        mask = dist >= $(_sliding_window_size)
      } else {
        mask = (dist >= $(0)) & (dist < $(_sliding_window_size))
      }
      qk = tl.where(mask, qk, float("-inf"))
    }
    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    qk = qk - m_ij[:, None]
    p = tl.math.exp2(qk)
    if $(_SLIDING_WINDOW) != $(0) {
      p = tl.where(mask, p, 0.0)
    }
    l_ij = tl.sum(p, 1)
    tmp = m_i - m_ij
    alpha = tl.math.exp2(tmp)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]
    v = tl.load(V_block_ptr)
    acc += tl.dot(p, v)
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
  tl.store(O_block_ptr, (acc).to(Out.type.element_ty))
}
```
</details>

<details><summary><code>active</code></summary>

```lean
def active
    (s : BlockState) (N_CTX HEAD_ACTIVE BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Prop :=
  mIndex s BLOCK_M idx.1 < N_CTX ∧ kIndex idx < HEAD_ACTIVE
```
</details>

<details><summary><code>outOffset</code></summary>

```lean
def outOffset
    (s : BlockState)
    (H stride_qz stride_qh stride_qm stride_qk BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  offZ s H * stride_qz + offH s H * stride_qh +
    mIndex s BLOCK_M idx.1 * stride_qm + kIndex idx * stride_qk
```
</details>

<details><summary><code>attentionFwdTriton3Case2OutSpecG</code></summary>

```
/-- General genuine closed form, case 2 (complement sliding window). -/
```
```lean
noncomputable def attentionFwdTriton3Case2OutSpecG
    (s : BlockState) (Q K V : RegionName)
    (base BM ND NC sqm sqk skn skk svk svn : Nat) (sc : ℝ) (BN off size : Nat)
    (idx : TileIndex [BM, ND]) : ℝ :=
  attentionRealBase2PerKeyScalePred (qTile3G s Q base BM ND sqm sqk)
    (kTile3G s K base NC ND skn skk) (vTile3G s V base NC ND svk svn)
    (keyScale3G sc NC) (fun i j => natComplementSlidingWindowKeepG (s.pids 0) BM BN off size i j) idx
```
</details>

<details><summary><code>lRowOffset</code></summary>

```
/-- Proof-oriented L (log-sum-exp) row store slice of `attention_fwd_triton3.py`.
Takes a precomputed `LPre` vector and proves the row writeback into `L` at
offset `off_hz * ROUND_CTX + offs_m`. -/
```
```lean
def lRowOffset (s : BlockState) (off_hz ROUND_CTX BLOCK_M : Nat)
    (i : Fin BLOCK_M) : Nat :=
  off_hz * ROUND_CTX + (s.pids 0 * BLOCK_M + i.val)
```
</details>

<details><summary><code>attentionFwdTriton3KMSpecG</code></summary>

```
/-- General genuine `M`-row spec (cases 1/2): raw `(M ⊔ … + log2 l).unbotD`. -/
```
```lean
noncomputable def attentionFwdTriton3KMSpecG
    (s : BlockState) (Q K V : RegionName)
    (base BM ND NC sqm sqk skn skk svk svn : Nat) (sc : ℝ)
    (keep : Fin BM → Fin NC → Prop) [∀ i j, Decidable (keep i j)] (i : Fin BM) (hND : 0 < ND) : ℝ :=
  (WithBot.realAdd
      (aft3RunningMaxG (qTile3G s Q base BM ND sqm sqk) (kTile3G s K base NC ND skn skk)
        (vTile3G s V base NC ND svk svn) (keyScale3G sc NC) keep NC i ⟨0, hND⟩)
      (WithBot.realLog2 (((aft3StateBotKG (qTile3G s Q base BM ND sqm sqk) (kTile3G s K base NC ND skn skk)
        (vTile3G s V base NC ND svk svn) (keyScale3G sc NC) keep NC i ⟨0, hND⟩).2.1 : ℝ) : WithBot ℝ))).unbotD 0
```
</details>

<details><summary><code>natComplementSlidingWindowKeepG</code></summary>

```
/-- General case-2 complement keep predicate: `size ≤ dist`. -/
```
```lean
def natComplementSlidingWindowKeepG (SM BM BN off size : Nat) {NC : Nat}
    (i : Fin BM) (j : Fin NC) : Prop :=
  size ≤ natDist3G SM BM BN off i j
```
</details>

<details><summary><code>mIndex</code></summary>

```lean
def mIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val
```
</details>

<details><summary><code>kIndex</code></summary>

```lean
def kIndex (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  idx.2.1.val
```
</details>

<details><summary><code>offZ</code></summary>

```
/-- Surface transcription/proof-oriented final output-store slice of `attention_fwd_triton3.py`'s
`_attn_fwd`.

The full kernel runs separate streaming attention stages, including the causal
stage when requested. This slice starts after those stages have produced a
precomputed normalized `Acc` tile and proves the final masked writeback into
`Out`, preserving the source store address and mask
`(offs_m < N_CTX) & (offs_k < HEAD_ACTIVE)`. The inner `tl.float32` accumulator is
outside this slice. -/
```
```lean
def offZ (s : BlockState) (H : Nat) : Nat :=
  s.pids 1 / H
```
</details>

<details><summary><code>offH</code></summary>

```lean
def offH (s : BlockState) (H : Nat) : Nat :=
  s.pids 1 % H
```
</details>

<details><summary><code>qTile3G</code></summary>

```
/-- General query tile: query row `i`, head lane `e`, at
`base + (pid0·BM + i)·sqm + e·sqk`. -/
```
```lean
noncomputable def qTile3G (s : BlockState) (Q : RegionName)
    (base BM ND sqm sqk : Nat) : TileIndex [BM, ND] → ℝ :=
  fun (i, e, _) => s.readMem Q (base + (s.pids 0 * BM + i.val) * sqm + e.val * sqk)
```
</details>

<details><summary><code>kTile3G</code></summary>

```
/-- General key tile: key `j` (global), head lane `e`, at `base + j·skn + e·skk`. -/
```
```lean
noncomputable def kTile3G (s : BlockState) (K : RegionName)
    (base NC ND skn skk : Nat) : TileIndex [NC, ND] → ℝ :=
  fun (j, e, _) => s.readMem K (base + j.val * skn + e.val * skk)
```
</details>

<details><summary><code>vTile3G</code></summary>

```
/-- General value tile: key `j` (global), head lane `d`, at `base + j·svk + d·svn`. -/
```
```lean
noncomputable def vTile3G (s : BlockState) (V : RegionName)
    (base NC ND svk svn : Nat) : TileIndex [NC, ND] → ℝ :=
  fun (j, d, _) => s.readMem V (base + j.val * svk + d.val * svn)
```
</details>

<details><summary><code>keyScale3G</code></summary>

```
/-- General per-key uniform score scale (= `sm_scale · log2e`). -/
```
```lean
noncomputable def keyScale3G (sc : ℝ) (NC : Nat) : Fin NC → ℝ := fun _ => sc
```
</details>

<details><summary><code>aft3RunningMaxG</code></summary>

```
/-- General ⊥-seeded running max over the windowed prefix. -/
```
```lean
noncomputable def aft3RunningMaxG {BM ND NC : Nat}
    (qT : TileIndex [BM, ND] → ℝ) (kT vT : TileIndex [NC, ND] → ℝ)
    (keyScale : Fin NC → ℝ) (keep : Fin BM → Fin NC → Prop)
    [∀ i j, Decidable (keep i j)] (hi : Nat) (i : Fin BM) (d : Fin ND) :
    WithBot ℝ :=
  ((aft3KeysUptoG qT kT vT keyScale keep hi i d).map
    (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥
```
</details>

<details><summary><code>aft3StateBotKG</code></summary>

```
/-- General faithful kernel ⊥-carry state (seed-1 at window 0, seed-0 ⊥-state after). -/
```
```lean
noncomputable def aft3StateBotKG {BM ND NC : Nat}
    (qT : TileIndex [BM, ND] → ℝ) (kT vT : TileIndex [NC, ND] → ℝ)
    (keyScale : Fin NC → ℝ) (keep : Fin BM → Fin NC → Prop)
    [∀ i j, Decidable (keep i j)] (hi : Nat) (i : Fin BM) (d : Fin ND) :
    WithBot ℝ × ℝ × ℝ :=
  if hi = 0 then (⊥, 1, 0)
  else aft3StateBotG qT kT vT keyScale keep hi i d
```
</details>

<details><summary><code>natDist3G</code></summary>

```
/-- General faithful nat-truncated sliding-window distance, block-local key
`jL = j mod BN`, block start `start_n = (j / BN)·BN`:
`dist = (i − jL : ℕ) + SM·BM − start_n + offset`. -/
```
```lean
def natDist3G (SM BM BN off : Nat) {NC : Nat} (i : Fin BM) (j : Fin NC) : Nat :=
  (i.val - j.val % BN) + SM * BM - (j.val / BN) * BN + off
```
</details>

<details><summary><code>aft3KeysUptoG</code></summary>

```
/-- General windowed prefix key list `[0, hi)`. -/
```
```lean
noncomputable def aft3KeysUptoG {BM ND NC : Nat}
    (qT : TileIndex [BM, ND] → ℝ) (kT vT : TileIndex [NC, ND] → ℝ)
    (keyScale : Fin NC → ℝ) (keep : Fin BM → Fin NC → Prop)
    [∀ i j, Decidable (keep i j)] (hi : Nat) (i : Fin BM) (d : Fin ND) :
    List (ℝ × ℝ) :=
  (List.finRange NC).filterMap (fun j : Fin NC =>
    if j.val < hi ∧ keep i j then
      some (keyScale j * Finset.univ.sum (fun e : Fin ND =>
              qT (i, e, PUnit.unit) * kT (j, e, PUnit.unit)),
            vT (j, d, PUnit.unit))
    else none)
```
</details>

<details><summary><code>aft3StateBotG</code></summary>

```
/-- General ⊥-seeded running `(max, denom, acc)` after the windowed prefix `[0, hi)`. -/
```
```lean
noncomputable def aft3StateBotG {BM ND NC : Nat}
    (qT : TileIndex [BM, ND] → ℝ) (kT vT : TileIndex [NC, ND] → ℝ)
    (keyScale : Fin NC → ℝ) (keep : Fin BM → Fin NC → Prop)
    [∀ i j, Decidable (keep i j)] (hi : Nat) (i : Fin BM) (d : Fin ND) :
    WithBot ℝ × ℝ × ℝ :=
  (aft3KeysUptoG qT kT vT keyScale keep hi i d).foldl aft3OsStepBot (⊥, 0, 0)
```
</details>

<details><summary><code>aft3OsStepBot</code></summary>

```
/-- **Body split (case 1).** The lowered algorithm body of the case-1 surface is
exactly `aft3PreLoopG ++ Stmt.forRange "start_n" 0 128 64 aft3LoopBodyG ::
aft3PostLoopG`. -/
```
```lean
noncomputable def aft3OsStepBot (st : WithBot ℝ × ℝ × ℝ) (sv : ℝ × ℝ) : WithBot ℝ × ℝ × ℝ :=
  let m := st.1; let l := st.2.1; let acc := st.2.2
  let s := sv.1; let v := sv.2
  let m' := m ⊔ ((s : ℝ) : WithBot ℝ)
  let α := (WithBot.realExp2 (WithBot.realSub m m')).unbotD 0
  let p := pow2 (s - m'.unbotD 0)
  (m', l * α + p, acc * α + p * v)
```
</details>

## Public theorem: `attention_fwd_triton3_python_case3_output_summary_general`

<details><summary>docstring</summary>

```
/-- **Case 3 general genuine output summary.** -/
```
</details>

**Statement:**
```lean
specification attention_fwd_triton3_python_case3_output_summary_general
    (Q K V M Out L : RegionName) (sm_scale : ℝ)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      Z H H_KV N_CTX ROUND_CTX NKV_CTX off BM ND BN : Nat) (s : BlockState)
    (hND : 0 < ND) (hBM : 0 < BM) (hBN : 0 < BN) (hNC : 0 < NKV_CTX) (hBNdvd : BN ∣ NKV_CTX)
    (hH : 0 < H) (hHKV : H_KV = H)
    (hskz : skz = sqz) (hskh : skh = sqh) (hsvz : svz = sqz) (hsvh : svh = sqh)
    (hsoz : soz = sqz) (hsoh : soh = sqh)
    (hMO : M ≠ Out) (hundef : ∀ rg o, s.undef rg o = 0)
    (hinjO : Function.Injective
      (fun idx : TileIndex [BM, ND] => (s.pids 1 / H * sqz + s.pids 1 % H * sqh) + (s.pids 0 * BM + idx.1.val) * som + idx.2.1.val * son))
    (hinjM : Function.Injective
      (fun r : TileIndex [BM] => s.pids 1 * ROUND_CTX + (s.pids 0 * BM + r.1.val))) :
    (∃ alg, (attention_fwd_triton3_surface Q K V M Out L sm_scale
      sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      Z H H_KV N_CTX ROUND_CTX NKV_CTX off 0 1 1 BM ND BN 1 1 0 0).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := attention_fwd_triton3_surface Q K V M Out L sm_scale
        sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
        Z H H_KV N_CTX ROUND_CTX NKV_CTX off 0 1 1 BM ND BN 1 1 0 0)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BM, ND] => active s N_CTX ND BM idx)
        (fun idx : TileIndex [BM, ND] => (Out, outOffset s H sqz sqh som son BM idx)))
      (expected := fun idx : TileIndex [BM, ND] =>
        attentionFwdTriton3Case3OutSpecG s Q K V (s.pids 1 / H * sqz + s.pids 1 % H * sqh) BM ND NKV_CTX sqm sqk skn skk svk svn (sm_scale * 1.4426950408889634) idx) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := attention_fwd_triton3_surface Q K V M Out L sm_scale
        sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
        Z H H_KV N_CTX ROUND_CTX NKV_CTX off 0 1 1 BM ND BN 1 1 0 0)
      (initialState := s)
      (write := fun i : Fin BM => some (M, lRowOffset s (s.pids 1) ROUND_CTX BM i))
      (expected := fun i : Fin BM =>
        attentionFwdTriton3Case3MSpecG s Q K V (s.pids 1 / H * sqz + s.pids 1 % H * sqh) BM ND NKV_CTX sqm sqk skn skk svk svn (sm_scale * 1.4426950408889634) i hND)
```

**Assumptions / layout contracts:**
- `hND : 0 < ND`
- `hBM : 0 < BM`
- `hBN : 0 < BN`
- `hNC : 0 < NKV_CTX`
- `hBNdvd : BN ∣ NKV_CTX`
- `hH : 0 < H`
- `hHKV : H_KV = H`
- `hskz : skz = sqz`
- `hskh : skh = sqh`
- `hsvz : svz = sqz`
- `hsvh : svh = sqh`
- `hsoz : soz = sqz`
- `hsoh : soh = sqh`
- `hMO : M ≠ Out`
- `hundef : ∀ rg o, s.undef rg o = 0`
- `fun idx : TileIndex [BM, ND] => (s.pids 1 / H * sqz + s.pids 1 % H * sqh) + (s.pids 0 * BM + idx.1.val) * som + idx.2.1.val * son`
- `fun r : TileIndex [BM] => s.pids 1 * ROUND_CTX + (s.pids 0 * BM + r.1.val)`
- `fun idx : TileIndex [BM, ND] => active s N_CTX ND BM idx`
- `fun idx : TileIndex [BM, ND] => (Out, outOffset s H sqz sqh som son BM idx)`

**Closed-form spec defs (transitive):** `attention_fwd_triton3_surface`, `active`, `outOffset`, `attentionFwdTriton3Case3OutSpecG`, `lRowOffset`, `attentionFwdTriton3Case3MSpecG`, `mIndex`, `kIndex`, `offZ`, `offH`, `qTile3G`, `kTile3G`, `vTile3G`, `keyScale3G`, `aft3RunningMaxG`, `aft3StateBot1G`, `aft3KeysUptoG`, `aft3OsStepBot`

<details><summary><code>attention_fwd_triton3_surface</code></summary>

```
/-- Full Lean port of `attention_fwd_triton3.py`'s `_attn_fwd`. -/
```
```lean
def attention_fwd_triton3_surface
    (Q K V M Out L : RegionName) (sm_scale : ℝ)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      _Z H H_KV N_CTX ROUND_CTX NKV_CTX
      _sliding_window_offset _sliding_window_size
      IS_EVEN_M _IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N END INIT
      _SLIDING_WINDOW _COMPLEMENT_SLIDING_WINDOW : Nat) :
    ComputeKernel := triton {
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
  qk_scale = $(sm_scale) * 1.0
  qk_scale *= 1.4426950408889634
  if $(IS_EVEN_M) != $(0) {
    q = tl.load(Q_block_ptr)
  } else {
    q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
  }
  for start_n in range($(0), $(NKV_CTX), $(BLOCK_N)) {
    k = tl.load(K_block_ptr)
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    qk += tl.dot(q, k)
    qk = qk * qk_scale
    if $(_SLIDING_WINDOW) != $(0) {
      dist = tl.arange(0, $(BLOCK_M))[:, None] - tl.arange(0, $(BLOCK_N))[None, :]
        + start_m * $(BLOCK_M) - start_n + $(_sliding_window_offset)
      if $(_COMPLEMENT_SLIDING_WINDOW) != $(0) {
        mask = dist >= $(_sliding_window_size)
      } else {
        mask = (dist >= $(0)) & (dist < $(_sliding_window_size))
      }
      qk = tl.where(mask, qk, float("-inf"))
    }
    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    qk = qk - m_ij[:, None]
    p = tl.math.exp2(qk)
    if $(_SLIDING_WINDOW) != $(0) {
      p = tl.where(mask, p, 0.0)
    }
    l_ij = tl.sum(p, 1)
    tmp = m_i - m_ij
    alpha = tl.math.exp2(tmp)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]
    v = tl.load(V_block_ptr)
    acc += tl.dot(p, v)
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
  tl.store(O_block_ptr, (acc).to(Out.type.element_ty))
}
```
</details>

<details><summary><code>active</code></summary>

```lean
def active
    (s : BlockState) (N_CTX HEAD_ACTIVE BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Prop :=
  mIndex s BLOCK_M idx.1 < N_CTX ∧ kIndex idx < HEAD_ACTIVE
```
</details>

<details><summary><code>outOffset</code></summary>

```lean
def outOffset
    (s : BlockState)
    (H stride_qz stride_qh stride_qm stride_qk BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  offZ s H * stride_qz + offH s H * stride_qh +
    mIndex s BLOCK_M idx.1 * stride_qm + kIndex idx * stride_qk
```
</details>

<details><summary><code>attentionFwdTriton3Case3OutSpecG</code></summary>

```
/-- General genuine closed form, case 3 (no window) — plain base-2 softmax. -/
```
```lean
noncomputable def attentionFwdTriton3Case3OutSpecG
    (s : BlockState) (Q K V : RegionName)
    (base BM ND NC sqm sqk skn skk svk svn : Nat) (sc : ℝ)
    (idx : TileIndex [BM, ND]) : ℝ :=
  attentionRealBase2PerKeyScalePred (qTile3G s Q base BM ND sqm sqk)
    (kTile3G s K base NC ND skn skk) (vTile3G s V base NC ND svk svn)
    (keyScale3G sc NC) (fun (i : Fin BM) (j : Fin NC) => noWindowKeep i j) idx
```
</details>

<details><summary><code>lRowOffset</code></summary>

```
/-- Proof-oriented L (log-sum-exp) row store slice of `attention_fwd_triton3.py`.
Takes a precomputed `LPre` vector and proves the row writeback into `L` at
offset `off_hz * ROUND_CTX + offs_m`. -/
```
```lean
def lRowOffset (s : BlockState) (off_hz ROUND_CTX BLOCK_M : Nat)
    (i : Fin BLOCK_M) : Nat :=
  off_hz * ROUND_CTX + (s.pids 0 * BLOCK_M + i.val)
```
</details>

<details><summary><code>attentionFwdTriton3Case3MSpecG</code></summary>

```
/-- General genuine `M`-row spec (case 3): `m_i + log2 l_i` finalize. -/
```
```lean
noncomputable def attentionFwdTriton3Case3MSpecG
    (s : BlockState) (Q K V : RegionName)
    (base BM ND NC sqm sqk skn skk svk svn : Nat) (sc : ℝ) (i : Fin BM) (hND : 0 < ND) : ℝ :=
  (aft3RunningMaxG (qTile3G s Q base BM ND sqm sqk) (kTile3G s K base NC ND skn skk)
      (vTile3G s V base NC ND svk svn) (keyScale3G sc NC) (fun i j => noWindowKeep i j) NC i ⟨0, hND⟩).unbotD 0
    + Real.log
      ((aft3StateBot1G (qTile3G s Q base BM ND sqm sqk) (kTile3G s K base NC ND skn skk)
          (vTile3G s V base NC ND svk svn) (keyScale3G sc NC) (fun i j => noWindowKeep i j) NC i ⟨0, hND⟩).2.1) / Real.log 2
```
</details>

<details><summary><code>mIndex</code></summary>

```lean
def mIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val
```
</details>

<details><summary><code>kIndex</code></summary>

```lean
def kIndex (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  idx.2.1.val
```
</details>

<details><summary><code>offZ</code></summary>

```
/-- Surface transcription/proof-oriented final output-store slice of `attention_fwd_triton3.py`'s
`_attn_fwd`.

The full kernel runs separate streaming attention stages, including the causal
stage when requested. This slice starts after those stages have produced a
precomputed normalized `Acc` tile and proves the final masked writeback into
`Out`, preserving the source store address and mask
`(offs_m < N_CTX) & (offs_k < HEAD_ACTIVE)`. The inner `tl.float32` accumulator is
outside this slice. -/
```
```lean
def offZ (s : BlockState) (H : Nat) : Nat :=
  s.pids 1 / H
```
</details>

<details><summary><code>offH</code></summary>

```lean
def offH (s : BlockState) (H : Nat) : Nat :=
  s.pids 1 % H
```
</details>

<details><summary><code>qTile3G</code></summary>

```
/-- General query tile: query row `i`, head lane `e`, at
`base + (pid0·BM + i)·sqm + e·sqk`. -/
```
```lean
noncomputable def qTile3G (s : BlockState) (Q : RegionName)
    (base BM ND sqm sqk : Nat) : TileIndex [BM, ND] → ℝ :=
  fun (i, e, _) => s.readMem Q (base + (s.pids 0 * BM + i.val) * sqm + e.val * sqk)
```
</details>

<details><summary><code>kTile3G</code></summary>

```
/-- General key tile: key `j` (global), head lane `e`, at `base + j·skn + e·skk`. -/
```
```lean
noncomputable def kTile3G (s : BlockState) (K : RegionName)
    (base NC ND skn skk : Nat) : TileIndex [NC, ND] → ℝ :=
  fun (j, e, _) => s.readMem K (base + j.val * skn + e.val * skk)
```
</details>

<details><summary><code>vTile3G</code></summary>

```
/-- General value tile: key `j` (global), head lane `d`, at `base + j·svk + d·svn`. -/
```
```lean
noncomputable def vTile3G (s : BlockState) (V : RegionName)
    (base NC ND svk svn : Nat) : TileIndex [NC, ND] → ℝ :=
  fun (j, d, _) => s.readMem V (base + j.val * svk + d.val * svn)
```
</details>

<details><summary><code>keyScale3G</code></summary>

```
/-- General per-key uniform score scale (= `sm_scale · log2e`). -/
```
```lean
noncomputable def keyScale3G (sc : ℝ) (NC : Nat) : Fin NC → ℝ := fun _ => sc
```
</details>

<details><summary><code>aft3RunningMaxG</code></summary>

```
/-- General ⊥-seeded running max over the windowed prefix. -/
```
```lean
noncomputable def aft3RunningMaxG {BM ND NC : Nat}
    (qT : TileIndex [BM, ND] → ℝ) (kT vT : TileIndex [NC, ND] → ℝ)
    (keyScale : Fin NC → ℝ) (keep : Fin BM → Fin NC → Prop)
    [∀ i j, Decidable (keep i j)] (hi : Nat) (i : Fin BM) (d : Fin ND) :
    WithBot ℝ :=
  ((aft3KeysUptoG qT kT vT keyScale keep hi i d).map
    (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥
```
</details>

<details><summary><code>aft3StateBot1G</code></summary>

```
/-- General ⊥-seeded running state from the kernel's `l_i = 1` seed. -/
```
```lean
noncomputable def aft3StateBot1G {BM ND NC : Nat}
    (qT : TileIndex [BM, ND] → ℝ) (kT vT : TileIndex [NC, ND] → ℝ)
    (keyScale : Fin NC → ℝ) (keep : Fin BM → Fin NC → Prop)
    [∀ i j, Decidable (keep i j)] (hi : Nat) (i : Fin BM) (d : Fin ND) :
    WithBot ℝ × ℝ × ℝ :=
  (aft3KeysUptoG qT kT vT keyScale keep hi i d).foldl aft3OsStepBot (⊥, 1, 0)
```
</details>

<details><summary><code>aft3KeysUptoG</code></summary>

```
/-- General windowed prefix key list `[0, hi)`. -/
```
```lean
noncomputable def aft3KeysUptoG {BM ND NC : Nat}
    (qT : TileIndex [BM, ND] → ℝ) (kT vT : TileIndex [NC, ND] → ℝ)
    (keyScale : Fin NC → ℝ) (keep : Fin BM → Fin NC → Prop)
    [∀ i j, Decidable (keep i j)] (hi : Nat) (i : Fin BM) (d : Fin ND) :
    List (ℝ × ℝ) :=
  (List.finRange NC).filterMap (fun j : Fin NC =>
    if j.val < hi ∧ keep i j then
      some (keyScale j * Finset.univ.sum (fun e : Fin ND =>
              qT (i, e, PUnit.unit) * kT (j, e, PUnit.unit)),
            vT (j, d, PUnit.unit))
    else none)
```
</details>

<details><summary><code>aft3OsStepBot</code></summary>

```
/-- **Body split (case 1).** The lowered algorithm body of the case-1 surface is
exactly `aft3PreLoopG ++ Stmt.forRange "start_n" 0 128 64 aft3LoopBodyG ::
aft3PostLoopG`. -/
```
```lean
noncomputable def aft3OsStepBot (st : WithBot ℝ × ℝ × ℝ) (sv : ℝ × ℝ) : WithBot ℝ × ℝ × ℝ :=
  let m := st.1; let l := st.2.1; let acc := st.2.2
  let s := sv.1; let v := sv.2
  let m' := m ⊔ ((s : ℝ) : WithBot ℝ)
  let α := (WithBot.realExp2 (WithBot.realSub m m')).unbotD 0
  let p := pow2 (s - m'.unbotD 0)
  (m', l * α + p, acc * α + p * v)
```
</details>

## Public theorem: `attention_fwd_triton3_python_case4_output_summary_general`

<details><summary>docstring</summary>

```
/-- **Dimension-general genuine `output_summary`, case 4** (`INIT=False` cross-launch
resume + sliding window). The executed surface writes the GENUINE
`attentionFwdTriton3Case4OutSpecG` (normalized resume-seeded sliding-window online
softmax, read over INPUT `Q`/`K`/`V` and the resume buffers `M`/`L`/`Out` — **no
self-reference** to this program's own executed output) into `O`, and the `m+log2 l`
finalize into `M`. The resume seed is read from `M`/`L`/`Out` at the initial state. -/
```
</details>

**Statement:**
```lean
specification attention_fwd_triton3_python_case4_output_summary_general
    (Q K V M Out L : RegionName) (sm_scale : ℝ)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      Z H H_KV N_CTX ROUND_CTX NKV_CTX off size BM ND BN : Nat) (s : BlockState)
    (hND : 0 < ND) (hBM : 0 < BM) (hBN : 0 < BN) (hNC : 0 < NKV_CTX) (hBNdvd : BN ∣ NKV_CTX)
    (hH : 0 < H) (hHKV : H_KV = H)
    (hskz : skz = sqz) (hskh : skh = sqh) (hsvz : svz = sqz) (hsvh : svh = sqh)
    (hsoz : soz = sqz) (hsoh : soh = sqh)
    (hMO : M ≠ Out) (hundef : ∀ rg o, s.undef rg o = 0)
    (hinjO : Function.Injective
      (fun idx : TileIndex [BM, ND] => (s.pids 1 / H * sqz + s.pids 1 % H * sqh) + (s.pids 0 * BM + idx.1.val) * som + idx.2.1.val * son))
    (hinjM : Function.Injective
      (fun r : TileIndex [BM] => s.pids 1 * ROUND_CTX + (s.pids 0 * BM + r.1.val))) :
    (∃ alg, (attention_fwd_triton3_surface Q K V M Out L sm_scale
      sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      Z H H_KV N_CTX ROUND_CTX NKV_CTX off size 1 1 BM ND BN 1 0 1 0).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := attention_fwd_triton3_surface Q K V M Out L sm_scale
        sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
        Z H H_KV N_CTX ROUND_CTX NKV_CTX off size 1 1 BM ND BN 1 0 1 0)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BM, ND] => active s N_CTX ND BM idx)
        (fun idx : TileIndex [BM, ND] => (Out, outOffset s H sqz sqh som son BM idx)))
      (expected := fun idx : TileIndex [BM, ND] =>
        attentionFwdTriton3Case4OutSpecG s Q K V M Out L (s.pids 1 / H * sqz + s.pids 1 % H * sqh) BM ND NKV_CTX sqm sqk skn skk svk svn som son ROUND_CTX (sm_scale * 1.4426950408889634) BN off size hND idx) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := attention_fwd_triton3_surface Q K V M Out L sm_scale
        sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
        Z H H_KV N_CTX ROUND_CTX NKV_CTX off size 1 1 BM ND BN 1 0 1 0)
      (initialState := s)
      (write := fun i : Fin BM => some (M, lRowOffset s (s.pids 1) ROUND_CTX BM i))
      (expected := fun i : Fin BM =>
        attentionFwdTriton3Case4MSpecG s Q K V M Out L (s.pids 1 / H * sqz + s.pids 1 % H * sqh) BM ND NKV_CTX sqm sqk skn skk svk svn som son ROUND_CTX (sm_scale * 1.4426950408889634) BN off size i hND)
```

**Assumptions / layout contracts:**
- `hND : 0 < ND`
- `hBM : 0 < BM`
- `hBN : 0 < BN`
- `hNC : 0 < NKV_CTX`
- `hBNdvd : BN ∣ NKV_CTX`
- `hH : 0 < H`
- `hHKV : H_KV = H`
- `hskz : skz = sqz`
- `hskh : skh = sqh`
- `hsvz : svz = sqz`
- `hsvh : svh = sqh`
- `hsoz : soz = sqz`
- `hsoh : soh = sqh`
- `hMO : M ≠ Out`
- `hundef : ∀ rg o, s.undef rg o = 0`
- `fun idx : TileIndex [BM, ND] => (s.pids 1 / H * sqz + s.pids 1 % H * sqh) + (s.pids 0 * BM + idx.1.val) * som + idx.2.1.val * son`
- `fun r : TileIndex [BM] => s.pids 1 * ROUND_CTX + (s.pids 0 * BM + r.1.val)`
- `fun idx : TileIndex [BM, ND] => active s N_CTX ND BM idx`
- `fun idx : TileIndex [BM, ND] => (Out, outOffset s H sqz sqh som son BM idx)`

**Closed-form spec defs (transitive):** `attention_fwd_triton3_surface`, `active`, `outOffset`, `attentionFwdTriton3Case4OutSpecG`, `lRowOffset`, `attentionFwdTriton3Case4MSpecG`, `mIndex`, `kIndex`, `offZ`, `offH`, `aft3Case4Seed`, `natSlidingWindowKeepG`, `aft3StateSeededG`, `qTile3G`, `kTile3G`, `vTile3G`, `keyScale3G`, `mlRow3G`, `outLane3G`, `natDist3G`, `aft3KeysUptoG`, `aft3OsStepBot`

<details><summary><code>attention_fwd_triton3_surface</code></summary>

```
/-- Full Lean port of `attention_fwd_triton3.py`'s `_attn_fwd`. -/
```
```lean
def attention_fwd_triton3_surface
    (Q K V M Out L : RegionName) (sm_scale : ℝ)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      _Z H H_KV N_CTX ROUND_CTX NKV_CTX
      _sliding_window_offset _sliding_window_size
      IS_EVEN_M _IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N END INIT
      _SLIDING_WINDOW _COMPLEMENT_SLIDING_WINDOW : Nat) :
    ComputeKernel := triton {
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
  qk_scale = $(sm_scale) * 1.0
  qk_scale *= 1.4426950408889634
  if $(IS_EVEN_M) != $(0) {
    q = tl.load(Q_block_ptr)
  } else {
    q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
  }
  for start_n in range($(0), $(NKV_CTX), $(BLOCK_N)) {
    k = tl.load(K_block_ptr)
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    qk += tl.dot(q, k)
    qk = qk * qk_scale
    if $(_SLIDING_WINDOW) != $(0) {
      dist = tl.arange(0, $(BLOCK_M))[:, None] - tl.arange(0, $(BLOCK_N))[None, :]
        + start_m * $(BLOCK_M) - start_n + $(_sliding_window_offset)
      if $(_COMPLEMENT_SLIDING_WINDOW) != $(0) {
        mask = dist >= $(_sliding_window_size)
      } else {
        mask = (dist >= $(0)) & (dist < $(_sliding_window_size))
      }
      qk = tl.where(mask, qk, float("-inf"))
    }
    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    qk = qk - m_ij[:, None]
    p = tl.math.exp2(qk)
    if $(_SLIDING_WINDOW) != $(0) {
      p = tl.where(mask, p, 0.0)
    }
    l_ij = tl.sum(p, 1)
    tmp = m_i - m_ij
    alpha = tl.math.exp2(tmp)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]
    v = tl.load(V_block_ptr)
    acc += tl.dot(p, v)
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
  tl.store(O_block_ptr, (acc).to(Out.type.element_ty))
}
```
</details>

<details><summary><code>active</code></summary>

```lean
def active
    (s : BlockState) (N_CTX HEAD_ACTIVE BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Prop :=
  mIndex s BLOCK_M idx.1 < N_CTX ∧ kIndex idx < HEAD_ACTIVE
```
</details>

<details><summary><code>outOffset</code></summary>

```lean
def outOffset
    (s : BlockState)
    (H stride_qz stride_qh stride_qm stride_qk BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  offZ s H * stride_qz + offH s H * stride_qh +
    mIndex s BLOCK_M idx.1 * stride_qm + kIndex idx * stride_qk
```
</details>

<details><summary><code>attentionFwdTriton3Case4OutSpecG</code></summary>

```
/-- **Genuine closed form, case 4** (`INIT=False` cross-launch resume, sliding
window). The output lane `(i,d)` is the normalized resume-seeded online softmax:
the `aft3StateSeededG` fold of the sliding-window-kept keys onto the loaded
`aft3Case4Seed`, read off as `acc / denom`. Genuinely over INPUT memory
(`Q`/`K`/`V` for the keys, `M`/`L`/`Out` for the resume seed) — **no
self-reference** to this program's executed output. -/
```
```lean
noncomputable def attentionFwdTriton3Case4OutSpecG
    (s : BlockState) (Q K V M Out L : RegionName)
    (base BM ND NC sqm sqk skn skk svk svn som son ROUND_CTX : Nat) (sc : ℝ)
    (BN off size : Nat) (hND : 0 < ND) (idx : TileIndex [BM, ND]) : ℝ :=
  let seed := aft3Case4Seed s M Out L base BM ND som son ROUND_CTX
  let kp := fun i j => natSlidingWindowKeepG (s.pids 0) BM BN off size i j
  (aft3StateSeededG (qTile3G s Q base BM ND sqm sqk) (kTile3G s K base NC ND skn skk)
      (vTile3G s V base NC ND svk svn) (keyScale3G sc NC) kp seed NC idx.1 idx.2.1).2.2
    / (aft3StateSeededG (qTile3G s Q base BM ND sqm sqk) (kTile3G s K base NC ND skn skk)
      (vTile3G s V base NC ND svk svn) (keyScale3G sc NC) kp seed NC idx.1 ⟨0, hND⟩).2.1
```
</details>

<details><summary><code>lRowOffset</code></summary>

```
/-- Proof-oriented L (log-sum-exp) row store slice of `attention_fwd_triton3.py`.
Takes a precomputed `LPre` vector and proves the row writeback into `L` at
offset `off_hz * ROUND_CTX + offs_m`. -/
```
```lean
def lRowOffset (s : BlockState) (off_hz ROUND_CTX BLOCK_M : Nat)
    (i : Fin BLOCK_M) : Nat :=
  off_hz * ROUND_CTX + (s.pids 0 * BLOCK_M + i.val)
```
</details>

<details><summary><code>attentionFwdTriton3Case4MSpecG</code></summary>

```
/-- **Case-4 finalize M closed form (general).** The `m + log2 l` finalize of the
resume-seeded sliding-window online-softmax fold (`aft3StateSeededG` over the seed
`aft3Case4Seed`, read off at column `0`), as written to `M[row]`. Named wrapper so
the `output_summary_general` statement stays a one-liner (mirrors
`attentionFwdTriton3Case3MSpecG`). -/
```
```lean
noncomputable def attentionFwdTriton3Case4MSpecG
    (s : BlockState) (Q K V M Out L : RegionName)
    (base BM ND NC sqm sqk skn skk svk svn som son ROUND_CTX : Nat) (sc : ℝ)
    (BN off size : Nat) (i : Fin BM) (hND : 0 < ND) : ℝ :=
  let seed := aft3Case4Seed s M Out L base BM ND som son ROUND_CTX
  let kp := fun i j => natSlidingWindowKeepG (s.pids 0) BM BN off size i j
  (WithBot.realAdd
    (aft3StateSeededG (qTile3G s Q base BM ND sqm sqk) (kTile3G s K base NC ND skn skk) (vTile3G s V base NC ND svk svn) (keyScale3G sc NC) kp seed NC i ⟨0, hND⟩).1
    (WithBot.realLog2 (((aft3StateSeededG (qTile3G s Q base BM ND sqm sqk) (kTile3G s K base NC ND skn skk) (vTile3G s V base NC ND svk svn) (keyScale3G sc NC) kp seed NC i ⟨0, hND⟩).2.1 : ℝ) : WithBot ℝ))).unbotD 0
```
</details>

<details><summary><code>mIndex</code></summary>

```lean
def mIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val
```
</details>

<details><summary><code>kIndex</code></summary>

```lean
def kIndex (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  idx.2.1.val
```
</details>

<details><summary><code>offZ</code></summary>

```
/-- Surface transcription/proof-oriented final output-store slice of `attention_fwd_triton3.py`'s
`_attn_fwd`.

The full kernel runs separate streaming attention stages, including the causal
stage when requested. This slice starts after those stages have produced a
precomputed normalized `Acc` tile and proves the final masked writeback into
`Out`, preserving the source store address and mask
`(offs_m < N_CTX) & (offs_k < HEAD_ACTIVE)`. The inner `tl.float32` accumulator is
outside this slice. -/
```
```lean
def offZ (s : BlockState) (H : Nat) : Nat :=
  s.pids 1 / H
```
</details>

<details><summary><code>offH</code></summary>

```lean
def offH (s : BlockState) (H : Nat) : Nat :=
  s.pids 1 % H
```
</details>

<details><summary><code>aft3Case4Seed</code></summary>

```
/-- The `INIT=False` resume seed loaded from input memory: per row `i`,
`m_i = M[off_hz·ROUND_CTX + start_m·BM + i]`, `l_i = L[…]`, and per lane `(i,d)`,
`acc = Out[base + (start_m·BM + i)·som + d·son]` — all read from the **initial**
state `s` (these are the running results of prior chunk launches, i.e. genuine
INPUT memory to this program, not this program's own executed output). -/
```
```lean
noncomputable def aft3Case4Seed
    (s : BlockState) (M Out L : RegionName)
    (base BM ND som son ROUND_CTX : Nat) :
    Fin BM → Fin ND → WithBot ℝ × ℝ × ℝ :=
  fun i d =>
    (((mlRow3G s M ROUND_CTX BM i.val : ℝ) : WithBot ℝ),
     mlRow3G s L ROUND_CTX BM i.val,
     outLane3G s Out base BM som son i.val d.val)
```
</details>

<details><summary><code>natSlidingWindowKeepG</code></summary>

```
/-- General case-1 keep predicate: `dist < size`. -/
```
```lean
def natSlidingWindowKeepG (SM BM BN off size : Nat) {NC : Nat}
    (i : Fin BM) (j : Fin NC) : Prop :=
  natDist3G SM BM BN off i j < size
```
</details>

<details><summary><code>aft3StateSeededG</code></summary>

```
/-- General resume-**SEEDED** running `(max, denom, acc)` after the windowed prefix
`[0, hi)`: the online-softmax `aft3OsStepBot` fold from an arbitrary initial state
`init i d`, rather than the `(⊥, 0, 0)` of `aft3StateBotG`. This is the case-4
(`INIT=False` cross-launch resume) analogue, where `init` is the prior
`(m_i, l_i, acc)` loaded from the input `M`/`L`/`Out` buffers. -/
```
```lean
noncomputable def aft3StateSeededG {BM ND NC : Nat}
    (qT : TileIndex [BM, ND] → ℝ) (kT vT : TileIndex [NC, ND] → ℝ)
    (keyScale : Fin NC → ℝ) (keep : Fin BM → Fin NC → Prop)
    [∀ i j, Decidable (keep i j)]
    (init : Fin BM → Fin ND → WithBot ℝ × ℝ × ℝ)
    (hi : Nat) (i : Fin BM) (d : Fin ND) : WithBot ℝ × ℝ × ℝ :=
  (aft3KeysUptoG qT kT vT keyScale keep hi i d).foldl aft3OsStepBot (init i d)
```
</details>

<details><summary><code>qTile3G</code></summary>

```
/-- General query tile: query row `i`, head lane `e`, at
`base + (pid0·BM + i)·sqm + e·sqk`. -/
```
```lean
noncomputable def qTile3G (s : BlockState) (Q : RegionName)
    (base BM ND sqm sqk : Nat) : TileIndex [BM, ND] → ℝ :=
  fun (i, e, _) => s.readMem Q (base + (s.pids 0 * BM + i.val) * sqm + e.val * sqk)
```
</details>

<details><summary><code>kTile3G</code></summary>

```
/-- General key tile: key `j` (global), head lane `e`, at `base + j·skn + e·skk`. -/
```
```lean
noncomputable def kTile3G (s : BlockState) (K : RegionName)
    (base NC ND skn skk : Nat) : TileIndex [NC, ND] → ℝ :=
  fun (j, e, _) => s.readMem K (base + j.val * skn + e.val * skk)
```
</details>

<details><summary><code>vTile3G</code></summary>

```
/-- General value tile: key `j` (global), head lane `d`, at `base + j·svk + d·svn`. -/
```
```lean
noncomputable def vTile3G (s : BlockState) (V : RegionName)
    (base NC ND svk svn : Nat) : TileIndex [NC, ND] → ℝ :=
  fun (j, d, _) => s.readMem V (base + j.val * svk + d.val * svn)
```
</details>

<details><summary><code>keyScale3G</code></summary>

```
/-- General per-key uniform score scale (= `sm_scale · log2e`). -/
```
```lean
noncomputable def keyScale3G (sc : ℝ) (NC : Nat) : Fin NC → ℝ := fun _ => sc
```
</details>

<details><summary><code>mlRow3G</code></summary>

```
/-- Per-row running-stat entry `R[off_hz·ROUND_CTX + start_m·BM + i]` — the
shared row layout of the `M` (running max) and `L` (running denom) buffers. -/
```
```lean
noncomputable def mlRow3G (s : BlockState) (R : RegionName)
    (ROUND_CTX BM : Nat) (i : Nat) : ℝ :=
  s.readMem R (s.pids 1 * ROUND_CTX + (s.pids 0 * BM + i))
```
</details>

<details><summary><code>outLane3G</code></summary>

```
/-- Output-buffer lane `Out[base + (start_m·BM + i)·som + d·son]` — the `Out`
block-pointer layout at row stride `som`, lane stride `son`. -/
```
```lean
noncomputable def outLane3G (s : BlockState) (Out : RegionName)
    (base BM som son : Nat) (i d : Nat) : ℝ :=
  s.readMem Out (base + (s.pids 0 * BM + i) * som + d * son)
```
</details>

<details><summary><code>natDist3G</code></summary>

```
/-- General faithful nat-truncated sliding-window distance, block-local key
`jL = j mod BN`, block start `start_n = (j / BN)·BN`:
`dist = (i − jL : ℕ) + SM·BM − start_n + offset`. -/
```
```lean
def natDist3G (SM BM BN off : Nat) {NC : Nat} (i : Fin BM) (j : Fin NC) : Nat :=
  (i.val - j.val % BN) + SM * BM - (j.val / BN) * BN + off
```
</details>

<details><summary><code>aft3KeysUptoG</code></summary>

```
/-- General windowed prefix key list `[0, hi)`. -/
```
```lean
noncomputable def aft3KeysUptoG {BM ND NC : Nat}
    (qT : TileIndex [BM, ND] → ℝ) (kT vT : TileIndex [NC, ND] → ℝ)
    (keyScale : Fin NC → ℝ) (keep : Fin BM → Fin NC → Prop)
    [∀ i j, Decidable (keep i j)] (hi : Nat) (i : Fin BM) (d : Fin ND) :
    List (ℝ × ℝ) :=
  (List.finRange NC).filterMap (fun j : Fin NC =>
    if j.val < hi ∧ keep i j then
      some (keyScale j * Finset.univ.sum (fun e : Fin ND =>
              qT (i, e, PUnit.unit) * kT (j, e, PUnit.unit)),
            vT (j, d, PUnit.unit))
    else none)
```
</details>

<details><summary><code>aft3OsStepBot</code></summary>

```
/-- **Body split (case 1).** The lowered algorithm body of the case-1 surface is
exactly `aft3PreLoopG ++ Stmt.forRange "start_n" 0 128 64 aft3LoopBodyG ::
aft3PostLoopG`. -/
```
```lean
noncomputable def aft3OsStepBot (st : WithBot ℝ × ℝ × ℝ) (sv : ℝ × ℝ) : WithBot ℝ × ℝ × ℝ :=
  let m := st.1; let l := st.2.1; let acc := st.2.2
  let s := sv.1; let v := sv.2
  let m' := m ⊔ ((s : ℝ) : WithBot ℝ)
  let α := (WithBot.realExp2 (WithBot.realSub m m')).unbotD 0
  let p := pow2 (s - m'.unbotD 0)
  (m', l * α + p, acc * α + p * v)
```
</details>
