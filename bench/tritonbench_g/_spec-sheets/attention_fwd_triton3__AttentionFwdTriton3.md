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

## Public theorem: `attention_fwd_triton3_case3_io_correctness`

<details><summary>docstring</summary>

```
/-- **The case-3 `⊨[R]` io headline (round-7 exemplar; the bench's first io
headline on a `make_block_ptr` kernel).** For every rounding model `R`, the
case-3 (`(END,INIT,SLIDING_WINDOW,CSW) = (1,1,0,0)`) `_attn_fwd` surface
implements, on its `StreamMasked3DKernelIO₃ₓ₂` signature, the **ideal-ℝ
online-softmax attention fold** over the three streamed tiles: output lane
`j = (i, d)` of `Out` holds the plain base-2 per-key-scale softmax
(`attentionFwdTriton3Case3IOOutSpec` = the existing
`attentionFwdTriton3Case3OutSpecG` closed form restated on the streams), and
row `i` of `M` holds the `m_i + log2 l_i` finalize
(`attentionFwdTriton3Case3IOMSpec` = `attentionFwdTriton3Case3MSpecG` on the
streams). Both output grids are the `.real` default (the store-side
`.to(Out.type.element_ty)` cast erases to the identity at translation), so at
every `R` the terminal cells carry the exact fold values.

**Hypothesis provenance** (inherited verbatim from the exact case-3 headline
`attention_fwd_triton3_python_case3_output_summary_general`, all
truth-forced): `0 < ND/BM/BN/NKV_CTX` and `BN ∣ NKV_CTX` shape the KV walk
(`T = NKV_CTX / BN` full blocks); `0 < H`, `H_KV = H` and the six stride
equalities collapse the four `q/k/v/o` plane offsets onto the shared base
`pid₁/H·sqz + pid₁%H·sqh` (the kernel's GQA-free regime); `M ≠ Out` keeps the
`O` store from clobbering the `M` row; `hinjO`/`hinjM` (∀-pids forms of the
exact stack's per-program injectivity) make the per-lane readback of the two
scatter stores well-defined. The exact headline's `hundef` is **not** a
hypothesis here — the skin's Hoare triple carries the `undef` pin itself.

**Scope disclosed**: this round showcases the io face on **case 3 only**; the
sliding-window cases 1/2 and the `INIT=0` resume case 4 stay on their
existing exact headlines (follow-ups on the same io pattern). The port's
known fidelity gaps (`IS_EVEN_N = 1` hardwired, casts erased to identity,
`@triton.heuristics` not modeled) are inherited as-is from the surface. -/
```
</details>

**Statement:**
```lean
specification attention_fwd_triton3_case3_io_correctness (R : RoundingModel)
    (Q K V M Out L : RegionName) (sm_scale : ℝ)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      Z H H_KV N_CTX ROUND_CTX NKV_CTX off BM ND BN : Nat)
    (hND : 0 < ND) (hBM : 0 < BM) (hBN : 0 < BN) (hNC : 0 < NKV_CTX) (hBNdvd : BN ∣ NKV_CTX)
    (hH : 0 < H) (hHKV : H_KV = H)
    (hskz : skz = sqz) (hskh : skh = sqh) (hsvz : svz = sqz) (hsvh : svh = sqh)
    (hsoz : soz = sqz) (hsoh : soh = sqh)
    (hMO : M ≠ Out)
    (hinjO : ∀ p₀ p₁ : Nat, Function.Injective
      (fun idx : TileIndex [BM, ND] =>
        (p₁ / H * sqz + p₁ % H * sqh) + (p₀ * BM + idx.1.val) * som + idx.2.1.val * son))
    (hinjM : ∀ p₀ p₁ : Nat, Function.Injective
      (fun r : TileIndex [BM] => p₁ * ROUND_CTX + (p₀ * BM + r.1.val))) :
    attentionFwdTriton3Case3IO Q K V M Out L sm_scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son Z H H_KV N_CTX ROUND_CTX NKV_CTX off BM ND BN ⊨[R]
      fun _ _ _ xs ys zs =>
        (fun j => attentionFwdTriton3Case3IOOutSpec BM ND BN NKV_CTX (NKV_CTX / BN)
            (Nat.div_pos (Nat.le_of_dvd hNC hBNdvd) hBN) (Nat.div_mul_cancel hBNdvd) hBN
            (sm_scale * 1.4426950408889634) xs ys zs j,
         fun i => attentionFwdTriton3Case3IOMSpec BM ND BN NKV_CTX (NKV_CTX / BN) hND
            (Nat.div_pos (Nat.le_of_dvd hNC hBNdvd) hBN) (Nat.div_mul_cancel hBNdvd) hBN
            (sm_scale * 1.4426950408889634) xs ys zs i)
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

**Closed-form spec defs (transitive):** `attentionFwdTriton3Case3IO`, `attentionFwdTriton3Case3IOOutSpec`, `attentionFwdTriton3Case3IOMSpec`, `attention_fwd_triton3_surface`, `aft3IOqT`, `aft3IOkT`, `aft3IOvT`, `keyScale3G`, `aft3RunningMaxG`, `aft3StateBot1G`, `aft3KeysUptoG`, `aft3OsStepBot`

<details><summary><code>attentionFwdTriton3Case3IO</code></summary>

```
/-- **Streaming IO signature** of the case-3 (`(END,INIT,SW,CSW) = (1,1,0,0)`)
`_attn_fwd` surface on the three-stream two-output attention fold skin. -/
```
```lean
def attentionFwdTriton3Case3IO (Q K V M Out L : RegionName) (sm_scale : ℝ)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      Z H H_KV N_CTX ROUND_CTX NKV_CTX off BM ND BN : Nat) :
    StreamMasked3DKernelIO₃ₓ₂ where
  kernel := attention_fwd_triton3_surface Q K V M Out L sm_scale
    sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
    Z H H_KV N_CTX ROUND_CTX NKV_CTX off 0 1 1 BM ND BN 1 1 0 0
  inp1 := Q
  inp2 := K
  inp3 := V
  out1 := Out
  out2 := M
  T := NKV_CTX / BN
  B1 := BM * ND
  B2 := ND * BN
  B3 := BN * ND
  C1 := BM * ND
  C2 := BM
  read1 := fun p₀ p₁ _ _ j =>
    (p₁ / H * sqz + p₁ % H * sqh) + (p₀ * BM + j.val / ND) * sqm + (j.val % ND) * sqk
  read2 := fun _ p₁ _ t j =>
    (p₁ / H * sqz + p₁ % H * sqh) + (j.val / BN) * skk + (t.val * BN + j.val % BN) * skn
  read3 := fun _ p₁ _ t j =>
    (p₁ / H * sqz + p₁ % H * sqh) + (t.val * BN + j.val / ND) * svk + (j.val % ND) * svn
  write1 := fun p₀ p₁ _ j =>
    (p₁ / H * sqz + p₁ % H * sqh) + (p₀ * BM + j.val / ND) * som + (j.val % ND) * son
  write2 := fun p₀ p₁ _ j => p₁ * ROUND_CTX + (p₀ * BM + j.val)
  mask1 := fun _ _ _ _ _ => True
  mask2 := fun _ _ _ _ _ => True
  mask3 := fun _ _ _ _ _ => True
  writeMask1 := fun _ _ _ _ => True
  writeMask2 := fun _ _ _ _ => True
```
</details>

<details><summary><code>attentionFwdTriton3Case3IOOutSpec</code></summary>

```
/-- **Case-3 `Out` closed form on the streams**: the plain base-2
per-key-scale softmax (`attentionRealBase2PerKeyScalePred`, exactly the
existing `attentionFwdTriton3Case3OutSpecG`) restated over the three streamed
tiles, at output lane `j = (i, d)` row-major over `[BM, ND]`. -/
```
```lean
noncomputable def attentionFwdTriton3Case3IOOutSpec (BM ND BN NC T : Nat)
    (hT : 0 < T) (hTB : T * BN = NC) (hBN : 0 < BN) (sc : ℝ)
    (xs : Fin T → Fin (BM * ND) → ℝ) (ys : Fin T → Fin (ND * BN) → ℝ)
    (zs : Fin T → Fin (BN * ND) → ℝ) (j : Fin (BM * ND)) : ℝ :=
  attentionRealBase2PerKeyScalePred (aft3IOqT BM ND T hT xs)
    (aft3IOkT ND BN NC T hTB hBN ys) (aft3IOvT ND BN NC T hTB hBN zs)
    (keyScale3G sc NC) (fun (i : Fin BM) (j : Fin NC) => noWindowKeep i j)
    (Lane2D.decode j)
```
</details>

<details><summary><code>attentionFwdTriton3Case3IOMSpec</code></summary>

```
/-- **Case-3 `M` closed form on the streams**: the `m_i + log2 l_i` finalize
(exactly the existing `attentionFwdTriton3Case3MSpecG`) restated over the
three streamed tiles, at output row `i`. -/
```
```lean
noncomputable def attentionFwdTriton3Case3IOMSpec (BM ND BN NC T : Nat)
    (hND : 0 < ND) (hT : 0 < T) (hTB : T * BN = NC) (hBN : 0 < BN) (sc : ℝ)
    (xs : Fin T → Fin (BM * ND) → ℝ) (ys : Fin T → Fin (ND * BN) → ℝ)
    (zs : Fin T → Fin (BN * ND) → ℝ) (i : Fin BM) : ℝ :=
  (aft3RunningMaxG (aft3IOqT BM ND T hT xs) (aft3IOkT ND BN NC T hTB hBN ys)
      (aft3IOvT ND BN NC T hTB hBN zs) (keyScale3G sc NC)
      (fun i j => noWindowKeep i j) NC i ⟨0, hND⟩).unbotD 0
    + Real.log
      ((aft3StateBot1G (aft3IOqT BM ND T hT xs) (aft3IOkT ND BN NC T hTB hBN ys)
          (aft3IOvT ND BN NC T hTB hBN zs) (keyScale3G sc NC)
          (fun i j => noWindowKeep i j) NC i ⟨0, hND⟩).2.1) / Real.log 2
```
</details>

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

<details><summary><code>aft3IOqT</code></summary>

```
/-- The `Q` tile read off the (static) first stream: the window ignores `t`,
so the step-`0` slice carries the whole tile. -/
```
```lean
noncomputable def aft3IOqT (BM ND T : Nat) (hT : 0 < T)
    (xs : Fin T → Fin (BM * ND) → ℝ) : TileIndex [BM, ND] → ℝ :=
  fun idx => xs ⟨0, hT⟩ (Lane2D.encode idx)
```
</details>

<details><summary><code>aft3IOkT</code></summary>

```
/-- The global `K` tile read off the second stream: global key `j` lives in
step `j / BN`, block-local column `j % BN`, at stream lane `(e, j % BN)`. -/
```
```lean
noncomputable def aft3IOkT (ND BN NC T : Nat) (hTB : T * BN = NC) (hBN : 0 < BN)
    (ys : Fin T → Fin (ND * BN) → ℝ) : TileIndex [NC, ND] → ℝ :=
  fun idx =>
    ys ⟨idx.1.val / BN, (Nat.div_lt_iff_lt_mul hBN).mpr (by rw [hTB]; exact idx.1.isLt)⟩
      (Lane2D.encode (idx.2.1, ⟨idx.1.val % BN, Nat.mod_lt _ hBN⟩, PUnit.unit))
```
</details>

<details><summary><code>aft3IOvT</code></summary>

```
/-- The global `V` tile read off the third stream: global key `j` lives in
step `j / BN`, block-local row `j % BN`, at stream lane `(j % BN, d)`. -/
```
```lean
noncomputable def aft3IOvT (ND BN NC T : Nat) (hTB : T * BN = NC) (hBN : 0 < BN)
    (zs : Fin T → Fin (BN * ND) → ℝ) : TileIndex [NC, ND] → ℝ :=
  fun idx =>
    zs ⟨idx.1.val / BN, (Nat.div_lt_iff_lt_mul hBN).mpr (by rw [hTB]; exact idx.1.isLt)⟩
      (Lane2D.encode (⟨idx.1.val % BN, Nat.mod_lt _ hBN⟩, idx.2.1, PUnit.unit))
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

## Public theorem: `attention_fwd_triton3_case1_io_correctness`

<details><summary>docstring</summary>

```
/-- **The case-1 `⊨[R]` io headline (sliding window).** For every rounding
model `R`, the case-1 (`(END,INIT,SLIDING_WINDOW,COMPLEMENT) = (1,1,1,0)`)
`_attn_fwd` surface implements, on its `StreamMasked3DKernelIO₃ₓ₂` signature,
the **ideal-ℝ sliding-window online-softmax attention fold** over the three
streamed tiles: output lane `j = (i, d)` of `Out` holds the window-filtered
base-2 per-key-scale softmax (`attentionFwdTriton3Case1IOOutSpec` = the
existing `attentionFwdTriton3Case1OutSpecG` closed form restated on the
streams — the `tl.where` window mask `0 ≤ dist ∧ dist < size` enters as the
`natSlidingWindowKeepG p₀` key filter), and row `i` of `M` holds the raw
`(m ⊔ … + log2 l).unbotD 0` finalize (`attentionFwdTriton3Case1IOMSpec` =
`attentionFwdTriton3KMSpecG` on the streams). Both output grids are the
`.real` default, so at every `R` the terminal cells carry the exact fold
values.

**Hypothesis provenance** (inherited verbatim from the exact case-1 headline
`attention_fwd_triton3_python_case1_output_summary_general`, all
truth-forced): `0 < ND/BM/BN/NKV_CTX` and `BN ∣ NKV_CTX` shape the KV walk;
`0 < H`, `H_KV = H` and the six stride equalities collapse the four plane
offsets onto the shared base `pid₁/H·sqz + pid₁%H·sqh`; `M ≠ Out` keeps the
`O` store off the `M` row; `hinjO`/`hinjM` (∀-pids forms of the exact
stack's per-program injectivity) make the scatter readbacks well-defined.
The exact headline's `hundef` is not a hypothesis — the skin's Hoare triple
carries the `undef` pin itself.

**Scope disclosed**: with this and the case-2 headline below, three of the
four constexpr cases carry io faces. Case 4 (`INIT=0` seeded resume) is
**not** converted: its surface reads six input regions (`Q`/`K`/`V` plus the
`M`/`L`/`Out` resume rows, `M` and `Out` simultaneously input and output) of
non-uniform per-step widths (`BM·ND`/`ND·BN`/`BN·ND`/`BM`/`BM`/`BM·ND`) —
neither the 3-input `StreamMasked3DKernelIO₃ₓ₂` skin nor the uniform-`B`
`StreamGroupedEmitMasked3DKernelIO` vector skin can state that honestly; it
needs a named-field `₆ₓ₂` (or per-channel-`B` grouped) variant and stays on
its exact headline `attention_fwd_triton3_python_case4_output_summary_general`
(deferred). The port's known fidelity gaps (`IS_EVEN_N = 1` hardwired, casts
erased to identity, `@triton.heuristics` not modeled) are inherited as-is. -/
```
</details>

**Statement:**
```lean
specification attention_fwd_triton3_case1_io_correctness (R : RoundingModel)
    (Q K V M Out L : RegionName) (sm_scale : ℝ)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      Z H H_KV N_CTX ROUND_CTX NKV_CTX off size BM ND BN : Nat)
    (hND : 0 < ND) (hBM : 0 < BM) (hBN : 0 < BN) (hNC : 0 < NKV_CTX) (hBNdvd : BN ∣ NKV_CTX)
    (hH : 0 < H) (hHKV : H_KV = H)
    (hskz : skz = sqz) (hskh : skh = sqh) (hsvz : svz = sqz) (hsvh : svh = sqh)
    (hsoz : soz = sqz) (hsoh : soh = sqh)
    (hMO : M ≠ Out)
    (hinjO : ∀ p₀ p₁ : Nat, Function.Injective
      (fun idx : TileIndex [BM, ND] =>
        (p₁ / H * sqz + p₁ % H * sqh) + (p₀ * BM + idx.1.val) * som + idx.2.1.val * son))
    (hinjM : ∀ p₀ p₁ : Nat, Function.Injective
      (fun r : TileIndex [BM] => p₁ * ROUND_CTX + (p₀ * BM + r.1.val))) :
    attentionFwdTriton3Case1IO Q K V M Out L sm_scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son Z H H_KV N_CTX ROUND_CTX NKV_CTX off size BM ND BN ⊨[R]
      fun p₀ _ _ xs ys zs =>
        (fun j => attentionFwdTriton3Case1IOOutSpec p₀ BM ND BN NKV_CTX (NKV_CTX / BN)
            (Nat.div_pos (Nat.le_of_dvd hNC hBNdvd) hBN) (Nat.div_mul_cancel hBNdvd) hBN
            (sm_scale * 1.4426950408889634) off size xs ys zs j,
         fun i => attentionFwdTriton3Case1IOMSpec p₀ BM ND BN NKV_CTX (NKV_CTX / BN) hND
            (Nat.div_pos (Nat.le_of_dvd hNC hBNdvd) hBN) (Nat.div_mul_cancel hBNdvd) hBN
            (sm_scale * 1.4426950408889634) off size xs ys zs i)
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

**Closed-form spec defs (transitive):** `attentionFwdTriton3Case1IO`, `attentionFwdTriton3Case1IOOutSpec`, `attentionFwdTriton3Case1IOMSpec`, `attention_fwd_triton3_surface`, `aft3IOqT`, `aft3IOkT`, `aft3IOvT`, `keyScale3G`, `natSlidingWindowKeepG`, `aft3RunningMaxG`, `aft3StateBotKG`, `natDist3G`, `aft3KeysUptoG`, `aft3StateBotG`, `aft3OsStepBot`

<details><summary><code>attentionFwdTriton3Case1IO</code></summary>

```
/-- **Streaming IO signature** of the case-1
(`(END,INIT,SLIDING_WINDOW,COMPLEMENT) = (1,1,1,0)`) `_attn_fwd` surface —
identical reads/writes/masks to the case-3 face (the window is a spec-side
key filter, not an addressing change). -/
```
```lean
def attentionFwdTriton3Case1IO (Q K V M Out L : RegionName) (sm_scale : ℝ)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      Z H H_KV N_CTX ROUND_CTX NKV_CTX off size BM ND BN : Nat) :
    StreamMasked3DKernelIO₃ₓ₂ where
  kernel := attention_fwd_triton3_surface Q K V M Out L sm_scale
    sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
    Z H H_KV N_CTX ROUND_CTX NKV_CTX off size 1 1 BM ND BN 1 1 1 0
  inp1 := Q
  inp2 := K
  inp3 := V
  out1 := Out
  out2 := M
  T := NKV_CTX / BN
  B1 := BM * ND
  B2 := ND * BN
  B3 := BN * ND
  C1 := BM * ND
  C2 := BM
  read1 := fun p₀ p₁ _ _ j =>
    (p₁ / H * sqz + p₁ % H * sqh) + (p₀ * BM + j.val / ND) * sqm + (j.val % ND) * sqk
  read2 := fun _ p₁ _ t j =>
    (p₁ / H * sqz + p₁ % H * sqh) + (j.val / BN) * skk + (t.val * BN + j.val % BN) * skn
  read3 := fun _ p₁ _ t j =>
    (p₁ / H * sqz + p₁ % H * sqh) + (t.val * BN + j.val / ND) * svk + (j.val % ND) * svn
  write1 := fun p₀ p₁ _ j =>
    (p₁ / H * sqz + p₁ % H * sqh) + (p₀ * BM + j.val / ND) * som + (j.val % ND) * son
  write2 := fun p₀ p₁ _ j => p₁ * ROUND_CTX + (p₀ * BM + j.val)
  mask1 := fun _ _ _ _ _ => True
  mask2 := fun _ _ _ _ _ => True
  mask3 := fun _ _ _ _ _ => True
  writeMask1 := fun _ _ _ _ => True
  writeMask2 := fun _ _ _ _ => True
```
</details>

<details><summary><code>attentionFwdTriton3Case1IOOutSpec</code></summary>

```
/-- **Case-1 `Out` closed form on the streams**: the sliding-window base-2
per-key-scale softmax (exactly the exact headline's
`attentionFwdTriton3Case1OutSpecG`, keep `natSlidingWindowKeepG`) restated
over the three streamed tiles, at output lane `j = (i, d)` row-major over
`[BM, ND]`. The window filter eats `p₀` (the `start_m` program id). -/
```
```lean
noncomputable def attentionFwdTriton3Case1IOOutSpec (p₀ BM ND BN NC T : Nat)
    (hT : 0 < T) (hTB : T * BN = NC) (hBN : 0 < BN) (sc : ℝ) (off size : Nat)
    (xs : Fin T → Fin (BM * ND) → ℝ) (ys : Fin T → Fin (ND * BN) → ℝ)
    (zs : Fin T → Fin (BN * ND) → ℝ) (j : Fin (BM * ND)) : ℝ :=
  attentionRealBase2PerKeyScalePred (aft3IOqT BM ND T hT xs)
    (aft3IOkT ND BN NC T hTB hBN ys) (aft3IOvT ND BN NC T hTB hBN zs)
    (keyScale3G sc NC) (fun i j => natSlidingWindowKeepG p₀ BM BN off size i j)
    (Lane2D.decode j)
```
</details>

<details><summary><code>attentionFwdTriton3Case1IOMSpec</code></summary>

```
/-- **Case-1 `M` closed form on the streams**: the raw
`(m ⊔ … + log2 l).unbotD 0` finalize (exactly the exact headline's
`attentionFwdTriton3KMSpecG` at the sliding-window keep) restated over the
three streamed tiles, at output row `i`. -/
```
```lean
noncomputable def attentionFwdTriton3Case1IOMSpec (p₀ BM ND BN NC T : Nat)
    (hND : 0 < ND) (hT : 0 < T) (hTB : T * BN = NC) (hBN : 0 < BN) (sc : ℝ)
    (off size : Nat)
    (xs : Fin T → Fin (BM * ND) → ℝ) (ys : Fin T → Fin (ND * BN) → ℝ)
    (zs : Fin T → Fin (BN * ND) → ℝ) (i : Fin BM) : ℝ :=
  (WithBot.realAdd
      (aft3RunningMaxG (aft3IOqT BM ND T hT xs) (aft3IOkT ND BN NC T hTB hBN ys)
        (aft3IOvT ND BN NC T hTB hBN zs) (keyScale3G sc NC)
        (fun i j => natSlidingWindowKeepG p₀ BM BN off size i j) NC i ⟨0, hND⟩)
      (WithBot.realLog2
        (((aft3StateBotKG (aft3IOqT BM ND T hT xs) (aft3IOkT ND BN NC T hTB hBN ys)
            (aft3IOvT ND BN NC T hTB hBN zs) (keyScale3G sc NC)
            (fun i j => natSlidingWindowKeepG p₀ BM BN off size i j) NC i ⟨0, hND⟩).2.1 : ℝ)
          : WithBot ℝ))).unbotD 0
```
</details>

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

<details><summary><code>aft3IOqT</code></summary>

```
/-- The `Q` tile read off the (static) first stream: the window ignores `t`,
so the step-`0` slice carries the whole tile. -/
```
```lean
noncomputable def aft3IOqT (BM ND T : Nat) (hT : 0 < T)
    (xs : Fin T → Fin (BM * ND) → ℝ) : TileIndex [BM, ND] → ℝ :=
  fun idx => xs ⟨0, hT⟩ (Lane2D.encode idx)
```
</details>

<details><summary><code>aft3IOkT</code></summary>

```
/-- The global `K` tile read off the second stream: global key `j` lives in
step `j / BN`, block-local column `j % BN`, at stream lane `(e, j % BN)`. -/
```
```lean
noncomputable def aft3IOkT (ND BN NC T : Nat) (hTB : T * BN = NC) (hBN : 0 < BN)
    (ys : Fin T → Fin (ND * BN) → ℝ) : TileIndex [NC, ND] → ℝ :=
  fun idx =>
    ys ⟨idx.1.val / BN, (Nat.div_lt_iff_lt_mul hBN).mpr (by rw [hTB]; exact idx.1.isLt)⟩
      (Lane2D.encode (idx.2.1, ⟨idx.1.val % BN, Nat.mod_lt _ hBN⟩, PUnit.unit))
```
</details>

<details><summary><code>aft3IOvT</code></summary>

```
/-- The global `V` tile read off the third stream: global key `j` lives in
step `j / BN`, block-local row `j % BN`, at stream lane `(j % BN, d)`. -/
```
```lean
noncomputable def aft3IOvT (ND BN NC T : Nat) (hTB : T * BN = NC) (hBN : 0 < BN)
    (zs : Fin T → Fin (BN * ND) → ℝ) : TileIndex [NC, ND] → ℝ :=
  fun idx =>
    zs ⟨idx.1.val / BN, (Nat.div_lt_iff_lt_mul hBN).mpr (by rw [hTB]; exact idx.1.isLt)⟩
      (Lane2D.encode (⟨idx.1.val % BN, Nat.mod_lt _ hBN⟩, idx.2.1, PUnit.unit))
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

## Public theorem: `attention_fwd_triton3_case2_io_correctness`

<details><summary>docstring</summary>

```
/-- **The case-2 `⊨[R]` io headline (complement sliding window).** As the
case-1 io headline with the complementary key filter: the constexpr
`COMPLEMENT=1` mask `dist ≥ size` enters the spec as
`natComplementSlidingWindowKeepG p₀`, so `Out` holds
`attentionFwdTriton3Case2IOOutSpec` (= `attentionFwdTriton3Case2OutSpecG` on
the streams) and `M` the matching raw finalize
`attentionFwdTriton3Case2IOMSpec`. Hypotheses inherited verbatim from
`attention_fwd_triton3_python_case2_output_summary_general` minus `hundef`
(carried by the skin's triple). See the case-1 headline's scope note for why
case 4 (`INIT=0` seeded resume, six non-uniform input channels) has no
honest io face on the existing skins and stays on its exact headline. -/
```
</details>

**Statement:**
```lean
specification attention_fwd_triton3_case2_io_correctness (R : RoundingModel)
    (Q K V M Out L : RegionName) (sm_scale : ℝ)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      Z H H_KV N_CTX ROUND_CTX NKV_CTX off size BM ND BN : Nat)
    (hND : 0 < ND) (hBM : 0 < BM) (hBN : 0 < BN) (hNC : 0 < NKV_CTX) (hBNdvd : BN ∣ NKV_CTX)
    (hH : 0 < H) (hHKV : H_KV = H)
    (hskz : skz = sqz) (hskh : skh = sqh) (hsvz : svz = sqz) (hsvh : svh = sqh)
    (hsoz : soz = sqz) (hsoh : soh = sqh)
    (hMO : M ≠ Out)
    (hinjO : ∀ p₀ p₁ : Nat, Function.Injective
      (fun idx : TileIndex [BM, ND] =>
        (p₁ / H * sqz + p₁ % H * sqh) + (p₀ * BM + idx.1.val) * som + idx.2.1.val * son))
    (hinjM : ∀ p₀ p₁ : Nat, Function.Injective
      (fun r : TileIndex [BM] => p₁ * ROUND_CTX + (p₀ * BM + r.1.val))) :
    attentionFwdTriton3Case2IO Q K V M Out L sm_scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son Z H H_KV N_CTX ROUND_CTX NKV_CTX off size BM ND BN ⊨[R]
      fun p₀ _ _ xs ys zs =>
        (fun j => attentionFwdTriton3Case2IOOutSpec p₀ BM ND BN NKV_CTX (NKV_CTX / BN)
            (Nat.div_pos (Nat.le_of_dvd hNC hBNdvd) hBN) (Nat.div_mul_cancel hBNdvd) hBN
            (sm_scale * 1.4426950408889634) off size xs ys zs j,
         fun i => attentionFwdTriton3Case2IOMSpec p₀ BM ND BN NKV_CTX (NKV_CTX / BN) hND
            (Nat.div_pos (Nat.le_of_dvd hNC hBNdvd) hBN) (Nat.div_mul_cancel hBNdvd) hBN
            (sm_scale * 1.4426950408889634) off size xs ys zs i)
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

**Closed-form spec defs (transitive):** `attentionFwdTriton3Case2IO`, `attentionFwdTriton3Case2IOOutSpec`, `attentionFwdTriton3Case2IOMSpec`, `attention_fwd_triton3_surface`, `aft3IOqT`, `aft3IOkT`, `aft3IOvT`, `keyScale3G`, `natComplementSlidingWindowKeepG`, `aft3RunningMaxG`, `aft3StateBotKG`, `natDist3G`, `aft3KeysUptoG`, `aft3StateBotG`, `aft3OsStepBot`

<details><summary><code>attentionFwdTriton3Case2IO</code></summary>

```
/-- **Streaming IO signature** of the case-2 (`(1,1,1,1)`, complement window)
surface — same io skeleton, complement constexpr instantiation. -/
```
```lean
def attentionFwdTriton3Case2IO (Q K V M Out L : RegionName) (sm_scale : ℝ)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      Z H H_KV N_CTX ROUND_CTX NKV_CTX off size BM ND BN : Nat) :
    StreamMasked3DKernelIO₃ₓ₂ where
  kernel := attention_fwd_triton3_surface Q K V M Out L sm_scale
    sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
    Z H H_KV N_CTX ROUND_CTX NKV_CTX off size 1 1 BM ND BN 1 1 1 1
  inp1 := Q
  inp2 := K
  inp3 := V
  out1 := Out
  out2 := M
  T := NKV_CTX / BN
  B1 := BM * ND
  B2 := ND * BN
  B3 := BN * ND
  C1 := BM * ND
  C2 := BM
  read1 := fun p₀ p₁ _ _ j =>
    (p₁ / H * sqz + p₁ % H * sqh) + (p₀ * BM + j.val / ND) * sqm + (j.val % ND) * sqk
  read2 := fun _ p₁ _ t j =>
    (p₁ / H * sqz + p₁ % H * sqh) + (j.val / BN) * skk + (t.val * BN + j.val % BN) * skn
  read3 := fun _ p₁ _ t j =>
    (p₁ / H * sqz + p₁ % H * sqh) + (t.val * BN + j.val / ND) * svk + (j.val % ND) * svn
  write1 := fun p₀ p₁ _ j =>
    (p₁ / H * sqz + p₁ % H * sqh) + (p₀ * BM + j.val / ND) * som + (j.val % ND) * son
  write2 := fun p₀ p₁ _ j => p₁ * ROUND_CTX + (p₀ * BM + j.val)
  mask1 := fun _ _ _ _ _ => True
  mask2 := fun _ _ _ _ _ => True
  mask3 := fun _ _ _ _ _ => True
  writeMask1 := fun _ _ _ _ => True
  writeMask2 := fun _ _ _ _ => True
```
</details>

<details><summary><code>attentionFwdTriton3Case2IOOutSpec</code></summary>

```
/-- **Case-2 `Out` closed form on the streams** (complement keep
`natComplementSlidingWindowKeepG`; otherwise as case 1). -/
```
```lean
noncomputable def attentionFwdTriton3Case2IOOutSpec (p₀ BM ND BN NC T : Nat)
    (hT : 0 < T) (hTB : T * BN = NC) (hBN : 0 < BN) (sc : ℝ) (off size : Nat)
    (xs : Fin T → Fin (BM * ND) → ℝ) (ys : Fin T → Fin (ND * BN) → ℝ)
    (zs : Fin T → Fin (BN * ND) → ℝ) (j : Fin (BM * ND)) : ℝ :=
  attentionRealBase2PerKeyScalePred (aft3IOqT BM ND T hT xs)
    (aft3IOkT ND BN NC T hTB hBN ys) (aft3IOvT ND BN NC T hTB hBN zs)
    (keyScale3G sc NC) (fun i j => natComplementSlidingWindowKeepG p₀ BM BN off size i j)
    (Lane2D.decode j)
```
</details>

<details><summary><code>attentionFwdTriton3Case2IOMSpec</code></summary>

```
/-- **Case-2 `M` closed form on the streams** (complement keep). -/
```
```lean
noncomputable def attentionFwdTriton3Case2IOMSpec (p₀ BM ND BN NC T : Nat)
    (hND : 0 < ND) (hT : 0 < T) (hTB : T * BN = NC) (hBN : 0 < BN) (sc : ℝ)
    (off size : Nat)
    (xs : Fin T → Fin (BM * ND) → ℝ) (ys : Fin T → Fin (ND * BN) → ℝ)
    (zs : Fin T → Fin (BN * ND) → ℝ) (i : Fin BM) : ℝ :=
  (WithBot.realAdd
      (aft3RunningMaxG (aft3IOqT BM ND T hT xs) (aft3IOkT ND BN NC T hTB hBN ys)
        (aft3IOvT ND BN NC T hTB hBN zs) (keyScale3G sc NC)
        (fun i j => natComplementSlidingWindowKeepG p₀ BM BN off size i j) NC i ⟨0, hND⟩)
      (WithBot.realLog2
        (((aft3StateBotKG (aft3IOqT BM ND T hT xs) (aft3IOkT ND BN NC T hTB hBN ys)
            (aft3IOvT ND BN NC T hTB hBN zs) (keyScale3G sc NC)
            (fun i j => natComplementSlidingWindowKeepG p₀ BM BN off size i j) NC i ⟨0, hND⟩).2.1 : ℝ)
          : WithBot ℝ))).unbotD 0
```
</details>

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

<details><summary><code>aft3IOqT</code></summary>

```
/-- The `Q` tile read off the (static) first stream: the window ignores `t`,
so the step-`0` slice carries the whole tile. -/
```
```lean
noncomputable def aft3IOqT (BM ND T : Nat) (hT : 0 < T)
    (xs : Fin T → Fin (BM * ND) → ℝ) : TileIndex [BM, ND] → ℝ :=
  fun idx => xs ⟨0, hT⟩ (Lane2D.encode idx)
```
</details>

<details><summary><code>aft3IOkT</code></summary>

```
/-- The global `K` tile read off the second stream: global key `j` lives in
step `j / BN`, block-local column `j % BN`, at stream lane `(e, j % BN)`. -/
```
```lean
noncomputable def aft3IOkT (ND BN NC T : Nat) (hTB : T * BN = NC) (hBN : 0 < BN)
    (ys : Fin T → Fin (ND * BN) → ℝ) : TileIndex [NC, ND] → ℝ :=
  fun idx =>
    ys ⟨idx.1.val / BN, (Nat.div_lt_iff_lt_mul hBN).mpr (by rw [hTB]; exact idx.1.isLt)⟩
      (Lane2D.encode (idx.2.1, ⟨idx.1.val % BN, Nat.mod_lt _ hBN⟩, PUnit.unit))
```
</details>

<details><summary><code>aft3IOvT</code></summary>

```
/-- The global `V` tile read off the third stream: global key `j` lives in
step `j / BN`, block-local row `j % BN`, at stream lane `(j % BN, d)`. -/
```
```lean
noncomputable def aft3IOvT (ND BN NC T : Nat) (hTB : T * BN = NC) (hBN : 0 < BN)
    (zs : Fin T → Fin (BN * ND) → ℝ) : TileIndex [NC, ND] → ℝ :=
  fun idx =>
    zs ⟨idx.1.val / BN, (Nat.div_lt_iff_lt_mul hBN).mpr (by rw [hTB]; exact idx.1.isLt)⟩
      (Lane2D.encode (⟨idx.1.val % BN, Nat.mod_lt _ hBN⟩, idx.2.1, PUnit.unit))
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

## Public theorem: `attention_fwd_triton3_case4_io_correctness`

<details><summary>docstring</summary>

```
/-- **The case-4 `⊨[R]` io headline (`INIT=0` cross-launch resume).** For every
rounding model `R`, the case-4
(`(END,INIT,SLIDING_WINDOW,COMPLEMENT) = (1,0,1,0)`) `_attn_fwd` surface
implements, on its `StreamMasked3DKernelIO₆ₓ₂` signature, the **ideal-ℝ
resume-seeded sliding-window online-softmax attention fold** over the six
streamed tiles: output lane `j = (i, d)` of `Out` holds the window-filtered
base-2 per-key-scale fold started from the loaded running state
(`attentionFwdTriton3Case4IOOutSpec` = the existing
`attentionFwdTriton3Case4OutSpecG` closed form restated on the streams, with
the seed `aft3IOseedT` read off streams 4/5/6), and row `i` of `M` holds the
raw `(m + log2 l).unbotD 0` finalize of that same fold
(`attentionFwdTriton3Case4IOMSpec` = `attentionFwdTriton3Case4MSpecG` on the
streams). Both output grids are the `.real` default, so at every `R` the
terminal cells carry the exact fold values.

**Aliasing** (the point of this exemplar): `inp6 = out1 = Out` and
`inp4 = out2 = M`, so the skin's region list repeats two names. See the
section note above for why that is sound — one name, one segment; the input
pins read `s₀` while the readbacks read `s'`; the frame still says only the
two write windows moved. `L` (`inp5`) is read-only in case 4, its store
living in the dead `END=0` branch.

**Hypothesis provenance** (inherited verbatim from the exact case-4 headline
`attention_fwd_triton3_python_case4_output_summary_general`, all
truth-forced): `0 < ND/BM/BN/NKV_CTX` and `BN ∣ NKV_CTX` shape the KV walk
(`T = NKV_CTX / BN` full blocks); `0 < H`, `H_KV = H` and the six stride
equalities collapse the four plane offsets onto the shared base
`pid₁/H·sqz + pid₁%H·sqh`; `M ≠ Out` keeps the `O` store off the `M` row;
`hinjO`/`hinjM` (∀-pids forms of the exact stack's per-program injectivity)
make the scatter readbacks well-defined. The exact headline's `hundef` is not
a hypothesis — the skin's Hoare triple carries the `undef` pin itself.

**Scope**: with this headline all **four** constexpr cases of the file carry
`⊨[R]` io faces. The port's known fidelity gaps (`IS_EVEN_N = 1` hardwired,
casts erased to identity, `@triton.heuristics` not modeled) are inherited
as-is from the surface. -/
```
</details>

**Statement:**
```lean
specification attention_fwd_triton3_case4_io_correctness (R : RoundingModel)
    (Q K V M Out L : RegionName) (sm_scale : ℝ)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      Z H H_KV N_CTX ROUND_CTX NKV_CTX off size BM ND BN : Nat)
    (hND : 0 < ND) (hBM : 0 < BM) (hBN : 0 < BN) (hNC : 0 < NKV_CTX) (hBNdvd : BN ∣ NKV_CTX)
    (hH : 0 < H) (hHKV : H_KV = H)
    (hskz : skz = sqz) (hskh : skh = sqh) (hsvz : svz = sqz) (hsvh : svh = sqh)
    (hsoz : soz = sqz) (hsoh : soh = sqh)
    (hMO : M ≠ Out)
    (hinjO : ∀ p₀ p₁ : Nat, Function.Injective
      (fun idx : TileIndex [BM, ND] =>
        (p₁ / H * sqz + p₁ % H * sqh) + (p₀ * BM + idx.1.val) * som + idx.2.1.val * son))
    (hinjM : ∀ p₀ p₁ : Nat, Function.Injective
      (fun r : TileIndex [BM] => p₁ * ROUND_CTX + (p₀ * BM + r.1.val))) :
    attentionFwdTriton3Case4IO Q K V M Out L sm_scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son Z H H_KV N_CTX ROUND_CTX NKV_CTX off size BM ND BN ⊨[R]
      fun p₀ _ _ x1s x2s x3s x4s x5s x6s =>
        (fun j => attentionFwdTriton3Case4IOOutSpec p₀ BM ND BN NKV_CTX (NKV_CTX / BN) hND
            (Nat.div_pos (Nat.le_of_dvd hNC hBNdvd) hBN) (Nat.div_mul_cancel hBNdvd) hBN
            (sm_scale * 1.4426950408889634) off size x1s x2s x3s x4s x5s x6s j,
         fun i => attentionFwdTriton3Case4IOMSpec p₀ BM ND BN NKV_CTX (NKV_CTX / BN) hND
            (Nat.div_pos (Nat.le_of_dvd hNC hBNdvd) hBN) (Nat.div_mul_cancel hBNdvd) hBN
            (sm_scale * 1.4426950408889634) off size x1s x2s x3s x4s x5s x6s i)
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

**Closed-form spec defs (transitive):** `attentionFwdTriton3Case4IO`, `attentionFwdTriton3Case4IOOutSpec`, `attentionFwdTriton3Case4IOMSpec`, `attention_fwd_triton3_surface`, `aft3StateSeededG`, `aft3IOqT`, `aft3IOkT`, `aft3IOvT`, `keyScale3G`, `natSlidingWindowKeepG`, `aft3IOseedT`, `aft3KeysUptoG`, `aft3OsStepBot`, `natDist3G`

<details><summary><code>attentionFwdTriton3Case4IO</code></summary>

```
/-- **Streaming IO signature** of the case-4 (`(END,INIT,SW,CSW) = (1,0,1,0)`)
`_attn_fwd` surface on the six-stream two-output attention fold skin. Channels
4/5/6 are the `M`/`L`/`Out` resume rows; `out1 = Out = inp6` and
`out2 = M = inp4` (see the section note on aliasing). -/
```
```lean
def attentionFwdTriton3Case4IO (Q K V M Out L : RegionName) (sm_scale : ℝ)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      Z H H_KV N_CTX ROUND_CTX NKV_CTX off size BM ND BN : Nat) :
    StreamMasked3DKernelIO₆ₓ₂ where
  kernel := attention_fwd_triton3_surface Q K V M Out L sm_scale
    sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
    Z H H_KV N_CTX ROUND_CTX NKV_CTX off size 1 1 BM ND BN 1 0 1 0
  inp1 := Q
  inp2 := K
  inp3 := V
  inp4 := M
  inp5 := L
  inp6 := Out
  out1 := Out
  out2 := M
  T := NKV_CTX / BN
  B1 := BM * ND
  B2 := ND * BN
  B3 := BN * ND
  B4 := BM
  B5 := BM
  B6 := BM * ND
  C1 := BM * ND
  C2 := BM
  read1 := fun p₀ p₁ _ _ j =>
    (p₁ / H * sqz + p₁ % H * sqh) + (p₀ * BM + j.val / ND) * sqm + (j.val % ND) * sqk
  read2 := fun _ p₁ _ t j =>
    (p₁ / H * sqz + p₁ % H * sqh) + (j.val / BN) * skk + (t.val * BN + j.val % BN) * skn
  read3 := fun _ p₁ _ t j =>
    (p₁ / H * sqz + p₁ % H * sqh) + (t.val * BN + j.val / ND) * svk + (j.val % ND) * svn
  read4 := fun p₀ p₁ _ _ j => p₁ * ROUND_CTX + (p₀ * BM + j.val)
  read5 := fun p₀ p₁ _ _ j => p₁ * ROUND_CTX + (p₀ * BM + j.val)
  read6 := fun p₀ p₁ _ _ j =>
    (p₁ / H * sqz + p₁ % H * sqh) + (p₀ * BM + j.val / ND) * som + (j.val % ND) * son
  write1 := fun p₀ p₁ _ j =>
    (p₁ / H * sqz + p₁ % H * sqh) + (p₀ * BM + j.val / ND) * som + (j.val % ND) * son
  write2 := fun p₀ p₁ _ j => p₁ * ROUND_CTX + (p₀ * BM + j.val)
  mask1 := fun _ _ _ _ _ => True
  mask2 := fun _ _ _ _ _ => True
  mask3 := fun _ _ _ _ _ => True
  mask4 := fun _ _ _ _ _ => True
  mask5 := fun _ _ _ _ _ => True
  mask6 := fun _ _ _ _ _ => True
  writeMask1 := fun _ _ _ _ => True
  writeMask2 := fun _ _ _ _ => True
```
</details>

<details><summary><code>attentionFwdTriton3Case4IOOutSpec</code></summary>

```
/-- **Case-4 `Out` closed form on the streams**: the resume-seeded
sliding-window online-softmax fold (exactly the exact headline's
`attentionFwdTriton3Case4OutSpecG`, keep `natSlidingWindowKeepG`) restated over
the six streamed tiles, at output lane `j = (i, d)` row-major over `[BM, ND]`.
The seed is `aft3IOseedT` — the prior running state read off streams 4/5/6. -/
```
```lean
noncomputable def attentionFwdTriton3Case4IOOutSpec (p₀ BM ND BN NC T : Nat)
    (hND : 0 < ND) (hT : 0 < T) (hTB : T * BN = NC) (hBN : 0 < BN) (sc : ℝ)
    (off size : Nat)
    (x1s : Fin T → Fin (BM * ND) → ℝ) (x2s : Fin T → Fin (ND * BN) → ℝ)
    (x3s : Fin T → Fin (BN * ND) → ℝ) (x4s x5s : Fin T → Fin BM → ℝ)
    (x6s : Fin T → Fin (BM * ND) → ℝ) (j : Fin (BM * ND)) : ℝ :=
  (aft3StateSeededG (aft3IOqT BM ND T hT x1s) (aft3IOkT ND BN NC T hTB hBN x2s)
      (aft3IOvT ND BN NC T hTB hBN x3s) (keyScale3G sc NC)
      (fun i j => natSlidingWindowKeepG p₀ BM BN off size i j)
      (aft3IOseedT BM ND T hT x4s x5s x6s) NC
      (Lane2D.decode j).1 (Lane2D.decode j).2.1).2.2
    / (aft3StateSeededG (aft3IOqT BM ND T hT x1s) (aft3IOkT ND BN NC T hTB hBN x2s)
      (aft3IOvT ND BN NC T hTB hBN x3s) (keyScale3G sc NC)
      (fun i j => natSlidingWindowKeepG p₀ BM BN off size i j)
      (aft3IOseedT BM ND T hT x4s x5s x6s) NC
      (Lane2D.decode j).1 ⟨0, hND⟩).2.1
```
</details>

<details><summary><code>attentionFwdTriton3Case4IOMSpec</code></summary>

```
/-- **Case-4 `M` closed form on the streams**: the raw
`(m + log2 l).unbotD 0` finalize of the resume-seeded fold (exactly the exact
headline's `attentionFwdTriton3Case4MSpecG`) restated over the six streamed
tiles, at output row `i`. -/
```
```lean
noncomputable def attentionFwdTriton3Case4IOMSpec (p₀ BM ND BN NC T : Nat)
    (hND : 0 < ND) (hT : 0 < T) (hTB : T * BN = NC) (hBN : 0 < BN) (sc : ℝ)
    (off size : Nat)
    (x1s : Fin T → Fin (BM * ND) → ℝ) (x2s : Fin T → Fin (ND * BN) → ℝ)
    (x3s : Fin T → Fin (BN * ND) → ℝ) (x4s x5s : Fin T → Fin BM → ℝ)
    (x6s : Fin T → Fin (BM * ND) → ℝ) (i : Fin BM) : ℝ :=
  (WithBot.realAdd
      (aft3StateSeededG (aft3IOqT BM ND T hT x1s) (aft3IOkT ND BN NC T hTB hBN x2s)
        (aft3IOvT ND BN NC T hTB hBN x3s) (keyScale3G sc NC)
        (fun i j => natSlidingWindowKeepG p₀ BM BN off size i j)
        (aft3IOseedT BM ND T hT x4s x5s x6s) NC i ⟨0, hND⟩).1
      (WithBot.realLog2
        (((aft3StateSeededG (aft3IOqT BM ND T hT x1s) (aft3IOkT ND BN NC T hTB hBN x2s)
            (aft3IOvT ND BN NC T hTB hBN x3s) (keyScale3G sc NC)
            (fun i j => natSlidingWindowKeepG p₀ BM BN off size i j)
            (aft3IOseedT BM ND T hT x4s x5s x6s) NC i ⟨0, hND⟩).2.1 : ℝ)
          : WithBot ℝ))).unbotD 0
```
</details>

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

<details><summary><code>aft3IOqT</code></summary>

```
/-- The `Q` tile read off the (static) first stream: the window ignores `t`,
so the step-`0` slice carries the whole tile. -/
```
```lean
noncomputable def aft3IOqT (BM ND T : Nat) (hT : 0 < T)
    (xs : Fin T → Fin (BM * ND) → ℝ) : TileIndex [BM, ND] → ℝ :=
  fun idx => xs ⟨0, hT⟩ (Lane2D.encode idx)
```
</details>

<details><summary><code>aft3IOkT</code></summary>

```
/-- The global `K` tile read off the second stream: global key `j` lives in
step `j / BN`, block-local column `j % BN`, at stream lane `(e, j % BN)`. -/
```
```lean
noncomputable def aft3IOkT (ND BN NC T : Nat) (hTB : T * BN = NC) (hBN : 0 < BN)
    (ys : Fin T → Fin (ND * BN) → ℝ) : TileIndex [NC, ND] → ℝ :=
  fun idx =>
    ys ⟨idx.1.val / BN, (Nat.div_lt_iff_lt_mul hBN).mpr (by rw [hTB]; exact idx.1.isLt)⟩
      (Lane2D.encode (idx.2.1, ⟨idx.1.val % BN, Nat.mod_lt _ hBN⟩, PUnit.unit))
```
</details>

<details><summary><code>aft3IOvT</code></summary>

```
/-- The global `V` tile read off the third stream: global key `j` lives in
step `j / BN`, block-local row `j % BN`, at stream lane `(j % BN, d)`. -/
```
```lean
noncomputable def aft3IOvT (ND BN NC T : Nat) (hTB : T * BN = NC) (hBN : 0 < BN)
    (zs : Fin T → Fin (BN * ND) → ℝ) : TileIndex [NC, ND] → ℝ :=
  fun idx =>
    zs ⟨idx.1.val / BN, (Nat.div_lt_iff_lt_mul hBN).mpr (by rw [hTB]; exact idx.1.isLt)⟩
      (Lane2D.encode (⟨idx.1.val % BN, Nat.mod_lt _ hBN⟩, idx.2.1, PUnit.unit))
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

<details><summary><code>aft3IOseedT</code></summary>

```
/-- The resume seed read off the (static) streams 4/5/6: row `i` takes its
running max from the `M` stream, its running denominator from the `L` stream,
and lane `(i, d)` its running accumulator from the `Out` stream. All three
windows ignore `t`, so the step-`0` slice carries the whole seed. -/
```
```lean
noncomputable def aft3IOseedT (BM ND T : Nat) (hT : 0 < T)
    (x4s x5s : Fin T → Fin BM → ℝ) (x6s : Fin T → Fin (BM * ND) → ℝ) :
    Fin BM → Fin ND → WithBot ℝ × ℝ × ℝ :=
  fun i d =>
    (((x4s ⟨0, hT⟩ i : ℝ) : WithBot ℝ), x5s ⟨0, hT⟩ i,
      x6s ⟨0, hT⟩ (Lane2D.encode (i, d, PUnit.unit)))
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
