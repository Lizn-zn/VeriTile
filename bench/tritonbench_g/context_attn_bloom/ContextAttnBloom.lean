import VeriTile.Triton

/-!
# `context_attn_bloom` — strict per-kernel correctness

`_fwd_kernel` is BLOOM-style varlen context (prefill) attention. Each program
`(cur_batch, cur_head, start_m)` loads a `[BLOCK_M, BLOCK_DMODEL]` query tile,
streams over the cached key/value tokens gathered through `Req_to_tokens`,
runs an online-softmax (`m_i`/`l_i`/`acc`) loop with a plain causal mask offset
by `prompt_cache_len` (`offs_m + prompt_cache_len ≥ start_n + offs_n`; there is
no ALiBi slope or positional bias anywhere in this kernel), and stores the
accumulated `acc` tile back to `Out`, masked by `offs_m < cur_batch_seq_len`
and `offs_d < head_dim`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`_fwd_kernel[grid](...)` with
`grid = (batch, head, cdiv(max_input_len, BLOCK))`, the scheduling over batch /
head / sequence blocks, and how the runtime composes per-program writes into one
buffer) is the *trusted boundary*, not a proof obligation here. Because the
program ids `(cur_batch, cur_head, start_m)` are universally quantified, the
per-program statement covers every program of the grid.

## Proof architecture

```
context_attn_bloom_surface_compute_correct_general        ← TOP THEOREM (symbolic, dimension-general)
  ├─ (surface lowers to the algorithm layer — discharged inline)
  └─ bloom_exec_general                                     genuine streaming-softmax exec, symbolic dims
       ├─ bloomPreLoopG_eval / bloom_attn_stepG / bloomPostLoopG_eval   online-softmax fold steps
       └─ bloomFwdGenuineOutValueG (= contextAttnBloomExactFoldMG)      genuine closed form over loaded Q/K/V
(honest side-condition: output-offset injectivity `hOInj`)
```

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); `@triton.autotune` and
`num_warps` are not modeled. The verified compute claim is scoped to the **final
masked writeback** of the accumulated `acc` tile into `Out`: every active lane
(`offs_m < cur_batch_seq_len ∧ offs_d < head_dim`) holds the surface-produced
`acc` value (`bloomFwdGenuineOutValueG`), and out-of-bounds lanes are
preserved. This is the genuine closed-form block-causal-guarded online-softmax
fold (`contextAttnBloomExactFoldMG`) of the loaded Q/K/V memory: the streaming
loop (`m_i`/`l_i`/`acc` updates, `tl.dot`, the `Req_to_tokens` gathers, and the
`prompt_cache_len`-offset causal mask) is proven to realize that fold via the
whole-kernel exec chain (`bloom_exec_general`). The top theorem is
dimension-general: it is stated over symbolic `BLOCK_M`/`BLOCK_N`/`BLOCK_DMODEL`/
`head_dim` and the per-axis strides. The Python test shape (`head_dim=96`,
`BLOCK_DMODEL=BLOCK_N=128`, `BLOCK_M ∈ {128, 64}`) is recovered as a concrete
special case.
-/

namespace VeriTile.Bench.TritonBenchG.ContextAttnBloom

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-! **★ Main theorem:** `context_attn_bloom_surface_compute_correct_general` -/

/-! # ══════════ CORRECT — genuine / dimension-general (review this) ══════════ -/

section Correct_without_Rounding

/-- Faithful DSL port of `context_attn_bloom.py`'s `_fwd_kernel`. -/
def context_attn_bloom_fwd_kernel_surface
    (Q K V : RegionName) (sm_scale : ℝ)
    (B_Start_Loc B_Seqlen : Region .nat) (Out : RegionName)
    (Req_to_tokens B_req_idx b_prompt_cache_len : Region .nat)
    (stride_qbs stride_qh stride_qd
      stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd
      stride_obs stride_oh stride_od
      stride_req_to_tokens_b stride_req_to_tokens_s
      kv_group_num head_dim BLOCK_M BLOCK_DMODEL BLOCK_N : Nat) :
    ComputeKernel := triton {
  cur_batch = tl.program_id(0)
  cur_head = tl.program_id(1)
  start_m = tl.program_id(2)

  cur_kv_head = cur_head // $(kv_group_num)

  cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
  prompt_cache_len = tl.load(b_prompt_cache_len + cur_batch)
  cur_batch_seq_len = tl.load(B_Seqlen + cur_batch) - prompt_cache_len
  cur_batch_req_idx = tl.load(B_req_idx + cur_batch)

  block_start_loc = $(BLOCK_M) * start_m

  offs_n = tl.arange(0, $(BLOCK_N))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  off_q = (cur_batch_in_all_start_index + offs_m[:, None]) * $(stride_qbs) +
    cur_head * $(stride_qh) + offs_d[None, :] * $(stride_qd)

  q = tl.load(Q + off_q,
    mask=(offs_m[:, None] < cur_batch_seq_len) & (offs_d[None, :] < $(head_dim)),
    other=0.0)

  m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)

  block_mask = tl.where(block_start_loc < cur_batch_seq_len, $(1), $(0))
  block_end_loc = tl.minimum((start_m + $(1)) * $(BLOCK_M) + prompt_cache_len,
    cur_batch_seq_len + prompt_cache_len)

  for start_n in range($(0), block_mask * block_end_loc, $(BLOCK_N)) {
    start_n = tl.multiple_of(start_n, $(BLOCK_N))
    kv_loc = tl.load(Req_to_tokens + $(stride_req_to_tokens_b) * cur_batch_req_idx +
      $(stride_req_to_tokens_s) * (start_n + offs_n),
      mask=(start_n + offs_n) < block_end_loc,
      other=0)
    off_k = kv_loc[None, :] * $(stride_kbs) + cur_kv_head * $(stride_kh) +
      offs_d[:, None] * $(stride_kd)
    k = tl.load(K + off_k,
      mask=((start_n + offs_n[None, :]) < block_end_loc) &
        (offs_d[:, None] < $(head_dim)),
      other=0.0)

    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    qk += tl.dot(q, k)
    qk *= $((sm_scale : ℝ))
    qk = tl.where(offs_m[:, None] + prompt_cache_len >= start_n + offs_n[None, :],
      qk, -100000000.0)

    m_ij = tl.max(qk, 1)
    p = tl.exp(qk - m_ij[:, None])
    l_ij = tl.sum(p, 1)
    m_i_new = tl.maximum(m_i, m_ij)
    alpha = tl.exp(m_i - m_i_new)
    beta = tl.exp(m_ij - m_i_new)
    l_i_new = alpha * l_i + beta * l_ij
    p_scale = beta / l_i_new
    p = p * p_scale[:, None]
    acc_scale = l_i / l_i_new * alpha
    acc_scale = tl.where(offs_m + prompt_cache_len >= start_n, acc_scale, 1.0)
    acc = acc * acc_scale[:, None]
    off_v = kv_loc[:, None] * $(stride_vbs) + cur_kv_head * $(stride_vh) +
      offs_d[None, :] * $(stride_vd)
    v = tl.load(V + off_v,
      mask=((start_n + offs_n[:, None]) < block_end_loc) &
        (offs_d[None, :] < $(head_dim)),
      other=0.0)
    p = (p).to(v.dtype)
    acc += tl.dot(p, v)
    l_i = l_i_new
    m_i = m_i_new
  }
  off_o = (cur_batch_in_all_start_index + offs_m[:, None]) * $(stride_obs) +
    cur_head * $(stride_oh) + offs_d[None, :] * $(stride_od)
  out_ptrs = Out + off_o
  tl.store(out_ptrs, acc,
    mask=(offs_m[:, None] < cur_batch_seq_len) & (offs_d[None, :] < $(head_dim)))
}

def promptLen (s : BlockState) (B_Prompt_Cache_Len : RegionName) : Nat :=
  s.readMemValue .nat B_Prompt_Cache_Len (s.pids 0)

def seqLen (s : BlockState) (B_Seqlen B_Prompt_Cache_Len : RegionName) : Nat :=
  s.readMemValue .nat B_Seqlen (s.pids 0) - promptLen s B_Prompt_Cache_Len

def startLoc (s : BlockState) (B_Start_Loc : RegionName) : Nat :=
  s.readMemValue .nat B_Start_Loc (s.pids 0)

def mIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 2 * BLOCK_M + i.val

def dIndex (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  idx.2.1.val

def active
    (s : BlockState) (B_Seqlen B_Prompt_Cache_Len : RegionName)
    (head_dim BLOCK_M : Nat) (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Prop :=
  mIndex s BLOCK_M idx.1 < seqLen s B_Seqlen B_Prompt_Cache_Len ∧
    dIndex idx < head_dim

instance activeDecidable
    (s : BlockState) (B_Seqlen B_Prompt_Cache_Len : RegionName)
    (head_dim BLOCK_M BLOCK_DMODEL : Nat) (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) :
    Decidable (active s B_Seqlen B_Prompt_Cache_Len head_dim BLOCK_M idx) := by
  unfold active
  infer_instance

def outOffset
    (s : BlockState) (B_Start_Loc : RegionName)
    (stride_obs stride_oh stride_od BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  (startLoc s B_Start_Loc + mIndex s BLOCK_M idx.1) * stride_obs +
    s.pids 1 * stride_oh + dIndex idx * stride_od

/-! ## Genuine closed-form context-attention spec (no self-reference)

The streaming-softmax loop in `_fwd_kernel` is *not* a self-referential black
box: it computes, for every active query lane, the prompt-cache-offset **causal
softmax attention** value over the request-gathered key/value tokens. Unlike the
`context_attn_fwd` PPL kernel (which uses base-2 `exp2` with a `log₂e`-scaled
`sm_scale` and a deferred final `acc /= l_i`), BLOOM uses **natural** `exp`
directly with `sm_scale = (√D)⁻¹` and *in-loop* normalization (`p_scale =
β/l_iⁿᵉʷ`, `acc_scale = (l_i/l_iⁿᵉʷ)·α`, `acc += dot(p,v)`, no final divide). Both
variants of online softmax produce the *same* final `acc/l` ratio — the genuine
scaled-dot causal softmax. This section makes that closed form explicit as a pure
function of `Q`/`K`/`V` memory and proves it is the library's
`attentionRealCausalBlock` reference; no `exec`, no self-reference.

### Score / scale / mask of this kernel (decoded lane-by-lane)

For program `(cur_batch, cur_head, start_m)`, query lane `i` (global row
`gi = start_m·BLOCK_M + i`), gathered key `j`, head channel `e`:

* **raw score** `raw i j = Σ_e Q[gi,e]·K[kvloc j,e]`  (`tl.dot q k`, line 89);
* **scale**     `qk·sm_scale` with `sm_scale = (√D)⁻¹` (line 90);
* **softmax**   natural `tl.exp` (lines 95);
* **mask**      `gi + prompt_cache_len ≥ j`  (line 91): future keys get score
  `-1e8` (≈ `exp → 0`), a causal mask shifted by `prompt_cache_len`.

So the kernel realizes the **natural-exp** causal softmax with effective scale
`sm_scale` exactly (no base conversion needed). -/

/-- Head index of this kernel's program (`cur_head = pids 1`, `kv_group_num = 1`
so `cur_kv_head = cur_head`). -/
def curHead (s : BlockState) : Nat := s.pids 1

/-- Request index for this batch: `cur_batch_req_idx = B_req_idx[cur_batch]`. -/
def reqIdx (s : BlockState) (B_req_idx : RegionName) : Nat :=
  s.readMemValue .nat B_req_idx (s.pids 0)

/-- One ⊥-seeded online-softmax step: running max in `WithBot ℝ` (seeded `⊥`), so
`α = realExp2(m ⊖ m')` is `0` on the first block. -/
noncomputable def osStepBot (st : WithBot ℝ × ℝ × ℝ) (sv : ℝ × ℝ) : WithBot ℝ × ℝ × ℝ :=
  let m := st.1; let l := st.2.1; let acc := st.2.2
  let sc := sv.1; let v := sv.2
  let m' := m ⊔ ((sc : ℝ) : WithBot ℝ)
  let α := (WithBot.realExp2 (WithBot.realSub m m')).unbotD 0
  let p := pow2 (sc - m'.unbotD 0)
  (m', l * α + p, acc * α + p * v)

theorem osStepBot_foldl_fst
    (xs : List (ℝ × ℝ)) (m₀ : WithBot ℝ) (l₀ acc₀ : ℝ) :
    (xs.foldl osStepBot (m₀, l₀, acc₀)).1
      = (xs.map (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldl (· ⊔ ·) m₀ := by
  induction xs generalizing m₀ l₀ acc₀ with
  | nil => rfl
  | cons x xs ih => simp only [List.foldl_cons, List.map_cons]; rw [ih]; rfl

/-- **⊥-seeded consistency.** Folding `osStepBot` from `(m, l, acc)` anchored to the
true (max-free) denominator `L` / accumulator `T` via the ⊥-aware factor keeps that
invariant (`l = κ(m)·L`, `acc = κ(m)·T`, `κ ⊥ = 0`, `κ (some r) = pow2(−r)`). -/
theorem osStepBot_foldl_consistent (xs : List (ℝ × ℝ)) (m : WithBot ℝ) (l acc T L : ℝ)
    (hl : l = (m.elim 0 (fun r => pow2 (-r))) * L)
    (hacc : acc = (m.elim 0 (fun r => pow2 (-r))) * T)
    (hmL : m = ⊥ → L = 0) (hmT : m = ⊥ → T = 0) :
    let st := xs.foldl osStepBot (m, l, acc)
    st.2.1 = (st.1.elim 0 (fun r => pow2 (-r))) * (L + (xs.map (fun p => pow2 p.1)).sum) ∧
    st.2.2 = (st.1.elim 0 (fun r => pow2 (-r))) * (T + (xs.map (fun p => pow2 p.1 * p.2)).sum) := by
  induction xs generalizing m l acc T L with
  | nil => simp [hl, hacc]
  | cons x xs ih =>
    obtain ⟨sc, v⟩ := x
    set m' : WithBot ℝ := m ⊔ ((sc : ℝ) : WithBot ℝ) with hm'
    have hm'r : ∃ r : ℝ, m' = (r : WithBot ℝ) := by
      cases m with
      | bot => exact ⟨sc, by rw [hm']; rfl⟩
      | coe a => exact ⟨max a sc, by rw [hm']; rw [← WithBot.coe_max]⟩
    obtain ⟨mr, hmr⟩ := hm'r
    have hκm' : m'.elim 0 (fun r => pow2 (-r)) = pow2 (-mr) := by rw [hmr]; rfl
    have hunbot : m'.unbotD 0 = mr := by rw [hmr]; rfl
    have hp : pow2 (sc - m'.unbotD 0) = pow2 (-mr) * pow2 sc := by
      rw [hunbot, ← pow2_add]; ring_nf
    have hl' : l * (WithBot.realExp2 (WithBot.realSub m m')).unbotD 0
        + pow2 (sc - m'.unbotD 0) = pow2 (-mr) * (L + pow2 sc) := by
      cases m with
      | bot =>
        rw [hmL rfl]
        have hz : (WithBot.realExp2 (WithBot.realSub (⊥ : WithBot ℝ) m')).unbotD 0 = 0 := by
          rw [WithBot.realSub_bot_left, WithBot.realExp2_bot]; rfl
        rw [hz, mul_zero, zero_add, hp]; ring
      | coe a =>
        have hm'a : m' = ((max a sc : ℝ) : WithBot ℝ) := by rw [hm']; rw [← WithBot.coe_max]
        have hmra : mr = max a sc := by rw [hm'a] at hmr; exact (WithBot.coe_inj.mp hmr.symm)
        have hαa : (pow2 (-a)) * (WithBot.realExp2 (WithBot.realSub (↑a) m')).unbotD 0
            = pow2 (-mr) := by
          rw [hm'a, WithBot.realSub_coe_coe, WithBot.realExp2_coe, WithBot.unbotD_coe]
          rw [show Real.exp ((a - max a sc) * Real.log 2) = pow2 (a - max a sc) from by
            simp [pow2, mul_comm], ← pow2_add, hmra]; ring_nf
        rw [hl, show ((↑a : WithBot ℝ).elim 0 (fun r => pow2 (-r))) = pow2 (-a) from rfl]
        rw [mul_right_comm, hαa, hp]; ring
    have hacc' : acc * (WithBot.realExp2 (WithBot.realSub m m')).unbotD 0
        + pow2 (sc - m'.unbotD 0) * v = pow2 (-mr) * (T + pow2 sc * v) := by
      cases m with
      | bot =>
        rw [hmT rfl]
        have hz : (WithBot.realExp2 (WithBot.realSub (⊥ : WithBot ℝ) m')).unbotD 0 = 0 := by
          rw [WithBot.realSub_bot_left, WithBot.realExp2_bot]; rfl
        rw [hz, mul_zero, zero_add, hp]; ring
      | coe a =>
        have hm'a : m' = ((max a sc : ℝ) : WithBot ℝ) := by rw [hm']; rw [← WithBot.coe_max]
        have hmra : mr = max a sc := by rw [hm'a] at hmr; exact (WithBot.coe_inj.mp hmr.symm)
        have hαa : (pow2 (-a)) * (WithBot.realExp2 (WithBot.realSub (↑a) m')).unbotD 0
            = pow2 (-mr) := by
          rw [hm'a, WithBot.realSub_coe_coe, WithBot.realExp2_coe, WithBot.unbotD_coe]
          rw [show Real.exp ((a - max a sc) * Real.log 2) = pow2 (a - max a sc) from by
            simp [pow2, mul_comm], ← pow2_add, hmra]; ring_nf
        rw [hacc, show ((↑a : WithBot ℝ).elim 0 (fun r => pow2 (-r))) = pow2 (-a) from rfl]
        rw [mul_right_comm, hαa, hp]; ring
    have step := ih m'
      (l * (WithBot.realExp2 (WithBot.realSub m m')).unbotD 0 + pow2 (sc - m'.unbotD 0))
      (acc * (WithBot.realExp2 (WithBot.realSub m m')).unbotD 0 + pow2 (sc - m'.unbotD 0) * v)
      (T + pow2 sc * v) (L + pow2 sc) (by rw [hl', hκm']) (by rw [hacc', hκm'])
      (by rw [hmr]; simp) (by rw [hmr]; simp)
    simpa [List.foldl_cons, osStepBot, hm', List.map_cons, add_assoc] using step

/-- The `WithBot ⊔`-fold is seed/direction-agnostic. -/
theorem foldl_sup_bot_eq_foldr (L : List (WithBot ℝ)) :
    L.foldl (· ⊔ ·) (⊥ : WithBot ℝ) = L.foldr (· ⊔ ·) (⊥ : WithBot ℝ) := by
  have gen : ∀ (m : WithBot ℝ), L.foldl (· ⊔ ·) m = m ⊔ L.foldr (· ⊔ ·) ⊥ := by
    induction L with
    | nil => intro m; simp
    | cons a t ih =>
      intro m
      simp only [List.foldl_cons, List.foldr_cons, ih]
      rw [max_assoc]
  rw [gen ⊥, bot_sup_eq]

/-- Any list member is `≤` the `foldr ⊔ ⊥`. -/
theorem mem_le_foldr_sup (a : WithBot ℝ) :
    ∀ (L : List (WithBot ℝ)), a ∈ L → a ≤ L.foldr (· ⊔ ·) ⊥ := by
  intro L
  induction L with
  | nil => intro h; simp at h
  | cons x t ih =>
    intro h
    simp only [List.foldr_cons]
    rcases List.mem_cons.mp h with h | h
    · rw [h]; exact le_sup_left
    · exact le_trans (ih h) le_sup_right

/-- Generic windowed key list `[0, hi)` over an abstract per-key `g`. -/
noncomputable def gKeysUpto (S hi : Nat) (g : Fin S → ℝ × ℝ) : List (ℝ × ℝ) :=
  (List.finRange S).filterMap (fun j : Fin S => if j.val < hi then some (g j) else none)

/-- Generic block-`c` key list (keys `c·BN ≤ j < (c+1)·BN`). -/
noncomputable def gBlock (S BN c : Nat) (g : Fin S → ℝ × ℝ) : List (ℝ × ℝ) :=
  (List.finRange S).filterMap (fun j : Fin S =>
    if c * BN ≤ j.val ∧ j.val < (c + 1) * BN then some (g j) else none)

/-- Generic ⊥-seeded running `(max, l, acc)` after streaming `[0, hi)`. -/
noncomputable def gStateBot (S hi : Nat) (g : Fin S → ℝ × ℝ) : WithBot ℝ × ℝ × ℝ :=
  (gKeysUpto S hi g).foldl osStepBot (⊥, 0, 0)

/-- Generic ⊥-seeded running max after streaming `[0, hi)`. -/
noncomputable def gRunningMax (S hi : Nat) (g : Fin S → ℝ × ℝ) : WithBot ℝ :=
  ((gKeysUpto S hi g).map (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥

/-- The running max component of `gStateBot` is the `WithBot ⊔`-fold `gRunningMax`. -/
theorem gStateBot_fst_eq_runningMax (S hi : Nat) (g : Fin S → ℝ × ℝ) :
    (gStateBot S hi g).1 = gRunningMax S hi g := by
  rw [gStateBot, osStepBot_foldl_fst, gRunningMax, foldl_sup_bot_eq_foldr]

/-- Threshold-split for a `.val`-ascending `Fin` list. -/
private theorem g_filterMap_window_split {n : Nat} (l : List (Fin n))
    (hsorted : l.Pairwise (fun a b => a.val < b.val))
    (t hi₂ : Nat) (g : Fin n → ℝ × ℝ) (hle : t ≤ hi₂) :
    l.filterMap (fun j => if j.val < hi₂ then some (g j) else none)
      = l.filterMap (fun j => if j.val < t then some (g j) else none)
        ++ l.filterMap (fun j => if t ≤ j.val ∧ j.val < hi₂ then some (g j) else none) := by
  induction l with
  | nil => simp
  | cons a tl ih =>
    have htl : tl.Pairwise (fun x y => x.val < y.val) := (List.pairwise_cons.mp hsorted).2
    have hahead : ∀ b ∈ tl, a.val < b.val := (List.pairwise_cons.mp hsorted).1
    rw [List.filterMap_cons, List.filterMap_cons, List.filterMap_cons]
    by_cases hlt : a.val < t
    · rw [ih htl]
      have hnb : ¬ (t ≤ a.val ∧ a.val < hi₂) := fun h => (Nat.not_le.mpr hlt) h.1
      rw [if_neg hnb, if_pos (lt_of_lt_of_le hlt hle), if_pos hlt]; rfl
    · have hge : t ≤ a.val := Nat.not_lt.mp hlt
      have htail_prefix : tl.filterMap (fun j => if j.val < t then some (g j) else none) = [] := by
        apply List.filterMap_eq_nil_iff.mpr
        intro b hb
        have hab : a.val < b.val := hahead b hb
        have hbt : ¬ (b.val < t) := by omega
        simp [hbt]
      rw [ih htl, htail_prefix, if_neg hlt]
      by_cases h2 : a.val < hi₂
      · rw [if_pos h2, if_pos (And.intro hge h2 : t ≤ a.val ∧ a.val < hi₂)]; rfl
      · rw [if_neg h2, if_neg (fun h : t ≤ a.val ∧ a.val < hi₂ => h2 h.2)]

/-- **Window split** (`hi = c·BN`) for the generic key list. -/
theorem gKeysUpto_succ (S BN c : Nat) (g : Fin S → ℝ × ℝ) :
    gKeysUpto S ((c + 1) * BN) g = gKeysUpto S (c * BN) g ++ gBlock S BN c g := by
  unfold gKeysUpto gBlock
  rw [g_filterMap_window_split (List.finRange S) (List.pairwise_lt_finRange S)
    (c * BN) ((c + 1) * BN) g (by nlinarith [Nat.zero_le BN])]

/-- **One-block advance** of the generic ⊥-seeded state. -/
theorem gStateBot_succ (S BN c : Nat) (g : Fin S → ℝ × ℝ) :
    gStateBot S ((c + 1) * BN) g
      = (gBlock S BN c g).foldl osStepBot (gStateBot S (c * BN) g) := by
  unfold gStateBot; rw [gKeysUpto_succ, List.foldl_append]

/-- The running max of an `osStepBot` fold over `block` from `(m, l, acc)` is
`m ⊔ (block max)`. -/
theorem osStepBot_block_fst (m : WithBot ℝ) (l acc : ℝ) (block : List (ℝ × ℝ)) :
    (block.foldl osStepBot (m, l, acc)).1
      = m ⊔ (block.map (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥ := by
  rw [osStepBot_foldl_fst]
  induction block generalizing m with
  | nil => simp
  | cons a t ih =>
    simp only [List.map_cons, List.foldl_cons, List.foldr_cons]
    rw [ih]
    rw [show (m ⊔ ((a.1 : ℝ) : WithBot ℝ)) ⊔ (List.foldr (· ⊔ ·) ⊥ (List.map (fun p => ((p.1 : ℝ) : WithBot ℝ)) t))
          = m ⊔ (((a.1 : ℝ) : WithBot ℝ) ⊔ (List.foldr (· ⊔ ·) ⊥ (List.map (fun p => ((p.1 : ℝ) : WithBot ℝ)) t))) from by
      rw [sup_assoc]]

/-- **The block-at-once update equals the key-by-key `osStepBot` fold.** -/
theorem osStepBot_block_eq (m : WithBot ℝ) (l acc T L : ℝ) (block : List (ℝ × ℝ))
    (hl : l = (m.elim 0 (fun r => pow2 (-r))) * L)
    (hacc : acc = (m.elim 0 (fun r => pow2 (-r))) * T)
    (hmL : m = ⊥ → L = 0) (hmT : m = ⊥ → T = 0) :
    let M' := m ⊔ (block.map (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥
    (M',
     l * (WithBot.realExp2 (WithBot.realSub m M')).unbotD 0
       + (block.map (fun p => pow2 (p.1 - M'.unbotD 0))).sum,
     acc * (WithBot.realExp2 (WithBot.realSub m M')).unbotD 0
       + (block.map (fun p => pow2 (p.1 - M'.unbotD 0) * p.2)).sum)
      = block.foldl osStepBot (m, l, acc) := by
  intro M'
  have hfst : (block.foldl osStepBot (m, l, acc)).1 = M' := by
    rw [osStepBot_block_fst]
  obtain ⟨hfold_l, hfold_acc⟩ := osStepBot_foldl_consistent block m l acc T L hl hacc hmL hmT
  rw [hfst] at hfold_l hfold_acc
  have hM'eq : M' = m ⊔ (block.map (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥ := rfl
  cases hM' : M' with
  | bot =>
    have hempty : block = [] := by
      rcases block with _ | ⟨a, t⟩
      · rfl
      · exfalso
        have : ((a.1 : ℝ) : WithBot ℝ) ≤ M' := by
          rw [hM'eq]
          exact le_sup_of_le_right (by simp only [List.map_cons, List.foldr_cons]; exact le_sup_left)
        rw [hM'] at this
        exact absurd (le_bot_iff.mp this) (WithBot.coe_ne_bot)
    have hm0 : m = ⊥ := by
      rw [hM'eq, hempty] at hM'
      simpa only [List.map_nil, List.foldr_nil, sup_bot_eq] using hM'
    have hl0 : l = 0 := by rw [hl, hm0]; simp [hmL hm0]
    have hacc0 : acc = 0 := by rw [hacc, hm0]; simp [hmT hm0]
    subst hempty
    rw [hl0, hacc0]
    simp only [List.foldl_nil, List.map_nil, List.sum_nil, add_zero, mul_zero, zero_mul]
    rw [hm0]
  | coe Mr =>
    rw [hM'] at hfst hfold_l hfold_acc
    have hlα : l * (WithBot.realExp2 (WithBot.realSub m (↑Mr : WithBot ℝ))).unbotD 0 = pow2 (-Mr) * L := by
      cases hm : m with
      | bot =>
        rw [hl, hm, show ((⊥ : WithBot ℝ).elim 0 (fun r => pow2 (-r))) = 0 from rfl,
          zero_mul, hmL hm]; ring
      | coe a =>
        rw [hl, hm, show ((↑a : WithBot ℝ).elim 0 (fun r => pow2 (-r))) = pow2 (-a) from rfl]
        rw [WithBot.realSub_coe_coe, WithBot.realExp2_coe, WithBot.unbotD_coe,
          show Real.exp ((a - Mr) * Real.log 2) = pow2 (a - Mr) from by simp [pow2, mul_comm]]
        rw [mul_right_comm, ← pow2_add]; ring_nf
    have haccα : acc * (WithBot.realExp2 (WithBot.realSub m (↑Mr : WithBot ℝ))).unbotD 0 = pow2 (-Mr) * T := by
      cases hm : m with
      | bot =>
        rw [hacc, hm, show ((⊥ : WithBot ℝ).elim 0 (fun r => pow2 (-r))) = 0 from rfl,
          zero_mul, hmT hm]; ring
      | coe a =>
        rw [hacc, hm, show ((↑a : WithBot ℝ).elim 0 (fun r => pow2 (-r))) = pow2 (-a) from rfl]
        rw [WithBot.realSub_coe_coe, WithBot.realExp2_coe, WithBot.unbotD_coe,
          show Real.exp ((a - Mr) * Real.log 2) = pow2 (a - Mr) from by simp [pow2, mul_comm]]
        rw [mul_right_comm, ← pow2_add]; ring_nf
    have hsumL : (block.map (fun p => pow2 (p.1 - (↑Mr : WithBot ℝ).unbotD 0))).sum
        = pow2 (-Mr) * (block.map (fun p => pow2 p.1)).sum := by
      have := sum_map_pow2_sub ((↑Mr : WithBot ℝ).unbotD 0) block (fun _ => 1)
      simp only [mul_one] at this
      rw [this, WithBot.unbotD_coe]
    have hsumT : (block.map (fun p => pow2 (p.1 - (↑Mr : WithBot ℝ).unbotD 0) * p.2)).sum
        = pow2 (-Mr) * (block.map (fun p => pow2 p.1 * p.2)).sum := by
      rw [sum_map_pow2_sub ((↑Mr : WithBot ℝ).unbotD 0) block (fun p => p.2), WithBot.unbotD_coe]
    refine Prod.ext hfst.symm (Prod.ext ?_ ?_)
    · rw [hfold_l, hlα, hsumL, show ((↑Mr : WithBot ℝ).elim 0 (fun r => pow2 (-r))) = pow2 (-Mr) from rfl]; ring
    · rw [hfold_acc, haccα, hsumT, show ((↑Mr : WithBot ℝ).elim 0 (fun r => pow2 (-r))) = pow2 (-Mr) from rfl]; ring

/-- filterMap-sum over `Fin n` with a guard collapses into the masked `Finset.sum`. -/
theorem g_filterMap_finRange_sum {α : Type*} (n : Nat)
    (p : Fin n → Prop) [DecidablePred p] (g : Fin n → α) (h : α → ℝ) :
    (((List.finRange n).filterMap (fun j => if p j then some (g j) else none)).map h).sum
      = ∑ j : Fin n, if p j then h (g j) else 0 := by
  rw [List.map_filterMap]
  rw [show (fun j : Fin n => Option.map h (if p j then some (g j) else none))
        = (fun j : Fin n => if p j then some (h (g j)) else none) from by
    funext j; by_cases hj : p j <;> simp [hj]]
  rw [show (((List.finRange n).filterMap (fun j => if p j then some (h (g j)) else none))).sum
        = ((List.finRange n).map (fun j => if p j then h (g j) else 0)).sum from by
    induction (List.finRange n) with
    | nil => simp
    | cons a t ih => by_cases ha : p a <;> simp [ha, ih]]
  rw [← List.sum_ofFn]; congr 1; rw [List.ofFn_eq_map]

/-- The `WithBot` `foldr` of a filtered score list (coerced) equals the `Finset.sup`. -/
theorem g_filterMap_foldr_sup (n : Nat) (P : Fin n → Prop) [DecidablePred P]
    (sc : Fin n → ℝ) :
    (((List.finRange n).filterMap (fun j => if P j then some (sc j) else none)).map
        (fun x => ((x : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥
      = Finset.univ.sup (fun j : Fin n => if P j then ((sc j : ℝ) : WithBot ℝ) else ⊥) := by
  rw [show (((List.finRange n).filterMap (fun j => if P j then some (sc j) else none)).map
        (fun x => ((x : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥
      = (List.finRange n).foldr (fun j a => (if P j then ((sc j : ℝ) : WithBot ℝ) else ⊥) ⊔ a) ⊥ from by
    induction (List.finRange n) with
    | nil => simp
    | cons a t ih => by_cases ha : P a <;> simp [ha, ih]]
  apply le_antisymm
  · induction (List.finRange n) with
    | nil => simp
    | cons a t ih =>
      simp only [List.foldr_cons]
      exact sup_le (Finset.le_sup (f := fun j : Fin n => if P j then ((sc j : ℝ) : WithBot ℝ) else ⊥)
        (Finset.mem_univ a)) ih
  · apply Finset.sup_le
    intro j _
    have key : ∀ (l : List (Fin n)), j ∈ l →
        (if P j then ((sc j : ℝ) : WithBot ℝ) else ⊥)
          ≤ l.foldr (fun j a => (if P j then ((sc j : ℝ) : WithBot ℝ) else ⊥) ⊔ a) ⊥ := by
      intro l hl
      induction l with
      | nil => simp at hl
      | cons a t ih =>
        simp only [List.foldr_cons]
        rcases List.mem_cons.mp hl with h | h
        · subst h; exact le_sup_left
        · exact le_trans (ih h) le_sup_right
    exact key _ (List.mem_finRange j)

/-- Block-local lane `jL : Fin BN` maps to a global key `c·BN + jL < S`. -/
theorem gBlock_idx_lt (S BN c : Nat) (hwin : (c + 1) * BN ≤ S) (jL : Fin BN) :
    c * BN + jL.val < S := by
  have hjlt := jL.isLt
  have heq : (c + 1) * BN = c * BN + BN := by ring
  omega

/-- Reindex a windowed `Finset.sup` over `Fin S` onto `Fin BN`. -/
theorem g_window_sup_reindex (BN c S : Nat) (hwin : (c + 1) * BN ≤ S)
    (F : Nat → WithBot ℝ) :
    Finset.univ.sup (fun j : Fin S =>
        if c * BN ≤ j.val ∧ j.val < (c + 1) * BN then F j.val else ⊥)
      = Finset.univ.sup (fun jL : Fin BN => F (c * BN + jL.val)) := by
  have hmul : (c + 1) * BN = c * BN + BN := by ring
  apply le_antisymm
  · apply Finset.sup_le
    intro j _
    by_cases hj : c * BN ≤ j.val ∧ j.val < (c + 1) * BN
    · rw [if_pos hj]
      have hjL : j.val - c * BN < BN := by omega
      refine le_trans ?_ (Finset.le_sup (f := fun jL : Fin BN => F (c * BN + jL.val))
        (Finset.mem_univ (⟨j.val - c * BN, hjL⟩ : Fin BN)))
      simp only
      rw [show c * BN + (j.val - c * BN) = j.val from by omega]
    · rw [if_neg hj]; exact bot_le
  · apply Finset.sup_le
    intro jL _
    have hb : c * BN + jL.val < S := by have := jL.isLt; omega
    refine le_trans ?_ (Finset.le_sup
      (f := fun j : Fin S => if c * BN ≤ j.val ∧ j.val < (c + 1) * BN then F j.val else ⊥)
      (Finset.mem_univ (⟨c * BN + jL.val, hb⟩ : Fin S)))
    simp only
    rw [if_pos (by have := jL.isLt; exact ⟨by omega, by omega⟩)]

/-- Reindex a masked `Fin S`-window sum onto `Fin BN`. -/
theorem g_window_sum_reindex (BN c S : Nat) (hwin : (c + 1) * BN ≤ S) (g : Nat → ℝ) :
    (∑ j : Fin S, if c * BN ≤ j.val ∧ j.val < (c + 1) * BN then g j.val else 0)
      = ∑ jL : Fin BN, g (c * BN + jL.val) := by
  have hmul : (c + 1) * BN = c * BN + BN := by ring
  rw [← Finset.sum_filter]
  symm
  refine Finset.sum_bij (i := fun jL _ => (⟨c * BN + jL.val, gBlock_idx_lt S BN c hwin jL⟩ : Fin S))
    ?_ ?_ ?_ ?_
  · intro jL _; simp only [Finset.mem_filter, Finset.mem_univ, true_and]; have := jL.isLt; omega
  · intro a _ b _ hab
    apply Fin.ext
    have : c * BN + a.val = c * BN + b.val := by simpa using congrArg Fin.val hab
    omega
  · intro j hj
    have hj2 : c * BN ≤ j.val ∧ j.val < (c + 1) * BN := (Finset.mem_filter.mp hj).2
    exact ⟨⟨j.val - c * BN, by omega⟩, Finset.mem_univ _, by apply Fin.ext; simp only; omega⟩
  · intro jL _; rfl

/-- The generic block list reindexes onto `Fin BN` (key `c·BN + jL`). -/
theorem gBlock_map_sum (S BN c : Nat) (g : Fin S → ℝ × ℝ)
    (hwin : (c + 1) * BN ≤ S) (h : ℝ × ℝ → ℝ) :
    ((gBlock S BN c g).map h).sum
      = ∑ jL : Fin BN, h (g ⟨c * BN + jL.val, gBlock_idx_lt S BN c hwin jL⟩) := by
  rw [gBlock, g_filterMap_finRange_sum S
    (fun j => c * BN ≤ j.val ∧ j.val < (c + 1) * BN) g h]
  rw [show (∑ j : Fin S, if c * BN ≤ j.val ∧ j.val < (c + 1) * BN then h (g j) else 0)
        = ∑ j : Fin S, if c * BN ≤ j.val ∧ j.val < (c + 1) * BN
            then (fun jg => if h' : jg < S then h (g ⟨jg, h'⟩) else 0) j.val else 0 from by
    apply Finset.sum_congr rfl; intro j _
    by_cases hw : c * BN ≤ j.val ∧ j.val < (c + 1) * BN
    · rw [if_pos hw, if_pos hw]; simp only [dif_pos j.isLt]
    · rw [if_neg hw, if_neg hw]]
  rw [g_window_sum_reindex BN c S hwin
    (fun jg => if h' : jg < S then h (g ⟨jg, h'⟩) else 0)]
  apply Finset.sum_congr rfl
  intro jL _
  simp only [dif_pos (gBlock_idx_lt S BN c hwin jL)]

/-- **One-block advance** of the generic ⊥-seeded running max. -/
theorem gRunningMax_succ (S BN c : Nat) (g : Fin S → ℝ × ℝ) (hwin : (c + 1) * BN ≤ S) :
    gRunningMax S ((c + 1) * BN) g
      = gRunningMax S (c * BN) g
        ⊔ Finset.univ.sup (fun jL : Fin BN =>
            ((g ⟨c * BN + jL.val, gBlock_idx_lt S BN c hwin jL⟩).1 : WithBot ℝ)) := by
  unfold gRunningMax
  rw [gKeysUpto_succ, List.map_append, List.foldr_append]
  have hblock : ((gBlock S BN c g).map (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥
      = Finset.univ.sup (fun jL : Fin BN =>
          ((g ⟨c * BN + jL.val, gBlock_idx_lt S BN c hwin jL⟩).1 : WithBot ℝ)) := by
    rw [show (gBlock S BN c g).map (fun p => ((p.1 : ℝ) : WithBot ℝ))
          = ((List.finRange S).filterMap (fun j : Fin S =>
              if c * BN ≤ j.val ∧ j.val < (c + 1) * BN
              then some ((g j).1) else none)).map (fun x : ℝ => ((x : ℝ) : WithBot ℝ)) from by
      unfold gBlock
      rw [List.map_filterMap, List.map_filterMap]
      apply List.filterMap_congr
      intro j _
      by_cases hj : c * BN ≤ j.val ∧ j.val < (c + 1) * BN <;> simp [hj]]
    rw [g_filterMap_foldr_sup S
      (fun j => c * BN ≤ j.val ∧ j.val < (c + 1) * BN) (fun j => (g j).1)]
    classical
    rw [show (Finset.univ.sup (fun j : Fin S =>
          if c * BN ≤ j.val ∧ j.val < (c + 1) * BN
          then (((g j).1 : ℝ) : WithBot ℝ) else ⊥))
        = Finset.univ.sup (fun j : Fin S =>
            if c * BN ≤ j.val ∧ j.val < (c + 1) * BN
            then (fun jg => if h : jg < S then (((g ⟨jg, h⟩).1 : ℝ) : WithBot ℝ) else ⊥) j.val else ⊥)
        from by
      apply Finset.sup_congr rfl
      intro j _
      by_cases hw : c * BN ≤ j.val ∧ j.val < (c + 1) * BN
      · rw [if_pos hw, if_pos hw]; simp only [dif_pos j.isLt]
      · rw [if_neg hw, if_neg hw]]
    rw [g_window_sup_reindex BN c S hwin
      (fun jg => if h : jg < S then (((g ⟨jg, h⟩).1 : ℝ) : WithBot ℝ) else ⊥)]
    apply Finset.sup_congr rfl
    intro jL _
    have hb : c * BN + jL.val < S := by
      have hjlt := jL.isLt; have heq : (c + 1) * BN = c * BN + BN := by ring
      omega
    simp only [dif_pos hb]
  rw [hblock]
  generalize (gKeysUpto S (c * BN) g).map (fun p => ((p.1 : ℝ) : WithBot ℝ)) = preL
  induction preL with
  | nil => simp
  | cons a t ih => simp only [List.foldr_cons, ih]; rw [sup_assoc]

/-- **Generic ⊥-seeded consistency**: the running `(l, acc)` of `gStateBot` are
`κ(runningMax)` times the max-free reference sums. -/
theorem gStateBot_consistent (S hi : Nat) (g : Fin S → ℝ × ℝ) :
    (gStateBot S hi g).2.1
        = ((gStateBot S hi g).1.elim 0 (fun r => pow2 (-r)))
          * ((gKeysUpto S hi g).map (fun p => pow2 p.1)).sum ∧
    (gStateBot S hi g).2.2
        = ((gStateBot S hi g).1.elim 0 (fun r => pow2 (-r)))
          * ((gKeysUpto S hi g).map (fun p => pow2 p.1 * p.2)).sum := by
  obtain ⟨hL, hT⟩ := osStepBot_foldl_consistent (gKeysUpto S hi g) ⊥ 0 0 0 0
    (by simp) (by simp) (fun _ => rfl) (fun _ => rfl)
  refine ⟨?_, ?_⟩
  · rw [gStateBot]; rw [show (List.foldl osStepBot (⊥, 0, 0) (gKeysUpto S hi g)).2.1 = _ from hL,
      zero_add]
  · rw [gStateBot]; rw [show (List.foldl osStepBot (⊥, 0, 0) (gKeysUpto S hi g)).2.2 = _ from hT,
      zero_add]

theorem gKeysUpto_map_sum_eq_zero_of_bot (S hi : Nat) (g : Fin S → ℝ × ℝ)
    (hbot : (gStateBot S hi g).1 = ⊥) (f : ℝ × ℝ → ℝ) :
    ((gKeysUpto S hi g).map f).sum = 0 := by
  rw [gStateBot_fst_eq_runningMax, gRunningMax] at hbot
  have hnil : gKeysUpto S hi g = [] := by
    rcases hk : gKeysUpto S hi g with _ | ⟨a, t⟩
    · rfl
    · exfalso
      have : ((a.1 : ℝ) : WithBot ℝ) ≤ ((gKeysUpto S hi g).map (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥ :=
        mem_le_foldr_sup _ _ (by rw [hk]; exact List.mem_cons_self ..)
      rw [hbot] at this
      exact absurd (le_bot_iff.mp this) WithBot.coe_ne_bot
  rw [hnil]; simp

/-- After `c+1` blocks the ⊥-seeded running max is finite. -/
theorem gRunningMax_succ_ne_bot (S BN c : Nat) (g : Fin S → ℝ × ℝ)
    (hBN : 0 < BN) (hwin : (c + 1) * BN ≤ S) :
    gRunningMax S ((c + 1) * BN) g ≠ ⊥ := by
  rw [gRunningMax_succ S BN c g hwin]
  intro h
  rw [max_eq_bot] at h
  have hle := Finset.le_sup (f := fun jL : Fin BN =>
      ((g ⟨c * BN + jL.val, gBlock_idx_lt S BN c hwin jL⟩).1 : WithBot ℝ))
      (Finset.mem_univ (⟨0, hBN⟩ : Fin BN))
  rw [h.2] at hle
  exact absurd (le_bot_iff.mp hle) WithBot.coe_ne_bot

/-- The running max / denominator of `gStateBot` depend only on the per-key scores. -/
theorem gStateBot_score_congr (S hi : Nat) (g1 g2 : Fin S → ℝ × ℝ)
    (h : ∀ j : Fin S, (g1 j).1 = (g2 j).1) :
    (gStateBot S hi g1).1 = (gStateBot S hi g2).1
      ∧ (gStateBot S hi g1).2.1 = (gStateBot S hi g2).2.1 := by
  have hkeys : (gKeysUpto S hi g1).map (fun p => ((p.1 : ℝ) : WithBot ℝ))
      = (gKeysUpto S hi g2).map (fun p => ((p.1 : ℝ) : WithBot ℝ)) := by
    unfold gKeysUpto
    rw [List.map_filterMap, List.map_filterMap]
    apply List.filterMap_congr; intro j _
    by_cases hj : j.val < hi <;> simp [hj, h j]
  have hkeys2 : (gKeysUpto S hi g1).map (fun p => pow2 p.1)
      = (gKeysUpto S hi g2).map (fun p => pow2 p.1) := by
    unfold gKeysUpto
    rw [List.map_filterMap, List.map_filterMap]
    apply List.filterMap_congr; intro j _
    by_cases hj : j.val < hi <;> simp [hj, h j]
  have hfst : (gStateBot S hi g1).1 = (gStateBot S hi g2).1 := by
    rw [gStateBot_fst_eq_runningMax, gStateBot_fst_eq_runningMax, gRunningMax, gRunningMax, hkeys]
  refine ⟨hfst, ?_⟩
  rw [(gStateBot_consistent S hi g1).1, (gStateBot_consistent S hi g2).1, hfst, hkeys2]

/-- **Block-step in explicit `Fin BN` form** (BN-parametric). -/
theorem gStateBot_succ_explicit (S BN c : Nat) (g : Fin S → ℝ × ℝ) (hwin : (c + 1) * BN ≤ S) :
    let st := gStateBot S (c * BN) g
    let M' := st.1 ⊔ Finset.univ.sup (fun jL : Fin BN =>
        ((g ⟨c * BN + jL.val, gBlock_idx_lt S BN c hwin jL⟩).1 : WithBot ℝ))
    gStateBot S ((c + 1) * BN) g
      = (M',
         st.2.1 * (WithBot.realExp2 (WithBot.realSub st.1 M')).unbotD 0
           + Finset.univ.sum (fun jL : Fin BN =>
               pow2 ((g ⟨c * BN + jL.val, gBlock_idx_lt S BN c hwin jL⟩).1 - M'.unbotD 0)),
         st.2.2 * (WithBot.realExp2 (WithBot.realSub st.1 M')).unbotD 0
           + Finset.univ.sum (fun jL : Fin BN =>
               pow2 ((g ⟨c * BN + jL.val, gBlock_idx_lt S BN c hwin jL⟩).1 - M'.unbotD 0)
                 * (g ⟨c * BN + jL.val, gBlock_idx_lt S BN c hwin jL⟩).2)) := by
  intro st M'
  obtain ⟨hLc, hTc⟩ := gStateBot_consistent S (c * BN) g
  have hMblock : M' = st.1 ⊔ ((gBlock S BN c g).map (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥ := by
    have hsup : ((gBlock S BN c g).map (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥
        = Finset.univ.sup (fun jL : Fin BN =>
            ((g ⟨c * BN + jL.val, gBlock_idx_lt S BN c hwin jL⟩).1 : WithBot ℝ)) := by
      rw [show (gBlock S BN c g).map (fun p => ((p.1 : ℝ) : WithBot ℝ))
            = ((List.finRange S).filterMap (fun j : Fin S =>
                if c * BN ≤ j.val ∧ j.val < (c + 1) * BN
                then some ((g j).1) else none)).map (fun x : ℝ => ((x : ℝ) : WithBot ℝ)) from by
        unfold gBlock
        rw [List.map_filterMap, List.map_filterMap]
        apply List.filterMap_congr
        intro j _
        by_cases hj : c * BN ≤ j.val ∧ j.val < (c + 1) * BN <;> simp [hj]]
      rw [g_filterMap_foldr_sup S (fun j => c * BN ≤ j.val ∧ j.val < (c + 1) * BN) (fun j => (g j).1)]
      classical
      rw [show (Finset.univ.sup (fun j : Fin S =>
            if c * BN ≤ j.val ∧ j.val < (c + 1) * BN then (((g j).1 : ℝ) : WithBot ℝ) else ⊥))
          = Finset.univ.sup (fun j : Fin S =>
              if c * BN ≤ j.val ∧ j.val < (c + 1) * BN
              then (fun jg => if h : jg < S then (((g ⟨jg, h⟩).1 : ℝ) : WithBot ℝ) else ⊥) j.val else ⊥)
          from by
        apply Finset.sup_congr rfl
        intro j _
        by_cases hw : c * BN ≤ j.val ∧ j.val < (c + 1) * BN
        · rw [if_pos hw, if_pos hw]; simp only [dif_pos j.isLt]
        · rw [if_neg hw, if_neg hw]]
      rw [g_window_sup_reindex BN c S hwin
        (fun jg => if h : jg < S then (((g ⟨jg, h⟩).1 : ℝ) : WithBot ℝ) else ⊥)]
      apply Finset.sup_congr rfl
      intro jL _
      simp only [dif_pos (gBlock_idx_lt S BN c hwin jL)]
    show _ = _
    rw [hsup]
  have hstep := osStepBot_block_eq st.1 st.2.1 st.2.2
    (((gKeysUpto S (c * BN) g).map (fun p => pow2 p.1 * p.2)).sum)
    (((gKeysUpto S (c * BN) g).map (fun p => pow2 p.1)).sum)
    (gBlock S BN c g) hLc hTc
    (fun hb => gKeysUpto_map_sum_eq_zero_of_bot S (c * BN) g hb _)
    (fun hb => gKeysUpto_map_sum_eq_zero_of_bot S (c * BN) g hb _)
  have hfold : (gBlock S BN c g).foldl osStepBot st = gStateBot S ((c + 1) * BN) g :=
    (gStateBot_succ S BN c g).symm
  rw [hfold] at hstep
  simp only [] at hstep
  rw [← hMblock] at hstep
  rw [show ((gBlock S BN c g).map (fun p => pow2 (p.1 - M'.unbotD 0))).sum
        = Finset.univ.sum (fun jL : Fin BN =>
            pow2 ((g ⟨c * BN + jL.val, gBlock_idx_lt S BN c hwin jL⟩).1 - M'.unbotD 0))
      from gBlock_map_sum S BN c g hwin (fun p => pow2 (p.1 - M'.unbotD 0))] at hstep
  rw [show ((gBlock S BN c g).map (fun p => pow2 (p.1 - M'.unbotD 0) * p.2)).sum
        = Finset.univ.sum (fun jL : Fin BN =>
            pow2 ((g ⟨c * BN + jL.val, gBlock_idx_lt S BN c hwin jL⟩).1 - M'.unbotD 0)
              * (g ⟨c * BN + jL.val, gBlock_idx_lt S BN c hwin jL⟩).2)
      from gBlock_map_sum S BN c g hwin (fun p => pow2 (p.1 - M'.unbotD 0) * p.2)] at hstep
  exact hstep.symm

/-! ### Block-causal-guarded normalized accumulator (faithfulness layer)

The bloom kernel normalizes `acc` *in loop* (`acc_scale = (lᵢ/lᵢⁿᵉʷ)·α`, `p_scale
= β/lᵢⁿᵉʷ`) **and** guards the rescale: `acc_scale = tl.where(offs_m + plen ≥
start_n, (lᵢ/lᵢⁿᵉʷ)·α, 1.0)`. For row `i`, a block `c` whose start `c·BN >
qpos = gᵢ+plen` is entirely causally-future (all keys get the `-1e8` sentinel), so
the kernel applies `acc_scale = 1` — it does **not** rescale `acc`, dropping the
sentinel residue from the normalization. `gAccN` is the faithful per-row
normalized accumulator: it tracks exactly the kernel's `acc` register, applying the
same block-level guard, so guard-fail blocks add only the (negligible) dot residue
without the spurious denominator rescale. The running `(max, l)` (`= gStateBot`)
are unaffected by the guard — the kernel's `m_i_new`/`l_i_new` carry no guard. -/

/-- **Faithful normalized accumulator** after `c` blocks for a row with causal
limit `qpos`. Mirrors the kernel's `acc` register: `acc_new = acc·acc_scale +
dot(p,v)`, with `acc_scale = (lᵢ/lᵢⁿᵉʷ)·α` on guard-pass blocks (`c·BN ≤ qpos`)
and `1` on guard-fail blocks, and `dot(p,v) = (numerⁿᵉʷ − numer·α)/lᵢⁿᵉʷ`. -/
noncomputable def gAccN (S BN qpos : Nat) (g : Fin S → ℝ × ℝ) : Nat → ℝ
  | 0 => 0
  | c + 1 =>
    let st := gStateBot S (c * BN) g
    let stn := gStateBot S ((c + 1) * BN) g
    let α := (WithBot.realExp2 (WithBot.realSub st.1 stn.1)).unbotD 0
    let accScale := if c * BN ≤ qpos then (st.2.1 / stn.2.1) * α else 1
    gAccN S BN qpos g c * accScale + (stn.2.2 - st.2.2 * α) / stn.2.1

@[simp] theorem gAccN_zero (S BN qpos : Nat) (g : Fin S → ℝ × ℝ) :
    gAccN S BN qpos g 0 = 0 := rfl

theorem gAccN_succ (S BN qpos c : Nat) (g : Fin S → ℝ × ℝ) :
    gAccN S BN qpos g (c + 1)
      = gAccN S BN qpos g c
          * (if c * BN ≤ qpos then
              ((gStateBot S (c * BN) g).2.1 / (gStateBot S ((c + 1) * BN) g).2.1)
                * (WithBot.realExp2 (WithBot.realSub (gStateBot S (c * BN) g).1
                    (gStateBot S ((c + 1) * BN) g).1)).unbotD 0
            else 1)
        + ((gStateBot S ((c + 1) * BN) g).2.2
            - (gStateBot S (c * BN) g).2.2
                * (WithBot.realExp2 (WithBot.realSub (gStateBot S (c * BN) g).1
                    (gStateBot S ((c + 1) * BN) g).1)).unbotD 0)
          / (gStateBot S ((c + 1) * BN) g).2.1 := rfl

/-! ### Tile-cell bridges (shape-generic) -/

theorem ctxg_reduceMaxDrop_data_row {M N : Nat} (hN : 0 < N) (qk : Tile .real [M, N])
    (rmaxT : Tile .real [M]) (hrm : Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [M,N].length) qk = some rmaxT)
    (r : Fin M) (g : Fin N → WithBot ℝ) (hqk : ∀ jL : Fin N, qk.data (r, jL, PUnit.unit) = g jL) :
    rmaxT.data (r, PUnit.unit) = Finset.univ.sup g := by
  unfold Tile.reduceMaxDrop at hrm
  rw [dif_pos (show 0 < TileShape.axisDim [M,N] (⟨1, by simp⟩ : Fin [M,N].length) from hN)] at hrm
  rw [← Option.some.inj hrm]
  simp only [Finset.sup'_eq_sup]
  exact Finset.sup_congr rfl (fun jL _ => hqk jL)

/-- A `WithBot ℝ` sum of `some`-valued cells is `some` of the real sum. -/
theorem ctxg_withBot_sum_some {N : Nat} (g : Fin N → ℝ) :
    @Finset.sum (Fin N) (WithBot ℝ) _ Finset.univ (fun k => (some (g k) : WithBot ℝ))
      = some (Finset.univ.sum g) :=
  (WithBot.coe_sum Finset.univ g).symm

/-- `realExp2` is total (never `⊥`). -/
theorem ctxg_realExp2_eq_some_unbotD (z : WithBot ℝ) :
    WithBot.realExp2 z = some ((WithBot.realExp2 z).unbotD 0) := by
  cases z <;> rfl

/-- `m_ij = select(m_i > rmax) m_i rmax` collapses to `max` in `WithBot ℝ`. -/
theorem ctxg_mij_max {M : Nat} (m_i rmaxT : Tile .real [M]) (r : Fin M)
    (a b : WithBot ℝ) (hmi : m_i.data (r, PUnit.unit) = a) (hrm : rmaxT.data (r, PUnit.unit) = b) :
    (Tile.select (Tile.cop ComparableDType.real.gt Broadcast.nil.consSame m_i rmaxT) m_i rmaxT).data
        (r, PUnit.unit) = max a b := by
  rw [Tile.select_data, Tile.cop_data]
  simp only [Broadcast.leftIndex, Broadcast.rightIndex, ComparableDType.gt, hmi, hrm]
  by_cases h : a ≤ b
  · rw [if_neg (by simp [not_lt.mpr h]), max_eq_right h]
  · rw [if_pos (by simpa using not_le.mp h), max_eq_left (le_of_lt (not_le.mp h))]

/-- A `dot` row over all `K` keys when both factors are all-`some`. -/
theorem ctxg_dot_row {M K N : Nat} (p : Tile .real [M, K]) (v : Tile .real [K, N])
    (r : Fin M) (d : Fin N) (fp fv : Fin K → ℝ)
    (hp : ∀ jL : Fin K, p.data (r, jL, PUnit.unit) = some (fp jL))
    (hv : ∀ jL : Fin K, v.data (jL, d, PUnit.unit) = some (fv jL)) :
    (Tile.dot [] p v).data (r, d, PUnit.unit) = some (Finset.univ.sum fun jL : Fin K => fp jL * fv jL) := by
  rw [Tile.dot_nil_data]
  rw [show (@Finset.sum (Fin K) (WithBot ℝ) _ Finset.univ
        (fun k => Option.map₂ (· * ·) (p.data (r, k, PUnit.unit)) (v.data (k, d, PUnit.unit))))
      = @Finset.sum (Fin K) (WithBot ℝ) _ Finset.univ (fun k => (some (fp k * fv k) : WithBot ℝ))
      from Finset.sum_congr rfl (fun k _ => by rw [hp k, hv k]; rfl)]
  exact ctxg_withBot_sum_some _

/-- The `qk -= m_ij[:, None]` cell readback (avoids deep recursion inline). -/
theorem ctx_qk_sub_mij_cell {BM BN : Nat} (qkT : Tile .real [BM, BN]) (mijT : Tile .real [BM])
    (i : Fin BM) (jL : Fin BN) (sc mij : ℝ)
    (hqk : qkT.data (i, jL, PUnit.unit) = some sc)
    (hmij : mijT.data (i, PUnit.unit) = some mij) :
    (Tile.bop NumericDType.real.sub Broadcast.nil.consR.consSame qkT
        (Tile.expandDim ⟨1, by simp⟩ mijT)).data (i, jL, PUnit.unit)
      = some (sc - mij) := by
  simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, Tile.expandDim_data,
    TileShape.dropInsertedIndex, TileShape.insertAxisIndex, hqk, hmij,
    NumericDType.sub, WithBot.realSub, Option.map₂, Option.bind, Option.map]

/-- The `q·k` dot cell is the (unscaled) score (shape-generic `[BM,D]·[D,BN]`). -/
theorem ctx_dot_score_cell {BM BN D : Nat}
    (qtile : Tile .real [BM, D]) (ktile : Tile .real [D, BN]) (i : Fin BM) (j : Fin BN)
    (qf : Fin BM → Fin D → ℝ) (kf : Fin BN → Fin D → ℝ)
    (hq : ∀ e : Fin D, qtile.data (i, e, PUnit.unit) = some (qf i e))
    (hk : ∀ e : Fin D, ktile.data (e, j, PUnit.unit) = some (kf j e)) :
    (Tile.dot [] qtile ktile).data (i, j, PUnit.unit)
      = some (Finset.univ.sum (fun e : Fin D => qf i e * kf j e)) := by
  rw [Tile.dot_nil_data]
  rw [show (@Finset.sum (Fin D) (WithBot ℝ) _ Finset.univ
        (fun e => Option.map₂ (· * ·) (qtile.data (i, e, PUnit.unit)) (ktile.data (e, j, PUnit.unit))))
      = @Finset.sum (Fin D) (WithBot ℝ) _ Finset.univ
          (fun e => (some (qf i e * kf j e) : WithBot ℝ))
      from Finset.sum_congr rfl (fun e _ => by rw [hq e, hk e]; rfl)]
  rw [show (fun e : Fin D => (some (qf i e * kf j e) : WithBot ℝ))
        = (fun e : Fin D => ((qf i e * kf j e : ℝ) : WithBot ℝ)) from rfl,
    ← WithBot.coe_sum]; rfl

/-- The bloom kernel's masked `qk` cell at lane `(i,j)`: active lane gets the
scaled dot `sm·Σ_e qf·kf`; future lane gets the `-1e8` sentinel. -/
noncomputable def bloomQkCell {BM BN D : Nat} (sm : ℝ) (SN plen : Nat) (gOM : Fin BM → Nat)
    (qf : Fin BM → Fin D → ℝ) (kf : Fin BN → Fin D → ℝ) (i : Fin BM) (j : Fin BN) : ℝ :=
  if SN + j.val ≤ gOM i + plen then
    sm * Finset.univ.sum (fun e : Fin D => qf i e * kf j e)
  else (0.0 - 100000000.0 : ℝ)

/-- **The bloom kernel's `qkT` cell is `some (bloomQkCell …)`** (shape-generic).
The `tl.where(mask, (0 + dot)·sm, -1e8)` register (the bloom `+0` from
`qk = tl.zeros; qk += dot`) reads `qf`/`kf` and has cell `(i,j) = bloomQkCell`. -/
theorem bloom_qkT_cell {BM BN D : Nat} (sm : ℝ) (SN plen : Nat) (gOM : Fin BM → Nat)
    (qtile : Tile .real [BM, D]) (kloadT : Tile .real [D, BN]) (qf : Fin BM → Fin D → ℝ) (kf : Fin BN → Fin D → ℝ)
    (hq : ∀ (i : Fin BM) (e : Fin D), qtile.data (i, e, PUnit.unit) = some (qf i e))
    (hk : ∀ (j : Fin BN) (e : Fin D), kloadT.data (e, j, PUnit.unit) = some (kf j e))
    (i : Fin BM) (j : Fin BN) :
    (Tile.select
        (⟨fun idx : TileIndex [BM, BN] => decide (SN + idx.2.1.val ≤ gOM idx.1 + plen)⟩ : Tile .bool [BM, BN])
        (Tile.bop NumericDType.real.mul Broadcast.scalarR
          (Tile.bop NumericDType.real.add Broadcast.nil.consSame.consSame
            (⟨fun _ : TileIndex [BM, BN] => some (0 : ℝ)⟩ : Tile .real [BM, BN])
            (Tile.dot [] qtile kloadT))
          (Tile.scalar (some sm)))
        (⟨fun _ : TileIndex [BM, BN] => some (0.0 - 100000000.0 : ℝ)⟩ : Tile .real [BM, BN])).data
      (i, j, PUnit.unit)
      = some (bloomQkCell sm SN plen gOM qf kf i j) := by
  rw [Tile.select_data, bloomQkCell]
  have hsel : (⟨fun idx : TileIndex [BM, BN] => decide (SN + idx.2.1.val ≤ gOM idx.1 + plen)⟩ : Tile .bool [BM, BN]).data (i, j, PUnit.unit)
      = decide (SN + j.val ≤ gOM i + plen) := rfl
  by_cases h : SN + j.val ≤ gOM i + plen
  · rw [hsel, if_pos h]
    simp only [decide_eq_true_eq.mpr h, if_true]
    rw [Tile.bop_data]
    simp only [Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.mul]
    have hadd : (Tile.bop NumericDType.real.add Broadcast.nil.consSame.consSame
        (⟨fun _ : TileIndex [BM, BN] => some (0 : ℝ)⟩ : Tile .real [BM, BN])
        (Tile.dot [] qtile kloadT)).data (i, j, PUnit.unit)
        = some (Finset.univ.sum (fun e : Fin D => qf i e * kf j e)) := by
      rw [Tile.bop_data]
      simp only [Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add]
      rw [ctx_dot_score_cell qtile kloadT i j qf kf (hq i) (hk j)]
      show WithBot.realAdd (some 0) (some _) = _
      simp only [WithBot.realAdd, Option.map₂, Option.bind, Option.map]
      rw [zero_add]
    rw [hadd]
    show Option.map₂ (· * ·) _ _ = _
    simp only [Tile.scalar_data, Option.map₂]
    refine congrArg some ?_; ring
  · rw [hsel, if_neg h]
    simp only [decide_eq_false_iff_not.mpr h, Bool.false_eq_true, if_false]

/-! ### Per-statement `evalOp` helpers (parametric) -/

/-- Axis-0 `expandDim` over a `nat` register (`offs_n[None, :]` row broadcast). -/
@[simp] theorem ctx_evalOp_expandDim_zero_nat {D : Nat} (name : RegName) (s : BlockState) :
    @evalOp .nat [1, D] (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] name)) s =
      (s.regs .nat [D] name).bind (fun v =>
        some ({ data := fun i : TileIndex [1, D] => v.data (i.2.1, PUnit.unit) } : Tile .nat [1, D])) := by
  unfold evalOp; simp [Tile.expandDim]; rfl

/-- Axis-1 `expandDim` over a `nat` register (`offs_m[:, None]` column broadcast). -/
@[simp] theorem ctx_evalOp_expandDim_one_nat {M : Nat} (name : RegName) (s : BlockState) :
    @evalOp .nat [M, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] name)) s =
      (s.regs .nat [M] name).bind (fun v =>
        some ({ data := fun i : TileIndex [M, 1] => v.data (i.1, PUnit.unit) } : Tile .nat [M, 1])) := by
  unfold evalOp; simp [Tile.expandDim]; rfl

/-- Axis-1 `expandDim` over a `real` register (`m_ij[:, None]`/`alpha[:, None]`). -/
@[simp] theorem ctx_evalOp_expandDim_one_real {M : Nat} (name : RegName) (s : BlockState) :
    @evalOp .real [M, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [M] name)) s =
      (s.regs .real [M] name).bind (fun v =>
        some ({ data := fun i : TileIndex [M, 1] => v.data (i.1, PUnit.unit) } : Tile .real [M, 1])) := by
  unfold evalOp; simp [Tile.expandDim]; rfl

/-- Eval helper for `ge` (causal mask comparison). -/
theorem ctx_evalOp_ge {dtype a b shape} (h : ComparableDType dtype) (bc : Broadcast a b shape)
    (x : Op dtype a) (y : Op dtype b) (s : BlockState) :
    evalOp (.ge h bc x y) s = (do
      let vx ← evalOp x s; let vy ← evalOp y s; some (Tile.cop h.ge bc vx vy)) := by
  simp [evalOp]

/-- Eval helper for `floorDiv` (`cur_kv_head = cur_head // kv_group_num`). -/
theorem ctx_evalOp_floorDiv {dtype a b shape} (h : IntegralDType dtype)
    (bc : Broadcast a b shape) (x : Op dtype a) (y : Op dtype b) (s : BlockState) :
    evalOp (.floorDiv h bc x y) s = (do
      let vx ← evalOp x s; let vy ← evalOp y s; some (Tile.bop h.floorDiv bc vx vy)) := by
  simp [evalOp]

/-- Scalar `nat` region load (`tl.load(b_prompt_cache_len + cur_batch)` etc.). -/
theorem ctx_evalOp_load_scalar_nat (region : Region .nat) (off : Op .nat [])
    (s : BlockState) (o : Nat) (hoff : evalOp off s = some (Tile.scalar o)) :
    evalOp (.load .nat (MemAccess.region region off) MaskOpt.none) s
      = some (Tile.scalar (s.readMemValue .nat (Region.cast region) o)) := by
  simp only [evalOp, hoff, Option.bind_eq_bind, Option.bind_some]
  rfl

/-- Masked pointer-arith region load (`tl.load(R + offs, mask=m, other=o)`). -/
theorem ctx_evalOp_load_region_maskOther {dtype : TileDType} {shape : TileShape}
    (region : Region dtype) (offsets : Op .nat shape)
    (mask : Op .bool shape) (other : Op dtype shape) (s : BlockState)
    (offsTile : Tile .nat shape) (maskTile : Tile .bool shape) (otherTile : Tile dtype shape)
    (hoff : evalOp offsets s = some offsTile)
    (hmask : evalOp mask s = some maskTile)
    (hother : evalOp other s = some otherTile) :
    evalOp (.load dtype (MemAccess.region region offsets) (MaskOpt.maskOther mask other)) s
      = some ⟨fun i => if maskTile.data i then
          s.readMemValue dtype (Region.cast region) (offsTile.data i) else otherTile.data i⟩ := by
  simp only [evalOp, hoff, hmask, hother, Option.bind_eq_bind, Option.bind_some]
  rfl

/-- `qk = tl.zeros([…])` full-zero eval. -/
theorem bloomQkFull_eval {BM BN : Nat} (s : BlockState) :
    evalOp (Op.full [BM, BN] (Op.const (0 : ℝ))) s
      = some (⟨fun _ : TileIndex [BM, BN] => some (0 : ℝ)⟩ : Tile .real [BM, BN]) := by
  simp [evalOp_full, evalOp_const]

/-- `qk += tl.dot(q, k)` eval (`qk = qk + dot q k`, `qk` is all-zero). -/
theorem bloomQkAddDot_eval {BM BN D : Nat} (s : BlockState)
    (qktile : Tile .real [BM, BN]) (qtile : Tile .real [BM, D]) (ktile : Tile .real [D, BN])
    (hqk : s.regs .real [BM, BN] "qk" = some qktile)
    (hq : s.regs .real [BM, D] "q" = some qtile) (hk : s.regs .real [D, BN] "k" = some ktile) :
    evalOp (Op.add .real Broadcast.nil.consSame.consSame (Op.ref .real [BM, BN] "qk")
        (Op.dot (batch := []) (Op.ref .real [BM, D] "q") (Op.ref .real [D, BN] "k"))) s
      = some (Tile.bop NumericDType.real.add Broadcast.nil.consSame.consSame qktile
          (Tile.dot [] qtile ktile)) := by
  have hdot : evalOp (Op.dot (batch := []) (Op.ref .real [BM, D] "q") (Op.ref .real [D, BN] "k")) s
      = some (Tile.dot [] qtile ktile) := by rw [evalOp_dot]; simp [hq, hk]
  have hdot2 : @evalOp TileDType.real [BM, BN]
      (Op.dot (batch := []) (Op.ref .real [BM, D] "q") (Op.ref .real [D, BN] "k")) s
      = some (Tile.dot [] qtile ktile) := hdot
  rw [evalOp_add]; simp only [evalOp_ref, hqk, hdot2, Option.bind_eq_bind, Option.bind_some]; rfl

/-- `qk *= sm_scale` eval. -/
theorem bloomQkScale_eval {BM BN : Nat} (s : BlockState) (sm : ℝ) (qktile : Tile .real [BM, BN])
    (hqk : s.regs .real [BM, BN] "qk" = some qktile) :
    evalOp (Op.mul .real Broadcast.scalarR (Op.ref .real [BM, BN] "qk") (Op.const sm)) s
      = some (Tile.bop NumericDType.real.mul Broadcast.scalarR qktile (Tile.scalar (some sm))) := by
  rw [evalOp_mul]; simp only [evalOp_ref, evalOp_const, hqk, Option.bind_eq_bind, Option.bind_some]

/-- `qk = tl.where(offs_m[:,None]+plen ≥ start_n+offs_n[None,:], qk, -1e8)` eval
(causal `-1e8` sentinel, mask `ge` inline). -/
theorem bloomQkWhere_eval (s : BlockState) (BM BN plen SN : Nat) (gOM : Fin BM → Nat)
    (qktile : Tile .real [BM, BN])
    (hm : s.regs .nat [BM] "offs_m" = some (Tile.vec gOM))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => j.val)))
    (hp : s.regs .nat [] "prompt_cache_len" = some (Tile.scalar plen))
    (hsn : s.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hqk : s.regs .real [BM, BN] "qk" = some qktile) :
    evalOp ((Op.ge .nat Broadcast.nil.consL.consR
          (Op.add .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m"))
            (Op.ref .nat [] "prompt_cache_len"))
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_n")))).where
        (Op.ref .real [BM, BN] "qk")
        ((Op.sub .real Broadcast.nil (Op.const (0.0 : ℝ)) (Op.const (100000000.0 : ℝ))).broadcast [BM, BN])) s
      = some (Tile.select
          (⟨fun idx : TileIndex [BM, BN] => decide (SN + idx.2.1.val ≤ gOM idx.1 + plen)⟩ : Tile .bool [BM, BN])
          qktile
          (⟨fun _ : TileIndex [BM, BN] => some (0.0 - 100000000.0 : ℝ)⟩ : Tile .real [BM, BN])) := by
  rw [evalOp_where]
  have hmask : evalOp (Op.ge .nat Broadcast.nil.consL.consR
        (Op.add .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m"))
          (Op.ref .nat [] "prompt_cache_len"))
        (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_n")))) s
      = some (⟨fun idx : TileIndex [BM, BN] => decide (SN + idx.2.1.val ≤ gOM idx.1 + plen)⟩ : Tile .bool [BM, BN]) := by
    rw [ctx_evalOp_ge]
    simp only [evalOp_add, evalOp_ref, ctx_evalOp_expandDim_one_nat, ctx_evalOp_expandDim_zero_nat,
      hm, hn, hp, hsn, Option.bind_eq_bind, Option.bind_some]
    refine congrArg some ?_; ext idx
    simp [Tile.cop, Tile.bop, Tile.expandDim, Tile.vec, ComparableDType.ge, NumericDType.add]
  have hother : @evalOp .real [BM, BN]
      ((Op.sub NumericDType.real Broadcast.nil (Op.const (0.0:ℝ)) (Op.const (100000000.0:ℝ))).broadcast [BM, BN]) s
      = some (⟨fun _ : TileIndex [BM, BN] => some (0.0 - 100000000.0 : ℝ)⟩ : Tile .real [BM, BN]) := by
    simp only [evalOp]; refine congrArg some ?_; ext idx; simp [Tile.bop, NumericDType.sub]
  simp only [hmask, evalOp_ref, hqk, hother, Option.bind_eq_bind, Option.bind_some]

/-- `m_ij = tl.max(qk, 1)` eval (block-max reduce). -/
theorem bloomMij_eval {BM BN : Nat} (s : BlockState) (qkfull : Tile .real [BM, BN]) (rmaxT : Tile .real [BM])
    (hqk : s.regs .real [BM, BN] "qk" = some qkfull)
    (hrm : Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [BM, BN].length) qkfull = some rmaxT) :
    evalOp (Op.reduceMax ⟨1, by simp⟩ Bool.false (Op.ref .real [BM, BN] "qk")) s = some rmaxT := by
  rw [evalOp_reduceMax]; simp only [evalOp_ref, hqk, Option.bind_some]; exact hrm

/-- `l_ij = tl.sum(p, 1)` eval. -/
theorem bloomLij_eval {BM BN : Nat} (s : BlockState) (ptile : Tile .real [BM, BN])
    (hp : s.regs .real [BM, BN] "p" = some ptile) :
    evalOp (Op.reduceSum (⟨1, by simp⟩ : Fin [BM, BN].length) Bool.false (Op.ref .real [BM, BN] "p")) s
      = some (Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [BM, BN].length) ptile) := by
  rw [evalOp_reduceSum]; simp only [evalOp_ref, hp, Option.bind_some]; rfl

/-- `p = tl.exp(qk − m_ij[:, None])` natural-exp eval. -/
theorem bloomP_eval {BM BN : Nat} (s : BlockState) (qk2tile : Tile .real [BM, BN])
    (mij : Tile .real [BM])
    (hqk : s.regs .real [BM, BN] "qk" = some qk2tile) (hmij : s.regs .real [BM] "m_ij" = some mij) :
    evalOp (Op.sub .real Broadcast.nil.consR.consSame (Op.ref .real [BM, BN] "qk")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BM] "m_ij"))).exp s
      = some (Tile.uop WithBot.realExp (Tile.bop NumericDType.real.sub Broadcast.nil.consR.consSame
          qk2tile (Tile.expandDim ⟨1, by simp⟩ mij))) := by
  rw [evalOp_exp, evalOp_sub]
  have hexp : @evalOp TileDType.real [BM, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BM] "m_ij")) s
      = some (Tile.expandDim ⟨1, by simp⟩ mij) := by erw [ctx_evalOp_expandDim_one_real, hmij]; rfl
  simp only [evalOp_ref, hqk, hexp, Option.bind_eq_bind, Option.bind_some]; rfl

/-- `m_i_new = tl.maximum(m_i, m_ij)` = `where(m_i > m_ij, m_i, m_ij)` eval. -/
theorem bloomMiNew_eval {BM : Nat} (s : BlockState) (mi mij : Tile .real [BM])
    (hmi : s.regs .real [BM] "m_i" = some mi) (hmij : s.regs .real [BM] "m_ij" = some mij) :
    evalOp ((Op.gt .real Broadcast.nil.consSame (Op.ref .real [BM] "m_i") (Op.ref .real [BM] "m_ij")).where
        (Op.ref .real [BM] "m_i") (Op.ref .real [BM] "m_ij")) s
      = some (Tile.select (Tile.cop ComparableDType.real.gt Broadcast.nil.consSame mi mij) mi mij) := by
  rw [evalOp_where]
  simp only [evalOp_gt, evalOp_ref, hmi, hmij, Option.bind_eq_bind, Option.bind_some]

/-- `alpha = tl.exp(m_i − m_i_new)` / `beta = tl.exp(m_ij − m_i_new)` natural-exp eval. -/
theorem bloomExpSub_eval {BM : Nat} (s : BlockState) (nm1 nm2 : RegName) (t1 t2 : Tile .real [BM])
    (h1 : s.regs .real [BM] nm1 = some t1) (h2 : s.regs .real [BM] nm2 = some t2) :
    evalOp (Op.sub .real Broadcast.nil.consSame (Op.ref .real [BM] nm1) (Op.ref .real [BM] nm2)).exp s
      = some (Tile.uop WithBot.realExp (Tile.bop NumericDType.real.sub Broadcast.nil.consSame t1 t2)) := by
  rw [evalOp_exp, evalOp_sub]; simp [h1, h2]

/-- `l_i_new = alpha·l_i + beta·l_ij` eval. -/
theorem bloomLiNew_eval {BM : Nat} (s : BlockState) (alpha li beta lij : Tile .real [BM])
    (ha : s.regs .real [BM] "alpha" = some alpha) (hli : s.regs .real [BM] "l_i" = some li)
    (hb : s.regs .real [BM] "beta" = some beta) (hlij : s.regs .real [BM] "l_ij" = some lij) :
    evalOp (Op.add .real Broadcast.nil.consSame
        (Op.mul .real Broadcast.nil.consSame (Op.ref .real [BM] "alpha") (Op.ref .real [BM] "l_i"))
        (Op.mul .real Broadcast.nil.consSame (Op.ref .real [BM] "beta") (Op.ref .real [BM] "l_ij"))) s
      = some (Tile.bop NumericDType.real.add Broadcast.nil.consSame
          (Tile.bop NumericDType.real.mul Broadcast.nil.consSame alpha li)
          (Tile.bop NumericDType.real.mul Broadcast.nil.consSame beta lij)) := by
  rw [evalOp_add]; simp [evalOp_mul, ha, hli, hb, hlij]

/-- `p_scale = beta / l_i_new` eval. -/
theorem bloomPscale_eval {BM : Nat} (s : BlockState) (beta lin : Tile .real [BM])
    (hb : s.regs .real [BM] "beta" = some beta) (hlin : s.regs .real [BM] "l_i_new" = some lin) :
    evalOp (Op.div .real Broadcast.nil.consSame (Op.ref .real [BM] "beta") (Op.ref .real [BM] "l_i_new")) s
      = some (Tile.bop NumericDType.real.div Broadcast.nil.consSame beta lin) := by
  rw [evalOp_div]; simp [hb, hlin]

/-- `p = p · p_scale[:, None]` eval. -/
theorem bloomP2_eval {BM BN : Nat} (s : BlockState) (ptile : Tile .real [BM, BN]) (pscale : Tile .real [BM])
    (hp : s.regs .real [BM, BN] "p" = some ptile) (hps : s.regs .real [BM] "p_scale" = some pscale) :
    evalOp (Op.mul .real Broadcast.nil.consR.consSame (Op.ref .real [BM, BN] "p")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BM] "p_scale"))) s
      = some (Tile.bop NumericDType.real.mul Broadcast.nil.consR.consSame ptile
          (Tile.expandDim ⟨1, by simp⟩ pscale)) := by
  have hexp : @evalOp TileDType.real [BM, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BM] "p_scale")) s
      = some (Tile.expandDim ⟨1, by simp⟩ pscale) := by erw [ctx_evalOp_expandDim_one_real, hps]; rfl
  rw [evalOp_mul]; simp only [evalOp_ref, hp, hexp, Option.bind_eq_bind, Option.bind_some]; rfl

/-- `acc_scale = (l_i / l_i_new) · alpha` eval. -/
theorem bloomAccScale1_eval {BM : Nat} (s : BlockState) (li lin alpha : Tile .real [BM])
    (hli : s.regs .real [BM] "l_i" = some li) (hlin : s.regs .real [BM] "l_i_new" = some lin)
    (ha : s.regs .real [BM] "alpha" = some alpha) :
    evalOp (Op.mul .real Broadcast.nil.consSame
        (Op.div .real Broadcast.nil.consSame (Op.ref .real [BM] "l_i") (Op.ref .real [BM] "l_i_new"))
        (Op.ref .real [BM] "alpha")) s
      = some (Tile.bop NumericDType.real.mul Broadcast.nil.consSame
          (Tile.bop NumericDType.real.div Broadcast.nil.consSame li lin) alpha) := by
  rw [evalOp_mul]; simp [evalOp_div, hli, hlin, ha]

/-- `acc_scale = tl.where(offs_m+plen ≥ start_n, acc_scale, 1.0)` eval (the
acc-rescale active-lane guard). -/
theorem bloomAccScale2_eval (s : BlockState) (BM plen SN : Nat) (gOM : Fin BM → Nat)
    (acctile : Tile .real [BM])
    (hm : s.regs .nat [BM] "offs_m" = some (Tile.vec gOM))
    (hp : s.regs .nat [] "prompt_cache_len" = some (Tile.scalar plen))
    (hsn : s.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hacc : s.regs .real [BM] "acc_scale" = some acctile) :
    evalOp ((Op.ge .nat Broadcast.scalarR
          (Op.add .nat Broadcast.scalarR (Op.ref .nat [BM] "offs_m") (Op.ref .nat [] "prompt_cache_len"))
          (Op.ref .nat [] "start_n")).where
        (Op.ref .real [BM] "acc_scale") ((Op.const (1.0 : ℝ)).broadcast [BM])) s
      = some (Tile.select
          (⟨fun idx : TileIndex [BM] => decide (SN ≤ gOM idx.1 + plen)⟩ : Tile .bool [BM])
          acctile (⟨fun _ : TileIndex [BM] => some (1.0 : ℝ)⟩ : Tile .real [BM])) := by
  rw [evalOp_where]
  have hmask : evalOp (Op.ge .nat Broadcast.scalarR
        (Op.add .nat Broadcast.scalarR (Op.ref .nat [BM] "offs_m") (Op.ref .nat [] "prompt_cache_len"))
        (Op.ref .nat [] "start_n")) s
      = some (⟨fun idx : TileIndex [BM] => decide (SN ≤ gOM idx.1 + plen)⟩ : Tile .bool [BM]) := by
    rw [ctx_evalOp_ge]
    simp only [evalOp_add, evalOp_ref, hm, hp, hsn, Option.bind_eq_bind, Option.bind_some]
    refine congrArg some ?_; ext idx
    simp [Tile.cop, Tile.bop, Tile.vec, Tile.scalar_data_index, ComparableDType.ge, NumericDType.add]
  have hother : @evalOp .real [BM] ((Op.const (1.0 : ℝ)).broadcast [BM]) s
      = some (⟨fun _ : TileIndex [BM] => some (1.0 : ℝ)⟩ : Tile .real [BM]) := by
    simp only [evalOp]; refine congrArg some ?_; ext idx; rfl
  simp only [hmask, evalOp_ref, hacc, hother, Option.bind_eq_bind, Option.bind_some]

/-- `acc = acc · acc_scale[:, None]` eval. -/
theorem bloomAcc1_eval {BM D : Nat} (s : BlockState) (acctile : Tile .real [BM, D]) (ascale : Tile .real [BM])
    (hacc : s.regs .real [BM, D] "acc" = some acctile) (has : s.regs .real [BM] "acc_scale" = some ascale) :
    evalOp (Op.mul .real Broadcast.nil.consR.consSame (Op.ref .real [BM, D] "acc")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BM] "acc_scale"))) s
      = some (Tile.bop NumericDType.real.mul Broadcast.nil.consR.consSame acctile
          (Tile.expandDim ⟨1, by simp⟩ ascale)) := by
  have hexp : @evalOp TileDType.real [BM, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BM] "acc_scale")) s
      = some (Tile.expandDim ⟨1, by simp⟩ ascale) := by erw [ctx_evalOp_expandDim_one_real, has]; rfl
  rw [evalOp_mul]; simp only [evalOp_ref, hacc, hexp, Option.bind_eq_bind, Option.bind_some]; rfl

/-- `acc += tl.dot(p, v)` eval (`acc = acc + dot p v`). -/
theorem bloomAcc2_eval {BM BN D : Nat} (s : BlockState) (acctile : Tile .real [BM, D])
    (ptile : Tile .real [BM, BN]) (vtile : Tile .real [BN, D])
    (hacc : s.regs .real [BM, D] "acc" = some acctile)
    (hp : s.regs .real [BM, BN] "p" = some ptile) (hv : s.regs .real [BN, D] "v" = some vtile) :
    evalOp (Op.add .real Broadcast.nil.consSame.consSame (Op.ref .real [BM, D] "acc")
        (Op.dot (batch := []) (Op.ref .real [BM, BN] "p") (Op.ref .real [BN, D] "v"))) s
      = some (Tile.bop NumericDType.real.add Broadcast.nil.consSame.consSame acctile
          (Tile.dot [] ptile vtile)) := by
  have hdot0 : evalOp (Op.dot (batch := []) (Op.ref .real [BM, BN] "p") (Op.ref .real [BN, D] "v")) s
      = some (Tile.dot [] ptile vtile) := by rw [evalOp_dot]; simp [hp, hv]
  have hdot : @evalOp TileDType.real [BM, D]
      (Op.dot (batch := []) (Op.ref .real [BM, BN] "p") (Op.ref .real [BN, D] "v")) s
      = some (Tile.dot [] ptile vtile) := hdot0
  rw [evalOp_add]; simp only [evalOp_ref, hacc, hdot, Option.bind_eq_bind, Option.bind_some]; rfl

/-- The ⊥-seeded fold over the empty window `[0, 0)` is the seed `(⊥, 0, 0)`. -/
theorem gStateBot_zero (S : Nat) (g : Fin S → ℝ × ℝ) : gStateBot S 0 g = (⊥, 0, 0) := by
  rw [gStateBot, show gKeysUpto S 0 g = [] from by
    rw [gKeysUpto]; apply List.filterMap_eq_nil_iff.mpr; intro j _; simp]
  rfl

/-- `block_end_loc = min((start_m+1)·BLOCK_M + plen, cur_batch_seq_len + plen)`. -/
def bloomFwdBel (s : BlockState) (B_Seqlen B_Prompt_Cache_Len : RegionName) (BLOCK_M : Nat) : Nat :=
  let plen := promptLen s B_Prompt_Cache_Len
  let sl := seqLen s B_Seqlen B_Prompt_Cache_Len
  let a := (s.pids 2 + 1) * BLOCK_M + plen
  let b := sl + plen
  if a < b then a else b

/-- General `Req_to_tokens` gather: physical token slot for streamed key index
`j`, gather strides free:
`kv_loc(j) = Req_to_tokens[stride_req_b·req_idx + stride_req_s·j]`. -/
noncomputable def bloomKvLocG
    (s : BlockState) (Req_to_tokens B_req_idx : RegionName)
    (stride_req_b stride_req_s j : Nat) : Nat :=
  s.readMemValue .nat Req_to_tokens (reqIdx s B_req_idx * stride_req_b + stride_req_s * j)

/-- General coordinate-faithful query tile `Q[gi, e]` at head/stride parameters. -/
noncomputable def bloomQTileG
    (s : BlockState) (Q B_Start_Loc : RegionName)
    (stride_qbs stride_qh stride_qd BLOCK_M : Nat) (i : Fin BLOCK_M) (e : Nat) : ℝ :=
  s.readMem Q
    ((startLoc s B_Start_Loc + (s.pids 2 * BLOCK_M + i.val)) * stride_qbs
      + curHead s * stride_qh + e * stride_qd)

/-- General coordinate-faithful key tile `K[kv_loc(j), cur_head, e]`. -/
noncomputable def bloomKTileG (s : BlockState) (K Req_to_tokens B_req_idx : RegionName)
    (stride_req_b stride_req_s stride_kbs stride_kh stride_kd S : Nat) (j : Fin S) (e : Nat) : ℝ :=
  s.readMem K (bloomKvLocG s Req_to_tokens B_req_idx stride_req_b stride_req_s j.val * stride_kbs
    + curHead s * stride_kh + e * stride_kd)

/-- General coordinate-faithful value tile `V[kv_loc(j), cur_head, d]`. -/
noncomputable def bloomVTileG (s : BlockState) (V Req_to_tokens B_req_idx : RegionName)
    (stride_req_b stride_req_s stride_vbs stride_vh stride_vd S : Nat) (j : Fin S) (d : Nat) : ℝ :=
  s.readMem V (bloomKvLocG s Req_to_tokens B_req_idx stride_req_b stride_req_s j.val * stride_vbs
    + curHead s * stride_vh + d * stride_vd)

/-- General `block_end_loc`/channel-masked key tile: genuine `bloomKTileG` for
`j < bel` and channel `e < head_dim`, else `0`. -/
noncomputable def bloomKTileMG (s : BlockState) (K Req_to_tokens B_req_idx : RegionName)
    (stride_req_b stride_req_s stride_kbs stride_kh stride_kd head_dim S bel : Nat)
    (j : Fin S) (e : Nat) : ℝ :=
  if (j.val < bel) ∧ (e < head_dim) then
    bloomKTileG s K Req_to_tokens B_req_idx stride_req_b stride_req_s stride_kbs stride_kh stride_kd S j e
  else 0

/-- General `block_end_loc`/channel-masked value tile. -/
noncomputable def bloomVTileMG (s : BlockState) (V Req_to_tokens B_req_idx : RegionName)
    (stride_req_b stride_req_s stride_vbs stride_vh stride_vd head_dim S bel : Nat)
    (j : Fin S) (d : Nat) : ℝ :=
  if (j.val < bel) ∧ (d < head_dim) then
    bloomVTileG s V Req_to_tokens B_req_idx stride_req_b stride_req_s stride_vbs stride_vh stride_vd S j d
  else 0

/-- General row/channel-masked query tile: genuine `bloomQTileG` on active rows
(`gi < seq_len`) and channel `e < head_dim`, else `0`. -/
noncomputable def bloomQTileMG
    (s : BlockState) (Q B_Start_Loc B_Seqlen B_Prompt_Cache_Len : RegionName)
    (stride_qbs stride_qh stride_qd head_dim BLOCK_M : Nat) (i : Fin BLOCK_M) (e : Nat) : ℝ :=
  if (s.pids 2 * BLOCK_M + i.val < seqLen s B_Seqlen B_Prompt_Cache_Len) ∧ (e < head_dim) then
    bloomQTileG s Q B_Start_Loc stride_qbs stride_qh stride_qd BLOCK_M i e
  else 0

/-- General faithful per-key `(base-2 score, value)` the loop folds, with the
`-1e8` sentinel kept and the `block_end_loc`/channel load masks; score fed
`/ log 2` so `pow2 score = exp (natural kernel score)`. -/
noncomputable def bloomKVMG
    (s : BlockState) (Q K V B_Start_Loc B_Seqlen B_Prompt_Cache_Len Req_to_tokens B_req_idx : RegionName)
    (sm_scale : ℝ)
    (stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd head_dim BLOCK_DMODEL BLOCK_M S bel : Nat)
    (i : Fin BLOCK_M) (d : Nat) (j : Fin S) : ℝ × ℝ :=
  ((if j.val ≤ s.pids 2 * BLOCK_M + i.val + promptLen s B_Prompt_Cache_Len then
      sm_scale * Finset.univ.sum (fun e : Fin BLOCK_DMODEL =>
        bloomQTileMG s Q B_Start_Loc B_Seqlen B_Prompt_Cache_Len stride_qbs stride_qh stride_qd head_dim BLOCK_M i e.val
          * bloomKTileMG s K Req_to_tokens B_req_idx stride_req_b stride_req_s stride_kbs stride_kh stride_kd head_dim S bel j e.val)
    else (0.0 - 10e7 : ℝ)) / Real.log 2,
    bloomVTileMG s V Req_to_tokens B_req_idx stride_req_b stride_req_s stride_vbs stride_vh stride_vd head_dim S bel j d)

/-- **General faithful kernel value** at output lane `(i,d)`: the block-causal-guarded
normalized accumulator `gAccN` of the ⊥-seeded online-softmax fold over `bloomKVMG`
for the full streamed window `[0, S)`. A pure function of `Q`/`K`/`V` memory. -/
noncomputable def contextAttnBloomExactFoldMG
    (s : BlockState) (Q K V B_Start_Loc B_Seqlen B_Prompt_Cache_Len Req_to_tokens B_req_idx : RegionName)
    (sm_scale : ℝ)
    (stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd head_dim BLOCK_DMODEL BLOCK_N BLOCK_M S bel : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : ℝ :=
  gAccN S BLOCK_N (s.pids 2 * BLOCK_M + idx.1.val + promptLen s B_Prompt_Cache_Len)
    (bloomKVMG s Q K V B_Start_Loc B_Seqlen B_Prompt_Cache_Len Req_to_tokens B_req_idx sm_scale
      stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd head_dim BLOCK_DMODEL BLOCK_M S bel idx.1 idx.2.1.val)
    (S / BLOCK_N)

/-! ### General op-eval gather/mask helpers (strides + head_dim free) -/

/-- General masked `kv_loc` gather eval (gather strides free). -/
theorem bloom_kvloc_gather_evalG {BN : Nat} (s : BlockState)
    (Req_to_tokens : RegionName) (rqi SN bel stride_req_b stride_req_s : Nat)
    (hrqi : s.regs .nat [] "cur_batch_req_idx" = some (Tile.scalar rqi))
    (hsn : s.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => j.val)))
    (hbel : s.regs .nat [] "block_end_loc" = some (Tile.scalar bel)) :
    evalOp (Op.load .nat
        (MemAccess.region Req_to_tokens
          (Op.add .nat Broadcast.scalarL
            (Op.mul .nat Broadcast.nil (Op.constNat stride_req_b) (Op.ref .nat [] "cur_batch_req_idx"))
            (Op.mul .nat Broadcast.scalarL (Op.constNat stride_req_s)
              (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n") (Op.ref .nat [BN] "offs_n")))))
        (MaskOpt.maskOther
          (Op.lt .nat Broadcast.scalarR
            (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n") (Op.ref .nat [BN] "offs_n"))
            (Op.ref .nat [] "block_end_loc"))
          (Op.broadcast (Op.constNat 0) [BN]))) s
      = some (⟨fun idx : TileIndex [BN] =>
          if decide (SN + idx.1.val < bel) then
            s.readMemValue .nat (Region.cast Req_to_tokens) (stride_req_b * rqi + stride_req_s * (SN + idx.1.val))
          else 0⟩ : Tile .nat [BN]) := by
  simp only [evalOp, hrqi, hsn, hn, hbel, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_; ext idx
  simp only [Tile.cop_data, Tile.bop_data, Tile.scalar_data, Tile.scalar_data_index, Tile.vec_data,
    Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul,
    ComparableDType.lt, Region.cast_cast]
  by_cases h : SN + idx.1.val < bel
  · simp only [h, decide_true, if_true]
  · simp only [h, decide_false, Bool.false_eq_true, if_false]

/-- General `off_k` gather recipe (K strides + head_dim free, shape `[D, BN]`). -/
theorem bloom_offk_gather_evalG {BN D : Nat} (s : BlockState) (ckvh : Nat) (kvf : Fin BN → Nat)
    (stride_kbs stride_kh stride_kd : Nat)
    (hkvloc : s.regs .nat [BN] "kv_loc" = some (Tile.vec kvf))
    (hckvh : s.regs .nat [] "cur_kv_head" = some (Tile.scalar ckvh))
    (hd : s.regs .nat [D] "offs_d" = some (Tile.vec (fun e : Fin D => e.val))) :
    evalOp (Op.add .nat Broadcast.nil.consR.consL
        (Op.add .nat Broadcast.scalarR
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "kv_loc"))
            (Op.constNat stride_kbs))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_kv_head") (Op.constNat stride_kh)))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [D] "offs_d"))
          (Op.constNat stride_kd))) s
      = some (⟨fun idx : TileIndex [D, BN] =>
          kvf idx.2.1 * stride_kbs + ckvh * stride_kh + idx.1.val * stride_kd⟩ : Tile .nat [D, BN]) := by
  simp only [evalOp_add, evalOp_mul, evalOp_constNat, evalOp_ref,
    ctx_evalOp_expandDim_one_nat, ctx_evalOp_expandDim_zero_nat,
    hkvloc, hckvh, hd, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_; ext idx
  simp [Tile.bop_data, Tile.scalar_data, Tile.vec_data, Tile.expandDim, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.add, NumericDType.mul]

/-- General `off_v` gather recipe (V strides + head_dim free, shape `[BN, D]`). -/
theorem bloom_offv_gather_evalG {BN D : Nat} (s : BlockState) (ckvh : Nat) (kvf : Fin BN → Nat)
    (stride_vbs stride_vh stride_vd : Nat)
    (hkvloc : s.regs .nat [BN] "kv_loc" = some (Tile.vec kvf))
    (hckvh : s.regs .nat [] "cur_kv_head" = some (Tile.scalar ckvh))
    (hd : s.regs .nat [D] "offs_d" = some (Tile.vec (fun e : Fin D => e.val))) :
    evalOp (Op.add .nat Broadcast.nil.consL.consR
        (Op.add .nat Broadcast.scalarR
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BN] "kv_loc"))
            (Op.constNat stride_vbs))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_kv_head") (Op.constNat stride_vh)))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d"))
          (Op.constNat stride_vd))) s
      = some (⟨fun idx : TileIndex [BN, D] =>
          kvf idx.1 * stride_vbs + ckvh * stride_vh + idx.2.1.val * stride_vd⟩ : Tile .nat [BN, D]) := by
  simp only [evalOp_add, evalOp_mul, evalOp_constNat, evalOp_ref,
    ctx_evalOp_expandDim_one_nat, ctx_evalOp_expandDim_zero_nat,
    hkvloc, hckvh, hd, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_; ext idx
  simp [Tile.bop_data, Tile.scalar_data, Tile.vec_data, Tile.expandDim, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.add, NumericDType.mul]

/-- General `k` masked-load mask (head_dim free, shape `[D, BN]`). -/
theorem bloomKMask_evalG {BN D : Nat} (s : BlockState) (SN bel head_dim : Nat)
    (hsn : s.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => j.val)))
    (hd : s.regs .nat [D] "offs_d" = some (Tile.vec (fun e : Fin D => e.val)))
    (hbel : s.regs .nat [] "block_end_loc" = some (Tile.scalar bel)) :
    evalOp (Op.boolAnd Broadcast.nil.consR.consL
        (Op.lt .nat Broadcast.scalarR
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_n")))
          (Op.ref .nat [] "block_end_loc"))
        (Op.lt .nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [D] "offs_d"))
          (Op.constNat head_dim))) s
      = some (⟨fun idx : TileIndex [D, BN] =>
          decide (SN + idx.2.1.val < bel) && decide (idx.1.val < head_dim)⟩ : Tile .bool [D, BN]) := by
  rw [evalOp]
  rw [evalOp_lt, evalOp_add]
  erw [ctx_evalOp_expandDim_zero_nat]
  rw [evalOp_lt]
  erw [ctx_evalOp_expandDim_one_nat]
  simp only [evalOp_ref, evalOp_constNat, hsn, hn, hd, hbel,
    Option.bind_eq_bind, Option.bind_some, Option.pure_def]
  refine congrArg some ?_; ext idx
  simp [Tile.bop_data, Tile.cop_data, Tile.expandDim, Tile.scalar_data_index,
    Tile.vec_data, Broadcast.leftIndex, Broadcast.rightIndex, ComparableDType.lt, NumericDType.add]

/-- General `v` masked-load mask (head_dim free, shape `[BN, D]`). -/
theorem bloomVMask_evalG {BN D : Nat} (s : BlockState) (SN bel head_dim : Nat)
    (hsn : s.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => j.val)))
    (hd : s.regs .nat [D] "offs_d" = some (Tile.vec (fun e : Fin D => e.val)))
    (hbel : s.regs .nat [] "block_end_loc" = some (Tile.scalar bel)) :
    evalOp (Op.boolAnd Broadcast.nil.consL.consR
        (Op.lt .nat Broadcast.scalarR
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BN] "offs_n")))
          (Op.ref .nat [] "block_end_loc"))
        (Op.lt .nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d"))
          (Op.constNat head_dim))) s
      = some (⟨fun idx : TileIndex [BN, D] =>
          decide (SN + idx.1.val < bel) && decide (idx.2.1.val < head_dim)⟩ : Tile .bool [BN, D]) := by
  rw [evalOp]
  rw [evalOp_lt, evalOp_add]
  erw [ctx_evalOp_expandDim_one_nat]
  rw [evalOp_lt]
  erw [ctx_evalOp_expandDim_zero_nat]
  simp only [evalOp_ref, evalOp_constNat, hsn, hn, hd, hbel,
    Option.bind_eq_bind, Option.bind_some, Option.pure_def]
  refine congrArg some ?_; ext idx
  simp [Tile.bop_data, Tile.cop_data, Tile.expandDim, Tile.scalar_data_index,
    Tile.vec_data, Broadcast.leftIndex, Broadcast.rightIndex, ComparableDType.lt, NumericDType.add]

/-- General `q`/store mask (head_dim free, shape `[BM, D]`). -/
theorem bloomQMask_evalG {BM D : Nat} (s : BlockState) (gOM : Fin BM → Nat) (sl head_dim : Nat)
    (hm : s.regs .nat [BM] "offs_m" = some (Tile.vec gOM))
    (hd : s.regs .nat [D] "offs_d" = some (Tile.vec (fun e : Fin D => e.val)))
    (hsl : s.regs .nat [] "cur_batch_seq_len" = some (Tile.scalar sl)) :
    evalOp (Op.boolAnd Broadcast.nil.consL.consR
        (Op.lt .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m"))
          (Op.ref .nat [] "cur_batch_seq_len"))
        (Op.lt .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d"))
          (Op.constNat head_dim))) s
      = some (⟨fun idx : TileIndex [BM, D] =>
          decide (gOM idx.1 < sl) && decide (idx.2.1.val < head_dim)⟩ : Tile .bool [BM, D]) := by
  rw [evalOp]
  rw [evalOp_lt]
  erw [ctx_evalOp_expandDim_one_nat]
  rw [evalOp_lt]
  erw [ctx_evalOp_expandDim_zero_nat]
  simp only [evalOp_ref, evalOp_constNat, hm, hd, hsl, Option.bind_eq_bind, Option.bind_some, Option.pure_def]
  refine congrArg some ?_; ext idx
  simp [Tile.bop_data, Tile.cop_data, Tile.expandDim, Tile.scalar_data_index, Tile.vec_data,
    Broadcast.leftIndex, Broadcast.rightIndex, ComparableDType.lt]


/-! ### General loop body + body split -/

/-- General lowered bloom loop-body statements (strides/head_dim/BLOCK_* free). -/
noncomputable def bloomLoopBodyG (Q K V Req_to_tokens B_req_idx : RegionName) (sm_scale : ℝ)
    (stride_req_b stride_req_s stride_kbs stride_kh stride_kd stride_vbs stride_vh stride_vd
      head_dim BLOCK_DMODEL BLOCK_M BLOCK_N : Nat) : List Stmt :=
  [ Stmt.assign .nat [] "start_n" (Op.ref .nat [] "start_n"),
    Stmt.assign .nat [BLOCK_N] "kv_loc"
      (Op.load .nat
        (MemAccess.region Req_to_tokens
          (Op.add .nat Broadcast.scalarL
            (Op.mul .nat Broadcast.nil (Op.constNat stride_req_b) (Op.ref .nat [] "cur_batch_req_idx"))
            (Op.mul .nat Broadcast.scalarL (Op.constNat stride_req_s)
              (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n") (Op.ref .nat [BLOCK_N] "offs_n")))))
        (MaskOpt.maskOther
          (Op.lt .nat Broadcast.scalarR
            (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n") (Op.ref .nat [BLOCK_N] "offs_n"))
            (Op.ref .nat [] "block_end_loc"))
          (Op.broadcast (Op.constNat 0) [BLOCK_N]))),
    Stmt.assign .nat [BLOCK_DMODEL, BLOCK_N] "off_k"
      (Op.add .nat Broadcast.nil.consR.consL
        (Op.add .nat Broadcast.scalarR
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_N] "kv_loc"))
            (Op.constNat stride_kbs))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_kv_head") (Op.constNat stride_kh)))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_DMODEL] "offs_d"))
          (Op.constNat stride_kd))),
    Stmt.assign .real [BLOCK_DMODEL, BLOCK_N] "k"
      (Op.load .real (MemAccess.region K (Op.ref .nat [BLOCK_DMODEL, BLOCK_N] "off_k"))
        (MaskOpt.maskOther
          (Op.boolAnd Broadcast.nil.consR.consL
            (Op.lt .nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
                (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_N] "offs_n")))
              (Op.ref .nat [] "block_end_loc"))
            (Op.lt .nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_DMODEL] "offs_d"))
              (Op.constNat head_dim)))
          ((Op.const (0.0 : ℝ)).broadcast [BLOCK_DMODEL, BLOCK_N]))),
    Stmt.assign .real [BLOCK_M, BLOCK_N] "qk"
      (Op.full [BLOCK_M, BLOCK_N] (Op.const (0 : ℝ))),
    Stmt.assign .real [BLOCK_M, BLOCK_N] "qk"
      (Op.add .real Broadcast.nil.consSame.consSame (Op.ref .real [BLOCK_M, BLOCK_N] "qk")
        (Op.dot (batch := []) (Op.ref .real [BLOCK_M, BLOCK_DMODEL] "q") (Op.ref .real [BLOCK_DMODEL, BLOCK_N] "k"))),
    Stmt.assign .real [BLOCK_M, BLOCK_N] "qk"
      (Op.mul .real Broadcast.scalarR (Op.ref .real [BLOCK_M, BLOCK_N] "qk") (Op.const (sm_scale : ℝ))),
    Stmt.assign .real [BLOCK_M, BLOCK_N] "qk"
      ((Op.ge .nat Broadcast.nil.consL.consR
          (Op.add .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m"))
            (Op.ref .nat [] "prompt_cache_len"))
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_N] "offs_n")))).where
        (Op.ref .real [BLOCK_M, BLOCK_N] "qk")
        ((Op.sub .real Broadcast.nil (Op.const (0.0 : ℝ)) (Op.const (100000000.0 : ℝ))).broadcast [BLOCK_M, BLOCK_N])),
    Stmt.assign .real [BLOCK_M] "m_ij"
      (Op.reduceMax ⟨1, by simp⟩ Bool.false (Op.ref .real [BLOCK_M, BLOCK_N] "qk")),
    Stmt.assign .real [BLOCK_M, BLOCK_N] "p"
      (Op.sub .real Broadcast.nil.consR.consSame (Op.ref .real [BLOCK_M, BLOCK_N] "qk")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BLOCK_M] "m_ij"))).exp,
    Stmt.assign .real [BLOCK_M] "l_ij"
      (Op.reduceSum ⟨1, by simp⟩ Bool.false (Op.ref .real [BLOCK_M, BLOCK_N] "p")),
    Stmt.assign .real [BLOCK_M] "m_i_new"
      ((Op.gt .real Broadcast.nil.consSame (Op.ref .real [BLOCK_M] "m_i") (Op.ref .real [BLOCK_M] "m_ij")).where
        (Op.ref .real [BLOCK_M] "m_i") (Op.ref .real [BLOCK_M] "m_ij")),
    Stmt.assign .real [BLOCK_M] "alpha"
      (Op.sub .real Broadcast.nil.consSame (Op.ref .real [BLOCK_M] "m_i") (Op.ref .real [BLOCK_M] "m_i_new")).exp,
    Stmt.assign .real [BLOCK_M] "beta"
      (Op.sub .real Broadcast.nil.consSame (Op.ref .real [BLOCK_M] "m_ij") (Op.ref .real [BLOCK_M] "m_i_new")).exp,
    Stmt.assign .real [BLOCK_M] "l_i_new"
      (Op.add .real Broadcast.nil.consSame
        (Op.mul .real Broadcast.nil.consSame (Op.ref .real [BLOCK_M] "alpha") (Op.ref .real [BLOCK_M] "l_i"))
        (Op.mul .real Broadcast.nil.consSame (Op.ref .real [BLOCK_M] "beta") (Op.ref .real [BLOCK_M] "l_ij"))),
    Stmt.assign .real [BLOCK_M] "p_scale"
      (Op.div .real Broadcast.nil.consSame (Op.ref .real [BLOCK_M] "beta") (Op.ref .real [BLOCK_M] "l_i_new")),
    Stmt.assign .real [BLOCK_M, BLOCK_N] "p"
      (Op.mul .real Broadcast.nil.consR.consSame (Op.ref .real [BLOCK_M, BLOCK_N] "p")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BLOCK_M] "p_scale"))),
    Stmt.assign .real [BLOCK_M] "acc_scale"
      (Op.mul .real Broadcast.nil.consSame
        (Op.div .real Broadcast.nil.consSame (Op.ref .real [BLOCK_M] "l_i") (Op.ref .real [BLOCK_M] "l_i_new"))
        (Op.ref .real [BLOCK_M] "alpha")),
    Stmt.assign .real [BLOCK_M] "acc_scale"
      ((Op.ge .nat Broadcast.scalarR
          (Op.add .nat Broadcast.scalarR (Op.ref .nat [BLOCK_M] "offs_m") (Op.ref .nat [] "prompt_cache_len"))
          (Op.ref .nat [] "start_n")).where
        (Op.ref .real [BLOCK_M] "acc_scale") ((Op.const (1.0 : ℝ)).broadcast [BLOCK_M])),
    Stmt.assign .real [BLOCK_M, BLOCK_DMODEL] "acc"
      (Op.mul .real Broadcast.nil.consR.consSame (Op.ref .real [BLOCK_M, BLOCK_DMODEL] "acc")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BLOCK_M] "acc_scale"))),
    Stmt.assign .nat [BLOCK_N, BLOCK_DMODEL] "off_v"
      (Op.add .nat Broadcast.nil.consL.consR
        (Op.add .nat Broadcast.scalarR
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_N] "kv_loc"))
            (Op.constNat stride_vbs))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_kv_head") (Op.constNat stride_vh)))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_DMODEL] "offs_d"))
          (Op.constNat stride_vd))),
    Stmt.assign .real [BLOCK_N, BLOCK_DMODEL] "v"
      (Op.load .real (MemAccess.region V (Op.ref .nat [BLOCK_N, BLOCK_DMODEL] "off_v"))
        (MaskOpt.maskOther
          (Op.boolAnd Broadcast.nil.consL.consR
            (Op.lt .nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
                (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_N] "offs_n")))
              (Op.ref .nat [] "block_end_loc"))
            (Op.lt .nat Broadcast.scalarR
              (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_DMODEL] "offs_d"))
              (Op.constNat head_dim)))
          ((Op.const (0.0 : ℝ)).broadcast [BLOCK_N, BLOCK_DMODEL]))),
    Stmt.assign .real [BLOCK_M, BLOCK_N] "p" (Op.ref .real [BLOCK_M, BLOCK_N] "p"),
    Stmt.assign .real [BLOCK_M, BLOCK_DMODEL] "acc"
      (Op.add .real Broadcast.nil.consSame.consSame (Op.ref .real [BLOCK_M, BLOCK_DMODEL] "acc")
        (Op.dot (batch := []) (Op.ref .real [BLOCK_M, BLOCK_N] "p") (Op.ref .real [BLOCK_N, BLOCK_DMODEL] "v"))),
    Stmt.assign .real [BLOCK_M] "l_i" (Op.ref .real [BLOCK_M] "l_i_new"),
    Stmt.assign .real [BLOCK_M] "m_i" (Op.ref .real [BLOCK_M] "m_i_new") ]


/-! ### General preLoop + postLoop + body split -/

/-- General lowered bloom prologue statements (strides/head_dim/BLOCK_* +
kv_group_num free). -/
noncomputable def bloomPreLoopG (Q : RegionName)
    (B_Start_Loc B_Seqlen B_req_idx b_prompt_cache_len : Region .nat)
    (stride_qbs stride_qh stride_qd head_dim kv_group_num BLOCK_DMODEL BLOCK_M BLOCK_N : Nat) : List Stmt :=
  [ Stmt.assign .nat [] "cur_batch" (Op.programId 0),
    Stmt.assign .nat [] "cur_head" (Op.programId 1),
    Stmt.assign .nat [] "start_m" (Op.programId 2),
    Stmt.assign .nat [] "cur_kv_head"
      (Op.floorDiv .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat kv_group_num)),
    Stmt.assign .nat [] "cur_batch_in_all_start_index"
      (Op.load .nat (MemAccess.region B_Start_Loc (Op.ref .nat [] "cur_batch")) MaskOpt.none),
    Stmt.assign .nat [] "prompt_cache_len"
      (Op.load .nat (MemAccess.region b_prompt_cache_len (Op.ref .nat [] "cur_batch")) MaskOpt.none),
    Stmt.assign .nat [] "cur_batch_seq_len"
      (Op.sub .nat Broadcast.nil
        (Op.load .nat (MemAccess.region B_Seqlen (Op.ref .nat [] "cur_batch")) MaskOpt.none)
        (Op.ref .nat [] "prompt_cache_len")),
    Stmt.assign .nat [] "cur_batch_req_idx"
      (Op.load .nat (MemAccess.region B_req_idx (Op.ref .nat [] "cur_batch")) MaskOpt.none),
    Stmt.assign .nat [] "block_start_loc"
      (Op.mul .nat Broadcast.nil (Op.constNat BLOCK_M) (Op.ref .nat [] "start_m")),
    Stmt.assign .nat [BLOCK_N] "offs_n" (Op.arange BLOCK_N),
    Stmt.assign .nat [BLOCK_DMODEL] "offs_d" (Op.arange BLOCK_DMODEL),
    Stmt.assign .nat [BLOCK_M] "offs_m"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat BLOCK_M)) (Op.arange BLOCK_M)),
    Stmt.assign .nat [BLOCK_M, BLOCK_DMODEL] "off_q"
      (Op.add .nat Broadcast.nil.consL.consR
        (Op.add .nat Broadcast.scalarR
          (Op.mul .nat Broadcast.scalarR
            (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "cur_batch_in_all_start_index")
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m")))
            (Op.constNat stride_qbs))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat stride_qh)))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_DMODEL] "offs_d"))
          (Op.constNat stride_qd))),
    Stmt.assign .real [BLOCK_M, BLOCK_DMODEL] "q"
      (Op.load .real (MemAccess.region Q (Op.ref .nat [BLOCK_M, BLOCK_DMODEL] "off_q"))
        (MaskOpt.maskOther
          (Op.boolAnd Broadcast.nil.consL.consR
            (Op.lt .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m"))
              (Op.ref .nat [] "cur_batch_seq_len"))
            (Op.lt .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_DMODEL] "offs_d"))
              (Op.constNat head_dim)))
          ((Op.const (0.0 : ℝ)).broadcast [BLOCK_M, BLOCK_DMODEL]))),
    Stmt.assign .real [BLOCK_M] "m_i"
      (Op.add .real Broadcast.scalarR (Op.full [BLOCK_M] (Op.const 0)) Op.negInf),
    Stmt.assign .real [BLOCK_M] "l_i" (Op.full [BLOCK_M] (Op.const 0)),
    Stmt.assign .real [BLOCK_M, BLOCK_DMODEL] "acc" (Op.full [BLOCK_M, BLOCK_DMODEL] (Op.const 0)),
    Stmt.assign .nat [] "block_mask"
      ((Op.lt .nat Broadcast.nil (Op.ref .nat [] "block_start_loc") (Op.ref .nat [] "cur_batch_seq_len")).where
        (Op.constNat 1) (Op.constNat 0)),
    Stmt.assign .nat [] "block_end_loc"
      ((Op.lt .nat Broadcast.nil
            (Op.add .nat Broadcast.nil
              (Op.mul .nat Broadcast.nil
                (Op.add .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat 1)) (Op.constNat BLOCK_M))
              (Op.ref .nat [] "prompt_cache_len"))
            (Op.add .nat Broadcast.nil (Op.ref .nat [] "cur_batch_seq_len")
              (Op.ref .nat [] "prompt_cache_len"))).where
        (Op.add .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil
            (Op.add .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat 1)) (Op.constNat BLOCK_M))
          (Op.ref .nat [] "prompt_cache_len"))
        (Op.add .nat Broadcast.nil (Op.ref .nat [] "cur_batch_seq_len")
          (Op.ref .nat [] "prompt_cache_len"))) ]

/-- General lowered bloom post-loop statements (NO `acc /= l_i`; in-loop
normalized). -/
noncomputable def bloomPostLoopG (Out : RegionName)
    (stride_obs stride_oh stride_od head_dim BLOCK_DMODEL BLOCK_M : Nat) : List Stmt :=
  [ Stmt.assign .nat [BLOCK_M, BLOCK_DMODEL] "off_o"
      (Op.add .nat Broadcast.nil.consL.consR
        (Op.add .nat Broadcast.scalarR
          (Op.mul .nat Broadcast.scalarR
            (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "cur_batch_in_all_start_index")
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m")))
            (Op.constNat stride_obs))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat stride_oh)))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_DMODEL] "offs_d"))
          (Op.constNat stride_od))),
    Stmt.assign .ptr [BLOCK_M, BLOCK_DMODEL] "out_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Out) (Op.ref .nat [BLOCK_M, BLOCK_DMODEL] "off_o")),
    Stmt.store .real [BLOCK_M, BLOCK_DMODEL] (MemAccess.ptr (Op.ref .ptr [BLOCK_M, BLOCK_DMODEL] "out_ptrs"))
      (Op.ref .real [BLOCK_M, BLOCK_DMODEL] "acc")
      (MaskOpt.mask
        (Op.boolAnd Broadcast.nil.consL.consR
          (Op.lt .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m"))
            (Op.ref .nat [] "cur_batch_seq_len"))
          (Op.lt .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_DMODEL] "offs_d"))
            (Op.constNat head_dim)))) ]

/-- **General body decomposition.** The general surface body splits as
`bloomPreLoopG ++ (forRangeDyn … bloomLoopBodyG :: bloomPostLoopG)` by `rfl`. -/
theorem bloomBody_splitG (Q K V Out : RegionName)
    (B_Start_Loc B_Seqlen Req_to_tokens B_req_idx b_prompt_cache_len : Region .nat) (sm_scale : ℝ)
    (stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
      stride_req_b stride_req_s kv_group_num head_dim BLOCK_M BLOCK_DMODEL BLOCK_N : Nat) :
    (context_attn_bloom_fwd_kernel_surface Q K V sm_scale B_Start_Loc B_Seqlen Out
        Req_to_tokens B_req_idx b_prompt_cache_len
        stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd
        stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
        stride_req_b stride_req_s kv_group_num head_dim BLOCK_M BLOCK_DMODEL BLOCK_N).toAlgKernel.body
      = bloomPreLoopG Q B_Start_Loc B_Seqlen B_req_idx b_prompt_cache_len
          stride_qbs stride_qh stride_qd head_dim kv_group_num BLOCK_DMODEL BLOCK_M BLOCK_N
        ++ (Stmt.forRangeDyn "start_n" (Op.constNat 0)
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "block_mask") (Op.ref .nat [] "block_end_loc"))
              (Op.constNat BLOCK_N)
              (bloomLoopBodyG Q K V Req_to_tokens B_req_idx sm_scale stride_req_b stride_req_s
                stride_kbs stride_kh stride_kd stride_vbs stride_vh stride_vd head_dim BLOCK_DMODEL BLOCK_M BLOCK_N)
            :: bloomPostLoopG Out stride_obs stride_oh stride_od head_dim BLOCK_DMODEL BLOCK_M) := by
  rfl

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **General LoopBody execution.** Mirror of `bloomLoopBody_steps` over symbolic
dims/strides + free gather/head_dim. Threads `0 < BLOCK_N` (for `reduceMaxDrop`
totality). -/
theorem bloomLoopBody_stepsG (Q K V Req_to_tokens B_req_idx : RegionName) (sm_scale : ℝ)
    (stride_req_b stride_req_s stride_kbs stride_kh stride_kd stride_vbs stride_vh stride_vd
      head_dim BLOCK_DMODEL BLOCK_M BLOCK_N : Nat) (hBN : 0 < BLOCK_N)
    (sin : BlockState) (SN : Nat)
    (gOM : Fin BLOCK_M → Nat) (cb ckvh ch plen sl bel cbsi rqi : Nat)
    (qtile : Tile .real [BLOCK_M, BLOCK_DMODEL]) (mtile ltile : Tile .real [BLOCK_M])
    (acctile : Tile .real [BLOCK_M, BLOCK_DMODEL])
    (hsn : sin.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hcb : sin.regs .nat [] "cur_batch" = some (Tile.scalar cb))
    (hckvh : sin.regs .nat [] "cur_kv_head" = some (Tile.scalar ckvh))
    (hch : sin.regs .nat [] "cur_head" = some (Tile.scalar ch))
    (hplen : sin.regs .nat [] "prompt_cache_len" = some (Tile.scalar plen))
    (hsl : sin.regs .nat [] "cur_batch_seq_len" = some (Tile.scalar sl))
    (hbel : sin.regs .nat [] "block_end_loc" = some (Tile.scalar bel))
    (hcbsi : sin.regs .nat [] "cur_batch_in_all_start_index" = some (Tile.scalar cbsi))
    (hrqi : sin.regs .nat [] "cur_batch_req_idx" = some (Tile.scalar rqi))
    (hm : sin.regs .nat [BLOCK_M] "offs_m" = some (Tile.vec gOM))
    (hn : sin.regs .nat [BLOCK_N] "offs_n" = some (Tile.vec (fun j : Fin BLOCK_N => j.val)))
    (hd : sin.regs .nat [BLOCK_DMODEL] "offs_d" = some (Tile.vec (fun e : Fin BLOCK_DMODEL => e.val)))
    (hq : sin.regs .real [BLOCK_M, BLOCK_DMODEL] "q" = some qtile)
    (hmi : sin.regs .real [BLOCK_M] "m_i" = some mtile)
    (hli : sin.regs .real [BLOCK_M] "l_i" = some ltile)
    (hacc : sin.regs .real [BLOCK_M, BLOCK_DMODEL] "acc" = some acctile)
    (hundef : ∀ rg o, sin.undef rg o = 0) :
    ∃ sF, stepStmts (bloomLoopBodyG Q K V Req_to_tokens B_req_idx sm_scale stride_req_b stride_req_s
        stride_kbs stride_kh stride_kd stride_vbs stride_vh stride_vd head_dim BLOCK_DMODEL BLOCK_M BLOCK_N) sin = some sF
      ∧ sF.pids = sin.pids ∧ sF.mem = sin.mem ∧ (∀ rg o, sF.undef rg o = 0)
      ∧ sF.regs .nat [] "start_n" = some (Tile.scalar SN)
      ∧ sF.regs .nat [] "cur_batch" = some (Tile.scalar cb)
      ∧ sF.regs .nat [] "cur_kv_head" = some (Tile.scalar ckvh)
      ∧ sF.regs .nat [] "cur_head" = some (Tile.scalar ch)
      ∧ sF.regs .nat [] "prompt_cache_len" = some (Tile.scalar plen)
      ∧ sF.regs .nat [] "cur_batch_seq_len" = some (Tile.scalar sl)
      ∧ sF.regs .nat [] "block_end_loc" = some (Tile.scalar bel)
      ∧ sF.regs .nat [] "cur_batch_in_all_start_index" = some (Tile.scalar cbsi)
      ∧ sF.regs .nat [] "cur_batch_req_idx" = some (Tile.scalar rqi)
      ∧ sF.regs .nat [BLOCK_M] "offs_m" = some (Tile.vec gOM)
      ∧ sF.regs .nat [BLOCK_N] "offs_n" = some (Tile.vec (fun j : Fin BLOCK_N => j.val))
      ∧ sF.regs .nat [BLOCK_DMODEL] "offs_d" = some (Tile.vec (fun e : Fin BLOCK_DMODEL => e.val))
      ∧ sF.regs .real [BLOCK_M, BLOCK_DMODEL] "q" = some qtile
      ∧ ∃ (kloadT : Tile .real [BLOCK_DMODEL, BLOCK_N]) (vloadT : Tile .real [BLOCK_N, BLOCK_DMODEL])
            (qkT : Tile .real [BLOCK_M, BLOCK_N])
            (rmaxT miNewT alphaT betaT lijT pscaleT liNewT accscale2T : Tile .real [BLOCK_M])
            (pexpT p2T : Tile .real [BLOCK_M, BLOCK_N]) (acc1T : Tile .real [BLOCK_M, BLOCK_DMODEL]),
          kloadT = (⟨fun idx : TileIndex [BLOCK_DMODEL, BLOCK_N] =>
              if decide (SN + idx.2.1.val < bel) && decide (idx.1.val < head_dim) then
                sin.readMemValue .real (Region.cast K)
                  ((if decide (SN + idx.2.1.val < bel) then
                      sin.readMemValue .nat (Region.cast Req_to_tokens) (stride_req_b * rqi + stride_req_s * (SN + idx.2.1.val))
                    else 0) * stride_kbs + ckvh * stride_kh + idx.1.val * stride_kd)
              else some (0.0 : ℝ)⟩ : Tile .real [BLOCK_DMODEL, BLOCK_N])
          ∧ vloadT = (⟨fun idx : TileIndex [BLOCK_N, BLOCK_DMODEL] =>
              if decide (SN + idx.1.val < bel) && decide (idx.2.1.val < head_dim) then
                sin.readMemValue .real (Region.cast V)
                  ((if decide (SN + idx.1.val < bel) then
                      sin.readMemValue .nat (Region.cast Req_to_tokens) (stride_req_b * rqi + stride_req_s * (SN + idx.1.val))
                    else 0) * stride_vbs + ckvh * stride_vh + idx.2.1.val * stride_vd)
              else some (0.0 : ℝ)⟩ : Tile .real [BLOCK_N, BLOCK_DMODEL])
          ∧ qkT = Tile.select
              (⟨fun idx : TileIndex [BLOCK_M, BLOCK_N] => decide (SN + idx.2.1.val ≤ gOM idx.1 + plen)⟩ : Tile .bool [BLOCK_M, BLOCK_N])
              (Tile.bop NumericDType.real.mul Broadcast.scalarR
                (Tile.bop NumericDType.real.add Broadcast.nil.consSame.consSame
                  (⟨fun _ : TileIndex [BLOCK_M, BLOCK_N] => some (0 : ℝ)⟩ : Tile .real [BLOCK_M, BLOCK_N])
                  (Tile.dot [] qtile kloadT))
                (Tile.scalar (some sm_scale)))
              (⟨fun _ : TileIndex [BLOCK_M, BLOCK_N] => some (0.0 - 100000000.0 : ℝ)⟩ : Tile .real [BLOCK_M, BLOCK_N])
          ∧ Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length) qkT = some rmaxT
          ∧ pexpT = Tile.uop WithBot.realExp (Tile.bop NumericDType.real.sub Broadcast.nil.consR.consSame
              qkT (Tile.expandDim ⟨1, by simp⟩ rmaxT))
          ∧ lijT = Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length) pexpT
          ∧ miNewT = Tile.select (Tile.cop ComparableDType.real.gt Broadcast.nil.consSame mtile rmaxT) mtile rmaxT
          ∧ alphaT = Tile.uop WithBot.realExp (Tile.bop NumericDType.real.sub Broadcast.nil.consSame mtile miNewT)
          ∧ betaT = Tile.uop WithBot.realExp (Tile.bop NumericDType.real.sub Broadcast.nil.consSame rmaxT miNewT)
          ∧ liNewT = Tile.bop NumericDType.real.add Broadcast.nil.consSame
              (Tile.bop NumericDType.real.mul Broadcast.nil.consSame alphaT ltile)
              (Tile.bop NumericDType.real.mul Broadcast.nil.consSame betaT lijT)
          ∧ pscaleT = Tile.bop NumericDType.real.div Broadcast.nil.consSame betaT liNewT
          ∧ p2T = Tile.bop NumericDType.real.mul Broadcast.nil.consR.consSame pexpT (Tile.expandDim ⟨1, by simp⟩ pscaleT)
          ∧ accscale2T = Tile.select
              (⟨fun idx : TileIndex [BLOCK_M] => decide (SN ≤ gOM idx.1 + plen)⟩ : Tile .bool [BLOCK_M])
              (Tile.bop NumericDType.real.mul Broadcast.nil.consSame
                (Tile.bop NumericDType.real.div Broadcast.nil.consSame ltile liNewT) alphaT)
              (⟨fun _ : TileIndex [BLOCK_M] => some (1.0 : ℝ)⟩ : Tile .real [BLOCK_M])
          ∧ acc1T = Tile.bop NumericDType.real.mul Broadcast.nil.consR.consSame acctile (Tile.expandDim ⟨1, by simp⟩ accscale2T)
          ∧ sF.regs .real [BLOCK_M] "m_i" = some miNewT
          ∧ sF.regs .real [BLOCK_M] "l_i" = some liNewT
          ∧ sF.regs .real [BLOCK_M, BLOCK_DMODEL] "acc" = some (Tile.bop NumericDType.real.add Broadcast.nil.consSame.consSame
              acc1T (Tile.dot [] p2T vloadT)) := by
  set kvf : Fin BLOCK_N → Nat := fun j : Fin BLOCK_N =>
      if decide (SN + j.val < bel) then
        sin.readMemValue .nat (Region.cast Req_to_tokens) (stride_req_b * rqi + stride_req_s * (SN + j.val))
      else 0 with hkvf
  set kloadT : Tile .real [BLOCK_DMODEL, BLOCK_N] := ⟨fun idx : TileIndex [BLOCK_DMODEL, BLOCK_N] =>
      if decide (SN + idx.2.1.val < bel) && decide (idx.1.val < head_dim) then
        sin.readMemValue .real (Region.cast K) (kvf idx.2.1 * stride_kbs + ckvh * stride_kh + idx.1.val * stride_kd)
      else some (0.0 : ℝ)⟩ with hkl
  set vloadT : Tile .real [BLOCK_N, BLOCK_DMODEL] := ⟨fun idx : TileIndex [BLOCK_N, BLOCK_DMODEL] =>
      if decide (SN + idx.1.val < bel) && decide (idx.2.1.val < head_dim) then
        sin.readMemValue .real (Region.cast V) (kvf idx.1 * stride_vbs + ckvh * stride_vh + idx.2.1.val * stride_vd)
      else some (0.0 : ℝ)⟩ with hvl
  set qkdotT : Tile .real [BLOCK_M, BLOCK_N] := Tile.bop NumericDType.real.add Broadcast.nil.consSame.consSame
      (⟨fun _ : TileIndex [BLOCK_M, BLOCK_N] => some (0 : ℝ)⟩ : Tile .real [BLOCK_M, BLOCK_N]) (Tile.dot [] qtile kloadT) with hqkdot
  set qkscaleT : Tile .real [BLOCK_M, BLOCK_N] := Tile.bop NumericDType.real.mul Broadcast.scalarR qkdotT
      (Tile.scalar (some sm_scale)) with hqkscale
  set qkT : Tile .real [BLOCK_M, BLOCK_N] := Tile.select
      (⟨fun idx : TileIndex [BLOCK_M, BLOCK_N] => decide (SN + idx.2.1.val ≤ gOM idx.1 + plen)⟩ : Tile .bool [BLOCK_M, BLOCK_N])
      qkscaleT
      (⟨fun _ : TileIndex [BLOCK_M, BLOCK_N] => some (0.0 - 100000000.0 : ℝ)⟩ : Tile .real [BLOCK_M, BLOCK_N]) with hqk
  obtain ⟨rmaxT, hrm⟩ : ∃ t, Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length) qkT = some t :=
    ⟨_, by unfold Tile.reduceMaxDrop; rw [dif_pos (show 0 < TileShape.axisDim [BLOCK_M, BLOCK_N] (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length) from hBN)]⟩
  set rmaxT' : Tile .real [BLOCK_M] := rmaxT with hrmaxT'
  set pexpT : Tile .real [BLOCK_M, BLOCK_N] := Tile.uop WithBot.realExp
      (Tile.bop NumericDType.real.sub Broadcast.nil.consR.consSame qkT (Tile.expandDim ⟨1, by simp⟩ rmaxT')) with hpexp
  set lijT : Tile .real [BLOCK_M] := Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length) pexpT with hlij
  set miNewT : Tile .real [BLOCK_M] := Tile.select (Tile.cop ComparableDType.real.gt Broadcast.nil.consSame mtile rmaxT) mtile rmaxT with hminew
  set alphaT : Tile .real [BLOCK_M] := Tile.uop WithBot.realExp (Tile.bop NumericDType.real.sub Broadcast.nil.consSame mtile miNewT) with hal
  set betaT : Tile .real [BLOCK_M] := Tile.uop WithBot.realExp (Tile.bop NumericDType.real.sub Broadcast.nil.consSame rmaxT miNewT) with hbeta
  set liNewT : Tile .real [BLOCK_M] := Tile.bop NumericDType.real.add Broadcast.nil.consSame
      (Tile.bop NumericDType.real.mul Broadcast.nil.consSame alphaT ltile)
      (Tile.bop NumericDType.real.mul Broadcast.nil.consSame betaT lijT) with hlinew
  set pscaleT : Tile .real [BLOCK_M] := Tile.bop NumericDType.real.div Broadcast.nil.consSame betaT liNewT with hpscale
  set p2T : Tile .real [BLOCK_M, BLOCK_N] := Tile.bop NumericDType.real.mul Broadcast.nil.consR.consSame pexpT (Tile.expandDim ⟨1, by simp⟩ pscaleT) with hp2
  set accscale1T : Tile .real [BLOCK_M] := Tile.bop NumericDType.real.mul Broadcast.nil.consSame
      (Tile.bop NumericDType.real.div Broadcast.nil.consSame ltile liNewT) alphaT with hascale1
  set accscale2T : Tile .real [BLOCK_M] := Tile.select
      (⟨fun idx : TileIndex [BLOCK_M] => decide (SN ≤ gOM idx.1 + plen)⟩ : Tile .bool [BLOCK_M]) accscale1T
      (⟨fun _ : TileIndex [BLOCK_M] => some (1.0 : ℝ)⟩ : Tile .real [BLOCK_M]) with hascale2
  set acc1T : Tile .real [BLOCK_M, BLOCK_DMODEL] := Tile.bop NumericDType.real.mul Broadcast.nil.consR.consSame acctile (Tile.expandDim ⟨1, by simp⟩ accscale2T) with hacc1
  unfold bloomLoopBodyG
  -- stmt 0: start_n = ref start_n
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ref .nat [] "start_n") sin = some (Tile.scalar SN) from by rw [evalOp_ref]; exact hsn))]
  -- stmt 1: kv_loc = masked gather
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (bloom_kvloc_gather_evalG _ Req_to_tokens rqi SN bel stride_req_b stride_req_s
      (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hrqi]) (by simp only [BlockState.setReg_same])
      (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hn]) (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hbel])))]
  -- stmt 2: off_k
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (bloom_offk_gather_evalG _ ckvh kvf stride_kbs stride_kh stride_kd
      (by rw [BlockState.setReg_same, hkvf]; refine congrArg some ?_; ext idx;
          simp only [Tile.vec, BlockState.setReg_readMemValue])
      (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hckvh]) (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hd])))]
  -- stmt 3: k = masked load (2-cond)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ctx_evalOp_load_region_maskOther K (Op.ref .nat [BLOCK_DMODEL, BLOCK_N] "off_k") _ _ _
      (⟨fun idx : TileIndex [BLOCK_DMODEL, BLOCK_N] => kvf idx.2.1 * stride_kbs + ckvh * stride_kh + idx.1.val * stride_kd⟩ : Tile .nat [BLOCK_DMODEL, BLOCK_N])
      (⟨fun idx : TileIndex [BLOCK_DMODEL, BLOCK_N] => decide (SN + idx.2.1.val < bel) && decide (idx.1.val < head_dim)⟩ : Tile .bool [BLOCK_DMODEL, BLOCK_N])
      (⟨fun _ : TileIndex [BLOCK_DMODEL, BLOCK_N] => some (0.0 : ℝ)⟩ : Tile .real [BLOCK_DMODEL, BLOCK_N])
      (by rw [evalOp_ref]; simp [BlockState.setReg_same])
      (bloomKMask_evalG _ SN bel head_dim (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hsn])
        (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hn]) (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hd]) (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hbel]))
      (by simp only [evalOp, Option.bind_eq_bind, Option.bind_some]; rfl)))]
  -- stmt 4: qk = full 0
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (bloomQkFull_eval _))]
  -- stmt 5: qk = qk + dot q k
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (bloomQkAddDot_eval _ (⟨fun _ : TileIndex [BLOCK_M, BLOCK_N] => some (0 : ℝ)⟩ : Tile .real [BLOCK_M, BLOCK_N]) qtile kloadT
      (by simp only [BlockState.setReg_same])
      (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hq])
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), BlockState.setReg_same, hkl];
          refine congrArg some ?_; ext idx; simp only [BlockState.setReg_readMemValue])))]
  -- stmt 6: qk = qk * sm_scale
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (bloomQkScale_eval _ sm_scale qkdotT (by simp only [BlockState.setReg_same, hqkdot])))]
  -- stmt 7: qk = where(ge, qk, -1e8)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (bloomQkWhere_eval _ BLOCK_M BLOCK_N plen SN gOM qkscaleT
      (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hm]) (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hn])
      (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hplen]) (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hsn])
      (by simp only [BlockState.setReg_same, hqkscale])))]
  -- stmt 8: m_ij = reduceMax
  rw [stepStmts.cons_some (@stepStmt_assign_eq_some .real [BLOCK_M] "m_ij"
    (Op.reduceMax ⟨1, by simp⟩ Bool.false (Op.ref .real [BLOCK_M, BLOCK_N] "qk")) _ rmaxT
    (bloomMij_eval _ qkT rmaxT (by simp only [BlockState.setReg_same, hqk]) hrm))]
  -- stmt 9: p = exp(qk - m_ij)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (bloomP_eval _ qkT rmaxT (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hqk]) (by simp only [BlockState.setReg_same, hpexp, hlij, hminew, hal, hbeta, hlinew, hpscale, hp2, hascale1, hascale2, hacc1]; try rfl)))]
  -- stmt 10: l_ij = sum p 1
  rw [stepStmts.cons_some (@stepStmt_assign_eq_some .real [BLOCK_M] "l_ij"
    (Op.reduceSum (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length) Bool.false (Op.ref .real [BLOCK_M, BLOCK_N] "p")) _ lijT
    (bloomLij_eval _ pexpT (by simp only [BlockState.setReg_same, hpexp]; try rfl)))]
  -- stmt 11: m_i_new = maximum(m_i, m_ij)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (bloomMiNew_eval _ mtile rmaxT (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hmi]) (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hpexp, hlij, hminew, hal, hbeta, hlinew, hpscale, hp2, hascale1, hascale2, hacc1]; try rfl)))]
  -- stmt 12: alpha = exp(m_i - m_i_new)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (bloomExpSub_eval _ "m_i" "m_i_new" mtile miNewT (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hmi]) (by simp only [BlockState.setReg_same, hpexp, hlij, hminew, hal, hbeta, hlinew, hpscale, hp2, hascale1, hascale2, hacc1]; try rfl)))]
  -- stmt 13: beta = exp(m_ij - m_i_new)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (bloomExpSub_eval _ "m_ij" "m_i_new" rmaxT miNewT (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hpexp, hlij, hminew, hal, hbeta, hlinew, hpscale, hp2, hascale1, hascale2, hacc1]; try rfl) (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hpexp, hlij, hminew, hal, hbeta, hlinew, hpscale, hp2, hascale1, hascale2, hacc1]; try rfl)))]
  -- stmt 14: l_i_new = alpha*l_i + beta*l_ij
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (bloomLiNew_eval _ alphaT ltile betaT lijT
      (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hpexp, hlij, hminew, hal, hbeta, hlinew, hpscale, hp2, hascale1, hascale2, hacc1]; try rfl)
      (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hli])
      (by simp only [BlockState.setReg_same, hpexp, hlij, hminew, hal, hbeta, hlinew, hpscale, hp2, hascale1, hascale2, hacc1]; try rfl)
      (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hpexp, hlij, hminew, hal, hbeta, hlinew, hpscale, hp2, hascale1, hascale2, hacc1]; try rfl)))]
  -- stmt 15: p_scale = beta / l_i_new
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (bloomPscale_eval _ betaT liNewT (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hpexp, hlij, hminew, hal, hbeta, hlinew, hpscale, hp2, hascale1, hascale2, hacc1]; try rfl) (by simp only [BlockState.setReg_same, hpexp, hlij, hminew, hal, hbeta, hlinew, hpscale, hp2, hascale1, hascale2, hacc1]; try rfl)))]
  -- stmt 16: p = p * p_scale
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (bloomP2_eval _ pexpT pscaleT
      (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hpexp, hlij, hminew, hal, hbeta, hlinew, hpscale, hp2, hascale1, hascale2, hacc1]; try rfl)
      (by simp only [BlockState.setReg_same, hpexp, hlij, hminew, hal, hbeta, hlinew, hpscale, hp2, hascale1, hascale2, hacc1]; try rfl)))]
  -- stmt 17: acc_scale = (l_i/l_i_new)*alpha
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (bloomAccScale1_eval _ ltile liNewT alphaT
      (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hli])
      (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hpexp, hlij, hminew, hal, hbeta, hlinew, hpscale, hp2, hascale1, hascale2, hacc1]; try rfl)
      (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hpexp, hlij, hminew, hal, hbeta, hlinew, hpscale, hp2, hascale1, hascale2, hacc1]; try rfl)))]
  -- stmt 18: acc_scale = where(ge, acc_scale, 1)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (bloomAccScale2_eval _ BLOCK_M plen SN gOM accscale1T
      (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hm])
      (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hplen])
      (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hsn])
      (by simp only [BlockState.setReg_same, hpexp, hlij, hminew, hal, hbeta, hlinew, hpscale, hp2, hascale1, hascale2, hacc1]; try rfl)))]
  -- stmt 19: acc = acc * acc_scale
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (bloomAcc1_eval _ acctile accscale2T
      (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hacc])
      (by simp only [BlockState.setReg_same, hpexp, hlij, hminew, hal, hbeta, hlinew, hpscale, hp2, hascale1, hascale2, hacc1]; try rfl)))]
  -- stmt 20: off_v = gather
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (bloom_offv_gather_evalG _ ckvh kvf stride_vbs stride_vh stride_vd
      (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hkvf];
          refine congrArg some ?_; ext idx; simp only [Tile.vec, BlockState.setReg_readMemValue])
      (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hckvh]) (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hd])))]
  -- stmt 21: v = masked load (2-cond)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ctx_evalOp_load_region_maskOther V (Op.ref .nat [BLOCK_N, BLOCK_DMODEL] "off_v") _ _ _
      (⟨fun idx : TileIndex [BLOCK_N, BLOCK_DMODEL] => kvf idx.1 * stride_vbs + ckvh * stride_vh + idx.2.1.val * stride_vd⟩ : Tile .nat [BLOCK_N, BLOCK_DMODEL])
      (⟨fun idx : TileIndex [BLOCK_N, BLOCK_DMODEL] => decide (SN + idx.1.val < bel) && decide (idx.2.1.val < head_dim)⟩ : Tile .bool [BLOCK_N, BLOCK_DMODEL])
      (⟨fun _ : TileIndex [BLOCK_N, BLOCK_DMODEL] => some (0.0 : ℝ)⟩ : Tile .real [BLOCK_N, BLOCK_DMODEL])
      (by rw [evalOp_ref]; simp [BlockState.setReg_same])
      (bloomVMask_evalG _ SN bel head_dim (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hsn])
        (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hn]) (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hd]) (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hbel]))
      (by simp only [evalOp, Option.bind_eq_bind, Option.bind_some]; rfl)))]
  -- stmt 22: p = ref p (noop)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ref .real [BLOCK_M, BLOCK_N] "p") _ = some p2T from by rw [evalOp_ref]; simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hp2]; try rfl))]
  -- stmt 23: acc = acc + dot(p, v)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (bloomAcc2_eval _ acc1T p2T vloadT
      (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hpexp, hlij, hminew, hal, hbeta, hlinew, hpscale, hp2, hascale1, hascale2, hacc1]; try rfl)
      (by simp only [BlockState.setReg_same, hpexp, hlij, hminew, hal, hbeta, hlinew, hpscale, hp2, hascale1, hascale2, hacc1]; try rfl)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), BlockState.setReg_same, hvl];
          refine congrArg some ?_; ext idx; simp only [BlockState.setReg_readMemValue])))]
  -- stmt 24: l_i = ref l_i_new
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ref .real [BLOCK_M] "l_i_new") _ = some liNewT from by rw [evalOp_ref]; simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hlinew]; try rfl))]
  -- stmt 25: m_i = ref m_i_new
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ref .real [BLOCK_M] "m_i_new") _ = some miNewT from by rw [evalOp_ref]; simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hminew]; try rfl))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_,
    kloadT, vloadT, qkT, rmaxT, miNewT, alphaT, betaT, lijT, pscaleT, liNewT, accscale2T,
    pexpT, p2T, acc1T,
    rfl, rfl, rfl, hrm, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, ?_, ?_, ?_⟩
  · simp only [BlockState.setReg_pids]
  · funext rg o; simp only [BlockState.setReg_mem]
  · intro rg o; simp only [BlockState.setReg_undef]; exact hundef rg o
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq]
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hcb]
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hckvh]
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hch]
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hplen]
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hsl]
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hbel]
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hcbsi]
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hrqi]
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hm]
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hn]
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hd]
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hq]
  · rw [BlockState.setReg_same]
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq]
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq]


/-! ### General per-key data + streaming-loop invariant -/

/-- General per-key data (`bloomKVMG`) carried by the loop registers at output
lane `(i, d)`. -/
noncomputable def bloomGG
    (s0 : BlockState) (Q K V B_Start_Loc B_Seqlen B_Prompt_Cache_Len Req_to_tokens B_req_idx : RegionName)
    (sm_scale : ℝ)
    (stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd head_dim BLOCK_DMODEL BLOCK_M S bel : Nat)
    (i : Fin BLOCK_M) (d : Nat) : Fin S → ℝ × ℝ :=
  bloomKVMG s0 Q K V B_Start_Loc B_Seqlen B_Prompt_Cache_Len Req_to_tokens B_req_idx sm_scale
    stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
    stride_vbs stride_vh stride_vd head_dim BLOCK_DMODEL BLOCK_M S bel i d

/-- **General streaming-loop invariant** unifying `bloomInvariant`/`bloomInvariant64`
over `BLOCK_M`/`BLOCK_N`/`BLOCK_DMODEL`/`head_dim`, the gather strides and the K/V/Q
strides. After `c` blocks the in-loop-normalized registers carry the ⊥-seeded fold
`gStateBot (c·BLOCK_N)` over `bloomGG`: `m_i = log2·(running max)`, `l_i = (running
denom)`, `acc = gAccN` (normalized accumulator). `kv_group_num = 1`; `0 <
BLOCK_DMODEL` for the channel-0 score lane. -/
noncomputable def bloomInvariantG
    (Q K V Out B_Start_Loc B_Seqlen Req_to_tokens B_req_idx B_Prompt_Cache_Len : RegionName) (s0 : BlockState)
    (sm_scale : ℝ)
    (stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd head_dim BLOCK_DMODEL BLOCK_N BLOCK_M : Nat) (hD : 0 < BLOCK_DMODEL)
    (S bel : Nat) (c : Nat) (s : BlockState) : Prop :=
  let plen := promptLen s0 B_Prompt_Cache_Len
  let sl := seqLen s0 B_Seqlen B_Prompt_Cache_Len
  let g := fun (i : Fin BLOCK_M) (d : Nat) =>
    bloomGG s0 Q K V B_Start_Loc B_Seqlen B_Prompt_Cache_Len Req_to_tokens B_req_idx sm_scale
      stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd head_dim BLOCK_DMODEL BLOCK_M S bel i d
  s.pids = s0.pids ∧ s.mem = s0.mem ∧ (∀ rg o, s.undef rg o = 0) ∧
  (s.regs .nat [] "cur_batch" = some (Tile.scalar (s0.pids 0))) ∧
  (s.regs .nat [] "cur_kv_head" = some (Tile.scalar (s0.pids 1 / 1))) ∧
  (s.regs .nat [] "cur_head" = some (Tile.scalar (s0.pids 1))) ∧
  (s.regs .nat [] "prompt_cache_len" = some (Tile.scalar plen)) ∧
  (s.regs .nat [] "cur_batch_seq_len" = some (Tile.scalar sl)) ∧
  (s.regs .nat [] "block_end_loc" = some (Tile.scalar bel)) ∧
  (s.regs .nat [] "cur_batch_in_all_start_index" = some (Tile.scalar (startLoc s0 B_Start_Loc))) ∧
  (s.regs .nat [] "cur_batch_req_idx" = some (Tile.scalar (reqIdx s0 B_req_idx))) ∧
  (s.regs .nat [BLOCK_M] "offs_m" = some (Tile.vec (fun r : Fin BLOCK_M => s0.pids 2 * BLOCK_M + r.val))) ∧
  (s.regs .nat [BLOCK_N] "offs_n" = some (Tile.vec (fun j : Fin BLOCK_N => j.val))) ∧
  (s.regs .nat [BLOCK_DMODEL] "offs_d" = some (Tile.vec (fun e : Fin BLOCK_DMODEL => e.val))) ∧
  (s.regs .real [BLOCK_M, BLOCK_DMODEL] "q" = some ⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
      if decide (s0.pids 2 * BLOCK_M + idx.1.val < sl) && decide (idx.2.1.val < head_dim) then
        s0.readMemValue .real (Region.cast Q)
          ((startLoc s0 B_Start_Loc + (s0.pids 2 * BLOCK_M + idx.1.val)) * stride_qbs
            + s0.pids 1 * stride_qh + idx.2.1.val * stride_qd)
      else some (0.0 : ℝ)⟩) ∧
  (s.regs .real [BLOCK_M] "m_i" = some ⟨fun r : TileIndex [BLOCK_M] =>
      (gStateBot S (c * BLOCK_N) (g r.1 (⟨0, hD⟩ : Fin BLOCK_DMODEL).val)).1.map (· * Real.log 2)⟩) ∧
  (s.regs .real [BLOCK_M] "l_i" = some ⟨fun r : TileIndex [BLOCK_M] =>
      some (gStateBot S (c * BLOCK_N) (g r.1 (⟨0, hD⟩ : Fin BLOCK_DMODEL).val)).2.1⟩) ∧
  (s.regs .real [BLOCK_M, BLOCK_DMODEL] "acc" = some ⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
      some (gAccN S BLOCK_N (s0.pids 2 * BLOCK_M + idx.1.val + plen) (g idx.1 idx.2.1.val) c)⟩) ∧
  (c * BLOCK_N ≤ S)

/-! ### General preLoop eval (→ `bloomInvariantG 0`) -/

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **General `bloomPreLoopG_eval`.** The general prologue steps to a state seeding
the loop-entry invariant (mirror of `bloomPreLoop_eval`, strides/head_dim/BLOCK_*
+ kv_group_num free; bloom pid layout `pids 0/1/2 = cur_batch/cur_head/start_m`). -/
theorem bloomPreLoopG_eval
    (s : BlockState) (Q : RegionName)
    (B_Start_Loc B_Seqlen B_req_idx b_prompt_cache_len : Region .nat)
    (stride_qbs stride_qh stride_qd head_dim kv_group_num BLOCK_DMODEL BLOCK_M BLOCK_N : Nat)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ∃ s0, stepStmts (bloomPreLoopG Q B_Start_Loc B_Seqlen B_req_idx b_prompt_cache_len
            stride_qbs stride_qh stride_qd head_dim kv_group_num BLOCK_DMODEL BLOCK_M BLOCK_N) s = some s0
      ∧ s0.pids = s.pids ∧ s0.mem = s.mem ∧ (∀ rg o, s0.undef rg o = 0)
      ∧ s0.regs .nat [] "cur_batch" = some (Tile.scalar (s.pids 0))
      ∧ s0.regs .nat [] "cur_head" = some (Tile.scalar (s.pids 1))
      ∧ s0.regs .nat [] "start_m" = some (Tile.scalar (s.pids 2))
      ∧ s0.regs .nat [] "cur_kv_head" = some (Tile.scalar (s.pids 1 / kv_group_num))
      ∧ s0.regs .nat [] "prompt_cache_len"
          = some (Tile.scalar (s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 0)))
      ∧ s0.regs .nat [] "cur_batch_in_all_start_index"
          = some (Tile.scalar (s.readMemValue .nat (Region.cast B_Start_Loc) (s.pids 0)))
      ∧ s0.regs .nat [] "cur_batch_req_idx"
          = some (Tile.scalar (s.readMemValue .nat (Region.cast B_req_idx) (s.pids 0)))
      ∧ s0.regs .nat [] "cur_batch_seq_len"
          = some (Tile.scalar (s.readMemValue .nat (Region.cast B_Seqlen) (s.pids 0)
              - s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 0)))
      ∧ s0.regs .nat [BLOCK_N] "offs_n" = some (Tile.vec (fun j : Fin BLOCK_N => j.val))
      ∧ s0.regs .nat [BLOCK_DMODEL] "offs_d" = some (Tile.vec (fun e : Fin BLOCK_DMODEL => e.val))
      ∧ s0.regs .nat [BLOCK_M] "offs_m" = some (Tile.vec (fun r : Fin BLOCK_M => s.pids 2 * BLOCK_M + r.val))
      ∧ s0.regs .real [BLOCK_M] "m_i" = some ⟨fun _ : TileIndex [BLOCK_M] => (⊥ : WithBot ℝ)⟩
      ∧ s0.regs .real [BLOCK_M] "l_i" = some ⟨fun _ : TileIndex [BLOCK_M] => some (0 : ℝ)⟩
      ∧ s0.regs .real [BLOCK_M, BLOCK_DMODEL] "acc" = some ⟨fun _ : TileIndex [BLOCK_M, BLOCK_DMODEL] => some (0 : ℝ)⟩
      ∧ s0.regs .nat [] "block_end_loc"
          = some (Tile.scalar
              (let a := (s.pids 2 + 1) * BLOCK_M + s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 0)
               let b := (s.readMemValue .nat (Region.cast B_Seqlen) (s.pids 0)
                   - s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 0))
                 + s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 0)
               if a < b then a else b))
      ∧ s0.regs .nat [] "block_mask"
          = some (Tile.scalar
              (if BLOCK_M * s.pids 2 < s.readMemValue .nat (Region.cast B_Seqlen) (s.pids 0)
                  - s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 0) then 1 else 0))
      ∧ s0.regs .real [BLOCK_M, BLOCK_DMODEL] "q" = some ⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          if decide (s.pids 2 * BLOCK_M + idx.1.val
              < s.readMemValue .nat (Region.cast B_Seqlen) (s.pids 0)
                - s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 0)) && decide (idx.2.1.val < head_dim) then
            s.readMemValue .real (Region.cast Q)
              ((s.readMemValue .nat (Region.cast B_Start_Loc) (s.pids 0) + (s.pids 2 * BLOCK_M + idx.1.val))
                  * stride_qbs + s.pids 1 * stride_qh + idx.2.1.val * stride_qd)
          else some (0.0 : ℝ)⟩ := by
  unfold bloomPreLoopG
  -- stmt 0: cur_batch = programId 0
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 0 s))]
  -- stmt 1: cur_head = programId 1
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 1 _))]
  -- stmt 2: start_m = programId 2
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 2 _))]
  -- stmt 3: cur_kv_head = cur_head // kv_group_num
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.floorDiv .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat kv_group_num)) _
        = some (Tile.scalar (s.pids 1 / kv_group_num)) from by
      rw [ctx_evalOp_floorDiv]
      simp only [evalOp_ref, evalOp_constNat, BlockState.setReg_same, BlockState.setReg_ne_name,
        ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, BlockState.setReg_pids,
        Option.bind_eq_bind, Option.bind_some]
      rfl))]
  -- stmt 4: cur_batch_in_all_start_index = load(B_Start_Loc + cur_batch)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ctx_evalOp_load_scalar_nat B_Start_Loc (Op.ref .nat [] "cur_batch") _ (s.pids 0)
      (by rw [evalOp_ref]; simp [BlockState.setReg_ne_name, BlockState.setReg_same, BlockState.setReg_pids])))]
  -- stmt 5: prompt_cache_len = load(bpc + cur_batch)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ctx_evalOp_load_scalar_nat b_prompt_cache_len (Op.ref .nat [] "cur_batch") _ (s.pids 0)
      (by rw [evalOp_ref]; simp [BlockState.setReg_ne_name, BlockState.setReg_same, BlockState.setReg_pids])))]
  -- stmt 6: cur_batch_seq_len = load(B_Seqlen + cur_batch) - prompt_cache_len
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.sub .nat Broadcast.nil
        (Op.load .nat (MemAccess.region B_Seqlen (Op.ref .nat [] "cur_batch")) MaskOpt.none)
        (Op.ref .nat [] "prompt_cache_len")) _
        = some (Tile.scalar (s.readMemValue .nat (Region.cast B_Seqlen) (s.pids 0)
            - s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 0))) from by
      rw [evalOp_sub, ctx_evalOp_load_scalar_nat B_Seqlen (Op.ref .nat [] "cur_batch") _ (s.pids 0)
        (by rw [evalOp_ref]; simp [BlockState.setReg_ne_name, BlockState.setReg_same, BlockState.setReg_pids])]
      simp only [evalOp_ref, BlockState.setReg_same, Option.bind_eq_bind, Option.bind_some]
      rfl))]
  -- stmt 7: cur_batch_req_idx = load(B_req_idx + cur_batch)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ctx_evalOp_load_scalar_nat B_req_idx (Op.ref .nat [] "cur_batch") _ (s.pids 0)
      (by rw [evalOp_ref]; simp [BlockState.setReg_ne_name, BlockState.setReg_same, BlockState.setReg_pids])))]
  -- stmt 8: block_start_loc = BLOCK_M * start_m
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.mul .nat Broadcast.nil (Op.constNat BLOCK_M) (Op.ref .nat [] "start_m")) _
        = some (Tile.scalar (BLOCK_M * s.pids 2)) from by
      rw [evalOp_mul]
      simp only [evalOp_constNat, evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name,
        ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, BlockState.setReg_pids,
        Option.bind_eq_bind, Option.bind_some]
      rfl))]
  -- stmt 9: offs_n = arange BLOCK_N
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.arange BLOCK_N) _ = some (Tile.vec (fun j : Fin BLOCK_N => j.val)) from evalOp_arange BLOCK_N _))]
  -- stmt 10: offs_d = arange BLOCK_DMODEL
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.arange BLOCK_DMODEL) _ = some (Tile.vec (fun e : Fin BLOCK_DMODEL => e.val)) from evalOp_arange BLOCK_DMODEL _))]
  -- stmt 11: offs_m = start_m * BLOCK_M + arange BLOCK_M
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat BLOCK_M)) (Op.arange BLOCK_M)) _
        = some (Tile.vec (fun r : Fin BLOCK_M => s.pids 2 * BLOCK_M + r.val)) from by
      rw [evalOp_add, evalOp_mul]
      simp only [evalOp_ref, evalOp_arange, evalOp_constNat, BlockState.setReg_same, BlockState.setReg_ne_name,
        ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, BlockState.setReg_pids,
        Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext r
      simp [Tile.bop_data, Tile.scalar_data, Tile.vec_data, Broadcast.leftIndex, Broadcast.rightIndex,
        NumericDType.add, NumericDType.mul]))]
  -- stmt 12: off_q
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .nat Broadcast.nil.consL.consR
        (Op.add .nat Broadcast.scalarR
          (Op.mul .nat Broadcast.scalarR
            (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "cur_batch_in_all_start_index")
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m")))
            (Op.constNat stride_qbs))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat stride_qh)))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_DMODEL] "offs_d"))
          (Op.constNat stride_qd))) _
        = some (⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
            (s.readMemValue .nat (Region.cast B_Start_Loc) (s.pids 0) + (s.pids 2 * BLOCK_M + idx.1.val))
                * stride_qbs + s.pids 1 * stride_qh + idx.2.1.val * stride_qd⟩ : Tile .nat [BLOCK_M, BLOCK_DMODEL]) from by
      simp only [evalOp_add, evalOp_mul, evalOp_constNat, evalOp_ref,
        ctx_evalOp_expandDim_one_nat, ctx_evalOp_expandDim_zero_nat,
        BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
        not_false_eq_true, reduceCtorEq, BlockState.setReg_pids, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp [Tile.bop_data, Tile.scalar_data, Tile.vec_data, Tile.expandDim, Broadcast.leftIndex,
        Broadcast.rightIndex, NumericDType.add, NumericDType.mul]))]
  -- stmt 13: q = masked load (2-cond)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ctx_evalOp_load_region_maskOther Q (Op.ref .nat [BLOCK_M, BLOCK_DMODEL] "off_q") _ _ _
      (⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          (s.readMemValue .nat (Region.cast B_Start_Loc) (s.pids 0) + (s.pids 2 * BLOCK_M + idx.1.val))
              * stride_qbs + s.pids 1 * stride_qh + idx.2.1.val * stride_qd⟩ : Tile .nat [BLOCK_M, BLOCK_DMODEL])
      (⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => decide (s.pids 2 * BLOCK_M + idx.1.val
          < s.readMemValue .nat (Region.cast B_Seqlen) (s.pids 0)
            - s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 0)) && decide (idx.2.1.val < head_dim)⟩ : Tile .bool [BLOCK_M, BLOCK_DMODEL])
      (⟨fun _ : TileIndex [BLOCK_M, BLOCK_DMODEL] => some (0.0 : ℝ)⟩ : Tile .real [BLOCK_M, BLOCK_DMODEL])
      (by rw [evalOp_ref]; simp [BlockState.setReg_same])
      (bloomQMask_evalG _ (fun r : Fin BLOCK_M => s.pids 2 * BLOCK_M + r.val) _ head_dim
        (by simp [BlockState.setReg_ne_name, BlockState.setReg_same])
        (by simp [BlockState.setReg_ne_name, BlockState.setReg_same])
        (by simp [BlockState.setReg_ne_name, BlockState.setReg_same]))
      (by simp only [evalOp, Option.bind_eq_bind, Option.bind_some]; rfl)))]
  -- stmt 14: m_i = full 0 + (-inf) = full ⊥
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .real Broadcast.scalarR (Op.full [BLOCK_M] (Op.const 0)) Op.negInf) _
        = some (⟨fun _ : TileIndex [BLOCK_M] => (⊥ : WithBot ℝ)⟩ : Tile .real [BLOCK_M]) from by
      rw [evalOp_add]
      simp only [evalOp_full, evalOp_const, evalOp_negInf, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp only [Tile.bop_data, Tile.scalar_data, Broadcast.leftIndex, Broadcast.rightIndex,
        NumericDType.add, WithBot.realAdd, Option.map₂, Option.bind, Option.map]
      rfl))]
  -- stmt 15: l_i = full 0
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.full [BLOCK_M] (Op.const 0)) _
        = some (⟨fun _ : TileIndex [BLOCK_M] => some (0 : ℝ)⟩ : Tile .real [BLOCK_M]) from by
      simp [evalOp_full, evalOp_const]))]
  -- stmt 16: acc = full 0
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.full [BLOCK_M, BLOCK_DMODEL] (Op.const 0)) _
        = some (⟨fun _ : TileIndex [BLOCK_M, BLOCK_DMODEL] => some (0 : ℝ)⟩ : Tile .real [BLOCK_M, BLOCK_DMODEL]) from by
      simp [evalOp_full, evalOp_const]))]
  -- stmt 17: block_mask = where (block_start_loc < seqlen) 1 0
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp ((Op.lt .nat Broadcast.nil (Op.ref .nat [] "block_start_loc")
            (Op.ref .nat [] "cur_batch_seq_len")).where (Op.constNat 1) (Op.constNat 0)) _
        = some (Tile.scalar (if BLOCK_M * s.pids 2 < s.readMemValue .nat (Region.cast B_Seqlen) (s.pids 0)
            - s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 0) then 1 else 0)) from by
      rw [evalOp_where]
      simp only [evalOp_lt, evalOp_ref, evalOp_constNat, BlockState.setReg_same,
        BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq,
        Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_
      ext _idx
      simp only [Tile.select_data, Tile.cop_data, Tile.scalar_data_index, ComparableDType.lt]
      by_cases h : BLOCK_M * s.pids 2 < s.readMemValue .nat (Region.cast B_Seqlen) (s.pids 0)
          - s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 0)
      · simp [h]
      · simp [h]))]
  -- stmt 18: block_end_loc
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp ((Op.lt .nat Broadcast.nil
            (Op.add .nat Broadcast.nil
              (Op.mul .nat Broadcast.nil
                (Op.add .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat 1)) (Op.constNat BLOCK_M))
              (Op.ref .nat [] "prompt_cache_len"))
            (Op.add .nat Broadcast.nil (Op.ref .nat [] "cur_batch_seq_len")
              (Op.ref .nat [] "prompt_cache_len"))).where
          (Op.add .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil
              (Op.add .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat 1)) (Op.constNat BLOCK_M))
            (Op.ref .nat [] "prompt_cache_len"))
          (Op.add .nat Broadcast.nil (Op.ref .nat [] "cur_batch_seq_len")
            (Op.ref .nat [] "prompt_cache_len"))) _
        = some (Tile.scalar
            (let a := (s.pids 2 + 1) * BLOCK_M + s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 0)
             let b := (s.readMemValue .nat (Region.cast B_Seqlen) (s.pids 0)
                 - s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 0))
               + s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 0)
             if a < b then a else b)) from by
      rw [evalOp_where]
      simp only [evalOp_lt, evalOp_add, evalOp_mul, evalOp_ref, evalOp_constNat, BlockState.setReg_same,
        BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq,
        Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_
      ext _idx
      simp only [Tile.select_data, Tile.cop_data, Tile.bop_data, Tile.scalar_data_index,
        ComparableDType.lt, NumericDType.add, NumericDType.mul, Broadcast.leftIndex, Broadcast.rightIndex]
      by_cases h : ((s.pids 2 + 1) * BLOCK_M + s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 0))
          < (s.readMemValue .nat (Region.cast B_Seqlen) (s.pids 0)
              - s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 0))
            + s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 0)
      · simp [h]
      · simp [h]))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp only [BlockState.setReg_pids]
  · funext rg o; simp only [BlockState.setReg_mem]
  · intro rg o; simp only [BlockState.setReg_undef]; exact hundef rg o
  all_goals
    simp only [BlockState.setReg_ne_name, BlockState.setReg_same, BlockState.setReg_pids,
      BlockState.setReg_readMemValue, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq]


/-! ### General `bloom_attn_stepG` (loop body advances `bloomInvariantG` one block) -/

set_option maxHeartbeats 2400000 in
set_option maxRecDepth 8000 in
/-- **General `bloom_attn_stepG`.** The streaming loop body advances `bloomInvariantG`
by one block. Mirror of `bloom_attn_step` over symbolic dims/strides + free gather
strides/head_dim (`kv_group_num = 1`; `0 < BLOCK_DMODEL` for the channel-0 score
lane, `0 < BLOCK_N` for the block reductions). Natural-exp in-loop normalization is
preserved: `m_i = log2·runningmax`, `l_i = denom`, `acc = gAccN` (normalized). -/
theorem bloom_attn_stepG
    (Q K V Out B_Start_Loc B_Seqlen Req_to_tokens B_req_idx B_Prompt_Cache_Len : RegionName) (s0 : BlockState)
    (sm_scale : ℝ)
    (stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd
      BLOCK_DMODEL BLOCK_N BLOCK_M head_dim : Nat) (hD : 0 < BLOCK_DMODEL) (hBN : 0 < BLOCK_N)
    (S bel : Nat) (c : Nat) (i : Nat) (s : BlockState)
    (hwin : (c + 1) * BLOCK_N ≤ S) (hieq : i = c * BLOCK_N)
    (hinv : bloomInvariantG Q K V Out B_Start_Loc B_Seqlen Req_to_tokens B_req_idx B_Prompt_Cache_Len s0
        sm_scale stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
        stride_vbs stride_vh stride_vd head_dim BLOCK_DMODEL BLOCK_N BLOCK_M hD S bel c s) :
    ∃ s', stepStmts (bloomLoopBodyG Q K V Req_to_tokens B_req_idx sm_scale stride_req_b stride_req_s
            stride_kbs stride_kh stride_kd stride_vbs stride_vh stride_vd head_dim BLOCK_DMODEL BLOCK_M BLOCK_N)
          (s.setReg "start_n" .nat [] (Tile.scalar i)) = some s'
      ∧ bloomInvariantG Q K V Out B_Start_Loc B_Seqlen Req_to_tokens B_req_idx B_Prompt_Cache_Len s0
          sm_scale stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
          stride_vbs stride_vh stride_vd head_dim BLOCK_DMODEL BLOCK_N BLOCK_M hD S bel (c + 1) s' := by
  subst hieq
  set plen := promptLen s0 B_Prompt_Cache_Len with hplend
  set sl := seqLen s0 B_Seqlen B_Prompt_Cache_Len with hsld
  set d0 : Nat := (⟨0, hD⟩ : Fin BLOCK_DMODEL).val with hd0def
  set g := fun (i : Fin BLOCK_M) (d : Nat) =>
    bloomGG s0 Q K V B_Start_Loc B_Seqlen B_Prompt_Cache_Len Req_to_tokens B_req_idx sm_scale
      stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd head_dim BLOCK_DMODEL BLOCK_M S bel i d with hgd
  simp only [bloomInvariantG] at hinv
  obtain ⟨hpids, hmem, hundef, hcb, hckvh, hch, hplen, hsl, hbel, hcbsi, hrqi, hom, hon, hod,
      hq, hmi, hli, hacc, hcle⟩ := hinv
  set rqi : Nat := reqIdx s0 B_req_idx with hrqid
  set qtile : Tile .real [BLOCK_M, BLOCK_DMODEL] := ⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
      if decide (s0.pids 2 * BLOCK_M + idx.1.val < sl) && decide (idx.2.1.val < head_dim) then
        s0.readMemValue .real (Region.cast Q)
          ((startLoc s0 B_Start_Loc + (s0.pids 2 * BLOCK_M + idx.1.val)) * stride_qbs
            + s0.pids 1 * stride_qh + idx.2.1.val * stride_qd)
      else some (0.0 : ℝ)⟩ with hqtile
  set mtile : Tile .real [BLOCK_M] := ⟨fun r : TileIndex [BLOCK_M] =>
      (gStateBot S (c * BLOCK_N) (g r.1 d0)).1.map (· * Real.log 2)⟩ with hmtile
  set ltile : Tile .real [BLOCK_M] := ⟨fun r : TileIndex [BLOCK_M] =>
      some (gStateBot S (c * BLOCK_N) (g r.1 d0)).2.1⟩ with hltile
  set acctile : Tile .real [BLOCK_M, BLOCK_DMODEL] := ⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
      some (gAccN S BLOCK_N (s0.pids 2 * BLOCK_M + idx.1.val + plen) (g idx.1 idx.2.1.val) c)⟩ with hacctile
  obtain ⟨sF, hchain, hpidsF, hmemF, hundefF, hsnF, hcbF, hckvhF, hchF, hplenF, hslF, hbelF,
      hcbsiF, hrqiF, homF, honF, hodF, hqF,
      kloadT, vloadT, qkT, rmaxT, miNewT, alphaT, betaT, lijT, pscaleT, liNewT, accscale2T,
      pexpT, p2T, acc1T,
      hkleq, hvleq, hqkeq, hrm, hpexpeq, hlijeq, hminew, haleq, hbetaeq, hlineweq, hpscaleeq, hp2eq,
      hascale2eq, hacc1eq, hm_iF, hl_iF, haccF⟩ :=
    bloomLoopBody_stepsG Q K V Req_to_tokens B_req_idx sm_scale stride_req_b stride_req_s
      stride_kbs stride_kh stride_kd stride_vbs stride_vh stride_vd head_dim BLOCK_DMODEL BLOCK_M BLOCK_N hBN
      (s.setReg "start_n" .nat [] (Tile.scalar (c * BLOCK_N))) (c * BLOCK_N)
      (fun r : Fin BLOCK_M => s0.pids 2 * BLOCK_M + r.val)
      (s0.pids 0) (s0.pids 1 / 1) (s0.pids 1) plen sl bel
      (startLoc s0 B_Start_Loc) rqi
      qtile mtile ltile acctile
      (by rw [BlockState.setReg_same])
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hcb)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hckvh)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hch)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hplen)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hsl)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hbel)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hcbsi)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hrqi)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hom)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hon)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hod)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hq)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hmi)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hli)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hacc)
      (by intro rg o; simp [BlockState.setReg_undef, hundef])
  refine ⟨sF, hchain, ?_⟩
  have hckvhd : s0.pids 1 / 1 = s0.pids 1 := Nat.div_one _
  have hrmem : ∀ (R : RegionName) (o : Nat),
      (s.setReg "start_n" .nat [] (Tile.scalar (c * BLOCK_N))).readMem R o = s0.readMem R o := by
    intro R o; rw [BlockState.setReg_readMem]; unfold BlockState.readMem; rw [hmem]
  have hrmemN : ∀ (R : RegionName) (o : Nat),
      s.readMemValue .nat R o = s0.readMemValue .nat R o := by
    intro R o; simp only [BlockState.readMemValue, BlockState.readMemTyped, hmem]
  have hrmemR : ∀ (R : RegionName) (o : Nat), s.readMem R o = s0.readMem R o := by
    intro R o; unfold BlockState.readMem; rw [hmem]
  -- per-cell q readback: qtile = bloomQTileMG
  have hqf : ∀ (a : Fin BLOCK_M) (e : Fin BLOCK_DMODEL),
      qtile.data (a, e, PUnit.unit)
        = some (bloomQTileMG s0 Q B_Start_Loc B_Seqlen B_Prompt_Cache_Len stride_qbs stride_qh stride_qd head_dim BLOCK_M a e.val) := by
    intro a e
    rw [hqtile]
    show (if decide (s0.pids 2 * BLOCK_M + a.val < sl) && decide (e.val < head_dim) then
        s0.readMemValue .real (Region.cast Q)
          ((startLoc s0 B_Start_Loc + (s0.pids 2 * BLOCK_M + a.val)) * stride_qbs + s0.pids 1 * stride_qh + e.val * stride_qd)
      else some (0.0 : ℝ)) = _
    simp only [bloomQTileMG, bloomQTileG, curHead, ← hsld]
    by_cases h1 : s0.pids 2 * BLOCK_M + a.val < sl
    · by_cases h2 : e.val < head_dim
      · rw [if_pos (by simp only [decide_eq_true_eq, Bool.and_eq_true]; exact ⟨h1, h2⟩),
          if_pos (And.intro h1 h2)]
        simp only [BlockState.readMemValue_real, Region.cast_id]
      · rw [if_neg (by simp only [decide_eq_false_iff_not.mpr h2, Bool.and_false, Bool.false_eq_true, not_false_eq_true]),
          if_neg (fun hcon => h2 hcon.2)]; norm_num
    · rw [if_neg (by simp only [decide_eq_false_iff_not.mpr h1, Bool.false_and, Bool.false_eq_true, not_false_eq_true]),
        if_neg (fun hcon => h1 hcon.1)]; norm_num
  -- per-cell k readback: kloadT = bloomKTileMG (at global key c*BN+jL)
  have hkf : ∀ (jL : Fin BLOCK_N) (e : Fin BLOCK_DMODEL),
      kloadT.data (e, jL, PUnit.unit)
        = some (bloomKTileMG s0 K Req_to_tokens B_req_idx stride_req_b stride_req_s stride_kbs stride_kh stride_kd head_dim S bel
            ⟨c * BLOCK_N + jL.val, gBlock_idx_lt S BLOCK_N c hwin jL⟩ e.val) := by
    intro jL e
    rw [hkleq]
    show (if decide (c * BLOCK_N + jL.val < bel) && decide (e.val < head_dim) then
        (s.setReg "start_n" .nat [] (Tile.scalar (c * BLOCK_N))).readMemValue .real (Region.cast K)
          ((if decide (c * BLOCK_N + jL.val < bel) then
              (s.setReg "start_n" .nat [] (Tile.scalar (c * BLOCK_N))).readMemValue .nat (Region.cast Req_to_tokens)
                (stride_req_b * rqi + stride_req_s * (c * BLOCK_N + jL.val))
            else 0) * stride_kbs + (s0.pids 1 / 1) * stride_kh + e.val * stride_kd)
        else some (0.0 : ℝ)) = _
    simp only [bloomKTileMG, bloomKTileG, bloomKvLocG, curHead]
    by_cases h : c * BLOCK_N + jL.val < bel
    · by_cases h2 : e.val < head_dim
      · rw [if_pos (by simp only [decide_eq_true_eq, Bool.and_eq_true]; exact ⟨h, h2⟩),
          if_pos (And.intro h h2), if_pos (by simp only [decide_eq_true_eq]; exact h)]
        simp only [BlockState.setReg_readMem, BlockState.setReg_readMemValue,
          BlockState.readMemValue_real, Region.cast_id, hckvhd, hrqid, hrmemN, hrmemR,
          Nat.mul_comm stride_req_b (reqIdx s0 B_req_idx)]
      · rw [if_neg (by simp [h2]), if_neg (by simp [h2])]; norm_num
    · rw [if_neg (by simp [h]), if_neg (by simp [h])]; norm_num
  -- per-cell v readback: vloadT = bloomVTileMG
  have hvf : ∀ (jL : Fin BLOCK_N) (d : Fin BLOCK_DMODEL),
      vloadT.data (jL, d, PUnit.unit)
        = some (bloomVTileMG s0 V Req_to_tokens B_req_idx stride_req_b stride_req_s stride_vbs stride_vh stride_vd head_dim S bel
            ⟨c * BLOCK_N + jL.val, gBlock_idx_lt S BLOCK_N c hwin jL⟩ d.val) := by
    intro jL d
    rw [hvleq]
    show (if decide (c * BLOCK_N + jL.val < bel) && decide (d.val < head_dim) then
        (s.setReg "start_n" .nat [] (Tile.scalar (c * BLOCK_N))).readMemValue .real (Region.cast V)
          ((if decide (c * BLOCK_N + jL.val < bel) then
              (s.setReg "start_n" .nat [] (Tile.scalar (c * BLOCK_N))).readMemValue .nat (Region.cast Req_to_tokens)
                (stride_req_b * rqi + stride_req_s * (c * BLOCK_N + jL.val))
            else 0) * stride_vbs + (s0.pids 1 / 1) * stride_vh + d.val * stride_vd)
        else some (0.0 : ℝ)) = _
    simp only [bloomVTileMG, bloomVTileG, bloomKvLocG, curHead]
    by_cases h : c * BLOCK_N + jL.val < bel
    · by_cases h2 : d.val < head_dim
      · rw [if_pos (by simp only [decide_eq_true_eq, Bool.and_eq_true]; exact ⟨h, h2⟩),
          if_pos (And.intro h h2), if_pos (by simp only [decide_eq_true_eq]; exact h)]
        simp only [BlockState.setReg_readMem, BlockState.setReg_readMemValue,
          BlockState.readMemValue_real, Region.cast_id, hckvhd, hrqid, hrmemN, hrmemR,
          Nat.mul_comm stride_req_b (reqIdx s0 B_req_idx)]
      · rw [if_neg (by simp [h2]), if_neg (by simp [h2])]; norm_num
    · rw [if_neg (by simp [h]), if_neg (by simp [h])]; norm_num
  -- the score cell (natural) = (g a dd ⟨c*BN+jL⟩).1 · log 2
  have hscorelog : ∀ (a : Fin BLOCK_M) (jL : Fin BLOCK_N) (dd : Nat),
      bloomQkCell (D := BLOCK_DMODEL) sm_scale (c * BLOCK_N) plen (fun r : Fin BLOCK_M => s0.pids 2 * BLOCK_M + r.val)
          (fun a e => bloomQTileMG s0 Q B_Start_Loc B_Seqlen B_Prompt_Cache_Len stride_qbs stride_qh stride_qd head_dim BLOCK_M a e.val)
          (fun jK e => bloomKTileMG s0 K Req_to_tokens B_req_idx stride_req_b stride_req_s stride_kbs stride_kh stride_kd head_dim S bel
            ⟨c * BLOCK_N + jK.val, gBlock_idx_lt S BLOCK_N c hwin jK⟩ e.val)
          a jL
        = (g a dd ⟨c * BLOCK_N + jL.val, gBlock_idx_lt S BLOCK_N c hwin jL⟩).1 * Real.log 2 := by
    intro a jL dd
    simp only [hgd, bloomGG, bloomKVMG, bloomQkCell]
    by_cases h : c * BLOCK_N + jL.val ≤ s0.pids 2 * BLOCK_M + a.val + plen
    · rw [if_pos h, if_pos (by rw [hplend] at *; omega)]
      rw [div_mul_cancel₀ _ (by positivity : Real.log 2 ≠ 0)]
    · rw [if_neg h, if_neg (by rw [hplend] at *; omega)]
      rw [div_mul_cancel₀ _ (by positivity : Real.log 2 ≠ 0)]; norm_num
  -- qkT cell = some (score · log 2)
  have hqk : ∀ (a : Fin BLOCK_M) (jL : Fin BLOCK_N) (dd : Nat),
      qkT.data (a, jL, PUnit.unit) = some ((g a dd ⟨c * BLOCK_N + jL.val, gBlock_idx_lt S BLOCK_N c hwin jL⟩).1 * Real.log 2) := by
    intro a jL dd
    rw [hqkeq]
    rw [bloom_qkT_cell sm_scale (c * BLOCK_N) plen (fun r : Fin BLOCK_M => s0.pids 2 * BLOCK_M + r.val)
      qtile kloadT
      (fun aa ee => bloomQTileMG s0 Q B_Start_Loc B_Seqlen B_Prompt_Cache_Len stride_qbs stride_qh stride_qd head_dim BLOCK_M aa ee.val)
      (fun jK ee => bloomKTileMG s0 K Req_to_tokens B_req_idx stride_req_b stride_req_s stride_kbs stride_kh stride_kd head_dim S bel
        ⟨c * BLOCK_N + jK.val, gBlock_idx_lt S BLOCK_N c hwin jK⟩ ee.val)
      hqf hkf a jL]
    rw [hscorelog a jL dd]
  set st := gStateBot S (c * BLOCK_N) with hstdef
  set M' := fun (a : Fin BLOCK_M) (dd : Nat) => (gStateBot S ((c + 1) * BLOCK_N) (g a dd)).1 with hM'def
  have hsucc : ∀ (a : Fin BLOCK_M) (dd : Nat),
      gStateBot S ((c + 1) * BLOCK_N) (g a dd)
        = (M' a dd,
           (st (g a dd)).2.1 * (WithBot.realExp2 (WithBot.realSub (st (g a dd)).1 (M' a dd))).unbotD 0
             + Finset.univ.sum (fun jL : Fin BLOCK_N =>
                 pow2 ((g a dd ⟨c * BLOCK_N + jL.val, gBlock_idx_lt S BLOCK_N c hwin jL⟩).1 - (M' a dd).unbotD 0)),
           (st (g a dd)).2.2 * (WithBot.realExp2 (WithBot.realSub (st (g a dd)).1 (M' a dd))).unbotD 0
             + Finset.univ.sum (fun jL : Fin BLOCK_N =>
                 pow2 ((g a dd ⟨c * BLOCK_N + jL.val, gBlock_idx_lt S BLOCK_N c hwin jL⟩).1 - (M' a dd).unbotD 0)
                   * (g a dd ⟨c * BLOCK_N + jL.val, gBlock_idx_lt S BLOCK_N c hwin jL⟩).2)) := by
    intro a dd
    have he := gStateBot_succ_explicit S BLOCK_N c (g a dd) hwin
    simp only [] at he
    have hM'eq : M' a dd = (gStateBot S (c * BLOCK_N) (g a dd)).1 ⊔ Finset.univ.sup (fun jL : Fin BLOCK_N =>
        ((g a dd ⟨c * BLOCK_N + jL.val, gBlock_idx_lt S BLOCK_N c hwin jL⟩).1 : WithBot ℝ)) := by
      rw [hM'def]; simp only []; rw [he]
    rw [hM'eq]; exact he
  have hM'ne : ∀ (a : Fin BLOCK_M) (dd : Nat), M' a dd ≠ ⊥ := by
    intro a dd
    rw [hM'def]; simp only []
    rw [gStateBot_fst_eq_runningMax]
    exact gRunningMax_succ_ne_bot S BLOCK_N c (g a dd) hBN hwin
  have hMdd : ∀ (a : Fin BLOCK_M) (dd : Nat), M' a d0 = M' a dd := by
    intro a dd
    rw [hM'def]; simp only []
    exact (gStateBot_score_congr S ((c + 1) * BLOCK_N) _ _
      (fun j => by simp only [hgd, bloomGG, bloomKVMG])).1
  have hst1 : ∀ (a : Fin BLOCK_M) (dd : Nat), (st (g a d0)).1 = (st (g a dd)).1 := by
    intro a dd
    rw [hstdef]
    exact (gStateBot_score_congr S (c * BLOCK_N) _ _
      (fun j => by simp only [hgd, bloomGG, bloomKVMG])).1
  have hlog2pos : (0 : ℝ) < Real.log 2 := Real.log_pos (by norm_num)
  set L : ℝ := Real.log 2 with hLdef
  have hmapsup2 : ∀ x y : WithBot ℝ, (x.map (· * L)) ⊔ (y.map (· * L)) = (x ⊔ y).map (· * L) := by
    intro x y
    cases x with
    | bot => simp
    | coe xr => cases y with
      | bot => simp
      | coe yr =>
        simp only [WithBot.map_coe, ← WithBot.coe_sup, WithBot.coe_le_coe]
        rcases le_total xr yr with h | h
        · rw [sup_eq_right.mpr h, sup_eq_right.mpr (mul_le_mul_of_nonneg_right h hlog2pos.le)]
        · rw [sup_eq_left.mpr h, sup_eq_left.mpr (mul_le_mul_of_nonneg_right h hlog2pos.le)]
  have hmapFsup : ∀ (f : Fin BLOCK_N → WithBot ℝ),
      Finset.univ.sup (fun jL : Fin BLOCK_N => (f jL).map (· * L)) = (Finset.univ.sup f).map (· * L) := by
    intro f
    refine Finset.cons_induction ?_ ?_ (Finset.univ : Finset (Fin BLOCK_N))
    · simp
    · intro a s ha ih
      rw [Finset.sup_cons, Finset.sup_cons, ih, hmapsup2]
  have hexpmap : ∀ x y : WithBot ℝ,
      WithBot.realExp (WithBot.realSub (x.map (· * L)) (y.map (· * L)))
        = WithBot.realExp2 (WithBot.realSub x y) := by
    intro x y
    cases x with
    | bot => cases y <;> rfl
    | coe xr => cases y with
      | bot => rfl
      | coe yr =>
        show WithBot.realExp (some (xr * L - yr * L)) = WithBot.realExp2 (some (xr - yr))
        rw [WithBot.realExp2_some]
        show some (Real.exp (xr * L - yr * L)) = some (Real.exp ((xr - yr) * Real.log 2))
        rw [hLdef]; congr 1; congr 1; ring
  -- m_ij cell = (block base-2 sup) · L
  have hmijc : ∀ a : Fin BLOCK_M, rmaxT.data (a, PUnit.unit)
      = (Finset.univ.sup (fun jL : Fin BLOCK_N =>
          ((g a d0 ⟨c * BLOCK_N + jL.val, gBlock_idx_lt S BLOCK_N c hwin jL⟩).1 : WithBot ℝ))).map (· * L) := by
    intro a
    rw [ctxg_reduceMaxDrop_data_row hBN qkT rmaxT hrm a
      (fun jL => ((g a d0 ⟨c * BLOCK_N + jL.val, gBlock_idx_lt S BLOCK_N c hwin jL⟩).1 : WithBot ℝ).map (· * L))
      (fun jL => by rw [hqk a jL d0]; rfl)]
    exact hmapFsup _
  have hminewc : ∀ a : Fin BLOCK_M, miNewT.data (a, PUnit.unit) = (M' a d0).map (· * L) := by
    intro a
    rw [hminew]
    rw [ctxg_mij_max mtile rmaxT a ((st (g a d0)).1.map (· * L)) _
      (by rw [hmtile]) (hmijc a)]
    rw [hM'def]; simp only []
    rw [congrArg Prod.fst (gStateBot_succ_explicit S BLOCK_N c (g a d0) hwin)]
    simp only []
    rw [hmapsup2]
  have halc : ∀ (a : Fin BLOCK_M) (dd : Nat), alphaT.data (a, PUnit.unit)
      = some ((WithBot.realExp2 (WithBot.realSub (st (g a dd)).1 (M' a dd))).unbotD 0) := by
    intro a dd
    rw [haleq]
    show WithBot.realExp ((Tile.bop NumericDType.real.sub Broadcast.nil.consSame mtile miNewT).data (a, PUnit.unit)) = _
    have hinner : (Tile.bop NumericDType.real.sub Broadcast.nil.consSame mtile miNewT).data (a, PUnit.unit)
        = WithBot.realSub ((st (g a dd)).1.map (· * L)) ((M' a dd).map (· * L)) := by
      simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.sub]
      rw [hmtile]; simp only []
      rw [hminewc a, hMdd a dd, hst1 a dd]
    rw [hinner, hexpmap]
    exact (ctxg_realExp2_eq_some_unbotD _)
  set Bsup := fun (a : Fin BLOCK_M) (dd : Nat) => Finset.univ.sup (fun jL : Fin BLOCK_N =>
      ((g a dd ⟨c * BLOCK_N + jL.val, gBlock_idx_lt S BLOCK_N c hwin jL⟩).1 : WithBot ℝ)) with hBsupdef
  have hBsupne : ∀ (a : Fin BLOCK_M) (dd : Nat), Bsup a dd ≠ ⊥ := by
    intro a dd
    rw [hBsupdef]; simp only []
    intro hcon
    have h0 := Finset.le_sup (f := fun jL : Fin BLOCK_N =>
        ((g a dd ⟨c * BLOCK_N + jL.val, gBlock_idx_lt S BLOCK_N c hwin jL⟩).1 : WithBot ℝ))
      (Finset.mem_univ (⟨0, hBN⟩ : Fin BLOCK_N))
    rw [hcon] at h0; exact absurd (le_bot_iff.mp h0) (by simp)
  have hBsupdd : ∀ (a : Fin BLOCK_M) (dd : Nat), Bsup a d0 = Bsup a dd := by
    intro a dd
    rw [hBsupdef]; simp only []
    refine Finset.sup_congr rfl (fun jL _ => ?_)
    simp only [hgd, bloomGG, bloomKVMG]
  have hBsup0 : ∀ a : Fin BLOCK_M, Bsup a d0
      = Finset.univ.sup (fun jL : Fin BLOCK_N =>
          ((g a d0 ⟨c * BLOCK_N + jL.val, gBlock_idx_lt S BLOCK_N c hwin jL⟩).1 : WithBot ℝ)) := by
    intro a; rw [hBsupdef]
  have hrmaxc : ∀ (a : Fin BLOCK_M) (dd : Nat), rmaxT.data (a, PUnit.unit) = some ((Bsup a dd).unbotD 0 * L) := by
    intro a dd
    rw [hmijc a, ← hBsup0 a, hBsupdd a dd]
    rcases hB : Bsup a dd with _ | br
    · exact absurd hB (hBsupne a dd)
    · simp [WithBot.map]
  have hpc : ∀ (a : Fin BLOCK_M) (jL : Fin BLOCK_N) (dd : Nat), pexpT.data (a, jL, PUnit.unit)
      = some (pow2 ((g a dd ⟨c * BLOCK_N + jL.val, gBlock_idx_lt S BLOCK_N c hwin jL⟩).1 - (Bsup a dd).unbotD 0)) := by
    intro a jL dd
    rw [hpexpeq]
    show WithBot.realExp ((Tile.bop NumericDType.real.sub Broadcast.nil.consR.consSame qkT
        (Tile.expandDim ⟨1, by simp⟩ rmaxT)).data (a, jL, PUnit.unit)) = _
    rw [ctx_qk_sub_mij_cell qkT rmaxT a jL _ _ (hqk a jL dd) (hrmaxc a dd)]
    show WithBot.realExp (some _) = _
    simp only [WithBot.realExp, pow2]
    rw [show (g a dd ⟨c * BLOCK_N + jL.val, gBlock_idx_lt S BLOCK_N c hwin jL⟩).1 * L - (Bsup a dd).unbotD 0 * L
          = Real.log 2 * ((g a dd ⟨c * BLOCK_N + jL.val, gBlock_idx_lt S BLOCK_N c hwin jL⟩).1 - (Bsup a dd).unbotD 0)
        from by rw [hLdef]; ring]
  have hlijc : ∀ (a : Fin BLOCK_M) (dd : Nat), lijT.data (a, PUnit.unit)
      = some (Finset.univ.sum (fun jL : Fin BLOCK_N =>
          pow2 ((g a dd ⟨c * BLOCK_N + jL.val, gBlock_idx_lt S BLOCK_N c hwin jL⟩).1 - (Bsup a dd).unbotD 0))) := by
    intro a dd
    rw [hlijeq, Tile.reduceSumDrop_data]
    simp only [TileShape.insertAxisIndex]
    refine Eq.trans (Finset.sum_congr rfl (fun (jL : Fin BLOCK_N) _ => hpc a jL dd)) ?_
    exact ctxg_withBot_sum_some
      (fun jL : Fin BLOCK_N => pow2 ((g a dd ⟨c * BLOCK_N + jL.val, gBlock_idx_lt S BLOCK_N c hwin jL⟩).1 - (Bsup a dd).unbotD 0))
  have hbetav : ∀ (a : Fin BLOCK_M) (dd : Nat), betaT.data (a, PUnit.unit)
      = some ((WithBot.realExp2 (WithBot.realSub (Bsup a dd) (M' a dd))).unbotD 0) := by
    intro a dd
    rw [hbetaeq]
    show WithBot.realExp ((Tile.bop NumericDType.real.sub Broadcast.nil.consSame rmaxT miNewT).data (a, PUnit.unit)) = _
    have hinner : (Tile.bop NumericDType.real.sub Broadcast.nil.consSame rmaxT miNewT).data (a, PUnit.unit)
        = WithBot.realSub ((Bsup a dd).map (· * L)) ((M' a dd).map (· * L)) := by
      simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.sub]
      rw [hmijc a, hminewc a, ← hBsup0 a, hBsupdd a dd, hMdd a dd]
    rw [hinner, hexpmap]
    exact (ctxg_realExp2_eq_some_unbotD _)
  have hbetapow : ∀ (a : Fin BLOCK_M) (jL : Fin BLOCK_N) (dd : Nat),
      (WithBot.realExp2 (WithBot.realSub (Bsup a dd) (M' a dd))).unbotD 0
        * pow2 ((g a dd ⟨c * BLOCK_N + jL.val, gBlock_idx_lt S BLOCK_N c hwin jL⟩).1 - (Bsup a dd).unbotD 0)
      = pow2 ((g a dd ⟨c * BLOCK_N + jL.val, gBlock_idx_lt S BLOCK_N c hwin jL⟩).1 - (M' a dd).unbotD 0) := by
    intro a jL dd
    rcases hB : Bsup a dd with _ | br
    · exact absurd hB (hBsupne a dd)
    · rcases hM : M' a dd with _ | Mr
      · exact absurd hM (hM'ne a dd)
      · have hfac : (WithBot.realExp2 (WithBot.realSub (some br) (some Mr))).unbotD 0
            = Real.exp ((br - Mr) * Real.log 2) := rfl
        have hb' : (WithBot.unbotD 0 (some br) : ℝ) = br := rfl
        have hm' : (WithBot.unbotD 0 (some Mr) : ℝ) = Mr := rfl
        rw [hfac, hb', hm', pow2, pow2, ← Real.exp_add]
        congr 1; ring
  have hliNewv : ∀ (a : Fin BLOCK_M) (dd : Nat), liNewT.data (a, PUnit.unit)
      = some ((gStateBot S ((c + 1) * BLOCK_N) (g a dd)).2.1) := by
    intro a dd
    rw [hlineweq]
    simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex]
    rw [halc a dd, show ltile.data (a, PUnit.unit) = some (st (g a dd)).2.1 from by
        rw [hltile]; simp only []; rw [hstdef]
        exact congrArg some (gStateBot_score_congr S (c * BLOCK_N) (g a d0) (g a dd)
          (fun j => by simp only [hgd, bloomGG, bloomKVMG])).2,
      hbetav a dd, hlijc a dd]
    rw [hsucc a dd]; simp only []
    show WithBot.realAdd (WithBot.realMul (some _) (some _)) (WithBot.realMul (some _) (some _)) = _
    simp only [WithBot.realAdd, WithBot.realMul, Option.map₂, Option.bind, Option.map]
    refine congrArg some ?_
    rw [Finset.mul_sum]
    rw [Finset.sum_congr rfl (fun (jL : Fin BLOCK_N) _ => hbetapow a jL dd)]
    ring
  have hpscalev : ∀ (a : Fin BLOCK_M) (dd : Nat), pscaleT.data (a, PUnit.unit)
      = some ((WithBot.realExp2 (WithBot.realSub (Bsup a dd) (M' a dd))).unbotD 0
          / (gStateBot S ((c + 1) * BLOCK_N) (g a dd)).2.1) := by
    intro a dd
    rw [hpscaleeq]
    simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.div]
    rw [hbetav a dd, hliNewv a dd]; rfl
  have hp2c : ∀ (a : Fin BLOCK_M) (jL : Fin BLOCK_N) (dd : Nat), p2T.data (a, jL, PUnit.unit)
      = some (pow2 ((g a dd ⟨c * BLOCK_N + jL.val, gBlock_idx_lt S BLOCK_N c hwin jL⟩).1 - (M' a dd).unbotD 0)
          / (gStateBot S ((c + 1) * BLOCK_N) (g a dd)).2.1) := by
    intro a jL dd
    rw [hp2eq]
    simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, Tile.expandDim_data,
      TileShape.dropInsertedIndex, TileShape.insertAxisIndex, NumericDType.mul]
    rw [hpc a jL dd, hpscalev a dd]
    show WithBot.realMul (some _) (some _) = _
    simp only [WithBot.realMul, Option.map₂, Option.bind, Option.map]
    refine congrArg some ?_
    rw [← hbetapow a jL dd]; ring
  have hascale2c : ∀ (a : Fin BLOCK_M) (dd : Nat), accscale2T.data (a, PUnit.unit)
      = some (if c * BLOCK_N ≤ s0.pids 2 * BLOCK_M + a.val + plen then
          ((st (g a dd)).2.1 / (gStateBot S ((c + 1) * BLOCK_N) (g a dd)).2.1)
            * (WithBot.realExp2 (WithBot.realSub (st (g a dd)).1 (M' a dd))).unbotD 0
          else 1) := by
    intro a dd
    rw [hascale2eq]
    rw [Tile.select_data]
    have hcond : (⟨fun idx : TileIndex [BLOCK_M] => decide (c * BLOCK_N ≤ (fun r : Fin BLOCK_M => s0.pids 2 * BLOCK_M + r.val) idx.1 + plen)⟩ : Tile .bool [BLOCK_M]).data (a, PUnit.unit)
        = decide (c * BLOCK_N ≤ s0.pids 2 * BLOCK_M + a.val + plen) := rfl
    rw [hcond]
    by_cases hg : c * BLOCK_N ≤ s0.pids 2 * BLOCK_M + a.val + plen
    · rw [if_pos (by simp only [decide_eq_true_eq]; exact hg), if_pos hg]
      simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.mul, NumericDType.div]
      rw [show ltile.data (a, PUnit.unit) = some (st (g a dd)).2.1 from by
          rw [hltile]; simp only []; rw [hstdef]
          exact congrArg some (gStateBot_score_congr S (c * BLOCK_N) (g a d0) (g a dd)
            (fun j => by simp only [hgd, bloomGG, bloomKVMG])).2,
        hliNewv a dd, halc a dd]
      show WithBot.realMul (WithBot.realDiv (some _) (some _)) (some _) = _
      simp only [WithBot.realMul, WithBot.realDiv, Option.map₂, Option.bind, Option.map]
    · rw [if_neg (by simp only [decide_eq_true_eq]; exact hg), if_neg hg]; norm_num
  have hacc1c : ∀ idx : TileIndex [BLOCK_M, BLOCK_DMODEL], acc1T.data idx
      = some (gAccN S BLOCK_N (s0.pids 2 * BLOCK_M + idx.1.val + plen) (g idx.1 idx.2.1.val) c
          * (if c * BLOCK_N ≤ s0.pids 2 * BLOCK_M + idx.1.val + plen then
              ((st (g idx.1 idx.2.1.val)).2.1 / (gStateBot S ((c + 1) * BLOCK_N) (g idx.1 idx.2.1.val)).2.1)
                * (WithBot.realExp2 (WithBot.realSub (st (g idx.1 idx.2.1.val)).1 (M' idx.1 idx.2.1.val))).unbotD 0
              else 1)) := by
    intro idx
    rw [hacc1eq]
    simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, Tile.expandDim_data,
      TileShape.dropInsertedIndex, TileShape.insertAxisIndex, NumericDType.mul]
    rw [show acctile.data idx = some (gAccN S BLOCK_N (s0.pids 2 * BLOCK_M + idx.1.val + plen) (g idx.1 idx.2.1.val) c) from by
        rw [hacctile], hascale2c idx.1 idx.2.1.val]
    show WithBot.realMul (some _) (some _) = _
    simp only [WithBot.realMul, Option.map₂, Option.bind, Option.map]
  -- build invariant at (c+1)
  have hgapp : ∀ (a : Fin BLOCK_M) (d : Nat),
      bloomGG s0 Q K V B_Start_Loc B_Seqlen B_Prompt_Cache_Len Req_to_tokens B_req_idx sm_scale
        stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
        stride_vbs stride_vh stride_vd head_dim BLOCK_DMODEL BLOCK_M S bel a d = g a d :=
    fun a d => rfl
  simp only [bloomInvariantG, hgapp]
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hpidsF, BlockState.setReg_pids]; exact hpids
  · rw [hmemF]; exact hmem
  · exact hundefF
  · rw [hcbF]
  · rw [hckvhF]
  · rw [hchF]
  · rw [hplenF]
  · rw [hslF]
  · rw [hbelF]
  · rw [hcbsiF]
  · rw [hrqiF]
  · rw [homF]
  · rw [honF]
  · rw [hodF]
  · rw [hqF]
  · rw [hm_iF]; refine congrArg some ?_; ext a
    show miNewT.data a = (gStateBot S ((c + 1) * BLOCK_N) (g a.1 d0)).1.map (· * Real.log 2)
    rw [hminewc a.1, hM'def, ← hLdef]
  · rw [hl_iF]; refine congrArg some ?_; ext a
    show liNewT.data a = some (gStateBot S ((c + 1) * BLOCK_N) (g a.1 d0)).2.1
    rw [hliNewv a.1 d0]
  · rw [haccF]; refine congrArg some ?_; ext idx
    show (Tile.bop NumericDType.real.add Broadcast.nil.consSame.consSame acc1T (Tile.dot [] p2T vloadT)).data idx
        = some (gAccN S BLOCK_N (s0.pids 2 * BLOCK_M + idx.1.val + plen) (g idx.1 idx.2.1.val) (c + 1))
    simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex]
    have hdotc : (Tile.dot [] p2T vloadT).data (idx.1, idx.2.1, PUnit.unit)
        = some (Finset.univ.sum (fun jL : Fin BLOCK_N =>
            pow2 ((g idx.1 idx.2.1.val ⟨c * BLOCK_N + jL.val, gBlock_idx_lt S BLOCK_N c hwin jL⟩).1 - (M' idx.1 idx.2.1.val).unbotD 0)
              * (g idx.1 idx.2.1.val ⟨c * BLOCK_N + jL.val, gBlock_idx_lt S BLOCK_N c hwin jL⟩).2)
            / (gStateBot S ((c + 1) * BLOCK_N) (g idx.1 idx.2.1.val)).2.1) := by
      rw [ctxg_dot_row p2T vloadT idx.1 idx.2.1 _ _ (fun jL => hp2c idx.1 jL idx.2.1.val)
        (fun jL => by rw [hvf jL idx.2.1])]
      refine congrArg some ?_
      rw [Finset.sum_div]
      refine Finset.sum_congr rfl (fun jL _ => ?_)
      simp only [hgd, bloomGG, bloomKVMG]
      ring
    rw [hdotc, hacc1c idx]
    show WithBot.realAdd (some _) (some _) = _
    simp only [WithBot.realAdd, Option.map₂, Option.bind, Option.map]
    refine congrArg some ?_
    rw [gAccN_succ]
    simp only [hstdef, hM'def]
    have hincr : (gStateBot S ((c + 1) * BLOCK_N) (g idx.1 idx.2.1.val)).2.2
          - (gStateBot S (c * BLOCK_N) (g idx.1 idx.2.1.val)).2.2
              * (WithBot.realExp2 (WithBot.realSub (gStateBot S (c * BLOCK_N) (g idx.1 idx.2.1.val)).1
                  (gStateBot S ((c + 1) * BLOCK_N) (g idx.1 idx.2.1.val)).1)).unbotD 0
        = Finset.univ.sum (fun jL : Fin BLOCK_N =>
            pow2 ((g idx.1 idx.2.1.val ⟨c * BLOCK_N + jL.val, gBlock_idx_lt S BLOCK_N c hwin jL⟩).1
                - ((gStateBot S ((c + 1) * BLOCK_N) (g idx.1 idx.2.1.val)).1).unbotD 0)
              * (g idx.1 idx.2.1.val ⟨c * BLOCK_N + jL.val, gBlock_idx_lt S BLOCK_N c hwin jL⟩).2) := by
      conv_lhs => rw [hsucc idx.1 idx.2.1.val]
      simp only [hstdef, hM'def]; ring
    rw [hincr]
  · exact hwin


/-! ### General postLoop eval (masked store off the faithful general fold) -/

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **General `bloomPostLoopG_eval`.** The in-loop-normalized `acc` (no final divide)
is masked-stored, reading off the faithful general fold `contextAttnBloomExactFoldMG`.
Mirror of `bloomPostLoop_eval` over symbolic dims/strides/head_dim; output-offset
injectivity is a hypothesis. -/
theorem bloomPostLoopG_eval
    (Q K V Out B_Start_Loc B_Seqlen Req_to_tokens B_req_idx B_Prompt_Cache_Len : RegionName) (s0 : BlockState)
    (sm_scale : ℝ)
    (stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
      head_dim BLOCK_DMODEL BLOCK_N BLOCK_M : Nat) (hD : 0 < BLOCK_DMODEL) (hBN : 0 < BLOCK_N)
    (S bel : Nat) (c : Nat) (s : BlockState) (hSc : S = c * BLOCK_N)
    (hOInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => outOffset s0 B_Start_Loc stride_obs stride_oh stride_od BLOCK_M idx))
    (hinv : bloomInvariantG Q K V Out B_Start_Loc B_Seqlen Req_to_tokens B_req_idx B_Prompt_Cache_Len s0
        sm_scale stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
        stride_vbs stride_vh stride_vd head_dim BLOCK_DMODEL BLOCK_N BLOCK_M hD S bel c s) :
    ∃ sP, stepStmts (bloomPostLoopG Out stride_obs stride_oh stride_od head_dim BLOCK_DMODEL BLOCK_M) s = some sP
      ∧ ∀ idx : TileIndex [BLOCK_M, BLOCK_DMODEL],
          sP.readMem Out (outOffset s0 B_Start_Loc stride_obs stride_oh stride_od BLOCK_M idx)
            = if active s0 B_Seqlen B_Prompt_Cache_Len head_dim BLOCK_M idx then
                contextAttnBloomExactFoldMG s0 Q K V B_Start_Loc B_Seqlen B_Prompt_Cache_Len
                  Req_to_tokens B_req_idx sm_scale stride_qbs stride_qh stride_qd stride_req_b stride_req_s
                  stride_kbs stride_kh stride_kd stride_vbs stride_vh stride_vd head_dim BLOCK_DMODEL BLOCK_N BLOCK_M S bel idx
              else s0.readMem Out (outOffset s0 B_Start_Loc stride_obs stride_oh stride_od BLOCK_M idx) := by
  subst hSc
  set plen := promptLen s0 B_Prompt_Cache_Len with hplend
  set sl := seqLen s0 B_Seqlen B_Prompt_Cache_Len with hsld
  set g := fun (i : Fin BLOCK_M) (d : Nat) =>
    bloomGG s0 Q K V B_Start_Loc B_Seqlen B_Prompt_Cache_Len Req_to_tokens B_req_idx sm_scale
      stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd head_dim BLOCK_DMODEL BLOCK_M (c * BLOCK_N) bel i d with hgd
  simp only [bloomInvariantG] at hinv
  obtain ⟨hpids, hmem, hundef, hcb, hckvh, hch, hplen, hsl, hbel, hcbsi, hrqi, hom, hon, hod,
      hq, hmi, hli, hacc, hcle⟩ := hinv
  set accTile : Tile .real [BLOCK_M, BLOCK_DMODEL] := ⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
      some (gAccN (c * BLOCK_N) BLOCK_N (s0.pids 2 * BLOCK_M + idx.1.val + plen) (g idx.1 idx.2.1.val) c)⟩
    with haccTile
  have haccref0 : s.regs .real [BLOCK_M, BLOCK_DMODEL] "acc" = some accTile := by
    rw [hacc]
  unfold bloomPostLoopG
  set offoTile : Tile .nat [BLOCK_M, BLOCK_DMODEL] :=
    ⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => outOffset s0 B_Start_Loc stride_obs stride_oh stride_od BLOCK_M idx⟩ with hoffo
  -- stmt 0: off_o
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .nat Broadcast.nil.consL.consR
        (Op.add .nat Broadcast.scalarR
          (Op.mul .nat Broadcast.scalarR
            (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "cur_batch_in_all_start_index")
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m")))
            (Op.constNat stride_obs))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat stride_oh)))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_DMODEL] "offs_d"))
          (Op.constNat stride_od))) s = some offoTile from by
      simp only [evalOp_add, evalOp_mul, evalOp_constNat, evalOp_ref,
        ctx_evalOp_expandDim_one_nat, ctx_evalOp_expandDim_zero_nat,
        BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq,
        hcbsi, hch, hom, hod, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp only [hoffo, outOffset, startLoc, mIndex, dIndex,
        Tile.bop_data, Tile.scalar_data, Tile.vec_data,
        Tile.expandDim, Broadcast.leftIndex, Broadcast.rightIndex,
        NumericDType.add, NumericDType.mul]))]
  set s1 := s.setReg "off_o" .nat [BLOCK_M, BLOCK_DMODEL] offoTile with hs1
  -- stmt 1: out_ptrs = Out + off_o
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Out) (Op.ref .nat [BLOCK_M, BLOCK_DMODEL] "off_o")) s1
        = some (⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
            (Out, offoTile.data idx)⟩ : Tile .ptr [BLOCK_M, BLOCK_DMODEL]) from by
      simp only [evalOp, hs1, BlockState.setReg_same, Region.cast_id,
        Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      · simp only [Tile.ptrAdd, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex]
      · simp only [Tile.ptrAdd, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex,
          Nat.zero_add]))]
  set s2 := s1.setReg "out_ptrs" .ptr [BLOCK_M, BLOCK_DMODEL]
    (⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => (Out, offoTile.data idx)⟩ : Tile .ptr [BLOCK_M, BLOCK_DMODEL]) with hs2
  -- stmt 2: masked store of acc into Out (2D boolAnd mask)
  set P : TileIndex [BLOCK_M, BLOCK_DMODEL] → Prop :=
    fun idx => s0.pids 2 * BLOCK_M + idx.1.val < sl ∧ idx.2.1.val < head_dim with hP
  have hmaskev : evalOp (Op.boolAnd Broadcast.nil.consL.consR
        (Op.lt .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m"))
          (Op.ref .nat [] "cur_batch_seq_len"))
        (Op.lt .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_DMODEL] "offs_d"))
          (Op.constNat head_dim))) s2
      = some (⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          decide (s0.pids 2 * BLOCK_M + idx.1.val < sl) && decide (idx.2.1.val < head_dim)⟩ : Tile .bool [BLOCK_M, BLOCK_DMODEL]) := by
    exact bloomQMask_evalG s2 (fun r : Fin BLOCK_M => s0.pids 2 * BLOCK_M + r.val) sl head_dim
      (by rw [hs2, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            hs1, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hom)
      (by rw [hs2, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            hs1, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hod)
      (by rw [hs2, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            hs1, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hsl)
  have haccref : evalOp (Op.ref .real [BLOCK_M, BLOCK_DMODEL] "acc") s2 = some accTile := by
    rw [evalOp_ref, hs2, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      hs1, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact haccref0
  have hptrref : evalOp (Op.ref .ptr [BLOCK_M, BLOCK_DMODEL] "out_ptrs") s2
      = some (⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => (Out, offoTile.data idx)⟩ : Tile .ptr [BLOCK_M, BLOCK_DMODEL]) := by
    rw [evalOp_ref, hs2, BlockState.setReg_same]
  have hstore : stepStmt (Stmt.store .real [BLOCK_M, BLOCK_DMODEL]
      (MemAccess.ptr (Op.ref .ptr [BLOCK_M, BLOCK_DMODEL] "out_ptrs"))
      (Op.ref .real [BLOCK_M, BLOCK_DMODEL] "acc")
      (MaskOpt.mask
        (Op.boolAnd Broadcast.nil.consL.consR
          (Op.lt .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m"))
            (Op.ref .nat [] "cur_batch_seq_len"))
          (Op.lt .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_DMODEL] "offs_d"))
            (Op.constNat head_dim))))) s2
      = some ((TileShape.allIndices [BLOCK_M, BLOCK_DMODEL]).foldl
          (fun acc idx => if P idx then acc.writeMem Out (offoTile.data idx)
            ((accTile.data idx).unbotD 0) else acc) s2) := by
    unfold stepStmt
    rw [haccref]
    simp only [hmaskev, Option.bind_eq_bind, Option.bind_some, Option.map_some]
    rw [hptrref]
    simp only [Option.bind_eq_bind, Option.bind_some]
    refine congrArg some ?_
    refine List.foldl_ext _ _ s2 ?_
    intro acc idx _
    by_cases h1 : s0.pids 2 * BLOCK_M + idx.1.val < sl
    · by_cases h2 : idx.2.1.val < head_dim
      · rw [decide_eq_true_eq.mpr h1, decide_eq_true_eq.mpr h2, Bool.and_true]
        simp only [if_true, BlockState.writeMemTyped_real, FloatDType.real_storeValue]
        rw [if_pos (show P idx from ⟨h1, h2⟩)]
      · rw [decide_eq_false_iff_not.mpr h2, Bool.and_false]
        simp only [Bool.false_eq_true, if_false]
        rw [if_neg (fun hc => h2 hc.2)]
    · rw [decide_eq_false_iff_not.mpr h1, Bool.false_and]
      simp only [Bool.false_eq_true, if_false]
      rw [if_neg (fun hc => h1 hc.1)]
  rw [stepStmts.cons_some hstore, stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  intro idx
  rw [show outOffset s0 B_Start_Loc stride_obs stride_oh stride_od BLOCK_M idx = offoTile.data idx from rfl]
  rw [BlockState.scatter_readback_prop_masked_nd (region := Out) s2 (fun idx => offoTile.data idx)
    (fun idx => (accTile.data idx).unbotD 0) P hOInj idx]
  have hactive_iff : active s0 B_Seqlen B_Prompt_Cache_Len head_dim BLOCK_M idx ↔ P idx := by
    have he : mIndex s0 BLOCK_M idx.1 = s0.pids 2 * BLOCK_M + idx.1.val := rfl
    simp only [active, hP, he, hsld, dIndex]
  by_cases hac : P idx
  · rw [if_pos hac, if_pos (hactive_iff.mpr hac)]
    simp only [haccTile, contextAttnBloomExactFoldMG, WithBot.unbotD_coe]
    have hkvm : g idx.1 idx.2.1.val
        = bloomKVMG s0 Q K V B_Start_Loc B_Seqlen B_Prompt_Cache_Len
            Req_to_tokens B_req_idx sm_scale stride_qbs stride_qh stride_qd stride_req_b stride_req_s
            stride_kbs stride_kh stride_kd stride_vbs stride_vh stride_vd head_dim BLOCK_DMODEL BLOCK_M (c * BLOCK_N) bel idx.1 idx.2.1.val := by
      rw [hgd]; rfl
    rw [hkvm, Nat.mul_div_cancel c hBN, hplend]
    rfl
  · rw [if_neg hac, if_neg (fun hcon => hac (hactive_iff.mp hcon))]
    show (s2.readMem Out (offoTile.data idx)) = _
    have hreadeq : s2.readMem Out (offoTile.data idx) = s.readMem Out (offoTile.data idx) := by
      unfold BlockState.readMem; rw [hs2, hs1]; simp only [BlockState.setReg_mem]
    rw [hreadeq]
    unfold BlockState.readMem; rw [hmem]


/-! ### General whole-kernel exec + genuine output value -/

/-- General kernel-decoded streamed window `S = BLOCK_N·⌈(block_mask·block_end_loc)/BLOCK_N⌉`
(loop step `BLOCK_N`; `block_end_loc` uses query block size `BLOCK_M`). -/
def bloomFwdWindowG (s : BlockState) (B_Seqlen B_Prompt_Cache_Len : RegionName) (BLOCK_M BLOCK_N : Nat) : Nat :=
  let plen := promptLen s B_Prompt_Cache_Len
  let sl := seqLen s B_Seqlen B_Prompt_Cache_Len
  let bel := bloomFwdBel s B_Seqlen B_Prompt_Cache_Len BLOCK_M
  let bm := if BLOCK_M * s.pids 2 < sl then 1 else 0
  BLOCK_N * ((bm * bel + (BLOCK_N - 1)) / BLOCK_N)

/-- **General genuine closed-form `Out` value**: the block-causal-guarded boundary-masked
in-loop-normalized online-softmax fold `contextAttnBloomExactFoldMG` of the gathered
Q/K/V memory — a pure function of memory, not the kernel's executed readback. -/
noncomputable def bloomFwdGenuineOutValueG
    (s : BlockState) (Q K V B_Start_Loc B_Seqlen Req_to_tokens B_req_idx B_Prompt_Cache_Len : RegionName)
    (sm_scale : ℝ)
    (stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd head_dim BLOCK_DMODEL BLOCK_N BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : ℝ :=
  contextAttnBloomExactFoldMG s Q K V B_Start_Loc B_Seqlen B_Prompt_Cache_Len Req_to_tokens B_req_idx sm_scale
    stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
    stride_vbs stride_vh stride_vd head_dim BLOCK_DMODEL BLOCK_N BLOCK_M
    (bloomFwdWindowG s B_Seqlen B_Prompt_Cache_Len BLOCK_M BLOCK_N)
    (bloomFwdBel s B_Seqlen B_Prompt_Cache_Len BLOCK_M) idx

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **General whole-kernel exec assembly.** Steps the faithful BLOOM surface and reads
off the genuine general masked fold at every active `Out` lane. Mirror of `bloom_exec`
over symbolic dims/strides/head_dim (`kv_group_num = 1`); output-offset injectivity is
a hypothesis. -/
theorem bloom_exec_general
    (Q K V Out : RegionName)
    (B_Start_Loc B_Seqlen Req_to_tokens B_req_idx B_Prompt_Cache_Len : Region .nat) (s : BlockState)
    (sm_scale : ℝ)
    (stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
      head_dim BLOCK_DMODEL BLOCK_M BLOCK_N : Nat) (hD : 0 < BLOCK_DMODEL) (hBN : 0 < BLOCK_N)
    (hOInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => outOffset s B_Start_Loc stride_obs stride_oh stride_od BLOCK_M idx))
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ∃ sF, exec (context_attn_bloom_fwd_kernel_surface Q K V
        sm_scale B_Start_Loc B_Seqlen Out Req_to_tokens B_req_idx B_Prompt_Cache_Len
        stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd
        stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
        stride_req_b stride_req_s 1 head_dim BLOCK_M BLOCK_DMODEL BLOCK_N) s = some sF
      ∧ ∀ idx : TileIndex [BLOCK_M, BLOCK_DMODEL],
          sF.readMem Out (outOffset s B_Start_Loc stride_obs stride_oh stride_od BLOCK_M idx)
            = if active s B_Seqlen B_Prompt_Cache_Len head_dim BLOCK_M idx then
                bloomFwdGenuineOutValueG s Q K V B_Start_Loc B_Seqlen Req_to_tokens B_req_idx B_Prompt_Cache_Len
                  sm_scale stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
                  stride_vbs stride_vh stride_vd head_dim BLOCK_DMODEL BLOCK_N BLOCK_M idx
              else s.readMem Out (outOffset s B_Start_Loc stride_obs stride_oh stride_od BLOCK_M idx) := by
  rw [show exec (context_attn_bloom_fwd_kernel_surface Q K V
        sm_scale B_Start_Loc B_Seqlen Out Req_to_tokens B_req_idx B_Prompt_Cache_Len
        stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd
        stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
        stride_req_b stride_req_s 1 head_dim BLOCK_M BLOCK_DMODEL BLOCK_N) s
      = stepStmts (context_attn_bloom_fwd_kernel_surface Q K V
        sm_scale B_Start_Loc B_Seqlen Out Req_to_tokens B_req_idx B_Prompt_Cache_Len
        stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd
        stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
        stride_req_b stride_req_s 1 head_dim BLOCK_M BLOCK_DMODEL BLOCK_N).toAlgKernel.body s from rfl]
  rw [bloomBody_splitG Q K V Out B_Start_Loc B_Seqlen Req_to_tokens B_req_idx B_Prompt_Cache_Len sm_scale
    stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd
    stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
    stride_req_b stride_req_s 1 head_dim BLOCK_M BLOCK_DMODEL BLOCK_N]
  -- preLoop
  obtain ⟨s0, hpre, hpids, hmem, hundef0, hcb, hch, hstart_m, hckvh, hplen, hcbsi, hrqi0, hsl0,
      hon, hod, hom, hmi, hli, hacc, hbel0, hbm0, hq⟩ :=
    bloomPreLoopG_eval s Q B_Start_Loc B_Seqlen B_req_idx B_Prompt_Cache_Len
      stride_qbs stride_qh stride_qd head_dim 1 BLOCK_DMODEL BLOCK_M BLOCK_N hundef
  rw [stepStmts.append_some hpre]
  set plen := promptLen s0 B_Prompt_Cache_Len with hplend
  set sl := seqLen s0 B_Seqlen B_Prompt_Cache_Len with hsld
  set bel := bloomFwdBel s0 B_Seqlen B_Prompt_Cache_Len BLOCK_M with hbeld
  set bm := (if BLOCK_M * s0.pids 2 < sl then 1 else 0) with hbmd
  set S := BLOCK_N * ((bm * bel + (BLOCK_N - 1)) / BLOCK_N) with hSd
  have hmemv : ∀ (rg : RegionName) (i : Nat),
      s.readMemValue .nat rg i = s0.readMemValue .nat rg i := by
    intro rg i; simp only [BlockState.readMemValue, BlockState.readMemTyped, hmem]
  have hmemvr : ∀ (rg : RegionName) (i : Nat),
      s.readMemValue .real rg i = s0.readMemValue .real rg i := by
    intro rg i; simp only [BlockState.readMemValue, BlockState.readMemAs, hmem]
  have hbelrb : s0.regs .nat [] "block_end_loc" = some (Tile.scalar bel) := by
    rw [hbel0]
    refine congrArg (fun x => some (Tile.scalar x)) ?_
    simp only [hbeld, bloomFwdBel, hsld, hplend, seqLen, promptLen, Region.cast_cast, ← hpids, hmemv]
  have hbmrb : s0.regs .nat [] "block_mask" = some (Tile.scalar bm) := by
    rw [hbm0]
    refine congrArg (fun x => some (Tile.scalar x)) ?_
    simp only [hbmd, hsld, hplend, seqLen, promptLen, Region.cast_cast, ← hpids, hmemv]
  -- loop-entry invariant at block 0
  have hinv0 : bloomInvariantG Q K V Out B_Start_Loc B_Seqlen Req_to_tokens B_req_idx B_Prompt_Cache_Len s0
      sm_scale stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd head_dim BLOCK_DMODEL BLOCK_N BLOCK_M hD S bel 0 s0 := by
    refine ⟨rfl, rfl, hundef0, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · rw [hcb, hpids]
    · rw [hckvh, hpids]
    · rw [hch, hpids]
    · rw [hplen]; simp only [promptLen, Region.cast_cast, hmemv, ← hpids]
    · rw [hsl0]; simp only [seqLen, promptLen, Region.cast_cast, hmemv, ← hpids]
    · exact hbelrb
    · rw [hcbsi]; simp only [startLoc, Region.cast_cast, hmemv, ← hpids]
    · rw [hrqi0]; simp only [reqIdx, Region.cast_cast, hmemv, ← hpids]
    · rw [hom, hpids]
    · rw [hon]
    · rw [hod]
    · rw [hq]; refine congrArg some ?_; ext idx
      simp only [seqLen, promptLen, startLoc, Region.cast_id, hpids, hmemv, hmemvr]
    · rw [hmi]; refine congrArg some ?_; ext r
      simp only [Nat.zero_mul, gStateBot_zero, WithBot.map_bot]
    · rw [hli]; refine congrArg some ?_; ext r
      simp only [Nat.zero_mul, gStateBot_zero]
    · rw [hacc]; refine congrArg some ?_; ext idx
      simp only [gAccN_zero]
    · omega
  -- run the streaming loop
  obtain ⟨final, sL, hloop, hfin, c_final, hfinaleq, hinvL⟩ :=
    forRangeDyn_inv (idx := "start_n")
      (startOp := Op.constNat 0)
      (stopOp := Op.mul .nat Broadcast.nil (Op.ref .nat [] "block_mask") (Op.ref .nat [] "block_end_loc"))
      (stepOp := Op.constNat BLOCK_N)
      (P := fun i st => ∃ c, i = c * BLOCK_N ∧
        bloomInvariantG Q K V Out B_Start_Loc B_Seqlen Req_to_tokens B_req_idx B_Prompt_Cache_Len s0
          sm_scale stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
          stride_vbs stride_vh stride_vd head_dim BLOCK_DMODEL BLOCK_N BLOCK_M hD S bel c st)
      (s_init := s0)
      (by rw [evalOp_constNat])
      (show evalOp (Op.mul .nat Broadcast.nil (Op.ref .nat [] "block_mask")
            (Op.ref .nat [] "block_end_loc")) s0 = some (Tile.scalar (bm * bel)) from by
        rw [evalOp_mul, evalOp_ref, evalOp_ref, hbmrb, hbelrb]
        simp only [Option.bind_eq_bind, Option.bind_some]; rfl)
      (by rw [evalOp_constNat])
      (Nat.pos_iff_ne_zero.mp hBN)
      ⟨0, by ring, hinv0⟩
      (fun i st hi hP => by
        obtain ⟨c, hic, hinvc⟩ := hP
        have hwin : (c + 1) * BLOCK_N ≤ S := by
          have hstop : c * BLOCK_N < bm * bel := hic ▸ hi
          rw [hSd]
          have hge : bm * bel ≤ BLOCK_N * ((bm * bel + (BLOCK_N - 1)) / BLOCK_N) := by
            have hdm := Nat.div_add_mod (bm * bel + (BLOCK_N - 1)) BLOCK_N
            have h3 : (bm * bel + (BLOCK_N - 1)) % BLOCK_N < BLOCK_N := Nat.mod_lt _ hBN
            omega
          have hcq : c < (bm * bel + (BLOCK_N - 1)) / BLOCK_N := by
            by_contra hcon
            have hcon' : (bm * bel + (BLOCK_N - 1)) / BLOCK_N ≤ c := Nat.not_lt.mp hcon
            have h1 : BLOCK_N * ((bm * bel + (BLOCK_N - 1)) / BLOCK_N) ≤ BLOCK_N * c :=
              Nat.mul_le_mul_left _ hcon'
            have h2 : bm * bel ≤ BLOCK_N * c := le_trans hge h1
            rw [Nat.mul_comm BLOCK_N c] at h2
            omega
          calc (c + 1) * BLOCK_N = BLOCK_N * (c + 1) := by ring
            _ ≤ BLOCK_N * ((bm * bel + (BLOCK_N - 1)) / BLOCK_N) := Nat.mul_le_mul_left _ hcq
        obtain ⟨s', hs', hinv'⟩ :=
          bloom_attn_stepG Q K V Out B_Start_Loc B_Seqlen Req_to_tokens B_req_idx B_Prompt_Cache_Len s0
            sm_scale stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
            stride_vbs stride_vh stride_vd BLOCK_DMODEL BLOCK_N BLOCK_M head_dim hD hBN
            S bel c i st hwin hic hinvc
        exact ⟨s', hs', c + 1, by rw [hic]; ring, hinv'⟩)
  subst hfinaleq
  have hcle : c_final * BLOCK_N ≤ S := hinvL.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have hScfinal : S = c_final * BLOCK_N := by
    have hfinge : bm * bel ≤ c_final * BLOCK_N := hfin
    have hqle : (bm * bel + (BLOCK_N - 1)) / BLOCK_N ≤ c_final := by
      rw [Nat.div_le_iff_le_mul_add_pred hBN]
      have hcomm : c_final * BLOCK_N = BLOCK_N * c_final := Nat.mul_comm _ _
      omega
    have hSub : S ≤ c_final * BLOCK_N := by
      rw [hSd, Nat.mul_comm BLOCK_N]
      exact Nat.mul_le_mul_right _ hqle
    omega
  -- postLoop
  obtain ⟨sP, hpostStep, hO⟩ :=
    bloomPostLoopG_eval Q K V Out B_Start_Loc B_Seqlen Req_to_tokens B_req_idx B_Prompt_Cache_Len s0
      sm_scale stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
      head_dim BLOCK_DMODEL BLOCK_N BLOCK_M hD hBN S bel c_final sL hScfinal
      (by
        have : (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => outOffset s0 B_Start_Loc stride_obs stride_oh stride_od BLOCK_M idx)
            = (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => outOffset s B_Start_Loc stride_obs stride_oh stride_od BLOCK_M idx) := by
          funext idx
          simp only [outOffset, startLoc, mIndex, dIndex, hpids, hmem,
            BlockState.readMemValue, BlockState.readMemTyped]
        rw [this]; exact hOInj)
      hinvL
  refine ⟨sP, ?_, ?_⟩
  · rw [stepStmts.cons_some hloop]; exact hpostStep
  · intro idx
    have hOidx := hO idx
    have houtoff : outOffset s0 B_Start_Loc stride_obs stride_oh stride_od BLOCK_M idx
        = outOffset s B_Start_Loc stride_obs stride_oh stride_od BLOCK_M idx := by
      simp only [outOffset, startLoc, mIndex, dIndex, hpids, hmemv]
    have hact : active s0 B_Seqlen B_Prompt_Cache_Len head_dim BLOCK_M idx
        ↔ active s B_Seqlen B_Prompt_Cache_Len head_dim BLOCK_M idx := by
      simp only [active, mIndex, seqLen, promptLen, dIndex, hpids, hmemv]
    rw [houtoff] at hOidx
    rw [hOidx]
    by_cases hac : active s0 B_Seqlen B_Prompt_Cache_Len head_dim BLOCK_M idx
    · rw [if_pos hac, if_pos (hact.mp hac)]
      have hreadmem : ∀ (rg : RegionName) (o : Nat), s0.readMem rg o = s.readMem rg o := by
        intro rg o; unfold BlockState.readMem; rw [hmem]
      have hbeleq : bel = bloomFwdBel s B_Seqlen B_Prompt_Cache_Len BLOCK_M := by
        simp only [hbeld, bloomFwdBel, hsld, hplend, seqLen, promptLen, ← hpids, hmemv]
      have hSeq : S = bloomFwdWindowG s B_Seqlen B_Prompt_Cache_Len BLOCK_M BLOCK_N := by
        simp only [bloomFwdWindowG, bloomFwdBel, hSd, hbmd, hbeld, bloomFwdBel, hsld, hplend,
          seqLen, promptLen, ← hpids, hmemv]
      rw [bloomFwdGenuineOutValueG, ← hbeleq, ← hSeq]
      have hkvm : bloomKVMG s0 Q K V B_Start_Loc B_Seqlen B_Prompt_Cache_Len Req_to_tokens B_req_idx sm_scale
            stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
            stride_vbs stride_vh stride_vd head_dim BLOCK_DMODEL BLOCK_M S bel idx.1 idx.2.1.val
          = bloomKVMG s Q K V B_Start_Loc B_Seqlen B_Prompt_Cache_Len Req_to_tokens B_req_idx sm_scale
            stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
            stride_vbs stride_vh stride_vd head_dim BLOCK_DMODEL BLOCK_M S bel idx.1 idx.2.1.val := by
        funext j
        simp only [bloomKVMG, bloomQTileMG, bloomKTileMG, bloomVTileMG, bloomQTileG, bloomKTileG, bloomVTileG,
          bloomKvLocG, promptLen, seqLen, reqIdx, startLoc, curHead, hpids, hmemv, hreadmem]
      simp only [contextAttnBloomExactFoldMG, hkvm]
      have hplenq : promptLen s0 B_Prompt_Cache_Len = promptLen s B_Prompt_Cache_Len := by
        simp only [promptLen, hpids, hmemv]
      rw [hplenq, hpids]
    · rw [if_neg hac, if_neg (fun h => hac (hact.mpr h))]
      unfold BlockState.readMem; rw [hmem]


/-! ### General top theorems -/

/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/
/-- **General surface compute-correctness** for `context_attn_bloom.py` over symbolic
`BLOCK_M`/`BLOCK_N`/`BLOCK_DMODEL`/`head_dim`, the per-axis K/V/Q/O strides and the
`Req_to_tokens` gather strides (`kv_group_num = 1`). Every active observable `Out`
write holds the genuine boundary-masked natural-exp in-loop-normalized causal-softmax
closed form `bloomFwdGenuineOutValueG` of the BLOOM gathered Q/K/V memory — a pure
function of memory, NOT the kernel's executed readback. The `-1e8` causal sentinel is
kept exactly. Side conditions: `0 < BLOCK_DMODEL`, `0 < BLOCK_N`, output-offset
injectivity. -/
specification context_attn_bloom_surface_compute_correct_general
    (Q K V Out B_Start_Loc B_Seqlen Req_to_tokens B_req_idx B_Prompt_Cache_Len : RegionName)
    (sm_scale : ℝ)
    (stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
      head_dim BLOCK_DMODEL BLOCK_M BLOCK_N : Nat) (hD : 0 < BLOCK_DMODEL) (hBN : 0 < BLOCK_N)
    (s : BlockState)
    (hOInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => outOffset s B_Start_Loc stride_obs stride_oh stride_od BLOCK_M idx))
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := context_attn_bloom_fwd_kernel_surface Q K V sm_scale
        B_Start_Loc B_Seqlen Out Req_to_tokens B_req_idx B_Prompt_Cache_Len
        stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd
        stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
        stride_req_b stride_req_s 1 head_dim BLOCK_M BLOCK_DMODEL BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          active s B_Seqlen B_Prompt_Cache_Len head_dim BLOCK_M idx)
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          (Out, outOffset s B_Start_Loc stride_obs stride_oh stride_od BLOCK_M idx)))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        bloomFwdGenuineOutValueG s Q K V B_Start_Loc B_Seqlen Req_to_tokens B_req_idx B_Prompt_Cache_Len
          sm_scale stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
          stride_vbs stride_vh stride_vd head_dim BLOCK_DMODEL BLOCK_N BLOCK_M idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [context_attn_bloom_fwd_kernel_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  obtain ⟨sF, hexec, hO⟩ :=
    bloom_exec_general Q K V Out B_Start_Loc B_Seqlen Req_to_tokens B_req_idx B_Prompt_Cache_Len s
      sm_scale stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
      head_dim BLOCK_DMODEL BLOCK_M BLOCK_N hD hBN hOInj hundef
  rw [hExec] at hexec
  obtain rfl : sF = s' := (Option.some.inj hexec).symm
  have hb := hO idx
  simp only [ComputeCorrect.OutputReadable.read_real, Region.cast_cast, Region.cast_id]
    at hb hActive ⊢
  rw [if_pos hActive] at hb
  rw [hb]

end Correct_without_Rounding


/-! # ══════════ IO-face (`⊨[R]`) — the paged-KV gather skin ══════════ -/

section IOFace

open scoped VeriTile.Triton.StreamMetaGatherMasked3DKernelIO₃

/-! ## Slot table and IO signature

`context_attn_bloom` is the **sibling consumer** of
`StreamMetaGatherMasked3DKernelIO₃` (metadata + a per-step gathered index
channel + three streamed float inputs + launch-legality `pre`), alongside
`context_attn_llama`. Its grid is genuinely **3-D**
(`cur_batch = pid₀`, `cur_head = pid₁`, `start_m = pid₂`), so all four `.nat`
metadata slots are read at cell `cur_batch = pid₀` of their own regions — no
`pid / H` head decode, and `cur_head` enters the windows as the bare `pid₁`.

Slots, in the kernel's own load order: slot `0` = `B_Start_Loc[pid₀]`
(`cur_batch_in_all_start_index`), slot `1` = `b_prompt_cache_len[pid₀]`
(`prompt_cache_len`), slot `2` = `B_Seqlen[pid₀]` **raw**, slot `3` =
`B_req_idx[pid₀]` (`cur_batch_req_idx`, the page-table row this program
gathers through). The `B_Seqlen` slot is the raw load because the kernel loads
the raw total length and subtracts in-register
(`cur_batch_seq_len = tl.load(B_Seqlen + cur_batch) - prompt_cache_len`), so
the honest slot is the raw load and every window/mask carries the
ℕ-truncated `m 2 - m 1` verbatim.

The **gather channel** is the `Req_to_tokens` page table, identical in shape
to llama's: per step the kernel loads a `[BLOCK_N]` `.nat` tile of physical
token slots at `stride_req_b·m 3 + stride_req_s·(t·BN + jL)`, masked by the
same `block_end_loc` liveness the K/V loads carry, with `other=0` — the
skin's `gbuf`/`gty`/`Bg`/`gread`/`gmask`/`gother` fields. Its *values*
address the `K`/`V` loads, so `read2`/`read3` eat the gathered tile `G`.

**The `head_dim` delta.** Unlike llama, every BLOOM float load *and* the
terminal store carries an extra channel guard `offs_d < head_dim`
(`BLOCK_DMODEL` is the padded tile width, `head_dim` the true head width), so
`mask1`/`mask2`/`mask3`/`writeMask` are **conjunctions**. The gather mask is
not: `gmask` is the 1-D `[BLOCK_N]` liveness guard `t·BN + jL <
block_end_loc` with no channel conjunct (the page-table load is a `[BLOCK_N]`
vector, not a tile). -/

/-- Slot-region table of the four per-batch metadata slots, in the kernel's
own load order. A shared def, never an inline `match` in a window/spec
position. -/
def bloomIOMetaBuf (B_Start_Loc B_Seqlen B_req_idx b_prompt_cache_len : Region .nat) :
    Fin 4 → RegionName
  | ⟨0, _⟩ => B_Start_Loc.cast
  | ⟨1, _⟩ => b_prompt_cache_len.cast
  | ⟨2, _⟩ => B_Seqlen.cast
  | ⟨_ + 3, _⟩ => B_req_idx.cast

/-- The pid-free step budget `T = ⌈NT·BLOCK_M / BLOCK_N⌉`: at a `pre`-legal
launch the live trip count `block_mask·block_end_loc ≤ B_Seqlen[b] ≤
NT·BLOCK_M` never outruns `T·BLOCK_N` (`bloomFwdIOBel_le` below). -/
def bloomFwdIOT (NT BLOCK_M BLOCK_N : Nat) : Nat :=
  (NT * BLOCK_M + (BLOCK_N - 1)) / BLOCK_N

/-- `0 < T` for nonempty tiles and a nonempty grid (the static `Q` stream is
read at step `0`, so the skin forces this). -/
theorem bloomFwdIOT_pos (NT BLOCK_M BLOCK_N : Nat) (hBM : 0 < BLOCK_M)
    (hBN : 0 < BLOCK_N) (hNT : 0 < NT) : 0 < bloomFwdIOT NT BLOCK_M BLOCK_N := by
  have h1 : 0 < NT * BLOCK_M := Nat.mul_pos hNT hBM
  exact Nat.div_pos (by omega) hBN

/-- Ceil property: `NT·BLOCK_M ≤ BLOCK_N·T`. -/
theorem bloomFwdIOT_mul_le (NT BLOCK_M BLOCK_N : Nat) (hBN : 0 < BLOCK_N) :
    NT * BLOCK_M ≤ BLOCK_N * bloomFwdIOT NT BLOCK_M BLOCK_N := by
  rw [bloomFwdIOT]
  have hdm := Nat.div_add_mod (NT * BLOCK_M + (BLOCK_N - 1)) BLOCK_N
  have h3 : (NT * BLOCK_M + (BLOCK_N - 1)) % BLOCK_N < BLOCK_N := Nat.mod_lt _ hBN
  omega

/-- The kernel-decoded `block_end_loc = min((pid₂+1)·BLOCK_M + plen,
(rawsl − plen) + plen)`, transcribed VERBATIM on the slot values
(ℕ-truncated subtraction and all) — `bloomFwdBel` freed from its state anchor
(they are definitionally equal at the slot reads). -/
def bloomFwdIOBel (BLOCK_M pid₂ plen rawsl : Nat) : Nat :=
  let sl := rawsl - plen
  let a := (pid₂ + 1) * BLOCK_M + plen
  let b := sl + plen
  if a < b then a else b

/-- The kernel-decoded streamed window `S = BLOCK_N·⌈bm·bel / BLOCK_N⌉`
(`bloomFwdWindowG` freed from its state anchor). -/
def bloomFwdIOWindow (BLOCK_M BLOCK_N pid₂ plen rawsl : Nat) : Nat :=
  BLOCK_N * (((if BLOCK_M * pid₂ < rawsl - plen then 1 else 0)
      * bloomFwdIOBel BLOCK_M pid₂ plen rawsl + (BLOCK_N - 1)) / BLOCK_N)

/-- Under the raw-slot launch bound the live trip count `bm·bel` is at most
`NT·BLOCK_M`: a live block (`bm = 1`) forces `plen < rawsl`, so
`bel ≤ (rawsl − plen) + plen = rawsl`. This is what makes the `pre`-forced
budget `T` citable for every live step. -/
private theorem bloomFwdIOBel_le (BLOCK_M pid₂ plen rawsl NT : Nat)
    (hraw : rawsl ≤ NT * BLOCK_M) :
    (if BLOCK_M * pid₂ < rawsl - plen then 1 else 0)
        * bloomFwdIOBel BLOCK_M pid₂ plen rawsl ≤ NT * BLOCK_M := by
  by_cases hbm : BLOCK_M * pid₂ < rawsl - plen
  · rw [if_pos hbm, one_mul]
    simp only [bloomFwdIOBel]
    split <;> omega
  · rw [if_neg hbm, zero_mul]
    exact Nat.zero_le _

/-- **Gather-indexed streaming metadata IO signature** of
`context_attn_bloom` on the gather-widened three-stream fold skin (S1:
online-softmax fold + terminal masked store, **3-D** pid grid
`(cur_batch, cur_head, start_m)`), at the existing headline's
`kv_group_num = 1` pin (so `cur_kv_head = cur_head`) and fully **symbolic
per-axis strides** and symbolic `head_dim`.

The kernel's `forRangeDyn` trip count `block_mask·block_end_loc` grows with
`pid₂` × the loaded slots, so the walk has **no pid-free step bound**:
`T := ⌈NT·BLOCK_M / BLOCK_N⌉` for a new `Nat` parameter `NT` (the host grid's
**third** dimension `cdiv(max_input_len, BLOCK)`), and the skin's
launch-legality field is

`pre := pid₂ < NT ∧ B_Seqlen[pid₀] ≤ NT·BLOCK_M`

— exactly the port's documented **trusted boundary** (see the file
docstring's Scope section): the host launches
`grid = (batch, head, cdiv(max_input_len, BLOCK))`, so every real program has
`start_m < NT`, and every raw `B_Seqlen[b] ≤ max_input_len ≤ NT·BLOCK_M`. The
`⊨[R]` triple says nothing about launches outside this boundary. The gather
adds **nothing** to `pre`.

Windows transcribe the kernel's pointer arithmetic exactly, with the loaded
slot vector `m` in place of the in-state metadata reads (`m 0 =
cur_batch_in_all_start_index`, `m 1 = prompt_cache_len`, `m 2 = raw
B_Seqlen[pid₀]`, `m 3 = cur_batch_req_idx`; the register `cur_batch_seq_len`
is `m 2 - m 1`) and the gathered tile `G` in place of the in-state `kv_loc`
register:

* `gread` (`Req_to_tokens`, the page table — `Bg = BLOCK_N`, `gty = .nat`,
  `gother = 0` from the load's `other=0`): lane `jL` at step `t` reads
  `stride_req_b·m 3 + stride_req_s·(t·BN + jL)`; `gmask` is the K/V liveness
  `t·BN + jL < block_end_loc(pid₂, m)` — the **same** guard the data loads
  carry, so dead lanes are never dereferenced. **No `head_dim` conjunct**
  (the page-table tile is 1-D).
* `read1` (`Q`, the **static** stream — the window ignores `t`): lane
  `j = (i, e)` row-major over `[BLOCK_M, BLOCK_DMODEL]` reads
  `(m 0 + (pid₂·BM + i))·stride_qbs + pid₁·stride_qh + e·stride_qd`;
  `mask1` is the row guard `pid₂·BM + i < m 2 − m 1` **∧** the channel guard
  `e < head_dim`.
* `read2` (`K`, **gather-addressed**, `kv_loc[None,:]` so the gathered lane
  sits on axis 1): lane `j = (e, jL)` over `[BLOCK_DMODEL, BLOCK_N]` reads
  `G t jL·stride_kbs + pid₁·stride_kh + e·stride_kd`; `mask2` is
  `t·BN + jL < block_end_loc(pid₂, m)` ∧ `e < head_dim`.
* `read3` (`V`, mirror at `[BLOCK_N, BLOCK_DMODEL]`, `kv_loc[:,None]` so the
  gathered lane sits on axis 0): lane `j = (jL, e)` reads
  `G t jL·stride_vbs + pid₁·stride_vh + e·stride_vd`; `mask3` is
  `t·BN + jL < block_end_loc(pid₂, m)` ∧ `e < head_dim`.
* `write` (`Out`, the terminal store — BLOOM normalizes **in loop**, so there
  is no post-loop divide): lane `j = (i, e)` writes
  `(m 0 + (pid₂·BM + i))·stride_obs + pid₁·stride_oh + e·stride_od`;
  `writeMask` is the row guard ∧ the channel guard.

The gathered *values* are host page-table contents — a trusted input, like
the slot values — so their in-bounds-ness is an `hbr2`/`hbr3` hypothesis of
the triple (quantified over `G`), not a `pre` fact.

`outDType` is the `.real` default: the terminal `tl.store` is untyped, so
there is no quantization event. -/
def contextAttnBloomIO (Q K V Out : RegionName)
    (B_Start_Loc B_Seqlen Req_to_tokens B_req_idx b_prompt_cache_len : Region .nat) (sm_scale : ℝ)
    (stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
      head_dim BLOCK_DMODEL BLOCK_M BLOCK_N NT : Nat) :
    StreamMetaGatherMasked3DKernelIO₃ where
  kernel := context_attn_bloom_fwd_kernel_surface Q K V sm_scale B_Start_Loc B_Seqlen Out
    Req_to_tokens B_req_idx b_prompt_cache_len
    stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd
    stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
    stride_req_b stride_req_s 1 head_dim BLOCK_M BLOCK_DMODEL BLOCK_N
  inp1 := Q
  inp2 := K
  inp3 := V
  out := Out
  nMeta := 4
  sty := fun _ => ChanTy.nat
  mbuf := bloomIOMetaBuf B_Start_Loc B_Seqlen B_req_idx b_prompt_cache_len
  mwin := fun _ pid₀ _ _ => pid₀
  gbuf := Req_to_tokens
  gty := ChanTy.nat
  Bg := BLOCK_N
  gother := 0
  T := bloomFwdIOT NT BLOCK_M BLOCK_N
  B1 := BLOCK_M * BLOCK_DMODEL
  B2 := BLOCK_DMODEL * BLOCK_N
  B3 := BLOCK_N * BLOCK_DMODEL
  C := BLOCK_M * BLOCK_DMODEL
  pre := fun _ _ pid₂ m =>
    pid₂ < NT ∧ m (⟨2, by omega⟩ : Fin 4) ≤ NT * BLOCK_M
  gread := fun _ _ _ m t j =>
    stride_req_b * m (⟨3, by omega⟩ : Fin 4) + stride_req_s * (t.val * BLOCK_N + j.val)
  gmask := fun _ _ pid₂ m t j =>
    t.val * BLOCK_N + j.val
      < bloomFwdIOBel BLOCK_M pid₂ (m (⟨1, by omega⟩ : Fin 4)) (m (⟨2, by omega⟩ : Fin 4))
  read1 := fun _ pid₁ pid₂ m _ j =>
    (m (⟨0, by omega⟩ : Fin 4) + (pid₂ * BLOCK_M + j.val / BLOCK_DMODEL)) * stride_qbs
      + pid₁ * stride_qh + j.val % BLOCK_DMODEL * stride_qd
  read2 := fun _ pid₁ _ _ G t j =>
    G t (Lane2D.decode j).2.1 * stride_kbs
      + pid₁ * stride_kh + j.val / BLOCK_N * stride_kd
  read3 := fun _ pid₁ _ _ G t j =>
    G t (Lane2D.decode j).1 * stride_vbs
      + pid₁ * stride_vh + j.val % BLOCK_DMODEL * stride_vd
  write := fun _ pid₁ pid₂ m j =>
    (m (⟨0, by omega⟩ : Fin 4) + (pid₂ * BLOCK_M + j.val / BLOCK_DMODEL)) * stride_obs
      + pid₁ * stride_oh + j.val % BLOCK_DMODEL * stride_od
  mask1 := fun _ _ pid₂ m _ j =>
    pid₂ * BLOCK_M + j.val / BLOCK_DMODEL
        < m (⟨2, by omega⟩ : Fin 4) - m (⟨1, by omega⟩ : Fin 4)
      ∧ j.val % BLOCK_DMODEL < head_dim
  mask2 := fun _ _ pid₂ m t j =>
    t.val * BLOCK_N + j.val % BLOCK_N
        < bloomFwdIOBel BLOCK_M pid₂ (m (⟨1, by omega⟩ : Fin 4)) (m (⟨2, by omega⟩ : Fin 4))
      ∧ j.val / BLOCK_N < head_dim
  mask3 := fun _ _ pid₂ m t j =>
    t.val * BLOCK_N + j.val / BLOCK_DMODEL
        < bloomFwdIOBel BLOCK_M pid₂ (m (⟨1, by omega⟩ : Fin 4)) (m (⟨2, by omega⟩ : Fin 4))
      ∧ j.val % BLOCK_DMODEL < head_dim
  writeMask := fun _ _ pid₂ m j =>
    pid₂ * BLOCK_M + j.val / BLOCK_DMODEL
        < m (⟨2, by omega⟩ : Fin 4) - m (⟨1, by omega⟩ : Fin 4)
      ∧ j.val % BLOCK_DMODEL < head_dim

/-! ## Stream-indexed tiles and the streamed closed form -/

/-- The `Q` cell read off the (static) first stream, row/channel-guarded like
the kernel's masked `q` load (`offs_m < cur_batch_seq_len` on the slot values
`sl = rawsl − plen`, `offs_d < head_dim`); the window ignores `t`, so the
step-`0` slice carries the whole `[BLOCK_M, BLOCK_DMODEL]` tile. -/
noncomputable def bloomFwdIOqM (BLOCK_M BLOCK_DMODEL T head_dim : Nat) (hT : 0 < T)
    (pid₂ sl : Nat) (xs : Fin T → Fin (BLOCK_M * BLOCK_DMODEL) → ℝ)
    (i : Fin BLOCK_M) (e : Fin BLOCK_DMODEL) : ℝ :=
  if pid₂ * BLOCK_M + i.val < sl ∧ e.val < head_dim then
    xs ⟨0, hT⟩ (Lane2D.encode (i, e, PUnit.unit))
  else 0

/-- The `block_end_loc`/channel-masked `K` cell at global key `jg` off the
second stream: step `jg / BLOCK_N`, block column `jg % BLOCK_N`, stream lane
`(e, jg % BLOCK_N)`; `0` at or beyond `bel`, off-channel (`e ≥ head_dim`)
— the kernel's `other=0.0` — and beyond the `T`-step budget (unreachable at
any `pre`-legal launch). -/
noncomputable def bloomFwdIOkM (BLOCK_DMODEL BLOCK_N T head_dim : Nat) (hBN : 0 < BLOCK_N)
    (bel : Nat) (ys : Fin T → Fin (BLOCK_DMODEL * BLOCK_N) → ℝ)
    (jg : Nat) (e : Fin BLOCK_DMODEL) : ℝ :=
  if jg < bel ∧ e.val < head_dim then
    if h : jg / BLOCK_N < T then
      ys ⟨jg / BLOCK_N, h⟩
        (Lane2D.encode (e, ⟨jg % BLOCK_N, Nat.mod_lt _ hBN⟩, PUnit.unit))
    else 0
  else 0

/-- The `block_end_loc`/channel-masked `V` cell at global key `jg` off the
third stream, at stream lane `(jg % BLOCK_N, d)`. -/
noncomputable def bloomFwdIOvM (BLOCK_N BLOCK_DMODEL T head_dim : Nat) (hBN : 0 < BLOCK_N)
    (bel : Nat) (zs : Fin T → Fin (BLOCK_N * BLOCK_DMODEL) → ℝ)
    (jg : Nat) (d : Fin BLOCK_DMODEL) : ℝ :=
  if jg < bel ∧ d.val < head_dim then
    if h : jg / BLOCK_N < T then
      zs ⟨jg / BLOCK_N, h⟩
        (Lane2D.encode (⟨jg % BLOCK_N, Nat.mod_lt _ hBN⟩, d, PUnit.unit))
    else 0
  else 0

/-- **The streamed closed form**: `bloomFwdGenuineOutValueG` restated VERBATIM
over the three streamed tiles — the **block-causal-guarded normalized
accumulator** `gAccN` (BLOOM normalizes *in loop*: `p_scale = β/lᵢⁿᵉʷ`,
`acc_scale = tl.where(offs_m+plen ≥ start_n, (lᵢ/lᵢⁿᵉʷ)·α, 1.0)`, and there
is **no** post-loop divide) of the ⊥-seeded online-softmax fold over the
kernel's live streamed window `S = BLOCK_N·⌈bm·bel / BLOCK_N⌉` (a function of
`pid₂` and the slots), with the prompt-cache-offset causal boundary and the
**finite `-1e8` sentinel** on future lanes — masked keys genuinely carry
weight `exp(-1e8 - m)`, so this is NOT cleaned up into a pure causal softmax.

The per-key score is divided by `Real.log 2` exactly as the port's
`bloomKVMG` does, so the shared `pow2`/`gStateBot` machinery expresses the
kernel's **natural** `tl.exp` with `sm_scale = 1/√D`. Output lane
`j = (i, e)` row-major over `[BLOCK_M, BLOCK_DMODEL]`. -/
noncomputable def contextAttnBloomIOSpec
    (BLOCK_M BLOCK_DMODEL BLOCK_N head_dim NT : Nat)
    (hBN : 0 < BLOCK_N) (hT : 0 < bloomFwdIOT NT BLOCK_M BLOCK_N) (sm_scale : ℝ)
    (pid₂ plen rawsl : Nat)
    (xs : Fin (bloomFwdIOT NT BLOCK_M BLOCK_N) → Fin (BLOCK_M * BLOCK_DMODEL) → ℝ)
    (ys : Fin (bloomFwdIOT NT BLOCK_M BLOCK_N) → Fin (BLOCK_DMODEL * BLOCK_N) → ℝ)
    (zs : Fin (bloomFwdIOT NT BLOCK_M BLOCK_N) → Fin (BLOCK_N * BLOCK_DMODEL) → ℝ)
    (j : Fin (BLOCK_M * BLOCK_DMODEL)) : ℝ :=
  gAccN (bloomFwdIOWindow BLOCK_M BLOCK_N pid₂ plen rawsl) BLOCK_N
    (pid₂ * BLOCK_M + (Lane2D.decode j).1.val + plen)
    (fun jg =>
      ((if jg.val ≤ pid₂ * BLOCK_M + (Lane2D.decode j).1.val + plen then
          sm_scale * Finset.univ.sum (fun e : Fin BLOCK_DMODEL =>
            bloomFwdIOqM BLOCK_M BLOCK_DMODEL (bloomFwdIOT NT BLOCK_M BLOCK_N) head_dim hT
                pid₂ (rawsl - plen) xs (Lane2D.decode j).1 e
              * bloomFwdIOkM BLOCK_DMODEL BLOCK_N (bloomFwdIOT NT BLOCK_M BLOCK_N) head_dim hBN
                  (bloomFwdIOBel BLOCK_M pid₂ plen rawsl) ys jg.val e)
        else (0.0 - 10e7 : ℝ)) / Real.log 2,
       bloomFwdIOvM BLOCK_N BLOCK_DMODEL (bloomFwdIOT NT BLOCK_M BLOCK_N) head_dim hBN
         (bloomFwdIOBel BLOCK_M pid₂ plen rawsl) zs jg.val (Lane2D.decode j).2.1))
    (bloomFwdIOWindow BLOCK_M BLOCK_N pid₂ plen rawsl / BLOCK_N)

set_option maxHeartbeats 3200000 in
/-- **Stream-spec bridge**: at a `pre`-legal launch (`rawsl ≤ NT·BLOCK_M`),
the port's genuine closed form `bloomFwdGenuineOutValueG` *is* the streamed
closed form `contextAttnBloomIOSpec`, under the skin's input pins. Both sides
are the same `gAccN`-over-`gStateBot` fold on the same live window — no
window extension is needed (the finite `-1e8` sentinel means dead keys carry
weight, so the window is transcribed, not extended); the per-key data agree
pointwise because every `bel`-live key sits inside the `T`-step budget
(`bloomFwdIOBel_le` + the ceil property).

This is the **only** place the gathered tile `G` is seen: the kernel's gather
closed form `bloomKvLocG` at a `bel`-live global key `jg` is rewritten to
`G ⟨jg/BN⟩ ⟨jg%BN⟩` through the gather pin's active leg (`hgv` below), after
which the `K`/`V` data pins apply verbatim. Dead keys (and off-`head_dim`
channels) need no gather fact: both sides substitute the load's `other=0.0`
there. -/
private theorem bloomFwdIOSpec_eq_genuine
    (s : BlockState) (Q K V : RegionName)
    (B_Start_Loc B_Seqlen Req_to_tokens B_req_idx b_prompt_cache_len : Region .nat) (sm_scale : ℝ)
    (stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd
      head_dim BLOCK_DMODEL BLOCK_M BLOCK_N NT : Nat)
    (hBN : 0 < BLOCK_N) (hT : 0 < bloomFwdIOT NT BLOCK_M BLOCK_N)
    (m0 m1 m2 m3 : Nat)
    (hm0 : s.readMemValue .nat ↑B_Start_Loc (s.pids 0) = m0)
    (hm1 : s.readMemValue .nat ↑b_prompt_cache_len (s.pids 0) = m1)
    (hm2 : s.readMemValue .nat ↑B_Seqlen (s.pids 0) = m2)
    (hm3 : s.readMemValue .nat ↑B_req_idx (s.pids 0) = m3)
    (hm2NT : m2 ≤ NT * BLOCK_M)
    (G : Fin (bloomFwdIOT NT BLOCK_M BLOCK_N) → Fin BLOCK_N → Nat)
    (xs : Fin (bloomFwdIOT NT BLOCK_M BLOCK_N) → Fin (BLOCK_M * BLOCK_DMODEL) → ℝ)
    (ys : Fin (bloomFwdIOT NT BLOCK_M BLOCK_N) → Fin (BLOCK_DMODEL * BLOCK_N) → ℝ)
    (zs : Fin (bloomFwdIOT NT BLOCK_M BLOCK_N) → Fin (BLOCK_N * BLOCK_DMODEL) → ℝ)
    (hgv : ∀ (t : Fin (bloomFwdIOT NT BLOCK_M BLOCK_N)) (jL : Fin BLOCK_N),
      t.val * BLOCK_N + jL.val < bloomFwdIOBel BLOCK_M (s.pids 2) m1 m2 →
      s.readMemValue .nat ↑Req_to_tokens
          (stride_req_b * m3 + stride_req_s * (t.val * BLOCK_N + jL.val)) = G t jL)
    (hx : ∀ (t : Fin (bloomFwdIOT NT BLOCK_M BLOCK_N)) (l : Fin (BLOCK_M * BLOCK_DMODEL)),
      s.pids 2 * BLOCK_M + l.val / BLOCK_DMODEL < m2 - m1
        ∧ l.val % BLOCK_DMODEL < head_dim →
      s.readMem Q ((m0 + (s.pids 2 * BLOCK_M + l.val / BLOCK_DMODEL)) * stride_qbs
          + s.pids 1 * stride_qh + l.val % BLOCK_DMODEL * stride_qd) = xs t l)
    (hy : ∀ (t : Fin (bloomFwdIOT NT BLOCK_M BLOCK_N)) (l : Fin (BLOCK_DMODEL * BLOCK_N)),
      t.val * BLOCK_N + l.val % BLOCK_N < bloomFwdIOBel BLOCK_M (s.pids 2) m1 m2
        ∧ l.val / BLOCK_N < head_dim →
      s.readMem K (G t (Lane2D.decode l).2.1 * stride_kbs
          + s.pids 1 * stride_kh + l.val / BLOCK_N * stride_kd) = ys t l)
    (hz : ∀ (t : Fin (bloomFwdIOT NT BLOCK_M BLOCK_N)) (l : Fin (BLOCK_N * BLOCK_DMODEL)),
      t.val * BLOCK_N + l.val / BLOCK_DMODEL < bloomFwdIOBel BLOCK_M (s.pids 2) m1 m2
        ∧ l.val % BLOCK_DMODEL < head_dim →
      s.readMem V (G t (Lane2D.decode l).1 * stride_vbs
          + s.pids 1 * stride_vh + l.val % BLOCK_DMODEL * stride_vd) = zs t l)
    (j : Fin (BLOCK_M * BLOCK_DMODEL)) :
    bloomFwdGenuineOutValueG s Q K V ↑B_Start_Loc ↑B_Seqlen ↑Req_to_tokens ↑B_req_idx
        ↑b_prompt_cache_len sm_scale
        stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
        stride_vbs stride_vh stride_vd head_dim BLOCK_DMODEL BLOCK_N BLOCK_M (Lane2D.decode j)
      = contextAttnBloomIOSpec BLOCK_M BLOCK_DMODEL BLOCK_N head_dim NT hBN hT sm_scale
          (s.pids 2) m1 m2 xs ys zs j := by
  have hplen : promptLen s ↑b_prompt_cache_len = m1 := hm1
  have hsl : seqLen s ↑B_Seqlen ↑b_prompt_cache_len = m2 - m1 := by
    simp only [seqLen, promptLen, hm1, hm2]
  have hbel : bloomFwdBel s ↑B_Seqlen ↑b_prompt_cache_len BLOCK_M
      = bloomFwdIOBel BLOCK_M (s.pids 2) m1 m2 := by
    simp only [bloomFwdBel, bloomFwdIOBel, seqLen, promptLen, hm1, hm2]
  have hSwin : bloomFwdWindowG s ↑B_Seqlen ↑b_prompt_cache_len BLOCK_M BLOCK_N
      = bloomFwdIOWindow BLOCK_M BLOCK_N (s.pids 2) m1 m2 := by
    simp only [bloomFwdWindowG, bloomFwdIOWindow, bloomFwdBel, bloomFwdIOBel, seqLen, promptLen,
      hm1, hm2]
  have hSxle : bloomFwdIOWindow BLOCK_M BLOCK_N (s.pids 2) m1 m2
      ≤ bloomFwdIOT NT BLOCK_M BLOCK_N * BLOCK_N := by
    have hbb := bloomFwdIOBel_le BLOCK_M (s.pids 2) m1 m2 NT hm2NT
    have h1 : ((if BLOCK_M * s.pids 2 < m2 - m1 then 1 else 0)
          * bloomFwdIOBel BLOCK_M (s.pids 2) m1 m2 + (BLOCK_N - 1)) / BLOCK_N
        ≤ bloomFwdIOT NT BLOCK_M BLOCK_N := by
      rw [bloomFwdIOT]
      exact Nat.div_le_div_right (by omega)
    calc bloomFwdIOWindow BLOCK_M BLOCK_N (s.pids 2) m1 m2
        ≤ BLOCK_N * bloomFwdIOT NT BLOCK_M BLOCK_N := by
          rw [bloomFwdIOWindow]; exact Nat.mul_le_mul_left _ h1
      _ = bloomFwdIOT NT BLOCK_M BLOCK_N * BLOCK_N := Nat.mul_comm _ _
  rw [bloomFwdGenuineOutValueG, hSwin, hbel]
  simp only [contextAttnBloomExactFoldMG, contextAttnBloomIOSpec, hplen]
  have hg : bloomKVMG s Q K V ↑B_Start_Loc ↑B_Seqlen ↑b_prompt_cache_len ↑Req_to_tokens ↑B_req_idx
      sm_scale
      stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd head_dim BLOCK_DMODEL BLOCK_M
      (bloomFwdIOWindow BLOCK_M BLOCK_N (s.pids 2) m1 m2)
      (bloomFwdIOBel BLOCK_M (s.pids 2) m1 m2)
      (Lane2D.decode j).1 (Lane2D.decode j).2.1.val
      = fun jg =>
        ((if jg.val ≤ s.pids 2 * BLOCK_M + (Lane2D.decode j).1.val + m1 then
            sm_scale * Finset.univ.sum (fun e : Fin BLOCK_DMODEL =>
              bloomFwdIOqM BLOCK_M BLOCK_DMODEL (bloomFwdIOT NT BLOCK_M BLOCK_N) head_dim hT
                  (s.pids 2) (m2 - m1) xs (Lane2D.decode j).1 e
                * bloomFwdIOkM BLOCK_DMODEL BLOCK_N (bloomFwdIOT NT BLOCK_M BLOCK_N) head_dim hBN
                    (bloomFwdIOBel BLOCK_M (s.pids 2) m1 m2) ys jg.val e)
          else (0.0 - 10e7 : ℝ)) / Real.log 2,
         bloomFwdIOvM BLOCK_N BLOCK_DMODEL (bloomFwdIOT NT BLOCK_M BLOCK_N) head_dim hBN
           (bloomFwdIOBel BLOCK_M (s.pids 2) m1 m2) zs jg.val (Lane2D.decode j).2.1) := by
    funext jg
    have hjT : jg.val / BLOCK_N < bloomFwdIOT NT BLOCK_M BLOCK_N :=
      (Nat.div_lt_iff_lt_mul hBN).mpr (lt_of_lt_of_le jg.isLt hSxle)
    -- the gather pin, transported to the kernel's own closed form at key `jg`
    have hgkv : jg.val < bloomFwdIOBel BLOCK_M (s.pids 2) m1 m2 →
        bloomKvLocG s ↑Req_to_tokens ↑B_req_idx stride_req_b stride_req_s jg.val
          = G ⟨jg.val / BLOCK_N, hjT⟩ ⟨jg.val % BLOCK_N, Nat.mod_lt _ hBN⟩ := by
      intro hb
      have hsplit : jg.val / BLOCK_N * BLOCK_N + jg.val % BLOCK_N = jg.val :=
        Nat.div_add_mod' jg.val BLOCK_N
      have hpin := hgv ⟨jg.val / BLOCK_N, hjT⟩ ⟨jg.val % BLOCK_N, Nat.mod_lt _ hBN⟩
        (by show jg.val / BLOCK_N * BLOCK_N + jg.val % BLOCK_N
              < bloomFwdIOBel BLOCK_M (s.pids 2) m1 m2
            rw [hsplit]; exact hb)
      rw [hsplit] at hpin
      rw [bloomKvLocG]
      show s.readMemValue .nat ↑Req_to_tokens
          (s.readMemValue .nat ↑B_req_idx (s.pids 0) * stride_req_b
            + stride_req_s * jg.val) = _
      rw [hm3, Nat.mul_comm m3 stride_req_b]
      exact hpin
    refine Prod.ext ?_ ?_
    · -- the score component (divided by `log 2`, as the port's `bloomKVMG` does)
      show ((if jg.val ≤ s.pids 2 * BLOCK_M + (Lane2D.decode j).1.val
              + promptLen s ↑b_prompt_cache_len then
            sm_scale * Finset.univ.sum (fun e : Fin BLOCK_DMODEL =>
              bloomQTileMG s Q ↑B_Start_Loc ↑B_Seqlen ↑b_prompt_cache_len
                  stride_qbs stride_qh stride_qd head_dim BLOCK_M (Lane2D.decode j).1 e.val
                * bloomKTileMG s K ↑Req_to_tokens ↑B_req_idx stride_req_b stride_req_s
                    stride_kbs stride_kh stride_kd head_dim _
                    (bloomFwdIOBel BLOCK_M (s.pids 2) m1 m2) jg e.val)
          else (0.0 - 10e7 : ℝ)) / Real.log 2) = _
      rw [hplen]
      refine congrArg (· / Real.log 2) ?_
      by_cases hc : jg.val ≤ s.pids 2 * BLOCK_M + (Lane2D.decode j).1.val + m1
      · rw [if_pos hc, if_pos hc]
        refine congrArg (sm_scale * ·) (Finset.sum_congr rfl (fun e _ => ?_))
        refine congrArg₂ (· * ·) ?_ ?_
        · -- the Q cell
          rw [bloomQTileMG, bloomFwdIOqM, hsl]
          by_cases hrow : s.pids 2 * BLOCK_M + (Lane2D.decode j).1.val < m2 - m1
              ∧ e.val < head_dim
          · rw [if_pos hrow, if_pos hrow, bloomQTileG,
              ← hx ⟨0, hT⟩ (Lane2D.encode ((Lane2D.decode j).1, e, PUnit.unit))
                (by rw [Lane2D.encode_div, Lane2D.encode_mod]; exact hrow)]
            rw [Lane2D.encode_div, Lane2D.encode_mod]
            simp only [BlockState.readMem, startLoc, curHead, hm0]
          · rw [if_neg hrow, if_neg hrow]
        · -- the K cell (gather-addressed)
          rw [bloomKTileMG, bloomFwdIOkM]
          by_cases hb : jg.val < bloomFwdIOBel BLOCK_M (s.pids 2) m1 m2 ∧ e.val < head_dim
          · rw [if_pos hb, if_pos hb, dif_pos hjT, bloomKTileG,
              ← hy ⟨jg.val / BLOCK_N, hjT⟩
                (Lane2D.encode (e, ⟨jg.val % BLOCK_N, Nat.mod_lt _ hBN⟩, PUnit.unit))
                (by rw [Lane2D.encode_mod, Lane2D.encode_div]
                    refine ⟨?_, hb.2⟩
                    show jg.val / BLOCK_N * BLOCK_N + jg.val % BLOCK_N
                      < bloomFwdIOBel BLOCK_M (s.pids 2) m1 m2
                    rw [Nat.div_add_mod']; exact hb.1)]
            rw [Lane2D.decode_encode, Lane2D.encode_div, hgkv hb.1]
            show s.readMem K (G ⟨jg.val / BLOCK_N, hjT⟩ ⟨jg.val % BLOCK_N, Nat.mod_lt _ hBN⟩
                  * stride_kbs + curHead s * stride_kh + e.val * stride_kd)
              = s.readMem K (G ⟨jg.val / BLOCK_N, hjT⟩ ⟨jg.val % BLOCK_N, Nat.mod_lt _ hBN⟩
                  * stride_kbs + s.pids 1 * stride_kh + e.val * stride_kd)
            rfl
          · rw [if_neg hb, if_neg hb]
      · rw [if_neg hc, if_neg hc]
    · -- the value component (gather-addressed)
      show bloomVTileMG s V ↑Req_to_tokens ↑B_req_idx stride_req_b stride_req_s
          stride_vbs stride_vh stride_vd head_dim _
          (bloomFwdIOBel BLOCK_M (s.pids 2) m1 m2) jg (Lane2D.decode j).2.1.val = _
      rw [bloomVTileMG, bloomFwdIOvM]
      by_cases hb : jg.val < bloomFwdIOBel BLOCK_M (s.pids 2) m1 m2
          ∧ (Lane2D.decode j).2.1.val < head_dim
      · rw [if_pos hb, if_pos hb, dif_pos hjT, bloomVTileG,
          ← hz ⟨jg.val / BLOCK_N, hjT⟩
            (Lane2D.encode (⟨jg.val % BLOCK_N, Nat.mod_lt _ hBN⟩,
              (Lane2D.decode j).2.1, PUnit.unit))
            (by rw [Lane2D.encode_div, Lane2D.encode_mod]
                refine ⟨?_, hb.2⟩
                show jg.val / BLOCK_N * BLOCK_N + jg.val % BLOCK_N
                  < bloomFwdIOBel BLOCK_M (s.pids 2) m1 m2
                rw [Nat.div_add_mod']; exact hb.1)]
        rw [Lane2D.decode_encode, Lane2D.encode_mod, hgkv hb.1]
        show s.readMem V (G ⟨jg.val / BLOCK_N, hjT⟩ ⟨jg.val % BLOCK_N, Nat.mod_lt _ hBN⟩
              * stride_vbs + curHead s * stride_vh + (Lane2D.decode j).2.1.val * stride_vd)
          = s.readMem V (G ⟨jg.val / BLOCK_N, hjT⟩ ⟨jg.val % BLOCK_N, Nat.mod_lt _ hBN⟩
              * stride_vbs + s.pids 1 * stride_vh + (Lane2D.decode j).2.1.val * stride_vd)
        rfl
      · rw [if_neg hb, if_neg hb]
  rw [hg]

/-! ## Flat-bridge coverage and cast-freedom

The LightLLM BLOOM port is cast-free post-erasure (`p.to(v.dtype)` lowered to
a self-assign; all loads/stores at `.real`/`.nat`, plain pointers throughout —
no `ptrSub`, no block pointers, no atomics), so `stepStmtsR R` collapses
verbatim onto the exact stepper on every segment. The only float unary is the
natural `Op.exp` (`tl.exp`), which is exact `.real` arithmetic just like
llama's `Op.exp2`. -/

set_option maxHeartbeats 4000000 in
/-- The surface sits inside the flat-memory bridge's covered fragment. -/
theorem bloomFwdIO_flattenOk (Q K V Out : RegionName)
    (B_Start_Loc B_Seqlen Req_to_tokens B_req_idx b_prompt_cache_len : Region .nat) (sm_scale : ℝ)
    (stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
      head_dim BLOCK_DMODEL BLOCK_M BLOCK_N : Nat) :
    ((context_attn_bloom_fwd_kernel_surface Q K V sm_scale B_Start_Loc B_Seqlen Out
      Req_to_tokens B_req_idx b_prompt_cache_len
      stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
      stride_req_b stride_req_s 1 head_dim BLOCK_M BLOCK_DMODEL BLOCK_N).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  rw [bloomBody_splitG]
  simp [bloomPreLoopG, bloomLoopBodyG, bloomPostLoopG, StmtList.FlattenOk,
    Stmt.FlattenOk, Op.FlattenOk]
  simp [Op.FlattenOk.eq_def]

/-- Per-statement cast-free collapse lifts to statement lists (walks the
actual successor chain; a failing step collapses on both sides). Private
copy of the llama/nopad helper (bench files never import each other). -/
private theorem bloomFwdIO_stepStmtsR_castFree_of_stmts (R : RoundingModel) :
    ∀ (l : List Stmt), (∀ st ∈ l, ∀ u, stepStmtR R st u = stepStmt st u) →
      ∀ s, stepStmtsR R l s = stepStmts l s
  | [], _, s => by simp only [stepStmtsR, stepStmts]
  | st :: rest, h, s => by
      simp only [stepStmtsR, stepStmts, h st List.mem_cons_self s]
      cases stepStmt st s with
      | none => rfl
      | some s' =>
          exact bloomFwdIO_stepStmtsR_castFree_of_stmts R rest
            (fun st' h' u => h st' (List.mem_cons_of_mem _ h') u) s'

set_option maxHeartbeats 4000000 in
/-- Every preLoop statement is cast-free (`.nat` slot loads, register-only
`.nat` arithmetic, the `other=0.0`-masked `.real` `q` load with its two-clause
`boolAnd` mask, `full`/`where` seeds). -/
private theorem bloomFwdIO_preLoop_stmt_castFree (R : RoundingModel)
    (Q : RegionName) (B_Start_Loc B_Seqlen B_req_idx b_prompt_cache_len : Region .nat)
    (stride_qbs stride_qh stride_qd head_dim kv_group_num BLOCK_DMODEL BLOCK_M BLOCK_N : Nat) :
    ∀ st ∈ bloomPreLoopG Q B_Start_Loc B_Seqlen B_req_idx b_prompt_cache_len
        stride_qbs stride_qh stride_qd head_dim kv_group_num BLOCK_DMODEL BLOCK_M BLOCK_N,
      ∀ u, stepStmtR R st u = stepStmt st u := by
  intro st hst u
  simp only [bloomPreLoopG, List.mem_cons, List.not_mem_nil, or_false] at hst
  rcases hst with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl |
    rfl | rfl | rfl | rfl | rfl | rfl
  all_goals simp only [stepStmtR, stepStmt, evalOpR.eq_def, evalOp.eq_def]

/-- The preLoop is cast-free as a list. -/
private theorem bloomFwdIO_preLoop_castFree (R : RoundingModel)
    (Q : RegionName) (B_Start_Loc B_Seqlen B_req_idx b_prompt_cache_len : Region .nat)
    (stride_qbs stride_qh stride_qd head_dim kv_group_num BLOCK_DMODEL BLOCK_M BLOCK_N : Nat)
    (t : BlockState) :
    stepStmtsR R (bloomPreLoopG Q B_Start_Loc B_Seqlen B_req_idx b_prompt_cache_len
        stride_qbs stride_qh stride_qd head_dim kv_group_num BLOCK_DMODEL BLOCK_M BLOCK_N) t
      = stepStmts (bloomPreLoopG Q B_Start_Loc B_Seqlen B_req_idx b_prompt_cache_len
          stride_qbs stride_qh stride_qd head_dim kv_group_num BLOCK_DMODEL BLOCK_M BLOCK_N) t :=
  bloomFwdIO_stepStmtsR_castFree_of_stmts R _
    (bloomFwdIO_preLoop_stmt_castFree R Q B_Start_Loc B_Seqlen B_req_idx b_prompt_cache_len
      stride_qbs stride_qh stride_qd head_dim kv_group_num BLOCK_DMODEL BLOCK_M BLOCK_N) t

set_option maxHeartbeats 4000000 in
/-- Every loop-body statement is cast-free (the erased `p.to(v.dtype)` is a
self-assign; the `kv_loc` gather is a `.nat` load with `other=0`, the `k`/`v`
loads carry `other=0.0` at `.real`; the natural `exp` and the in-loop
normalization are exact `.real` arithmetic). -/
private theorem bloomFwdIO_body_stmt_castFree (R : RoundingModel)
    (Q K V Req_to_tokens B_req_idx : RegionName) (sm_scale : ℝ)
    (stride_req_b stride_req_s stride_kbs stride_kh stride_kd stride_vbs stride_vh stride_vd
      head_dim BLOCK_DMODEL BLOCK_M BLOCK_N : Nat) :
    ∀ st ∈ bloomLoopBodyG Q K V Req_to_tokens B_req_idx sm_scale
        stride_req_b stride_req_s stride_kbs stride_kh stride_kd stride_vbs stride_vh stride_vd
        head_dim BLOCK_DMODEL BLOCK_M BLOCK_N,
      ∀ u, stepStmtR R st u = stepStmt st u := by
  intro st hst u
  simp only [bloomLoopBodyG, List.mem_cons, List.not_mem_nil, or_false] at hst
  rcases hst with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl |
    rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
  all_goals simp only [stepStmtR, stepStmt, evalOpR.eq_def, evalOp.eq_def]

/-- The loop body is cast-free as a list. -/
private theorem bloomFwdIO_body_castFree (R : RoundingModel)
    (Q K V Req_to_tokens B_req_idx : RegionName) (sm_scale : ℝ)
    (stride_req_b stride_req_s stride_kbs stride_kh stride_kd stride_vbs stride_vh stride_vd
      head_dim BLOCK_DMODEL BLOCK_M BLOCK_N : Nat) (t : BlockState) :
    stepStmtsR R (bloomLoopBodyG Q K V Req_to_tokens B_req_idx sm_scale
        stride_req_b stride_req_s stride_kbs stride_kh stride_kd stride_vbs stride_vh stride_vd
        head_dim BLOCK_DMODEL BLOCK_M BLOCK_N) t
      = stepStmts (bloomLoopBodyG Q K V Req_to_tokens B_req_idx sm_scale
          stride_req_b stride_req_s stride_kbs stride_kh stride_kd stride_vbs stride_vh stride_vd
          head_dim BLOCK_DMODEL BLOCK_M BLOCK_N) t :=
  bloomFwdIO_stepStmtsR_castFree_of_stmts R _
    (bloomFwdIO_body_stmt_castFree R Q K V Req_to_tokens B_req_idx sm_scale
      stride_req_b stride_req_s stride_kbs stride_kh stride_kd stride_vbs stride_vh stride_vd
      head_dim BLOCK_DMODEL BLOCK_M BLOCK_N) t

set_option maxHeartbeats 4000000 in
/-- Every postLoop statement is cast-free (`writeMemTypedR R .real` *is*
`writeMemTyped .real`); there are only three — BLOOM has **no** post-loop
`acc /= l_i`. -/
private theorem bloomFwdIO_postLoop_stmt_castFree (R : RoundingModel)
    (Out : RegionName) (stride_obs stride_oh stride_od head_dim BLOCK_DMODEL BLOCK_M : Nat) :
    ∀ st ∈ bloomPostLoopG Out stride_obs stride_oh stride_od head_dim BLOCK_DMODEL BLOCK_M,
      ∀ u, stepStmtR R st u = stepStmt st u := by
  intro st hst u
  simp only [bloomPostLoopG, List.mem_cons, List.not_mem_nil, or_false] at hst
  rcases hst with rfl | rfl | rfl
  all_goals
    simp only [stepStmtR, stepStmt, evalOpR.eq_def, evalOp.eq_def,
      BlockState.writeMemTypedR]

/-- The postLoop is cast-free as a list. -/
private theorem bloomFwdIO_postLoop_castFree (R : RoundingModel)
    (Out : RegionName) (stride_obs stride_oh stride_od head_dim BLOCK_DMODEL BLOCK_M : Nat)
    (t : BlockState) :
    stepStmtsR R (bloomPostLoopG Out stride_obs stride_oh stride_od head_dim
        BLOCK_DMODEL BLOCK_M) t
      = stepStmts (bloomPostLoopG Out stride_obs stride_oh stride_od head_dim
          BLOCK_DMODEL BLOCK_M) t :=
  bloomFwdIO_stepStmtsR_castFree_of_stmts R _
    (bloomFwdIO_postLoop_stmt_castFree R Out stride_obs stride_oh stride_od head_dim
      BLOCK_DMODEL BLOCK_M) t

/-- `evalOpR` of a `constNat` (R-independent). -/
private theorem bloomFwdIO_evalOpR_constNat (R : RoundingModel) (n : Nat) (u : BlockState) :
    evalOpR R (Op.constNat n) u = some (Tile.scalar n) := by
  simp [evalOpR]

/-- `evalOpR` of the `forRangeDyn` stop expression
`block_mask * block_end_loc` is the exact evaluation (register-only `.nat`
arithmetic — R-independent). -/
private theorem bloomFwdIO_stopOpR_castFree (R : RoundingModel) (u : BlockState) :
    evalOpR R (Op.mul .nat Broadcast.nil (Op.ref .nat [] "block_mask")
        (Op.ref .nat [] "block_end_loc")) u
      = evalOp (Op.mul .nat Broadcast.nil (Op.ref .nat [] "block_mask")
          (Op.ref .nat [] "block_end_loc")) u := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

set_option maxHeartbeats 1600000 in
/-- The streaming `forRangeDyn` statement is cast-free per-state: its bound
expressions are register-only `.nat` arithmetic and its body is cast-free,
so `stepStmtR R` on the whole loop *is* `stepStmt`. -/
private theorem bloomFwdIO_dyn_castFree (R : RoundingModel)
    (Q K V Req_to_tokens B_req_idx : RegionName) (sm_scale : ℝ)
    (stride_req_b stride_req_s stride_kbs stride_kh stride_kd stride_vbs stride_vh stride_vd
      head_dim BLOCK_DMODEL BLOCK_M BLOCK_N : Nat) :
    ∀ u, stepStmtR R (Stmt.forRangeDyn "start_n" (Op.constNat 0)
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "block_mask")
          (Op.ref .nat [] "block_end_loc"))
        (Op.constNat BLOCK_N)
        (bloomLoopBodyG Q K V Req_to_tokens B_req_idx sm_scale
          stride_req_b stride_req_s stride_kbs stride_kh stride_kd stride_vbs stride_vh stride_vd
          head_dim BLOCK_DMODEL BLOCK_M BLOCK_N)) u
      = stepStmt (Stmt.forRangeDyn "start_n" (Op.constNat 0)
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "block_mask")
            (Op.ref .nat [] "block_end_loc"))
          (Op.constNat BLOCK_N)
          (bloomLoopBodyG Q K V Req_to_tokens B_req_idx sm_scale
            stride_req_b stride_req_s stride_kbs stride_kh stride_kd stride_vbs stride_vh stride_vd
            head_dim BLOCK_DMODEL BLOCK_M BLOCK_N)) u := by
  intro u
  rw [stepForRangeAux.forRangeDyn_unfold]
  simp only [stepStmtR, bloomFwdIO_evalOpR_constNat, bloomFwdIO_stopOpR_castFree,
    evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
  cases hstop : evalOp (Op.mul .nat Broadcast.nil (Op.ref .nat [] "block_mask")
      (Op.ref .nat [] "block_end_loc")) u with
  | none => rfl
  | some t =>
      simp only [Option.bind_some]
      exact stepForRangeAuxR_castFree R _
        (bloomFwdIO_body_castFree R Q K V Req_to_tokens B_req_idx sm_scale
          stride_req_b stride_req_s stride_kbs stride_kh stride_kd stride_vbs stride_vh stride_vd
          head_dim BLOCK_DMODEL BLOCK_M BLOCK_N)
        "start_n" _ _ _ u

/-- The whole lowered body is cast-free, statement by statement: `execR R`
on the surface *is* the exact `stepStmts` run. -/
private theorem bloomFwdIO_execR_collapse (R : RoundingModel)
    (Q K V Out : RegionName)
    (B_Start_Loc B_Seqlen Req_to_tokens B_req_idx b_prompt_cache_len : Region .nat) (sm_scale : ℝ)
    (stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
      head_dim BLOCK_DMODEL BLOCK_M BLOCK_N : Nat) (s : BlockState) :
    execR R ((context_attn_bloom_fwd_kernel_surface Q K V sm_scale B_Start_Loc B_Seqlen Out
        Req_to_tokens B_req_idx b_prompt_cache_len
        stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd
        stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
        stride_req_b stride_req_s 1 head_dim BLOCK_M BLOCK_DMODEL BLOCK_N).toAlgKernel) s
      = stepStmts ((context_attn_bloom_fwd_kernel_surface Q K V sm_scale B_Start_Loc B_Seqlen Out
          Req_to_tokens B_req_idx b_prompt_cache_len
          stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd
          stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
          stride_req_b stride_req_s 1 head_dim BLOCK_M BLOCK_DMODEL
          BLOCK_N).toAlgKernel.body) s := by
  unfold execR
  rw [bloomBody_splitG]
  refine bloomFwdIO_stepStmtsR_castFree_of_stmts R _ ?_ s
  intro st hst
  rcases List.mem_append.mp hst with hpre | hrest
  · exact bloomFwdIO_preLoop_stmt_castFree R Q B_Start_Loc B_Seqlen B_req_idx b_prompt_cache_len
      stride_qbs stride_qh stride_qd head_dim 1 BLOCK_DMODEL BLOCK_M BLOCK_N st hpre
  · rcases List.mem_cons.mp hrest with rfl | hpost
    · exact bloomFwdIO_dyn_castFree R Q K V Req_to_tokens B_req_idx sm_scale
        stride_req_b stride_req_s stride_kbs stride_kh stride_kd stride_vbs stride_vh stride_vd
        head_dim BLOCK_DMODEL BLOCK_M BLOCK_N
    · exact bloomFwdIO_postLoop_stmt_castFree R Out stride_obs stride_oh stride_od head_dim
        BLOCK_DMODEL BLOCK_M st hpost

/-! ## The weak safety stack (`hts`)

The skin's `hts` obligation quantifies over **arbitrary** launch states (no
clean-`undef` pin), so the exact stack's `bloomInvariantG` (whose third
conjunct pins `undef`) is unavailable there. Since every load in this kernel
is masked with an explicit `other` (`0.0` for `q`/`k`/`v`, `0` for the
`kv_loc` gather) or an unmasked `.nat` scalar (never `other=None`), the whole
walk is `undef`-independent: the weak stack re-runs the register chain with
exact pins for the address-bearing registers and **existential** pins for the
value registers (`q`/`m_i`/`l_i`/`acc`) — the llama/nopad pattern. The `pre`
hypothesis (`raw B_Seqlen slot ≤ NT·BLOCK_M`) is what makes the per-step
`Fin T` window bounds citable at all. -/

/-- Combined walk cons: safety of the head at the current state, the R-step
it actually takes, and the pair (safety, run) of the tail from the successor
give the pair for the whole list. Private copy of the llama/nopad
combinator. -/
private theorem bloomFwdIO_walkCons {R : RoundingModel} {bounds : RegionBounds}
    {P : BlockState → Prop} {st : Stmt} {rest : List Stmt} {s s' : BlockState}
    (h1 : Stmt.TraceSafeR R bounds st s)
    (hstep : stepStmtR R st s = some s')
    (h2 : Stmt.TraceSafeListR R bounds rest s'
      ∧ ∃ sF, stepStmtsR R rest s' = some sF ∧ P sF) :
    Stmt.TraceSafeListR R bounds (st :: rest) s
      ∧ ∃ sF, stepStmtsR R (st :: rest) s = some sF ∧ P sF :=
  ⟨Stmt.TraceSafeListR.cons_intro h1 (fun u hu => by
      rw [hstep] at hu
      exact (Option.some.inj hu) ▸ h2.1),
    by rw [stepStmtsR_cons_some hstep]; exact h2.2⟩

/-- R-step of an assign whose op is cast-free: the two collapse into one
walk-ready equation. -/
private theorem bloomFwdIO_stepR_of_assign {R : RoundingModel} {dt : TileDType}
    {sh : TileShape} {nm : RegName} {e : Op dt sh} {s : BlockState} {v : Tile dt sh}
    (hcf : evalOpR R e s = evalOp e s) (h : evalOp e s = some v) :
    stepStmtR R (.assign dt sh nm e) s = some (s.setReg nm dt sh v) :=
  stepStmtR_assign_eq_some (hcf.trans h)

/-- Masked `.real` region load, read back on a **memory-equal anchor state**.

The body writes registers only, so the in-loop state `u` (a deep `setReg`
tower) has `u.mem = base.mem`; this lemma discharges that bridge by an
explicit `rw` on `hmem` instead of leaving it to `whnf`. Without it the
`k`/`v` load steps would have to prove `u.readMemValue = base.readMemValue`
by *defeq* through the whole tower, which is where the elaborator blows up at
the 21-deep `v` load. Everything else is `ctx_evalOp_load_region_maskOther`
verbatim. -/
private theorem bloomFwdIO_maskedLoad_eval {sh : TileShape}
    (Rg : Region .real) (offsets : Op .nat sh) (mask : Op .bool sh) (other : Op .real sh)
    (u base : BlockState) (hmem : u.mem = base.mem)
    (offsTile : Tile .nat sh) (maskTile : Tile .bool sh) (otherTile : Tile .real sh)
    (hoff : evalOp offsets u = some offsTile)
    (hmask : evalOp mask u = some maskTile)
    (hother : evalOp other u = some otherTile) :
    evalOp (.load .real (MemAccess.region Rg offsets) (MaskOpt.maskOther mask other)) u
      = some ⟨fun i => if maskTile.data i then
          base.readMemValue .real (Region.cast Rg) (offsTile.data i)
        else otherTile.data i⟩ := by
  rw [ctx_evalOp_load_region_maskOther Rg offsets mask other u offsTile maskTile otherTile
    hoff hmask hother]
  refine congrArg some ?_
  ext i
  simp only [BlockState.readMemValue, BlockState.readMemAs, hmem]

/-- The `other=0.0` broadcast tile of the `q`/`k`/`v` loads, at **any** state:
hoisted out of the walk so the elaborator never re-runs `simp only [evalOp]`
against a deep `setReg` tower. -/
private theorem bloomFwdIO_zeroBroadcast_eval (sh : TileShape) (u : BlockState) :
    @evalOp .real sh ((Op.const (0.0 : ℝ)).broadcast sh) u
      = some (⟨fun _ : TileIndex sh => some (0.0 : ℝ)⟩ : Tile .real sh) := by
  simp only [evalOp, Option.bind_eq_bind, Option.bind_some]; rfl

/-- `evalOpR` of the `k`-load mask `((start_n + offs_n[None,:]) < block_end_loc)
& (offs_d[:,None] < head_dim)` at pinned registers (R-side wrapper of
`bloomKMask_evalG`). -/
private theorem bloomFwdIO_kmaskR_eval (R : RoundingModel) {BN D : Nat} (u : BlockState)
    (SN bel head_dim : Nat)
    (hsn : u.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hn : u.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => j.val)))
    (hd : u.regs .nat [D] "offs_d" = some (Tile.vec (fun e : Fin D => e.val)))
    (hbel : u.regs .nat [] "block_end_loc" = some (Tile.scalar bel)) :
    evalOpR R (Op.boolAnd Broadcast.nil.consR.consL
        (Op.lt .nat Broadcast.scalarR
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_n")))
          (Op.ref .nat [] "block_end_loc"))
        (Op.lt .nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [D] "offs_d"))
          (Op.constNat head_dim))) u
      = some (⟨fun idx : TileIndex [D, BN] =>
          decide (SN + idx.2.1.val < bel) && decide (idx.1.val < head_dim)⟩
            : Tile .bool [D, BN]) := by
  rw [show evalOpR R (Op.boolAnd Broadcast.nil.consR.consL
        (Op.lt .nat Broadcast.scalarR
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_n")))
          (Op.ref .nat [] "block_end_loc"))
        (Op.lt .nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [D] "offs_d"))
          (Op.constNat head_dim))) u
      = evalOp (Op.boolAnd Broadcast.nil.consR.consL
          (Op.lt .nat Broadcast.scalarR
            (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
              (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_n")))
            (Op.ref .nat [] "block_end_loc"))
          (Op.lt .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [D] "offs_d"))
            (Op.constNat head_dim))) u from by
    simp only [evalOpR.eq_def, evalOp.eq_def]]
  exact bloomKMask_evalG u SN bel head_dim hsn hn hd hbel

/-- `evalOpR` of the `v`-load mask `((start_n + offs_n[:,None]) < block_end_loc)
& (offs_d[None,:] < head_dim)` at pinned registers (R-side wrapper of
`bloomVMask_evalG`). -/
private theorem bloomFwdIO_vmaskR_eval (R : RoundingModel) {BN D : Nat} (u : BlockState)
    (SN bel head_dim : Nat)
    (hsn : u.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hn : u.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => j.val)))
    (hd : u.regs .nat [D] "offs_d" = some (Tile.vec (fun e : Fin D => e.val)))
    (hbel : u.regs .nat [] "block_end_loc" = some (Tile.scalar bel)) :
    evalOpR R (Op.boolAnd Broadcast.nil.consL.consR
        (Op.lt .nat Broadcast.scalarR
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BN] "offs_n")))
          (Op.ref .nat [] "block_end_loc"))
        (Op.lt .nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d"))
          (Op.constNat head_dim))) u
      = some (⟨fun idx : TileIndex [BN, D] =>
          decide (SN + idx.1.val < bel) && decide (idx.2.1.val < head_dim)⟩
            : Tile .bool [BN, D]) := by
  rw [show evalOpR R (Op.boolAnd Broadcast.nil.consL.consR
        (Op.lt .nat Broadcast.scalarR
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BN] "offs_n")))
          (Op.ref .nat [] "block_end_loc"))
        (Op.lt .nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d"))
          (Op.constNat head_dim))) u
      = evalOp (Op.boolAnd Broadcast.nil.consL.consR
          (Op.lt .nat Broadcast.scalarR
            (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BN] "offs_n")))
            (Op.ref .nat [] "block_end_loc"))
          (Op.lt .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d"))
            (Op.constNat head_dim))) u from by
    simp only [evalOpR.eq_def, evalOp.eq_def]]
  exact bloomVMask_evalG u SN bel head_dim hsn hn hd hbel

/-- `evalOpR` of the shared row/channel mask
`(offs_m[:,None] < cur_batch_seq_len) & (offs_d[None,:] < head_dim)` (the `q`
load's and the terminal store's mask) at pinned registers (R-side wrapper of
`bloomQMask_evalG`). -/
private theorem bloomFwdIO_qmaskR_eval (R : RoundingModel) {BM D : Nat} (u : BlockState)
    (gOM : Fin BM → Nat) (sl head_dim : Nat)
    (hm : u.regs .nat [BM] "offs_m" = some (Tile.vec gOM))
    (hd : u.regs .nat [D] "offs_d" = some (Tile.vec (fun e : Fin D => e.val)))
    (hsl : u.regs .nat [] "cur_batch_seq_len" = some (Tile.scalar sl)) :
    evalOpR R (Op.boolAnd Broadcast.nil.consL.consR
        (Op.lt .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m"))
          (Op.ref .nat [] "cur_batch_seq_len"))
        (Op.lt .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d"))
          (Op.constNat head_dim))) u
      = some (⟨fun idx : TileIndex [BM, D] =>
          decide (gOM idx.1 < sl) && decide (idx.2.1.val < head_dim)⟩
            : Tile .bool [BM, D]) := by
  rw [show evalOpR R (Op.boolAnd Broadcast.nil.consL.consR
        (Op.lt .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m"))
          (Op.ref .nat [] "cur_batch_seq_len"))
        (Op.lt .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d"))
          (Op.constNat head_dim))) u
      = evalOp (Op.boolAnd Broadcast.nil.consL.consR
          (Op.lt .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m"))
            (Op.ref .nat [] "cur_batch_seq_len"))
          (Op.lt .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d"))
            (Op.constNat head_dim))) u from by
    simp only [evalOpR.eq_def, evalOp.eq_def]]
  exact bloomQMask_evalG u gOM sl head_dim hm hd hsl

/-- Pinned-register eval of the `kv_loc` gather's **address** tree
`stride_req_b·cur_batch_req_idx + stride_req_s·(start_n + offs_n)` (the
skin's `gread` window, one chain level: pids and slots only). -/
private theorem bloomFwdIO_gread_eval {BN : Nat} (u : BlockState)
    (rqi SN stride_req_b stride_req_s : Nat)
    (hrqi : u.regs .nat [] "cur_batch_req_idx" = some (Tile.scalar rqi))
    (hsn : u.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hn : u.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => j.val))) :
    evalOp (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.constNat stride_req_b)
          (Op.ref .nat [] "cur_batch_req_idx"))
        (Op.mul .nat Broadcast.scalarL (Op.constNat stride_req_s)
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
            (Op.ref .nat [BN] "offs_n")))) u
      = some (Tile.vec (fun j : Fin BN =>
          stride_req_b * rqi + stride_req_s * (SN + j.val))) := by
  simp only [evalOp_add, evalOp_mul, evalOp_constNat, evalOp_ref, hrqi, hsn, hn,
    Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_; ext idx
  simp [Tile.bop_data, Tile.scalar_data, Tile.vec_data, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.add, NumericDType.mul]

/-- `evalOpR` of the gather address tree is the exact evaluation
(register-only `.nat` arithmetic — R-independent). -/
private theorem bloomFwdIO_greadR_eval (R : RoundingModel) {BN : Nat} (u : BlockState)
    (rqi SN stride_req_b stride_req_s : Nat)
    (hrqi : u.regs .nat [] "cur_batch_req_idx" = some (Tile.scalar rqi))
    (hsn : u.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hn : u.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => j.val))) :
    evalOpR R (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.constNat stride_req_b)
          (Op.ref .nat [] "cur_batch_req_idx"))
        (Op.mul .nat Broadcast.scalarL (Op.constNat stride_req_s)
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
            (Op.ref .nat [BN] "offs_n")))) u
      = some (Tile.vec (fun j : Fin BN =>
          stride_req_b * rqi + stride_req_s * (SN + j.val))) := by
  rw [show evalOpR R (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.constNat stride_req_b)
          (Op.ref .nat [] "cur_batch_req_idx"))
        (Op.mul .nat Broadcast.scalarL (Op.constNat stride_req_s)
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
            (Op.ref .nat [BN] "offs_n")))) u
      = evalOp (Op.add .nat Broadcast.scalarL
          (Op.mul .nat Broadcast.nil (Op.constNat stride_req_b)
            (Op.ref .nat [] "cur_batch_req_idx"))
          (Op.mul .nat Broadcast.scalarL (Op.constNat stride_req_s)
            (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
              (Op.ref .nat [BN] "offs_n")))) u from by
    simp only [evalOpR.eq_def, evalOp.eq_def]]
  exact bloomFwdIO_gread_eval u rqi SN stride_req_b stride_req_s hrqi hsn hn

/-- Pinned-register eval of the `kv_loc` gather's **mask**
`(start_n + offs_n) < block_end_loc` — the skin's `gmask`, the liveness guard
the `K`/`V` loads also carry, `[BN]`-shaped with **no** `head_dim` conjunct. -/
private theorem bloomFwdIO_gmask_eval {BN : Nat} (u : BlockState) (SN bel : Nat)
    (hsn : u.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hn : u.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => j.val)))
    (hbel : u.regs .nat [] "block_end_loc" = some (Tile.scalar bel)) :
    evalOp (Op.lt .nat Broadcast.scalarR
        (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
          (Op.ref .nat [BN] "offs_n"))
        (Op.ref .nat [] "block_end_loc")) u
      = some (⟨fun idx : TileIndex [BN] => decide (SN + idx.1.val < bel)⟩
          : Tile .bool [BN]) := by
  simp only [evalOp_lt, evalOp_add, evalOp_ref, hsn, hn, hbel,
    Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_; ext idx
  simp [Tile.cop_data, Tile.bop_data, Tile.scalar_data, Tile.scalar_data_index, Tile.vec_data,
    Broadcast.leftIndex, Broadcast.rightIndex, ComparableDType.lt, NumericDType.add]

/-- `evalOpR` of the gather mask at pinned registers (R-side wrapper of
`bloomFwdIO_gmask_eval`). -/
private theorem bloomFwdIO_gmaskR_eval (R : RoundingModel) {BN : Nat} (u : BlockState)
    (SN bel : Nat)
    (hsn : u.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hn : u.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => j.val)))
    (hbel : u.regs .nat [] "block_end_loc" = some (Tile.scalar bel)) :
    evalOpR R (Op.lt .nat Broadcast.scalarR
        (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
          (Op.ref .nat [BN] "offs_n"))
        (Op.ref .nat [] "block_end_loc")) u
      = some (⟨fun idx : TileIndex [BN] => decide (SN + idx.1.val < bel)⟩
          : Tile .bool [BN]) := by
  rw [show evalOpR R (Op.lt .nat Broadcast.scalarR
        (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
          (Op.ref .nat [BN] "offs_n"))
        (Op.ref .nat [] "block_end_loc")) u
      = evalOp (Op.lt .nat Broadcast.scalarR
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
            (Op.ref .nat [BN] "offs_n"))
          (Op.ref .nat [] "block_end_loc")) u from by
    simp only [evalOpR.eq_def, evalOp.eq_def]]
  exact bloomFwdIO_gmask_eval u SN bel hsn hn hbel

/-- Pinned-register `off_o` eval (the postLoop offset tree, generic row
vector `gOM`; bloom's `cur_head` is the bare `pid₁`, no `% H` decode). -/
private theorem bloomFwdIO_offo_eval {BM D : Nat} (u : BlockState) (gOM : Fin BM → Nat)
    (cbsi ch stride_obs stride_oh stride_od : Nat)
    (hcbsi : u.regs .nat [] "cur_batch_in_all_start_index" = some (Tile.scalar cbsi))
    (hch : u.regs .nat [] "cur_head" = some (Tile.scalar ch))
    (hom : u.regs .nat [BM] "offs_m" = some (Tile.vec gOM))
    (hod : u.regs .nat [D] "offs_d" = some (Tile.vec (fun e : Fin D => e.val))) :
    evalOp (Op.add .nat Broadcast.nil.consL.consR
        (Op.add .nat Broadcast.scalarR
          (Op.mul .nat Broadcast.scalarR
            (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "cur_batch_in_all_start_index")
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m")))
            (Op.constNat stride_obs))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat stride_oh)))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d"))
          (Op.constNat stride_od))) u
      = some (⟨fun idx : TileIndex [BM, D] =>
          (cbsi + gOM idx.1) * stride_obs + ch * stride_oh + idx.2.1.val * stride_od⟩
            : Tile .nat [BM, D]) := by
  simp only [evalOp_add, evalOp_mul, evalOp_constNat, evalOp_ref,
    ctx_evalOp_expandDim_one_nat, ctx_evalOp_expandDim_zero_nat,
    hcbsi, hch, hom, hod, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_; ext idx
  simp [Tile.bop_data, Tile.scalar_data, Tile.vec_data, Tile.expandDim, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.add, NumericDType.mul]

/-- `evalOpR` of the `off_o` offset tree is the exact evaluation
(register-only `.nat` arithmetic — R-independent, ∀-state). -/
private theorem bloomFwdIO_offoR_castFree (R : RoundingModel)
    {BM D : Nat} (stride_obs stride_oh stride_od : Nat) (u : BlockState) :
    evalOpR R (Op.add .nat Broadcast.nil.consL.consR
        (Op.add .nat Broadcast.scalarR
          (Op.mul .nat Broadcast.scalarR
            (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "cur_batch_in_all_start_index")
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m")))
            (Op.constNat stride_obs))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat stride_oh)))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d"))
          (Op.constNat stride_od))) u
      = evalOp (Op.add .nat Broadcast.nil.consL.consR
          (Op.add .nat Broadcast.scalarR
            (Op.mul .nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "cur_batch_in_all_start_index")
                (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m")))
              (Op.constNat stride_obs))
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat stride_oh)))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d"))
            (Op.constNat stride_od))) u := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

/-- `evalOpR` of the `out_ptrs` pointer tree is the exact evaluation. -/
private theorem bloomFwdIO_outptrR_castFree (R : RoundingModel) (Out : RegionName)
    (BM D : Nat) (u : BlockState) :
    evalOpR R (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Out) (Op.ref .nat [BM, D] "off_o")) u
      = evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Out)
          (Op.ref .nat [BM, D] "off_o")) u := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

/-- Weak (safety-walk) invariant: address-register pins + value-register
existence, anchored to the launch state `s`. All pins are state-based (slot
reads spelled through `readMemValue` at cell `pids 0`, `block_end_loc`
through `bloomFwdIOBel` at `pids 2`); `kv_group_num = 1` leaves the register
`cur_kv_head` at the literal `pids 1 / 1`. The `kv_loc` register is **not**
pinned: it is re-derived inside each body pass (the gather is a loop-local
read). -/
private def bloomFwdIOSafeInvW (B_Start_Loc B_Seqlen B_req_idx b_prompt_cache_len : Region .nat)
    (BLOCK_DMODEL BLOCK_M BLOCK_N : Nat) (s s' : BlockState) : Prop :=
  s'.mem = s.mem
  ∧ s'.pids = s.pids
  ∧ s'.regs .nat [] "cur_batch" = some (Tile.scalar (s.pids 0))
  ∧ s'.regs .nat [] "cur_head" = some (Tile.scalar (s.pids 1))
  ∧ s'.regs .nat [] "cur_kv_head" = some (Tile.scalar (s.pids 1 / 1))
  ∧ s'.regs .nat [] "prompt_cache_len"
      = some (Tile.scalar (s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 0)))
  ∧ s'.regs .nat [] "cur_batch_in_all_start_index"
      = some (Tile.scalar (s.readMemValue .nat (Region.cast B_Start_Loc) (s.pids 0)))
  ∧ s'.regs .nat [] "cur_batch_seq_len"
      = some (Tile.scalar (s.readMemValue .nat (Region.cast B_Seqlen) (s.pids 0)
          - s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 0)))
  ∧ s'.regs .nat [] "cur_batch_req_idx"
      = some (Tile.scalar (s.readMemValue .nat (Region.cast B_req_idx) (s.pids 0)))
  ∧ s'.regs .nat [] "block_mask"
      = some (Tile.scalar (if BLOCK_M * s.pids 2
          < s.readMemValue .nat (Region.cast B_Seqlen) (s.pids 0)
            - s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 0)
          then 1 else 0))
  ∧ s'.regs .nat [] "block_end_loc"
      = some (Tile.scalar (bloomFwdIOBel BLOCK_M (s.pids 2)
          (s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 0))
          (s.readMemValue .nat (Region.cast B_Seqlen) (s.pids 0))))
  ∧ s'.regs .nat [BLOCK_N] "offs_n" = some (Tile.vec (fun jn : Fin BLOCK_N => jn.val))
  ∧ s'.regs .nat [BLOCK_DMODEL] "offs_d" = some (Tile.vec (fun e : Fin BLOCK_DMODEL => e.val))
  ∧ s'.regs .nat [BLOCK_M] "offs_m"
      = some (Tile.vec (fun r : Fin BLOCK_M => s.pids 2 * BLOCK_M + r.val))
  ∧ (∃ qT : Tile .real [BLOCK_M, BLOCK_DMODEL],
      s'.regs .real [BLOCK_M, BLOCK_DMODEL] "q" = some qT)
  ∧ (∃ mT : Tile .real [BLOCK_M], s'.regs .real [BLOCK_M] "m_i" = some mT)
  ∧ (∃ lT : Tile .real [BLOCK_M], s'.regs .real [BLOCK_M] "l_i" = some lT)
  ∧ (∃ aT : Tile .real [BLOCK_M, BLOCK_DMODEL],
      s'.regs .real [BLOCK_M, BLOCK_DMODEL] "acc" = some aT)

set_option maxHeartbeats 8000000 in
set_option maxRecDepth 8000 in
/-- **Weak preLoop walk** (single pass): from an **arbitrary** launch state
the 19 preLoop statements are trace-safe (the four `.nat` metadata loads
bounded by the slot windows at cell `pid₀`, the masked `q` load by the
`read1` window under the row-∧-channel mask) and step to a state satisfying
`bloomFwdIOSafeInvW`. -/
private theorem bloomFwdIO_preLoopW (R : RoundingModel) (bounds : RegionBounds)
    (Q : RegionName) (B_Start_Loc B_Seqlen B_req_idx b_prompt_cache_len : Region .nat)
    (stride_qbs stride_qh stride_qd head_dim BLOCK_DMODEL BLOCK_M BLOCK_N : Nat)
    (s : BlockState)
    (hbPC : s.pids 0 < bounds (Region.cast b_prompt_cache_len))
    (hbSL : s.pids 0 < bounds (Region.cast B_Start_Loc))
    (hbSQ : s.pids 0 < bounds (Region.cast B_Seqlen))
    (hbRQ : s.pids 0 < bounds (Region.cast B_req_idx))
    (hbQ : ∀ (i : Fin BLOCK_M) (e : Fin BLOCK_DMODEL),
      s.pids 2 * BLOCK_M + i.val < s.readMemValue .nat (Region.cast B_Seqlen) (s.pids 0)
          - s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 0) →
      e.val < head_dim →
      (s.readMemValue .nat (Region.cast B_Start_Loc) (s.pids 0)
          + (s.pids 2 * BLOCK_M + i.val)) * stride_qbs
        + s.pids 1 * stride_qh + e.val * stride_qd < bounds Q) :
    Stmt.TraceSafeListR R bounds (bloomPreLoopG Q B_Start_Loc B_Seqlen B_req_idx
        b_prompt_cache_len stride_qbs stride_qh stride_qd head_dim 1
        BLOCK_DMODEL BLOCK_M BLOCK_N) s
      ∧ ∃ s0, stepStmtsR R (bloomPreLoopG Q B_Start_Loc B_Seqlen B_req_idx b_prompt_cache_len
            stride_qbs stride_qh stride_qd head_dim 1 BLOCK_DMODEL BLOCK_M BLOCK_N) s = some s0
          ∧ bloomFwdIOSafeInvW B_Start_Loc B_Seqlen B_req_idx b_prompt_cache_len
              BLOCK_DMODEL BLOCK_M BLOCK_N s s0 := by
  unfold bloomPreLoopG
  -- stmt 0: cur_batch = programId 0
  refine bloomFwdIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (bloomFwdIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp _ _ = some (Tile.scalar (s.pids 0)) from evalOp_programId 0 s)) ?_
  -- stmt 1: cur_head = programId 1
  refine bloomFwdIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (bloomFwdIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp _ _ = some (Tile.scalar (s.pids 1)) from evalOp_programId 1 _)) ?_
  -- stmt 2: start_m = programId 2
  refine bloomFwdIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (bloomFwdIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp _ _ = some (Tile.scalar (s.pids 2)) from evalOp_programId 2 _)) ?_
  -- stmt 3: cur_kv_head = cur_head // 1
  refine bloomFwdIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (bloomFwdIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp (Op.floorDiv .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat 1)) _
          = some (Tile.scalar (s.pids 1 / 1)) from by
        rw [ctx_evalOp_floorDiv]
        simp only [evalOp_ref, evalOp_constNat, BlockState.setReg_same, BlockState.setReg_ne_name,
          ne_eq, String.reduceEq, not_false_eq_true, BlockState.setReg_pids,
          Option.bind_eq_bind, Option.bind_some]
        rfl)) ?_
  -- stmt 4: cur_batch_in_all_start_index = load(B_Start_Loc + cur_batch)
  refine bloomFwdIO_walkCons ?_
    (bloomFwdIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp (Op.load .nat
          (MemAccess.region B_Start_Loc (Op.ref .nat [] "cur_batch")) MaskOpt.none) _
          = some (Tile.scalar (s.readMemValue .nat (Region.cast B_Start_Loc)
              (s.pids 0))) from by
        rw [ctx_evalOp_load_scalar_nat B_Start_Loc (Op.ref .nat [] "cur_batch") _
          (s.pids 0)
          (by rw [evalOp_ref]; simp [BlockState.setReg_ne_name, BlockState.setReg_same])]
        rfl)) ?_
  · simp only [Stmt.TraceSafeR, Op.SafeAtR, MaskOpt.ActiveR, MemAccess.ActiveAddressSafeR,
      memAccessActiveAddressSafeR]
    refine ⟨by simp [Op.SafeAtR.eq_def], by simp [MaskOpt.SafeAtR], ?_⟩
    intro offsets hoff idx _
    rw [evalOpR_ref] at hoff
    simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same] at hoff
    obtain rfl := Option.some.inj hoff
    exact hbSL
  -- stmt 5: prompt_cache_len = load(bpc + cur_batch)  [slot-window bound]
  refine bloomFwdIO_walkCons ?_
    (bloomFwdIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp (Op.load .nat
          (MemAccess.region b_prompt_cache_len (Op.ref .nat [] "cur_batch")) MaskOpt.none) _
          = some (Tile.scalar (s.readMemValue .nat (Region.cast b_prompt_cache_len)
              (s.pids 0))) from by
        rw [ctx_evalOp_load_scalar_nat b_prompt_cache_len (Op.ref .nat [] "cur_batch") _
          (s.pids 0)
          (by rw [evalOp_ref]; simp [BlockState.setReg_ne_name, BlockState.setReg_same])]
        rfl)) ?_
  · simp only [Stmt.TraceSafeR, Op.SafeAtR, MaskOpt.ActiveR, MemAccess.ActiveAddressSafeR,
      memAccessActiveAddressSafeR]
    refine ⟨by simp [Op.SafeAtR.eq_def], by simp [MaskOpt.SafeAtR], ?_⟩
    intro offsets hoff idx _
    rw [evalOpR_ref] at hoff
    simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same] at hoff
    obtain rfl := Option.some.inj hoff
    exact hbPC
  -- stmt 6: cur_batch_seq_len = load(B_Seqlen + cur_batch) - prompt_cache_len
  refine bloomFwdIO_walkCons ?_
    (bloomFwdIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp (Op.sub .nat Broadcast.nil
          (Op.load .nat (MemAccess.region B_Seqlen (Op.ref .nat [] "cur_batch")) MaskOpt.none)
          (Op.ref .nat [] "prompt_cache_len")) _
          = some (Tile.scalar (s.readMemValue .nat (Region.cast B_Seqlen) (s.pids 0)
              - s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 0))) from by
        rw [evalOp_sub, ctx_evalOp_load_scalar_nat B_Seqlen (Op.ref .nat [] "cur_batch") _
          (s.pids 0)
          (by rw [evalOp_ref]; simp [BlockState.setReg_ne_name, BlockState.setReg_same])]
        simp only [evalOp_ref, BlockState.setReg_same, Option.bind_eq_bind, Option.bind_some]
        rfl)) ?_
  · simp only [Stmt.TraceSafeR, Op.SafeAtR, MaskOpt.ActiveR, MemAccess.ActiveAddressSafeR,
      memAccessActiveAddressSafeR]
    refine ⟨⟨by simp [Op.SafeAtR.eq_def], by simp [MaskOpt.SafeAtR], ?_⟩,
      by simp [Op.SafeAtR.eq_def]⟩
    intro offsets hoff idx _
    rw [evalOpR_ref] at hoff
    simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same] at hoff
    obtain rfl := Option.some.inj hoff
    exact hbSQ
  -- stmt 7: cur_batch_req_idx = load(B_req_idx + cur_batch)  [slot-window bound]
  refine bloomFwdIO_walkCons ?_
    (bloomFwdIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp (Op.load .nat
          (MemAccess.region B_req_idx (Op.ref .nat [] "cur_batch")) MaskOpt.none) _
          = some (Tile.scalar (s.readMemValue .nat (Region.cast B_req_idx)
              (s.pids 0))) from by
        rw [ctx_evalOp_load_scalar_nat B_req_idx (Op.ref .nat [] "cur_batch") _
          (s.pids 0)
          (by rw [evalOp_ref]; simp [BlockState.setReg_ne_name, BlockState.setReg_same])]
        rfl)) ?_
  · simp only [Stmt.TraceSafeR, Op.SafeAtR, MaskOpt.ActiveR, MemAccess.ActiveAddressSafeR,
      memAccessActiveAddressSafeR]
    refine ⟨by simp [Op.SafeAtR.eq_def], by simp [MaskOpt.SafeAtR], ?_⟩
    intro offsets hoff idx _
    rw [evalOpR_ref] at hoff
    simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same] at hoff
    obtain rfl := Option.some.inj hoff
    exact hbRQ
  -- stmt 8: block_start_loc = BLOCK_M * start_m
  refine bloomFwdIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (bloomFwdIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp (Op.mul .nat Broadcast.nil (Op.constNat BLOCK_M)
            (Op.ref .nat [] "start_m")) _
          = some (Tile.scalar (BLOCK_M * s.pids 2)) from by
        rw [evalOp_mul]
        simp only [evalOp_constNat, evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name,
          ne_eq, String.reduceEq, not_false_eq_true, BlockState.setReg_pids,
          Option.bind_eq_bind, Option.bind_some]
        rfl)) ?_
  -- stmt 9: offs_n = arange BLOCK_N
  refine bloomFwdIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (bloomFwdIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp (Op.arange BLOCK_N) _ = some (Tile.vec (fun j : Fin BLOCK_N => j.val)) from
        evalOp_arange BLOCK_N _)) ?_
  -- stmt 10: offs_d = arange BLOCK_DMODEL
  refine bloomFwdIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (bloomFwdIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp (Op.arange BLOCK_DMODEL) _
          = some (Tile.vec (fun e : Fin BLOCK_DMODEL => e.val)) from
        evalOp_arange BLOCK_DMODEL _)) ?_
  -- stmt 11: offs_m = start_m * BLOCK_M + arange BLOCK_M
  refine bloomFwdIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (bloomFwdIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp (Op.add .nat Broadcast.scalarL
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat BLOCK_M))
            (Op.arange BLOCK_M)) _
          = some (Tile.vec (fun r : Fin BLOCK_M => s.pids 2 * BLOCK_M + r.val)) from by
        rw [evalOp_add, evalOp_mul]
        simp only [evalOp_ref, evalOp_arange, evalOp_constNat, BlockState.setReg_same,
          BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
          BlockState.setReg_pids, Option.bind_eq_bind, Option.bind_some]
        refine congrArg some ?_; ext r
        simp [Tile.bop_data, Tile.scalar_data, Tile.vec_data, Broadcast.leftIndex,
          Broadcast.rightIndex, NumericDType.add, NumericDType.mul])) ?_
  -- stmt 12: off_q
  refine bloomFwdIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (bloomFwdIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp (Op.add .nat Broadcast.nil.consL.consR
          (Op.add .nat Broadcast.scalarR
            (Op.mul .nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "cur_batch_in_all_start_index")
                (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m")))
              (Op.constNat stride_qbs))
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat stride_qh)))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_DMODEL] "offs_d"))
            (Op.constNat stride_qd))) _
          = some (⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
              (s.readMemValue .nat (Region.cast B_Start_Loc) (s.pids 0)
                  + (s.pids 2 * BLOCK_M + idx.1.val))
                  * stride_qbs + s.pids 1 * stride_qh + idx.2.1.val * stride_qd⟩
                : Tile .nat [BLOCK_M, BLOCK_DMODEL]) from by
        simp only [evalOp_add, evalOp_mul, evalOp_constNat, evalOp_ref,
          ctx_evalOp_expandDim_one_nat, ctx_evalOp_expandDim_zero_nat,
          BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
          not_false_eq_true, Option.bind_eq_bind, Option.bind_some]
        refine congrArg some ?_; ext idx
        simp [Tile.bop_data, Tile.scalar_data, Tile.vec_data, Tile.expandDim, Broadcast.leftIndex,
          Broadcast.rightIndex, NumericDType.add, NumericDType.mul])) ?_
  -- stmt 13: q = masked load  [read1-window bound, row ∧ channel]
  refine bloomFwdIO_walkCons ?_
    (bloomFwdIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (ctx_evalOp_load_region_maskOther Q (Op.ref .nat [BLOCK_M, BLOCK_DMODEL] "off_q") _ _ _
        (⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
            (s.readMemValue .nat (Region.cast B_Start_Loc) (s.pids 0)
                + (s.pids 2 * BLOCK_M + idx.1.val))
                * stride_qbs + s.pids 1 * stride_qh + idx.2.1.val * stride_qd⟩
              : Tile .nat [BLOCK_M, BLOCK_DMODEL])
        (⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
            decide (s.pids 2 * BLOCK_M + idx.1.val
              < s.readMemValue .nat (Region.cast B_Seqlen) (s.pids 0)
                - s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 0))
            && decide (idx.2.1.val < head_dim)⟩
              : Tile .bool [BLOCK_M, BLOCK_DMODEL])
        (⟨fun _ : TileIndex [BLOCK_M, BLOCK_DMODEL] => some (0.0 : ℝ)⟩
              : Tile .real [BLOCK_M, BLOCK_DMODEL])
        (by rw [evalOp_ref]; simp [BlockState.setReg_same])
        (bloomQMask_evalG _ (fun r : Fin BLOCK_M => s.pids 2 * BLOCK_M + r.val) _ head_dim
          (by simp [BlockState.setReg_ne_name, BlockState.setReg_same])
          (by simp [BlockState.setReg_ne_name, BlockState.setReg_same])
          (by simp [BlockState.setReg_ne_name, BlockState.setReg_same]))
        (by simp only [evalOp, Option.bind_eq_bind, Option.bind_some]; rfl))) ?_
  · simp only [Stmt.TraceSafeR, Op.SafeAtR, MaskOpt.ActiveR, MemAccess.ActiveAddressSafeR,
      memAccessActiveAddressSafeR]
    refine ⟨by simp [Op.SafeAtR.eq_def], ⟨by simp [Op.SafeAtR.eq_def],
      by simp [Op.SafeAtR.eq_def]⟩, ?_⟩
    intro offsets hoff idx hactive
    rw [evalOpR_ref] at hoff
    simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same] at hoff
    obtain rfl := Option.some.inj hoff
    obtain ⟨masks, hmask, hmi⟩ := hactive
    rw [bloomFwdIO_qmaskR_eval R _ (fun r : Fin BLOCK_M => s.pids 2 * BLOCK_M + r.val)
      (s.readMemValue .nat (Region.cast B_Seqlen) (s.pids 0)
        - s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 0)) head_dim
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
        BlockState.setReg_same])
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
        BlockState.setReg_same])
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
        BlockState.setReg_same])] at hmask
    obtain rfl := Option.some.inj hmask
    obtain ⟨i, e, pu⟩ := idx
    have hcond : s.pids 2 * BLOCK_M + i.val
          < s.readMemValue .nat (Region.cast B_Seqlen) (s.pids 0)
            - s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 0)
        ∧ e.val < head_dim := by
      simpa using hmi
    exact hbQ i e hcond.1 hcond.2
  -- stmt 14: m_i = full 0 + (-inf)
  refine bloomFwdIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (bloomFwdIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp (Op.add .real Broadcast.scalarR (Op.full [BLOCK_M] (Op.const 0)) Op.negInf) _
          = some (⟨fun _ : TileIndex [BLOCK_M] => (⊥ : WithBot ℝ)⟩
              : Tile .real [BLOCK_M]) from by
        rw [evalOp_add]
        simp only [evalOp_full, evalOp_const, evalOp_negInf, Option.bind_eq_bind, Option.bind_some]
        refine congrArg some ?_; ext idx
        simp only [Tile.bop_data, Tile.scalar_data, Broadcast.leftIndex, Broadcast.rightIndex,
          NumericDType.add, WithBot.realAdd, Option.map₂, Option.bind, Option.map]
        rfl)) ?_
  -- stmt 15: l_i = full 0
  refine bloomFwdIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (bloomFwdIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp (Op.full [BLOCK_M] (Op.const 0)) _
          = some (⟨fun _ : TileIndex [BLOCK_M] => some (0 : ℝ)⟩
              : Tile .real [BLOCK_M]) from by
        simp [evalOp_full, evalOp_const])) ?_
  -- stmt 16: acc = full 0
  refine bloomFwdIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (bloomFwdIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp (Op.full [BLOCK_M, BLOCK_DMODEL] (Op.const 0)) _
          = some (⟨fun _ : TileIndex [BLOCK_M, BLOCK_DMODEL] => some (0 : ℝ)⟩
              : Tile .real [BLOCK_M, BLOCK_DMODEL]) from by
        simp [evalOp_full, evalOp_const])) ?_
  -- stmt 17: block_mask
  refine bloomFwdIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (bloomFwdIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp ((Op.lt .nat Broadcast.nil (Op.ref .nat [] "block_start_loc")
              (Op.ref .nat [] "cur_batch_seq_len")).where (Op.constNat 1) (Op.constNat 0)) _
          = some (Tile.scalar (if BLOCK_M * s.pids 2
              < s.readMemValue .nat (Region.cast B_Seqlen) (s.pids 0)
                - s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 0)
              then 1 else 0)) from by
        rw [evalOp_where]
        simp only [evalOp_lt, evalOp_ref, evalOp_constNat, BlockState.setReg_same,
          BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
          Option.bind_eq_bind, Option.bind_some]
        refine congrArg some ?_
        ext _idx
        simp only [Tile.select_data, Tile.cop_data, Tile.scalar_data_index, ComparableDType.lt]
        by_cases h : BLOCK_M * s.pids 2
            < s.readMemValue .nat (Region.cast B_Seqlen) (s.pids 0)
              - s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 0)
        · simp [h]
        · simp [h])) ?_
  -- stmt 18: block_end_loc
  refine bloomFwdIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (bloomFwdIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp ((Op.lt .nat Broadcast.nil
              (Op.add .nat Broadcast.nil
                (Op.mul .nat Broadcast.nil
                  (Op.add .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat 1))
                  (Op.constNat BLOCK_M))
                (Op.ref .nat [] "prompt_cache_len"))
              (Op.add .nat Broadcast.nil (Op.ref .nat [] "cur_batch_seq_len")
                (Op.ref .nat [] "prompt_cache_len"))).where
            (Op.add .nat Broadcast.nil
              (Op.mul .nat Broadcast.nil
                (Op.add .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat 1))
                (Op.constNat BLOCK_M))
              (Op.ref .nat [] "prompt_cache_len"))
            (Op.add .nat Broadcast.nil (Op.ref .nat [] "cur_batch_seq_len")
              (Op.ref .nat [] "prompt_cache_len"))) _
          = some (Tile.scalar (bloomFwdIOBel BLOCK_M (s.pids 2)
              (s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 0))
              (s.readMemValue .nat (Region.cast B_Seqlen) (s.pids 0)))) from by
        simp only [bloomFwdIOBel]
        rw [evalOp_where]
        simp only [evalOp_lt, evalOp_add, evalOp_mul, evalOp_ref, evalOp_constNat,
          BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
          not_false_eq_true, Option.bind_eq_bind, Option.bind_some]
        refine congrArg some ?_
        ext _idx
        simp only [Tile.select_data, Tile.cop_data, Tile.bop_data, Tile.scalar_data_index,
          ComparableDType.lt, NumericDType.add, NumericDType.mul, Broadcast.leftIndex,
          Broadcast.rightIndex]
        by_cases h : (s.pids 2 + 1) * BLOCK_M
              + s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 0)
            < (s.readMemValue .nat (Region.cast B_Seqlen) (s.pids 0)
                - s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 0))
              + s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 0)
        · simp [h]
        · simp [h])) ?_
  refine ⟨Stmt.TraceSafeListR.nil_intro, _, stepStmtsR_nil R _, ?_⟩
  -- the weak-invariant pins of the final state
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_,
    ⟨_, by
      simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq,
        not_false_eq_true, reduceCtorEq]
      rfl⟩,
    ⟨_, by
      simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq,
        not_false_eq_true, reduceCtorEq]
      rfl⟩,
    ⟨_, by
      simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq,
        not_false_eq_true, reduceCtorEq]
      rfl⟩,
    ⟨_, by
      simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq,
        not_false_eq_true, reduceCtorEq]
      rfl⟩⟩
  · funext rg o; simp only [BlockState.setReg_mem]
  · simp only [BlockState.setReg_pids]
  all_goals
    (simp only [BlockState.setReg_ne_name, BlockState.setReg_same, BlockState.setReg_pids,
      BlockState.setReg_readMemValue, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq]
     try rfl)

set_option maxHeartbeats 8000000 in
set_option maxRecDepth 8000 in
/-- **Weak loop-body walk** (single pass): from any `bloomFwdIOSafeInvW` state
with the counter register set, the 26 body statements are trace-safe (the
`kv_loc` gather bounded by the skin's `gread` window and the two masked data
loads by the `read2`/`read3` windows at the current step — those two eat the
gathered tile, so their bounds are stated on the loaded page-table value, and
both carry the extra `offs_d < head_dim` channel clause) and step to a state
satisfying `bloomFwdIOSafeInvW` again — the value registers advance
existentially, the address registers are untouched. -/
private theorem bloomFwdIO_bodyW (R : RoundingModel) (bounds : RegionBounds)
    (Q K V Req_to_tokens B_req_idx : RegionName)
    (B_Start_Loc B_Seqlen B_req_idxR b_prompt_cache_len : Region .nat)
    (sm_scale : ℝ)
    (stride_req_b stride_req_s stride_kbs stride_kh stride_kd stride_vbs stride_vh stride_vd
      head_dim BLOCK_DMODEL BLOCK_M BLOCK_N : Nat) (hBN : 0 < BLOCK_N)
    (s stt : BlockState) (c : Nat)
    (hP : bloomFwdIOSafeInvW B_Start_Loc B_Seqlen B_req_idxR b_prompt_cache_len
        BLOCK_DMODEL BLOCK_M BLOCK_N s stt)
    (hbG : ∀ jL : Fin BLOCK_N,
      c + jL.val < bloomFwdIOBel BLOCK_M (s.pids 2)
          (s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 0))
          (s.readMemValue .nat (Region.cast B_Seqlen) (s.pids 0)) →
      stride_req_b * s.readMemValue .nat (Region.cast B_req_idxR) (s.pids 0)
          + stride_req_s * (c + jL.val) < bounds Req_to_tokens)
    (hbK : ∀ (e : Fin BLOCK_DMODEL) (jL : Fin BLOCK_N),
      c + jL.val < bloomFwdIOBel BLOCK_M (s.pids 2)
          (s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 0))
          (s.readMemValue .nat (Region.cast B_Seqlen) (s.pids 0)) →
      e.val < head_dim →
      s.readMemValue .nat (Region.cast Req_to_tokens)
            (stride_req_b * s.readMemValue .nat (Region.cast B_req_idxR) (s.pids 0)
              + stride_req_s * (c + jL.val)) * stride_kbs
        + s.pids 1 * stride_kh + e.val * stride_kd < bounds K)
    (hbV : ∀ (jL : Fin BLOCK_N) (e : Fin BLOCK_DMODEL),
      c + jL.val < bloomFwdIOBel BLOCK_M (s.pids 2)
          (s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 0))
          (s.readMemValue .nat (Region.cast B_Seqlen) (s.pids 0)) →
      e.val < head_dim →
      s.readMemValue .nat (Region.cast Req_to_tokens)
            (stride_req_b * s.readMemValue .nat (Region.cast B_req_idxR) (s.pids 0)
              + stride_req_s * (c + jL.val)) * stride_vbs
        + s.pids 1 * stride_vh + e.val * stride_vd < bounds V) :
    Stmt.TraceSafeListR R bounds
        (bloomLoopBodyG Q K V Req_to_tokens B_req_idx sm_scale stride_req_b stride_req_s
          stride_kbs stride_kh stride_kd stride_vbs stride_vh stride_vd
          head_dim BLOCK_DMODEL BLOCK_M BLOCK_N)
        (stt.setReg "start_n" .nat [] (Tile.scalar c))
      ∧ ∃ s', stepStmtsR R
            (bloomLoopBodyG Q K V Req_to_tokens B_req_idx sm_scale stride_req_b stride_req_s
              stride_kbs stride_kh stride_kd stride_vbs stride_vh stride_vd
              head_dim BLOCK_DMODEL BLOCK_M BLOCK_N)
            (stt.setReg "start_n" .nat [] (Tile.scalar c)) = some s'
          ∧ bloomFwdIOSafeInvW B_Start_Loc B_Seqlen B_req_idxR b_prompt_cache_len
              BLOCK_DMODEL BLOCK_M BLOCK_N s s' := by
  obtain ⟨hmem, hpids, hcb, hch, hckvh, hplen, hcbsi, hseq, hrqi, hbmk, hbel, hn, hd, hom,
    ⟨qT, hq⟩, ⟨mT, hmi⟩, ⟨lT, hli⟩, ⟨aT, hacc⟩⟩ := hP
  set plenv := s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 0) with hplenv
  set rqiv := s.readMemValue .nat (Region.cast B_req_idxR) (s.pids 0) with hrqiv
  set belv := bloomFwdIOBel BLOCK_M (s.pids 2)
    (s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 0))
    (s.readMemValue .nat (Region.cast B_Seqlen) (s.pids 0)) with hbelv
  set gOM : Fin BLOCK_M → Nat := fun r => s.pids 2 * BLOCK_M + r.val with hgOM
  -- the body writes registers only, so every read is the launch state's
  have hmemvN : ∀ (rg : RegionName) (o : Nat),
      stt.readMemValue .nat rg o = s.readMemValue .nat rg o := by
    intro rg o
    simp only [BlockState.readMemValue, BlockState.readMemTyped, hmem]
  -- the gathered page-table tile at this step (`other = 0` on dead lanes)
  set kvf : Fin BLOCK_N → Nat := fun jL =>
    if decide (c + jL.val < belv) then
      stt.readMemValue .nat (Region.cast Req_to_tokens)
        (stride_req_b * rqiv + stride_req_s * (c + jL.val))
    else 0 with hkvf
  have hkvfs : ∀ jL : Fin BLOCK_N, c + jL.val < belv →
      kvf jL = s.readMemValue .nat (Region.cast Req_to_tokens)
        (stride_req_b * rqiv + stride_req_s * (c + jL.val)) := by
    intro jL hlt
    simp only [hkvf]
    rw [if_pos (by simp only [decide_eq_true_eq]; exact hlt), hmemvN]
  set kloadT : Tile .real [BLOCK_DMODEL, BLOCK_N] :=
      ⟨fun idx : TileIndex [BLOCK_DMODEL, BLOCK_N] =>
        if decide (c + idx.2.1.val < belv) && decide (idx.1.val < head_dim) then
          stt.readMemValue .real (Region.cast K)
            (kvf idx.2.1 * stride_kbs
              + s.pids 1 / 1 * stride_kh + idx.1.val * stride_kd)
        else some (0.0 : ℝ)⟩ with hkl
  set vloadT : Tile .real [BLOCK_N, BLOCK_DMODEL] :=
      ⟨fun idx : TileIndex [BLOCK_N, BLOCK_DMODEL] =>
        if decide (c + idx.1.val < belv) && decide (idx.2.1.val < head_dim) then
          stt.readMemValue .real (Region.cast V)
            (kvf idx.1 * stride_vbs
              + s.pids 1 / 1 * stride_vh + idx.2.1.val * stride_vd)
        else some (0.0 : ℝ)⟩ with hvl
  set qkfullT : Tile .real [BLOCK_M, BLOCK_N] :=
    ⟨fun _ : TileIndex [BLOCK_M, BLOCK_N] => some (0 : ℝ)⟩ with hqkf
  set qkdotT : Tile .real [BLOCK_M, BLOCK_N] := Tile.bop NumericDType.real.add
    Broadcast.nil.consSame.consSame qkfullT (Tile.dot [] qT kloadT) with hqkdot
  set qkscaleT : Tile .real [BLOCK_M, BLOCK_N] := Tile.bop NumericDType.real.mul
    Broadcast.scalarR qkdotT (Tile.scalar (some sm_scale)) with hqksc
  set qkT : Tile .real [BLOCK_M, BLOCK_N] := Tile.select
      (⟨fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
          decide (c + idx.2.1.val ≤ gOM idx.1 + plenv)⟩ : Tile .bool [BLOCK_M, BLOCK_N])
      qkscaleT
      (⟨fun _ : TileIndex [BLOCK_M, BLOCK_N] => some (0.0 - 100000000.0 : ℝ)⟩
        : Tile .real [BLOCK_M, BLOCK_N]) with hqk
  obtain ⟨rmaxT, hrm⟩ : ∃ t : Tile .real [BLOCK_M], Tile.reduceMaxDrop
      (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length) qkT = some t := by
    unfold Tile.reduceMaxDrop
    rw [dif_pos (show 0 < TileShape.axisDim [BLOCK_M, BLOCK_N]
      (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length) from hBN)]
    exact ⟨_, rfl⟩
  set pT : Tile .real [BLOCK_M, BLOCK_N] := Tile.uop WithBot.realExp
    (Tile.bop NumericDType.real.sub Broadcast.nil.consR.consSame qkT
      (Tile.expandDim ⟨1, by simp⟩ rmaxT)) with hpT
  set lijT : Tile .real [BLOCK_M] := Tile.reduceSumDrop
    (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length) pT with hlij
  set miNewT : Tile .real [BLOCK_M] :=
    Tile.select (Tile.cop ComparableDType.real.gt Broadcast.nil.consSame mT rmaxT)
      mT rmaxT with hminew
  set alphaT : Tile .real [BLOCK_M] := Tile.uop WithBot.realExp
    (Tile.bop NumericDType.real.sub Broadcast.nil.consSame mT miNewT) with hal
  set betaT : Tile .real [BLOCK_M] := Tile.uop WithBot.realExp
    (Tile.bop NumericDType.real.sub Broadcast.nil.consSame rmaxT miNewT) with hbe
  set liNewT : Tile .real [BLOCK_M] := Tile.bop NumericDType.real.add Broadcast.nil.consSame
    (Tile.bop NumericDType.real.mul Broadcast.nil.consSame alphaT lT)
    (Tile.bop NumericDType.real.mul Broadcast.nil.consSame betaT lijT) with hlin
  set pscaleT : Tile .real [BLOCK_M] :=
    Tile.bop NumericDType.real.div Broadcast.nil.consSame betaT liNewT with hps
  set p2T : Tile .real [BLOCK_M, BLOCK_N] := Tile.bop NumericDType.real.mul
    Broadcast.nil.consR.consSame pT (Tile.expandDim ⟨1, by simp⟩ pscaleT) with hp2
  set accScale1T : Tile .real [BLOCK_M] := Tile.bop NumericDType.real.mul Broadcast.nil.consSame
    (Tile.bop NumericDType.real.div Broadcast.nil.consSame lT liNewT) alphaT with has1
  set accScale2T : Tile .real [BLOCK_M] := Tile.select
    (⟨fun idx : TileIndex [BLOCK_M] => decide (c ≤ gOM idx.1 + plenv)⟩ : Tile .bool [BLOCK_M])
    accScale1T (⟨fun _ : TileIndex [BLOCK_M] => some (1.0 : ℝ)⟩ : Tile .real [BLOCK_M]) with has2
  set acc1T : Tile .real [BLOCK_M, BLOCK_DMODEL] := Tile.bop NumericDType.real.mul
    Broadcast.nil.consR.consSame aT (Tile.expandDim ⟨1, by simp⟩ accScale2T) with hacc1
  unfold bloomLoopBodyG
  -- stmt 0: start_n = ref start_n
  refine bloomFwdIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (bloomFwdIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp (Op.ref .nat [] "start_n") _ = some (Tile.scalar c) from by
        rw [evalOp_ref, BlockState.setReg_same])) ?_
  -- stmt 1: kv_loc = masked `.nat` page-table gather  [gread-window bound]
  refine bloomFwdIO_walkCons ?_
    (bloomFwdIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp (Op.load .nat
          (MemAccess.region Req_to_tokens
            (Op.add .nat Broadcast.scalarL
              (Op.mul .nat Broadcast.nil (Op.constNat stride_req_b)
                (Op.ref .nat [] "cur_batch_req_idx"))
              (Op.mul .nat Broadcast.scalarL (Op.constNat stride_req_s)
                (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
                  (Op.ref .nat [BLOCK_N] "offs_n")))))
          (MaskOpt.maskOther
            (Op.lt .nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
                (Op.ref .nat [BLOCK_N] "offs_n"))
              (Op.ref .nat [] "block_end_loc"))
            (Op.broadcast (Op.constNat 0) [BLOCK_N]))) _
          = some (Tile.vec kvf) from by
        rw [bloom_kvloc_gather_evalG _ Req_to_tokens rqiv c belv stride_req_b stride_req_s
          (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
                reduceCtorEq]
              exact hrqi)
          (by rw [BlockState.setReg_same])
          (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
                reduceCtorEq]
              exact hn)
          (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
                reduceCtorEq]
              exact hbel)]
        refine congrArg some ?_
        ext idx
        simp only [hkvf, Tile.vec_data, BlockState.setReg_readMemValue])) ?_
  · simp only [Stmt.TraceSafeR, Op.SafeAtR, MaskOpt.ActiveR, MemAccess.ActiveAddressSafeR,
      memAccessActiveAddressSafeR]
    refine ⟨by simp [Op.SafeAtR.eq_def],
      ⟨by simp [Op.SafeAtR.eq_def], by simp [Op.SafeAtR.eq_def]⟩, ?_⟩
    intro offsets hoff idx hactive
    rw [bloomFwdIO_greadR_eval R _ rqiv c stride_req_b stride_req_s
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
            reduceCtorEq]
          exact hrqi)
      (by rw [BlockState.setReg_same])
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
            reduceCtorEq]
          exact hn)] at hoff
    obtain rfl := Option.some.inj hoff
    obtain ⟨masks, hmask, hactl⟩ := hactive
    rw [bloomFwdIO_gmaskR_eval R _ c belv
      (by rw [BlockState.setReg_same])
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
            reduceCtorEq]
          exact hn)
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
            reduceCtorEq]
          exact hbel)] at hmask
    obtain rfl := Option.some.inj hmask
    obtain ⟨jL, pu⟩ := idx
    have hcol : c + jL.val < belv := by simpa using hactl
    show stride_req_b * rqiv + stride_req_s * (c + jL.val)
      < bounds (Region.cast Req_to_tokens)
    exact hbG jL hcol
  -- stmt 2: off_k (gather-addressed)
  refine bloomFwdIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (bloomFwdIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (bloom_offk_gather_evalG _ (s.pids 1 / 1) kvf stride_kbs stride_kh stride_kd
        (by rw [BlockState.setReg_same])
        (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
              reduceCtorEq]
            exact hckvh)
        (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
              reduceCtorEq]
            exact hd))) ?_
  -- stmt 3: k = masked load  [read2-window bound, liveness ∧ channel]
  refine bloomFwdIO_walkCons ?_
    (bloomFwdIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp (Op.load .real
          (MemAccess.region K (Op.ref .nat [BLOCK_DMODEL, BLOCK_N] "off_k"))
          (MaskOpt.maskOther
            (Op.boolAnd Broadcast.nil.consR.consL
              (Op.lt .nat Broadcast.scalarR
                (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
                  (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_N] "offs_n")))
                (Op.ref .nat [] "block_end_loc"))
              (Op.lt .nat Broadcast.scalarR
                (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_DMODEL] "offs_d"))
                (Op.constNat head_dim)))
            ((Op.const (0.0 : ℝ)).broadcast [BLOCK_DMODEL, BLOCK_N]))) _
          = some kloadT from
        bloomFwdIO_maskedLoad_eval K (Op.ref .nat [BLOCK_DMODEL, BLOCK_N] "off_k") _ _ _ stt
          (by funext rg o; simp only [BlockState.setReg_mem])
          (⟨fun idx : TileIndex [BLOCK_DMODEL, BLOCK_N] =>
              kvf idx.2.1 * stride_kbs
                + s.pids 1 / 1 * stride_kh + idx.1.val * stride_kd⟩
              : Tile .nat [BLOCK_DMODEL, BLOCK_N])
          (⟨fun idx : TileIndex [BLOCK_DMODEL, BLOCK_N] =>
              decide (c + idx.2.1.val < belv) && decide (idx.1.val < head_dim)⟩
              : Tile .bool [BLOCK_DMODEL, BLOCK_N])
          (⟨fun _ : TileIndex [BLOCK_DMODEL, BLOCK_N] => some (0.0 : ℝ)⟩
              : Tile .real [BLOCK_DMODEL, BLOCK_N])
          (by rw [evalOp_ref, BlockState.setReg_same])
          (bloomKMask_evalG _ c belv head_dim
            (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
                  reduceCtorEq]
                rw [BlockState.setReg_same])
            (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
                  reduceCtorEq]
                exact hn)
            (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
                  reduceCtorEq]
                exact hd)
            (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
                  reduceCtorEq]
                exact hbel))
          (bloomFwdIO_zeroBroadcast_eval _ _))) ?_
  · simp only [Stmt.TraceSafeR, Op.SafeAtR, MaskOpt.ActiveR, MemAccess.ActiveAddressSafeR,
      memAccessActiveAddressSafeR]
    refine ⟨by simp [Op.SafeAtR.eq_def],
      ⟨by simp [Op.SafeAtR.eq_def], by simp [Op.SafeAtR.eq_def]⟩, ?_⟩
    intro offsets hoff idx hactive
    rw [evalOpR_ref] at hoff
    simp only [BlockState.setReg_same] at hoff
    obtain rfl := Option.some.inj hoff
    obtain ⟨masks, hmask, hactl⟩ := hactive
    rw [bloomFwdIO_kmaskR_eval R _ c belv head_dim
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
            reduceCtorEq]
          rw [BlockState.setReg_same])
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
            reduceCtorEq]
          exact hn)
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
            reduceCtorEq]
          exact hd)
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
            reduceCtorEq]
          exact hbel)] at hmask
    obtain rfl := Option.some.inj hmask
    obtain ⟨e, jL, pu⟩ := idx
    have hcond : c + jL.val < belv ∧ e.val < head_dim := by simpa using hactl
    have hb := hbK e jL hcond.1 hcond.2
    show kvf jL * stride_kbs
        + s.pids 1 / 1 * stride_kh + e.val * stride_kd < bounds (Region.cast K)
    rw [Nat.div_one, hkvfs jL hcond.1]
    exact hb
  -- stmt 4: qk = tl.zeros
  refine bloomFwdIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (bloomFwdIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp (Op.full [BLOCK_M, BLOCK_N] (Op.const (0 : ℝ))) _ = some qkfullT from
        bloomQkFull_eval _)) ?_
  -- stmt 5: qk += dot(q, k)
  refine bloomFwdIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (bloomFwdIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (bloomQkAddDot_eval _ qkfullT qT kloadT
        (by rw [BlockState.setReg_same])
        (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
              reduceCtorEq]
            exact hq)
        (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
              reduceCtorEq]
            rw [BlockState.setReg_same]))) ?_
  -- stmt 6: qk *= sm_scale
  refine bloomFwdIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (bloomFwdIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (bloomQkScale_eval _ sm_scale qkdotT (by rw [BlockState.setReg_same]))) ?_
  -- stmt 7: qk = where(causal, qk, -1e8)
  refine bloomFwdIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (bloomFwdIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (bloomQkWhere_eval _ BLOCK_M BLOCK_N plenv c gOM qkscaleT
        (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
              reduceCtorEq]
            exact hom)
        (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
              reduceCtorEq]
            exact hn)
        (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
              reduceCtorEq]
            exact hplen)
        (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
              reduceCtorEq]
            rw [BlockState.setReg_same])
        (by rw [BlockState.setReg_same]))) ?_
  -- stmt 8: m_ij = tl.max(qk, 1)
  refine bloomFwdIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (bloomFwdIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (bloomMij_eval _ qkT rmaxT (by rw [BlockState.setReg_same]) hrm)) ?_
  -- stmt 9: p = tl.exp(qk − m_ij[:, None])
  refine bloomFwdIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (bloomFwdIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (bloomP_eval _ qkT rmaxT
        (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
              reduceCtorEq]
            rw [BlockState.setReg_same])
        (by rw [BlockState.setReg_same]))) ?_
  -- stmt 10: l_ij = tl.sum(p, 1)
  refine bloomFwdIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (bloomFwdIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (bloomLij_eval _ pT (by rw [BlockState.setReg_same]))) ?_
  -- stmt 11: m_i_new = tl.maximum(m_i, m_ij)
  refine bloomFwdIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (bloomFwdIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (bloomMiNew_eval _ mT rmaxT
        (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
              reduceCtorEq]
            exact hmi)
        (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
              reduceCtorEq]
            rw [BlockState.setReg_same]))) ?_
  -- stmt 12: alpha = tl.exp(m_i − m_i_new)
  refine bloomFwdIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (bloomFwdIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (bloomExpSub_eval _ "m_i" "m_i_new" mT miNewT
        (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
              reduceCtorEq]
            exact hmi)
        (by rw [BlockState.setReg_same]))) ?_
  -- stmt 13: beta = tl.exp(m_ij − m_i_new)
  refine bloomFwdIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (bloomFwdIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (bloomExpSub_eval _ "m_ij" "m_i_new" rmaxT miNewT
        (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
              reduceCtorEq]
            rw [BlockState.setReg_same])
        (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
              reduceCtorEq]
            rw [BlockState.setReg_same]))) ?_
  -- stmt 14: l_i_new = alpha·l_i + beta·l_ij
  refine bloomFwdIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (bloomFwdIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (bloomLiNew_eval _ alphaT lT betaT lijT
        (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
              reduceCtorEq]
            rw [BlockState.setReg_same])
        (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
              reduceCtorEq]
            exact hli)
        (by rw [BlockState.setReg_same])
        (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
              reduceCtorEq]
            rw [BlockState.setReg_same]))) ?_
  -- stmt 15: p_scale = beta / l_i_new
  refine bloomFwdIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (bloomFwdIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (bloomPscale_eval _ betaT liNewT
        (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
              reduceCtorEq]
            rw [BlockState.setReg_same])
        (by rw [BlockState.setReg_same]))) ?_
  -- stmt 16: p = p · p_scale[:, None]
  refine bloomFwdIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (bloomFwdIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (bloomP2_eval _ pT pscaleT
        (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
              reduceCtorEq]
            rw [BlockState.setReg_same])
        (by rw [BlockState.setReg_same]))) ?_
  -- stmt 17: acc_scale = (l_i / l_i_new)·alpha
  refine bloomFwdIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (bloomFwdIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (bloomAccScale1_eval _ lT liNewT alphaT
        (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
              reduceCtorEq]
            exact hli)
        (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
              reduceCtorEq]
            rw [BlockState.setReg_same])
        (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
              reduceCtorEq]
            rw [BlockState.setReg_same]))) ?_
  -- stmt 18: acc_scale = where(offs_m + plen ≥ start_n, acc_scale, 1.0)
  refine bloomFwdIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (bloomFwdIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (bloomAccScale2_eval _ BLOCK_M plenv c gOM accScale1T
        (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
              reduceCtorEq]
            exact hom)
        (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
              reduceCtorEq]
            exact hplen)
        (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
              reduceCtorEq]
            rw [BlockState.setReg_same])
        (by rw [BlockState.setReg_same]))) ?_
  -- stmt 19: acc = acc · acc_scale[:, None]
  refine bloomFwdIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (bloomFwdIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (bloomAcc1_eval _ aT accScale2T
        (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
              reduceCtorEq]
            exact hacc)
        (by rw [BlockState.setReg_same]))) ?_
  -- stmt 20: off_v
  refine bloomFwdIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (bloomFwdIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (bloom_offv_gather_evalG _ (s.pids 1 / 1) kvf stride_vbs stride_vh stride_vd
        (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
              reduceCtorEq]
            rw [BlockState.setReg_same])
        (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
              reduceCtorEq]
            exact hckvh)
        (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
              reduceCtorEq]
            exact hd))) ?_
  -- stmt 21: v = masked load  [read3-window bound, liveness ∧ channel]
  refine bloomFwdIO_walkCons ?_
    (bloomFwdIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp (Op.load .real
          (MemAccess.region V (Op.ref .nat [BLOCK_N, BLOCK_DMODEL] "off_v"))
          (MaskOpt.maskOther
            (Op.boolAnd Broadcast.nil.consL.consR
              (Op.lt .nat Broadcast.scalarR
                (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
                  (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_N] "offs_n")))
                (Op.ref .nat [] "block_end_loc"))
              (Op.lt .nat Broadcast.scalarR
                (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_DMODEL] "offs_d"))
                (Op.constNat head_dim)))
            ((Op.const (0.0 : ℝ)).broadcast [BLOCK_N, BLOCK_DMODEL]))) _
          = some vloadT from
        bloomFwdIO_maskedLoad_eval V (Op.ref .nat [BLOCK_N, BLOCK_DMODEL] "off_v") _ _ _ stt
          (by funext rg o; simp only [BlockState.setReg_mem])
          (⟨fun idx : TileIndex [BLOCK_N, BLOCK_DMODEL] =>
              kvf idx.1 * stride_vbs
                + s.pids 1 / 1 * stride_vh + idx.2.1.val * stride_vd⟩
              : Tile .nat [BLOCK_N, BLOCK_DMODEL])
          (⟨fun idx : TileIndex [BLOCK_N, BLOCK_DMODEL] =>
              decide (c + idx.1.val < belv) && decide (idx.2.1.val < head_dim)⟩
              : Tile .bool [BLOCK_N, BLOCK_DMODEL])
          (⟨fun _ : TileIndex [BLOCK_N, BLOCK_DMODEL] => some (0.0 : ℝ)⟩
              : Tile .real [BLOCK_N, BLOCK_DMODEL])
          (by rw [evalOp_ref, BlockState.setReg_same])
          (bloomVMask_evalG _ c belv head_dim
            (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
                  reduceCtorEq]
                rw [BlockState.setReg_same])
            (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
                  reduceCtorEq]
                exact hn)
            (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
                  reduceCtorEq]
                exact hd)
            (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
                  reduceCtorEq]
                exact hbel))
          (bloomFwdIO_zeroBroadcast_eval _ _))) ?_
  · simp only [Stmt.TraceSafeR, Op.SafeAtR, MaskOpt.ActiveR, MemAccess.ActiveAddressSafeR,
      memAccessActiveAddressSafeR]
    refine ⟨by simp [Op.SafeAtR.eq_def],
      ⟨by simp [Op.SafeAtR.eq_def], by simp [Op.SafeAtR.eq_def]⟩, ?_⟩
    intro offsets hoff idx hactive
    rw [evalOpR_ref] at hoff
    simp only [BlockState.setReg_same] at hoff
    obtain rfl := Option.some.inj hoff
    obtain ⟨masks, hmask, hactl⟩ := hactive
    rw [bloomFwdIO_vmaskR_eval R _ c belv head_dim
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
            reduceCtorEq]
          rw [BlockState.setReg_same])
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
            reduceCtorEq]
          exact hn)
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
            reduceCtorEq]
          exact hd)
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
            reduceCtorEq]
          exact hbel)] at hmask
    obtain rfl := Option.some.inj hmask
    obtain ⟨jL, e, pu⟩ := idx
    have hcond : c + jL.val < belv ∧ e.val < head_dim := by simpa using hactl
    have hb := hbV jL e hcond.1 hcond.2
    show kvf jL * stride_vbs
        + s.pids 1 / 1 * stride_vh + e.val * stride_vd < bounds (Region.cast V)
    rw [Nat.div_one, hkvfs jL hcond.1]
    exact hb
  -- stmt 22: p = ref p (erased cast)
  refine bloomFwdIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (bloomFwdIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp (Op.ref .real [BLOCK_M, BLOCK_N] "p") _ = some p2T from by
        rw [evalOp_ref]
        simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
          reduceCtorEq]
        rw [BlockState.setReg_same])) ?_
  -- stmt 23: acc += dot(p, v)
  refine bloomFwdIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (bloomFwdIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (bloomAcc2_eval _ acc1T p2T vloadT
        (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
              reduceCtorEq]
            rw [BlockState.setReg_same])
        (by rw [BlockState.setReg_same])
        (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
              reduceCtorEq]
            rw [BlockState.setReg_same]))) ?_
  -- stmt 24: l_i = l_i_new
  refine bloomFwdIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (bloomFwdIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp (Op.ref .real [BLOCK_M] "l_i_new") _ = some liNewT from by
        rw [evalOp_ref]
        simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
          reduceCtorEq]
        rw [BlockState.setReg_same])) ?_
  -- stmt 25: m_i = m_i_new
  refine bloomFwdIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (bloomFwdIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp (Op.ref .real [BLOCK_M] "m_i_new") _ = some miNewT from by
        rw [evalOp_ref]
        simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
          reduceCtorEq]
        rw [BlockState.setReg_same])) ?_
  refine ⟨Stmt.TraceSafeListR.nil_intro, _, stepStmtsR_nil R _, ?_⟩
  refine ⟨?_, ?_,
    (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
          reduceCtorEq]
        exact hcb),
    (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
          reduceCtorEq]
        exact hch),
    (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
          reduceCtorEq]
        exact hckvh),
    (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
          reduceCtorEq]
        exact hplen),
    (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
          reduceCtorEq]
        exact hcbsi),
    (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
          reduceCtorEq]
        exact hseq),
    (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
          reduceCtorEq]
        exact hrqi),
    (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
          reduceCtorEq]
        exact hbmk),
    (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
          reduceCtorEq]
        exact hbel),
    (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
          reduceCtorEq]
        exact hn),
    (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
          reduceCtorEq]
        exact hd),
    (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
          reduceCtorEq]
        exact hom),
    ⟨qT, by
      simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
        reduceCtorEq]
      exact hq⟩,
    ⟨miNewT, by rw [BlockState.setReg_same]⟩,
    ⟨liNewT, by
      simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
        reduceCtorEq]
      rw [BlockState.setReg_same]⟩,
    ⟨_, by
      simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
        reduceCtorEq]
      rw [BlockState.setReg_same]⟩⟩
  · funext rg o
    simp only [BlockState.setReg_mem]
    rw [hmem]
  · simp only [BlockState.setReg_pids]
    exact hpids

set_option maxHeartbeats 8000000 in
set_option maxRecDepth 8000 in
/-- **The `TraceSafeR` walk for the whole kernel** from an arbitrary launch
state under the `pre`-legal window bounds (`raw B_Seqlen slot ≤ NT·BLOCK_M`):
weak preLoop walk, the streaming `forRangeDyn` driven by
`Stmt.forRangeTraceSafeR_inv` over `bloomFwdIOSafeInvW` (every live step
`cc < bm·bel ≤ NT·BLOCK_M ≤ BLOCK_N·T` indexes `Fin T`), and the terminal
masked store bounded by the `write` window. -/
private theorem bloomFwdIO_traceSafeR (R : RoundingModel) (bounds : RegionBounds)
    (Q K V Out : RegionName)
    (B_Start_Loc B_Seqlen Req_to_tokens B_req_idx b_prompt_cache_len : Region .nat)
    (sm_scale : ℝ)
    (stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
      head_dim BLOCK_DMODEL BLOCK_M BLOCK_N NT : Nat) (hBN : 0 < BLOCK_N)
    (s : BlockState)
    (hm2NT : s.readMemValue .nat (Region.cast B_Seqlen) (s.pids 0) ≤ NT * BLOCK_M)
    (hbPC : s.pids 0 < bounds (Region.cast b_prompt_cache_len))
    (hbSL : s.pids 0 < bounds (Region.cast B_Start_Loc))
    (hbSQ : s.pids 0 < bounds (Region.cast B_Seqlen))
    (hbRQ : s.pids 0 < bounds (Region.cast B_req_idx))
    (hbQ : ∀ (i : Fin BLOCK_M) (e : Fin BLOCK_DMODEL),
      s.pids 2 * BLOCK_M + i.val < s.readMemValue .nat (Region.cast B_Seqlen) (s.pids 0)
          - s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 0) →
      e.val < head_dim →
      (s.readMemValue .nat (Region.cast B_Start_Loc) (s.pids 0)
          + (s.pids 2 * BLOCK_M + i.val)) * stride_qbs
        + s.pids 1 * stride_qh + e.val * stride_qd < bounds Q)
    (hbG : ∀ (t : Fin (bloomFwdIOT NT BLOCK_M BLOCK_N)) (jL : Fin BLOCK_N),
      t.val * BLOCK_N + jL.val < bloomFwdIOBel BLOCK_M (s.pids 2)
          (s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 0))
          (s.readMemValue .nat (Region.cast B_Seqlen) (s.pids 0)) →
      stride_req_b * s.readMemValue .nat (Region.cast B_req_idx) (s.pids 0)
          + stride_req_s * (t.val * BLOCK_N + jL.val) < bounds Req_to_tokens)
    (hbK : ∀ (t : Fin (bloomFwdIOT NT BLOCK_M BLOCK_N)) (e : Fin BLOCK_DMODEL)
        (jL : Fin BLOCK_N),
      t.val * BLOCK_N + jL.val < bloomFwdIOBel BLOCK_M (s.pids 2)
          (s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 0))
          (s.readMemValue .nat (Region.cast B_Seqlen) (s.pids 0)) →
      e.val < head_dim →
      s.readMemValue .nat (Region.cast Req_to_tokens)
            (stride_req_b * s.readMemValue .nat (Region.cast B_req_idx) (s.pids 0)
              + stride_req_s * (t.val * BLOCK_N + jL.val)) * stride_kbs
        + s.pids 1 * stride_kh + e.val * stride_kd < bounds K)
    (hbV : ∀ (t : Fin (bloomFwdIOT NT BLOCK_M BLOCK_N)) (jL : Fin BLOCK_N)
        (e : Fin BLOCK_DMODEL),
      t.val * BLOCK_N + jL.val < bloomFwdIOBel BLOCK_M (s.pids 2)
          (s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 0))
          (s.readMemValue .nat (Region.cast B_Seqlen) (s.pids 0)) →
      e.val < head_dim →
      s.readMemValue .nat (Region.cast Req_to_tokens)
            (stride_req_b * s.readMemValue .nat (Region.cast B_req_idx) (s.pids 0)
              + stride_req_s * (t.val * BLOCK_N + jL.val)) * stride_vbs
        + s.pids 1 * stride_vh + e.val * stride_vd < bounds V)
    (hbO : ∀ (i : Fin BLOCK_M) (e : Fin BLOCK_DMODEL),
      s.pids 2 * BLOCK_M + i.val < s.readMemValue .nat (Region.cast B_Seqlen) (s.pids 0)
          - s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 0) →
      e.val < head_dim →
      (s.readMemValue .nat (Region.cast B_Start_Loc) (s.pids 0)
          + (s.pids 2 * BLOCK_M + i.val)) * stride_obs
        + s.pids 1 * stride_oh + e.val * stride_od < bounds Out) :
    ((context_attn_bloom_fwd_kernel_surface Q K V sm_scale B_Start_Loc B_Seqlen Out
      Req_to_tokens B_req_idx b_prompt_cache_len
      stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
      stride_req_b stride_req_s 1 head_dim BLOCK_M BLOCK_DMODEL
      BLOCK_N).toAlgKernel).TraceSafeR R bounds s := by
  set plenv := s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 0) with hplenv
  set rawv := s.readMemValue .nat (Region.cast B_Seqlen) (s.pids 0) with hrawv
  set bmv := (if BLOCK_M * s.pids 2 < rawv - plenv then 1 else 0) with hbmv
  set belv := bloomFwdIOBel BLOCK_M (s.pids 2) plenv rawv with hbelv
  have hstople : bmv * belv ≤ NT * BLOCK_M :=
    bloomFwdIOBel_le BLOCK_M (s.pids 2) plenv rawv NT hm2NT
  have hTle : NT * BLOCK_M ≤ BLOCK_N * bloomFwdIOT NT BLOCK_M BLOCK_N :=
    bloomFwdIOT_mul_le NT BLOCK_M BLOCK_N hBN
  -- shared per-iteration body handler
  have hbody : ∀ (cc : Nat) (st : BlockState),
      cc < bmv * belv →
      bloomFwdIOSafeInvW B_Start_Loc B_Seqlen B_req_idx b_prompt_cache_len
          BLOCK_DMODEL BLOCK_M BLOCK_N s st ∧ cc % BLOCK_N = 0 →
      Stmt.TraceSafeListR R bounds
          (bloomLoopBodyG Q K V Req_to_tokens B_req_idx sm_scale stride_req_b stride_req_s
            stride_kbs stride_kh stride_kd stride_vbs stride_vh stride_vd
            head_dim BLOCK_DMODEL BLOCK_M BLOCK_N)
          (st.setReg "start_n" .nat [] (Tile.scalar cc))
        ∧ ∃ s', stepStmtsR R
              (bloomLoopBodyG Q K V Req_to_tokens B_req_idx sm_scale stride_req_b stride_req_s
                stride_kbs stride_kh stride_kd stride_vbs stride_vh stride_vd
                head_dim BLOCK_DMODEL BLOCK_M BLOCK_N)
              (st.setReg "start_n" .nat [] (Tile.scalar cc)) = some s'
            ∧ (bloomFwdIOSafeInvW B_Start_Loc B_Seqlen B_req_idx b_prompt_cache_len
                BLOCK_DMODEL BLOCK_M BLOCK_N s s'
              ∧ (cc + BLOCK_N) % BLOCK_N = 0) := by
    intro cc st hcc hPP
    obtain ⟨hPinv, hPmod⟩ := hPP
    have hceq : cc / BLOCK_N * BLOCK_N = cc := Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hPmod)
    have hcT : cc / BLOCK_N < bloomFwdIOT NT BLOCK_M BLOCK_N := by
      refine (Nat.div_lt_iff_lt_mul hBN).mpr ?_
      calc cc < bmv * belv := hcc
        _ ≤ NT * BLOCK_M := hstople
        _ ≤ BLOCK_N * bloomFwdIOT NT BLOCK_M BLOCK_N := hTle
        _ = bloomFwdIOT NT BLOCK_M BLOCK_N * BLOCK_N := Nat.mul_comm _ _
    obtain ⟨hsafeB, s', hrunB, hInvB⟩ :=
      bloomFwdIO_bodyW R bounds Q K V Req_to_tokens B_req_idx B_Start_Loc B_Seqlen B_req_idx
        b_prompt_cache_len sm_scale
        stride_req_b stride_req_s stride_kbs stride_kh stride_kd stride_vbs stride_vh stride_vd
        head_dim BLOCK_DMODEL BLOCK_M BLOCK_N hBN s st cc hPinv
        (fun jL hc => by
          have hb := hbG ⟨cc / BLOCK_N, hcT⟩ jL (by rw [hceq]; exact hc)
          rwa [hceq] at hb)
        (fun e jL hc hhd => by
          have hb := hbK ⟨cc / BLOCK_N, hcT⟩ e jL (by rw [hceq]; exact hc) hhd
          rwa [hceq] at hb)
        (fun jL e hc hhd => by
          have hb := hbV ⟨cc / BLOCK_N, hcT⟩ jL e (by rw [hceq]; exact hc) hhd
          rwa [hceq] at hb)
    exact ⟨hsafeB, s', hrunB, hInvB, by rw [Nat.add_mod_right]; exact hPmod⟩
  unfold Kernel.TraceSafeR
  rw [bloomBody_splitG]
  obtain ⟨hsafePre, s0, hrun0, hInv0⟩ :=
    bloomFwdIO_preLoopW R bounds Q B_Start_Loc B_Seqlen B_req_idx b_prompt_cache_len
      stride_qbs stride_qh stride_qd head_dim BLOCK_DMODEL BLOCK_M BLOCK_N s
      hbPC hbSL hbSQ hbRQ hbQ
  have hInv0' := hInv0
  obtain ⟨hmem0, hpids0, hcb0, hch0, hckvh0, hplen0, hcbsi0, hseq0, hrqi0, hbmk0, hbel0,
    hn0, hd0, hom0, hqE, hmiE, hliE, haccE⟩ := hInv0
  have hstopExact : evalOp (Op.mul .nat Broadcast.nil (Op.ref .nat [] "block_mask")
      (Op.ref .nat [] "block_end_loc")) s0 = some (Tile.scalar (bmv * belv)) := by
    rw [evalOp_mul, evalOp_ref, evalOp_ref, hbmk0, hbel0]
    simp only [Option.bind_eq_bind, Option.bind_some]
    rfl
  refine Stmt.TraceSafeListR.append_intro _ _ hsafePre ?_
  intro s1 hs1
  rw [hrun0] at hs1
  obtain rfl := Option.some.inj hs1
  refine Stmt.TraceSafeListR.cons_intro ?_ (fun s2 hs2 => ?_)
  · -- TraceSafeR of the forRangeDyn itself
    simp only [Stmt.TraceSafeR]
    refine ⟨by simp [Op.SafeAtR.eq_def], by simp [Op.SafeAtR.eq_def],
      by simp [Op.SafeAtR.eq_def], ?_⟩
    rw [bloomFwdIO_evalOpR_constNat, bloomFwdIO_evalOpR_constNat,
      bloomFwdIO_stopOpR_castFree, hstopExact]
    refine Stmt.forRangeTraceSafeR_inv R bounds "start_n" _ _
      (bloomLoopBodyG Q K V Req_to_tokens B_req_idx sm_scale stride_req_b stride_req_s
        stride_kbs stride_kh stride_kd stride_vbs stride_vh stride_vd
        head_dim BLOCK_DMODEL BLOCK_M BLOCK_N)
      (fun i st => bloomFwdIOSafeInvW B_Start_Loc B_Seqlen B_req_idx b_prompt_cache_len
        BLOCK_DMODEL BLOCK_M BLOCK_N s st ∧ i % BLOCK_N = 0)
      ?_ _ s0 ⟨hInv0', Nat.zero_mod BLOCK_N⟩
    intro cc st hcc hPP
    obtain ⟨hsafeB, s', hrunB, hInvB⟩ := hbody cc st hcc hPP
    exact ⟨hsafeB, s', hrunB, hInvB⟩
  · -- the loop's actual successor, then the postLoop
    obtain ⟨finalC, sL, hLoopExact, hfin, hPL⟩ :=
      forRangeDyn_inv (idx := "start_n")
        (startOp := Op.constNat 0)
        (stopOp := Op.mul .nat Broadcast.nil (Op.ref .nat [] "block_mask")
          (Op.ref .nat [] "block_end_loc"))
        (stepOp := Op.constNat BLOCK_N)
        (P := fun i st => bloomFwdIOSafeInvW B_Start_Loc B_Seqlen B_req_idx b_prompt_cache_len
          BLOCK_DMODEL BLOCK_M BLOCK_N s st ∧ i % BLOCK_N = 0)
        (s_init := s0)
        (evalOp_constNat 0 s0) hstopExact (evalOp_constNat BLOCK_N s0)
        hBN.ne'
        ⟨hInv0', Nat.zero_mod BLOCK_N⟩
        (fun i st hi hP => by
          obtain ⟨hsafeB, s', hrunB, hInvB⟩ := hbody i st hi hP
          exact ⟨s', by
            rw [← bloomFwdIO_body_castFree R Q K V Req_to_tokens B_req_idx sm_scale
              stride_req_b stride_req_s stride_kbs stride_kh stride_kd
              stride_vbs stride_vh stride_vd head_dim BLOCK_DMODEL BLOCK_M BLOCK_N]
            exact hrunB, hInvB⟩)
    rw [bloomFwdIO_dyn_castFree R Q K V Req_to_tokens B_req_idx sm_scale
      stride_req_b stride_req_s stride_kbs stride_kh stride_kd stride_vbs stride_vh stride_vd
      head_dim BLOCK_DMODEL BLOCK_M BLOCK_N s0, hLoopExact] at hs2
    obtain rfl := Option.some.inj hs2
    obtain ⟨⟨hmemL, hpidsL, hcbL, hchL, hckvhL, hplenL, hcbsiL, hseqL, hrqiL, hbmkL, hbelL,
      hnL, hdL, homL, hqEL, hmiEL, hliEL, haccEL⟩, hmodL⟩ := hPL
    -- postLoop: `off_o`, `out_ptrs`, the masked terminal store (no post-divide)
    refine Stmt.TraceSafeListR.cons_intro
      (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s4 hs4 => ?_)
    obtain ⟨v4, hv4, rfl⟩ := stepStmtR_assign_inv hs4
    rw [bloomFwdIO_offoR_castFree R stride_obs stride_oh stride_od,
      bloomFwdIO_offo_eval _ (fun r : Fin BLOCK_M => s.pids 2 * BLOCK_M + r.val)
        (s.readMemValue .nat (Region.cast B_Start_Loc) (s.pids 0)) (s.pids 1)
        stride_obs stride_oh stride_od
        hcbsiL hchL homL hdL] at hv4
    obtain rfl := Option.some.inj hv4
    refine Stmt.TraceSafeListR.cons_intro
      (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s5 hs5 => ?_)
    obtain ⟨v5, hv5, rfl⟩ := stepStmtR_assign_inv hs5
    rw [bloomFwdIO_outptrR_castFree R Out BLOCK_M BLOCK_DMODEL] at hv5
    simp only [evalOp, evalOp_ref, BlockState.setReg_same, Option.bind] at hv5
    obtain rfl := Option.some.inj hv5
    refine Stmt.TraceSafeListR.cons_intro ?_ (fun _ _ => Stmt.TraceSafeListR.nil_intro)
    simp only [Stmt.TraceSafeR, MemAccess.SafeAtR, MaskOpt.SafeAtR, MaskOpt.ActiveR,
      MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
    refine ⟨by simp [MemAccess.SafeAtR, Op.SafeAtR.eq_def],
      by simp [Op.SafeAtR.eq_def], by simp [MaskOpt.SafeAtR, Op.SafeAtR.eq_def], ?_⟩
    intro ptrs hptrs idx hactive
    rw [evalOpR_ref] at hptrs
    simp only [BlockState.setReg_same] at hptrs
    obtain rfl := Option.some.inj hptrs
    obtain ⟨masks, hmask, hactl⟩ := hactive
    rw [bloomFwdIO_qmaskR_eval R _ (fun r : Fin BLOCK_M => s.pids 2 * BLOCK_M + r.val)
      (rawv - plenv) head_dim
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
            reduceCtorEq]
          exact homL)
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
            reduceCtorEq]
          exact hdL)
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
            reduceCtorEq]
          exact hseqL)] at hmask
    obtain rfl := Option.some.inj hmask
    obtain ⟨i2, e2, pu⟩ := idx
    have hcond : s.pids 2 * BLOCK_M + i2.val < rawv - plenv ∧ e2.val < head_dim := by
      simpa using hactl
    have hb := hbO i2 e2 hcond.1 hcond.2
    simpa [Tile.ptrAdd_data, Tile.scalar_data, Broadcast.leftIndex, Broadcast.rightIndex,
      Region.cast_id] using hb

/-! ## The rounded Hoare triple (`hrun`): framed exact run

`hrun` rides the exact `bloomPreLoopG_eval` → `forRangeDyn_inv` →
`bloomPostLoopG_eval` stack unchanged (everything is cast-free, so `execR R`
collapses onto it verbatim); the only new obligation the skin adds over the
existing headline is the **memory frame**, recovered by replaying the
deterministic 3-statement postLoop and framing its masked scatter. -/

/-- A masked `writeMem` scatter `foldl` leaves every cell not hit by an
active lane untouched. Private copy of the llama/nopad helper. -/
private theorem bloomFwdIO_foldl_writeMem_frame_masked {α : Type} (region : RegionName)
    (offFn : α → Nat) (valFn : α → ℝ) (P : α → Prop) [DecidablePred P] :
    ∀ (l : List α) (st : BlockState) (r : RegionName) (o : Nat),
      (r = region → ∀ k ∈ l, P k → offFn k ≠ o) →
      ((l.foldl (fun acc k =>
          if P k then acc.writeMem region (offFn k) (valFn k) else acc) st).mem r o
        = st.mem r o)
  | [], _, _, _, _ => rfl
  | k :: rest, st, r, o, h => by
      rw [List.foldl_cons]
      by_cases hPk : P k
      · rw [if_pos hPk,
          bloomFwdIO_foldl_writeMem_frame_masked region offFn valFn P rest _ r o
            (fun hr k' hk' hPk' => h hr k' (List.mem_cons_of_mem _ hk') hPk'),
          BlockState.writeMem_mem]
        rw [if_neg (fun hro => h hro.1 k List.mem_cons_self hPk hro.2.symm)]
      · rw [if_neg hPk]
        exact bloomFwdIO_foldl_writeMem_frame_masked region offFn valFn P rest _ r o
          (fun hr k' hk' hPk' => h hr k' (List.mem_cons_of_mem _ hk') hPk')

set_option maxHeartbeats 3200000 in
set_option maxRecDepth 8000 in
/-- **PostLoop frame**: every cell outside the terminal store's active window
is untouched by the postLoop (replay of `bloomPostLoopG_eval`'s deterministic
3-statement chain, framed). -/
private theorem bloomFwdIO_postLoop_frame
    (Q K V Out B_Start_Loc B_Seqlen Req_to_tokens B_req_idx B_Prompt_Cache_Len : RegionName)
    (s0 : BlockState) (sm_scale : ℝ)
    (stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
      head_dim BLOCK_DMODEL BLOCK_N BLOCK_M : Nat) (hD : 0 < BLOCK_DMODEL)
    (S bel : Nat) (c : Nat) (sL sP : BlockState) (hSc : S = c * BLOCK_N)
    (hinv : bloomInvariantG Q K V Out B_Start_Loc B_Seqlen Req_to_tokens B_req_idx
        B_Prompt_Cache_Len s0 sm_scale
        stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
        stride_vbs stride_vh stride_vd head_dim BLOCK_DMODEL BLOCK_N BLOCK_M hD S bel c sL)
    (hpost : stepStmts (bloomPostLoopG Out stride_obs stride_oh stride_od head_dim
        BLOCK_DMODEL BLOCK_M) sL = some sP) :
    ∀ r o,
      (r = Out → ∀ idx : TileIndex [BLOCK_M, BLOCK_DMODEL],
        active s0 B_Seqlen B_Prompt_Cache_Len head_dim BLOCK_M idx →
        o ≠ outOffset s0 B_Start_Loc stride_obs stride_oh stride_od BLOCK_M idx) →
      sP.mem r o = sL.mem r o := by
  subst hSc
  set plen := promptLen s0 B_Prompt_Cache_Len with hplend
  set sl := seqLen s0 B_Seqlen B_Prompt_Cache_Len with hsld
  set g := fun (i : Fin BLOCK_M) (d : Nat) =>
    bloomGG s0 Q K V B_Start_Loc B_Seqlen B_Prompt_Cache_Len Req_to_tokens B_req_idx sm_scale
      stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd head_dim BLOCK_DMODEL BLOCK_M (c * BLOCK_N) bel i d with hgd
  simp only [bloomInvariantG] at hinv
  obtain ⟨hpids, hmem, hundef, hcb, hckvh, hch, hplen, hsl, hbel, hcbsi, hrqi, hom, hon, hod,
      hq, hmi, hli, hacc, hcle⟩ := hinv
  set accTile : Tile .real [BLOCK_M, BLOCK_DMODEL] :=
    ⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
      some (gAccN (c * BLOCK_N) BLOCK_N (s0.pids 2 * BLOCK_M + idx.1.val + plen)
        (g idx.1 idx.2.1.val) c)⟩ with haccTile
  have haccref0 : sL.regs .real [BLOCK_M, BLOCK_DMODEL] "acc" = some accTile := by rw [hacc]
  unfold bloomPostLoopG at hpost
  set offoTile : Tile .nat [BLOCK_M, BLOCK_DMODEL] :=
    ⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
      outOffset s0 B_Start_Loc stride_obs stride_oh stride_od BLOCK_M idx⟩ with hoffo
  -- stmt 0: off_o
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .nat Broadcast.nil.consL.consR
        (Op.add .nat Broadcast.scalarR
          (Op.mul .nat Broadcast.scalarR
            (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "cur_batch_in_all_start_index")
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m")))
            (Op.constNat stride_obs))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat stride_oh)))
        (Op.mul .nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_DMODEL] "offs_d"))
          (Op.constNat stride_od))) sL = some offoTile from by
      simp only [evalOp_add, evalOp_mul, evalOp_constNat, evalOp_ref,
        ctx_evalOp_expandDim_one_nat, ctx_evalOp_expandDim_zero_nat,
        hcbsi, hch, hom, hod, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp only [hoffo, outOffset, startLoc, mIndex, dIndex,
        Tile.bop_data, Tile.scalar_data, Tile.vec_data,
        Tile.expandDim, Broadcast.leftIndex, Broadcast.rightIndex,
        NumericDType.add, NumericDType.mul]))] at hpost
  set s1 := sL.setReg "off_o" .nat [BLOCK_M, BLOCK_DMODEL] offoTile with hs1
  -- stmt 1: out_ptrs = Out + off_o
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Out)
        (Op.ref .nat [BLOCK_M, BLOCK_DMODEL] "off_o")) s1
        = some (⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
            (Out, offoTile.data idx)⟩ : Tile .ptr [BLOCK_M, BLOCK_DMODEL]) from by
      simp only [evalOp, hs1, BlockState.setReg_same, Region.cast_id,
        Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      · simp only [Tile.ptrAdd, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex]
      · simp only [Tile.ptrAdd, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex,
          Nat.zero_add]))] at hpost
  set s2 := s1.setReg "out_ptrs" .ptr [BLOCK_M, BLOCK_DMODEL]
    (⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => (Out, offoTile.data idx)⟩
      : Tile .ptr [BLOCK_M, BLOCK_DMODEL]) with hs2
  -- stmt 2: masked store
  set P : TileIndex [BLOCK_M, BLOCK_DMODEL] → Prop :=
    fun idx => s0.pids 2 * BLOCK_M + idx.1.val < sl ∧ idx.2.1.val < head_dim with hP
  have hmaskev : evalOp (Op.boolAnd Broadcast.nil.consL.consR
        (Op.lt .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m"))
          (Op.ref .nat [] "cur_batch_seq_len"))
        (Op.lt .nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_DMODEL] "offs_d"))
          (Op.constNat head_dim))) s2
      = some (⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          decide (s0.pids 2 * BLOCK_M + idx.1.val < sl) && decide (idx.2.1.val < head_dim)⟩
            : Tile .bool [BLOCK_M, BLOCK_DMODEL]) :=
    bloomQMask_evalG s2 (fun r : Fin BLOCK_M => s0.pids 2 * BLOCK_M + r.val) sl head_dim
      (by rw [hs2, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            hs1, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hom)
      (by rw [hs2, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            hs1, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hod)
      (by rw [hs2, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            hs1, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hsl)
  have haccref : evalOp (Op.ref .real [BLOCK_M, BLOCK_DMODEL] "acc") s2 = some accTile := by
    rw [evalOp_ref, hs2, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      hs1, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
    exact haccref0
  have hptrref : evalOp (Op.ref .ptr [BLOCK_M, BLOCK_DMODEL] "out_ptrs") s2
      = some (⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => (Out, offoTile.data idx)⟩
          : Tile .ptr [BLOCK_M, BLOCK_DMODEL]) := by
    rw [evalOp_ref, hs2, BlockState.setReg_same]
  have hstore : stepStmt (Stmt.store .real [BLOCK_M, BLOCK_DMODEL]
      (MemAccess.ptr (Op.ref .ptr [BLOCK_M, BLOCK_DMODEL] "out_ptrs"))
      (Op.ref .real [BLOCK_M, BLOCK_DMODEL] "acc")
      (MaskOpt.mask
        (Op.boolAnd Broadcast.nil.consL.consR
          (Op.lt .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m"))
            (Op.ref .nat [] "cur_batch_seq_len"))
          (Op.lt .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_DMODEL] "offs_d"))
            (Op.constNat head_dim))))) s2
      = some ((TileShape.allIndices [BLOCK_M, BLOCK_DMODEL]).foldl
          (fun acc idx => if P idx then acc.writeMem Out (offoTile.data idx)
            ((accTile.data idx).unbotD 0) else acc) s2) := by
    unfold stepStmt
    rw [haccref]
    simp only [hmaskev, Option.bind_eq_bind, Option.bind_some, Option.map_some]
    rw [hptrref]
    simp only [Option.bind_eq_bind, Option.bind_some]
    refine congrArg some ?_
    refine List.foldl_ext _ _ s2 ?_
    intro acc idx _
    by_cases h1 : s0.pids 2 * BLOCK_M + idx.1.val < sl
    · by_cases h2 : idx.2.1.val < head_dim
      · rw [decide_eq_true_eq.mpr h1, decide_eq_true_eq.mpr h2, Bool.and_true]
        simp only [if_true, BlockState.writeMemTyped_real, FloatDType.real_storeValue]
        rw [if_pos (show P idx from ⟨h1, h2⟩)]
      · rw [decide_eq_false_iff_not.mpr h2, Bool.and_false]
        simp only [Bool.false_eq_true, if_false]
        rw [if_neg (fun hc => h2 hc.2)]
    · rw [decide_eq_false_iff_not.mpr h1, Bool.false_and]
      simp only [Bool.false_eq_true, if_false]
      rw [if_neg (fun hc => h1 hc.1)]
  rw [stepStmts.cons_some hstore, stepStmts.nil] at hpost
  obtain rfl := Option.some.inj hpost
  intro r o hno
  have hactive_of_P : ∀ idx : TileIndex [BLOCK_M, BLOCK_DMODEL],
      P idx → active s0 B_Seqlen B_Prompt_Cache_Len head_dim BLOCK_M idx := by
    intro idx hPk
    have hactive_iff : active s0 B_Seqlen B_Prompt_Cache_Len head_dim BLOCK_M idx ↔ P idx := by
      have he : mIndex s0 BLOCK_M idx.1 = s0.pids 2 * BLOCK_M + idx.1.val := rfl
      simp only [active, hP, he, hsld, dIndex]
    exact hactive_iff.mpr hPk
  refine (bloomFwdIO_foldl_writeMem_frame_masked Out _ _ _ _ s2 r o ?_).trans ?_
  · intro hr k _ hPk
    exact Ne.symm (hno hr k (hactive_of_P k hPk))
  · rw [hs2, hs1]
    simp only [BlockState.setReg_mem]

set_option maxHeartbeats 3200000 in
set_option maxRecDepth 8000 in
/-- **Framed general execution**: `bloom_exec_general` extended with the
per-cell memory frame (this is the exact run `hrun` rides). -/
private theorem bloomFwdIO_exec_framed
    (Q K V Out : RegionName)
    (B_Start_Loc B_Seqlen Req_to_tokens B_req_idx B_Prompt_Cache_Len : Region .nat)
    (s : BlockState) (sm_scale : ℝ)
    (stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
      head_dim BLOCK_DMODEL BLOCK_M BLOCK_N : Nat) (hD : 0 < BLOCK_DMODEL) (hBN : 0 < BLOCK_N)
    (hOInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        outOffset s B_Start_Loc stride_obs stride_oh stride_od BLOCK_M idx))
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ∃ sF, stepStmts ((context_attn_bloom_fwd_kernel_surface Q K V sm_scale
        B_Start_Loc B_Seqlen Out Req_to_tokens B_req_idx B_Prompt_Cache_Len
        stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd
        stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
        stride_req_b stride_req_s 1 head_dim BLOCK_M BLOCK_DMODEL
        BLOCK_N).toAlgKernel.body) s = some sF
      ∧ (∀ idx : TileIndex [BLOCK_M, BLOCK_DMODEL],
          active s B_Seqlen B_Prompt_Cache_Len head_dim BLOCK_M idx →
          sF.readMem Out (outOffset s B_Start_Loc stride_obs stride_oh stride_od BLOCK_M idx)
            = bloomFwdGenuineOutValueG s Q K V B_Start_Loc B_Seqlen Req_to_tokens B_req_idx
                B_Prompt_Cache_Len sm_scale
                stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh
                stride_kd stride_vbs stride_vh stride_vd head_dim BLOCK_DMODEL BLOCK_N
                BLOCK_M idx)
      ∧ (∀ r o, (r ≠ Out ∨ ∀ idx : TileIndex [BLOCK_M, BLOCK_DMODEL],
            active s B_Seqlen B_Prompt_Cache_Len head_dim BLOCK_M idx →
            o ≠ outOffset s B_Start_Loc stride_obs stride_oh stride_od BLOCK_M idx) →
          sF.mem r o = s.mem r o) := by
  rw [bloomBody_splitG Q K V Out B_Start_Loc B_Seqlen Req_to_tokens B_req_idx B_Prompt_Cache_Len
    sm_scale stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd
    stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
    stride_req_b stride_req_s 1 head_dim BLOCK_M BLOCK_DMODEL BLOCK_N]
  -- preLoop
  obtain ⟨s0, hpre, hpids, hmem, hundef0, hcb, hch, hstart_m, hckvh, hplen, hcbsi, hrqi0, hsl0,
      hon, hod, hom, hmi, hli, hacc, hbel0, hbm0, hq⟩ :=
    bloomPreLoopG_eval s Q B_Start_Loc B_Seqlen B_req_idx B_Prompt_Cache_Len
      stride_qbs stride_qh stride_qd head_dim 1 BLOCK_DMODEL BLOCK_M BLOCK_N hundef
  rw [stepStmts.append_some hpre]
  set plen := promptLen s0 B_Prompt_Cache_Len with hplend
  set sl := seqLen s0 B_Seqlen B_Prompt_Cache_Len with hsld
  set bel := bloomFwdBel s0 B_Seqlen B_Prompt_Cache_Len BLOCK_M with hbeld
  set bm := (if BLOCK_M * s0.pids 2 < sl then 1 else 0) with hbmd
  set S := BLOCK_N * ((bm * bel + (BLOCK_N - 1)) / BLOCK_N) with hSd
  have hmemv : ∀ (rg : RegionName) (i : Nat),
      s.readMemValue .nat rg i = s0.readMemValue .nat rg i := by
    intro rg i; simp only [BlockState.readMemValue, BlockState.readMemTyped, hmem]
  have hmemvr : ∀ (rg : RegionName) (i : Nat),
      s.readMemValue .real rg i = s0.readMemValue .real rg i := by
    intro rg i; simp only [BlockState.readMemValue, BlockState.readMemAs, hmem]
  have hbelrb : s0.regs .nat [] "block_end_loc" = some (Tile.scalar bel) := by
    rw [hbel0]
    refine congrArg (fun x => some (Tile.scalar x)) ?_
    simp only [hbeld, bloomFwdBel, hsld, hplend, seqLen, promptLen, Region.cast_cast, ← hpids,
      hmemv]
  have hbmrb : s0.regs .nat [] "block_mask" = some (Tile.scalar bm) := by
    rw [hbm0]
    refine congrArg (fun x => some (Tile.scalar x)) ?_
    simp only [hbmd, hsld, hplend, seqLen, promptLen, Region.cast_cast, ← hpids, hmemv]
  -- loop-entry invariant at block 0
  have hinv0 : bloomInvariantG Q K V Out B_Start_Loc B_Seqlen Req_to_tokens B_req_idx
      B_Prompt_Cache_Len s0 sm_scale
      stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd head_dim BLOCK_DMODEL BLOCK_N BLOCK_M hD S bel 0 s0 := by
    refine ⟨rfl, rfl, hundef0, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · rw [hcb, hpids]
    · rw [hckvh, hpids]
    · rw [hch, hpids]
    · rw [hplen]; simp only [promptLen, Region.cast_cast, hmemv, ← hpids]
    · rw [hsl0]; simp only [seqLen, promptLen, Region.cast_cast, hmemv, ← hpids]
    · exact hbelrb
    · rw [hcbsi]; simp only [startLoc, Region.cast_cast, hmemv, ← hpids]
    · rw [hrqi0]; simp only [reqIdx, Region.cast_cast, hmemv, ← hpids]
    · rw [hom, hpids]
    · rw [hon]
    · rw [hod]
    · rw [hq]; refine congrArg some ?_; ext idx
      simp only [seqLen, promptLen, startLoc, Region.cast_id, hpids, hmemv, hmemvr]
    · rw [hmi]; refine congrArg some ?_; ext r
      simp only [Nat.zero_mul, gStateBot_zero, WithBot.map_bot]
    · rw [hli]; refine congrArg some ?_; ext r
      simp only [Nat.zero_mul, gStateBot_zero]
    · rw [hacc]; refine congrArg some ?_; ext idx
      simp only [gAccN_zero]
    · omega
  -- run the streaming loop
  obtain ⟨final, sL, hloop, hfin, c_final, hfinaleq, hinvL⟩ :=
    forRangeDyn_inv (idx := "start_n")
      (startOp := Op.constNat 0)
      (stopOp := Op.mul .nat Broadcast.nil (Op.ref .nat [] "block_mask")
        (Op.ref .nat [] "block_end_loc"))
      (stepOp := Op.constNat BLOCK_N)
      (P := fun i st => ∃ c, i = c * BLOCK_N ∧
        bloomInvariantG Q K V Out B_Start_Loc B_Seqlen Req_to_tokens B_req_idx
          B_Prompt_Cache_Len s0 sm_scale
          stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
          stride_vbs stride_vh stride_vd head_dim BLOCK_DMODEL BLOCK_N BLOCK_M hD S bel c st)
      (s_init := s0)
      (by rw [evalOp_constNat])
      (show evalOp (Op.mul .nat Broadcast.nil (Op.ref .nat [] "block_mask")
            (Op.ref .nat [] "block_end_loc")) s0 = some (Tile.scalar (bm * bel)) from by
        rw [evalOp_mul, evalOp_ref, evalOp_ref, hbmrb, hbelrb]
        simp only [Option.bind_eq_bind, Option.bind_some]; rfl)
      (by rw [evalOp_constNat])
      (Nat.pos_iff_ne_zero.mp hBN)
      ⟨0, by ring, hinv0⟩
      (fun i st hi hP => by
        obtain ⟨c, hic, hinvc⟩ := hP
        have hwin : (c + 1) * BLOCK_N ≤ S := by
          have hstop : c * BLOCK_N < bm * bel := hic ▸ hi
          rw [hSd]
          have hge : bm * bel ≤ BLOCK_N * ((bm * bel + (BLOCK_N - 1)) / BLOCK_N) := by
            have hdm := Nat.div_add_mod (bm * bel + (BLOCK_N - 1)) BLOCK_N
            have h3 : (bm * bel + (BLOCK_N - 1)) % BLOCK_N < BLOCK_N := Nat.mod_lt _ hBN
            omega
          have hcq : c < (bm * bel + (BLOCK_N - 1)) / BLOCK_N := by
            by_contra hcon
            have hcon' : (bm * bel + (BLOCK_N - 1)) / BLOCK_N ≤ c := Nat.not_lt.mp hcon
            have h1 : BLOCK_N * ((bm * bel + (BLOCK_N - 1)) / BLOCK_N) ≤ BLOCK_N * c :=
              Nat.mul_le_mul_left _ hcon'
            have h2 : bm * bel ≤ BLOCK_N * c := le_trans hge h1
            rw [Nat.mul_comm BLOCK_N c] at h2
            omega
          calc (c + 1) * BLOCK_N = BLOCK_N * (c + 1) := by ring
            _ ≤ BLOCK_N * ((bm * bel + (BLOCK_N - 1)) / BLOCK_N) := Nat.mul_le_mul_left _ hcq
        obtain ⟨s', hs', hinv'⟩ :=
          bloom_attn_stepG Q K V Out B_Start_Loc B_Seqlen Req_to_tokens B_req_idx
            B_Prompt_Cache_Len s0 sm_scale
            stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
            stride_vbs stride_vh stride_vd BLOCK_DMODEL BLOCK_N BLOCK_M head_dim hD hBN
            S bel c i st hwin hic hinvc
        exact ⟨s', hs', c + 1, by rw [hic]; ring, hinv'⟩)
  subst hfinaleq
  have hcle : c_final * BLOCK_N ≤ S := hinvL.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have hScfinal : S = c_final * BLOCK_N := by
    have hfinge : bm * bel ≤ c_final * BLOCK_N := hfin
    have hqle : (bm * bel + (BLOCK_N - 1)) / BLOCK_N ≤ c_final := by
      rw [Nat.div_le_iff_le_mul_add_pred hBN]
      have hcomm : c_final * BLOCK_N = BLOCK_N * c_final := Nat.mul_comm _ _
      omega
    have hSub : S ≤ c_final * BLOCK_N := by
      rw [hSd, Nat.mul_comm BLOCK_N]
      exact Nat.mul_le_mul_right _ hqle
    omega
  -- postLoop
  obtain ⟨sP, hpostStep, hO⟩ :=
    bloomPostLoopG_eval Q K V Out B_Start_Loc B_Seqlen Req_to_tokens B_req_idx B_Prompt_Cache_Len
      s0 sm_scale stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh
      stride_kd stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
      head_dim BLOCK_DMODEL BLOCK_N BLOCK_M hD hBN S bel c_final sL hScfinal
      (by
        have : (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
              outOffset s0 B_Start_Loc stride_obs stride_oh stride_od BLOCK_M idx)
            = (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
              outOffset s B_Start_Loc stride_obs stride_oh stride_od BLOCK_M idx) := by
          funext idx
          simp only [outOffset, startLoc, mIndex, dIndex, hpids, hmem,
            BlockState.readMemValue, BlockState.readMemTyped]
        rw [this]; exact hOInj)
      hinvL
  have hframeP := bloomFwdIO_postLoop_frame Q K V Out
    (Region.cast B_Start_Loc) (Region.cast B_Seqlen) (Region.cast Req_to_tokens)
    (Region.cast B_req_idx) (Region.cast B_Prompt_Cache_Len) s0 sm_scale
    stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
    stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
    head_dim BLOCK_DMODEL BLOCK_N BLOCK_M hD S bel c_final sL sP hScfinal hinvL hpostStep
  refine ⟨sP, ?_, ?_, ?_⟩
  · rw [stepStmts.cons_some hloop]; exact hpostStep
  · intro idx hac
    have hOidx := hO idx
    have houtoff : outOffset s0 B_Start_Loc stride_obs stride_oh stride_od BLOCK_M idx
        = outOffset s B_Start_Loc stride_obs stride_oh stride_od BLOCK_M idx := by
      simp only [outOffset, startLoc, mIndex, dIndex, hpids, hmemv]
    have hact : active s0 B_Seqlen B_Prompt_Cache_Len head_dim BLOCK_M idx
        ↔ active s B_Seqlen B_Prompt_Cache_Len head_dim BLOCK_M idx := by
      simp only [active, mIndex, seqLen, promptLen, dIndex, hpids, hmemv]
    rw [houtoff] at hOidx
    rw [hOidx, if_pos (hact.mpr hac)]
    have hreadmem : ∀ (rg : RegionName) (o : Nat), s0.readMem rg o = s.readMem rg o := by
      intro rg o; unfold BlockState.readMem; rw [hmem]
    have hbeleq : bel = bloomFwdBel s B_Seqlen B_Prompt_Cache_Len BLOCK_M := by
      simp only [hbeld, bloomFwdBel, hsld, hplend, seqLen, promptLen, ← hpids, hmemv]
    have hSeq : S = bloomFwdWindowG s B_Seqlen B_Prompt_Cache_Len BLOCK_M BLOCK_N := by
      simp only [bloomFwdWindowG, bloomFwdBel, hSd, hbmd, hbeld, bloomFwdBel, hsld, hplend,
        seqLen, promptLen, ← hpids, hmemv]
    rw [bloomFwdGenuineOutValueG, ← hbeleq, ← hSeq]
    have hkvm : bloomKVMG s0 Q K V B_Start_Loc B_Seqlen B_Prompt_Cache_Len Req_to_tokens
          B_req_idx sm_scale
          stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
          stride_vbs stride_vh stride_vd head_dim BLOCK_DMODEL BLOCK_M S bel idx.1 idx.2.1.val
        = bloomKVMG s Q K V B_Start_Loc B_Seqlen B_Prompt_Cache_Len Req_to_tokens B_req_idx
          sm_scale
          stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
          stride_vbs stride_vh stride_vd head_dim BLOCK_DMODEL BLOCK_M S bel idx.1 idx.2.1.val := by
      funext j
      simp only [bloomKVMG, bloomQTileMG, bloomKTileMG, bloomVTileMG, bloomQTileG, bloomKTileG,
        bloomVTileG, bloomKvLocG, promptLen, seqLen, reqIdx, startLoc, curHead, hpids, hmemv,
        hreadmem]
    simp only [contextAttnBloomExactFoldMG, hkvm]
    have hplenq : promptLen s0 B_Prompt_Cache_Len = promptLen s B_Prompt_Cache_Len := by
      simp only [promptLen, hpids, hmemv]
    rw [hplenq, hpids]
  · intro r o hcond
    have hcond0 : r = Out → ∀ idx : TileIndex [BLOCK_M, BLOCK_DMODEL],
        active s0 (Region.cast B_Seqlen) (Region.cast B_Prompt_Cache_Len) head_dim BLOCK_M idx →
        o ≠ outOffset s0 (Region.cast B_Start_Loc) stride_obs stride_oh stride_od BLOCK_M idx := by
      intro hr idx hact0
      rcases hcond with hne | hno
      · exact absurd hr hne
      · have hactIff : active s0 B_Seqlen B_Prompt_Cache_Len head_dim BLOCK_M idx
            ↔ active s B_Seqlen B_Prompt_Cache_Len head_dim BLOCK_M idx := by
          simp only [active, mIndex, seqLen, promptLen, dIndex, hpids, hmemv]
        have houtoff : outOffset s0 B_Start_Loc stride_obs stride_oh stride_od BLOCK_M idx
            = outOffset s B_Start_Loc stride_obs stride_oh stride_od BLOCK_M idx := by
          simp only [outOffset, startLoc, mIndex, dIndex, hpids, hmemv]
        have h := hno idx (hactIff.mp hact0)
        rw [houtoff]
        exact h
    rw [hframeP r o hcond0]
    have hmemL : sL.mem = s0.mem := hinvL.2.1
    rw [hmemL, hmem]

/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/
set_option maxHeartbeats 8000000 in
set_option maxRecDepth 8000 in
/-- **The `⊨[R]` gather-skin headline** — `context_attn_bloom` on
`StreamMetaGatherMasked3DKernelIO₃`, the paged-KV gather skin, at fully
symbolic per-axis strides and symbolic `head_dim`. For every rounding model
`R`, the faithful surface implements, on its metadata + gather + three-stream
signature, the streamed closed form `contextAttnBloomIOSpec`: every
write-active output lane `j = (i, e)` holds the **block-causal-guarded
normalized accumulator** `gAccN` of the ⊥-seeded online-softmax fold over the
live streamed window, with the prompt-cache-offset causal boundary and the
**finite `-1e8` sentinel** — masked keys genuinely carry weight
`exp(-1e8 − m)`, so the spec transcribes the kernel's fold verbatim rather
than a cleaned-up causal softmax (the `-1e8` honesty is inherited from the
exact headline, not repaired here).

**BLOOM's shape, versus its `context_attn_llama` sibling.** The grid is
genuinely 3-D (`cur_batch = pid₀`, `cur_head = pid₁`, `start_m = pid₂`), so
all four `.nat` slots are read at cell `pid₀` and `cur_head` enters the
windows as the bare `pid₁` (no `pid / H` decode). Normalization is **in
loop** (`p_scale = β/lᵢⁿᵉʷ`, `acc_scale = tl.where(offs_m+plen ≥ start_n,
(lᵢ/lᵢⁿᵉʷ)·α, 1.0)`) with **no** post-loop divide, so the spec is the `gAccN`
shape rather than llama's `acc/l` ratio; the block-causal guard on
`acc_scale` is modelled exactly. The kernel uses natural `tl.exp` with
`sm_scale = 1/√D`; the per-key score is divided by `Real.log 2` exactly as
the port's `bloomKVMG` does, so the shared `pow2`/`gStateBot` machinery
expresses it. The slots are `m 0 = cur_batch_in_all_start_index`,
`m 1 = prompt_cache_len`, `m 2` = the **raw** `B_Seqlen` load — the kernel
subtracts in-register, so masks carry the ℕ-truncated `m 2 - m 1` verbatim —
and `m 3 = cur_batch_req_idx`, the page-table row.

**The `head_dim` delta.** Every float load and the terminal store carry the
extra channel guard `offs_d < head_dim` (`BLOCK_DMODEL` is the padded tile
width), so `mask1`/`mask2`/`mask3`/`writeMask` are conjunctions. `gmask` is
*not*: the `Req_to_tokens` tile is a `[BLOCK_N]` vector with only the
`block_end_loc` liveness guard.

**The gather layer.** `Req_to_tokens` is the skin's index channel: a
`[BLOCK_N]` `.nat` tile `G t ·` per step, read at
`stride_req_b·m 3 + stride_req_s·(t·BN + jL)` under the same `block_end_loc`
liveness the `K`/`V` loads carry, `other = 0` on dead lanes. `G` is
universally quantified beside the slot vector and pinned with the skin's two
legs; the `K`/`V` windows eat it (`read2`/`read3`), while the spec `f` does
**not** — the gathered index moves addresses only, and the data pins already
deliver the gathered cells as the ordinary streams
(`bloomFwdIOSpec_eq_genuine` is the single place `bloomKvLocG ↦ G` is
rewritten, on the pin's active leg). Consequently the in-bounds-ness of the
*gathered values* is a hypothesis of the triple (`hbr2`/`hbr3`, quantified
over `G`), exactly like the slot values: host page-table contents are trusted
input.

The kernel has **zero rounding events** (`.nat` slot/gather loads,
`other=0.0`-masked `.real` loads, `.real` in-loop arithmetic, untyped terminal
store), so the skin's boundary quantization degenerates: the readback's
`R.round .real` is the identity by the model's defining `round_real`.

**Launch legality (`pre` = the trusted-launch boundary).** The triple is
guarded by `io.pre pid₂ m = (pid₂ < NT ∧ m 2 ≤ NT·BLOCK_M)` — the gather adds
**nothing** to `pre` — exactly the port's documented trusted boundary (see the
file docstring's Scope section): the host launches
`grid = (batch, head, cdiv(max_input_len, BLOCK))` with `NT` the third grid
dimension and every raw `B_Seqlen[b] ≤ max_input_len ≤ NT·BLOCK_M`. The live
trip count `block_mask·block_end_loc` has no pid-free bound; under `pre` it is
`≤ NT·BLOCK_M ≤ BLOCK_N·T` (`bloomFwdIOBel_le`/`bloomFwdIOT_mul_le`), which is
what makes the `T = ⌈NT·BLOCK_M/BLOCK_N⌉`-step window citable in both the
safety walk and the value bridge. The `⊨[R]` triple says nothing about
launches outside this boundary.

**Hypothesis provenance**: `0 < BLOCK_DMODEL`, `0 < BLOCK_N` are the exact
headline's side conditions (nonempty tiles; `reduceMax` totality);
`0 < BLOCK_M` and `0 < NT` are truth-forced by the static `Q` stream's
step-`0` read (`0 < T`); `hOInj` restates the exact headline's **open**
output-offset injectivity side condition in ∀-pids/∀-base form (per-axis
strides are symbolic here — unlike `context_attn_nopad`'s contiguous pin, no
`DM ≤ rs` discharge is available). `kv_group_num = 1` is inherited from the
exact headline's pin. The exact headline's `hundef` is **not** a hypothesis
here — the skin's Hoare triple carries the `undef` pin itself.

Relation to the exact surface: the `Realizes_without_Rounding` headline above
is retained unchanged; this `⊨[R]` face restates the same genuine closed form
(`bloomFwdGenuineOutValueG`) on the gather skin, for every `R` at once. -/
specification context_attn_bloom_io_correctness (R : RoundingModel)
    (Q K V Out : RegionName)
    (B_Start_Loc B_Seqlen Req_to_tokens B_req_idx b_prompt_cache_len : Region .nat)
    (sm_scale : ℝ)
    (stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
      head_dim BLOCK_DMODEL BLOCK_M BLOCK_N NT : Nat)
    (hD : 0 < BLOCK_DMODEL) (hBN : 0 < BLOCK_N) (hBM : 0 < BLOCK_M) (hNT : 0 < NT)
    (hOInj : ∀ pid₁ pid₂ base : Nat, Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        (base + (pid₂ * BLOCK_M + idx.1.val)) * stride_obs + pid₁ * stride_oh
          + idx.2.1.val * stride_od)) :
    contextAttnBloomIO Q K V Out B_Start_Loc B_Seqlen Req_to_tokens B_req_idx
        b_prompt_cache_len sm_scale
        stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
        stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
        head_dim BLOCK_DMODEL BLOCK_M BLOCK_N NT ⊨[R]
      fun _ _ pid₂ m xs ys zs j =>
        contextAttnBloomIOSpec BLOCK_M BLOCK_DMODEL BLOCK_N head_dim NT hBN
          (bloomFwdIOT_pos NT BLOCK_M BLOCK_N hBM hBN hNT) sm_scale pid₂
          (m (⟨1, by omega⟩ : Fin 4)) (m (⟨2, by omega⟩ : Fin 4)) xs ys zs j := by
  refine StreamMetaGatherMasked3DKernelIO₃.ImplementsR.intro _ ?_ ?_ ?_
  · exact bloomFwdIO_flattenOk Q K V Out B_Start_Loc B_Seqlen Req_to_tokens B_req_idx
      b_prompt_cache_len sm_scale
      stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
      head_dim BLOCK_DMODEL BLOCK_M BLOCK_N
  · -- the safety walk
    intro bounds s m G xs ys zs hpre hm hg _hgo _hx _hy _hz hbm hbrG hbr1 hbr2 hbr3 hbw
    simp only [contextAttnBloomIO] at hpre hm hg hbm hbrG hbr1 hbr2 hbr3 hbw ⊢
    have hm0 : s.readMemValue .nat ↑B_Start_Loc (s.pids 0)
        = m (⟨0, by omega⟩ : Fin 4) := hm (⟨0, by omega⟩ : Fin 4)
    have hm1 : s.readMemValue .nat ↑b_prompt_cache_len (s.pids 0)
        = m (⟨1, by omega⟩ : Fin 4) := hm (⟨1, by omega⟩ : Fin 4)
    have hm2 : s.readMemValue .nat ↑B_Seqlen (s.pids 0)
        = m (⟨2, by omega⟩ : Fin 4) := hm (⟨2, by omega⟩ : Fin 4)
    have hm3 : s.readMemValue .nat ↑B_req_idx (s.pids 0)
        = m (⟨3, by omega⟩ : Fin 4) := hm (⟨3, by omega⟩ : Fin 4)
    refine bloomFwdIO_traceSafeR R bounds Q K V Out B_Start_Loc B_Seqlen Req_to_tokens B_req_idx
      b_prompt_cache_len sm_scale
      stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
      head_dim BLOCK_DMODEL BLOCK_M BLOCK_N NT hBN s
      (by rw [hm2]; exact hpre.2)
      (hbm (⟨1, by omega⟩ : Fin 4)) (hbm (⟨0, by omega⟩ : Fin 4))
      (hbm (⟨2, by omega⟩ : Fin 4)) (hbm (⟨3, by omega⟩ : Fin 4)) ?_ ?_ ?_ ?_ ?_
    · -- `Q`: the static stream's step-0 read
      intro i e hrow hhd
      have h := hbr1 ⟨0, bloomFwdIOT_pos NT BLOCK_M BLOCK_N hBM hBN hNT⟩
        (Lane2D.encode (i, e, PUnit.unit))
        (by rw [Lane2D.encode_div, Lane2D.encode_mod, ← hm1, ← hm2]; exact ⟨hrow, hhd⟩)
      rw [Lane2D.encode_div, Lane2D.encode_mod, ← hm0] at h
      exact h
    · -- the gather window itself
      intro t jL hcol
      have h := hbrG t jL (by rw [← hm1, ← hm2]; exact hcol)
      rw [← hm3] at h
      exact h
    · -- `K`: the gathered value's window, pinned back to memory by `hg`
      intro t e jL hcol hhd
      have hcolm : t.val * BLOCK_N + jL.val
          < bloomFwdIOBel BLOCK_M (s.pids 2) (m (⟨1, by omega⟩ : Fin 4))
              (m (⟨2, by omega⟩ : Fin 4)) := by
        rw [← hm1, ← hm2]; exact hcol
      have hgv : s.readMemValue .nat ↑Req_to_tokens
          (stride_req_b * m (⟨3, by omega⟩ : Fin 4)
            + stride_req_s * (t.val * BLOCK_N + jL.val)) = G t jL := hg t jL hcolm
      have h := hbr2 t (Lane2D.encode (e, jL, PUnit.unit))
        (by rw [Lane2D.encode_mod, Lane2D.encode_div]; exact ⟨hcolm, hhd⟩)
      rw [Lane2D.decode_encode, Lane2D.encode_div, ← hgv, ← hm3] at h
      exact h
    · -- `V`: mirror
      intro t jL e hcol hhd
      have hcolm : t.val * BLOCK_N + jL.val
          < bloomFwdIOBel BLOCK_M (s.pids 2) (m (⟨1, by omega⟩ : Fin 4))
              (m (⟨2, by omega⟩ : Fin 4)) := by
        rw [← hm1, ← hm2]; exact hcol
      have hgv : s.readMemValue .nat ↑Req_to_tokens
          (stride_req_b * m (⟨3, by omega⟩ : Fin 4)
            + stride_req_s * (t.val * BLOCK_N + jL.val)) = G t jL := hg t jL hcolm
      have h := hbr3 t (Lane2D.encode (jL, e, PUnit.unit))
        (by rw [Lane2D.encode_div, Lane2D.encode_mod]; exact ⟨hcolm, hhd⟩)
      rw [Lane2D.decode_encode, Lane2D.encode_mod, ← hgv, ← hm3] at h
      exact h
    · -- `Out`: the terminal store's window
      intro i e hrow hhd
      have h := hbw (Lane2D.encode (i, e, PUnit.unit))
        (by rw [Lane2D.encode_div, Lane2D.encode_mod, ← hm1, ← hm2]; exact ⟨hrow, hhd⟩)
      rw [Lane2D.encode_div, Lane2D.encode_mod, ← hm0] at h
      exact h
  · -- the rounded Hoare triple: framed exact stack + cast-free collapse
    intro s₀ m G xs ys zs hpre hu hm hg _hgo hx hy hz
    simp only [contextAttnBloomIO] at hpre hm hg hx hy hz ⊢
    have hundef' : ∀ rg o, s₀.undef rg o = 0 := fun rg o => by rw [hu]
    have hm0 : s₀.readMemValue .nat ↑B_Start_Loc (s₀.pids 0)
        = m (⟨0, by omega⟩ : Fin 4) := hm (⟨0, by omega⟩ : Fin 4)
    have hm1 : s₀.readMemValue .nat ↑b_prompt_cache_len (s₀.pids 0)
        = m (⟨1, by omega⟩ : Fin 4) := hm (⟨1, by omega⟩ : Fin 4)
    have hm2 : s₀.readMemValue .nat ↑B_Seqlen (s₀.pids 0)
        = m (⟨2, by omega⟩ : Fin 4) := hm (⟨2, by omega⟩ : Fin 4)
    have hm3 : s₀.readMemValue .nat ↑B_req_idx (s₀.pids 0)
        = m (⟨3, by omega⟩ : Fin 4) := hm (⟨3, by omega⟩ : Fin 4)
    have hOInj' : Function.Injective
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          outOffset s₀ ↑B_Start_Loc stride_obs stride_oh stride_od BLOCK_M idx) :=
      hOInj (s₀.pids 1) (s₀.pids 2) (startLoc s₀ ↑B_Start_Loc)
    obtain ⟨sF, hstep, hOut, hframe⟩ :=
      bloomFwdIO_exec_framed Q K V Out B_Start_Loc B_Seqlen Req_to_tokens B_req_idx
        b_prompt_cache_len s₀ sm_scale
        stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
        stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
        head_dim BLOCK_DMODEL BLOCK_M BLOCK_N hD hBN hOInj' hundef'
    refine ⟨sF, ?_, ?_, ?_⟩
    · rw [bloomFwdIO_execR_collapse]
      exact hstep
    · -- readback: the genuine closed form = the streamed closed form
      intro j hj
      have hact : active s₀ ↑B_Seqlen ↑b_prompt_cache_len head_dim BLOCK_M (Lane2D.decode j) := by
        refine ⟨?_, ?_⟩
        · show s₀.pids 2 * BLOCK_M + (Lane2D.decode j).1.val
            < s₀.readMemValue .nat ↑B_Seqlen (s₀.pids 0)
              - s₀.readMemValue .nat ↑b_prompt_cache_len (s₀.pids 0)
          rw [Lane2D.decode_row, hm1, hm2]
          exact hj.1
        · show (Lane2D.decode j).2.1.val < head_dim
          rw [Lane2D.decode_col]
          exact hj.2
      have hOutj := hOut (Lane2D.decode j) hact
      have haddr : (s₀.readMemValue .nat ↑B_Start_Loc (s₀.pids 0)
            + (s₀.pids 2 * BLOCK_M + j.val / BLOCK_DMODEL)) * stride_obs
          + s₀.pids 1 * stride_oh + j.val % BLOCK_DMODEL * stride_od
          = outOffset s₀ ↑B_Start_Loc stride_obs stride_oh stride_od BLOCK_M
              (Lane2D.decode j) := by
        simp only [outOffset, startLoc, mIndex, dIndex, Lane2D.decode_row, Lane2D.decode_col]
      rw [BlockState.readMemAs_real, ← hm0, haddr, hOutj, R.round_real_apply]
      refine congrArg some ?_
      exact bloomFwdIOSpec_eq_genuine s₀ Q K V B_Start_Loc B_Seqlen Req_to_tokens B_req_idx
        b_prompt_cache_len sm_scale
        stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
        stride_vbs stride_vh stride_vd head_dim BLOCK_DMODEL BLOCK_M BLOCK_N NT hBN
        (bloomFwdIOT_pos NT BLOCK_M BLOCK_N hBM hBN hNT)
        (m (⟨0, by omega⟩ : Fin 4)) (m (⟨1, by omega⟩ : Fin 4)) (m (⟨2, by omega⟩ : Fin 4))
        (m (⟨3, by omega⟩ : Fin 4))
        hm0 hm1 hm2 hm3 hpre.2 G xs ys zs hg hx hy hz j
    · -- the frame
      intro r o hcond
      refine hframe r o ?_
      rcases hcond with hne | hno
      · exact Or.inl hne
      · refine Or.inr (fun idx hact hoeq => ?_)
        exact hno (Lane2D.encode idx)
          (by rw [Lane2D.encode_div, Lane2D.encode_mod, ← hm2, ← hm1]
              exact ⟨hact.1, hact.2⟩)
          (hoeq.trans (by
            simp only [outOffset, startLoc, mIndex, dIndex,
              Lane2D.encode_div, Lane2D.encode_mod, hm0]))

end IOFace

end VeriTile.Bench.TritonBenchG.ContextAttnBloom
