import VeriTile.Triton

/-!
# `context_attn_nopad` — strict per-kernel correctness

`_fwd_kernel` is varlen ("no padding") context (prefill) attention. Each program
`(cur_batch, cur_head, start_m)` loads a `[BLOCK_M, BLOCK_DMODEL]` query tile,
runs an online-softmax (`m_i`/`l_i`/`acc`) loop over the key/value tokens with a
plain causal mask (`offs_m >= start_n + offs_n`), and stores the accumulated
`acc` tile to `Out`, masked by `offs_m < cur_batch_seq_len`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`_fwd_kernel[grid](...)` with
`grid = (batch, head, cdiv(...))`, the scheduling over batch / head / sequence
blocks, and how the runtime composes per-program writes into one buffer) is the
*trusted boundary*, not a proof obligation here. Because the program ids
`(cur_batch, cur_head, start_m)` are universally quantified, the per-program
statement covers every program of the grid.

## Proof architecture

```
context_attn_nopad_output_summary_general                   ← TOP THEOREM (symbolic, dimension-general)
  └─ nopad_exec_general                      full surface exec → genuine closed form (dimension-general)
       ├─ nopadPreLoop_evalG                 19 preLoop stmts → nopadInvariantG … 0
       ├─ forRangeDyn_inv ∘ nopad_attn_stepG loop body advances nopadInvariantG c → c+1
       │    └─ osNormStepBot_block_eq + nopadBlockMG_{sup,lij,acc} bridges
       └─ nopadPostLoop_evalG                masked store → ctxNopadGenuineOutValueG
            └─ nopadFoldUptoG_full_eq_genuine ∘ ctxNopad_fold_eq_exactFoldMG
(genuine spec: ctxNopadGenuineOutValueG = contextAttnNopadExactFoldMG, the boundary-
 masked causal softmax; = attentionRealCausalBlock via the closed-form bridge.)
```

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); `@triton.autotune` is not
modeled. The verified compute claim is **genuine, not self-referential**: the full
surface kernel (preLoop + the online-softmax streaming `forRangeDyn` loop +
masked store) realizes the closed-form causal-softmax fold
`ctxNopadGenuineOutValueG` — the boundary-masked
`numer/denom = (Σ_{j ≤ gi} exp(sm_scale·rawᵢⱼ)·V[j]) / (Σ_{j ≤ gi} exp(sm_scale·rawᵢⱼ))`
of the loaded Q/K/V memory — at every active lane (`offs_m < cur_batch_seq_len`,
with `offs_d < head_dim` folded into the slice), and preserves out-of-bounds lanes.
The online-softmax streaming loop (`m_i`/`l_i`/`acc` updates, `tl.dot`, the causal
`-inf` mask, in-loop normalization) is decoded statement-by-statement
(`nopadPreLoop_evalG`/`nopad_attn_stepG`/`nopadPostLoop_evalG`, assembled in
`nopad_exec_general`) and *proven* to collapse to this closed form, not re-stated as a
spec. The top theorem is
dimension-general: it is stated over symbolic `BLK`/`DM` (with
`BLOCK_M = BLOCK_N = BLK`, `BLOCK_DMODEL = DM`) and the contiguous layout strides
`(rs, hs, 1)`. The Python test shape (`BLK = DM = 128`, `rs = 768`, `hs = 128`) is
recovered as a concrete special case.
-/

namespace VeriTile.Bench.TritonBenchG.ContextAttnNopad

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-! **★ Main theorem:** `context_attn_nopad_output_summary_general` -/

/-! # ══════════ CORRECT — genuine / dimension-general (review this) ══════════ -/

section Correct_without_Rounding

/-- Faithful DSL port of `context_attn_nopad.py`'s `_fwd_kernel`. -/
def context_attn_nopad_fwd_kernel_surface
    (Q K V : RegionName) (sm_scale : ℝ)
    (B_Start_Loc B_Seqlen : Region .nat) (Out : RegionName)
    (stride_qbs stride_qh stride_qd
      stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd
      stride_obs stride_oh stride_od
      BLOCK_M BLOCK_DMODEL BLOCK_N : Nat) :
    ComputeKernel := triton {
  cur_batch = tl.program_id(0)
  cur_head = tl.program_id(1)
  start_m = tl.program_id(2)

  cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
  cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)

  block_start_loc = $(BLOCK_M) * start_m

  offs_n = tl.arange(0, $(BLOCK_N))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  off_q = (cur_batch_in_all_start_index + offs_m[:, None]) * $(stride_qbs) +
    cur_head * $(stride_qh) + offs_d[None, :] * $(stride_qd)
  off_k = offs_n[None, :] * $(stride_kbs) + cur_head * $(stride_kh) +
    offs_d[:, None] * $(stride_kd)
  off_v = offs_n[:, None] * $(stride_vbs) + cur_head * $(stride_vh) +
    offs_d[None, :] * $(stride_vd)

  q = tl.load(Q + off_q, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)

  k_ptrs = K + off_k
  v_ptrs = V + off_v

  m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)

  block_mask = tl.where(block_start_loc < cur_batch_seq_len, $(1), $(0))

  for start_n in range($(0), block_mask * (start_m + $(1)) * $(BLOCK_M), $(BLOCK_N)) {
    start_n = tl.multiple_of(start_n, $(BLOCK_N))
    k = tl.load(k_ptrs + (cur_batch_in_all_start_index + start_n) * $(stride_kbs),
      mask=(start_n + offs_n[None, :]) < cur_batch_seq_len, other=0.0)

    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    qk += tl.dot(q, k)
    qk *= $((sm_scale : ℝ))
    qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))

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
    acc = acc * acc_scale[:, None]
    v = tl.load(v_ptrs + (cur_batch_in_all_start_index + start_n) * $(stride_vbs),
      mask=(start_n + offs_n[:, None]) < cur_batch_seq_len, other=0.0)

    p = (p).to(v.dtype)
    acc += tl.dot(p, v)
    l_i = l_i_new
    m_i = m_i_new
  }
  off_o = (cur_batch_in_all_start_index + offs_m[:, None]) * $(stride_obs) +
    cur_head * $(stride_oh) + offs_d[None, :] * $(stride_od)
  out_ptrs = Out + off_o
  tl.store(out_ptrs, acc, mask=offs_m[:, None] < cur_batch_seq_len)
}

def seqLen (s : BlockState) (B_Seqlen : RegionName) : Nat :=
  s.readMemValue .nat B_Seqlen (s.pids 0)

def startLoc (s : BlockState) (B_Start_Loc : RegionName) : Nat :=
  s.readMemValue .nat B_Start_Loc (s.pids 0)

def dIndex (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  idx.2.1.val

/-- Kernel-decoded k/v load boundary `bel = cur_batch_seq_len`. -/
def ctxNopadBel (s : BlockState) (B_Seqlen : RegionName) : Nat :=
  seqLen s B_Seqlen

/-! ## Online-normalized streaming recurrence (the loop invariant's math)

Unlike the int8-KV context kernel (which divides `acc /= l_i` once after the
loop), `context_attn_nopad` **normalizes inside the loop**: each block it rescales
`p` by `beta/l_i_new` and `acc` by `l_i/l_i_new·alpha`, so after every block the
`acc` register already holds the *running normalized softmax ratio* `numer/denom`
and `l_i` holds the running denominator shifted by `exp(-m_i)` (i.e.
`Σ exp(score − m_i)`). This section is the mathematical heart of nopad's loop: the
⊥-seeded normalized recurrence (running max in `WithBot ℝ`, seeded `⊥` to model
`m_i = −inf`, `l_i = acc = 0`), and the proof that its running `(l, acc)` stay
equal to `exp(−m)·Σexp(score)` and `Σexp(score)v / Σexp(score)` — so the
full-window `acc` reads off the genuine causal-softmax closed form with NO
post-loop division. -/

/-- One ⊥-seeded **online-normalized** softmax step absorbing key `(sc, v)`. The
running max lives in `WithBot ℝ` (seeded `⊥`); `α = realExp(m ⊖ m')` is `0` on the
first key (faithful to `m_i = −inf`, `l_i = acc = 0`). `l` is the running
shifted denominator `Σ exp(score − m)`; `acc` is the running normalized ratio
`numer/denom`. The block update is `l' = α·l + exp(sc − m')`,
`acc' = acc·(l/l'·α) + (exp(sc − m')/l')·v`. -/
noncomputable def osNormStepBot
    (st : WithBot ℝ × ℝ × ℝ) (sv : ℝ × ℝ) : WithBot ℝ × ℝ × ℝ :=
  let m := st.1; let l := st.2.1; let acc := st.2.2
  let sc := sv.1; let v := sv.2
  let m' := m ⊔ ((sc : ℝ) : WithBot ℝ)
  let α := (WithBot.realExp (WithBot.realSub m m')).unbotD 0
  let l' := l * α + Real.exp (sc - m'.unbotD 0)
  let acc' := acc * (l / l' * α) + (Real.exp (sc - m'.unbotD 0) / l') * v
  (m', l', acc')

/-- The running `max` component of an `osNormStepBot` fold is the `WithBot ⊔`-fold
of the per-key scores — independent of the normalized `l`/`acc` carried. -/
theorem osNormStepBot_foldl_fst
    (xs : List (ℝ × ℝ)) (m₀ : WithBot ℝ) (l₀ acc₀ : ℝ) :
    (xs.foldl osNormStepBot (m₀, l₀, acc₀)).1
      = (xs.map (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldl (· ⊔ ·) m₀ := by
  induction xs generalizing m₀ l₀ acc₀ with
  | nil => rfl
  | cons x xs ih => simp only [List.foldl_cons, List.map_cons]; rw [ih]; rfl

/-- **⊥-seeded normalized consistency.** Folding `osNormStepBot` from a state
consistent with batch denominator `L` and unnormalized accumulator `T`
(`l = κ(m)·L`, `acc = T/L`, with the convention `acc = 0` when `L = 0`) keeps that
invariant: the final `l = κ(m_final)·(L + Σexp)`, `acc = (T + Σexp·v)/(L + Σexp)`.
`κ ⊥ = 0`, `κ (some r) = exp(−r)`. -/
theorem osNormStepBot_foldl_consistent
    (xs : List (ℝ × ℝ)) (m : WithBot ℝ) (l acc T L : ℝ)
    (hL0 : 0 ≤ L)
    (hxpos : ∀ p ∈ xs, True)
    (hl : l = (m.elim 0 (fun r => Real.exp (-r))) * L)
    (hacc : acc * L = T)
    (hmL : m = ⊥ → L = 0) (hmT : m = ⊥ → T = 0)
    (hLpos : 0 < L → l ≠ 0) :
    let st := xs.foldl osNormStepBot (m, l, acc)
    let L' := L + (xs.map (fun p => Real.exp p.1)).sum
    let T' := T + (xs.map (fun p => Real.exp p.1 * p.2)).sum
    st.2.1 = (st.1.elim 0 (fun r => Real.exp (-r))) * L'
      ∧ st.2.2 * L' = T'
      ∧ 0 ≤ L'
      ∧ (st.1 = ⊥ → L' = 0)
      ∧ (st.1 = ⊥ → T' = 0)
      ∧ (0 < L' → st.2.1 ≠ 0) := by
  induction xs generalizing m l acc T L with
  | nil => exact ⟨by simpa using hl, by simpa using hacc, by simpa using hL0,
      by simpa using hmL, by simpa using hmT, by simpa using hLpos⟩
  | cons x xs ih =>
    obtain ⟨sc, v⟩ := x
    set m' : WithBot ℝ := m ⊔ ((sc : ℝ) : WithBot ℝ) with hm'
    have hm'r : ∃ r : ℝ, m' = (r : WithBot ℝ) := by
      cases m with
      | bot => exact ⟨sc, by rw [hm']; rfl⟩
      | coe a => exact ⟨max a sc, by rw [hm']; rw [← WithBot.coe_max]⟩
    obtain ⟨mr, hmr⟩ := hm'r
    have hκm' : m'.elim 0 (fun r => Real.exp (-r)) = Real.exp (-mr) := by rw [hmr]; rfl
    have hunbot : m'.unbotD 0 = mr := by rw [hmr]; rfl
    set L' := L + Real.exp sc with hL'd
    set T' := T + Real.exp sc * v with hT'd
    set α : ℝ := (WithBot.realExp (WithBot.realSub m m')).unbotD 0 with hαd
    set p : ℝ := Real.exp (sc - m'.unbotD 0) with hpd
    set l' : ℝ := l * α + p with hl'd
    set acc' : ℝ := acc * (l / l' * α) + (p / l') * v with hacc'd
    -- l·α = exp(-mr)·L : `m = ⊥` uses `L = 0`; `m = ↑a` uses `l = exp(-a)·L`, `α = exp(a − mr)`.
    have hlα : l * α = Real.exp (-mr) * L := by
      cases m with
      | bot =>
        have hL0' : L = 0 := hmL rfl
        rw [hl, hL0']; simp
      | coe a =>
        have hm'a : m' = ((max a sc : ℝ) : WithBot ℝ) := by rw [hm']; rw [← WithBot.coe_max]
        have hmra : mr = max a sc := by rw [hm'a] at hmr; exact (WithBot.coe_inj.mp hmr.symm)
        have hαv : α = Real.exp (a - max a sc) := by
          rw [hαd, hm'a, WithBot.realSub_coe_coe, WithBot.realExp_coe, WithBot.unbotD_coe]
        rw [hl, hαv,
          show ((↑a : WithBot ℝ).elim 0 (fun r => Real.exp (-r))) = Real.exp (-a) from rfl]
        rw [show Real.exp (-a) * L * Real.exp (a - max a sc)
            = (Real.exp (-a) * Real.exp (a - max a sc)) * L from by ring,
          ← Real.exp_add, hmra]
        ring_nf
    -- new l' = κ(m')·L'
    have hl'eq : l' = Real.exp (-mr) * L' := by
      rw [hl'd, hlα, hpd, hunbot, hL'd]
      have e2 : Real.exp (sc - mr) = Real.exp (-mr) * Real.exp sc := by
        rw [← Real.exp_add]; ring_nf
      rw [e2]; ring
    -- L' ≥ 0
    have hL'pos : 0 ≤ L' := by rw [hL'd]; positivity
    -- L' > 0 (key always contributes exp sc > 0)
    have hL'strict : 0 < L' := by rw [hL'd]; positivity
    -- l' ≠ 0
    have hl'ne : l' ≠ 0 := by rw [hl'eq]; positivity
    -- acc' * L' = T'
    have hacc'eq : acc' * L' = T' := by
      rw [hacc'd, hpd, hunbot]
      rw [add_mul]
      -- term 2: (exp(sc-mr)/l')·v·L' = exp(sc-mr)·L'/l' · v ; l' = exp(-mr)L'
      have hl'val : l' = Real.exp (-mr) * L' := hl'eq
      have e2 : Real.exp (sc - mr) / l' * v * L'
          = Real.exp sc * v := by
        rw [hl'val]
        rw [show Real.exp (sc - mr) = Real.exp (-mr) * Real.exp sc from by
          rw [← Real.exp_add]; ring_nf]
        have hexpne : Real.exp (-mr) ≠ 0 := Real.exp_ne_zero _
        have hL'ne : (L' : ℝ) ≠ 0 := ne_of_gt hL'strict
        field_simp
      -- term 1: acc·(l/l'·α)·L' = acc·(l·α)·(L'/l') = acc·(exp(-mr)L)·(L'/(exp(-mr)L')) = acc·L = T
      have e1 : acc * (l / l' * α) * L' = T := by
        have hexpne : Real.exp (-mr) ≠ 0 := Real.exp_ne_zero _
        have hL'ne : (L' : ℝ) ≠ 0 := ne_of_gt hL'strict
        have hrw : acc * (l / l' * α) * L'
            = acc * (l * α) * L' / l' := by ring
        rw [hrw, hlα, hl'val]
        rw [show acc * (Real.exp (-mr) * L) * L' / (Real.exp (-mr) * L')
            = acc * L * (Real.exp (-mr) * L' / (Real.exp (-mr) * L')) from by ring]
        rw [div_self (mul_ne_zero hexpne hL'ne), mul_one, hacc]
      rw [e1, e2, hT'd]
    have hL'bot : m' = ⊥ → L' = 0 := fun h => absurd h (by rw [hmr]; simp)
    have hT'bot : m' = ⊥ → T' = 0 := fun h => absurd h (by rw [hmr]; simp)
    have hL'ne0 : 0 < L' → l' ≠ 0 := fun _ => hl'ne
    have step := ih m' l' acc' T' L' hL'pos (fun _ _ => trivial)
      (by rw [hl'eq, hκm']) hacc'eq hL'bot hT'bot hL'ne0
    -- rewrite the goal's fold/sum to match step
    simpa [List.foldl_cons, osNormStepBot, hm', hαd, hpd, hl'd, hacc'd,
      List.map_cons, List.sum_cons, hL'd, hT'd, add_assoc, add_comm, add_left_comm,
      mul_comm, mul_left_comm] using step

/-! ### Full-window readback: the loop's `acc` is the genuine closed form

The kernel's `tl.where(mask, qk, −inf)` makes future keys carry softmax weight
exactly `0`, so they are inert in both numerator and denominator. The fold the
loop realizes is therefore `osNormStepBot` over the *active-key* list (causal
`j ≤ gi`, value `ctxVTileMG`); its final `acc` is the genuine boundary-masked
closed form `contextAttnNopadExactFoldMG` (assembled in
`ctxNopad_fold_eq_exactFoldMG`). This subsection provides the generic
filterMap/sum helpers that reduction uses. -/

/-- filterMap-sum over `Fin n` with a guard collapses into the masked `Finset.sum`. -/
theorem ctxNopad_filterMap_finRange_sum {α : Type*} (n : Nat)
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

/-! ### Block-windowed key lists (per-block invariant advance)

Mirror of #307's `srKeysUpto`/`srBlock`/`srKeysUpto_succ`/`srStateBot_succ`,
adapted to nopad's *causal* per-key filter. The streamed prefix after `c` blocks
is `ctxNopadKeysUptoG … (c·BLK)` — the causal-and-window key list — and one loop
iteration appends `nopadBlockMG c` (the keys in `[c·BLK, (c+1)·BLK)` that are causal
for row `i`). -/

/-- Threshold-split for a `.val`-ascending `Fin` list under a causal-and-window
guard: `j.val < hi₂` window splits into `< t` prefix and `t ≤ j.val < hi₂` block. -/
private theorem nopad_filterMap_window_split {n : Nat} (l : List (Fin n))
    (hsorted : l.Pairwise (fun a b => a.val < b.val))
    (gi t hi₂ : Nat) (g : Fin n → ℝ × ℝ) (hle : t ≤ hi₂) :
    l.filterMap (fun j => if j.val ≤ gi ∧ j.val < hi₂ then some (g j) else none)
      = l.filterMap (fun j => if j.val ≤ gi ∧ j.val < t then some (g j) else none)
        ++ l.filterMap (fun j => if j.val ≤ gi ∧ t ≤ j.val ∧ j.val < hi₂ then some (g j) else none) := by
  induction l with
  | nil => simp
  | cons a tl ih =>
    have htl : tl.Pairwise (fun x y => x.val < y.val) := (List.pairwise_cons.mp hsorted).2
    have hahead : ∀ b ∈ tl, a.val < b.val := (List.pairwise_cons.mp hsorted).1
    rw [List.filterMap_cons, List.filterMap_cons, List.filterMap_cons]
    by_cases hca : a.val ≤ gi
    · by_cases hlt : a.val < t
      · rw [ih htl]
        rw [if_pos ⟨hca, lt_of_lt_of_le hlt hle⟩, if_pos ⟨hca, hlt⟩,
          if_neg (fun h : a.val ≤ gi ∧ t ≤ a.val ∧ a.val < hi₂ => by omega)]
        rfl
      · have hge : t ≤ a.val := Nat.not_lt.mp hlt
        have htail_prefix : tl.filterMap (fun j => if j.val ≤ gi ∧ j.val < t then some (g j) else none) = [] := by
          apply List.filterMap_eq_nil_iff.mpr
          intro b hb
          have := hahead b hb
          rw [if_neg (fun h : b.val ≤ gi ∧ b.val < t => by omega)]
        rw [ih htl, htail_prefix, if_neg (fun h : a.val ≤ gi ∧ a.val < t => by omega)]
        by_cases h2 : a.val < hi₂
        · rw [if_pos ⟨hca, h2⟩, if_pos ⟨hca, hge, h2⟩]; rfl
        · rw [if_neg (fun h => h2 h.2), if_neg (fun h => h2 h.2.2)]
    · rw [if_neg (fun h => hca h.1), if_neg (fun h => hca h.1), if_neg (fun h => hca h.1)]
      rw [ih htl]

/-! ### 128-lane ↔ `nopadBlockMG` reduction bridges (per row `i` / channel `d`)

Mirror of #307's `srBlock_sup_eq`/`srBlock_esum_sum`/`srBlock_acc_sum`, in 2D and
under nopad's *causal* per-lane guard. The loop body reduces a `Fin 128` masked
row (lane `jL` ↦ global key `c·128 + jL`, causal-visible when `c·128 + jL ≤ gi`
and in-window `c·128 + jL < S`); the `osNormStepBot` math uses `nopadBlockMG`'s
`Fin S` filterMap over the causal window `[c·128, (c+1)·128)`. -/

/-- filterMap-sum over `Fin n` with a guard collapses into the masked
`Finset.sum` (2D copy of `ctxNopad_filterMap_finRange_sum`, kept local). -/
private theorem nopad_filterMap_finRange_sum {α : Type*} (n : Nat)
    (p : Fin n → Prop) [DecidablePred p] (g : Fin n → α) (h : α → ℝ) :
    (((List.finRange n).filterMap (fun j => if p j then some (g j) else none)).map h).sum
      = ∑ j : Fin n, if p j then h (g j) else 0 :=
  ctxNopad_filterMap_finRange_sum n p g h

/-- The `WithBot` `foldr` of a guarded score list (coerced) equals the
`Finset.sup` over `Fin n` of the lane terms (`⊥` on filtered-out lanes). 2D copy
of #307's `sr_filterMap_foldr_sup`. -/
private theorem nopad_filterMap_foldr_sup (n : Nat) (P : Fin n → Prop) [DecidablePred P]
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

/-! ### Loop invariant and exec-side stepping

`nopadInvariantG … c s` states that, after streaming `c` `BLOCK_N = BLK`-blocks, the
live `[BLK]` `m_i`/`l_i` vectors and the `[BLK,DM]` `acc` matrix hold (per row `i`,
channel `d`) the `osNormStepBot` fold of `ctxNopadKeysUptoG … (c·BLK)` from the
kernel seed `(⊥, 0, 0)` — and every preLoop-seeded register is preserved. The fold's
`.1` (running max) / `.2.1` (running denom) are channel-independent; `.2.2` is the
running normalized ratio for channel `d`. -/

/-- `realExp` always returns `some`; restate it as `some (·.unbotD 0)`. -/
theorem nopad_realExp_eq_some_unbotD (x : WithBot ℝ) :
    WithBot.realExp x = some ((WithBot.realExp x).unbotD 0) := by
  cases x <;> rfl

/-- The running `max` of an `osNormStepBot` block fold (from a coerced-real state)
is `m ⊔ blockSup`. -/
theorem osNormStepBot_block_fst (m : WithBot ℝ) (l acc : ℝ) (block : List (ℝ × ℝ)) :
    (block.foldl osNormStepBot (m, l, acc)).1
      = m ⊔ (block.map (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥ := by
  rw [osNormStepBot_foldl_fst]
  induction block generalizing m with
  | nil => simp
  | cons a t ih =>
    simp only [List.map_cons, List.foldl_cons, List.foldr_cons]
    rw [ih]
    rw [show (m ⊔ ((a.1 : ℝ) : WithBot ℝ)) ⊔ (List.foldr (· ⊔ ·) ⊥ (List.map (fun p => ((p.1 : ℝ) : WithBot ℝ)) t))
          = m ⊔ (((a.1 : ℝ) : WithBot ℝ) ⊔ (List.foldr (· ⊔ ·) ⊥ (List.map (fun p => ((p.1 : ℝ) : WithBot ℝ)) t))) from by
      rw [sup_assoc]]

/-- The coerced-score `⊔`-foldr is `⊥` iff the list is empty (coerced reals ≠ `⊥`). -/
theorem foldr_sup_coe_bot_iff (xs : List (ℝ × ℝ)) :
    (xs.map (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥ = ⊥ ↔ xs = [] := by
  induction xs with
  | nil => simp
  | cons a t ih =>
    simp only [List.map_cons, List.foldr_cons]
    constructor
    · intro h
      exact absurd (le_bot_iff.mp (h ▸ le_sup_left)) WithBot.coe_ne_bot
    · intro h; exact absurd h (by simp)

/-- `lij`-style block sum rescaled by `β = exp(blockSup ⊖ M')` collapses to
`exp(−Mr)·Σ exp(s)·g` where `Mr = M'.unbotD 0` (empty-block case: both sides `0`).
Auxiliary for `osNormStepBot_block_eq`. -/
private theorem osNormStepBot_blockSum_rescale (block : List (ℝ × ℝ)) (Mr : ℝ)
    (g : ℝ × ℝ → ℝ) :
    let blockSup := (block.map (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥
    (WithBot.realExp (WithBot.realSub blockSup ((Mr : ℝ) : WithBot ℝ))).unbotD 0
        * (block.map (fun p => Real.exp (p.1 - blockSup.unbotD 0) * g p)).sum
      = Real.exp (-Mr) * (block.map (fun p => Real.exp p.1 * g p)).sum := by
  intro blockSup
  cases hbs : blockSup with
  | bot =>
    have hempty : block = [] := by
      rcases block with _ | ⟨a, t⟩
      · rfl
      · exfalso
        have hle : ((a.1 : ℝ) : WithBot ℝ) ≤ blockSup := by
          simp only [blockSup, List.map_cons, List.foldr_cons]; exact le_sup_left
        rw [hbs] at hle
        exact absurd (le_bot_iff.mp hle) (WithBot.coe_ne_bot)
    subst hempty; simp
  | coe bsr =>
    have hbsunbot : blockSup.unbotD 0 = bsr := by rw [hbs]; rfl
    simp only [hbs, WithBot.realSub_coe_coe, WithBot.realExp_coe, WithBot.unbotD_coe, hbsunbot]
    rw [show (block.map (fun p => Real.exp (p.1 - bsr) * g p)).sum
          = Real.exp (-bsr) * (block.map (fun p => Real.exp p.1 * g p)).sum from by
      rw [← List.sum_map_mul_left]
      congr 1; apply List.map_congr_left; intro p _
      rw [show p.1 - bsr = -bsr + p.1 from by ring, Real.exp_add]; ring]
    rw [show Real.exp (bsr - Mr) * (Real.exp (-bsr) * (block.map (fun p => Real.exp p.1 * g p)).sum)
          = (Real.exp (bsr - Mr) * Real.exp (-bsr)) * (block.map (fun p => Real.exp p.1 * g p)).sum from by ring,
      ← Real.exp_add]
    ring_nf

/-- **The block-at-once *normalized* update equals the key-by-key `osNormStepBot`
fold.** Mirror of #307's `srOsStepBot_block_eq` for nopad's *in-loop normalized*
recurrence (the kernel rescales `acc` by `l/l'·α` and adds `(exp/l')·v`, so `acc`
already holds the running ratio). Given a state `(m, ↑l, ↑acc)` anchored to the true
denominator `L` (`l = κ(m)·L`) and accumulator `T` (`acc·L = T`, with `acc = 0` when
`L = 0`), and block max `M' = m ⊔ blockSup`, the kernel's one-shot update — block
denominator `l_ij = Σ exp(s − blockSup)`, `m_i_new = M'`, `α = exp(m ⊖ M')`,
`β = exp(blockSup ⊖ M')`, `l_i_new = α·l + β·l_ij`, and
`acc' = acc·(l/l_i_new·α) + Σ (exp(s−blockSup)·β/l_i_new)·v` — lands on the coerced
`block.foldl osNormStepBot (m, l, acc)`. -/
theorem osNormStepBot_block_eq (m : WithBot ℝ) (l acc T L : ℝ) (block : List (ℝ × ℝ))
    (hbne : block ≠ [])
    (hL0 : 0 ≤ L)
    (hl : l = (m.elim 0 (fun r => Real.exp (-r))) * L)
    (hacc : acc * L = T)
    (hmL : m = ⊥ → L = 0) (hmT : m = ⊥ → T = 0)
    (hLpos : 0 < L → l ≠ 0) :
    let blockSup := (block.map (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥
    let M' := m ⊔ blockSup
    let α := (WithBot.realExp (WithBot.realSub m M')).unbotD 0
    let β := (WithBot.realExp (WithBot.realSub blockSup M')).unbotD 0
    let lij := (block.map (fun p => Real.exp (p.1 - blockSup.unbotD 0))).sum
    let l' := α * l + β * lij
    let acc' := acc * (l / l' * α)
      + (block.map (fun p => Real.exp (p.1 - blockSup.unbotD 0) * β / l' * p.2)).sum
    (M', l', acc') = block.foldl osNormStepBot (m, l, acc) := by
  intro blockSup M' α β lij l' acc'
  have hfst : (block.foldl osNormStepBot (m, l, acc)).1 = M' := by
    rw [osNormStepBot_block_fst]
  obtain ⟨hfold_l, hfold_acc, _hL'0, _hbot1, _hbot2, _hne⟩ :=
    osNormStepBot_foldl_consistent block m l acc T L hL0 (fun _ _ => trivial) hl hacc hmL hmT hLpos
  rw [hfst] at hfold_l
  set L'b := L + (block.map (fun p => Real.exp p.1)).sum with hL'b
  set T'b := T + (block.map (fun p => Real.exp p.1 * p.2)).sum with hT'b
  -- α·l = exp(-Mr)·L for any finite Mr = M'.unbotD 0 (uses anchor `l = κ(m)·L`)
  have hαl : ∀ Mr : ℝ, M' = (Mr : WithBot ℝ) → l * α = Real.exp (-Mr) * L := by
    intro Mr hM'
    cases hm : m with
    | bot => have : L = 0 := hmL hm; rw [hl, hm]; simp [this]
    | coe a =>
      have hαv : α = Real.exp (a - Mr) := by
        simp only [α, hm, hM', WithBot.realSub_coe_coe, WithBot.realExp_coe, WithBot.unbotD_coe]
      rw [hαv, hl, hm, show ((↑a : WithBot ℝ).elim 0 (fun r => Real.exp (-r))) = Real.exp (-a) from rfl]
      rw [show Real.exp (-a) * L * Real.exp (a - Mr) = (Real.exp (-a) * Real.exp (a - Mr)) * L from by ring,
        ← Real.exp_add]
      ring_nf
  -- block ≠ [] ⟹ blockSup ≠ ⊥ ⟹ M' ≠ ⊥
  have hbsne : blockSup ≠ ⊥ := fun h => hbne ((foldr_sup_coe_bot_iff block).mp h)
  cases hM' : M' with
  | bot =>
    exact absurd (le_bot_iff.mp (hM' ▸ (le_sup_right : blockSup ≤ M'))) hbsne
  | coe Mr =>
    -- l' = exp(-Mr)·L'b
    have hl'eq : l' = Real.exp (-Mr) * L'b := by
      show α * l + β * lij = Real.exp (-Mr) * L'b
      have h1 : α * l = Real.exp (-Mr) * L := by rw [mul_comm]; exact hαl Mr hM'
      have h2 : β * lij = Real.exp (-Mr) * (block.map (fun p => Real.exp p.1)).sum := by
        have hr := osNormStepBot_blockSum_rescale block Mr (fun _ => 1)
        simp only [mul_one] at hr
        rw [show β = (WithBot.realExp (WithBot.realSub blockSup ((Mr:ℝ):WithBot ℝ))).unbotD 0 from by simp only [β, hM']]
        rw [show lij = (block.map (fun p => Real.exp (p.1 - blockSup.unbotD 0))).sum from rfl]
        exact hr
      rw [h1, h2, hL'b]; ring
    have hL'pos : 0 < L'b := by
      rw [hL'b]
      have hsumpos : 0 < (block.map (fun p => Real.exp p.1)).sum := by
        rcases block with _ | ⟨a, t⟩
        · exact absurd rfl hbne
        · rw [List.map_cons, List.sum_cons]
          have h1 : 0 ≤ (t.map (fun p => Real.exp p.1)).sum := by
            apply List.sum_nonneg; intro x hx
            simp only [List.mem_map] at hx; obtain ⟨p, _, rfl⟩ := hx; exact le_of_lt (Real.exp_pos _)
          have := Real.exp_pos a.1; linarith
      linarith
    have hl'ne : l' ≠ 0 := by rw [hl'eq]; positivity
    refine Prod.ext (hfst.trans hM').symm (Prod.ext ?_ ?_)
    · show l' = (block.foldl osNormStepBot (m, l, acc)).2.1
      rw [hM'] at hfold_l
      rw [hfold_l, hl'eq, show ((↑Mr : WithBot ℝ).elim 0 (fun r => Real.exp (-r))) = Real.exp (-Mr) from rfl]
    · show acc' = (block.foldl osNormStepBot (m, l, acc)).2.2
      have hfacc : (block.foldl osNormStepBot (m, l, acc)).2.2 = T'b / L'b := by
        rw [eq_div_iff (ne_of_gt hL'pos), hfold_acc]
      rw [hfacc]
      show acc * (l / l' * α) + (block.map (fun p => Real.exp (p.1 - blockSup.unbotD 0) * β / l' * p.2)).sum
        = T'b / L'b
      have hβsum : (block.map (fun p => Real.exp (p.1 - blockSup.unbotD 0) * β / l' * p.2)).sum
          = (Real.exp (-Mr) * (block.map (fun p => Real.exp p.1 * p.2)).sum) / l' := by
        have hmapeq : (block.map (fun p => Real.exp (p.1 - blockSup.unbotD 0) * β / l' * p.2))
            = (block.map (fun p => (Real.exp (p.1 - blockSup.unbotD 0) * (β * p.2)) * (1 / l'))) := by
          apply List.map_congr_left; intro p _; ring
        rw [hmapeq, List.sum_map_mul_right, ← div_eq_mul_one_div]
        congr 1
        rw [show (block.map (fun p => Real.exp (p.1 - blockSup.unbotD 0) * (β * p.2))).sum
              = β * (block.map (fun p => Real.exp (p.1 - blockSup.unbotD 0) * p.2)).sum from by
          rw [← List.sum_map_mul_left]; congr 1; apply List.map_congr_left; intro p _; ring]
        rw [show β = (WithBot.realExp (WithBot.realSub blockSup ((Mr:ℝ):WithBot ℝ))).unbotD 0 from by simp only [β, hM']]
        exact osNormStepBot_blockSum_rescale block Mr (fun p => p.2)
      rw [hβsum]
      rw [show acc * (l / l' * α) = acc * (l * α) / l' from by ring, hαl Mr hM', hl'eq, hT'b]
      rw [← add_div]
      rw [show acc * (Real.exp (-Mr) * L) + Real.exp (-Mr) * (block.map (fun p => Real.exp p.1 * p.2)).sum
            = Real.exp (-Mr) * (acc * L + (block.map (fun p => Real.exp p.1 * p.2)).sum) from by ring]
      rw [hacc, mul_div_mul_left _ _ (Real.exp_ne_zero _)]

/-- `seqLen` depends only on `mem`/`pids`. -/
theorem seqLen_eq_of_mem_pids (s s0 : BlockState) (B_Seqlen : RegionName)
    (hmem : s.mem = s0.mem) (hpids : s.pids = s0.pids) :
    seqLen s B_Seqlen = seqLen s0 B_Seqlen := by
  simp only [seqLen, BlockState.readMemValue, BlockState.readMemTyped, hmem, hpids]

/-- `startLoc` depends only on `mem`/`pids`. -/
theorem startLoc_eq_of_mem_pids (s s0 : BlockState) (B_Start_Loc : RegionName)
    (hmem : s.mem = s0.mem) (hpids : s.pids = s0.pids) :
    startLoc s B_Start_Loc = startLoc s0 B_Start_Loc := by
  simp only [startLoc, BlockState.readMemValue, BlockState.readMemTyped, hmem, hpids]

/-! ## PostLoop store and full-kernel exec assembly -/

/-- General coordinate-faithful query tile: row `i` is the global packed row
`B_Start_Loc[cur_batch] + start_m·BLK + i`, channel `e`, contiguous strides
`(rs, hs, 1)` (= `(H·DM, DM, 1)`). -/
noncomputable def ctxQTileG
    (s : BlockState) (Q B_Start_Loc : RegionName) (rs hs BLK DM : Nat) :
    TileIndex [BLK, DM] → ℝ :=
  fun (i, e, _) =>
    s.readMem Q
      ((startLoc s B_Start_Loc + (s.pids 2 * BLK + i.val)) * rs
        + s.pids 1 * hs + e.val)

/-- General coordinate-faithful key tile. -/
noncomputable def ctxKTileG
    (s : BlockState) (K B_Start_Loc : RegionName) (rs hs S DM : Nat) :
    TileIndex [S, DM] → ℝ :=
  fun (j, e, _) =>
    s.readMem K
      ((startLoc s B_Start_Loc + j.val) * rs + s.pids 1 * hs + e.val)

/-- General coordinate-faithful value tile. -/
noncomputable def ctxVTileG
    (s : BlockState) (V B_Start_Loc : RegionName) (rs hs S DM : Nat) :
    TileIndex [S, DM] → ℝ :=
  fun (j, d, _) =>
    s.readMem V
      ((startLoc s B_Start_Loc + j.val) * rs + s.pids 1 * hs + d.val)

/-- General sequence-length-masked key tile. -/
noncomputable def ctxKTileMG
    (s : BlockState) (K B_Start_Loc : RegionName) (rs hs S DM bel : Nat) :
    TileIndex [S, DM] → ℝ :=
  fun (j, e, u) => if j.val < bel then ctxKTileG s K B_Start_Loc rs hs S DM (j, e, u) else 0

/-- General sequence-length-masked value tile. -/
noncomputable def ctxVTileMG
    (s : BlockState) (V B_Start_Loc : RegionName) (rs hs S DM bel : Nat) :
    TileIndex [S, DM] → ℝ :=
  fun (j, d, u) => if j.val < bel then ctxVTileG s V B_Start_Loc rs hs S DM (j, d, u) else 0

/-- General row-masked query tile: `ctxQTileG` on active rows (`gi < bel`), else `0`. -/
noncomputable def ctxQTileMRowG
    (s : BlockState) (Q B_Start_Loc : RegionName) (rs hs BLK DM bel : Nat) :
    TileIndex [BLK, DM] → ℝ :=
  fun (i, e, u) =>
    if s.pids 2 * BLK + i.val < bel then ctxQTileG s Q B_Start_Loc rs hs BLK DM (i, e, u) else 0

/-- General boundary-masked causal-softmax fold (the faithful kernel value). -/
noncomputable def contextAttnNopadExactFoldMG
    (s : BlockState) (Q K V B_Start_Loc : RegionName)
    (sm_scale : ℝ) (rs hs BLK DM S bel : Nat) (idx : TileIndex [BLK, DM]) : ℝ :=
  let i := idx.1
  let d := idx.2.1
  let gi := s.pids 2 * BLK + i.val
  let raw := fun j : Fin S =>
    Finset.univ.sum (fun e : Fin DM =>
      ctxQTileG s Q B_Start_Loc rs hs BLK DM (i, e, PUnit.unit)
        * ctxKTileMG s K B_Start_Loc rs hs S DM bel (j, e, PUnit.unit))
  let weight := fun j : Fin S =>
    if j.val ≤ gi then Real.exp (sm_scale * raw j) else 0
  let denom := Finset.univ.sum (fun j : Fin S => weight j)
  let numer := Finset.univ.sum (fun j : Fin S =>
    weight j * ctxVTileMG s V B_Start_Loc rs hs S DM bel (j, d, PUnit.unit))
  numer / denom

/-- General kernel-decoded streamed window `S = block_mask·(start_m+1)·BLK`. -/
def ctxNopadWindowG (s : BlockState) (B_Seqlen : RegionName) (BLK : Nat) : Nat :=
  let sl := seqLen s B_Seqlen
  let bm := if BLK * s.pids 2 < sl then 1 else 0
  bm * (s.pids 2 + 1) * BLK

/-- General genuine closed-form output value (boundary-masked causal-softmax fold). -/
noncomputable def ctxNopadGenuineOutValueG
    (s : BlockState) (Q K V B_Start_Loc B_Seqlen : RegionName)
    (sm_scale : ℝ) (rs hs BLK DM : Nat) (idx : TileIndex [BLK, DM]) : ℝ :=
  contextAttnNopadExactFoldMG s Q K V B_Start_Loc sm_scale rs hs BLK DM
    (ctxNopadWindowG s B_Seqlen BLK) (ctxNopadBel s B_Seqlen) idx

/-! ### General key lists, blocks, and the masked fold -/

/-- General active-key `(score, value)` list (causal `j ≤ gi`). -/
noncomputable def ctxNopadKeyListG
    (s : BlockState) (Q K V B_Start_Loc : RegionName) (sm_scale : ℝ)
    (rs hs BLK DM S bel : Nat) (i : Fin BLK) (d : Fin DM) : List (ℝ × ℝ) :=
  let gi := s.pids 2 * BLK + i.val
  (List.finRange S).filterMap (fun j : Fin S =>
    if j.val ≤ gi then
      some (sm_scale * Finset.univ.sum (fun e : Fin DM =>
              ctxQTileG s Q B_Start_Loc rs hs BLK DM (i, e, PUnit.unit)
                * ctxKTileMG s K B_Start_Loc rs hs S DM bel (j, e, PUnit.unit)),
            ctxVTileMG s V B_Start_Loc rs hs S DM bel (j, d, PUnit.unit))
    else none)

/-- General causal-window key list (causal `j ≤ gi` AND `j < hi`). -/
noncomputable def ctxNopadKeysUptoG
    (s : BlockState) (Q K V B_Start_Loc : RegionName) (sm_scale : ℝ)
    (rs hs BLK DM S bel hi : Nat) (i : Fin BLK) (d : Fin DM) : List (ℝ × ℝ) :=
  let gi := s.pids 2 * BLK + i.val
  (List.finRange S).filterMap (fun j : Fin S =>
    if j.val ≤ gi ∧ j.val < hi then
      some (sm_scale * Finset.univ.sum (fun e : Fin DM =>
              ctxQTileG s Q B_Start_Loc rs hs BLK DM (i, e, PUnit.unit)
                * ctxKTileMG s K B_Start_Loc rs hs S DM bel (j, e, PUnit.unit)),
            ctxVTileMG s V B_Start_Loc rs hs S DM bel (j, d, PUnit.unit))
    else none)

/-- General row-masked causal-window key list. -/
noncomputable def ctxNopadKeysUptoMG
    (s : BlockState) (Q K V B_Start_Loc : RegionName) (sm_scale : ℝ)
    (rs hs BLK DM S bel hi : Nat) (i : Fin BLK) (d : Fin DM) : List (ℝ × ℝ) :=
  let gi := s.pids 2 * BLK + i.val
  (List.finRange S).filterMap (fun j : Fin S =>
    if j.val ≤ gi ∧ j.val < hi then
      some (sm_scale * Finset.univ.sum (fun e : Fin DM =>
              ctxQTileMRowG s Q B_Start_Loc rs hs BLK DM bel (i, e, PUnit.unit)
                * ctxKTileMG s K B_Start_Loc rs hs S DM bel (j, e, PUnit.unit)),
            ctxVTileMG s V B_Start_Loc rs hs S DM bel (j, d, PUnit.unit))
    else none)

/-- General row-masked block-`c` causal key list. -/
noncomputable def nopadBlockMG
    (s : BlockState) (Q K V B_Start_Loc : RegionName) (sm_scale : ℝ)
    (rs hs BLK DM S bel c : Nat) (i : Fin BLK) (d : Fin DM) : List (ℝ × ℝ) :=
  let gi := s.pids 2 * BLK + i.val
  (List.finRange S).filterMap (fun j : Fin S =>
    if j.val ≤ gi ∧ c * BLK ≤ j.val ∧ j.val < (c + 1) * BLK then
      some (sm_scale * Finset.univ.sum (fun e : Fin DM =>
              ctxQTileMRowG s Q B_Start_Loc rs hs BLK DM bel (i, e, PUnit.unit)
                * ctxKTileMG s K B_Start_Loc rs hs S DM bel (j, e, PUnit.unit)),
            ctxVTileMG s V B_Start_Loc rs hs S DM bel (j, d, PUnit.unit))
    else none)

/-- General masked fold of the row-masked causal-window key prefix `[0, hi)`. -/
noncomputable def nopadFoldUptoG
    (s : BlockState) (Q K V B_Start_Loc : RegionName) (sm_scale : ℝ)
    (rs hs BLK DM S bel hi : Nat) (i : Fin BLK) (d : Fin DM) : WithBot ℝ × ℝ × ℝ :=
  (ctxNopadKeysUptoMG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel hi i d).foldl
    osNormStepBot (⊥, 0, 0)

/-- General: full causal key list is the causal-window list at `hi = S`. -/
theorem ctxNopadKeysUptoG_full
    (s : BlockState) (Q K V B_Start_Loc : RegionName) (sm_scale : ℝ)
    (rs hs BLK DM S bel : Nat) (i : Fin BLK) (d : Fin DM) :
    ctxNopadKeysUptoG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel S i d
      = ctxNopadKeyListG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel i d := by
  unfold ctxNopadKeysUptoG ctxNopadKeyListG
  apply List.filterMap_congr
  intro j _
  by_cases hj : j.val ≤ s.pids 2 * BLK + i.val
  · simp [hj, j.isLt]
  · simp [hj]

/-- General masked window split. -/
theorem ctxNopadKeysUptoMG_succ
    (s : BlockState) (Q K V B_Start_Loc : RegionName) (sm_scale : ℝ)
    (rs hs BLK DM S bel c : Nat) (i : Fin BLK) (d : Fin DM) (hBLK : 0 < BLK) :
    ctxNopadKeysUptoMG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel ((c + 1) * BLK) i d
      = ctxNopadKeysUptoMG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel (c * BLK) i d
        ++ nopadBlockMG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel c i d := by
  unfold ctxNopadKeysUptoMG nopadBlockMG
  exact nopad_filterMap_window_split (List.finRange S) (List.pairwise_lt_finRange S)
    (s.pids 2 * BLK + i.val) (c * BLK) ((c + 1) * BLK) _
    (by nlinarith [Nat.zero_le BLK])

/-- General masked causal key list at `hi = 0` is empty. -/
theorem ctxNopadKeysUptoMG_zero
    (s : BlockState) (Q K V B_Start_Loc : RegionName) (sm_scale : ℝ)
    (rs hs BLK DM S bel : Nat) (i : Fin BLK) (d : Fin DM) :
    ctxNopadKeysUptoMG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel 0 i d = [] := by
  unfold ctxNopadKeysUptoMG
  apply List.filterMap_eq_nil_iff.mpr
  intro j _; simp

/-- General: on an active row the masked key list equals the genuine one. -/
theorem ctxNopadKeysUptoMG_active
    (s : BlockState) (Q K V B_Start_Loc : RegionName) (sm_scale : ℝ)
    (rs hs BLK DM S bel hi : Nat) (i : Fin BLK) (d : Fin DM)
    (hact : s.pids 2 * BLK + i.val < bel) :
    ctxNopadKeysUptoMG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel hi i d
      = ctxNopadKeysUptoG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel hi i d := by
  unfold ctxNopadKeysUptoMG ctxNopadKeysUptoG
  apply List.filterMap_congr
  intro j _
  by_cases hj : j.val ≤ s.pids 2 * BLK + i.val ∧ j.val < hi
  · rw [if_pos hj, if_pos hj]
    refine congrArg some ?_
    refine Prod.ext ?_ rfl
    refine congrArg (sm_scale * ·) (Finset.sum_congr rfl (fun e _ => ?_))
    rw [show ctxQTileMRowG s Q B_Start_Loc rs hs BLK DM bel (i, e, PUnit.unit)
          = ctxQTileG s Q B_Start_Loc rs hs BLK DM (i, e, PUnit.unit) from by
      simp only [ctxQTileMRowG, hact, if_true]]
  · rw [if_neg hj, if_neg hj]

/-- General masked one-block advance. -/
theorem nopadFoldUptoMG_succ
    (s : BlockState) (Q K V B_Start_Loc : RegionName) (sm_scale : ℝ)
    (rs hs BLK DM S bel c : Nat) (i : Fin BLK) (d : Fin DM) (hBLK : 0 < BLK) :
    nopadFoldUptoG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel ((c + 1) * BLK) i d
      = (nopadBlockMG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel c i d).foldl osNormStepBot
          (nopadFoldUptoG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel (c * BLK) i d) := by
  rw [nopadFoldUptoG, nopadFoldUptoG, ctxNopadKeysUptoMG_succ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ hBLK,
    List.foldl_append]

/-- General base case: masked fold at `hi = 0` is the kernel seed. -/
theorem nopadFoldUptoG_zero
    (s : BlockState) (Q K V B_Start_Loc : RegionName) (sm_scale : ℝ)
    (rs hs BLK DM S bel : Nat) (i : Fin BLK) (d : Fin DM) :
    nopadFoldUptoG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel 0 i d = (⊥, 0, 0) := by
  rw [nopadFoldUptoG, ctxNopadKeysUptoMG_zero]; rfl

/-- General masked-fold anchor (consistency from the `(⊥,0,0)` seed). -/
theorem nopadFoldUptoG_anchor
    (s : BlockState) (Q K V B_Start_Loc : RegionName) (sm_scale : ℝ)
    (rs hs BLK DM S bel hi : Nat) (i : Fin BLK) (d : Fin DM) :
    let st := nopadFoldUptoG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel hi i d
    let L := ((ctxNopadKeysUptoMG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel hi i d).map
                (fun p => Real.exp p.1)).sum
    let T := ((ctxNopadKeysUptoMG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel hi i d).map
                (fun p => Real.exp p.1 * p.2)).sum
    st.2.1 = (st.1.elim 0 (fun r => Real.exp (-r))) * L
      ∧ st.2.2 * L = T
      ∧ 0 ≤ L
      ∧ (st.1 = ⊥ → L = 0)
      ∧ (st.1 = ⊥ → T = 0)
      ∧ (0 < L → st.2.1 ≠ 0) := by
  have h := osNormStepBot_foldl_consistent
    (ctxNopadKeysUptoMG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel hi i d) ⊥ 0 0 0 0
    (le_refl 0) (fun _ _ => trivial) (by simp) (by ring) (fun _ => rfl) (fun _ => rfl)
    (by intro h; exact absurd h (lt_irrefl 0))
  simpa only [nopadFoldUptoG, zero_add] using h

/-- General channel-independence of the masked fold's `.1`/`.2.1`. -/
theorem nopadFoldUptoG_channel_indep
    (s : BlockState) (Q K V B_Start_Loc : RegionName) (sm_scale : ℝ)
    (rs hs BLK DM S bel hi : Nat) (i : Fin BLK) (d d' : Fin DM) :
    (nopadFoldUptoG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel hi i d).1
        = (nopadFoldUptoG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel hi i d').1
      ∧ (nopadFoldUptoG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel hi i d).2.1
          = (nopadFoldUptoG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel hi i d').2.1 := by
  have hkeys : (ctxNopadKeysUptoMG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel hi i d).map Prod.fst
      = (ctxNopadKeysUptoMG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel hi i d').map Prod.fst := by
    unfold ctxNopadKeysUptoMG
    rw [List.map_filterMap, List.map_filterMap]
    apply List.filterMap_congr
    intro j _
    by_cases hj : j.val ≤ s.pids 2 * BLK + i.val ∧ j.val < hi <;> simp [hj]
  have hfst : (nopadFoldUptoG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel hi i d).1
      = (nopadFoldUptoG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel hi i d').1 := by
    rw [nopadFoldUptoG, nopadFoldUptoG, osNormStepBot_foldl_fst, osNormStepBot_foldl_fst]
    congr 1
    rw [show (ctxNopadKeysUptoMG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel hi i d).map
            (fun p => ((p.1 : ℝ) : WithBot ℝ))
          = ((ctxNopadKeysUptoMG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel hi i d).map Prod.fst).map
            (fun r : ℝ => ((r : ℝ) : WithBot ℝ)) from by rw [List.map_map]; rfl,
        hkeys, List.map_map]
    rfl
  refine ⟨hfst, ?_⟩
  have hden : ∀ dd : Fin DM, (nopadFoldUptoG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel hi i dd).2.1
      = ((nopadFoldUptoG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel hi i dd).1.elim 0 (fun r => Real.exp (-r)))
        * ((ctxNopadKeysUptoMG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel hi i dd).map
            (fun p => Real.exp p.1)).sum := by
    intro dd
    have h := (osNormStepBot_foldl_consistent
      (ctxNopadKeysUptoMG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel hi i dd) ⊥ 0 0 0 0
      (le_refl 0) (fun _ _ => trivial) (by simp) (by ring) (fun _ => rfl) (fun _ => rfl)
      (by intro h; exact absurd h (lt_irrefl 0))).1
    rw [nopadFoldUptoG]
    rw [show (List.foldl osNormStepBot (⊥, 0, 0)
          (ctxNopadKeysUptoMG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel hi i dd)).2.1 = _ from h]
    rw [zero_add]
  rw [hden d, hden d', hfst]
  congr 2
  rw [show (ctxNopadKeysUptoMG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel hi i d).map
          (fun p => Real.exp p.1)
        = ((ctxNopadKeysUptoMG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel hi i d).map Prod.fst).map
          Real.exp from by rw [List.map_map]; rfl,
      hkeys, List.map_map]
  rfl

/-- General block score-list channel-independence. -/
theorem nopadBlockMG_fst_channel_indep
    (s : BlockState) (Q K V B_Start_Loc : RegionName) (sm_scale : ℝ)
    (rs hs BLK DM S bel c : Nat) (i : Fin BLK) (d d' : Fin DM) :
    (nopadBlockMG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel c i d).map Prod.fst
      = (nopadBlockMG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel c i d').map Prod.fst := by
  unfold nopadBlockMG
  rw [List.map_filterMap, List.map_filterMap]
  apply List.filterMap_congr
  intro j _
  by_cases hj : j.val ≤ s.pids 2 * BLK + i.val ∧ c * BLK ≤ j.val ∧ j.val < (c + 1) * BLK <;> simp [hj]

/-- General: block `c` is non-empty when in-window and causal-active. -/
theorem nopadBlockMG_ne_nil
    (s : BlockState) (Q K V B_Start_Loc : RegionName) (sm_scale : ℝ)
    (rs hs BLK DM S bel c : Nat) (i : Fin BLK) (d : Fin DM)
    (hwin : (c + 1) * BLK ≤ S) (hcle : c * BLK ≤ s.pids 2 * BLK + i.val) (hBLK : 0 < BLK) :
    nopadBlockMG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel c i d ≠ [] := by
  intro hnil
  have h0 : (⟨c * BLK, by nlinarith [Nat.zero_le BLK]⟩ : Fin S) ∈ List.finRange S := List.mem_finRange _
  have := List.filterMap_eq_nil_iff.mp hnil (⟨c * BLK, by nlinarith [Nat.zero_le BLK]⟩ : Fin S) h0
  simp only [Fin.val_mk] at this
  rw [if_pos ⟨by omega, by omega, by nlinarith⟩] at this
  exact absurd this (by simp)

/-- General lane bound: every block-`c` lane is in-window. -/
private theorem nopad_lane_lt_SG (c S BLK : Nat) (hwin : (c + 1) * BLK ≤ S) (jL : Fin BLK) :
    c * BLK + jL.val < S := by have := jL.isLt; nlinarith [Nat.zero_le BLK]

/-- General windowed `Finset.sup` reindex (`Fin S` causal-window → `Fin BLK` lane). -/
private theorem nopad_window_sup_reindexG (c S gi BLK : Nat) (hwin : (c + 1) * BLK ≤ S)
    (F : Nat → WithBot ℝ) :
    Finset.univ.sup (fun j : Fin S =>
        if j.val ≤ gi ∧ c * BLK ≤ j.val ∧ j.val < (c + 1) * BLK then F j.val else ⊥)
      = Finset.univ.sup (fun jL : Fin BLK =>
          if c * BLK + jL.val ≤ gi then F (c * BLK + jL.val) else ⊥) := by
  apply le_antisymm
  · apply Finset.sup_le
    intro j _
    by_cases hj : j.val ≤ gi ∧ c * BLK ≤ j.val ∧ j.val < (c + 1) * BLK
    · rw [if_pos hj]
      have hjL : j.val - c * BLK < BLK := by
        have : j.val < c * BLK + BLK := by have := hj.2.2; nlinarith [Nat.zero_le BLK]
        omega
      refine le_trans ?_ (Finset.le_sup
        (f := fun jL : Fin BLK => if c * BLK + jL.val ≤ gi then F (c * BLK + jL.val) else ⊥)
        (Finset.mem_univ (⟨j.val - c * BLK, hjL⟩ : Fin BLK)))
      simp only
      rw [if_pos (by show c * BLK + (j.val - c * BLK) ≤ gi; omega),
        show c * BLK + (j.val - c * BLK) = j.val from by omega]
    · rw [if_neg hj]; exact bot_le
  · apply Finset.sup_le
    intro jL _
    have hb : c * BLK + jL.val < S := nopad_lane_lt_SG c S BLK hwin jL
    by_cases hc : c * BLK + jL.val ≤ gi
    · rw [if_pos hc]
      refine le_trans ?_ (Finset.le_sup
        (f := fun j : Fin S =>
          if j.val ≤ gi ∧ c * BLK ≤ j.val ∧ j.val < (c + 1) * BLK then F j.val else ⊥)
        (Finset.mem_univ (⟨c * BLK + jL.val, hb⟩ : Fin S)))
      simp only
      rw [if_pos (by have := jL.isLt; exact ⟨hc, by omega, by nlinarith⟩)]
    · rw [if_neg hc]; exact bot_le

/-- General `nopadBlockMG` map-and-sum bridge. -/
theorem nopadBlockMG_map_sum
    (s : BlockState) (Q K V B_Start_Loc : RegionName) (sm_scale : ℝ)
    (rs hs BLK DM S bel c : Nat) (i : Fin BLK) (d : Fin DM)
    (hwin : (c + 1) * BLK ≤ S) (h : ℝ × ℝ → ℝ) :
    ((nopadBlockMG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel c i d).map h).sum
      = ∑ jL : Fin BLK,
          (if c * BLK + jL.val ≤ s.pids 2 * BLK + i.val ∧ c * BLK + jL.val < S then
            h (sm_scale * Finset.univ.sum (fun e : Fin DM =>
                  ctxQTileMRowG s Q B_Start_Loc rs hs BLK DM bel (i, e, PUnit.unit)
                    * ctxKTileMG s K B_Start_Loc rs hs S DM bel (⟨c * BLK + jL.val, by have := jL.isLt; nlinarith [Nat.zero_le BLK]⟩, e, PUnit.unit)),
                ctxVTileMG s V B_Start_Loc rs hs S DM bel (⟨c * BLK + jL.val, by have := jL.isLt; nlinarith [Nat.zero_le BLK]⟩, d, PUnit.unit))
           else 0) := by
  classical
  rw [nopadBlockMG, nopad_filterMap_finRange_sum S
    (fun j : Fin S => j.val ≤ s.pids 2 * BLK + i.val ∧ c * BLK ≤ j.val ∧ j.val < (c + 1) * BLK)
    (fun j => (sm_scale * Finset.univ.sum (fun e : Fin DM =>
                ctxQTileMRowG s Q B_Start_Loc rs hs BLK DM bel (i, e, PUnit.unit)
                  * ctxKTileMG s K B_Start_Loc rs hs S DM bel (j, e, PUnit.unit)),
              ctxVTileMG s V B_Start_Loc rs hs S DM bel (j, d, PUnit.unit))) h]
  rw [← Finset.sum_filter
        (fun j : Fin S => j.val ≤ s.pids 2 * BLK + i.val ∧ c * BLK ≤ j.val ∧ j.val < (c + 1) * BLK)
        (fun j => h (sm_scale * Finset.univ.sum (fun e : Fin DM =>
                ctxQTileMRowG s Q B_Start_Loc rs hs BLK DM bel (i, e, PUnit.unit)
                  * ctxKTileMG s K B_Start_Loc rs hs S DM bel (j, e, PUnit.unit)),
              ctxVTileMG s V B_Start_Loc rs hs S DM bel (j, d, PUnit.unit)))]
  symm
  rw [← Finset.sum_filter
        (fun jL : Fin BLK => c * BLK + jL.val ≤ s.pids 2 * BLK + i.val ∧ c * BLK + jL.val < S)
        (fun jL => h (sm_scale * Finset.univ.sum (fun e : Fin DM =>
                ctxQTileMRowG s Q B_Start_Loc rs hs BLK DM bel (i, e, PUnit.unit)
                  * ctxKTileMG s K B_Start_Loc rs hs S DM bel (⟨c * BLK + jL.val, by have := jL.isLt; nlinarith [Nat.zero_le BLK]⟩, e, PUnit.unit)),
              ctxVTileMG s V B_Start_Loc rs hs S DM bel (⟨c * BLK + jL.val, by have := jL.isLt; nlinarith [Nat.zero_le BLK]⟩, d, PUnit.unit)))]
  refine Finset.sum_bij
    (i := fun jL (_ : jL ∈ Finset.univ.filter
        (fun jL : Fin BLK => c * BLK + jL.val ≤ s.pids 2 * BLK + i.val ∧ c * BLK + jL.val < S)) =>
      (⟨c * BLK + jL.val, by have := jL.isLt; nlinarith [Nat.zero_le BLK]⟩ : Fin S)) ?_ ?_ ?_ ?_
  · intro jL hjL
    have hmem := (Finset.mem_filter.mp hjL).2
    rw [Finset.mem_filter]
    refine ⟨Finset.mem_univ _, ?_⟩
    show c * BLK + jL.val ≤ s.pids 2 * BLK + i.val ∧ c * BLK ≤ c * BLK + jL.val ∧ c * BLK + jL.val < (c + 1) * BLK
    have := jL.isLt; exact ⟨hmem.1, by omega, by nlinarith⟩
  · intro a _ b _ hab
    apply Fin.ext
    have : c * BLK + a.val = c * BLK + b.val := by simpa using congrArg Fin.val hab
    omega
  · intro j hj
    have hj2 : j.val ≤ s.pids 2 * BLK + i.val ∧ c * BLK ≤ j.val ∧ j.val < (c + 1) * BLK := (Finset.mem_filter.mp hj).2
    have hjlt : j.val < c * BLK + BLK := by have := hj2.2.2; nlinarith [Nat.zero_le BLK]
    refine ⟨⟨j.val - c * BLK, by omega⟩, ?_, by apply Fin.ext; simp only; omega⟩
    rw [Finset.mem_filter]
    refine ⟨Finset.mem_univ _, ?_⟩
    show c * BLK + (j.val - c * BLK) ≤ s.pids 2 * BLK + i.val ∧ c * BLK + (j.val - c * BLK) < S
    have := j.isLt; constructor <;> omega
  · intro jL _; rfl

/-- General `nopadBlockMG` `l_ij` lane-sum bridge. -/
theorem nopadBlockMG_lij_sum
    (s : BlockState) (Q K V B_Start_Loc : RegionName) (sm_scale : ℝ)
    (rs hs BLK DM S bel c : Nat) (i : Fin BLK) (d : Fin DM) (Mr : ℝ)
    (hwin : (c + 1) * BLK ≤ S) :
    (∑ jL : Fin BLK, WithBot.realExp (WithBot.realSub
        (if c * BLK + jL.val ≤ s.pids 2 * BLK + i.val then
          ((sm_scale * Finset.univ.sum (fun e : Fin DM =>
              ctxQTileMRowG s Q B_Start_Loc rs hs BLK DM bel (i, e, PUnit.unit)
                * ctxKTileMG s K B_Start_Loc rs hs S DM bel
                    (⟨c * BLK + jL.val, nopad_lane_lt_SG c S BLK hwin jL⟩, e, PUnit.unit)) : ℝ) : WithBot ℝ)
         else (⊥ : WithBot ℝ)) ((Mr : ℝ) : WithBot ℝ)))
      = some ((nopadBlockMG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel c i d).map
          (fun p => Real.exp (p.1 - Mr))).sum := by
  have hcell : ∀ jL : Fin BLK,
      WithBot.realExp (WithBot.realSub
        (if c * BLK + jL.val ≤ s.pids 2 * BLK + i.val then
          ((sm_scale * Finset.univ.sum (fun e : Fin DM =>
              ctxQTileMRowG s Q B_Start_Loc rs hs BLK DM bel (i, e, PUnit.unit)
                * ctxKTileMG s K B_Start_Loc rs hs S DM bel
                    (⟨c * BLK + jL.val, nopad_lane_lt_SG c S BLK hwin jL⟩, e, PUnit.unit)) : ℝ) : WithBot ℝ)
         else (⊥ : WithBot ℝ)) ((Mr : ℝ) : WithBot ℝ))
        = some (if c * BLK + jL.val ≤ s.pids 2 * BLK + i.val ∧ c * BLK + jL.val < S
            then Real.exp ((sm_scale * Finset.univ.sum (fun e : Fin DM =>
              ctxQTileMRowG s Q B_Start_Loc rs hs BLK DM bel (i, e, PUnit.unit)
                * ctxKTileMG s K B_Start_Loc rs hs S DM bel
                    (⟨c * BLK + jL.val, nopad_lane_lt_SG c S BLK hwin jL⟩, e, PUnit.unit))) - Mr) else 0) := by
    intro jL
    have hS : c * BLK + jL.val < S := nopad_lane_lt_SG c S BLK hwin jL
    by_cases hj : c * BLK + jL.val ≤ s.pids 2 * BLK + i.val
    · rw [if_pos hj, if_pos ⟨hj, hS⟩, WithBot.realSub_coe_coe, WithBot.realExp_coe]; rfl
    · rw [if_neg hj, if_neg (fun h => hj h.1), WithBot.realSub_bot_left, WithBot.realExp_bot]; rfl
  simp only [hcell]
  rw [WithBot.sum_someTerm_eq_some]
  refine congrArg some ?_
  rw [nopadBlockMG_map_sum s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel c i d hwin
    (fun p => Real.exp (p.1 - Mr))]

/-- General `nopadBlockMG` `acc` lane-sum bridge. -/
theorem nopadBlockMG_acc_sum
    (s : BlockState) (Q K V B_Start_Loc : RegionName) (sm_scale : ℝ)
    (rs hs BLK DM S bel c : Nat) (i : Fin BLK) (d : Fin DM) (Mr : ℝ)
    (hwin : (c + 1) * BLK ≤ S)
    (rawV : Fin BLK → ℝ)
    (hrawV : ∀ jL : Fin BLK, c * BLK + jL.val ≤ s.pids 2 * BLK + i.val →
      rawV jL = ctxVTileMG s V B_Start_Loc rs hs S DM bel
        (⟨c * BLK + jL.val, nopad_lane_lt_SG c S BLK hwin jL⟩, d, PUnit.unit)) :
    (∑ jL : Fin BLK, WithBot.realMul
        (WithBot.realExp (WithBot.realSub
          (if c * BLK + jL.val ≤ s.pids 2 * BLK + i.val then
            ((sm_scale * Finset.univ.sum (fun e : Fin DM =>
                ctxQTileMRowG s Q B_Start_Loc rs hs BLK DM bel (i, e, PUnit.unit)
                  * ctxKTileMG s K B_Start_Loc rs hs S DM bel
                      (⟨c * BLK + jL.val, nopad_lane_lt_SG c S BLK hwin jL⟩, e, PUnit.unit)) : ℝ) : WithBot ℝ)
           else (⊥ : WithBot ℝ)) ((Mr : ℝ) : WithBot ℝ)))
        ((rawV jL : ℝ) : WithBot ℝ))
      = some ((nopadBlockMG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel c i d).map
          (fun p => Real.exp (p.1 - Mr) * p.2)).sum := by
  have hcell : ∀ jL : Fin BLK,
      WithBot.realMul
        (WithBot.realExp (WithBot.realSub
          (if c * BLK + jL.val ≤ s.pids 2 * BLK + i.val then
            ((sm_scale * Finset.univ.sum (fun e : Fin DM =>
                ctxQTileMRowG s Q B_Start_Loc rs hs BLK DM bel (i, e, PUnit.unit)
                  * ctxKTileMG s K B_Start_Loc rs hs S DM bel
                      (⟨c * BLK + jL.val, nopad_lane_lt_SG c S BLK hwin jL⟩, e, PUnit.unit)) : ℝ) : WithBot ℝ)
           else (⊥ : WithBot ℝ)) ((Mr : ℝ) : WithBot ℝ)))
        ((rawV jL : ℝ) : WithBot ℝ)
        = some (if c * BLK + jL.val ≤ s.pids 2 * BLK + i.val ∧ c * BLK + jL.val < S
            then Real.exp ((sm_scale * Finset.univ.sum (fun e : Fin DM =>
              ctxQTileMRowG s Q B_Start_Loc rs hs BLK DM bel (i, e, PUnit.unit)
                * ctxKTileMG s K B_Start_Loc rs hs S DM bel
                    (⟨c * BLK + jL.val, nopad_lane_lt_SG c S BLK hwin jL⟩, e, PUnit.unit))) - Mr)
                  * ctxVTileMG s V B_Start_Loc rs hs S DM bel
                      (⟨c * BLK + jL.val, nopad_lane_lt_SG c S BLK hwin jL⟩, d, PUnit.unit)
            else 0) := by
    intro jL
    have hS : c * BLK + jL.val < S := nopad_lane_lt_SG c S BLK hwin jL
    by_cases hj : c * BLK + jL.val ≤ s.pids 2 * BLK + i.val
    · rw [if_pos hj, if_pos ⟨hj, hS⟩, WithBot.realSub_coe_coe, WithBot.realExp_coe,
        WithBot.realMul_coe_coe, hrawV jL hj]; rfl
    · rw [if_neg hj, if_neg (fun h => hj h.1), WithBot.realSub_bot_left, WithBot.realExp_bot]
      show WithBot.realMul ((0:ℝ):WithBot ℝ) ((rawV jL : ℝ):WithBot ℝ) = some 0
      rw [WithBot.realMul_coe_coe, zero_mul]; rfl
  simp only [hcell]
  rw [WithBot.sum_someTerm_eq_some]
  refine congrArg some ?_
  rw [nopadBlockMG_map_sum s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel c i d hwin
    (fun p => Real.exp (p.1 - Mr) * p.2)]

/-- General `nopadBlockMG` running-sup bridge. -/
theorem nopadBlockMG_sup_eq
    (s : BlockState) (Q K V B_Start_Loc : RegionName) (sm_scale : ℝ)
    (rs hs BLK DM S bel c : Nat) (i : Fin BLK) (d : Fin DM)
    (hwin : (c + 1) * BLK ≤ S) :
    Finset.univ.sup (fun jL : Fin BLK =>
        if c * BLK + jL.val ≤ s.pids 2 * BLK + i.val then
          ((sm_scale * Finset.univ.sum (fun e : Fin DM =>
              ctxQTileMRowG s Q B_Start_Loc rs hs BLK DM bel (i, e, PUnit.unit)
                * ctxKTileMG s K B_Start_Loc rs hs S DM bel
                    (⟨c * BLK + jL.val, nopad_lane_lt_SG c S BLK hwin jL⟩, e, PUnit.unit)) : ℝ) : WithBot ℝ)
        else (⊥ : WithBot ℝ))
      = ((nopadBlockMG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel c i d).map
          (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥ := by
  classical
  set F : Nat → WithBot ℝ := fun jg =>
    if h : jg < S then
      ((sm_scale * Finset.univ.sum (fun e : Fin DM =>
          ctxQTileMRowG s Q B_Start_Loc rs hs BLK DM bel (i, e, PUnit.unit)
            * ctxKTileMG s K B_Start_Loc rs hs S DM bel (⟨jg, h⟩, e, PUnit.unit)) : ℝ) : WithBot ℝ)
    else ⊥ with hF
  rw [show (nopadBlockMG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel c i d).map (fun p => ((p.1 : ℝ) : WithBot ℝ))
        = ((List.finRange S).filterMap (fun j : Fin S =>
            if j.val ≤ s.pids 2 * BLK + i.val ∧ c * BLK ≤ j.val ∧ j.val < (c + 1) * BLK then
              some (sm_scale * Finset.univ.sum (fun e : Fin DM =>
                ctxQTileMRowG s Q B_Start_Loc rs hs BLK DM bel (i, e, PUnit.unit)
                  * ctxKTileMG s K B_Start_Loc rs hs S DM bel (j, e, PUnit.unit)))
            else none)).map (fun x => ((x : ℝ) : WithBot ℝ)) from by
    unfold nopadBlockMG
    rw [List.map_filterMap, List.map_filterMap]
    apply List.filterMap_congr
    intro j _
    by_cases hj : j.val ≤ s.pids 2 * BLK + i.val ∧ c * BLK ≤ j.val ∧ j.val < (c + 1) * BLK <;> simp [hj]]
  rw [nopad_filterMap_foldr_sup S
    (fun j => j.val ≤ s.pids 2 * BLK + i.val ∧ c * BLK ≤ j.val ∧ j.val < (c + 1) * BLK)
    (fun j => sm_scale * Finset.univ.sum (fun e : Fin DM =>
        ctxQTileMRowG s Q B_Start_Loc rs hs BLK DM bel (i, e, PUnit.unit)
          * ctxKTileMG s K B_Start_Loc rs hs S DM bel (j, e, PUnit.unit)))]
  rw [show (Finset.univ.sup (fun j : Fin S =>
        if j.val ≤ s.pids 2 * BLK + i.val ∧ c * BLK ≤ j.val ∧ j.val < (c + 1) * BLK then
          ((sm_scale * Finset.univ.sum (fun e : Fin DM =>
            ctxQTileMRowG s Q B_Start_Loc rs hs BLK DM bel (i, e, PUnit.unit)
              * ctxKTileMG s K B_Start_Loc rs hs S DM bel (j, e, PUnit.unit)) : ℝ) : WithBot ℝ)
        else ⊥))
      = Finset.univ.sup (fun j : Fin S =>
          if j.val ≤ s.pids 2 * BLK + i.val ∧ c * BLK ≤ j.val ∧ j.val < (c + 1) * BLK then F j.val else ⊥) from by
    apply Finset.sup_congr rfl
    intro j _
    by_cases hw : j.val ≤ s.pids 2 * BLK + i.val ∧ c * BLK ≤ j.val ∧ j.val < (c + 1) * BLK
    · rw [if_pos hw, if_pos hw, hF]; simp only [dif_pos j.isLt]
    · rw [if_neg hw, if_neg hw]]
  rw [nopad_window_sup_reindexG c S (s.pids 2 * BLK + i.val) BLK hwin F]
  apply Finset.sup_congr rfl
  intro jL _
  have hb : c * BLK + jL.val < S := nopad_lane_lt_SG c S BLK hwin jL
  by_cases hc : c * BLK + jL.val ≤ s.pids 2 * BLK + i.val
  · rw [if_pos hc, if_pos hc, hF]; simp only [dif_pos hb]
  · rw [if_neg hc, if_neg hc]

/-- General: the full-window `osNormStepBot` `acc` over the genuine causal-key list is
the genuine closed form `contextAttnNopadExactFoldMG`. -/
theorem ctxNopad_fold_eq_exactFoldMG
    (s : BlockState) (Q K V B_Start_Loc : RegionName) (sm_scale : ℝ)
    (rs hs BLK DM S bel : Nat) (idx : TileIndex [BLK, DM])
    (hvis : (0 : Nat) < S ∧ (0 : Nat) ≤ s.pids 2 * BLK + idx.1.val) :
    ((ctxNopadKeyListG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel idx.1 idx.2.1).foldl
        osNormStepBot (⊥, 0, 0)).2.2
      = contextAttnNopadExactFoldMG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel idx := by
  obtain ⟨i, d, u⟩ := idx
  set xs := ctxNopadKeyListG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel i d with hxs
  have hL : (xs.map (fun p => Real.exp p.1)).sum
      = Finset.univ.sum (fun j : Fin S =>
          if j.val ≤ s.pids 2 * BLK + i.val then
            Real.exp (sm_scale * Finset.univ.sum (fun e : Fin DM =>
              ctxQTileG s Q B_Start_Loc rs hs BLK DM (i, e, PUnit.unit)
                * ctxKTileMG s K B_Start_Loc rs hs S DM bel (j, e, PUnit.unit)))
          else 0) := by
    rw [hxs, ctxNopadKeyListG]
    rw [ctxNopad_filterMap_finRange_sum S
      (fun j : Fin S => j.val ≤ s.pids 2 * BLK + i.val)]
  have hT : (xs.map (fun p => Real.exp p.1 * p.2)).sum
      = Finset.univ.sum (fun j : Fin S =>
          (if j.val ≤ s.pids 2 * BLK + i.val then
            Real.exp (sm_scale * Finset.univ.sum (fun e : Fin DM =>
              ctxQTileG s Q B_Start_Loc rs hs BLK DM (i, e, PUnit.unit)
                * ctxKTileMG s K B_Start_Loc rs hs S DM bel (j, e, PUnit.unit)))
          else 0) * ctxVTileMG s V B_Start_Loc rs hs S DM bel (j, d, PUnit.unit)) := by
    rw [hxs, ctxNopadKeyListG]
    rw [ctxNopad_filterMap_finRange_sum S
      (fun j : Fin S => j.val ≤ s.pids 2 * BLK + i.val)]
    apply Finset.sum_congr rfl; intro j _
    by_cases hj : j.val ≤ s.pids 2 * BLK + i.val <;> simp [hj]
  have hcons := osNormStepBot_foldl_consistent xs ⊥ 0 0 0 0
    (le_refl 0) (fun _ _ => trivial) (by simp) (by ring) (fun _ => rfl) (fun _ => rfl)
    (by intro h; exact absurd h (lt_irrefl 0))
  obtain ⟨_hl', hacc', _hL'0, _hbot1, _hbot2, _hne⟩ := hcons
  simp only [zero_add] at hacc' hL hT
  rw [contextAttnNopadExactFoldMG]
  have hdenom_pos : 0 < Finset.univ.sum (fun j : Fin S =>
        if j.val ≤ s.pids 2 * BLK + i.val then
          Real.exp (sm_scale * Finset.univ.sum (fun e : Fin DM =>
            ctxQTileG s Q B_Start_Loc rs hs BLK DM (i, e, PUnit.unit)
              * ctxKTileMG s K B_Start_Loc rs hs S DM bel (j, e, PUnit.unit)))
        else 0) := by
    apply Finset.sum_pos'
    · intro j _
      by_cases hj : j.val ≤ s.pids 2 * BLK + i.val
      · simp [hj, le_of_lt (Real.exp_pos _)]
      · simp [hj]
    · refine ⟨⟨0, hvis.1⟩, Finset.mem_univ _, ?_⟩
      have h0 : (0 : Nat) ≤ s.pids 2 * BLK + i.val := Nat.zero_le _
      simp only [Fin.val_mk, h0, if_true]
      exact Real.exp_pos _
  rw [eq_div_iff (ne_of_gt hdenom_pos)]
  rw [← hL, ← hT, hacc']

/-- General active-row boundary-mask subsumption: faithful fold = idealized closed form. -/
theorem nopadFoldUptoG_full_eq_genuine
    (s0 : BlockState) (Q K V B_Start_Loc B_Seqlen : RegionName) (sm_scale : ℝ)
    (rs hs BLK DM : Nat) (idx : TileIndex [BLK, DM]) (hBLK : 0 < BLK)
    (hact : s0.pids 2 * BLK + idx.1.val < seqLen s0 B_Seqlen) :
    (nopadFoldUptoG s0 Q K V B_Start_Loc sm_scale rs hs BLK DM
        (ctxNopadWindowG s0 B_Seqlen BLK) (seqLen s0 B_Seqlen)
        (ctxNopadWindowG s0 B_Seqlen BLK) idx.1 idx.2.1).2.2
      = ctxNopadGenuineOutValueG s0 Q K V B_Start_Loc B_Seqlen sm_scale rs hs BLK DM idx := by
  have hpos : 0 < ctxNopadWindowG s0 B_Seqlen BLK := by
    rw [ctxNopadWindowG]
    have hbm : BLK * s0.pids 2 < seqLen s0 B_Seqlen := by
      have := idx.1.isLt; nlinarith [Nat.zero_le BLK]
    rw [if_pos hbm]
    have := idx.1.isLt; positivity
  set S := ctxNopadWindowG s0 B_Seqlen BLK with hSdef
  set bel := seqLen s0 B_Seqlen with hbeldef
  rw [nopadFoldUptoG, ctxNopadKeysUptoMG_active s0 Q K V B_Start_Loc sm_scale rs hs BLK DM S bel S idx.1 idx.2.1 hact,
    ctxNopadKeysUptoG_full]
  rw [show ((ctxNopadKeyListG s0 Q K V B_Start_Loc sm_scale rs hs BLK DM S bel idx.1 idx.2.1).foldl osNormStepBot (⊥, 0, 0)).2.2
        = contextAttnNopadExactFoldMG s0 Q K V B_Start_Loc sm_scale rs hs BLK DM S bel idx from
    ctxNopad_fold_eq_exactFoldMG s0 Q K V B_Start_Loc sm_scale rs hs BLK DM S bel idx ⟨hpos, Nat.zero_le _⟩]
  rw [ctxNopadGenuineOutValueG, ctxNopadBel]

/-! ### General exec-side stepping (contiguous layout `(rs, hs, 1) = (H·DM, DM, 1)`) -/

/-- General preLoop statement list (19 stmts), strides `(rs, hs, 1)`. -/
def nopadPreLoopG (Q K V : RegionName) (B_Start_Loc B_Seqlen : Region .nat)
    (rs hs BLK DM : Nat) : List Stmt :=
  [ Stmt.assign .nat [] "cur_batch" (Op.programId 0),
    Stmt.assign .nat [] "cur_head" (Op.programId 1),
    Stmt.assign .nat [] "start_m" (Op.programId 2),
    Stmt.assign .nat [] "cur_batch_seq_len"
      (Op.load .nat (MemAccess.region B_Seqlen (Op.ref .nat [] "cur_batch")) MaskOpt.none),
    Stmt.assign .nat [] "cur_batch_in_all_start_index"
      (Op.load .nat (MemAccess.region B_Start_Loc (Op.ref .nat [] "cur_batch")) MaskOpt.none),
    Stmt.assign .nat [] "block_start_loc"
      (Op.mul .nat Broadcast.nil (Op.constNat BLK) (Op.ref .nat [] "start_m")),
    Stmt.assign .nat [BLK] "offs_n" (Op.arange BLK),
    Stmt.assign .nat [DM] "offs_d" (Op.arange DM),
    Stmt.assign .nat [BLK] "offs_m"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat BLK))
        (Op.arange BLK)),
    Stmt.assign .nat [BLK, DM] "off_q"
      (Op.add .nat Broadcast.nil.consL.consR
        (Op.add .nat Broadcast.scalarR
          (Op.mul .nat Broadcast.scalarR
            (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "cur_batch_in_all_start_index")
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLK] "offs_m")))
            (Op.constNat rs))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat hs)))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [DM] "offs_d"))
          (Op.constNat 1))),
    Stmt.assign .nat [DM, BLK] "off_k"
      (Op.add .nat Broadcast.nil.consR.consL
        (Op.add .nat Broadcast.scalarR
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLK] "offs_n"))
            (Op.constNat rs))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat hs)))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [DM] "offs_d"))
          (Op.constNat 1))),
    Stmt.assign .nat [BLK, DM] "off_v"
      (Op.add .nat Broadcast.nil.consL.consR
        (Op.add .nat Broadcast.scalarR
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLK] "offs_n"))
            (Op.constNat rs))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat hs)))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [DM] "offs_d"))
          (Op.constNat 1))),
    Stmt.assign .real [BLK, DM] "q"
      (Op.load .real (MemAccess.region Q (Op.ref .nat [BLK, DM] "off_q"))
        (MaskOpt.maskOther
          (Op.remap [BLK, DM] Broadcast.nil.consL.consSame.leftIndex
            (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLK] "offs_m"))
              (Op.ref .nat [] "cur_batch_seq_len")))
          ((Op.const 0.0).broadcast [BLK, DM]))),
    Stmt.assign .ptr [DM, BLK] "k_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase K) (Op.ref .nat [DM, BLK] "off_k")),
    Stmt.assign .ptr [BLK, DM] "v_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase V) (Op.ref .nat [BLK, DM] "off_v")),
    Stmt.assign .real [BLK] "m_i"
      (Op.add .real Broadcast.scalarR (Op.full [BLK] (Op.const 0)) Op.negInf),
    Stmt.assign .real [BLK] "l_i" (Op.full [BLK] (Op.const 0)),
    Stmt.assign .real [BLK, DM] "acc" (Op.full [BLK, DM] (Op.const 0)),
    Stmt.assign .nat [] "block_mask"
      ((Op.lt ComparableDType.nat Broadcast.nil (Op.ref .nat [] "block_start_loc")
            (Op.ref .nat [] "cur_batch_seq_len")).where
        (Op.constNat 1) (Op.constNat 0)) ]

/-- General loop-body statement list (22 stmts), strides `(rs, hs, 1)`. -/
def nopadLoopBodyG (sc : ℝ) (rs hs BLK DM : Nat) : List Stmt :=
  [ Stmt.assign .nat [] "start_n" (Op.ref .nat [] "start_n"),
    Stmt.assign .real [DM, BLK] "k"
      (Op.load .real
        (MemAccess.ptr
          (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [DM, BLK] "k_ptrs")
            (Op.mul .nat Broadcast.nil
              (Op.add .nat Broadcast.nil (Op.ref .nat [] "cur_batch_in_all_start_index")
                (Op.ref .nat [] "start_n"))
              (Op.constNat rs))))
        (MaskOpt.maskOther
          (Op.remap [DM, BLK] Broadcast.nil.consSame.consL.leftIndex
            (Op.lt ComparableDType.nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
                (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLK] "offs_n")))
              (Op.ref .nat [] "cur_batch_seq_len")))
          ((Op.const 0.0).broadcast [DM, BLK]))),
    Stmt.assign .real [BLK, BLK] "qk" (Op.full [BLK, BLK] (Op.const 0)),
    Stmt.assign .real [BLK, BLK] "qk"
      (Op.add .real Broadcast.nil.consSame.consSame (Op.ref .real [BLK, BLK] "qk")
        (Op.dot (batch := []) (Op.ref .real [BLK, DM] "q") (Op.ref .real [DM, BLK] "k"))),
    Stmt.assign .real [BLK, BLK] "qk"
      (Op.mul .real Broadcast.scalarR (Op.ref .real [BLK, BLK] "qk") (Op.const sc)),
    Stmt.assign .real [BLK, BLK] "qk"
      ((Op.ge ComparableDType.nat Broadcast.nil.consL.consR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLK] "offs_m"))
            (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
              (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLK] "offs_n")))).where
        (Op.ref .real [BLK, BLK] "qk") (Op.negInf.broadcast [BLK, BLK])),
    Stmt.assign .real [BLK] "m_ij" (Op.reduceMax ⟨1, by simp⟩ Bool.false (Op.ref .real [BLK, BLK] "qk")),
    Stmt.assign .real [BLK, BLK] "p"
      (Op.sub .real Broadcast.nil.consR.consSame (Op.ref .real [BLK, BLK] "qk")
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BLK] "m_ij"))).exp,
    Stmt.assign .real [BLK] "l_ij" (Op.reduceSum ⟨1, by simp⟩ Bool.false (Op.ref .real [BLK, BLK] "p")),
    Stmt.assign .real [BLK] "m_i_new"
      ((Op.gt ComparableDType.real Broadcast.nil.consSame (Op.ref .real [BLK] "m_i")
            (Op.ref .real [BLK] "m_ij")).where
        (Op.ref .real [BLK] "m_i") (Op.ref .real [BLK] "m_ij")),
    Stmt.assign .real [BLK] "alpha"
      (Op.sub .real Broadcast.nil.consSame (Op.ref .real [BLK] "m_i")
          (Op.ref .real [BLK] "m_i_new")).exp,
    Stmt.assign .real [BLK] "beta"
      (Op.sub .real Broadcast.nil.consSame (Op.ref .real [BLK] "m_ij")
          (Op.ref .real [BLK] "m_i_new")).exp,
    Stmt.assign .real [BLK] "l_i_new"
      (Op.add .real Broadcast.nil.consSame
        (Op.mul .real Broadcast.nil.consSame (Op.ref .real [BLK] "alpha")
          (Op.ref .real [BLK] "l_i"))
        (Op.mul .real Broadcast.nil.consSame (Op.ref .real [BLK] "beta")
          (Op.ref .real [BLK] "l_ij"))),
    Stmt.assign .real [BLK] "p_scale"
      (Op.div .real Broadcast.nil.consSame (Op.ref .real [BLK] "beta")
        (Op.ref .real [BLK] "l_i_new")),
    Stmt.assign .real [BLK, BLK] "p"
      (Op.mul .real Broadcast.nil.consR.consSame (Op.ref .real [BLK, BLK] "p")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BLK] "p_scale"))),
    Stmt.assign .real [BLK] "acc_scale"
      (Op.mul .real Broadcast.nil.consSame
        (Op.div .real Broadcast.nil.consSame (Op.ref .real [BLK] "l_i")
          (Op.ref .real [BLK] "l_i_new"))
        (Op.ref .real [BLK] "alpha")),
    Stmt.assign .real [BLK, DM] "acc"
      (Op.mul .real Broadcast.nil.consR.consSame (Op.ref .real [BLK, DM] "acc")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BLK] "acc_scale"))),
    Stmt.assign .real [BLK, DM] "v"
      (Op.load .real
        (MemAccess.ptr
          (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BLK, DM] "v_ptrs")
            (Op.mul .nat Broadcast.nil
              (Op.add .nat Broadcast.nil (Op.ref .nat [] "cur_batch_in_all_start_index")
                (Op.ref .nat [] "start_n"))
              (Op.constNat rs))))
        (MaskOpt.maskOther
          (Op.remap [BLK, DM] Broadcast.nil.consL.consSame.leftIndex
            (Op.lt ComparableDType.nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
                (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLK] "offs_n")))
              (Op.ref .nat [] "cur_batch_seq_len")))
          ((Op.const 0.0).broadcast [BLK, DM]))),
    Stmt.assign .real [BLK, BLK] "p" (Op.ref .real [BLK, BLK] "p"),
    Stmt.assign .real [BLK, DM] "acc"
      (Op.add .real Broadcast.nil.consSame.consSame (Op.ref .real [BLK, DM] "acc")
        (Op.dot (batch := []) (Op.ref .real [BLK, BLK] "p") (Op.ref .real [BLK, DM] "v"))),
    Stmt.assign .real [BLK] "l_i" (Op.ref .real [BLK] "l_i_new"),
    Stmt.assign .real [BLK] "m_i" (Op.ref .real [BLK] "m_i_new") ]

/-- General postLoop statement list (3 stmts), strides `(rs, hs, 1)`. -/
def nopadPostLoopG (Out : RegionName) (rs hs BLK DM : Nat) : List Stmt :=
  [ Stmt.assign .nat [BLK, DM] "off_o"
      (Op.add .nat Broadcast.nil.consL.consR
        (Op.add .nat Broadcast.scalarR
          (Op.mul .nat Broadcast.scalarR
            (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "cur_batch_in_all_start_index")
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLK] "offs_m")))
            (Op.constNat rs))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat hs)))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [DM] "offs_d"))
          (Op.constNat 1))),
    Stmt.assign .ptr [BLK, DM] "out_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Out) (Op.ref .nat [BLK, DM] "off_o")),
    Stmt.store .real [BLK, DM] (MemAccess.ptr (Op.ref .ptr [BLK, DM] "out_ptrs"))
      (Op.ref .real [BLK, DM] "acc")
      (MaskOpt.mask
        (Op.remap [BLK, DM] Broadcast.nil.consL.consSame.leftIndex
          (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLK] "offs_m"))
            (Op.ref .nat [] "cur_batch_seq_len")))) ]

set_option maxRecDepth 8000 in
/-- General body split: the lowered forward body is `nopadPreLoopG ++ forRangeDyn :: nopadPostLoopG`. -/
theorem nopad_body_splitG
    (Q K V : RegionName) (B_Start_Loc B_Seqlen : Region .nat) (Out : RegionName) (sc : ℝ)
    (rs hs BLK DM : Nat) :
    (context_attn_nopad_fwd_kernel_surface Q K V sc B_Start_Loc B_Seqlen Out
      rs hs 1 rs hs 1 rs hs 1 rs hs 1 BLK DM BLK).toAlgKernel.body
      = nopadPreLoopG Q K V B_Start_Loc B_Seqlen rs hs BLK DM
        ++ (Stmt.forRangeDyn "start_n" (Op.constNat 0)
              (Op.mul .nat Broadcast.nil
                (Op.mul .nat Broadcast.nil (Op.ref .nat [] "block_mask")
                  (Op.add .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat 1)))
                (Op.constNat BLK))
              (Op.constNat BLK) (nopadLoopBodyG sc rs hs BLK DM)
            :: nopadPostLoopG Out rs hs BLK DM) := by
  rfl

set_option maxHeartbeats 1600000 in
/-- General `off_q` recipe. -/
theorem nopad_offq_evalG (s : BlockState) (sl head pid2 rs hs BLK DM : Nat)
    (hsl : s.regs .nat [] "cur_batch_in_all_start_index" = some (Tile.scalar sl))
    (hch : s.regs .nat [] "cur_head" = some (Tile.scalar head))
    (hm : s.regs .nat [BLK] "offs_m" = some (Tile.vec (fun i : Fin BLK => pid2 * BLK + i.val)))
    (hd : s.regs .nat [DM] "offs_d" = some (Tile.vec (fun e : Fin DM => e.val))) :
    evalOp (Op.add .nat Broadcast.nil.consL.consR
        (Op.add .nat Broadcast.scalarR
          (Op.mul .nat Broadcast.scalarR
            (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "cur_batch_in_all_start_index")
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLK] "offs_m")))
            (Op.constNat rs))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat hs)))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [DM] "offs_d"))
          (Op.constNat 1))) s
      = some (⟨fun idx : TileIndex [BLK, DM] =>
          (sl + (pid2 * BLK + idx.1.val)) * rs + head * hs + idx.2.1.val⟩ : Tile .nat [BLK, DM]) := by
  have hexpM := evalOp_expandDim_ref_of_regs .nat [BLK] ⟨1, by simp⟩ "offs_m" s _ hm
  have hexpD := evalOp_expandDim_ref_of_regs .nat [DM] ⟨0, by simp⟩ "offs_d" s _ hd
  rw [evalOp_add, evalOp_add, evalOp_mul, evalOp_add, evalOp_ref, hsl]
  erw [hexpM]
  rw [evalOp_mul, evalOp_ref, hch, evalOp_mul]
  erw [hexpD]
  rw [evalOp_constNat, evalOp_constNat, evalOp_constNat]
  simp only [Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_; ext idx
  obtain ⟨i, e, u⟩ := idx
  simp only [Tile.bop_data, Tile.bop, Tile.scalar, Tile.vec, Tile.expandDim,
    Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul,
    TileShape.dropInsertedIndex]
  ring

set_option maxHeartbeats 1600000 in
/-- General `off_k` recipe (lane `(e, jL)`). -/
theorem nopad_offk_evalG (s : BlockState) (head rs hs BLK DM : Nat)
    (hch : s.regs .nat [] "cur_head" = some (Tile.scalar head))
    (hn : s.regs .nat [BLK] "offs_n" = some (Tile.vec (fun j : Fin BLK => j.val)))
    (hd : s.regs .nat [DM] "offs_d" = some (Tile.vec (fun e : Fin DM => e.val))) :
    evalOp (Op.add .nat Broadcast.nil.consR.consL
        (Op.add .nat Broadcast.scalarR
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLK] "offs_n"))
            (Op.constNat rs))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat hs)))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [DM] "offs_d"))
          (Op.constNat 1))) s
      = some (⟨fun idx : TileIndex [DM, BLK] =>
          idx.2.1.val * rs + head * hs + idx.1.val⟩ : Tile .nat [DM, BLK]) := by
  have hexpN := evalOp_expandDim_ref_of_regs .nat [BLK] ⟨0, by simp⟩ "offs_n" s _ hn
  have hexpD := evalOp_expandDim_ref_of_regs .nat [DM] ⟨1, by simp⟩ "offs_d" s _ hd
  rw [evalOp_add, evalOp_add, evalOp_mul]
  erw [hexpN]
  rw [evalOp_mul, evalOp_ref, hch, evalOp_mul]
  erw [hexpD]
  rw [evalOp_constNat, evalOp_constNat, evalOp_constNat]
  simp only [Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_; ext idx
  obtain ⟨e, jL, u⟩ := idx
  simp only [Tile.bop_data, Tile.bop, Tile.scalar, Tile.vec, Tile.expandDim,
    Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul,
    TileShape.dropInsertedIndex]
  ring

set_option maxHeartbeats 1600000 in
/-- General `off_v` recipe (lane `(jL, d)`). -/
theorem nopad_offv_evalG (s : BlockState) (head rs hs BLK DM : Nat)
    (hch : s.regs .nat [] "cur_head" = some (Tile.scalar head))
    (hn : s.regs .nat [BLK] "offs_n" = some (Tile.vec (fun j : Fin BLK => j.val)))
    (hd : s.regs .nat [DM] "offs_d" = some (Tile.vec (fun e : Fin DM => e.val))) :
    evalOp (Op.add .nat Broadcast.nil.consL.consR
        (Op.add .nat Broadcast.scalarR
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLK] "offs_n"))
            (Op.constNat rs))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat hs)))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [DM] "offs_d"))
          (Op.constNat 1))) s
      = some (⟨fun idx : TileIndex [BLK, DM] =>
          idx.1.val * rs + head * hs + idx.2.1.val⟩ : Tile .nat [BLK, DM]) := by
  have hexpN := evalOp_expandDim_ref_of_regs .nat [BLK] ⟨1, by simp⟩ "offs_n" s _ hn
  have hexpD := evalOp_expandDim_ref_of_regs .nat [DM] ⟨0, by simp⟩ "offs_d" s _ hd
  rw [evalOp_add, evalOp_add, evalOp_mul]
  erw [hexpN]
  rw [evalOp_mul, evalOp_ref, hch, evalOp_mul]
  erw [hexpD]
  rw [evalOp_constNat, evalOp_constNat, evalOp_constNat]
  simp only [Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_; ext idx
  obtain ⟨jL, d, u⟩ := idx
  simp only [Tile.bop_data, Tile.bop, Tile.scalar, Tile.vec, Tile.expandDim,
    Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul,
    TileShape.dropInsertedIndex]
  ring

set_option maxHeartbeats 1600000 in
/-- General `q` masked-load recipe. -/
theorem nopad_q_load_evalG (s : BlockState) (Q B_Start_Loc B_Seqlen : RegionName) (rs hs BLK DM : Nat)
    (hoffq : s.regs .nat [BLK, DM] "off_q" = some (⟨fun idx : TileIndex [BLK, DM] =>
        (startLoc s B_Start_Loc + (s.pids 2 * BLK + idx.1.val)) * rs + s.pids 1 * hs + idx.2.1.val⟩
        : Tile .nat [BLK, DM]))
    (hm : s.regs .nat [BLK] "offs_m" = some (Tile.vec (fun i : Fin BLK => s.pids 2 * BLK + i.val)))
    (hseq : s.regs .nat [] "cur_batch_seq_len" = some (Tile.scalar (seqLen s B_Seqlen))) :
    evalOp (Op.load .real (MemAccess.region Q (Op.ref .nat [BLK, DM] "off_q"))
        (MaskOpt.maskOther
          (Op.remap [BLK, DM] Broadcast.nil.consL.consSame.leftIndex
            (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLK] "offs_m"))
              (Op.ref .nat [] "cur_batch_seq_len")))
          ((Op.const 0.0).broadcast [BLK, DM]))) s
      = some (⟨fun idx : TileIndex [BLK, DM] =>
          some (ctxQTileMRowG s Q B_Start_Loc rs hs BLK DM (seqLen s B_Seqlen) (idx.1, idx.2.1, PUnit.unit))⟩
          : Tile .real [BLK, DM]) := by
  have hexpM := evalOp_expandDim_ref_of_regs .nat [BLK] ⟨1, by simp⟩ "offs_m" s _ hm
  simp only [evalOp, hoffq, hseq, Option.bind]
  erw [hexpM]
  refine congrArg some ?_; ext idx
  obtain ⟨i, e, u⟩ := idx
  simp only [Tile.cop_data, Tile.bop_data, Tile.bop, Tile.remap, Tile.expandDim, Tile.vec,
    Tile.scalar, NumericDType.add, NumericDType.mul, ComparableDType.lt, Broadcast.leftIndex,
    Broadcast.rightIndex, BlockState.readMemValue_real, Region.cast_id,
    TileShape.dropInsertedIndex, ctxQTileMRowG, ctxQTileG]
  by_cases hlt : s.pids 2 * BLK + i.val < seqLen s B_Seqlen
  · rw [if_pos hlt]
    simp only [hlt, decide_true, if_true, if_pos hlt]
  · rw [if_neg hlt]
    simp only [hlt, decide_false, if_false, Bool.false_eq_true]
    norm_num

set_option maxHeartbeats 1600000 in
/-- General `k` masked-load recipe (lane `(e, jL)`). -/
theorem nopad_k_load_evalG (s : BlockState) (K B_Start_Loc B_Seqlen : RegionName) (SN rs hs BLK DM : Nat)
    (hkp : s.regs .ptr [DM, BLK] "k_ptrs" =
      some (⟨fun idx : TileIndex [DM, BLK] =>
        (K, idx.2.1.val * rs + s.pids 1 * hs + idx.1.val)⟩
        : Tile .ptr [DM, BLK]))
    (hsl : s.regs .nat [] "cur_batch_in_all_start_index" = some (Tile.scalar (startLoc s B_Start_Loc)))
    (hsn : s.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hn : s.regs .nat [BLK] "offs_n" = some (Tile.vec (fun j : Fin BLK => j.val)))
    (hseq : s.regs .nat [] "cur_batch_seq_len" = some (Tile.scalar (seqLen s B_Seqlen))) :
    evalOp (Op.load .real
        (MemAccess.ptr
          (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [DM, BLK] "k_ptrs")
            (Op.mul .nat Broadcast.nil
              (Op.add .nat Broadcast.nil (Op.ref .nat [] "cur_batch_in_all_start_index")
                (Op.ref .nat [] "start_n"))
              (Op.constNat rs))))
        (MaskOpt.maskOther
          (Op.remap [DM, BLK] Broadcast.nil.consSame.consL.leftIndex
            (Op.lt ComparableDType.nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
                (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLK] "offs_n")))
              (Op.ref .nat [] "cur_batch_seq_len")))
          ((Op.const 0.0).broadcast [DM, BLK]))) s
      = some (⟨fun idx : TileIndex [DM, BLK] =>
          if SN + idx.2.1.val < seqLen s B_Seqlen then
            some (s.readMem K ((startLoc s B_Start_Loc + (SN + idx.2.1.val)) * rs
              + s.pids 1 * hs + idx.1.val))
          else some (0.0 : ℝ)⟩ : Tile .real [DM, BLK]) := by
  have hexp : @evalOp .nat [1, BLK] (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLK] "offs_n")) s
      = some (Tile.expandDim ⟨0, by simp⟩ (Tile.vec (fun j : Fin BLK => j.val))) :=
    evalOp_expandDim_ref_of_regs .nat [BLK] ⟨0, by simp⟩ "offs_n" s _ hn
  simp only [evalOp, hkp, hsl, hsn, hexp, hseq, Option.bind, Option.some.injEq]
  refine congrArg some ?_
  ext idx
  obtain ⟨e, jL, u⟩ := idx
  simp only [Tile.ptrAdd_data, Tile.cop_data, Tile.bop_data, Tile.bop, Tile.remap, Tile.expandDim,
    Tile.vec, Tile.scalar, NumericDType.add, NumericDType.mul, ComparableDType.lt,
    Broadcast.leftIndex, Broadcast.rightIndex, BlockState.readMemValue_real, Region.cast_id,
    BlockState.readMem, TileShape.dropInsertedIndex]
  by_cases hlt : SN + jL.val < seqLen s B_Seqlen
  · simp only [hlt, decide_true, if_true, if_pos hlt, BlockState.readMem]
    rw [show jL.val * rs + s.pids 1 * hs + e.val + (startLoc s B_Start_Loc + SN) * rs
        = (startLoc s B_Start_Loc + (SN + jL.val)) * rs + s.pids 1 * hs + e.val from by ring]
  · simp only [hlt, decide_false, if_false, if_neg hlt, Bool.false_eq_true]

set_option maxHeartbeats 1600000 in
/-- General `v` masked-load recipe (lane `(jL, d)`). -/
theorem nopad_v_load_evalG (s : BlockState) (V B_Start_Loc B_Seqlen : RegionName) (SN rs hs BLK DM : Nat)
    (hvp : s.regs .ptr [BLK, DM] "v_ptrs" =
      some (⟨fun idx : TileIndex [BLK, DM] =>
        (V, idx.1.val * rs + s.pids 1 * hs + idx.2.1.val)⟩ : Tile .ptr [BLK, DM]))
    (hsl : s.regs .nat [] "cur_batch_in_all_start_index" = some (Tile.scalar (startLoc s B_Start_Loc)))
    (hsn : s.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hn : s.regs .nat [BLK] "offs_n" = some (Tile.vec (fun j : Fin BLK => j.val)))
    (hseq : s.regs .nat [] "cur_batch_seq_len" = some (Tile.scalar (seqLen s B_Seqlen))) :
    evalOp (Op.load .real
        (MemAccess.ptr
          (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BLK, DM] "v_ptrs")
            (Op.mul .nat Broadcast.nil
              (Op.add .nat Broadcast.nil (Op.ref .nat [] "cur_batch_in_all_start_index")
                (Op.ref .nat [] "start_n"))
              (Op.constNat rs))))
        (MaskOpt.maskOther
          (Op.remap [BLK, DM] Broadcast.nil.consL.consSame.leftIndex
            (Op.lt ComparableDType.nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
                (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLK] "offs_n")))
              (Op.ref .nat [] "cur_batch_seq_len")))
          ((Op.const 0.0).broadcast [BLK, DM]))) s
      = some (⟨fun idx : TileIndex [BLK, DM] =>
          if SN + idx.1.val < seqLen s B_Seqlen then
            some (s.readMem V ((startLoc s B_Start_Loc + (SN + idx.1.val)) * rs
              + s.pids 1 * hs + idx.2.1.val))
          else some (0.0 : ℝ)⟩ : Tile .real [BLK, DM]) := by
  have hexp : @evalOp .nat [BLK, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLK] "offs_n")) s
      = some (Tile.expandDim ⟨1, by simp⟩ (Tile.vec (fun j : Fin BLK => j.val))) :=
    evalOp_expandDim_ref_of_regs .nat [BLK] ⟨1, by simp⟩ "offs_n" s _ hn
  simp only [evalOp, hvp, hsl, hsn, hexp, hseq, Option.bind, Option.some.injEq]
  refine congrArg some ?_
  ext idx
  obtain ⟨jL, d, u⟩ := idx
  simp only [Tile.ptrAdd_data, Tile.cop_data, Tile.bop_data, Tile.bop, Tile.remap, Tile.expandDim,
    Tile.vec, Tile.scalar, NumericDType.add, NumericDType.mul, ComparableDType.lt,
    Broadcast.leftIndex, Broadcast.rightIndex, BlockState.readMemValue_real, Region.cast_id,
    BlockState.readMem, TileShape.dropInsertedIndex]
  by_cases hlt : SN + jL.val < seqLen s B_Seqlen
  · simp only [hlt, decide_true, if_true, if_pos hlt, BlockState.readMem]
    rw [show jL.val * rs + s.pids 1 * hs + d.val + (startLoc s B_Start_Loc + SN) * rs
        = (startLoc s B_Start_Loc + (SN + jL.val)) * rs + s.pids 1 * hs + d.val from by ring]
  · simp only [hlt, decide_false, if_false, if_neg hlt, Bool.false_eq_true]

set_option maxHeartbeats 1600000 in
/-- General `qk += dot(q, k)` recipe. -/
theorem nopad_qk_dot_evalG (s : BlockState) (BLK DM : Nat) (qFn : Fin BLK → Fin DM → ℝ)
    (kFn : Fin DM → Fin BLK → ℝ)
    (hqk0 : s.regs .real [BLK, BLK] "qk"
      = some (⟨fun _ : TileIndex [BLK, BLK] => some (0 : ℝ)⟩ : Tile .real [BLK, BLK]))
    (hq : s.regs .real [BLK, DM] "q"
      = some (⟨fun idx : TileIndex [BLK, DM] => some (qFn idx.1 idx.2.1)⟩ : Tile .real [BLK, DM]))
    (hk : s.regs .real [DM, BLK] "k"
      = some (⟨fun idx : TileIndex [DM, BLK] => some (kFn idx.1 idx.2.1)⟩ : Tile .real [DM, BLK])) :
    evalOp (Op.add .real Broadcast.nil.consSame.consSame (Op.ref .real [BLK, BLK] "qk")
        (Op.dot (batch := []) (Op.ref .real [BLK, DM] "q") (Op.ref .real [DM, BLK] "k"))) s
      = some (⟨fun idx : TileIndex [BLK, BLK] =>
          some (Finset.univ.sum (fun e : Fin DM => qFn idx.1 e * kFn e idx.2.1))⟩
          : Tile .real [BLK, BLK]) := by
  have hdot : evalOp (Op.dot (batch := []) (Op.ref .real [BLK, DM] "q") (Op.ref .real [DM, BLK] "k")) s
      = some (Tile.dot [] (⟨fun idx : TileIndex [BLK, DM] => some (qFn idx.1 idx.2.1)⟩ : Tile .real [BLK, DM])
          (⟨fun idx : TileIndex [DM, BLK] => some (kFn idx.1 idx.2.1)⟩ : Tile .real [DM, BLK])) := by
    rw [evalOp_dot]; erw [evalOp_ref, hq, evalOp_ref, hk]; rfl
  rw [evalOp_add, evalOp_ref, hqk0]
  show Option.bind (evalOp (Op.dot (batch := []) (Op.ref .real [BLK, DM] "q") (Op.ref .real [DM, BLK] "k")) s) _ = _
  rw [hdot]
  simp only [Option.bind]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, jL, u⟩ := idx
  simp only [Tile.bop_data, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.add, WithBot.realAdd]
  rw [Tile.dot_nil_data]
  rw [show (@Finset.sum (Fin DM) (WithBot ℝ) _ Finset.univ fun e : Fin DM =>
        Option.map₂ (· * ·) ((⟨fun idx : TileIndex [BLK, DM] => some (qFn idx.1 idx.2.1)⟩ : Tile .real [BLK, DM]).data (i, e, PUnit.unit))
          ((⟨fun idx : TileIndex [DM, BLK] => some (kFn idx.1 idx.2.1)⟩ : Tile .real [DM, BLK]).data (e, jL, PUnit.unit)))
      = (@Finset.sum (Fin DM) (WithBot ℝ) _ Finset.univ fun e : Fin DM => ((qFn i e * kFn e jL : ℝ) : WithBot ℝ)) from by
    apply Finset.sum_congr rfl; intro e _; rfl]
  rw [WithBot.sum_some_eq_some]
  show Option.map₂ (· + ·) (some 0) (some _) = _
  simp only [Option.map₂, Option.bind, Option.map, zero_add]

set_option maxHeartbeats 1600000 in
/-- General `qk *= sm_scale` recipe. -/
theorem nopad_qk_scale_evalG (s : BlockState) (BLK : Nat) (sc : ℝ) (rawFn : Fin BLK → Fin BLK → ℝ)
    (hqk : s.regs .real [BLK, BLK] "qk"
      = some (⟨fun idx : TileIndex [BLK, BLK] => some (rawFn idx.1 idx.2.1)⟩ : Tile .real [BLK, BLK])) :
    evalOp (Op.mul .real Broadcast.scalarR (Op.ref .real [BLK, BLK] "qk") (Op.const sc)) s
      = some (⟨fun idx : TileIndex [BLK, BLK] => some (rawFn idx.1 idx.2.1 * sc)⟩
          : Tile .real [BLK, BLK]) := by
  rw [evalOp_mul, evalOp_ref, hqk, evalOp_const]
  simp only [Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, jL, u⟩ := idx
  simp only [Tile.bop_data, Tile.bop, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.mul]
  rfl

set_option maxHeartbeats 1600000 in
/-- General `qk = where(offs_m ≥ start_n+offs_n, qk, -inf)` recipe. -/
theorem nopad_qk_where_evalG (s : BlockState) (BLK : Nat) (SN : Nat) (qkFn : Fin BLK → Fin BLK → ℝ)
    (offsM : Fin BLK → Nat)
    (hqk : s.regs .real [BLK, BLK] "qk"
      = some (⟨fun idx : TileIndex [BLK, BLK] => some (qkFn idx.1 idx.2.1)⟩ : Tile .real [BLK, BLK]))
    (hm : s.regs .nat [BLK] "offs_m" = some (Tile.vec offsM))
    (hsn : s.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hn : s.regs .nat [BLK] "offs_n" = some (Tile.vec (fun j : Fin BLK => j.val))) :
    evalOp ((Op.ge ComparableDType.nat Broadcast.nil.consL.consR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLK] "offs_m"))
            (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
              (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLK] "offs_n")))).where
        (Op.ref .real [BLK, BLK] "qk") (Op.negInf.broadcast [BLK, BLK])) s
      = some (⟨fun idx : TileIndex [BLK, BLK] =>
          if SN + idx.2.1.val ≤ offsM idx.1 then some (qkFn idx.1 idx.2.1)
          else (⊥ : WithBot ℝ)⟩ : Tile .real [BLK, BLK]) := by
  have hexpM : @evalOp .nat [BLK, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLK] "offs_m")) s
      = some (Tile.expandDim ⟨1, by simp⟩ (Tile.vec offsM)) :=
    evalOp_expandDim_ref_of_regs .nat [BLK] ⟨1, by simp⟩ "offs_m" s _ hm
  have hexpN : @evalOp .nat [1, BLK] (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLK] "offs_n")) s
      = some (Tile.expandDim ⟨0, by simp⟩ (Tile.vec (fun j : Fin BLK => j.val))) :=
    evalOp_expandDim_ref_of_regs .nat [BLK] ⟨0, by simp⟩ "offs_n" s _ hn
  simp only [evalOp, hexpM, hsn, hexpN, hqk, Option.bind, Option.some.injEq]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, jL, u⟩ := idx
  simp only [Tile.select_data, Tile.cop_data, Tile.bop_data, Tile.bop, Tile.expandDim, Tile.vec,
    Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex, ComparableDType.ge, NumericDType.add,
    TileShape.dropInsertedIndex]
  by_cases hle : SN + jL.val ≤ offsM i
  · rw [if_pos (by simp only [decide_eq_true_eq]; omega), if_pos hle]
  · rw [if_neg (by simp only [decide_eq_true_eq]; omega), if_neg hle]; rfl

set_option maxHeartbeats 1600000 in
/-- General `m_ij = tl.max(qk, 1)` recipe. -/
theorem nopad_mij_evalG (s : BlockState) (BLK : Nat) (hBLK : 0 < BLK) (qkFn : Fin BLK → Fin BLK → WithBot ℝ)
    (hqk : s.regs .real [BLK, BLK] "qk"
      = some (⟨fun idx : TileIndex [BLK, BLK] => qkFn idx.1 idx.2.1⟩ : Tile .real [BLK, BLK])) :
    evalOp (Op.reduceMax ⟨1, by simp⟩ Bool.false (Op.ref .real [BLK, BLK] "qk")) s
      = some (⟨fun idx : TileIndex [BLK] =>
          Finset.univ.sup' (⟨⟨0, hBLK⟩, Finset.mem_univ _⟩ : (Finset.univ : Finset (Fin BLK)).Nonempty)
            (fun jL : Fin BLK => qkFn idx.1 jL)⟩ : Tile .real [BLK]) := by
  rw [evalOp_reduceMax, evalOp_ref, hqk]
  simp only [Option.bind_eq_bind, Option.bind_some, Tile.reduceMax_false]
  unfold Tile.reduceMaxDrop
  rw [dif_pos (show 0 < TileShape.axisDim [BLK, BLK] (⟨1, by simp⟩ : Fin [BLK,BLK].length) from hBLK)]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, u⟩ := idx
  rfl

set_option maxHeartbeats 1600000 in
/-- General `p = tl.exp(qk - m_ij[:,None])` recipe. -/
theorem nopad_p_evalG (s : BlockState) (BLK : Nat) (qkFn : Fin BLK → Fin BLK → WithBot ℝ)
    (mijFn : Fin BLK → WithBot ℝ)
    (hqk : s.regs .real [BLK, BLK] "qk"
      = some (⟨fun idx : TileIndex [BLK, BLK] => qkFn idx.1 idx.2.1⟩ : Tile .real [BLK, BLK]))
    (hmij : s.regs .real [BLK] "m_ij" = some (⟨fun idx : TileIndex [BLK] => mijFn idx.1⟩ : Tile .real [BLK])) :
    evalOp (Op.sub .real Broadcast.nil.consR.consSame (Op.ref .real [BLK, BLK] "qk")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BLK] "m_ij"))).exp s
      = some (⟨fun idx : TileIndex [BLK, BLK] =>
          WithBot.realExp (WithBot.realSub (qkFn idx.1 idx.2.1) (mijFn idx.1))⟩ : Tile .real [BLK, BLK]) := by
  have hexp : @evalOp .real [BLK, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BLK] "m_ij")) s
      = some (Tile.expandDim ⟨1, by simp⟩ (⟨fun idx : TileIndex [BLK] => mijFn idx.1⟩ : Tile .real [BLK])) :=
    evalOp_expandDim_ref_of_regs .real [BLK] ⟨1, by simp⟩ "m_ij" s _ hmij
  rw [evalOp_exp, evalOp_sub, evalOp_ref, hqk, hexp]
  simp only [Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, jL, u⟩ := idx
  simp only [Tile.uop_data, Tile.bop_data, Tile.bop, Tile.expandDim, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.sub, TileShape.dropInsertedIndex]

set_option maxHeartbeats 1600000 in
/-- General `l_ij = tl.sum(p, 1)` recipe. -/
theorem nopad_lij_evalG (s : BlockState) (BLK : Nat) (pFn : Fin BLK → Fin BLK → WithBot ℝ)
    (hp : s.regs .real [BLK, BLK] "p"
      = some (⟨fun idx : TileIndex [BLK, BLK] => pFn idx.1 idx.2.1⟩ : Tile .real [BLK, BLK])) :
    evalOp (Op.reduceSum ⟨1, by simp⟩ Bool.false (Op.ref .real [BLK, BLK] "p")) s
      = some (⟨fun idx : TileIndex [BLK] =>
          (Finset.univ.sum (fun jL : Fin BLK => pFn idx.1 jL))⟩ : Tile .real [BLK]) := by
  rw [evalOp_reduceSum, evalOp_ref, hp]
  simp only [Option.bind_eq_bind, Option.bind_some, Tile.reduceSum_false]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, u⟩ := idx
  rfl

set_option maxHeartbeats 1600000 in
/-- General `m_i_new = tl.maximum(m_i, m_ij)` recipe. -/
theorem nopad_minew_evalG (s : BlockState) (BLK : Nat) (miFn mijFn : Fin BLK → WithBot ℝ)
    (hmi : s.regs .real [BLK] "m_i" = some (⟨fun idx : TileIndex [BLK] => miFn idx.1⟩ : Tile .real [BLK]))
    (hmij : s.regs .real [BLK] "m_ij" = some (⟨fun idx : TileIndex [BLK] => mijFn idx.1⟩ : Tile .real [BLK])) :
    evalOp ((Op.gt ComparableDType.real Broadcast.nil.consSame (Op.ref .real [BLK] "m_i")
            (Op.ref .real [BLK] "m_ij")).where
        (Op.ref .real [BLK] "m_i") (Op.ref .real [BLK] "m_ij")) s
      = some (⟨fun idx : TileIndex [BLK] => miFn idx.1 ⊔ mijFn idx.1⟩ : Tile .real [BLK]) := by
  rw [evalOp_where, evalOp_gt]
  simp only [evalOp_ref, hmi, hmij, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, u⟩ := idx
  simp only [Tile.select_data, Tile.cop_data, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
    ComparableDType.gt]
  by_cases h : mijFn i < miFn i
  · rw [if_pos (by simpa using h), max_eq_left (le_of_lt h)]
  · rw [if_neg (by simpa using h), max_eq_right (not_lt.mp h)]

set_option maxHeartbeats 1600000 in
/-- General `alpha = tl.exp(m_i - m_i_new)` recipe. -/
theorem nopad_alpha_evalG (s : BlockState) (BLK : Nat) (miFn minewFn : Fin BLK → WithBot ℝ)
    (hmi : s.regs .real [BLK] "m_i" = some (⟨fun idx : TileIndex [BLK] => miFn idx.1⟩ : Tile .real [BLK]))
    (hminew : s.regs .real [BLK] "m_i_new" = some (⟨fun idx : TileIndex [BLK] => minewFn idx.1⟩ : Tile .real [BLK])) :
    evalOp (Op.sub .real Broadcast.nil.consSame (Op.ref .real [BLK] "m_i")
        (Op.ref .real [BLK] "m_i_new")).exp s
      = some (⟨fun idx : TileIndex [BLK] => WithBot.realExp (WithBot.realSub (miFn idx.1) (minewFn idx.1))⟩
          : Tile .real [BLK]) := by
  rw [evalOp_exp, evalOp_sub, evalOp_ref, hmi, evalOp_ref, hminew]
  simp only [Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, u⟩ := idx
  simp only [Tile.uop_data, Tile.bop_data, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.sub]

set_option maxHeartbeats 1600000 in
/-- General `beta = tl.exp(m_ij - m_i_new)` recipe. -/
theorem nopad_beta_evalG (s : BlockState) (BLK : Nat) (mijFn minewFn : Fin BLK → WithBot ℝ)
    (hmij : s.regs .real [BLK] "m_ij" = some (⟨fun idx : TileIndex [BLK] => mijFn idx.1⟩ : Tile .real [BLK]))
    (hminew : s.regs .real [BLK] "m_i_new" = some (⟨fun idx : TileIndex [BLK] => minewFn idx.1⟩ : Tile .real [BLK])) :
    evalOp (Op.sub .real Broadcast.nil.consSame (Op.ref .real [BLK] "m_ij")
        (Op.ref .real [BLK] "m_i_new")).exp s
      = some (⟨fun idx : TileIndex [BLK] => WithBot.realExp (WithBot.realSub (mijFn idx.1) (minewFn idx.1))⟩
          : Tile .real [BLK]) := by
  rw [evalOp_exp, evalOp_sub, evalOp_ref, hmij, evalOp_ref, hminew]
  simp only [Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, u⟩ := idx
  simp only [Tile.uop_data, Tile.bop_data, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.sub]

set_option maxHeartbeats 1600000 in
/-- General `l_i_new = alpha*l_i + beta*l_ij` recipe. -/
theorem nopad_linew_evalG (s : BlockState) (BLK : Nat) (alphaFn liFn betaFn lijFn : Fin BLK → WithBot ℝ)
    (halpha : s.regs .real [BLK] "alpha" = some (⟨fun idx : TileIndex [BLK] => alphaFn idx.1⟩ : Tile .real [BLK]))
    (hli : s.regs .real [BLK] "l_i" = some (⟨fun idx : TileIndex [BLK] => liFn idx.1⟩ : Tile .real [BLK]))
    (hbeta : s.regs .real [BLK] "beta" = some (⟨fun idx : TileIndex [BLK] => betaFn idx.1⟩ : Tile .real [BLK]))
    (hlij : s.regs .real [BLK] "l_ij" = some (⟨fun idx : TileIndex [BLK] => lijFn idx.1⟩ : Tile .real [BLK])) :
    evalOp (Op.add .real Broadcast.nil.consSame
        (Op.mul .real Broadcast.nil.consSame (Op.ref .real [BLK] "alpha") (Op.ref .real [BLK] "l_i"))
        (Op.mul .real Broadcast.nil.consSame (Op.ref .real [BLK] "beta") (Op.ref .real [BLK] "l_ij"))) s
      = some (⟨fun idx : TileIndex [BLK] =>
          WithBot.realAdd (WithBot.realMul (alphaFn idx.1) (liFn idx.1))
            (WithBot.realMul (betaFn idx.1) (lijFn idx.1))⟩ : Tile .real [BLK]) := by
  rw [evalOp_add, evalOp_mul, evalOp_mul]
  simp only [evalOp_ref, halpha, hli, hbeta, hlij, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, u⟩ := idx
  simp only [Tile.bop_data, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.add, NumericDType.mul]

set_option maxHeartbeats 1600000 in
/-- General `p_scale = beta / l_i_new` recipe. -/
theorem nopad_pscale_evalG (s : BlockState) (BLK : Nat) (betaFn linewFn : Fin BLK → WithBot ℝ)
    (hbeta : s.regs .real [BLK] "beta" = some (⟨fun idx : TileIndex [BLK] => betaFn idx.1⟩ : Tile .real [BLK]))
    (hlinew : s.regs .real [BLK] "l_i_new" = some (⟨fun idx : TileIndex [BLK] => linewFn idx.1⟩ : Tile .real [BLK])) :
    evalOp (Op.div .real Broadcast.nil.consSame (Op.ref .real [BLK] "beta")
        (Op.ref .real [BLK] "l_i_new")) s
      = some (⟨fun idx : TileIndex [BLK] => WithBot.realDiv (betaFn idx.1) (linewFn idx.1)⟩ : Tile .real [BLK]) := by
  rw [evalOp_div, evalOp_ref, hbeta, evalOp_ref, hlinew]
  simp only [Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, u⟩ := idx
  simp only [Tile.bop_data, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.div]

set_option maxHeartbeats 1600000 in
/-- General `p = p * p_scale[:,None]` recipe. -/
theorem nopad_pmul_evalG (s : BlockState) (BLK : Nat) (pFn : Fin BLK → Fin BLK → WithBot ℝ)
    (psFn : Fin BLK → WithBot ℝ)
    (hp : s.regs .real [BLK, BLK] "p"
      = some (⟨fun idx : TileIndex [BLK, BLK] => pFn idx.1 idx.2.1⟩ : Tile .real [BLK, BLK]))
    (hps : s.regs .real [BLK] "p_scale" = some (⟨fun idx : TileIndex [BLK] => psFn idx.1⟩ : Tile .real [BLK])) :
    evalOp (Op.mul .real Broadcast.nil.consR.consSame (Op.ref .real [BLK, BLK] "p")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BLK] "p_scale"))) s
      = some (⟨fun idx : TileIndex [BLK, BLK] => WithBot.realMul (pFn idx.1 idx.2.1) (psFn idx.1)⟩
          : Tile .real [BLK, BLK]) := by
  have hexp : @evalOp .real [BLK, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BLK] "p_scale")) s
      = some (Tile.expandDim ⟨1, by simp⟩ (⟨fun idx : TileIndex [BLK] => psFn idx.1⟩ : Tile .real [BLK])) :=
    evalOp_expandDim_ref_of_regs .real [BLK] ⟨1, by simp⟩ "p_scale" s _ hps
  rw [evalOp_mul, evalOp_ref, hp, hexp]
  simp only [Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, jL, u⟩ := idx
  simp only [Tile.bop_data, Tile.bop, Tile.expandDim, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.mul, TileShape.dropInsertedIndex]

set_option maxHeartbeats 1600000 in
/-- General `acc_scale = l_i / l_i_new * alpha` recipe. -/
theorem nopad_accscale_evalG (s : BlockState) (BLK : Nat) (liFn linewFn alphaFn : Fin BLK → WithBot ℝ)
    (hli : s.regs .real [BLK] "l_i" = some (⟨fun idx : TileIndex [BLK] => liFn idx.1⟩ : Tile .real [BLK]))
    (hlinew : s.regs .real [BLK] "l_i_new" = some (⟨fun idx : TileIndex [BLK] => linewFn idx.1⟩ : Tile .real [BLK]))
    (halpha : s.regs .real [BLK] "alpha" = some (⟨fun idx : TileIndex [BLK] => alphaFn idx.1⟩ : Tile .real [BLK])) :
    evalOp (Op.mul .real Broadcast.nil.consSame
        (Op.div .real Broadcast.nil.consSame (Op.ref .real [BLK] "l_i") (Op.ref .real [BLK] "l_i_new"))
        (Op.ref .real [BLK] "alpha")) s
      = some (⟨fun idx : TileIndex [BLK] =>
          WithBot.realMul (WithBot.realDiv (liFn idx.1) (linewFn idx.1)) (alphaFn idx.1)⟩ : Tile .real [BLK]) := by
  rw [evalOp_mul, evalOp_div]
  simp only [evalOp_ref, hli, hlinew, halpha, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, u⟩ := idx
  simp only [Tile.bop_data, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.mul, NumericDType.div]

set_option maxHeartbeats 1600000 in
/-- General `acc = acc * acc_scale[:,None]` recipe. -/
theorem nopad_accmul_evalG (s : BlockState) (BLK DM : Nat) (accFn : Fin BLK → Fin DM → WithBot ℝ)
    (asFn : Fin BLK → WithBot ℝ)
    (hacc : s.regs .real [BLK, DM] "acc"
      = some (⟨fun idx : TileIndex [BLK, DM] => accFn idx.1 idx.2.1⟩ : Tile .real [BLK, DM]))
    (has : s.regs .real [BLK] "acc_scale" = some (⟨fun idx : TileIndex [BLK] => asFn idx.1⟩ : Tile .real [BLK])) :
    evalOp (Op.mul .real Broadcast.nil.consR.consSame (Op.ref .real [BLK, DM] "acc")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BLK] "acc_scale"))) s
      = some (⟨fun idx : TileIndex [BLK, DM] => WithBot.realMul (accFn idx.1 idx.2.1) (asFn idx.1)⟩
          : Tile .real [BLK, DM]) := by
  have hexp : @evalOp .real [BLK, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BLK] "acc_scale")) s
      = some (Tile.expandDim ⟨1, by simp⟩ (⟨fun idx : TileIndex [BLK] => asFn idx.1⟩ : Tile .real [BLK])) :=
    evalOp_expandDim_ref_of_regs .real [BLK] ⟨1, by simp⟩ "acc_scale" s _ has
  rw [evalOp_mul, evalOp_ref, hacc, hexp]
  simp only [Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, d, u⟩ := idx
  simp only [Tile.bop_data, Tile.bop, Tile.expandDim, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.mul, TileShape.dropInsertedIndex]

set_option maxHeartbeats 1600000 in
/-- General `acc += dot(p, v)` recipe. -/
theorem nopad_acc_dot_evalG (s : BlockState) (BLK DM : Nat) (accFn : Fin BLK → Fin DM → WithBot ℝ)
    (pFn : Fin BLK → Fin BLK → ℝ) (vFn : Fin BLK → Fin DM → ℝ)
    (hacc : s.regs .real [BLK, DM] "acc"
      = some (⟨fun idx : TileIndex [BLK, DM] => accFn idx.1 idx.2.1⟩ : Tile .real [BLK, DM]))
    (hp : s.regs .real [BLK, BLK] "p"
      = some (⟨fun idx : TileIndex [BLK, BLK] => some (pFn idx.1 idx.2.1)⟩ : Tile .real [BLK, BLK]))
    (hv : s.regs .real [BLK, DM] "v"
      = some (⟨fun idx : TileIndex [BLK, DM] => some (vFn idx.1 idx.2.1)⟩ : Tile .real [BLK, DM])) :
    evalOp (Op.add .real Broadcast.nil.consSame.consSame (Op.ref .real [BLK, DM] "acc")
        (Op.dot (batch := []) (Op.ref .real [BLK, BLK] "p") (Op.ref .real [BLK, DM] "v"))) s
      = some (⟨fun idx : TileIndex [BLK, DM] =>
          WithBot.realAdd (accFn idx.1 idx.2.1)
            (some (Finset.univ.sum (fun jL : Fin BLK => pFn idx.1 jL * vFn jL idx.2.1)))⟩
          : Tile .real [BLK, DM]) := by
  have hdot : evalOp (Op.dot (batch := []) (Op.ref .real [BLK, BLK] "p") (Op.ref .real [BLK, DM] "v")) s
      = some (Tile.dot [] (⟨fun idx : TileIndex [BLK, BLK] => some (pFn idx.1 idx.2.1)⟩ : Tile .real [BLK, BLK])
          (⟨fun idx : TileIndex [BLK, DM] => some (vFn idx.1 idx.2.1)⟩ : Tile .real [BLK, DM])) := by
    rw [evalOp_dot]; erw [evalOp_ref, hp, evalOp_ref, hv]; rfl
  rw [evalOp_add, evalOp_ref, hacc]
  show Option.bind (evalOp (Op.dot (batch := []) (Op.ref .real [BLK, BLK] "p") (Op.ref .real [BLK, DM] "v")) s) _ = _
  rw [hdot]
  simp only [Option.bind]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, d, u⟩ := idx
  simp only [Tile.bop_data, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add]
  rw [Tile.dot_nil_data]
  rw [show (@Finset.sum (Fin BLK) (WithBot ℝ) _ Finset.univ fun jL : Fin BLK =>
        Option.map₂ (· * ·) ((⟨fun idx : TileIndex [BLK, BLK] => some (pFn idx.1 idx.2.1)⟩ : Tile .real [BLK, BLK]).data (i, jL, PUnit.unit))
          ((⟨fun idx : TileIndex [BLK, DM] => some (vFn idx.1 idx.2.1)⟩ : Tile .real [BLK, DM]).data (jL, d, PUnit.unit)))
      = (@Finset.sum (Fin BLK) (WithBot ℝ) _ Finset.univ fun jL : Fin BLK => ((pFn i jL * vFn jL d : ℝ) : WithBot ℝ)) from by
    apply Finset.sum_congr rfl; intro jL _; rfl]
  rw [WithBot.sum_some_eq_some]; rfl

/-- General output offset (contiguous layout, head stride `hs`). -/
def outOffsetG
    (s : BlockState) (B_Start_Loc : RegionName)
    (rs hs BLK DM : Nat) (idx : TileIndex [BLK, DM]) : Nat :=
  (startLoc s B_Start_Loc + (s.pids 2 * BLK + idx.1.val)) * rs
    + s.pids 1 * hs + idx.2.1.val

/-- General loop invariant after `c` `BLK`-blocks (contiguous layout `(rs, hs, 1)`). -/
noncomputable def nopadInvariantG
    (Q K V : RegionName) (B_Start_Loc B_Seqlen : RegionName)
    (sm_scale : ℝ) (rs hs BLK DM : Nat) (hDM0 : 0 < DM)
    (s0 : BlockState) (c : Nat) (s : BlockState) : Prop :=
  s.pids = s0.pids
  ∧ s.mem = s0.mem
  ∧ (∀ rg o, s.undef rg o = 0)
  ∧ s.regs .nat [] "cur_batch" = some (Tile.scalar (s0.pids 0))
  ∧ s.regs .nat [] "cur_head" = some (Tile.scalar (s0.pids 1))
  ∧ s.regs .nat [] "start_m" = some (Tile.scalar (s0.pids 2))
  ∧ s.regs .nat [] "cur_batch_seq_len" = some (Tile.scalar (seqLen s0 B_Seqlen))
  ∧ s.regs .nat [] "cur_batch_in_all_start_index" = some (Tile.scalar (startLoc s0 B_Start_Loc))
  ∧ s.regs .nat [BLK] "offs_n" = some (Tile.vec (fun j : Fin BLK => j.val))
  ∧ s.regs .nat [DM] "offs_d" = some (Tile.vec (fun e : Fin DM => e.val))
  ∧ s.regs .nat [BLK] "offs_m" = some (Tile.vec (fun i : Fin BLK => s0.pids 2 * BLK + i.val))
  ∧ s.regs .real [BLK, DM] "q" =
      some (⟨fun idx : TileIndex [BLK, DM] =>
        some (ctxQTileMRowG s0 Q B_Start_Loc rs hs BLK DM (seqLen s0 B_Seqlen) (idx.1, idx.2.1, PUnit.unit))⟩
        : Tile .real [BLK, DM])
  ∧ s.regs .ptr [DM, BLK] "k_ptrs" =
      some (⟨fun idx : TileIndex [DM, BLK] =>
        (K, idx.2.1.val * rs + s0.pids 1 * hs + idx.1.val)⟩ : Tile .ptr [DM, BLK])
  ∧ s.regs .ptr [BLK, DM] "v_ptrs" =
      some (⟨fun idx : TileIndex [BLK, DM] =>
        (V, idx.1.val * rs + s0.pids 1 * hs + idx.2.1.val)⟩ : Tile .ptr [BLK, DM])
  ∧ s.regs .nat [] "block_mask" =
      some (Tile.scalar (if BLK * s0.pids 2 < seqLen s0 B_Seqlen then 1 else 0))
  ∧ s.regs .real [BLK] "m_i" =
      some (⟨fun idx : TileIndex [BLK] =>
        (nopadFoldUptoG s0 Q K V B_Start_Loc sm_scale rs hs BLK DM
          (ctxNopadWindowG s0 B_Seqlen BLK) (seqLen s0 B_Seqlen) (c * BLK) idx.1 ⟨0, hDM0⟩).1⟩
        : Tile .real [BLK])
  ∧ s.regs .real [BLK] "l_i" =
      some (⟨fun idx : TileIndex [BLK] =>
        ((nopadFoldUptoG s0 Q K V B_Start_Loc sm_scale rs hs BLK DM
          (ctxNopadWindowG s0 B_Seqlen BLK) (seqLen s0 B_Seqlen) (c * BLK) idx.1 ⟨0, hDM0⟩).2.1 : WithBot ℝ)⟩
        : Tile .real [BLK])
  ∧ s.regs .real [BLK, DM] "acc" =
      some (⟨fun idx : TileIndex [BLK, DM] =>
        ((nopadFoldUptoG s0 Q K V B_Start_Loc sm_scale rs hs BLK DM
          (ctxNopadWindowG s0 B_Seqlen BLK) (seqLen s0 B_Seqlen) (c * BLK) idx.1 idx.2.1).2.2 : WithBot ℝ)⟩
        : Tile .real [BLK, DM])

/-- General `ctxQTileMRowG` depends only on `mem`/`pids`. -/
theorem ctxQTileMRowG_eq_of_mem_pids (s s0 : BlockState) (Q B_Start_Loc : RegionName)
    (rs hs BLK DM bel : Nat) (hmem : s.mem = s0.mem) (hpids : s.pids = s0.pids)
    (idx : TileIndex [BLK, DM]) :
    ctxQTileMRowG s Q B_Start_Loc rs hs BLK DM bel (idx.1, idx.2.1, PUnit.unit)
      = ctxQTileMRowG s0 Q B_Start_Loc rs hs BLK DM bel (idx.1, idx.2.1, PUnit.unit) := by
  simp only [ctxQTileMRowG, ctxQTileG, startLoc, BlockState.readMem, BlockState.readMemValue,
    BlockState.readMemTyped, hmem, hpids]

/-- General `contextAttnNopadExactFoldMG` transports across mem/pids-equal states. -/
theorem contextAttnNopadExactFoldMG_eq_of_mem_pids
    (s s0 : BlockState) (Q K V B_Start_Loc : RegionName) (sm_scale : ℝ) (rs hs BLK DM S bel : Nat)
    (idx : TileIndex [BLK, DM]) (hmem : s.mem = s0.mem) (hpids : s.pids = s0.pids) :
    contextAttnNopadExactFoldMG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel idx
      = contextAttnNopadExactFoldMG s0 Q K V B_Start_Loc sm_scale rs hs BLK DM S bel idx := by
  have hsl : startLoc s B_Start_Loc = startLoc s0 B_Start_Loc := startLoc_eq_of_mem_pids s s0 B_Start_Loc hmem hpids
  have hQt : ctxQTileG s Q B_Start_Loc rs hs BLK DM = ctxQTileG s0 Q B_Start_Loc rs hs BLK DM := by
    funext j; simp only [ctxQTileG, hsl, hpids, BlockState.readMem, hmem]
  have hKt : ctxKTileMG s K B_Start_Loc rs hs S DM bel = ctxKTileMG s0 K B_Start_Loc rs hs S DM bel := by
    funext j; simp only [ctxKTileMG, ctxKTileG, hsl, hpids, BlockState.readMem, hmem]
  have hVt : ctxVTileMG s V B_Start_Loc rs hs S DM bel = ctxVTileMG s0 V B_Start_Loc rs hs S DM bel := by
    funext j; simp only [ctxVTileMG, ctxVTileG, hsl, hpids, BlockState.readMem, hmem]
  simp only [contextAttnNopadExactFoldMG, hQt, hKt, hVt, hpids]

/-- General `ctxNopadGenuineOutValueG` transports across mem/pids-equal states. -/
theorem ctxNopadGenuineOutValueG_eq_of_mem_pids
    (s s0 : BlockState) (Q K V B_Start_Loc B_Seqlen : RegionName) (sm_scale : ℝ) (rs hs BLK DM : Nat)
    (hmem : s.mem = s0.mem) (hpids : s.pids = s0.pids) (idx : TileIndex [BLK, DM]) :
    ctxNopadGenuineOutValueG s Q K V B_Start_Loc B_Seqlen sm_scale rs hs BLK DM idx
      = ctxNopadGenuineOutValueG s0 Q K V B_Start_Loc B_Seqlen sm_scale rs hs BLK DM idx := by
  have hseqE : seqLen s B_Seqlen = seqLen s0 B_Seqlen := seqLen_eq_of_mem_pids s s0 B_Seqlen hmem hpids
  have hWin : ctxNopadWindowG s B_Seqlen BLK = ctxNopadWindowG s0 B_Seqlen BLK := by
    simp only [ctxNopadWindowG, hseqE, hpids]
  have hBel : ctxNopadBel s B_Seqlen = ctxNopadBel s0 B_Seqlen := by simp only [ctxNopadBel, hseqE]
  rw [ctxNopadGenuineOutValueG, ctxNopadGenuineOutValueG,
    contextAttnNopadExactFoldMG_eq_of_mem_pids s s0 Q K V B_Start_Loc sm_scale rs hs BLK DM _ _ idx hmem hpids,
    hWin, hBel]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- General preLoop execution. -/
theorem nopadPreLoop_evalG
    (s : BlockState) (Q K V : RegionName) (B_Start_Loc B_Seqlen : Region .nat)
    (sm_scale : ℝ) (rs hs BLK DM : Nat) (hBLK : 0 < BLK) (hDM : 0 < DM)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ∃ s0, stepStmts (nopadPreLoopG Q K V B_Start_Loc B_Seqlen rs hs BLK DM) s = some s0
      ∧ nopadInvariantG Q K V B_Start_Loc B_Seqlen sm_scale rs hs BLK DM hDM s 0 s0 := by
  unfold nopadPreLoopG
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 0 s))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 1 _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 2 _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.load .nat (MemAccess.region B_Seqlen (Op.ref .nat [] "cur_batch")) MaskOpt.none) _
        = some (Tile.scalar (seqLen s B_Seqlen)) from by
      simp only [evalOp_load_region_none, evalOp_ref, BlockState.setReg_same,
        BlockState.setReg_ne_name, Option.bind, Option.pure_def, seqLen, BlockState.readMemValue]
      rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.load .nat (MemAccess.region B_Start_Loc (Op.ref .nat [] "cur_batch")) MaskOpt.none) _
        = some (Tile.scalar (startLoc s B_Start_Loc)) from by
      simp only [evalOp_load_region_none, evalOp_ref, BlockState.setReg_same,
        BlockState.setReg_ne_name, Option.bind, Option.pure_def, startLoc, BlockState.readMemValue]
      rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.mul .nat Broadcast.nil (Op.constNat BLK) (Op.ref .nat [] "start_m")) _
        = some (Tile.scalar (BLK * s.pids 2)) from by
      rw [evalOp_mul, evalOp_constNat, evalOp_ref]
      simp only [BlockState.setReg_same, BlockState.setReg_ne_name, BlockState.setReg_pids, ne_eq, String.reduceEq,
        not_false_eq_true, Option.bind_eq_bind, Option.bind_some]
      rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.arange BLK) _ = some (Tile.vec (fun j : Fin BLK => j.val)) from evalOp_arange BLK _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.arange DM) _ = some (Tile.vec (fun e : Fin DM => e.val)) from evalOp_arange DM _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .nat Broadcast.scalarL
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat BLK))
          (Op.arange BLK)) _
        = some (Tile.vec (fun i : Fin BLK => s.pids 2 * BLK + i.val)) from by
      rw [evalOp_add, evalOp_mul, evalOp_ref, evalOp_constNat, evalOp_arange]
      simp only [BlockState.setReg_same, BlockState.setReg_ne_name, BlockState.setReg_pids, ne_eq, String.reduceEq,
        not_false_eq_true, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp only [Tile.bop_data, Tile.bop, Tile.scalar, Tile.vec, Broadcast.leftIndex,
        Broadcast.rightIndex, NumericDType.add, NumericDType.mul]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ _ = some (⟨fun idx : TileIndex [BLK, DM] =>
            (startLoc s B_Start_Loc + (s.pids 2 * BLK + idx.1.val)) * rs
              + s.pids 1 * hs + idx.2.1.val⟩ : Tile .nat [BLK, DM]) from
      nopad_offq_evalG _ (startLoc s B_Start_Loc) (s.pids 1) (s.pids 2) rs hs BLK DM
        (by simp)
        (by simp)
        (by simp)
        (by simp [Tile.vec])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ _ = some (⟨fun idx : TileIndex [DM, BLK] =>
            idx.2.1.val * rs + s.pids 1 * hs + idx.1.val⟩ : Tile .nat [DM, BLK]) from
      nopad_offk_evalG _ (s.pids 1) rs hs BLK DM
        (by simp)
        (by simp [Tile.vec])
        (by simp [Tile.vec])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ _ = some (⟨fun idx : TileIndex [BLK, DM] =>
            idx.1.val * rs + s.pids 1 * hs + idx.2.1.val⟩ : Tile .nat [BLK, DM]) from
      nopad_offv_evalG _ (s.pids 1) rs hs BLK DM
        (by simp)
        (by simp [Tile.vec])
        (by simp [Tile.vec])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (nopad_q_load_evalG _ Q B_Start_Loc B_Seqlen rs hs BLK DM
      (by simp [startLoc, BlockState.readMemValue, BlockState.readMemTyped])
      (by simp)
      (by simp [seqLen, BlockState.readMemValue, BlockState.readMemTyped])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase K) (Op.ref .nat [DM, BLK] "off_k")) _
        = some (⟨fun idx : TileIndex [DM, BLK] =>
            (K, idx.2.1.val * rs + s.pids 1 * hs + idx.1.val)⟩ : Tile .ptr [DM, BLK]) from by
      simp only [evalOp, evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name, BlockState.setReg_pids, ne_eq,
        String.reduceEq, not_false_eq_true, BlockState.setReg_pids, Option.bind]
      refine congrArg some (Tile.ext (fun idx => ?_))
      obtain ⟨e, jL, u⟩ := idx
      simp only [Tile.ptrAdd_data, Tile.scalar_data, Broadcast.leftIndex, Broadcast.rightIndex,
        Region.cast_id, Nat.zero_add, Prod.mk.injEq, true_and]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase V) (Op.ref .nat [BLK, DM] "off_v")) _
        = some (⟨fun idx : TileIndex [BLK, DM] =>
            (V, idx.1.val * rs + s.pids 1 * hs + idx.2.1.val)⟩ : Tile .ptr [BLK, DM]) from by
      simp only [evalOp, evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name, BlockState.setReg_pids, ne_eq,
        String.reduceEq, not_false_eq_true, BlockState.setReg_pids, Option.bind]
      refine congrArg some (Tile.ext (fun idx => ?_))
      obtain ⟨jL, d, u⟩ := idx
      simp only [Tile.ptrAdd_data, Tile.scalar_data, Broadcast.leftIndex, Broadcast.rightIndex,
        Region.cast_id, Nat.zero_add, Prod.mk.injEq, true_and]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .real Broadcast.scalarR (Op.full [BLK] (Op.const 0)) Op.negInf) _
        = some (⟨fun _ : TileIndex [BLK] => (⊥ : WithBot ℝ)⟩ : Tile .real [BLK]) from by
      rw [evalOp_add, evalOp_full, evalOp_const, evalOp_negInf]
      simp only [Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp only [Tile.bop_data, Tile.bop, Tile.scalar, Broadcast.leftIndex,
        Broadcast.rightIndex, NumericDType.add]
      rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.full [BLK] (Op.const 0)) _
        = some (⟨fun _ : TileIndex [BLK] => some (0 : ℝ)⟩ : Tile .real [BLK]) from by
      simp [evalOp_full, evalOp_const]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.full [BLK, DM] (Op.const 0)) _
        = some (⟨fun _ : TileIndex [BLK, DM] => some (0 : ℝ)⟩ : Tile .real [BLK, DM]) from by
      simp [evalOp_full, evalOp_const]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp ((Op.lt ComparableDType.nat Broadcast.nil (Op.ref .nat [] "block_start_loc")
            (Op.ref .nat [] "cur_batch_seq_len")).where (Op.constNat 1) (Op.constNat 0)) _
        = some (Tile.scalar (if BLK * s.pids 2 < seqLen s B_Seqlen then 1 else 0)) from by
      rw [evalOp_where, evalOp_lt, evalOp_constNat, evalOp_constNat]
      simp only [evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name, BlockState.setReg_pids, ne_eq,
        String.reduceEq, not_false_eq_true, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp only [Tile.select_data, Tile.cop_data, Tile.bop, Tile.scalar, Broadcast.leftIndex,
        Broadcast.rightIndex, ComparableDType.lt]
      by_cases hlt : BLK * s.pids 2 < seqLen s B_Seqlen
      · rw [if_pos (by simpa [seqLen, BlockState.readMemValue] using hlt), if_pos hlt]
      · rw [if_neg (by simpa [seqLen, BlockState.readMemValue] using hlt), if_neg hlt]))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  refine ⟨by simp, ?_, ?_, by simp, by simp, by simp, by simp, by simp, by simp [Tile.vec], by simp [Tile.vec],
    by simp [Tile.vec], ?_, by simp, by simp, by simp, ?_, ?_, ?_⟩
  · funext rg o; simp
  · intro rg o; simp [hundef]
  · simp only [BlockState.setReg_same, BlockState.setReg_ne_name, BlockState.setReg_pids, ne_eq,
      String.reduceEq, not_false_eq_true]
    refine congrArg some (Tile.ext (fun idx => ?_))
    simp only [ctxQTileMRowG, ctxQTileG, seqLen, startLoc, BlockState.readMem, BlockState.readMemValue,
      BlockState.readMemTyped, BlockState.setReg_mem, BlockState.setReg_pids]
  · simp only [BlockState.setReg_same, BlockState.setReg_ne_name, BlockState.setReg_pids, ne_eq, String.reduceEq,
      not_false_eq_true]
    refine congrArg some (Tile.ext (fun idx => ?_))
    simp only [Nat.zero_mul, nopadFoldUptoG_zero]
  · simp only [BlockState.setReg_same, BlockState.setReg_ne_name, BlockState.setReg_pids, ne_eq, String.reduceEq,
      not_false_eq_true]
    refine congrArg some (Tile.ext (fun idx => ?_))
    simp only [Nat.zero_mul, nopadFoldUptoG_zero]; rfl
  · simp only [BlockState.setReg_same, BlockState.setReg_ne_name, BlockState.setReg_pids, ne_eq, String.reduceEq,
      not_false_eq_true]
    refine congrArg some (Tile.ext (fun idx => ?_))
    simp only [Nat.zero_mul, nopadFoldUptoG_zero]; rfl

/-- General `c = 0` invariant re-anchor to `s0`. -/
theorem nopadInvariantG_zero_reanchor
    (Q K V : RegionName) (B_Start_Loc B_Seqlen : RegionName) (sm_scale : ℝ) (rs hs BLK DM : Nat)
    (hDM : 0 < DM) (s s0 : BlockState)
    (h : nopadInvariantG Q K V B_Start_Loc B_Seqlen sm_scale rs hs BLK DM hDM s 0 s0) :
    nopadInvariantG Q K V B_Start_Loc B_Seqlen sm_scale rs hs BLK DM hDM s0 0 s0 := by
  obtain ⟨hpids, hmem, hundef, hcb, hch, hsm, hseq, hsl, hn, hd, hoffm, hq, hkp, hvp, hbmask, hmi, hli, hacc⟩ := h
  have hseqEq : seqLen s0 B_Seqlen = seqLen s B_Seqlen := seqLen_eq_of_mem_pids s0 s B_Seqlen hmem hpids
  have hslEq : startLoc s0 B_Start_Loc = startLoc s B_Start_Loc := startLoc_eq_of_mem_pids s0 s B_Start_Loc hmem hpids
  refine ⟨rfl, rfl, hundef, ?_, ?_, ?_, ?_, ?_, hn, hd, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hcb, hpids]
  · rw [hch, hpids]
  · rw [hsm, hpids]
  · rw [hseq, hseqEq]
  · rw [hsl, hslEq]
  · rw [hoffm, hpids]
  · rw [hq]; refine congrArg some (Tile.ext (fun idx => ?_))
    show some (ctxQTileMRowG s Q B_Start_Loc rs hs BLK DM (seqLen s B_Seqlen) (idx.1, idx.2.1, PUnit.unit))
      = some (ctxQTileMRowG s0 Q B_Start_Loc rs hs BLK DM (seqLen s0 B_Seqlen) (idx.1, idx.2.1, PUnit.unit))
    rw [hseqEq, ← ctxQTileMRowG_eq_of_mem_pids s0 s Q B_Start_Loc rs hs BLK DM (seqLen s B_Seqlen) hmem hpids idx]
  · rw [hkp, hpids]
  · rw [hvp, hpids]
  · rw [hbmask, hpids, hseqEq]
  · rw [hmi]; refine congrArg some (Tile.ext (fun idx => ?_))
    simp only [Nat.zero_mul, nopadFoldUptoG_zero]
  · rw [hli]; refine congrArg some (Tile.ext (fun idx => ?_))
    simp only [Nat.zero_mul, nopadFoldUptoG_zero]
  · rw [hacc]; refine congrArg some (Tile.ext (fun idx => ?_))
    simp only [Nat.zero_mul, nopadFoldUptoG_zero]

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- General one loop-body step advances the invariant by one block. -/
theorem nopad_attn_stepG
    (Q K V : RegionName) (B_Start_Loc B_Seqlen : RegionName) (sm_scale : ℝ)
    (rs hs BLK DM : Nat) (hBLK : 0 < BLK) (hDM : 0 < DM)
    (s0 : BlockState) (i : Nat) (s : BlockState)
    (hilt : i < ctxNopadWindowG s0 B_Seqlen BLK)
    (hinv : nopadInvariantG Q K V B_Start_Loc B_Seqlen sm_scale rs hs BLK DM hDM s0 (i / BLK) s)
    (hi : i = (i / BLK) * BLK) :
    ∃ s', stepStmts (nopadLoopBodyG sm_scale rs hs BLK DM) (s.setReg "start_n" .nat [] (Tile.scalar i)) = some s'
      ∧ nopadInvariantG Q K V B_Start_Loc B_Seqlen sm_scale rs hs BLK DM hDM s0 (i / BLK + 1) s' := by
  set S := ctxNopadWindowG s0 B_Seqlen BLK with hSdef
  set bel := seqLen s0 B_Seqlen with hbeldef
  set c := i / BLK with hc_def
  set sc := sm_scale with hscdef
  have hwin : (c + 1) * BLK ≤ S := by
    have hSmul : S = (if BLK * s0.pids 2 < bel then 1 else 0) * (s0.pids 2 + 1) * BLK := by
      simp only [hSdef, ctxNopadWindowG, hbeldef]
    by_cases hbm : BLK * s0.pids 2 < bel
    · rw [hSmul, if_pos hbm, one_mul] at hilt ⊢
      have : c < s0.pids 2 + 1 := by
        by_contra hcon
        push_neg at hcon
        have : (s0.pids 2 + 1) * BLK ≤ c * BLK := by exact Nat.mul_le_mul_right BLK hcon
        omega
      have hle : c + 1 ≤ s0.pids 2 + 1 := by omega
      exact Nat.mul_le_mul_right BLK hle
    · rw [hSmul, if_neg hbm] at hilt; omega
  have hcS : c ≤ s0.pids 2 := by
    have hSmul : S = (if BLK * s0.pids 2 < bel then 1 else 0) * (s0.pids 2 + 1) * BLK := by
      simp only [hSdef, ctxNopadWindowG, hbeldef]
    by_cases hbm : BLK * s0.pids 2 < bel
    · rw [hSmul, if_pos hbm, one_mul] at hwin
      by_contra hcon
      push_neg at hcon
      have : (s0.pids 2 + 2) * BLK ≤ (c + 1) * BLK := Nat.mul_le_mul_right BLK (by omega)
      nlinarith [Nat.zero_le BLK]
    · rw [hSmul, if_neg hbm, Nat.zero_mul, Nat.zero_mul] at hwin
      have : (c + 1) * BLK ≥ BLK := by nlinarith [Nat.zero_le (c * BLK)]
      omega
  obtain ⟨hpids, hmem, hundef, hcb, hch, hsm, hseq, hsl, hn, hd, hoffm, hq, hkp, hvp, hbmask, hmi, hli, hacc⟩ := hinv
  set fold0 : Fin BLK → Fin DM → WithBot ℝ × ℝ × ℝ := fun ir dd =>
    nopadFoldUptoG s0 Q K V B_Start_Loc sc rs hs BLK DM S bel (c * BLK) ir dd with hfold0
  set blk : Fin BLK → Fin DM → List (ℝ × ℝ) := fun ir dd =>
    nopadBlockMG s0 Q K V B_Start_Loc sc rs hs BLK DM S bel c ir dd with hblk
  set rawSum : Fin BLK → Fin BLK → ℝ := fun ir jL =>
    Finset.univ.sum (fun e : Fin DM =>
      ctxQTileMRowG s0 Q B_Start_Loc rs hs BLK DM bel (ir, e, PUnit.unit)
        * ctxKTileMG s0 K B_Start_Loc rs hs S DM bel (⟨c * BLK + jL.val, nopad_lane_lt_SG c S BLK hwin jL⟩, e, PUnit.unit))
    with hrawSum
  unfold nopadLoopBodyG
  set s1 := s.setReg "start_n" .nat [] (Tile.scalar i) with hs1d
  have hs1pids : s1.pids = s0.pids := by rw [hs1d, BlockState.setReg_pids]; exact hpids
  have hs1mem : s1.mem = s0.mem := by funext rg o; rw [hs1d, BlockState.setReg_mem]; exact congrFun (congrFun hmem rg) o
  have e1 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "start_n" → s.regs dt sh nm = some t → s1.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs1d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs1sn : s1.regs .nat [] "start_n" = some (Tile.scalar i) := by rw [hs1d, BlockState.setReg_same]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ref .nat [] "start_n") s1 = some (Tile.scalar i) from by rw [evalOp_ref]; exact hs1sn))]
  set s2 := s1.setReg "start_n" .nat [] (Tile.scalar i) with hs2d
  have hs2pids : s2.pids = s0.pids := by rw [hs2d, BlockState.setReg_pids]; exact hs1pids
  have hs2mem : s2.mem = s0.mem := by funext rg o; rw [hs2d, BlockState.setReg_mem]; exact hs1mem ▸ rfl
  have e2 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "start_n" → s1.regs dt sh nm = some t → s2.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs2d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs2sn : s2.regs .nat [] "start_n" = some (Tile.scalar i) := by rw [hs2d, BlockState.setReg_same]
  have hs2seq : s2.regs .nat [] "cur_batch_seq_len" = some (Tile.scalar bel) := e2 (by decide) (e1 (by decide) hseq)
  have hs2slStart : startLoc s2 B_Start_Loc = startLoc s0 B_Start_Loc := startLoc_eq_of_mem_pids s2 s0 B_Start_Loc hs2mem hs2pids
  have hs2sl : s2.regs .nat [] "cur_batch_in_all_start_index" = some (Tile.scalar (startLoc s2 B_Start_Loc)) := by
    rw [hs2slStart]; exact e2 (by decide) (e1 (by decide) hsl)
  have hs2n : s2.regs .nat [BLK] "offs_n" = some (Tile.vec (fun j : Fin BLK => j.val)) := e2 (by decide) (e1 (by decide) hn)
  have hs2m : s2.regs .nat [BLK] "offs_m" = some (Tile.vec (fun ir : Fin BLK => s0.pids 2 * BLK + ir.val)) := e2 (by decide) (e1 (by decide) hoffm)
  have hs2kp : s2.regs .ptr [DM, BLK] "k_ptrs" = some (⟨fun idx : TileIndex [DM, BLK] =>
      (K, idx.2.1.val * rs + s2.pids 1 * hs + idx.1.val)⟩ : Tile .ptr [DM, BLK]) := by
    rw [hs2pids]; exact e2 (by decide) (e1 (by decide) hkp)
  have hs2seqB : seqLen s2 B_Seqlen = bel := by rw [hbeldef]; exact seqLen_eq_of_mem_pids s2 s0 B_Seqlen hs2mem hs2pids
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ s2 = _ from by
      have h := nopad_k_load_evalG s2 K B_Start_Loc B_Seqlen i rs hs BLK DM hs2kp hs2sl hs2sn hs2n (by rw [hs2seqB]; exact hs2seq)
      rw [hs2seqB] at h; exact h))]
  set kFn : Fin DM → Fin BLK → ℝ := fun e jL =>
    ctxKTileMG s0 K B_Start_Loc rs hs S DM bel (⟨c * BLK + jL.val, nopad_lane_lt_SG c S BLK hwin jL⟩, e, PUnit.unit) with hkFn
  rw [show (⟨fun idx : TileIndex [DM, BLK] =>
        if i + idx.2.1.val < bel then
          some (s2.readMem K ((startLoc s2 B_Start_Loc + (i + idx.2.1.val)) * rs + s2.pids 1 * hs + idx.1.val))
        else some (0.0 : ℝ)⟩ : Tile .real [DM, BLK])
      = (⟨fun idx : TileIndex [DM, BLK] => some (kFn idx.1 idx.2.1)⟩ : Tile .real [DM, BLK]) from by
    refine Tile.ext (fun idx => ?_)
    obtain ⟨e, jL, u⟩ := idx
    simp only [hkFn, ctxKTileMG, ctxKTileG, hs2pids,
      show startLoc s2 B_Start_Loc = startLoc s0 B_Start_Loc from startLoc_eq_of_mem_pids s2 s0 B_Start_Loc hs2mem hs2pids,
      show i + jL.val = c * BLK + jL.val from by rw [hi]]
    by_cases hlt : c * BLK + jL.val < bel
    · rw [if_pos hlt, if_pos hlt]
      simp only [BlockState.readMem, hs2mem]
    · rw [if_neg hlt, if_neg hlt]; norm_num]
  set s3 := s2.setReg "k" .real [DM, BLK] (⟨fun idx : TileIndex [DM, BLK] => some (kFn idx.1 idx.2.1)⟩ : Tile .real [DM, BLK]) with hs3d
  have e3 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "k" → s2.regs dt sh nm = some t → s3.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs3d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs3pids : s3.pids = s0.pids := by rw [hs3d, BlockState.setReg_pids]; exact hs2pids
  have hs3mem : s3.mem = s0.mem := by funext rg o; rw [hs3d, BlockState.setReg_mem]; exact hs2mem ▸ rfl
  have hs3k : s3.regs .real [DM, BLK] "k" = some (⟨fun idx : TileIndex [DM, BLK] => some (kFn idx.1 idx.2.1)⟩ : Tile .real [DM, BLK]) := by
    rw [hs3d, BlockState.setReg_same]
  set qFn : Fin BLK → Fin DM → ℝ := fun ir e => ctxQTileMRowG s0 Q B_Start_Loc rs hs BLK DM bel (ir, e, PUnit.unit) with hqFn
  have hs3q : s3.regs .real [BLK, DM] "q" = some (⟨fun idx : TileIndex [BLK, DM] => some (qFn idx.1 idx.2.1)⟩ : Tile .real [BLK, DM]) :=
    e3 (by decide) (e2 (by decide) (e1 (by decide) hq))
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.full [BLK, BLK] (Op.const 0)) s3
        = some (⟨fun _ : TileIndex [BLK, BLK] => some (0 : ℝ)⟩ : Tile .real [BLK, BLK]) from by
      simp [evalOp_full, evalOp_const]))]
  set s4 := s3.setReg "qk" .real [BLK, BLK] (⟨fun _ : TileIndex [BLK, BLK] => some (0 : ℝ)⟩ : Tile .real [BLK, BLK]) with hs4d
  have e4 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "qk" → s3.regs dt sh nm = some t → s4.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs4d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs4qk0 : s4.regs .real [BLK, BLK] "qk" = some (⟨fun _ : TileIndex [BLK, BLK] => some (0 : ℝ)⟩ : Tile .real [BLK, BLK]) := by
    rw [hs4d, BlockState.setReg_same]
  have hs4q : s4.regs .real [BLK, DM] "q" = some (⟨fun idx : TileIndex [BLK, DM] => some (qFn idx.1 idx.2.1)⟩ : Tile .real [BLK, DM]) := e4 (by decide) hs3q
  have hs4k : s4.regs .real [DM, BLK] "k" = some (⟨fun idx : TileIndex [DM, BLK] => some (kFn idx.1 idx.2.1)⟩ : Tile .real [DM, BLK]) := e4 (by decide) hs3k
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (nopad_qk_dot_evalG s4 BLK DM qFn kFn hs4qk0 hs4q hs4k))]
  rw [show (⟨fun idx : TileIndex [BLK, BLK] =>
        some (Finset.univ.sum (fun e : Fin DM => qFn idx.1 e * kFn e idx.2.1))⟩ : Tile .real [BLK, BLK])
      = (⟨fun idx : TileIndex [BLK, BLK] => some (rawSum idx.1 idx.2.1)⟩ : Tile .real [BLK, BLK]) from by
    refine Tile.ext (fun idx => ?_); obtain ⟨ir, jL, u⟩ := idx
    simp only [hrawSum, hqFn, hkFn]]
  set s5 := s4.setReg "qk" .real [BLK, BLK] (⟨fun idx : TileIndex [BLK, BLK] => some (rawSum idx.1 idx.2.1)⟩ : Tile .real [BLK, BLK]) with hs5d
  have e5 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "qk" → s4.regs dt sh nm = some t → s5.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs5d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs5qk : s5.regs .real [BLK, BLK] "qk" = some (⟨fun idx : TileIndex [BLK, BLK] => some (rawSum idx.1 idx.2.1)⟩ : Tile .real [BLK, BLK]) := by
    rw [hs5d, BlockState.setReg_same]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (nopad_qk_scale_evalG s5 BLK sc rawSum hs5qk))]
  set s6 := s5.setReg "qk" .real [BLK, BLK] (⟨fun idx : TileIndex [BLK, BLK] => some (rawSum idx.1 idx.2.1 * sc)⟩ : Tile .real [BLK, BLK]) with hs6d
  have e6 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "qk" → s5.regs dt sh nm = some t → s6.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs6d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs6qk : s6.regs .real [BLK, BLK] "qk" = some (⟨fun idx : TileIndex [BLK, BLK] => some (rawSum idx.1 idx.2.1 * sc)⟩ : Tile .real [BLK, BLK]) := by
    rw [hs6d, BlockState.setReg_same]
  have hs6m : s6.regs .nat [BLK] "offs_m" = some (Tile.vec (fun ir : Fin BLK => s0.pids 2 * BLK + ir.val)) :=
    e6 (by decide) (e5 (by decide) (e4 (by decide) (e3 (by decide) hs2m)))
  have hs6sn : s6.regs .nat [] "start_n" = some (Tile.scalar i) :=
    e6 (by decide) (e5 (by decide) (e4 (by decide) (e3 (by decide) hs2sn)))
  have hs6n : s6.regs .nat [BLK] "offs_n" = some (Tile.vec (fun j : Fin BLK => j.val)) :=
    e6 (by decide) (e5 (by decide) (e4 (by decide) (e3 (by decide) hs2n)))
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (nopad_qk_where_evalG s6 BLK i (fun ir jL => rawSum ir jL * sc) (fun ir => s0.pids 2 * BLK + ir.val)
      hs6qk hs6m hs6sn hs6n))]
  set qkW : Fin BLK → Fin BLK → WithBot ℝ := fun ir jL =>
    if c * BLK + jL.val ≤ s0.pids 2 * BLK + ir.val then ((sc * rawSum ir jL : ℝ) : WithBot ℝ) else (⊥ : WithBot ℝ)
    with hqkW
  rw [show (⟨fun idx : TileIndex [BLK, BLK] =>
        if i + idx.2.1.val ≤ s0.pids 2 * BLK + idx.1.val then some (rawSum idx.1 idx.2.1 * sc)
        else (⊥ : WithBot ℝ)⟩ : Tile .real [BLK, BLK])
      = (⟨fun idx : TileIndex [BLK, BLK] => qkW idx.1 idx.2.1⟩ : Tile .real [BLK, BLK]) from by
    refine Tile.ext (fun idx => ?_); obtain ⟨ir, jL, u⟩ := idx
    simp only [hqkW, show i + jL.val = c * BLK + jL.val from by rw [hi]]
    by_cases hca : c * BLK + jL.val ≤ s0.pids 2 * BLK + ir.val
    · rw [if_pos hca, if_pos hca]; rw [mul_comm]; rfl
    · rw [if_neg hca, if_neg hca]]
  set s7 := s6.setReg "qk" .real [BLK, BLK] (⟨fun idx : TileIndex [BLK, BLK] => qkW idx.1 idx.2.1⟩ : Tile .real [BLK, BLK]) with hs7d
  have e7 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "qk" → s6.regs dt sh nm = some t → s7.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs7d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs7qk : s7.regs .real [BLK, BLK] "qk" = some (⟨fun idx : TileIndex [BLK, BLK] => qkW idx.1 idx.2.1⟩ : Tile .real [BLK, BLK]) := by
    rw [hs7d, BlockState.setReg_same]
  set mijFn : Fin BLK → WithBot ℝ := fun ir =>
    (blk ir ⟨0, hDM⟩).map (fun p => ((p.1 : ℝ) : WithBot ℝ)) |>.foldr (· ⊔ ·) ⊥ with hmijFn
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (nopad_mij_evalG s7 BLK hBLK qkW hs7qk))]
  rw [show (⟨fun idx : TileIndex [BLK] =>
        Finset.univ.sup' (⟨⟨0, hBLK⟩, Finset.mem_univ _⟩ : (Finset.univ : Finset (Fin BLK)).Nonempty) (fun jL : Fin BLK => qkW idx.1 jL)⟩ : Tile .real [BLK])
      = (⟨fun idx : TileIndex [BLK] => mijFn idx.1⟩ : Tile .real [BLK]) from by
    refine Tile.ext (fun idx => ?_); obtain ⟨ir, u⟩ := idx
    show Finset.univ.sup' (⟨⟨0, hBLK⟩, Finset.mem_univ _⟩ : (Finset.univ : Finset (Fin BLK)).Nonempty) (fun jL : Fin BLK => qkW ir jL) = mijFn ir
    rw [Finset.sup'_eq_sup]
    rw [hmijFn]
    simp only [hblk]
    rw [← nopadBlockMG_sup_eq s0 Q K V B_Start_Loc sc rs hs BLK DM S bel c ir ⟨0, hDM⟩ hwin]]
  set s8 := s7.setReg "m_ij" .real [BLK] (⟨fun idx : TileIndex [BLK] => mijFn idx.1⟩ : Tile .real [BLK]) with hs8d
  have e8 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "m_ij" → s7.regs dt sh nm = some t → s8.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs8d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs8qk : s8.regs .real [BLK, BLK] "qk" = some (⟨fun idx : TileIndex [BLK, BLK] => qkW idx.1 idx.2.1⟩ : Tile .real [BLK, BLK]) := e8 (by decide) hs7qk
  have hs8mij : s8.regs .real [BLK] "m_ij" = some (⟨fun idx : TileIndex [BLK] => mijFn idx.1⟩ : Tile .real [BLK]) := by
    rw [hs8d, BlockState.setReg_same]
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (nopad_p_evalG s8 BLK qkW mijFn hs8qk hs8mij))]
  set pFn : Fin BLK → Fin BLK → WithBot ℝ := fun ir jL => WithBot.realExp (WithBot.realSub (qkW ir jL) (mijFn ir)) with hpFn
  set s9 := s8.setReg "p" .real [BLK, BLK] (⟨fun idx : TileIndex [BLK, BLK] => pFn idx.1 idx.2.1⟩ : Tile .real [BLK, BLK]) with hs9d
  have e9 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "p" → s8.regs dt sh nm = some t → s9.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs9d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs9p : s9.regs .real [BLK, BLK] "p" = some (⟨fun idx : TileIndex [BLK, BLK] => pFn idx.1 idx.2.1⟩ : Tile .real [BLK, BLK]) := by
    rw [hs9d, BlockState.setReg_same]
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (nopad_lij_evalG s9 BLK pFn hs9p))]
  set lijFn : Fin BLK → WithBot ℝ := fun ir => Finset.univ.sum (fun jL : Fin BLK => pFn ir jL) with hlijFn
  set s10 := s9.setReg "l_ij" .real [BLK] (⟨fun idx : TileIndex [BLK] => lijFn idx.1⟩ : Tile .real [BLK]) with hs10d
  have e10 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "l_ij" → s9.regs dt sh nm = some t → s10.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs10d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs10lij : s10.regs .real [BLK] "l_ij" = some (⟨fun idx : TileIndex [BLK] => lijFn idx.1⟩ : Tile .real [BLK]) := by
    rw [hs10d, BlockState.setReg_same]
  set miFn : Fin BLK → WithBot ℝ := fun ir => (fold0 ir ⟨0, hDM⟩).1 with hmiFn
  have hs10mi : s10.regs .real [BLK] "m_i" = some (⟨fun idx : TileIndex [BLK] => miFn idx.1⟩ : Tile .real [BLK]) :=
    e10 (by decide) (e9 (by decide) (e8 (by decide) (e7 (by decide) (e6 (by decide) (e5 (by decide) (e4 (by decide) (e3 (by decide) (e2 (by decide) (e1 (by decide) hmi)))))))))
  have hs10mij : s10.regs .real [BLK] "m_ij" = some (⟨fun idx : TileIndex [BLK] => mijFn idx.1⟩ : Tile .real [BLK]) := e10 (by decide) (e9 (by decide) hs8mij)
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (nopad_minew_evalG s10 BLK miFn mijFn hs10mi hs10mij))]
  set minewFn : Fin BLK → WithBot ℝ := fun ir => miFn ir ⊔ mijFn ir with hminewFn
  set s11 := s10.setReg "m_i_new" .real [BLK] (⟨fun idx : TileIndex [BLK] => minewFn idx.1⟩ : Tile .real [BLK]) with hs11d
  have e11 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "m_i_new" → s10.regs dt sh nm = some t → s11.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs11d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs11minew : s11.regs .real [BLK] "m_i_new" = some (⟨fun idx : TileIndex [BLK] => minewFn idx.1⟩ : Tile .real [BLK]) := by
    rw [hs11d, BlockState.setReg_same]
  have hs11mi : s11.regs .real [BLK] "m_i" = some (⟨fun idx : TileIndex [BLK] => miFn idx.1⟩ : Tile .real [BLK]) := e11 (by decide) hs10mi
  have hs11mij : s11.regs .real [BLK] "m_ij" = some (⟨fun idx : TileIndex [BLK] => mijFn idx.1⟩ : Tile .real [BLK]) := e11 (by decide) hs10mij
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (nopad_alpha_evalG s11 BLK miFn minewFn hs11mi hs11minew))]
  set alphaFn : Fin BLK → WithBot ℝ := fun ir => WithBot.realExp (WithBot.realSub (miFn ir) (minewFn ir)) with halphaFn
  set s12 := s11.setReg "alpha" .real [BLK] (⟨fun idx : TileIndex [BLK] => alphaFn idx.1⟩ : Tile .real [BLK]) with hs12d
  have e12 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "alpha" → s11.regs dt sh nm = some t → s12.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs12d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs12alpha : s12.regs .real [BLK] "alpha" = some (⟨fun idx : TileIndex [BLK] => alphaFn idx.1⟩ : Tile .real [BLK]) := by
    rw [hs12d, BlockState.setReg_same]
  have hs12mij : s12.regs .real [BLK] "m_ij" = some (⟨fun idx : TileIndex [BLK] => mijFn idx.1⟩ : Tile .real [BLK]) := e12 (by decide) hs11mij
  have hs12minew : s12.regs .real [BLK] "m_i_new" = some (⟨fun idx : TileIndex [BLK] => minewFn idx.1⟩ : Tile .real [BLK]) := e12 (by decide) hs11minew
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (nopad_beta_evalG s12 BLK mijFn minewFn hs12mij hs12minew))]
  set betaFn : Fin BLK → WithBot ℝ := fun ir => WithBot.realExp (WithBot.realSub (mijFn ir) (minewFn ir)) with hbetaFn
  set s13 := s12.setReg "beta" .real [BLK] (⟨fun idx : TileIndex [BLK] => betaFn idx.1⟩ : Tile .real [BLK]) with hs13d
  have e13 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "beta" → s12.regs dt sh nm = some t → s13.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs13d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs13beta : s13.regs .real [BLK] "beta" = some (⟨fun idx : TileIndex [BLK] => betaFn idx.1⟩ : Tile .real [BLK]) := by
    rw [hs13d, BlockState.setReg_same]
  have hs13alpha : s13.regs .real [BLK] "alpha" = some (⟨fun idx : TileIndex [BLK] => alphaFn idx.1⟩ : Tile .real [BLK]) := e13 (by decide) hs12alpha
  set liFn : Fin BLK → WithBot ℝ := fun ir => ((fold0 ir ⟨0, hDM⟩).2.1 : WithBot ℝ) with hliFn
  have hs13li : s13.regs .real [BLK] "l_i" = some (⟨fun idx : TileIndex [BLK] => liFn idx.1⟩ : Tile .real [BLK]) :=
    e13 (by decide) (e12 (by decide) (e11 (by decide) (e10 (by decide) (e9 (by decide) (e8 (by decide) (e7 (by decide) (e6 (by decide) (e5 (by decide) (e4 (by decide) (e3 (by decide) (e2 (by decide) (e1 (by decide) hli))))))))))))
  have hs13lij : s13.regs .real [BLK] "l_ij" = some (⟨fun idx : TileIndex [BLK] => lijFn idx.1⟩ : Tile .real [BLK]) :=
    e13 (by decide) (e12 (by decide) (e11 (by decide) hs10lij))
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (nopad_linew_evalG s13 BLK alphaFn liFn betaFn lijFn hs13alpha hs13li hs13beta hs13lij))]
  set linewFn : Fin BLK → WithBot ℝ := fun ir =>
    WithBot.realAdd (WithBot.realMul (alphaFn ir) (liFn ir)) (WithBot.realMul (betaFn ir) (lijFn ir)) with hlinewFn
  set s14 := s13.setReg "l_i_new" .real [BLK] (⟨fun idx : TileIndex [BLK] => linewFn idx.1⟩ : Tile .real [BLK]) with hs14d
  have e14 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "l_i_new" → s13.regs dt sh nm = some t → s14.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs14d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs14linew : s14.regs .real [BLK] "l_i_new" = some (⟨fun idx : TileIndex [BLK] => linewFn idx.1⟩ : Tile .real [BLK]) := by
    rw [hs14d, BlockState.setReg_same]
  have hs14beta : s14.regs .real [BLK] "beta" = some (⟨fun idx : TileIndex [BLK] => betaFn idx.1⟩ : Tile .real [BLK]) := e14 (by decide) hs13beta
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (nopad_pscale_evalG s14 BLK betaFn linewFn hs14beta hs14linew))]
  set psFn : Fin BLK → WithBot ℝ := fun ir => WithBot.realDiv (betaFn ir) (linewFn ir) with hpsFn
  set s15 := s14.setReg "p_scale" .real [BLK] (⟨fun idx : TileIndex [BLK] => psFn idx.1⟩ : Tile .real [BLK]) with hs15d
  have e15 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "p_scale" → s14.regs dt sh nm = some t → s15.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs15d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs15ps : s15.regs .real [BLK] "p_scale" = some (⟨fun idx : TileIndex [BLK] => psFn idx.1⟩ : Tile .real [BLK]) := by
    rw [hs15d, BlockState.setReg_same]
  have hs15p : s15.regs .real [BLK, BLK] "p" = some (⟨fun idx : TileIndex [BLK, BLK] => pFn idx.1 idx.2.1⟩ : Tile .real [BLK, BLK]) :=
    e15 (by decide) (e14 (by decide) (e13 (by decide) (e12 (by decide) (e11 (by decide) (e10 (by decide) hs9p)))))
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (nopad_pmul_evalG s15 BLK pFn psFn hs15p hs15ps))]
  set pFinal : Fin BLK → Fin BLK → WithBot ℝ := fun ir jL => WithBot.realMul (pFn ir jL) (psFn ir) with hpFinal
  set s16 := s15.setReg "p" .real [BLK, BLK] (⟨fun idx : TileIndex [BLK, BLK] => pFinal idx.1 idx.2.1⟩ : Tile .real [BLK, BLK]) with hs16d
  have e16 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "p" → s15.regs dt sh nm = some t → s16.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs16d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs16p : s16.regs .real [BLK, BLK] "p" = some (⟨fun idx : TileIndex [BLK, BLK] => pFinal idx.1 idx.2.1⟩ : Tile .real [BLK, BLK]) := by
    rw [hs16d, BlockState.setReg_same]
  have hs16li : s16.regs .real [BLK] "l_i" = some (⟨fun idx : TileIndex [BLK] => liFn idx.1⟩ : Tile .real [BLK]) :=
    e16 (by decide) (e15 (by decide) (e14 (by decide) hs13li))
  have hs16linew : s16.regs .real [BLK] "l_i_new" = some (⟨fun idx : TileIndex [BLK] => linewFn idx.1⟩ : Tile .real [BLK]) :=
    e16 (by decide) (e15 (by decide) hs14linew)
  have hs16alpha : s16.regs .real [BLK] "alpha" = some (⟨fun idx : TileIndex [BLK] => alphaFn idx.1⟩ : Tile .real [BLK]) :=
    e16 (by decide) (e15 (by decide) (e14 (by decide) hs13alpha))
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (nopad_accscale_evalG s16 BLK liFn linewFn alphaFn hs16li hs16linew hs16alpha))]
  set asFn : Fin BLK → WithBot ℝ := fun ir => WithBot.realMul (WithBot.realDiv (liFn ir) (linewFn ir)) (alphaFn ir) with hasFn
  set s17 := s16.setReg "acc_scale" .real [BLK] (⟨fun idx : TileIndex [BLK] => asFn idx.1⟩ : Tile .real [BLK]) with hs17d
  have e17 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "acc_scale" → s16.regs dt sh nm = some t → s17.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs17d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs17as : s17.regs .real [BLK] "acc_scale" = some (⟨fun idx : TileIndex [BLK] => asFn idx.1⟩ : Tile .real [BLK]) := by
    rw [hs17d, BlockState.setReg_same]
  set accFn : Fin BLK → Fin DM → WithBot ℝ := fun ir dd => ((fold0 ir dd).2.2 : WithBot ℝ) with haccFn
  have hs17acc : s17.regs .real [BLK, DM] "acc" = some (⟨fun idx : TileIndex [BLK, DM] => accFn idx.1 idx.2.1⟩ : Tile .real [BLK, DM]) :=
    e17 (by decide) (e16 (by decide) (e15 (by decide) (e14 (by decide) (e13 (by decide) (e12 (by decide) (e11 (by decide) (e10 (by decide) (e9 (by decide) (e8 (by decide) (e7 (by decide) (e6 (by decide) (e5 (by decide) (e4 (by decide) (e3 (by decide) (e2 (by decide) (e1 (by decide) hacc))))))))))))))))
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (nopad_accmul_evalG s17 BLK DM accFn asFn hs17acc hs17as))]
  set accMul : Fin BLK → Fin DM → WithBot ℝ := fun ir dd => WithBot.realMul (accFn ir dd) (asFn ir) with haccMul
  set s18 := s17.setReg "acc" .real [BLK, DM] (⟨fun idx : TileIndex [BLK, DM] => accMul idx.1 idx.2.1⟩ : Tile .real [BLK, DM]) with hs18d
  have e18 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "acc" → s17.regs dt sh nm = some t → s18.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs18d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs18acc : s18.regs .real [BLK, DM] "acc" = some (⟨fun idx : TileIndex [BLK, DM] => accMul idx.1 idx.2.1⟩ : Tile .real [BLK, DM]) := by
    rw [hs18d, BlockState.setReg_same]
  have hs18mem : s18.mem = s0.mem := by
    funext rg o; rw [hs18d, BlockState.setReg_mem, hs17d, BlockState.setReg_mem, hs16d, BlockState.setReg_mem,
      hs15d, BlockState.setReg_mem, hs14d, BlockState.setReg_mem, hs13d, BlockState.setReg_mem,
      hs12d, BlockState.setReg_mem, hs11d, BlockState.setReg_mem, hs10d, BlockState.setReg_mem,
      hs9d, BlockState.setReg_mem, hs8d, BlockState.setReg_mem, hs7d, BlockState.setReg_mem,
      hs6d, BlockState.setReg_mem, hs5d, BlockState.setReg_mem, hs4d, BlockState.setReg_mem]; exact hs3mem ▸ rfl
  have hs18pids : s18.pids = s0.pids := by
    rw [hs18d, BlockState.setReg_pids, hs17d, BlockState.setReg_pids, hs16d, BlockState.setReg_pids,
      hs15d, BlockState.setReg_pids, hs14d, BlockState.setReg_pids, hs13d, BlockState.setReg_pids,
      hs12d, BlockState.setReg_pids, hs11d, BlockState.setReg_pids, hs10d, BlockState.setReg_pids,
      hs9d, BlockState.setReg_pids, hs8d, BlockState.setReg_pids, hs7d, BlockState.setReg_pids,
      hs6d, BlockState.setReg_pids, hs5d, BlockState.setReg_pids, hs4d, BlockState.setReg_pids, hs3pids]
  have hs18pids1 : s18.pids 1 = s0.pids 1 := by rw [hs18pids]
  have hs18seqB : seqLen s18 B_Seqlen = bel := by rw [hbeldef]; exact seqLen_eq_of_mem_pids s18 s0 B_Seqlen hs18mem hs18pids
  have hs18slB : startLoc s18 B_Start_Loc = startLoc s0 B_Start_Loc := startLoc_eq_of_mem_pids s18 s0 B_Start_Loc hs18mem hs18pids
  have hs18vp : s18.regs .ptr [BLK, DM] "v_ptrs" = some (⟨fun idx : TileIndex [BLK, DM] =>
      (V, idx.1.val * rs + s18.pids 1 * hs + idx.2.1.val)⟩ : Tile .ptr [BLK, DM]) := by
    rw [hs18pids1]
    exact e18 (by decide) (e17 (by decide) (e16 (by decide) (e15 (by decide) (e14 (by decide) (e13 (by decide) (e12 (by decide) (e11 (by decide) (e10 (by decide) (e9 (by decide) (e8 (by decide) (e7 (by decide) (e6 (by decide) (e5 (by decide) (e4 (by decide) (e3 (by decide) (e2 (by decide) (e1 (by decide) hvp)))))))))))))))))
  have hs18sl : s18.regs .nat [] "cur_batch_in_all_start_index" = some (Tile.scalar (startLoc s18 B_Start_Loc)) := by
    rw [hs18slB, ← hs2slStart]
    exact e18 (by decide) (e17 (by decide) (e16 (by decide) (e15 (by decide) (e14 (by decide) (e13 (by decide) (e12 (by decide) (e11 (by decide) (e10 (by decide) (e9 (by decide) (e8 (by decide) (e7 (by decide) (e6 (by decide) (e5 (by decide) (e4 (by decide) (e3 (by decide) hs2sl)))))))))))))))
  have hs18sn : s18.regs .nat [] "start_n" = some (Tile.scalar i) :=
    e18 (by decide) (e17 (by decide) (e16 (by decide) (e15 (by decide) (e14 (by decide) (e13 (by decide) (e12 (by decide) (e11 (by decide) (e10 (by decide) (e9 (by decide) (e8 (by decide) (e7 (by decide) hs6sn)))))))))))
  have hs18n : s18.regs .nat [BLK] "offs_n" = some (Tile.vec (fun j : Fin BLK => j.val)) :=
    e18 (by decide) (e17 (by decide) (e16 (by decide) (e15 (by decide) (e14 (by decide) (e13 (by decide) (e12 (by decide) (e11 (by decide) (e10 (by decide) (e9 (by decide) (e8 (by decide) (e7 (by decide) hs6n)))))))))))
  have hs18seq : s18.regs .nat [] "cur_batch_seq_len" = some (Tile.scalar (seqLen s18 B_Seqlen)) := by
    rw [hs18seqB]
    exact e18 (by decide) (e17 (by decide) (e16 (by decide) (e15 (by decide) (e14 (by decide) (e13 (by decide) (e12 (by decide) (e11 (by decide) (e10 (by decide) (e9 (by decide) (e8 (by decide) (e7 (by decide) (e6 (by decide) (e5 (by decide) (e4 (by decide) (e3 (by decide) hs2seq)))))))))))))))
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ s18 = _ from by
      have h := nopad_v_load_evalG s18 V B_Start_Loc B_Seqlen i rs hs BLK DM hs18vp hs18sl hs18sn hs18n hs18seq
      exact h))]
  set vFn : Fin BLK → Fin DM → ℝ := fun jL dd =>
    if i + jL.val < seqLen s18 B_Seqlen then
      s18.readMem V ((startLoc s18 B_Start_Loc + (i + jL.val)) * rs + s18.pids 1 * hs + dd.val)
    else (0.0 : ℝ) with hvFn
  set s19 := s18.setReg "v" .real [BLK, DM] (⟨fun idx : TileIndex [BLK, DM] => some (vFn idx.1 idx.2.1)⟩ : Tile .real [BLK, DM]) with hs19d
  rw [show (⟨fun idx : TileIndex [BLK, DM] =>
        if i + idx.1.val < seqLen s18 B_Seqlen then
          some (s18.readMem V ((startLoc s18 B_Start_Loc + (i + idx.1.val)) * rs + s18.pids 1 * hs + idx.2.1.val))
        else some (0.0 : ℝ)⟩ : Tile .real [BLK, DM])
      = (⟨fun idx : TileIndex [BLK, DM] => some (vFn idx.1 idx.2.1)⟩ : Tile .real [BLK, DM]) from by
    refine Tile.ext (fun idx => ?_); obtain ⟨jL, dd, u⟩ := idx
    show (if i + jL.val < seqLen s18 B_Seqlen then
        some (s18.readMem V ((startLoc s18 B_Start_Loc + (i + jL.val)) * rs + s18.pids 1 * hs + dd.val))
      else some (0.0 : ℝ)) = some (vFn jL dd)
    by_cases hlt : i + jL.val < seqLen s18 B_Seqlen
    · simp only [hvFn, hlt, if_true]
    · simp only [hvFn, hlt, if_false]]
  have e19 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "v" → s18.regs dt sh nm = some t → s19.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs19d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs19v : s19.regs .real [BLK, DM] "v" = some (⟨fun idx : TileIndex [BLK, DM] => some (vFn idx.1 idx.2.1)⟩ : Tile .real [BLK, DM]) := by
    rw [hs19d, BlockState.setReg_same]
  have hs19p : s19.regs .real [BLK, BLK] "p" = some (⟨fun idx : TileIndex [BLK, BLK] => pFinal idx.1 idx.2.1⟩ : Tile .real [BLK, BLK]) := e19 (by decide) hs16p
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ref .real [BLK, BLK] "p") s19 = some (⟨fun idx : TileIndex [BLK, BLK] => pFinal idx.1 idx.2.1⟩ : Tile .real [BLK, BLK]) from by
      rw [evalOp_ref]; exact hs19p))]
  set s20 := s19.setReg "p" .real [BLK, BLK] (⟨fun idx : TileIndex [BLK, BLK] => pFinal idx.1 idx.2.1⟩ : Tile .real [BLK, BLK]) with hs20d
  have e20 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "p" → s19.regs dt sh nm = some t → s20.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs20d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs20p : s20.regs .real [BLK, BLK] "p" = some (⟨fun idx : TileIndex [BLK, BLK] => pFinal idx.1 idx.2.1⟩ : Tile .real [BLK, BLK]) := by
    rw [hs20d, BlockState.setReg_same]
  have hs20v : s20.regs .real [BLK, DM] "v" = some (⟨fun idx : TileIndex [BLK, DM] => some (vFn idx.1 idx.2.1)⟩ : Tile .real [BLK, DM]) := e20 (by decide) hs19v
  have hs20acc : s20.regs .real [BLK, DM] "acc" = some (⟨fun idx : TileIndex [BLK, DM] => accMul idx.1 idx.2.1⟩ : Tile .real [BLK, DM]) :=
    e20 (by decide) (e19 (by decide) hs18acc)
  set pReal : Fin BLK → Fin BLK → ℝ := fun ir jL =>
    (WithBot.realExp (WithBot.realSub (qkW ir jL) (mijFn ir))).unbotD 0 with hpReal
  have hpFnSome : ∀ ir jL : Fin BLK, pFn ir jL = some (pReal ir jL) := by
    intro ir jL
    show WithBot.realExp (WithBot.realSub (qkW ir jL) (mijFn ir))
      = some ((WithBot.realExp (WithBot.realSub (qkW ir jL) (mijFn ir))).unbotD 0)
    exact nopad_realExp_eq_some_unbotD _
  have hlijSome : ∀ ir : Fin BLK, lijFn ir = some (Finset.univ.sum (fun jL : Fin BLK => pReal ir jL)) := by
    intro ir
    show Finset.univ.sum (fun jL : Fin BLK => pFn ir jL) = some (Finset.univ.sum (fun jL : Fin BLK => pReal ir jL))
    rw [show (fun jL : Fin BLK => pFn ir jL) = (fun jL : Fin BLK => (some (pReal ir jL) : WithBot ℝ)) from by
      funext jL; exact hpFnSome ir jL]
    rw [WithBot.sum_someTerm_eq_some]
  set lijReal : Fin BLK → ℝ := fun ir => Finset.univ.sum (fun jL : Fin BLK => pReal ir jL) with hlijReal
  set αReal : Fin BLK → ℝ := fun ir => (WithBot.realExp (WithBot.realSub (miFn ir) (minewFn ir))).unbotD 0 with hαReal
  set βReal : Fin BLK → ℝ := fun ir => (WithBot.realExp (WithBot.realSub (mijFn ir) (minewFn ir))).unbotD 0 with hβReal
  have hαSome : ∀ ir, alphaFn ir = some (αReal ir) := by
    intro ir
    show WithBot.realExp (WithBot.realSub (miFn ir) (minewFn ir))
      = some ((WithBot.realExp (WithBot.realSub (miFn ir) (minewFn ir))).unbotD 0)
    exact nopad_realExp_eq_some_unbotD _
  have hβSome : ∀ ir, betaFn ir = some (βReal ir) := by
    intro ir
    show WithBot.realExp (WithBot.realSub (mijFn ir) (minewFn ir))
      = some ((WithBot.realExp (WithBot.realSub (mijFn ir) (minewFn ir))).unbotD 0)
    exact nopad_realExp_eq_some_unbotD _
  have hliSome : ∀ ir, liFn ir = some ((fold0 ir ⟨0, hDM⟩).2.1) := by intro ir; rfl
  set linewReal : Fin BLK → ℝ := fun ir => αReal ir * (fold0 ir ⟨0, hDM⟩).2.1 + βReal ir * lijReal ir with hlinewReal
  have hlinewSome : ∀ ir, linewFn ir = some (linewReal ir) := by
    intro ir
    show WithBot.realAdd (WithBot.realMul (alphaFn ir) (liFn ir)) (WithBot.realMul (betaFn ir) (lijFn ir))
      = some (linewReal ir)
    rw [hαSome, hliSome, hβSome, hlijSome]; rfl
  set psReal : Fin BLK → ℝ := fun ir => βReal ir / linewReal ir with hpsReal
  have hpsSome : ∀ ir, psFn ir = some (psReal ir) := by
    intro ir
    show WithBot.realDiv (betaFn ir) (linewFn ir) = some (psReal ir)
    rw [hβSome, hlinewSome]; rfl
  have hpFinalSome : ∀ ir jL, pFinal ir jL = some (pReal ir jL * psReal ir) := by
    intro ir jL
    show WithBot.realMul (pFn ir jL) (psFn ir) = some (pReal ir jL * psReal ir)
    rw [hpFnSome, hpsSome]; rfl
  set asReal : Fin BLK → ℝ := fun ir => (fold0 ir ⟨0, hDM⟩).2.1 / linewReal ir * αReal ir with hasReal
  have hasSome : ∀ ir, asFn ir = some (asReal ir) := by
    intro ir
    show WithBot.realMul (WithBot.realDiv (liFn ir) (linewFn ir)) (alphaFn ir) = some (asReal ir)
    rw [hliSome, hlinewSome, hαSome]; rfl
  have haccFnSome : ∀ ir dd, accFn ir dd = some ((fold0 ir dd).2.2) := by intro ir dd; rfl
  have haccMulSome : ∀ ir dd, accMul ir dd = some ((fold0 ir dd).2.2 * asReal ir) := by
    intro ir dd
    show WithBot.realMul (accFn ir dd) (asFn ir) = some ((fold0 ir dd).2.2 * asReal ir)
    rw [haccFnSome, hasSome]; rfl
  rw [show (⟨fun idx : TileIndex [BLK, BLK] => pFinal idx.1 idx.2.1⟩ : Tile .real [BLK, BLK])
        = (⟨fun idx : TileIndex [BLK, BLK] => some (pReal idx.1 idx.2.1 * psReal idx.1)⟩ : Tile .real [BLK, BLK]) from by
    refine Tile.ext (fun idx => ?_); obtain ⟨ir, jL, u⟩ := idx; exact hpFinalSome ir jL] at hs20p
  rw [show (⟨fun idx : TileIndex [BLK, DM] => accMul idx.1 idx.2.1⟩ : Tile .real [BLK, DM])
        = (⟨fun idx : TileIndex [BLK, DM] => some ((fold0 idx.1 idx.2.1).2.2 * asReal idx.1)⟩ : Tile .real [BLK, DM]) from by
    refine Tile.ext (fun idx => ?_); obtain ⟨ir, dd, u⟩ := idx; exact haccMulSome ir dd] at hs20acc
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (nopad_acc_dot_evalG s20 BLK DM (fun ir dd => some ((fold0 ir dd).2.2 * asReal ir))
      (fun ir jL => pReal ir jL * psReal ir) vFn hs20acc hs20p hs20v))]
  set accFinal : Fin BLK → Fin DM → ℝ := fun ir dd =>
    (fold0 ir dd).2.2 * asReal ir + Finset.univ.sum (fun jL : Fin BLK => (pReal ir jL * psReal ir) * vFn jL dd) with haccFinal
  rw [show (⟨fun idx : TileIndex [BLK, DM] =>
        WithBot.realAdd (some ((fold0 idx.1 idx.2.1).2.2 * asReal idx.1))
          (some (Finset.univ.sum (fun jL : Fin BLK => (pReal idx.1 jL * psReal idx.1) * vFn jL idx.2.1)))⟩ : Tile .real [BLK, DM])
      = (⟨fun idx : TileIndex [BLK, DM] => some (accFinal idx.1 idx.2.1)⟩ : Tile .real [BLK, DM]) from by
    refine Tile.ext (fun idx => ?_); obtain ⟨ir, dd, u⟩ := idx
    show WithBot.realAdd (some ((fold0 ir dd).2.2 * asReal ir))
        (some (Finset.univ.sum (fun jL : Fin BLK => (pReal ir jL * psReal ir) * vFn jL dd)))
      = some (accFinal ir dd)
    rfl]
  set s21 := s20.setReg "acc" .real [BLK, DM] (⟨fun idx : TileIndex [BLK, DM] => some (accFinal idx.1 idx.2.1)⟩ : Tile .real [BLK, DM]) with hs21d
  have e21 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "acc" → s20.regs dt sh nm = some t → s21.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs21d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs21acc : s21.regs .real [BLK, DM] "acc" = some (⟨fun idx : TileIndex [BLK, DM] => some (accFinal idx.1 idx.2.1)⟩ : Tile .real [BLK, DM]) := by
    rw [hs21d, BlockState.setReg_same]
  have hs21linew : s21.regs .real [BLK] "l_i_new" = some (⟨fun idx : TileIndex [BLK] => linewFn idx.1⟩ : Tile .real [BLK]) :=
    e21 (by decide) (e20 (by decide) (e19 (by decide) (e18 (by decide) (e17 (by decide) (e16 (by decide) (e15 (by decide) hs14linew))))))
  have hs21minew : s21.regs .real [BLK] "m_i_new" = some (⟨fun idx : TileIndex [BLK] => minewFn idx.1⟩ : Tile .real [BLK]) :=
    e21 (by decide) (e20 (by decide) (e19 (by decide) (e18 (by decide) (e17 (by decide) (e16 (by decide) (e15 (by decide) (e14 (by decide) (e13 (by decide) (e12 (by decide) hs11minew)))))))))
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ref .real [BLK] "l_i_new") s21 = some (⟨fun idx : TileIndex [BLK] => linewFn idx.1⟩ : Tile .real [BLK]) from by
      rw [evalOp_ref]; exact hs21linew))]
  set s22 := s21.setReg "l_i" .real [BLK] (⟨fun idx : TileIndex [BLK] => linewFn idx.1⟩ : Tile .real [BLK]) with hs22d
  have e22 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "l_i" → s21.regs dt sh nm = some t → s22.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs22d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs22minew : s22.regs .real [BLK] "m_i_new" = some (⟨fun idx : TileIndex [BLK] => minewFn idx.1⟩ : Tile .real [BLK]) := e22 (by decide) hs21minew
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ref .real [BLK] "m_i_new") s22 = some (⟨fun idx : TileIndex [BLK] => minewFn idx.1⟩ : Tile .real [BLK]) from by
      rw [evalOp_ref]; exact hs22minew))]
  rw [stepStmts.nil]
  set s23 := s22.setReg "m_i" .real [BLK] (⟨fun idx : TileIndex [BLK] => minewFn idx.1⟩ : Tile .real [BLK]) with hs23d
  refine ⟨s23, rfl, ?_⟩
  have hbridge : ∀ ir : Fin BLK, ∀ dd : Fin DM,
      (minewFn ir, linewReal ir, accFinal ir dd)
        = nopadFoldUptoG s0 Q K V B_Start_Loc sc rs hs BLK DM S bel ((c + 1) * BLK) ir dd := by
    intro ir dd
    rw [nopadFoldUptoMG_succ s0 Q K V B_Start_Loc sc rs hs BLK DM S bel c ir dd hBLK]
    have hbne : nopadBlockMG s0 Q K V B_Start_Loc sc rs hs BLK DM S bel c ir dd ≠ [] :=
      nopadBlockMG_ne_nil s0 Q K V B_Start_Loc sc rs hs BLK DM S bel c ir dd hwin (by have := hcS; nlinarith [Nat.zero_le BLK]) hBLK
    obtain ⟨ha_l, ha_acc, ha_L0, ha_botL, ha_botT, ha_ne⟩ :=
      nopadFoldUptoG_anchor s0 Q K V B_Start_Loc sc rs hs BLK DM S bel (c * BLK) ir dd
    obtain ⟨hci1, hci2⟩ := nopadFoldUptoG_channel_indep s0 Q K V B_Start_Loc sc rs hs BLK DM S bel (c * BLK) ir dd ⟨0, hDM⟩
    have hblockEq := osNormStepBot_block_eq (fold0 ir dd).1 (fold0 ir dd).2.1 (fold0 ir dd).2.2
      (((ctxNopadKeysUptoMG s0 Q K V B_Start_Loc sc rs hs BLK DM S bel (c * BLK) ir dd).map (fun p => Real.exp p.1 * p.2)).sum)
      (((ctxNopadKeysUptoMG s0 Q K V B_Start_Loc sc rs hs BLK DM S bel (c * BLK) ir dd).map (fun p => Real.exp p.1)).sum)
      (nopadBlockMG s0 Q K V B_Start_Loc sc rs hs BLK DM S bel c ir dd)
      hbne ha_L0 ha_l ha_acc ha_botL ha_botT ha_ne
    simp only at hblockEq
    set bsupDD : WithBot ℝ := (nopadBlockMG s0 Q K V B_Start_Loc sc rs hs BLK DM S bel c ir dd).map (fun p => ((p.1 : ℝ) : WithBot ℝ)) |>.foldr (· ⊔ ·) ⊥ with hbsupDD
    have hmijEq : mijFn ir = bsupDD := by
      simp only [hmijFn, hbsupDD, hblk]
      congr 1
      rw [show (fun p : ℝ × ℝ => ((p.1 : ℝ) : WithBot ℝ)) = (fun r : ℝ => ((r : ℝ) : WithBot ℝ)) ∘ Prod.fst from rfl,
        ← List.map_map, ← List.map_map,
        nopadBlockMG_fst_channel_indep s0 Q K V B_Start_Loc sc rs hs BLK DM S bel c ir ⟨0, hDM⟩ dd]
    have hmiEq : miFn ir = (fold0 ir dd).1 := by simp only [hmiFn]; exact hci1.symm
    have hMnew : minewFn ir = (fold0 ir dd).1 ⊔ bsupDD := by simp only [hminewFn]; rw [hmiEq, hmijEq]
    have hliEq : (fold0 ir ⟨0, hDM⟩).2.1 = (fold0 ir dd).2.1 := hci2.symm
    have hαEqB : αReal ir = (WithBot.realExp (WithBot.realSub (fold0 ir dd).1 (minewFn ir))).unbotD 0 := by
      simp only [hαReal]; rw [hmiEq]
    have hβEqB : βReal ir = (WithBot.realExp (WithBot.realSub bsupDD (minewFn ir))).unbotD 0 := by
      simp only [hβReal]; rw [hmijEq]
    have hbsupNe : bsupDD ≠ ⊥ := fun h => hbne ((foldr_sup_coe_bot_iff _).mp h)
    have hbsupCoe : bsupDD = ((bsupDD.unbotD 0 : ℝ) : WithBot ℝ) := by
      cases hb : bsupDD with
      | bot => exact absurd hb hbsupNe
      | coe r => rfl
    have hlijEqB : lijReal ir = ((nopadBlockMG s0 Q K V B_Start_Loc sc rs hs BLK DM S bel c ir dd).map (fun p => Real.exp (p.1 - bsupDD.unbotD 0))).sum := by
      have hMr := nopadBlockMG_lij_sum s0 Q K V B_Start_Loc sc rs hs BLK DM S bel c ir dd (bsupDD.unbotD 0) hwin
      have hpRealEq : ∀ jL : Fin BLK, (pReal ir jL : ℝ)
          = (WithBot.realExp (WithBot.realSub
              (if c * BLK + jL.val ≤ s0.pids 2 * BLK + ir.val then
                ((sc * Finset.univ.sum (fun e : Fin DM =>
                    ctxQTileMRowG s0 Q B_Start_Loc rs hs BLK DM bel (ir, e, PUnit.unit)
                      * ctxKTileMG s0 K B_Start_Loc rs hs S DM bel (⟨c * BLK + jL.val, nopad_lane_lt_SG c S BLK hwin jL⟩, e, PUnit.unit)) : ℝ) : WithBot ℝ)
               else (⊥ : WithBot ℝ)) ((bsupDD.unbotD 0 : ℝ) : WithBot ℝ))).unbotD 0 := by
        intro jL; simp only [hpReal, hqkW, hrawSum, hmijEq]; rw [← hbsupCoe]
      have hsum : ((Finset.univ.sum (fun jL : Fin BLK => pReal ir jL) : ℝ) : WithBot ℝ)
          = some (((nopadBlockMG s0 Q K V B_Start_Loc sc rs hs BLK DM S bel c ir dd).map (fun p => Real.exp (p.1 - bsupDD.unbotD 0))).sum) := by
        rw [← hMr, ← WithBot.sum_some_eq_some]
        apply Finset.sum_congr rfl; intro jL _
        rw [hpRealEq jL]
        exact (nopad_realExp_eq_some_unbotD _).symm
      simp only [hlijReal]
      exact WithBot.coe_inj.mp hsum
    rw [← hblockEq]
    refine Prod.ext ?_ (Prod.ext ?_ ?_)
    · show minewFn ir = (fold0 ir dd).1 ⊔ bsupDD; exact hMnew
    · show linewReal ir = _
      simp only [hlinewReal]; rw [hαEqB, hβEqB, hlijEqB, hliEq, hMnew]
    · show accFinal ir dd = _
      simp only [haccFinal]
      have hasEqB : asReal ir = (fold0 ir dd).2.1 / linewReal ir * αReal ir := by
        simp only [hasReal]; rw [hliEq]
      have hvEq : ∀ jL : Fin BLK, vFn jL dd
          = ctxVTileMG s0 V B_Start_Loc rs hs S DM bel (⟨c * BLK + jL.val, nopad_lane_lt_SG c S BLK hwin jL⟩, dd, PUnit.unit) := by
        intro jL
        show (if i + jL.val < seqLen s18 B_Seqlen then
            s18.readMem V ((startLoc s18 B_Start_Loc + (i + jL.val)) * rs + s18.pids 1 * hs + dd.val)
          else (0.0 : ℝ))
          = ctxVTileMG s0 V B_Start_Loc rs hs S DM bel (⟨c * BLK + jL.val, nopad_lane_lt_SG c S BLK hwin jL⟩, dd, PUnit.unit)
        rw [ctxVTileMG, ctxVTileG, hs18seqB, hs18slB, hs18pids1,
          show i + jL.val = c * BLK + jL.val from by rw [hi]]
        by_cases hlt : c * BLK + jL.val < bel
        · rw [if_pos hlt, if_pos hlt]; simp only [BlockState.readMem, hs18mem]
        · rw [if_neg hlt, if_neg hlt]; norm_num
      have haccsum : Finset.univ.sum (fun jL : Fin BLK => (pReal ir jL * psReal ir) * vFn jL dd)
          = psReal ir * ((nopadBlockMG s0 Q K V B_Start_Loc sc rs hs BLK DM S bel c ir dd).map (fun p => Real.exp (p.1 - bsupDD.unbotD 0) * p.2)).sum := by
        have hbr := nopadBlockMG_acc_sum s0 Q K V B_Start_Loc sc rs hs BLK DM S bel c ir dd (bsupDD.unbotD 0) hwin
          (fun jL => vFn jL dd) (fun jL _ => hvEq jL)
        have hcoe : ((Finset.univ.sum (fun jL : Fin BLK => pReal ir jL * vFn jL dd) : ℝ) : WithBot ℝ)
            = some (((nopadBlockMG s0 Q K V B_Start_Loc sc rs hs BLK DM S bel c ir dd).map (fun p => Real.exp (p.1 - bsupDD.unbotD 0) * p.2)).sum) := by
          rw [← hbr, ← WithBot.sum_some_eq_some]
          apply Finset.sum_congr rfl; intro jL _
          rw [show WithBot.realExp (WithBot.realSub
                (if c * BLK + jL.val ≤ s0.pids 2 * BLK + ir.val then
                  ((sc * Finset.univ.sum (fun e : Fin DM =>
                      ctxQTileMRowG s0 Q B_Start_Loc rs hs BLK DM bel (ir, e, PUnit.unit)
                        * ctxKTileMG s0 K B_Start_Loc rs hs S DM bel (⟨c * BLK + jL.val, nopad_lane_lt_SG c S BLK hwin jL⟩, e, PUnit.unit)) : ℝ) : WithBot ℝ)
                 else (⊥ : WithBot ℝ)) ((bsupDD.unbotD 0 : ℝ) : WithBot ℝ))
              = ((pReal ir jL : ℝ) : WithBot ℝ) from by
            have : pReal ir jL = (WithBot.realExp (WithBot.realSub
                (if c * BLK + jL.val ≤ s0.pids 2 * BLK + ir.val then
                  ((sc * Finset.univ.sum (fun e : Fin DM =>
                      ctxQTileMRowG s0 Q B_Start_Loc rs hs BLK DM bel (ir, e, PUnit.unit)
                        * ctxKTileMG s0 K B_Start_Loc rs hs S DM bel (⟨c * BLK + jL.val, nopad_lane_lt_SG c S BLK hwin jL⟩, e, PUnit.unit)) : ℝ) : WithBot ℝ)
                 else (⊥ : WithBot ℝ)) ((bsupDD.unbotD 0 : ℝ) : WithBot ℝ))).unbotD 0 := by
              simp only [hpReal, hqkW, hrawSum, hmijEq]; rw [← hbsupCoe]
            rw [this]; exact nopad_realExp_eq_some_unbotD _]
          rw [WithBot.realMul_coe_coe]
        rw [show (fun jL : Fin BLK => pReal ir jL * psReal ir * vFn jL dd)
              = (fun jL : Fin BLK => psReal ir * (pReal ir jL * vFn jL dd)) from by funext jL; ring]
        rw [← Finset.mul_sum, WithBot.coe_inj.mp hcoe]
      rw [hasEqB, haccsum, ← hMnew, ← hαEqB, ← hβEqB, ← hlijEqB]
      rw [show αReal ir * (fold0 ir dd).2.1 + βReal ir * lijReal ir = linewReal ir from by
        simp only [hlinewReal]; rw [hliEq]]
      congr 1
      rw [show ((nopadBlockMG s0 Q K V B_Start_Loc sc rs hs BLK DM S bel c ir dd).map (fun p => Real.exp (p.1 - bsupDD.unbotD 0) * βReal ir / linewReal ir * p.2)).sum
            = ((nopadBlockMG s0 Q K V B_Start_Loc sc rs hs BLK DM S bel c ir dd).map (fun p => (Real.exp (p.1 - bsupDD.unbotD 0) * p.2) * psReal ir)).sum from by
        apply congrArg; apply List.map_congr_left; intro p _; rw [hpsReal]; ring]
      rw [List.sum_map_mul_right]; ring
  have e23 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "m_i" → s22.regs dt sh nm = some t → s23.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs23d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have chainAll : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "start_n" → nm ≠ "k" → nm ≠ "qk" → nm ≠ "m_ij" → nm ≠ "p" → nm ≠ "l_ij"
      → nm ≠ "m_i_new" → nm ≠ "alpha" → nm ≠ "beta" → nm ≠ "l_i_new" → nm ≠ "p_scale"
      → nm ≠ "acc_scale" → nm ≠ "acc" → nm ≠ "v" → nm ≠ "l_i" → nm ≠ "m_i"
      → s.regs dt sh nm = some t → s23.regs dt sh nm = some t := by
    intro dt sh nm t h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12 h13 h14 h15 h16 h
    exact e23 h16 (e22 h15 (e21 h13 (e20 h5 (e19 h14 (e18 h13 (e17 h12 (e16 h5 (e15 h11 (e14 h10 (e13 h9 (e12 h8 (e11 h7 (e10 h6 (e9 h5 (e8 h4 (e7 h3 (e6 h3 (e5 h3 (e4 h3 (e3 h2 (e2 h1 (e1 h1 h))))))))))))))))))))))
  have hs23pids : s23.pids = s0.pids := by
    rw [hs23d, BlockState.setReg_pids, hs22d, BlockState.setReg_pids, hs21d, BlockState.setReg_pids,
      hs20d, BlockState.setReg_pids, hs19d, BlockState.setReg_pids, hs18pids]
  have hs23mem : s23.mem = s0.mem := by
    funext rg o; rw [hs23d, BlockState.setReg_mem, hs22d, BlockState.setReg_mem, hs21d, BlockState.setReg_mem,
      hs20d, BlockState.setReg_mem, hs19d, BlockState.setReg_mem]; exact hs18mem ▸ rfl
  have hs23undef : ∀ rg o, s23.undef rg o = 0 := by
    intro rg o
    rw [hs23d, BlockState.setReg_undef, hs22d, BlockState.setReg_undef, hs21d, BlockState.setReg_undef,
      hs20d, BlockState.setReg_undef, hs19d, BlockState.setReg_undef, hs18d, BlockState.setReg_undef,
      hs17d, BlockState.setReg_undef, hs16d, BlockState.setReg_undef, hs15d, BlockState.setReg_undef,
      hs14d, BlockState.setReg_undef, hs13d, BlockState.setReg_undef, hs12d, BlockState.setReg_undef,
      hs11d, BlockState.setReg_undef, hs10d, BlockState.setReg_undef, hs9d, BlockState.setReg_undef,
      hs8d, BlockState.setReg_undef, hs7d, BlockState.setReg_undef, hs6d, BlockState.setReg_undef,
      hs5d, BlockState.setReg_undef, hs4d, BlockState.setReg_undef, hs3d, BlockState.setReg_undef,
      hs2d, BlockState.setReg_undef, hs1d, BlockState.setReg_undef]
    exact hundef rg o
  unfold nopadInvariantG
  refine ⟨hs23pids, hs23mem, hs23undef, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hcb
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hch
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hsm
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hseq
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hsl
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hn
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hd
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hoffm
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hq
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hkp
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hvp
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hbmask
  · rw [hs23d, BlockState.setReg_same]
    refine congrArg some (Tile.ext (fun idx => ?_)); obtain ⟨ir, u⟩ := idx
    show minewFn ir = (nopadFoldUptoG s0 Q K V B_Start_Loc sm_scale rs hs BLK DM (ctxNopadWindowG s0 B_Seqlen BLK) (seqLen s0 B_Seqlen) ((c + 1) * BLK) ir ⟨0, hDM⟩).1
    exact congrArg (Prod.fst) (hbridge ir ⟨0, hDM⟩)
  · rw [show s23.regs .real [BLK] "l_i" = some (⟨fun idx : TileIndex [BLK] => linewFn idx.1⟩ : Tile .real [BLK]) from by
      rw [hs23d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hs22d, BlockState.setReg_same]]
    refine congrArg some (Tile.ext (fun idx => ?_)); obtain ⟨ir, u⟩ := idx
    show linewFn ir = ((nopadFoldUptoG s0 Q K V B_Start_Loc sm_scale rs hs BLK DM (ctxNopadWindowG s0 B_Seqlen BLK) (seqLen s0 B_Seqlen) ((c + 1) * BLK) ir ⟨0, hDM⟩).2.1 : WithBot ℝ)
    rw [hlinewSome ir]
    exact congrArg (fun p => ((p.2.1 : ℝ) : WithBot ℝ)) (hbridge ir ⟨0, hDM⟩)
  · rw [show s23.regs .real [BLK, DM] "acc" = some (⟨fun idx : TileIndex [BLK, DM] => some (accFinal idx.1 idx.2.1)⟩ : Tile .real [BLK, DM]) from by
      rw [hs23d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hs22d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hs21d, BlockState.setReg_same]]
    refine congrArg some (Tile.ext (fun idx => ?_)); obtain ⟨ir, dd, u⟩ := idx
    show some (accFinal ir dd) = ((nopadFoldUptoG s0 Q K V B_Start_Loc sm_scale rs hs BLK DM (ctxNopadWindowG s0 B_Seqlen BLK) (seqLen s0 B_Seqlen) ((c + 1) * BLK) ir dd).2.2 : WithBot ℝ)
    exact congrArg (fun p => ((p.2.2 : ℝ) : WithBot ℝ)) (hbridge ir dd)

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- General postLoop execution + genuine readback (contiguous layout `(rs, hs, 1)`,
`DM ≤ rs` for output-offset injectivity). -/
theorem nopadPostLoop_evalG
    (Q K V : RegionName) (B_Start_Loc B_Seqlen : RegionName) (Out : RegionName)
    (sm_scale : ℝ) (rs hs BLK DM : Nat) (hDM : 0 < DM) (hDMrs : DM ≤ rs)
    (s0 : BlockState) (cF : Nat) (s : BlockState)
    (hfull : cF * BLK = ctxNopadWindowG s0 B_Seqlen BLK)
    (hinv : nopadInvariantG Q K V B_Start_Loc B_Seqlen sm_scale rs hs BLK DM hDM s0 cF s) :
    ∃ sP, stepStmts (nopadPostLoopG Out rs hs BLK DM) s = some sP
      ∧ ∀ idx : TileIndex [BLK, DM],
          sP.readMem Out (outOffsetG s0 B_Start_Loc rs hs BLK DM idx)
            = if s0.pids 2 * BLK + idx.1.val < seqLen s0 B_Seqlen then
                ctxNopadGenuineOutValueG s0 Q K V B_Start_Loc B_Seqlen sm_scale rs hs BLK DM idx
              else s.readMem Out (outOffsetG s0 B_Start_Loc rs hs BLK DM idx) := by
  set bel := seqLen s0 B_Seqlen with hbeldef
  obtain ⟨hpids, hmem, hundef, hcb, hch, hsm, hseq, hsl, hn, hd, hoffm, hq, hkp, hvp, hbmask, hmi, hli, hacc⟩ := hinv
  unfold nopadPostLoopG
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (nopad_offq_evalG s (startLoc s0 B_Start_Loc) (s0.pids 1) (s0.pids 2) rs hs BLK DM hsl hch hoffm hd))]
  set s1 := s.setReg "off_o" .nat [BLK, DM]
    (⟨fun idx : TileIndex [BLK, DM] => (startLoc s0 B_Start_Loc + (s0.pids 2 * BLK + idx.1.val)) * rs + s0.pids 1 * hs + idx.2.1.val⟩ : Tile .nat [BLK, DM]) with hs1d
  have e1 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "off_o" → s.regs dt sh nm = some t → s1.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs1d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs1offo : s1.regs .nat [BLK, DM] "off_o" = some
      (⟨fun idx : TileIndex [BLK, DM] => (startLoc s0 B_Start_Loc + (s0.pids 2 * BLK + idx.1.val)) * rs + s0.pids 1 * hs + idx.2.1.val⟩ : Tile .nat [BLK, DM]) := by
    rw [hs1d, BlockState.setReg_same]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Out) (Op.ref .nat [BLK, DM] "off_o")) s1
        = some (⟨fun idx : TileIndex [BLK, DM] =>
            (Out, (startLoc s0 B_Start_Loc + (s0.pids 2 * BLK + idx.1.val)) * rs + s0.pids 1 * hs + idx.2.1.val)⟩ : Tile .ptr [BLK, DM]) from by
      simp only [evalOp, evalOp_ref, hs1offo, Option.bind]
      refine congrArg some (Tile.ext (fun idx => ?_))
      obtain ⟨ir, dd, u⟩ := idx
      simp only [Tile.ptrAdd_data, Tile.scalar_data, Broadcast.leftIndex, Broadcast.rightIndex,
        Region.cast_id, Nat.zero_add, Prod.mk.injEq, true_and]))]
  set s2 := s1.setReg "out_ptrs" .ptr [BLK, DM]
    (⟨fun idx : TileIndex [BLK, DM] =>
      (Out, (startLoc s0 B_Start_Loc + (s0.pids 2 * BLK + idx.1.val)) * rs + s0.pids 1 * hs + idx.2.1.val)⟩ : Tile .ptr [BLK, DM]) with hs2d
  have e2 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "out_ptrs" → s1.regs dt sh nm = some t → s2.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs2d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs2ptr : s2.regs .ptr [BLK, DM] "out_ptrs" = some
      (⟨fun idx : TileIndex [BLK, DM] => (Out, (startLoc s0 B_Start_Loc + (s0.pids 2 * BLK + idx.1.val)) * rs + s0.pids 1 * hs + idx.2.1.val)⟩ : Tile .ptr [BLK, DM]) := by
    rw [hs2d, BlockState.setReg_same]
  have hs2acc : s2.regs .real [BLK, DM] "acc" = some
      (⟨fun idx : TileIndex [BLK, DM] =>
        ((nopadFoldUptoG s0 Q K V B_Start_Loc sm_scale rs hs BLK DM (ctxNopadWindowG s0 B_Seqlen BLK) (seqLen s0 B_Seqlen) (cF * BLK) idx.1 idx.2.1).2.2 : WithBot ℝ)⟩ : Tile .real [BLK, DM]) :=
    e2 (by decide) (e1 (by decide) hacc)
  have hs2m : s2.regs .nat [BLK] "offs_m" = some (Tile.vec (fun i : Fin BLK => s0.pids 2 * BLK + i.val)) :=
    e2 (by decide) (e1 (by decide) hoffm)
  have hs2seq : s2.regs .nat [] "cur_batch_seq_len" = some (Tile.scalar (seqLen s0 B_Seqlen)) :=
    e2 (by decide) (e1 (by decide) hseq)
  have hs2mem : s2.mem = s0.mem := by
    funext rg o; rw [hs2d, BlockState.setReg_mem, hs1d, BlockState.setReg_mem]; exact hmem ▸ rfl
  have hs2pids : s2.pids = s0.pids := by rw [hs2d, BlockState.setReg_pids, hs1d, BlockState.setReg_pids]; exact hpids
  have hstore : stepStmt (Stmt.store .real [BLK, DM] (MemAccess.ptr (Op.ref .ptr [BLK, DM] "out_ptrs"))
      (Op.ref .real [BLK, DM] "acc")
      (MaskOpt.mask (Op.remap [BLK, DM] Broadcast.nil.consL.consSame.leftIndex
        (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLK] "offs_m"))
          (Op.ref .nat [] "cur_batch_seq_len"))))) s2
      = some ((TileShape.allIndices [BLK, DM]).foldl
          (fun acc idx =>
            if s0.pids 2 * BLK + idx.1.val < seqLen s0 B_Seqlen then
              acc.writeMem Out ((startLoc s0 B_Start_Loc + (s0.pids 2 * BLK + idx.1.val)) * rs + s0.pids 1 * hs + idx.2.1.val)
                ((nopadFoldUptoG s0 Q K V B_Start_Loc sm_scale rs hs BLK DM (ctxNopadWindowG s0 B_Seqlen BLK) (seqLen s0 B_Seqlen) (cF * BLK) idx.1 idx.2.1).2.2 : ℝ)
            else acc) s2) := by
    have hexpM : @evalOp .nat [BLK, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLK] "offs_m")) s2
        = some (Tile.expandDim ⟨1, by simp⟩ (Tile.vec (fun i : Fin BLK => s0.pids 2 * BLK + i.val))) :=
      evalOp_expandDim_ref_of_regs .nat [BLK] ⟨1, by simp⟩ "offs_m" s2 _ hs2m
    unfold stepStmt
    simp only [evalOp_ref, hs2acc, hs2ptr, hs2seq, evalOp, hexpM, Option.bind, Option.map]
    refine congrArg some ?_
    congr 1
    funext acc idx
    obtain ⟨ir, dd, u⟩ := idx
    simp only [Tile.cop_data, Tile.bop_data, Tile.bop, Tile.remap, Tile.expandDim, Tile.vec,
      Tile.scalar, ComparableDType.lt, Broadcast.leftIndex, Broadcast.rightIndex,
      TileShape.dropInsertedIndex, BlockState.writeMemTyped_real, FloatDType.real_storeValue,
      decide_eq_true_eq, WithBot.unbotD_coe]
  rw [stepStmts.cons_some hstore, stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  intro idx
  have hinj : Function.Injective (fun idx : TileIndex [BLK, DM] =>
      (startLoc s0 B_Start_Loc + (s0.pids 2 * BLK + idx.1.val)) * rs + s0.pids 1 * hs + idx.2.1.val) := by
    rintro ⟨⟨ma, hma⟩, ⟨da, hda⟩, _⟩ ⟨⟨mb, hmb⟩, ⟨db, hdb⟩, _⟩ h
    simp only at h
    have hda' : da < rs := lt_of_lt_of_le hda hDMrs
    have hdb' : db < rs := lt_of_lt_of_le hdb hDMrs
    have hm : ma = mb := by nlinarith [h, hda', hdb', Nat.zero_le rs]
    subst hm
    have hd2 : da = db := by omega
    subst hd2; rfl
  rw [show outOffsetG s0 B_Start_Loc rs hs BLK DM idx
        = (fun idx : TileIndex [BLK, DM] => (startLoc s0 B_Start_Loc + (s0.pids 2 * BLK + idx.1.val)) * rs + s0.pids 1 * hs + idx.2.1.val) idx from by
    simp only [outOffsetG]]
  rw [BlockState.scatter_readback_prop_masked_nd s2
    (fun idx : TileIndex [BLK, DM] => (startLoc s0 B_Start_Loc + (s0.pids 2 * BLK + idx.1.val)) * rs + s0.pids 1 * hs + idx.2.1.val)
    (fun idx : TileIndex [BLK, DM] =>
      ((nopadFoldUptoG s0 Q K V B_Start_Loc sm_scale rs hs BLK DM (ctxNopadWindowG s0 B_Seqlen BLK) (seqLen s0 B_Seqlen) (cF * BLK) idx.1 idx.2.1).2.2 : ℝ))
    (fun idx : TileIndex [BLK, DM] => s0.pids 2 * BLK + idx.1.val < seqLen s0 B_Seqlen) hinj idx]
  by_cases hact : s0.pids 2 * BLK + idx.1.val < seqLen s0 B_Seqlen
  · rw [if_pos hact, if_pos hact]
    rw [show cF * BLK = ctxNopadWindowG s0 B_Seqlen BLK from hfull]
    exact nopadFoldUptoG_full_eq_genuine s0 Q K V B_Start_Loc B_Seqlen sm_scale rs hs BLK DM idx
      (by have := idx.1.isLt; omega) hact
  · rw [if_neg hact, if_neg hact]
    simp only [BlockState.readMem, hs2mem, hmem]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **General full-kernel execution chain.** Running the lowered forward body
(preLoop + `forRangeDyn` streaming-softmax loop + postLoop) of the surface at the
contiguous layout `(rs, hs, 1) = (H·DM, DM, 1)` from a clean state reaches a final
state whose `Out` store holds the genuine causal-softmax closed form at every active
output lane and preserves out-of-bounds lanes. -/
theorem nopad_exec_general
    (Q K V : RegionName) (B_Start_Loc B_Seqlen : Region .nat) (Out : RegionName)
    (sm_scale : ℝ) (rs hs BLK DM : Nat) (hBLK : 0 < BLK) (hDM : 0 < DM) (hDMrs : DM ≤ rs)
    (s : BlockState) (hundef : ∀ rg o, s.undef rg o = 0) :
    ∃ sF, stepStmts ((context_attn_nopad_fwd_kernel_surface Q K V sm_scale B_Start_Loc B_Seqlen Out
        rs hs 1 rs hs 1 rs hs 1 rs hs 1 BLK DM BLK).toAlgKernel.body) s = some sF
      ∧ ∀ idx : TileIndex [BLK, DM],
          s.pids 2 * BLK + idx.1.val < seqLen s B_Seqlen →
            sF.readMem Out (outOffsetG s B_Start_Loc rs hs BLK DM idx)
              = ctxNopadGenuineOutValueG s Q K V B_Start_Loc B_Seqlen sm_scale rs hs BLK DM idx := by
  rw [nopad_body_splitG]
  obtain ⟨s0, hpre, hinv0⟩ := nopadPreLoop_evalG s Q K V B_Start_Loc B_Seqlen sm_scale rs hs BLK DM hBLK hDM hundef
  rw [stepStmts.append_some hpre]
  obtain ⟨hpids0, hmem0, hundef0, hcb0, hch0, hsm0, hseq0, hsl0, hn0, hd0, hoffm0, hq0, hkp0, hvp0, hbmask0, hmi0, hli0, hacc0⟩ := hinv0
  have hs0seq : seqLen s0 B_Seqlen = seqLen s B_Seqlen := seqLen_eq_of_mem_pids s0 s B_Seqlen hmem0 hpids0
  set S := ctxNopadWindowG s0 B_Seqlen BLK with hSdef
  have hSmulBLK : S % BLK = 0 := by
    rw [hSdef, ctxNopadWindowG]
    by_cases h : BLK * s0.pids 2 < seqLen s0 B_Seqlen
    · simp only [h, if_true, one_mul]; exact Nat.mul_mod_left _ _
    · simp only [h, if_false, Nat.zero_mul]; exact Nat.zero_mod BLK
  have hbmS : (if BLK * s0.pids 2 < seqLen s0 B_Seqlen then 1 else 0) * (s0.pids 2 + 1) * BLK = S := by
    rw [hSdef, ctxNopadWindowG]
  have hinv0' : nopadInvariantG Q K V B_Start_Loc B_Seqlen sm_scale rs hs BLK DM hDM s0 0 s0 :=
    nopadInvariantG_zero_reanchor Q K V B_Start_Loc B_Seqlen sm_scale rs hs BLK DM hDM s s0 ⟨hpids0, hmem0, hundef0, hcb0, hch0, hsm0, hseq0, hsl0, hn0, hd0, hoffm0, hq0, hkp0, hvp0, hbmask0, hmi0, hli0, hacc0⟩
  obtain ⟨final, sL, hloop, hfin, hinvL⟩ :=
    VeriTile.Triton.forRangeDyn_inv (idx := "start_n")
      (startOp := Op.constNat 0)
      (stopOp := Op.mul .nat Broadcast.nil
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "block_mask")
          (Op.add .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat 1)))
        (Op.constNat BLK))
      (stepOp := Op.constNat BLK)
      (P := fun i st => nopadInvariantG Q K V B_Start_Loc B_Seqlen sm_scale rs hs BLK DM hDM s0 (i / BLK) st ∧ i % BLK = 0 ∧ i ≤ S)
      (s_init := s0) (start := 0) (stop := S) (step := BLK)
      (by rw [evalOp_constNat])
      (by
        rw [evalOp_mul, evalOp_mul, evalOp_ref, hbmask0, evalOp_add, evalOp_ref, hsm0, evalOp_constNat, evalOp_constNat]
        simp only [Option.bind_eq_bind, Option.bind_some]
        refine congrArg some ?_
        refine Tile.ext (fun idx => ?_)
        simp only [Tile.bop_data, Tile.bop, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.mul, NumericDType.add]
        show (if BLK * s.pids 2 < seqLen s B_Seqlen.cast then 1 else 0) * (s.pids 2 + 1) * BLK = S
        rw [hSdef, ctxNopadWindowG, hpids0, hs0seq])
      (by rw [evalOp_constNat])
      (by omega)
      ⟨by rw [Nat.zero_div]; exact hinv0', Nat.zero_mod BLK, Nat.zero_le _⟩
      (fun i st hi hP => by
        obtain ⟨hPinv, hPmod, hPle⟩ := hP
        have hdvd : i = (i / BLK) * BLK := (Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hPmod)).symm
        obtain ⟨s', hstep, hinv'⟩ := nopad_attn_stepG Q K V B_Start_Loc B_Seqlen sm_scale rs hs BLK DM hBLK hDM s0 i st hi hPinv hdvd
        have hmod' : (i + BLK) % BLK = 0 := by rw [Nat.add_mod_right]; exact hPmod
        have hle' : i + BLK ≤ S := by
          have hlt : i < S := hi
          obtain ⟨a, ha⟩ : BLK ∣ i := Nat.dvd_of_mod_eq_zero hPmod
          obtain ⟨b, hb⟩ : BLK ∣ S := Nat.dvd_of_mod_eq_zero hSmulBLK
          have hab : a < b := by
            rw [ha, hb] at hlt
            by_contra hc; push_neg at hc
            have : BLK * b ≤ BLK * a := Nat.mul_le_mul_left BLK hc
            omega
          rw [ha, hb]
          calc BLK * a + BLK = BLK * (a + 1) := by ring
            _ ≤ BLK * b := Nat.mul_le_mul_left BLK (by omega)
        refine ⟨s', hstep, ?_, hmod', hle'⟩
        rw [show (i + BLK) / BLK = i / BLK + 1 from by
          rw [Nat.add_div_right i hBLK]]; exact hinv')
  rw [stepStmts.cons_some hloop]
  obtain ⟨hinvLinv, hinvLmod, hinvLle⟩ := hinvL
  have hfinS : final = S := Nat.le_antisymm hinvLle hfin
  rw [hfinS] at hinvLinv
  have hfull : (S / BLK) * BLK = S := Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hSmulBLK)
  obtain ⟨sP, hpost, hOut⟩ := nopadPostLoop_evalG Q K V B_Start_Loc B_Seqlen Out sm_scale rs hs BLK DM hDM hDMrs s0 (S / BLK) sL
    (by rw [hfull]) hinvLinv
  rw [hpost]
  refine ⟨sP, rfl, ?_⟩
  intro idx hact
  have hslEq0 : startLoc s B_Start_Loc = startLoc s0 B_Start_Loc :=
    startLoc_eq_of_mem_pids s s0 B_Start_Loc hmem0.symm hpids0.symm
  have hoeq : outOffsetG s B_Start_Loc rs hs BLK DM idx = outOffsetG s0 B_Start_Loc rs hs BLK DM idx := by
    simp only [outOffsetG, ← hpids0, hslEq0]
  have hactS0 : s0.pids 2 * BLK + idx.1.val < seqLen s0 B_Seqlen := by
    rw [hs0seq, hpids0]; exact hact
  rw [hoeq, hOut idx, if_pos hactS0]
  exact ctxNopadGenuineOutValueG_eq_of_mem_pids s0 s Q K V B_Start_Loc.cast B_Seqlen.cast sm_scale rs hs BLK DM hmem0 hpids0 idx

/-- General active-output predicate. -/
def activeG (s : BlockState) (B_Seqlen : RegionName) (BLK : Nat)
    (idx : TileIndex [BLK, DM]) : Prop :=
  s.pids 2 * BLK + idx.1.val < seqLen s B_Seqlen

instance activeGDecidable (s : BlockState) (B_Seqlen : RegionName) (BLK DM : Nat)
    (idx : TileIndex [BLK, DM]) : Decidable (activeG s B_Seqlen BLK idx) := by
  unfold activeG; infer_instance

end Correct_without_Rounding

/-! # ══════════ GENERAL — dimension-general public summary ══════════ -/


section General


/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/
/-- **General Public summary for `context_attn_nopad.py`.**

The full faithful `_fwd_kernel` surface (preLoop + streaming-softmax `forRangeDyn`
loop + masked store) *realizes* the genuine causal-softmax closed form
`ctxNopadGenuineOutValueG` — the boundary-masked causal-softmax fold of the loaded
Q/K/V memory — at every active output lane, at the **dimension-parameterized**
contiguous layout `(stride_*bs, stride_*h, stride_*d) = (rs, hs, 1)` with
`BLOCK_M = BLOCK_N = BLK`, `BLOCK_DMODEL = DM`. NOT a self-referential executed
value: the streaming `m_i`/`l_i`/`acc` recurrence is decoded statement-by-statement
and proven to collapse to the closed form. Side conditions: `0 < BLK`, `0 < DM`,
`DM ≤ rs` (output-offset injectivity; contiguous layout has `rs = H·DM ≥ DM`),
`hundef`. Instantiating `BLK = DM = 128`, `rs = 768`, `hs = 128` recovers the
concrete Python test shape. -/
specification context_attn_nopad_output_summary_general
    (Q K V : RegionName) (B_Start_Loc B_Seqlen : Region .nat)
    (Out : RegionName) (sm_scale : ℝ) (rs hs BLK DM : Nat)
    (hBLK : 0 < BLK) (hDM : 0 < DM) (hDMrs : DM ≤ rs)
    (s : BlockState) (hundef : ∀ rg o, s.undef rg o = 0) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := context_attn_nopad_fwd_kernel_surface Q K V sm_scale B_Start_Loc B_Seqlen Out
        rs hs 1 rs hs 1 rs hs 1 rs hs 1 BLK DM BLK)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLK, DM] => activeG s B_Seqlen BLK idx)
        (fun idx : TileIndex [BLK, DM] =>
          (Out, outOffsetG s B_Start_Loc rs hs BLK DM idx)))
      (expected := fun idx : TileIndex [BLK, DM] =>
        ctxNopadGenuineOutValueG s Q K V B_Start_Loc B_Seqlen sm_scale rs hs BLK DM idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [context_attn_nopad_fwd_kernel_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  obtain ⟨sF, hstep, hOut⟩ := nopad_exec_general Q K V B_Start_Loc B_Seqlen Out sm_scale rs hs BLK DM hBLK hDM hDMrs s hundef
  rw [show exec _ s = stepStmts _ s from rfl, hstep] at hExec
  obtain rfl : sF = s' := Option.some.inj hExec
  simp only [ComputeCorrect.OutputReadable.read_real]
  exact hOut idx hActive

end General

/-! # ══════════ IO FACE — the `⊨[R]` streaming-metadata headline ══════════ -/

section IOFace

open scoped VeriTile.Triton.StreamMetaMasked3DKernelIO₃

/-! ## Slot table and IO signature

`context_attn_nopad` is the exemplar consumer of
`StreamMetaMasked3DKernelIO₃` (metadata + three streamed float inputs +
launch-legality `pre`): two `.nat` metadata slots loaded at cell
`cur_batch = pid₀` of their own regions (slot `0` = `B_Seqlen[pid₀]`, the
per-batch sequence length; slot `1` = `B_Start_Loc[pid₀]`, the packed-row
offset), plus the `Q`/`K`/`V` streams whose windows and masks eat the
loaded slot vector. -/

/-- Slot-region table of the two per-batch metadata slots, in the kernel's
own load order: slot `0` = `B_Seqlen` (`cur_batch_seq_len`), slot `1` =
`B_Start_Loc` (`cur_batch_in_all_start_index`). A shared def, never an
inline `match` in a window/spec position. -/
def ctxNopadMetaBuf (B_Start_Loc B_Seqlen : Region .nat) : Fin 2 → RegionName
  | ⟨0, _⟩ => B_Seqlen.cast
  | ⟨_ + 1, _⟩ => B_Start_Loc.cast

/-- **Streaming metadata IO signature** of `context_attn_nopad` on the
metadata-parametrized three-stream fold skin (S1: online-softmax fold +
terminal masked store, 3-D pid grid `(cur_batch, cur_head, start_m)`).

The kernel's `forRangeDyn` trip count `block_mask·(start_m+1)·BLOCK_M/BLOCK_N`
grows with `pid₂`, so the walk has **no pid-free step bound**: `T := NT`, a
new `Nat` parameter (the host grid's third dimension
`cdiv(max_input_len, BLOCK_M)`), and the skin's launch-legality field is

`pre := pid₂ < NT ∧ B_Seqlen[pid₀] ≤ NT·BLOCK_M`

— exactly the port's documented **trusted boundary** (see the file
docstring's Scope section): the host launches
`grid = (batch, head, cdiv(max_input_len, BLOCK_M))`, so every real program
has `start_m < NT`, and every `B_Seqlen[b] ≤ max_input_len ≤ NT·BLOCK_M`.
The `⊨[R]` triple says nothing about launches outside this boundary.

Windows transcribe the kernel's pointer arithmetic exactly, with the loaded
slot vector `m` in place of the in-state metadata reads
(`m 0 = cur_batch_seq_len`, `m 1 = cur_batch_in_all_start_index`), at the
contiguous strides `(rs, hs, 1)` and `BLOCK_M = BLOCK_N = BLK`,
`BLOCK_DMODEL = DM`:

* `read1` (`Q`, the **static** stream — the window ignores `t`): lane
  `j = (i, e)` row-major over `[BLK, DM]` reads
  `(m 1 + (pid₂·BLK + i))·rs + pid₁·hs + e` (the `off_q` cell);
  `mask1` is the row guard `pid₂·BLK + i < m 0` (`offs_m < seq_len`).
* `read2` (`K`, slot-shifted by `t·BLK` **columns** per step): lane
  `j = (e, jL)` over `[DM, BLK]` reads `(m 1 + (t·BLK + jL))·rs + pid₁·hs + e`;
  `mask2` is `t·BLK + jL < m 0` — the **slot-eating step mask** that keeps
  the dead tail steps (`t·BLK ≥` the live window) unpinned.
* `read3` (`V`, mirror at `[BLK, DM]`): lane `j = (jL, e)` reads
  `(m 1 + (t·BLK + jL))·rs + pid₁·hs + e`; `mask3` is `t·BLK + jL < m 0`.
* `write` (`Out`, the terminal store): lane `j = (i, e)` writes
  `(m 1 + (pid₂·BLK + i))·rs + pid₁·hs + e`; `writeMask` is the row guard
  (the kernel's `offs_m < cur_batch_seq_len` store mask).

`outDType` is the `.real` default: the loop normalizes in place and the
terminal `tl.store` is untyped, so there is no quantization event. -/
def contextAttnNopadIO (Q K V : RegionName) (B_Start_Loc B_Seqlen : Region .nat)
    (Out : RegionName) (sm_scale : ℝ) (rs hs BLK DM NT : Nat) :
    StreamMetaMasked3DKernelIO₃ where
  kernel := context_attn_nopad_fwd_kernel_surface Q K V sm_scale B_Start_Loc B_Seqlen Out
    rs hs 1 rs hs 1 rs hs 1 rs hs 1 BLK DM BLK
  inp1 := Q
  inp2 := K
  inp3 := V
  out := Out
  nMeta := 2
  sty := fun _ => ChanTy.nat
  mbuf := ctxNopadMetaBuf B_Start_Loc B_Seqlen
  mwin := fun _ pid₀ _ _ => pid₀
  T := NT
  B1 := BLK * DM
  B2 := DM * BLK
  B3 := BLK * DM
  C := BLK * DM
  pre := fun _ _ pid₂ m => pid₂ < NT ∧ m (⟨0, by omega⟩ : Fin 2) ≤ NT * BLK
  read1 := fun _ pid₁ pid₂ m _ j =>
    (m (⟨1, by omega⟩ : Fin 2) + (pid₂ * BLK + j.val / DM)) * rs + pid₁ * hs + j.val % DM
  read2 := fun _ pid₁ _ m t j =>
    (m (⟨1, by omega⟩ : Fin 2) + (t.val * BLK + j.val % BLK)) * rs + pid₁ * hs + j.val / BLK
  read3 := fun _ pid₁ _ m t j =>
    (m (⟨1, by omega⟩ : Fin 2) + (t.val * BLK + j.val / DM)) * rs + pid₁ * hs + j.val % DM
  write := fun _ pid₁ pid₂ m j =>
    (m (⟨1, by omega⟩ : Fin 2) + (pid₂ * BLK + j.val / DM)) * rs + pid₁ * hs + j.val % DM
  mask1 := fun _ _ pid₂ m _ j => pid₂ * BLK + j.val / DM < m (⟨0, by omega⟩ : Fin 2)
  mask2 := fun _ _ _ m t j => t.val * BLK + j.val % BLK < m (⟨0, by omega⟩ : Fin 2)
  mask3 := fun _ _ _ m t j => t.val * BLK + j.val / DM < m (⟨0, by omega⟩ : Fin 2)
  writeMask := fun _ _ pid₂ m j => pid₂ * BLK + j.val / DM < m (⟨0, by omega⟩ : Fin 2)

/-! ## Stream-indexed tiles and the streamed closed form -/

/-- The `Q` tile read off the (static) first stream: the window ignores `t`,
so the step-`0` slice carries the whole `[BLK, DM]` tile. -/
noncomputable def ctxNopadIOqT (BLK DM NT : Nat) (hNT : 0 < NT)
    (xs : Fin NT → Fin (BLK * DM) → ℝ) : TileIndex [BLK, DM] → ℝ :=
  fun idx => xs ⟨0, hNT⟩ (Lane2D.encode idx)

/-- The boundary-masked global `K` tile read off the second stream: global
key `jg` lives in step `jg / BLK`, block-local column `jg % BLK`, at stream
lane `(e, jg % BLK)`; keys at or beyond the sequence boundary `m0` are `0`
(the kernel's `other=0.0` load mask). -/
noncomputable def ctxNopadIOkT (BLK DM NT m0 : Nat) (hBLK : 0 < BLK)
    (ys : Fin NT → Fin (DM * BLK) → ℝ) : TileIndex [NT * BLK, DM] → ℝ :=
  fun idx =>
    if idx.1.val < m0 then
      ys ⟨idx.1.val / BLK, (Nat.div_lt_iff_lt_mul hBLK).mpr idx.1.isLt⟩
        (Lane2D.encode (idx.2.1, ⟨idx.1.val % BLK, Nat.mod_lt _ hBLK⟩, PUnit.unit))
    else 0

/-- The boundary-masked global `V` tile read off the third stream: global
key `jg` at stream lane `(jg % BLK, e)`. -/
noncomputable def ctxNopadIOvT (BLK DM NT m0 : Nat) (hBLK : 0 < BLK)
    (zs : Fin NT → Fin (BLK * DM) → ℝ) : TileIndex [NT * BLK, DM] → ℝ :=
  fun idx =>
    if idx.1.val < m0 then
      zs ⟨idx.1.val / BLK, (Nat.div_lt_iff_lt_mul hBLK).mpr idx.1.isLt⟩
        (Lane2D.encode (⟨idx.1.val % BLK, Nat.mod_lt _ hBLK⟩, idx.2.1, PUnit.unit))
    else 0

/-- **The streamed closed form**: `contextAttnNopadExactFoldMG` — the honest
causal softmax `Σ_{jg ≤ gi} exp(sm_scale·raw)·v / Σ_{jg ≤ gi} exp(sm_scale·raw)`
with boundary-masked `K`/`V` — restated over the three streamed tiles on the
`pre`-legal global-key window `[0, NT·BLK)` (which covers the kernel's live
window `(pid₂+1)·BLK` for every legal launch; causality `jg ≤ gi` cuts the
extension to the same keys). Output lane `j = (i, e)` row-major over
`[BLK, DM]`. -/
noncomputable def contextAttnNopadIOSpec (BLK DM NT m0 : Nat) (hBLK : 0 < BLK)
    (hNT : 0 < NT) (sm_scale : ℝ) (pid₂ : Nat)
    (xs : Fin NT → Fin (BLK * DM) → ℝ) (ys : Fin NT → Fin (DM * BLK) → ℝ)
    (zs : Fin NT → Fin (BLK * DM) → ℝ) (j : Fin (BLK * DM)) : ℝ :=
  let i := (Lane2D.decode j).1
  let d := (Lane2D.decode j).2.1
  let gi := pid₂ * BLK + i.val
  let raw := fun jg : Fin (NT * BLK) =>
    Finset.univ.sum (fun e : Fin DM =>
      ctxNopadIOqT BLK DM NT hNT xs (i, e, PUnit.unit)
        * ctxNopadIOkT BLK DM NT m0 hBLK ys (jg, e, PUnit.unit))
  let weight := fun jg : Fin (NT * BLK) =>
    if jg.val ≤ gi then Real.exp (sm_scale * raw jg) else 0
  let denom := Finset.univ.sum (fun jg : Fin (NT * BLK) => weight jg)
  let numer := Finset.univ.sum (fun jg : Fin (NT * BLK) =>
    weight jg * ctxNopadIOvT BLK DM NT m0 hBLK zs (jg, d, PUnit.unit))
  numer / denom

/-- Causal-sum window extension: a `Fin`-indexed real sum whose integrand
vanishes above `gi` is invariant under enlarging the window past `gi`. -/
private theorem ctxNopadIO_sum_extend (S S' gi : Nat) (hSS : S ≤ S') (hgi : gi < S)
    (F : Nat → ℝ) (h0 : ∀ n, gi < n → F n = 0) :
    (∑ jf : Fin S, F jf.val) = ∑ jf : Fin S', F jf.val := by
  rw [Fin.sum_univ_eq_sum_range, Fin.sum_univ_eq_sum_range]
  refine Finset.sum_subset
    (fun x hx => Finset.mem_range.mpr (lt_of_lt_of_le (Finset.mem_range.mp hx) hSS)) ?_
  intro x _ hxn
  exact h0 x (by have := Finset.mem_range.not.mp hxn; omega)

/-- The `Nat`-indexed denominator integrand shared by both spellings of the
causal softmax (memory-side over the live window `Fin ((pid₂+1)·BLK)`,
stream-side over the legal window `Fin (NT·BLK)`): key `n`'s weight
`exp(sm_scale·Σ_e Q[row gi, e]·K_masked[n, e])`, causal-guarded. -/
private noncomputable def ctxNopadIOFden (s : BlockState) (Q K : RegionName)
    (sm_scale : ℝ) (rs hs DM m0 m1 gi p1 : Nat) : Nat → ℝ :=
  fun n =>
    if n ≤ gi then
      Real.exp (sm_scale * ∑ e : Fin DM,
        s.readMem Q ((m1 + gi) * rs + p1 * hs + e.val)
          * (if n < m0 then s.readMem K ((m1 + n) * rs + p1 * hs + e.val) else 0))
    else 0

/-- The `Nat`-indexed numerator integrand: the weight times the
boundary-masked `V` cell of channel `dv`. -/
private noncomputable def ctxNopadIOFnum (s : BlockState) (Q K V : RegionName)
    (sm_scale : ℝ) (rs hs DM m0 m1 gi p1 dv : Nat) : Nat → ℝ :=
  fun n =>
    ctxNopadIOFden s Q K sm_scale rs hs DM m0 m1 gi p1 n
      * (if n < m0 then s.readMem V ((m1 + n) * rs + p1 * hs + dv) else 0)

set_option maxHeartbeats 3200000 in
/-- **Stream-spec bridge**: at a legal launch (`pid₂ < NT`) and an active
output row, the port's genuine closed form `ctxNopadGenuineOutValueG` *is*
the streamed closed form `contextAttnNopadIOSpec`, under the skin's input
pins. Both sides collapse onto the shared `Nat`-indexed integrands
(`ctxNopadIOFden`/`ctxNopadIOFnum`); causality (`jg ≤ gi < (pid₂+1)·BLK ≤
NT·BLK`) makes the window extension inert (`ctxNopadIO_sum_extend`). -/
private theorem ctxNopadIOSpec_eq_genuine
    (s : BlockState) (Q K V : RegionName) (B_Start_Loc B_Seqlen : Region .nat)
    (sm_scale : ℝ) (rs hs BLK DM NT m0 m1 : Nat) (hBLK : 0 < BLK) (hNT : 0 < NT)
    (hpid : s.pids 2 < NT)
    (hm0 : seqLen s B_Seqlen.cast = m0) (hm1 : startLoc s B_Start_Loc.cast = m1)
    (xs : Fin NT → Fin (BLK * DM) → ℝ) (ys : Fin NT → Fin (DM * BLK) → ℝ)
    (zs : Fin NT → Fin (BLK * DM) → ℝ)
    (hx : ∀ (t : Fin NT) (l : Fin (BLK * DM)), s.pids 2 * BLK + l.val / DM < m0 →
      s.readMem Q ((m1 + (s.pids 2 * BLK + l.val / DM)) * rs + s.pids 1 * hs + l.val % DM)
        = xs t l)
    (hy : ∀ (t : Fin NT) (l : Fin (DM * BLK)), t.val * BLK + l.val % BLK < m0 →
      s.readMem K ((m1 + (t.val * BLK + l.val % BLK)) * rs + s.pids 1 * hs + l.val / BLK)
        = ys t l)
    (hz : ∀ (t : Fin NT) (l : Fin (BLK * DM)), t.val * BLK + l.val / DM < m0 →
      s.readMem V ((m1 + (t.val * BLK + l.val / DM)) * rs + s.pids 1 * hs + l.val % DM)
        = zs t l)
    (j : Fin (BLK * DM)) (hact : s.pids 2 * BLK + j.val / DM < m0) :
    ctxNopadGenuineOutValueG s Q K V B_Start_Loc.cast B_Seqlen.cast sm_scale rs hs BLK DM
        (Lane2D.decode j)
      = contextAttnNopadIOSpec BLK DM NT m0 hBLK hNT sm_scale (s.pids 2) xs ys zs j := by
  have hiv : (Lane2D.decode j).1.val = j.val / DM := Lane2D.decode_row j
  have hdv : (Lane2D.decode j).2.1.val = j.val % DM := Lane2D.decode_col j
  have hact' : s.pids 2 * BLK + (Lane2D.decode j).1.val < m0 := by rw [hiv]; exact hact
  have hbm : BLK * s.pids 2 < seqLen s B_Seqlen.cast := by
    rw [hm0]
    calc BLK * s.pids 2 = s.pids 2 * BLK := Nat.mul_comm _ _
      _ ≤ s.pids 2 * BLK + (Lane2D.decode j).1.val := Nat.le_add_right _ _
      _ < m0 := hact'
  have hS : ctxNopadWindowG s B_Seqlen.cast BLK = (s.pids 2 + 1) * BLK := by
    rw [ctxNopadWindowG]
    show (if BLK * s.pids 2 < seqLen s B_Seqlen.cast then 1 else 0) * (s.pids 2 + 1) * BLK
      = (s.pids 2 + 1) * BLK
    rw [if_pos hbm, one_mul]
  have hgiS : s.pids 2 * BLK + (Lane2D.decode j).1.val < (s.pids 2 + 1) * BLK := by
    have h := (Lane2D.decode j).1.isLt
    calc s.pids 2 * BLK + (Lane2D.decode j).1.val
        < s.pids 2 * BLK + BLK := Nat.add_lt_add_left h _
      _ = (s.pids 2 + 1) * BLK := (Nat.succ_mul _ _).symm
  have hSle : (s.pids 2 + 1) * BLK ≤ NT * BLK := Nat.mul_le_mul_right _ hpid
  have hden0 : ∀ n, s.pids 2 * BLK + (Lane2D.decode j).1.val < n →
      ctxNopadIOFden s Q K sm_scale rs hs DM m0 m1
        (s.pids 2 * BLK + (Lane2D.decode j).1.val) (s.pids 1) n = 0 := by
    intro n hn
    rw [ctxNopadIOFden]
    exact if_neg (by omega)
  have hnum0 : ∀ n, s.pids 2 * BLK + (Lane2D.decode j).1.val < n →
      ctxNopadIOFnum s Q K V sm_scale rs hs DM m0 m1
        (s.pids 2 * BLK + (Lane2D.decode j).1.val) (s.pids 1) (j.val % DM) n = 0 := by
    intro n hn
    rw [ctxNopadIOFnum, hden0 n hn, zero_mul]
  rw [ctxNopadGenuineOutValueG, hS, ctxNopadBel, hm0]
  refine Eq.trans (b :=
    (∑ jg : Fin (NT * BLK), ctxNopadIOFnum s Q K V sm_scale rs hs DM m0 m1
        (s.pids 2 * BLK + (Lane2D.decode j).1.val) (s.pids 1) (j.val % DM) jg.val)
      / (∑ jg : Fin (NT * BLK), ctxNopadIOFden s Q K sm_scale rs hs DM m0 m1
          (s.pids 2 * BLK + (Lane2D.decode j).1.val) (s.pids 1) jg.val)) ?_ ?_
  · -- memory side = the shared integrand over the live window, then extend
    rw [contextAttnNopadExactFoldMG]
    have hnum : (∑ jS : Fin ((s.pids 2 + 1) * BLK),
          (if jS.val ≤ s.pids 2 * BLK + (Lane2D.decode j).1.val then
            Real.exp (sm_scale * ∑ e : Fin DM,
              ctxQTileG s Q B_Start_Loc.cast rs hs BLK DM ((Lane2D.decode j).1, e, PUnit.unit)
                * ctxKTileMG s K B_Start_Loc.cast rs hs ((s.pids 2 + 1) * BLK) DM m0
                    (jS, e, PUnit.unit))
          else 0)
            * ctxVTileMG s V B_Start_Loc.cast rs hs ((s.pids 2 + 1) * BLK) DM m0
                (jS, (Lane2D.decode j).2.1, PUnit.unit))
        = ∑ jg : Fin (NT * BLK), ctxNopadIOFnum s Q K V sm_scale rs hs DM m0 m1
            (s.pids 2 * BLK + (Lane2D.decode j).1.val) (s.pids 1) (j.val % DM) jg.val := by
      rw [show (∑ jS : Fin ((s.pids 2 + 1) * BLK),
            (if jS.val ≤ s.pids 2 * BLK + (Lane2D.decode j).1.val then
              Real.exp (sm_scale * ∑ e : Fin DM,
                ctxQTileG s Q B_Start_Loc.cast rs hs BLK DM ((Lane2D.decode j).1, e, PUnit.unit)
                  * ctxKTileMG s K B_Start_Loc.cast rs hs ((s.pids 2 + 1) * BLK) DM m0
                      (jS, e, PUnit.unit))
            else 0)
              * ctxVTileMG s V B_Start_Loc.cast rs hs ((s.pids 2 + 1) * BLK) DM m0
                  (jS, (Lane2D.decode j).2.1, PUnit.unit))
          = ∑ jS : Fin ((s.pids 2 + 1) * BLK), ctxNopadIOFnum s Q K V sm_scale rs hs DM m0 m1
              (s.pids 2 * BLK + (Lane2D.decode j).1.val) (s.pids 1) (j.val % DM) jS.val from by
        refine Finset.sum_congr rfl (fun jS _ => ?_)
        rw [ctxNopadIOFnum, ctxNopadIOFden]
        simp only [ctxQTileG, ctxKTileMG, ctxKTileG, ctxVTileMG, ctxVTileG, hm1, hdv]]
      exact ctxNopadIO_sum_extend _ _ _ hSle hgiS _ hnum0
    have hden : (∑ jS : Fin ((s.pids 2 + 1) * BLK),
          if jS.val ≤ s.pids 2 * BLK + (Lane2D.decode j).1.val then
            Real.exp (sm_scale * ∑ e : Fin DM,
              ctxQTileG s Q B_Start_Loc.cast rs hs BLK DM ((Lane2D.decode j).1, e, PUnit.unit)
                * ctxKTileMG s K B_Start_Loc.cast rs hs ((s.pids 2 + 1) * BLK) DM m0
                    (jS, e, PUnit.unit))
          else 0)
        = ∑ jg : Fin (NT * BLK), ctxNopadIOFden s Q K sm_scale rs hs DM m0 m1
            (s.pids 2 * BLK + (Lane2D.decode j).1.val) (s.pids 1) jg.val := by
      rw [show (∑ jS : Fin ((s.pids 2 + 1) * BLK),
            if jS.val ≤ s.pids 2 * BLK + (Lane2D.decode j).1.val then
              Real.exp (sm_scale * ∑ e : Fin DM,
                ctxQTileG s Q B_Start_Loc.cast rs hs BLK DM ((Lane2D.decode j).1, e, PUnit.unit)
                  * ctxKTileMG s K B_Start_Loc.cast rs hs ((s.pids 2 + 1) * BLK) DM m0
                      (jS, e, PUnit.unit))
            else 0)
          = ∑ jS : Fin ((s.pids 2 + 1) * BLK), ctxNopadIOFden s Q K sm_scale rs hs DM m0 m1
              (s.pids 2 * BLK + (Lane2D.decode j).1.val) (s.pids 1) jS.val from by
        refine Finset.sum_congr rfl (fun jS _ => ?_)
        rw [ctxNopadIOFden]
        simp only [ctxQTileG, ctxKTileMG, ctxKTileG, hm1]]
      exact ctxNopadIO_sum_extend _ _ _ hSle hgiS _ hden0
    rw [← hnum, ← hden]
  · -- shared integrand = the streamed closed form
    rw [contextAttnNopadIOSpec]
    beta_reduce
    congr 1
    · refine Finset.sum_congr rfl (fun jg _ => ?_)
      rw [ctxNopadIOFnum, ctxNopadIOFden]
      by_cases hc : jg.val ≤ s.pids 2 * BLK + (Lane2D.decode j).1.val
      · rw [if_pos hc, if_pos hc]
        refine congrArg₂ (· * ·) (congrArg Real.exp (congrArg (sm_scale * ·)
          (Finset.sum_congr rfl (fun e _ => ?_)))) ?_
        · refine congrArg₂ (· * ·) ?_ ?_
          · -- Q cell = the static stream cell
            rw [ctxNopadIOqT,
              ← hx ⟨0, hNT⟩ (Lane2D.encode ((Lane2D.decode j).1, e, PUnit.unit))
                (by rw [Lane2D.encode_div]; exact hact')]
            rw [Lane2D.encode_div, Lane2D.encode_mod]
          · -- K cell = the streamed K cell
            rw [ctxNopadIOkT]
            by_cases hb : jg.val < m0
            · rw [if_pos hb, if_pos hb,
                ← hy ⟨jg.val / BLK, (Nat.div_lt_iff_lt_mul hBLK).mpr jg.isLt⟩
                  (Lane2D.encode (e, ⟨jg.val % BLK, Nat.mod_lt _ hBLK⟩, PUnit.unit))
                  (by rw [Lane2D.encode_mod]
                      show jg.val / BLK * BLK + jg.val % BLK < m0
                      rw [Nat.div_add_mod']; exact hb)]
              rw [Lane2D.encode_div, Lane2D.encode_mod]
              show s.readMem K ((m1 + jg.val) * rs + s.pids 1 * hs + e.val)
                = s.readMem K ((m1 + (jg.val / BLK * BLK + jg.val % BLK)) * rs
                    + s.pids 1 * hs + e.val)
              rw [Nat.div_add_mod']
            · rw [if_neg hb, if_neg hb]
        · -- V cell = the streamed V cell
          rw [ctxNopadIOvT]
          by_cases hb : jg.val < m0
          · rw [if_pos hb, if_pos hb,
              ← hz ⟨jg.val / BLK, (Nat.div_lt_iff_lt_mul hBLK).mpr jg.isLt⟩
                (Lane2D.encode (⟨jg.val % BLK, Nat.mod_lt _ hBLK⟩, (Lane2D.decode j).2.1,
                  PUnit.unit))
                (by rw [Lane2D.encode_div]
                    show jg.val / BLK * BLK + jg.val % BLK < m0
                    rw [Nat.div_add_mod']; exact hb)]
            rw [Lane2D.encode_div, Lane2D.encode_mod, hdv]
            show s.readMem V ((m1 + jg.val) * rs + s.pids 1 * hs + j.val % DM)
              = s.readMem V ((m1 + (jg.val / BLK * BLK + jg.val % BLK)) * rs
                  + s.pids 1 * hs + j.val % DM)
            rw [Nat.div_add_mod']
          · rw [if_neg hb, if_neg hb]
      · rw [if_neg hc, if_neg hc, zero_mul, zero_mul]
    · refine Finset.sum_congr rfl (fun jg _ => ?_)
      rw [ctxNopadIOFden]
      by_cases hc : jg.val ≤ s.pids 2 * BLK + (Lane2D.decode j).1.val
      · rw [if_pos hc, if_pos hc]
        refine congrArg Real.exp (congrArg (sm_scale * ·)
          (Finset.sum_congr rfl (fun e _ => ?_)))
        refine congrArg₂ (· * ·) ?_ ?_
        · rw [ctxNopadIOqT,
            ← hx ⟨0, hNT⟩ (Lane2D.encode ((Lane2D.decode j).1, e, PUnit.unit))
              (by rw [Lane2D.encode_div]; exact hact')]
          rw [Lane2D.encode_div, Lane2D.encode_mod]
        · rw [ctxNopadIOkT]
          by_cases hb : jg.val < m0
          · rw [if_pos hb, if_pos hb,
              ← hy ⟨jg.val / BLK, (Nat.div_lt_iff_lt_mul hBLK).mpr jg.isLt⟩
                (Lane2D.encode (e, ⟨jg.val % BLK, Nat.mod_lt _ hBLK⟩, PUnit.unit))
                (by rw [Lane2D.encode_mod]
                    show jg.val / BLK * BLK + jg.val % BLK < m0
                    rw [Nat.div_add_mod']; exact hb)]
            rw [Lane2D.encode_div, Lane2D.encode_mod]
            show s.readMem K ((m1 + jg.val) * rs + s.pids 1 * hs + e.val)
              = s.readMem K ((m1 + (jg.val / BLK * BLK + jg.val % BLK)) * rs
                  + s.pids 1 * hs + e.val)
            rw [Nat.div_add_mod']
          · rw [if_neg hb, if_neg hb]
      · rw [if_neg hc, if_neg hc]

/-! ## Flat-bridge coverage and cast-freedom

The LightLLM port is cast-free post-erasure (`p.to(v.dtype)` lowered to a
self-assign; all loads/stores at `.real`/`.nat`), so `stepStmtsR R` collapses
verbatim onto the exact stepper on every segment. -/

set_option maxHeartbeats 4000000 in
/-- The surface sits inside the flat-memory bridge's covered fragment (plain
pointers throughout — no `ptrSub`, no block pointers). -/
theorem ctxNopad_flattenOk (Q K V : RegionName) (B_Start_Loc B_Seqlen : Region .nat)
    (Out : RegionName) (sm_scale : ℝ) (rs hs BLK DM : Nat) :
    ((context_attn_nopad_fwd_kernel_surface Q K V sm_scale B_Start_Loc B_Seqlen Out
      rs hs 1 rs hs 1 rs hs 1 rs hs 1 BLK DM BLK).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  rw [nopad_body_splitG]
  simp [nopadPreLoopG, nopadLoopBodyG, nopadPostLoopG, StmtList.FlattenOk,
    Stmt.FlattenOk, Op.FlattenOk]
  simp [Op.FlattenOk.eq_def]

/-- Per-statement cast-free collapse lifts to statement lists (walks the
actual successor chain; a failing step collapses on both sides). Private
copy of the aft3 helper (bench files never import each other). -/
private theorem ctxNopadIO_stepStmtsR_castFree_of_stmts (R : RoundingModel) :
    ∀ (l : List Stmt), (∀ st ∈ l, ∀ u, stepStmtR R st u = stepStmt st u) →
      ∀ s, stepStmtsR R l s = stepStmts l s
  | [], _, s => by simp only [stepStmtsR, stepStmts]
  | st :: rest, h, s => by
      simp only [stepStmtsR, stepStmts, h st List.mem_cons_self s]
      cases stepStmt st s with
      | none => rfl
      | some s' =>
          exact ctxNopadIO_stepStmtsR_castFree_of_stmts R rest
            (fun st' h' u => h st' (List.mem_cons_of_mem _ h') u) s'

set_option maxHeartbeats 4000000 in
/-- Every preLoop statement is cast-free. -/
private theorem ctxNopadIO_preLoop_stmt_castFree (R : RoundingModel)
    (Q K V : RegionName) (B_Start_Loc B_Seqlen : Region .nat) (rs hs BLK DM : Nat) :
    ∀ st ∈ nopadPreLoopG Q K V B_Start_Loc B_Seqlen rs hs BLK DM,
      ∀ u, stepStmtR R st u = stepStmt st u := by
  intro st hst u
  simp only [nopadPreLoopG, List.mem_cons, List.not_mem_nil, or_false] at hst
  rcases hst with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl |
    rfl | rfl | rfl | rfl | rfl | rfl
  all_goals simp only [stepStmtR, stepStmt, evalOpR.eq_def, evalOp.eq_def]

/-- The preLoop is cast-free as a list. -/
private theorem ctxNopadIO_preLoop_castFree (R : RoundingModel)
    (Q K V : RegionName) (B_Start_Loc B_Seqlen : Region .nat) (rs hs BLK DM : Nat)
    (t : BlockState) :
    stepStmtsR R (nopadPreLoopG Q K V B_Start_Loc B_Seqlen rs hs BLK DM) t
      = stepStmts (nopadPreLoopG Q K V B_Start_Loc B_Seqlen rs hs BLK DM) t :=
  ctxNopadIO_stepStmtsR_castFree_of_stmts R _
    (ctxNopadIO_preLoop_stmt_castFree R Q K V B_Start_Loc B_Seqlen rs hs BLK DM) t

set_option maxHeartbeats 4000000 in
/-- Every loop-body statement is cast-free (the erased `p.to(v.dtype)` is a
self-assign; both masked loads carry `other=0.0` at `.real`). -/
private theorem ctxNopadIO_body_stmt_castFree (R : RoundingModel)
    (sc : ℝ) (rs hs BLK DM : Nat) :
    ∀ st ∈ nopadLoopBodyG sc rs hs BLK DM,
      ∀ u, stepStmtR R st u = stepStmt st u := by
  intro st hst u
  simp only [nopadLoopBodyG, List.mem_cons, List.not_mem_nil, or_false] at hst
  rcases hst with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl |
    rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
  all_goals simp only [stepStmtR, stepStmt, evalOpR.eq_def, evalOp.eq_def]

/-- The loop body is cast-free as a list. -/
private theorem ctxNopadIO_body_castFree (R : RoundingModel)
    (sc : ℝ) (rs hs BLK DM : Nat) (t : BlockState) :
    stepStmtsR R (nopadLoopBodyG sc rs hs BLK DM) t
      = stepStmts (nopadLoopBodyG sc rs hs BLK DM) t :=
  ctxNopadIO_stepStmtsR_castFree_of_stmts R _
    (ctxNopadIO_body_stmt_castFree R sc rs hs BLK DM) t

set_option maxHeartbeats 4000000 in
/-- Every postLoop statement is cast-free (`writeMemTypedR R .real` *is*
`writeMemTyped .real`). -/
private theorem ctxNopadIO_postLoop_stmt_castFree (R : RoundingModel)
    (Out : RegionName) (rs hs BLK DM : Nat) :
    ∀ st ∈ nopadPostLoopG Out rs hs BLK DM,
      ∀ u, stepStmtR R st u = stepStmt st u := by
  intro st hst u
  simp only [nopadPostLoopG, List.mem_cons, List.not_mem_nil, or_false] at hst
  rcases hst with rfl | rfl | rfl
  all_goals
    simp only [stepStmtR, stepStmt, evalOpR.eq_def, evalOp.eq_def,
      BlockState.writeMemTypedR]

/-- The postLoop is cast-free as a list. -/
private theorem ctxNopadIO_postLoop_castFree (R : RoundingModel)
    (Out : RegionName) (rs hs BLK DM : Nat) (t : BlockState) :
    stepStmtsR R (nopadPostLoopG Out rs hs BLK DM) t
      = stepStmts (nopadPostLoopG Out rs hs BLK DM) t :=
  ctxNopadIO_stepStmtsR_castFree_of_stmts R _
    (ctxNopadIO_postLoop_stmt_castFree R Out rs hs BLK DM) t

/-- `evalOpR` of a `constNat` (R-independent). -/
private theorem ctxNopadIO_evalOpR_constNat (R : RoundingModel) (n : Nat) (u : BlockState) :
    evalOpR R (Op.constNat n) u = some (Tile.scalar n) := by
  simp [evalOpR]

/-- `evalOpR` of the `forRangeDyn` stop expression
`block_mask * (start_m + 1) * BLK` is the exact evaluation (register-only
`.nat` arithmetic — R-independent). -/
private theorem ctxNopadIO_stopOpR_castFree (R : RoundingModel) (BLK : Nat) (u : BlockState) :
    evalOpR R (Op.mul .nat Broadcast.nil
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "block_mask")
          (Op.add .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat 1)))
        (Op.constNat BLK)) u
      = evalOp (Op.mul .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "block_mask")
            (Op.add .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat 1)))
          (Op.constNat BLK)) u := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

set_option maxHeartbeats 1600000 in
/-- The streaming `forRangeDyn` statement is cast-free per-state: its bound
expressions are register-only `.nat` arithmetic and its body is cast-free,
so `stepStmtR R` on the whole loop *is* `stepStmt`. -/
private theorem ctxNopadIO_dyn_castFree (R : RoundingModel) (sc : ℝ) (rs hs BLK DM : Nat) :
    ∀ u, stepStmtR R (Stmt.forRangeDyn "start_n" (Op.constNat 0)
        (Op.mul .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "block_mask")
            (Op.add .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat 1)))
          (Op.constNat BLK))
        (Op.constNat BLK) (nopadLoopBodyG sc rs hs BLK DM)) u
      = stepStmt (Stmt.forRangeDyn "start_n" (Op.constNat 0)
          (Op.mul .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "block_mask")
              (Op.add .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat 1)))
            (Op.constNat BLK))
          (Op.constNat BLK) (nopadLoopBodyG sc rs hs BLK DM)) u := by
  intro u
  rw [stepForRangeAux.forRangeDyn_unfold]
  simp only [stepStmtR, ctxNopadIO_evalOpR_constNat, ctxNopadIO_stopOpR_castFree,
    evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
  cases hstop : evalOp (Op.mul .nat Broadcast.nil
      (Op.mul .nat Broadcast.nil (Op.ref .nat [] "block_mask")
        (Op.add .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat 1)))
      (Op.constNat BLK)) u with
  | none => rfl
  | some t =>
      simp only [Option.bind_some]
      exact stepForRangeAuxR_castFree R _ (ctxNopadIO_body_castFree R sc rs hs BLK DM)
        "start_n" _ _ _ u

/-- The whole lowered body is cast-free, statement by statement: `execR R`
on the surface *is* the exact `stepStmts` run. -/
private theorem ctxNopadIO_execR_collapse (R : RoundingModel)
    (Q K V : RegionName) (B_Start_Loc B_Seqlen : Region .nat) (Out : RegionName)
    (sm_scale : ℝ) (rs hs BLK DM : Nat) (s : BlockState) :
    execR R ((context_attn_nopad_fwd_kernel_surface Q K V sm_scale B_Start_Loc B_Seqlen Out
        rs hs 1 rs hs 1 rs hs 1 rs hs 1 BLK DM BLK).toAlgKernel) s
      = stepStmts ((context_attn_nopad_fwd_kernel_surface Q K V sm_scale B_Start_Loc B_Seqlen Out
          rs hs 1 rs hs 1 rs hs 1 rs hs 1 BLK DM BLK).toAlgKernel.body) s := by
  unfold execR
  rw [nopad_body_splitG]
  refine ctxNopadIO_stepStmtsR_castFree_of_stmts R _ ?_ s
  intro st hst
  rcases List.mem_append.mp hst with hpre | hrest
  · exact ctxNopadIO_preLoop_stmt_castFree R Q K V B_Start_Loc B_Seqlen rs hs BLK DM st hpre
  · rcases List.mem_cons.mp hrest with rfl | hpost
    · exact ctxNopadIO_dyn_castFree R sm_scale rs hs BLK DM
    · exact ctxNopadIO_postLoop_stmt_castFree R Out rs hs BLK DM st hpost

/-! ## The weak safety stack (`hts`)

The skin's `hts` obligation quantifies over **arbitrary** launch states (no
clean-`undef` pin), so the exact stack's `nopadInvariantG` is unavailable
there. Since every load in this kernel is `other=0.0`-masked or unmasked
(never `other=None`), the whole walk is `undef`-independent: the weak stack
re-runs the register chain with exact pins for the address-bearing registers
and **existential** pins for the value registers (`q`/`m_i`/`l_i`/`acc`) —
the aft3 `aft3SafeInv` pattern. The `pre` hypothesis (`pid₂ < NT`) is what
makes the per-step `Fin NT` window bounds citable at all. -/

/-- Weak (safety-walk) invariant: address-register pins + value-register
existence, anchored to the launch state `s`. Counter-free: `k_ptrs`/`v_ptrs`
never advance in this kernel (the per-step shift is added inside the loads). -/
private def nopadSafeInvW (Q K V : RegionName) (B_Start_Loc B_Seqlen : RegionName)
    (rs hs BLK DM : Nat) (s s' : BlockState) : Prop :=
  s'.mem = s.mem
  ∧ s'.pids = s.pids
  ∧ s'.regs .nat [] "cur_batch_seq_len" = some (Tile.scalar (seqLen s B_Seqlen))
  ∧ s'.regs .nat [] "cur_batch_in_all_start_index"
      = some (Tile.scalar (startLoc s B_Start_Loc))
  ∧ s'.regs .nat [] "start_m" = some (Tile.scalar (s.pids 2))
  ∧ s'.regs .nat [] "cur_head" = some (Tile.scalar (s.pids 1))
  ∧ s'.regs .nat [] "block_mask"
      = some (Tile.scalar (if BLK * s.pids 2 < seqLen s B_Seqlen then 1 else 0))
  ∧ s'.regs .nat [BLK] "offs_n" = some (Tile.vec (fun jn : Fin BLK => jn.val))
  ∧ s'.regs .nat [DM] "offs_d" = some (Tile.vec (fun e : Fin DM => e.val))
  ∧ s'.regs .nat [BLK] "offs_m" = some (Tile.vec (fun i : Fin BLK => s.pids 2 * BLK + i.val))
  ∧ s'.regs .ptr [DM, BLK] "k_ptrs" = some (⟨fun idx : TileIndex [DM, BLK] =>
      (K, idx.2.1.val * rs + s.pids 1 * hs + idx.1.val)⟩ : Tile .ptr [DM, BLK])
  ∧ s'.regs .ptr [BLK, DM] "v_ptrs" = some (⟨fun idx : TileIndex [BLK, DM] =>
      (V, idx.1.val * rs + s.pids 1 * hs + idx.2.1.val)⟩ : Tile .ptr [BLK, DM])
  ∧ (∃ qT : Tile .real [BLK, DM], s'.regs .real [BLK, DM] "q" = some qT)
  ∧ (∃ mT : Tile .real [BLK], s'.regs .real [BLK] "m_i" = some mT)
  ∧ (∃ lT : Tile .real [BLK], s'.regs .real [BLK] "l_i" = some lT)
  ∧ (∃ aT : Tile .real [BLK, DM], s'.regs .real [BLK, DM] "acc" = some aT)

/-- Combined walk cons: safety of the head at the current state, the
R-step it actually takes, and the pair (safety, run) of the tail from the
successor give the pair for the whole list. The single-pass combinator the
weak preLoop/body walks chain through. -/
private theorem nopadIO_walkCons {R : RoundingModel} {bounds : RegionBounds}
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

set_option maxHeartbeats 1600000 in
/-- `evalOpR` of the K-load pointer tree at pinned registers. -/
private theorem ctxNopadIO_kptrR_eval (R : RoundingModel) (u : BlockState)
    (K : RegionName) (m1v SN P1 rs hs BLK DM : Nat)
    (hkp : u.regs .ptr [DM, BLK] "k_ptrs" = some (⟨fun idx : TileIndex [DM, BLK] =>
      (K, idx.2.1.val * rs + P1 * hs + idx.1.val)⟩ : Tile .ptr [DM, BLK]))
    (hsl : u.regs .nat [] "cur_batch_in_all_start_index" = some (Tile.scalar m1v))
    (hsn : u.regs .nat [] "start_n" = some (Tile.scalar SN)) :
    evalOpR R (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [DM, BLK] "k_ptrs")
        (Op.mul .nat Broadcast.nil
          (Op.add .nat Broadcast.nil (Op.ref .nat [] "cur_batch_in_all_start_index")
            (Op.ref .nat [] "start_n"))
          (Op.constNat rs))) u
      = some (⟨fun idx : TileIndex [DM, BLK] =>
          (K, idx.2.1.val * rs + P1 * hs + idx.1.val + (m1v + SN) * rs)⟩
          : Tile .ptr [DM, BLK]) := by
  rw [show evalOpR R (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [DM, BLK] "k_ptrs")
      (Op.mul .nat Broadcast.nil
        (Op.add .nat Broadcast.nil (Op.ref .nat [] "cur_batch_in_all_start_index")
          (Op.ref .nat [] "start_n"))
        (Op.constNat rs))) u
      = evalOp (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [DM, BLK] "k_ptrs")
        (Op.mul .nat Broadcast.nil
          (Op.add .nat Broadcast.nil (Op.ref .nat [] "cur_batch_in_all_start_index")
            (Op.ref .nat [] "start_n"))
          (Op.constNat rs))) u from by
    simp only [evalOpR.eq_def, evalOp.eq_def]]
  simp only [evalOp, hkp, hsl, hsn, Option.bind]
  refine congrArg some (Tile.ext fun idx => ?_)
  obtain ⟨e, jL, pu⟩ := idx
  simp only [Tile.ptrAdd_data, Tile.bop_data, Tile.bop, Tile.scalar, NumericDType.add,
    NumericDType.mul, Broadcast.leftIndex, Broadcast.rightIndex, Prod.mk.injEq, true_and]

set_option maxHeartbeats 1600000 in
/-- `evalOpR` of the V-load pointer tree at pinned registers. -/
private theorem ctxNopadIO_vptrR_eval (R : RoundingModel) (u : BlockState)
    (V : RegionName) (m1v SN P1 rs hs BLK DM : Nat)
    (hvp : u.regs .ptr [BLK, DM] "v_ptrs" = some (⟨fun idx : TileIndex [BLK, DM] =>
      (V, idx.1.val * rs + P1 * hs + idx.2.1.val)⟩ : Tile .ptr [BLK, DM]))
    (hsl : u.regs .nat [] "cur_batch_in_all_start_index" = some (Tile.scalar m1v))
    (hsn : u.regs .nat [] "start_n" = some (Tile.scalar SN)) :
    evalOpR R (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BLK, DM] "v_ptrs")
        (Op.mul .nat Broadcast.nil
          (Op.add .nat Broadcast.nil (Op.ref .nat [] "cur_batch_in_all_start_index")
            (Op.ref .nat [] "start_n"))
          (Op.constNat rs))) u
      = some (⟨fun idx : TileIndex [BLK, DM] =>
          (V, idx.1.val * rs + P1 * hs + idx.2.1.val + (m1v + SN) * rs)⟩
          : Tile .ptr [BLK, DM]) := by
  rw [show evalOpR R (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BLK, DM] "v_ptrs")
      (Op.mul .nat Broadcast.nil
        (Op.add .nat Broadcast.nil (Op.ref .nat [] "cur_batch_in_all_start_index")
          (Op.ref .nat [] "start_n"))
        (Op.constNat rs))) u
      = evalOp (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BLK, DM] "v_ptrs")
        (Op.mul .nat Broadcast.nil
          (Op.add .nat Broadcast.nil (Op.ref .nat [] "cur_batch_in_all_start_index")
            (Op.ref .nat [] "start_n"))
          (Op.constNat rs))) u from by
    simp only [evalOpR.eq_def, evalOp.eq_def]]
  simp only [evalOp, hvp, hsl, hsn, Option.bind]
  refine congrArg some (Tile.ext fun idx => ?_)
  obtain ⟨jL, e, pu⟩ := idx
  simp only [Tile.ptrAdd_data, Tile.bop_data, Tile.bop, Tile.scalar, NumericDType.add,
    NumericDType.mul, Broadcast.leftIndex, Broadcast.rightIndex, Prod.mk.injEq, true_and]

set_option maxHeartbeats 1600000 in
/-- `evalOpR` of the K-load column mask `(start_n + offs_n[None,:]) < seq_len`
at pinned registers. -/
private theorem ctxNopadIO_kmaskR_eval (R : RoundingModel) (u : BlockState)
    (slv SN BLK DM : Nat)
    (hsn : u.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hn : u.regs .nat [BLK] "offs_n" = some (Tile.vec (fun jn : Fin BLK => jn.val)))
    (hseq : u.regs .nat [] "cur_batch_seq_len" = some (Tile.scalar slv)) :
    evalOpR R (Op.remap [DM, BLK] Broadcast.nil.consSame.consL.leftIndex
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLK] "offs_n")))
          (Op.ref .nat [] "cur_batch_seq_len"))) u
      = some (⟨fun idx : TileIndex [DM, BLK] => decide (SN + idx.2.1.val < slv)⟩
          : Tile .bool [DM, BLK]) := by
  rw [show evalOpR R (Op.remap [DM, BLK] Broadcast.nil.consSame.consL.leftIndex
      (Op.lt ComparableDType.nat Broadcast.scalarR
        (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLK] "offs_n")))
        (Op.ref .nat [] "cur_batch_seq_len"))) u
      = evalOp (Op.remap [DM, BLK] Broadcast.nil.consSame.consL.leftIndex
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLK] "offs_n")))
          (Op.ref .nat [] "cur_batch_seq_len"))) u from by
    simp only [evalOpR.eq_def, evalOp.eq_def]]
  have hexp : @evalOp .nat [1, BLK] (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLK] "offs_n")) u
      = some (Tile.expandDim ⟨0, by simp⟩ (Tile.vec (fun jn : Fin BLK => jn.val))) :=
    evalOp_expandDim_ref_of_regs .nat [BLK] ⟨0, by simp⟩ "offs_n" u _ hn
  simp only [evalOp, hsn, hexp, hseq, Option.bind]
  refine congrArg some (Tile.ext fun idx => ?_)
  obtain ⟨e, jL, pu⟩ := idx
  simp only [Tile.remap, Tile.cop_data, Tile.bop_data, Tile.bop, Tile.expandDim, Tile.vec,
    Tile.scalar, ComparableDType.lt, NumericDType.add, Broadcast.leftIndex,
    Broadcast.rightIndex, TileShape.dropInsertedIndex]
  rfl

set_option maxHeartbeats 1600000 in
/-- `evalOpR` of the V-load row mask `(start_n + offs_n[:,None]) < seq_len`
at pinned registers. -/
private theorem ctxNopadIO_vmaskR_eval (R : RoundingModel) (u : BlockState)
    (slv SN BLK DM : Nat)
    (hsn : u.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hn : u.regs .nat [BLK] "offs_n" = some (Tile.vec (fun jn : Fin BLK => jn.val)))
    (hseq : u.regs .nat [] "cur_batch_seq_len" = some (Tile.scalar slv)) :
    evalOpR R (Op.remap [BLK, DM] Broadcast.nil.consL.consSame.leftIndex
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLK] "offs_n")))
          (Op.ref .nat [] "cur_batch_seq_len"))) u
      = some (⟨fun idx : TileIndex [BLK, DM] => decide (SN + idx.1.val < slv)⟩
          : Tile .bool [BLK, DM]) := by
  rw [show evalOpR R (Op.remap [BLK, DM] Broadcast.nil.consL.consSame.leftIndex
      (Op.lt ComparableDType.nat Broadcast.scalarR
        (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLK] "offs_n")))
        (Op.ref .nat [] "cur_batch_seq_len"))) u
      = evalOp (Op.remap [BLK, DM] Broadcast.nil.consL.consSame.leftIndex
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLK] "offs_n")))
          (Op.ref .nat [] "cur_batch_seq_len"))) u from by
    simp only [evalOpR.eq_def, evalOp.eq_def]]
  have hexp : @evalOp .nat [BLK, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLK] "offs_n")) u
      = some (Tile.expandDim ⟨1, by simp⟩ (Tile.vec (fun jn : Fin BLK => jn.val))) :=
    evalOp_expandDim_ref_of_regs .nat [BLK] ⟨1, by simp⟩ "offs_n" u _ hn
  simp only [evalOp, hsn, hexp, hseq, Option.bind]
  refine congrArg some (Tile.ext fun idx => ?_)
  obtain ⟨jL, e, pu⟩ := idx
  simp only [Tile.remap, Tile.cop_data, Tile.bop_data, Tile.bop, Tile.expandDim, Tile.vec,
    Tile.scalar, ComparableDType.lt, NumericDType.add, Broadcast.leftIndex,
    Broadcast.rightIndex, TileShape.dropInsertedIndex]
  rfl

set_option maxHeartbeats 1600000 in
/-- `evalOpR` of the shared row mask `offs_m[:,None] < seq_len` (the `q`
load's and the terminal store's mask) at pinned registers. -/
private theorem ctxNopadIO_rowmaskR_eval (R : RoundingModel) (u : BlockState)
    (slv P2v BLK DM : Nat)
    (hm : u.regs .nat [BLK] "offs_m" = some (Tile.vec (fun i : Fin BLK => P2v * BLK + i.val)))
    (hseq : u.regs .nat [] "cur_batch_seq_len" = some (Tile.scalar slv)) :
    evalOpR R (Op.remap [BLK, DM] Broadcast.nil.consL.consSame.leftIndex
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLK] "offs_m"))
          (Op.ref .nat [] "cur_batch_seq_len"))) u
      = some (⟨fun idx : TileIndex [BLK, DM] => decide (P2v * BLK + idx.1.val < slv)⟩
          : Tile .bool [BLK, DM]) := by
  rw [show evalOpR R (Op.remap [BLK, DM] Broadcast.nil.consL.consSame.leftIndex
      (Op.lt ComparableDType.nat Broadcast.scalarR
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLK] "offs_m"))
        (Op.ref .nat [] "cur_batch_seq_len"))) u
      = evalOp (Op.remap [BLK, DM] Broadcast.nil.consL.consSame.leftIndex
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLK] "offs_m"))
          (Op.ref .nat [] "cur_batch_seq_len"))) u from by
    simp only [evalOpR.eq_def, evalOp.eq_def]]
  have hexp : @evalOp .nat [BLK, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLK] "offs_m")) u
      = some (Tile.expandDim ⟨1, by simp⟩ (Tile.vec (fun i : Fin BLK => P2v * BLK + i.val))) :=
    evalOp_expandDim_ref_of_regs .nat [BLK] ⟨1, by simp⟩ "offs_m" u _ hm
  simp only [evalOp, hexp, hseq, Option.bind]
  refine congrArg some (Tile.ext fun idx => ?_)
  obtain ⟨i, e, pu⟩ := idx
  simp only [Tile.remap, Tile.cop_data, Tile.bop_data, Tile.bop, Tile.expandDim, Tile.vec,
    Tile.scalar, ComparableDType.lt, Broadcast.leftIndex, Broadcast.rightIndex,
    TileShape.dropInsertedIndex]
  rfl

/-- R-step of an assign whose op is cast-free: the two collapse into one
walk-ready equation. -/
private theorem nopadIO_stepR_of_assign {R : RoundingModel} {dt : TileDType}
    {sh : TileShape} {nm : RegName} {e : Op dt sh} {s : BlockState} {v : Tile dt sh}
    (hcf : evalOpR R e s = evalOp e s) (h : evalOp e s = some v) :
    stepStmtR R (.assign dt sh nm e) s = some (s.setReg nm dt sh v) :=
  stepStmtR_assign_eq_some (hcf.trans h)

set_option maxHeartbeats 8000000 in
set_option maxRecDepth 8000 in
/-- **Weak preLoop walk** (single pass): from an **arbitrary** launch state
the 19 preLoop statements are trace-safe (the two `.nat` metadata loads
bounded by the slot windows, the masked `q` load by the `read1` window) and
step to a state satisfying `nopadSafeInvW`. -/
private theorem nopadIO_preLoopW (R : RoundingModel) (bounds : RegionBounds)
    (Q K V : RegionName) (B_Start_Loc B_Seqlen : Region .nat)
    (rs hs BLK DM : Nat) (s : BlockState)
    (hbSeq : s.pids 0 < bounds (Region.cast B_Seqlen))
    (hbLoc : s.pids 0 < bounds (Region.cast B_Start_Loc))
    (hbQ : ∀ (i : Fin BLK) (e : Fin DM), s.pids 2 * BLK + i.val < seqLen s ↑B_Seqlen →
      (startLoc s ↑B_Start_Loc + (s.pids 2 * BLK + i.val)) * rs + s.pids 1 * hs + e.val
        < bounds Q) :
    Stmt.TraceSafeListR R bounds (nopadPreLoopG Q K V B_Start_Loc B_Seqlen rs hs BLK DM) s
      ∧ ∃ s0, stepStmtsR R (nopadPreLoopG Q K V B_Start_Loc B_Seqlen rs hs BLK DM) s = some s0
          ∧ nopadSafeInvW Q K V ↑B_Start_Loc ↑B_Seqlen rs hs BLK DM s s0 := by
  unfold nopadPreLoopG
  refine nopadIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (nopadIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp _ _ = some (Tile.scalar (s.pids 0)) from evalOp_programId 0 s)) ?_
  refine nopadIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (nopadIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp _ _ = some (Tile.scalar (s.pids 1)) from evalOp_programId 1 _)) ?_
  refine nopadIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (nopadIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp _ _ = some (Tile.scalar (s.pids 2)) from evalOp_programId 2 _)) ?_
  -- the `cur_batch_seq_len` slot load
  refine nopadIO_walkCons ?_
    (nopadIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp _ _ = some (Tile.scalar (seqLen s B_Seqlen)) from by
        simp only [evalOp_load_region_none, evalOp_ref, BlockState.setReg_same,
          BlockState.setReg_ne_name, Option.bind, Option.pure_def, seqLen,
          BlockState.readMemValue]
        rfl)) ?_
  · simp only [Stmt.TraceSafeR, Op.SafeAtR, MaskOpt.ActiveR, MemAccess.ActiveAddressSafeR,
      memAccessActiveAddressSafeR]
    refine ⟨by simp [Op.SafeAtR.eq_def], by simp [MaskOpt.SafeAtR], ?_⟩
    intro offsets hoff idx _
    rw [evalOpR_ref] at hoff
    simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same] at hoff
    obtain rfl := Option.some.inj hoff
    exact hbSeq
  -- the `cur_batch_in_all_start_index` slot load
  refine nopadIO_walkCons ?_
    (nopadIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp _ _ = some (Tile.scalar (startLoc s B_Start_Loc)) from by
        simp only [evalOp_load_region_none, evalOp_ref, BlockState.setReg_same,
          BlockState.setReg_ne_name, Option.bind, Option.pure_def, startLoc,
          BlockState.readMemValue]
        rfl)) ?_
  · simp only [Stmt.TraceSafeR, Op.SafeAtR, MaskOpt.ActiveR, MemAccess.ActiveAddressSafeR,
      memAccessActiveAddressSafeR]
    refine ⟨by simp [Op.SafeAtR.eq_def], by simp [MaskOpt.SafeAtR], ?_⟩
    intro offsets hoff idx _
    rw [evalOpR_ref] at hoff
    simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same] at hoff
    obtain rfl := Option.some.inj hoff
    exact hbLoc
  refine nopadIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (nopadIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp _ _ = some (Tile.scalar (BLK * s.pids 2)) from by
        rw [evalOp_mul, evalOp_constNat, evalOp_ref]
        simp only [BlockState.setReg_same, BlockState.setReg_ne_name, BlockState.setReg_pids,
          ne_eq, String.reduceEq, not_false_eq_true, Option.bind_eq_bind, Option.bind_some]
        rfl)) ?_
  refine nopadIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (nopadIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp _ _ = some (Tile.vec (fun jn : Fin BLK => jn.val)) from
        evalOp_arange BLK _)) ?_
  refine nopadIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (nopadIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp _ _ = some (Tile.vec (fun e : Fin DM => e.val)) from
        evalOp_arange DM _)) ?_
  refine nopadIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (nopadIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp _ _
          = some (Tile.vec (fun i : Fin BLK => s.pids 2 * BLK + i.val)) from by
        rw [evalOp_add, evalOp_mul, evalOp_ref, evalOp_constNat, evalOp_arange]
        simp only [BlockState.setReg_same, BlockState.setReg_ne_name, BlockState.setReg_pids,
          ne_eq, String.reduceEq, not_false_eq_true, Option.bind_eq_bind, Option.bind_some]
        refine congrArg some ?_; ext idx
        simp only [Tile.bop_data, Tile.bop, Tile.scalar, Tile.vec, Broadcast.leftIndex,
          Broadcast.rightIndex, NumericDType.add, NumericDType.mul])) ?_
  refine nopadIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (nopadIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp _ _ = some (⟨fun idx : TileIndex [BLK, DM] =>
            (startLoc s B_Start_Loc + (s.pids 2 * BLK + idx.1.val)) * rs
              + s.pids 1 * hs + idx.2.1.val⟩ : Tile .nat [BLK, DM]) from
        nopad_offq_evalG _ (startLoc s B_Start_Loc) (s.pids 1) (s.pids 2) rs hs BLK DM
          (by simp) (by simp) (by simp) (by simp [Tile.vec]))) ?_
  refine nopadIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (nopadIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp _ _ = some (⟨fun idx : TileIndex [DM, BLK] =>
            idx.2.1.val * rs + s.pids 1 * hs + idx.1.val⟩ : Tile .nat [DM, BLK]) from
        nopad_offk_evalG _ (s.pids 1) rs hs BLK DM
          (by simp) (by simp [Tile.vec]) (by simp [Tile.vec]))) ?_
  refine nopadIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (nopadIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp _ _ = some (⟨fun idx : TileIndex [BLK, DM] =>
            idx.1.val * rs + s.pids 1 * hs + idx.2.1.val⟩ : Tile .nat [BLK, DM]) from
        nopad_offv_evalG _ (s.pids 1) rs hs BLK DM
          (by simp) (by simp [Tile.vec]) (by simp [Tile.vec]))) ?_
  -- the masked `q` load
  refine nopadIO_walkCons ?_
    (nopadIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (nopad_q_load_evalG _ Q B_Start_Loc B_Seqlen rs hs BLK DM
        (by simp [startLoc, BlockState.readMemValue, BlockState.readMemTyped])
        (by simp)
        (by simp [seqLen, BlockState.readMemValue, BlockState.readMemTyped]))) ?_
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
    rw [ctxNopadIO_rowmaskR_eval R _ (seqLen s ↑B_Seqlen) (s.pids 2) BLK DM
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
        BlockState.setReg_same])
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
        BlockState.setReg_same])] at hmask
    obtain rfl := Option.some.inj hmask
    obtain ⟨i, e, pu⟩ := idx
    have hrow : s.pids 2 * BLK + i.val < seqLen s ↑B_Seqlen := by simpa using hmi
    have hb := hbQ i e hrow
    simpa [Region.cast_id] using hb
  refine nopadIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (nopadIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp _ _ = some (⟨fun idx : TileIndex [DM, BLK] =>
            (K, idx.2.1.val * rs + s.pids 1 * hs + idx.1.val)⟩ : Tile .ptr [DM, BLK]) from by
        simp only [evalOp, evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name,
          BlockState.setReg_pids, ne_eq, String.reduceEq, not_false_eq_true, Option.bind]
        refine congrArg some (Tile.ext (fun idx => ?_))
        obtain ⟨e, jL, pu⟩ := idx
        simp only [Tile.ptrAdd_data, Tile.scalar_data, Broadcast.leftIndex,
          Broadcast.rightIndex, Region.cast_id, Nat.zero_add, Prod.mk.injEq, true_and])) ?_
  refine nopadIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (nopadIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp _ _ = some (⟨fun idx : TileIndex [BLK, DM] =>
            (V, idx.1.val * rs + s.pids 1 * hs + idx.2.1.val)⟩ : Tile .ptr [BLK, DM]) from by
        simp only [evalOp, evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name,
          BlockState.setReg_pids, ne_eq, String.reduceEq, not_false_eq_true, Option.bind]
        refine congrArg some (Tile.ext (fun idx => ?_))
        obtain ⟨jL, e, pu⟩ := idx
        simp only [Tile.ptrAdd_data, Tile.scalar_data, Broadcast.leftIndex,
          Broadcast.rightIndex, Region.cast_id, Nat.zero_add, Prod.mk.injEq, true_and])) ?_
  refine nopadIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (nopadIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp _ _ = some (⟨fun _ : TileIndex [BLK] => (⊥ : WithBot ℝ)⟩
          : Tile .real [BLK]) from by
        rw [evalOp_add, evalOp_full, evalOp_const, evalOp_negInf]
        simp only [Option.bind_eq_bind, Option.bind_some]
        refine congrArg some ?_; ext idx
        simp only [Tile.bop_data, Tile.bop, Tile.scalar, Broadcast.leftIndex,
          Broadcast.rightIndex, NumericDType.add]
        rfl)) ?_
  refine nopadIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (nopadIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp _ _ = some (⟨fun _ : TileIndex [BLK] => some (0 : ℝ)⟩
          : Tile .real [BLK]) from by
        simp [evalOp_full, evalOp_const])) ?_
  refine nopadIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (nopadIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp _ _ = some (⟨fun _ : TileIndex [BLK, DM] => some (0 : ℝ)⟩
          : Tile .real [BLK, DM]) from by
        simp [evalOp_full, evalOp_const])) ?_
  refine nopadIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (nopadIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp _ _
          = some (Tile.scalar (if BLK * s.pids 2 < seqLen s B_Seqlen then 1 else 0)) from by
        rw [evalOp_where, evalOp_lt, evalOp_constNat, evalOp_constNat]
        simp only [evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name,
          BlockState.setReg_pids, ne_eq, String.reduceEq, not_false_eq_true,
          Option.bind_eq_bind, Option.bind_some]
        refine congrArg some ?_; ext idx
        simp only [Tile.select_data, Tile.cop_data, Tile.bop, Tile.scalar,
          Broadcast.leftIndex, Broadcast.rightIndex, ComparableDType.lt]
        by_cases hlt : BLK * s.pids 2 < seqLen s B_Seqlen
        · rw [if_pos (by simpa [seqLen, BlockState.readMemValue] using hlt), if_pos hlt]
        · rw [if_neg (by simpa [seqLen, BlockState.readMemValue] using hlt), if_neg hlt])) ?_
  refine ⟨Stmt.TraceSafeListR.nil_intro, _, stepStmtsR_nil R _, ?_⟩
  -- the weak-invariant pins of the final state
  refine ⟨?_, ?_, by simp, by simp, by simp, by simp, by simp, by simp [Tile.vec],
    by simp [Tile.vec], by simp [Tile.vec], by simp, by simp,
    ⟨⟨fun idx : TileIndex [BLK, DM] =>
        some (ctxQTileMRowG s Q ↑B_Start_Loc rs hs BLK DM (seqLen s ↑B_Seqlen)
          (idx.1, idx.2.1, PUnit.unit))⟩, ?_⟩,
    ⟨⟨fun _ : TileIndex [BLK] => (⊥ : WithBot ℝ)⟩, by simp⟩,
    ⟨⟨fun _ : TileIndex [BLK] => some (0 : ℝ)⟩, by simp⟩,
    ⟨⟨fun _ : TileIndex [BLK, DM] => some (0 : ℝ)⟩, by simp⟩⟩
  · funext rg o; simp
  · simp
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same]
    refine congrArg some (Tile.ext (fun idx => ?_))
    simp only [ctxQTileMRowG, ctxQTileG, seqLen, startLoc, BlockState.readMem,
      BlockState.readMemValue, BlockState.readMemTyped, BlockState.setReg_mem,
      BlockState.setReg_pids]

set_option maxHeartbeats 1600000 in
/-- Generic-value K-load eval (the in-loop slot-shifted masked load), pinned
by plain scalar values — the `nopad_k_load_evalG` recipe freed from its
state-anchored pin spellings, for the weak walk. -/
private theorem ctxNopadIO_kload_eval (u : BlockState) (K : RegionName)
    (m1v slv SN P1 rs hs BLK DM : Nat)
    (hkp : u.regs .ptr [DM, BLK] "k_ptrs" = some (⟨fun idx : TileIndex [DM, BLK] =>
      (K, idx.2.1.val * rs + P1 * hs + idx.1.val)⟩ : Tile .ptr [DM, BLK]))
    (hsl : u.regs .nat [] "cur_batch_in_all_start_index" = some (Tile.scalar m1v))
    (hsn : u.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hn : u.regs .nat [BLK] "offs_n" = some (Tile.vec (fun jn : Fin BLK => jn.val)))
    (hseq : u.regs .nat [] "cur_batch_seq_len" = some (Tile.scalar slv)) :
    evalOp (Op.load .real
        (MemAccess.ptr
          (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [DM, BLK] "k_ptrs")
            (Op.mul .nat Broadcast.nil
              (Op.add .nat Broadcast.nil (Op.ref .nat [] "cur_batch_in_all_start_index")
                (Op.ref .nat [] "start_n"))
              (Op.constNat rs))))
        (MaskOpt.maskOther
          (Op.remap [DM, BLK] Broadcast.nil.consSame.consL.leftIndex
            (Op.lt ComparableDType.nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
                (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLK] "offs_n")))
              (Op.ref .nat [] "cur_batch_seq_len")))
          ((Op.const 0.0).broadcast [DM, BLK]))) u
      = some (⟨fun idx : TileIndex [DM, BLK] =>
          if SN + idx.2.1.val < slv then
            some (u.readMem K ((m1v + (SN + idx.2.1.val)) * rs + P1 * hs + idx.1.val))
          else some (0.0 : ℝ)⟩ : Tile .real [DM, BLK]) := by
  have hexp : @evalOp .nat [1, BLK] (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLK] "offs_n")) u
      = some (Tile.expandDim ⟨0, by simp⟩ (Tile.vec (fun jn : Fin BLK => jn.val))) :=
    evalOp_expandDim_ref_of_regs .nat [BLK] ⟨0, by simp⟩ "offs_n" u _ hn
  simp only [evalOp, hkp, hsl, hsn, hexp, hseq, Option.bind, Option.some.injEq]
  refine congrArg some ?_
  ext idx
  obtain ⟨e, jL, pu⟩ := idx
  simp only [Tile.ptrAdd_data, Tile.cop_data, Tile.bop_data, Tile.bop, Tile.remap,
    Tile.expandDim, Tile.vec, Tile.scalar, NumericDType.add, NumericDType.mul,
    ComparableDType.lt, Broadcast.leftIndex, Broadcast.rightIndex,
    BlockState.readMemValue_real, Region.cast_id, BlockState.readMem,
    TileShape.dropInsertedIndex]
  by_cases hlt : SN + jL.val < slv
  · simp only [hlt, decide_true, if_true, if_pos hlt, BlockState.readMem]
    rw [show jL.val * rs + P1 * hs + e.val + (m1v + SN) * rs
        = (m1v + (SN + jL.val)) * rs + P1 * hs + e.val from by ring]
  · simp only [hlt, decide_false, if_false, if_neg hlt, Bool.false_eq_true]

set_option maxHeartbeats 1600000 in
/-- Generic-value V-load eval (mirror of `ctxNopadIO_kload_eval` at the
`[BLK, DM]` value tile). -/
private theorem ctxNopadIO_vload_eval (u : BlockState) (V : RegionName)
    (m1v slv SN P1 rs hs BLK DM : Nat)
    (hvp : u.regs .ptr [BLK, DM] "v_ptrs" = some (⟨fun idx : TileIndex [BLK, DM] =>
      (V, idx.1.val * rs + P1 * hs + idx.2.1.val)⟩ : Tile .ptr [BLK, DM]))
    (hsl : u.regs .nat [] "cur_batch_in_all_start_index" = some (Tile.scalar m1v))
    (hsn : u.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hn : u.regs .nat [BLK] "offs_n" = some (Tile.vec (fun jn : Fin BLK => jn.val)))
    (hseq : u.regs .nat [] "cur_batch_seq_len" = some (Tile.scalar slv)) :
    evalOp (Op.load .real
        (MemAccess.ptr
          (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BLK, DM] "v_ptrs")
            (Op.mul .nat Broadcast.nil
              (Op.add .nat Broadcast.nil (Op.ref .nat [] "cur_batch_in_all_start_index")
                (Op.ref .nat [] "start_n"))
              (Op.constNat rs))))
        (MaskOpt.maskOther
          (Op.remap [BLK, DM] Broadcast.nil.consL.consSame.leftIndex
            (Op.lt ComparableDType.nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
                (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLK] "offs_n")))
              (Op.ref .nat [] "cur_batch_seq_len")))
          ((Op.const 0.0).broadcast [BLK, DM]))) u
      = some (⟨fun idx : TileIndex [BLK, DM] =>
          if SN + idx.1.val < slv then
            some (u.readMem V ((m1v + (SN + idx.1.val)) * rs + P1 * hs + idx.2.1.val))
          else some (0.0 : ℝ)⟩ : Tile .real [BLK, DM]) := by
  have hexp : @evalOp .nat [BLK, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLK] "offs_n")) u
      = some (Tile.expandDim ⟨1, by simp⟩ (Tile.vec (fun jn : Fin BLK => jn.val))) :=
    evalOp_expandDim_ref_of_regs .nat [BLK] ⟨1, by simp⟩ "offs_n" u _ hn
  simp only [evalOp, hvp, hsl, hsn, hexp, hseq, Option.bind, Option.some.injEq]
  refine congrArg some ?_
  ext idx
  obtain ⟨jL, e, pu⟩ := idx
  simp only [Tile.ptrAdd_data, Tile.cop_data, Tile.bop_data, Tile.bop, Tile.remap,
    Tile.expandDim, Tile.vec, Tile.scalar, NumericDType.add, NumericDType.mul,
    ComparableDType.lt, Broadcast.leftIndex, Broadcast.rightIndex,
    BlockState.readMemValue_real, Region.cast_id, BlockState.readMem,
    TileShape.dropInsertedIndex]
  by_cases hlt : SN + jL.val < slv
  · simp only [hlt, decide_true, if_true, if_pos hlt, BlockState.readMem]
    rw [show jL.val * rs + P1 * hs + e.val + (m1v + SN) * rs
        = (m1v + (SN + jL.val)) * rs + P1 * hs + e.val from by ring]
  · simp only [hlt, decide_false, if_false, if_neg hlt, Bool.false_eq_true]

set_option maxHeartbeats 1600000 in
/-- Generic-tile `qk += dot(q, k)` eval for the weak walk. -/
private theorem ctxNopadIO_qkdot_eval (u : BlockState) (BLK DM : Nat)
    (qk0T : Tile .real [BLK, BLK]) (qt : Tile .real [BLK, DM]) (kt : Tile .real [DM, BLK])
    (hqk0 : u.regs .real [BLK, BLK] "qk" = some qk0T)
    (hqr : u.regs .real [BLK, DM] "q" = some qt)
    (hkr : u.regs .real [DM, BLK] "k" = some kt) :
    evalOp (Op.add .real Broadcast.nil.consSame.consSame (Op.ref .real [BLK, BLK] "qk")
        (Op.dot (batch := []) (Op.ref .real [BLK, DM] "q") (Op.ref .real [DM, BLK] "k"))) u
      = some (Tile.bop NumericDType.real.add Broadcast.nil.consSame.consSame qk0T
          (Tile.dot [] qt kt)) := by
  have hdot : evalOp (Op.dot (batch := [])
      (Op.ref .real [BLK, DM] "q") (Op.ref .real [DM, BLK] "k")) u
      = some (Tile.dot [] qt kt) := by
    rw [evalOp_dot]; erw [evalOp_ref, hqr, evalOp_ref, hkr]; rfl
  rw [evalOp_add, evalOp_ref, hqk0]
  show Option.bind (evalOp (Op.dot (batch := [])
    (Op.ref .real [BLK, DM] "q") (Op.ref .real [DM, BLK] "k")) u) _ = _
  rw [hdot]
  rfl

set_option maxHeartbeats 1600000 in
/-- Generic-tile `acc += dot(p, v)` eval for the weak walk. -/
private theorem ctxNopadIO_accdot_eval (u : BlockState) (BLK DM : Nat)
    (accT : Tile .real [BLK, DM]) (pt : Tile .real [BLK, BLK]) (vt : Tile .real [BLK, DM])
    (haccr : u.regs .real [BLK, DM] "acc" = some accT)
    (hpr : u.regs .real [BLK, BLK] "p" = some pt)
    (hvr : u.regs .real [BLK, DM] "v" = some vt) :
    evalOp (Op.add .real Broadcast.nil.consSame.consSame (Op.ref .real [BLK, DM] "acc")
        (Op.dot (batch := []) (Op.ref .real [BLK, BLK] "p") (Op.ref .real [BLK, DM] "v"))) u
      = some (Tile.bop NumericDType.real.add Broadcast.nil.consSame.consSame accT
          (Tile.dot [] pt vt)) := by
  have hdot : evalOp (Op.dot (batch := [])
      (Op.ref .real [BLK, BLK] "p") (Op.ref .real [BLK, DM] "v")) u
      = some (Tile.dot [] pt vt) := by
    rw [evalOp_dot]; erw [evalOp_ref, hpr, evalOp_ref, hvr]; rfl
  rw [evalOp_add, evalOp_ref, haccr]
  show Option.bind (evalOp (Op.dot (batch := [])
    (Op.ref .real [BLK, BLK] "p") (Op.ref .real [BLK, DM] "v")) u) _ = _
  rw [hdot]
  rfl

set_option maxHeartbeats 1600000 in
/-- Generic-tile causal-`where` eval for the weak walk. -/
private theorem ctxNopadIO_qkwhere_eval (u : BlockState) (BLK : Nat) (SN : Nat)
    (qkT : Tile .real [BLK, BLK]) (offsM : Fin BLK → Nat)
    (hqk : u.regs .real [BLK, BLK] "qk" = some qkT)
    (hm : u.regs .nat [BLK] "offs_m" = some (Tile.vec offsM))
    (hsn : u.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hn : u.regs .nat [BLK] "offs_n" = some (Tile.vec (fun jn : Fin BLK => jn.val))) :
    evalOp ((Op.ge ComparableDType.nat Broadcast.nil.consL.consR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLK] "offs_m"))
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLK] "offs_n")))).where
        (Op.ref .real [BLK, BLK] "qk") (Op.negInf.broadcast [BLK, BLK])) u
      = some (⟨fun idx : TileIndex [BLK, BLK] =>
          if SN + idx.2.1.val ≤ offsM idx.1 then qkT.data idx else (⊥ : WithBot ℝ)⟩
          : Tile .real [BLK, BLK]) := by
  have hexpM : @evalOp .nat [BLK, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLK] "offs_m")) u
      = some (Tile.expandDim ⟨1, by simp⟩ (Tile.vec offsM)) :=
    evalOp_expandDim_ref_of_regs .nat [BLK] ⟨1, by simp⟩ "offs_m" u _ hm
  have hexpN : @evalOp .nat [1, BLK] (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLK] "offs_n")) u
      = some (Tile.expandDim ⟨0, by simp⟩ (Tile.vec (fun jn : Fin BLK => jn.val))) :=
    evalOp_expandDim_ref_of_regs .nat [BLK] ⟨0, by simp⟩ "offs_n" u _ hn
  simp only [evalOp, hexpM, hsn, hexpN, hqk, Option.bind, Option.some.injEq]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, jL, pu⟩ := idx
  simp only [Tile.select_data, Tile.cop_data, Tile.bop_data, Tile.bop, Tile.expandDim, Tile.vec,
    Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex, ComparableDType.ge, NumericDType.add,
    TileShape.dropInsertedIndex]
  by_cases hle : SN + jL.val ≤ offsM i
  · rw [if_pos (by simp only [decide_eq_true_eq]; omega), if_pos hle]
  · rw [if_neg (by simp only [decide_eq_true_eq]; omega), if_neg hle]; rfl

set_option maxHeartbeats 1600000 in
/-- Generic-tile `m_ij = tl.max(qk, 1)` eval for the weak walk. -/
private theorem ctxNopadIO_mij_eval (u : BlockState) (BLK : Nat) (hBLK : 0 < BLK)
    (qkT : Tile .real [BLK, BLK])
    (hqk : u.regs .real [BLK, BLK] "qk" = some qkT) :
    evalOp (Op.reduceMax ⟨1, by simp⟩ Bool.false (Op.ref .real [BLK, BLK] "qk")) u
      = some (⟨fun idx : TileIndex [BLK] =>
          Finset.univ.sup' (⟨⟨0, hBLK⟩, Finset.mem_univ _⟩ : (Finset.univ : Finset (Fin BLK)).Nonempty)
            (fun jL : Fin BLK => qkT.data (idx.1, jL, PUnit.unit))⟩ : Tile .real [BLK]) := by
  rw [evalOp_reduceMax, evalOp_ref, hqk]
  simp only [Option.bind_eq_bind, Option.bind_some, Tile.reduceMax_false]
  unfold Tile.reduceMaxDrop
  rw [dif_pos (show 0 < TileShape.axisDim [BLK, BLK]
    (⟨1, by simp⟩ : Fin ([BLK, BLK] : TileShape).length) from hBLK)]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, pu⟩ := idx
  rfl

set_option maxHeartbeats 1600000 in
/-- Generic-tile `p = exp(qk - m_ij[:, None])` eval for the weak walk. -/
private theorem ctxNopadIO_pexp_eval (u : BlockState) (BLK : Nat)
    (qkT : Tile .real [BLK, BLK]) (mijT : Tile .real [BLK])
    (hqk : u.regs .real [BLK, BLK] "qk" = some qkT)
    (hmij : u.regs .real [BLK] "m_ij" = some mijT) :
    evalOp (Op.sub .real Broadcast.nil.consR.consSame (Op.ref .real [BLK, BLK] "qk")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BLK] "m_ij"))).exp u
      = some (⟨fun idx : TileIndex [BLK, BLK] =>
          WithBot.realExp (WithBot.realSub (qkT.data idx) (mijT.data (idx.1, PUnit.unit)))⟩
          : Tile .real [BLK, BLK]) := by
  have hexp : @evalOp .real [BLK, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BLK] "m_ij")) u
      = some (Tile.expandDim ⟨1, by simp⟩ mijT) :=
    evalOp_expandDim_ref_of_regs .real [BLK] ⟨1, by simp⟩ "m_ij" u _ hmij
  rw [evalOp_exp, evalOp_sub, evalOp_ref, hqk, hexp]
  simp only [Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, jL, pu⟩ := idx
  simp only [Tile.uop_data, Tile.bop_data, Tile.bop, Tile.expandDim, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.sub, TileShape.dropInsertedIndex]

set_option maxHeartbeats 1600000 in
/-- Generic-tile `l_ij = tl.sum(p, 1)` eval for the weak walk. -/
private theorem ctxNopadIO_lij_eval (u : BlockState) (BLK : Nat)
    (pT : Tile .real [BLK, BLK])
    (hp : u.regs .real [BLK, BLK] "p" = some pT) :
    evalOp (Op.reduceSum ⟨1, by simp⟩ Bool.false (Op.ref .real [BLK, BLK] "p")) u
      = some (⟨fun idx : TileIndex [BLK] =>
          Finset.univ.sum (fun jL : Fin BLK => pT.data (idx.1, jL, PUnit.unit))⟩
          : Tile .real [BLK]) := by
  rw [evalOp_reduceSum, evalOp_ref, hp]
  simp only [Option.bind_eq_bind, Option.bind_some, Tile.reduceSum_false]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, pu⟩ := idx
  rfl

set_option maxHeartbeats 1600000 in
/-- Generic-tile `p = p * p_scale[:, None]` eval for the weak walk. -/
private theorem ctxNopadIO_pmul_eval (u : BlockState) (BLK : Nat)
    (pT : Tile .real [BLK, BLK]) (psT : Tile .real [BLK])
    (hp : u.regs .real [BLK, BLK] "p" = some pT)
    (hps : u.regs .real [BLK] "p_scale" = some psT) :
    evalOp (Op.mul .real Broadcast.nil.consR.consSame (Op.ref .real [BLK, BLK] "p")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BLK] "p_scale"))) u
      = some (⟨fun idx : TileIndex [BLK, BLK] =>
          WithBot.realMul (pT.data idx) (psT.data (idx.1, PUnit.unit))⟩
          : Tile .real [BLK, BLK]) := by
  have hexp : @evalOp .real [BLK, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BLK] "p_scale")) u
      = some (Tile.expandDim ⟨1, by simp⟩ psT) :=
    evalOp_expandDim_ref_of_regs .real [BLK] ⟨1, by simp⟩ "p_scale" u _ hps
  rw [evalOp_mul, evalOp_ref, hp, hexp]
  simp only [Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, jL, pu⟩ := idx
  simp only [Tile.bop_data, Tile.bop, Tile.expandDim, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.mul, TileShape.dropInsertedIndex]

set_option maxHeartbeats 1600000 in
/-- Generic-tile `acc = acc * acc_scale[:, None]` eval for the weak walk. -/
private theorem ctxNopadIO_accmul_eval (u : BlockState) (BLK DM : Nat)
    (accT : Tile .real [BLK, DM]) (asT : Tile .real [BLK])
    (hacc : u.regs .real [BLK, DM] "acc" = some accT)
    (has : u.regs .real [BLK] "acc_scale" = some asT) :
    evalOp (Op.mul .real Broadcast.nil.consR.consSame (Op.ref .real [BLK, DM] "acc")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BLK] "acc_scale"))) u
      = some (⟨fun idx : TileIndex [BLK, DM] =>
          WithBot.realMul (accT.data idx) (asT.data (idx.1, PUnit.unit))⟩
          : Tile .real [BLK, DM]) := by
  have hexp : @evalOp .real [BLK, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BLK] "acc_scale")) u
      = some (Tile.expandDim ⟨1, by simp⟩ asT) :=
    evalOp_expandDim_ref_of_regs .real [BLK] ⟨1, by simp⟩ "acc_scale" u _ has
  rw [evalOp_mul, evalOp_ref, hacc, hexp]
  simp only [Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, dd, pu⟩ := idx
  simp only [Tile.bop_data, Tile.bop, Tile.expandDim, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.mul, TileShape.dropInsertedIndex]

set_option maxHeartbeats 8000000 in
set_option maxRecDepth 8000 in
/-- **Weak loop-body walk** (single pass): from any `nopadSafeInvW` state
with the counter register set, the 22 body statements are trace-safe (the
two masked loads bounded by the skin's `read2`/`read3` windows at the
current step) and step to a state satisfying `nopadSafeInvW` again — the
value registers advance existentially, the address registers are untouched
(`k_ptrs`/`v_ptrs` never advance in this kernel). -/
private theorem nopadIO_bodyW (R : RoundingModel) (bounds : RegionBounds)
    (Q K V : RegionName) (B_Start_Loc B_Seqlen : RegionName) (sc : ℝ)
    (rs hs BLK DM : Nat) (hBLK : 0 < BLK)
    (s stt : BlockState) (c : Nat)
    (hP : nopadSafeInvW Q K V B_Start_Loc B_Seqlen rs hs BLK DM s stt)
    (hbK : ∀ (e : Fin DM) (jL : Fin BLK), c + jL.val < seqLen s B_Seqlen →
      (startLoc s B_Start_Loc + (c + jL.val)) * rs + s.pids 1 * hs + e.val < bounds K)
    (hbV : ∀ (jL : Fin BLK) (e : Fin DM), c + jL.val < seqLen s B_Seqlen →
      (startLoc s B_Start_Loc + (c + jL.val)) * rs + s.pids 1 * hs + e.val < bounds V) :
    Stmt.TraceSafeListR R bounds (nopadLoopBodyG sc rs hs BLK DM)
        (stt.setReg "start_n" .nat [] (Tile.scalar c))
      ∧ ∃ s', stepStmtsR R (nopadLoopBodyG sc rs hs BLK DM)
            (stt.setReg "start_n" .nat [] (Tile.scalar c)) = some s'
          ∧ nopadSafeInvW Q K V B_Start_Loc B_Seqlen rs hs BLK DM s s' := by
  obtain ⟨hmem, hpids, hseq, hsl, hsm, hch, hbm, hn, hd, hom, hkp, hvp,
    ⟨qT, hq⟩, ⟨mT, hmi⟩, ⟨lT, hli⟩, ⟨aT, hacc⟩⟩ := hP
  unfold nopadLoopBodyG
  -- (1) `start_n` self-assign
  refine nopadIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (nopadIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp _ _ = some (Tile.scalar c) from by
        rw [evalOp_ref, BlockState.setReg_same])) ?_
  -- (2) the masked `k` load
  refine nopadIO_walkCons ?_
    (nopadIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (ctxNopadIO_kload_eval _ K (startLoc s B_Start_Loc) (seqLen s B_Seqlen) c
        (s.pids 1) rs hs BLK DM
        (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq,
              not_false_eq_true]
            exact hkp)
        (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq,
              not_false_eq_true]
            exact hsl)
        (by simp only [BlockState.setReg_same])
        (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq,
              not_false_eq_true]
            exact hn)
        (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq,
              not_false_eq_true]
            exact hseq))) ?_
  · simp only [Stmt.TraceSafeR, Op.SafeAtR, MaskOpt.ActiveR, MemAccess.ActiveAddressSafeR,
      memAccessActiveAddressSafeR]
    refine ⟨by simp [Op.SafeAtR.eq_def],
      ⟨by simp [Op.SafeAtR.eq_def], by simp [Op.SafeAtR.eq_def]⟩, ?_⟩
    intro ptrs hptrs idx hactive
    rw [ctxNopadIO_kptrR_eval R _ K (startLoc s B_Start_Loc) c (s.pids 1) rs hs BLK DM
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]
          exact hkp)
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]
          exact hsl)
      (by simp only [BlockState.setReg_same])] at hptrs
    obtain rfl := Option.some.inj hptrs
    obtain ⟨masks, hmask, hactl⟩ := hactive
    rw [ctxNopadIO_kmaskR_eval R _ (seqLen s B_Seqlen) c BLK DM
      (by simp only [BlockState.setReg_same])
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]
          exact hn)
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]
          exact hseq)] at hmask
    obtain rfl := Option.some.inj hmask
    obtain ⟨e, jL, pu⟩ := idx
    have hcol : c + jL.val < seqLen s B_Seqlen := by simpa using hactl
    have hb := hbK e jL hcol
    show jL.val * rs + s.pids 1 * hs + e.val + (startLoc s B_Start_Loc + c) * rs < bounds K
    calc jL.val * rs + s.pids 1 * hs + e.val + (startLoc s B_Start_Loc + c) * rs
        = (startLoc s B_Start_Loc + (c + jL.val)) * rs + s.pids 1 * hs + e.val := by ring
      _ < bounds K := hb
  -- (3) `qk = zeros`
  refine nopadIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (nopadIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (by simp [evalOp_full, evalOp_const]
          try rfl)) ?_
  -- (4) `qk += dot(q, k)`
  refine nopadIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (nopadIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (ctxNopadIO_qkdot_eval _ BLK DM _ _ _
        (by rw [BlockState.setReg_same])
        (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]
            exact hq)
        (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
          BlockState.setReg_same]
            rfl))) ?_
  -- (5) `qk *= sm_scale`
  refine nopadIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (nopadIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (by simp only [evalOp, evalOp_expandDim_ref_of_regs, evalOp_expandDim_ref, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
        String.reduceEq, not_false_eq_true, Option.bind]
          try rfl)) ?_
  -- (6) the causal `where`
  refine nopadIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (nopadIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (ctxNopadIO_qkwhere_eval _ BLK c _ _
        (by rw [BlockState.setReg_same])
        (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]
            exact hom)
        (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
          BlockState.setReg_same])
        (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]
            exact hn))) ?_
  -- (7) `m_ij = tl.max(qk, 1)`
  refine nopadIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (nopadIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (ctxNopadIO_mij_eval _ BLK hBLK _
        (by rw [BlockState.setReg_same]))) ?_
  -- (8) `p = exp(qk - m_ij[:, None])`
  refine nopadIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (nopadIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (ctxNopadIO_pexp_eval _ BLK _ _
        (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
          BlockState.setReg_same]
            rfl)
        (by rw [BlockState.setReg_same]))) ?_
  -- (9) `l_ij = tl.sum(p, 1)`
  refine nopadIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (nopadIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (ctxNopadIO_lij_eval _ BLK _
        (by rw [BlockState.setReg_same]))) ?_
  -- (10) `m_i_new = maximum(m_i, m_ij)`
  refine nopadIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (nopadIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (by simp only [evalOp, evalOp_expandDim_ref_of_regs, evalOp_expandDim_ref, hmi, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
        String.reduceEq, not_false_eq_true, Option.bind]
          try rfl)) ?_
  -- (11) `alpha = exp(m_i - m_i_new)`
  refine nopadIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (nopadIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (by simp only [evalOp, evalOp_expandDim_ref_of_regs, evalOp_expandDim_ref, hmi, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
        String.reduceEq, not_false_eq_true, Option.bind]
          try rfl)) ?_
  -- (12) `beta = exp(m_ij - m_i_new)`
  refine nopadIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (nopadIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (by simp only [evalOp, evalOp_expandDim_ref_of_regs, evalOp_expandDim_ref, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
        String.reduceEq, not_false_eq_true, Option.bind]
          try rfl)) ?_
  -- (13) `l_i_new = alpha*l_i + beta*l_ij`
  refine nopadIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (nopadIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (by simp only [evalOp, evalOp_expandDim_ref_of_regs, evalOp_expandDim_ref, hli, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
        String.reduceEq, not_false_eq_true, Option.bind]
          try rfl)) ?_
  -- (14) `p_scale = beta / l_i_new`
  refine nopadIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (nopadIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (by simp only [evalOp, evalOp_expandDim_ref_of_regs, evalOp_expandDim_ref, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
        String.reduceEq, not_false_eq_true, Option.bind]
          try rfl)) ?_
  -- (15) `p = p * p_scale[:, None]`
  refine nopadIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (nopadIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (ctxNopadIO_pmul_eval _ BLK _ _
        (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
          BlockState.setReg_same]
            rfl)
        (by rw [BlockState.setReg_same]))) ?_
  -- (16) `acc_scale = l_i / l_i_new * alpha`
  refine nopadIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (nopadIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (by simp only [evalOp, evalOp_expandDim_ref_of_regs, evalOp_expandDim_ref, hli, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
        String.reduceEq, not_false_eq_true, Option.bind]
          try rfl)) ?_
  -- (17) `acc = acc * acc_scale[:, None]`
  refine nopadIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (nopadIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (ctxNopadIO_accmul_eval _ BLK DM _ _
        (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]
            exact hacc)
        (by rw [BlockState.setReg_same]))) ?_
  -- (18) the masked `v` load
  refine nopadIO_walkCons ?_
    (nopadIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (ctxNopadIO_vload_eval _ V (startLoc s B_Start_Loc) (seqLen s B_Seqlen) c
        (s.pids 1) rs hs BLK DM
        (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq,
              not_false_eq_true]
            exact hvp)
        (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq,
              not_false_eq_true]
            exact hsl)
        (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq,
              not_false_eq_true, BlockState.setReg_same])
        (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq,
              not_false_eq_true]
            exact hn)
        (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq,
              not_false_eq_true]
            exact hseq))) ?_
  · simp only [Stmt.TraceSafeR, Op.SafeAtR, MaskOpt.ActiveR, MemAccess.ActiveAddressSafeR,
      memAccessActiveAddressSafeR]
    refine ⟨by simp [Op.SafeAtR.eq_def],
      ⟨by simp [Op.SafeAtR.eq_def], by simp [Op.SafeAtR.eq_def]⟩, ?_⟩
    intro ptrs hptrs idx hactive
    rw [ctxNopadIO_vptrR_eval R _ V (startLoc s B_Start_Loc) c (s.pids 1) rs hs BLK DM
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]
          exact hvp)
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]
          exact hsl)
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
        BlockState.setReg_same])] at hptrs
    obtain rfl := Option.some.inj hptrs
    obtain ⟨masks, hmask, hactl⟩ := hactive
    rw [ctxNopadIO_vmaskR_eval R _ (seqLen s B_Seqlen) c BLK DM
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
        BlockState.setReg_same])
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]
          exact hn)
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]
          exact hseq)] at hmask
    obtain rfl := Option.some.inj hmask
    obtain ⟨jL, e, pu⟩ := idx
    have hrow : c + jL.val < seqLen s B_Seqlen := by simpa using hactl
    have hb := hbV jL e hrow
    show jL.val * rs + s.pids 1 * hs + e.val + (startLoc s B_Start_Loc + c) * rs < bounds V
    calc jL.val * rs + s.pids 1 * hs + e.val + (startLoc s B_Start_Loc + c) * rs
        = (startLoc s B_Start_Loc + (c + jL.val)) * rs + s.pids 1 * hs + e.val := by ring
      _ < bounds V := hb
  -- (19) `p` self-assign (the erased `.to(v.dtype)` cast)
  refine nopadIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (nopadIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (by simp only [evalOp, evalOp_expandDim_ref_of_regs, evalOp_expandDim_ref, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
        String.reduceEq, not_false_eq_true, Option.bind]
          try rfl)) ?_
  -- (20) `acc += dot(p, v)`
  refine nopadIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (nopadIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (ctxNopadIO_accdot_eval _ BLK DM _ _ _
        (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
          BlockState.setReg_same]
            rfl)
        (by rw [BlockState.setReg_same])
        (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
          BlockState.setReg_same]
            rfl))) ?_
  -- (21) `l_i = l_i_new`
  refine nopadIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (nopadIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (by simp only [evalOp, evalOp_expandDim_ref_of_regs, evalOp_expandDim_ref, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
        String.reduceEq, not_false_eq_true, Option.bind]
          try rfl)) ?_
  -- (22) `m_i = m_i_new`
  refine nopadIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (nopadIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (by simp only [evalOp, evalOp_expandDim_ref_of_regs, evalOp_expandDim_ref, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
        String.reduceEq, not_false_eq_true, Option.bind]
          try rfl)) ?_
  refine ⟨Stmt.TraceSafeListR.nil_intro, _, stepStmtsR_nil R _, ?_⟩
  refine ⟨?_, ?_,
    (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]
        exact hseq),
    (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]
        exact hsl),
    (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]
        exact hsm),
    (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]
        exact hch),
    (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]
        exact hbm),
    (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]
        exact hn),
    (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]
        exact hd),
    (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]
        exact hom),
    (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]
        exact hkp),
    (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]
        exact hvp),
    ⟨qT, by
      simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]
      exact hq⟩,
    ⟨_, by rw [BlockState.setReg_same]⟩,
    ⟨_, by
      simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
        BlockState.setReg_same]
      rfl⟩,
    ⟨_, by
      simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
        BlockState.setReg_same]
      rfl⟩⟩
  · funext rg o
    simp only [BlockState.setReg_mem]
    rw [hmem]
  · simp only [BlockState.setReg_pids]
    exact hpids

/-- `evalOpR` of the `off_o` offset tree is the exact evaluation
(register-only `.nat` arithmetic — R-independent, ∀-state). -/
private theorem ctxNopadIO_offoR_castFree (R : RoundingModel) (rs hs BLK DM : Nat)
    (u : BlockState) :
    evalOpR R (Op.add .nat Broadcast.nil.consL.consR
        (Op.add .nat Broadcast.scalarR
          (Op.mul .nat Broadcast.scalarR
            (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "cur_batch_in_all_start_index")
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLK] "offs_m")))
            (Op.constNat rs))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat hs)))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [DM] "offs_d"))
          (Op.constNat 1))) u
      = evalOp (Op.add .nat Broadcast.nil.consL.consR
          (Op.add .nat Broadcast.scalarR
            (Op.mul .nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "cur_batch_in_all_start_index")
                (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLK] "offs_m")))
              (Op.constNat rs))
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat hs)))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [DM] "offs_d"))
            (Op.constNat 1))) u := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

/-- `evalOpR` of the `out_ptrs` pointer tree is the exact evaluation. -/
private theorem ctxNopadIO_outptrR_castFree (R : RoundingModel) (Out : RegionName)
    (BLK DM : Nat) (u : BlockState) :
    evalOpR R (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Out) (Op.ref .nat [BLK, DM] "off_o")) u
      = evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Out) (Op.ref .nat [BLK, DM] "off_o")) u := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

set_option maxHeartbeats 8000000 in
set_option maxRecDepth 8000 in
/-- **The `TraceSafeR` walk for the whole kernel** from an arbitrary launch
state under the `pre`-legal window bounds (`pid₂ < NT`): weak preLoop walk,
the streaming `forRangeDyn` driven by `Stmt.forRangeTraceSafeR_inv` over
`nopadSafeInvW`, and the terminal masked store bounded by the `write`
window. -/
private theorem nopadIO_traceSafeR (R : RoundingModel) (bounds : RegionBounds)
    (Q K V : RegionName) (B_Start_Loc B_Seqlen : Region .nat) (Out : RegionName)
    (sm_scale : ℝ) (rs hs BLK DM NT : Nat) (hBLK : 0 < BLK)
    (s : BlockState) (hpid2 : s.pids 2 < NT)
    (hbSeq : s.pids 0 < bounds (Region.cast B_Seqlen))
    (hbLoc : s.pids 0 < bounds (Region.cast B_Start_Loc))
    (hbQ : ∀ (i : Fin BLK) (e : Fin DM), s.pids 2 * BLK + i.val < seqLen s ↑B_Seqlen →
      (startLoc s ↑B_Start_Loc + (s.pids 2 * BLK + i.val)) * rs + s.pids 1 * hs + e.val
        < bounds Q)
    (hbK : ∀ (t : Fin NT) (e : Fin DM) (jL : Fin BLK),
      t.val * BLK + jL.val < seqLen s ↑B_Seqlen →
      (startLoc s ↑B_Start_Loc + (t.val * BLK + jL.val)) * rs + s.pids 1 * hs + e.val
        < bounds K)
    (hbV : ∀ (t : Fin NT) (jL : Fin BLK) (e : Fin DM),
      t.val * BLK + jL.val < seqLen s ↑B_Seqlen →
      (startLoc s ↑B_Start_Loc + (t.val * BLK + jL.val)) * rs + s.pids 1 * hs + e.val
        < bounds V)
    (hbO : ∀ (i : Fin BLK) (e : Fin DM), s.pids 2 * BLK + i.val < seqLen s ↑B_Seqlen →
      (startLoc s ↑B_Start_Loc + (s.pids 2 * BLK + i.val)) * rs + s.pids 1 * hs + e.val
        < bounds Out) :
    ((context_attn_nopad_fwd_kernel_surface Q K V sm_scale B_Start_Loc B_Seqlen Out
      rs hs 1 rs hs 1 rs hs 1 rs hs 1 BLK DM BLK).toAlgKernel).TraceSafeR R bounds s := by
  -- shared per-iteration body facts
  have hSTOPle : (if BLK * s.pids 2 < seqLen s ↑B_Seqlen then 1 else 0) * (s.pids 2 + 1) * BLK
      ≤ (s.pids 2 + 1) * BLK := by
    by_cases h : BLK * s.pids 2 < seqLen s ↑B_Seqlen
    · simp [h]
    · simp [h]
  have hbody : ∀ (cc : Nat) (st : BlockState),
      cc < (if BLK * s.pids 2 < seqLen s ↑B_Seqlen then 1 else 0) * (s.pids 2 + 1) * BLK →
      nopadSafeInvW Q K V ↑B_Start_Loc ↑B_Seqlen rs hs BLK DM s st ∧ cc % BLK = 0 →
      Stmt.TraceSafeListR R bounds (nopadLoopBodyG sm_scale rs hs BLK DM)
          (st.setReg "start_n" .nat [] (Tile.scalar cc))
        ∧ ∃ s', stepStmtsR R (nopadLoopBodyG sm_scale rs hs BLK DM)
              (st.setReg "start_n" .nat [] (Tile.scalar cc)) = some s'
            ∧ (nopadSafeInvW Q K V ↑B_Start_Loc ↑B_Seqlen rs hs BLK DM s s'
                ∧ (cc + BLK) % BLK = 0) := by
    intro cc st hcc hPP
    obtain ⟨hPinv, hPmod⟩ := hPP
    have hceq : cc / BLK * BLK = cc := Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hPmod)
    have hcNT : cc / BLK < NT := by
      have h1 : cc < (s.pids 2 + 1) * BLK := lt_of_lt_of_le hcc hSTOPle
      have h2 : cc / BLK < s.pids 2 + 1 := (Nat.div_lt_iff_lt_mul hBLK).mpr h1
      omega
    obtain ⟨hsafeB, s', hrunB, hInvB⟩ :=
      nopadIO_bodyW R bounds Q K V ↑B_Start_Loc ↑B_Seqlen sm_scale rs hs BLK DM hBLK s st cc
        hPinv
        (fun e jL hc => by
          have hb := hbK ⟨cc / BLK, hcNT⟩ e jL (by rw [hceq]; exact hc)
          rwa [hceq] at hb)
        (fun jL e hc => by
          have hb := hbV ⟨cc / BLK, hcNT⟩ jL e (by rw [hceq]; exact hc)
          rwa [hceq] at hb)
    exact ⟨hsafeB, s', hrunB, hInvB, by rw [Nat.add_mod_right]; exact hPmod⟩
  unfold Kernel.TraceSafeR
  rw [nopad_body_splitG]
  obtain ⟨hsafePre, s0, hrun0, hInv0⟩ :=
    nopadIO_preLoopW R bounds Q K V B_Start_Loc B_Seqlen rs hs BLK DM s hbSeq hbLoc hbQ
  have hInv0' := hInv0
  obtain ⟨hmem0, hpids0, hseq0, hsl0, hsm0, hch0, hbm0, hn0, hd0, hom0, hkp0, hvp0,
    hqE, hmiE, hliE, haccE⟩ := hInv0
  have hstopExact : evalOp (Op.mul .nat Broadcast.nil
      (Op.mul .nat Broadcast.nil (Op.ref .nat [] "block_mask")
        (Op.add .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat 1)))
      (Op.constNat BLK)) s0
      = some (Tile.scalar ((if BLK * s.pids 2 < seqLen s ↑B_Seqlen then 1 else 0)
          * (s.pids 2 + 1) * BLK)) := by
    rw [evalOp_mul, evalOp_mul, evalOp_ref, hbm0, evalOp_add, evalOp_ref, hsm0,
      evalOp_constNat, evalOp_constNat]
    simp only [Option.bind_eq_bind, Option.bind_some]
    refine congrArg some ?_
    refine Tile.ext (fun idx => ?_)
    simp only [Tile.bop_data, Tile.bop, Tile.scalar, Broadcast.leftIndex,
      Broadcast.rightIndex, NumericDType.mul, NumericDType.add]
  refine Stmt.TraceSafeListR.append_intro _ _ hsafePre ?_
  intro s1 hs1
  rw [hrun0] at hs1
  obtain rfl := Option.some.inj hs1
  refine Stmt.TraceSafeListR.cons_intro ?_ (fun s2 hs2 => ?_)
  · -- TraceSafeR of the forRangeDyn itself
    simp only [Stmt.TraceSafeR]
    refine ⟨by simp [Op.SafeAtR.eq_def], by simp [Op.SafeAtR.eq_def],
      by simp [Op.SafeAtR.eq_def], ?_⟩
    rw [ctxNopadIO_evalOpR_constNat, ctxNopadIO_evalOpR_constNat,
      ctxNopadIO_stopOpR_castFree, hstopExact]
    refine Stmt.forRangeTraceSafeR_inv R bounds "start_n" _ _
      (nopadLoopBodyG sm_scale rs hs BLK DM)
      (fun i st => nopadSafeInvW Q K V ↑B_Start_Loc ↑B_Seqlen rs hs BLK DM s st
        ∧ i % BLK = 0) ?_ _ s0 ⟨hInv0', Nat.zero_mod BLK⟩
    intro cc st hcc hPP
    obtain ⟨hsafeB, s', hrunB, hInvB⟩ := hbody cc st hcc hPP
    exact ⟨hsafeB, s', hrunB, hInvB⟩
  · -- the loop's actual successor, then the postLoop
    obtain ⟨finalC, sL, hLoopExact, hfin, hPL⟩ :=
      forRangeDyn_inv (idx := "start_n")
        (startOp := Op.constNat 0)
        (stopOp := Op.mul .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "block_mask")
            (Op.add .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat 1)))
          (Op.constNat BLK))
        (stepOp := Op.constNat BLK)
        (P := fun i st => nopadSafeInvW Q K V ↑B_Start_Loc ↑B_Seqlen rs hs BLK DM s st
          ∧ i % BLK = 0)
        (s_init := s0)
        (evalOp_constNat 0 s0) hstopExact (evalOp_constNat BLK s0)
        hBLK.ne'
        ⟨hInv0', Nat.zero_mod BLK⟩
        (fun i st hi hP => by
          obtain ⟨hsafeB, s', hrunB, hInvB⟩ := hbody i st hi hP
          exact ⟨s', by
            rw [← ctxNopadIO_body_castFree R sm_scale rs hs BLK DM]
            exact hrunB, hInvB⟩)
    rw [ctxNopadIO_dyn_castFree R sm_scale rs hs BLK DM s0, hLoopExact] at hs2
    obtain rfl := Option.some.inj hs2
    obtain ⟨⟨hmemL, hpidsL, hseqL, hslL, hsmL, hchL, hbmL, hnL, hdL, homL, hkpL, hvpL,
      hqEL, hmiEL, hliEL, haccEL⟩, hmodL⟩ := hPL
    -- postLoop: `off_o` assign, `out_ptrs` assign, the masked terminal store
    refine Stmt.TraceSafeListR.cons_intro
      (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s3 hs3 => ?_)
    obtain ⟨v3, hv3, rfl⟩ := stepStmtR_assign_inv hs3
    rw [ctxNopadIO_offoR_castFree R rs hs BLK DM,
      nopad_offq_evalG sL (startLoc s ↑B_Start_Loc) (s.pids 1) (s.pids 2) rs hs BLK DM
        hslL hchL homL hdL] at hv3
    obtain rfl := Option.some.inj hv3
    refine Stmt.TraceSafeListR.cons_intro
      (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s4 hs4 => ?_)
    obtain ⟨v4, hv4, rfl⟩ := stepStmtR_assign_inv hs4
    rw [ctxNopadIO_outptrR_castFree R Out BLK DM] at hv4
    simp only [evalOp, evalOp_ref, BlockState.setReg_same, Option.bind] at hv4
    obtain rfl := Option.some.inj hv4
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
    rw [ctxNopadIO_rowmaskR_eval R _ (seqLen s ↑B_Seqlen) (s.pids 2) BLK DM
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]
          exact homL)
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]
          exact hseqL)] at hmask
    obtain rfl := Option.some.inj hmask
    obtain ⟨i2, e2, pu⟩ := idx
    have hrow : s.pids 2 * BLK + i2.val < seqLen s ↑B_Seqlen := by simpa using hactl
    have hb := hbO i2 e2 hrow
    simpa [Tile.ptrAdd_data, Tile.scalar_data, Broadcast.leftIndex, Broadcast.rightIndex,
      Region.cast_id] using hb

/-! ## The rounded Hoare triple (`hrun`): framed exact run

`hrun` rides the exact `nopadPreLoop_evalG` → `forRangeDyn_inv` →
`nopadPostLoop_evalG` stack unchanged (everything is cast-free, so `execR R`
collapses onto it verbatim); the only new obligation the skin adds over the
existing headline is the **memory frame**, recovered by replaying the
deterministic 3-statement postLoop and framing its masked scatter. -/

/-- A masked `writeMem` scatter `foldl` leaves every cell not hit by an
active lane untouched. -/
private theorem nopadIO_foldl_writeMem_frame_masked {α : Type} (region : RegionName)
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
          nopadIO_foldl_writeMem_frame_masked region offFn valFn P rest _ r o
            (fun hr k' hk' hPk' => h hr k' (List.mem_cons_of_mem _ hk') hPk'),
          BlockState.writeMem_mem]
        rw [if_neg (fun hro => h hro.1 k List.mem_cons_self hPk hro.2.symm)]
      · rw [if_neg hPk]
        exact nopadIO_foldl_writeMem_frame_masked region offFn valFn P rest _ r o
          (fun hr k' hk' hPk' => h hr k' (List.mem_cons_of_mem _ hk') hPk')

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **PostLoop frame**: every cell outside the terminal store's active
window is untouched by the postLoop (replay of `nopadPostLoop_evalG`'s
deterministic chain, framed). -/
private theorem nopadIO_postLoop_frame
    (Q K V : RegionName) (B_Start_Loc B_Seqlen : RegionName) (Out : RegionName)
    (sm_scale : ℝ) (rs hs BLK DM : Nat) (hDM : 0 < DM)
    (s0 : BlockState) (cF : Nat) (sL sP : BlockState)
    (hinv : nopadInvariantG Q K V B_Start_Loc B_Seqlen sm_scale rs hs BLK DM hDM s0 cF sL)
    (hpost : stepStmts (nopadPostLoopG Out rs hs BLK DM) sL = some sP) :
    ∀ r o,
      (r = Out → ∀ idx : TileIndex [BLK, DM],
        s0.pids 2 * BLK + idx.1.val < seqLen s0 B_Seqlen →
        o ≠ outOffsetG s0 B_Start_Loc rs hs BLK DM idx) →
      sP.mem r o = sL.mem r o := by
  obtain ⟨hpids, hmem, hundef, hcb, hch, hsm, hseq, hsl, hn, hd, hoffm, hq, hkp, hvp,
    hbmask, hmi, hli, hacc⟩ := hinv
  unfold nopadPostLoopG at hpost
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (nopad_offq_evalG sL (startLoc s0 B_Start_Loc) (s0.pids 1) (s0.pids 2) rs hs BLK DM
      hsl hch hoffm hd))] at hpost
  set s1 := sL.setReg "off_o" .nat [BLK, DM]
    (⟨fun idx : TileIndex [BLK, DM] =>
      (startLoc s0 B_Start_Loc + (s0.pids 2 * BLK + idx.1.val)) * rs
        + s0.pids 1 * hs + idx.2.1.val⟩ : Tile .nat [BLK, DM]) with hs1d
  have hs1offo : s1.regs .nat [BLK, DM] "off_o" = some
      (⟨fun idx : TileIndex [BLK, DM] =>
        (startLoc s0 B_Start_Loc + (s0.pids 2 * BLK + idx.1.val)) * rs
          + s0.pids 1 * hs + idx.2.1.val⟩ : Tile .nat [BLK, DM]) := by
    rw [hs1d, BlockState.setReg_same]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Out) (Op.ref .nat [BLK, DM] "off_o")) s1
        = some (⟨fun idx : TileIndex [BLK, DM] =>
            (Out, (startLoc s0 B_Start_Loc + (s0.pids 2 * BLK + idx.1.val)) * rs
              + s0.pids 1 * hs + idx.2.1.val)⟩ : Tile .ptr [BLK, DM]) from by
      simp only [evalOp, evalOp_ref, hs1offo, Option.bind]
      refine congrArg some (Tile.ext (fun idx => ?_))
      obtain ⟨ir, dd, pu⟩ := idx
      simp only [Tile.ptrAdd_data, Tile.scalar_data, Broadcast.leftIndex, Broadcast.rightIndex,
        Region.cast_id, Nat.zero_add, Prod.mk.injEq, true_and]))] at hpost
  set s2 := s1.setReg "out_ptrs" .ptr [BLK, DM]
    (⟨fun idx : TileIndex [BLK, DM] =>
      (Out, (startLoc s0 B_Start_Loc + (s0.pids 2 * BLK + idx.1.val)) * rs
        + s0.pids 1 * hs + idx.2.1.val)⟩ : Tile .ptr [BLK, DM]) with hs2d
  have hs2ptr : s2.regs .ptr [BLK, DM] "out_ptrs" = some
      (⟨fun idx : TileIndex [BLK, DM] =>
        (Out, (startLoc s0 B_Start_Loc + (s0.pids 2 * BLK + idx.1.val)) * rs
          + s0.pids 1 * hs + idx.2.1.val)⟩ : Tile .ptr [BLK, DM]) := by
    rw [hs2d, BlockState.setReg_same]
  have hs2acc : s2.regs .real [BLK, DM] "acc" = some
      (⟨fun idx : TileIndex [BLK, DM] =>
        ((nopadFoldUptoG s0 Q K V B_Start_Loc sm_scale rs hs BLK DM
          (ctxNopadWindowG s0 B_Seqlen BLK) (seqLen s0 B_Seqlen) (cF * BLK)
          idx.1 idx.2.1).2.2 : WithBot ℝ)⟩ : Tile .real [BLK, DM]) := by
    rw [hs2d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      hs1d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
    exact hacc
  have hs2m : s2.regs .nat [BLK] "offs_m"
      = some (Tile.vec (fun i : Fin BLK => s0.pids 2 * BLK + i.val)) := by
    rw [hs2d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      hs1d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
    exact hoffm
  have hs2seq : s2.regs .nat [] "cur_batch_seq_len"
      = some (Tile.scalar (seqLen s0 B_Seqlen)) := by
    rw [hs2d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      hs1d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
    exact hseq
  have hstore : stepStmt (Stmt.store .real [BLK, DM] (MemAccess.ptr (Op.ref .ptr [BLK, DM] "out_ptrs"))
      (Op.ref .real [BLK, DM] "acc")
      (MaskOpt.mask (Op.remap [BLK, DM] Broadcast.nil.consL.consSame.leftIndex
        (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLK] "offs_m"))
          (Op.ref .nat [] "cur_batch_seq_len"))))) s2
      = some ((TileShape.allIndices [BLK, DM]).foldl
          (fun acc idx =>
            if s0.pids 2 * BLK + idx.1.val < seqLen s0 B_Seqlen then
              acc.writeMem Out ((startLoc s0 B_Start_Loc + (s0.pids 2 * BLK + idx.1.val)) * rs + s0.pids 1 * hs + idx.2.1.val)
                ((nopadFoldUptoG s0 Q K V B_Start_Loc sm_scale rs hs BLK DM (ctxNopadWindowG s0 B_Seqlen BLK) (seqLen s0 B_Seqlen) (cF * BLK) idx.1 idx.2.1).2.2 : ℝ)
            else acc) s2) := by
    have hexpM : @evalOp .nat [BLK, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLK] "offs_m")) s2
        = some (Tile.expandDim ⟨1, by simp⟩ (Tile.vec (fun i : Fin BLK => s0.pids 2 * BLK + i.val))) :=
      evalOp_expandDim_ref_of_regs .nat [BLK] ⟨1, by simp⟩ "offs_m" s2 _ hs2m
    unfold stepStmt
    simp only [evalOp_ref, hs2acc, hs2ptr, hs2seq, evalOp, hexpM, Option.bind, Option.map]
    refine congrArg some ?_
    congr 1
    funext acc idx
    obtain ⟨ir, dd, pu⟩ := idx
    simp only [Tile.cop_data, Tile.bop_data, Tile.bop, Tile.remap, Tile.expandDim, Tile.vec,
      Tile.scalar, ComparableDType.lt, Broadcast.leftIndex, Broadcast.rightIndex,
      TileShape.dropInsertedIndex, BlockState.writeMemTyped_real, FloatDType.real_storeValue,
      decide_eq_true_eq, WithBot.unbotD_coe]
  rw [stepStmts.cons_some hstore, stepStmts.nil] at hpost
  obtain rfl := Option.some.inj hpost
  intro r o hno
  refine (nopadIO_foldl_writeMem_frame_masked Out _ _ _ _ s2 r o ?_).trans ?_
  · intro hr k _ hPk
    exact Ne.symm (hno hr k hPk)
  · rw [hs2d, hs1d]
    simp only [BlockState.setReg_mem]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **Framed general execution**: `nopad_exec_general` extended with the
per-cell memory frame (this is the exact run `hrun` rides). -/
private theorem nopadIO_exec_framed
    (Q K V : RegionName) (B_Start_Loc B_Seqlen : Region .nat) (Out : RegionName)
    (sm_scale : ℝ) (rs hs BLK DM : Nat) (hBLK : 0 < BLK) (hDM : 0 < DM) (hDMrs : DM ≤ rs)
    (s : BlockState) (hundef : ∀ rg o, s.undef rg o = 0) :
    ∃ sF, stepStmts ((context_attn_nopad_fwd_kernel_surface Q K V sm_scale B_Start_Loc B_Seqlen Out
        rs hs 1 rs hs 1 rs hs 1 rs hs 1 BLK DM BLK).toAlgKernel.body) s = some sF
      ∧ (∀ idx : TileIndex [BLK, DM],
          s.pids 2 * BLK + idx.1.val < seqLen s B_Seqlen →
            sF.readMem Out (outOffsetG s B_Start_Loc rs hs BLK DM idx)
              = ctxNopadGenuineOutValueG s Q K V B_Start_Loc B_Seqlen sm_scale rs hs BLK DM idx)
      ∧ (∀ r o, (r ≠ Out ∨ ∀ idx : TileIndex [BLK, DM],
            s.pids 2 * BLK + idx.1.val < seqLen s B_Seqlen →
              o ≠ outOffsetG s B_Start_Loc rs hs BLK DM idx) →
          sF.mem r o = s.mem r o) := by
  rw [nopad_body_splitG]
  obtain ⟨s0, hpre, hinv0⟩ := nopadPreLoop_evalG s Q K V B_Start_Loc B_Seqlen sm_scale rs hs BLK DM hBLK hDM hundef
  rw [stepStmts.append_some hpre]
  obtain ⟨hpids0, hmem0, hundef0, hcb0, hch0, hsm0, hseq0, hsl0, hn0, hd0, hoffm0, hq0, hkp0, hvp0, hbmask0, hmi0, hli0, hacc0⟩ := hinv0
  have hs0seq : seqLen s0 B_Seqlen = seqLen s B_Seqlen := seqLen_eq_of_mem_pids s0 s B_Seqlen hmem0 hpids0
  set S := ctxNopadWindowG s0 B_Seqlen BLK with hSdef
  have hSmulBLK : S % BLK = 0 := by
    rw [hSdef, ctxNopadWindowG]
    by_cases h : BLK * s0.pids 2 < seqLen s0 B_Seqlen
    · simp only [h, if_true, one_mul]; exact Nat.mul_mod_left _ _
    · simp only [h, if_false, Nat.zero_mul]; exact Nat.zero_mod BLK
  have hinv0' : nopadInvariantG Q K V B_Start_Loc B_Seqlen sm_scale rs hs BLK DM hDM s0 0 s0 :=
    nopadInvariantG_zero_reanchor Q K V B_Start_Loc B_Seqlen sm_scale rs hs BLK DM hDM s s0 ⟨hpids0, hmem0, hundef0, hcb0, hch0, hsm0, hseq0, hsl0, hn0, hd0, hoffm0, hq0, hkp0, hvp0, hbmask0, hmi0, hli0, hacc0⟩
  obtain ⟨final, sL, hloop, hfin, hinvL⟩ :=
    VeriTile.Triton.forRangeDyn_inv (idx := "start_n")
      (startOp := Op.constNat 0)
      (stopOp := Op.mul .nat Broadcast.nil
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "block_mask")
          (Op.add .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat 1)))
        (Op.constNat BLK))
      (stepOp := Op.constNat BLK)
      (P := fun i st => nopadInvariantG Q K V B_Start_Loc B_Seqlen sm_scale rs hs BLK DM hDM s0 (i / BLK) st ∧ i % BLK = 0 ∧ i ≤ S)
      (s_init := s0) (start := 0) (stop := S) (step := BLK)
      (by rw [evalOp_constNat])
      (by
        rw [evalOp_mul, evalOp_mul, evalOp_ref, hbmask0, evalOp_add, evalOp_ref, hsm0, evalOp_constNat, evalOp_constNat]
        simp only [Option.bind_eq_bind, Option.bind_some]
        refine congrArg some ?_
        refine Tile.ext (fun idx => ?_)
        simp only [Tile.bop_data, Tile.bop, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.mul, NumericDType.add]
        show (if BLK * s.pids 2 < seqLen s B_Seqlen.cast then 1 else 0) * (s.pids 2 + 1) * BLK = S
        rw [hSdef, ctxNopadWindowG, hpids0, hs0seq])
      (by rw [evalOp_constNat])
      (by omega)
      ⟨by rw [Nat.zero_div]; exact hinv0', Nat.zero_mod BLK, Nat.zero_le _⟩
      (fun i st hi hP => by
        obtain ⟨hPinv, hPmod, hPle⟩ := hP
        have hdvd : i = (i / BLK) * BLK := (Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hPmod)).symm
        obtain ⟨s', hstep, hinv'⟩ := nopad_attn_stepG Q K V B_Start_Loc B_Seqlen sm_scale rs hs BLK DM hBLK hDM s0 i st hi hPinv hdvd
        have hmod' : (i + BLK) % BLK = 0 := by rw [Nat.add_mod_right]; exact hPmod
        have hle' : i + BLK ≤ S := by
          have hlt : i < S := hi
          obtain ⟨a, ha⟩ : BLK ∣ i := Nat.dvd_of_mod_eq_zero hPmod
          obtain ⟨b, hb⟩ : BLK ∣ S := Nat.dvd_of_mod_eq_zero hSmulBLK
          have hab : a < b := by
            rw [ha, hb] at hlt
            by_contra hc
            have : BLK * b ≤ BLK * a := Nat.mul_le_mul_left BLK (Nat.le_of_not_lt hc)
            omega
          rw [ha, hb]
          calc BLK * a + BLK = BLK * (a + 1) := by ring
            _ ≤ BLK * b := Nat.mul_le_mul_left BLK (by omega)
        refine ⟨s', hstep, ?_, hmod', hle'⟩
        rw [show (i + BLK) / BLK = i / BLK + 1 from by
          rw [Nat.add_div_right i hBLK]]
        exact hinv')
  rw [stepStmts.cons_some hloop]
  obtain ⟨hinvLinv, hinvLmod, hinvLle⟩ := hinvL
  have hfinS : final = S := Nat.le_antisymm hinvLle hfin
  rw [hfinS] at hinvLinv
  have hfull : (S / BLK) * BLK = S := Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hSmulBLK)
  obtain ⟨sP, hpost, hOut⟩ := nopadPostLoop_evalG Q K V B_Start_Loc B_Seqlen Out sm_scale rs hs BLK DM hDM hDMrs s0 (S / BLK) sL
    (by rw [hfull]) hinvLinv
  have hframeP := nopadIO_postLoop_frame Q K V ↑B_Start_Loc ↑B_Seqlen Out sm_scale rs hs BLK DM hDM
    s0 (S / BLK) sL sP hinvLinv hpost
  have hmemL : sL.mem = s0.mem := hinvLinv.2.1
  rw [hpost]
  refine ⟨sP, rfl, ?_, ?_⟩
  · intro idx hact
    have hslEq0 : startLoc s B_Start_Loc = startLoc s0 B_Start_Loc :=
      startLoc_eq_of_mem_pids s s0 B_Start_Loc hmem0.symm hpids0.symm
    have hoeq : outOffsetG s B_Start_Loc rs hs BLK DM idx = outOffsetG s0 B_Start_Loc rs hs BLK DM idx := by
      simp only [outOffsetG, ← hpids0, hslEq0]
    have hactS0 : s0.pids 2 * BLK + idx.1.val < seqLen s0 B_Seqlen := by
      rw [hs0seq, hpids0]; exact hact
    rw [hoeq, hOut idx, if_pos hactS0]
    exact ctxNopadGenuineOutValueG_eq_of_mem_pids s0 s Q K V B_Start_Loc.cast B_Seqlen.cast sm_scale rs hs BLK DM hmem0 hpids0 idx
  · intro r o hcond
    have hslEq0 : startLoc s B_Start_Loc = startLoc s0 B_Start_Loc :=
      startLoc_eq_of_mem_pids s s0 B_Start_Loc hmem0.symm hpids0.symm
    have hcond0 : r = Out → ∀ idx : TileIndex [BLK, DM],
        s0.pids 2 * BLK + idx.1.val < seqLen s0 B_Seqlen →
        o ≠ outOffsetG s0 B_Start_Loc rs hs BLK DM idx := by
      intro hr idx hact0
      rcases hcond with hne | hno
      · exact absurd hr hne
      · have hacts : s.pids 2 * BLK + idx.1.val < seqLen s B_Seqlen := by
          rw [← hs0seq, ← hpids0]; exact hact0
        have h := hno idx hacts
        rw [show outOffsetG s B_Start_Loc rs hs BLK DM idx
            = outOffsetG s0 B_Start_Loc rs hs BLK DM idx from by
          simp only [outOffsetG, ← hpids0, hslEq0]] at h
        exact h
    rw [hframeP r o hcond0, hmemL]
    rw [show s0.mem = s.mem from hmem0]

/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/
set_option maxHeartbeats 8000000 in
set_option maxRecDepth 8000 in
/-- **The `⊨[R]` streaming-metadata headline (wave-5) — the exemplar
consumer of `StreamMetaMasked3DKernelIO₃`, the context-attention trio's
skin.** For every rounding model `R`, the faithful `context_attn_nopad`
surface implements, on its metadata three-stream signature, the **honest
causal softmax** over the streamed `Q`/`K`/`V` tiles: every write-active
output lane `j = (i, e)` holds

`Σ_{jg ≤ pid₂·BLK+i} exp(sm_scale·raw jg)·V[jg,e] / Σ_{jg ≤ pid₂·BLK+i} exp(sm_scale·raw jg)`

(`contextAttnNopadIOSpec` — exactly the port's genuine closed form
`contextAttnNopadExactFoldMG` restated on the streams: the `float("-inf")`
causal mask genuinely drops future keys, the `K`/`V` boundary masks zero
out-of-sequence keys, and the in-loop-normalized `acc` needs no
post-divide). The two `.nat` slots enter only through the windows/masks
(`m 0 = B_Seqlen[pid₀]` the boundary, `m 1 = B_Start_Loc[pid₀]` the packed
row offset). The kernel has **zero rounding events** (`.nat` slot loads,
`other=0.0`-masked `.real` loads, `.real` in-loop arithmetic, untyped
terminal store), so the skin's boundary quantization degenerates: the
readback's `R.round .real` is the identity by the model's defining
`round_real`.

**Launch legality (the skin's genre novelty).** The triple is guarded by

`io.pre pid₂ m = (pid₂ < NT ∧ m 0 ≤ NT·BLK)`

— the port's documented **trusted boundary** (see the file docstring's
Scope section): the host launches
`grid = (batch, head, cdiv(max_input_len, BLOCK_M))` with `NT` the third
dimension and every `B_Seqlen[b] ≤ max_input_len ≤ NT·BLOCK_M`. The live
trip count `block_mask·(pid₂+1)` has no pid-free bound, so `pid₂ < NT` is
what makes the `T = NT`-step window citable at all — it is threaded into
both the safety walk (every live step `c ≤ pid₂ < NT` indexes `Fin NT`)
and the value bridge (`gi < (pid₂+1)·BLK ≤ NT·BLK` makes the causal window
extension inert).

**Hypothesis provenance** (all truth-forced, inherited from the exact
headline `context_attn_nopad_output_summary_general`): `0 < BLK`, `0 < DM`
(nonempty tiles); `DM ≤ rs` (output-offset injectivity — the contiguous
layout has `rs = H·DM ≥ DM`, so no open `hInj` side condition); `0 < NT`
(the host grid's third dimension is `cdiv(max_input_len, BLOCK_M) ≥ 1`;
also forced by the static `Q` stream's step-`0` read). The exact headline's
`hundef` is **not** a hypothesis here — the skin's Hoare triple carries the
`undef` pin itself.

Relation to the exact surface: the `Realizes_without_Rounding` headline
above is retained unchanged; this `⊨[R]` face restates the same causal
softmax on the streaming metadata skin, for every `R` at once. -/
specification context_attn_nopad_io_correctness (R : RoundingModel)
    (Q K V : RegionName) (B_Start_Loc B_Seqlen : Region .nat) (Out : RegionName)
    (sm_scale : ℝ) (rs hs BLK DM NT : Nat)
    (hBLK : 0 < BLK) (hDM : 0 < DM) (hDMrs : DM ≤ rs) (hNT : 0 < NT) :
    contextAttnNopadIO Q K V B_Start_Loc B_Seqlen Out sm_scale rs hs BLK DM NT ⊨[R]
      fun _ _ pid₂ m xs ys zs j =>
        contextAttnNopadIOSpec BLK DM NT (m (⟨0, by omega⟩ : Fin 2)) hBLK hNT sm_scale
          pid₂ xs ys zs j := by
  refine StreamMetaMasked3DKernelIO₃.ImplementsR.intro _ ?_ ?_ ?_
  · exact ctxNopad_flattenOk Q K V B_Start_Loc B_Seqlen Out sm_scale rs hs BLK DM
  · -- the safety walk
    intro bounds s m xs ys zs hpre hm _hx _hy _hz hbm hbr1 hbr2 hbr3 hbw
    simp only [contextAttnNopadIO] at hpre hm hbm hbr1 hbr2 hbr3 hbw ⊢
    obtain ⟨hpid2, _hm0NT⟩ := hpre
    have hm0 : seqLen s ↑B_Seqlen = m (⟨0, by omega⟩ : Fin 2) := hm (⟨0, by omega⟩ : Fin 2)
    have hm1 : startLoc s ↑B_Start_Loc = m (⟨1, by omega⟩ : Fin 2) :=
      hm (⟨1, by omega⟩ : Fin 2)
    refine nopadIO_traceSafeR R bounds Q K V B_Start_Loc B_Seqlen Out sm_scale
      rs hs BLK DM NT hBLK s hpid2 (hbm (⟨0, by omega⟩ : Fin 2))
      (hbm (⟨1, by omega⟩ : Fin 2)) ?_ ?_ ?_ ?_
    · intro i e hrow
      have h := hbr1 ⟨0, hNT⟩ (Lane2D.encode (i, e, PUnit.unit))
        (by rw [Lane2D.encode_div, ← hm0]; exact hrow)
      rw [Lane2D.encode_div, Lane2D.encode_mod, ← hm1] at h
      exact h
    · intro t e jL hcol
      have h := hbr2 t (Lane2D.encode (e, jL, PUnit.unit))
        (by rw [Lane2D.encode_mod, ← hm0]; exact hcol)
      rw [Lane2D.encode_div, Lane2D.encode_mod, ← hm1] at h
      exact h
    · intro t jL e hcol
      have h := hbr3 t (Lane2D.encode (jL, e, PUnit.unit))
        (by rw [Lane2D.encode_div, ← hm0]; exact hcol)
      rw [Lane2D.encode_div, Lane2D.encode_mod, ← hm1] at h
      exact h
    · intro i e hrow
      have h := hbw (Lane2D.encode (i, e, PUnit.unit))
        (by rw [Lane2D.encode_div, ← hm0]; exact hrow)
      rw [Lane2D.encode_div, Lane2D.encode_mod, ← hm1] at h
      exact h
  · -- the rounded Hoare triple: framed exact stack + cast-free collapse
    intro s₀ m xs ys zs hpre hu hm hx hy hz
    simp only [contextAttnNopadIO] at hpre hm hx hy hz ⊢
    obtain ⟨hpid2, _hm0NT⟩ := hpre
    have hundef' : ∀ rg o, s₀.undef rg o = 0 := fun rg o => by rw [hu]
    have hm0 : seqLen s₀ ↑B_Seqlen = m (⟨0, by omega⟩ : Fin 2) :=
      hm (⟨0, by omega⟩ : Fin 2)
    have hm1 : startLoc s₀ ↑B_Start_Loc = m (⟨1, by omega⟩ : Fin 2) :=
      hm (⟨1, by omega⟩ : Fin 2)
    obtain ⟨sF, hstep, hOut, hframe⟩ :=
      nopadIO_exec_framed Q K V B_Start_Loc B_Seqlen Out sm_scale rs hs BLK DM
        hBLK hDM hDMrs s₀ hundef'
    refine ⟨sF, ?_, ?_, ?_⟩
    · rw [ctxNopadIO_execR_collapse]
      exact hstep
    · -- readback: the genuine closed form = the streamed closed form
      intro j hj
      have hactIdx : s₀.pids 2 * BLK + (Lane2D.decode j).1.val < seqLen s₀ B_Seqlen := by
        rw [Lane2D.decode_row, hm0]
        exact hj
      have hOutj := hOut (Lane2D.decode j) hactIdx
      have haddr : (startLoc s₀ ↑B_Start_Loc + (s₀.pids 2 * BLK + j.val / DM)) * rs
            + s₀.pids 1 * hs + j.val % DM
          = outOffsetG s₀ ↑B_Start_Loc rs hs BLK DM (Lane2D.decode j) := by
        simp only [outOffsetG, Lane2D.decode_row, Lane2D.decode_col]
      rw [BlockState.readMemAs_real, ← hm1, haddr, hOutj, R.round_real_apply]
      refine congrArg some ?_
      exact ctxNopadIOSpec_eq_genuine s₀ Q K V B_Start_Loc B_Seqlen sm_scale rs hs BLK DM
        NT (m (⟨0, by omega⟩ : Fin 2)) (m (⟨1, by omega⟩ : Fin 2)) hBLK hNT hpid2 hm0 hm1
        xs ys zs hx hy hz j hj
    · -- the frame
      intro r o hcond
      refine hframe r o ?_
      rcases hcond with hne | hno
      · exact Or.inl hne
      · refine Or.inr (fun idx hact hoeq => ?_)
        exact hno (Lane2D.encode idx)
          (by rw [Lane2D.encode_div, ← hm0]; exact hact)
          (hoeq.trans (by simp only [outOffsetG, Lane2D.encode_div, Lane2D.encode_mod, hm1]))

end IOFace

end VeriTile.Bench.TritonBenchG.ContextAttnNopad
