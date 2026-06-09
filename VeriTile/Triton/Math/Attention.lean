/-
VeriTile.Triton.Math.Attention

Pure real-valued attention specifications and shape helpers.
-/

import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Tactic
import VeriTile.Triton.Core.Shape

namespace VeriTile.Triton

/-! ## Pure attention math -/

/-- Scaled score `(Q · K^T · scale)[i, j]` for real-valued Q/K slices. -/
noncomputable def scaledScore {M S D : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S, D] → ℝ) (scale : ℝ)
    (i : Fin M) (j : Fin S) : ℝ :=
  scale * Finset.univ.sum (fun d : Fin D =>
    Q (i, d, PUnit.unit) * K (j, d, PUnit.unit))

/-- ℝ-valued reference attention:
`softmax(Q · K^T · scale) · V`. -/
noncomputable def attentionReal {M S D : Nat}
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (scale : ℝ) : TileIndex [M, D] → ℝ :=
  fun (i, d, _) =>
    let score := fun j : Fin S => scaledScore Q K scale i j
    let denom := Finset.univ.sum (fun j : Fin S => Real.exp (score j))
    let numer := Finset.univ.sum (fun j : Fin S =>
      Real.exp (score j) * V (j, d, PUnit.unit))
    numer / denom

/-- Base-2 reference attention with a per-key score scale.

Matches quantized flash-attention kernels that (a) use `exp2` instead of `exp`
for the softmax and (b) scale each key's score by a per-block quantization
factor (so different keys may carry different scales). For query `i`, output
channel `d`:

`out[i,d] = (Σ_j 2 ^ (keyScale j · raw i j) · V[j,d]) / (Σ_j 2 ^ (keyScale j · raw i j))`,

where `raw i j = Σ_e Q[i,e] · K[j,e]` is the unscaled score. Using
`2 ^ x = Real.exp (Real.log 2 · x)`, this is the base-`e` softmax of
`Real.log 2 · keyScale j · raw i j`. With a constant `keyScale ≡ scale`, it
specialises to `attentionReal` with effective scale `Real.log 2 · scale`. -/
noncomputable def attentionRealBase2PerKeyScale {M S D : Nat}
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (keyScale : Fin S → ℝ) : TileIndex [M, D] → ℝ :=
  fun (i, d, _) =>
    let raw := fun j : Fin S =>
      Finset.univ.sum (fun e : Fin D => Q (i, e, PUnit.unit) * K (j, e, PUnit.unit))
    let weight := fun j : Fin S => Real.exp (Real.log 2 * (keyScale j * raw j))
    let denom := Finset.univ.sum (fun j : Fin S => weight j)
    let numer := Finset.univ.sum (fun j : Fin S => weight j * V (j, d, PUnit.unit))
    numer / denom

/-! ## Online (flash) softmax recurrence == batch softmax

The flash-attention kernel never materializes the full score row. Instead it
streams keys, maintaining a running max `m`, a running denominator `l`, and a
running (unnormalized) accumulator `acc`, rescaling by `2^(m_old - m_new)` each
step. This section proves — purely over ℝ, independent of any kernel — that the
streamed result equals the batch base-2 softmax `Σ 2^s v / Σ 2^s`.

Everything is stated generically over the number of keys `S`, the head width
`D`, and arbitrary per-key scores (so per-key / per-block quantization scaling
is free). It is the mathematical heart of the `attention_forward_triton`
correctness proof. The exponential is concretely `exp2` (`pow2`) — the only base
this family of kernels uses. -/

/-- Base-2 power `2^x = exp(log 2 · x)` — the `exp2` used by flash kernels. -/
noncomputable def pow2 (x : ℝ) : ℝ := Real.exp (Real.log 2 * x)

@[simp] theorem pow2_zero : pow2 0 = 1 := by simp [pow2]

theorem pow2_pos (x : ℝ) : 0 < pow2 x := Real.exp_pos _

theorem pow2_add (a b : ℝ) : pow2 (a + b) = pow2 a * pow2 b := by
  unfold pow2; rw [mul_add, Real.exp_add]

/-- One online-softmax update step for a single output channel: absorb key
`(s, v)` (score `s`, value `v`) into running state `(m, l, acc)`. -/
noncomputable def osStep (st : ℝ × ℝ × ℝ) (sv : ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let m := st.1; let l := st.2.1; let acc := st.2.2
  let s := sv.1; let v := sv.2
  let m' := max m s
  let α := pow2 (m - m')
  let p := pow2 (s - m')
  (m', l * α + p, acc * α + p * v)

/-- **Online-softmax invariant.** Folding `osStep` over a key list keeps the
running denominator and accumulator equal to `2^(-m)` times the *true*
(un-shifted) batch denominator `Σ 2^s` and accumulator `Σ 2^s · v`. Stated for
an arbitrary consistent start state so it proves by a clean `foldl` induction. -/
theorem osStep_foldl_consistent (xs : List (ℝ × ℝ)) (m l acc T L : ℝ)
    (hl : l = pow2 (-m) * L) (hacc : acc = pow2 (-m) * T) :
    let st := xs.foldl osStep (m, l, acc)
    st.2.1 = pow2 (-st.1) * (L + (xs.map (fun p => pow2 p.1)).sum) ∧
    st.2.2 = pow2 (-st.1) * (T + (xs.map (fun p => pow2 p.1 * p.2)).sum) := by
  induction xs generalizing m l acc T L with
  | nil => simp [hl, hacc]
  | cons x xs ih =>
    obtain ⟨s, v⟩ := x
    set m' := max m s with hm'
    -- New running state after absorbing `(s, v)`.
    have key : pow2 (-m) * pow2 (m - m') = pow2 (-m') := by
      rw [← pow2_add]; ring_nf
    have hl' : l * pow2 (m - m') + pow2 (s - m') = pow2 (-m') * (L + pow2 s) := by
      rw [hl]
      have e1 : pow2 (-m) * L * pow2 (m - m') = pow2 (-m') * L := by
        rw [mul_right_comm, key]
      have e2 : pow2 (s - m') = pow2 (-m') * pow2 s := by
        rw [← pow2_add]; ring_nf
      rw [e1, e2]; ring
    have hacc' : acc * pow2 (m - m') + pow2 (s - m') * v
        = pow2 (-m') * (T + pow2 s * v) := by
      rw [hacc]
      have e1 : pow2 (-m) * T * pow2 (m - m') = pow2 (-m') * T := by
        rw [mul_right_comm, key]
      have e2 : pow2 (s - m') = pow2 (-m') * pow2 s := by
        rw [← pow2_add]; ring_nf
      rw [e1, e2]; ring
    have step := ih m' (l * pow2 (m - m') + pow2 (s - m'))
      (acc * pow2 (m - m') + pow2 (s - m') * v)
      (T + pow2 s * v) (L + pow2 s) hl' hacc'
    simpa [List.foldl_cons, osStep, hm', List.map_cons, add_assoc] using step

/-- **Online softmax = batch softmax (single channel).** Folding `osStep` over
all keys from the neutral start `(0, 0, 0)` yields `acc / l` equal to the batch
base-2 softmax `Σ 2^s v / Σ 2^s`. The running max cancels in the ratio. Per-key /
per-block score scaling is already covered: the scores `p.1` are arbitrary. -/
theorem osStep_foldl_eq_batch (xs : List (ℝ × ℝ)) :
    let st := xs.foldl osStep (0, 0, 0)
    st.2.2 / st.2.1
      = (xs.map (fun p => pow2 p.1 * p.2)).sum / (xs.map (fun p => pow2 p.1)).sum := by
  have h := osStep_foldl_consistent xs 0 0 0 0 0 (by simp) (by simp)
  obtain ⟨hL, hT⟩ := h
  simp only at hL hT ⊢
  rw [hT, hL]
  by_cases hm : pow2 (-(xs.foldl osStep (0, 0, 0)).1) = 0
  · -- `pow2` is never zero, so this branch is vacuous.
    exact absurd hm (ne_of_gt (pow2_pos _))
  · rw [zero_add, zero_add, mul_div_mul_left _ _ hm]

/-- Per-key list of `(score, value)` pairs that the kernel streams for output
`(i, d)`: score `keyScale j · (Q row i · K row j)`, value `V (j, d)`. -/
noncomputable def attnKeyList {M S D : Nat}
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (keyScale : Fin S → ℝ) (i : Fin M) (d : Fin D) : List (ℝ × ℝ) :=
  List.ofFn (fun j : Fin S =>
    (keyScale j * Finset.univ.sum (fun e : Fin D =>
        Q (i, e, PUnit.unit) * K (j, e, PUnit.unit)),
      V (j, d, PUnit.unit)))

/-- **Bridge: the base-2 per-key-scale attention spec IS the online-softmax
fold.** The closed-form spec `attentionRealBase2PerKeyScale` equals the
`acc / l` ratio produced by streaming `osStep` over `attnKeyList` from the
neutral start. The eventual `exec` proof only has to show the kernel's loop
realizes this fold; this lemma then delivers the closed form. -/
theorem attentionRealBase2PerKeyScale_eq_streaming {M S D : Nat}
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (keyScale : Fin S → ℝ) (i : Fin M) (d : Fin D) :
    attentionRealBase2PerKeyScale Q K V keyScale (i, d, PUnit.unit)
      = (let st := (attnKeyList Q K V keyScale i d).foldl osStep (0, 0, 0)
         st.2.2 / st.2.1) := by
  rw [osStep_foldl_eq_batch]
  simp only [attentionRealBase2PerKeyScale, attnKeyList, List.map_ofFn, List.sum_ofFn,
    Function.comp]
  rfl

/-! ### Block-level online softmax (matching the kernel's loop)

The kernel does not stream one key at a time; each loop iteration absorbs a whole
`BLOCK_N`-key block, taking the block max and rescaling the accumulator *once* per
block. `osBlockStep` models exactly that, and `osBlockStep_foldl_consistent` shows
the block recurrence keeps the same `l = 2^(-m)·Σ2^s`, `acc = 2^(-m)·Σ2^s·v`
invariant — so the block fold and the per-key fold agree on the final ratio. This
is the form the `exec` loop invariant carries (accumulator state after `k`
blocks). -/

/-- Factor the running-max shift out of a block sum: `Σ 2^(s-m')·g = 2^(-m')·Σ 2^s·g`. -/
theorem sum_map_pow2_sub (m' : ℝ) (block : List (ℝ × ℝ)) (g : ℝ × ℝ → ℝ) :
    (block.map (fun p => pow2 (p.1 - m') * g p)).sum
      = pow2 (-m') * (block.map (fun p => pow2 p.1 * g p)).sum := by
  induction block with
  | nil => simp
  | cons a t ih =>
    have e : pow2 (a.1 - m') = pow2 (-m') * pow2 a.1 := by
      rw [← pow2_add]; ring_nf
    simp only [List.map_cons, List.sum_cons, ih, e]; ring

/-- Running block max: `max` of the entry max `m₀` and every score in the block. -/
def blockMax (m₀ : ℝ) (block : List (ℝ × ℝ)) : ℝ :=
  block.foldl (fun a p => max a p.1) m₀

/-- One online-softmax update absorbing an entire `BLOCK_N`-key block: take the
block max, rescale by `2^(m_old - m_new)` once, add the block's contributions. -/
noncomputable def osBlockStep (st : ℝ × ℝ × ℝ) (block : List (ℝ × ℝ)) : ℝ × ℝ × ℝ :=
  let m := st.1; let l := st.2.1; let acc := st.2.2
  let m' := blockMax m block
  let α := pow2 (m - m')
  (m', l * α + (block.map (fun p => pow2 (p.1 - m'))).sum,
       acc * α + (block.map (fun p => pow2 (p.1 - m') * p.2)).sum)

/-- **Block-level online-softmax invariant.** Same consistency as the per-key
version: folding `osBlockStep` over a list of blocks keeps `l` and `acc` equal to
`2^(-m)` times the true batch denominator/accumulator over all keys (the flattened
block list). The block max chosen each step is irrelevant to this identity. -/
theorem osBlockStep_foldl_consistent (blocks : List (List (ℝ × ℝ)))
    (m l acc T L : ℝ) (hl : l = pow2 (-m) * L) (hacc : acc = pow2 (-m) * T) :
    let st := blocks.foldl osBlockStep (m, l, acc)
    st.2.1 = pow2 (-st.1) * (L + (blocks.flatten.map (fun p => pow2 p.1)).sum) ∧
    st.2.2 = pow2 (-st.1) * (T + (blocks.flatten.map (fun p => pow2 p.1 * p.2)).sum) := by
  induction blocks generalizing m l acc T L with
  | nil => simp [hl, hacc]
  | cons b bs ih =>
    set m' := blockMax m b with hm'
    have key : pow2 (-m) * pow2 (m - m') = pow2 (-m') := by
      rw [← pow2_add]; ring_nf
    -- The new l after absorbing block `b`.
    have hl' : l * pow2 (m - m') + (b.map (fun p => pow2 (p.1 - m'))).sum
        = pow2 (-m') * (L + (b.map (fun p => pow2 p.1)).sum) := by
      rw [hl]
      have e1 : pow2 (-m) * L * pow2 (m - m') = pow2 (-m') * L := by
        rw [mul_right_comm, key]
      have e2 : (b.map (fun p => pow2 (p.1 - m'))).sum
          = pow2 (-m') * (b.map (fun p => pow2 p.1)).sum := by
        have := sum_map_pow2_sub m' b (fun _ => 1)
        simpa using this
      rw [e1, e2]; ring
    have hacc' : acc * pow2 (m - m') + (b.map (fun p => pow2 (p.1 - m') * p.2)).sum
        = pow2 (-m') * (T + (b.map (fun p => pow2 p.1 * p.2)).sum) := by
      rw [hacc]
      have e1 : pow2 (-m) * T * pow2 (m - m') = pow2 (-m') * T := by
        rw [mul_right_comm, key]
      rw [e1, sum_map_pow2_sub m' b (fun p => p.2)]; ring
    have step := ih m'
      (l * pow2 (m - m') + (b.map (fun p => pow2 (p.1 - m'))).sum)
      (acc * pow2 (m - m') + (b.map (fun p => pow2 (p.1 - m') * p.2)).sum)
      (T + (b.map (fun p => pow2 p.1 * p.2)).sum)
      (L + (b.map (fun p => pow2 p.1)).sum) hl' hacc'
    simpa [List.foldl_cons, osBlockStep, hm', List.flatten_cons, List.map_append,
      List.sum_append, add_assoc] using step

/-- **Block online softmax = batch softmax.** Folding `osBlockStep` over all
blocks from the neutral start gives `acc / l` equal to the batch base-2 softmax
over every key (the flattened block list). -/
theorem osBlockStep_foldl_eq_batch (blocks : List (List (ℝ × ℝ))) :
    let st := blocks.foldl osBlockStep (0, 0, 0)
    st.2.2 / st.2.1
      = (blocks.flatten.map (fun p => pow2 p.1 * p.2)).sum
        / (blocks.flatten.map (fun p => pow2 p.1)).sum := by
  have h := osBlockStep_foldl_consistent blocks 0 0 0 0 0 (by simp) (by simp)
  obtain ⟨hL, hT⟩ := h
  simp only at hL hT ⊢
  rw [hT, hL]
  by_cases hm : pow2 (-(blocks.foldl osBlockStep (0, 0, 0)).1) = 0
  · exact absurd hm (ne_of_gt (pow2_pos _))
  · rw [zero_add, zero_add, mul_div_mul_left _ _ hm]

/-- Summing `h` over the flattened block grid `[[g 0 0, g 0 1, …], [g 1 0, …], …]`
equals the double sum. Bridges the kernel's block-indexed loop (a list of blocks)
to the spec's flat per-key `Finset.sum`. -/
theorem sum_flatten_ofFn_ofFn {α : Type*} (m n : Nat)
    (g : Fin m → Fin n → α) (h : α → ℝ) :
    (((List.ofFn (fun b : Fin m => List.ofFn (fun a : Fin n => g b a))).flatten).map h).sum
      = ∑ b : Fin m, ∑ a : Fin n, h (g b a) := by
  simp [List.map_flatten, List.sum_flatten, List.map_ofFn, List.sum_ofFn, Function.comp]

/-- Reindex a flat per-key sum `∑ j : Fin (BLOCK_N · numKVBlocks)` as a block grid
`∑ b ∑ a` with global key `b · BLOCK_N + a`. The bridge from the spec's per-key
`Finset.sum` to the kernel's `(block, key-in-block)` loop structure. -/
theorem sum_fin_eq_block_grid (BLOCK_N numKVBlocks : Nat)
    (F : Fin (BLOCK_N * numKVBlocks) → ℝ) :
    (∑ j : Fin (BLOCK_N * numKVBlocks), F j)
      = ∑ b : Fin numKVBlocks, ∑ a : Fin BLOCK_N,
          F (Fin.cast (Nat.mul_comm numKVBlocks BLOCK_N) (finProdFinEquiv (b, a))) := by
  rw [← Equiv.sum_comp (finProdFinEquiv.trans (finCongr (Nat.mul_comm numKVBlocks BLOCK_N))) F,
      Fintype.sum_prod_type]
  rfl

/-! ## Generalized base-2 streaming-softmax recurrence (arbitrary per-key score)

The flash-attention loop maintains running `(m, l, acc)` per query row, absorbing
one `BLOCK_N`-key block per iteration. The lemmas below model that recurrence and
prove it equals the batch base-2 softmax — **generically over an arbitrary
per-key score function** `score : Fin Mq → Fin (BLOCK_N · numKVBlocks) → ℝ`
(rather than the hardcoded `keyScale · rawScore`). Concrete kernels recover their
specialization by instantiating `score i j := keyScale j · rawScore Q K i j` (see
`attnGenScore_eq_streaming` and the `attentionRealBase2PerKeyScale` bridge).

`gkey b a` is the global key index `b · BLOCK_N + a` of local lane `a` in block
`b`; the `g`-suffixed names (`mPg`, `lPg`, `oPg`, …) are the generalized partials.
-/

/-- Global key index of local lane `a` in block `b` (`b < nB`): `b · BN + a`. -/
def gkey (BN nB : Nat) (b : Nat) (hb : b < nB) (a : Fin BN) : Fin (BN * nB) :=
  ⟨b * BN + a.val, by
    calc b * BN + a.val < (b + 1) * BN := by ring_nf; omega
      _ ≤ nB * BN := Nat.mul_le_mul_right _ hb
      _ = BN * nB := by ring⟩

/-- `gkey b a` (global key `b·BN + a`) is the block-grid reindex element. -/
theorem gkey_eq_cast {BN nB : Nat} (b : Fin nB) (a : Fin BN) :
    gkey BN nB b.val b.isLt a = Fin.cast (Nat.mul_comm nB BN) (finProdFinEquiv (b, a)) := by
  apply Fin.ext
  simp only [gkey, finProdFinEquiv_apply_val, Fin.coe_cast]
  ring

/-- Running per-row max of per-key scores over the first `c` blocks, seeded `⊥`. -/
noncomputable def mPg {Mq : Nat} (BN nB : Nat)
    (score : Fin Mq → Fin (BN * nB) → ℝ) (i : Fin Mq) : Nat → WithBot ℝ
  | 0 => (⊥ : WithBot ℝ)
  | c + 1 =>
      if h : c + 1 ≤ nB then
        max (mPg BN nB score i c)
          (Finset.univ.sup (fun a : Fin BN =>
            ((score i (gkey BN nB c (by omega) a) : ℝ) : WithBot ℝ)))
      else
        mPg BN nB score i c

/-- The running max is finite (`≠ ⊥`) once at least one nonempty block is seen. -/
theorem mPg_ne_bot {Mq : Nat} {BN nB : Nat}
    (score : Fin Mq → Fin (BN * nB) → ℝ) (hBN : 0 < BN) (i : Fin Mq)
    (c : Nat) (hc1 : 1 ≤ c) (hc : c ≤ nB) :
    mPg BN nB score i c ≠ (⊥ : WithBot ℝ) := by
  obtain ⟨c, rfl⟩ : ∃ c', c = c' + 1 := ⟨c - 1, by omega⟩
  simp only [mPg, dif_pos hc]
  have hle := Finset.le_sup (f := fun a : Fin BN =>
      ((score i (gkey BN nB c (by omega) a) : ℝ) : WithBot ℝ))
      (Finset.mem_univ (⟨0, hBN⟩ : Fin BN))
  intro hmax
  rw [max_eq_bot] at hmax
  rw [hmax.2] at hle
  exact absurd (le_bot_iff.mp hle) (WithBot.coe_ne_bot)

/-- Real running max (the kernel's `m_i`, with `⊥` read as `0` — only used at
`c ≥ 1` where it is genuinely finite, see `mPg_ne_bot`). -/
noncomputable def mRg {Mq : Nat} {BN nB : Nat}
    (score : Fin Mq → Fin (BN * nB) → ℝ) (i : Fin Mq) (c : Nat) : ℝ :=
  (mPg BN nB score i c).unbotD 0

theorem mPg_eq_coe {Mq : Nat} {BN nB : Nat}
    (score : Fin Mq → Fin (BN * nB) → ℝ) (hBN : 0 < BN) (i : Fin Mq)
    (c : Nat) (hc1 : 1 ≤ c) (hc : c ≤ nB) :
    mPg BN nB score i c = ((mRg score i c : ℝ) : WithBot ℝ) := by
  have hne := mPg_ne_bot score hBN i c hc1 hc
  unfold mRg
  cases h : mPg BN nB score i c with
  | bot => exact absurd h hne
  | coe r => simp

/-- `m`-free running denominator `Σ_{b<c} Σ_a pow2(score)` (no max shift). -/
noncomputable def lFree2g {Mq : Nat} {BN nB : Nat}
    (score : Fin Mq → Fin (BN * nB) → ℝ) (i : Fin Mq) : Nat → ℝ
  | 0 => 0
  | c + 1 =>
      if h : c + 1 ≤ nB then
        lFree2g score i c +
          Finset.univ.sum (fun a : Fin BN => pow2 (score i (gkey BN nB c (by omega) a)))
      else lFree2g score i c

/-- Online-softmax rescale multiplier `α = exp2(m_c − m_{c+1})` (and `0` at `c = 0`
where the running max is `⊥`, matching the kernel's first-block α-kill). -/
noncomputable def alphaPg {Mq : Nat} {BN nB : Nat}
    (score : Fin Mq → Fin (BN * nB) → ℝ) (i : Fin Mq) (c : Nat) : ℝ :=
  match mPg BN nB score i c with
  | ⊥ => 0
  | (r : ℝ) => pow2 (r - mRg score i (c + 1))

/-- Running per-row denominator `l_i` after `c` blocks (kernel seed `l = 1`). -/
noncomputable def lPg {Mq : Nat} {BN nB : Nat}
    (score : Fin Mq → Fin (BN * nB) → ℝ) (i : Fin Mq) : Nat → ℝ
  | 0 => 1
  | c + 1 =>
      if h : c + 1 ≤ nB then
        alphaPg score i c * lPg score i c +
          Finset.univ.sum (fun a : Fin BN =>
            pow2 (score i (gkey BN nB c (by omega) a) - mRg score i (c + 1)))
      else lPg score i c

theorem alphaPg_zero {Mq : Nat} {BN nB : Nat}
    (score : Fin Mq → Fin (BN * nB) → ℝ) (i : Fin Mq) :
    alphaPg score i 0 = 0 := by
  unfold alphaPg
  rw [show mPg BN nB score i 0 = (⊥ : WithBot ℝ) from rfl]

theorem alphaPg_succ {Mq : Nat} {BN nB : Nat}
    (score : Fin Mq → Fin (BN * nB) → ℝ) (hBN : 0 < BN) (i : Fin Mq)
    (c : Nat) (hc1 : 1 ≤ c) (hc : c + 1 ≤ nB) :
    alphaPg score i c = pow2 (mRg score i c - mRg score i (c + 1)) := by
  unfold alphaPg
  rw [mPg_eq_coe score hBN i c hc1 (by omega)]

/-- Factor the max-shift out of a per-block denominator sum. -/
theorem block_sum_shiftg {Mq : Nat} {BN nB : Nat}
    (score : Fin Mq → Fin (BN * nB) → ℝ) (i : Fin Mq) (c : Nat) (hc : c < nB) (m : ℝ) :
    Finset.univ.sum (fun a : Fin BN =>
        pow2 (score i (gkey BN nB c (by omega) a) - m))
      = pow2 (-m) * Finset.univ.sum (fun a : Fin BN =>
          pow2 (score i (gkey BN nB c (by omega) a))) := by
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro a _
  rw [show score i (gkey BN nB c (by omega) a) - m
        = (-m) + score i (gkey BN nB c (by omega) a) by ring, pow2_add]

/-- **Denominator consistency.** For `1 ≤ c ≤ nB`, the kernel's running
denominator `l_i` equals `pow2(−m_i)` times the max-free reference sum. The
`α`-rescale telescopes the max shift; the `l = 1` seed is killed at `c = 0` by
`alphaPg_zero`. -/
theorem lPg_eq {Mq : Nat} {BN nB : Nat}
    (score : Fin Mq → Fin (BN * nB) → ℝ) (hBN : 0 < BN) (i : Fin Mq) :
    ∀ c, 1 ≤ c → c ≤ nB →
      lPg score i c = pow2 (-(mRg score i c)) * lFree2g score i c := by
  intro c hc1 hc
  induction c, hc1 using Nat.le_induction with
  | base =>
    have h1 : (1 : Nat) ≤ nB := hc
    simp only [lPg, dif_pos h1, alphaPg_zero score i, zero_mul, zero_add]
    rw [block_sum_shiftg score i 0 (by omega)]
    simp only [lFree2g, dif_pos h1]
    ring
  | succ c hc1 ih =>
    have hc' : c ≤ nB := by omega
    have ihc := ih hc'
    simp only [lPg, dif_pos hc]
    rw [ihc, alphaPg_succ score hBN i c hc1 hc,
        block_sum_shiftg score i c (by omega)]
    simp only [lFree2g, dif_pos hc]
    rw [show pow2 (mRg score i c - mRg score i (c+1))
            * (pow2 (-(mRg score i c)) * lFree2g score i c)
          = pow2 ((mRg score i c - mRg score i (c+1)) + (-(mRg score i c)))
              * lFree2g score i c by rw [pow2_add]; ring]
    rw [show (mRg score i c - mRg score i (c+1)) + (-(mRg score i c))
          = -(mRg score i (c+1)) by ring]
    ring

/-- `m`-free running numerator `Σ_{b<c} Σ_a pow2(score)·V[gkey, d]`. -/
noncomputable def oFree2g {Mq Dh : Nat} {BN nB : Nat}
    (score : Fin Mq → Fin (BN * nB) → ℝ) (V : TileIndex [BN * nB, Dh] → ℝ)
    (i : Fin Mq) (d : Fin Dh) : Nat → ℝ
  | 0 => 0
  | c + 1 =>
      if h : c + 1 ≤ nB then
        oFree2g score V i d c +
          Finset.univ.sum (fun a : Fin BN =>
            pow2 (score i (gkey BN nB c (by omega) a))
              * V (gkey BN nB c (by omega) a, d, PUnit.unit))
      else oFree2g score V i d c

/-- Running per-row unnormalized output `acc[i,d]` after `c` blocks. -/
noncomputable def oPg {Mq Dh : Nat} {BN nB : Nat}
    (score : Fin Mq → Fin (BN * nB) → ℝ) (V : TileIndex [BN * nB, Dh] → ℝ)
    (i : Fin Mq) (d : Fin Dh) : Nat → ℝ
  | 0 => 0
  | c + 1 =>
      if h : c + 1 ≤ nB then
        alphaPg score i c * oPg score V i d c +
          Finset.univ.sum (fun a : Fin BN =>
            pow2 (score i (gkey BN nB c (by omega) a) - mRg score i (c + 1))
              * V (gkey BN nB c (by omega) a, d, PUnit.unit))
      else oPg score V i d c

/-- Numerator analogue of `block_sum_shiftg`. -/
theorem block_sum_shiftg_v {Mq Dh : Nat} {BN nB : Nat}
    (score : Fin Mq → Fin (BN * nB) → ℝ) (V : TileIndex [BN * nB, Dh] → ℝ)
    (i : Fin Mq) (d : Fin Dh) (c : Nat) (hc : c < nB) (m : ℝ) :
    Finset.univ.sum (fun a : Fin BN =>
        pow2 (score i (gkey BN nB c (by omega) a) - m)
          * V (gkey BN nB c (by omega) a, d, PUnit.unit))
      = pow2 (-m) * Finset.univ.sum (fun a : Fin BN =>
          pow2 (score i (gkey BN nB c (by omega) a))
            * V (gkey BN nB c (by omega) a, d, PUnit.unit)) := by
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro a _
  rw [show score i (gkey BN nB c (by omega) a) - m
        = (-m) + score i (gkey BN nB c (by omega) a) by ring, pow2_add]
  ring

/-- **Numerator consistency** (the `acc` analogue of `lPg_eq`). -/
theorem oPg_eq {Mq Dh : Nat} {BN nB : Nat}
    (score : Fin Mq → Fin (BN * nB) → ℝ) (V : TileIndex [BN * nB, Dh] → ℝ)
    (hBN : 0 < BN) (i : Fin Mq) (d : Fin Dh) :
    ∀ c, 1 ≤ c → c ≤ nB →
      oPg score V i d c = pow2 (-(mRg score i c)) * oFree2g score V i d c := by
  intro c hc1 hc
  induction c, hc1 using Nat.le_induction with
  | base =>
    have h1 : (1 : Nat) ≤ nB := hc
    simp only [oPg, dif_pos h1, alphaPg_zero score i, zero_mul, zero_add]
    rw [block_sum_shiftg_v score V i d 0 (by omega)]
    simp only [oFree2g, dif_pos h1]
    ring
  | succ c hc1 ih =>
    have hc' : c ≤ nB := by omega
    have ihc := ih hc'
    simp only [oPg, dif_pos hc]
    rw [ihc, alphaPg_succ score hBN i c hc1 hc,
        block_sum_shiftg_v score V i d c (by omega)]
    simp only [oFree2g, dif_pos hc]
    rw [show pow2 (mRg score i c - mRg score i (c+1))
            * (pow2 (-(mRg score i c)) * oFree2g score V i d c)
          = pow2 ((mRg score i c - mRg score i (c+1)) + (-(mRg score i c)))
              * oFree2g score V i d c by rw [pow2_add]; ring]
    rw [show (mRg score i c - mRg score i (c+1)) + (-(mRg score i c))
          = -(mRg score i (c+1)) by ring]
    ring

/-- **Ratio invariance.** The running max shift cancels in `acc / l_i`
(`pow2` is never zero), so the kernel's normalized output after `c ≥ 1` blocks
equals the max-free `oFree2g / lFree2g`. -/
theorem ratioPg {Mq Dh : Nat} {BN nB : Nat}
    (score : Fin Mq → Fin (BN * nB) → ℝ) (V : TileIndex [BN * nB, Dh] → ℝ)
    (hBN : 0 < BN) (i : Fin Mq) (d : Fin Dh)
    (c : Nat) (hc1 : 1 ≤ c) (hc : c ≤ nB) :
    oPg score V i d c / lPg score i c
      = oFree2g score V i d c / lFree2g score i c := by
  rw [oPg_eq score V hBN i d c hc1 hc, lPg_eq score hBN i c hc1 hc,
      mul_div_mul_left _ _ (ne_of_gt (pow2_pos _))]

theorem lFree2g_range {Mq : Nat} {BN nB : Nat}
    (score : Fin Mq → Fin (BN * nB) → ℝ) (i : Fin Mq) : ∀ c,
    lFree2g score i c
      = Finset.sum (Finset.range c) (fun b =>
          if h : b < nB then
            Finset.univ.sum (fun a : Fin BN => pow2 (score i (gkey BN nB b h a)))
          else 0) := by
  intro c
  induction c with
  | zero => simp [lFree2g]
  | succ c ih =>
    rw [Finset.sum_range_succ, ← ih]
    by_cases h : c + 1 ≤ nB
    · simp only [lFree2g, dif_pos h, dif_pos (show c < nB by omega)]
    · simp only [lFree2g, dif_neg h, dif_neg (show ¬ c < nB by omega), add_zero]

/-- Block-grid → flat reindex of the generalized denominator: after all `nB`
blocks, `lFree2g` is the flat per-key sum `Σ_j pow2(score i j)`. -/
theorem lFree2g_flat {Mq : Nat} {BN nB : Nat}
    (score : Fin Mq → Fin (BN * nB) → ℝ) (i : Fin Mq) :
    lFree2g score i nB
      = Finset.univ.sum (fun j : Fin (BN * nB) => pow2 (score i j)) := by
  rw [lFree2g_range score i nB,
      ← Fin.sum_univ_eq_sum_range (fun b => if h : b < nB then
            Finset.univ.sum (fun a : Fin BN => pow2 (score i (gkey BN nB b h a))) else 0) nB,
      sum_fin_eq_block_grid BN nB (fun j => pow2 (score i j))]
  apply Finset.sum_congr rfl; intro b _
  rw [dif_pos b.isLt]; apply Finset.sum_congr rfl; intro a _; rw [← gkey_eq_cast]

theorem oFree2g_range {Mq Dh : Nat} {BN nB : Nat}
    (score : Fin Mq → Fin (BN * nB) → ℝ) (V : TileIndex [BN * nB, Dh] → ℝ)
    (i : Fin Mq) (d : Fin Dh) : ∀ c,
    oFree2g score V i d c
      = Finset.sum (Finset.range c) (fun b =>
          if h : b < nB then
            Finset.univ.sum (fun a : Fin BN =>
              pow2 (score i (gkey BN nB b h a)) * V (gkey BN nB b h a, d, PUnit.unit))
          else 0) := by
  intro c
  induction c with
  | zero => simp [oFree2g]
  | succ c ih =>
    rw [Finset.sum_range_succ, ← ih]
    by_cases h : c + 1 ≤ nB
    · simp only [oFree2g, dif_pos h, dif_pos (show c < nB by omega)]
    · simp only [oFree2g, dif_neg h, dif_neg (show ¬ c < nB by omega), add_zero]

/-- Block-grid → flat reindex of the generalized numerator. -/
theorem oFree2g_flat {Mq Dh : Nat} {BN nB : Nat}
    (score : Fin Mq → Fin (BN * nB) → ℝ) (V : TileIndex [BN * nB, Dh] → ℝ)
    (i : Fin Mq) (d : Fin Dh) :
    oFree2g score V i d nB
      = Finset.univ.sum (fun j : Fin (BN * nB) =>
          pow2 (score i j) * V (j, d, PUnit.unit)) := by
  rw [oFree2g_range score V i d nB,
      ← Fin.sum_univ_eq_sum_range (fun b => if h : b < nB then
            Finset.univ.sum (fun a : Fin BN =>
              pow2 (score i (gkey BN nB b h a)) * V (gkey BN nB b h a, d, PUnit.unit))
          else 0) nB,
      sum_fin_eq_block_grid BN nB
        (fun j => pow2 (score i j) * V (j, d, PUnit.unit))]
  apply Finset.sum_congr rfl; intro b _
  rw [dif_pos b.isLt]; apply Finset.sum_congr rfl; intro a _; rw [← gkey_eq_cast]

/-- Generic attention spec for an arbitrary per-key score function: the base-2
softmax `(Σ_j 2^(score i j)·V[j,d]) / (Σ_j 2^(score i j))`. -/
noncomputable def attnGenScore {Mq Dh : Nat} {S : Nat}
    (fscore : Fin Mq → Fin S → ℝ) (V : TileIndex [S, Dh] → ℝ) :
    TileIndex [Mq, Dh] → ℝ :=
  fun (i, d, _) =>
    let denom := Finset.univ.sum (fun j : Fin S => pow2 (fscore i j))
    let numer := Finset.univ.sum (fun j : Fin S => pow2 (fscore i j) * V (j, d, PUnit.unit))
    numer / denom

/-- **The math heart of the generalized closed-form.** The streamed running
ratio `oPg / lPg` after all `nB` blocks equals the generic batch base-2 softmax
`attnGenScore`. Combines `ratioPg` with the block-grid → flat reindex. -/
theorem closed_form_g {Mq Dh : Nat} {BN nB : Nat}
    (score : Fin Mq → Fin (BN * nB) → ℝ) (V : TileIndex [BN * nB, Dh] → ℝ)
    (hBN : 0 < BN) (hnB : 1 ≤ nB) (i : Fin Mq) (d : Fin Dh) :
    oPg score V i d nB / lPg score i nB
      = attnGenScore score V (i, d, PUnit.unit) := by
  rw [ratioPg score V hBN i d nB hnB le_rfl, oFree2g_flat, lFree2g_flat]
  rfl

/-- `attnGenScore` is exactly the all-keys `osStep` streaming ratio (the per-key
fold from `osStep_foldl_eq_batch`, which is already generic over scores). -/
theorem attnGenScore_eq_streaming {Mq Dh : Nat} {S : Nat}
    (fscore : Fin Mq → Fin S → ℝ) (V : TileIndex [S, Dh] → ℝ) (i : Fin Mq) (d : Fin Dh) :
    attnGenScore fscore V (i, d, PUnit.unit)
      = (let st := (List.ofFn (fun j : Fin S => (fscore i j, V (j, d, PUnit.unit)))).foldl
            osStep (0, 0, 0)
         st.2.2 / st.2.1) := by
  rw [osStep_foldl_eq_batch]
  simp only [attnGenScore, List.map_ofFn, List.sum_ofFn, Function.comp]

/-- **Specialization bridge.** The existing per-block key-scale spec
`attentionRealBase2PerKeyScale` is the special case of the generic `attnGenScore`
with `fscore i j := keyScale j · (Σ_e Q[i,e]·K[j,e])` (the per-key-scaled unscaled
score). So the generic recurrence/closed-form lemmas above subsume the
hardcoded-`kscore` ones. -/
theorem attentionRealBase2PerKeyScale_eq_attnGenScore {M S D : Nat}
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (keyScale : Fin S → ℝ) (i : Fin M) (d : Fin D) :
    attentionRealBase2PerKeyScale Q K V keyScale (i, d, PUnit.unit)
      = attnGenScore
          (fun (i : Fin M) (j : Fin S) =>
            keyScale j * Finset.univ.sum (fun e : Fin D =>
              Q (i, e, PUnit.unit) * K (j, e, PUnit.unit))) V (i, d, PUnit.unit) := by
  simp only [attentionRealBase2PerKeyScale, attnGenScore, pow2]

/-- Causal attention for one 2D slice. A key position `j` contributes to
query position `i` exactly when `j ≤ i`; future keys contribute zero
softmax mass. -/
noncomputable def attentionRealCausal {M S D : Nat}
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (scale : ℝ) : TileIndex [M, D] → ℝ :=
  fun (i, d, _) =>
    let score := fun j : Fin S => scaledScore Q K scale i j
    let weight := fun j : Fin S =>
      if j.val ≤ i.val then Real.exp (score j) else 0
    let denom := Finset.univ.sum (fun j : Fin S => weight j)
    let numer := Finset.univ.sum (fun j : Fin S =>
      weight j * V (j, d, PUnit.unit))
    numer / denom

/-- Causal attention for a local Q block whose row `i` corresponds to
global query row `qStart + i`. -/
noncomputable def attentionRealCausalBlock {M S D : Nat}
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (scale : ℝ) : TileIndex [M, D] → ℝ :=
  fun (i, d, _) =>
    let score := fun j : Fin S => scaledScore Q K scale i j
    let weight := fun j : Fin S =>
      if j.val ≤ qStart + i.val then Real.exp (score j) else 0
    let denom := Finset.univ.sum (fun j : Fin S => weight j)
    let numer := Finset.univ.sum (fun j : Fin S =>
      weight j * V (j, d, PUnit.unit))
    numer / denom

/-! ## 4D layout-independent views -/

/-- Slice a 4D `[B, H, S, D]` tile-as-function at fixed `(batch, head)`,
yielding a 2D `[S, D]` tile-as-function. -/
def sliceBH {B H S D : Nat}
    (T : TileIndex [B, H, S, D] → ℝ)
    (b : Fin B) (h : Fin H) : TileIndex [S, D] → ℝ :=
  fun (i, d, _) => T (b, h, i, d, PUnit.unit)

/-- 4D ℝ-valued reference attention. Each `(batch, head)` slice is
independent and computes ordinary 2D `attentionReal`. -/
noncomputable def attentionReal4D {B H S_q S_k D : Nat}
    (Q : TileIndex [B, H, S_q, D] → ℝ)
    (K V : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) : TileIndex [B, H, S_q, D] → ℝ :=
  fun (b, h, i, d, _) =>
    attentionReal (sliceBH Q b h) (sliceBH K b h) (sliceBH V b h)
      scale (i, d, PUnit.unit)

/-- 4D ℝ-valued causal reference attention. -/
noncomputable def attentionReal4DCausal {B H S_q S_k D : Nat}
    (Q : TileIndex [B, H, S_q, D] → ℝ)
    (K V : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) : TileIndex [B, H, S_q, D] → ℝ :=
  fun (b, h, i, d, _) =>
    attentionRealCausal (sliceBH Q b h) (sliceBH K b h) (sliceBH V b h)
      scale (i, d, PUnit.unit)

@[simp] theorem attentionReal4D_slice {B H S_q S_k D : Nat}
    (Q : TileIndex [B, H, S_q, D] → ℝ)
    (K V : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) (b : Fin B) (h : Fin H) (i : Fin S_q) (d : Fin D) :
    attentionReal4D Q K V scale (b, h, i, d, PUnit.unit)
      = attentionReal (sliceBH Q b h) (sliceBH K b h) (sliceBH V b h)
          scale (i, d, PUnit.unit) := rfl

@[simp] theorem attentionReal4DCausal_slice {B H S_q S_k D : Nat}
    (Q : TileIndex [B, H, S_q, D] → ℝ)
    (K V : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) (b : Fin B) (h : Fin H) (i : Fin S_q) (d : Fin D) :
    attentionReal4DCausal Q K V scale (b, h, i, d, PUnit.unit)
      = attentionRealCausal (sliceBH Q b h) (sliceBH K b h) (sliceBH V b h)
          scale (i, d, PUnit.unit) := rfl

/-- M-row slice of the `(b, h)` plane of a 4D tensor. -/
def slice4DQRows {B H S D : Nat} (M : Nat)
    (T : TileIndex [B, H, S, D] → ℝ) (b : Fin B) (h : Fin H)
    (start : Nat) (hBnd : start * M + M ≤ S) : TileIndex [M, D] → ℝ :=
  fun (i, d, _) =>
    T (b, h, ⟨start * M + i.val, by
      have := i.isLt
      omega⟩, d, PUnit.unit)

/-- Boundary-masked M-row Q block. In-bounds rows read the logical Q tensor;
out-of-bounds rows are the `other=0` value supplied to `tl.load`. -/
def slice4DQRowsBoundary {B H S D : Nat} (M : Nat)
    (T : TileIndex [B, H, S, D] → ℝ) (b : Fin B) (h : Fin H)
    (start : Nat) : TileIndex [M, D] → ℝ :=
  fun (i, d, _) =>
    if hIn : start * M + i.val < S then
      T (b, h, ⟨start * M + i.val, hIn⟩, d, PUnit.unit)
    else
      0

@[simp] theorem slice4DQRowsBoundary_of_lt {B H S D : Nat} (M : Nat)
    (T : TileIndex [B, H, S, D] → ℝ) (b : Fin B) (h : Fin H)
    (start : Nat) (i : Fin M) (d : Fin D)
    (hIn : start * M + i.val < S) :
    slice4DQRowsBoundary M T b h start (i, d, PUnit.unit) =
      T (b, h, ⟨start * M + i.val, hIn⟩, d, PUnit.unit) := by
  simp [slice4DQRowsBoundary, hIn]

@[simp] theorem slice4DQRowsBoundary_of_not_lt {B H S D : Nat} (M : Nat)
    (T : TileIndex [B, H, S, D] → ℝ) (b : Fin B) (h : Fin H)
    (start : Nat) (i : Fin M) (d : Fin D)
    (hOut : ¬ start * M + i.val < S) :
    slice4DQRowsBoundary M T b h start (i, d, PUnit.unit) = 0 := by
  simp [slice4DQRowsBoundary, hOut]

/-- Pad a logical hidden dimension `D` to a block hidden dimension `Bd`.
Out-of-range hidden lanes are zero. -/
def padHeadD {S D Bd : Nat} (X : TileIndex [S, D] → ℝ) :
    TileIndex [S, Bd] → ℝ :=
  fun (i, d, _) =>
    if h : d.val < D then
      X (i, ⟨d.val, h⟩, PUnit.unit)
    else
      0

@[simp] theorem padHeadD_of_lt {S D Bd : Nat}
    (X : TileIndex [S, D] → ℝ) (i : Fin S) (d : Fin Bd)
    (h : d.val < D) :
    padHeadD X (i, d, PUnit.unit) = X (i, ⟨d.val, h⟩, PUnit.unit) := by
  simp [padHeadD, h]

@[simp] theorem padHeadD_of_not_lt {S D Bd : Nat}
    (X : TileIndex [S, D] → ℝ) (i : Fin S) (d : Fin Bd)
    (h : ¬ d.val < D) :
    padHeadD X (i, d, PUnit.unit) = 0 := by
  simp [padHeadD, h]

/-- A zero-padded hidden-dimension sum over `Fin Bd` reduces to the logical
sum over `Fin D` when the logical dimension is covered by the block width. -/
theorem sum_padHeadD_eq {D Bd : Nat} (hDLe : D ≤ Bd) (f : Fin D → ℝ) :
    (Finset.univ : Finset (Fin Bd)).sum (fun d : Fin Bd =>
      if h : d.val < D then f ⟨d.val, h⟩ else 0)
      = (Finset.univ : Finset (Fin D)).sum f := by
  rw [show (Finset.univ.sum (fun d : Fin Bd =>
        if h : d.val < D then f ⟨d.val, h⟩ else 0)) =
        ((Finset.univ : Finset (Fin Bd)).filter (fun d => d.val < D)).sum
          (fun d : Fin Bd => if h : d.val < D then f ⟨d.val, h⟩ else 0)
        from ?_]
  · refine (Finset.sum_bij (fun (d : Fin D) (_ : d ∈ Finset.univ) =>
      Fin.castLE hDLe d) ?_ ?_ ?_ ?_).symm
    · intro d _
      simp [Fin.val_castLE, d.isLt]
    · intro d₁ _ d₂ _ heq
      apply Fin.ext
      have := congrArg Fin.val heq
      simpa [Fin.val_castLE] using this
    · intro d hd
      simp at hd
      refine ⟨⟨d.val, hd⟩, Finset.mem_univ _, ?_⟩
      apply Fin.ext
      simp
    · intro d _
      have hLt : (Fin.castLE hDLe d).val < D := by
        rw [Fin.val_castLE]; exact d.isLt
      rw [dif_pos hLt]
      congr 1
  · refine (Finset.sum_filter_of_ne ?_).symm
    intro d _ hNe
    by_contra hLt
    apply hNe
    rw [dif_neg hLt]

theorem scaledScore_padHeadD_eq {M S D Bd : Nat} (hDLe : D ≤ Bd)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S, D] → ℝ)
    (scale : ℝ) (i : Fin M) (j : Fin S) :
    scaledScore (D := Bd) (padHeadD (Bd := Bd) Q) (padHeadD (Bd := Bd) K) scale i j =
      scaledScore Q K scale i j := by
  unfold scaledScore
  congr 1
  rw [show (Finset.univ : Finset (Fin Bd)).sum (fun d : Fin Bd =>
        padHeadD Q (i, d, PUnit.unit) * padHeadD K (j, d, PUnit.unit)) =
      (Finset.univ : Finset (Fin Bd)).sum (fun d : Fin Bd =>
        if h : d.val < D then
          Q (i, ⟨d.val, h⟩, PUnit.unit) * K (j, ⟨d.val, h⟩, PUnit.unit)
        else
          0) by
    apply Finset.sum_congr rfl
    intro d _
    by_cases h : d.val < D
    · simp [padHeadD, h]
    · simp [padHeadD, h]]
  rw [sum_padHeadD_eq hDLe (fun d : Fin D =>
    Q (i, d, PUnit.unit) * K (j, d, PUnit.unit))]

/-- Causal block attention is invariant under zero-padding the hidden
dimension, on logical output lanes `d < D`. -/
theorem attentionRealCausalBlock_padHeadD_eq {M S D Bd : Nat} (hDLe : D ≤ Bd)
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (scale : ℝ) (idx : TileIndex [M, Bd]) (hDIdx : idx.2.1.val < D) :
    attentionRealCausalBlock qStart
        (padHeadD (Bd := Bd) Q) (padHeadD (Bd := Bd) K) (padHeadD (Bd := Bd) V)
        scale idx =
    attentionRealCausalBlock qStart Q K V scale
        (idx.1, ⟨idx.2.1.val, hDIdx⟩, PUnit.unit) := by
  obtain ⟨i, d, u⟩ := idx
  cases u
  have hD : d.val < D := hDIdx
  unfold attentionRealCausalBlock
  have hV : ∀ j : Fin S,
      padHeadD V (j, d, PUnit.unit) = V (j, ⟨d.val, hD⟩, PUnit.unit) := by
    intro j
    simp [padHeadD, hD]
  have hScoreT : ∀ j : Fin S,
      scaledScore (padHeadD (Bd := Bd) Q)
          (padHeadD (Bd := Bd) K) scale i j =
        scaledScore Q K scale i j := by
    intro j
    simpa using scaledScore_padHeadD_eq hDLe Q K scale i j
  simp [hScoreT, hV]

/-- Causal attention is invariant under zero-padding the hidden dimension,
on logical output lanes `d < D`. -/
theorem attentionRealCausal_padHeadD_eq {M S D Bd : Nat} (hDLe : D ≤ Bd)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (scale : ℝ) (idx : TileIndex [M, Bd]) (hDIdx : idx.2.1.val < D) :
    attentionRealCausal
        (padHeadD (Bd := Bd) Q) (padHeadD (Bd := Bd) K) (padHeadD (Bd := Bd) V)
        scale idx =
    attentionRealCausal Q K V scale
        (idx.1, ⟨idx.2.1.val, hDIdx⟩, PUnit.unit) := by
  obtain ⟨i, d, u⟩ := idx
  cases u
  have hD : d.val < D := hDIdx
  unfold attentionRealCausal
  have hV : ∀ j : Fin S,
      padHeadD V (j, d, PUnit.unit) = V (j, ⟨d.val, hD⟩, PUnit.unit) := by
    intro j
    simp [padHeadD, hD]
  have hScoreT : ∀ j : Fin S,
      scaledScore (padHeadD (Bd := Bd) Q)
          (padHeadD (Bd := Bd) K) scale i j =
        scaledScore Q K scale i j := by
    intro j
    simpa using scaledScore_padHeadD_eq hDLe Q K scale i j
  simp [hScoreT, hV]

/-- Non-causal attention is invariant under zero-padding the hidden
dimension, on logical output lanes `d < D`. -/
theorem attentionReal_padHeadD_eq {M S D Bd : Nat} (hDLe : D ≤ Bd)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (scale : ℝ) (idx : TileIndex [M, Bd]) (hDIdx : idx.2.1.val < D) :
    attentionReal
        (padHeadD (Bd := Bd) Q) (padHeadD (Bd := Bd) K) (padHeadD (Bd := Bd) V)
        scale idx =
      attentionReal Q K V scale
        (idx.1, ⟨idx.2.1.val, hDIdx⟩, PUnit.unit) := by
  obtain ⟨i, d, u⟩ := idx
  cases u
  have hD : d.val < D := hDIdx
  have hV : ∀ j : Fin S,
      padHeadD V (j, d, PUnit.unit) = V (j, ⟨d.val, hD⟩, PUnit.unit) := by
    intro j
    simp [padHeadD, hD]
  have hScoreT : ∀ j : Fin S,
      scaledScore (padHeadD (Bd := Bd) Q)
          (padHeadD (Bd := Bd) K) scale i j =
        scaledScore Q K scale i j := by
    intro j
    simpa using scaledScore_padHeadD_eq hDLe Q K scale i j
  unfold attentionReal
  simp [hScoreT, hV]

/-- Reinterpret the `(b, h)` plane of a 4D `[B, H, S, D]` tensor as a
flat `[Bk * numKVBlocks, D]` view, given `Bk * numKVBlocks = S`. -/
def slice4DFlat {B H S D : Nat} (Bk numKVBlocks : Nat)
    (T : TileIndex [B, H, S, D] → ℝ) (b : Fin B) (h : Fin H)
    (hSk : Bk * numKVBlocks = S) : TileIndex [Bk * numKVBlocks, D] → ℝ :=
  fun (j, d, _) =>
    T (b, h, ⟨j.val, by
      have := j.isLt
      omega⟩, d, PUnit.unit)

/-! ### Base-2 attention with a scalar score scale and an additive bias

Matches relative-position-bias flash kernels (`attention_kernel_aligned`) that
(a) use `exp2` for the softmax, (b) fold a *scalar* score scale `qk_scale =
sm_scale · log2(e)` into `q` (so every key shares the same scale), and (c) add a
per-`(query, key)` bias `bias i j` to the score *after* scaling — here
`bias i j = b0 + b1` of the fused `rel_h + rel_w` position table. For query `i`,
output channel `d`:

`out[i,d] = (Σ_j 2^(scale · raw i j + bias i j) · V[j,d])
              / (Σ_j 2^(scale · raw i j + bias i j))`,

where `raw i j = Σ_e Q[i,e] · K[j,e]`. Using `2^x = exp(log 2 · x)`, this is the
base-`e` softmax of `log 2 · (scale · raw + bias)`. With `bias ≡ 0` it
specialises to `attentionRealBase2PerKeyScale` with the constant key scale
`scale`. -/
noncomputable def attentionRealBase2ScalarScaleBias {M S D : Nat}
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (scale : ℝ) (bias : Fin M → Fin S → ℝ) : TileIndex [M, D] → ℝ :=
  fun (i, d, _) =>
    let raw := fun j : Fin S =>
      Finset.univ.sum (fun e : Fin D => Q (i, e, PUnit.unit) * K (j, e, PUnit.unit))
    let weight := fun j : Fin S => Real.exp (Real.log 2 * (scale * raw j + bias i j))
    let denom := Finset.univ.sum (fun j : Fin S => weight j)
    let numer := Finset.univ.sum (fun j : Fin S => weight j * V (j, d, PUnit.unit))
    numer / denom

/-- The streamed `(score, value)` list the aligned kernel folds over: score
`scale · raw i j + bias i j` (the `qk` register after the dot, scale-folded into
`q`, and the `b0 + b1` bias add), value `V[j,d]`. -/
noncomputable def attnKeyListBias {M S D : Nat}
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (scale : ℝ) (bias : Fin M → Fin S → ℝ) (i : Fin M) (d : Fin D) :
    List (ℝ × ℝ) :=
  List.ofFn (fun j : Fin S =>
    (scale * Finset.univ.sum (fun e : Fin D =>
        Q (i, e, PUnit.unit) * K (j, e, PUnit.unit)) + bias i j,
      V (j, d, PUnit.unit)))

/-- **Bridge: the base-2 scalar-scale + additive-bias attention spec IS the
online-softmax fold.** Mirrors `attentionRealBase2PerKeyScale_eq_streaming`: the
closed form equals the `acc / l` ratio produced by streaming `osStep` over
`attnKeyListBias` from the neutral start. The remaining `exec` proof only has to
show the kernel's loop realizes this fold; this lemma then delivers the closed
form. -/
theorem attentionRealBase2ScalarScaleBias_eq_streaming {M S D : Nat}
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (scale : ℝ) (bias : Fin M → Fin S → ℝ) (i : Fin M) (d : Fin D) :
    attentionRealBase2ScalarScaleBias Q K V scale bias (i, d, PUnit.unit)
      = (let st := (attnKeyListBias Q K V scale bias i d).foldl osStep (0, 0, 0)
         st.2.2 / st.2.1) := by
  rw [osStep_foldl_eq_batch]
  simp only [attentionRealBase2ScalarScaleBias, attnKeyListBias, List.map_ofFn,
    List.sum_ofFn, Function.comp]
  rfl

end VeriTile.Triton
