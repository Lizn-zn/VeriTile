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

/-! ### Causal base-2 per-key-scale attention

Streaming flash kernels with a causal mask (e.g. `flash_attn.py`'s
`tl.where(off_m[:,None] >= start_n+off_n[None,:], qk, -inf)`) mask each *future*
key's score to `-inf`, so its softmax weight `2^(score)` is `0`. The genuine
closed form is therefore the base-2 per-key-scale softmax restricted to keys
`j ≤ qStart + i` (global query row `qStart + i`). This section gives that closed
form and its bridge to the same `osStep` online-softmax fold over a *causally
filtered* key list — the exec-side loop only has to realize this fold. -/

/-- Causal base-2 per-key-scale attention: key `j` contributes to output row `i`
exactly when its global index `j ≤ qStart + i.val`; future keys carry zero
softmax mass (the `-inf`-masked score). -/
noncomputable def attentionRealBase2PerKeyScaleCausal {M S D : Nat}
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (keyScale : Fin S → ℝ) (qStart : Nat) : TileIndex [M, D] → ℝ :=
  fun (i, d, _) =>
    let raw := fun j : Fin S =>
      Finset.univ.sum (fun e : Fin D => Q (i, e, PUnit.unit) * K (j, e, PUnit.unit))
    let weight := fun j : Fin S =>
      if j.val ≤ qStart + i.val then pow2 (keyScale j * raw j) else 0
    let denom := Finset.univ.sum (fun j : Fin S => weight j)
    let numer := Finset.univ.sum (fun j : Fin S => weight j * V (j, d, PUnit.unit))
    numer / denom

/-- Causally-filtered streamed key list for output `(i, d)`: only keys with
global index `≤ qStart + i.val` are emitted (future keys are dropped, exactly the
`-inf`-masked / zero-weight keys), each as a `(score, value)` pair with score
`keyScale j · (Q row i · K row j)`. -/
noncomputable def attnKeyListCausal {M S D : Nat}
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (keyScale : Fin S → ℝ) (qStart : Nat) (i : Fin M) (d : Fin D) : List (ℝ × ℝ) :=
  (List.finRange S).filterMap (fun j : Fin S =>
    if j.val ≤ qStart + i.val then
      some (keyScale j * Finset.univ.sum (fun e : Fin D =>
              Q (i, e, PUnit.unit) * K (j, e, PUnit.unit)),
            V (j, d, PUnit.unit))
    else none)

/-- Sum of a `filterMap (if p then some (g·) else none)` collapses the guard into
the summand. -/
private theorem list_sum_filterMap_ite {α β : Type*} [AddCommMonoid β]
    (l : List α) (p : α → Prop) [DecidablePred p] (g : α → β) :
    ((l.filterMap (fun a => if p a then some (g a) else none))).sum
      = (l.map (fun a => if p a then g a else 0)).sum := by
  induction l with
  | nil => simp
  | cons a t ih => by_cases ha : p a <;> simp [ha, ih]

/-- `h`-image sum of a causally filtered `finRange` list = the masked
`Finset.sum` over `Fin n`. The list/Finset bridge for the causal numerator and
denominator. -/
private theorem list_sum_filterMap_finRange {α : Type*} (n : Nat)
    (p : Fin n → Prop) [DecidablePred p] (g : Fin n → α) (h : α → ℝ) :
    (((List.finRange n).filterMap (fun j => if p j then some (g j) else none)).map h).sum
      = ∑ j : Fin n, if p j then h (g j) else 0 := by
  rw [List.map_filterMap]
  rw [show (fun j : Fin n => Option.map h (if p j then some (g j) else none))
        = (fun j : Fin n => if p j then some (h (g j)) else none) from by
    funext j; by_cases hj : p j <;> simp [hj]]
  rw [list_sum_filterMap_ite (List.finRange n) p (fun j => h (g j))]
  rw [← List.sum_ofFn]; congr 1; rw [List.ofFn_eq_map]

/-- **Bridge: causal base-2 per-key-scale attention IS the online-softmax fold
over the causally filtered key list.** The eventual `exec` proof only has to show
the kernel's masked loop realizes this fold; this lemma then delivers the causal
closed form (`attentionRealBase2PerKeyScaleCausal`). -/
theorem attentionRealBase2PerKeyScaleCausal_eq_streaming {M S D : Nat}
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (keyScale : Fin S → ℝ) (qStart : Nat) (i : Fin M) (d : Fin D) :
    attentionRealBase2PerKeyScaleCausal Q K V keyScale qStart (i, d, PUnit.unit)
      = (let st := (attnKeyListCausal Q K V keyScale qStart i d).foldl osStep (0, 0, 0)
         st.2.2 / st.2.1) := by
  rw [osStep_foldl_eq_batch]
  simp only [attentionRealBase2PerKeyScaleCausal, attnKeyListCausal]
  congr 1
  · rw [list_sum_filterMap_finRange S (fun j => j.val ≤ qStart + i.val)
      (fun j => (keyScale j * Finset.univ.sum (fun e : Fin D =>
          Q (i, e, PUnit.unit) * K (j, e, PUnit.unit)), V (j, d, PUnit.unit)))
      (fun p => pow2 p.1 * p.2)]
    refine Finset.sum_congr rfl (fun j _ => ?_)
    by_cases hj : j.val ≤ qStart + i.val <;> simp [hj]
  · rw [list_sum_filterMap_finRange S (fun j => j.val ≤ qStart + i.val)
      (fun j => (keyScale j * Finset.univ.sum (fun e : Fin D =>
          Q (i, e, PUnit.unit) * K (j, e, PUnit.unit)), V (j, d, PUnit.unit)))
      (fun p => pow2 p.1)]

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

/-- The running max after folding `osStep` over a list is `blockMax`. -/
theorem osStep_foldl_fst (xs : List (ℝ × ℝ)) (m l acc : ℝ) :
    (xs.foldl osStep (m, l, acc)).1 = blockMax m xs := by
  induction xs generalizing m l acc with
  | nil => simp [blockMax]
  | cons x xs ih =>
    obtain ⟨s, v⟩ := x
    simp only [List.foldl_cons, osStep, blockMax, List.foldl_cons]
    exact ih (max m s) _ _

/-- **One block step = the per-key `osStep` fold over that block.** The kernel
takes the block max and rescales once per block; this equals streaming the block's
keys one at a time. Both keep `l = 2^(-m)·Σ2^s`, `acc = 2^(-m)·Σ2^s·v` and end at
the same running max `blockMax`. -/
theorem osBlockStep_eq_foldl_osStep (st : ℝ × ℝ × ℝ) (block : List (ℝ × ℝ)) :
    osBlockStep st block = block.foldl osStep st := by
  obtain ⟨m, l, acc⟩ := st
  set m' := blockMax m block with hm'
  have hfst : (block.foldl osStep (m, l, acc)).1 = m' := osStep_foldl_fst block m l acc
  -- both consistency facts at the same anchor L = 2^m·l, T = 2^m·acc
  have hlL : l = pow2 (-m) * (pow2 m * l) := by
    rw [← mul_assoc, ← pow2_add]; ring_nf; simp
  have haccT : acc = pow2 (-m) * (pow2 m * acc) := by
    rw [← mul_assoc, ← pow2_add]; ring_nf; simp
  obtain ⟨hl1, ha1⟩ := osStep_foldl_consistent block m l acc (pow2 m * acc) (pow2 m * l) hlL haccT
  obtain ⟨hl2, ha2⟩ := osBlockStep_foldl_consistent [block] m l acc (pow2 m * acc) (pow2 m * l) hlL haccT
  simp only [List.foldl_cons, List.foldl_nil, List.flatten_cons, List.flatten_nil,
    List.append_nil] at hl2 ha2
  apply Prod.ext
  · -- max components
    show (osBlockStep (m, l, acc) block).1 = (block.foldl osStep (m, l, acc)).1
    rw [hfst]; rfl
  apply Prod.ext
  · -- l components
    show (osBlockStep (m, l, acc) block).2.1 = (block.foldl osStep (m, l, acc)).2.1
    rw [hl1, hl2]
    congr 1
    show pow2 (-(osBlockStep (m, l, acc) block).1) = pow2 (-(block.foldl osStep (m, l, acc)).1)
    rw [hfst]; rfl
  · -- acc components
    show (osBlockStep (m, l, acc) block).2.2 = (block.foldl osStep (m, l, acc)).2.2
    rw [ha1, ha2]
    congr 1
    show pow2 (-(osBlockStep (m, l, acc) block).1) = pow2 (-(block.foldl osStep (m, l, acc)).1)
    rw [hfst]; rfl

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

end VeriTile.Triton
