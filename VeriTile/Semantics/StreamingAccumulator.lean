/-
VeriTile.Semantics.StreamingAccumulator

Online-softmax streaming `(m, l, O)` accumulator recurrence and tile helper
lemmas for blockwise attention reductions.
-/

import VeriTile.Math.Attention
import VeriTile.Semantics.TileOps

namespace VeriTile
/-! ## Streaming math model (online softmax recurrence)

Three running quantities, each parameterized by the number of completed KV blocks
`k ∈ [0, numKVBlocks]`:

* `mPartial` — running per-row max of scaled scores. Uses `WithBot ℝ`
  (with seed `⊥` at `k = 0`) so the streaming `max` matches the
  kernel's `m_i := tl.full([M], -inf)` initialization exactly.
* `lPartial` — running per-row normalizer (sum of `exp(score - m)`).
  Real-valued (seed `0` at `k = 0`).
* `oPartial` — running unnormalized output (`Σ exp(score - m) · V`).
  Real-valued (seed `0` at `k = 0`).

The final theorem `streaming_eq_attentionReal` (math identity, no
kernel) proves that
`oPartial numKVBlocks idx / lPartial numKVBlocks i = attentionReal idx`,
which the kernel's post-loop `out := o_acc / l_i[:, None]` realizes. -/

namespace StreamingAccumulator

/-- Scaled score used by the streaming accumulator recurrence. This matches
`VeriTile.scaledScore` from `Math.Attention`, but stays under the
accumulator namespace for proof-local rewriting. -/
noncomputable def scaledScore {M S D : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S, D] → ℝ) (scale : ℝ)
    (i : Fin M) (j : Fin S) : ℝ :=
  scale * Finset.univ.sum (fun d : Fin D =>
    Q (i, d, PUnit.unit) * K (j, d, PUnit.unit))

/-- Row-wise softmax along the trailing axis on a `Tile .real`.
Local tile-level helper used only to connect `Tile.dot` data lemmas to the
pure real attention formulas. -/
noncomputable def softmaxRow {M N : Nat} (s : Tile .real [M, N]) :
    Tile .real [M, N] :=
  ⟨fun (m, n, _) =>
    let row := fun j : Fin N => (s.data (m, j, PUnit.unit)).unbotD 0
    let num := Real.exp (row n)
    let denom := Finset.univ.sum (fun j : Fin N => Real.exp (row j))
    some (num / denom)⟩

/-- Flat KV index of the `jLocal`-th lane of the `k`-th block. Factored
out as a non-dependent helper so `Finset.le_sup`-style lemmas can
unify on it without tripping over the inline proof. -/
def blockIndex (Bk numKVBlocks k : Nat) (h : k + 1 ≤ numKVBlocks)
    (jLocal : Fin Bk) : Fin (Bk * numKVBlocks) :=
  ⟨k * Bk + jLocal.val, by
    have hk : k < numKVBlocks := by omega
    calc k * Bk + jLocal.val
        < k * Bk + Bk := by omega
      _ = (k + 1) * Bk := by ring
      _ ≤ numKVBlocks * Bk := Nat.mul_le_mul_right Bk (by omega)
      _ = Bk * numKVBlocks := by ring⟩

/-- Running per-row max of scaled scores over the first `k` KV blocks.
Seeded at `⊥` so the `max` is right at `k = 0` (no scores seen).
Closely mirrors the standard online-softmax max recurrence. -/
noncomputable def mPartial {M D : Nat} (Bk : Nat)
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ) :
    Nat → Fin M → WithBot ℝ
  | 0,     _ => (⊥ : WithBot ℝ)
  | k + 1, i =>
      if h : k + 1 ≤ numKVBlocks then
        max (mPartial Bk Q numKVBlocks K scale k i)
          (Finset.univ.sup (fun jLocal : Fin Bk =>
            ((scaledScore Q K scale i (blockIndex Bk numKVBlocks k h jLocal)
              : ℝ) : WithBot ℝ)))
      else
        mPartial Bk Q numKVBlocks K scale k i

/-- The α multiplier `exp(m_k - m_{k+1})` produced by the online
softmax. At `k = 0` this collapses to `0` (since `m_0 = ⊥` and
`exp ⊥ = 0`), zeroing out the first iteration's contribution from the
empty seed. -/
noncomputable def alphaPartial {M D Bk : Nat}
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ)
    (k : Nat) (i : Fin M) : ℝ :=
  ((WithBot.realExp
      (WithBot.realSub
        (mPartial Bk Q numKVBlocks K scale k i)
      (mPartial Bk Q numKVBlocks K scale (k + 1) i)))).unbotD 0

/-- `alphaPartial` is exactly the real payload of the operational
`exp(m_i - m_new)` expression. The result is always a real `some`: when the
subtraction sees `⊥`, `WithBot.realExp ⊥ = 0`. -/
theorem alphaPartial_toWithBot {M D Bk : Nat}
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ)
    (k : Nat) (i : Fin M) :
    WithBot.realExp
      (WithBot.realSub
        (mPartial Bk Q numKVBlocks K scale k i)
        (mPartial Bk Q numKVBlocks K scale (k + 1) i))
      =
      ((alphaPartial Q numKVBlocks K scale k i : ℝ) : WithBot ℝ) := by
  unfold alphaPartial
  cases h :
      WithBot.realSub
        (mPartial Bk Q numKVBlocks K scale k i)
        (mPartial Bk Q numKVBlocks K scale (k + 1) i) with
  | bot =>
      simp
  | coe a =>
      simp

/-- Running per-row normalizer `Σ exp(score - m)` over the first `k`
KV blocks. -/
noncomputable def lPartial {M D Bk : Nat}
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ) :
    Nat → Fin M → ℝ
  | 0,     _ => 0
  | k + 1, i =>
      if h : k + 1 ≤ numKVBlocks then
        let mNew : ℝ := (mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0
        alphaPartial Q numKVBlocks K scale k i *
          lPartial Q numKVBlocks K scale k i +
          Finset.univ.sum (fun jLocal : Fin Bk =>
            Real.exp (scaledScore Q K scale i
                (blockIndex Bk numKVBlocks k h jLocal) - mNew))
      else
        lPartial Q numKVBlocks K scale k i

/-- Running unnormalized per-row output `Σ exp(score - m) · V` over
the first `k` KV blocks. -/
noncomputable def oPartial {M D Bk : Nat}
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ) :
    Nat → TileIndex [M, D] → ℝ
  | 0,     _ => 0
  | k + 1, idx =>
      let i := idx.1
      let d := idx.2.1
      if h : k + 1 ≤ numKVBlocks then
        let mNew : ℝ := (mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0
        alphaPartial Q numKVBlocks K scale k i *
          oPartial Q numKVBlocks K V scale k idx +
          Finset.univ.sum (fun jLocal : Fin Bk =>
            Real.exp (scaledScore Q K scale i
                (blockIndex Bk numKVBlocks k h jLocal) - mNew) *
              V (blockIndex Bk numKVBlocks k h jLocal, d, PUnit.unit))
      else
        oPartial Q numKVBlocks K V scale k idx

/-! ### Stage A — m-free reference sums (`lFree`, `oFree`)

Reference sums *without* the `m` shift, used to bridge the streaming
form (which subtracts `m_k` for numerical stability) to `attentionReal`
(which doesn't). The key identity:

```text
lPartial k i = exp(-m_k) · lFree k i        (m-shift factors out)
oPartial k i d = exp(-m_k) · oFree k i d
```

so the ratio `oPartial / lPartial = oFree / lFree`, which matches
`attentionReal` directly. -/

/-- Σ over the first `k` KV blocks of `exp(scaledScore)`, no `m`-shift. -/
noncomputable def lFree {M D Bk N : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [Bk * N, D] → ℝ) (scale : ℝ)
    (k : Nat) (hk : k ≤ N) (i : Fin M) : ℝ :=
  Finset.univ.sum (fun n : Fin k => Finset.univ.sum (fun jL : Fin Bk =>
    Real.exp (scaledScore Q K scale i
      (blockIndex Bk N n.val (Nat.lt_of_lt_of_le n.isLt hk) jL))))

/-- Σ over the first `k` KV blocks of `exp(scaledScore) · V`, no `m`-shift. -/
noncomputable def oFree {M D Bk N : Nat}
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [Bk * N, D] → ℝ) (scale : ℝ)
    (k : Nat) (hk : k ≤ N) (idx : TileIndex [M, D]) : ℝ :=
  let i := idx.1
  let d := idx.2.1
  Finset.univ.sum (fun n : Fin k => Finset.univ.sum (fun jL : Fin Bk =>
    Real.exp (scaledScore Q K scale i
      (blockIndex Bk N n.val (Nat.lt_of_lt_of_le n.isLt hk) jL)) *
    V (blockIndex Bk N n.val (Nat.lt_of_lt_of_le n.isLt hk) jL, d, PUnit.unit)))

/-- Recurrence: `lFree (k+1)` adds the next block's sum on top of `lFree k`.
Uses `Fin.sum_univ_castSucc` to peel the last index off the outer sum;
the remaining `Fin k` sum matches `lFree k` modulo proof irrelevance in
the `blockIndex` proof argument, and the peeled-off term reduces via
`Fin.val_last`. -/
theorem lFree_succ {M D Bk N : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [Bk * N, D] → ℝ) (scale : ℝ)
    (k : Nat) (hk : k + 1 ≤ N) (i : Fin M) :
    lFree Q K scale (k + 1) hk i =
      lFree Q K scale k (Nat.le_of_succ_le hk) i +
      Finset.univ.sum (fun jL : Fin Bk =>
        Real.exp (scaledScore Q K scale i (blockIndex Bk N k hk jL))) := by
  unfold lFree
  rw [Fin.sum_univ_castSucc]
  simp [Fin.val_last]

/-- Recurrence companion: `oFree (k+1)` adds the next block's `exp · V` sum
on top of `oFree k`. -/
theorem oFree_succ {M D Bk N : Nat}
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [Bk * N, D] → ℝ) (scale : ℝ)
    (k : Nat) (hk : k + 1 ≤ N) (idx : TileIndex [M, D]) :
    oFree Q K V scale (k + 1) hk idx =
      oFree Q K V scale k (Nat.le_of_succ_le hk) idx +
      Finset.univ.sum (fun jL : Fin Bk =>
        Real.exp (scaledScore Q K scale idx.1 (blockIndex Bk N k hk jL)) *
        V (blockIndex Bk N k hk jL, idx.2.1, PUnit.unit)) := by
  unfold oFree
  rw [Fin.sum_univ_castSucc]
  simp [Fin.val_last]

/-- `lFree 0 = 0` (empty sum). -/
@[simp] theorem lFree_zero {M D Bk N : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [Bk * N, D] → ℝ) (scale : ℝ)
    (i : Fin M) :
    lFree Q K scale 0 (Nat.zero_le _) i = 0 := by
  unfold lFree
  simp

/-- `oFree 0 = 0` (empty sum). -/
@[simp] theorem oFree_zero {M D Bk N : Nat}
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [Bk * N, D] → ℝ) (scale : ℝ)
    (idx : TileIndex [M, D]) :
    oFree Q K V scale 0 (Nat.zero_le _) idx = 0 := by
  unfold oFree
  simp

/-! ### Stage A foundations — `mPartial` non-`⊥` for `k ≥ 1`

The streaming algebra (α-cancellation in `lPartial` / `oPartial`)
needs `mPartial k` to be a real number, not `⊥`, whenever any block
has been seen. With `0 < Bk`, the block-max `Finset.univ.sup` over
`Fin Bk` is non-`⊥`, so `mPartial 1` is non-`⊥` and the property
propagates upward through `max`. -/

theorem mPartial_succ_ne_bot {M D : Nat} {Bk : Nat} (hBk : 0 < Bk)
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ)
    (k : Nat) (hk : k + 1 ≤ numKVBlocks) (i : Fin M) :
    mPartial Bk Q numKVBlocks K scale (k + 1) i ≠ (⊥ : WithBot ℝ) := by
  unfold mPartial
  simp only [hk, ↓reduceDIte]
  -- Result is `max prev (Finset.sup f)` where `f j = some (scaledScore _)`.
  -- Pick j₀ = ⟨0, hBk⟩; sup ≥ f j₀ = some _, so sup ≠ ⊥, so max ≠ ⊥.
  set f : Fin Bk → WithBot ℝ := fun jLocal =>
    ((scaledScore Q K scale i (blockIndex Bk numKVBlocks k hk jLocal) : ℝ)
      : WithBot ℝ)
  have hSup : f ⟨0, hBk⟩ ≤ Finset.univ.sup f :=
    Finset.le_sup (Finset.mem_univ _)
  intro hMaxBot
  have hSupBot : Finset.univ.sup f ≤ (⊥ : WithBot ℝ) := hMaxBot ▸ le_max_right _ _
  have : f ⟨0, hBk⟩ = ⊥ :=
    le_antisymm (le_trans hSup hSupBot) (OrderBot.bot_le _)
  exact WithBot.coe_ne_bot this

/-- In-bounds recurrence form of `mPartial`; avoids repeatedly unfolding the
guarded definition and discharging `k + 1 ≤ numKVBlocks`. -/
theorem mPartial_succ_of_lt {M D Bk : Nat}
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ)
    (k : Nat) (hk : k < numKVBlocks) (i : Fin M) :
    mPartial Bk Q numKVBlocks K scale (k + 1) i =
      max (mPartial Bk Q numKVBlocks K scale k i)
        (Finset.univ.sup (fun jLocal : Fin Bk =>
          ((scaledScore Q K scale i
            (blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) jLocal)
            : ℝ) : WithBot ℝ))) := by
  change (if h : k + 1 ≤ numKVBlocks then
      max (mPartial Bk Q numKVBlocks K scale k i)
        (Finset.univ.sup (fun jLocal : Fin Bk =>
          ((scaledScore Q K scale i
            (blockIndex Bk numKVBlocks k h jLocal) : ℝ) : WithBot ℝ)))
    else mPartial Bk Q numKVBlocks K scale k i) = _
  simp [Nat.succ_le_iff.mpr hk]

/-- In-bounds recurrence form of `lPartial`. -/
theorem lPartial_succ_of_lt {M D Bk : Nat}
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ)
    (k : Nat) (hk : k < numKVBlocks) (i : Fin M) :
    lPartial Q numKVBlocks K scale (k + 1) i =
      let mNew : ℝ := (mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0
      alphaPartial Q numKVBlocks K scale k i *
        lPartial Q numKVBlocks K scale k i +
        Finset.univ.sum (fun jLocal : Fin Bk =>
          Real.exp (scaledScore Q K scale i
              (blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) jLocal) - mNew)) := by
  change (if h : k + 1 ≤ numKVBlocks then
      (let mNew : ℝ := (mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0
       alphaPartial Q numKVBlocks K scale k i *
          lPartial Q numKVBlocks K scale k i +
        Finset.univ.sum (fun jLocal : Fin Bk =>
          Real.exp (scaledScore Q K scale i
              (blockIndex Bk numKVBlocks k h jLocal) - mNew)))
    else lPartial Q numKVBlocks K scale k i) = _
  simp [Nat.succ_le_iff.mpr hk]

/-- In-bounds recurrence form of `oPartial`. -/
theorem oPartial_succ_of_lt {M D Bk : Nat}
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ)
    (k : Nat) (hk : k < numKVBlocks) (idx : TileIndex [M, D]) :
    oPartial Q numKVBlocks K V scale (k + 1) idx =
      let i := idx.1
      let d := idx.2.1
      let mNew : ℝ := (mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0
      alphaPartial Q numKVBlocks K scale k i *
        oPartial Q numKVBlocks K V scale k idx +
        Finset.univ.sum (fun jLocal : Fin Bk =>
          Real.exp (scaledScore Q K scale i
              (blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) jLocal) - mNew) *
            V (blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) jLocal,
              d, PUnit.unit)) := by
  change (if h : k + 1 ≤ numKVBlocks then
      (let i := idx.1
       let d := idx.2.1
       let mNew : ℝ := (mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0
       alphaPartial Q numKVBlocks K scale k i *
          oPartial Q numKVBlocks K V scale k idx +
        Finset.univ.sum (fun jLocal : Fin Bk =>
          Real.exp (scaledScore Q K scale i
              (blockIndex Bk numKVBlocks k h jLocal) - mNew) *
            V (blockIndex Bk numKVBlocks k h jLocal, d, PUnit.unit)))
    else oPartial Q numKVBlocks K V scale k idx) = _
  simp [Nat.succ_le_iff.mpr hk]

/-- The streaming `lPartial k i` equals `exp(-m_k) · lFree k i`, where
`m_k = (mPartial k i).unbotD 0`. By induction on `k`: the α-factor
`exp(m_k - m_{k+1})` absorbs the shift difference at each step. -/
theorem lPartial_eq_mShifted {M D Bk : Nat} (_hBk : 0 < Bk)
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ)
    (k : Nat) (hk : k ≤ numKVBlocks) (i : Fin M) :
    lPartial Q numKVBlocks K scale k i =
      Real.exp (-(mPartial Bk Q numKVBlocks K scale k i).unbotD 0) *
        lFree Q K scale k hk i := by
  induction k with
  | zero =>
    -- LHS: lPartial 0 i = 0; RHS: exp(-((⊥).unbotD 0)) · lFree 0 = 1 · 0 = 0.
    show (0 : ℝ) =
      Real.exp (-((⊥ : WithBot ℝ).unbotD 0)) *
        lFree Q K scale 0 hk i
    rw [lFree_zero]
    ring
  | succ k ih =>
    have hk' : k ≤ numKVBlocks := Nat.le_of_succ_le hk
    -- Unfold LHS: αₖ * lPartial k + Σ exp(scaledScore - m_{k+1}).
    unfold lPartial
    simp only [hk, ↓reduceDIte]
    -- Apply IH on the lPartial k subterm.
    rw [ih hk']
    -- Expand RHS via lFree_succ.
    rw [lFree_succ Q K scale k hk i, mul_add]
    -- Goal:
    --   αₖ * (exp(-mₖ) * lFree k _) + Σ_jL exp(scaledScore - m_{k+1})
    --   = exp(-m_{k+1}) * lFree k _ + exp(-m_{k+1}) * Σ_jL exp(scaledScore)
    -- Match each term separately.
    have hExpSub : ∀ s : ℝ,
        Real.exp (s - (mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0) =
          Real.exp (-(mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0) *
            Real.exp s := by
      intro s
      rw [show s - (mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0
            = -(mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0 + s by ring,
          Real.exp_add]
    -- Term (b): Σ_jL exp(s - m_{k+1}) = exp(-m_{k+1}) * Σ_jL exp(s).
    have hSumB :
        Finset.univ.sum (fun jL : Fin Bk =>
          Real.exp (scaledScore Q K scale i (blockIndex Bk numKVBlocks k hk jL)
            - (mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0))
        = Real.exp (-(mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0) *
            Finset.univ.sum (fun jL : Fin Bk =>
              Real.exp (scaledScore Q K scale i
                (blockIndex Bk numKVBlocks k hk jL))) := by
      rw [Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro jL _
      exact hExpSub _
    -- Term (a): αₖ * (exp(-mₖ) * lFree k) = exp(-m_{k+1}) * lFree k.
    -- Case split on k.
    have hSumA :
        alphaPartial Q numKVBlocks K scale k i *
          (Real.exp (-(mPartial Bk Q numKVBlocks K scale k i).unbotD 0) *
            lFree Q K scale k hk' i)
        = Real.exp (-(mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0) *
            lFree Q K scale k hk' i := by
      rcases Nat.eq_zero_or_pos k with hkz | hkpos
      · -- k = 0: lFree 0 _ = 0, so both sides are zero.
        subst hkz
        rw [lFree_zero]
        ring
      · -- k ≥ 1: mPartial k is some real, so αₖ = exp(mₖ - m_{k+1}) and
        -- αₖ · exp(-mₖ) = exp(-m_{k+1}).
        -- Get the underlying reals.
        obtain ⟨k', rfl⟩ := Nat.exists_eq_succ_of_ne_zero (Nat.pos_iff_ne_zero.mp hkpos)
        -- Now `k = k' + 1`. Have hk : k' + 1 + 1 ≤ numKVBlocks ⇒ k' + 1 ≤ numKVBlocks.
        have hk_succ : k' + 1 ≤ numKVBlocks := Nat.le_of_succ_le hk
        have hmk_ne : mPartial Bk Q numKVBlocks K scale (k' + 1) i ≠ ⊥ :=
          mPartial_succ_ne_bot _hBk Q numKVBlocks K scale k' hk_succ i
        have hmk1_ne :
            mPartial Bk Q numKVBlocks K scale (k' + 1 + 1) i ≠ ⊥ :=
          mPartial_succ_ne_bot _hBk Q numKVBlocks K scale (k' + 1) hk i
        -- Extract real values.
        obtain ⟨rk, hrk⟩ := WithBot.ne_bot_iff_exists.mp hmk_ne
        obtain ⟨rk1, hrk1⟩ := WithBot.ne_bot_iff_exists.mp hmk1_ne
        -- Compute αₖ.
        have hAlpha :
            alphaPartial Q numKVBlocks K scale (k' + 1) i =
              Real.exp (rk - rk1) := by
          unfold alphaPartial
          rw [← hrk, ← hrk1]
          simp [WithBot.realSub]
        rw [hAlpha, ← hrk, ← hrk1]
        simp only [WithBot.unbotD_coe]
        -- Goal: exp(rk - rk1) * (exp(-rk) * lFree k' k _) = exp(-rk1) * lFree k' k _
        rw [show Real.exp (rk - rk1) = Real.exp (-rk1) * Real.exp rk by
              rw [show rk - rk1 = -rk1 + rk by ring, Real.exp_add]]
        rw [show Real.exp (-rk1) * Real.exp rk *
                  (Real.exp (-rk) * lFree Q K scale (k' + 1) hk' i)
              = Real.exp (-rk1) *
                  (Real.exp rk * Real.exp (-rk) * lFree Q K scale (k' + 1) hk' i) by ring]
        rw [show Real.exp rk * Real.exp (-rk) = 1 by
              rw [← Real.exp_add, add_neg_cancel, Real.exp_zero]]
        ring
    -- Combine.
    linarith [hSumA, hSumB]

/-- Companion identity for `oPartial`: `exp(-m_k) · oFree k i d`. Same
shape as `lPartial_eq_mShifted` plus `· V[j, d]` on each summand. -/
theorem oPartial_eq_mShifted {M D Bk : Nat} (_hBk : 0 < Bk)
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ)
    (k : Nat) (hk : k ≤ numKVBlocks) (idx : TileIndex [M, D]) :
    oPartial Q numKVBlocks K V scale k idx =
      Real.exp (-(mPartial Bk Q numKVBlocks K scale k idx.1).unbotD 0) *
        oFree Q K V scale k hk idx := by
  induction k with
  | zero =>
    show (0 : ℝ) =
      Real.exp (-((⊥ : WithBot ℝ).unbotD 0)) *
        oFree Q K V scale 0 hk idx
    rw [oFree_zero]
    ring
  | succ k ih =>
    have hk' : k ≤ numKVBlocks := Nat.le_of_succ_le hk
    unfold oPartial
    simp only [hk, ↓reduceDIte]
    rw [ih hk']
    rw [oFree_succ Q K V scale k hk idx, mul_add]
    -- Match the two terms; identical shape to `lPartial_eq_mShifted` plus `· V`.
    have hExpSub : ∀ s : ℝ,
        Real.exp (s - (mPartial Bk Q numKVBlocks K scale (k + 1) idx.1).unbotD 0) =
          Real.exp (-(mPartial Bk Q numKVBlocks K scale (k + 1) idx.1).unbotD 0) *
            Real.exp s := by
      intro s
      rw [show s - (mPartial Bk Q numKVBlocks K scale (k + 1) idx.1).unbotD 0
            = -(mPartial Bk Q numKVBlocks K scale (k + 1) idx.1).unbotD 0 + s by ring,
          Real.exp_add]
    have hSumB :
        Finset.univ.sum (fun jL : Fin Bk =>
          Real.exp (scaledScore Q K scale idx.1
              (blockIndex Bk numKVBlocks k hk jL)
            - (mPartial Bk Q numKVBlocks K scale (k + 1) idx.1).unbotD 0) *
          V (blockIndex Bk numKVBlocks k hk jL, idx.2.1, PUnit.unit))
        = Real.exp (-(mPartial Bk Q numKVBlocks K scale (k + 1) idx.1).unbotD 0) *
            Finset.univ.sum (fun jL : Fin Bk =>
              Real.exp (scaledScore Q K scale idx.1
                  (blockIndex Bk numKVBlocks k hk jL)) *
              V (blockIndex Bk numKVBlocks k hk jL, idx.2.1, PUnit.unit)) := by
      rw [Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro jL _
      rw [hExpSub]
      ring
    have hSumA :
        alphaPartial Q numKVBlocks K scale k idx.1 *
          (Real.exp (-(mPartial Bk Q numKVBlocks K scale k idx.1).unbotD 0) *
            oFree Q K V scale k hk' idx)
        = Real.exp (-(mPartial Bk Q numKVBlocks K scale (k + 1) idx.1).unbotD 0) *
            oFree Q K V scale k hk' idx := by
      rcases Nat.eq_zero_or_pos k with hkz | hkpos
      · subst hkz
        rw [oFree_zero]
        ring
      · obtain ⟨k', rfl⟩ := Nat.exists_eq_succ_of_ne_zero (Nat.pos_iff_ne_zero.mp hkpos)
        have hk_succ : k' + 1 ≤ numKVBlocks := Nat.le_of_succ_le hk
        have hmk_ne : mPartial Bk Q numKVBlocks K scale (k' + 1) idx.1 ≠ ⊥ :=
          mPartial_succ_ne_bot _hBk Q numKVBlocks K scale k' hk_succ idx.1
        have hmk1_ne :
            mPartial Bk Q numKVBlocks K scale (k' + 1 + 1) idx.1 ≠ ⊥ :=
          mPartial_succ_ne_bot _hBk Q numKVBlocks K scale (k' + 1) hk idx.1
        obtain ⟨rk, hrk⟩ := WithBot.ne_bot_iff_exists.mp hmk_ne
        obtain ⟨rk1, hrk1⟩ := WithBot.ne_bot_iff_exists.mp hmk1_ne
        have hAlpha :
            alphaPartial Q numKVBlocks K scale (k' + 1) idx.1 =
              Real.exp (rk - rk1) := by
          unfold alphaPartial
          rw [← hrk, ← hrk1]
          simp [WithBot.realSub]
        rw [hAlpha, ← hrk, ← hrk1]
        simp only [WithBot.unbotD_coe]
        rw [show Real.exp (rk - rk1) = Real.exp (-rk1) * Real.exp rk by
              rw [show rk - rk1 = -rk1 + rk by ring, Real.exp_add]]
        rw [show Real.exp (-rk1) * Real.exp rk *
                  (Real.exp (-rk) * oFree Q K V scale (k' + 1) hk' idx)
              = Real.exp (-rk1) *
                  (Real.exp rk * Real.exp (-rk) * oFree Q K V scale (k' + 1) hk' idx) by ring]
        rw [show Real.exp rk * Real.exp (-rk) = 1 by
              rw [← Real.exp_add, add_neg_cancel, Real.exp_zero]]
        ring
    linarith [hSumA, hSumB]

/-- Fin (Bk * N) ≃ Fin N × Fin Bk, the bijection underlying
`oFree`/`lFree`'s double-sum form. Composes `finProdFinEquiv.symm` with
a `mul_comm` cast. -/
def blockIndexEquiv (Bk N : Nat) : Fin (Bk * N) ≃ Fin N × Fin Bk :=
  (Fin.castOrderIso (Nat.mul_comm Bk N)).toEquiv.trans finProdFinEquiv.symm

/-- Helper: blockIndex round-trips through `blockIndexEquiv`. -/
theorem blockIndex_blockIndexEquiv {Bk N : Nat} (j : Fin (Bk * N)) :
    blockIndex Bk N ((blockIndexEquiv Bk N) j).1.val
        (by have := ((blockIndexEquiv Bk N) j).1.isLt; omega)
        ((blockIndexEquiv Bk N) j).2 = j := by
  apply Fin.ext
  show ((blockIndexEquiv Bk N) j).1.val * Bk +
        ((blockIndexEquiv Bk N) j).2.val = j.val
  -- `blockIndexEquiv = (Fin.castOrderIso mul_comm).toEquiv.trans
  --                     finProdFinEquiv.symm`.
  -- Apply to `j`: cast to `Fin (N * Bk)` then divNat/modNat. So
  -- `(blockIndexEquiv j).1.val = j.val / Bk` and
  -- `(blockIndexEquiv j).2.val = j.val % Bk`. Then standard
  -- `Nat.div_add_mod`.
  show ((finProdFinEquiv.symm
            ((Fin.castOrderIso (Nat.mul_comm Bk N)).toEquiv j))).1.val * Bk +
       ((finProdFinEquiv.symm
            ((Fin.castOrderIso (Nat.mul_comm Bk N)).toEquiv j))).2.val = j.val
  show (Fin.divNat ((Fin.castOrderIso (Nat.mul_comm Bk N)).toEquiv j)).val * Bk +
       (Fin.modNat ((Fin.castOrderIso (Nat.mul_comm Bk N)).toEquiv j)).val = j.val
  -- `(castOrderIso _ j).val = j.val` (cast preserves val).
  -- `Fin.divNat j' .val = j'.val / Bk`, `Fin.modNat j' .val = j'.val % Bk`.
  -- (These are unfoldings of definitions; if simp's not pulling them
  -- through, fall back on the well-known Nat identity.)
  have : (((Fin.castOrderIso (Nat.mul_comm Bk N)).toEquiv j)).val = j.val := rfl
  rw [show (Fin.divNat ((Fin.castOrderIso (Nat.mul_comm Bk N)).toEquiv j)).val
        = j.val / Bk from rfl,
      show (Fin.modNat ((Fin.castOrderIso (Nat.mul_comm Bk N)).toEquiv j)).val
        = j.val % Bk from rfl]
  rw [Nat.mul_comm (j.val / Bk) Bk]
  exact Nat.div_add_mod j.val Bk

/-- The flat `Σ over Fin (Bk*N)` form of `lFree N (le_refl _)`. Bridges
the double-sum form (used by streaming proofs) and the flat form (used
by `attentionReal` through `softmaxRow`). -/
theorem lFree_eq_flat {M D Bk N : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [Bk * N, D] → ℝ) (scale : ℝ)
    (i : Fin M) :
    lFree Q K scale N (le_refl _) i =
      Finset.univ.sum (fun j : Fin (Bk * N) =>
        Real.exp (scaledScore Q K scale i j)) := by
  unfold lFree
  rw [← Finset.sum_product', Finset.univ_product_univ]
  refine (Finset.sum_equiv (blockIndexEquiv Bk N) ?_ ?_).symm
  · intro _; simp
  · intro j _
    rw [blockIndex_blockIndexEquiv]

/-- Companion flat-form for `oFree`. -/
theorem oFree_eq_flat {M D Bk N : Nat}
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [Bk * N, D] → ℝ) (scale : ℝ)
    (idx : TileIndex [M, D]) :
    oFree Q K V scale N (le_refl _) idx =
      Finset.univ.sum (fun j : Fin (Bk * N) =>
        Real.exp (scaledScore Q K scale idx.1 j) *
        V (j, idx.2.1, PUnit.unit)) := by
  unfold oFree
  rw [← Finset.sum_product', Finset.univ_product_univ]
  refine (Finset.sum_equiv (blockIndexEquiv Bk N) ?_ ?_).symm
  · intro _; simp
  · intro j _
    rw [blockIndex_blockIndexEquiv]

/-- Auxiliary: the WithBot ℝ-valued `qkT` tile data computes the real
`scaledScore`-related dot product (no scale yet). All-`some` because
all operands are lifted via `Tile.ofReal`. -/
theorem qkT_data_eq {M D Bk N : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [Bk * N, D] → ℝ)
    (i : Fin M) (j : Fin (Bk * N)) :
    (Tile.dot [] (Tile.ofReal Q) (Tile.transpose [] (Tile.ofReal K))).data
        (i, j, PUnit.unit)
      = ((Finset.univ.sum (fun d : Fin D =>
            Q (i, d, PUnit.unit) * K (j, d, PUnit.unit)) : ℝ) : WithBot ℝ) := by
  rw [Tile.dot_nil_data]
  have hPush : ∑ d : Fin D, (((Q (i, d, PUnit.unit) * K (j, d, PUnit.unit)
                : ℝ) : WithBot ℝ)) =
      (((∑ d : Fin D, Q (i, d, PUnit.unit) * K (j, d, PUnit.unit)
          : ℝ) : WithBot ℝ)) :=
    (map_sum (WithBot.addHom : ℝ →+ WithBot ℝ) _ _).symm
  rw [← hPush]
  apply Finset.sum_congr rfl
  intro k _
  rw [Tile.transpose_nil_data, Tile.ofReal_data, Tile.ofReal_data]
  rfl

/-- The `scaled` tile inside `attention` data-evaluates to a `WithBot ℝ`
coercion of `scaledScore`. -/
theorem scaled_data_eq {M D Bk N : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [Bk * N, D] → ℝ)
    (scale : ℝ) (i : Fin M) (j : Fin (Bk * N)) :
    Option.map (· * scale)
        ((Tile.dot [] (Tile.ofReal Q)
          (Tile.transpose [] (Tile.ofReal K))).data (i, j, PUnit.unit))
      = ((scaledScore Q K scale i j : ℝ) : WithBot ℝ) := by
  rw [qkT_data_eq]
  unfold scaledScore
  show some _ = some _
  congr 1
  ring

/-- `softmaxRow` of `scaled` evaluates per-cell as
`some(exp(scaledScore) / Σ_j' exp(scaledScore))`. -/
theorem softmaxRow_scaled_data_eq {M D Bk N : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [Bk * N, D] → ℝ)
    (scale : ℝ) (i : Fin M) (n : Fin (Bk * N)) :
    let scaled : Tile .real [M, Bk * N] :=
      ⟨fun idx => Option.map (· * scale)
        ((Tile.dot [] (Tile.ofReal Q)
          (Tile.transpose [] (Tile.ofReal K))).data idx)⟩
    (softmaxRow scaled).data (i, n, PUnit.unit) =
      ((Real.exp (scaledScore Q K scale i n) /
        Finset.univ.sum (fun j' : Fin (Bk * N) =>
          Real.exp (scaledScore Q K scale i j')) : ℝ) : WithBot ℝ) := by
  -- Unfold softmaxRow; the row function evaluates to scaledScore via
  -- `scaled_data_eq` + `WithBot.unbotD_coe`.
  unfold softmaxRow
  show some _ = some _
  congr 1
  -- Goal: exp(row n) / Σ exp(row j) = exp(scaledScore i n) / Σ exp(scaledScore i j)
  -- where row j = (scaled.data (i, j, _)).unbotD 0
  have hRow : ∀ j' : Fin (Bk * N),
      ((⟨fun idx => Option.map (· * scale)
          ((Tile.dot [] (Tile.ofReal Q)
            (Tile.transpose [] (Tile.ofReal K))).data idx)⟩ : Tile .real _).data
        (i, j', PUnit.unit)).unbotD 0
        = scaledScore Q K scale i j' := by
    intro j'
    show WithBot.unbotD 0 (Option.map (· * scale)
        ((Tile.dot [] (Tile.ofReal Q)
          (Tile.transpose [] (Tile.ofReal K))).data (i, j', PUnit.unit)))
        = scaledScore Q K scale i j'
    rw [scaled_data_eq]
    rfl
  congr 1
  · exact congrArg Real.exp (hRow n)
  · apply Finset.sum_congr rfl
    intro j _
    exact congrArg Real.exp (hRow j)

/-! ### Shape-polymorphic variants

These mirror `qkT_data_eq` / `scaled_data_eq` / `softmaxRow_scaled_data_eq`
but take a generic `S` (not necessarily of the form `Bk * N`). Used by
the boundary streaming proofs where K/V live in `[S_k, D]` with no
factorization commitment. The proofs are textually identical — Lean's
definitional unfolding is the same; only the implicit shape changes. -/

/-- Shape-polymorphic version of `qkT_data_eq`. -/
theorem qkT_data_eq' {M D S : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S, D] → ℝ)
    (i : Fin M) (j : Fin S) :
    (Tile.dot [] (Tile.ofReal Q) (Tile.transpose [] (Tile.ofReal K))).data
        (i, j, PUnit.unit)
      = ((Finset.univ.sum (fun d : Fin D =>
            Q (i, d, PUnit.unit) * K (j, d, PUnit.unit)) : ℝ) : WithBot ℝ) := by
  rw [Tile.dot_nil_data]
  have hPush : ∑ d : Fin D, (((Q (i, d, PUnit.unit) * K (j, d, PUnit.unit)
                : ℝ) : WithBot ℝ)) =
      (((∑ d : Fin D, Q (i, d, PUnit.unit) * K (j, d, PUnit.unit)
          : ℝ) : WithBot ℝ)) :=
    (map_sum (WithBot.addHom : ℝ →+ WithBot ℝ) _ _).symm
  rw [← hPush]
  apply Finset.sum_congr rfl
  intro k _
  rw [Tile.transpose_nil_data, Tile.ofReal_data, Tile.ofReal_data]
  rfl

/-- Shape-polymorphic version of `scaled_data_eq`. -/
theorem scaled_data_eq' {M D S : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S, D] → ℝ)
    (scale : ℝ) (i : Fin M) (j : Fin S) :
    Option.map (· * scale)
        ((Tile.dot [] (Tile.ofReal Q)
          (Tile.transpose [] (Tile.ofReal K))).data (i, j, PUnit.unit))
      = ((scaledScore Q K scale i j : ℝ) : WithBot ℝ) := by
  rw [qkT_data_eq']
  unfold scaledScore
  show some _ = some _
  congr 1
  ring

/-- Shape-polymorphic version of `softmaxRow_scaled_data_eq`. -/
theorem softmaxRow_scaled_data_eq' {M D S : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S, D] → ℝ)
    (scale : ℝ) (i : Fin M) (n : Fin S) :
    let scaled : Tile .real [M, S] :=
      ⟨fun idx => Option.map (· * scale)
        ((Tile.dot [] (Tile.ofReal Q)
          (Tile.transpose [] (Tile.ofReal K))).data idx)⟩
    (softmaxRow scaled).data (i, n, PUnit.unit) =
      ((Real.exp (scaledScore Q K scale i n) /
        Finset.univ.sum (fun j' : Fin S =>
          Real.exp (scaledScore Q K scale i j')) : ℝ) : WithBot ℝ) := by
  unfold softmaxRow
  show some _ = some _
  congr 1
  have hRow : ∀ j' : Fin S,
      ((⟨fun idx => Option.map (· * scale)
          ((Tile.dot [] (Tile.ofReal Q)
            (Tile.transpose [] (Tile.ofReal K))).data idx)⟩ : Tile .real _).data
        (i, j', PUnit.unit)).unbotD 0
        = scaledScore Q K scale i j' := by
    intro j'
    show WithBot.unbotD 0 (Option.map (· * scale)
        ((Tile.dot [] (Tile.ofReal Q)
          (Tile.transpose [] (Tile.ofReal K))).data (i, j', PUnit.unit)))
        = scaledScore Q K scale i j'
    rw [scaled_data_eq']
    rfl
  congr 1
  · exact congrArg Real.exp (hRow n)
  · apply Finset.sum_congr rfl
    intro j _
    exact congrArg Real.exp (hRow j)

/-- Dot product against the `n`-th KV block computes the corresponding
flat-index score before scaling. This is the block-local companion to
`qkT_data_eq`, used by the operational loop proof. -/
theorem block_qkT_data_eq {M D Bk N : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [Bk * N, D] → ℝ)
    (n : Nat) (hn : n < N)
    (i : Fin M) (j : Fin Bk) :
    (Tile.dot [] (Tile.ofReal Q)
      (Tile.transpose [] (Tile.ofReal
        (fun idx : TileIndex [Bk, D] =>
          K (blockIndex Bk N n (Nat.succ_le_iff.mpr hn) idx.1,
            idx.2.1, PUnit.unit))))).data (i, j, PUnit.unit)
      =
        ((Finset.univ.sum (fun d : Fin D =>
          Q (i, d, PUnit.unit) *
          K (blockIndex Bk N n (Nat.succ_le_iff.mpr hn) j,
            d, PUnit.unit)) : ℝ) : WithBot ℝ) := by
  rw [Tile.dot_nil_data]
  have hPush : ∑ d : Fin D,
        (((Q (i, d, PUnit.unit) *
            K (blockIndex Bk N n (Nat.succ_le_iff.mpr hn) j,
              d, PUnit.unit) : ℝ) : WithBot ℝ)) =
      (((∑ d : Fin D,
          Q (i, d, PUnit.unit) *
            K (blockIndex Bk N n (Nat.succ_le_iff.mpr hn) j,
              d, PUnit.unit) : ℝ) : WithBot ℝ)) :=
    (map_sum (WithBot.addHom : ℝ →+ WithBot ℝ) _ _).symm
  rw [← hPush]
  apply Finset.sum_congr rfl
  intro d _
  rw [Tile.transpose_nil_data, Tile.ofReal_data, Tile.ofReal_data]
  rfl

/-- Scaled block score data: the exact value assigned to the loop-local
`scores[i,j]`. The AST multiplies the dot product by a scalar on the right;
`scaledScore` writes the scale on the left, so the proof finishes by
commutativity of real multiplication. -/
theorem block_scaled_data_eq {M D Bk N : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [Bk * N, D] → ℝ)
    (scale : ℝ) (n : Nat) (hn : n < N)
    (i : Fin M) (j : Fin Bk) :
    WithBot.realMul
      ((Tile.dot [] (Tile.ofReal Q)
        (Tile.transpose [] (Tile.ofReal
          (fun idx : TileIndex [Bk, D] =>
            K (blockIndex Bk N n (Nat.succ_le_iff.mpr hn) idx.1,
              idx.2.1, PUnit.unit))))).data (i, j, PUnit.unit))
      ((scale : ℝ) : WithBot ℝ)
      = ((scaledScore Q K scale i
          (blockIndex Bk N n (Nat.succ_le_iff.mpr hn) j) : ℝ) : WithBot ℝ) := by
  rw [block_qkT_data_eq Q K n hn i j]
  unfold scaledScore
  let S : ℝ := Finset.univ.sum (fun d : Fin D =>
    Q (i, d, PUnit.unit) *
      K (blockIndex Bk N n (Nat.succ_le_iff.mpr hn) j, d, PUnit.unit))
  change WithBot.realMul ((S : ℝ) : WithBot ℝ) ((scale : ℝ) : WithBot ℝ) =
      ((scale * S : ℝ) : WithBot ℝ)
  simp [WithBot.realMul]
  rw [mul_comm]

/-- Row-max of the current score block, matching the second argument of
`mPartial_succ_of_lt`. -/
theorem block_mBlock_data_eq {M D Bk N : Nat} (hBk : 0 < Bk)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [Bk * N, D] → ℝ)
    (scale : ℝ) (n : Nat) (hn : n < N) :
    (Tile.reduceMax (shape := [M, Bk]) ⟨1, by simp⟩ Bool.false
      (Tile.ofReal fun idx : TileIndex [M, Bk] =>
        scaledScore Q K scale idx.1
          (blockIndex Bk N n (Nat.succ_le_iff.mpr hn) idx.2.1)))
      =
        some ⟨fun idx : TileIndex [M] =>
          (Finset.univ : Finset (Fin Bk)).sup
            (fun j => ((scaledScore Q K scale idx.1
              (blockIndex Bk N n (Nat.succ_le_iff.mpr hn) j) : ℝ) :
                WithBot ℝ))⟩ := by
  unfold Tile.reduceMax
  simp [Tile.reduceMaxDrop, TileShape.axisDim, TileShape.eraseAxis,
    TileShape.insertAxisIndex, hBk, Tile.ofReal]
  funext idx
  change (((Finset.univ : Finset (Fin Bk)).sup'
      (by exact ⟨⟨0, hBk⟩, Finset.mem_univ _⟩)
      (fun j => scaledScore Q K scale idx.1
        (blockIndex Bk N n (Nat.succ_le_iff.mpr hn) j)) : ℝ) : WithBot ℝ) =
    (Finset.univ : Finset (Fin Bk)).sup
      (fun j => ((scaledScore Q K scale idx.1
        (blockIndex Bk N n (Nat.succ_le_iff.mpr hn) j) : ℝ) : WithBot ℝ))
  rw [← WithBot.sup'_coe]
  rw [Finset.sup'_eq_sup]

/-- Data form of the loop-local unnormalized probabilities
`p = exp(scores - m_new[:, None])`. -/
theorem block_p_data_eq {M D Bk N : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [Bk * N, D] → ℝ)
    (scale : ℝ) (n : Nat) (hn : n < N)
    (mNew : Fin M → WithBot ℝ) (i : Fin M) (j : Fin Bk) :
    WithBot.realExp
      (WithBot.realSub
        ((scaledScore Q K scale i
          (blockIndex Bk N n (Nat.succ_le_iff.mpr hn) j) : ℝ) : WithBot ℝ)
        (mNew i))
      =
      WithBot.realExp
        (WithBot.realSub
          ((scaledScore Q K scale i
            (blockIndex Bk N n (Nat.succ_le_iff.mpr hn) j) : ℝ) : WithBot ℝ)
          (mNew i)) := rfl

/-- The operational `p` entry is a real value once `mPartial (k+1)` is known
not to be `⊥` (true for non-empty KV blocks). -/
theorem block_p_toWithBot {M D Bk N : Nat} (hBk : 0 < Bk)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [Bk * N, D] → ℝ)
    (scale : ℝ) (k : Nat) (hk : k < N) (i : Fin M) (j : Fin Bk) :
    WithBot.realExp
      (WithBot.realSub
        ((scaledScore Q K scale i
          (blockIndex Bk N k (Nat.succ_le_iff.mpr hk) j) : ℝ) : WithBot ℝ)
        (mPartial Bk Q N K scale (k + 1) i))
      =
      ((Real.exp (scaledScore Q K scale i
          (blockIndex Bk N k (Nat.succ_le_iff.mpr hk) j) -
        (mPartial Bk Q N K scale (k + 1) i).unbotD 0) : ℝ) :
          WithBot ℝ) := by
  have hm : mPartial Bk Q N K scale (k + 1) i ≠ ⊥ :=
    mPartial_succ_ne_bot hBk Q N K scale k (Nat.succ_le_iff.mpr hk) i
  obtain ⟨m, hm_eq⟩ := WithBot.ne_bot_iff_exists.mp hm
  rw [← hm_eq]
  simp [WithBot.realSub]

/-- Row-sum of the current `p` block in the non-`⊥` case where `m_new` is
known to be real-valued. This is the additive block contribution in
`lPartial_succ_of_lt`. -/
theorem block_p_rowSum_eq {M D Bk N : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [Bk * N, D] → ℝ)
    (scale : ℝ) (n : Nat) (hn : n < N)
    (mNew : Fin M → ℝ) :
    (Tile.reduceSum (shape := [M, Bk]) ⟨1, by simp⟩ Bool.false
      (Tile.ofReal fun idx : TileIndex [M, Bk] =>
        Real.exp (scaledScore Q K scale idx.1
          (blockIndex Bk N n (Nat.succ_le_iff.mpr hn) idx.2.1) - mNew idx.1)))
      =
        (Tile.ofReal fun idx : TileIndex [M] =>
          Finset.univ.sum (fun j : Fin Bk =>
            Real.exp (scaledScore Q K scale idx.1
              (blockIndex Bk N n (Nat.succ_le_iff.mpr hn) j) - mNew idx.1))) := by
  ext idx
  unfold Tile.reduceSum
  simp [Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
    TileShape.insertAxisIndex, Tile.ofReal]
  rfl

/-- Dotting the current probability block with the current V block gives the
block contribution used by `oPartial_succ_of_lt`. -/
theorem block_pv_data_eq {M D Bk N : Nat}
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * N, D] → ℝ) (scale : ℝ)
    (n : Nat) (hn : n < N) (mNew : Fin M → ℝ)
    (i : Fin M) (d : Fin D) :
    (Tile.dot [] (M := M) (K := Bk) (N := D)
      (Tile.ofReal fun idx : TileIndex [M, Bk] =>
        Real.exp (scaledScore Q K scale idx.1
          (blockIndex Bk N n (Nat.succ_le_iff.mpr hn) idx.2.1) - mNew idx.1))
      (Tile.ofReal fun idx : TileIndex [Bk, D] =>
        V (blockIndex Bk N n (Nat.succ_le_iff.mpr hn) idx.1,
          idx.2.1, PUnit.unit))).data (i, d, PUnit.unit)
      =
        ((Finset.univ.sum (fun j : Fin Bk =>
          Real.exp (scaledScore Q K scale i
              (blockIndex Bk N n (Nat.succ_le_iff.mpr hn) j) - mNew i) *
            V (blockIndex Bk N n (Nat.succ_le_iff.mpr hn) j,
              d, PUnit.unit)) : ℝ) : WithBot ℝ) := by
  rw [Tile.dot_nil_data]
  have hPush : ∑ j : Fin Bk,
        (((Real.exp (scaledScore Q K scale i
              (blockIndex Bk N n (Nat.succ_le_iff.mpr hn) j) - mNew i) *
            V (blockIndex Bk N n (Nat.succ_le_iff.mpr hn) j,
              d, PUnit.unit) : ℝ) : WithBot ℝ)) =
      (((∑ j : Fin Bk,
          Real.exp (scaledScore Q K scale i
              (blockIndex Bk N n (Nat.succ_le_iff.mpr hn) j) - mNew i) *
            V (blockIndex Bk N n (Nat.succ_le_iff.mpr hn) j,
              d, PUnit.unit) : ℝ) : WithBot ℝ)) :=
    (map_sum (WithBot.addHom : ℝ →+ WithBot ℝ) _ _).symm
  rw [← hPush]
  apply Finset.sum_congr rfl
  intro j _
  rw [Tile.ofReal_data, Tile.ofReal_data]
  rfl

/-- The loop-local `m_new = max(m_i, m_block)` realizes the streaming
`mPartial` recurrence at `k + 1`. -/
theorem block_mNew_tile_eq {M D Bk N : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [Bk * N, D] → ℝ)
    (scale : ℝ) (k : Nat) (hk : k < N) :
    Tile.bop max (Broadcast.consSame Broadcast.nil)
      (⟨fun idx : TileIndex [M] => mPartial Bk Q N K scale k idx.1⟩ : Tile .real [M])
      (⟨fun idx : TileIndex [M] =>
        (Finset.univ : Finset (Fin Bk)).sup
          (fun j => ((scaledScore Q K scale idx.1
            (blockIndex Bk N k (Nat.succ_le_iff.mpr hk) j) : ℝ) :
              WithBot ℝ))⟩ : Tile .real [M])
      =
      ⟨fun idx : TileIndex [M] => mPartial Bk Q N K scale (k + 1) idx.1⟩ := by
  ext idx
  simp [Tile.bop]
  rw [mPartial_succ_of_lt Q N K scale k hk idx.1]

/-- Bridge between `Option ℝ`'s `Max` instance (used by `Tile.bop max bc`
when the carrier is the raw `Option ℝ`) and `WithBot ℝ`'s `Max` instance
(coming from `SemilatticeSup.toMax`). The two are propositionally — but not
definitionally — equal, so we materialize the equality as a rewrite lemma
that `simp_rw` can fire inside the `fa1_step` proof. -/
theorem option_max_eq_withbot_max (a b : WithBot ℝ) :
    @max (Option ℝ) Option.instMax a b = max a b := by
  cases a <;> cases b <;> rfl

/-- Proof irrelevance for `Finset.sup'`: any two `Nonempty` witnesses give the
same value. Used in the `fa1_step` proof to align `Finset.univ.sup'` proof
arguments coming from `Tile.bop` elaboration with proof arguments coming from
the helper hypotheses (`hAlphaDataMaxOf` etc.). -/
theorem sup'_proof_irrel {α ι : Type*} [SemilatticeSup α] {s : Finset ι}
    (h₁ h₂ : s.Nonempty) (f : ι → α) : s.sup' h₁ f = s.sup' h₂ f := rfl

/-- The loop-local `l_new` expression realizes the streaming `lPartial`
recurrence at `k + 1`. -/
theorem block_lNew_tile_eq {M D Bk N : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [Bk * N, D] → ℝ)
    (scale : ℝ) (k : Nat) (hk : k < N) :
    Tile.bop WithBot.realAdd (Broadcast.consSame Broadcast.nil)
      (Tile.bop WithBot.realMul (Broadcast.consSame Broadcast.nil)
        (Tile.ofReal fun idx : TileIndex [M] =>
          alphaPartial Q N K scale k idx.1)
        (Tile.ofReal fun idx : TileIndex [M] =>
          lPartial Q N K scale k idx.1))
      (Tile.ofReal fun idx : TileIndex [M] =>
        Finset.univ.sum (fun j : Fin Bk =>
          Real.exp (scaledScore Q K scale idx.1
            (blockIndex Bk N k (Nat.succ_le_iff.mpr hk) j) -
              (mPartial Bk Q N K scale (k + 1) idx.1).unbotD 0)))
      =
      Tile.ofReal (fun idx : TileIndex [M] =>
        lPartial Q N K scale (k + 1) idx.1) := by
  ext idx
  simp [Tile.bop, Tile.ofReal]
  rw [lPartial_succ_of_lt Q N K scale k hk idx.1]

/-- The loop-local `o_acc` update realizes the streaming `oPartial`
recurrence at `k + 1`. -/
theorem block_oAcc_tile_eq {M D Bk N : Nat}
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * N, D] → ℝ) (scale : ℝ)
    (k : Nat) (hk : k < N) :
    Tile.bop WithBot.realAdd (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
      (Tile.bop WithBot.realMul (Broadcast.consSame (Broadcast.consL Broadcast.nil))
        (Tile.expandDim ⟨1, by simp⟩
          (Tile.ofReal fun idx : TileIndex [M] =>
            alphaPartial Q N K scale k idx.1))
        (Tile.ofReal fun idx : TileIndex [M, D] =>
          oPartial Q N K V scale k idx))
      (Tile.ofReal fun idx : TileIndex [M, D] =>
        Finset.univ.sum (fun j : Fin Bk =>
          Real.exp (scaledScore Q K scale idx.1
              (blockIndex Bk N k (Nat.succ_le_iff.mpr hk) j) -
                (mPartial Bk Q N K scale (k + 1) idx.1).unbotD 0) *
            V (blockIndex Bk N k (Nat.succ_le_iff.mpr hk) j,
              idx.2.1, PUnit.unit)))
      =
      Tile.ofReal (fun idx : TileIndex [M, D] =>
        oPartial Q N K V scale (k + 1) idx) := by
  ext idx
  rcases idx with ⟨i, d, u⟩
  cases u
  simp [Tile.bop, Tile.expandDim, Tile.ofReal, TileShape.dropInsertedIndex]
  rw [oPartial_succ_of_lt Q N K V scale k hk (i, d, PUnit.unit)]

/-- The m-free reference sums `oFree N` and `lFree N` connect to
`attentionReal` directly. This is the specification-side identity
(no streaming algebra involved). -/
theorem oFree_div_lFree_eq_attentionReal {M D Bk N : Nat}
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [Bk * N, D] → ℝ)
    (scale : ℝ) (idx : TileIndex [M, D])
    (_hlFree : lFree Q K scale N (le_refl N) idx.1 ≠ 0) :
    oFree Q K V scale N (le_refl N) idx /
        lFree Q K scale N (le_refl N) idx.1
      = attentionReal Q K V scale idx := by
  rcases idx with ⟨i, d, u⟩
  cases u
  rw [oFree_eq_flat, lFree_eq_flat]
  unfold attentionReal
  rfl

/-- **Math identity (paper centerpiece).** After all `numKVBlocks`
iterations, the streaming `(oPartial, lPartial)` ratio computes the
same value as `attentionReal`. The proof factors `exp(-m_N)` out of
both numerator and denominator (via `lPartial_eq_mShifted` and
`oPartial_eq_mShifted`), cancels it, and matches the residual
`oFree / lFree` against `attentionReal`. -/
theorem streaming_eq_attentionReal {M D Bk : Nat} (hBk : 0 < Bk)
    (Q : TileIndex [M, D] → ℝ)
    (numKVBlocks : Nat) (_hN : 0 < numKVBlocks)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ)
    (idx : TileIndex [M, D])
    (hl : lPartial Q numKVBlocks K scale numKVBlocks idx.1 ≠ 0) :
    oPartial Q numKVBlocks K V scale numKVBlocks idx /
        lPartial Q numKVBlocks K scale numKVBlocks idx.1
      = attentionReal Q K V scale idx := by
  -- Factor exp(-m_N) out of both sides, then cancel.
  rw [oPartial_eq_mShifted hBk Q numKVBlocks K V scale numKVBlocks
        (le_refl _) idx,
      lPartial_eq_mShifted hBk Q numKVBlocks K scale numKVBlocks
        (le_refl _) idx.1]
  -- After rewriting:
  --   (exp(-m) · oFree) / (exp(-m) · lFree) = attentionReal
  -- Use `mul_div_mul_left` to cancel exp(-m), which is non-zero.
  rw [mul_div_mul_left _ _ (Real.exp_ne_zero _)]
  -- Now goal: oFree N idx / lFree N idx.1 = attentionReal idx.
  -- We need lFree ≠ 0 to invoke the spec-side identity. Derive from `hl`:
  -- `lPartial = exp(-m) · lFree` and `lPartial ≠ 0` ⇒ `lFree ≠ 0`.
  have hlFree : lFree Q K scale numKVBlocks (le_refl _) idx.1 ≠ 0 := by
    intro h
    apply hl
    rw [lPartial_eq_mShifted hBk Q numKVBlocks K scale numKVBlocks
        (le_refl _) idx.1, h, mul_zero]
  exact oFree_div_lFree_eq_attentionReal Q K V scale idx hlFree

/-- The final unshifted softmax normalizer is strictly positive when the
KV domain is non-empty. -/
theorem lFree_final_pos {M D Bk N : Nat} (hBk : 0 < Bk) (hN : 0 < N)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [Bk * N, D] → ℝ)
    (scale : ℝ) (i : Fin M) :
    0 < lFree Q K scale N (le_refl N) i := by
  rw [lFree_eq_flat]
  apply Finset.sum_pos
  · intro j _
    exact Real.exp_pos _
  · exact ⟨⟨0, Nat.mul_pos hBk hN⟩, Finset.mem_univ _⟩

/-- The final streaming normalizer is non-zero under a non-empty KV scope.
This is the denominator fact needed by the readout stage. -/
theorem lPartial_final_ne_zero {M D Bk : Nat} (hBk : 0 < Bk)
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (hN : 0 < numKVBlocks)
    (K : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ)
    (i : Fin M) :
    lPartial Q numKVBlocks K scale numKVBlocks i ≠ 0 := by
  rw [lPartial_eq_mShifted hBk Q numKVBlocks K scale numKVBlocks
      (le_refl _) i]
  exact mul_ne_zero (Real.exp_ne_zero _)
    (ne_of_gt (lFree_final_pos hBk hN Q K scale i))

end StreamingAccumulator

end VeriTile
