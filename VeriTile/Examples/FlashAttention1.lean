/-
VeriTile.Examples.FlashAttention1

FlashAttention-1 forward kernel — DSL definition + math model + correctness
*statement* (proof = sorry, to be discharged in issue #38).

v0 scope (issue #37):
  * One `program_id` ↔ one Q-row block of `[BLOCK_M, D]`.
  * K, V are full `[S, D]` matrices in memory (single-block-output).
  * Inner KV-block loop with the canonical online-softmax update
    (running `m_i`, `l_i`, `O`).
  * `1/√D` scaling (Triton-faithful).
  * **Non-causal** — causal masking (uses `tl.where`) is a follow-up.

The proof of `fa1_forward_correct` is left as `sorry` here so the
kernel definition + math model + statement can land independently of
the proof effort.
-/

import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.DSL
import VeriTile.Examples.Common

namespace VeriTile.Examples

open VeriTile.Triton

/-! ## FA-1 forward kernel — DSL definition

Online-softmax recurrence over KV blocks:

  m_i, l_i, O ← running max / normalizer / output (all per Q-row)

  for n in 0..numKVBlocks:
    K_b, V_b ← load
    s   = (Q · K_bᵀ) · scale                       -- [M, Bk]
    m'  = max(m_i, rowmax(s))                          -- [M]
    α   = exp(m_i - m')                                -- [M]
    p   = exp(s - m'[:, None])                         -- [M, Bk]
    l'  = α * l_i + rowsum(p)                          -- [M]
    O'  = α[:, None] * O + p · V_b                     -- [M, D]
    (m_i, l_i, O) ← (m', l', O')

  out = O / l_i[:, None]                               -- [M, D]

The kernel is parameterized by the four regions (Q, K, V, output) and
by `M` (Q-block / output rows), `D` (head dim), `Bk` (KV-block size),
and `numKVBlocks` (`S = numKVBlocks * Bk` is the implicit invariant).
The scaling factor `1/√D` enters as a Lean-level `ℝ` antiquote so the
kernel can be parameterized by precomputed constants. -/

def fa1ForwardKernel
    (qReg kReg vReg outReg : RegionName)
    (M D Bk numKVBlocks : Nat) (scale : ℝ) : Kernel := triton {
  pid    := tl.program_id(0)

  offs_m := pid * $(M) + tl.arange(0, $(M))
  offs_d := tl.arange(0, $(D))

  -- Load Q-block once (reused across all KV blocks).
  q_ptrs := offs_m[:, None] * $(D) + offs_d[None, :]
  q      := tl.load($(qReg) + q_ptrs)

  -- Initialize accumulators at the right tile shape (so the loop
  -- registers stay rank-correct across iterations).
  m_i    := tl.full([$(M)], -inf)
  l_i    := tl.zeros([$(M)])
  o_acc  := tl.zeros([$(M), $(D)])

  tl.for n in $(numKVBlocks) {
    offs_n  := n * $(Bk) + tl.arange(0, $(Bk))
    k_ptrs  := offs_n[:, None] * $(D) + offs_d[None, :]
    v_ptrs  := offs_n[:, None] * $(D) + offs_d[None, :]
    k       := tl.load($(kReg) + k_ptrs)
    v       := tl.load($(vReg) + v_ptrs)

    -- Scaled scores: [M, Bk]
    scores  := tl.dot(q, tl.trans(k)) * $ℝ(scale)

    -- Online-softmax update.
    m_block := tl.max(scores, axis = 1)
    m_new   := tl.max(m_i, m_block)
    alpha   := tl.exp(m_i - m_new)
    p       := tl.exp(scores - m_new[:, None])
    l_new   := alpha * l_i + tl.sum(p, axis = 1)
    o_acc   := alpha[:, None] * o_acc + tl.dot(p, v)
    m_i     := m_new
    l_i     := l_new
  }

  -- Final normalize and store.
  out    := o_acc / l_i[:, None]
  o_ptrs := offs_m[:, None] * $(D) + offs_d[None, :]
  tl.store($(outReg) + o_ptrs, out)
}

/-! ## Math model — softmax-attention

Spec layer is ℝ-valued: `BlockState.mem` only reads ℝ, never `⊥`, so
the natural type of a `tl.load`-fed kernel input is
`TileIndex shape → ℝ`. We lift to `Tile .real` (whose carrier is
`WithBot ℝ`) only as an internal staging step — `Tile.ofReal` /
`unbotD 0` round-trip — to reuse the existing `Tile.dot` /
`Tile.transpose` machinery. The `⊥` sentinel is reserved for things
like `-inf` / `tl.full(_, -inf)` / masked-off lanes that arise
*inside* a kernel, not at its inputs/outputs. -/

/-- Row-wise softmax along the trailing axis on a `Tile .real`. (Math
reference — no row-max subtraction; correctness is what matters at
the spec layer.) -/
noncomputable def softmaxRow {M N : Nat} (s : Tile .real [M, N]) :
    Tile .real [M, N] :=
  ⟨fun (m, n, _) =>
    let row := fun j : Fin N => (s.data (m, j, PUnit.unit)).unbotD 0
    let num := Real.exp (row n)
    let denom := Finset.univ.sum (fun j : Fin N => Real.exp (row j))
    some (num / denom)⟩

/-- Internal `Tile`-level attention helper. Takes `Tile .real` operands
and reuses `Tile.dot` / `Tile.transpose`. The user-facing spec is
`attentionReal` below; this helper exists so the proof can still pivot
through tile-level lemmas. -/
noncomputable def attention {M S D : Nat}
    (Q : Tile .real [M, D]) (K V : Tile .real [S, D])
    (scale : ℝ) : Tile .real [M, D] :=
  let qkT : Tile .real [M, S] :=
    Tile.dot [] Q (Tile.transpose [] K)
  let scaled : Tile .real [M, S] :=
    ⟨fun idx => Option.map (· * scale) (qkT.data idx)⟩
  let p : Tile .real [M, S] := softmaxRow scaled
  Tile.dot [] p V

/-- ℝ-valued reference attention: `softmax(Q · Kᵀ · scale) · V` on
plain `TileIndex → ℝ` inputs. Lifts through `Tile.ofReal`, runs
`attention`, projects back via `unbotD 0`. This is what
`fa1_forward_correct` compares the kernel against. -/
noncomputable def attentionReal {M S D : Nat}
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (scale : ℝ) : TileIndex [M, D] → ℝ :=
  fun idx =>
    ((attention (Tile.ofReal Q) (Tile.ofReal K) (Tile.ofReal V)
        scale).data idx).unbotD 0

/-! ## Streaming math model (FA-1 online softmax recurrence)

Mirrors what `fa1ForwardKernel` computes block by block. Three running
quantities, each parameterized by the number of completed KV blocks
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

namespace FA1Math

/-- Scaled score `(Q · Kᵀ · scale)[i, j]` at a flat KV index `j ∈ [0, S)`.
The streaming form indexes K/V by `k * Bk + jLocal` for the k-th block;
this helper normalizes that to a single `j` so the recurrence can sum
over the whole input length uniformly. -/
noncomputable def scaledScore {M S D : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S, D] → ℝ) (scale : ℝ)
    (i : Fin M) (j : Fin S) : ℝ :=
  scale * Finset.univ.sum (fun d : Fin D =>
    Q (i, d, PUnit.unit) * K (j, d, PUnit.unit))

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
Closely mirrors `onlineSoftmaxM` from `Examples/OnlineSoftmax.lean`. -/
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

/-- The α multiplier `exp(m_k - m_{k+1})` produced by FA-1's online
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

/-- The flat `Σ over Fin (Bk*N)` form of `lFree N (le_refl _)`. Bridges
the double-sum form (used by streaming proofs) and the flat form (used
by `attentionReal` through `softmaxRow`). -/
theorem lFree_eq_flat {M D Bk N : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [Bk * N, D] → ℝ) (scale : ℝ)
    (i : Fin M) :
    lFree Q K scale N (le_refl _) i =
      Finset.univ.sum (fun j : Fin (Bk * N) =>
        Real.exp (scaledScore Q K scale i j)) := by
  -- Reindex via `blockIndexEquiv`. Mechanical Finset.sum bookkeeping
  -- (Finset.sum_product' + Finset.sum_equiv); deferred.
  sorry

/-- Companion flat-form for `oFree`. -/
theorem oFree_eq_flat {M D Bk N : Nat}
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [Bk * N, D] → ℝ) (scale : ℝ)
    (idx : TileIndex [M, D]) :
    oFree Q K V scale N (le_refl _) idx =
      Finset.univ.sum (fun j : Fin (Bk * N) =>
        Real.exp (scaledScore Q K scale idx.1 j) *
        V (j, idx.2.1, PUnit.unit)) := by
  sorry

/-- The m-free reference sums `oFree N` and `lFree N` connect to
`attentionReal` directly through `Tile.dot` / `softmaxRow`. This is the
specification-side identity (no streaming algebra involved). The proof
flat-forms both sides (`*_eq_flat`) and unfolds `attentionReal` through
its `Tile.dot` / `softmaxRow` chain. -/
theorem oFree_div_lFree_eq_attentionReal {M D Bk N : Nat}
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [Bk * N, D] → ℝ)
    (scale : ℝ) (idx : TileIndex [M, D])
    (_hlFree : lFree Q K scale N (le_refl N) idx.1 ≠ 0) :
    oFree Q K V scale N (le_refl N) idx /
        lFree Q K scale N (le_refl N) idx.1
      = attentionReal Q K V scale idx := by
  -- The remaining content: unfold `attentionReal` through `attention`,
  -- `Tile.dot []`, `Tile.transpose []`, `softmaxRow`, and the
  -- `Tile.ofReal`-lifted operands; show the result equals
  -- `(Σ over Fin (Bk*N), exp(scaledScore i j) · V j d) / Σ exp(...)`,
  -- which by `oFree_eq_flat` and `lFree_eq_flat` matches the LHS. The
  -- unfolding chain is mechanical but lengthy; deferred.
  sorry

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

end FA1Math

/-! ## Operational layer — `P_fa1` invariant + proof skeleton (issue #38)

Carries Stage A's streaming functions through the kernel as a
`forLoop_inv` invariant, the way `Examples/OnlineSoftmax.lean` does
for `onlineSoftmaxM` / `onlineSoftmaxL`. Proofs at this layer remain
`sorry` while we land the structural scaffold; they reduce to:

* `P_fa1_init` — the post-pre-loop state satisfies `P_fa1 0`.
* `P_fa1_step` — the loop body preserves `P_fa1` under `n → n+1`.
* `P_fa1_readout` — given `P_fa1 numKVBlocks s`, the post-store
  memory at `outReg` equals `attentionReal Q K V scale idx`,
  bridging through `streaming_eq_attentionReal`.

These three combine via `forLoop_inv` and explicit `stmts_cons`
walks to discharge `fa1_forward_correct`. -/

/-- Loop-carried predicate: at iteration count `k ∈ [0, numKVBlocks]`,
the kernel state holds the streaming math values `mPartial k`,
`lPartial k`, `oPartial k` in the corresponding registers, plus the
preset Q-block and book-keeping registers. K, V remain loaded. -/
def P_fa1
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg : RegionName)
    (origPid : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ) (k : Nat) (s : BlockState) : Prop :=
  -- `pid` register and value preserved.
  s.regs .nat [] "pid" = some (Tile.scalar origPid) ∧
  s.pid = origPid ∧
  -- Q-row-block index vector (set pre-loop, reused post-loop for the
  -- final `tl.store` ptrs).
  s.regs .nat [M] "offs_m" = some
      (Tile.vec (fun i : Fin M => origPid * M + i.val)) ∧
  -- D-axis index vector.
  s.regs .nat [D] "offs_d" = some
      (Tile.vec (fun d : Fin D => d.val)) ∧
  -- Loaded Q-row-block.
  s.regs .real [M, D] "q" = some (Tile.ofReal Q) ∧
  -- Streaming accumulators at iteration `k`.
  s.regs .real [M] "m_i" = some
      ⟨fun idx : TileIndex [M] => FA1Math.mPartial Bk Q numKVBlocks K scale k idx.1⟩ ∧
  s.regs .real [M] "l_i" = some
      (Tile.ofReal fun idx : TileIndex [M] =>
        FA1Math.lPartial Q numKVBlocks K scale k idx.1) ∧
  s.regs .real [M, D] "o_acc" = some
      (Tile.ofReal fun idx : TileIndex [M, D] =>
        FA1Math.oPartial Q numKVBlocks K V scale k idx) ∧
  -- Input regions still loaded as the user's hypothesis says.
  InputAt s qReg
      (Offset.rowMajor2D (rows := M) (cols := D) (origPid * M * D) D) Q ∧
  InputAt s kReg
      (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) K ∧
  InputAt s vReg
      (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) V

/-! ## Correctness statement (sorry — discharged in issue #38)

The statement uses the standard `InputAt` / `observeTileAt` predicates
from `Examples.Common` to relate the kernel's memory effects to the
math model on `Tile`s. The kernel's optional final state is fed
directly into `observeTileAt`; the proof in #38 must therefore also
discharge \"`exec` succeeds\" (presumably via positivity of `M`, `D`,
`Bk`, `numKVBlocks` once those become required by the inductive
loop-invariant argument). -/

/-- After running `fa1ForwardKernel` on a state where Q is loaded as a
contiguous `[M, D]` row-major block at `(s.pid * M, ⋯)` and K, V are
loaded as full `[S, D]` row-major matrices at `(0, ⋯)`, the output
region holds `attentionReal(Q, K, V, scale)` row-major in the same
`[M, D]` block.

`scale : ℝ` is an arbitrary user-supplied scaling factor (the kernel
does not enforce `scale = 1 / √D`). The standard \"Triton attention\"
specialization is the caller's responsibility.

Spec is in plain ℝ: `Q`, `K`, `V` are `TileIndex shape → ℝ` matching
`BlockState.mem`'s ℝ carrier. The `WithBot ℝ` sentinel `⊥` does not
appear at the input boundary; it's an internal sentinel for things
like `tl.full(_, -inf)` / masked-off lanes / `-inf` introduced by the
kernel.

**Proof skeleton (see issue #38).** The proof follows the same shape
as `online_softmax_correct` in `Examples/OnlineSoftmax.lean`, scaled
to FA-1's bigger statement:

```text
1. Stage B init — exec walks through the 8 pre-loop statements
   (program_id, two aranges, Q-load, three tile-init fulls), each
   matched by an explicit `stepStmt` rewrite. Resulting state
   satisfies `P_fa1 0`.

2. Stage C step — invoke `forLoop_inv` with `P := P_fa1 …` and the
   per-iteration body. The body's 13 statements need their
   evalOp / Tile.dot / Tile.transpose / Tile.expandDim semantics
   threaded through; the math-side recurrence (mPartial / lPartial /
   oPartial at k → k+1) closes the inductive step.

3. Stage D readout — given `P_fa1 numKVBlocks s_final`, the
   post-loop assigns `out := o_acc / l_i[:, None]` and
   `tl.store(outReg + ptrs, out)` realize
   `streaming_eq_attentionReal` at the operational layer; the
   `observeTileAt … = some (attentionReal …)` follows.
```

Each Stage above is itself a multi-hundred-line proof (mirroring
OnlineSoftmax's scale × the rank-2 / batched / multi-stage
complexity of FA-1). v0 leaves the body as `sorry`. -/
theorem fa1_forward_correct
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg outReg : RegionName)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ)
    (s : BlockState)
    (_hQ : InputAt s qReg
        (Offset.rowMajor2D (rows := M) (cols := D) (s.pid * M * D) D) Q)
    (_hK : InputAt s kReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) K)
    (_hV : InputAt s vReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) V) :
    ∀ idx : TileIndex [M, D],
      observeTileAt
          (exec (fa1ForwardKernel qReg kReg vReg outReg M D Bk numKVBlocks scale) s)
          outReg
          (Offset.rowMajor2D (rows := M) (cols := D) (s.pid * M * D) D) idx
        = some (attentionReal Q K V scale idx) := by
  -- See doc above for the proof skeleton (Stages B / C / D); each
  -- step is multi-hundred lines and depends on the streaming math
  -- identity (`streaming_eq_attentionReal`) being proved first.
  sorry

end VeriTile.Examples
