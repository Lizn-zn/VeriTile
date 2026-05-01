/-
VeriTile.Examples.FlashAttention1

FlashAttention-1 forward kernel — DSL definition + math model + full
correctness theorem.

v0 scope:
  * One `program_id` ↔ one Q-row block of `[BLOCK_M, D]`.
  * K, V are full `[S, D]` matrices in memory (single-block-output).
  * Inner KV-block loop with the canonical online-softmax update
    (running `m_i`, `l_i`, `O`).
  * Arbitrary user-supplied `scale : ℝ` (the standard `1/√D` Triton
    specialization is the caller's responsibility).
  * **Non-causal** — causal masking (uses `tl.where`) is a follow-up.

`fa1_forward_correct` is fully proven end-to-end via the four-stage
decomposition (math identity + pre-loop init + loop step via
`forLoop_inv` + post-loop readout) described above the theorem.
-/

import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.DSL
import VeriTile.Triton.LoopInvariant
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
`attentionReal` directly through `Tile.dot` / `softmaxRow`. This is the
specification-side identity (no streaming algebra involved). -/
theorem oFree_div_lFree_eq_attentionReal {M D Bk N : Nat}
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [Bk * N, D] → ℝ)
    (scale : ℝ) (idx : TileIndex [M, D])
    (_hlFree : lFree Q K scale N (le_refl N) idx.1 ≠ 0) :
    oFree Q K V scale N (le_refl N) idx /
        lFree Q K scale N (le_refl N) idx.1
      = attentionReal Q K V scale idx := by
  rw [oFree_eq_flat, lFree_eq_flat]
  unfold attentionReal attention
  rw [Tile.dot_nil_data]
  -- Match the WithBot Σ to a coerced real Σ via Finset.sum_congr +
  -- softmaxRow_scaled_data_eq, then push the coercion out via map_sum,
  -- unbotD it, and factor the constant denominator with Finset.sum_div.
  rw [Finset.sum_congr rfl (g := fun k : Fin (Bk * N) =>
        (((Real.exp (scaledScore Q K scale idx.1 k) /
          Finset.univ.sum (fun j' : Fin (Bk * N) =>
            Real.exp (scaledScore Q K scale idx.1 j')) *
          V (k, idx.2.1, PUnit.unit)) : ℝ) : WithBot ℝ))
    (by
      intro k _
      let scaled : Tile .real [M, Bk * N] :=
        { data := fun idx =>
            Option.map (fun x => x * scale)
              ((Tile.dot [] (Tile.ofReal Q)
                (Tile.transpose [] (Tile.ofReal K))).data idx) }
      change
        Option.map₂ (fun x1 x2 => x1 * x2)
          ((softmaxRow scaled).data (idx.1, k, PUnit.unit))
          (some (V (k, idx.2.1, PUnit.unit))) =
        (((Real.exp (scaledScore Q K scale idx.1 k) /
            Finset.univ.sum (fun j' : Fin (Bk * N) =>
              Real.exp (scaledScore Q K scale idx.1 j')) *
            V (k, idx.2.1, PUnit.unit)) : ℝ) : WithBot ℝ)
      have hsoft := softmaxRow_scaled_data_eq Q K scale idx.1 k
      change
        (softmaxRow scaled).data (idx.1, k, PUnit.unit) =
          ((Real.exp (scaledScore Q K scale idx.1 k) /
            Finset.univ.sum (fun j' : Fin (Bk * N) =>
              Real.exp (scaledScore Q K scale idx.1 j')) : ℝ) : WithBot ℝ) at hsoft
      rw [hsoft]
      rfl)]
  rw [WithBot.sum_some_eq_some]
  rw [WithBot.unbotD_coe]
  rw [Finset.sum_div]
  apply Finset.sum_congr rfl
  intro k _
  ring

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

/-- The final streaming normalizer is non-zero under FA-1's non-empty KV
scope. This is the denominator fact needed by the readout stage. -/
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

end FA1Math

/-! ## Operational layer — `P_fa1` invariant + four-stage proof

Carries Stage A's streaming functions through the kernel as a
`forLoop_inv` invariant, the way `Examples/OnlineSoftmax.lean` does
for `onlineSoftmaxM` / `onlineSoftmaxL`. The three pieces that combine
via `forLoop_inv` to discharge `fa1_forward_correct` are:

* `fa1_preLoop_correct` — the post-pre-loop state satisfies `P_fa1 0`.
* `fa1_step` — the loop body preserves `P_fa1` under `n → n+1`.
* `fa1_postLoop_correct` — given `P_fa1 numKVBlocks s`, the
  post-store memory at `outReg` equals `attentionReal Q K V scale idx`,
  bridging through `streaming_eq_attentionReal`. -/

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

/-- Statements before the KV-block loop: pid/offset setup, Q-block load,
and accumulator initialization. -/
private def fa1PreLoop (qReg : RegionName) (M D : Nat) : List Stmt :=
  [ Stmt.assign .nat [] "pid" (Op.programId 0)
  , Stmt.assign .nat [M] "offs_m"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil
          (Op.ref .nat [] "pid")
          (Op.constNat M))
        (Op.arange M))
  , Stmt.assign .nat [D] "offs_d" (Op.arange D)
  , Stmt.assign .nat [M, D] "q_ptrs"
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.mul .nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "offs_m"))
          (Op.constNat D))
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d")))
  , Stmt.assign .real [M, D] "q"
      (Op.load qReg (Op.ref .nat [M, D] "q_ptrs"))
  , Stmt.assign .real [M] "m_i"
      (Op.full [M] Op.negInf)
  , Stmt.assign .real [M] "l_i"
      (Op.full [M] (Op.const 0))
  , Stmt.assign .real [M, D] "o_acc"
      (Op.full [M, D] (Op.const 0))
  ]

/-- Body of the inner KV-block loop in `fa1ForwardKernel`, factored out so
the loop invariant proof can name the operational step directly. -/
private def fa1LoopBody (kReg vReg : RegionName)
    (M D Bk : Nat) (scale : ℝ) : List Stmt :=
  [ Stmt.assign .nat [Bk] "offs_n"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil
          (Op.ref .nat [] "n")
          (Op.constNat Bk))
        (Op.arange Bk))
  , Stmt.assign .nat [Bk, D] "k_ptrs"
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.mul .nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [Bk] "offs_n"))
          (Op.constNat D))
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d")))
  , Stmt.assign .nat [Bk, D] "v_ptrs"
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.mul .nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [Bk] "offs_n"))
          (Op.constNat D))
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d")))
  , Stmt.assign .real [Bk, D] "k"
      (Op.load kReg (Op.ref .nat [Bk, D] "k_ptrs"))
  , Stmt.assign .real [Bk, D] "v"
      (Op.load vReg (Op.ref .nat [Bk, D] "v_ptrs"))
  , Stmt.assign .real [M, Bk] "scores"
      (Op.mul .real Broadcast.scalarR
        (Op.dot (batch := []) (M := M) (K := D) (N := Bk)
          (Op.ref .real [M, D] "q")
          (Op.transpose (batch := []) (M := Bk) (N := D)
            (Op.ref .real [Bk, D] "k")))
        (Op.const scale))
  , Stmt.assign .real [M] "m_block"
      (Op.reduceMax (shape := [M, Bk]) ⟨1, by simp⟩ (keepDims := Bool.false)
        (Op.ref .real [M, Bk] "scores"))
  , Stmt.assign .real [M] "m_new"
      (Op.max2 (Broadcast.consSame Broadcast.nil)
        (Op.ref .real [M] "m_i")
        (Op.ref .real [M] "m_block"))
  , Stmt.assign .real [M] "alpha"
      (Op.exp
        (Op.sub .real (Broadcast.consSame Broadcast.nil)
          (Op.ref .real [M] "m_i")
          (Op.ref .real [M] "m_new")))
  , Stmt.assign .real [M, Bk] "p"
      (Op.exp
        (Op.sub .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
          (Op.ref .real [M, Bk] "scores")
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [M] "m_new"))))
  , Stmt.assign .real [M] "l_new"
      (Op.add .real (Broadcast.consSame Broadcast.nil)
        (Op.mul .real (Broadcast.consSame Broadcast.nil)
          (Op.ref .real [M] "alpha")
          (Op.ref .real [M] "l_i"))
        (Op.reduceSum (shape := [M, Bk]) ⟨1, by simp⟩ (keepDims := Bool.false)
          (Op.ref .real [M, Bk] "p")))
  , Stmt.assign .real [M, D] "o_acc"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.mul .real (Broadcast.consSame (Broadcast.consL Broadcast.nil))
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [M] "alpha"))
          (Op.ref .real [M, D] "o_acc"))
        (Op.dot (batch := []) (M := M) (K := Bk) (N := D)
          (Op.ref .real [M, Bk] "p")
          (Op.ref .real [Bk, D] "v")))
  , Stmt.assign .real [M] "m_i"
      (Op.ref .real [M] "m_new")
  , Stmt.assign .real [M] "l_i"
      (Op.ref .real [M] "l_new")
  ]

/-- Initialization stage: the pre-loop statements establish `P_fa1 0`. -/
theorem fa1_preLoop_correct
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg : RegionName)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ) (s : BlockState)
    (hQ : InputAt s qReg
        (Offset.rowMajor2D (rows := M) (cols := D) (s.pid * M * D) D) Q)
    (hK : InputAt s kReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) K)
    (hV : InputAt s vReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) V) :
    ∃ s0,
      stepStmts (fa1PreLoop qReg M D) s = some s0 ∧
      P_fa1 qReg kReg vReg s.pid Q K V scale 0 s0 := by
  let qPtrs : Tile .nat [M, D] :=
    ⟨fun idx => (s.pid * M + idx.1.val) * D + idx.2.1.val⟩
  let qLoaded : Tile .real [M, D] :=
    ⟨fun idx => some (s.readMem qReg ((s.pid * M + idx.1.val) * D + idx.2.1.val))⟩
  let s0 :=
    ((((((((s.setReg "pid" .nat [] (Tile.scalar s.pid))
      ).setReg "offs_m" .nat [M] (Tile.vec fun i : Fin M => s.pid * M + i.val)
      ).setReg "offs_d" .nat [D] (Tile.vec fun d : Fin D => d.val)
      ).setReg "q_ptrs" .nat [M, D] qPtrs
      ).setReg "q" .real [M, D] qLoaded
      ).setReg "m_i" .real [M] ⟨fun _ => (⊥ : WithBot ℝ)⟩
      ).setReg "l_i" .real [M] (Tile.ofReal fun _ => 0)
      ).setReg "o_acc" .real [M, D] (Tile.ofReal fun _ => 0)
  have hQ_loaded_eq : qLoaded = Tile.ofReal Q := by
    ext idx
    simp [qLoaded, Tile.ofReal, BlockState.readMem]
    rw [show (s.pid * M + idx.1.val) * D + idx.2.1.val =
        Offset.rowMajor2D (rows := M) (cols := D) (s.pid * M * D) D idx by
          simp [Offset.rowMajor2D, Offset.strided, Nat.add_mul, Nat.mul_assoc,
            Nat.add_assoc]]
    exact congrArg some (hQ idx)
  refine ⟨s0, ?_, ?_⟩
  · simp [fa1PreLoop, stepStmts, stepStmt, evalOp, Tile.bop, Tile.expandDim,
      NumericDType.add, NumericDType.mul, Option.bind, TileShape.dropInsertedIndex,
      BlockState.readMem, Tile.vec, Tile.ofReal, qPtrs, qLoaded, s0]
    rfl
  · refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · simp [s0]
    · simp [s0]
    · simp [s0]
    · simp [s0]
    · simp [s0, hQ_loaded_eq]
    · simp [s0, FA1Math.mPartial]
    · simp [s0, FA1Math.lPartial, Tile.ofReal]
    · simp [s0, FA1Math.oPartial, Tile.ofReal]
    · intro idx
      simpa [s0] using hQ idx
    · intro idx
      simpa [s0] using hK idx
    · intro idx
      simpa [s0] using hV idx

/-- The three statements after the FA-1 KV-block loop: normalize the
accumulator, rebuild output pointers, and store the `[M, D]` tile. Factored
out so the readout proof can be checked independently of the loop proof. -/
private def fa1PostLoop (outReg : RegionName) (M D : Nat) : List Stmt :=
  [ Stmt.assign .real [M, D] "out"
      (Op.div .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [M, D] "o_acc")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [M] "l_i")))
  , Stmt.assign .nat [M, D] "o_ptrs"
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.mul .nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "offs_m"))
          (Op.constNat D))
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d")))
  , Stmt.store outReg [M, D]
      (Op.ref .nat [M, D] "o_ptrs")
      (Op.ref .real [M, D] "out")
  ]

/-- Readout stage: once the loop invariant holds at `numKVBlocks`, the
post-loop normalization and store produce the ℝ-level attention spec. -/
theorem fa1_postLoop_correct
    {M D Bk numKVBlocks : Nat}
    (hBk : 0 < Bk) (hNumKVBlocks : 0 < numKVBlocks)
    (qReg kReg vReg outReg : RegionName)
    (origPid : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ) (sLoop : BlockState)
    (hP : P_fa1 qReg kReg vReg origPid Q K V scale numKVBlocks sLoop) :
    ∀ idx : TileIndex [M, D],
      observeTileAt
          (stepStmts (fa1PostLoop outReg M D) sLoop)
          outReg
          (Offset.rowMajor2D (rows := M) (cols := D) (origPid * M * D) D) idx
        = some (attentionReal Q K V scale idx) := by
  intro idx
  rcases hP with
    ⟨_hpidReg, _hpid, hoffs_m, hoffs_d, _hq, _hm, hl, ho, _hQ, _hK, _hV⟩
  have h_inj :
      Function.Injective
        (Offset.rowMajor2D (rows := M) (cols := D) (origPid * M * D) D) :=
    Offset.rowMajor2D_inj (base := origPid * M * D) (rowStride := D) (le_refl D)
  have h_inj_store :
      Function.Injective
        (fun i : TileIndex [M, D] => (origPid * M + i.1.val) * D + i.2.1.val) := by
    intro a b h
    apply h_inj
    simpa [Offset.rowMajor2D, Offset.strided, Nat.add_mul, Nat.mul_assoc,
      Nat.add_assoc] using h
  simp [observeTileAt, fa1PostLoop, stepStmts, stepStmt, evalOp,
        BlockState.setReg, Tile.ofReal, hoffs_m, hoffs_d, hl, ho,
        Tile.bop, Tile.expandDim, NumericDType.add, NumericDType.mul,
        NumericDType.div, Offset.rowMajor2D, Offset.strided, Option.bind,
        TileShape.dropInsertedIndex]
  rw [show origPid * M * D + idx.1.val * D + idx.2.1.val =
      (origPid * M + idx.1.val) * D + idx.2.1.val by
        rw [Nat.add_mul]]
  simp only [BlockState.readMem]
  rw [BlockState.scatter_readback_nd _ _ _ h_inj_store idx]
  simp [FA1Math.streaming_eq_attentionReal hBk Q numKVBlocks hNumKVBlocks K V scale idx
        (FA1Math.lPartial_final_ne_zero hBk Q numKVBlocks hNumKVBlocks K scale idx.1)]

/-- The DSL-expanded kernel body is exactly the factored operational shape:
pre-loop setup, one `forLoop`, then post-loop readout. -/
@[simp] theorem fa1ForwardKernel_body_eq
    (qReg kReg vReg outReg : RegionName)
    (M D Bk numKVBlocks : Nat) (scale : ℝ) :
    (fa1ForwardKernel qReg kReg vReg outReg M D Bk numKVBlocks scale).body =
      fa1PreLoop qReg M D ++
      [Stmt.forLoop "n" numKVBlocks (fa1LoopBody kReg vReg M D Bk scale)] ++
      fa1PostLoop outReg M D := by
  rfl

/-- Reading the `n`-th KV block through the row-major address expression
used by the loop gives the corresponding `blockIndex` cell of a full
`[Bk * numKVBlocks, D]` input. -/
theorem fa1_block_read
    {D Bk numKVBlocks : Nat}
    (region : RegionName) (s : BlockState)
    (X : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (hX : InputAt s region
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) X)
    (n : Nat) (hn : n < numKVBlocks)
    (j : Fin Bk) (d : Fin D) :
    s.readMem region ((n * Bk + j.val) * D + d.val) =
      X (FA1Math.blockIndex Bk numKVBlocks n (Nat.succ_le_iff.mpr hn) j,
        d, PUnit.unit) := by
  rw [BlockState.readMem]
  have haddr :
      (n * Bk + j.val) * D + d.val =
        Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D
          (FA1Math.blockIndex Bk numKVBlocks n (Nat.succ_le_iff.mpr hn) j,
            d, PUnit.unit) := by
    simp [Offset.rowMajor2D, Offset.strided, FA1Math.blockIndex]
  rw [haddr]
  exact hX _

/-- Tile-level version of `fa1_block_read`: the loop-local block load is the
`Tile.ofReal` view of the corresponding full K/V matrix block. -/
theorem fa1_block_load_tile_eq
    {D Bk numKVBlocks : Nat}
    (region : RegionName) (s : BlockState)
    (X : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (hX : InputAt s region
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) X)
    (n : Nat) (hn : n < numKVBlocks) :
    (⟨fun idx : TileIndex [Bk, D] =>
        some (s.readMem region ((n * Bk + idx.1.val) * D + idx.2.1.val))⟩
      : Tile .real [Bk, D])
      =
      Tile.ofReal (fun idx : TileIndex [Bk, D] =>
        X (FA1Math.blockIndex Bk numKVBlocks n (Nat.succ_le_iff.mpr hn) idx.1,
          idx.2.1, PUnit.unit)) := by
  ext idx
  rw [Tile.ofReal_data]
  exact congrArg some (fa1_block_read region s X hX n hn idx.1 idx.2.1)

set_option maxHeartbeats 800000 in
/-- One FA-1 KV-block iteration preserves the loop invariant. -/
theorem fa1_step
    {M D Bk numKVBlocks : Nat} (hBk : 0 < Bk)
    (qReg kReg vReg : RegionName)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ) (origPid k : Nat) (s : BlockState)
    (hk : k < numKVBlocks)
    (hP : P_fa1 qReg kReg vReg origPid Q K V scale k s) :
    ∃ s',
      stepStmts (fa1LoopBody kReg vReg M D Bk scale)
        (s.setReg "n" .nat [] (Tile.scalar k)) = some s' ∧
      P_fa1 qReg kReg vReg origPid Q K V scale (k + 1) s' := by
  rcases hP with
    ⟨hpidReg, hpid, hoffs_m, hoffs_d, hq, hm, hl, ho, hQ, hK, hV⟩
  let offsN : Tile .nat [Bk] :=
    Tile.vec fun j : Fin Bk => k * Bk + j.val
  let ptrs : Tile .nat [Bk, D] :=
    ⟨fun idx : TileIndex [Bk, D] => (k * Bk + idx.1.val) * D + idx.2.1.val⟩
  let kTile : Tile .real [Bk, D] :=
    Tile.ofReal fun idx : TileIndex [Bk, D] =>
      K (FA1Math.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) idx.1,
        idx.2.1, PUnit.unit)
  let vTile : Tile .real [Bk, D] :=
    Tile.ofReal fun idx : TileIndex [Bk, D] =>
      V (FA1Math.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) idx.1,
        idx.2.1, PUnit.unit)
  let scores : Tile .real [M, Bk] :=
    Tile.ofReal fun idx : TileIndex [M, Bk] =>
      FA1Math.scaledScore Q K scale idx.1
        (FA1Math.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) idx.2.1)
  let mBlock : Tile .real [M] :=
    ⟨fun idx : TileIndex [M] =>
      (Finset.univ : Finset (Fin Bk)).sup
        (fun j => ((FA1Math.scaledScore Q K scale idx.1
          (FA1Math.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) j)
          : ℝ) : WithBot ℝ))⟩
  let mNew : Tile .real [M] :=
    ⟨fun idx : TileIndex [M] =>
      FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) idx.1⟩
  let alpha : Tile .real [M] :=
    Tile.ofReal fun idx : TileIndex [M] =>
      FA1Math.alphaPartial Q numKVBlocks K scale k idx.1
  let p : Tile .real [M, Bk] :=
    Tile.ofReal fun idx : TileIndex [M, Bk] =>
      Real.exp (FA1Math.scaledScore Q K scale idx.1
        (FA1Math.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) idx.2.1) -
          (FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) idx.1).unbotD 0)
  let lNew : Tile .real [M] :=
    Tile.ofReal fun idx : TileIndex [M] =>
      FA1Math.lPartial Q numKVBlocks K scale (k + 1) idx.1
  let oNew : Tile .real [M, D] :=
    Tile.ofReal fun idx : TileIndex [M, D] =>
      FA1Math.oPartial Q numKVBlocks K V scale (k + 1) idx
  let s0 := s.setReg "n" .nat [] (Tile.scalar k)
  let s1 := s0.setReg "offs_n" .nat [Bk] offsN
  let s2 := s1.setReg "k_ptrs" .nat [Bk, D] ptrs
  let s3 := s2.setReg "v_ptrs" .nat [Bk, D] ptrs
  let s4 := s3.setReg "k" .real [Bk, D] kTile
  let s5 := s4.setReg "v" .real [Bk, D] vTile
  let s6 := s5.setReg "scores" .real [M, Bk] scores
  let s7 := s6.setReg "m_block" .real [M] mBlock
  let s8 := s7.setReg "m_new" .real [M] mNew
  let s9 := s8.setReg "alpha" .real [M] alpha
  let s10 := s9.setReg "p" .real [M, Bk] p
  let s11 := s10.setReg "l_new" .real [M] lNew
  let s12 := s11.setReg "o_acc" .real [M, D] oNew
  let s13 := s12.setReg "m_i" .real [M] mNew
  let s' := s13.setReg "l_i" .real [M] lNew
  apply Exists.intro
  constructor
  · simp [fa1LoopBody, stepStmts, stepStmt, evalOp, Tile.bop, Tile.expandDim,
      Tile.transpose, Tile.dot, Tile.reduceMax, Tile.reduceMaxDrop,
      Tile.reduceSum, Tile.reduceSumDrop, TileShape.axisDim,
      TileShape.eraseAxis, TileShape.insertAxisIndex, TileShape.dropInsertedIndex,
      NumericDType.add, NumericDType.mul, NumericDType.sub,
      BlockState.readMem, Option.bind, hBk, hoffs_d, hq, hm, hl, ho]
    have hKmem : ∀ (j : Fin Bk) (d : Fin D),
        s.mem kReg ((k * Bk + j.val) * D + d.val) =
          K (FA1Math.blockIndex Bk numKVBlocks k
            (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit) :=
      fa1_block_read kReg s K hK k hk
    have hVmem : ∀ (j : Fin Bk) (d : Fin D),
        s.mem vReg ((k * Bk + j.val) * D + d.val) =
          V (FA1Math.blockIndex Bk numKVBlocks k
            (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit) :=
      fa1_block_read vReg s V hV k hk
    simp_rw [hKmem, hVmem]
    rfl
  · have hmNewData : ∀ i : Fin M,
        max (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
          (some ((Finset.univ : Finset (Fin Bk)).sup'
            (by exact ⟨⟨0, hBk⟩, Finset.mem_univ _⟩)
            (fun j =>
              (Finset.univ.sum (fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (FA1Math.blockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale))) =
          FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i := by
      intro i
      rw [FA1Math.mPartial_succ_of_lt Q numKVBlocks K scale k hk i]
      congr 1
      change ((((Finset.univ : Finset (Fin Bk)).sup'
          (by exact ⟨⟨0, hBk⟩, Finset.mem_univ _⟩)
          (fun j => (Finset.univ.sum (fun d : Fin D =>
            Q (i, d, PUnit.unit) *
              K (FA1Math.blockIndex Bk numKVBlocks k
                (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)) : ℝ) : WithBot ℝ) =
        (Finset.univ : Finset (Fin Bk)).sup (fun j =>
          ((FA1Math.scaledScore Q K scale i
            (FA1Math.blockIndex Bk numKVBlocks k
              (Nat.succ_le_iff.mpr hk) j) : ℝ) : WithBot ℝ))
      rw [← WithBot.sup'_coe]
      rw [Finset.sup'_eq_sup]
      apply Finset.sup_congr rfl
      intro j _
      simp [FA1Math.scaledScore, mul_comm]
    have hmNewDataOf : ∀ (i : Fin M)
        (H : (Finset.univ : Finset (Fin Bk)).Nonempty),
        (FA1Math.mPartial Bk Q numKVBlocks K scale k i) ⊔
          (some ((Finset.univ : Finset (Fin Bk)).sup' H
            (fun j =>
              (Finset.univ.sum (fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (FA1Math.blockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale))) =
          FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i := by
      intro i H
      rw [FA1Math.mPartial_succ_of_lt Q numKVBlocks K scale k hk i]
      congr 1
      change ((((Finset.univ : Finset (Fin Bk)).sup' H
          (fun j => (Finset.univ.sum (fun d : Fin D =>
            Q (i, d, PUnit.unit) *
              K (FA1Math.blockIndex Bk numKVBlocks k
                (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)) : ℝ) : WithBot ℝ) =
        (Finset.univ : Finset (Fin Bk)).sup (fun j =>
          ((FA1Math.scaledScore Q K scale i
            (FA1Math.blockIndex Bk numKVBlocks k
              (Nat.succ_le_iff.mpr hk) j) : ℝ) : WithBot ℝ))
      rw [← WithBot.sup'_coe]
      rw [Finset.sup'_eq_sup]
      apply Finset.sup_congr rfl
      intro j _
      simp [FA1Math.scaledScore, mul_comm]
    have hmNewDataComm : ∀ i : Fin M,
        max (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
          (some ((Finset.univ : Finset (Fin Bk)).sup'
            (by exact ⟨⟨0, hBk⟩, Finset.mem_univ _⟩)
            (fun j =>
              scale * (Finset.univ.sum (fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (FA1Math.blockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit)))))) =
          FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i := by
      intro i
      rw [← hmNewData i]
      congr 2
      apply Finset.sup'_congr (H := by exact ⟨⟨0, hBk⟩, Finset.mem_univ _⟩) rfl
      intro j _
      rw [mul_comm]
    have hAlphaData : ∀ i : Fin M,
        WithBot.realExp
          (Option.map₂ (fun x1 x2 => x1 - x2)
            (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
            (FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i))
          =
          some (FA1Math.alphaPartial Q numKVBlocks K scale k i) := by
      intro i
      simpa [WithBot.realSub] using
        FA1Math.alphaPartial_toWithBot Q numKVBlocks K scale k i
    have hPData : ∀ (i : Fin M) (j : Fin Bk),
        WithBot.realExp
          (Option.map
            (fun b =>
              (Finset.univ.sum (fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (FA1Math.blockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale - b)
            (FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i))
          =
          some (Real.exp (FA1Math.scaledScore Q K scale i
            (FA1Math.blockIndex Bk numKVBlocks k
              (Nat.succ_le_iff.mpr hk) j) -
            (FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0)) := by
      intro i j
      have hm : FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i ≠ ⊥ :=
        FA1Math.mPartial_succ_ne_bot hBk Q numKVBlocks K scale k
          (Nat.succ_le_iff.mpr hk) i
      obtain ⟨m, hm_eq⟩ := WithBot.ne_bot_iff_exists.mp hm
      rw [← hm_eq]
      simp [FA1Math.scaledScore, mul_comm]
    have hAlphaDataOf : ∀ (i : Fin M)
        (H : (Finset.univ : Finset (Fin Bk)).Nonempty),
        WithBot.realExp
          (Option.map₂ (fun x1 x2 => x1 - x2)
            (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (@max (Option ℝ) Option.instMax
                (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
                (some ((Finset.univ : Finset (Fin Bk)).sup' H
                  (fun j =>
                    (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (FA1Math.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)))))
          =
          some (FA1Math.alphaPartial Q numKVBlocks K scale k i) := by
      intro i H
      change WithBot.realExp
          (Option.map₂ (fun x1 x2 => x1 - x2)
            (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (@max (Option ℝ) Option.instMax
                (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
                (some ((Finset.univ : Finset (Fin Bk)).sup' H
                  (fun j =>
                    (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (FA1Math.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)))))
          =
          some (FA1Math.alphaPartial Q numKVBlocks K scale k i)
      have hM :
          @max (Option ℝ) Option.instMax
            (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
            (some ((Finset.univ : Finset (Fin Bk)).sup' H
              (fun j =>
                (Finset.univ.sum (fun d : Fin D =>
                  Q (i, d, PUnit.unit) *
                    K (FA1Math.blockIndex Bk numKVBlocks k
                      (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale))) =
            FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i := by
        have hOptionMax :
            @max (Option ℝ) Option.instMax
              (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (some ((Finset.univ : Finset (Fin Bk)).sup' H
                (fun j =>
                  (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (FA1Math.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)))
            =
            max (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (some ((Finset.univ : Finset (Fin Bk)).sup' H
                (fun j =>
                  (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (FA1Math.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale))) := by
          cases FA1Math.mPartial Bk Q numKVBlocks K scale k i <;> rfl
        rw [hOptionMax]
        exact hmNewDataOf i H
      exact (congrArg
        (fun z => WithBot.realExp
          (Option.map₂ (fun x1 x2 => x1 - x2)
            (FA1Math.mPartial Bk Q numKVBlocks K scale k i) z)) hM).trans
        (hAlphaData i)
    have hPDataOf : ∀ (i : Fin M) (j : Fin Bk)
        (H : (Finset.univ : Finset (Fin Bk)).Nonempty),
      WithBot.realExp
        (Option.map
          (fun b =>
            (Finset.univ.sum (fun d : Fin D =>
              Q (i, d, PUnit.unit) *
                K (FA1Math.blockIndex Bk numKVBlocks k
                  (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale - b)
            (@max (Option ℝ) Option.instMax
              (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (some ((Finset.univ : Finset (Fin Bk)).sup' H
                (fun j =>
                  (Finset.univ.sum (fun d : Fin D =>
                  Q (i, d, PUnit.unit) *
                    K (FA1Math.blockIndex Bk numKVBlocks k
                      (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)))))
        =
        some (Real.exp (FA1Math.scaledScore Q K scale i
          (FA1Math.blockIndex Bk numKVBlocks k
            (Nat.succ_le_iff.mpr hk) j) -
          (FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0)) := by
      intro i j H
      change WithBot.realExp
          (Option.map
            (fun b =>
              (Finset.univ.sum (fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (FA1Math.blockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale - b)
              (@max (Option ℝ) Option.instMax
                (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
                (some ((Finset.univ : Finset (Fin Bk)).sup' H
                  (fun j =>
                    (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (FA1Math.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)))))
          =
          some (Real.exp (FA1Math.scaledScore Q K scale i
            (FA1Math.blockIndex Bk numKVBlocks k
              (Nat.succ_le_iff.mpr hk) j) -
            (FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0))
      have hM :
          @max (Option ℝ) Option.instMax
            (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
            (some ((Finset.univ : Finset (Fin Bk)).sup' H
              (fun j =>
                (Finset.univ.sum (fun d : Fin D =>
                  Q (i, d, PUnit.unit) *
                    K (FA1Math.blockIndex Bk numKVBlocks k
                      (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale))) =
            FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i := by
        have hOptionMax :
            @max (Option ℝ) Option.instMax
              (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (some ((Finset.univ : Finset (Fin Bk)).sup' H
                (fun j =>
                  (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (FA1Math.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)))
            =
            max (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (some ((Finset.univ : Finset (Fin Bk)).sup' H
                (fun j =>
                  (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (FA1Math.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale))) := by
          cases FA1Math.mPartial Bk Q numKVBlocks K scale k i <;> rfl
        rw [hOptionMax]
        exact hmNewDataOf i H
      exact (congrArg
        (fun z => WithBot.realExp
          (Option.map
            (fun b =>
              (Finset.univ.sum (fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (FA1Math.blockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale - b) z)) hM).trans
          (hPData i j)
    have hAlphaDataComm : ∀ (i : Fin M)
        (H : (Finset.univ : Finset (Fin Bk)).Nonempty),
        WithBot.realExp
          (Option.map₂ (fun x1 x2 => x1 - x2)
            (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (@max (Option ℝ) Option.instMax
                (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
                (some ((Finset.univ : Finset (Fin Bk)).sup' H
                  (fun j =>
                    scale * (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (FA1Math.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))))))))
          =
          some (FA1Math.alphaPartial Q numKVBlocks K scale k i) := by
      intro i H
      have hSup :
          ((Finset.univ : Finset (Fin Bk)).sup' H
            (fun j =>
              scale * (Finset.univ.sum (fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (FA1Math.blockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))))) =
          ((Finset.univ : Finset (Fin Bk)).sup' H
            (fun j =>
              (Finset.univ.sum (fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (FA1Math.blockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)) := by
        apply Finset.sup'_congr (H := H) rfl
        intro j _
        rw [mul_comm]
      rw [hSup]
      exact hAlphaDataOf i H
    have hPDataComm : ∀ (i : Fin M) (j : Fin Bk)
        (H : (Finset.univ : Finset (Fin Bk)).Nonempty),
      WithBot.realExp
        (Option.map
          (fun b =>
            scale * (Finset.univ.sum (fun d : Fin D =>
              Q (i, d, PUnit.unit) *
                K (FA1Math.blockIndex Bk numKVBlocks k
                  (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) - b)
            (@max (Option ℝ) Option.instMax
              (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (some ((Finset.univ : Finset (Fin Bk)).sup' H
                (fun j =>
                  scale * (Finset.univ.sum (fun d : Fin D =>
                  Q (i, d, PUnit.unit) *
                    K (FA1Math.blockIndex Bk numKVBlocks k
                      (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))))))))
        =
        some (Real.exp (FA1Math.scaledScore Q K scale i
          (FA1Math.blockIndex Bk numKVBlocks k
            (Nat.succ_le_iff.mpr hk) j) -
          (FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0)) := by
      intro i j H
      have hSup :
          ((Finset.univ : Finset (Fin Bk)).sup' H
            (fun j =>
              scale * (Finset.univ.sum (fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (FA1Math.blockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))))) =
          ((Finset.univ : Finset (Fin Bk)).sup' H
            (fun j =>
              (Finset.univ.sum (fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (FA1Math.blockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)) := by
        apply Finset.sup'_congr (H := H) rfl
        intro j _
        rw [mul_comm]
      rw [hSup]
      simpa [mul_comm] using hPDataOf i j H
    have hAlphaDataMaxComm : ∀ (i : Fin M)
        (H : (Finset.univ : Finset (Fin Bk)).Nonempty),
        WithBot.realExp
          (Option.map₂ (fun x1 x2 => x1 - x2)
            (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (@max (Option ℝ) Option.instMax
                (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
                (some ((Finset.univ : Finset (Fin Bk)).sup' H
                  (fun j =>
                    scale * (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (FA1Math.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))))))))
          =
          some (FA1Math.alphaPartial Q numKVBlocks K scale k i) := by
      intro i H
      exact hAlphaDataComm i H
    have hPDataMaxComm : ∀ (i : Fin M) (j : Fin Bk)
        (H : (Finset.univ : Finset (Fin Bk)).Nonempty),
      WithBot.realExp
        (Option.map
          (fun b =>
            scale * (Finset.univ.sum (fun d : Fin D =>
              Q (i, d, PUnit.unit) *
                K (FA1Math.blockIndex Bk numKVBlocks k
                  (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) - b)
            (@max (Option ℝ) Option.instMax
              (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (some ((Finset.univ : Finset (Fin Bk)).sup' H
                (fun j =>
                  scale * (Finset.univ.sum (fun d : Fin D =>
                  Q (i, d, PUnit.unit) *
                    K (FA1Math.blockIndex Bk numKVBlocks k
                      (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))))))))
        =
        some (Real.exp (FA1Math.scaledScore Q K scale i
          (FA1Math.blockIndex Bk numKVBlocks k
            (Nat.succ_le_iff.mpr hk) j) -
          (FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0)) := by
      intro i j H
      exact hPDataComm i j H
    have hAlphaDataMaxComm' : ∀ (i : Fin M)
        (H : (Finset.univ : Finset (Fin Bk)).Nonempty),
        WithBot.realExp
          (Option.map₂ (fun x1 x2 => x1 - x2)
            (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (max (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
                (some ((Finset.univ : Finset (Fin Bk)).sup' H
                  (fun j =>
                    scale * (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (FA1Math.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))))))))
          =
          some (FA1Math.alphaPartial Q numKVBlocks K scale k i) := by
      intro i H
      exact hAlphaDataComm i H
    have hPDataMaxComm' : ∀ (i : Fin M) (j : Fin Bk)
        (H : (Finset.univ : Finset (Fin Bk)).Nonempty),
      WithBot.realExp
        (Option.map
          (fun b =>
            scale * (Finset.univ.sum (fun d : Fin D =>
              Q (i, d, PUnit.unit) *
                K (FA1Math.blockIndex Bk numKVBlocks k
                  (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) - b)
            (max (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (some ((Finset.univ : Finset (Fin Bk)).sup' H
                (fun j =>
                  scale * (Finset.univ.sum (fun d : Fin D =>
                  Q (i, d, PUnit.unit) *
                    K (FA1Math.blockIndex Bk numKVBlocks k
                      (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))))))))
        =
        some (Real.exp (FA1Math.scaledScore Q K scale i
          (FA1Math.blockIndex Bk numKVBlocks k
            (Nat.succ_le_iff.mpr hk) j) -
          (FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0)) := by
      intro i j H
      exact hPDataComm i j H
    have hAlphaDataMaxOf : ∀ (i : Fin M)
        (H : (Finset.univ : Finset (Fin Bk)).Nonempty),
        WithBot.realExp
          (Option.map₂ (fun x1 x2 => x1 - x2)
            (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (max (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
                (some ((Finset.univ : Finset (Fin Bk)).sup' H
                  (fun j =>
                    (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (FA1Math.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)))))
          =
          some (FA1Math.alphaPartial Q numKVBlocks K scale k i) := by
      intro i H
      exact hAlphaDataOf i H
    have hPDataMaxOf : ∀ (i : Fin M) (j : Fin Bk)
        (H : (Finset.univ : Finset (Fin Bk)).Nonempty),
      WithBot.realExp
        (Option.map
          (fun b =>
            (Finset.univ.sum (fun d : Fin D =>
              Q (i, d, PUnit.unit) *
                K (FA1Math.blockIndex Bk numKVBlocks k
                  (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale - b)
            (max (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (some ((Finset.univ : Finset (Fin Bk)).sup' H
                (fun j =>
                  (Finset.univ.sum (fun d : Fin D =>
                  Q (i, d, PUnit.unit) *
                    K (FA1Math.blockIndex Bk numKVBlocks k
                      (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)))))
        =
        some (Real.exp (FA1Math.scaledScore Q K scale i
          (FA1Math.blockIndex Bk numKVBlocks k
            (Nat.succ_le_iff.mpr hk) j) -
          (FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0)) := by
      intro i j H
      exact hPDataOf i j H
    have max_eq_sup : ∀ (a b : WithBot ℝ), max a b = a ⊔ b := by
      intro a b
      rfl
    refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · simp [hpidReg]
    · simp [hpid]
    · simp [hoffs_m]
    · simp [hoffs_d]
    · simp [hq]
    · simp
      ext idx
      exact hmNewData idx.1
    · rw [← FA1Math.block_lNew_tile_eq Q K scale k hk]
      simp
      ext idx
      simp only [Tile.bop, Tile.ofReal, Tile.uop,
        Broadcast.leftIndex_consSame, Broadcast.rightIndex_consSame,
        Broadcast.leftIndex_nil, Broadcast.rightIndex_nil]
      congr 1
      · -- LHS: Option.map (· * lPartial k) (WithBot.realExp (Option.map₂ - mPartial(k) (max ...)))
        -- RHS: Option.map₂ * (some alphaPartial) (some lPartial k)
        rw [show (Option.map₂ (· * ·) (some (FA1Math.alphaPartial Q numKVBlocks K scale k idx.1))
              (some (FA1Math.lPartial Q numKVBlocks K scale k idx.1)) :
              WithBot ℝ) =
            Option.map (· * FA1Math.lPartial Q numKVBlocks K scale k idx.1)
              (some (FA1Math.alphaPartial Q numKVBlocks K scale k idx.1)) from rfl]
        congr 1
        -- Goal: WithBot.realExp (Option.map₂ - mPartial(k) (max mPartial(k) (some sup'))) =
        --       some alphaPartial.
        -- hAlphaData idx.1 says it's WithBot.realExp (Option.map₂ - mPartial(k) mPartial(k+1)).
        -- We bridge via congrArg.
        refine Eq.trans ?_ (hAlphaData idx.1)
        congr 2
        exact hmNewData idx.1
      · -- LHS: ∑ x, WithBot.realExp (Option.map (... - max ...) ...)
        -- RHS: some (∑ j, Real.exp (...))
        -- Step 1: replace each summand by `some (Real.exp ...)` via the same
        -- congr-2 / hmNewData / hPData pattern used in the alpha branch.
        -- Step 2: collapse `∑ some _ = some (∑ _)` via `WithBot.sum_someTerm_eq_some`.
        refine Eq.trans
          (@Finset.sum_congr (Fin Bk) (WithBot ℝ) _ _ _ _ _ rfl
            (fun j _ => ?_))
          (WithBot.sum_someTerm_eq_some _ _)
        refine Eq.trans ?_ (hPData idx.1 j)
        congr 2
        exact hmNewData idx.1
    · rw [← FA1Math.block_oAcc_tile_eq Q K V scale k hk]
      simp
      ext idx
      simp only [Tile.bop, Tile.ofReal, Tile.uop, Tile.expandDim,
        Broadcast.leftIndex_consSame, Broadcast.rightIndex_consSame,
        Broadcast.leftIndex_consL, Broadcast.rightIndex_consL,
        Broadcast.leftIndex_nil, Broadcast.rightIndex_nil,
        TileShape.dropInsertedIndex]
      congr 1
      · -- LHS: Option.map (· * oPartial k (idx)) (WithBot.realExp (Option.map₂ - mPartial(k) (max ...)))
        -- RHS: Option.map₂ * (some alphaPartial) (some (oPartial k idx))
        rw [show (Option.map₂ (· * ·)
              (some (FA1Math.alphaPartial Q numKVBlocks K scale k idx.1))
              (some (FA1Math.oPartial Q numKVBlocks K V scale k
                (idx.1, idx.2.1, PUnit.unit))) :
              WithBot ℝ) =
            Option.map (· * FA1Math.oPartial Q numKVBlocks K V scale k
                (idx.1, idx.2.1, PUnit.unit))
              (some (FA1Math.alphaPartial Q numKVBlocks K scale k idx.1)) from rfl]
        congr 1
        refine Eq.trans ?_ (hAlphaData idx.1)
        congr 2
        exact hmNewData idx.1
      · -- LHS: ∑ x, Option.map (· * V (...)) (WithBot.realExp (Option.map (... - max ...)))
        -- RHS: some (∑ j, Real.exp (...) * V (...))
        refine Eq.trans
          (@Finset.sum_congr (Fin Bk) (WithBot ℝ) _ _ _ _ _ rfl
            (fun j _ => ?_))
          (WithBot.sum_someTerm_eq_some _ _)
        -- Goal at j: Option.map (· * V (...)) (WithBot.realExp (Option.map (...) (max ...)))
        --           = some (Real.exp (...) * V (...))
        -- Strategy: descend through Option.map (·*V) and WithBot.realExp via `congr 1`s
        -- (the `· * V` factor on the outside aligns with the `· * V` factor in the RHS),
        -- then use the same congr-2 + hmNewData + hPData pattern as the alpha branch.
        -- We can't use `congr 1` directly because LHS has `Option.map` head, RHS has
        -- `some` head. So we first reshape RHS to match: `some y = Option.map (·*V) (some r)`
        -- where `r * V = y`, holds by `rfl`.
        change Option.map _ _ =
            Option.map (fun a : ℝ => a *
              V (FA1Math.blockIndex Bk numKVBlocks k
                (Nat.succ_le_iff.mpr hk) j, idx.2.1, PUnit.unit))
              (some (Real.exp (FA1Math.scaledScore Q K scale idx.1
                (FA1Math.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) j) -
                  WithBot.unbotD 0
                    (FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) idx.1))))
        congr 1
        refine Eq.trans ?_ (hPData idx.1 j)
        congr 2
        exact hmNewData idx.1
    · intro idx
      simpa [s', s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0] using hQ idx
    · intro idx
      simpa [s', s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0] using hK idx
    · intro idx
      simpa [s', s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0] using hV idx

/-! ## Correctness statement + theorem

The statement uses the standard `InputAt` / `observeTileAt` predicates
from `Examples.Common` to relate the kernel's memory effects to the
math model on `Tile`s. The kernel's optional final state is fed
directly into `observeTileAt`; the proof also discharges "`exec`
succeeds" along the way, threading through the four-stage
decomposition. -/

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

**Proof structure.** Follows the same shape as `online_softmax_correct`
in `Examples/OnlineSoftmax.lean`, scaled to FA-1's bigger statement:

```text
1. Stage A math — `streaming_eq_attentionReal`: the streaming
   `mPartial` / `lPartial` / `oPartial` recurrence over KV blocks
   matches the closed-form `attentionReal` softmax-weighted average.

2. Stage B init — `fa1_preLoop_correct`: exec walks through the 8
   pre-loop statements (program_id, two aranges, Q-load, three
   tile-init fulls), each matched by an explicit `stepStmt` rewrite.
   Resulting state satisfies `P_fa1 0`.

3. Stage C step — `fa1_step`: the loop body's 13 statements thread
   evalOp / Tile.dot / Tile.transpose / Tile.expandDim semantics
   through; the math-side recurrence (mPartial / lPartial / oPartial
   at k → k+1) closes the inductive step. Combined with Stage B via
   `forLoop_inv` to obtain `P_fa1 numKVBlocks` at the loop exit.

4. Stage D readout — `fa1_postLoop_correct`: given `P_fa1 numKVBlocks
   s_final`, the post-loop assigns `out := o_acc / l_i[:, None]` and
   `tl.store(outReg + ptrs, out)` realize `streaming_eq_attentionReal`
   at the operational layer; `observeTileAt … = some (attentionReal …)`
   follows.
```

The non-empty hypotheses `0 < Bk` and `0 < numKVBlocks` are semantic,
not proof-only: `tl.max(scores, axis = 1)` is undefined on an empty KV
block in the current operational model, and attention over zero KV
blocks is outside the intended FA-1 v0 scope. -/
theorem fa1_forward_correct
    {M D Bk numKVBlocks : Nat}
    (hBk : 0 < Bk) (hNumKVBlocks : 0 < numKVBlocks)
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
  -- Helper: `stepStmts` distributes over list append.
  have stepStmts_cons : ∀ (st : Stmt) (rest : List Stmt) (sa sb : BlockState),
      stepStmt st sa = some sb →
      stepStmts (st :: rest) sa = stepStmts rest sb := by
    intro st rest sa sb h
    conv_lhs => unfold stepStmts
    rw [h]
  have stepStmts_append : ∀ (l1 l2 : List Stmt) (sa sb : BlockState),
      stepStmts l1 sa = some sb →
      stepStmts (l1 ++ l2) sa = stepStmts l2 sb := by
    intro l1
    induction l1 with
    | nil =>
        intro l2 sa sb h
        conv_lhs at h => unfold stepStmts
        injection h with h
        rw [List.nil_append, ← h]
    | cons st rest ih =>
        intro l2 sa sb h
        conv_lhs at h => unfold stepStmts
        cases hst : stepStmt st sa with
        | none => rw [hst] at h; simp at h
        | some sm =>
            rw [hst] at h
            simp at h
            rw [List.cons_append, stepStmts_cons _ _ _ _ hst]
            exact ih l2 sm sb h
  -- Stage B: pre-loop establishes P_fa1 0.
  obtain ⟨s0, hPre, hP0⟩ :=
    fa1_preLoop_correct qReg kReg vReg Q K V scale s _hQ _hK _hV
  -- Stage C: forLoop_inv chains fa1_step over numKVBlocks iterations.
  obtain ⟨sLoop, hLoopStmt, hPLoop⟩ :=
    forLoop_inv (P := P_fa1 qReg kReg vReg s.pid Q K V scale) hP0
      (fun i st hi hPi => fa1_step hBk qReg kReg vReg Q K V scale s.pid i st hi hPi)
  -- Stage D: post-loop readout matches `attentionReal`.
  intro idx
  -- Reshape `exec` through body = preLoop ++ [forLoop] ++ postLoop.
  show observeTileAt (stepStmts _ s) outReg _ idx = _
  rw [show (fa1ForwardKernel qReg kReg vReg outReg M D Bk numKVBlocks scale).body =
        fa1PreLoop qReg M D ++
        [Stmt.forLoop "n" numKVBlocks (fa1LoopBody kReg vReg M D Bk scale)] ++
        fa1PostLoop outReg M D from rfl]
  -- Walk preLoop: stepStmts (preLoop ++ [forLoop] ++ postLoop) s
  --             = stepStmts ([forLoop] ++ postLoop) s0
  rw [List.append_assoc,
      stepStmts_append (fa1PreLoop qReg M D)
        ([Stmt.forLoop "n" numKVBlocks (fa1LoopBody kReg vReg M D Bk scale)] ++
          fa1PostLoop outReg M D) s s0 hPre]
  -- Walk forLoop: stepStmts ([forLoop] ++ postLoop) s0 = stepStmts postLoop sLoop.
  rw [stepStmts_append [Stmt.forLoop "n" numKVBlocks (fa1LoopBody kReg vReg M D Bk scale)]
        (fa1PostLoop outReg M D) s0 sLoop ?_]
  · exact fa1_postLoop_correct hBk hNumKVBlocks qReg kReg vReg outReg s.pid Q K V scale
      sLoop hPLoop idx
  · -- stepStmts [forLoop] s0 = stepStmts [] sLoop = some sLoop
    rw [stepStmts_cons _ [] _ _ hLoopStmt]
    show stepStmts [] sLoop = some sLoop
    unfold stepStmts
    rfl

end VeriTile.Examples
