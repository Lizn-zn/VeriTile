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
        let prev := mPartial Bk Q numKVBlocks K scale k i
        let blockMax : WithBot ℝ :=
          Finset.univ.sup (fun jLocal : Fin Bk =>
            ((scaledScore Q K scale i
                ⟨k * Bk + jLocal.val, by
                  have : k < numKVBlocks := by omega
                  calc k * Bk + jLocal.val
                      < k * Bk + Bk := by omega
                    _ = (k + 1) * Bk := by ring
                    _ ≤ numKVBlocks * Bk := Nat.mul_le_mul_right Bk (by omega)
                    _ = Bk * numKVBlocks := by ring⟩
              : ℝ) : WithBot ℝ))
        max prev blockMax
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
        let sumBlock : ℝ :=
          Finset.univ.sum (fun jLocal : Fin Bk =>
            Real.exp (scaledScore Q K scale i
                ⟨k * Bk + jLocal.val, by
                  have : k < numKVBlocks := by omega
                  calc k * Bk + jLocal.val
                      < k * Bk + Bk := by omega
                    _ = (k + 1) * Bk := by ring
                    _ ≤ numKVBlocks * Bk := Nat.mul_le_mul_right Bk (by omega)
                    _ = Bk * numKVBlocks := by ring⟩
              - mNew))
        alphaPartial Q numKVBlocks K scale k i *
          lPartial Q numKVBlocks K scale k i + sumBlock
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
        let sumBlock : ℝ :=
          Finset.univ.sum (fun jLocal : Fin Bk =>
            let jFlat : Fin (Bk * numKVBlocks) :=
              ⟨k * Bk + jLocal.val, by
                have : k < numKVBlocks := by omega
                calc k * Bk + jLocal.val
                    < k * Bk + Bk := by omega
                  _ = (k + 1) * Bk := by ring
                  _ ≤ numKVBlocks * Bk := Nat.mul_le_mul_right Bk (by omega)
                  _ = Bk * numKVBlocks := by ring⟩
            Real.exp (scaledScore Q K scale i jFlat - mNew) *
              V (jFlat, d, PUnit.unit))
        alphaPartial Q numKVBlocks K scale k i *
          oPartial Q numKVBlocks K V scale k idx + sumBlock
      else
        oPartial Q numKVBlocks K V scale k idx

/-- **Math identity (paper centerpiece).** After all `numKVBlocks`
iterations, the streaming `(oPartial, lPartial)` ratio computes the
same value as `attentionReal` (the math reference using
`softmaxRow`). The kernel's post-loop `out := o_acc / l_i[:, None]`
realizes this identity at the operational layer.

**Status:** the recurrence is well-defined; this identity is the
remaining math obligation for issue #38 — proof deferred. The proof
strategy: (1) at `k = numKVBlocks`, induction over `k` shows that
`oPartial k = (Σ_{j ∈ first k*Bk indices} exp(s[i,j] - m_k[i]) · V[j])`
and analogously for `lPartial`; (2) use `mPartial`-cancellation
(numerator and denominator share the same shift) to convert
`oPartial / lPartial` into `softmaxRow`'s ratio without the row-max
subtraction. -/
theorem streaming_eq_attentionReal {M D Bk : Nat}
    (Q : TileIndex [M, D] → ℝ)
    (numKVBlocks : Nat) (hN : 0 < numKVBlocks)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ)
    (idx : TileIndex [M, D])
    (hl : lPartial Q numKVBlocks K scale numKVBlocks idx.1 ≠ 0) :
    oPartial Q numKVBlocks K V scale numKVBlocks idx /
        lPartial Q numKVBlocks K scale numKVBlocks idx.1
      = attentionReal Q K V scale idx := by
  sorry

end FA1Math

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

**Proof status: v0 leaves this as `sorry` — see issue #38 for the
online-softmax invariant + `forLoop_inv` discharge.** -/
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
  sorry

end VeriTile.Examples
