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
    s   = (Q · K_bᵀ) · invSqrtD                       -- [M, Bk]
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
    (M D Bk numKVBlocks : Nat) (invSqrtD : ℝ) : Kernel := triton {
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
    scores  := tl.dot(q, tl.trans(k)) * $ℝ(invSqrtD)

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

Standard scaled-dot-product attention on real-valued tiles, all
post-`unbotD 0` (i.e. `WithBot ℝ` is degenerate by hypothesis on
inputs). Single-Q-block view: `Q : [M, D]`, `K, V : [S, D]`. -/

/-- Row-wise softmax along the trailing axis: divide each row by the
sum of its `exp`. (We do *not* need the row-max subtraction here since
this is the math reference, not a numerical implementation —
correctness is what matters.) Operates on the underlying ℝ values via
`unbotD 0`; the proof in #38 will assume well-formed inputs (no `⊥`). -/
noncomputable def softmaxRow {M N : Nat} (s : Tile .real [M, N]) :
    Tile .real [M, N] :=
  ⟨fun (m, n, _) =>
    let row := fun j : Fin N => (s.data (m, j, PUnit.unit)).unbotD 0
    let num := Real.exp (row n)
    let denom := Finset.univ.sum (fun j : Fin N => Real.exp (row j))
    some (num / denom)⟩

/-- Reference attention: `softmax(Q · Kᵀ · invSqrtD) · V`. Math-side
analogue of `fa1ForwardKernel`. -/
noncomputable def attention {M S D : Nat}
    (Q : Tile .real [M, D]) (K V : Tile .real [S, D])
    (invSqrtD : ℝ) : Tile .real [M, D] :=
  -- (Q · Kᵀ) — element-wise scale by invSqrtD — softmax row-wise — · V
  let qkT : Tile .real [M, S] :=
    Tile.dot [] Q (Tile.transpose [] K)
  let scaled : Tile .real [M, S] :=
    ⟨fun idx => Option.map (· * invSqrtD) (qkT.data idx)⟩
  let p : Tile .real [M, S] := softmaxRow scaled
  Tile.dot [] p V

/-! ## Correctness statement (sorry — discharged in issue #38)

The statement uses the standard `InputAt` / `observeTileAt` predicates
from `Examples.Common` to relate the kernel's memory effects to the
math model on `Tile`s. Single-block: one Q-row block per `program_id`,
which we expose as a hypothesis on the input state's `pid`. -/

/-- After running `fa1ForwardKernel` on a state where Q is loaded as a
contiguous `[M, D]` row-major block at `(s.pid * M, ⋯)` and K, V are
loaded as full `[S, D]` row-major matrices at `(0, ⋯)`, the output
region holds `attention(Q, K, V, invSqrtD)` row-major in the same
`[M, D]` block.

**Proof status: v0 leaves this as `sorry` — see issue #38 for the
online-softmax invariant + `forLoop_inv` discharge.** -/
theorem fa1_forward_correct
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg outReg : RegionName)
    (Q : Tile .real [M, D]) (K V : Tile .real [Bk * numKVBlocks, D])
    (invSqrtD : ℝ)
    (s s' : BlockState)
    (_hQ : InputAt s qReg
        (Offset.rowMajor2D (rows := M) (cols := D) (s.pid * M * D) D)
        (fun idx => (Q.data idx).unbotD 0))
    (_hK : InputAt s kReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D)
        (fun idx => (K.data idx).unbotD 0))
    (_hV : InputAt s vReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D)
        (fun idx => (V.data idx).unbotD 0))
    (_hExec :
      exec (fa1ForwardKernel qReg kReg vReg outReg M D Bk numKVBlocks invSqrtD)
        s = some s') :
    ∀ idx : TileIndex [M, D],
      observeTileAt (some s') outReg
          (Offset.rowMajor2D (rows := M) (cols := D) (s.pid * M * D) D) idx
        = some (((attention Q K V invSqrtD).data idx).unbotD 0) := by
  sorry

end VeriTile.Examples
