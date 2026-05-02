/-
VeriTile.Examples.OnlineSoftmax

Online softmax recurrence and typed Triton kernel skeleton.
-/

import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.DSL
import VeriTile.Triton.LoopInvariant
import VeriTile.Examples.Common
import VeriTile.Examples.SoftmaxEq

namespace VeriTile.Examples

open VeriTile.Triton

def batchSoftmaxKernel (xReg yReg : RegionName) (N : Nat) : Kernel :=
  stableSoftmaxKernel xReg yReg N

def onlineSoftmaxKernel (xReg _yReg : RegionName) (N : Nat) : Kernel := triton {
  pid := tl.program_id(0)
  m   := -inf
  l   := 0
  tl.for i in $(N) {
    xi    := tl.load(tl.ptr($(xReg)) + (pid * $(N) + i))
    m_new := tl.max(m, xi)
    l     := tl.exp(m - m_new) * l + tl.exp(xi - m_new)
    m     := m_new
  }
}

private def onlineSoftmaxLoopBody (xReg : RegionName) (N : Nat) : List Stmt :=
  [Stmt.assign .real [] "xi"
      (Op.load xReg
        (Op.add .nat .nil
          (Op.mul .nat .nil (Op.ref .nat [] "pid")
            (Op.constNat N))
          (Op.ref .nat [] "i"))),
    Stmt.assign .real [] "m_new"
      (Op.max2 .nil (Op.ref .real [] "m") (Op.ref .real [] "xi")),
    Stmt.assign .real [] "l"
      (Op.add .real .nil
        (Op.mul .real .nil
          (Op.exp (Op.sub .real .nil
            (Op.ref .real [] "m") (Op.ref .real [] "m_new")))
          (Op.ref .real [] "l"))
        (Op.exp (Op.sub .real .nil
          (Op.ref .real [] "xi") (Op.ref .real [] "m_new")))),
    Stmt.assign .real [] "m" (Op.ref .real [] "m_new")]

/-! ## Math model — `WithBot ℝ`-valued

The seed `M_0 = ⊥` is genuinely below every real number, so the first
iteration's `max ⊥ (xs 0) = some (xs 0)` reproduces the batch base case
*without* a magnitude precondition on the input data. This is the entire
point of the `WithBot` refactor (issue #21): no more `h_lo`. -/

noncomputable def onlineSoftmaxM {N : Nat} (xs : Fin N → ℝ) : Nat → WithBot ℝ
  | 0     => ⊥
  | k + 1 =>
      if h : k < N then max (onlineSoftmaxM xs k) (((xs ⟨k, h⟩ : ℝ) : WithBot ℝ))
      else onlineSoftmaxM xs k

/-- L-recurrence in `WithBot ℝ`. After `k ≥ 1` iterations the result is
`some (∑ exp(xs i - M_k))`; at `k = 0` the seed is `some 0`. -/
noncomputable def onlineSoftmaxL {N : Nat} (xs : Fin N → ℝ) : Nat → WithBot ℝ
  | 0     => ((0 : ℝ) : WithBot ℝ)
  | k + 1 =>
      if h : k < N then
        let m_old := onlineSoftmaxM xs k
        let m_new := onlineSoftmaxM xs (k + 1)
        WithBot.realAdd
          (WithBot.realMul (WithBot.realExp (WithBot.realSub m_old m_new))
                           (onlineSoftmaxL xs k))
          (WithBot.realExp (WithBot.realSub (((xs ⟨k, h⟩ : ℝ) : WithBot ℝ)) m_new))
      else onlineSoftmaxL xs k

noncomputable def batchSoftmaxM {N : Nat} (hN : 0 < N) (xs : Fin N → ℝ) : ℝ :=
  tileMax hN xs

noncomputable def batchSoftmaxL {N : Nat} (hN : 0 < N) (xs : Fin N → ℝ) : ℝ :=
  ∑ i, Real.exp (xs i - batchSoftmaxM hN xs)

/-! ### Prefix lemmas — no `h_lo` precondition needed -/

private def P_online_softmax {N : Nat} (xs : Fin N → ℝ) (xReg : RegionName)
    (origPid : Nat) (k : Nat) (s : BlockState) : Prop :=
  -- `onlineSoftmaxM/L xs k : WithBot ℝ` directly populates the tile.
  s.regs .real [] "m" = some (Tile.scalar (onlineSoftmaxM xs k))
  ∧ s.regs .real [] "l" = some (Tile.scalar (onlineSoftmaxL xs k))
  ∧ s.regs .nat [] "pid" = some (Tile.scalar origPid)
  ∧ s.pid = origPid
  ∧ InputLoadedAt s xReg N xs

end VeriTile.Examples
