/-
VeriTile.Examples.OnlineSoftmax

Tier 2 kernel-pair (PAPER CENTERPIECE): online softmax recurrence ≡ batch
softmax. The streaming form

  m_0 = -∞,  l_0 = 0
  m_{k+1} = max(m_k, x_k)
  l_{k+1} = exp(m_k − m_{k+1}) · l_k + exp(x_k − m_{k+1})

produces the same (m, l) as the one-shot batch form

  m = max(x_0, ..., x_{N-1})
  l = Σ exp(x_i − m)

This is the algorithmic core of FlashAttention; Phase C will reuse the
recurrence at the kernel level.

For Phase B we use a degenerate "block size 1" form so that the loop
iterates over single elements, exposing the recurrence directly. Phase C's
FA kernel will instantiate this with full Bk-size blocks.
-/

import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.DSL
import VeriTile.Triton.LoopInvariant
import VeriTile.Examples.Common
import VeriTile.Examples.SoftmaxEq  -- reuse stableSoftmaxKernel + tileMax

namespace VeriTile.Examples

open VeriTile.Triton

/-- Batch softmax kernel — same as the existing `stableSoftmaxKernel` from
    `Examples/SoftmaxEq.lean`. Restated here for clarity / direct reference. -/
def batchSoftmaxKernel (xReg yReg : RegionName) (N : Nat) : Kernel :=
  stableSoftmaxKernel xReg yReg N

/-- Online softmax kernel: maintains `(m, l)` registers across single elements
    of the input tile. The Phase C FA kernel will lift this to KV-blocks. -/
def onlineSoftmaxKernel (xReg yReg : RegionName) (N : Nat) : Kernel := triton {
  pid := tl.program_id(0)
  m   := -inf    -- ℝ-channel sentinel (Op.negInf), finite stand-in for -∞
  l   := 0
  tl.for i in $(N) {
    xi    := tl.load($(xReg) + (pid * $(N) + i))
    m_new := tl.max(m, xi)
    l     := tl.exp(m - m_new) * l + tl.exp(xi - m_new)
    m     := m_new
  }
  -- Phase B focuses on (m, l) at the end. Phase C will add the per-element
  -- Y[i] = exp(x_i - m) / l divide.
}

/-- Online softmax math: streaming `m_k` after k iterations. -/
noncomputable def onlineSoftmaxM {N : Nat} (xs : Fin N → ℝ) : Nat → ℝ
  | 0     => -1e38
  | k + 1 =>
      if h : k < N then max (onlineSoftmaxM xs k) (xs ⟨k, h⟩)
      else onlineSoftmaxM xs k

/-- Online softmax math: streaming `l_k` after k iterations. -/
noncomputable def onlineSoftmaxL {N : Nat} (xs : Fin N → ℝ) : Nat → ℝ
  | 0     => 0
  | k + 1 =>
      if h : k < N then
        let m_old := onlineSoftmaxM xs k
        let m_new := onlineSoftmaxM xs (k + 1)
        Real.exp (m_old - m_new) * onlineSoftmaxL xs k + Real.exp (xs ⟨k, h⟩ - m_new)
      else onlineSoftmaxL xs k

/-- Batch softmax math. -/
noncomputable def batchSoftmaxM {N : Nat} (hN : 0 < N) (xs : Fin N → ℝ) : ℝ :=
  tileMax hN xs  -- defined in SoftmaxEq.lean

noncomputable def batchSoftmaxL {N : Nat} (hN : 0 < N) (xs : Fin N → ℝ) : ℝ :=
  ∑ i, Real.exp (xs i - batchSoftmaxM hN xs)

/-- **The math identity (paper centerpiece)**: the online recurrence at depth
    N produces the same (m, l) as the batch form. -/
theorem online_softmax_recurrence_eq_batch
    {N : Nat} (hN : 0 < N) (xs : Fin N → ℝ) :
    onlineSoftmaxM xs N = batchSoftmaxM hN xs ∧
    onlineSoftmaxL xs N = batchSoftmaxL hN xs := by
  sorry

/-- Operational correctness: the online softmax kernel computes (m, l)
    matching `onlineSoftmaxM xs N` and `onlineSoftmaxL xs N`. -/
theorem online_softmax_correct
    (xReg yReg : RegionName) (N : Nat) (hN : 0 < N)
    (s : BlockState) (xs : Fin N → ℝ)
    (h_x : InputLoadedAt s xReg N xs) :
    let final := exec (onlineSoftmaxKernel xReg yReg N) s
    final.bind (fun s' => s'.regs "m" >>= Value.asScalar)
        = some (onlineSoftmaxM xs N)
    ∧ final.bind (fun s' => s'.regs "l" >>= Value.asScalar)
        = some (onlineSoftmaxL xs N) := by
  sorry

end VeriTile.Examples
