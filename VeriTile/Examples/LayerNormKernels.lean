/-
VeriTile.Examples.LayerNormKernels

Tier 2 kernel-pair: fused single-pass LayerNorm ≡ two-pass LayerNorm.
Composes the Welford recurrence (Phase B #4) with the affine
`(x − μ)/√(var+ε) · γ + β` transform.

Skeleton only — the three correctness theorems are sorry'd here. Per the
recent project decision (focus on getting all skeletons + operators in
place first), the proofs are deferred to a follow-up task. Theorem
signatures include the alias-hazard hypotheses (`yReg ≠ xReg`, etc.) now
so future proof work doesn't have to re-shape the API.
-/

import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.DSL
import VeriTile.Triton.LoopInvariant
import VeriTile.Examples.Common
import VeriTile.Examples.WelfordKernels
import VeriTile.Examples.WelfordMath

namespace VeriTile.Examples

open VeriTile.Triton

/-- Two-pass LayerNorm kernel: `tl.sum` twice (mean and var), then affine. -/
def twoPassLayerNormKernel
    (xReg γReg βReg yReg : RegionName) (N : Nat) (ε : ℝ) : Kernel := triton {
  pid    := tl.program_id(0)
  offs   := pid * $(N) + tl.arange($(N))
  x      := tl.load($(xReg) + offs)
  s_x    := tl.sum(x)
  μ      := s_x / tl.toReal($(N))
  d      := x - μ
  s_d2   := tl.sum(d * d)
  v      := s_d2 / tl.toReal($(N))
  γ      := tl.load($(γReg) + tl.arange($(N)))
  β      := tl.load($(βReg) + tl.arange($(N)))
  σ_inv  := 1 / tl.sqrt(v + $ℝ(ε))
  y      := (x - μ) * σ_inv * γ + β
  tl.store($(yReg) + offs, y)
}

/-- Fused single-pass LayerNorm kernel: Welford `forLoop`, then affine. -/
def fusedLayerNormKernel
    (xReg γReg βReg yReg : RegionName) (N : Nat) (ε : ℝ) : Kernel := triton {
  pid := tl.program_id(0)
  M   := 0
  S   := 0
  tl.for i in $(N) {
    xi      := tl.load($(xReg) + (pid * $(N) + i))
    delta   := xi - M
    M       := M + delta / (tl.toReal(i) + 1)
    delta2  := xi - M
    S       := S + delta * delta2
  }
  μ       := M
  v       := S / tl.toReal($(N))
  σ_inv   := 1 / tl.sqrt(v + $ℝ(ε))
  -- Second pass to compute Y. The "fused" gain is that μ/var were
  -- computed in a single pass over `x`; the residual `(x − μ)` still
  -- needs the second read of x.
  offs    := pid * $(N) + tl.arange($(N))
  x       := tl.load($(xReg) + offs)
  γ       := tl.load($(γReg) + tl.arange($(N)))
  β       := tl.load($(βReg) + tl.arange($(N)))
  y       := (x - μ) * σ_inv * γ + β
  tl.store($(yReg) + offs, y)
}

/-- LayerNorm spec: `y_i = (x_i − μ) / √(var + ε) · γ_i + β_i`. -/
noncomputable def layerNormSpec {N : Nat}
    (xs γs βs : Fin N → ℝ) (ε : ℝ) (i : Fin N) : ℝ :=
  let μ : ℝ := twoPassMean xs
  let v : ℝ := twoPassS xs / N
  (xs i - μ) / Real.sqrt (v + ε) * γs i + βs i

theorem twopass_layernorm_correct
    (xReg γReg βReg yReg : RegionName) (N : Nat) (_hN : 0 < N) (ε : ℝ)
    (s : BlockState) (xs γs βs : Fin N → ℝ)
    (_h_x : InputLoadedAt s xReg N xs)
    (_h_γ : InputLoadedAt s γReg N γs)
    (_h_β : InputLoadedAt s βReg N βs)
    (_h_yx : yReg ≠ xReg) (_h_yγ : yReg ≠ γReg) (_h_yβ : yReg ≠ βReg) :
    ∀ i : Fin N,
      observeAt (exec (twoPassLayerNormKernel xReg γReg βReg yReg N ε) s)
                yReg N s.pid i
        = some (layerNormSpec xs γs βs ε i) := by
  sorry

theorem fused_layernorm_correct
    (xReg γReg βReg yReg : RegionName) (N : Nat) (_hN : 0 < N) (ε : ℝ)
    (s : BlockState) (xs γs βs : Fin N → ℝ)
    (_h_x : InputLoadedAt s xReg N xs)
    (_h_γ : InputLoadedAt s γReg N γs)
    (_h_β : InputLoadedAt s βReg N βs)
    (_h_yx : yReg ≠ xReg) (_h_yγ : yReg ≠ γReg) (_h_yβ : yReg ≠ βReg) :
    ∀ i : Fin N,
      observeAt (exec (fusedLayerNormKernel xReg γReg βReg yReg N ε) s)
                yReg N s.pid i
        = some (layerNormSpec xs γs βs ε i) := by
  sorry

theorem layernorm_kernels_refinement
    (xReg γReg βReg yReg : RegionName) (N : Nat) (_hN : 0 < N) (ε : ℝ)
    (s : BlockState) (xs γs βs : Fin N → ℝ)
    (_h_x : InputLoadedAt s xReg N xs)
    (_h_γ : InputLoadedAt s γReg N γs)
    (_h_β : InputLoadedAt s βReg N βs)
    (_h_yx : yReg ≠ xReg) (_h_yγ : yReg ≠ γReg) (_h_yβ : yReg ≠ βReg) :
    ∀ i : Fin N,
      observeAt (exec (twoPassLayerNormKernel xReg γReg βReg yReg N ε) s)
                yReg N s.pid i
        = observeAt (exec (fusedLayerNormKernel xReg γReg βReg yReg N ε) s)
                yReg N s.pid i := by
  sorry

end VeriTile.Examples
