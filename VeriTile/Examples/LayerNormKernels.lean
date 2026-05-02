/-
VeriTile.Examples.LayerNormKernels

Tier 2 kernel-pair: fused single-pass LayerNorm ≡ two-pass LayerNorm.
Composes the Welford recurrence (Phase B #4) with the affine
`(x − μ)/√(var+ε) · γ + β` transform.

The two-pass kernel proof is closed over the typed tile semantics. The fused
kernel proof still depends on the Welford loop invariant.
-/

import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.DSL
import VeriTile.Triton.LoopInvariant
import VeriTile.Examples.Common
import VeriTile.Examples.WelfordKernels

namespace VeriTile.Examples

open VeriTile.Triton

/-- Two-pass LayerNorm kernel: `tl.sum` twice (mean and var), then affine. -/
def twoPassLayerNormKernel
    (xReg γReg βReg yReg : RegionName) (N : Nat) (ε : ℝ) : Kernel := triton {
  pid    := tl.program_id(0)
  offs   := pid * $(N) + tl.arange($(N))
  x      := tl.load(tl.ptr($(xReg)) + offs)
  s_x    := tl.sum(x)
  μ      := s_x / tl.toReal($(N))
  d      := x - μ
  s_d2   := tl.sum(d * d)
  v      := s_d2 / tl.toReal($(N))
  γ      := tl.load(tl.ptr($(γReg)) + tl.arange($(N)))
  β      := tl.load(tl.ptr($(βReg)) + tl.arange($(N)))
  σ_inv  := 1 / tl.sqrt(v + $ℝ(ε))
  y      := (x - μ) * σ_inv * γ + β
  tl.store(tl.ptr($(yReg)) + offs, y)
}

/-- Fused single-pass LayerNorm kernel: Welford `forLoop`, then affine. -/
def fusedLayerNormKernel
    (xReg γReg βReg yReg : RegionName) (N : Nat) (ε : ℝ) : Kernel := triton {
  pid := tl.program_id(0)
  M   := 0
  S   := 0
  tl.for i in $(N) {
    xi      := tl.load(tl.ptr($(xReg)) + (pid * $(N) + i))
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
  x       := tl.load(tl.ptr($(xReg)) + offs)
  γ       := tl.load(tl.ptr($(γReg)) + tl.arange($(N)))
  β       := tl.load(tl.ptr($(βReg)) + tl.arange($(N)))
  y       := (x - μ) * σ_inv * γ + β
  tl.store(tl.ptr($(yReg)) + offs, y)
}

def layerNormAffineTailKernel
    (xReg γReg βReg yReg : RegionName) (N : Nat) (ε : ℝ) : Kernel :=
  { inputs := [xReg, γReg, βReg]
    outputs := [yReg]
    body :=
      [ .assign .real [] "μ" (.ref .real [] "M")
      , .assign .real [] "v"
          (.div .real .nil (.ref .real [] "S") (Op.natToReal (.constNat N)))
      , .assign .real [] "σ_inv"
          (.div .real .nil (.const 1)
            (.sqrt (.add .real .nil (.ref .real [] "v") (.const ε))))
      , .assign .nat [N] "offs"
          (.add .nat .scalarL
            (.mul .nat .nil (.ref .nat [] "pid") (.constNat N))
            (.arange N))
      , .assign .real [N] "x" (.load xReg (.ref .nat [N] "offs"))
      , .assign .real [N] "γ" (.load γReg (.arange N))
      , .assign .real [N] "β" (.load βReg (.arange N))
      , .assign .real [N] "y"
          (.add .real (.consSame .nil)
            (.mul .real (.consSame .nil)
              (.mul .real .scalarR
                (.sub .real .scalarR (.ref .real [N] "x") (.ref .real [] "μ"))
                (.ref .real [] "σ_inv"))
              (.ref .real [N] "γ"))
            (.ref .real [N] "β"))
      , .store yReg [N] (.ref .nat [N] "offs") (.ref .real [N] "y")
      ] }

/-- LayerNorm spec: `y_i = (x_i − μ) / √(var + ε) · γ_i + β_i`. -/
noncomputable def layerNormSpec {N : Nat}
    (xs γs βs : Fin N → ℝ) (ε : ℝ) (i : Fin N) : ℝ :=
  let μ : ℝ := twoPassMean xs
  let v : ℝ := twoPassS xs / N
  (xs i - μ) / Real.sqrt (v + ε) * γs i + βs i

private def P_layernorm {N : Nat}
    (xs γs βs : Fin N → ℝ) (xReg γReg βReg : RegionName)
    (origPid : Nat) (k : Nat) (s : BlockState) : Prop :=
  s.regs .real [] "M" = some (Tile.scalar (welfordMean xs k))
  ∧ s.regs .real [] "S" = some (Tile.scalar (welfordS xs k))
  ∧ s.regs .nat [] "pid" = some (Tile.scalar origPid)
  ∧ s.pid = origPid
  ∧ InputLoadedAt s xReg N xs
  ∧ InputFeatureLoadedAt s γReg N γs
  ∧ InputFeatureLoadedAt s βReg N βs

end VeriTile.Examples
