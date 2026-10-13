import VeriTile.Triton.DSL
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.FloatDivisionPromotion
open VeriTile.Triton

def mixedHalf (X Y : RegionName) (B : Nat) : ComputeKernel := triton {
  tl.region X = tl.float16, Y = tl.bfloat16
  x = tl.load(X + tl.arange(0, $(B)))
  y = tl.load(Y + tl.arange(0, $(B)))
  forward = x / y
  reverse = y / x
  product = x * y
}

/-- Division casts both original operands directly to fp32, in either order. -/
example (X Y : RegionName) (B : Nat) :
    (mixedHalf X Y B).surfaceBody[2]? = some
      (.assign .real [B] "forward" (.compute (.alg .fp32
        (.div .real (.consSame .nil)
          (.castFloat .fp16 .real (.ref .fp16 [B] "x"))
          (.castFloat .bf16 .real (.ref .bf16 [B] "y")))))) ∧
    (mixedHalf X Y B).surfaceBody[3]? = some
      (.assign .real [B] "reverse" (.compute (.alg .fp32
        (.div .real (.consSame .nil)
          (.castFloat .bf16 .real (.ref .bf16 [B] "y"))
          (.castFloat .fp16 .real (.ref .fp16 [B] "x")))))) := ⟨rfl, rfl⟩

/-- Multiplication still uses the fp16 common type selected by Triton. -/
example (X Y : RegionName) (B : Nat) :
    (mixedHalf X Y B).surfaceBody[4]? = some
      (.assign .fp16 [B] "product" (.alg
        (.mul .fp16 (.consSame .nil) (.ref .fp16 [B] "x")
          (.castFloat .bf16 .fp16 (.ref .bf16 [B] "y"))))) := rfl

def wideDivision (X Y : RegionName) (B : Nat) : ComputeKernel := triton {
  tl.region X = tl.float16, Y = tl.float64
  x = tl.load(X + tl.arange(0, $(B)))
  y = tl.load(Y + tl.arange(0, $(B)))
  out = x / y
}

/-- An fp64 operand dominates division; the half input is not rounded through fp32. -/
example (X Y : RegionName) (B : Nat) :
    (wideDivision X Y B).surfaceBody[2]? = some
      (.assign .real [B] "out" (.compute (.alg .fp64
        (.div .real (.consSame .nil)
          (.castFloat .fp16 .real
            (.withPrecision .fp16 .fp32 (.ref .fp16 [B] "x")))
          (.ref .real [B] "y"))))) := rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Tests.FloatDivisionPromotion
