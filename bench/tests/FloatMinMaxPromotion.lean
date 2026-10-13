import VeriTile.Triton.DSL
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.FloatMinMaxPromotion
open VeriTile.Triton

def mixedMax (x : RegionName) (B : Nat) : ComputeKernel := triton {
  tl.region x = tl.bfloat16
  m = tl.full([$(B)], value=-float("inf"), dtype=tl.float32)
  inp = tl.load(x + tl.arange(0, $(B)))
  m_new = tl.maximum(m, inp)
}

def mixedMin (x : RegionName) (B : Nat) : ComputeKernel := triton {
  tl.region x = tl.float16
  m = tl.full([$(B)], value=0.0, dtype=tl.float32)
  inp = tl.load(x + tl.arange(0, $(B)))
  m_new = tl.minimum(inp, m)
}

def literalMax (x : RegionName) (B : Nat) : ComputeKernel := triton {
  tl.region x = tl.float16
  inp = tl.load(x + tl.arange(0, $(B)))
  m_new = tl.maximum(0.0, inp)
}

/-- The input remains bf16 storage and is widened at the comparison/select. -/
example (x : RegionName) (B : Nat) :
    (mixedMax x B).surfaceBody.getLast? = some
      (.assign .real [B] "m_new" (.compute (.alg .fp32
        (.where (.gt .real (.consSame .nil) (.ref .real [B] "m")
          (.castFloat .bf16 .real (.ref .bf16 [B] "inp")))
          (.ref .real [B] "m") (.castFloat .bf16 .real (.ref .bf16 [B] "inp")))))) := rfl

/-- Reversing operand order still promotes the half tensor to fp32. -/
example (x : RegionName) (B : Nat) :
    (mixedMin x B).surfaceBody.getLast? = some
      (.assign .real [B] "m_new" (.compute (.alg .fp32
        (.where (.lt .real (.consSame .nil)
          (.castFloat .fp16 .real (.ref .fp16 [B] "inp")) (.ref .real [B] "m"))
          (.castFloat .fp16 .real (.ref .fp16 [B] "inp")) (.ref .real [B] "m"))))) := rfl

private def assignedHalf : ComputeStmt → Bool
  | .assign .fp16 _ _ _ => Bool.true
  | _ => Bool.false

/-- A constexpr literal does not promote a half tensor to fp32. -/
example (x : RegionName) (B : Nat) :
    (literalMax x B).surfaceBody.getLast?.map assignedHalf = some Bool.true := rfl

#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Tests.FloatMinMaxPromotion
