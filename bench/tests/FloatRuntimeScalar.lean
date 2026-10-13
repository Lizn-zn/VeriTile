import VeriTile.Triton.DSL
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.FloatRuntimeScalar
open VeriTile.Triton

def halfProduct (x : RegionName) (alpha : ℝ) : ComputeKernel := triton {
  tl.region x = tl.float16
  value = tl.load(x)
  product = value * ($((alpha : ℝ))).to(tl.float32)
}

/-- An explicitly fp32 scalar promotes the half tensor before multiplication. -/
example (x : RegionName) (alpha : ℝ) :
    (halfProduct x alpha).surfaceBody.getLast? = some
      (.assign .real [] "product" (.compute (.alg .fp32
        (.mul .real .nil (.castFloat .fp16 .real (.ref .fp16 [] "value"))
          (.castFloat .real .real (.const alpha)))))) := rfl

#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Tests.FloatRuntimeScalar
