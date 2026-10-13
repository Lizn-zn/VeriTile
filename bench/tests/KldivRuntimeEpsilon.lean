import VeriTile.Triton.DSL
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.KldivRuntimeEpsilon
open VeriTile.Triton

/-- Python-float runtime epsilon has fp32 type before the target clamp. -/
def halfClamp (target : RegionName) (eps : ℝ) : ComputeKernel := triton {
  tl.region target = tl.float16
  value = tl.load(target)
  clamped = tl.maximum(value, ($((eps : ℝ))).to(tl.float32))
  logarithm = tl.log(clamped)
}

/-- The clamp widens the loaded half value, retaining the runtime epsilon cast. -/
example (target : RegionName) (eps : ℝ) :
    (halfClamp target eps).surfaceBody[1]? = some
      (.assign .real [] "clamped" (.compute (.alg .fp32
        (.where (.gt .real .nil
          (.castFloat .fp16 .real (.ref .fp16 [] "value"))
          (.castFloat .real .real (.const eps)))
          (.castFloat .fp16 .real (.ref .fp16 [] "value"))
          (.castFloat .real .real (.const eps)))))) := rfl

/-- The logarithm receives the promoted fp32 clamp result. -/
example (target : RegionName) (eps : ℝ) :
    (halfClamp target eps).surfaceBody[2]? = some
      (.assign .real [] "logarithm" (.compute (.alg .fp32
        (.log (.ref .real [] "clamped"))))) := rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Tests.KldivRuntimeEpsilon
