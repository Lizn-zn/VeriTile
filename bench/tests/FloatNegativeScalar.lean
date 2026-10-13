import VeriTile.Triton.DSL
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.FloatNegativeScalar
open VeriTile.Triton

def halfNegative (T O : RegionName) : ComputeKernel := triton {
  tl.region T = tl.float16, O = tl.float16
  target = tl.load(T)
  res = target * -1
  tl.store(O, res)
}

/-- A negative constexpr is materialized in the half tensor's type without runtime subtraction. -/
example (T O : RegionName) :
    (halfNegative T O).surfaceBody[1]? = some (.assign .fp16 [] "res"
      (.alg (.mul .fp16 .nil (.ref .fp16 [] "target") (.constFloat .fp16 (-1))))) := rfl

/-- The same-storage store retains the narrow value without an unrequested wider carrier. -/
example (T O : RegionName) :
    (halfNegative T O).surfaceBody[2]? = some (.store .fp16 []
      (.region O.cast (.constNat 0)) (.alg (.ref .fp16 [] "res")) .none) := rfl

def bfloatNegative (T : RegionName) : ComputeKernel := triton {
  tl.region T = tl.bfloat16
  target = tl.load(T)
  res = (-1) * target
}

/-- Parentheses and left scalar position retain the same weak-scalar rule. -/
example (T : RegionName) :
    (bfloatNegative T).surfaceBody[1]? = some (.assign .bf16 [] "res"
      (.alg (.mul .bf16 .nil (.constFloat .bf16 (-1)) (.ref .bf16 [] "target")))) := rfl

def singleNegative (T : RegionName) : ComputeKernel := triton {
  tl.region T = tl.float32
  target = tl.load(T)
  res = target * -1
}

/-- Typed fp32 likewise receives one constant, with no additional backend subtraction. -/
example (T : RegionName) :
    (singleNegative T).surfaceBody[1]? = some (.assign .real [] "res"
      (.compute (.alg .fp32 (.mul .real .nil (.ref .real [] "target") (.const (-1)))))) := rfl

def doubleNegative (T : RegionName) : ComputeKernel := triton {
  tl.region T = tl.float64
  target = tl.load(T)
  res = target * -1
}

/-- Typed fp64 also materializes the constexpr directly in the selected computation type. -/
example (T : RegionName) :
    (doubleNegative T).surfaceBody[1]? = some (.assign .real [] "res"
      (.compute (.alg .fp64 (.mul .real .nil (.ref .real [] "target") (.const (-1)))))) := rfl

def halfRuntimeNegation (T : RegionName) : ComputeKernel := triton {
  tl.region T = tl.float16
  target = tl.load(T)
  res = -target
}

/-- Runtime tensor negation remains a subtraction in the tensor's own type. -/
example (T : RegionName) :
    (halfRuntimeNegation T).surfaceBody[1]? = some (.assign .fp16 [] "res"
      (.alg (.sub .fp16 .nil (.constFloat .fp16 0.0) (.ref .fp16 [] "target")))) := rfl

def legacyRealNegative (T : RegionName) : ComputeKernel := triton {
  target = tl.load(T)
  res = target * -1
}

/-- The untyped mathematical surface retains its historical expression tree. -/
example (T : RegionName) :
    (legacyRealNegative T).surfaceBody[1]? = some (.assign .real [] "res"
      (.alg (.mul .real .nil (.ref .real [] "target")
        (.sub .real .nil (.const 0.0) (.const 1))))) := rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Tests.FloatNegativeScalar
