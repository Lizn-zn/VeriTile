import VeriTile.Triton.DSL
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.ScalarPrecision
open Lean VeriTile.Triton

syntax "scalarMultiply%" "[" tritonDType "]" : term
set_option hygiene false in
macro_rules
  | `(scalarMultiply%[$dtype:tritonDType]) => `(fun (x : RegionName) (scale : ℝ) => triton {
      tl.region x = $dtype
      a = tl.load(x)
      b = $((scale : ℝ)) * a
      c = a * $((scale : ℝ))
      d = 2.0 * a
    })

def half : RegionName → ℝ → ComputeKernel := scalarMultiply%[tl.float16]
def bfloat : RegionName → ℝ → ComputeKernel := scalarMultiply%[tl.bfloat16]

def assignmentTypes : List ComputeStmt → List TileDType
  | [] => []
  | .assign d _ _ _ :: xs => d :: assignmentTypes xs
  | _ :: xs => assignmentTypes xs

/-- Constexpr scaling before an explicit fp32 cast still rounds in the input
precision; a scalar value must not silently widen all half multiplications. -/
example (x : RegionName) (scale : ℝ) :
    assignmentTypes (half x scale).surfaceBody = [.fp16,.fp16,.fp16,.fp16] := rfl
example (x : RegionName) (scale : ℝ) :
    assignmentTypes (bfloat x scale).surfaceBody = [.bf16,.bf16,.bf16,.bf16] := rfl

def runtimeScalar (x scale : RegionName) : ComputeKernel := triton {
  tl.region x = tl.float16, scale = tl.float32
  a = tl.load(x)
  s = tl.load(scale)
  b = s * a
}
/-- A loaded fp32 scalar participates in promotion even though its shape is []. -/
example (x s : RegionName) :
    assignmentTypes (runtimeScalar x s).surfaceBody = [.fp16,.real,.real] := rfl

def division (x : RegionName) (scale : ℝ) : ComputeKernel := triton {
  tl.region x = tl.float16
  a = tl.load(x)
  b = a / $((scale : ℝ))
  c = a / a
}
/-- Triton promotes both scalar and tensor half division to fp32. -/
example (x : RegionName) (scale : ℝ) :
    assignmentTypes (division x scale).surfaceBody = [.fp16,.real,.real] := rfl

def explicitCast (x : RegionName) : ComputeKernel := triton {
  tl.region x = tl.float16
  a = tl.load(x)
  b = (2.0).to(tl.float32) * a
  c = a + 2.0
  d = 2.0 - a
}
/-- An explicit fp32 cast remains a promotion boundary; unannotated scalar
addition/subtraction still use the input precision. -/
example (x : RegionName) :
    assignmentTypes (explicitCast x).surfaceBody = [.fp16,.real,.fp16,.fp16] := rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Tests.ScalarPrecision
