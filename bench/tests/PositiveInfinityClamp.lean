import VeriTile.Triton.DSL
import VeriTile.Triton.Float.Structural
import VeriTile.Triton.Float.Erasure
import VeriTile.Triton.Semantics
import VeriTile.Meta.StatementAudit

/-! Positive infinity is supported as the right operand of a minimum.
The real projection is identity; the FP operation and precision are retained.
The FP equation below holds for every backend, including one where this
operation changes its operand. No identity axiom is introduced. -/
namespace VeriTile.Tests.PositiveInfinityClamp
open VeriTile Triton
noncomputable section

private def source : ComputeKernel :=
  let x : RegionName := "x"
  let y : RegionName := "y"
  triton {
    tl.region x = tl.bfloat16, y = tl.float32
    v = tl.load(x + 0).to(tl.float32)
    capped = tl.minimum(v, float("inf"))
    tl.store(y + 0, capped)
  }

/-- The source retains a distinct operation after the explicit fp32 cast. -/
theorem source_precision :
    source.surfaceBody[1]? = some (.assign .real [] "capped"
      (.compute (.alg .fp32 (.minPosInf (.ref .real [] "v"))))) := by rfl

/-- Real semantics of this operation does not require a finite replacement
for +infinity, even when the operand contains the model's negative infinity. -/
theorem real_minimum (e : Op .real sh) (s : BlockState) :
    evalOp (.minPosInf e) s = evalOp e s := by
  simp only [evalOp]
  cases h : evalOp e s with
  | none => rfl
  | some v => rfl

/-- FP evaluation calls the backend's minimum-with-infinity interpretation;
this intentionally does not assert that its output equals the input. -/
theorem fp_minimum {F : Type} [Inhabited F] (M : FP.Structural.Algebra F) (p : Option ComputeDType)
    (s : FP.Structural.State F) (v : FP.Structural.Values F .real sh)
    (h : s.regs .real sh "v" = some v) :
    FP.Structural.evalOp M p (.minPosInf (.ref .real sh "v")) s =
      some ((M.unary p .minPosInf) ∘ v) := by
  simp [FP.Structural.evalOp,h]


private def source64 : ComputeKernel :=
  let x : RegionName := "x"
  let y : RegionName := "y"
  triton {
    tl.region x = tl.bfloat16, y = tl.float64
    v = tl.load(x + 0).to(tl.float64)
    capped = tl.minimum(v, float("inf"))
    tl.store(y + 0, capped)
  }

theorem source_fp64_precision :
    source64.surfaceBody[1]? = some (.assign .real [] "capped"
      (.compute (.alg .fp64 (.minPosInf (.ref .real [] "v"))))) := by rfl

/-- error: tl.minimum(..., +inf): explicitly cast to fp32 or fp64 first: dtype mismatch -/
#guard_msgs (error, drop info) in
#check fun (x : RegionName) => (triton {
  tl.region x = tl.float16
  v = tl.load(x + 0)
  capped = tl.minimum(v, float("inf"))
} : ComputeKernel)

/-- error: tl.minimum(..., +inf): explicitly cast to fp32 or fp64 first: dtype mismatch -/
#guard_msgs (error, drop info) in
#check fun (x : RegionName) => (triton {
  tl.region x = tl.bfloat16
  v = tl.load(x + 0)
  capped = tl.minimum(v, float("inf"))
} : ComputeKernel)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Tests.PositiveInfinityClamp
