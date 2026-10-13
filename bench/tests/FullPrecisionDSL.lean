import VeriTile.Triton.DSL
import VeriTile.Meta.StatementAudit

/-! Typed full/zeros constructors must compose without dropping their fp32
precision. The negative-infinity expression is used by online attention. -/
namespace FullPrecisionDSLTests
open VeriTile.Triton

def finite := triton {
  y = tl.zeros([2], dtype=tl.float32) + 1.0
}

theorem finite_preserves_full : finite.surfaceBody = [
    .assign .real [2] "y" (.compute (.alg .fp32
      (.add .real .scalarR (.full [2] (.const 0)) (.const 1.0))))] := rfl

def negativeInfinity := triton {
  y = tl.zeros([2], dtype=tl.float32) - float("inf")
}

theorem infinity_preserves_full : negativeInfinity.surfaceBody = [
    .assign .real [2] "y" (.compute (.alg .fp32
      (.add .real .scalarR (.full [2] (.const 0)) .negInf)))] := rfl

def widened := triton {
  w = $((5 : ℝ)).to(tl.float64)
  y = tl.full([2], 2.0, dtype=tl.float32) * w
}

/-- The fp32 constructor remains inside its own precision scope under fp64 arithmetic. -/
theorem full_widening_boundary : widened.surfaceBody.drop 1 = [
    .assign .real [2] "y" (.compute (.alg .fp64
      (.mul .real .scalarR (.withPrecision .real .fp32 (.full [2] (.const 2.0)))
        (.ref .real [] "w"))))] := rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end FullPrecisionDSLTests
