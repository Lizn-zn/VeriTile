import VeriTile.Triton.DSL
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.LocalPrecisionDSL
open VeriTile.Triton
set_option maxRecDepth 6000

/-- Explicit fp64 storage supplies provenance; a mathematical real region does not. -/
def raw64 (W : RegionName) := triton {
  tl.region W = tl.float64
  w = tl.load(W + tl.arange(0, 2))
  y = w * w
}

theorem raw64_exact (W : RegionName) : (raw64 W).surfaceBody = [
    .assign .real [2] "w" (.compute (.alg .fp64
      (.load .real (.region W (.arange 2)) .none))),
    .assign .real [2] "y" (.compute (.alg .fp64
      (.mul .real (.consSame .nil) (.ref .real [2] "w") (.ref .real [2] "w"))))] := rfl

def mathematical (W : Region .real) := triton {
  w = tl.load($(W) + tl.arange(0, 2))
  y = w * w
}

theorem mathematical_unchanged (W : Region .real) : (mathematical W).surfaceBody = [
    .assign .real [2] "w" (.alg (.load .real (.region W (.arange 2)) .none)),
    .assign .real [2] "y" (.alg
      (.mul .real (.consSame .nil) (.ref .real [2] "w") (.ref .real [2] "w")))] := rfl

def nested := triton {
  x = $((2 : ℝ)).to(tl.float32)
  w = $((5 : ℝ)).to(tl.float64)
  y = (x * x).to(tl.float64) * w
}

theorem nested_local_boundary : nested.surfaceBody.drop 2 = [
    .assign .real [] "y" (.compute (.alg .fp64
      (.mul .real .nil
        (.castFloat .real .real (.withPrecision .real .fp32
          (.mul .real .nil (.ref .real [] "x") (.ref .real [] "x"))))
        (.ref .real [] "w"))))] := rfl

def mixedHalf (X W : RegionName) := triton {
  tl.region X = tl.float16, W = tl.bfloat16
  x = tl.load(X + tl.arange(0, 2))
  w = tl.load(W + tl.arange(0, 2))
  y = x * w
}

/-- Triton prefers fp16 for the fp16/bf16 pair (its usual tensor promotion rule). -/
theorem mixed_half_prefers_fp16 (X W : RegionName) : (mixedHalf X W).surfaceBody.drop 2 = [
    .assign .fp16 [2] "y" (.alg
      (.mul .fp16 (.consSame .nil) (.ref .fp16 [2] "x")
        (.castFloat .bf16 .fp16 (.ref .bf16 [2] "w"))))] := rfl

def halfBeforeWeight := triton {
  x = $((2 : ℝ)).to(tl.float32)
  w = $((5 : ℝ)).to(tl.float64)
  y = (x * x).to(tl.float16) * w
}

theorem half_cast_stays_before_weight : halfBeforeWeight.surfaceBody.drop 2 = [
    .assign .real [] "y" (.compute (.alg .fp64
      (.mul .real .nil
        (.castFloat .fp16 .real (.withPrecision .fp16 .fp32
          (.castFloat .real .fp16
            (.mul .real .nil (.ref .real [] "x") (.ref .real [] "x")))))
        (.ref .real [] "w"))))] := rfl

def halfBeforeWeight32 := triton {
  x = $((2 : ℝ)).to(tl.float32)
  w = $((5 : ℝ)).to(tl.float32)
  y = (x * x).to(tl.float16) * w
}

/-- Homogeneous fp32 composition introduces no redundant precision scope. -/
theorem half_cast_under_fp32 : halfBeforeWeight32.surfaceBody.drop 2 = [
    .assign .real [] "y" (.compute (.alg .fp32
      (.mul .real .nil
        (.castFloat .fp16 .real (.castFloat .real .fp16
          (.mul .real .nil (.ref .real [] "x") (.ref .real [] "x"))))
        (.ref .real [] "w"))))] := rfl

def scalarPromoted := triton {
  x = $((2 : ℝ)).to(tl.float64)
  y = x * 3.0
}

/-- An unannotated scalar literal takes the fp64 tensor context. -/
theorem scalar_literal_promotes : scalarPromoted.surfaceBody.drop 1 = [
    .assign .real [] "y" (.compute (.alg .fp64
      (.mul .real .nil (.ref .real [] "x") (.const 3.0))))] := rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Tests.LocalPrecisionDSL
