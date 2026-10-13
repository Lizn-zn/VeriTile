import VeriTile.Triton.DSL
import VeriTile.Meta.StatementAudit

/-! Triton v3.5.1 math.py lines 137–159 retains the input's fp32/fp64 type
for sqrt and rsqrt. These checks pin both operation precision and nested
precision boundaries; they make no hardware rounding claim. -/
namespace RootPrecisionDSLTests
open VeriTile.Triton

def roots64 := triton {
  x = $((4 : ℝ)).to(tl.float64)
  a = tl.sqrt(x)
  b = tl.rsqrt(x)
  c = tl.math.rsqrt(x)
}

theorem roots_keep_fp64 : roots64.surfaceBody.drop 1 = [
    .assign .real [] "a" (.compute (.alg .fp64 (.sqrt (.ref .real [] "x")))),
    .assign .real [] "b" (.compute (.alg .fp64 (.rsqrt (.ref .real [] "x")))),
    .assign .real [] "c" (.compute (.alg .fp64 (.rsqrt (.ref .real [] "x"))))] := rfl

def roots32 := triton {
  x = $((4 : ℝ)).to(tl.float32)
  a = tl.sqrt(x)
  b = tl.rsqrt(x)
  c = tl.math.rsqrt(x)
}

theorem roots_keep_fp32 : roots32.surfaceBody.drop 1 = [
    .assign .real [] "a" (.compute (.alg .fp32 (.sqrt (.ref .real [] "x")))),
    .assign .real [] "b" (.compute (.alg .fp32 (.rsqrt (.ref .real [] "x")))),
    .assign .real [] "c" (.compute (.alg .fp32 (.rsqrt (.ref .real [] "x"))))] := rfl

def mixed := triton {
  x = $((4 : ℝ)).to(tl.float32)
  w = $((5 : ℝ)).to(tl.float64)
  a = tl.sqrt(x) * w
  b = tl.rsqrt(x) * w
  c = tl.math.rsqrt(x) * w
}

/-- A later fp64 multiplication must not retroactively widen the root operation. -/
theorem nested_roots_keep_fp32 : mixed.surfaceBody.drop 2 = [
    .assign .real [] "a" (.compute (.alg .fp64 (.mul .real .nil
      (.withPrecision .real .fp32 (.sqrt (.ref .real [] "x"))) (.ref .real [] "w")))),
    .assign .real [] "b" (.compute (.alg .fp64 (.mul .real .nil
      (.withPrecision .real .fp32 (.rsqrt (.ref .real [] "x"))) (.ref .real [] "w")))),
    .assign .real [] "c" (.compute (.alg .fp64 (.mul .real .nil
      (.withPrecision .real .fp32 (.rsqrt (.ref .real [] "x"))) (.ref .real [] "w"))))] := rfl

def mathematical (x : ℝ) := triton {
  a = tl.sqrt($((x : ℝ)))
  b = tl.rsqrt($((x : ℝ)))
}

theorem mathematical_carrier_unchanged (x : ℝ) : (mathematical x).surfaceBody = [
    .assign .real [] "a" (.alg (.sqrt (.const x))),
    .assign .real [] "b" (.alg (.rsqrt (.const x)))] := rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end RootPrecisionDSLTests
