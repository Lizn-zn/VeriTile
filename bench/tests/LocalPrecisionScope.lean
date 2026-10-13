import VeriTile.Triton.Float.Structural
import VeriTile.Triton.Float.Erasure
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.LocalPrecisionScope
open VeriTile.Triton
open VeriTile.Triton.FP.Structural

/-- This carrier records actual evaluator callbacks, without numerical laws. -/
inductive Trace where
  | leaf
  | literal (p : Option ComputeDType) (d : FloatDType)
  | binary (p : Option ComputeDType) (d : FloatDType) (op : Binary) (a b : Trace)
  | unary (p : Option ComputeDType) (op : Unary) (a : Trace)
  | cast (p : Option ComputeDType) (src dst : FloatDType) (a : Trace)
  deriving Inhabited

noncomputable def backend : Algebra Trace where
  literal p d _ := .literal p d
  negInf := .leaf
  binary := Trace.binary
  unary := Trace.unary
  cast := Trace.cast
  fromNat _ _ := .leaf
  fromInt _ _ := .leaf
  fp32Bits _ := .leaf
  fp32Load := id
  reduceMax _ {_} _ _ _ := fun _ => .leaf
  reduceSum _ {_} _ _ _ := fun _ => .leaf

def empty : State Trace where
  mem _ _ := .mk .real .leaf
  regs _ _ _ := none
  pids _ := 0
  numPids _ := 1
  undef d _ _ := defaultValue d

noncomputable def inner : Op .real [] := .mul .real .nil (.const 2) (.const 3)
noncomputable def mixed : Op .real [] :=
  .mul .real .nil
    (.castFloat .real .real (.withPrecision .real .fp32 inner)) (.const 5)

/-- The inner product remains fp32; its cast and the enclosing product are fp64. -/
theorem records_nested_precisions :
    evalOp backend (some .fp64) mixed empty = some (fun _ =>
      Trace.binary (some .fp64) .real .mul
        (.cast (some .fp64) .real .real
          (.binary (some .fp32) .real .mul
            (.literal (some .fp32) .real) (.literal (some .fp32) .real)))
        (.literal (some .fp64) .real)) := rfl

noncomputable def preHalf : Op .fp16 [] := .castFloat .real .fp16 inner
noncomputable def halfPromoted : Op .real [] :=
  .mul .real .nil
    (.castFloat .fp16 .real (.withPrecision .fp16 .fp32 preHalf)) (.const 5)

/-- A cast to half remains before multiplication and has its own fp32 context. -/
theorem records_half_cast_before_promotion :
    evalOp backend (some .fp64) halfPromoted empty = some (fun _ =>
      Trace.binary (some .fp64) .real .mul
        (.cast (some .fp64) .fp16 .real
          (.cast (some .fp32) .real .fp16
            (.binary (some .fp32) .real .mul
              (.literal (some .fp32) .real) (.literal (some .fp32) .real))))
        (.literal (some .fp64) .real)) := rfl

theorem scope_preserves_missing_register_failure :
    evalOp backend (some .fp64)
      (.withPrecision .real .fp32 (.ref .real [] "missing")) empty = none := rfl

theorem scope_erases_exactly (d : FloatDType) (p : IndexComputePrecision)
    (e : Op d.toTileDType sh) : (Op.withPrecision d p e).eraseDType = e.eraseDType := by
  simp only [Op.eraseDType]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Tests.LocalPrecisionScope
