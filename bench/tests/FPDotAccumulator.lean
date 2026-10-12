import VeriTile.Triton.DSL
import VeriTile.Triton.Float.ExecutionProfile
import VeriTile.Triton.Float.RewriteTactics
import VeriTile.Meta.StatementAudit

/-! The explicit accumulator is part of dot, including its precision and
options. The deliberately distinguishing backend below is an interpreter
regression, not a floating-point hardware simulator. -/
namespace FPDotAccumulatorTests
open VeriTile.Triton FP.Structural

def halfSource : ComputeKernel := triton {
  a = tl.full([16, 16], 2, dtype=tl.float16)
  b = tl.full([16, 16], 3, dtype=tl.float16)
  acc = tl.full([16, 16], 7, dtype=tl.float32)
  acc = tl.dot(a, b, acc)
}

theorem source_retains_accumulator : halfSource.surfaceBody[3]? =
    some (.assign .real [16,16] "acc" (.compute (.alg .fp32
      (.dotAcc .fp16 .fp16 .real (batch := [])
        (.ref .fp16 [16,16] "a") (.ref .fp16 [16,16] "b")
        (.ref .real [16,16] "acc"))))) := rfl

def bfloatSource : ComputeKernel := triton {
  a = tl.full([16, 16], 2, dtype=tl.bfloat16)
  b = tl.full([16, 16], 3, dtype=tl.bfloat16)
  acc = tl.full([16, 16], 7, dtype=tl.float32)
  acc = tl.dot(a, b, acc, input_precision="ieee")
}

theorem source_retains_bfloat_and_options : bfloatSource.surfaceBody[3]? =
    some (.assign .real [16,16] "acc" (.compute (.alg .fp32
      (.dotAcc .bf16 .bf16 .real (batch := [])
        (.ref .bf16 [16,16] "a") (.ref .bf16 [16,16] "b")
        (.ref .real [16,16] "acc") { inputPrecision := some "ieee" })))) := rfl

def halfAccumulator : ComputeKernel := triton {
  a = tl.full([16, 16], 2, dtype=tl.float16)
  b = tl.full([16, 16], 3, dtype=tl.float16)
  acc = tl.full([16, 16], 7, dtype=tl.float16)
  acc = tl.dot(a, b, acc, out_dtype=tl.float16)
}

theorem explicit_half_output_matches_accumulator : halfAccumulator.surfaceBody[3]? =
    some (.assign .fp16 [16,16] "acc" (.alg
      (.dotAcc .fp16 .fp16 .fp16 (batch := [])
        (.ref .fp16 [16,16] "a") (.ref .fp16 [16,16] "b")
        (.ref .fp16 [16,16] "acc") { outDType := some .fp16 }))) := rfl

/-- error: tl.dot: out_dtype must match accumulator precision -/
#guard_msgs in
def conflictingOutput : ComputeKernel := triton {
  a = tl.full([16, 16], 2, dtype=tl.float16)
  b = tl.full([16, 16], 3, dtype=tl.float16)
  acc = tl.full([16, 16], 7, dtype=tl.float16)
  acc = tl.dot(a, b, acc, out_dtype=tl.float32)
}

/-- error: tl.dot: bfloat16 output requires a float32/float16 dot followed by a cast -/
#guard_msgs in
def bfloatAccumulator : ComputeKernel := triton {
  a = tl.full([16, 16], 2, dtype=tl.bfloat16)
  b = tl.full([16, 16], 3, dtype=tl.bfloat16)
  acc = tl.full([16, 16], 7, dtype=tl.bfloat16)
  acc = tl.dot(a, b, acc, out_dtype=tl.bfloat16)
}

/-- error: tl.dot: out_dtype must match accumulator precision -/
#guard_msgs in
def halfAccumulatorWithDefaultOutput : ComputeKernel := triton {
  a = tl.full([16, 16], 2, dtype=tl.float16)
  b = tl.full([16, 16], 3, dtype=tl.float16)
  acc = tl.full([16, 16], 7, dtype=tl.float16)
  acc = tl.dot(a, b, acc)
}

private def model : Algebra Nat where
  literal := fun _ _ _ => 7
  negInf := 0
  binary := fun _ _ _ a b => a + b
  unary := fun _ _ a => a
  cast := fun _ _ _ a => a
  fromNat := fun _ n => n
  fromInt := fun _ n => n.toNat
  fp32Bits := fun _ => 0
  fp32Load := id
  reduceMax := fun _ _ _ _ _ => 0
  reduceSum := fun _ _ _ _ _ => 0
  dot := fun _ {_} {_} {_} {_} _ _ => some (fun _ => 10)
  dotAcc := fun p left right out options {_} {_} {_} {_} _ _ c =>
    some (fun i => ofFloat out (toFloat out (c i) +
      (if p = some .fp64 then 64 else
       if left = .fp16 ∧ right = .fp16 ∧ options.inputPrecision = some "ieee"
       then 200 else 100)))

private def initial : State Nat where
  mem := fun _ _ => .mk .real 0
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d
private def tile (n : Nat) : Op .real [16,16] := .full [16,16] (.const n)
private def fused : Op .real [16,16] := .dotAcc .real .real .real (batch := [])
  (tile 2) (tile 3) (tile 7)
private def separated : Op .real [16,16] := .add .real (.consSame (.consSame .nil))
  (.dot (batch := []) (tile 2) (tile 3)) (tile 7)

theorem fused_result_depends_on_accumulator :
    FP.Structural.evalOp model none fused initial = some (fun _ => 107) := rfl

theorem decomposition_is_not_assumed :
    FP.Structural.evalOp model none fused initial ≠
      FP.Structural.evalOp model none separated initial := by
  intro h
  have := congrArg (fun v => v.map (fun f => f (0,0,PUnit.unit))) h
  change some (107 : Nat) = some 17 at this
  contradiction

theorem unsupported_backend_fails :
    FP.Structural.evalOp { model with dotAcc := fun _ _ _ _ _ {_} {_} {_} {_} _ _ _ => none }
      none fused initial = none := rfl

theorem missing_accumulator_fails :
    FP.Structural.evalOp model none
      (.dotAcc .real .real .real (batch := []) (tile 2) (tile 3)
        (.ref .real [16,16] "missing")) initial = none := rfl

theorem default_profile_reaches_dot_accumulator :
    FP.Structural.evalOp (model.withDefaultPrecision .fp64) none fused initial =
      some (fun _ => 71) := rfl

private def typedFused : Op .fp32 [16,16] :=
  .dotAcc .fp16 .fp16 .fp32 (batch := [])
    (.full [16,16] (.constFloat .fp16 2))
    (.full [16,16] (.constFloat .fp16 3))
    (.full [16,16] (.constFloat .fp32 7)) { inputPrecision := some "ieee" }

theorem typed_channels_and_options_reach_backend :
    FP.Structural.evalOp { model with cast := fun _ _ _ x => x + 1 }
      none typedFused initial = some (fun _ => 207) := rfl

set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false
/-- Actual source execution with a nonzero accumulator; no preinitialized
register or overall execution premise is supplied. -/
theorem full_source_executes : ∃ t,
    FP.Structural.exec (model.withDefaultPrecision .fp32) halfSource initial = some t ∧
      t.regs .real [16,16] "acc" = some (fun _ => 107) := by
  simp [halfSource, FP.Structural.exec, run, step, FP.Structural.evalExpr,
    evalComputeOp, evalOp_unfold, Option.bind, model, initial, Algebra.withDefaultPrecision,
    resolvePrecision, ComputeDType.eraseDType, numeric, bop, FP.Structural.store,
    State.setReg, ofFloat, toFloat]

open FP.ProfiledRewrite
open scoped VeriTile.Spec
private def whereCode (extra : List ComputeStmt) (swapped : Bool) : List ComputeStmt :=
  extra ++ [
    .assign .real [4] "x" (.alg (.load .real (.region "x" (.arange 4)) .none)),
    .assign .real [4] "value" (.alg (.where
      (.lt .nat .scalarR (.arange 4) (.constNat 2))
      (if swapped then .mul .real .scalarL (.const 2) (.ref .real [4] "x")
       else .mul .real .scalarR (.ref .real [4] "x") (.const 2))
      (.broadcast (.const 0) [4]))),
    .store .real [4] (.region "out" (.arange 4)) (.alg (.ref .real [4] "value")) .none]
private def whereProgram (extra : List ComputeStmt) (swapped : Bool) : Program where
  kernel := .mk ["x"] ["out"] (whereCode extra swapped)
  domain := [.mulScalar (Source.prefixBefore (whereCode extra Bool.false) "value")
    4 [] (.ref .real [4] "x") (.const 2)]
private def padding : ComputeStmt := .assign .nat [] "unrelated" (.alg (.constNat 17))

theorem eager_where_tactic (R : FP.ScalarArithmetic.Rules) :
    whereProgram [padding] Bool.false ≡[R] whereProgram [padding] Bool.true := by
  equiv_decompose
  fp_prove

/-- The generic eager-selection rule must not erase its operand guards. -/
example (R : FP.ScalarArithmetic.Rules) : True := by
  fail_if_success
    have : ({ whereProgram [padding] Bool.false with domain := [] } : Program) ≡[R]
        ({ whereProgram [padding] Bool.true with domain := [] } : Program) := by
      equiv_decompose
      fp_prove
  trivial

#guard_msgs (drop info) in
#auditModuleAxioms
end FPDotAccumulatorTests
