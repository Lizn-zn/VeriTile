import VeriTile.Triton.DSL
import VeriTile.Triton.Float.Structural
import VeriTile.Triton.Float.ExecutionProfile
import VeriTile.Triton.Float.Erasure
import VeriTile.Meta.StatementAudit

/-! Reduction fidelity: a floating sum retains its dtype, a full tile is
flattened once, explicit widening is visible, and the FP interpreter does not
identify flat and nested reductions. The distinguishing backend is a syntax
witness, not a GPU implementation or an experimental rejection. -/
namespace TypedSumTests
open VeriTile.Triton FP.Structural
set_option maxHeartbeats 1000000

def halfSource (M N : Nat) : ComputeKernel := triton {
  x = tl.zeros([$(M), $(N)], dtype=tl.float16)
  whole = tl.sum(x)
  rows = tl.sum(x, axis=1)
  kept = tl.sum(x, keep_dims=true)
  promoted = tl.sum(x, dtype=tl.float32)
  explicitNone = tl.sum(x, axis=None)
}

private def halfInput (M N : Nat) : Op .fp16 [M,N] := .ref .fp16 [M,N] "x"
private def wholeHalf (M N : Nat) : Op .fp16 [] :=
  .reduceSumTyped .fp16 (shape := [M*N]) ⟨0, by simp⟩ Bool.false
    (.reshape [M*N] (halfInput M N))

theorem whole_keeps_half (M N : Nat) : (halfSource M N).surfaceBody[1]? =
    some (.assign .fp16 [] "whole" (.alg (wholeHalf M N))) := rfl

theorem rows_keep_half (M N : Nat) : (halfSource M N).surfaceBody[2]? =
    some (.assign .fp16 [M] "rows" (.alg
      (.reduceSumTyped .fp16 ⟨1, by simp⟩ Bool.false (halfInput M N)))) := rfl

theorem kept_is_one_reduction (M N : Nat) : (halfSource M N).surfaceBody[3]? =
    some (.assign .fp16 [1,1] "kept" (.alg (.reshape [1,1] (wholeHalf M N)))) := rfl

theorem explicit_promotion (M N : Nat) : (halfSource M N).surfaceBody[4]? =
    some (.assign .fp32 [] "promoted" (.alg
      (.reduceSumTyped .fp32 (shape := [M*N]) ⟨0, by simp⟩ Bool.false
        (.reshape [M*N] (.castFloat .fp16 .fp32 (halfInput M N)))))) := rfl

theorem explicit_none_matches_default (M N : Nat) : (halfSource M N).surfaceBody[5]? =
    some (.assign .fp16 [] "explicitNone" (.alg (wholeHalf M N))) := rfl

def bfloatSource : ComputeKernel := triton {
  x = tl.zeros([2, 4], dtype=tl.bfloat16)
  y = tl.sum(x)
  z = y * y
}

theorem bfloat_sum_controls_next_operation : bfloatSource.surfaceBody[2]? =
    some (.assign .bf16 [] "z" (.alg (.mul .bf16 .nil
      (.ref .bf16 [] "y") (.ref .bf16 [] "y")))) := rfl

def fp32Source (M N : Nat) : ComputeKernel := triton {
  x = tl.zeros([$(M), $(N)], dtype=tl.float32)
  y = tl.sum(x)
  z = tl.sum(x, dtype=tl.float32)
}

private def whole32 (M N : Nat) : Op .real [] :=
  .reduceSum (shape := [M*N]) ⟨0, by simp⟩ Bool.false
    (.reshape [M*N] (.ref .real [M,N] "x"))

theorem fp32_annotation_survives (M N : Nat) : (fp32Source M N).surfaceBody[1]? =
    some (.assign .real [] "y" (.compute (.alg .fp32 (whole32 M N)))) := rfl

theorem fp32_same_dtype_needs_no_cast (M N : Nat) : (fp32Source M N).surfaceBody[2]? =
    some (.assign .real [] "z" (.compute (.alg .fp32 (whole32 M N)))) := rfl

private def code (d : FloatDType) : Nat :=
  match d with
  | .real => 64 | .fp32 => 32 | .fp16 => 16 | .bf16 => 116 | .f8e4 => 84 | .f8e5 => 85

private def payloadSum {shape : TileShape} (v : Values Nat .real shape) : Nat :=
  (TileShape.allIndices shape).foldl (fun acc i => acc + v i) 0

private def model : Algebra Nat where
  literal := fun _ _ _ => 0
  negInf := 0
  binary := fun _ _ _ a b => a + b
  unary := fun _ _ a => a
  cast := fun _ _ _ a => a + 10000
  fromNat := fun _ n => n
  fromInt := fun _ n => n.toNat
  fp32Bits := fun _ => 0
  fp32Load := id
  reduceMax := fun _ _ _ _ _ => 0
  reduceSum := fun _ {_} _ _ v _ => 100 + payloadSum v
  reduceSumTyped := fun p d {_} _ _ v => some (fun _ => ofFloat d
    (code d + (if p = some .fp64 then 1000 else 0) + payloadSum (fun i => toFloat d (v i))))

private def initial : State Nat where
  mem := fun _ _ => .mk .real 0
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

private def typed (d : FloatDType) : Op d.toTileDType [] :=
  .reduceSumTyped d (shape := [4]) ⟨0, by simp⟩ Bool.false
    (.full [4] (.constFloat d 0))

theorem typed_sum_has_no_extra_cast :
    FP.Structural.evalOp model none (typed .fp16) initial = some (fun _ => ofFloat .fp16 16) := rfl

theorem bfloat_uses_own_backend :
    FP.Structural.evalOp model none (typed .bf16) initial = some (fun _ => ofFloat .bf16 116) := rfl

theorem fp32_uses_own_backend :
    FP.Structural.evalOp model none (typed .fp32) initial = some (fun _ => ofFloat .fp32 32) := rfl

theorem profile_reaches_typed_sum :
    FP.Structural.evalOp (model.withDefaultPrecision .fp64) none (typed .fp16) initial =
      some (fun _ => ofFloat .fp16 1016) := rfl

theorem unsupported_typed_sum_fails :
    FP.Structural.evalOp { model with reduceSumTyped := fun _ _ {_} _ _ _ => none }
      none (typed .fp16) initial = none := rfl

theorem missing_input_fails :
    FP.Structural.evalOp model none (wholeHalf 2 2) initial = none := rfl

private def matrixState : State Nat :=
  initial.setReg "matrix" .fp16 [2,3] (fun i => i.1.val * 3 + i.2.1.val)

theorem reshape_keeps_row_major_order :
    (FP.Structural.evalOp model none
      (.reshape [6] (.ref .fp16 [2,3] "matrix")) matrixState).map
        (fun f => (TileShape.allIndices [6]).map f) = some [0,1,2,3,4,5] := rfl

theorem mismatched_reshape_fails :
    FP.Structural.evalOp model none
      (.reshape [5] (.ref .fp16 [2,3] "matrix")) matrixState = none := rfl

theorem empty_reshape_succeeds :
    (FP.Structural.evalOp model none
      (.reshape [0] (.full [0,3] (.constFloat .fp16 0))) initial).isSome = Bool.true := rfl

/-- The frontend's complete program executes with typed reduction support;
the widened sum sees four extra casts after the initial typed-zero casts. -/
theorem source_executes : ∃ t,
    FP.Structural.exec model (halfSource 2 2) initial = some t ∧
    t.regs .fp16 [] "whole" = some (fun _ => 40016) ∧
    t.regs .fp32 [] "promoted" = some (fun _ => 80032) ∧
    t.mem = initial.mem := by
  refine ⟨(FP.Structural.exec model (halfSource 2 2) initial).getD initial, ?_, ?_, ?_, ?_⟩
  all_goals
    simp [halfSource, FP.Structural.exec, run, step, FP.Structural.evalExpr,
      evalOp_unfold, Option.bind, model, initial, State.setReg, payloadSum, TileShape.allIndices,
      TileShape.reshapeIndex, TileShape.linearIndex, TileShape.product,
      TileShape.eraseAxis, TileShape.reduceShape, ofFloat, toFloat, code]

private def flat : Op .real [] :=
  .reduceSum (shape := [4]) ⟨0, by simp⟩ Bool.false
    (.reshape [4] (.full [2,2] (.const 0)))
private def nested : Op .real [] :=
  .reduceSum (shape := [2]) ⟨0, by simp⟩ Bool.false
    (.reduceSum (shape := [2,2]) ⟨1, by simp⟩ Bool.false (.full [2,2] (.const 0)))

theorem flat_and_nested_are_not_implicitly_equal :
    FP.Structural.evalOp model none flat initial ≠ FP.Structural.evalOp model none nested initial := by
  intro h
  have hv := congrArg (fun v => v.map (fun f => f PUnit.unit)) h
  change some (100 : Nat) = some 300 at hv
  contradiction

theorem erasure_is_real_sum : (typed .fp16).eraseDType =
    Op.reduceSum (shape := [4]) ⟨0, by simp⟩ Bool.false (.full [4] (.const 0)) := by
  simp only [typed, Op.eraseDType.eq_def]
  rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end TypedSumTests
