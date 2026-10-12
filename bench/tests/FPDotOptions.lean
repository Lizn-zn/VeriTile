import VeriTile.Triton
import VeriTile.Triton.Float.ExecutionProfile
import VeriTile.Meta.StatementAudit

/-! Source dot options remain visible to the FP backend, including under an
outer fp32 addition. Real evaluation still uses the mathematical matrix product.
The witness backend below distinguishes operations; it is not a GPU emulator. -/
namespace FPDotOptionsTests
open VeriTile.Triton FP.Structural

def source (out : RegionName) : ComputeKernel := triton {
  a = tl.full([16, 16], 1, dtype=tl.float16)
  b = tl.full([16, 16], 1, dtype=tl.float16)
  acc = tl.zeros([16, 16], dtype=tl.float32)
  acc += tl.dot(a, b, out_dtype=tl.float16)
  i = tl.arange(0, 16)
  offsets = i[:, None] * 16 + i[None, :]
  tl.store(out + offsets, acc)
}

private def a : Op .fp16 [16, 16] := .ref .fp16 [16, 16] "a"
private def b : Op .fp16 [16, 16] := .ref .fp16 [16, 16] "b"

theorem nested_source_options : (source "out").surfaceBody[3]? = some
    (.assign .real [16, 16] "acc" (.compute (.alg .fp32
      (.add .real (.consSame (.consSame .nil)) (.ref .real [16, 16] "acc")
        (.castFloat .fp16 .real (.dotTyped .fp16 .fp16 .fp16
          (options := { outDType := some .fp16 }) (batch := []) a b)))))) := rfl

def typedResults : ComputeKernel := triton {
  a = tl.full([16, 16], 0, dtype=tl.float16)
  b = tl.full([16, 16], 0, dtype=tl.float16)
  c = tl.dot(a, b, out_dtype=tl.float16)
  squared = c * c
  d = tl.dot(a, b)
  mixed = d + c
}

theorem result_controls_following_arithmetic : typedResults.surfaceBody[3]? =
    some (.assign .fp16 [16, 16] "squared" (.alg
      (.mul .fp16 (.consSame (.consSame .nil))
        (.ref .fp16 [16, 16] "c") (.ref .fp16 [16, 16] "c")))) := rfl

theorem default_typed_result_is_fp32 : typedResults.surfaceBody[4]? =
    some (.assign .fp32 [16, 16] "d" (.alg
      (.dotTyped .fp16 .fp16 .fp32 (batch := []) a b))) := rfl

theorem mixed_result_promotes_to_fp32 : typedResults.surfaceBody[5]? =
    some (.assign .fp32 [16, 16] "mixed" (.alg
      (.add .fp32 (.consSame (.consSame .nil)) (.ref .fp32 [16, 16] "d")
        (.castFloat .fp16 .fp32 (.ref .fp16 [16, 16] "c"))))) := rfl

def bf16Result : ComputeKernel := triton {
  a = tl.full([16, 16], 0, dtype=tl.bfloat16)
  b = tl.full([16, 16], 0, dtype=tl.bfloat16)
  c = tl.dot(a, b)
}

theorem bf16_operands_produce_fp32 : bf16Result.surfaceBody[2]? =
    some (.assign .fp32 [16, 16] "c" (.alg
      (.dotTyped .bf16 .bf16 .fp32 (batch := [])
        (.ref .bf16 [16, 16] "a") (.ref .bf16 [16, 16] "b")))) := rfl

def modes : ComputeKernel := triton {
  a = tl.full([16, 16], 0, dtype=tl.float32)
  b = tl.full([16, 16], 0, dtype=tl.float32)
  c = tl.dot(a, b, out_dtype=tl.float32, input_precision="ieee", max_num_imprecise_acc=8)
  d = tl.dot(a, b, allow_tf32=False)
}

private def explicitOptions : DotOptions where
  outDType := some .fp32
  inputPrecision := some "ieee"
  maxNumImpreciseAcc := some 8

theorem source_precision_options : (modes.surfaceBody[2]?, modes.surfaceBody[3]?) =
    (some (.assign .real [16, 16] "c" (.alg
      (.dot (options := explicitOptions) (batch := [])
        (.ref .real [16, 16] "a") (.ref .real [16, 16] "b")))),
     some (.assign .real [16, 16] "d" (.alg
      (.dot (options := { allowTF32 := some Bool.false }) (batch := [])
        (.ref .real [16, 16] "a") (.ref .real [16, 16] "b"))))) := rfl

private def input : Op .real [16, 16] := .full [16, 16] (.const 0)
private def dot (options : DotOptions) : Op .real [16, 16] :=
  .dot (options := options) (batch := []) input input

private def model : Algebra Nat where
  literal := fun _ _ _ => 0
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
  dot := fun _ {_} {_} {_} {_} _ _ => some (fun _ => 0)
  dotWithOptions := fun p options {_} {_} {_} {_} _ _ =>
    some (fun _ => if p = some .fp64 then 64
      else if options.outDType = some .fp16 then 16 else 32)
  dotTyped := fun p left right out _ {_} {_} {_} {_} _ _ =>
    some (fun _ => ofFloat out (if p = some .fp64 then 64
      else if left = .fp16 ∧ right = .fp16 ∧ out = .fp16 then 16 else 32))

private def initial : State Nat where
  mem := fun _ _ => .mk .real 0
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

theorem configured_backend :
    FP.Structural.evalOp model none (dot { outDType := some .fp16 }) initial =
      some (fun _ => 16) := rfl

theorem default_backend :
    FP.Structural.evalOp model none (dot {}) initial = some (fun _ => 0) := rfl

-- The default backend does not provide support for an explicitly chosen mode.
theorem unsupported_configuration_fails :
    FP.Structural.evalOp { model with
      dotWithOptions := fun _ _ {_} {_} {_} {_} _ _ => none }
      none (dot { outDType := some .fp16 }) initial = none := rfl

theorem output_mode_is_not_a_post_cast :
    FP.Structural.evalOp model none (dot { outDType := some .fp16 }) initial ≠
      FP.Structural.evalOp model none (.castFloat .fp16 .real
        (.castFloat .real .fp16 (dot {}))) initial := by
  intro h
  have hv := congrArg (fun v => v.map (fun f => f (0, 0, PUnit.unit))) h
  change some (16 : Nat) = some 0 at hv
  contradiction

theorem profile_reaches_configured_dot :
    FP.Structural.evalOp (model.withDefaultPrecision .fp64) none
      (dot { outDType := some .fp16 }) initial = some (fun _ => 64) := rfl

theorem dtype_erasure_retains_options :
    (dot { outDType := some .fp16, allowTF32 := some Bool.false }).eraseDType =
      dot { outDType := some .fp16, allowTF32 := some Bool.false } := by
  conv_lhs => unfold dot Op.eraseDType
  simp [dot, input, Op.eraseDType]
  rfl

theorem flatten_retains_options (A : FlatAlloc) (options : DotOptions) :
    A.flattenOp (dot options) = dot options := by
  conv_lhs => unfold dot FlatAlloc.flattenOp
  simp [dot, input, FlatAlloc.flattenOp]

theorem real_projection_ignores_options (s : BlockState) (options : DotOptions) :
    VeriTile.Triton.evalOp (dot options) s = VeriTile.Triton.evalOp (dot {}) s := by
  exact (evalOp_dot (options := options) [] input input s).trans
    (evalOp_dot (options := {}) [] input input s).symm

theorem configured_dot_still_requires_operands :
    FP.Structural.evalOp model none
      (.dot (options := explicitOptions) (batch := [])
        (.ref .real [16, 16] "missing") input) initial = none := rfl

private def typedDot : Op .fp16 [16, 16] :=
  .dotTyped .fp16 .fp16 .fp16 (batch := [])
    (.full [16, 16] (.constFloat .fp16 0))
    (.full [16, 16] (.constFloat .fp16 0)) { outDType := some .fp16 }

theorem typed_backend_has_no_artificial_cast :
    FP.Structural.evalOp { model with cast := fun _ _ _ x => x + 1 }
      none typedDot initial = some (fun _ => 16) := rfl

theorem typed_backend_requires_support :
    FP.Structural.evalOp { model with
      dotTyped := fun _ _ _ _ _ {_} {_} {_} {_} _ _ => none }
      none typedDot initial = none := rfl

theorem typed_profile_reaches_backend :
    FP.Structural.evalOp (model.withDefaultPrecision .fp64) none typedDot initial =
      some (fun _ => 64) := rfl

theorem flatten_retains_typed_channels (A : FlatAlloc) :
    A.flattenOp typedDot = typedDot := by
  conv_lhs => unfold typedDot FlatAlloc.flattenOp
  simp [typedDot, FlatAlloc.flattenOp]
  erw [FlatAlloc.flattenOp]

set_option maxHeartbeats 1600000
set_option linter.unusedSimpArgs false

theorem full_source_executes : ∃ t,
    FP.Structural.exec (model.withDefaultPrecision .fp32) (source "out") initial = some t := by
  simp [source, FP.Structural.exec, run, step, FP.Structural.evalExpr,
    evalComputeOp, evalOp_unfold, Option.bind, model, initial, Algebra.withDefaultPrecision,
    resolvePrecision, ComputeDType.eraseDType, numeric, bop, FP.Structural.store, State.setReg]

#guard_msgs (drop info) in
#auditModuleAxioms
end FPDotOptionsTests
