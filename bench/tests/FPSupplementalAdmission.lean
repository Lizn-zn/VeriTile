/- Boundary checks for conditional admission and precision-preserving rewrites. -/
import bench.examples.SoftmaxReciprocalFPEquiv
import bench.examples.FloatDTypeSoftmaxFPEquiv

namespace FPSupplementalAdmissionTests
open VeriTile Triton
open FP.Structural FP.Guarded FP.Reciprocal
open scoped VeriTile.Spec

-- Binary64 payloads keep their width and partial Real projection.
example : ComputeDType.fp64.width ≠ ComputeDType.fp32.width := by decide
#guard Float64Bits.decodeRat ⟨BitVec.ofNat 64 0x3ff0000000000000⟩ = some 1
#guard Float64Bits.decodeRat ⟨BitVec.ofNat 64 0xc000000000000000⟩ = some (-2)
#guard Float64Bits.decodeRat ⟨BitVec.ofNat 64 0⟩ = none
#guard Float64Bits.decodeRat ⟨BitVec.ofNat 64 0x7ff0000000000000⟩ = none
example : ComputeOp.bitcastPayload .fp64 .fp32 ⟨BitVec.ofNat 64 0x3ff0000000000000⟩ = none := rfl

-- The measured casted binary64 atom cannot be used as an uncast binary64 law.
example : (entry .fp64_fp32).rule.lhs = [lhs .fp64_fp32] := rfl
example : (entry .fp64_fp32).rule.rhs = [rhs .fp64_fp32] := rfl
example : (entry .fp64_fp32).rule.lhs ≠ [lhs .fp32] := by
  intro h
  have hn := congrArg (fun xs => xs.map (fun x => x.code.length)) h
  norm_num [entry, report, FP.ReportedScalarRule.bind, FP.ReportedRule.bind,
    lhs, FP.SupplementalAdmission.fp64_fp64_fp32_div_mul_rcp] at hn

private def M : Algebra Nat where
  literal := fun _ _ _ => 1
  negInf := 0
  binary := fun _ _ op a b => match op with
    | .div => if b = 0 then 7 else a / b
    | .mul => a * b
    | _ => 0
  unary := fun _ _ a => a
  cast := fun _ _ _ a => a + 1
  fromNat := fun _ a => a
  fromInt := fun _ a => a.toNat
  fp32Bits := fun a => a.bits.toNat
  fp32Load := id
  reduceMax := fun _ _ _ _ _ => 0
  reduceSum := fun _ _ _ _ _ => 0

private def D : Domain Nat
  | .finite => fun _ => True
  | .nonzero => fun a => a ≠ 0
  | .positive => fun a => 0 < a

private def initial : State Nat where
  mem := fun _ _ => .mk .real 0
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

-- Dropping the domain would add a false numerical identity even for a tiny model.
example : quotient .fp32 M 2 0 ≠ reciprocal .fp32 M 2 0 := by decide
example : ¬ ScalarDomain D guards
    ((initial.setReg "a" .real [] (fun _ => 2)).setReg "b" .real [] (fun _ => 0)) := by
  intro h
  have hb := h ⟨"b", .nonzero⟩ (by simp [guards])
  simp [State.setReg, D] at hb

-- Finite/nonzero inputs are possible; the contract does not force an empty domain.
example : ScalarDomain D guards
    ((initial.setReg "a" .real [] (fun _ => 2)).setReg "b" .real [] (fun _ => 1)) := by
  intro g hg
  simp [guards] at hg
  rcases hg with rfl | rfl | rfl <;> simp [State.setReg, D]

-- The final cast is observable; an admitted casted equality cannot erase it.
example : quotient .fp64_fp32 M 2 1 ≠ M.binary (some .fp64) .real .div 2 1 := by decide

-- Both examples retain symbolic dimensions and their actual input contracts.
open VeriTile.Bench.Examples
example (B : Nat) (hB : 0 < B) (R : SoftmaxReciprocalFPEquiv.Rules) :
    SoftmaxReciprocalFPEquiv.originalIO B ≡[R] SoftmaxReciprocalFPEquiv.reciprocalIO B :=
  SoftmaxReciprocalFPEquiv.softmax_reciprocal_equiv R B hB
example (B : Nat) (hB : 0 < B) (R : FloatDTypeSoftmaxFPEquiv.Rules) :
    FloatDTypeSoftmaxFPEquiv.originalIO B ≡[R] FloatDTypeSoftmaxFPEquiv.reciprocalIO B :=
  FloatDTypeSoftmaxFPEquiv.softmax_reciprocal_equiv R B hB

-- An input-domain change is a public signature change, not a structural rewrite.
example (B : Nat) :
    (Spec.ProgramSyntax.signature (SoftmaxReciprocalFPEquiv.originalIO B)).2.guards.length = 3 := rfl

-- Precision boundaries of the actual DSL output are retained in the AST.
private def precisionOf : ComputeStmt → Option ComputeDType
  | .assign _ _ _ (.compute (.alg d _)) => some d
  | .assign _ _ _ (.compute (.load d _ _)) => some d
  | .store _ _ _ (.compute (.alg d _)) _ => some d
  | _ => none

example (B : Nat) :
    ((FloatDTypeSoftmaxFPEquiv.originalKernel "x" "y" B).surfaceBody.map precisionOf) =
      [none, none, some .fp32, some .fp64, some .fp64, some .fp64,
       some .fp64, some .fp64, some .fp32] := rfl
example (B : Nat) :
    ((FloatDTypeSoftmaxFPEquiv.reciprocalKernel "x" "y" B).surfaceBody.map precisionOf) =
      [none, none, some .fp32, some .fp64, some .fp64, some .fp64,
       some .fp64, some .fp64, some .fp64, some .fp32] := rfl

-- No correctness module supplies a real-valued numerical law to these FP proofs.
open Lean Elab Command in
run_cmd do
  let env ← getEnv
  for n in [`VeriTile.Bench.Examples.SoftmaxReciprocalCorrect.divIO,
            `VeriTile.Bench.Examples.FloatDTypeSoftmaxCorrect.divIO] do
    if env.contains n then throwError "FP examples imported a Correct counterpart"

#axiomsClean FP.Reciprocal.apply_rule
#axiomsClean FP.Reciprocal.admitted
#axiomsClean SoftmaxReciprocalFPEquiv.softmax_reciprocal_equiv
#axiomsClean FloatDTypeSoftmaxFPEquiv.softmax_reciprocal_equiv
end FPSupplementalAdmissionTests
