/-
FloatDTypeAdd: fp32 addition with symbolic kernel dimensions.
The kernel retains its explicit fp32 output quantization; only x+y changes to y+x.
Both kernel definitions are written here so this example is self-contained.

The experiment selects ADD-COMMUTE as an atomic assumption. The proof uses
that assumption at the kernel's symbolic tile size, without matching it to
the experiment's shape or launch configuration. No new global axiom.
-/
import VeriTile.Triton.DSL
import VeriTile.Meta.StatementAudit
import VeriTile.Triton.Float.Equivalence
import VeriTile.Triton.Float.ReportedAdmission

namespace VeriTile.Bench.Examples.FloatDTypeAddFPEquiv

open VeriTile Triton
open scoped VeriTile.Spec

abbrev admitted := FP.ReportedAdmission.fp32_add_commute

/-- Original FloatDTypeAdd addition, transcribed with fp32 regions. -/
def originalKernel (blockSize : Nat) : ComputeKernel :=
  let x_ptr : Region .fp32 := ⟨"x"⟩
  let y_ptr : Region .fp32 := ⟨"y"⟩
  let out_ptr : Region .fp32 := ⟨"out"⟩
  triton {
  pid = tl.program_id(axis=0)
  offs = pid * $(blockSize) + tl.arange(0, $(blockSize))
  x = tl.load(x_ptr + offs)
  y = tl.load(y_ptr + offs)
  out = x + y
  tl.store(out_ptr + offs, (out).round_to(tl.float32))
}

/-- The sole rewrite is out = y + x. Addresses/masks are unchanged. -/
def optimizedKernel (blockSize : Nat) : ComputeKernel :=
  let x_ptr : Region .fp32 := ⟨"x"⟩
  let y_ptr : Region .fp32 := ⟨"y"⟩
  let out_ptr : Region .fp32 := ⟨"out"⟩
  triton {
  pid = tl.program_id(axis=0)
  offs = pid * $(blockSize) + tl.arange(0, $(blockSize))
  x = tl.load(x_ptr + offs)
  y = tl.load(y_ptr + offs)
  out = y + x
  tl.store(out_ptr + offs, (out).round_to(tl.float32))
}

abbrev body := Spec.ProgramSyntax.body (Program := ComputeKernel)

/-- Parameterized local fp32 addition: register renaming is explicit. -/
def addFragment (blockSize : Nat) (out x y : RegName) : List ComputeStmt :=
  [.assign .real [blockSize] out
    (.compute (.alg .fp32 (.add .real (.consSame .nil)
      (.ref .real [blockSize] x) (.ref .real [blockSize] y))))]

def originalAdd (blockSize : Nat) := addFragment blockSize "out" "x" "y"
def optimizedAdd (blockSize : Nat) := addFragment blockSize "out" "y" "x"
def beforeAdd (blockSize : Nat) : List ComputeStmt :=
  (body (originalKernel blockSize)).take 4
def afterAdd (blockSize : Nat) : List ComputeStmt :=
  (body (originalKernel blockSize)).drop 5

theorem original_decomposition (blockSize : Nat) :
    body (originalKernel blockSize) =
      beforeAdd blockSize ++ originalAdd blockSize ++ afterAdd blockSize := rfl

theorem optimized_decomposition (blockSize : Nat) :
    body (optimizedKernel blockSize) =
      beforeAdd blockSize ++ optimizedAdd blockSize ++ afterAdd blockSize := rfl

/-- Admission selects the fp32 operation. Experimental shape and launch are
provenance for that selection, not restrictions on the subsequent derivation. -/
theorem admitted_operation_matches :
    admitted.ruleID = "ADD-COMMUTE" ∧ admitted.input = "fp32" ∧
    admitted.compute = "fp32" ∧ admitted.accumulator = "fp32" ∧
    admitted.output = "fp32" := by decide

/-- Only this frozen accepted row is bound, never a user-supplied PASS label. -/
def addCommute (blockSize : Nat) : Spec.RuleEntry ComputeStmt :=
  admitted.bind (originalAdd blockSize) (optimizedAdd blockSize)

/-- The experiment-selected atom, instantiated at a symbolic tile size.
`blockSize` only indexes the typed syntax; no experiment-size condition remains.
This is the atomic modeling assumption exposed by #print_fp_assumptions. -/
structure Rules (blockSize : Nat) where
  add_comm : Spec.EvidenceValidated (addCommute blockSize).rule (addCommute blockSize).evidence

def Rules.assumptions {blockSize : Nat} (_ : Rules blockSize) :
    Spec.Assumptions ComputeStmt := [addCommute blockSize]

instance {blockSize : Nat} :
    CoeOut (Rules blockSize) (Spec.Assumptions (Spec.ProgramSyntax.Statement ComputeKernel)) :=
  ⟨Rules.assumptions⟩

@[spec_rule] theorem admitted_add_commute {blockSize : Nat} (R : Rules blockSize) :
    Spec.Derivation R.assumptions (originalAdd blockSize) (optimizedAdd blockSize) := by
  exact .atom (addCommute blockSize) (by simp [Rules.assumptions])
    (admitted.admit (originalAdd blockSize) (optimizedAdd blockSize) R.add_comm)

/-- Public specification: a kernel equivalence derived from one accepted atom. -/
specification float_add_equiv (blockSize : Nat) (R : Rules blockSize) :
    originalKernel blockSize ≡[R] optimizedKernel blockSize := by
  refine ⟨rfl, ?_⟩
  change Spec.Derivation R.assumptions
    (body (originalKernel blockSize)) (body (optimizedKernel blockSize))
  rw [original_decomposition, optimized_decomposition]
  exact .frame (beforeAdd blockSize) (afterAdd blockSize) (admitted_add_commute R)

#print_fp_assumptions float_add_equiv
-- Keep the proof audit active without adding its success log to the example.
#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.Examples.FloatDTypeAddFPEquiv
