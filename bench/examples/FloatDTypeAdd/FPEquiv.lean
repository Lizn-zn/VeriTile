import bench.examples.FloatDTypeAdd.Kernels
/-
FloatDTypeAdd: fp32 addition with symbolic kernel dimensions.
The kernel retains its explicit fp32 output quantization; only x+y changes to y+x.
Both implementations are defined in Kernels.lean and shared by the proofs.

The experiment selects ADD-COMMUTE as an atomic assumption. The proof uses
that assumption at the kernel's symbolic tile size, without matching it to
the experiment's shape or launch configuration. No new global axiom.
-/
import VeriTile.Triton.DSL
import VeriTile.Meta.StatementAudit
import VeriTile.Meta.FPProve
import VeriTile.Triton.Float.Equivalence
import VeriTile.Triton.Float.ScalarArithmetic

namespace VeriTile.Bench.Examples.FloatDTypeAddFPEquiv
open VeriTile.Bench.Examples.FloatDTypeAdd.Kernels
open VeriTile Triton
open scoped VeriTile.Spec

abbrev admitted := (FP.ScalarArithmetic.report .addCommute (by decide)).report

/-- Parameterized local fp32 addition: register renaming is explicit. -/
def addFragment (blockSize : Nat) (out x y : RegName) : List ComputeStmt :=
  [.assign .real [blockSize] out
    (.compute (.alg .fp32 (.add .real (.consSame .nil)
      (.ref .real [blockSize] x) (.ref .real [blockSize] y))))]

def originalAdd (blockSize : Nat) := addFragment blockSize "out" "x" "y"
def optimizedAdd (blockSize : Nat) := addFragment blockSize "out" "y" "x"

/-- The candidate catalog selects the accepted fp32 relation before it is instantiated. -/
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
  equiv_decompose
  all_goals fp_prove

#print_fp_assumptions float_add_equiv
-- Keep the proof audit active without adding its success log to the example.
#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.Examples.FloatDTypeAddFPEquiv
