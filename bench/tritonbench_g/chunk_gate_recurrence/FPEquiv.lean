import bench.tritonbench_g.chunk_gate_recurrence.Kernels
import bench.tritonbench_g.chunk_gate_recurrence.Proofs.FP

/-! FP equivalence of the complete sources in Kernels. The candidate commutes
only the fp32 addition in acc*d+S or Dacc*d+DS. Source/gate/seed/incoming
storage remain independently fp32, fp16 or bf16, with all casts retained.
Products, DG reductions, pointer updates and stores keep their context.

forwardRequirements checks finite fp32 operands at every reached loop update,
after storage conversion. backwardRequirements checks every reached reverse
update and the final DL update after the entire original loop. These checks
refer to actual intermediate values, with no output-equation premise. Both
proofs use only the admitted add_commute atom. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.Equivalence
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec

section
variable (source gate seed : TypedSource.Storage) (S D O last_kv : RegionName)
    (H N K V BK BV : Nat) (hasSeed : Bool)

def forward : Program where
  kernel := TypedSource.forward source gate seed S D O last_kv H N K V BK BV hasSeed
  domain := FPEquivProofs.forwardRequirements
    (TypedSource.forward source gate seed S D O last_kv H N K V BK BV hasSeed) source gate BK BV

def forwardSumLeft : Program :=
  { forward source gate seed S D O last_kv H N K V BK BV hasSeed with
    kernel := TypedSource.forwardSumLeft source gate seed S D O last_kv H N K V BK BV hasSeed }

specification forward_equiv (R : FP.ScalarArithmetic.Rules) :
    forward source gate seed S D O last_kv H N K V BK BV hasSeed ≡[R]
    forwardSumLeft source gate seed S D O last_kv H N K V BK BV hasSeed := by
  exact FPEquivProofs.forward_equiv source gate seed S D O last_kv H N K V BK BV hasSeed R
end

section
variable (source gate incoming : TypedSource.Storage) (S D DI DG DL DS : RegionName)
    (H N K V BK BV : Nat)

def backward : Program where
  kernel := TypedSource.backward source gate incoming S D DI DG DL DS H N K V BK BV
  domain := FPEquivProofs.backwardRequirements
    (TypedSource.backward source gate incoming S D DI DG DL DS H N K V BK BV) gate incoming BK BV

def backwardSumLeft : Program :=
  { backward source gate incoming S D DI DG DL DS H N K V BK BV with
    kernel := TypedSource.backwardSumLeft source gate incoming S D DI DG DL DS H N K V BK BV }

specification backward_equiv (R : FP.ScalarArithmetic.Rules) :
    backward source gate incoming S D DI DG DL DS H N K V BK BV ≡[R]
    backwardSumLeft source gate incoming S D DI DG DL DS H N K V BK BV := by
  exact FPEquivProofs.backward_equiv source gate incoming S D DI DG DL DS H N K V BK BV R
end

#print_fp_assumptions forward_equiv
#print_fp_assumptions backward_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.Equivalence
