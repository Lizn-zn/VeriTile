import FifthSphDraft.Kernels
import bench.tritonbench_g.destindex_copy.Proofs.IndexRewrite
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.FPProof
open VeriTile Triton FP.ProfiledRewrite TypedSource
open scoped VeriTile.Spec
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 30000

def assumptions : Spec.Assumptions FP.GuardedFragment := []

def forwardProgram (input output : Storage) (swapped : Bool) (X Y : RegionName) (B N M off stride : Nat) : Program where
  kernel := forward input output swapped X Y B N M off stride
  domain := []

theorem forward_equivalent (input output : Storage) (X Y : RegionName) (B N M off stride : Nat) :
    forwardProgram input output Bool.false X Y B N M off stride ≡[assumptions]
    forwardProgram input output Bool.true X Y B N M off stride := by
  cases input <;> cases output <;> equiv_decompose <;> fp_prove

#print_fp_assumptions forward_equivalent

def backwardProgram (input gradient upstream : Storage) (swapped : Bool) (X dX g : RegionName) (B N M off stride : Nat) : Program where
  kernel := backward input gradient upstream swapped X dX g B N M off stride
  domain := []

theorem backward_equivalent (input gradient upstream : Storage) (X dX g : RegionName) (B N M off stride : Nat) :
    backwardProgram input gradient upstream Bool.false X dX g B N M off stride ≡[assumptions]
    backwardProgram input gradient upstream Bool.true X dX g B N M off stride := by
  cases input <;> cases gradient <;> cases upstream <;> equiv_decompose <;> fp_prove

#print_fp_assumptions backward_equivalent

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.FPProof
