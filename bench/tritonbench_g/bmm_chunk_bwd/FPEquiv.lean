import bench.tritonbench_g.bmm_chunk_bwd.Kernels
import bench.tritonbench_g.bmm_chunk_bwd.Proofs.FP
import VeriTile.Meta.StatementAudit

/-! Whole-program admitted FP equivalence: commute acc+res after the complete
chunk loop and the actual residual load/conversion. Only the reached fp32
operands must be finite. All nine input storage pairs, dot precision, casts,
partial tails, optional branch, output conversion and memory effects remain.
No dot/reduction reassociation or concrete IEEE equality is claimed. -/
namespace VeriTile.Bench.TritonBenchG.BmmChunkBwd.Equivalence
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec

section
variable (aStorage doutStorage : TypedSource.Storage) (dtype out_dtype : FloatDType)
    (a dout db res : RegionName)
    (seqlen chunk K groups sab sas sah sak sdb sdc sdh sdm sdn sob sos soh sok srb srs srh srk BM BN BCS : Nat)
    (hasResidual : Bool)

def original : Program where
  kernel := TypedSource.original aStorage doutStorage dtype out_dtype a dout db res
    seqlen chunk K groups sab sas sah sak sdb sdc sdh sdm sdn sob sos soh sok srb srs srh srk BM BN BCS hasResidual
  domain := FPEquivProofs.requirementsFor (TypedSource.original aStorage doutStorage dtype out_dtype a dout db res
    seqlen chunk K groups sab sas sah sak sdb sdc sdh sdm sdn sob sos soh sok srb srs srh srk BM BN BCS hasResidual) BM BN

def residualLeft : Program :=
  { original aStorage doutStorage dtype out_dtype a dout db res
      seqlen chunk K groups sab sas sah sak sdb sdc sdh sdm sdn sob sos soh sok srb srs srh srk BM BN BCS hasResidual with
    kernel := TypedSource.residualLeft aStorage doutStorage dtype out_dtype a dout db res
      seqlen chunk K groups sab sas sah sak sdb sdc sdh sdm sdn sob sos soh sok srb srs srh srk BM BN BCS hasResidual }


specification bmm_residual_equiv (R : FP.ScalarArithmetic.Rules) :
    original aStorage doutStorage dtype out_dtype a dout db res
      seqlen chunk K groups sab sas sah sak sdb sdc sdh sdm sdn sob sos soh sok srb srs srh srk BM BN BCS hasResidual ≡[R]
    residualLeft aStorage doutStorage dtype out_dtype a dout db res
      seqlen chunk K groups sab sas sah sak sdb sdc sdh sdm sdn sob sos soh sok srb srs srh srk BM BN BCS hasResidual := by
  exact FPEquivProofs.residual_equiv aStorage doutStorage dtype out_dtype a dout db res
      seqlen chunk K groups sab sas sah sak sdb sdc sdh sdm sdn sob sos soh sok srb srs srh srk BM BN BCS hasResidual R

end
#print_fp_assumptions bmm_residual_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.BmmChunkBwd.Equivalence
