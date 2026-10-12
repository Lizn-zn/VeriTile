import bench.tritonbench_g.bmm_chunk_bwd.Kernels
import VeriTile.Triton.Float.Erasure

/-! Precision erasure of the complete source, with the in-body chunk limit,
all partial-tile masks and the optional residual. This connects source dtypes
to the real program; it is not yet the independent matrix-product IO proof. -/
namespace VeriTile.Bench.TritonBenchG.BmmChunkBwd.RealSource
open VeriTile.Triton
set_option maxHeartbeats 8000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

section
variable (aStorage doutStorage : TypedSource.Storage) (swapped : Bool) (dtype out_dtype : FloatDType)
    (a dout db res : RegionName)
    (seqlen chunk K groups sab sas sah sak sdb sdc sdh sdm sdn sob sos soh sok srb srs srh srk BM BN BCS : Nat)
    (hasResidual : Bool)

def source := TypedSource.specialize aStorage doutStorage swapped dtype out_dtype a dout db res
  seqlen chunk K groups sab sas sah sak sdb sdc sdh sdm sdn sob sos soh sok srb srs srh srk BM BN BCS hasResidual

theorem source_projection :
    (source aStorage doutStorage swapped dtype out_dtype a dout db res
      seqlen chunk K groups sab sas sah sak sdb sdc sdh sdm sdn sob sos soh sok srb srs srh srk BM BN BCS hasResidual).toAlgorithm? =
    .ok (source aStorage doutStorage swapped dtype out_dtype a dout db res
      seqlen chunk K groups sab sas sah sak sdb sdc sdh sdm sdn sob sos soh sok srb srs srh srk BM BN BCS hasResidual).toAlgKernel := by
  cases aStorage <;> cases doutStorage <;> cases swapped <;> rfl

/-- All represented storage combinations have the same real program after
precision erasure. No equivalence between different FP dtypes is asserted. -/
theorem storage_erasure :
    (source aStorage doutStorage swapped dtype out_dtype a dout db res
      seqlen chunk K groups sab sas sah sak sdb sdc sdh sdm sdn sob sos soh sok srb srs srh srk BM BN BCS hasResidual).eraseDType =
    (source .fp32 .fp32 swapped .real .real a dout db res
      seqlen chunk K groups sab sas sah sak sdb sdc sdh sdm sdn sob sos soh sok srb srs srh srk BM BN BCS hasResidual).eraseDType := by
  cases aStorage <;> cases doutStorage <;> cases swapped
  all_goals
    simp only [source, TypedSource.specialize, ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
      ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
      ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
      Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
      NumericDType.eraseDType, ComparableDType.eraseDType,
      IntegralDType.eraseDType, Except.bind, bind, eraseDType_float]
    simp only [VeriTile.Triton.eraseDType]
    cases dtype <;> cases out_dtype <;> rfl

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.BmmChunkBwd.RealSource
