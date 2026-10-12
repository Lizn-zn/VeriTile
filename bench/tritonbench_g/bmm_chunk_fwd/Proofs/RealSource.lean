import bench.tritonbench_g.bmm_chunk_fwd.Kernels
import VeriTile.Triton.Float.Erasure

/-! Erase float precision in the complete two source variants. Signed metadata,
causal block selection, tail masks and all casts' value dependencies remain.
This is the source connection, not the independent matrix-product theorem. -/
namespace VeriTile.Bench.TritonBenchG.BmmChunkFwd.RealSource
open VeriTile.Triton
set_option maxHeartbeats 9000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false

section
variable (aStorage bStorage : TypedSource.Storage) (swapped : Bool) (out_dtype : FloatDType)
    (a b out : RegionName) (seq : Region .int)
    (seqlen chunk K groups sab sas sah sak sbb sbs sbh sbk sob soc soh som son ssb sss BM BN BK : Nat)
    (causal hasSeq : Bool)

def source := TypedSource.specialize aStorage bStorage swapped out_dtype a b out seq
  seqlen chunk K groups sab sas sah sak sbb sbs sbh sbk sob soc soh som son ssb sss BM BN BK causal hasSeq

theorem source_projection :
    (source aStorage bStorage swapped out_dtype a b out seq
      seqlen chunk K groups sab sas sah sak sbb sbs sbh sbk sob soc soh som son ssb sss BM BN BK causal hasSeq).toAlgorithm? =
    .ok (source aStorage bStorage swapped out_dtype a b out seq
      seqlen chunk K groups sab sas sah sak sbb sbs sbh sbk sob soc soh som son ssb sss BM BN BK causal hasSeq).toAlgKernel := by
  cases aStorage <;> cases bStorage <;> cases swapped <;> rfl

theorem storage_erasure :
    (source aStorage bStorage swapped out_dtype a b out seq
      seqlen chunk K groups sab sas sah sak sbb sbs sbh sbk sob soc soh som son ssb sss BM BN BK causal hasSeq).eraseDType =
    (source .fp32 .fp32 swapped .real a b out seq
      seqlen chunk K groups sab sas sah sak sbb sbs sbh sbk sob soc soh som son ssb sss BM BN BK causal hasSeq).eraseDType := by
  cases aStorage <;> cases bStorage <;> cases swapped
  all_goals
    simp only [source, TypedSource.specialize, ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
      ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
      ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
      Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
      NumericDType.eraseDType, ComparableDType.eraseDType,
      IntegralDType.eraseDType, Except.bind, bind, eraseDType_float]
    simp only [VeriTile.Triton.eraseDType]
    cases out_dtype <;> rfl

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.BmmChunkFwd.RealSource
