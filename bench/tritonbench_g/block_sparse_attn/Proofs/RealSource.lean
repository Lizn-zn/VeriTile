import bench.tritonbench_g.block_sparse_attn.Kernels
import VeriTile.Triton.Float.Erasure

/-! Connection from the complete typed sources to their real programs.
Signed CSR metadata, all mask/channel branches and the program structure
survive erasure. The independent attention formula and memory safety are
separate obligations; FP equivalence imports no result from this module. -/
namespace VeriTile.Bench.TritonBenchG.BlockSparseAttn.RealSource
open VeriTile.Triton
set_option maxHeartbeats 3000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false

section
variable (storage : TypedSource.Storage) (swapped : Bool) (out Q K V : RegionName)
    (rows cols : Region .int) (rs cs layouts : Nat) (scale : ℝ)
    (sqb sqh sqm skb skh skn svb svh svn sob soh som heads kvHeads N BM BN BD blocks : Nat)
    (evenM evenN : Bool)

def source := TypedSource.specialize storage swapped out Q K V rows cols rs cs layouts scale
  sqb sqh sqm skb skh skn svb svh svn sob soh som heads kvHeads N BM BN BD blocks evenM evenN

theorem source_projection :
    (source storage swapped out Q K V rows cols rs cs layouts scale
      sqb sqh sqm skb skh skn svb svh svn sob soh som heads kvHeads N BM BN BD blocks evenM evenN).toAlgorithm? =
    .ok (source storage swapped out Q K V rows cols rs cs layouts scale
      sqb sqh sqm skb skh skn svb svh svn sob soh som heads kvHeads N BM BN BD blocks evenM evenN).toAlgKernel := by
  cases storage <;> cases swapped <;> rfl

/-- The same full real source for all represented storage dtypes. No FP
equivalence between different storage configurations is asserted here. -/
theorem storage_erasure :
    (source storage swapped out Q K V rows cols rs cs layouts scale
      sqb sqh sqm skb skh skn svb svh svn sob soh som heads kvHeads N BM BN BD blocks evenM evenN).eraseDType =
    (source .fp16 swapped out Q K V rows cols rs cs layouts scale
      sqb sqh sqm skb skh skn svb svh svn sob soh som heads kvHeads N BM BN BD blocks evenM evenN).eraseDType := by
  cases storage <;> cases swapped
  all_goals
    simp only [source, TypedSource.specialize, ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
      ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
      ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
      Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
      IntegralDType.eraseDType, Except.bind, bind, eraseDType_float]
    try rfl

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.BlockSparseAttn.RealSource
