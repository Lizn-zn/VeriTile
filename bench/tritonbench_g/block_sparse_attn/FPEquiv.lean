import bench.tritonbench_g.block_sparse_attn.Kernels
import bench.tritonbench_g.block_sparse_attn.Proofs.FP
import VeriTile.Meta.StatementAudit

/-! Whole-program FP equivalence of the original and scalar-left sources.
Only qk*scale becomes scale*qk, after both score dots and before masking.
At every reached signed CSR visit, the domain checks the actual fp32 score
tile and scale for finiteness. Prefix and earlier visits stay in that context.

fp16/bf16/fp32 storage, probability/output casts, tl.exp, online normalization,
all reductions, masks and stores are preserved. No exponential identity or
reduction reassociation is needed. Dimensions and strides stay symbolic.
This is equivalence under the admitted FP atom, not concrete IEEE equality. -/
namespace VeriTile.Bench.TritonBenchG.BlockSparseAttn.Equivalence
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec

section
variable (storage : TypedSource.Storage) (out Q K V : RegionName)
    (rows cols : Region .int) (rs cs layouts : Nat) (scale : ℝ)
    (sqb sqh sqm skb skh skn svb svh svn sob soh som heads kvHeads N BM BN BD blocks : Nat)
    (evenM evenN : Bool)

def original : Program where
  kernel := TypedSource.original storage out Q K V rows cols rs cs layouts scale
    sqb sqh sqm skb skh skn svb svh svn sob soh som heads kvHeads N BM BN BD blocks evenM evenN
  domain := FPEquivProofs.requirementsFor (TypedSource.original storage out Q K V rows cols rs cs layouts scale
    sqb sqh sqm skb skh skn svb svh svn sob soh som heads kvHeads N BM BN BD blocks evenM evenN) BM BN scale

def scalarLeft : Program :=
  { original storage out Q K V rows cols rs cs layouts scale
      sqb sqh sqm skb skh skn svb svh svn sob soh som heads kvHeads N BM BN BD blocks evenM evenN with
    kernel := TypedSource.scalarLeft storage out Q K V rows cols rs cs layouts scale
      sqb sqh sqm skb skh skn svb svh svn sob soh som heads kvHeads N BM BN BD blocks evenM evenN }

specification block_sparse_scaling_equiv (R : FP.ScalarArithmetic.Rules) :
    original storage out Q K V rows cols rs cs layouts scale
      sqb sqh sqm skb skh skn svb svh svn sob soh som heads kvHeads N BM BN BD blocks evenM evenN ≡[R]
    scalarLeft storage out Q K V rows cols rs cs layouts scale
      sqb sqh sqm skb skh skn svb svh svn sob soh som heads kvHeads N BM BN BD blocks evenM evenN := by
  exact FPEquivProofs.scaling_equiv storage out Q K V rows cols rs cs layouts scale
    sqb sqh sqm skb skh skn svb svh svn sob soh som heads kvHeads N BM BN BD blocks evenM evenN R

end
#print_fp_assumptions block_sparse_scaling_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.BlockSparseAttn.Equivalence
