import bench.tritonbench_g.block_sparse_attn.Kernels
import VeriTile.Triton.Float.RewriteTactics
import VeriTile.Meta.StatementAudit

/-! Score-scaling commutation in the complete signed CSR loop. Each reached
iteration checks its actual fp32 qk tile and scalar after the score dots.
Nothing about tl.exp, the reductions or online normalization is assumed. -/
namespace VeriTile.Bench.TritonBenchG.BlockSparseAttn.FPEquivProofs
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec

/-- The third top-level qk assignment is scaling. The optional second dot
remains in its preceding branch. Signed-loop reachability keeps the metadata,
prefix and earlier iterations in the guard's execution context. -/
def requirementsFor (kernel : ComputeKernel) (BM BN : Nat) (scale : ℝ) : List Site :=
  match Source.intRange? kernel.surfaceBody "col_idx_idx" with
  | none => []
  | some loop => [.intRange loop.before "col_idx_idx" loop.start loop.stop loop.stride loop.body
      (.mulScalarFP32 (Source.prefixBefore loop.body "qk" 2) BM [BN]
        (.ref .real [BM, BN] "qk") (.const scale))]

section
variable (storage : TypedSource.Storage) (out Q K V : RegionName)
    (rows cols : Region .int) (rs cs layouts : Nat) (scale : ℝ)
    (sqb sqh sqm skb skh skn svb svh svn sob soh som heads kvHeads N BM BN BD blocks : Nat)
    (evenM evenN : Bool)

def original : Program where
  kernel := TypedSource.original storage out Q K V rows cols rs cs layouts scale
    sqb sqh sqm skb skh skn svb svh svn sob soh som heads kvHeads N BM BN BD blocks evenM evenN
  domain := requirementsFor (TypedSource.original storage out Q K V rows cols rs cs layouts scale
    sqb sqh sqm skb skh skn svb svh svn sob soh som heads kvHeads N BM BN BD blocks evenM evenN)
      BM BN scale

def scalarLeft : Program :=
  { original storage out Q K V rows cols rs cs layouts scale
      sqb sqh sqm skb skh skn svb svh svn sob soh som heads kvHeads N BM BN BD blocks evenM evenN with
    kernel := TypedSource.scalarLeft storage out Q K V rows cols rs cs layouts scale
      sqb sqh sqm skb skh skn svb svh svn sob soh som heads kvHeads N BM BN BD blocks evenM evenN }

theorem scaling_equiv (R : FP.ScalarArithmetic.Rules) :
    original storage out Q K V rows cols rs cs layouts scale
      sqb sqh sqm skb skh skn svb svh svn sob soh som heads kvHeads N BM BN BD blocks evenM evenN ≡[R]
    scalarLeft storage out Q K V rows cols rs cs layouts scale
      sqb sqh sqm skb skh skn svb svh svn sob soh som heads kvHeads N BM BN BD blocks evenM evenN := by
  cases storage <;> equiv_decompose <;> fp_prove

end
#print_fp_assumptions scaling_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.BlockSparseAttn.FPEquivProofs
