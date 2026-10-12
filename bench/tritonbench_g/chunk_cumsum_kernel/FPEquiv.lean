import bench.tritonbench_g.chunk_cumsum_kernel.Kernels
import bench.tritonbench_g.chunk_cumsum_kernel.Proofs.FP
import VeriTile.Meta.StatementAudit

/-! Complete typed equivalence: exchange only the carry and block total in
each fp32 addition. The scan, sum, memory accesses, casts and stores keep their
original order and backend interpretation. At each reached iteration, finite
checks apply to the actual carry and computed total. They do not require a
particular output equation, scan identity or reduction association. -/
namespace VeriTile.Bench.TritonBenchG.ChunkCumsumKernel.Equivalence
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec

section
variable (dtype out_dtype : FloatDType) (s o : RegionName) (T BT : Nat)

def original : Program where
  kernel := TypedSource.original dtype out_dtype s o T BT
  domain := FPEquivProofs.requirementsFor (TypedSource.original dtype out_dtype s o T BT)

def sumLeft : Program :=
  { original dtype out_dtype s o T BT with
    kernel := TypedSource.sumLeft dtype out_dtype s o T BT }

specification chunk_sum_left_equiv (R : FP.ScalarArithmetic.Rules) :
    original dtype out_dtype s o T BT ≡[R] sumLeft dtype out_dtype s o T BT := by
  exact FPEquivProofs.sum_left_equiv dtype out_dtype s o T BT R
end
#print_fp_assumptions chunk_sum_left_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ChunkCumsumKernel.Equivalence
