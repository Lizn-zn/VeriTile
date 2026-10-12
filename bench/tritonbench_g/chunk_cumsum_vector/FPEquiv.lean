import bench.tritonbench_g.chunk_cumsum_vector.Kernels
import bench.tritonbench_g.chunk_cumsum_vector.Proofs.FP
import VeriTile.Meta.StatementAudit

/-! Complete typed equivalence: exchange only the carry and block total in
each fp32 addition. The triangular dot, sum, memory accesses, casts and stores
keep their original order and backend interpretation. At each reached iteration, finite
checks apply to the actual carry and computed total. They do not require a
particular output equation, dot identity or reduction association. -/
namespace VeriTile.Bench.TritonBenchG.ChunkCumsumVector.Equivalence
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec

section
variable (dtype out_dtype : FloatDType) (s o : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat)

def original : Program where
  kernel := TypedSource.original dtype out_dtype s o s_s_h s_s_t s_s_d T S BT BS
  domain := FPEquivProofs.requirementsFor
    (TypedSource.original dtype out_dtype s o s_s_h s_s_t s_s_d T S BT BS) BT BS

def sumLeft : Program :=
  { original dtype out_dtype s o s_s_h s_s_t s_s_d T S BT BS with
    kernel := TypedSource.sumLeft dtype out_dtype s o s_s_h s_s_t s_s_d T S BT BS }

specification chunk_sum_left_equiv (R : FP.ScalarArithmetic.Rules) :
    original dtype out_dtype s o s_s_h s_s_t s_s_d T S BT BS ≡[R]
      sumLeft dtype out_dtype s o s_s_h s_s_t s_s_d T S BT BS := by
  exact FPEquivProofs.sum_left_equiv dtype out_dtype s o s_s_h s_s_t s_s_d T S BT BS R
end
#print_fp_assumptions chunk_sum_left_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ChunkCumsumVector.Equivalence
