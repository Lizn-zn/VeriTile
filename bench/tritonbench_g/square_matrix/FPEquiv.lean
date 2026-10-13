import bench.tritonbench_g.square_matrix.Kernels
import bench.tritonbench_g.destindex_copy.Proofs.IndexRewrite

namespace VeriTile.Bench.TritonBenchG.SquareMatrix.FPEquiv
open VeriTile Triton FP.ProfiledRewrite
open scoped VeriTile.Spec

def assumptions : Spec.Assumptions FP.GuardedFragment := []

def program (storage : Storage) (commuted : Bool) (out inp : RegionName)
    (inputStride outputStride n B : Nat) : Program where
  kernel := typedKernel storage commuted out inp inputStride outputStride n B
  domain := []

/-- Exact index arithmetic preserves complete typed execution, including every
register and memory cell and failure. No floating-point atom is needed. -/
specification square_index_equiv (storage : Storage) (out inp : RegionName)
    (inputStride outputStride n B : Nat) (_hB : 0 < B) :
    program storage Bool.false out inp inputStride outputStride n B ≡[assumptions]
      program storage Bool.true out inp inputStride outputStride n B := by
  cases storage <;> equiv_decompose <;> fp_prove

#print_fp_assumptions square_index_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SquareMatrix.FPEquiv
