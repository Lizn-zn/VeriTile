import bench.tritonbench_g.matrix_transpose.Kernels
import bench.tritonbench_g.matrix_transpose.Proofs.IndexRewrite

namespace VeriTile.Bench.TritonBenchG.MatrixTranspose.FPEquiv
open VeriTile Triton FP.ProfiledRewrite
open scoped VeriTile.Spec

def assumptions : Spec.Assumptions FP.GuardedFragment := []

def program (v : Bool) (M Out : RegionName) (sx sy ox oy m d : Nat) : Program where
  kernel := typedKernel v M Out sx sy ox oy m d
  domain := []

/-- Full float16 execution agrees, including all registers, memory, and failure.
Only natural-number address multiplication changes, so no floating atom is used. -/
specification transpose_index_equiv (M Out : RegionName) (sx sy ox oy m d : Nat) :
    program Bool.false M Out sx sy ox oy m d ≡[assumptions]
      program Bool.true M Out sx sy ox oy m d := by
  equiv_decompose <;> fp_prove

#print_fp_assumptions transpose_index_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.MatrixTranspose.FPEquiv
