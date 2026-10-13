import bench.tritonbench_g.rmsnorm_implementation.Kernels
import bench.tritonbench_g.cache_transform_triton.Proofs.IndexRewrite

namespace VeriTile.Bench.TritonBenchG.RmsnormImplementation.FPEquiv
open VeriTile Triton FP.ProfiledRewrite
open scoped VeriTile.Spec
set_option maxHeartbeats 16000000
set_option maxRecDepth 16000

def assumptions : Spec.Assumptions FP.GuardedFragment := []

def program (input weight output : Storage) (v : Bool)
    (X W O : RegionName) (sxb sxm sxk sw sob som sok N B : Nat) (eps : ℝ) : Program where
  kernel := typedKernel input weight output v X W O sxb sxm sxk sw sob som sok N B eps
  domain := []

/-- The whole two-pass program preserves all backend results and failures, all
registers and all memory, including aliases and arbitrary masked-load undefined
values. Only exact natural-address multiplication changes; no numerical atom,
finite-input assumption or successful-output assumption is required. -/
specification source_equivalence (input weight output : Storage)
    (X W O : RegionName) (sxb sxm sxk sw sob som sok N B : Nat) (eps : ℝ) :
    program input weight output Bool.false X W O sxb sxm sxk sw sob som sok N B eps ≡[assumptions]
      program input weight output Bool.true X W O sxb sxm sxk sw sob som sok N B eps := by
  cases input <;> cases weight <;> cases output <;> equiv_decompose <;> fp_prove

#print_fp_assumptions source_equivalence
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsnormImplementation.FPEquiv
