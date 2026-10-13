import bench.tritonbench_g.rmsnorm_fused_llama.Kernels
import bench.tritonbench_g.l2_norm_triton1.Proofs.FP

namespace VeriTile.Bench.TritonBenchG.RmsnormFusedLlama.FPEquiv
open VeriTile Triton FP.ProfiledRewrite
open scoped VeriTile.Spec
set_option maxHeartbeats 16000000
set_option maxRecDepth 16000

def assumptions : Spec.Assumptions FP.GuardedFragment := []

def program (input weight output : Storage) (v : Bool)
    (X Y W : RegionName) (stride N B : Nat) (eps : ℝ) : Program where
  kernel := typedKernel input weight output v X Y W stride N B eps
  domain := []

/-- The whole two-pass program preserves all backend results and failures, all
registers and all memory, including aliases and arbitrary masked-load undefined
values. Only exact natural-address multiplication changes; no numerical atom,
finite-input assumption or successful-output assumption is required. -/
specification source_equivalence (input weight output : Storage)
    (X Y W : RegionName) (stride N B : Nat) (eps : ℝ) :
    program input weight output Bool.false X Y W stride N B eps ≡[assumptions]
      program input weight output Bool.true X Y W stride N B eps := by
  cases input <;> cases weight <;> cases output <;> equiv_decompose <;> fp_prove

#print_fp_assumptions source_equivalence
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsnormFusedLlama.FPEquiv
