import bench.tritonbench_g.chunk_gla_simple.Proofs.FP

/-! Complete typed-source equivalence for the scale-first candidate in Kernels.
The sole numeric rewrite commutes the final fp32 multiplication by scale.
Q/K/H share the chosen dot storage; V and G storage are independent. The source
score-to-V cast, output cast, intrinsic tl.exp, TF32-disabled dots, checked
loads, causal mask and full K loop are retained. No exponential identity is used.

The domain checks the actual total (gated history plus the final dot) and scale
after the common source prefix. It does not constrain the experimental shape.
A nonempty mixed-storage execution is checked in Proofs/TypedExecution. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaSimple.FPEquiv
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec
section
variable (dotStorage valueStorage gateStorage : Storage) (out : FloatDType) (q k v h g o : RegionName)
  (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat) (scale : ℝ) (T K V BT BK BV : Nat)

def original : Program where
  kernel := ChunkGlaSimple.original dotStorage valueStorage gateStorage out q k v h g o
    s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV
  domain := FPProof.requirements
    (ChunkGlaSimple.original dotStorage valueStorage gateStorage out q k v h g o
      s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV) valueStorage BT BV scale

def scaleFirst : Program :=
  { original dotStorage valueStorage gateStorage out q k v h g o
      s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV with
    kernel := ChunkGlaSimple.scaleFirst dotStorage valueStorage gateStorage out q k v h g o
      s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV }

specification scale_equiv (R : FP.ScalarArithmetic.Rules) :
    original dotStorage valueStorage gateStorage out q k v h g o
      s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV ≡[R]
    scaleFirst dotStorage valueStorage gateStorage out q k v h g o
      s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV := by
  exact FPProof.equivalent dotStorage valueStorage gateStorage out q k v h g o
    s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV R
end
#print_fp_assumptions scale_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ChunkGlaSimple.FPEquiv
