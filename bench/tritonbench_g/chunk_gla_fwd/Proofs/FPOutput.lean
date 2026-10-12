import bench.tritonbench_g.chunk_gla_fwd.Kernels
import VeriTile.Triton.Float.RewriteTactics
import VeriTile.Meta.StatementAudit

/-! Complete output-source equivalence. Only the final fp32 carry/dot addition
is commuted. Its guard executes the complete original prefix, including the K
loop, query casts, intrinsic exp, both dots and A/V loads. No real projection,
exp identity, reassociation or final-output equation is used. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.FPOutput
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec
set_option maxHeartbeats 8000000
set_option maxRecDepth 18000

def intraDot (storage : TypedOutput.Storage) (BT BV : Nat) : Op .real [BT,BV] :=
  match storage with
  | .fp32 => .dot (batch := []) (.ref .real [BT,BT] "b_A") (.ref .real [BT,BV] "b_v")
      {allowTF32 := some Bool.false}
  | .fp16 => .castFloat .fp32 .real (.dotTyped .fp16 .fp16 .fp32 (batch := [])
      (.ref .fp16 [BT,BT] "b_A") (.ref .fp16 [BT,BV] "b_v") {allowTF32 := some Bool.false})
  | .bf16 => .castFloat .fp32 .real (.dotTyped .bf16 .bf16 .fp32 (batch := [])
      (.ref .bf16 [BT,BT] "b_A") (.ref .bf16 [BT,BV] "b_v") {allowTF32 := some Bool.false})

def requirements (kernel : ComputeKernel) (storage : TypedOutput.Storage) (BT BV : Nat) : List Site :=
  [.addFP32 (Source.prefixBefore kernel.surfaceBody "b_o" 1) [BT,BV]
    (.ref .real [BT,BV] "b_o") (intraDot storage BT BV)]

section
variable (queryStorage valueStorage gateStorage : TypedOutput.Storage)
    (dtype out_dtype : FloatDType) (q v g h o A : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat) (scale : ℝ) (T K V BT BK BV : Nat)

def original : Program where
  kernel := TypedOutput.original queryStorage valueStorage gateStorage dtype out_dtype q v g h o A
    s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV
  domain := requirements (TypedOutput.original queryStorage valueStorage gateStorage dtype out_dtype q v g h o A
    s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV) valueStorage BT BV

def sumLeft : Program :=
  { original queryStorage valueStorage gateStorage dtype out_dtype q v g h o A
      s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV with
    kernel := TypedOutput.sumLeft queryStorage valueStorage gateStorage dtype out_dtype q v g h o A
      s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV }

theorem output_equiv (R : FP.ScalarArithmetic.Rules) :
    original queryStorage valueStorage gateStorage dtype out_dtype q v g h o A
      s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV ≡[R]
    sumLeft queryStorage valueStorage gateStorage dtype out_dtype q v g h o A
      s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV := by
  cases queryStorage <;> cases valueStorage <;> cases gateStorage <;> equiv_decompose <;> fp_prove
end

#print_fp_assumptions output_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.FPOutput
