import bench.tritonbench_g.chunk_gla_simple.Kernels
import VeriTile.Triton.Float.RewriteTactics

namespace VeriTile.Bench.TritonBenchG.ChunkGlaSimple.FPProof
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec
set_option maxRecDepth 18000
set_option maxHeartbeats 8000000

def dotResult (storage : Storage) (BT BV : Nat) : Op .real [BT,BV] :=
  match storage with
  | .fp32 => .dot (batch := []) (.castFloat .real .real (.ref .real [BT,BT] "b_s"))
      (.ref .real [BT,BV] "b_v") {allowTF32 := some Bool.false}
  | .fp16 => .castFloat .fp32 .real (.dotTyped .fp16 .fp16 .fp32 (batch := [])
      (.castFloat .real .fp16 (.ref .real [BT,BT] "b_s")) (.ref .fp16 [BT,BV] "b_v")
      {allowTF32 := some Bool.false})
  | .bf16 => .castFloat .fp32 .real (.dotTyped .bf16 .bf16 .fp32 (batch := [])
      (.castFloat .real .bf16 (.ref .real [BT,BT] "b_s")) (.ref .bf16 [BT,BV] "b_v")
      {allowTF32 := some Bool.false})
def total (storage : Storage) (BT BV : Nat) : Op .real [BT,BV] :=
  .add .real (.consSame (.consSame .nil)) (.ref .real [BT,BV] "b_o") (dotResult storage BT BV)
def requirements (kernel : ComputeKernel) (storage : Storage) (BT BV : Nat) (scale : ℝ) : List Site :=
  [.mulScalar (Source.prefixBefore kernel.surfaceBody "b_o" 2) BT [BV] (total storage BT BV) (.const scale)]
section
variable (dotStorage valueStorage gateStorage : Storage) (out_dtype : FloatDType) (q k v h g o : RegionName)
  (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat) (scale : ℝ) (T K V BT BK BV : Nat)
def original : Program where
  kernel := ChunkGlaSimple.original dotStorage valueStorage gateStorage out_dtype q k v h g o
    s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV
  domain := requirements (ChunkGlaSimple.original dotStorage valueStorage gateStorage out_dtype q k v h g o
    s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV) valueStorage BT BV scale
def scaleFirst : Program :=
  { original dotStorage valueStorage gateStorage out_dtype q k v h g o
      s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV with
    kernel := ChunkGlaSimple.scaleFirst dotStorage valueStorage gateStorage out_dtype q k v h g o
      s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV }
theorem equivalent (R : FP.ScalarArithmetic.Rules) :
    original dotStorage valueStorage gateStorage out_dtype q k v h g o
      s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV ≡[R]
    scaleFirst dotStorage valueStorage gateStorage out_dtype q k v h g o
      s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV := by
  cases dotStorage <;> cases valueStorage <;> cases gateStorage <;> equiv_decompose <;> fp_prove
end
#print_fp_assumptions equivalent
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ChunkGlaSimple.FPProof
