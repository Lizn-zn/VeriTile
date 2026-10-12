import bench.tritonbench_g.decay_cumsum.Kernels
import bench.tritonbench_g.cache_transform_triton.Proofs.IndexRewrite
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.DecayCumsum.MaskEquivalence
open VeriTile Triton FP.Structural FP.ProfiledRewrite TypedSource
open scoped VeriTile.Spec
set_option maxHeartbeats 8000000
set_option maxRecDepth 20000

/-- Congruence for the exact natural-number lane comparison. -/
@[spec_rule] theorem op_nat_lt {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) (before : List ComputeStmt) (bc : Broadcast sa sb sh)
    (a a' : Op .nat sa) (b b' : Op .nat sb)
    (ha : OpEquivalent M s before a a') (hb : OpEquivalent M s before b b') :
    OpEquivalent M s before (.lt .nat bc a b) (.lt .nat bc a' b') := by
  intro t ht
  simp only [FP.Structural.evalOp]
  rw [ha t ht, hb t ht]

def assumptions : Spec.Assumptions FP.GuardedFragment := []

section
variable (storage : Storage) (g g_o : RegionName) (s_qk_h s_qk_t s_qk_d B H T BT BK DK : Nat) (scale : ℝ)
def forwardProgram (swapped : Bool) : Program where
  kernel := forward storage swapped g g_o s_qk_h s_qk_t s_qk_d B H T BT BK DK scale
  domain := []

theorem forward_equiv :
    forwardProgram storage g g_o s_qk_h s_qk_t s_qk_d B H T BT BK DK scale Bool.false ≡[assumptions]
    forwardProgram storage g g_o s_qk_h s_qk_t s_qk_d B H T BT BK DK scale Bool.true := by
  cases storage <;> equiv_decompose <;> fp_prove
end
#print_fp_assumptions forward_equiv

section
variable (source gate : Storage) (q k g qg kg : RegionName) (s_qk_h s_qk_t s_qk_d B H T BT BK DK : Nat) (scale : ℝ)
def prepareProgram (swapped : Bool) : Program where
  kernel := prepare source gate swapped q k g qg kg s_qk_h s_qk_t s_qk_d B H T BT BK DK scale
  domain := []

theorem prepare_equiv :
    prepareProgram source gate q k g qg kg s_qk_h s_qk_t s_qk_d B H T BT BK DK scale Bool.false ≡[assumptions]
    prepareProgram source gate q k g qg kg s_qk_h s_qk_t s_qk_d B H T BT BK DK scale Bool.true := by
  cases source <;> cases gate <;> equiv_decompose <;> fp_prove
end
#print_fp_assumptions prepare_equiv

section
variable (source gate : Storage) (dq_inner dq_inter dk_inner dk_inter q k g dg : RegionName) (s_qk_h s_qk_t s_qk_d B H T BT BK DK : Nat) (scale : ℝ)
def backwardProgram (swapped : Bool) : Program where
  kernel := backward source gate swapped dq_inner dq_inter dk_inner dk_inter q k g dg s_qk_h s_qk_t s_qk_d B H T BT BK DK scale
  domain := []

theorem backward_equiv :
    backwardProgram source gate dq_inner dq_inter dk_inner dk_inter q k g dg s_qk_h s_qk_t s_qk_d B H T BT BK DK scale Bool.false ≡[assumptions]
    backwardProgram source gate dq_inner dq_inter dk_inner dk_inter q k g dg s_qk_h s_qk_t s_qk_d B H T BT BK DK scale Bool.true := by
  cases source <;> cases gate <;> equiv_decompose <;> fp_prove
end
#print_fp_assumptions backward_equiv

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.DecayCumsum.MaskEquivalence
