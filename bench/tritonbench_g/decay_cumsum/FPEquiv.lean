import bench.tritonbench_g.decay_cumsum.Kernels
import bench.tritonbench_g.decay_cumsum.Proofs.MaskEquivalence
/-! Whole-program FP equivalence for the forward scan, preparation and reverse
scan. Only i_k * BK in the natural-number lane mask becomes BK * i_k.
All numerical operations, casts, reads, writes and failures are preserved.
This structural change needs no numerical atom or numerical-domain restriction.
Source and gate storage vary independently where the source permits them. -/
namespace VeriTile.Bench.TritonBenchG.DecayCumsum.FPEquivalence
open VeriTile Triton FP.ProfiledRewrite TypedSource
open scoped VeriTile.Spec

def assumptions : Spec.Assumptions FP.GuardedFragment := []

section
variable (storage : Storage) (g g_o : RegionName) (s_qk_h s_qk_t s_qk_d B H T BT BK DK : Nat) (scale : ℝ)
def forwardProgram (swapped : Bool) : Program where
  kernel := forward storage swapped g g_o s_qk_h s_qk_t s_qk_d B H T BT BK DK scale
  domain := []

specification forward_index_commuted_equiv :
    forwardProgram storage g g_o s_qk_h s_qk_t s_qk_d B H T BT BK DK scale Bool.false ≡[assumptions]
    forwardProgram storage g g_o s_qk_h s_qk_t s_qk_d B H T BT BK DK scale Bool.true := by
  exact MaskEquivalence.forward_equiv storage g g_o s_qk_h s_qk_t s_qk_d B H T BT BK DK scale
end
#print_fp_assumptions forward_index_commuted_equiv

section
variable (source gate : Storage) (q k g qg kg : RegionName) (s_qk_h s_qk_t s_qk_d B H T BT BK DK : Nat) (scale : ℝ)
def prepareProgram (swapped : Bool) : Program where
  kernel := prepare source gate swapped q k g qg kg s_qk_h s_qk_t s_qk_d B H T BT BK DK scale
  domain := []

specification prepare_index_commuted_equiv :
    prepareProgram source gate q k g qg kg s_qk_h s_qk_t s_qk_d B H T BT BK DK scale Bool.false ≡[assumptions]
    prepareProgram source gate q k g qg kg s_qk_h s_qk_t s_qk_d B H T BT BK DK scale Bool.true := by
  exact MaskEquivalence.prepare_equiv source gate q k g qg kg s_qk_h s_qk_t s_qk_d B H T BT BK DK scale
end
#print_fp_assumptions prepare_index_commuted_equiv

section
variable (source gate : Storage) (dq_inner dq_inter dk_inner dk_inter q k g dg : RegionName) (s_qk_h s_qk_t s_qk_d B H T BT BK DK : Nat) (scale : ℝ)
def backwardProgram (swapped : Bool) : Program where
  kernel := backward source gate swapped dq_inner dq_inter dk_inner dk_inter q k g dg s_qk_h s_qk_t s_qk_d B H T BT BK DK scale
  domain := []

specification backward_index_commuted_equiv :
    backwardProgram source gate dq_inner dq_inter dk_inner dk_inter q k g dg s_qk_h s_qk_t s_qk_d B H T BT BK DK scale Bool.false ≡[assumptions]
    backwardProgram source gate dq_inner dq_inter dk_inner dk_inter q k g dg s_qk_h s_qk_t s_qk_d B H T BT BK DK scale Bool.true := by
  exact MaskEquivalence.backward_equiv source gate dq_inner dq_inter dk_inner dk_inter q k g dg s_qk_h s_qk_t s_qk_d B H T BT BK DK scale
end
#print_fp_assumptions backward_index_commuted_equiv

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.DecayCumsum.FPEquivalence
