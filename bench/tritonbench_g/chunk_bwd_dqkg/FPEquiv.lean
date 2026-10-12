import bench.tritonbench_g.chunk_bwd_dqkg.Kernels
import bench.tritonbench_g.chunk_bwd_dqkg.Proofs.FP
import VeriTile.Meta.StatementAudit

/-! FP equivalence of the two complete typed programs in Kernels. The sole
rewrite exchanges the final dq carry and dot result. All value-loop iterations,
loads, casts, opaque reductions/dots, masks and stores remain in context.

The domain checks finite fp32 operands at the reached final addition: the
accumulator and the dot result after the score's cast to q/k storage. This is
conditional equivalence under the admitted atomic rule, not IEEE bit equality
or an end-to-end error bound. -/
namespace VeriTile.Bench.TritonBenchG.ChunkBwdDqkg.Equivalence
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec

section
variable (qk valueStorage hist dhist : TypedSource.Storage)
    (q k v h g dout dh dq dk dg : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
    (scale : ℝ) (T K V BT BK BV NT : Nat)

def original : Program where
  kernel := TypedSource.original qk valueStorage hist dhist q k v h g dout dh dq dk dg
    s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV NT
  domain := FPEquivProofs.requirementsFor
    (TypedSource.original qk valueStorage hist dhist q k v h g dout dh dq dk dg
      s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV NT) qk BT BK

def dotLeft : Program :=
  { original qk valueStorage hist dhist q k v h g dout dh dq dk dg
      s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV NT with
    kernel := TypedSource.dotLeft qk valueStorage hist dhist q k v h g dout dh dq dk dg
      s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV NT }

specification chunk_dot_left_equiv (R : FP.ScalarArithmetic.Rules) :
    original qk valueStorage hist dhist q k v h g dout dh dq dk dg
      s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV NT ≡[R]
    dotLeft qk valueStorage hist dhist q k v h g dout dh dq dk dg
      s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV NT := by
  exact FPEquivProofs.dot_left_equiv qk valueStorage hist dhist q k v h g dout dh dq dk dg
    s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV NT R
end

#print_fp_assumptions chunk_dot_left_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ChunkBwdDqkg.Equivalence
