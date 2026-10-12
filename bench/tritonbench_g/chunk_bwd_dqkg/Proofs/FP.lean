import bench.tritonbench_g.chunk_bwd_dqkg.Kernels
import VeriTile.Triton.Float.RewriteTactics
import VeriTile.Meta.StatementAudit

/-! The full-source candidate commutes only the final fp32 dq addition.
Its domain checks the reached carry and the actual typed dot result after
the complete value loop, gating and score conversion. Reductions and dots
remain opaque; no reassociation or dot distributivity is assumed.
Successful execution and real output correctness are separate obligations. -/
namespace VeriTile.Bench.TritonBenchG.ChunkBwdDqkg.FPEquivProofs
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000

def finalDot (qk : TypedSource.Storage) (BT BK : Nat) : Op .real [BT, BK] :=
  .castFloat .fp32 .real (.dotTyped qk.dtype qk.dtype .fp32 (batch := [])
    (.ref qk.dtype.toTileDType [BT, BT] "b_ds")
    (.ref qk.dtype.toTileDType [BT, BK] "b_k"))

def requirementsFor (kernel : ComputeKernel) (qk : TypedSource.Storage)
    (BT BK : Nat) : List Site :=
  [.addFP32 (Source.prefixBefore kernel.surfaceBody "b_dq" 2) [BT, BK]
    (.ref .real [BT, BK] "b_dq") (finalDot qk BT BK)]

section
variable (qk value hist dhist : TypedSource.Storage)
    (q k v h g do_ dh dq dk dg : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
    (scale : ℝ) (T K V BT BK BV NT : Nat)

def original : Program where
  kernel := TypedSource.original qk value hist dhist q k v h g do_ dh dq dk dg
    s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV NT
  domain := requirementsFor
    (TypedSource.original qk value hist dhist q k v h g do_ dh dq dk dg
      s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV NT) qk BT BK

def dotLeft : Program :=
  { original qk value hist dhist q k v h g do_ dh dq dk dg
      s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV NT with
    kernel := TypedSource.dotLeft qk value hist dhist q k v h g do_ dh dq dk dg
      s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV NT }

theorem dot_left_equiv (R : FP.ScalarArithmetic.Rules) :
    original qk value hist dhist q k v h g do_ dh dq dk dg
      s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV NT ≡[R]
    dotLeft qk value hist dhist q k v h g do_ dh dq dk dg
      s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV NT := by
  cases hist <;> cases dhist <;> equiv_decompose <;> fp_prove

end
#print_fp_assumptions dot_left_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ChunkBwdDqkg.FPEquivProofs
