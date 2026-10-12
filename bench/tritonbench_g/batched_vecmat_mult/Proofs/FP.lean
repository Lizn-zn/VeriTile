import bench.tritonbench_g.batched_vecmat_mult.Kernels
import VeriTile.Triton.Float.RewriteTactics
import VeriTile.Meta.StatementAudit

/-! Whole-loop fp32 equivalence. Each reached iteration checks the actual
carry and transposed product sum before the addition. Reductions are kept in
place; no reduction identity, positivity or output equation is assumed. -/
namespace VeriTile.Bench.TritonBenchG.BatchedVecmatMult.FPEquivProofs
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec

def blockSum (BM BN BK : Nat) : Op .real [BM,BN] :=
  .transpose (batch := []) (.reduceSum ⟨2, by simp⟩ Bool.false
    (.mul .real (.consL (.consSame (.consSame .nil)))
      (.ref .real [1,BM,BK] "expanded_a") (.ref .real [BN,BM,BK] "b")))

def requirements (A B out : RegionName) (M N K BM BN BK : Nat) : List Site :=
  match Source.dynamicRange? (TypedSource.original A B out M N K BM BN BK).surfaceBody "k_index" with
  | none => []
  | some loop => [.range loop.before "k_index" loop.start loop.stop loop.stride loop.body
      (.add (Source.prefixBefore loop.body "vecmat") [BM,BN]
        (.ref .real [BM,BN] "vecmat") (blockSum BM BN BK))]

def originalProgram (A B out : RegionName) (M N K BM BN BK : Nat) : Program where
  kernel := TypedSource.original A B out M N K BM BN BK
  domain := requirements A B out M N K BM BN BK

def blockFirstProgram (A B out : RegionName) (M N K BM BN BK : Nat) : Program :=
  { originalProgram A B out M N K BM BN BK with kernel := TypedSource.blockFirst A B out M N K BM BN BK }

theorem block_first_equiv (A B out : RegionName) (M N K BM BN BK : Nat) (R : FP.ScalarArithmetic.Rules) :
    originalProgram A B out M N K BM BN BK ≡[R] blockFirstProgram A B out M N K BM BN BK := by
  equiv_decompose
  all_goals fp_prove

#print_fp_assumptions block_first_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.BatchedVecmatMult.FPEquivProofs
