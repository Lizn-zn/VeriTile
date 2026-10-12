import bench.tritonbench_g.batched_vecmat_mult.Kernels
import bench.tritonbench_g.batched_vecmat_mult.Proofs.FP
import VeriTile.Meta.StatementAudit

/-! Whole-program FP equivalence for the two sources in Kernels.lean.
A/B/output are fp32 and ProfiledRewrite supplies default fp32 arithmetic for
products, reductions and accumulation. Only each carry+blockSum addition is
commuted. The domain checks both actual operands in every reached iteration;
reduction order, unmasked memory accesses, stores and failures are preserved.
There is no positivity requirement, output equation or experiment-shape bound.
TypedExecution supplies a consistent nonempty execution/domain witness. -/
namespace VeriTile.Bench.TritonBenchG.BatchedVecmatMult.FPEquivalence
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec
abbrev Rules := FP.ScalarArithmetic.Rules

/-- The actual RHS block product, reduction and transpose. The broadcasted A
register has leading dimension 1; multiplication expands it to [BN,BM,BK]. -/
def blockSum (BM BN BK : Nat) : Op .real [BM,BN] :=
  .transpose (batch := []) (.reduceSum ⟨2, by simp⟩ Bool.false
    (.mul .real (.consL (.consSame (.consSame .nil)))
      (.ref .real [1,BM,BK] "expanded_a") (.ref .real [BN,BM,BK] "b")))

/-- Finiteness of the current carry and computed block sum at every reached
addition, after actual loads and broadcast. No reassociation is assumed. -/
def requirements (A B out : RegionName) (M N K BM BN BK : Nat) : List Site :=
  match Source.dynamicRange? (TypedSource.original A B out M N K BM BN BK).surfaceBody "k_index" with
  | none => []
  | some loop => [.range loop.before "k_index" loop.start loop.stop loop.stride loop.body
      (.add (Source.prefixBefore loop.body "vecmat") [BM,BN]
        (.ref .real [BM,BN] "vecmat") (blockSum BM BN BK))]

def original (A B out : RegionName) (M N K BM BN BK : Nat) : Program where
  kernel := TypedSource.original A B out M N K BM BN BK
  domain := requirements A B out M N K BM BN BK

def blockFirst (A B out : RegionName) (M N K BM BN BK : Nat) : Program :=
  { original A B out M N K BM BN BK with kernel := TypedSource.blockFirst A B out M N K BM BN BK }

specification vecmat_block_first_equiv (A B out : RegionName) (M N K BM BN BK : Nat) (R : Rules) :
    original A B out M N K BM BN BK ≡[R] blockFirst A B out M N K BM BN BK := by
  exact FPEquivProofs.block_first_equiv A B out M N K BM BN BK R

#print_fp_assumptions vecmat_block_first_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.BatchedVecmatMult.FPEquivalence
