import bench.tritonbench_g.bmm_chunk_fwd.Kernels
import bench.tritonbench_g.bmm_chunk_fwd.Proofs.FP
import VeriTile.Meta.StatementAudit

/-! Whole-program admitted FP equivalence for all nine A/B storage pairs.
The candidate replaces acc + tl.dot(a,b) by tl.dot(a,b) + acc in every reached
K iteration. Only add_commute is used; the typed dot, its fp32 accumulator,
load/output conversions, sequence filter and causal branch remain intact.

The domain checks the actual fp32 carry and typed dot result after each pair
of loads, in the selected causal branch and each reached K iteration. It
requires those operands to be finite, without prescribing the output or
restricting signed sequence IDs. This is equivalence under admitted atoms,
not a concrete IEEE equality or a dot/reduction reassociation. -/
namespace VeriTile.Bench.TritonBenchG.BmmChunkFwd.Equivalence
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec

section
variable (aStorage bStorage : TypedSource.Storage) (out_dtype : FloatDType)
    (a b out : RegionName) (seq : Region .int)
    (seqlen chunk K groups sab sas sah sak sbb sbs sbh sbk sob soc soh som son ssb sss BM BN BK : Nat)
    (causal hasSeq : Bool)

def original : Program where
  kernel := TypedSource.original aStorage bStorage out_dtype a b out seq
    seqlen chunk K groups sab sas sah sak sbb sbs sbh sbk sob soc soh som son ssb sss BM BN BK causal hasSeq
  domain := FPEquivProofs.requirementsFor (TypedSource.original aStorage bStorage out_dtype a b out seq
    seqlen chunk K groups sab sas sah sak sbb sbs sbh sbk sob soc soh som son ssb sss BM BN BK causal hasSeq)
    aStorage bStorage BM BN BK

def dotFirst : Program :=
  { original aStorage bStorage out_dtype a b out seq
      seqlen chunk K groups sab sas sah sak sbb sbs sbh sbk sob soc soh som son ssb sss BM BN BK causal hasSeq with
    kernel := TypedSource.dotFirst aStorage bStorage out_dtype a b out seq
      seqlen chunk K groups sab sas sah sak sbb sbs sbh sbk sob soc soh som son ssb sss BM BN BK causal hasSeq }

specification bmm_dot_first_equiv (R : FP.ScalarArithmetic.Rules) :
    original aStorage bStorage out_dtype a b out seq
      seqlen chunk K groups sab sas sah sak sbb sbs sbh sbk sob soc soh som son ssb sss BM BN BK causal hasSeq ≡[R]
    dotFirst aStorage bStorage out_dtype a b out seq
      seqlen chunk K groups sab sas sah sak sbb sbs sbh sbk sob soc soh som son ssb sss BM BN BK causal hasSeq := by
  exact FPEquivProofs.dot_first_equiv aStorage bStorage out_dtype a b out seq
    seqlen chunk K groups sab sas sah sak sbb sbs sbh sbk sob soc soh som son ssb sss BM BN BK causal hasSeq R

end
#print_fp_assumptions bmm_dot_first_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.BmmChunkFwd.Equivalence
