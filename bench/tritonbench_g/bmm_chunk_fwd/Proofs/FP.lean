import bench.tritonbench_g.bmm_chunk_fwd.Kernels
import VeriTile.Triton.Float.RewriteTactics
import VeriTile.Meta.StatementAudit

/-! Commute each reached fp32 carry-plus-dot addition, preserving the complete
causal branch, K loop, signed sequence filter and output memory effects.
The local domain checks the actual typed dot and carry at each reached step;
it asserts no dot reassociation or final output equation. -/
namespace VeriTile.Bench.TritonBenchG.BmmChunkFwd.FPEquivProofs
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec
set_option maxHeartbeats 6000000
set_option maxRecDepth 18000

def dotDType (a b : TypedSource.Storage) : FloatDType :=
  if a = .bf16 ∨ b = .bf16 then .bf16
  else if a = .fp16 ∨ b = .fp16 then .fp16 else .fp32

def blockDot (a b : TypedSource.Storage) (BM BN BK : Nat) : Op .real [BM, BN] :=
  match dotDType a b with
  | .fp32 => .dot (batch := []) (.ref .real [BM, BK] "a") (.ref .real [BK, BN] "b")
  | d => .castFloat .fp32 .real (.dotTyped d d .fp32 (batch := [])
      (.ref d.toTileDType [BM, BK] "a") (.ref d.toTileDType [BK, BN] "b"))

def requirementsFor (kernel : ComputeKernel) (a b : TypedSource.Storage)
    (BM BN BK : Nat) : List Site :=
  (do
    let branch ← Source.conditional? kernel.surfaceBody
      (fun body => (Source.dynamicRange? body "k").isSome)
    let loop ← Source.dynamicRange? branch.yes "k"
    return Site.branch branch.before branch.condition Bool.true
      (.range loop.before "k" loop.start loop.stop loop.stride loop.body
        (.addFP32 (Source.prefixBefore loop.body "acc") [BM, BN]
          (.ref .real [BM, BN] "acc") (blockDot a b BM BN BK)))).toList

section
variable (aStorage bStorage : TypedSource.Storage) (out_dtype : FloatDType)
    (a b out : RegionName) (seq : Region .int)
    (seqlen chunk K groups sab sas sah sak sbb sbs sbh sbk sob soc soh som son ssb sss BM BN BK : Nat)
    (causal hasSeq : Bool)

def original : Program where
  kernel := TypedSource.original aStorage bStorage out_dtype a b out seq
    seqlen chunk K groups sab sas sah sak sbb sbs sbh sbk sob soc soh som son ssb sss BM BN BK causal hasSeq
  domain := requirementsFor (TypedSource.original aStorage bStorage out_dtype a b out seq
    seqlen chunk K groups sab sas sah sak sbb sbs sbh sbk sob soc soh som son ssb sss BM BN BK causal hasSeq)
    aStorage bStorage BM BN BK

def dotFirst : Program :=
  { original aStorage bStorage out_dtype a b out seq
      seqlen chunk K groups sab sas sah sak sbb sbs sbh sbk sob soc soh som son ssb sss BM BN BK causal hasSeq with
    kernel := TypedSource.dotFirst aStorage bStorage out_dtype a b out seq
      seqlen chunk K groups sab sas sah sak sbb sbs sbh sbk sob soc soh som son ssb sss BM BN BK causal hasSeq }

theorem requirements_present :
    (original aStorage bStorage out_dtype a b out seq
      seqlen chunk K groups sab sas sah sak sbb sbs sbh sbk sob soc soh som son ssb sss BM BN BK causal hasSeq).domain.length = 1 := by
  cases aStorage <;> cases bStorage <;> rfl

theorem dot_first_equiv (R : FP.ScalarArithmetic.Rules) :
    original aStorage bStorage out_dtype a b out seq
      seqlen chunk K groups sab sas sah sak sbb sbs sbh sbk sob soc soh som son ssb sss BM BN BK causal hasSeq ≡[R]
    dotFirst aStorage bStorage out_dtype a b out seq
      seqlen chunk K groups sab sas sah sak sbb sbs sbh sbk sob soc soh som son ssb sss BM BN BK causal hasSeq := by
  cases aStorage <;> cases bStorage <;> equiv_decompose <;> fp_prove

end
#print_fp_assumptions dot_first_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.BmmChunkFwd.FPEquivProofs
