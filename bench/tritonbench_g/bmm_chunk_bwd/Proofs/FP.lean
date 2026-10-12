import bench.tritonbench_g.bmm_chunk_bwd.Kernels
import VeriTile.Triton.Float.RewriteTactics
import VeriTile.Meta.StatementAudit

/-! Residual-add commutation in the complete source. The actual fp32
accumulator and converted residual must be finite in the reached branch,
after its complete chunk loop and residual load. Both storage types are
independent; no-residual execution stays unchanged. Executability and the
independent real output formula are separate obligations. -/
namespace VeriTile.Bench.TritonBenchG.BmmChunkBwd.FPEquivProofs
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000

def requirementsFor (kernel : ComputeKernel) (BM BN : Nat) : List Site :=
  (do
    let branch ← Source.conditional? kernel.surfaceBody
      (fun body => decide ((Source.prefixBefore body "res").length < body.length))
    return Site.branch branch.before branch.condition Bool.true
      (.addFP32 (Source.prefixBefore branch.yes "acc") [BM, BN]
        (.ref .real [BM, BN] "acc") (.ref .real [BM, BN] "res"))).toList

section
variable (aStorage doutStorage : TypedSource.Storage) (dtype out_dtype : FloatDType)
    (a dout db res : RegionName)
    (seqlen chunk K groups sab sas sah sak sdb sdc sdh sdm sdn sob sos soh sok srb srs srh srk BM BN BCS : Nat)
    (hasResidual : Bool)

def original : Program where
  kernel := TypedSource.original aStorage doutStorage dtype out_dtype a dout db res
    seqlen chunk K groups sab sas sah sak sdb sdc sdh sdm sdn sob sos soh sok srb srs srh srk BM BN BCS hasResidual
  domain := requirementsFor (TypedSource.original aStorage doutStorage dtype out_dtype a dout db res
    seqlen chunk K groups sab sas sah sak sdb sdc sdh sdm sdn sob sos soh sok srb srs srh srk BM BN BCS hasResidual) BM BN

def residualLeft : Program :=
  { original aStorage doutStorage dtype out_dtype a dout db res
      seqlen chunk K groups sab sas sah sak sdb sdc sdh sdm sdn sob sos soh sok srb srs srh srk BM BN BCS hasResidual with
    kernel := TypedSource.residualLeft aStorage doutStorage dtype out_dtype a dout db res
      seqlen chunk K groups sab sas sah sak sdb sdc sdh sdm sdn sob sos soh sok srb srs srh srk BM BN BCS hasResidual }

theorem requirements_present :
    (original aStorage doutStorage dtype out_dtype a dout db res
      seqlen chunk K groups sab sas sah sak sdb sdc sdh sdm sdn sob sos soh sok srb srs srh srk BM BN BCS hasResidual).domain.length = 1 := by
  cases aStorage <;> cases doutStorage <;> rfl

theorem residual_equiv (R : FP.ScalarArithmetic.Rules) :
    original aStorage doutStorage dtype out_dtype a dout db res
      seqlen chunk K groups sab sas sah sak sdb sdc sdh sdm sdn sob sos soh sok srb srs srh srk BM BN BCS hasResidual ≡[R]
    residualLeft aStorage doutStorage dtype out_dtype a dout db res
      seqlen chunk K groups sab sas sah sak sdb sdc sdh sdm sdn sob sos soh sok srb srs srh srk BM BN BCS hasResidual := by
  cases aStorage <;> cases doutStorage <;> equiv_decompose <;> fp_prove

end
#print_fp_assumptions residual_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.BmmChunkBwd.FPEquivProofs
