import bench.tritonbench_g.attention_kernel.Proofs.RealSource
import bench.tritonbench_g.attention_kernel.Proofs.RealNormalization
import VeriTile.Triton.Memory.KernelSpec.StreamAttentionReal
import VeriTile.Meta.StatementAudit

/-! Connect the cast-free IO proof to both complete typed sources after erasure.
The real and FP proofs remain independent. The reciprocal transfer uses equality
of complete flat-memory executions, retaining successful output and the frame. -/
namespace VeriTile.Bench.TritonBenchG.AttentionKernel.RealProofs
open VeriTile.Triton
open scoped VeriTile.Triton.StreamMasked3DKernelIO₄
set_option maxRecDepth 16000
set_option maxHeartbeats 600000

section
variable (storage : TypedSource.Storage) (Q K V B0 Out : RegionName) (sm_scale : ℝ)
    (sqh skh sbh BM BN HD BIAS sbm nB : Nat)

def originalIO : StreamMasked3DKernelIO₄ :=
  { RealKernel.attentionKernelIO Q K V B0 Out sm_scale sqh skh sbh BM BN HD BIAS sbm nB with
    kernel := (TypedSource.original storage Q K V B0 Out sm_scale sqh HD 1 skh HD 1 skh HD 1 sqh HD 1 sbh sbm 2 4 (BN * nB) 0 BIAS 128 HD BM BN).eraseDType
    projection := RealSource.original_projection_supported .. }

def reciprocalIO : StreamMasked3DKernelIO₄ :=
  { originalIO storage Q K V B0 Out sm_scale sqh skh sbh BM BN HD BIAS sbm nB with
    kernel := (TypedSource.reciprocal storage Q K V B0 Out sm_scale sqh HD 1 skh HD 1 skh HD 1 sqh HD 1 sbh sbm 2 4 (BN * nB) 0 BIAS 128 HD BM BN).eraseDType
    projection := RealSource.reciprocal_projection_supported .. }

/-- Reference: successful flat execution and the independent weighted sum. -/
theorem original_correct (hBM : 0 < BM) (hBN : 0 < BN) (hHD : 0 < HD) (hnB : 1 ≤ nB) :
    originalIO storage Q K V B0 Out sm_scale sqh skh sbh BM BN HD BIAS sbm nB ⊨
      fun _ _ _ xs ys zs ws j =>
        attnGenScore (RealKernel.akIOscoreT nB BM BN HD sm_scale xs ys ws)
          (RealKernel.akIOvT nB BN HD zs) (Lane2D.decode j) := by
  have h := StreamMasked3DKernelIO₄.Implements.of_identityRounding
    (RealKernel.attentionKernelIO Q K V B0 Out sm_scale sqh skh sbh BM BN HD BIAS sbm nB) rfl _
    (RealKernel.attention_kernel_io_correctness .triv rfl Q K V B0 Out sm_scale sqh skh sbh BM BN HD BIAS sbm nB hBM hBN hHD hnB)
  simpa only [StreamMasked3DKernelIO₄.Implements, originalIO,
    RealSource.original_real_source, RealKernel.attentionKernelIO] using h

/-- Candidate: same result, successful execution and unchanged-memory frame. -/
theorem reciprocal_correct (hBM : 0 < BM) (hBN : 0 < BN) (hHD : 0 < HD) (hnB : 1 ≤ nB) :
    reciprocalIO storage Q K V B0 Out sm_scale sqh skh sbh BM BN HD BIAS sbm nB ⊨
      fun _ _ _ xs ys zs ws j =>
        attnGenScore (RealKernel.akIOscoreT nB BM BN HD sm_scale xs ys ws)
          (RealKernel.akIOvT nB BN HD zs) (Lane2D.decode j) := by
  have h := original_correct storage Q K V B0 Out sm_scale sqh skh sbh BM BN HD BIAS sbm nB hBM hBN hHD hnB
  simpa only [StreamMasked3DKernelIO₄.Implements, reciprocalIO, originalIO,
    ← RealNormalization.flat_executions_agree] using h

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionKernel.RealProofs
