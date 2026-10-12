import bench.tritonbench_g.attention_fwd_triton2.Proofs.RealProjection
import bench.tritonbench_g.attention_forward_triton.Proofs.Real

/-! Complete real IO contracts transferred through checked equality of the
precision-erased full kernels. The five input windows, output mask, independent
formula, successful flat execution and frame all remain in the contract. -/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton2.RealProofs
open VeriTile.Triton
open scoped StreamMasked3DKernelIO₅
set_option maxHeartbeats 2400000
set_option maxRecDepth 16000

section
variable (Q K V QScale KScale Out : RegionName)
    (stride_qz stride_qh Z H HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE
      numKVBlocks : Nat)

def originalIO : StreamMasked3DKernelIO₅ :=
  { AttentionForwardTriton.RealProofs.originalIO Q K V QScale KScale Out
      stride_qz stride_qh Z H HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE numKVBlocks with
    kernel := (TypedSource.original Q K V QScale KScale Out
      stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
      stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
      Z H (BLOCK_N * numKVBlocks) HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE).eraseDType
    projection := RealProjection.original_projection_supported .. }

def reciprocalIO : StreamMasked3DKernelIO₅ :=
  { AttentionForwardTriton.RealProofs.reciprocalIO Q K V QScale KScale Out
      stride_qz stride_qh Z H HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE numKVBlocks with
    kernel := (TypedSource.reciprocal Q K V QScale KScale Out
      stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
      stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
      Z H (BLOCK_N * numKVBlocks) HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE).eraseDType
    projection := RealProjection.reciprocal_projection_supported .. }

end

/-- Reconstruct the mathematical tiles from the five pinned input streams.
Q uses the repeated stream's first slice; K/V and key scales use the block
containing each global key. The helpers only index inputs, never execute code. -/
noncomputable def expected (p₀ BM BN BD HA T : Nat)
    (xs : Fin T → Fin (BM * BD) → ℝ)
    (ys : Fin T → Fin (BD * BN) → ℝ)
    (zs : Fin T → Fin (BN * BD) → ℝ)
    (ws vs : Fin T → Fin 1 → ℝ) (j : Fin (BM * BD)) : ℝ :=
  if h : j.val % BD < HA then
    let Q := AttentionForwardTriton.attnIOqTm p₀ (BN * T) BM BD HA T xs
    let K := AttentionForwardTriton.attnIOkTm BN BD HA T ys
    let V := AttentionForwardTriton.attnIOvTm BN BD HA T zs
    let scale := AttentionForwardTriton.attnIOkeyScale BN T ws vs
    let i := (Lane2D.decode j).1
    let d : Fin HA := ⟨j.val % BD, h⟩
    let weight := fun k : Fin (BN * T) => Real.exp (Real.log 2 *
      (scale k * ∑ e : Fin HA, Q (i, e, PUnit.unit) * K (k, e, PUnit.unit)))
    (∑ k : Fin (BN * T), weight k * V (k, d, PUnit.unit)) /
      ∑ k : Fin (BN * T), weight k
  else 0

theorem original_correct
    (Q K V QScale KScale Out : RegionName)
    (stride_qz stride_qh Z H HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE
      numKVBlocks : Nat)
    (hBN : 0 < BLOCK_N) (hnum : 0 < numKVBlocks)
    (hActiveLe : HEAD_ACTIVE ≤ BLOCK_DMODEL) (hHD : HEAD_ACTIVE ≤ HEAD_DIM) :
    originalIO Q K V QScale KScale Out
      stride_qz stride_qh Z H HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE numKVBlocks ⊨
      fun p₀ _ _ xs ys zs ws vs j =>
        expected p₀ BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks xs ys zs ws vs j := by
  have h := AttentionForwardTriton.RealProofs.original_correct Q K V QScale KScale Out
      stride_qz stride_qh Z H HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE numKVBlocks
    hBN hnum hActiveLe hHD
  simpa only [StreamMasked3DKernelIO₅.Implements, originalIO,
    RealProjection.original_real_source] using h

theorem reciprocal_correct
    (Q K V QScale KScale Out : RegionName)
    (stride_qz stride_qh Z H HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE
      numKVBlocks : Nat)
    (hBN : 0 < BLOCK_N) (hnum : 0 < numKVBlocks)
    (hActiveLe : HEAD_ACTIVE ≤ BLOCK_DMODEL) (hHD : HEAD_ACTIVE ≤ HEAD_DIM) :
    reciprocalIO Q K V QScale KScale Out
      stride_qz stride_qh Z H HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE numKVBlocks ⊨
      fun p₀ _ _ xs ys zs ws vs j =>
        expected p₀ BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks xs ys zs ws vs j := by
  have h := AttentionForwardTriton.RealProofs.reciprocal_correct Q K V QScale KScale Out
      stride_qz stride_qh Z H HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE numKVBlocks
    hBN hnum hActiveLe hHD
  simpa only [StreamMasked3DKernelIO₅.Implements, reciprocalIO,
    RealProjection.reciprocal_real_source] using h

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton2.RealProofs
