import bench.tritonbench_g.attention_forward_triton.Kernels
import bench.tritonbench_g.attention_forward_triton.Proofs.Real
import VeriTile.Meta.StatementAudit

/-! Real correctness of both complete sources in Kernels.lean.
The independent formula is Σⱼ 2^(qScale·kScaleⱼ·⟨qᵢ,kⱼ⟩)·vⱼ / Σⱼ 2^(qScale·kScaleⱼ·⟨qᵢ,kⱼ⟩).
The IO contract proves successful execution, active-address safety and the frame
outside the masked output window for each program; host grid composition is
outside this statement.

Scope: N_CTX = BLOCK_N * numKVBlocks, shared batch/head strides, contiguous head
rows and unit inner stride. Block and head dimensions remain symbolic. The
source's partially filled KV block does not compute standard attention:
Proofs/TailCounterexample.lean gives a checked counterexample. -/
namespace VeriTile.Bench.TritonBenchG.AttentionForwardTriton.Correctness
open VeriTile.Triton
open scoped VeriTile.Triton.StreamMasked3DKernelIO₅

/-- Five input streams: static Q and Q_scale, and streamed K, V and K_scale.
The flat allocation must cover each active read/write address. Q and output
have the same row/head mask; the two scale loads are unmasked. -/
def originalIO (Q K V QScale KScale Out : RegionName)
    (stride_qz stride_qh Z H HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE
      numKVBlocks : Nat) : StreamMasked3DKernelIO₅ where
  kernel := (TypedSource.original Q K V QScale KScale Out
    stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
    stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
    Z H (BLOCK_N * numKVBlocks) HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE).eraseDType
  projection := RealNormalization.original_projection_supported ..
  inp1 := Q
  inp2 := K
  inp3 := V
  inp4 := QScale
  inp5 := KScale
  out := Out
  T := numKVBlocks
  B1 := BLOCK_M * BLOCK_DMODEL
  B2 := BLOCK_DMODEL * BLOCK_N
  B3 := BLOCK_N * BLOCK_DMODEL
  B4 := 1
  B5 := 1
  C := BLOCK_M * BLOCK_DMODEL
  outDType := .real
  read1 := fun p₀ p₁ _ _ j =>
    p₁ / H * stride_qz + p₁ % H * stride_qh
      + (p₀ * BLOCK_M + j.val / BLOCK_DMODEL) * HEAD_DIM + j.val % BLOCK_DMODEL
  read2 := fun _ p₁ _ t j =>
    p₁ / H * stride_qz + p₁ % H * stride_qh
      + j.val / BLOCK_N + (t.val * BLOCK_N + j.val % BLOCK_N) * HEAD_DIM
  read3 := fun _ p₁ _ t j =>
    p₁ / H * stride_qz + p₁ % H * stride_qh
      + (t.val * BLOCK_N + j.val / BLOCK_DMODEL) * HEAD_DIM + j.val % BLOCK_DMODEL
  read4 := fun p₀ p₁ _ _ _ =>
    p₁ * ((BLOCK_N * numKVBlocks + BLOCK_M - 1) / BLOCK_M) + p₀
  read5 := fun _ p₁ _ t _ =>
    p₁ * ((BLOCK_N * numKVBlocks + BLOCK_N - 1) / BLOCK_N) + t.val
  write := fun p₀ p₁ _ j =>
    p₁ / H * stride_qz + p₁ % H * stride_qh
      + (p₀ * BLOCK_M + j.val / BLOCK_DMODEL) * HEAD_DIM + j.val % BLOCK_DMODEL
  mask1 := fun p₀ _ _ _ j =>
    p₀ * BLOCK_M + j.val / BLOCK_DMODEL < BLOCK_N * numKVBlocks
      ∧ j.val % BLOCK_DMODEL < HEAD_ACTIVE
  mask2 := fun _ _ _ t j =>
    j.val % BLOCK_N < BLOCK_N * numKVBlocks - t.val * BLOCK_N ∧ j.val / BLOCK_N < HEAD_ACTIVE
  mask3 := fun _ _ _ t j =>
    j.val / BLOCK_DMODEL < BLOCK_N * numKVBlocks - t.val * BLOCK_N
      ∧ j.val % BLOCK_DMODEL < HEAD_ACTIVE
  mask4 := fun _ _ _ _ _ => True
  mask5 := fun _ _ _ _ _ => True
  writeMask := fun p₀ _ _ j =>
    p₀ * BLOCK_M + j.val / BLOCK_DMODEL < BLOCK_N * numKVBlocks
      ∧ j.val % BLOCK_DMODEL < HEAD_ACTIVE

/-- The reciprocal candidate retains all six windows and masks. -/
def reciprocalIO (Q K V QScale KScale Out : RegionName)
    (stride_qz stride_qh Z H HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE
      numKVBlocks : Nat) : StreamMasked3DKernelIO₅ :=
  { originalIO Q K V QScale KScale Out stride_qz stride_qh Z H HEAD_DIM BLOCK_M BLOCK_N
      BLOCK_DMODEL HEAD_ACTIVE STAGE numKVBlocks with
    kernel := (TypedSource.reciprocal Q K V QScale KScale Out
      stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
      stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
      Z H (BLOCK_N * numKVBlocks) HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE).eraseDType
    projection := RealNormalization.reciprocal_projection_supported .. }

/-- Reconstruct the mathematical tiles from the five pinned input streams.
Q uses the repeated stream's first slice; K/V and key scales use the block
containing each global key. The helpers only index inputs, never execute code. -/
noncomputable def expected (p₀ BM BN BD HA T : Nat)
    (xs : Fin T → Fin (BM * BD) → ℝ)
    (ys : Fin T → Fin (BD * BN) → ℝ)
    (zs : Fin T → Fin (BN * BD) → ℝ)
    (ws vs : Fin T → Fin 1 → ℝ) (j : Fin (BM * BD)) : ℝ :=
  if h : j.val % BD < HA then
    let Q := attnIOqTm p₀ (BN * T) BM BD HA T xs
    let K := attnIOkTm BN BD HA T ys
    let V := attnIOvTm BN BD HA T zs
    let scale := attnIOkeyScale BN T ws vs
    let i := (Lane2D.decode j).1
    let d : Fin HA := ⟨j.val % BD, h⟩
    let weight := fun k : Fin (BN * T) => Real.exp (Real.log 2 *
      (scale k * ∑ e : Fin HA, Q (i, e, PUnit.unit) * K (k, e, PUnit.unit)))
    (∑ k : Fin (BN * T), weight k * V (k, d, PUnit.unit)) /
      ∑ k : Fin (BN * T), weight k
  else 0

/-- Both positive key-block dimensions are needed for the nonempty reduction.
Active head lanes fit the tile and distinct output rows do not overlap. -/
specification attention_correctness
    (Q K V QScale KScale Out : RegionName)
    (stride_qz stride_qh Z H HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE
      numKVBlocks : Nat)
    (hBN : 0 < BLOCK_N) (hnum : 0 < numKVBlocks)
    (hActiveLe : HEAD_ACTIVE ≤ BLOCK_DMODEL) (hHD : HEAD_ACTIVE ≤ HEAD_DIM) :
    Spec.Real (originalIO Q K V QScale KScale Out stride_qz stride_qh Z H HEAD_DIM
      BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE numKVBlocks ⊨
      fun p₀ _ _ xs ys zs ws vs j =>
        expected p₀ BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks xs ys zs ws vs j) := by
  exact RealProofs.original_correct Q K V QScale KScale Out stride_qz stride_qh Z H HEAD_DIM
    BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE numKVBlocks hBN hnum hActiveLe hHD

/-- The complete candidate satisfies the same independent attention formula. -/
specification attention_reciprocal_correctness
    (Q K V QScale KScale Out : RegionName)
    (stride_qz stride_qh Z H HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE
      numKVBlocks : Nat)
    (hBN : 0 < BLOCK_N) (hnum : 0 < numKVBlocks)
    (hActiveLe : HEAD_ACTIVE ≤ BLOCK_DMODEL) (hHD : HEAD_ACTIVE ≤ HEAD_DIM) :
    Spec.Real (reciprocalIO Q K V QScale KScale Out stride_qz stride_qh Z H HEAD_DIM
      BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE numKVBlocks ⊨
      fun p₀ _ _ xs ys zs ws vs j =>
        expected p₀ BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks xs ys zs ws vs j) := by
  exact RealProofs.reciprocal_correct Q K V QScale KScale Out stride_qz stride_qh Z H HEAD_DIM
    BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE numKVBlocks hBN hnum hActiveLe hHD

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionForwardTriton.Correctness
