import bench.tritonbench_g.attn_fwd_causal.Proofs.RealContract

/-! A partial query block and an unaligned diagonal range. Its last tile has
no active K/V lanes but still reads a key scale. Bounds must retain that read.
The first range visits 0/2 and the reset second range visits 3/5. -/
namespace RealAttnFwdCausalIOTests
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.AttnFwdCausal
noncomputable section

private def configuration : RealLoads.Config where
  Q := "Q"
  K := "K"
  V := "V"
  QS := "QS"
  KS := "KS"
  base := 0
  queryScaleBase := 0
  keyScaleBase := 0
  strideM := 2
  strideD := 1
  strideKN := 2
  headDim := 2
  context := 4
  activeHead := 1
  startM := 1

private def allocated (scales : Nat) : RegionBounds :=
  fun r => if r = "KS" then scales else if r = "QS" then 2 else 8

theorem partial_unaligned_bounds :
    RealAccessSafety.Windows configuration "Out" 3 2 2 3 (allocated 3) := by
  constructor
  · intro r d hr hd
    simp +decide [configuration, allocated] at hr hd ⊢
    omega
  · intro v _ j d hj hd
    simp +decide [configuration, allocated] at hj hd ⊢
    omega
  · intro v _ j d hj hd
    simp +decide [configuration, allocated] at hj hd ⊢
    omega
  · decide
  · intro v hv
    have hv' : v = ⟨0,Bool.false⟩ ∨ v = ⟨2,Bool.false⟩ ∨ v = ⟨3,Bool.true⟩ ∨ v = ⟨5,Bool.true⟩ := by
      change v ∈ [⟨0,Bool.false⟩,⟨2,Bool.false⟩,⟨3,Bool.true⟩,⟨5,Bool.true⟩] at hv
      simpa only [List.mem_cons, List.not_mem_nil, or_false] using hv
    rcases hv' with rfl | rfl | rfl | rfl <;> decide
  · rintro ⟨r,d,⟨⟩⟩ hi
    simp +decide [RealEpilogue.active, RealEpilogue.address, configuration, allocated] at hi ⊢
    omega

theorem last_tile_has_no_active_values (j d : Fin 2) :
    ¬ (5+j.val < configuration.context ∧ d.val < configuration.activeHead) := by
  dsimp [configuration]
  omega

theorem missing_padded_scale :
    ¬ RealAccessSafety.Windows configuration "Out" 3 2 2 3 (allocated 2) := by
  intro h
  have hs := h.keyScale ⟨5,Bool.true⟩ (by decide)
  norm_num [configuration, allocated] at hs

#guard_msgs (drop info) in
#auditModuleAxioms
end
end RealAttnFwdCausalIOTests
