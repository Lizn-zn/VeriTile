import bench.tritonbench_g.chunk_delta_fwd.Correct
import bench.tritonbench_g.chunk_delta_fwd.Proofs.UndefinedTail
import VeriTile.Meta.Disprove

/-! Complete-source regression with all four initial/final-state switch pairs.
BT=3, BC=2 makes four-row update windows overlap at row 3. The second window
must overwrite row 3 using the new H, while earlier V rows and masked neighbors
remain intact. K/V/time tails are all present. -/
namespace VeriTile.Tests.ChunkDeltaRealExecution
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkDeltaFwd
open RealProgram
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

private abbrev p : Layout where
  keyHead := 15
  keyTime := 3
  keyDim := 1
  valueHead := 15
  valueTime := 3
  valueDim := 1
  historyHead := 18
  historyTime := 3
  H := 1
  T := 5
  K := 3
  V := 3
  BT := 3
  BC := 2
  BK := 4
  BV := 2
  NT := 2

private def r : Regions := ⟨"k","d","v","v_new","h","h0","ht"⟩
private def a : Inputs := ⟨fun _ => 2,fun _ => 1,fun _ => 20⟩
private def a₀ : Nat → ℝ := fun _ => 1
private def u : BlockState where
  mem := fun region _ =>
    if region = "k" then MemCell.real 2
    else if region = "d" ∨ region = "h0" then MemCell.real 1
    else if region = "v" then MemCell.real 20
    else MemCell.of .nat 99
  regs := fun _ _ _ => none
  pids := fun ax => if ax = 1 then 1 else 0

private def l := RealInit.launch u

private theorem values_injective : RealInnerLoop.NoCollision p l := by
  intro t s i j ha hb he
  dsimp [activeValue, valueAddress, valueIndex, p, l, RealInit.launch, u] at ha hb he
  have heq : t = s ∧ i.val = j.val := by omega
  exact ⟨heq.1,Fin.ext heq.2⟩

private theorem history_injective : RealOuterLoop.HistoryNoCollision p l := by
  intro c d e f i j _ _ ha hb he
  dsimp [RealStateStore.active, keyIndex, valueIndex, RealOuterStep.historyAddress,
    RealOuterStep.historyBase, RealStateStore.address, p, l, RealInit.launch, u] at ha hb he
  have heq : c = d ∧ e.val = f.val ∧ i.val = j.val := by omega
  exact ⟨heq.1,Fin.ext heq.2.1,Fin.ext heq.2.2⟩

private theorem final_injective : RealStateStore.NoCollision p l (RealInit.base p l) p.V := by
  intro e f i j ha hb he
  dsimp [RealStateStore.active, keyIndex, valueIndex, RealInit.base, RealStateStore.address,
    p, l, RealInit.launch, u] at ha hb he
  have heq : e.val = f.val ∧ i.val = j.val := by omega
  exact ⟨Fin.ext heq.1,Fin.ext heq.2⟩

private theorem state_zero (useSeed : Bool) :
    RealWindow.expected p l a (RealInit.seed p l a₀) useSeed 0 0 2 =
      if useSeed then 1 else 0 := by
  cases useSeed <;> rfl

private theorem state_one (useSeed : Bool) :
    RealWindow.expected p l a (RealInit.seed p l a₀) useSeed 1 0 2 =
      if useSeed then 137 else 160 := by
  simp only [RealWindow.expected, show RealWindow.span p = 4 from rfl]
  cases useSeed <;> norm_num [RealWindow.expected, RealWindow.state, RealWindow.span,
    RealInnerLoop.count, RealMath.corrected, RealValues.arrays, RealInit.seed, a, a₀,
    p, Fin.sum_univ_succ]

private theorem state_two (useSeed : Bool) :
    RealWindow.expected p l a (RealInit.seed p l a₀) useSeed 2 0 2 =
      if useSeed then -1427 else -1680 := by
  simp only [RealWindow.expected, show RealWindow.span p = 4 from rfl]
  cases useSeed <;> norm_num [RealWindow.expected, RealWindow.state, RealWindow.span,
    RealInnerLoop.count, RealMath.corrected, RealValues.arrays, RealInit.seed, a, a₀,
    p, Fin.sum_univ_succ]

private theorem value_zero (useSeed : Bool) :
    RealMath.corrected (RealValues.arrays p l a (RealInit.seed p l a₀)) p.K
      (RealWindow.expected p l a (RealInit.seed p l a₀) useSeed (0 / p.BT)) 0 2 =
      if useSeed then 17 else 20 := by
  cases useSeed <;> norm_num [RealMath.corrected, RealValues.arrays, RealWindow.expected,
    RealWindow.state, RealInit.seed, a, a₀, p, Fin.sum_univ_succ]

private theorem value_overlap (useSeed : Bool) :
    RealMath.corrected (RealValues.arrays p l a (RealInit.seed p l a₀)) p.K
      (RealWindow.expected p l a (RealInit.seed p l a₀) useSeed (3 / p.BT)) 3 2 =
      if useSeed then -391 else -460 := by
  simp only [RealWindow.expected, show RealWindow.span p = 4 from rfl]
  cases useSeed <;> norm_num [RealMath.corrected, RealValues.arrays, RealWindow.expected,
    RealWindow.state, RealWindow.span, RealInnerLoop.count, RealInit.seed, a, a₀, p, Fin.sum_univ_succ]

theorem complete_source_all_switches (useSeed storeFinal : Bool) : ∃ v : BlockState,
    exec (RealExecution.source p r .fp16 .bf16 .bf16 .fp32 useSeed storeFinal).eraseDType.toAlgKernel u = some v ∧
    v.readMem "h" 2 = (if useSeed then 1 else 0) ∧
    v.readMem "h" 11 = (if useSeed then 137 else 160) ∧
    v.readMem "v_new" 2 = (if useSeed then 17 else 20) ∧
    v.readMem "v_new" 11 = (if useSeed then -391 else -460) ∧
    (storeFinal = Bool.true → v.readMem "ht" 2 = (if useSeed then -1427 else -1680)) ∧
    v.mem "v_new" 3 = MemCell.of .nat 99 ∧
    (storeFinal = Bool.false → v.mem "ht" 2 = MemCell.of .nat 99) := by
  obtain ⟨v,he,h,hframe⟩ := RealExecution.executes p r .fp16 .bf16 .bf16 .fp32 useSeed storeFinal a a₀ u
    (by rfl) (by rfl) (by decide) (by decide) (by decide) (by decide)
    (by constructor <;> decide) (by intro _; constructor <;> decide)
    values_injective history_injective (fun _ => final_injective)
    (by intro off; simp [u,r,a,BlockState.readMem,MemCell.readAs,MemCell.real,MemCell.of])
    (by intro off; simp [u,r,a,BlockState.readMem,MemCell.readAs,MemCell.real,MemCell.of])
    (by intro off; simp [u,r,a,BlockState.readMem,MemCell.readAs,MemCell.real,MemCell.of])
    (by intro off; simp [u,r,a₀,BlockState.readMem,MemCell.readAs,MemCell.real,MemCell.of])
  refine ⟨v,he,?_,?_,?_,?_,?_,?_,?_⟩
  · exact (h.history 0 0 0 (by decide) (by decide)).trans (state_zero useSeed)
  · exact (h.history 1 0 0 (by decide) (by decide)).trans (state_one useSeed)
  · exact (h.corrected 0 0 (by decide)).trans (value_zero useSeed)
  · exact (h.corrected 3 0 (by decide)).trans (value_overlap useSeed)
  · intro hon
    exact (h.final hon 0 0 (by decide)).trans (state_two useSeed)
  · have hf := hframe "v_new" 3 (by
      rintro ((hh | hv) | hf)
      · exact (by decide : ("v_new" : RegionName) ≠ "h") hh.1
      · rcases hv with ⟨_,t,j,ha,hoff⟩
        dsimp [activeValue,valueAddress,valueIndex,p,RealInit.launch,u] at ha hoff
        omega
      · exact (by decide : ("v_new" : RegionName) ≠ "ht") hf.2.1)
    exact hf.symm
  · intro hoff
    have hf := hframe "ht" 2 (by
      rintro ((hh | hv) | hf)
      · exact (by decide : ("ht" : RegionName) ≠ "h") hh.1
      · exact (by decide : ("ht" : RegionName) ≠ "v_new") hv.1
      · simp [RealFinal.footprint,hoff] at hf)
    exact hf.symm

/-- Bounds use active lanes only, including the overlapping time window and
all K/V/time tails of this concrete source geometry. -/
theorem source_bounds (useSeed storeFinal : Bool) :
    RealSafety.Bounds p l r useSeed storeFinal (fun _ => 18) := by
  refine ⟨?_,⟨⟨?_,?_,?_,?_⟩,?_⟩,?_⟩
  · intro _ e j ha
    dsimp [RealStateStore.active,RealStateStore.address,RealInit.base,keyIndex,valueIndex,p,l,RealInit.launch,u] at ha ⊢
    omega
  · intro t e ha
    dsimp [keyIndex,keyAddress,p,l,RealInit.launch,u] at ha ⊢
    omega
  · intro t e ha
    dsimp [keyIndex,deltaAddress,p,l,RealInit.launch,u] at ha ⊢
    omega
  · intro t j ha
    dsimp [activeValue,valueIndex,valueAddress,p,l,RealInit.launch,u] at ha ⊢
    omega
  · intro t j ha
    dsimp [activeValue,valueIndex,valueAddress,p,l,RealInit.launch,u] at ha ⊢
    omega
  · intro c hc e j ha
    dsimp [RealStateStore.active,RealStateStore.address,RealOuterStep.historyBase,keyIndex,valueIndex,p,l,RealInit.launch,u] at ha hc ⊢
    omega
  · intro _ e j ha
    dsimp [RealStateStore.active,RealStateStore.address,RealInit.base,keyIndex,valueIndex,p,l,RealInit.launch,u] at ha ⊢
    omega

theorem source_safe_all_switches (useSeed storeFinal : Bool) :
    (RealExecution.source p r .fp16 .bf16 .bf16 .fp32 useSeed storeFinal).eraseDType.toAlgKernel.TraceSafe
      (fun _ => 18) u :=
  RealSafety.source_safe p r .fp16 .bf16 .bf16 .fp32 useSeed storeFinal u rfl rfl
    (by decide) (by decide) (by constructor <;> decide) values_injective history_injective
    _ (source_bounds useSeed storeFinal)

/-- An extent of 17 misses the last active history cell, at offset 17. -/
theorem undersized_history_rejected (useSeed storeFinal : Bool) :
    ¬ RealSafety.Bounds p l r useSeed storeFinal (fun _ => 17) := by
  intro h
  have bad := h.loop.history 1 (by decide) 2 0 (by decide)
  norm_num [RealStateStore.address,RealOuterStep.historyBase,keyIndex,valueIndex,p,l,RealInit.launch,u] at bad

/-- Both complete variants retain the same observed overwritten row. -/
theorem candidate_all_switches (useSeed storeFinal : Bool) : ∃ v,
    exec (RealCandidate.source p r .fp16 .bf16 .bf16 .fp32 Bool.true useSeed storeFinal).eraseDType.toAlgKernel u = some v ∧
      v.readMem "v_new" 11 = (if useSeed then -391 else -460) := by
  obtain ⟨v,he,_,_,_,hv,_⟩ := complete_source_all_switches useSeed storeFinal
  refine ⟨v,?_,hv⟩
  rw [← RealCandidate.execution_eq]
  exact he

/-- For this source geometry, silently replacing its four-row update by a
three-row disjoint update is false. This concerns real source semantics, not
GPU compilation or a two-gates admission. -/
refutation disjoint_window_claim : ∀ v : BlockState,
    exec (RealExecution.source p r .fp16 .bf16 .bf16 .fp32 Bool.true Bool.true).eraseDType.toAlgKernel u = some v →
    v.readMem "h" 11 =
      RealMath.state (RealValues.arrays p l a (RealInit.seed p l a₀)) p.T p.K p.BT Bool.true 1 0 2 := by
  obtain ⟨v,he,_,hh,_⟩ := complete_source_all_switches Bool.true Bool.true
  have hstandard : RealMath.state (RealValues.arrays p l a (RealInit.seed p l a₀))
      p.T p.K p.BT Bool.true 1 0 2 = 103 := by
    norm_num [RealMath.state,RealMath.corrected,RealValues.arrays,RealInit.seed,a,a₀,p,Fin.sum_univ_succ]
  intro h
  have bad := h v he
  rw [hh,hstandard] at bad
  norm_num at bad

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Tests.ChunkDeltaRealExecution
