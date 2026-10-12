import bench.tritonbench_g.chunk_gate_recurrence.Proofs.RealForwardFlat
import bench.tritonbench_g.chunk_gate_recurrence.Proofs.RealForwardBounds

/-! Complete-source real regressions: three chunks, both seed flags and both
versions; seed aliases the output at the same addresses. A neighboring tagged
integer cell must stay intact. Allocation checks also cover N=1 without input
allocations and reject a missing final output address. -/
namespace VeriTile.Tests.ChunkGateForwardReal
open VeriTile.Triton VeriTile.Bench.TritonBenchG.ChunkGateRecurrence RealProgram
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

private abbrev p : Layout := ⟨3,4,4,2,2⟩
private def u : BlockState where
  mem := fun r off =>
    if r = "S" then MemCell.real (if off < 16 then 1 else 2)
    else if r = "D" then MemCell.real 2
    else if r = "O" ∧ off ≠ 9 then MemCell.real 3
    else MemCell.of .nat 77
  regs := fun _ _ _ => none
  pids := fun ax => if ax = 0 then 0 else 1
  undef := fun _ _ => 7

private def a : Inputs p where
  source := fun c _ _ => if c = 0 then 1 else 2
  gate := fun _ => 2
  seed := fun _ _ => 3
  incoming := fun _ _ _ => 0

private theorem source_values : ∀ (t : Nat) (e : Fin p.BK) (j : Fin p.BV), t < p.N →
    u.readMem "S" (address p (launch u) t e.val j.val) = a.source t e j := by
  intro t e j ht
  change t < 3 at ht
  interval_cases t <;> fin_cases e <;> fin_cases j <;>
    simp [BlockState.readMem,u,address,launch,p,a,String.reduceEq]


private theorem gate_values : ∀ t, t < p.N → u.readMem "D" (gateAddress p (launch u) t) = a.gate t := by
  intro t _
  simp [u,a,BlockState.readMem,MemCell.real,MemCell.readAs,MemCell.of]

private theorem seed_values (hasSeed : Bool) : hasSeed = Bool.true → ∀ e j,
    u.readMem "O" (seedAddress p (launch u) e.val j.val) = a.seed e j := by
  intro _ e j
  fin_cases e <;> fin_cases j <;>
    simp [BlockState.readMem,u,seedAddress,launch,p,a,String.reduceEq]


private theorem value_zero (hasSeed : Bool) :
    RealMath.forward a hasSeed 0 0 0 = if hasSeed then 3 else 0 := by
  rw [RealMath.forward_zero]
  cases hasSeed <;> simp [RealMath.initial,a]

private theorem value_one (hasSeed : Bool) :
    RealMath.forward a hasSeed 1 0 0 = if hasSeed then 7 else 1 := by
  rw [RealMath.forward_step a hasSeed 0, value_zero]
  cases hasSeed <;> norm_num [a]

private theorem value_two (hasSeed : Bool) :
    RealMath.forward a hasSeed 2 0 0 = if hasSeed then 16 else 4 := by
  rw [RealMath.forward_step a hasSeed 1, value_one]
  cases hasSeed <;> norm_num [a]

theorem complete_source (swapped hasSeed : Bool) : ∃ v : BlockState,
    exec (RealForwardFlat.source swapped p "S" "D" "O" "O" hasSeed .fp16 .bf16 .fp16 1).eraseDType.toAlgKernel u = some v ∧
    v.readMem "O" 10 = (if hasSeed then 3 else 0) ∧
    v.readMem "O" 26 = (if hasSeed then 7 else 1) ∧
    v.readMem "O" 42 = (if hasSeed then 16 else 4) ∧
    v.mem "O" 9 = MemCell.of .nat 77 := by
  obtain ⟨v,he,ho,hf⟩ := RealForward.original_executes p "S" "D" "O" "O" a hasSeed u
    .fp16 .bf16 .fp16 1 (by decide) (by decide) (by decide) (by decide) (by decide)
    source_values gate_values (seed_values hasSeed)
  refine ⟨v,?_,?_,?_,?_,?_⟩
  · cases swapped
    · exact he
    · change exec (TypedSource.forwardSumLeft .fp16 .bf16 .fp16 "S" "D" "O" "O" 1 3 4 4 2 2 hasSeed).eraseDType.toAlgKernel u = _
      rw [← RealForwardCandidate.region_exec_equal p]
      exact he
  · exact (ho 0 0 0 (by decide)).trans (value_zero hasSeed)
  · exact (ho 1 0 0 (by decide)).trans (value_one hasSeed)
  · exact (ho 2 0 0 (by decide)).trans (value_two hasSeed)
  · have h := hf "O" 9 (by
      rintro ⟨_,t,e,j,ht,heq⟩
      dsimp [address,launch,u,p] at heq
      omega)
    exact h.symm

theorem sufficient_bounds (hasSeed : Bool) :
    RealForwardSafety.Bounds p (launch u) "S" "D" "O" "O" hasSeed
      (fun r => if r = "S" then 32 else if r = "D" then 2 else 48) := by
  apply RealForwardBounds.sufficient
  all_goals norm_num [p,launch,u]
  all_goals simp [String.reduceEq]

/-- The same source/value/frame check survives arbitrary disjoint placement in
one flat allocation. The initial unspecified-value oracle is seven, not zero. -/
theorem complete_flat (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0)
    (hS : 32 ≤ A.extent "S") (hD : 2 ≤ A.extent "D") (hO : 48 ≤ A.extent "O")
    (swapped hasSeed : Bool) : ∃ v : BlockState,
    exec (A.flattenKernel
      (RealForwardFlat.source swapped p "S" "D" "O" "O" hasSeed .fp16 .bf16 .fp16 1).eraseDType.toAlgKernel)
      (A.flattenState u) = some v ∧
    v.readMem A.flat (A.addr "O" 10) = (if hasSeed then 3 else 0) ∧
    v.readMem A.flat (A.addr "O" 26) = (if hasSeed then 7 else 1) ∧
    v.readMem A.flat (A.addr "O" 42) = (if hasSeed then 16 else 4) ∧
    v.mem A.flat (A.addr "O" 9) = MemCell.of .nat 77 := by
  have hb : RealForwardSafety.Bounds p (launch u) "S" "D" "O" "O" hasSeed A.extent := by
    apply RealForwardBounds.sufficient
    · decide
    · decide
    · intro _; exact hS
    · intro _; exact hD
    · exact hO
    · intro _; exact le_trans (by decide : 16 ≤ 48) hO
  obtain ⟨v,he,ho,hf⟩ := RealForwardFlat.source_executes A hd hcov p "S" "D" "O" "O" a
    swapped hasSeed .fp16 .bf16 .fp16 1 u (by decide) (by decide) (by decide)
    (by decide) (by decide) source_values gate_values (seed_values hasSeed) hb
  refine ⟨v,he,?_,?_,?_,?_⟩
  · exact (ho 0 0 0 (by decide)).trans (value_zero hasSeed)
  · exact (ho 1 0 0 (by decide)).trans (value_one hasSeed)
  · exact (ho 2 0 0 (by decide)).trans (value_two hasSeed)
  · have h := hf A.flat (A.addr "O" 9) (by
      rintro ⟨_,t,e,j,ht,heq⟩
      dsimp [FlatAlloc.addr,address,launch,u,p] at heq
      omega)
    rw [← h]
    have hr : ("O" : RegionName) ∈ A.regions := by
      by_contra hn
      rw [hcov "O" hn] at hO
      omega
    rw [A.flattenState_mem_addr hd u hr (show 9 < A.extent "O" by omega)]
    rfl

/-- The kernel must still store its last active tile, at address 47 here. -/
theorem short_output_rejected (hasSeed : Bool) :
    ¬ RealForwardSafety.Bounds p (launch u) "S" "D" "O" "O" hasSeed
      (fun r => if r = "O" then 47 else 48) := by
  intro h
  have hlast := h.output 2 1 1 (by decide)
  norm_num [p,launch,u,address] at hlast

/-- No loop iteration means no S/D read; an absent seed requires no allocation
either. The output tile still has to fit. -/
theorem single_chunk_bounds :
    RealForwardSafety.Bounds ⟨1,4,4,2,2⟩ (launch u) "S" "D" "O" "seed" Bool.false
      (fun r => if r = "O" then 16 else 0) := by
  apply RealForwardBounds.sufficient
  all_goals norm_num [launch,u]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Tests.ChunkGateForwardReal
