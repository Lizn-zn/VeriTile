import bench.tritonbench_g.chunk_gate_recurrence.Proofs.RealBackwardCandidate
import bench.tritonbench_g.chunk_gate_recurrence.Proofs.RealBackwardBounds

/-! Complete backward real runs, with both source variants and mixed storage.
Three chunks exercise both reverse updates and the final DL step; S varies
across both tile axes so DG must use all four lanes. DL aliases S, while last
DI/DG chunks and a neighboring tagged cell remain untouched. N=1 exercises an
empty loop with no source or DI/DG access. -/
namespace VeriTile.Tests.ChunkGateBackwardReal
open VeriTile.Triton VeriTile.Bench.TritonBenchG.ChunkGateRecurrence RealProgram RealBackwardPointers
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false
open scoped BigOperators

private abbrev p : Layout := ⟨3,4,4,2,2⟩
private def u : BlockState where
  mem := fun r off =>
    if r = "S" then MemCell.real ((off / 16 + 1 + off % 16 / 4 + off % 4 : Nat) : ℝ)
    else if r = "D" then MemCell.real 2
    else if r = "DS" then MemCell.real ((off / 16 + 1 : Nat) : ℝ)
    else MemCell.of .nat 77
  regs := fun _ _ _ => none
  pids := fun _ => 0
  undef := fun _ _ => 7

private def a : Inputs p where
  source := fun c e j => ((c+1+e.val+j.val : Nat) : ℝ)
  gate := fun _ => 2
  seed := fun _ _ => 0
  incoming := fun c _ _ => ((c+1 : Nat) : ℝ)

private theorem source_values : ∀ (t : Nat) (e : Fin p.BK) (j : Fin p.BV), t < p.N-1 →
    u.readMem "S" (address p (launch u) t e.val j.val) = a.source t e j := by
  intro t e j ht
  change t < 2 at ht
  interval_cases t <;> fin_cases e <;> fin_cases j <;>
    simp [BlockState.readMem,u,address,launch,p,a,String.reduceEq]

private theorem incoming_values : ∀ (t : Nat) (e : Fin p.BK) (j : Fin p.BV), t < p.N →
    u.readMem "DS" (address p (launch u) t e.val j.val) = a.incoming t e j := by
  intro t e j ht
  change t < 3 at ht
  interval_cases t <;> fin_cases e <;> fin_cases j <;>
    simp [BlockState.readMem,u,address,launch,p,a,String.reduceEq]

private theorem gate_values : ∀ t, t < p.N → u.readMem "D" (gateAddress p (launch u) t) = a.gate t := by
  intro t _
  simp [BlockState.readMem,u,a,String.reduceEq]

private theorem suffix_two (e : Fin p.BK) (j : Fin p.BV) : RealMath.reverse a 3 2 e j = 3 := by
  rw [RealMath.reverse_step a 3 2 (by decide),RealMath.reverse_end]
  norm_num [a]

private theorem suffix_one (e : Fin p.BK) (j : Fin p.BV) : RealMath.reverse a 3 1 e j = 8 := by
  rw [RealMath.reverse_step a 3 1 (by decide),suffix_two]
  norm_num [a]

private theorem suffix_zero (e : Fin p.BK) (j : Fin p.BV) : RealMath.reverse a 3 0 e j = 17 := by
  rw [RealMath.reverse_step a 3 0 (by decide),suffix_one]
  norm_num [a]

private theorem gradient_zero : RealMath.gateGradient a 3 0 = 64 := by
  simp only [RealMath.gateGradient,show 0+1=1 from rfl,suffix_one]
  norm_num [p,a,Fin.sum_univ_succ]

private theorem gradient_one : RealMath.gateGradient a 3 1 = 36 := by
  simp only [RealMath.gateGradient,show 1+1=2 from rfl,suffix_two]
  norm_num [p,a,Fin.sum_univ_succ]

private def source (swapped : Bool) (N : Nat) :=
  if swapped then TypedSource.backwardSumLeft .fp16 .bf16 .fp32 "S" "D" "DI" "DG" "S" "DS" 1 N 4 4 2 2
  else TypedSource.backward .fp16 .bf16 .fp32 "S" "D" "DI" "DG" "S" "DS" 1 N 4 4 2 2

theorem complete_source (swapped : Bool) : ∃ v : BlockState,
    exec (source swapped 3).eraseDType.toAlgKernel u = some v ∧
    v.readMem "DI" 0 = 8 ∧ v.readMem "DI" 16 = 3 ∧
    v.readMem "DG" 0 = 64 ∧ v.readMem "DG" 4 = 36 ∧
    v.readMem "S" 0 = 17 ∧
    v.mem "DI" 32 = MemCell.of .nat 77 ∧
    v.mem "DG" 8 = MemCell.of .nat 77 ∧
    v.mem "DI" 2 = MemCell.of .nat 77 := by
  obtain ⟨v,he,hi,hg,hl,hf⟩ := RealBackward.original_executes p "S" "D" "DI" "DG" "S" "DS" a u
    .fp16 .bf16 .fp32 1 (by decide) (by decide) (by decide) (by decide) (by decide)
    (by constructor <;> decide) (by decide) (by decide) source_values incoming_values gate_values
  refine ⟨v,?_,(hi 0 0 0 (by decide)).trans (suffix_one 0 0),
    (hi 1 0 0 (by decide)).trans (suffix_two 0 0),
    (hg 0 (by decide)).trans gradient_zero,(hg 1 (by decide)).trans gradient_one,
    (hl 0 0).trans (suffix_zero 0 0),?_,?_,?_⟩
  · cases swapped
    · exact he
    · change exec (TypedSource.backwardSumLeft .fp16 .bf16 .fp32 "S" "D" "DI" "DG" "S" "DS" 1 3 4 4 2 2).eraseDType.toAlgKernel u = _
      rw [← RealBackwardCandidate.region_exec_equal p]
      exact he
  · have h := hf "DI" 32 (by
      simp only [RealBackward.footprint,WriteFootprint.union,RealBackwardLoop.footprint,
        RealBackwardPost.footprint,Region.mk.injEq,String.reduceEq,true_and,false_and,or_false]
      rintro ⟨t,e,j,ht,heq⟩
      dsimp [address,launch,u,p] at heq ht
      have he := e.isLt
      have hj := j.isLt
      omega)
    exact h.symm
  · have h := hf "DG" 8 (by
      simp only [RealBackward.footprint,WriteFootprint.union,RealBackwardLoop.footprint,
        RealBackwardPost.footprint,Region.mk.injEq,String.reduceEq,true_and,false_and,false_or,or_false]
      rintro ⟨t,ht,heq⟩
      dsimp [gradientAddress,numK,numV,launch,u,p] at heq ht
      omega)
    exact h.symm
  · have h := hf "DI" 2 (by
      simp only [RealBackward.footprint,WriteFootprint.union,RealBackwardLoop.footprint,
        RealBackwardPost.footprint,Region.mk.injEq,String.reduceEq,true_and,false_and,or_false]
      rintro ⟨t,e,j,ht,heq⟩
      dsimp [address,launch,u,p] at heq ht
      have he := e.isLt
      have hj := j.isLt
      omega)
    exact h.symm

/-- Both complete versions also execute when the reverse loop is empty. Every
DI/DG cell is unchanged, while DL still performs its chunk-zero update. -/
theorem single_chunk (swapped : Bool) : ∃ v : BlockState,
    exec (source swapped 1).eraseDType.toAlgKernel u = some v ∧
    v.readMem "S" 0 = 1 ∧
    (∀ off, v.mem "DI" off = u.mem "DI" off) ∧
    (∀ off, v.mem "DG" off = u.mem "DG" off) := by
  let q : Layout := ⟨1,4,4,2,2⟩
  obtain ⟨v,he,_,_,hl,hf⟩ := RealBackward.original_executes q "S" "D" "DI" "DG" "S" "DS" a u
    .fp16 .bf16 .fp32 1 (by decide) (by decide) (by decide) (by decide) (by decide)
    (by constructor <;> decide) (by decide) (by decide)
    (by intro t e j ht; exact False.elim (Nat.not_lt_zero t ht))
    (by
      intro t e j ht
      have ht0 : t = 0 := by change t < 1 at ht; omega
      subst t
      fin_cases e <;> fin_cases j <;>
        simp [BlockState.readMem,u,address,launch,q,a,String.reduceEq])
    (by intro t ht; simp [BlockState.readMem,u,a,String.reduceEq])
  refine ⟨v,?_,?_,?_,?_⟩
  · cases swapped
    · exact he
    · change exec (TypedSource.backwardSumLeft .fp16 .bf16 .fp32 "S" "D" "DI" "DG" "S" "DS" 1 1 4 4 2 2).eraseDType.toAlgKernel u = _
      rw [← RealBackwardCandidate.region_exec_equal q]
      exact he
  · simpa [q,seedAddress,launch,u,RealMath.reverse_single,a] using hl 0 0
  · intro off
    exact (hf "DI" off (by simp [RealBackward.footprint,WriteFootprint.union,RealBackwardLoop.footprint,
      RealBackwardPost.footprint,q,String.reduceEq])).symm
  · intro off
    exact (hf "DG" off (by simp [RealBackward.footprint,WriteFootprint.union,RealBackwardLoop.footprint,
      RealBackwardPost.footprint,q,String.reduceEq])).symm


theorem sufficient_bounds : RealBackwardSafety.Bounds p (launch u) "S" "D" "DI" "DG" "S" "DS"
    (fun r => if r = "D" then 3 else if r = "DG" then 8 else if r = "DS" then 48 else 32) := by
  apply RealBackwardBounds.sufficient
  all_goals norm_num [p,launch,u,numK,numV]
  all_goals simp [String.reduceEq]

theorem short_gradient_rejected : ¬ RealBackwardSafety.Bounds p (launch u) "S" "D" "DI" "DG" "S" "DS"
    (fun r => if r = "DG" then 4 else 48) := by
  intro h
  have hlast := h.outputDG 1 (by decide)
  norm_num [gradientAddress,numK,numV,p,launch,u] at hlast

theorem short_DI_rejected : ¬ RealBackwardSafety.Bounds p (launch u) "S" "D" "DI" "DG" "S" "DS"
    (fun r => if r = "DI" then 21 else 48) := by
  intro h
  have hlast := h.outputDI 1 1 1 (by decide)
  norm_num [address,p,launch,u] at hlast

/-- An empty reverse loop needs no S/DI/DG allocation; DS, D and DL remain live. -/
theorem single_chunk_bounds :
    RealBackwardSafety.Bounds ⟨1,4,4,2,2⟩ (launch u) "S" "D" "DI" "DG" "DL" "DS"
      (fun r => if r = "D" then 1 else if r = "DS" ∨ r = "DL" then 16 else 0) := by
  apply RealBackwardBounds.sufficient
  all_goals norm_num [launch,u,numK,numV]
  all_goals simp [String.reduceEq]

theorem single_gate_required : ¬
    RealBackwardSafety.Bounds ⟨1,4,4,2,2⟩ (launch u) "S" "D" "DI" "DG" "DL" "DS"
      (fun r => if r = "D" then 0 else 16) := by
  intro h
  have hzero := h.gate 0 (by decide)
  norm_num [gateAddress,launch,u] at hzero


/-- After the final loop decrement, dead S lanes need not equal the chunk-zero
pointer. This is why the flat bridge must ignore those register values. -/
theorem dead_pointer_not_chunk_zero :
    backwardPointer p (launch u) "S" (p.N-2) (p.N-1) ≠ pointer p (launch u) "S" 0 := by
  intro h
  have hv := congrArg (fun t : Tile .ptr [2,2] => t.data (1,1,PUnit.unit)) h
  norm_num [backwardPointer,pointer,address,p,launch,u] at hv

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Tests.ChunkGateBackwardReal
