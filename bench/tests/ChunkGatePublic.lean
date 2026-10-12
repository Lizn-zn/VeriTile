import bench.tritonbench_g.chunk_gate_recurrence.Correct
import bench.tritonbench_g.chunk_gate_recurrence.FPEquiv

/-! Exercise the reader-facing contracts at the empty-loop boundary. The
forward seed aliases its output; backward allocates no DI/DG region. Both
source variants run through the public specifications with a nonzero initial
padding oracle, rather than through execution helpers. -/
namespace VeriTile.Tests.ChunkGatePublic
open VeriTile.Triton VeriTile.Bench.TritonBenchG.ChunkGateRecurrence RealProgram
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

private abbrev p : Layout := ⟨1,1,1,1,1⟩
private def allocation : FlatAlloc where
  flat := "flat"
  regions := ["S","D","DS"]
  base := fun r => if r = "S" then 10 else if r = "D" then 20 else 30
  extent := fun r => if r ∈ (["S","D","DS"] : List RegionName) then 1 else 0
private def initial : BlockState where
  mem := fun _ _ => .real 7
  regs := fun _ _ _ => none
  pids := fun _ => 0
  undef := fun _ _ => 9

private theorem disjoint : allocation.Disjoint := by
  intro r hr s hs hne
  simp [allocation] at hr hs
  rcases hr with rfl | rfl | rfl <;> rcases hs with rfl | rfl | rfl <;>
    (first | exact (hne rfl).elim | decide)

private theorem covered : ∀ r, r ∉ allocation.regions → allocation.extent r = 0 := by
  intro r h
  exact if_neg h


private def forwardIO (swapped hasSeed : Bool) :=
  if swapped then Correctness.sumLeftForwardIO p "Input" "D" "S" "S" .fp16 .bf16 .fp32 1 hasSeed
  else Correctness.originalForwardIO p "Input" "D" "S" "S" .fp16 .bf16 .fp32 1 hasSeed

private theorem forward_bounds (swapped hasSeed : Bool) :
    (forwardIO swapped hasSeed).bounds initial allocation.extent := by
  cases swapped <;> change RealForwardSafety.Bounds p (launch initial) "Input" "D" "S" "S" hasSeed allocation.extent
  all_goals
    constructor
    · intro t e j ht; exact False.elim (Nat.not_lt_zero t ht)
    · intro t ht; exact False.elim (Nat.not_lt_zero t ht)
    · intro t e j ht
      have ht0 : t = 0 := by change t < 1 at ht; omega
      subst t
      fin_cases e; fin_cases j; decide
    · intro _ e j
      fin_cases e; fin_cases j; decide

theorem forward_boundary (swapped hasSeed : Bool) :
    ∃ v, exec (allocation.flattenKernel (forwardIO swapped hasSeed).kernel.toAlgKernel)
      (allocation.flattenState initial) = some v ∧
      v.readMem allocation.flat 10 = (if hasSeed then 7 else 0) := by
  have h : RealIO.ForwardIO.Implements (forwardIO swapped hasSeed) (Correctness.expectedForward p hasSeed) := by
    cases swapped
    · exact Correctness.forward_correctness p "Input" "D" "S" "S" .fp16 .bf16 .fp32 1 hasSeed
        (by decide) (by decide) (by decide) (by decide) (by decide)
    · exact Correctness.forward_sum_left_correctness p "Input" "D" "S" "S" .fp16 .bf16 .fp32 1 hasSeed
        (by decide) (by decide) (by decide) (by decide) (by decide)
  obtain ⟨v,he,ho,_⟩ := h allocation disjoint covered initial (forward_bounds swapped hasSeed)
  refine ⟨v,he,?_⟩
  have hzero := ho 0 0 0 (by decide)
  cases swapped <;> cases hasSeed <;>
    simpa [forwardIO,Correctness.originalForwardIO,Correctness.sumLeftForwardIO,Correctness.expectedForward,
      p,FlatAlloc.addr,allocation,address,seedAddress,launch,initial,BlockState.readMem,String.reduceEq] using hzero

private def backwardIO (swapped : Bool) :=
  if swapped then Correctness.sumLeftBackwardIO p "Input" "D" "DI" "DG" "S" "DS" .fp16 .bf16 .fp32 1
  else Correctness.originalBackwardIO p "Input" "D" "DI" "DG" "S" "DS" .fp16 .bf16 .fp32 1

private theorem backward_bounds (swapped : Bool) :
    (backwardIO swapped).bounds initial allocation.extent := by
  cases swapped <;> change RealBackwardSafety.Bounds p (launch initial) "Input" "D" "DI" "DG" "S" "DS" allocation.extent
  all_goals
    constructor
    · intro t e j ht; exact False.elim (Nat.not_lt_zero t ht)
    · intro t e j ht
      have ht0 : t = 0 := by change t < 1 at ht; omega
      subst t
      fin_cases e; fin_cases j; decide
    · intro t ht
      have ht0 : t = 0 := by change t < 1 at ht; omega
      subst t; decide
    · intro t e j ht; exact False.elim (Nat.not_lt_zero t ht)
    · intro t ht; exact False.elim (Nat.not_lt_zero t ht)
    · intro e j
      fin_cases e; fin_cases j; decide

theorem backward_boundary (swapped : Bool) :
    ∃ v, exec (allocation.flattenKernel (backwardIO swapped).kernel.toAlgKernel)
      (allocation.flattenState initial) = some v ∧ v.readMem allocation.flat 10 = 7 := by
  have h : RealIO.BackwardIO.Implements (backwardIO swapped) (Correctness.expectedBackward p) := by
    cases swapped
    · exact Correctness.backward_correctness p "Input" "D" "DI" "DG" "S" "DS" .fp16 .bf16 .fp32 1
        (by decide) (by decide) (by decide) (by decide) (by decide)
        (by constructor <;> decide) (by decide) (by decide)
    · exact Correctness.backward_sum_left_correctness p "Input" "D" "DI" "DG" "S" "DS" .fp16 .bf16 .fp32 1
        (by decide) (by decide) (by decide) (by decide) (by decide)
        (by constructor <;> decide) (by decide) (by decide)
  obtain ⟨v,he,_,_,hl,_⟩ := h allocation disjoint covered initial (backward_bounds swapped)
  refine ⟨v,he,?_⟩
  have hzero := hl 0 0
  cases swapped <;>
    simpa [backwardIO,Correctness.originalBackwardIO,Correctness.sumLeftBackwardIO,Correctness.expectedBackward,
      p,FlatAlloc.addr,allocation,address,seedAddress,launch,initial,BlockState.readMem,String.reduceEq] using hzero

/-- info: FP assumptions used by forward_equiv:
---
info:   add_commute -/
#guard_msgs in
#print_fp_assumptions Equivalence.forward_equiv

/-- info: FP assumptions used by backward_equiv:
---
info:   add_commute -/
#guard_msgs in
#print_fp_assumptions Equivalence.backward_equiv

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Tests.ChunkGatePublic
