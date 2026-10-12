import bench.tritonbench_g.bgmv_shrink_kernel.Proofs.RealIO
import bench.tritonbench_g.bgmv_shrink_kernel.Proofs.RealExecution

/-! A nonempty complete IO contract on four disjoint concrete allocations.
Both output branches, both source versions, all represented precisions, the
signed sentinel and masked K/N tails use the actual complete flat program. -/
namespace VeriTile.Bench.TritonBenchG.BgmvShrinkKernel.RealIOExecution
open VeriTile.Triton RealSetupSource RealExecution
noncomputable section
set_option maxHeartbeats 2200000
set_option linter.unusedSimpArgs false

private def allocation : FlatAlloc where
  flat := "flat"
  regions := ["X", "W", "Y", "Index"]
  base := fun r => if r = "X" then 0 else if r = "W" then 32 else if r = "Y" then 128 else 160
  extent := fun r => if r = "X" then 16 else if r = "W" then 65 else if r = "Y" then 20 else if r = "Index" then 2 else 0

private def inputs (atomic : Bool) : RealIO.Inputs (parameters atomic) where
  vector := fun k => (k.val : ℝ)+1
  weights := fun n k => 9*(n.val : ℝ)+(k.val : ℝ)+1
  initialOutput := fun n => 113+2*(n.val : ℝ)

private theorem disjoint : allocation.Disjoint := by
  simp +decide [FlatAlloc.Disjoint]

private theorem cover (r : RegionName) (h : r ∉ allocation.regions) : allocation.extent r = 0 := by
  simp_all [allocation]

private theorem read_input (atomic : Bool) (o : Nat) : (initial atomic 1).readMem "X" o = (o : ℝ)-8 := rfl
private theorem read_weight (atomic : Bool) (o : Nat) : (initial atomic 1).readMem "W" o = (o : ℝ)-39 := rfl
private theorem read_output (atomic : Bool) (o : Nat) : (initial atomic 1).readMem "Y" o = 100+(o : ℝ) := rfl

private theorem bounds (storage : TypedSource.Storage) (out_dtype : FloatDType) (swapped atomic : Bool) :
    (RealIO.sourceIO (parameters atomic) storage out_dtype swapped).Bounds
      allocation.extent (if atomic then 1 else 0) 1 1 := by
  constructor
  · change 1 < 2; decide
  · intro _ k _
    change 9+k.val < 16
    have hk : k.val < 7 := k.isLt
    omega
  · intro _ n k _
    change 40+n.val*9+k.val*1 < 65
    have hn : n.val < 3 := n.isLt
    have hk : k.val < 7 := k.isLt
    omega
  · intro _ n
    change 13+n.val*2 < 20
    have hn : n.val < 3 := n.isLt
    omega

private theorem pins (storage : TypedSource.Storage) (out_dtype : FloatDType) (swapped atomic : Bool) :
    (RealIO.sourceIO (parameters atomic) storage out_dtype swapped).Pins
      (initial atomic 1) 1 (inputs atomic) := by
  constructor
  · rfl
  · intro _ k _
    change (initial atomic 1).readMem "X" (9+k.val) = (k.val : ℝ)+1
    rw [read_input]
    push_cast
    ring
  · intro _ n k _
    change (initial atomic 1).readMem "W" (40+n.val*9+k.val*1) = 9*(n.val : ℝ)+(k.val : ℝ)+1
    rw [read_weight]
    push_cast
    ring
  · intro _ _ n
    change (initial atomic 1).readMem "Y" (13+n.val*2) = 113+2*(n.val : ℝ)
    rw [read_output]
    push_cast
    ring

private theorem values (atomic : Bool) (n : Fin 3) :
    RealIO.expected (parameters atomic) (if atomic then 1 else 0) (inputs atomic) n =
      if atomic then 335+380*(n.val : ℝ) else 420+756*(n.val : ℝ) := by
  dsimp only [RealIO.expected, inputs, parameters, RealIO.rank, RealLoop.numKIters]
  cases atomic <;> fin_cases n <;>
    norm_num [Fin.sum_univ_two, Finset.sum_range_succ]

/-- The actual IO theorem executes from allocated independent inputs and
writes only flat[141], flat[143], flat[145]. Masked padding stays unchanged;
the masked K lane can exceed the input/weight allocation without being read. -/
theorem executes (storage : TypedSource.Storage) (out_dtype : FloatDType) (swapped atomic : Bool) :
    ∃ final, exec (allocation.flattenKernel
        ((parameters atomic).source storage out_dtype swapped).eraseDType.toAlgKernel)
        (allocation.flattenState (initial atomic 1)) = some final ∧
      (∀ n : Fin 3, final.readMem "flat" (141+n.val*2) =
        if atomic then 335+380*(n.val : ℝ) else 420+756*(n.val : ℝ)) ∧
      (∀ (r : RegionName) o, (r ≠ "flat" ∨ (o ≠ 141 ∧ o ≠ 143 ∧ o ≠ 145)) →
        final.mem r o = (allocation.flattenState (initial atomic 1)).mem r o) := by
  obtain ⟨final, he, hv, hf⟩ := RealIO.implements (parameters atomic) storage out_dtype swapped
    (by cases atomic <;> decide) (by norm_num [parameters]) (by norm_num [parameters])
    allocation disjoint rfl cover (initial atomic 1) 1 (inputs atomic) (Or.inr (by decide)) rfl
    (bounds storage out_dtype swapped atomic) (pins storage out_dtype swapped atomic)
  refine ⟨final, he, ?_, ?_⟩
  · intro n
    have h := (hv n (by decide)).trans (values atomic n)
    convert h using 1
    simp [RealIO.sourceIO, parameters, initial, FlatAlloc.addr, allocation, ← Nat.add_assoc]
  · intro r o hm
    apply hf
    rcases hm with hr | ho
    · exact Or.inl hr
    · right
      intro n _ heq
      have hn : n.val < 3 := n.isLt
      have heq' : o = 141+n.val*2 := by
        simpa [RealIO.sourceIO, parameters, initial, FlatAlloc.addr, allocation, ← Nat.add_assoc] using heq
      omega

theorem sentinel_skips (storage : TypedSource.Storage) (out_dtype : FloatDType) (swapped atomic : Bool) :
    ∃ final, exec (allocation.flattenKernel
        ((parameters atomic).source storage out_dtype swapped).eraseDType.toAlgKernel)
        (allocation.flattenState (initial atomic (-1))) = some final ∧
      final.mem = (allocation.flattenState (initial atomic (-1))).mem := by
  exact RealFlat.sentinel_skips allocation disjoint cover (parameters atomic) storage out_dtype
    swapped (initial atomic (-1)) rfl rfl (by change 1 < 2; decide)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BgmvShrinkKernel.RealIOExecution
