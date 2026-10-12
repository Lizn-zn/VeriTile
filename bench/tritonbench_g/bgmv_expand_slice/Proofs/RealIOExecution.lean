import bench.tritonbench_g.bgmv_expand_slice.Proofs.RealIO
import bench.tritonbench_g.bgmv_expand_slice.Proofs.RealExecution

/-! Nonempty witness of the complete IO contract on four concrete disjoint
allocations. All type/cast/load choices and both versions share this execution.
The vector's padded lane is in bounds and nonzero; masked weight padding
extends beyond the weight allocation and is correctly not required in bounds. -/
namespace VeriTile.Bench.TritonBenchG.BgmvExpandSlice.RealIOExecution
open VeriTile.Triton RealSetupSource RealExecution
noncomputable section
set_option maxHeartbeats 2200000
set_option linter.unusedSimpArgs false

private def allocation : FlatAlloc where
  flat := "flat"
  regions := ["X", "W", "Y", "I"]
  base := fun r => if r = "X" then 0 else if r = "W" then 16 else if r = "Y" then 80 else 104
  extent := fun r => if r = "X" then 12 else if r = "W" then 45 else if r = "Y" then 21 else if r = "I" then 2 else 0

private def inputs : RealIO.Inputs parameters where
  vector := fun k => match k.val with | 0 => 2 | 1 => -1 | 2 => 3 | _ => 99
  weights := fun g k => 5+7*(g.val : ℝ)+2*(k.val : ℝ)
  initialOutput := fun g => 111+3*(g.val : ℝ)

private theorem disjoint : allocation.Disjoint := by
  simp +decide [FlatAlloc.Disjoint]

private theorem cover (r : RegionName) (h : r ∉ allocation.regions) : allocation.extent r = 0 := by
  simp_all [allocation]

private theorem read_weight (o : Nat) : (initial 1).readMem "W" o = (o : ℝ)-21 := rfl
private theorem read_output (o : Nat) : (initial 1).readMem "Y" o = 100+(o : ℝ) := rfl

private theorem bounds (input : TypedSource.InputStorage) (weight : TypedSource.WeightStorage)
    (out_dtype : FloatDType) (cast swapped even add : Bool) :
    (RealIO.sourceIO parameters input weight out_dtype cast swapped even add).Bounds
      allocation.extent 1 1 1 := by
  constructor
  · change 1 < 2; decide
  · intro _ k _
    change 5+k.val*2 < 12
    have hk : k.val < 4 := k.isLt
    omega
  · intro _ g k
    change 26+g.val*7+k.val*2 < 45
    have hg : g.val < 3 := g.isLt
    have hk : k.val < 3 := k.isLt
    omega
  · intro _ g
    change 11+g.val*3 < 21
    have hg : g.val < 3 := g.isLt
    omega

private theorem pins (input : TypedSource.InputStorage) (weight : TypedSource.WeightStorage)
    (out_dtype : FloatDType) (cast swapped even add : Bool) :
    (RealIO.sourceIO parameters input weight out_dtype cast swapped even add).Pins (initial 1) 1 inputs := by
  constructor
  · rfl
  · intro _ k _
    fin_cases k <;> rfl
  · intro _ g k
    change (initial 1).readMem "W" (26+g.val*7+k.val*2) = 5+7*(g.val : ℝ)+2*(k.val : ℝ)
    rw [read_weight]
    push_cast
    ring
  · intro _ _ g
    change (initial 1).readMem "Y" (11+g.val*3) = 111+3*(g.val : ℝ)
    rw [read_output]
    push_cast
    ring

private theorem values (add : Bool) (g : Fin 3) :
    RealIO.expected parameters (by decide) add inputs g =
      if add then 141+31*(g.val : ℝ) else 30+28*(g.val : ℝ) := by
  change (let v : ℝ := ∑ k : Fin 3,
      (match k.val with | 0 => 2 | 1 => -1 | 2 => 3 | _ => 99) *
        (5+7*(g.val : ℝ)+2*(k.val : ℝ))
    if add then v+(111+3*(g.val : ℝ)) else v) = _
  cases add <;> fin_cases g <;>
    norm_num [Fin.sum_univ_succ]

/-- This instantiates the complete independent-input IO theorem, including
metadata, disjoint allocations, all active bounds, initial values, flat
execution and the full frame. Only flat[91], flat[94], flat[97] may change. -/
theorem executes (input : TypedSource.InputStorage) (weight : TypedSource.WeightStorage)
    (out_dtype : FloatDType) (cast swapped even add : Bool) :
    ∃ final, exec (allocation.flattenKernel
        (parameters.source input weight out_dtype cast swapped even add).eraseDType.toAlgKernel)
        (allocation.flattenState (initial 1)) = some final ∧
      (∀ g : Fin 3, final.readMem "flat" (91+g.val*3) =
        if add then 141+31*(g.val : ℝ) else 30+28*(g.val : ℝ)) ∧
      (∀ (r : RegionName) o, (r ≠ "flat" ∨ (o ≠ 91 ∧ o ≠ 94 ∧ o ≠ 97)) →
        final.mem r o = (allocation.flattenState (initial 1)).mem r o) := by
  obtain ⟨final, he, hv, hf⟩ := RealIO.implements parameters input weight out_dtype cast swapped even add
    (by decide) (by decide) (by decide) (by decide) (by decide)
    allocation disjoint rfl cover (initial 1) 1 inputs (Or.inr (by decide)) rfl
    (bounds input weight out_dtype cast swapped even add) (pins input weight out_dtype cast swapped even add)
  refine ⟨final, he, ?_, ?_⟩
  · intro g
    have h := (hv g (by decide)).trans (values add g)
    convert h using 1
    simp [RealIO.sourceIO, parameters, Parameters.length, initial, FlatAlloc.addr, allocation, ← Nat.add_assoc]
  · intro r o hm
    apply hf
    rcases hm with hr | ho
    · exact Or.inl hr
    · right
      intro g _ heq
      have hg : g.val < 3 := g.isLt
      have heq' : o = 91+g.val*3 := by
        simpa [RealIO.sourceIO, parameters, Parameters.length, initial, FlatAlloc.addr, allocation, ← Nat.add_assoc] using heq
      omega

theorem sentinel_skips (input : TypedSource.InputStorage) (weight : TypedSource.WeightStorage)
    (out_dtype : FloatDType) (cast swapped even add : Bool) :
    ∃ final, exec (allocation.flattenKernel
        (parameters.source input weight out_dtype cast swapped even add).eraseDType.toAlgKernel)
        (allocation.flattenState (initial (-1))) = some final ∧
      final.mem = (allocation.flattenState (initial (-1))).mem := by
  exact RealFlat.sentinel_skips allocation disjoint cover parameters input weight out_dtype
    cast swapped even add (initial (-1)) rfl rfl (by decide)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BgmvExpandSlice.RealIOExecution
