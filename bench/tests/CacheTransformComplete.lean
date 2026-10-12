import bench.tritonbench_g.cache_transform_triton.Correct
import bench.tritonbench_g.cache_transform_triton.FPEquiv
import bench.tritonbench_g.cache_transform_triton.Proofs.HostSafety

/-! Exercise the complete decoding IO contract on the second block of five
sequences, with two hidden lanes and mixed source dtypes. Check both outputs,
an inactive tagged cell, and the original unpadded-prefill allocation failure.
The successful run supplies actual input pins and bounds, not an assumed run. -/
namespace CacheTransformCompleteTests
open VeriTile.Triton VeriTile.Bench.TritonBenchG.CacheTransformTriton
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000

def p : Layout := ⟨"cos", "sin", ⟨"metadata"⟩, "cosOut", "sinOut", 2, 1, 2, 4⟩
def allocation : FlatAlloc where
  flat := "flat"
  regions := ["cos", "sin", "metadata", "cosOut", "sinOut"]
  base := fun r => if r = "cos" then 0 else if r = "sin" then 32 else
    if r = "metadata" then 64 else if r = "cosOut" then 96 else 128
  extent := fun r => if r ∈ (["cos", "sin", "metadata", "cosOut", "sinOut"] : List RegionName) then 16 else 0

def initial : BlockState where
  mem := fun r o => if r = "metadata" then MemCell.of .int 2 else
    if r = "cos" then MemCell.real (100 + o) else
    if r = "sin" then MemCell.real (200 + o) else MemCell.of .bf16 (some 9)
  regs := fun _ _ _ => none
  pids := fun _ => 1
  numPids := fun _ => 2
  undef := fun _ _ => 0

def inputs : DecodeIO.Inputs 2 5 where
  rows := fun _ => 2
  cos := fun r i => 100 + (r * 2 + i.val : Nat)
  sin := fun r i => 200 + (r * 2 + i.val : Nat)

theorem allocation_disjoint : allocation.Disjoint := by simp +decide [FlatAlloc.Disjoint]
theorem allocation_covers (r : RegionName) (hr : r ∉ allocation.regions) : allocation.extent r = 0 := by
  simp_all [allocation]

theorem bounds (commuted : Bool) :
    (DecodeIO.sourceIO p .fp16 .bf16 commuted 5).Bounds allocation.extent 5 (initial.pids 0) inputs := by
  constructor
  · intro i ha
    simp +decide [DecodeIO.sourceIO, DecodeIO.token, allocation, initial, p, Region.cast] at ha ⊢
    omega
  · intro i ha
    norm_num [DecodeIO.sourceIO, allocation, inputs, p]
    have := i.2.1.isLt
    change i.2.1.val < 2 at this
    omega
  · intro i ha
    norm_num [DecodeIO.sourceIO, allocation, inputs, p]
    have := i.2.1.isLt
    change i.2.1.val < 2 at this
    omega
  all_goals
    intro i ha
    simp +decide [DecodeIO.sourceIO, DecodeIO.token, allocation, initial, p] at ha ⊢
    have := i.2.1.isLt
    change i.2.1.val < 2 at this
    omega

theorem pins (commuted : Bool) :
    (DecodeIO.sourceIO p .fp16 .bf16 commuted 5).Pins initial 5 inputs := by
  constructor
  · intro i ha; rfl
  all_goals
    intro i ha
    simp +decide [DecodeIO.sourceIO, DecodeIO.expected, initial, inputs, p, BlockState.readMem]
    norm_num

theorem complete_flat_source (commuted : Bool) :
    ∃ final, exec (allocation.flattenKernel (p.decode .fp16 .bf16 commuted 5).eraseDType.toAlgKernel)
        (allocation.flattenState initial) = some final ∧
      final.readMem "flat" 104 = 104 ∧ final.readMem "flat" 137 = 205 ∧
      final.mem "flat" 106 = MemCell.of .bf16 (some 9) := by
  have hLayout : Function.Injective (fun i : TileIndex [p.B,p.H] =>
      i.1.val * p.rowStride + i.2.1.val * p.hiddenStride) := by
    intro a b h
    rcases a with ⟨i,j,⟨⟩⟩
    rcases b with ⟨i',j',⟨⟩⟩
    have hj := j.isLt
    have hj' := j'.isLt
    simp only [p, Nat.mul_one] at h hj hj'
    have hi : i = i' := Fin.ext (by omega)
    have hjj : j = j' := Fin.ext (by omega)
    subst hi; subst hjj; rfl
  obtain ⟨final, he, hv, hf⟩ := DecodeIO.implements p .fp16 .bf16 commuted 5 (by decide) hLayout
    allocation allocation_disjoint rfl allocation_covers initial inputs
    (by intros; norm_num [inputs]) rfl (bounds commuted) (pins commuted)
  refine ⟨final, he, ?_, ?_, ?_⟩
  · have h := (hv (⟨0,by decide⟩,⟨0,by decide⟩,PUnit.unit) (by decide)).1
    simp +decide [DecodeIO.sourceIO, DecodeIO.token, DecodeIO.expected, inputs, p, initial, allocation,
      FlatAlloc.addr] at h
    norm_num at h
    exact h
  · have h := (hv (⟨0,by decide⟩,⟨1,by decide⟩,PUnit.unit) (by decide)).2
    simp +decide [DecodeIO.sourceIO, DecodeIO.token, DecodeIO.expected, inputs, p, initial, allocation,
      FlatAlloc.addr] at h
    norm_num at h
    exact h
  · have h := hf "flat" 106 (Or.inr (by
      intro i ha
      have hj := i.2.1.isLt
      simp +decide [DecodeIO.sourceIO, DecodeIO.token, p, initial, allocation, FlatAlloc.addr] at ha hj ⊢
      constructor <;> omega))
    exact h.trans rfl

theorem three_entries_do_not_support_four_loads (s : BlockState) (commuted : Bool) :
    ¬ (RealSource.prefill "cos" "sin" ⟨"metadata"⟩ "cosOut" "sinOut" 2 1 6 2 4 16 commuted).TraceSafe
      (fun r => if r = "metadata" then 3 else 16) s := by
  exact HostSafety.prefill_unpadded_metadata_safe _ _ _ _ _ _ _ _ _ _ _ _ commuted s (by decide)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end CacheTransformCompleteTests
