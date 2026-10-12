import bench.tritonbench_g.bmm_optimized.Proofs.RealMemory
import bench.tritonbench_g.bmm_optimized.Proofs.RealSafety

/-! Complete real/flat execution with actual supported 32×32 dot tiles,
nonzero batch and both nonzero grid coordinates. The last short CTA group
selects row64/column32 of a 65×48 output, and K=33 exercises the second K
block. Both output tails and untouched tagged cells remain observable. -/
namespace BmmOptimizedCompleteTests
open VeriTile.Triton VeriTile.Bench.TritonBenchG.BmmOptimized
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false

def p : RealProgram.Parameters := ⟨"A","B","O",65,48,33,2,Bool.false,Bool.false,Bool.false⟩
def initial : BlockState where
  mem := fun r _ => if r = "A" ∨ r = "B" then MemCell.real 1 else MemCell.of .fp16 (some 9)
  regs := fun _ _ _ => none
  pids := fun ax => if ax = 0 then 2 else 1
  numPids := fun ax => if ax = 0 then 3 else 2
  undef := fun _ _ => 0

theorem host : RealMemory.Host 32 32 32 p := by constructor <;> norm_num [p]
theorem launch : RealMemory.Launch initial 32 32 p := by constructor <;> decide

theorem complete_source (swapped : Bool) :
    ∃ t, exec (RealProgram.kernel swapped 32 32 32 p).toAlgKernel initial = some t ∧
      t.mem "O" 6224 = MemCell.real 33 ∧ t.mem "O" 6239 = MemCell.real 33 ∧
      t.mem "O" 6240 = MemCell.of .fp16 (some 9) ∧
      t.mem "O" 6272 = MemCell.of .fp16 (some 9) ∧
      t.mem "O" 6223 = MemCell.of .fp16 (some 9) := by
  obtain ⟨t,he,hv,hf⟩ := RealMemory.executes swapped initial 32 32 32 p host launch (by intros; rfl)
  refine ⟨t,he,?_,?_,?_,?_,?_⟩
  · have h := hv (0,0,PUnit.unit) (by unfold RealMemory.active; decide)
    norm_num [RealMemory.address, RealMemory.formula, RealMath.contraction,
      RealLoop.aEntry, RealLoop.bEntry, RealSetup.aBase, RealSetup.bBase, RealSetup.oBase,
      RealSetup.rows, RealSetup.columns, RealSetup.tileM, RealSetup.tileN,
      RealGrid.rowTile, RealGrid.columnTile, RealGrid.groupSize, BlockState.readMem, initial, p] at h
    exact h
  · have h := hv (0,15,PUnit.unit) (by unfold RealMemory.active; decide)
    norm_num [RealMemory.address, RealMemory.formula, RealMath.contraction,
      RealLoop.aEntry, RealLoop.bEntry, RealSetup.aBase, RealSetup.bBase, RealSetup.oBase,
      RealSetup.rows, RealSetup.columns, RealSetup.tileM, RealSetup.tileN,
      RealGrid.rowTile, RealGrid.columnTile, RealGrid.groupSize, BlockState.readMem, initial, p] at h
    exact h
  all_goals
    rw [hf "O" _]
    · rfl
    · rintro ⟨i,j,⟨⟩⟩ hi he
      norm_num [RealMemory.active, RealMemory.address, RealSetup.oBase,
        RealSetup.rows, RealSetup.columns, RealSetup.tileM, RealSetup.tileN,
        RealGrid.rowTile, RealGrid.columnTile, RealGrid.groupSize, initial, p] at hi he
      omega

def allocation : FlatAlloc where
  flat := "flat"
  regions := ["A","B","O"]
  base := fun r => if r = "A" then 0 else if r = "B" then 32768 else 65536
  extent := fun r => if r ∈ (["A","B","O"] : List RegionName) then 16384 else 0

theorem disjoint : allocation.Disjoint := by simp +decide [FlatAlloc.Disjoint]
theorem covers (r : RegionName) (hr : r ∉ allocation.regions) : allocation.extent r = 0 := by
  simp_all [allocation]

theorem address_bounds : RealSafety.Bounds initial 32 32 p allocation.extent := by
  refine ⟨⟨?_,?_⟩,?_⟩
  · intro i k hi hk
    norm_num [RealMath.active, RealSetup.aBase, RealSetup.rows, RealSetup.tileM,
      RealGrid.rowTile, RealGrid.groupSize, initial, p, allocation] at hi hk ⊢
    omega
  · intro k j hk hj
    norm_num [RealMath.active, RealSetup.bBase, RealSetup.columns, RealSetup.tileN,
      RealGrid.columnTile, RealGrid.groupSize, initial, p, allocation] at hk hj ⊢
    omega
  · intro i j hi hj
    norm_num [RealMath.active, RealSetup.oBase, RealSetup.rows, RealSetup.columns, RealSetup.tileM, RealSetup.tileN,
      RealGrid.rowTile, RealGrid.columnTile, RealGrid.groupSize, initial, p, allocation] at hi hj ⊢
    omega

theorem complete_flat_source (swapped : Bool) :
    ∃ t, exec (allocation.flattenKernel (RealProgram.kernel swapped 32 32 32 p).toAlgKernel)
        (allocation.flattenState initial) = some t ∧
      t.mem "flat" 71760 = MemCell.real 33 ∧ t.mem "flat" 71775 = MemCell.real 33 ∧
      t.mem "flat" 71776 = MemCell.of .fp16 (some 9) ∧
      t.mem "flat" 71808 = MemCell.of .fp16 (some 9) ∧
      t.mem "flat" 71759 = MemCell.of .fp16 (some 9) := by
  obtain ⟨t,he,h0,h1,hn,hm,hp⟩ := complete_source swapped
  refine ⟨allocation.flattenState t,?_,?_,?_,?_,?_,?_⟩
  · rw [FlatAlloc.exec_flatten allocation disjoint covers _ initial
      (RealSafety.safe swapped initial 32 32 32 p host.kPositive host.divisibleK (by intros; rfl)
        allocation.extent address_bounds)
      (RealPureSafety.flatten_ok swapped 32 32 32 p) rfl, he, Option.map_some]
  all_goals
    have hf (a : Nat) (ha : a < 16384) :
        (allocation.flattenState t).mem "flat" (65536+a) = allocation.trCell (t.mem "O" a) :=
      FlatAlloc.flattenState_mem_addr allocation disjoint t (r := "O") (o := a)
        (by simp [allocation]) (by simpa [allocation] using ha)
    first
    | exact (hf 6224 (by decide)).trans (by rw [h0]; rfl)
    | exact (hf 6239 (by decide)).trans (by rw [h1]; rfl)
    | exact (hf 6240 (by decide)).trans (by rw [hn]; rfl)
    | exact (hf 6272 (by decide)).trans (by rw [hm]; rfl)
    | exact (hf 6223 (by decide)).trans (by rw [hp]; rfl)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end BmmOptimizedCompleteTests
