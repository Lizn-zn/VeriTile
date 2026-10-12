import bench.tritonbench_g.bmm_chunk_bwd.Proofs.RealMemory
import bench.tritonbench_g.bmm_chunk_bwd.Proofs.RealSafety

/-! Nonzero full-source execution with all three program IDs nonzero, a
length-three final chunk, a masked output column/row, and both residual arms.
The result is obtained from initial memory through the proved source execution;
inactive tagged fp16 cells remain untouched. -/
namespace BmmChunkBwdCompleteTests
open VeriTile.Triton VeriTile.Bench.TritonBenchG.BmmChunkBwd
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

private def parameters (residual : Bool) : RealProgram.Parameters where
  A := "A"
  Dout := "Dout"
  Db := "Db"
  Res := "Res"
  seqlen := 7
  chunk := 4
  K := 3
  groups := 2
  aBatch := 100
  aRow := 1
  aHead := 20
  aColumn := 10
  dBatch := 1000
  dChunk := 100
  dHead := 20
  dRow := 1
  dColumn := 4
  oBatch := 1000
  oRow := 4
  oHead := 100
  oColumn := 1
  rBatch := 100
  rRow := 4
  rHead := 10
  rColumn := 1
  hasResidual := residual

private def initial : BlockState where
  mem := fun r a =>
    if r = "Dout" then MemCell.real (
      if a = 1120 then 2 else if a = 1121 then 3 else if a = 1122 then 5
      else if a = 1124 then 7 else if a = 1125 then 11 else if a = 1126 then 13 else 1000)
    else if r = "A" then MemCell.real (
      if a = 144 then 7 else if a = 145 then 11 else if a = 146 then 13 else 1000)
    else if r = "Res" then MemCell.real (if a = 128 then 17 else if a = 132 then 19 else 1000)
    else MemCell.of .fp16 (some 9)
  undef := fun _ _ => 0
  regs := fun _ _ _ => none
  pids := fun pidAxis => if pidAxis = 0 then 1 else if pidAxis = 1 then 1 else 3
  numPids := fun _ => 4

private theorem addresses_distinct (residual : Bool) :
    ∀ a b, RealMemory.active initial 4 2 (parameters residual) a →
      RealMemory.active initial 4 2 (parameters residual) b →
      RealMemory.address initial 4 2 (parameters residual) 1000 4 100 1 a =
        RealMemory.address initial 4 2 (parameters residual) 1000 4 100 1 b → a = b := by
  rintro ⟨i,j,⟨⟩⟩ ⟨i',j',⟨⟩⟩ ha hb he
  simp only [RealMemory.active, RealMemory.address, RealMemory.base, RealResidual.baseValue,
    RealSetup.rows, RealSetup.columns, RealSetup.tileM, RealSetup.tileN, RealSetup.tilesN,
    RealSetup.limit, RealSetup.chunk, RealSetup.head, RealMath.chunkLimit, initial, parameters] at ha hb he
  simp only [Nat.reduceAdd, Nat.reduceMul, Nat.reduceDiv, Nat.reduceMod, Nat.reduceSub,
    Nat.reduceLT, Nat.reduceBEq, ite_true, ite_false, Nat.min_self] at ha hb he
  have hj : j = j' := Fin.ext (by omega)
  have hi : i = i' := Fin.ext (by omega)
  cases hi; cases hj; rfl

theorem complete_source_with_residual (swapped residual : Bool) :
    ∃ t, exec (RealProgram.kernel swapped 4 2 2 (parameters residual)).toAlgKernel initial = some t ∧
      t.mem "Db" 1118 = MemCell.real (if residual then 129 else 112) ∧
      t.mem "Db" 1122 = MemCell.real (if residual then 358 else 339) ∧
      t.mem "Db" 1119 = MemCell.of .fp16 (some 9) ∧
      t.mem "Db" 1130 = MemCell.of .fp16 (some 9) ∧
      (∀ (r : RegionName) a, r ≠ "Db" → t.mem r a = initial.mem r a) := by
  obtain ⟨t, he, hv, hf⟩ := RealMemory.executes swapped initial 4 2 2 (parameters residual)
    (by decide) (addresses_distinct residual)
  refine ⟨t, he, ?_, ?_, ?_, ?_, ?_⟩
  · have h := hv (0,0,PUnit.unit) (by norm_num [RealMemory.active, RealSetup.rows, RealSetup.columns, RealSetup.tileM, RealSetup.tileN, RealSetup.tilesN, RealSetup.limit, RealSetup.chunk, RealMath.chunkLimit, parameters, initial])
    cases residual <;> norm_num [RealMemory.formula, RealMemory.address, RealMemory.base, RealResidual.baseValue,
      RealLoop.product, RealSetup.rows, RealSetup.columns, RealSetup.tileM, RealSetup.tileN,
      RealSetup.tilesN, RealSetup.limit, RealMath.chunkLimit, RealSetup.chunk, RealSetup.head,
      RealSetup.dBase, RealSetup.aBase, initial, parameters, BlockState.readMem, Finset.sum_range_succ]
      at h ⊢
    all_goals
      simp only [String.reduceEq, if_false] at h
      norm_num at h
      exact h
  · have h := hv (1,0,PUnit.unit) (by norm_num [RealMemory.active, RealSetup.rows, RealSetup.columns, RealSetup.tileM, RealSetup.tileN, RealSetup.tilesN, RealSetup.tilesN, RealSetup.limit, RealSetup.chunk, RealMath.chunkLimit, parameters, initial])
    cases residual <;> norm_num [RealMemory.formula, RealMemory.address, RealMemory.base, RealResidual.baseValue,
      RealLoop.product, RealSetup.rows, RealSetup.columns, RealSetup.tileM, RealSetup.tileN,
      RealSetup.tilesN, RealSetup.limit, RealMath.chunkLimit, RealSetup.chunk, RealSetup.head,
      RealSetup.dBase, RealSetup.aBase, initial, parameters, BlockState.readMem, Finset.sum_range_succ]
      at h ⊢
    all_goals
      simp only [String.reduceEq, if_false] at h
      norm_num at h
      exact h
  · rw [hf "Db" 1119]
    · rfl
    · rintro ⟨i,j,⟨⟩⟩ ha he
      simp [RealMemory.active, RealMemory.address, RealMemory.base, RealResidual.baseValue,
        RealSetup.rows, RealSetup.columns, RealSetup.tileM, RealSetup.tileN, RealSetup.tilesN,
        RealSetup.limit, RealSetup.chunk, RealSetup.head, RealMath.chunkLimit, initial, parameters] at ha he
      omega
  · rw [hf "Db" 1130]
    · rfl
    · rintro ⟨i,j,⟨⟩⟩ ha he
      simp [RealMemory.active, RealMemory.address, RealMemory.base, RealResidual.baseValue,
        RealSetup.rows, RealSetup.columns, RealSetup.tileM, RealSetup.tileN, RealSetup.tilesN,
        RealSetup.limit, RealSetup.chunk, RealSetup.head, RealMath.chunkLimit, initial, parameters] at ha he
      omega
  · intro r a hr
    exact hf r a (fun _ _ he => hr he.1.symm)


private def allocation : FlatAlloc where
  flat := "flat"
  regions := ["Dout", "A", "Res", "Db"]
  base := fun r => if r = "Dout" then 0 else if r = "A" then 2048 else if r = "Res" then 4096 else 6144
  extent := fun r => if r ∈ (["Dout", "A", "Res", "Db"] : List RegionName) then 2048 else 0

private theorem disjoint : allocation.Disjoint := by
  simp +decide [FlatAlloc.Disjoint]

private theorem covers (r : RegionName) (hr : r ∉ allocation.regions) : allocation.extent r = 0 := by
  simp_all [allocation]

private theorem address_bounds (residual : Bool) :
    RealSafety.Bounds initial 4 2 (parameters residual) allocation.extent := by
  refine ⟨⟨?_, ?_⟩, ?_, ?_⟩
  · intro i k hi hk
    norm_num [RealSetup.rows, RealSetup.columns, RealSetup.tileM, RealSetup.tileN,
      RealSetup.tilesN, RealSetup.limit, RealSetup.chunk, RealSetup.head, RealSetup.dBase,
      RealMath.chunkLimit, initial, parameters, allocation] at hi hk ⊢
    omega
  · intro k j hk hj
    norm_num [RealSetup.rows, RealSetup.columns, RealSetup.tileM, RealSetup.tileN,
      RealSetup.tilesN, RealSetup.limit, RealSetup.chunk, RealSetup.head, RealSetup.aBase,
      RealMath.chunkLimit, initial, parameters, allocation] at hk hj ⊢
    omega
  · intro _ i hi
    norm_num [RealMemory.active, RealMemory.address, RealMemory.base, RealResidual.baseValue,
      RealSetup.rows, RealSetup.columns, RealSetup.tileM, RealSetup.tileN, RealSetup.tilesN,
      RealSetup.limit, RealSetup.chunk, RealSetup.head, RealMath.chunkLimit, initial, parameters, allocation] at hi ⊢
    omega
  · intro i hi
    norm_num [RealMemory.active, RealMemory.address, RealMemory.base, RealResidual.baseValue,
      RealSetup.rows, RealSetup.columns, RealSetup.tileM, RealSetup.tileN, RealSetup.tilesN,
      RealSetup.limit, RealSetup.chunk, RealSetup.head, RealMath.chunkLimit, initial, parameters, allocation] at hi ⊢
    omega

/-- The same nonzero source executions survive flattening. This checks the
real bridge, three-axis windows and preservation of both masked tagged cells. -/
theorem complete_flat_source (swapped residual : Bool) :
    ∃ t, exec (allocation.flattenKernel (RealProgram.kernel swapped 4 2 2 (parameters residual)).toAlgKernel)
        (allocation.flattenState initial) = some t ∧
      t.mem "flat" 7262 = MemCell.real (if residual then 129 else 112) ∧
      t.mem "flat" 7266 = MemCell.real (if residual then 358 else 339) ∧
      t.mem "flat" 7263 = MemCell.of .fp16 (some 9) ∧
      t.mem "flat" 7274 = MemCell.of .fp16 (some 9) := by
  obtain ⟨t, he, h0, h1, hc, hr, _⟩ := complete_source_with_residual swapped residual
  refine ⟨allocation.flattenState t, ?_, ?_, ?_, ?_, ?_⟩
  · rw [FlatAlloc.exec_flatten allocation disjoint covers _ initial
      (RealSafety.safe swapped initial 4 2 2 (parameters residual) allocation.extent (address_bounds residual))
      (RealSafety.flatten_ok swapped 4 2 2 (parameters residual)) rfl, he, Option.map_some]
  all_goals
    have hf (a : Nat) (ha : a < 2048) :
        (allocation.flattenState t).mem "flat" (6144 + a) = allocation.trCell (t.mem "Db" a) :=
      FlatAlloc.flattenState_mem_addr allocation disjoint t (r := "Db") (o := a) (by simp [allocation]) (by simpa [allocation] using ha)
    first
    | exact (hf 1118 (by decide)).trans (by rw [h0]; rfl)
    | exact (hf 1122 (by decide)).trans (by rw [h1]; rfl)
    | exact (hf 1119 (by decide)).trans (by rw [hc]; rfl)
    | exact (hf 1130 (by decide)).trans (by rw [hr]; rfl)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end BmmChunkBwdCompleteTests
