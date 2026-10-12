import bench.tritonbench_g.bmm_chunk_fwd.Proofs.RealMemory
import bench.tritonbench_g.bmm_chunk_fwd.Proofs.RealSafety

/-! Complete region and flat-memory executions at nonzero batch/chunk/head
coordinates, including partial K blocks, padded output zeros, signed sequence
filtering and causal tile skipping. Nonzero padding detects missing masks. -/
namespace BmmChunkFwdCompleteTests
open VeriTile.Triton VeriTile.Bench.TritonBenchG.BmmChunkFwd
noncomputable section
set_option linter.unusedTactic false
set_option linter.unusedSimpArgs false
set_option maxHeartbeats 2000000
set_option maxRecDepth 12000
private def parameters (causal hasSeq : Bool) : RealProgram.Parameters where
  A := "A"
  B := "B"
  Out := "Out"
  Seq := ⟨"Seq"⟩
  seqlen := 7
  chunk := 4
  K := 3
  groups := 2
  aBatch := 100
  aRow := 10
  aHead := 4
  aColumn := 1
  bBatch := 120
  bRow := 11
  bHead := 5
  bColumn := 1
  oBatch := 200
  oChunk := 24
  oHead := 16
  oRow := 4
  oColumn := 1
  sBatch := 90
  sRow := 1
  causal := causal
  hasSeq := hasSeq

private def initial : BlockState where
  mem := fun r o =>
    if r = "A" then MemCell.real (
      if o = 144 then 2 else if o = 145 then 3 else if o = 146 then 5
      else if o = 154 then 7 else if o = 155 then 11 else if o = 156 then 13 else 1000)
    else if r = "B" then MemCell.real (
      if o = 191 then 29 else if o = 192 then 31 else if o = 193 then 37 else 1000)
    else if r = "Seq" then MemCell.of .int (if o = 95 then -2 else -1)
    else MemCell.of .fp16 (some 9)
  undef := fun _ _ => 0
  regs := fun _ _ _ => none
  pids := fun ax => if ax = 2 then 3 else 1
  numPids := fun _ => 8

private theorem addresses_distinct (causal hasSeq : Bool) :
    ∀ a b, RealMemory.active initial 4 2 (parameters causal hasSeq) a →
      RealMemory.active initial 4 2 (parameters causal hasSeq) b →
      RealMemory.address initial 4 2 (parameters causal hasSeq) a =
        RealMemory.address initial 4 2 (parameters causal hasSeq) b → a = b := by
  rintro ⟨i,j,⟨⟩⟩ ⟨i',j',⟨⟩⟩ _ _ he
  norm_num [RealMemory.address, RealMemory.base, RealSetup.rows, RealSetup.columns,
    RealSetup.tileM, RealSetup.tileN, RealSetup.tilesN, RealSetup.chunk, RealSetup.head,
    initial, parameters] at he
  have hj : j = j' := Fin.ext (by omega)
  have hi : i = i' := Fin.ext (by omega)
  cases hi; cases hj; rfl

theorem complete_source (swapped causal hasSeq : Bool) :
    ∃ t, exec (RealProgram.kernel swapped 4 2 2 (parameters causal hasSeq)).toAlgKernel initial = some t ∧
      t.mem "Out" 242 = MemCell.real 336 ∧
      t.mem "Out" 246 = MemCell.real (if hasSeq then 0 else 1025) ∧
      t.mem "Out" 243 = MemCell.real 0 ∧
      t.mem "Out" 254 = MemCell.real 0 ∧
      t.mem "Out" 241 = MemCell.of .fp16 (some 9) ∧
      (∀ (r : RegionName) a, r ≠ "Out" → t.mem r a = initial.mem r a) := by
  obtain ⟨t, he, hv, hf⟩ := RealMemory.executes swapped initial 4 2 2 (parameters causal hasSeq)
    (by decide) (addresses_distinct causal hasSeq)
  refine ⟨t, he, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · have h := hv (0,0,PUnit.unit) (by cases causal <;> norm_num [RealMemory.active, RealSetup.active, RealSetup.rows, RealSetup.columns, RealSetup.tileM, RealSetup.tileN, RealSetup.tilesN, parameters, initial])
    cases hasSeq <;> norm_num [RealMemory.formula, RealMemory.sequenceID, RealMemory.address, RealMemory.base,
      RealLoop.product, RealSetup.rows, RealSetup.columns, RealSetup.tileM, RealSetup.tileN,
      RealSetup.tilesN, RealSetup.limit, RealSetup.chunk, RealSetup.head, RealSetup.seqBase,
      RealSetup.aBase, RealSetup.bBase, initial, parameters, BlockState.readMem,
      BlockState.readMemValue, BlockState.readMemTyped, Region.cast,
      Finset.sum_range_succ] at h ⊢
    all_goals
      try simp only [String.reduceEq, if_false] at h
      norm_num at h
      exact h
  · have h := hv (1,0,PUnit.unit) (by cases causal <;> norm_num [RealMemory.active, RealSetup.active, RealSetup.rows, RealSetup.columns, RealSetup.tileM, RealSetup.tileN, RealSetup.tilesN, parameters, initial])
    cases hasSeq <;> norm_num [RealMemory.formula, RealMemory.sequenceID, RealMemory.address, RealMemory.base,
      RealLoop.product, RealSetup.rows, RealSetup.columns, RealSetup.tileM, RealSetup.tileN,
      RealSetup.tilesN, RealSetup.limit, RealSetup.chunk, RealSetup.head, RealSetup.seqBase,
      RealSetup.aBase, RealSetup.bBase, initial, parameters, BlockState.readMem,
      BlockState.readMemValue, BlockState.readMemTyped, Region.cast,
      Finset.sum_range_succ] at h ⊢
    all_goals
      try simp only [String.reduceEq, if_false] at h
      norm_num at h
      exact h
  · have h := hv (0,1,PUnit.unit) (by cases causal <;> norm_num [RealMemory.active, RealSetup.active, RealSetup.rows, RealSetup.columns, RealSetup.tileM, RealSetup.tileN, RealSetup.tilesN, parameters, initial])
    cases hasSeq <;> norm_num [RealMemory.formula, RealMemory.sequenceID, RealMemory.address, RealMemory.base,
      RealLoop.product, RealSetup.rows, RealSetup.columns, RealSetup.tileM, RealSetup.tileN,
      RealSetup.tilesN, RealSetup.limit, RealSetup.chunk, RealSetup.head, RealSetup.seqBase,
      RealSetup.aBase, RealSetup.bBase, initial, parameters, BlockState.readMem,
      BlockState.readMemValue, BlockState.readMemTyped, Region.cast,
      Finset.sum_range_succ] at h ⊢
    all_goals
      try simp only [String.reduceEq, if_false] at h
      norm_num at h
      exact h
  · have h := hv (3,0,PUnit.unit) (by cases causal <;> norm_num [RealMemory.active, RealSetup.active, RealSetup.rows, RealSetup.columns, RealSetup.tileM, RealSetup.tileN, RealSetup.tilesN, parameters, initial])
    cases hasSeq <;> norm_num [RealMemory.formula, RealMemory.sequenceID, RealMemory.address, RealMemory.base,
      RealLoop.product, RealSetup.rows, RealSetup.columns, RealSetup.tileM, RealSetup.tileN,
      RealSetup.tilesN, RealSetup.limit, RealSetup.chunk, RealSetup.head, RealSetup.seqBase,
      RealSetup.aBase, RealSetup.bBase, initial, parameters, BlockState.readMem,
      BlockState.readMemValue, BlockState.readMemTyped, Region.cast,
      Finset.sum_range_succ] at h ⊢
    all_goals
      try simp only [String.reduceEq, if_false] at h
      norm_num at h
      exact h
  · rw [hf "Out" 241]
    · rfl
    · rintro ⟨i,j,⟨⟩⟩ _ he
      norm_num [RealMemory.address, RealMemory.base, RealSetup.rows, RealSetup.columns,
        RealSetup.tileM, RealSetup.tileN, RealSetup.tilesN, RealSetup.chunk, RealSetup.head,
        initial, parameters] at he
      omega
  · intro r a hr
    exact hf r a (fun _ _ he => hr he.1.symm)

theorem causal_skip_preserves_all_cells (swapped hasSeq : Bool) :
    ∃ t, exec (RealProgram.kernel swapped 2 2 2 (parameters Bool.true hasSeq)).toAlgKernel initial = some t ∧
      t.mem = initial.mem := by
  refine ⟨RealSetup.prologueResult initial 2 2 (parameters Bool.true hasSeq), ?_, ?_⟩
  · exact RealExecution.skipped_executes swapped initial 2 2 2 (parameters Bool.true hasSeq) (by cases hasSeq <;> decide)
  · rfl

private def allocation : FlatAlloc where
  flat := "flat"
  regions := ["A", "B", "Seq", "Out"]
  base := fun r => if r = "A" then 0 else if r = "B" then 2048 else if r = "Seq" then 4096 else 6144
  extent := fun r => if r ∈ (["A", "B", "Seq", "Out"] : List RegionName) then 2048 else 0

private theorem disjoint : allocation.Disjoint := by
  simp +decide [FlatAlloc.Disjoint]

private theorem covers (r : RegionName) (hr : r ∉ allocation.regions) : allocation.extent r = 0 := by
  simp_all [allocation]

private theorem address_bounds (causal hasSeq : Bool) :
    RealSafety.Bounds initial 4 2 (parameters causal hasSeq) allocation.extent := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro _
    refine ⟨?_, ?_⟩
    · intro i k hi hk
      norm_num [RealSetup.rows, RealSetup.columns, RealSetup.tileM, RealSetup.tileN,
        RealSetup.tilesN, RealSetup.limit, RealSetup.chunk, RealSetup.head, RealSetup.aBase,
        initial, parameters, allocation] at hi hk ⊢
      omega
    · intro k j hk hj
      norm_num [RealSetup.rows, RealSetup.columns, RealSetup.tileM, RealSetup.tileN,
        RealSetup.tilesN, RealSetup.limit, RealSetup.chunk, RealSetup.head, RealSetup.bBase,
        initial, parameters, allocation] at hk hj ⊢
      omega
  · intro _ _ i hi
    norm_num [RealSetup.rows, RealSetup.tileM, RealSetup.tilesN, RealSetup.limit,
      RealSetup.chunk, RealSetup.seqBase, initial, parameters, allocation, Region.cast] at hi ⊢
    omega
  · intro _ _ j hj
    norm_num [RealSetup.columns, RealSetup.tileN, RealSetup.tilesN, RealSetup.limit,
      RealSetup.chunk, RealSetup.seqBase, initial, parameters, allocation, Region.cast] at hj ⊢
    omega
  · intro i hi
    norm_num [RealMemory.active, RealMemory.address, RealMemory.base,
      RealSetup.rows, RealSetup.columns, RealSetup.tileM, RealSetup.tileN, RealSetup.tilesN,
      RealSetup.limit, RealSetup.chunk, RealSetup.head, initial, parameters, allocation] at hi ⊢
    omega

/-- Flattening preserves both computed values, the zero-filled padded cells,
and an untouched tagged output cell, for both sequence arms and both versions. -/
theorem complete_flat_source (swapped causal hasSeq : Bool) :
    ∃ t, exec (allocation.flattenKernel (RealProgram.kernel swapped 4 2 2 (parameters causal hasSeq)).toAlgKernel)
        (allocation.flattenState initial) = some t ∧
      t.mem "flat" 6386 = MemCell.real 336 ∧
      t.mem "flat" 6390 = MemCell.real (if hasSeq then 0 else 1025) ∧
      t.mem "flat" 6387 = MemCell.real 0 ∧
      t.mem "flat" 6398 = MemCell.real 0 ∧
      t.mem "flat" 6385 = MemCell.of .fp16 (some 9) := by
  obtain ⟨t, he, h0, h1, hc, hr, hu, _⟩ := complete_source swapped causal hasSeq
  refine ⟨allocation.flattenState t, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [FlatAlloc.exec_flatten allocation disjoint covers _ initial
      (RealSafety.safe swapped initial 4 2 2 (parameters causal hasSeq) allocation.extent (address_bounds causal hasSeq))
      (RealSafety.flatten_ok swapped 4 2 2 (parameters causal hasSeq)) rfl, he, Option.map_some]
  all_goals
    have hf (a : Nat) (ha : a < 2048) :
        (allocation.flattenState t).mem "flat" (6144 + a) = allocation.trCell (t.mem "Out" a) :=
      FlatAlloc.flattenState_mem_addr allocation disjoint t (r := "Out") (o := a) (by simp [allocation]) (by simpa [allocation] using ha)
    first
    | exact (hf 242 (by decide)).trans (by rw [h0]; rfl)
    | exact (hf 246 (by decide)).trans (by rw [h1]; rfl)
    | exact (hf 243 (by decide)).trans (by rw [hc]; rfl)
    | exact (hf 254 (by decide)).trans (by rw [hr]; rfl)
    | exact (hf 241 (by decide)).trans (by rw [hu]; rfl)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end BmmChunkFwdCompleteTests
