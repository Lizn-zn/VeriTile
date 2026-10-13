import bench.tritonbench_g.flash_attn.Proofs.FlatReal
import bench.tritonbench_g.flash_attn.Proofs.WitnessData

/-! Concrete finite real executions with non-contiguous addresses, two blocks,
a padded final KV tile and both outputs. The fourth (padding) key has V=4;
ignoring that physical load would incorrectly change noncausal O from 1 to 0. -/
namespace VeriTile.Bench.TritonBenchG.FlashAttn.WitnessData
open VeriTile.Triton Formula RealExecution
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 20000

theorem full_run (swapped causal : Bool) :
    ∃ t, exec (allocation.flattenKernel (Candidate.source swapped "Q" "K" "V" "L" "O" (layout causal)))
        (allocation.flattenState initial) = some (allocation.flattenState t) ∧
      t.mem = (referenceState (layout causal) initial "Q" "K" "V" "L" "O").mem ∧
      ∀ r o, ¬writeWindow (layout causal) "L" "O" 0 1 r o → t.mem r o = initial.mem r o := by
  apply source_flat_correct allocation allocation_disjoint allocation_coverage swapped
    "Q" "K" "V" "L" "O" (layout causal) initial
  · exact ⟨input_bounds causal,output_bounds causal⟩
  · rfl
  · norm_num [layout]
  · cases causal <;> decide

theorem reference_outputs_noncausal :
    (referenceState (layout Bool.false) initial "Q" "K" "V" "L" "O").mem "O" 20 = .real 1 ∧
    (referenceState (layout Bool.false) initial "Q" "K" "V" "L" "O").mem "L" 3 = .real 2 := by
  simp only [referenceState]
  change (writeState (layout Bool.false) 0 1 "L" "O" _ _ initial).mem "O" 20 = _ ∧
    (writeState (layout Bool.false) 0 1 "L" "O" _ _ initial).mem "L" 3 = _
  simp only [writeState]
  simp [layout,oOffset,lOffset,base,row,TileShape.allIndices,List.finRange_succ,
    BlockState.writeMem,MemCell.real,MemCell.of]
  exact ⟨congrArg some (noncausal_output 0 0),congrArg some (noncausal_lse 0)⟩


theorem reference_outputs_causal :
    (referenceState (layout Bool.true) initial "Q" "K" "V" "L" "O").mem "O" 20 = .real 0 ∧
    (referenceState (layout Bool.true) initial "Q" "K" "V" "L" "O").mem "L" 3 = .real 0 := by
  simp only [referenceState]
  change (writeState (layout Bool.true) 0 1 "L" "O" _ _ initial).mem "O" 20 = _ ∧
    (writeState (layout Bool.true) 0 1 "L" "O" _ _ initial).mem "L" 3 = _
  simp only [writeState]
  simp [layout,oOffset,lOffset,base,row,TileShape.allIndices,List.finRange_succ,
    BlockState.writeMem,MemCell.real,MemCell.of]
  exact ⟨congrArg some (causal_output 0 0),congrArg some (by simpa using causal_lse (0 : Fin 2))⟩


theorem numerical_run (swapped : Bool) :
    ∃ t, exec (allocation.flattenKernel
        (Candidate.source swapped "Q" "K" "V" "L" "O" (layout Bool.false)))
        (allocation.flattenState initial) = some (allocation.flattenState t) ∧
      t.mem "O" 20 = .real 1 ∧ t.mem "L" 3 = .real 2 := by
  obtain ⟨t,ht,hm,_⟩ := full_run swapped Bool.false
  refine ⟨t,ht,?_⟩
  rw [hm]
  exact reference_outputs_noncausal

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.FlashAttn.WitnessData
