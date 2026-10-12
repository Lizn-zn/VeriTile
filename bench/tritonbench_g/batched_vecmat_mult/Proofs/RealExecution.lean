import bench.tritonbench_g.batched_vecmat_mult.Proofs.Real

/-! Nonempty flat-memory witness for both complete real sources. A 2×3 tile
at program IDs (1,1) reads two nonempty K blocks from disjoint allocations.
All inputs are 2, so each of the six outputs is 16. The theorem instantiates
the complete contract, including actual bounds and the memory frame. -/
namespace VeriTile.Bench.TritonBenchG.BatchedVecmatMult.RealExecution
open VeriTile.Triton
noncomputable section

private def allocation : FlatAlloc where
  flat := "flat"
  regions := ["A", "B", "Out"]
  base := fun r => if r = "A" then 0 else if r = "B" then 16 else 112
  extent := fun r => if r = "A" then 16 else if r = "B" then 96
    else if r = "Out" then 24 else 0

private def initial : BlockState where
  mem := fun _ _ => MemCell.real 2
  regs := fun _ _ _ => none
  pids := fun _ => 1

private def io (candidate : Bool) := RealProofs.io "A" "B" "Out" 4 6 2 3 2 2 candidate

private theorem allocation_disjoint : allocation.Disjoint := by
  simp +decide [FlatAlloc.Disjoint]

private theorem io_fields (candidate : Bool) :
    (io candidate).inp1 = "A" ∧ (io candidate).inp2 = "B" ∧ (io candidate).out = "Out" := by
  cases candidate <;> exact ⟨rfl,rfl,rfl⟩

theorem executes (candidate : Bool) :
    ∃ t, exec (allocation.flattenKernel (io candidate).kernel.toAlgKernel)
        (allocation.flattenState initial) = some t ∧
      (∀ j : Fin 6, t.readMem "flat" (112 + ((2+j.val/3)*6+(3+j.val%3))) = 16) ∧
      (∀ (r : RegionName) o, (r ≠ "flat" ∨ ∀ j : Fin 6, o ≠ 112 + ((2+j.val/3)*6+(3+j.val%3))) →
        t.mem r o = (allocation.flattenState initial).mem r o) := by
  have proof : StreamMasked2DKernelIO₂.Implements (io candidate)
      (fun _ _ xs ys l => RealProofs.expected 2 3 2 2 xs ys l) := by
    cases candidate
    · exact RealProofs.original_correct _ _ _ _ _ _ _ _ _ (by decide) (by decide)
    · exact RealProofs.candidate_correct _ _ _ _ _ _ _ _ _ (by decide) (by decide)
  have hbA : ∀ (t : Fin 2) (l : Fin 4), (io candidate).read1 1 1 t l < 16 := by cases candidate <;> decide
  have hbB : ∀ (t : Fin 2) (l : Fin 12), (io candidate).read2 1 1 t l < 96 := by cases candidate <;> decide
  have hbO : ∀ l : Fin 6, (io candidate).write 1 1 l < 24 := by cases candidate <;> decide
  obtain ⟨t, he, hy, hf⟩ := proof allocation allocation_disjoint
    (by simp [allocation, (io_fields candidate).1, (io_fields candidate).2.1, (io_fields candidate).2.2])
    (by intro r hr; simp_all [allocation]) 1 1 (fun _ _ => 2) (fun _ _ => 2) initial
    rfl rfl rfl
    (by simpa [io, RealProofs.io, batchedVecmatIO, allocation] using fun t l (_ : True) => hbA t l)
    (by simpa [io, RealProofs.io, batchedVecmatIO, allocation] using fun t l (_ : True) => hbB t l)
    (by simpa [io, RealProofs.io, batchedVecmatIO, allocation] using fun l (_ : True) => hbO l)
    (by intros; rfl) (by intros; rfl)
  refine ⟨t, he, ?_, ?_⟩
  · intro j
    have h := hy j trivial
    simpa [io, RealProofs.io, batchedVecmatIO, FlatAlloc.addr, allocation,
      RealProofs.expected, Fin.sum_univ_succ, show (2 : ℝ) * (2 * (2*2)) = 16 by norm_num] using h
  · simpa [io, RealProofs.io, batchedVecmatIO, FlatAlloc.addr, allocation] using hf

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BatchedVecmatMult.RealExecution
