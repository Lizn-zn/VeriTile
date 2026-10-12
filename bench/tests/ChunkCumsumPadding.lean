import bench.tritonbench_g.chunk_cumsum_kernel.Proofs.RealPadding

/-! Complete source executions with three chunks and a partial final chunk.
The padding value is arbitrary, including seven; valid outputs are still
1, 2, 3, 4, 5. This checks real semantics after precision erasure, not GPU FP. -/
namespace ChunkCumsumPaddingTests
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkCumsumKernel
noncomputable section

private def initial (padding : ℝ) : BlockState where
  mem := fun _ _ => .real 1
  regs := fun _ _ _ => none
  pids := fun _ => 1
  undef := fun _ _ => padding

/-- Both complete sources retain every valid prefix for all real padding values. -/
theorem complete_outputs (swapped : Bool) (padding : ℝ) :
    ∃ v, exec (RealSource.source swapped .fp16 .bf16 "s" "o" 5 2).eraseDType.toAlgKernel
      (initial padding) = some v ∧
      ∀ i : Fin 5, v.readMem "o" (5 + i.val) = (i.val + 1 : Nat) := by
  obtain ⟨v, he, ho⟩ := RealPadding.source_outputs swapped .fp16 .bf16 "s" "o" 5 2
    (initial padding) (by decide) (by decide)
  refine ⟨v, he, ?_⟩
  intro i
  have hi := ho i.val i.isLt
  have hm (a : Nat) : (initial padding).readMem "s" a = 1 := rfl
  simp only [hm] at hi
  fin_cases i <;>
    norm_num [initial, Finset.sum_range_succ, Finset.sum_filter] at hi ⊢ <;> exact hi

theorem nonzero_padding_outputs (swapped : Bool) :
    ∃ v, exec (RealSource.source swapped .fp16 .bf16 "s" "o" 5 2).eraseDType.toAlgKernel
      (initial 7) = some v ∧
      ∀ i : Fin 5, v.readMem "o" (5 + i.val) = (i.val + 1 : Nat) :=
  complete_outputs swapped 7

#guard_msgs (drop info) in
#auditModuleAxioms
end
end ChunkCumsumPaddingTests
