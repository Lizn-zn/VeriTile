import bench.tritonbench_g.chunk_cumsum_vector.Correct

/-! Both complete sources run three chunks with a partial time row and a
partial feature block. Nonzero feature/head IDs test the actual addressing.
Arbitrary padding cannot affect valid prefixes or neighboring output cells.
These are real executions after precision erasure, not GPU numerical results. -/
namespace VectorCumsumPaddingTests
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkCumsumVector
noncomputable section

private def layout : RealProgram.Layout := ⟨15,3,1,5,3,2,2⟩

private def initial (padding : ℝ) : BlockState where
  mem := fun r _ => if r = "s" then .real 2 else .real (-9)
  regs := fun _ _ _ => none
  pids := fun _ => 1
  undef := fun _ _ => padding

theorem complete_outputs_and_frame (swapped : Bool) (padding : ℝ) :
    ∃ v, exec (RealSource.source swapped .bf16 .fp16 "s" "o"
      15 3 1 5 3 2 2).eraseDType.toAlgKernel (initial padding) = some v ∧
      (∀ i : Fin 5, v.readMem "o" (17 + i.val * 3) = (2 * (i.val + 1) : Nat)) ∧
      (∀ off, off = 16 ∨ off = 18 ∨ off = 30 →
        v.mem "o" off = (initial padding).mem "o" off) := by
  have hinj : RealLoop.NoCollision layout (RealInit.launch (initial padding)) := by
    intro t u i j ht hu heq
    dsimp [RealStep.active, RealStep.address, RealInit.launch, initial, layout] at ht hu heq
    exact ⟨by omega, Fin.ext (by omega)⟩
  obtain ⟨v, he, ho, hf⟩ := RealPadding.source_outputs swapped layout "s" "o"
    .bf16 .fp16 (initial padding) (by decide) (by decide) hinj
  refine ⟨v, he, ?_, ?_⟩
  · intro i
    have hi := ho i.val ⟨0, by decide⟩
      ⟨i.isLt, by norm_num [RealInit.launch, initial, layout]⟩
    have hm (off : Nat) : (initial padding).readMem "s" off = 2 := rfl
    simp only [Finset.sum_filter, hm] at hi
    fin_cases i <;>
      norm_num [layout, RealStep.address, RealInit.launch, initial,
        Finset.sum_range_succ, Finset.sum_filter] at hi ⊢ <;> exact hi
  · intro off hoff
    apply (hf "o" off ?_).symm
    rintro ⟨_, t, j, ht, ha⟩
    dsimp [RealStep.active, RealStep.address, RealInit.launch, initial, layout] at ht ha
    rcases hoff with rfl | rfl | rfl <;> omega

/-- In particular, neither masked axis has to supply zero. -/
theorem nonzero_padding_outputs (swapped : Bool) :
    ∃ v, exec (RealSource.source swapped .bf16 .fp16 "s" "o"
      15 3 1 5 3 2 2).eraseDType.toAlgKernel (initial 7) = some v ∧
      ∀ i : Fin 5, v.readMem "o" (17 + i.val * 3) = (2 * (i.val + 1) : Nat) := by
  obtain ⟨v, he, ho, _⟩ := complete_outputs_and_frame swapped 7
  exact ⟨v, he, ho⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VectorCumsumPaddingTests
