import bench.tritonbench_g.bgmv_shrink_kernel.Proofs.RealProgram

/-! Full-source real arithmetic witness with empty registers, signed slot 1,
batch 1, nonunit output stride, a masked N tail and a masked K tail. Store
runs four K blocks; atomic addition runs two with split ID 1. Unlike the
abstract FP execution witness, the reduction here is the actual real sum. -/
namespace VeriTile.Bench.TritonBenchG.BgmvShrinkKernel.RealExecution
open VeriTile.Triton RealSetupSource
noncomputable section
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false

def parameters (atomic : Bool) : Parameters where
  input := "X"
  weights := "W"
  output := "Y"
  indices := "Index"
  N := 3
  K := 7
  scaling := 3
  inputBatchStride := 9
  inputStride := 17
  slotStride := 40
  rowStride := 9
  colStride := 1
  outputBatchStride := 13
  outputStride := 2
  BN := 4
  BK := 2
  splits := if atomic then 2 else 1

def initial (atomic : Bool) (slot : Int) : BlockState where
  mem := fun r o => if r = "Index" then ⟨.int, slot⟩ else .real
    (if r = "X" then (o : ℝ)-8 else if r = "W" then (o : ℝ)-39 else 100+o)
  regs := fun _ _ _ => none
  pids := fun ax => if ax = 0 then if atomic then 1 else 0 else 1

private theorem read_input (atomic : Bool) (slot : Int) (o : Nat) :
    (initial atomic slot).readMem "X" o = (o : ℝ)-8 := rfl
private theorem read_weight (atomic : Bool) (slot : Int) (o : Nat) :
    (initial atomic slot).readMem "W" o = (o : ℝ)-39 := rfl
private theorem read_output (atomic : Bool) (slot : Int) (o : Nat) :
    (initial atomic slot).readMem "Y" o = 100+(o : ℝ) := rfl

private theorem formula_values (atomic : Bool) (n : Fin 3) :
    RealProgram.formula (parameters atomic) (initial atomic 1) 1 n.val =
      if atomic then 335+380*(n.val : ℝ) else 420+756*(n.val : ℝ) := by
  dsimp only [RealProgram.formula, RealProgram.contribution, parameters, RealTail.offset]
  simp only [read_input, read_weight, read_output]
  cases atomic <;> fin_cases n <;>
    norm_num [initial, parameters, Fin.sum_univ_two, Finset.sum_range_succ]

/-- Both complete versions and every represented precision choice produce
store values [420,1176,1932] or atomic values [335,715,1095] at Y[13,15,17].
Every other typed cell, including masked Y[19], remains unchanged. -/
theorem executes (storage : TypedSource.Storage) (out_dtype : FloatDType)
    (swapped atomic : Bool) :
    ∃ final, exec ((parameters atomic).source storage out_dtype swapped).eraseDType.toAlgKernel
        (initial atomic 1) = some final ∧
      (∀ n : Fin 3, final.readMem "Y" (13+n.val*2) =
        if atomic then 335+380*(n.val : ℝ) else 420+756*(n.val : ℝ)) ∧
      (∀ (r : RegionName) o, (r ≠ "Y" ∨ (o ≠ 13 ∧ o ≠ 15 ∧ o ≠ 17)) →
        final.mem r o = (initial atomic 1).mem r o) := by
  obtain ⟨final, he, hv, hf⟩ := RealProgram.source_executes (parameters atomic) storage out_dtype
    swapped (initial atomic 1) 1 (by rfl) (by cases atomic <;> decide)
    (by norm_num [parameters]) (by norm_num [parameters])
  refine ⟨final, he, ?_, ?_⟩
  · intro n
    exact (hv n.val n.isLt).trans (formula_values atomic n)
  · intro r o hm
    apply hf
    intro n hn hbad
    rcases hm with hr | ho
    · exact hr hbad.1.symm
    · have heq : 13+n.val*2 = o := hbad.2
      have hn' : n.val < 3 := hn
      omega

theorem sentinel_skips (storage : TypedSource.Storage) (out_dtype : FloatDType)
    (swapped atomic : Bool) :
    ∃ final, exec ((parameters atomic).source storage out_dtype swapped).eraseDType.toAlgKernel
        (initial atomic (-1)) = some final ∧ final.mem = (initial atomic (-1)).mem :=
  RealProgram.sentinel_skips (parameters atomic) storage out_dtype swapped (initial atomic (-1)) (by rfl)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BgmvShrinkKernel.RealExecution
