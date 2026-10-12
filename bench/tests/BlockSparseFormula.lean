import bench.tritonbench_g.block_sparse_attn.Proofs.RealFormula

/-! A complete independent-array witness: signed visits [-1,2), separate CSR
strides, repeated key blocks, a partial query tile and a masked final key.
Five visible occurrences have values 4,7,10,4,7, giving 32/5 at zero scale.
The second channel doubles the value. The invalid query row keeps its tag. -/
namespace BlockSparseFormulaTests
open VeriTile.Triton VeriTile.Bench.TritonBenchG.BlockSparseAttn
open RealAddressFormula
noncomputable section
set_option linter.unusedSimpArgs false

private def config (evenM evenN two : Bool) : RealProgram.Parameters where
  Out := "Out"
  Q := "Queries"
  K := "Keys"
  V := "Values"
  rows := ⟨"Rows"⟩
  cols := ⟨"Cols"⟩
  rowStride := 5
  columnStride := 7
  layouts := 2
  scale := 0
  qBatch := 17
  qHead := 19
  qRow := 3
  kBatch := 23
  kHead := 29
  kRow := 5
  vBatch := 31
  vHead := 37
  vRow := 7
  oBatch := 41
  oHead := 43
  oRow := 11
  heads := 2
  kvHeads := 2
  context := 3
  blocks := if two then 2 else 1
  evenM := evenM
  evenN := evenN

private def inputs : CSRMath.Inputs where
  query := fun r d => r + d + 1
  key := fun r d => r + d + 2
  value := fun r d => (if d = 0 then 1 else 2) * (3 * r + 4)
  rowIndex := fun a => if a = 6 then -1 else if a = 7 then 2 else 0
  columnIndex := fun a => if a = 7 then 1 else 0

private def initial : BlockState where
  mem := fun region offset =>
    if region.name == "Rows" then MemCell.of .int (inputs.rowIndex offset)
    else if region.name == "Cols" then MemCell.of .int (inputs.columnIndex offset)
    else if region.name == "Queries" then MemCell.real (inputs.query ((offset-36)/3) ((offset-36)%3))
    else if region.name == "Keys" then MemCell.real (inputs.key (((offset-52)/5 : Nat) : Int) ((offset-52)%5))
    else if region.name == "Values" then MemCell.real (inputs.value (((offset-68)/7 : Nat) : Int) ((offset-68)%7))
    else MemCell.of .nat 99
  undef := fun _ offset => 1000 + offset
  regs := fun _ _ _ => none
  pids := fun gridAxis => if gridAxis = 0 then 1 else if gridAxis = 1 then 3 else 0
  numPids := fun _ => 8

private theorem interval (evenM evenN two : Bool) :
    CSRMath.first (schedule initial (config evenM evenN two)) inputs = -1 ∧
      CSRMath.last (schedule initial (config evenM evenN two)) inputs = 2 ∧
      CSRMath.count (schedule initial (config evenM evenN two)) inputs = 3 := by
  norm_num [Int.toNat, CSRMath.first, CSRMath.last, CSRMath.count, schedule, RealSetup.head, initial, config, inputs]

private theorem channels (evenM evenN two : Bool) (d : Nat) (hd : d < RealInputs.width 1 (config evenM evenN two)) :
    d < 2 := by
  cases two <;> simp [RealInputs.width, config] at hd <;> omega

theorem independent_inputs (evenM evenN two : Bool) : RealInputs.Memory initial 1 (config evenM evenN two) inputs := by
  constructor
  · intro next
    simp [BlockState.readMemValue, BlockState.readMemTyped, initial, config, Region.cast]
  · intro visit hv
    rw [(interval evenM evenN two).2.2] at hv
    interval_cases visit <;>
      norm_num [RealInputs.columnSlot, CSRMath.first, Int.toNat, schedule, RealSetup.head,
        initial, config, inputs, BlockState.readMemValue, BlockState.readMemTyped, Region.cast] <;> rfl
  · intro r d hr hd
    have hd' := channels evenM evenN two d hd
    change r < 3 at hr
    interval_cases r <;> interval_cases d <;>
      norm_num [BlockState.readMem, RealSetup.qBase, RealSetup.batch, RealSetup.head,
        config, initial, inputs, MemCell.real, Int.toNat] <;> rfl
  · intro r d hr hlo hd
    have hd' := channels evenM evenN two d hd
    change r < (3 : Int) at hlo
    interval_cases r <;> interval_cases d <;>
      norm_num [BlockState.readMem, RealSetup.kBase, RealSetup.batch, RealSetup.head, RealSetup.keyHead,
        config, initial, inputs, MemCell.real, Int.toNat] <;> rfl
  · intro r d hr hlo hd
    have hd' := channels evenM evenN two d hd
    change r < (3 : Int) at hlo
    interval_cases r <;> interval_cases d <;>
      norm_num [BlockState.readMem, RealSetup.vBase, RealSetup.batch, RealSetup.head, RealSetup.keyHead,
        config, initial, inputs, MemCell.real, Int.toNat] <;> rfl

theorem signed_schedule_valid (evenM evenN two : Bool) : RealInputs.Valid initial (config evenM evenN two) inputs := by
  constructor <;> intro visit hv
  all_goals
    rw [(interval evenM evenN two).2.2] at hv
    interval_cases visit <;>
      norm_num [RealInputs.columnSlot, CSRMath.first, Int.toNat, schedule, RealSetup.head,
        initial, config, inputs]

theorem signed_interval_and_repeated_keys (evenM evenN two : Bool) :
    CSRMath.first (schedule initial (config evenM evenN two)) inputs = -1 ∧
      CSRMath.selectedKey (schedule initial (config evenM evenN two)) inputs 2 0 0 = 0 ∧
      CSRMath.selectedKey (schedule initial (config evenM evenN two)) inputs 2 1 0 = 2 ∧
      CSRMath.selectedKey (schedule initial (config evenM evenN two)) inputs 2 2 0 = 0 := by
  norm_num [CSRMath.selectedKey, CSRMath.first, Int.toNat, schedule, RealSetup.head, initial, config, inputs]

theorem first_formula (evenM evenN two : Bool) :
    CSRMath.attention (schedule initial (config evenM evenN two)) inputs 2 1 (config evenM evenN two).blocks 2 0 0 = 32/5 := by
  simp only [CSRMath.attention, (interval evenM evenN two).2.2, Finset.sum_range_succ,
    Finset.sum_range_zero, Fin.sum_univ_two]
  norm_num [CSRMath.dotScore, CSRMath.selectedKey, CSRMath.first, Int.toNat,
    schedule, RealSetup.head, initial, config, inputs]

theorem second_formula (evenM evenN two : Bool) :
    CSRMath.attention (schedule initial (config evenM evenN two)) inputs 2 1 (config evenM evenN two).blocks 2 1 0 = 64/5 := by
  simp only [CSRMath.attention, (interval evenM evenN two).2.2, Finset.sum_range_succ,
    Finset.sum_range_zero, Fin.sum_univ_two]
  norm_num [CSRMath.dotScore, CSRMath.selectedKey, CSRMath.first, Int.toNat,
    schedule, RealSetup.head, initial, config, inputs]

/-- Both complete kernels run from the independently pinned initial arrays;
no intermediate-state, score or output-value premise is supplied. -/
theorem complete_source (swapped evenM evenN two : Bool) :
    ∃ t, stepStmts (RealProgram.kernel swapped 2 2 1 (config evenM evenN two)).toAlgKernel.body initial = some t ∧
      t.mem "Out" 106 = MemCell.real (32/5) ∧
      (two = Bool.true → t.mem "Out" 107 = MemCell.real (64/5)) ∧
      t.mem "Out" 117 = MemCell.of .nat 99 := by
  obtain ⟨t, he, hfirst, hsecond, hframe⟩ := RealFormula.source_correct swapped initial 2 2 1
    (config evenM evenN two) inputs (independent_inputs ..) (signed_schedule_valid ..)
    (by decide) (by norm_num [config]) (by intro _; norm_num [config])
  have hr : RealSetup.rows initial 2 (0 : Fin 2) < (config evenM evenN two).context := by
    norm_num [RealSetup.rows, initial, config]
  refine ⟨t, he, ?_, ?_, ?_⟩
  · have h := hfirst 0 0 hr
    have hf := first_formula evenM evenN two
    convert h.trans (congrArg MemCell.real hf) using 1
    norm_num [RealStores.offset, RealSetup.rows, RealSetup.batch, RealSetup.head, initial, config]
  · intro htwo
    have h := hsecond (by simp [config, htwo]) 0 0 hr
    have hf := second_formula evenM evenN two
    convert h.trans (congrArg MemCell.real hf) using 1
    norm_num [RealStores.offset, RealSetup.rows, RealSetup.batch, RealSetup.head, initial, config]
  · rw [hframe "Out" 117]
    · rfl
    · intro r d hr'
      have hd : d.val = 0 := by omega
      have hrv : r.val = 0 := by simp [RealSetup.rows, initial, config] at hr'; omega
      norm_num [RealStores.offset, RealSetup.rows, RealSetup.batch, RealSetup.head, initial, config, hd, hrv]
    · intro _ r d hr'
      have hd : d.val = 0 := by omega
      have hrv : r.val = 0 := by simp [RealSetup.rows, initial, config] at hr'; omega
      norm_num [RealStores.offset, RealSetup.rows, RealSetup.batch, RealSetup.head, initial, config, hd, hrv]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end BlockSparseFormulaTests
