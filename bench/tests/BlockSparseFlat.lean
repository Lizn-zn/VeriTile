import bench.tritonbench_g.block_sparse_attn.Correct

/-! A complete flat-memory witness with distinct allocation bases, signed
CSR visits, repeated keys, a partial query tile and both channel branches.
Only row zero is causal/visible; the output is 3. Untouched output padding
retains its original natural-number tag. All mask flags remain parameters. -/
namespace BlockSparseFlatTests
open VeriTile.Triton VeriTile.Bench.TritonBenchG.BlockSparseAttn
open RealAddressFormula
noncomputable section
set_option linter.unusedSimpArgs false

private def config (evenM evenN two : Bool) : RealProgram.Parameters where
  Out := "O"
  Q := "Q"
  K := "K"
  V := "V"
  rows := ⟨"R"⟩
  cols := ⟨"C"⟩
  rowStride := 3
  columnStride := 5
  layouts := 2
  scale := 0
  qBatch := 0
  qHead := 0
  qRow := 4
  kBatch := 0
  kHead := 0
  kRow := 6
  vBatch := 0
  vHead := 0
  vRow := 8
  oBatch := 0
  oHead := 0
  oRow := 4
  heads := 2
  kvHeads := 2
  context := 1
  blocks := if two then 2 else 1
  evenM := evenM
  evenN := evenN

private def initial : BlockState where
  mem := fun region offset =>
    if region = "R" then MemCell.of .int (if offset = 3 then -1 else 1)
    else if region = "C" then MemCell.of .int 0
    else if region = "O" then MemCell.of .nat 99 else MemCell.real 3
  undef := fun _ _ => 0
  regs := fun _ _ _ => none
  pids := fun gridAxis => if gridAxis = 1 then 1 else 0
  numPids := fun _ => 8


private def inputs : CSRMath.Inputs where
  query := fun _ _ => 3
  key := fun _ _ => 3
  value := fun _ _ => 3
  rowIndex := fun a => if a = 3 then -1 else 1
  columnIndex := fun _ => 0

private def allocation (evenM evenN : Bool) : FlatAlloc where
  flat := "flat"
  regions := ["Q", "K", "V", "R", "C", "O"]
  base := fun region =>
    if region = "K" then 16 else if region = "V" then 32
    else if region = "R" then 48 else if region = "C" then 64
    else if region = "O" then 80 else 0
  extent := fun region =>
    if region = "Q" then (if evenM then 6 else 2)
    else if region = "K" then (if evenN then 8 else 2)
    else if region = "V" then (if evenN then 10 else 2)
    else if region = "R" then 5 else if region = "C" then 6
    else if region = "O" then 4 else 0

private theorem interval (evenM evenN two : Bool) :
    CSRMath.count (schedule initial (config evenM evenN two)) inputs = 2 := by
  norm_num [Int.toNat, CSRMath.count, CSRMath.first, CSRMath.last, schedule, RealSetup.head, initial, config, inputs]

private theorem channels (evenM evenN two : Bool) (d : Nat)
    (hd : d < RealInputs.width 1 (config evenM evenN two)) : d < 2 := by
  cases two <;> simp [RealInputs.width, config] at hd <;> omega

theorem pins (evenM evenN two : Bool) : RealInputs.Memory initial 1 (config evenM evenN two) inputs := by
  constructor
  · intro next
    cases next <;>
      norm_num [RealInputs.rowAddress, schedule, RealSetup.head, BlockState.readMemValue,
        BlockState.readMemTyped, initial, config, inputs, Region.cast]
  · intro visit _
    simp [BlockState.readMemValue, BlockState.readMemTyped, initial, config, inputs, Region.cast]
  · intro r d _ _
    simp [BlockState.readMem, initial, config, inputs, MemCell.real]
  · intro r d _ _ _
    simp [BlockState.readMem, initial, config, inputs, MemCell.real]
  · intro r d _ _ _
    simp [BlockState.readMem, initial, config, inputs, MemCell.real]

theorem valid (evenM evenN two : Bool) : RealInputs.Valid initial (config evenM evenN two) inputs := by
  constructor <;> intro visit hv
  · norm_num [RealInputs.columnSlot, CSRMath.first, schedule, RealSetup.head, initial, config, inputs]
    omega
  · simp [inputs]

theorem windows (evenM evenN two : Bool) :
    RealWindows.Bounds (allocation evenM evenN).extent initial 2 2 1 (config evenM evenN two) inputs := by
  constructor
  · intro next
    change RealInputs.rowAddress initial (config evenM evenN two) next < 5
    cases next <;> norm_num [RealInputs.rowAddress, schedule, RealSetup.head, initial, config, allocation, Region.cast, Region.mk.injEq]
  · intro visit hv
    change (RealInputs.columnSlot initial (config evenM evenN two) inputs visit).toNat < 6
    rw [interval] at hv
    norm_num [RealInputs.columnSlot, CSRMath.first, schedule, RealSetup.head, initial, config, inputs, allocation, Region.cast, Region.mk.injEq]
    omega
  · intro r d hd ha
    have hd' := channels evenM evenN two d hd
    have hr := r.isLt
    cases evenM <;>
      simp [RealWindows.query, RealSetup.qBase, RealSetup.batch, RealSetup.head, initial, config, allocation] at ha ⊢ <;> omega
  · intro visit _ j d hd ha
    have hd' := channels evenM evenN two d hd
    have hj := j.isLt
    cases evenN <;>
      simp [RealWindows.key, CSRMath.selectedKey, RealSetup.kBase, RealSetup.batch, RealSetup.head,
        RealSetup.keyHead, initial, config, inputs, allocation] at ha ⊢ <;> omega
  · intro visit _ j d hd ha
    have hd' := channels evenM evenN two d hd
    have hj := j.isLt
    cases evenN <;>
      simp [RealWindows.value, CSRMath.selectedKey, RealSetup.vBase, RealSetup.batch, RealSetup.head,
        RealSetup.keyHead, initial, config, inputs, allocation] at ha ⊢ <;> omega
  · intro r d hd ha
    have hd' := channels evenM evenN two d hd
    simp [RealWindows.output, RealSetup.batch, RealSetup.head, initial, config, allocation] at ha ⊢
    omega

theorem disjoint (evenM evenN : Bool) : (allocation evenM evenN).Disjoint := by
  intro r hr r' hr' hne
  simp only [allocation, List.mem_cons, List.mem_nil_iff, or_false] at hr hr'
  rcases hr with rfl | rfl | rfl | rfl | rfl | rfl <;>
    rcases hr' with rfl | rfl | rfl | rfl | rfl | rfl <;>
    cases evenM <;> cases evenN <;> norm_num [allocation, Region.mk.injEq] at * <;> decide

theorem covers (evenM evenN : Bool) (r : RegionName) (hr : r ∉ (allocation evenM evenN).regions) :
    (allocation evenM evenN).extent r = 0 := by
  simp only [allocation, List.mem_cons, List.mem_nil_iff, or_false, not_or] at hr
  simp [allocation, hr.1, hr.2.1, hr.2.2.1, hr.2.2.2.1, hr.2.2.2.2.1, hr.2.2.2.2.2]

private theorem formula (evenM evenN two : Bool) (d : Nat) :
    CSRMath.attention (schedule initial (config evenM evenN two)) inputs 2 1
      (config evenM evenN two).blocks 0 d 0 = 3 := by
  simp only [CSRMath.attention, interval, Finset.sum_range_succ, Finset.sum_range_zero, Fin.sum_univ_two]
  norm_num [CSRMath.dotScore, CSRMath.selectedKey, inputs]

/-- Exercise the public specifications for every storage and source choice,
with actual independent input pins and a concrete, disjoint allocation. -/
theorem complete_flat_source (storage : TypedSource.Storage) (swapped evenM evenN two : Bool) :
    ∃ t, exec ((allocation evenM evenN).flattenKernel
      (RealProgram.source storage swapped 2 2 1 (config evenM evenN two)).eraseDType.toAlgKernel)
      ((allocation evenM evenN).flattenState initial) = some t ∧
      t.mem "flat" 80 = MemCell.real 3 ∧
      (two = Bool.true → t.mem "flat" 81 = MemCell.real 3) ∧
      t.mem "flat" 83 = MemCell.of .nat 99 := by
  have hspec : (RealContract.io storage swapped 2 2 1 (config evenM evenN two)).Implements 2
      (Correctness.expected 2 2 1 (config evenM evenN two)) := by
    cases swapped
    · exact Correctness.block_sparse_attention_correctness storage 2 2 1 (config evenM evenN two)
        (by decide) (by norm_num [config]) (by intro _; norm_num [config])
    · exact Correctness.block_sparse_scalar_left_correctness storage 2 2 1 (config evenM evenN two)
        (by decide) (by norm_num [config]) (by intro _; norm_num [config])
  have hp := pins evenM evenN two
  have hv := valid evenM evenN two
  have hw := windows evenM evenN two
  obtain ⟨t, he, hy, hf⟩ := hspec (allocation evenM evenN) (disjoint evenM evenN)
    (covers evenM evenN) initial inputs ⟨hv.slots, hv.columns⟩ rfl
    ⟨hw.rows, hw.columns, hw.query, hw.key, hw.value, hw.output⟩
    ⟨hp.rows, hp.columns, hp.query, hp.key, hp.value⟩
  have hr : initial.pids 0 * 2 + (0 : Fin 2).val < (config evenM evenN two).context := by
    norm_num [initial, config]
  have hd : 0 < (RealContract.io storage swapped 2 2 1 (config evenM evenN two)).width := by
    cases two <;> norm_num [RealContract.io, config]
  refine ⟨t, he, ?_, ?_, ?_⟩
  · exact (hy 0 0 hd hr).trans (congrArg MemCell.real (formula evenM evenN two 0))
  · intro hb
    have hd2 : 1 < (RealContract.io storage swapped 2 2 1 (config evenM evenN two)).width := by
      simp [RealContract.io, config, hb]
    exact (hy 0 1 hd2 hr).trans (congrArg MemCell.real (formula evenM evenN two 1))
  · have hframe : t.mem "flat" 83 = ((allocation evenM evenN).flattenState initial).mem "flat" 83 := by
      apply hf "flat" 83
      intro r d hd' hr'
      have hdlt := channels evenM evenN two d hd'
      have hrv : r.val = 0 := by simp [initial, RealContract.io, config] at hr'; omega
      interval_cases d <;>
        norm_num [FlatAlloc.addr, allocation, RealContract.io, initial, config, hrv, Region.mk.injEq] <;> decide
    rw [hframe]
    change ((allocation evenM evenN).flattenState initial).mem (allocation evenM evenN).flat
      ((allocation evenM evenN).addr "O" 3) = _
    rw [FlatAlloc.flattenState_mem_addr _ (disjoint evenM evenN) initial (by simp [allocation]) (by change 3 < 4; decide)]
    rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end BlockSparseFlatTests
