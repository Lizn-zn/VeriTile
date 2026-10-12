import bench.tritonbench_g.block_sparse_attn.Proofs.RealLoopSafety

/-! Complete source access safety on a partial 2×1 query tile and repeated
two-lane key blocks. Masked Q/K/V need only two cells; unmasked branches need
padding. The signed interval [-1,1) reads CSR column slots 4 and 5. -/
namespace BlockSparseSafetyTests
open VeriTile.Triton VeriTile.Bench.TritonBenchG.BlockSparseAttn
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
  scale := 1
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
    else if region = "C" then MemCell.of .int 0 else MemCell.real 3
  undef := fun _ offset => 1000 + offset
  regs := fun _ _ _ => none
  pids := fun gridAxis => if gridAxis = 1 then 1 else 0
  numPids := fun _ => 8

private def allocation (evenM evenN : Bool) : RegionBounds := fun region =>
  if region = "Q" then (if evenM then 6 else 2)
  else if region = "K" then (if evenN then 8 else 2)
  else if region = "V" then (if evenN then 10 else 2)
  else if region = "R" then 5 else if region = "C" then 6 else 2

@[simp] private theorem query_extent (evenM evenN : Bool) :
    allocation evenM evenN "Q" = if evenM then 6 else 2 := by cases evenM <;> rfl
@[simp] private theorem key_extent (evenM evenN : Bool) :
    allocation evenM evenN "K" = if evenN then 8 else 2 := by cases evenN <;> rfl
@[simp] private theorem value_extent (evenM evenN : Bool) :
    allocation evenM evenN "V" = if evenN then 10 else 2 := by cases evenN <;> rfl
@[simp] private theorem row_extent (evenM evenN : Bool) : allocation evenM evenN "R" = 5 := rfl
@[simp] private theorem column_extent (evenM evenN : Bool) : allocation evenM evenN "C" = 6 := rfl
@[simp] private theorem output_extent (evenM evenN : Bool) : allocation evenM evenN "O" = 2 := rfl

private theorem interval (evenM evenN two : Bool) :
    RealLoop.first initial (config evenM evenN two) = -1 ∧
      RealLoop.last initial (config evenM evenN two) = 1 ∧
      RealLoop.count initial (config evenM evenN two) = 2 := by
  norm_num [RealLoop.first, RealLoop.last, RealLoop.count, RealRows.bound, RealRows.address,
    RealSetup.head, initial, config, BlockState.readMemValue, BlockState.readMemTyped, Region.cast, Int.toNat]

private theorem start_zero (evenM evenN two : Bool) (visit : Nat) :
    RealVisit.start initial (RealLoop.operands initial 2 2 1 (config evenM evenN two) visit)
      (config evenM evenN two) = 0 := by
  have hc (offset : Nat) : initial.readMemValue .int "C" offset = 0 := by
    have hcr : ("C" : RegionName) ≠ "R" := by decide
    simp [BlockState.readMemValue, BlockState.readMemTyped, initial, hcr]
  change initial.readMemValue .int "C" _ * 2 = 0
  rw [hc]
  norm_num

theorem query_bounds (evenM evenN two second : Bool) :
    RealQuery.Bounds (allocation evenM evenN) (RealSetup.queryPointers initial 2 1 (config evenM evenN two))
      (RealSetup.rows initial 2) (config evenM evenN two).context evenM second := by
  intro r d ha
  have hd : d.val = 0 := by omega
  have hr := r.isLt
  cases evenM <;> cases second <;>
    simp [RealQuery.active, RealQuery.address, RealSetup.queryPointers, RealSetup.qBase, RealSetup.rows,
      RealSetup.batch, RealSetup.head, initial, config, hd] at ha ⊢ <;> omega

theorem key_bounds (evenM evenN two second : Bool) :
    RealKey.Bounds (allocation evenM evenN) (RealSetup.keyPointers initial 2 1 (config evenM evenN two))
      0 (config evenM evenN two).kRow (config evenM evenN two).context evenN second := by
  intro d j ha
  have hd : d.val = 0 := by omega
  have hj := j.isLt
  cases evenN <;> cases second <;>
    simp [RealKey.active, RealKey.address, RealSetup.keyPointers, RealSetup.kBase,
      RealSetup.batch, RealSetup.head, RealSetup.keyHead, initial, config, hd] at ha ⊢ <;> omega

theorem value_bounds (evenM evenN two second : Bool) :
    RealValueSafety.Bounds (allocation evenM evenN) (RealSetup.valuePointers initial 2 1 (config evenM evenN two))
      0 (config evenM evenN two).vRow (config evenM evenN two).context evenN second := by
  intro j d ha
  have hd : d.val = 0 := by omega
  have hj := j.isLt
  cases evenN <;> cases second <;>
    simp [RealValueLoad.active, RealValueLoad.address, RealSetup.valuePointers, RealSetup.vBase,
      RealSetup.batch, RealSetup.head, RealSetup.keyHead, initial, config, hd] at ha ⊢ <;> omega

theorem output_bounds (evenM evenN two second : Bool) :
    RealStores.Bounds (allocation evenM evenN) initial 2 1 (config evenM evenN two) second := by
  intro r d ha
  have hd : d.val = 0 := by omega
  have hr : r.val = 0 := by simp [RealSetup.rows, initial, config] at ha; omega
  cases second <;>
    norm_num [RealStores.offset, RealSetup.rows, RealSetup.batch, RealSetup.head, initial, config, hd, hr]

theorem full_loop_bounds (evenM evenN two : Bool) :
    RealLoopSafety.Bounds (allocation evenM evenN) initial 2 2 1 (config evenM evenN two) := by
  constructor
  · intro visit hv
    rw [(interval evenM evenN two).2.2] at hv
    simp only [(interval evenM evenN two).1]
    norm_num [RealColumn.address, RealSetup.head, config, initial, Region.cast]
    omega
  · intro visit _
    rw [start_zero]
    exact key_bounds evenM evenN two Bool.false
  · intro _ visit _
    rw [start_zero]
    exact key_bounds evenM evenN two Bool.true
  · intro visit _
    rw [start_zero]
    exact value_bounds evenM evenN two Bool.false
  · intro _ visit _
    rw [start_zero]
    exact value_bounds evenM evenN two Bool.true

theorem complete_source_safe (swapped evenM evenN two : Bool) :
    Stmt.TraceSafeList (allocation evenM evenN)
      (RealProgram.kernel swapped 2 2 1 (config evenM evenN two)).toAlgKernel.body initial := by
  apply RealLoopSafety.source_safe _ _ _ _ _ _ _ (by decide) (full_loop_bounds ..)
  · exact query_bounds evenM evenN two Bool.false
  · exact fun _ => query_bounds evenM evenN two Bool.true
  · norm_num [RealRows.address, RealSetup.head, config, initial, Region.cast]
  · exact output_bounds evenM evenN two Bool.false
  · exact fun _ => output_bounds evenM evenN two Bool.true

/-- Removing the K mask without increasing its allocation violates the
second lane's bound, even for the first channel. -/
theorem unmasked_key_needs_padding (evenM two : Bool) :
    ¬ RealKey.Bounds (allocation evenM Bool.false)
      (RealSetup.keyPointers initial 2 1 (config evenM Bool.true two))
      0 (config evenM Bool.true two).kRow (config evenM Bool.true two).context Bool.true Bool.false := by
  intro hb
  have h := hb 0 1 (Or.inl rfl)
  norm_num [RealKey.address, RealSetup.keyPointers, RealSetup.kBase, RealSetup.batch,
    RealSetup.head, RealSetup.keyHead, initial, config] at h

#guard_msgs (drop info) in
#auditModuleAxioms
end
end BlockSparseSafetyTests
