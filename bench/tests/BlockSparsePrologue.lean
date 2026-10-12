import bench.tritonbench_g.block_sparse_attn.Proofs.RealPrologue

/-! Real pre-loop execution from an empty register file. Nonzero batch/head
and grouped-query mapping, distinct Q/K/V and CSR strides, a partial query
block, signed CSR endpoints and the optional second channel are exercised. -/
namespace BlockSparsePrologueTests
open VeriTile.Triton VeriTile.Bench.TritonBenchG.BlockSparseAttn
open RealNormalizationValues
noncomputable section
set_option linter.unusedSimpArgs false

private def config (evenM evenN two : Bool) : RealProgram.Parameters where
  Out := "Out"
  Q := "Queries"
  K := "Keys"
  V := "Values"
  rows := ⟨"Rows"⟩
  cols := ⟨"Cols"⟩
  rowStride := 11
  columnStride := 13
  layouts := 2
  scale := 1/2
  qBatch := 100
  qHead := 20
  qRow := 7
  kBatch := 200
  kHead := 30
  kRow := 11
  vBatch := 300
  vHead := 40
  vRow := 13
  oBatch := 400
  oHead := 50
  oRow := 17
  heads := 4
  kvHeads := 2
  context := 4
  blocks := if two then 2 else 1
  evenM := evenM
  evenN := evenN

private def initial : BlockState where
  mem := fun region offset => if region = "Rows" then
    MemCell.of .int (if offset = 12 then -2 else if offset = 13 then 1 else 9)
    else MemCell.real (offset : ℝ)
  undef := fun _ offset => 1000 + offset
  regs := fun _ _ _ => none
  pids := fun gridAxis => if gridAxis = 0 then 1 else if gridAxis = 1 then 7 else 0
  numPids := fun _ => 8

private def preloop (evenM evenN two : Bool) : BlockState :=
  RealPrologue.result initial 3 3 2 (config evenM evenN two)

theorem actual_prologue_executes (swapped evenM evenN two : Bool) :
    stepStmts (RealProgram.realPart (RealProgram.parts swapped 3 3 2 (config evenM evenN two)).before) initial =
      some (preloop evenM evenN two) := RealPrologue.source_executes ..

theorem signed_row_bounds (evenM evenN two : Bool) :
    (preloop evenM evenN two).regs .int [] "start_l" = some (Tile.scalar (-2)) ∧
    (preloop evenM evenN two).regs .int [] "end_l" = some (Tile.scalar 1) := by
  have h := RealPrologue.outputs initial 3 3 2 (config evenM evenN two)
  have hf : RealRows.bound initial (config evenM evenN two).rows
      (RealSetup.head initial (config evenM evenN two)) (config evenM evenN two).layouts
      (config evenM evenN two).rowStride (initial.pids 0) Bool.false = -2 := by
    norm_num [RealSetup.head, config, initial]
    rfl
  have hl : RealRows.bound initial (config evenM evenN two).rows
      (RealSetup.head initial (config evenM evenN two)) (config evenM evenN two).layouts
      (config evenM evenN two).rowStride (initial.pids 0) Bool.true = 1 := by
    norm_num [RealSetup.head, config, initial]
    rfl
  exact ⟨by simpa only [hf] using h.first, by simpa only [hl] using h.last⟩

theorem selected_head_and_layout (evenM evenN two : Bool) :
    (preloop evenM evenN two).regs .nat [] "off_h" = some (Tile.scalar 3) ∧
    (preloop evenM evenN two).regs .nat [] "layout_h" = some (Tile.scalar 1) := by
  have h := RealPrologue.outputs initial 3 3 2 (config evenM evenN two)
  constructor
  · simpa [RealSetup.head, config, initial] using h.queryHead
  · simpa [RealSetup.head, config, initial] using h.layout

theorem grouped_key_and_value_pointers (evenM evenN two : Bool) :
    ((preloop evenM evenN two).regs .ptr [2, 3] "k_ptrs").map (fun t => t.data (1, 2, PUnit.unit)) =
      some (("Keys" : RegionName), 253) ∧
    ((preloop evenM evenN two).regs .ptr [3, 2] "v_ptrs").map (fun t => t.data (2, 1, PUnit.unit)) =
      some (("Values" : RegionName), 367) := by
  have h := RealPrologue.outputs initial 3 3 2 (config evenM evenN two)
  rw [preloop, h.keyPointers, h.valuePointers]
  constructor <;> norm_num [RealSetup.keyPointers, RealSetup.valuePointers, RealSetup.kBase,
    RealSetup.vBase, RealSetup.keyHead, RealSetup.batch, RealSetup.head, config, initial]

theorem active_query_value (evenM evenN two : Bool) :
    ((preloop evenM evenN two).regs .real [3, 2] "q").map (fun t => t.data (0, 0, PUnit.unit)) =
      some (some 181) := by
  have h := RealPrologue.outputs initial 3 3 2 (config evenM evenN two)
  rw [preloop, h.query]
  have hqr : ("Queries" : RegionName) ≠ "Rows" := by decide
  norm_num [RealQuery.loaded, RealQuery.active, RealQuery.address, RealSetup.queryPointers,
    RealSetup.qBase, RealSetup.batch, RealSetup.head, RealSetup.rows, config, initial,
    BlockState.readMem, MemCell.real, matrix, finite, hqr]
  rfl

theorem masked_query_value (evenN two : Bool) :
    ((preloop Bool.false evenN two).regs .real [3, 2] "q").map (fun t => t.data (1, 0, PUnit.unit)) =
      some (some 1188) := by
  have h := RealPrologue.outputs initial 3 3 2 (config Bool.false evenN two)
  rw [preloop, h.query]
  norm_num [RealQuery.loaded, RealQuery.active, RealQuery.address, RealSetup.queryPointers,
    RealSetup.qBase, RealSetup.batch, RealSetup.head, RealSetup.rows, config, initial, matrix, finite]
  rfl

theorem unmasked_query_value (evenN two : Bool) :
    ((preloop Bool.true evenN two).regs .real [3, 2] "q").map (fun t => t.data (1, 0, PUnit.unit)) =
      some (some 188) := by
  have h := RealPrologue.outputs initial 3 3 2 (config Bool.true evenN two)
  rw [preloop, h.query]
  have hqr : ("Queries" : RegionName) ≠ "Rows" := by decide
  norm_num [RealQuery.loaded, RealQuery.active, RealQuery.address, RealSetup.queryPointers,
    RealSetup.qBase, RealSetup.batch, RealSetup.head, RealSetup.rows, config, initial,
    BlockState.readMem, MemCell.real, matrix, finite, hqr]
  rfl

theorem second_channel_value (evenM evenN : Bool) :
    ((preloop evenM evenN Bool.true).regs .real [3, 2] "q2").map (fun t => t.data (0, 0, PUnit.unit)) =
      some (some 183) := by
  have h := RealPrologue.outputs initial 3 3 2 (config evenM evenN Bool.true)
  rw [preloop, h.secondQuery (by simp [config])]
  have hqr : ("Queries" : RegionName) ≠ "Rows" := by decide
  norm_num [RealQuery.loaded, RealQuery.active, RealQuery.address, RealSetup.queryPointers,
    RealSetup.qBase, RealSetup.batch, RealSetup.head, RealSetup.rows, config, initial,
    BlockState.readMem, MemCell.real, matrix, finite, hqr]
  rfl

theorem single_channel_registers_absent (evenM evenN : Bool) :
    (preloop evenM evenN Bool.false).regs .real [3, 2] "q2" = none ∧
    (preloop evenM evenN Bool.false).regs .real [3, 2] "acc2" = none := by
  constructor <;> simp [preloop, RealPrologue.result, RealPrologue.queried, RealRows.result,
    RealQuery.result, RealSetup.result, config, initial]

theorem all_state_preserved (evenM evenN two : Bool) :
    {preloop evenM evenN two with regs := initial.regs} = initial := RealPrologue.state_frame ..

private def allocation (two : Bool) : RegionBounds := fun region =>
  if region = "Rows" then 14 else if two then 185 else 183

private theorem query_bounds (evenN two second : Bool) (hsecond : second = Bool.true → two = Bool.true) :
    RealQuery.Bounds (allocation two) (RealSetup.queryPointers initial 3 2 (config Bool.false evenN two))
      (RealSetup.rows initial 3) 4 Bool.false second := by
  have hqr : ("Queries" : RegionName) ≠ "Rows" := by decide
  intro r d hr
  have hrow : r.val = 0 := by
    simp [RealQuery.active, RealSetup.rows, initial] at hr
    omega
  have hd := d.isLt
  cases two <;> cases second <;>
    simp_all [RealQuery.address, RealSetup.queryPointers, RealSetup.qBase, RealSetup.head,
      RealSetup.batch, RealSetup.rows, config, initial, allocation]
  all_goals omega

theorem masked_prologue_safe (swapped evenN two : Bool) :
    Stmt.TraceSafeList (allocation two)
      (RealProgram.realPart (RealProgram.parts swapped 3 3 2 (config Bool.false evenN two)).before) initial := by
  apply RealPrologue.source_safe
  · exact query_bounds evenN two Bool.false (by simp)
  · intro hb
    have ht : two = Bool.true := by cases two <;> simp [config] at hb ⊢
    exact query_bounds evenN two Bool.true (fun _ => ht)
  · norm_num [RealRows.address, RealSetup.head, config, initial, allocation, Region.cast]

theorem first_extent_does_not_cover_second_channel (evenN : Bool) :
    ¬ RealQuery.Bounds (allocation Bool.false)
      (RealSetup.queryPointers initial 3 2 (config Bool.false evenN Bool.true))
      (RealSetup.rows initial 3) 4 Bool.false Bool.true := by
  intro h
  have hqr : ("Queries" : RegionName) ≠ "Rows" := by decide
  have hb := h 0 1 (Or.inr (by decide))
  norm_num [RealQuery.address, RealSetup.queryPointers, RealSetup.qBase, RealSetup.head,
    RealSetup.batch, RealSetup.rows, config, initial, allocation, hqr] at hb

#guard_msgs (drop info) in
#auditModuleAxioms
end
end BlockSparsePrologueTests
