import bench.tritonbench_g.block_sparse_attn.Proofs.RealStoreOutputs
import bench.tritonbench_g.block_sparse_attn.Proofs.RealStoreSafety

/-! A partial 3×2 output tile with nonzero batch/head and row stride 17.
Only row 3 is valid. Stores must preserve the tagged cells at rows 4 and 5,
even with EVEN_M=true; the optional second channel has separate bounds. -/
namespace BlockSparseStoreTests
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
  mem := fun region _ => if region = "Rows" then MemCell.of .int 0 else MemCell.of .nat 99
  undef := fun _ offset => 1000 + offset
  regs := fun _ _ _ => none
  pids := fun gridAxis => if gridAxis = 0 then 1 else if gridAxis = 1 then 7 else 0
  numPids := fun _ => 8

private def values : Inputs 3 3 2 where
  score := fun _ _ => ⊥
  maximum := fun _ => ⊥
  normalizer := fun _ => 0
  accumulator := fun r d => 71 + 10 * r.val + d.val
  secondAccumulator := fun r d => 91 + 10 * r.val + d.val

private def atStore (evenM evenN two : Bool) : BlockState :=
  ((RealPrologue.result initial 3 3 2 (config evenM evenN two)).setReg "acc" .real [3, 2]
    (matrix values.accumulator)).setReg "acc2" .real [3, 2] (matrix values.secondAccumulator)

private theorem fixed (evenM evenN two : Bool) :
    RealLoop.Fixed initial 3 3 2 (config evenM evenN two) (atStore evenM evenN two) := by
  have h := RealLoop.prologue_fixed initial 3 3 2 (config evenM evenN two)
  rcases h with ⟨hl, hk, hv, hr, hn, hq, hq2, hctx, hb, hh, hd, hf⟩
  constructor <;> simp_all [atStore, BlockState.setReg]

private theorem carries (evenM evenN two : Bool) :
    RealLoop.CarryAt values (config evenM evenN two).blocks (atStore evenM evenN two) := by
  have h := RealPrologue.outputs initial 3 3 2 (config evenM evenN two)
  constructor
  · simpa [atStore, values] using h.maximum
  · simpa [atStore, values] using h.normalizer
  · simp [atStore]
  · intro _
    simp [atStore]

private def output (evenM evenN two : Bool) : BlockState :=
  RealStores.result initial (atStore evenM evenN two) 3 3 2 (config evenM evenN two) values

theorem actual_tail (swapped evenM evenN two : Bool) :
    stepStmts (RealProgram.realPart (RealProgram.parts swapped 3 3 2 (config evenM evenN two)).after)
      (atStore evenM evenN two) = some (output evenM evenN two) :=
  RealStores.source_executes _ _ _ _ _ _ _ _ (fixed ..) (carries ..)

theorem active_cells (evenM evenN two : Bool) :
    (output evenM evenN two).mem "Out" 601 = MemCell.real 71 ∧
    (output evenM evenN two).mem "Out" 602 = MemCell.real 72 := by
  constructor
  · have h := RealStores.first_output initial (atStore evenM evenN two) 3 3 2 (config evenM evenN two)
      values 0 0 (by norm_num [RealSetup.rows, initial, config]) (by norm_num [config]) (by intro _; norm_num [config])
    norm_num [values] at h
    simpa [output, RealStores.offset, RealSetup.rows, RealSetup.batch, RealSetup.head, initial, config, values] using h
  · have h := RealStores.first_output initial (atStore evenM evenN two) 3 3 2 (config evenM evenN two)
      values 0 1 (by norm_num [RealSetup.rows, initial, config]) (by norm_num [config]) (by intro _; norm_num [config])
    norm_num [values] at h
    simpa [output, RealStores.offset, RealSetup.rows, RealSetup.batch, RealSetup.head, initial, config, values] using h

theorem second_cells (evenM evenN : Bool) :
    (output evenM evenN Bool.true).mem "Out" 603 = MemCell.real 91 ∧
    (output evenM evenN Bool.true).mem "Out" 604 = MemCell.real 92 := by
  constructor
  · have h := RealStores.second_output initial (atStore evenM evenN Bool.true) 3 3 2
      (config evenM evenN Bool.true) values 0 0 (by norm_num [RealSetup.rows, initial, config]) (by norm_num [config]) (by norm_num [config])
    norm_num [values] at h
    simpa [output, RealStores.offset, RealSetup.rows, RealSetup.batch, RealSetup.head, initial, config, values] using h
  · have h := RealStores.second_output initial (atStore evenM evenN Bool.true) 3 3 2
      (config evenM evenN Bool.true) values 0 1 (by norm_num [RealSetup.rows, initial, config]) (by norm_num [config]) (by norm_num [config])
    norm_num [values] at h
    simpa [output, RealStores.offset, RealSetup.rows, RealSetup.batch, RealSetup.head, initial, config, values] using h

theorem masked_rows_keep_tags (evenM evenN two : Bool) (address : Nat) (haddr : 618 ≤ address) :
    (output evenM evenN two).mem "Out" address = MemCell.of .nat 99 := by
  have h := RealStores.kernel_frame initial (atStore evenM evenN two) 3 3 2
    (config evenM evenN two) values (fixed ..) "Out" address
  have hnot (second : Bool) (r : Fin 3) (d : Fin 2)
      (hr : RealSetup.rows initial 3 r < (config evenM evenN two).context) :
      ¬((config evenM evenN two).Out = "Out" ∧
        RealStores.offset initial 3 2 (config evenM evenN two) second r d = address) := by
    have hd := d.isLt
    have hrv : r.val = 0 := by simp [RealSetup.rows, initial, config] at hr; omega
    cases second <;> norm_num [RealStores.offset, RealSetup.rows, RealSetup.batch, RealSetup.head,
      initial, config, hrv] <;> omega
  simpa [output, initial] using h (hnot Bool.false) (fun _ => hnot Bool.true)

theorem absent_second_keeps_tag (evenM evenN : Bool) :
    (output evenM evenN Bool.false).mem "Out" 603 = MemCell.of .nat 99 := by
  have h := RealStores.kernel_frame initial (atStore evenM evenN Bool.false) 3 3 2
    (config evenM evenN Bool.false) values (fixed ..) "Out" 603
  apply Eq.trans (h ?_ (by simp [config]))
  · rfl
  · intro r d hr he
    have hd := d.isLt
    have hrv : r.val = 0 := by simp [RealSetup.rows, initial, config] at hr; omega
    norm_num [RealStores.offset, RealSetup.rows, RealSetup.batch, RealSetup.head, initial, config, hrv] at he
    omega

private def allocation (extent : Nat) : RegionBounds := fun _ => extent

theorem output_bounds (evenM evenN two : Bool) :
    RealStores.Bounds (allocation 603) initial 3 2 (config evenM evenN two) Bool.false ∧
    RealStores.Bounds (allocation 605) initial 3 2 (config evenM evenN two) Bool.true := by
  constructor <;> intro r d hr
  all_goals
    have hd := d.isLt
    have hrv : r.val = 0 := by simp [RealSetup.rows, initial, config] at hr; omega
    norm_num [RealStores.offset, RealSetup.rows, RealSetup.batch, RealSetup.head, initial, config, hrv, allocation]
    omega

theorem first_allocation_insufficient_for_second (evenM evenN two : Bool) :
    ¬ RealStores.Bounds (allocation 603) initial 3 2 (config evenM evenN two) Bool.true := by
  intro hb
  have h := hb 0 0 (by norm_num [RealSetup.rows, initial, config])
  norm_num [RealStores.offset, RealSetup.rows, RealSetup.batch, RealSetup.head, initial, config, allocation] at h

theorem actual_tail_safe (swapped evenM evenN two : Bool) :
    Stmt.TraceSafeList (allocation 605)
      (RealProgram.realPart (RealProgram.parts swapped 3 3 2 (config evenM evenN two)).after)
      (atStore evenM evenN two) := by
  apply RealStores.source_safe _ _ _ _ _ _ _ _ values (fixed ..) (carries ..)
  · intro r d hr
    have h := (output_bounds evenM evenN two).1 r d hr
    change _ < 605
    change _ < 603 at h
    omega
  · exact fun _ => (output_bounds evenM evenN two).2

/-- Empty CSR rows still run the actual final stores, writing zero to valid
outputs and preserving masked cells. This exercises the full source assembly. -/
theorem complete_empty_source (swapped evenM evenN two : Bool) :
    ∃ t, stepStmts (RealProgram.kernel swapped 3 3 2 (config evenM evenN two)).toAlgKernel.body initial = some t ∧
      t.mem "Out" 601 = MemCell.real 0 ∧ t.mem "Out" 618 = MemCell.of .nat 99 := by
  obtain ⟨t, hf, he⟩ := RealStores.kernel_executes swapped initial 3 3 2 (config evenM evenN two) (by norm_num [config])
  refine ⟨_, he, ?_, ?_⟩
  · have hcount : RealLoop.count initial (config evenM evenN two) = 0 := by
      norm_num [RealLoop.count, RealLoop.first, RealLoop.last, RealRows.bound,
        BlockState.readMemValue, BlockState.readMemTyped, initial, config, Region.cast]
    have h := RealStores.first_output initial t 3 3 2 (config evenM evenN two)
      (RealLoop.carry initial 3 3 2 (config evenM evenN two) (RealLoop.count initial (config evenM evenN two)))
      0 0 (by norm_num [RealSetup.rows, initial, config]) (by norm_num [config]) (by intro _; norm_num [config])
    simpa [RealStores.offset, RealSetup.rows, RealSetup.batch, RealSetup.head, initial, config,
      hcount, RealLoop.carry, RealNormalizationMath.inputs, StreamingMath.accumulated] using h
  · rw [RealStores.kernel_frame initial t 3 3 2 (config evenM evenN two) _ hf "Out" 618]
    · rfl
    · intro r d hr
      have hd := d.isLt
      have hrv : r.val = 0 := by simp [RealSetup.rows, initial, config] at hr; omega
      norm_num [RealStores.offset, RealSetup.rows, RealSetup.batch, RealSetup.head, initial, config, hrv]
      omega
    · intro _ r d hr
      have hd := d.isLt
      have hrv : r.val = 0 := by simp [RealSetup.rows, initial, config] at hr; omega
      norm_num [RealStores.offset, RealSetup.rows, RealSetup.batch, RealSetup.head, initial, config, hrv]
      omega

#guard_msgs (drop info) in
#auditModuleAxioms
end
end BlockSparseStoreTests
