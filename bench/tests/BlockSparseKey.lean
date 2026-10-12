import bench.tritonbench_g.block_sparse_attn.Proofs.RealKey

/-! Nonsquare K tiles distinguish the channel and key axes. Actual source
loads are checked for both mask modes and both channel blocks, with nonzero
bases/stride and unspecified masked values. -/
namespace BlockSparseKeyTests
open VeriTile.Triton VeriTile.Bench.TritonBenchG.BlockSparseAttn
open RealNormalizationValues
noncomputable section

private def pointers : Tile .ptr [2, 3] :=
  Tile.mat fun d j => ("Keys", 41 + d.val + j.val * 7)

private def initial : BlockState :=
  let s : BlockState :=
    { mem := fun _ o => MemCell.real (o : ℝ),
      undef := fun _ o => 1000 + o,
      regs := fun _ _ _ => none, pids := fun a => a + 2, numPids := fun _ => 7 }
  let s := s.setReg "k_ptrs" .ptr [2, 3] pointers
  let s := s.setReg "start_n" .int [] (Tile.scalar 2)
  s.setReg "offs_n" .nat [3] (Tile.vec Fin.val)

private def config (p : RealProgram.Parameters) (even : Bool) : RealProgram.Parameters :=
  {p with kRow := 7, context := 4, evenN := even}

private def result (even second : Bool) : BlockState :=
  initial.setReg "k" .real [2, 3] (matrix (RealKey.loaded initial pointers 2 7 4 even second))

theorem no_preloaded_key : initial.regs .real [2, 3] "k" = none := by
  simp [initial, BlockState.setReg]

theorem first_source_executes (p : RealProgram.Parameters) (swapped even : Bool) (BM : Nat) :
    stepStmts (RealProgram.realPart (RealKey.firstSource swapped BM 3 2 (config p even))) initial =
      some (result even Bool.false) := by
  rw [RealKey.first_matches]
  apply RealKey.executes
  all_goals simp [initial]

theorem second_source_executes (p : RealProgram.Parameters) (swapped even : Bool) (BM : Nat) :
    stepStmts (RealProgram.realPart (RealKey.secondSource swapped BM 3 2 (config p even))) initial =
      some (result even Bool.true) := by
  rw [RealKey.second_matches]
  apply RealKey.executes
  all_goals simp [initial]

theorem distinct_channel_and_key_addresses :
    RealKey.address pointers 2 7 Bool.false 1 0 = (("Keys" : RegionName), 56) ∧
    RealKey.address pointers 2 7 Bool.false 0 1 = (("Keys" : RegionName), 62) ∧
    RealKey.address pointers 2 7 Bool.true 1 0 = (("Keys" : RegionName), 58) := by
  decide

theorem active_key_readback (even : Bool) :
    RealKey.loaded initial pointers 2 7 4 even Bool.false 1 0 = 56 ∧
    RealKey.loaded initial pointers 2 7 4 even Bool.true 1 0 = 58 := by
  simp [RealKey.loaded, RealKey.active, RealKey.address, initial, pointers,
    BlockState.readMem, Int.toNat]

theorem masked_key_is_not_zero :
    RealKey.loaded initial pointers 2 7 4 Bool.false Bool.false 1 2 = 1070 ∧
    RealKey.loaded initial pointers 2 7 4 Bool.false Bool.true 1 2 = 1072 := by
  norm_num [RealKey.loaded, RealKey.active, RealKey.address, initial, pointers,
    BlockState.setReg, Int.toNat]

theorem even_mode_reads_last_key :
    RealKey.loaded initial pointers 2 7 4 Bool.true Bool.false 1 2 = 70 ∧
    RealKey.loaded initial pointers 2 7 4 Bool.true Bool.true 1 2 = 72 := by
  norm_num [RealKey.loaded, RealKey.active, RealKey.address, initial, pointers,
    BlockState.readMem, Int.toNat]

theorem complete_state_frame (even second : Bool) :
    {result even second with regs := initial.regs} = initial := rfl

/-- With the upper-bound mask, only keys 2 and 3 need an allocation. The
second channel requires two additional cells; masked key 4 is out of bounds. -/
theorem active_bounds (second : Bool) :
    RealKey.Bounds (fun _ => if second then 66 else 64) pointers 2 7 4 Bool.false second := by
  intro d j ha
  have hj : j.val < 2 := by
    simp [RealKey.active] at ha
    omega
  have hd := d.isLt
  cases second <;> simp [RealKey.address, pointers, Int.toNat] <;> omega

theorem actual_load_safe (second : Bool) :
    Stmt.TraceSafeList (fun _ => if second then 66 else 64) (RealKey.code 3 2 7 4 Bool.false second) initial := by
  apply RealKey.code_safe (ptrs := pointers) (start := 2)
  · simp [initial]
  · simp [initial]
  · simp [initial]
  · exact active_bounds second

theorem first_channel_bound_does_not_cover_second :
    ¬ RealKey.Bounds (fun _ => 64) pointers 2 7 4 Bool.false Bool.true := by
  intro h
  have hb := h 1 1 (by norm_num [RealKey.active])
  norm_num [RealKey.address, pointers, Int.toNat] at hb

#guard_msgs (drop info) in
#auditModuleAxioms
end
end BlockSparseKeyTests
