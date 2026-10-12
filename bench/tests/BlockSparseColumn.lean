import bench.tritonbench_g.block_sparse_attn.Proofs.RealColumn

/-! The real source must use the signed CSR visit in the complete layout
offset. Slot 11 and slot 13 contain different column indices to expose early
clipping of the negative visit. -/
namespace BlockSparseColumnTests
open VeriTile.Triton VeriTile.Bench.TritonBenchG.BlockSparseAttn
noncomputable section

private def initial : BlockState :=
  let s : BlockState :=
    { mem := fun _ o => MemCell.of .int (if o = 11 then 4 else 9),
      regs := fun _ _ _ => none, pids := fun a => a+2, numPids := fun _ => 7 }
  (s.setReg "layout_h" .nat [] (Tile.scalar 1)).setReg "col_idx_idx" .int [] (Tile.scalar (-2))

private def config (p : RealProgram.Parameters) : RealProgram.Parameters :=
  {p with cols := ⟨"Cols"⟩, columnStride := 13}

theorem actual_source_reads_negative_visit (p : RealProgram.Parameters) (swapped : Bool) (BM BD : Nat) :
    stepStmts (RealProgram.realPart (RealColumn.source swapped BM 2 BD (config p))) initial =
      some (RealColumn.result initial 2 ⟨"Cols"⟩ 1 13 (-2)) := by
  rw [RealColumn.matches_source]
  apply RealColumn.executes
  all_goals simp [initial]

private theorem loaded_column : RealColumn.column initial ⟨"Cols"⟩ 1 13 (-2) = 4 := by
  rfl

theorem actual_column_and_key_start :
    (RealColumn.result initial 2 ⟨"Cols"⟩ 1 13 (-2)).regs .int [] "col_idx" = some (Tile.scalar 4) ∧
    (RealColumn.result initial 2 ⟨"Cols"⟩ 1 13 (-2)).regs .int [] "start_n" = some (Tile.scalar 8) := by
  simpa only [loaded_column, show (4 : Int) * (2 : Int) = 8 by decide] using
    RealColumn.readbacks initial 2 ⟨"Cols"⟩ 1 13 (-2)

theorem complete_state_frame :
    { RealColumn.result initial 2 ⟨"Cols"⟩ 1 13 (-2) with regs := initial.regs } = initial :=
  RealColumn.state_frame ..

theorem signed_visit_is_preserved :
    (RealColumn.result initial 2 ⟨"Cols"⟩ 1 13 (-2)).regs .int [] "col_idx_idx" = some (Tile.scalar (-2)) := by
  rw [RealColumn.register_frame _ _ _ _ _ _ _ _ _ (by decide) (by decide)]
  simp [initial]

/-- The unmasked metadata address is 11, so extent 12 suffices even though
the signed visit is negative. -/
theorem actual_column_access_is_safe :
    Stmt.TraceSafeList (fun _ => 12) (RealColumn.code 2 ⟨"Cols"⟩ 13) initial := by
  apply RealColumn.code_safe (layout := 1) (visit := -2)
  · simp [initial]
  · simp [initial]
  · decide

theorem extent_eleven_is_insufficient :
    ¬ RealColumn.address 1 13 (-2) < 11 := by decide

#guard_msgs (drop info) in
#auditModuleAxioms
end
end BlockSparseColumnTests
