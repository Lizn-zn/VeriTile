import bench.tritonbench_g.bmm_chunk_bwd.Proofs.RealDot

/-! The second contraction block has only one valid lane, and only one of
two output columns is live. Inactive memory and undef are deliberately large;
both loads must produce zero there before the real dot is taken. -/
namespace BmmChunkBwdLoadsTests
open VeriTile.Triton VeriTile.Bench.TritonBenchG.BmmChunkBwd
noncomputable section

private def dp : Tile .ptr [1, 2] := Tile.mat fun _ k => ("Dout", 168 + 4 * k.val)
private def ap : Tile .ptr [2, 2] := Tile.mat fun k j => ("A", 166 + 10 * k.val + j.val)
private def rows : Fin 1 → Nat := fun _ => 0
private def columns : Fin 2 → Nat := fun j => 2 + j.val

private def initial : BlockState :=
  let s : BlockState := {
    mem := fun r o => MemCell.real (if r = "Dout" ∧ o = 168 then 13
      else if r = "A" ∧ o = 166 then 5 else 1000)
    undef := fun _ _ => 2000
    regs := fun _ _ _ => none
    pids := fun _ => 0
    numPids := fun _ => 1 }
  let s := s.setReg "dout_ptrs" .ptr [1, 2] dp
  let s := s.setReg "a_ptrs" .ptr [2, 2] ap
  let s := s.setReg "offs_m" .nat [1] (Tile.vec rows)
  let s := s.setReg "offs_n" .nat [2] (Tile.vec columns)
  let s := s.setReg "offs_cs" .nat [2] (Tile.vec Fin.val)
  let s := s.setReg "chunk_size_limit" .nat [] (Tile.scalar 3)
  s.setReg "cs" .nat [] (Tile.scalar 1)

theorem dout_actual_load : evalOp (RealLoads.doutLoad 1 2 4) initial =
    some (RealLoads.doutValues initial dp rows 4 3 1) := by
  apply RealLoads.dout_load
  all_goals simp [initial]

theorem a_actual_load : evalOp (RealLoads.aLoad 2 2 3) initial =
    some (RealLoads.aValues initial ap columns 3 3 1) := by
  apply RealLoads.a_load
  all_goals simp [initial]

theorem load_values_and_masks :
    (RealLoads.doutValues initial dp rows 4 3 1).data (0,0,PUnit.unit) = some 13 ∧
    (RealLoads.doutValues initial dp rows 4 3 1).data (0,1,PUnit.unit) = some 0 ∧
    (RealLoads.aValues initial ap columns 3 3 1).data (0,0,PUnit.unit) = some 5 ∧
    (RealLoads.aValues initial ap columns 3 3 1).data (0,1,PUnit.unit) = some 0 ∧
    (RealLoads.aValues initial ap columns 3 3 1).data (1,0,PUnit.unit) = some 0 := by
  norm_num [RealLoads.doutValues, RealLoads.aValues, Tile.mat,
    initial, dp, ap, rows, columns, BlockState.readMem]

theorem partial_dot_and_masked_column :
    (Tile.dot [] (RealLoads.doutValues initial dp rows 4 3 1)
      (RealLoads.aValues initial ap columns 3 3 1)).data (0,0,PUnit.unit) = some 65 ∧
    (Tile.dot [] (RealLoads.doutValues initial dp rows 4 3 1)
      (RealLoads.aValues initial ap columns 3 3 1)).data (0,1,PUnit.unit) = some 0 := by
  rw [RealDot.masked_dot]
  norm_num [RealDot.value, Tile.mat, Fin.sum_univ_two,
    initial, dp, ap, rows, columns, BlockState.readMem]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end BmmChunkBwdLoadsTests
