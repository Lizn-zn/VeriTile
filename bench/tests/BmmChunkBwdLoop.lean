import bench.tritonbench_g.bmm_chunk_bwd.Proofs.RealLoop

/-! Execute the actual setup and two dynamic contraction iterations. The
length-three contraction has a padded lane containing 1000, and only one
output column is active. Values are derived from initial memory, not a dot
oracle or an assumed loop execution. Row 3 still participates in the dot
because its load mask uses chunk_size=4; the later store uses length=3. -/
namespace BmmChunkBwdLoopTests
open VeriTile.Triton VeriTile.Bench.TritonBenchG.BmmChunkBwd
noncomputable section

private def parameters : RealProgram.Parameters where
  A := "A"
  Dout := "Dout"
  Db := "Db"
  Res := "Res"
  seqlen := 3
  chunk := 4
  K := 1
  groups := 1
  aBatch := 0
  aRow := 1
  aHead := 0
  aColumn := 1
  dBatch := 0
  dChunk := 0
  dHead := 0
  dRow := 1
  dColumn := 4
  oBatch := 0
  oRow := 1
  oHead := 0
  oColumn := 1
  rBatch := 0
  rRow := 1
  rHead := 0
  rColumn := 1
  hasResidual := Bool.false

private def initial : BlockState where
  mem := fun r o => MemCell.real (
    if r = "Dout" then
      if o = 0 then 2 else if o = 1 then 3 else if o = 2 then 5
      else if o = 4 then 7 else if o = 5 then 11 else if o = 6 then 13 else 1000
    else if r = "A" then
      if o = 0 then 7 else if o = 1 then 11 else if o = 2 then 13 else 1000
    else 1000)
  undef := fun _ _ => 2000
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1

private def code (swapped : Bool) : List Stmt :=
  RealProgram.realPart (RealProgram.parts swapped 4 2 2 parameters).before ++
    [.forRangeDyn "cs" (.constNat 0) (RealProgram.loopBound 2) (.constNat 1)
      (RealProgram.realPart (RealProgram.parts swapped 4 2 2 parameters).body)]

theorem complete_contraction (swapped : Bool) :
    ∃ t, stepStmts (code swapped) initial = some t ∧
      ((t.regs .real [4, 2] "acc").map (fun tile => tile.data (0,0,PUnit.unit)) = some (some 112)) ∧
      ((t.regs .real [4, 2] "acc").map (fun tile => tile.data (1,0,PUnit.unit)) = some (some 339)) ∧
      ((t.regs .real [4, 2] "acc").map (fun tile => tile.data (0,1,PUnit.unit)) = some (some 0)) ∧
      ((t.regs .real [4, 2] "acc").map (fun tile => tile.data (3,0,PUnit.unit)) = some (some 31000)) ∧
      t.mem = initial.mem := by
  obtain ⟨t, he, h⟩ := RealLoop.prefix_executes swapped initial 4 2 2 parameters
  refine ⟨t, he, ?_, ?_, ?_, ?_, h.mem⟩
  all_goals
    rw [RealLoop.full_sum initial t 4 2 2 parameters (by decide) h]
    norm_num [Tile.mat, RealLoop.product, RealSetup.rows, RealSetup.columns,
      RealSetup.tileM, RealSetup.tileN, RealSetup.tilesN, RealSetup.limit,
      RealMath.chunkLimit, RealSetup.chunk, RealSetup.head,
      RealSetup.dBase, RealSetup.aBase, initial, parameters,
      BlockState.readMem, Finset.sum_range_succ]
  all_goals
    try simp only [String.reduceEq, if_false]
    first
    | rfl
    | congr 2 <;> norm_num [show (3 : Fin 4) ≠ 0 from by decide]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end BmmChunkBwdLoopTests
