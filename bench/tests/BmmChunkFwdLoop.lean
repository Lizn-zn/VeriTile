import bench.tritonbench_g.bmm_chunk_fwd.Proofs.RealLoop

/-! Actual prologue, masked loads and two K-block iterations at nonzero
batch/chunk/head/tile coordinates. Padding and undef are nonzero, so missing
row/column/K masks cannot accidentally satisfy the expected zero results. -/
namespace BmmChunkFwdLoopTests
open VeriTile.Triton VeriTile.Bench.TritonBenchG.BmmChunkFwd
noncomputable section
private def parameters : RealProgram.Parameters where
  A := "A"
  B := "B"
  Out := "Out"
  Seq := ⟨"Seq"⟩
  seqlen := 7
  chunk := 4
  K := 3
  groups := 2
  aBatch := 100
  aRow := 10
  aHead := 4
  aColumn := 1
  bBatch := 120
  bRow := 11
  bHead := 5
  bColumn := 1
  oBatch := 200
  oChunk := 24
  oHead := 16
  oRow := 4
  oColumn := 1
  sBatch := 90
  sRow := 1
  causal := Bool.true
  hasSeq := Bool.true

private def initial : BlockState where
  mem := fun r o => MemCell.real (
    if r = "A" then
      if o = 144 then 2 else if o = 145 then 3 else if o = 146 then 5
      else if o = 154 then 7 else if o = 155 then 11 else if o = 156 then 13 else 1000
    else if r = "B" then
      if o = 191 then 29 else if o = 192 then 31 else if o = 193 then 37 else 1000
    else 1000)
  undef := fun _ _ => 2000
  regs := fun _ _ _ => none
  pids := fun ax => if ax = 2 then 3 else 1
  numPids := fun _ => 8

private def prefixCode (swapped : Bool) : List Stmt :=
  RealProgram.realPart (RealProgram.parts swapped 4 2 2 parameters).before ++
    [.forRangeDyn "k" (.constNat 0) (RealProgram.loopBound 3 2) (.constNat 1)
      (RealProgram.realPart (RealProgram.parts swapped 4 2 2 parameters).body)]

/-- This is a surviving diagonal tile even though its (row 0, column 2)
entry lies above the elementwise diagonal. The source has no such mask. -/
theorem diagonal_tile_is_active :
    RealSetup.active initial 4 2 parameters = Bool.true ∧
    RealSetup.rows initial 4 2 parameters 0 < RealSetup.columns initial 2 parameters 0 := by
  decide

theorem complete_contraction (swapped : Bool) :
    ∃ t, stepStmts (RealSetup.prologue 4 2 2 parameters ++ prefixCode swapped) initial = some t ∧
      ((t.regs .real [4, 2] "acc").map (fun tile => tile.data (0,0,PUnit.unit)) = some (some 336)) ∧
      ((t.regs .real [4, 2] "acc").map (fun tile => tile.data (1,0,PUnit.unit)) = some (some 1025)) ∧
      ((t.regs .real [4, 2] "acc").map (fun tile => tile.data (0,1,PUnit.unit)) = some (some 0)) ∧
      ((t.regs .real [4, 2] "acc").map (fun tile => tile.data (3,0,PUnit.unit)) = some (some 0)) ∧
      t.mem = initial.mem := by
  obtain ⟨t, he, h⟩ := RealLoop.prefix_executes swapped initial 4 2 2 parameters
  refine ⟨t, ?_, ?_, ?_, ?_, ?_, h.mem⟩
  · rw [stepStmts.append_some (RealSetup.prologue_executes initial 4 2 2 parameters)]
    exact he
  all_goals
    rw [RealLoop.full_sum initial t 4 2 2 parameters (by decide) h]
    norm_num [Tile.mat, RealLoop.product, RealSetup.rows, RealSetup.columns,
      RealSetup.tileM, RealSetup.tileN, RealSetup.tilesN, RealSetup.limit,
      RealSetup.chunk, RealSetup.head, RealSetup.aBase, RealSetup.bBase,
      initial, parameters, BlockState.readMem, Finset.sum_range_succ]
  all_goals
    try simp only [String.reduceEq, if_false]
    rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end BmmChunkFwdLoopTests
