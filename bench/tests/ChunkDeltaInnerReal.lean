import bench.tritonbench_g.chunk_delta_fwd.Proofs.RealValues
import bench.tritonbench_g.chunk_delta_fwd.Proofs.RealInnerSafety

/-! Regression for multiple inner blocks, nonzero outer/value offsets, incoming
accumulator 5, and simultaneous K/V/time tails. Three valid rows contribute
2 * (20 - 3 * 3) = 22 each, so the final accumulated cell is 71. Stores outside
the valid output lanes preserve their original non-real memory tags. -/
namespace VeriTile.Tests.ChunkDeltaInnerReal
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkDeltaFwd
open RealProgram
noncomputable section
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false

private abbrev p : Layout where
  keyHead := 21
  keyTime := 3
  keyDim := 1
  valueHead := 21
  valueTime := 3
  valueDim := 1
  historyHead := 18
  historyTime := 3
  H := 1
  T := 7
  K := 3
  V := 3
  BT := 4
  BC := 2
  BK := 4
  BV := 2
  NT := 2

private def l : Launch := ⟨0,1,0⟩
private def r : Regions := ⟨"k","d","v","v_new","h","h0","ht"⟩
private def a : Inputs := ⟨fun _ => 2,fun _ => 3,fun _ => 20⟩
private def frozen : StateValue p := fun e j => if e.val < 3 ∧ j.val = 0 then 1 else 0
private def seed : StateValue p := fun _ _ => 5

private def initial : BlockState :=
  let raw : BlockState := {
    mem := fun region _ =>
      if region = "k" then MemCell.real 2
      else if region = "d" then MemCell.real 3
      else if region = "v" then MemCell.real 20
      else MemCell.of .nat 99
    regs := fun _ _ _ => none
    pids := fun ax => if ax = 1 then 1 else 0 }
  let u1 := raw.setReg "i_k" .nat [] (Tile.scalar 0)
  let u2 := u1.setReg "i_v" .nat [] (Tile.scalar 1)
  let u3 := u2.setReg "i_bh" .nat [] (Tile.scalar 0)
  let u4 := u3.setReg "i_t" .nat [] (Tile.scalar 1)
  let u5 := u4.setReg "b_h" .real [p.BK,p.BV] (stateTile p frozen)
  u5.setReg "b_h_cumsum" .real [p.BK,p.BV] (stateTile p seed)

private theorem no_collision : RealInnerLoop.NoCollision p l := by
  intro t u i j _ _ he
  simp only [valueAddress, valueIndex, p, l] at he
  have hi := i.isLt
  have hj := j.isLt
  dsimp [p] at hi hj
  have heq : t = u ∧ i.val = j.val := by omega
  exact ⟨heq.1,Fin.ext heq.2⟩

private theorem entry : RealInnerLoop.Holds initial p l r a 1 frozen seed 0 initial := by
  apply RealInnerLoop.entry
  all_goals simp [initial, l, r, a, BlockState.readMem, MemCell.readAs, MemCell.real, MemCell.of]

private theorem final_acc : RealInnerLoop.accumulated p l a 1 (RealInnerLoop.count p) frozen seed 0 0 = 71 := by
  norm_num [RealInnerLoop.accumulated, RealInnerLoop.count, RealInnerLoop.contribution,
    correctedValue, keyValue, deltaValue, inputValue, activeValue, keyAddress, deltaAddress,
    valueAddress, keyIndex, valueIndex, timeBase, p, l, a, frozen, seed,
    Finset.sum_range_succ, Fin.sum_univ_succ]

private theorem row_value (t : Nat) (ht : t < 7) : correctedValue p l a frozen t 0 = 11 := by
  norm_num [correctedValue, deltaValue, inputValue, activeValue, deltaAddress, valueAddress,
    keyIndex, valueIndex, p, l, a, frozen, ht, Fin.sum_univ_succ]

/-- The composed theorem supplies an actual successful run; the mask check
compares full memory cells, including the unchanged `.nat` tag. -/
theorem multiple_blocks_and_tails : ∃ v : BlockState,
    stepStmt (RealInnerLoop.statement p r) initial = some v ∧
    (∃ tile, v.regs .real [p.BK,p.BV] "b_h_cumsum" = some tile ∧ tile.data (0,0,PUnit.unit) = some 71) ∧
    v.readMem "v_new" 14 = 11 ∧ v.readMem "v_new" 17 = 11 ∧ v.readMem "v_new" 20 = 11 ∧
    v.mem "v_new" 15 = MemCell.of .nat 99 := by
  obtain ⟨v,he,h⟩ := RealInnerLoop.executes initial p l r a 1 frozen seed (by rfl)
    (by decide) (by decide) (by decide) no_collision initial entry
  refine ⟨v,he,⟨_,h.acc,?_⟩,?_,?_,?_,?_⟩
  · change some (RealInnerLoop.accumulated p l a 1 (RealInnerLoop.count p) frozen seed 0 0) = some 71
    rw [final_acc]
  · exact (h.output 4 0 (by decide) (by decide) (by decide)).trans (row_value 4 (by decide))
  · exact (h.output 5 0 (by decide) (by decide) (by decide)).trans (row_value 5 (by decide))
  · exact (h.output 6 0 (by decide) (by decide) (by decide)).trans (row_value 6 (by decide))
  · have hf := h.frame "v_new" 15 (by
      rintro ⟨_,t,j,ha,_,_,hoff⟩
      dsimp [activeValue, valueAddress, valueIndex, p, l] at ha hoff
      omega)
    exact hf.symm

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Tests.ChunkDeltaInnerReal
