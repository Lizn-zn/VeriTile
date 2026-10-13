import bench.tritonbench_g.softmax_flaggems.Proofs.RealForwardInnerFullStep
import bench.tritonbench_g.softmax_flaggems.Proofs.RealForwardModel
import VeriTile.Triton.Semantics.ProgramRewrite

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardInner
open VeriTile Triton FullSource
set_option maxHeartbeats 1200000
set_option linter.unusedSimpArgs false

noncomputable def laneState (sb : BlockState) (X : RegionName)
    (row N B c : Nat) (i : TileIndex [B]) : WithBot ℝ × ℝ :=
  RealForwardModel.state N B i.1.val c (fun n ↦ sb.readMem X (row * N + n))

theorem laneState_zero (sb : BlockState) (X : RegionName)
    (row N B : Nat) (i : TileIndex [B]) :
    laneState sb X row N B 0 i = (⊥, 0) := by
  simp [laneState, RealForwardModel.state, RealForwardModel.values]

theorem laneState_succ (sb : BlockState) (X : RegionName)
    (row N B c : Nat) (i : TileIndex [B]) :
    laneState sb X row N B (c + 1) i =
      RealMath.maskedOnlineStep (laneState sb X row N B c i)
        (score sb X row N (c * B) B i) := by
  simpa [laneState, score] using RealForwardModel.state_succ N B i.1.val c
    (fun n ↦ sb.readMem X (row * N + n))

structure Pass1Inv (sb : BlockState) (X Y : RegionName)
    (row N B c : Nat) (st : BlockState) : Prop where
  memory : st.mem = sb.mem
  pids : st.pids = sb.pids
  numPids : st.numPids = sb.numPids
  undef : st.undef = sb.undef
  rowReg : st.regs .nat [] "pid_m" = some (Tile.scalar row)
  inputReg : st.regs .ptr [] "input_ptr" = some (Tile.scalar (X, row * N))
  outputReg : st.regs .ptr [] "output_ptr" = some (Tile.scalar (Y, row * N))
  previousReg : st.regs .nat [] "previous_multiple" = some (Tile.scalar (RealAddress.previous N B))
  maxReg : st.regs .real [B] "m" =
    some ⟨fun i ↦ (laneState sb X row N B c i).1⟩
  sumReg : st.regs .real [B] "z" =
    some ⟨fun i ↦ some (laneState sb X row N B c i).2⟩

 theorem pass1_invariant_step (sb st : BlockState) (Y X : RegionName)
    (M row N B c : Nat) (h : Pass1Inv sb X Y row N B c st) :
    ∃ next, stepStmts (pass1Body Y X M N B)
        (st.setReg "start_n" .nat [] (Tile.scalar (c * B))) = some next ∧
      Pass1Inv sb X Y row N B (c + 1) next := by
  obtain ⟨next, hexec, hmem, hpids, hnum, hundef, hout, hprev, hrow, hk, hm, hz⟩ :=
    pass1_step Y X M N B row (c * B)
      (st.setReg "start_n" .nat [] (Tile.scalar (c * B)))
      (fun i ↦ (laneState sb X row N B c i).1)
      (fun i ↦ (laneState sb X row N B c i).2)
      (by simpa using h.rowReg) (by simpa using h.inputReg)
      (by simp) (by simpa using h.maxReg) (by simpa using h.sumReg)
  refine ⟨next, hexec, ?_⟩
  refine ⟨hmem.trans h.memory, hpids.trans h.pids, hnum.trans h.numPids,
    hundef.trans h.undef, hrow, hk,
    (by simpa using hout.trans h.outputReg), (by simpa using hprev.trans h.previousReg), ?_, ?_⟩
  · rw [hm]
    apply congrArg some
    apply Tile.ext
    intro i
    dsimp only
    rw [laneState_succ]
    simp only [score, BlockState.setReg_readMem, BlockState.readMem, BlockState.setReg_mem, h.memory]
  · rw [hz]
    apply congrArg some
    apply Tile.ext
    intro i
    dsimp only
    rw [laneState_succ]
    simp only [score, BlockState.setReg_readMem, BlockState.readMem, BlockState.setReg_mem, h.memory]



 theorem full_invariant_step (sb st : BlockState) (Y X : RegionName)
    (M row N B c : Nat) (hfull : c * B + B ≤ N) (h : Pass1Inv sb X Y row N B c st) :
    ∃ next, stepStmts (fullBody Y X M N B)
        (st.setReg "start_n" .nat [] (Tile.scalar (c * B))) = some next ∧
      Pass1Inv sb X Y row N B (c + 1) next := by
  obtain ⟨next, hexec, hmem, hpids, hnum, hundef, hout, hprev, hrow, hk, hm, hz⟩ :=
    full_step Y X M N B row (c * B)
      (st.setReg "start_n" .nat [] (Tile.scalar (c * B)))
      (fun i ↦ (laneState sb X row N B c i).1)
      (fun i ↦ (laneState sb X row N B c i).2)
      (by simpa using h.rowReg) (by simpa using h.inputReg)
      (by simp) (by simpa using h.maxReg) (by simpa using h.sumReg)
  refine ⟨next, hexec, ?_⟩
  refine ⟨hmem.trans h.memory, hpids.trans h.pids, hnum.trans h.numPids,
    hundef.trans h.undef, hrow, hk,
    (by simpa using hout.trans h.outputReg), (by simpa using hprev.trans h.previousReg), ?_, ?_⟩
  · rw [hm]
    apply congrArg some
    apply Tile.ext
    intro i
    dsimp only
    rw [laneState_succ]
    rw [fullScore_eq_score _ _ _ _ _ _ hfull]
    simp only [score, BlockState.setReg_readMem, BlockState.readMem, BlockState.setReg_mem, h.memory]
  · rw [hz]
    apply congrArg some
    apply Tile.ext
    intro i
    dsimp only
    rw [laneState_succ]
    rw [fullScore_eq_score _ _ _ _ _ _ hfull]
    simp only [score, BlockState.setReg_readMem, BlockState.readMem, BlockState.setReg_mem, h.memory]





/-- Exact aligned source loop, including a zero-iteration prefix. -/
theorem range_aligned {idx : RegName} {q B : Nat} {body : List Stmt}
    {P : Nat → BlockState → Prop} (hB : 0 < B)
    (hstep : ∀ c s, c < q → P c s →
      ∃ s', stepStmts body (s.setReg idx .nat [] (Tile.scalar (c * B))) = some s' ∧ P (c + 1) s') :
    ∀ c s, c ≤ q → P c s →
      ∃ s', stepForRangeAux idx (c * B) (q * B) B body s = some s' ∧ P q s' := by
  intro c s
  induction h : q - c using Nat.strong_induction_on generalizing c s with
  | _ remaining ih =>
    intro hc hp
    by_cases he : c = q
    · subst c
      exact ⟨s, stepForRangeAux.step_ge (Nat.ne_of_gt hB) (Nat.le_refl _), hp⟩
    · have hlt : c < q := by omega
      obtain ⟨next, hexec, hn⟩ := hstep c s hlt hp
      obtain ⟨last, hr, hl⟩ := ih (q - (c + 1)) (by omega) (c + 1) next rfl (by omega) hn
      refine ⟨last, ?_, hl⟩
      rw [stepForRangeAux.step_lt (Nat.ne_of_gt hB) (Nat.mul_lt_mul_of_pos_right hlt hB), hexec]
      simpa only [Option.bind_some, Nat.add_mul, Nat.one_mul] using hr

theorem full_loop (sb st : BlockState) (Y X : RegionName) (M row N B : Nat)
    (hN : 0 < N) (hB : 0 < B) (h : Pass1Inv sb X Y row N B 0 st) :
    ∃ next, stepForRangeAux "start_n" 0 (RealAddress.previous N B) B (fullBody Y X M N B) st = some next ∧
      Pass1Inv sb X Y row N B (RealAddress.rounds N B - 1) next := by
  rw [RealAddress.previous_eq]
  suffices ∃ next, stepForRangeAux "start_n" (0 * B) ((RealAddress.rounds N B - 1) * B)
      B (fullBody Y X M N B) st = some next ∧
      Pass1Inv sb X Y row N B (RealAddress.rounds N B - 1) next by simpa using this
  apply range_aligned hB ?_ 0 st (Nat.zero_le _) h
  intro c s hc hs
  apply full_invariant_step sb s Y X M row N B c ?_ hs
  have hlast := RealAddress.previous_lt hN hB
  rw [RealAddress.previous_eq] at hlast
  have hmul := Nat.mul_le_mul_right B (show c + 1 ≤ RealAddress.rounds N B - 1 by omega)
  simp only [Nat.add_mul, Nat.one_mul] at hmul
  omega

theorem tail_loop (sb st : BlockState) (Y X : RegionName) (M row N B : Nat)
    (hN : 0 < N) (hB : 0 < B)
    (h : Pass1Inv sb X Y row N B (RealAddress.rounds N B - 1) st) :
    ∃ next, stepForRangeAux "start_n" (RealAddress.previous N B) N B (pass1Body Y X M N B) st = some next ∧
      Pass1Inv sb X Y row N B (RealAddress.rounds N B) next := by
  rw [RealAddress.previous_eq]
  exact RealLoop.range_tiles hN hB
    (fun c s _ hs ↦ pass1_invariant_step sb s Y X M row N B c hs)
    _ st (Nat.sub_le _ _) h

end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardInner
