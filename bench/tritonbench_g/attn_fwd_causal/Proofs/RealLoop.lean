import bench.tritonbench_g.attn_fwd_causal.Proofs.RealHelperSource
import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealKeys

/-! Full execution of either source helper range from its loop-entry
register contract. Bounds, dimensions, strides and partial tails stay
symbolic. The final carry is the independent closed formula over every
visited tile, including zero-padded lanes and finite-penalty lanes. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealLoop
open VeriTile.Triton RealLoads RealMemory RealIteration
open AttentionFwdTriton3.RealScores (Frame)
open AttentionFwdTriton3.RealLoopFormula (Carry closed closed_zero closed_succ)
noncomputable section
set_option maxHeartbeats 1000000

def sourceBody (BM BN BD N HA HD : Nat) (second firstDiagonal : Bool) : List Stmt :=
  if second then RealBody.second BM BN BD N HA HD else RealBody.first BM BN BD N HA HD firstDiagonal

def cursor (lower BN i : Nat) : Nat := lower+i*BN

theorem cursor_succ (lower BN i : Nat) : cursor lower BN (i+1) = cursor lower BN i+BN := by
  simp [cursor, Nat.add_mul, Nat.add_assoc]

theorem cursor_lt (lower upper BN i : Nat) (hN : 0 < BN) :
    cursor lower BN i < upper ↔ i < RealFormula.tripCount lower upper BN := by
  have hc : RealFormula.tripCount lower upper BN =
      AttentionFwdTriton3.RealKeys.blockCount lower upper BN := by
    by_cases h : lower < upper
    · simp only [RealFormula.tripCount, AttentionFwdTriton3.RealKeys.blockCount, if_pos h]
      rw [show upper-lower+BN-1 = (upper-lower-1)+BN from by omega, Nat.add_div_right _ hN]
    · simp [RealFormula.tripCount, AttentionFwdTriton3.RealKeys.blockCount, h,
        Nat.sub_eq_zero_of_le (show upper ≤ lower by omega), Nat.div_eq_of_lt (show BN-1 < BN by omega)]
  simpa only [cursor, hc] using (AttentionFwdTriton3.RealKeys.iteration_lt_count lower upper BN i hN).symm

def score {BM BN BD : Nat} (xs : RealFormula.Inputs BM BN BD) (c : Config)
    (lower : Nat) (second firstDiagonal : Bool) (i : Nat) : Fin BM → Fin BN → ℝ :=
  RealFormula.logit xs c.startM c.context c.activeHead ⟨cursor lower BN i,second || firstDiagonal⟩

def value {BM BN BD : Nat} (xs : RealFormula.Inputs BM BN BD) (c : Config)
    (lower : Nat) (second firstDiagonal : Bool) (i : Nat) : TileIndex [BN,BD] → ℝ :=
  fun j => RealFormula.loadedValue xs c.context c.activeHead ⟨cursor lower BN i,second || firstDiagonal⟩ j.1 j.2.1

def after {BM BN BD : Nat} (xs : RealFormula.Inputs BM BN BD) (c : Config)
    (lower : Nat) (second firstDiagonal : Bool) (seed : Carry BM BD) (i : Nat) : Carry BM BD :=
  closed seed (score xs c lower second firstDiagonal) (fun _ => RealTiles.keep)
    (value xs c lower second firstDiagonal) i

structure Registers {BM BN BD : Nat} (xs : RealFormula.Inputs BM BN BD) (c : Config)
    (carry : Carry BM BD) (start : Nat) (second : Bool) (s : BlockState) : Prop where
  query : s.regs .real [BM,BD] "q" = some (RealTiles.query xs c.startM c.context c.activeHead)
  queryScale : s.regs .real [] "q_scale" = some (Tile.scalar (some xs.queryScale))
  rows : s.regs .nat [BM] "offs_m" = some ⟨fun i => c.startM*BM+i.1.val⟩
  keys : s.regs .nat [BN] "offs_n" = some (Tile.vec (fun i => i.val))
  carry : Carries s carry
  pointers : Pointers s c BN BD start second
  undef : ∀ r a, s.undef r a = 0

theorem source_of_flat (BM BN BD N HA HD : Nat) (second firstDiagonal : Bool) (s t : BlockState)
    (h : stepStmts (RealIteration.body BM BN BD N HA HD second (second || firstDiagonal)) s = some t) :
    stepStmts (sourceBody BM BN BD N HA HD second firstDiagonal) s = some t := by
  cases second with
  | «false» => exact RealHelperSource.first_executes BM BN BD N HA HD firstDiagonal s t h
  | «true» => exact RealHelperSource.second_executes BM BN BD N HA HD s t h

theorem invariant_step {BM BN BD : Nat} (mem s : BlockState) (c : Config)
    (lower : Nat) (second firstDiagonal : Bool) (seed : Carry BM BD) (hN : 0 < BN) (i : Nat)
    (hmem : s.mem = mem.mem)
    (hr : Registers (inputs mem c BM BN BD) c
      (after (inputs mem c BM BN BD) c lower second firstDiagonal seed i) (cursor lower BN i) second s) :
    ∃ t, stepStmts (sourceBody BM BN BD c.context c.activeHead c.headDim second firstDiagonal)
        (s.setReg "start_n" .nat [] (Tile.scalar (cursor lower BN i))) = some t ∧
      Registers (inputs mem c BM BN BD) c
        (after (inputs mem c BM BN BD) c lower second firstDiagonal seed (i+1)) (cursor lower BN (i+1)) second t ∧
      Frame (writes second) s t := by
  let si := s.setReg "start_n" .nat [] (Tile.scalar (cursor lower BN i))
  have hinputs := inputs_same_memory si mem c BM BN BD hmem
  have hc : Carries si (after (inputs mem c BM BN BD) c lower second firstDiagonal seed i) := by
    exact ⟨by simpa [si] using hr.carry.center, by simpa [si] using hr.carry.denominator,
      by simpa [si] using hr.carry.numerator⟩
  have hp : Pointers si c BN BD (cursor lower BN i) second := by
    refine ⟨?_, ?_, ?_⟩
    · cases second <;> simpa [si, keyName] using hr.pointers.key
    · cases second <;> simpa [si, scaleName] using hr.pointers.scale
    · cases second <;> simpa [si, valueName] using hr.pointers.value
  obtain ⟨t, hstep, hcarry, hptr, hf⟩ := RealIteration.executes si c BM BN BD (cursor lower BN i) hN
    second (second || firstDiagonal) (after (inputs mem c BM BN BD) c lower second firstDiagonal seed i)
    (by rw [hinputs]; simpa [si] using hr.query)
    (by rw [hinputs]; simpa [si] using hr.queryScale)
    (by simpa [si] using hr.rows) (by simpa [si] using hr.keys) (by simp [si]) hc hp
    (by simpa [si] using hr.undef)
  have hf' : Frame (writes second) s t :=
    (Frame.setReg _ s "start_n" _ _ _ (by simp [writes, RealPreparation.writes])).trans hf
  rw [hinputs] at hcarry
  refine ⟨t, source_of_flat _ _ _ _ _ _ second firstDiagonal si t hstep, ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_⟩, hf'⟩
  · exact (hf'.regs _ _ "q" (by cases second <;>
      simp [writes, RealPreparation.writes, RealPreparation.maskName, keyName, scaleName, valueName])).trans hr.query
  · exact (hf'.regs _ _ "q_scale" (by cases second <;>
      simp [writes, RealPreparation.writes, RealPreparation.maskName, keyName, scaleName, valueName])).trans hr.queryScale
  · exact (hf'.regs _ _ "offs_m" (by cases second <;>
      simp [writes, RealPreparation.writes, RealPreparation.maskName, keyName, scaleName, valueName])).trans hr.rows
  · exact (hf'.regs _ _ "offs_n" (by cases second <;>
      simp [writes, RealPreparation.writes, RealPreparation.maskName, keyName, scaleName, valueName])).trans hr.keys
  · simpa only [after, closed_succ, updated, score, value] using hcarry
  · simpa only [cursor_succ] using hptr
  · rw [hf'.undef]; exact hr.undef

theorem loop_from {BM BN BD : Nat} (mem s : BlockState) (c : Config)
    (lower upper : Nat) (second firstDiagonal : Bool) (seed : Carry BM BD) (hN : 0 < BN)
    (i : Nat) (hi : i ≤ RealFormula.tripCount lower upper BN) (hmem : s.mem = mem.mem)
    (hr : Registers (inputs mem c BM BN BD) c
      (after (inputs mem c BM BN BD) c lower second firstDiagonal seed i) (cursor lower BN i) second s) :
    ∃ t, stepForRangeAux "start_n" (cursor lower BN i) upper BN
        (sourceBody BM BN BD c.context c.activeHead c.headDim second firstDiagonal) s = some t ∧
      Registers (inputs mem c BM BN BD) c
        (after (inputs mem c BM BN BD) c lower second firstDiagonal seed (RealFormula.tripCount lower upper BN))
        (cursor lower BN (RealFormula.tripCount lower upper BN)) second t ∧ Frame (writes second) s t := by
  by_cases h : i < RealFormula.tripCount lower upper BN
  · obtain ⟨sm, hs, hnext, hf⟩ := invariant_step mem s c lower second firstDiagonal seed hN i hmem hr
    obtain ⟨t, ht, hfinal, hframe⟩ := loop_from mem sm c lower upper second firstDiagonal seed hN
      (i+1) (by omega) (hf.mem.trans hmem) hnext
    refine ⟨t, ?_, hfinal, hf.trans hframe⟩
    rw [stepForRangeAux.step_lt (by omega) ((cursor_lt lower upper BN i hN).mpr h), hs, Option.bind_some]
    simpa only [cursor_succ] using ht
  · have he : i = RealFormula.tripCount lower upper BN := by omega
    subst i
    exact ⟨s, stepForRangeAux.step_ge (by omega) (by
      have ht := cursor_lt lower upper BN (RealFormula.tripCount lower upper BN) hN; omega), hr, Frame.refl _ _⟩
termination_by RealFormula.tripCount lower upper BN-i

/-- Either full helper range, including an empty first prefix and an
unaligned diagonal lower bound, computes its independent closed carry. -/
theorem loop_executes {BM BN BD : Nat} (s : BlockState) (c : Config)
    (lower upper : Nat) (second firstDiagonal : Bool) (seed : Carry BM BD) (hN : 0 < BN)
    (hr : Registers (inputs s c BM BN BD) c seed lower second s) :
    ∃ t, stepForRangeAux "start_n" lower upper BN
        (sourceBody BM BN BD c.context c.activeHead c.headDim second firstDiagonal) s = some t ∧
      Registers (inputs s c BM BN BD) c
        (after (inputs s c BM BN BD) c lower second firstDiagonal seed (RealFormula.tripCount lower upper BN))
        (cursor lower BN (RealFormula.tripCount lower upper BN)) second t ∧ Frame (writes second) s t := by
  simpa only [cursor, Nat.zero_mul, Nat.add_zero] using
    loop_from s s c lower upper second firstDiagonal seed hN 0 (Nat.zero_le _) rfl
      (by simpa only [after, closed_zero, cursor, Nat.zero_mul, Nat.add_zero] using hr)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealLoop
