import bench.tritonbench_g.chunk_bwd_dqkg.Proofs.UndefinedTailLoop
import bench.tritonbench_g.chunk_bwd_dqkg.Proofs.RealFrame
import VeriTile.Meta.Disprove

/-! The original full source can store a nonzero dq for all-zero inputs when
its unspecified value-axis tail is one. This refutes unconditional real
correctness; it is not a GPU measurement or a floating-point atom rejection. -/
namespace VeriTile.Bench.TritonBenchG.ChunkBwdDqkg.UndefinedTail
open VeriTile.Triton RealMath UndefinedTailLoop
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false

def kernel := (RealSource.source .fp32 .fp32 .fp32 .fp32 Bool.false
  "q" "k" "v" "h" "g" "do" "dh" "dq" "dk" "dg"
  256 16 256 16 256 16 1 16 16 1 16 16 16 1).eraseDType.toAlgKernel

theorem carries_match (s : BlockState) :
    (result s).regs .real [1] "b_dg_last" =
      some (cbdDgLastTile model "h" "dh" 256 16 16 16 16 16 1) ∧
    (result s).regs .real [16,16] "b_ds" =
      some (cbdDsTile model "v" "do" 256 16 16 16 16 16) ∧
    (result s).regs .real [16,16] "b_dq" =
      some (cbdDqTile model "h" "do" 256 16 256 16 16 16 16 16 16 16 1) ∧
    (result s).regs .real [16,16] "b_dk" =
      some (cbdDkTile model "v" "dh" 256 16 256 16 16 16 16 16 16 16 1) := by
  simp [result, p, wide, acc, add, block, blockSum, dqTerm, dkTerm, dsTerm,
    dgLastTerm, value, history, a, l, model, initial, BlockState.readMem,
    cbdDgLastTile, cbdDsTile, cbdDqTile, cbdDkTile, dgLastAcc, dsAcc, dqAcc, dkAcc,
    voElem, hElem]
  exact ⟨rfl, rfl, rfl⟩

/-- All-zero actual arrays; only the unspecified padded lanes supply the 15. -/
theorem source_stores_fifteen :
    ∃ sf, exec kernel initial = some sf ∧ sf.readMem "dq" 0 = 15 := by
  obtain ⟨setup, hsetup, _, _, hm, hik, hit, hib, hnb, hoi, hg, hgl,
      hdq, hdk, hds, hdgl, hdg⟩ :=
    cbdPreLoop_run "g" 16 16 16 initial (by simp [initial])
  have hu := cbdPreLoop_undef "g" 16 16 16 initial setup hsetup
  let entry := setup.setReg "i_v" .nat [] (Tile.scalar 0)
  have hi : stepStmts (RealStep.code p r) entry = some (result entry) := by
    apply UndefinedTailLoop.executes entry
    · exact hu
    · intro off; simpa [entry, initial, r, BlockState.readMem] using hm "v" off
    · intro off; simpa [entry, initial, r, BlockState.readMem] using hm "h" off
    · intro off; simpa [entry, initial, r, BlockState.readMem] using hm "do" off
    · intro off; simpa [entry, initial, r, BlockState.readMem] using hm "dh" off
    · simpa [entry, initial, l] using hik
    · simpa [entry, initial, l] using hit
    · simpa [entry, initial, l] using hib
    · simp [entry]
    · simpa [entry, p, acc, cbdZero, Tile.mat] using hdq
    · simpa [entry, p, acc, cbdZero, Tile.mat] using hdk
    · simpa [entry, p, acc, cbdZero, Tile.mat] using hds
    · simpa [entry, p, acc, cbdZero, Tile.vec] using hdgl
  have hloop : stepStmt (RealLoop.statement p r) setup = some (result entry) := by
    simp only [RealLoop.statement, stepForRangeAux.forRangeDyn_unfold,
      RealLoop.stop_eval, evalOp_constNat, bind, Option.bind_some, Tile.scalar]
    change stepForRangeAux "i_v" 0 1 1 (RealStep.code p r) setup = _
    change stepStmts (RealStep.code p r) (setup.setReg "i_v" .nat [] (Tile.scalar 0)) = _ at hi
    rw [stepForRangeAux]
    simp only [Nat.reduceLT, ↓reduceIte, hi, bind, Option.bind_some]
    rw [stepForRangeAux]
    rfl
  have hinj : ∀ i j : TileIndex [16,16],
      (model.pids 1 * 16 + i.1.val < 16 ∧ model.pids 0 * 16 + i.2.1.val < 16) →
      (model.pids 1 * 16 + j.1.val < 16 ∧ model.pids 0 * 16 + j.2.1.val < 16) →
      cbdQkAddr model 256 16 16 16 i = cbdQkAddr model 256 16 16 16 j → i = j := by
    intro i j _ _ he
    apply RealFrame.qk_injective_on p l (by decide) i j
    · simp [RealFrame.qkActive, p, l]
    · simp [RealFrame.qkActive, p, l]
    · simpa [RealFrame.qkAddress, cbdQkAddr, p, l, model, initial] using he
  obtain ⟨hgCarry, hsCarry, hqCarry, hkCarry⟩ := carries_match entry
  obtain ⟨sf, hpost, hq, _, _⟩ := cbdPostLoop_run
    "q" "k" "dq" "dk" "dg" "g" "v" "h" "do" "dh" model (result entry)
    256 16 256 16 256 16 1 16 16 16 16 16 16 1
    (by simpa only [result, entry, BlockState.setReg] using
      (show setup.undef "q" = (fun _ => 0) ∧ setup.undef "k" = (fun _ => 0) by simp [hu, initial]))
    (fun off => by simpa [result, entry, model, initial, BlockState.readMem] using hm "q" off)
    (fun off => by simpa [result, entry, model, initial, BlockState.readMem] using hm "k" off)
    (by decide) (by decide) (by decide) hinj
    (by simpa [result, entry, model] using hik)
    (by simpa [result, entry, model] using hit)
    (by simpa [result, entry, model] using hib)
    (by simpa [result, entry, model] using hnb)
    (by simpa [result, entry] using hoi)
    (by simpa [result, entry, cbdBgTile, model, initial, BlockState.readMem] using hg)
    (by simpa [result, entry, cbdBgLastTile, model, initial, BlockState.readMem] using hgl)
    (by simpa [result, entry] using hdg)
    hgCarry hsCarry hqCarry hkCarry
  refine ⟨sf, ?_, ?_⟩
  · rw [kernel, exec, RealSource.real_body_matches, cbd_body_eq, List.append_assoc,
      stepStmts.append_some hsetup]
    change stepStmts (RealLoop.statement p r :: cbdPostLoop "q" "k" "dq" "dk" "dg"
      256 16 1 16 16 16 16) setup = _
    rw [stepStmts.cons_some hloop]
    exact hpost
  · have hout := hq (0,0,PUnit.unit) (by decide)
    norm_num [cbdQkAddr, dqSpec, dqGated, dqAcc, qkElem, voElem, hElem, gElem,
      model, initial, BlockState.readMem, sample, Fin.sum_univ_succ] at hout
    exact hout

/-- The independent formula sums only the one actual value coordinate. -/
theorem mathematical_value :
    RealMath.dq p (RealLoop.launch initial)
      (RealPost.inputs initial "q" "k" "v" "h" "g" "do" "dh") ⟨0,by decide⟩ ⟨0,by decide⟩ = 0 := by
  simp [RealMath.dq, gatedDq, total, dqTerm, score, dsTerm, value, history,
    queryKey, gate, RealPost.inputs, p, initial, RealLoop.launch, BlockState.readMem]

/-- The full source, its actual initial arrays and its output disagree if the
zero-unspecified-value condition is omitted. -/
refutation dq_without_zero_padding :
    ∀ s sf : BlockState, exec kernel s = some sf →
      sf.readMem "dq" 0 = RealMath.dq p (RealLoop.launch s)
        (RealPost.inputs s "q" "k" "v" "h" "g" "do" "dh") ⟨0,by decide⟩ ⟨0,by decide⟩ := by
  intro h
  obtain ⟨sf, he, hout⟩ := source_stores_fifteen
  have bad := h initial sf he
  rw [hout, mathematical_value] at bad
  norm_num at bad

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkBwdDqkg.UndefinedTail
