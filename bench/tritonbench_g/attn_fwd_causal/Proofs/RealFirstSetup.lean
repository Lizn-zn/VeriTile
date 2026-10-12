import bench.tritonbench_g.attn_fwd_causal.Proofs.RealTwoLoops

/-! The first helper's local pointer copies and constant stage branch.
The prefix stage starts at zero; the diagonal stage advances those copies
to the query block while leaving the caller bases intact. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealFirstSetup
open VeriTile.Triton RealLoads
open RealIteration (Pointers)
open RealLoopFormula (firstLower firstUpper)
open AttentionFwdTriton3.RealScores (Frame)
noncomputable section
set_option maxHeartbeats 800000

def copies (BM BN BD : Nat) : List Stmt :=
  [.assign .nat [] "first_lo" (.constNat 0),
   .assign .nat [] "first_hi" (RealSetup.lowerExpr BM),
   .assign .ptr [BD,BN] "K_ptrs_first" (.ref .ptr [BD,BN] "K_ptrs"),
   .assign .ptr [] "K_scale_ptr_first" (.ref .ptr [] "K_scale_ptr"),
   .assign .ptr [BN,BD] "V_ptrs_first" (.ref .ptr [BN,BD] "V_ptrs")]

def code (BM BN BD HD : Nat) (diagonal : Bool) : List Stmt :=
  copies BM BN BD ++ [.ifThen (.constBool diagonal) (RealSetup.code BM BN BD HD Bool.false)]

theorem copies_execute (s : BlockState) (c : Config) (BM BN BD : Nat)
    (caller : RealTwoLoops.Caller s c BN BD) :
    ∃ t, stepStmts (copies BM BN BD) s = some t ∧
      t.regs .nat [] "first_lo" = some (Tile.scalar 0) ∧
      t.regs .nat [] "first_hi" = some (Tile.scalar (c.startM*BM)) ∧
      Pointers t c BN BD 0 Bool.false ∧ Frame (RealSetup.writes Bool.false) s t := by
  let sl := s.setReg "first_lo" .nat [] (Tile.scalar 0)
  let sh := sl.setReg "first_hi" .nat [] (Tile.scalar (c.startM*BM))
  let sk := sh.setReg "K_ptrs_first" .ptr [BD,BN] (keyPointer c BD BN 0)
  let ss := sk.setReg "K_scale_ptr_first" .ptr [] (keyScalePointer c BN 0)
  have hhigh : evalOp (RealSetup.lowerExpr BM) sl = some (Tile.scalar (c.startM*BM)) := by
    simp [RealSetup.lowerExpr, evalOp.eq_def, sl, caller.row]; rfl
  have hk : evalOp (.ref .ptr [BD,BN] "K_ptrs") sh = some (keyPointer c BD BN 0) := by
    simp [sh, sl, caller.key]
  have hs : evalOp (.ref .ptr [] "K_scale_ptr") sk = some (keyScalePointer c BN 0) := by
    simp [sk, sh, sl, caller.scale]
  have hv : evalOp (.ref .ptr [BN,BD] "V_ptrs") ss = some (valuePointer c BN BD 0) := by
    simp [ss, sk, sh, sl, caller.value]
  simp only [copies]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_constNat 0 s)),
    stepStmts.cons_some (stepStmt_assign_eq_some hhigh),
    stepStmts.cons_some (stepStmt_assign_eq_some hk),
    stepStmts.cons_some (stepStmt_assign_eq_some hs),
    stepStmts.cons_some (stepStmt_assign_eq_some hv), stepStmts.nil]
  refine ⟨_, rfl, by simp [ss, sk, sh, sl], by simp [ss, sk, sh],
    ⟨by simp [ss, sk, RealMemory.keyName], by simp [ss, RealMemory.scaleName], by simp [RealMemory.valueName]⟩,
    rfl, rfl, rfl, ?_⟩
  intro dt shape name hn
  simp only [RealSetup.writes, RealSetup.lowerName, RealSetup.upperName, RealMemory.keyName,
    RealMemory.scaleName, RealMemory.valueName, Bool.false_eq_true, ↓reduceIte, List.mem_cons, not_or] at hn
  simp [ss, sk, sh, sl, hn.1, hn.2.1, hn.2.2.1, hn.2.2.2.1, hn.2.2.2.2.1]

theorem executes (s : BlockState) (c : Config) (BM BN BD : Nat) (diagonal : Bool)
    (caller : RealTwoLoops.Caller s c BN BD) :
    ∃ t, stepStmts (code BM BN BD c.headDim diagonal) s = some t ∧
      t.regs .nat [] "first_lo" = some (Tile.scalar (firstLower c.startM BM diagonal)) ∧
      t.regs .nat [] "first_hi" = some (Tile.scalar (firstUpper c.startM BM diagonal)) ∧
      Pointers t c BN BD (firstLower c.startM BM diagonal) Bool.false ∧
      Frame (RealSetup.writes Bool.false) s t := by
  obtain ⟨sc, hcopy, hlo, hhi, hp, hf⟩ := copies_execute s c BM BN BD caller
  cases diagonal with
  | «false» =>
    refine ⟨sc, ?_, hlo, hhi, hp, hf⟩
    rw [code, stepStmts.append_some hcopy]
    have hskip : stepStmt (.ifThen (.constBool Bool.false) (RealSetup.code BM BN BD c.headDim Bool.false)) sc = some sc := by
      conv_lhs => unfold stepStmt
      simp only [evalOp.eq_def, bind, Option.bind, Tile.scalar, Bool.false_eq_true, ↓reduceIte]
    rw [stepStmts.cons_some hskip, stepStmts.nil]
  | «true» =>
    obtain ⟨t, ht, hl, hh, hptr, hframe⟩ := RealSetup.executes sc c BM BN BD Bool.false
      ((hf.regs _ _ "start_m" (by decide)).trans caller.row) hp.key hp.scale hp.value
    refine ⟨t, ?_, hl, hh, hptr, hf.trans hframe⟩
    rw [code, stepStmts.append_some hcopy]
    have hbranch : stepStmt (.ifThen (.constBool Bool.true) (RealSetup.code BM BN BD c.headDim Bool.false)) sc = some t := by
      conv_lhs => unfold stepStmt
      simpa only [evalOp.eq_def, bind, Option.bind, Tile.scalar, Bool.true_eq, ↓reduceIte] using ht
    rw [stepStmts.cons_some hbranch, stepStmts.nil]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealFirstSetup
