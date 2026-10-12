import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealMemorySafety

/-! Access safety of the complete signed loop. The invariant used for real
execution also establishes the actual K/V pointer registers at each iteration.
The bound premises concern memory addresses, not successful execution or outputs. -/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealSafetyLoop
open VeriTile.Triton RealAccess RealRegisterSafety RealMemorySafety RealLoop RealTiles
noncomputable section

theorem iteration_safe {BM BN BD : Nat} (bounds : RegionBounds) (s : BlockState)
    (p : Parameters BM BN BD) (c : RealLoopFormula.Carry BM BD) (i : Nat) (hN : 0 < BN)
    (hregs : Registers p c i s)
    (hbK : BlockBounds bounds (keyPointer p i) (RealAccumulation.checks p.even))
    (hbV : BlockBounds bounds (valuePointer p i) (RealAccumulation.checks p.even)) :
    Stmt.TraceSafeList bounds (RealBody.body BM BN BD p.context p.window p.complement p.even p.offset p.size)
      (s.setReg "start_n" .int [] (Tile.scalar (cursor p i : Int))) := by
  let sn := s.setReg "start_n" .int [] (Tile.scalar (cursor p i : Int))
  let sk := sn.setReg "k" .real [BD, BN]
    (finite (RealAccumulation.values sn (keyPointer p i) (RealAccumulation.checks p.even)))
  have hn : stepStmt (.assign .int [] "start_n" (.ref .int [] "start_n")) sn = some sn := by
    rw [stepStmt_assign_eq_some (by simp [sn] : evalOp (.ref .int [] "start_n") sn =
      some (Tile.scalar (cursor p i : Int)))]
    apply congrArg some
    refine BlockState.ext (fun _ _ => rfl) (fun dt sh name => ?_) (fun _ => rfl)
      (fun _ _ => rfl) (fun _ => rfl)
    by_cases h : name = "start_n" <;> simp [sn, BlockState.setReg, h]
  have hk : stepStmt (RealBody.loadKey BD BN p.even) sn = some sk :=
    RealAccumulation.load_key_executes sn BD BN p.even (keyPointer p i) (by simp [sn, hregs.key])
  have hstart : stepStmts [.assign .int [] "start_n" (.ref .int [] "start_n"), RealBody.loadKey BD BN p.even] sn = some sk := by
    rw [stepStmts.cons_some hn, stepStmts.cons_some hk, stepStmts.nil]
  obtain ⟨sp, ep, hmij, hp, fp⟩ := RealProbabilities.scores_probabilities_execute sk
    BM BN BD p.startM p.context p.window p.complement p.even hN (cursor p i) p.offset p.size p.q
    (RealAccumulation.values sn (keyPointer p i) (RealAccumulation.checks p.even)) p.scale c.center
    (by simp [sk, sn, hregs.q]) (by simp [sk]) (by simp [sk, sn, hregs.scale])
    (by simp [sk, sn, hregs.startM]) (by simp [sk, sn]) (by simp [sk, sn, hregs.center])
  obtain ⟨sr, er, _, _, _, _, _, _, _, fr⟩ := RealNumeric.rescale_executes sp BM BN BD
    (⟨fun j => c.center j.1⟩ : Tile .real [BM]) _ (finite (fun j : TileIndex [BM] => c.denominator j.1))
    _ (finite c.numerator)
    ((fp.regs _ _ _ (by simp)).trans (by simp [sk, sn, hregs.center])) hmij
    ((fp.regs _ _ _ (by simp)).trans (by simp [sk, sn, hregs.denominator])) hp
    ((fp.regs _ _ _ (by simp)).trans (by simp [sk, sn, hregs.numerator]))
  have hv : sr.regs .blockPtr [BN, BD] "V_block_ptr" = some (valuePointer p i) :=
    (fr _ _ _ (by simp)).trans ((fp.regs _ _ _ (by simp)).trans (by simp [sk, sn, hregs.value]))
  have hfirst : Stmt.TraceSafeList bounds
      [.assign .int [] "start_n" (.ref .int [] "start_n"), RealBody.loadKey BD BN p.even] sn := by
    apply cons_of_run (by simp [Stmt.TraceSafe, Op.SafeAt.eq_def]) hn
    apply Stmt.TraceSafeList.cons_intro (load_key_safe bounds sn BD BN p.even _ (by simp [sn, hregs.key]) hbK)
    intro _ _; exact Stmt.TraceSafeList.nil_intro
  have hmiddle : Stmt.TraceSafeList bounds
      (RealBody.scores BM BN BD p.context p.window p.complement p.even p.offset p.size ++
       RealBody.probabilities BM BN p.context p.window p.even) sk :=
    append_safe _ sk (scores_safe _ _ _ _ _ _ _ _ _ bounds sk)
      (fun t _ => probabilities_safe _ _ _ _ _ bounds t)
  have htail := append_of_run (rescale_safe BM BN BD bounds sp) er
    (accumulate_safe bounds sr BM BN BD p.even (valuePointer p i) hv hbV)
  have hall := append_of_run hfirst hstart (append_of_run hmiddle ep htail)
  simpa only [RealBody.body, List.append_assoc] using hall

theorem loop_safe_from {BM BN BD : Nat} (bounds : RegionBounds) (mem s : BlockState)
    (p : Parameters BM BN BD) (seed : RealLoopFormula.Carry BM BD) (hN : 0 < BN)
    (hbK : ∀ j, j < count p → BlockBounds bounds (keyPointer p j) (RealAccumulation.checks p.even))
    (hbV : ∀ j, j < count p → BlockBounds bounds (valuePointer p j) (RealAccumulation.checks p.even))
    (i : Nat) (hi : i ≤ count p) (hmem : s.mem = mem.mem)
    (hregs : Registers p (after mem p seed i) i s) :
    Stmt.forRangeIntTraceSafe bounds "start_n" (cursor p i) p.hi BN
      (RealBody.body BM BN BD p.context p.window p.complement p.even p.offset p.size) s := by
  rw [Stmt.forRangeIntTraceSafe, if_neg (by omega)]
  by_cases hlt : i < count p
  · rw [if_pos ((cursor_lt p hN i).mpr hlt)]
    refine ⟨iteration_safe bounds s p _ i hN hregs (hbK i hlt) (hbV i hlt), ?_⟩
    obtain ⟨t, er, ht, hf⟩ := invariant_step mem s p seed hN i hmem hregs
    rw [er]
    simpa only [cursor_succ, Nat.cast_add] using loop_safe_from bounds mem t p seed hN hbK hbV
      (i + 1) (by omega) (hf.mem.trans hmem) ht
  · rw [if_neg (by have ht := cursor_lt p hN i; omega)]
    trivial
termination_by count p - i

theorem loop_safe {BM BN BD : Nat} (bounds : RegionBounds) (s : BlockState)
    (p : Parameters BM BN BD) (seed : RealLoopFormula.Carry BM BD) (hN : 0 < BN)
    (hbK : ∀ j, j < count p → BlockBounds bounds (keyPointer p j) (RealAccumulation.checks p.even))
    (hbV : ∀ j, j < count p → BlockBounds bounds (valuePointer p j) (RealAccumulation.checks p.even))
    (hregs : Registers p seed 0 s) :
    Stmt.forRangeIntTraceSafe bounds "start_n" p.lo p.hi BN
      (RealBody.body BM BN BD p.context p.window p.complement p.even p.offset p.size) s := by
  simpa [cursor] using loop_safe_from bounds s s p seed hN hbK hbV 0 (Nat.zero_le _) rfl
    (by simpa [after, RealLoopFormula.closed_zero] using hregs)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealSafetyLoop
