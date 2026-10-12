import bench.tritonbench_g.attn_fwd_causal.Proofs.RealIteration

/-! Transport the flat iteration execution to the two actual erased helper
bodies. The first helper's constant branch is executed explicitly; the second
always uses the diagonal finite penalty. RealBody connects these helpers to
both complete typed kernels for every valid source stage. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealHelperSource
open VeriTile.Triton RealMemory
noncomputable section

theorem first_flat (BM BN BD N HA HD : Nat) (diagonal : Bool) :
    RealIteration.body BM BN BD N HA HD Bool.false diagonal =
      (RealBody.loadPrefix "K_ptrs_first" "K_scale_ptr_first" BD BN N HA ++ [RealBody.score BM BN BD]) ++
      ((if diagonal then RealBody.diagonal BM BN "mask_first" ++ RealBody.center BM BN else RealBody.center BM BN) ++
        RealBody.suffix "K_ptrs_first" "K_scale_ptr_first" "V_ptrs_first" BM BN BD N HA HD) := by
  cases diagonal <;> simp [RealIteration.body, RealPreparation.prepare, RealPreparation.maskName,
    keyName, scaleName, valueName, RealBody.suffix, List.append_assoc]

theorem first_executes (BM BN BD N HA HD : Nat) (diagonal : Bool) (s t : BlockState)
    (h : stepStmts (RealIteration.body BM BN BD N HA HD Bool.false diagonal) s = some t) :
    stepStmts (RealBody.first BM BN BD N HA HD diagonal) s = some t := by
  rw [first_flat] at h
  obtain ⟨u, hu, hrest⟩ := stepStmts.append_some_iff.mp h
  obtain ⟨v, hv, ht⟩ := stepStmts.append_some_iff.mp hrest
  have hb : stepStmt (.ifThenElse (.constBool diagonal)
      (RealBody.diagonal BM BN "mask_first" ++ RealBody.center BM BN) (RealBody.center BM BN)) u = some v := by
    conv_lhs => unfold stepStmt
    cases diagonal <;> simpa only [evalOp.eq_def, bind, Option.bind, Tile.scalar,
      Bool.false_eq_true, Bool.true_eq, ↓reduceIte] using hv
  have htail := (stepStmts.cons_some hb).trans ht
  simpa only [RealBody.first, List.append_assoc, List.cons_append, List.nil_append] using
    (stepStmts.append_some hu).trans htail

theorem second_flat (BM BN BD N HA HD : Nat) :
    RealIteration.body BM BN BD N HA HD Bool.true Bool.true = RealBody.second BM BN BD N HA HD := by
  simp [RealIteration.body, RealPreparation.prepare, RealPreparation.maskName,
    keyName, scaleName, valueName, RealBody.second, RealBody.suffix, List.append_assoc]

theorem second_executes (BM BN BD N HA HD : Nat) (s t : BlockState)
    (h : stepStmts (RealIteration.body BM BN BD N HA HD Bool.true Bool.true) s = some t) :
    stepStmts (RealBody.second BM BN BD N HA HD) s = some t := by
  simpa only [second_flat] using h

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealHelperSource
