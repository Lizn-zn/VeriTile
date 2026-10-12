import bench.tritonbench_g.chunk_gated_attention.Proofs.RealHInit

/-! Optional final-state store, with its source mask, contiguous state layout
and full-cell frame. Disabling it leaves memory unchanged. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGatedAttention.RealHFinal
open VeriTile.Triton RealHMath RealHProgram RealHPointers RealHTiles
noncomputable section
set_option maxHeartbeats 400000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

def base (p : Layout) (l : Launch) : Nat := l.head*p.K*p.V

def output (p : Layout) (l : Launch) (r : Regions) (storeFinal : Bool)
    (f : StateValue p) (u : BlockState) : BlockState :=
  if storeFinal then RealHStore.scatter p l r.final (base p l) p.V 1 f
    (u.setReg "p_h" .blockPtr [p.BK,p.BV] (statePointer p l r.final))
  else u

def footprint (p : Layout) (l : Launch) (r : Regions) (storeFinal : Bool) : WriteFootprint :=
  fun cell => storeFinal = Bool.true ∧ RealHStore.footprint p l r.final (base p l) p.V 1 cell

theorem executes (p : Layout) (l : Launch) (r : Regions) (storeFinal : Bool)
    (f : StateValue p) (u : BlockState)
    (hik : u.regs .nat [] "i_k" = some (Tile.scalar l.key))
    (hiv : u.regs .nat [] "i_v" = some (Tile.scalar l.value))
    (hih : u.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (hh : u.regs .real [p.BK,p.BV] "b_h" = some (stateTile p f)) :
    stepStmts (finalization p r storeFinal) u = some (output p l r storeFinal f u) := by
  unfold finalization
  cases storeFinal
  · rw [stepStmts.cons_some (RealHInit.if_false _ _),stepStmts.nil]
    rfl
  · rw [stepStmts.cons_some (st := .ifThen (.constBool Bool.true) _) (by
      rw [RealHInit.if_true]
      rw [stepStmts.cons_some (stepStmt_assign_eq_some (state_eval p l r.final u hih hik hiv))]
      erw [stepStmts.cons_some (RealHStore.store_eval p l r.final (base p l) p.V 1 f "p_h" _
        (by simp [hh]) (by simp; rfl)),stepStmts.nil])]
    rw [stepStmts.nil]
    rfl

theorem readback (p : Layout) (l : Launch) (r : Regions) (f : StateValue p) (u : BlockState)
    (hinj : RealHStore.NoCollision p l (base p l) p.V 1)
    (e : Fin p.BK) (j : Fin p.BV) (ha : RealHStore.active p l e.val j.val) :
    (output p l r Bool.true f u).readMem r.final
      (stateAddress p l (keyIndex p l e.val) (valueIndex p l j.val)) = f e j := by
  simpa [output,RealHStore.address,base,stateAddress] using
    RealHStore.readback p l r.final (base p l) p.V 1 f _ hinj e j ha

theorem frame (p : Layout) (l : Launch) (r : Regions) (storeFinal : Bool)
    (f : StateValue p) (u : BlockState) :
    BlockState.WriteWithin (footprint p l r storeFinal) u (output p l r storeFinal f u) := by
  cases storeFinal
  · intro _ _ _; rfl
  · intro region off hn
    apply RealHStore.frame p l r.final (base p l) p.V 1 f
      (u.setReg "p_h" .blockPtr [p.BK,p.BV] (statePointer p l r.final)) region off
    intro h
    exact hn ⟨rfl,h⟩

theorem unchanged (p : Layout) (l : Launch) (r : Regions) (f : StateValue p) (u : BlockState) :
    (output p l r Bool.false f u).mem = u.mem := rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGatedAttention.RealHFinal
