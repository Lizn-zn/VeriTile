import bench.tritonbench_g.chunk_delta_fwd.Proofs.RealInit

/-! Both final-state switch branches. Disabled means no store; enabled writes
the final matrix through the original masked contiguous K-by-V block pointer. -/
namespace VeriTile.Bench.TritonBenchG.ChunkDeltaFwd.RealFinal
open VeriTile.Triton RealProgram
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def pointerOp (p : Layout) (out : RegionName) : Op .blockPtr [p.BK,p.BV] :=
  .makeBlockPtrDynOffsets out
    (.mul .nat .nil (.mul .nat .nil (.ref .nat [] "i_bh") (.constNat p.K)) (.constNat p.V))
    [p.K,p.V] [p.BK,p.BV] [p.V,1]
    [.mul .nat .nil (.ref .nat [] "i_k") (.constNat p.BK),
     .mul .nat .nil (.ref .nat [] "i_v") (.constNat p.BV)]

def result (p : Layout) (l : Launch) (r : Regions) (storeFinal : Bool)
    (frozen : StateValue p) (u : BlockState) : BlockState :=
  if storeFinal then
    RealStateStore.scatter p l r.final (RealInit.base p l) p.V frozen
      (u.setReg "p_ht" .blockPtr [p.BK,p.BV]
        (RealStateStore.pointer p l r.final (RealInit.base p l) p.V))
  else u

def footprint (p : Layout) (l : Launch) (r : Regions) (storeFinal : Bool) : WriteFootprint :=
  fun cell => storeFinal = Bool.true ∧ RealStateStore.footprint p l r.final (RealInit.base p l) p.V cell

theorem pointer_eval (p : Layout) (l : Launch) (out : RegionName) (u : BlockState)
    (hik : u.regs .nat [] "i_k" = some (Tile.scalar l.key))
    (hiv : u.regs .nat [] "i_v" = some (Tile.scalar l.value))
    (hih : u.regs .nat [] "i_bh" = some (Tile.scalar l.head)) :
    evalOp (pointerOp p out) u = some (RealStateStore.pointer p l out (RealInit.base p l) p.V) := by
  exact makeBlockPtr_2d_eval out u _ _ _ [p.K,p.V] [p.BK,p.BV] [p.V,1]
    (RealInit.base p l) (l.key * p.BK) (l.value * p.BV)
    (mulMulConst_eval u "i_bh" l.head p.K p.V hih)
    (mulConst_eval u "i_k" l.key p.BK hik) (mulConst_eval u "i_v" l.value p.BV hiv)

theorem executes (p : Layout) (l : Launch) (r : Regions) (storeFinal : Bool)
    (frozen : StateValue p) (u : BlockState)
    (hik : u.regs .nat [] "i_k" = some (Tile.scalar l.key))
    (hiv : u.regs .nat [] "i_v" = some (Tile.scalar l.value))
    (hih : u.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (hh : u.regs .real [p.BK,p.BV] "b_h" = some (stateTile p frozen)) :
    stepStmts (cdfEpilogue r.final p.K p.V p.BK p.BV storeFinal) u =
      some (result p l r storeFinal frozen u) := by
  unfold cdfEpilogue
  cases storeFinal
  · rw [stepStmts.cons_some (cdf_ifThen_false_noop _ _),stepStmts.nil]
    rfl
  · rw [stepStmts.cons_some (st := .ifThen (.constBool Bool.true) _) (by
      rw [cdf_ifThen_true_run]
      have hp := pointer_eval p l r.final u hik hiv hih
      unfold pointerOp at hp
      rw [stepStmts.cons_some (stepStmt_assign_eq_some hp)]
      rw [stepStmts.cons_some (RealStateStore.store_eval p l r.final (RealInit.base p l) p.V
        frozen "p_ht" _ (by simp [hh]) (by simp)),stepStmts.nil])]
    rw [stepStmts.nil]
    rfl

theorem frame (p : Layout) (l : Launch) (r : Regions) (storeFinal : Bool)
    (frozen : StateValue p) (u : BlockState) :
    BlockState.WriteWithin (footprint p l r storeFinal) u (result p l r storeFinal frozen u) := by
  cases storeFinal
  · intro _ _ _; rfl
  · have h := RealStateStore.frame p l r.final (RealInit.base p l) p.V frozen
      (u.setReg "p_ht" .blockPtr [p.BK,p.BV]
        (RealStateStore.pointer p l r.final (RealInit.base p l) p.V))
    intro rg off hn
    exact h rg off (fun hh => hn ⟨rfl,hh⟩)

theorem readback (p : Layout) (l : Launch) (r : Regions) (frozen : StateValue p) (u : BlockState)
    (hinj : RealStateStore.NoCollision p l (RealInit.base p l) p.V)
    (e : Fin p.BK) (j : Fin p.BV) (ha : RealStateStore.active p l e.val j.val) :
    (result p l r Bool.true frozen u).readMem r.final
      (RealStateStore.address p l (RealInit.base p l) p.V e.val j.val) = frozen e j :=
  RealStateStore.readback p l r.final (RealInit.base p l) p.V frozen _ hinj e j ha

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkDeltaFwd.RealFinal
