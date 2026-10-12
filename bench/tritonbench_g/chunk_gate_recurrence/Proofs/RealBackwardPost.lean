import bench.tritonbench_g.chunk_gate_recurrence.Proofs.RealBackwardInit

/-! Actual DL postlude after the reverse loop. It loads chunk zero, performs
the final multiply-add and stores the seed-gradient tile. It emits no DI/DG
store, including when the main loop is empty. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.RealBackwardPost
open VeriTile.Triton RealProgram RealBackwardPointers RealBackwardProgram RealBackwardLoop
open RealForwardProgram (eraseCode)
open RealBackwardInit (seedPointerOp seedPointer_eval)
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

def code (p : Layout) (DL : RegionName) : List Stmt :=
  [.assign .ptr [p.BK,p.BV] "DL" (seedPointerOp p DL),
   .assign .real [p.BK,p.BV] "DS_i" (.load .real (.ptr (.ref .ptr [p.BK,p.BV] "DS")) .none),
   .assign .real [] "d_i" (.load .real (.ptr (.ref .ptr [] "D")) .none),
   .assign .real [p.BK,p.BV] "Dacc" (update p),
   .store .real [p.BK,p.BV] (.ptr (.ref .ptr [p.BK,p.BV] "DL")) (.ref .real [p.BK,p.BV] "Dacc") .none]

theorem code_exact (p : Layout) (S D DI DG DL DS : RegionName) :
    postCode p S D DI DG DL DS = code p DL := by
  simp only [postCode, eraseCode, parts, chunk_gate_recurrence_bwd_surface,
    ComputeKernel.surfaceBody, Source.dynamicRange?, Option.map, Option.getD,
    ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
    Except.bind, bind, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType, NumericDType.eraseDType,
    List.map_cons, List.map_nil, Bool.false_eq_true, ↓reduceIte]
  simp only [Op.eraseDType.eq_def,VeriTile.Triton.eraseDType]
  rfl

def beforeStore (p : Layout) (l : Launch) (DL : RegionName) (a : Inputs p)
    (acc : Values p) (u : BlockState) : BlockState :=
  let u1 := u.setReg "DL" .ptr [p.BK,p.BV] (seedPointer p l DL)
  let u2 := u1.setReg "DS_i" .real [p.BK,p.BV] (tile p (a.incoming 0))
  let u3 := u2.setReg "d_i" .real [] (Tile.scalar (some (a.gate 0)))
  u3.setReg "Dacc" .real [p.BK,p.BV] (tile p (RealBackwardStep.next p a 0 acc))

def seedScatter (p : Layout) (l : Launch) (DL : RegionName) (f : Values p) (u : BlockState) : BlockState :=
  (TileShape.allIndices [p.BK,p.BV]).foldl (fun st i =>
    st.writeMem DL (seedAddress p l i.1.val i.2.1.val) (f i.1 i.2.1)) u

def result (p : Layout) (l : Launch) (DL : RegionName) (a : Inputs p)
    (acc : Values p) (u : BlockState) : BlockState :=
  seedScatter p l DL (RealBackwardStep.next p a 0 acc) (beforeStore p l DL a acc u)

def footprint (p : Layout) (l : Launch) (DL : RegionName) : WriteFootprint := fun cell =>
  cell.1 = DL ∧ ∃ (e : Fin p.BK) (j : Fin p.BV), cell.2 = seedAddress p l e.val j.val

theorem seedScatter_read (p : Layout) (l : Launch) (DL : RegionName) (f : Values p) (u : BlockState)
    (hi : NoCollision p l) (e : Fin p.BK) (j : Fin p.BV) :
    (seedScatter p l DL f u).readMem DL (seedAddress p l e.val j.val) = f e j := by
  apply BlockState.scatter_readback_nd (shape := [p.BK,p.BV]) u
    (fun i => seedAddress p l i.1.val i.2.1.val) (fun i => f i.1 i.2.1) ?_ (e,j,PUnit.unit)
  rintro ⟨e₁,j₁,⟨⟩⟩ ⟨e₂,j₂,⟨⟩⟩ heq
  have heq' : address p l 0 e₁.val j₁.val = address p l 0 e₂.val j₂.val := by
    dsimp [seedAddress] at heq
    dsimp [address]
    omega
  obtain ⟨_,he,hj⟩ := hi 0 0 e₁ e₂ j₁ j₂ heq'
  subst e₂; subst j₂; rfl

theorem seedScatter_frame (p : Layout) (l : Launch) (DL : RegionName) (f : Values p) (u : BlockState) :
    BlockState.WriteWithin (footprint p l DL) u (seedScatter p l DL f u) := by
  have h := BlockState.foldl_writeMem_prop_writeWithin (P := footprint p l DL) DL
    (fun i : TileIndex [p.BK,p.BV] => seedAddress p l i.1.val i.2.1.val)
    (fun i => f i.1 i.2.1) (fun _ => True) (TileShape.allIndices [p.BK,p.BV]) u
    (fun i _ _ => ⟨rfl,i.1,i.2.1,rfl⟩)
  simpa only [seedScatter,ite_true] using h

theorem store_seed (p : Layout) (l : Launch) (DL : RegionName) (f : Values p) (u : BlockState)
    (hp : u.regs .ptr [p.BK,p.BV] "DL" = some (seedPointer p l DL))
    (hf : u.regs .real [p.BK,p.BV] "Dacc" = some (tile p f)) :
    stepStmt (.store .real [p.BK,p.BV] (.ptr (.ref .ptr [p.BK,p.BV] "DL"))
      (.ref .real [p.BK,p.BV] "Dacc") .none) u = some (seedScatter p l DL f u) := by
  simp only [stepStmt,evalOp_ref,hp,hf,Option.bind_some,Option.bind_eq_bind]
  rfl

theorem executes (p : Layout) (l : Launch) (D DL DS : RegionName) (a : Inputs p)
    (acc : Values p) (u : BlockState) (h : Fixed p l u)
    (hds : u.regs .ptr [p.BK,p.BV] "DS" = some (pointer p l DS 0))
    (hd : u.regs .ptr [] "D" = some (gatePointer p l D 0))
    (ha : u.regs .real [p.BK,p.BV] "Dacc" = some (tile p acc))
    (hDS : ∀ e j, u.readMem DS (address p l 0 e.val j.val) = a.incoming 0 e j)
    (hD : u.readMem D (gateAddress p l 0) = a.gate 0) :
    stepStmts (code p DL) u = some (result p l DL a acc u) := by
  unfold code
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (seedPointer_eval p l DL u h))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (RealForwardStep.load_tile p l DS 0 "DS" _ _
    (by intro e j; simpa using hDS e j) (by simp [hds])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (RealForwardStep.load_gate p l D 0 "D" _ _
    (by simpa using hD) (by simp [hd])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (RealBackwardStep.update_eval p a 0 acc _
    (by simp) (by simp) (by simp [ha])))]
  rw [stepStmts.cons_some (store_seed p l DL (RealBackwardStep.next p a 0 acc) _ (by simp) (by simp)),stepStmts.nil]
  rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.RealBackwardPost
