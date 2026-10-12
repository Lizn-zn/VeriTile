import bench.tritonbench_g.chunk_gate_recurrence.Proofs.RealBackwardSafety

/-! The reverse loop's actual memory operations, before its five register-only
pointer decrements. This part commutes strictly with flat translation; the
following tail treats final dead pointers separately. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.RealBackwardFlatCore
open VeriTile.Triton RealProgram RealBackwardProgram RealBackwardPointers RealBackwardStep
open RealForwardProgram (stride)
open RealForwardStep (load_tile load_gate)
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

def core (p : Layout) : List Stmt :=
  [.assign .real [p.BK,p.BV] "S_i" (.load .real (.ptr (.ref .ptr [p.BK,p.BV] "S")) .none),
   .assign .real [p.BK,p.BV] "DS_i" (.load .real (.ptr (.ref .ptr [p.BK,p.BV] "DS")) .none),
   .assign .real [] "d_i" (.load .real (.ptr (.ref .ptr [] "D")) .none),
   .assign .real [p.BK,p.BV] "Dacc" (update p),
   .assign .real [] "DG_i" (gradient p),
   .store .real [] (.ptr (.ref .ptr [] "DG")) (.ref .real [] "DG_i") .none,
   .store .real [p.BK,p.BV] (.ptr (.ref .ptr [p.BK,p.BV] "DI")) (.ref .real [p.BK,p.BV] "Dacc") .none]

def tail (p : Layout) : List Stmt :=
  [.assign .ptr [p.BK,p.BV] "S" (.ptrSub .scalarR (.ref .ptr [p.BK,p.BV] "S") (stride p)),
   .assign .ptr [p.BK,p.BV] "DI" (.ptrSub .scalarR (.ref .ptr [p.BK,p.BV] "DI") (stride p)),
   .assign .ptr [p.BK,p.BV] "DS" (.ptrSub .scalarR (.ref .ptr [p.BK,p.BV] "DS") (stride p)),
   .assign .ptr [] "DG" (.ptrSub .nil (.ref .ptr [] "DG") gradientStride),
   .assign .ptr [] "D" (.ptrSub .nil (.ref .ptr [] "D") (.constNat 1))]

theorem split (p : Layout) : loopBody p = core p ++ tail p := rfl

theorem executes (p : Layout) (l : Launch) (S D DI DG DS : RegionName) (a : Inputs p)
    (k : Nat) (acc : Values p) (u : BlockState) (hk : k < p.N-1)
    (hs : u.regs .ptr [p.BK,p.BV] "S" = some (backwardPointer p l S (p.N-2) k))
    (hdi : u.regs .ptr [p.BK,p.BV] "DI" = some (backwardPointer p l DI (p.N-2) k))
    (hds : u.regs .ptr [p.BK,p.BV] "DS" = some (backwardPointer p l DS (p.N-1) k))
    (hdg : u.regs .ptr [] "DG" = some (backwardGradient p l DG k))
    (hd : u.regs .ptr [] "D" = some (backwardGate p l D k))
    (ha : u.regs .real [p.BK,p.BV] "Dacc" = some (tile p acc))
    (hS : ∀ e j, u.readMem S (address p l (p.N-2-k) e.val j.val) = a.source (p.N-2-k) e j)
    (hDS : ∀ e j, u.readMem DS (address p l (p.N-1-k) e.val j.val) = a.incoming (p.N-1-k) e j)
    (hD : u.readMem D (gateAddress p l (p.N-1-k)) = a.gate (p.N-1-k)) :
    stepStmts (core p) u = some (stored p l DI DG a k acc u) := by
  have hs' := hs.trans (congrArg some (backwardPointer_live p l S (p.N-2) k (by omega)))
  have hdi' := hdi.trans (congrArg some (backwardPointer_live p l DI (p.N-2) k (by omega)))
  have hds' := hds.trans (congrArg some (backwardPointer_live p l DS (p.N-1) k (by omega)))
  have hdg' := hdg.trans (congrArg some (backwardGradient_live p l DG k (by omega)))
  have hd' := hd.trans (congrArg some (backwardGate_live p l D k (by omega)))
  unfold core
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (load_tile p l S (p.N-2-k) "S" _ u hS hs'))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (load_tile p l DS (p.N-1-k) "DS" _ _
    (by intro e j; simpa using hDS e j) (by simp [hds'])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (load_gate p l D (p.N-1-k) "D" _ _
    (by simpa using hD) (by simp [hd'])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (update_eval p a (p.N-1-k) acc _
    (by simp) (by simp) (by simp [ha])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (gradient_eval p a (p.N-2-k) (next p a (p.N-1-k) acc) _
    (by simp) (by simp)))]
  rw [stepStmts.cons_some (store_scalar DG (gradientAddress p l (p.N-2-k))
    (contraction p a (p.N-2-k) (next p a (p.N-1-k) acc)) _ (by simp [hdg']) (by simp))]
  rw [stepStmts.cons_some (store_tile p l DI (p.N-2-k) (next p a (p.N-1-k) acc) _ (by simp [hdi']) (by simp))]
  rw [stepStmts.nil]
  rfl


private theorem prefix_safe (bounds : RegionBounds) (first rest : List Stmt) (u : BlockState)
    (h : Stmt.TraceSafeList bounds (first ++ rest) u) : Stmt.TraceSafeList bounds first u := by
  induction first generalizing u with
  | nil => exact Stmt.TraceSafeList.nil_intro
  | cons st first ih =>
    rw [List.cons_append,Stmt.TraceSafeList] at h
    apply Stmt.TraceSafeList.cons_intro h.1
    intro v he
    have hh := h.2
    rw [he] at hh
    exact ih v hh

theorem safe (p : Layout) (l : Launch) (S D DI DG DL DS : RegionName)
    (bounds : RegionBounds) (hb : RealBackwardSafety.Bounds p l S D DI DG DL DS bounds)
    (k : Nat) (hk : k < p.N-1) (u : BlockState)
    (hs : u.regs .ptr [p.BK,p.BV] "S" = some (backwardPointer p l S (p.N-2) k))
    (hdi : u.regs .ptr [p.BK,p.BV] "DI" = some (backwardPointer p l DI (p.N-2) k))
    (hds : u.regs .ptr [p.BK,p.BV] "DS" = some (backwardPointer p l DS (p.N-1) k))
    (hdg : u.regs .ptr [] "DG" = some (backwardGradient p l DG k))
    (hd : u.regs .ptr [] "D" = some (backwardGate p l D k)) :
    Stmt.TraceSafeList bounds (core p) u := by
  have h := RealBackwardSafety.iteration_safe p l S D DI DG DL DS bounds hb k hk u hs hdi hds hdg hd
  rw [split] at h
  exact prefix_safe bounds (core p) (tail p) u h

theorem flatten_ok (p : Layout) : StmtList.FlattenOk (core p) := by
  simp [core,update,gradient,StmtList.FlattenOk,Stmt.FlattenOk,Op.FlattenOk,Op.FlattenOk.eq_def]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.RealBackwardFlatCore
