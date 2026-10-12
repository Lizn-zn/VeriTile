import bench.tritonbench_g.diag_ssm_triton.Proofs.ZeroTime
import VeriTile.Triton.Memory.TraceProofs
namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton.ZeroSafety
open VeriTile.Triton TypedSource
noncomputable section
set_option maxHeartbeats 600000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

/-- Bounds for the static input loads that remain at zero sequence length. -/
def forwardBounds (bounds : RegionBounds) (u : BlockState)
    (s l : RegionName) (batch dim block : Nat) : Prop :=
  (∀ j : Fin block, u.pids 0*block+j.val < batch*dim →
    u.pids 0*block+j.val < bounds s) ∧
  (∀ j : Fin block, u.pids 0*block+j.val < batch*dim →
    (u.pids 0*block+j.val)%dim < bounds l)

theorem forward_safe (bounds : RegionBounds) (u : BlockState)
    (s x l y : RegionName) (batch dim block : Nat)
    (hb : forwardBounds bounds u s l batch dim block) :
    (Erasure.forward Bool.false s x l y 0 batch dim block).toAlgKernel.TraceSafe bounds u := by
  simp only [Erasure.forward,ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,ComputeKernel.surfaceBody,
        ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,Except.bind,bind,ite_false,Bool.false_eq_true,↓reduceIte]
  simp only [Kernel.TraceSafe,Stmt.TraceSafeList,Stmt.TraceSafe,Stmt.forLoopTraceSafe,
    Op.SafeAt.eq_def,MemAccess.SafeAt,MaskOpt.SafeAt,MemAccess.ActiveAddressSafe,
    memAccessActiveAddressSafe,MaskOpt.Active,
    stepStmts,stepStmt,evalOp,evalOp.eq_def,Option.bind,bind,pure,
    Option.map,Option.pure_def,Broadcast.leftIndex,Broadcast.rightIndex,Tile.vec,
    String.reduceEq,FlatAlloc.shiftBroadcast,
    Tile.bop,Tile.cop,Tile.scalar,NumericDType.mul,NumericDType.add,
    IntegralDType.mod,ComparableDType.lt,stepForLoopAux.forLoop_unfold,
    stepForLoopAux.step_ge,Region.cast,BlockState.setReg_same,
    BlockState.setReg_ne_name,BlockState.setReg_pids,BlockState.setReg_readMem,
    BlockState.setReg_readMemValue,Option.bind_some,Option.map_some,
    Bool.true_eq_false,↓reduceIte,ne_eq,not_false_eq_true,Nat.le_refl,Nat.not_lt_zero,False.elim,forall_const,true_and,and_true]
  simp only [Option.some.injEq,forall_eq_apply_imp_iff,exists_eq_left,
    decide_eq_true_eq] 
  constructor
  · intro offsets hoff i ⟨masks,hm,hact⟩
    subst offsets; subst masks
    exact hb.1 i.1 (of_decide_eq_true hact)
  · intro offsets hoff i ⟨masks,hm,hact⟩
    subst offsets; subst masks
    exact hb.2 i.1 (of_decide_eq_true hact)


/-- The final component address bounds both interleaved components. -/
def complexForwardBounds (bounds : RegionBounds) (u : BlockState)
    (s l : RegionName) (batch dim block : Nat) : Prop :=
  (∀ j : Fin block, u.pids 0*block+j.val < batch*dim →
    (u.pids 0*block+j.val)*2+1 < bounds s) ∧
  (∀ j : Fin block, u.pids 0*block+j.val < batch*dim →
    ((u.pids 0*block+j.val)%dim)*2+1 < bounds l)

theorem complex_forward_safe (bounds : RegionBounds) (u : BlockState)
    (s x y l : RegionName) (batch dim block : Nat)
    (hb : complexForwardBounds bounds u s l batch dim block) :
    (Erasure.complexForward Bool.false s x y l 0 batch dim block).toAlgKernel.TraceSafe bounds u := by
  simp only [Erasure.complexForward,ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,ComputeKernel.surfaceBody,
        ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,Except.bind,bind,ite_false,Bool.false_eq_true,↓reduceIte]
  simp only [Kernel.TraceSafe,Stmt.TraceSafeList,Stmt.TraceSafe,Stmt.forLoopTraceSafe,
    Op.SafeAt.eq_def,MemAccess.SafeAt,MaskOpt.SafeAt,MemAccess.ActiveAddressSafe,
    memAccessActiveAddressSafe,MaskOpt.Active,
    stepStmts,stepStmt,evalOp,evalOp.eq_def,Option.bind,bind,pure,
    Option.map,Option.pure_def,Broadcast.leftIndex,Broadcast.rightIndex,Tile.vec,
    String.reduceEq,FlatAlloc.shiftBroadcast,
    Tile.bop,Tile.cop,Tile.scalar,NumericDType.mul,NumericDType.add,
    IntegralDType.mod,ComparableDType.lt,stepForLoopAux.forLoop_unfold,
    stepForLoopAux.step_ge,Region.cast,BlockState.setReg_same,
    BlockState.setReg_ne_name,BlockState.setReg_pids,BlockState.setReg_readMem,
    BlockState.setReg_readMemValue,Option.bind_some,Option.map_some,
    Bool.true_eq_false,↓reduceIte,ne_eq,not_false_eq_true,Nat.le_refl,Nat.not_lt_zero,False.elim,forall_const,true_and,and_true]
  simp only [Option.some.injEq,forall_eq_apply_imp_iff,exists_eq_left,
    decide_eq_true_eq] 
  refine ⟨?_,?_,?_,?_⟩
  · intro offsets hoff i ⟨masks,hm,hact⟩
    subst offsets; subst masks
    have h := hb.1 i.1 (of_decide_eq_true hact)
    exact Nat.lt_of_le_of_lt (Nat.le_add_right _ 1) h
  · intro offsets hoff i ⟨masks,hm,hact⟩
    subst offsets; subst masks
    exact hb.1 i.1 (of_decide_eq_true hact)
  · intro offsets hoff i ⟨masks,hm,hact⟩
    subst offsets; subst masks
    have h := hb.2 i.1 (of_decide_eq_true hact)
    exact Nat.lt_of_le_of_lt (Nat.le_add_right _ 1) h
  · intro offsets hoff i ⟨masks,hm,hact⟩
    subst offsets; subst masks
    exact hb.2 i.1 (of_decide_eq_true hact)


/-- Zero-time reverse execution reads only the transition and still writes
both terminal gradients. No sequence input or grad_x window is accessed. -/
def backwardBounds (bounds : RegionBounds) (u : BlockState)
    (l gs gl : RegionName) (batch dim block : Nat) : Prop :=
  (∀ j : Fin block, u.pids 0*block+j.val < batch*dim →
    (u.pids 0*block+j.val)%dim < bounds l) ∧
  (∀ j : Fin block, u.pids 0*block+j.val < batch*dim →
    u.pids 0*block+j.val < bounds gs) ∧
  (∀ j : Fin block, u.pids 0*block+j.val < batch*dim →
    u.pids 0*block+j.val < bounds gl)

theorem backward_safe (bounds : RegionBounds) (u : BlockState)
    (s l y gs gx gl gy : RegionName) (batch dim block : Nat)
    (hb : backwardBounds bounds u l gs gl batch dim block) :
    (Erasure.backward Bool.false s l y gs gx gl gy 0 batch dim block).toAlgKernel.TraceSafe bounds u := by
  simp only [Erasure.backward,ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,ComputeKernel.surfaceBody,
        ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,Except.bind,bind,ite_false,Bool.false_eq_true,↓reduceIte]
  simp only [Kernel.TraceSafe,Stmt.TraceSafeList,Stmt.TraceSafe,Stmt.forLoopTraceSafe,
    Op.SafeAt.eq_def,MemAccess.SafeAt,MaskOpt.SafeAt,MemAccess.ActiveAddressSafe,
    memAccessActiveAddressSafe,MaskOpt.Active,
    stepStmts,stepStmt,evalOp,evalOp.eq_def,Option.bind,bind,pure,
    Option.map,Option.pure_def,Broadcast.leftIndex,Broadcast.rightIndex,Tile.vec,
    String.reduceEq,FlatAlloc.shiftBroadcast,
    Tile.bop,Tile.cop,Tile.scalar,NumericDType.mul,NumericDType.add,
    IntegralDType.mod,ComparableDType.lt,stepForLoopAux.forLoop_unfold,
    stepForLoopAux.step_ge,Region.cast,BlockState.writeMemTyped_real,BlockState.foldl_writeMem_masked_regs,BlockState.setReg_same,
    BlockState.setReg_ne_name,BlockState.setReg_pids,BlockState.setReg_readMem,
    BlockState.setReg_readMemValue,Option.bind_some,Option.map_some,
    Bool.true_eq_false,↓reduceIte,ne_eq,not_false_eq_true,Nat.le_refl,Nat.not_lt_zero,False.elim,forall_const,true_and,and_true]
  simp only [Option.some.injEq,forall_eq_apply_imp_iff,exists_eq_left,
    decide_eq_true_eq] 
  refine ⟨?_,?_,?_⟩
  · intro offsets hoff i ⟨masks,hm,hact⟩
    subst offsets; subst masks
    exact hb.1 i.1 (of_decide_eq_true hact)
  · intro offsets hoff i ⟨masks,hm,hact⟩
    subst offsets; subst masks
    exact hb.2.1 i.1 (of_decide_eq_true hact)
  · intro offsets hoff i ⟨masks,hm,hact⟩
    subst offsets; subst masks
    exact hb.2.2 i.1 (of_decide_eq_true hact)


/-- Interleaved transition loads and both terminal gradient pairs remain
active when the complex reverse loop has no iterations. -/
def complexBackwardBounds (bounds : RegionBounds) (u : BlockState)
    (l gs gl : RegionName) (batch dim block : Nat) : Prop :=
  (∀ j : Fin block, u.pids 0*block+j.val < batch*dim →
    ((u.pids 0*block+j.val)%dim)*2+1 < bounds l) ∧
  (∀ j : Fin block, u.pids 0*block+j.val < batch*dim →
    (u.pids 0*block+j.val)*2+1 < bounds gs) ∧
  (∀ j : Fin block, u.pids 0*block+j.val < batch*dim →
    (u.pids 0*block+j.val)*2+1 < bounds gl)

theorem complex_backward_safe (bounds : RegionBounds) (u : BlockState)
    (s l y gs gx gl gy : RegionName) (batch dim block : Nat)
    (hb : complexBackwardBounds bounds u l gs gl batch dim block) :
    (Erasure.complexBackward Bool.false s l y gs gx gl gy 0 batch dim block).toAlgKernel.TraceSafe bounds u := by
  simp only [Erasure.complexBackward,ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,ComputeKernel.surfaceBody,
        ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,Except.bind,bind,ite_false,Bool.false_eq_true,↓reduceIte]
  simp only [Kernel.TraceSafe,Stmt.TraceSafeList,Stmt.TraceSafe,Stmt.forLoopTraceSafe,
    Op.SafeAt.eq_def,MemAccess.SafeAt,MaskOpt.SafeAt,MemAccess.ActiveAddressSafe,
    memAccessActiveAddressSafe,MaskOpt.Active,
    stepStmts,stepStmt,evalOp,evalOp.eq_def,Option.bind,bind,pure,
    Option.map,Option.pure_def,Broadcast.leftIndex,Broadcast.rightIndex,Tile.vec,
    String.reduceEq,FlatAlloc.shiftBroadcast,
    Tile.bop,Tile.cop,Tile.scalar,NumericDType.mul,NumericDType.add,
    IntegralDType.mod,ComparableDType.lt,stepForLoopAux.forLoop_unfold,
    stepForLoopAux.step_ge,Region.cast,BlockState.writeMemTyped_real,BlockState.foldl_writeMem_masked_regs,BlockState.setReg_same,
    BlockState.setReg_ne_name,BlockState.setReg_pids,BlockState.setReg_readMem,
    BlockState.setReg_readMemValue,Option.bind_some,Option.map_some,
    Bool.true_eq_false,↓reduceIte,ne_eq,not_false_eq_true,Nat.le_refl,Nat.not_lt_zero,False.elim,forall_const,true_and,and_true]
  simp only [Option.some.injEq,forall_eq_apply_imp_iff,exists_eq_left,
    decide_eq_true_eq] 
  refine ⟨?_,?_,?_,?_,?_,?_⟩
  · intro offsets hoff i ⟨masks,hm,hact⟩
    subst offsets; subst masks
    have h := hb.1 i.1 (of_decide_eq_true hact)
    exact Nat.lt_of_le_of_lt (Nat.le_add_right _ 1) h
  · intro offsets hoff i ⟨masks,hm,hact⟩
    subst offsets; subst masks
    exact hb.1 i.1 (of_decide_eq_true hact)
  · intro offsets hoff i ⟨masks,hm,hact⟩
    subst offsets; subst masks
    have h := hb.2.1 i.1 (of_decide_eq_true hact)
    exact Nat.lt_of_le_of_lt (Nat.le_add_right _ 1) h
  · intro offsets hoff i ⟨masks,hm,hact⟩
    subst offsets; subst masks
    exact hb.2.1 i.1 (of_decide_eq_true hact)
  · intro offsets hoff i ⟨masks,hm,hact⟩
    subst offsets; subst masks
    have h := hb.2.2 i.1 (of_decide_eq_true hact)
    exact Nat.lt_of_le_of_lt (Nat.le_add_right _ 1) h
  · intro offsets hoff i ⟨masks,hm,hact⟩
    subst offsets; subst masks
    exact hb.2.2 i.1 (of_decide_eq_true hact)


/-- Commuting the pure integer index does not change the states at which
later address checks are made. -/
theorem index_trace (bounds : RegionBounds) (block : Nat) (tail : List Stmt) (u : BlockState) :
    Stmt.TraceSafeList bounds ([Candidate.index Bool.false block] ++ tail) u ↔
      Stmt.TraceSafeList bounds ([Candidate.index Bool.true block] ++ tail) u := by
  have hsafe (swapped : Bool) : (Candidate.index swapped block).TraceSafe bounds u := by
    cases swapped <;> simp [Candidate.index,Stmt.TraceSafe,Op.SafeAt.eq_def]
  simp only [List.singleton_append,Stmt.TraceSafeList,hsafe,true_and,Candidate.index_step]

theorem forward_safe_typed (storage : Storage) (swapped : Bool)
    (bounds : RegionBounds) (u : BlockState) (s x l y : RegionName)
    (batch dim block : Nat) (hb : forwardBounds bounds u s l batch dim block) :
    (TypedSource.forward storage swapped s x l y 0 batch dim block).eraseDType.toAlgKernel.TraceSafe bounds u := by
  unfold Kernel.TraceSafe
  rw [Erasure.forward_body]
  cases swapped
  · exact forward_safe bounds u s x l y batch dim block hb
  · rw [Candidate.forward_body Bool.true,← index_trace]
    rw [← Candidate.forward_body Bool.false]
    exact forward_safe bounds u s x l y batch dim block hb

theorem complex_forward_safe_typed (swapped : Bool)
    (bounds : RegionBounds) (u : BlockState) (s x y l : RegionName)
    (batch dim block : Nat) (hb : complexForwardBounds bounds u s l batch dim block) :
    (TypedSource.complexForward swapped s x y l 0 batch dim block).eraseDType.toAlgKernel.TraceSafe bounds u := by
  unfold Kernel.TraceSafe
  rw [Erasure.complexForward_body]
  cases swapped
  · exact complex_forward_safe bounds u s x y l batch dim block hb
  · rw [Candidate.complexForward_body Bool.true,← index_trace]
    rw [← Candidate.complexForward_body Bool.false]
    exact complex_forward_safe bounds u s x y l batch dim block hb

theorem backward_safe_typed (storage : Storage) (swapped : Bool)
    (bounds : RegionBounds) (u : BlockState) (s l y gs gx gl gy : RegionName)
    (batch dim block : Nat) (hb : backwardBounds bounds u l gs gl batch dim block) :
    (TypedSource.backward storage swapped s l y gs gx gl gy 0 batch dim block).eraseDType.toAlgKernel.TraceSafe bounds u := by
  unfold Kernel.TraceSafe
  rw [Erasure.backward_body]
  cases swapped
  · exact backward_safe bounds u s l y gs gx gl gy batch dim block hb
  · rw [Candidate.backward_body Bool.true,← index_trace]
    rw [← Candidate.backward_body Bool.false]
    exact backward_safe bounds u s l y gs gx gl gy batch dim block hb

theorem complex_backward_safe_typed (swapped : Bool)
    (bounds : RegionBounds) (u : BlockState) (s l y gs gx gl gy : RegionName)
    (batch dim block : Nat) (hb : complexBackwardBounds bounds u l gs gl batch dim block) :
    (TypedSource.complexBackward swapped s l y gs gx gl gy 0 batch dim block).eraseDType.toAlgKernel.TraceSafe bounds u := by
  unfold Kernel.TraceSafe
  rw [Erasure.complexBackward_body]
  cases swapped
  · exact complex_backward_safe bounds u s l y gs gx gl gy batch dim block hb
  · rw [Candidate.complexBackward_body Bool.true,← index_trace]
    rw [← Candidate.complexBackward_body Bool.false]
    exact complex_backward_safe bounds u s l y gs gx gl gy batch dim block hb

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DiagSsmTriton.ZeroSafety
