import bench.tritonbench_g.chunk_gate_recurrence.Proofs.RealBackwardProgram
import bench.tritonbench_g.chunk_gate_recurrence.Proofs.RealBackwardPointers
import bench.tritonbench_g.chunk_gate_recurrence.Proofs.RealForwardStep
import VeriTile.Triton.Semantics.WholeReduction

/-! Execution of all twelve reverse-loop statements. The scalar DG store is
proved to be the complete tile contraction, not an abstract reduction witness.
The incoming carry is arbitrary here and is derived by the loop invariant. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.RealBackwardStep
open VeriTile.Triton RealProgram RealBackwardProgram RealBackwardPointers
open RealForwardProgram (stride)
open RealForwardStep (load_tile load_gate)
open scoped BigOperators
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

def next (p : Layout) (a : Inputs p) (q : Nat) (acc : Values p) : Values p :=
  fun e j => acc e j * a.gate q + a.incoming q e j

def contraction (p : Layout) (a : Inputs p) (c : Nat) (f : Values p) : ℝ :=
  ∑ e, ∑ j, f e j * a.source c e j

def beforeStore (p : Layout) (a : Inputs p) (k : Nat) (acc : Values p) (u : BlockState) : BlockState :=
  let u1 := u.setReg "S_i" .real [p.BK,p.BV] (tile p (a.source (p.N-2-k)))
  let u2 := u1.setReg "DS_i" .real [p.BK,p.BV] (tile p (a.incoming (p.N-1-k)))
  let u3 := u2.setReg "d_i" .real [] (Tile.scalar (some (a.gate (p.N-1-k))))
  let u4 := u3.setReg "Dacc" .real [p.BK,p.BV] (tile p (next p a (p.N-1-k) acc))
  u4.setReg "DG_i" .real [] (Tile.scalar (some (contraction p a (p.N-2-k) (next p a (p.N-1-k) acc))))

def stored (p : Layout) (l : Launch) (DI DG : RegionName) (a : Inputs p)
    (k : Nat) (acc : Values p) (u : BlockState) : BlockState :=
  let f := next p a (p.N-1-k) acc
  let u1 := (beforeStore p a k acc u).writeMem DG (gradientAddress p l (p.N-2-k))
    (contraction p a (p.N-2-k) f)
  scatter p l DI (p.N-2-k) f u1

def result (p : Layout) (l : Launch) (S D DI DG DS : RegionName) (a : Inputs p)
    (k : Nat) (acc : Values p) (u : BlockState) : BlockState :=
  let u1 := (stored p l DI DG a k acc u).setReg "S" .ptr [p.BK,p.BV] (backwardPointer p l S (p.N-2) (k+1))
  let u2 := u1.setReg "DI" .ptr [p.BK,p.BV] (backwardPointer p l DI (p.N-2) (k+1))
  let u3 := u2.setReg "DS" .ptr [p.BK,p.BV] (backwardPointer p l DS (p.N-1) (k+1))
  let u4 := u3.setReg "DG" .ptr [] (backwardGradient p l DG (k+1))
  u4.setReg "D" .ptr [] (backwardGate p l D (k+1))

theorem update_eval (p : Layout) (a : Inputs p) (q : Nat) (acc : Values p) (u : BlockState)
    (hg : u.regs .real [] "d_i" = some (Tile.scalar (some (a.gate q))))
    (hs : u.regs .real [p.BK,p.BV] "DS_i" = some (tile p (a.incoming q)))
    (ha : u.regs .real [p.BK,p.BV] "Dacc" = some (tile p acc)) :
    evalOp (update p) u = some (tile p (next p a q acc)) := by
  rw [update,evalOp_add,evalOp_mul]
  simp only [evalOp_ref,hg,hs,ha,Option.bind_some,Option.bind_eq_bind]
  rfl

theorem gradient_eval (p : Layout) (a : Inputs p) (c : Nat) (f : Values p) (u : BlockState)
    (hs : u.regs .real [p.BK,p.BV] "S_i" = some (tile p (a.source c)))
    (ha : u.regs .real [p.BK,p.BV] "Dacc" = some (tile p f)) :
    evalOp (gradient p) u = some (Tile.scalar (some (contraction p a c f))) := by
  simp only [gradient,evalOp.eq_def,hs,ha,Option.bind,bind]
  apply congrArg some
  apply Tile.ext
  intro i
  cases i
  rw [Tile.reduceSum_reshape_two]
  change (∑ e : Fin p.BK, ∑ j : Fin p.BV, (↑(f e j * a.source c e j) : WithBot ℝ)) =
    (↑(contraction p a c f) : WithBot ℝ)
  simp only [← WithBot.coe_sum,contraction]

theorem store_tile (p : Layout) (l : Launch) (r : RegionName) (c : Nat)
    (f : Values p) (u : BlockState)
    (hp : u.regs .ptr [p.BK,p.BV] "DI" = some (pointer p l r c))
    (hf : u.regs .real [p.BK,p.BV] "Dacc" = some (tile p f)) :
    stepStmt (.store .real [p.BK,p.BV] (.ptr (.ref .ptr [p.BK,p.BV] "DI"))
      (.ref .real [p.BK,p.BV] "Dacc") .none) u = some (scatter p l r c f u) := by
  simp only [stepStmt,evalOp_ref,hp,hf,Option.bind_some,Option.bind_eq_bind]
  rfl

theorem store_scalar (r : RegionName) (off : Nat) (f : ℝ) (u : BlockState)
    (hp : u.regs .ptr [] "DG" = some (Tile.scalar (r,off)))
    (hf : u.regs .real [] "DG_i" = some (Tile.scalar (some f))) :
    stepStmt (.store .real [] (.ptr (.ref .ptr [] "DG")) (.ref .real [] "DG_i") .none) u =
      some (u.writeMem r off f) := by
  simp only [stepStmt,evalOp_ref,hp,hf,Option.bind_some,Option.bind_eq_bind]
  rfl

theorem decrement_tile (p : Layout) (l : Launch) (r : RegionName) (start k : Nat)
    (name : RegName) (u : BlockState)
    (hp : u.regs .ptr [p.BK,p.BV] name = some (backwardPointer p l r start k)) :
    evalOp (.ptrSub .scalarR (.ref .ptr [p.BK,p.BV] name) (stride p)) u =
      some (backwardPointer p l r start (k+1)) := by
  simp only [stride,evalOp.eq_def,hp,Option.bind,bind]
  exact congrArg some (pointer_decrement p l r start k)

theorem decrement_gate (p : Layout) (l : Launch) (r : RegionName) (k : Nat) (u : BlockState)
    (hp : u.regs .ptr [] "D" = some (backwardGate p l r k)) :
    evalOp (.ptrSub .nil (.ref .ptr [] "D") (.constNat 1)) u = some (backwardGate p l r (k+1)) := by
  simp only [evalOp.eq_def,hp,Option.bind,bind]
  exact congrArg some (gate_decrement p l r k)

theorem decrement_gradient (p : Layout) (l : Launch) (r : RegionName) (k : Nat) (u : BlockState)
    (hp : u.regs .ptr [] "DG" = some (backwardGradient p l r k))
    (hk : u.regs .nat [] "NUM_K" = some (Tile.scalar (numK p)))
    (hv : u.regs .nat [] "NUM_V" = some (Tile.scalar (numV p))) :
    evalOp (.ptrSub .nil (.ref .ptr [] "DG") gradientStride) u = some (backwardGradient p l r (k+1)) := by
  simp only [gradientStride,evalOp.eq_def,hp,hk,hv,Option.bind,bind]
  exact congrArg some (gradient_decrement p l r k)

theorem executes (p : Layout) (l : Launch) (S D DI DG DS : RegionName) (a : Inputs p)
    (k : Nat) (acc : Values p) (u : BlockState) (hk : k < p.N-1)
    (hs : u.regs .ptr [p.BK,p.BV] "S" = some (backwardPointer p l S (p.N-2) k))
    (hdi : u.regs .ptr [p.BK,p.BV] "DI" = some (backwardPointer p l DI (p.N-2) k))
    (hds : u.regs .ptr [p.BK,p.BV] "DS" = some (backwardPointer p l DS (p.N-1) k))
    (hdg : u.regs .ptr [] "DG" = some (backwardGradient p l DG k))
    (hd : u.regs .ptr [] "D" = some (backwardGate p l D k))
    (ha : u.regs .real [p.BK,p.BV] "Dacc" = some (tile p acc))
    (hnk : u.regs .nat [] "NUM_K" = some (Tile.scalar (numK p)))
    (hnv : u.regs .nat [] "NUM_V" = some (Tile.scalar (numV p)))
    (hS : ∀ e j, u.readMem S (address p l (p.N-2-k) e.val j.val) = a.source (p.N-2-k) e j)
    (hDS : ∀ e j, u.readMem DS (address p l (p.N-1-k) e.val j.val) = a.incoming (p.N-1-k) e j)
    (hD : u.readMem D (gateAddress p l (p.N-1-k)) = a.gate (p.N-1-k)) :
    stepStmts (loopBody p) u = some (result p l S D DI DG DS a k acc u) := by
  have hs' := hs.trans (congrArg some (backwardPointer_live p l S (p.N-2) k (by omega)))
  have hdi' := hdi.trans (congrArg some (backwardPointer_live p l DI (p.N-2) k (by omega)))
  have hds' := hds.trans (congrArg some (backwardPointer_live p l DS (p.N-1) k (by omega)))
  have hdg' := hdg.trans (congrArg some (backwardGradient_live p l DG k (by omega)))
  have hd' := hd.trans (congrArg some (backwardGate_live p l D k (by omega)))
  unfold loopBody
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
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (decrement_tile p l S (p.N-2) k "S" _ (by simp [scatter,hs])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (decrement_tile p l DI (p.N-2) k "DI" _ (by simp [scatter,hdi])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (decrement_tile p l DS (p.N-1) k "DS" _ (by simp [scatter,hds])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (decrement_gradient p l DG k _
    (by simp [scatter,hdg]) (by simp [scatter,hnk]) (by simp [scatter,hnv])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (decrement_gate p l D k _ (by simp [scatter,hd])))]
  rw [stepStmts.nil]
  rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.RealBackwardStep
