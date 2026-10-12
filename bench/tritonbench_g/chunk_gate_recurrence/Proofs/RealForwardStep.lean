import bench.tritonbench_g.chunk_gate_recurrence.Proofs.RealForwardProgram

/-! One actual forward iteration, for an arbitrary incoming register tile.
Every source statement executes in order, including the output store and
pointer advances. The following loop proof derives that tile from the inputs. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.RealForwardStep
open VeriTile.Triton RealProgram RealForwardProgram
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

def next (p : Layout) (a : Inputs p) (c : Nat) (acc : Values p) : Values p :=
  fun e j => acc e j * a.gate c + a.source c e j

def beforeStore (p : Layout) (a : Inputs p) (c : Nat) (acc : Values p) (u : BlockState) : BlockState :=
  let u1 := u.setReg "d_i" .real [] (Tile.scalar (some (a.gate c)))
  let u2 := u1.setReg "S_i" .real [p.BK,p.BV] (tile p (a.source c))
  u2.setReg "acc" .real [p.BK,p.BV] (tile p (next p a c acc))

def result (p : Layout) (l : Launch) (S D O : RegionName) (a : Inputs p)
    (c : Nat) (acc : Values p) (u : BlockState) : BlockState :=
  let u1 := scatter p l O (c+1) (next p a c acc) (beforeStore p a c acc u)
  let u2 := u1.setReg "D" .ptr [] (gatePointer p l D (c+1))
  let u3 := u2.setReg "S" .ptr [p.BK,p.BV] (pointer p l S (c+1))
  u3.setReg "O" .ptr [p.BK,p.BV] (pointer p l O (c+2))

theorem load_tile (p : Layout) (l : Launch) (r : RegionName) (c : Nat)
    (name : RegName) (f : Values p) (u : BlockState)
    (hm : ∀ e j, u.readMem r (address p l c e.val j.val) = f e j)
    (hp : u.regs .ptr [p.BK,p.BV] name = some (pointer p l r c)) :
    evalOp (.load .real (.ptr (.ref .ptr [p.BK,p.BV] name)) .none) u = some (tile p f) := by
  simp only [evalOp, evalOp_ref, hp, Option.bind_some, Option.bind_eq_bind]
  apply congrArg some
  apply Tile.ext
  rintro ⟨e,j,⟨⟩⟩
  simp [pointer, tile, Tile.mat, BlockState.readMemValue_real, hm]

theorem load_gate (p : Layout) (l : Launch) (r : RegionName) (c : Nat)
    (name : RegName) (value : ℝ) (u : BlockState)
    (hm : u.readMem r (gateAddress p l c) = value)
    (hp : u.regs .ptr [] name = some (gatePointer p l r c)) :
    evalOp (.load .real (.ptr (.ref .ptr [] name)) .none) u = some (Tile.scalar (some value)) := by
  simp only [evalOp, evalOp_ref, hp, Option.bind_some, Option.bind_eq_bind]
  apply congrArg some
  apply Tile.ext
  intro i
  simp [gatePointer, BlockState.readMemValue_real, hm]

theorem update_eval (p : Layout) (a : Inputs p) (c : Nat) (acc : Values p) (u : BlockState)
    (hg : u.regs .real [] "d_i" = some (Tile.scalar (some (a.gate c))))
    (hs : u.regs .real [p.BK,p.BV] "S_i" = some (tile p (a.source c)))
    (ha : u.regs .real [p.BK,p.BV] "acc" = some (tile p acc)) :
    evalOp (update p) u = some (tile p (next p a c acc)) := by
  rw [update, evalOp_add, evalOp_mul]
  simp only [evalOp_ref, hg, hs, ha, Option.bind_some, Option.bind_eq_bind]
  rfl

theorem store_tile (p : Layout) (l : Launch) (r : RegionName) (c : Nat)
    (name : RegName) (f : Values p) (u : BlockState)
    (hp : u.regs .ptr [p.BK,p.BV] name = some (pointer p l r c))
    (hf : u.regs .real [p.BK,p.BV] "acc" = some (tile p f)) :
    stepStmt (.store .real [p.BK,p.BV] (.ptr (.ref .ptr [p.BK,p.BV] name))
      (.ref .real [p.BK,p.BV] "acc") .none) u = some (scatter p l r c f u) := by
  simp only [stepStmt, evalOp_ref, hp, hf, Option.bind_some, Option.bind_eq_bind]
  rfl

theorem advance_tile (p : Layout) (l : Launch) (r : RegionName) (c : Nat)
    (name : RegName) (u : BlockState)
    (hp : u.regs .ptr [p.BK,p.BV] name = some (pointer p l r c)) :
    evalOp (.ptrAdd .scalarR (.ref .ptr [p.BK,p.BV] name) (stride p)) u =
      some (pointer p l r (c+1)) := by
  rw [evalOp_ptrAdd, evalOp_ref, hp]
  simp only [stride, evalOp_mul, evalOp_constNat, Option.bind_some, Option.bind_eq_bind]
  exact congrArg some (pointer_advance p l r c)

theorem advance_gate (p : Layout) (l : Launch) (r : RegionName) (c : Nat)
    (u : BlockState) (hp : u.regs .ptr [] "D" = some (gatePointer p l r c)) :
    evalOp (.ptrAdd .nil (.ref .ptr [] "D") (.constNat 1)) u =
      some (gatePointer p l r (c+1)) := by
  rw [evalOp_ptrAdd, evalOp_ref, hp]
  simp only [evalOp_constNat, Option.bind_some, Option.bind_eq_bind]
  exact congrArg some (gatePointer_advance p l r c)

theorem executes (p : Layout) (l : Launch) (S D O : RegionName) (a : Inputs p)
    (c : Nat) (acc : Values p) (u : BlockState)
    (hs : u.regs .ptr [p.BK,p.BV] "S" = some (pointer p l S c))
    (hd : u.regs .ptr [] "D" = some (gatePointer p l D c))
    (ho : u.regs .ptr [p.BK,p.BV] "O" = some (pointer p l O (c+1)))
    (ha : u.regs .real [p.BK,p.BV] "acc" = some (tile p acc))
    (hS : ∀ e j, u.readMem S (address p l c e.val j.val) = a.source c e j)
    (hD : u.readMem D (gateAddress p l c) = a.gate c) :
    stepStmts (loopBody p) u = some (result p l S D O a c acc u) := by
  unfold loopBody
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (load_gate p l D c "D" _ u hD hd))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (load_tile p l S c "S" _ _
    (by intro e j; simpa using hS e j) (by simp [hs])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (update_eval p a c acc _
    (by simp) (by simp) (by simp [ha])))]
  rw [stepStmts.cons_some (store_tile p l O (c+1) "O" (next p a c acc) _ (by simp [ho]) (by simp))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (advance_gate p l D c _
    (by simp [scatter, hd])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (advance_tile p l S c "S" _
    (by simp [scatter, hs])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (advance_tile p l O (c+1) "O" _
    (by simp [scatter, ho])))]
  rw [stepStmts.nil]
  rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.RealForwardStep
