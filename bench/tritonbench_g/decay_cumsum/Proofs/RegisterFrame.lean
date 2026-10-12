import VeriTile.Triton.Semantics.ProgramRewrite
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.DecayCumsum.RegisterFrame
open VeriTile.Triton

/-- Installing an unrelated register commutes with an assignment. -/
theorem setReg_comm (s : BlockState) (a b : RegName) (hn : a≠b)
    (da db : TileDType) (sa sb : TileShape) (va : Tile da sa) (vb : Tile db sb) :
    (s.setReg a da sa va).setReg b db sb vb =
      (s.setReg b db sb vb).setReg a da sa va := by
  apply BlockState.ext
  · intros; rfl
  · intro d sh n
    by_cases ha : n=a <;> by_cases hb : n=b
    · exact False.elim (hn (ha.symm.trans hb))
    all_goals simp [BlockState.setReg,ha,hb,hn,Ne.symm hn]
  · intros; rfl
  · intros; rfl
  · intros; rfl

theorem setReg_write (s : BlockState) (name : RegName)
    (d : TileDType) (sh : TileShape) (v : Tile d sh)
    (out : TileDType) (r : RegionName) (o : Nat) (x : TileCarrier out) :
    (s.setReg name d sh v).writeMemTyped out r o x =
      (s.writeMemTyped out r o x).setReg name d sh v := by
  cases out <;> rfl

theorem fold_map {ι : Type} (f : BlockState → BlockState)
    (g : BlockState → ι → BlockState)
    (h : ∀ s i, g (f s) i = f (g s i)) (xs : List ι) (s : BlockState) :
    xs.foldl g (f s) = f (xs.foldl g s) := by
  induction xs generalizing s with
  | nil => rfl
  | cons i xs ih => simpa only [List.foldl_cons,h] using ih (g s i)

/-- A complete statement transports a state map, preserving both failure and
the entire mapped final state. -/
def Transports (f : BlockState → BlockState) (st : Stmt) : Prop :=
  ∀ s, stepStmt st (f s) = (stepStmt st s).map f

theorem assign (f : BlockState → BlockState) (name : RegName)
    (d : TileDType) (sh : TileShape) (op : Op d sh)
    (he : ∀ s, evalOp op (f s) = evalOp op s)
    (hr : ∀ s v, (f s).setReg name d sh v = f (s.setReg name d sh v)) :
    Transports f (.assign d sh name op) := by
  intro s
  simp only [stepStmt,he]
  cases evalOp op s <;> simp [hr]

theorem store_ptr_mask (f : BlockState → BlockState)
    (d : TileDType) (sh : TileShape) (ptr : Op .ptr sh)
    (value : Op d sh) (mask : Op .bool sh)
    (hp : ∀ s, evalOp ptr (f s) = evalOp ptr s)
    (hv : ∀ s, evalOp value (f s) = evalOp value s)
    (hm : ∀ s, evalOp mask (f s) = evalOp mask s)
    (hw : ∀ s r o x, (f s).writeMemTyped d r o x = f (s.writeMemTyped d r o x)) :
    Transports f (.store d sh (.ptr ptr) value (.mask mask)) := by
  intro s
  simp only [stepStmt,hv,hm,hp]
  cases evalOp value s <;> cases evalOp mask s <;> cases evalOp ptr s
  all_goals simp only [bind,Option.bind_none,Option.bind_some,Option.map_none,Option.map_some]
  rename_i values masks ptrs
  congr 1
  apply fold_map
  intro u i
  split <;> simp_all only [hw]

theorem list (f : BlockState → BlockState) (code : List Stmt)
    (h : ∀ st ∈ code, Transports f st) (s : BlockState) :
    stepStmts code (f s) = (stepStmts code s).map f := by
  induction code generalizing s with
  | nil => simp [stepStmts]
  | cons st rest ih =>
    have ht : stepStmt st (f s) = (stepStmt st s).map f := h st (by simp) s
    simp only [stepStmts,ht]
    cases hstep : stepStmt st s <;> simp only [Option.map_none,Option.map_some]
    exact ih (fun st hs => h st (by simp [hs])) _

theorem branch (f : BlockState → BlockState) (cond : Op .bool []) (body : List Stmt)
    (hc : ∀ s, evalOp cond (f s) = evalOp cond s)
    (hb : ∀ s, stepStmts body (f s) = (stepStmts body s).map f) :
    Transports f (.ifThen cond body) := by
  intro s
  simp only [stepStmt,hc]
  cases h : evalOp cond s
  · rfl
  · rename_i value
    simp only [bind,Option.bind_some]
    cases he : value.data PUnit.unit <;> simp [he,hb]

theorem range (f : BlockState → BlockState) (idx : RegName)
    (cur stop stride : Nat) (body : List Stmt)
    (hb : ∀ s, stepStmts body (f s) = (stepStmts body s).map f)
    (hr : ∀ s n, (f s).setReg idx .nat [] (Tile.scalar n) =
      f (s.setReg idx .nat [] (Tile.scalar n))) (s : BlockState) :
    stepForRangeAux idx cur stop stride body (f s) =
      (stepForRangeAux idx cur stop stride body s).map f := by
  by_cases hz : stride=0
  · subst stride
    simp only [stepForRangeAux.step_zero_step,Option.map_some]
  · have hs : 0 < stride := Nat.pos_of_ne_zero hz
    by_cases hc : cur < stop
    · rw [stepForRangeAux.step_lt hz hc,stepForRangeAux.step_lt hz hc,hr,hb]
      cases he : stepStmts body (s.setReg idx .nat [] (Tile.scalar cur))
      · rfl
      · simp only [Option.map_some]
        exact range f idx (cur+stride) stop stride body hb hr _
    · have hge : stop ≤ cur := by omega
      rw [stepForRangeAux.step_ge hz hge,stepForRangeAux.step_ge hz hge]
      rfl
termination_by stop-cur
decreasing_by omega

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.DecayCumsum.RegisterFrame
