import VeriTile.Triton.Memory.TraceProofs
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.SoftmaxOptimize.Safety
open VeriTile.Triton

/-- Pointer stores change memory only, including arbitrary masked stores. -/
theorem pointer_store_regs {dtype : TileDType} {shape : TileShape}
    (ptr : Op .ptr shape) (value : Op dtype shape) (mask : MaskOpt dtype shape)
    (s t : BlockState)
    (h : stepStmt (.store dtype shape (.ptr ptr) value mask) s = some t) :
    t.regs = s.regs := by
  unfold stepStmt at h
  simp only [Option.bind_eq_bind, Option.bind_eq_some_iff] at h
  obtain ⟨values, _, active, _, ptrs, _, ht⟩ := h
  cases ht
  funext d sh name
  exact BlockState.foldl_writeMemTypedAt_masked_regs dtype
    (fun i ↦ (ptrs.data i).1) (fun i ↦ (ptrs.data i).2)
    values.data active (TileShape.allIndices shape) s d sh name

/-- The straight-line source bodies only assign other names or store through pointers. -/
def AvoidsRegister (name : RegName) : Stmt → Prop
  | .assign _ _ target _ => name ≠ target
  | .store _ _ (.ptr _) _ _ => True
  | _ => False

theorem step_preserves_register (name : RegName) (dtype : TileDType) (shape : TileShape)
    (st : Stmt) (s t : BlockState) (havoid : AvoidsRegister name st)
    (hrun : stepStmt st s = some t) : t.regs dtype shape name = s.regs dtype shape name := by
  cases st with
  | assign d sh target value =>
    obtain ⟨v, _, rfl⟩ := TraceProofs.assign_inv hrun
    exact BlockState.setReg_ne_name s target name d dtype sh shape v havoid
  | store d sh address value mask =>
    cases address with
    | ptr ptr => rw [pointer_store_regs ptr value mask s t hrun]
    | region _ _ => contradiction
    | blockPtr _ _ => contradiction
  | _ => contradiction

theorem list_preserves_register (name : RegName) (dtype : TileDType) (shape : TileShape)
    (body : List Stmt) (s t : BlockState)
    (havoid : ∀ st ∈ body, AvoidsRegister name st)
    (hrun : stepStmts body s = some t) : t.regs dtype shape name = s.regs dtype shape name := by
  induction body generalizing s with
  | nil =>
    simp only [stepStmts] at hrun
    cases hrun
    rfl
  | cons st rest ih =>
    obtain ⟨u, hu, ht⟩ := TraceProofs.cons_inv hrun
    rw [ih u (fun st hs ↦ havoid st (List.mem_cons_of_mem _ hs)) ht]
    exact step_preserves_register name dtype shape st s u
      (havoid st List.mem_cons_self) hu

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxOptimize.Safety
