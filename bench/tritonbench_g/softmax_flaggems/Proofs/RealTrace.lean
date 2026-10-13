import VeriTile.Triton.Memory.Flatten
import VeriTile.Triton.Semantics.ProgramRewrite

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealTrace
open VeriTile Triton

theorem of_all (bounds : RegionBounds) (body : List Stmt)
    (h : ∀ statement ∈ body, ∀ s, Stmt.TraceSafe bounds statement s)
    (s : BlockState) : Stmt.TraceSafeList bounds body s := by
  induction body generalizing s with
  | nil => simp only [Stmt.TraceSafeList]
  | cons statement tail ih =>
    rw [Stmt.TraceSafeList]
    refine ⟨h statement List.mem_cons_self s, ?_⟩
    cases stepStmt statement s with
    | none => exact True.intro
    | some next => exact ih (fun st hs ↦ h st (List.mem_cons_of_mem _ hs)) next

theorem singleton (bounds : RegionBounds) (statement : Stmt) (s : BlockState)
    (h : Stmt.TraceSafe bounds statement s) :
    Stmt.TraceSafeList bounds [statement] s := by
  rw [Stmt.TraceSafeList]
  refine ⟨h, ?_⟩
  cases stepStmt statement s <;> simp [Stmt.TraceSafeList]

theorem append_iff (bounds : RegionBounds) (before after : List Stmt) (s : BlockState) :
    Stmt.TraceSafeList bounds (before ++ after) s ↔
      Stmt.TraceSafeList bounds before s ∧
        ∀ t, stepStmts before s = some t → Stmt.TraceSafeList bounds after t := by
  induction before generalizing s with
  | nil => simp [Stmt.TraceSafeList, stepStmts]
  | cons head tail ih =>
    simp only [List.cons_append, Stmt.TraceSafeList, stepStmts]
    cases heq : stepStmt head s with
    | none => simp
    | some next => simp only [Option.bind_some, ih, and_assoc]

theorem append (bounds : RegionBounds) (before after : List Stmt) (s t : BlockState)
    (hbefore : Stmt.TraceSafeList bounds before s)
    (hexec : stepStmts before s = some t)
    (hafter : Stmt.TraceSafeList bounds after t) :
    Stmt.TraceSafeList bounds (before ++ after) s := by
  apply (append_iff bounds before after s).mpr
  refine ⟨hbefore, ?_⟩
  intro u hu
  have : u = t := Option.some.inj (hu.symm.trans hexec)
  simpa only [this] using hafter

theorem false_branch (bounds : RegionBounds) (before yes no : List Stmt) (s : BlockState) :
    Stmt.TraceSafeList bounds (before ++ [.ifThenElse (.constBool Bool.false) yes no]) s ↔
      Stmt.TraceSafeList bounds (before ++ no) s := by
  rw [append_iff, append_iff]
  simp [Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt, evalOp.eq_def,
    stepStmt, stepStmts]
  intro _
  constructor
  · intro h t ht
    exact (h t ht).1
  · intro h t ht
    refine ⟨h t ht, ?_⟩
    cases stepStmts no t <;> exact True.intro

end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealTrace
