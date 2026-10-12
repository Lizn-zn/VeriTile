import VeriTile.Triton.Memory.TraceRewrite
import VeriTile.Meta.StatementAudit

/-! Execution equality alone cannot transport address safety. Branches and
ranges must also retain the safety obligations of the accesses they reach. -/
namespace TraceRewriteTests
open VeriTile.Triton

private def initial : BlockState :=
  { mem := fun _ _ => .real 0, regs := fun _ _ _ => none, pids := fun _ => 0 }
private def bounds : RegionBounds := fun _ => 0
private def missing : Stmt := .assign .real [] "x" (.ref .real [] "missing")
private def badStore : Stmt :=
  .store .real [] (.region "Out" (.constNat 0)) (.ref .real [] "missing") .none

private theorem bad_store_unsafe (s : BlockState) : ¬ badStore.TraceSafe bounds s := by
  simp [badStore, bounds, Stmt.TraceSafe, MemAccess.SafeAt, Op.SafeAt.eq_def,
    MaskOpt.SafeAt, MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    MaskOpt.Active]

/-- Both executions fail, but only one meets the address-safety predicate. -/
theorem equal_failure_does_not_imply_equal_safety :
    stepStmts [badStore] initial = stepStmts [missing] initial ∧
      ¬ Stmt.TraceSafeList bounds [badStore] initial ∧
      Stmt.TraceSafeList bounds [missing] initial := by
  refine ⟨?_, ?_, ?_⟩
  · simp [badStore, missing, stepStmts, stepStmt, initial]
  · intro h
    rw [Stmt.TraceSafeList] at h
    exact bad_store_unsafe initial h.1
  · simp [missing, Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
      stepStmt, initial]

/-- The same out-of-bounds store is safe to skip, but unsafe to select. -/
theorem selected_branch_controls_accesses (s : BlockState) :
    Stmt.TraceSafeList bounds [.ifThen (.constBool Bool.false) [badStore]] s ∧
      ¬ Stmt.TraceSafeList bounds [.ifThen (.constBool Bool.true) [badStore]] s := by
  simp only [TraceRewrite.constant_branch, Bool.false_eq_true, ↓reduceIte,
    List.append_nil]
  constructor
  · exact Stmt.TraceSafeList.nil_intro
  · intro h
    rw [Stmt.TraceSafeList] at h
    exact bad_store_unsafe s h.1

/-- Empty and zero-stride ranges skip the access; a nonempty range checks it. -/
theorem range_boundary_controls_accesses (s : BlockState) :
    Stmt.forRangeTraceSafe bounds "i" 2 2 1 [badStore] s ∧
      Stmt.forRangeTraceSafe bounds "i" 0 2 0 [badStore] s ∧
      ¬ Stmt.forRangeTraceSafe bounds "i" 0 2 1 [badStore] s := by
  refine ⟨?_, ?_, ?_⟩
  · rw [Stmt.forRangeTraceSafe]; trivial
  · rw [Stmt.forRangeTraceSafe]; trivial
  · rw [Stmt.forRangeTraceSafe]
    simp only [Nat.one_ne_zero, ↓reduceIte, Nat.zero_lt_succ]
    intro h
    have hb := h.1
    rw [Stmt.TraceSafeList] at hb
    exact bad_store_unsafe _ hb.1

/-- An earlier failed step has no successor at which to demand tail safety. -/
theorem unreachable_tail_has_no_access (s : BlockState)
    (hm : s.regs .real [] "missing" = none) :
    Stmt.TraceSafeList bounds ([missing] ++ [badStore]) s := by
  rw [TraceRewrite.append_iff]
  simp [missing, Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    stepStmts, stepStmt, hm]

#guard_msgs (drop info) in
#auditModuleAxioms
end TraceRewriteTests
