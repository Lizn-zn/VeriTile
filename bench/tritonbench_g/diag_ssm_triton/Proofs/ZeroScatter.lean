import VeriTile.Triton.Semantics.Step
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton.ZeroScatter
open VeriTile.Triton
noncomputable section
attribute [local instance] Classical.propDecidable

theorem memory {α : Type} (region : RegionName) (off : α → Nat) (mask : α → Bool)
    (items : List α) (u : BlockState) (r : RegionName) (o : Nat) :
    (items.foldl (fun acc j => if mask j then acc.writeMem region (off j) 0 else acc) u).mem r o =
      if r = region ∧ ∃ j ∈ items, mask j = true ∧ o = off j
      then MemCell.real 0 else u.mem r o := by
  classical
  induction items generalizing u with
  | nil => simp
  | cons a rest ih =>
    rw [List.foldl_cons,ih]
    by_cases hm : mask a = true
    · rw [if_pos hm,BlockState.writeMem_mem]
      by_cases hr : r = region
      · subst r
        by_cases ho : o = off a
        · subst o; simp [hm]
        · simp [hm,ho]
      · simp [hr]
    · simp [hm]


/-- Propositional-mask form, useful after simplifying an evaluated mask. -/
theorem memory_prop {α : Type} (region : RegionName) (off : α → Nat)
    (mask : α → Prop) [DecidablePred mask] (items : List α) (u : BlockState)
    (r : RegionName) (o : Nat) :
    (items.foldl (fun acc j => if mask j then acc.writeMem region (off j) 0 else acc) u).mem r o =
      if r = region ∧ ∃ j ∈ items, mask j ∧ o = off j
      then MemCell.real 0 else u.mem r o := by
  simpa only [decide_eq_true_eq] using
    memory region off (fun j => decide (mask j)) items u r o

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DiagSsmTriton.ZeroScatter
