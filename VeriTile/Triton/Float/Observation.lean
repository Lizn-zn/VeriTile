import VeriTile.Triton.Float.StructuralIO

/-!
Execution summaries for decomposition of single-output IO contracts. A summary
certifies a successful run, its observed cell and exact preservation of all
other cells. It contains no equivalence or numerical assumption.
-/
namespace VeriTile.Triton.FP.Structural
open Equational

def ScalarRun {α : Type} [Inhabited α] (io : KernelIO₁) (M : Algebra α)
    (s : State α) (value : Cell α) : Prop :=
  ∃ t, exec M io.kernel s = some t ∧
    t.mem io.out (io.write (s.pids 0)) = value ∧
    ∀ (r : RegionName) o, (r ≠ io.out ∨ o ≠ io.write (s.pids 0)) → t.mem r o = s.mem r o

theorem privateScratch_of_nil (io : KernelIO₁) (h : io.scratch = []) : IO₁PrivateScratch io := by
  simp [IO₁PrivateScratch, h]

/-- Convert two execution summaries into output and frame obligations. The
single-cell restriction is explicit; no output can be shortened or forgotten. -/
theorem scalar_runs {α : Type} [Inhabited α] {R : Spec.Assumptions ComputeStmt}
    {lhs rhs : KernelIO₁} {plans : Schedules} {s : State (Term α)}
    {va vb : Cell (Term α)} (left : ScalarRun lhs (algebra plans) s va)
    (right : ScalarRun rhs (algebra plans) s vb) (size : lhs.Bout = 1) (rightSize : rhs.Bout = 1)
    (related : CellRelated R va vb) :
    ∃ a b, exec (algebra plans) lhs.kernel s = some a ∧
      exec (algebra plans) rhs.kernel s = some b ∧
      (∀ i : Fin lhs.Bout,
        CellRelated R (a.mem lhs.out (lhs.write (s.pids 0) + i.val))
          (b.mem rhs.out (rhs.write (s.pids 0) + i.val))) ∧
      IO₁Frame lhs s a ∧ IO₁Frame rhs s b := by
  obtain ⟨a, ha, hva, hfa⟩ := left
  obtain ⟨b, hb, hvb, hfb⟩ := right
  refine ⟨a, b, ha, hb, ?_, ?_, ?_⟩
  · intro i
    have hi : i.val = 0 := by have := i.isLt; omega
    simpa only [hi, Nat.add_zero, hva, hvb] using related
  · intro r o ho _
    apply hfa r o
    rcases ho with hr | ho
    · exact Or.inl hr
    · exact Or.inr (by simpa using ho ⟨0, by omega⟩)
  · intro r o ho _
    apply hfb r o
    rcases ho with hr | ho
    · exact Or.inl hr
    · exact Or.inr (by simpa using ho ⟨0, by omega⟩)

theorem cellRelated_mk {R : Spec.Assumptions ComputeStmt} {d : TileDType}
    {a b : Value (Term α) d} (h : ValueRelated R d a b) :
    CellRelated R (.mk d a) (.mk d b) := ⟨d, a, b, rfl, rfl, h⟩

end VeriTile.Triton.FP.Structural
