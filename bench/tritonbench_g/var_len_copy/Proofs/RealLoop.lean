import VeriTile.Meta.StatementAudit
import bench.tritonbench_g.var_len_copy.Proofs.RealInvariant

namespace VeriTile.Bench.TritonBenchG.VarLenCopy.FullReal
open VeriTile Triton

/-- Every chunk follows the actual signed range; zero and negative lengths
terminate before entering the body. -/
theorem loop_run (initial : BlockState) (os ol ns : Region .int)
    (src dst : RegionName) (B : Nat) (length : Int) (oldStart newStart : Nat)
    (hB : 0 < B) (hsd : src ≠ dst) :
    ∀ c s, c ≤ Math.rounds length B → Inv initial src dst B length oldStart newStart c s →
      ∃ final, stepForRangeIntAux "i" (c * B : Nat) length B (body os ol ns src dst B) s = some final ∧
        Inv initial src dst B length oldStart newStart (Math.rounds length B) final := by
  intro c s
  induction h : (Math.rounds length B - c) using Nat.strong_induction_on generalizing c s with
  | _ remaining ih =>
    intro hc hi
    by_cases hd : c = Math.rounds length B
    · subst c
      have hcover := Math.rounds_coverage length hB
      refine ⟨s, ?_, hi⟩
      exact stepForRangeIntAux.step_ge (Nat.ne_of_gt hB) (by omega)
    · have hlt : c < Math.rounds length B := by omega
      have htest := (Math.loop_test length hB).mpr hlt
      obtain ⟨mid,he,hm⟩ := invariant_step initial s os ol ns src dst B length oldStart newStart c hB hlt hsd hi
      obtain ⟨final,hf,hfinal⟩ := ih (Math.rounds length B - (c + 1)) (by omega)
        (c + 1) mid rfl (by omega) hm
      refine ⟨final, ?_, hfinal⟩
      rw [stepForRangeIntAux.step_lt (Nat.ne_of_gt hB) (by omega), he]
      simpa only [Option.bind_some, Nat.add_mul, Nat.one_mul, Nat.cast_add] using hf

end VeriTile.Bench.TritonBenchG.VarLenCopy.FullReal

#guard_msgs (drop info) in
#auditModuleAxioms
