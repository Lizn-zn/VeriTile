import bench.tritonbench_g.softmax_flaggems.Proofs.RealAddress
import VeriTile.Triton.Semantics.Step

/-! Successful dynamic-loop execution indexed by the number of visited tiles.
The invariant and body transition remain explicit premises; this is an execution
principle, not the completed FlagGems output proof. -/
namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealLoop
open VeriTile Triton

/-- Unlike an aligned-stop rule, this handles a final partial source tile. -/
theorem range_tiles
    {idx : RegName} {N B : Nat} {body : List Stmt}
    {P : Nat → BlockState → Prop} (hN : 0 < N) (hB : 0 < B)
    (hstep : ∀ c s, c < RealAddress.rounds N B → P c s →
      ∃ s', stepStmts body (s.setReg idx .nat [] (Tile.scalar (c * B))) = some s' ∧
        P (c + 1) s') :
    ∀ c s, c ≤ RealAddress.rounds N B → P c s →
      ∃ s', stepForRangeAux idx (c * B) N B body s = some s' ∧
        P (RealAddress.rounds N B) s' := by
  intro c s
  induction h : (RealAddress.rounds N B - c) using Nat.strong_induction_on
      generalizing c s with
  | _ remaining ih =>
    intro hc hp
    by_cases hdone : c = RealAddress.rounds N B
    · subst c
      exact ⟨s, stepForRangeAux.step_ge (Nat.ne_of_gt hB)
        (RealAddress.coverage hB), hp⟩
    · have hlt : c < RealAddress.rounds N B := by omega
      obtain ⟨next, hbody, hnext⟩ := hstep c s hlt hp
      obtain ⟨final, hexec, hfinal⟩ :=
        ih (RealAddress.rounds N B - (c + 1)) (by omega)
          (c + 1) next rfl (by omega) hnext
      refine ⟨final, ?_, hfinal⟩
      rw [stepForRangeAux.step_lt (Nat.ne_of_gt hB)
        ((RealAddress.tile_start_lt_iff hN hB).mpr hlt), hbody]
      simpa only [Option.bind_some, Nat.add_mul, Nat.one_mul] using hexec

/-- Execution of the source's `range(0, N, B)` from its initial invariant. -/
theorem range_dyn_tiles
    {idx : RegName} {N B : Nat} {body : List Stmt}
    {startOp stopOp stepOp : Op .nat []} {s : BlockState}
    {P : Nat → BlockState → Prop} (hN : 0 < N) (hB : 0 < B)
    (hstart : evalOp startOp s = some (Tile.scalar 0))
    (hstop : evalOp stopOp s = some (Tile.scalar N))
    (hwidth : evalOp stepOp s = some (Tile.scalar B)) (hinit : P 0 s)
    (hstep : ∀ c st, c < RealAddress.rounds N B → P c st →
      ∃ next, stepStmts body (st.setReg idx .nat [] (Tile.scalar (c * B))) = some next ∧
        P (c + 1) next) :
    ∃ final, stepStmt (.forRangeDyn idx startOp stopOp stepOp body) s = some final ∧
      P (RealAddress.rounds N B) final := by
  obtain ⟨final, hexec, hfinal⟩ := range_tiles hN hB hstep 0 s (Nat.zero_le _) hinit
  refine ⟨final, ?_, hfinal⟩
  rw [stepForRangeAux.forRangeDyn_unfold, hstart, hstop, hwidth]
  simpa only [Option.bind_some, Tile.scalar, Nat.zero_mul] using hexec

end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealLoop
