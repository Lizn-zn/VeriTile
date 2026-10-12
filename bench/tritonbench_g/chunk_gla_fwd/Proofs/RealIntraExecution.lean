import bench.tritonbench_g.chunk_gla_fwd.Proofs.RealIntraLoop

/-! Complete original and scale-first real executions. Valid output cells
equal the independent K-feature gated contraction; all other tagged memory
cells are preserved. No assumption about unspecified padding values is used.
Key/gate arrays remain separate from output because loads are streamed after
stores; the cached query array may alias output. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealIntraExecution
open VeriTile.Triton RealIntraMath RealIntraProgram RealIntraMemory RealIntraLoop
open RealIntraSetup (inputs initial initial_grid initialization_executes)
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def footprint (p : Layout) (l : Launch) (out : RegionName) : WriteFootprint :=
  fun cell => enabled p l ∧ RealIntraLoop.footprint p l out (columns p l) cell

theorem condition_eval (p : Layout) (l : Launch) (s : BlockState) (hg : Grid l s) :
    evalOp (condition p) s = some (Tile.scalar (decide (rowStart p l < p.T))) := by
  simp only [condition,timeOp,times,evalOp,evalOp_ref,hg.chunk,hg.row,Option.bind_some,Option.bind_eq_bind]
  apply congrArg some
  apply Tile.ext
  intro i
  simp [-Tile.scalar_eta,Tile.scalar,Tile.cop,Tile.bop,Broadcast.leftIndex,Broadcast.rightIndex,NumericDType.add,NumericDType.mul,
    ComparableDType.ge,rowStart]
  by_cases h : l.chunk*p.BT+l.block*p.BC < p.T
  · simp [h,Nat.not_le.mpr h]
  · simp [h,Nat.le_of_not_gt h]

theorem body_executes (p : Layout) (r : Regions) (scale : ℝ) (swapped : Bool) (s : BlockState)
    (hkey : r.key ≠ r.output) (hgate : r.gate ≠ r.output) (hcols : columns p (launch s) ≤ p.BT)
    (hK : p.K ≤ p.BK) :
    ∃ sf, stepStmts (body p r scale swapped) s = some sf ∧
      (enabled p (launch s) → ∀ j, j < columns p (launch s) → ∀ i : Fin p.BC,
        rowActive p (launch s) i.val → sf.readMem r.output (outputAddress p (launch s) i.val j) =
          expected p (launch s) (inputs r s) scale i.val j) ∧
      BlockState.WriteWithin (footprint p (launch s) r.output) s sf := by
  rw [body,stepStmts.append_some (initialization_executes s)]
  simp only [stepStmts,stepStmt,condition_eval p (launch s) (initial s) (initial_grid s),
    Tile.scalar,Option.bind_some,Option.bind_eq_bind]
  by_cases hen : enabled p (launch s)
  · have hrow : rowStart p (launch s) < p.T := hen
    simp only [hrow,decide_true,↓reduceIte]
    rw [stepStmts.append_some (RealIntraSetup.executes p (launch s) r (initial s) (initial_grid s))]
    obtain ⟨v,he,hi⟩ := RealIntraLoop.executes p r scale s swapped hkey hgate hcols
      (start p r s) (start_holds p r scale s)
    have hel : stepStmts [loop p r scale swapped] (RealIntraSetup.result p (launch s) r (initial s)) = some v := by
      change stepStmts [loop p r scale swapped] (start p r s) = _
      rw [stepStmts.cons_some he,stepStmts.nil]
    rw [hel]
    refine ⟨v,rfl,?_,?_⟩
    · intro _ j hj i hrow
      rw [hi.output j hj i hrow]
      exact result_valid p (launch s) (inputs r s) (s.undef r.query) (s.undef r.gate) scale i.val j hrow hK
    · intro region off hn
      exact hi.frame region off (fun h => hn ⟨hen,h⟩)
  · have hrow : ¬rowStart p (launch s) < p.T := hen
    simp only [hrow,decide_false,Bool.false_eq_true,↓reduceIte]
    exact ⟨initial s,rfl,fun h => (hen h).elim,by intro region off hn; rfl⟩

theorem source_executes (p : Layout) (r : Regions) (qs ks gs : TypedIntra.Storage)
    (scale : ℝ) (swapped : Bool) (s : BlockState)
    (hkey : r.key ≠ r.output) (hgate : r.gate ≠ r.output) (hcols : columns p (launch s) ≤ p.BT)
    (hK : p.K ≤ p.BK) :
    ∃ sf, exec (source p r qs ks gs scale swapped).eraseDType.toAlgKernel s = some sf ∧
      (enabled p (launch s) → ∀ j, j < columns p (launch s) → ∀ i : Fin p.BC,
        rowActive p (launch s) i.val → sf.readMem r.output (outputAddress p (launch s) i.val j) =
          expected p (launch s) (inputs r s) scale i.val j) ∧
      BlockState.WriteWithin (footprint p (launch s) r.output) s sf := by
  rw [exec,source_body]
  exact body_executes p r scale swapped s hkey hgate hcols hK

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealIntraExecution
