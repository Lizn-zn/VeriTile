import bench.tritonbench_g.cross_entropy1.Proofs.BackwardProgram
import VeriTile.Examples.Common
namespace VeriTile.Bench.TritonBenchG.CrossEntropy1.BackwardReal
open VeriTile.Triton BackwardFormula BackwardProgram
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

theorem offsets_injective (p : Layout) (l : Launch) :
    Function.Injective (fun i : TileIndex [p.B] => l.row*p.gradientStride+(l.block*p.B+i.1.val)) := by
  intro a b h
  have hab : a.1 = b.1 := Fin.ext (Nat.add_left_cancel (Nat.add_left_cancel h))
  cases a
  cases b
  simp only at hab
  cases hab
  rfl

/-- Every active output is derived from initial logits, LSE, upstream loss and
label inputs, including both smoothing choices and the ignored-label branch. -/
theorem output (p : Layout) (r : Regions) (u v : BlockState)
    (he : exec (mathematical p r).toAlgKernel u = some v) (i : Fin p.B) :
    v.readMem r.gradient (gradientAddress p (launch u) i.val) =
      if active p (launch u) i.val then expected p (launch u) (inputs u r) i.val
      else u.readMem r.gradient (gradientAddress p (launch u) i.val) := by
  have hinj := offsets_injective p (launch u)
  cases hs : p.hasSmoothing <;>
    by_cases hi : u.readMemValue .int r.label (u.pids 0) = p.ignored
  all_goals simp only [Region.cast] at hi
  all_goals
    simp [exec,mathematical,realKernel,ComputeKernel.toAlgKernel,ComputeKernel.surfaceBody,
      ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,stepStmts,stepStmt,evalOp.eq_def,
      Option.bind,Tile.bop,Tile.uop,Tile.cop,Tile.ptrAdd,NumericDType.add,
      NumericDType.mul,NumericDType.sub,NumericDType.div,ComparableDType.lt,ComparableDType.eq,
      ComparableDType.ne,Region.cast,hs,hi] at he
  all_goals rw [← he]
  all_goals simp only [gradientAddress,launch,column] at hinj ⊢
  all_goals rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hinj (i,PUnit.unit)]
  all_goals
    by_cases ha : u.pids 1*p.B+i.val < p.nCols
    all_goals
      simp [active,column,launch,expected,upstream,adjusted,probability,logitAddress,
        shiftedLabel,inputs,hs,hi,ha,WithBot.realExp,WithBot.realSub,WithBot.realAdd,
        WithBot.realMul,WithBot.realDiv,Option.map₂,Region.cast]
  all_goals split_ifs <;> norm_num [Function.comp_def,Option.map,WithBot.unbotD]
  all_goals first | rfl | exact Or.inl rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.CrossEntropy1.BackwardReal
