import bench.tritonbench_g.softmax_optimize.Proofs.Source

namespace VeriTile.Bench.TritonBenchG.SoftmaxOptimize.FullSource
open VeriTile.Triton
set_option maxRecDepth 16000
set_option maxHeartbeats 4000000

def signedQuotient (a b : Op .int []) : Op .int [] :=
  let z := Op.full [] (.constInt 0)
  let aa := Op.add .int .nil a z
  let bb := Op.add .int .nil z b
  (Op.lt .int .nil aa z).where
    (.sub .int .nil z (.floorDiv .int .nil (.sub .int .nil z aa) bb))
    (.floorDiv .int .nil aa bb)

def previousExpression (N B : Nat) : Op .int [] :=
  .sub .int .nil
    (.mul .int .nil
      (signedQuotient
        (.sub .int .nil (.add .int .nil (.constInt N) (.constInt B)) (.constInt 1))
        (.constInt B)) (.constInt B)) (.constInt B)

/-- On positive tile widths the source's truncating division agrees with the
checked nonnegative-numerator signed helper, including N=0. -/
theorem previousExpression_eval (N B : Nat) (hB : 0<B) (s : BlockState) :
    evalOp (previousExpression N B) s = some (Tile.scalar (prevMultipleInt N B)) := by
  have hn : ¬ ((N : Int)+(B : Int)-1<0) := by omega
  simp only [previousExpression, signedQuotient, evalOp.eq_def, Option.bind_some,
    Option.bind_eq_bind]
  apply congrArg some
  apply Tile.ext
  intro i
  change ((if decide (((N : Int)+B-1)+0 < 0) then
      0 - (0-(((N : Int)+B-1)+0))/(0+B)
    else (((N : Int)+B-1)+0)/(0+B)) * B - B) = prevMultipleInt N B
  simp [hn, prevMultipleInt]

/-- Both helper assignments retain the signed expression, including N=0. -/
theorem previous_assignments (input : Storage) (output : Accumulation) (v : Bool)
    (Y X : RegionName) (M N B : Nat) :
    (kernel input output v Y X M N B).surfaceBody[3]? =
      some (.assign .int [] "prev_multiple" (.alg (previousExpression N B))) ∧
    (kernel input output v Y X M N B).surfaceBody[9]? =
      some (.assign .int [] "prev_multiple" (.alg (previousExpression N B))) := by
  cases input <;> cases output <;> cases v <;> constructor <;> rfl

def isSignedLoop : ComputeStmt → Bool
  | .forRangeInt _ _ _ _ _ => Bool.true
  | _ => Bool.false

/-- The source contains exactly the four signed source loops, with no branch
that silently replaces the zero-column behavior. -/
theorem four_signed_loops (input : Storage) (output : Accumulation) (v : Bool)
    (Y X : RegionName) (M N B : Nat) :
    ((kernel input output v Y X M N B).surfaceBody.filter isSignedLoop).length = 4 := by
  cases input <;> cases output <;> cases v <;> rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxOptimize.FullSource
