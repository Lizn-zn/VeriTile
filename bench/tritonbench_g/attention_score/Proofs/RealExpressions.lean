import bench.tritonbench_g.attention_score.Kernels
import bench.tritonbench_g.attention_score.Proofs.RealFormula
import VeriTile.Triton.Semantics.Integer
import VeriTile.Meta.StatementAudit

/-! Signed window expressions from the actual source, with their input-index
meaning. The full execution and independent IO proof are separate obligations. -/
namespace VeriTile.Bench.TritonBenchG.AttentionScore.RealExpressions
open VeriTile.Triton

def distanceExpr (BM BN : Nat) (offset : Int) : Op .int [BM, BN] :=
  .add .int .scalarR
    (.sub .int .scalarR
      (.add .int .scalarR
        (.sub .int (.consR (.consL .nil))
          (.castNatToInt (.expandDim ⟨1, by simp⟩ (.arange BM)))
          (.castNatToInt (.expandDim ⟨0, by simp⟩ (.arange BN))))
        (.castNatToInt (.ref .nat [] "start_m")))
      (.mul .int .nil (.castNatToInt (.ref .nat [] "start_n")) (.constInt BN)))
    (.constInt offset)

def distances (BM BN startM startN : Nat) (offset : Int) : Tile .int [BM, BN] :=
  ⟨fun i => RealFormula.distance (startM + i.1.val) (startN * BN + i.2.1.val) offset⟩

theorem eval_distance (s : BlockState) (BM BN startM startN : Nat) (offset : Int)
    (hm : s.regs .nat [] "start_m" = some (Tile.scalar startM))
    (hn : s.regs .nat [] "start_n" = some (Tile.scalar startN)) :
    evalOp (distanceExpr BM BN offset) s = some (distances BM BN startM startN offset) := by
  simp only [distanceExpr, evalOp.eq_def, hm, hn, bind, Option.bind]
  congr 1
  apply Tile.ext
  intro i
  rcases i with ⟨row, key, ⟨⟩⟩
  simp [distances, RealFormula.distance, Tile.bop_data, Tile.scalar, Tile.vec,
    Tile.expandDim_data, TileShape.dropInsertedIndex, NumericDType.add,
    NumericDType.sub, NumericDType.mul, -Tile.scalar_eta]
  ring

/-- Locate the signed distance without assuming a fixed statement offset. -/
def distance? (BM BN : Nat) : Nat → List ComputeStmt → Option (Op .int [BM, BN])
  | 0, _ => none
  | _ + 1, [] => none
  | fuel + 1, st :: rest =>
      let found := match st with
        | .assign .int sh "dist" (.alg value) =>
            if h : sh = [BM, BN] then some (h ▸ value) else none
        | .ifThen _ body => distance? BM BN fuel body
        | .ifThenElse _ yes no => distance? BM BN fuel yes <|> distance? BM BN fuel no
        | .forRangeDyn _ _ _ _ body => distance? BM BN fuel body
        | _ => none
      found <|> distance? BM BN fuel rest

section
variable (Q K M Out : RegionName)
    (sqz sqh sqm sqk skz skh skn skk soz soh son Z H HK N RN NK BM BD BN : Nat)
    (offset size : Int) (scale : ℝ) (window complement evenM evenN : Bool)
    (hBlock : BM = BN)

theorem original_distance :
    distance? BM BN 64 (TypedSource.original Q K M Out
      sqz sqh sqm sqk skz skh skn skk soz soh son Z H HK N RN NK BM BD BN
      offset size scale window complement evenM evenN hBlock).surfaceBody =
      some (distanceExpr BM BN offset) := by
  subst BM
  simp [distance?, TypedSource.original,
    ComputeKernel.surfaceBody, distanceExpr]

theorem candidate_distance :
    distance? BM BN 64 (TypedSource.scalarLeft Q K M Out
      sqz sqh sqm sqk skz skh skn skk soz soh son Z H HK N RN NK BM BD BN
      offset size scale window complement evenM evenN hBlock).surfaceBody =
      some (distanceExpr BM BN offset) := by
  subst BM
  simp [distance?, TypedSource.scalarLeft,
    ComputeKernel.surfaceBody, distanceExpr]

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionScore.RealExpressions
