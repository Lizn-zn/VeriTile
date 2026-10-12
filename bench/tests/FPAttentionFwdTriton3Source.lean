import bench.tritonbench_g.attention_fwd_triton3.FPEquiv
import VeriTile.Triton.Semantics.Integer

/-! Checks on the full replacement source. The signed window and range, final
normalization guard, and retained stores must survive frontend changes. -/
namespace FPAttentionFwdTriton3SourceTests
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3

private def original : ComputeKernel := TypedSource.original "Q" "K" "V" "M" "Out" "L" 1
  512 256 16 1 240 240 16 1 240 240 16 1 512 256 16 1
  1 2 1 15 16 15 0 0 16 16 16 1 1 1 0 0 16
private def reciprocal : ComputeKernel := TypedSource.reciprocal "Q" "K" "V" "M" "Out" "L" 1
  512 256 16 1 240 240 16 1 240 240 16 1 512 256 16 1
  1 2 1 15 16 15 0 0 16 16 16 1 1 1 0 0 16

private def loop? : List ComputeStmt → Option ComputeStmt
  | [] => none
  | st@(.forRangeInt ..) :: _ => some st
  | _ :: rest => loop? rest

private def distance? : Nat → List ComputeStmt → Option (Op .int [16, 16])
  | 0, _ => none
  | _ + 1, [] => none
  | n + 1, st :: rest =>
      let found := match st with
        | .assign .int shape "dist" (.alg value) =>
            if h : shape = [16, 16] then some (h ▸ value) else none
        | .ifThen _ body => distance? n body
        | .ifThenElse _ yes no => distance? n yes <|> distance? n no
        | .forRangeInt _ _ _ _ body => distance? n body
        | _ => none
      found <|> distance? n rest

private def signedDistance : Op .int [16, 16] :=
  .add .int .scalarR
    (.sub .int .scalarR
      (.add .int .scalarR
        (.sub .int (.consR (.consL .nil))
          (.castNatToInt (.expandDim ⟨1, by simp⟩ (.arange 16)))
          (.castNatToInt (.expandDim ⟨0, by simp⟩ (.arange 16))))
        (.mul .int .nil (.castNatToInt (.ref .nat [] "start_m")) (.constInt 16)))
      (.ref .int [] "start_n")) (.constInt 0)

theorem actual_signed_distance :
    distance? 128 original.surfaceBody = some signedDistance := rfl

theorem future_key_has_negative_distance (s : BlockState)
    (hm : s.regs .nat [] "start_m" = some (Tile.scalar 0))
    (hn : s.regs .int [] "start_n" = some (Tile.scalar 0)) :
    (evalOp signedDistance s).map (fun t => t.data (0, 1, PUnit.unit)) = some (-1) := by
  simp [signedDistance, evalOp.eq_def, hm, hn, bind, Option.bind, Tile.bop_data, Tile.scalar,
    Tile.expandDim_data, Tile.vec, TileShape.dropInsertedIndex,
    NumericDType.sub, NumericDType.add, NumericDType.mul, -Tile.scalar_eta]

theorem candidate_preserves_signed_loop :
    loop? reciprocal.surfaceBody = loop? original.surfaceBody := rfl

theorem range_uses_signed_registers :
    (loop? original.surfaceBody).map (fun st => match st with
      | .forRangeInt index lo hi stride _ =>
          (index, lo, hi, stride)
      | _ => ("missing", .constInt 0, .constInt 0, .constNat 0)) =
      some ("start_n", .ref .int [] "lo", .ref .int [] "hi", .constNat 16) := rfl

theorem end_branch_is_found :
    (Source.conditional? original.surfaceBody
      (fun yes => yes.any FPEquivalence.isNormalization)).isSome = Bool.true := rfl

theorem normalization_site_is_present :
    (FPEquivalence.normalizationSites original.surfaceBody 16 16).length = 1 := rfl

theorem output_store_is_preserved :
    reciprocal.surfaceBody.getLast? = original.surfaceBody.getLast? := rfl

private def tailMask? : Nat → List ComputeStmt → Option (Op .bool [16, 16])
  | 0, _ => none
  | _ + 1, [] => none
  | n + 1, st :: rest =>
      let found := match st with
        | .assign .real shape "qk" (.alg (.where mask@(.remap ..) _ _)) =>
            if h : shape = [16, 16] then some (h ▸ mask) else none
        | .ifThen _ body => tailMask? n body
        | .ifThenElse _ yes no => tailMask? n yes <|> tailMask? n no
        | .forRangeInt _ _ _ _ body => tailMask? n body
        | _ => none
      found <|> tailMask? n rest

private def tailMask : Op .bool [16, 16] :=
  .remap [16, 16] (fun i => (0, i.2))
    (.expandDim ⟨0, by simp⟩
      (.lt .int .scalarR
        (.add .int .scalarR (.castNatToInt (.arange 16)) (.ref .int [] "start_n"))
        (.castNatToInt (.constNat 15))))

/-- This is the broadcast mask extracted from the complete source, rather
than an independently constructed approximation of its tail condition. -/
theorem actual_tail_mask : tailMask? 128 original.surfaceBody = some tailMask := rfl

/-- The final KV lane is masked in every row, while key 14 remains active.
This execution previously failed at the singleton-dimension remap node. -/
theorem tail_mask_executes {α : Type} [Inhabited α] (M : FP.Structural.Algebra α)
    (s : FP.Structural.State α)
    (hn : s.regs .int [] "start_n" = some (fun _ => 0)) :
    FP.Structural.evalOp M none tailMask s =
      some (fun i => decide (i.2.1.val < 15)) := by
  simp [tailMask, FP.Structural.evalOp_unfold, hn, FP.Structural.numeric,
    FP.Structural.numericLt, FP.Structural.bop, TileShape.dropInsertedIndex]

#guard_msgs (drop info) in
#auditModuleAxioms
end FPAttentionFwdTriton3SourceTests
