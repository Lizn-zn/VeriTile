import bench.tritonbench_g.attention_score.Proofs.FP
import bench.tritonbench_g.attention_score.Proofs.RealExpressions

/-! Source regression for precision, signed windows and the upstream query-tail
condition. The executable IO proof must use these complete sources. -/
namespace FPAttentionScoreSourceTests
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.AttentionScore

private def original (w c em en : Bool) := TypedSource.original "Q" "K" "M" "Out"
  512 256 16 1 256 256 16 1 32 16 99 1 2 1 15 16 15 16 16 16
  0 8 1 w c em en rfl
private def candidate (w c em en : Bool) := TypedSource.scalarLeft "Q" "K" "M" "Out"
  512 256 16 1 256 256 16 1 32 16 99 1 2 1 15 16 15 16 16 16
  0 8 1 w c em en rfl
private def loop (w c em en : Bool) :=
  ((Source.dynamicRange? (original w c em en).surfaceBody "start_m").map (·.body)).getD []
private def assignment (code : List ComputeStmt) (name : RegName) (occurrence : Nat := 0) :=
  (code.drop (Source.prefixBefore code name occurrence).length).head?

theorem source_regions (w c em en : Bool) :
    (original w c em en).inputs = ["Q", "K", "M"] ∧
      (original w c em en).outputs = ["Out"] := ⟨rfl, rfl⟩

theorem score_dot_precision (w c em en : Bool) : assignment (loop w c em en) "qk" 1 = some
    (.assign .real [16, 16] "qk" (.compute (.alg .fp32
      (.add .real (.consSame (.consSame .nil)) (.ref .real [16, 16] "qk")
        (.castFloat .fp32 .real (.dotTyped .fp16 .fp16 .fp32 (batch := [])
          (.ref .fp16 [16, 16] "q") (.ref .fp16 [16, 16] "k"))))))) := rfl

theorem normalizer_load_precision (w c em en : Bool) : assignment (loop w c em en) "m" = some
    (.assign .real [16] "m" (.compute (.load .fp32
      (.ptr (.ref .ptr [16] "m_ptrs")) .none))) := rfl

theorem source_distance (w c em en : Bool) :
    RealExpressions.distance? 16 16 64 (original w c em en).surfaceBody =
      some (RealExpressions.distanceExpr 16 16 0) := rfl

theorem future_key_is_negative (s : BlockState)
    (hm : s.regs .nat [] "start_m" = some (Tile.scalar 0))
    (hn : s.regs .nat [] "start_n" = some (Tile.scalar 0)) :
    (evalOp (RealExpressions.distanceExpr 16 16 0) s).map
      (fun t => t.data (0, 1, PUnit.unit)) = some (-1) := by
  rw [RealExpressions.eval_distance s 16 16 0 0 0 hm hn]
  rfl

/-- The historical natural-subtraction window admitted this future key;
its proof cannot be reused as a proof of the signed source window. -/
theorem truncated_window_disagrees :
    (0 ≤ (((0 : Nat) - 1) + 0 - 0) + 0 ∧ (((0 : Nat) - 1) + 0 - 0) + 0 < 8) ∧
    ¬ RealFormula.windowAllows Bool.true Bool.false 0 8 0 1 := by decide

private def tailCondition? : List ComputeStmt → Option (Op .bool [])
  | [] => none
  | .ifThen condition@(.boolNot _) _ :: _ => some condition
  | _ :: rest => tailCondition? rest

/-- The source query-tail mask is selected by the key-even flag. Replacing
this with evenM would change the modeled program, even if it looked tidier. -/
theorem query_tail_uses_key_flag (w c em en : Bool) :
    tailCondition? (loop w c em en) = some (.boolNot (.constBool en)) := rfl

theorem masked_fp16_store (w c em en : Bool) :
    (original w c em en).surfaceBody.getLast? = some
      (.store .fp16 [16] (.ptr (.ref .ptr [16] "o_ptrs"))
        (.alg (.castFloat .real .fp16 (.ref .real [16] "o")))
        (.mask (.lt .nat .scalarR (.ref .nat [16] "o_range") (.constNat 15)))) := rfl

theorem candidate_retains_store (w c em en : Bool) :
    (candidate w c em en).surfaceBody.getLast? = (original w c em en).surfaceBody.getLast? := rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end FPAttentionScoreSourceTests
