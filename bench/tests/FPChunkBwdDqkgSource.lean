import bench.tritonbench_g.chunk_bwd_dqkg.Proofs.FP

/-! Precision-sensitive sites in the complete shared source. Real erasure
cannot detect missing casts, so these checks compare the FP AST directly. -/
namespace VeriTile.Bench.Tests.FPChunkBwdDqkgSource
open VeriTile.Triton VeriTile.Bench.TritonBenchG.ChunkBwdDqkg
set_option maxRecDepth 16000

private def original (BT BK BV : Nat) := TypedSource.original .fp16 .bf16 .fp32 .fp16
  "q" "k" "v" "h" "g" "do" "dh" "dq" "dk" "dg"
  9 3 9 3 18 3 2 3 3 3 BT BK BV 2

private def loop (BT BK BV : Nat) :=
  (Source.dynamicRange? (original BT BK BV).surfaceBody "i_v").getD
    ⟨[], .constNat 0, .constNat 0, .constNat 0, [], []⟩

/-- History storage is fp32; the dot receives the explicit bf16 conversion. -/
theorem loop_history_cast (BT BK BV : Nat) :
    ((loop BT BK BV).body.drop
      (Source.prefixBefore (loop BT BK BV).body "b_dq").length).head? =
    some (.assign .real [BT, BK] "b_dq" (.compute (.alg .fp32
      (.add .real (.consSame (.consSame .nil)) (.ref .real [BT, BK] "b_dq")
        (.castFloat .fp32 .real (.dotTyped .bf16 .bf16 .fp32 (batch := [])
          (.ref .bf16 [BT, BV] "b_do")
          (.castFloat .real .bf16 (.ref .real [BV, BK] "b_h")))))))) := by
  rfl

/-- The score must be converted from fp32 to the key storage before its dot. -/
theorem score_cast (BT BK BV : Nat) :
    ((original BT BK BV).surfaceBody.drop
      (Source.prefixBefore (original BT BK BV).surfaceBody "b_ds" 2).length).head? =
    some (.assign .fp16 [BT, BT] "b_ds"
      (.alg (.castFloat .real .fp16 (.ref .real [BT, BT] "b_ds")))) := by
  rfl

theorem final_add_operands (BT BK BV : Nat) :
    ((original BT BK BV).surfaceBody.drop
      (Source.prefixBefore (original BT BK BV).surfaceBody "b_dq" 2).length).head? =
    some (.assign .real [BT, BK] "b_dq" (.compute (.alg .fp32
      (.add .real (.consSame (.consSame .nil)) (.ref .real [BT, BK] "b_dq")
        (FPEquivProofs.finalDot .fp16 BT BK))))) := by
  rfl

theorem output_cast (BT BK BV : Nat) :
    ((original BT BK BV).surfaceBody.drop
      (Source.prefixBeforeStore (original BT BK BV).surfaceBody).length).head? =
    some (.store .fp16 [BT, BK]
      (.blockPtr (.ref .blockPtr [BT, BK] "p_dq") [0, 1])
      (.alg (.castFloat .real .fp16 (.ref .real [BT, BK] "b_dq"))) .none) := by
  rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Tests.FPChunkBwdDqkgSource
