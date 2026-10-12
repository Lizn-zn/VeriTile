import bench.tritonbench_g.attention_fwd_triton2.Kernels
import VeriTile.Triton.Source
import VeriTile.Meta.StatementAudit

/-! Precision checks on the actual source: the additional V cast and bf16
output must survive lowering even when a real-erasure proof still passes. -/
namespace FPAttentionFwdTriton2SourceTests
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton2

private def original : ComputeKernel := TypedSource.original "Q" "K" "V" "QS" "KS" "Out"
  2048 2048 128 1 2048 2048 128 1 2048 2048 128 1 2048 2048 128 1
  1 1 16 128 16 16 128 96 1
private def reciprocal : ComputeKernel := TypedSource.reciprocal "Q" "K" "V" "QS" "KS" "Out"
  2048 2048 128 1 2048 2048 128 1 2048 2048 128 1 2048 2048 128 1
  1 1 16 128 16 16 128 96 1

private def loopBody : List ComputeStmt → List ComputeStmt
  | [] => []
  | .forRange _ _ _ _ body :: _ => body
  | _ :: rest => loopBody rest

private def accumulation (source : ComputeKernel) : Option ComputeStmt :=
  let body := loopBody source.surfaceBody
  (body.drop (Source.prefixBefore body "acc" 1).length).head?

/-- The V cast remains nested inside the fp16-output dot, followed by promotion
into the fp32 accumulation. Replacing it with real or cast-free arithmetic fails. -/
theorem explicit_v_cast_and_fp32_accumulation :
    accumulation original = some
      (.assign .real [16, 128] "acc" (.compute (.alg .fp32
        (.add .real (.consSame (.consSame .nil)) (.ref .real [16, 128] "acc")
          (.castFloat .fp16 .real (.dotTyped .fp16 .fp16 .fp16
            (options := { outDType := some .fp16 }) (batch := [])
            (.ref .fp16 [16, 16] "p")
            (.castFloat .fp16 .fp16 (.ref .fp16 [16, 128] "v")))))))) := rfl

theorem bf16_output_cast_and_store : original.surfaceBody.getLast? = some
    (.store .bf16 [16, 128] (.ptr (.ref .ptr [16, 128] "O_block_ptr"))
      (.alg (.castFloat .real .bf16 (.ref .real [16, 128] "acc")))
      (.mask (.boolAnd (.consR (.consL .nil))
        (.lt .nat .scalarR (.expandDim ⟨1, by simp⟩ (.ref .nat [16] "offs_m")) (.constNat 16))
        (.expandDim ⟨0, by simp⟩ (.lt .nat .scalarR (.arange 128) (.constNat 96)))))) := rfl

theorem candidate_preserves_loop :
    loopBody reciprocal.surfaceBody = loopBody original.surfaceBody := rfl

theorem candidate_keeps_output_conversion :
    reciprocal.surfaceBody.getLast? = original.surfaceBody.getLast? := rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end FPAttentionFwdTriton2SourceTests
