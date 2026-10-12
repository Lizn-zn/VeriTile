import bench.tritonbench_g.attention_forward_triton.Kernels
import VeriTile.Meta.StatementAudit

namespace FPAttentionSourceTests
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.AttentionForwardTriton

private def original : ComputeKernel := TypedSource.original "Q" "K" "V" "QS" "KS" "Out"
  2048 2048 128 1 2048 2048 128 1 2048 2048 128 1 2048 2048 128 1
  1 1 16 128 16 16 128 96 1
private def reciprocal : ComputeKernel := TypedSource.reciprocal "Q" "K" "V" "QS" "KS" "Out"
  2048 2048 128 1 2048 2048 128 1 2048 2048 128 1 2048 2048 128 1
  1 1 16 128 16 16 128 96 1

-- Inspect the actual final source statement: a cast and fp16 store, with the
-- original row/head mask. A real-carrier store would silently drop output precision.
theorem fp16_output_cast_and_store : original.surfaceBody.getLast? = some
    (.store .fp16 [16, 128] (.ptr (.ref .ptr [16, 128] "O_block_ptr"))
      (.alg (.castFloat .real .fp16 (.ref .real [16, 128] "acc")))
      (.mask (.boolAnd (.consR (.consL .nil))
        (.lt .nat .scalarR (.expandDim ⟨1, by simp⟩ (.ref .nat [16] "offs_m")) (.constNat 16))
        (.expandDim ⟨0, by simp⟩ (.lt .nat .scalarR (.arange 128) (.constNat 96)))))) := rfl

theorem candidate_keeps_output_conversion :
    reciprocal.surfaceBody.getLast? = original.surfaceBody.getLast? := rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end FPAttentionSourceTests
