import bench.tritonbench_g.attention_kernel.FPEquiv

/-! Check precision-sensitive parts of the full source, including the bias
scalar conversion and default dot output, which real erasure cannot detect. -/
namespace FPAttentionKernelSourceTests
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.AttentionKernel

private def original (storage : TypedSource.Storage) := TypedSource.original storage
  "Q" "K" "V" "B0" "Out" 1
  256 16 1 256 16 1 256 16 1 256 16 1 512 32
  1 1 16 0 16 32 16 16 16
private def reciprocal (storage : TypedSource.Storage) := TypedSource.reciprocal storage
  "Q" "K" "V" "B0" "Out" 1
  256 16 1 256 16 1 256 16 1 256 16 1 512 32
  1 1 16 0 16 32 16 16 16

private def loopBody (source : ComputeKernel) : List ComputeStmt :=
  (Source.dynamicRange? source.surfaceBody "start_n").map (·.body) |>.getD []

private def assignment (body : List ComputeStmt) (name : RegName) (occurrence : Nat := 0) :
    Option ComputeStmt := (body.drop (Source.prefixBefore body name occurrence).length).head?

/-- The unchecked source load omits padding; retain that tag even though no
lane can select it without a boundary check. -/
theorem fp16_q_load : assignment (original .fp16).surfaceBody "q" = some
    (.assign .fp16 [16, 16] "q" (.alg
      (.load .fp16 (.blockPtr (.ref .blockPtr [16, 16] "Q_block_ptr") []) .none .unspecified))) := rfl

theorem fp16_default_score_dot : assignment (loopBody (original .fp16)) "qk" 1 = some
    (.assign .fp32 [16, 16] "qk" (.alg
      (.add .fp32 (.consSame (.consSame .nil))
        (.castFloat .fp16 .fp32 (.ref .fp16 [16, 16] "qk"))
        (.dotTyped .fp16 .fp16 .fp32 (batch := [])
          (.ref .fp16 [16, 16] "q") (.ref .fp16 [16, 16] "k"))))) := rfl

/-- Bias addition, multiplication and scalar conversion stay at storage precision,
then promote to the fp32 score. Replacing the constant with an untyped literal
or lifting the bias arithmetic to fp32 breaks this check. -/
theorem fp16_bias_precision : assignment (loopBody (original .fp16)) "qk" 2 = some
    (.assign .fp32 [16, 16] "qk" (.alg
      (.add .fp32 (.consSame (.consSame .nil)) (.ref .fp32 [16, 16] "qk")
        (.castFloat .fp16 .fp32 (.mul .fp16 .scalarR
          (.add .fp16 (.consSame (.consL .nil))
            (.ref .fp16 [16, 1] "b0") (.ref .fp16 [16, 16] "b1"))
          (.castFloat .real .fp16 (.const 1.44269504))))))) := rfl

theorem fp16_probability_cast_and_default_accumulation :
    assignment (loopBody (original .fp16)) "acc" 1 = some
      (.assign .real [16, 16] "acc" (.compute (.alg .fp32
        (.add .real (.consSame (.consSame .nil)) (.ref .real [16, 16] "acc")
          (.castFloat .fp32 .real (.dotTyped .fp16 .fp16 .fp32 (batch := [])
            (.castFloat .real .fp16 (.ref .real [16, 16] "p"))
            (.ref .fp16 [16, 16] "v"))))))) := rfl

theorem fp16_output_conversion : (original .fp16).surfaceBody.getLast? = some
    (.store .fp16 [16, 16] (.blockPtr (.ref .blockPtr [16, 16] "O_block_ptr") [])
      (.alg (.castFloat .real .fp16 (.ref .real [16, 16] "acc"))) .none) := rfl

theorem bf16_q_load : assignment (original .bf16).surfaceBody "q" = some
    (.assign .bf16 [16, 16] "q" (.alg
      (.load .bf16 (.blockPtr (.ref .blockPtr [16, 16] "Q_block_ptr") []) .none .unspecified))) := rfl

theorem bf16_default_score_dot : assignment (loopBody (original .bf16)) "qk" 1 = some
    (.assign .fp32 [16, 16] "qk" (.alg
      (.add .fp32 (.consSame (.consSame .nil))
        (.castFloat .bf16 .fp32 (.ref .bf16 [16, 16] "qk"))
        (.dotTyped .bf16 .bf16 .fp32 (batch := [])
          (.ref .bf16 [16, 16] "q") (.ref .bf16 [16, 16] "k"))))) := rfl

/-- Bias addition, multiplication and scalar conversion stay at storage precision,
then promote to the fp32 score. Replacing the constant with an untyped literal
or lifting the bias arithmetic to fp32 breaks this check. -/
theorem bf16_bias_precision : assignment (loopBody (original .bf16)) "qk" 2 = some
    (.assign .fp32 [16, 16] "qk" (.alg
      (.add .fp32 (.consSame (.consSame .nil)) (.ref .fp32 [16, 16] "qk")
        (.castFloat .bf16 .fp32 (.mul .bf16 .scalarR
          (.add .bf16 (.consSame (.consL .nil))
            (.ref .bf16 [16, 1] "b0") (.ref .bf16 [16, 16] "b1"))
          (.castFloat .real .bf16 (.const 1.44269504))))))) := rfl

theorem bf16_probability_cast_and_default_accumulation :
    assignment (loopBody (original .bf16)) "acc" 1 = some
      (.assign .real [16, 16] "acc" (.compute (.alg .fp32
        (.add .real (.consSame (.consSame .nil)) (.ref .real [16, 16] "acc")
          (.castFloat .fp32 .real (.dotTyped .bf16 .bf16 .fp32 (batch := [])
            (.castFloat .real .bf16 (.ref .real [16, 16] "p"))
            (.ref .bf16 [16, 16] "v"))))))) := rfl

theorem bf16_output_conversion : (original .bf16).surfaceBody.getLast? = some
    (.store .bf16 [16, 16] (.blockPtr (.ref .blockPtr [16, 16] "O_block_ptr") [])
      (.alg (.castFloat .real .bf16 (.ref .real [16, 16] "acc"))) .none) := rfl

theorem candidate_preserves_loop (storage : TypedSource.Storage) :
    loopBody (reciprocal storage) = loopBody (original storage) := by
  cases storage <;> rfl

theorem candidate_keeps_output_conversion (storage : TypedSource.Storage) :
    (reciprocal storage).surfaceBody.getLast? = (original storage).surfaceBody.getLast? := by
  cases storage <;> rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end FPAttentionKernelSourceTests
