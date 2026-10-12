import bench.tritonbench_g.attention_llama.FPEquiv

/-! Precision and source quirks that real erasure would hide. -/
namespace FPAttentionLlamaSourceTests
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.AttentionLlama

private def original (causal : Bool) := TypedSource.original "Q" "K" "V" "Out" 1
    256 256 16 1 256 256 16 1 256 256 16 1 256 256 16 1
    1 15 15 0 16 16 16 causal
private def body (causal : Bool) :=
  ((Source.dynamicRange? (original causal).surfaceBody "start_n").map (·.body)).getD []
private def assignment (code : List ComputeStmt) (name : RegName) (occurrence : Nat := 0) :=
  (code.drop (Source.prefixBefore code name occurrence).length).head?

theorem source_io (causal : Bool) :
    (original causal).inputs = ["Q", "K", "V"] ∧
      (original causal).outputs = ["Out"] := ⟨rfl, rfl⟩

theorem typed_score_dot (causal : Bool) : assignment (body causal) "qk" 1 = some
    (.assign .real [16, 16] "qk" (.compute (.alg .fp32
      (.add .real (.consSame (.consSame .nil)) (.ref .real [16, 16] "qk")
        (.castFloat .fp32 .real (.dotTyped .fp16 .fp16 .fp32 (batch := [])
          (.ref .fp16 [16, 16] "q") (.transpose (batch := []) (.ref .fp16 [16, 16] "k")))))))) := rfl

theorem key_mask_uses_block_local_offsets (causal : Bool) :
    assignment (body causal) "qk" 2 = some
      (.assign .real [16, 16] "qk" (.alg
        (.where (.remap [16, 16] (Broadcast.nil.consSame.consL).leftIndex
          (.lt .nat .scalarR (.expandDim ⟨0, by simp⟩ (.ref .nat [16] "offs_n")) (.constNat 15)))
          (.ref .real [16, 16] "qk") (.broadcast .negInf [16, 16])))) := rfl

theorem probability_cast (causal : Bool) : assignment (body causal) "p" 2 = some
    (.assign .fp16 [16, 16] "p" (.alg
      (.castFloat .real .fp16 (.ref .real [16, 16] "p")))) := rfl

theorem typed_value_dot (causal : Bool) : assignment (body causal) "acc" 1 = some
    (.assign .real [16, 16] "acc" (.compute (.alg .fp32
      (.add .real (.consSame (.consSame .nil)) (.ref .real [16, 16] "acc")
        (.castFloat .fp32 .real (.dotTyped .fp16 .fp16 .fp32 (batch := [])
          (.ref .fp16 [16, 16] "p") (.ref .fp16 [16, 16] "v"))))))) := rfl

theorem masked_fp16_store (causal : Bool) : (original causal).surfaceBody.getLast? = some
    (.store .fp16 [16, 16] (.ptr (.ref .ptr [16, 16] "out_ptrs"))
      (.alg (.castFloat .real .fp16 (.ref .real [16, 16] "acc")))
      (.mask (.remap [16, 16] (Broadcast.nil.consL.consSame).leftIndex
        (.lt .nat .scalarR (.expandDim ⟨1, by simp⟩ (.ref .nat [16] "offs_m")) (.constNat 15))))) := rfl

private def fp8 (causal optimized : Bool) :=
  (if optimized then TypedSource.carryFirstFP8 else TypedSource.originalFP8)
    "Q" "K" "V" "Out" 1
    256 256 16 1 256 256 16 1 256 256 16 1 256 256 16 1
    1 15 15 0 16 16 16 causal
private def fp8Body (causal optimized : Bool) :=
  ((Source.dynamicRange? (fp8 causal optimized).surfaceBody "start_n").map (·.body)).getD []

/-- Both K/V inputs remain signed-byte loads with integer zero padding.
The reinterpretation does not change addresses or the block-dependent mask. -/
theorem fp8_byte_loads (causal optimized : Bool) :
    ∀ (name ptr : RegName), (name, ptr) ∈ [("k", "k_ptrs"), ("v", "v_ptrs")] →
      assignment (fp8Body causal optimized) name = some
        (.assign .int [16, 16] name (.alg
          (.load .int (.ptr (.ref .ptr [16, 16] ptr))
            (.maskOther (.remap [16, 16] (Broadcast.nil.consL.consSame).leftIndex
              (.lt .nat .scalarR (.expandDim ⟨1, by simp⟩
                (.ref .nat [16] "block_n_offs")) (.constNat 15)))
              (.broadcast (.constInt 0) [16, 16]))))) := by
  intro name ptr h
  simp only [List.mem_cons, List.not_mem_nil, or_false, Prod.mk.injEq] at h
  rcases h with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ <;> cases optimized <;> rfl

/-- The raw bits and subsequent fp16 conversion are two distinct statements,
in both source variants and for both causal flags. -/
theorem fp8_conversion_steps (causal optimized : Bool) :
    ∀ name ∈ (["k", "v"] : List RegName),
      assignment (fp8Body causal optimized) name 1 = some
        (.assign .f8e5 [16, 16] name (.alg
          (.decodeFloat8E5 .f8e5 (.ref .int [16, 16] name)))) ∧
      assignment (fp8Body causal optimized) name 2 = some
        (.assign .fp16 [16, 16] name (.alg
          (.castFloat .f8e5 .fp16 (.ref .f8e5 [16, 16] name)))) := by
  intro name h
  simp only [List.mem_cons, List.not_mem_nil, or_false] at h
  rcases h with rfl | rfl <;> cases optimized <;> exact ⟨rfl, rfl⟩

theorem fp8_preserves_dot_cast_mask_store (causal optimized : Bool) :
    assignment (fp8Body causal optimized) "qk" 1 = assignment (body causal) "qk" 1 ∧
    assignment (fp8Body causal optimized) "qk" 2 = assignment (body causal) "qk" 2 ∧
    assignment (fp8Body causal optimized) "p" 2 = assignment (body causal) "p" 2 ∧
    assignment (fp8Body causal optimized) "acc" 1 = assignment (body causal) "acc" 1 ∧
    (fp8 causal optimized).surfaceBody.getLast? = (original causal).surfaceBody.getLast? := by
  cases optimized <;> exact ⟨rfl, rfl, rfl, rfl, rfl⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end FPAttentionLlamaSourceTests
