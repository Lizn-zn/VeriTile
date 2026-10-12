import bench.tritonbench_g.cache_transform_triton.Proofs.TypedExecution

/-! Full-source storage/mask checks and successful output observations.
These regressions do not replace the symbolic real IO contract. -/
namespace CacheTransformTests
open VeriTile.Triton VeriTile.Bench.TritonBenchG.CacheTransformTriton
open FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000

private def assignment (code : List ComputeStmt) (name : RegName) :=
  (code.drop (Source.prefixBefore code name).length).head?

theorem unmasked_signed_metadata (cosType sinType : TypedSource.Storage) (commuted : Bool) (N : Nat) :
    assignment (TypedSource.prefill cosType sinType commuted "cos" "sin" ⟨"metadata"⟩ "cosOut" "sinOut"
      2 1 6 2 N 4).surfaceBody "cumsum_lens" =
      some (.assign .int [N] "cumsum_lens"
        (.alg (.load .int (.region ⟨"metadata"⟩ (.arange N)) .none))) := by
  cases cosType <;> cases sinType <;> cases commuted <;> rfl

theorem signed_prefill_subtraction (cosType sinType : TypedSource.Storage) (commuted : Bool) (N : Nat) :
    assignment (TypedSource.prefill cosType sinType commuted "cos" "sin" ⟨"metadata"⟩ "cosOut" "sinOut"
      2 1 6 2 N 4).surfaceBody "ori_seq_idx" =
      some (.assign .int [] "ori_seq_idx" (.alg
        (.sub .int .nil (.castNatToInt (.ref .nat [] "idx"))
          (.reduceMaxInt ⟨0, by simp⟩ Bool.false
            (.where (.le .int .scalarR (.ref .int [N] "cumsum_lens")
                (.castNatToInt (.ref .nat [] "idx")))
              (.ref .int [N] "cumsum_lens") (.broadcast (.castNatToInt (.constNat 0)) [N])))))) := by
  cases cosType <;> cases sinType <;> cases commuted <;> rfl

theorem decode_undefined_metadata (cosType sinType : TypedSource.Storage) (commuted : Bool) (B H N : Nat) :
    ∃ mask, assignment (TypedSource.decode cosType sinType commuted "cos" "sin" ⟨"metadata"⟩ "cosOut" "sinOut"
      2 1 H N B).surfaceBody "ori_seq_idx" =
      some (.assign .int [B] "ori_seq_idx"
        (.alg (.load .int (.region ⟨"metadata"⟩ (.ref .nat [B] "idx")) (.mask mask)))) := by
  cases cosType <;> cases sinType <;> cases commuted <;> exact ⟨_, rfl⟩

theorem decode_full_tile (commuted : Bool) (B H N : Nat) :
    ∃ access value mask, (TypedSource.decode .fp16 .bf16 commuted "cos" "sin" ⟨"metadata"⟩ "cosOut" "sinOut"
      2 1 H N B).surfaceBody.getLast? = some (.store .bf16 [B,H] access value mask) := by
  cases commuted <;> exact ⟨_,_,_,rfl⟩

theorem prefill_outputs {α : Type} [Inhabited α] (M : Algebra α)
    (payload : RegionName → Nat → α) :
    (TypedExecution.prefillFinish M .fp16 .bf16 payload).val.mem "cosOut" 10 = .mk .fp16 (payload "cos" 4) ∧
    (TypedExecution.prefillFinish M .fp16 .bf16 payload).val.mem "sinOut" 11 = .mk .bf16 (payload "sin" 5) := by
  constructor <;> rfl

theorem decode_outputs_and_tail {α : Type} [Inhabited α] (M : Algebra α)
    (payload : RegionName → Nat → α) :
    (TypedExecution.decodeFinish M .bf16 .fp16 payload).val.mem "cosOut" 8 = .mk .bf16 (payload "cos" 4) ∧
    (TypedExecution.decodeFinish M .bf16 .fp16 payload).val.mem "sinOut" 9 = .mk .fp16 (payload "sin" 5) ∧
    (TypedExecution.decodeFinish M .bf16 .fp16 payload).val.mem "cosOut" 10 = .mk .bf16 (payload "cosOut" 10) ∧
    (TypedExecution.decodeFinish M .bf16 .fp16 payload).val.mem "metadata" 4 = .mk .int 2 := by
  repeat' constructor

/-- info: FP assumptions used by prefill_equiv:
---
info:   none -/
#guard_msgs in
#print_fp_assumptions FPEquivProofs.prefill_equiv

/-- info: FP assumptions used by decode_equiv:
---
info:   none -/
#guard_msgs in
#print_fp_assumptions FPEquivProofs.decode_equiv

#guard_msgs (drop info) in
#auditModuleAxioms
end
end CacheTransformTests
