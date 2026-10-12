import bench.tritonbench_g.attn_fwd_causal.Proofs.FP

/-! Regressions for the two helper calls, local pointer bases, finite additive
masking and mixed precision. These assertions inspect both complete typed
sources; mathematical execution and flat IO are proved separately. -/
namespace FPAttnFwdCausalSourceTests
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.AttnFwdCausal

private def source (optimized : Bool) (stage : Nat) (h : stage = 2 ∨ stage = 3) :=
  (if optimized then TypedSource.reciprocal else TypedSource.original)
    "Q" "K" "V" "QS" "KS" "Out"
    4 4 1 1 4 4 1 1 4 4 1 1 4 4 1 1 1 1 4 1 2 2 1 1 stage h
private def loop (optimized : Bool) (stage : Nat) (h : stage = 2 ∨ stage = 3) (n : Nat) :=
  ((Source.dynamicRange? (source optimized stage h).surfaceBody "start_n" n).map (·.body)).getD []
private def assignment (code : List ComputeStmt) (name : RegName) (occurrence : Nat := 0) :=
  (code.drop (Source.prefixBefore code name occurrence).length).head?
private def bounds (optimized : Bool) (stage : Nat) (h : stage = 2 ∨ stage = 3) (n : Nat) :=
  (Source.dynamicRange? (source optimized stage h).surfaceBody "start_n" n).map
    (fun r => (r.start,r.stop,r.stride))

private def condition? : ComputeStmt → Option (Op .bool [])
  | .ifThen condition _ => some condition
  | _ => none
private def storeType? : ComputeStmt → Option TileDType
  | .store dtype _ _ _ _ => some dtype
  | _ => none

section
variable (optimized : Bool) (stage : Nat) (h : stage = 2 ∨ stage = 3)

theorem two_helper_ranges :
    bounds optimized stage h 0 = some (.ref .nat [] "first_lo", .ref .nat [] "first_hi", .constNat 2) ∧
    bounds optimized stage h 1 = some (.ref .nat [] "second_lo", .ref .nat [] "second_hi", .constNat 2) ∧
    Source.dynamicRange? (source optimized stage h).surfaceBody "start_n" 2 = none := by
  cases optimized <;> exact ⟨rfl,rfl,rfl⟩

theorem first_helper_selects_outer_stage_two :
    (Source.dynamicRange? (source optimized stage h).surfaceBody "start_n").bind
      (fun r => r.before.getLast?.bind condition?) = some (.constBool (decide (stage = 2))) := by
  cases optimized <;> rfl

theorem output_has_fp16_storage :
    (source optimized stage h).surfaceBody.getLast?.bind storeType? = some .fp16 := by
  cases optimized <;> rfl

theorem candidate_preserves_complete_attention_prefix :
    Source.prefixBefore (source optimized stage h).surfaceBody "acc" 1 =
      Source.prefixBefore (source Bool.false stage h).surfaceBody "acc" 1 := by
  cases optimized <;> rfl

theorem second_call_uses_caller_key_base :
    assignment (source optimized stage h).surfaceBody "K_ptrs_second" = some
      (.assign .ptr [1,2] "K_ptrs_second" (.alg
        (.ptrAdd .scalarR (.ref .ptr [1,2] "K_ptrs")
          (.mul .nat .nil (.constNat 1) (.ref .nat [] "second_lo"))))) := by
  cases optimized <;> rfl

theorem second_call_uses_caller_scale_base :
    assignment (source optimized stage h).surfaceBody "K_scale_ptr_second" = some
      (.assign .ptr [] "K_scale_ptr_second" (.alg
        (.ptrAdd .nil (.ref .ptr [] "K_scale_ptr")
          (.floorDiv .nat .nil (.ref .nat [] "second_lo") (.constNat 2))))) := by
  cases optimized <;> rfl

private def penaltyAssignment (mask : RegName) : ComputeStmt :=
  .assign .real [2,2] "qk" (.compute (.alg .fp32
    (.add .real (.consSame (.consSame .nil)) (.ref .real [2,2] "qk")
      (.where (.ref .bool [2,2] mask) (.broadcast (.const 0) [2,2])
        (.broadcast (.sub .real .nil (.const 0.0) (.const 1.0e6)) [2,2])))))

theorem second_call_adds_penalty :
    assignment (loop optimized stage h 1) "qk" 1 = some (penaltyAssignment "mask") := by
  cases optimized <;> rfl

theorem first_diagonal_adds_penalty :
    ((Source.conditional? (loop optimized stage h 0) (fun _ => Bool.true)).map
      (fun c => assignment c.yes "qk")) = some (some (penaltyAssignment "mask_first")) := by
  cases optimized <;> rfl

theorem source_score_precision : assignment (loop optimized stage h 1) "qk" = some
    (.assign .real [2,2] "qk" (.compute (.alg .fp32
      (.mul .real .scalarR
        (.mul .real .scalarR
          (.castFloat .fp32 .real (.dotTyped .fp16 .fp16 .fp32 (batch := [])
            (.ref .fp16 [2,1] "q") (.ref .fp16 [1,2] "k")))
          (.ref .real [] "q_scale")) (.ref .real [] "k_scale"))))) := by
  cases optimized <;> rfl

theorem source_value_dot_precision : assignment (loop optimized stage h 1) "acc" 1 = some
    (.assign .real [2,1] "acc" (.compute (.alg .fp32
      (.add .real (.consSame (.consSame .nil)) (.ref .real [2,1] "acc")
        (.castFloat .fp16 .real (.dotTyped .fp16 .fp16 .fp16 (batch := [])
          (.ref .fp16 [2,2] "p") (.ref .fp16 [2,1] "v") {outDType := some .fp16})))))) := by
  cases optimized <;> rfl

/-- There are exactly two writes to p: exp2 and fp16 cast. In particular the
historical p=where(mask,p,0) rewrite must not reappear. -/
theorem probability_has_no_extra_write :
    assignment (loop optimized stage h 1) "p" 2 = none := by
  cases optimized <;> rfl

end
#guard_msgs (drop info) in
#auditModuleAxioms
end FPAttnFwdCausalSourceTests
