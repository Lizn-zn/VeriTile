import bench.tritonbench_g.bmm_chunk_fwd.Proofs.FP

/-! Regressions for independent storage, actual K-loop bounds and reached FP
sites. Causal and sequence branches remain in the complete typed program. -/
namespace FPBmmChunkFwdSourceTests
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.BmmChunkFwd

private def source (a b : TypedSource.Storage) (swapped causal hasSeq : Bool) :=
  TypedSource.specialize a b swapped .fp16 "A" "B" "Out" ⟨"Seq"⟩
    7 4 3 2 100 10 4 1 120 11 5 1 200 24 16 4 1 90 1 4 2 2 causal hasSeq
private def branch (a b : TypedSource.Storage) (swapped causal hasSeq : Bool) :=
  (Source.conditional? (source a b swapped causal hasSeq).surfaceBody
    (fun body => (Source.dynamicRange? body "k").isSome)).getD
    ⟨[], .constBool Bool.false, [], [], []⟩
private def loop (a b : TypedSource.Storage) (swapped causal hasSeq : Bool) :=
  (Source.dynamicRange? (branch a b swapped causal hasSeq).yes "k").getD
    ⟨[], .constNat 0, .constNat 0, .constNat 0, [], []⟩
private def assignment (body : List ComputeStmt) (name : RegName) :=
  (body.drop (Source.prefixBefore body name).length).head?

theorem loop_uses_full_K (a b : TypedSource.Storage) (swapped causal hasSeq : Bool) :
    ((loop a b swapped causal hasSeq).start, (loop a b swapped causal hasSeq).stop) =
      (.constNat 0, .div .nat .nil
        (.sub .nat .nil (.add .nat .nil (.constNat 3) (.constNat 2)) (.constNat 1)) (.constNat 2)) := by
  cases a <;> cases b <;> cases swapped <;> rfl

theorem mixed_storage_dot (causal hasSeq : Bool) :
    assignment (loop .fp16 .bf16 Bool.false causal hasSeq).body "acc" = some
      (.assign .real [4, 2] "acc" (.compute (.alg .fp32
        (.add .real (.consSame (.consSame .nil)) (.ref .real [4, 2] "acc")
          (.castFloat .fp32 .real (.dotTyped .bf16 .bf16 .fp32 (batch := [])
            (.ref .bf16 [4, 2] "a") (.ref .bf16 [2, 2] "b"))))))) := rfl

theorem bf16_priority_is_symmetric (swapped causal hasSeq : Bool) :
    assignment (loop .bf16 .fp16 swapped causal hasSeq).body "acc" =
      assignment (loop .fp16 .bf16 swapped causal hasSeq).body "acc" := by
  cases swapped <;> rfl

theorem a_storage_precedes_dot_cast (swapped causal hasSeq : Bool) :
    ∃ access mask, assignment (loop .fp16 .bf16 swapped causal hasSeq).body "a" = some
      (.assign .bf16 [4, 2] "a" (.alg
        (.castFloat .fp16 .bf16 (.load .fp16 access mask)))) := by
  cases swapped <;> exact ⟨_, _, rfl⟩

theorem candidate_preserves_loop_context (a b : TypedSource.Storage) (causal hasSeq : Bool) :
    (loop a b Bool.false causal hasSeq).before = (loop a b Bool.true causal hasSeq).before ∧
    (loop a b Bool.false causal hasSeq).after = (loop a b Bool.true causal hasSeq).after ∧
    (loop a b Bool.false causal hasSeq).stop = (loop a b Bool.true causal hasSeq).stop := by
  cases a <;> cases b <;> exact ⟨rfl, rfl, rfl⟩

theorem numerical_domain_is_present (a b : TypedSource.Storage) (causal hasSeq : Bool) :
    (FPEquivProofs.requirementsFor (source a b Bool.false causal hasSeq) a b 4 2 2).length = 1 := by
  cases a <;> cases b <;> rfl

/-- info: FP assumptions used by dot_first_equiv:
---
info:   add_commute -/
#guard_msgs in
#print_fp_assumptions FPEquivProofs.dot_first_equiv

#guard_msgs (drop info) in
#auditModuleAxioms
end FPBmmChunkFwdSourceTests
