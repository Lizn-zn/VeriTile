import bench.tritonbench_g.bmm_chunk_bwd.Proofs.FP

/-! Source regressions for mixed input dtypes, the runtime chunk limit and
the location of the admitted addition. Full execution is checked separately
in the case's TypedExecution module. -/
namespace FPBmmChunkBwdSourceTests
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.BmmChunkBwd

private def source (a dout : TypedSource.Storage) (swapped hasResidual : Bool) :=
  TypedSource.specialize a dout swapped .bf16 .fp16 "A" "Dout" "Db" "Res"
    7 4 3 2 100 10 4 1 120 24 16 4 1 200 12 5 1 180 11 6 1 4 2 2 hasResidual

private def loop (a dout : TypedSource.Storage) (swapped hasResidual : Bool) :=
  (Source.dynamicRange? (source a dout swapped hasResidual).surfaceBody "cs").getD
    ⟨[], .constNat 0, .constNat 0, .constNat 0, [], []⟩

private def assignment (body : List ComputeStmt) (name : RegName) :=
  (body.drop (Source.prefixBefore body name).length).head?

private def residual (swapped hasResidual : Bool) :=
  ((Source.conditional? (loop .fp16 .bf16 swapped hasResidual).after (fun _ => Bool.true)).getD
    ⟨[], .constBool Bool.false, [], [], []⟩).yes

theorem runtime_chunk_limit (a dout : TypedSource.Storage) (swapped hasResidual : Bool) :
    assignment (loop a dout swapped hasResidual).before "chunk_size_limit" = some
      (.assign .nat [] "chunk_size_limit" (.alg (.where
        (.lt .nat .nil (.constNat 4)
          (.sub .nat .nil (.constNat 7) (.mul .nat .nil (.ref .nat [] "pid_c") (.constNat 4))))
        (.constNat 4)
        (.sub .nat .nil (.constNat 7) (.mul .nat .nil (.ref .nat [] "pid_c") (.constNat 4)))))) := by
  cases a <;> cases dout <;> cases swapped <;> rfl

theorem loop_uses_computed_limit (a dout : TypedSource.Storage) (swapped hasResidual : Bool) :
    ((loop a dout swapped hasResidual).start, (loop a dout swapped hasResidual).stop) =
      (.constNat 0, .div .nat .nil
        (.sub .nat .nil (.add .nat .nil (.ref .nat [] "chunk_size_limit") (.constNat 2))
          (.constNat 1)) (.constNat 2)) := by
  cases a <;> cases dout <;> cases swapped <;> rfl

/-- Either bf16 input selects bf16 for both dot operands, with fp32 output
and fp32 accumulator arithmetic. No scalar expansion of dot is assumed. -/
theorem mixed_storage_dot (swapped hasResidual : Bool) :
    assignment (loop .fp16 .bf16 swapped hasResidual).body "acc" = some
      (.assign .real [4, 2] "acc" (.compute (.alg .fp32
        (.add .real (.consSame (.consSame .nil)) (.ref .real [4, 2] "acc")
          (.castFloat .fp32 .real (.dotTyped .bf16 .bf16 .fp32 (batch := [])
            (.ref .bf16 [4, 2] "dout") (.ref .bf16 [2, 2] "a"))))))) := by
  cases swapped <;> rfl

theorem bf16_priority_is_symmetric (swapped hasResidual : Bool) :
    assignment (loop .bf16 .fp16 swapped hasResidual).body "acc" =
      assignment (loop .fp16 .bf16 swapped hasResidual).body "acc" := by
  cases swapped <;> rfl

/-- A mixed dot must read the original fp16 A before converting to bf16.
Loading bf16 directly would read a different typed cell in the semantics. -/
theorem a_storage_precedes_dot_cast (swapped hasResidual : Bool) :
    ∃ access mask, assignment (loop .fp16 .bf16 swapped hasResidual).body "a" = some
      (.assign .bf16 [2, 2] "a" (.alg
        (.castFloat .fp16 .bf16 (.load .fp16 access mask)))) := by
  cases swapped <;> exact ⟨_, _, rfl⟩

theorem residual_storage_precedes_fp32_cast (swapped hasResidual : Bool) :
    ∃ access mask, assignment (residual swapped hasResidual) "res" = some
      (.assign .real [4, 2] "res" (.compute (.alg .fp32
        (.castFloat .bf16 .real (.load .bf16 access mask))))) := by
  cases swapped <;> exact ⟨_, _, rfl⟩

theorem candidate_preserves_whole_contraction (a dout : TypedSource.Storage) (hasResidual : Bool) :
    (loop a dout Bool.false hasResidual).before = (loop a dout Bool.true hasResidual).before ∧
    (loop a dout Bool.false hasResidual).body = (loop a dout Bool.true hasResidual).body ∧
    (loop a dout Bool.false hasResidual).stop = (loop a dout Bool.true hasResidual).stop := by
  cases a <;> cases dout <;> exact ⟨rfl, rfl, rfl⟩

theorem numerical_domain_is_present (a dout : TypedSource.Storage) (hasResidual : Bool) :
    (FPEquivProofs.requirementsFor (source a dout Bool.false hasResidual) 4 2).length = 1 := by
  cases a <;> cases dout <;> rfl

/-- info: FP assumptions used by residual_equiv:
---
info:   add_commute -/
#guard_msgs in
#print_fp_assumptions FPEquivProofs.residual_equiv

#guard_msgs (drop info) in
#auditModuleAxioms
end FPBmmChunkBwdSourceTests
