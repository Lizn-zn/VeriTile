import bench.tritonbench_g.chunk_gla_fwd.Proofs.FPMerge

/-! Merge source regressions before real erasure. Head count comes from the
runtime grid, the source early return remains, input padding is unspecified,
and output retains its bf16 conversion. Removing the reached-operand domain
must prevent automatic admission of the numerical rewrite. -/
namespace FPChunkGLAMergeTests
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.ChunkGlaFwd
open scoped VeriTile.Spec
set_option linter.unusedVariables false
set_option maxHeartbeats 3000000
set_option maxRecDepth 18000

private def source (T BT BC NK : Nat) := TypedMerge.original .bf16 "A" "out" T BT BC NK
private def branch (T BT BC NK : Nat) :=
  (Source.conditional? (source T BT BC NK).surfaceBody (fun _ => Bool.true)).getD
    ⟨[],.constBool Bool.false,[],[],[]⟩
private def loop (T BT BC NK : Nat) :=
  (Source.dynamicRange? (branch T BT BC NK).yes "i_k").getD
    ⟨[],.constNat 0,.constNat 0,.constNat 0,[],[]⟩
private def assignment (body : List ComputeStmt) (name : RegName) :=
  (body.drop (Source.prefixBefore body name).length).head?

theorem enabled_branch (T BT BC NK : Nat) :
    (branch T BT BC NK).condition = .boolNot (.ge .nat .nil
      (.add .nat .nil (.mul .nat .nil (.ref .nat [] "i_t") (.constNat BT))
        (.mul .nat .nil (.ref .nat [] "i_c") (.constNat BC))) (.constNat T)) := rfl

theorem runtime_heads (T BT BC NK : Nat) :
    assignment (branch T BT BC NK).yes "n_bh" = some (.assign .nat [] "n_bh" (.alg (.numPrograms 2))) := rfl

theorem input_slice (T BT BC NK : Nat) :
    assignment (loop T BT BC NK).body "p_A" = some
      (.assign .blockPtr [BC,BC] "p_A" (.alg (.makeBlockPtrDynOffsets "A"
        (.mul .nat .nil (.mul .nat .nil
          (.add .nat .nil (.ref .nat [] "i_bh") (.mul .nat .nil (.ref .nat [] "i_k") (.ref .nat [] "n_bh")))
          (.constNat T)) (.constNat BC)) [T,BC] [BC,BC] [BC,1]
        [.add .nat .nil (.mul .nat .nil (.ref .nat [] "i_t") (.constNat BT))
          (.mul .nat .nil (.ref .nat [] "i_c") (.constNat BC)),.constNat 0]))) := rfl

theorem addition_precision (T BT BC NK : Nat) :
    assignment (loop T BT BC NK).body "b_A" = some
      (.assign .real [BC,BC] "b_A" (.compute (.alg .fp32
        (.add .real (.consSame (.consSame .nil)) (.ref .real [BC,BC] "b_A")
          (.load .real (.blockPtr (.ref .blockPtr [BC,BC] "p_A") [0,1]) .none .unspecified))))) := rfl

theorem output_storage (T BT BC NK : Nat) :
    ((loop T BT BC NK).after.drop (Source.prefixBeforeStore (loop T BT BC NK).after).length).head? = some
      (.store .bf16 [BC,BC] (.blockPtr (.ref .blockPtr [BC,BC] "p_A2") [0,1])
        (.alg (.castFloat .real .bf16 (.ref .real [BC,BC] "b_A"))) .none) := rfl

theorem domain_present (T BT BC NK : Nat) : (FPMerge.originalProgram .bf16 "A" "out" T BT BC NK).domain.length = 1 := rfl

example (R : FP.ScalarArithmetic.Rules) : True := by
  fail_if_success
    have :
        ({ FPMerge.originalProgram .bf16 "A" "out" 5 4 2 3 with domain := [] } : Program) ≡[R]
        ({ FPMerge.loadFirstProgram .bf16 "A" "out" 5 4 2 3 with domain := [] } : Program) := by
      equiv_decompose; fp_prove
  trivial

/-- info: FP assumptions used by merge_equiv:
---
info:   add_commute -/
#guard_msgs in
#print_fp_assumptions FPMerge.merge_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end FPChunkGLAMergeTests
