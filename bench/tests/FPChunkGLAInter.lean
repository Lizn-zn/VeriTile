import bench.tritonbench_g.chunk_gla_fwd.Proofs.FPInter

/-! Preserve both early exits, full K trip count, the gate reference's actual
layout, half-precision gate subtraction, intrinsic exp, default dot options
and bf16 output. The numerical rewrite requires its reached-operand domain. -/
namespace FPChunkGLAInterTests
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.ChunkGlaFwd
open scoped VeriTile.Spec
set_option maxHeartbeats 4000000
set_option maxRecDepth 18000
set_option linter.unusedVariables false

private def source (T K BT BC BK NC sh st : Nat) :=
  TypedInter.original .fp16 .bf16 .fp16 .bf16 "q" "k" "g" "A" sh st 2 T K BT BC BK NC
private def outerBranch (T K BT BC BK NC sh st : Nat) :=
  (Source.conditional? (source T K BT BC BK NC sh st).surfaceBody (fun _ => Bool.true)).getD
    ⟨[],.constBool Bool.false,[],[],[]⟩
private def innerBranch (T K BT BC BK NC sh st : Nat) :=
  (Source.conditional? (outerBranch T K BT BC BK NC sh st).yes (fun _ => Bool.true)).getD
    ⟨[],.constBool Bool.false,[],[],[]⟩
private def loop (T K BT BC BK NC sh st : Nat) :=
  (Source.dynamicRange? (innerBranch T K BT BC BK NC sh st).yes "i_k").getD
    ⟨[],.constNat 0,.constNat 0,.constNat 0,[],[]⟩
private def assignment (body : List ComputeStmt) (name : RegName) :=
  (body.drop (Source.prefixBefore body name).length).head?

section
variable (T K BT BC BK NC sh st : Nat)
theorem time_branch : (outerBranch T K BT BC BK NC sh st).condition =
    .boolNot (.ge .nat .nil (.add .nat .nil (.mul .nat .nil (.ref .nat [] "i_t") (.constNat BT))
      (.mul .nat .nil (.ref .nat [] "i_i") (.constNat BC))) (.constNat T)) := rfl

theorem triangular_branch : (innerBranch T K BT BC BK NC sh st).condition =
    .boolNot (.le .nat .nil (.ref .nat [] "i_i") (.ref .nat [] "i_j")) := rfl

theorem all_K_blocks : (loop T K BT BC BK NC sh st).stop =
    .div .nat .nil (.sub .nat .nil (.add .nat .nil (.constNat K) (.constNat BK)) (.constNat 1)) (.constNat BK) := rfl

theorem half_gate_subtract_before_exp :
    assignment (loop T K BT BC BK NC sh st).body "b_qg" = some
      (.assign .real [BC,BK] "b_qg" (.alg
        (.mul .real .scalarR (.mul .real (.consSame (.consSame .nil))
          (.castFloat .fp16 .real (.ref .fp16 [BC,BK] "b_q"))
          (.exp (.castFloat .fp16 .real (.sub .fp16 (.consR (.consSame .nil))
            (.ref .fp16 [BC,BK] "b_g") (.expandDim ⟨0,by simp⟩ (.ref .fp16 [BK] "b_gn"))))))
          (.const 2)))) := rfl

theorem fp32_default_dot_add :
    assignment (loop T K BT BC BK NC sh st).body "b_A" = some
      (.assign .real [BC,BC] "b_A" (.compute (.alg .fp32
        (.add .real (.consSame (.consSame .nil)) (.ref .real [BC,BC] "b_A")
          (.dot (batch := []) (.ref .real [BC,BK] "b_qg") (.ref .real [BK,BC] "b_kg") {}))))) := rfl

theorem output_storage :
    ((loop T K BT BC BK NC sh st).after.drop (Source.prefixBeforeStore (loop T K BT BC BK NC sh st).after).length).head? = some
      (.store .bf16 [BC,BC] (.blockPtr (.ref .blockPtr [BC,BC] "p_A") [0,1])
        (.alg (.castFloat .real .bf16 (.ref .real [BC,BC] "b_A"))) .none) := rfl
end

theorem domain_present : (FPInter.original .fp16 .bf16 .fp16 .bf16 "q" "k" "g" "A" 40 5 2 7 3 4 2 2 2).domain.length = 1 := rfl

example (R : FP.ScalarArithmetic.Rules) : True := by
  fail_if_success
    have :
        ({ FPInter.original .fp16 .bf16 .fp16 .bf16 "q" "k" "g" "A" 40 5 2 7 3 4 2 2 2 with domain := [] } : Program) ≡[R]
        ({ FPInter.dotFirst .fp16 .bf16 .fp16 .bf16 "q" "k" "g" "A" 40 5 2 7 3 4 2 2 2 with domain := [] } : Program) := by
      equiv_decompose; fp_prove
  trivial

/-- info: FP assumptions used by inter_equiv:
---
info:   add_commute -/
#guard_msgs in
#print_fp_assumptions FPInter.inter_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end FPChunkGLAInterTests
