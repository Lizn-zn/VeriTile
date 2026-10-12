import bench.tritonbench_g.chunk_gla_fwd.Proofs.FPOutput

/-! Regressions before real erasure: both query rounding sites, distinct dot
options, typed output, unspecified loads and the reached-addition domain. -/
namespace FPChunkGLAOutputTests
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.ChunkGlaFwd
open scoped VeriTile.Spec
set_option maxHeartbeats 3000000
set_option maxRecDepth 18000

private def source (BT BK BV : Nat) :=
  TypedOutput.original .fp16 .bf16 .bf16 .bf16 .fp16 "q" "v" "g" "h" "o" "A"
    15 3 15 3 18 3 1 5 3 3 BT BK BV
private def loop (BT BK BV : Nat) :=
  (Source.dynamicRange? (source BT BK BV).surfaceBody "i_k").getD
    ⟨[],.constNat 0,.constNat 0,.constNat 0,[],[]⟩
private def branch (BT BK BV : Nat) :=
  (Source.conditional? (loop BT BK BV).body (fun _ => Bool.true)).getD
    ⟨[],.constBool Bool.false,[],[],[]⟩
private def assignment (body : List ComputeStmt) (name : RegName) (occurrence : Nat := 0) :=
  (body.drop (Source.prefixBefore body name occurrence).length).head?

theorem query_padding (BT BK BV : Nat) :
    assignment (loop BT BK BV).body "b_q" = some
      (.assign .fp16 [BT,BK] "b_q" (.alg (.load .fp16
        (.blockPtr (.ref .blockPtr [BT,BK] "p_q") [0,1]) .none .unspecified))) := rfl

theorem scale_rounding (BT BK BV : Nat) :
    assignment (loop BT BK BV).body "b_q" 1 = some
      (.assign .fp16 [BT,BK] "b_q" (.alg (.castFloat .real .fp16
        (.mul .real .scalarR (.castFloat .fp16 .real (.ref .fp16 [BT,BK] "b_q")) (.const 1))))) := rfl

theorem gate_rounding (BT BK BV : Nat) :
    assignment (loop BT BK BV).body "b_qg" = some
      (.assign .fp16 [BT,BK] "b_qg" (.alg (.castFloat .real .fp16
        (.mul .real (.consSame (.consSame .nil))
          (.castFloat .fp16 .real (.ref .fp16 [BT,BK] "b_q"))
          (.exp (.castFloat .bf16 .real (.ref .bf16 [BT,BK] "b_g"))))))) := rfl

/-- The history conversion is inside the default-option inter-chunk dot. -/
theorem inter_dot (BT BK BV : Nat) :
    assignment (branch BT BK BV).yes "b_o" = some
      (.assign .real [BT,BV] "b_o" (.compute (.alg .fp32
        (.add .real (.consSame (.consSame .nil)) (.ref .real [BT,BV] "b_o")
          (.castFloat .fp32 .real (.dotTyped .fp16 .fp16 .fp32 (batch := [])
            (.ref .fp16 [BT,BK] "b_qg")
            (.castFloat .bf16 .fp16 (.ref .bf16 [BK,BV] "b_h")))))))) := rfl

/-- The intra-chunk dot has a different storage type and disables TF32. -/
theorem intra_dot (BT BK BV : Nat) :
    assignment (loop BT BK BV).after "b_o" = some
      (.assign .real [BT,BV] "b_o" (.compute (.alg .fp32
        (.add .real (.consSame (.consSame .nil)) (.ref .real [BT,BV] "b_o")
          (FPOutput.intraDot .bf16 BT BV))))) := rfl

theorem output_storage (BT BK BV : Nat) :
    ((loop BT BK BV).after.drop (Source.prefixBeforeStore (loop BT BK BV).after).length).head? = some
      (.store .fp16 [BT,BV] (.blockPtr (.ref .blockPtr [BT,BV] "p_o") [0,1])
        (.alg (.castFloat .real .fp16 (.ref .real [BT,BV] "b_o"))) .none) := rfl

example (R : FP.ScalarArithmetic.Rules) : True := by
  fail_if_success
    have :
        ({ FPOutput.original .fp16 .bf16 .bf16 .bf16 .fp16 "q" "v" "g" "h" "o" "A"
          15 3 15 3 18 3 1 5 3 3 4 2 2 with domain := [] } : Program) ≡[R]
        ({ FPOutput.sumLeft .fp16 .bf16 .bf16 .bf16 .fp16 "q" "v" "g" "h" "o" "A"
          15 3 15 3 18 3 1 5 3 3 4 2 2 with domain := [] } : Program) := by
      equiv_decompose; fp_prove
  trivial

/-- info: FP assumptions used by output_equiv:
---
info:   add_commute -/
#guard_msgs in
#print_fp_assumptions FPOutput.output_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end FPChunkGLAOutputTests
