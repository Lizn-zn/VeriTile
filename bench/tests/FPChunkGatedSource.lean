import bench.tritonbench_g.chunk_gated_attention.Proofs.FP

/-! Check the typed source before erasure: mixed storage, optional seed
addition, history/final store conversions, exact endpoint addressing and
intrinsic exp. The FP tactic must retain the actual reached domain. -/
namespace FPChunkGatedSourceTests
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
open scoped VeriTile.Spec
set_option linter.unusedVariables false
set_option maxRecDepth 18000
set_option maxHeartbeats 3000000

private def source (BT BK BV : Nat) (gateK initial final : Bool) :=
  TypedSource.hOriginal .fp16 .bf16 .bf16 .bf16 .fp32 "k" "v" "g" "h" "h0" "ht"
    15 3 1 15 3 1 18 3 1 5 3 3 BT BK BV 2 gateK initial final
private def loop (BT BK BV : Nat) (gateK initial final : Bool) :=
  (Source.dynamicRange? (source BT BK BV gateK initial final).surfaceBody "i_t").getD
    ⟨[],.constNat 0,.constNat 0,.constNat 0,[],[]⟩
private def gate (BT BK BV : Nat) (gateK initial final : Bool) :=
  (Source.conditional? (loop BT BK BV gateK initial final).body (fun _ => Bool.true)).getD
    ⟨[],.constBool Bool.false,[],[],[]⟩
private def seed (BT BK BV : Nat) (gateK initial final : Bool) :=
  (Source.conditional? (loop BT BK BV gateK initial final).before (fun _ => Bool.true)).getD
    ⟨[],.constBool Bool.false,[],[],[]⟩
private def finalBranch (BT BK BV : Nat) (gateK initial final : Bool) :=
  (Source.conditional? (loop BT BK BV gateK initial final).after (fun _ => Bool.true)).getD
    ⟨[],.constBool Bool.false,[],[],[]⟩
private def assignment (body : List ComputeStmt) (name : RegName) (occurrence : Nat := 0) :=
  (body.drop (Source.prefixBefore body name occurrence).length).head?

/-- Loading the bf16 seed is followed by an addition, not an assignment. -/
theorem seed_addition (BT BK BV : Nat) (gateK initial final : Bool) :
    assignment (seed BT BK BV gateK initial final).yes "b_h" = some
      (.assign .real [BK,BV] "b_h" (.compute (.alg .fp32
        (.add .real (.consSame (.consSame .nil)) (.ref .real [BK,BV] "b_h")
          (.castFloat .bf16 .real (.load .bf16
            (.blockPtr (.ref .blockPtr [BK,BV] "p_h") [0,1]) .none .unspecified)))))) := rfl

theorem history_storage (BT BK BV : Nat) (gateK initial final : Bool) :
    ((loop BT BK BV gateK initial final).body.drop
      (Source.prefixBeforeStore (loop BT BK BV gateK initial final).body).length).head? = some
      (.store .bf16 [BK,BV] (.blockPtr (.ref .blockPtr [BK,BV] "p_h") [0,1])
        (.alg (.castFloat .real .bf16 (.ref .real [BK,BV] "b_h"))) .none) := rfl

theorem final_storage (BT BK BV : Nat) (gateK initial final : Bool) :
    ((finalBranch BT BK BV gateK initial final).yes.drop
      (Source.prefixBeforeStore (finalBranch BT BK BV gateK initial final).yes).length).head? = some
      (.store .fp32 [BK,BV] (.blockPtr (.ref .blockPtr [BK,BV] "p_h") [0,1])
        (.alg (.castFloat .real .fp32 (.ref .real [BK,BV] "b_h"))) .none) := rfl

/-- The endpoint is at the full chunk end in a flattened parent of size T*K. -/
theorem key_endpoint (BT BK BV : Nat) (gateK initial final : Bool) :
    assignment (gate BT BK BV gateK initial final).yes "p_gn" = some
      (.assign .blockPtr [BK] "p_gn" (.alg (.makeBlockPtrDynOffsets "g"
        (.mul .nat .nil (.ref .nat [] "i_bh") (.constNat 15)) [15] [BK] [1]
        [.add .nat .nil
          (.mul .nat .nil (.sub .nat .nil
            (.add .nat .nil (.mul .nat .nil (.ref .nat [] "i_t") (.constNat BT)) (.constNat BT))
            (.constNat 1)) (.constNat 3))
          (.mul .nat .nil (.ref .nat [] "i_k") (.constNat BK))]))) := rfl

theorem intrinsic_key_gate (BT BK BV : Nat) (gateK initial final : Bool) :
    assignment (gate BT BK BV gateK initial final).yes "b_h" = some
      (.assign .real [BK,BV] "b_h" (.compute (.alg .fp32
        (.mul .real (.consSame (.consR .nil)) (.ref .real [BK,BV] "b_h")
          (.expandDim ⟨1,by simp⟩ (.exp (.castFloat .bf16 .real (.ref .bf16 [BK] "b_gn")))))))) := rfl

theorem intrinsic_value_gate (BT BK BV : Nat) (gateK initial final : Bool) :
    assignment (gate BT BK BV gateK initial final).no "b_h" = some
      (.assign .real [BK,BV] "b_h" (.compute (.alg .fp32
        (.mul .real (.consR (.consSame .nil)) (.ref .real [BK,BV] "b_h")
          (.expandDim ⟨0,by simp⟩ (.exp (.castFloat .bf16 .real (.ref .bf16 [BV] "b_gn")))))))) := rfl

/-- Both dot operands remain fp16 after the gated operand's storage cast. -/
theorem final_addition_precision (BT BK BV : Nat) (gateK initial final : Bool) :
    assignment (loop BT BK BV gateK initial final).body "b_h" = some
      (.assign .real [BK,BV] "b_h" (.compute (.alg .fp32
        (.add .real (.consSame (.consSame .nil)) (.ref .real [BK,BV] "b_h")
          (FPEquivProofs.blockDot .fp16 BK BT BV))))) := rfl

example (R : FP.ScalarArithmetic.Rules) : True := by
  fail_if_success
    have :
        ({ FPEquivProofs.recurrent .fp16 .bf16 .bf16 .bf16 .fp32 "k" "v" "g" "h" "h0" "ht"
          15 3 1 15 3 1 18 3 1 5 3 3 4 2 2 2 Bool.true Bool.true Bool.true with domain := [] } : Program) ≡[R]
        ({ FPEquivProofs.recurrentSumLeft .fp16 .bf16 .bf16 .bf16 .fp32 "k" "v" "g" "h" "h0" "ht"
          15 3 1 15 3 1 18 3 1 5 3 3 4 2 2 2 Bool.true Bool.true Bool.true with domain := [] } : Program) := by
      equiv_decompose; fp_prove
  trivial

private def ptrProgram (swapped : Bool) (rowOffset : Nat) : Program where
  kernel := .mk [] [] [.assign .blockPtr [2,2] "p" (.alg
    (.makeBlockPtrDynOffsets "x"
      (if swapped then .mul .nat .nil (.constNat 8) (.programId 0)
       else .mul .nat .nil (.programId 0) (.constNat 8)) [4,4] [2,2] [4,1]
      [.constNat rowOffset,.constNat 0]))]
  domain := []

theorem integer_block_pointer :
    ptrProgram Bool.false 0 ≡[([] : Spec.Assumptions FP.GuardedFragment)] ptrProgram Bool.true 0 := by
  equiv_decompose; fp_prove

-- The new base congruence must not silently change the pointer's offsets.
example : True := by
  fail_if_success
    have : ptrProgram Bool.false 0 ≡[([] : Spec.Assumptions FP.GuardedFragment)] ptrProgram Bool.true 1 := by
      equiv_decompose; fp_prove
  trivial

/-- info: FP assumptions used by cumulative_equiv:
---
info:   none -/
#guard_msgs in
#print_fp_assumptions FPEquivProofs.cumulative_equiv
/-- info: FP assumptions used by recurrent_equiv:
---
info:   add_commute -/
#guard_msgs in
#print_fp_assumptions FPEquivProofs.recurrent_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end FPChunkGatedSourceTests
