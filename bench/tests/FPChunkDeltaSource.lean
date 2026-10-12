import bench.tritonbench_g.chunk_delta_fwd.Proofs.FP

/-! These checks inspect the typed source before real erasure. They preserve
mixed storage, promotion at subtraction, dot precision, masks and both stores. -/
namespace FPChunkDeltaSourceTests
open VeriTile.Triton VeriTile.Bench.TritonBenchG.ChunkDeltaFwd
set_option maxRecDepth 16000

private def source (BC BK BV : Nat) (initial final : Bool) :=
  TypedSource.original .fp16 .bf16 .bf16 .fp32 "k" "v" "d" "v_new" "h" "h0" "ht"
    15 3 1 15 3 1 18 3 1 5 3 3 4 BC BK BV 2 initial final
private def outer (BC BK BV : Nat) (initial final : Bool) :=
  (Source.dynamicRange? (source BC BK BV initial final).surfaceBody "i_t").getD
    ⟨[], .constNat 0, .constNat 0, .constNat 0, [], []⟩
private def inner (BC BK BV : Nat) (initial final : Bool) :=
  (Source.dynamicRange? (outer BC BK BV initial final).body "i_c").getD
    ⟨[], .constNat 0, .constNat 0, .constNat 0, [], []⟩
private def assignment (body : List ComputeStmt) (name : RegName) (occurrence : Nat := 0) :=
  (body.drop (Source.prefixBefore body name occurrence).length).head?

/-- bf16 V widens before subtraction; both dot inputs remain fp16. -/
theorem residual_promotion (BC BK BV : Nat) (initial final : Bool) :
    assignment (inner BC BK BV initial final).body "b_v" 1 = some
      (.assign .fp32 [BC, BV] "b_v" (.alg
        (.sub .fp32 (.consSame (.consSame .nil))
          (.castFloat .bf16 .fp32 (.ref .bf16 [BC, BV] "b_v"))
          (.dotTyped .fp16 .fp16 .fp32 (batch := [])
            (.ref .fp16 [BC, BK] "b_d")
            (.castFloat .real .fp16 (.ref .real [BK, BV] "b_h"))
            {allowTF32 := some Bool.false})))) := rfl

/-- Corrected fp32 V converts back to K storage before the second dot. -/
theorem corrected_value_dot_cast (BC BK BV : Nat) (initial final : Bool) :
    assignment (inner BC BK BV initial final).body "b_h_cumsum" = some
      (.assign .real [BK, BV] "b_h_cumsum" (.compute (.alg .fp32
        (.add .real (.consSame (.consSame .nil)) (.ref .real [BK, BV] "b_h_cumsum")
          (.castFloat .fp32 .real (.dotTyped .fp16 .fp16 .fp32 (batch := [])
            (.ref .fp16 [BK, BC] "b_k")
            (.castFloat .fp32 .fp16 (.ref .fp32 [BC, BV] "b_v"))
            {allowTF32 := some Bool.false})))))) := rfl

theorem value_store_cast_and_masks (BC BK BV : Nat) (initial final : Bool) :
    ((inner BC BK BV initial final).body.drop
      (Source.prefixBeforeStore (inner BC BK BV initial final).body).length).head? = some
      (.store .bf16 [BC, BV] (.blockPtr (.ref .blockPtr [BC, BV] "p_v_new") [0, 1])
        (.alg (.castFloat .fp32 .bf16 (.ref .fp32 [BC, BV] "b_v"))) .none) := rfl

theorem history_precedes_inner_loop (BC BK BV : Nat) (initial final : Bool) :
    ((inner BC BK BV initial final).before.drop
      (Source.prefixBeforeStore (inner BC BK BV initial final).before).length).head? = some
      (.store .fp16 [BK, BV] (.blockPtr (.ref .blockPtr [BK, BV] "p_h") [0, 1])
        (.alg (.castFloat .real .fp16 (.ref .real [BK, BV] "b_h"))) .none) := rfl

theorem source_loop_bounds (BC BK BV : Nat) (initial final : Bool) :
    (outer BC BK BV initial final).stop = .constNat 2 ∧
    (inner BC BK BV initial final).stop =
      .div .nat .nil (.sub .nat .nil (.add .nat .nil (.constNat 4) (.constNat BC))
        (.constNat 1)) (.constNat BC) := ⟨rfl, rfl⟩

/-- info: FP assumptions used by sum_left_equiv:
---
info:   add_commute -/
#guard_msgs in
#print_fp_assumptions FPEquivProofs.sum_left_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end FPChunkDeltaSourceTests
