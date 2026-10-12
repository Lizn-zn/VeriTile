import bench.tritonbench_g.chunk_gla_fwd.Proofs.FPSplit

/-! Source-fidelity checks for the split intra-subblock producer. In particular,
masked half/bfloat loads must precede their fp32 casts; the full column loop,
causal mask, implicit fp32 store and K-strided pointer advances remain present.
An empty numerical domain must not prove the scaling rewrite. -/
namespace FPChunkGLASplitTests
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.ChunkGlaFwd
open scoped VeriTile.Spec
set_option maxHeartbeats 4000000
set_option maxRecDepth 18000

private def source (T K BT BC BK NC sh st : Nat) :=
  TypedSplit.original .fp16 .bf16 .fp16 "q" "k" "g" "A" sh st 2 T K BT BC BK NC
private def active (T K BT BC BK NC sh st : Nat) :=
  (Source.conditional? (source T K BT BC BK NC sh st).surfaceBody (fun _ => Bool.true)).getD
    ⟨[],.constBool Bool.false,[],[],[]⟩
private def loop (T K BT BC BK NC sh st : Nat) :=
  (Source.dynamicRange? (active T K BT BC BK NC sh st).yes "j").getD
    ⟨[],.constNat 0,.constNat 0,.constNat 0,[],[]⟩
private def assignment (body : List ComputeStmt) (name : RegName) (occurrence : Nat := 0) :=
  (body.drop (Source.prefixBefore body name occurrence).length).head?

section
variable (T K BT BC BK NC sh st : Nat)
theorem source_return : (active T K BT BC BK NC sh st).condition =
    .boolNot (.ge .nat .nil (.add .nat .nil (.mul .nat .nil (.ref .nat [] "i_t") (.constNat BT))
      (.mul .nat .nil (.ref .nat [] "i_i") (.constNat BC))) (.constNat T)) := rfl

theorem runtime_heads : assignment (active T K BT BC BK NC sh st).before "n_bh" =
    some (.assign .nat [] "n_bh" (.alg (.numPrograms 2))) := rfl

theorem feature_grid : assignment (loop T K BT BC BK NC sh st).before "o_k" = some
    (.assign .nat [BK] "o_k" (.alg (.add .nat .scalarL
      (.mul .nat .nil (.ref .nat [] "i_k") (.constNat BK)) (.arange BK)))) := rfl

theorem zero_accumulator : assignment (loop T K BT BC BK NC sh st).body "b_A" = some
    (.assign .real [BC] "b_A" (.compute (.full [BC] (.alg .fp32 (.const 0))))) := rfl

theorem sum_added_to_accumulator : assignment (loop T K BT BC BK NC sh st).body "b_A" 1 = some
    (.assign .real [BC] "b_A" (.compute (.alg .fp32
      (.add .real (.consSame .nil) (.ref .real [BC] "b_A")
        (.reduceSum ⟨1,by simp⟩ Bool.false
          (.mul .real (.consSame (.consSame .nil))
            (.mul .real (.consR (.consSame .nil)) (.castFloat .fp16 .real (.ref .fp16 [BC,BK] "b_q"))
              (.expandDim ⟨0,by simp⟩ (.ref .real [BK] "b_k")))
            (.exp (.sub .real (.consR (.consSame .nil))
              (.castFloat .fp16 .real (.ref .fp16 [BC,BK] "b_g"))
              (.expandDim ⟨0,by simp⟩ (.ref .real [BK] "b_gk")))))))))) := rfl

theorem temporary_slice_stride : assignment (loop T K BT BC BK NC sh st).before "o_A" = some
    (.assign .nat [BC] "o_A" (.alg (.add .nat .scalarL
      (.mul .nat .nil (.mul .nat .nil
        (.add .nat .nil (.ref .nat [] "i_bh") (.mul .nat .nil (.ref .nat [] "i_k") (.ref .nat [] "n_bh")))
        (.constNat T)) (.constNat BC))
      (.mul .nat .scalarR
        (.add .nat .scalarL
          (.add .nat .nil (.mul .nat .nil (.ref .nat [] "i_t") (.constNat BT))
            (.mul .nat .nil (.ref .nat [] "i_i") (.constNat BC))) (.arange BC)) (.constNat BC))))) := rfl

theorem key_load_then_cast : assignment (loop T K BT BC BK NC sh st).body "b_k" = some
    (.assign .real [BK] "b_k" (.compute (.alg .fp32
      (.castFloat .bf16 .real (.load .bf16 (.ptr (.ref .ptr [BK] "p_k"))
        (.maskOther (.ref .bool [BK] "m_k") (.broadcast (.castFloat .real .bf16 (.const 0)) [BK]))))))) := rfl

theorem gate_load_then_cast : assignment (loop T K BT BC BK NC sh st).body "b_gk" = some
    (.assign .real [BK] "b_gk" (.compute (.alg .fp32
      (.castFloat .fp16 .real (.load .fp16 (.ptr (.ref .ptr [BK] "p_gk"))
        (.maskOther (.ref .bool [BK] "m_k") (.broadcast (.castFloat .real .fp16 (.const 0)) [BK]))))))) := rfl

theorem unspecified_matrix_gate : assignment (loop T K BT BC BK NC sh st).before "b_g" = some
    (.assign .fp16 [BC,BK] "b_g" (.alg (.load .fp16
      (.blockPtr (.ref .blockPtr [BC,BK] "p_g") [0,1]) .none .unspecified))) := rfl

theorem causal_where : assignment (loop T K BT BC BK NC sh st).body "b_A" 2 = some
    (.assign .real [BC] "b_A" (.alg (.where
      (.ge .nat .scalarR (.ref .nat [BC] "o_i") (.ref .nat [] "j"))
      (.mul .real .scalarR (.ref .real [BC] "b_A") (.const 2)) (.broadcast (.const 0.0) [BC])))) := rfl

theorem fp32_store : ((loop T K BT BC BK NC sh st).body.drop
    (Source.prefixBeforeStore (loop T K BT BC BK NC sh st).body).length).head? = some
    (.store .real [BC] (.region "A" (.add .nat .scalarR (.ref .nat [BC] "o_A") (.ref .nat [] "j")))
      (.alg (.ref .real [BC] "b_A")) (.mask (.ref .bool [BC] "m_A"))) := rfl

theorem key_advance : assignment (loop T K BT BC BK NC sh st).body "p_k" = some
    (.assign .ptr [BK] "p_k" (.alg (.ptrAdd .scalarR (.ref .ptr [BK] "p_k") (.constNat K)))) := rfl

theorem gate_advance : assignment (loop T K BT BC BK NC sh st).body "p_gk" = some
    (.assign .ptr [BK] "p_gk" (.alg (.ptrAdd .scalarR (.ref .ptr [BK] "p_gk") (.constNat K)))) := rfl

theorem all_valid_columns : (loop T K BT BC BK NC sh st).stop =
    .where (.lt .nat .nil (.constNat BC)
      (.sub .nat .nil (.sub .nat .nil (.constNat T) (.mul .nat .nil (.ref .nat [] "i_t") (.constNat BT)))
        (.mul .nat .nil (.ref .nat [] "i_i") (.constNat BC)))) (.constNat BC)
      (.sub .nat .nil (.sub .nat .nil (.constNat T) (.mul .nat .nil (.ref .nat [] "i_t") (.constNat BT)))
        (.mul .nat .nil (.ref .nat [] "i_i") (.constNat BC))) := rfl
end

theorem domain_present :
    (FPSplit.original .fp16 .bf16 .fp16 "q" "k" "g" "A" 40 5 2 7 3 4 2 4 2).domain.length = 1 := rfl

example (R : FP.ScalarArithmetic.Rules) : True := by
  fail_if_success
    have :
        ({ FPSplit.original .fp16 .bf16 .fp16 "q" "k" "g" "A" 40 5 2 7 3 4 2 4 2 with domain := [] } : Program) ≡[R]
        ({ FPSplit.scaleFirst .fp16 .bf16 .fp16 "q" "k" "g" "A" 40 5 2 7 3 4 2 4 2 with domain := [] } : Program) := by
      equiv_decompose; fp_prove
  trivial

/-- info: FP assumptions used by split_equiv:
---
info:   mul_commute -/
#guard_msgs in
#print_fp_assumptions FPSplit.split_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end FPChunkGLASplitTests
