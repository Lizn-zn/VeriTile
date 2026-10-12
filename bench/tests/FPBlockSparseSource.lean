import bench.tritonbench_g.block_sparse_attn.Proofs.FP

/-! Precision and signed-CSR checks on the complete source. Independent row
and column strides prevent the legacy num_heads/column-stride mix-up from
being hidden by a single-layout fixture. -/
namespace FPBlockSparseSourceTests
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.BlockSparseAttn

private def source (storage : TypedSource.Storage) (swapped evenM evenN : Bool) (blocks : Nat) :=
  TypedSource.specialize storage swapped "Out" "Q" "K" "V" ⟨"Rows"⟩ ⟨"Cols"⟩
    5 13 2 3 100 20 4 80 16 4 80 16 4 200 40 6 4 2 3 2 2 2 blocks evenM evenN

private def loop (storage : TypedSource.Storage) (swapped evenM evenN : Bool) (blocks : Nat) :=
  (Source.intRange? (source storage swapped evenM evenN blocks).surfaceBody "col_idx_idx").getD
    ⟨[], .constInt 0, .constInt 0, .constNat 0, [], []⟩

private def assignment (body : List ComputeStmt) (name : RegName) (occurrence : Nat := 0) :=
  (body.drop (Source.prefixBefore body name occurrence).length).head?

theorem actual_signed_bounds (storage : TypedSource.Storage) (swapped em en : Bool) (blocks : Nat) :
    ((loop storage swapped em en blocks).start, (loop storage swapped em en blocks).stop,
      (loop storage swapped em en blocks).stride) =
        (.ref .int [] "start_l", .ref .int [] "end_l", .constNat 1) := by
  cases storage <;> cases swapped <;> rfl

theorem row_pointer_load_is_signed (storage : TypedSource.Storage) (swapped em en : Bool) (blocks : Nat) :
    assignment (loop storage swapped em en blocks).before "start_l" = some
      (.assign .int [] "start_l" (.alg (.load .int (.ptr (.ref .ptr [] "layout_ptr")) .none))) := by
  cases storage <;> cases swapped <;> rfl

theorem signed_rank_product (storage : TypedSource.Storage) (swapped em en : Bool) (blocks : Nat) :
    assignment (loop storage swapped em en blocks).body "start_n" = some
      (.assign .int [] "start_n" (.alg (.mul .int .nil
        (.ref .int [] "col_idx") (.castNatToInt (.constNat 2))))) := by
  cases storage <;> cases swapped <;> rfl

/-- Combine the layout offset and signed visit before the address conversion.
For layout=1 and visit=-2 this reads column slot 11, rather than slot 13. -/
theorem signed_column_address (storage : TypedSource.Storage) (swapped em en : Bool) (blocks : Nat) :
    assignment (loop storage swapped em en blocks).body "col_idx" = some
      (.assign .int [] "col_idx" (.alg (.load .int
        (.region ⟨"Cols"⟩ (.castIntToNat (.add .int .nil
          (.castNatToInt (.mul .nat .nil (.ref .nat [] "layout_h") (.constNat 13)))
          (.ref .int [] "col_idx_idx")))) .none))) := by
  cases storage <;> cases swapped <;> rfl

theorem fp16_probability_cast (swapped em en : Bool) (blocks : Nat) :
    assignment (loop .fp16 swapped em en blocks).body "p" 2 = some
      (.assign .fp16 [2, 2] "p" (.alg (.castFloat .real .fp16 (.ref .real [2, 2] "p")))) := by
  cases swapped <;> rfl

theorem bf16_probability_cast (swapped em en : Bool) (blocks : Nat) :
    assignment (loop .bf16 swapped em en blocks).body "p" 2 = some
      (.assign .bf16 [2, 2] "p" (.alg (.castFloat .real .bf16 (.ref .real [2, 2] "p")))) := by
  cases swapped <;> rfl

theorem bf16_score_dot_and_fp32_carry (swapped em en : Bool) (blocks : Nat) :
    assignment (loop .bf16 swapped em en blocks).body "qk" 1 = some
      (.assign .real [2, 2] "qk" (.compute (.alg .fp32
        (.add .real (.consSame (.consSame .nil)) (.ref .real [2, 2] "qk")
          (.castFloat .fp32 .real (.dotTyped .bf16 .bf16 .fp32 (batch := [])
            (.ref .bf16 [2, 2] "q") (.ref .bf16 [2, 2] "k"))))))) := by
  cases swapped <;> rfl

theorem bf16_value_dot_and_fp32_carry (swapped em en : Bool) (blocks : Nat) :
    assignment (loop .bf16 swapped em en blocks).body "acc" 1 = some
      (.assign .real [2, 2] "acc" (.compute (.alg .fp32
        (.add .real (.consSame (.consSame .nil)) (.ref .real [2, 2] "acc")
          (.castFloat .fp32 .real (.dotTyped .bf16 .bf16 .fp32 (batch := [])
            (.ref .bf16 [2, 2] "p") (.ref .bf16 [2, 2] "v"))))))) := by
  cases swapped <;> rfl

theorem intrinsic_exp_preserved (storage : TypedSource.Storage) (swapped em en : Bool) (blocks : Nat) :
    assignment (loop storage swapped em en blocks).body "p" = some
      (.assign .real [2, 2] "p" (.compute (.alg .fp32
        (.exp (.sub .real (.consSame (.consR .nil)) (.ref .real [2, 2] "qk")
          (.expandDim ⟨1, by simp⟩ (.ref .real [2] "m_ij"))))))) := by
  cases storage <;> cases swapped <;> rfl

theorem numerical_domain_is_present (storage : TypedSource.Storage) (em en : Bool) (blocks : Nat) :
    (FPEquivProofs.requirementsFor (source storage Bool.false em en blocks) 2 2 3).length = 1 := by
  cases storage <;> rfl

theorem candidate_preserves_prefix_and_stores (storage : TypedSource.Storage) (em en : Bool) (blocks : Nat) :
    (loop storage Bool.false em en blocks).before = (loop storage Bool.true em en blocks).before ∧
    (loop storage Bool.false em en blocks).after = (loop storage Bool.true em en blocks).after := by
  cases storage <;> exact ⟨rfl, rfl⟩

/-- info: FP assumptions used by scaling_equiv:
---
info:   mul_commute -/
#guard_msgs in
#print_fp_assumptions FPEquivProofs.scaling_equiv

#guard_msgs (drop info) in
#auditModuleAxioms
end FPBlockSparseSourceTests
