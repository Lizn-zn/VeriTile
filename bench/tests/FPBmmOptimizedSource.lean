import bench.tritonbench_g.bmm_optimized.Proofs.FP

/-! Source-level regressions: omitted `other`, mask elision, typed dot options,
loop bounds and actual admitted atom. These do not certify value correctness. -/
namespace BmmOptimizedSourceTests
open VeriTile.Triton VeriTile.Bench.TritonBenchG.BmmOptimized

private def source (dtype : TypedSource.Storage) (dm dn dk swapped : Bool) :=
  TypedSource.specialize dtype dm dn dk swapped "A" "B" "O" 5 5 3 2 4 2 2
private def parts (dtype : TypedSource.Storage) (dm dn dk swapped : Bool) :=
  (Source.dynamicRange? (source dtype dm dn dk swapped).surfaceBody "_i").getD
    ⟨[], .constNat 0, .constNat 0, .constNat 0, [], []⟩
private def assignment (code : List ComputeStmt) (name : RegName) :=
  (code.drop (Source.prefixBefore code name).length).head?

theorem masked_load_keeps_undefined (swapped : Bool) :
    ∃ access mask, assignment (parts .fp16 Bool.false Bool.false Bool.false swapped).body "a" =
      some (.assign .fp16 [2,2] "a" (.alg (.load .fp16 access (.mask mask)))) := by
  cases swapped <;> exact ⟨_,_,rfl⟩

theorem unmasked_load_has_no_mask (swapped : Bool) :
    ∃ access, assignment (parts .bf16 Bool.true Bool.false Bool.true swapped).body "a" =
      some (.assign .bf16 [2,2] "a" (.alg (.load .bf16 access .none))) := by
  cases swapped <;> exact ⟨_,rfl⟩

theorem dot_options_and_precision (dm dn dk : Bool) :
    assignment (parts .fp16 dm dn dk Bool.false).body "o" =
      some (.assign .real [2,4] "o" (.compute (.alg .fp32
        (.add .real (.consSame (.consSame .nil)) (.ref .real [2,4] "o")
          (.castFloat .fp32 .real (.dotTyped .fp16 .fp16 .fp32 (batch := [])
            (.ref .fp16 [2,2] "a") (.ref .fp16 [2,4] "b")
            {allowTF32 := some Bool.false})))))) := by
  cases dm <;> cases dn <;> cases dk <;> rfl

theorem variants_keep_context (dtype : TypedSource.Storage) (dm dn dk : Bool) :
    (parts dtype dm dn dk Bool.false).before = (parts dtype dm dn dk Bool.true).before ∧
    (parts dtype dm dn dk Bool.false).after = (parts dtype dm dn dk Bool.true).after ∧
    (parts dtype dm dn dk Bool.false).stop = (parts dtype dm dn dk Bool.true).stop := by
  cases dtype <;> cases dm <;> cases dn <;> cases dk <;> exact ⟨rfl,rfl,rfl⟩

theorem actual_trip_count (dtype : TypedSource.Storage) (dm dn dk swapped : Bool) :
    assignment (parts dtype dm dn dk swapped).before "num_iters" =
      some (.assign .nat [] "num_iters" (.alg (.div .nat .nil
        (.sub .nat .nil (.add .nat .nil (.constNat 3) (.constNat 2)) (.constNat 1)) (.constNat 2)))) := by
  cases dtype <;> cases dm <;> cases dn <;> cases dk <;> cases swapped <;> rfl

/-- A fp32 accumulator is stored through the fp16 output channel. The source
makes Triton's implicit pointer-element conversion explicit before the store. -/
theorem output_storage_is_retained (dm dn dk swapped : Bool) :
    ∃ access value mask, (parts .fp16 dm dn dk swapped).after.getLast? =
      some (.store .fp16 [2,4] access value mask) := by
  cases dm <;> cases dn <;> cases dk <;> cases swapped <;> exact ⟨_,_,_,rfl⟩

/-- info: FP assumptions used by dot_first_equiv:
---
info:   add_commute -/
#guard_msgs in
#print_fp_assumptions FPEquivProofs.dot_first_equiv

#guard_msgs (drop info) in
#auditModuleAxioms
end BmmOptimizedSourceTests
