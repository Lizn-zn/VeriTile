import bench.tritonbench_g.batched_vecmat_mult.Proofs.FP

/-! Precision and geometry regressions for the complete vector-matrix source.
The asymmetric tile detects accidental reduction/transpose or broadcast-shape
changes. Both candidates must keep the same dynamic bounds, loads and store. -/
namespace FPBatchedVecmatSourceTests
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.BatchedVecmatMult

private def source (candidate : Bool) :=
  (if candidate then TypedSource.blockFirst else TypedSource.original) "A" "B" "Out" 4 6 4 2 3 2
private def range (candidate : Bool) :=
  Source.dynamicRange? (source candidate).surfaceBody "k_index"
private def body (candidate : Bool) := ((range candidate).map (·.body)).getD []
private def assignment (code : List ComputeStmt) (name : RegName) :=
  (code.drop (Source.prefixBefore code name).length).head?

section
variable (candidate : Bool)

theorem dynamic_key_loop :
    (range candidate).map (fun r => (r.start,r.stop,r.stride)) =
      some (.constNat 0, .ref .nat [] "k_blocks", .constNat 1) := by
  cases candidate <;> rfl

theorem query_load_fp32 : assignment (body candidate) "a" = some
    (.assign .real [2,2] "a" (.compute
      (.load .fp32 (.region "A" (.ref .nat [2,2] "a_tile")) .none))) := by
  cases candidate <;> rfl

theorem matrix_load_fp32 : assignment (body candidate) "b" = some
    (.assign .real [3,2,2] "b" (.compute
      (.load .fp32 (.region "B" (.ref .nat [3,2,2] "b_tile")) .none))) := by
  cases candidate <;> rfl

theorem broadcast_keeps_unit_axis : assignment (body candidate) "expanded_a" = some
    (.assign .real [1,2,2] "expanded_a" (.alg
      (.expandDim ⟨0, by simp⟩ (.ref .real [2,2] "a")))) := by
  cases candidate <;> rfl

theorem candidate_preserves_input_prefix :
    Source.prefixBefore (body candidate) "vecmat" =
      Source.prefixBefore (body Bool.false) "vecmat" := by
  cases candidate <;> rfl

/-- fp32 is the execution profile; the stored value is the already computed
fp32 accumulator carried by the algorithm-typed .real register. -/
theorem complete_unmasked_store : (source candidate).surfaceBody.getLast? = some
    (.store .real [2,3] (.region "Out" (.ref .nat [2,3] "output_tile"))
      (.alg (.ref .real [2,3] "vecmat")) .none) := by
  cases candidate <;> rfl

end

theorem original_update : assignment (body Bool.false) "vecmat" = some
    (.assign .real [2,3] "vecmat" (.alg
      (.add .real (.consSame (.consSame .nil)) (.ref .real [2,3] "vecmat")
        (FPEquivProofs.blockSum 2 3 2)))) := rfl

theorem block_first_update : assignment (body Bool.true) "vecmat" = some
    (.assign .real [2,3] "vecmat" (.alg
      (.add .real (.consSame (.consSame .nil)) (FPEquivProofs.blockSum 2 3 2)
        (.ref .real [2,3] "vecmat")))) := rfl

/-- The symbolic source always supplies its loop guard; a failed extraction
cannot silently turn requirements into an empty list. -/
theorem symbolic_requirements_present (A B out : RegionName) (M N K BM BN BK : Nat) :
    (FPEquivProofs.requirements A B out M N K BM BN BK).length = 1 := rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end FPBatchedVecmatSourceTests
