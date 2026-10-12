import bench.tritonbench_g.chunk_gate_recurrence.Proofs.FP

/-! Regressions for storage promotion, the complete DG reduction, both rewrite
sites and reverse pointer placement. These inspect the actual typed source. -/
namespace FPChunkGateSourceTests
open VeriTile.Triton VeriTile.Bench.TritonBenchG.ChunkGateRecurrence

private def fwd (swapped hasSeed : Bool) :=
  TypedSource.specializeForward .fp16 .bf16 swapped .fp32
    "S" "D" "O" "Seed" 2 3 4 4 2 2 hasSeed
private def bwd (swapped : Bool) :=
  TypedSource.specializeBackward .bf16 .fp32 swapped .fp16
    "S" "D" "DI" "DG" "DL" "DS" 2 3 4 4 2 2
private def parts (kernel : ComputeKernel) : Source.DynamicRange :=
  (Source.dynamicRange? kernel.surfaceBody "_i").getD
    ⟨[], .constNat 0, .constNat 0, .constNat 0, [], []⟩
private def assignment (body : List ComputeStmt) (name : RegName) :=
  (body.drop (Source.prefixBefore body name).length).head?
private def storeAt (body : List ComputeStmt) (occurrence : Nat := 0) :=
  (body.drop (Source.prefixBeforeStore body occurrence).length).head?

/-- Both operands are converted from their independent storage channels before
fp32 addition. The gate conversion occurs inside the fp32 multiplication. -/
theorem forward_precision (hasSeed : Bool) :
    assignment (parts (fwd Bool.false hasSeed)).body "acc" = some
      (.assign .real [2,2] "acc" (.compute (.alg .fp32
        (.add .real (.consSame (.consSame .nil))
          (.mul .real .scalarR (.ref .real [2,2] "acc")
            (.castFloat .bf16 .real (.ref .bf16 [] "d_i")))
          (.castFloat .fp16 .real (.ref .fp16 [2,2] "S_i")))))) := rfl

theorem forward_load_before_cast (swapped hasSeed : Bool) :
    assignment (parts (fwd swapped hasSeed)).body "S_i" = some
      (.assign .fp16 [2,2] "S_i" (.alg (.load .fp16 (.ptr (.ref .ptr [2,2] "S")) .none))) := by
  cases swapped <;> rfl

/-- No-axis tl.sum contracts the whole tile in fp32, not just one row/column.
The source S load keeps fp16 storage until the explicit conversion. -/
theorem backward_full_tile_sum (swapped : Bool) :
    assignment (parts (bwd swapped)).body "DG_i" = some
      (.assign .real [] "DG_i" (.compute (.alg .fp32
        (.reduceSum ⟨0, by decide⟩ Bool.false (.reshape [4]
          (.mul .real (.consSame (.consSame .nil)) (.ref .real [2,2] "Dacc")
            (.castFloat .fp16 .real (.ref .fp16 [2,2] "S_i")))))))) := by
  cases swapped <;> rfl

theorem backward_final_update :
    assignment (parts (bwd Bool.false)).after "Dacc" = some
      (.assign .real [2,2] "Dacc" (.compute (.alg .fp32
        (.add .real (.consSame (.consSame .nil))
          (.mul .real .scalarR (.ref .real [2,2] "Dacc")
            (.castFloat .bf16 .real (.ref .bf16 [] "d_i")))
          (.ref .real [2,2] "DS_i"))))) := rfl

/-- Two guards: each reached reverse iteration, then the separate post-loop
update. The latter prefix includes the ORIGINAL loop and subsequent loads. -/
theorem backward_final_guard :
    (FPEquivProofs.backwardRequirements (bwd Bool.false) .bf16 .fp32 2 2)[1]? = some
      (.addFP32 (Source.prefixBefore (bwd Bool.false).surfaceBody "Dacc" 1) [2,2]
        (FPEquivProofs.product .bf16 2 2 "Dacc") (.ref .real [2,2] "DS_i")) := rfl

theorem unchanged_initialization (hasSeed : Bool) :
    (parts (fwd Bool.false hasSeed)).before = (parts (fwd Bool.true hasSeed)).before ∧
    (parts (bwd Bool.false)).before = (parts (bwd Bool.true)).before := ⟨rfl,rfl⟩

/-- Pointer element-type attributes must resolve to physical output storage,
not silently drop the conversion and store an fp32 accumulator. -/
theorem forward_store_precision (hasSeed : Bool) :
    storeAt (parts (fwd Bool.false hasSeed)).before = some
      (.store .fp16 [2,2] (.ptr (.ref .ptr [2,2] "O"))
        (.alg (.castFloat .real .fp16 (.ref .real [2,2] "acc"))) .none) := rfl

theorem gate_store_precision (swapped : Bool) :
    storeAt (parts (bwd swapped)).body = some
      (.store .bf16 [] (.ptr (.ref .ptr [] "DG"))
        (.alg (.castFloat .real .bf16 (.ref .real [] "DG_i"))) .none) := by
  cases swapped <;> rfl

theorem boundary_store_precision (swapped : Bool) :
    storeAt (parts (bwd swapped)).after = some
      (.store .fp16 [2,2] (.ptr (.ref .ptr [2,2] "DL"))
        (.alg (.castFloat .real .fp16 (.ref .real [2,2] "Dacc"))) .none) := by
  cases swapped <;> rfl

/-- info: FP assumptions used by forward_equiv:
---
info:   add_commute -/
#guard_msgs in
#print_fp_assumptions FPEquivProofs.forward_equiv

/-- info: FP assumptions used by backward_equiv:
---
info:   add_commute -/
#guard_msgs in
#print_fp_assumptions FPEquivProofs.backward_equiv

#guard_msgs (drop info) in
#auditModuleAxioms
end FPChunkGateSourceTests
