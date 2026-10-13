import bench.tritonbench_g.bgmv_expand_slice.Kernels

/-! Source regressions for all input/weight/cast choices, with a partial K tile
and two output-loop steps. Offsetted loads retain their storage dtypes. The
checks also prevent a failed branch/loop locator from removing the domain. -/
namespace FPBgmvExpandSliceSourceTests
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.BgmvExpandSlice
set_option maxHeartbeats 2200000

private def source (input : TypedSource.InputStorage) (weight : TypedSource.WeightStorage)
    (cast swapped : Bool) (out_dtype : FloatDType) (even add : Bool) :=
  TypedSource.specialize input weight cast swapped out_dtype "X" "W" "Y" 5 3 ⟨"Index"⟩
    3 1 15 3 1 5 1 0 2 4 2 even add

private def activeBody (kernel : ComputeKernel) :=
  ((Source.conditional? kernel.surfaceBody
    (fun body => (Source.dynamicRange? body "n").isSome)).map (·.yes)).getD []

private def loopBody (kernel : ComputeKernel) :=
  ((Source.dynamicRange? (activeBody kernel) "n").map (fun (r : Source.DynamicRange) => r.body)).getD []

private def yesBody (code : List ComputeStmt) :=
  ((Source.conditional? code (fun _ => Bool.true)).map (·.yes)).getD []

private def assignment (code : List ComputeStmt) (name : RegName) :=
  (code.drop (Source.prefixBefore code name).length).head?

private def loadDType? : ComputeStmt → Option TileDType
  | .assign _ _ _ (.alg (.load dtype _ _)) => some dtype
  | .assign _ _ _ (.compute (.load .fp32 _ _)) => some .fp32
  | _ => none

private def assignedLoad (code : List ComputeStmt) (name : RegName) :=
  (assignment code name).bind loadDType?

private def inputDType : TypedSource.InputStorage → TileDType
  | .fp16 => .fp16 | .bf16 => .bf16 | .fp32 => .fp32
private def weightDType : TypedSource.WeightStorage → TileDType
  | .fp16 => .fp16 | .bf16 => .bf16

section
variable (input : TypedSource.InputStorage) (weight : TypedSource.WeightStorage)
    (cast swapped : Bool) (out_dtype : FloatDType) (even add : Bool)

theorem input_load_keeps_storage :
    assignedLoad (yesBody (activeBody (source input weight cast swapped out_dtype even add))) "tiled_a" =
      some (inputDType input) := by
  cases input <;> cases weight <;> cases cast <;> cases swapped <;> cases out_dtype <;> rfl

theorem offsetted_weight_load_keeps_storage :
    assignedLoad (loopBody (source input weight cast swapped out_dtype even add)) "tiled_b" =
      some (weightDType weight) := by
  cases input <;> cases weight <;> cases cast <;> cases swapped <;> cases out_dtype <;> rfl

theorem old_output_load_keeps_storage :
    assignedLoad (yesBody (loopBody (source input weight cast swapped out_dtype even add))) "tiled_out" =
      some out_dtype.toTileDType := by
  cases input <;> cases weight <;> cases cast <;> cases swapped <;> cases out_dtype <;> rfl

private def assignedDType? : ComputeStmt → Option TileDType
  | .assign dtype _ _ _ => some dtype
  | _ => none

theorem cast_rebinds_to_weight_dtype :
    (assignment (activeBody (source input weight cast swapped out_dtype even add)) "tiled_a").bind assignedDType? =
      if cast then some (weightDType weight) else none := by
  cases input <;> cases weight <;> cases cast <;> cases swapped <;> cases out_dtype <;> rfl

private def storedDType? : ComputeStmt → Option TileDType
  | .store _ _ _ (.compute (.alg .fp32 _)) _ => some .fp32
  | .store dtype _ _ _ _ => some dtype
  | _ => none

theorem store_keeps_output_dtype :
    ((loopBody (source input weight cast swapped out_dtype even add)).getLast?).bind storedDType? =
      some out_dtype.toTileDType := by
  cases input <;> cases weight <;> cases cast <;> cases swapped <;> cases out_dtype <;> rfl

theorem loop_keeps_dynamic_bounds :
    (Source.dynamicRange? (activeBody (source input weight cast swapped out_dtype even add)) "n").map
      (fun r => (r.start,r.stop,r.stride)) =
      some (.constNat 0, .ref .nat [] "split_n_length", .constNat 2) := by
  cases input <;> cases weight <;> cases cast <;> cases swapped <;> cases out_dtype <;> rfl

theorem complete_io_signature :
    (source input weight cast swapped out_dtype even add).inputs = ["X","W","Y","Index"] ∧
    (source input weight cast swapped out_dtype even add).outputs = ["Y"] := by
  cases input <;> cases weight <;> cases cast <;> cases swapped <;> cases out_dtype <;> exact ⟨rfl,rfl⟩

end

private def halfSum : Op .fp16 [2] :=
  .reduceSumTyped .fp16 ⟨1, by decide⟩ Bool.false
    (.mul .fp16 (.leadL (.consSame .nil)) (.ref .fp16 [4] "tiled_a")
      (.ref .fp16 [2, 4] "tiled_b"))

/-- Half output requires a half addition of the actual half reduction. -/
theorem half_sum_and_addition_keep_dtype :
    assignment (yesBody (loopBody
      (source .fp16 .fp16 Bool.false Bool.false .fp16 Bool.false Bool.true))) "accumulator" =
      some (.assign .fp16 [2] "accumulator" (.alg (.castFloat .fp16 .fp16
        (.add .fp16 (.consSame .nil) halfSum (.ref .fp16 [2] "tiled_out"))))) := rfl

/-- Promotion occurs after the typed reduction when output storage is fp32. -/
theorem fp32_output_promotes_after_half_sum :
    assignment (yesBody (loopBody
      (source .fp16 .fp16 Bool.false Bool.false .fp32 Bool.false Bool.true))) "accumulator" =
      some (.assign .real [2] "accumulator" (.compute (.alg .fp32 (.castFloat .real .real
        (.add .real (.consSame .nil) (.castFloat .fp16 .real halfSum)
          (.ref .real [2] "tiled_out")))))) := rfl

/-- The candidate swaps exactly these operands without widening their arithmetic. -/
theorem half_candidate_preserves_sum_dtype :
    assignment (yesBody (loopBody
      (source .fp16 .fp16 Bool.false Bool.true .fp16 Bool.false Bool.true))) "accumulator" =
      some (.assign .fp16 [2] "accumulator" (.alg (.castFloat .fp16 .fp16
        (.add .fp16 (.consSame .nil) (.ref .fp16 [2] "tiled_out") halfSum)))) := rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end FPBgmvExpandSliceSourceTests
