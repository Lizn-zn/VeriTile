import bench.tritonbench_g.bgmv_shrink_kernel.Proofs.FP

/-! Full-source checks for dtype preservation, the exact split choice and
implicit output conversion. The unused input stride must not enter any load. -/
namespace FPBgmvShrinkSourceTests
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.BgmvShrinkKernel
set_option maxHeartbeats 2200000

private def source (storage : TypedSource.Storage) (swapped : Bool)
    (out_dtype : FloatDType) (K splits sk : Nat) :=
  TypedSource.specialize storage swapped out_dtype "X" "W" "Y" 3 K ⟨"Index"⟩ 3
    9 sk 40 9 1 13 2 4 2 splits

private def activeBody (kernel : ComputeKernel) :=
  ((Source.conditional? kernel.surfaceBody (fun _ => Bool.true)).map (·.yes)).getD []

private def loop? (code : List ComputeStmt) : Option (Nat × Nat × Nat × List ComputeStmt) :=
  code.findSome? fun st => match st with
    | .forRange "k" a b c body => some (a,b,c,body)
    | _ => none

private def loopBody (kernel : ComputeKernel) :=
  ((loop? (activeBody kernel)).map (·.2.2.2)).getD []

private def assignment (code : List ComputeStmt) (name : RegName) :=
  (code.drop (Source.prefixBefore code name).length).head?

private def loadDType? : ComputeStmt → Option TileDType
  | .assign _ _ _ (.alg (.load dtype _ _)) => some dtype
  | _ => none

private def dtype : TypedSource.Storage → TileDType
  | .fp16 => .fp16 | .bf16 => .bf16

section
variable (storage : TypedSource.Storage) (swapped : Bool)
    (out_dtype : FloatDType) (K splits sk sk' : Nat)

theorem both_offset_loads_keep_storage :
    (assignment (loopBody (source storage swapped out_dtype K splits sk)) "tiled_a").bind loadDType? = some (dtype storage) ∧
    (assignment (loopBody (source storage swapped out_dtype K splits sk)) "tiled_b").bind loadDType? = some (dtype storage) := by
  cases storage <;> cases swapped <;> exact ⟨rfl,rfl⟩

theorem source_does_not_use_input_stride :
    source storage swapped out_dtype K splits sk = source storage swapped out_dtype K splits sk' := by
  cases storage <;> cases swapped <;> rfl

theorem loop_keeps_split_stride :
    (loop? (activeBody (source storage swapped out_dtype K splits sk))).map
      (fun r => (r.1,r.2.1,r.2.2.1)) = some (0,K,2*splits) := by
  cases storage <;> cases swapped <;> rfl

theorem both_tails_keep_output_conversion_and_mask :
    (activeBody (source storage swapped out_dtype K splits sk)).getLast? = some
      (.ifThenElse (.eq .nat .nil (.constNat splits) (.constNat 1))
        [.store out_dtype.toTileDType [4] (.ptr (.ref .ptr [4] "c_ptr"))
          (.alg (.castFloat .real out_dtype (.ref .real [4] "accumulator")))
          (.mask (.ref .bool [4] "c_mask"))]
        [.atomicAdd out_dtype.numericDType [4] (.ptr (.ref .ptr [4] "c_ptr"))
          (.alg (.castFloat .real out_dtype (.ref .real [4] "accumulator")))
          (.mask (.ref .bool [4] "c_mask"))]) := by
  cases storage <;> cases swapped <;> rfl

theorem complete_io_signature :
    (source storage swapped out_dtype K splits sk).inputs = ["X","W","Y","Index"] ∧
    (source storage swapped out_dtype K splits sk).outputs = ["Y"] := by
  cases storage <;> cases swapped <;> exact ⟨rfl,rfl⟩

end

/-- Products round in bf16 before conversion into the fp32 sum/accumulator. -/
theorem bf16_product_and_fp32_accumulation :
    assignment (loopBody (source .bf16 Bool.false .fp16 7 2 17)) "accumulator" = some
      (.assign .real [4] "accumulator" (.compute (.alg .fp32
        (.add .real (.consSame .nil) (.ref .real [4] "accumulator")
          (.reduceSum ⟨1, by decide⟩ Bool.false (.castFloat .bf16 .real
            (.mul .bf16 (.leadL (.consSame .nil)) (.ref .bf16 [2] "tiled_a")
              (.ref .bf16 [4,2] "tiled_b")))))))) := rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end FPBgmvShrinkSourceTests
