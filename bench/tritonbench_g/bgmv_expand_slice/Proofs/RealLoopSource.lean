import bench.tritonbench_g.bgmv_expand_slice.Proofs.RealBlock

/-! Exact source correspondence for the complete output loop. The equations
below preserve the calculated split register, loop bounds, weight/output masks,
old-output load, both accumulation branches and the store. This is a source
bridge for the full real proof, not a standalone correctness specification. -/
namespace VeriTile.Bench.TritonBenchG.BgmvExpandSlice.RealLoopSource
open VeriTile.Triton
open RealBlock
set_option maxHeartbeats 2000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

def loadPrefix (rowStride colStride K BN BK : Nat) : List Stmt :=
  [Stmt.assign TileDType.nat [BN] "current_n"
      (Op.add NumericDType.nat Broadcast.scalarL (Op.ref TileDType.nat [] "n")
        (Op.ref TileDType.nat [BN] "offset_n")),
    Stmt.assign TileDType.bool [BN, BK] "b_ptr_mask"
      (Op.boolAnd Broadcast.nil.consL.consR
        (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref TileDType.nat [BN] "current_n"))
          (Op.ref .nat [] "split_n_length"))
        (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref TileDType.nat [BK] "offset_k"))
          (Op.constNat K))),
    Stmt.assign TileDType.bool [BN] "c_mask"
      (Op.lt ComparableDType.nat Broadcast.scalarR (Op.ref TileDType.nat [BN] "current_n") (Op.ref .nat [] "split_n_length")),
    Stmt.assign TileDType.real [BN, BK] "tiled_b"
      (Op.load TileDType.real
        (MemAccess.ptr
          (Op.ptrAdd Broadcast.nil.consL.consR
            (Op.ptrAdd Broadcast.scalarL (Op.ref TileDType.ptr [] "b_ptr")
              (Op.mul NumericDType.nat Broadcast.scalarR
                (Op.expandDim ⟨1, by simp⟩ (Op.ref TileDType.nat [BN] "current_n")) (Op.constNat rowStride)))
            (Op.mul NumericDType.nat Broadcast.scalarR
              (Op.expandDim ⟨0, by simp⟩ (Op.ref TileDType.nat [BK] "offset_k")) (Op.constNat colStride))))
        (MaskOpt.maskOther (Op.ref TileDType.bool [BN, BK] "b_ptr_mask")
          ((Op.const (0.0:ℝ)).broadcast [BN, BK])))]

def body (rowStride colStride outputStride K BN BK : Nat) (add swapped : Bool) : List Stmt :=
  loadPrefix rowStride colStride K BN BK ++ tail outputStride BN BK add swapped

/-- Retain the complete loop statement from the final active branch. The
source equation below checks both successful location and every loop bound. -/
def locatedLoop (kernel : ComputeKernel) : Option Stmt :=
  match kernel.toAlgKernel.body.getLast? with
  | some (.ifThen _ active) => active.getLast?
  | _ => none

theorem source_loop (input : TypedSource.InputStorage) (weight : TypedSource.WeightStorage)
    (CAST_TYPE swapped : Bool) (out_dtype : FloatDType)
    (x w out : RegionName) (N K : Nat) (indices : Region .int)
    (sx sk sw0 swn swk sy syn slice BN BK splits : Nat) (EVEN_K ADD_INPUTS : Bool) :
    locatedLoop (RealSource.source input weight CAST_TYPE swapped out_dtype x w out N K indices
      sx sk sw0 swn swk sy syn slice BN BK splits EVEN_K ADD_INPUTS).eraseDType =
    some (.forRangeDyn "n" (.constNat 0) (.ref .nat [] "split_n_length") (.constNat BN)
      (body swn swk syn K BN BK ADD_INPUTS swapped)) := by
  rw [RealSource.storage_erasure]
  cases CAST_TYPE <;> cases swapped
  all_goals
    simp only [locatedLoop, RealSource.source, TypedSource.specialize, ComputeKernel.surfaceBody,
      ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
      ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Kernel.eraseDType,
      Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def,
      MaskOpt.eraseDType.eq_def, VeriTile.Triton.eraseDType, NumericDType.eraseDType,
      ComparableDType.eraseDType, IntegralDType.eraseDType, Except.bind, bind]
    rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.BgmvExpandSlice.RealLoopSource
