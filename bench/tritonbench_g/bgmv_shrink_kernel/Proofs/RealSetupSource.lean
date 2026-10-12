import bench.tritonbench_g.bgmv_shrink_kernel.Proofs.RealLoop
import bench.tritonbench_g.bgmv_shrink_kernel.Proofs.RealSource

/-! Decomposition of the entire precision-erased source. The signed metadata
load, sentinel guard, split-K loop and computed store/atomic choice remain
part of the program; internal ASTs are connected to that source below. -/
namespace VeriTile.Bench.TritonBenchG.BgmvShrinkKernel.RealSetupSource
open VeriTile.Triton
set_option maxHeartbeats 2200000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

structure Parameters where
  input : RegionName
  weights : RegionName
  output : RegionName
  indices : Region .int
  N : Nat
  K : Nat
  scaling : ℝ
  inputBatchStride : Nat
  inputStride : Nat
  slotStride : Nat
  rowStride : Nat
  colStride : Nat
  outputBatchStride : Nat
  outputStride : Nat
  BN : Nat
  BK : Nat
  splits : Nat

namespace Parameters

def source (p : Parameters) (storage : TypedSource.Storage)
    (out_dtype : FloatDType) (swapped : Bool) :=
  RealSource.source storage swapped out_dtype p.input p.weights p.output p.N p.K p.indices p.scaling
    p.inputBatchStride p.inputStride p.slotStride p.rowStride p.colStride
    p.outputBatchStride p.outputStride p.BN p.BK p.splits

def prelude (p : Parameters) : List Stmt :=
  [.assign .nat [] "pid_sk" (.programId 0),
   .assign .nat [] "cur_batch" (.programId 1),
   .assign .int [] "lora_index" (.load .int
     (.region p.indices (.ref .nat [] "cur_batch")) .none)]

def guard : Op .bool [] :=
  .ne .int .nil (.ref .int [] "lora_index") (.constInt (-1))

def weightPointer (p : Parameters) : Op .ptr [] :=
  .ptrAdd .nil (.ptrBase p.weights)
    (.castIntToNat (.mul .int .nil (.castNatToInt (.constNat p.slotStride))
      (.ref .int [] "lora_index")))

def setup (p : Parameters) : List Stmt :=
  [.assign .nat [p.BN] "offset_n" (.arange p.BN),
   .assign .nat [p.BK] "offset_k"
     (.add .nat .scalarR (.arange p.BK)
       (.mul .nat .nil (.ref .nat [] "pid_sk") (.constNat p.BK))),
   .assign .ptr [] "a_ptr" (.ptrAdd .nil (.ptrBase p.input)
     (.mul .nat .nil (.ref .nat [] "cur_batch") (.constNat p.inputBatchStride))),
   .assign .ptr [] "b_ptr" p.weightPointer,
   .assign .real [p.BN] "accumulator" (.full [p.BN] (.const 0))]

def loop (p : Parameters) : Stmt :=
  .forRange "k" 0 p.K (p.BK * p.splits)
    (shrinkLoopBody p.N p.K p.rowStride p.colStride p.BN p.BK)

def scale (p : Parameters) (swapped : Bool) : Stmt :=
  .assign .real [p.BN] "accumulator"
    (if swapped then .mul .real .scalarL (.const p.scaling) (.ref .real [p.BN] "accumulator")
     else .mul .real .scalarR (.ref .real [p.BN] "accumulator") (.const p.scaling))

def tail (p : Parameters) (swapped : Bool) : List Stmt :=
  p.scale swapped ::
    (shrinkTailCommon p.output p.N p.outputBatchStride p.outputStride p.BN p.scaling).drop 1 ++
  [.ifThenElse (.eq .nat .nil (.constNat p.splits) (.constNat 1))
    [shrinkStoreStmt p.BN] [shrinkAtomicStmt p.BN]]

def body (p : Parameters) (swapped : Bool) : List Stmt :=
  p.prelude ++ [.ifThen guard (p.setup ++ p.loop :: p.tail swapped)]

/-- All statements of both typed implementations occur in this decomposition.
No execution, branch outcome or output equation is assumed. -/
theorem source_body (p : Parameters) (storage : TypedSource.Storage)
    (out_dtype : FloatDType) (swapped : Bool) :
    (p.source storage out_dtype swapped).eraseDType.toAlgKernel.body = p.body swapped := by
  unfold source
  rw [RealSource.storage_erasure]
  cases swapped
  all_goals
    simp only [RealSource.source, TypedSource.specialize, ComputeKernel.surfaceBody,
      ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
      ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Kernel.eraseDType,
      Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def,
      MaskOpt.eraseDType.eq_def, VeriTile.Triton.eraseDType, NumericDType.eraseDType,
      ComparableDType.eraseDType, IntegralDType.eraseDType, Except.bind, bind]
    rfl

end Parameters
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.BgmvShrinkKernel.RealSetupSource
