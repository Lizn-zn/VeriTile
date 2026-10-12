import bench.tritonbench_g.bgmv_expand_slice.Proofs.RealLoop

/-! Full real source decomposition, including metadata, the sentinel branch,
input loading, the computed split, the staged cast's register write and both
pointers. Internal ASTs are related to the entire erased source body below. -/
namespace VeriTile.Bench.TritonBenchG.BgmvExpandSlice.RealSetupSource
open VeriTile.Triton RealLoop
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
  inputBatchStride : Nat
  inputStride : Nat
  slotStride : Nat
  rowStride : Nat
  colStride : Nat
  outputBatchStride : Nat
  outputStride : Nat
  slice : Nat
  BN : Nat
  BK : Nat
  splits : Nat

namespace Parameters

def length (p : Parameters) := (p.N + p.splits - 1) / p.splits

def source (p : Parameters) (input : TypedSource.InputStorage)
    (weight : TypedSource.WeightStorage) (out_dtype : FloatDType)
    (cast swapped even add : Bool) :=
  RealSource.source input weight cast swapped out_dtype p.input p.weights p.output p.N p.K p.indices
    p.inputBatchStride p.inputStride p.slotStride p.rowStride p.colStride
    p.outputBatchStride p.outputStride p.slice p.BN p.BK p.splits even add

def geometry (p : Parameters) (initial : BlockState) (slot : Nat) : Geometry where
  weights := p.weights
  output := p.output
  weightBase := p.slotStride * slot + initial.pids 0 * p.length * p.rowStride
  rowStride := p.rowStride
  colStride := p.colStride
  outputBase := initial.pids 1 * p.outputBatchStride + initial.pids 0 * p.length + p.slice * p.outputStride
  outputStride := p.outputStride
  K := p.K
  length := p.length
  BN := p.BN
  BK := p.BK

def prelude (p : Parameters) : List Stmt :=
  [.assign .nat [] "pid_sn" (.programId 0),
   .assign .nat [] "cur_batch" (.programId 1),
   .assign .int [] "lora_index" (.load .int
     (.region p.indices (.ref .nat [] "cur_batch")) .none)]

def guard : Op .bool [] :=
  .ne .int .nil (.ref .int [] "lora_index") (.constInt (-1))

def inputAddress (p : Parameters) : MemAccess .real [p.BK] :=
  .region p.input (.add .nat .scalarL
    (.mul .nat .nil (.ref .nat [] "cur_batch") (.constNat p.inputBatchStride))
    (.mul .nat .scalarR (.ref .nat [p.BK] "offset_k") (.constNat p.inputStride)))

def inputLoad (p : Parameters) (even : Bool) : Op .real [p.BK] :=
  .load .real p.inputAddress (if even then .none else
    .maskOther (.lt .nat .scalarR (.ref .nat [p.BK] "offset_k") (.constNat p.K))
      ((Op.const (0 : ℝ)).broadcast [p.BK]))

def split : Parameters → Op .nat [] := fun p =>
  .div .nat .nil (.sub .nat .nil (.add .nat .nil (.constNat p.N) (.constNat p.splits))
    (.constNat 1)) (.constNat p.splits)

def weightPointer (p : Parameters) : Op .ptr [] :=
  .ptrAdd .nil (.ptrBase p.weights) (.castIntToNat (.add .int .nil
    (.mul .int .nil (.castNatToInt (.constNat p.slotStride))
      (.ref .int [] "lora_index"))
    (.castNatToInt (.mul .nat .nil (.mul .nat .nil (.ref .nat [] "pid_sn")
      (.ref .nat [] "split_n_length")) (.constNat p.rowStride)))))

def outputPointer (p : Parameters) : Op .ptr [] :=
  .ptrAdd .nil (.ptrBase p.output) (.add .nat .nil
    (.add .nat .nil (.mul .nat .nil (.ref .nat [] "cur_batch") (.constNat p.outputBatchStride))
      (.mul .nat .nil (.ref .nat [] "pid_sn") (.ref .nat [] "split_n_length")))
    (.mul .nat .nil (.constNat p.slice) (.constNat p.outputStride)))

def setup (p : Parameters) (cast even : Bool) : List Stmt :=
  [.assign .nat [p.BK] "offset_k" (.arange p.BK),
   .assign .nat [p.BN] "offset_n" (.arange p.BN),
   .ifThenElse (.constBool even)
     [.assign .real [p.BK] "tiled_a" (p.inputLoad Bool.true)]
     [.assign .real [p.BK] "tiled_a" (p.inputLoad Bool.false)],
   .assign .nat [] "split_n_length" p.split] ++
  (if cast then [.assign .real [p.BK] "tiled_a" (.ref .real [p.BK] "tiled_a")] else []) ++
  [.assign .ptr [] "b_ptr" p.weightPointer,
   .assign .ptr [] "c_ptr" p.outputPointer]

def loop (p : Parameters) (add swapped : Bool) : Stmt :=
  .forRangeDyn "n" (.constNat 0) (.ref .nat [] "split_n_length") (.constNat p.BN)
    (RealLoopSource.body p.rowStride p.colStride p.outputStride p.K p.BN p.BK add swapped)

def body (p : Parameters) (cast swapped even add : Bool) : List Stmt :=
  p.prelude ++ [.ifThen guard (p.setup cast even ++ [p.loop add swapped])]

/-- Every statement of the original and candidate, including the source
prefix, is accounted for. The cast erases to a real identity assignment. -/
theorem source_body (p : Parameters) (input : TypedSource.InputStorage)
    (weight : TypedSource.WeightStorage) (out_dtype : FloatDType)
    (cast swapped even add : Bool) :
    (p.source input weight out_dtype cast swapped even add).eraseDType.toAlgKernel.body =
      p.body cast swapped even add := by
  unfold source
  rw [RealSource.storage_erasure]
  cases cast <;> cases swapped
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
end VeriTile.Bench.TritonBenchG.BgmvExpandSlice.RealSetupSource
