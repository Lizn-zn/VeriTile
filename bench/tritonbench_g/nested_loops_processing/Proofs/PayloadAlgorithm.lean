import bench.tritonbench_g.nested_loops_processing.Proofs.RealInitialize
import bench.tritonbench_g.nested_loops_processing.Proofs.SourceChecks

namespace VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.PayloadAlgorithm
open VeriTile.Triton TypedSource
set_option maxHeartbeats 1000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

/-- The payload carrier varies; the exact signed pointer expressions are the
ones extracted from the source. No loop or load is replaced by its output. -/
def inputAdvance (src dst : RegionName) (sm sn : Int) : Stmt :=
  ((FullReal.inner src dst sm sn)[0]?).getD (.forRange "" 0 0 1 [])

def outputAdvance (src dst : RegionName) (sm sn : Int) : Stmt :=
  ((FullReal.inner src dst sm sn)[3]?).getD (.forRange "" 0 0 1 [])

def load (dtype : TileDType) (name : RegName) : Stmt :=
  .assign dtype [2,2] name (.load dtype (.ptr (.ref .ptr [2,2] "a_ptrs")) .none)

def store (dtype : TileDType) (name : RegName) : Stmt :=
  .store dtype [2,2] (.ptr (.ref .ptr [2,2] "c_ptrs")) (.ref dtype [2,2] name) .none

def inner (dtype : TileDType) (src dst : RegionName) (sm sn : Int) : List Stmt :=
  [inputAdvance src dst sm sn, load dtype "a3", store dtype "a1", outputAdvance src dst sm sn,
    store dtype "a2", outputAdvance src dst sm sn, store dtype "a3", outputAdvance src dst sm sn]

def middle (dtype : TileDType) (src dst : RegionName) (sm sn : Int) : List Stmt :=
  [inputAdvance src dst sm sn, load dtype "a2", .forRange "k" 0 2 1 (inner dtype src dst sm sn)]

def outer (dtype : TileDType) (src dst : RegionName) (sm sn : Int) : List Stmt :=
  [load dtype "a1", .forRange "j" 0 2 1 (middle dtype src dst sm sn), inputAdvance src dst sm sn]

def program (dtype : TileDType) (src dst : RegionName) (sm sn : Int) : Kernel :=
  .mk [src] [dst] (FullReal.prefixBody src dst sm sn ++
    [.forRange "i" 0 2 1 (outer dtype src dst sm sn)])

/-- Exact complete erasure for every homogeneous source storage profile. -/
theorem source_projection (storage : Storage) (src dst : RegionName) (sm sn : Int) :
    (kernel storage Bool.false src dst sm sn).eraseDType.toAlgKernel =
      program (VeriTile.Triton.eraseDType (SourceChecks.dtype storage)) src dst sm sn := by
  cases storage
  all_goals simp [program, outer, middle, inner, load, store, inputAdvance, outputAdvance,
    FullReal.prefixBody, FullReal.inner, FullReal.middle, FullReal.outer, FullReal.source,
    SourceChecks.dtype, kernel, ComputeKernel.surfaceBody,
    ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    NumericDType.eraseDType, ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType, Region.cast, Except.bind, bind]

end VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.PayloadAlgorithm

#guard_msgs (drop info) in
#auditModuleAxioms
