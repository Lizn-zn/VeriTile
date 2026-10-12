import bench.tritonbench_g.bmm_chunk_fwd.Proofs.RealSource

/-! Parameters and named pieces of the complete source. The outer active-tile
branch contains the full dynamic K loop and optional signed metadata filter. -/
namespace VeriTile.Bench.TritonBenchG.BmmChunkFwd.RealProgram
open VeriTile.Triton
set_option maxHeartbeats 6000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false

structure Parameters where
  A : RegionName
  B : RegionName
  Out : RegionName
  Seq : Region .int
  seqlen : Nat
  chunk : Nat
  K : Nat
  groups : Nat
  aBatch : Nat
  aRow : Nat
  aHead : Nat
  aColumn : Nat
  bBatch : Nat
  bRow : Nat
  bHead : Nat
  bColumn : Nat
  oBatch : Nat
  oChunk : Nat
  oHead : Nat
  oRow : Nat
  oColumn : Nat
  sBatch : Nat
  sRow : Nat
  causal : Bool
  hasSeq : Bool

def source (aStorage bStorage : TypedSource.Storage) (swapped : Bool)
    (outputDType : FloatDType) (BM BN BK : Nat) (p : Parameters) : ComputeKernel :=
  TypedSource.specialize aStorage bStorage swapped outputDType p.A p.B p.Out p.Seq
    p.seqlen p.chunk p.K p.groups p.aBatch p.aRow p.aHead p.aColumn
    p.bBatch p.bRow p.bHead p.bColumn p.oBatch p.oChunk p.oHead p.oRow p.oColumn
    p.sBatch p.sRow BM BN BK p.causal p.hasSeq

def kernel (swapped : Bool) (BM BN BK : Nat) (p : Parameters) : ComputeKernel :=
  (source .fp32 .fp32 swapped .real BM BN BK p).eraseDType

theorem source_projects (aStorage bStorage : TypedSource.Storage) (swapped : Bool)
    (outputDType : FloatDType) (BM BN BK : Nat) (p : Parameters) :
    (source aStorage bStorage swapped outputDType BM BN BK p).toAlgorithm? =
      .ok (source aStorage bStorage swapped outputDType BM BN BK p).toAlgKernel := by
  cases aStorage <;> cases bStorage <;> cases swapped <;> rfl

theorem storage_erasure (aStorage bStorage : TypedSource.Storage) (swapped : Bool)
    (outputDType : FloatDType) (BM BN BK : Nat) (p : Parameters) :
    (source aStorage bStorage swapped outputDType BM BN BK p).eraseDType =
      kernel swapped BM BN BK p :=
  RealSource.storage_erasure aStorage bStorage swapped outputDType p.A p.B p.Out p.Seq
    p.seqlen p.chunk p.K p.groups p.aBatch p.aRow p.aHead p.aColumn
    p.bBatch p.bRow p.bHead p.bColumn p.oBatch p.oChunk p.oHead p.oRow p.oColumn
    p.sBatch p.sRow BM BN BK p.causal p.hasSeq

def branch (swapped : Bool) (BM BN BK : Nat) (p : Parameters) : Source.Conditional :=
  (Source.conditional? (source .fp32 .fp32 swapped .real BM BN BK p).surfaceBody
    (fun body => (Source.dynamicRange? body "k").isSome)).getD
    ⟨[], .constBool Bool.false, [], [], []⟩

def parts (swapped : Bool) (BM BN BK : Nat) (p : Parameters) : Source.DynamicRange :=
  (Source.dynamicRange? (branch swapped BM BN BK p).yes "k").getD
    ⟨[], .constNat 0, .constNat 0, .constNat 0, [], []⟩

def realPart (code : List ComputeStmt) : List Stmt :=
  Stmt.eraseDTypeList (ComputeKernel.mk [] [] code).toAlgKernel.body

def loopBound (K BK : Nat) : Op .nat [] :=
  .div .nat .nil (.sub .nat .nil
    (.add .nat .nil (.constNat K) (.constNat BK)) (.constNat 1)) (.constNat BK)

theorem body_split (swapped : Bool) (BM BN BK : Nat) (p : Parameters) :
    (kernel swapped BM BN BK p).toAlgKernel.body =
      realPart (branch swapped BM BN BK p).before ++
        [.ifThen (.ref .bool [] "active_block")
          (realPart (parts swapped BM BN BK p).before ++
           [.forRangeDyn "k" (.constNat 0) (loopBound p.K BK) (.constNat 1)
             (realPart (parts swapped BM BN BK p).body)] ++
           realPart (parts swapped BM BN BK p).after)] := by
  cases swapped
  all_goals
    simp [kernel, source, branch, parts, realPart, loopBound, TypedSource.specialize,
      Source.conditional?, Source.dynamicRange?, ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
      ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?, ComputeKernel.fromKernelBody,
      ComputeStmt.listToAlgorithm?_map_alg, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Kernel.eraseDType,
      Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
      MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
      IntegralDType.eraseDType, Except.bind, bind, eraseDType_float]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.BmmChunkFwd.RealProgram
