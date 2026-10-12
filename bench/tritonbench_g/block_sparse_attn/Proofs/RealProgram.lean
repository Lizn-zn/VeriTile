import bench.tritonbench_g.block_sparse_attn.Proofs.RealSource

/-! Named slices of the complete signed source. Dimensions, strides and
branches remain symbolic. The checked decomposition is used by the real
execution proof; locating a slice asserts no execution or output property. -/
namespace VeriTile.Bench.TritonBenchG.BlockSparseAttn.RealProgram
open VeriTile.Triton
set_option maxRecDepth 22000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

structure Parameters where
  Out : RegionName
  Q : RegionName
  K : RegionName
  V : RegionName
  rows : Region .int
  cols : Region .int
  rowStride : Nat
  columnStride : Nat
  layouts : Nat
  scale : ℝ
  qBatch : Nat
  qHead : Nat
  qRow : Nat
  kBatch : Nat
  kHead : Nat
  kRow : Nat
  vBatch : Nat
  vHead : Nat
  vRow : Nat
  oBatch : Nat
  oHead : Nat
  oRow : Nat
  heads : Nat
  kvHeads : Nat
  context : Nat
  blocks : Nat
  evenM : Bool
  evenN : Bool

def source (storage : TypedSource.Storage) (swapped : Bool) (BM BN BD : Nat)
    (p : Parameters) : ComputeKernel :=
  TypedSource.specialize storage swapped p.Out p.Q p.K p.V p.rows p.cols
    p.rowStride p.columnStride p.layouts p.scale
    p.qBatch p.qHead p.qRow p.kBatch p.kHead p.kRow p.vBatch p.vHead p.vRow
    p.oBatch p.oHead p.oRow p.heads p.kvHeads p.context BM BN BD p.blocks p.evenM p.evenN

/-- A canonical storage representative of the common real program. -/
def kernel (swapped : Bool) (BM BN BD : Nat) (p : Parameters) : ComputeKernel :=
  (source .fp16 swapped BM BN BD p).eraseDType

theorem source_projects (storage : TypedSource.Storage) (swapped : Bool)
    (BM BN BD : Nat) (p : Parameters) :
    (source storage swapped BM BN BD p).toAlgorithm? =
      .ok (source storage swapped BM BN BD p).toAlgKernel := by
  cases storage <;> cases swapped <;> rfl

theorem storage_erasure (storage : TypedSource.Storage) (swapped : Bool)
    (BM BN BD : Nat) (p : Parameters) :
    (source storage swapped BM BN BD p).eraseDType = kernel swapped BM BN BD p :=
  RealSource.storage_erasure storage swapped p.Out p.Q p.K p.V p.rows p.cols
    p.rowStride p.columnStride p.layouts p.scale
    p.qBatch p.qHead p.qRow p.kBatch p.kHead p.kRow p.vBatch p.vHead p.vRow
    p.oBatch p.oHead p.oRow p.heads p.kvHeads p.context BM BN BD p.blocks p.evenM p.evenN

def parts (swapped : Bool) (BM BN BD : Nat) (p : Parameters) : Source.IntRange :=
  (Source.intRange? (source .fp16 swapped BM BN BD p).surfaceBody "col_idx_idx").getD
    ⟨[], .constInt 0, .constInt 0, .constNat 0, [], []⟩

def realPart (code : List ComputeStmt) : List Stmt :=
  Stmt.eraseDTypeList (ComputeKernel.mk [] [] code).toAlgKernel.body

/-- The entire actual source, including the signed loop, its prefix and all
stores. The successful projection rules out a default empty algorithm body. -/
theorem body_split (swapped : Bool) (BM BN BD : Nat) (p : Parameters) :
    (kernel swapped BM BN BD p).toAlgKernel.body =
      realPart (parts swapped BM BN BD p).before ++
        [.forRangeInt "col_idx_idx" (.ref .int [] "start_l") (.ref .int [] "end_l")
          (.constNat 1) (realPart (parts swapped BM BN BD p).body)] ++
      realPart (parts swapped BM BN BD p).after := by
  cases swapped
  all_goals
    simp [kernel, source, parts, realPart, TypedSource.specialize, Source.intRange?, Source.prefixBefore,
      ComputeKernel.surfaceBody, ComputeKernel.eraseDType, ComputeKernel.toAlgKernel,
      ComputeKernel.toAlgorithm?, ComputeKernel.fromKernelBody,
      ComputeStmt.listToAlgorithm?_map_alg, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
      Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
      IntegralDType.eraseDType, Except.bind, bind, eraseDType_float]

/-- Start at the row maximum, ending after the probability storage cast.
The occurrence count distinguishes the initial, scaled and cast p writes. -/
def normalizationPrefix (swapped : Bool) (BM BN BD : Nat) (p : Parameters) :=
  Source.prefixBefore (parts swapped BM BN BD p).body "m_ij"

def normalizationRest (swapped : Bool) (BM BN BD : Nat) (p : Parameters) :=
  (parts swapped BM BN BD p).body.drop (normalizationPrefix swapped BM BN BD p).length

def normalization (swapped : Bool) (BM BN BD : Nat) (p : Parameters) :=
  let rest := normalizationRest swapped BM BN BD p
  rest.take ((Source.prefixBefore rest "p" 2).length + 1)

def accumulation (swapped : Bool) (BM BN BD : Nat) (p : Parameters) :=
  let rest := normalizationRest swapped BM BN BD p
  rest.drop ((Source.prefixBefore rest "p" 2).length + 1)

theorem loop_split (swapped : Bool) (BM BN BD : Nat) (p : Parameters) :
    realPart (parts swapped BM BN BD p).body =
      realPart (normalizationPrefix swapped BM BN BD p) ++
      realPart (normalization swapped BM BN BD p) ++
      realPart (accumulation swapped BM BN BD p) := by
  cases swapped
  all_goals
    simp [kernel, source, parts, realPart, normalizationPrefix, normalizationRest,
      normalization, accumulation, TypedSource.specialize, Source.intRange?, Source.prefixBefore,
      ComputeKernel.surfaceBody, ComputeKernel.eraseDType, ComputeKernel.toAlgKernel,
      ComputeKernel.toAlgorithm?, ComputeKernel.fromKernelBody,
      ComputeStmt.listToAlgorithm?_map_alg, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
      Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
      IntegralDType.eraseDType, Except.bind, bind, eraseDType_float]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.BlockSparseAttn.RealProgram
