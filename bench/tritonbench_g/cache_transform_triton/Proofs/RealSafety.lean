import bench.tritonbench_g.cache_transform_triton.Proofs.RealExecution
import VeriTile.Triton.Memory.Flatten

/-! Actual access bounds for both complete kernels. Prefill's metadata load
is unmasked. Decode masks metadata by sequence count and floating payloads by
the same row predicate broadcast over the hidden dimension. -/
namespace VeriTile.Bench.TritonBenchG.CacheTransformTriton.RealSafety
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def prefillRead (metadata : Region .int) (rowStride hiddenStride N B : Nat)
    (s : BlockState) (hN : 0 < N) (i : Nat) : Nat :=
  (RealMath.relativeRow hN (s.pids 0 * B + s.pids 1)
    (fun j => s.readMemValue .int metadata.cast j.val) * (rowStride : Int) +
      (i * hiddenStride : Nat)).toNat

def decodeRead (metadata : Region .int) (rowStride hiddenStride B : Nat)
    (s : BlockState) (i j : Nat) : Nat :=
  (s.readMemValue .int metadata.cast (s.pids 0 * B + i) * (rowStride : Int) +
    (j * hiddenStride : Nat)).toNat

structure PrefillBounds (bounds : RegionBounds) (cos sin : RegionName)
    (metadata : Region .int) (cosOut sinOut : RegionName)
    (rowStride hiddenStride total H N B : Nat) (s : BlockState) (hN : 0 < N) : Prop where
  metadataBound : ∀ i : Fin N, i.val < bounds metadata.cast
  cos : ∀ i : Fin H, s.pids 0 * B + s.pids 1 < total →
    prefillRead metadata rowStride hiddenStride N B s hN i.val < bounds cos
  sin : ∀ i : Fin H, s.pids 0 * B + s.pids 1 < total →
    prefillRead metadata rowStride hiddenStride N B s hN i.val < bounds sin
  cosOut : ∀ i : Fin H, s.pids 0 * B + s.pids 1 < total →
    (s.pids 0 * B + s.pids 1) * rowStride + i.val * hiddenStride < bounds cosOut
  sinOut : ∀ i : Fin H, s.pids 0 * B + s.pids 1 < total →
    (s.pids 0 * B + s.pids 1) * rowStride + i.val * hiddenStride < bounds sinOut

structure DecodeBounds (bounds : RegionBounds) (cos sin : RegionName)
    (metadata : Region .int) (cosOut sinOut : RegionName)
    (rowStride hiddenStride H N B : Nat) (s : BlockState) : Prop where
  metadataBound : ∀ i : Fin B, s.pids 0 * B + i.val < N →
    s.pids 0 * B + i.val < bounds metadata.cast
  cos : ∀ i : TileIndex [B,H], s.pids 0 * B + i.1.val < N →
    decodeRead metadata rowStride hiddenStride B s i.1.val i.2.1.val < bounds cos
  sin : ∀ i : TileIndex [B,H], s.pids 0 * B + i.1.val < N →
    decodeRead metadata rowStride hiddenStride B s i.1.val i.2.1.val < bounds sin
  cosOut : ∀ i : TileIndex [B,H], s.pids 0 * B + i.1.val < N →
    (s.pids 0 * B + i.1.val) * rowStride + i.2.1.val * hiddenStride < bounds cosOut
  sinOut : ∀ i : TileIndex [B,H], s.pids 0 * B + i.1.val < N →
    (s.pids 0 * B + i.1.val) * rowStride + i.2.1.val * hiddenStride < bounds sinOut

section
variable (cos sin : RegionName) (metadata : Region .int) (cosOut sinOut : RegionName)
    (rowStride hiddenStride total H N B : Nat)

theorem prefill_flattenOk (commuted : Bool) :
    (RealSource.prefill cos sin metadata cosOut sinOut rowStride hiddenStride total H N B commuted).FlattenOk := by
  cases commuted <;> simp [Kernel.FlattenOk, RealSource.prefill, TypedSource.prefill,
    ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?, ComputeKernel.surfaceBody,
      ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, Except.bind, bind, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

theorem decode_flattenOk (commuted : Bool) :
    (RealSource.decode cos sin metadata cosOut sinOut rowStride hiddenStride H N B commuted).FlattenOk := by
  cases commuted <;> simp [Kernel.FlattenOk, RealSource.decode, TypedSource.decode,
    ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?, ComputeKernel.surfaceBody,
      ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, Except.bind, bind, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

theorem prefill_traceSafe (bounds : RegionBounds) (commuted : Bool) (s : BlockState) (hN : 0 < N)
    (hb : PrefillBounds bounds cos sin metadata cosOut sinOut rowStride hiddenStride total H N B s hN) :
    (RealSource.prefill cos sin metadata cosOut sinOut rowStride hiddenStride total H N B commuted).TraceSafe bounds s := by
  cases commuted
  all_goals
    unfold Kernel.TraceSafe
    simp [RealSource.prefill, TypedSource.prefill, ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?, ComputeKernel.surfaceBody,
      ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, Except.bind, bind,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Stmt.TraceSafeList, Stmt.TraceSafe,
      Op.SafeAt.eq_def, MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, evalOp.eq_def,
      MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe, MaskOpt.Active,
      Op.PointerAddressesSafeOn, Op.MemorySafe, Option.bind, Option.map, tile_elementwise,
      Bool.and_eq_true, Tile.bop, Tile.cop, Tile.uop, Tile.ptrAdd, Tile.select,
      Tile.reduceMaxInt, TileShape.discreteMaximum, TileShape.axisDim,
      TileShape.insertAxisIndex, NumericDType.add, NumericDType.mul, NumericDType.sub,
      ComparableDType.lt, ComparableDType.le, hN, _root_.mul_comm]
    simpa [prefillRead, RealMath.relativeRow, RealMath.selectedEnd, _root_.mul_comm]
      using And.intro hb.metadataBound (And.intro hb.cos (And.intro hb.sin (And.intro hb.cosOut hb.sinOut)))

theorem decode_traceSafe (bounds : RegionBounds) (commuted : Bool) (s : BlockState)
    (hb : DecodeBounds bounds cos sin metadata cosOut sinOut rowStride hiddenStride H N B s) :
    (RealSource.decode cos sin metadata cosOut sinOut rowStride hiddenStride H N B commuted).TraceSafe bounds s := by
  cases commuted
  all_goals
    unfold Kernel.TraceSafe
    simp [RealSource.decode, TypedSource.decode, ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?, ComputeKernel.surfaceBody,
      ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, Except.bind, bind,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Stmt.TraceSafeList, Stmt.TraceSafe,
      Op.SafeAt.eq_def, MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, evalOp.eq_def,
      MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe, MaskOpt.Active,
      Op.PointerAddressesSafeOn, Op.MemorySafe, Option.bind, Option.map, tile_elementwise,
      Bool.and_eq_true, Tile.bop, Tile.cop, Tile.uop, Tile.ptrAdd, Tile.select,
      Tile.remap, Broadcast.leftIndex, NumericDType.add, NumericDType.mul, NumericDType.sub,
      ComparableDType.lt, ComparableDType.le, _root_.mul_comm]
    refine ⟨?_, ?_, ?_, ?_, ?_⟩
    · simpa [Nat.mul_comm] using hb.metadataBound
    · intro i j ha
      have h := hb.cos (i, j, PUnit.unit) (by simpa [Nat.mul_comm] using ha)
      simp only [decodeRead, Nat.mul_comm] at h
      simp only [if_pos ha]
      convert h using 1
      congr 2
      exact Int.mul_comm _ _
    · intro i j ha
      have h := hb.sin (i, j, PUnit.unit) (by simpa [Nat.mul_comm] using ha)
      simp only [decodeRead, Nat.mul_comm] at h
      simp only [if_pos ha]
      convert h using 1
      congr 2
      exact Int.mul_comm _ _
    · simpa [Nat.mul_comm] using hb.cosOut
    · simpa [Nat.mul_comm] using hb.sinOut
end

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.CacheTransformTriton.RealSafety
