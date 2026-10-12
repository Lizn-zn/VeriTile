import bench.tritonbench_g.chunk_gla_simple.Kernels
import VeriTile.Triton
import VeriTile.Triton.Semantics.BlockPadding
import VeriTile.Meta.StatementAudit
import VeriTile.Triton.Float.Erasure

namespace VeriTile.Bench.TritonBenchG.ChunkGlaSimple.Real
open VeriTile.Triton
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false
/-- The compiled (algorithm-lowered) loop body of the key-block loop: the three
block-pointer constructions, the three loads, and the two matmul accumulations. -/
def chunkGlaLoopBody (q k v h : RegionName) (s_k_h s_k_t s_h_h s_h_t T K V BT BK BV : Nat) :
    List Stmt :=
  [ Stmt.assign .blockPtr [BT, BK] "p_q"
      (Op.makeBlockPtrDynOffsets q
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_k_h)) [T, K]
        [BT, BK] [s_k_t, 1]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat BT),
          Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat BK)]),
    Stmt.assign .blockPtr [BK, BT] "p_k"
      (Op.makeBlockPtrDynOffsets k
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_k_h)) [K, T]
        [BK, BT] [1, s_k_t]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat BK),
          Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat BT)]),
    Stmt.assign .blockPtr [BK, BV] "p_h"
      (Op.makeBlockPtrDynOffsets h
        (Op.add .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_h_h))
          (Op.mul .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat K)) (Op.constNat V)))
        [K, V] [BK, BV] [s_h_t, 1]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat BK),
          Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_v") (Op.constNat BV)]),
    Stmt.assign .real [BT, BK] "b_q"
      (Op.load .real (.blockPtr (Op.ref .blockPtr [BT, BK] "p_q") [0, 1]) .none .unspecified),
    Stmt.assign .real [BK, BT] "b_k"
      (Op.load .real (.blockPtr (Op.ref .blockPtr [BK, BT] "p_k") [0, 1]) .none .unspecified),
    Stmt.assign .real [BK, BV] "b_h"
      (Op.load .real (.blockPtr (Op.ref .blockPtr [BK, BV] "p_h") [0, 1]) .none .unspecified),
    Stmt.assign .real [BT, BV] "b_o"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BT, BV] "b_o")
        (Op.dot (batch := []) (Op.ref .real [BT, BK] "b_q") (Op.ref .real [BK, BV] "b_h") {allowTF32 := some Bool.false})),
    Stmt.assign .real [BT, BT] "b_s"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BT, BT] "b_s")
        (Op.dot (batch := []) (Op.ref .real [BT, BK] "b_q") (Op.ref .real [BK, BT] "b_k") {allowTF32 := some Bool.false})) ]

/-- The compiled (algorithm-lowered) front of the kernel: everything except the
final block-pointer store. -/
def chunkGlaFront (q k v h g o : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t T K V BT BK BV : Nat) (scale : ℝ) : List Stmt :=
  [ Stmt.assign .nat [] "i_v" (Op.programId 0),
    Stmt.assign .nat [] "i_t" (Op.programId 1),
    Stmt.assign .nat [] "i_bh" (Op.programId 2),
    Stmt.assign .nat [BT] "o_i" (Op.arange BT),
    Stmt.assign .bool [BT, BT] "m_s"
      (Op.ge ComparableDType.nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BT] "o_i"))
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BT] "o_i"))),
    Stmt.assign .real [BT, BV] "b_o" (Op.full [BT, BV] (Op.const 0)),
    Stmt.assign .real [BT, BT] "b_s" (Op.full [BT, BT] (Op.const 0)),
    Stmt.forRangeDyn "i_k" (.constNat 0)
      (.div .nat .nil (.sub .nat .nil (.add .nat .nil (.constNat K) (.constNat BK)) (.constNat 1)) (.constNat BK))
      (.constNat 1)
      (chunkGlaLoopBody q k v h s_k_h s_k_t s_h_h s_h_t T K V BT BK BV),
    Stmt.assign .blockPtr [BT] "p_g"
      (Op.makeBlockPtrDynOffsets g
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat T)) [T] [BT] [1]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat BT)]),
    Stmt.assign .real [BT] "b_g"
      (Op.load .real (.blockPtr (Op.ref .blockPtr [BT] "p_g") [0]) .none .unspecified),
    Stmt.assign .real [BT, BV] "b_o"
      (Op.mul .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [BT, BV] "b_o")
        (Op.expandDim ⟨1, by simp⟩ (Op.exp (Op.ref .real [BT] "b_g")))),
    Stmt.assign .real [BT, BT] "b_s"
      (Op.mul .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BT, BT] "b_s")
        (Op.exp (Op.sub .real (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BT] "b_g"))
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .real [BT] "b_g"))))),
    Stmt.assign .real [BT, BT] "b_s"
      (Op.where (Op.ref .bool [BT, BT] "m_s") (Op.ref .real [BT, BT] "b_s")
        (Op.broadcast (Op.const 0) [BT, BT])),
    Stmt.assign .blockPtr [BT, BV] "p_v"
      (Op.makeBlockPtrDynOffsets v
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_v_h)) [T, V]
        [BT, BV] [s_v_t, 1]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat BT),
          Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_v") (Op.constNat BV)]),
    Stmt.assign .real [BT, BV] "b_v"
      (Op.load .real (.blockPtr (Op.ref .blockPtr [BT, BV] "p_v") [0, 1]) .none .unspecified),
    Stmt.assign .real [BT, BV] "b_o"
      (Op.mul .real Broadcast.scalarR
        (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
          (Op.ref .real [BT, BV] "b_o")
          (Op.dot (batch := []) (Op.ref .real [BT, BT] "b_s") (Op.ref .real [BT, BV] "b_v") {allowTF32 := some Bool.false}))
        (Op.const scale)),
    Stmt.assign .blockPtr [BT, BV] "p_o"
      (Op.makeBlockPtrDynOffsets o
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_v_h)) [T, V]
        [BT, BV] [s_v_t, 1]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat BT),
          Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_v") (Op.constNat BV)]) ]

/-- The final block-pointer store statement. -/
def chunkGlaStoreStmt (o : RegionName) (BT BV : Nat) : Stmt :=
  Stmt.store .real [BT, BV] (.blockPtr (Op.ref .blockPtr [BT, BV] "p_o") [0, 1])
    (Op.ref .real [BT, BV] "b_o") .none


#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ChunkGlaSimple.Real

namespace VeriTile.Bench.TritonBenchG.ChunkGlaSimple.RealSource
open VeriTile.Triton
set_option maxRecDepth 20000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false

variable (swapped : Bool) (query value gate : Storage)
    (out_dtype : FloatDType) (q k v h g o : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
    (scale : ℝ) (T K V BT BK BV : Nat)

theorem projects :
    (source query value gate swapped out_dtype q k v h g o
      s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV).toAlgorithm? =
    .ok (source query value gate swapped out_dtype q k v h g o
      s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV).toAlgKernel := by
  cases swapped <;> cases query <;> cases value <;> cases gate <;> rfl

theorem storage_erasure :
    (source query value gate swapped out_dtype q k v h g o
      s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV).eraseDType.toAlgKernel.body =
    (source .fp32 .fp32 .fp32 swapped .real q k v h g o
      s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV).eraseDType.toAlgKernel.body := by
  cases swapped <;> cases query <;> cases value <;> cases gate
  all_goals simp only [source, Bool.false_eq_true, ↓reduceIte,
    ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
    ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, ComputeDType.eraseDType, Kernel.eraseDType, Stmt.eraseDTypeList,
    Stmt.eraseDType, Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
    NumericDType.eraseDType, ComparableDType.eraseDType, IntegralDType.eraseDType,
    Except.bind, bind, List.map_cons, List.map_nil, eraseDType_float]
  all_goals simp only [VeriTile.Triton.eraseDType]
  all_goals simp [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def, eraseDType_float]
  all_goals cases out_dtype <;>
    simp [VeriTile.Triton.eraseDType, FloatDType.toTileDType, NumericDType.eraseDType]

theorem body_split :
    (source .fp32 .fp32 .fp32 Bool.false .real q k v h g o
      s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV).eraseDType.toAlgKernel.body =
    Real.chunkGlaFront q k v h g o s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t T K V BT BK BV scale ++
      [Real.chunkGlaStoreStmt o BT BV] := by
  simp only [source, Bool.false_eq_true, ↓reduceIte, Real.chunkGlaFront, Real.chunkGlaLoopBody,
    Real.chunkGlaStoreStmt, ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
    ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, ComputeDType.eraseDType, Kernel.eraseDType, Stmt.eraseDTypeList,
    Stmt.eraseDType, Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
    NumericDType.eraseDType, ComparableDType.eraseDType, IntegralDType.eraseDType,
    Except.bind, bind, List.map_cons, List.map_nil, eraseDType_float]
  simp [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def, eraseDType_float,VeriTile.Triton.eraseDType,
    FloatDType.toTileDType,NumericDType.eraseDType]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ChunkGlaSimple.RealSource
