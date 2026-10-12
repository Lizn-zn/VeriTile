import bench.tritonbench_g.chunk_linear_attn.Kernels
import VeriTile.Triton
import VeriTile.Triton.Semantics.BlockPadding
import VeriTile.Meta.StatementAudit
import VeriTile.Triton.Float.Erasure
import bench.tritonbench_g.chunk_gla_simple.Proofs.RealStep

namespace VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealOutput
open VeriTile.Triton
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false
/-- The compiled (algorithm-lowered) loop body of the key-block loop: the three
block-pointer constructions, the three loads, and the two matmul accumulations. -/
def linearOutputLoopBody (q k v h : RegionName) (s_k_h s_k_t s_k_d s_h_h s_h_t T K V BT BK BV : Nat) :
    List Stmt :=
  [ Stmt.assign .blockPtr [BT, BK] "p_q"
      (Op.makeBlockPtrDynOffsets q
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_k_h)) [T, K]
        [BT, BK] [s_k_t, s_k_d]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat BT),
          Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat BK)]),
    Stmt.assign .blockPtr [BK, BT] "p_k"
      (Op.makeBlockPtrDynOffsets k
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_k_h)) [K, T]
        [BK, BT] [s_k_d, s_k_t]
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
def linearOutputFront (q k v h o : RegionName)
    (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t T K V BT BK BV : Nat) (scale : ℝ) : List Stmt :=
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
      (linearOutputLoopBody q k v h s_k_h s_k_t s_k_d s_h_h s_h_t T K V BT BK BV),
    Stmt.assign .real [BT, BT] "b_s"
      (Op.where (Op.ref .bool [BT, BT] "m_s") (Op.ref .real [BT, BT] "b_s")
        (Op.broadcast (Op.const 0) [BT, BT])),
    Stmt.assign .blockPtr [BT, BV] "p_v"
      (Op.makeBlockPtrDynOffsets v
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_v_h)) [T, V]
        [BT, BV] [s_v_t, s_v_d]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat BT),
          Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_v") (Op.constNat BV)]),
    Stmt.assign .blockPtr [BT, BV] "p_o"
      (Op.makeBlockPtrDynOffsets o
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_v_h)) [T, V]
        [BT, BV] [s_v_t, s_v_d]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat BT),
          Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_v") (Op.constNat BV)]),
    Stmt.assign .real [BT, BV] "b_v"
      (Op.load .real (.blockPtr (Op.ref .blockPtr [BT, BV] "p_v") [0, 1]) .none .unspecified),
    Stmt.assign .real [BT, BV] "b_o"
      (Op.mul .real Broadcast.scalarR
        (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
          (Op.ref .real [BT, BV] "b_o")
          (Op.dot (batch := []) (Op.ref .real [BT, BT] "b_s") (Op.ref .real [BT, BV] "b_v") {allowTF32 := some Bool.false}))
        (Op.const scale))
 ]

/-- The final block-pointer store statement. -/
def linearOutputStoreStmt (o : RegionName) (BT BV : Nat) : Stmt :=
  Stmt.store .real [BT, BV] (.blockPtr (Op.ref .blockPtr [BT, BV] "p_o") [0, 1])
    (Op.ref .real [BT, BV] "b_o") .none


end VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealOutput

namespace VeriTile.Bench.TritonBenchG.ChunkLinearAttn.OutputSource
open VeriTile.Triton
set_option maxRecDepth 20000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false

variable (swapped : Bool) (query value : Storage)
    (out_dtype : FloatDType) (q k v h o : RegionName)
    (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t : Nat)
    (scale : ℝ) (T K V BT BK BV : Nat)

theorem projects :
    (forwardOSource query value swapped out_dtype q k v h o
      s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t scale T K V BT BK BV).toAlgorithm? =
    .ok (forwardOSource query value swapped out_dtype q k v h o
      s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t scale T K V BT BK BV).toAlgKernel := by
  cases swapped <;> cases query <;> cases value <;> rfl

theorem storage_erasure :
    (forwardOSource query value swapped out_dtype q k v h o
      s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t scale T K V BT BK BV).eraseDType.toAlgKernel.body =
    (forwardOSource .fp32 .fp32 swapped .real q k v h o
      s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t scale T K V BT BK BV).eraseDType.toAlgKernel.body := by
  cases swapped <;> cases query <;> cases value
  all_goals simp only [forwardOSource, Bool.false_eq_true, ↓reduceIte,
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
    (forwardOSource .fp32 .fp32 Bool.false .real q k v h o
      s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t scale T K V BT BK BV).eraseDType.toAlgKernel.body =
    RealOutput.linearOutputFront q k v h o s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t T K V BT BK BV scale ++
      [RealOutput.linearOutputStoreStmt o BT BV] := by
  simp only [forwardOSource, Bool.false_eq_true, ↓reduceIte, RealOutput.linearOutputFront, RealOutput.linearOutputLoopBody,
    RealOutput.linearOutputStoreStmt, ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
    ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, ComputeDType.eraseDType, Kernel.eraseDType, Stmt.eraseDTypeList,
    Stmt.eraseDType, Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
    NumericDType.eraseDType, ComparableDType.eraseDType, IntegralDType.eraseDType,
    Except.bind, bind, List.map_cons, List.map_nil, eraseDType_float]
  simp [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def, eraseDType_float,VeriTile.Triton.eraseDType,
    FloatDType.toTileDType,NumericDType.eraseDType]

end VeriTile.Bench.TritonBenchG.ChunkLinearAttn.OutputSource

namespace VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealOutput
open VeriTile.Triton
open ChunkGlaSimple.Real (makeBlockPtr_2d_eval load_bp_2d_ref mulConst_eval accDot_op_eval)
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false
/-- Loaded-tile abbreviations: the masked block-pointer loads of `q`, `k`, `h`.
Stated exactly as `load_bp_2d_ref` emits them (offset `kb * BK` on the key axis). -/
noncomputable def bqTile (s : BlockState) (q : RegionName) (s_k_h s_k_t s_k_d T K BT BK kb : Nat) :
    Tile .real [BT, BK] :=
  ⟨fun idx => if (s.pids 1 * BT + idx.1.val < T ∧ kb * BK + idx.2.1.val < K) then
      some (s.readMem q (s.pids 2 * s_k_h + (s.pids 1 * BT + idx.1.val) * s_k_t
        + (kb * BK + idx.2.1.val) * s_k_d)) else some 0⟩

noncomputable def bkTile (s : BlockState) (k : RegionName) (s_k_h s_k_t s_k_d T K BT BK kb : Nat) :
    Tile .real [BK, BT] :=
  ⟨fun idx => if (kb * BK + idx.1.val < K ∧ s.pids 1 * BT + idx.2.1.val < T) then
      some (s.readMem k (s.pids 2 * s_k_h + (kb * BK + idx.1.val) * s_k_d
        + (s.pids 1 * BT + idx.2.1.val) * s_k_t)) else some 0⟩

noncomputable def bhTile (s : BlockState) (h : RegionName) (s_h_h s_h_t K V BT BK BV kb : Nat) :
    Tile .real [BK, BV] :=
  ⟨fun idx => if (kb * BK + idx.1.val < K ∧ s.pids 0 * BV + idx.2.1.val < V) then
      some (s.readMem h (s.pids 2 * s_h_h + s.pids 1 * K * V + (kb * BK + idx.1.val) * s_h_t
        + (s.pids 0 * BV + idx.2.1.val) * 1)) else some 0⟩

/-- **Loop-body execution at key block `kb`.** Given the prologue register
readbacks on the loop-entry state `sl` (program ids, `i_k = kb`, the current
`b_o`/`b_s`), the 8-statement loop body steps to a final state whose `b_o`/`b_s`
registers hold the two matmul accumulations of the masked `q`/`k`/`h` loads. -/
theorem linearOutputLoopBody_steps (q k v h : RegionName)
    (s_k_h s_k_t s_k_d s_h_h s_h_t T K V BT BK BV : Nat) (kb : Nat) (sl : BlockState)
    (boTile : Tile .real [BT, BV]) (bsTile : Tile .real [BT, BT])
    (hiv : sl.regs .nat [] "i_v" = some (Tile.scalar (sl.pids 0)))
    (hit : sl.regs .nat [] "i_t" = some (Tile.scalar (sl.pids 1)))
    (hibh : sl.regs .nat [] "i_bh" = some (Tile.scalar (sl.pids 2)))
    (hik : sl.regs .nat [] "i_k" = some (Tile.scalar kb))
    (hbo : sl.regs .real [BT, BV] "b_o" = some boTile)
    (hbs : sl.regs .real [BT, BT] "b_s" = some bsTile)
    (hu : sl.undef = fun _ _ => 0) :
    ∃ sF, stepStmts (linearOutputLoopBody q k v h s_k_h s_k_t s_k_d s_h_h s_h_t T K V BT BK BV) sl = some sF
      ∧ sF.pids = sl.pids ∧ sF.mem = sl.mem ∧ sF.undef = sl.undef
      ∧ sF.regs .nat [] "i_v" = some (Tile.scalar (sl.pids 0))
      ∧ sF.regs .nat [] "i_t" = some (Tile.scalar (sl.pids 1))
      ∧ sF.regs .nat [] "i_bh" = some (Tile.scalar (sl.pids 2))
      ∧ sF.regs .bool [BT, BT] "m_s" = sl.regs .bool [BT, BT] "m_s"
      ∧ sF.regs .real [BT, BV] "b_o" = some
          (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
            boTile (Tile.dot [] (bqTile sl q s_k_h s_k_t s_k_d T K BT BK kb)
              (bhTile sl h s_h_h s_h_t K V BT BK BV kb)))
      ∧ sF.regs .real [BT, BT] "b_s" = some
          (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
            bsTile (Tile.dot [] (bqTile sl q s_k_h s_k_t s_k_d T K BT BK kb)
              (bkTile sl k s_k_h s_k_t s_k_d T K BT BK kb))) := by
  -- abbreviations for the makeBlockPtr base/offset scalar evals (proved by simp,
  -- peeling the accumulated setReg chain via the @[simp 1100] ref/setReg lemmas)
  have hbaseH : ∀ st : BlockState, st.regs .nat [] "i_bh" = some (Tile.scalar (sl.pids 2)) →
      st.regs .nat [] "i_t" = some (Tile.scalar (sl.pids 1)) →
      evalOp (Op.add .nat Broadcast.nil
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_h_h))
        (Op.mul .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat K)) (Op.constNat V))) st
      = some (Tile.scalar (sl.pids 2 * s_h_h + sl.pids 1 * K * V)) := by
    intro st hbh ht
    rw [evalOp_add, mulConst_eval st "i_bh" _ _ hbh, evalOp_mul,
      mulConst_eval st "i_t" (sl.pids 1) K ht]
    simp only [evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
    rfl
  -- step p_q, p_k, p_h
  unfold linearOutputLoopBody
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (makeBlockPtr_2d_eval q sl _ _ _ [T, K] [BT, BK] [s_k_t, s_k_d] (sl.pids 2 * s_k_h)
      (sl.pids 1 * BT) (kb * BK)
      (mulConst_eval sl "i_bh" _ _ hibh) (mulConst_eval sl "i_t" _ _ hit)
      (mulConst_eval sl "i_k" _ _ hik)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (makeBlockPtr_2d_eval k _ _ _ _ [K, T] [BK, BT] [s_k_d, s_k_t] (sl.pids 2 * s_k_h)
      (kb * BK) (sl.pids 1 * BT)
      (mulConst_eval _ "i_bh" _ _ (by simp [hibh])) (mulConst_eval _ "i_k" _ _ (by simp [hik]))
      (mulConst_eval _ "i_t" _ _ (by simp [hit]))))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (makeBlockPtr_2d_eval h _ _ _ _ [K, V] [BK, BV] [s_h_t, 1]
      (sl.pids 2 * s_h_h + sl.pids 1 * K * V) (kb * BK) (sl.pids 0 * BV)
      (hbaseH _ (by simp [hibh]) (by simp [hit])) (mulConst_eval _ "i_k" _ _ (by simp [hik]))
      (mulConst_eval _ "i_v" _ _ (by simp [hiv]))))]
  -- step loads b_q, b_k, b_h
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (load_bp_2d_ref q _ "p_q" (sl.pids 2 * s_k_h) T K BT BK s_k_t s_k_d (sl.pids 1 * BT) (kb * BK)
      (by simp) (by simpa using hu)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (load_bp_2d_ref k _ "p_k" (sl.pids 2 * s_k_h) K T BK BT s_k_d s_k_t (kb * BK) (sl.pids 1 * BT)
      (by simp) (by simpa using hu)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (load_bp_2d_ref h _ "p_h" (sl.pids 2 * s_h_h + sl.pids 1 * K * V) K V BK BV s_h_t 1
      (kb * BK) (sl.pids 0 * BV) (by simp) (by simpa using hu)))]
  -- step b_o += dot(b_q, b_h) and b_s += dot(b_q, b_k)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (accDot_op_eval _ BT BK BV "b_o" "b_q" "b_h" boTile (bqTile sl q s_k_h s_k_t s_k_d T K BT BK kb)
      (bhTile sl h s_h_h s_h_t K V BT BK BV kb) (by simp [hbo]) (by simp [bqTile]) (by simp [bhTile])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (accDot_op_eval _ BT BK BT "b_s" "b_q" "b_k" bsTile (bqTile sl q s_k_h s_k_t s_k_d T K BT BK kb)
      (bkTile sl k s_k_h s_k_t s_k_d T K BT BK kb) (by simp [hbs]) (by simp [bqTile]) (by simp [bkTile])))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, rfl, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp
  · funext region offset; simp
  · simp [hiv]
  · simp [hit]
  · simp [hibh]
  · simp
  · simp
  · simp


end VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealOutput

/-! Independent zero-extended contraction over initial arrays. The kernel's
unspecified boundary values require an explicit zero-oracle premise in the
execution theorem; they are not silently changed in the source. -/
namespace VeriTile.Bench.TritonBenchG.ChunkLinearAttn.OutputMath
open scoped BigOperators
noncomputable section
structure Layout where
  keyHead : Nat
  keyTime : Nat
  keyDimension : Nat
  valueHead : Nat
  valueTime : Nat
  valueDimension : Nat
  historyHead : Nat
  historyKey : Nat
  T : Nat
  K : Nat
  V : Nat
  BT : Nat
  BK : Nat
  BV : Nat
structure Launch where
  value : Nat
  chunk : Nat
  head : Nat
structure Inputs where
  query : Nat → ℝ
  key : Nat → ℝ
  value : Nat → ℝ
  history : Nat → ℝ

def numBlocks (p : Layout) := (p.K+p.BK-1)/p.BK
def row (p : Layout) (l : Launch) (i : Nat) := l.chunk*p.BT+i
def column (p : Layout) (l : Launch) (j : Nat) := l.value*p.BV+j
def keyAddress (p : Layout) (l : Launch) (t k : Nat) := l.head*p.keyHead+t*p.keyTime+k*p.keyDimension
def historyAddress (p : Layout) (l : Launch) (k j : Nat) :=
  l.head*p.historyHead+l.chunk*p.K*p.V+k*p.historyKey+column p l j

def outputAddress (p : Layout) (l : Launch) (i j : Nat) :=
  l.head*p.valueHead+row p l i*p.valueTime+column p l j*p.valueDimension

def active (p : Layout) (l : Launch) (i j : Nat) : Prop := row p l i < p.T ∧ column p l j < p.V

instance (p : Layout) (l : Launch) (i j : Nat) : Decidable (active p l i j) :=
  inferInstanceAs (Decidable (_ ∧ _))

def query (p : Layout) (l : Launch) (a : Inputs) (i k : Nat) : ℝ :=
  if row p l i < p.T ∧ k < p.K then a.query (keyAddress p l (row p l i) k) else 0

def key (p : Layout) (l : Launch) (a : Inputs) (k j : Nat) : ℝ :=
  if k < p.K ∧ row p l j < p.T then a.key (l.head*p.keyHead+k*p.keyDimension+row p l j*p.keyTime) else 0

def history (p : Layout) (l : Launch) (a : Inputs) (k j : Nat) : ℝ :=
  if k < p.K ∧ column p l j < p.V then a.history (historyAddress p l k j) else 0

def value (p : Layout) (l : Launch) (a : Inputs) (i j : Nat) : ℝ :=
  if active p l i j then a.value (outputAddress p l i j) else 0

/-- Contributions from all completed key blocks, including the checked tail. -/
def qhSum (p : Layout) (l : Launch) (a : Inputs) (n i j : Nat) : ℝ :=
  ∑ kb ∈ Finset.range n, ∑ e : Fin p.BK,
    query p l a i (kb*p.BK+e.val)*history p l a (kb*p.BK+e.val) j

def qkSum (p : Layout) (l : Launch) (a : Inputs) (n i j : Nat) : ℝ :=
  ∑ kb ∈ Finset.range n, ∑ e : Fin p.BK,
    query p l a i (kb*p.BK+e.val)*key p l a (kb*p.BK+e.val) j

def score (p : Layout) (l : Launch) (a : Inputs) (i j : Nat) : ℝ :=
  if j ≤ i then qkSum p l a (numBlocks p) i j else 0

/-- History plus the causal query/key/value contraction, then scale. -/
def expected (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (i j : Nat) : ℝ :=
  (qhSum p l a (numBlocks p) i j+
    ∑ t : Fin p.BT, score p l a i t.val*value p l a t.val j)*scale

end
end VeriTile.Bench.TritonBenchG.ChunkLinearAttn.OutputMath


namespace VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealOutput
open VeriTile.Triton OutputMath
open ChunkGlaSimple.Real (dot2d_elem)
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

structure Regions where
  query : RegionName
  key : RegionName
  value : RegionName
  history : RegionName
  output : RegionName

def launch (s : BlockState) : Launch := ⟨s.pids 0,s.pids 1,s.pids 2⟩
def inputs (r : Regions) (s : BlockState) : Inputs :=
  ⟨s.readMem r.query,s.readMem r.key,s.readMem r.value,s.readMem r.history⟩

def queryTile (p : Layout) (l : Launch) (a : Inputs) (kb : Nat) : Tile .real [p.BT,p.BK] :=
  Tile.mat fun i e => some (query p l a i.val (kb*p.BK+e.val))
def keyTile (p : Layout) (l : Launch) (a : Inputs) (kb : Nat) : Tile .real [p.BK,p.BT] :=
  Tile.mat fun e j => some (key p l a (kb*p.BK+e.val) j.val)
def historyTile (p : Layout) (l : Launch) (a : Inputs) (kb : Nat) : Tile .real [p.BK,p.BV] :=
  Tile.mat fun e j => some (history p l a (kb*p.BK+e.val) j.val)
def carryO (p : Layout) (l : Launch) (a : Inputs) (n : Nat) : Tile .real [p.BT,p.BV] :=
  Tile.mat fun i j => some (qhSum p l a n i.val j.val)
def carryS (p : Layout) (l : Launch) (a : Inputs) (n : Nat) : Tile .real [p.BT,p.BT] :=
  Tile.mat fun i j => some (qkSum p l a n i.val j.val)
def maskTile (p : Layout) : Tile .bool [p.BT,p.BT] :=
  Tile.mat fun i j => decide (j.val ≤ i.val)

theorem query_tile (p : Layout) (r : Regions) (s : BlockState) (kb : Nat) :
    bqTile s r.query p.keyHead p.keyTime p.keyDimension p.T p.K p.BT p.BK kb = queryTile p (launch s) (inputs r s) kb := by
  apply Tile.ext
  rintro ⟨i,e,⟨⟩⟩
  simp only [bqTile,queryTile,Tile.mat,query,keyAddress,row,launch,inputs,mul_one]
  split_ifs <;> rfl

theorem key_tile (p : Layout) (r : Regions) (s : BlockState) (kb : Nat) :
    bkTile s r.key p.keyHead p.keyTime p.keyDimension p.T p.K p.BT p.BK kb = keyTile p (launch s) (inputs r s) kb := by
  apply Tile.ext
  rintro ⟨e,j,⟨⟩⟩
  simp only [bkTile,keyTile,Tile.mat,key,row,launch,inputs,mul_one]
  split_ifs <;> rfl

theorem history_tile (p : Layout) (r : Regions) (s : BlockState) (kb : Nat) :
    bhTile s r.history p.historyHead p.historyKey p.K p.V p.BT p.BK p.BV kb = historyTile p (launch s) (inputs r s) kb := by
  apply Tile.ext
  rintro ⟨e,j,⟨⟩⟩
  simp only [bhTile,historyTile,Tile.mat,history,historyAddress,column,launch,inputs,mul_one]
  split_ifs <;> rfl

theorem output_update (p : Layout) (l : Launch) (a : Inputs) (n : Nat) :
    Tile.bop NumericDType.real.add (.consSame (.consSame .nil))
      (carryO p l a n) (Tile.dot [] (queryTile p l a n) (historyTile p l a n)) = carryO p l a (n+1) := by
  apply Tile.ext
  rintro ⟨i,j,⟨⟩⟩
  rw [Tile.bop_data]
  have hd := dot2d_elem (queryTile p l a n) (historyTile p l a n) i j
    (fun e => query p l a i.val (n*p.BK+e.val))
    (fun e => history p l a (n*p.BK+e.val) j.val) (fun _ => rfl) (fun _ => rfl)
  simp only [Broadcast.leftIndex,Broadcast.rightIndex,hd,carryO,Tile.mat,
    NumericDType.add,WithBot.realAdd,Option.map₂_some_some,qhSum,Finset.sum_range_succ]

theorem score_update (p : Layout) (l : Launch) (a : Inputs) (n : Nat) :
    Tile.bop NumericDType.real.add (.consSame (.consSame .nil))
      (carryS p l a n) (Tile.dot [] (queryTile p l a n) (keyTile p l a n)) = carryS p l a (n+1) := by
  apply Tile.ext
  rintro ⟨i,j,⟨⟩⟩
  rw [Tile.bop_data]
  have hd := dot2d_elem (queryTile p l a n) (keyTile p l a n) i j
    (fun e => query p l a i.val (n*p.BK+e.val))
    (fun e => key p l a (n*p.BK+e.val) j.val) (fun _ => rfl) (fun _ => rfl)
  simp only [Broadcast.leftIndex,Broadcast.rightIndex,hd,carryS,Tile.mat,
    NumericDType.add,WithBot.realAdd,Option.map₂_some_some,qkSum,Finset.sum_range_succ]

end
end VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealOutput


namespace VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealOutput
open VeriTile.Triton OutputMath
open ChunkGlaSimple.Real (dot2d_elem)
noncomputable section
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

def valueTile (p : Layout) (l : Launch) (a : Inputs) : Tile .real [p.BT,p.BV] :=
  Tile.mat fun i j => some (value p l a i.val j.val)
def maskedS (p : Layout) (l : Launch) (a : Inputs) : Tile .real [p.BT,p.BT] :=
  Tile.select (maskTile p) (carryS p l a (numBlocks p)) (⟨fun _ => some 0⟩ : Tile .real [p.BT,p.BT])
def resultTile (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) : Tile .real [p.BT,p.BV] :=
  Tile.bop NumericDType.real.mul .scalarR
    (Tile.bop NumericDType.real.add (.consSame (.consSame .nil))
      (carryO p l a (numBlocks p)) (Tile.dot [] (maskedS p l a) (valueTile p l a))) (Tile.scalar (some scale))

theorem masked_value (p : Layout) (l : Launch) (a : Inputs) (i j : Fin p.BT) :
    (maskedS p l a).data (i,j,⟨⟩) = some (score p l a i.val j.val) := by
  by_cases h : j.val ≤ i.val
  all_goals simp [maskedS,maskTile,carryS,Tile.select,Tile.bop,Tile.uop,Tile.expandDim,
    Tile.mat,Tile.vec,TileShape.dropInsertedIndex,Broadcast.leftIndex,Broadcast.rightIndex,
    NumericDType.mul,NumericDType.sub,WithBot.realExp,score,h,-Fin.val_fin_le]

theorem result_value (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (i : Fin p.BT) (j : Fin p.BV) :
    (resultTile p l a scale).data (i,j,⟨⟩) = some (expected p l a scale i.val j.val) := by
  have hd := dot2d_elem (maskedS p l a) (valueTile p l a) i j
    (fun t => score p l a i.val t.val) (fun t => value p l a t.val j.val)
    (masked_value p l a i) (fun _ => rfl)
  simp only [resultTile,carryO,Tile.bop_data,Tile.uop_data,
    Tile.expandDim_data,TileShape.dropInsertedIndex,Broadcast.leftIndex,Broadcast.rightIndex,
    Tile.mat,Tile.vec,hd,Tile.scalar,NumericDType.mul,NumericDType.add,WithBot.realMul,
    WithBot.realAdd,WithBot.realExp,Option.map₂_some_some,Option.map_some,expected]

end
end VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealOutput

#guard_msgs (drop info) in
#auditModuleAxioms
