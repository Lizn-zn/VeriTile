import bench.tritonbench_g.block_sparse_attn.Proofs.RealQuery

/-! Actual address setup and zero carries before the query loads. This keeps
program ids, grouped-query head mapping, all independent batch/head/row strides
and Q/K/V tile orientations. It performs no memory accesses. -/
namespace VeriTile.Bench.TritonBenchG.BlockSparseAttn.RealSetup
open VeriTile.Triton RealNormalizationValues
noncomputable section
set_option linter.unusedSimpArgs false

def code (BM BN BD : Nat) (p : RealProgram.Parameters) : List Stmt :=
[Stmt.ifThen (Op.constBool Bool.false) [], Stmt.assign TileDType.nat [] "q_seq_len" (Op.constNat p.context),
      Stmt.assign TileDType.nat [] "start_m" (Op.programId 0), Stmt.assign TileDType.nat [] "off_bh" (Op.programId 1),
      Stmt.assign TileDType.nat [] "off_h"
        (Op.mod IntegralDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_bh") (Op.constNat p.heads)),
      Stmt.assign TileDType.nat [] "off_b"
        (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_bh") (Op.constNat p.heads)),
      Stmt.assign TileDType.nat [] "head_groups"
        (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.constNat p.heads) (Op.constNat p.kvHeads)),
      Stmt.assign TileDType.nat [] "off_h_kv"
        (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_h")
          (Op.ref TileDType.nat [] "head_groups")),
      Stmt.assign TileDType.ptr [] "Q"
        (Op.ptrAdd Broadcast.nil (Op.ptrBase p.Q)
          (Op.add NumericDType.nat Broadcast.nil
            (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_b") (Op.constNat p.qBatch))
            (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_h") (Op.constNat p.qHead)))),
      Stmt.assign TileDType.ptr [] "K"
        (Op.ptrAdd Broadcast.nil (Op.ptrBase p.K)
          (Op.add NumericDType.nat Broadcast.nil
            (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_b") (Op.constNat p.kBatch))
            (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_h_kv") (Op.constNat p.kHead)))),
      Stmt.assign TileDType.ptr [] "V"
        (Op.ptrAdd Broadcast.nil (Op.ptrBase p.V)
          (Op.add NumericDType.nat Broadcast.nil
            (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_b") (Op.constNat p.vBatch))
            (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_h_kv") (Op.constNat p.vHead)))),
      Stmt.assign TileDType.nat [BM] "offs_m"
        (Op.add NumericDType.nat Broadcast.scalarL
          (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "start_m") (Op.constNat BM)) (Op.arange BM)),
      Stmt.assign TileDType.nat [BN] "offs_n" (Op.arange BN), Stmt.assign TileDType.nat [BD] "offs_d" (Op.arange BD),
      Stmt.assign TileDType.nat [BM, BD] "off_q"
        (Op.add NumericDType.nat Broadcast.nil.consL.consR
          (Op.mul NumericDType.nat Broadcast.scalarR (Op.expandDim 1 (Op.ref TileDType.nat [BM] "offs_m"))
            (Op.constNat p.qRow))
          (Op.expandDim 0 (Op.ref TileDType.nat [BD] "offs_d"))),
      Stmt.assign TileDType.nat [BD, BN] "off_k"
        (Op.add NumericDType.nat Broadcast.nil.consR.consL
          (Op.mul NumericDType.nat Broadcast.scalarR (Op.expandDim 0 (Op.ref TileDType.nat [BN] "offs_n"))
            (Op.constNat p.kRow))
          (Op.expandDim 1 (Op.ref TileDType.nat [BD] "offs_d"))),
      Stmt.assign TileDType.nat [BN, BD] "off_v"
        (Op.add NumericDType.nat Broadcast.nil.consL.consR
          (Op.mul NumericDType.nat Broadcast.scalarR (Op.expandDim 1 (Op.ref TileDType.nat [BN] "offs_n"))
            (Op.constNat p.vRow))
          (Op.expandDim 0 (Op.ref TileDType.nat [BD] "offs_d"))),
      Stmt.assign TileDType.ptr [BM, BD] "q_ptrs"
        (Op.ptrAdd Broadcast.scalarL (Op.ref TileDType.ptr [] "Q") (Op.ref TileDType.nat [BM, BD] "off_q")),
      Stmt.assign TileDType.ptr [BD, BN] "k_ptrs"
        (Op.ptrAdd Broadcast.scalarL (Op.ref TileDType.ptr [] "K") (Op.ref TileDType.nat [BD, BN] "off_k")),
      Stmt.assign TileDType.ptr [BN, BD] "v_ptrs"
        (Op.ptrAdd Broadcast.scalarL (Op.ref TileDType.ptr [] "V") (Op.ref TileDType.nat [BN, BD] "off_v")),
      Stmt.assign TileDType.real [BM] "m_i"
        (Op.add NumericDType.real Broadcast.scalarR (Op.full [BM] (Op.const 0)) Op.negInf),
      Stmt.assign TileDType.real [BM] "l_i" (Op.full [BM] (Op.const 0)),
      Stmt.assign TileDType.real [BM, BD] "acc" (Op.full [BM, BD] (Op.const 0)),
      Stmt.ifThen (Op.constBool (decide (2 ≤ p.blocks)))
        [Stmt.assign TileDType.real [BM, BD] "acc2" (Op.full [BM, BD] (Op.const 0))]]

def source (swapped : Bool) (BM BN BD : Nat) (p : RealProgram.Parameters) : List ComputeStmt :=
  ((Source.conditional? (RealProgram.parts swapped BM BN BD p).before RealQuery.writesQuery).getD
    ⟨[], .constBool Bool.false, [], [], []⟩).before

set_option maxRecDepth 22000 in
set_option maxHeartbeats 4000000 in
theorem matches_source (swapped : Bool) (BM BN BD : Nat) (p : RealProgram.Parameters) :
    RealProgram.realPart (source swapped BM BN BD p) = code BM BN BD p := by
  cases swapped
  all_goals
    simp [source, code, RealQuery.writesQuery, RealProgram.realPart, RealProgram.parts,
      RealProgram.source, TypedSource.specialize, Source.intRange?, Source.prefixBefore,
      Source.conditional?, ComputeKernel.surfaceBody, ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?,
      ComputeStmt.listToAlgorithm?_map_alg, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Stmt.eraseDTypeList, Stmt.eraseDType,
      Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
      IntegralDType.eraseDType, Except.bind, bind, eraseDType_float]
    exact ⟨⟨rfl, rfl⟩, ⟨rfl, rfl⟩, rfl, rfl⟩

def head (s : BlockState) (p : RealProgram.Parameters) : Nat := s.pids 1 % p.heads
def batch (s : BlockState) (p : RealProgram.Parameters) : Nat := s.pids 1 / p.heads
def keyHead (s : BlockState) (p : RealProgram.Parameters) : Nat := head s p / (p.heads / p.kvHeads)
def qBase (s : BlockState) (p : RealProgram.Parameters) : Nat := batch s p * p.qBatch + head s p * p.qHead
def kBase (s : BlockState) (p : RealProgram.Parameters) : Nat := batch s p * p.kBatch + keyHead s p * p.kHead
def vBase (s : BlockState) (p : RealProgram.Parameters) : Nat := batch s p * p.vBatch + keyHead s p * p.vHead

def rows (s : BlockState) (BM : Nat) : Fin BM → Nat := fun r => s.pids 0 * BM + r.val
def qOffsets (s : BlockState) (BM BD : Nat) (p : RealProgram.Parameters) : Tile .nat [BM, BD] :=
  Tile.mat fun r d => rows s BM r * p.qRow + d.val
def kOffsets (BN BD : Nat) (p : RealProgram.Parameters) : Tile .nat [BD, BN] :=
  Tile.mat fun d j => j.val * p.kRow + d.val
def vOffsets (BN BD : Nat) (p : RealProgram.Parameters) : Tile .nat [BN, BD] :=
  Tile.mat fun j d => j.val * p.vRow + d.val

def queryPointers (s : BlockState) (BM BD : Nat) (p : RealProgram.Parameters) : Tile .ptr [BM, BD] :=
  Tile.mat fun r d => (p.Q, qBase s p + rows s BM r * p.qRow + d.val)
def keyPointers (s : BlockState) (BN BD : Nat) (p : RealProgram.Parameters) : Tile .ptr [BD, BN] :=
  Tile.mat fun d j => (p.K, kBase s p + (j.val * p.kRow + d.val))
def valuePointers (s : BlockState) (BN BD : Nat) (p : RealProgram.Parameters) : Tile .ptr [BN, BD] :=
  Tile.mat fun j d => (p.V, vBase s p + (j.val * p.vRow + d.val))

def result (s : BlockState) (BM BN BD : Nat) (p : RealProgram.Parameters) : BlockState :=
  let t := s.setReg "q_seq_len" .nat [] (Tile.scalar p.context)
  let t := t.setReg "start_m" .nat [] (Tile.scalar (s.pids 0))
  let t := t.setReg "off_bh" .nat [] (Tile.scalar (s.pids 1))
  let t := t.setReg "off_h" .nat [] (Tile.scalar (head s p))
  let t := t.setReg "off_b" .nat [] (Tile.scalar (batch s p))
  let t := t.setReg "head_groups" .nat [] (Tile.scalar (p.heads / p.kvHeads))
  let t := t.setReg "off_h_kv" .nat [] (Tile.scalar (keyHead s p))
  let t := t.setReg "Q" .ptr [] (Tile.scalar (p.Q, qBase s p))
  let t := t.setReg "K" .ptr [] (Tile.scalar (p.K, kBase s p))
  let t := t.setReg "V" .ptr [] (Tile.scalar (p.V, vBase s p))
  let t := t.setReg "offs_m" .nat [BM] (Tile.vec (rows s BM))
  let t := t.setReg "offs_n" .nat [BN] (Tile.vec Fin.val)
  let t := t.setReg "offs_d" .nat [BD] (Tile.vec Fin.val)
  let t := t.setReg "off_q" .nat [BM, BD] (qOffsets s BM BD p)
  let t := t.setReg "off_k" .nat [BD, BN] (kOffsets BN BD p)
  let t := t.setReg "off_v" .nat [BN, BD] (vOffsets BN BD p)
  let t := t.setReg "q_ptrs" .ptr [BM, BD] (queryPointers s BM BD p)
  let t := t.setReg "k_ptrs" .ptr [BD, BN] (keyPointers s BN BD p)
  let t := t.setReg "v_ptrs" .ptr [BN, BD] (valuePointers s BN BD p)
  let t := t.setReg "m_i" .real [BM] (extendedRow (fun _ => ⊥))
  let t := t.setReg "l_i" .real [BM] (row (fun _ => 0))
  let t := t.setReg "acc" .real [BM, BD] (matrix (fun _ _ => 0))
  if 2 ≤ p.blocks then t.setReg "acc2" .real [BM, BD] (matrix (fun _ _ => 0)) else t

set_option maxRecDepth 22000 in
set_option maxHeartbeats 4000000 in
theorem executes (s : BlockState) (BM BN BD : Nat) (p : RealProgram.Parameters) :
    stepStmts (code BM BN BD p) s = some (result s BM BN BD p) := by
  by_cases hb : 2 ≤ p.blocks
  all_goals
    simp [code, result, stepStmts, stepStmt, evalOp.eq_def, bind, Option.bind, hb,
      head, batch, keyHead, qBase, kBase, vBase, rows, qOffsets, kOffsets, vOffsets,
      queryPointers, keyPointers, valuePointers]
    simp only [Tile.ptrAdd, Tile.bop, Tile.scalar, Tile.vec, Tile.mat,
      Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul,
      IntegralDType.mod, IntegralDType.floorDiv, Nat.zero_add, Nat.add_assoc]
    rfl

structure Outputs (s : BlockState) (BM BN BD : Nat) (p : RealProgram.Parameters) (t : BlockState) : Prop where
  context : t.regs .nat [] "q_seq_len" = some (Tile.scalar p.context)
  queryBlock : t.regs .nat [] "start_m" = some (Tile.scalar (s.pids 0))
  queryHead : t.regs .nat [] "off_h" = some (Tile.scalar (head s p))
  queryPointers : t.regs .ptr [BM, BD] "q_ptrs" = some (queryPointers s BM BD p)
  keyPointers : t.regs .ptr [BD, BN] "k_ptrs" = some (keyPointers s BN BD p)
  valuePointers : t.regs .ptr [BN, BD] "v_ptrs" = some (valuePointers s BN BD p)
  queryRows : t.regs .nat [BM] "offs_m" = some (Tile.vec (rows s BM))
  keyLanes : t.regs .nat [BN] "offs_n" = some (Tile.vec Fin.val)
  maximum : t.regs .real [BM] "m_i" = some (extendedRow (fun _ => ⊥))
  normalizer : t.regs .real [BM] "l_i" = some (row (fun _ => 0))
  accumulator : t.regs .real [BM, BD] "acc" = some (matrix (fun _ _ => 0))
  secondAccumulator : 2 ≤ p.blocks → t.regs .real [BM, BD] "acc2" = some (matrix (fun _ _ => 0))

theorem outputs (s : BlockState) (BM BN BD : Nat) (p : RealProgram.Parameters) :
    Outputs s BM BN BD p (result s BM BN BD p) := by
  constructor <;> by_cases hb : 2 ≤ p.blocks <;> simp [result, hb]

theorem state_frame (s : BlockState) (BM BN BD : Nat) (p : RealProgram.Parameters) :
    {result s BM BN BD p with regs := s.regs} = s := by
  by_cases hb : 2 ≤ p.blocks <;> simp [result, hb, BlockState.setReg]

def writes : List RegName := ["q_seq_len", "start_m", "off_bh", "off_h", "off_b", "head_groups", "off_h_kv",
  "Q", "K", "V", "offs_m", "offs_n", "offs_d", "off_q", "off_k", "off_v", "q_ptrs", "k_ptrs", "v_ptrs",
  "m_i", "l_i", "acc", "acc2"]

theorem register_frame (s : BlockState) (BM BN BD : Nat) (p : RealProgram.Parameters)
    (dtype : TileDType) (shape : TileShape) (name : RegName) (hn : name ∉ writes) :
    (result s BM BN BD p).regs dtype shape name = s.regs dtype shape name := by
  simp only [writes, List.mem_cons, List.not_mem_nil, or_false, not_or] at hn
  by_cases hb : 2 ≤ p.blocks <;> simp [result, hb, hn]

theorem code_safe (bounds : RegionBounds) (s : BlockState) (BM BN BD : Nat) (p : RealProgram.Parameters) :
    Stmt.TraceSafeList bounds (code BM BN BD p) s := by
  apply TraceProofs.all_safe
  intro st hst t
  simp only [code, List.mem_cons, List.not_mem_nil, or_false] at hst
  rcases hst with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl |
    rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
  all_goals simp [Stmt.TraceSafe, Op.SafeAt.eq_def, evalOp.eq_def, Stmt.TraceSafeList, stepStmt]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BlockSparseAttn.RealSetup
