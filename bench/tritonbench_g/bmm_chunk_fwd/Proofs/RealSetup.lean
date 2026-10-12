import bench.tritonbench_g.bmm_chunk_fwd.Proofs.RealProgram
import bench.tritonbench_g.bmm_chunk_bwd.Proofs.RealMath
import VeriTile.Triton

/-! Execute the real grid/causal prologue and the complete active-tile setup.
The runtime chunk limit and zero accumulator are derived from initial state.
No prepared-register or execution premise is required. -/
namespace VeriTile.Bench.TritonBenchG.BmmChunkFwd.RealSetup
open VeriTile.Triton
noncomputable section
set_option maxRecDepth 18000
set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

def chunk (s : BlockState) (p : RealProgram.Parameters) := s.pids 2 / p.groups
def head (s : BlockState) (p : RealProgram.Parameters) := s.pids 2 - chunk s p * p.groups
def tilesN (BN : Nat) (p : RealProgram.Parameters) := (p.chunk + BN - 1) / BN
def tileM (s : BlockState) (BN : Nat) (p : RealProgram.Parameters) := s.pids 0 / tilesN BN p
def tileN (s : BlockState) (BN : Nat) (p : RealProgram.Parameters) := s.pids 0 % tilesN BN p
def rows (s : BlockState) (BM BN : Nat) (p : RealProgram.Parameters) : Fin BM → Nat :=
  fun i => tileM s BN p * BM + i.val
def columns (s : BlockState) (BN : Nat) (p : RealProgram.Parameters) : Fin BN → Nat :=
  fun j => tileN s BN p * BN + j.val
def active (s : BlockState) (BM BN : Nat) (p : RealProgram.Parameters) : Bool :=
  !p.causal || decide (tileN s BN p * BN < (tileM s BN p + 1) * BM)
def limit (s : BlockState) (p : RealProgram.Parameters) :=
  min p.chunk (p.seqlen - chunk s p * p.chunk)
def aBase (s : BlockState) (p : RealProgram.Parameters) :=
  s.pids 1 * p.aBatch + chunk s p * p.chunk * p.aRow + head s p * p.aHead
def bBase (s : BlockState) (p : RealProgram.Parameters) :=
  s.pids 1 * p.bBatch + chunk s p * p.chunk * p.bRow + head s p * p.bHead
def seqBase (s : BlockState) (p : RealProgram.Parameters) :=
  s.pids 1 * p.sBatch + chunk s p * p.chunk * p.sRow

def prologueResult (s : BlockState) (BM BN : Nat) (p : RealProgram.Parameters) : BlockState :=
  let t := s.setReg "pid_b" .nat [] (Tile.scalar (s.pids 1))
  let t := t.setReg "pid_ch" .nat [] (Tile.scalar (s.pids 2))
  let t := t.setReg "pid_c" .nat [] (Tile.scalar (chunk s p))
  let t := t.setReg "pid_h" .nat [] (Tile.scalar (head s p))
  let t := t.setReg "num_pid_n" .nat [] (Tile.scalar (tilesN BN p))
  let t := t.setReg "pid_m" .nat [] (Tile.scalar (tileM s BN p))
  let t := t.setReg "pid_n" .nat [] (Tile.scalar (tileN s BN p))
  let t := t.setReg "active_block" .bool [] (Tile.scalar Bool.true)
  if p.causal then t.setReg "active_block" .bool [] (Tile.scalar (active s BM BN p)) else t

def prologue (BM BN BK : Nat) (p : RealProgram.Parameters) : List Stmt :=
  RealProgram.realPart (RealProgram.branch Bool.false BM BN BK p).before

theorem prologue_executes (s : BlockState) (BM BN BK : Nat) (p : RealProgram.Parameters) :
    stepStmts (prologue BM BN BK p) s = some (prologueResult s BM BN p) := by
  simp [prologue, RealProgram.realPart, RealProgram.branch, RealProgram.source,
    TypedSource.specialize, Source.conditional?, Source.dynamicRange?, ComputeKernel.surfaceBody,
    ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    IntegralDType.eraseDType, Except.bind, bind, eraseDType_float]
  cases hca : p.causal <;>
    simp [prologueResult, stepStmts, stepStmt, evalOp.eq_def, bind, Option.bind,
      chunk, head, tilesN, tileM, tileN, active, hca, Tile.bop, Tile.cop, Tile.scalar,
      Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.sub,
      NumericDType.mul, NumericDType.div, IntegralDType.mod, IntegralDType.floorDiv,
      ComparableDType.lt, -Tile.scalar_eta]
  all_goals rfl

def aPointers (s : BlockState) (BM BN BK : Nat) (p : RealProgram.Parameters) : Tile .ptr [BM, BK] :=
  Tile.mat fun i k => (p.A, aBase s p + (rows s BM BN p i * p.aRow + k.val * p.aColumn))
def bPointers (s : BlockState) (BN BK : Nat) (p : RealProgram.Parameters) : Tile .ptr [BK, BN] :=
  Tile.mat fun k j => (p.B, bBase s p + (k.val * p.bColumn + columns s BN p j * p.bRow))

def result (s : BlockState) (BM BN BK : Nat) (p : RealProgram.Parameters) : BlockState :=
  let t := prologueResult s BM BN p
  let t := t.setReg "a_ptr" .ptr [] (Tile.scalar (p.A, aBase s p))
  let t := t.setReg "b_ptr" .ptr [] (Tile.scalar (p.B, bBase s p))
  let t := if p.hasSeq then t.setReg "seq_idx_ptr" .ptr [] (Tile.scalar (Region.cast p.Seq, seqBase s p)) else t
  let t := t.setReg "offs_m" .nat [BM] (Tile.vec (rows s BM BN p))
  let t := t.setReg "offs_n" .nat [BN] (Tile.vec (columns s BN p))
  let t := t.setReg "offs_k" .nat [BK] (Tile.vec Fin.val)
  let t := t.setReg "a_ptrs" .ptr [BM, BK] (aPointers s BM BN BK p)
  let t := t.setReg "b_ptrs" .ptr [BK, BN] (bPointers s BN BK p)
  let t := t.setReg "chunk_size_limit" .nat [] (Tile.scalar (limit s p))
  t.setReg "acc" .real [BM, BN] (Tile.mat fun _ _ => some 0)

def code (BM BN BK : Nat) (p : RealProgram.Parameters) : List Stmt :=
  RealProgram.realPart (RealProgram.parts Bool.false BM BN BK p).before

theorem original_candidate_setup (swapped : Bool) (BM BN BK : Nat) (p : RealProgram.Parameters) :
    RealProgram.realPart (RealProgram.parts swapped BM BN BK p).before = code BM BN BK p := by
  cases swapped <;> rfl

theorem executes (s : BlockState) (BM BN BK : Nat) (p : RealProgram.Parameters) :
    stepStmts (code BM BN BK p) (prologueResult s BM BN p) = some (result s BM BN BK p) := by
  simp [code, RealProgram.realPart, RealProgram.parts, RealProgram.branch, RealProgram.source,
    TypedSource.specialize, Source.conditional?, Source.dynamicRange?, ComputeKernel.surfaceBody,
    ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    IntegralDType.eraseDType, Except.bind, bind, eraseDType_float]
  cases hs : p.hasSeq <;> cases hc : p.causal
  all_goals
    simp [result, prologueResult, hs, hc, stepStmts, stepStmt, evalOp.eq_def, bind, Option.bind,
      chunk, head, tilesN, tileM, tileN, rows, columns, aBase, bBase, seqBase, aPointers, bPointers]
    simp only [Tile.ptrAdd, Tile.bop, Tile.scalar, Tile.vec, Tile.mat,
      Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.sub,
      NumericDType.mul, NumericDType.div, IntegralDType.mod, IntegralDType.floorDiv,
      Nat.zero_add, Nat.add_assoc]
    by_cases h : p.chunk < p.seqlen - s.pids 2 / p.groups * p.chunk
    · simp [limit, chunk, ComparableDType.lt, Tile.select, h, Nat.min_eq_left (Nat.le_of_lt h)]
      rfl
    · simp [limit, chunk, ComparableDType.lt, Tile.select, h, Nat.min_eq_right (Nat.le_of_not_gt h)]
      rfl

theorem state_frame (s : BlockState) (BM BN BK : Nat) (p : RealProgram.Parameters) :
    {result s BM BN BK p with regs := s.regs} = s := by
  cases hs : p.hasSeq <;> cases hc : p.causal <;>
    simp [result, prologueResult, hs, hc, BlockState.setReg]

theorem chunk_limit (s : BlockState) (BM BN BK : Nat) (p : RealProgram.Parameters) :
    (result s BM BN BK p).regs .nat [] "chunk_size_limit" = some (Tile.scalar (limit s p)) := by
  simp [result]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmChunkFwd.RealSetup
