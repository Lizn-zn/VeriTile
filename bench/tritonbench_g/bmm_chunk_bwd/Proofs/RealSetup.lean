import bench.tritonbench_g.bmm_chunk_bwd.Proofs.RealProgram
import bench.tritonbench_g.bmm_chunk_bwd.Proofs.RealMath
import VeriTile.Triton

/-! Execute the actual symbolic pre-loop source. It derives the tile, batch,
chunk and group indices, pointer grids, runtime chunk limit and zero carry
from the initial state. No pre-populated register or execution assumption is
used. Setup performs no memory access. -/
namespace VeriTile.Bench.TritonBenchG.BmmChunkBwd.RealSetup
open VeriTile.Triton
noncomputable section
set_option maxRecDepth 18000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

def chunk (s : BlockState) (p : RealProgram.Parameters) := s.pids 2 / p.groups
def head (s : BlockState) (p : RealProgram.Parameters) := s.pids 2 - chunk s p * p.groups
def tilesN (BN : Nat) (p : RealProgram.Parameters) := (p.K + BN - 1) / BN
def tileM (s : BlockState) (BN : Nat) (p : RealProgram.Parameters) := s.pids 0 / tilesN BN p
def tileN (s : BlockState) (BN : Nat) (p : RealProgram.Parameters) := s.pids 0 % tilesN BN p
def rows (s : BlockState) (BM BN : Nat) (p : RealProgram.Parameters) : Fin BM → Nat :=
  fun i => tileM s BN p * BM + i.val
def columns (s : BlockState) (BN : Nat) (p : RealProgram.Parameters) : Fin BN → Nat :=
  fun j => tileN s BN p * BN + j.val
def limit (s : BlockState) (p : RealProgram.Parameters) :=
  RealMath.chunkLimit p.seqlen p.chunk (chunk s p)
def aBase (s : BlockState) (p : RealProgram.Parameters) :=
  s.pids 1 * p.aBatch + chunk s p * p.chunk * p.aRow + head s p * p.aHead
def dBase (s : BlockState) (p : RealProgram.Parameters) :=
  s.pids 1 * p.dBatch + chunk s p * p.dChunk + head s p * p.dHead
def aPointers (s : BlockState) (BN BCS : Nat) (p : RealProgram.Parameters) : Tile .ptr [BCS, BN] :=
  Tile.mat fun k j => (p.A, aBase s p + (k.val * p.aRow + columns s BN p j * p.aColumn))
def dPointers (s : BlockState) (BM BN BCS : Nat) (p : RealProgram.Parameters) : Tile .ptr [BM, BCS] :=
  Tile.mat fun i k => (p.Dout, dBase s p + (rows s BM BN p i * p.dColumn + k.val * p.dRow))

def result (s : BlockState) (BM BN BCS : Nat) (p : RealProgram.Parameters) : BlockState :=
  let t := s.setReg "pid_b" .nat [] (Tile.scalar (s.pids 1))
  let t := t.setReg "pid_ch" .nat [] (Tile.scalar (s.pids 2))
  let t := t.setReg "pid_c" .nat [] (Tile.scalar (chunk s p))
  let t := t.setReg "pid_h" .nat [] (Tile.scalar (head s p))
  let t := t.setReg "num_pid_n" .nat [] (Tile.scalar (tilesN BN p))
  let t := t.setReg "pid_m" .nat [] (Tile.scalar (tileM s BN p))
  let t := t.setReg "pid_n" .nat [] (Tile.scalar (tileN s BN p))
  let t := t.setReg "a_ptr" .ptr [] (Tile.scalar (p.A, aBase s p))
  let t := t.setReg "dout_ptr" .ptr [] (Tile.scalar (p.Dout, dBase s p))
  let t := t.setReg "offs_m" .nat [BM] (Tile.vec (rows s BM BN p))
  let t := t.setReg "offs_n" .nat [BN] (Tile.vec (columns s BN p))
  let t := t.setReg "offs_cs" .nat [BCS] (Tile.vec Fin.val)
  let t := t.setReg "dout_ptrs" .ptr [BM, BCS] (dPointers s BM BN BCS p)
  let t := t.setReg "a_ptrs" .ptr [BCS, BN] (aPointers s BN BCS p)
  let t := t.setReg "chunk_size_limit" .nat [] (Tile.scalar (limit s p))
  t.setReg "acc" .real [BM, BN] (Tile.mat fun _ _ => some 0)

def code (BM BN BCS : Nat) (p : RealProgram.Parameters) : List Stmt :=
  RealProgram.realPart (RealProgram.parts Bool.false BM BN BCS p).before

theorem original_candidate_setup (swapped : Bool) (BM BN BCS : Nat) (p : RealProgram.Parameters) :
    RealProgram.realPart (RealProgram.parts swapped BM BN BCS p).before = code BM BN BCS p := by
  cases swapped <;> rfl

theorem executes (s : BlockState) (BM BN BCS : Nat) (p : RealProgram.Parameters) :
    stepStmts (code BM BN BCS p) s = some (result s BM BN BCS p) := by
  simp [code, RealProgram.realPart, RealProgram.parts, RealProgram.source,
    TypedSource.specialize, Source.dynamicRange?, ComputeKernel.surfaceBody,
    ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    IntegralDType.eraseDType, Except.bind, bind, eraseDType_float]
  simp [result, stepStmts, stepStmt, evalOp.eq_def, bind, Option.bind,
    chunk, head, tilesN, tileM, tileN, rows, columns, aBase, dBase, aPointers, dPointers]
  simp only [Tile.ptrAdd, Tile.bop, Tile.scalar, Tile.vec, Tile.mat,
    Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.sub,
    NumericDType.mul, NumericDType.div, IntegralDType.mod, IntegralDType.floorDiv,
    Nat.zero_add, Nat.add_assoc]
  by_cases h : p.chunk < p.seqlen - s.pids 2 / p.groups * p.chunk
  · simp [limit, RealMath.chunkLimit, chunk, ComparableDType.lt, Tile.select,
      h, Nat.min_eq_left (Nat.le_of_lt h)]
    rfl
  · simp [limit, RealMath.chunkLimit, chunk, ComparableDType.lt, Tile.select,
      h, Nat.min_eq_right (Nat.le_of_not_gt h)]
    rfl

theorem state_frame (s : BlockState) (BM BN BCS : Nat) (p : RealProgram.Parameters) :
    {result s BM BN BCS p with regs := s.regs} = s := by
  cases s
  rfl

theorem chunk_limit (s : BlockState) (BM BN BCS : Nat) (p : RealProgram.Parameters) :
    (result s BM BN BCS p).regs .nat [] "chunk_size_limit" = some (Tile.scalar (limit s p)) := by
  simp [result]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmChunkBwd.RealSetup
