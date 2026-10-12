import bench.tritonbench_g.bmm_optimized.Proofs.RealProgram
import bench.tritonbench_g.bmm_optimized.Proofs.RealGrid
import VeriTile.Triton

/-! Execute the complete prefix from an arbitrary initial state. This includes
all parameter-pointer updates, both CTA-ordering branches, the final short
group, optional row/column masks, initial pointers, trip count and zero carry.
No prepared-register or successful-execution premise is assumed. -/
namespace VeriTile.Bench.TritonBenchG.BmmOptimized.RealSetup
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 14000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def tileM (s : BlockState) (p : RealProgram.Parameters) : Nat :=
  RealGrid.rowTile (s.pids 0) (s.pids 1) (s.numPids 0) (s.numPids 1) p.groupM
def tileN (s : BlockState) (p : RealProgram.Parameters) : Nat :=
  RealGrid.columnTile (s.pids 0) (s.pids 1) (s.numPids 0) (s.numPids 1) p.groupM
def rows (s : BlockState) (BM : Nat) (p : RealProgram.Parameters) : Fin BM → Nat :=
  fun i => tileM s p * BM + i.val
def columns (s : BlockState) (BN : Nat) (p : RealProgram.Parameters) : Fin BN → Nat :=
  fun j => tileN s p * BN + j.val

def aBase (s : BlockState) (p : RealProgram.Parameters) := s.pids 2 * p.M * p.K
def bBase (s : BlockState) (p : RealProgram.Parameters) := s.pids 2 * p.K * p.N
def oBase (s : BlockState) (p : RealProgram.Parameters) := s.pids 2 * p.M * p.N

def gridResult (s : BlockState) (p : RealProgram.Parameters) : BlockState :=
  let t := s.setReg "pid_b" .nat [] (Tile.scalar (s.pids 2))
  let t := t.setReg "A" .ptr [] (Tile.scalar (p.A, aBase s p))
  let t := t.setReg "B" .ptr [] (Tile.scalar (p.B, bBase s p))
  let t := t.setReg "O" .ptr [] (Tile.scalar (p.O, oBase s p))
  let t := t.setReg "pidx" .nat [] (Tile.scalar (s.pids 0))
  let t := t.setReg "pidy" .nat [] (Tile.scalar (s.pids 1))
  if p.groupM = 1 then
    (t.setReg "pid_m" .nat [] (Tile.scalar (s.pids 0))).setReg "pid_n" .nat [] (Tile.scalar (s.pids 1))
  else
    let pid := s.pids 0 + s.pids 1 * s.numPids 0
    let count := s.numPids 1 * p.groupM
    let group := pid / count
    let t := t.setReg "gridx" .nat [] (Tile.scalar (s.numPids 0))
    let t := t.setReg "gridy" .nat [] (Tile.scalar (s.numPids 1))
    let t := t.setReg "pid" .nat [] (Tile.scalar pid)
    let t := t.setReg "num_CTA_per_group" .nat [] (Tile.scalar count)
    let t := t.setReg "group_id" .nat [] (Tile.scalar group)
    let t := t.setReg "inner_group_id" .nat [] (Tile.scalar (pid % count))
    let t := t.setReg "GROUP_SIZE" .nat [] (Tile.scalar (RealGrid.groupSize (s.numPids 0) group p.groupM))
    (t.setReg "pid_m" .nat [] (Tile.scalar (tileM s p))).setReg "pid_n" .nat [] (Tile.scalar (tileN s p))

def aPointers (s : BlockState) (BM BK : Nat) (p : RealProgram.Parameters) : Tile .ptr [BM,BK] :=
  Tile.mat fun i k => (p.A, aBase s p + rows s BM p i * p.K + k.val)
def bPointers (s : BlockState) (BN BK : Nat) (p : RealProgram.Parameters) : Tile .ptr [BK,BN] :=
  Tile.mat fun k j => (p.B, bBase s p + k.val * p.N + columns s BN p j)
def oPointers (s : BlockState) (BM BN : Nat) (p : RealProgram.Parameters) : Tile .ptr [BM,BN] :=
  Tile.mat fun i j => (p.O, oBase s p + rows s BM p i * p.N + columns s BN p j)

def result (s : BlockState) (BM BN BK : Nat) (p : RealProgram.Parameters) : BlockState :=
  let t := gridResult s p
  let t := t.setReg "offs_m" .nat [BM] (Tile.vec (rows s BM p))
  let t := t.setReg "offs_n" .nat [BN] (Tile.vec (columns s BN p))
  let t := t.setReg "offs_k" .nat [BK] (Tile.vec Fin.val)
  let t := if !p.divM then t.setReg "mask_m" .bool [BM] (Tile.vec fun i => decide (rows s BM p i < p.M)) else t
  let t := if !p.divN then t.setReg "mask_n" .bool [BN] (Tile.vec fun j => decide (columns s BN p j < p.N)) else t
  let t := t.setReg "a_ptrs" .ptr [BM,BK] (aPointers s BM BK p)
  let t := t.setReg "b_ptrs" .ptr [BK,BN] (bPointers s BN BK p)
  let t := t.setReg "o_ptrs" .ptr [BM,BN] (oPointers s BM BN p)
  let t := t.setReg "num_iters" .nat [] (Tile.scalar (BmmChunkBwd.RealMath.blocks p.K BK))
  t.setReg "o" .real [BM,BN] (Tile.mat fun _ _ => some 0)

def code (BM BN BK : Nat) (p : RealProgram.Parameters) :=
  RealProgram.realPart (RealProgram.parts Bool.false BM BN BK p).before

theorem original_candidate_setup (swapped : Bool) (BM BN BK : Nat) (p : RealProgram.Parameters) :
    RealProgram.realPart (RealProgram.parts swapped BM BN BK p).before = code BM BN BK p := by
  rcases p with ⟨A,B,O,M,N,K,G,dm,dn,dk⟩
  cases dm <;> cases dn <;> cases dk <;> cases swapped <;> rfl

theorem executes (s : BlockState) (BM BN BK : Nat) (p : RealProgram.Parameters) :
    stepStmts (code BM BN BK p) s = some (result s BM BN BK p) := by
  rcases p with ⟨A,B,O,M,N,K,G,dm,dn,dk⟩
  cases dm <;> cases dn <;> cases dk
  all_goals
    simp only [code, RealProgram.realPart, RealProgram.parts, RealProgram.source,
      TypedSource.specialize, Source.dynamicRange?, ComputeKernel.surfaceBody,
      ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?,
      ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
      Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
      MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
      IntegralDType.eraseDType, Except.bind, bind, eraseDType_float,
      Option.map, Option.getD, beq_self_eq_true, ↓reduceIte]
    by_cases hg : G = 1
    all_goals
      by_cases hb : (s.pids 0 + s.pids 1 * s.numPids 0) / (s.numPids 1 * G) * G + G > s.numPids 0
      all_goals
        simp [result, gridResult, hg, hb, aBase, bBase, oBase, tileM, tileN, rows, columns,
          aPointers, bPointers, oPointers, RealGrid.rowTile, RealGrid.columnTile, RealGrid.groupSize,
          BmmChunkBwd.RealMath.blocks, stepStmts, stepStmt, evalOp.eq_def, bind, Option.bind,
          Tile.ptrAdd, Tile.bop, Tile.cop, Tile.scalar, Tile.vec, Tile.mat,
          Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.sub,
          NumericDType.mul, NumericDType.div, IntegralDType.mod, IntegralDType.floorDiv,
          ComparableDType.lt, ComparableDType.gt, ComparableDType.eq, -Tile.scalar_eta]
        try rfl

theorem state_frame (s : BlockState) (BM BN BK : Nat) (p : RealProgram.Parameters) :
    {result s BM BN BK p with regs := s.regs} = s := by
  cases hm : p.divM <;> cases hn : p.divN <;> by_cases hg : p.groupM = 1 <;>
    simp [result, gridResult, hm, hn, hg, BlockState.setReg]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmOptimized.RealSetup
