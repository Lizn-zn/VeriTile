import bench.tritonbench_g.diag_ssm_triton.Proofs.ComplexForwardStep
namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton.ComplexForwardInit
open VeriTile.Triton ComplexForwardStep
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def columns (u : BlockState) (B : Nat) : TileIndex [B] → Nat :=
  fun j => u.pids 0*B+j.1.val

def active (u : BlockState) (batch dim B : Nat) : TileIndex [B] → Bool :=
  fun j => decide (columns u B j < batch*dim)

def initial (u : BlockState) (s : RegionName) (batch dim B : Nat) : TileIndex [B] → ℂ :=
  input u s (columns u B) (active u batch dim B)

def transition (u : BlockState) (l : RegionName) (batch dim B : Nat) : TileIndex [B] → ℂ :=
  input u l (fun j => columns u B j % dim) (active u batch dim B)

def result (u : BlockState) (s l : RegionName) (batch dim B : Nat) : BlockState :=
  let indexed := u.setReg "col_idx" .nat [] (Tile.scalar (u.pids 0*B))
  let columnsSet := indexed.setReg "col_offsets" .nat [B] ⟨columns u B⟩
  let masked := columnsSet.setReg "mask" .bool [B] ⟨active u batch dim B⟩
  let sr := masked.setReg "s_real" .real [B] (realTile (initial u s batch dim B))
  let si := sr.setReg "s_imag" .real [B] (imagTile (initial u s batch dim B))
  let lr := si.setReg "lambda_real" .real [B] (realTile (transition u l batch dim B))
  lr.setReg "lambda_imag" .real [B] (imagTile (transition u l batch dim B))

theorem prefix_step (u : BlockState) (s l : RegionName) (batch dim B : Nat) :
    stepStmts (ComplexForwardSource.prelude s l batch dim B) u = some (result u s l batch dim B) := by
  simp only [ComplexForwardSource.prelude,Erasure.complexForward,
    ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,ComputeKernel.surfaceBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,
    Except.bind,bind,Bool.false_eq_true,↓reduceIte,List.take]
  simp only [stepStmts,stepStmt,evalOp,evalOp.eq_def,Option.bind,bind,pure,
    Option.map,Option.pure_def,Broadcast.leftIndex,Broadcast.rightIndex,Tile.vec,
    String.reduceEq,Tile.bop,Tile.cop,Tile.scalar,NumericDType.mul,NumericDType.add,
    IntegralDType.mod,ComparableDType.lt,Region.cast,
    BlockState.setReg_same,BlockState.setReg_ne_name,BlockState.setReg_pids,
    BlockState.setReg_readMemValue,BlockState.setReg_readMem,BlockState.readMemValue_real,
    Option.bind_some,Option.map_some,Bool.true_eq_false,↓reduceIte,ne_eq,not_false_eq_true]
  simp only [result,columns,active,initial,transition,input,realTile,imagTile,Tile.scalar]
  simp only [apply_ite]
  rfl

theorem result_registers (u : BlockState) (s l : RegionName) (batch dim B : Nat) :
    Registers (result u s l batch dim B) (columns u B) (active u batch dim B)
      (initial u s batch dim B) (transition u l batch dim B) := by
  constructor <;> simp [result]

@[simp] theorem result_memory (u : BlockState) (s l : RegionName) (batch dim B : Nat)
    (r : RegionName) (o : Nat) :
    (result u s l batch dim B).mem r o = u.mem r o := rfl

@[simp] theorem result_pids (u : BlockState) (s l : RegionName) (batch dim B : Nat) :
    (result u s l batch dim B).pids = u.pids := rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DiagSsmTriton.ComplexForwardInit
