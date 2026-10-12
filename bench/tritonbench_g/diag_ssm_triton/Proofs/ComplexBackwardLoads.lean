import bench.tritonbench_g.diag_ssm_triton.Proofs.ComplexBackwardStep
namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton.ComplexBackwardLoads
open VeriTile.Triton ComplexForwardStep
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def source (gy : RegionName) (length batch dim B : Nat) : List Stmt :=
  (ComplexBackwardSource.body "s" "y" "gx" gy length batch dim B).take 4

def result {B : Nat} (u : BlockState) (gy : RegionName) (batch dim t : Nat)
    (col : TileIndex [B] → Nat) (mask : TileIndex [B] → Bool) : BlockState :=
  let off := offset batch dim t col
  let values := fun j => starRingEnd ℂ (input u gy off mask j)
  let timed := u.setReg "t" .nat [] (Tile.scalar t)
  let indexed := timed.setReg "offsets" .nat [B] ⟨fun j => off j*2⟩
  let realLoaded := indexed.setReg "grad_y_real" .real [B] (realTile values)
  realLoaded.setReg "grad_y_imag" .real [B] (imagTile values)

theorem step (gy : RegionName) (length batch dim B k : Nat) (u : BlockState)
    (col : TileIndex [B] → Nat) (mask : TileIndex [B] → Bool)
    (hi : u.regs .nat [] "i"=some (Tile.scalar k))
    (hc : u.regs .nat [B] "col_offsets"=some ⟨col⟩)
    (hm : u.regs .bool [B] "mask"=some ⟨mask⟩) :
    stepStmts (source gy length batch dim B) u = some (result u gy batch dim (length-1-k) col mask) := by
  simp only [source,ComplexBackwardSource.body,Erasure.complexBackward,
    ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,ComputeKernel.surfaceBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,
    Except.bind,bind,Bool.false_eq_true,↓reduceIte,List.drop,List.take]
  simp only [stepStmts,stepStmt,evalOp,evalOp.eq_def,Option.bind,bind,pure,
    Option.map,Option.pure_def,Broadcast.leftIndex,Broadcast.rightIndex,Tile.vec,
    String.reduceEq,Tile.bop,Tile.cop,Tile.scalar,NumericDType.mul,NumericDType.add,NumericDType.sub,
    WithBot.realSub,Option.map₂_some_some,Region.cast,
    BlockState.setReg_same,BlockState.setReg_ne_name,BlockState.setReg_pids,
    BlockState.setReg_readMemValue,BlockState.setReg_readMem,BlockState.readMemValue_real,
    Option.bind_some,Option.map_some,Bool.true_eq_false,↓reduceIte,ne_eq,not_false_eq_true,
    hi,hc,hm,Nat.mul_assoc]
  simp only [result,realTile,imagTile,input,offset,Complex.conj_re,Complex.conj_im]
  simp only [apply_ite,Option.map₂_some_some]
  norm_num only [zero_sub]
  rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DiagSsmTriton.ComplexBackwardLoads
