import bench.tritonbench_g.diag_ssm_triton.Proofs.ComplexBackwardStep
namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton.ComplexBackwardUpdate
open VeriTile.Triton ComplexForwardStep
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def source (gx : RegionName) (B : Nat) : List Stmt :=
  (ComplexBackwardSource.body "s" "y" gx "gy" 0 0 0 B).drop 5

def result {B : Nat} (u : BlockState) (gx : RegionName) (off : TileIndex [B] → Nat)
    (mask : TileIndex [B] → Bool) (g prev carry acc transition : TileIndex [B] → ℂ) : BlockState :=
  let grad := fun j => g j+carry j
  let newAcc := fun j => acc j+grad j*prev j
  let newCarry := fun j => grad j*transition j
  let gr := u.setReg "grad_s_real" .real [B] (realTile grad)
  let gi := gr.setReg "grad_s_imag" .real [B] (imagTile grad)
  let xr := gi.setReg "grad_x_real" .real [B] (realTile grad)
  let xi := xr.setReg "grad_x_imag" .real [B] (imagTile grad)
  let ar := xi.setReg "grad_lambda_real" .real [B] (realTile newAcc)
  let ai := ar.setReg "grad_lambda_imag" .real [B] (imagTile newAcc)
  let cr := ai.setReg "grad_s_real" .real [B] (realTile newCarry)
  let ci := cr.setReg "grad_s_imag" .real [B] (imagTile newCarry)
  ComplexMemory.store gx off (fun j => starRingEnd ℂ (grad j)) mask ci

theorem step {B : Nat} (u : BlockState) (gx : RegionName)
    (col off : TileIndex [B] → Nat) (mask : TileIndex [B] → Bool)
    (g prev carry acc transition : TileIndex [B] → ℂ)
    (h : ComplexBackwardStep.Registers u col mask carry acc transition)
    (ho : u.regs .nat [B] "offsets"=some ⟨fun j => off j*2⟩)
    (hgr : u.regs .real [B] "grad_y_real"=some (realTile g))
    (hgi : u.regs .real [B] "grad_y_imag"=some (imagTile g))
    (hpr : u.regs .real [B] "s_real"=some (realTile prev))
    (hpi : u.regs .real [B] "s_imag"=some (imagTile prev)) :
    stepStmts (source gx B) u=some (result u gx off mask g prev carry acc transition) := by
  rcases h with ⟨hc,hm,hcr,hci,har,hai,hlr,hli⟩
  simp only [source,ComplexBackwardSource.body,Erasure.complexBackward,
    ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,ComputeKernel.surfaceBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,
    Except.bind,bind,Bool.false_eq_true,↓reduceIte,List.drop]
  simp only [stepStmts,stepStmt,evalOp,evalOp.eq_def,Option.bind,bind,pure,
    Option.map,Option.pure_def,Broadcast.leftIndex,Broadcast.rightIndex,Tile.vec,
    String.reduceEq,Tile.bop,Tile.cop,Tile.scalar,NumericDType.mul,NumericDType.add,NumericDType.sub,
    WithBot.realMul,WithBot.realAdd,WithBot.realSub,Option.map₂_some_some,
    Region.cast,BlockState.writeMemTyped_real,BlockState.foldl_writeMem_masked_regs,
    BlockState.setReg_same,BlockState.setReg_ne_name,BlockState.setReg_pids,
    BlockState.setReg_readMemValue,BlockState.setReg_readMem,BlockState.readMemValue_real,
    Option.bind_some,Option.map_some,Bool.true_eq_false,↓reduceIte,ne_eq,not_false_eq_true,
    hc,hm,hcr,hci,har,hai,hlr,hli,ho,hgr,hgi,hpr,hpi,realTile,imagTile,
    FloatDType.storeValue,FloatDType.toWithBot,WithBot.unbotD_some]
  simp only [result,ComplexMemory.store,realTile,imagTile,
    Complex.add_re,Complex.add_im,Complex.mul_re,Complex.mul_im,Complex.conj_re,Complex.conj_im]
  norm_num only [zero_sub]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DiagSsmTriton.ComplexBackwardUpdate
