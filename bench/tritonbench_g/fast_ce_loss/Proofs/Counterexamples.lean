import bench.tritonbench_g.fast_ce_loss.Kernels
import VeriTile.Triton.Float.Erasure
import VeriTile.Triton
import VeriTile.Meta.Disprove
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.FastCeLoss.Counterexamples
open VeriTile.Triton TypedSource
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def chunkKernel : ComputeKernel :=
  (chunked .fp32 Bool.false Bool.false Bool.false "logits" "loss" "lse" ⟨"label"⟩ 2 2 2 1 1 1).eraseDType

def chunkInitial : BlockState where
  mem := fun r _ => if r="label" then .of .int 0 else .real (if r="logits" then 2 else 17)
  regs := fun _ _ _ => none
  pids := fun ax => if ax=1 then 1 else 0
  numPids := fun _ => 2
  undef := fun _ _ => 0

/-- The complete nonzero-chunk source executes and leaves the LSE cell at its
initial value. This is the actual nested-store source, not a repaired variant. -/
theorem nonzero_chunk_unchanged :
    (exec chunkKernel chunkInitial).map (·.readMem "lse" 1)=some 17 := by
  simp only [chunkKernel,chunked,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType,ComputeDType.eraseDType,Bool.false_eq_true,↓reduceIte,Except.bind,bind]
  simp [exec,stepStmts,stepStmt,evalOp,evalOp.eq_def,Option.bind,Option.map,
    Tile.bop,Tile.cop,Tile.ptrAdd,Tile.uop,Tile.scalar,-Tile.scalar_eta,
    Tile.reduceMax,Tile.reduceMaxDrop,Tile.reduceSum,Tile.reduceSumDrop,
    TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,TileShape.allIndices,
    NumericDType.add,NumericDType.sub,NumericDType.mul,NumericDType.div,
    ComparableDType.lt,ComparableDType.ne,ComparableDType.eq,FloatDType.cast,
    chunkInitial,Region.cast,BlockState.readMemValue,BlockState.setReg,BlockState.readMem,BlockState.readMemTyped,MemCell.real,MemCell.of,MemCell.readAs]

/-- One active logit of value two has mathematical log-sum-exp two. -/
refutation all_chunks_write_lse :
    (exec chunkKernel chunkInitial).map (·.readMem "lse" 1)=some 2 := by
  rw [nonzero_chunk_unchanged]
  norm_num

def softcapKernel : ComputeKernel :=
  (forward .fp32 Bool.true Bool.false Bool.false "logits" "loss" "lse" ⟨"label"⟩ 1 1 2 1 1).eraseDType

def softcapInitial : BlockState where
  mem := fun r _ => if r="label" then .of .int 0 else .real 0
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun _ _ => 0

private theorem softcap_read (i : Nat) : softcapInitial.readMem "logits" i = 0 := by
  rfl

private theorem softcap_label : softcapInitial.readMemValue .int ⟨"label"⟩ 0 = 0 := by
  simp [softcapInitial,BlockState.readMemValue,BlockState.readMemTyped,MemCell.of,MemCell.readAs]

private theorem sup_two (f : Fin 2 → WithBot ℝ) (h : (Finset.univ : Finset (Fin 2)).Nonempty) :
    Finset.univ.sup' h f = max (f 0) (f 1) := by
  rw [Finset.sup'_eq_sup]
  simp [Finset.univ_fin2]

private theorem sum_two (f : Fin 2 → WithBot ℝ) : ∑ j, f j = f 0 + f 1 := by
  simp [Fin.sum_univ_succ]

/-- The masked second lane becomes -SOFTCAP and contributes exp(-1). -/
theorem softcap_padding_loss :
    (exec softcapKernel softcapInitial).map (·.readMem "loss" 0) =
      some (Real.log (1 + Real.exp (-1))) := by
  simp only [softcapKernel,forward,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType,ComputeDType.eraseDType,Bool.false_eq_true,↓reduceIte,Except.bind,bind]
  simp [exec,stepStmts,stepStmt,evalOp.eq_def,Option.bind,Option.map,
    Tile.bop,Tile.cop,Tile.ptrAdd,Tile.uop,Tile.scalar,-Tile.scalar_eta,
    Tile.reduceMax,Tile.reduceMaxDrop,Tile.reduceSum,Tile.reduceSumDrop,
    TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,TileShape.allIndices,
    NumericDType.add,NumericDType.sub,NumericDType.mul,NumericDType.div,
    ComparableDType.lt,ComparableDType.ne,ComparableDType.eq,FloatDType.cast,
    softcap_label,softcap_read,show softcapInitial.pids=(fun _ => 0) from rfl,Region.cast,
    WithBot.realTanh,WithBot.realMul,WithBot.realDiv,WithBot.realSub,
    WithBot.realExp,WithBot.realLog,Real.tanh_zero,
    Finset.univ_fin2,Finset.sup'_insert,Finset.sup'_singleton,Fin.sum_univ_succ]
  erw [sup_two,sum_two]
  have hm : max (some (0 : ℝ) : WithBot ℝ) (some (-1) : WithBot ℝ) = (some 0 : WithBot ℝ) := by
    change some (max (0 : ℝ) (-1)) = some 0
    norm_num
  norm_num [WithBot.unbotD,WithBot.recBotCoe,WithBot.realTanh,WithBot.realExp,
    WithBot.realLog,Option.map₂,hm,WithBot.add,WithTop.add]
  simp_all [WithBot.unbotD,WithBot.recBotCoe,Option.map₂,WithBot.add,WithTop.add]
  erw [← WithBot.coe_max]
  norm_num [WithBot.unbotD,WithBot.recBotCoe,Option.map₂,Option.bind,Option.map]
  rfl

/-- Ordinary one-class cross entropy is zero, but padded softcapping is not. -/
refutation softcap_padding_is_ordinary_cross_entropy :
    (exec softcapKernel softcapInitial).map (·.readMem "loss" 0) = some 0 := by
  rw [softcap_padding_loss]
  simp only [Option.some.injEq]
  exact ne_of_gt (Real.log_pos (by have h := Real.exp_pos (-1); linarith))


#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FastCeLoss.Counterexamples
