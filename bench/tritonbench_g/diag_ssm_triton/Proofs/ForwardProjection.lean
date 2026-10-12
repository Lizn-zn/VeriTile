import bench.tritonbench_g.diag_ssm_triton.Proofs.Erasure
import bench.tritonbench_g.diag_ssm_triton.Proofs.RealReference
import VeriTile.Triton.Semantics.ProgramRewrite
namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton.ForwardProjection
open VeriTile.Triton TypedSource
set_option maxHeartbeats 6000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def mask (batch dim block : Nat) : Stmt :=
  .assign .bool [block] "mask" (.lt .nat .scalarR (.ref .nat [block] "col_offsets")
    (.mul .nat .nil (.constNat batch) (.constNat dim)))

def offset (batch dim block : Nat) : Stmt :=
  .assign .nat [block] "offsets" (.add .nat .scalarL
    (.mul .nat .nil (.mul .nat .nil (.ref .nat [] "t") (.constNat batch)) (.constNat dim))
    (.ref .nat [block] "col_offsets"))

def prelude (s l : RegionName) (batch dim block : Nat) : List Stmt :=
  let old := RealReference.diagSsmForwardPreLoop s l batch dim block
  old.take 2 ++ [mask batch dim block] ++ old.drop 3

def loopBody (x y : RegionName) (batch dim block : Nat) : List Stmt :=
  offset batch dim block :: (RealReference.diagSsmForwardLoopBody x y batch dim block).drop 1

theorem real_source_body (s x l y : RegionName) (length batch dim block : Nat) :
    (Erasure.forward Bool.false s x l y length batch dim block).toAlgKernel.body =
      prelude s l batch dim block ++ [.forLoop "t" length (loopBody x y batch dim block)] := by
  rfl

theorem index_product (batch dim : Nat) (u : BlockState) :
    evalOp (.mul .nat .nil (.mul .nat .nil (.ref .nat [] "t") (.constNat batch)) (.constNat dim)) u =
      evalOp (.mul .nat .nil (.ref .nat [] "t") (.constNat (batch*dim))) u := by
  simp only [evalOp_mul,evalOp_constNat]
  cases ht : evalOp (.ref .nat [] "t") u
  · rfl
  · simp only [bind,Option.bind_some]
    congr 1
    ext i
    exact Nat.mul_assoc _ _ _

theorem mask_step (batch dim block : Nat) (u : BlockState) :
    stepStmt (mask batch dim block) u =
      stepStmt (.assign .bool [block] "mask"
        (.lt .nat .scalarR (.ref .nat [block] "col_offsets") (.constNat (batch*dim)))) u := by
  simp [mask,stepStmt,evalOp,Tile.bop,Tile.scalar,NumericDType.mul,-Tile.scalar_eta]

theorem offset_step (batch dim block : Nat) (u : BlockState) :
    stepStmt (offset batch dim block) u =
      stepStmt (.assign .nat [block] "offsets" (.add .nat .scalarL
        (.mul .nat .nil (.ref .nat [] "t") (.constNat (batch*dim)))
        (.ref .nat [block] "col_offsets"))) u := by
  simp only [offset,stepStmt,evalOp_add,index_product]

theorem prelude_steps (s l : RegionName) (batch dim block : Nat) (u : BlockState) :
    stepStmts (prelude s l batch dim block) u =
      stepStmts (RealReference.diagSsmForwardPreLoop s l batch dim block) u := by
  simp only [prelude,RealReference.diagSsmForwardPreLoop,List.take,List.drop,List.cons_append,List.nil_append]
  simp only [stepStmts,mask_step]

theorem loop_steps (x y : RegionName) (batch dim block : Nat) (u : BlockState) :
    stepStmts (loopBody x y batch dim block) u =
      stepStmts (RealReference.diagSsmForwardLoopBody x y batch dim block) u := by
  simp only [loopBody,RealReference.diagSsmForwardLoopBody,List.drop,stepStmts,offset_step]

/-- Loop-body equality includes failure, so it composes at every reached state. -/
theorem loop_congr (idx : RegName) (start stop : Nat) (left right : List Stmt)
    (h : ∀ u, stepStmts left u = stepStmts right u) (u : BlockState) :
    stepForLoopAux idx start stop left u = stepForLoopAux idx start stop right u := by
  by_cases hc : start < stop
  · rw [stepForLoopAux.step_lt hc,stepForLoopAux.step_lt hc,h]
    cases stepStmts right (u.setReg idx .nat [] (Tile.scalar start)) with
    | none => rfl
    | some v => exact loop_congr idx (start+1) stop left right h v
  · rw [stepForLoopAux.step_ge (by omega),stepForLoopAux.step_ge (by omega)]
termination_by stop-start
decreasing_by omega

theorem real_source_exec (s x l y : RegionName) (length batch dim block : Nat) (u : BlockState) :
    exec (Erasure.forward Bool.false s x l y length batch dim block).toAlgKernel u =
      exec (RealReference.diag_ssm_forward_kernel s x l y length batch dim block).toAlgKernel u := by
  unfold exec
  rw [real_source_body,RealReference.diag_ssm_forward_kernel_toAlg_body]
  simp only [RealReference.diagSsmForwardProjectedBody,ProgramRewrite.append,prelude_steps]
  congr 1
  funext v
  simp only [stepStmts,stepForLoopAux.forLoop_unfold]
  rw [loop_congr "t" 0 length _ _ (loop_steps x y batch dim block)]

theorem flat_mask_step (A : FlatAlloc) (batch dim block : Nat) (u : BlockState) :
    stepStmt (A.flattenStmt (mask batch dim block)) u =
      stepStmt (A.flattenStmt (.assign .bool [block] "mask"
        (.lt .nat .scalarR (.ref .nat [block] "col_offsets") (.constNat (batch*dim))))) u := by
  simpa only [mask,FlatAlloc.flattenStmt,FlatAlloc.flattenOp] using mask_step batch dim block u

theorem flat_offset_step (A : FlatAlloc) (batch dim block : Nat) (u : BlockState) :
    stepStmt (A.flattenStmt (offset batch dim block)) u =
      stepStmt (A.flattenStmt (.assign .nat [block] "offsets" (.add .nat .scalarL
        (.mul .nat .nil (.ref .nat [] "t") (.constNat (batch*dim)))
        (.ref .nat [block] "col_offsets")))) u := by
  simpa only [offset,FlatAlloc.flattenStmt,FlatAlloc.flattenOp] using offset_step batch dim block u

theorem flat_prelude_steps (A : FlatAlloc) (s l : RegionName) (batch dim block : Nat) (u : BlockState) :
    stepStmts (A.flattenStmts (prelude s l batch dim block)) u =
      stepStmts (A.flattenStmts (RealReference.diagSsmForwardPreLoop s l batch dim block)) u := by
  simp only [prelude,RealReference.diagSsmForwardPreLoop,List.take,List.drop,List.cons_append,List.nil_append,
    FlatAlloc.flattenStmts,stepStmts,flat_mask_step]

theorem flat_loop_steps (A : FlatAlloc) (x y : RegionName) (batch dim block : Nat) (u : BlockState) :
    stepStmts (A.flattenStmts (loopBody x y batch dim block)) u =
      stepStmts (A.flattenStmts (RealReference.diagSsmForwardLoopBody x y batch dim block)) u := by
  simp only [loopBody,RealReference.diagSsmForwardLoopBody,List.drop,
    FlatAlloc.flattenStmts,stepStmts,flat_offset_step]

theorem real_source_flat_exec (A : FlatAlloc) (s x l y : RegionName)
    (length batch dim block : Nat) (u : BlockState) :
    exec (A.flattenKernel (Erasure.forward Bool.false s x l y length batch dim block).toAlgKernel) u =
      exec (A.flattenKernel (RealReference.diag_ssm_forward_kernel s x l y length batch dim block).toAlgKernel) u := by
  unfold exec FlatAlloc.flattenKernel
  rw [real_source_body,RealReference.diag_ssm_forward_kernel_toAlg_body]
  simp only [RealReference.diagSsmForwardProjectedBody,FlatAlloc.flattenStmts_append,
    ProgramRewrite.append,flat_prelude_steps]
  congr 1
  funext v
  simp only [FlatAlloc.flattenStmts,FlatAlloc.flattenStmt,stepStmts,stepForLoopAux.forLoop_unfold]
  rw [loop_congr "t" 0 length _ _ (flat_loop_steps A x y batch dim block)]

theorem typed_source_flat_exec (storage : Storage) (A : FlatAlloc) (s x l y : RegionName)
    (length batch dim block : Nat) (u : BlockState) :
    exec (A.flattenKernel (TypedSource.forward storage Bool.false s x l y length batch dim block).eraseDType.toAlgKernel) u =
      exec (A.flattenKernel (RealReference.diag_ssm_forward_kernel s x l y length batch dim block).toAlgKernel) u := by
  have h := real_source_flat_exec A s x l y length batch dim block u
  simpa only [exec,FlatAlloc.flattenKernel,Erasure.forward_body] using h

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.DiagSsmTriton.ForwardProjection
