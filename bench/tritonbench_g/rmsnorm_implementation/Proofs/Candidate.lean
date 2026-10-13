import bench.tritonbench_g.rmsnorm_implementation.Proofs.Source
import bench.tritonbench_g.rmsnorm_implementation.Proofs.Real
import VeriTile.Triton.Semantics.ProgramRewrite

namespace VeriTile.Bench.TritonBenchG.RmsnormImplementation
open VeriTile.Triton
set_option maxHeartbeats 1000000

def indexPrefix (v : Bool) (sxb sxm : Nat) : List Stmt :=
  [.assign .nat [] "pid_batch" (.programId 0),
   .assign .nat [] "pid_m" (.programId 1),
   .assign .nat [] "offset_m"
     (.add .nat .nil
       (if v then .mul .nat .nil (.constNat sxb) (.ref .nat [] "pid_batch")
        else .mul .nat .nil (.ref .nat [] "pid_batch") (.constNat sxb))
       (.mul .nat .nil (.ref .nat [] "pid_m") (.constNat sxm)))]

theorem prefix_source (v : Bool) (X W O : RegionName)
    (sxb sxm sxk sw sob som sok N B : Nat) (eps : ℝ) :
    (mathKernel v X W O sxb sxm sxk sw sob som sok N B eps).body.take 3 =
      indexPrefix v sxb sxm := by
  cases v <;> rfl

theorem prefix_flat (A : FlatAlloc) (X W O : RegionName)
    (sxb sxm sxk sw sob som sok N B : Nat) (eps : ℝ) (s : BlockState) :
    stepStmts (A.flattenStmts ((mathKernel Bool.false X W O sxb sxm sxk sw sob som sok N B eps).body.take 3)) s =
      stepStmts (A.flattenStmts ((mathKernel Bool.true X W O sxb sxm sxk sw sob som sok N B eps).body.take 3)) s := by
  rw [prefix_source, prefix_source]
  simp [indexPrefix, FlatAlloc.flattenStmts, FlatAlloc.flattenStmt, FlatAlloc.flattenOp.eq_def,
    stepStmts, stepStmt, evalOp.eq_def, Tile.bop, NumericDType.mul, Nat.mul_comm]

/-- Full execution transport preserves both loops, all stores and the entire state. -/
theorem typed_flat_to_math (A : FlatAlloc) (input weight output : Storage) (v : Bool)
    (X W O : RegionName) (sxb sxm sxk sw sob som sok N B : Nat) (eps : ℝ) (s : BlockState) :
    exec (A.flattenKernel
      (typedKernel input weight output v X W O sxb sxm sxk sw sob som sok N B eps).eraseDType.toAlgKernel) s =
      exec (A.flattenKernel
        (uncastMathKernel X W O sxb sxm sxk sw sob som sok N B eps).toAlgKernel) s := by
  rw [typed_projection]
  cases v
  · rfl
  · change exec (A.flattenKernel (mathKernel Bool.true X W O sxb sxm sxk sw sob som sok N B eps)) s =
      exec (A.flattenKernel (mathKernel Bool.false X W O sxb sxm sxk sw sob som sok N B eps)) s
    unfold exec FlatAlloc.flattenKernel
    rw [← List.take_append_drop 3 (mathKernel Bool.true X W O sxb sxm sxk sw sob som sok N B eps).body,
      ← List.take_append_drop 3 (mathKernel Bool.false X W O sxb sxm sxk sw sob som sok N B eps).body]
    have hs : (mathKernel Bool.true X W O sxb sxm sxk sw sob som sok N B eps).body.drop 3 =
        (mathKernel Bool.false X W O sxb sxm sxk sw sob som sok N B eps).body.drop 3 := rfl
    simp only [FlatAlloc.flattenStmts_append, ProgramRewrite.append, prefix_flat, hs]

def uncastVarBody (X : RegionName) (sxk N B : Nat) : List Stmt :=
  (rmsVarLoopBody X sxk N B).take 3 ++
    [.assign .real [B] "xf" (.ref .real [B] "x")] ++
    (rmsVarLoopBody X sxk N B).drop 4

theorem uncast_var_flat (A : FlatAlloc) (X : RegionName) (sxk N B : Nat) (s : BlockState) :
    stepStmts (A.flattenStmts (uncastVarBody X sxk N B)) s =
      stepStmts (A.flattenStmts (rmsVarLoopBody X sxk N B)) s := by
  have hcast : ∀ t : BlockState,
      stepStmts (A.flattenStmts [.assign .real [B] "xf" (.ref .real [B] "x")]) t =
      stepStmts (A.flattenStmts [.assign .real [B] "xf"
        (.castFloat .real .real (.ref .real [B] "x"))]) t := by
    intro t
    simp [FlatAlloc.flattenStmts, FlatAlloc.flattenStmt, FlatAlloc.flattenOp.eq_def,
      stepStmts, stepStmt, evalOp.eq_def, FloatDType.cast]
    cases t.regs .real [B] "x" <;> rfl
  have hb : rmsVarLoopBody X sxk N B =
      (rmsVarLoopBody X sxk N B).take 3 ++
      [.assign .real [B] "xf" (.castFloat .real .real (.ref .real [B] "x"))] ++
      (rmsVarLoopBody X sxk N B).drop 4 := rfl
  conv_rhs => rw [hb]
  unfold uncastVarBody
  simp only [FlatAlloc.flattenStmts_append, ProgramRewrite.append, hcast]

theorem uncast_flat_exec (A : FlatAlloc) (X W O : RegionName)
    (sxb sxm sxk sw sob som sok N B : Nat) (eps : ℝ) (s : BlockState) :
    exec (A.flattenKernel (uncastMathKernel X W O sxb sxm sxk sw sob som sok N B eps).toAlgKernel) s =
      exec (A.flattenKernel (rmsnorm_implementation X W O sxb sxm sxk sw sob som sok N B eps).toAlgKernel) s := by
  have hbody : (uncastMathKernel X W O sxb sxm sxk sw sob som sok N B eps).toAlgKernel.body =
      rmsVarPreLoop sxb sxm B ++
      [.forRange "block_n_strart_ptr" 0 N B (uncastVarBody X sxk N B)] ++
      rmsStdPostLoop N B eps ++
      [.forRange "block_n_strart_ptr" 0 N B (rmsOutLoopBody X W O sxk sw sob som sok N B)] := by
    simp [uncastMathKernel, rmsVarPreLoop, uncastVarBody, rmsVarLoopBody,
      rmsStdPostLoop, rmsOutLoopBody, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
    repeat' constructor
  have hloop : ∀ t : BlockState,
      stepStmts (A.flattenStmts [.forRange "block_n_strart_ptr" 0 N B (uncastVarBody X sxk N B)]) t =
      stepStmts (A.flattenStmts [.forRange "block_n_strart_ptr" 0 N B (rmsVarLoopBody X sxk N B)]) t := by
    intro t
    simp only [FlatAlloc.flattenStmts, FlatAlloc.flattenStmt, stepStmts, stepStmt]
    rw [ProgramRewrite.range_congr _ _ _ _ _ _ (uncast_var_flat A X sxk N B)]
  unfold exec FlatAlloc.flattenKernel
  rw [hbody, rmsnorm_implementation_toAlg_body]
  simp only [FlatAlloc.flattenStmts_append, ProgramRewrite.append, hloop]

/-- Both typed variants execute the preserved real kernel, including all loops,
all stores and all failure outcomes. Removing a real-to-real cast is proved
separately through every variance-loop iteration. -/
theorem typed_flat_exec (A : FlatAlloc) (input weight output : Storage) (v : Bool)
    (X W O : RegionName) (sxb sxm sxk sw sob som sok N B : Nat) (eps : ℝ) (s : BlockState) :
    exec (A.flattenKernel
      (typedKernel input weight output v X W O sxb sxm sxk sw sob som sok N B eps).eraseDType.toAlgKernel) s =
      exec (A.flattenKernel
        (rmsnorm_implementation X W O sxb sxm sxk sw sob som sok N B eps).toAlgKernel) s :=
  (typed_flat_to_math A input weight output v X W O sxb sxm sxk sw sob som sok N B eps s).trans
    (uncast_flat_exec A X W O sxb sxm sxk sw sob som sok N B eps s)

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsnormImplementation
