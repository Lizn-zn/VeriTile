import bench.tritonbench_g.sin_kernel.Kernels
import bench.tritonbench_g.cache_transform_triton.Proofs.IndexRewrite

namespace VeriTile.Bench.TritonBenchG.SinKernel.FPProof
open VeriTile.Triton FP.Structural FP.ProfiledRewrite

def indexStmt (v : Bool) (B : Nat) : ComputeStmt :=
  .assign .nat [] "block_start" (.alg
    (if v then .mul .nat .nil (.constNat B) (.ref .nat [] "pid")
     else .mul .nat .nil (.ref .nat [] "pid") (.constNat B)))

def sourcePrefix (v : Bool) (B : Nat) : List ComputeStmt :=
  [.assign .nat [] "pid" (.alg (.programId 0)), indexStmt v B]

theorem index_step {α : Type} [Inhabited α] (M : Algebra α) (s : State α) (B : Nat) :
    step (engine M) (indexStmt Bool.false B) s =
      step (engine M) (indexStmt Bool.true B) s := by
  have h := CacheTransformTriton.IndexRewrite.index_product_commutes M s []
    (.ref .nat [] "pid") (.constNat B) s (by simp [run])
  simpa only [indexStmt,
    Bool.false_eq_true, ↓reduceIte, step, FP.Structural.evalExpr] using
    congrArg (fun x => x >>= fun value => some (s.setReg "block_start" .nat [] value)) h

/-- No floating operation changes; all states and failure paths are retained. -/
theorem prefix_run {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (B : Nat) (after : List ComputeStmt) :
    run (engine M) (sourcePrefix Bool.false B ++ after) s =
      run (engine M) (sourcePrefix Bool.true B ++ after) s := by
  rw [run_append, run_append]
  apply congrArg (fun (result : Option (State α)) => result.bind (run (engine M) after))
  simp only [sourcePrefix, run, index_step]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SinKernel.FPProof
