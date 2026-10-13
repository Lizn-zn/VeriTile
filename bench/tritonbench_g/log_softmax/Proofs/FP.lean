import bench.tritonbench_g.log_softmax.Kernels
import bench.tritonbench_g.cache_transform_triton.Proofs.IndexRewrite

namespace VeriTile.Bench.TritonBenchG.LogSoftmax.FPProof
open VeriTile.Triton FP.Structural FP.ProfiledRewrite

def indexStmt (v : Bool) (R : Nat) : ComputeStmt :=
  .assign .nat [R] "m_offset" (.alg (.add .nat .scalarL
    (if v then .mul .nat .nil (.constNat R) (.ref .nat [] "pid_m")
     else .mul .nat .nil (.ref .nat [] "pid_m") (.constNat R)) (.arange R)))

def sourcePrefix (v : Bool) (R : Nat) : List ComputeStmt :=
  [.assign .nat [] "pid_m" (.alg (.programId 0)),
   .assign .nat [] "pid_k" (.alg (.programId 1)), indexStmt v R]

/-- Exact index commutation preserves arbitrary states and failures. -/
theorem prefix_run {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (R : Nat) (after : List ComputeStmt) :
    run (engine M) (sourcePrefix Bool.false R ++ after) s =
      run (engine M) (sourcePrefix Bool.true R ++ after) s := by
  change Contextual M s
    [.assign .nat [] "pid_m" (.alg (.programId 0)),
     .assign .nat [] "pid_k" (.alg (.programId 1))] after
    (indexStmt Bool.false R) (indexStmt Bool.true R)
  unfold indexStmt
  simp only [Bool.false_eq_true, ↓reduceIte]
  fp_prove

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.LogSoftmax.FPProof
