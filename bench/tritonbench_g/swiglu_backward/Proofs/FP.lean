import bench.tritonbench_g.swiglu_backward.Kernels
import bench.tritonbench_g.swiglu_fwd.Proofs.FP
namespace VeriTile.Bench.TritonBenchG.SwigluBackward.FPProof
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open SwigluFwd.FPProof
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000

def sourcePrefix (v : Bool) (X Y DOUT OUT DX DY : RegionName)
    (sx sy sd so sdx sdy B : Nat) (R : Bool) : List ComputeStmt :=
  [.assign .nat [] "row" (.alg (.programId 0)),
   .assign .nat [] "start_col" (.alg (.mul .nat .nil (.programId 1) (.constNat B))),
   base v X "X" sx, base v Y "Y" sy, base v DOUT "DOUT" sd,
   .ifThen (.constBool R) [base v OUT "OUT" so],
   base v DX "DX" sdx, base v DY "DY" sdy,
   .assign .nat [B] "cols" (.alg
     (.add .nat .scalarL (.ref .nat [] "start_col") (.arange B)))]

theorem conditional_base_step {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (r : RegionName) (name : RegName) (stride : Nat) (R : Bool) :
    step (engine M) (.ifThen (.constBool R) [base Bool.false r name stride]) s =
      step (engine M) (.ifThen (.constBool R) [base Bool.true r name stride]) s := by
  cases R
  · simp [step, evalOp_unfold, run, engine]
  · simpa [step, evalOp_unfold, run, engine] using base_step M s r name stride

theorem prefix_run {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (X Y DOUT OUT DX DY : RegionName) (sx sy sd so sdx sdy B : Nat) (R : Bool)
    (after : List ComputeStmt) :
    run (engine M) (sourcePrefix Bool.false X Y DOUT OUT DX DY sx sy sd so sdx sdy B R ++ after) s =
      run (engine M) (sourcePrefix Bool.true X Y DOUT OUT DX DY sx sy sd so sdx sdy B R ++ after) s := by
  rw [run_append, run_append]
  apply congrArg (fun (result : Option (State α)) => result.bind (run (engine M) after))
  simp only [sourcePrefix, run, base_step, conditional_base_step]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SwigluBackward.FPProof
