import bench.tritonbench_g.l2_norm_triton1.Proofs.FP

namespace VeriTile.Bench.TritonBenchG.SwigluFwd.FPProof
open VeriTile.Triton FP.Structural FP.ProfiledRewrite

def base (v : Bool) (r : RegionName) (name : RegName) (stride : Nat) : ComputeStmt :=
  let a : Op .nat [] := .ref .nat [] "row"
  let b : Op .nat [] := .constNat stride
  .assign .ptr [] name (.alg (.ptrAdd .nil (.ptrBase r)
    (if v then .mul .nat .nil b a else .mul .nat .nil a b)))

def sourcePrefix (v : Bool) (X Y OUT : RegionName) (sx sy so B : Nat) : List ComputeStmt :=
  [.assign .nat [] "row" (.alg (.programId 0)),
   .assign .nat [] "start_col" (.alg (.mul .nat .nil (.programId 1) (.constNat B))),
   base v X "X" sx, base v Y "Y" sy, base v OUT "OUT" so]

theorem base_step {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (r : RegionName) (name : RegName) (stride : Nat) :
    step (engine M) (base Bool.false r name stride) s =
      step (engine M) (base Bool.true r name stride) s := by
  have h := L2NormTriton1.FPProof.ptr_index_commutes M s [] r
    (.ref .nat [] "row") (.constNat stride) s (by simp [run])
  simpa only [base, Bool.false_eq_true, ↓reduceIte, step, FP.Structural.evalExpr] using
    congrArg (fun x => x >>= fun value => some (s.setReg name .ptr [] value)) h

/-- Full suffix equivalence from an arbitrary initial state, including failure. -/
theorem prefix_run {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (X Y OUT : RegionName) (sx sy so B : Nat) (after : List ComputeStmt) :
    run (engine M) (sourcePrefix Bool.false X Y OUT sx sy so B ++ after) s =
      run (engine M) (sourcePrefix Bool.true X Y OUT sx sy so B ++ after) s := by
  rw [run_append, run_append]
  apply congrArg (fun (result : Option (State α)) => result.bind (run (engine M) after))
  simp only [sourcePrefix, run, base_step]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SwigluFwd.FPProof
