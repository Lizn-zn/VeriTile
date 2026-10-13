import bench.tritonbench_g.var_len_copy.Proofs.FP

namespace VeriTile.Bench.TritonBenchG.VarLenCopy.DiscreteFP
open VeriTile.Triton FP.Structural DiscreteSource
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000

def body (storage : Storage) (v : Bool) (os ol ns : Region .int)
    (src dst : RegionName) (B : Nat) : List ComputeStmt :=
  match ((kernel storage v os ol ns src dst B).surfaceBody[6]? : Option ComputeStmt) with
  | some (.forRangeInt _ _ _ _ body) => body
  | _ => []

theorem body_equivalence {α : Type} [Inhabited α] (M : Algebra α)
    (storage : Storage) (os ol ns : Region .int) (src dst : RegionName)
    (B : Nat) (s : State α) :
    run M (body storage Bool.false os ol ns src dst B) s =
      run M (body storage Bool.true os ol ns src dst B) s := by
  cases storage
  all_goals simp only [body, kernel, ComputeKernel.surfaceBody,
    List.getElem?_cons_zero, List.getElem?_cons_succ, run, step, evalExpr, evalComputeOp]
  all_goals
    cases ho : s.regs .int [] "old_start" <;> cases hi : s.regs .int [] "i"
    all_goals simp [evalOp_unfold, ho, hi, numeric, FP.Structural.bop,
      Broadcast.leftIndex, Broadcast.rightIndex, Int.add_comm]
  all_goals unfold FP.Structural.bop
  all_goals simp only [Broadcast.leftIndex, Broadcast.rightIndex, Int.add_comm]

theorem source_decomposition (storage : Storage) (v : Bool)
    (os ol ns : Region .int) (src dst : RegionName) (B : Nat) :
    kernel storage v os ol ns src dst B = .mk [ol.cast,os.cast,ns.cast,src] [dst]
      (FPProof.prefixBody os ol ns src dst B ++
        [.forRangeInt "i" (.constInt 0) (.ref .int [] "length")
          (.constNat B) (body storage v os ol ns src dst B)]) := by
  cases storage <;> cases v <;> rfl

/-- All aliases and all evaluator failures are preserved across the complete
signed loop; only exact mathematical signed address addition is commuted. -/
theorem source_equivalence {α : Type} [Inhabited α] (M : Algebra α)
    (storage : Storage) (os ol ns : Region .int) (src dst : RegionName)
    (B : Nat) (s : State α) :
    exec M (kernel storage Bool.false os ol ns src dst B) s =
      exec M (kernel storage Bool.true os ol ns src dst B) s := by
  rw [source_decomposition, source_decomposition]
  simp only [FP.Structural.exec, run_append]
  apply congrArg (Option.bind (run M (FPProof.prefixBody os ol ns src dst B) s))
  funext t
  simp only [run, step, evalExpr]
  simp only [evalOp_unfold]
  cases h : t.regs .int [] "length" with
  | none => rfl
  | some stop =>
    change (rangeInt M "i" 0 (stop PUnit.unit) B (body storage Bool.false os ol ns src dst B) t >>= fun x ↦ some x) =
      (rangeInt M "i" 0 (stop PUnit.unit) B (body storage Bool.true os ol ns src dst B) t >>= fun x ↦ some x)
    rw [FPProof.rangeInt_congr M "i" (stop PUnit.unit) B _ _
      (body_equivalence M storage os ol ns src dst B)]

end VeriTile.Bench.TritonBenchG.VarLenCopy.DiscreteFP

#guard_msgs (drop info) in
#auditModuleAxioms
