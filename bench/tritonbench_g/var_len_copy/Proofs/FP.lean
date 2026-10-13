import bench.tritonbench_g.var_len_copy.Proofs.FPBody

namespace VeriTile.Bench.TritonBenchG.VarLenCopy.FPProof
open VeriTile.Triton FP.Structural TypedSource
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000

def prefixBody (os ol ns : Region .int) (src dst : RegionName) (B : Nat) : List ComputeStmt :=
  (kernel .fp32 .fp32 Bool.false os ol ns src dst B).surfaceBody.take 6

theorem source_decomposition (input output : Storage) (v : Bool)
    (os ol ns : Region .int) (src dst : RegionName) (B : Nat) :
    kernel input output v os ol ns src dst B = .mk [ol.cast,os.cast,ns.cast,src] [dst]
      (prefixBody os ol ns src dst B ++
        [.forRangeInt "i" (.constInt 0) (.ref .int [] "length")
          (.constNat B) (body input output v os ol ns src dst B)]) := by
  cases input <;> cases output <;> cases v <;> rfl

/-- All aliases and all evaluator failures are preserved across the complete
signed loop; only exact mathematical signed address addition is commuted. -/
theorem source_equivalence {α : Type} [Inhabited α] (M : Algebra α)
    (input output : Storage) (os ol ns : Region .int) (src dst : RegionName)
    (B : Nat) (s : State α) :
    exec M (kernel input output Bool.false os ol ns src dst B) s =
      exec M (kernel input output Bool.true os ol ns src dst B) s := by
  rw [source_decomposition, source_decomposition]
  simp only [FP.Structural.exec, run_append]
  apply congrArg (Option.bind (run M (prefixBody os ol ns src dst B) s))
  funext t
  simp only [run, step, evalExpr]
  simp only [evalOp_unfold]
  cases h : t.regs .int [] "length" with
  | none => rfl
  | some stop =>
    change (rangeInt M "i" 0 (stop PUnit.unit) B (body input output Bool.false os ol ns src dst B) t >>= fun x ↦ some x) =
      (rangeInt M "i" 0 (stop PUnit.unit) B (body input output Bool.true os ol ns src dst B) t >>= fun x ↦ some x)
    rw [rangeInt_congr M "i" (stop PUnit.unit) B _ _
      (body_equivalence M input output os ol ns src dst B)]

end VeriTile.Bench.TritonBenchG.VarLenCopy.FPProof

#guard_msgs (drop info) in
#auditModuleAxioms
