import bench.tritonbench_g.var_len_copy.Kernels
import bench.tritonbench_g.var_len_copy.Proofs.FPAddress
import bench.tritonbench_g.var_len_copy.Proofs.FPRange

namespace VeriTile.Bench.TritonBenchG.VarLenCopy.FPProof
open VeriTile.Triton FP.Structural TypedSource
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000


def body (input output : Storage) (v : Bool) (os ol ns : Region .int)
    (src dst : RegionName) (B : Nat) : List ComputeStmt :=
  match ((kernel input output v os ol ns src dst B).surfaceBody[6]? : Option ComputeStmt) with
  | some (.forRangeInt _ _ _ _ body) => body
  | _ => []

def loadStatement (input : Storage) (v : Bool) (os ol ns : Region .int)
    (src dst : RegionName) (B : Nat) : ComputeStmt :=
  (body input .fp32 v os ol ns src dst B).headD (.assign .nat [] "unused" (.alg (.constNat 0)))

theorem load_equivalence {α : Type} [Inhabited α] (M : Algebra α)
    (input : Storage) (os ol ns : Region .int) (src dst : RegionName)
    (B : Nat) (s : State α) :
    step M (loadStatement input Bool.false os ol ns src dst B) s =
      step M (loadStatement input Bool.true os ol ns src dst B) s := by
  cases input
  all_goals simp only [loadStatement, body, kernel, ComputeKernel.surfaceBody,
    List.getElem?_cons_zero, List.getElem?_cons_succ, List.headD_cons]
  all_goals simp only [step, evalExpr, evalComputeOp]
  all_goals
    cases ho : s.regs .int [] "old_start" <;> cases hi : s.regs .int [] "i"
    all_goals simp [evalOp_unfold, ho, hi, numeric, FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Int.add_comm]
  all_goals unfold FP.Structural.bop
  all_goals simp only [Broadcast.leftIndex, Broadcast.rightIndex, Int.add_comm]

theorem body_decomposition (input output : Storage) (v : Bool)
    (os ol ns : Region .int) (src dst : RegionName) (B : Nat) :
    body input output v os ol ns src dst B =
      loadStatement input v os ol ns src dst B ::
        (body input output Bool.false os ol ns src dst B).tail := by
  cases input <;> cases output <;> cases v <;> rfl

theorem body_equivalence {α : Type} [Inhabited α] (M : Algebra α)
    (input output : Storage) (os ol ns : Region .int) (src dst : RegionName)
    (B : Nat) (s : State α) :
    run M (body input output Bool.false os ol ns src dst B) s =
      run M (body input output Bool.true os ol ns src dst B) s := by
  rw [body_decomposition input output Bool.false, body_decomposition input output Bool.true]
  simp only [run, load_equivalence]

end VeriTile.Bench.TritonBenchG.VarLenCopy.FPProof

#guard_msgs (drop info) in
#auditModuleAxioms
