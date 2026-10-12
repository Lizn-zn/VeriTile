import bench.tritonbench_g.attn_fwd_causal.Kernels
import bench.tritonbench_g.attn_fwd_causal.Proofs.RealFormula
import VeriTile.Triton.Semantics.EvalOp
import VeriTile.Triton.Source
import VeriTile.Meta.StatementAudit

/-! The actual diagonal mask and additive score expression, evaluated in real
semantics. Register hypotheses describe intermediate operands; a full source
loop invariant must establish them from memory inputs. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealExpressions
open VeriTile.Triton
noncomputable section
set_option linter.unusedSimpArgs false

def maskExpr (BM BN : Nat) : Op .bool [BM,BN] :=
  .ge .nat (.consR (.consL .nil))
    (.expandDim ⟨1, by simp⟩ (.ref .nat [BM] "offs_m"))
    (.add .nat .scalarL (.ref .nat [] "start_n")
      (.expandDim ⟨0, by simp⟩ (.ref .nat [BN] "offs_n")))

def maskTile (BM BN startM startN : Nat) : Tile .bool [BM,BN] :=
  ⟨fun i => decide (startN+i.2.1.val ≤ startM*BM+i.1.val)⟩

def penaltyExpr (BM BN : Nat) (mask : RegName) : Op .real [BM,BN] :=
  .add .real (.consSame (.consSame .nil)) (.ref .real [BM,BN] "qk")
    (.where (.ref .bool [BM,BN] mask) (.broadcast (.const 0) [BM,BN])
      (.broadcast (.sub .real .nil (.const 0.0) (.const 1.0e6)) [BM,BN]))

def scoreTile {BM BN : Nat} (scores : Fin BM → Fin BN → ℝ) : Tile .real [BM,BN] :=
  ⟨fun i => some (scores i.1 i.2.1)⟩

theorem eval_mask (s : BlockState) (BM BN startM startN : Nat)
    (hm : s.regs .nat [BM] "offs_m" = some ⟨fun i => startM*BM+i.1.val⟩)
    (hn : s.regs .nat [BN] "offs_n" = some ⟨fun i => i.1.val⟩)
    (hs : s.regs .nat [] "start_n" = some (Tile.scalar startN)) :
    evalOp (maskExpr BM BN) s = some (maskTile BM BN startM startN) := by
  simp only [maskExpr, evalOp.eq_def, hm, hn, hs, bind, Option.bind]
  rfl

theorem eval_penalty (s : BlockState) (BM BN startM startN : Nat) (mask : RegName)
    (scores : Fin BM → Fin BN → ℝ)
    (hq : s.regs .real [BM,BN] "qk" = some (scoreTile scores))
    (hm : s.regs .bool [BM,BN] mask = some (maskTile BM BN startM startN)) :
    evalOp (penaltyExpr BM BN mask) s = some (scoreTile (fun row key =>
      scores row key + RealFormula.penalty Bool.true (startM*BM+row.val) (startN+key.val))) := by
  simp only [penaltyExpr, evalOp.eq_def, hq, hm, bind, Option.bind]
  congr 1
  apply Tile.ext
  intro i
  rcases i with ⟨row,key,⟨⟩⟩
  by_cases h : startM*BM+row.val < startN+key.val
  · have hn : ¬ startN+key.val ≤ startM*BM+row.val := by omega
    norm_num [scoreTile, maskTile, Tile.bop_data, Tile.select,
      Tile.scalar, NumericDType.add, NumericDType.sub, WithBot.realAdd, WithBot.realSub,
      RealFormula.penalty, h, hn, -Tile.scalar_eta]
  · have hn : startN+key.val ≤ startM*BM+row.val := by omega
    norm_num [scoreTile, maskTile, Tile.bop_data, Tile.select,
      Tile.scalar, NumericDType.add, NumericDType.sub, WithBot.realAdd, WithBot.realSub,
      RealFormula.penalty, h, hn, -Tile.scalar_eta]

/-- The natural subtraction in the load mask has the same active lanes as
an absolute key bound, including an empty tail when the loop starts past N. -/
theorem tail_mask_iff (N startN lane : Nat) : lane < N-startN ↔ startN+lane < N := by
  omega

def assignment? (code : List ComputeStmt) (name : RegName) (occurrence : Nat := 0) :=
  (code.drop (Source.prefixBefore code name occurrence).length).head?

section
variable (Q K V QS KS Out : RegionName)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      Z H N HD BM BN BD HA stage : Nat) (hs : stage = 2 ∨ stage = 3)

/-- Extract the second qk write in the second helper from the complete source,
then relate that exact operation to the real expression above. -/
theorem source_penalty (candidate : Bool) :
    ((Source.dynamicRange? ((if candidate then TypedSource.reciprocal else TypedSource.original) Q K V QS KS Out
      sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      Z H N HD BM BN BD HA stage hs).surfaceBody "start_n" 1).map
      (fun r => assignment? r.body "qk" 1)) =
        some (some (.assign .real [BM,BN] "qk" (.compute (.alg .fp32
          (penaltyExpr BM BN "mask"))))) := by
  cases candidate
  all_goals
    simp [TypedSource.original, TypedSource.reciprocal, ComputeKernel.surfaceBody,
      Source.dynamicRange?, Source.prefixBefore, assignment?, penaltyExpr]
    rfl

theorem source_mask (candidate : Bool) :
    ((Source.dynamicRange? ((if candidate then TypedSource.reciprocal else TypedSource.original)
      Q K V QS KS Out sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      Z H N HD BM BN BD HA stage hs).surfaceBody "start_n" 1).map
      (fun r => assignment? r.body "mask")) = some (some
        (.assign .bool [BM,BN] "mask" (.alg (maskExpr BM BN)))) := by
  cases candidate <;> rfl

end
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealExpressions
