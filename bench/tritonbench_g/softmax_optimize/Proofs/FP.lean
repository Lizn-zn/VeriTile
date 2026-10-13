import VeriTile.Triton.Float.ExpressionRewrite
import VeriTile.Triton.Float.RewriteTactics
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.SoftmaxOptimize.FPProof
open VeriTile.Triton FP.Structural FP.ProfiledRewrite

/-- Exact integer address arithmetic, independent of every floating interpretation. -/
@[spec_rule] theorem int_product_commutes {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) (before : List ComputeStmt) (a b : Op .int []) :
    OpEquivalent M s before (.mul .int .nil a b) (.mul .int .nil b a) := by
  intro t _
  change (evalOp (engine M) none a t >>= fun x =>
    evalOp (engine M) none b t >>= fun y => some (bop Int.mul .nil x y)) =
    (evalOp (engine M) none b t >>= fun y =>
    evalOp (engine M) none a t >>= fun x => some (bop Int.mul .nil y x))
  cases ha : evalOp (engine M) none a t <;>
    cases hb : evalOp (engine M) none b t
  all_goals first
    | rfl
    | apply congrArg some
      funext i
      exact Int.mul_comm _ _

@[spec_rule] theorem nat_product_commutes {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) (before : List ComputeStmt) (a b : Op .nat []) :
    OpEquivalent M s before (.mul .nat .nil a b) (.mul .nat .nil b a) := by
  intro t _
  change (evalOp (engine M) none a t >>= fun x =>
    evalOp (engine M) none b t >>= fun y => some (bop Nat.mul .nil x y)) =
    (evalOp (engine M) none b t >>= fun y =>
    evalOp (engine M) none a t >>= fun x => some (bop Nat.mul .nil y x))
  cases ha : evalOp (engine M) none a t <;>
    cases hb : evalOp (engine M) none b t
  all_goals first
    | rfl
    | apply congrArg some
      funext i
      exact Nat.mul_comm _ _

/-- Lift an unconditional exact body equality through signed ranges, including
negative starts and failure. No artificial floating domain is introduced. -/
theorem contextual_signed_range {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) (before after : List ComputeStmt) (index : RegName)
    (start stop : Op .int []) (stride : Op .nat []) (lhs rhs : List ComputeStmt)
    (hbody : ∀ t, run (engine M) lhs t = run (engine M) rhs t) :
    Contextual M s before after (.forRangeInt index start stop stride lhs)
      (.forRangeInt index start stop stride rhs) := by
  unfold Contextual
  rw [run_append, run_append]
  cases hp : run (engine M) before s with
  | none => rfl
  | some t =>
    simp only [bind, Option.bind_some, run, step]
    cases ha : FP.Structural.evalOp (engine M) none start t with
    | none => rfl
    | some a =>
      simp only [Option.bind_some]
      cases hb : FP.Structural.evalOp (engine M) none stop t with
      | none => rfl
      | some b =>
        simp only [Option.bind_some]
        cases hdt : FP.Structural.evalOp (engine M) none stride t with
        | none => rfl
        | some d =>
          simp only [Option.bind_some]
          rw [rangeInt_congr_on_reachable (engine M) index (a PUnit.unit)
            (b PUnit.unit) (d PUnit.unit) lhs rhs t (fun _ u _ _ _ ↦ hbody _)]

@[spec_rule] theorem contextual_index_loop {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) (before after : List ComputeStmt) (index : RegName)
    (start stop : Op .int []) (stride : Op .nat []) (init : ComputeStmt)
    (rest : List ComputeStmt) (B : Nat) (a b : Op .int []) :
    Contextual M s before after
      (.forRangeInt index start stop stride
        (init :: .assign .int [B] "offset" (.alg (.add .int .scalarL
          (.mul .int .nil a b) (.ref .int [B] "n_offsets"))) :: rest))
      (.forRangeInt index start stop stride
        (init :: .assign .int [B] "offset" (.alg (.add .int .scalarL
          (.mul .int .nil b a) (.ref .int [B] "n_offsets"))) :: rest)) := by
  apply contextual_signed_range
  intro t
  change Contextual M t [init] rest
    (.assign .int [B] "offset" (.alg (.add .int .scalarL
      (.mul .int .nil a b) (.ref .int [B] "n_offsets"))))
    (.assign .int [B] "offset" (.alg (.add .int .scalarL
      (.mul .int .nil b a) (.ref .int [B] "n_offsets"))))
  fp_prove

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxOptimize.FPProof
