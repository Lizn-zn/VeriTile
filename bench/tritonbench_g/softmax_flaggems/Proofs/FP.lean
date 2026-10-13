import VeriTile.Triton.Float.ExpressionRewrite
import VeriTile.Triton.Float.RewriteTactics

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.FPProofs
open VeriTile Triton FP.Structural FP.ProfiledRewrite

/-- Exact scalar address arithmetic; no floating-point commutativity is asserted. -/
@[spec_rule] theorem scalar_nat_mul_comm {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) (before : List ComputeStmt) (a b : Op .nat []) :
    OpEquivalent M s before (.mul .nat .nil a b) (.mul .nat .nil b a) := by
  intro t _
  change (evalOp (engine M) none a t >>= fun va =>
    evalOp (engine M) none b t >>= fun vb =>
      some (bop (numeric (engine M) none .mul .nat) .nil va vb)) =
    (evalOp (engine M) none b t >>= fun vb =>
    evalOp (engine M) none a t >>= fun va =>
      some (bop (numeric (engine M) none .mul .nat) .nil vb va))
  cases ha : evalOp (engine M) none a t <;> cases hb : evalOp (engine M) none b t <;>
    simp [numeric]
  funext i
  exact Nat.mul_comm _ _


private theorem step_pointer_comm {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) (name : RegName) (r : RegionName) (a b : Op .nat []) :
    step (engine M) (.assign .ptr [] name (.alg
      (.ptrAdd .nil (.ptrBase r) (.mul .nat .nil a b)))) s =
    step (engine M) (.assign .ptr [] name (.alg
      (.ptrAdd .nil (.ptrBase r) (.mul .nat .nil b a)))) s := by
  have h := scalar_nat_mul_comm M s [] a b s (by simp [run])
  have hp : evalOp (engine M) none (.ptrAdd .nil (.ptrBase r) (.mul .nat .nil a b)) s =
      evalOp (engine M) none (.ptrAdd .nil (.ptrBase r) (.mul .nat .nil b a)) s := by
    change (evalOp (engine M) none (.ptrBase r) s >>= fun vp =>
      evalOp (engine M) none (.mul .nat .nil a b) s >>= fun vo =>
        some (FP.Structural.bop (fun p o => (p.1, p.2 + o)) .nil vp vo)) = _
    rw [h]
    rfl
  simp only [step, evalExpr, hp]

private theorem contextual_else_unconditional {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) (before after : List ComputeStmt) (condition : Op .bool [])
    (otherwise lhs rhs : List ComputeStmt)
    (hbody : ∀ t, run (engine M) lhs t = run (engine M) rhs t) :
    Contextual M s before after (.ifThenElse condition otherwise lhs)
      (.ifThenElse condition otherwise rhs) := by
  unfold Contextual
  rw [run_append, run_append]
  cases hp : run (engine M) before s with
  | none => rfl
  | some t =>
    simp only [Option.bind_some, run, step]
    cases hc : evalOp (engine M) none condition t with
    | none => rfl
    | some value =>
      simp only [bind, Option.bind_some]
      cases hv : value PUnit.unit with
      | false => simp only [Bool.false_eq_true, ↓reduceIte]; rw [hbody t]
      | true => rfl

/-- Structural descent into the fallback needs no artificial numerical domain. -/
@[spec_rule] theorem contextual_pointer_pair {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) (before after otherwise rest : List ComputeStmt)
    (condition : Op .bool []) (initM initZ : ComputeStmt)
    (input output : RegionName) (a b : Op .nat []) :
    Contextual M s before after
      (.ifThenElse condition otherwise (initM :: initZ ::
        .assign .ptr [] "input_ptr" (.alg
          (.ptrAdd .nil (.ptrBase input) (.mul .nat .nil a b))) ::
        .assign .ptr [] "output_ptr" (.alg
          (.ptrAdd .nil (.ptrBase output) (.mul .nat .nil a b))) :: rest))
      (.ifThenElse condition otherwise (initM :: initZ ::
        .assign .ptr [] "input_ptr" (.alg
          (.ptrAdd .nil (.ptrBase input) (.mul .nat .nil b a))) ::
        .assign .ptr [] "output_ptr" (.alg
          (.ptrAdd .nil (.ptrBase output) (.mul .nat .nil b a))) :: rest)) := by
  have hb (t : State α) :
      run (engine M) (initM :: initZ ::
        .assign .ptr [] "input_ptr" (.alg
          (.ptrAdd .nil (.ptrBase input) (.mul .nat .nil a b))) ::
        .assign .ptr [] "output_ptr" (.alg
          (.ptrAdd .nil (.ptrBase output) (.mul .nat .nil a b))) :: rest) t =
      run (engine M) (initM :: initZ ::
        .assign .ptr [] "input_ptr" (.alg
          (.ptrAdd .nil (.ptrBase input) (.mul .nat .nil b a))) ::
        .assign .ptr [] "output_ptr" (.alg
          (.ptrAdd .nil (.ptrBase output) (.mul .nat .nil b a))) :: rest) t := by
    simp only [run, step_pointer_comm M _ _ _ a b]
  exact contextual_else_unconditional M s before after condition otherwise _ _ hb

end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.FPProofs
