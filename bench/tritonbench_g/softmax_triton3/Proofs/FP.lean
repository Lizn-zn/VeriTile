import VeriTile.Triton.Float.RewriteTactics

namespace VeriTile.Bench.TritonBenchG.SoftmaxTriton3.FPProof
open VeriTile Triton FP.Structural FP.Guarded FP.ProfiledRewrite

/-- Lift the existing fp32 div/reciprocal atom through an explicitly fp32 scalar
normalization. Guards inspect the actual numerator and denominator registers. -/
@[spec_rule] theorem contextual_fp32_div_scalar_mul_rcp {α : Type} [Inhabited α]
    (R : FP.ScalarArithmetic.Rules) (M : Algebra α) (D : Domain α)
    (hM : Models R.assumptions M D) (s : State α) (before after : List ComputeStmt)
    (out numerator denominator : RegName)
    (n : Nat)
    (hd : (Site.divScalar before n [] (.ref .real [n] numerator) (.ref .real [] denominator)).Holds M D s) :
    Contextual M s before after
      (.assign .real (n :: []) out (.compute (.alg .fp32 (.div .real .scalarR
        (.ref .real [n] numerator) (.ref .real [] denominator)))))
      (.assign .real (n :: []) out
        (.compute (.alg .fp32 (.mul .real .scalarR (.ref .real [n] numerator)
          (.div .real .nil (.const 1) (.ref .real [] denominator)))))) := by
  let a : Op .real [n] := .ref .real [n] numerator
  let b : Op .real [] := .ref .real [] denominator
  unfold Contextual
  rw [run_append, run_append]
  cases hp : run (engine M) before s with
  | none => rfl
  | some t =>
    simp only [Option.bind_some, run, step, FP.Structural.evalExpr, evalComputeOp]
    have ediv : FP.Structural.evalOp (engine M) (some .fp32) (.div .real .scalarR a b) t =
        (FP.Structural.evalOp (engine M) (some .fp32) a t >>= fun va =>
          FP.Structural.evalOp (engine M) (some .fp32) b t >>= fun vb =>
            some (fun i => (engine M).binary (some .fp32) .real .div (va i) (vb PUnit.unit))) := rfl
    have emul : FP.Structural.evalOp (engine M) (some .fp32)
        (.mul .real .scalarR a (.div .real .nil (.const 1) b)) t =
        (FP.Structural.evalOp (engine M) (some .fp32) a t >>= fun va =>
          FP.Structural.evalOp (engine M) (some .fp32) b t >>= fun vb =>
            some (fun i => (engine M).binary (some .fp32) .real .mul (va i)
              ((engine M).binary (some .fp32) .real .div ((engine M).literal (some .fp32) .real 1)
                (vb PUnit.unit)))) := by
      change (FP.Structural.evalOp (engine M) (some .fp32) a t >>= fun va =>
        (FP.Structural.evalOp (engine M) (some .fp32) b t >>= fun vb =>
          some (fun _ : TileIndex [] => (engine M).binary (some .fp32) .real .div
            ((engine M).literal (some .fp32) .real 1) (vb PUnit.unit))) >>= fun inv =>
          some (fun i => (engine M).binary (some .fp32) .real .mul (va i) (inv PUnit.unit))) = _
      cases FP.Structural.evalOp (engine M) (some .fp32) a t <;>
        cases FP.Structural.evalOp (engine M) (some .fp32) b t <;> rfl
    erw [ediv, emul]
    cases ha : FP.Structural.evalOp (engine M) (some .fp32) a t <;>
      cases hb : FP.Structural.evalOp (engine M) (some .fp32) b t
    all_goals simp only [bind, pure, Option.bind_none, Option.bind_some]
    rename_i va vb
    obtain ⟨hva, hvb, hn⟩ := hd t hp va vb ha hb
    congr 2
    funext i
    exact FP.ScalarArithmetic.div_mul_rcp R M D hM t _ _ (hva i) hvb hn


end VeriTile.Bench.TritonBenchG.SoftmaxTriton3.FPProof
