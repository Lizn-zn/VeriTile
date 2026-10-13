import VeriTile.Triton.Float.ExpressionRewrite
import VeriTile.Triton.Float.RewriteTactics
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.GegluTanhTriton.FPProof
open VeriTile.Triton FP.Structural FP.ProfiledRewrite

@[spec_rule] theorem signed_index_product_commutes {α : Type} [Inhabited α] (M : Algebra α)
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



/-- Signed row-base commutation preserves conversion to a natural pointer offset,
including negative intermediate values and failed operand evaluation. -/
@[spec_rule] theorem signed_ptr_index_commutes {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) (before : List ComputeStmt) (r : RegionName) (a b : Op .int []) :
    OpEquivalent M s before
      (.ptrAdd .nil (.ptrBase r) (.castIntToNat (.mul .int .nil a b)))
      (.ptrAdd .nil (.ptrBase r) (.castIntToNat (.mul .int .nil b a))) := by
  intro t ht
  have h := signed_index_product_commutes M s before a b t ht
  change ((evalOp (engine M) none (.mul .int .nil a b) t >>= fun v =>
    some (fun i => (v i).toNat)) >>= fun y =>
    some (bop (fun (p : RegionName × Nat) (n : Nat) => (p.1,p.2+n))
      .nil (fun _ => (r,0)) y)) = _
  rw [h]
  rfl

def base (v : Bool) (r : RegionName) (name : RegName) (stride : Nat) : ComputeStmt :=
  let a : Op .int [] := .ref .int [] "program_id"
  let b : Op .int [] := .castNatToInt (.constNat stride)
  .assign .ptr [] name (.alg (.ptrAdd .nil (.ptrBase r)
    (.castIntToNat (if v then .mul .int .nil b a else .mul .int .nil a b))))

def sourcePrefix (v : Bool) (r₁ r₂ r₃ : RegionName) (n₁ n₂ n₃ : RegName)
    (stride : Nat) : List ComputeStmt :=
  [.assign .int [] "program_id" (.alg (.castNatToInt (.programId 0))),
   base v r₁ n₁ stride, base v r₂ n₂ stride, base v r₃ n₃ stride]

theorem base_step {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (r : RegionName) (name : RegName) (stride : Nat) :
    step (engine M) (base Bool.false r name stride) s =
      step (engine M) (base Bool.true r name stride) s := by
  have h := signed_ptr_index_commutes M s [] r
    (.ref .int [] "program_id") (.castNatToInt (.constNat stride)) s (by simp [run])
  simpa only [base, Bool.false_eq_true, ↓reduceIte, step, FP.Structural.evalExpr] using
    congrArg (fun x => x >>= fun value => some (s.setReg name .ptr [] value)) h

/-- A single proof for the common four-statement prefix, independent of every
floating operation in the unchanged suffix. -/
theorem prefix_run {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (r₁ r₂ r₃ : RegionName) (n₁ n₂ n₃ : RegName) (stride : Nat)
    (after : List ComputeStmt) :
    run (engine M) (sourcePrefix Bool.false r₁ r₂ r₃ n₁ n₂ n₃ stride ++ after) s =
      run (engine M) (sourcePrefix Bool.true r₁ r₂ r₃ n₁ n₂ n₃ stride ++ after) s := by
  rw [run_append, run_append]
  apply congrArg (fun (result : Option (State α)) => result.bind (run (engine M) after))
  simp only [sourcePrefix, run, base_step]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.GegluTanhTriton.FPProof
