import bench.tritonbench_g.bgmv_expand_slice.Kernels
import VeriTile.Triton.Float.RewriteTactics
import VeriTile.Meta.StatementAudit

/-! Supported output-fp32 specialization of the original output-first candidate.
Every input/weight/cast choice retains its actual product and reduction dtype;
only the final addition is promoted to fp32 by the fp32 output operand.
The domain checks the actual promoted operands in each reached accumulation arm.
This does not certify the pending arbitrary-output-dtype theorem in FP.lean. -/
namespace VeriTile.Bench.TritonBenchG.BgmvExpandSlice.FP32Proofs
open VeriTile Triton FP FP.Structural FP.ProfiledRewrite FP.Guarded
open _root_.VeriTile.Triton.FP.GuardedRewrite (sameShape sameShape_left sameShape_right)
open scoped VeriTile.Spec
set_option maxHeartbeats 2200000
set_option maxRecDepth 10000

/-- Extract exactly the scalar-addition site supported by the admitted rule.
The symbolic source check below rejects missing sites in every specialization.
Both operands keep the precision context of the actual assignment. -/
def additionOperands? : {dtype : TileDType} → {shape : TileShape} →
    Op dtype shape → Option (Op .real shape × Op .real shape)
  | _, _, .add .real (.consSame .nil) left right => some (left, right)
  | _, _, .castFloat _ _ operand => additionOperands? operand
  | _, _, _ => none

def additionSite? (code : List ComputeStmt) : Option Site :=
  let before := Source.prefixBefore code "accumulator"
  match (code.drop before.length).head? with
  | some (.assign _ shape _ (.compute (.alg .fp32 operand))) =>
      (additionOperands? operand).map (fun (left, right) => .addFP32 before shape left right)
  | _ => none

def requirementsFor (kernel : ComputeKernel) : List Site :=
  (do
    let active ← Source.conditional? kernel.surfaceBody
      (fun body => (Source.dynamicRange? body "n").isSome)
    let loop ← Source.dynamicRange? active.yes "n"
    let accumulation ← Source.conditional? loop.body (fun _ => Bool.true)
    let addition ← additionSite? accumulation.yes
    return Site.branch active.before active.condition Bool.true
      (.range loop.before "n" loop.start loop.stop loop.stride loop.body
        (.branch accumulation.before accumulation.condition Bool.true addition))).toList

/-- Preserve the explicit output conversion around the admitted fp32 rewrite. -/
@[spec_rule] theorem contextual_cast_add_fp32_commute {α : Type} [Inhabited α]
    (R : ScalarArithmetic.Rules) (M : Algebra α) (D : Domain α)
    (hM : Models R.assumptions M D) (s : State α) (before after : List ComputeStmt)
    (out : RegName) (a b : Op .real shape)
    (hd : (Site.addFP32 before shape a b).Holds M D s) :
    Contextual M s before after
      (.assign .real shape out
        (.compute (.alg .fp32 (.castFloat .real .real (.add .real (sameShape shape) a b)))))
      (.assign .real shape out
        (.compute (.alg .fp32 (.castFloat .real .real (.add .real (sameShape shape) b a))))) := by
  unfold Contextual
  rw [run_append, run_append]
  cases hp : run (engine M) before s with
  | none => rfl
  | some t =>
    simp only [Option.bind_some, run, step, Structural.evalExpr, evalComputeOp]
    have ecast (c : Op .real shape) :
        Structural.evalOp (engine M) (some .fp32) (.castFloat .real .real c) t =
          (Structural.evalOp (engine M) (some .fp32) c t).bind
            (fun v => some (fun i => (engine M).cast (some .fp32) .real .real (v i))) := rfl
    erw [ecast, ecast]
    have eadd (x y : Op .real shape) :
        Structural.evalOp (engine M) (some .fp32) (.add .real (sameShape shape) x y) t =
          ((Structural.evalOp (engine M) (some .fp32) x t).bind fun vx =>
            (Structural.evalOp (engine M) (some .fp32) y t).bind fun vy =>
              some (bop ((engine M).binary (some .fp32) .real .add) (sameShape shape) vx vy)) := rfl
    erw [eadd, eadd]
    cases ha : Structural.evalOp (engine M) (some .fp32) a t <;>
      cases hb : Structural.evalOp (engine M) (some .fp32) b t
    all_goals simp only [Option.bind_none, Option.bind_some]
    rename_i va vb
    have hvalues : bop ((engine M).binary (some .fp32) .real .add) (sameShape shape) va vb =
        bop ((engine M).binary (some .fp32) .real .add) (sameShape shape) vb va := by
      funext i
      simp only [bop, sameShape_left, sameShape_right]
      exact ScalarArithmetic.add_comm R M D hM t _ _ ((hd t hp va vb ha hb).1 i)
        ((hd t hp va vb ha hb).2 i)
    rw [hvalues]

section
variable (input : TypedSource.InputStorage) (weight : TypedSource.WeightStorage)
    (CAST_TYPE : Bool)
    (x w out : RegionName) (N K : Nat) (indices : Region .int)
    (sx sk sw0 swn swk sy syn slice BN BK splits : Nat) (EVEN_K ADD_INPUTS : Bool)

def original : Program where
  kernel := TypedSource.original input weight CAST_TYPE .fp32 x w out N K indices
    sx sk sw0 swn swk sy syn slice BN BK splits EVEN_K ADD_INPUTS
  domain := requirementsFor (TypedSource.original input weight CAST_TYPE .fp32 x w out N K indices
    sx sk sw0 swn swk sy syn slice BN BK splits EVEN_K ADD_INPUTS)

def outputFirst : Program :=
  { original input weight CAST_TYPE x w out N K indices
      sx sk sw0 swn swk sy syn slice BN BK splits EVEN_K ADD_INPUTS with
    kernel := TypedSource.outputFirst input weight CAST_TYPE .fp32 x w out N K indices
      sx sk sw0 swn swk sy syn slice BN BK splits EVEN_K ADD_INPUTS }

/-- A failed source locator cannot silently remove the required domain. -/
theorem requirements_present :
    (original input weight CAST_TYPE x w out N K indices
      sx sk sw0 swn swk sy syn slice BN BK splits EVEN_K ADD_INPUTS).domain.length = 1 := by
  cases input <;> cases weight <;> cases CAST_TYPE <;> rfl

theorem output_first_equiv (R : FP.ScalarArithmetic.Rules) :
    original input weight CAST_TYPE x w out N K indices
      sx sk sw0 swn swk sy syn slice BN BK splits EVEN_K ADD_INPUTS ≡[R]
    outputFirst input weight CAST_TYPE x w out N K indices
      sx sk sw0 swn swk sy syn slice BN BK splits EVEN_K ADD_INPUTS := by
  cases input <;> cases weight <;> cases CAST_TYPE
  all_goals equiv_decompose
  all_goals fp_prove

end
#print_fp_assumptions output_first_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.BgmvExpandSlice.FP32Proofs
