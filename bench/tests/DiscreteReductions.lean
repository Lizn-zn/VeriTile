import VeriTile.Triton.DSL
import VeriTile.Triton.Float.Structural
import VeriTile.Triton.Float.EvalOpR
import VeriTile.Meta.StatementAudit

/-! Discrete maximum must execute in both interpreters. In particular, signed
metadata is not clipped to zero or delegated to an opaque floating operation. -/
namespace DiscreteReductionTests
open VeriTile.Triton TileShape
noncomputable section

theorem natural_compatibility {shape : TileShape} (axisIdx : Fin shape.length)
    (keepDims : Bool) (x : Tile .nat shape) :
    (discreteMaximum axisIdx keepDims x.data).map Tile.mk = Tile.reduceMaxNat axisIdx keepDims x := by
  cases keepDims <;>
    dsimp [discreteMaximum, Tile.reduceMaxNat, Tile.reduceMaxNatDrop, Tile.reduceMaxNatKeep] <;>
    split_ifs <;> rfl

theorem signed_empty :
    discreteMaximum (shape := [0]) (β := Int) ⟨0,by simp⟩ Bool.false
      (fun i => Fin.elim0 i.1) = none := by
  simp [discreteMaximum, axisDim]

theorem signed_negative :
    discreteMaximum (shape := [2]) (β := Int) ⟨0,by simp⟩ Bool.false
      (fun i => if i.1.val = 0 then -7 else -2) = some (fun _ => -2) := by
  simp only [discreteMaximum, axisDim, show 0 < (2 : Nat) by decide, dif_pos, Option.some.injEq]
  funext i
  change (Finset.univ : Finset (Fin 2)).sup' _
    (fun k => if k.val = 0 then (-7 : Int) else -2) = (-2 : Int)
  norm_num [Finset.univ_fin2]

theorem signed_negative_keep :
    discreteMaximum (shape := [2]) (β := Int) ⟨0,by simp⟩ Bool.true
      (fun i => if i.1.val = 0 then -7 else -2) = some (fun _ => -2) := by
  simp only [discreteMaximum, axisDim, show 0 < (2 : Nat) by decide, dif_pos, Option.some.injEq]
  funext i
  change (Finset.univ : Finset (Fin 2)).sup' _
    (fun k => if k.val = 0 then (-7 : Int) else -2) = (-2 : Int)
  norm_num [Finset.univ_fin2]

/-- An axis inside a multidimensional tile retains the other coordinates. -/
theorem second_axis :
    discreteMaximum (shape := [3,2]) (β := Int) ⟨1,by simp⟩ Bool.false
      (fun i => (i.1.val : Int) + if i.2.1.val = 0 then -7 else -2) =
      some (fun i => (i.1.val : Int) - 2) := by
  simp only [discreteMaximum, axisDim, show 0 < (2 : Nat) by decide, dif_pos, Option.some.injEq]
  funext i
  change (Finset.univ : Finset (Fin 2)).sup' _
    (fun k => (i.1.val : Int) + if k.val = 0 then -7 else -2) = _
  norm_num [Finset.univ_fin2]
  omega

def signedSource (metadata : Region .int) : ComputeKernel := triton {
  values = tl.load(metadata + tl.arange(0, $(2)))
  maximum = tl.max(values)
}

theorem signed_source_projection (metadata : Region .int) :
    (signedSource metadata).surfaceBody.getLast? =
      some (.assign .int [] "maximum" (.alg
        (.reduceMaxInt ⟨0, by simp⟩ Bool.false (.ref .int [2] "values")))) := rfl

/-- These evaluations work for every floating algebra, including ones with no
floating comparison support. The input is an actual loaded/register value. -/
theorem signed_fp_execution {α : Type} [Inhabited α]
    (M : FP.Structural.Algebra α) (s : FP.Structural.State α) :
    FP.Structural.evalOp M none
      (.reduceMaxInt ⟨0, by simp⟩ Bool.false (.ref .int [2] "values"))
      (s.setReg "values" .int [2] (fun i => if i.1.val = 0 then -7 else -2)) =
      some (fun _ => -2) := by
  simp only [FP.Structural.evalOp, FP.Structural.State.setReg_same]
  exact signed_negative

theorem natural_fp_execution {α : Type} [Inhabited α]
    (M : FP.Structural.Algebra α) (s : FP.Structural.State α) :
    FP.Structural.evalOp M none
      (.reduceMaxNat ⟨0, by simp⟩ Bool.false (.arange 2)) s = some (fun _ => 1) := by
  change discreteMaximum (shape := [2]) ⟨0, by simp⟩ Bool.false
    (fun i => i.1.val) = some (fun _ => 1)
  simp only [discreteMaximum, axisDim,
    show 0 < (2 : Nat) by decide, dif_pos, Option.some.injEq]
  funext i
  change (Finset.univ : Finset (Fin 2)).sup' _ (fun k => k.val) = 1
  norm_num [Finset.univ_fin2]

theorem missing_operand_fails {α : Type} [Inhabited α]
    (M : FP.Structural.Algebra α) (s : FP.Structural.State α)
    (h : s.regs .int [2] "missing" = none) :
    FP.Structural.evalOp M none
      (.reduceMaxInt ⟨0, by simp⟩ Bool.false (.ref .int [2] "missing")) s = none := by
  simp [FP.Structural.evalOp, h]

/-- Real and rounded execution use the same signed maximum after their operand
has evaluated. This covers every axis and both keep_dims choices. -/
theorem signed_real_evaluation {shape : TileShape} (axisIdx : Fin shape.length)
    (keepDims : Bool) (e : Op .int shape) (s : BlockState) (v : Tile .int shape)
    (h : evalOp e s = some v) :
    evalOp (.reduceMaxInt axisIdx keepDims e) s =
      (discreteMaximum axisIdx keepDims v.data).map Tile.mk := by
  simp [evalOp, h, Tile.reduceMaxInt]

theorem signed_rounded_evaluation {shape : TileShape} (axisIdx : Fin shape.length)
    (keepDims : Bool) (R : RoundingModel) (e : Op .int shape)
    (s : BlockState) (v : Tile .int shape) (h : evalOpR R e s = some v) :
    evalOpR R (.reduceMaxInt axisIdx keepDims e) s =
      (discreteMaximum axisIdx keepDims v.data).map Tile.mk := by
  simp [evalOpR, h, Tile.reduceMaxInt]

def signedInitial {α : Type} [Inhabited α] : FP.Structural.State α where
  mem := fun _ offset => .mk .int (if offset = 0 then -7 else -2)
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => FP.Structural.defaultValue d

/-- The entire source executes: it loads signed cells, reduces them and leaves
memory unchanged. This rules out equality of two failed executions. -/
theorem signed_source_executes {α : Type} [Inhabited α]
    (M : FP.Structural.Algebra α) :
    FP.Structural.exec M (signedSource ⟨"metadata"⟩) signedInitial =
      some (((signedInitial (α := α)).setReg "values" .int [2]
        (fun i => if i.1.val = 0 then -7 else -2)).setReg
          "maximum" .int [] (fun _ => -2)) := by
  simp [signedSource, FP.Structural.exec, FP.Structural.run, FP.Structural.step,
    FP.Structural.evalExpr, FP.Structural.evalOp, Option.bind, signedInitial,
    Region.cast, FP.Structural.Cell.read, discreteMaximum, axisDim,
    insertAxisIndex]
  congr 1

#guard_msgs (drop info) in
#auditModuleAxioms
end
end DiscreteReductionTests
