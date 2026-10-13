import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.FP64ConversionBackend
open VeriTile.Triton FP.Structural

/-- A casted real-carrier expression invokes conversion at the selected fp32
profile. The backend's fp32-load hook is not a replacement for this event. -/
example {α : Type} [Inhabited α] (M : Algebra α)
    (e : Op .real shape) (s : State α) :
    evalComputeOp M (.alg .fp32 (.castFloat .real .real e)) s =
      (do let v ← evalOp M (some .fp32) e s
          pure (fun i => M.cast (some .fp32) .real .real (v i))) := by
  simp only [evalComputeOp, evalOp]
  rfl

/-- Widening the already-computed mean selects the fp64 conversion profile. -/
example {α : Type} [Inhabited α] (M : Algebra α)
    (e : Op .real shape) (s : State α) :
    evalComputeOp M (.alg .fp64 (.castFloat .real .real e)) s =
      (do let v ← evalOp M (some .fp64) e s
          pure (fun i => M.cast (some .fp64) .real .real (v i))) := by
  simp only [evalComputeOp, evalOp]
  rfl

/-- A genuine fp32 load instead uses its distinct load hook. No equality
between the two backend operations is assumed by these regressions. -/
example {α : Type} [Inhabited α] (M : Algebra α)
    (mem : MemAccess .real shape) (mask : MaskOpt .real shape) (s : State α) :
    evalComputeOp M (.load .fp32 mem mask) s =
      (do let v ← evalOp M (some .fp32) (.load .real mem mask) s
          pure (fun i => M.fp32Load (v i))) := rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Tests.FP64ConversionBackend
