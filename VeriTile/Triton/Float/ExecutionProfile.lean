/- Select the floating computation precision of algorithm-typed source.
Explicit ComputeOp tags keep their own precision. This adapter changes no
numerical laws and performs no Real projection or cast erasure. -/
import VeriTile.Triton.Float.Structural

namespace VeriTile.Triton.FP.Structural

def resolvePrecision (defaultPrecision : ComputeDType) (explicit : Option ComputeDType) :
    Option ComputeDType := some (explicit.getD defaultPrecision)

/-- A source without explicit ComputeOp tags still needs an execution profile
before it can use a precision-specific admission. Select only its implicit
precision; retain casts, conversions, reductions and explicit tags. -/
def Algebra.withDefaultPrecision {α : Type} (M : Algebra α) (p : ComputeDType) : Algebra α :=
  { M with
    literal := fun q => M.literal (resolvePrecision p q)
    binary := fun q => M.binary (resolvePrecision p q)
    unary := fun q => M.unary (resolvePrecision p q)
    compareLt := fun q => M.compareLt (resolvePrecision p q)
    compareLe := fun q => M.compareLe (resolvePrecision p q)
    compareEq := fun q => M.compareEq (resolvePrecision p q)
    cast := fun q => M.cast (resolvePrecision p q)
    fromNat := fun q => M.fromNat (resolvePrecision p q)
    fromInt := fun q => M.fromInt (resolvePrecision p q)
    reduceMax := fun q => M.reduceMax (resolvePrecision p q)
    reduceSum := fun q => M.reduceSum (resolvePrecision p q) }

end VeriTile.Triton.FP.Structural
