import VeriTile.Triton
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.FastRmsLayernorm.RealMath
open VeriTile.Triton
noncomputable section

def padded {B : Nat} (N : Nat) (xs : Fin B → ℝ) (j : Fin B) :=
  if j.val<N then xs j else 0

def squareSum {B : Nat} (N : Nat) (xs : Fin B → ℝ) := ∑ j, (padded N xs j)^2

def meanSquare {B : Nat} (N : Nat) (xs : Fin B → ℝ) := squareSum N xs/(N : ℝ)

def invStd {B : Nat} (N : Nat) (eps : ℝ) (xs : Fin B → ℝ) :=
  1/Real.sqrt (meanSquare N xs+eps)

def output {B : Nat} (gemma : Bool) (N : Nat) (eps : ℝ) (xs ws : Fin B → ℝ) (j : Fin B) :=
  (xs j*invStd N eps xs)*(if gemma then ws j+1 else ws j)

/-- Backward consumes caller statistics, and always reads weights contiguously. -/
def gradient {B : Nat} (gemma : Bool) (N : Nat) (inv : ℝ) (xs dys ws : Fin B → ℝ) (j : Fin B) :=
  let q := fun k => padded N dys k*(if gemma then padded N ws k+1 else padded N ws k)
  let z := fun k => padded N xs k*inv
  inv/(N : ℝ)*((N : ℝ)*q j-z j*(∑ k,q k*z k))

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FastRmsLayernorm.RealMath
