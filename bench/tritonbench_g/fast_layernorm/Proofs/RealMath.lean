import VeriTile.Triton
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.FastLayernorm.RealMath
open VeriTile.Triton
noncomputable section

def padded {B : Nat} (N : Nat) (xs : Fin B → ℝ) (j : Fin B) :=
  if j.val<N then xs j else 0

def mean {B : Nat} (N : Nat) (xs : Fin B → ℝ) := (∑ j, padded N xs j)/(N : ℝ)

/-- Every lane is centered, including the initially zero-padded lanes. -/
def variance {B : Nat} (N : Nat) (xs : Fin B → ℝ) :=
  (∑ j, (padded N xs j-mean N xs)^2)/(N : ℝ)

def invStd {B : Nat} (N : Nat) (eps : ℝ) (xs : Fin B → ℝ) :=
  1/Real.sqrt (variance N xs+eps)

def output {B : Nat} (N : Nat) (eps : ℝ) (xs ws bs : Fin B → ℝ) (j : Fin B) :=
  ((xs j-mean N xs)*invStd N eps xs)*ws j+bs j

/-- The backward body reads these statistics from the caller. Bias is loaded
but is not an operand of the returned gradient. -/
def gradient {B : Nat} (N : Nat) (inv avg : ℝ) (xs dys ws : Fin B → ℝ) (j : Fin B) :=
  let q := fun k => padded N dys k*padded N ws k
  let normed := fun k => (padded N xs k-avg)*inv
  (q j-(∑ k, q k)/(N : ℝ)-normed j*(∑ k, q k*normed k)/(N : ℝ))*inv

/-- A real formula sanity check showing centered padding enters variance. -/
theorem padded_variance_example : variance 1 (fun _ : Fin 2 => (1 : ℝ))=1 := by
  norm_num [variance,mean,padded,Fin.sum_univ_succ]

/-- The host chooses block size four for a three-element row. -/
theorem host_padding_example :
    mean 3 (fun _ : Fin 4 => (3 : ℝ))=3 ∧ variance 3 (fun _ : Fin 4 => (3 : ℝ))=3 := by
  norm_num [variance,mean,padded,Fin.sum_univ_succ]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FastLayernorm.RealMath
