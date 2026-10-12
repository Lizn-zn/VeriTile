import bench.examples.FusedLayerNorm.Kernels
/- Original online-statistics versus two-pass LayerNorm. The shared affine
suffix retains its sqrt, epsilon, gamma/beta and bf16 cast. Only the admitted
scalar arithmetic and bounded count atoms are used. Empty output rows remain
covered because both sources preserve every memory cell at N = 0. -/
import bench.examples.FusedLayerNorm.Proofs.FP
import VeriTile.Triton.Float.CountConversionLaws
import VeriTile.Triton.Float.Kernel
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Examples.FusedLayerNormFPEquiv
open VeriTile.Bench.Examples.FusedLayerNorm.Kernels
open VeriTile Triton FP.Structural FP.CountConversion
open LayerNormFPContract LayerNormFPExecution
open _root_.VeriTile.Triton.FP.Equational (ReductionPlan)
open scoped VeriTile.Spec
open _root_.VeriTile.Triton.FP.Equational (Schedules)

/-- Declare the three public input ports once each. The DSL's metadata lists
each antiquoted load occurrence, including the fused kernel's second x read;
the explicit IO declaration preserves the entire original statement body. -/
def twopassIO (x g b y : RegionName) (N stride : Nat) (ε : ℝ) : KernelIO₃ where
  kernel := .mk [x, g, b] [y] (twoPassLayerNormKernel x g b y N stride ε).surfaceBody
  in1 := x
  in2 := g
  in3 := b
  out := y
  B1 := N
  B2 := N
  B3 := N
  Bout := N
  read1 := fun pid => pid * stride
  read2 := fun _ => 0
  read3 := fun _ => 0
  write := fun pid => pid * stride

def fusedIO (x g b y : RegionName) (N stride : Nat) (ε : ℝ) : KernelIO₃ :=
  (twopassIO x g b y N stride ε).withKernel
    (.mk [x, g, b] [y] (fusedLayerNormKernel x g b y N stride ε).surfaceBody)

/-- Nonempty rows use exactly Welford's compiled statistics domain. Empty
rows have no numerical output and require no domain checks. -/
def requirements (x : RegionName) (N stride : Nat) (empty : ReductionPlan 0)
    (plans : Schedules) : FP.GuardExpression.Condition :=
  if N = 0 then .top else WelfordFPContract.requirements x N stride empty plans

def twopass (x g b y : RegionName) (N stride : Nat) (ε : ℝ) (empty : ReductionPlan 0) :
    FP.Kernel₃ where
  io := twopassIO x g b y N stride ε
  profile := FP.fp32
  domain := requirements x N stride empty

def fused (x g b y : RegionName) (N stride : Nat) (ε : ℝ) (empty : ReductionPlan 0) :
    FP.Kernel₃ :=
  { twopass x g b y N stride ε empty with io := fusedIO x g b y N stride ε }

specification layernorm_equiv (N stride : Nat) (hBound : N ≤ limit) (ε : ℝ)
    (empty : ReductionPlan 0) (R : Rules) :
    fused "x" "gamma" "beta" "y" N stride ε empty ≡[R]
      twopass "x" "gamma" "beta" "y" N stride ε empty := by
  apply Spec.FloatingPoint.ofNumerical
    (lhs := fused "x" "gamma" "beta" "y" N stride ε empty)
    (rhs := twopass "x" "gamma" "beta" "y" N stride ε empty)
    (structural := fun _ _ => False) rfl rfl
  exact LayerNormFPContract.equivalent N stride hBound ε empty R

#print_fp_assumptions layernorm_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Examples.FusedLayerNormFPEquiv
