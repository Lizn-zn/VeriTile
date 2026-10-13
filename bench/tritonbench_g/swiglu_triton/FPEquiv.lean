import bench.tritonbench_g.swiglu_triton.Proofs.Execution
import bench.tritonbench_g.swiglu_triton.Kernels
import bench.tritonbench_g.geglu_tanh_triton.Proofs.FP

namespace VeriTile.Bench.TritonBenchG.SwigluTriton.FPEquiv
open VeriTile Triton FP.ProfiledRewrite FP.Structural
open GegluTanhTriton.FPProof
open scoped VeriTile.Spec
set_option maxHeartbeats 8000000
set_option maxRecDepth 10000

def assumptions : Spec.Assumptions FP.GuardedFragment := []

def forward (storage partner : Storage) (commuted : Bool)
    (a b c : RegionName) (stride n_cols BLOCK_SIZE : Nat) : Program where
  kernel := typedFwd storage partner commuted a b c stride n_cols BLOCK_SIZE
  domain := []

def backward (storage partner gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : Program where
  kernel := typedBwd storage partner gradient commuted dc a b stride n_cols BLOCK_SIZE
  domain := []

theorem forward_decomposition (storage partner : Storage) (v : Bool)
    (a b c : RegionName) (stride n_cols BLOCK_SIZE : Nat) :
    (typedFwd storage partner v a b c stride n_cols BLOCK_SIZE).surfaceBody =
      sourcePrefix v a b c "a_ptr" "b_ptr" "c_ptr" stride ++
        (typedFwd storage partner Bool.false a b c stride n_cols BLOCK_SIZE).surfaceBody.drop 4 := by
  cases storage <;> cases partner <;> cases v <;> rfl

theorem backward_decomposition (storage partner gradient : Storage) (v : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) :
    (typedBwd storage partner gradient v dc a b stride n_cols BLOCK_SIZE).surfaceBody =
      sourcePrefix v dc a b "dc_ptr" "a_ptr" "b_ptr" stride ++
        (typedBwd storage partner gradient Bool.false dc a b stride n_cols BLOCK_SIZE).surfaceBody.drop 4 := by
  cases storage <;> cases partner <;> cases gradient <;> cases v <;> rfl

/-- Full forward state/failure equivalence. Only signed row-base products commute;
all floating operations and conversions remain in the source order. -/
specification forward_index_equiv (storage partner : Storage)
    (a b c : RegionName) (stride n_cols BLOCK_SIZE : Nat) :
    forward storage partner Bool.false a b c stride n_cols BLOCK_SIZE ≡[assumptions]
      forward storage partner Bool.true a b c stride n_cols BLOCK_SIZE := by
  apply Spec.FloatingPoint.ofNumerical (structural := fun _ _ => False) rfl
    (by cases storage <;> cases partner <;> rfl)
  intro α _ M D _ s _
  change run (engine M)
      (typedFwd storage partner Bool.false a b c stride n_cols BLOCK_SIZE).surfaceBody s =
    run (engine M)
      (typedFwd storage partner Bool.true a b c stride n_cols BLOCK_SIZE).surfaceBody s
  rw [forward_decomposition storage partner Bool.false a b c stride n_cols BLOCK_SIZE,
    forward_decomposition storage partner Bool.true a b c stride n_cols BLOCK_SIZE]
  exact prefix_run M s a b c "a_ptr" "b_ptr" "c_ptr" stride _

/-- Full backward state/failure equivalence, including both ordered in-place
stores and every input dtype combination, without a numerical-domain premise. -/
specification backward_index_equiv (storage partner gradient : Storage)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) :
    backward storage partner gradient Bool.false dc a b stride n_cols BLOCK_SIZE ≡[assumptions]
      backward storage partner gradient Bool.true dc a b stride n_cols BLOCK_SIZE := by
  apply Spec.FloatingPoint.ofNumerical (structural := fun _ _ => False) rfl
    (by cases storage <;> cases partner <;> cases gradient <;> rfl)
  intro α _ M D _ s _
  change run (engine M)
      (typedBwd storage partner gradient Bool.false dc a b stride n_cols BLOCK_SIZE).surfaceBody s =
    run (engine M)
      (typedBwd storage partner gradient Bool.true dc a b stride n_cols BLOCK_SIZE).surfaceBody s
  rw [backward_decomposition storage partner gradient Bool.false dc a b stride n_cols BLOCK_SIZE,
    backward_decomposition storage partner gradient Bool.true dc a b stride n_cols BLOCK_SIZE]
  exact prefix_run M s dc a b "dc_ptr" "a_ptr" "b_ptr" stride _

#print_fp_assumptions forward_index_equiv
#print_fp_assumptions backward_index_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SwigluTriton.FPEquiv
