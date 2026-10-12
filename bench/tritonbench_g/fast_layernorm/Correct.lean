import bench.tritonbench_g.fast_layernorm.Proofs.ForwardIO
import bench.tritonbench_g.fast_layernorm.Proofs.BackwardIO
import bench.tritonbench_g.fast_layernorm.Proofs.Counterexamples
/-! Whole-source finite-flat contracts. Forward centers the padded block and
therefore includes padded lanes in variance. Backward consumes caller-provided
statistics; this is a source formula theorem, not a calculus or host-composition
claim. Program scheduling, races, fixed-width overflow and physical rounding
are outside these per-program contracts. -/
namespace VeriTile.Bench.TritonBenchG.FastLayernorm.Correctness
open VeriTile VeriTile.Triton TypedSource
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


structure ForwardBounds (Y X W b r mu : RegionName) (YR XR N B : Nat)
    (u : BlockState) (bounds : RegionBounds) : Prop where
  input : ∀ j : Fin B, j.val<N → u.pids 0*XR+j.val<bounds X
  weight : ∀ j : Fin B, j.val<N → j.val<bounds W
  bias : ∀ j : Fin B, j.val<N → j.val<bounds b
  output : ∀ j : Fin B, j.val<N → u.pids 0*YR+j.val<bounds Y
  inv : u.pids 0<bounds r
  mean : u.pids 0<bounds mu

def forwardOutside (A : FlatAlloc) (u : BlockState) (Y r mu : RegionName) (YR N B : Nat)
    (q : RegionName) (o : Nat) : Prop :=
  q≠A.flat ∨ ((∀ j : Fin B,j.val<N → o≠A.addr Y (u.pids 0*YR+j.val)) ∧
    o≠A.addr r (u.pids 0) ∧ o≠A.addr mu (u.pids 0))

def forward_correct_target (kernel : Kernel) (Y X W b r mu : RegionName) (YR XR N B : Nat) (eps : ℝ) : Prop :=
  ∀ A : FlatAlloc, A.Disjoint → (∀ q,q∉A.regions → A.extent q=0) →
  ∀ u : BlockState, u.undef=(fun _ _ => 0) → ForwardBounds Y X W b r mu YR XR N B u A.extent →
  0<N → N≤B → Y≠r → Y≠mu → r≠mu →
  0<variance N (fun j : Fin B => u.readMem X (u.pids 0*XR+j.val))+eps →
    ∃ v,exec (A.flattenKernel kernel) (A.flattenState u)=some v ∧
      (∀ j : Fin B,j.val<N → v.readMem A.flat (A.addr Y (u.pids 0*YR+j.val))=
        output N eps (fun j : Fin B => u.readMem X (u.pids 0*XR+j.val))
          (fun j : Fin B => u.readMem W j.val) (fun j : Fin B => u.readMem b j.val) j) ∧
      v.readMem A.flat (A.addr r (u.pids 0))=
        invStd N eps (fun j : Fin B => u.readMem X (u.pids 0*XR+j.val)) ∧
      v.readMem A.flat (A.addr mu (u.pids 0))=
        mean N (fun j : Fin B => u.readMem X (u.pids 0*XR+j.val)) ∧
      (∀ q o,forwardOutside A u Y r mu YR N B q o → v.mem q o=(A.flattenState u).mem q o)

specification forward_correctness (input weight bias : Storage) (Y X W b r mu : RegionName)
    (YR XR N B : Nat) (eps : ℝ) :
    Spec.Real (forward_correct_target
      (forward input weight bias Bool.false Y X W b r mu YR XR N B eps).eraseDType.toAlgKernel
      Y X W b r mu YR XR N B eps) := by
  intro A hd hc u hu hb hN hB hYr hYmu hRmu hpos
  have hb' : ForwardIO.Bounds Y X W b r mu YR XR N B u A.extent :=
    ⟨hb.input,hb.weight,hb.bias,hb.output,hb.inv,hb.mean⟩
  exact ForwardIO.typed_correct input weight bias Y X W b r mu YR XR N B eps
    A hd hc u hu hb' hN hB hYr hYmu hRmu hpos

specification forward_index_commuted_correctness (input weight bias : Storage) (Y X W b r mu : RegionName)
    (YR XR N B : Nat) (eps : ℝ) :
    Spec.Real (forward_correct_target
      (forward input weight bias Bool.true Y X W b r mu YR XR N B eps).eraseDType.toAlgKernel
      Y X W b r mu YR XR N B eps) := by
  intro A hd hc u hu hb hN hB hYr hYmu hRmu hpos
  have hb' : ForwardIO.Bounds Y X W b r mu YR XR N B u A.extent :=
    ⟨hb.input,hb.weight,hb.bias,hb.output,hb.inv,hb.mean⟩
  exact ForwardIO.candidate_correct input weight bias Y X W b r mu YR XR N B eps
    A hd hc u hu hb' hN hB hYr hYmu hRmu hpos

structure BackwardBounds (dY X W b r mu : RegionName) (YR XR N B : Nat)
    (u : BlockState) (bounds : RegionBounds) : Prop where
  dy : ∀ j : Fin B, j.val<N → u.pids 0*YR+j.val<bounds dY
  input : ∀ j : Fin B, j.val<N → u.pids 0*XR+j.val<bounds X
  weight : ∀ j : Fin B, j.val<N → j.val<bounds W
  bias : ∀ j : Fin B, j.val<N → j.val<bounds b
  inv : u.pids 0<bounds r
  mean : u.pids 0<bounds mu

def backwardOutside (A : FlatAlloc) (u : BlockState) (dY : RegionName) (YR N B : Nat)
    (q : RegionName) (o : Nat) : Prop :=
  q≠A.flat ∨ ∀ j : Fin B,j.val<N → o≠A.addr dY (u.pids 0*YR+j.val)

def backward_correct_target (kernel : Kernel) (dY X W b r mu : RegionName) (YR XR N B : Nat) : Prop :=
  ∀ A : FlatAlloc, A.Disjoint → (∀ q,q∉A.regions → A.extent q=0) →
  ∀ u : BlockState, u.undef=(fun _ _ => 0) → BackwardBounds dY X W b r mu YR XR N B u A.extent →
  0<N → N≤B →
    ∃ v,exec (A.flattenKernel kernel) (A.flattenState u)=some v ∧
      (∀ j : Fin B,j.val<N → v.readMem A.flat (A.addr dY (u.pids 0*YR+j.val))=
        gradient N (u.readMem r (u.pids 0)) (u.readMem mu (u.pids 0))
          (fun j : Fin B => u.readMem X (u.pids 0*XR+j.val))
          (fun j : Fin B => u.readMem dY (u.pids 0*YR+j.val))
          (fun j : Fin B => u.readMem W j.val) j) ∧
      (∀ q o,backwardOutside A u dY YR N B q o → v.mem q o=(A.flattenState u).mem q o)

specification backward_correctness (input weight bias : Storage) (dY X W b r mu : RegionName)
    (YR XR N B : Nat) (eps : ℝ) :
    Spec.Real (backward_correct_target
      (backward input weight bias Bool.false dY X W b r mu YR XR N B eps).eraseDType.toAlgKernel
      dY X W b r mu YR XR N B) := by
  intro A hd hc u hu hb hN hB
  have hb' : BackwardIO.Bounds dY X W b r mu YR XR N B u A.extent :=
    ⟨hb.dy,hb.input,hb.weight,hb.bias,hb.inv,hb.mean⟩
  exact BackwardIO.typed_correct input weight bias dY X W b r mu YR XR N B eps
    A hd hc u hu hb' hN hB

specification backward_index_commuted_correctness (input weight bias : Storage) (dY X W b r mu : RegionName)
    (YR XR N B : Nat) (eps : ℝ) :
    Spec.Real (backward_correct_target
      (backward input weight bias Bool.true dY X W b r mu YR XR N B eps).eraseDType.toAlgKernel
      dY X W b r mu YR XR N B) := by
  intro A hd hc u hu hb hN hB
  have hb' : BackwardIO.Bounds dY X W b r mu YR XR N B u A.extent :=
    ⟨hb.dy,hb.input,hb.weight,hb.bias,hb.inv,hb.mean⟩
  exact BackwardIO.candidate_correct input weight bias dY X W b r mu YR XR N B eps
    A hd hc u hu hb' hN hB

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FastLayernorm.Correctness
