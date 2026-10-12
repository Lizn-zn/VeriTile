import bench.tritonbench_g.fast_rms_layernorm.Proofs.ForwardIO
import bench.tritonbench_g.fast_rms_layernorm.Proofs.BackwardIO
/-! Complete finite-flat contracts for three original/candidate sources.
Regular forward uses W stride; Gemma forward and backward ignore it. Backward
uses caller statistics and does not access dW. No host composition, calculus,
fixed-width index, scheduling/race or physical rounding claim is implicit. -/
namespace VeriTile.Bench.TritonBenchG.FastRmsLayernorm.Correctness
open VeriTile VeriTile.Triton TypedSource
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


def weightOffset (isGemma : Bool) (WR j : Nat) := if isGemma then j else j*WR

structure ForwardBounds (isGemma : Bool) (Y X W r : RegionName) (YR XR WR RR N B : Nat)
    (u : BlockState) (bounds : RegionBounds) : Prop where
  input : ∀ j : Fin B,j.val<N → u.pids 0*XR+j.val<bounds X
  weight : ∀ j : Fin B,j.val<N → weightOffset isGemma WR j.val<bounds W
  output : ∀ j : Fin B,j.val<N → u.pids 0*YR+j.val<bounds Y
  inv : u.pids 0*RR<bounds r

def forwardOutside (A : FlatAlloc) (u : BlockState) (Y r : RegionName) (YR RR N B : Nat)
    (q : RegionName) (o : Nat) : Prop :=
  q≠A.flat ∨ ((∀ j : Fin B,j.val<N → o≠A.addr Y (u.pids 0*YR+j.val)) ∧
    o≠A.addr r (u.pids 0*RR))

def forward_correct_target (kernel : Kernel) (isGemma : Bool) (Y X W r : RegionName)
    (YR XR WR RR N B : Nat) (eps : ℝ) : Prop :=
  ∀ A : FlatAlloc,A.Disjoint → (∀ q,q∉A.regions → A.extent q=0) →
  ∀ u : BlockState,u.undef=(fun _ _ => 0) → ForwardBounds isGemma Y X W r YR XR WR RR N B u A.extent →
  0<N → N≤B → Y≠r →
  0<meanSquare N (fun j : Fin B => u.readMem X (u.pids 0*XR+j.val))+eps →
    ∃ v,exec (A.flattenKernel kernel) (A.flattenState u)=some v ∧
      (∀ j : Fin B,j.val<N → v.readMem A.flat (A.addr Y (u.pids 0*YR+j.val))=
        output isGemma N eps (fun k : Fin B => u.readMem X (u.pids 0*XR+k.val))
          (fun k : Fin B => u.readMem W (weightOffset isGemma WR k.val)) j) ∧
      v.readMem A.flat (A.addr r (u.pids 0*RR))=
        invStd N eps (fun j : Fin B => u.readMem X (u.pids 0*XR+j.val)) ∧
      (∀ q o,forwardOutside A u Y r YR RR N B q o → v.mem q o=(A.flattenState u).mem q o)

specification forward_correctness (input weight : Storage) (Y X W r : RegionName)
    (YR XR WR RR N B : Nat) (eps : ℝ) :
    Spec.Real (forward_correct_target (forward input weight Bool.false Y X W r YR XR WR RR N B eps).eraseDType.toAlgKernel
      Bool.false Y X W r YR XR WR RR N B eps) := by
  intro A hd hc u hu hb hN hB hne hpos
  have hb' : ForwardIO.Bounds Bool.false Y X W r YR XR WR RR N B u A.extent :=
    ⟨hb.input,hb.weight,hb.output,hb.inv⟩
  exact ForwardIO.typed_correct input weight Bool.false Y X W r YR XR WR RR N B eps A hd hc u hu hb' hN hB hne hpos

specification forward_index_commuted_correctness (input weight : Storage) (Y X W r : RegionName)
    (YR XR WR RR N B : Nat) (eps : ℝ) :
    Spec.Real (forward_correct_target (forward input weight Bool.true Y X W r YR XR WR RR N B eps).eraseDType.toAlgKernel
      Bool.false Y X W r YR XR WR RR N B eps) := by
  intro A hd hc u hu hb hN hB hne hpos
  have hb' : ForwardIO.Bounds Bool.false Y X W r YR XR WR RR N B u A.extent :=
    ⟨hb.input,hb.weight,hb.output,hb.inv⟩
  exact ForwardIO.candidate_correct input weight Bool.false Y X W r YR XR WR RR N B eps A hd hc u hu hb' hN hB hne hpos

specification gemma_correctness (input weight : Storage) (Y X W r : RegionName)
    (YR XR WR RR N B : Nat) (eps : ℝ) :
    Spec.Real (forward_correct_target (gemma input weight Bool.false Y X W r YR XR WR RR N B eps).eraseDType.toAlgKernel
      Bool.true Y X W r YR XR WR RR N B eps) := by
  intro A hd hc u hu hb hN hB hne hpos
  have hb' : ForwardIO.Bounds Bool.true Y X W r YR XR WR RR N B u A.extent :=
    ⟨hb.input,hb.weight,hb.output,hb.inv⟩
  exact ForwardIO.typed_correct input weight Bool.true Y X W r YR XR WR RR N B eps A hd hc u hu hb' hN hB hne hpos

specification gemma_index_commuted_correctness (input weight : Storage) (Y X W r : RegionName)
    (YR XR WR RR N B : Nat) (eps : ℝ) :
    Spec.Real (forward_correct_target (gemma input weight Bool.true Y X W r YR XR WR RR N B eps).eraseDType.toAlgKernel
      Bool.true Y X W r YR XR WR RR N B eps) := by
  intro A hd hc u hu hb hN hB hne hpos
  have hb' : ForwardIO.Bounds Bool.true Y X W r YR XR WR RR N B u A.extent :=
    ⟨hb.input,hb.weight,hb.output,hb.inv⟩
  exact ForwardIO.candidate_correct input weight Bool.true Y X W r YR XR WR RR N B eps A hd hc u hu hb' hN hB hne hpos

structure BackwardBounds (Y X W r : RegionName) (YR XR RR N B : Nat)
    (u : BlockState) (bounds : RegionBounds) : Prop where
  dy : ∀ j : Fin B,j.val<N → u.pids 0*YR+j.val<bounds Y
  input : ∀ j : Fin B,j.val<N → u.pids 0*XR+j.val<bounds X
  weight : ∀ j : Fin B,j.val<N → j.val<bounds W
  inv : u.pids 0*RR<bounds r

def backwardOutside (A : FlatAlloc) (u : BlockState) (Y : RegionName) (YR N B : Nat)
    (q : RegionName) (o : Nat) : Prop :=
  q≠A.flat ∨ ∀ j : Fin B,j.val<N → o≠A.addr Y (u.pids 0*YR+j.val)

def backward_correct_target (kernel : Kernel) (isGemma : Bool) (Y X W r : RegionName) (YR XR RR N B : Nat) : Prop :=
  ∀ A : FlatAlloc,A.Disjoint → (∀ q,q∉A.regions → A.extent q=0) →
  ∀ u : BlockState,u.undef=(fun _ _ => 0) → BackwardBounds Y X W r YR XR RR N B u A.extent →
  0<N → N≤B →
    ∃ v,exec (A.flattenKernel kernel) (A.flattenState u)=some v ∧
      (∀ j : Fin B,j.val<N → v.readMem A.flat (A.addr Y (u.pids 0*YR+j.val))=
        gradient isGemma N (u.readMem r (u.pids 0*RR))
          (fun k : Fin B => u.readMem X (u.pids 0*XR+k.val))
          (fun k : Fin B => u.readMem Y (u.pids 0*YR+k.val))
          (fun k : Fin B => u.readMem W k.val) j) ∧
      (∀ q o,backwardOutside A u Y YR N B q o → v.mem q o=(A.flattenState u).mem q o)

specification backward_correctness (input weight : Storage) (isGemma : Bool) (Y X W r dW : RegionName)
    (YR XR WR RR DWR N B : Nat) (eps : ℝ) :
    Spec.Real (backward_correct_target (backward input weight Bool.false Y X W r dW YR XR WR RR DWR N B eps isGemma).eraseDType.toAlgKernel
      isGemma Y X W r YR XR RR N B) := by
  intro A hd hc u hu hb hN hB
  have hb' : BackwardIO.Bounds Y X W r YR XR RR N B u A.extent :=
    ⟨hb.dy,hb.input,hb.weight,hb.inv⟩
  exact BackwardIO.typed_correct input weight isGemma Y X W r dW YR XR WR RR DWR N B eps A hd hc u hu hb' hN hB

specification backward_index_commuted_correctness (input weight : Storage) (isGemma : Bool) (Y X W r dW : RegionName)
    (YR XR WR RR DWR N B : Nat) (eps : ℝ) :
    Spec.Real (backward_correct_target (backward input weight Bool.true Y X W r dW YR XR WR RR DWR N B eps isGemma).eraseDType.toAlgKernel
      isGemma Y X W r YR XR RR N B) := by
  intro A hd hc u hu hb hN hB
  have hb' : BackwardIO.Bounds Y X W r YR XR RR N B u A.extent :=
    ⟨hb.dy,hb.input,hb.weight,hb.inv⟩
  exact BackwardIO.candidate_correct input weight isGemma Y X W r dW YR XR WR RR DWR N B eps A hd hc u hu hb' hN hB

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FastRmsLayernorm.Correctness
