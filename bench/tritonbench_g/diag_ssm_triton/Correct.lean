import bench.tritonbench_g.diag_ssm_triton.Proofs.ForwardIO
import bench.tritonbench_g.diag_ssm_triton.Proofs.BackwardIO
import bench.tritonbench_g.diag_ssm_triton.Proofs.ComplexForwardIO
import bench.tritonbench_g.diag_ssm_triton.Proofs.ComplexBackwardIO
/-! Real correctness of the four complete typed sources in Kernels.lean.

Forward: h₀=s and hₜ₊₁=hₜ λ+xₜ; row t stores hₜ₊₁.
Backward: starting with carry=0, reverse time computes gxₜ=gyₜ+carry,
then carry=gxₜ λ* and accumulates gl += gxₜ hₜ*; the last carry is gs.
Here * denotes complex conjugation (identity for real values). Reference
contains these independent mathematical recurrences and the forward closed
form. All formulas read the initial memory, not intermediate kernel states.

The contracts below cover any sequence length, including zero, and masked
partial tiles. Bounds are in real storage cells; complex64 uses interleaved
fp32 components. The full tagged memory outside the stated writes is preserved.
Storage is erased only for these mathematical correctness statements. These
are per-program contracts; host batch reduction, launch races and fixed-width
index overflow are outside their scope. -/
namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton.Correctness
open VeriTile VeriTile.Triton TypedSource
noncomputable section

namespace RealForward
abbrev Layout := ForwardLayout

def original (storage : Storage) (p : Layout) (length : Nat) : ComputeKernel :=
  (TypedSource.forward storage Bool.false p.s p.x p.l p.y length p.batch p.dim p.B).eraseDType

def indexCommuted (storage : Storage) (p : Layout) (length : Nat) : ComputeKernel :=
  (TypedSource.forward storage Bool.true p.s p.x p.l p.y length p.batch p.dim p.B).eraseDType

def column (p : Layout) (u : BlockState) (j : Fin p.B) : Nat := u.pids 0*p.B+j.val
def active (p : Layout) (u : BlockState) (j : Fin p.B) : Prop := column p u j<p.batch*p.dim
def offset (p : Layout) (u : BlockState) (t : Nat) (j : Fin p.B) : Nat := t*(p.batch*p.dim)+column p u j

structure Bounds (p : Layout) (u : BlockState) (length : Nat) (bounds : RegionBounds) : Prop where
  initial : ∀ j, active p u j → column p u j < bounds p.s
  transition : ∀ j, active p u j → column p u j%p.dim < bounds p.l
  input : ∀ t, t<length → ∀ j, active p u j → offset p u t j < bounds p.x
  output : ∀ t, t<length → ∀ j, active p u j → offset p u t j < bounds p.y

/-- Independent scalar recurrence over the initial input memory. -/
def value (p : Layout) (u : BlockState) (t : Nat) (j : Fin p.B) : ℝ :=
  Reference.forwardState (u.readMem p.s (column p u j)) (u.readMem p.l (column p u j%p.dim))
    (fun q => u.readMem p.x (offset p u q j)) (t+1)

def Outside (p : Layout) (A : FlatAlloc) (u : BlockState) (length : Nat) (r : RegionName) (o : Nat) : Prop :=
  r≠A.flat ∨ ∀ t, t<length → ∀ j, active p u j → o≠A.addr p.y (offset p u t j)

/-- Uniform real contract, including zero length. The initial-state and
transition bounds remain required because their loads precede the loop. -/
def contract (kernel : ComputeKernel) (p : Layout) (length : Nat) : Prop :=
  (0<length → p.x≠p.y) →
  ∀ A : FlatAlloc, A.Disjoint → A.regions=[p.s,p.l,p.x,p.y] →
    (∀ r, r ∉ A.regions → A.extent r=0) →
  ∀ u : BlockState, u.undef=(fun _ _ => 0) → Bounds p u length A.extent →
    ∃ v, exec (A.flattenKernel kernel.toAlgKernel) (A.flattenState u)=some v ∧
      (∀ t, t<length → ∀ j, active p u j → v.readMem A.flat (A.addr p.y (offset p u t j))=value p u t j) ∧
      (∀ r o, Outside p A u length r o → v.mem r o=(A.flattenState u).mem r o)
specification forward_correctness (storage : Storage) (p : Layout) (length : Nat) :
    Spec.Real (contract (original storage p length) p length) := by
  intro hsep A hd hr hc u hu hb
  exact ForwardIO.original_correct storage p length hsep A hd hr hc u hu
    ⟨hb.initial,hb.transition,hb.input,hb.output⟩

specification forward_index_commuted_correctness (storage : Storage) (p : Layout) (length : Nat) :
    Spec.Real (contract (indexCommuted storage p length) p length) := by
  intro hsep A hd hr hc u hu hb
  exact ForwardIO.candidate_correct storage p length hsep A hd hr hc u hu
    ⟨hb.initial,hb.transition,hb.input,hb.output⟩

end RealForward

namespace RealBackward
abbrev Layout := BackwardLayout

def original (storage : Storage) (p : Layout) (length : Nat) : ComputeKernel :=
  (TypedSource.backward storage Bool.false p.s p.l p.y p.gs p.gx p.gl p.gy length p.batch p.dim p.B).eraseDType

def indexCommuted (storage : Storage) (p : Layout) (length : Nat) : ComputeKernel :=
  (TypedSource.backward storage Bool.true p.s p.l p.y p.gs p.gx p.gl p.gy length p.batch p.dim p.B).eraseDType

def column (p : Layout) (u : BlockState) (j : Fin p.B) : Nat := u.pids 0*p.B+j.val
def active (p : Layout) (u : BlockState) (j : Fin p.B) : Prop := column p u j<p.batch*p.dim
def offset (p : Layout) (u : BlockState) (t : Nat) (j : Fin p.B) : Nat := t*(p.batch*p.dim)+column p u j

/-- Sufficient region separation for the reused positive-length stream proof. -/
structure Separation (p : Layout) : Prop where
  initial : p.gx≠p.s
  output : p.gx≠p.y
  gradient : p.gx≠p.gy
  transition : p.gx≠p.l
  initialGradient : p.gs≠p.gx
  transitionGradient : p.gl≠p.gx
  terminal : p.gl≠p.gs

structure Bounds (p : Layout) (u : BlockState) (length : Nat) (bounds : RegionBounds) : Prop where
  initial : 0<length → ∀ j, active p u j → column p u j < bounds p.s
  transition : ∀ j, active p u j → column p u j%p.dim < bounds p.l
  previous : ∀ t, t+1<length → ∀ j, active p u j → offset p u t j < bounds p.y
  gradOutput : ∀ t, t<length → ∀ j, active p u j → offset p u t j < bounds p.gy
  gradInput : ∀ t, t<length → ∀ j, active p u j → offset p u t j < bounds p.gx
  gradInitial : ∀ j, active p u j → column p u j < bounds p.gs
  gradTransition : ∀ j, active p u j → column p u j < bounds p.gl

def inputValue (p : Layout) (u : BlockState) (length t : Nat) (j : Fin p.B) : ℝ :=
  Reference.inputGradient length (u.readMem p.l (column p u j%p.dim))
    (fun q => u.readMem p.gy (offset p u q j)) t
def initialValue (p : Layout) (u : BlockState) (length : Nat) (j : Fin p.B) : ℝ :=
  Reference.reverseCarry length (u.readMem p.l (column p u j%p.dim))
    (fun q => u.readMem p.gy (offset p u q j)) length
def transitionValue (p : Layout) (u : BlockState) (length : Nat) (j : Fin p.B) : ℝ :=
  Reference.transitionGradient length (u.readMem p.l (column p u j%p.dim)) (u.readMem p.s (column p u j))
    (fun q => u.readMem p.y (offset p u q j)) (fun q => u.readMem p.gy (offset p u q j)) length

def Outside (p : Layout) (A : FlatAlloc) (u : BlockState) (length : Nat) (r : RegionName) (o : Nat) : Prop :=
  r≠A.flat ∨
    (∀ t, t<length → ∀ j, active p u j → o≠A.addr p.gx (offset p u t j)) ∧
    (∀ j, active p u j → o≠A.addr p.gs (column p u j)) ∧
    (∀ j, active p u j → o≠A.addr p.gl (column p u j))

/-- Zero length still reads the transition and writes both terminal gradients.
No output separation is required for that constant-zero path. -/
def contract (kernel : ComputeKernel) (p : Layout) (length : Nat) : Prop :=
  (0<length → Separation p) →
  ∀ A : FlatAlloc, A.Disjoint → A.regions=[p.s,p.l,p.y,p.gs,p.gx,p.gl,p.gy] →
    (∀ r, r ∉ A.regions → A.extent r=0) →
  ∀ u : BlockState, u.undef=(fun _ _ => 0) → Bounds p u length A.extent →
    ∃ v, exec (A.flattenKernel kernel.toAlgKernel) (A.flattenState u)=some v ∧
      (∀ t, t<length → ∀ j, active p u j → v.readMem A.flat (A.addr p.gx (offset p u t j))=inputValue p u length t j) ∧
      (∀ j, active p u j → v.readMem A.flat (A.addr p.gs (column p u j))=initialValue p u length j) ∧
      (∀ j, active p u j → v.readMem A.flat (A.addr p.gl (column p u j))=transitionValue p u length j) ∧
      (∀ r o, Outside p A u length r o → v.mem r o=(A.flattenState u).mem r o)
specification backward_correctness (storage : Storage) (p : Layout) (length : Nat) :
    Spec.Real (contract (original storage p length) p length) := by
  intro hsep A hd hr hc u hu hb
  exact BackwardIO.original_correct storage p length (fun hp => let s := hsep hp; ⟨s.initial,s.output,s.gradient,s.transition,s.initialGradient,s.transitionGradient,s.terminal⟩) A hd hr hc u hu
    ⟨hb.initial,hb.transition,hb.previous,hb.gradOutput,hb.gradInput,hb.gradInitial,hb.gradTransition⟩

specification backward_index_commuted_correctness (storage : Storage) (p : Layout) (length : Nat) :
    Spec.Real (contract (indexCommuted storage p length) p length) := by
  intro hsep A hd hr hc u hu hb
  exact BackwardIO.candidate_correct storage p length (fun hp => let s := hsep hp; ⟨s.initial,s.output,s.gradient,s.transition,s.initialGradient,s.transitionGradient,s.terminal⟩) A hd hr hc u hu
    ⟨hb.initial,hb.transition,hb.previous,hb.gradOutput,hb.gradInput,hb.gradInitial,hb.gradTransition⟩

end RealBackward

/-- Read the two masked components of an original complex input. -/
def complexInput {B : Nat} (u : BlockState) (r : RegionName) (offset : TileIndex [B] → Nat)
    (mask : TileIndex [B] → Bool) : TileIndex [B] → ℂ := fun j =>
  ⟨if mask j then u.readMem r (offset j*2) else 0,
   if mask j then u.readMem r (offset j*2+1) else 0⟩

/-- Relocate components separately; the allocation base may be odd. -/
def complexOutput (A : FlatAlloc) (v : BlockState) (r : RegionName) (offset : Nat) : ℂ :=
  ⟨v.readMem A.flat (A.addr r (offset*2)),v.readMem A.flat (A.addr r (offset*2+1))⟩

namespace ComplexForward
abbrev Layout := ForwardLayout
abbrev Lane (p : Layout) := TileIndex [p.B]

def original (p : Layout) (length : Nat) : ComputeKernel :=
  (TypedSource.complexForward Bool.false p.s p.x p.y p.l length p.batch p.dim p.B).eraseDType

def indexCommuted (p : Layout) (length : Nat) : ComputeKernel :=
  (TypedSource.complexForward Bool.true p.s p.x p.y p.l length p.batch p.dim p.B).eraseDType

def columns (p : Layout) (u : BlockState) (j : Lane p) : Nat := u.pids 0*p.B+j.1.val
def mask (p : Layout) (u : BlockState) (j : Lane p) : Bool := decide (columns p u j<p.batch*p.dim)
def offset (p : Layout) (u : BlockState) (t : Nat) (j : Lane p) : Nat := t*(p.batch*p.dim)+columns p u j

def initial (p : Layout) (u : BlockState) : Lane p → ℂ := complexInput u p.s (columns p u) (mask p u)
def transition (p : Layout) (u : BlockState) : Lane p → ℂ :=
  complexInput u p.l (fun j => columns p u j%p.dim) (mask p u)
def input (p : Layout) (u : BlockState) (t : Nat) : Lane p → ℂ :=
  complexInput u p.x (offset p u t) (mask p u)

def state (p : Layout) (u : BlockState) (t : Nat) (j : Lane p) : ℂ :=
  Reference.forwardState (initial p u j) (transition p u j) (fun q => input p u q j) t

structure Bounds (p : Layout) (u : BlockState) (length : Nat) (bounds : RegionBounds) : Prop where
  initial : ∀ j, mask p u j=Bool.true → columns p u j*2+1 < bounds p.s
  transition : ∀ j, mask p u j=Bool.true → (columns p u j%p.dim)*2+1 < bounds p.l
  input : ∀ t, t<length → ∀ j, mask p u j=Bool.true → offset p u t j*2+1 < bounds p.x
  output : ∀ t, t<length → ∀ j, mask p u j=Bool.true → offset p u t j*2+1 < bounds p.y
def Outside (p : Layout) (A : FlatAlloc) (u : BlockState) (length : Nat)
    (r : RegionName) (o : Nat) : Prop :=
  r ≠ A.flat ∨ ∀ t, t<length → ∀ j, mask p u j=Bool.true →
    A.addr p.y (offset p u t j*2) ≠ o ∧ A.addr p.y (offset p u t j*2+1) ≠ o

/-- Full finite real IO for the two component stores. Static and temporal
access bounds, successful flat execution, the independent recurrence and
the complete tagged-cell frame are all part of this contract. -/
def contract (kernel : ComputeKernel) (p : Layout) (length : Nat) : Prop :=
  ∀ A : FlatAlloc, A.Disjoint →
    (∀ r, r ∉ A.regions → A.extent r=0) →
  ∀ u : BlockState, u.undef=(fun _ _ => 0) →
    Bounds p u length A.extent →
    kernel.toAlgKernel.TraceSafe A.extent u ∧
    ∃ v, exec (A.flattenKernel kernel.toAlgKernel) (A.flattenState u)=some v ∧
      (∀ t, t<length → ∀ j, mask p u j=Bool.true →
        complexOutput A v p.y (offset p u t j)=state p u (t+1) j) ∧
      (∀ r o, Outside p A u length r o → v.mem r o=(A.flattenState u).mem r o)
specification complex_forward_correctness (p : Layout) (length : Nat) :
    Spec.Real (contract (original p length) p length) := by
  intro A hd hc u hu hb
  exact ComplexForwardIO.original_correct p length A hd hc u hu
    ⟨hb.initial,hb.transition,hb.input,hb.output⟩

specification complex_forward_index_commuted_correctness (p : Layout) (length : Nat) :
    Spec.Real (contract (indexCommuted p length) p length) := by
  intro A hd hc u hu hb
  exact ComplexForwardIO.candidate_correct p length A hd hc u hu
    ⟨hb.initial,hb.transition,hb.input,hb.output⟩

end ComplexForward

namespace ComplexBackward
abbrev Layout := BackwardLayout
abbrev Lane (p : Layout) := TileIndex [p.B]

def original (p : Layout) (length : Nat) : ComputeKernel :=
  (TypedSource.complexBackward Bool.false p.s p.l p.y p.gs p.gx p.gl p.gy length p.batch p.dim p.B).eraseDType

def indexCommuted (p : Layout) (length : Nat) : ComputeKernel :=
  (TypedSource.complexBackward Bool.true p.s p.l p.y p.gs p.gx p.gl p.gy length p.batch p.dim p.B).eraseDType

def columns (p : Layout) (u : BlockState) (j : Lane p) : Nat := u.pids 0*p.B+j.1.val
def mask (p : Layout) (u : BlockState) (j : Lane p) : Bool := decide (columns p u j<p.batch*p.dim)
def offset (p : Layout) (u : BlockState) (t : Nat) (j : Lane p) : Nat := t*(p.batch*p.dim)+columns p u j

def initial (p : Layout) (u : BlockState) : Lane p → ℂ := complexInput u p.s (columns p u) (mask p u)
def transition (p : Layout) (u : BlockState) : Lane p → ℂ :=
  complexInput u p.l (fun j => columns p u j%p.dim) (mask p u)
def output (p : Layout) (u : BlockState) (t : Nat) : Lane p → ℂ :=
  complexInput u p.y (offset p u t) (mask p u)
def gradOutput (p : Layout) (u : BlockState) (t : Nat) : Lane p → ℂ :=
  complexInput u p.gy (offset p u t) (mask p u)

/-- Sufficient alias conditions for nonempty reverse scans. All terminal
stores follow the reads; input regions can alias one another. -/
structure Separation (p : Layout) : Prop where
  initial : p.s≠p.gx
  output : p.y≠p.gx
  gradient : p.gy≠p.gx
  terminal : p.gs≠p.gl
  initialGradient : p.gx≠p.gs
  transitionGradient : p.gx≠p.gl

structure Bounds (p : Layout) (u : BlockState) (length : Nat) (bounds : RegionBounds) : Prop where
  initial : 0<length → ∀ j, mask p u j=Bool.true → columns p u j*2+1 < bounds p.s
  transition : ∀ j, mask p u j=Bool.true → (columns p u j%p.dim)*2+1 < bounds p.l
  previous : ∀ t, t+1<length → ∀ j, mask p u j=Bool.true → offset p u t j*2+1 < bounds p.y
  gradOutput : ∀ t, t<length → ∀ j, mask p u j=Bool.true → offset p u t j*2+1 < bounds p.gy
  gradInput : ∀ t, t<length → ∀ j, mask p u j=Bool.true → offset p u t j*2+1 < bounds p.gx
  gradInitial : ∀ j, mask p u j=Bool.true → columns p u j*2+1 < bounds p.gs
  gradTransition : ∀ j, mask p u j=Bool.true → columns p u j*2+1 < bounds p.gl
def Outside (p : Layout) (A : FlatAlloc) (u : BlockState) (length : Nat)
    (r : RegionName) (o : Nat) : Prop :=
  r≠A.flat ∨
    (∀ t, t<length → ∀ j, mask p u j=Bool.true →
      A.addr p.gx (offset p u t j*2)≠o ∧ A.addr p.gx (offset p u t j*2+1)≠o) ∧
    (∀ j, mask p u j=Bool.true → A.addr p.gs (columns p u j*2)≠o ∧ A.addr p.gs (columns p u j*2+1)≠o) ∧
    (∀ j, mask p u j=Bool.true → A.addr p.gl (columns p u j*2)≠o ∧ A.addr p.gl (columns p u j*2+1)≠o)

/-- Actual finite access safety, successful flat execution, all three
independent reverse-scan outputs and the full tagged-cell frame. -/
def contract (kernel : ComputeKernel) (p : Layout) (length : Nat) : Prop :=
  (0<length → Separation p) →
  ∀ A : FlatAlloc, A.Disjoint → (∀ r, r ∉ A.regions → A.extent r=0) →
  ∀ u : BlockState, u.undef=(fun _ _ => 0) → Bounds p u length A.extent →
    kernel.toAlgKernel.TraceSafe A.extent u ∧
    ∃ v, exec (A.flattenKernel kernel.toAlgKernel) (A.flattenState u)=some v ∧
      (∀ t, t<length → ∀ j, mask p u j=Bool.true →
        complexOutput A v p.gx (offset p u t j)=Reference.inputGradient length
          (starRingEnd ℂ (transition p u j)) (fun q => gradOutput p u q j) t) ∧
      (∀ j, mask p u j=Bool.true → complexOutput A v p.gs (columns p u j)=
        Reference.reverseCarry length (starRingEnd ℂ (transition p u j)) (fun q => gradOutput p u q j) length) ∧
      (∀ j, mask p u j=Bool.true → complexOutput A v p.gl (columns p u j)=
        Reference.transitionGradient length (starRingEnd ℂ (transition p u j)) (starRingEnd ℂ (initial p u j))
          (fun q => starRingEnd ℂ (output p u q j)) (fun q => gradOutput p u q j) length) ∧
      (∀ r o, Outside p A u length r o → v.mem r o=(A.flattenState u).mem r o)
specification complex_backward_correctness (p : Layout) (length : Nat) :
    Spec.Real (contract (original p length) p length) := by
  intro hsep A hd hc u hu hb
  exact ComplexBackwardIO.original_correct p length (fun hp => let s := hsep hp; ⟨⟨s.initial,s.output,s.gradient⟩,s.terminal,s.initialGradient,s.transitionGradient⟩) A hd hc u hu
    ⟨hb.initial,hb.transition,hb.previous,hb.gradOutput,hb.gradInput,hb.gradInitial,hb.gradTransition⟩

specification complex_backward_index_commuted_correctness (p : Layout) (length : Nat) :
    Spec.Real (contract (indexCommuted p length) p length) := by
  intro hsep A hd hc u hu hb
  exact ComplexBackwardIO.candidate_correct p length (fun hp => let s := hsep hp; ⟨⟨s.initial,s.output,s.gradient⟩,s.terminal,s.initialGradient,s.transitionGradient⟩) A hd hc u hu
    ⟨hb.initial,hb.transition,hb.previous,hb.gradOutput,hb.gradInput,hb.gradInitial,hb.gradTransition⟩

end ComplexBackward

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DiagSsmTriton.Correctness
