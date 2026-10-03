/- Two-output FP specifications must observe both typed output windows and
preserve every other cell. Store reordering needs no numerical assumption. -/
import VeriTile.Triton.DSL
import VeriTile.Triton.Float.GuardedIO
import VeriTile.Meta.StatementAudit

namespace FPDualOutputTests
open VeriTile Triton FP.Structural
open scoped VeriTile.Spec

def forwardKernel (xReg mReg vReg : RegionName) : ComputeKernel := triton {
  x := tl.load($(xReg))
  tl.store($(mReg), (x).to(tl.bfloat16))
  tl.store($(vReg), (x).to(tl.bfloat16))
}

def reverseKernel (xReg mReg vReg : RegionName) : ComputeKernel := triton {
  x := tl.load($(xReg))
  tl.store($(vReg), (x).to(tl.bfloat16))
  tl.store($(mReg), (x).to(tl.bfloat16))
}

def forwardIO : KernelIO₁ₓ₂ where
  kernel := forwardKernel "x" "mean" "variance"
  inp := "x"
  out1 := "mean"
  out2 := "variance"
  Bin := 1
  Bout1 := 1
  Bout2 := 1
  read := fun _ => 0
  write1 := fun _ => 0
  write2 := fun _ => 0

def reverseIO : KernelIO₁ₓ₂ :=
  { forwardIO with
    -- Keep the declared port order while reordering the stores in the body.
    kernel := .mk ["x"] ["mean", "variance"] (reverseKernel "x" "mean" "variance").surfaceBody
    projection := by rfl }

private theorem reordered : IO₁ₓ₂Equiv forwardIO reverseIO := by
  refine ⟨by simp [IO₁ₓ₂PrivateScratch, forwardIO],
    by simp [IO₁ₓ₂PrivateScratch, reverseIO, forwardIO], ?_⟩
  intro α _ M s
  simp [forwardIO, reverseIO, forwardKernel, reverseKernel,
    ComputeKernel.surfaceBody, FP.Structural.exec, run, step, evalExpr, evalOp_unfold, store,
    ofFloat, toFloat, TileShape.allIndices, State.write, IO₁ₓ₂Outputs, IO₁ₓ₂Frame]
  constructor <;> intro r o hm hv
  all_goals
    simp only [if_neg (by tauto : ¬ (r = "variance" ∧ o = 0)),
      if_neg (by tauto : ¬ (r = "mean" ∧ o = 0))]

def R : Spec.Assumptions ComputeStmt := []

specification reordered_outputs : forwardIO ≡[R] reverseIO :=
  Spec.FloatingPoint.ofStructural (structural := IO₁ₓ₂Equiv) rfl rfl reordered

def guardedForward : FP.Guarded.IO₁ₓ₂ := ⟨forwardIO, ⟨[], []⟩⟩
def guardedReverse : FP.Guarded.IO₁ₓ₂ := ⟨reverseIO, ⟨[], []⟩⟩
def guardedRules : Spec.Assumptions FP.GuardedFragment := []

specification guarded_reordered_outputs : guardedForward ≡[guardedRules] guardedReverse :=
  Spec.FloatingPoint.ofNumerical (structural := fun _ _ => False) rfl rfl
    (FP.Guarded.Equivalent₁ₓ₂.ofStructural guardedRules reordered)

-- An opaque whole-kernel premise is not a list of admitted scalar atoms.
specification opaque_guarded_outputs
    (h : FP.Guarded.Equivalent₁ₓ₂ guardedRules guardedForward guardedReverse) :
    guardedForward ≡[guardedRules] guardedReverse :=
  Spec.FloatingPoint.ofNumerical (structural := fun _ _ => False) rfl rfl h

private def M : Algebra Nat where
  literal := fun _ _ _ => 0
  negInf := 0
  binary := fun _ _ _ a b => a + b
  unary := fun _ _ a => a + 1
  cast := fun _ _ _ a => a + 1
  fromNat := fun _ n => n
  fromInt := fun _ n => n.toNat
  fp32Bits := fun b => b.bits.toNat
  fp32Load := id
  reduceMax := fun _ _ _ _ _ => 0
  reduceSum := fun _ _ _ _ _ => 0

private def initial : State Nat where
  mem := fun _ _ => .mk .real 0
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

private def changedKernel (xReg mReg vReg : RegionName) : ComputeKernel := triton {
  x := tl.load($(xReg))
  tl.store($(mReg), (x).to(tl.bfloat16))
  tl.store($(vReg), ((x).to(tl.bfloat16)).to(tl.bfloat16))
}

private def changedVariance : KernelIO₁ₓ₂ :=
  { forwardIO with kernel := changedKernel "x" "mean" "variance", projection := by rfl }

theorem second_output_cannot_be_ignored : ¬ IO₁ₓ₂Equiv forwardIO changedVariance := by
  intro h
  obtain ⟨a, b, ha, hb, ho, _⟩ := h.2.2 Nat M initial
  have hv := ho.2 ⟨0, by decide⟩
  simp [forwardIO, forwardKernel, changedVariance, changedKernel, FP.Structural.exec, run, step,
    evalExpr, evalOp_unfold, store, ofFloat, toFloat, TileShape.allIndices] at ha hb
  cases ha
  cases hb
  simp [forwardIO, changedVariance, M, initial] at hv

theorem empty_first_does_not_hide_second :
    ¬ IO₁ₓ₂Outputs { forwardIO with Bout1 := 0 } { forwardIO with Bout1 := 0 }
      initial initial (initial.write "variance" 0 (.mk .real 1)) := by
  intro h
  have hv := h.2 ⟨0, by decide⟩
  simp [forwardIO, initial] at hv

theorem second_dtype_is_observable :
    ¬ IO₁ₓ₂Outputs forwardIO forwardIO initial initial
      (initial.write "variance" 0 (.mk .bf16 0)) := by
  intro h
  have hv := h.2 ⟨0, by decide⟩
  simp [forwardIO, initial] at hv

private def unsupported : KernelIO₁ₓ₂ :=
  { forwardIO with
    kernel := .mk ["x"] ["mean", "variance"] [.effectMarker "tl.debug_barrier"]
    projection := by rfl }

theorem failed_runs_cannot_certify : ¬ IO₁ₓ₂Equiv unsupported unsupported := by
  intro h
  obtain ⟨a, _, ha, _⟩ := h.2.2 Nat M initial
  simp [unsupported, FP.Structural.exec, run, step] at ha

theorem output_neighbor_stays_framed :
    ¬ IO₁ₓ₂Frame forwardIO initial (initial.write "variance" 1 (.mk .real 9)) := by
  intro h
  have hv := h "variance" 1 (Or.inl (by decide))
    (Or.inr (by intro i; have := i.isLt; change i.val < 1 at this; simp [forwardIO]))
    (by simp [forwardIO])
  simp [initial] at hv

theorem scratch_neighbor_stays_framed :
    ¬ IO₁ₓ₂Frame
      { forwardIO with scratch := [{ buf := "tmp", win := fun _ => 0, len := 1 }] }
      initial (initial.write "tmp" 1 (.mk .real 9)) := by
  intro h
  have ht := h "tmp" 1 (Or.inl (by decide)) (Or.inl (by decide)) (by
    intro p hp _ i
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hp
    subst p
    simp)
  simp [initial] at ht

theorem public_regions_are_not_scratch (r : RegionName)
    (hr : r = "x" ∨ r = "mean" ∨ r = "variance") :
    ¬ IO₁ₓ₂PrivateScratch
      { forwardIO with scratch := [{ buf := r, win := fun _ => 0, len := 1 }] } := by
  rcases hr with rfl | rfl | rfl <;> simp [IO₁ₓ₂PrivateScratch, forwardIO]

theorem signature_keeps_second_name :
    io₁ₓ₂Signature forwardIO ≠ io₁ₓ₂Signature { forwardIO with out2 := "hidden" } := by
  intro h
  have := congrArg IO₁ₓ₂Signature.out2 h
  contradiction

theorem signature_keeps_second_length :
    io₁ₓ₂Signature forwardIO ≠ io₁ₓ₂Signature { forwardIO with Bout2 := 0 } := by
  intro h
  have := congrArg IO₁ₓ₂Signature.Bout2 h
  contradiction

theorem signature_keeps_second_address :
    io₁ₓ₂Signature forwardIO ≠ io₁ₓ₂Signature { forwardIO with write2 := fun _ => 1 } := by
  intro h
  have := congrArg (fun sig : IO₁ₓ₂Signature => sig.write2 0) h
  contradiction

theorem signature_keeps_ports :
    io₁ₓ₂Signature forwardIO ≠ io₁ₓ₂Signature
      { forwardIO with
        kernel := .mk ["x"] ["mean"] (forwardKernel "x" "mean" "variance").surfaceBody
        projection := by rfl } := by
  intro h
  have := congrArg IO₁ₓ₂Signature.ports h
  contradiction

theorem guarded_signature_keeps_domain :
    Spec.ProgramSyntax.signature guardedForward ≠ Spec.ProgramSyntax.signature
      { guardedForward with domain := ⟨[], [⟨"x", [], .finite⟩]⟩ } := by
  intro h
  have := congrArg (fun sig : IO₁ₓ₂Signature × FP.Guarded.Precondition =>
    sig.2.guards.length) h
  contradiction

theorem syntax_preserves_scratch :
    ¬ Spec.ProgramSyntax.sameContext forwardIO
      { forwardIO with scratch := [{ buf := "tmp", win := fun _ => 0, len := 1 }] } := by
  change ¬ ([] : List ScratchSpec) = [_]
  simp

#axiomsClean reordered_outputs
#axiomsClean guarded_reordered_outputs
#print_fp_assumptions reordered_outputs
#print_fp_assumptions guarded_reordered_outputs
#print_fp_assumptions opaque_guarded_outputs

end FPDualOutputTests
