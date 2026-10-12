import bench.tritonbench_g.diag_ssm_triton.Proofs.ComplexBackwardLoop
import bench.tritonbench_g.diag_ssm_triton.Proofs.ComplexBackwardFinal
namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton.ComplexBackwardRun
open VeriTile.Triton ComplexBackwardLoop
noncomputable section
set_option maxHeartbeats 1500000
set_option maxRecDepth 20000

/-- Terminal output regions must remain separately observable. All terminal
stores occur after the reverse loop's reads, so no further input separation
is needed here. -/
structure Separation (p : Layout) : Prop extends ReadSeparation p where
  terminal : p.gs≠p.gl
  initialGradient : p.gx≠p.gs
  transitionGradient : p.gx≠p.gl

def TerminalOutside (p : Layout) (u : BlockState) (out r : RegionName) (o : Nat) : Prop :=
  r≠out ∨ ∀ j, mask p u j=Bool.true → columns p u j*2≠o ∧ columns p u j*2+1≠o

def Outside (p : Layout) (u : BlockState) (length : Nat) (r : RegionName) (o : Nat) : Prop :=
  ComplexBackwardLoop.Outside p u length length r o ∧
    TerminalOutside p u p.gs r o ∧ TerminalOutside p u p.gl r o

/-- Complete regional-memory execution of the original complex reverse
source, including both terminal gradient stores. All formulas below use
original memory and the independent mathematical reference. -/
theorem run (p : Layout) (u : BlockState) (length : Nat) (sep : Separation p) :
    ∃ w, exec (Erasure.complexBackward Bool.false p.s p.l p.y p.gs p.gx p.gl p.gy
      length p.batch p.dim p.B).toAlgKernel u=some w ∧
      (∀ t, t<length → ∀ j, mask p u j=Bool.true →
        ComplexMemory.read w p.gx (offset p u t j)=Reference.inputGradient length
          (starRingEnd ℂ (transition p u j)) (fun q => gradOutput p u q j) t) ∧
      (∀ j, mask p u j=Bool.true → ComplexMemory.read w p.gs (columns p u j)=
        Reference.reverseCarry length (starRingEnd ℂ (transition p u j)) (fun q => gradOutput p u q j) length) ∧
      (∀ j, mask p u j=Bool.true → ComplexMemory.read w p.gl (columns p u j)=
        Reference.transitionGradient length (starRingEnd ℂ (transition p u j)) (starRingEnd ℂ (initial p u j))
          (fun q => starRingEnd ℂ (output p u q j)) (fun q => gradOutput p u q j) length) ∧
      (∀ r o, Outside p u length r o → w.mem r o=u.mem r o) := by
  obtain ⟨v,he,hv⟩ := run_loop p u length sep.toReadSeparation
  let w := ComplexBackwardFinal.result v p.gs p.gl (columns p u) (mask p u)
    (carry p u length length) (acc p u length length)
  have hpost := ComplexBackwardFinal.post_step v p.gs p.gl (columns p u) (mask p u)
    (carry p u length length) (acc p u length length) (transition p u) hv.registers
  have hinj : Function.Injective (columns p u) := ComplexMemory.lane_injective p.B (u.pids 0*p.B)
  refine ⟨w,?_,?_,?_,?_,?_⟩
  · unfold exec
    rw [ComplexBackwardSource.source_body]
    simp only [ProgramRewrite.append,ComplexBackwardInit.prefix_step,Option.bind_some,stepStmts,he,hpost]
    rfl
  · intro t ht j hm
    have hf (c : Fin 2) : w.mem p.gx (offset p u t j*2+c.val)=v.mem p.gx (offset p u t j*2+c.val) := by
      exact ComplexBackwardFinal.frame _ _ _ _ _ _ _ _ _ (Or.inl sep.initialGradient) (Or.inl sep.transitionGradient)
    rw [ComplexMemory.read_of_mem (by simpa using hf 0) (by simpa using hf 1)]
    rw [hv.outputs t (by omega) ht j hm]
    exact ComplexBackwardReference.conjugate_input length _ _ t
  · intro j hm
    rw [ComplexBackwardFinal.read_carry v p.gs p.gl (columns p u) (mask p u)
      (carry p u length length) (acc p u length length) sep.terminal hinj j hm]
    exact ComplexBackwardReference.conjugate_carry length _ _ length
  · intro j hm
    rw [ComplexBackwardFinal.read_acc v p.gs p.gl (columns p u) (mask p u)
      (carry p u length length) (acc p u length length) hinj j hm]
    exact ComplexBackwardReference.conjugate_transition length _ _ _ _ length
  · intro r o ho
    exact (ComplexBackwardFinal.frame _ _ _ _ _ _ _ r o ho.2.1 ho.2.2).trans (hv.frame r o ho.1)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DiagSsmTriton.ComplexBackwardRun
