import bench.tritonbench_g.diag_ssm_triton.Proofs.ComplexBackwardInit
import bench.tritonbench_g.diag_ssm_triton.Proofs.ComplexBackwardStages
import bench.tritonbench_g.diag_ssm_triton.Proofs.ComplexBackwardReference
import VeriTile.Triton.KernelLemmas.LoopInvariant
import VeriTile.Triton.Semantics.ProgramRewrite
namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton.ComplexBackwardLoop
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 20000

abbrev Layout := BackwardLayout
abbrev Lane (p : Layout) := TileIndex [p.B]

def columns (p : Layout) (u : BlockState) : Lane p → Nat := ComplexForwardInit.columns u p.B
def mask (p : Layout) (u : BlockState) : Lane p → Bool := ComplexForwardInit.active u p.batch p.dim p.B
def offset (p : Layout) (u : BlockState) (t : Nat) : Lane p → Nat :=
  ComplexForwardStep.offset p.batch p.dim t (columns p u)
def initial (p : Layout) (u : BlockState) : Lane p → ℂ := ComplexForwardInit.initial u p.s p.batch p.dim p.B
def transition (p : Layout) (u : BlockState) : Lane p → ℂ := ComplexForwardInit.transition u p.l p.batch p.dim p.B
def output (p : Layout) (u : BlockState) (t : Nat) : Lane p → ℂ :=
  ComplexForwardStep.input u p.y (offset p u t) (mask p u)
def gradOutput (p : Layout) (u : BlockState) (t : Nat) : Lane p → ℂ :=
  ComplexForwardStep.input u p.gy (offset p u t) (mask p u)

def carry (p : Layout) (u : BlockState) (length k : Nat) : Lane p → ℂ := fun j =>
  Reference.reverseCarry length (transition p u j) (fun t => starRingEnd ℂ (gradOutput p u t j)) k
def acc (p : Layout) (u : BlockState) (length k : Nat) : Lane p → ℂ := fun j =>
  Reference.transitionGradient length (transition p u j) (initial p u j) (fun t => output p u t j)
    (fun t => starRingEnd ℂ (gradOutput p u t j)) k

/-- Source gradients are observed after conjugation; the final IO theorem
rewrites this formula to the independent conjugated-transition recurrence. -/
def inputGrad (p : Layout) (u : BlockState) (length t : Nat) : Lane p → ℂ := fun j =>
  starRingEnd ℂ (Reference.inputGradient length (transition p u j)
    (fun q => starRingEnd ℂ (gradOutput p u q j)) t)

/-- Sufficient alias conditions for the reverse loop. Input regions may
alias one another. The transition is loaded before every loop write. -/
structure ReadSeparation (p : Layout) : Prop where
  initial : p.s≠p.gx
  output : p.y≠p.gx
  gradient : p.gy≠p.gx

def Outside (p : Layout) (u : BlockState) (length k : Nat) (r : RegionName) (o : Nat) : Prop :=
  r≠p.gx ∨ ∀ t, length-k≤t → t<length → ∀ j, mask p u j=Bool.true →
    offset p u t j*2≠o ∧ offset p u t j*2+1≠o

structure Invariant (p : Layout) (u : BlockState) (length k : Nat) (v : BlockState) : Prop where
  registers : ComplexBackwardStep.Registers v (columns p u) (mask p u)
    (carry p u length k) (acc p u length k) (transition p u)
  pids : v.pids=u.pids
  outputs : ∀ t, length-k≤t → t<length → ∀ j, mask p u j=Bool.true →
    ComplexMemory.read v p.gx (offset p u t j)=inputGrad p u length t j
  frame : ∀ r o, Outside p u length k r o → v.mem r o=u.mem r o

theorem input_unchanged {p : Layout} {u v : BlockState} {length k : Nat}
    (h : Invariant p u length k v) (r : RegionName) (hr : r≠p.gx) (off : Lane p → Nat) :
    ComplexForwardStep.input v r off (mask p u) = ComplexForwardStep.input u r off (mask p u) := by
  funext j
  apply Complex.ext
  · simp only [ComplexForwardStep.input]
    have hc := h.frame r (off j*2) (Or.inl hr)
    unfold BlockState.readMem
    rw [hc]
  · simp only [ComplexForwardStep.input]
    have hc := h.frame r (off j*2+1) (Or.inl hr)
    unfold BlockState.readMem
    rw [hc]

theorem separated (p : Layout) (u : BlockState) (t q : Nat) (htq : t≠q)
    (j i : Lane p) (hj : mask p u j=Bool.true) (hi : mask p u i=Bool.true)
    (a b : Fin 2) : offset p u t j*2+a.val ≠ offset p u q i*2+b.val := by
  have hj' : columns p u j < p.batch*p.dim := of_decide_eq_true hj
  have hi' : columns p u i < p.batch*p.dim := of_decide_eq_true hi
  have h := ComplexMemory.time_separated (p.batch*p.dim) t q
    (columns p u j) (columns p u i) hj' hi' htq
  change (t*(p.batch*p.dim)+columns p u j)*2+a.val ≠
    (q*(p.batch*p.dim)+columns p u i)*2+b.val
  have ha := a.isLt
  have hb := b.isLt
  omega

theorem initial_invariant (p : Layout) (u : BlockState) (length : Nat) :
    Invariant p u length 0 (ComplexBackwardInit.result u p.l p.batch p.dim p.B) := by
  refine ⟨?_,rfl,?_,?_⟩
  · exact ComplexBackwardInit.result_registers u p.l p.batch p.dim p.B
  · intro t ht hlt; omega
  · intro r o _; rfl

/-- A reverse step reads the original input windows, updates both independent
recurrences and preserves already written gradients and every other cell. -/
theorem step (p : Layout) (u : BlockState) (length k : Nat) (hk : k<length)
    (sep : ReadSeparation p) (v : BlockState) (h : Invariant p u length k v) :
    ∃ w, stepStmts (ComplexBackwardSource.body p.s p.y p.gx p.gy length p.batch p.dim p.B)
      (v.setReg "i" .nat [] (Tile.scalar k)) = some w ∧ Invariant p u length (k+1) w := by
  let t := length-1-k
  let vt := v.setReg "i" .nat [] (Tile.scalar k)
  let w := ComplexBackwardStep.result vt p.s p.y p.gx p.gy p.batch p.dim t (columns p u) (mask p u)
    (carry p u length k) (acc p u length k) (transition p u)
  have ht := h.registers.setCounter k
  have hexec := ComplexBackwardStep.body_step p.s p.y p.gx p.gy length p.batch p.dim p.B k vt
    (columns p u) (mask p u) (carry p u length k) (acc p u length k) (transition p u) (by simp [vt]) ht
  have hg : ComplexBackwardStep.gradient vt p.gy (offset p u t) (mask p u) (carry p u length k) =
      fun j => starRingEnd ℂ (gradOutput p u t j)+carry p u length k j := by
    unfold ComplexBackwardStep.gradient
    rw [show ComplexForwardStep.input vt p.gy (offset p u t) (mask p u) = gradOutput p u t from
      input_unchanged h p.gy sep.gradient (offset p u t)]
  have hp : ComplexBackwardStep.previous vt p.s p.y p.batch p.dim t (columns p u) (mask p u) =
      fun j => Reference.previousState (initial p u j) (fun q => output p u q j) t := by
    by_cases hz : t=0
    · simp only [ComplexBackwardStep.previous,hz,↓reduceIte,Reference.previousState]
      exact input_unchanged h p.s sep.initial (columns p u)
    · simp only [ComplexBackwardStep.previous,hz,↓reduceIte,Reference.previousState]
      exact input_unchanged h p.y sep.output (offset p u (t-1))
  have hcarry : (fun j => ComplexBackwardStep.gradient vt p.gy (offset p u t) (mask p u)
      (carry p u length k) j * transition p u j) = carry p u length (k+1) := by
    rw [hg]
    rfl
  have hacc : (fun j => acc p u length k j + ComplexBackwardStep.gradient vt p.gy (offset p u t)
      (mask p u) (carry p u length k) j *
      ComplexBackwardStep.previous vt p.s p.y p.batch p.dim t (columns p u) (mask p u) j) =
      acc p u length (k+1) := by
    rw [hg,hp]
    rfl
  have hreg := ComplexBackwardStep.result_registers vt p.s p.y p.gx p.gy p.batch p.dim t
    (columns p u) (mask p u) (carry p u length k) (acc p u length k) (transition p u) ht
  have hreg' : ComplexBackwardStep.Registers w (columns p u) (mask p u)
      (carry p u length (k+1)) (acc p u length (k+1)) (transition p u) := by
    rw [←hcarry,←hacc]
    exact hreg
  have hinj : Function.Injective (offset p u t) := by
    simpa only [offset,ComplexForwardStep.offset,columns,ComplexForwardInit.columns,Nat.add_assoc] using
      ComplexMemory.lane_injective p.B (t*(p.batch*p.dim)+u.pids 0*p.B)
  dsimp only [offset] at hg
  refine ⟨w,hexec,hreg',?_,?_,?_⟩
  · exact (ComplexBackwardStep.result_pids _ _ _ _ _ _ _ _ _ _ _ _ _).trans h.pids
  · intro q hq hql j hm
    by_cases hqt : q=t
    · subst q
      have hout := ComplexBackwardStep.result_read vt p.s p.y p.gx p.gy p.batch p.dim t
        (columns p u) (mask p u) (carry p u length k) (acc p u length k) (transition p u) hinj j hm
      rw [hg] at hout
      have hk' : length-1-t=k := by dsimp [t]; omega
      simpa only [inputGrad,Reference.inputGradient,hk',carry] using hout
    · have hq' : length-k≤q := by dsimp [t] at hqt; omega
      have hpast (c : Fin 2) : w.mem p.gx (offset p u q j*2+c.val) =
          v.mem p.gx (offset p u q j*2+c.val) := by
        apply ComplexBackwardStep.result_frame
        right
        intro i hmi
        constructor
        · simpa using separated p u t q (Ne.symm hqt) i j hmi hm 0 c
        · simpa using separated p u t q (Ne.symm hqt) i j hmi hm 1 c
      exact (ComplexMemory.read_of_mem (by simpa using hpast 0) (by simpa using hpast 1)).trans
        (h.outputs q hq' hql j hm)
  · intro r o ho
    have hcurrent : r≠p.gx ∨ ∀ j, mask p u j=Bool.true →
        offset p u t j*2≠o ∧ offset p u t j*2+1≠o := by
      rcases ho with hr | ho
      · exact Or.inl hr
      · exact Or.inr (fun j hj => ho t (by dsimp [t];omega) (by dsimp [t];omega) j hj)
    have hpast : Outside p u length k r o := by
      rcases ho with hr | ho
      · exact Or.inl hr
      · exact Or.inr (fun q hq hql j hj => ho q (by omega) hql j hj)
    exact (ComplexBackwardStep.result_frame vt p.s p.y p.gx p.gy p.batch p.dim t (columns p u)
      (mask p u) (carry p u length k) (acc p u length k) (transition p u) r o hcurrent).trans
        (h.frame r o hpast)

theorem run_loop (p : Layout) (u : BlockState) (length : Nat) (sep : ReadSeparation p) :
    ∃ v, stepStmt (.forLoop "i" length
      (ComplexBackwardSource.body p.s p.y p.gx p.gy length p.batch p.dim p.B))
      (ComplexBackwardInit.result u p.l p.batch p.dim p.B) = some v ∧ Invariant p u length length v := by
  exact forLoop_inv (idx := "i") (n := length)
    (body := ComplexBackwardSource.body p.s p.y p.gx p.gy length p.batch p.dim p.B)
    (P := Invariant p u length)
    (s_init := ComplexBackwardInit.result u p.l p.batch p.dim p.B)
    (initial_invariant p u length) (fun k v hk h => step p u length k hk sep v h)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DiagSsmTriton.ComplexBackwardLoop
