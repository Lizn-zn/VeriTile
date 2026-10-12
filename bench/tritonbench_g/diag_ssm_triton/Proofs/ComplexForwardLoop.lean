import bench.tritonbench_g.diag_ssm_triton.Proofs.ComplexForwardInit
import VeriTile.Triton.KernelLemmas.LoopInvariant
import VeriTile.Triton.Semantics.ProgramRewrite
namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton.ComplexForwardLoop
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 20000

abbrev Layout := ForwardLayout
abbrev Lane (p : Layout) := TileIndex [p.B]

def columns (p : Layout) (u : BlockState) : Lane p → Nat := ComplexForwardInit.columns u p.B
def mask (p : Layout) (u : BlockState) : Lane p → Bool := ComplexForwardInit.active u p.batch p.dim p.B
def offset (p : Layout) (u : BlockState) (t : Nat) : Lane p → Nat :=
  ComplexForwardStep.offset p.batch p.dim t (columns p u)
def initial (p : Layout) (u : BlockState) : Lane p → ℂ := ComplexForwardInit.initial u p.s p.batch p.dim p.B
def transition (p : Layout) (u : BlockState) : Lane p → ℂ := ComplexForwardInit.transition u p.l p.batch p.dim p.B
def input (p : Layout) (u : BlockState) (t : Nat) : Lane p → ℂ :=
  ComplexForwardStep.input u p.x (offset p u t) (mask p u)

/-- Independent recurrence, with input windows read from the initial
memory. It does not refer to execution, registers or intermediate outputs. -/
def state (p : Layout) (u : BlockState) (t : Nat) : Lane p → ℂ := fun j =>
  Reference.forwardState (initial p u j) (transition p u j) (fun t => input p u t j) t

/-- Cells outside all active output components written before k. -/
def Outside (p : Layout) (u : BlockState) (k : Nat) (r : RegionName) (o : Nat) : Prop :=
  r ≠ p.y ∨ ∀ t, t<k → ∀ j, mask p u j = Bool.true →
    offset p u t j*2 ≠ o ∧ offset p u t j*2+1 ≠ o

structure Invariant (p : Layout) (u : BlockState) (k : Nat) (v : BlockState) : Prop where
  registers : ComplexForwardStep.Registers v (columns p u) (mask p u) (state p u k) (transition p u)
  pids : v.pids = u.pids
  outputs : ∀ t, t<k → ∀ j, mask p u j = Bool.true →
    ComplexMemory.read v p.y (offset p u t j) = state p u (t+1) j
  frame : ∀ r o, Outside p u k r o → v.mem r o = u.mem r o

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

/-- Earlier writes cannot alter an input component of a future row, even
when input and output name the same buffer. -/
theorem outside_future (p : Layout) (u : BlockState) (k t : Nat) (hkt : k≤t)
    (j : Lane p) (hj : mask p u j=Bool.true) (c : Fin 2) (r : RegionName) :
    Outside p u k r (offset p u t j*2+c.val) := by
  right
  intro q hq i hi
  constructor
  · simpa using separated p u q t (by omega) i j hi hj 0 c
  · simpa using separated p u q t (by omega) i j hi hj 1 c

theorem input_unchanged {p : Layout} {u v : BlockState} {k : Nat}
    (h : Invariant p u k v) (t : Nat) (hkt : k≤t) :
    ComplexForwardStep.input v p.x (offset p u t) (mask p u) = input p u t := by
  funext j
  by_cases hm : mask p u j=Bool.true
  · have h0 := h.frame p.x (offset p u t j*2)
      (by simpa using outside_future p u k t hkt j hm 0 p.x)
    have h1 := h.frame p.x (offset p u t j*2+1)
      (by simpa using outside_future p u k t hkt j hm 1 p.x)
    apply Complex.ext
    · simp only [input,ComplexForwardStep.input,hm,↓reduceIte]
      unfold BlockState.readMem
      rw [h0]
    · simp only [input,ComplexForwardStep.input,hm,↓reduceIte]
      unfold BlockState.readMem
      rw [h1]
  · simp [input,ComplexForwardStep.input,hm]

theorem initial_invariant (p : Layout) (u : BlockState) :
    Invariant p u 0 (ComplexForwardInit.result u p.s p.l p.batch p.dim p.B) := by
  refine ⟨?_,rfl,?_,?_⟩
  · exact ComplexForwardInit.result_registers u p.s p.l p.batch p.dim p.B
  · intro t ht; omega
  · intro r o _; rfl


/-- One source iteration preserves the mathematical recurrence, every
previous output, and the complete write footprint. -/
theorem step (p : Layout) (u : BlockState) (k : Nat) (v : BlockState)
    (h : Invariant p u k v) :
    ∃ w, stepStmts (ComplexForwardSource.body p.x p.y p.batch p.dim p.B)
        (v.setReg "t" .nat [] (Tile.scalar k)) = some w ∧ Invariant p u (k+1) w := by
  let vt := v.setReg "t" .nat [] (Tile.scalar k)
  let w := ComplexForwardStep.result vt p.x p.y (offset p u k) (mask p u)
    (state p u k) (transition p u)
  have ht := h.registers.setCounter k
  have hexec := ComplexForwardStep.body_step p.x p.y p.batch p.dim p.B k vt
    (columns p u) (mask p u) (state p u k) (transition p u)
    (by simp [vt]) ht.columns ht.active ht.realState ht.imagState ht.realTransition ht.imagTransition
  have hinput : ComplexForwardStep.input vt p.x (offset p u k) (mask p u) = input p u k :=
    input_unchanged h k (le_refl k)
  have hnext : ComplexForwardStep.next vt p.x (offset p u k) (mask p u)
      (state p u k) (transition p u) = state p u (k+1) := by
    funext j
    unfold ComplexForwardStep.next
    rw [hinput]
    rfl
  have hreg := ComplexForwardStep.result_registers vt p.x p.y (offset p u k)
    (columns p u) (mask p u) (state p u k) (transition p u) ht
  rw [hnext] at hreg
  have hi : Function.Injective (offset p u k) := by
    simpa only [offset,ComplexForwardStep.offset,columns,ComplexForwardInit.columns,Nat.add_assoc] using
      ComplexMemory.lane_injective p.B (k*(p.batch*p.dim)+u.pids 0*p.B)
  refine ⟨w,hexec,hreg,?_,?_,?_⟩
  · exact (ComplexForwardStep.result_pids _ _ _ _ _ _ _).trans h.pids
  · intro t htk j hm
    by_cases htk' : t=k
    · subst t
      have hv := ComplexForwardStep.result_read vt p.x p.y (offset p u k)
        (mask p u) (state p u k) (transition p u) hi j hm
      rw [hnext] at hv
      exact hv
    · have htk0 : t<k := by omega
      have hp (c : Fin 2) : w.mem p.y (offset p u t j*2+c.val) =
          v.mem p.y (offset p u t j*2+c.val) := by
        apply ComplexForwardStep.result_frame
        right
        intro i hmi
        constructor
        · simpa using separated p u k t (Ne.symm htk') i j hmi hm 0 c
        · simpa using separated p u k t (Ne.symm htk') i j hmi hm 1 c
      exact (ComplexMemory.read_of_mem (by simpa using hp 0) (by simpa using hp 1)).trans
        (h.outputs t htk0 j hm)
  · intro r o ho
    have hcurrent : r ≠ p.y ∨ ∀ j, mask p u j=Bool.true →
        offset p u k j*2 ≠ o ∧ offset p u k j*2+1 ≠ o := by
      rcases ho with hr | ho
      · exact Or.inl hr
      · exact Or.inr (fun j hj => ho k (Nat.lt_succ_self k) j hj)
    have hpast : Outside p u k r o := by
      rcases ho with hr | ho
      · exact Or.inl hr
      · exact Or.inr (fun t ht j hj => ho t (by omega) j hj)
    exact (ComplexForwardStep.result_frame vt p.x p.y (offset p u k)
      (mask p u) (state p u k) (transition p u) r o hcurrent).trans (h.frame r o hpast)

/-- Whole-source real execution for every sequence length. This is a
regional-memory result; finite allocation and address safety are discharged
separately when forming the public flattened IO contract. -/
theorem run (p : Layout) (u : BlockState) (length : Nat) :
    ∃ v, exec (Erasure.complexForward Bool.false p.s p.x p.y p.l length p.batch p.dim p.B).toAlgKernel u = some v ∧
      Invariant p u length v := by
  obtain ⟨v,he,hv⟩ := forLoop_inv (idx := "t") (n := length)
    (body := ComplexForwardSource.body p.x p.y p.batch p.dim p.B)
    (P := Invariant p u)
    (s_init := ComplexForwardInit.result u p.s p.l p.batch p.dim p.B)
    (initial_invariant p u) (fun k v _ h => step p u k v h)
  refine ⟨v,?_,hv⟩
  unfold exec
  rw [ComplexForwardSource.source_body]
  simp only [ProgramRewrite.append,ComplexForwardInit.prefix_step,Option.bind_some,stepStmts,he]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DiagSsmTriton.ComplexForwardLoop
