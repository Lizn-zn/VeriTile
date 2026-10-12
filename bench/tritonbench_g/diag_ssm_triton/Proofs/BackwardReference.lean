import bench.tritonbench_g.diag_ssm_triton.Proofs.RealReference
import bench.tritonbench_g.diag_ssm_triton.Proofs.Reference
namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton.BackwardReference
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 20000

def extend {T : Nat} (x : Fin T → ℝ) (n : Nat) : ℝ :=
  if h : n<T then x ⟨n,h⟩ else 0

theorem carry_reference {T : Nat} (gy : Fin T → ℝ) (l : ℝ) (k : Nat) :
    RealReference.dssmbStreamGradS gy l k = Reference.reverseCarry T l (extend gy) k := by
  induction k with
  | zero => rfl
  | succ k ih => simp [RealReference.dssmbStreamGradS,Reference.reverseCarry,RealReference.reverseTime,extend,ih]

/-- The y stream is already windowed at the previous time by the IO reader.
Undo that windowing to use the independent mathematical previous-state formula. -/
theorem previous_reference {T : Nat} (ys ss : Fin T → ℝ) (k : Nat) :
    RealReference.dssmbStreamPrev ys ss k =
      Reference.previousState (extend ss 0) (fun t => extend ys (t+1)) (T-1-k) := by
  unfold RealReference.dssmbStreamPrev RealReference.reverseTime Reference.previousState extend
  by_cases hT : 0<T
  · have hrt : T-1-k<T := by omega
    rw [dif_pos hrt]
    by_cases hz : T-1-k=0
    · simp [hz,hT]
    · have hp : 0<T-1-k := by omega
      have hidx : T-1-k-1+1=T-1-k := by omega
      simp [hz,hp,hidx,hrt]
  · have hz : T=0 := by omega
    subst T
    simp

theorem weight_reference {T : Nat} (gy ys ss : Fin T → ℝ) (l : ℝ) (k : Nat) :
    RealReference.dssmbStreamGradLambda gy ys ss l k =
      Reference.transitionGradient T l (extend ss 0) (fun t => extend ys (t+1)) (extend gy) k := by
  induction k with
  | zero => rfl
  | succ k ih =>
    simp [RealReference.dssmbStreamGradLambda,Reference.transitionGradient,ih,carry_reference,
      previous_reference,RealReference.reverseTime,extend]

/-- Three outputs of the reverse recurrence: grad_x at the requested time,
the final state adjoint, and the per-column transition adjoint. -/
def gradients {T B : Nat} (xs : Fin 4 → Fin T → Fin B → ℝ)
    (o : Fin 3) (t : Fin T) (j : Fin B) : ℝ :=
  let l := xs 3 t j
  let gy := extend (fun u => xs 0 u j)
  let previous := fun u => extend (fun v => xs 1 v j) (u+1)
  let initial := extend (fun u => xs 2 u j) 0
  match o with
  | ⟨0,_⟩ => Reference.inputGradient T l gy t.val
  | ⟨1,_⟩ => Reference.reverseCarry T l gy T
  | ⟨_+2,_⟩ => Reference.transitionGradient T l initial previous gy T

theorem gradients_reference {T B : Nat} (xs : Fin 4 → Fin T → Fin B → ℝ)
    (o : Fin 3) (t : Fin T) (j : Fin B) :
    RealReference.diagSsmBackwardStreamSpec xs o t j = gradients xs o t j := by
  obtain ⟨o,ho⟩ := o
  rcases o with _|_|o
  all_goals simp [RealReference.diagSsmBackwardStreamSpec,gradients,Reference.inputGradient,
    carry_reference,weight_reference,extend,t.isLt]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DiagSsmTriton.BackwardReference
