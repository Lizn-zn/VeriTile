import bench.tritonbench_g.chunk_delta_fwd.Proofs.RealValues

/-! Independent matrix recurrence with a window width and a stride. The source
uses stride BT and width ceil(BT/BC)*BC. These may differ: BC not dividing BT
makes adjacent update windows overlap. Corrected rows can be overwritten by
later windows. At width=BT the recurrence is the usual disjoint-chunk formula. -/
namespace VeriTile.Bench.TritonBenchG.ChunkDeltaFwd.RealWindow
open VeriTile.Triton RealProgram
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

/-- Width and stride are independent mathematical parameters. -/
def state (xs : RealMath.Inputs) (T K stride width : Nat) (useSeed : Bool) :
    Nat → Nat → Nat → ℝ
  | 0, e, j => if useSeed then xs.seed e j else 0
  | c + 1, e, j =>
      state xs T K stride width useSeed c e j +
        ∑ t : Fin width, if c * stride + t.val < T then
          xs.key (c * stride + t.val) e *
            RealMath.corrected xs K (state xs T K stride width useSeed c) (c * stride + t.val) j
        else 0

theorem nonoverlapping (xs : RealMath.Inputs) (T K BT : Nat) (useSeed : Bool) (c : Nat) :
    state xs T K BT BT useSeed c = RealMath.state xs T K BT useSeed c := by
  induction c with
  | zero => rfl
  | succ c ih =>
    funext e j
    simp only [state, RealMath.state, ih]

def span (p : Layout) : Nat := RealInnerLoop.count p * p.BC

theorem span_ge (p : Layout) (hBC : 0 < p.BC) : p.BT ≤ span p := by
  have h := Nat.mod_add_div (p.BT + p.BC - 1) p.BC
  have hm := Nat.mod_lt (p.BT + p.BC - 1) hBC
  rw [Nat.mul_comm p.BC] at h
  dsimp [span, RealInnerLoop.count]
  omega

theorem span_eq (p : Layout) (hBC : 0 < p.BC) (hd : p.BC ∣ p.BT) : span p = p.BT :=
  RealMath.complete_inner_span p.BT p.BC hBC hd

/-- All rows actually visited by the source contribute, with the global T mask. -/
theorem accumulated_eq (p : Layout) (l : Launch) (a : Inputs) (seed current : Nat → Nat → ℝ)
    (outer : Nat) (frozen acc : StateValue p) (hk : l.key = 0) (hK : p.K ≤ p.BK)
    (hf : RealValues.Represents p l frozen current) (e : Fin p.BK) (j : Fin p.BV)
    (he : e.val < p.K) (hj : valueIndex p l j.val < p.V) :
    RealInnerLoop.accumulated p l a outer (RealInnerLoop.count p) frozen acc e j =
      acc e j + ∑ t : Fin (span p), if outer * p.BT + t.val < p.T then
        (RealValues.arrays p l a seed).key (outer * p.BT + t.val) e.val *
          RealMath.corrected (RealValues.arrays p l a seed) p.K current
            (outer * p.BT + t.val) (valueIndex p l j.val)
      else 0 := by
  rw [RealInnerLoop.accumulated_flat]
  congr 1
  apply Finset.sum_congr rfl
  intro t _
  by_cases ht : outer * p.BT + t.val < p.T
  · rw [RealValues.corrected_eq p l a seed current frozen hk hK hf _ j ⟨ht,hj⟩]
    simp only [keyValue, keyIndex, hk, Nat.zero_mul, Nat.zero_add, he, ht, and_self,
      ite_true, keyAddress, RealValues.arrays]
  · simp only [keyValue, ht, and_false, ite_false, zero_mul]

/-- Source carry update after the entire inner loop. -/
def advance (p : Layout) (l : Launch) (a : Inputs) (outer : Nat)
    (frozen : StateValue p) : StateValue p := fun e j =>
  frozen e j + RealInnerLoop.accumulated p l a outer (RealInnerLoop.count p) frozen (fun _ _ => 0) e j

def expected (p : Layout) (l : Launch) (a : Inputs) (seed : Nat → Nat → ℝ) (useSeed : Bool)
    (outer : Nat) : Nat → Nat → ℝ :=
  state (RealValues.arrays p l a seed) p.T p.K p.BT (span p) useSeed outer

theorem advance_represents (p : Layout) (l : Launch) (a : Inputs) (seed : Nat → Nat → ℝ)
    (useSeed : Bool) (outer : Nat) (frozen : StateValue p)
    (hk : l.key = 0) (hK : p.K ≤ p.BK)
    (hf : RealValues.Represents p l frozen (expected p l a seed useSeed outer)) :
    RealValues.Represents p l (advance p l a outer frozen) (expected p l a seed useSeed (outer + 1)) := by
  intro e j he hj
  rw [advance, accumulated_eq p l a seed _ outer frozen (fun _ _ => 0) hk hK hf e j he hj]
  simp only [zero_add, hf e j he hj, expected, state]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkDeltaFwd.RealWindow
