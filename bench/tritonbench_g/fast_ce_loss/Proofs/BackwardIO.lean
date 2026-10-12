import bench.tritonbench_g.fast_ce_loss.Proofs.BackwardReal
import bench.tritonbench_g.fast_ce_loss.Proofs.Candidate
import VeriTile.Triton.Memory.Denotation
namespace VeriTile.Bench.TritonBenchG.FastCeLoss.BackwardIO
open VeriTile VeriTile.Triton TypedSource BackwardReal
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 20000

structure Bounds (X D Z : RegionName) (L : Region .int) (V R DR B : Nat)
    (u : BlockState) (bounds : RegionBounds) : Prop where
  label : u.pids 0 < bounds (Region.cast L)
  labelTyped : ∃ lab : Int, u.mem (Region.cast L) (u.pids 0)=MemCell.of .int lab
  labelValid : let lab := u.readMemValue .int (Region.cast L) (u.pids 0)
    lab = -100 ∨ (0≤lab ∧ lab < (V : Int))
  lse : u.pids 0 < bounds Z
  upstream : u.readMemValue .int (Region.cast L) (u.pids 0) ≠ -100 → u.pids 0*DR < bounds D
  logits : ∀ j : Fin B, col u B j<V → address u R B j < bounds X

def Outside (A : FlatAlloc) (u : BlockState) (X : RegionName) (V R B : Nat)
    (r : RegionName) (o : Nat) : Prop :=
  r≠A.flat ∨ ∀ j : Fin B, col u B j<V → o≠A.addr X (address u R B j)

def Implements (kernel : Kernel) (cap scale : Bool) (X D Z : RegionName) (L : Region .int)
    (V R DR B : Nat) (SC SL : ℝ) : Prop :=
  (cap=Bool.true → SC≠0) →
  ∀ A : FlatAlloc, A.Disjoint → (∀ r, r ∉ A.regions → A.extent r=0) →
  ∀ u : BlockState, u.undef=(fun _ _ => 0) → Bounds X D Z L V R DR B u A.extent →
    ∃ v, exec (A.flattenKernel kernel) (A.flattenState u)=some v ∧
      (∀ j : Fin B, col u B j<V → v.readMem A.flat (A.addr X (address u R B j)) =
        gradient cap scale SC SL (u.readMemValue .int (Region.cast L) (u.pids 0))
          (col u B j) (u.readMem X (address u R B j))
          (u.readMem Z (u.pids 0)) (u.readMem D (u.pids 0*DR))) ∧
      (∀ r o, Outside A u X V R B r o → v.mem r o=(A.flattenState u).mem r o)

theorem canonical_correct (cap scale : Bool) (X D Z : RegionName) (L : Region .int)
    (V R DR B : Nat) (SC SL : ℝ) :
    Spec.Real (Implements (original cap scale X D Z L V R DR B SC SL)
      cap scale X D Z L V R DR B SC SL) := by
  intro hSC A hd hc u hu hb
  have hs := trace_safe cap scale X D Z L V R DR B SC SL A.extent u hb.label hb.lse hb.upstream hb.logits
  have hok := flatten_ok cap scale X D Z L V R DR B SC SL
  obtain ⟨v,he,hv,hf⟩ := run cap scale X D Z L V R DR B SC SL u
  have hbridge := A.exec_flatten hd hc _ u hs hok hu
  refine ⟨A.flattenState v,?_,?_,?_⟩
  · rw [hbridge,he,Option.map_some]
  · intro j hj
    have hx := hb.logits j hj
    have hr : X ∈ A.regions := by
      by_contra hn
      rw [hc X hn] at hx
      omega
    rw [A.flattenState_readMem hd v hr hx]
    exact hv j hj
  · intro r o ho
    by_cases hr : r=A.flat
    · subst r
      simp only [FlatAlloc.flattenState]
      unfold FlatAlloc.readFlat
      cases hdcd : A.decode o with
      | none => rfl
      | some pair =>
        obtain ⟨q,z⟩ := pair
        obtain ⟨_,hz,_⟩ := A.decode_sound hdcd
        show A.trCell (v.mem q z)=A.trCell (u.mem q z)
        apply congrArg A.trCell (hf q z ?_)
        intro j hj ⟨hx,ha⟩
        subst q
        rcases ho with hn | hn
        · exact hn rfl
        · exact hn j hj (by rw [ha]; exact hz)
    · simp only [FlatAlloc.flattenState,if_neg hr]

theorem typed_correct (storage : Storage) (cap scale : Bool) (X D Z : RegionName) (L : Region .int)
    (V R DR B : Nat) (SC SL : ℝ) :
    Spec.Real (Implements
      (backward storage cap scale Bool.false X D Z L V R DR B SC SL).eraseDType.toAlgKernel
      cap scale X D Z L V R DR B SC SL) := by
  rw [Erasure.backward_body]
  exact canonical_correct cap scale X D Z L V R DR B SC SL

theorem candidate_correct (storage : Storage) (cap scale : Bool) (X D Z : RegionName) (L : Region .int)
    (V R DR B : Nat) (SC SL : ℝ) :
    Spec.Real (Implements
      (backward storage cap scale Bool.true X D Z L V R DR B SC SL).eraseDType.toAlgKernel
      cap scale X D Z L V R DR B SC SL) := by
  rw [Erasure.backward_body]
  intro hSC A hd hc u hu hb
  obtain ⟨v,he,hv,hf⟩ := canonical_correct cap scale X D Z L V R DR B SC SL hSC A hd hc u hu hb
  refine ⟨v,?_,hv,hf⟩
  have hcand := Candidate.flat_exec A .backward cap scale X D Z L V R B 0 DR SC SL (A.flattenState u)
  exact hcand.symm.trans he

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FastCeLoss.BackwardIO
