import bench.tritonbench_g.fast_ce_loss.Proofs.ForwardSafety
import bench.tritonbench_g.fast_ce_loss.Proofs.Candidate
import VeriTile.Triton.Memory.Denotation
namespace VeriTile.Bench.TritonBenchG.FastCeLoss.ForwardIO
open VeriTile VeriTile.Triton TypedSource ForwardReal
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 20000

structure Bounds (X Y Z : RegionName) (L : Region .int) (V R B : Nat)
    (u : BlockState) (bounds : RegionBounds) : Prop where
  label : u.pids 0 < bounds (Region.cast L)
  labelTyped : ∃ lab : Int, u.mem (Region.cast L) (u.pids 0)=MemCell.of .int lab
  labelValid : let lab := u.readMemValue .int (Region.cast L) (u.pids 0)
    lab = -100 ∨ (0≤lab ∧ lab < (V : Int))
  loss : u.pids 0 < bounds Y
  lse : u.pids 0 < bounds Z
  gather : u.readMemValue .int (Region.cast L) (u.pids 0) ≠ -100 →
    u.pids 0*R+(u.readMemValue .int (Region.cast L) (u.pids 0)).toNat < bounds X
  logits : ∀ j : Fin B, j.val<V → u.pids 0*R+j.val < bounds X

def Outside (A : FlatAlloc) (u : BlockState) (Y Z : RegionName)
    (r : RegionName) (o : Nat) : Prop :=
  r≠A.flat ∨ (o≠A.addr Y (u.pids 0) ∧ o≠A.addr Z (u.pids 0))

def Implements (kernel : Kernel) (cap scale : Bool) (X Y Z : RegionName) (L : Region .int)
    (V R B : Nat) (SC SL : ℝ) : Prop :=
  (cap=Bool.true → 0<SC) → (scale=Bool.true → 0<SL) →
  ∀ A : FlatAlloc, A.Disjoint → (∀ r, r ∉ A.regions → A.extent r=0) →
  ∀ u : BlockState, u.undef=(fun _ _ => 0) → Bounds X Y Z L V R B u A.extent →
  0<V → V≤B → Y≠Z →
    ∃ v, exec (A.flattenKernel kernel) (A.flattenState u)=some v ∧
      v.readMem A.flat (A.addr Z (u.pids 0)) = RealMath.lse cap scale SC SL V 0
        (fun j : Fin B => u.readMem X (u.pids 0*R+j.val)) ∧
      v.readMem A.flat (A.addr Y (u.pids 0)) = RealMath.loss cap scale SC SL V
        (u.readMemValue .int (Region.cast L) (u.pids 0))
        (fun j : Fin B => u.readMem X (u.pids 0*R+j.val))
        (u.readMem X (u.pids 0*R+(u.readMemValue .int (Region.cast L) (u.pids 0)).toNat)) ∧
      (∀ r o, Outside A u Y Z r o → v.mem r o=(A.flattenState u).mem r o)

theorem canonical_correct (cap scale : Bool) (X Y Z : RegionName) (L : Region .int)
    (V R B : Nat) (SC SL : ℝ) :
    Spec.Real (Implements (original cap scale X Y Z L V R B SC SL)
      cap scale X Y Z L V R B SC SL) := by
  intro hSC hSL A hd hc u hu hb hv hB hne
  have hbpos : 0<B := lt_of_lt_of_le hv hB
  have hs := trace_safe cap scale X Y Z L V R B SC SL A.extent u hbpos hb.label hb.loss hb.lse hb.gather hb.logits
  have hok := flatten_ok cap scale X Y Z L V R B SC SL
  obtain ⟨v,he,hz,hy,hf⟩ := run cap scale X Y Z L V R B SC SL u hbpos hv hne
  have hbridge := A.exec_flatten hd hc _ u hs hok hu
  have hmem (q : RegionName) (hq : u.pids 0<A.extent q) : q∈A.regions := by
    by_contra hn
    rw [hc q hn] at hq
    omega
  refine ⟨A.flattenState v,?_,?_,?_,?_⟩
  · rw [hbridge,he,Option.map_some]
  · rw [A.flattenState_readMem hd v (hmem Z hb.lse) hb.lse]
    exact hz
  · rw [A.flattenState_readMem hd v (hmem Y hb.loss) hb.loss]
    exact hy
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
        apply congrArg A.trCell (hf q z ?_ ?_)
        · intro ⟨hq,hzval⟩
          subst q; subst z
          exact ho.elim (fun hn => hn rfl) (fun hn => hn.1 hz)
        · intro ⟨hq,hzval⟩
          subst q; subst z
          exact ho.elim (fun hn => hn rfl) (fun hn => hn.2 hz)
    · simp only [FlatAlloc.flattenState,if_neg hr]

theorem typed_correct (storage : Storage) (cap scale : Bool) (X Y Z : RegionName) (L : Region .int)
    (V R B : Nat) (SC SL : ℝ) :
    Spec.Real (Implements
      (forward storage cap scale Bool.false X Y Z L V R B SC SL).eraseDType.toAlgKernel
      cap scale X Y Z L V R B SC SL) := by
  rw [Erasure.forward_body]
  exact canonical_correct cap scale X Y Z L V R B SC SL

theorem candidate_correct (storage : Storage) (cap scale : Bool) (X Y Z : RegionName) (L : Region .int)
    (V R B : Nat) (SC SL : ℝ) :
    Spec.Real (Implements
      (forward storage cap scale Bool.true X Y Z L V R B SC SL).eraseDType.toAlgKernel
      cap scale X Y Z L V R B SC SL) := by
  rw [Erasure.forward_body]
  intro hSC hSL A hd hc u hu hb hv hB hne
  obtain ⟨v,he,hz,hy,hf⟩ := canonical_correct cap scale X Y Z L V R B SC SL hSC hSL A hd hc u hu hb hv hB hne
  refine ⟨v,?_,hz,hy,hf⟩
  have hcand := Candidate.flat_exec A .forward cap scale X Y Z L V R B 0 0 SC SL (A.flattenState u)
  exact hcand.symm.trans he

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FastCeLoss.ForwardIO
