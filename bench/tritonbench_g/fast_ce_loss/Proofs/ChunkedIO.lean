import bench.tritonbench_g.fast_ce_loss.Proofs.ChunkedSafety
import bench.tritonbench_g.fast_ce_loss.Proofs.Candidate
import VeriTile.Triton.Memory.Denotation
namespace VeriTile.Bench.TritonBenchG.FastCeLoss.ChunkedIO
open VeriTile VeriTile.Triton TypedSource ChunkedReal
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 20000

structure Bounds (X Y Z : RegionName) (L : Region .int) (V C R B : Nat)
    (u : BlockState) (bounds : RegionBounds) : Prop where
  label : u.pids 0 < bounds (Region.cast L)
  labelTyped : ∃ lab : Int, u.mem (Region.cast L) (u.pids 0)=MemCell.of .int lab
  labelValid : let lab := u.readMemValue .int (Region.cast L) (u.pids 0)
    lab = -100 ∨ (0≤lab ∧ lab < (V : Int))
  loss : u.pids 1=0 → u.pids 0 < bounds Y
  lse : u.pids 1=0 → u.pids 0*C < bounds Z
  gather : u.pids 1=0 → u.readMemValue .int (Region.cast L) (u.pids 0) ≠ -100 →
    u.pids 0*R+(u.readMemValue .int (Region.cast L) (u.pids 0)).toNat < bounds X
  logits : ∀ j : Fin B, u.pids 1*B+j.val<V → u.pids 0*R+(u.pids 1*B+j.val) < bounds X

def Outside (A : FlatAlloc) (u : BlockState) (Y Z : RegionName) (C : Nat)
    (r : RegionName) (o : Nat) : Prop :=
  r≠A.flat ∨ (o≠A.addr Y (u.pids 0) ∧ o≠A.addr Z (u.pids 0*C))

def Outputs (A : FlatAlloc) (u v : BlockState) (cap scale : Bool)
    (X Y Z : RegionName) (L : Region .int) (V C R B : Nat) (SC SL : ℝ) : Prop :=
  if u.pids 1=0 then
    v.readMem A.flat (A.addr Z (u.pids 0*C)) = RealMath.lse cap scale SC SL V 0
      (fun j : Fin B => u.readMem X (u.pids 0*R+j.val)) ∧
    v.readMem A.flat (A.addr Y (u.pids 0)) = partialLoss cap scale SC SL
      (u.readMemValue .int (Region.cast L) (u.pids 0))
      (u.readMem X (u.pids 0*R+(u.readMemValue .int (Region.cast L) (u.pids 0)).toNat)) ∧
    (∀ r o, Outside A u Y Z C r o → v.mem r o=(A.flattenState u).mem r o)
  else v.mem=(A.flattenState u).mem

/-- Conditional source behavior: only chunk zero writes its partial loss and LSE. -/
def Implements (kernel : Kernel) (cap scale : Bool) (X Y Z : RegionName) (L : Region .int)
    (V C R B : Nat) (SC SL : ℝ) : Prop :=
  (cap=Bool.true → 0<SC) → (scale=Bool.true → 0<SL) →
  ∀ A : FlatAlloc, A.Disjoint → (∀ r, r ∉ A.regions → A.extent r=0) →
  ∀ u : BlockState, u.undef=(fun _ _ => 0) → Bounds X Y Z L V C R B u A.extent →
  0<V → 0<B → Y≠Z →
    ∃ v, exec (A.flattenKernel kernel) (A.flattenState u)=some v ∧
      Outputs A u v cap scale X Y Z L V C R B SC SL

theorem canonical_correct (cap scale : Bool) (X Y Z : RegionName) (L : Region .int)
    (V C R B : Nat) (SC SL : ℝ) :
    Spec.Real (Implements (original cap scale X Y Z L V C R B SC SL)
      cap scale X Y Z L V C R B SC SL) := by
  intro hSC hSL A hd hc u hu hb hv hB hne
  have hs := trace_safe cap scale X Y Z L V C R B SC SL A.extent u hB hb.label hb.loss hb.lse hb.gather hb.logits
  have hok := flatten_ok cap scale X Y Z L V C R B SC SL
  have hbridge := A.exec_flatten hd hc _ u hs hok hu
  by_cases hcz : u.pids 1=0
  · obtain ⟨v,he,hz,hy,hf⟩ := zero_run cap scale X Y Z L V C R B SC SL u hB hv hne hcz
    have hmem (q : RegionName) (o : Nat) (hq : o<A.extent q) : q∈A.regions := by
      by_contra hn
      rw [hc q hn] at hq
      omega
    refine ⟨A.flattenState v,?_,?_⟩
    · rw [hbridge,he,Option.map_some]
    · simp only [Outputs,if_pos hcz]
      refine ⟨?_,?_,?_⟩
      · rw [A.flattenState_readMem hd v (hmem Z _ (hb.lse hcz)) (hb.lse hcz)]
        exact hz
      · rw [A.flattenState_readMem hd v (hmem Y _ (hb.loss hcz)) (hb.loss hcz)]
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
  · obtain ⟨v,he,hm⟩ := later_run cap scale X Y Z L V C R B SC SL u hB hcz
    refine ⟨A.flattenState v,?_,?_⟩
    · rw [hbridge,he,Option.map_some]
    · simp only [Outputs,if_neg hcz,FlatAlloc.flattenState,FlatAlloc.readFlat,hm]

theorem typed_correct (storage : Storage) (cap scale : Bool) (X Y Z : RegionName) (L : Region .int)
    (V C R B : Nat) (SC SL : ℝ) :
    Spec.Real (Implements
      (chunked storage cap scale Bool.false X Y Z L V C R B SC SL).eraseDType.toAlgKernel
      cap scale X Y Z L V C R B SC SL) := by
  rw [Erasure.chunked_body]
  exact canonical_correct cap scale X Y Z L V C R B SC SL

theorem candidate_correct (storage : Storage) (cap scale : Bool) (X Y Z : RegionName) (L : Region .int)
    (V C R B : Nat) (SC SL : ℝ) :
    Spec.Real (Implements
      (chunked storage cap scale Bool.true X Y Z L V C R B SC SL).eraseDType.toAlgKernel
      cap scale X Y Z L V C R B SC SL) := by
  rw [Erasure.chunked_body]
  intro hSC hSL A hd hc u hu hb hv hB hne
  obtain ⟨v,he,hout⟩ := canonical_correct cap scale X Y Z L V C R B SC SL hSC hSL A hd hc u hu hb hv hB hne
  refine ⟨v,?_,hout⟩
  have hcand := Candidate.flat_exec A .chunked cap scale X Y Z L V R B C 0 SC SL (A.flattenState u)
  exact hcand.symm.trans he

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FastCeLoss.ChunkedIO
