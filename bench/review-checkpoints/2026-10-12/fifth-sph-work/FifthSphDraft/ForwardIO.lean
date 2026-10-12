import FifthSphDraft.ForwardReal
import FifthSphDraft.Candidate
import VeriTile.Triton.Memory.Denotation
namespace VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.ForwardIO
open VeriTile VeriTile.Triton TypedSource
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 30000

def coord (pid B j c : Nat) := j*3+B*3*pid+c
def outputAddress (pid B stride off j c : Nat) := j*stride+B*stride*pid+off+c

structure Bounds (X Y : RegionName) (B N M off stride : Nat) (u : BlockState) (bounds : RegionBounds) : Prop where
  coordinates : ∀ c : Fin 3,∀ j : Fin B,coord u.pid B j.val c.val<N → coord u.pid B j.val c.val<bounds X
  output : ∀ c : Fin 11,∀ j : Fin B,outputAddress u.pid B stride off j.val c.val<M → outputAddress u.pid B stride off j.val c.val<bounds Y

def Cover (B N M off stride : Nat) : Prop := ∀ pid,∀ j : Fin B,
  outputAddress pid B stride off j.val 0<M → coord pid B j.val 2<N

def Outside (A : FlatAlloc) (u : BlockState) (Y : RegionName) (B M off stride : Nat) (r : RegionName) (o : Nat) : Prop :=
  r≠A.flat ∨ ∀ c : Fin 11,∀ j : Fin B,outputAddress u.pid B stride off j.val c.val<M →
    o≠A.addr Y (outputAddress u.pid B stride off j.val c.val)

def Implements (kernel : Kernel) (X Y : RegionName) (B N M off stride : Nat) : Prop :=
  ∀ A : FlatAlloc,A.Disjoint → (∀ r,r∉A.regions → A.extent r=0) →
  ∀ u : BlockState,u.undef=(fun _ _ => 0) → Bounds X Y B N M off stride u A.extent →
  10<stride → Cover B N M off stride →
    ∃ v,exec (A.flattenKernel kernel) (A.flattenState u)=some v ∧
      (∀ c : Fin 11,∀ j : Fin B,outputAddress u.pid B stride off j.val c.val<M →
        v.readMem A.flat (A.addr Y (outputAddress u.pid B stride off j.val c.val))=
          RealMath.output c (u.readMem X (coord u.pid B j.val 0))
            (u.readMem X (coord u.pid B j.val 1)) (u.readMem X (coord u.pid B j.val 2))) ∧
      (∀ r o,Outside A u Y B M off stride r o → v.mem r o=(A.flattenState u).mem r o)

theorem polynomial_eq (c : Fin 11) (x y z : ℝ) : ForwardReal.sphY c x y z=RealMath.output c x y z := by
  fin_cases c <;> norm_num [ForwardReal.sphY,RealMath.output] <;> ring

theorem canonical_correct (X Y : RegionName) (B N M off stride : Nat) :
    Spec.Real (Implements (ForwardReal.fifth_order_fwd_surface X Y B N M off stride).toAlgKernel X Y B N M off stride) := by
  intro A hd ha u hu hb hstride hcover
  have hs := ForwardReal.fifth_order_fwd_surface_traceSafe X Y B N M off stride A.extent u
    (by intro c j hj;simpa only [ForwardReal.sphInWin_eq] using hb.coordinates c j (by simpa only [ForwardReal.sphInWin_eq] using hj))
    (by intro c j hj;simpa only [ForwardReal.sphOutWin_eq] using hb.output c j (by simpa only [ForwardReal.sphOutWin_eq] using hj))
  have hok := ForwardReal.fifth_order_fwd_surface_flattenOk X Y B N M off stride
  obtain ⟨v,he,hv,hf⟩ := ForwardReal.fifth_order_fwd_surface_region_run X Y B N M off stride hstride
    (by simpa only [Cover,outputAddress,coord,Nat.add_zero] using hcover)
    u (fun c : Fin 3 => fun j : Fin B => u.readMem X (ForwardReal.sphInWin B u.pid c j)) (fun _ _ _ => rfl)
  have hbridge := A.exec_flatten hd ha _ u hs hok hu
  have hmem (r : RegionName) (o : Nat) (ho : o<A.extent r) : r∈A.regions := by
    by_contra hn
    rw [ha r hn] at ho
    omega
  refine ⟨A.flattenState v,?_,?_,?_⟩
  · rw [hbridge,he,Option.map_some]
  · intro c j hj
    rw [A.flattenState_readMem hd v (hmem Y _ (hb.output c j hj)) (hb.output c j hj)]
    have h := hv c j (by simpa only [ForwardReal.sphOutWin_eq] using hj)
    simpa only [ForwardReal.sphOutWin_eq,ForwardReal.sphInWin_eq,coord,outputAddress,polynomial_eq] using h
  · intro r o ho
    by_cases hr : r=A.flat
    · subst r
      simp only [FlatAlloc.flattenState]
      unfold FlatAlloc.readFlat
      cases hdec : A.decode o with
      | none => rfl
      | some pair =>
        obtain ⟨r,z⟩ := pair
        obtain ⟨_,hz,_⟩ := A.decode_sound hdec
        show A.trCell (v.mem r z)=A.trCell (u.mem r z)
        apply congrArg A.trCell
        apply hf r z
        intro c j hj
        by_cases hr : r=Y
        · subst r
          right
          intro heq
          have houtside := ho.elim (fun hn => False.elim (hn rfl)) (fun hn => hn c j (by simpa only [ForwardReal.sphOutWin_eq] using hj))
          apply houtside
          simpa only [heq,ForwardReal.sphOutWin_eq] using hz
        · exact Or.inl hr
    · simp only [FlatAlloc.flattenState,if_neg hr]

theorem typed_correct (input output : Storage) (X Y : RegionName) (B N M off stride : Nat) :
    Spec.Real (Implements (forward input output Bool.false X Y B N M off stride).eraseDType.toAlgKernel X Y B N M off stride) := by
  rw [Erasure.forward_body]
  exact canonical_correct X Y B N M off stride

theorem candidate_correct (input output : Storage) (X Y : RegionName) (B N M off stride : Nat) :
    Spec.Real (Implements (forward input output Bool.true X Y B N M off stride).eraseDType.toAlgKernel X Y B N M off stride) := by
  rw [Erasure.forward_body]
  intro A hd ha u hu hb hstride hcover
  obtain ⟨v,he,hv,hf⟩ := canonical_correct X Y B N M off stride A hd ha u hu hb hstride hcover
  exact ⟨v,(Candidate.flat_exec A .forward X Y X B N M off stride (A.flattenState u)).symm.trans he,hv,hf⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.ForwardIO
