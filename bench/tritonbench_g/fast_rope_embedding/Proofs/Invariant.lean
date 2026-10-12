import bench.tritonbench_g.fast_rope_embedding.Proofs.Readback
namespace VeriTile.Bench.TritonBenchG.FastRopeEmbedding.LoopInvariant
open VeriTile.Triton Code
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 25000

def Written (row QR H B firstHead cur o : Nat) : Prop :=
  ∃ t,firstHead≤t ∧ t<cur ∧ ∃ j : Fin B,j.val<H/2 ∧
    (o=Reference.first row QR H t j.val ∨ o=Reference.second row QR H t j.val)

structure Holds (s u : BlockState) (Q : RegionName) (row QR H B firstHead cur : Nat)
    (cs ss : Fin B → ℝ) : Prop where
  context : Context u row H B cs ss
  pids : u.pids=s.pids
  values : ∀ t,firstHead≤t → t<cur → ∀ j : Fin B,j.val<H/2 →
    u.readMem Q (Reference.first row QR H t j.val)=
      s.readMem Q (Reference.first row QR H t j.val)*cs j-s.readMem Q (Reference.second row QR H t j.val)*ss j ∧
    u.readMem Q (Reference.second row QR H t j.val)=
      s.readMem Q (Reference.second row QR H t j.val)*cs j+s.readMem Q (Reference.first row QR H t j.val)*ss j
  frame : ∀ r o,(r≠Q ∨ ¬Written row QR H B firstHead cur o) → u.mem r o=s.mem r o

theorem initial (s u : BlockState) (Q : RegionName) (row QR H B firstHead : Nat) (cs ss : Fin B → ℝ)
    (hc : Context u row H B cs ss) (hp : u.pids=s.pids) (hm : u.mem=s.mem) :
    Holds s u Q row QR H B firstHead firstHead cs ss := by
  refine ⟨hc,hp,?_,?_⟩
  · intro t ht hs
    omega
  · intro r o _
    exact congrFun (congrFun hm r) o

theorem earlier (row QR H a b : Nat) (i j : Fin B) (hab : a<b) (hi : i.val<H/2) :
    Reference.first row QR H a i.val≠Reference.first row QR H b j.val ∧
    Reference.first row QR H a i.val≠Reference.second row QR H b j.val ∧
    Reference.second row QR H a i.val≠Reference.first row QR H b j.val ∧
    Reference.second row QR H a i.val≠Reference.second row QR H b j.val := by
  have h := Reference.earlier_head row QR H a b i j hab hi
  simp only [Reference.second,Reference.first] at *
  omega

theorem advance (s u v : BlockState) (Q C S : RegionName) (QR CR SR seq H heads B row firstHead k : Nat)
    (backwards : Bool) (cs ss : Fin B → ℝ)
    (hi : Holds s u Q row QR H B firstHead k cs ss) (hk : firstHead≤k)
    (he : stepStmts (body Q C S QR CR SR seq H heads B backwards) (u.setReg "k" .nat [] (Tile.scalar k))=some v) :
    Holds s v Q row QR H B firstHead (k+1) cs ss := by
  obtain ⟨hc,hp⟩ := Step.preserves_context Q C S QR CR SR seq H heads B row k backwards cs ss u v hi.context he
  refine ⟨hc,hp.trans hi.pids,?_,?_⟩
  · intro t ht htk j hj
    by_cases hold : t<k
    · have hf := Step.frame Q C S QR CR SR seq H heads B row k backwards cs ss u v hi.context he Q
        (Reference.first row QR H t j.val) (by
          intro i _ heq
          exact (earlier row QR H t k j i hold hj).1 heq.2.symm) (by
          intro i _ heq
          exact (earlier row QR H t k j i hold hj).2.1 heq.2.symm)
      have hs := Step.frame Q C S QR CR SR seq H heads B row k backwards cs ss u v hi.context he Q
        (Reference.second row QR H t j.val) (by
          intro i _ heq
          exact (earlier row QR H t k j i hold hj).2.2.1 heq.2.symm) (by
          intro i _ heq
          exact (earlier row QR H t k j i hold hj).2.2.2 heq.2.symm)
      simpa only [BlockState.readMem,hf,hs] using hi.values t ht hold j hj
    · have htk : t=k := by omega
      subst t
      have hf : u.mem Q (Reference.first row QR H k j.val)=s.mem Q (Reference.first row QR H k j.val) := by
        apply hi.frame _ _ (Or.inr ?_)
        rintro ⟨t,_,ht,i,hi,heq⟩
        rcases heq with heq|heq
        · exact (earlier row QR H t k i j ht hi).1 heq.symm
        · exact (earlier row QR H t k i j ht hi).2.2.1 heq.symm
      have hs : u.mem Q (Reference.second row QR H k j.val)=s.mem Q (Reference.second row QR H k j.val) := by
        apply hi.frame _ _ (Or.inr ?_)
        rintro ⟨t,_,ht,i,hi,heq⟩
        rcases heq with heq|heq
        · exact (earlier row QR H t k i j ht hi).2.1 heq.symm
        · exact (earlier row QR H t k i j ht hi).2.2.2 heq.symm
      have h1 := Readback.first Q C S QR CR SR seq H heads B row k backwards cs ss u v hi.context he j hj
      have h2 := Readback.second Q C S QR CR SR seq H heads B row k backwards cs ss u v hi.context he j hj
      simpa only [BlockState.readMem,hf,hs] using And.intro h1 h2
  · intro r o ho
    have hold : u.mem r o=s.mem r o := hi.frame r o (by
      rcases ho with hr|ho
      · exact Or.inl hr
      · right
        rintro ⟨t,ht,hk,j,hj,heq⟩
        exact ho ⟨t,ht,by omega,j,hj,heq⟩)
    refine (Step.frame Q C S QR CR SR seq H heads B row k backwards cs ss u v hi.context he r o ?_ ?_).trans hold
    · intro j hj heq
      rcases ho with hr|ho
      · exact hr heq.1.symm
      · exact ho ⟨k,hk,by omega,j,hj,Or.inl heq.2.symm⟩
    · intro j hj heq
      rcases ho with hr|ho
      · exact hr heq.1.symm
      · exact ho ⟨k,hk,by omega,j,hj,Or.inr heq.2.symm⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FastRopeEmbedding.LoopInvariant
