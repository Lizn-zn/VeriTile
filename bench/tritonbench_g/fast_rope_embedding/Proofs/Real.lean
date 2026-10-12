import bench.tritonbench_g.fast_rope_embedding.Proofs.Invariant
import bench.tritonbench_g.fast_rope_embedding.Proofs.Prelude
import VeriTile.Triton.Semantics.ProgramRewrite
namespace VeriTile.Bench.TritonBenchG.FastRopeEmbedding.RealProof
open VeriTile.Triton Code LoopInvariant
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 25000

theorem loop (s u : BlockState) (Q C S : RegionName) (QR CR SR seq H heads B row firstHead cur stop : Nat)
    (backwards : Bool) (cs ss : Fin B → ℝ)
    (hi : Holds s u Q row QR H B firstHead cur cs ss) (hc : firstHead≤cur) :
    ∃ v K,stepForRangeAux "k" cur stop 1 (body Q C S QR CR SR seq H heads B backwards) u=some v ∧
      Holds s v Q row QR H B firstHead K cs ss ∧ cur≤K ∧
      (∀ t,cur≤t → t<K → t<stop) ∧ stop≤K := by
  by_cases hcur : cur<stop
  · obtain ⟨v,he⟩ := Step.executes Q C S QR CR SR seq H heads B row cur backwards cs ss u hi.context
    have hiv := advance s u v Q C S QR CR SR seq H heads B row firstHead cur backwards cs ss hi hc he
    obtain ⟨w,K,hw,hiw,hK,hvisited,hstop⟩ := loop s v Q C S QR CR SR seq H heads B row firstHead (cur+1) stop backwards cs ss hiv (by omega)
    refine ⟨w,K,?_,hiw,by omega,?_,hstop⟩
    · rw [stepForRangeAux.step_lt (by omega) hcur,he]
      exact hw
    · intro t ht htK
      by_cases htc : t=cur
      · simpa only [htc] using hcur
      · exact hvisited t (by omega) htK
  · refine ⟨u,cur,?_,hi,le_rfl,?_,by omega⟩
    · rw [stepForRangeAux.step_ge (by omega) (by omega)]
    · intro t ht hk
      omega
termination_by stop-cur
decreasing_by omega

/-- Both results use the two halves and coefficient tables in the initial
state. Coefficients are loaded before the loop, so their regions may alias Q. -/
theorem run (Q C S : RegionName) (QR CR SR seq H heads B : Nat) (backwards : Bool) (u : BlockState) :
    ∃ v,exec (Code.original Q C S QR CR SR seq H heads B backwards) u=some v ∧
      (∀ k,Reference.start (u.pids 1)≤k → k<Reference.stop (u.pids 1) heads →
        ∀ j : Fin B,j.val<H/2 →
        v.readMem Q (Reference.first u.pid QR H k j.val)=
          Reference.firstValue backwards
            (u.readMem Q (Reference.first u.pid QR H k j.val))
            (u.readMem Q (Reference.second u.pid QR H k j.val))
            (u.readMem C (Reference.coefficientAddress u.pid CR seq j.val))
            (u.readMem S (Reference.coefficientAddress u.pid SR seq j.val)) ∧
        v.readMem Q (Reference.second u.pid QR H k j.val)=
          Reference.secondValue backwards
            (u.readMem Q (Reference.first u.pid QR H k j.val))
            (u.readMem Q (Reference.second u.pid QR H k j.val))
            (u.readMem C (Reference.coefficientAddress u.pid CR seq j.val))
            (u.readMem S (Reference.coefficientAddress u.pid SR seq j.val))) ∧
      (∀ r o,(r≠Q ∨ ¬Reference.Written u.pid (u.pids 1) QR H heads B o) → v.mem r o=u.mem r o) := by
  obtain ⟨v,hp,hcv,hs,he,hpv,hmv⟩ := Prelude.executes Q C S QR CR SR seq H heads B backwards u
  have hi := LoopInvariant.initial u v Q u.pid QR H B (Reference.start (u.pids 1))
    (Reference.coefficient u C CR seq H B) (fun j => Reference.signedSine backwards (Reference.coefficient u S SR seq H B j)) hcv hpv hmv
  obtain ⟨w,K,heq,hiw,_,hvisited,hstop⟩ := loop u v Q C S QR CR SR seq H heads B u.pid
    (Reference.start (u.pids 1)) (Reference.start (u.pids 1)) (Reference.stop (u.pids 1) heads) backwards
    (Reference.coefficient u C CR seq H B) (fun j => Reference.signedSine backwards (Reference.coefficient u S SR seq H B j)) hi le_rfl
  refine ⟨w,?_,?_,?_⟩
  · change stepStmts (Code.original Q C S QR CR SR seq H heads B backwards).body u=some w
    rw [Code.decomposition,ProgramRewrite.append,hp]
    simp only [Option.bind_some,stepStmts,stepStmt,evalOp.eq_def,hs,he]
    change (match stepForRangeAux "k" (Reference.start (u.pids 1)) (Reference.stop (u.pids 1) heads) 1
      (body Q C S QR CR SR seq H heads B backwards) v with | some t => some t | none => none)=some w
    rw [heq]
  · intro k hk ht j hj
    have hv := hiw.values k hk (by omega) j hj
    simpa only [Reference.firstValue,Reference.secondValue,Reference.coefficient,hj,if_pos] using hv
  · intro r o ho
    apply hiw.frame r o
    rcases ho with hr|ho
    · exact Or.inl hr
    · right
      rintro ⟨t,ht,htK,j,hj,heq⟩
      exact ho ⟨t,ht,hvisited t ht htK,j,hj,heq⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FastRopeEmbedding.RealProof
