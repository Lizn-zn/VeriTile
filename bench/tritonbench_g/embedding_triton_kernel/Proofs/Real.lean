import bench.tritonbench_g.embedding_triton_kernel.Proofs.Invariant
import VeriTile.Triton.Semantics.ProgramRewrite
namespace VeriTile.Bench.TritonBenchG.EmbeddingTritonKernel.RealProof
open VeriTile.Triton Chunk LoopInvariant
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 20000

/-- Every executed iteration advances the invariant. The returned count
records the actual source range, including an overshooting final chunk. -/
theorem loop (s u : BlockState) (weight ids out : RegionName) (first last : Int)
    (sw so n h D N NN k : Nat) (hi : Holds s u weight ids out first last sw so n h D N NN k)
    (hw : weight≠out) (hid : ids≠out) (hs : min D h≤so) :
    ∃ v K, stepForRangeAux "start_nn" (k*NN) N NN (body weight ids out first last sw so n h D N NN) u=some v ∧
      Holds s v weight ids out first last sw so n h D N NN K ∧ k≤K ∧
      (∀ t, k≤t → t<K → t*NN<N) ∧ (NN=0 ∨ N≤K*NN) := by
  by_cases hz : NN=0
  · refine ⟨u,k,?_,hi,le_rfl,?_,Or.inl hz⟩
    · subst NN
      rw [stepForRangeAux.step_zero_step]
    · intro t ht hk
      omega
  · by_cases hk : k*NN<N
    · obtain ⟨v,he⟩ := Chunk.executes weight ids out first last sw so n h D N NN (k*NN) u hi.context
      have hiv := advance s u v weight ids out first last sw so n h D N NN k hi hw hid hs he
      obtain ⟨w,K,heq,hiw,hK,hvisited,hstop⟩ := loop s v weight ids out first last sw so n h D N NN (k+1) hiv hw hid hs
      refine ⟨w,K,?_,hiw,by omega,?_,hstop⟩
      · rw [stepForRangeAux.step_lt hz hk,he]
        simpa only [Nat.add_mul,Nat.one_mul] using heq
      · intro t ht htK
        by_cases htk : t=k
        · simpa only [htk] using hk
        · exact hvisited t (by omega) htK
    · refine ⟨u,k,?_,hi,le_rfl,?_,Or.inr (by omega)⟩
      · rw [stepForRangeAux.step_ge hz (by omega)]
      · intro t ht hk
        omega
termination_by N-k*NN
decreasing_by simp only [Nat.add_mul,Nat.one_mul]; omega

/-- Complete-source successful execution, independent initial-memory lookup
at every source output, and preservation of every other tagged cell. -/
theorem run (weight ids out : RegionName) (first last : Int) (sw so n h D N NN : Nat)
    (u : BlockState) (hw : weight≠out) (hid : ids≠out) (hs : min D h≤so) :
    ∃ v, exec (Erasure.source Bool.false weight ids out first last sw so n h D N NN).toAlgKernel u=some v ∧
      (∀ t, t*NN<N → ∀ jr : Fin NN, ∀ jd : Fin D,
        Reference.row u.pid N NN t jr.val<n → jd.val<h →
        v.readMem out (Reference.row u.pid N NN t jr.val*so+jd.val)=
          Reference.value u weight ids first last sw (Reference.row u.pid N NN t jr.val) jd.val) ∧
      (∀ r o, (r≠out ∨ ¬Reference.Written u.pid N NN D n h so o) → v.mem r o=u.mem r o) := by
  obtain ⟨v,hp,hcv,hpv,hmv⟩ := prefix_executes weight ids out first last sw so n h D N NN u
  have hi := LoopInvariant.initial u v weight ids out first last sw so n h D N NN hcv hpv hmv
  obtain ⟨w,K,heq,hiw,_,hvisited,hstop⟩ := loop u v weight ids out first last sw so n h D N NN 0 hi hw hid hs
  refine ⟨w,?_,?_,?_⟩
  · change stepStmts (code weight ids out first last sw so n h D N NN) u=some w
    rw [source_decomposition,ProgramRewrite.append,hp]
    simp only [Nat.zero_mul] at heq
    simp only [Option.bind_some,stepStmts,stepStmt,heq]
  · intro t ht jr jd hn hd
    have htK : t<K := by
      rcases hstop with hz|hstop
      · have := jr.isLt
        omega
      · by_contra hnot
        have hmul := Nat.mul_le_mul_right NN (show K≤t by omega)
        omega
    exact hiw.values t htK jr jd hn hd
  · intro r o ho
    apply hiw.frame r o
    rcases ho with hr|ho
    · exact Or.inl hr
    · right
      rintro ⟨t,ht,jr,jd,hn,hd,he⟩
      exact ho ⟨t,hvisited t (Nat.zero_le _) ht,jr.val,jr.isLt,jd.val,jd.isLt,hn,hd,he⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.EmbeddingTritonKernel.RealProof
