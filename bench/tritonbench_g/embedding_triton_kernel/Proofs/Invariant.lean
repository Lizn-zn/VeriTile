import bench.tritonbench_g.embedding_triton_kernel.Proofs.Chunk
namespace VeriTile.Bench.TritonBenchG.EmbeddingTritonKernel.LoopInvariant
open VeriTile.Triton Chunk
noncomputable section
set_option maxHeartbeats 2000000

/-- Exact output footprint of the first `k` complete source chunks. -/
def Written (pid N NN D n h so k o : Nat) : Prop :=
  ∃ t, t<k ∧ ∃ jr : Fin NN, ∃ jd : Fin D,
    Reference.row pid N NN t jr.val<n ∧ jd.val<h ∧
    o=Reference.row pid N NN t jr.val*so+jd.val

structure Holds (s u : BlockState) (weight ids out : RegionName) (first last : Int)
    (sw so n h D N NN k : Nat) : Prop where
  context : Context u N NN D
  pids : u.pids=s.pids
  values : ∀ t, t<k → ∀ jr : Fin NN, ∀ jd : Fin D,
    Reference.row s.pid N NN t jr.val<n → jd.val<h →
    u.readMem out (Reference.row s.pid N NN t jr.val*so+jd.val)=
      Reference.value s weight ids first last sw (Reference.row s.pid N NN t jr.val) jd.val
  frame : ∀ r o, (r≠out ∨ ¬Written s.pid N NN D n h so k o) → u.mem r o=s.mem r o

theorem pid_eq (hi : Holds s u weight ids out first last sw so n h D N NN k) :
    u.pid=s.pid := congrFun hi.pids 0

theorem initial (s u : BlockState) (weight ids out : RegionName) (first last : Int)
    (sw so n h D N NN : Nat) (hc : Context u N NN D) (hp : u.pids=s.pids) (hm : u.mem=s.mem) :
    Holds s u weight ids out first last sw so n h D N NN 0 := by
  refine ⟨hc,hp,?_,?_⟩
  · intro t ht
    omega
  · intro r o _
    exact congrFun (congrFun hm r) o

theorem row_eq (hi : Holds s u weight ids out first last sw so n h D N NN k)
    (jr : Fin NN) (jd : Fin D) :
    row u N (k*NN) (jr,jd,PUnit.unit)=Reference.row s.pid N NN k jr.val := by
  simp only [row,Reference.row,pid_eq hi]
  omega

/-- Earlier chunks occupy strictly earlier logical rows, including padded
lanes in a final chunk. This uses no restriction to single-row chunks. -/
theorem earlier_row (pid N NN t k : Nat) (jr kr : Fin NN) (ht : t<k) :
    Reference.row pid N NN t jr.val<Reference.row pid N NN k kr.val := by
  have hmul := Nat.mul_le_mul_right NN (show t+1≤k by omega)
  simp only [Nat.add_mul,Nat.one_mul] at hmul
  simp only [Reference.row]
  omega

/-- One actual source iteration extends all output values and the complete
cell-tag frame. The input separation prevents later chunks reading outputs
as token IDs or weights. -/
theorem advance (s u v : BlockState) (weight ids out : RegionName) (first last : Int)
    (sw so n h D N NN k : Nat) (hi : Holds s u weight ids out first last sw so n h D N NN k)
    (hw : weight≠out) (hid : ids≠out) (hs : min D h≤so)
    (he : stepStmts (body weight ids out first last sw so n h D N NN)
      (u.setReg "start_nn" .nat [] (Tile.scalar (k*NN)))=some v) :
    Holds s v weight ids out first last sw so n h D N NN (k+1) := by
  obtain ⟨hcv,hpv⟩ := Chunk.preserves_context weight ids out first last sw so n h D N NN (k*NN) u v hi.context he
  refine ⟨hcv,hpv.trans hi.pids,?_,?_⟩
  · intro t ht jr jd hn hd
    by_cases htk : t<k
    · have hmem := Chunk.frame weight ids out first last sw so n h D N NN (k*NN) u v hi.context he
        out (Reference.row s.pid N NN t jr.val*so+jd.val) ?_
      · simpa only [BlockState.readMem,hmem] using hi.values t htk jr jd hn hd
      · intro j hj haddr
        obtain ⟨kr,kd,⟨⟩⟩ := j
        have hr := row_eq hi kr kd
        have heq : Reference.row s.pid N NN k kr.val*so+kd.val=
            Reference.row s.pid N NN t jr.val*so+jd.val := by
          simpa only [address,hr] using haddr.2
        obtain ⟨heq,_⟩ := Reference.row_major_injective so D h _ _ kd jd hs hj.2 hd heq
        exact (Nat.ne_of_lt (earlier_row s.pid N NN t k jr kr htk)) heq.symm
    · have htk : t=k := by omega
      subst t
      let j : TileIndex [NN,D] := (jr,jd,PUnit.unit)
      have hr : row u N (k*NN) j=Reference.row s.pid N NN k jr.val := row_eq hi jr jd
      have hj : Active u n h N (k*NN) j := ⟨by simpa only [hr] using hn,hd⟩
      have hv := Chunk.readback weight ids out first last sw so n h D N NN (k*NN) u v hi.context he j hj
        (fun q hq heq => Chunk.no_collision u so n h N (k*NN) q j hs hq hj heq)
      have href := Reference.value_congr s u weight ids first last sw (Reference.row s.pid N NN k jr.val) jd.val
        (fun o => hi.frame weight o (Or.inl hw)) (fun o => hi.frame ids o (Or.inl hid))
      simpa only [address,hr,j,href] using hv
  · intro r o ho
    have hold : u.mem r o=s.mem r o := hi.frame r o (by
      rcases ho with hr|ho
      · exact Or.inl hr
      · right
        rintro ⟨t,ht,jr,jd,hn,hd,heq⟩
        exact ho ⟨t,by omega,jr,jd,hn,hd,heq⟩)
    refine (Chunk.frame weight ids out first last sw so n h D N NN (k*NN) u v hi.context he r o ?_).trans hold
    intro j hj heq
    rcases ho with hr|ho
    · exact hr heq.1.symm
    · obtain ⟨jr,jd,⟨⟩⟩ := j
      have hr := row_eq hi jr jd
      exact ho ⟨k,by omega,jr,jd,by simpa only [hr] using hj.1,hj.2,
        by simpa only [address,hr] using heq.2.symm⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.EmbeddingTritonKernel.LoopInvariant
