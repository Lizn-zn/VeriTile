import bench.tritonbench_g.embedding_triton_kernel.Proofs.ChunkSafety
import bench.tritonbench_g.embedding_triton_kernel.Proofs.Invariant
import VeriTile.Triton.Memory.TraceProofs
namespace VeriTile.Bench.TritonBenchG.EmbeddingTritonKernel.Safety
open VeriTile.Triton Chunk LoopInvariant
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 20000

/-- Only source-active token, weight and output addresses require storage.
Bounds are on initial inputs, independently of any execution result. -/
structure Bounds (u : BlockState) (weight ids out : RegionName) (first last : Int)
    (sw so n h D N NN : Nat) (bounds : RegionBounds) : Prop where
  idAccess : ∀ t, t*NN<N → ∀ jr : Fin NN,
    Reference.row u.pid N NN t jr.val<n → Reference.row u.pid N NN t jr.val<bounds ids
  weightAccess : ∀ t, t*NN<N → ∀ jr : Fin NN, ∀ jd : Fin D,
    Reference.row u.pid N NN t jr.val<n → jd.val<h →
    first≤Reference.token u ids (Reference.row u.pid N NN t jr.val) →
    Reference.token u ids (Reference.row u.pid N NN t jr.val)<last →
    (Reference.token u ids (Reference.row u.pid N NN t jr.val)-first).toNat*sw+jd.val<bounds weight
  outputAccess : ∀ t, t*NN<N → ∀ jr : Fin NN, ∀ jd : Fin D,
    Reference.row u.pid N NN t jr.val<n → jd.val<h →
    Reference.row u.pid N NN t jr.val*so+jd.val<bounds out

theorem chunk_bounds (s u : BlockState) (weight ids out : RegionName) (first last : Int)
    (sw so n h D N NN k : Nat) (bounds : RegionBounds)
    (hi : Holds s u weight ids out first last sw so n h D N NN k)
    (hid : ids≠out) (hk : k*NN<N) (hb : Bounds s weight ids out first last sw so n h D N NN bounds) :
    ChunkSafety.Bounds u weight ids out first last sw so n h D N NN (k*NN) bounds := by
  have hp := pid_eq hi
  have hr (jr : Fin NN) : k*NN+(u.pid*N+jr.val)=Reference.row s.pid N NN k jr.val := by
    simp only [Reference.row,hp]
    omega
  have ht (o : Nat) : Reference.token u ids o=Reference.token s ids o := by
    simp only [Reference.token,BlockState.readMemValue,BlockState.readMemTyped,hi.frame ids o (Or.inl hid)]
  constructor
  · intro jr hn
    simpa only [hr] using hb.idAccess k hk jr (by simpa only [hr] using hn)
  · rintro ⟨jr,jd,⟨⟩⟩ ⟨hn,hd⟩ hlo hhi
    simp only [row,hr,ht] at hn hlo hhi ⊢
    exact hb.weightAccess k hk jr jd hn hd hlo hhi
  · rintro ⟨jr,jd,⟨⟩⟩ ⟨hn,hd⟩
    simp only [address,row,hr] at hn ⊢
    exact hb.outputAccess k hk jr jd hn hd

theorem prefix_safe (weight ids out : RegionName) (first last : Int) (sw so n h D N NN : Nat)
    (bounds : RegionBounds) (u : BlockState) :
    Stmt.TraceSafeList bounds (prefixCode weight ids out first last sw so n h D N NN) u := by
  simp [prefixCode,code,Erasure.source,ComputeKernel.surfaceBody,ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,Except.bind,bind,Option.bind,
    Stmt.TraceSafeList,Stmt.TraceSafe,Op.SafeAt.eq_def,stepStmt,evalOp.eq_def]

/-- Memory safety follows the same invariant through the complete range.
No output equation is required from the caller. -/
theorem trace_safe (weight ids out : RegionName) (first last : Int) (sw so n h D N NN : Nat)
    (bounds : RegionBounds) (u : BlockState) (hw : weight≠out) (hid : ids≠out) (hs : min D h≤so)
    (hb : Bounds u weight ids out first last sw so n h D N NN bounds) :
    Stmt.TraceSafeList bounds (Erasure.source Bool.false weight ids out first last sw so n h D N NN).toAlgKernel.body u := by
  change Stmt.TraceSafeList bounds (code weight ids out first last sw so n h D N NN) u
  rw [source_decomposition]
  apply TraceProofs.append_safe _ u (prefix_safe weight ids out first last sw so n h D N NN bounds u)
  intro v hv
  obtain ⟨v',hv',hcv,hpv,hmv⟩ := prefix_executes weight ids out first last sw so n h D N NN u
  have he : v'=v := Option.some.inj (hv'.symm.trans hv)
  subst v'
  have hi := LoopInvariant.initial u v weight ids out first last sw so n h D N NN hcv hpv hmv
  have hrange := TraceProofs.range_safe
    (P := fun cur state => ∃ k, cur=k*NN ∧ Holds u state weight ids out first last sw so n h D N NN k)
    (bounds:=bounds) (index:="start_nn") (stop:=N) (stride:=NN)
    (body:=body weight ids out first last sw so n h D N NN)
    (fun cur s hcur hp => by
      obtain ⟨k,rfl,hk⟩ := hp
      exact ChunkSafety.trace_safe weight ids out first last sw so n h D N NN (k*NN) bounds s hk.context
        (chunk_bounds u s weight ids out first last sw so n h D N NN k bounds hk hid hcur hb))
    (fun cur s t _ hp he => by
      obtain ⟨k,rfl,hk⟩ := hp
      exact ⟨k+1,by simp only [Nat.add_mul,Nat.one_mul],advance u s t weight ids out first last sw so n h D N NN k hk hw hid hs he⟩)
    0 v ⟨0,by omega,hi⟩
  exact Stmt.TraceSafeList.cons_intro (by simpa only [Stmt.TraceSafe] using hrange) (fun _ _ => Stmt.TraceSafeList.nil_intro)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.EmbeddingTritonKernel.Safety
