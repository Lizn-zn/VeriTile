import bench.tritonbench_g.chunk_bwd_dqkg.Proofs.RealSafety
import bench.tritonbench_g.chunk_bwd_dqkg.Proofs.RealCandidate

/-! Both complete sources execute in one flat address space, write the
independent whole-V real formulas and preserve every other tagged cell.
Active address bounds, allocation disjointness and omitted-padding values
are explicit premises. Candidate transfer is ordinary real arithmetic. -/
namespace VeriTile.Bench.TritonBenchG.ChunkBwdDqkg.RealFlat
open VeriTile.Triton RealMath RealFrame
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 14000
set_option linter.unusedSimpArgs false

private theorem flatten_append (A : FlatAlloc) (xs ys : List Stmt) :
    A.flattenStmts (xs ++ ys) = A.flattenStmts xs ++ A.flattenStmts ys := by
  induction xs with
  | nil => rfl
  | cons x xs ih => simp only [List.cons_append, FlatAlloc.flattenStmts, ih]

theorem candidate_executes_like (A : FlatAlloc) (s t : BlockState) (p : Layout)
    (qk valueStorage hist dhist : TypedSource.Storage)
    (q k v h g dout dh dqOut dkOut dgOut : RegionName)
    (he : exec (A.flattenKernel
      (RealSource.source qk valueStorage hist dhist Bool.false q k v h g dout dh dqOut dkOut dgOut
        p.s_k_h p.s_k_t p.s_v_h p.s_v_t p.s_h_h p.s_h_t p.scale
        p.T p.K p.V p.BT p.BK p.BV p.NT).eraseDType.toAlgKernel) s = some t) :
    exec (A.flattenKernel
      (RealSource.source qk valueStorage hist dhist Bool.true q k v h g dout dh dqOut dkOut dgOut
        p.s_k_h p.s_k_t p.s_v_h p.s_v_t p.s_h_h p.s_h_t p.scale
        p.T p.K p.V p.BT p.BK p.BV p.NT).eraseDType.toAlgKernel) s = some t := by
  rw [exec, FlatAlloc.flattenKernel, RealCandidate.original_split, flatten_append] at he
  rw [exec, FlatAlloc.flattenKernel, RealCandidate.candidate_split, flatten_append]
  simp only [FlatAlloc.flattenStmts, FlatAlloc.flattenStmt, FlatAlloc.flattenOp] at he ⊢
  exact MatrixRewrite.add_commute _ _ _ _ _ s t he

def flatFootprint (A : FlatAlloc) (p : Layout) (l : Launch) (dq dk dg : RegionName) : WriteFootprint :=
  fun addr => addr.1 = A.flat ∧
    ((∃ i, qkActive p l i ∧ A.addr dq (qkAddress p l i) = addr.2) ∨
     (∃ i, qkActive p l i ∧ A.addr dk (qkAddress p l i) = addr.2) ∨
     (∃ i, dgActive p l i ∧ A.addr dg (dgAddress p l i) = addr.2))

theorem flatten_frame (A : FlatAlloc) (s t : BlockState) (p : Layout) (l : Launch)
    (dq dk dg : RegionName) (hf : BlockState.WriteWithin (footprint p l dq dk dg) s t) :
    BlockState.WriteWithin (flatFootprint A p l dq dk dg) (A.flattenState s) (A.flattenState t) := by
  intro r o hmiss
  by_cases hr : r = A.flat
  · subst r
    simp only [FlatAlloc.flattenState, if_pos rfl, FlatAlloc.readFlat]
    cases hd : A.decode o with
    | none => rfl
    | some cell =>
      obtain ⟨r',o'⟩ := cell
      obtain ⟨_, heq, _⟩ := A.decode_sound hd
      apply congrArg A.trCell
      apply hf
      intro hhit
      apply hmiss
      refine ⟨rfl, ?_⟩
      rcases hhit with hq | hk | hg
      · obtain ⟨hr,i,hi,hoff⟩ := hq
        exact Or.inl ⟨i,hi, by rw [← hr, hoff, ← heq]⟩
      · obtain ⟨hr,i,hi,hoff⟩ := hk
        exact Or.inr (Or.inl ⟨i,hi, by rw [← hr, hoff, ← heq]⟩)
      · obtain ⟨hr,i,hi,hoff⟩ := hg
        exact Or.inr (Or.inr ⟨i,hi, by rw [← hr, hoff, ← heq]⟩)
  · simp only [FlatAlloc.flattenState, if_neg hr]

theorem source_executes (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0)
    (s : BlockState) (p : Layout) (swapped : Bool)
    (qk valueStorage hist dhist : TypedSource.Storage)
    (q k v h g dout dh dqOut dkOut dgOut : RegionName)
    (hu : s.undef = (fun _ _ => 0)) (hBV : 0 < p.BV)
    (hDqDk : dqOut ≠ dkOut) (hDqDg : dqOut ≠ dgOut) (hDkDg : dkOut ≠ dgOut)
    (hInj : ∀ i j, qkActive p (RealLoop.launch s) i → qkActive p (RealLoop.launch s) j →
      qkAddress p (RealLoop.launch s) i = qkAddress p (RealLoop.launch s) j → i = j)
    (hb : RealSafety.Bounds s p q k v h g dout dh dqOut dkOut dgOut A.extent) :
    ∃ sf, exec (A.flattenKernel
      (RealSource.source qk valueStorage hist dhist swapped q k v h g dout dh dqOut dkOut dgOut
        p.s_k_h p.s_k_t p.s_v_h p.s_v_t p.s_h_h p.s_h_t p.scale
        p.T p.K p.V p.BT p.BK p.BV p.NT).eraseDType.toAlgKernel) (A.flattenState s) = some sf ∧
      (∀ i, qkActive p (RealLoop.launch s) i →
        sf.readMem A.flat (A.addr dqOut (qkAddress p (RealLoop.launch s) i)) =
          RealMath.dq p (RealLoop.launch s) (RealPost.inputs s q k v h g dout dh) i.1 i.2.1) ∧
      (∀ i, qkActive p (RealLoop.launch s) i →
        sf.readMem A.flat (A.addr dkOut (qkAddress p (RealLoop.launch s) i)) =
          RealMath.dk p (RealLoop.launch s) (RealPost.inputs s q k v h g dout dh) i.1 i.2.1) ∧
      (∀ i, dgActive p (RealLoop.launch s) i →
        sf.readMem A.flat (A.addr dgOut (dgAddress p (RealLoop.launch s) i)) =
          RealMath.dg p (RealLoop.launch s) (RealPost.inputs s q k v h g dout dh) i.1) ∧
      BlockState.WriteWithin (flatFootprint A p (RealLoop.launch s) dqOut dkOut dgOut)
        (A.flattenState s) sf := by
  have hinj : ∀ i j : TileIndex [p.BT,p.BK],
      (s.pids 1 * p.BT + i.1.val < p.T ∧ s.pids 0 * p.BK + i.2.1.val < p.K) →
      (s.pids 1 * p.BT + j.1.val < p.T ∧ s.pids 0 * p.BK + j.2.1.val < p.K) →
      cbdQkAddr s p.s_k_h p.s_k_t p.BT p.BK i = cbdQkAddr s p.s_k_h p.s_k_t p.BT p.BK j → i = j := by
    intro i j hi hj he
    apply hInj i j hi hj
    simpa only [qkAddress, RealLoop.launch, cbdQkAddr, Nat.mul_one, Nat.add_assoc] using he
  obtain ⟨t, he, hq, hk, hg⟩ := RealPost.original_executes s p q k v h g dout dh
    qk valueStorage hist dhist dqOut dkOut dgOut hu hBV hDqDk hDqDg hDkDg hinj
  have hflat := A.exec_flatten hd hcov _ s
    (RealSafety.source_safe s p qk valueStorage hist dhist q k v h g dout dh dqOut dkOut dgOut A.extent hb hu)
    (RealSafety.source_flatten_ok p qk valueStorage hist dhist q k v h g dout dh dqOut dkOut dgOut) hu
  rw [he, Option.map_some] at hflat
  have hframe := flatten_frame A s t p (RealLoop.launch s) dqOut dkOut dgOut
    (original_frame s t p qk valueStorage hist dhist q k v h g dout dh dqOut dkOut dgOut hu he)
  have hmem (r : RegionName) (off : Nat) (ho : off < A.extent r) :
      (A.flattenState t).readMem A.flat (A.addr r off) = t.readMem r off := by
    have hr : r ∈ A.regions := by
      by_contra hn
      rw [hcov r hn] at ho
      exact Nat.not_lt_zero _ ho
    exact A.flattenState_readMem hd t hr ho
  refine ⟨A.flattenState t, ?_, ?_, ?_, ?_, hframe⟩
  · cases swapped
    · exact hflat
    · exact candidate_executes_like A _ _ p qk valueStorage hist dhist
        q k v h g dout dh dqOut dkOut dgOut hflat
  · intro i hi
    rw [hmem _ _ (hb.post.dq i hi)]
    simpa only [qkAddress, RealLoop.launch, cbdQkAddr, Nat.mul_one, Nat.add_assoc] using hq i hi
  · intro i hi
    rw [hmem _ _ (hb.post.dk i hi)]
    simpa only [qkAddress, RealLoop.launch, cbdQkAddr, Nat.mul_one, Nat.add_assoc] using hk i hi
  · intro i hi
    rw [hmem _ _ (hb.post.dg i hi)]
    simpa only [dgAddress, RealLoop.launch, cbdDgAddr, Nat.mul_one, Nat.add_assoc] using hg i hi

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkBwdDqkg.RealFlat
