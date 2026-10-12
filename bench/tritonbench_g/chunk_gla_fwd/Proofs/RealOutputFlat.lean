import bench.tritonbench_g.chunk_gla_fwd.Proofs.RealOutputSafety
import bench.tritonbench_g.chunk_gla_fwd.Proofs.RealOutputCandidate

/-! Complete real consumer execution in a flat allocation. Active address bounds,
allocation disjointness, zero unspecified values and distinct output addresses
are explicit. The conclusion includes the independent gated contraction and
preservation of every other tagged memory cell for both source orders. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealOutputFlat
open VeriTile.Triton RealOutput
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false

private theorem flatten_append (alloc : FlatAlloc) (xs ys : List Stmt) :
    alloc.flattenStmts (xs ++ ys) = alloc.flattenStmts xs ++ alloc.flattenStmts ys := by
  induction xs with
  | nil => rfl
  | cons x xs ih => simp only [List.cons_append,FlatAlloc.flattenStmts,ih]

section
variable (query value gate : TypedOutput.Storage)
    (dtype out_dtype : FloatDType) (q v g h o A : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
    (scale : ℝ) (T K V BT BK BV : Nat)

theorem candidate_executes_like (alloc : FlatAlloc) (s t : BlockState)
    (he : exec (alloc.flattenKernel
      (TypedOutput.source query value gate Bool.false dtype out_dtype q v g h o A s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV).eraseDType.toAlgKernel) s = some t) :
    exec (alloc.flattenKernel
      (TypedOutput.source query value gate Bool.true dtype out_dtype q v g h o A s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV).eraseDType.toAlgKernel) s = some t := by
  rw [exec,FlatAlloc.flattenKernel,RealOutputCandidate.original_split,flatten_append] at he
  rw [exec,FlatAlloc.flattenKernel,RealOutputCandidate.candidate_split,flatten_append]
  simp only [FlatAlloc.flattenStmts,FlatAlloc.flattenStmt,FlatAlloc.flattenOp] at he ⊢
  exact MatrixRewrite.add_commute _ _ _ _ _ s t he

def footprint (alloc : FlatAlloc) (o : RegionName) (s_v_h s_v_t T V BT BV : Nat) (s : BlockState) : WriteFootprint :=
  fun cell => cell.1 = alloc.flat ∧ ∃ idx : TileIndex [BT,BV],
    (tIndex s BT idx.1.val < T ∧ vIndex s BV idx.2.1.val < V) ∧
    cell.2 = alloc.addr o (outOffset s s_v_h s_v_t BT BV idx)

theorem flatten_frame (alloc : FlatAlloc) (s t : BlockState)
    (hf : BlockState.WriteWithin (RealOutput.footprint o s_v_h s_v_t T V BT BV s) s t) :
    BlockState.WriteWithin (footprint alloc o s_v_h s_v_t T V BT BV s) (alloc.flattenState s) (alloc.flattenState t) := by
  intro region off hn
  by_cases hr : region = alloc.flat
  · subst region
    simp only [FlatAlloc.flattenState,if_pos rfl,FlatAlloc.readFlat]
    cases hd : alloc.decode off with
    | none => rfl
    | some cell =>
      rcases cell with ⟨r,k⟩
      obtain ⟨_,heq,_⟩ := alloc.decode_sound hd
      apply congrArg alloc.trCell
      apply hf
      rintro ⟨hr,idx,hi,haddr⟩
      apply hn
      refine ⟨rfl,idx,hi,?_⟩
      rw [← hr,← haddr,← heq]
  · simp only [FlatAlloc.flattenState,if_neg hr]

theorem source_executes (alloc : FlatAlloc) (hd : alloc.Disjoint)
    (hcov : ∀ r, r ∉ alloc.regions → alloc.extent r = 0)
    (swapped : Bool) (s : BlockState)
    (hu : s.undef = fun _ _ => 0)
    (hInj : Function.Injective (outOffset s s_v_h s_v_t BT BV))
    (hb : RealOutputSafety.Bounds s q v g h o A
      s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t T K V BT BK BV alloc.extent) :
    ∃ sf, exec (alloc.flattenKernel
      (TypedOutput.source query value gate swapped dtype out_dtype q v g h o A s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV).eraseDType.toAlgKernel)
      (alloc.flattenState s) = some sf ∧
      (∀ idx : TileIndex [BT,BV],
        (tIndex s BT idx.1.val < T ∧ vIndex s BV idx.2.1.val < V) →
        sf.readMem alloc.flat (alloc.addr o (outOffset s s_v_h s_v_t BT BV idx)) =
          cgfOutput s q v g h A s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV idx.1.val idx.2.1.val) ∧
      BlockState.WriteWithin (footprint alloc o s_v_h s_v_t T V BT BV s) (alloc.flattenState s) sf := by
  obtain ⟨t,he,ho,hf⟩ := RealOutputCandidate.source_executes query value gate dtype out_dtype q v g h o A s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV Bool.false s hu hInj
  have hflat := alloc.exec_flatten hd hcov _ s
    (RealOutputSafety.source_safe s query value gate dtype out_dtype q v g h o A s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV alloc.extent hb hu)
    (RealOutputSafety.source_flatten_ok query value gate dtype out_dtype q v g h o A s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV) hu
  rw [he,Option.map_some] at hflat
  refine ⟨alloc.flattenState t,?_,?_,flatten_frame o s_v_h s_v_t T V BT BV alloc s t hf⟩
  · cases swapped
    · exact hflat
    · exact candidate_executes_like query value gate dtype out_dtype q v g h o A s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV alloc _ _ hflat
  · intro idx hi
    have hoff : outOffset s s_v_h s_v_t BT BV idx < alloc.extent o := hb.post.output idx.1 idx.2.1 hi
    have hr : o ∈ alloc.regions := by
      by_contra hn
      rw [hcov o hn] at hoff
      exact Nat.not_lt_zero _ hoff
    rw [alloc.flattenState_readMem hd t hr hoff]
    exact ho idx hi
end
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealOutputFlat
