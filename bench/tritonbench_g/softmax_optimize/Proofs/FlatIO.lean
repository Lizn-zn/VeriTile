import bench.tritonbench_g.softmax_optimize.Proofs.SignedIO
import bench.tritonbench_g.softmax_optimize.Proofs.SafetyCandidate
import bench.tritonbench_g.softmax_flaggems.Proofs.RealFlatFrame
import VeriTile.Triton.Memory.KernelCore

namespace VeriTile.Bench.TritonBenchG.SoftmaxOptimize.FullIO
open VeriTile Triton FullSource SignedTransport
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000

/-- Exact row footprint; no cell outside the logical row is writable. -/
def footprint (Y : RegionName) (row N : Nat) : WriteFootprint :=
  fun a ↦ a.1=Y ∧ row*N ≤ a.2 ∧ a.2 < row*N+N

theorem frame_writeWithin (Y : RegionName) (N : Nat) (s t : BlockState)
    (h : RowFrame Y N s t) : BlockState.WriteWithin (footprint Y s.pid N) s t := by
  intro r o hn
  apply Eq.symm
  apply h
  by_cases hr : r=Y
  · subst r
    right
    by_cases hlo : o<s.pid*N
    · exact Or.inl hlo
    · right
      by_contra hi
      exact hn ⟨rfl, by omega, by omega⟩
  · exact Or.inl hr

theorem original_flattenOk (input : Storage) (output : Accumulation)
    (Y X : RegionName) (M N B : Nat) :
    (kernel input output Bool.false Y X M N B).eraseDType.toAlgKernel.FlattenOk := by
  rw [projection]
  unfold Kernel.FlattenOk
  rw [source_body]
  simp [StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def,
    inputFull, inputTail, outputFull, outputTail, indexPrefix, maskStatement, outputPointer,
    loopAbody, loopBbody, loopApass2body, loopBpass2body, reductionStmts,
    previousExpression, signedQuotient]

theorem candidate_flattenOk (input : Storage) (output : Accumulation)
    (Y X : RegionName) (M N B : Nat) :
    (kernel input output Bool.true Y X M N B).eraseDType.toAlgKernel.FlattenOk := by
  rw [projection]
  unfold Kernel.FlattenOk
  rw [candidate_source_body]
  simp [StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def,
    inputFullCandidate, inputTailCandidate, outputFullCandidate, outputTailCandidate, indexPrefixCandidate, maskStatement, outputPointer,
    loopAbody, loopBbody, loopApass2body, loopBpass2body, reductionStmts,
    previousExpression, signedQuotient]

/-- Flat execution bridge with the existing flattenState zero-undef premise
made explicit. The primary region IO theorem permits arbitrary undef values. -/
theorem original_flat_io (input : Storage) (output : Accumulation)
    (A : FlatAlloc) (hd : A.Disjoint) (hcov : ∀ r, r∉A.regions → A.extent r=0)
    (s : BlockState) (Y X : RegionName) (M N B : Nat)
    (hN : 0<N) (hB : 0<B) (hYX : Y≠X) (hr : s.pids 0<M)
    (hX : M*N≤A.extent X) (hY : M*N≤A.extent Y) (hYr : Y∈A.regions)
    (hu : s.undef=fun _ _ ↦ 0) : ∃ t,
      exec (A.flattenKernel (kernel input output Bool.false Y X M N B).eraseDType.toAlgKernel)
        (A.flattenState s) = some t ∧
      (∀ j : Fin N, t.readMem A.flat (A.addr Y (linearOffset s N j)) =
        softmaxOptimizeFullSpec s X N j) ∧
      BlockState.WriteWithin
        (SoftmaxFlaggems.RealFlatFrame.footprint A (footprint Y s.pid N)) (A.flattenState s) t := by
  obtain ⟨t, ht, hv, hf⟩ := original_io input output Y X M N B hN hB hYX s
  refine ⟨A.flattenState t, ?_, ?_, ?_⟩
  · rw [A.exec_flatten hd hcov _ s
      (Safety.original_source_safe input output A.extent s Y X M N B hN hB hr hX hY)
      (original_flattenOk input output Y X M N B) hu, ht, Option.map_some]
  · intro j
    rw [A.flattenState_readMem hd t hYr]
    · exact hv j
    · exact Safety.row_address_bound s.pid M N j.val (A.extent Y) hr j.isLt hY
  · exact SoftmaxFlaggems.RealFlatFrame.transport A _ s t (frame_writeWithin Y N s t hf)

theorem candidate_flat_io (input : Storage) (output : Accumulation)
    (A : FlatAlloc) (hd : A.Disjoint) (hcov : ∀ r, r∉A.regions → A.extent r=0)
    (s : BlockState) (Y X : RegionName) (M N B : Nat)
    (hN : 0<N) (hB : 0<B) (hYX : Y≠X) (hr : s.pids 0<M)
    (hX : M*N≤A.extent X) (hY : M*N≤A.extent Y) (hYr : Y∈A.regions)
    (hu : s.undef=fun _ _ ↦ 0) : ∃ t,
      exec (A.flattenKernel (kernel input output Bool.true Y X M N B).eraseDType.toAlgKernel)
        (A.flattenState s) = some t ∧
      (∀ j : Fin N, t.readMem A.flat (A.addr Y (linearOffset s N j)) =
        softmaxOptimizeFullSpec s X N j) ∧
      BlockState.WriteWithin
        (SoftmaxFlaggems.RealFlatFrame.footprint A (footprint Y s.pid N)) (A.flattenState s) t := by
  obtain ⟨t, ht, hv, hf⟩ := candidate_io input output Y X M N B hN hB hYX s
  refine ⟨A.flattenState t, ?_, ?_, ?_⟩
  · rw [A.exec_flatten hd hcov _ s
      (Safety.candidate_source_safe input output A.extent s Y X M N B hN hB hr hX hY)
      (candidate_flattenOk input output Y X M N B) hu, ht, Option.map_some]
  · intro j
    rw [A.flattenState_readMem hd t hYr]
    · exact hv j
    · exact Safety.row_address_bound s.pid M N j.val (A.extent Y) hr j.isLt hY
  · exact SoftmaxFlaggems.RealFlatFrame.transport A _ s t (frame_writeWithin Y N s t hf)

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxOptimize.FullIO
