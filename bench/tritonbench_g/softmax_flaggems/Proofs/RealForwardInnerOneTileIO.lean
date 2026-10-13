import bench.tritonbench_g.softmax_flaggems.Proofs.RealForwardInnerOneTileStep
import bench.tritonbench_g.softmax_flaggems.Proofs.RealCandidate
import bench.tritonbench_g.softmax_flaggems.Proofs.RealProjection

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardInnerOneTile
open VeriTile Triton FullSource
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

theorem scatter_value {B : Nat} (s : BlockState) (Y : RegionName) (row N : Nat)
    (value : TileIndex [B] → ℝ) (i : TileIndex [B]) (hi : i.1.val<N) :
    (scatter s Y (tileOffset row N) value (tileActive N)).readMem Y (tileOffset row N i) = value i := by
  apply BlockState.scatter_readback_prop_masked_nd_of_true s (tileOffset row N) value
    (fun i ↦ tileActive N i=Bool.true) i (by simpa [tileActive] using hi)
  intro j _ he
  apply Prod.ext
  · exact Fin.ext (Nat.add_left_cancel he)
  · exact Subsingleton.elim _ _

theorem scatter_frame {B : Nat} (s : BlockState) (Y : RegionName) (row N : Nat)
    (value : TileIndex [B] → ℝ) :
    BlockState.WriteWithin (RealForwardInner.outputFootprint Y row N) s
      (scatter s Y (tileOffset row N) value (tileActive N)) := by
  apply BlockState.foldl_writeMem_prop_writeWithin Y (tileOffset row N) value
    (fun i ↦ tileActive N i=Bool.true) (TileShape.allIndices [B]) s
  intro i _ hi
  exact ⟨rfl,i.1.val,of_decide_eq_true hi,rfl⟩

theorem scatter_launch {B : Nat} (s : BlockState) (Y : RegionName)
    (off : TileIndex [B] → Nat) (value : TileIndex [B] → ℝ) (active : TileIndex [B] → Bool) :
    (scatter s Y off value active).pids=s.pids ∧
      (scatter s Y off value active).numPids=s.numPids ∧
      (scatter s Y off value active).undef=s.undef := by
  unfold scatter
  generalize TileShape.allIndices [B]=lanes
  induction lanes generalizing s with
  | nil => exact ⟨rfl,rfl,rfl⟩
  | cons i tail ih =>
    simp only [List.foldl_cons]
    by_cases hi : active i=Bool.true
    · simpa only [if_pos hi, BlockState.writeMem_pids, BlockState.writeMem_numPids,
        BlockState.writeMem_undef] using ih (s.writeMem Y (off i) (value i))
    · simpa only [if_neg hi] using ih s

theorem body_io (s : BlockState) (Y X : RegionName) (M N B row : Nat)
    (hN : 0<N) (hB : 0<B) (hNB : N≤B)
    (hrow : s.regs .nat [] "pid_m"=some (Tile.scalar row)) :
    ∃ final, stepStmts (body Y X M N B) s=some final ∧
      BlockState.WriteWithin (RealForwardInner.outputFootprint Y row N) s final ∧
      (∀ n, n<N → final.readMem Y (row*N+n)=RealForwardInner.expected s X row N n) ∧
      final.pids=s.pids ∧ final.numPids=s.numPids ∧ final.undef=s.undef := by
  obtain ⟨mid,he,hm,hp,hn,hu⟩ := step Y X M N B row hB s hrow
  let value := storeValue s X row N B hB
  let final := scatter mid Y (tileOffset row N) value (tileActive N)
  have hf := scatter_frame mid Y row N value
  have hl := scatter_launch mid Y (tileOffset row N) value (tileActive N)
  refine ⟨final,he,?_,?_,hl.1.trans hp,hl.2.1.trans hn,hl.2.2.trans hu⟩
  · intro r o ho
    rw [←hm]
    exact hf r o ho
  · intro n hn'
    let i : TileIndex [B] := (⟨n,lt_of_lt_of_le hn' hNB⟩,PUnit.unit)
    exact (scatter_value mid Y row N value i hn').trans
      (storeValue_expected s X row N B hN hB hNB i hn')

theorem candidate_exec (input output : Storage) (Y X : RegionName) (M N B : Nat)
    (s : BlockState) :
    exec (softmax_kernel_inner input output Y X M N B Bool.true).eraseDType.toAlgKernel s =
      exec (softmax_kernel_inner_candidate input output Y X M N B Bool.true).eraseDType.toAlgKernel s := by
  unfold exec
  conv_lhs => rw [RealCandidate.forward_inner_original_body]
  conv_rhs => rw [RealCandidate.forward_inner_body]
  simp [stepStmts,stepStmt,evalOp.eq_def]

theorem original_io (input output : Storage) (s : BlockState) (Y X : RegionName)
    (M N B : Nat) (hN : 0<N) (hB : 0<B) (hNB : N≤B) :
    ∃ final, exec (softmax_kernel_inner input output Y X M N B Bool.true).eraseDType.toAlgKernel s=some final ∧
      BlockState.WriteWithin (RealForwardInner.outputFootprint Y (s.pids 0) N) s final ∧
      (∀ n, n<N → final.readMem Y (s.pids 0*N+n)=RealForwardInner.expected s X (s.pids 0) N n) ∧
      final.pids=s.pids ∧ final.numPids=s.numPids ∧ final.undef=s.undef := by
  rw [RealProjection.softmax_kernel_inner_erased]
  let st := s.setReg "pid_m" .nat [] (Tile.scalar (s.pids 0))
  obtain ⟨final,he,hf,hv,hp,hn,hu⟩ := body_io st Y X M N B (s.pids 0) hN hB hNB (by simp [st])
  refine ⟨final,?_,hf,?_,hp,hn,hu⟩
  · unfold exec
    rw [RealCandidate.forward_inner_original_body]
    simp only [stepStmts,stepStmt,evalOp.eq_def,Tile.scalar_data,
      Bool.true_eq,↓reduceIte,Option.bind_eq_bind,Option.bind,Option.pure_def]
    have he' : stepStmts (RealCandidate.thenBody
        (softmax_kernel_inner .fp32 .fp32 Y X M N B Bool.true).eraseDType.toAlgKernel) st =
        some final := he
    rw [he']
  · intro n hn'
    simpa only [st,RealForwardInner.expected,RealForwardInner.denominator,
      BlockState.readMem,BlockState.setReg_mem] using hv n hn'

theorem candidate_io (input output : Storage) (s : BlockState) (Y X : RegionName)
    (M N B : Nat) (hN : 0<N) (hB : 0<B) (hNB : N≤B) :
    ∃ final, exec (softmax_kernel_inner_candidate input output Y X M N B Bool.true).eraseDType.toAlgKernel s=some final ∧
      BlockState.WriteWithin (RealForwardInner.outputFootprint Y (s.pids 0) N) s final ∧
      (∀ n, n<N → final.readMem Y (s.pids 0*N+n)=RealForwardInner.expected s X (s.pids 0) N n) ∧
      final.pids=s.pids ∧ final.numPids=s.numPids ∧ final.undef=s.undef := by
  rw [←candidate_exec]
  exact original_io input output s Y X M N B hN hB hNB

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardInnerOneTile
