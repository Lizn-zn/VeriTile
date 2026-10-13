import bench.tritonbench_g.flash_attn.Proofs.RealWritebackMath
import bench.tritonbench_g.flash_attn.Proofs.RealPostMemory

/-! Successful execution and complete memory agreement with the independent
attention formula. Both final stores and all source loop iterations are covered. -/
namespace VeriTile.Bench.TritonBenchG.FlashAttn.RealExecution
open VeriTile.Triton Formula NormalizedSource
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3
noncomputable section

theorem algorithm_correct (Q K V L O : RegionName) (p : Layout) (s : BlockState)
    (hBN : 0 < p.BLOCK_N) (hhi : 0 < hi p (s.pids 0)) :
    ∃ u, exec (algorithm Q K V L O p) s = some u ∧
      u.mem = (referenceState p s Q K V L O).mem := by
  let initial := entry Q K V L O p s
  have hinit := entry_invariant p s Q K V L O
  obtain ⟨t,ht,hinv⟩ := loop_invariant p s Q K V initial.val hBN hinit
  let c := Geometry.tiles (hi p (s.pids 0)) p.BLOCK_N
  let result := post Q K V L O p t (s.pids 0) (s.pids 1) (base p (s.pids 1))
    (Tile.vec (fun i => row p (s.pids 0) i.val)) (prefixMax p s Q K V c)
    (prefixDen p s Q K V c) (prefixAcc p s Q K V c)
    hinv.startm hinv.bshead hinv.base hinv.offm hinv.maximum hinv.denominator hinv.accumulator
  refine ⟨result.val, ?_, ?_⟩
  · unfold exec
    rw [decomposition, stepStmts.append_some initial.property,
      stepStmts.cons_some ht]
    exact result.property
  · have hm := post_mem Q K V L O p s t (prefixMax p s Q K V c)
      (prefixDen p s Q K V c) (prefixAcc p s Q K V c)
      hinv.startm hinv.bshead hinv.base hinv.offm hinv.maximum hinv.denominator hinv.accumulator
    have hn : normalizedTile p (prefixDen p s Q K V c) (prefixAcc p s Q K V c) =
        RealTiles.finite (fun i : TileIndex [p.BLOCK_M,p.DIM] =>
          output p s.readMem Q K V (s.pids 0) (s.pids 1) i.1.val i.2.1.val) :=
      final_output_tile p s Q K V hhi hBN
    have hl : logsumTile p (prefixMax p s Q K V c) (prefixDen p s Q K V c) =
        RealTiles.finite (fun i : TileIndex [p.BLOCK_M] =>
          lse p s.readMem Q K (s.pids 0) (s.pids 1) i.1.val) :=
      final_lse_tile p s Q K V hhi hBN
    rw [hn, hl] at hm
    change result.val.mem = (writeState p (s.pids 0) (s.pids 1) L O
      (fun i => lse p s.readMem Q K (s.pids 0) (s.pids 1) i.val)
      (fun i => output p s.readMem Q K V (s.pids 0) (s.pids 1) i.1.val i.2.1.val) t).mem at hm
    exact hm.trans (writeState_mem_congr p (s.pids 0) (s.pids 1) L O _ _ s t hinv.mem)

theorem algorithm_correct_frame (Q K V L O : RegionName) (p : Layout) (s : BlockState)
    (hBN : 0 < p.BLOCK_N) (hhi : 0 < hi p (s.pids 0)) :
    ∃ u, exec (algorithm Q K V L O p) s = some u ∧
      u.mem = (referenceState p s Q K V L O).mem ∧
      ∀ r o, ¬writeWindow p L O (s.pids 0) (s.pids 1) r o → u.mem r o = s.mem r o := by
  obtain ⟨u,hu,hm⟩ := algorithm_correct Q K V L O p s hBN hhi
  refine ⟨u,hu,hm,?_⟩
  intro r o h
  rw [hm]
  exact referenceState_frame p s Q K V L O r o h

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.FlashAttn.RealExecution
