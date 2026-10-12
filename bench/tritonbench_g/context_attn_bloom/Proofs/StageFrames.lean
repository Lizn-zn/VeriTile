import bench.tritonbench_g.context_attn_bloom.Proofs.RegisterFrame
import bench.tritonbench_g.context_attn_bloom.Proofs.LoopProgram
import bench.tritonbench_g.context_attn_bloom.Proofs.LoopKeysExecution
namespace VeriTile.Bench.TritonBenchG.ContextAttnBloom.StageFrames
open VeriTile.Triton Math Program RealSetup RegisterFrame
noncomputable section
set_option maxHeartbeats 500000
set_option maxRecDepth 2000

theorem keys (p : Layout) (r : Regions) (sn req last : Int) (kvHead : Nat) (u : BlockState) :
    Frame ["k","off_k","kv_loc","start_n"] u (LoopKeysExecution.prepared p r sn req last kvHead u) := by
  unfold LoopKeysExecution.prepared
  repeat apply write
  exact same u

theorem scores (p : Layout) (swapped : Bool) (scale : ℝ) (rows : Fin p.BM → Nat) (prompt sn : Int)
    (q : Tile .real [p.BM,p.D]) (k : Tile .real [p.D,p.BN]) (u : BlockState) :
    Frame ["qk","qk","qk","qk"] u (ScoreTiles.prepared p swapped scale rows prompt sn q k u) := by
  unfold ScoreTiles.prepared
  repeat apply write
  exact same u

theorem stats (p : Layout) (qk : Tile .real [p.BM,p.BN]) (maximum : Tile .real [p.BM]) (u : BlockState) :
    Frame ["l_ij","p","m_ij"] u (StatsExecution.prepared p qk maximum u) := by
  unfold StatsExecution.prepared
  repeat apply write
  exact same u

theorem normalizer (p : Layout) (mi mij li lij : Tile .real [p.BM]) (u : BlockState) :
    Frame ["l_i_new","beta","alpha","m_i_new"] u (NormalizerExecution.prepared p mi mij li lij u) := by
  unfold NormalizerExecution.prepared
  repeat apply write
  exact same u

theorem scales (p : Layout) (rows : Fin p.BM → Nat) (prompt sn : Int) (li lin alpha beta : Tile .real [p.BM])
    (ptile : Tile .real [p.BM,p.BN]) (acc : Tile .real [p.BM,p.D]) (u : BlockState) :
    Frame ["acc","acc_scale","acc_scale","p","p_scale"] u (ScalesExecution.prepared p rows prompt sn li lin alpha beta ptile acc u) := by
  unfold ScalesExecution.prepared
  repeat apply write
  exact same u

theorem values (p : Layout) (r : Regions) (sn last : Int) (kvHead : Nat) (pages : Fin p.BN → Int) (u : BlockState) :
    Frame ["v","off_v"] u (ValuesExecution.prepared p r sn last kvHead pages u) := by
  unfold ValuesExecution.prepared
  repeat apply write
  exact same u

theorem commit (p : Layout) (prob : Tile .real [p.BM,p.BN]) (value : Tile .real [p.BN,p.D]) (acc : Tile .real [p.BM,p.D])
    (li mi : Tile .real [p.BM]) (u : BlockState) :
    Frame ["m_i","l_i","acc","p"] u (CommitExecution.prepared p prob value acc li mi u) := by
  unfold CommitExecution.prepared
  repeat apply write
  exact same u

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnBloom.StageFrames
