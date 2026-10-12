import bench.tritonbench_g.context_attn_nopad.Proofs.Invariant
import bench.tritonbench_g.context_attn_nopad.Proofs.KeysExecution
import bench.tritonbench_g.context_attn_nopad.Proofs.ScoresExecution
import bench.tritonbench_g.context_attn_nopad.Proofs.StatsExecution
import bench.tritonbench_g.context_attn_nopad.Proofs.NormalizerExecution
import bench.tritonbench_g.context_attn_nopad.Proofs.ScalesExecution
import bench.tritonbench_g.context_attn_nopad.Proofs.ValuesExecution
import bench.tritonbench_g.context_attn_nopad.Proofs.CommitExecution
import bench.tritonbench_g.context_attn_fwd.Proofs.Frames
namespace VeriTile.Bench.TritonBenchG.ContextAttnNopad.Frames
open VeriTile.Triton Formula Program PreludeComplete
open VeriTile.Bench.TritonBenchG.ContextAttnFwd.Frames (Frame same write then_frame)
noncomputable section
set_option maxHeartbeats 1200000
set_option maxRecDepth 4000
def readNames : List RegName :=
  ["cur_batch","cur_head","cur_batch_in_all_start_index",
   "cur_batch_seq_len","offs_m","offs_n","offs_d","q","k_ptrs","v_ptrs"]

theorem preserved (p : Layout) (r : Regions) (initial before after : BlockState) (writes : List RegName)
    (f : Frame writes before after) (h : Invariant.Static p r initial before)
    (hsafe : ∀ name ∈ readNames, name ∉ writes) : Invariant.Static p r initial after := by
  refine ⟨f.memory.trans h.memory,f.pids.trans h.pids,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_⟩
  · exact (f.registers .nat [] "cur_batch" (hsafe _ (by simp [readNames]))).trans h.batch
  · exact (f.registers .nat [] "cur_head" (hsafe _ (by simp [readNames]))).trans h.head
  · exact (f.registers .int [] "cur_batch_in_all_start_index" (hsafe _ (by simp [readNames]))).trans h.start
  · exact (f.registers .int [] "cur_batch_seq_len" (hsafe _ (by simp [readNames]))).trans h.sequence
  · exact (f.registers .nat [p.BM] "offs_m" (hsafe _ (by simp [readNames]))).trans h.rows
  · exact (f.registers .nat [p.BN] "offs_n" (hsafe _ (by simp [readNames]))).trans h.lanes
  · exact (f.registers .nat [p.D] "offs_d" (hsafe _ (by simp [readNames]))).trans h.channels
  · exact (f.registers .real [p.BM,p.D] "q" (hsafe _ (by simp [readNames]))).trans h.query

  · exact (f.registers .ptr [p.D,p.BN] "k_ptrs" (hsafe _ (by simp [readNames]))).trans h.key_ptrs
  · exact (f.registers .ptr [p.BN,p.D] "v_ptrs" (hsafe _ (by simp [readNames]))).trans h.value_ptrs

theorem keys (p : Layout) (r : Regions) (head : Nat) (start sn limit : Int) (u : BlockState) :
    Frame ["k","start_n"] u (KeysExecution.prepared p r head start sn limit u) := by
  unfold KeysExecution.prepared
  repeat' apply write
  exact same u

theorem scores (p : Layout) (swapped : Bool) (scale : ℝ) (rows : Fin p.BM → Nat) (sn : Int)
    (q : Tile .real [p.BM,p.D]) (k : Tile .real [p.D,p.BN]) (u : BlockState) :
    Frame ["qk","qk","qk","qk"] u (ScoresExecution.prepared p swapped scale rows sn q k u) := by
  unfold ScoresExecution.prepared
  repeat' apply write
  exact same u

theorem stats (p : Layout) (qk : Tile .real [p.BM,p.BN]) (rm : Tile .real [p.BM]) (u : BlockState) :
    Frame ["l_ij","p","m_ij"] u (StatsExecution.prepared p qk rm u) := by
  unfold StatsExecution.prepared
  repeat' apply write
  exact same u

theorem normalizer (p : Layout) (mi mij li lij : Tile .real [p.BM]) (u : BlockState) :
    Frame ["l_i_new","beta","alpha","m_i_new"] u (NormalizerExecution.prepared p mi mij li lij u) := by
  unfold NormalizerExecution.prepared
  repeat' apply write
  exact same u

theorem scales (p : Layout) (li lin alpha beta : Tile .real [p.BM])
    (prob : Tile .real [p.BM,p.BN]) (acc : Tile .real [p.BM,p.D]) (u : BlockState) :
    Frame ["acc","acc_scale","p","p_scale"] u (ScalesExecution.prepared p li lin alpha beta prob acc u) := by
  unfold ScalesExecution.prepared
  repeat' apply write
  exact same u

theorem values (p : Layout) (r : Regions) (head : Nat) (start sn limit : Int) (u : BlockState) :
    Frame ["v"] u (ValuesExecution.prepared p r head start sn limit u) := by
  unfold ValuesExecution.prepared
  repeat' apply write
  exact same u

theorem commit (p : Layout) (prob : Tile .real [p.BM,p.BN]) (value : Tile .real [p.BN,p.D])
    (acc : Tile .real [p.BM,p.D]) (li mi : Tile .real [p.BM]) (u : BlockState) :
    Frame ["m_i","l_i","acc","p"] u (CommitExecution.prepared p prob value acc li mi u) := by
  unfold CommitExecution.prepared
  repeat' apply write
  exact same u

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnNopad.Frames
