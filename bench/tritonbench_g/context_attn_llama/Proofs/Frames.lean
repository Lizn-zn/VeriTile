import bench.tritonbench_g.context_attn_llama.Proofs.Invariant
import bench.tritonbench_g.context_attn_llama.Proofs.KeysExecution
import bench.tritonbench_g.context_attn_llama.Proofs.ScoresExecution
import bench.tritonbench_g.context_attn_llama.Proofs.StatsExecution
import bench.tritonbench_g.context_attn_llama.Proofs.ValuesExecution
namespace VeriTile.Bench.TritonBenchG.ContextAttnLlama.Frames
open VeriTile.Triton Formula Program PreludeComplete
noncomputable section
set_option maxHeartbeats 1200000
set_option maxRecDepth 4000
/-- A local execution fragment changes only the named registers. All tagged
memory cells, launch ids and undefined-load values are preserved exactly. -/
structure Frame (writes : List RegName) (before after : BlockState) : Prop where
  memory : after.mem = before.mem
  pids : after.pids = before.pids
  numPids : after.numPids = before.numPids
  undef : after.undef = before.undef
  registers : ∀ dtype shape name, name ∉ writes → after.regs dtype shape name = before.regs dtype shape name

theorem same (u : BlockState) : Frame [] u u :=
  ⟨rfl,rfl,rfl,rfl,fun _ _ _ _ => rfl⟩

theorem write {writes : List RegName} {before after : BlockState} (h : Frame writes before after)
    (name : RegName) (dtype : TileDType) (shape : TileShape) (value : Tile dtype shape) :
    Frame (name::writes) before (after.setReg name dtype shape value) := by
  constructor
  · funext r o
    simpa only [BlockState.setReg_mem] using congrFun (congrFun h.memory r) o
  · simpa only [BlockState.setReg_pids] using h.pids
  · simpa only [BlockState.setReg_numPids] using h.numPids
  · funext r o
    simpa only [BlockState.setReg_undef] using congrFun (congrFun h.undef r) o
  · intro dt sh nm hn
    have hname : nm ≠ name := by
      intro heq
      exact hn (List.mem_cons.mpr (Or.inl heq))
    have hrest : nm ∉ writes := by
      intro hm
      exact hn (List.mem_cons_of_mem name hm)
    rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hname,h.registers dt sh nm hrest]

theorem then_frame {first second : List RegName} {u v w : BlockState}
    (h1 : Frame first u v) (h2 : Frame second v w) : Frame (second++first) u w := by
  refine ⟨h2.memory.trans h1.memory,h2.pids.trans h1.pids,h2.numPids.trans h1.numPids,
    h2.undef.trans h1.undef,?_⟩
  intro dt sh name hn
  have hs : name ∉ second := fun h => hn (List.mem_append_left first h)
  have hf : name ∉ first := fun h => hn (List.mem_append_right second h)
  rw [h2.registers dt sh name hs,h1.registers dt sh name hf]



def readNames : List RegName :=
  ["cur_batch","cur_head","cur_kv_head","cur_batch_in_all_start_index","prompt_cache_len",
   "cur_batch_seq_len","cur_batch_req_idx","block_end_loc","offs_m","offs_n","offs_d","q"]

theorem preserved (p : Layout) (r : Regions) (initial before after : BlockState) (writes : List RegName)
    (f : Frame writes before after) (h : Invariant.Static p r initial before)
    (hsafe : ∀ name ∈ readNames, name ∉ writes) : Invariant.Static p r initial after := by
  refine ⟨f.memory.trans h.memory,f.pids.trans h.pids,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_⟩
  · exact (f.registers .nat [] "cur_batch" (hsafe _ (by simp [readNames]))).trans h.batch
  · exact (f.registers .nat [] "cur_head" (hsafe _ (by simp [readNames]))).trans h.head
  · exact (f.registers .nat [] "cur_kv_head" (hsafe _ (by simp [readNames]))).trans h.kv_head
  · exact (f.registers .int [] "cur_batch_in_all_start_index" (hsafe _ (by simp [readNames]))).trans h.start
  · exact (f.registers .int [] "prompt_cache_len" (hsafe _ (by simp [readNames]))).trans h.prompt
  · exact (f.registers .int [] "cur_batch_seq_len" (hsafe _ (by simp [readNames]))).trans h.sequence
  · exact (f.registers .int [] "cur_batch_req_idx" (hsafe _ (by simp [readNames]))).trans h.request
  · exact (f.registers .int [] "block_end_loc" (hsafe _ (by simp [readNames]))).trans h.ending
  · exact (f.registers .nat [p.BM] "offs_m" (hsafe _ (by simp [readNames]))).trans h.rows
  · exact (f.registers .nat [p.BN] "offs_n" (hsafe _ (by simp [readNames]))).trans h.lanes
  · exact (f.registers .nat [p.D] "offs_d" (hsafe _ (by simp [readNames]))).trans h.channels
  · exact (f.registers .real [p.BM,p.D] "q" (hsafe _ (by simp [readNames]))).trans h.query

theorem keys (p : Layout) (r : Regions) (head : Nat) (req sn limit : Int) (u : BlockState) :
    Frame ["k","off_k","kv_loc","start_n"] u (KeysExecution.prepared p r head req sn limit u) := by
  unfold KeysExecution.prepared
  repeat' apply write
  exact same u

theorem scores (p : Layout) (swapped : Bool) (scale : ℝ) (rows : Fin p.BM → Nat) (prompt sn : Int)
    (q : Tile .real [p.BM,p.D]) (k : Tile .real [p.D,p.BN]) (u : BlockState) :
    Frame ["qk","mask","qk"] u (ScoresExecution.prepared p swapped scale rows prompt sn q k u) := by
  unfold ScoresExecution.prepared
  repeat' apply write
  exact same u

theorem stats (p : Layout) (qk : Tile .real [p.BM,p.BN]) (mi li rm : Tile .real [p.BM])
    (acc : Tile .real [p.BM,p.D]) (u : BlockState) :
    Frame ["acc","l_i","alpha","l_ij","p","qk","m_ij"] u (StatsExecution.prepared p qk mi li rm acc u) := by
  unfold StatsExecution.prepared
  repeat' apply write
  exact same u

theorem values (p : Layout) (r : Regions) (head : Nat) (pages : Fin p.BN → Int) (sn limit : Int)
    (prob : Tile .real [p.BM,p.BN]) (mx : Tile .real [p.BM]) (acc : Tile .real [p.BM,p.D]) (u : BlockState) :
    Frame ["m_i","acc","p","v","off_v"] u (ValuesExecution.prepared p r head pages sn limit prob mx acc u) := by
  unfold ValuesExecution.prepared ValuesExecution.loadedState
  repeat' apply write
  exact same u

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnLlama.Frames
