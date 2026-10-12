import bench.tritonbench_g.context_attn_bloom.Proofs.StageFrames
namespace VeriTile.Bench.TritonBenchG.ContextAttnBloom.StaticFrame
open VeriTile.Triton Math Program RealSetup RegisterFrame

def readNames : List RegName :=
  ["cur_batch","cur_head","cur_kv_head","cur_batch_in_all_start_index","prompt_cache_len",
   "cur_batch_seq_len","block_end_loc","cur_batch_req_idx","offs_m","offs_n","offs_d","q"]

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
  · exact (f.registers .int [] "block_end_loc" (hsafe _ (by simp [readNames]))).trans h.ending
  · exact (f.registers .int [] "cur_batch_req_idx" (hsafe _ (by simp [readNames]))).trans h.request
  · exact (f.registers .nat [p.BM] "offs_m" (hsafe _ (by simp [readNames]))).trans h.rows
  · exact (f.registers .nat [p.BN] "offs_n" (hsafe _ (by simp [readNames]))).trans h.lanes
  · exact (f.registers .nat [p.D] "offs_d" (hsafe _ (by simp [readNames]))).trans h.channels
  · exact (f.registers .real [p.BM,p.D] "q" (hsafe _ (by simp [readNames]))).trans h.query

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ContextAttnBloom.StaticFrame
