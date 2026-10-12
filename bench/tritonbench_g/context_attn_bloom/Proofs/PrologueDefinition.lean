import bench.tritonbench_g.context_attn_bloom.Proofs.Program
import VeriTile.Triton.Semantics.BroadcastReshape

namespace VeriTile.Bench.TritonBenchG.ContextAttnBloom.RealSetup
open VeriTile.Triton Math Program
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def parts (p : Layout) (r : Regions) (swapped : Bool) (scale : ℝ) : Source.IntRange :=
  match Source.intRange? (mathematical p r swapped scale).surfaceBody "start_n" with
  | some x => x
  | none => ⟨[],.constInt 0,.constInt 0,.constNat 1,[],[]⟩

def lower (code : List ComputeStmt) : List Stmt :=
  ((ComputeStmt.listToAlgorithm? code).toOption.getD []).map Stmt.eraseDType

def prelude (p : Layout) (r : Regions) : List Stmt :=
  lower (parts p r Bool.false 0).before

theorem prelude_shared (p : Layout) (r : Regions) (swapped : Bool) (scale : ℝ) :
    lower (parts p r swapped scale).before = prelude p r := by
  cases swapped <;> rfl

def launch (u : BlockState) : Launch := ⟨u.pids 0,u.pids 1,u.pids 2⟩

def inputs (u : BlockState) (r : Regions) : Inputs where
  q := u.readMem r.query
  k := u.readMem r.key
  v := u.readMem r.value
  start := u.readMemValue .int r.start
  sequence := u.readMemValue .int r.sequence
  prompt := u.readMemValue .int r.prompt
  request := u.readMemValue .int r.request
  tokens := u.readMemValue .int r.tokens

set_option linter.unusedVariables false in
def prologueSource := fun
    (Q K V : RegionName) (sm_scale : ℝ)
    (B_Start_Loc B_Seqlen : RegionName) (Out : RegionName)
    (Req_to_tokens B_req_idx b_prompt_cache_len : RegionName)
    (stride_qbs stride_qh stride_qd
      stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd
      stride_obs stride_oh stride_od
      stride_req_to_tokens_b stride_req_to_tokens_s
      kv_group_num head_dim BLOCK_M BLOCK_DMODEL BLOCK_N : Nat)=> (triton {
  tl.region Q = tl.float32, K = tl.float32, V = tl.float32, Out = tl.float32,
    B_Start_Loc = tl.int32, B_Seqlen = tl.int32, Req_to_tokens = tl.int32,
    B_req_idx = tl.int32, b_prompt_cache_len = tl.int32

  cur_batch = tl.program_id(0)
  cur_head = tl.program_id(1)
  start_m = tl.program_id(2)

  cur_kv_head = cur_head // $(kv_group_num)

  cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
  prompt_cache_len = tl.load(b_prompt_cache_len + cur_batch)
  cur_batch_seq_len = tl.load(B_Seqlen + cur_batch) - prompt_cache_len
  cur_batch_req_idx = tl.load(B_req_idx + cur_batch)

  block_start_loc = $(BLOCK_M) * start_m

  offs_n = tl.arange(0, $(BLOCK_N))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  off_q = (cur_batch_in_all_start_index + offs_m[:, None]) * $(stride_qbs) +
    cur_head * $(stride_qh) + offs_d[None, :] * $(stride_qd)

  q = tl.load(Q + off_q,
    mask=(offs_m[:, None] < cur_batch_seq_len) & (offs_d[None, :] < $(head_dim)),
    other=0.0)

  m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)

  block_mask = tl.where(block_start_loc < cur_batch_seq_len, $(1), $(0))
  block_end_loc = tl.minimum((start_m + $(1)) * $(BLOCK_M) + prompt_cache_len,
    cur_batch_seq_len + prompt_cache_len)

})

def explicitPrelude (p : Layout) (r : Regions) : List Stmt :=
  (prologueSource r.query r.key r.value 0 r.start r.sequence r.out r.tokens r.request r.prompt
    p.qRow p.qHead p.qDim p.kRow p.kHead p.kDim p.vRow p.vHead p.vDim
    p.outRow p.outHead p.outDim p.requestRow p.requestToken p.kvGroup p.headDim p.BM p.D p.BN).eraseDType.toAlgKernel.body

end
end VeriTile.Bench.TritonBenchG.ContextAttnBloom.RealSetup
