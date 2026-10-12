import bench.tritonbench_g.context_attn_mistral.Proofs.Program
import VeriTile.Triton.Semantics.BroadcastReshape
namespace VeriTile.Bench.TritonBenchG.ContextAttnMistral.Prelude
open VeriTile.Triton Formula Program
set_option maxHeartbeats 5000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def prefixSource (p : Layout) (r : Regions) : ComputeKernel :=
  let Q := r.query
  let K := r.key
  let V := r.value
  let Out := r.out
  let B_Start_Loc := r.start
  let B_Seqlen := r.sequence
  triton {
  tl.region Q = tl.float32, K = tl.float32, V = tl.float32, Out = tl.float32,
    B_Start_Loc = tl.int32, B_Seqlen = tl.int32

  cur_batch = tl.program_id(0)
  cur_head = tl.program_id(1)
  start_m = tl.program_id(2)

  cur_kv_head = cur_head // $(p.kvGroup)

  cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
  cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)

  block_start_loc = $(p.BM) * start_m

  offs_n = tl.arange(0, $(p.BN))
  offs_d = tl.arange(0, $(p.D))
  offs_m = start_m * $(p.BM) + tl.arange(0, $(p.BM))
  off_q = (cur_batch_in_all_start_index + offs_m[:, None]) * $(p.qRow) +
    cur_head * $(p.qHead) + offs_d[None, :] * $(p.qDim)
  off_k = offs_n[None, :] * $(p.kRow) + cur_kv_head * $(p.kHead) +
    offs_d[:, None] * $(p.kDim)
  off_v = offs_n[:, None] * $(p.vRow) + cur_kv_head * $(p.vHead) +
    offs_d[None, :] * $(p.vDim)

  q = tl.load(Q + off_q, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)

  k_ptrs = K + off_k
  v_ptrs = V + off_v

  m_i = tl.zeros([$(p.BM)], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([$(p.BM)], dtype=tl.float32)
  acc = tl.zeros([$(p.BM), $(p.D)], dtype=tl.float32)

  block_mask = tl.where(block_start_loc < cur_batch_seq_len, $(1), $(0))

}

theorem prefix_source (p : Layout) (r : Regions) :
    prelude p r = (prefixSource p r).eraseDType.toAlgKernel.body := by
  simp only [prelude,prefixSource,lower,parts,mathematical,kernel,VeriTile.Bench.TritonBenchG.ContextAttnMistral.original,
    ComputeKernel.surfaceBody,ComputeKernel.eraseDType,ComputeKernel.toAlgorithm?,
    ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType,Except.bind,bind,Except.toOption,Option.getD,
    Kernel.eraseDType,Stmt.eraseDTypeList,List.map_cons,List.map_nil,Stmt.eraseDType,
    Op.eraseDType,MemAccess.eraseDType,MaskOpt.eraseDType,VeriTile.Triton.eraseDType,
    NumericDType.eraseDType,IntegralDType.eraseDType,ComparableDType.eraseDType,
    Source.intRange?,String.reduceEq,Bool.false_eq_true,↓reduceIte,Option.map_some]

end VeriTile.Bench.TritonBenchG.ContextAttnMistral.Prelude
