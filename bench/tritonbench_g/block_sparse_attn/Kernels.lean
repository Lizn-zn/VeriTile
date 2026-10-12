import VeriTile.Triton.DSL
import VeriTile.Triton.Source
import VeriTile.Meta.StatementAudit

/-! Complete typed source of `block_sparse_attn.py`'s
`block_sparse_attention_kernel`. Both sources retain signed CSR metadata and
loop bounds, GQA/head layout, all strides, EVEN_M/EVEN_N load branches, the
one/two-channel branches, causal mask, tl.exp and online normalization.
The candidate changes only qk*softmax_scale to softmax_scale*qk.

Q/K/V/output share the selected storage dtype: fp32 (the supplied Python
example), fp16 or bf16. Scores and accumulators are fp32; probability is cast
back to Q storage before the value dot. Masked loads have no added other
value. NUM_D_BLOCKS>=2 selects exactly one extra channel block, as in Python.
Integer arithmetic models mathematical signed integers, not fixed-width
int32 overflow. This is a per-program source, not a host launch proof.

Translation-surface blocker: storage declarations and the Q dtype cast are
specialized by this macro. The output-pointer conversions are written
explicitly. static_print has a fixed debug payload. The original
Python source remains in this directory. Correct.lean proves both complete
real IO contracts; FPEquiv.lean proves score-scale commutation in context. -/
open Lean
open VeriTile.Triton
namespace VeriTile.Bench.TritonBenchG.BlockSparseAttn.TypedSource

syntax "blockSparseSource%" "[" tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(blockSparseSource%[$storage:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `scalarLeft do
      Macro.throwError "expected original or scalarLeft"
    let scale : TSyntax `tritonExpr ←
      if version.getId == `scalarLeft then `(tritonExpr| $(softmax_scale) * qk)
      else `(tritonExpr| qk * $(softmax_scale))
    `(fun (out Q K V : RegionName)
        (layout_csr_row_indices layout_csr_col_indices : Region .int)
        (layout_csr_row_stride_h layout_csr_col_stride_h num_layout : Nat)
        (softmax_scale : ℝ)
        (stride_qb stride_qh stride_qm stride_kb stride_kh stride_kn
          stride_vb stride_vh stride_vn stride_ob stride_oh stride_om
          num_heads num_kv_heads total_seq_len BLOCK_M BLOCK_N BLOCK_D
          NUM_D_BLOCKS : Nat) (EVEN_M EVEN_N : Bool) =>
    ComputeKernel.mk [Q, K, V, Region.cast layout_csr_row_indices,
      Region.cast layout_csr_col_indices] [out] (triton {
  tl.region Q = $storage, K = $storage, V = $storage, out = $storage
  tl.region layout_csr_row_indices = tl.int32, layout_csr_col_indices = tl.int32
  tl.static_print("block_sparse_attention_kernel")
  q_seq_len = $(total_seq_len)
  start_m = tl.program_id(0)
  off_bh = tl.program_id(1)
  off_h = off_bh % $(num_heads)
  off_b = off_bh // $(num_heads)
  head_groups = $(num_heads) // $(num_kv_heads)
  off_h_kv = off_h // head_groups
  Q += off_b * $(stride_qb) + off_h * $(stride_qh)
  K += off_b * $(stride_kb) + off_h_kv * $(stride_kh)
  V += off_b * $(stride_vb) + off_h_kv * $(stride_vh)
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_n = tl.arange(0, $(BLOCK_N))
  offs_d = tl.arange(0, $(BLOCK_D))
  off_q = offs_m[:, None] * $(stride_qm) + offs_d[None, :]
  off_k = offs_n[None, :] * $(stride_kn) + offs_d[:, None]
  off_v = offs_n[:, None] * $(stride_vn) + offs_d[None, :]
  q_ptrs = Q + off_q
  k_ptrs = K + off_k
  v_ptrs = V + off_v
  m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_D)], dtype=tl.float32)
  if $((NUM_D_BLOCKS >= 2 : Bool)) {
    acc2 = tl.zeros([$(BLOCK_M), $(BLOCK_D)], dtype=tl.float32)
  }
  if EVEN_M {
    q = tl.load(q_ptrs)
    if $((NUM_D_BLOCKS >= 2 : Bool)) {
      q2 = tl.load(q_ptrs + $(BLOCK_D))
    }
  } else {
    q = tl.load(q_ptrs, mask=offs_m[:, None] < q_seq_len)
    if $((NUM_D_BLOCKS >= 2 : Bool)) {
      q2 = tl.load(q_ptrs + $(BLOCK_D), mask=offs_m[:, None] < q_seq_len)
    }
  }
  layout_h = off_h % $(num_layout)
  layout_ptr = layout_csr_row_indices + layout_h * $(layout_csr_row_stride_h) + start_m
  start_l = (tl.load(layout_ptr)).to(tl.int32)
  end_l = (tl.load(layout_ptr + $(1))).to(tl.int32)
  for col_idx_idx in range(start_l, end_l) {
    col_idx = (tl.load(layout_csr_col_indices +
      layout_h * $(layout_csr_col_stride_h) + col_idx_idx)).to(tl.int32)
    start_n = col_idx * $(BLOCK_N)
    if EVEN_N {
      k = tl.load(k_ptrs + start_n * $(stride_kn))
    } else {
      k = tl.load(k_ptrs + start_n * $(stride_kn),
        mask=offs_n[None, :] + start_n < $(total_seq_len))
    }
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    qk += tl.dot(q, k)
    if $((NUM_D_BLOCKS >= 2 : Bool)) {
      if EVEN_N {
        k = tl.load(k_ptrs + start_n * $(stride_kn) + $(BLOCK_D))
      } else {
        k = tl.load(k_ptrs + start_n * $(stride_kn) + $(BLOCK_D),
          mask=offs_n[None, :] + start_n < $(total_seq_len))
      }
      qk += tl.dot(q2, k)
    }
    qk = $scale
    qk += tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), 0, float("-inf"))
    m_ij = tl.max(qk, 1)
    p = tl.exp(qk - m_ij[:, None])
    l_ij = tl.sum(p, 1)
    m_i_new = tl.maximum(m_i, m_ij)
    alpha = tl.exp(m_i - m_i_new)
    beta = tl.exp(m_ij - m_i_new)
    l_i_new = alpha * l_i + beta * l_ij
    p_scale = beta / l_i_new
    p = p * p_scale[:, None]
    acc_scale = l_i / l_i_new * alpha
    acc = acc * acc_scale[:, None]
    if $((NUM_D_BLOCKS >= 2 : Bool)) {
      acc2 = acc2 * acc_scale[:, None]
    }
    p = (p).to($storage:tritonDType)
    if EVEN_N {
      v = tl.load(v_ptrs + start_n * $(stride_vn))
    } else {
      v = tl.load(v_ptrs + start_n * $(stride_vn),
        mask=offs_n[:, None] + start_n < $(total_seq_len))
    }
    acc += tl.dot(p, v)
    if $((NUM_D_BLOCKS >= 2 : Bool)) {
      if EVEN_N {
        v = tl.load(v_ptrs + start_n * $(stride_vn) + $(BLOCK_D))
      } else {
        v = tl.load(v_ptrs + start_n * $(stride_vn) + $(BLOCK_D),
          mask=offs_n[:, None] + start_n < $(total_seq_len))
      }
      acc2 += tl.dot(p, v)
    }
    l_i = l_i_new
    m_i = m_i_new
  }
  off_o = off_b * $(stride_ob) + off_h * $(stride_oh) +
    offs_m[:, None] * $(stride_om) + offs_d[None, :]
  out_ptrs = out + off_o
  tl.store(out_ptrs, (acc).to($storage:tritonDType), mask=offs_m[:, None] < q_seq_len)
  if $((NUM_D_BLOCKS >= 2 : Bool)) {
    tl.store(out_ptrs + $(BLOCK_D), (acc2).to($storage:tritonDType), mask=offs_m[:, None] < q_seq_len)
  }
}).surfaceBody)

inductive Storage where
  | fp32 | fp16 | bf16
  deriving DecidableEq

abbrev KernelSource := RegionName → RegionName → RegionName → RegionName →
  Region .int → Region .int → Nat → Nat → Nat → ℝ →
  Nat → Nat → Nat → Nat → Nat → Nat → Nat → Nat → Nat → Nat → Nat → Nat →
  Nat → Nat → Nat → Nat → Nat → Nat → Nat → Bool → Bool → ComputeKernel

def specialize (storage : Storage) (swapped : Bool) : KernelSource :=
  match storage, swapped with
  | .fp32, Bool.false => blockSparseSource%[tl.float32, original]
  | .fp32, Bool.true => blockSparseSource%[tl.float32, scalarLeft]
  | .fp16, Bool.false => blockSparseSource%[tl.float16, original]
  | .fp16, Bool.true => blockSparseSource%[tl.float16, scalarLeft]
  | .bf16, Bool.false => blockSparseSource%[tl.bfloat16, original]
  | .bf16, Bool.true => blockSparseSource%[tl.bfloat16, scalarLeft]

def original (storage : Storage) : KernelSource := specialize storage Bool.false

def scalarLeft (storage : Storage) : KernelSource := specialize storage Bool.true

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.BlockSparseAttn.TypedSource
