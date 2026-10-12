import VeriTile.Triton.DSL

/-! Full source of `batched_vecmat_mult.py`'s `batched_vecmat_kernel`.
A/B/output and accumulation are fp32, as in the supplied wrapper. Each
program computes one BM×BN tile by streaming unmasked BM×BK and BN×BM×BK
input tiles. The candidate swaps only the operands of each vecmat addition;
broadcast, axis-2 reduction, transpose, loop order and store are unchanged.
The host requires whole blocks. The public real contracts describe all actual
input/output windows; host launch composition and fixed-width address overflow
are outside the per-program model. -/
namespace VeriTile.Bench.TritonBenchG.BatchedVecmatMult.TypedSource
open VeriTile.Triton

def original
    (A B output : RegionName)
    (_dim_m dim_n dim_k BLOCK_M BLOCK_N BLOCK_K : Nat) :
    ComputeKernel := triton {
  tl.region A = tl.float32, B = tl.float32, output = tl.float32
  m_index = tl.program_id(0)
  n_index = tl.program_id(1)
  output_tile = (m_index * $(BLOCK_M) + tl.arange(0, $(BLOCK_M)))[:, None] * $(dim_n) +
    (n_index * $(BLOCK_N) + tl.arange(0, $(BLOCK_N)))[None, :]
  vecmat = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=A.dtype.element_ty)
  k_blocks = $(dim_k) // $(BLOCK_K)
  for k_index in range(k_blocks) {
    a_tile = (m_index * $(BLOCK_M) + tl.arange(0, $(BLOCK_M)))[:, None] * $(dim_k) +
      (k_index * $(BLOCK_K) + tl.arange(0, $(BLOCK_K)))[None, :]
    a = tl.load(A + a_tile)
    b_tile = (m_index * $(BLOCK_M) + tl.arange(0, $(BLOCK_M)))[None, :, None] *
      $(dim_n) * $(dim_k) +
      (n_index * $(BLOCK_N) + tl.arange(0, $(BLOCK_N)))[:, None, None] * $(dim_k) +
      (k_index * $(BLOCK_K) + tl.arange(0, $(BLOCK_K)))[None, None, :]
    b = tl.load(B + b_tile)
    expanded_a, _ = tl.broadcast(a, b)
    vecmat += tl.trans(tl.sum(expanded_a * b, axis=2))
  }
  tl.store(output + output_tile, vecmat)
}

def blockFirst
    (A B output : RegionName)
    (_dim_m dim_n dim_k BLOCK_M BLOCK_N BLOCK_K : Nat) :
    ComputeKernel := triton {
  tl.region A = tl.float32, B = tl.float32, output = tl.float32
  m_index = tl.program_id(0)
  n_index = tl.program_id(1)
  output_tile = (m_index * $(BLOCK_M) + tl.arange(0, $(BLOCK_M)))[:, None] * $(dim_n) +
    (n_index * $(BLOCK_N) + tl.arange(0, $(BLOCK_N)))[None, :]
  vecmat = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=A.dtype.element_ty)
  k_blocks = $(dim_k) // $(BLOCK_K)
  for k_index in range(k_blocks) {
    a_tile = (m_index * $(BLOCK_M) + tl.arange(0, $(BLOCK_M)))[:, None] * $(dim_k) +
      (k_index * $(BLOCK_K) + tl.arange(0, $(BLOCK_K)))[None, :]
    a = tl.load(A + a_tile)
    b_tile = (m_index * $(BLOCK_M) + tl.arange(0, $(BLOCK_M)))[None, :, None] *
      $(dim_n) * $(dim_k) +
      (n_index * $(BLOCK_N) + tl.arange(0, $(BLOCK_N)))[:, None, None] * $(dim_k) +
      (k_index * $(BLOCK_K) + tl.arange(0, $(BLOCK_K)))[None, None, :]
    b = tl.load(B + b_tile)
    expanded_a, _ = tl.broadcast(a, b)
    vecmat = tl.trans(tl.sum(expanded_a * b, axis=2)) + vecmat
  }
  tl.store(output + output_tile, vecmat)
}


end VeriTile.Bench.TritonBenchG.BatchedVecmatMult.TypedSource
