
import torch
import triton
import triton.language as tl

@triton.jit
def embedding_kernel(
    weight,
    input_ids,
    out,
    vob_start_id,
    vob_end_id,
    stride_weight_seq,
    stride_out_seq,
    n_ctx,
    hiden_size: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_NN: tl.constexpr,
):
    start_n = tl.program_id(0) * BLOCK_N

    offs_nn = start_n + tl.arange(0, BLOCK_NN)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    for start_nn in range(0, BLOCK_N, BLOCK_NN):
        start_nn = tl.multiple_of(start_nn, BLOCK_NN)
        offs_seq = start_nn + offs_nn
        n_ctx_mask = offs_seq < n_ctx
        token_ids = tl.load(input_ids + offs_seq, mask=n_ctx_mask, other=vob_end_id)
        id_mask = (token_ids >= vob_start_id) & (token_ids < vob_end_id)
        token_ids = token_ids - vob_start_id
        dim_mask = offs_d < hiden_size
        load_mask = id_mask[:, None] & dim_mask[None, :]
        store_mask = n_ctx_mask[:, None] & dim_mask[None, :]
        vecs = tl.load(weight + token_ids[:, None] * stride_weight_seq + offs_d[None, :], mask=load_mask, other=0.0)
        tl.store(out + offs_seq[:, None] * stride_out_seq + offs_d[None, :], vecs, mask=store_mask)

@torch.no_grad()
def embedding(input_ids, weight: torch.Tensor, vob_start_id, vob_end_id, out: torch.Tensor):
    BLOCK_N = 64
    BLOCK_NN = 1
    BLOCK_DMODEL = triton.next_power_of_2(weight.shape[1])
    n_ctx = input_ids.shape[0]

    grid = (triton.cdiv(n_ctx, BLOCK_N), 1, 1)

    embedding_kernel[grid](
        weight,
        input_ids,
        out,
        vob_start_id,
        vob_end_id,
        weight.stride(0),
        out.stride(0),
        n_ctx=n_ctx,
        hiden_size=weight.shape[1],
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_N=BLOCK_N,
        BLOCK_NN=BLOCK_NN,
        num_warps=1,
        num_stages=1,
    )




##################################################################################################################################################


import torch

def test_embedding():
    # Parameter definitions
    vocab_size = 1000         # Vocabulary size
    embedding_dim = 512       # Embedding dimension
    sequence_length = 128     # Input sequence length
    vob_start_id = 10         # Vocabulary start ID
    vob_end_id = 1000         # Vocabulary end ID

    # Create the test input tensors.
    input_ids = torch.randint(
        vob_start_id, vob_end_id, (sequence_length,), dtype=torch.int32, device='cuda'
    )
    weight = torch.randn(
        vocab_size, embedding_dim, dtype=torch.float32, device='cuda'
    )
    out = torch.zeros(
        sequence_length, embedding_dim, dtype=torch.float32, device='cuda'
    )

    # Call the embedding function.
    embedding(input_ids, weight, vob_start_id, vob_end_id, out)

    # Save the results.
    results = {}
    results['test_case_1'] = out.clone()

    # Test different inputs.
    input_ids = torch.randint(
        vob_start_id, vob_end_id, (sequence_length,), dtype=torch.int32, device='cuda'
    )
    embedding(input_ids, weight, vob_start_id, vob_end_id, out)
    results['test_case_2'] = out.clone()

    # Test different vocabulary ranges.
    vob_start_id = 0
    vob_end_id = 500
    input_ids = torch.randint(
        vob_start_id, vob_end_id, (sequence_length,), dtype=torch.int32, device='cuda'
    )
    embedding(input_ids, weight, vob_start_id, vob_end_id, out)
    results['test_case_3'] = out.clone()

    # Test different embedding dimensions.
    embedding_dim = 256
    weight = torch.randn(
        vocab_size, embedding_dim, dtype=torch.float32, device='cuda'
    )
    out = torch.zeros(
        sequence_length, embedding_dim, dtype=torch.float32, device='cuda'
    )
    embedding(input_ids, weight, vob_start_id, vob_end_id, out)
    results['test_case_4'] = out.clone()

    return results

result_gold = test_embedding()
