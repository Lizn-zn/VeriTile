import torch
import triton
import triton.language as tl


@triton.jit
def kldivergence_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x * tl.log(x / y)
    tl.store(output_ptr + offsets, output, mask=mask)


def kldivergence(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    kldivergence_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


##################################################################################################################################################


def test_kldivergence():
    results = {}
    x1 = torch.rand(1024, device="cuda")
    y1 = torch.rand(1024, device="cuda")
    results["test_case_1"] = kldivergence(x1, y1)

    x2 = torch.rand(2048, device="cuda")
    y2 = torch.rand(2048, device="cuda")
    results["test_case_2"] = kldivergence(x2, y2)

    x3 = torch.rand(4096, device="cuda")
    y3 = torch.rand(4096, device="cuda")
    results["test_case_3"] = kldivergence(x3, y3)

    x4 = torch.rand(8192, device="cuda")
    y4 = torch.rand(8192, device="cuda")
    results["test_case_4"] = kldivergence(x4, y4)

    return results


result_gold = test_kldivergence()
