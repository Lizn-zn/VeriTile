import VeriTile.Triton.Core
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.SoftmaxTriton1

open VeriTile.Triton

/-- Faithful 1:1 transcription of `softmax_triton1.py`'s `softmax_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python pointer args → Lean `RegionName` injected via `$(...)`.
- Python `BLOCK_SIZE: tl.constexpr` → Lean `Nat` parameter. -/
def softmax_kernel
    (output_ptr input_ptr : RegionName)
    (input_row_stride output_row_stride n_cols BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  row_idx = tl.program_id(0)
  row_start_ptr = $(input_ptr) + row_idx * $(input_row_stride)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  input_ptrs = row_start_ptr + col_offsets
  row = tl.load(input_ptrs, mask=col_offsets < $(n_cols), other=-inf)
  row_minus_max = row - tl.max(row, axis=0)
  numerator = tl.exp(row_minus_max)
  denominator = tl.sum(numerator, axis=0)
  softmax_output = numerator / denominator
  output_row_start_ptr = $(output_ptr) + row_idx * $(output_row_stride)
  output_ptrs = output_row_start_ptr + col_offsets
  tl.store(output_ptrs, softmax_output, mask=col_offsets < $(n_cols))
}

end VeriTile.Bench.TritonBenchG.SoftmaxTriton1
