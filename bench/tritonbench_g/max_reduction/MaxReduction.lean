import VeriTile.Triton.Core
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.MaxReduction

open VeriTile.Triton

/-- Faithful 1:1 transcription of `max_reduction.py`'s `max_kernel_1`.

Allowed mechanical Lean-syntax-only changes:
- Python `=` → Lean `:=` for register binding.
- Python pointer args → Lean `RegionName` injected via `$(...)`.
- Python `BLOCK_SIZE: tl.constexpr` → Lean `Nat` parameter. -/
def max_kernel_1
    (inp mid : RegionName)
    (M BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid := tl.program_id(0)
  offset := pid * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
  inp_ptrs := $(inp) + offset
  mask := offset < $(M)
  inp_val := tl.load(inp_ptrs, mask=mask, other=-inf)
  max_val := tl.max(inp_val)
  mid_ptr := $(mid) + pid
  tl.store(mid_ptr, max_val)
}

/-- Faithful 1:1 transcription of `max_reduction.py`'s `max_kernel_2`.

Same allowed Lean-syntax-only changes as `max_kernel_1`. -/
def max_kernel_2
    (mid out : RegionName)
    (mid_size BLOCK_MID : Nat) :
    ComputeKernel := triton {
  offset := tl.arange(0, $(BLOCK_MID))
  mid_ptrs := $(mid) + offset
  mask := offset < $(mid_size)
  mid_val := tl.load(mid_ptrs, mask=mask, other=-inf)
  max_val := tl.max(mid_val)
  tl.store($(out), max_val)
}

/-- Faithful 1:1 transcription of `max_reduction.py`'s `max_kernel`
(autotuned, returns value + index via `tl.max(..., return_indices=True)`).

Allowed mechanical Lean-syntax-only changes apply. The `@triton.autotune`
and `@triton.heuristics` decorators that wrap `max_kernel` are not DSL-side
constructs (they configure the launch grid, not the kernel body); they have
no in-kernel transcription. -/
def max_kernel
    (inp out_value out_index : RegionName)
    (M N K BLOCK_M BLOCK_N : Nat) :
    ComputeKernel := triton {
  pid_m := tl.program_id(0)
  pid_k := tl.program_id(1)
  m_offset := pid_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  n_offset := tl.arange(0, $(BLOCK_N))
  offset := m_offset[:, None] * $(N) * $(K) + n_offset[None, :] * $(K) + pid_k
  offset_index := m_offset * $(K) + pid_k
  mask1 := m_offset < $(M)
  mask := m_offset[:, None] < $(M) and n_offset[None, :] < $(N)
  inp_ptrs := $(inp) + offset
  inp_vals := tl.load(inp_ptrs, mask=mask, other=-inf)
  result_value, result_index := tl.max(inp_vals, axis=1, return_indices=True)
  out_value_ptrs := $(out_value) + offset_index
  out_index_ptrs := $(out_index) + offset_index
  tl.store(out_value_ptrs, result_value, mask=mask1)
  tl.store(out_index_ptrs, result_index, mask=mask1)
}

end VeriTile.Bench.TritonBenchG.MaxReduction
