import VeriTile.Triton

/-! `add_kernel` from vector_addition. The original preserves the complete per-program body
from vector_addition.py. The candidate changes only the operand order of addition;
this is a rewrite example, with no performance claim. The host launch is outside
the proofs. Element count and block size remain symbolic. -/

namespace VeriTile.Bench.TritonBenchG.VectorAddition
open VeriTile.Triton

/-- Faithful transcription of `vector_addition.py`'s `add_kernel`. -/
def add_kernel
    (x_ptr y_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  x = tl.load(x_ptr + offsets, mask=mask)
  y = tl.load(y_ptr + offsets, mask=mask)
  output = x + y
  tl.store(output_ptr + offsets, output, mask=mask)
}

/-- Candidate with the same loads, mask and store, adding y to x. -/
def add_commuted_kernel
    (x_ptr y_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  x = tl.load(x_ptr + offsets, mask=mask)
  y = tl.load(y_ptr + offsets, mask=mask)
  output = y + x
  tl.store(output_ptr + offsets, output, mask=mask)
}

end VeriTile.Bench.TritonBenchG.VectorAddition
