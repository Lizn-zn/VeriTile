import VeriTile.Triton.Source
import VeriTile.Triton.DSL
/-! Source: dequantize_rowwise.py, _dequantize_rowwise

Complete _dequantize_rowwise JIT. Signed int8 weights, fp32 row scales and
fp16 output retain the original load masks and the unmasked row-scale load.
The candidate commutes only the exact integer block-start multiplication. -/
namespace VeriTile.Bench.TritonBenchG.DequantizeRowwise.TypedSource
open Lean VeriTile.Triton
syntax "dequantizeRowwiseSource%" ident : term
set_option hygiene false in
macro_rules
  | `(dequantizeRowwiseSource% $version:ident) => do
    unless version.getId == `original || version.getId == `indexCommuted do
      Macro.throwError "expected original or indexCommuted"
    let indexStmt : TSyntax `tritonStmt ←
      if version.getId == `indexCommuted then `(tritonStmt| block_start = $(BLOCK_SIZE) * pid)
      else `(tritonStmt| block_start = pid * $(BLOCK_SIZE))
    `(fun (x_ptr state_x output_ptr : RegionName) (inv_127 : ℝ) (n_elements BLOCK_SIZE P2 : Nat) =>
      ComputeKernel.mk [x_ptr,state_x] [output_ptr] (triton {
  tl.region x_ptr = tl.int8, state_x = tl.float32, output_ptr = tl.float16
  pid = tl.program_id(axis=0)
  $indexStmt
  arange = tl.arange(0, $(P2))
  offsets = block_start + arange
  row_mask = arange < $(BLOCK_SIZE)
  x = tl.load(x_ptr + offsets, mask=row_mask)
  max_val = tl.load(state_x + pid)
  output = max_val * x * $((inv_127 : ℝ))
  tl.store(output_ptr + offsets, (output).to(tl.float16), mask=row_mask)
}).surfaceBody)
def original := dequantizeRowwiseSource% original
def indexCommuted := dequantizeRowwiseSource% indexCommuted
end VeriTile.Bench.TritonBenchG.DequantizeRowwise.TypedSource
