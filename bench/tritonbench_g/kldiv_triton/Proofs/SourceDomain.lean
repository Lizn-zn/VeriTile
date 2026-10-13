import bench.tritonbench_g.kldiv_triton.Kernels

namespace VeriTile.Bench.TritonBenchG.KldivTriton.TypedSource

/-- The default backward source contains the nonzero weak scalar `-1`.
Triton v3.5.1 `semantic.py` lines 60–71 and 185–186 select the tensor dtype
and materialize that scalar directly at the selected dtype. Its `scalar_constant`
(lines 540–549) requires a `get_<dtype>` builder method for nonzero constants;
`python/src/ir.cc` lines 848–868 provides these for bf16/fp16/fp32/fp64, but
lines 912–930 provides only fp8 type constructors, not fp8 value constructors.
Thus the two fp8 target branches remain an abstract AST extension, not a claim
of source-frontend acceptance. This conclusion is from the pinned compiler
source, not a native GPU compilation experiment. Output storage is independent. -/
def BackwardSourceSupported (target : Storage) : Prop :=
  match target with
  | .fp32 | .fp64 | .fp16 | .bf16 => True
  | .f8e4 | .f8e5 => False

example : BackwardSourceSupported .fp16 := trivial
example : ¬ BackwardSourceSupported .f8e4 := id
example : ¬ BackwardSourceSupported .f8e5 := id

end VeriTile.Bench.TritonBenchG.KldivTriton.TypedSource
