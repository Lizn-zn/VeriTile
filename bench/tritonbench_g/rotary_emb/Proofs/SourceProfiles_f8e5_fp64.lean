import bench.tritonbench_g.rotary_emb.Proofs.SourceTypes

namespace VeriTile.Bench.TritonBenchG.RotaryEmb.FullSource
open Lean VeriTile.Triton
set_option linter.unusedVariables false
set_option maxHeartbeats 16000000
set_option maxRecDepth 16000

def template_f8e5_fp64 (sin : Storage) : Source :=
  match sin with
  | .fp32 => rotaryFull%[tl.float8e5, tl.float64, tl.float32]
  | .fp64 => rotaryFull%[tl.float8e5, tl.float64, tl.float64]
  | .fp16 => rotaryFull%[tl.float8e5, tl.float64, tl.float16]
  | .bf16 => rotaryFull%[tl.float8e5, tl.float64, tl.bfloat16]
  | .f8e4 => rotaryFull%[tl.float8e5, tl.float64, tl.float8e4nv]
  | .f8e5 => rotaryFull%[tl.float8e5, tl.float64, tl.float8e5]

end VeriTile.Bench.TritonBenchG.RotaryEmb.FullSource
