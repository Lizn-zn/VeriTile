import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.MeanReduction
open Lean VeriTile.Triton

/-- Faithful transcription of `mean_reduction.py`'s `mean_dim_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `[:, None]` / `[None, :]` dimension annotations preserved.
- Python `BLOCK_M` / `BLOCK_N: tl.constexpr` → Lean `Nat` parameters.

The proof below connects the full-row spec to a loop invariant for the
`for off in range(...)` accumulation.
-/
def mean_dim_kernel
    (X Mean : RegionName)
    (M N BLOCK_M BLOCK_N : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0) * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))[:, None]
  X = X + pid * $(N)
  Mean = Mean + pid
  row_mask = pid < $(M)
  _mean = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
  for off in range(0, $(N), $(BLOCK_N)) {
    cols = off + tl.arange(0, $(BLOCK_N))[None, :]
    col_mask = cols < $(N)
    mask = row_mask and col_mask
    a = tl.load(X + cols, mask, other=0.0).to(tl.float32)
    _mean += a
  }
  mean = tl.sum(_mean, axis=1) / $(N)
  mean = mean[:, None]
  tl.store(Mean, mean, row_mask)
}

/-! Typed source family. The candidate commutes only the natural-number
program-base product. Loaded values still explicitly convert to fp32 before
the complete accumulation loop; the output conversion makes the store dtype
explicit after pointer rebinding. -/
syntax "meanSource%" "[" tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(meanSource%[$input:tritonDType, $output:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `indexCommuted do
      Macro.throwError "expected original or indexCommuted"
    let pid : TSyntax `tritonStmt ←
      if version.getId == `original then
        `(tritonStmt| pid = tl.program_id(0) * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))[:, None])
      else
        `(tritonStmt| pid = $(BLOCK_M) * tl.program_id(0) + tl.arange(0, $(BLOCK_M))[:, None])
    let write : TSyntax `tritonStmt ←
      match output with
      | `(tritonDType| tl.float32) => `(tritonStmt| tl.store(Mean, mean, row_mask))
      | _ => `(tritonStmt| tl.store(Mean, mean.to($output), row_mask))
    `(fun (X Mean : RegionName) (M N BLOCK_M BLOCK_N : Nat) =>
      ComputeKernel.mk [X] [Mean] (triton {
  tl.region X = $input, Mean = $output
  $pid
  X = X + pid * $(N)
  Mean = Mean + pid
  row_mask = pid < $(M)
  _mean = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
  for off in range(0, $(N), $(BLOCK_N)) {
    cols = off + tl.arange(0, $(BLOCK_N))[None, :]
    col_mask = cols < $(N)
    mask = row_mask and col_mask
    a = tl.load(X + cols, mask, other=0.0).to(tl.float32)
    _mean += a
  }
  mean = tl.sum(_mean, axis=1) / $(N)
  mean = mean[:, None]
  $write
}).surfaceBody)

inductive Storage where
  | fp32 | fp16 | bf16 | fp64
  deriving DecidableEq

abbrev Source := RegionName → RegionName → Nat → Nat → Nat → Nat → ComputeKernel

syntax "meanOutputs%" "[" tritonDType "," ident "]" : term
macro_rules
  | `(meanOutputs%[$input:tritonDType, $version:ident]) =>
    `(fun output : Storage => match output with
      | .fp32 => meanSource%[$input, tl.float32, $version]
      | .fp16 => meanSource%[$input, tl.float16, $version]
      | .bf16 => meanSource%[$input, tl.bfloat16, $version]
      | .fp64 => meanSource%[$input, tl.float64, $version])

def typedKernel (input output : Storage) (commuted : Bool) : Source :=
  match input, commuted with
  | .fp32, Bool.false => meanOutputs%[tl.float32, original] output
  | .fp32, Bool.true => meanOutputs%[tl.float32, indexCommuted] output
  | .fp16, Bool.false => meanOutputs%[tl.float16, original] output
  | .fp16, Bool.true => meanOutputs%[tl.float16, indexCommuted] output
  | .bf16, Bool.false => meanOutputs%[tl.bfloat16, original] output
  | .bf16, Bool.true => meanOutputs%[tl.bfloat16, indexCommuted] output
  | .fp64, Bool.false => meanOutputs%[tl.float64, original] output
  | .fp64, Bool.true => meanOutputs%[tl.float64, indexCommuted] output

end VeriTile.Bench.TritonBenchG.MeanReduction
