import bench.tritonbench_g.softmax_optimize.Proofs.Geometry

/-! Full signed-loop transcription. The signed helper remains negative at N=0;
real correctness is stated separately on a meaningful nonempty shape domain.
Input storage is independent; exp-compatible output/accumulator storage is fp32
or fp64. Destination conversions spell the source store conversion explicitly. -/
namespace VeriTile.Bench.TritonBenchG.SoftmaxOptimize.FullSource
open Lean VeriTile.Triton
set_option linter.unusedVariables false

syntax "softmaxOptimizeFull%" "[" tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(softmaxOptimizeFull%[$inputType:tritonDType, $outputType:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `commuted do
      Macro.throwError "expected original or commuted"
    let rowBase : TSyntax `tritonExpr ← if version.getId == `original then
      `(tritonExpr| pid_m * $(N)) else `(tritonExpr| $(N) * pid_m)
    `(fun (output_ptr input_ptr : RegionName) (M N TILE_N : Nat) =>
      ComputeKernel.mk [input_ptr] [output_ptr] (triton {
  tl.region input_ptr = $inputType, output_ptr = $outputType
  pid_m = tl.program_id(0)
  m = tl.full([$(TILE_N)], value=-float("inf"), dtype=$outputType:tritonDType)
  z = tl.full([$(TILE_N)], value=0, dtype=$outputType:tritonDType)
  prev_multiple = (($((N : Int)) + $((TILE_N : Int)) - $((1 : Int))) // $((TILE_N : Int))) * $((TILE_N : Int)) - $((TILE_N : Int))
  for start_n in range($(0), prev_multiple, $(TILE_N)) {
    n_offsets = start_n + tl.arange(0, $(TILE_N))
    offset = $rowBase + n_offsets
    input_ptrs = input_ptr + offset
    inp = (tl.load(input_ptrs)).to($outputType:tritonDType)
    new_m = tl.maximum(m, inp)
    new_z = tl.exp(m - new_m) * z + tl.exp(inp - new_m)
    m = new_m
    z = new_z
  }
  for start_n in range(prev_multiple, $(N), $(TILE_N)) {
    n_offsets = start_n + tl.arange(0, $(TILE_N))
    offset = $rowBase + n_offsets
    input_ptrs = input_ptr + offset
    mask = n_offsets < $(N)
    inp = (tl.load(input_ptrs, mask=mask, other=-float("inf"))).to($outputType:tritonDType)
    new_m = tl.maximum(m, inp)
    new_z = tl.exp(m - new_m) * z + tl.exp(inp - new_m)
    m = new_m
    z = new_z
  }
  final_m = tl.max(m, 0)
  z = tl.sum(tl.exp(m - final_m) * z)
  m = final_m

  prev_multiple = (($((N : Int)) + $((TILE_N : Int)) - $((1 : Int))) // $((TILE_N : Int))) * $((TILE_N : Int)) - $((TILE_N : Int))
  for start_n in range($(0), prev_multiple, $(TILE_N)) {
    n_offsets = start_n + tl.arange(0, $(TILE_N))
    offset = $rowBase + n_offsets
    input_ptrs = input_ptr + offset
    inp = (tl.load(input_ptrs)).to($outputType:tritonDType)
    e = tl.exp(inp - m)
    out = e / z
    output_ptrs = output_ptr + offset
    tl.store(output_ptrs, out)
  }
  for start_n in range(prev_multiple, $(N), $(TILE_N)) {
    n_offsets = start_n + tl.arange(0, $(TILE_N))
    offset = $rowBase + n_offsets
    input_ptrs = input_ptr + offset
    mask = n_offsets < $(N)
    inp = (tl.load(input_ptrs, mask=mask, other=-float("inf"))).to($outputType:tritonDType)
    e = tl.exp(inp - m)
    out = e / z
    output_ptrs = output_ptr + offset
    tl.store(output_ptrs, out, mask=mask)
  }
}).surfaceBody)

inductive Storage where
  | fp32 | fp16 | bf16 | fp64 | f8e4 | f8e5
  deriving DecidableEq, Repr

/-- Pinned Triton exp admits fp32/fp64; other output dtypes are unsupported source profiles. -/
inductive Accumulation where
  | fp32 | fp64
  deriving DecidableEq, Repr

abbrev Source := RegionName → RegionName → Nat → Nat → Nat → ComputeKernel

def kernel (input : Storage) (output : Accumulation) (v : Bool) : Source :=
  match input, output, v with
  | .fp32, .fp32, Bool.false => softmaxOptimizeFull%[tl.float32, tl.float32, original]
  | .fp32, .fp32, Bool.true => softmaxOptimizeFull%[tl.float32, tl.float32, commuted]
  | .fp32, .fp64, Bool.false => softmaxOptimizeFull%[tl.float32, tl.float64, original]
  | .fp32, .fp64, Bool.true => softmaxOptimizeFull%[tl.float32, tl.float64, commuted]
  | .fp16, .fp32, Bool.false => softmaxOptimizeFull%[tl.float16, tl.float32, original]
  | .fp16, .fp32, Bool.true => softmaxOptimizeFull%[tl.float16, tl.float32, commuted]
  | .fp16, .fp64, Bool.false => softmaxOptimizeFull%[tl.float16, tl.float64, original]
  | .fp16, .fp64, Bool.true => softmaxOptimizeFull%[tl.float16, tl.float64, commuted]
  | .bf16, .fp32, Bool.false => softmaxOptimizeFull%[tl.bfloat16, tl.float32, original]
  | .bf16, .fp32, Bool.true => softmaxOptimizeFull%[tl.bfloat16, tl.float32, commuted]
  | .bf16, .fp64, Bool.false => softmaxOptimizeFull%[tl.bfloat16, tl.float64, original]
  | .bf16, .fp64, Bool.true => softmaxOptimizeFull%[tl.bfloat16, tl.float64, commuted]
  | .fp64, .fp32, Bool.false => softmaxOptimizeFull%[tl.float64, tl.float32, original]
  | .fp64, .fp32, Bool.true => softmaxOptimizeFull%[tl.float64, tl.float32, commuted]
  | .fp64, .fp64, Bool.false => softmaxOptimizeFull%[tl.float64, tl.float64, original]
  | .fp64, .fp64, Bool.true => softmaxOptimizeFull%[tl.float64, tl.float64, commuted]
  | .f8e4, .fp32, Bool.false => softmaxOptimizeFull%[tl.float8e4nv, tl.float32, original]
  | .f8e4, .fp32, Bool.true => softmaxOptimizeFull%[tl.float8e4nv, tl.float32, commuted]
  | .f8e4, .fp64, Bool.false => softmaxOptimizeFull%[tl.float8e4nv, tl.float64, original]
  | .f8e4, .fp64, Bool.true => softmaxOptimizeFull%[tl.float8e4nv, tl.float64, commuted]
  | .f8e5, .fp32, Bool.false => softmaxOptimizeFull%[tl.float8e5, tl.float32, original]
  | .f8e5, .fp32, Bool.true => softmaxOptimizeFull%[tl.float8e5, tl.float32, commuted]
  | .f8e5, .fp64, Bool.false => softmaxOptimizeFull%[tl.float8e5, tl.float64, original]
  | .f8e5, .fp64, Bool.true => softmaxOptimizeFull%[tl.float8e5, tl.float64, commuted]

end VeriTile.Bench.TritonBenchG.SoftmaxOptimize.FullSource

/-! Historical kernel declarations retained verbatim for their existing contracts.
The Nat helper here is a legacy model; FullSource above is the exact signed source. -/
namespace VeriTile.Bench.TritonBenchG.SoftmaxOptimize
open VeriTile.Triton

/-- Faithful transcription of `softmax_optimize.py`'s
`softmax_kernel_online_v2`.

This keeps the online max/sum recurrence over full tiles, the masked tail pass,
the final scalar normalization, and the two writeback loops. -/
def softmax_kernel_online_v2_surface
    (output_ptr input_ptr : RegionName)
    (M N TILE_N : Nat) :
    ComputeKernel := triton {
  pid_m = tl.program_id(0)
  m = tl.full([$(TILE_N)], value=-float("inf"), dtype=output_ptr.dtype.element_ty)
  z = tl.full([$(TILE_N)], value=0, dtype=output_ptr.dtype.element_ty)
  prev_multiple = $((N + TILE_N - 1)) // $(TILE_N) * $(TILE_N) - $(TILE_N)
  for start_n in range($(0), prev_multiple, $(TILE_N)) {
    n_offsets = start_n + tl.arange(0, $(TILE_N))
    offset = pid_m * $(N) + n_offsets
    input_ptrs = input_ptr + offset
    inp = (tl.load(input_ptrs)).to(output_ptr.dtype.element_ty)
    new_m = tl.maximum(m, inp)
    new_z = tl.exp(m - new_m) * z + tl.exp(inp - new_m)
    m = new_m
    z = new_z
  }
  for start_n in range(prev_multiple, $(N), $(TILE_N)) {
    n_offsets = start_n + tl.arange(0, $(TILE_N))
    offset = pid_m * $(N) + n_offsets
    input_ptrs = input_ptr + offset
    mask = n_offsets < $(N)
    inp = (tl.load(input_ptrs, mask=mask, other=-float("inf"))).to(output_ptr.dtype.element_ty)
    new_m = tl.maximum(m, inp)
    new_z = tl.exp(m - new_m) * z + tl.exp(inp - new_m)
    m = new_m
    z = new_z
  }
  final_m = tl.max(m, 0)
  z = tl.sum(tl.exp(m - final_m) * z)
  m = final_m

  prev_multiple = $((N + TILE_N - 1)) // $(TILE_N) * $(TILE_N) - $(TILE_N)
  for start_n in range($(0), prev_multiple, $(TILE_N)) {
    n_offsets = start_n + tl.arange(0, $(TILE_N))
    offset = pid_m * $(N) + n_offsets
    input_ptrs = input_ptr + offset
    inp = (tl.load(input_ptrs)).to(output_ptr.dtype.element_ty)
    e = tl.exp(inp - m)
    out = e / z
    output_ptrs = output_ptr + offset
    tl.store(output_ptrs, out)
  }
  for start_n in range(prev_multiple, $(N), $(TILE_N)) {
    n_offsets = start_n + tl.arange(0, $(TILE_N))
    offset = pid_m * $(N) + n_offsets
    input_ptrs = input_ptr + offset
    mask = n_offsets < $(N)
    inp = (tl.load(input_ptrs, mask=mask, other=-float("inf"))).to(output_ptr.dtype.element_ty)
    e = tl.exp(inp - m)
    out = e / z
    output_ptrs = output_ptr + offset
    tl.store(output_ptrs, out, mask=mask)
  }
}

/-- Proof-oriented one-tile specialization of `softmax_kernel_online_v2`.

When `N <= TILE_N`, the online loops collapse to one masked row tile. This kernel
is kept as the small executable target for the existing algorithm proof. -/
def softmax_kernel_online_v2_one_tile
    (output_ptr input_ptr : RegionName)
    (N TILE_N : Nat) :
    ComputeKernel := triton {
  pid_m = tl.program_id(0)
  n_offsets = tl.arange(0, $(TILE_N))
  offset = pid_m * $(N) + n_offsets
  mask = n_offsets < $(N)
  input_ptrs = input_ptr + offset
  inp = (tl.load(input_ptrs, mask=mask, other=-float("inf"))).to(output_ptr.dtype.element_ty)
  m = tl.max(inp, 0)
  e = tl.exp(inp - m)
  z = tl.sum(e, 0)
  out = e / z
  output_ptrs = output_ptr + offset
  tl.store(output_ptrs, out, mask=mask)
}

end VeriTile.Bench.TritonBenchG.SoftmaxOptimize
