import VeriTile.Triton.DSL

/-! Complete forward/backward FlagGems source family, retaining both constexpr
branches and all loops. The legacy one-tile definitions remain untouched.
Typed output conversion spells the source store's implicit conversion explicitly.
Host launch composition and IEEE behavior are not claimed by source elaboration. -/
namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.FullSource
open Lean VeriTile.Triton

syntax "flaggems0%" "[" tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(flaggems0%[$inputType:tritonDType, $outputType:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `address_commuted do
      Macro.throwError "expected original or address_commuted"
    let programBase : TSyntax `tritonExpr ←
      if version.getId == `address_commuted then
        `(tritonExpr| $(TILE_K) * pid_k)
      else `(tritonExpr| pid_k * $(TILE_K))
    `(fun (output_ptr input_ptr : RegionName)
        (M N K TILE_N TILE_K : Nat) (ONE_TILE_PER_CTA : Bool) => triton {
  tl.region output_ptr = $outputType, input_ptr = $inputType
  pid_k = tl.program_id(1)
  pid_m = tl.program_id(0)
  k_offsets = $programBase + tl.arange(0, $(TILE_K))
  if ONE_TILE_PER_CTA {
    n_offsets = tl.arange(0, $(TILE_N))
    offset = pid_m * $(N) * $(K) + n_offsets[:, None] * $(K) + k_offsets
    mask = (n_offsets[:, None] < $(N)) & (k_offsets < $(K))
    input_ptrs = input_ptr + offset
    inp = tl.load(input_ptrs, mask=mask, other=-float("inf"))
    m = tl.max(inp, 0)
    e = tl.exp(inp - m[None, :])
    z = tl.sum(e, 0)
    out = e / z
    output_ptrs = output_ptr + offset
    tl.store(output_ptrs, (out).to($outputType:tritonDType), mask=mask)
  } else {
    m = tl.full([$(TILE_N), $(TILE_K)], value=float("-inf"), dtype=tl.float32)
    z = tl.full([$(TILE_N), $(TILE_K)], value=0.0, dtype=tl.float32)
    for start_n in range($(0), $(N), $(TILE_N)) {
      n_offsets = start_n + tl.arange(0, $(TILE_N))
      offsets = pid_m * $(N) * $(K) + n_offsets[:, None] * $(K) + k_offsets
      mask = (n_offsets[:, None] < $(N)) & (k_offsets < $(K))
      inp = tl.load(input_ptr + offsets, mask=mask, other=-float("inf"))
      m_new = tl.maximum(m, inp)
      alpha = tl.exp(m - m_new)
      z = z * alpha + tl.exp(inp - m_new)
      m = m_new
    }
    m_reduced = tl.max(m, 0)
    z = tl.sum(z * tl.exp(m - m_reduced[None, :]), 0)
    m = m_reduced
    previous_multiple = tl.cdiv($(N), $(TILE_N)) * $(TILE_N) - $(TILE_N)
    for start_n in range($(0), $(N), $(TILE_N)) {
      n_offsets = previous_multiple - start_n + tl.arange(0, $(TILE_N))
      offsets = pid_m * $(N) * $(K) + n_offsets[:, None] * $(K) + k_offsets
      mask = (n_offsets[:, None] < $(N)) & (k_offsets[None, :] < $(K))
      inp = tl.load(input_ptr + offsets, mask=mask, other=-float("inf"))
      o = tl.exp(inp - m[None, :]) / z[None, :]
      tl.store(output_ptr + offsets, (o).to($outputType:tritonDType), mask=mask)
    }
  }
})

def softmax_kernel_non_inner_fp32 := flaggems0%[tl.float32, tl.float32, original]

syntax "flaggems1%" "[" tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(flaggems1%[$inputType:tritonDType, $outputType:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `address_commuted do
      Macro.throwError "expected original or address_commuted"
    let programBase : TSyntax `tritonExpr ←
      if version.getId == `address_commuted then
        `(tritonExpr| $(N) * pid_m)
      else `(tritonExpr| pid_m * $(N))
    `(fun (output_ptr input_ptr : RegionName)
        (M N TILE_N : Nat) (ONE_TILE_PER_CTA : Bool) => triton {
  tl.region output_ptr = $outputType, input_ptr = $inputType
  pid_m = tl.program_id(0)
  if ONE_TILE_PER_CTA {
    n_offsets = tl.arange(0, $(TILE_N))
    offset = pid_m * $(N) + n_offsets
    input_ptrs = input_ptr + offset
    mask = n_offsets < $(N)
    inp = tl.load(input_ptrs, mask=mask, other=-float("inf")).to($outputType:tritonDType)
    m = tl.max(inp, 0)
    e = tl.exp(inp - m)
    z = tl.sum(e, 0)
    out = e / z
    output_ptrs = output_ptr + offset
    tl.store(output_ptrs, (out).to($outputType:tritonDType), mask=mask)
  } else {
    m = tl.full([$(TILE_N)], value=float("-inf"), dtype=tl.float32)
    z = tl.full([$(TILE_N)], value=0.0, dtype=tl.float32)
    input_ptr += $programBase
    output_ptr += $programBase
    previous_multiple = tl.cdiv($(N), $(TILE_N)) * $(TILE_N) - $(TILE_N)
    for start_n in range($(0), previous_multiple, $(TILE_N)) {
      n_offsets = start_n + tl.arange(0, $(TILE_N))
      inp = tl.load(input_ptr + n_offsets)
      m_new = tl.maximum(m, inp)
      z = z * tl.exp(m - m_new) + tl.exp(inp - m_new)
      m = m_new
    }
    for start_n in range(previous_multiple, $(N), $(TILE_N)) {
      n_offsets = start_n + tl.arange(0, $(TILE_N))
      mask = n_offsets < $(N)
      inp = tl.load(input_ptr + n_offsets, mask=mask, other=-float("inf"))
      m_new = tl.maximum(m, inp)
      z = z * tl.exp(m - m_new) + tl.exp(inp - m_new)
      m = m_new
    }
    m_reduced = tl.max(m, 0)
    z = tl.sum(z * tl.exp(m - m_reduced), 0)
    m = m_reduced
    previous_multiple = tl.cdiv($(N), $(TILE_N)) * $(TILE_N) - $(TILE_N)
    for start_n in range($(0), $(TILE_N), $(TILE_N)) {
      n_offsets = previous_multiple - start_n + tl.arange(0, $(TILE_N))
      mask = n_offsets < $(N)
      inp = tl.load(input_ptr + n_offsets, mask=mask, other=-float("inf"), eviction_policy="evict_first")
      o = tl.exp(inp - m) / z
      tl.store(output_ptr + n_offsets, (o).to($outputType:tritonDType), mask=mask)
    }
    for start_n in range($(TILE_N), $(N), $(TILE_N)) {
      n_offsets = previous_multiple - start_n + tl.arange(0, $(TILE_N))
      inp = tl.load(input_ptr + n_offsets, eviction_policy="evict_first")
      o = tl.exp(inp - m) / z
      tl.store(output_ptr + n_offsets, (o).to($outputType:tritonDType))
    }
  }
})

def softmax_kernel_inner_fp32 := flaggems1%[tl.float32, tl.float32, original]

syntax "flaggems2%" "[" tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(flaggems2%[$inputType:tritonDType, $outputType:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `address_commuted do
      Macro.throwError "expected original or address_commuted"
    let programBase : TSyntax `tritonExpr ←
      if version.getId == `address_commuted then
        `(tritonExpr| $(TILE_K) * pid_k)
      else `(tritonExpr| pid_k * $(TILE_K))
    `(fun (out_ptr out_grad_ptr in_grad_ptr : RegionName)
        (M N K TILE_N TILE_K : Nat) (ONE_TILE_PER_CTA : Bool) => triton {
  tl.region out_ptr = $inputType, out_grad_ptr = $inputType, in_grad_ptr = $outputType
  pid_m = tl.program_id(0)
  pid_k = tl.program_id(1)
  offsets_k = $programBase + tl.arange(0, $(TILE_K))
  if ONE_TILE_PER_CTA {
    offsets_n = tl.arange(0, $(TILE_N))
    offsets = pid_m * $(N) * $(K) + offsets_n[:, None] * $(K) + offsets_k
    mask = (offsets_n < $(N))[:, None] & (offsets_k < $(K))
    out_tile = tl.load(out_ptr + offsets, mask=mask)
    out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask)
    scale = tl.sum(out_tile * out_grad_tile, axis=0)
    in_grad_tile = out_tile * (out_grad_tile - scale[None, :])
    tl.store(in_grad_ptr + offsets, (in_grad_tile).to($outputType:tritonDType), mask=mask)
  } else {
    offsets_n = tl.arange(0, $(TILE_N))
    offsets = pid_m * $(N) * $(K) + offsets_n[:, None] * $(K) + offsets_k
    scale = tl.zeros([$(TILE_N), $(TILE_K)], dtype=tl.float32)
    for unused_start_n in range($(0), $(N), $(TILE_N)) {
      mask = (offsets_n < $(N))[:, None] & (offsets_k < $(K))
      out_tile = tl.load(out_ptr + offsets, mask=mask)
      out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask)
      scale += out_tile * out_grad_tile
      offsets_n += $(TILE_N)
      offsets += $(TILE_N) * $(K)
    }
    scale = tl.sum(scale, axis=0)
    offsets_n = tl.arange(0, $(TILE_N))
    offsets = pid_m * $(N) * $(K) + offsets_n[:, None] * $(K) + offsets_k
    for unused_start_n in range($(0), $(N), $(TILE_N)) {
      mask = (offsets_n < $(N))[:, None] & (offsets_k < $(K))
      out_tile = tl.load(out_ptr + offsets, mask=mask)
      out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask)
      in_grad_tile = out_tile * (out_grad_tile - scale[None, :])
      tl.store(in_grad_ptr + offsets, (in_grad_tile).to($outputType:tritonDType), mask=mask)
      offsets_n += $(TILE_N)
      offsets += $(TILE_N) * $(K)
    }
  }
})

def softmax_backward_kernel_non_inner_fp32 := flaggems2%[tl.float32, tl.float32, original]

syntax "flaggems3%" "[" tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(flaggems3%[$inputType:tritonDType, $outputType:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `address_commuted do
      Macro.throwError "expected original or address_commuted"
    let programBase : TSyntax `tritonExpr ←
      if version.getId == `address_commuted then
        `(tritonExpr| $(TILE_M) * pid_m)
      else `(tritonExpr| pid_m * $(TILE_M))
    `(fun (out_ptr out_grad_ptr in_grad_ptr : RegionName)
        (M N TILE_M TILE_N : Nat) (ONE_TILE_PER_CTA : Bool) => triton {
  tl.region out_ptr = $inputType, out_grad_ptr = $inputType, in_grad_ptr = $outputType
  pid_m = tl.program_id(0)
  m_offsets = $programBase + tl.arange(0, $(TILE_M))
  if ONE_TILE_PER_CTA {
    n_offsets = tl.arange(0, $(TILE_N))
    offsets = m_offsets[:, None] * $(N) + n_offsets
    mask = (m_offsets[:, None] < $(M)) & (n_offsets < $(N))
    out_tile = tl.load(out_ptr + offsets, mask=mask)
    out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask)
    scale = tl.sum(out_tile * out_grad_tile, 1)
    in_grad_tile = out_tile * (out_grad_tile - scale[:, None])
    tl.store(in_grad_ptr + offsets, (in_grad_tile).to($outputType:tritonDType), mask=mask)
  } else {
    scale = tl.zeros([$(TILE_M), $(TILE_N)], dtype=tl.float32)
    n_offsets = tl.arange(0, $(TILE_N))
    offsets = m_offsets[:, None] * $(N) + n_offsets
    for unused_start_n in range($(0), $(N), $(TILE_N)) {
      mask = (m_offsets[:, None] < $(M)) & (n_offsets < $(N))
      out_tile = tl.load(out_ptr + offsets, mask=mask, eviction_policy="evict_last")
      out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask)
      scale += out_tile * out_grad_tile
      n_offsets += $(TILE_N)
      offsets += $(TILE_N)
    }
    scale = tl.sum(scale, 1)
    n_offsets = tl.arange(0, $(TILE_N))
    offsets = m_offsets[:, None] * $(N) + n_offsets
    for unused_start_n in range($(0), $(N), $(TILE_N)) {
      mask = (m_offsets[:, None] < $(M)) & (n_offsets < $(N))
      out_tile = tl.load(out_ptr + offsets, mask=mask, eviction_policy="evict_first")
      out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask)
      in_grad_tile = out_tile * (out_grad_tile - scale[:, None])
      tl.store(in_grad_ptr + offsets, (in_grad_tile).to($outputType:tritonDType), mask=mask)
      n_offsets += $(TILE_N)
      offsets += $(TILE_N)
    }
  }
})

def softmax_backward_kernel_inner_fp32 := flaggems3%[tl.float32, tl.float32, original]

inductive Storage where
  | fp16 | bf16 | fp32 | fp64
  deriving DecidableEq, Repr

def softmax_kernel_non_inner (inputType outputType : Storage) :=
  match inputType, outputType with
  | .fp16, .fp16 => flaggems0%[tl.float16, tl.float16, original]
  | .fp16, .bf16 => flaggems0%[tl.float16, tl.bfloat16, original]
  | .fp16, .fp32 => flaggems0%[tl.float16, tl.float32, original]
  | .fp16, .fp64 => flaggems0%[tl.float16, tl.float64, original]
  | .bf16, .fp16 => flaggems0%[tl.bfloat16, tl.float16, original]
  | .bf16, .bf16 => flaggems0%[tl.bfloat16, tl.bfloat16, original]
  | .bf16, .fp32 => flaggems0%[tl.bfloat16, tl.float32, original]
  | .bf16, .fp64 => flaggems0%[tl.bfloat16, tl.float64, original]
  | .fp32, .fp16 => flaggems0%[tl.float32, tl.float16, original]
  | .fp32, .bf16 => flaggems0%[tl.float32, tl.bfloat16, original]
  | .fp32, .fp32 => flaggems0%[tl.float32, tl.float32, original]
  | .fp32, .fp64 => flaggems0%[tl.float32, tl.float64, original]
  | .fp64, .fp16 => flaggems0%[tl.float64, tl.float16, original]
  | .fp64, .bf16 => flaggems0%[tl.float64, tl.bfloat16, original]
  | .fp64, .fp32 => flaggems0%[tl.float64, tl.float32, original]
  | .fp64, .fp64 => flaggems0%[tl.float64, tl.float64, original]

def softmax_kernel_non_inner_candidate (inputType outputType : Storage) :=
  match inputType, outputType with
  | .fp16, .fp16 => flaggems0%[tl.float16, tl.float16, address_commuted]
  | .fp16, .bf16 => flaggems0%[tl.float16, tl.bfloat16, address_commuted]
  | .fp16, .fp32 => flaggems0%[tl.float16, tl.float32, address_commuted]
  | .fp16, .fp64 => flaggems0%[tl.float16, tl.float64, address_commuted]
  | .bf16, .fp16 => flaggems0%[tl.bfloat16, tl.float16, address_commuted]
  | .bf16, .bf16 => flaggems0%[tl.bfloat16, tl.bfloat16, address_commuted]
  | .bf16, .fp32 => flaggems0%[tl.bfloat16, tl.float32, address_commuted]
  | .bf16, .fp64 => flaggems0%[tl.bfloat16, tl.float64, address_commuted]
  | .fp32, .fp16 => flaggems0%[tl.float32, tl.float16, address_commuted]
  | .fp32, .bf16 => flaggems0%[tl.float32, tl.bfloat16, address_commuted]
  | .fp32, .fp32 => flaggems0%[tl.float32, tl.float32, address_commuted]
  | .fp32, .fp64 => flaggems0%[tl.float32, tl.float64, address_commuted]
  | .fp64, .fp16 => flaggems0%[tl.float64, tl.float16, address_commuted]
  | .fp64, .bf16 => flaggems0%[tl.float64, tl.bfloat16, address_commuted]
  | .fp64, .fp32 => flaggems0%[tl.float64, tl.float32, address_commuted]
  | .fp64, .fp64 => flaggems0%[tl.float64, tl.float64, address_commuted]

def softmax_kernel_inner (inputType outputType : Storage) :=
  match inputType, outputType with
  | .fp16, .fp16 => flaggems1%[tl.float16, tl.float16, original]
  | .fp16, .bf16 => flaggems1%[tl.float16, tl.bfloat16, original]
  | .fp16, .fp32 => flaggems1%[tl.float16, tl.float32, original]
  | .fp16, .fp64 => flaggems1%[tl.float16, tl.float64, original]
  | .bf16, .fp16 => flaggems1%[tl.bfloat16, tl.float16, original]
  | .bf16, .bf16 => flaggems1%[tl.bfloat16, tl.bfloat16, original]
  | .bf16, .fp32 => flaggems1%[tl.bfloat16, tl.float32, original]
  | .bf16, .fp64 => flaggems1%[tl.bfloat16, tl.float64, original]
  | .fp32, .fp16 => flaggems1%[tl.float32, tl.float16, original]
  | .fp32, .bf16 => flaggems1%[tl.float32, tl.bfloat16, original]
  | .fp32, .fp32 => flaggems1%[tl.float32, tl.float32, original]
  | .fp32, .fp64 => flaggems1%[tl.float32, tl.float64, original]
  | .fp64, .fp16 => flaggems1%[tl.float64, tl.float16, original]
  | .fp64, .bf16 => flaggems1%[tl.float64, tl.bfloat16, original]
  | .fp64, .fp32 => flaggems1%[tl.float64, tl.float32, original]
  | .fp64, .fp64 => flaggems1%[tl.float64, tl.float64, original]

def softmax_kernel_inner_candidate (inputType outputType : Storage) :=
  match inputType, outputType with
  | .fp16, .fp16 => flaggems1%[tl.float16, tl.float16, address_commuted]
  | .fp16, .bf16 => flaggems1%[tl.float16, tl.bfloat16, address_commuted]
  | .fp16, .fp32 => flaggems1%[tl.float16, tl.float32, address_commuted]
  | .fp16, .fp64 => flaggems1%[tl.float16, tl.float64, address_commuted]
  | .bf16, .fp16 => flaggems1%[tl.bfloat16, tl.float16, address_commuted]
  | .bf16, .bf16 => flaggems1%[tl.bfloat16, tl.bfloat16, address_commuted]
  | .bf16, .fp32 => flaggems1%[tl.bfloat16, tl.float32, address_commuted]
  | .bf16, .fp64 => flaggems1%[tl.bfloat16, tl.float64, address_commuted]
  | .fp32, .fp16 => flaggems1%[tl.float32, tl.float16, address_commuted]
  | .fp32, .bf16 => flaggems1%[tl.float32, tl.bfloat16, address_commuted]
  | .fp32, .fp32 => flaggems1%[tl.float32, tl.float32, address_commuted]
  | .fp32, .fp64 => flaggems1%[tl.float32, tl.float64, address_commuted]
  | .fp64, .fp16 => flaggems1%[tl.float64, tl.float16, address_commuted]
  | .fp64, .bf16 => flaggems1%[tl.float64, tl.bfloat16, address_commuted]
  | .fp64, .fp32 => flaggems1%[tl.float64, tl.float32, address_commuted]
  | .fp64, .fp64 => flaggems1%[tl.float64, tl.float64, address_commuted]

def softmax_backward_kernel_non_inner (storage : Storage) :=
  match storage with
  | .fp16 => flaggems2%[tl.float16, tl.float16, original]
  | .bf16 => flaggems2%[tl.bfloat16, tl.bfloat16, original]
  | .fp32 => flaggems2%[tl.float32, tl.float32, original]
  | .fp64 => flaggems2%[tl.float64, tl.float64, original]

def softmax_backward_kernel_non_inner_candidate (storage : Storage) :=
  match storage with
  | .fp16 => flaggems2%[tl.float16, tl.float16, address_commuted]
  | .bf16 => flaggems2%[tl.bfloat16, tl.bfloat16, address_commuted]
  | .fp32 => flaggems2%[tl.float32, tl.float32, address_commuted]
  | .fp64 => flaggems2%[tl.float64, tl.float64, address_commuted]

def softmax_backward_kernel_inner (storage : Storage) :=
  match storage with
  | .fp16 => flaggems3%[tl.float16, tl.float16, original]
  | .bf16 => flaggems3%[tl.bfloat16, tl.bfloat16, original]
  | .fp32 => flaggems3%[tl.float32, tl.float32, original]
  | .fp64 => flaggems3%[tl.float64, tl.float64, original]

def softmax_backward_kernel_inner_candidate (storage : Storage) :=
  match storage with
  | .fp16 => flaggems3%[tl.float16, tl.float16, address_commuted]
  | .bf16 => flaggems3%[tl.bfloat16, tl.bfloat16, address_commuted]
  | .fp32 => flaggems3%[tl.float32, tl.float32, address_commuted]
  | .fp64 => flaggems3%[tl.float64, tl.float64, address_commuted]

end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.FullSource

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems
open VeriTile.Triton
set_option linter.unusedSimpArgs false
set_option linter.unusedVariables false

/-- Faithful transcription of `softmax_flaggems.py`'s
`softmax_kernel_non_inner`.

This keeps both the one-tile CTA path and the multi-tile online fallback for
the non-inner softmax dimension. -/
def softmax_kernel_non_inner_surface
    (output_ptr input_ptr : RegionName)
    (M N K TILE_N TILE_K : Nat)
    (ONE_TILE_PER_CTA : Bool) :
    ComputeKernel := triton {
  pid_k = tl.program_id(1)
  pid_m = tl.program_id(0)

  k_offsets = pid_k * $(TILE_K) + tl.arange(0, $(TILE_K))

  if ONE_TILE_PER_CTA {
    n_offsets = tl.arange(0, $(TILE_N))
    offset = pid_m * $(N) * $(K) + n_offsets[:, None] * $(K) + k_offsets
    mask = (n_offsets[:, None] < $(N)) & (k_offsets < $(K))
    input_ptrs = input_ptr + offset
    inp = tl.load(input_ptrs, mask=mask, other=-float("inf"))
    m = tl.max(inp, 0)
    e = tl.exp(inp - m[None, :])
    z = tl.sum(e, 0)
    out = e / z
    output_ptrs = output_ptr + offset
    tl.store(output_ptrs, out, mask=mask)
  } else {
    m = tl.full([$(TILE_N), $(TILE_K)], value=float("-inf"), dtype=tl.float32)
    z = tl.full([$(TILE_N), $(TILE_K)], value=0.0, dtype=tl.float32)

    for start_n in range($(0), $(N), $(TILE_N)) {
      n_offsets = start_n + tl.arange(0, $(TILE_N))
      offsets = pid_m * $(N) * $(K) + n_offsets[:, None] * $(K) + k_offsets
      mask = (n_offsets[:, None] < $(N)) & (k_offsets < $(K))
      inp = tl.load(input_ptr + offsets, mask=mask, other=-float("inf"))
      m_new = tl.maximum(m, inp)
      alpha = tl.exp(m - m_new)
      z = z * alpha + tl.exp(inp - m_new)
      m = m_new
    }

    m_reduced = tl.max(m, 0)
    z = tl.sum(z * tl.exp(m - m_reduced[None, :]), 0)
    m = m_reduced

    previous_multiple = $((N + TILE_N - 1)) // $(TILE_N) * $(TILE_N) - $(TILE_N)
    for start_n in range($(0), $(N), $(TILE_N)) {
      n_offsets = (previous_multiple - start_n) + tl.arange(0, $(TILE_N))
      offsets = pid_m * $(N) * $(K) + n_offsets[:, None] * $(K) + k_offsets
      mask = (n_offsets[:, None] < $(N)) & (k_offsets[None, :] < $(K))
      inp = tl.load(input_ptr + offsets, mask=mask, other=-float("inf"))
      o = tl.exp(inp - m[None, :]) / z[None, :]
      tl.store(output_ptr + offsets, o, mask=mask)
    }
  }
}

/-- Proof-oriented specialization of `softmax_kernel_non_inner` to
`ONE_TILE_PER_CTA=true`.

This covers the `K > 1` forward path used when the softmax dimension is not the
innermost physical dimension: each CTA handles one `(m, k-block)`, loads the
`[TILE_N, TILE_K]` tile, reduces along `N`, and stores the normalized tile.
The multi-tile fallback's `tl.float32` running `m/z` state is outside this
specialized surface. -/
def softmax_kernel_non_inner_one_tile_surface
    (output_ptr input_ptr : RegionName)
    (N K TILE_N TILE_K : Nat) :
    ComputeKernel := triton {
  pid_k = tl.program_id(1)
  pid_m = tl.program_id(0)
  k_offsets = pid_k * $(TILE_K) + tl.arange(0, $(TILE_K))
  n_offsets = tl.arange(0, $(TILE_N))
  offset = pid_m * $(N) * $(K) + n_offsets[:, None] * $(K) + k_offsets
  mask = (n_offsets[:, None] < $(N)) & (k_offsets < $(K))
  input_ptrs = input_ptr + offset
  inp = tl.load(input_ptrs, mask=mask, other=-float("inf"))
  m = tl.max(inp, 0)
  e = tl.exp(inp - m[None, :])
  z = tl.sum(e, 0)
  out = e / z
  output_ptrs = output_ptr + offset
  tl.store(output_ptrs, out, mask=mask)
}

/-- Proof-oriented `ONE_TILE_PER_CTA=true` slice of
`softmax_flaggems.py`'s `softmax_kernel_inner`.

This covers the inner-dimension fast path where one CTA covers a full row. It
preserves the source kernel's row program id, masked load with `-inf`, stable
softmax max/sum normalization, output-dtype load cast, and masked store. The
multi-tile online fallback remains future work. -/
def softmax_kernel_inner_one_tile
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

/-- Surface transcription of `softmax_flaggems.py`'s
`softmax_backward_kernel_non_inner`, specialized to `ONE_TILE_PER_CTA=true`.

This preserves the non-inner `(m, k-block)` addressing and the standard softmax
VJP formula `out * (out_grad - sum(out * out_grad, axis=0))`. -/
def softmax_backward_kernel_non_inner_one_tile_surface
    (out_ptr out_grad_ptr in_grad_ptr : RegionName)
    (N K TILE_N TILE_K : Nat) :
    ComputeKernel := triton {
  pid_m = tl.program_id(0)
  pid_k = tl.program_id(1)
  offsets_k = pid_k * $(TILE_K) + tl.arange(0, $(TILE_K))
  offsets_n = tl.arange(0, $(TILE_N))
  offsets = pid_m * $(N) * $(K) + offsets_n[:, None] * $(K) + offsets_k
  mask = (offsets_n < $(N))[:, None] & (offsets_k < $(K))
  out_tile = tl.load(out_ptr + offsets, mask=mask)
  out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask)
  scale = tl.sum(out_tile * out_grad_tile, axis=0)
  in_grad_tile = out_tile * (out_grad_tile - scale[None, :])
  tl.store(in_grad_ptr + offsets, in_grad_tile, mask=mask)
}

/-- Surface transcription of `softmax_flaggems.py`'s
`softmax_backward_kernel_inner`, specialized to `ONE_TILE_PER_CTA=true`.

This is the contiguous-row backward path: a CTA handles `[TILE_M, TILE_N]`,
reduces along `N`, and writes the masked input-gradient tile. -/
def softmax_backward_kernel_inner_one_tile_surface
    (out_ptr out_grad_ptr in_grad_ptr : RegionName)
    (M N TILE_M TILE_N : Nat) :
    ComputeKernel := triton {
  pid_m = tl.program_id(0)
  m_offsets = pid_m * $(TILE_M) + tl.arange(0, $(TILE_M))
  n_offsets = tl.arange(0, $(TILE_N))
  offsets = m_offsets[:, None] * $(N) + n_offsets
  mask = (m_offsets[:, None] < $(M)) & (n_offsets < $(N))
  out_tile = tl.load(out_ptr + offsets, mask=mask)
  out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask)
  scale = tl.sum(out_tile * out_grad_tile, 1)
  in_grad_tile = out_tile * (out_grad_tile - scale[:, None])
  tl.store(in_grad_ptr + offsets, in_grad_tile, mask=mask)
}



end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems
