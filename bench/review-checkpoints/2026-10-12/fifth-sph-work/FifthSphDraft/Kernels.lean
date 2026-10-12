import VeriTile.Triton.DSL
import VeriTile.Triton.Source
/-! Source: fifth_order_sph_harmonics.py, fifth_order_fwd.
Both fifth_order_fwd and fifth_order_bwd are transcribed completely.
Named decimal coefficient assignments become fp32 scalar tensors, following
ordinary Triton assignment semantics. They promote polynomial arithmetic to
fp32; powers formed before the coefficient multiplication retain input type.
Negative decimal literals are staged as single constants; negating a named
tensor coefficient remains an arithmetic operation.
The candidate only commutes the natural-number coordinate block product.
All implicit destination conversions are explicit. Missing load padding stays
unspecified; it is not translated as zero. Backward adds to the initial dX.
-/
namespace VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.TypedSource
open Lean VeriTile.Triton
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000
inductive Storage where
  | fp16 | bf16 | fp32
  deriving DecidableEq

syntax "fifthFwd%" "[" tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(fifthFwd%[$inputTy:tritonDType,$outTy:tritonDType,$version:ident]) => do
    unless version.getId == `original || version.getId == `indexCommuted do
      Macro.throwError "expected original or indexCommuted"
    let coordinateOffset : TSyntax `tritonStmt ←
      if version.getId == `indexCommuted then
        `(tritonStmt| coord_row_offset = coord_striding + block_id * ($(block_size) * coord_stride))
      else
        `(tritonStmt| coord_row_offset = coord_striding + ($(block_size) * coord_stride) * block_id)
    `(fun (coord_ptr output_ptr : RegionName) (block_size coord_numel output_numel col_offset output_stride : Nat) =>
      ComputeKernel.mk [coord_ptr,output_ptr] [output_ptr] (triton {
  tl.region coord_ptr = $inputTy, output_ptr = $outTy
  coord_stride = $(3)
  block_id = tl.program_id($(0))
  coord_striding = (tl.arange($(0), $(block_size)) * coord_stride)
  $coordinateOffset
  x = tl.load(coord_ptr + coord_row_offset, mask=coord_row_offset < $(coord_numel))
  y = tl.load(coord_ptr + coord_row_offset + $(1), mask=(coord_row_offset + $(1)) < $(coord_numel))
  z = tl.load(coord_ptr + coord_row_offset + $(2), mask=(coord_row_offset + $(2)) < $(coord_numel))
  CONST000 = 1.73430461568895
  CONST001 = 2.32681380862329
  CONST002 = 1.60565407233314
  CONST003 = 3.21130814466628
  CONST004 = 3.31662479035540
  CONST005 = 6.21867148191637
  CONST006 = 6.21867148191637
  CONST007 = 1.60565407233314
  CONST009 = 11.6340690431164
  CONST010 = 12.8452325786651
  CONST011 = 12.4373429638327
  CONST012 = 12.8452325786651
  CONST013 = 13.8744369255116
  CONST017 = 33.9852909359329
  CONST018 = 7.35803132638072
  CONST020 = $((-44.1481879582843 : ℝ))
  CONST021 = $((-41.6233107765348 : ℝ))
  CONST022 = $((-29.4321253055229 : ℝ))
  CONST023 = $((-23.2681380862329 : ℝ))
  CONST024 = $((-19.2678488679977 : ℝ))
  CONST025 = $((-19.2678488679977 : ℝ))
  CONST026 = $((-16.9926454679664 : ℝ))
  CONST027 = $((-16.9926454679664 : ℝ))
  CONST028 = $((-13.8744369255116 : ℝ))
  CONST029 = $((-16.5831239517770 : ℝ))
  CONST030 = 3.46860923137790
  CONST031 = $((-8.49632273398321 : ℝ))
  CONST032 = $((-5.20291384706685 : ℝ))
  CONST033 = $((-3.46860923137790 : ℝ))
  CONST034 = $((-1.73430461568895 : ℝ))
  VAR05 = ((((x * x) * x) * x) * x)
  VAR06 = (((x * x) * x) * x)
  VAR07 = ((x * x) * x)
  VAR08 = (x * x)
  VAR14 = ((((y * y) * y) * y) * y)
  VAR15 = (((y * y) * y) * y)
  VAR16 = ((y * y) * y)
  VAR17 = (y * y)
  VAR23 = ((((z * z) * z) * z) * z)
  VAR24 = (((z * z) * z) * z)
  VAR25 = ((z * z) * z)
  VAR26 = (z * z)
  Y00 = (((CONST001 * VAR05) + ((CONST009 * VAR24) * x)) + ((CONST023 * VAR07) * VAR26))
  Y01 = (y * (((CONST022 * VAR07) * z) - ((CONST022 * VAR25) * x)))
  Y02 = (((CONST000 * VAR05) + (VAR07 * ((CONST028 * VAR17) + (CONST033 * VAR26)))) + (x * ((((-CONST021) * VAR17) * VAR26) + (CONST032 * VAR24))))
  Y03 = ((((CONST027 * VAR07) * y) * z) + (x * (((CONST017 * VAR16) * z) + ((CONST026 * VAR25) * y))))
  Y04 = (((CONST002 * VAR05) + (VAR07 * ((CONST003 * VAR26) + (CONST025 * VAR17)))) + (x * (((CONST002 * VAR24) + (CONST010 * VAR15)) + ((CONST024 * VAR17) * VAR26))))
  Y05 = (((CONST004 * VAR14) + (VAR16 * ((CONST029 * VAR08) + (CONST029 * VAR26)))) + (y * (((CONST005 * VAR06) + (CONST006 * VAR24)) + ((CONST011 * VAR08) * VAR26))))
  Y06 = (((CONST002 * VAR23) + (VAR25 * ((CONST003 * VAR08) + (CONST024 * VAR17)))) + (z * (((CONST007 * VAR06) + (CONST012 * VAR15)) + ((CONST024 * VAR08) * VAR17))))
  Y07 = ((VAR16 * ((CONST026 * VAR08) - (CONST026 * VAR26))) + (y * (((-CONST031) * VAR06) + (CONST031 * VAR24))))
  Y08 = (((CONST034 * VAR23) + (VAR25 * ((CONST013 * VAR17) + (CONST030 * VAR08)))) + (z * (((CONST021 * VAR08) * VAR17) - (CONST032 * VAR06))))
  Y09 = (y * (((CONST018 * VAR06) + (CONST018 * VAR24)) + ((CONST020 * VAR08) * VAR26)))
  Y10 = (((CONST001 * VAR23) + ((CONST009 * VAR06) * z)) + ((CONST023 * VAR08) * VAR25))
  output_striding = (tl.arange($(0), $(block_size)) * $(output_stride))
  output_row_offset = ((output_striding + (($(block_size) * $(output_stride)) * block_id)) + $(col_offset))
  tl.store(output_ptr + output_row_offset, (Y00).to($outTy:tritonDType), mask=output_row_offset < $(output_numel))
  tl.store(output_ptr + output_row_offset + $(1), (Y01).to($outTy:tritonDType), mask=(output_row_offset + $(1)) < $(output_numel))
  tl.store(output_ptr + output_row_offset + $(2), (Y02).to($outTy:tritonDType), mask=(output_row_offset + $(2)) < $(output_numel))
  tl.store(output_ptr + output_row_offset + $(3), (Y03).to($outTy:tritonDType), mask=(output_row_offset + $(3)) < $(output_numel))
  tl.store(output_ptr + output_row_offset + $(4), (Y04).to($outTy:tritonDType), mask=(output_row_offset + $(4)) < $(output_numel))
  tl.store(output_ptr + output_row_offset + $(5), (Y05).to($outTy:tritonDType), mask=(output_row_offset + $(5)) < $(output_numel))
  tl.store(output_ptr + output_row_offset + $(6), (Y06).to($outTy:tritonDType), mask=(output_row_offset + $(6)) < $(output_numel))
  tl.store(output_ptr + output_row_offset + $(7), (Y07).to($outTy:tritonDType), mask=(output_row_offset + $(7)) < $(output_numel))
  tl.store(output_ptr + output_row_offset + $(8), (Y08).to($outTy:tritonDType), mask=(output_row_offset + $(8)) < $(output_numel))
  tl.store(output_ptr + output_row_offset + $(9), (Y09).to($outTy:tritonDType), mask=(output_row_offset + $(9)) < $(output_numel))
  tl.store(output_ptr + output_row_offset + $(10), (Y10).to($outTy:tritonDType), mask=(output_row_offset + $(10)) < $(output_numel))
}).surfaceBody)

abbrev forward_fp16_fp16 (swapped : Bool) := if swapped then fifthFwd%[tl.float16,tl.float16,indexCommuted] else fifthFwd%[tl.float16,tl.float16,original]
abbrev forward_fp16_bf16 (swapped : Bool) := if swapped then fifthFwd%[tl.float16,tl.bfloat16,indexCommuted] else fifthFwd%[tl.float16,tl.bfloat16,original]
abbrev forward_fp16_fp32 (swapped : Bool) := if swapped then fifthFwd%[tl.float16,tl.float32,indexCommuted] else fifthFwd%[tl.float16,tl.float32,original]
abbrev forward_bf16_fp16 (swapped : Bool) := if swapped then fifthFwd%[tl.bfloat16,tl.float16,indexCommuted] else fifthFwd%[tl.bfloat16,tl.float16,original]
abbrev forward_bf16_bf16 (swapped : Bool) := if swapped then fifthFwd%[tl.bfloat16,tl.bfloat16,indexCommuted] else fifthFwd%[tl.bfloat16,tl.bfloat16,original]
abbrev forward_bf16_fp32 (swapped : Bool) := if swapped then fifthFwd%[tl.bfloat16,tl.float32,indexCommuted] else fifthFwd%[tl.bfloat16,tl.float32,original]
abbrev forward_fp32_fp16 (swapped : Bool) := if swapped then fifthFwd%[tl.float32,tl.float16,indexCommuted] else fifthFwd%[tl.float32,tl.float16,original]
abbrev forward_fp32_bf16 (swapped : Bool) := if swapped then fifthFwd%[tl.float32,tl.bfloat16,indexCommuted] else fifthFwd%[tl.float32,tl.bfloat16,original]
abbrev forward_fp32_fp32 (swapped : Bool) := if swapped then fifthFwd%[tl.float32,tl.float32,indexCommuted] else fifthFwd%[tl.float32,tl.float32,original]

def forward (input output : Storage) (swapped : Bool) := match input,output with
  | .fp16,.fp16 => forward_fp16_fp16 swapped
  | .fp16,.bf16 => forward_fp16_bf16 swapped
  | .fp16,.fp32 => forward_fp16_fp32 swapped
  | .bf16,.fp16 => forward_bf16_fp16 swapped
  | .bf16,.bf16 => forward_bf16_bf16 swapped
  | .bf16,.fp32 => forward_bf16_fp32 swapped
  | .fp32,.fp16 => forward_fp32_fp16 swapped
  | .fp32,.bf16 => forward_fp32_bf16 swapped
  | .fp32,.fp32 => forward_fp32_fp32 swapped

syntax "fifthBwd%" "[" tritonDType "," tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(fifthBwd%[$inputTy:tritonDType,$gradTy:tritonDType,$upstreamTy:tritonDType,$version:ident]) => do
    unless version.getId == `original || version.getId == `indexCommuted do
      Macro.throwError "expected original or indexCommuted"
    let coordinateOffset : TSyntax `tritonStmt ←
      if version.getId == `indexCommuted then
        `(tritonStmt| coord_row_offset = coord_striding + block_id * ($(block_size) * coord_stride))
      else
        `(tritonStmt| coord_row_offset = coord_striding + ($(block_size) * coord_stride) * block_id)
    `(fun (coord_ptr coord_grad_ptr sph_grad_ptr : RegionName) (block_size coord_numel output_numel col_offset output_stride : Nat) =>
      ComputeKernel.mk [coord_ptr,coord_grad_ptr,sph_grad_ptr] [coord_grad_ptr] (triton {
  tl.region coord_ptr = $inputTy, coord_grad_ptr = $gradTy, sph_grad_ptr = $upstreamTy
  block_id = tl.program_id($(0))
  coord_stride = $(3)
  coord_striding = (tl.arange($(0), $(block_size)) * coord_stride)
  $coordinateOffset
  x = tl.load(coord_ptr + coord_row_offset, mask=coord_row_offset < $(coord_numel))
  y = tl.load(coord_ptr + coord_row_offset + $(1), mask=(coord_row_offset + $(1)) < $(coord_numel))
  z = tl.load(coord_ptr + coord_row_offset + $(2), mask=(coord_row_offset + $(2)) < $(coord_numel))
  output_striding = (tl.arange($(0), $(block_size)) * $(output_stride))
  output_row_offset = ((output_striding + (($(block_size) * $(output_stride)) * block_id)) + $(col_offset))
  g_0 = tl.load(sph_grad_ptr + output_row_offset, mask=output_row_offset < $(output_numel))
  g_1 = tl.load(sph_grad_ptr + output_row_offset + $(1), mask=(output_row_offset + $(1)) < $(output_numel))
  g_2 = tl.load(sph_grad_ptr + output_row_offset + $(2), mask=(output_row_offset + $(2)) < $(output_numel))
  g_3 = tl.load(sph_grad_ptr + output_row_offset + $(3), mask=(output_row_offset + $(3)) < $(output_numel))
  g_4 = tl.load(sph_grad_ptr + output_row_offset + $(4), mask=(output_row_offset + $(4)) < $(output_numel))
  g_5 = tl.load(sph_grad_ptr + output_row_offset + $(5), mask=(output_row_offset + $(5)) < $(output_numel))
  g_6 = tl.load(sph_grad_ptr + output_row_offset + $(6), mask=(output_row_offset + $(6)) < $(output_numel))
  g_7 = tl.load(sph_grad_ptr + output_row_offset + $(7), mask=(output_row_offset + $(7)) < $(output_numel))
  g_8 = tl.load(sph_grad_ptr + output_row_offset + $(8), mask=(output_row_offset + $(8)) < $(output_numel))
  g_9 = tl.load(sph_grad_ptr + output_row_offset + $(9), mask=(output_row_offset + $(9)) < $(output_numel))
  g_10 = tl.load(sph_grad_ptr + output_row_offset + $(10), mask=(output_row_offset + $(10)) < $(output_numel))
  CONST000 = 1.60565407233314
  CONST001 = 3.00000000000000
  CONST002 = 3.21130814466628
  CONST003 = 1.60565407233314
  CONST004 = 6.42261628933256
  CONST005 = 6.42261628933256
  CONST006 = 8.67152307844476
  CONST007 = 8.02827036166571
  CONST008 = 6.93721846275580
  CONST009 = 11.6340690431164
  CONST010 = 12.8452325786651
  CONST011 = 6.21867148191637
  CONST012 = 6.21867148191637
  CONST014 = 12.4373429638327
  CONST017 = 12.8452325786651
  CONST018 = 13.8744369255116
  CONST019 = 24.8746859276655
  CONST020 = 24.8746859276655
  CONST021 = 27.7488738510232
  CONST024 = 29.4321253055229
  CONST027 = 7.35803132638072
  CONST029 = 46.5362761724657
  CONST030 = 51.3809303146605
  CONST031 = 51.3809303146605
  CONST034 = 101.955872807799
  CONST036 = $((-8.67152307844475 : ℝ))
  CONST037 = 3.46860923137790
  CONST038 = $((-88.2963759165686 : ℝ))
  CONST039 = $((-83.2466215530696 : ℝ))
  CONST040 = $((-69.8044142586986 : ℝ))
  CONST041 = $((-50.9779364038993 : ℝ))
  CONST042 = $((-50.9779364038993 : ℝ))
  CONST043 = $((-46.5362761724657 : ℝ))
  CONST044 = $((-44.1481879582843 : ℝ))
  CONST045 = $((-41.6233107765348 : ℝ))
  CONST046 = $((-38.5356977359954 : ℝ))
  CONST047 = $((-38.5356977359954 : ℝ))
  CONST048 = $((-33.1662479035540 : ℝ))
  CONST049 = $((-33.9852909359329 : ℝ))
  CONST050 = 6.42261628933257
  CONST051 = $((-33.9852909359329 : ℝ))
  CONST052 = $((-29.4321253055229 : ℝ))
  CONST053 = $((-27.7488738510232 : ℝ))
  CONST054 = $((-20.8116553882674 : ℝ))
  CONST055 = $((-19.2678488679977 : ℝ))
  CONST056 = $((-19.2678488679977 : ℝ))
  CONST057 = $((-16.9926454679664 : ℝ))
  CONST058 = $((-16.9926454679664 : ℝ))
  CONST059 = $((-13.8744369255116 : ℝ))
  CONST060 = $((-16.5831239517770 : ℝ))
  CONST061 = $((-8.49632273398321 : ℝ))
  CONST062 = $((-6.93721846275580 : ℝ))
  CONST063 = $((-5.20291384706685 : ℝ))
  CONST064 = $((-3.46860923137790 : ℝ))
  VAR06 = (((x * x) * x) * x)
  VAR07 = ((x * x) * x)
  VAR08 = (x * x)
  VAR15 = (((y * y) * y) * y)
  VAR16 = ((y * y) * y)
  VAR17 = (y * y)
  VAR24 = (((z * z) * z) * z)
  VAR25 = ((z * z) * z)
  VAR26 = (z * z)
  g_x = tl.load(coord_grad_ptr + coord_row_offset, mask=coord_row_offset < $(coord_numel))
  g_y = tl.load(coord_grad_ptr + coord_row_offset + $(1), mask=(coord_row_offset + $(1)) < $(coord_numel))
  g_z = tl.load(coord_grad_ptr + coord_row_offset + $(2), mask=(coord_row_offset + $(2)) < $(coord_numel))
  g_x += (((((((((((g_0 * (((CONST009 * VAR06) + (CONST009 * VAR24)) + ((CONST040 * VAR08) * VAR26))) + ((g_1 * y) * (((CONST038 * VAR08) * z) - (CONST052 * VAR25)))) + (g_10 * (((CONST029 * VAR07) * z) + ((CONST043 * VAR25) * x)))) + (g_2 * (((((CONST001 * VAR08) * ((CONST059 * VAR17) + (CONST064 * VAR26))) + (CONST006 * VAR06)) - ((CONST045 * VAR17) * VAR26)) + (CONST063 * VAR24)))) + (g_3 * (((((CONST041 * VAR08) * y) * z) - ((CONST049 * VAR16) * z)) + ((CONST057 * VAR25) * y)))) + (g_4 * (((((CONST000 * VAR24) + ((CONST001 * VAR08) * ((CONST002 * VAR26) + (CONST055 * VAR17)))) + (CONST007 * VAR06)) + (CONST010 * VAR15)) + ((CONST056 * VAR17) * VAR26)))) + (g_5 * (((CONST048 * VAR16) * x) + (y * ((CONST019 * VAR07) + ((CONST019 * VAR26) * x)))))) + (g_6 * (((CONST005 * VAR25) * x) + (z * ((CONST004 * VAR07) + ((CONST046 * VAR17) * x)))))) + (g_7 * (((CONST049 * VAR16) * x) - ((CONST051 * VAR07) * y)))) + (g_8 * (((CONST008 * VAR25) * x) + (z * (((CONST039 * VAR17) * x) - (CONST054 * VAR07)))))) + ((g_9 * y) * ((CONST024 * VAR07) + ((CONST038 * VAR26) * x))))
  g_y += (((((((((g_1 * (((CONST052 * VAR07) * z) - ((CONST052 * VAR25) * x))) + (g_2 * (((((-CONST039) * VAR26) * x) * y) + ((CONST053 * VAR07) * y)))) + (g_3 * (((CONST058 * VAR07) * z) + (x * (((CONST034 * VAR17) * z) + (CONST057 * VAR25)))))) + (g_4 * (((CONST047 * VAR07) * y) + (x * ((CONST030 * VAR16) + ((CONST046 * VAR26) * y)))))) + (g_5 * ((((((CONST001 * VAR17) * ((CONST060 * VAR08) + (CONST060 * VAR26))) + (CONST011 * VAR06)) + (CONST012 * VAR24)) + ((CONST014 * VAR08) * VAR26)) - (CONST060 * VAR15)))) + (g_6 * (((CONST046 * VAR25) * y) + (z * ((CONST031 * VAR16) + ((CONST046 * VAR08) * y)))))) + (g_7 * ((((CONST001 * VAR17) * ((CONST057 * VAR08) - (CONST057 * VAR26))) - (CONST061 * VAR06)) + (CONST061 * VAR24)))) + (g_8 * (((CONST021 * VAR25) * y) + (((CONST039 * VAR08) * y) * z)))) + (g_9 * (((CONST027 * VAR06) + (CONST027 * VAR24)) + ((CONST044 * VAR08) * VAR26))))
  g_z += (((((((((((g_0 * (((CONST029 * VAR25) * x) + ((CONST043 * VAR07) * z))) + ((g_1 * y) * ((((-CONST038) * VAR26) * x) + (CONST052 * VAR07)))) + (g_10 * (((CONST009 * VAR06) + (CONST009 * VAR24)) + ((CONST040 * VAR08) * VAR26)))) + (g_2 * (((CONST062 * VAR07) * z) + (x * ((((-CONST039) * VAR17) * z) + (CONST054 * VAR25)))))) + (g_3 * (((CONST058 * VAR07) * y) + (x * (((CONST042 * VAR26) * y) - (CONST049 * VAR16)))))) + (g_4 * (((CONST005 * VAR07) * z) + (x * (((CONST046 * VAR17) * z) + (CONST050 * VAR25)))))) + (g_5 * (((CONST048 * VAR16) * z) + (y * (((CONST019 * VAR08) * z) + (CONST020 * VAR25)))))) + (g_6 * ((((((CONST001 * VAR26) * ((CONST002 * VAR08) + (CONST056 * VAR17))) + (CONST003 * VAR06)) + (CONST007 * VAR24)) + (CONST017 * VAR15)) + ((CONST056 * VAR08) * VAR17)))) + (g_7 * ((((-CONST049) * VAR16) * z) + ((CONST051 * VAR25) * y)))) + (g_8 * (((((CONST001 * VAR26) * ((CONST018 * VAR17) + (CONST037 * VAR08))) + (CONST036 * VAR24)) + ((CONST045 * VAR08) * VAR17)) - (CONST063 * VAR06)))) + ((g_9 * y) * ((CONST024 * VAR25) + ((CONST038 * VAR08) * z))))
  tl.store(coord_grad_ptr + coord_row_offset, (g_x).to($gradTy:tritonDType), mask=coord_row_offset < $(coord_numel))
  tl.store(coord_grad_ptr + coord_row_offset + $(1), (g_y).to($gradTy:tritonDType), mask=(coord_row_offset + $(1)) < $(coord_numel))
  tl.store(coord_grad_ptr + coord_row_offset + $(2), (g_z).to($gradTy:tritonDType), mask=(coord_row_offset + $(2)) < $(coord_numel))
}).surfaceBody)

abbrev backward_fp16_fp16_fp16 (swapped : Bool) := if swapped then fifthBwd%[tl.float16,tl.float16,tl.float16,indexCommuted] else fifthBwd%[tl.float16,tl.float16,tl.float16,original]
abbrev backward_fp16_fp16_bf16 (swapped : Bool) := if swapped then fifthBwd%[tl.float16,tl.float16,tl.bfloat16,indexCommuted] else fifthBwd%[tl.float16,tl.float16,tl.bfloat16,original]
abbrev backward_fp16_fp16_fp32 (swapped : Bool) := if swapped then fifthBwd%[tl.float16,tl.float16,tl.float32,indexCommuted] else fifthBwd%[tl.float16,tl.float16,tl.float32,original]
abbrev backward_fp16_bf16_fp16 (swapped : Bool) := if swapped then fifthBwd%[tl.float16,tl.bfloat16,tl.float16,indexCommuted] else fifthBwd%[tl.float16,tl.bfloat16,tl.float16,original]
abbrev backward_fp16_bf16_bf16 (swapped : Bool) := if swapped then fifthBwd%[tl.float16,tl.bfloat16,tl.bfloat16,indexCommuted] else fifthBwd%[tl.float16,tl.bfloat16,tl.bfloat16,original]
abbrev backward_fp16_bf16_fp32 (swapped : Bool) := if swapped then fifthBwd%[tl.float16,tl.bfloat16,tl.float32,indexCommuted] else fifthBwd%[tl.float16,tl.bfloat16,tl.float32,original]
abbrev backward_fp16_fp32_fp16 (swapped : Bool) := if swapped then fifthBwd%[tl.float16,tl.float32,tl.float16,indexCommuted] else fifthBwd%[tl.float16,tl.float32,tl.float16,original]
abbrev backward_fp16_fp32_bf16 (swapped : Bool) := if swapped then fifthBwd%[tl.float16,tl.float32,tl.bfloat16,indexCommuted] else fifthBwd%[tl.float16,tl.float32,tl.bfloat16,original]
abbrev backward_fp16_fp32_fp32 (swapped : Bool) := if swapped then fifthBwd%[tl.float16,tl.float32,tl.float32,indexCommuted] else fifthBwd%[tl.float16,tl.float32,tl.float32,original]
abbrev backward_bf16_fp16_fp16 (swapped : Bool) := if swapped then fifthBwd%[tl.bfloat16,tl.float16,tl.float16,indexCommuted] else fifthBwd%[tl.bfloat16,tl.float16,tl.float16,original]
abbrev backward_bf16_fp16_bf16 (swapped : Bool) := if swapped then fifthBwd%[tl.bfloat16,tl.float16,tl.bfloat16,indexCommuted] else fifthBwd%[tl.bfloat16,tl.float16,tl.bfloat16,original]
abbrev backward_bf16_fp16_fp32 (swapped : Bool) := if swapped then fifthBwd%[tl.bfloat16,tl.float16,tl.float32,indexCommuted] else fifthBwd%[tl.bfloat16,tl.float16,tl.float32,original]
abbrev backward_bf16_bf16_fp16 (swapped : Bool) := if swapped then fifthBwd%[tl.bfloat16,tl.bfloat16,tl.float16,indexCommuted] else fifthBwd%[tl.bfloat16,tl.bfloat16,tl.float16,original]
abbrev backward_bf16_bf16_bf16 (swapped : Bool) := if swapped then fifthBwd%[tl.bfloat16,tl.bfloat16,tl.bfloat16,indexCommuted] else fifthBwd%[tl.bfloat16,tl.bfloat16,tl.bfloat16,original]
abbrev backward_bf16_bf16_fp32 (swapped : Bool) := if swapped then fifthBwd%[tl.bfloat16,tl.bfloat16,tl.float32,indexCommuted] else fifthBwd%[tl.bfloat16,tl.bfloat16,tl.float32,original]
abbrev backward_bf16_fp32_fp16 (swapped : Bool) := if swapped then fifthBwd%[tl.bfloat16,tl.float32,tl.float16,indexCommuted] else fifthBwd%[tl.bfloat16,tl.float32,tl.float16,original]
abbrev backward_bf16_fp32_bf16 (swapped : Bool) := if swapped then fifthBwd%[tl.bfloat16,tl.float32,tl.bfloat16,indexCommuted] else fifthBwd%[tl.bfloat16,tl.float32,tl.bfloat16,original]
abbrev backward_bf16_fp32_fp32 (swapped : Bool) := if swapped then fifthBwd%[tl.bfloat16,tl.float32,tl.float32,indexCommuted] else fifthBwd%[tl.bfloat16,tl.float32,tl.float32,original]
abbrev backward_fp32_fp16_fp16 (swapped : Bool) := if swapped then fifthBwd%[tl.float32,tl.float16,tl.float16,indexCommuted] else fifthBwd%[tl.float32,tl.float16,tl.float16,original]
abbrev backward_fp32_fp16_bf16 (swapped : Bool) := if swapped then fifthBwd%[tl.float32,tl.float16,tl.bfloat16,indexCommuted] else fifthBwd%[tl.float32,tl.float16,tl.bfloat16,original]
abbrev backward_fp32_fp16_fp32 (swapped : Bool) := if swapped then fifthBwd%[tl.float32,tl.float16,tl.float32,indexCommuted] else fifthBwd%[tl.float32,tl.float16,tl.float32,original]
abbrev backward_fp32_bf16_fp16 (swapped : Bool) := if swapped then fifthBwd%[tl.float32,tl.bfloat16,tl.float16,indexCommuted] else fifthBwd%[tl.float32,tl.bfloat16,tl.float16,original]
abbrev backward_fp32_bf16_bf16 (swapped : Bool) := if swapped then fifthBwd%[tl.float32,tl.bfloat16,tl.bfloat16,indexCommuted] else fifthBwd%[tl.float32,tl.bfloat16,tl.bfloat16,original]
abbrev backward_fp32_bf16_fp32 (swapped : Bool) := if swapped then fifthBwd%[tl.float32,tl.bfloat16,tl.float32,indexCommuted] else fifthBwd%[tl.float32,tl.bfloat16,tl.float32,original]
abbrev backward_fp32_fp32_fp16 (swapped : Bool) := if swapped then fifthBwd%[tl.float32,tl.float32,tl.float16,indexCommuted] else fifthBwd%[tl.float32,tl.float32,tl.float16,original]
abbrev backward_fp32_fp32_bf16 (swapped : Bool) := if swapped then fifthBwd%[tl.float32,tl.float32,tl.bfloat16,indexCommuted] else fifthBwd%[tl.float32,tl.float32,tl.bfloat16,original]
abbrev backward_fp32_fp32_fp32 (swapped : Bool) := if swapped then fifthBwd%[tl.float32,tl.float32,tl.float32,indexCommuted] else fifthBwd%[tl.float32,tl.float32,tl.float32,original]

def backward (input gradient upstream : Storage) (swapped : Bool) := match input,gradient,upstream with
  | .fp16,.fp16,.fp16 => backward_fp16_fp16_fp16 swapped
  | .fp16,.fp16,.bf16 => backward_fp16_fp16_bf16 swapped
  | .fp16,.fp16,.fp32 => backward_fp16_fp16_fp32 swapped
  | .fp16,.bf16,.fp16 => backward_fp16_bf16_fp16 swapped
  | .fp16,.bf16,.bf16 => backward_fp16_bf16_bf16 swapped
  | .fp16,.bf16,.fp32 => backward_fp16_bf16_fp32 swapped
  | .fp16,.fp32,.fp16 => backward_fp16_fp32_fp16 swapped
  | .fp16,.fp32,.bf16 => backward_fp16_fp32_bf16 swapped
  | .fp16,.fp32,.fp32 => backward_fp16_fp32_fp32 swapped
  | .bf16,.fp16,.fp16 => backward_bf16_fp16_fp16 swapped
  | .bf16,.fp16,.bf16 => backward_bf16_fp16_bf16 swapped
  | .bf16,.fp16,.fp32 => backward_bf16_fp16_fp32 swapped
  | .bf16,.bf16,.fp16 => backward_bf16_bf16_fp16 swapped
  | .bf16,.bf16,.bf16 => backward_bf16_bf16_bf16 swapped
  | .bf16,.bf16,.fp32 => backward_bf16_bf16_fp32 swapped
  | .bf16,.fp32,.fp16 => backward_bf16_fp32_fp16 swapped
  | .bf16,.fp32,.bf16 => backward_bf16_fp32_bf16 swapped
  | .bf16,.fp32,.fp32 => backward_bf16_fp32_fp32 swapped
  | .fp32,.fp16,.fp16 => backward_fp32_fp16_fp16 swapped
  | .fp32,.fp16,.bf16 => backward_fp32_fp16_bf16 swapped
  | .fp32,.fp16,.fp32 => backward_fp32_fp16_fp32 swapped
  | .fp32,.bf16,.fp16 => backward_fp32_bf16_fp16 swapped
  | .fp32,.bf16,.bf16 => backward_fp32_bf16_bf16 swapped
  | .fp32,.bf16,.fp32 => backward_fp32_bf16_fp32 swapped
  | .fp32,.fp32,.fp16 => backward_fp32_fp32_fp16 swapped
  | .fp32,.fp32,.bf16 => backward_fp32_fp32_bf16 swapped
  | .fp32,.fp32,.fp32 => backward_fp32_fp32_fp32 swapped

end
end VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.TypedSource
