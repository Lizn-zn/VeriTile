import VeriTile.Triton.DSL
import VeriTile.Triton.Source
import VeriTile.Meta.StatementAudit
/-! Complete source of `bgmv_expand_slice.py`'s `_bgmv_expand_slice_kernel`.
The source template below is specialized at the input and weight dtypes and
CAST_TYPE, which are compile-time choices in Triton. Every accepted input dtype
(fp16/bf16/fp32), both weight dtypes, both cast choices, both load-mask choices
and both ADD_INPUTS branches remain representable. Output dtype is explicit.

The two update expressions are the sole candidate change. Both expanded
kernels keep the sentinel, N/SPLIT_N calculation, dynamic output loop, exact
source pointer arithmetic, masks, reduction and output conversion. The host
chooses CAST_TYPE for fp32 input, while the dispatcher also covers the other
compile-time choices. The template stages that conditional before register
typing, so its cast cannot be lost at a branch-type merge. The output conversion that Triton inserts for the store is written at each
branch's accumulator assignment. Only that register write separates the sum
from the store; there is no intervening arithmetic. Staging the same conversion
in both arms avoids giving the no-addition arm the addition arm's promoted dtype.
The sum itself retains its product dtype, and only the ADD_INPUTS arithmetic
uses promotion against the loaded output.

Correct.lean proves the independent full-rank real formula with flat-memory
IO; FPEquiv.lean proves the complete typed-program equivalence. The legacy
real helper remains in Proofs/RealFoundation solely for reusable lemmas.

Translation-surface blocker: the macro specializes storage dtypes and stages
CAST_TYPE before register typing; the store makes Triton’s implicit output
conversion explicit. The original early return is a non-sentinel active branch.
The source correspondence and complete configuration regression check these
choices; README.md records the mapping and scopes.
-/
open Lean
open VeriTile.Triton
namespace VeriTile.Bench.TritonBenchG.BgmvExpandSlice.TypedSource
set_option maxHeartbeats 2400000
syntax "bgmvConcreteSource%" "[" tritonDType "," tritonDType "," tritonDType "," ident "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(bgmvConcreteSource%[$inTy:tritonDType, $weightTy:tritonDType, $outTy:tritonDType, $castFlag:ident, $swapFlag:ident]) => do
    unless castFlag.getId == `cast || castFlag.getId == `keep do
      Macro.throwError "expected cast or keep"
    unless swapFlag.getId == `original || swapFlag.getId == `swapped do
      Macro.throwError "expected original or swapped"
    let casts : Array (TSyntax `tritonStmt) ←
      if castFlag.getId == `cast then pure #[← `(tritonStmt| tiled_a = (tiled_a).to($weightTy:tritonDType))]
      else pure #[]
    let update : TSyntax `tritonExpr ←
      if swapFlag.getId == `swapped then `(tritonExpr| tiled_out + tl.sum(tiled_a * tiled_b, 1))
      else `(tritonExpr| tl.sum(tiled_a * tiled_b, 1) + tiled_out)
    `(fun (input_ptr lora_ptr out_ptr : RegionName)
        (N K : Nat) (lora_indices : Region .int)
        (xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride
          cn_stride slice_offset BLOCK_N BLOCK_K SPLIT_N : Nat)
        (EVEN_K ADD_INPUTS : Bool) =>
  ComputeKernel.mk [input_ptr, lora_ptr, out_ptr, Region.cast lora_indices] [out_ptr]
  (triton {
  tl.region input_ptr = $inTy, lora_ptr = $weightTy, out_ptr = $outTy
  pid_sn = tl.program_id(axis=0)
  cur_batch = tl.program_id(axis=1)
  lora_index = tl.load(lora_indices + cur_batch)
  if lora_index != $((-1 : Int)) {
    offset_k = tl.arange(0, $(BLOCK_K))
    offset_n = tl.arange(0, $(BLOCK_N))
    if EVEN_K {
      tiled_a = tl.load(input_ptr + cur_batch * $(xm_stride) + offset_k * $(xk_stride))
    } else {
      tiled_a = tl.load(input_ptr + cur_batch * $(xm_stride) + offset_k * $(xk_stride),
        mask=offset_k < $(K), other=0)
    }
    split_n_length = tl.cdiv($(N), $(SPLIT_N))
    $casts:tritonStmt*
    b_ptr = lora_ptr + $(l0_stride) * lora_index +
      pid_sn * split_n_length * $(lora_k_stride)
    c_ptr = out_ptr + cur_batch * $(cm_stride) + pid_sn * split_n_length +
      $(slice_offset) * $(cn_stride)
    for n in range($(0), split_n_length, $(BLOCK_N)) {
      current_n = n + offset_n
      b_ptr_mask = (current_n[:, None] < split_n_length) & (offset_k[None, :] < $(K))
      c_mask = current_n < split_n_length
      tiled_b = tl.load(
        b_ptr + current_n[:, None] * $(lora_k_stride) +
          offset_k[None, :] * $(lora_n_stride),
        mask=b_ptr_mask, other=0.0)
      if ADD_INPUTS {
        tiled_out = tl.load(c_ptr + current_n * $(cn_stride), mask=c_mask)
        accumulator = ($update).to($outTy:tritonDType)
      } else {
        accumulator = (tl.sum(tiled_a * tiled_b, 1)).to($outTy:tritonDType)
      }
      tl.store(c_ptr + current_n * $(cn_stride), accumulator, mask=c_mask)
    }
  }
}).surfaceBody)

/-- Specialize output storage before arithmetic promotion. A half reduction
plus a half output must stay half; it cannot use an implicit fp32 sum. -/
syntax "bgmvSource%" "[" tritonDType "," tritonDType "," ident "," ident "]" : term
macro_rules
  | `(bgmvSource%[$inTy:tritonDType, $weightTy:tritonDType, $castFlag:ident, $swapFlag:ident]) =>
    `(fun (out_dtype : FloatDType) => match out_dtype with
      | .real => bgmvConcreteSource%[$inTy, $weightTy, tl.float64, $castFlag, $swapFlag]
      | .fp32 => bgmvConcreteSource%[$inTy, $weightTy, tl.float32, $castFlag, $swapFlag]
      | .fp16 => bgmvConcreteSource%[$inTy, $weightTy, tl.float16, $castFlag, $swapFlag]
      | .bf16 => bgmvConcreteSource%[$inTy, $weightTy, tl.bfloat16, $castFlag, $swapFlag]
      | .f8e4 => bgmvConcreteSource%[$inTy, $weightTy, tl.float8e4nv, $castFlag, $swapFlag]
      | .f8e5 => bgmvConcreteSource%[$inTy, $weightTy, tl.float8e5, $castFlag, $swapFlag])

/-- Input storage types accepted by the host wrapper. -/
inductive InputStorage where
  | fp16 | bf16 | fp32
  deriving DecidableEq

/-- Weight storage types accepted by the host wrapper. -/
inductive WeightStorage where
  | fp16 | bf16
  deriving DecidableEq

abbrev KernelSource := FloatDType → RegionName → RegionName → RegionName → Nat → Nat →
  Region .int → Nat → Nat → Nat → Nat → Nat → Nat → Nat → Nat → Nat → Nat → Nat →
  Bool → Bool → ComputeKernel

/-- Specialize only compile-time precision/cast choices. All shape, stride,
program-ID, sentinel, masking and accumulation behavior remains in the source. -/
def specialize (input : InputStorage) (weight : WeightStorage)
    (CAST_TYPE swapped : Bool) : KernelSource :=
  match input, weight, CAST_TYPE, swapped with
  | .fp16, .fp16, Bool.false, Bool.false => bgmvSource%[tl.float16, tl.float16, keep, original]
  | .fp16, .fp16, Bool.false, Bool.true => bgmvSource%[tl.float16, tl.float16, keep, swapped]
  | .fp16, .fp16, Bool.true, Bool.false => bgmvSource%[tl.float16, tl.float16, cast, original]
  | .fp16, .fp16, Bool.true, Bool.true => bgmvSource%[tl.float16, tl.float16, cast, swapped]
  | .fp16, .bf16, Bool.false, Bool.false => bgmvSource%[tl.float16, tl.bfloat16, keep, original]
  | .fp16, .bf16, Bool.false, Bool.true => bgmvSource%[tl.float16, tl.bfloat16, keep, swapped]
  | .fp16, .bf16, Bool.true, Bool.false => bgmvSource%[tl.float16, tl.bfloat16, cast, original]
  | .fp16, .bf16, Bool.true, Bool.true => bgmvSource%[tl.float16, tl.bfloat16, cast, swapped]
  | .bf16, .fp16, Bool.false, Bool.false => bgmvSource%[tl.bfloat16, tl.float16, keep, original]
  | .bf16, .fp16, Bool.false, Bool.true => bgmvSource%[tl.bfloat16, tl.float16, keep, swapped]
  | .bf16, .fp16, Bool.true, Bool.false => bgmvSource%[tl.bfloat16, tl.float16, cast, original]
  | .bf16, .fp16, Bool.true, Bool.true => bgmvSource%[tl.bfloat16, tl.float16, cast, swapped]
  | .bf16, .bf16, Bool.false, Bool.false => bgmvSource%[tl.bfloat16, tl.bfloat16, keep, original]
  | .bf16, .bf16, Bool.false, Bool.true => bgmvSource%[tl.bfloat16, tl.bfloat16, keep, swapped]
  | .bf16, .bf16, Bool.true, Bool.false => bgmvSource%[tl.bfloat16, tl.bfloat16, cast, original]
  | .bf16, .bf16, Bool.true, Bool.true => bgmvSource%[tl.bfloat16, tl.bfloat16, cast, swapped]
  | .fp32, .fp16, Bool.false, Bool.false => bgmvSource%[tl.float32, tl.float16, keep, original]
  | .fp32, .fp16, Bool.false, Bool.true => bgmvSource%[tl.float32, tl.float16, keep, swapped]
  | .fp32, .fp16, Bool.true, Bool.false => bgmvSource%[tl.float32, tl.float16, cast, original]
  | .fp32, .fp16, Bool.true, Bool.true => bgmvSource%[tl.float32, tl.float16, cast, swapped]
  | .fp32, .bf16, Bool.false, Bool.false => bgmvSource%[tl.float32, tl.bfloat16, keep, original]
  | .fp32, .bf16, Bool.false, Bool.true => bgmvSource%[tl.float32, tl.bfloat16, keep, swapped]
  | .fp32, .bf16, Bool.true, Bool.false => bgmvSource%[tl.float32, tl.bfloat16, cast, original]
  | .fp32, .bf16, Bool.true, Bool.true => bgmvSource%[tl.float32, tl.bfloat16, cast, swapped]

def original (input : InputStorage) (weight : WeightStorage) (CAST_TYPE : Bool) : KernelSource :=
  specialize input weight CAST_TYPE Bool.false

def outputFirst (input : InputStorage) (weight : WeightStorage) (CAST_TYPE : Bool) : KernelSource :=
  specialize input weight CAST_TYPE Bool.true

/-- The host's CAST_TYPE selection; the sources also support explicit choices. -/
def hostCast : InputStorage → Bool
  | .fp32 => Bool.true
  | _ => Bool.false

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.BgmvExpandSlice.TypedSource
