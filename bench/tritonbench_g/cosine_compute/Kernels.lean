import VeriTile.Triton.DSL

/-! Source: cosine_compute.py, cos_func.
Both complete sources retain the masked
load with unspecified padding, explicit fp32 conversion, intrinsic tl.cos and
destination conversion. The candidate only commutes the integer product in
the block offset; it does not change the numerical expression. -/
namespace VeriTile.Bench.TritonBenchG.CosineCompute
open Lean VeriTile.Triton

inductive Storage where
  | fp16 | bf16 | fp32
  deriving DecidableEq

syntax "cosineSource%" "[" tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(cosineSource%[$storage:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `indexCommuted do
      Macro.throwError "expected original or indexCommuted"
    if version.getId == `original then
      `(fun (a b : RegionName) (n_elements BLOCK_SIZE : Nat) => ComputeKernel.mk [a] [b]
      (triton {
  tl.region a = $storage, b = $storage
  offset = tl.program_id(0) * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
  mask = offset < $(n_elements)
  a_value = tl.load(a + offset, mask=mask)
  b_value = tl.cos((a_value).to(tl.float32))
  tl.store(b + offset, (b_value).to($storage:tritonDType), mask=mask)
}).surfaceBody)
    else
      `(fun (a b : RegionName) (n_elements BLOCK_SIZE : Nat) => ComputeKernel.mk [a] [b]
      (triton {
  tl.region a = $storage, b = $storage
  offset = $(BLOCK_SIZE) * tl.program_id(0) + tl.arange(0, $(BLOCK_SIZE))
  mask = offset < $(n_elements)
  a_value = tl.load(a + offset, mask=mask)
  b_value = tl.cos((a_value).to(tl.float32))
  tl.store(b + offset, (b_value).to($storage:tritonDType), mask=mask)
}).surfaceBody)

def original (storage : Storage) := match storage with
  | .fp16 => cosineSource%[tl.float16,original]
  | .bf16 => cosineSource%[tl.bfloat16,original]
  | .fp32 => cosineSource%[tl.float32,original]

def indexCommuted (storage : Storage) := match storage with
  | .fp16 => cosineSource%[tl.float16,indexCommuted]
  | .bf16 => cosineSource%[tl.bfloat16,indexCommuted]
  | .fp32 => cosineSource%[tl.float32,indexCommuted]

def kernel (storage : Storage) (swapped : Bool) (a b : RegionName) (n B : Nat) :=
  (if swapped then indexCommuted storage else original storage) a b n B
end VeriTile.Bench.TritonBenchG.CosineCompute
