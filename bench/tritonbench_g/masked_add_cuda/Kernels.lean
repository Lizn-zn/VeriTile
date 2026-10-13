import VeriTile.Triton

namespace VeriTile.Bench.TritonBenchG.MaskedAddCuda
open VeriTile.Triton

/-- Faithful transcription of `masked_add_cuda.py`'s `masked_add_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` -> Lean `Nat` parameter. -/
def masked_add_kernel
    (grad_ptr p_ptr p_mask_ptr : RegionName)
    (n_elements : Nat) (alpha : ℝ) (BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  p_mask = tl.load(p_mask_ptr + offsets, mask=mask).to(tl.int1)
  mask = mask & ~p_mask
  p = tl.load(p_ptr + offsets, mask=mask)
  grad = tl.load(grad_ptr + offsets, mask=mask)
  grad += p * $(alpha)
  tl.store(grad_ptr + offsets, grad, mask=mask)
}


/-- Commuted addition for the preserved boolean-buffer specialization. -/
def masked_add_commuted_kernel
    (grad_ptr p_ptr p_mask_ptr : RegionName)
    (n_elements : Nat) (alpha : ℝ) (BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  p_mask = tl.load(p_mask_ptr + offsets, mask=mask).to(tl.int1)
  mask = mask & ~p_mask
  p = tl.load(p_ptr + offsets, mask=mask)
  grad = tl.load(grad_ptr + offsets, mask=mask)
  grad = p * $(alpha) + grad
  tl.store(grad_ptr + offsets, grad, mask=mask)
}


/-- Tested Python integer mask specializations. Both use signed integer memory. -/
inductive MaskStorage where
  | int32 | int64
  deriving DecidableEq, Repr

open Lean
set_option hygiene false in
macro "maskedAddSource%[" gradType:tritonDType "," pType:tritonDType "," maskType:tritonDType "," candidate:ident "]" : term => do
  let update ← if candidate.getId == `original then
    `(tritonStmt| grad += p * $((alpha : ℝ)).to(tl.float32))
  else if candidate.getId == `commuted then
    `(tritonStmt| grad = p * $((alpha : ℝ)).to(tl.float32) + grad)
  else Macro.throwError "expected original or commuted"
  let store ← match gradType with
    | `(tritonDType| tl.float32) | `(tritonDType| tl.float64) =>
        `(tritonStmt| tl.store(grad_ptr + offsets, grad, mask=mask))
    | _ => `(tritonStmt| tl.store(grad_ptr + offsets, grad.to($gradType:tritonDType), mask=mask))
  `(fun (grad_ptr p_ptr p_mask_ptr : RegionName) (n_elements : Nat) (alpha : ℝ)
      (BLOCK_SIZE : Nat) => triton {
    tl.region grad_ptr = $gradType, p_ptr = $pType, p_mask_ptr = $maskType
    pid = tl.program_id(axis=0)
    block_start = pid * $(BLOCK_SIZE)
    offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
    mask = offsets < $(n_elements)
    p_mask = tl.load(p_mask_ptr + offsets, mask=mask).to(tl.int1)
    mask = mask & ~p_mask
    p = tl.load(p_ptr + offsets, mask=mask)
    grad = tl.load(grad_ptr + offsets, mask=mask)
    $update:tritonStmt
    $store:tritonStmt
  })

/-- Complete fp32 data source, with signed int32 or int64 mask storage. -/
def integer_kernel (storage : MaskStorage) (candidate : Bool)
    (grad_ptr p_ptr p_mask_ptr : RegionName)
    (n_elements : Nat) (alpha : ℝ) (BLOCK_SIZE : Nat) : ComputeKernel :=
  match storage, candidate with
  | .int32, Bool.false =>
      maskedAddSource%[tl.float32, tl.float32, tl.int32, original]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .int64, Bool.false =>
      maskedAddSource%[tl.float32, tl.float32, tl.int64, original]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .int32, Bool.true =>
      maskedAddSource%[tl.float32, tl.float32, tl.int32, commuted]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .int64, Bool.true =>
      maskedAddSource%[tl.float32, tl.float32, tl.int64, commuted]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE

/-- Float64-region transcription for real-only verification. Current DSL load
profile erasure and mixed runtime-scalar precision prevent a faithful FP claim. -/
def integer_fp64_kernel (storage : MaskStorage) (candidate : Bool)
    (grad_ptr p_ptr p_mask_ptr : RegionName)
    (n_elements : Nat) (alpha : ℝ) (BLOCK_SIZE : Nat) : ComputeKernel :=
  match storage, candidate with
  | .int32, Bool.false =>
      maskedAddSource%[tl.float64, tl.float64, tl.int32, original]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .int64, Bool.false =>
      maskedAddSource%[tl.float64, tl.float64, tl.int64, original]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .int32, Bool.true =>
      maskedAddSource%[tl.float64, tl.float64, tl.int32, commuted]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .int64, Bool.true =>
      maskedAddSource%[tl.float64, tl.float64, tl.int64, commuted]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE

/-- Floating storage profiles whose runtime fp32 alpha promotes arithmetic to fp32. -/
inductive FloatStorage where
  | fp32 | fp16 | bf16
  deriving DecidableEq, Repr

/-- Full independent gradient/data storage combinations. The final cast spells
out Triton's implicit conversion back to the gradient pointer element type. -/
def typed_integer_kernel (gradType pType : FloatStorage) (maskType : MaskStorage)
    (candidate : Bool) (grad_ptr p_ptr p_mask_ptr : RegionName)
    (n_elements : Nat) (alpha : ℝ) (BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradType, pType, maskType, candidate with
  | .fp32, .fp32, .int32, Bool.false =>
      maskedAddSource%[tl.float32, tl.float32, tl.int32, original]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .fp32, .fp32, .int32, Bool.true =>
      maskedAddSource%[tl.float32, tl.float32, tl.int32, commuted]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .fp32, .fp32, .int64, Bool.false =>
      maskedAddSource%[tl.float32, tl.float32, tl.int64, original]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .fp32, .fp32, .int64, Bool.true =>
      maskedAddSource%[tl.float32, tl.float32, tl.int64, commuted]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .fp32, .fp16, .int32, Bool.false =>
      maskedAddSource%[tl.float32, tl.float16, tl.int32, original]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .fp32, .fp16, .int32, Bool.true =>
      maskedAddSource%[tl.float32, tl.float16, tl.int32, commuted]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .fp32, .fp16, .int64, Bool.false =>
      maskedAddSource%[tl.float32, tl.float16, tl.int64, original]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .fp32, .fp16, .int64, Bool.true =>
      maskedAddSource%[tl.float32, tl.float16, tl.int64, commuted]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .fp32, .bf16, .int32, Bool.false =>
      maskedAddSource%[tl.float32, tl.bfloat16, tl.int32, original]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .fp32, .bf16, .int32, Bool.true =>
      maskedAddSource%[tl.float32, tl.bfloat16, tl.int32, commuted]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .fp32, .bf16, .int64, Bool.false =>
      maskedAddSource%[tl.float32, tl.bfloat16, tl.int64, original]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .fp32, .bf16, .int64, Bool.true =>
      maskedAddSource%[tl.float32, tl.bfloat16, tl.int64, commuted]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .fp16, .fp32, .int32, Bool.false =>
      maskedAddSource%[tl.float16, tl.float32, tl.int32, original]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .fp16, .fp32, .int32, Bool.true =>
      maskedAddSource%[tl.float16, tl.float32, tl.int32, commuted]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .fp16, .fp32, .int64, Bool.false =>
      maskedAddSource%[tl.float16, tl.float32, tl.int64, original]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .fp16, .fp32, .int64, Bool.true =>
      maskedAddSource%[tl.float16, tl.float32, tl.int64, commuted]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .fp16, .fp16, .int32, Bool.false =>
      maskedAddSource%[tl.float16, tl.float16, tl.int32, original]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .fp16, .fp16, .int32, Bool.true =>
      maskedAddSource%[tl.float16, tl.float16, tl.int32, commuted]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .fp16, .fp16, .int64, Bool.false =>
      maskedAddSource%[tl.float16, tl.float16, tl.int64, original]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .fp16, .fp16, .int64, Bool.true =>
      maskedAddSource%[tl.float16, tl.float16, tl.int64, commuted]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .fp16, .bf16, .int32, Bool.false =>
      maskedAddSource%[tl.float16, tl.bfloat16, tl.int32, original]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .fp16, .bf16, .int32, Bool.true =>
      maskedAddSource%[tl.float16, tl.bfloat16, tl.int32, commuted]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .fp16, .bf16, .int64, Bool.false =>
      maskedAddSource%[tl.float16, tl.bfloat16, tl.int64, original]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .fp16, .bf16, .int64, Bool.true =>
      maskedAddSource%[tl.float16, tl.bfloat16, tl.int64, commuted]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .bf16, .fp32, .int32, Bool.false =>
      maskedAddSource%[tl.bfloat16, tl.float32, tl.int32, original]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .bf16, .fp32, .int32, Bool.true =>
      maskedAddSource%[tl.bfloat16, tl.float32, tl.int32, commuted]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .bf16, .fp32, .int64, Bool.false =>
      maskedAddSource%[tl.bfloat16, tl.float32, tl.int64, original]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .bf16, .fp32, .int64, Bool.true =>
      maskedAddSource%[tl.bfloat16, tl.float32, tl.int64, commuted]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .bf16, .fp16, .int32, Bool.false =>
      maskedAddSource%[tl.bfloat16, tl.float16, tl.int32, original]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .bf16, .fp16, .int32, Bool.true =>
      maskedAddSource%[tl.bfloat16, tl.float16, tl.int32, commuted]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .bf16, .fp16, .int64, Bool.false =>
      maskedAddSource%[tl.bfloat16, tl.float16, tl.int64, original]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .bf16, .fp16, .int64, Bool.true =>
      maskedAddSource%[tl.bfloat16, tl.float16, tl.int64, commuted]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .bf16, .bf16, .int32, Bool.false =>
      maskedAddSource%[tl.bfloat16, tl.bfloat16, tl.int32, original]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .bf16, .bf16, .int32, Bool.true =>
      maskedAddSource%[tl.bfloat16, tl.bfloat16, tl.int32, commuted]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .bf16, .bf16, .int64, Bool.false =>
      maskedAddSource%[tl.bfloat16, tl.bfloat16, tl.int64, original]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .bf16, .bf16, .int64, Bool.true =>
      maskedAddSource%[tl.bfloat16, tl.bfloat16, tl.int64, commuted]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE

end VeriTile.Bench.TritonBenchG.MaskedAddCuda
