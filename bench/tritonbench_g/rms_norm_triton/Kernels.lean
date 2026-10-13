import VeriTile.Triton.DSL
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.RmsNormTriton
open VeriTile.Triton
/-- Faithful transcription of `rms_norm_triton.py`'s `rms_norm_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` -> Lean `Nat` parameter. -/
def rms_norm_kernel
    (Y X W : RegionName)
    (y_stride_r y_stride_c x_stride_r x_stride_c N : Nat)
    (eps : ℝ) (BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  Y += pid * $(y_stride_r)
  X += pid * $(x_stride_r)
  mask = tl.arange(0, $(BLOCK_SIZE)) < $(N)
  cols = tl.arange(0, $(BLOCK_SIZE))
  x = tl.load(X + cols * $(x_stride_c), mask, other=0.0).to(tl.float32)
  var = tl.sum(x * x, axis=0) / $(N)
  rrms = 1 / tl.sqrt(var + $(eps))
  w = tl.load(W + tl.arange(0, $(BLOCK_SIZE)), mask=mask, other=0.0)
  y = (x * rrms).to(Y.dtype.element_ty) * w
  tl.store(Y + cols * $(y_stride_c), y, mask=mask)
}


/-! The complete floating storage family retains the cast before multiplication
by W. Runtime eps is fp32; the final cast models the implicit store conversion
only when multiplication promoted the value beyond Y's storage type. -/
open Lean
syntax "rmsTyped%" "[" tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(rmsTyped%[$dtype:tritonDType, $weight:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `commuted do
      Macro.throwError "expected original or commuted"
    let ybase : TSyntax `tritonStmt ←
      if version.getId == `original then `(tritonStmt| Y += pid * $(y_stride_r))
      else `(tritonStmt| Y += $(y_stride_r) * pid)
    let xbase : TSyntax `tritonStmt ←
      if version.getId == `original then `(tritonStmt| X += pid * $(x_stride_r))
      else `(tritonStmt| X += $(x_stride_r) * pid)
    let narrow := match dtype, weight with
      | `(tritonDType| tl.float64), _ => Bool.false
      | `(tritonDType| tl.float32), `(tritonDType| tl.float64) => Bool.true
      | `(tritonDType| tl.float32), _ => Bool.false
      | `(tritonDType| tl.float16), `(tritonDType| tl.float16) => Bool.false
      | `(tritonDType| tl.float16), `(tritonDType| tl.bfloat16) => Bool.false
      | `(tritonDType| tl.bfloat16), `(tritonDType| tl.bfloat16) => Bool.false
      | _, _ => Bool.true
    let write : TSyntax `tritonStmt ← if narrow then
      `(tritonStmt| tl.store(Y + cols * $(y_stride_c), y.to($dtype:tritonDType), mask=mask))
      else `(tritonStmt| tl.store(Y + cols * $(y_stride_c), y, mask=mask))
    `(fun (Y X W : RegionName) (y_stride_r y_stride_c x_stride_r x_stride_c N : Nat)
        (eps : ℝ) (BLOCK_SIZE : Nat) =>
      ComputeKernel.mk [X, W] [Y] (triton {
  tl.region Y = $dtype, X = $dtype, W = $weight
  pid = tl.program_id(0)
  $ybase
  $xbase
  mask = tl.arange(0, $(BLOCK_SIZE)) < $(N)
  cols = tl.arange(0, $(BLOCK_SIZE))
  x = tl.load(X + cols * $(x_stride_c), mask, other=0.0).to(tl.float32)
  var = tl.sum(x * x, axis=0) / $(N)
  rrms = 1 / tl.sqrt(var + $((eps : ℝ)).to(tl.float32))
  w = tl.load(W + tl.arange(0, $(BLOCK_SIZE)), mask=mask, other=0.0)
  y = (x * rrms).to($dtype:tritonDType) * w
  $write
}).surfaceBody)

inductive Storage where
  | fp32 | fp16 | bf16 | fp64
  deriving DecidableEq, Repr

def typed (storage weight : Storage) (commuted : Bool)
    (Y X W : RegionName) (y_stride_r y_stride_c x_stride_r x_stride_c N : Nat)
    (eps : ℝ) (BLOCK_SIZE : Nat) : ComputeKernel :=
  match storage, weight, commuted with
  | .fp32, .fp32, Bool.false =>
    rmsTyped%[tl.float32, tl.float32, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp32, .fp32, Bool.true =>
    rmsTyped%[tl.float32, tl.float32, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp32, .fp16, Bool.false =>
    rmsTyped%[tl.float32, tl.float16, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp32, .fp16, Bool.true =>
    rmsTyped%[tl.float32, tl.float16, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp32, .bf16, Bool.false =>
    rmsTyped%[tl.float32, tl.bfloat16, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp32, .bf16, Bool.true =>
    rmsTyped%[tl.float32, tl.bfloat16, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp32, .fp64, Bool.false =>
    rmsTyped%[tl.float32, tl.float64, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp32, .fp64, Bool.true =>
    rmsTyped%[tl.float32, tl.float64, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp16, .fp32, Bool.false =>
    rmsTyped%[tl.float16, tl.float32, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp16, .fp32, Bool.true =>
    rmsTyped%[tl.float16, tl.float32, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp16, .fp16, Bool.false =>
    rmsTyped%[tl.float16, tl.float16, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp16, .fp16, Bool.true =>
    rmsTyped%[tl.float16, tl.float16, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp16, .bf16, Bool.false =>
    rmsTyped%[tl.float16, tl.bfloat16, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp16, .bf16, Bool.true =>
    rmsTyped%[tl.float16, tl.bfloat16, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp16, .fp64, Bool.false =>
    rmsTyped%[tl.float16, tl.float64, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp16, .fp64, Bool.true =>
    rmsTyped%[tl.float16, tl.float64, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .bf16, .fp32, Bool.false =>
    rmsTyped%[tl.bfloat16, tl.float32, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .bf16, .fp32, Bool.true =>
    rmsTyped%[tl.bfloat16, tl.float32, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .bf16, .fp16, Bool.false =>
    rmsTyped%[tl.bfloat16, tl.float16, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .bf16, .fp16, Bool.true =>
    rmsTyped%[tl.bfloat16, tl.float16, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .bf16, .bf16, Bool.false =>
    rmsTyped%[tl.bfloat16, tl.bfloat16, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .bf16, .bf16, Bool.true =>
    rmsTyped%[tl.bfloat16, tl.bfloat16, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .bf16, .fp64, Bool.false =>
    rmsTyped%[tl.bfloat16, tl.float64, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .bf16, .fp64, Bool.true =>
    rmsTyped%[tl.bfloat16, tl.float64, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp64, .fp32, Bool.false =>
    rmsTyped%[tl.float64, tl.float32, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp64, .fp32, Bool.true =>
    rmsTyped%[tl.float64, tl.float32, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp64, .fp16, Bool.false =>
    rmsTyped%[tl.float64, tl.float16, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp64, .fp16, Bool.true =>
    rmsTyped%[tl.float64, tl.float16, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp64, .bf16, Bool.false =>
    rmsTyped%[tl.float64, tl.bfloat16, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp64, .bf16, Bool.true =>
    rmsTyped%[tl.float64, tl.bfloat16, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp64, .fp64, Bool.false =>
    rmsTyped%[tl.float64, tl.float64, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp64, .fp64, Bool.true =>
    rmsTyped%[tl.float64, tl.float64, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsNormTriton
