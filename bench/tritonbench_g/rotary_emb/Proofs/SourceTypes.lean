import VeriTile.Triton

/-! Complete 3D source. The unused sine strides and the Q-before-K store order
are intentional source facts. Storage composition keeps every source statement. -/
namespace VeriTile.Bench.TritonBenchG.RotaryEmb.FullSource
open Lean VeriTile.Triton
set_option linter.unusedVariables false

inductive Storage where
  | fp32 | fp64 | fp16 | bf16 | f8e4 | f8e5
  deriving DecidableEq, Repr

private def dtypeCode (stx : TSyntax `tritonDType) : MacroM Nat :=
  match stx with
  | `(tritonDType| tl.float32) => pure 0
  | `(tritonDType| tl.float64) => pure 1
  | `(tritonDType| tl.float16) => pure 2
  | `(tritonDType| tl.bfloat16) => pure 3
  | `(tritonDType| tl.float8e4nv) => pure 4
  | `(tritonDType| tl.float8e5) => pure 5
  | _ => Macro.throwError "unsupported rotary storage"

/-- Triton v3.5.1 tensor arithmetic promotion, used only to spell the implicit
store conversion; operation precision remains determined by the actual DSL. -/
private def promotedCode (a b : Nat) : Nat :=
  if a=1 || b=1 then 1 else if a=0 || b=0 then 0 else
  if a=2 || b=2 then 2 else if a=3 && b=3 then 3 else
  if a=3 || b=3 then 0 else if a=b then a else 2

syntax "rotaryFull%" "[" tritonDType "," tritonDType "," tritonDType "]" : term
set_option hygiene false in
macro_rules
  | `(rotaryFull%[$data:tritonDType, $cos:tritonDType, $sin:tritonDType]) => do
    let d ← dtypeCode data
    let c ← dtypeCode cos
    let sn ← dtypeCode sin
    let same := promotedCode (promotedCode d c) (promotedCode d sn) == d
    let storeQ0 : TSyntax `tritonStmt ← if same then
      `(tritonStmt| tl.store(Q + off_q0, out0,
        mask=(cur_seq_range[:, None, None] < $(max_total_len)) &
          (cur_head_range[None, :, None] < $(HEAD_Q))))
      else `(tritonStmt| tl.store(Q + off_q0, out0.to($data),
        mask=(cur_seq_range[:, None, None] < $(max_total_len)) &
          (cur_head_range[None, :, None] < $(HEAD_Q))))
    let storeQ1 : TSyntax `tritonStmt ← if same then
      `(tritonStmt| tl.store(Q + off_q1, out1,
        mask=(cur_seq_range[:, None, None] < $(max_total_len)) &
          (cur_head_range[None, :, None] < $(HEAD_Q))))
      else `(tritonStmt| tl.store(Q + off_q1, out1.to($data),
        mask=(cur_seq_range[:, None, None] < $(max_total_len)) &
          (cur_head_range[None, :, None] < $(HEAD_Q))))
    let storeK0 : TSyntax `tritonStmt ← if same then
      `(tritonStmt| tl.store(K + off_k0, out_k0,
        mask=(cur_seq_range[:, None, None] < $(max_total_len)) &
          (cur_head_range[None, :, None] < $(HEAD_K))))
      else `(tritonStmt| tl.store(K + off_k0, out_k0.to($data),
        mask=(cur_seq_range[:, None, None] < $(max_total_len)) &
          (cur_head_range[None, :, None] < $(HEAD_K))))
    let storeK1 : TSyntax `tritonStmt ← if same then
      `(tritonStmt| tl.store(K + off_k1, out_k1,
        mask=(cur_seq_range[:, None, None] < $(max_total_len)) &
          (cur_head_range[None, :, None] < $(HEAD_K))))
      else `(tritonStmt| tl.store(K + off_k1, out_k1.to($data),
        mask=(cur_seq_range[:, None, None] < $(max_total_len)) &
          (cur_head_range[None, :, None] < $(HEAD_K))))
    `(fun (Q K Cos Sin : RegionName)
      (stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd
        stride_cosbs stride_cosd stride_sinbs stride_sind max_total_len
        HEAD_Q HEAD_K BLOCK_HEAD BLOCK_SEQ BLOCK_DMODEL : Nat) =>
      ComputeKernel.mk [Q,K,Cos,Sin] [Q,K] (triton {
  tl.region Q = $data, K = $data, Cos = $cos, Sin = $sin
  cur_head_index = tl.program_id(0)
  cur_seq_index = tl.program_id(1)

  cur_head_range = cur_head_index * $(BLOCK_HEAD) + tl.arange(0, $(BLOCK_HEAD))
  cur_seq_range = cur_seq_index * $(BLOCK_SEQ) + tl.arange(0, $(BLOCK_SEQ))

  dim_range0 = tl.arange(0, $(BLOCK_DMODEL) // $(2)) * $(2)
  dim_range1 = tl.arange(0, $(BLOCK_DMODEL) // $(2)) * $(2) + $(1)

  off_q0 = cur_seq_range[:, None, None] * $(stride_qbs) +
    cur_head_range[None, :, None] * $(stride_qh) +
    dim_range0[None, None, :] * $(stride_qd)
  off_q1 = cur_seq_range[:, None, None] * $(stride_qbs) +
    cur_head_range[None, :, None] * $(stride_qh) +
    dim_range1[None, None, :] * $(stride_qd)

  off_dimcos_sin0 = cur_seq_range[:, None, None] * $(stride_cosbs) +
    dim_range0[None, None, :] * $(stride_cosd)
  off_dimcos_sin1 = cur_seq_range[:, None, None] * $(stride_cosbs) +
    dim_range1[None, None, :] * $(stride_cosd)

  q0 = tl.load(Q + off_q0,
    mask=(cur_seq_range[:, None, None] < $(max_total_len)) &
      (cur_head_range[None, :, None] < $(HEAD_Q)),
    other=0.0)
  q1 = tl.load(Q + off_q1,
    mask=(cur_seq_range[:, None, None] < $(max_total_len)) &
      (cur_head_range[None, :, None] < $(HEAD_Q)),
    other=0.0)

  cos0 = tl.load(Cos + off_dimcos_sin0,
    mask=cur_seq_range[:, None, None] < $(max_total_len), other=0.0)
  sin0 = tl.load(Sin + off_dimcos_sin0,
    mask=cur_seq_range[:, None, None] < $(max_total_len), other=0.0)

  cos1 = tl.load(Cos + off_dimcos_sin1,
    mask=cur_seq_range[:, None, None] < $(max_total_len), other=0.0)
  sin1 = tl.load(Sin + off_dimcos_sin1,
    mask=cur_seq_range[:, None, None] < $(max_total_len), other=0.0)

  out0 = q0 * cos0 - q1 * sin0
  out1 = q0 * sin1 + q1 * cos1

  $storeQ0
  $storeQ1

  off_k0 = cur_seq_range[:, None, None] * $(stride_kbs) +
    cur_head_range[None, :, None] * $(stride_kh) +
    dim_range0[None, None, :] * $(stride_kd)
  off_k1 = cur_seq_range[:, None, None] * $(stride_kbs) +
    cur_head_range[None, :, None] * $(stride_kh) +
    dim_range1[None, None, :] * $(stride_kd)

  off_dimcos_sin0 = cur_seq_range[:, None, None] * $(stride_cosbs) +
    dim_range0[None, None, :] * $(stride_cosd)
  off_dimcos_sin1 = cur_seq_range[:, None, None] * $(stride_cosbs) +
    dim_range1[None, None, :] * $(stride_cosd)

  k0 = tl.load(K + off_k0,
    mask=(cur_seq_range[:, None, None] < $(max_total_len)) &
      (cur_head_range[None, :, None] < $(HEAD_K)),
    other=0.0)
  k1 = tl.load(K + off_k1,
    mask=(cur_seq_range[:, None, None] < $(max_total_len)) &
      (cur_head_range[None, :, None] < $(HEAD_K)),
    other=0.0)

  cos0 = tl.load(Cos + off_dimcos_sin0,
    mask=cur_seq_range[:, None, None] < $(max_total_len), other=0.0)
  sin0 = tl.load(Sin + off_dimcos_sin0,
    mask=cur_seq_range[:, None, None] < $(max_total_len), other=0.0)

  cos1 = tl.load(Cos + off_dimcos_sin1,
    mask=cur_seq_range[:, None, None] < $(max_total_len), other=0.0)
  sin1 = tl.load(Sin + off_dimcos_sin1,
    mask=cur_seq_range[:, None, None] < $(max_total_len), other=0.0)

  out_k0 = k0 * cos0 - k1 * sin0
  out_k1 = k0 * sin1 + k1 * cos1

  $storeK0
  $storeK1
}).surfaceBody)

abbrev Source := RegionName → RegionName → RegionName → RegionName →
  Nat → Nat → Nat → Nat → Nat → Nat → Nat → Nat →
  Nat → Nat → Nat → Nat → Nat → Nat → Nat → Nat → ComputeKernel


end VeriTile.Bench.TritonBenchG.RotaryEmb.FullSource
