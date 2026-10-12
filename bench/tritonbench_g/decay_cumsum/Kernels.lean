import VeriTile.Triton.Source
import VeriTile.Triton.DSL
/-! Source: decay_cumsum.py. Complete forward scan, preparation and reverse
scan sources. Candidates commute only the exact integer product in the lane
mask. Storage, fp32 conversions, exp2, loop order and all accesses remain.
The preparation's last_decay load is unmasked. The reverse loop retains its
final unused pointer decrements. Pointer arithmetic uses the natural-address
model; no fixed-width overflow or host launch claim is made. -/
namespace VeriTile.Bench.TritonBenchG.DecayCumsum.TypedSource
open Lean VeriTile.Triton
inductive Storage where
  | fp16 | bf16 | fp32
  deriving DecidableEq
def inv_ln2 : ℝ := 1.44269504

/-! Source: decay_cumsum.py, fwd_decay_cumsum. -/
syntax "decayForwardSource%" "[" tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(decayForwardSource%[$dataTy:tritonDType, $gateTy:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `indexCommuted do
      Macro.throwError "expected original or indexCommuted"
    let maskStmt : TSyntax `tritonStmt ←
      if version.getId == `indexCommuted then `(tritonStmt| mask = ($(BK) * i_k + tl.arange(0, $(BK)) < $(DK)))
      else `(tritonStmt| mask = (i_k * $(BK) + tl.arange(0, $(BK)) < $(DK)))
    `(fun (g g_o : RegionName) (s_qk_h s_qk_t s_qk_d B H T BT BK DK : Nat) (scale : ℝ) =>
      ComputeKernel.mk [g] [g_o] (triton {
  tl.region g = $gateTy, g_o = $gateTy
  i_k, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
  p_g = g + i_bh * $(s_qk_h) + i_c * $(BT) * $(DK) + i_k * $(BK) + tl.arange(0, $(BK))
  p_go = g_o + i_bh * $(s_qk_h) + i_c * $(BT) * $(DK) + i_k * $(BK) + tl.arange(0, $(BK))
  cum_decay = tl.zeros([$(BK)], dtype=tl.float32)
  $maskStmt
  for i in range($(BT)) {
    _g = (tl.load(p_g, mask=mask, other=0)).to(tl.float32)
    cum_decay += _g * $((inv_ln2 : ℝ))
    tl.store(p_go, (cum_decay).to($gateTy:tritonDType), mask=mask)
    p_g += $(DK)
    p_go += $(DK)
  }
}).surfaceBody)

def forward (storage : Storage) (swapped : Bool) := match storage with
  | .fp16 => if swapped then decayForwardSource%[tl.float16,tl.float16,indexCommuted] else decayForwardSource%[tl.float16,tl.float16,original]
  | .bf16 => if swapped then decayForwardSource%[tl.bfloat16,tl.bfloat16,indexCommuted] else decayForwardSource%[tl.bfloat16,tl.bfloat16,original]
  | .fp32 => if swapped then decayForwardSource%[tl.float32,tl.float32,indexCommuted] else decayForwardSource%[tl.float32,tl.float32,original]

/-! Source: decay_cumsum.py, prepare_qg_kg. -/
syntax "decayPrepareSource%" "[" tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(decayPrepareSource%[$dataTy:tritonDType, $gateTy:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `indexCommuted do
      Macro.throwError "expected original or indexCommuted"
    let maskStmt : TSyntax `tritonStmt ←
      if version.getId == `indexCommuted then `(tritonStmt| mask = ($(BK) * i_k + tl.arange(0, $(BK)) < $(DK)))
      else `(tritonStmt| mask = (i_k * $(BK) + tl.arange(0, $(BK)) < $(DK)))
    `(fun (q k g qg kg : RegionName) (s_qk_h s_qk_t s_qk_d B H T BT BK DK : Nat) (scale : ℝ) =>
      ComputeKernel.mk [q, k, g] [qg, kg] (triton {
  tl.region q = $dataTy, k = $dataTy, g = $gateTy, qg = $dataTy, kg = $dataTy
  i_k, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
  p_q = q + i_bh * $(s_qk_h) + i_c * $(BT) * $(DK) + i_k * $(BK) + tl.arange(0, $(BK))
  p_g = g + i_bh * $(s_qk_h) + i_c * $(BT) * $(DK) + i_k * $(BK) + tl.arange(0, $(BK))
  p_k = k + i_bh * $(s_qk_h) + i_c * $(BT) * $(DK) + i_k * $(BK) + tl.arange(0, $(BK))
  p_qg = qg + i_bh * $(s_qk_h) + i_c * $(BT) * $(DK) + i_k * $(BK) + tl.arange(0, $(BK))
  p_kg = kg + i_bh * $(s_qk_h) + i_c * $(BT) * $(DK) + i_k * $(BK) + tl.arange(0, $(BK))
  $maskStmt
  last_decay = tl.load(g + i_bh * $(s_qk_h) + (i_c * $(BT) + $(BT) - $(1)) * $(DK) + i_k * $(BK) + tl.arange(0, $(BK)))
  for i in range($(BT)) {
    _q = tl.load(p_q, mask=mask, other=0)
    _k = tl.load(p_k, mask=mask, other=0)
    _g = (tl.load(p_g, mask=mask, other=0)).to(tl.float32)
    _q *= tl.math.exp2(_g) * $((scale : ℝ))
    _k *= tl.math.exp2(last_decay - _g)
    tl.store(p_kg, (_k).to($dataTy:tritonDType), mask=mask)
    tl.store(p_qg, (_q).to($dataTy:tritonDType), mask=mask)
    p_q += $(DK)
    p_g += $(DK)
    p_k += $(DK)
    p_kg += $(DK)
    p_qg += $(DK)
  }
}).surfaceBody)

def prepare (source gate : Storage) (swapped : Bool) := match source, gate with
  | .fp16, .fp16 => if swapped then decayPrepareSource%[tl.float16,tl.float16,indexCommuted] else decayPrepareSource%[tl.float16,tl.float16,original]
  | .fp16, .bf16 => if swapped then decayPrepareSource%[tl.float16,tl.bfloat16,indexCommuted] else decayPrepareSource%[tl.float16,tl.bfloat16,original]
  | .fp16, .fp32 => if swapped then decayPrepareSource%[tl.float16,tl.float32,indexCommuted] else decayPrepareSource%[tl.float16,tl.float32,original]
  | .bf16, .fp16 => if swapped then decayPrepareSource%[tl.bfloat16,tl.float16,indexCommuted] else decayPrepareSource%[tl.bfloat16,tl.float16,original]
  | .bf16, .bf16 => if swapped then decayPrepareSource%[tl.bfloat16,tl.bfloat16,indexCommuted] else decayPrepareSource%[tl.bfloat16,tl.bfloat16,original]
  | .bf16, .fp32 => if swapped then decayPrepareSource%[tl.bfloat16,tl.float32,indexCommuted] else decayPrepareSource%[tl.bfloat16,tl.float32,original]
  | .fp32, .fp16 => if swapped then decayPrepareSource%[tl.float32,tl.float16,indexCommuted] else decayPrepareSource%[tl.float32,tl.float16,original]
  | .fp32, .bf16 => if swapped then decayPrepareSource%[tl.float32,tl.bfloat16,indexCommuted] else decayPrepareSource%[tl.float32,tl.bfloat16,original]
  | .fp32, .fp32 => if swapped then decayPrepareSource%[tl.float32,tl.float32,indexCommuted] else decayPrepareSource%[tl.float32,tl.float32,original]

/-! Source: decay_cumsum.py, bwd_decay_global_cumsum. -/
syntax "decayBackwardSource%" "[" tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(decayBackwardSource%[$dataTy:tritonDType, $gateTy:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `indexCommuted do
      Macro.throwError "expected original or indexCommuted"
    let maskStmt : TSyntax `tritonStmt ←
      if version.getId == `indexCommuted then `(tritonStmt| mask = ($(BK) * i_k + tl.arange(0, $(BK)) < $(DK)))
      else `(tritonStmt| mask = (i_k * $(BK) + tl.arange(0, $(BK)) < $(DK)))
    `(fun (dq_inner dq_inter dk_inner dk_inter q k g dg : RegionName) (s_qk_h s_qk_t s_qk_d B H T BT BK DK : Nat) (scale : ℝ) =>
      ComputeKernel.mk [dq_inner, dq_inter, dk_inner, dk_inter, q, k, g] [dq_inter, dk_inter, dg] (triton {
  tl.region dq_inner = $dataTy, dq_inter = $dataTy, dk_inner = $dataTy, dk_inter = $dataTy, q = $dataTy, k = $dataTy, g = $gateTy, dg = $gateTy
  i_k, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
  p_q = q + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK)) + (i_c * $(BT) + $(BT) - $(1)) * $(DK)
  p_k = k + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK)) + (i_c * $(BT) + $(BT) - $(1)) * $(DK)
  p_g = g + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK)) + (i_c * $(BT) + $(BT) - $(1)) * $(DK)
  p_dg = dg + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK)) + (i_c * $(BT) + $(BT) - $(1)) * $(DK)
  p_dq_inner = dq_inner + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK)) + (i_c * $(BT) + $(BT) - $(1)) * $(DK)
  p_dk_inner = dk_inner + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK)) + (i_c * $(BT) + $(BT) - $(1)) * $(DK)
  p_dq_inter = dq_inter + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK)) + (i_c * $(BT) + $(BT) - $(1)) * $(DK)
  p_dk_inter = dk_inter + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK)) + (i_c * $(BT) + $(BT) - $(1)) * $(DK)
  cum_grad_dg = tl.zeros([$(BK)], dtype=tl.float32)
  $maskStmt
  last_g = tl.zeros([$(BK)], dtype=tl.float32)
  for j in range($(BT) - $(1), -$(1), -$(1)) {
    _g = (tl.load(p_g, mask=mask, other=0)).to(tl.float32)
    if (j == $(BT) - $(1)) {
      last_g = _g
    }
    _dq1 = tl.load(p_dq_inner, mask=mask, other=0)
    _dq2 = tl.load(p_dq_inter, mask=mask, other=0)
    _dq2 *= tl.math.exp2(_g)
    _dq = _dq1 + _dq2
    tl.store(p_dq_inter, (_dq).to($dataTy:tritonDType), mask=mask)
    _dk1 = tl.load(p_dk_inner, mask=mask, other=0)
    _dk2 = tl.load(p_dk_inter, mask=mask, other=0)
    _dk2 *= tl.math.exp2(last_g - _g)
    _dk = _dk1 + _dk2
    tl.store(p_dk_inter, (_dk).to($dataTy:tritonDType), mask=mask)
    _q = tl.load(p_q, mask=mask, other=0)
    _k = tl.load(p_k, mask=mask, other=0)
    _dg = _dq * _q - _dk * _k
    cum_grad_dg += _dg
    tl.store(p_dg, (cum_grad_dg).to($gateTy:tritonDType), mask=mask)
    p_g -= $(DK)
    p_k -= $(DK)
    p_q -= $(DK)
    p_dq_inner -= $(DK)
    p_dk_inner -= $(DK)
    p_dq_inter -= $(DK)
    p_dk_inter -= $(DK)
    p_dg -= $(DK)
  }
}).surfaceBody)

def backward (source gate : Storage) (swapped : Bool) := match source, gate with
  | .fp16, .fp16 => if swapped then decayBackwardSource%[tl.float16,tl.float16,indexCommuted] else decayBackwardSource%[tl.float16,tl.float16,original]
  | .fp16, .bf16 => if swapped then decayBackwardSource%[tl.float16,tl.bfloat16,indexCommuted] else decayBackwardSource%[tl.float16,tl.bfloat16,original]
  | .fp16, .fp32 => if swapped then decayBackwardSource%[tl.float16,tl.float32,indexCommuted] else decayBackwardSource%[tl.float16,tl.float32,original]
  | .bf16, .fp16 => if swapped then decayBackwardSource%[tl.bfloat16,tl.float16,indexCommuted] else decayBackwardSource%[tl.bfloat16,tl.float16,original]
  | .bf16, .bf16 => if swapped then decayBackwardSource%[tl.bfloat16,tl.bfloat16,indexCommuted] else decayBackwardSource%[tl.bfloat16,tl.bfloat16,original]
  | .bf16, .fp32 => if swapped then decayBackwardSource%[tl.bfloat16,tl.float32,indexCommuted] else decayBackwardSource%[tl.bfloat16,tl.float32,original]
  | .fp32, .fp16 => if swapped then decayBackwardSource%[tl.float32,tl.float16,indexCommuted] else decayBackwardSource%[tl.float32,tl.float16,original]
  | .fp32, .bf16 => if swapped then decayBackwardSource%[tl.float32,tl.bfloat16,indexCommuted] else decayBackwardSource%[tl.float32,tl.bfloat16,original]
  | .fp32, .fp32 => if swapped then decayBackwardSource%[tl.float32,tl.float32,indexCommuted] else decayBackwardSource%[tl.float32,tl.float32,original]

end VeriTile.Bench.TritonBenchG.DecayCumsum.TypedSource
