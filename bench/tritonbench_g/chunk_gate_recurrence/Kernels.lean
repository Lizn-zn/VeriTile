import VeriTile.Triton.DSL
import VeriTile.Triton.Source

/-! Source: chunk_gate_recurrence.py, _fwd_recurrence

Complete per-program sources for `_fwd_recurrence` and `_bwd_recurrence`
in chunk_gate_recurrence.py. The sum-left candidates commute only the addition
in acc*d+S (forward) and Dacc*d+DS (backward, including its final DL update).
Products, fp32 accumulation, source storage, every cast, the whole-tile DG sum,
optional seed and all pointer updates/stores stay in their original order.

Forward O uses S storage. Backward DI uses DS storage, DG uses gate storage,
and DL uses S storage. These three storage choices remain independent.
Pointer element-type casts are resolved explicitly from these storage choices;
they must not be erased by the attribute-cast fallback.
The optional seed is specialized by a Boolean. Sizes and program IDs are
symbolic; the real contracts require a positive chunk count and safe unmasked
tile accesses. Addresses use mathematical naturals: the unused S/DI/DG pointer
positions before chunk zero are clipped, and no load/store observes them.
Host grid composition and fixed-width overflow are outside this source model.

The Python host clears the last DI/DG slices after the backward kernel and
reduces DG across K/V tiles. Those host operations are not kernel stores.
-/
namespace VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.TypedSource
open Lean VeriTile.Triton

syntax "gateForwardSource%" "[" tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(gateForwardSource%[$sTy:tritonDType, $gateTy:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `sumLeft do
      Macro.throwError "expected original or sumLeft"
    let update : TSyntax `tritonStmt ←
      if version.getId == `sumLeft then `(tritonStmt| acc = S_i + acc * d_i)
      else `(tritonStmt| acc = acc * d_i + S_i)
    `(fun (dtype : FloatDType) (S D O last_kv : RegionName)
        (_NUM_HEAD NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat)
        (HAS_LAST_KV : Bool) =>
      ComputeKernel.mk [S, D, last_kv] [O] (triton {
        tl.region S = $sTy, O = $sTy, D = $gateTy, last_kv = DTYPE
        offset_bh = tl.program_id(0)
        offset_d = tl.program_id(1)
        offset_s = tl.program_id(2)

        S = S + offset_bh * $(NUM_BLOCK) * $(D_MODEL_K) * $(D_MODEL_V) +
          offset_d * $(D_MODEL_V) * $(BLOCK_MODEL_K) +
          tl.arange(0, $(BLOCK_MODEL_K))[:, None] * $(D_MODEL_V) +
          offset_s * $(BLOCK_MODEL_V) + tl.arange(0, $(BLOCK_MODEL_V))[None, :]
        O = O + offset_bh * $(NUM_BLOCK) * $(D_MODEL_K) * $(D_MODEL_V) +
          offset_d * $(D_MODEL_V) * $(BLOCK_MODEL_K) +
          tl.arange(0, $(BLOCK_MODEL_K))[:, None] * $(D_MODEL_V) +
          offset_s * $(BLOCK_MODEL_V) + tl.arange(0, $(BLOCK_MODEL_V))[None, :]
        if HAS_LAST_KV {
          last_kv = last_kv + offset_bh * $(D_MODEL_K) * $(D_MODEL_V) +
            offset_d * $(D_MODEL_V) * $(BLOCK_MODEL_K) +
            tl.arange(0, $(BLOCK_MODEL_K))[:, None] * $(D_MODEL_V) +
            offset_s * $(BLOCK_MODEL_V) + tl.arange(0, $(BLOCK_MODEL_V))[None, :]
          acc = tl.load(last_kv).to(tl.float32)
        } else {
          acc = tl.zeros([$(BLOCK_MODEL_K), $(BLOCK_MODEL_V)], dtype=tl.float32)
        }
        tl.store(O, (acc).to($sTy:tritonDType))
        O += $(D_MODEL_K) * $(D_MODEL_V)
        D = D + offset_bh * $(NUM_BLOCK)
        for _i in range($(0), $(NUM_BLOCK) - $(1), $(1)) {
          d_i = tl.load(D)
          S_i = tl.load(S)
          $update
          tl.store(O, (acc).to($sTy:tritonDType))
          D += $(1)
          S += $(D_MODEL_K) * $(D_MODEL_V)
          O += $(D_MODEL_K) * $(D_MODEL_V)
        }
      }).surfaceBody)

/-- Source: chunk_gate_recurrence.py, _bwd_recurrence. The offs_k/offs_v
registers name repeated arange expressions from the Python source. -/
syntax "gateBackwardSource%" "[" tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(gateBackwardSource%[$gateTy:tritonDType, $dsTy:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `sumLeft do
      Macro.throwError "expected original or sumLeft"
    let update : TSyntax `tritonStmt ←
      if version.getId == `sumLeft then `(tritonStmt| Dacc = DS_i + Dacc * d_i)
      else `(tritonStmt| Dacc = Dacc * d_i + DS_i)
    `(fun (dtype : FloatDType) (S D DI DG DL DS : RegionName)
        (_NUM_HEAD NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat) =>
      ComputeKernel.mk [S, D, DS] [DI, DG, DL] (triton {
        tl.region S = DTYPE, DL = DTYPE, D = $gateTy, DG = $gateTy,
          DS = $dsTy, DI = $dsTy
        offset_bh = tl.program_id(0)
        offset_d = tl.program_id(1)
        offset_s = tl.program_id(2)
        NUM_K = $(D_MODEL_K) // $(BLOCK_MODEL_K)
        NUM_V = $(D_MODEL_V) // $(BLOCK_MODEL_V)
        offs_k = tl.arange(0, $(BLOCK_MODEL_K))
        offs_v = tl.arange(0, $(BLOCK_MODEL_V))
        S = S + offset_bh * $(NUM_BLOCK) * $(D_MODEL_K) * $(D_MODEL_V) +
          offset_d * $(D_MODEL_V) * $(BLOCK_MODEL_K) +
          offs_k[:, None] * $(D_MODEL_V) + offset_s * $(BLOCK_MODEL_V) +
          offs_v[None, :] + ($(NUM_BLOCK) - $(2)) * $(D_MODEL_K) * $(D_MODEL_V)
        DI = DI + offset_bh * $(NUM_BLOCK) * $(D_MODEL_K) * $(D_MODEL_V) +
          offset_d * $(D_MODEL_V) * $(BLOCK_MODEL_K) +
          offs_k[:, None] * $(D_MODEL_V) + offset_s * $(BLOCK_MODEL_V) +
          offs_v[None, :] + ($(NUM_BLOCK) - $(2)) * $(D_MODEL_K) * $(D_MODEL_V)
        DS = DS + offset_bh * $(NUM_BLOCK) * $(D_MODEL_K) * $(D_MODEL_V) +
          offset_d * $(D_MODEL_V) * $(BLOCK_MODEL_K) +
          offs_k[:, None] * $(D_MODEL_V) + offset_s * $(BLOCK_MODEL_V) +
          offs_v[None, :] + ($(NUM_BLOCK) - $(1)) * $(D_MODEL_K) * $(D_MODEL_V)
        DG = DG + offset_bh * $(NUM_BLOCK) * NUM_K * NUM_V +
          offset_d * NUM_V + offset_s + ($(NUM_BLOCK) - $(2)) * NUM_K * NUM_V
        D = D + offset_bh * $(NUM_BLOCK) + ($(NUM_BLOCK) - $(1))
        Dacc = tl.zeros([$(BLOCK_MODEL_K), $(BLOCK_MODEL_V)], dtype=tl.float32)
        for _i in range($(0), $(NUM_BLOCK) - $(1), $(1)) {
          S_i = tl.load(S)
          DS_i = tl.load(DS)
          d_i = tl.load(D)
          $update
          DG_i = tl.sum(Dacc * (S_i).to(tl.float32))
          tl.store(DG, (DG_i).to($gateTy:tritonDType))
          tl.store(DI, (Dacc).to($dsTy:tritonDType))
          S -= $(D_MODEL_K) * $(D_MODEL_V)
          DI -= $(D_MODEL_K) * $(D_MODEL_V)
          DS -= $(D_MODEL_K) * $(D_MODEL_V)
          DG -= NUM_K * NUM_V
          D -= $(1)
        }
        DL = DL + offset_bh * $(D_MODEL_K) * $(D_MODEL_V) +
          offset_d * $(D_MODEL_V) * $(BLOCK_MODEL_K) +
          offs_k[:, None] * $(D_MODEL_V) + offset_s * $(BLOCK_MODEL_V) +
          offs_v[None, :]
        DS_i = tl.load(DS)
        d_i = tl.load(D)
        $update
        tl.store(DL, (Dacc).to(DTYPE))
      }).surfaceBody)

inductive Storage where
  | fp32 | fp16 | bf16
  deriving DecidableEq

def Storage.dtype : Storage → FloatDType
  | .fp32 => .fp32 | .fp16 => .fp16 | .bf16 => .bf16

def specializeForward (source gate : Storage) (swapped : Bool) :=
  match source, gate, swapped with
  | .fp32, .fp32, Bool.false => gateForwardSource%[tl.float32, tl.float32, original]
  | .fp32, .fp32, Bool.true => gateForwardSource%[tl.float32, tl.float32, sumLeft]
  | .fp32, .fp16, Bool.false => gateForwardSource%[tl.float32, tl.float16, original]
  | .fp32, .fp16, Bool.true => gateForwardSource%[tl.float32, tl.float16, sumLeft]
  | .fp32, .bf16, Bool.false => gateForwardSource%[tl.float32, tl.bfloat16, original]
  | .fp32, .bf16, Bool.true => gateForwardSource%[tl.float32, tl.bfloat16, sumLeft]
  | .fp16, .fp32, Bool.false => gateForwardSource%[tl.float16, tl.float32, original]
  | .fp16, .fp32, Bool.true => gateForwardSource%[tl.float16, tl.float32, sumLeft]
  | .fp16, .fp16, Bool.false => gateForwardSource%[tl.float16, tl.float16, original]
  | .fp16, .fp16, Bool.true => gateForwardSource%[tl.float16, tl.float16, sumLeft]
  | .fp16, .bf16, Bool.false => gateForwardSource%[tl.float16, tl.bfloat16, original]
  | .fp16, .bf16, Bool.true => gateForwardSource%[tl.float16, tl.bfloat16, sumLeft]
  | .bf16, .fp32, Bool.false => gateForwardSource%[tl.bfloat16, tl.float32, original]
  | .bf16, .fp32, Bool.true => gateForwardSource%[tl.bfloat16, tl.float32, sumLeft]
  | .bf16, .fp16, Bool.false => gateForwardSource%[tl.bfloat16, tl.float16, original]
  | .bf16, .fp16, Bool.true => gateForwardSource%[tl.bfloat16, tl.float16, sumLeft]
  | .bf16, .bf16, Bool.false => gateForwardSource%[tl.bfloat16, tl.bfloat16, original]
  | .bf16, .bf16, Bool.true => gateForwardSource%[tl.bfloat16, tl.bfloat16, sumLeft]

def specializeBackward (gate incoming : Storage) (swapped : Bool) :=
  match gate, incoming, swapped with
  | .fp32, .fp32, Bool.false => gateBackwardSource%[tl.float32, tl.float32, original]
  | .fp32, .fp32, Bool.true => gateBackwardSource%[tl.float32, tl.float32, sumLeft]
  | .fp32, .fp16, Bool.false => gateBackwardSource%[tl.float32, tl.float16, original]
  | .fp32, .fp16, Bool.true => gateBackwardSource%[tl.float32, tl.float16, sumLeft]
  | .fp32, .bf16, Bool.false => gateBackwardSource%[tl.float32, tl.bfloat16, original]
  | .fp32, .bf16, Bool.true => gateBackwardSource%[tl.float32, tl.bfloat16, sumLeft]
  | .fp16, .fp32, Bool.false => gateBackwardSource%[tl.float16, tl.float32, original]
  | .fp16, .fp32, Bool.true => gateBackwardSource%[tl.float16, tl.float32, sumLeft]
  | .fp16, .fp16, Bool.false => gateBackwardSource%[tl.float16, tl.float16, original]
  | .fp16, .fp16, Bool.true => gateBackwardSource%[tl.float16, tl.float16, sumLeft]
  | .fp16, .bf16, Bool.false => gateBackwardSource%[tl.float16, tl.bfloat16, original]
  | .fp16, .bf16, Bool.true => gateBackwardSource%[tl.float16, tl.bfloat16, sumLeft]
  | .bf16, .fp32, Bool.false => gateBackwardSource%[tl.bfloat16, tl.float32, original]
  | .bf16, .fp32, Bool.true => gateBackwardSource%[tl.bfloat16, tl.float32, sumLeft]
  | .bf16, .fp16, Bool.false => gateBackwardSource%[tl.bfloat16, tl.float16, original]
  | .bf16, .fp16, Bool.true => gateBackwardSource%[tl.bfloat16, tl.float16, sumLeft]
  | .bf16, .bf16, Bool.false => gateBackwardSource%[tl.bfloat16, tl.bfloat16, original]
  | .bf16, .bf16, Bool.true => gateBackwardSource%[tl.bfloat16, tl.bfloat16, sumLeft]

def forward (source gate seed : Storage) :=
  specializeForward source gate Bool.false seed.dtype

def forwardSumLeft (source gate seed : Storage) :=
  specializeForward source gate Bool.true seed.dtype

def backward (source gate incoming : Storage) :=
  specializeBackward gate incoming Bool.false source.dtype

def backwardSumLeft (source gate incoming : Storage) :=
  specializeBackward gate incoming Bool.true source.dtype

end VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.TypedSource
