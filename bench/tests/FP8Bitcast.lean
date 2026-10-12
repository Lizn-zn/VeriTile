import VeriTile.Triton.DSL
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

/-! Source width, pointer aliases, positional masks and the separate fp16
conversion must survive the frontend's byte-reinterpretation path. -/
namespace FP8BitcastTests
open VeriTile.Triton

def source (K Out : RegionName) : ComputeKernel := triton {
  tl.region K = tl.int8, Out = tl.float16
  offsets = tl.arange(0, 2)
  ptrs = K + offsets
  raw = tl.load(ptrs, offsets < $(1), other=0.0)
  bits = raw.to(tl.float8e5, bitcast=True)
  half = bits.to(tl.float16)
  tl.store(Out + offsets, half)
}

theorem runtime_reinterpretation : (source "K" "O").surfaceBody[3]? = some
    (.assign .f8e5 [2] "bits" (.alg
      (.decodeFloat8E5 .f8e5 (.ref .int [2] "raw")))) := rfl

theorem following_cast : (source "K" "O").surfaceBody[4]? = some
    (.assign .fp16 [2] "half" (.alg
      (.castFloat .f8e5 .fp16 (.ref .f8e5 [2] "bits")))) := rfl

def explicitLoad (K : RegionName) : ComputeKernel := triton {
  raw = tl.load(K + $(0), dtype=tl.int8)
  bits = tl.bitcast(raw, tl.float8e5)
}

theorem explicit_load_width : (explicitLoad "K").surfaceBody[1]? = some
    (.assign .f8e5 [] "bits" (.alg
      (.decodeFloat8E5 .f8e5 (.ref .int [] "raw")))) := rfl

def parenthesized (K : RegionName) : ComputeKernel := triton {
  tl.region K = tl.int8
  raw = tl.load(K + $(0))
  bits = (raw).to(tl.float8e5, bitcast=True)
}

theorem parenthesized_form : (parenthesized "K").surfaceBody[1]? = some
    (.assign .f8e5 [] "bits" (.alg
      (.decodeFloat8E5 .f8e5 (.ref .int [] "raw")))) := rfl

/-- Storage provenance must not interfere with existing integer stores.
Their carrier remains mathematical Int; this asserts no narrowing rule. -/
def integerStores (K Out : RegionName) : ComputeKernel := triton {
  tl.region K = tl.int8
  raw = tl.load(K + $(0))
  tl.store(Out + $(0), raw, dtype=tl.int8)
  tl.store(Out + $(1), raw + raw, dtype=tl.int8)
}

theorem byte_store_uses_integer_channel : (integerStores "K" "O").surfaceBody[1]? = some
    (.store .int [] (.region "O" (.constNat 0))
      (.alg (.ref .int [] "raw")) .none) := rfl

theorem arithmetic_store_uses_integer_channel : (integerStores "K" "O").surfaceBody[2]? = some
    (.store .int [] (.region "O" (.constNat 1))
      (.alg (.add .int .nil (.ref .int [] "raw") (.ref .int [] "raw"))) .none) := rfl

/-- error: tl.bitcast(..., tl.float8e5): requires a known int8 source width -/
#guard_msgs in
def wrongWidth (K : RegionName) : ComputeKernel := triton {
  tl.region K = tl.int32
  raw = tl.load(K + $(0))
  bits = raw.to(tl.float8e5, bitcast=True)
}

/-- error: tl.bitcast(..., tl.float8e5): requires a known int8 source width -/
#guard_msgs in
def numericalCastLosesByteWidth (K : RegionName) : ComputeKernel := triton {
  tl.region K = tl.int8
  raw = tl.load(K + $(0))
  wide = raw.to(tl.int32)
  bits = wide.to(tl.float8e5, bitcast=True)
}

/-- error: tl.bitcast(..., tl.float8e5): requires a known int8 source width -/
#guard_msgs in
def arithmeticIsNotAByteLoad (K : RegionName) : ComputeKernel := triton {
  tl.region K = tl.int8
  raw = tl.load(K + $(0))
  computed = raw + raw
  bits = computed.to(tl.float8e5, bitcast=True)
}

/-- error: tl.bitcast(..., tl.float8e5): requires a known int8 source width -/
#guard_msgs in
def atomicReturnIsNotAByteLoad (K Other : RegionName) : ComputeKernel := triton {
  tl.region K = tl.int8, Other = tl.int32
  raw = tl.load(K + $(0))
  previous = tl.atomic_xchg(Other + $(0), raw)
  bits = previous.to(tl.float8e5, bitcast=True)
}

open FP.Structural

/-- A trace backend: the cast adds a marker to the byte payload, so dropping
either conversion changes the observed output. This is not GPU arithmetic. -/
private noncomputable def model : Algebra Nat where
  literal := fun _ _ _ => 0
  negInf := 0
  binary := fun _ _ _ _ _ => 0
  unary := fun _ _ _ => 0
  cast := fun _ _ _ a => 1000 + a
  fromNat := fun _ n => n
  fromInt := fun _ n => n.toNat
  fp32Bits := fun _ => 0
  float8E5Bits := some (fun _ b => b.val)
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ _ => 0
  reduceSum := fun _ {_} _ _ _ _ => 0

private def initial : State Nat where
  mem := fun _ i => .mk .int (if i = 0 then 60 else 127)
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

/-- Lane one is masked: it uses the integer zero `other`, not the NaN byte
in memory. Both lanes then undergo bit reinterpretation and the fp16 cast. -/
theorem masked_load_cast_store :
    ∃ t, run model (source "K" "O").surfaceBody initial = some t ∧
      t.mem "O" 0 = .mk .fp16 1060 ∧
      t.mem "O" 1 = .mk .fp16 1000 ∧
      t.mem "K" 0 = .mk .int 60 ∧
      t.mem "K" 1 = .mk .int 127 := by
  refine ⟨?t, ?execution, ?outputs⟩
  case execution =>
    simp [source, ComputeKernel.surfaceBody, run, step, evalExpr,
      evalOp_unfold, model, initial, State.setReg,
      store, natLt,
      Float8E5Bits.isInt8, Float8E5Bits.ofInt, ofFloat, toFloat, Int.toNat]
    rfl
  case outputs => exact ⟨rfl, rfl, rfl, rfl⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end FP8BitcastTests
