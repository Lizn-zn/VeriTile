import VeriTile.Triton.Semantics.EvalOp
import VeriTile.Triton.Float.Structural
import VeriTile.Triton.Float.Erasure
import VeriTile.Meta.StatementAudit
import Mathlib.Tactic.NormNum

/-! Runtime byte loads, bit reinterpretation, and the following fp16 cast
remain separate operations. The FP witness records payloads and casts; it is
not an IEEE emulator and asserts no floating algebraic identity. -/
set_option maxRecDepth 4096

namespace Float8E5ExecutionTests
open VeriTile.Triton
open FP.Structural

private def loadedByte : Op .int [] :=
  .load .int (.region "bytes" (.constNat 0)) .none

private def widenedByte : Op .fp16 [] :=
  .castFloat .f8e5 .fp16 (.decodeFloat8E5 .f8e5 loadedByte)

theorem erasure_keeps_reinterpretation :
    widenedByte.eraseDType = .decodeFloat8E5 .real loadedByte := by
  simp only [widenedByte, loadedByte, Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def, eraseDType, FloatDType.toTileDType]
  rfl

private def realState (byte : Int) : BlockState :=
  { (default : BlockState) with mem := fun _ _ => .of .int byte }

private theorem load_byte (byte : Int) :
    VeriTile.Triton.evalOp loadedByte (realState byte) = some (Tile.scalar byte) := by
  simp only [loadedByte, VeriTile.Triton.evalOp.eq_def, bind, Option.bind_some]
  rfl

private theorem decode_byte (dst : FloatDType) (byte : Int) :
    VeriTile.Triton.evalOp (.decodeFloat8E5 dst loadedByte) (realState byte) =
      Float8E5Bits.decodeTile dst (Tile.scalar byte) := by
  cases dst <;> simp only [VeriTile.Triton.evalOp, load_byte, bind, Option.bind_some]

private theorem widened_byte (byte : Int) :
    VeriTile.Triton.evalOp widenedByte (realState byte) = (do
      let v ← Float8E5Bits.decodeTile .f8e5 (Tile.scalar byte)
      pure ⟨fun i => FloatDType.cast .f8e5 .fp16 (v.data i)⟩) := by
  unfold widenedByte
  have h := evalOp_castFloat .f8e5 .fp16 (shape := [])
    (.decodeFloat8E5 .f8e5 loadedByte) (realState byte)
  exact h.trans (by rw [decode_byte]; rfl)

theorem real_byte_is_not_integer_conversion :
    VeriTile.Triton.evalOp widenedByte (realState 60) =
      some (Tile.scalar (some (1 : ℝ))) := by
  rw [widened_byte]
  norm_num [-Tile.scalar_eta, Float8E5Bits.decodeTile, Float8E5Bits.isInt8,
    Float8E5Bits.decodeRat, Float8E5Bits.ofInt, Float8E5Bits.exponent,
    Float8E5Bits.fraction, Float8E5Bits.sign, Tile.scalar, FloatDType.cast,
    FloatDType.ofReal, FloatDType.ofWithBot, FloatDType.toWithBot, Int.toNat]

theorem real_infinity_has_no_projection :
    VeriTile.Triton.evalOp widenedByte (realState 124) = none := by
  rw [widened_byte]
  norm_num [-Tile.scalar_eta, Float8E5Bits.decodeTile, Float8E5Bits.isInt8,
    Float8E5Bits.decodeRat, Float8E5Bits.ofInt, Float8E5Bits.exponent,
    Tile.scalar, Int.toNat]

theorem real_nan_has_no_projection :
    VeriTile.Triton.evalOp widenedByte (realState 127) = none := by
  rw [widened_byte]
  norm_num [-Tile.scalar_eta, Float8E5Bits.decodeTile, Float8E5Bits.isInt8,
    Float8E5Bits.decodeRat, Float8E5Bits.ofInt, Float8E5Bits.exponent,
    Tile.scalar, Int.toNat]

inductive Trace where
  | byte : FloatDType → Float8E5Bits → Trace
  | cast : FloatDType → FloatDType → Trace → Trace
  | other
  deriving Inhabited

private noncomputable def model : Algebra Trace where
  literal := fun _ _ _ => .other
  negInf := .other
  binary := fun _ _ _ _ _ => .other
  unary := fun _ _ _ => .other
  cast := fun _ src dst a => .cast src dst a
  fromNat := fun _ _ => .other
  fromInt := fun _ _ => .other
  fp32Bits := fun _ => .other
  float8E5Bits := some Trace.byte
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ _ => .other
  reduceSum := fun _ {_} _ _ _ _ => .other

private def fpState (byte : Int) : State Trace where
  mem := fun _ _ => .mk .int byte
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

/-- This includes negative zero, infinities and NaNs. No finite-input premise
is needed by a backend that accepts the raw E5M2 representation. -/
theorem fp_runtime_keeps_payload (byte : Int)
    (hbyte : Float8E5Bits.isInt8 byte = true) :
    FP.Structural.evalOp model none widenedByte (fpState byte) =
      some (fun _ => Trace.cast .f8e5 .fp16
        (.byte .f8e5 (Float8E5Bits.ofInt byte))) := by
  simp [widenedByte, loadedByte, FP.Structural.evalOp, model, fpState,
    Cell.read, hbyte, ofFloat, toFloat]

theorem fp_rejects_invalid_byte :
    FP.Structural.evalOp model none widenedByte (fpState 128) = none := by
  simp [widenedByte, loadedByte, FP.Structural.evalOp, model, fpState,
    Cell.read, Float8E5Bits.isInt8]

theorem negative_zero_reaches_cast :
    FP.Structural.evalOp model none widenedByte (fpState (-128)) =
      some (fun _ => Trace.cast .f8e5 .fp16 (.byte .f8e5 ⟨128, by decide⟩)) :=
  fp_runtime_keeps_payload (-128) (by decide)

theorem nan_payload_reaches_cast :
    FP.Structural.evalOp model none widenedByte (fpState 126) =
      some (fun _ => Trace.cast .f8e5 .fp16 (.byte .f8e5 ⟨126, by decide⟩)) :=
  fp_runtime_keeps_payload 126 (by decide)

#guard_msgs (drop info) in
#auditModuleAxioms
end Float8E5ExecutionTests
