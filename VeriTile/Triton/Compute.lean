/-
VeriTile.Triton.Compute

Compute-facing dtype and bit-payload examples.
-/

import VeriTile.Triton.Core

namespace VeriTile.Triton

namespace Float32Bits

example :
    decodeRat { bits := 0x3f800000#32 } = some 1 := by
  native_decide

end Float32Bits

namespace ComputeOp

def oneBits : UInt32Bits :=
  { bits := 0x3f800000#32 }

def oneBitcast : ComputeOp .fp32 [] :=
  ComputeOp.bitcast .uint32 .fp32 rfl (ComputeOp.const oneBits)

example :
    constOpToAlgorithm? oneBitcast = Except.ok (Op.const 1) := by
  have hdecode :
      Float32Bits.decodeRat ({ bits := 0x3f800000#32 } : Float32Bits) = some 1 := by
    native_decide
  simp [constOpToAlgorithm?, oneBitcast, oneBits, constPayload?, bitcastPayload,
    constToAlgorithm?]
  change
    (match Float32Bits.decodeRat ({ bits := 0x3f800000#32 } : Float32Bits) with
      | some q => Except.ok (Op.const (q : ℝ))
      | none => Except.error (.unsupportedBitcast "unsupported fp32 decode")) =
    Except.ok (Op.const 1)
  rw [hdecode]
  norm_num

theorem oneBitcast_toAlgorithm :
    constOpToAlgorithm? oneBitcast = Except.ok (Op.const 1) := by
  have hdecode :
      Float32Bits.decodeRat ({ bits := 0x3f800000#32 } : Float32Bits) = some 1 := by
    native_decide
  simp [constOpToAlgorithm?, oneBitcast, oneBits, constPayload?, bitcastPayload,
    constToAlgorithm?]
  change
    (match Float32Bits.decodeRat ({ bits := 0x3f800000#32 } : Float32Bits) with
      | some q => Except.ok (Op.const (q : ℝ))
      | none => Except.error (.unsupportedBitcast "unsupported fp32 decode")) =
    Except.ok (Op.const 1)
  rw [hdecode]
  norm_num

end ComputeOp

end VeriTile.Triton
