import VeriTile.Triton.Core.Float8
import VeriTile.Meta.StatementAudit
import Mathlib.Tactic.NormNum

set_option maxRecDepth 4096
set_option maxHeartbeats 800000

namespace Float8E5BitsTests
open VeriTile.Triton Float8E5Bits

private def bytes : List Float8E5Bits := List.finRange 256
private def count (c : Class) : Nat := (bytes.filter (fun b => b.classify = c)).length

/-- Exhaust the byte space, including both zero signs and all NaN payloads. -/
theorem encoding_counts :
    count .zero = 2 ∧ count .subnormal = 6 ∧ count .normal = 240 ∧
    count .infinity = 2 ∧ count .nan = 6 := by decide

theorem zero_sign_preserved :
    (ofInt 0).sign = false ∧ (ofInt (-128)).sign = true ∧
    (ofInt 0).decodeRat = some 0 ∧ (ofInt (-128)).decodeRat = some 0 := by
  norm_num [ofInt, sign, exponent, fraction, decodeRat, classify, isInt8, Int.toNat]

theorem finite_boundaries :
    (ofInt 1).decodeRat = some (1 / 65536) ∧
    (ofInt 3).decodeRat = some (3 / 65536) ∧
    (ofInt 4).decodeRat = some (1 / 16384) ∧
    (ofInt 123).decodeRat = some 57344 ∧
    (ofInt (-5)).decodeRat = some (-57344) := by
  norm_num [ofInt, sign, exponent, fraction, decodeRat, classify, isInt8, Int.toNat]

theorem numerical_conversion_differs :
    (ofInt 60).decodeRat = some 1 ∧ (ofInt 71).decodeRat = some 7 ∧
    (ofInt (-68)).decodeRat = some (-1) := by
  norm_num [ofInt, sign, exponent, fraction, decodeRat, classify, isInt8, Int.toNat]

theorem special_payloads_preserved :
    (ofInt 124).classify = .infinity ∧ (ofInt (-4)).classify = .infinity ∧
    (ofInt 125).classify = .nan ∧ (ofInt 126).classify = .nan ∧
    ofInt 125 ≠ ofInt 126 ∧ (ofInt 124).decodeRat = none ∧
    (ofInt (-4)).decodeRat = none ∧ (ofInt 127).decodeRat = none := by
  norm_num [ofInt, sign, exponent, fraction, decodeRat, classify, isInt8, Int.toNat]

theorem signed_byte_bounds :
    isInt8 (-129) = false ∧ isInt8 (-128) = true ∧
    isInt8 127 = true ∧ isInt8 128 = false := by
  norm_num [ofInt, sign, exponent, fraction, decodeRat, classify, isInt8, Int.toNat]

#guard_msgs (drop info) in
#auditModuleAxioms
end Float8E5BitsTests
