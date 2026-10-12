import VeriTile.Triton.Semantics.ProgramRewrite
import VeriTile.Meta.StatementAudit

/-! Real execution of the FP8 input-conversion fragment. Precision erasure
keeps the byte load and E5M2 decoding, while the subsequent fp16 cast becomes
an identity assignment. The fragment can be fused without assuming decoding
succeeds. This is a helper for the real proof, not an attention specification.
It does not use FP admission rules. -/
namespace VeriTile.Bench.TritonBenchG.AttentionLlama.FP8RealLoads
open VeriTile.Triton

private theorem overwrite (s : BlockState) (name : RegName)
    (a : Tile da sha) (b : Tile db shb) :
    (s.setReg name da sha a).setReg name db shb b = s.setReg name db shb b := by
  apply BlockState.ext <;> intros <;> try rfl
  simp only [BlockState.setReg]
  split <;> rfl

/-- The three erased statements from each K/V conversion site. -/
def conversion (name : RegName) (bytes : Op .int shape) : List Stmt :=
  [.assign .int shape name bytes,
   .assign .real shape name (.decodeFloat8E5 .real (.ref .int shape name)),
   .assign .real shape name (.ref .real shape name)]

private theorem eval_decode (bytes : Op .int shape) (s : BlockState) :
    @evalOp .real shape (.decodeFloat8E5 .real bytes) s =
      (evalOp bytes s).bind (Float8E5Bits.decodeTile .real) := by
  rw [evalOp.eq_def]
  rfl

/-- Preserve the complete final state and all failure cases, including an
invalid byte or a nonfinite encoding. Decoding is retained in the fused load. -/
theorem fuse_conversion (name : RegName) (bytes : Op .int shape) (s : BlockState) :
    stepStmts (conversion name bytes) s =
      stepStmts [.assign .real shape name (.decodeFloat8E5 .real bytes)] s := by
  simp only [conversion, stepStmts, stepStmt, eval_decode]
  cases hbytes : evalOp bytes s with
  | none => simp
  | some raw =>
    simp only [evalOp, bind, Option.bind_some, BlockState.setReg_same]
    cases hdecode : Float8E5Bits.decodeTile .real raw <;>
      simp [overwrite]

/-- The only numerical premise is about the loaded input bytes themselves.
The decoded values are exact E5M2 rationals embedded into the real carrier. -/
theorem conversion_of_finite (name : RegName) (bytes : Op .int shape)
    (s : BlockState) (raw : Tile .int shape) (values : TileIndex shape → ℚ)
    (hload : evalOp bytes s = some raw)
    (hbyte : ∀ i, Float8E5Bits.isInt8 (raw.data i) = true)
    (hfinite : ∀ i, (Float8E5Bits.ofInt (raw.data i)).decodeRat = some (values i)) :
    stepStmts (conversion name bytes) s =
      some (s.setReg name .real shape ⟨fun i => some (values i : ℝ)⟩) := by
  rw [fuse_conversion]
  simp only [stepStmts, stepStmt, eval_decode, hload, bind, Option.bind_some,
    Float8E5Bits.decodeTile_of_finite .real raw values hbyte hfinite]
  rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionLlama.FP8RealLoads
