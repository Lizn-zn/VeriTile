import bench.tritonbench_g.attention_fwd_triton3.Proofs.LegacySource
import VeriTile.Meta.Disprove
import VeriTile.Meta.StatementAudit

/-! Counterexample to the historical window transcription. This refutes source
correspondence for the mask; it does not claim an FP countermodel for the full
kernel. All expressions below are connected to the actual historical AST. -/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.SourceReview
open VeriTile.Triton

/-- Signed distance in `_attn_fwd_inner`, before any mask is applied. -/
def signedDistance (SM BM BN : Nat) (off : Int) {NC : Nat}
    (i : Fin BM) (j : Fin NC) : Int :=
  (i.val : Int) - (j.val % BN : Nat) + (SM * BM : Nat) -
    ((j.val / BN) * BN : Nat) + off

def signedKeep (SM BM BN : Nat) (off size : Int) {NC : Nat}
    (i : Fin BM) (j : Fin NC) : Prop :=
  0 ≤ signedDistance SM BM BN off i j ∧ signedDistance SM BM BN off i j < size

/-- Locate the actual distance assignment, including nested branches/loops.
Fuel bounds the search only: the equality below fails if the source is absent
or different. -/
private def distance? : Nat → List Stmt → Option (Op .nat [64, 64])
  | 0, _ => none
  | _ + 1, [] => none
  | n + 1, st :: rest =>
      let found := match st with
        | .assign .nat shape name value =>
            if name = "dist" then
              if h : shape = [64, 64] then some (h ▸ value) else none
            else none
        | .ifThen _ body => distance? n body
        | .ifThenElse _ yes no => distance? n yes <|> distance? n no
        | .forRange _ _ _ _ body => distance? n body
        | _ => none
      found <|> distance? n rest

private def oldSource : ComputeKernel := attention_fwd_triton3_surface
  "Q" "K" "V" "M" "Out" "L" 1
  8192 8192 64 1 8192 8192 64 1 8192 8192 64 1 8192 8192 64 1
  1 1 1 128 128 128 0 64 1 1 64 64 64 1 1 1 0

private def oldDistance : Op .nat [64, 64] :=
  .add .nat .scalarR
    (.sub .nat .scalarR
      (.add .nat .scalarR
        (.sub .nat (.consR (.consL .nil))
          (.expandDim ⟨1, by simp⟩ (.arange 64))
          (.expandDim ⟨0, by simp⟩ (.arange 64)))
        (.mul .nat .nil (.ref .nat [] "start_m") (.constNat 64)))
      (.ref .nat [] "start_n")) (.constNat 0)

set_option maxRecDepth 8000 in
/-- This is the distance from the full historical kernel, not a detached toy
expression or a guessed statement index. -/
theorem historical_source_distance :
    distance? 128 oldSource.toAlgKernel.body = some oldDistance := by rfl

/-- The actual old expression retains a future key in the first block. -/
theorem historical_distance_at_future_key (s : BlockState)
    (hm : s.regs .nat [] "start_m" = some (Tile.scalar 0))
    (hn : s.regs .nat [] "start_n" = some (Tile.scalar 0)) :
    (evalOp oldDistance s).map (fun t => t.data (0, 1, PUnit.unit)) = some 0 := by
  rw [show evalOp oldDistance s = some (⟨fun idx : TileIndex [64, 64] =>
      ((idx.1.val - idx.2.1.val) + 0 * 64 - 0) + 0⟩ : Tile .nat [64, 64]) from
    aft3_dist_evalG s 0 0 64 64 0 (by simp) (by simp) hm hn]
  rfl

theorem window_mask_counterexample :
    natSlidingWindowKeepG 0 64 64 0 64 (0 : Fin 64) (1 : Fin 128) ∧
    signedDistance 0 64 64 0 (0 : Fin 64) (1 : Fin 128) = -1 ∧
    ¬ signedKeep 0 64 64 0 64 (0 : Fin 64) (1 : Fin 128) := by
  norm_num [natSlidingWindowKeepG, natDist3G, signedDistance, signedKeep]

/-- A universal claim that the old and signed source masks agree is false. -/
refutation historical_window_matches_source :
    ∀ (SM : Nat) (i : Fin 64) (j : Fin 128),
      natSlidingWindowKeepG SM 64 64 0 64 i j ↔ signedKeep SM 64 64 0 64 i j := by
  intro h
  exact window_mask_counterexample.2.2 ((h 0 0 1).mp window_mask_counterexample.1)

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.SourceReview
