import bench.tritonbench_g.flash_attn.Proofs.RealPointers
import bench.tritonbench_g.flash_attn.Proofs.RealNumericStep
import bench.tritonbench_g.flash_attn.Proofs.RealSummary
import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealTiles

/-! The actual query/key dot and value tiles, expressed using the initial-memory
addresses in the independent formula. All loads are unmasked, including tails. -/
namespace VeriTile.Bench.TritonBenchG.FlashAttn.RealExecution
open VeriTile.Triton Formula
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3
noncomputable section

def keyTile (p : Layout) (s : BlockState) (K : RegionName) (pid₁ start : Nat) :
    Tile .real [p.DIM, p.BLOCK_N] :=
  RealTiles.finite (fun i => s.readMem K (kOffset p pid₁ (start + i.2.1.val) i.1.val))

def valueTile (p : Layout) (s : BlockState) (V : RegionName) (pid₁ start : Nat) :
    Tile .real [p.BLOCK_N, p.DIM] :=
  RealTiles.finite (fun i => s.readMem V (vOffset p pid₁ (start + i.1.val) i.2.1.val))

theorem keyTile_address (p : Layout) (s : BlockState) (K : RegionName)
    (pid₁ start : Nat) :
    (⟨fun i => s.readMemValue .real K
      (((keyPointer p K pid₁ start).data i).address (TileShape.indexToList _ i))⟩ :
        Tile .real [p.DIM, p.BLOCK_N]) = keyTile p s K pid₁ start := by
  apply Tile.ext
  intro i
  simp only [keyPointer_address, BlockState.readMemValue_real]
  rfl

theorem valueTile_address (p : Layout) (s : BlockState) (V : RegionName)
    (pid₁ start : Nat) :
    (⟨fun i => s.readMemValue .real V
      (((valuePointer p V pid₁ start).data i).address (TileShape.indexToList _ i))⟩ :
        Tile .real [p.BLOCK_N, p.DIM]) = valueTile p s V pid₁ start := by
  apply Tile.ext
  intro i
  simp only [valuePointer_address, BlockState.readMemValue_real]
  rfl

theorem query_key_dot (p : Layout) (s : BlockState) (Q K : RegionName) (start : Nat) :
    Tile.dot [] (query p s Q) (keyTile p s K (s.pids 1) start) =
      RealTiles.finite (fun i : TileIndex [p.BLOCK_M, p.BLOCK_N] =>
        score p s.readMem Q K (s.pids 0) (s.pids 1) i.1.val (start + i.2.1.val)) := by
  apply RealTiles.dot_finite

theorem loaded_keyTile (p : Layout) (s : BlockState) (K : RegionName) (pid₁ start : Nat) :
    loaded s (keyPointer p K pid₁ start) = keyTile p s K pid₁ start := by
  apply keyTile_address

theorem loaded_valueTile (p : Layout) (s : BlockState) (V : RegionName) (pid₁ start : Nat) :
    loaded s (valuePointer p V pid₁ start) = valueTile p s V pid₁ start := by
  apply valueTile_address

theorem source_scores (p : Layout) (s : BlockState) (Q K : RegionName) (start : Nat) :
    scores p s start (Tile.vec (fun i => row p (s.pids 0) i.val))
      (Tile.vec (fun j : Fin p.BLOCK_N => j.val)) (query p s Q)
      (keyPointer p K (s.pids 1) start) =
        (⟨fun i => if visible p (s.pids 0) i.1.val (start + i.2.1.val) then
          (score p s.readMem Q K (s.pids 0) (s.pids 1) i.1.val (start + i.2.1.val) : WithBot ℝ)
          else ⊥⟩ : Tile .real [p.BLOCK_M,p.BLOCK_N]) := by
  simp only [scores]
  apply Tile.ext
  intro i
  dsimp only
  erw [loaded_keyTile, query_key_dot]
  by_cases h : start + i.2.1.val ≤ row p (s.pids 0) i.1.val
  · cases hc : p.causal <;> simp [visible, hc, h, Tile.vec, RealTiles.finite, NumericDType.add] <;> rfl
  · cases hc : p.causal <;> simp [visible, hc, h, Tile.vec, RealTiles.finite, NumericDType.add] <;> rfl

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.FlashAttn.RealExecution
