import bench.tritonbench_g.flash_attn.Proofs.RealLoopInvariant

/-! The final division and log2 expressions equal the independent output/LSE
formulas. All values are read from the initial memory through the prefix sums. -/
namespace VeriTile.Bench.TritonBenchG.FlashAttn.RealExecution
open VeriTile.Triton Formula
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3
noncomputable section

theorem final_output_tile (p : Layout) (s : BlockState) (Q K V : RegionName)
    (hhi : 0 < hi p (s.pids 0)) (hBN : 0 < p.BLOCK_N) :
    let c := Geometry.tiles (hi p (s.pids 0)) p.BLOCK_N
    Tile.bop NumericDType.real.div (.consSame (.consR .nil)) (prefixAcc p s Q K V c)
      (Tile.expandDim ⟨1, by simp⟩ (prefixDen p s Q K V c)) =
        RealTiles.finite (fun i : TileIndex [p.BLOCK_M,p.DIM] =>
          output p s.readMem Q K V (s.pids 0) (s.pids 1) i.1.val i.2.1.val) := by
  dsimp only
  apply Tile.ext
  rintro ⟨i,d,⟨⟩⟩
  have hind := prefixSummary_value_independent p s.readMem Q K V (s.pids 0) (s.pids 1)
    i.val 0 d.val (Geometry.tiles (hi p (s.pids 0)) p.BLOCK_N)
  have hf := prefixSummary_full p s.readMem Q K V (s.pids 0) (s.pids 1) i.val d.val
  have hd := congrArg (fun z : WithBot ℝ × ℝ × ℝ => z.2.1) hf
  have ha := congrArg (fun z : WithBot ℝ × ℝ × ℝ => z.2.2) hf
  dsimp only at hd ha
  change some ((prefixSummary p s.readMem Q K V (s.pids 0) (s.pids 1) i.val d.val _).numerator /
    (prefixSummary p s.readMem Q K V (s.pids 0) (s.pids 1) i.val 0 _).denominator) = some _
  rw [hind.2, hd, ha]
  exact congrArg some (online_output p s.readMem Q K V (s.pids 0) (s.pids 1) i.val d.val hhi hBN)

theorem final_lse_tile (p : Layout) (s : BlockState) (Q K V : RegionName)
    (hhi : 0 < hi p (s.pids 0)) (hBN : 0 < p.BLOCK_N) :
    let c := Geometry.tiles (hi p (s.pids 0)) p.BLOCK_N
    Tile.bop NumericDType.real.add (.consSame .nil) (prefixMax p s Q K V c)
      (Tile.uop WithBot.realLog2 (prefixDen p s Q K V c)) =
        RealTiles.finite (fun i : TileIndex [p.BLOCK_M] =>
          lse p s.readMem Q K (s.pids 0) (s.pids 1) i.1.val) := by
  dsimp only
  apply Tile.ext
  rintro ⟨i,⟨⟩⟩
  let xs := pairs p s.readMem Q K V (s.pids 0) (s.pids 1) i.val 0
  have hf := prefixSummary_full p s.readMem Q K V (s.pids 0) (s.pids 1) i.val 0
  have hm := congrArg (fun z : WithBot ℝ × ℝ × ℝ => z.1) hf
  have hd := congrArg (fun z : WithBot ℝ × ℝ × ℝ => z.2.1) hf
  dsimp only at hm hd
  obtain ⟨m,hfinite⟩ := RealMath.maximum_finite xs
    (pairs_nonempty p s.readMem Q K V (s.pids 0) (s.pids 1) i.val 0 hhi hBN)
  change WithBot.realAdd (prefixSummary p s.readMem Q K V (s.pids 0) (s.pids 1) i.val 0 _).center
    (WithBot.realLog2 (some (prefixSummary p s.readMem Q K V
      (s.pids 0) (s.pids 1) i.val 0 _).denominator)) = some _
  rw [hm, hd, RealMath.running_max]
  change WithBot.realAdd (RealMath.maximum xs) (WithBot.realLog2 (some (RealMath.running xs).2.1)) = _
  rw [hfinite]
  have hout := online_lse p s.readMem Q K V (s.pids 0) (s.pids 1) i.val 0 hhi hBN
  change (RealMath.maximum xs).unbotD 0 + Real.log (RealMath.running xs).2.1 / Real.log 2 = _ at hout
  rw [hfinite] at hout
  simpa only [WithBot.realLog2_some, WithBot.realAdd_coe_coe] using congrArg some hout

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.FlashAttn.RealExecution
