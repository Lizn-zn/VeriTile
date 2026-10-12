import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.Tiles
namespace VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.Scan
open VeriTile.Triton Math Tiles
open scoped BigOperators
noncomputable section
private theorem foldl_realAdd_coe (l : List ℝ) (init : ℝ) :
    (l.map (fun r => ((r : ℝ) : WithBot ℝ))).foldl WithBot.realAdd
        ((init : ℝ) : WithBot ℝ)
      = (((l.foldl (· + ·) init : ℝ)) : WithBot ℝ) := by
  induction l generalizing init with
  | nil => simp
  | cons a t ih =>
      simp only [List.map_cons, List.foldl_cons, WithBot.realAdd_coe_coe]
      exact ih (init + a)

/-- A `ScanOp.sum` fold over a list of `some`-valued lanes equals `some` of the
underlying real list-sum. -/
private theorem scan_prefix_eq {α : Type*} (g : α → ℝ) (L : List α) :
    (L.map (fun k => (some (g k) : WithBot ℝ))).foldl WithBot.realAdd
        ((0 : ℝ) : WithBot ℝ)
      = some ((L.map g).sum) := by
  rw [show (L.map (fun k => (some (g k) : WithBot ℝ)))
      = List.map (fun r => ((r : ℝ) : WithBot ℝ)) (L.map g) from by
        rw [List.map_map]; rfl]
  rw [foldl_realAdd_coe]; congr 1; rw [List.sum_eq_foldl]

/-- **2D axis-1 cumsum = within-row prefix sum.** The `Tile.scan .sum` along
axis 1 of a `some`-valued `[H, CH]` tile, evaluated at index `idx`, is
`some (Σ_{k ≤ idx.chunk} g (idx.head, k))` — the prefix sum over the chunk axis
at fixed head. -/
theorem scan2d_axis1_sum (H CH : Nat) (g : Fin H → Fin CH → ℝ)
    (idx : TileIndex [H, CH]) :
    (Tile.scan .sum ⟨1, by simp⟩ .forward
      (⟨fun idx => some (g idx.1 idx.2.1)⟩ : Tile .real [H, CH])).data idx
      = some (∑ k ∈ (Finset.univ.filter
          (fun k : Fin CH => k.val ≤ idx.2.1.val)), g idx.1 k) := by
  rw [Tile.scan_data]
  simp only [ScanOp.eval_sum, TileShape.axisCoord, TileShape.replaceAxisCoord,
    TileShape.axisDim]
  refine Eq.trans (scan_prefix_eq (fun k : Fin CH => g idx.1 k) _) ?_
  rw [Finset.sum_map_toList]; rfl


theorem eval_scan (p : Layout) (u : BlockState) (f : Fin p.BH → Fin p.BC → ℝ)
    (hf : u.regs .real [p.BH,p.BC] "dA" = some (matrix p f)) :
    evalOp (.scan .sum ⟨1,by simp⟩ .forward (.ref .real [p.BH,p.BC] "dA")) u =
      some (matrix p (fun i j => ∑ t : Fin p.BC, if t.val ≤ j.val then f i t else 0)) := by
  simp only [evalOp.eq_def,hf,Option.bind_eq_bind,Option.bind_some,Option.pure_def]
  apply congrArg some
  apply Tile.ext
  rintro ⟨i,j,⟨⟩⟩
  rw [show matrix p f = (⟨fun idx => some (f idx.1 idx.2.1)⟩ : Tile .real [p.BH,p.BC]) from rfl]
  rw [scan2d_axis1_sum]
  simp [matrix,Finset.sum_filter]
end
end VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.Scan
