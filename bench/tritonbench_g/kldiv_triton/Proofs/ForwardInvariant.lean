import bench.tritonbench_g.kldiv_triton.Proofs.ForwardContext

namespace VeriTile.Bench.TritonBenchG.KldivTriton.FullForward
open VeriTile.Triton

/-- Independent scalar output formulas for both actual source store paths. -/
structure Inv (initial : BlockState) (Y T O : RegionName)
    (yBase tBase base N B c reduction : Nat) (flag : Bool) (s : BlockState) : Prop where
  context : Context initial Y T O yBase tBase base N B c reduction s
  elements : reduction = 0 → ∀ t lane, t < c → lane < B → t * B + lane < N →
    s.readMem O (base + (t * B + lane)) =
      FullMath.loss flag (initial.readMem Y (yBase + (t * B + lane)))
        (initial.readMem T (tBase + (t * B + lane)))
  chunks : reduction ≠ 0 → ∀ t, t < c → s.readMem O (base + t) =
    FullMath.chunkLoss flag (fun k ↦ initial.readMem Y (yBase + k))
      (fun k ↦ initial.readMem T (tBase + k)) N B t

theorem Inv.setIndex {initial s : BlockState} {Y T O : RegionName}
    {yBase tBase base N B c reduction : Nat} {flag : Bool}
    (h : Inv initial Y T O yBase tBase base N B c reduction flag s) (index : Int) :
    Inv initial Y T O yBase tBase base N B c reduction flag
      (s.setReg "i" .int [] (Tile.scalar index)) := by
  exact ⟨h.context.setIndex index, h.elements, h.chunks⟩

/-- One complete chunk extends the independent output formulas and preserves the full frame. -/
theorem invariant_step (initial s : BlockState) (Y T O : RegionName)
    (sy st so yBase tBase base N B c reduction : Nat) (flag : Bool)
    (hB : 0 < B) (hc : c * B < N) (hYO : Y ≠ O) (hTO : T ≠ O)
    (h : Inv initial Y T O yBase tBase base N B c reduction flag s) :
    ∃ next, stepStmts (body Y T O sy st so N B flag reduction)
        (s.setReg "i" .int [] (Tile.scalar (c * B : Nat))) = some next ∧
      Inv initial Y T O yBase tBase base N B (c + 1) reduction flag next := by
  let indexed := s.setReg "i" .int [] (Tile.scalar (c * B : Nat))
  have hi : Inv initial Y T O yBase tBase base N B c reduction flag indexed := h.setIndex _
  let next := resultState (B := B) indexed Y T O yBase tBase (outputBase base c reduction)
    N (c * B : Nat) flag reduction
  have hf := result_frame indexed Y T O yBase tBase (outputBase base c reduction)
    N B (c * B) flag reduction
  refine ⟨next, step indexed Y T O sy st so N B yBase tBase (outputBase base c reduction)
    (c * B : Nat) flag reduction (by simp [indexed]) hi.context.yReg hi.context.tReg
    hi.context.oReg hi.context.baseReg, ?_⟩
  refine ⟨context_step initial indexed Y T O yBase tBase base N B c reduction flag
    hB hc hYO hTO hi.context, ?_, ?_⟩
  · intro hr t lane ht hl hn
    by_cases he : t = c
    · subst t
      let j : TileIndex [B] := (⟨lane, hl⟩, PUnit.unit)
      have hv := element_value indexed Y T O yBase tBase base N B (c * B) flag j hn
      have hv' : next.readMem O (base + (c * B + lane)) =
          FullMath.loss flag (indexed.readMem Y (yBase + (c * B + lane)))
            (indexed.readMem T (tBase + (c * B + lane))) := by
        simpa only [next, hr, outputBase, if_true, j, Nat.add_assoc] using hv
      rw [hv']
      simp only [BlockState.readMem, hi.context.inputY, hi.context.inputT]
    · have htc : t < c := by omega
      have hmul := Nat.mul_le_mul_right B (show t + 1 ≤ c by omega)
      simp only [Nat.add_mul, Nat.one_mul] at hmul
      have hcell := hf O (base + (t * B + lane)) (by
        simp only [blockFootprint, outputBase, hr, if_true]
        rintro ⟨_, other, _, _, he⟩
        omega)
      have hrm : next.readMem O (base + (t * B + lane)) =
          indexed.readMem O (base + (t * B + lane)) := by
        simp only [next, BlockState.readMem, ← hcell]
      exact hrm.trans (hi.elements hr t lane htc hl hn)
  · intro hr t ht
    by_cases he : t = c
    · subst t
      have hv := reduced_value indexed Y T O yBase tBase
        (outputBase base c reduction) N B c reduction flag hr
      have hv' : next.readMem O (base + c) =
          FullMath.chunkLoss flag (fun k ↦ indexed.readMem Y (yBase + k))
            (fun k ↦ indexed.readMem T (tBase + k)) N B c := by
        simpa only [next, outputBase, hr, if_false] using hv
      rw [hv']
      simp only [BlockState.readMem, hi.context.inputY, hi.context.inputT]
    · have htc : t < c := by omega
      have hcell := hf O (base + t) (by
        simp only [blockFootprint, outputBase, hr, if_false]
        omega)
      have hrm : next.readMem O (base + t) = indexed.readMem O (base + t) := by
        simp only [next, BlockState.readMem, ← hcell]
      exact hrm.trans (hi.chunks hr t htc)

end VeriTile.Bench.TritonBenchG.KldivTriton.FullForward
#guard_msgs (drop info) in
#auditModuleAxioms
