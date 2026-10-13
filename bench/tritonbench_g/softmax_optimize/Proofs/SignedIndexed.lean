import bench.tritonbench_g.softmax_optimize.Proofs.SignedState

namespace VeriTile.Bench.TritonBenchG.SoftmaxOptimize.SignedTransport
open VeriTile.Triton

def castIndex {shape : TileShape} (x : Tile .nat shape) : Tile .int shape :=
  ⟨fun i ↦ (x.data i : Int)⟩

/-- After the three address assignments both vectors agree through Int.ofNat.
This relation is local to an iteration; arbitrary initial scratch values need
not already be nonnegative or initialized. -/
structure IndexedRelated (B : Nat) (s t : BlockState) : Prop extends LiveRelated s t where
  columns_eq : s.regs .int [B] "n_offsets" = (t.regs .nat [B] "n_offsets").map castIndex
  offset_eq : s.regs .int [B] "offset" = (t.regs .nat [B] "offset").map castIndex

namespace IndexedRelated

theorem setShared {B : Nat} {s t : BlockState} (h : IndexedRelated B s t)
    (name : RegName) (dtype : TileDType) (shape : TileShape) (value : Tile dtype shape)
    (hname : Shared name) :
    IndexedRelated B (s.setReg name dtype shape value) (t.setReg name dtype shape value) := by
  have hc : "n_offsets" ≠ name := by
    intro he; subst name; exact hname.1 (Or.inr (Or.inl rfl))
  have ho : "offset" ≠ name := by
    intro he; subst name; exact hname.1 (Or.inr (Or.inr rfl))
  exact ⟨h.toLiveRelated.setShared name dtype shape value hname,
    by simpa only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hc] using h.columns_eq,
    by simpa only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ ho] using h.offset_eq⟩

theorem offset_eval {B : Nat} {s t : BlockState} (h : IndexedRelated B s t) :
    evalOp (.castIntToNat (.ref .int [B] "offset")) s =
      evalOp (.ref .nat [B] "offset") t := by
  simp only [evalOp.eq_def, h.offset_eq]
  cases he : t.regs .nat [B] "offset" with
  | none => rfl
  | some v =>
    simp only [Option.map_some, Option.bind_some]
    apply congrArg some
    apply Tile.ext
    intro i
    exact Int.toNat_natCast _

end IndexedRelated

/-- The first helper assignment establishes the cross-dtype relation from a
single arbitrary initial state; stale scratch registers are allowed. -/
theorem seed_previous (s : BlockState) (n : Nat) :
    LiveRelated (s.setReg "prev_multiple" .int [] (Tile.scalar (n : Int)))
      (s.setReg "prev_multiple" .nat [] (Tile.scalar n)) := by
  constructor
  · rfl
  · rfl
  · rfl
  · rfl
  · intro dt sh name hname
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hname.2]
  · simp only [BlockState.setReg_same, Option.map_some, Tile.scalar_data]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxOptimize.SignedTransport
