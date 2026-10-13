import bench.tritonbench_g.softmax_flaggems.Proofs.RealBackwardInnerStore

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardInner
open VeriTile Triton

theorem tile_active_injective {B C M N rbase start : Nat} {i j : TileIndex [C,B]}
    (hi : tileActive M N rbase start i = Bool.true)
    (hj : tileActive M N rbase start j = Bool.true)
    (he : tileOffset N rbase start i = tileOffset N rbase start j) : i=j := by
  have hi' : rbase+i.1.val<M ∧ start+i.2.1.val<N := of_decide_eq_true hi
  have hj' : rbase+j.1.val<M ∧ start+j.2.1.val<N := of_decide_eq_true hj
  obtain ⟨hr,hn⟩ := RealAddress.row_column_injective hi'.2 hj'.2 he
  apply Prod.ext
  · exact Fin.ext (Nat.add_left_cancel hr)
  · apply Prod.ext
    · exact Fin.ext (Nat.add_left_cancel hn)
    · exact Subsingleton.elim _ _

theorem tile_access_bound {B C M N rbase start : Nat} (i : TileIndex [C,B])
    (hi : tileActive M N rbase start i = Bool.true) : tileOffset N rbase start i < M*N := by
  have hi' : rbase+i.1.val<M ∧ start+i.2.1.val<N := of_decide_eq_true hi
  simpa [tileOffset, offset] using RealAddress.non_inner_bound hi'.1 (show 0<1 by decide) hi'.2

theorem scatter_value {B C : Nat} (s : BlockState) (Y : RegionName)
    (M N rbase start : Nat) (value : TileIndex [C,B] → ℝ)
    (i : TileIndex [C,B]) (hi : tileActive M N rbase start i = Bool.true) :
    (scatter s Y (tileOffset N rbase start) value (tileActive M N rbase start)).readMem Y
      (tileOffset N rbase start i) = value i := by
  apply BlockState.scatter_readback_prop_masked_nd_of_true s
    (tileOffset N rbase start) value (fun i ↦ tileActive M N rbase start i = Bool.true) i hi
  intro j hj he
  exact tile_active_injective hj hi he

theorem scatter_preserve_processed {B C : Nat} (s : BlockState) (Y : RegionName)
    (M N rbase c r n : Nat) (value : TileIndex [C,B] → ℝ)
    (hB : 0<B) (hold : n/B<c) (hn : n<N) :
    (scatter s Y (tileOffset N rbase (c*B)) value (tileActive M N rbase (c*B))).mem Y
      (offset N r n) = s.mem Y (offset N r n) := by
  apply foldl_store_preserve_cell _ value (fun i ↦ tileActive M N rbase (c*B) i = Bool.true)
  intro i _ hi he
  have hi' : rbase+i.1.val<M ∧ c*B+i.2.1.val<N := of_decide_eq_true hi
  have heq := (RealAddress.row_column_injective hi'.2 hn he.2).2
  have hlt := (Nat.div_lt_iff_lt_mul hB).mp hold
  omega

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardInner
