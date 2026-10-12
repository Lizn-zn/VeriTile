import VeriTile.Triton.Semantics.TileOps
import VeriTile.Triton.Core.Reshape

/-! Real whole-tile reduction identities used to relate faithful source
lowering to independently stated finite-sum formulas. -/
namespace VeriTile.Triton
theorem TileShape.sum_allIndices_two {A : Type*} [AddCommMonoid A]
    (M N : Nat) (f : TileIndex [M,N] → A) :
    ((allIndices [M,N]).map f).sum = ∑ i : Fin M, ∑ j : Fin N, f (i,j,PUnit.unit) := by
  simp [allIndices, List.flatMap_def, List.map_flatten, List.sum_flatten,
    List.map_ofFn, List.sum_ofFn, Function.comp, List.finRange]

private theorem sum_getElem_optional {A B : Type*} [AddCommMonoid B]
    (xs : List A) (f : A → B) :
    (∑ i : Fin xs.length, (xs[i.val]?.map f).getD 0) = (xs.map f).sum := by
  have hs (i : Fin xs.length) : (xs[i.val]?.map f).getD 0 = f xs[i.val] := by simp
  simp only [hs]
  rw [← List.sum_ofFn]
  simpa only [Function.comp_def, List.ofFn_getElem] using
    congrArg List.sum (List.map_ofFn (f := fun i : Fin xs.length => xs[i.val]) (g := f)).symm
/-- Flattening a matrix before its real sum yields the ordinary double sum.
This is a real-semantics theorem, not an FP regrouping assumption. -/
theorem Tile.reduceSum_reshape_two (M N : Nat) (x : Tile .real [M,N]) :
    (Tile.reduceSum (shape := [M*N]) ⟨0, by simp⟩ Bool.false
      (Tile.reshape x)).data PUnit.unit =
    @Finset.sum (Fin M) (WithBot ℝ) _ Finset.univ fun i =>
      @Finset.sum (Fin N) (WithBot ℝ) _ Finset.univ fun j => x.data (i,j,PUnit.unit) := by
  have hlen : (TileShape.allIndices [M,N]).length = M*N := by
    rw [TileShape.length_allIndices]
    simp only [TileShape.product, Nat.mul_one]
  have hsum := sum_getElem_optional (B := WithBot ℝ) (TileShape.allIndices [M,N]) x.data
  rw [TileShape.sum_allIndices_two] at hsum
  rw [hlen] at hsum
  rw [← hsum]
  simp only [Tile.reduceSum_false, Tile.reduceSumDrop_data, Tile.reshape,
    TileShape.axisDim, TileShape.insertAxisIndex, TileShape.linearIndex,
    TileShape.product, Nat.mul_one, Nat.add_zero]
  apply Finset.sum_congr rfl
  intro i _
  cases (TileShape.allIndices [M,N])[i.val]? <;> rfl
end VeriTile.Triton
