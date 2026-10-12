import VeriTile.Triton.Core.Shape
import Mathlib.Data.Finset.Lattice.Fold
import Mathlib.Data.Fintype.Fin

/-! Exact discrete reductions shared by real and FP execution. Numerical
floating algebras supply no assumption or operation for these control values. -/
namespace VeriTile.Triton.TileShape
noncomputable section

/-- Exact maximum for discrete carriers. A zero-sized reduced axis fails;
no zero initializer is inserted for a nonempty signed tile. -/
def discreteMaximum {β : Type} [SemilatticeSup β] {shape : TileShape}
    (axis : Fin shape.length) : (keepDims : Bool) → (TileIndex shape → β) →
      Option (TileIndex (reduceShape shape axis keepDims) → β)
  | Bool.false, x =>
    if h : 0 < axisDim shape axis then
      some (fun out => (Finset.univ : Finset (Fin (axisDim shape axis))).sup'
        ⟨⟨0,h⟩, Finset.mem_univ _⟩ (fun k => x (insertAxisIndex shape axis out k)))
    else none
  | Bool.true, x =>
    if h : 0 < axisDim shape axis then
      some (fun out => (Finset.univ : Finset (Fin (axisDim shape axis))).sup'
        ⟨⟨0,h⟩, Finset.mem_univ _⟩ (fun k => x (replaceAxisIndex shape axis out k)))
    else none

end
end VeriTile.Triton.TileShape
