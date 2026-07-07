/-
VeriTile.Triton.EvalHelpers

Small, generic `evalOp` unfolding lemmas (and the `cdiv` ceiling-division helper)
shared by the bench kernel transcriptions. Each is a one-line `simp [evalOp]`
specialization that several kernels re-proved verbatim; collecting them here keeps
the per-kernel files focused on layout/spec work.
-/
import VeriTile.Triton.Semantics

namespace VeriTile.Triton

/-- Ceiling division `⌈a / b⌉`, matching Triton's `tl.cdiv`. -/
def cdiv (a b : Nat) : Nat := (a + b - 1) / b

/-- `evalOp` of `ptrBase region`: the scalar base pointer `(region, 0)`. -/
theorem evalOp_ptrBase (region : RegionName) (s : BlockState) :
    evalOp (.ptrBase region) s = some (Tile.scalar (region.cast, 0)) := by simp [evalOp]

/-- `evalOp` of `ptrAdd`: monadic add of the evaluated pointer and offset tiles. -/
theorem evalOp_ptrAdd {a b shape} (bc : Broadcast a b shape)
    (ptr : Op .ptr a) (off : Op .nat b) (s : BlockState) :
    evalOp (.ptrAdd bc ptr off) s = (do
      let ptrs ← evalOp ptr s; let offs ← evalOp off s;
      some (Tile.ptrAdd bc ptrs offs)) := by simp [evalOp]

/-- `evalOp` of integer `mod`. -/
theorem evalOp_mod {dtype a b shape} (h : IntegralDType dtype)
    (bc : Broadcast a b shape) (x : Op dtype a) (y : Op dtype b) (s : BlockState) :
    evalOp (.mod h bc x y) s = (do
      let vx ← evalOp x s; let vy ← evalOp y s; some (Tile.bop h.mod bc vx vy)) := by
  simp [evalOp]

/-- `evalOp` of integer `floorDiv`. -/
theorem evalOp_floorDiv {dtype a b shape} (h : IntegralDType dtype)
    (bc : Broadcast a b shape) (x : Op dtype a) (y : Op dtype b) (s : BlockState) :
    evalOp (.floorDiv h bc x y) s = (do
      let vx ← evalOp x s; let vy ← evalOp y s; some (Tile.bop h.floorDiv bc vx vy)) := by
  simp [evalOp]

end VeriTile.Triton
