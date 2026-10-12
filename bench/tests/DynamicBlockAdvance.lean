import VeriTile.Triton.DSL
import VeriTile.Triton.Float.Erasure
import VeriTile.Meta.StatementAudit

/-! The sliding-window helper passes a runtime coordinate to tl.advance.
Check the actual frontend lowering, including signed offsets, mixed constant
coordinates, tuple/list syntax and failure to silently erase the coordinate. -/
namespace DynamicBlockAdvanceTests
open VeriTile.Triton

def tupleSource (X : RegionName) (shift : Int) : ComputeKernel := triton {
  p = tl.make_block_ptr(base=X, shape=(8, 8), strides=(8, 1),
    offsets=(0, 0), block_shape=(2, 2), order=(1, 0))
  delta = $((shift : Int))
  p = tl.advance(p, (0, delta))
}

def listSource (X : RegionName) (shift : Int) : ComputeKernel := triton {
  p = tl.make_block_ptr(base=X, shape=(8, 8), strides=(8, 1),
    offsets=(0, 0), block_shape=(2, 2), order=(1, 0))
  delta = $((shift : Int))
  p = tl.advance(p, [0, delta])
}

theorem tuple_list_agree (X : RegionName) (shift : Int) : tupleSource X shift = listSource X shift := rfl

theorem keeps_dynamic_signed_coordinate (X : RegionName) (shift : Int) :
    (tupleSource X shift).surfaceBody.getLast? = some
      (.assign .blockPtr [2, 2] "p" (.alg
        (.advanceBlockPtrDyn (.ref .blockPtr [2, 2] "p")
          [.constInt 0, .ref .int [] "delta"]))) := rfl

/-- Floating erasure leaves integer coordinates and their register reads intact. -/
theorem erasure_keeps_dynamic_advance :
    (Op.advanceBlockPtrDyn (.ref .blockPtr [2, 2] "p")
      [.constInt 0, .ref .int [] "delta"]).eraseDType =
    .advanceBlockPtrDyn (.ref .blockPtr [2, 2] "p")
      [.constInt 0, .ref .int [] "delta"] := by
  simp [Op.eraseDType, VeriTile.Triton.eraseDType]

#guard_msgs (drop info) in
#auditModuleAxioms
end DynamicBlockAdvanceTests
