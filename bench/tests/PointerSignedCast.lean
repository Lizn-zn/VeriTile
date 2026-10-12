import VeriTile.Triton.DSL
import VeriTile.Triton.Source
import VeriTile.Meta.StatementAudit

/-! A casted load from a declared signed region must retain that storage
type through derived pointers, CSR bounds and downstream address arithmetic.
The source casts here use the existing mathematical Int channel. -/
namespace PointerSignedCastTests
open VeriTile.Triton

def kernel (rows cols out : Region .int) : ComputeKernel := triton {
  tl.region rows = tl.int32, cols = tl.int32, out = tl.int32
  ptr = rows + $(3)
  lo = (tl.load(ptr)).to(tl.int32)
  hi = (tl.load(ptr + $(1))).to(tl.int64)
  total = $((0 : Int))
  for i in range(lo, hi) {
    col = (tl.load(cols + i)).to(tl.int32)
    shifted = col * $(2)
    total = total + shifted
  }
  tl.store(out, total)
}

private def assignment (body : List ComputeStmt) (name : RegName) : Option ComputeStmt :=
  (body.drop (Source.prefixBefore body name).length).head?

theorem signed_load_through_pointer (rows cols out : Region .int) :
    assignment (kernel rows cols out).surfaceBody "lo" = some
      (.assign .int [] "lo" (.alg
        (.load .int (.ptr (.ref .ptr [] "ptr")) .none))) := rfl

theorem signed_bounds (rows cols out : Region .int) :
    (Source.intRange? (kernel rows cols out).surfaceBody "i").map
      (fun loop => (loop.start, loop.stop, loop.stride)) =
        some (.ref .int [] "lo", .ref .int [] "hi", .constNat 1) := rfl

theorem signed_product_used_by_address (rows cols out : Region .int) :
    ((Source.intRange? (kernel rows cols out).surfaceBody "i").map (·.body)).bind
      (fun body => assignment body "shifted") = some
        (.assign .int [] "shifted" (.alg (.mul .int .nil
          (.ref .int [] "col") (.castNatToInt (.constNat 2))))) := rfl

theorem signed_load_is_not_natural_range (rows cols out : Region .int) :
    Source.dynamicRange? (kernel rows cols out).surfaceBody "i" = none := rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end PointerSignedCastTests
