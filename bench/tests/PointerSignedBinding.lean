import VeriTile.Triton.DSL
import VeriTile.Meta.StatementAudit

/-! A signed address expression must not turn a named pointer into an integer
register. The complete signed offset is converted only at the pointer boundary. -/
namespace PointerSignedBindingTests
open VeriTile.Triton

def kernel (input_ptr : RegionName) (B : Nat) : ComputeKernel := triton {
  tl.region input_ptr = tl.float32
  offset = $((-1 : Int)) + tl.arange(0, $(B))
  input_ptrs = input_ptr + offset
  alias_ptrs = input_ptrs
  inp = (tl.load(alias_ptrs)).to(tl.float32)
}

/-- The inferred signed offset remains signed until the pointer is constructed. -/
theorem named_pointer_keeps_pointer_type (X : RegionName) (B : Nat) :
    (kernel X B).toAlgKernel.body[1]? = some
      (.assign .ptr [B] "input_ptrs" (.ptrAdd .scalarL (.ptrBase X)
        (.castIntToNat (.ref .int [B] "offset")))) := rfl

/-- Signed scalar arithmetic retains its integer register type independently of
pointer inference and of its eventual use as a memory offset. -/
theorem offset_keeps_signed_type (X : RegionName) (B : Nat) :
    (kernel X B).toAlgKernel.body[0]? = some
      (.assign .int [B] "offset" (.add .int .scalarL (.constInt (-1))
        (.castNatToInt (.arange B)))) := rfl

/-- Copying the derived pointer must preserve its pointer register channel too. -/
theorem alias_keeps_pointer_type (X : RegionName) (B : Nat) :
    (kernel X B).toAlgKernel.body[2]? = some
      (.assign .ptr [B] "alias_ptrs" (.ref .ptr [B] "input_ptrs")) := rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end PointerSignedBindingTests
