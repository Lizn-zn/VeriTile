import VeriTile.Triton.DSL
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.SignedRangeAddressInference
open VeriTile.Triton

def explicitBounds (X : RegionName) (B : Nat) : ComputeKernel := triton {
  for i in range($((-2 : Int)), $((3 : Int)), $(2)) {
    offsets = i + tl.arange(0, $(B))
    tl.store(X + offsets, 0.0)
  }
}

def parenthesizedBounds (X : RegionName) (B : Nat) : ComputeKernel := triton {
  for i in range(($((-2 : Int))), $((3 : Int)), $(2)) {
    offsets = i + tl.arange(0, $(B))
    tl.store(X + offsets, 0.0)
  }
}

/-- Parentheses do not change signed index propagation into stored addresses. -/
example (X : RegionName) (B : Nat) :
    (explicitBounds X B).surfaceBody = (parenthesizedBounds X B).surfaceBody := rfl

def zeroStart (X : RegionName) (N : Int) (B : Nat) : ComputeKernel := triton {
  for i in range(0, $((N : Int)), $(B)) {
    offsets = i + tl.arange(0, $(B))
    tl.store(X + offsets, 0.0)
  }
}

def parenthesizedZero (X : RegionName) (N : Int) (B : Nat) : ComputeKernel := triton {
  for i in range((0), $((N : Int)), $(B)) {
    offsets = i + tl.arange(0, $(B))
    tl.store(X + offsets, 0.0)
  }
}

/-- The literal-zero fast path preserves the same complete signed loop AST. -/
example (X : RegionName) (N : Int) (B : Nat) :
    (zeroStart X N B).surfaceBody = (parenthesizedZero X N B).surfaceBody := rfl

private def signedAddressBody : ComputeStmt → Bool
  | .forRangeInt _ _ _ _ (.assign .int _ "offsets" _ :: _) => Bool.true
  | _ => Bool.false

/-- Address intermediates stay signed inside both source spellings. -/
example (X : RegionName) (N : Int) (B : Nat) :
    (explicitBounds X B).surfaceBody.head?.map signedAddressBody = some Bool.true ∧
    (zeroStart X N B).surfaceBody.head?.map signedAddressBody = some Bool.true := ⟨rfl, rfl⟩

def naturalBounds (X : RegionName) (N B : Nat) : ComputeKernel := triton {
  for i in range(0, $(N), $(B)) {
    offsets = i + tl.arange(0, $(B))
    tl.store(X + offsets, 0.0)
  }
}

private def naturalAddressBody : ComputeStmt → Bool
  | .forRange _ _ _ _ (.assign .nat _ "offsets" _ :: _) => Bool.true
  | _ => Bool.false

/-- Natural bounds retain their natural loop and address channels. -/
example (X : RegionName) (N B : Nat) :
    (naturalBounds X N B).surfaceBody.head?.map naturalAddressBody = some Bool.true := rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Tests.SignedRangeAddressInference
