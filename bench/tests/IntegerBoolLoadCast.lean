import VeriTile.Triton.DSL
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.IntegerBoolLoadCast
open VeriTile.Triton

def source (mask : RegionName) : ComputeKernel := triton {
  tl.region mask = tl.int64
  flag = tl.load(mask).to(tl.int1)
}

def offsetSource (mask : RegionName) (N B : Nat) : ComputeKernel := triton {
  tl.region mask = tl.int32
  offsets = tl.program_id(0) * $(B) + tl.arange(0, $(B))
  active = offsets < $(N)
  flag = (tl.load(mask + offsets, mask=active)).to(tl.int1)
}

def positionalSource (mask : RegionName) (N B : Nat) : ComputeKernel := triton {
  tl.region mask = tl.int32
  offsets = tl.program_id(0) * $(B) + tl.arange(0, $(B))
  active = offsets < $(N)
  flag = tl.load(mask + offsets, active).to(tl.int1)
}

def legacyBoolSource (mask : RegionName) : ComputeKernel := triton {
  flag = tl.load(mask).to(tl.int1)
}

private def readsIntegerBeforeTruthiness : ComputeStmt → Bool
  | .assign .bool _ _ (.alg (.ne .int _ (.load .int _ _ _) _)) => Bool.true
  | _ => Bool.false

private def comparesIntegerWithZero : ComputeStmt → Bool
  | .assign .bool _ _ (.alg (.ne .int _ (.load .int _ _ _) (.constInt 0))) => Bool.true
  | _ => Bool.false

/-- Casting the loaded value does not change the storage dtype being read. -/
example (mask : RegionName) :
    (source mask).surfaceBody.length = 1 ∧
      (source mask).surfaceBody.all comparesIntegerWithZero = Bool.true := by
  exact ⟨rfl, rfl⟩

/-- Offset arithmetic and a keyword mask retain the signed storage read. -/
example (mask : RegionName) (N B : Nat) :
    (offsetSource mask N B).surfaceBody.getLast?.map readsIntegerBeforeTruthiness =
      some Bool.true := by
  rfl

/-- Positional and keyword mask syntax denote the same load and conversion. -/
example (mask : RegionName) (N B : Nat) :
    (offsetSource mask N B).surfaceBody = (positionalSource mask N B).surfaceBody := by
  rfl

private def readsBool : ComputeStmt → Bool
  | .assign .bool _ _ (.alg (.load .bool _ _ _)) => Bool.true
  | _ => Bool.false

/-- Existing untyped bool-buffer specialization keeps its historical AST. -/
example (mask : RegionName) :
    (legacyBoolSource mask).surfaceBody.length = 1 ∧
      (legacyBoolSource mask).surfaceBody.all readsBool = Bool.true := by
  exact ⟨rfl, rfl⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Tests.IntegerBoolLoadCast
