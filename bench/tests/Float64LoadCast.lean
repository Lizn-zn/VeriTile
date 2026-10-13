import VeriTile.Triton.DSL
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.Float64LoadCast
open VeriTile.Triton

def direct (x : RegionName) : ComputeKernel := triton {
  tl.region x = tl.float64
  value = tl.load(x).to(tl.float64)
}

def parenthesized (x : RegionName) : ComputeKernel := triton {
  tl.region x = tl.float64
  value = (tl.load(x)).to(tl.float64)
}

def narrowDirect (x : RegionName) : ComputeKernel := triton {
  tl.region x = tl.float64
  value = tl.load(x).to(tl.float32)
}

def narrowParenthesized (x : RegionName) : ComputeKernel := triton {
  tl.region x = tl.float64
  value = (tl.load(x)).to(tl.float32)
}

def legacyNarrow (x : RegionName) : ComputeKernel := triton {
  value = tl.load(x).to(tl.float32)
}

def legacyAnnotated (x : Region .real) : ComputeKernel := triton {
  value = tl.load($((x : Region .real))).to(tl.float32)
}

/-- A mathematical region parameter is not an explicit fp64 storage declaration. -/
example (x : RegionName) : (legacyNarrow x).surfaceBody =
    [.assign .real [] "value" (.compute (.load .fp32
      (.region x (.constNat 0)) .none))] := rfl

example (x : Region .real) : (legacyAnnotated x).surfaceBody =
    (legacyNarrow x).surfaceBody := rfl

def narrowViaAlias (x : RegionName) (B : Nat) : ComputeKernel := triton {
  tl.region x = tl.float64
  ptr = x + tl.arange(0, $(B))
  value = tl.load(ptr).to(tl.float32)
}

def narrowOverwrittenAlias (x y : RegionName) (B : Nat) : ComputeKernel := triton {
  tl.region x = tl.float64
  ptr = x + tl.arange(0, $(B))
  ptr = y + tl.arange(0, $(B))
  value = tl.load(ptr).to(tl.float32)
}

/-- Declared fp64 provenance follows an offset pointer alias. -/
example (x : RegionName) (B : Nat) : (narrowViaAlias x B).surfaceBody.getLast? =
    some (.assign .real [B] "value" (.compute (.alg .fp32
      (.castFloat .real .real
        (.withPrecision .real .fp64
          (.load .real (.ptr (.ref .ptr [B] "ptr")) .none)))))) := rfl

/-- Overwriting that alias with a mathematical pointer removes its fp64 provenance. -/
example (x y : RegionName) (B : Nat) :
    (narrowOverwrittenAlias x y B).surfaceBody.getLast? =
      some (.assign .real [B] "value" (.compute (.load .fp32
        (.ptr (.ref .ptr [B] "ptr")) .none))) := rfl

/-- Narrowing a double load invokes the conversion, not the fp32-load backend. -/
example (x : RegionName) :
    (narrowDirect x).surfaceBody = (narrowParenthesized x).surfaceBody := rfl

example (x : RegionName) : (narrowDirect x).surfaceBody =
    [.assign .real [] "value" (.compute (.alg .fp32
      (.castFloat .real .real (.withPrecision .real .fp64
        (.load .real (.region x (.constNat 0)) .none)))))] := rfl

/-- Parentheses cannot erase the requested fp64 computation profile. -/
example (x : RegionName) : (direct x).surfaceBody = (parenthesized x).surfaceBody := rfl

private def hasFP64Profile : ComputeStmt → Bool
  | .assign _ _ _ (.compute (.alg .fp64 _)) => Bool.true
  | _ => Bool.false

/-- The matching syntax forms both retain an explicit fp64 computation profile. -/
example (x : RegionName) :
    (direct x).surfaceBody.length = 1 ∧
      (direct x).surfaceBody.all hasFP64Profile = Bool.true := ⟨rfl, rfl⟩

def maskedDirect (x : RegionName) (N B : Nat) : ComputeKernel := triton {
  tl.region x = tl.float64
  offsets = tl.arange(0, $(B))
  value = tl.load(x + offsets, mask=offsets < $(N), other=0).to(tl.float64)
}

def maskedParenthesized (x : RegionName) (N B : Nat) : ComputeKernel := triton {
  tl.region x = tl.float64
  offsets = tl.arange(0, $(B))
  value = (tl.load(x + offsets, mask=offsets < $(N), other=0)).to(tl.float64)
}

def maskedPositional (x : RegionName) (N B : Nat) : ComputeKernel := triton {
  tl.region x = tl.float64
  offsets = tl.arange(0, $(B))
  value = tl.load(x + offsets, offsets < $(N), other=0).to(tl.float64)
}

/-- Mask spelling and parentheses preserve the offset load, padding and fp64 cast. -/
example (x : RegionName) (N B : Nat) :
    (maskedDirect x N B).surfaceBody = (maskedParenthesized x N B).surfaceBody ∧
      (maskedDirect x N B).surfaceBody = (maskedPositional x N B).surfaceBody ∧
      (maskedDirect x N B).surfaceBody.getLast?.map hasFP64Profile = some Bool.true :=
  ⟨rfl, rfl, rfl⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Tests.Float64LoadCast
