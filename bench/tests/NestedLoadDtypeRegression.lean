import VeriTile.Triton.DSL
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.NestedLoadDtypeRegression
open VeriTile.Triton
set_option maxHeartbeats 2000000
set_option maxRecDepth 6000

/-- The storage and cast must survive being nested under arithmetic, including
an advanced pointer and a positional mask. -/
def nested (x y : RegionName) : ComputeKernel := triton {
  tl.region x = tl.float16, y = tl.float32
  p = x + $(3)
  v = tl.load(p, true, other=0.0).to(tl.float32) * 2.0
  tl.store(y, v)
}

def nestedBF16 (x y : RegionName) : ComputeKernel := triton {
  tl.region x = tl.bfloat16, y = tl.float32
  v = tl.exp((tl.load(x)).to(tl.float32)) + 2.0
  tl.store(y, v)
}

def nestedUntyped (x y : RegionName) : ComputeKernel := triton {
  v = tl.load(x).to(tl.float32) * 2.0
  tl.store(y, v)
}

/-- Each nested load uses the current pointer binding's element type. -/
def rebound (x y out : RegionName) : ComputeKernel := triton {
  tl.region x = tl.float16, y = tl.bfloat16, out = tl.float32
  p = x + $(1)
  first = tl.load(p).to(tl.float32) * 2.0
  p = y + $(2)
  second = tl.load(p).to(tl.float32) * 3.0
  tl.store(out, first + second)
}

/-- The internal hint never replaces an explicit dtype, and applies equally
when the nested load is an update operand. -/
def explicitStorage (x y : RegionName) : ComputeKernel := triton {
  tl.region x = tl.bfloat16, y = tl.float32
  v = tl.load(x, dtype=tl.float16).to(tl.float32) * 2.0
  tl.store(y, v)
}

def nestedUpdate (x y : RegionName) : ComputeKernel := triton {
  tl.region x = tl.float16, y = tl.float32
  v = 0.0
  v += tl.load(x).to(tl.float32) * 2.0
  tl.store(y, v)
}

def loadedType? {d : TileDType} {sh : TileShape} : Op d sh → Option TileDType
  | .load d _ _ _ => some d
  | .castFloat _ _ a => loadedType? a
  | .mul _ _ a _ => loadedType? a
  | .add _ _ a _ => loadedType? a
  | .exp a => loadedType? a
  | _ => none

def assignedType? (code : List ComputeStmt) (name : RegName := "v") : Option TileDType :=
  match code with
  | [] => none
  | .assign _ _ dest e :: rest =>
    if dest=name then e.toAlgorithm?.toOption.bind loadedType? else assignedType? rest name
  | _ :: rest => assignedType? rest name

theorem fp16_nested_load_retained : assignedType? (nested "x" "y").surfaceBody = some .fp16 := by
  simp [assignedType?,loadedType?,nested,ComputeKernel.surfaceBody,
    ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,Except.toOption,Option.bind]

theorem bf16_nested_load_retained : assignedType? (nestedBF16 "x" "y").surfaceBody = some .bf16 := by
  simp [assignedType?,loadedType?,nestedBF16,ComputeKernel.surfaceBody,
    ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,Except.toOption,Option.bind]

theorem unknown_storage_retains_real_default : assignedType? (nestedUntyped "x" "y").surfaceBody = some .real := by
  simp [assignedType?,loadedType?,nestedUntyped,ComputeKernel.surfaceBody,
    ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,Except.toOption,Option.bind]

theorem rebound_storage_tracks_current_pointer :
    assignedType? (rebound "x" "y" "out").surfaceBody "first" = some .fp16 ∧
    assignedType? (rebound "x" "y" "out").surfaceBody "second" = some .bf16 := by
  simp [assignedType?,loadedType?,rebound,ComputeKernel.surfaceBody,
    ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,Except.toOption,Option.bind]

theorem explicit_storage_is_preserved :
    assignedType? (explicitStorage "x" "y").surfaceBody = some .fp16 := by
  simp [assignedType?,loadedType?,explicitStorage,ComputeKernel.surfaceBody,
    ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,Except.toOption,Option.bind]

def updateType? {d : TileDType} {sh : TileShape} : Op d sh → Option TileDType
  | .add _ _ _ b => loadedType? b
  | _ => none

def secondAssignmentType? (code : List ComputeStmt) : Option TileDType :=
  match code with
  | .assign _ _ _ _ :: .assign _ _ _ e :: _ => e.toAlgorithm?.toOption.bind updateType?
  | _ => none

theorem nested_update_storage_retained :
    secondAssignmentType? (nestedUpdate "x" "y").surfaceBody = some .fp16 := by
  simp [secondAssignmentType?,updateType?,loadedType?,nestedUpdate,ComputeKernel.surfaceBody,
    ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,Except.toOption,Option.bind]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Tests.NestedLoadDtypeRegression
