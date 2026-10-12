import VeriTile.Triton.DSL
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.SignedLoadCast
open VeriTile.Triton

def direct (labels : Region .int) (x y : RegionName) : ComputeKernel := triton {
  tl.region labels = tl.int32, x = tl.float32, y = tl.float32
  labels += tl.program_id(0)
  label = tl.load(labels).to(tl.int32)
  if label != $((-100 : Int)) {
    v = tl.load(x + label)
    tl.store(y, v)
  }
}
def parenthesized (labels : Region .int) (x y : RegionName) : ComputeKernel := triton {
  tl.region labels = tl.int32, x = tl.float32, y = tl.float32
  labels += tl.program_id(0)
  label = (tl.load(labels)).to(tl.int32)
  if label != $((-100 : Int)) {
    v = tl.load(x + label)
    tl.store(y, v)
  }
}
/-- Direct-call syntax retains the same signed label and live negative
sentinel as the parenthesized spelling, even when used as a gather address. -/
example (labels : Region .int) (x y : RegionName) : direct labels x y = parenthesized labels x y := rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Tests.SignedLoadCast
