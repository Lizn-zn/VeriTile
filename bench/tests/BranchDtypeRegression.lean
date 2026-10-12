import VeriTile.Triton.DSL
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.BranchDtypeRegression
open Lean VeriTile.Triton

/-- A second common binding used to make the old branch intersection discard
z when only its explicit compute tag differed between the two arms. -/
def joined (label : Region .int) (x y : RegionName) : ComputeKernel := triton {
  tl.region x = tl.float32, y = tl.float32
  c = tl.load(label)
  if c == $((0 : Int)) {
    z = 0.0
    common = 1.0
  } else {
    z = tl.load(x).to(tl.float32)
    common = 2.0
  }
  tl.store(y, (z).to(tl.float32))
}

def reversed (label : Region .int) (x y : RegionName) : ComputeKernel := triton {
  tl.region x = tl.float32, y = tl.float32
  c = tl.load(label)
  if c == $((0 : Int)) {
    z = tl.load(x).to(tl.float32)
    common = 1.0
  } else {
    z = 0.0
    common = 2.0
  }
  tl.store(y, (z).to(tl.float32))
}

run_cmd do
  let scalar := VeriTile.Triton.DSL.SInfo.scalar
  let merge := VeriTile.Triton.DSL.mergeBranchEnv
  let fp := ("z", VeriTile.Triton.DSL.DInfo.real, scalar, some VeriTile.Triton.DSL.DInfo.fp32)
  let plain := ("z", VeriTile.Triton.DSL.DInfo.real, scalar, none)
  let common := ("common", VeriTile.Triton.DSL.DInfo.real, scalar, none)
  for (a,b) in [(fp,plain),(plain,fp)] do
    let joined := merge [] [a,common] [b,common]
    unless joined.any (fun e => e.1=="z" && e.2.2.2==some .fp32) do
      throwError "fp32 branch join lost its live scalar binding"
  let wide := ("z", VeriTile.Triton.DSL.DInfo.real, scalar, some VeriTile.Triton.DSL.DInfo.real)
  let incompatible := merge [] [fp,common] [wide,common]
  if incompatible.any (fun e => e.1=="z") then
    throwError "different explicit compute precisions were silently merged"

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Tests.BranchDtypeRegression
