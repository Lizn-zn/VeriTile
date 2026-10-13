import VeriTile.Triton
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.KldivTriton.FullAlgorithm
open VeriTile.Triton

/-- Complete forward control flow for the real projection, with signed loop counters.
Storage precision is supplied separately by the typed-source projection proof. -/
def forwardSurface (Y T O : RegionName) (sy st so N B : Nat)
    (logTarget : Bool) (reduction : Nat) : ComputeKernel :=
  ComputeKernel.mk [Y, T] [O] (triton {
    tl.region Y = tl.float32, T = tl.float32, O = tl.float32
    pid = tl.program_id(0).to(tl.int64)
    Y += pid * $(sy)
    T += pid * $(st)
    O += pid * $(so)
    base_offsets = tl.arange(0, $(B))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + base_offsets
      mask = offsets < $((N : Int))
      y = tl.load(Y + offsets, mask=mask, other=0.0)
      y_true = tl.load(T + offsets, mask=mask, other=0.0)
      if not logTarget {
        loss = y_true * (tl.log(y_true) - y)
      } else {
        loss = tl.exp(y_true) * (y_true - y)
      }
      if $(reduction) == $(0) {
        tl.store(O + offsets, loss, mask=mask)
      } else {
        loss = tl.sum(loss, axis=0)
        tl.store(O, loss)
        O += $(1)
      }
    }
  }).surfaceBody

/-- Complete backward control flow, retaining the source's pre-loop assignments. -/
def backwardSurface (T O : RegionName) (st so N B : Nat)
    (logTarget : Bool) : ComputeKernel :=
  if logTarget then
    ComputeKernel.mk [T] [O] (triton {
      tl.region T = tl.float32, O = tl.float32
      pid = tl.program_id(0).to(tl.int64)
      O += pid * $(so)
      T += pid * $(st)
      offsets = tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      for i in range(0, $((N : Int)), $(B)) {
        offsets = i + tl.arange(0, $(B))
        mask = offsets < $((N : Int))
        target = tl.load(T + offsets, mask=mask, other=0.0)
        res = -tl.exp(target)
        tl.store(O + offsets, res, mask=mask)
      }
    }).surfaceBody
  else
    ComputeKernel.mk [T] [O] (triton {
      tl.region T = tl.float32, O = tl.float32
      pid = tl.program_id(0).to(tl.int64)
      O += pid * $(so)
      T += pid * $(st)
      offsets = tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      for i in range(0, $((N : Int)), $(B)) {
        offsets = i + tl.arange(0, $(B))
        mask = offsets < $((N : Int))
        target = tl.load(T + offsets, mask=mask, other=0.0)
        res = target * -1
        tl.store(O + offsets, res, mask=mask)
      }
    }).surfaceBody

def forward (Y T O : RegionName) (sy st so N B : Nat)
    (logTarget : Bool) (reduction : Nat) : Kernel :=
  (forwardSurface Y T O sy st so N B logTarget reduction).toAlgKernel

def backward (T O : RegionName) (st so N B : Nat) (logTarget : Bool) : Kernel :=
  (backwardSurface T O st so N B logTarget).toAlgKernel

theorem forward_supported (Y T O : RegionName) (sy st so N B : Nat)
    (logTarget : Bool) (reduction : Nat) :
    (forwardSurface Y T O sy st so N B logTarget reduction).toAlgorithm? =
      .ok (forward Y T O sy st so N B logTarget reduction) := by
  rfl

theorem backward_supported (T O : RegionName) (st so N B : Nat) (logTarget : Bool) :
    (backwardSurface T O st so N B logTarget).toAlgorithm? =
      .ok (backward T O st so N B logTarget) := by
  cases logTarget <;> rfl

end VeriTile.Bench.TritonBenchG.KldivTriton.FullAlgorithm

#guard_msgs (drop info) in
#auditModuleAxioms
