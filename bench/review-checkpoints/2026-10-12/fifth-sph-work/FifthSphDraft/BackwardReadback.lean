import FifthSphDraft.BackwardReadback0
import FifthSphDraft.BackwardReadback1
import FifthSphDraft.BackwardReadback2
namespace VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.BackwardReadback
open VeriTile.Triton BackwardBase
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
theorem all_components (X dX g : RegionName) (B N M off stride : Nat) (u v : BlockState)
    (he : exec (source X dX g B N M off stride) u=some v) (j : Fin B)
    (hcoords : ∀ d : Fin 3,coord u.pid B j.val d.val<N)
    (hg : ∀ d : Fin 11,output u.pid B stride off j.val d.val<M) (c : Fin 3) :
    v.readMem dX (coord u.pid B j.val c.val)=
      RealMath.gradient c (u.readMem X (coord u.pid B j.val 0))
        (u.readMem X (coord u.pid B j.val 1)) (u.readMem X (coord u.pid B j.val 2))
        (u.readMem dX (coord u.pid B j.val c.val))
        (fun d => u.readMem g (output u.pid B stride off j.val d.val)) := by
  fin_cases c
  · exact component0 X dX g B N M off stride u v he j hcoords hg
  · exact component1 X dX g B N M off stride u v he j hcoords hg
  · exact component2 X dX g B N M off stride u v he j hcoords hg


#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.BackwardReadback
