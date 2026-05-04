/-
VeriTile.Triton.Concurrency.ProducerConsumer

First producer-consumer final-state refinement capstone for #81.
-/

import VeriTile.Triton.Concurrency.Refinement

namespace VeriTile.Triton

namespace ProducerConsumer

/-- Sequential reference for the Phase-0 one-tile copy consumer. -/
def sequentialTileCopy (inReg outReg : RegionName) (n : Nat)
    (s : BlockState) : Option BlockState :=
  some <| (List.range n).foldl
    (fun acc i => acc.writeMem outReg i (s.readMem inReg i)) s

/-- Abstract concurrent producer-consumer copy.

The first capstone intentionally abstracts away scheduling: async/barrier and
ownership obligations are explicit theorem assumptions, and the final-state
meaning is the same tile copy as the sequential reference. -/
def producerConsumerTileCopy (inReg outReg : RegionName) (n : Nat) :
    BlockState → Option BlockState :=
  sequentialTileCopy inReg outReg n

theorem oneTileProducerConsumer_refines_sequentialCopy
    {Perm : Type} [PermissionModel Perm]
    (trace : EffectTrace) (before after : OwnershipMap Perm)
    (tile : WriteFootprint)
    (inReg outReg : RegionName) (n : Nat)
    (_hAsync : EffectTrace.AsyncVisibility trace)
    (_hBarrier : EffectTrace.BarrierDiscipline trace)
    (_hOwner : EffectTrace.OwnershipTransfer before after tile)
    (_hNoRace : EffectTrace.NoRaceOnOwnedCells after tile) :
    ConcurrentTrace.RefinesSequential
      (producerConsumerTileCopy inReg outReg n)
      (sequentialTileCopy inReg outReg n) := by
  simpa [producerConsumerTileCopy] using
    ConcurrentTrace.refinesSequential_refl (sequentialTileCopy inReg outReg n)

end ProducerConsumer

end VeriTile.Triton

