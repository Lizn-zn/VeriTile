import bench.tritonbench_g.rotary_emb.Proofs.Parameters

namespace VeriTile.Bench.TritonBenchG.RotaryEmb.FullMath
open VeriTile.Triton

def Parameters.dataOffsets (p : Parameters) (s : BlockState)
    (sb sh sd : Nat) (odd : Bool) : Tile .nat [p.blockSeq,p.blockHead,p.dimension/2] :=
  ⟨address s sb sh sd p.blockSeq p.blockHead p.dimension odd⟩

def Parameters.trigOffsets (p : Parameters) (s : BlockState) (odd : Bool) :
    Tile .nat [p.blockSeq,1,p.dimension/2] :=
  ⟨FullMath.trigAddress s p.cosBatch p.cosDim p.blockSeq 1 p.dimension odd⟩

noncomputable def Parameters.dataValues (p : Parameters) (s : BlockState)
    (X : RegionName) (sb sh sd H : Nat) (odd : Bool) :
    Tile .real [p.blockSeq,p.blockHead,p.dimension/2] := by
  classical
  exact ⟨fun i ↦ some (if active s p.length H p.blockSeq p.blockHead p.dimension i then
    s.readMem X (address s sb sh sd p.blockSeq p.blockHead p.dimension odd i) else 0)⟩

noncomputable def Parameters.trigValues (p : Parameters) (s : BlockState)
    (X : RegionName) (odd : Bool) : Tile .real [p.blockSeq,1,p.dimension/2] :=
  ⟨fun i ↦ some (if sequence s p.blockSeq i < p.length then
    s.readMem X (FullMath.trigAddress s p.cosBatch p.cosDim p.blockSeq 1 p.dimension odd i)
    else 0)⟩

noncomputable def Parameters.resultValues (p : Parameters) (s : BlockState)
    (X Cos Sin : RegionName) (sb sh sd H : Nat) (odd : Bool) :
    Tile .real [p.blockSeq,p.blockHead,p.dimension/2] := by
  classical
  exact ⟨fun i ↦ some (if active s p.length H p.blockSeq p.blockHead p.dimension i then
    expected s X Cos Sin sb sh sd p.cosBatch p.cosDim p.blockSeq p.blockHead p.dimension odd i
    else 0)⟩

/-- The broadcasted trigonometric row is independent of the data head lane. -/
def Parameters.toTrigLane (p : Parameters) (i : p.Lane) : p.TrigLane :=
  (i.1,⟨0,by omega⟩,i.2.2)

theorem trigAddress_broadcast (p : Parameters) (s : BlockState) (odd : Bool) (i : p.Lane) :
    FullMath.trigAddress s p.cosBatch p.cosDim p.blockSeq 1 p.dimension odd (p.toTrigLane i) =
      p.trigAddress s odd i := rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RotaryEmb.FullMath
