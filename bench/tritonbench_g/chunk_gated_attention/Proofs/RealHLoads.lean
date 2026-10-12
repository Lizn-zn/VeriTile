import bench.tritonbench_g.chunk_gated_attention.Proofs.RealHTiles

/-! The source loads produce the independently indexed tiles, with each
pointer's exact boundary checks. Inactive lanes retain arbitrary oracle values.
In particular, endpoints are checked against the flattened parent. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGatedAttention.RealHLoads
open VeriTile.Triton RealHMath RealHProgram RealHPointers RealHTiles
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 14000
set_option linter.unusedSimpArgs false

theorem key_load (p : Layout) (l : Launch) (r : RegionName) (a padding : Nat → ℝ)
    (c : Nat) (name : RegName) (u : BlockState)
    (hm : ∀ off, u.readMem r off = a off) (hu : u.undef r = padding)
    (hp : u.regs .blockPtr [p.BK,p.BT] name = some (keyPointer p l r c)) :
    evalOp (load2 p.BK p.BT name) u = some (Tile.mat (fun e t => some (keyData p l a padding c e t))) := by
  rw [load2_eval r u name (l.head*p.keyHead) p.K p.T p.BK p.BT p.keyFeature p.keyTime (l.key*p.BK) (c*p.BT) hp]
  apply congrArg some
  apply Tile.ext
  rintro ⟨e,t,⟨⟩⟩
  simp only [Tile.mat,keyData,keyAddress,keyIndex,timeIndex,hm,hu]
  split <;> rfl

theorem value_load (p : Layout) (l : Launch) (r : RegionName) (a padding : Nat → ℝ)
    (c : Nat) (name : RegName) (u : BlockState)
    (hm : ∀ off, u.readMem r off = a off) (hu : u.undef r = padding)
    (hp : u.regs .blockPtr [p.BT,p.BV] name = some (valuePointer p l r c)) :
    evalOp (load2 p.BT p.BV name) u = some (Tile.mat (fun t j => some (valueData p l a padding c t j))) := by
  rw [load2_eval r u name (l.head*p.valueHead) p.T p.V p.BT p.BV p.valueTime p.valueFeature (c*p.BT) (l.value*p.BV) hp]
  apply congrArg some
  apply Tile.ext
  rintro ⟨t,j,⟨⟩⟩
  simp only [Tile.mat,valueData,valueAddress,valueIndex,timeIndex,hm,hu]
  split <;> rfl

theorem endpoint_load (p : Layout) (l : Launch) (r : RegionName) (a padding : Nat → ℝ)
    (c head stride size block start : Nat) (name : RegName) (u : BlockState)
    (hm : ∀ off, u.readMem r off = a off) (hu : u.undef r = padding)
    (hp : u.regs .blockPtr [block] name = some (endpointPointer p l r c head stride size block start)) :
    evalOp (load1 block name) u = some (Tile.vec (fun i : Fin block =>
      endpointData p l a padding c head stride size start i.val)) := by
  rw [load1_eval r u name (l.head*head) (p.T*size) block stride ((c*p.BT+p.BT-1)*size+start) hp]
  apply congrArg some
  apply Tile.ext
  rintro ⟨i,⟨⟩⟩
  simp only [Tile.vec,endpointData,hm,hu]
  split <;> rfl

def seedData (p : Layout) (l : Launch) (a padding : Nat → ℝ) (e : Fin p.BK) (j : Fin p.BV) : ℝ :=
  let k := keyIndex p l e.val
  let v := valueIndex p l j.val
  if k < p.K ∧ v < p.V then a (stateAddress p l k v) else padding (stateAddress p l k v)

theorem seed_load (p : Layout) (l : Launch) (r : RegionName) (a padding : Nat → ℝ)
    (name : RegName) (u : BlockState)
    (hm : ∀ off, u.readMem r off = a off) (hu : u.undef r = padding)
    (hp : u.regs .blockPtr [p.BK,p.BV] name = some (statePointer p l r)) :
    evalOp (load2 p.BK p.BV name) u = some (stateTile p (seedData p l a padding)) := by
  rw [load2_eval r u name (l.head*p.K*p.V) p.K p.V p.BK p.BV p.V 1 (l.key*p.BK) (l.value*p.BV) hp]
  apply congrArg some
  apply Tile.ext
  rintro ⟨e,j,⟨⟩⟩
  simp only [stateTile,Tile.mat,seedData,stateAddress,keyIndex,valueIndex,hm,hu,Nat.mul_one]
  split <;> rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGatedAttention.RealHLoads
