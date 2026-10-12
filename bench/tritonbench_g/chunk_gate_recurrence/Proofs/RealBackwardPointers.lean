import bench.tritonbench_g.chunk_gate_recurrence.Proofs.RealProgram

/-! Reverse registers retain absolute addresses minus all executed strides.
A dead pointer after the last iteration is not replaced by chunk zero.
The ordinary chunk interpretation is proved only while its index is live. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.RealBackwardPointers
open VeriTile.Triton RealProgram
noncomputable section

abbrev numK (p : Layout) := p.K / p.BK
abbrev numV (p : Layout) := p.V / p.BV

def gradientAddress (p : Layout) (l : Launch) (c : Nat) : Nat :=
  l.head * p.N * numK p * numV p + l.key * numV p + l.value + c * numK p * numV p

def backwardPointer (p : Layout) (l : Launch) (r : RegionName) (start k : Nat) : Tile .ptr [p.BK,p.BV] :=
  ⟨fun i => (r, address p l start i.1.val i.2.1.val - k * (p.K * p.V))⟩

def backwardGate (p : Layout) (l : Launch) (r : RegionName) (k : Nat) : Tile .ptr [] :=
  Tile.scalar (r, gateAddress p l (p.N-1) - k)

def backwardGradient (p : Layout) (l : Launch) (r : RegionName) (k : Nat) : Tile .ptr [] :=
  Tile.scalar (r, gradientAddress p l (p.N-2) - k * (numK p * numV p))

theorem subtract_stride (base start k stride : Nat) (hk : k ≤ start) :
    base + start * stride - k * stride = base + (start-k) * stride := by
  rw [Nat.sub_mul, Nat.add_sub_assoc (Nat.mul_le_mul_right stride hk)]

theorem backwardPointer_live (p : Layout) (l : Launch) (r : RegionName) (start k : Nat)
    (hk : k ≤ start) : backwardPointer p l r start k = pointer p l r (start-k) := by
  apply Tile.ext
  rintro ⟨e,j,⟨⟩⟩
  simp only [backwardPointer,pointer,address,Nat.mul_assoc]
  exact congrArg (fun off => (r,off)) (subtract_stride _ _ _ _ hk)

theorem backwardGate_live (p : Layout) (l : Launch) (r : RegionName) (k : Nat)
    (hk : k ≤ p.N-1) : backwardGate p l r k = gatePointer p l r (p.N-1-k) := by
  simp only [backwardGate,gatePointer,gateAddress,Nat.add_sub_assoc hk]

theorem backwardGradient_live (p : Layout) (l : Launch) (r : RegionName) (k : Nat)
    (hk : k ≤ p.N-2) :
    backwardGradient p l r k = Tile.scalar (r, gradientAddress p l (p.N-2-k)) := by
  simp only [backwardGradient,gradientAddress,Nat.mul_assoc,subtract_stride _ _ _ _ hk]

theorem pointer_decrement (p : Layout) (l : Launch) (r : RegionName) (start k : Nat) :
    Tile.ptrSub .scalarR (backwardPointer p l r start k) (Tile.scalar (p.K*p.V)) =
      backwardPointer p l r start (k+1) := by
  apply Tile.ext
  rintro ⟨e,j,⟨⟩⟩
  simp [Tile.ptrSub,Broadcast.leftIndex,Broadcast.rightIndex,backwardPointer,
    Nat.sub_sub,Nat.add_mul]

theorem gate_decrement (p : Layout) (l : Launch) (r : RegionName) (k : Nat) :
    Tile.ptrSub .nil (backwardGate p l r k) (Tile.scalar 1) = backwardGate p l r (k+1) := by
  apply Tile.ext
  intro i
  simp [Tile.ptrSub,Broadcast.leftIndex,Broadcast.rightIndex,backwardGate,Nat.sub_sub]

theorem gradient_decrement (p : Layout) (l : Launch) (r : RegionName) (k : Nat) :
    Tile.ptrSub .nil (backwardGradient p l r k) (Tile.scalar (numK p * numV p)) =
      backwardGradient p l r (k+1) := by
  apply Tile.ext
  intro i
  simp [Tile.ptrSub,Broadcast.leftIndex,Broadcast.rightIndex,backwardGradient,
    Nat.sub_sub,Nat.add_mul]

theorem gradient_injective (p : Layout) (l : Launch)
    (hk : 0 < numK p) (hv : 0 < numV p) : Function.Injective (gradientAddress p l) := by
  intro c d h
  have hp : 0 < numK p * numV p := Nat.mul_pos hk hv
  apply Nat.eq_of_mul_eq_mul_right hp
  simpa only [gradientAddress,Nat.mul_assoc,Nat.add_right_inj] using h

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.RealBackwardPointers
