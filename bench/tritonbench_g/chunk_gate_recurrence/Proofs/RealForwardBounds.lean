import bench.tritonbench_g.chunk_gate_recurrence.Proofs.RealForwardSafety

/-! Ordinary block-fit and allocation lengths imply every active address bound.
Forward never reads the final source/gate chunk; for N=1 neither input region
needs an allocation. These are per-program conditions, with symbolic sizes and
all three program IDs. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.RealForwardBounds
open VeriTile.Triton RealProgram
set_option maxHeartbeats 1000000

theorem lane_lt (block extent index : Nat) (hfit : (index+1)*block ≤ extent) (i : Fin block) :
    index*block+i.val < extent := by
  have hi := i.isLt
  simp only [Nat.add_mul,Nat.one_mul] at hfit
  omega

theorem address_lt (p : Layout) (l : Launch) (count t : Nat) (ht : t < count)
    (hk : (l.key+1)*p.BK ≤ p.K) (hv : (l.value+1)*p.BV ≤ p.V)
    (e : Fin p.BK) (j : Fin p.BV) :
    address p l t e.val j.val < (l.head*p.N+count)*p.K*p.V := by
  have he := lane_lt p.BK p.K l.key hk e
  have hj := lane_lt p.BV p.V l.value hv j
  have hinner := TraceProofs.matrix_offset_lt 0 p.K p.V ⟨_,he⟩ ⟨_,hj⟩
  simp only [Nat.zero_add] at hinner
  have hle := Nat.mul_le_mul_right (p.K*p.V)
    (show l.head*p.N+t+1 ≤ l.head*p.N+count by omega)
  calc
    address p l t e.val j.val = (l.head*p.N+t)*(p.K*p.V) +
        ((l.key*p.BK+e.val)*p.V+(l.value*p.BV+j.val)) := by unfold address; ring
    _ < (l.head*p.N+t+1)*(p.K*p.V) := by nlinarith
    _ ≤ (l.head*p.N+count)*(p.K*p.V) := hle
    _ = (l.head*p.N+count)*p.K*p.V := by ring

theorem seedAddress_lt (p : Layout) (l : Launch)
    (hk : (l.key+1)*p.BK ≤ p.K) (hv : (l.value+1)*p.BV ≤ p.V)
    (e : Fin p.BK) (j : Fin p.BV) : seedAddress p l e.val j.val < (l.head+1)*p.K*p.V := by
  have he := lane_lt p.BK p.K l.key hk e
  have hj := lane_lt p.BV p.V l.value hv j
  have h := TraceProofs.matrix_offset_lt (l.head*p.K*p.V) p.K p.V ⟨_,he⟩ ⟨_,hj⟩
  convert h using 1 <;> dsimp [seedAddress] <;> ring

theorem sufficient (p : Layout) (l : Launch) (S D O seed : RegionName)
    (hasSeed : Bool) (bounds : RegionBounds)
    (hk : (l.key+1)*p.BK ≤ p.K) (hv : (l.value+1)*p.BV ≤ p.V)
    (hS : 1 < p.N → (l.head*p.N+(p.N-1))*p.K*p.V ≤ bounds S)
    (hD : 1 < p.N → l.head*p.N+(p.N-1) ≤ bounds D)
    (hO : (l.head+1)*p.N*p.K*p.V ≤ bounds O)
    (hSeed : hasSeed = Bool.true → (l.head+1)*p.K*p.V ≤ bounds seed) :
    RealForwardSafety.Bounds p l S D O seed hasSeed bounds := by
  constructor
  · intro t e j ht
    exact lt_of_lt_of_le (address_lt p l (p.N-1) t ht hk hv e j) (hS (by omega))
  · intro t ht
    have h := hD (by omega)
    dsimp [gateAddress]
    omega
  · intro t e j ht
    have h := address_lt p l p.N t ht hk hv e j
    have hEq : (l.head*p.N+p.N)*p.K*p.V = (l.head+1)*p.N*p.K*p.V := by ring
    rw [hEq] at h
    exact lt_of_lt_of_le h hO
  · intro hon e j
    exact lt_of_lt_of_le (seedAddress_lt p l hk hv e j) (hSeed hon)

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.RealForwardBounds
