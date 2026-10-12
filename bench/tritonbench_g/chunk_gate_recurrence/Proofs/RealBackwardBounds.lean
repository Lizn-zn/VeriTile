import bench.tritonbench_g.chunk_gate_recurrence.Proofs.RealBackwardSafety
import bench.tritonbench_g.chunk_gate_recurrence.Proofs.RealForwardBounds

/-! Allocation lengths and ordinary whole-block launch fit imply every actual
backward access bound. With one chunk, S, DI and DG need no allocation: only
DS/D reads and the DL output are reached. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.RealBackwardBounds
open VeriTile.Triton RealProgram RealBackwardPointers
set_option maxHeartbeats 1000000

theorem gradientAddress_lt (p : Layout) (l : Launch) (count t : Nat) (ht : t < count)
    (hk : l.key < numK p) (hv : l.value < numV p) :
    gradientAddress p l t < (l.head*p.N+count)*numK p*numV p := by
  have hinner := TraceProofs.matrix_offset_lt 0 (numK p) (numV p) ⟨_,hk⟩ ⟨_,hv⟩
  simp only [Nat.zero_add] at hinner
  have hle := Nat.mul_le_mul_right (numK p*numV p)
    (show l.head*p.N+t+1 ≤ l.head*p.N+count by omega)
  calc
    gradientAddress p l t = (l.head*p.N+t)*(numK p*numV p)+(l.key*numV p+l.value) := by
      unfold gradientAddress; ring
    _ < (l.head*p.N+t+1)*(numK p*numV p) := by nlinarith
    _ ≤ (l.head*p.N+count)*(numK p*numV p) := hle
    _ = (l.head*p.N+count)*numK p*numV p := by ring

theorem sufficient (p : Layout) (l : Launch) (S D DI DG DL DS : RegionName) (bounds : RegionBounds)
    (hBK : 0 < p.BK) (hBV : 0 < p.BV)
    (hk : (l.key+1)*p.BK ≤ p.K) (hv : (l.value+1)*p.BV ≤ p.V)
    (hS : 1 < p.N → (l.head*p.N+(p.N-1))*p.K*p.V ≤ bounds S)
    (hDS : (l.head+1)*p.N*p.K*p.V ≤ bounds DS)
    (hD : (l.head+1)*p.N ≤ bounds D)
    (hDI : 1 < p.N → (l.head*p.N+(p.N-1))*p.K*p.V ≤ bounds DI)
    (hDG : 1 < p.N → (l.head*p.N+(p.N-1))*numK p*numV p ≤ bounds DG)
    (hDL : (l.head+1)*p.K*p.V ≤ bounds DL) :
    RealBackwardSafety.Bounds p l S D DI DG DL DS bounds := by
  have hkey : l.key < numK p := by
    have h := (Nat.le_div_iff_mul_le hBK).2 hk
    exact lt_of_lt_of_le (Nat.lt_succ_self _) h
  have hval : l.value < numV p := by
    have h := (Nat.le_div_iff_mul_le hBV).2 hv
    exact lt_of_lt_of_le (Nat.lt_succ_self _) h
  constructor
  · intro t e j ht
    exact lt_of_lt_of_le (RealForwardBounds.address_lt p l (p.N-1) t ht hk hv e j) (hS (by omega))
  · intro t e j ht
    have h := RealForwardBounds.address_lt p l p.N t ht hk hv e j
    have heq : (l.head*p.N+p.N)*p.K*p.V = (l.head+1)*p.N*p.K*p.V := by ring
    rw [heq] at h
    exact lt_of_lt_of_le h hDS
  · intro t ht
    dsimp [gateAddress]
    simp only [Nat.add_mul,Nat.one_mul] at hD
    omega
  · intro t e j ht
    exact lt_of_lt_of_le (RealForwardBounds.address_lt p l (p.N-1) t ht hk hv e j) (hDI (by omega))
  · intro t ht
    exact lt_of_lt_of_le (gradientAddress_lt p l (p.N-1) t ht hkey hval) (hDG (by omega))
  · intro e j
    exact lt_of_lt_of_le (RealForwardBounds.seedAddress_lt p l hk hv e j) hDL

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.RealBackwardBounds
