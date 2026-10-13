import VeriTile.Triton.Semantics
import Mathlib.Algebra.Order.Floor.Div
import VeriTile.Meta.StatementAudit
/-! The source reads every lane of each visited KV tile. `SEQLEN` and the
causal `hi` are loop stopping bounds, not masks on these block loads. -/
namespace VeriTile.Bench.TritonBenchG.FlashAttn.Geometry
open VeriTile.Triton

def stop (causal : Bool) (SEQLEN BLOCK_M pid₀ : Nat) : Nat :=
  if causal then (pid₀+1)*BLOCK_M else SEQLEN

def tiles (hi BLOCK_N : Nat) : Nat := hi ⌈/⌉ BLOCK_N

def visited (hi BLOCK_N : Nat) : Nat := tiles hi BLOCK_N * BLOCK_N

theorem stop_le_visited (hi BLOCK_N : Nat) (hBN : 0<BLOCK_N) :
    hi≤visited hi BLOCK_N := by
  simpa [visited,tiles,Nat.mul_comm] using
    (le_smul_ceilDiv (α:=Nat) (β:=Nat) (b:=hi) hBN)

theorem tile_start_lt (hi BLOCK_N c : Nat) (hBN : 0<BLOCK_N) :
    c<tiles hi BLOCK_N ↔ c*BLOCK_N<hi := by
  simpa only [tiles,← not_le,ceilDiv_le_iff_le_mul hBN,Nat.mul_comm]

theorem tiles_positive (hi BLOCK_N : Nat) (hhi : 0<hi) (hBN : 0<BLOCK_N) :
    0<tiles hi BLOCK_N := by
  exact (tile_start_lt hi BLOCK_N 0 hBN).mpr (by simpa using hhi)

theorem visited_positive (hi BLOCK_N : Nat) (hhi : 0<hi) (hBN : 0<BLOCK_N) :
    0<visited hi BLOCK_N := Nat.mul_pos (tiles_positive hi BLOCK_N hhi hBN) hBN

theorem visited_aligned (hi BLOCK_N : Nat) : BLOCK_N ∣ visited hi BLOCK_N := by
  exact ⟨tiles hi BLOCK_N,Nat.mul_comm _ _⟩

theorem tile_lane_bound (hi BLOCK_N c : Nat) (hc : c<tiles hi BLOCK_N)
    (j : Fin BLOCK_N) : c*BLOCK_N+j.val<visited hi BLOCK_N := by
  have h := Nat.mul_le_mul_right BLOCK_N hc
  have h' := j.isLt
  unfold visited
  calc c*BLOCK_N+j.val < c*BLOCK_N+BLOCK_N := Nat.add_lt_add_left h' _
    _ = (c+1)*BLOCK_N := by ring
    _ ≤ tiles hi BLOCK_N*BLOCK_N := h

theorem visited_minimal (hi BLOCK_N n : Nat) (hBN : 0<BLOCK_N)
    (hhi : hi≤n) (hn : BLOCK_N∣n) : visited hi BLOCK_N≤n := by
  obtain ⟨k,rfl⟩ := hn
  have hk : tiles hi BLOCK_N≤k := (ceilDiv_le_iff_le_mul hBN).mpr hhi
  simpa [visited,Nat.mul_comm] using Nat.mul_le_mul_right BLOCK_N hk

theorem stop_positive (causal : Bool) (SEQLEN BLOCK_M pid₀ : Nat)
    (hBM : 0<BLOCK_M) (hSEQ : causal=Bool.false → 0<SEQLEN) :
    0<stop causal SEQLEN BLOCK_M pid₀ := by
  cases causal
  · exact hSEQ rfl
  · exact Nat.mul_pos (Nat.succ_pos _) hBM

/-- A noncausal tail `hi=3, BLOCK_N=2` loads and uses four keys. -/
example : visited 3 2=4 := by decide
/-- A causal query block also walks the entire final KV tile. -/
example : visited (stop Bool.true 9 3 0) 2=4 := by decide

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.FlashAttn.Geometry
