import VeriTile.Triton.Memory.Frame.Basic
import bench.tritonbench_g.softmax_flaggems.Proofs.RealForwardInnerStore
import bench.tritonbench_g.softmax_flaggems.Proofs.RealForwardInnerPrepare
import bench.tritonbench_g.softmax_flaggems.Proofs.RealReverseAddress

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardInner
open VeriTile Triton

theorem storeValue_expected {B : Nat} (sb st : BlockState) (X : RegionName)
    (row N base : Nat) (hN : 0 < N) (hB : 0 < B)
    (hinput : ∀ o, st.readMem X o = sb.readMem X o)
    (i : TileIndex [B]) (hi : tileActive N base i = Bool.true) :
    storeValue st X row N base
        (finalMax sb X row N B hB)
        (finalSum sb X row N B hB) i =
      expected sb X row N (base + i.1.val) := by
  have hi' : base + i.1.val < N := of_decide_eq_true hi
  obtain ⟨m, hm, hz⟩ := final_denominator sb X row N B hN hB
  unfold storeValue score
  rw [if_pos hi', hm, hz, hinput]
  exact RealForwardNonInner.stored_value _ _ _

 theorem tile_active_injective {B row N base : Nat} {i j : TileIndex [B]}
    (heq : tileOffset row N base i = tileOffset row N base j) : i = j := by
  unfold tileOffset at heq
  apply Prod.ext
  · apply Fin.ext
    omega
  · exact Subsingleton.elim _ _

 theorem tile_access_bound {B M row N base : Nat} (hrow : row < M)
    (i : TileIndex [B]) (hi : tileActive N base i = Bool.true) :
    tileOffset row N base i < M * N := by
  have hi' : base + i.1.val < N := of_decide_eq_true hi
  have h := RealAddress.non_inner_bound (K := 1) (k := 0) hrow hi' (by omega)
  simpa [tileOffset] using h

 theorem scatter_value {B : Nat} (s : BlockState) (Y : RegionName)
    (row N base : Nat) (value : TileIndex [B] → ℝ)
    (i : TileIndex [B]) (hi : tileActive N base i = Bool.true) :
    (scatter s Y (tileOffset row N base) value (tileActive N base)).readMem Y (tileOffset row N base i) = value i := by
  apply BlockState.scatter_readback_prop_masked_nd_of_true s
    (tileOffset row N base) value (fun i ↦ tileActive N base i = Bool.true) i hi
  intro j hj heq
  exact tile_active_injective heq

 theorem scatter_frame {B : Nat} (s : BlockState) (Y : RegionName)
    (off : TileIndex [B] → Nat) (value : TileIndex [B] → ℝ)
    (active : TileIndex [B] → Bool) (P : WriteFootprint)
    (hP : ∀ i, active i = Bool.true → P (Y, off i)) :
    BlockState.WriteWithin P s (scatter s Y off value active) := by
  apply BlockState.foldl_writeMem_prop_writeWithin Y off value
    (fun i ↦ active i = Bool.true) (TileShape.allIndices [B]) s
  intro i _ hi
  exact hP i hi

 theorem scatter_regs {B : Nat} (s : BlockState) (Y : RegionName)
    (off : TileIndex [B] → Nat) (value : TileIndex [B] → ℝ)
    (active : TileIndex [B] → Bool) : (scatter s Y off value active).regs = s.regs := by
  funext dtype shape name
  exact foldl_store_regs off value active (TileShape.allIndices [B]) s dtype shape name

 theorem scatter_pids {B : Nat} (s : BlockState) (Y : RegionName)
    (off : TileIndex [B] → Nat) (value : TileIndex [B] → ℝ)
    (active : TileIndex [B] → Bool) : (scatter s Y off value active).pids = s.pids := by
  exact foldl_store_pids off value active (TileShape.allIndices [B]) s

 theorem scatter_launch_padding {B : Nat} (s : BlockState) (Y : RegionName)
    (off : TileIndex [B] → Nat) (value : TileIndex [B] → ℝ)
    (active : TileIndex [B] → Bool) :
    (scatter s Y off value active).numPids = s.numPids ∧
      (scatter s Y off value active).undef = s.undef := by
  unfold scatter
  generalize TileShape.allIndices [B] = lanes
  induction lanes generalizing s with
  | nil => exact ⟨rfl, rfl⟩
  | cons i tail ih =>
    simp only [List.foldl_cons]
    by_cases hi : active i = Bool.true
    · simpa only [if_pos hi, BlockState.writeMem_numPids, BlockState.writeMem_undef] using
        ih (s.writeMem Y (off i) (value i))
    · simpa only [if_neg hi] using ih s

theorem scatter_preserve_input {B : Nat} (s : BlockState) (Y X : RegionName)
    (off : TileIndex [B] → Nat) (value : TileIndex [B] → ℝ)
    (active : TileIndex [B] → Bool) (hXY : X ≠ Y) (o : Nat) :
    (scatter s Y off value active).mem X o = s.mem X o := by
  apply foldl_store_preserve_cell off value (fun i ↦ active i = Bool.true)
  intro i _ _ heq
  exact hXY heq.1.symm

theorem scatter_preserve_processed {B : Nat} (s : BlockState) (Y : RegionName)
    (row N c n : Nat) (value : TileIndex [B] → ℝ)
    (hB : 0 < B) (hc : c < RealAddress.rounds N B)
    (hold : RealAddress.rounds N B - c ≤ n / B) :
    (scatter s Y (tileOffset row N (RealAddress.previous N B - c * B))
        value (tileActive N (RealAddress.previous N B - c * B))).mem Y (row * N + n) = s.mem Y (row * N + n) := by
  apply foldl_store_preserve_cell _ value (fun i ↦ tileActive N (RealAddress.previous N B - c * B) i = Bool.true)
  intro i _ hi heq
  have haddr := heq.2
  unfold tileOffset at haddr
  have hn : n = RealAddress.previous N B - c * B + i.1.val := by omega
  exact RealAddress.reverse_old_disjoint hB hc hold i.1.isLt hn

end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardInner
