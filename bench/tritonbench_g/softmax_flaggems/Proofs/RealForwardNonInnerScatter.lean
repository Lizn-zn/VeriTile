import bench.tritonbench_g.softmax_flaggems.Proofs.RealForwardNonInnerStore
import bench.tritonbench_g.softmax_flaggems.Proofs.RealForwardNonInnerPrepare
import bench.tritonbench_g.softmax_flaggems.Proofs.RealReverseAddress

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardNonInner
open VeriTile Triton

theorem storeValue_expected {B C : Nat} (sb st : BlockState) (X : RegionName)
    (row N K kbase base : Nat) (hN : 0 < N) (hB : 0 < B)
    (hinput : ∀ o, st.readMem X o = sb.readMem X o)
    (i : TileIndex [B, C]) (hi : tileActive N K kbase base i = Bool.true) :
    storeValue st X row N K kbase base
        (finalMax sb X row N K kbase B C hB)
        (finalSum sb X row N K kbase B C hB) i =
      expected sb X row N K (base + i.1.val) (kbase + i.2.1.val) := by
  have hi' : base + i.1.val < N ∧ kbase + i.2.1.val < K := of_decide_eq_true hi
  obtain ⟨m, hm, hz⟩ := final_denominator sb X row N K kbase B C hN hB i.2.1 hi'.2
  unfold storeValue score
  rw [if_pos hi', hm, hz, hinput]
  exact stored_value _ _ _

 theorem tile_active_injective {B C row N K kbase base : Nat}
    {i j : TileIndex [B, C]}
    (hi : tileActive N K kbase base i = Bool.true)
    (hj : tileActive N K kbase base j = Bool.true)
    (heq : tileOffset row N K kbase base i = tileOffset row N K kbase base j) : i = j := by
  have hi' : base + i.1.val < N ∧ kbase + i.2.1.val < K := of_decide_eq_true hi
  have hj' : base + j.1.val < N ∧ kbase + j.2.1.val < K := of_decide_eq_true hj
  unfold tileOffset at heq
  have hcancel : (base + i.1.val) * K + (kbase + i.2.1.val) =
      (base + j.1.val) * K + (kbase + j.2.1.val) := by omega
  obtain ⟨hn, hk⟩ := RealAddress.row_column_injective hi'.2 hj'.2 hcancel
  apply Prod.ext
  · exact Fin.ext (Nat.add_left_cancel hn)
  · apply Prod.ext
    · exact Fin.ext (Nat.add_left_cancel hk)
    · exact Subsingleton.elim _ _

 theorem tile_access_bound {B C M row N K kbase base : Nat}
    (hrow : row < M) (i : TileIndex [B, C])
    (hi : tileActive N K kbase base i = Bool.true) :
    tileOffset row N K kbase base i < M * N * K := by
  have hi' : base + i.1.val < N ∧ kbase + i.2.1.val < K := of_decide_eq_true hi
  exact RealAddress.non_inner_bound hrow hi'.1 hi'.2

 theorem scatter_value {B C : Nat} (s : BlockState) (Y : RegionName)
    (row N K kbase base : Nat) (value : TileIndex [B, C] → ℝ)
    (i : TileIndex [B, C]) (hi : tileActive N K kbase base i = Bool.true) :
    (scatter s Y (tileOffset row N K kbase base) value (tileActive N K kbase base)).readMem Y (tileOffset row N K kbase base i) = value i := by
  apply BlockState.scatter_readback_prop_masked_nd_of_true s
    (tileOffset row N K kbase base) value (fun i ↦ tileActive N K kbase base i = Bool.true) i hi
  intro j hj heq
  exact tile_active_injective hj hi heq

 theorem scatter_frame {B C : Nat} (s : BlockState) (Y : RegionName)
    (off : TileIndex [B, C] → Nat) (value : TileIndex [B, C] → ℝ)
    (active : TileIndex [B, C] → Bool) (P : WriteFootprint)
    (hP : ∀ i, active i = Bool.true → P (Y, off i)) :
    BlockState.WriteWithin P s (scatter s Y off value active) := by
  apply BlockState.foldl_writeMem_prop_writeWithin Y off value
    (fun i ↦ active i = Bool.true) (TileShape.allIndices [B, C]) s
  intro i _ hi
  exact hP i hi

 theorem scatter_regs {B C : Nat} (s : BlockState) (Y : RegionName)
    (off : TileIndex [B, C] → Nat) (value : TileIndex [B, C] → ℝ)
    (active : TileIndex [B, C] → Bool) : (scatter s Y off value active).regs = s.regs := by
  funext dtype shape name
  exact foldl_store_regs off value active (TileShape.allIndices [B, C]) s dtype shape name

 theorem scatter_pids {B C : Nat} (s : BlockState) (Y : RegionName)
    (off : TileIndex [B, C] → Nat) (value : TileIndex [B, C] → ℝ)
    (active : TileIndex [B, C] → Bool) : (scatter s Y off value active).pids = s.pids := by
  exact foldl_store_pids off value active (TileShape.allIndices [B, C]) s

 theorem scatter_launch_padding {B C : Nat} (s : BlockState) (Y : RegionName)
    (off : TileIndex [B, C] → Nat) (value : TileIndex [B, C] → ℝ)
    (active : TileIndex [B, C] → Bool) :
    (scatter s Y off value active).numPids = s.numPids ∧
      (scatter s Y off value active).undef = s.undef := by
  unfold scatter
  generalize TileShape.allIndices [B, C] = lanes
  induction lanes generalizing s with
  | nil => exact ⟨rfl, rfl⟩
  | cons i tail ih =>
    simp only [List.foldl_cons]
    by_cases hi : active i = Bool.true
    · simpa only [if_pos hi, BlockState.writeMem_numPids, BlockState.writeMem_undef] using
        ih (s.writeMem Y (off i) (value i))
    · simpa only [if_neg hi] using ih s

theorem scatter_preserve_input {B C : Nat} (s : BlockState) (Y X : RegionName)
    (off : TileIndex [B, C] → Nat) (value : TileIndex [B, C] → ℝ)
    (active : TileIndex [B, C] → Bool) (hXY : X ≠ Y) (o : Nat) :
    (scatter s Y off value active).mem X o = s.mem X o := by
  apply foldl_store_preserve_cell off value (fun i ↦ active i = Bool.true)
  intro i _ _ heq
  exact hXY heq.1.symm

theorem scatter_preserve_processed {B C : Nat} (s : BlockState) (Y : RegionName)
    (row N K kbase c n k : Nat) (value : TileIndex [B, C] → ℝ)
    (hB : 0 < B) (hc : c < RealAddress.rounds N B)
    (hold : RealAddress.rounds N B - c ≤ n / B) (hk : k < K) :
    (scatter s Y (tileOffset row N K kbase (RealAddress.previous N B - c * B))
        value (tileActive N K kbase (RealAddress.previous N B - c * B))).mem Y (row * N * K + n * K + k) = s.mem Y (row * N * K + n * K + k) := by
  apply foldl_store_preserve_cell _ value (fun i ↦
    tileActive N K kbase (RealAddress.previous N B - c * B) i = Bool.true)
  intro i _ hi heq
  have hi' : RealAddress.previous N B - c * B + i.1.val < N ∧
      kbase + i.2.1.val < K := of_decide_eq_true hi
  have haddr := heq.2
  unfold tileOffset at haddr
  have hcancel : (RealAddress.previous N B - c * B + i.1.val) * K +
      (kbase + i.2.1.val) = n * K + k := by omega
  have hn := (RealAddress.row_column_injective hi'.2 hk hcancel).1
  exact RealAddress.reverse_old_disjoint hB hc hold i.1.isLt hn.symm

end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardNonInner
