import bench.tritonbench_g.var_len_copy.Proofs.DiscreteRealStep
import bench.tritonbench_g.var_len_copy.Proofs.Math

namespace VeriTile.Bench.TritonBenchG.VarLenCopy.DiscreteReal
open VeriTile Triton DiscreteSource

 theorem address_nat {B : Nat} (start current : Nat) (i : TileIndex [B]) :
    address (B := B) start current i = start + current + i.1.val := by
  simp only [address, ← Int.natCast_add, Int.toNat_natCast]

 theorem address_injective {B : Nat} (start current : Nat) :
    Function.Injective (address (B := B) start current) := by
  intro i j h
  simp only [address_nat] at h
  apply Prod.ext
  · exact Fin.ext (Nat.add_left_cancel h)
  · exact Subsingleton.elim _ _

theorem read_write (k : Kind) (s : BlockState) (dst r : RegionName)
    (off o : Nat) (v : TileCarrier k.dtype) :
    (s.writeMemTyped k.dtype dst off v).readMemValue k.dtype r o =
      if r = dst ∧ o = off then v else s.readMemValue k.dtype r o := by
  by_cases h : r = dst ∧ o = off
  all_goals cases k <;> simp [h, Kind.dtype, BlockState.writeMemTyped, BlockState.readMemValue,
    BlockState.readMemTyped]
  all_goals simp [h, MemCell.readAs, MemCell.of]

theorem scatter_value (k : Kind) {B : Nat} (s : BlockState) (dst : RegionName)
    (start current : Nat) (length : Int) (v : TileIndex [B] → TileCarrier k.dtype)
    (i : TileIndex [B]) (hi : (i.1.val : Int) < length) :
    (scatter k s dst start current length v).readMemValue k.dtype dst
      (start + current + i.1.val) = v i := by
  rw [← address_nat start current i]
  have ha : active length i = Bool.true := by simpa [active] using hi
  have hf (l : List (TileIndex [B])) (st : BlockState)
      (h : i ∈ l ∨ st.readMemValue k.dtype dst (address start current i) = v i) :
      (l.foldl (fun acc j ↦ if active length j then
        acc.writeMemTyped k.dtype dst (address start current j) (v j) else acc) st).readMemValue k.dtype dst (address start current i) = v i := by
    induction l generalizing st with
    | nil => simpa using h
    | cons j tail ih =>
      rw [List.foldl_cons]
      by_cases hj : active length j = Bool.true
      · rw [if_pos hj]
        apply ih
        rcases h with hm | hv
        · rcases List.mem_cons.mp hm with he | hm
          · subst j
            right
            simp [read_write]
          · exact Or.inl hm
        · right
          rw [read_write]
          by_cases he : address start current i = address start current j
          · have hij := address_injective start current he
            subst j
            simp
          · simpa [he] using hv
      · rw [if_neg hj]
        apply ih
        rcases h with hm | hv
        · rcases List.mem_cons.mp hm with he | hm
          · subst j; exact False.elim (hj ha)
          · exact Or.inl hm
        · exact Or.inr hv
  exact hf _ s (Or.inl (TileShape.mem_allIndices [B] i))

/-- Each iteration writes only its actual active lanes. -/
 theorem scatter_frame (k : Kind) {B : Nat} (s : BlockState) (dst : RegionName)
    (start current : Nat) (length : Int) (v : TileIndex [B] → TileCarrier k.dtype) :
    BlockState.WriteWithin
      (fun addr ↦ addr.1 = dst ∧ ∃ lane, lane < B ∧ (lane : Int) < length ∧
        addr.2 = start + current + lane) s (scatter k s dst start current length v) := by
  apply BlockState.foldl_writeMemTypedAt_masked_writeWithin k.dtype (fun _ ↦ dst)
    (address start current) v (active length) (TileShape.allIndices [B]) s
  intro i _ hi
  exact ⟨rfl, i.1.val, i.1.isLt, of_decide_eq_true hi, address_nat start current i⟩

 theorem scatter_state (k : Kind) {B : Nat} (s : BlockState) (dst : RegionName)
    (start current length : Int) (v : TileIndex [B] → TileCarrier k.dtype) :
    (scatter k s dst start current length v).regs = s.regs ∧
    (scatter k s dst start current length v).pids = s.pids ∧
    (scatter k s dst start current length v).numPids = s.numPids ∧
    (scatter k s dst start current length v).undef = s.undef := by
  cases k <;> unfold scatter
  all_goals
    generalize TileShape.allIndices [B] = lanes
    induction lanes generalizing s with
    | nil => exact ⟨rfl,rfl,rfl,rfl⟩
    | cons i tail ih =>
      simp only [List.foldl_cons]
      by_cases hi : active length i = Bool.true
      · simpa only [if_pos hi, BlockState.writeMemTyped, Kind.dtype] using
          ih (s.writeMemTyped _ dst (address start current i) (v i))
      · simpa only [if_neg hi] using ih s

end VeriTile.Bench.TritonBenchG.VarLenCopy.DiscreteReal

#guard_msgs (drop info) in
#auditModuleAxioms
