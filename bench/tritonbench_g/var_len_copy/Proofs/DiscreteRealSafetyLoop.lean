import bench.tritonbench_g.var_len_copy.Proofs.DiscreteRealSafetyStep

namespace VeriTile.Bench.TritonBenchG.VarLenCopy.DiscreteReal
open VeriTile Triton DiscreteSource

 theorem invariant_safe (k : Kind) (bounds : RegionBounds) (initial s : BlockState)
    (os ol ns : Region .int) (src dst : RegionName)
    (B : Nat) (length : Int) (oldStart newStart c : Nat)
    (hB : 0 < B) (hc : c < Math.rounds length B)
    (hsrc : oldStart + Math.copiedLength length B ≤ bounds src)
    (hdst : newStart + Math.copiedLength length B ≤ bounds dst)
    (h : Inv k initial src dst B length oldStart newStart c s) :
    Stmt.TraceSafeList bounds (body k os ol ns src dst B)
      (s.setReg "i" .int [] (Tile.scalar (c * B : Nat))) := by
  have hi := h.setIndex (c * B : Nat)
  have hstart := (Math.loop_test length hB).mpr hc
  apply body_safe k bounds os ol ns src dst B _ length oldStart newStart (c * B : Nat)
    hi.lengthReg hi.oldReg hi.newReg (by simp) hi.oldOffset hi.newOffset
  · intro i hL
    have hn := (Math.touched_iff length hB).mp
      (show Math.touched length B (c * B + i.1.val) from ⟨c,i.1.val,hstart,i.1.isLt,hL,rfl⟩)
    rw [address_nat]
    omega
  · intro i hL
    have hn := (Math.touched_iff length hB).mp
      (show Math.touched length B (c * B + i.1.val) from ⟨c,i.1.val,hstart,i.1.isLt,hL,rfl⟩)
    rw [address_nat]
    omega

 theorem loop_safe (k : Kind) (bounds : RegionBounds) (initial : BlockState)
    (os ol ns : Region .int) (src dst : RegionName)
    (B : Nat) (length : Int) (oldStart newStart : Nat)
    (hB : 0 < B) (hsd : src ≠ dst)
    (hsrc : oldStart + Math.copiedLength length B ≤ bounds src)
    (hdst : newStart + Math.copiedLength length B ≤ bounds dst) :
    ∀ c s, c ≤ Math.rounds length B → Inv k initial src dst B length oldStart newStart c s →
      Stmt.forRangeIntTraceSafe bounds "i" (c * B : Nat) length B (body k os ol ns src dst B) s := by
  intro c s
  induction h : (Math.rounds length B - c) using Nat.strong_induction_on generalizing c s with
  | _ remaining ih =>
    intro hc hi
    by_cases hd : c = Math.rounds length B
    · subst c
      have hcover := Math.rounds_coverage length hB
      rw [Stmt.forRangeIntTraceSafe, if_neg (Nat.ne_of_gt hB), if_neg (by omega)]
      trivial
    · have hlt : c < Math.rounds length B := by omega
      have htest := (Math.loop_test length hB).mpr hlt
      obtain ⟨mid,he,hm⟩ := invariant_step k initial s os ol ns src dst B length oldStart newStart c hB hlt hsd hi
      rw [Stmt.forRangeIntTraceSafe, if_neg (Nat.ne_of_gt hB), if_pos (by omega)]
      refine ⟨invariant_safe k bounds initial s os ol ns src dst B length oldStart newStart c hB hlt hsrc hdst hi, ?_⟩
      rw [he]
      have ht := ih (Math.rounds length B - (c + 1)) (by omega)
        (c + 1) mid rfl (by omega) hm
      simpa only [Nat.add_mul,Nat.one_mul,Nat.cast_add] using ht

end VeriTile.Bench.TritonBenchG.VarLenCopy.DiscreteReal

#guard_msgs (drop info) in
#auditModuleAxioms
