import VeriTile.Meta.StatementAudit
import bench.tritonbench_g.var_len_copy.Proofs.RealScatter

namespace VeriTile.Bench.TritonBenchG.VarLenCopy.FullReal
open VeriTile Triton
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000

/-- The footprint is the actual copied span, not the requested segment length. -/
def footprint (dst : RegionName) (start : Nat) (length : Int) (B : Nat)
    (addr : RegionName × Nat) : Prop :=
  addr.1 = dst ∧ ∃ n, n < Math.copiedLength length B ∧ addr.2 = start + n

structure Inv (initial : BlockState) (src dst : RegionName)
    (B : Nat) (length : Int) (oldStart newStart c : Nat) (s : BlockState) : Prop where
  frame : BlockState.WriteWithin (footprint dst newStart length B) initial s
  input : ∀ o, s.mem src o = initial.mem src o
  pids : s.pids = initial.pids
  numPids : s.numPids = initial.numPids
  undef : s.undef = initial.undef
  lengthReg : s.regs .int [] "length" = some (Tile.scalar length)
  oldReg : s.regs .int [] "old_start" = some (Tile.scalar (oldStart : Int))
  newReg : s.regs .int [] "new_start" = some (Tile.scalar (newStart : Int))
  oldOffset : s.regs .nat [B] "old_offset" = some (Tile.vec fun i ↦ i.val)
  newOffset : s.regs .nat [B] "new_offset" = some (Tile.vec fun i ↦ i.val)
  values : ∀ t lane, t < c → lane < B → (lane : Int) < length →
    s.readMem dst (newStart + (t * B + lane)) = initial.readMem src (oldStart + (t * B + lane))

 theorem Inv.setIndex {initial s : BlockState} {src dst : RegionName}
    {B : Nat} {length : Int} {oldStart newStart c : Nat}
    (h : Inv initial src dst B length oldStart newStart c s) (index : Int) :
    Inv initial src dst B length oldStart newStart c
      (s.setReg "i" .int [] (Tile.scalar index)) := by
  exact ⟨h.frame,h.input,h.pids,h.numPids,h.undef,
    by simpa using h.lengthReg, by simpa using h.oldReg, by simpa using h.newReg,
    by simpa using h.oldOffset, by simpa using h.newOffset,h.values⟩

 theorem Inv.mid {initial s : BlockState} {src dst : RegionName}
    {B : Nat} {length : Int} {oldStart newStart c : Nat}
    (h : Inv initial src dst B length oldStart newStart c s) :
    Inv initial src dst B length oldStart newStart c
      (midState (B := B) s src length oldStart (c * B : Nat)) := by
  exact ⟨h.frame,h.input,h.pids,h.numPids,h.undef,
    by simpa [midState] using h.lengthReg, by simpa [midState] using h.oldReg,
    by simpa [midState] using h.newReg, by simpa [midState] using h.oldOffset,
    by simpa [midState] using h.newOffset,h.values⟩

 theorem invariant_step (initial s : BlockState) (os ol ns : Region .int)
    (src dst : RegionName) (B : Nat) (length : Int) (oldStart newStart c : Nat)
    (hB : 0 < B) (hc : c < Math.rounds length B) (hsd : src ≠ dst)
    (h : Inv initial src dst B length oldStart newStart c s) :
    ∃ next, stepStmts (body os ol ns src dst B)
        (s.setReg "i" .int [] (Tile.scalar (c * B : Nat))) = some next ∧
      Inv initial src dst B length oldStart newStart (c + 1) next := by
  let indexed := s.setReg "i" .int [] (Tile.scalar (c * B : Nat))
  have hi : Inv initial src dst B length oldStart newStart c indexed := h.setIndex _
  let middle := midState (B := B) indexed src length oldStart (c * B : Nat)
  have hm : Inv initial src dst B length oldStart newStart c middle := hi.mid
  let vals := value (B := B) indexed src length oldStart (c * B : Nat)
  let next := scatter middle dst newStart (c * B : Nat) length vals
  have hf := scatter_frame middle dst newStart (c * B : Nat) length vals
  have hstate := scatter_state middle dst newStart (c * B : Nat) length vals
  have hstart := (Math.loop_test length hB).mpr hc
  refine ⟨next, ?_, ?_⟩
  · exact step os ol ns src dst B indexed length oldStart newStart (c * B : Nat)
      hi.lengthReg hi.oldReg hi.newReg (by simp [indexed]) hi.oldOffset hi.newOffset
  · constructor
    · intro r o hout
      rw [hm.frame r o hout]
      apply hf r o
      rintro ⟨rfl,lane,hl,hL,ho⟩
      apply hout
      refine ⟨rfl,c * B + lane, (Math.touched_iff length hB).mp ?_, ?_⟩
      · exact ⟨c,lane,hstart,hl,hL,rfl⟩
      · omega
    · intro o
      exact (hf src o (by rintro ⟨he,_⟩; exact hsd he)).symm.trans (hm.input o)
    · exact hstate.2.1.trans hm.pids
    · exact hstate.2.2.1.trans hm.numPids
    · exact hstate.2.2.2.trans hm.undef
    · rw [hstate.1]; exact hm.lengthReg
    · rw [hstate.1]; exact hm.oldReg
    · rw [hstate.1]; exact hm.newReg
    · rw [hstate.1]; exact hm.oldOffset
    · rw [hstate.1]; exact hm.newOffset
    · intro t lane ht hl hL
      by_cases he : t = c
      · subst t
        let i : TileIndex [B] := (⟨lane,hl⟩,PUnit.unit)
        have hv := scatter_value middle dst newStart (c * B : Nat) length vals i hL
        have hread : indexed.readMem src (oldStart + (c * B + lane)) =
            initial.readMem src (oldStart + (c * B + lane)) := by
          simp only [BlockState.readMem, hi.input]
        simpa [next,vals,value,active,i,address_nat,hL,Nat.add_assoc] using hv.trans
          (show vals i = initial.readMem src (oldStart + (c * B + lane)) by
            simpa only [vals,value,active,i,decide_eq_true_eq,if_pos hL,address_nat,Nat.add_assoc] using hread)
      · have htc : t < c := by omega
        have hmul := Nat.mul_le_mul_right B (show t + 1 ≤ c by omega)
        simp only [Nat.add_mul, Nat.one_mul] at hmul
        have hcell := hf dst (newStart + (t * B + lane)) (by
          rintro ⟨_,other,_,_,ho⟩
          omega)
        have hread : next.readMem dst (newStart + (t * B + lane)) =
            middle.readMem dst (newStart + (t * B + lane)) := by
          simp only [next, BlockState.readMem, ← hcell]
        exact hread.trans (hm.values t lane htc hl hL)

end VeriTile.Bench.TritonBenchG.VarLenCopy.FullReal

#guard_msgs (drop info) in
#auditModuleAxioms
