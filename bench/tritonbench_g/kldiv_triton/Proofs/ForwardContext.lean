import bench.tritonbench_g.kldiv_triton.Proofs.ForwardMemory

namespace VeriTile.Bench.TritonBenchG.KldivTriton.FullForward
open VeriTile.Triton

/-- Independent whole-program footprint, with one scalar for each executed reduced chunk. -/
def footprint (O : RegionName) (base N B reduction : Nat) (address : RegionName × Nat) : Prop :=
  address.1 = O ∧ ∃ k, k < FullMath.outputCount N B reduction ∧ address.2 = base + k

def outputBase (base chunk reduction : Nat) : Nat :=
  if reduction = 0 then base else base + chunk

structure Context (initial : BlockState) (Y T O : RegionName)
    (yBase tBase base N B c reduction : Nat) (s : BlockState) : Prop where
  frame : BlockState.WriteWithin (footprint O base N B reduction) initial s
  inputY : ∀ k, s.mem Y k = initial.mem Y k
  inputT : ∀ k, s.mem T k = initial.mem T k
  pids : s.pids = initial.pids
  numPids : s.numPids = initial.numPids
  undef : s.undef = initial.undef
  yReg : s.regs .ptr [] "Y" = some (Tile.scalar (Y, yBase))
  tReg : s.regs .ptr [] "T" = some (Tile.scalar (T, tBase))
  oReg : s.regs .ptr [] "O" = some (Tile.scalar (O, outputBase base c reduction))
  baseReg : s.regs .nat [B] "base_offsets" = some (Tile.vec fun j ↦ j.val)

/-- The signed loop index is a register update, so it preserves the entire memory invariant. -/
theorem Context.setIndex {initial s : BlockState} {Y T O : RegionName}
    {yBase tBase base N B c reduction : Nat}
    (h : Context initial Y T O yBase tBase base N B c reduction s) (index : Int) :
    Context initial Y T O yBase tBase base N B c reduction
      (s.setReg "i" .int [] (Tile.scalar index)) := by
  exact ⟨h.frame, h.inputY, h.inputT, h.pids, h.numPids, h.undef,
    by simpa using h.yReg, by simpa using h.tReg,
    by simpa using h.oReg, by simpa using h.baseReg⟩

/-- An executed chunk lies inside the source's full footprint, even on its partial last tile. -/
theorem block_within (O : RegionName) (base N B c reduction : Nat)
    (hB : 0 < B) (hc : c * B < N) (address : RegionName × Nat)
    (h : blockFootprint O (outputBase base c reduction) N B (c * B) reduction address) :
    footprint O base N B reduction address := by
  by_cases hr : reduction = 0
  · simp only [blockFootprint, outputBase, hr, if_true] at h
    rcases h with ⟨hO, lane, _, hn, he⟩
    refine ⟨hO, c * B + lane, ?_, ?_⟩
    · simpa only [FullMath.outputCount, hr, if_true] using hn
    · omega
  · simp only [blockFootprint, outputBase, hr, if_false] at h
    refine ⟨h.1, c, ?_, h.2⟩
    simpa only [FullMath.outputCount, hr, if_false] using
      (FullMath.lt_rounds_iff N B c hB).mpr hc

/-- Source execution preserves both input snapshots and advances only the actual output pointer. -/
theorem context_step (initial s : BlockState) (Y T O : RegionName)
    (yBase tBase base N B c reduction : Nat) (flag : Bool)
    (hB : 0 < B) (hc : c * B < N) (hYO : Y ≠ O) (hTO : T ≠ O)
    (h : Context initial Y T O yBase tBase base N B c reduction s) :
    Context initial Y T O yBase tBase base N B (c + 1) reduction
      (resultState (B := B) s Y T O yBase tBase (outputBase base c reduction)
        N (c * B : Nat) flag reduction) := by
  have hf := result_frame s Y T O yBase tBase (outputBase base c reduction)
    N B (c * B) flag reduction
  have hm := result_metadata s Y T O yBase tBase (outputBase base c reduction)
    N B (c * B : Nat) flag reduction
  have hp := result_pointers s Y T O yBase tBase (outputBase base c reduction)
    N B (c * B : Nat) flag reduction h.yReg h.tReg h.oReg h.baseReg
  constructor
  · exact BlockState.writeWithin_trans h.frame (fun r k hn ↦ hf r k
      (fun hb ↦ hn (block_within O base N B c reduction hB hc (r,k) hb)))
  · intro k
    exact (hf Y k (by
      intro he
      have hh := block_within O base N B c reduction hB hc (Y,k) he
      exact hYO hh.1)).symm.trans (h.inputY k)
  · intro k
    exact (hf T k (by
      intro he
      have hh := block_within O base N B c reduction hB hc (T,k) he
      exact hTO hh.1)).symm.trans (h.inputT k)
  · exact hm.1.trans h.pids
  · exact hm.2.1.trans h.numPids
  · exact hm.2.2.trans h.undef
  · exact hp.1
  · exact hp.2.1
  · by_cases hr : reduction = 0 <;>
      simpa [outputBase, hr, Nat.add_assoc] using hp.2.2.1
  · exact hp.2.2.2

end VeriTile.Bench.TritonBenchG.KldivTriton.FullForward
#guard_msgs (drop info) in
#auditModuleAxioms
