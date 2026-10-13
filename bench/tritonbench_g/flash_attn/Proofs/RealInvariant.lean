import bench.tritonbench_g.flash_attn.Proofs.RealInputTiles
import bench.tritonbench_g.flash_attn.Proofs.RealNumericAlgebra

/-! The source loop invariant. Numeric registers describe initial-memory keys
strictly before the next physical tile; read/write pointers and launch registers
are tracked independently of the mathematical sums. -/
namespace VeriTile.Bench.TritonBenchG.FlashAttn.RealExecution
open VeriTile.Triton Formula
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3
noncomputable section

def prefixMax (p : Layout) (s : BlockState) (Q K V : RegionName) (c : Nat) :
    Tile .real [p.BLOCK_M] :=
  ⟨fun i => (prefixSummary p s.readMem Q K V (s.pids 0) (s.pids 1) i.1.val 0 c).center⟩

def prefixDen (p : Layout) (s : BlockState) (Q K V : RegionName) (c : Nat) :
    Tile .real [p.BLOCK_M] :=
  RealTiles.finite (fun i =>
    (prefixSummary p s.readMem Q K V (s.pids 0) (s.pids 1) i.1.val 0 c).denominator)

def prefixAcc (p : Layout) (s : BlockState) (Q K V : RegionName) (c : Nat) :
    Tile .real [p.BLOCK_M,p.DIM] :=
  RealTiles.finite (fun i =>
    (prefixSummary p s.readMem Q K V (s.pids 0) (s.pids 1) i.1.val i.2.1.val c).numerator)

structure PrefixInvariant (p : Layout) (s₀ : BlockState) (Q K V : RegionName)
    (c : Nat) (t : BlockState) : Prop where
  mem : t.mem = s₀.mem
  pids : t.pids = s₀.pids
  undef : t.undef = s₀.undef
  startm : t.regs .nat [] "start_m" = some (Tile.scalar (s₀.pids 0))
  bshead : t.regs .nat [] "off_bs_head" = some (Tile.scalar (s₀.pids 1))
  base : t.regs .nat [] "qkv_base_offset" = some (Tile.scalar (Formula.base p (s₀.pids 1)))
  offm : t.regs .nat [p.BLOCK_M] "off_m" =
    some (Tile.vec (fun i => row p (s₀.pids 0) i.val))
  offn : t.regs .nat [p.BLOCK_N] "off_n" = some (Tile.vec (fun j => j.val))
  query : t.regs .real [p.BLOCK_M,p.DIM] "q" = some (RealExecution.query p s₀ Q)
  maximum : t.regs .real [p.BLOCK_M] "max" = some (prefixMax p s₀ Q K V c)
  denominator : t.regs .real [p.BLOCK_M] "denom" = some (prefixDen p s₀ Q K V c)
  accumulator : t.regs .real [p.BLOCK_M,p.DIM] "out_buffer" = some (prefixAcc p s₀ Q K V c)
  key : t.regs .blockPtr [p.DIM,p.BLOCK_N] "K_block_ptr" =
    some (keyPointer p K (s₀.pids 1) (c*p.BLOCK_N))
  value : t.regs .blockPtr [p.BLOCK_N,p.DIM] "V_block_ptr" =
    some (valuePointer p V (s₀.pids 1) (c*p.BLOCK_N))
  lo : t.regs .nat [] "lo" = some (Tile.scalar 0)
  hi : t.regs .nat [] "hi" = some (Tile.scalar (Formula.hi p (s₀.pids 0)))

theorem entry_invariant (p : Layout) (s₀ : BlockState) (Q K V L O : RegionName) :
    PrefixInvariant p s₀ Q K V 0 (entry Q K V L O p s₀).val := by
  obtain ⟨hm,hp,hu⟩ := entry_frame Q K V L O p s₀
  obtain ⟨hsm,hbs,hbase,hom,hon,hlo,hhi⟩ := entry_indices Q K V L O p s₀
  obtain ⟨hmax,hden,hacc⟩ := entry_seed Q K V L O p s₀
  obtain ⟨hk,hv⟩ := entry_pointers Q K V L O p s₀
  refine { mem := hm, pids := hp, undef := hu, startm := hsm, bshead := hbs,
           base := hbase, offm := hom, offn := hon, query := entry_query Q K V L O p s₀,
           maximum := ?_, denominator := ?_, accumulator := ?_, key := ?_, value := ?_,
           lo := hlo, hi := hhi }
  · simpa [prefixMax] using hmax
  · simpa [prefixDen, RealTiles.finite] using hden
  · simpa [prefixAcc, RealTiles.finite] using hacc
  · simpa using hk
  · simpa using hv

theorem loaded_mem_eq {shape : TileShape} (s t : BlockState) (hm : t.mem = s.mem)
    (ptr : Tile .blockPtr shape) : loaded t ptr = loaded s ptr := by
  apply Tile.ext
  intro i
  simp [loaded, BlockState.readMemValue_real, BlockState.readMem, hm]

theorem prefix_update (p : Layout) (s : BlockState) (Q K V : RegionName) (c : Nat) :
    let sc := scores p s (c*p.BLOCK_N) (Tile.vec (fun i => row p (s.pids 0) i.val))
      (Tile.vec (fun j => j.val)) (query p s Q) (keyPointer p K (s.pids 1) (c*p.BLOCK_N))
    let mn := updatedMax p (prefixMax p s Q K V c) sc
    let alpha := rescale p (prefixMax p s Q K V c) mn
    let prob := probabilities p sc mn
    mn = prefixMax p s Q K V (c+1) ∧
      updatedDenom p (prefixDen p s Q K V c) alpha prob = prefixDen p s Q K V (c+1) ∧
      updatedAcc p s (prefixDen p s Q K V c) alpha (prefixAcc p s Q K V c) prob
        (valuePointer p V (s.pids 1) (c*p.BLOCK_N)) = prefixAcc p s Q K V (c+1) := by
  let f := fun i : Fin p.BLOCK_M => fun j : Fin p.BLOCK_N =>
    score p s.readMem Q K (s.pids 0) (s.pids 1) i.val (c*p.BLOCK_N+j.val)
  let keep := fun i : Fin p.BLOCK_M => fun j : Fin p.BLOCK_N =>
    visible p (s.pids 0) i.val (c*p.BLOCK_N+j.val)
  let z := fun (i d : Nat) => prefixSummary p s.readMem Q K V (s.pids 0) (s.pids 1) i d c
  let old := fun i : Fin p.BLOCK_M => (z i.val 0).center
  let den := fun i : Fin p.BLOCK_M => (z i.val 0).denominator
  let ac := fun i : TileIndex [p.BLOCK_M,p.DIM] => (z i.1.val i.2.1.val).numerator
  let v := fun i : TileIndex [p.BLOCK_N,p.DIM] =>
    s.readMem V (vOffset p (s.pids 1) (c*p.BLOCK_N+i.1.val) i.2.1.val)
  have hsc : scores p s (c*p.BLOCK_N) (Tile.vec (fun i => row p (s.pids 0) i.val))
      (Tile.vec (fun j => j.val)) (query p s Q) (keyPointer p K (s.pids 1) (c*p.BLOCK_N)) =
        maskedTile p f keep := source_scores p s Q K (c*p.BLOCK_N)
  have hstep (i d : Nat) := prefixSummary_succ p s.readMem Q K V (s.pids 0) (s.pids 1) i d c
  dsimp only
  rw [hsc]
  refine ⟨?_, ?_, ?_⟩
  · change updatedMax p ⟨fun i => old i.1⟩ (maskedTile p f keep) = _
    rw [maximum_matches]
    apply Tile.ext
    rintro ⟨i, ⟨⟩⟩
    change RealTiles.center f keep old i = _
    have h := congrArg RealFormula.RowState.center (hstep i.val 0)
    rw [blockPairs_finRange] at h
    exact (RealTiles.center_matches_summary f keep old i (den i) (z i.val 0).numerator
      (fun j => s.readMem V (vOffset p (s.pids 1) (c*p.BLOCK_N+j.val) 0))).trans h
  · apply Tile.ext
    rintro ⟨i, ⟨⟩⟩
    have h := congrArg (fun t : RealFormula.RowState => some t.denominator) (hstep i.val 0)
    rw [blockPairs_finRange] at h
    exact (denominator_summary p f keep old den i
      (fun j => s.readMem V (vOffset p (s.pids 1) (c*p.BLOCK_N+j.val) 0))
      (z i.val 0).numerator).trans h
  · apply Tile.ext
    rintro ⟨i, d, ⟨⟩⟩
    have h := congrArg (fun t : RealFormula.RowState => some t.numerator) (hstep i.val d.val)
    rw [blockPairs_finRange] at h
    have hv : loaded s (valuePointer p V (s.pids 1) (c*p.BLOCK_N)) = RealTiles.finite v :=
      loaded_valueTile p s V (s.pids 1) (c*p.BLOCK_N)
    have hn := accumulator_summary p s f keep old den ac v
      (valuePointer p V (s.pids 1) (c*p.BLOCK_N)) hv i d
    have hind := prefixSummary_value_independent p s.readMem Q K V
      (s.pids 0) (s.pids 1) i.val 0 d.val c
    dsimp only [old,den,ac,z] at hn
    rw [hind.1, hind.2] at hn
    exact hn.trans h

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.FlashAttn.RealExecution
