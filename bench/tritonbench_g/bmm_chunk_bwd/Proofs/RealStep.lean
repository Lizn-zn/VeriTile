import bench.tritonbench_g.bmm_chunk_bwd.Proofs.RealDot

/-! One complete contraction iteration of the actual source, including both
masked loads, the dot/add and both pointer advances. Register hypotheses are
local loop-invariant facts; no execution or precomputed dot is assumed. -/
namespace VeriTile.Bench.TritonBenchG.BmmChunkBwd.RealStep
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

def advance {shape : TileShape} (ptrs : Tile .ptr shape) (offset : Nat) : Tile .ptr shape :=
  ⟨fun i => ((ptrs.data i).1, (ptrs.data i).2 + offset)⟩

def result {BM BN BCS : Nat} (s : BlockState) (p : RealProgram.Parameters)
    (dp : Tile .ptr [BM, BCS]) (ap : Tile .ptr [BCS, BN])
    (rows : Fin BM → Nat) (columns : Fin BN → Nat) (length c : Nat)
    (carry : Fin BM → Fin BN → ℝ) : BlockState :=
  let t := s.setReg "dout" .real [BM, BCS] (RealLoads.doutValues s dp rows p.chunk length c)
  let t := t.setReg "a" .real [BCS, BN] (RealLoads.aValues s ap columns p.K length c)
  let t := t.setReg "acc" .real [BM, BN] (Tile.mat fun i j => some
    (carry i j + RealDot.value s dp ap rows columns p.chunk p.K length c i j))
  let t := t.setReg "dout_ptrs" .ptr [BM, BCS] (advance dp (BCS * p.dRow))
  t.setReg "a_ptrs" .ptr [BCS, BN] (advance ap (BCS * p.aRow))

theorem executes (BM BN BCS : Nat) (s : BlockState) (p : RealProgram.Parameters)
    (dp : Tile .ptr [BM, BCS]) (ap : Tile .ptr [BCS, BN])
    (rows : Fin BM → Nat) (columns : Fin BN → Nat) (length c : Nat)
    (carry : Fin BM → Fin BN → ℝ)
    (hdp : s.regs .ptr [BM, BCS] "dout_ptrs" = some dp)
    (hap : s.regs .ptr [BCS, BN] "a_ptrs" = some ap)
    (hr : s.regs .nat [BM] "offs_m" = some (Tile.vec rows))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec columns))
    (hk : s.regs .nat [BCS] "offs_cs" = some (Tile.vec Fin.val))
    (hl : s.regs .nat [] "chunk_size_limit" = some (Tile.scalar length))
    (hc : s.regs .nat [] "cs" = some (Tile.scalar c))
    (hz : s.regs .real [BM, BN] "acc" = some (Tile.mat fun i j => some (carry i j))) :
    stepStmts (RealLoads.code BM BN BCS p) s =
      some (result s p dp ap rows columns length c carry) := by
  let sd := s.setReg "dout" .real [BM, BCS] (RealLoads.doutValues s dp rows p.chunk length c)
  let sa := sd.setReg "a" .real [BCS, BN] (RealLoads.aValues s ap columns p.K length c)
  let sz := sa.setReg "acc" .real [BM, BN] (Tile.mat fun i j => some
    (carry i j + RealDot.value s dp ap rows columns p.chunk p.K length c i j))
  let sp := sz.setReg "dout_ptrs" .ptr [BM, BCS] (advance dp (BCS * p.dRow))
  have hd := RealLoads.dout_load s BM BCS p.chunk length c rows dp hdp hr hk hl hc
  have ha : evalOp (RealLoads.aLoad BN BCS p.K) sd =
      some (RealLoads.aValues s ap columns p.K length c) := by
    apply RealLoads.a_load sd BN BCS p.K length c columns ap
    all_goals simpa [sd] using (by assumption)
  have hadd : evalOp (.add .real (.consSame (.consSame .nil)) (.ref .real [BM, BN] "acc")
      (.dot (batch := []) (.ref .real [BM, BCS] "dout") (.ref .real [BCS, BN] "a"))) sa =
      some (Tile.mat fun i j => some
        (carry i j + RealDot.value s dp ap rows columns p.chunk p.K length c i j)) := by
    apply RealDot.accumulation_eval BM BN BCS sa dp ap rows columns p.chunk p.K length c carry
    · simpa [sa, sd] using hz
    · simp [sa, sd]
      rfl
    · simp [sa]
      rfl
  have hdptr : evalOp (.ptrAdd .scalarR (.ref .ptr [BM, BCS] "dout_ptrs")
      (.mul .nat .nil (.constNat BCS) (.constNat p.dRow))) sz = some (advance dp (BCS * p.dRow)) := by
    simp [evalOp.eq_def, sz, sa, sd, hdp, bind, Option.bind]
    rfl
  have haptr : evalOp (.ptrAdd .scalarR (.ref .ptr [BCS, BN] "a_ptrs")
      (.mul .nat .nil (.constNat BCS) (.constNat p.aRow))) sp = some (advance ap (BCS * p.aRow)) := by
    simp [evalOp.eq_def, sp, sz, sa, sd, hap, bind, Option.bind]
    rfl
  dsimp only [sp, sz, sa, sd] at ha hadd hdptr haptr
  simp only [RealLoads.code, stepStmts, stepStmt, hd, ha, hadd, hdptr, haptr,
    bind, Option.bind_some]
  rfl

theorem state_frame {BM BN BCS : Nat} (s : BlockState) (p : RealProgram.Parameters)
    (dp : Tile .ptr [BM, BCS]) (ap : Tile .ptr [BCS, BN])
    (rows : Fin BM → Nat) (columns : Fin BN → Nat) (length c : Nat)
    (carry : Fin BM → Fin BN → ℝ) :
    {result s p dp ap rows columns length c carry with regs := s.regs} = s := by
  cases s
  rfl

/-- The loop leaves the runtime bound, lane indices and all unrelated
registers intact for the next iteration. -/
theorem register_frame {BM BN BCS : Nat} (s : BlockState) (p : RealProgram.Parameters)
    (dp : Tile .ptr [BM, BCS]) (ap : Tile .ptr [BCS, BN])
    (rows : Fin BM → Nat) (columns : Fin BN → Nat) (length c : Nat)
    (carry : Fin BM → Fin BN → ℝ) (dtype : TileDType) (shape : TileShape) (name : RegName)
    (hn : name ∉ ["dout", "a", "acc", "dout_ptrs", "a_ptrs"]) :
    (result s p dp ap rows columns length c carry).regs dtype shape name =
      s.regs dtype shape name := by
  simp only [List.mem_cons, List.not_mem_nil, or_false, not_or] at hn
  rcases hn with ⟨hd, ha, hz, hdp, hap⟩
  simp [result, BlockState.setReg, hd, ha, hz, hdp, hap]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmChunkBwd.RealStep
