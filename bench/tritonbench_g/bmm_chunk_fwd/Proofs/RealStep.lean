import bench.tritonbench_g.bmm_chunk_fwd.Proofs.RealDot

/-! One complete contraction iteration of the actual source, including both
masked loads, the dot/add and both pointer advances. Register hypotheses are
local loop-invariant facts; no execution or precomputed dot is assumed. -/
namespace VeriTile.Bench.TritonBenchG.BmmChunkFwd.RealStep
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

def advance {shape : TileShape} (ptrs : Tile .ptr shape) (offset : Nat) : Tile .ptr shape :=
  ⟨fun i => ((ptrs.data i).1, (ptrs.data i).2 + offset)⟩

def result {BM BN BK : Nat} (s : BlockState) (p : RealProgram.Parameters)
    (dp : Tile .ptr [BM, BK]) (ap : Tile .ptr [BK, BN])
    (rows : Fin BM → Nat) (columns : Fin BN → Nat) (length c : Nat)
    (carry : Fin BM → Fin BN → ℝ) : BlockState :=
  let t := s.setReg "a" .real [BM, BK] (RealLoads.aValues s dp rows length p.K c)
  let t := t.setReg "b" .real [BK, BN] (RealLoads.bValues s ap columns length p.K c)
  let t := t.setReg "acc" .real [BM, BN] (Tile.mat fun i j => some
    (carry i j + RealDot.value s dp ap rows columns length p.K c i j))
  let t := t.setReg "a_ptrs" .ptr [BM, BK] (advance dp (BK * p.aColumn))
  t.setReg "b_ptrs" .ptr [BK, BN] (advance ap (BK * p.bColumn))

theorem executes (swapped : Bool) (BM BN BK : Nat) (s : BlockState) (p : RealProgram.Parameters)
    (dp : Tile .ptr [BM, BK]) (ap : Tile .ptr [BK, BN])
    (rows : Fin BM → Nat) (columns : Fin BN → Nat) (length c : Nat)
    (carry : Fin BM → Fin BN → ℝ)
    (hdp : s.regs .ptr [BM, BK] "a_ptrs" = some dp)
    (hap : s.regs .ptr [BK, BN] "b_ptrs" = some ap)
    (hr : s.regs .nat [BM] "offs_m" = some (Tile.vec rows))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec columns))
    (hk : s.regs .nat [BK] "offs_k" = some (Tile.vec Fin.val))
    (hl : s.regs .nat [] "chunk_size_limit" = some (Tile.scalar length))
    (hc : s.regs .nat [] "k" = some (Tile.scalar c))
    (hz : s.regs .real [BM, BN] "acc" = some (Tile.mat fun i j => some (carry i j))) :
    stepStmts (RealLoads.code swapped BM BN BK p) s =
      some (result s p dp ap rows columns length c carry) := by
  let sd := s.setReg "a" .real [BM, BK] (RealLoads.aValues s dp rows length p.K c)
  let sa := sd.setReg "b" .real [BK, BN] (RealLoads.bValues s ap columns length p.K c)
  let sz := sa.setReg "acc" .real [BM, BN] (Tile.mat fun i j => some
    (carry i j + RealDot.value s dp ap rows columns length p.K c i j))
  let sp := sz.setReg "a_ptrs" .ptr [BM, BK] (advance dp (BK * p.aColumn))
  have hd := RealLoads.a_load s BM BK p.K length c rows dp hdp hr hk hl hc
  have ha : evalOp (RealLoads.bLoad BN BK p.K) sd =
      some (RealLoads.bValues s ap columns length p.K c) := by
    apply RealLoads.b_load sd BN BK p.K length c columns ap
    all_goals simpa [sd] using (by assumption)
  have hadd : evalOp (RealLoads.update swapped BM BN BK) sa =
      some (Tile.mat fun i j => some
        (carry i j + RealDot.value s dp ap rows columns length p.K c i j)) := by
    apply RealDot.accumulation_eval swapped BM BN BK sa dp ap rows columns length p.K c carry
    · simpa [sa, sd] using hz
    · simp [sa, sd]
      rfl
    · simp [sa]
      rfl
  have hdptr : evalOp (.ptrAdd .scalarR (.ref .ptr [BM, BK] "a_ptrs")
      (.mul .nat .nil (.constNat BK) (.constNat p.aColumn))) sz = some (advance dp (BK * p.aColumn)) := by
    simp [evalOp.eq_def, sz, sa, sd, hdp, bind, Option.bind]
    rfl
  have haptr : evalOp (.ptrAdd .scalarR (.ref .ptr [BK, BN] "b_ptrs")
      (.mul .nat .nil (.constNat BK) (.constNat p.bColumn))) sp = some (advance ap (BK * p.bColumn)) := by
    simp [evalOp.eq_def, sp, sz, sa, sd, hap, bind, Option.bind]
    rfl
  dsimp only [sp, sz, sa, sd] at ha hadd hdptr haptr
  simp only [RealLoads.code, stepStmts, stepStmt, hd, ha, hadd, hdptr, haptr,
    bind, Option.bind_some]
  rfl

theorem state_frame {BM BN BK : Nat} (s : BlockState) (p : RealProgram.Parameters)
    (dp : Tile .ptr [BM, BK]) (ap : Tile .ptr [BK, BN])
    (rows : Fin BM → Nat) (columns : Fin BN → Nat) (length c : Nat)
    (carry : Fin BM → Fin BN → ℝ) :
    {result s p dp ap rows columns length c carry with regs := s.regs} = s := by
  cases s
  rfl

/-- The loop leaves the runtime bound, lane indices and all unrelated
registers intact for the next iteration. -/
theorem register_frame {BM BN BK : Nat} (s : BlockState) (p : RealProgram.Parameters)
    (dp : Tile .ptr [BM, BK]) (ap : Tile .ptr [BK, BN])
    (rows : Fin BM → Nat) (columns : Fin BN → Nat) (length c : Nat)
    (carry : Fin BM → Fin BN → ℝ) (dtype : TileDType) (shape : TileShape) (name : RegName)
    (hn : name ∉ ["a", "b", "acc", "a_ptrs", "b_ptrs"]) :
    (result s p dp ap rows columns length c carry).regs dtype shape name =
      s.regs dtype shape name := by
  simp only [List.mem_cons, List.not_mem_nil, or_false, not_or] at hn
  rcases hn with ⟨hd, ha, hz, hdp, hap⟩
  simp [result, BlockState.setReg, hd, ha, hz, hdp, hap]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmChunkFwd.RealStep
