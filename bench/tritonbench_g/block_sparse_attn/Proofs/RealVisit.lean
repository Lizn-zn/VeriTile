import bench.tritonbench_g.block_sparse_attn.Proofs.RealScore
import bench.tritonbench_g.block_sparse_attn.Proofs.RealLoopTail

/-! Execute one entire actual CSR visit, from the column read through score
production, normalization, V loads and carry writes. The hypotheses describe
only incoming registers; no score register, loaded K/V tile, or whole-body
execution is assumed. The prologue and induction over signed visits remain
separate obligations. -/
namespace VeriTile.Bench.TritonBenchG.BlockSparseAttn.RealVisit
open VeriTile.Triton RealNormalizationValues
noncomputable section

/-- Values supplied by the prologue and the signed range iterator. -/
structure Operands (BM BN BD : Nat) where
  layout : Nat
  visit : Int
  keyPointers : Tile .ptr [BD, BN]
  valuePointers : Tile .ptr [BN, BD]
  rows : Fin BM → Nat
  query : Fin BM → Fin BD → ℝ
  secondQuery : Fin BM → Fin BD → ℝ

def start {BM BN BD : Nat} (s : BlockState) (o : Operands BM BN BD)
    (p : RealProgram.Parameters) : Int :=
  RealColumn.column s p.cols o.layout p.columnStride o.visit * (BN : Int)

def scored {BM BN BD : Nat} (s : BlockState) (o : Operands BM BN BD)
    (p : RealProgram.Parameters) : BlockState :=
  RealScore.result (RealColumn.result s BN p.cols o.layout p.columnStride o.visit)
    o.keyPointers (start s o p) p.kRow p.context p.blocks p.scale p.evenN
    o.rows o.query o.secondQuery

/-- Replace the unused incoming score with the score actually produced by
this visit. The other fields are the incoming normalization carry. -/
def inputs {BM BN BD : Nat} (s : BlockState) (o : Operands BM BN BD)
    (p : RealProgram.Parameters) (carry : Inputs BM BN BD) : Inputs BM BN BD :=
  let score := RealScore.masked
    (fun r j => RealScore.unscaled
      (RealColumn.result s BN p.cols o.layout p.columnStride o.visit)
      o.keyPointers (start s o p) p.kRow p.context p.blocks p.evenN o.query o.secondQuery r j * p.scale)
    (start s o p) o.rows
  {carry with score := score}

def result {BM BN BD : Nat} (s : BlockState) (o : Operands BM BN BD)
    (p : RealProgram.Parameters) (carry : Inputs BM BN BD) : BlockState :=
  RealLoopTail.result (scored s o p) (inputs s o p carry) o.valuePointers
    (start s o p) p.vRow p.context p.blocks p.evenN

theorem scored_frame {BM BN BD : Nat} (s : BlockState) (o : Operands BM BN BD)
    (p : RealProgram.Parameters) : {scored s o p with regs := s.regs} = s := by
  have h := RealScore.state_frame (RealColumn.result s BN p.cols o.layout p.columnStride o.visit)
    o.keyPointers (start s o p) p.kRow p.context p.blocks p.scale p.evenN o.rows o.query o.secondQuery
  exact (congrArg (fun t => {t with regs := s.regs}) h).trans
    (RealColumn.state_frame s BN p.cols o.layout p.columnStride o.visit)

theorem scored_register_frame {BM BN BD : Nat} (s : BlockState) (o : Operands BM BN BD)
    (p : RealProgram.Parameters) (dtype : TileDType) (shape : TileShape) (name : RegName)
    (hn : name ∉ ["col_idx", "start_n", "k", "qk"]) :
    (scored s o p).regs dtype shape name = s.regs dtype shape name := by
  simp only [List.mem_cons, List.not_mem_nil, or_false, not_or] at hn
  rw [scored, RealScore.register_frame _ _ _ _ _ _ _ _ _ _ _ _ _ _ hn.2.2.1 hn.2.2.2,
    RealColumn.register_frame _ _ _ _ _ _ _ _ _ hn.1 hn.2.1]

theorem score_readback {BM BN BD : Nat} (s : BlockState) (o : Operands BM BN BD)
    (p : RealProgram.Parameters) (carry : Inputs BM BN BD) :
    (scored s o p).regs .real [BM, BN] "qk" = some (extendedMatrix (inputs s o p carry).score) :=
  RealScore.readback ..

theorem start_readback {BM BN BD : Nat} (s : BlockState) (o : Operands BM BN BD)
    (p : RealProgram.Parameters) :
    (scored s o p).regs .int [] "start_n" = some (Tile.scalar (start s o p)) := by
  rw [scored, RealScore.register_frame _ _ _ _ _ _ _ _ _ _ _ _ _ _ (by decide) (by decide)]
  exact (RealColumn.readbacks s BN p.cols o.layout p.columnStride o.visit).2

/-- The full source body succeeds from its incoming loop registers. The
carry hypotheses will be supplied by induction, and the fixed Q/pointer
hypotheses by the prologue. Both scaling directions use this same proof. -/
theorem source_executes {BM BN BD : Nat} (swapped : Bool) (hN : 0 < BN)
    (s : BlockState) (o : Operands BM BN BD) (p : RealProgram.Parameters) (carry : Inputs BM BN BD)
    (hlayout : s.regs .nat [] "layout_h" = some (Tile.scalar o.layout))
    (hvisit : s.regs .int [] "col_idx_idx" = some (Tile.scalar o.visit))
    (hk : s.regs .ptr [BD, BN] "k_ptrs" = some o.keyPointers)
    (hv : s.regs .ptr [BN, BD] "v_ptrs" = some o.valuePointers)
    (hm : s.regs .nat [BM] "offs_m" = some (Tile.vec o.rows))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec Fin.val))
    (hq : s.regs .real [BM, BD] "q" = some (matrix o.query))
    (hq2 : 2 ≤ p.blocks → s.regs .real [BM, BD] "q2" = some (matrix o.secondQuery))
    (hmax : s.regs .real [BM] "m_i" = some (extendedRow carry.maximum))
    (hnorm : s.regs .real [BM] "l_i" = some (row carry.normalizer))
    (hacc : s.regs .real [BM, BD] "acc" = some (matrix carry.accumulator))
    (hacc2 : 2 ≤ p.blocks → s.regs .real [BM, BD] "acc2" = some (matrix carry.secondAccumulator)) :
    stepStmts (RealProgram.realPart (RealProgram.parts swapped BM BN BD p).body) s =
      some (result s o p carry) := by
  rw [RealProgram.loop_split, List.append_assoc]
  have hs := RealScore.source_executes swapped BM BN BD p s o.layout o.visit o.keyPointers
    o.rows o.query o.secondQuery hlayout hvisit hk hm hn hq hq2
  rw [stepStmts.append_some hs, RealNormalizationSource.matches_source, RealAccumulation.matches_source]
  change stepStmts (RealNormalizationSource.code BM BN BD p.blocks ++
      RealAccumulation.code BM BN BD p.vRow p.context p.blocks p.evenN) (scored s o p) = _
  apply RealLoopTail.executes hN
  · rw [scored_register_frame s o p .ptr [BN, BD] "v_ptrs" (by simp)]
    exact hv
  · exact start_readback s o p
  · rw [scored_register_frame s o p .nat [BN] "offs_n" (by simp)]
    exact hn
  · exact score_readback s o p carry
  · rw [scored_register_frame s o p .real [BM] "m_i" (by simp)]
    exact hmax
  · rw [scored_register_frame s o p .real [BM] "l_i" (by simp)]
    exact hnorm
  · rw [scored_register_frame s o p .real [BM, BD] "acc" (by simp)]
    exact hacc
  · intro hb
    rw [scored_register_frame s o p .real [BM, BD] "acc2" (by simp)]
    exact hacc2 hb

theorem state_frame {BM BN BD : Nat} (s : BlockState) (o : Operands BM BN BD)
    (p : RealProgram.Parameters) (carry : Inputs BM BN BD) :
    {result s o p carry with regs := s.regs} = s := by
  have h := RealLoopTail.state_frame (scored s o p) (inputs s o p carry) o.valuePointers
    (start s o p) p.vRow p.context p.blocks p.evenN
  exact (congrArg (fun t => {t with regs := s.regs}) h).trans (scored_frame s o p)

def writes : List RegName :=
  ["col_idx", "start_n", "k", "qk"] ++ RealNormalization.writes ++ RealAccumulation.writes

theorem register_frame {BM BN BD : Nat} (s : BlockState) (o : Operands BM BN BD)
    (p : RealProgram.Parameters) (carry : Inputs BM BN BD)
    (dtype : TileDType) (shape : TileShape) (name : RegName) (hn : name ∉ writes) :
    (result s o p carry).regs dtype shape name = s.regs dtype shape name := by
  simp only [writes, List.mem_append, not_or] at hn
  have hs := hn.1.1
  have ht : name ∉ RealNormalization.writes ++ RealAccumulation.writes := by
    simpa only [List.mem_append, not_or] using And.intro hn.1.2 hn.2
  rw [result, RealLoopTail.register_frame _ _ _ _ _ _ _ _ _ _ _ ht,
    scored_register_frame _ _ _ _ _ _ hs]

/-- Score production does not change the memory or arbitrary values read by
masked V lanes. This lets the induction express all loaded tiles at entry. -/
theorem loaded_scored {BM BN BD : Nat} (s : BlockState) (o : Operands BM BN BD)
    (p : RealProgram.Parameters) (second : Bool) :
    RealValueLoad.loaded (scored s o p) o.valuePointers (start s o p) p.vRow p.context p.evenN second =
      RealValueLoad.loaded s o.valuePointers (start s o p) p.vRow p.context p.evenN second := by
  have hm := congrArg BlockState.mem (scored_frame s o p)
  have hu := congrArg BlockState.undef (scored_frame s o p)
  change (scored s o p).mem = s.mem at hm
  change (scored s o p).undef = s.undef at hu
  funext j d
  simp only [RealValueLoad.loaded, BlockState.readMem, hm, hu]

theorem outputs {BM BN BD : Nat} (s : BlockState) (o : Operands BM BN BD)
    (p : RealProgram.Parameters) (carry : Inputs BM BN BD) :
    RealAccumulation.Outputs s (inputs s o p carry) o.valuePointers (start s o p)
      p.vRow p.context p.blocks p.evenN (result s o p carry) := by
  have h := RealLoopTail.outputs (scored s o p) (inputs s o p carry) o.valuePointers
    (start s o p) p.vRow p.context p.blocks p.evenN
  exact ⟨h.maximum, h.normalizer, by simpa only [loaded_scored] using h.accumulator,
    fun hb => by simpa only [loaded_scored] using h.secondAccumulator hb⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BlockSparseAttn.RealVisit
