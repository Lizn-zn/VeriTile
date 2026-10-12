import bench.tritonbench_g.bmm_chunk_bwd.Proofs.RealStep
import bench.tritonbench_g.bmm_chunk_bwd.Proofs.RealSetup

/-! Execute the complete dynamic contraction loop from the actual setup.
The invariant relates pointers and the accumulator to initial memory, keeps
all other registers and state fields, and counts the source's ceiling-divided
iterations. Partial final blocks are included through the actual load masks.
No loop execution or precomputed product is assumed. -/
namespace VeriTile.Bench.TritonBenchG.BmmChunkBwd.RealLoop
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false

def dPointers (s₀ : BlockState) (BM BN BCS : Nat) (p : RealProgram.Parameters)
    (c : Nat) : Tile .ptr [BM, BCS] :=
  Tile.mat fun i k => (p.Dout, RealSetup.dBase s₀ p +
    (RealSetup.rows s₀ BM BN p i * p.dColumn + (c * BCS + k.val) * p.dRow))

def aPointers (s₀ : BlockState) (BN BCS : Nat) (p : RealProgram.Parameters)
    (c : Nat) : Tile .ptr [BCS, BN] :=
  Tile.mat fun k j => (p.A, RealSetup.aBase s₀ p +
    ((c * BCS + k.val) * p.aRow + RealSetup.columns s₀ BN p j * p.aColumn))

def product (s₀ : BlockState) (BM BN : Nat) (p : RealProgram.Parameters)
    (i : Fin BM) (j : Fin BN) (k : Nat) : ℝ :=
  s₀.readMem p.Dout (RealSetup.dBase s₀ p + (RealSetup.rows s₀ BM BN p i * p.dColumn + k * p.dRow)) *
  s₀.readMem p.A (RealSetup.aBase s₀ p + (k * p.aRow + RealSetup.columns s₀ BN p j * p.aColumn))

def carry (s₀ : BlockState) (BM BN BCS : Nat) (p : RealProgram.Parameters)
    (c : Nat) (i : Fin BM) (j : Fin BN) : ℝ :=
  if RealSetup.rows s₀ BM BN p i < p.chunk ∧ RealSetup.columns s₀ BN p j < p.K then
    RealMath.partialSum (RealSetup.limit s₀ p) BCS c (product s₀ BM BN p i j)
  else 0

def writes : List RegName := ["cs", "dout", "a", "acc", "dout_ptrs", "a_ptrs"]

structure Holds (s₀ : BlockState) (BM BN BCS : Nat) (p : RealProgram.Parameters)
    (c : Nat) (s : BlockState) : Prop where
  count : c ≤ RealMath.blocks (RealSetup.limit s₀ p) BCS
  mem : s.mem = s₀.mem
  undef : s.undef = s₀.undef
  pids : s.pids = s₀.pids
  numPids : s.numPids = s₀.numPids
  fixed : ∀ dtype shape name, name ∉ writes → s.regs dtype shape name =
    (RealSetup.result s₀ BM BN BCS p).regs dtype shape name
  dout : s.regs .ptr [BM, BCS] "dout_ptrs" = some (dPointers s₀ BM BN BCS p c)
  a : s.regs .ptr [BCS, BN] "a_ptrs" = some (aPointers s₀ BN BCS p c)
  acc : s.regs .real [BM, BN] "acc" = some (Tile.mat fun i j => some (carry s₀ BM BN BCS p c i j))

theorem initial (s₀ : BlockState) (BM BN BCS : Nat) (p : RealProgram.Parameters) :
    Holds s₀ BM BN BCS p 0 (RealSetup.result s₀ BM BN BCS p) := by
  refine ⟨Nat.zero_le _, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp [RealSetup.result, BlockState.setReg]
  · simp [RealSetup.result, BlockState.setReg]
  · simp [RealSetup.result, BlockState.setReg]
  · simp [RealSetup.result, BlockState.setReg]
  · intros; rfl
  · simp [RealSetup.result, dPointers, RealSetup.dPointers]
  · simp [RealSetup.result, aPointers, RealSetup.aPointers]
  · simp [RealSetup.result, carry, RealMath.partial_zero]

theorem d_advance (s₀ : BlockState) (BM BN BCS : Nat) (p : RealProgram.Parameters) (c : Nat) :
    RealStep.advance (dPointers s₀ BM BN BCS p c) (BCS * p.dRow) =
      dPointers s₀ BM BN BCS p (c + 1) := by
  apply Tile.ext
  rintro ⟨i, k, ⟨⟩⟩
  simp only [RealStep.advance, dPointers, Tile.mat, Prod.mk.injEq, true_and]
  ring

theorem a_advance (s₀ : BlockState) (BN BCS : Nat) (p : RealProgram.Parameters) (c : Nat) :
    RealStep.advance (aPointers s₀ BN BCS p c) (BCS * p.aRow) =
      aPointers s₀ BN BCS p (c + 1) := by
  apply Tile.ext
  rintro ⟨k, j, ⟨⟩⟩
  simp only [RealStep.advance, aPointers, Tile.mat, Prod.mk.injEq, true_and]
  ring

theorem carry_succ (s₀ s : BlockState) (BM BN BCS : Nat) (p : RealProgram.Parameters)
    (c : Nat) (hm : s.mem = s₀.mem) (i : Fin BM) (j : Fin BN) :
    carry s₀ BM BN BCS p c i j +
      RealDot.value s (dPointers s₀ BM BN BCS p c) (aPointers s₀ BN BCS p c)
        (RealSetup.rows s₀ BM BN p) (RealSetup.columns s₀ BN p) p.chunk p.K
        (RealSetup.limit s₀ p) c i j = carry s₀ BM BN BCS p (c + 1) i j := by
  simp only [carry, RealDot.value]
  split
  · rw [RealMath.partial_succ]
    congr 1
    apply Finset.sum_congr rfl
    intro k _
    simp only [dPointers, aPointers, Tile.mat, product, BlockState.readMem, hm]
  · simp


theorem iteration (s₀ : BlockState) (BM BN BCS : Nat) (p : RealProgram.Parameters)
    (c : Nat) (s : BlockState) (hc : c < RealMath.blocks (RealSetup.limit s₀ p) BCS)
    (h : Holds s₀ BM BN BCS p c s) :
    ∃ t, stepStmts (RealLoads.code BM BN BCS p) (s.setReg "cs" .nat [] (Tile.scalar c)) = some t ∧
      Holds s₀ BM BN BCS p (c + 1) t := by
  let entry := s.setReg "cs" .nat [] (Tile.scalar c)
  have hr : entry.regs .nat [BM] "offs_m" = some (Tile.vec (RealSetup.rows s₀ BM BN p)) := by
    have hv := h.fixed .nat [BM] "offs_m" (by decide)
    simpa [entry, RealSetup.result] using hv
  have hn : entry.regs .nat [BN] "offs_n" = some (Tile.vec (RealSetup.columns s₀ BN p)) := by
    have hv := h.fixed .nat [BN] "offs_n" (by decide)
    simpa [entry, RealSetup.result] using hv
  have hk : entry.regs .nat [BCS] "offs_cs" = some (Tile.vec Fin.val) := by
    have hv := h.fixed .nat [BCS] "offs_cs" (by decide)
    simpa [entry, RealSetup.result] using hv
  have hl : entry.regs .nat [] "chunk_size_limit" = some (Tile.scalar (RealSetup.limit s₀ p)) := by
    have hv := h.fixed .nat [] "chunk_size_limit" (by decide)
    simpa [entry, RealSetup.result] using hv
  have he := RealStep.executes BM BN BCS entry p (dPointers s₀ BM BN BCS p c)
    (aPointers s₀ BN BCS p c) (RealSetup.rows s₀ BM BN p) (RealSetup.columns s₀ BN p)
    (RealSetup.limit s₀ p) c (carry s₀ BM BN BCS p c)
    (by simpa [entry] using h.dout) (by simpa [entry] using h.a)
    hr hn hk hl (by simp [entry]) (by simpa [entry] using h.acc)
  refine ⟨_, he, Nat.succ_le_of_lt hc, h.mem, h.undef, h.pids, h.numPids, ?_, ?_, ?_, ?_⟩
  · intro dtype shape name hname
    have hx : name ∉ ["dout", "a", "acc", "dout_ptrs", "a_ptrs"] := by
      have hs := hname
      simp only [writes, List.mem_cons, not_or] at hs
      simpa only [List.mem_cons, not_or] using hs.2
    rw [RealStep.register_frame _ _ _ _ _ _ _ _ _ _ _ _ hx]
    have hcs : name ≠ "cs" := by
      have hs := hname
      simp only [writes, List.mem_cons, not_or] at hs
      exact hs.1
    simpa [entry, BlockState.setReg, hcs] using h.fixed dtype shape name hname
  · simp [RealStep.result, d_advance]
  · simp [RealStep.result, a_advance]
  · simp only [RealStep.result, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, BlockState.setReg_same]
    congr 1
    apply Tile.ext
    rintro ⟨i,j,⟨⟩⟩
    exact congrArg some (carry_succ s₀ entry BM BN BCS p c h.mem i j)


/-- The source's dynamic contraction loop, with its actual register bound. -/
def statement (BM BN BCS : Nat) (p : RealProgram.Parameters) : Stmt :=
  .forRangeDyn "cs" (.constNat 0) (RealProgram.loopBound BCS) (.constNat 1)
    (RealLoads.code BM BN BCS p)

theorem executes (s₀ : BlockState) (BM BN BCS : Nat) (p : RealProgram.Parameters) :
    ∃ t, stepStmt (statement BM BN BCS p) (RealSetup.result s₀ BM BN BCS p) = some t ∧
      Holds s₀ BM BN BCS p (RealMath.blocks (RealSetup.limit s₀ p) BCS) t := by
  have resolve : stepStmt (statement BM BN BCS p) (RealSetup.result s₀ BM BN BCS p) =
      stepForRangeAux "cs" 0 (RealMath.blocks (RealSetup.limit s₀ p) BCS) 1
        (RealLoads.code BM BN BCS p) (RealSetup.result s₀ BM BN BCS p) := by
    simp only [statement, stepForRangeAux.forRangeDyn_unfold, RealProgram.loopBound,
      evalOp.eq_def, RealSetup.chunk_limit, bind, Option.bind_some]
    rfl
  rw [resolve]
  obtain ⟨final, t, he, hf, ht⟩ := forRangeAux_inv
    (idx := "cs") (stop := RealMath.blocks (RealSetup.limit s₀ p) BCS) (step := 1)
    (P := Holds s₀ BM BN BCS p) (by decide)
    (fun c s hc h => iteration s₀ BM BN BCS p c s hc h)
    0 (RealSetup.result s₀ BM BN BCS p) (initial s₀ BM BN BCS p)
  have final_eq : final = RealMath.blocks (RealSetup.limit s₀ p) BCS := Nat.le_antisymm ht.count hf
  subst final
  exact ⟨t, he, ht⟩

/-- Readout is a sum over the initial memory, including a partial last block. -/
theorem full_sum (s₀ s : BlockState) (BM BN BCS : Nat) (p : RealProgram.Parameters)
    (hBCS : 0 < BCS)
    (h : Holds s₀ BM BN BCS p (RealMath.blocks (RealSetup.limit s₀ p) BCS) s) :
    s.regs .real [BM, BN] "acc" = some (Tile.mat fun i j => some
      (if RealSetup.rows s₀ BM BN p i < p.chunk ∧ RealSetup.columns s₀ BN p j < p.K then
        ∑ k ∈ Finset.range (RealSetup.limit s₀ p), product s₀ BM BN p i j k else 0)) := by
  rw [h.acc]
  simp only [carry, RealMath.partial_complete _ _ hBCS]

/-- The prologue and loop are reached by the same full source in either order
of the later residual addition. No pre-populated register is assumed. -/
theorem prefix_executes (swapped : Bool) (s₀ : BlockState) (BM BN BCS : Nat)
    (p : RealProgram.Parameters) :
    ∃ t, stepStmts
      (RealProgram.realPart (RealProgram.parts swapped BM BN BCS p).before ++
        [.forRangeDyn "cs" (.constNat 0) (RealProgram.loopBound BCS) (.constNat 1)
          (RealProgram.realPart (RealProgram.parts swapped BM BN BCS p).body)]) s₀ = some t ∧
      Holds s₀ BM BN BCS p (RealMath.blocks (RealSetup.limit s₀ p) BCS) t := by
  obtain ⟨t, he, ht⟩ := executes s₀ BM BN BCS p
  refine ⟨t, ?_, ht⟩
  rw [RealSetup.original_candidate_setup, RealLoads.matches_source]
  rw [stepStmts.append_some (RealSetup.executes s₀ BM BN BCS p)]
  dsimp only [statement] at he
  rw [stepStmts.cons_some he]
  exact stepStmts.nil

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmChunkBwd.RealLoop
