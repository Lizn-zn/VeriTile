import bench.tritonbench_g.bmm_chunk_fwd.Proofs.RealStep
import bench.tritonbench_g.bmm_chunk_fwd.Proofs.RealSetup

/-! Execute the complete dynamic contraction loop from the actual setup.
The invariant relates pointers and the accumulator to initial memory, keeps
all other registers and state fields, and counts the source's ceiling-divided
iterations. Partial final blocks are included through the actual load masks.
No loop execution or precomputed product is assumed. -/
namespace VeriTile.Bench.TritonBenchG.BmmChunkFwd.RealLoop
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false

def aPointers (s₀ : BlockState) (BM BN BK : Nat) (p : RealProgram.Parameters)
    (c : Nat) : Tile .ptr [BM, BK] :=
  Tile.mat fun i k => (p.A, RealSetup.aBase s₀ p +
    (RealSetup.rows s₀ BM BN p i * p.aRow + (c * BK + k.val) * p.aColumn))

def bPointers (s₀ : BlockState) (BN BK : Nat) (p : RealProgram.Parameters)
    (c : Nat) : Tile .ptr [BK, BN] :=
  Tile.mat fun k j => (p.B, RealSetup.bBase s₀ p +
    ((c * BK + k.val) * p.bColumn + RealSetup.columns s₀ BN p j * p.bRow))

def product (s₀ : BlockState) (BM BN : Nat) (p : RealProgram.Parameters)
    (i : Fin BM) (j : Fin BN) (k : Nat) : ℝ :=
  s₀.readMem p.A (RealSetup.aBase s₀ p + (RealSetup.rows s₀ BM BN p i * p.aRow + k * p.aColumn)) *
  s₀.readMem p.B (RealSetup.bBase s₀ p + (k * p.bColumn + RealSetup.columns s₀ BN p j * p.bRow))

def carry (s₀ : BlockState) (BM BN BK : Nat) (p : RealProgram.Parameters)
    (c : Nat) (i : Fin BM) (j : Fin BN) : ℝ :=
  if RealSetup.rows s₀ BM BN p i < RealSetup.limit s₀ p ∧ RealSetup.columns s₀ BN p j < RealSetup.limit s₀ p then
    BmmChunkBwd.RealMath.partialSum p.K BK c (product s₀ BM BN p i j)
  else 0

def writes : List RegName := ["k", "a", "b", "acc", "a_ptrs", "b_ptrs"]

structure Holds (s₀ : BlockState) (BM BN BK : Nat) (p : RealProgram.Parameters)
    (c : Nat) (s : BlockState) : Prop where
  count : c ≤ BmmChunkBwd.RealMath.blocks p.K BK
  mem : s.mem = s₀.mem
  undef : s.undef = s₀.undef
  pids : s.pids = s₀.pids
  numPids : s.numPids = s₀.numPids
  fixed : ∀ dtype shape name, name ∉ writes → s.regs dtype shape name =
    (RealSetup.result s₀ BM BN BK p).regs dtype shape name
  a : s.regs .ptr [BM, BK] "a_ptrs" = some (aPointers s₀ BM BN BK p c)
  b : s.regs .ptr [BK, BN] "b_ptrs" = some (bPointers s₀ BN BK p c)
  acc : s.regs .real [BM, BN] "acc" = some (Tile.mat fun i j => some (carry s₀ BM BN BK p c i j))

theorem initial (s₀ : BlockState) (BM BN BK : Nat) (p : RealProgram.Parameters) :
    Holds s₀ BM BN BK p 0 (RealSetup.result s₀ BM BN BK p) := by
  refine ⟨Nat.zero_le _, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · have hf := RealSetup.state_frame s₀ BM BN BK p
    have hx := congrArg BlockState.mem hf
    exact hx
  · have hf := RealSetup.state_frame s₀ BM BN BK p
    have hx := congrArg BlockState.undef hf
    exact hx
  · have hf := RealSetup.state_frame s₀ BM BN BK p
    have hx := congrArg BlockState.pids hf
    exact hx
  · have hf := RealSetup.state_frame s₀ BM BN BK p
    have hx := congrArg BlockState.numPids hf
    exact hx
  · intros; rfl
  · simp [RealSetup.result, aPointers, RealSetup.aPointers]
  · simp [RealSetup.result, bPointers, RealSetup.bPointers]
  · simp [RealSetup.result, carry, BmmChunkBwd.RealMath.partial_zero]

theorem a_advance (s₀ : BlockState) (BM BN BK : Nat) (p : RealProgram.Parameters) (c : Nat) :
    RealStep.advance (aPointers s₀ BM BN BK p c) (BK * p.aColumn) =
      aPointers s₀ BM BN BK p (c + 1) := by
  apply Tile.ext
  rintro ⟨i, k, ⟨⟩⟩
  simp only [RealStep.advance, aPointers, Tile.mat, Prod.mk.injEq, true_and]
  ring

theorem b_advance (s₀ : BlockState) (BN BK : Nat) (p : RealProgram.Parameters) (c : Nat) :
    RealStep.advance (bPointers s₀ BN BK p c) (BK * p.bColumn) =
      bPointers s₀ BN BK p (c + 1) := by
  apply Tile.ext
  rintro ⟨k, j, ⟨⟩⟩
  simp only [RealStep.advance, bPointers, Tile.mat, Prod.mk.injEq, true_and]
  ring

theorem carry_succ (s₀ s : BlockState) (BM BN BK : Nat) (p : RealProgram.Parameters)
    (c : Nat) (hm : s.mem = s₀.mem) (i : Fin BM) (j : Fin BN) :
    carry s₀ BM BN BK p c i j +
      RealDot.value s (aPointers s₀ BM BN BK p c) (bPointers s₀ BN BK p c)
        (RealSetup.rows s₀ BM BN p) (RealSetup.columns s₀ BN p) (RealSetup.limit s₀ p) p.K c i j = carry s₀ BM BN BK p (c + 1) i j := by
  simp only [carry, RealDot.value]
  split
  · rw [BmmChunkBwd.RealMath.partial_succ]
    congr 1
    apply Finset.sum_congr rfl
    intro k _
    simp only [aPointers, bPointers, Tile.mat, product, BlockState.readMem, hm]
  · simp


theorem iteration (swapped : Bool) (s₀ : BlockState) (BM BN BK : Nat) (p : RealProgram.Parameters)
    (c : Nat) (s : BlockState) (hc : c < BmmChunkBwd.RealMath.blocks p.K BK)
    (h : Holds s₀ BM BN BK p c s) :
    ∃ t, stepStmts (RealLoads.code swapped BM BN BK p) (s.setReg "k" .nat [] (Tile.scalar c)) = some t ∧
      Holds s₀ BM BN BK p (c + 1) t := by
  let entry := s.setReg "k" .nat [] (Tile.scalar c)
  have hr : entry.regs .nat [BM] "offs_m" = some (Tile.vec (RealSetup.rows s₀ BM BN p)) := by
    have hv := h.fixed .nat [BM] "offs_m" (by decide)
    simpa [entry, RealSetup.result] using hv
  have hn : entry.regs .nat [BN] "offs_n" = some (Tile.vec (RealSetup.columns s₀ BN p)) := by
    have hv := h.fixed .nat [BN] "offs_n" (by decide)
    simpa [entry, RealSetup.result] using hv
  have hk : entry.regs .nat [BK] "offs_k" = some (Tile.vec Fin.val) := by
    have hv := h.fixed .nat [BK] "offs_k" (by decide)
    simpa [entry, RealSetup.result] using hv
  have hl : entry.regs .nat [] "chunk_size_limit" = some (Tile.scalar (RealSetup.limit s₀ p)) := by
    have hv := h.fixed .nat [] "chunk_size_limit" (by decide)
    simpa [entry, RealSetup.result] using hv
  have he := RealStep.executes swapped BM BN BK entry p (aPointers s₀ BM BN BK p c)
    (bPointers s₀ BN BK p c) (RealSetup.rows s₀ BM BN p) (RealSetup.columns s₀ BN p)
    (RealSetup.limit s₀ p) c (carry s₀ BM BN BK p c)
    (by simpa [entry] using h.a) (by simpa [entry] using h.b)
    hr hn hk hl (by simp [entry]) (by simpa [entry] using h.acc)
  refine ⟨_, he, Nat.succ_le_of_lt hc, h.mem, h.undef, h.pids, h.numPids, ?_, ?_, ?_, ?_⟩
  · intro dtype shape name hname
    have hx : name ∉ ["a", "b", "acc", "a_ptrs", "b_ptrs"] := by
      have hs := hname
      simp only [writes, List.mem_cons, not_or] at hs
      simpa only [List.mem_cons, not_or] using hs.2
    rw [RealStep.register_frame _ _ _ _ _ _ _ _ _ _ _ _ hx]
    have hcs : name ≠ "k" := by
      have hs := hname
      simp only [writes, List.mem_cons, not_or] at hs
      exact hs.1
    simpa [entry, BlockState.setReg, hcs] using h.fixed dtype shape name hname
  · simp [RealStep.result, a_advance]
  · simp [RealStep.result, b_advance]
  · simp only [RealStep.result, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, BlockState.setReg_same]
    congr 1
    apply Tile.ext
    rintro ⟨i,j,⟨⟩⟩
    exact congrArg some (carry_succ s₀ entry BM BN BK p c h.mem i j)


/-- The source's dynamic contraction loop, with its actual register bound. -/
def statement (swapped : Bool) (BM BN BK : Nat) (p : RealProgram.Parameters) : Stmt :=
  .forRangeDyn "k" (.constNat 0) (RealProgram.loopBound p.K BK) (.constNat 1)
    (RealLoads.code swapped BM BN BK p)

theorem executes (swapped : Bool) (s₀ : BlockState) (BM BN BK : Nat) (p : RealProgram.Parameters) :
    ∃ t, stepStmt (statement swapped BM BN BK p) (RealSetup.result s₀ BM BN BK p) = some t ∧
      Holds s₀ BM BN BK p (BmmChunkBwd.RealMath.blocks p.K BK) t := by
  have resolve : stepStmt (statement swapped BM BN BK p) (RealSetup.result s₀ BM BN BK p) =
      stepForRangeAux "k" 0 (BmmChunkBwd.RealMath.blocks p.K BK) 1
        (RealLoads.code swapped BM BN BK p) (RealSetup.result s₀ BM BN BK p) := by
    simp only [statement, stepForRangeAux.forRangeDyn_unfold, RealProgram.loopBound,
      evalOp.eq_def, RealSetup.chunk_limit, bind, Option.bind_some]
    rfl
  rw [resolve]
  obtain ⟨final, t, he, hf, ht⟩ := forRangeAux_inv
    (idx := "k") (stop := BmmChunkBwd.RealMath.blocks p.K BK) (step := 1)
    (P := Holds s₀ BM BN BK p) (by decide)
    (fun c s hc h => iteration swapped s₀ BM BN BK p c s hc h)
    0 (RealSetup.result s₀ BM BN BK p) (initial s₀ BM BN BK p)
  have final_eq : final = BmmChunkBwd.RealMath.blocks p.K BK := Nat.le_antisymm ht.count hf
  subst final
  exact ⟨t, he, ht⟩

/-- Reaa is a sum over the initial memory, including a partial last block. -/
theorem full_sum (s₀ s : BlockState) (BM BN BK : Nat) (p : RealProgram.Parameters)
    (hBK : 0 < BK)
    (h : Holds s₀ BM BN BK p (BmmChunkBwd.RealMath.blocks p.K BK) s) :
    s.regs .real [BM, BN] "acc" = some (Tile.mat fun i j => some
      (if RealSetup.rows s₀ BM BN p i < RealSetup.limit s₀ p ∧ RealSetup.columns s₀ BN p j < RealSetup.limit s₀ p then
        ∑ k ∈ Finset.range p.K, product s₀ BM BN p i j k else 0)) := by
  rw [h.acc]
  simp only [carry, BmmChunkBwd.RealMath.partial_complete _ _ hBK]

/-- The prologue and loop are reached by the same full source in either order
of the block addition. No pre-populated register is assumed. -/
theorem prefix_executes (swapped : Bool) (s₀ : BlockState) (BM BN BK : Nat)
    (p : RealProgram.Parameters) :
    ∃ t, stepStmts
      (RealProgram.realPart (RealProgram.parts swapped BM BN BK p).before ++
        [.forRangeDyn "k" (.constNat 0) (RealProgram.loopBound p.K BK) (.constNat 1)
          (RealProgram.realPart (RealProgram.parts swapped BM BN BK p).body)])
        (RealSetup.prologueResult s₀ BM BN p) = some t ∧
      Holds s₀ BM BN BK p (BmmChunkBwd.RealMath.blocks p.K BK) t := by
  obtain ⟨t, he, ht⟩ := executes swapped s₀ BM BN BK p
  refine ⟨t, ?_, ht⟩
  rw [RealSetup.original_candidate_setup, RealLoads.matches_source]
  rw [stepStmts.append_some (RealSetup.executes s₀ BM BN BK p)]
  dsimp only [statement] at he
  rw [stepStmts.cons_some he]
  exact stepStmts.nil

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmChunkFwd.RealLoop
