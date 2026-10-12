import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealIteration
import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealLoopFormula

/-! The entire signed source loop, with symbolic dimensions and independent
K/V strides. The entry contract names values established by the prologue.
The loop reads actual memory and returns independent closed row formulas;
initialization, final stores and address safety are proved separately. -/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealLoop
open VeriTile.Triton
open RealTiles RealScores RealAccumulation RealLoopFormula
noncomputable section

structure Parameters (BM BN BD : Nat) where
  q : TileIndex [BM, BD] → ℝ
  scale : ℝ
  startM : Nat
  context : Nat
  window : Nat
  complement : Nat
  even : Nat
  offset : Int
  size : Int
  lo : Nat
  hi : Int
  K : RegionName
  V : RegionName
  kBase : Nat
  vBase : Nat
  strideKD : Nat
  strideKN : Nat
  strideVN : Nat
  strideVD : Nat

def cursor {BM BN BD : Nat} (p : Parameters BM BN BD) (i : Nat) : Nat := p.lo + i * BN

def count {BM BN BD : Nat} (p : Parameters BM BN BD) : Nat := RealKeys.blockCount p.lo p.hi.toNat BN

def keyPointer {BM BN BD : Nat} (p : Parameters BM BN BD) (i : Nat) : Tile .blockPtr [BD, BN] :=
  ⟨fun _ =>
    { region := p.K, baseOffset := p.kBase, parentShape := [BD, p.context],
      blockShape := [BD, BN], strides := [p.strideKD, p.strideKN], offsets := [0, cursor p i] }⟩

def valuePointer {BM BN BD : Nat} (p : Parameters BM BN BD) (i : Nat) : Tile .blockPtr [BN, BD] :=
  ⟨fun _ =>
    { region := p.V, baseOffset := p.vBase, parentShape := [p.context, BD],
      blockShape := [BN, BD], strides := [p.strideVN, p.strideVD], offsets := [cursor p i, 0] }⟩

theorem cursor_succ {BM BN BD : Nat} (p : Parameters BM BN BD) (i : Nat) :
    cursor p (i + 1) = cursor p i + BN := by simp [cursor, Nat.add_mul, Nat.add_assoc]

theorem cursor_lt {BM BN BD : Nat} (p : Parameters BM BN BD) (hN : 0 < BN) (i : Nat) :
    (cursor p i : Int) < p.hi ↔ i < count p := by
  have h := RealKeys.iteration_lt_count p.lo p.hi.toNat BN i hN
  dsimp [cursor, count]
  omega

theorem key_pointer_succ {BM BN BD : Nat} (p : Parameters BM BN BD) (i : Nat) :
    (⟨fun j => ((keyPointer p i).data j).advance [0, BN]⟩ : Tile .blockPtr [BD, BN]) =
      keyPointer p (i + 1) := by
  apply Tile.ext
  intro j
  simp [keyPointer, cursor_succ]

theorem value_pointer_succ {BM BN BD : Nat} (p : Parameters BM BN BD) (i : Nat) :
    (⟨fun j => ((valuePointer p i).data j).advance [BN, 0]⟩ : Tile .blockPtr [BN, BD]) =
      valuePointer p (i + 1) := by
  apply Tile.ext
  intro j
  simp [valuePointer, cursor_succ]

def score {BM BN BD : Nat} (mem : BlockState) (p : Parameters BM BN BD) (i : Nat) :
    Fin BM → Fin BN → ℝ := fun row key =>
  RealNumeric.scores BM BN BD p.q (values mem (keyPointer p i) (checks p.even)) p.scale
    (row, key, PUnit.unit)

def active {BM BN BD : Nat} (p : Parameters BM BN BD) (i : Nat) : Fin BM → Fin BN → Bool :=
  keep BM BN p.startM p.context p.window p.complement p.even (cursor p i) p.offset p.size

def value {BM BN BD : Nat} (mem : BlockState) (p : Parameters BM BN BD) (i : Nat) :
    TileIndex [BN, BD] → ℝ := values mem (valuePointer p i) (checks p.even)

def after {BM BN BD : Nat} (mem : BlockState) (p : Parameters BM BN BD) (seed : Carry BM BD)
    (i : Nat) : Carry BM BD := closed seed (score mem p) (active p) (value mem p) i

/-- Entry and loop-carried registers. The invariant includes actual pointers,
Q and scale, not an equation about the result of executing the kernel. -/
structure Registers {BM BN BD : Nat} (p : Parameters BM BN BD) (c : Carry BM BD)
    (i : Nat) (s : BlockState) : Prop where
  q : s.regs .real [BM, BD] "q" = some (finite p.q)
  scale : s.regs .real [] "qk_scale" = some (Tile.scalar (some p.scale))
  startM : s.regs .nat [] "start_m" = some (Tile.scalar p.startM)
  center : s.regs .real [BM] "m_i" = some (⟨fun j => c.center j.1⟩ : Tile .real [BM])
  denominator : s.regs .real [BM] "l_i" = some (finite (fun j : TileIndex [BM] => c.denominator j.1))
  numerator : s.regs .real [BM, BD] "acc" = some (finite c.numerator)
  key : s.regs .blockPtr [BD, BN] "K_block_ptr" = some (keyPointer p i)
  value : s.regs .blockPtr [BN, BD] "V_block_ptr" = some (valuePointer p i)

def writes : List RegName := ["start_n", "k", "qk", "dist", "mask", "m_ij", "p", "l_ij", "tmp",
  "alpha_mask", "alpha", "l_i", "acc", "v", "m_i", "V_block_ptr", "K_block_ptr"]

/-- Instantiate the checked body theorem and advance the closed-form
invariant. K/V loads use the fixed original memory, since the loop has no store. -/
theorem invariant_step {BM BN BD : Nat} (mem s : BlockState) (p : Parameters BM BN BD)
    (seed : Carry BM BD) (hN : 0 < BN) (i : Nat)
    (hmem : s.mem = mem.mem) (hregs : Registers p (after mem p seed i) i s) :
    ∃ s', stepStmts (RealBody.body BM BN BD p.context p.window p.complement p.even p.offset p.size)
        (s.setReg "start_n" .int [] (Tile.scalar (cursor p i : Int))) = some s' ∧
      Registers p (after mem p seed (i + 1)) (i + 1) s' ∧ Frame writes s s' := by
  let si := s.setReg "start_n" .int [] (Tile.scalar (cursor p i : Int))
  let c := after mem p seed i
  obtain ⟨sf, hs, hc, hd, ha, hk, hv, hf⟩ := RealIteration.iteration_executes si
    BM BN BD p.startM p.context p.window p.complement p.even hN (cursor p i) p.offset p.size
    p.q p.scale c.center c.denominator c.numerator (keyPointer p i) (valuePointer p i)
    (by simp [si, hregs.q]) (by simp [si, hregs.scale]) (by simp [si, hregs.startM]) (by simp [si])
    (by simp [si, c, hregs.center]) (by simp [si, c, hregs.denominator])
    (by simp [si, c, hregs.numerator]) (by simp [si, hregs.key]) (by simp [si, hregs.value])
  have hkvalues := RealIteration.values_of_same_memory si mem (keyPointer p i) (checks p.even) hmem
  have hvvalues := RealIteration.values_of_same_memory si mem (valuePointer p i) (checks p.even) hmem
  rw [hkvalues] at hc hd ha
  rw [hvvalues] at ha
  rw [key_pointer_succ] at hk
  rw [value_pointer_succ] at hv
  have hframe : Frame writes s sf := (Frame.setReg writes s "start_n" _ _ _ (by simp [writes])).trans hf
  refine ⟨sf, hs, ?_, hframe⟩
  rw [after, closed_succ]
  exact ⟨(hframe.regs .real [BM, BD] "q" (by simp [writes])).trans hregs.q,
    (hframe.regs .real [] "qk_scale" (by simp [writes])).trans hregs.scale,
    (hframe.regs .nat [] "start_m" (by simp [writes])).trans hregs.startM,
    hc, hd, ha, hk, hv⟩

/-- Execute the signed range from any prefix invariant. The integer upper
bound may be negative; only actual loop tests decide whether a block runs. -/
theorem loop_from {BM BN BD : Nat} (mem s : BlockState) (p : Parameters BM BN BD)
    (seed : Carry BM BD) (hN : 0 < BN) (i : Nat) (hi : i ≤ count p)
    (hmem : s.mem = mem.mem) (hregs : Registers p (after mem p seed i) i s) :
    ∃ s', stepForRangeIntAux "start_n" (cursor p i) p.hi BN
        (RealBody.body BM BN BD p.context p.window p.complement p.even p.offset p.size) s = some s' ∧
      Registers p (after mem p seed (count p)) (count p) s' ∧ Frame writes s s' := by
  by_cases h : i < count p
  · have hcur := (cursor_lt p hN i).mpr h
    obtain ⟨sm, hs, hnext, hf⟩ := invariant_step mem s p seed hN i hmem hregs
    obtain ⟨sf, hrun, hfinal, hframe⟩ := loop_from mem sm p seed hN (i + 1) (by omega)
      (hf.mem.trans hmem) hnext
    refine ⟨sf, ?_, hfinal, hf.trans hframe⟩
    rw [stepForRangeIntAux.step_lt (by omega) hcur, hs, Option.bind_some]
    simpa only [cursor_succ, Nat.cast_add] using hrun
  · have hieq : i = count p := by omega
    subst i
    exact ⟨s, stepForRangeIntAux.step_ge (by omega) (by
      have ht := cursor_lt p hN (count p); omega), hregs, Frame.refl _ _⟩
termination_by count p - i

/-- The whole signed loop returns the independent closed row formulas and
advanced pointers. This theorem still starts at the loop-entry register
contract; the source prologue and final stores are separate obligations. -/
theorem loop_executes {BM BN BD : Nat} (s : BlockState) (p : Parameters BM BN BD)
    (seed : Carry BM BD) (hN : 0 < BN) (hregs : Registers p seed 0 s) :
    ∃ s', stepForRangeIntAux "start_n" p.lo p.hi BN
        (RealBody.body BM BN BD p.context p.window p.complement p.even p.offset p.size) s = some s' ∧
      Registers p (after s p seed (count p)) (count p) s' ∧ Frame writes s s' := by
  simpa [cursor] using loop_from s s p seed hN 0 (Nat.zero_le _) rfl
    (by simpa [after, closed_zero] using hregs)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealLoop
