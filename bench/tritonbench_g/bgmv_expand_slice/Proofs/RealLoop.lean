import bench.tritonbench_g.bgmv_expand_slice.Proofs.RealBlockExecution

/-! The complete dynamic output loop, including in-place accumulation. Each
active output cell is written exactly once when the output stride is positive.
The loop invariant retains initial values for future output blocks, so the
mathematical formula uses initial output memory rather than an intermediate
value or an output-equation premise. Setup and flat bounds remain separate. -/
namespace VeriTile.Bench.TritonBenchG.BgmvExpandSlice.RealLoop
open VeriTile.Triton RealBlock RealBlockExecution
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

structure Geometry where
  weights : RegionName
  output : RegionName
  weightBase : Nat
  rowStride : Nat
  colStride : Nat
  outputBase : Nat
  outputStride : Nat
  K : Nat
  length : Nat
  BN : Nat
  BK : Nat

namespace Geometry

def body (p : Geometry) (add swapped : Bool) :=
  RealLoopSource.body p.rowStride p.colStride p.outputStride p.K p.BN p.BK add swapped

def statement (p : Geometry) (add swapped : Bool) : Stmt :=
  .forRangeDyn "n" (.constNat 0) (.ref .nat [] "split_n_length") (.constNat p.BN) (p.body add swapped)

def outputAddress (p : Geometry) (g : Nat) := p.outputBase + g*p.outputStride

def expected (p : Geometry) (a : Fin p.BK → ℝ) (initial : BlockState) (add : Bool) (g : Nat) :=
  let v := dot a (initial.readMem p.weights) p.weightBase p.rowStride p.colStride p.K g
  if add then v + initial.readMem p.output (p.outputAddress g) else v

def Misses (p : Geometry) (r : RegionName) (o : Nat) : Prop :=
  ∀ g, g < p.length → ¬(p.output = r ∧ p.outputAddress g = o)

structure Registers (p : Geometry) (a : Fin p.BK → ℝ) (s : BlockState) : Prop where
  split : s.regs .nat [] "split_n_length" = some (Tile.scalar p.length)
  offsetN : s.regs .nat [p.BN] "offset_n" = some (Tile.vec (fun i => i.val))
  offsetK : s.regs .nat [p.BK] "offset_k" = some (Tile.vec (fun k => k.val))
  weightPtr : s.regs .ptr [] "b_ptr" = some (Tile.scalar (p.weights,p.weightBase))
  outputPtr : s.regs .ptr [] "c_ptr" = some (Tile.scalar (p.output,p.outputBase))
  input : s.regs .real [p.BK] "tiled_a" = some ⟨fun i => some (a i.1)⟩

def after (p : Geometry) (a : Fin p.BK → ℝ) (s : BlockState) (add : Bool) (c : Nat) :=
  afterBlock s a p.weights p.output p.weightBase p.rowStride p.colStride
    p.outputBase p.outputStride p.K p.length p.BN c add

theorem registers_after (p : Geometry) (a : Fin p.BK → ℝ) (s : BlockState) (add : Bool) (c : Nat)
    (h : p.Registers a s) : p.Registers a (p.after a s add c) := by
  have reg := block_register s a p.weights p.output p.weightBase p.rowStride p.colStride
    p.outputBase p.outputStride p.K p.length p.BN c add
  constructor
  · simpa only [after, reg .nat [] "split_n_length" (by decide)] using h.split
  · simpa only [after, reg .nat [p.BN] "offset_n" (by decide)] using h.offsetN
  · simpa only [after, reg .nat [p.BK] "offset_k" (by decide)] using h.offsetK
  · simpa only [after, reg .ptr [] "b_ptr" (by decide)] using h.weightPtr
  · simpa only [after, reg .ptr [] "c_ptr" (by decide)] using h.outputPtr
  · simpa only [after, reg .real [p.BK] "tiled_a" (by decide)] using h.input

structure Invariant (p : Geometry) (a : Fin p.BK → ℝ) (initial : BlockState)
    (add : Bool) (n : Nat) (s : BlockState) : Prop where
  multiple : p.BN ∣ n
  registers : p.Registers a s
  weights : s.readMem p.weights = initial.readMem p.weights
  output : ∀ g, g < p.length → s.readMem p.output (p.outputAddress g) =
    if g < n then p.expected a initial add g else initial.readMem p.output (p.outputAddress g)
  frame : ∀ r o, p.Misses r o → s.mem r o = initial.mem r o

private theorem address_injective (p : Geometry) (hstride : 0 < p.outputStride) :
    Function.Injective p.outputAddress := by
  intro i j h
  exact Nat.eq_of_mul_eq_mul_right hstride (Nat.add_left_cancel h)

theorem invariant_step (p : Geometry) (a : Fin p.BK → ℝ) (initial : BlockState)
    (add swapped : Bool) (hstride : 0 < p.outputStride) (hw : p.weights ≠ p.output)
    (n : Nat) (s : BlockState) (_hlt : n < p.length)
    (h : p.Invariant a initial add n s) :
    ∃ s', stepStmts (p.body add swapped) (s.setReg "n" .nat [] (Tile.scalar n)) = some s' ∧
      p.Invariant a initial add (n+p.BN) s' := by
  obtain ⟨c, hc⟩ := h.multiple
  have hn : n = c*p.BN := hc.trans (Nat.mul_comm _ _)
  clear hc
  subst n
  refine ⟨p.after a s add c, ?_, ?_⟩
  · exact block_exec s a p.weights p.output p.weightBase p.rowStride p.colStride
      p.outputBase p.outputStride p.K p.length p.BN c add swapped
      h.registers.split h.registers.offsetN h.registers.offsetK
      h.registers.weightPtr h.registers.outputPtr h.registers.input
  · refine ⟨⟨c+1, by ring⟩, p.registers_after a s add c h.registers, ?_, ?_, ?_⟩
    · funext o
      have hm := block_frame s a p.weights p.output p.weights p.weightBase p.rowStride p.colStride
        p.outputBase p.outputStride p.K p.length p.BN c o add
        (fun _ _ hbad => hw hbad.1.symm)
      simpa only [after, BlockState.readMem, hm] using congrFun h.weights o
    · intro g hg
      by_cases hit : c*p.BN ≤ g ∧ g < c*p.BN+p.BN
      · let i : Fin p.BN := ⟨g-c*p.BN, by omega⟩
        have hi : c*p.BN+i.val = g := by dsimp [i]; omega
        have hr := block_readback s a p.weights p.output p.weightBase p.rowStride p.colStride
          p.outputBase p.outputStride p.K p.length p.BN c add hstride i (by simpa [hi] using hg)
        have hold := h.output g hg
        simp only [outputAddress, if_neg (show ¬g < c*p.BN by omega)] at hold
        change (p.after a s add c).readMem p.output (p.outputAddress g) = _
        simpa only [after, outputAddress, hi, h.weights, hold, if_pos hit.2, expected] using hr
      · have hm := block_frame s a p.weights p.output p.output p.weightBase p.rowStride p.colStride
          p.outputBase p.outputStride p.K p.length p.BN c (p.outputAddress g) add (by
            intro i _ hbad
            have heq := p.address_injective hstride hbad.2
            have hi := i.isLt
            omega)
        have hr : (p.after a s add c).readMem p.output (p.outputAddress g) =
            s.readMem p.output (p.outputAddress g) := by simp only [after, BlockState.readMem, hm]
        rw [hr, h.output g hg]
        by_cases hprev : g < c*p.BN
        · rw [if_pos hprev, if_pos (show g < c*p.BN+p.BN by omega)]
        · rw [if_neg hprev, if_neg (show ¬g < c*p.BN+p.BN by omega)]
    · intro r o hmiss
      have hm := block_frame s a p.weights p.output r p.weightBase p.rowStride p.colStride
        p.outputBase p.outputStride p.K p.length p.BN c o add (fun i hi => hmiss _ hi)
      exact hm.trans (h.frame r o hmiss)

/-- Full real loop result, for arbitrary length/rank/block sizes, row/column
strides and both accumulation flags. Positive output stride separates stores;
the weight region is preserved across blocks. Zero-length loops are included. -/
theorem executes (p : Geometry) (a : Fin p.BK → ℝ) (initial : BlockState)
    (add swapped : Bool) (hBN : 0 < p.BN) (hstride : 0 < p.outputStride)
    (hw : p.weights ≠ p.output) (hregs : p.Registers a initial) :
    ∃ final, stepStmt (p.statement add swapped) initial = some final ∧
      (∀ g, g < p.length → final.readMem p.output (p.outputAddress g) = p.expected a initial add g) ∧
      (∀ r o, p.Misses r o → final.mem r o = initial.mem r o) := by
  have hinit : p.Invariant a initial add 0 initial := by
    refine ⟨dvd_zero _, hregs, rfl, ?_, ?_⟩
    · intro g _; simp
    · intros; rfl
  obtain ⟨n, final, hexec, hdone, hinv⟩ := forRangeDyn_inv
    (P := p.Invariant a initial add) (idx := "n") (body := p.body add swapped)
    (startOp := .constNat 0) (stopOp := .ref .nat [] "split_n_length") (stepOp := .constNat p.BN)
    (by simp) (by simpa using hregs.split) (by simp) (Nat.ne_of_gt hBN) hinit
    (p.invariant_step a initial add swapped hstride hw)
  refine ⟨final, hexec, ?_, hinv.frame⟩
  intro g hg
  simpa only [if_pos (show g < n by omega)] using hinv.output g hg

end Geometry
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BgmvExpandSlice.RealLoop
