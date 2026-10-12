import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealEpilogueSource

/-! Whole-source execution and mathematical memory readback. The contract here
is over region memory: it proves successful execution, all three output views
and preservation outside their addresses. Allocation bounds and transport to
flat memory are separate obligations of the final public IO specification. -/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealExecution
open VeriTile.Triton RealPrologue
noncomputable section

def rowAddress (BM : Nat) (p : Parameters) (s : BlockState) : TileIndex [BM] → Nat :=
  let c := RealHeader.coordinates s p.heads p.kvHeads
  fun i => c.headBatch * p.layout.roundedContext + c.startM * BM + i.1.val

def outputPointer (BM BD : Nat) (p : Parameters) (s : BlockState) : BlockPtr :=
  let c := RealHeader.coordinates s p.heads p.kvHeads
  let b := RealHeader.bases c p.strides
  { region := p.layout.Out, baseOffset := b.out, parentShape := [p.layout.roundedContext, BD],
    blockShape := [BM, BD], strides := [p.layout.oRow, p.layout.oDim], offsets := [c.startM * BM, 0] }

def outputAddress (BM BD : Nat) (p : Parameters) (s : BlockState) : TileIndex [BM, BD] → Nat :=
  RealEpilogue.blockAddress BM BD (outputPointer BM BD p s)

def finalCarry (BM BN BD : Nat) (p : Parameters) (s : BlockState) : RealLoopFormula.Carry BM BD :=
  RealLoopContext.fullResult s (loopParameters BM BN BD p s) (seed BM BD p s)

/-- Only output uniqueness is needed for region-memory readback. The source
reads all six input views before writing, so input/output aliasing is not
silently excluded here. Flat allocation and grid safety require further bounds. -/
structure OutputLayout (BM BD : Nat) (p : Parameters) (s : BlockState) : Prop where
  injective : Function.Injective (outputAddress BM BD p s)
  center_output : p.layout.M ≠ p.layout.Out
  denominator_center : p.layout.L ≠ p.layout.M
  denominator_output : p.layout.L ≠ p.layout.Out

structure Results (BM BN BD finish : Nat) (p : Parameters) (initial current : BlockState) : Prop where
  pids : current.pids = initial.pids
  undef : current.undef = initial.undef
  center : ∀ i, current.readMem p.layout.M (rowAddress BM p initial i) =
    RealEpilogue.centerValue finish (finalCarry BM BN BD p initial) i.1
  output : ∀ i, current.readMem p.layout.Out (outputAddress BM BD p initial i) =
    RealEpilogue.outputValue finish (finalCarry BM BN BD p initial) i
  denominator : finish = 0 → ∀ i, current.readMem p.layout.L (rowAddress BM p initial i) =
    (finalCarry BM BN BD p initial).denominator i.1
  finished_L : finish ≠ 0 → ∀ o, current.mem p.layout.L o = initial.mem p.layout.L o
  frame : ∀ r o,
    (r ≠ p.layout.M ∨ ∀ i, o ≠ rowAddress BM p initial i) →
    (finish ≠ 0 ∨ r ≠ p.layout.L ∨ ∀ i, o ≠ rowAddress BM p initial i) →
    (r ≠ p.layout.Out ∨ ∀ i, o ≠ outputAddress BM BD p initial i) →
    current.mem r o = initial.mem r o

theorem row_address_injective (BM : Nat) (p : Parameters) (s : BlockState) :
    Function.Injective (rowAddress BM p s) := by
  rintro ⟨i, ⟨⟩⟩ ⟨j, ⟨⟩⟩ h
  have hij : i = j := Fin.ext (by simpa only [rowAddress, Nat.add_left_cancel_iff] using h)
  subst j
  rfl

/-- Internal composition: the verified prefix supplies registers and unchanged
memory; the verified suffix supplies stores. The source theorems below obtain
these premises by executing the complete original/candidate implementations. -/
theorem completes (BM BN BD finish : Nat) (p : Parameters) (reciprocal : Bool)
    (s : BlockState) (body : List Stmt) (hlayout : OutputLayout BM BD p s)
    (htail : RealEpilogueSource.afterLoop body = RealEpilogue.code BM BD finish reciprocal)
    (hcompute : ∃ t, stepStmts (RealPrologueSource.throughLoop body) s = some t ∧
      RealLoop.Registers (loopParameters BM BN BD p s) (finalCarry BM BN BD p s)
        (RealLoop.count (loopParameters BM BN BD p s)) t ∧
      RealScores.Frame (RealPrologue.writes ++ RealLoop.writes) s t ∧ OutputRegisters BM BD p s t) :
    ∃ sf, stepStmts body s = some sf ∧ Results BM BN BD finish p s sf := by
  obtain ⟨t, erun, hc, hf, ho⟩ := hcompute
  let c := finalCarry BM BN BD p s
  let addr := rowAddress BM p s
  let bp := outputPointer BM BD p s
  let sf := RealEpilogue.writtenState t finish c p.layout.M p.layout.L addr bp
  have hs : stepStmts (RealEpilogue.code BM BD finish reciprocal) t = some sf :=
    RealEpilogue.executes t finish reciprocal c p.layout.M p.layout.L addr bp
      ⟨hc.center, hc.denominator, hc.numerator⟩ ho.center ho.denominator ho.output
  have hi := row_address_injective BM p s
  have hcontrol := RealEpilogue.written_control t finish c p.layout.M p.layout.L addr bp
  refine ⟨sf, ?_, ?_⟩
  · rw [← RealEpilogueSource.through_after_loop body, stepStmts.append_some erun, htail]
    exact hs
  · refine ⟨hcontrol.1.trans hf.pids, hcontrol.2.trans hf.undef, ?_, ?_, ?_, ?_, ?_⟩
    · exact RealEpilogue.center_readback t finish c p.layout.M p.layout.L addr bp hi hlayout.center_output
    · exact RealEpilogue.output_readback t finish c p.layout.M p.layout.L addr bp hlayout.injective
    · intro he i
      subst finish
      exact RealEpilogue.denominator_readback t c p.layout.M p.layout.L addr bp hi
        hlayout.denominator_center hlayout.denominator_output i
    · intro he o
      exact (RealEpilogue.finished_preserves_L t finish c p.layout.M p.layout.L addr bp he
        hlayout.denominator_center hlayout.denominator_output o).trans (congrFun (congrFun hf.mem p.layout.L) o)
    · intro r o hm hl hout
      exact (RealEpilogue.written_frame t finish c p.layout.M p.layout.L addr bp r o hm hl hout).trans
        (congrFun (congrFun hf.mem r) o)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealExecution
