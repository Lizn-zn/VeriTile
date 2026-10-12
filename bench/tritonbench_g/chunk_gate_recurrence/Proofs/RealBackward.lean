import bench.tritonbench_g.chunk_gate_recurrence.Proofs.RealBackwardPost

/-! Complete original backward execution from input-memory facts. DI is the
reverse suffix beginning at the following chunk, DG is its complete tile
contraction with S, and DL is suffix zero. The last DI/DG slices are untouched;
host zeroing and summing DG across programs are outside this kernel theorem.
Address bounds and flat-memory correspondence are separate obligations. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.RealBackward
open VeriTile.Triton RealProgram RealBackwardPointers RealBackwardProgram RealBackwardLoop RealBackwardInit
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

def footprint (p : Layout) (l : Launch) (DI DG DL : RegionName) : WriteFootprint :=
  WriteFootprint.union (RealBackwardLoop.footprint p l DI DG) (RealBackwardPost.footprint p l DL)

theorem invariant (p : Layout) (l : Launch) (S D DI DG DS : RegionName) (a : Inputs p) (u : BlockState)
    (hS : ∀ (t : Nat) (e : Fin p.BK) (j : Fin p.BV), t < p.N-1 →
      u.readMem S (address p l t e.val j.val) = a.source t e j)
    (hDS : ∀ (t : Nat) (e : Fin p.BK) (j : Fin p.BV), t < p.N →
      u.readMem DS (address p l t e.val j.val) = a.incoming t e j)
    (hD : ∀ t, t < p.N → u.readMem D (gateAddress p l t) = a.gate t) :
    Holds u p l S D DI DG DS a 0 (initial p l S D DI DG DS u) := by
  have hzero : RealMath.reverse a p.N p.N = fun _ _ => 0 := by
    funext e j
    exact RealMath.reverse_end a p.N e j
  refine ⟨Nat.zero_le _,?_,?_,?_,?_,?_,?_,?_,hS,hDS,hD,?_,?_,?_⟩
  · constructor <;> simp [initial,pointers,prepared]
  · simp [initial,pointers]
  · simp [initial,pointers]
  · simp [initial,pointers]
  · simp [initial,pointers]
  · simp [initial,pointers]
  · simp [initial,pointers,hzero]
  · intro t e j hlo hhi; omega
  · intro t hlo hhi; omega
  · intro r off hn; rfl

theorem original_executes (p : Layout) (S D DI DG DL DS : RegionName) (a : Inputs p)
    (u : BlockState) (storage gate incoming : TypedSource.Storage) (H : Nat)
    (hN : 0 < p.N) (hBKpos : 0 < p.BK) (hBVpos : 0 < p.BV)
    (hBK : p.BK ≤ p.K) (hBV : p.BV ≤ p.V)
    (sep : Separation S D DI DG DS) (hIDL : DI ≠ DL) (hGDL : DG ≠ DL)
    (hS : ∀ (t : Nat) (e : Fin p.BK) (j : Fin p.BV), t < p.N-1 →
      u.readMem S (address p (launch u) t e.val j.val) = a.source t e j)
    (hDS : ∀ (t : Nat) (e : Fin p.BK) (j : Fin p.BV), t < p.N →
      u.readMem DS (address p (launch u) t e.val j.val) = a.incoming t e j)
    (hD : ∀ t, t < p.N → u.readMem D (gateAddress p (launch u) t) = a.gate t) :
    ∃ v, exec (TypedSource.backward storage gate incoming S D DI DG DL DS H
      p.N p.K p.V p.BK p.BV).eraseDType.toAlgKernel u = some v ∧
      (∀ (t : Nat) (e : Fin p.BK) (j : Fin p.BV), t < p.N-1 →
        v.readMem DI (address p (launch u) t e.val j.val) = RealMath.reverse a p.N (t+1) e j) ∧
      (∀ t, t < p.N-1 → v.readMem DG (gradientAddress p (launch u) t) = RealMath.gateGradient a p.N t) ∧
      (∀ e j, v.readMem DL (seedAddress p (launch u) e.val j.val) = RealMath.reverse a p.N 0 e j) ∧
      BlockState.WriteWithin (footprint p (launch u) DI DG DL) u v := by
  let l := launch u
  have hi := noCollision p l hBK hBV
  have hgi := gradient_injective p l (Nat.div_pos hBK hBKpos) (Nat.div_pos hBV hBVpos)
  obtain ⟨mid,he,h⟩ := RealBackwardLoop.executes u p l S D DI DG DS a sep hi hgi
    (initial p l S D DI DG DS u) (invariant p l S D DI DG DS a u hS hDS hD)
  have hremain : p.N-(p.N-1) = 1 := by omega
  have hds : mid.regs .ptr [p.BK,p.BV] "DS" = some (pointer p l DS 0) := by
    simpa only [backwardPointer_live p l DS (p.N-1) (p.N-1) (Nat.le_refl _),Nat.sub_self] using h.incoming
  have hd : mid.regs .ptr [] "D" = some (gatePointer p l D 0) := by
    simpa only [backwardGate_live p l D (p.N-1) (Nat.le_refl _),Nat.sub_self] using h.gate
  have ha : mid.regs .real [p.BK,p.BV] "Dacc" = some (tile p (RealMath.reverse a p.N 1)) := by
    simpa only [hremain] using h.carry
  have hp := RealBackwardPost.executes p l D DL DS a (RealMath.reverse a p.N 1) mid h.fixed
    hds hd ha (fun e j => h.inputIncoming 0 e j hN) (h.inputGate 0 hN)
  have hnext : RealBackwardStep.next p a 0 (RealMath.reverse a p.N 1) = RealMath.reverse a p.N 0 := by
    funext e j
    exact (RealMath.reverse_step a p.N 0 hN e j).symm
  let pre := RealBackwardPost.beforeStore p l DL a (RealMath.reverse a p.N 1) mid
  let v := RealBackwardPost.result p l DL a (RealMath.reverse a p.N 1) mid
  refine ⟨v,?_,?_,?_,?_,?_⟩
  · rw [exec,source_body,ProgramRewrite.append,ProgramRewrite.append,
      RealBackwardInit.executes p S D DI DG DL DS u]
    change (stepStmts [statement p] _).bind (stepStmts (postCode p S D DI DG DL DS)) = _
    rw [stepStmts.cons_some he,stepStmts.nil]
    simpa only [Option.bind_some,RealBackwardPost.code_exact] using hp
  · intro t e j ht
    change (RealBackwardPost.seedScatter p l DL _ pre).readMem DI _ = _
    rw [RealBackwardPost.seedScatter,BlockState.scatter_preserves_other_region _ _ _ _ hIDL]
    exact h.outputDI t e j (by omega) ht
  · intro t ht
    change (RealBackwardPost.seedScatter p l DL _ pre).readMem DG _ = _
    rw [RealBackwardPost.seedScatter,BlockState.scatter_preserves_other_region _ _ _ _ hGDL]
    exact h.outputDG t (by omega) ht
  · intro e j
    change (RealBackwardPost.seedScatter p l DL _ pre).readMem DL _ = _
    rw [RealBackwardPost.seedScatter_read p l DL _ pre hi e j,hnext]
  · have hf := RealBackwardPost.seedScatter_frame p l DL (RealBackwardStep.next p a 0 (RealMath.reverse a p.N 1)) pre
    intro r off hn
    have hnloop : ¬ RealBackwardLoop.footprint p l DI DG (r,off) := fun hh => hn (Or.inl hh)
    have hnpost : ¬ RealBackwardPost.footprint p l DL (r,off) := fun hh => hn (Or.inr hh)
    exact (h.frame r off hnloop).trans (hf r off hnpost)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.RealBackward
