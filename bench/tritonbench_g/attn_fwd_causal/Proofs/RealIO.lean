import bench.tritonbench_g.attn_fwd_causal.Proofs.RealAccessSafety

/-! Real IO over the finite input windows actually visited by this program.
The number of visits may depend on the program ID; no fixed experiment shape
or context-sized stream truncates the source prefix. Input values are pinned
only where the source reads them. The existing FlatAlloc execution bridge
provides successful flat-memory execution, output readback and the full frame.
This is a per-program specification, with launch composition kept separate. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealIO
open VeriTile.Triton RealLoads
noncomputable section

structure IO (BM BN BD : Nat) where
  kernel : ComputeKernel
  projection : kernel.toAlgorithm? = .ok kernel.toAlgKernel
  configuration : Nat → Nat → Config
  out : RegionName
  stage : Nat

def IO.at {BM BN BD : Nat} (io : IO BM BN BD) (s : BlockState) : Config :=
  io.configuration (s.pids 0) (s.pids 1)

def IO.regions {BM BN BD : Nat} (io : IO BM BN BD) (s : BlockState) : List RegionName :=
  let c := io.at s
  [c.Q,c.K,c.V,c.QS,c.KS,io.out]

def InjectiveOutput (c : Config) (BM BD : Nat) : Prop :=
  ∀ a b, RealEpilogue.active c BM BD a = Bool.true → RealEpilogue.active c BM BD b = Bool.true →
    RealEpilogue.address c BM BD a = RealEpilogue.address c BM BD b → a = b

structure Results {BM BN BD : Nat} (io : IO BM BN BD) (s t : BlockState)
    (ys : TileIndex [BM,BD] → ℝ) : Prop where
  output : ∀ i, RealEpilogue.active (io.at s) BM BD i = Bool.true →
    t.readMem io.out (RealEpilogue.address (io.at s) BM BD i) = ys i
  frame : ∀ r offset, (r ≠ io.out ∨ ∀ i, RealEpilogue.active (io.at s) BM BD i = Bool.true →
    offset ≠ RealEpilogue.address (io.at s) BM BD i) → t.mem r offset = s.mem r offset

/-- Execute the actual projected kernel on disjoint physical allocations.
The premises name finite active address bounds, input values, the zero-undef
convention and nonoverlapping active output cells. None refers to a computed
output or a successful intermediate execution. -/
def IO.Implements {BM BN BD : Nat} (io : IO BM BN BD)
    (f : Nat → Nat → RealFormula.Inputs BM BN BD → TileIndex [BM,BD] → ℝ) : Prop :=
  ∀ (A : FlatAlloc), A.Disjoint →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ (s : BlockState) (xs : RealFormula.Inputs BM BN BD),
    A.regions = io.regions s →
    s.undef = (fun _ _ => 0) →
    RealAccessSafety.Windows (io.at s) io.out BM BN BD io.stage A.extent →
    RealInputPins.Pins s (io.at s) io.stage xs →
    InjectiveOutput (io.at s) BM BD →
    ∃ t, exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s) = some t ∧
      (∀ i, RealEpilogue.active (io.at s) BM BD i = Bool.true →
        t.readMem A.flat (A.addr io.out (RealEpilogue.address (io.at s) BM BD i)) =
          f (s.pids 0) (s.pids 1) xs i) ∧
      (∀ r offset, (r ≠ A.flat ∨ ∀ i, RealEpilogue.active (io.at s) BM BD i = Bool.true →
        offset ≠ A.addr io.out (RealEpilogue.address (io.at s) BM BD i)) →
        t.mem r offset = (A.flattenState s).mem r offset)

scoped notation:25 io " ⊨ " f => IO.Implements io f

/-- All semantic obligations are checked against the same source kernel.
The bridge is the library's FlatAlloc.exec_flatten theorem, with its usual
bounded-access, supported-syntax and zero-undef hypotheses discharged here. -/
theorem IO.Implements.intro {BM BN BD : Nat} (io : IO BM BN BD)
    (f : Nat → Nat → RealFormula.Inputs BM BN BD → TileIndex [BM,BD] → ℝ)
    (hok : io.kernel.toAlgKernel.FlattenOk)
    (hsafe : ∀ bounds s, (∀ r a, s.undef r a = 0) →
      RealAccessSafety.Windows (io.at s) io.out BM BN BD io.stage bounds →
      io.kernel.toAlgKernel.TraceSafe bounds s)
    (hrun : ∀ s xs, (∀ r a, s.undef r a = 0) →
      RealInputPins.Pins s (io.at s) io.stage xs → InjectiveOutput (io.at s) BM BD →
      ∃ t, exec io.kernel.toAlgKernel s = some t ∧ Results io s t (f (s.pids 0) (s.pids 1) xs)) :
    io.Implements f := by
  intro A hd hc s xs hregions hu hb hp hinj
  have hu' : ∀ r a, s.undef r a = 0 := by rw [hu]; intros; rfl
  obtain ⟨t, he, hy⟩ := hrun s xs hu' hp hinj
  have bridge := A.exec_flatten hd hc io.kernel.toAlgKernel s (hsafe A.extent s hu' hb) hok hu
  refine ⟨A.flattenState t, ?_, ?_, ?_⟩
  · rw [bridge, he, Option.map_some]
  · intro i hi
    rw [A.flattenState_readMem hd t (by simp [hregions, IO.regions]) (hb.output i hi)]
    exact hy.output i hi
  · intro r offset h
    by_cases hr : r = A.flat
    · subst r
      simp only [FlatAlloc.flattenState]
      unfold FlatAlloc.readFlat
      cases hdec : A.decode offset with
      | none => rfl
      | some p =>
        obtain ⟨r,off⟩ := p
        obtain ⟨_, haddr, _⟩ := A.decode_sound hdec
        show A.trCell (t.mem r off) = A.trCell (s.mem r off)
        apply congrArg A.trCell
        apply hy.frame r off
        by_cases ho : r = io.out
        · right
          intro i hi heq
          rcases h with hn | hn
          · exact hn rfl
          · exact hn i hi (by rw [haddr, ho, heq])
        · exact Or.inl ho
    · simp only [FlatAlloc.flattenState, if_neg hr]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealIO
