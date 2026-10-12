import bench.tritonbench_g.attention_fwd_triton1.Proofs.RealSafety
import VeriTile.Triton.Memory.KernelSpec.StreamAttentionReal

/-! Assemble the three input streams and both output windows. The H window
is inactive when STORE is false, so its entire memory is covered by the
frame. Mathematical formulas use the input streams, never kernel results. -/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton1.RealIO
open VeriTile.Triton
open scoped StreamMasked3DKernelIO₃ₓ₂
set_option linter.unusedSimpArgs false
set_option maxHeartbeats 2400000
set_option maxRecDepth 10000

def originalIO (Q K V H O : RegionName) (s_qh s_hh s_ht : Nat)
    (scale : ℝ) (BT BD NT : Nat) (STORE IFCOND : Bool) : StreamMasked3DKernelIO₃ₓ₂ where
  kernel := (TypedSource.original Q K V H O s_qh BD 1 s_hh s_ht (NT * BT)
    scale BT BD NT STORE IFCOND).eraseDType
  projection := RealProjection.original_projection_supported ..
  inp1 := Q
  inp2 := K
  inp3 := V
  out1 := O
  out2 := H
  T := NT
  B1 := BT * BD
  B2 := BD * BT
  B3 := BT * BD
  C1 := NT * (BT * BD)
  C2 := NT * (BD * BD)
  out1DType := .real
  out2DType := .real
  pre := fun _ _ _ => Q ≠ O ∧ K ≠ O ∧ V ≠ O ∧ Q ≠ H ∧ K ≠ H ∧ V ≠ H ∧ H ≠ O
  read1 := fun p _ _ c j => p * s_qh + (c.val * BT + j.val / BD) * BD + j.val % BD
  read2 := fun p _ _ c j => p * s_qh + j.val / BT + (c.val * BT + j.val % BT) * BD
  read3 := fun p _ _ c j => p * s_qh + (c.val * BT + j.val / BD) * BD + j.val % BD
  write1 := fun p _ _ j =>
    let chunk := Lane2D.decode j
    let cell := Lane2D.decode chunk.2.1
    p * s_qh + (chunk.1.val * BT + cell.1.val) * BD + cell.2.1.val
  write2 := fun p _ _ j =>
    let chunk := Lane2D.decode j
    let cell := Lane2D.decode chunk.2.1
    p * s_hh + (chunk.1.val * BD + cell.1.val) * s_ht + cell.2.1.val
  mask1 := fun _ _ _ _ _ => True
  mask2 := fun _ _ _ _ _ => True
  mask3 := fun _ _ _ _ _ => True
  writeMask1 := fun _ _ _ _ => True
  writeMask2 := fun _ _ _ _ => STORE = Bool.true

def scalarLeftIO (Q K V H O : RegionName) (s_qh s_hh s_ht : Nat)
    (scale : ℝ) (BT BD NT : Nat) (STORE IFCOND : Bool) : StreamMasked3DKernelIO₃ₓ₂ :=
  { originalIO Q K V H O s_qh s_hh s_ht scale BT BD NT STORE IFCOND with
    kernel := (TypedSource.scalarLeft Q K V H O s_qh BD 1 s_hh s_ht (NT * BT)
      scale BT BD NT STORE IFCOND).eraseDType
    projection := RealProjection.scalarLeft_projection_supported .. }

/-- Flatten the chunk index and a matrix cell into one output-window lane. -/
def emittedLane {NT rows cols : Nat} (c : Fin NT) (i : Fin rows) (j : Fin cols) :
    Fin (NT * (rows * cols)) :=
  Lane2D.encode (c, Lane2D.encode (i, j, PUnit.unit), PUnit.unit)

/-- H contains the sum of KᵀV from chunks strictly before this one. -/
noncomputable def expectedCarry (NT BT BD : Nat)
    (ys : Fin NT → Fin (BD * BT) → ℝ) (zs : Fin NT → Fin (BT * BD) → ℝ)
    (c : Fin NT) (i j : Fin BD) : ℝ :=
  ∑ past : Fin c.val, ∑ t : Fin BT,
    ys (Fin.castLE (Nat.le_of_lt c.isLt) past) (Lane2D.encode (i, t, PUnit.unit)) *
      zs (Fin.castLE (Nat.le_of_lt c.isLt) past) (Lane2D.encode (t, j, PUnit.unit))

/-- O is the local contribution plus Q times the pre-update carry. -/
noncomputable def expectedOutput (scale : ℝ) (NT BT BD : Nat)
    (xs : Fin NT → Fin (BT * BD) → ℝ) (ys : Fin NT → Fin (BD * BT) → ℝ)
    (zs : Fin NT → Fin (BT * BD) → ℝ) (c : Fin NT) (i : Fin BT) (j : Fin BD) : ℝ :=
  (∑ t : Fin BT,
    (∑ e : Fin BD, (xs c (Lane2D.encode (i, e, PUnit.unit)) * scale) *
      ys c (Lane2D.encode (e, t, PUnit.unit))) * zs c (Lane2D.encode (t, j, PUnit.unit))) +
  ∑ e : Fin BD, (xs c (Lane2D.encode (i, e, PUnit.unit)) * scale) *
    expectedCarry NT BT BD ys zs c e j

noncomputable def expected (scale : ℝ) (NT BT BD : Nat)
    (xs : Fin NT → Fin (BT * BD) → ℝ) (ys : Fin NT → Fin (BD * BT) → ℝ)
    (zs : Fin NT → Fin (BT * BD) → ℝ) :
    (Fin (NT * (BT * BD)) → ℝ) × (Fin (NT * (BD * BD)) → ℝ) :=
  (fun lane =>
    let chunk := Lane2D.decode lane
    let cell := Lane2D.decode chunk.2.1
    expectedOutput scale NT BT BD xs ys zs chunk.1 cell.1 cell.2.1,
   fun lane =>
    let chunk := Lane2D.decode lane
    let cell := Lane2D.decode chunk.2.1
    expectedCarry NT BT BD ys zs chunk.1 cell.1 cell.2.1)

theorem output_formula (Q K V : RegionName) (s₀ : BlockState)
    (s_qh : Nat) (scale : ℝ) (BT BD NT : Nat)
    (xs : Fin NT → Fin (BT * BD) → ℝ) (ys : Fin NT → Fin (BD * BT) → ℝ)
    (zs : Fin NT → Fin (BT * BD) → ℝ)
    (hx : ∀ (t : Fin NT) (j : Fin (BT * BD)),
      s₀.readMem Q (s₀.pids 0 * s_qh + (t.val * BT + j.val / BD) * BD
        + j.val % BD) = xs t j)
    (hy : ∀ (t : Fin NT) (j : Fin (BD * BT)),
      s₀.readMem K (s₀.pids 0 * s_qh + j.val / BT + (t.val * BT + j.val % BT)
        * BD) = ys t j)
    (hz : ∀ (t : Fin NT) (j : Fin (BT * BD)),
      s₀.readMem V (s₀.pids 0 * s_qh + (t.val * BT + j.val / BD) * BD
        + j.val % BD) = zs t j)
    (c : Fin NT) (j : Fin (BT * BD)) :
    expectedOutput scale NT BT BD xs ys zs c (Lane2D.decode j).1 (Lane2D.decode j).2.1
      = aft1OutG s₀ Q K V s_qh scale BT BD c.val (Lane2D.decode j).1
          (Lane2D.decode j).2.1 := by
  have hq : ∀ (t : Fin NT) (r : Fin BT) (e : Fin BD),
      xs t (Lane2D.encode (r, e, PUnit.unit))
        = s₀.readMem Q (s₀.pids 0 * s_qh + (t.val * BT + r.val) * BD + e.val) := by
    intro t r e
    rw [← hx t (Lane2D.encode (r, e, PUnit.unit))]
    simp only [Lane2D.encode_div, Lane2D.encode_mod]
  have hk : ∀ (t : Fin NT) (e : Fin BD) (tk : Fin BT),
      ys t (Lane2D.encode (e, tk, PUnit.unit))
        = s₀.readMem K (s₀.pids 0 * s_qh + e.val + (t.val * BT + tk.val) * BD) := by
    intro t e tk
    rw [← hy t (Lane2D.encode (e, tk, PUnit.unit))]
    simp only [Lane2D.encode_div, Lane2D.encode_mod]
  have hv : ∀ (t : Fin NT) (tk : Fin BT) (d : Fin BD),
      zs t (Lane2D.encode (tk, d, PUnit.unit))
        = s₀.readMem V (s₀.pids 0 * s_qh + (t.val * BT + tk.val) * BD + d.val) := by
    intro t tk d
    rw [← hz t (Lane2D.encode (tk, d, PUnit.unit))]
    simp only [Lane2D.encode_div, Lane2D.encode_mod]
  simp only [expectedOutput, expectedCarry, aft1OutG, aft1LocalOutG, aft1RecOutG, aft1RecStateG,
    aft1QCellG, aft1KCellG, aft1VCellG]
  refine congrArg₂ (· + ·) ?_ ?_
  · refine Finset.sum_congr rfl fun tk _ => ?_
    refine congrArg₂ (· * ·) (Finset.sum_congr rfl fun e _ => ?_)
      (hv c tk (Lane2D.decode j).2.1)
    exact congrArg₂ (· * ·)
      (congrArg₂ (· * ·) (hq c (Lane2D.decode j).1 e) rfl) (hk c e tk)
  · refine Finset.sum_congr rfl fun d' _ => ?_
    refine congrArg₂ (· * ·)
      (congrArg₂ (· * ·) (hq c (Lane2D.decode j).1 d') rfl) ?_
    rw [← Fin.sum_univ_eq_sum_range (fun u => ∑ tk : Fin BT,
      s₀.readMem K (s₀.pids 0 * s_qh + d'.val + (u * BT + tk.val) * BD)
        * s₀.readMem V (s₀.pids 0 * s_qh + (u * BT + tk.val) * BD
            + (Lane2D.decode j).2.1.val)) c.val]
    refine Finset.sum_congr rfl fun u _ => Finset.sum_congr rfl fun tk _ => ?_
    exact congrArg₂ (· * ·)
      (hk (Fin.castLE (Nat.le_of_lt c.isLt) u) d' tk)
      (hv (Fin.castLE (Nat.le_of_lt c.isLt) u) tk (Lane2D.decode j).2.1)

theorem carry_formula (K V : RegionName) (s : BlockState) (s_qh BT BD NT : Nat)
    (ys : Fin NT → Fin (BD * BT) → ℝ) (zs : Fin NT → Fin (BT * BD) → ℝ)
    (hy : ∀ (t : Fin NT) (j : Fin (BD * BT)),
      s.readMem K (s.pids 0 * s_qh + j.val / BT + (t.val * BT + j.val % BT) * BD) = ys t j)
    (hz : ∀ (t : Fin NT) (j : Fin (BT * BD)),
      s.readMem V (s.pids 0 * s_qh + (t.val * BT + j.val / BD) * BD + j.val % BD) = zs t j)
    (c : Fin NT) (i j : Fin BD) :
    expectedCarry NT BT BD ys zs c i j = aft1RecStateG s K V s_qh BT BD c.val i j := by
  simp only [expectedCarry, aft1RecStateG, aft1KCellG, aft1VCellG]
  rw [← Fin.sum_univ_eq_sum_range (fun u => ∑ t : Fin BT,
    s.readMem K (s.pids 0 * s_qh + i.val + (u * BT + t.val) * BD) *
      s.readMem V (s.pids 0 * s_qh + (u * BT + t.val) * BD + j.val)) c.val]
  refine Finset.sum_congr rfl fun past _ => Finset.sum_congr rfl fun t _ => ?_
  rw [← hy (Fin.castLE (Nat.le_of_lt c.isLt) past) (Lane2D.encode (i, t, PUnit.unit)),
    ← hz (Fin.castLE (Nat.le_of_lt c.isLt) past) (Lane2D.encode (t, j, PUnit.unit))]
  simp only [Lane2D.encode_div, Lane2D.encode_mod]
  rfl

theorem original_flattenOk (Q K V H O : RegionName) (s_qh s_hh s_ht : Nat)
    (scale : ℝ) (BT BD NT : Nat) (STORE IFCOND : Bool) :
    (originalIO Q K V H O s_qh s_hh s_ht scale BT BD NT STORE IFCOND).kernel.toAlgKernel.FlattenOk := by
  simp only [originalIO, RealProjection.original_matches_transcription]
  unfold Kernel.FlattenOk
  rw [attention_fwd_triton1_body_splitG]
  simp [aft1PrologueG, aft1LoopBodyG, aft1StopOpG, StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk]
  simp [Op.FlattenOk.eq_def]

theorem scalarLeft_flattenOk (Q K V H O : RegionName) (s_qh s_hh s_ht : Nat)
    (scale : ℝ) (BT BD NT : Nat) (STORE IFCOND : Bool) :
    (scalarLeftIO Q K V H O s_qh s_hh s_ht scale BT BD NT STORE IFCOND).kernel.toAlgKernel.FlattenOk := by
  unfold Kernel.FlattenOk
  simp only [scalarLeftIO, RealProjection.scalarLeft_matches_body]
  simp [RealProjection.scalarLeftBody, aft1PrologueG, aft1LoopBodyG, aft1StopOpG,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk]
  simp [Op.FlattenOk.eq_def]

/-- The complete original implements the independent formulas through the
flat-memory IO interface, with H disabled or active according to STORE. -/
theorem original_correct (Q K V H O : RegionName) (s_qh s_hh s_ht : Nat)
    (scale : ℝ) (BT BD NT : Nat) (STORE IFCOND : Bool)
    (hBT : 0 < BT) (hstride : BD ≤ s_ht) :
    originalIO Q K V H O s_qh s_hh s_ht scale BT BD NT STORE IFCOND ⊨
      fun _ _ _ xs ys zs => expected scale NT BT BD xs ys zs := by
  refine StreamMasked3DKernelIO₃ₓ₂.Implements.intro _
    (original_flattenOk Q K V H O s_qh s_hh s_ht scale BT BD NT STORE IFCOND) ?_ ?_
  · intro bounds st xs ys zs hpre _hx _hy _hz hbQ hbK hbV hbO hbH
    rcases hpre with ⟨hQO, hKO, hVO, hQH, hKH, hVH, hHO⟩
    apply RealSafety.original_safe bounds Q K V H O s_qh s_hh s_ht scale BT BD NT
      STORE IFCOND st hBT hstride hQO hKO hVO hQH hKH hVH hHO
    · intro c hc i j
      simpa only [originalIO, Lane2D.encode_div, Lane2D.encode_mod] using
        hbQ ⟨c, hc⟩ (Lane2D.encode (i, j, PUnit.unit)) trivial
    · intro c hc i j
      simpa only [originalIO, Lane2D.encode_div, Lane2D.encode_mod] using
        hbK ⟨c, hc⟩ (Lane2D.encode (i, j, PUnit.unit)) trivial
    · intro c hc i j
      simpa only [originalIO, Lane2D.encode_div, Lane2D.encode_mod] using
        hbV ⟨c, hc⟩ (Lane2D.encode (i, j, PUnit.unit)) trivial
    · intro c hc i j
      simpa only [originalIO, emittedLane, Lane2D.decode_encode] using
        hbO (emittedLane ⟨c, hc⟩ i j) trivial
    · intro hs c hc i j
      simpa only [originalIO, emittedLane, Lane2D.decode_encode] using
        hbH (emittedLane ⟨c, hc⟩ i j) hs
  · intro st xs ys zs hpre _hu hx hy hz
    rcases hpre with ⟨hQO, hKO, hVO, hQH, hKH, hVH, hHO⟩
    obtain ⟨final, hrun, hinv⟩ := RealLoop.original_executes Q K V H O s_qh s_hh s_ht scale
      BT BD NT STORE IFCOND st hBT hstride hQO hKO hVO hQH hKH hVH hHO
    refine ⟨final, hrun, ?_, ?_, ?_⟩
    · intro lane _
      let chunk := Lane2D.decode lane
      let cell := Lane2D.decode chunk.2.1
      exact (hinv.output chunk.1.val chunk.1.isLt cell.1 cell.2.1).trans
        (output_formula Q K V st s_qh scale BT BD NT xs ys zs
          (fun t j => hx t j trivial) (fun t j => hy t j trivial) (fun t j => hz t j trivial)
          chunk.1 chunk.2.1).symm
    · intro lane hs
      let chunk := Lane2D.decode lane
      let cell := Lane2D.decode chunk.2.1
      exact (hinv.snapshots hs chunk.1.val chunk.1.isLt cell.1 cell.2.1).trans
        (carry_formula K V st s_qh BT BD NT ys zs
          (fun t j => hy t j trivial) (fun t j => hz t j trivial) chunk.1 cell.1 cell.2.1).symm
    · intro rg off hmiss
      apply hinv.frame rg off
      · by_cases hr : rg = O
        · right
          intro c hc i j
          simpa only [originalIO, emittedLane, Lane2D.decode_encode] using
            hmiss.1 (emittedLane ⟨c, hc⟩ i j) trivial hr
        · exact Or.inl hr
      · by_cases hs : STORE = Bool.true
        · by_cases hr : rg = H
          · right; right
            intro c hc i j
            simpa only [originalIO, emittedLane, Lane2D.decode_encode] using
              hmiss.2 (emittedLane ⟨c, hc⟩ i j) hs hr
          · exact Or.inr (Or.inl hr)
        · exact Or.inl (Bool.eq_false_iff.mpr hs)

/-- The scalar-left candidate implements the independent formulas through the
flat-memory IO interface, with H disabled or active according to STORE. -/
theorem scalarLeft_correct (Q K V H O : RegionName) (s_qh s_hh s_ht : Nat)
    (scale : ℝ) (BT BD NT : Nat) (STORE IFCOND : Bool)
    (hBT : 0 < BT) (hstride : BD ≤ s_ht) :
    scalarLeftIO Q K V H O s_qh s_hh s_ht scale BT BD NT STORE IFCOND ⊨
      fun _ _ _ xs ys zs => expected scale NT BT BD xs ys zs := by
  refine StreamMasked3DKernelIO₃ₓ₂.Implements.intro _
    (scalarLeft_flattenOk Q K V H O s_qh s_hh s_ht scale BT BD NT STORE IFCOND) ?_ ?_
  · intro bounds st xs ys zs hpre _hx _hy _hz hbQ hbK hbV hbO hbH
    rcases hpre with ⟨hQO, hKO, hVO, hQH, hKH, hVH, hHO⟩
    apply RealSafety.candidate_safe bounds Q K V H O s_qh s_hh s_ht scale BT BD NT
      STORE IFCOND st hBT hstride hQO hKO hVO hQH hKH hVH hHO
    · intro c hc i j
      simpa only [scalarLeftIO, originalIO, Lane2D.encode_div, Lane2D.encode_mod] using
        hbQ ⟨c, hc⟩ (Lane2D.encode (i, j, PUnit.unit)) trivial
    · intro c hc i j
      simpa only [scalarLeftIO, originalIO, Lane2D.encode_div, Lane2D.encode_mod] using
        hbK ⟨c, hc⟩ (Lane2D.encode (i, j, PUnit.unit)) trivial
    · intro c hc i j
      simpa only [scalarLeftIO, originalIO, Lane2D.encode_div, Lane2D.encode_mod] using
        hbV ⟨c, hc⟩ (Lane2D.encode (i, j, PUnit.unit)) trivial
    · intro c hc i j
      simpa only [scalarLeftIO, originalIO, emittedLane, Lane2D.decode_encode] using
        hbO (emittedLane ⟨c, hc⟩ i j) trivial
    · intro hs c hc i j
      simpa only [scalarLeftIO, originalIO, emittedLane, Lane2D.decode_encode] using
        hbH (emittedLane ⟨c, hc⟩ i j) hs
  · intro st xs ys zs hpre _hu hx hy hz
    rcases hpre with ⟨hQO, hKO, hVO, hQH, hKH, hVH, hHO⟩
    obtain ⟨final, hrun, hinv⟩ := RealCandidate.executes Q K V H O s_qh s_hh s_ht scale
      BT BD NT STORE IFCOND st hBT hstride hQO hKO hVO hQH hKH hVH hHO
    refine ⟨final, hrun, ?_, ?_, ?_⟩
    · intro lane _
      let chunk := Lane2D.decode lane
      let cell := Lane2D.decode chunk.2.1
      exact (hinv.output chunk.1.val chunk.1.isLt cell.1 cell.2.1).trans
        (output_formula Q K V st s_qh scale BT BD NT xs ys zs
          (fun t j => hx t j trivial) (fun t j => hy t j trivial) (fun t j => hz t j trivial)
          chunk.1 chunk.2.1).symm
    · intro lane hs
      let chunk := Lane2D.decode lane
      let cell := Lane2D.decode chunk.2.1
      exact (hinv.snapshots hs chunk.1.val chunk.1.isLt cell.1 cell.2.1).trans
        (carry_formula K V st s_qh BT BD NT ys zs
          (fun t j => hy t j trivial) (fun t j => hz t j trivial) chunk.1 cell.1 cell.2.1).symm
    · intro rg off hmiss
      apply hinv.frame rg off
      · by_cases hr : rg = O
        · right
          intro c hc i j
          simpa only [scalarLeftIO, originalIO, emittedLane, Lane2D.decode_encode] using
            hmiss.1 (emittedLane ⟨c, hc⟩ i j) trivial hr
        · exact Or.inl hr
      · by_cases hs : STORE = Bool.true
        · by_cases hr : rg = H
          · right; right
            intro c hc i j
            simpa only [scalarLeftIO, originalIO, emittedLane, Lane2D.decode_encode] using
              hmiss.2 (emittedLane ⟨c, hc⟩ i j) hs hr
          · exact Or.inr (Or.inl hr)
        · exact Or.inl (Bool.eq_false_iff.mpr hs)

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton1.RealIO
