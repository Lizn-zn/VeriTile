import bench.tritonbench_g.softmax_optimize.Proofs.RealStoreFrame
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.SoftmaxOptimize
open VeriTile.Triton
set_option maxHeartbeats 5000000
set_option linter.unusedVariables false

/-- Exact raw-cell frame outside the active output row. -/
def RowFrame (Y : RegionName) (N : Nat) (s0 s : BlockState) : Prop :=
  ∀ r o, (r ≠ Y ∨ o < s0.pid*N ∨ s0.pid*N+N ≤ o) → s.mem r o = s0.mem r o

theorem pass2LoopA_run_frame (N TILE_N : Nat) (hT : 0 < TILE_N)
    (output_ptr input_ptr : RegionName) (hne : output_ptr ≠ input_ptr)
    (cc S : ℝ) (s0 sInit : BlockState)
    (hStop : evalOp (Op.ref .nat [] "prev_multiple") sInit
        = some (Tile.scalar (prevMultiple N TILE_N)))
    (hFrame : RowFrame output_ptr N s0 sInit)
    (hInv : pass2InvA output_ptr input_ptr N TILE_N cc S s0 0 sInit) :
    ∃ sFinal,
      stepStmt (.forRangeDyn "start_n" (Op.constNat 0) (Op.ref .nat [] "prev_multiple")
          (Op.constNat TILE_N) (loopApass2body output_ptr input_ptr N TILE_N)) sInit = some sFinal ∧
      pass2InvA output_ptr input_ptr N TILE_N cc S s0 (prevMultiple N TILE_N) sFinal ∧ RowFrame output_ptr N s0 sFinal := by
  apply forRangeDyn_inv_aligned (start := 0) (stop := prevMultiple N TILE_N) (step := TILE_N)
    (P := fun i t ↦ pass2InvA output_ptr input_ptr N TILE_N cc S s0 i t ∧
      RowFrame output_ptr N s0 t)
    (by simp [evalOp]) hStop (by simp [evalOp]) hT.ne' (Nat.zero_le _)
    (Dvd.intro 0 rfl) (TILE_N_dvd_prevMultiple N TILE_N) ⟨hInv, hFrame⟩
  intro i s hi ⟨hPi, hFrame⟩
  obtain ⟨⟨hpids, hpid, hm, hz, hinpEq, hout, hprevInv⟩, hdvd⟩ := hPi
  have hpidEq : s.pid = s0.pid := by unfold BlockState.pid; rw [hpids]
  have halign : i = (i / TILE_N) * TILE_N := (Nat.div_mul_cancel hdvd).symm
  set c := i / TILE_N with hc
  have hpidS : s.regs .nat [] "pid_m" = some (Tile.scalar s.pid) := by rw [hpid, hpidEq]
  obtain ⟨s', hstep, hpid', hm', hz', hpids', hother', hwritten, hpreserved, hprev'⟩ :=
    loopApass2_body_step N TILE_N output_ptr input_ptr hne c cc S s hpidS hm hz
  have hNext : pass2InvA output_ptr input_ptr N TILE_N cc S s0 (i+TILE_N) s' := by
    constructor
    · -- new pass2Inv at i + TILE_N
      refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
      · rw [hpids']; exact hpids
      · rw [hpid', hpidEq]
      · exact hm'
      · exact hz'
      · -- input region untouched
        intro o
        rw [hother' input_ptr o hne.symm, hinpEq]
      · -- output readback advances by one tile
        intro j hj
        by_cases hjt : j < i + TILE_N
        · -- in the newly written tile, or earlier
          rw [if_pos hjt]
          by_cases hji : j < i
          · -- earlier tile: this lane's offset s0.pid*N+j was not touched by tile c
            rw [hpreserved (s0.pid * N + j) (by
              intro idx
              rw [pass2Off, hpidEq]
              -- s.pid*N + (c*TILE_N + idx) ≠ s0.pid*N + j  since c*TILE_N + idx ≥ i > j
              have : i ≤ c * TILE_N + idx.1.val := by rw [← halign]; exact Nat.le_add_right _ _
              omega)]
            rw [hout j hj, if_pos hji]
          · -- newly written tile: j = c*TILE_N + ℓ for ℓ = j - i
            have hjr : i ≤ j := Nat.le_of_not_lt hji
            have hlt : j - i < TILE_N := by omega
            set ℓ : TileIndex [TILE_N] := (⟨j - i, hlt⟩, PUnit.unit) with hℓ
            have hcol : c * TILE_N + ℓ.1.val = j := by
              show c * TILE_N + (j - i) = j; rw [← halign]; omega
            have hoff : pass2Off N TILE_N c s ℓ = s0.pid * N + j := by
              rw [pass2Off, hpidEq, hcol]
            have hw := hwritten ℓ
            rw [hoff] at hw
            rw [hw, pass2Val, hcol, hpidEq, hinpEq]
        · -- j ≥ i + TILE_N: untouched this tile, was untouched before too
          rw [if_neg hjt]
          rw [hpreserved (s0.pid * N + j) (by
            intro idx
            rw [pass2Off, hpidEq]
            have : c * TILE_N + idx.1.val < i + TILE_N := by
              rw [← halign]; have := idx.2; omega
            omega)]
          rw [hout j hj, if_neg (by omega)]
      · rw [hprev', hprevInv]
    · exact Dvd.dvd.add hdvd (dvd_refl _)
  obtain ⟨t, ht, hf⟩ := loopApass2_raw_frame N TILE_N output_ptr input_ptr hne c cc S s hpidS hm hz
  have heq : t = s' := Option.some.inj (ht.symm.trans hstep)
  subst t
  refine ⟨s', ?_, hNext, ?_⟩
  · rw [← halign] at hstep
    exact hstep
  · intro r o houtside
    rw [hf r o ?_]
    · exact hFrame r o houtside
    · intro lane
      intro he
      have hactive : c*TILE_N+lane.1.val<N :=
        loopA_inRange N TILE_N hT c lane.1 (by simpa [← halign] using hi)
      have hoff : o = s0.pid*N+(c*TILE_N+lane.1.val) := by
        rw [← he.2, pass2Off, hpidEq]
      rcases houtside with hregion | hlow | hhigh
      · exact hregion he.1.symm
      · omega
      · omega


theorem pass2LoopB_run_frame (N TILE_N : Nat) (hT : 0 < TILE_N)
    (output_ptr input_ptr : RegionName) (hne : output_ptr ≠ input_ptr)
    (cc S : ℝ) (s0 sInit : BlockState)
    (hStart : evalOp (Op.ref .nat [] "prev_multiple") sInit
        = some (Tile.scalar (prevMultiple N TILE_N)))
    (hFrame : RowFrame output_ptr N s0 sInit)
    (hInv : pass2InvA output_ptr input_ptr N TILE_N cc S s0 (prevMultiple N TILE_N) sInit) :
    ∃ final sFinal,
      stepStmt (.forRangeDyn "start_n" (Op.ref .nat [] "prev_multiple") (Op.constNat N)
          (Op.constNat TILE_N) (loopBpass2body output_ptr input_ptr N TILE_N)) sInit = some sFinal ∧
      N ≤ final ∧
      pass2InvA output_ptr input_ptr N TILE_N cc S s0 final sFinal ∧ RowFrame output_ptr N s0 sFinal := by
  apply forRangeDyn_inv (start := prevMultiple N TILE_N) (stop := N) (step := TILE_N)
    (P := fun i t ↦ pass2InvA output_ptr input_ptr N TILE_N cc S s0 i t ∧
      RowFrame output_ptr N s0 t)
    hStart (by simp [evalOp]) (by simp [evalOp]) hT.ne' ⟨hInv, hFrame⟩
  intro i s hi ⟨hPi, hFrame⟩
  obtain ⟨⟨hpids, hpid, hm, hz, hinpEq, hout, hprevInv⟩, hdvd⟩ := hPi
  have hpidEq : s.pid = s0.pid := by unfold BlockState.pid; rw [hpids]
  have halign : i = (i / TILE_N) * TILE_N := (Nat.div_mul_cancel hdvd).symm
  set c := i / TILE_N with hc
  have hpidS : s.regs .nat [] "pid_m" = some (Tile.scalar s.pid) := by rw [hpid, hpidEq]
  obtain ⟨s', hstep, hpid', hm', hz', hpids', hother', hwritten, hpreserved, hprev'⟩ :=
    loopBpass2_body_step N TILE_N output_ptr input_ptr hne c cc S s hpidS hm hz
  have hNext : pass2InvA output_ptr input_ptr N TILE_N cc S s0 (i+TILE_N) s' := by
    constructor
    · refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
      · rw [hpids']; exact hpids
      · rw [hpid', hpidEq]
      · exact hm'
      · exact hz'
      · intro o; rw [hother' input_ptr o hne.symm, hinpEq]
      · intro j hj
        by_cases hjt : j < i + TILE_N
        · rw [if_pos hjt]
          by_cases hji : j < i
          · -- earlier column: untouched by this tile
            rw [hpreserved (s0.pid * N + j) (by
              intro idx hidx
              rw [pass2Off, hpidEq]
              have : i ≤ c * TILE_N + idx.1.val := by rw [← halign]; exact Nat.le_add_right _ _
              omega)]
            rw [hout j hj, if_pos hji]
          · -- newly written column: j = c*TILE_N + ℓ, and j < N so mask active
            have hjr : i ≤ j := Nat.le_of_not_lt hji
            have hlt : j - i < TILE_N := by omega
            set ℓ : TileIndex [TILE_N] := (⟨j - i, hlt⟩, PUnit.unit) with hℓ
            have hcol : c * TILE_N + ℓ.1.val = j := by
              show c * TILE_N + (j - i) = j; rw [← halign]; omega
            have hinr : c * TILE_N + ℓ.1.val < N := by rw [hcol]; exact hj
            have hoff : pass2Off N TILE_N c s ℓ = s0.pid * N + j := by
              rw [pass2Off, hpidEq, hcol]
            have hw := hwritten ℓ hinr
            rw [hoff] at hw
            rw [hw, pass2Val, hcol, hpidEq, hinpEq]
        · rw [if_neg hjt]
          rw [hpreserved (s0.pid * N + j) (by
            intro idx hidx
            rw [pass2Off, hpidEq]
            have : c * TILE_N + idx.1.val < i + TILE_N := by
              rw [← halign]; have := idx.2; omega
            omega)]
          rw [hout j hj, if_neg (by omega)]
      · rw [hprev', hprevInv]
    · exact Dvd.dvd.add hdvd (dvd_refl _)
  obtain ⟨t, ht, hf⟩ := loopBpass2_raw_frame N TILE_N output_ptr input_ptr hne c cc S s hpidS hm hz
  have heq : t = s' := Option.some.inj (ht.symm.trans hstep)
  subst t
  refine ⟨s', ?_, hNext, ?_⟩
  · rw [← halign] at hstep
    exact hstep
  · intro r o houtside
    rw [hf r o ?_]
    · exact hFrame r o houtside
    · intro lane hactive
      intro he
      have hoff : o = s0.pid*N+(c*TILE_N+lane.1.val) := by
        rw [← he.2, pass2Off, hpidEq]
      rcases houtside with hregion | hlow | hhigh
      · exact hregion he.1.symm
      · omega
      · omega


#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxOptimize
