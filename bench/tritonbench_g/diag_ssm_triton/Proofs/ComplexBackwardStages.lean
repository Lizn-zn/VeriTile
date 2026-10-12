import bench.tritonbench_g.diag_ssm_triton.Proofs.ComplexBackwardLoads
import bench.tritonbench_g.diag_ssm_triton.Proofs.ComplexBackwardPrevious
import bench.tritonbench_g.diag_ssm_triton.Proofs.ComplexBackwardUpdate
import VeriTile.Triton.Semantics.ProgramRewrite
namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton.ComplexBackwardStep
open VeriTile.Triton ComplexForwardStep
noncomputable section
set_option maxHeartbeats 1500000
set_option maxRecDepth 20000

/-- The three checked source fragments cover the entire original loop,
including the previous-state branch and both output component stores. -/
theorem source_parts (s y gx gy : RegionName) (length batch dim B : Nat) :
    ComplexBackwardSource.body s y gx gy length batch dim B =
      ComplexBackwardLoads.source gy length batch dim B ++
      ComplexBackwardPrevious.source s y batch dim B ++ ComplexBackwardUpdate.source gx B := rfl

theorem body_step (s y gx gy : RegionName) (length batch dim B k : Nat) (u : BlockState)
    (col : TileIndex [B] → Nat) (mask : TileIndex [B] → Bool)
    (carry acc transition : TileIndex [B] → ℂ)
    (hi : u.regs .nat [] "i" = some (Tile.scalar k))
    (h : Registers u col mask carry acc transition) :
    stepStmts (ComplexBackwardSource.body s y gx gy length batch dim B) u =
      some (result u s y gx gy batch dim (length-1-k) col mask carry acc transition) := by
  let t := length-1-k
  let ul := ComplexBackwardLoads.result u gy batch dim t col mask
  let up := ComplexBackwardPrevious.result ul s y batch dim t col mask
  let g := fun j => starRingEnd ℂ (input u gy (offset batch dim t col) mask j)
  let prev := previous u s y batch dim t col mask
  have hl := ComplexBackwardLoads.step gy length batch dim B k u col mask hi h.columns h.active
  have hp := ComplexBackwardPrevious.step s y batch dim B t ul col mask
    (by simp [ul,ComplexBackwardLoads.result])
    (by simpa [ul,ComplexBackwardLoads.result] using h.columns)
    (by simp [ul,ComplexBackwardLoads.result])
    (by simpa [ul,ComplexBackwardLoads.result] using h.active)
  have hr : Registers up col mask carry acc transition := by
    constructor
    · simpa [up,ul,ComplexBackwardPrevious.result,ComplexBackwardLoads.result] using h.columns
    · simpa [up,ul,ComplexBackwardPrevious.result,ComplexBackwardLoads.result] using h.active
    · simpa [up,ul,ComplexBackwardPrevious.result,ComplexBackwardLoads.result] using h.carryReal
    · simpa [up,ul,ComplexBackwardPrevious.result,ComplexBackwardLoads.result] using h.carryImag
    · simpa [up,ul,ComplexBackwardPrevious.result,ComplexBackwardLoads.result] using h.accReal
    · simpa [up,ul,ComplexBackwardPrevious.result,ComplexBackwardLoads.result] using h.accImag
    · simpa [up,ul,ComplexBackwardPrevious.result,ComplexBackwardLoads.result] using h.transitionReal
    · simpa [up,ul,ComplexBackwardPrevious.result,ComplexBackwardLoads.result] using h.transitionImag
  have hprev : previous ul s y batch dim t col mask = prev := rfl
  have hu := ComplexBackwardUpdate.step up gx col (offset batch dim t col) mask g prev carry acc transition hr
    (by simp [up,ul,ComplexBackwardPrevious.result,ComplexBackwardLoads.result])
    (by simp [up,ul,ComplexBackwardPrevious.result,ComplexBackwardLoads.result,g])
    (by simp [up,ul,ComplexBackwardPrevious.result,ComplexBackwardLoads.result,g])
    (by simp [up,ComplexBackwardPrevious.result,hprev])
    (by simp [up,ComplexBackwardPrevious.result,hprev])
  rw [source_parts]
  simp only [ProgramRewrite.append,hl,Option.bind_some]
  change (stepStmts (ComplexBackwardPrevious.source s y batch dim B) ul >>= fun v =>
    stepStmts (ComplexBackwardUpdate.source gx B) v) = _
  rw [hp]
  change stepStmts (ComplexBackwardUpdate.source gx B) up = _
  rw [hu]
  rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DiagSsmTriton.ComplexBackwardStep
