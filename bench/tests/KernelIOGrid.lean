import VeriTile.Triton
import VeriTile.Triton.Memory.KernelCore
import VeriTile.Meta.StatementAudit
import VeriTile.Triton.Memory.TraceProofs

/-! A three-axis contract must pass the third program ID to its formula,
not just to the memory windows. This kernel stores that ID as a nat cell. -/
namespace VeriTile.Bench.Tests.KernelIOGrid
open VeriTile.Triton VeriTile.Triton.DSL
noncomputable section
set_option linter.unusedSimpArgs false

def source (out : Region .nat) : ComputeKernel := triton {
  tl.store($((out : Region .nat)), tl.program_id(2))
}

def io (out : Region .nat) : UKernelIO where
  kernel := source out
  projection := by rfl
  nIn := 0
  nOut := 1
  nScr := 0
  bufs := [Region.cast out]
  ity := Fin.elim0
  iarity := Fin.elim0
  ibuf := Fin.elim0
  oty := fun _ => .nat
  oarity := fun _ => 1
  obuf := fun _ => Region.cast out
  obuf_mem := by simp
  sarity := Fin.elim0
  sbuf := Fin.elim0
  iwin := fun i => Fin.elim0 i
  imask := fun i => Fin.elim0 i
  owin := fun _ _ _ _ _ _ => 0
  omask := fun _ _ _ _ _ _ => True
  swin := fun i => Fin.elim0 i
  smask := fun i => Fin.elim0 i

theorem third_axis_formula (out : Region .nat) :
    (io out).Implements3D (fun _ _ pid₂ _ _ _ => pid₂) := by
  apply UKernelIO.Implements3D.intro
  · simp [io, source, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
      Kernel.FlattenOk, StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]
  · intro bounds s vals _ _ ho _
    have hb := ho (0 : Fin 1) (0 : Fin 1) trivial
    simp only [io] at hb
    change Stmt.TraceSafeList bounds
      [.store .nat [] (.region out (.constNat 0)) (.programId 2) .none] s
    apply Stmt.TraceSafeList.cons_intro
    · rw [Stmt.TraceSafe]
      refine ⟨by simp [MemAccess.SafeAt, Op.SafeAt.eq_def],
        by simp [Op.SafeAt.eq_def], by simp [MaskOpt.SafeAt], ?_⟩
      exact TraceProofs.region_safe out (.constNat 0) bounds s (Tile.scalar 0) (by simp [evalOp.eq_def]) (fun _ => hb) _
    · intro _ _
      exact Stmt.TraceSafeList.nil_intro
  · intro s vals _ _
    refine ⟨s.writeMemTyped .nat (Region.cast out) 0 (s.pids 2), ?_, ?_, ?_⟩
    · change stepStmts [Stmt.store .nat [] (.region out (.constNat 0)) (.programId 2) .none] s = _
      have he : stepStmt (Stmt.store .nat [] (.region out (.constNat 0)) (.programId 2) .none) s =
          some (s.writeMemTyped .nat (Region.cast out) 0 (s.pids 2)) := by
        rw [stepStmt]
        simp only [evalOp.eq_def, bind, Option.bind_some, TileShape.allIndices,
          List.foldl_cons, List.foldl_nil, Tile.scalar]
        rfl
      rw [stepStmts.cons_some he, stepStmts.nil]
    · intro o j _
      simp [io, ChanTy.read]
    · intro r a ho _
      have h := ho (0 : Fin 1) (0 : Fin 1) trivial
      simp only [io] at h
      simp only [BlockState.writeMemTyped]
      rw [if_neg (by intro he; rcases h with hr | ha; exact hr he.1; exact ha he.2)]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Tests.KernelIOGrid
