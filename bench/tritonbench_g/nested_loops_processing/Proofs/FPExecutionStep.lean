import bench.tritonbench_g.nested_loops_processing.Proofs.SourceChecks
import VeriTile.Triton.Float.ProfiledSoundness

namespace VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.TypedExecution
open VeriTile.Triton FP.Structural TypedSource
set_option maxHeartbeats 1500000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

abbrev PointerTile := TileIndex [2,2] → RegionName × Nat
abbrev Payload (α : Type) (storage : Storage) := Values α (SourceChecks.dtype storage) [2,2]

def bump (ptr : PointerTile) (delta : Nat) : PointerTile :=
  fun i ↦ ((ptr i).1, (ptr i).2 + delta)

def loaded {α : Type} [Inhabited α] (M : Algebra α) :
    (storage : Storage) → State α → PointerTile → Payload α storage
  | .int8, s, ptr => fun i ↦ (s.mem (ptr i).1 (ptr i).2).read .int
  | .int16, s, ptr => fun i ↦ (s.mem (ptr i).1 (ptr i).2).read .int
  | .int32, s, ptr => fun i ↦ (s.mem (ptr i).1 (ptr i).2).read .int
  | .int64, s, ptr => fun i ↦ (s.mem (ptr i).1 (ptr i).2).read .int
  | .uint8, s, ptr => fun i ↦ (s.mem (ptr i).1 (ptr i).2).read .nat
  | .uint16, s, ptr => fun i ↦ (s.mem (ptr i).1 (ptr i).2).read .nat
  | .uint32, s, ptr => fun i ↦ (s.mem (ptr i).1 (ptr i).2).read .nat
  | .uint64, s, ptr => fun i ↦ (s.mem (ptr i).1 (ptr i).2).read .nat
  | .boolean, s, ptr => fun i ↦ (s.mem (ptr i).1 (ptr i).2).read .bool
  | .fp16, s, ptr => fun i ↦ (s.mem (ptr i).1 (ptr i).2).read .fp16
  | .bf16, s, ptr => fun i ↦ (s.mem (ptr i).1 (ptr i).2).read .bf16
  | .fp64, s, ptr => fun i ↦ (s.mem (ptr i).1 (ptr i).2).read .real
  | .f8e4, s, ptr => fun i ↦ (s.mem (ptr i).1 (ptr i).2).read .f8e4
  | .f8e5, s, ptr => fun i ↦ (s.mem (ptr i).1 (ptr i).2).read .f8e5
  | .fp32, s, ptr => fun i ↦ M.fp32Load ((s.mem (ptr i).1 (ptr i).2).read .real)

def scatter {α : Type} (storage : Storage) (s : State α)
    (ptr : PointerTile) (value : Payload α storage) : State α :=
  (TileShape.allIndices [2,2]).foldl (fun t i ↦
    t.write (ptr i).1 (ptr i).2 (.mk (SourceChecks.dtype storage) (value i))) s

@[simp] theorem foldl_write_regs {α ι : Type} (region : ι → RegionName) (offset : ι → Nat)
    (value : ι → Cell α) (lanes : List ι) (s : State α) :
    (lanes.foldl (fun t i ↦ t.write (region i) (offset i) (value i)) s).regs = s.regs := by
  induction lanes generalizing s with
  | nil => rfl
  | cons i lanes ih => rw [List.foldl_cons, ih]; rfl

def innerState {α : Type} [Inhabited α] (M : Algebra α) (storage : Storage)
    (s : State α) (a c : PointerTile) (v1 v2 : Payload α storage) (delta : Nat) : State α :=
  let a' := bump a delta
  let v3 := loaded M storage s a'
  let t := s.setReg "a_ptrs" .ptr [2,2] a'
  let t := t.setReg "a3" (SourceChecks.dtype storage) [2,2] v3
  let t := scatter storage t c v1
  let t := t.setReg "c_ptrs" .ptr [2,2] (bump c delta)
  let t := scatter storage t (bump c delta) v2
  let t := t.setReg "c_ptrs" .ptr [2,2] (bump (bump c delta) delta)
  let t := scatter storage t (bump (bump c delta) delta) v3
  t.setReg "c_ptrs" .ptr [2,2] (bump (bump (bump c delta) delta) delta)

theorem inner_step {α : Type} [Inhabited α] (M : Algebra α) (storage : Storage)
    (src dst : RegionName) (sm sn : Int) (s : State α)
    (a c : PointerTile) (v1 v2 : Payload α storage)
    (ha : s.regs .ptr [2,2] "a_ptrs" = some a)
    (hc : s.regs .ptr [2,2] "c_ptrs" = some c)
    (h1 : s.regs (SourceChecks.dtype storage) [2,2] "a1" = some v1)
    (h2 : s.regs (SourceChecks.dtype storage) [2,2] "a2" = some v2) :
    run M (SourceChecks.inner storage Bool.false src dst sm sn) s =
      some (innerState M storage s a c v1 v2 (2 * sn).toNat) := by
  cases storage
  all_goals simp only [SourceChecks.dtype] at h1 h2
  all_goals simp only [SourceChecks.inner, SourceChecks.middle, SourceChecks.outer,
    kernel, ComputeKernel.surfaceBody, List.getElem?_cons_zero, List.getElem?_cons_succ,
    run, step, evalExpr, evalComputeOp]
  all_goals simp [ComputeDType.eraseDType, Option.bind, evalOp_unfold, FP.Structural.store, numeric, SourceChecks.dtype,
    ha, hc, h1, h2, scatter, innerState, loaded, bump]
  all_goals unfold FP.Structural.bop
  all_goals simp [ComputeDType.eraseDType, Option.bind, Broadcast.leftIndex, Broadcast.rightIndex, ha, hc, h1, h2,
    scatter, innerState, loaded, bump, SourceChecks.dtype]
  all_goals rfl

end VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.TypedExecution

#guard_msgs (drop info) in
#auditModuleAxioms
