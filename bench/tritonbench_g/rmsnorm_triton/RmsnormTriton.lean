import VeriTile.Triton

/-!
# `rmsnorm_triton` — strict per-kernel correctness

`rmsnorm_triton` is a 3D `(batch, M, K)` RMS normalization over the last
dimension: programs `(pid_batch, pid_m)` accumulate `var += pow(x, 2)` over a
`BLOCK_N_SIZE`-tiled loop across `N_SIZE` columns (masked by `offs_n < N_SIZE`,
masked lanes read as `0`), reduce to `var = sum/N_SIZE`, take
`rstd = rsqrt(var + eps)`, then in a second loop store `(x * rstd) * rms_w` back,
masked by `offs_n < N_SIZE`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`rmsnorm_triton[(batch, M)](...)`, the 2D
program-per-(batch,row) grid, the host `BLOCK_N_SIZE=1024` choice, and how the
runtime composes per-row writes into one buffer) is the *trusted boundary*, not
a proof obligation here. Because `(pid_batch, pid_m)` are universally
quantified, the per-program statement covers every program of the grid.

## Proof architecture

The verified result is the FULL multi-block correctness:

```
Full.Final2.rmsnorm_full_output_summary       ← FULL multi-block TOP THEOREM
  ├─ (toAlgorithm? = Except.ok _)             surface lowers to the algorithm layer
  └─ Full.Final2.rmsnorm_full_compute_correct ← ComputeCorrect over the masked store
       └─ Full.Final2.rmsnorm_full_correct    ← per global lane k < N_SIZE readback,
                                                 GENERAL ⌈N_SIZE/BLOCK_N_SIZE⌉-iteration loops
```

The **full** development (namespace `Full`) verifies both tiled loops over
arbitrarily many `BLOCK_N_SIZE`-blocks via loop invariants composed through
`forRange_inv`: a variance-accumulation invariant (`varInv`: after `c` blocks the
`var` register tile holds the elementwise partial sum-of-squares) and a writeback
invariant (`wbInv`: after `c` blocks every global lane `k < c·BLOCK_N_SIZE` of the
output holds the closed form, later blocks don't clobber earlier ones — needs
global output-offset injectivity). The combinatorial heart
(`sum_sq_mean`: the block×lane double sum equals `Σ_{k<N_SIZE} x[k]²`) closes the
mean. Final claim: for every `k < N_SIZE`,
`out[k] = x[k] · rsqrt((Σ_{k'<N} x[k']²)/N_SIZE + eps) · w[k]` (`rmsSpecFull`).

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); `@triton.autotune` is not
modeled. The `.to(tl.float32)` casts on the loaded `x` reduce to the identity at
the algorithm layer (post-erasure all dtypes unify to `ℝ`);
`tl.extra.cuda.libdevice.pow(·, 2)` is modeled as `x*x` and `tl.math.rsqrt` as
exact `WithBot.realRsqrt`. The **full** theorem holds for arbitrary `N_SIZE` and
`BLOCK_N_SIZE` (`0 < BLOCK_N_SIZE`, `0 < N_SIZE`), with the tiled loops running
`⌈N_SIZE/BLOCK_N_SIZE⌉` times; masked lanes load `0` (matching `other=0.0`). It
assumes the output region is distinct from the input regions (`o ≠ x`, `o ≠ w`)
and that the output column stride is nonzero (`0 < stride_out_k`, the one-glance
well-formedness bound that yields per-lane output-offset injectivity via
`outOff_injective_of_pos`). The wrapper launch shape is the special case
`N_SIZE ≤ BLOCK_N_SIZE` (here `N_SIZE = K`, `BLOCK_N_SIZE = 1024`).
-/

namespace VeriTile.Bench.TritonBenchG.RmsnormTriton

open VeriTile.Triton

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

/-- Faithful transcription of `rmsnorm_triton.py`'s `rmsnorm_triton`.

Allowed mechanical Lean-syntax-only changes:
- Python `N_SIZE: tl.constexpr` / `eps: tl.constexpr` / `BLOCK_N_SIZE: tl.constexpr`
  -> Lean `Nat` / `ℝ` parameters. -/
def rmsnorm_triton
    (x_ptr rms_w_ptr output_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k : Nat)
    (N_SIZE BLOCK_N_SIZE : Nat) (eps : ℝ) :
    ComputeKernel := triton {
  pid_batch = tl.program_id(0)
  pid_m = tl.program_id(1)
  offs_m = pid_batch * $(stride_x_batch) + pid_m * $(stride_x_m)
  block_N = tl.arange(0, $(BLOCK_N_SIZE))
  var = tl.zeros([$(BLOCK_N_SIZE)], tl.float32)
  for block_n_start_idx in range(0, $(N_SIZE), $(BLOCK_N_SIZE)) {
    offs_n = block_n_start_idx + block_N
    x_ptr_mask = offs_n < $(N_SIZE)
    x = tl.load(x_ptr + offs_m + offs_n * $(stride_x_k), mask=x_ptr_mask, other=0.0)
    var += tl.extra.cuda.libdevice.pow((x).to(tl.float32), 2)
  }
  var = tl.sum(var, axis=0) / $(N_SIZE)
  rstd = tl.math.rsqrt(var + $(eps))
  for block_n_start_idx in range(0, $(N_SIZE), $(BLOCK_N_SIZE)) {
    offs_n = block_n_start_idx + block_N
    x_ptr_mask = offs_n < $(N_SIZE)
    rms_w = tl.load(rms_w_ptr + offs_n * $(stride_rms_w), mask=x_ptr_mask)
    x = tl.load(x_ptr + offs_m + offs_n * $(stride_x_k), mask=x_ptr_mask, other=0.0).to(tl.float32)
    x_hat = x * rstd
    out = x_hat * rms_w
    out_off = pid_batch * $(stride_out_batch) + pid_m * $(stride_out_m) +
      offs_n * $(stride_out_k)
    tl.store(output_ptr + out_off, out, mask=x_ptr_mask)
  }
}

/-! ## Full multi-block correctness (general `N_SIZE`, no `N_SIZE ≤ BLOCK` hypothesis)

The development below proves the FULL general-loop correctness of `rmsnorm_triton`:
both the variance-accumulation loop and the masked-store writeback loop are
verified for arbitrarily many `BLOCK_N_SIZE`-tiled iterations (`⌈N_SIZE/BLOCK⌉`),
via loop invariants composed through `forRange_inv`. The capstone
`Full.Final2.rmsnorm_full_correct` shows that for every global lane `k < N_SIZE`,
the output equals `x[k] · rsqrt((Σ_{k'<N} x[k']²)/N + eps) · w[k]` — the genuine
RMS-norm closed form over all `N_SIZE` columns. -/
namespace Full

open Finset
set_option maxHeartbeats 2000000
set_option linter.unusedVariables false

namespace ScratchRms

def xOff (s : BlockState) (sxb sxm sxk : Nat) (k : Nat) : Nat :=
  s.pids 0 * sxb + s.pids 1 * sxm + k * sxk

def varBody (x : RegionName) (sxk N B : Nat) : List Stmt :=
  [ Stmt.assign .nat [B] "offs_n"
      (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "block_n_start_idx") (Op.ref .nat [B] "block_N")),
    Stmt.assign .bool [B] "x_ptr_mask"
      (Op.lt .nat Broadcast.scalarR (Op.ref .nat [B] "offs_n") (Op.constNat N)),
    Stmt.assign .real [B] "x"
      (Op.load .real
        (MemAccess.region x
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "offs_m")
            (Op.mul .nat Broadcast.scalarR (Op.ref .nat [B] "offs_n") (Op.constNat sxk))))
        (MaskOpt.maskOther (Op.ref .bool [B] "x_ptr_mask") (Op.broadcast (Op.const 0.0) [B]))),
    Stmt.assign .real [B] "var"
      (Op.add .real (Broadcast.consSame Broadcast.nil) (Op.ref .real [B] "var")
        (Op.mul .real (Broadcast.consSame Broadcast.nil)
          (Op.castFloat .real .real (Op.ref .real [B] "x"))
          (Op.castFloat .real .real (Op.ref .real [B] "x")))) ]

noncomputable def maskedSq (s0 : BlockState) (x : RegionName) (sxb sxm sxk N : Nat) (k : Nat) : ℝ :=
  if k < N then (s0.readMem x (xOff s0 sxb sxm sxk k))^2 else 0

noncomputable def varAcc (s0 : BlockState) (x : RegionName) (sxb sxm sxk N B : Nat)
    (c : Nat) (j : Fin B) : ℝ :=
  ∑ b : Fin c, maskedSq s0 x sxb sxm sxk N (b.val*B+j.val)

theorem offsn_eval (B c : Nat) (s : BlockState)
    (hblockN : s.regs .nat [B] "block_N" = some (Tile.vec (fun j : Fin B => j.val))) :
    evalOp (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "block_n_start_idx") (Op.ref .nat [B] "block_N"))
        (s.setReg "block_n_start_idx" .nat [] (Tile.scalar (c*B)))
      = some (Tile.vec (fun j : Fin B => c*B + j.val)) := by
  rw [evalOp_add, evalOp_ref_setReg_same,
    evalOp_ref_setReg_ne_name _ _ _ _ _ _ _ _ (show ("block_N":RegName) ≠ "block_n_start_idx" by decide),
    evalOp_ref, hblockN]
  apply congrArg some
  ext j
  simp only [Tile.bop_data, Broadcast.leftIndex_scalarL, Broadcast.rightIndex_scalarL,
    Tile.scalar, Tile.vec, NumericDType.add]

theorem mask_eval (N B : Nat) (s : BlockState) (c : Nat)
    (hoffsn : s.regs .nat [B] "offs_n" = some (Tile.vec (fun j : Fin B => c*B + j.val))) :
    evalOp (Op.lt .nat Broadcast.scalarR (Op.ref .nat [B] "offs_n") (Op.constNat N)) s
      = some (Tile.vec (fun j : Fin B => decide (c*B + j.val < N))) := by
  rw [evalOp_lt, evalOp_ref, hoffsn, evalOp_constNat]
  apply congrArg some
  ext j
  simp only [Tile.cop_data, Tile.vec_data, Tile.scalar_data, Broadcast.leftIndex_scalarR,
    Broadcast.rightIndex_scalarR, ComparableDType.lt, decide_eq_decide]

theorem xload_eval (x : RegionName) (sxb sxm sxk N B : Nat) (s0 s : BlockState) (c : Nat)
    (hpids : s.pids = s0.pids) (hrm : s.readMem x = s0.readMem x)
    (hoffsn : s.regs .nat [B] "offs_n" = some (Tile.vec (fun j : Fin B => c*B + j.val)))
    (hmask : s.regs .bool [B] "x_ptr_mask" = some (Tile.vec (fun j : Fin B => decide (c*B + j.val < N))))
    (hoffsm : s.regs .nat [] "offs_m" = some (Tile.scalar (s0.pids 0 * sxb + s0.pids 1 * sxm))) :
    evalOp (Op.load .real
        (MemAccess.region x
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "offs_m")
            (Op.mul .nat Broadcast.scalarR (Op.ref .nat [B] "offs_n") (Op.constNat sxk))))
        (MaskOpt.maskOther (Op.ref .bool [B] "x_ptr_mask") (Op.broadcast (Op.const 0.0) [B]))) s
      = some ⟨fun j : TileIndex [B] => some (if c*B + j.1.val < N then s0.readMem x (xOff s0 sxb sxm sxk (c*B+j.1.val)) else 0)⟩ := by
  simp only [evalOp, hoffsm, hoffsn, hmask]
  apply congrArg some
  ext j
  simp only [Tile.bop_data, Broadcast.rightIndex_scalarL,
    Broadcast.rightIndex_scalarR, Tile.scalar, Tile.vec,
    NumericDType.mul, NumericDType.add, BlockState.readMemValue_real, xOff, Region.cast_id, hrm]
  by_cases h : c*B + j.1.val < N
  · simp [h]
  · norm_num [h]

theorem varacc_eval (B : Nat) (s : BlockState) (g h : Fin B → ℝ)
    (hvar : s.regs .real [B] "var" = some ⟨fun j : TileIndex [B] => some (g j.1)⟩)
    (hx : s.regs .real [B] "x" = some ⟨fun j : TileIndex [B] => some (h j.1)⟩) :
    evalOp (Op.add .real (Broadcast.consSame Broadcast.nil) (Op.ref .real [B] "var")
        (Op.mul .real (Broadcast.consSame Broadcast.nil)
          (Op.castFloat .real .real (Op.ref .real [B] "x"))
          (Op.castFloat .real .real (Op.ref .real [B] "x")))) s
      = some ⟨fun j : TileIndex [B] => some (g j.1 + h j.1 * h j.1)⟩ := by
  have hxc : evalOp (Op.castFloat .real .real (Op.ref .real [B] "x")) s
      = some (⟨fun j : TileIndex [B] => some (h j.1)⟩ : Tile .real [B]) := by
    rw [evalOp_castFloat]; simp only [FloatDType.toTileDType_real]
    rw [evalOp_ref, hx]; apply congrArg some; ext j; simp [FloatDType.cast]
  have hxc2 : @evalOp TileDType.real [B] (Op.castFloat .real .real (Op.ref .real [B] "x")) s
      = some (⟨fun j : TileIndex [B] => some (h j.1)⟩ : Tile .real [B]) := hxc
  rw [evalOp_add, evalOp_ref, hvar, evalOp_mul, hxc2]
  apply congrArg some
  ext j
  simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.add, NumericDType.mul, WithBot.realAdd, WithBot.realMul, Option.map₂, Option.bind, Option.map]

-- recurrence
theorem varAcc_succ (s0 : BlockState) (x : RegionName) (sxb sxm sxk N B : Nat) (c : Nat) (j : Fin B) :
    varAcc s0 x sxb sxm sxk N B (c+1) j = varAcc s0 x sxb sxm sxk N B c j + maskedSq s0 x sxb sxm sxk N (c*B+j.val) := by
  unfold varAcc
  rw [Fin.sum_univ_castSucc]
  simp [Fin.last]

-- THE VAR STEP
theorem varStep (x : RegionName) (sxb sxm sxk N B : Nat) (s0 s : BlockState) (c : Nat)
    (hpids : s.pids = s0.pids) (hrm : s.readMem = s0.readMem)
    (hblockN : s.regs .nat [B] "block_N" = some (Tile.vec (fun j : Fin B => j.val)))
    (hoffsm : s.regs .nat [] "offs_m" = some (Tile.scalar (s0.pids 0 * sxb + s0.pids 1 * sxm)))
    (hvar : s.regs .real [B] "var" = some ⟨fun idx : TileIndex [B] => some (varAcc s0 x sxb sxm sxk N B c idx.1)⟩) :
    ∃ s', stepStmts (varBody x sxk N B)
        (s.setReg "block_n_start_idx" .nat [] (Tile.scalar (c*B))) = some s'
      ∧ s'.pids = s0.pids ∧ s'.readMem = s0.readMem
      ∧ s'.regs .nat [B] "block_N" = some (Tile.vec (fun j : Fin B => j.val))
      ∧ s'.regs .nat [] "offs_m" = some (Tile.scalar (s0.pids 0 * sxb + s0.pids 1 * sxm))
      ∧ s'.regs .real [B] "var" = some ⟨fun idx : TileIndex [B] => some (varAcc s0 x sxb sxm sxk N B (c+1) idx.1)⟩
      ∧ (∀ (dt : TileDType) (sh : TileShape) (R : RegName), R ≠ "offs_n" → R ≠ "x_ptr_mask" → R ≠ "x" → R ≠ "var" → R ≠ "block_n_start_idx" →
          s'.regs dt sh R = s.regs dt sh R) := by
  unfold varBody
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (offsn_eval B c s hblockN))]
  set s1 := (s.setReg "block_n_start_idx" .nat [] (Tile.scalar (c*B))).setReg "offs_n" .nat [B] (Tile.vec (fun j : Fin B => c*B + j.val)) with hs1
  have hoffsn1 : s1.regs .nat [B] "offs_n" = some (Tile.vec (fun j : Fin B => c*B + j.val)) := by
    simp [hs1]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (mask_eval N B s1 c hoffsn1))]
  set s2 := s1.setReg "x_ptr_mask" .bool [B] (Tile.vec (fun j : Fin B => decide (c*B + j.val < N))) with hs2
  have hpids2 : s2.pids = s0.pids := by rw [hs2, hs1]; simp [hpids]
  have hrm2 : s2.readMem = s0.readMem := by funext rg o; rw [hs2, hs1]; simp [hrm]
  have hoffsn2 : s2.regs .nat [B] "offs_n" = some (Tile.vec (fun j : Fin B => c*B + j.val)) := by
    rw [hs2]; simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offs_n":RegName) ≠ "x_ptr_mask" by decide)]; exact hoffsn1
  have hmask2 : s2.regs .bool [B] "x_ptr_mask" = some (Tile.vec (fun j : Fin B => decide (c*B + j.val < N))) := by
    rw [hs2]; simp
  have hoffsm2 : s2.regs .nat [] "offs_m" = some (Tile.scalar (s0.pids 0 * sxb + s0.pids 1 * sxm)) := by
    rw [hs2, hs1]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offs_m":RegName) ≠ "x_ptr_mask" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offs_m":RegName) ≠ "offs_n" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offs_m":RegName) ≠ "block_n_start_idx" by decide)]
    exact hoffsm
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (xload_eval x sxb sxm sxk N B s0 s2 c hpids2 (congrFun hrm2 x) hoffsn2 hmask2 hoffsm2))]
  set s3 := s2.setReg "x" .real [B] ⟨fun j : TileIndex [B] => some (if c*B + j.1.val < N then s0.readMem x (xOff s0 sxb sxm sxk (c*B+j.1.val)) else 0)⟩ with hs3
  have hvar3 : s3.regs .real [B] "var" = some ⟨fun idx : TileIndex [B] => some (varAcc s0 x sxb sxm sxk N B c idx.1)⟩ := by
    rw [hs3, hs2, hs1]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("var":RegName) ≠ "x" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("var":RegName) ≠ "x_ptr_mask" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("var":RegName) ≠ "offs_n" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("var":RegName) ≠ "block_n_start_idx" by decide)]
    exact hvar
  have hx3 : s3.regs .real [B] "x" = some ⟨fun j : TileIndex [B] => some ((if c*B + j.1.val < N then s0.readMem x (xOff s0 sxb sxm sxk (c*B+j.1.val)) else 0))⟩ := by
    rw [hs3]; simp
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (varacc_eval B s3 (fun j => varAcc s0 x sxb sxm sxk N B c j)
      (fun j => if c*B + j.val < N then s0.readMem x (xOff s0 sxb sxm sxk (c*B+j.val)) else 0) hvar3 hx3))]
  rw [stepStmts.nil]
  set s4 := s3.setReg "var" .real [B] ⟨fun j : TileIndex [B] => some (varAcc s0 x sxb sxm sxk N B c j.1 + (if c*B + j.1.val < N then s0.readMem x (xOff s0 sxb sxm sxk (c*B+j.1.val)) else 0) * (if c*B + j.1.val < N then s0.readMem x (xOff s0 sxb sxm sxk (c*B+j.1.val)) else 0))⟩ with hs4
  refine ⟨s4, rfl, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hs4, hs3]; simp [hpids2]
  · funext rg o; rw [hs4, hs3]; simp [hrm2]
  · rw [hs4]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("block_N":RegName) ≠ "var" by decide)]
    rw [hs3, hs2, hs1]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("block_N":RegName) ≠ "x" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("block_N":RegName) ≠ "x_ptr_mask" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("block_N":RegName) ≠ "offs_n" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("block_N":RegName) ≠ "block_n_start_idx" by decide)]
    exact hblockN
  · rw [hs4]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offs_m":RegName) ≠ "var" by decide)]
    exact hoffsm2
  · rw [hs4]; simp only [BlockState.setReg_same]
    apply congrArg some; ext idx
    show some _ = some _
    congr 1
    rw [varAcc_succ]
    congr 1
    unfold maskedSq
    by_cases h : c*B + idx.1.val < N
    · simp only [h, if_true]; ring
    · simp only [h, if_false]; ring
  · intro dt sh R h1 h2 h3 h4 h5
    rw [hs4, hs3, hs2, hs1]
    rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h4,
        BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h3,
        BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h2,
        BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h1,
        BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h5]

end ScratchRms

namespace Wb
open ScratchRms

def wbBody (x w o : RegionName) (sxb sxm sxk srw sob som sok N B : Nat) : List Stmt :=
  [ Stmt.assign .nat [B] "offs_n"
      (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "block_n_start_idx") (Op.ref .nat [B] "block_N")),
    Stmt.assign .bool [B] "x_ptr_mask"
      (Op.lt .nat Broadcast.scalarR (Op.ref .nat [B] "offs_n") (Op.constNat N)),
    Stmt.assign .real [B] "rms_w"
      (Op.load .real
        (MemAccess.region w
          (Op.mul .nat Broadcast.scalarR (Op.ref .nat [B] "offs_n") (Op.constNat srw)))
        (MaskOpt.mask (Op.ref .bool [B] "x_ptr_mask"))),
    Stmt.assign .real [B] "x"
      (Op.load .real
        (MemAccess.region x
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "offs_m")
            (Op.mul .nat Broadcast.scalarR (Op.ref .nat [B] "offs_n") (Op.constNat sxk))))
        (MaskOpt.maskOther (Op.ref .bool [B] "x_ptr_mask") (Op.broadcast (Op.const 0.0) [B]))),
    Stmt.assign .real [B] "x_hat"
      (Op.mul .real Broadcast.scalarR (Op.ref .real [B] "x") (Op.ref .real [] "rstd")),
    Stmt.assign .real [B] "out"
      (Op.mul .real (Broadcast.consSame Broadcast.nil) (Op.ref .real [B] "x_hat") (Op.ref .real [B] "rms_w")),
    Stmt.assign .nat [B] "out_off"
      (Op.add .nat Broadcast.scalarL
        (Op.add .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_batch") (Op.constNat sob))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_m") (Op.constNat som)))
        (Op.mul .nat Broadcast.scalarR (Op.ref .nat [B] "offs_n") (Op.constNat sok))),
    Stmt.store .real [B] (MemAccess.region o (Op.ref .nat [B] "out_off"))
      (Op.ref .real [B] "out") (MaskOpt.mask (Op.ref .bool [B] "x_ptr_mask")) ]

end Wb

namespace Wb

def outOff (s : BlockState) (sob som sok : Nat) (k : Nat) : Nat :=
  s.pids 0 * sob + s.pids 1 * som + k * sok

end Wb

namespace VarLoop
open ScratchRms

-- var-loop invariant
noncomputable def varInv (x : RegionName) (sxb sxm sxk N B : Nat) (s0 : BlockState)
    (i : Nat) (s : BlockState) : Prop :=
  s.pids = s0.pids ∧ s.readMem = s0.readMem ∧ (B ∣ i) ∧
  s.regs .nat [] "pid_batch" = some (Tile.scalar (s0.pids 0)) ∧
  s.regs .nat [] "pid_m" = some (Tile.scalar (s0.pids 1)) ∧
  s.regs .nat [B] "block_N" = some (Tile.vec (fun j : Fin B => j.val)) ∧
  s.regs .nat [] "offs_m" = some (Tile.scalar (s0.pids 0 * sxb + s0.pids 1 * sxm)) ∧
  s.regs .real [B] "var" = some ⟨fun idx : TileIndex [B] => some (varAcc s0 x sxb sxm sxk N B (i/B) idx.1)⟩

-- step preserves invariant (the forRange_inv h_step form)
theorem varInv_step (x : RegionName) (sxb sxm sxk N B : Nat) (s0 : BlockState) (hB : 0 < B)
    (i : Nat) (s : BlockState) (hlt : i < N) (hinv : varInv x sxb sxm sxk N B s0 i s) :
    ∃ s', stepStmts (varBody x sxk N B) (s.setReg "block_n_start_idx" .nat [] (Tile.scalar i)) = some s'
      ∧ varInv x sxb sxm sxk N B s0 (i+B) s' := by
  obtain ⟨hpids, hrm, ⟨c, hc⟩, hpb, hpm, hblockN, hoffsm, hvar⟩ := hinv
  subst hc
  have hcB : (B*c)/B = c := by rw [Nat.mul_div_cancel_left _ hB]
  rw [hcB] at hvar
  rw [show B*c = c*B from Nat.mul_comm B c]
  obtain ⟨s', hstep, hp', hr', hbN', hom', hv', hpres⟩ :=
    varStep x sxb sxm sxk N B s0 s c hpids hrm hblockN hoffsm hvar
  refine ⟨s', hstep, hp', hr', ⟨c+1, by ring⟩, ?_, ?_, hbN', hom', ?_⟩
  · rw [hpres _ _ _ (by decide) (by decide) (by decide) (by decide) (by decide)]; exact hpb
  · rw [hpres _ _ _ (by decide) (by decide) (by decide) (by decide) (by decide)]; exact hpm
  · rwa [show (c*B+B)/B = c+1 by rw [show c*B+B = (c+1)*B by ring, Nat.mul_div_cancel _ hB]]

end VarLoop

namespace VarFinal
open ScratchRms VarLoop

-- the forRange var loop, starting from a state with varInv ... 0, terminates with varInv ... final, N ≤ final
theorem var_forRange (x : RegionName) (sxb sxm sxk N B : Nat) (s0 : BlockState) (hB : 0 < B)
    (s : BlockState) (hinit : varInv x sxb sxm sxk N B s0 0 s) :
    ∃ final s', stepStmt (.forRange "block_n_start_idx" 0 N B (varBody x sxk N B)) s = some s'
      ∧ N ≤ final ∧ varInv x sxb sxm sxk N B s0 final s' := by
  exact forRange_inv (idx := "block_n_start_idx") (start := 0) (stop := N) (step := B)
    (Nat.pos_iff_ne_zero.mp hB) hinit
    (fun i st hlt hP => varInv_step x sxb sxm sxk N B s0 hB i st hlt hP)

end VarFinal

namespace MathHeart
open Finset

theorem sum_blocks_lanes (BLOCK c : Nat) (hBLOCK : 0 < BLOCK) (H : Nat → ℝ) :
    (∑ b : Fin c, ∑ j : Fin BLOCK, H (b.val * BLOCK + j.val)) = ∑ k : Fin (c * BLOCK), H k.val := by
  rw [← Fintype.sum_prod_type']
  apply Finset.sum_nbij' (i := fun p : Fin c × Fin BLOCK => (⟨p.1.val * BLOCK + p.2.val, by
        have h1 := p.1.isLt; have h2 := p.2.isLt
        calc p.1.val * BLOCK + p.2.val < p.1.val * BLOCK + BLOCK := by omega
          _ = (p.1.val + 1) * BLOCK := by ring
          _ ≤ c * BLOCK := Nat.mul_le_mul_right _ (by omega)⟩ : Fin (c*BLOCK)))
    (j := fun k : Fin (c*BLOCK) => (⟨k.val / BLOCK,
        (Nat.div_lt_iff_lt_mul hBLOCK).mpr (by have := k.isLt; omega)⟩, ⟨k.val % BLOCK, Nat.mod_lt _ hBLOCK⟩))
  · intro p _; simp
  · intro k _; simp
  · intro p _
    apply Prod.ext
    · apply Fin.ext; show (p.1.val * BLOCK + p.2.val) / BLOCK = p.1.val
      rw [Nat.mul_comm p.1.val BLOCK, Nat.mul_add_div hBLOCK, Nat.div_eq_of_lt p.2.isLt, Nat.add_zero]
    · apply Fin.ext; show (p.1.val * BLOCK + p.2.val) % BLOCK = p.2.val
      rw [Nat.mul_comm p.1.val BLOCK, Nat.mul_add_mod, Nat.mod_eq_of_lt p.2.isLt]
  · intro k _; apply Fin.ext; show (k.val / BLOCK) * BLOCK + k.val % BLOCK = k.val
    rw [Nat.mul_comm (k.val / BLOCK) BLOCK]; exact Nat.div_add_mod k.val BLOCK
  · intro p _; rfl

theorem sum_fin_extend (N M : Nat) (hNM : N ≤ M) (f : Nat → ℝ) :
    (∑ k : Fin M, (if k.val < N then f k.val else 0)) = ∑ k : Fin N, f k.val := by
  rw [Fin.sum_univ_eq_sum_range (fun k => if k < N then f k else 0) M,
      Fin.sum_univ_eq_sum_range (fun k => f k) N]
  rw [← Finset.sum_subset (s₁ := range N) (s₂ := range M)
        (fun x hx => Finset.mem_range.mpr (lt_of_lt_of_le (Finset.mem_range.mp hx) hNM))
        (by intro k hk hknotN; simp only [Finset.mem_range] at hk hknotN; simp [Nat.not_lt.mp hknotN])]
  apply Finset.sum_congr rfl
  intro k hk; simp only [Finset.mem_range] at hk; simp [hk]

theorem sum_sq_mean (BLOCK c N : Nat) (hBLOCK : 0 < BLOCK) (hge : N ≤ c * BLOCK) (f : Nat → ℝ) :
    (∑ j : Fin BLOCK, ∑ b : Fin c, (if (b.val * BLOCK + j.val) < N then f (b.val * BLOCK + j.val) else 0))
      = ∑ k : Fin N, f k.val := by
  rw [Finset.sum_comm]
  rw [sum_blocks_lanes BLOCK c hBLOCK (fun m => if m < N then f m else 0)]
  exact sum_fin_extend N (c*BLOCK) hge f

end MathHeart

namespace Mean
open ScratchRms VarLoop MathHeart Finset

theorem var_sum (s0 : BlockState) (x : RegionName) (sxb sxm sxk N B c : Nat) (s : BlockState)
    (hvar : s.regs .real [B] "var" = some ⟨fun idx : TileIndex [B] => some (varAcc s0 x sxb sxm sxk N B c idx.1)⟩) :
    evalOp (Op.reduceSum (⟨0, by simp⟩ : Fin [B].length) Bool.false (Op.ref .real [B] "var")) s
      = some (Tile.scalar (some (∑ j : Fin B, varAcc s0 x sxb sxm sxk N B c j))) := by
  rw [evalOp_reduceSum, evalOp_ref, hvar]
  apply congrArg some
  ext idx
  rw [Tile.reduceSum_false, Tile.reduceSumDrop_data]
  have : (Tile.scalar (some (∑ j : Fin B, varAcc s0 x sxb sxm sxk N B c j)) : Tile .real []).data idx
      = some (∑ j : Fin B, varAcc s0 x sxb sxm sxk N B c j) := rfl
  rw [this, ← withBot_sum_some]
  rfl

-- the sum of varAcc over lanes = sum of squares over k<N, when c*B >= N
theorem varAcc_sum_eq (s0 : BlockState) (x : RegionName) (sxb sxm sxk N B c : Nat)
    (hB : 0 < B) (hge : N ≤ c * B) :
    (∑ j : Fin B, varAcc s0 x sxb sxm sxk N B c j)
      = ∑ k : Fin N, (s0.readMem x (xOff s0 sxb sxm sxk k.val))^2 := by
  have := sum_sq_mean B c N hB hge (fun k => (s0.readMem x (xOff s0 sxb sxm sxk k))^2)
  rw [← this]
  apply Finset.sum_congr rfl
  intro j _
  unfold varAcc maskedSq
  rfl

end Mean

namespace WbBody
open ScratchRms Wb Mean Finset

-- eval helpers for writeback
theorem rmsw_eval (w : RegionName) (srw N B : Nat) (s : BlockState) (c : Nat)
    (hoffsn : s.regs .nat [B] "offs_n" = some (Tile.vec (fun j : Fin B => c*B + j.val)))
    (hmask : s.regs .bool [B] "x_ptr_mask" = some (Tile.vec (fun j : Fin B => decide (c*B + j.val < N)))) :
    evalOp (Op.load .real
        (MemAccess.region w
          (Op.mul .nat Broadcast.scalarR (Op.ref .nat [B] "offs_n") (Op.constNat srw)))
        (MaskOpt.mask (Op.ref .bool [B] "x_ptr_mask"))) s
      = some ⟨fun j : TileIndex [B] => some (if c*B + j.1.val < N then s.readMem w ((c*B+j.1.val)*srw) else s.undef w ((c*B+j.1.val)*srw))⟩ := by
  simp only [evalOp, hoffsn, hmask]
  apply congrArg some
  ext j
  simp only [Tile.bop_data, Tile.vec, Tile.scalar, BlockState.readMemValue_real, NumericDType.mul]
  by_cases h : c*B + j.1.val < N
  · simp [h]
  · simp [h]

theorem xhat_eval (B : Nat) (s : BlockState) (h : Fin B → ℝ) (rstd : ℝ)
    (hx : s.regs .real [B] "x" = some ⟨fun j : TileIndex [B] => some (h j.1)⟩)
    (hrstd : s.regs .real [] "rstd" = some (Tile.scalar (some rstd))) :
    evalOp (Op.mul .real Broadcast.scalarR (Op.ref .real [B] "x") (Op.ref .real [] "rstd")) s
      = some ⟨fun j : TileIndex [B] => some (h j.1 * rstd)⟩ := by
  rw [evalOp_mul, evalOp_ref, hx, evalOp_ref, hrstd]
  apply congrArg some
  ext j
  simp only [Tile.bop_data, Tile.vec, Tile.scalar_data, Broadcast.leftIndex_scalarR,
    Broadcast.rightIndex_scalarR, NumericDType.mul, WithBot.realMul, Option.map₂, Option.bind, Option.map]

theorem out_eval (B : Nat) (s : BlockState) (g h : Fin B → ℝ)
    (hxh : s.regs .real [B] "x_hat" = some ⟨fun j : TileIndex [B] => some (g j.1)⟩)
    (hrw : s.regs .real [B] "rms_w" = some ⟨fun j : TileIndex [B] => some (h j.1)⟩) :
    evalOp (Op.mul .real (Broadcast.consSame Broadcast.nil) (Op.ref .real [B] "x_hat") (Op.ref .real [B] "rms_w")) s
      = some ⟨fun j : TileIndex [B] => some (g j.1 * h j.1)⟩ := by
  rw [evalOp_mul, evalOp_ref, hxh, evalOp_ref, hrw]
  apply congrArg some
  ext j
  simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.mul,
    WithBot.realMul, Option.map₂, Option.bind, Option.map]

theorem outoff_eval (sob som sok B : Nat) (s : BlockState) (c : Nat)
    (hpb : s.regs .nat [] "pid_batch" = some (Tile.scalar (s.pids 0)))
    (hpm : s.regs .nat [] "pid_m" = some (Tile.scalar (s.pids 1)))
    (hoffsn : s.regs .nat [B] "offs_n" = some (Tile.vec (fun j : Fin B => c*B + j.val))) :
    evalOp (Op.add .nat Broadcast.scalarL
        (Op.add .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_batch") (Op.constNat sob))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_m") (Op.constNat som)))
        (Op.mul .nat Broadcast.scalarR (Op.ref .nat [B] "offs_n") (Op.constNat sok))) s
      = some (Tile.vec (fun j : Fin B => outOff s sob som sok (c*B+j.val))) := by
  rw [evalOp_add, evalOp_add, evalOp_mul, evalOp_ref, hpb, evalOp_constNat, evalOp_mul, evalOp_ref, hpm,
    evalOp_constNat, evalOp_mul, evalOp_ref, hoffsn, evalOp_constNat]
  apply congrArg some
  ext j
  simp only [Tile.bop_data, Tile.vec, Tile.scalar, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.add, NumericDType.mul, outOff]

end WbBody

namespace StoreLemma
open ScratchRms Wb

theorem store_step (o : RegionName) (B : Nat) (s : BlockState)
    (offs : Fin B → Nat) (vals : Fin B → ℝ) (mask : Fin B → Bool)
    (hoff : s.regs .nat [B] "out_off" = some (Tile.vec offs))
    (hout : s.regs .real [B] "out" = some ⟨fun j : TileIndex [B] => some (vals j.1)⟩)
    (hmask : s.regs .bool [B] "x_ptr_mask" = some (Tile.vec mask)) :
    stepStmt (Stmt.store .real [B] (MemAccess.region o (Op.ref .nat [B] "out_off"))
      (Op.ref .real [B] "out") (MaskOpt.mask (Op.ref .bool [B] "x_ptr_mask"))) s
      = some ((TileShape.allIndices [B]).foldl
          (fun acc (j : TileIndex [B]) =>
            if mask j.1 then acc.writeMem o (offs j.1) (vals j.1) else acc) s) := by
  simp only [stepStmt, evalOp_ref, hoff, hout, hmask, Option.bind_eq_bind, Option.bind_some,
    Option.map, Tile.vec, Tile.scalar, Region.cast, BlockState.writeMemTyped_real,
    FloatDType.real_storeValue, WithBot.unbotD_some]


end StoreLemma

namespace WbStep
open ScratchRms Wb WbBody Mean Finset

-- the store value at lane j (active) = spec, computed from x and rms_w loads
-- out[j] = (readMem x (xOff (c*B+j)) * RS) * readMem w ((c*B+j)*srw)

theorem wbStep (x w o : RegionName) (sxb sxm sxk srw sob som sok N B : Nat) (s0 s : BlockState)
    (c : Nat) (RS : ℝ)
    (hinj : Function.Injective (fun k : Fin N => outOff s0 sob som sok k.val))
    (hox : o ≠ x) (how : o ≠ w)
    (hpids : s.pids = s0.pids) (hrmx : s.readMem x = s0.readMem x) (hrmw : s.readMem w = s0.readMem w)
    (hpb : s.regs .nat [] "pid_batch" = some (Tile.scalar (s0.pids 0)))
    (hpm : s.regs .nat [] "pid_m" = some (Tile.scalar (s0.pids 1)))
    (hblockN : s.regs .nat [B] "block_N" = some (Tile.vec (fun j : Fin B => j.val)))
    (hoffsm : s.regs .nat [] "offs_m" = some (Tile.scalar (s0.pids 0 * sxb + s0.pids 1 * sxm)))
    (hrstd : s.regs .real [] "rstd" = some (Tile.scalar (some RS))) :
    ∃ s', stepStmts (wbBody x w o sxb sxm sxk srw sob som sok N B)
        (s.setReg "block_n_start_idx" .nat [] (Tile.scalar (c*B))) = some s'
      ∧ s'.pids = s0.pids
      ∧ s'.regs .nat [] "pid_batch" = some (Tile.scalar (s0.pids 0))
      ∧ s'.regs .nat [] "pid_m" = some (Tile.scalar (s0.pids 1))
      ∧ s'.regs .nat [B] "block_N" = some (Tile.vec (fun j : Fin B => j.val))
      ∧ s'.regs .nat [] "offs_m" = some (Tile.scalar (s0.pids 0 * sxb + s0.pids 1 * sxm))
      ∧ s'.regs .real [] "rstd" = some (Tile.scalar (some RS))
      ∧ s'.readMem x = s.readMem x ∧ s'.readMem w = s.readMem w
      ∧ ∀ k : Fin N, s'.readMem o (outOff s0 sob som sok k.val)
          = if (c*B ≤ k.val ∧ k.val < c*B+B) then
              s0.readMem x (xOff s0 sxb sxm sxk k.val) * RS * s0.readMem w (k.val*srw)
            else s.readMem o (outOff s0 sob som sok k.val) := by
  unfold wbBody
  -- offs_n
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (offsn_eval B c s hblockN))]
  set s1 := (s.setReg "block_n_start_idx" .nat [] (Tile.scalar (c*B))).setReg "offs_n" .nat [B] (Tile.vec (fun j : Fin B => c*B + j.val)) with hs1
  have hoffsn1 : s1.regs .nat [B] "offs_n" = some (Tile.vec (fun j : Fin B => c*B + j.val)) := by simp [hs1]
  -- x_ptr_mask
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (mask_eval N B s1 c hoffsn1))]
  set s2 := s1.setReg "x_ptr_mask" .bool [B] (Tile.vec (fun j : Fin B => decide (c*B + j.val < N))) with hs2
  have hoffsn2 : s2.regs .nat [B] "offs_n" = some (Tile.vec (fun j : Fin B => c*B + j.val)) := by
    rw [hs2]; simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offs_n":RegName) ≠ "x_ptr_mask" by decide)]; exact hoffsn1
  have hmask2 : s2.regs .bool [B] "x_ptr_mask" = some (Tile.vec (fun j : Fin B => decide (c*B + j.val < N))) := by rw [hs2]; simp
  -- rms_w (per-region w)
  have hrm2w : s2.readMem w = s0.readMem w := by funext ofs; rw [hs2, hs1]; simp only [BlockState.setReg_readMem]; exact congrFun hrmw ofs
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (rmsw_eval w srw N B s2 c hoffsn2 hmask2))]
  set s3 := s2.setReg "rms_w" .real [B] ⟨fun j : TileIndex [B] => some (if c*B + j.1.val < N then s2.readMem w ((c*B+j.1.val)*srw) else s2.undef w ((c*B+j.1.val)*srw))⟩ with hs3
  -- x load
  have hpids3 : s3.pids = s0.pids := by rw [hs3, hs2, hs1]; simp [hpids]
  have hrm3x : s3.readMem x = s0.readMem x := by funext ofs; rw [hs3, hs2, hs1]; simp only [BlockState.setReg_readMem]; exact congrFun hrmx ofs
  have hoffsn3 : s3.regs .nat [B] "offs_n" = some (Tile.vec (fun j : Fin B => c*B + j.val)) := by
    rw [hs3]; simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offs_n":RegName) ≠ "rms_w" by decide)]; exact hoffsn2
  have hmask3 : s3.regs .bool [B] "x_ptr_mask" = some (Tile.vec (fun j : Fin B => decide (c*B + j.val < N))) := by
    rw [hs3]; simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("x_ptr_mask":RegName) ≠ "rms_w" by decide)]; exact hmask2
  have hoffsm3 : s3.regs .nat [] "offs_m" = some (Tile.scalar (s0.pids 0 * sxb + s0.pids 1 * sxm)) := by
    rw [hs3, hs2, hs1]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offs_m":RegName) ≠ "rms_w" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offs_m":RegName) ≠ "x_ptr_mask" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offs_m":RegName) ≠ "offs_n" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offs_m":RegName) ≠ "block_n_start_idx" by decide)]
    exact hoffsm
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (xload_eval x sxb sxm sxk N B s0 s3 c hpids3 hrm3x hoffsn3 hmask3 hoffsm3))]
  set s4 := s3.setReg "x" .real [B] ⟨fun j : TileIndex [B] => some (if c*B + j.1.val < N then s0.readMem x (xOff s0 sxb sxm sxk (c*B+j.1.val)) else 0)⟩ with hs4
  -- x_hat
  have hx4 : s4.regs .real [B] "x" = some ⟨fun j : TileIndex [B] => some (if c*B + j.1.val < N then s0.readMem x (xOff s0 sxb sxm sxk (c*B+j.1.val)) else 0)⟩ := by
    rw [hs4]; exact BlockState.setReg_same _ _ _ _ _
  have hrstd4 : s4.regs .real [] "rstd" = some (Tile.scalar (some RS)) := by
    rw [hs4, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("rstd":RegName) ≠ "x" by decide), hs3,
        BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("rstd":RegName) ≠ "rms_w" by decide), hs2,
        BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("rstd":RegName) ≠ "x_ptr_mask" by decide), hs1,
        BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("rstd":RegName) ≠ "offs_n" by decide),
        BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("rstd":RegName) ≠ "block_n_start_idx" by decide)]
    exact hrstd
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (xhat_eval B s4 (fun j => if c*B + j.val < N then s0.readMem x (xOff s0 sxb sxm sxk (c*B+j.val)) else 0) RS hx4 hrstd4))]
  set s5 := s4.setReg "x_hat" .real [B] ⟨fun j : TileIndex [B] => some ((if c*B + j.1.val < N then s0.readMem x (xOff s0 sxb sxm sxk (c*B+j.1.val)) else 0) * RS)⟩ with hs5
  -- out
  have hxh5 : s5.regs .real [B] "x_hat" = some ⟨fun j : TileIndex [B] => some ((if c*B + j.1.val < N then s0.readMem x (xOff s0 sxb sxm sxk (c*B+j.1.val)) else 0) * RS)⟩ := by
    rw [hs5]; exact BlockState.setReg_same _ _ _ _ _
  have hrw5 : s5.regs .real [B] "rms_w" = some ⟨fun j : TileIndex [B] => some (if c*B + j.1.val < N then s2.readMem w ((c*B+j.1.val)*srw) else s2.undef w ((c*B+j.1.val)*srw))⟩ := by
    rw [hs5, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("rms_w":RegName) ≠ "x_hat" by decide), hs4,
        BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("rms_w":RegName) ≠ "x" by decide), hs3]
    exact BlockState.setReg_same _ _ _ _ _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (out_eval B s5 (fun j => (if c*B + j.val < N then s0.readMem x (xOff s0 sxb sxm sxk (c*B+j.val)) else 0) * RS)
      (fun j => if c*B + j.val < N then s2.readMem w ((c*B+j.val)*srw) else s2.undef w ((c*B+j.val)*srw)) hxh5 hrw5))]
  set s6 := s5.setReg "out" .real [B] ⟨fun j : TileIndex [B] => some (((if c*B + j.1.val < N then s0.readMem x (xOff s0 sxb sxm sxk (c*B+j.1.val)) else 0) * RS) * (if c*B + j.1.val < N then s2.readMem w ((c*B+j.1.val)*srw) else s2.undef w ((c*B+j.1.val)*srw)))⟩ with hs6
  -- out_off
  have hpids6 : s6.pids = s0.pids := by rw [hs6, hs5, hs4]; simp [hpids3]
  have hpb6 : s6.regs .nat [] "pid_batch" = some (Tile.scalar (s6.pids 0)) := by
    rw [hpids6, hs6, hs5, hs4, hs3, hs2, hs1]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("pid_batch":RegName) ≠ "out" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("pid_batch":RegName) ≠ "x_hat" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("pid_batch":RegName) ≠ "x" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("pid_batch":RegName) ≠ "rms_w" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("pid_batch":RegName) ≠ "x_ptr_mask" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("pid_batch":RegName) ≠ "offs_n" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("pid_batch":RegName) ≠ "block_n_start_idx" by decide)]
    exact hpb
  have hpm6 : s6.regs .nat [] "pid_m" = some (Tile.scalar (s6.pids 1)) := by
    rw [hpids6, hs6, hs5, hs4, hs3, hs2, hs1]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("pid_m":RegName) ≠ "out" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("pid_m":RegName) ≠ "x_hat" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("pid_m":RegName) ≠ "x" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("pid_m":RegName) ≠ "rms_w" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("pid_m":RegName) ≠ "x_ptr_mask" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("pid_m":RegName) ≠ "offs_n" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("pid_m":RegName) ≠ "block_n_start_idx" by decide)]
    exact hpm
  have hoffsn6 : s6.regs .nat [B] "offs_n" = some (Tile.vec (fun j : Fin B => c*B + j.val)) := by
    rw [hs6, hs5, hs4]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offs_n":RegName) ≠ "out" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offs_n":RegName) ≠ "x_hat" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offs_n":RegName) ≠ "x" by decide)]
    exact hoffsn3
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (outoff_eval sob som sok B s6 c hpb6 hpm6 hoffsn6))]
  set s7 := s6.setReg "out_off" .nat [B] (Tile.vec (fun j : Fin B => outOff s6 sob som sok (c*B+j.val))) with hs7
  -- store
  have houtoff7 : s7.regs .nat [B] "out_off" = some (Tile.vec (fun j : Fin B => outOff s0 sob som sok (c*B+j.val))) := by
    rw [hs7]; rw [BlockState.setReg_same]
    apply congrArg some; apply Tile.ext; intro j; unfold outOff; rw [hpids6]
  have hout7 : s7.regs .real [B] "out" = some ⟨fun j : TileIndex [B] => some (((if c*B + j.1.val < N then s0.readMem x (xOff s0 sxb sxm sxk (c*B+j.1.val)) else 0) * RS) * (if c*B + j.1.val < N then s2.readMem w ((c*B+j.1.val)*srw) else s2.undef w ((c*B+j.1.val)*srw)))⟩ := by
    rw [hs7, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("out":RegName) ≠ "out_off" by decide), hs6]
    exact BlockState.setReg_same _ _ _ _ _
  have hmask7 : s7.regs .bool [B] "x_ptr_mask" = some (Tile.vec (fun j : Fin B => decide (c*B + j.val < N))) := by
    rw [hs7, hs6, hs5, hs4]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("x_ptr_mask":RegName) ≠ "out_off" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("x_ptr_mask":RegName) ≠ "out" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("x_ptr_mask":RegName) ≠ "x_hat" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("x_ptr_mask":RegName) ≠ "x" by decide)]
    exact hmask3
  rw [stepStmts.cons_some (StoreLemma.store_step o B s7
    (fun j : Fin B => outOff s0 sob som sok (c*B+j.val))
    (fun j : Fin B => ((if c*B + j.val < N then s0.readMem x (xOff s0 sxb sxm sxk (c*B+j.val)) else 0) * RS) * (if c*B + j.val < N then s2.readMem w ((c*B+j.val)*srw) else s2.undef w ((c*B+j.val)*srw)))
    (fun j : Fin B => decide (c*B + j.val < N)) houtoff7 hout7 hmask7)]
  rw [stepStmts.nil]
  set sF := (TileShape.allIndices [B]).foldl
        (fun acc (j : TileIndex [B]) =>
          if decide (c*B + j.1.val < N) then acc.writeMem o (outOff s0 sob som sok (c*B+j.1.val))
            (((if c*B + j.1.val < N then s0.readMem x (xOff s0 sxb sxm sxk (c*B+j.1.val)) else 0) * RS) * (if c*B + j.1.val < N then s2.readMem w ((c*B+j.1.val)*srw) else s2.undef w ((c*B+j.1.val)*srw)))
          else acc) s7 with hsF
  have hs7pids : s7.pids = s0.pids := by rw [hs7]; simp only [BlockState.setReg_pids]; exact hpids6
  have hsFpids : sF.pids = s0.pids := by rw [hsF, foldl_store_pids]; exact hs7pids
  -- register facts at s7
  have hpb7 : s7.regs .nat [] "pid_batch" = some (Tile.scalar (s0.pids 0)) := by
    rw [hs7, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("pid_batch":RegName) ≠ "out_off" by decide)]
    rw [hpb6, hpids6]
  have hpm7 : s7.regs .nat [] "pid_m" = some (Tile.scalar (s0.pids 1)) := by
    rw [hs7, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("pid_m":RegName) ≠ "out_off" by decide)]
    rw [hpm6, hpids6]
  have hbN7 : s7.regs .nat [B] "block_N" = some (Tile.vec (fun j : Fin B => j.val)) := by
    rw [hs7, hs6, hs5, hs4, hs3, hs2, hs1]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("block_N":RegName) ≠ "out_off" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("block_N":RegName) ≠ "out" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("block_N":RegName) ≠ "x_hat" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("block_N":RegName) ≠ "x" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("block_N":RegName) ≠ "rms_w" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("block_N":RegName) ≠ "x_ptr_mask" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("block_N":RegName) ≠ "offs_n" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("block_N":RegName) ≠ "block_n_start_idx" by decide)]
    exact hblockN
  have hom7 : s7.regs .nat [] "offs_m" = some (Tile.scalar (s0.pids 0 * sxb + s0.pids 1 * sxm)) := by
    rw [hs7, hs6, hs5, hs4]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offs_m":RegName) ≠ "out_off" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offs_m":RegName) ≠ "out" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offs_m":RegName) ≠ "x_hat" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offs_m":RegName) ≠ "x" by decide)]
    exact hoffsm3
  have hrs7 : s7.regs .real [] "rstd" = some (Tile.scalar (some RS)) := by
    rw [hs7, hs6, hs5, hs4]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("rstd":RegName) ≠ "out_off" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("rstd":RegName) ≠ "out" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("rstd":RegName) ≠ "x_hat" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("rstd":RegName) ≠ "x" by decide)]
    exact hrstd4
  -- s7.readMem r = s.readMem r for any region
  have hs7rmAll : ∀ (rg : RegionName) (ofs : Nat), s7.readMem rg ofs = s.readMem rg ofs := by
    intro rg ofs; rw [hs7, hs6, hs5, hs4, hs3, hs2, hs1]; simp only [BlockState.setReg_readMem]
  refine ⟨sF, rfl, hsFpids, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hsF, foldl_store_regs]; exact hpb7
  · rw [hsF, foldl_store_regs]; exact hpm7
  · rw [hsF, foldl_store_regs]; exact hbN7
  · rw [hsF, foldl_store_regs]; exact hom7
  · rw [hsF, foldl_store_regs]; exact hrs7
  · funext ofs; rw [hsF, foldl_store_other_region _ _ _ _ _ _ _ (Ne.symm hox), hs7rmAll]
  · funext ofs; rw [hsF, foldl_store_other_region _ _ _ _ _ _ _ (Ne.symm how), hs7rmAll]
  · -- the readback
    intro k
    rw [hsF]
    have hk : k.val < N := k.isLt
    by_cases hin : (c*B ≤ k.val ∧ k.val < c*B+B)
    · -- k in this block: foldl_store_at at j_k
      rw [if_pos hin]
      have hjk : c*B + (k.val - c*B) = k.val := by omega
      set jk : Fin B := ⟨k.val - c*B, by omega⟩ with hjk_def
      have hofeq : outOff s0 sob som sok (c*B + jk.val) = outOff s0 sob som sok k.val := by
        simp only [hjk_def]; rw [hjk]
      rw [foldl_store_at
            (fun j : TileIndex [B] => outOff s0 sob som sok (c*B+j.1.val))
            (fun j : TileIndex [B] => ((if c*B + j.1.val < N then s0.readMem x (xOff s0 sxb sxm sxk (c*B+j.1.val)) else 0) * RS) * (if c*B + j.1.val < N then s2.readMem w ((c*B+j.1.val)*srw) else s2.undef w ((c*B+j.1.val)*srw)))
            (fun j : TileIndex [B] => decide (c*B + j.1.val < N))
            (outOff s0 sob som sok k.val) (TileShape.allIndices [B]) s7
            (jk, PUnit.unit) (TileShape.mem_allIndices [B] _)
            (by simp only [hjk_def]; rw [hjk]; exact decide_eq_true hk)
            hofeq
            (by -- uniqueness
              intro b hb hmb hofb
              -- offsets equal and both active (decide true => c*B+b<N) => use hinj on Fin N
              have hbN : c*B + b.1.val < N := by simpa using hmb
              have heq : (⟨c*B + b.1.val, hbN⟩ : Fin N) = k := hinj (by simpa using hofb)
              have : c*B + b.1.val = k.val := by rw [← heq]
              apply Prod.ext
              · apply Fin.ext; simp only [hjk_def]; omega
              · rfl)
            (TileShape.allIndices_nodup [B])]
      -- value at jk = spec; convert s2.readMem -> s0.readMem and the ifs collapse
      simp only [hjk_def]
      rw [hjk]
      rw [if_pos hk, if_pos hk, hrm2w]
    · -- k not in this block: preserved
      rw [if_neg hin]
      rw [foldl_store_preserve
            (fun j : TileIndex [B] => outOff s0 sob som sok (c*B+j.1.val))
            (fun j : TileIndex [B] => ((if c*B + j.1.val < N then s0.readMem x (xOff s0 sxb sxm sxk (c*B+j.1.val)) else 0) * RS) * (if c*B + j.1.val < N then s2.readMem w ((c*B+j.1.val)*srw) else s2.undef w ((c*B+j.1.val)*srw)))
            (fun j : TileIndex [B] => decide (c*B + j.1.val < N))
            (outOff s0 sob som sok k.val) (TileShape.allIndices [B]) s7
            (by
              intro b hb hmb hofb
              have hbN : c*B + b.1.val < N := by simpa using hmb
              have heq : (⟨c*B + b.1.val, hbN⟩ : Fin N) = k := hinj (by simpa using hofb)
              have : c*B + b.1.val = k.val := by rw [← heq]
              omega)]
      -- s7.readMem = s.readMem
      have hs7rm : s7.readMem = s.readMem := by
        funext rg ofs
        rw [hs7, hs6, hs5, hs4, hs3, hs2, hs1]
        simp only [BlockState.setReg_readMem]
      rw [hs7rm]


end WbStep

namespace WbLoop
open ScratchRms Wb WbStep

noncomputable def wbInv (x w o : RegionName) (sxb sxm sxk srw sob som sok N B : Nat)
    (s0 : BlockState) (RS : ℝ) (sorig : BlockState) (i : Nat) (s : BlockState) : Prop :=
  s.pids = s0.pids ∧ (B ∣ i) ∧
  s.regs .nat [] "pid_batch" = some (Tile.scalar (s0.pids 0)) ∧
  s.regs .nat [] "pid_m" = some (Tile.scalar (s0.pids 1)) ∧
  s.regs .nat [B] "block_N" = some (Tile.vec (fun j : Fin B => j.val)) ∧
  s.regs .nat [] "offs_m" = some (Tile.scalar (s0.pids 0 * sxb + s0.pids 1 * sxm)) ∧
  s.regs .real [] "rstd" = some (Tile.scalar (some RS)) ∧
  s.readMem x = s0.readMem x ∧ s.readMem w = s0.readMem w ∧
  (∀ k : Fin N, s.readMem o (outOff s0 sob som sok k.val)
    = if k.val < i then
        s0.readMem x (xOff s0 sxb sxm sxk k.val) * RS * s0.readMem w (k.val*srw)
      else sorig.readMem o (outOff s0 sob som sok k.val))

theorem wbInv_step (x w o : RegionName) (sxb sxm sxk srw sob som sok N B : Nat)
    (s0 : BlockState) (RS : ℝ) (sorig : BlockState) (hB : 0 < B)
    (hinj : Function.Injective (fun k : Fin N => outOff s0 sob som sok k.val))
    (hox : o ≠ x) (how : o ≠ w)
    (i : Nat) (s : BlockState) (hlt : i < N)
    (hinv : wbInv x w o sxb sxm sxk srw sob som sok N B s0 RS sorig i s) :
    ∃ s', stepStmts (wbBody x w o sxb sxm sxk srw sob som sok N B)
        (s.setReg "block_n_start_idx" .nat [] (Tile.scalar i)) = some s'
      ∧ wbInv x w o sxb sxm sxk srw sob som sok N B s0 RS sorig (i+B) s' := by
  obtain ⟨hpids, ⟨c, hc⟩, hpb, hpm, hbN, hom, hrs, hrx, hrw, hmem⟩ := hinv
  subst hc
  rw [show B*c = c*B from Nat.mul_comm B c]
  obtain ⟨s', hstep, hp', hpb', hpm', hbN', hom', hrs', hrx', hrw', hread'⟩ :=
    wbStep x w o sxb sxm sxk srw sob som sok N B s0 s c RS hinj hox how hpids hrx hrw hpb hpm hbN hom hrs
  refine ⟨s', hstep, hp', ⟨c+1, by ring⟩, hpb', hpm', hbN', hom', hrs', ?_, ?_, ?_⟩
  · rw [hrx']; exact hrx
  · rw [hrw']; exact hrw
  · -- the combined readback
    intro k
    rw [hread' k]
    by_cases hin : (c*B ≤ k.val ∧ k.val < c*B+B)
    · rw [if_pos hin, if_pos (by omega : k.val < c*B + B)]
    · rw [if_neg hin]
      rw [hmem k]
      have hcomm : B*c = c*B := Nat.mul_comm B c
      by_cases hlt2 : k.val < B*c
      · rw [if_pos hlt2, if_pos (by omega : k.val < c*B+B)]
      · rw [if_neg hlt2, if_neg (by omega : ¬ k.val < c*B+B)]

end WbLoop

namespace WbFinal
open ScratchRms Wb WbLoop

theorem wb_forRange (x w o : RegionName) (sxb sxm sxk srw sob som sok N B : Nat)
    (s0 : BlockState) (RS : ℝ) (sorig : BlockState) (hB : 0 < B)
    (hinj : Function.Injective (fun k : Fin N => outOff s0 sob som sok k.val))
    (hox : o ≠ x) (how : o ≠ w)
    (s : BlockState) (hinit : wbInv x w o sxb sxm sxk srw sob som sok N B s0 RS sorig 0 s) :
    ∃ final s', stepStmt (.forRange "block_n_start_idx" 0 N B (wbBody x w o sxb sxm sxk srw sob som sok N B)) s = some s'
      ∧ N ≤ final ∧ wbInv x w o sxb sxm sxk srw sob som sok N B s0 RS sorig final s' := by
  exact forRange_inv (idx := "block_n_start_idx") (start := 0) (stop := N) (step := B)
    (Nat.pos_iff_ne_zero.mp hB) hinit
    (fun i st hlt hP => wbInv_step x w o sxb sxm sxk srw sob som sok N B s0 RS sorig hB hinj hox how i st hlt hP)

end WbFinal

namespace Postfix
open ScratchRms Mean MathHeart VarLoop Finset

-- var = sum(var,0)/N  then  rstd = rsqrt(var+eps)
-- eval of the div assign
theorem vardiv_eval (s0 : BlockState) (x : RegionName) (sxb sxm sxk N B c : Nat) (s : BlockState)
    (hvar : s.regs .real [B] "var" = some ⟨fun idx : TileIndex [B] => some (varAcc s0 x sxb sxm sxk N B c idx.1)⟩) :
    evalOp (Op.div .real Broadcast.nil
        (Op.reduceSum (⟨0, by simp⟩ : Fin [B].length) Bool.false (Op.ref .real [B] "var"))
        (Op.const (N : ℝ))) s
      = some (Tile.scalar (some ((∑ j : Fin B, varAcc s0 x sxb sxm sxk N B c j) / (N : ℝ)))) := by
  have hs := var_sum s0 x sxb sxm sxk N B c s hvar
  simp only [evalOp_div, evalOp_const, Option.bind_eq_bind]
  erw [hs]
  simp only [Option.bind_some]
  apply congrArg some
  apply Tile.ext; intro idx
  rfl

theorem rsqrt_eval (RS : ℝ) (s : BlockState) (eps : ℝ)
    (hvar : s.regs .real [] "var" = some (Tile.scalar (some RS))) :
    evalOp (Op.rsqrt (Op.add .real Broadcast.nil (Op.ref .real [] "var") (Op.const eps))) s
      = some (Tile.scalar (WithBot.realRsqrt (some (RS + eps)))) := by
  rw [show (Op.rsqrt (Op.add .real Broadcast.nil (Op.ref .real [] "var") (Op.const eps)))
        = Op.rsqrt (Op.add .real Broadcast.nil (Op.ref .real [] "var") (Op.const eps)) from rfl]
  simp only [evalOp, hvar, Option.bind]
  apply congrArg some
  apply Tile.ext; intro idx
  simp only [Tile.uop_data, Tile.bop_data, Tile.scalar_data, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.add, WithBot.realAdd, Option.map₂, Option.bind, Option.map, evalOp_const, evalOp_ref]
  rfl

end Postfix

namespace Prefix
open ScratchRms VarLoop Finset

def prefixStmts (x : RegionName) (sxb sxm B : Nat) : List Stmt :=
  [ Stmt.assign .nat [] "pid_batch" (Op.programId 0),
    Stmt.assign .nat [] "pid_m" (Op.programId 1),
    Stmt.assign .nat [] "offs_m"
      (Op.add .nat Broadcast.nil
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_batch") (Op.constNat sxb))
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_m") (Op.constNat sxm))),
    Stmt.assign .nat [B] "block_N" (Op.arange B),
    Stmt.assign .real [B] "var" (Op.full [B] (Op.const (0:ℝ))) ]

theorem preLoop (x : RegionName) (sxb sxm sxk N B : Nat) (s : BlockState) :
    ∃ s', stepStmts (prefixStmts x sxb sxm B) s = some s'
      ∧ varInv x sxb sxm sxk N B s 0 s' := by
  unfold prefixStmts
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (by simp [evalOp_programId] : evalOp (Op.programId 0) s = some (Tile.scalar (s.pids 0))))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (by simp [evalOp_programId] : evalOp (Op.programId 1) _ = some (Tile.scalar (s.pids 1))))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (by
    rw [evalOp_add, evalOp_mul, evalOp_ref_setReg_ne_name _ _ _ _ _ _ _ _ (show ("pid_batch":RegName) ≠ "pid_m" by decide),
      evalOp_ref_setReg_same, evalOp_constNat, evalOp_mul, evalOp_ref_setReg_same, evalOp_constNat]
    rfl : evalOp (Op.add .nat Broadcast.nil
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_batch") (Op.constNat sxb))
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_m") (Op.constNat sxm))) _
      = some (Tile.scalar (s.pids 0 * sxb + s.pids 1 * sxm))))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (by simp [evalOp_arange] : evalOp (Op.arange B) _ = some (Tile.vec (fun j : Fin B => j.val))))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (by
    simp only [evalOp, Option.bind]
    rfl : evalOp (Op.full [B] (Op.const (0:ℝ))) _ = some ⟨fun _ : TileIndex [B] => some (0:ℝ)⟩))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ⟨0, rfl⟩, ?_, ?_, ?_, ?_, ?_⟩
  · simp
  · funext rg ofs; simp
  · simp [BlockState.setReg_ne_name]
  · simp [BlockState.setReg_ne_name]
  · simp [BlockState.setReg_ne_name]
  · simp [BlockState.setReg_ne_name]
  · -- var = zeros = varAcc 0 (empty sum)
    rw [BlockState.setReg_same]
    apply congrArg some; apply Tile.ext; intro idx
    show some (0:ℝ) = some (varAcc s x sxb sxm sxk N B (0/B) idx.1)
    rw [Nat.zero_div]
    unfold varAcc
    simp

end Prefix

namespace PostStep
open ScratchRms VarLoop Mean Postfix Finset

def postStmts (N B : Nat) (eps : ℝ) : List Stmt :=
  [ Stmt.assign .real [] "var"
      (Op.div .real Broadcast.nil
        (Op.reduceSum (⟨0, by simp⟩ : Fin [B].length) Bool.false (Op.ref .real [B] "var"))
        (Op.const (N:ℝ))),
    Stmt.assign .real [] "rstd"
      (Op.rsqrt (Op.add .real Broadcast.nil (Op.ref .real [] "var") (Op.const eps))) ]

theorem postStep (s0 : BlockState) (x : RegionName) (sxb sxm sxk N B cv : Nat) (eps : ℝ) (s : BlockState)
    (hvar : s.regs .real [B] "var" = some ⟨fun idx : TileIndex [B] => some (varAcc s0 x sxb sxm sxk N B cv idx.1)⟩) :
    ∃ s', stepStmts (postStmts N B eps) s = some s'
      ∧ s'.regs .real [] "rstd" = some (Tile.scalar (WithBot.realRsqrt (some ((∑ j : Fin B, varAcc s0 x sxb sxm sxk N B cv j) / (N:ℝ) + eps))))
      ∧ (∀ (dt : TileDType) (sh : TileShape) (R : RegName), R ≠ "var" → R ≠ "rstd" →
          s'.regs dt sh R = s.regs dt sh R)
      ∧ s'.pids = s.pids ∧ s'.readMem = s.readMem := by
  unfold postStmts
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (vardiv_eval s0 x sxb sxm sxk N B cv s hvar))]
  set s1 := s.setReg "var" .real [] (Tile.scalar (some ((∑ j : Fin B, varAcc s0 x sxb sxm sxk N B cv j) / (N:ℝ)))) with hs1
  have hv1 : s1.regs .real [] "var" = some (Tile.scalar (some ((∑ j : Fin B, varAcc s0 x sxb sxm sxk N B cv j) / (N:ℝ)))) := by
    rw [hs1]; exact BlockState.setReg_same _ _ _ _ _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (rsqrt_eval _ s1 eps hv1))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_⟩
  · rw [BlockState.setReg_same]
  · intro dt sh R h1 h2
    rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h2, hs1, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h1]
  · rw [hs1]; simp
  · funext rg ofs; rw [hs1]; simp

end PostStep

namespace Final
open ScratchRms VarLoop Mean MathHeart Postfix PostStep Prefix WbLoop WbFinal VarFinal Wb Finset



-- the toAlgKernel body decomposes into prefix ++ [varLoop] ++ post ++ [wbLoop]
theorem body_decomp (x w o : RegionName) (sxb sxm sxk srw sob som sok N B : Nat) (eps : ℝ) :
    (VeriTile.Bench.TritonBenchG.RmsnormTriton.rmsnorm_triton x w o sxb sxm sxk srw sob som sok N B eps).toAlgKernel.body
      = prefixStmts x sxb sxm B
        ++ [Stmt.forRange "block_n_start_idx" 0 N B (varBody x sxk N B)]
        ++ postStmts N B eps
        ++ [Stmt.forRange "block_n_start_idx" 0 N B (wbBody x w o sxb sxm sxk srw sob som sok N B)] := by
  rfl

end Final

namespace Final2
open ScratchRms VarLoop Mean MathHeart Postfix PostStep Prefix WbLoop WbFinal VarFinal Wb Final Finset

-- the genuine mean of squares over global lanes
noncomputable def meanSq (s : BlockState) (x : RegionName) (sxb sxm sxk N : Nat) : ℝ :=
  (∑ k : Fin N, (s.readMem x (xOff s sxb sxm sxk k.val))^2) / (N:ℝ)

noncomputable def rmsSpecFull (s : BlockState) (x w : RegionName) (sxb sxm sxk srw N : Nat) (eps : ℝ) (k : Nat) : ℝ :=
  s.readMem x (xOff s sxb sxm sxk k) * (WithBot.unbotD 0 (WithBot.realRsqrt (some (meanSq s x sxb sxm sxk N + eps)))) * s.readMem w (k*srw)

/-- **Output-offset injectivity from a one-glance well-formedness bound.**

`outOff k = pid_batch·sob + pid_m·som + k·sok` is injective over global lanes as
soon as the output column stride is nonzero (`0 < sok` ≡ `0 < stride_out_k`,
always true for a real tensor). The constant `(pid_batch, pid_m)` base cancels;
`k·sok = k'·sok` with `sok > 0` forces `k = k'`. Discharges the abstract
`Function.Injective` hypothesis from a single numeric fact. -/
theorem outOff_injective_of_pos (s : BlockState) (sob som sok N : Nat) (hsok : 0 < sok) :
    Function.Injective (fun k : Fin N => outOff s sob som sok k.val) := by
  have heq : (fun k : Fin N => outOff s sob som sok k.val)
      = (fun k : Fin N => (s.pids 0 * sob + s.pids 1 * som) + k.val * sok) := by
    funext k; simp only [outOff]
  rw [heq]
  exact affine1D_inj _ sok hsok

theorem rmsnorm_full_correct
    (x w o : RegionName) (sxb sxm sxk srw sob som sok N B : Nat) (eps : ℝ)
    (s s' : BlockState) (hB : 0 < B) (hNpos : 0 < N)
    (hox : o ≠ x) (how : o ≠ w)
    (hsok : 0 < sok)
    (hExec : exec (VeriTile.Bench.TritonBenchG.RmsnormTriton.rmsnorm_triton x w o sxb sxm sxk srw sob som sok N B eps) s = some s') :
    ∀ k : Fin N, s'.readMem o (outOff s sob som sok k.val)
      = rmsSpecFull s x w sxb sxm sxk srw N eps k.val := by
  have hinj : Function.Injective (fun k : Fin N => outOff s sob som sok k.val) :=
    outOff_injective_of_pos s sob som sok N hsok
  -- decompose exec
  rw [exec, body_decomp] at hExec
  simp only [List.append_assoc] at hExec
  -- now hExec : stepStmts (prefix ++ ([varLoop] ++ (post ++ [wbLoop]))) s = some s'
  -- prefix
  obtain ⟨sp, hsp, hinvP⟩ := preLoop x sxb sxm sxk N B s
  rw [stepStmts.append_some hsp] at hExec
  -- var loop
  obtain ⟨finalv, sv, hsv, hNlev, hinvV⟩ := var_forRange x sxb sxm sxk N B s hB sp hinvP
  have hsvL : stepStmts [Stmt.forRange "block_n_start_idx" 0 N B (varBody x sxk N B)] sp = some sv := by
    rw [stepStmts.cons_some hsv, stepStmts.nil]
  rw [stepStmts.append_some hsvL] at hExec
  obtain ⟨hpidsV, hrmV, ⟨cv, hcv⟩, hpbV, hpmV, hbNV, homV, hvarV⟩ := hinvV
  -- postfix
  obtain ⟨sr, hsr, hrstdR, hpresR, hpidsR, hrmR⟩ :=
    postStep s x sxb sxm sxk N B (finalv/B) eps sv hvarV
  rw [stepStmts.append_some hsr] at hExec
  -- mean = meanSq
  have hcvge : N ≤ (finalv/B) * B := by
    have hdvd : finalv/B * B = finalv := by
      have : B ∣ finalv := ⟨cv, by rw [hcv]⟩
      exact Nat.div_mul_cancel this
    omega
  have hmean : (∑ j : Fin B, varAcc s x sxb sxm sxk N B (finalv/B) j) / (N:ℝ) = meanSq s x sxb sxm sxk N := by
    unfold meanSq
    rw [varAcc_sum_eq s x sxb sxm sxk N B (finalv/B) hB hcvge]
  -- rstd value at sr
  have hRS : sr.regs .real [] "rstd" = some (Tile.scalar (WithBot.realRsqrt (some (meanSq s x sxb sxm sxk N + eps)))) := by
    rw [hrstdR]; rw [hmean]
  set RS : ℝ := WithBot.unbotD 0 (WithBot.realRsqrt (some (meanSq s x sxb sxm sxk N + eps))) with hRSdef
  have hRSsome : WithBot.realRsqrt (some (meanSq s x sxb sxm sxk N + eps)) = some RS := by
    rw [hRSdef]; simp [WithBot.realRsqrt]
  -- wbInv base at sr
  have hwbBase : wbInv x w o sxb sxm sxk srw sob som sok N B s RS s 0 sr := by
    refine ⟨?_, ⟨0, rfl⟩, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · rw [hpidsR, hpidsV]
    · rw [hpresR _ _ _ (by decide) (by decide)]; exact hpbV
    · rw [hpresR _ _ _ (by decide) (by decide)]; exact hpmV
    · rw [hpresR _ _ _ (by decide) (by decide)]; exact hbNV
    · rw [hpresR _ _ _ (by decide) (by decide)]; exact homV
    · rw [hRS]; rw [hRSsome]
    · funext ofs; rw [congrFun (congrFun hrmR x) ofs, congrFun (congrFun hrmV x) ofs]
    · funext ofs; rw [congrFun (congrFun hrmR w) ofs, congrFun (congrFun hrmV w) ofs]
    · intro k; rw [if_neg (by omega)]
      rw [congrFun (congrFun hrmR o) _, congrFun (congrFun hrmV o) _]
  -- wb loop
  obtain ⟨finalw, sw, hsw, hNlew, hinvW⟩ :=
    wb_forRange x w o sxb sxm sxk srw sob som sok N B s RS s hB hinj hox how sr hwbBase
  rw [stepStmts.cons_some hsw, stepStmts.nil] at hExec
  injection hExec with hExec
  subst hExec
  obtain ⟨_, _, _, _, _, _, _, _, _, hreadW⟩ := hinvW
  intro k
  rw [hreadW k, if_pos (by exact lt_of_lt_of_le k.isLt hNlew)]
  rfl

end Final2

namespace Final2

open ScratchRms Wb

/-- **General compute-facing correctness** for the full multi-block kernel: the
masked store to `output_ptr` is compute-correct over every global lane `k <
N_SIZE`, with no `N_SIZE ≤ BLOCK_N_SIZE` hypothesis. Each in-bounds lane holds
the genuine RMS-norm closed form `rmsSpecFull`. -/
theorem rmsnorm_full_compute_correct
    (x w o : RegionName) (sxb sxm sxk srw sob som sok N B : Nat) (eps : ℝ)
    (s : BlockState) (hB : 0 < B) (hNpos : 0 < N)
    (hox : o ≠ x) (how : o ≠ w)
    (hsok : 0 < sok) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := VeriTile.Bench.TritonBenchG.RmsnormTriton.rmsnorm_triton x w o
        sxb sxm sxk srw sob som sok N B eps)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin N => True)
        (fun k => (o, outOff s sob som sok k.val)))
      (expected := fun k : Fin N => rmsSpecFull s x w sxb sxm sxk srw N eps k.val) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [VeriTile.Bench.TritonBenchG.RmsnormTriton.rmsnorm_triton,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro k _
  exact rmsnorm_full_correct x w o sxb sxm sxk srw sob som sok N B eps s s' hB hNpos hox how hsok hExec k

/-- **Full output summary**: the surface lowers to the algorithm layer, and the
masked store realizes the genuine multi-block RMS-norm closed form at every
global lane `k < N_SIZE`. Holds for arbitrary `N_SIZE` — no
`N_SIZE ≤ BLOCK_N_SIZE` hypothesis. -/
specification rmsnorm_full_output_summary
    (x w o : RegionName) (sxb sxm sxk srw sob som sok N B : Nat) (eps : ℝ)
    (s : BlockState) (hB : 0 < B) (hNpos : 0 < N)
    (hox : o ≠ x) (how : o ≠ w)
    (hsok : 0 < sok) :
    (∃ alg, (VeriTile.Bench.TritonBenchG.RmsnormTriton.rmsnorm_triton x w o
        sxb sxm sxk srw sob som sok N B eps).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := VeriTile.Bench.TritonBenchG.RmsnormTriton.rmsnorm_triton x w o
        sxb sxm sxk srw sob som sok N B eps)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin N => True)
        (fun k => (o, outOff s sob som sok k.val)))
      (expected := fun k : Fin N => rmsSpecFull s x w sxb sxm sxk srw N eps k.val) := by
  refine ⟨?_, rmsnorm_full_compute_correct x w o sxb sxm sxk srw sob som sok N B eps s hB hNpos hox how hsok⟩
  simp [VeriTile.Bench.TritonBenchG.RmsnormTriton.rmsnorm_triton,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

end Final2

end Full

/-! ## The `⊨[R]` streaming headline (wave-5 S3 per-step emit genre)

Everything below is purely additive; the exact surface above is untouched.
This is the first consumer of the per-step emit skin
`StreamEmitMasked2DKernelIO₂` (streaming genre, style S3): the store sits
**inside** the second pass, so the output is a per-step `BLOCK_N_SIZE`-lane
window family rather than one terminal tile, and the kernel's spec `f t j`
is the genre's *two-pass* shape — the step-`t` tile combined with a fold
over the entire stream.

Structure of the `execR R` story: this kernel has **zero rounding events**.
Every load and store is at `.real`, and the only `castFloat`s are the
erased `.to(tl.float32)`s, i.e. `.real → .real` — exact under every `R` by
the model's defining `round_real` (`Rcast_real_real` below). Both passes
therefore collapse verbatim onto the exact stepper
(`stepForRangeAuxR_castFree`), and the whole proven
`preLoop` / `varInv` / `postStep` / `wbInv` invariant stack above is reused
unchanged; the `⊨[R]` face adds only the `TraceSafeR` walk, the per-cell
memory frame, and the stream-lane spec bridge. The skin's readback contract
at the default `outDType := .real` grid carries `R.round .real`, the
identity by `round_real` — the ∀-`R` face is the exact streaming contract
via the model's `.real` identity fields, not a `.triv` special case. -/

section IOFace

open Full Full.ScratchRms Full.Wb Full.VarLoop Full.VarFinal Full.Mean
  Full.MathHeart Full.PostStep Full.Prefix Full.WbLoop Full.WbFinal
  Full.WbBody Full.StoreLemma Full.Final Full.Final2 Finset
open scoped VeriTile.Triton.StreamEmitMasked2DKernelIO₂

set_option maxHeartbeats 4000000
set_option linter.unusedVariables false

/-! ### Stream geometry: trip count and windows -/

/-- Trip count of both `for block_n_start_idx in range(0, N_SIZE,
BLOCK_N_SIZE)` passes: `⌈N_SIZE / BLOCK_N_SIZE⌉`. -/
def rmsNumSteps (N B : Nat) : Nat := (N + B - 1) / B

private theorem rmsNumSteps_mul_ge (N B : Nat) (hB : 0 < B) :
    N ≤ rmsNumSteps N B * B := by
  rcases Nat.eq_zero_or_pos N with rfl | hN
  · exact Nat.zero_le _
  · unfold rmsNumSteps
    have heq : N + B - 1 = (N - 1) + B := by omega
    rw [heq, Nat.add_div_right _ hB]
    have h2 : (N - 1) % B + 1 ≤ B := Nat.mod_lt _ hB
    calc N = (N - 1) + 1 := by omega
      _ = (N - 1) / B * B + ((N - 1) % B + 1) := by
          rw [← Nat.add_assoc, Nat.div_add_mod']
      _ ≤ (N - 1) / B * B + B := Nat.add_le_add_left h2 _
      _ = ((N - 1) / B + 1) * B := (Nat.succ_mul _ _).symm

private theorem rmsStep_lt_numSteps (N B i : Nat) (hB : 0 < B) (hi : i < N) :
    i / B < rmsNumSteps N B := by
  have h2 : i / B * B < rmsNumSteps N B * B :=
    Nat.lt_of_le_of_lt (Nat.div_mul_le_self i B)
      (Nat.lt_of_lt_of_le hi (rmsNumSteps_mul_ge N B hB))
  exact Nat.lt_of_mul_lt_mul_right h2

/-! ### IO signature -/

/-- **Streaming IO signature** of `rmsnorm_triton` on the two-stream
per-step emit skin (S3: in-loop store). Step `t` of either pass (at
`block_n_start_idx = t·BLOCK_N_SIZE`) reads the `BLOCK_N_SIZE`-lane `x`
tile (`read1`, both passes read the same addresses) and the `rms_w` tile
(`read2`, pass 2 only); step `t` of pass 2 stores the `BLOCK_N_SIZE`-lane
output window (`write`) at the **`.real`** grid (`outDType` default — the
kernel's store is untyped `tl.store(output_ptr + out_off, out)` at `.real`,
so the per-step stores have no quantization event). The windows transcribe
the kernel's pointer arithmetic verbatim (`offs_n = t·BLOCK_N_SIZE + j`):

* `read1` step `t`, lane `j`:
  `pid₀·stride_x_batch + pid₁·stride_x_m + (t·BLOCK_N_SIZE + j)·stride_x_k`
  — the kernel's `x_ptr + offs_m + offs_n * stride_x_k`.
* `read2` step `t`, lane `j`: `(t·BLOCK_N_SIZE + j)·stride_rms_w` — the
  kernel's `rms_w_ptr + offs_n * stride_rms_w`.
* `write` step `t`, lane `j`:
  `pid₀·stride_out_batch + pid₁·stride_out_m + (t·BLOCK_N_SIZE + j)·stride_out_k`
  — the kernel's `out_off`.

All three masks are the kernel's single `x_ptr_mask`:
`t·BLOCK_N_SIZE + j < N_SIZE`. -/
def rmsnormKernelIO (x w o : RegionName)
    (sxb sxm sxk srw sob som sok N B : Nat) (eps : ℝ) :
    StreamEmitMasked2DKernelIO₂ where
  kernel := rmsnorm_triton x w o sxb sxm sxk srw sob som sok N B eps
  inp1 := x
  inp2 := w
  out := o
  T := rmsNumSteps N B
  B1 := B
  B2 := B
  C := B
  read1 := fun p₀ p₁ t j => p₀ * sxb + p₁ * sxm + (t.val * B + j.val) * sxk
  read2 := fun _ _ t j => (t.val * B + j.val) * srw
  write := fun p₀ p₁ t j => p₀ * sob + p₁ * som + (t.val * B + j.val) * sok
  mask1 := fun _ _ t j => t.val * B + j.val < N
  mask2 := fun _ _ t j => t.val * B + j.val < N
  writeMask := fun _ _ t j => t.val * B + j.val < N

/-! ### The stream-level spec -/

/-- The guarded stream-level sum of squares: the pass-1 fold
`var += pow(x, 2)` over the whole curried `x` stream, guarded by the
kernel's window (`t·B + e < N`) — the contract only pins `xs` on masked
lanes, so the spec must not read unmasked lanes. -/
noncomputable def rmsStreamSumSq (N B : Nat)
    (xs : Fin (rmsNumSteps N B) → Fin B → ℝ) : ℝ :=
  ∑ u : Fin (rmsNumSteps N B), ∑ e : Fin B,
    if u.val * B + e.val < N then xs u e ^ 2 else 0

/-- The stream-level RMS-norm spec (the genre's two-pass shape): output
window `(t, j)` holds the step-`t` `x` value times
`rsqrt(Σ x²/N_SIZE + eps)` — the fold over the *entire* stream — times the
step-`t` `rms_w` value. Algebraically `rmsSpecFull` with the `Fin N_SIZE`
sum re-blocked to the guarded stream double sum. -/
noncomputable def rmsStreamSpec (N B : Nat) (eps : ℝ)
    (xs ws : Fin (rmsNumSteps N B) → Fin B → ℝ)
    (t : Fin (rmsNumSteps N B)) (j : Fin B) : ℝ :=
  xs t j * (Real.sqrt (rmsStreamSumSq N B xs / (N : ℝ) + eps))⁻¹ * ws t j

/-! ### The stream-lane spec bridge -/

/-- Under the stream pin, the guarded stream double sum **is** the exact
stack's `Σ_{k<N} x[k]²` (re-blocking `k ↔ (k/B, k%B)` via the proven
`sum_sq_mean`). -/
private theorem rmsStreamSumSq_eq_sumSq (x : RegionName) (s₀ : BlockState)
    (sxb sxm sxk N B : Nat) (hB : 0 < B)
    (xs : Fin (rmsNumSteps N B) → Fin B → ℝ)
    (hx : ∀ (t : Fin (rmsNumSteps N B)) (e : Fin B), t.val * B + e.val < N →
      s₀.readMem x (s₀.pids 0 * sxb + s₀.pids 1 * sxm + (t.val * B + e.val) * sxk)
        = xs t e) :
    rmsStreamSumSq N B xs
      = ∑ k : Fin N, (s₀.readMem x (xOff s₀ sxb sxm sxk k.val)) ^ 2 := by
  unfold rmsStreamSumSq
  rw [Finset.sum_comm,
    ← sum_sq_mean B (rmsNumSteps N B) N hB (rmsNumSteps_mul_ge N B hB)
      (fun k => (s₀.readMem x (xOff s₀ sxb sxm sxk k)) ^ 2)]
  refine Finset.sum_congr rfl fun e _ => Finset.sum_congr rfl fun t _ => ?_
  by_cases h : t.val * B + e.val < N
  · rw [if_pos h, if_pos h, ← hx t e h]
    rfl
  · rw [if_neg h, if_neg h]

/-- Per-lane spec bridge: at a masked window `(t, j)` the stream spec **is**
the exact stack's `rmsSpecFull` at global lane `t·B + j` (rsqrt spelling:
`WithBot.realRsqrt (some v) = some (1/√v)`, and `1/√v = (√v)⁻¹`). -/
private theorem rmsStreamSpec_eq_rmsSpecFull (x w : RegionName)
    (s₀ : BlockState) (sxb sxm sxk srw N B : Nat) (eps : ℝ) (hB : 0 < B)
    (xs ws : Fin (rmsNumSteps N B) → Fin B → ℝ)
    (hx : ∀ (t : Fin (rmsNumSteps N B)) (e : Fin B), t.val * B + e.val < N →
      s₀.readMem x (s₀.pids 0 * sxb + s₀.pids 1 * sxm + (t.val * B + e.val) * sxk)
        = xs t e)
    (hw : ∀ (t : Fin (rmsNumSteps N B)) (e : Fin B), t.val * B + e.val < N →
      s₀.readMem w ((t.val * B + e.val) * srw) = ws t e)
    (t : Fin (rmsNumSteps N B)) (j : Fin B) (hj : t.val * B + j.val < N) :
    rmsStreamSpec N B eps xs ws t j
      = rmsSpecFull s₀ x w sxb sxm sxk srw N eps (t.val * B + j.val) := by
  unfold rmsStreamSpec rmsSpecFull meanSq
  rw [rmsStreamSumSq_eq_sumSq x s₀ sxb sxm sxk N B hB xs hx, ← hx t j hj,
    ← hw t j hj, WithBot.realRsqrt_some, WithBot.unbotD_some, one_div]
  rfl

/-! ### Cast-free collapses and the covered fragment -/

/-- The erased `.to(tl.float32)` is a `.real → .real` cast: exact under
every `R` — `R.roundW .real` is the identity by the model's defining
`round_real`. The kernel's only `castFloat`s are of this shape, so the
collapse lemmas below rewrite them to the exact cast. -/
private theorem Rcast_real_real (R : RoundingModel) :
    R.cast .real .real = FloatDType.cast .real .real := by
  funext v
  simp [RoundingModel.cast, FloatDType.cast]

/-- The prologue is cast-free: it steps identically under `stepStmtsR R`. -/
private theorem rmsPrefix_castFree (R : RoundingModel) (x : RegionName)
    (sxb sxm B : Nat) (t : BlockState) :
    stepStmtsR R (prefixStmts x sxb sxm B) t
      = stepStmts (prefixStmts x sxb sxm B) t := by
  simp only [prefixStmts, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def]
  rfl

/-- The pass-1 body is cast-free (masked `.real` load, `.real → .real`
casts, a real multiply-add): it steps identically under `stepStmtsR R`, so
the exact `varInv` stack transports to `execR`. -/
private theorem rmsVarBody_castFree (R : RoundingModel) (x : RegionName)
    (sxk N B : Nat) (t : BlockState) :
    stepStmtsR R (varBody x sxk N B) t = stepStmts (varBody x sxk N B) t := by
  simp only [varBody, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def, Rcast_real_real]
  rfl

/-- The reduce/rsqrt tail is cast-free. -/
private theorem rmsPost_castFree (R : RoundingModel) (N B : Nat) (eps : ℝ)
    (t : BlockState) :
    stepStmtsR R (postStmts N B eps) t = stepStmts (postStmts N B eps) t := by
  simp only [postStmts, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def]
  rfl

/-- The pass-2 body is cast-free **including its in-loop masked `.real`
store**: `stepStmtR` delegates a `.real`-typed store to the exact
`writeMemTyped` (`writeMemTypedR R .real` is definitionally the exact
write), so the whole storing loop steps identically under `stepStmtsR R`
and the exact `wbInv` stack transports to `execR`. -/
private theorem rmsWbBody_castFree (R : RoundingModel) (x w o : RegionName)
    (sxb sxm sxk srw sob som sok N B : Nat) (t : BlockState) :
    stepStmtsR R (wbBody x w o sxb sxm sxk srw sob som sok N B) t
      = stepStmts (wbBody x w o sxb sxm sxk srw sob som sok N B) t := by
  simp only [wbBody, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def, Rcast_real_real, BlockState.writeMemTypedR]
  rfl

/-- The full two-pass surface sits inside the flat-memory bridge's covered
fragment (`FlattenOk`; the two `forRange` clauses recurse into the
cast-free bodies). -/
theorem rmsnorm_triton_flattenOk (x w o : RegionName)
    (sxb sxm sxk srw sob som sok N B : Nat) (eps : ℝ) :
    ((rmsnorm_triton x w o sxb sxm sxk srw sob som sok N B eps).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  rw [body_decomp]
  simp [prefixStmts, varBody, postStmts, wbBody, StmtList.FlattenOk,
    Stmt.FlattenOk, Op.FlattenOk.eq_def]

/-! ### Cell-level memory frames

The exact stack proves per-lane `readMem` values; the `⊨[R]` Hoare triple
additionally needs a per-**cell** frame, so the register-only segments get
a generic assigns-don't-touch-memory lemma and the storing pass-2 body gets
a cell-level frame twin of `wbStep`. -/

/-- A run of `assign` statements never touches memory. -/
private theorem stepStmts_assigns_mem :
    ∀ (l : List Stmt),
      (∀ stmt ∈ l, ∃ dt sh nm, ∃ e : Op dt sh, stmt = Stmt.assign dt sh nm e) →
      ∀ {s s' : BlockState}, stepStmts l s = some s' → s'.mem = s.mem
  | [], _, s, s', h => by
      rw [stepStmts.nil] at h
      obtain rfl := Option.some.inj h
      rfl
  | stmt :: rest, hall, s, s', h => by
      obtain ⟨dt, sh, nm, e, rfl⟩ := hall _ List.mem_cons_self
      cases hv : evalOp e s with
      | none => simp [stepStmts, stepStmt, hv] at h
      | some v =>
          rw [stepStmts.cons_some (stepStmt_assign_eq_some hv)] at h
          rw [stepStmts_assigns_mem rest
            (fun st' hst' => hall st' (List.mem_cons_of_mem _ hst')) h]
          rfl

/-- Cell-level frame of a `Bool`-masked exact `writeMem` scatter `foldl`:
every cell not hit by an active lane is untouched (the cell-level sibling
of the shared `foldl_store_preserve`). -/
private theorem foldl_writeMem_bool_preserve_cell {α : Type}
    {region : RegionName} (ofn : α → Nat) (vfn : α → ℝ) (mask : α → Bool)
    (r : RegionName) (oo : Nat) (l : List α) (s : BlockState)
    (hnot : ∀ k ∈ l, mask k → ¬(r = region ∧ oo = ofn k)) :
    (l.foldl (fun acc k =>
        if mask k then acc.writeMem region (ofn k) (vfn k) else acc) s).mem r oo
      = s.mem r oo := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons]
      cases hm : mask hd
      · simp only [hm, Bool.false_eq_true, if_false]
        exact ih _ fun k hk hmk => hnot k (List.mem_cons_of_mem hd hk) hmk
      · simp only [hm, if_true]
        rw [ih _ fun k hk hmk => hnot k (List.mem_cons_of_mem hd hk) hmk,
          BlockState.writeMem_mem]
        exact if_neg (hnot hd List.mem_cons_self (by rw [hm]))

set_option maxHeartbeats 8000000 in
/-- **Cell-level frame of one pass-2 iteration** (the `mem` twin of
`wbStep`, same walk): from the `wbInv` register pins, one storing body
iteration leaves every cell off the `{(out, outOff k) : k < N_SIZE}`
write window untouched — the masked scatter store only hits active lanes
`i + j < N_SIZE`, whose offsets are `outOff (i + j)`. -/
private theorem rms_wbBody_step_frame (x w o : RegionName)
    (sxb sxm sxk srw sob som sok N B : Nat)
    (s0 st st' : BlockState) (RS : ℝ) (i : Nat) (hiB : B ∣ i)
    (hpids : st.pids = s0.pids)
    (hrmx : st.readMem x = s0.readMem x) (hrmw : st.readMem w = s0.readMem w)
    (hpb : st.regs .nat [] "pid_batch" = some (Tile.scalar (s0.pids 0)))
    (hpm : st.regs .nat [] "pid_m" = some (Tile.scalar (s0.pids 1)))
    (hblockN : st.regs .nat [B] "block_N" = some (Tile.vec (fun j : Fin B => j.val)))
    (hoffsm : st.regs .nat [] "offs_m" = some (Tile.scalar (s0.pids 0 * sxb + s0.pids 1 * sxm)))
    (hrstd : st.regs .real [] "rstd" = some (Tile.scalar (some RS)))
    (hstep : stepStmts (wbBody x w o sxb sxm sxk srw sob som sok N B)
        (st.setReg "block_n_start_idx" .nat [] (Tile.scalar i)) = some st')
    (r : RegionName) (oo : Nat)
    (hcond : r ≠ o ∨ ∀ k : Fin N, oo ≠ outOff s0 sob som sok k.val) :
    st'.mem r oo = st.mem r oo := by
  obtain ⟨c, hc⟩ := hiB
  subst hc
  rw [show B * c = c * B from Nat.mul_comm B c] at hstep
  unfold wbBody at hstep
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (offsn_eval B c st hblockN))] at hstep
  set s1 := (st.setReg "block_n_start_idx" .nat [] (Tile.scalar (c*B))).setReg "offs_n" .nat [B] (Tile.vec (fun j : Fin B => c*B + j.val)) with hs1
  have hoffsn1 : s1.regs .nat [B] "offs_n" = some (Tile.vec (fun j : Fin B => c*B + j.val)) := by simp [hs1]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (mask_eval N B s1 c hoffsn1))] at hstep
  set s2 := s1.setReg "x_ptr_mask" .bool [B] (Tile.vec (fun j : Fin B => decide (c*B + j.val < N))) with hs2
  have hoffsn2 : s2.regs .nat [B] "offs_n" = some (Tile.vec (fun j : Fin B => c*B + j.val)) := by
    rw [hs2]; simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offs_n":RegName) ≠ "x_ptr_mask" by decide)]; exact hoffsn1
  have hmask2 : s2.regs .bool [B] "x_ptr_mask" = some (Tile.vec (fun j : Fin B => decide (c*B + j.val < N))) := by rw [hs2]; simp
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (rmsw_eval w srw N B s2 c hoffsn2 hmask2))] at hstep
  set s3 := s2.setReg "rms_w" .real [B] ⟨fun j : TileIndex [B] => some (if c*B + j.1.val < N then s2.readMem w ((c*B+j.1.val)*srw) else s2.undef w ((c*B+j.1.val)*srw))⟩ with hs3
  have hpids3 : s3.pids = s0.pids := by rw [hs3, hs2, hs1]; simp [hpids]
  have hrm3x : s3.readMem x = s0.readMem x := by funext ofs; rw [hs3, hs2, hs1]; simp only [BlockState.setReg_readMem]; exact congrFun hrmx ofs
  have hoffsn3 : s3.regs .nat [B] "offs_n" = some (Tile.vec (fun j : Fin B => c*B + j.val)) := by
    rw [hs3]; simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offs_n":RegName) ≠ "rms_w" by decide)]; exact hoffsn2
  have hmask3 : s3.regs .bool [B] "x_ptr_mask" = some (Tile.vec (fun j : Fin B => decide (c*B + j.val < N))) := by
    rw [hs3]; simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("x_ptr_mask":RegName) ≠ "rms_w" by decide)]; exact hmask2
  have hoffsm3 : s3.regs .nat [] "offs_m" = some (Tile.scalar (s0.pids 0 * sxb + s0.pids 1 * sxm)) := by
    rw [hs3, hs2, hs1]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offs_m":RegName) ≠ "rms_w" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offs_m":RegName) ≠ "x_ptr_mask" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offs_m":RegName) ≠ "offs_n" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offs_m":RegName) ≠ "block_n_start_idx" by decide)]
    exact hoffsm
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (xload_eval x sxb sxm sxk N B s0 s3 c hpids3 hrm3x hoffsn3 hmask3 hoffsm3))] at hstep
  set s4 := s3.setReg "x" .real [B] ⟨fun j : TileIndex [B] => some (if c*B + j.1.val < N then s0.readMem x (xOff s0 sxb sxm sxk (c*B+j.1.val)) else 0)⟩ with hs4
  have hx4 : s4.regs .real [B] "x" = some ⟨fun j : TileIndex [B] => some (if c*B + j.1.val < N then s0.readMem x (xOff s0 sxb sxm sxk (c*B+j.1.val)) else 0)⟩ := by
    rw [hs4]; exact BlockState.setReg_same _ _ _ _ _
  have hrstd4 : s4.regs .real [] "rstd" = some (Tile.scalar (some RS)) := by
    rw [hs4, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("rstd":RegName) ≠ "x" by decide), hs3,
        BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("rstd":RegName) ≠ "rms_w" by decide), hs2,
        BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("rstd":RegName) ≠ "x_ptr_mask" by decide), hs1,
        BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("rstd":RegName) ≠ "offs_n" by decide),
        BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("rstd":RegName) ≠ "block_n_start_idx" by decide)]
    exact hrstd
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (xhat_eval B s4 (fun j => if c*B + j.val < N then s0.readMem x (xOff s0 sxb sxm sxk (c*B+j.val)) else 0) RS hx4 hrstd4))] at hstep
  set s5 := s4.setReg "x_hat" .real [B] ⟨fun j : TileIndex [B] => some ((if c*B + j.1.val < N then s0.readMem x (xOff s0 sxb sxm sxk (c*B+j.1.val)) else 0) * RS)⟩ with hs5
  have hxh5 : s5.regs .real [B] "x_hat" = some ⟨fun j : TileIndex [B] => some ((if c*B + j.1.val < N then s0.readMem x (xOff s0 sxb sxm sxk (c*B+j.1.val)) else 0) * RS)⟩ := by
    rw [hs5]; exact BlockState.setReg_same _ _ _ _ _
  have hrw5 : s5.regs .real [B] "rms_w" = some ⟨fun j : TileIndex [B] => some (if c*B + j.1.val < N then s2.readMem w ((c*B+j.1.val)*srw) else s2.undef w ((c*B+j.1.val)*srw))⟩ := by
    rw [hs5, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("rms_w":RegName) ≠ "x_hat" by decide), hs4,
        BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("rms_w":RegName) ≠ "x" by decide), hs3]
    exact BlockState.setReg_same _ _ _ _ _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (out_eval B s5 (fun j => (if c*B + j.val < N then s0.readMem x (xOff s0 sxb sxm sxk (c*B+j.val)) else 0) * RS)
      (fun j => if c*B + j.val < N then s2.readMem w ((c*B+j.val)*srw) else s2.undef w ((c*B+j.val)*srw)) hxh5 hrw5))] at hstep
  set s6 := s5.setReg "out" .real [B] ⟨fun j : TileIndex [B] => some (((if c*B + j.1.val < N then s0.readMem x (xOff s0 sxb sxm sxk (c*B+j.1.val)) else 0) * RS) * (if c*B + j.1.val < N then s2.readMem w ((c*B+j.1.val)*srw) else s2.undef w ((c*B+j.1.val)*srw)))⟩ with hs6
  have hpids6 : s6.pids = s0.pids := by rw [hs6, hs5, hs4]; simp [hpids3]
  have hpb6 : s6.regs .nat [] "pid_batch" = some (Tile.scalar (s6.pids 0)) := by
    rw [hpids6, hs6, hs5, hs4, hs3, hs2, hs1]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("pid_batch":RegName) ≠ "out" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("pid_batch":RegName) ≠ "x_hat" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("pid_batch":RegName) ≠ "x" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("pid_batch":RegName) ≠ "rms_w" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("pid_batch":RegName) ≠ "x_ptr_mask" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("pid_batch":RegName) ≠ "offs_n" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("pid_batch":RegName) ≠ "block_n_start_idx" by decide)]
    exact hpb
  have hpm6 : s6.regs .nat [] "pid_m" = some (Tile.scalar (s6.pids 1)) := by
    rw [hpids6, hs6, hs5, hs4, hs3, hs2, hs1]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("pid_m":RegName) ≠ "out" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("pid_m":RegName) ≠ "x_hat" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("pid_m":RegName) ≠ "x" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("pid_m":RegName) ≠ "rms_w" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("pid_m":RegName) ≠ "x_ptr_mask" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("pid_m":RegName) ≠ "offs_n" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("pid_m":RegName) ≠ "block_n_start_idx" by decide)]
    exact hpm
  have hoffsn6 : s6.regs .nat [B] "offs_n" = some (Tile.vec (fun j : Fin B => c*B + j.val)) := by
    rw [hs6, hs5, hs4]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offs_n":RegName) ≠ "out" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offs_n":RegName) ≠ "x_hat" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offs_n":RegName) ≠ "x" by decide)]
    exact hoffsn3
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (outoff_eval sob som sok B s6 c hpb6 hpm6 hoffsn6))] at hstep
  set s7 := s6.setReg "out_off" .nat [B] (Tile.vec (fun j : Fin B => outOff s6 sob som sok (c*B+j.val))) with hs7
  have houtoff7 : s7.regs .nat [B] "out_off" = some (Tile.vec (fun j : Fin B => outOff s0 sob som sok (c*B+j.val))) := by
    rw [hs7]; rw [BlockState.setReg_same]
    apply congrArg some; apply Tile.ext; intro j; unfold outOff; rw [hpids6]
  have hout7 : s7.regs .real [B] "out" = some ⟨fun j : TileIndex [B] => some (((if c*B + j.1.val < N then s0.readMem x (xOff s0 sxb sxm sxk (c*B+j.1.val)) else 0) * RS) * (if c*B + j.1.val < N then s2.readMem w ((c*B+j.1.val)*srw) else s2.undef w ((c*B+j.1.val)*srw)))⟩ := by
    rw [hs7, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("out":RegName) ≠ "out_off" by decide), hs6]
    exact BlockState.setReg_same _ _ _ _ _
  have hmask7 : s7.regs .bool [B] "x_ptr_mask" = some (Tile.vec (fun j : Fin B => decide (c*B + j.val < N))) := by
    rw [hs7, hs6, hs5, hs4]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("x_ptr_mask":RegName) ≠ "out_off" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("x_ptr_mask":RegName) ≠ "out" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("x_ptr_mask":RegName) ≠ "x_hat" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("x_ptr_mask":RegName) ≠ "x" by decide)]
    exact hmask3
  rw [stepStmts.cons_some (StoreLemma.store_step o B s7
    (fun j : Fin B => outOff s0 sob som sok (c*B+j.val))
    (fun j : Fin B => ((if c*B + j.val < N then s0.readMem x (xOff s0 sxb sxm sxk (c*B+j.val)) else 0) * RS) * (if c*B + j.val < N then s2.readMem w ((c*B+j.val)*srw) else s2.undef w ((c*B+j.val)*srw)))
    (fun j : Fin B => decide (c*B + j.val < N)) houtoff7 hout7 hmask7)] at hstep
  rw [stepStmts.nil] at hstep
  obtain rfl := Option.some.inj hstep
  refine Eq.trans (foldl_writeMem_bool_preserve_cell _ _ _ r oo _ _ ?_) ?_
  · intro b _ hmb hbad
    rcases hcond with hne | hno
    · exact hne hbad.1
    · have hbN2 : c * B + b.1.val < N := by simpa using hmb
      exact hno ⟨c * B + b.1.val, hbN2⟩ hbad.2
  · rw [hs7, hs6, hs5, hs4, hs3, hs2, hs1]
    rfl

/-! ### The `TraceSafeR` walk -/

/-- `evalOpR` = `evalOp` on the cast-free nat/bool index ops of the two
passes (register refs and nat arithmetic — `R` never enters). -/
private theorem rms_offsnR_eq (R : RoundingModel) (B : Nat) (s : BlockState) :
    evalOpR R (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "block_n_start_idx")
        (Op.ref .nat [B] "block_N")) s
      = evalOp (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "block_n_start_idx")
        (Op.ref .nat [B] "block_N")) s := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

private theorem rms_maskR_eq (R : RoundingModel) (N B : Nat) (s : BlockState) :
    evalOpR R (Op.lt .nat Broadcast.scalarR (Op.ref .nat [B] "offs_n") (Op.constNat N)) s
      = evalOp (Op.lt .nat Broadcast.scalarR (Op.ref .nat [B] "offs_n") (Op.constNat N)) s := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

private theorem rms_xaddrR_eq (R : RoundingModel) (sxk B : Nat) (s : BlockState) :
    evalOpR R (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "offs_m")
        (Op.mul .nat Broadcast.scalarR (Op.ref .nat [B] "offs_n") (Op.constNat sxk))) s
      = evalOp (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "offs_m")
        (Op.mul .nat Broadcast.scalarR (Op.ref .nat [B] "offs_n") (Op.constNat sxk))) s := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

private theorem rms_waddrR_eq (R : RoundingModel) (srw B : Nat) (s : BlockState) :
    evalOpR R (Op.mul .nat Broadcast.scalarR (Op.ref .nat [B] "offs_n") (Op.constNat srw)) s
      = evalOp (Op.mul .nat Broadcast.scalarR (Op.ref .nat [B] "offs_n") (Op.constNat srw)) s := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

private theorem rms_outoffR_eq (R : RoundingModel) (sob som sok B : Nat) (s : BlockState) :
    evalOpR R (Op.add .nat Broadcast.scalarL
        (Op.add .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_batch") (Op.constNat sob))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_m") (Op.constNat som)))
        (Op.mul .nat Broadcast.scalarR (Op.ref .nat [B] "offs_n") (Op.constNat sok))) s
      = evalOp (Op.add .nat Broadcast.scalarL
        (Op.add .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_batch") (Op.constNat sob))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_m") (Op.constNat som)))
        (Op.mul .nat Broadcast.scalarR (Op.ref .nat [B] "offs_n") (Op.constNat sok))) s := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

/-- The per-lane value of the `x` load's address op (`offs_m + offs_n * sxk`). -/
private theorem rms_xaddr_eval (sxk B : Nat) (base : Nat) (s : BlockState) (c : Nat)
    (hom : s.regs .nat [] "offs_m" = some (Tile.scalar base))
    (hoffsn : s.regs .nat [B] "offs_n" = some (Tile.vec (fun j : Fin B => c*B + j.val))) :
    evalOp (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "offs_m")
        (Op.mul .nat Broadcast.scalarR (Op.ref .nat [B] "offs_n") (Op.constNat sxk))) s
      = some (Tile.vec (fun j : Fin B => base + (c*B + j.val) * sxk)) := by
  rw [evalOp_add, evalOp_ref, hom, evalOp_mul, evalOp_ref, hoffsn, evalOp_constNat]
  apply congrArg some
  ext j
  simp only [Tile.bop_data, Tile.vec, Tile.scalar, Broadcast.leftIndex_scalarL,
    Broadcast.rightIndex_scalarL, Broadcast.leftIndex_scalarR,
    Broadcast.rightIndex_scalarR, NumericDType.add, NumericDType.mul]

/-- The per-lane value of the `rms_w` load's address op (`offs_n * srw`). -/
private theorem rms_waddr_eval (srw B : Nat) (s : BlockState) (c : Nat)
    (hoffsn : s.regs .nat [B] "offs_n" = some (Tile.vec (fun j : Fin B => c*B + j.val))) :
    evalOp (Op.mul .nat Broadcast.scalarR (Op.ref .nat [B] "offs_n") (Op.constNat srw)) s
      = some (Tile.vec (fun j : Fin B => (c*B + j.val) * srw)) := by
  rw [evalOp_mul, evalOp_ref, hoffsn, evalOp_constNat]
  apply congrArg some
  ext j
  simp only [Tile.bop_data, Tile.vec, Tile.scalar, Broadcast.leftIndex_scalarR,
    Broadcast.rightIndex_scalarR, NumericDType.mul]

set_option maxHeartbeats 4000000 in
/-- Per-iteration `TraceSafeListR` for the pass-1 body: the index/mask
assigns and the multiply-add are register-only; the masked `x` load's
**active** lanes are exactly the skin's `mask1` window at step `i / B`, in
bounds by the `read1` window bound (instantiated at raw counter `i`). -/
private theorem rms_varBodySafeR (R : RoundingModel) (bounds : RegionBounds)
    (x : RegionName) (sxb sxm sxk N B : Nat)
    (s0 st : BlockState) (i : Nat) (hiB : B ∣ i)
    (hbN : st.regs .nat [B] "block_N" = some (Tile.vec (fun j : Fin B => j.val)))
    (hom : st.regs .nat [] "offs_m" = some (Tile.scalar (s0.pids 0 * sxb + s0.pids 1 * sxm)))
    (hbx : ∀ j : Fin B, i + j.val < N →
      s0.pids 0 * sxb + s0.pids 1 * sxm + (i + j.val) * sxk < bounds x) :
    Stmt.TraceSafeListR R bounds (varBody x sxk N B)
      (st.setReg "block_n_start_idx" .nat [] (Tile.scalar i)) := by
  obtain ⟨c, rfl⟩ := hiB
  rw [show B * c = c * B from Nat.mul_comm B c] at hbx ⊢
  unfold varBody
  -- offs_n
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun t1 ht1 => ?_)
  rw [stepStmtR_assign_eq_some
    ((rms_offsnR_eq R B _).trans (offsn_eval B c st hbN))] at ht1
  obtain rfl := Option.some.inj ht1
  set q1 := (st.setReg "block_n_start_idx" .nat [] (Tile.scalar (c*B))).setReg
    "offs_n" .nat [B] (Tile.vec (fun j : Fin B => c*B + j.val)) with hq1
  have hoffsn1 : q1.regs .nat [B] "offs_n"
      = some (Tile.vec (fun j : Fin B => c*B + j.val)) := by simp [hq1]
  -- x_ptr_mask
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun t2 ht2 => ?_)
  rw [stepStmtR_assign_eq_some
    ((rms_maskR_eq R N B _).trans (mask_eval N B q1 c hoffsn1))] at ht2
  obtain rfl := Option.some.inj ht2
  set q2 := q1.setReg "x_ptr_mask" .bool [B]
    (Tile.vec (fun j : Fin B => decide (c*B + j.val < N))) with hq2
  have hoffsn2 : q2.regs .nat [B] "offs_n"
      = some (Tile.vec (fun j : Fin B => c*B + j.val)) := by
    rw [hq2]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
      (show ("offs_n":RegName) ≠ "x_ptr_mask" by decide)]
    exact hoffsn1
  have hmask2 : q2.regs .bool [B] "x_ptr_mask"
      = some (Tile.vec (fun j : Fin B => decide (c*B + j.val < N))) := by
    rw [hq2]; simp
  have hom2 : q2.regs .nat [] "offs_m"
      = some (Tile.scalar (s0.pids 0 * sxb + s0.pids 1 * sxm)) := by
    rw [hq2, hq1]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("offs_m":RegName) ≠ "x_ptr_mask" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("offs_m":RegName) ≠ "offs_n" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("offs_m":RegName) ≠ "block_n_start_idx" by decide)]
    exact hom
  -- the masked x load
  refine Stmt.TraceSafeListR.cons_intro ?_ (fun t3 ht3 => ?_)
  · simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR,
      MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR,
      and_true, true_and, and_self]
    intro offsets hoffsets idx hactive
    rw [rms_xaddrR_eq,
      rms_xaddr_eval sxk B (s0.pids 0 * sxb + s0.pids 1 * sxm) q2 c hom2 hoffsn2] at hoffsets
    obtain rfl := Option.some.inj hoffsets
    obtain ⟨masks, hm, hmi⟩ := hactive
    rw [evalOpR_ref, hmask2] at hm
    obtain rfl := Option.some.inj hm
    have hlt : c * B + idx.1.val < N := by simpa [Tile.vec] using hmi
    simpa [Region.cast_id, Tile.vec] using hbx idx.1 hlt
  · obtain ⟨v3, -, rfl⟩ := stepStmtR_assign_inv ht3
    -- var accumulate: register-only
    exact Stmt.TraceSafeListR.cons_intro
      (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
      (fun _ _ => Stmt.TraceSafeListR.nil_intro)

set_option maxHeartbeats 8000000 in
/-- Per-iteration `TraceSafeListR` for the pass-2 body: the index/mask/
scaling assigns are register-only; the masked `rms_w` / `x` loads' and the
masked store's **active** lanes are the skin's `mask2` / `mask1` /
`writeMask` windows at step `i / B`, in bounds by the corresponding window
bounds (instantiated at raw counter `i`). -/
private theorem rms_wbBodySafeR (R : RoundingModel) (bounds : RegionBounds)
    (x w o : RegionName) (sxb sxm sxk srw sob som sok N B : Nat)
    (s0 st : BlockState) (i : Nat) (hiB : B ∣ i)
    (hpids : st.pids = s0.pids)
    (hpb : st.regs .nat [] "pid_batch" = some (Tile.scalar (s0.pids 0)))
    (hpm : st.regs .nat [] "pid_m" = some (Tile.scalar (s0.pids 1)))
    (hbN : st.regs .nat [B] "block_N" = some (Tile.vec (fun j : Fin B => j.val)))
    (hom : st.regs .nat [] "offs_m" = some (Tile.scalar (s0.pids 0 * sxb + s0.pids 1 * sxm)))
    (hbx : ∀ j : Fin B, i + j.val < N →
      s0.pids 0 * sxb + s0.pids 1 * sxm + (i + j.val) * sxk < bounds x)
    (hbw : ∀ j : Fin B, i + j.val < N → (i + j.val) * srw < bounds w)
    (hbo : ∀ j : Fin B, i + j.val < N →
      s0.pids 0 * sob + s0.pids 1 * som + (i + j.val) * sok < bounds o) :
    Stmt.TraceSafeListR R bounds (wbBody x w o sxb sxm sxk srw sob som sok N B)
      (st.setReg "block_n_start_idx" .nat [] (Tile.scalar i)) := by
  obtain ⟨c, rfl⟩ := hiB
  rw [show B * c = c * B from Nat.mul_comm B c] at hbx hbw hbo ⊢
  unfold wbBody
  -- offs_n
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun t1 ht1 => ?_)
  rw [stepStmtR_assign_eq_some
    ((rms_offsnR_eq R B _).trans (offsn_eval B c st hbN))] at ht1
  obtain rfl := Option.some.inj ht1
  set q1 := (st.setReg "block_n_start_idx" .nat [] (Tile.scalar (c*B))).setReg
    "offs_n" .nat [B] (Tile.vec (fun j : Fin B => c*B + j.val)) with hq1
  have hoffsn1 : q1.regs .nat [B] "offs_n"
      = some (Tile.vec (fun j : Fin B => c*B + j.val)) := by simp [hq1]
  -- x_ptr_mask
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun t2 ht2 => ?_)
  rw [stepStmtR_assign_eq_some
    ((rms_maskR_eq R N B _).trans (mask_eval N B q1 c hoffsn1))] at ht2
  obtain rfl := Option.some.inj ht2
  set q2 := q1.setReg "x_ptr_mask" .bool [B]
    (Tile.vec (fun j : Fin B => decide (c*B + j.val < N))) with hq2
  have hoffsn2 : q2.regs .nat [B] "offs_n"
      = some (Tile.vec (fun j : Fin B => c*B + j.val)) := by
    rw [hq2]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
      (show ("offs_n":RegName) ≠ "x_ptr_mask" by decide)]
    exact hoffsn1
  have hmask2 : q2.regs .bool [B] "x_ptr_mask"
      = some (Tile.vec (fun j : Fin B => decide (c*B + j.val < N))) := by
    rw [hq2]; simp
  have hom2 : q2.regs .nat [] "offs_m"
      = some (Tile.scalar (s0.pids 0 * sxb + s0.pids 1 * sxm)) := by
    rw [hq2, hq1]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("offs_m":RegName) ≠ "x_ptr_mask" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("offs_m":RegName) ≠ "offs_n" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("offs_m":RegName) ≠ "block_n_start_idx" by decide)]
    exact hom
  have hpb2 : q2.regs .nat [] "pid_batch" = some (Tile.scalar (s0.pids 0)) := by
    rw [hq2, hq1]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("pid_batch":RegName) ≠ "x_ptr_mask" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("pid_batch":RegName) ≠ "offs_n" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("pid_batch":RegName) ≠ "block_n_start_idx" by decide)]
    exact hpb
  have hpm2 : q2.regs .nat [] "pid_m" = some (Tile.scalar (s0.pids 1)) := by
    rw [hq2, hq1]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("pid_m":RegName) ≠ "x_ptr_mask" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("pid_m":RegName) ≠ "offs_n" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("pid_m":RegName) ≠ "block_n_start_idx" by decide)]
    exact hpm
  have hpids2 : q2.pids = s0.pids := by rw [hq2, hq1]; simp [hpids]
  -- the masked rms_w load
  refine Stmt.TraceSafeListR.cons_intro ?_ (fun t3 ht3 => ?_)
  · simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR,
      MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR,
      and_true, true_and, and_self]
    intro offsets hoffsets idx hactive
    rw [rms_waddrR_eq, rms_waddr_eval srw B q2 c hoffsn2] at hoffsets
    obtain rfl := Option.some.inj hoffsets
    obtain ⟨masks, hm, hmi⟩ := hactive
    rw [evalOpR_ref, hmask2] at hm
    obtain rfl := Option.some.inj hm
    have hlt : c * B + idx.1.val < N := by simpa [Tile.vec] using hmi
    simpa [Region.cast_id, Tile.vec] using hbw idx.1 hlt
  · obtain ⟨v3, -, rfl⟩ := stepStmtR_assign_inv ht3
    set q3 := q2.setReg "rms_w" .real [B] v3 with hq3
    have hoffsn3 : q3.regs .nat [B] "offs_n"
        = some (Tile.vec (fun j : Fin B => c*B + j.val)) := by
      rw [hq3]
      simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("offs_n":RegName) ≠ "rms_w" by decide)]
      exact hoffsn2
    have hmask3 : q3.regs .bool [B] "x_ptr_mask"
        = some (Tile.vec (fun j : Fin B => decide (c*B + j.val < N))) := by
      rw [hq3]
      simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("x_ptr_mask":RegName) ≠ "rms_w" by decide)]
      exact hmask2
    have hom3 : q3.regs .nat [] "offs_m"
        = some (Tile.scalar (s0.pids 0 * sxb + s0.pids 1 * sxm)) := by
      rw [hq3]
      simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("offs_m":RegName) ≠ "rms_w" by decide)]
      exact hom2
    -- the masked x load
    refine Stmt.TraceSafeListR.cons_intro ?_ (fun t4 ht4 => ?_)
    · simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR,
        MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR,
        and_true, true_and, and_self]
      intro offsets hoffsets idx hactive
      rw [rms_xaddrR_eq,
        rms_xaddr_eval sxk B (s0.pids 0 * sxb + s0.pids 1 * sxm) q3 c hom3 hoffsn3] at hoffsets
      obtain rfl := Option.some.inj hoffsets
      obtain ⟨masks, hm, hmi⟩ := hactive
      rw [evalOpR_ref, hmask3] at hm
      obtain rfl := Option.some.inj hm
      have hlt : c * B + idx.1.val < N := by simpa [Tile.vec] using hmi
      simpa [Region.cast_id, Tile.vec] using hbx idx.1 hlt
    · obtain ⟨v4, -, rfl⟩ := stepStmtR_assign_inv ht4
      set q4 := q3.setReg "x" .real [B] v4 with hq4
      -- x_hat (register-only)
      refine Stmt.TraceSafeListR.cons_intro
        (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun t5 ht5 => ?_)
      obtain ⟨v5, -, rfl⟩ := stepStmtR_assign_inv ht5
      set q5 := q4.setReg "x_hat" .real [B] v5 with hq5
      -- out (register-only)
      refine Stmt.TraceSafeListR.cons_intro
        (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun t6 ht6 => ?_)
      obtain ⟨v6, -, rfl⟩ := stepStmtR_assign_inv ht6
      set q6 := q5.setReg "out" .real [B] v6 with hq6
      -- out_off
      have hpids6 : q6.pids = s0.pids := by
        rw [hq6, hq5, hq4, hq3]; simp [hpids2]
      have hpb6 : q6.regs .nat [] "pid_batch" = some (Tile.scalar (q6.pids 0)) := by
        rw [hpids6, hq6, hq5, hq4, hq3]
        simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
            (show ("pid_batch":RegName) ≠ "out" by decide),
          BlockState.setReg_ne_name _ _ _ _ _ _ _ _
            (show ("pid_batch":RegName) ≠ "x_hat" by decide),
          BlockState.setReg_ne_name _ _ _ _ _ _ _ _
            (show ("pid_batch":RegName) ≠ "x" by decide),
          BlockState.setReg_ne_name _ _ _ _ _ _ _ _
            (show ("pid_batch":RegName) ≠ "rms_w" by decide)]
        exact hpb2
      have hpm6 : q6.regs .nat [] "pid_m" = some (Tile.scalar (q6.pids 1)) := by
        rw [hpids6, hq6, hq5, hq4, hq3]
        simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
            (show ("pid_m":RegName) ≠ "out" by decide),
          BlockState.setReg_ne_name _ _ _ _ _ _ _ _
            (show ("pid_m":RegName) ≠ "x_hat" by decide),
          BlockState.setReg_ne_name _ _ _ _ _ _ _ _
            (show ("pid_m":RegName) ≠ "x" by decide),
          BlockState.setReg_ne_name _ _ _ _ _ _ _ _
            (show ("pid_m":RegName) ≠ "rms_w" by decide)]
        exact hpm2
      have hoffsn6 : q6.regs .nat [B] "offs_n"
          = some (Tile.vec (fun j : Fin B => c*B + j.val)) := by
        rw [hq6, hq5, hq4]
        simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
            (show ("offs_n":RegName) ≠ "out" by decide),
          BlockState.setReg_ne_name _ _ _ _ _ _ _ _
            (show ("offs_n":RegName) ≠ "x_hat" by decide),
          BlockState.setReg_ne_name _ _ _ _ _ _ _ _
            (show ("offs_n":RegName) ≠ "x" by decide)]
        exact hoffsn3
      have hmask6 : q6.regs .bool [B] "x_ptr_mask"
          = some (Tile.vec (fun j : Fin B => decide (c*B + j.val < N))) := by
        rw [hq6, hq5, hq4]
        simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
            (show ("x_ptr_mask":RegName) ≠ "out" by decide),
          BlockState.setReg_ne_name _ _ _ _ _ _ _ _
            (show ("x_ptr_mask":RegName) ≠ "x_hat" by decide),
          BlockState.setReg_ne_name _ _ _ _ _ _ _ _
            (show ("x_ptr_mask":RegName) ≠ "x" by decide)]
        exact hmask3
      have hoo : evalOp (Op.add .nat Broadcast.scalarL
          (Op.add .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_batch") (Op.constNat sob))
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_m") (Op.constNat som)))
          (Op.mul .nat Broadcast.scalarR (Op.ref .nat [B] "offs_n") (Op.constNat sok))) q6
          = some (Tile.vec (fun j : Fin B => outOff s0 sob som sok (c*B + j.val))) := by
        rw [outoff_eval sob som sok B q6 c hpb6 hpm6 hoffsn6]
        apply congrArg some
        apply Tile.ext
        intro j
        unfold outOff
        rw [hpids6]
      refine Stmt.TraceSafeListR.cons_intro
        (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun t7 ht7 => ?_)
      rw [stepStmtR_assign_eq_some ((rms_outoffR_eq R sob som sok B q6).trans hoo)] at ht7
      obtain rfl := Option.some.inj ht7
      set q7 := q6.setReg "out_off" .nat [B]
        (Tile.vec (fun j : Fin B => outOff s0 sob som sok (c*B + j.val))) with hq7
      have houtoff7 : q7.regs .nat [B] "out_off"
          = some (Tile.vec (fun j : Fin B => outOff s0 sob som sok (c*B + j.val))) := by
        rw [hq7]; simp
      have hmask7 : q7.regs .bool [B] "x_ptr_mask"
          = some (Tile.vec (fun j : Fin B => decide (c*B + j.val < N))) := by
        rw [hq7]
        simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
          (show ("x_ptr_mask":RegName) ≠ "out_off" by decide)]
        exact hmask6
      -- the masked store
      refine Stmt.TraceSafeListR.cons_intro ?_
        (fun _ _ => Stmt.TraceSafeListR.nil_intro)
      simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MemAccess.SafeAtR,
        MaskOpt.SafeAtR, MaskOpt.ActiveR, MemAccess.ActiveAddressSafeR,
        memAccessActiveAddressSafeR, and_true, true_and, and_self]
      intro offsets hoffsets idx hactive
      rw [evalOpR_ref, houtoff7] at hoffsets
      obtain rfl := Option.some.inj hoffsets
      obtain ⟨masks, hm, hmi⟩ := hactive
      rw [evalOpR_ref, hmask7] at hm
      obtain rfl := Option.some.inj hm
      have hlt : c * B + idx.1.val < N := by simpa [Tile.vec] using hmi
      simpa [Region.cast_id, Tile.vec, outOff] using hbo idx.1 hlt

set_option maxHeartbeats 8000000 in
/-- **The `TraceSafeR` walk for the whole two-pass kernel** — driven by
`Stmt.forRangeTraceSafeR_inv` over the (launch-state-robust) `varInv` for
pass 1 and `wbInv` for pass 2, with the counter advancing by the loops'
stride `BLOCK_N_SIZE`. The three bound groups are the skin's
`read1`/`read2`/`write` windows; `hB` converts the raw counter into the
step index `i / B < ⌈N/B⌉` the windows are phrased over, and
`hox`/`how`/`hsok` feed the `wbInv` step (its readback clause needs the
store not to clobber the streamed inputs and the write window to be
injective). -/
private theorem rmsnorm_traceSafeR (R : RoundingModel) (bounds : RegionBounds)
    (x w o : RegionName) (sxb sxm sxk srw sob som sok N B : Nat) (eps : ℝ)
    (hB : 0 < B) (hox : o ≠ x) (how : o ≠ w) (hsok : 0 < sok)
    (s : BlockState)
    (hbx : ∀ (t : Fin (rmsNumSteps N B)) (j : Fin B), t.val * B + j.val < N →
      s.pids 0 * sxb + s.pids 1 * sxm + (t.val * B + j.val) * sxk < bounds x)
    (hbw : ∀ (t : Fin (rmsNumSteps N B)) (j : Fin B), t.val * B + j.val < N →
      (t.val * B + j.val) * srw < bounds w)
    (hbo : ∀ (t : Fin (rmsNumSteps N B)) (j : Fin B), t.val * B + j.val < N →
      s.pids 0 * sob + s.pids 1 * som + (t.val * B + j.val) * sok < bounds o) :
    ((rmsnorm_triton x w o sxb sxm sxk srw sob som sok N B eps).toAlgKernel).TraceSafeR R bounds s := by
  have hinj : Function.Injective (fun k : Fin N => outOff s sob som sok k.val) :=
    outOff_injective_of_pos s sob som sok N hsok
  -- window instantiators at a raw in-range counter
  have hstepT : ∀ i, B ∣ i → ∀ j : Fin B, i + j.val < N →
      ∃ t : Fin (rmsNumSteps N B), t.val * B + j.val = i + j.val := by
    intro i hiB j hij
    have hiN : i < N := Nat.lt_of_le_of_lt (Nat.le_add_right _ _) hij
    refine ⟨⟨i / B, rmsStep_lt_numSteps N B i hB hiN⟩, ?_⟩
    simp [Nat.div_mul_cancel hiB]
  have hbx' : ∀ i, B ∣ i → ∀ j : Fin B, i + j.val < N →
      s.pids 0 * sxb + s.pids 1 * sxm + (i + j.val) * sxk < bounds x := by
    intro i hiB j hij
    obtain ⟨t, ht⟩ := hstepT i hiB j hij
    have h := hbx t j (by rw [ht]; exact hij)
    rwa [ht] at h
  have hbw' : ∀ i, B ∣ i → ∀ j : Fin B, i + j.val < N →
      (i + j.val) * srw < bounds w := by
    intro i hiB j hij
    obtain ⟨t, ht⟩ := hstepT i hiB j hij
    have h := hbw t j (by rw [ht]; exact hij)
    rwa [ht] at h
  have hbo' : ∀ i, B ∣ i → ∀ j : Fin B, i + j.val < N →
      s.pids 0 * sob + s.pids 1 * som + (i + j.val) * sok < bounds o := by
    intro i hiB j hij
    obtain ⟨t, ht⟩ := hstepT i hiB j hij
    have h := hbo t j (by rw [ht]; exact hij)
    rwa [ht] at h
  unfold Kernel.TraceSafeR
  rw [body_decomp]
  simp only [List.append_assoc, List.singleton_append]
  refine Stmt.TraceSafeListR.append_intro _ _ ?_ ?_
  · -- prologue: register-only assigns, safe at every state
    refine Stmt.TraceSafeListR.of_forall _ _ ?_
    intro stmt hst s'
    simp only [prefixStmts, List.mem_cons, List.not_mem_nil, or_false] at hst
    rcases hst with rfl | rfl | rfl | rfl | rfl <;>
      simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]
  · intro s1 hs1
    obtain ⟨s1x, hpre, hP0⟩ := preLoop x sxb sxm sxk N B s
    rw [rmsPrefix_castFree R x sxb sxm B s, hpre] at hs1
    obtain rfl := Option.some.inj hs1
    refine Stmt.TraceSafeListR.cons_intro ?_ ?_
    · -- pass 1 (invariant principle over `varInv`)
      simp only [Stmt.TraceSafeR]
      refine Stmt.forRangeTraceSafeR_inv R bounds "block_n_start_idx" N B
        (varBody x sxk N B) (varInv x sxb sxm sxk N B s) ?_ 0 s1x hP0
      intro i stt hi hP
      have hPd := hP
      obtain ⟨hpids, hrm, hiB, hpb, hpm, hbNr, homr, hvar⟩ := hPd
      refine ⟨rms_varBodySafeR R bounds x sxb sxm sxk N B s stt i hiB hbNr homr
        (fun j hj => hbx' i hiB j hj), ?_⟩
      obtain ⟨st', hstep, hP'⟩ := varInv_step x sxb sxm sxk N B s hB i stt hi hP
      exact ⟨st', by rw [rmsVarBody_castFree]; exact hstep, hP'⟩
    · intro s2 hs2
      obtain ⟨finalv, sv, hsv, hNlev, hinvV⟩ := var_forRange x sxb sxm sxk N B s hB s1x hP0
      rw [stepStmtR_forRange,
        stepForRangeAuxR_castFree R _ (rmsVarBody_castFree R x sxk N B) "block_n_start_idx",
        ← stepForRangeAux.forRange_unfold, hsv] at hs2
      obtain rfl := Option.some.inj hs2
      obtain ⟨hpidsV, hrmV, hdvdV, hpbV, hpmV, hbNV, homV, hvarV⟩ := hinvV
      refine Stmt.TraceSafeListR.append_intro _ _ ?_ ?_
      · -- reduce/rsqrt tail: register-only assigns
        refine Stmt.TraceSafeListR.of_forall _ _ ?_
        intro stmt hst s'
        simp only [postStmts, List.mem_cons, List.not_mem_nil, or_false] at hst
        rcases hst with rfl | rfl <;> simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]
      · intro s3 hs3
        obtain ⟨sr, hsr, hrstdR, hpresR, hpidsR, hrmR⟩ :=
          postStep s x sxb sxm sxk N B (finalv / B) eps sv hvarV
        rw [rmsPost_castFree R N B eps sv, hsr] at hs3
        obtain rfl := Option.some.inj hs3
        -- the `wbInv` base at the post state (mirrors the exact headline)
        have hcvge : N ≤ finalv / B * B := by
          have hdvd : finalv / B * B = finalv := Nat.div_mul_cancel hdvdV
          omega
        have hmean : (∑ j : Fin B, varAcc s x sxb sxm sxk N B (finalv / B) j) / (N:ℝ)
            = meanSq s x sxb sxm sxk N := by
          unfold meanSq
          rw [varAcc_sum_eq s x sxb sxm sxk N B (finalv / B) hB hcvge]
        set RS : ℝ := WithBot.unbotD 0 (WithBot.realRsqrt (some (meanSq s x sxb sxm sxk N + eps))) with hRSdef
        have hRS0 : sr.regs .real [] "rstd" = some (Tile.scalar (WithBot.realRsqrt (some (meanSq s x sxb sxm sxk N + eps)))) := by
          rw [hrstdR, hmean]
        have hRSsome : WithBot.realRsqrt (some (meanSq s x sxb sxm sxk N + eps)) = some RS := by
          rw [hRSdef]; simp [WithBot.realRsqrt]
        have hwbBase : wbInv x w o sxb sxm sxk srw sob som sok N B s RS s 0 sr := by
          refine ⟨?_, ⟨0, rfl⟩, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
          · rw [hpidsR, hpidsV]
          · rw [hpresR _ _ _ (by decide) (by decide)]; exact hpbV
          · rw [hpresR _ _ _ (by decide) (by decide)]; exact hpmV
          · rw [hpresR _ _ _ (by decide) (by decide)]; exact hbNV
          · rw [hpresR _ _ _ (by decide) (by decide)]; exact homV
          · rw [hRS0, hRSsome]
          · funext ofs; rw [congrFun (congrFun hrmR x) ofs, congrFun (congrFun hrmV x) ofs]
          · funext ofs; rw [congrFun (congrFun hrmR w) ofs, congrFun (congrFun hrmV w) ofs]
          · intro k; rw [if_neg (by omega)]
            rw [congrFun (congrFun hrmR o) _, congrFun (congrFun hrmV o) _]
        -- pass 2 (invariant principle over `wbInv`)
        refine Stmt.TraceSafeListR.cons_intro ?_ (fun s4 _ => Stmt.TraceSafeListR.nil_intro)
        simp only [Stmt.TraceSafeR]
        refine Stmt.forRangeTraceSafeR_inv R bounds "block_n_start_idx" N B
          (wbBody x w o sxb sxm sxk srw sob som sok N B)
          (wbInv x w o sxb sxm sxk srw sob som sok N B s RS s) ?_ 0 sr hwbBase
        intro i stt hi hQ
        have hQd := hQ
        obtain ⟨hpidsQ, hiBQ, hpbQ, hpmQ, hbNQ, homQ, hrsQ, hrxQ, hrwQ, hmemQ⟩ := hQd
        refine ⟨rms_wbBodySafeR R bounds x w o sxb sxm sxk srw sob som sok N B s stt i hiBQ
            hpidsQ hpbQ hpmQ hbNQ homQ
            (fun j hj => hbx' i hiBQ j hj) (fun j hj => hbw' i hiBQ j hj)
            (fun j hj => hbo' i hiBQ j hj), ?_⟩
        obtain ⟨st', hstep, hQ'⟩ := wbInv_step x w o sxb sxm sxk srw sob som sok N B s RS s hB hinj hox how i stt hi hQ
        exact ⟨st', by rw [rmsWbBody_castFree]; exact hstep, hQ'⟩

/-! ### The rounded Hoare triple (`hrun`) -/

set_option maxHeartbeats 8000000 in
/-- Termination, per-lane values and the per-cell frame of the whole
two-pass kernel under `execR R`, from an **arbitrary** launch state: the
exact `preLoop` / `varInv` / `postStep` / `wbInv` stack runs verbatim (both
passes are cast-free, so `execR R` collapses onto the exact stepper),
extended with the per-segment memory frames. -/
private theorem rmsnorm_runR (R : RoundingModel) (x w o : RegionName)
    (sxb sxm sxk srw sob som sok N B : Nat) (eps : ℝ)
    (hB : 0 < B) (hox : o ≠ x) (how : o ≠ w) (hsok : 0 < sok)
    (s₀ : BlockState) :
    ∃ sfin,
      execR R (rmsnorm_triton x w o sxb sxm sxk srw sob som sok N B eps).toAlgKernel s₀ = some sfin
      ∧ (∀ k : Fin N, sfin.readMem o (outOff s₀ sob som sok k.val)
          = rmsSpecFull s₀ x w sxb sxm sxk srw N eps k.val)
      ∧ (∀ r oo, (r ≠ o ∨ ∀ k : Fin N, oo ≠ outOff s₀ sob som sok k.val) →
          sfin.mem r oo = s₀.mem r oo) := by
  have hinj : Function.Injective (fun k : Fin N => outOff s₀ sob som sok k.val) :=
    outOff_injective_of_pos s₀ sob som sok N hsok
  -- prologue
  obtain ⟨sp, hsp, hP0⟩ := preLoop x sxb sxm sxk N B s₀
  have hspMem : sp.mem = s₀.mem :=
    stepStmts_assigns_mem (prefixStmts x sxb sxm B)
      (by
        intro stmt hst
        simp only [prefixStmts, List.mem_cons, List.not_mem_nil, or_false] at hst
        rcases hst with rfl | rfl | rfl | rfl | rfl <;> exact ⟨_, _, _, _, rfl⟩)
      hsp
  -- pass 1 with the memory frame carried alongside `varInv`
  obtain ⟨finalv, sv, hsv, hNlev, hPV⟩ :=
    forRange_inv (idx := "block_n_start_idx") (start := 0) (stop := N) (step := B)
      (body := varBody x sxk N B)
      (P := fun i stt => varInv x sxb sxm sxk N B s₀ i stt ∧ stt.mem = s₀.mem)
      (Nat.pos_iff_ne_zero.mp hB) ⟨hP0, hspMem⟩
      (fun i stt hlt hP => by
        obtain ⟨st', hstep, hinv'⟩ := varInv_step x sxb sxm sxk N B s₀ hB i stt hlt hP.1
        refine ⟨st', hstep, hinv', ?_⟩
        have hmm : st'.mem = (stt.setReg "block_n_start_idx" .nat [] (Tile.scalar i)).mem :=
          stepStmts_assigns_mem (varBody x sxk N B)
            (by
              intro stmt hst
              simp only [varBody, List.mem_cons, List.not_mem_nil, or_false] at hst
              rcases hst with rfl | rfl | rfl | rfl <;> exact ⟨_, _, _, _, rfl⟩)
            hstep
        rw [hmm]
        exact hP.2)
  obtain ⟨⟨hpidsV, hrmV, hdvdV, hpbV, hpmV, hbNV, homV, hvarV⟩, hmemV⟩ := hPV
  -- reduce/rsqrt tail
  obtain ⟨sr, hsr, hrstdR, hpresR, hpidsR, hrmR⟩ :=
    postStep s₀ x sxb sxm sxk N B (finalv / B) eps sv hvarV
  have hsrMem : sr.mem = s₀.mem := by
    rw [stepStmts_assigns_mem (postStmts N B eps)
      (by
        intro stmt hst
        simp only [postStmts, List.mem_cons, List.not_mem_nil, or_false] at hst
        rcases hst with rfl | rfl <;> exact ⟨_, _, _, _, rfl⟩)
      hsr]
    exact hmemV
  -- the `wbInv` base at the post state (mirrors the exact headline)
  have hcvge : N ≤ finalv / B * B := by
    have hdvd : finalv / B * B = finalv := Nat.div_mul_cancel hdvdV
    omega
  have hmean : (∑ j : Fin B, varAcc s₀ x sxb sxm sxk N B (finalv / B) j) / (N:ℝ)
      = meanSq s₀ x sxb sxm sxk N := by
    unfold meanSq
    rw [varAcc_sum_eq s₀ x sxb sxm sxk N B (finalv / B) hB hcvge]
  set RS : ℝ := WithBot.unbotD 0 (WithBot.realRsqrt (some (meanSq s₀ x sxb sxm sxk N + eps))) with hRSdef
  have hRS0 : sr.regs .real [] "rstd" = some (Tile.scalar (WithBot.realRsqrt (some (meanSq s₀ x sxb sxm sxk N + eps)))) := by
    rw [hrstdR, hmean]
  have hRSsome : WithBot.realRsqrt (some (meanSq s₀ x sxb sxm sxk N + eps)) = some RS := by
    rw [hRSdef]; simp [WithBot.realRsqrt]
  have hwbBase : wbInv x w o sxb sxm sxk srw sob som sok N B s₀ RS s₀ 0 sr := by
    refine ⟨?_, ⟨0, rfl⟩, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · rw [hpidsR, hpidsV]
    · rw [hpresR _ _ _ (by decide) (by decide)]; exact hpbV
    · rw [hpresR _ _ _ (by decide) (by decide)]; exact hpmV
    · rw [hpresR _ _ _ (by decide) (by decide)]; exact hbNV
    · rw [hpresR _ _ _ (by decide) (by decide)]; exact homV
    · rw [hRS0, hRSsome]
    · funext ofs; rw [congrFun (congrFun hrmR x) ofs, congrFun (congrFun hrmV x) ofs]
    · funext ofs; rw [congrFun (congrFun hrmR w) ofs, congrFun (congrFun hrmV w) ofs]
    · intro k; rw [if_neg (by omega)]
      rw [congrFun (congrFun hrmR o) _, congrFun (congrFun hrmV o) _]
  -- pass 2 with the conditional cell frame carried alongside `wbInv`
  obtain ⟨finalw, sw, hsw, hNlew, hQW⟩ :=
    forRange_inv (idx := "block_n_start_idx") (start := 0) (stop := N) (step := B)
      (body := wbBody x w o sxb sxm sxk srw sob som sok N B)
      (P := fun i stt => wbInv x w o sxb sxm sxk srw sob som sok N B s₀ RS s₀ i stt
        ∧ ∀ r oo, (r ≠ o ∨ ∀ k : Fin N, oo ≠ outOff s₀ sob som sok k.val) →
            stt.mem r oo = s₀.mem r oo)
      (Nat.pos_iff_ne_zero.mp hB)
      ⟨hwbBase, fun r oo _ => by rw [hsrMem]⟩
      (fun i stt hlt hQ => by
        obtain ⟨st', hstep, hinv'⟩ := wbInv_step x w o sxb sxm sxk srw sob som sok N B s₀ RS s₀ hB hinj hox how i stt hlt hQ.1
        refine ⟨st', hstep, hinv', ?_⟩
        intro r oo hcond
        obtain ⟨hpidsQ, hiBQ, hpbQ, hpmQ, hbNQ, homQ, hrsQ, hrxQ, hrwQ, hmemQ⟩ := hQ.1
        rw [rms_wbBody_step_frame x w o sxb sxm sxk srw sob som sok N B s₀ stt st' RS i
          hiBQ hpidsQ hrxQ hrwQ hpbQ hpmQ hbNQ homQ hrsQ hstep r oo hcond]
        exact hQ.2 r oo hcond)
  obtain ⟨⟨_, _, _, _, _, _, _, _, _, hreadW⟩, hframeW⟩ := hQW
  -- assemble the `execR` run through the cast-free collapses
  have hVarR : stepStmtR R (Stmt.forRange "block_n_start_idx" 0 N B (varBody x sxk N B)) sp = some sv := by
    rw [stepStmtR_forRange,
      stepForRangeAuxR_castFree R _ (rmsVarBody_castFree R x sxk N B) "block_n_start_idx",
      ← stepForRangeAux.forRange_unfold]
    exact hsv
  have hWbR : stepStmtR R (Stmt.forRange "block_n_start_idx" 0 N B
      (wbBody x w o sxb sxm sxk srw sob som sok N B)) sr = some sw := by
    rw [stepStmtR_forRange,
      stepForRangeAuxR_castFree R _ (rmsWbBody_castFree R x w o sxb sxm sxk srw sob som sok N B) "block_n_start_idx",
      ← stepForRangeAux.forRange_unfold]
    exact hsw
  refine ⟨sw, ?_, ?_, ?_⟩
  · show execR R (rmsnorm_triton x w o sxb sxm sxk srw sob som sok N B eps).toAlgKernel s₀ = some sw
    unfold execR
    rw [body_decomp x w o sxb sxm sxk srw sob som sok N B eps]
    simp only [List.append_assoc, List.singleton_append, List.cons_append,
      List.nil_append]
    rw [stepStmtsR_append R (prefixStmts x sxb sxm B) _ s₀,
      rmsPrefix_castFree R x sxb sxm B s₀, hsp, Option.bind_some,
      stepStmtsR_cons_some hVarR,
      stepStmtsR_append R (postStmts N B eps)
        [Stmt.forRange "block_n_start_idx" 0 N B (wbBody x w o sxb sxm sxk srw sob som sok N B)] sv,
      rmsPost_castFree R N B eps sv, hsr,
      Option.bind_some, stepStmtsR_cons_some hWbR, stepStmtsR_nil]
  · intro k
    rw [hreadW k, if_pos (lt_of_lt_of_le k.isLt hNlew)]
    rfl
  · exact hframeW

/-! ### The headline -/

set_option maxHeartbeats 4000000 in
/-- **The `⊨[R]` streaming headline (wave-5 S3 per-step emit genre).** For
every rounding model `R`, the faithful `rmsnorm_triton` surface implements,
on its `StreamEmitMasked2DKernelIO₂` signature, the **ideal ℝ two-pass
RMS-norm** over the streamed tiles: emitted window `(t, j)` holds
`x[t,j] · (√(Σ_guarded x²/N_SIZE + eps))⁻¹ · w[t,j]`, where the guarded
double sum folds the *entire* `x` stream — the spec `f` is exact real
arithmetic. The kernel has **zero rounding events** (loads, both passes'
arithmetic and the per-step stores are all at `.real`; the erased
`.to(tl.float32)`s are `.real → .real`), so the skin's boundary
quantization degenerates: the readback contract's `R.round .real` is the
identity by the model's defining `round_real`, and the `.real` in-loop
stores are exact under `execR R` — the ∀-`R` face holds via the
`RoundingModel` `.real` identity fields, not as a `.triv` special case.

Layer map: both passes are cast-free, so under `execR R` they collapse
verbatim onto the exact stepper and the proven
`preLoop` / `varInv` / `postStep` / `wbInv` invariant stack above is reused
unchanged; the `⊨[R]` face adds the `TraceSafeR` walk, the per-cell memory
frame (`rms_wbBody_step_frame`, the `mem` twin of `wbStep`), and the
stream-lane spec bridge (`sum_sq_mean` re-blocking `k ↔ (k/B, k%B)`).

All four hypotheses are truth-forced (exactly the exact headline
`rmsnorm_full_output_summary`'s side conditions):

* `hB : 0 < BLOCK_N_SIZE` — both loops step by `BLOCK_N_SIZE`
  (`range(0, N_SIZE, BLOCK_N_SIZE)`); at `BLOCK_N_SIZE = 0` and
  `N_SIZE > 0` neither `forRange` advances, `execR` cannot terminate the
  way the invariant stack requires, and the step index `i / B` is
  meaningless. It holds for every real launch.
* `hox : o ≠ x`, `how : o ≠ w` — pass 2 stores into `out` **between** its
  re-reads of `x` and `rms_w`; if `out` aliased either input, later blocks
  would re-read already-overwritten values and the closed form would be
  false.
* `hsok : 0 < stride_out_k` — the per-lane write window
  `pid₀·sob + pid₁·som + k·sok` is injective over global lanes only when
  the output column stride is nonzero (`outOff_injective_of_pos`); with
  `sok = 0` all lanes collide and the per-lane readback would be
  last-writer-wins. Always true for a real tensor.

Relation to the exact surface: the exact headline
`rmsnorm_full_output_summary` (`Realizes_without_Rounding`) above is
retained unchanged; this `⊨[R]` face restates the same RMS-norm content on
the streaming emit skin, for every `R` at once (at the `.real` grid the two
faces carry the same exact cell). Both faces are kept per the
rounding-as-default doctrine. -/
specification rmsnorm_full_io_correctness (R : RoundingModel)
    (x w o : RegionName) (sxb sxm sxk srw sob som sok N B : Nat) (eps : ℝ)
    (hB : 0 < B) (hox : o ≠ x) (how : o ≠ w) (hsok : 0 < sok) :
    rmsnormKernelIO x w o sxb sxm sxk srw sob som sok N B eps ⊨[R]
      fun _ _ xs ws t j => rmsStreamSpec N B eps xs ws t j := by
  refine StreamEmitMasked2DKernelIO₂.ImplementsR.intro _ ?_ ?_ ?_
  · exact rmsnorm_triton_flattenOk x w o sxb sxm sxk srw sob som sok N B eps
  · -- safety walk
    intro bounds s xs ws _hx _hw hbr1 hbr2 hbw
    simp only [rmsnormKernelIO] at hbr1 hbr2 hbw ⊢
    exact rmsnorm_traceSafeR R bounds x w o sxb sxm sxk srw sob som sok N B eps
      hB hox how hsok s hbr1 hbr2 hbw
  · -- the rounded Hoare triple
    intro s₀ xs ws _hundef hx hw
    simp only [rmsnormKernelIO] at hx hw ⊢
    obtain ⟨sfin, hexec, hval, hframe⟩ :=
      rmsnorm_runR R x w o sxb sxm sxk srw sob som sok N B eps hB hox how hsok s₀
    refine ⟨sfin, hexec, ?_, ?_⟩
    · intro t j hj
      have hk : t.val * B + j.val < N := hj
      have hval' : sfin.readMem o (s₀.pids 0 * sob + s₀.pids 1 * som + (t.val * B + j.val) * sok)
          = rmsSpecFull s₀ x w sxb sxm sxk srw N eps (t.val * B + j.val) :=
        hval ⟨t.val * B + j.val, hk⟩
      rw [BlockState.readMemAs_real, hval',
        ← rmsStreamSpec_eq_rmsSpecFull x w s₀ sxb sxm sxk srw N B eps hB xs ws hx hw t j hj]
      simp [FloatDType.ofReal]
    · intro r oo hcond
      refine hframe r oo ?_
      rcases hcond with hne | hno
      · exact Or.inl hne
      · refine Or.inr fun k => ?_
        have hdm : k.val / B * B + k.val % B = k.val := by
          rw [Nat.mul_comm]
          exact Nat.div_add_mod k.val B
        have h := hno ⟨k.val / B, rmsStep_lt_numSteps N B k.val hB k.isLt⟩
          ⟨k.val % B, Nat.mod_lt _ hB⟩ (by simp [hdm, k.isLt])
        simpa [hdm, outOff] using h

end IOFace

end VeriTile.Bench.TritonBenchG.RmsnormTriton
