import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Triton.LoopInvariant
import VeriTile.Triton.Math.Attention

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

There are two verified results:

```
Full.Final2.rmsnorm_full_output_summary       ← FULL multi-block TOP THEOREM
  ├─ (toAlgorithm? = Except.ok _)             surface lowers to the algorithm layer
  └─ Full.Final2.rmsnorm_full_compute_correct ← ComputeCorrect over the masked store
       └─ Full.Final2.rmsnorm_full_correct    ← per global lane k < N_SIZE readback,
                                                 GENERAL ⌈N_SIZE/BLOCK_N_SIZE⌉-iteration loops

rmsnorm_triton_output_summary                 ← one-block specialization (kept)
  └─ rmsnorm_triton_compute_correct
       └─ rmsnorm_triton_correct              (launch shape N_SIZE ≤ BLOCK_N_SIZE)
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

The **one-block** spec (kept intact) threads `rmsVarCarrier`
(`reduceSum (x*x) / N_SIZE`) and `rmsInvCarrier` (`rsqrt (var + eps)`) into
`rmsnormSpec = (x * inv) * w` lane-wise. In-bounds lanes hold `rmsnormSpec`,
out-of-bounds lanes are preserved.

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); `@triton.autotune` is not
modeled. The `.to(tl.float32)` casts on the loaded `x` reduce to the identity at
the algorithm layer (post-erasure all dtypes unify to `ℝ`);
`tl.extra.cuda.libdevice.pow(·, 2)` is modeled as `x*x` and `tl.math.rsqrt` as
exact `WithBot.realRsqrt`. The **full** theorem holds for arbitrary `N_SIZE` and
`BLOCK_N_SIZE` (`0 < BLOCK_N_SIZE`, `0 < N_SIZE`), with the tiled loops running
`⌈N_SIZE/BLOCK_N_SIZE⌉` times; masked lanes load `0` (matching `other=0.0`). It
assumes the output region is distinct from the input regions (`o ≠ x`, `o ≠ w`)
and that the per-program output offsets over global lanes are injective (`hinj`).
The one-block theorem is the specialization to the wrapper launch shape
(`N_SIZE ≤ BLOCK_N_SIZE`, here `N_SIZE = K`, `BLOCK_N_SIZE = 1024`).
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

def xOffset
    (s : BlockState) (stride_x_batch stride_x_m stride_x_k : Nat)
    (i : Fin BLOCK_N_SIZE) : Nat :=
  s.pids 0 * stride_x_batch + s.pids 1 * stride_x_m + i.val * stride_x_k

def wOffset (stride_rms_w : Nat) (i : Fin BLOCK_N_SIZE) : Nat :=
  i.val * stride_rms_w

def outOffset
    (s : BlockState) (stride_out_batch stride_out_m stride_out_k : Nat)
    (i : Fin BLOCK_N_SIZE) : Nat :=
  s.pids 0 * stride_out_batch + s.pids 1 * stride_out_m + i.val * stride_out_k

noncomputable def rmsInputTile
    (s : BlockState) (x_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k N_SIZE BLOCK_N_SIZE : Nat) :
    Tile .real [BLOCK_N_SIZE] :=
  { data := fun idx =>
      if idx.1.val < N_SIZE then
        some (s.readMem x_ptr
          (xOffset s stride_x_batch stride_x_m stride_x_k idx.1))
      else some (0.0 : ℝ) }

noncomputable def rmsVarCarrier
    (s : BlockState) (x_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k N_SIZE BLOCK_N_SIZE : Nat) :
    WithBot ℝ :=
  Option.map₂ (fun a n => a / n)
    ((Tile.reduceSum (shape := [BLOCK_N_SIZE]) ⟨0, by simp⟩ Bool.false
      (Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
        (rmsInputTile s x_ptr stride_x_batch stride_x_m stride_x_k
          N_SIZE BLOCK_N_SIZE)
        (rmsInputTile s x_ptr stride_x_batch stride_x_m stride_x_k
          N_SIZE BLOCK_N_SIZE))).data PUnit.unit)
    ((Tile.scalar (dtype := .real) (some (N_SIZE : ℝ) : WithBot ℝ)).data PUnit.unit)

noncomputable def rmsInvCarrier
    (s : BlockState) (x_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k N_SIZE BLOCK_N_SIZE : Nat)
    (eps : ℝ) : WithBot ℝ :=
  WithBot.realRsqrt
    (Option.map (fun a => a + eps)
      (rmsVarCarrier s x_ptr stride_x_batch stride_x_m stride_x_k
        N_SIZE BLOCK_N_SIZE))

noncomputable def rmsnormSpec
    (s : BlockState) (x_ptr rms_w_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k stride_rms_w N_SIZE BLOCK_N_SIZE : Nat)
    (eps : ℝ) (i : Fin BLOCK_N_SIZE) : ℝ :=
  WithBot.unbotD 0
    (Option.map₂ (fun scaled w => scaled * w)
      (Option.map₂ (fun x inv => x * inv)
        (some (s.readMem x_ptr
          (xOffset s stride_x_batch stride_x_m stride_x_k i)))
        (rmsInvCarrier s x_ptr stride_x_batch stride_x_m stride_x_k
          N_SIZE BLOCK_N_SIZE eps))
      (some (s.readMem rms_w_ptr (wOffset stride_rms_w i))))

/-- Algorithm-layer correctness under the wrapper's one-block launch shape
(`N_SIZE <= BLOCK_N_SIZE`). -/
theorem rmsnorm_triton_correct
    (x_ptr rms_w_ptr output_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE : Nat)
    (eps : ℝ) (s s' : BlockState)
    (hNpos : 0 < N_SIZE) (hNle : N_SIZE ≤ BLOCK_N_SIZE)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N_SIZE =>
        outOffset s stride_out_batch stride_out_m stride_out_k i))
    (hExec : exec (rmsnorm_triton x_ptr rms_w_ptr output_ptr
        stride_x_batch stride_x_m stride_x_k stride_rms_w
        stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE eps) s =
        some s') :
    ∀ i : Fin BLOCK_N_SIZE,
      s'.readMem output_ptr
          (outOffset s stride_out_batch stride_out_m stride_out_k i) =
        if i.val < N_SIZE then
          rmsnormSpec s x_ptr rms_w_ptr stride_x_batch stride_x_m stride_x_k
            stride_rms_w N_SIZE BLOCK_N_SIZE eps i
        else s.readMem output_ptr
          (outOffset s stride_out_batch stride_out_m stride_out_k i) := by
  intro i
  have h_inj : Function.Injective
      (fun idx : TileIndex [BLOCK_N_SIZE] =>
        s.pids 0 * stride_out_batch + s.pids 1 * stride_out_m +
          idx.1.val * stride_out_k) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [outOffset] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  by_cases hB : 0 < BLOCK_N_SIZE
  · have hStep : BLOCK_N_SIZE ≠ 0 := Nat.ne_of_gt hB
    simp [exec, rmsnorm_triton, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
          stepForRangeAux.step_lt, stepForRangeAux.step_ge,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.reduceSum,
          Tile.reduceSumDrop, Tile.select, TileShape.axisDim, TileShape.eraseAxis,
          TileShape.insertAxisIndex, NumericDType.add, NumericDType.mul,
          NumericDType.div, ComparableDType.lt, hNpos, hNle,
          Nat.not_lt.mpr hNle, hStep] at hExec
    subst s'
    simp only [outOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj (i, PUnit.unit)]
    by_cases hi : i.val < N_SIZE
    · simp only [hi, ↓reduceIte]
      simp [hi, rmsnormSpec, rmsInvCarrier, rmsVarCarrier, rmsInputTile,
            xOffset, wOffset, Tile.reduceSum, Tile.reduceSumDrop, Tile.select,
            TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
            WithBot.realRsqrt, NumericDType.mul, FloatDType.cast]
      rfl
    · simp [hi]
  · exact False.elim (hB (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing correctness under the wrapper's one-block launch shape
(`N_SIZE <= BLOCK_N_SIZE`). -/
theorem rmsnorm_triton_compute_correct
    (x_ptr rms_w_ptr output_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE : Nat)
    (eps : ℝ) (s : BlockState)
    (hNpos : 0 < N_SIZE) (hNle : N_SIZE ≤ BLOCK_N_SIZE)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N_SIZE =>
        outOffset s stride_out_batch stride_out_m stride_out_k i)) :
    ComputeCorrect.Realizes
      (kernel := rmsnorm_triton x_ptr rms_w_ptr output_ptr
        stride_x_batch stride_x_m stride_x_k stride_rms_w
        stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE eps)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N_SIZE => i.val < N_SIZE)
        (fun i => (output_ptr,
          outOffset s stride_out_batch stride_out_m stride_out_k i)))
      (expected := fun i =>
        rmsnormSpec s x_ptr rms_w_ptr stride_x_batch stride_x_m stride_x_k
          stride_rms_w N_SIZE BLOCK_N_SIZE eps i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [rmsnorm_triton, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := rmsnorm_triton_correct x_ptr rms_w_ptr output_ptr
    stride_x_batch stride_x_m stride_x_k stride_rms_w
    stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE eps
    s s' hNpos hNle hOutInj hExec i
  simpa [hActive] using h

/-- Per-kernel output summary for `rmsnorm_triton`: the DSL surface lowers to the
algorithm layer, and the masked store to `output_ptr` is compute-correct under
the wrapper's one-block launch shape (`N_SIZE ≤ BLOCK_N_SIZE`) — every in-bounds
lane holds `rmsnormSpec`, out-of-bounds lanes are preserved. -/
theorem rmsnorm_triton_output_summary
    (x_ptr rms_w_ptr output_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE : Nat)
    (eps : ℝ) (s : BlockState)
    (hNpos : 0 < N_SIZE) (hNle : N_SIZE ≤ BLOCK_N_SIZE)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N_SIZE =>
        outOffset s stride_out_batch stride_out_m stride_out_k i)) :
    (∃ alg, (rmsnorm_triton x_ptr rms_w_ptr output_ptr
        stride_x_batch stride_x_m stride_x_k stride_rms_w
        stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE eps).toAlgorithm? =
        Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := rmsnorm_triton x_ptr rms_w_ptr output_ptr
        stride_x_batch stride_x_m stride_x_k stride_rms_w
        stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE eps)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N_SIZE => i.val < N_SIZE)
        (fun i => (output_ptr,
          outOffset s stride_out_batch stride_out_m stride_out_k i)))
      (expected := fun i =>
        rmsnormSpec s x_ptr rms_w_ptr stride_x_batch stride_x_m stride_x_k
          stride_rms_w N_SIZE BLOCK_N_SIZE eps i) := by
  refine ⟨?_, ?_⟩
  · simp [rmsnorm_triton, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  · exact rmsnorm_triton_compute_correct x_ptr rms_w_ptr output_ptr
      stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE eps
      s hNpos hNle hOutInj


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

noncomputable def wbSpec (s0 : BlockState) (x w : RegionName) (sxb sxm sxk srw N : Nat) (rstd : ℝ) (k : Nat) : ℝ :=
  s0.readMem x (ScratchRms.xOff s0 sxb sxm sxk k) * rstd * s0.readMem w (k*srw)

theorem foldl_store_preserve {α : Type} {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → ℝ) (mask : α → Bool) (o : Nat) (l : List α)
    (s : BlockState) (hnot : ∀ k ∈ l, mask k → offsetFn k ≠ o) :
    (l.foldl (fun acc k => if mask k then acc.writeMem region (offsetFn k) (valueFn k) else acc) s).readMem region o
      = s.readMem region o := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
    rw [List.foldl_cons]
    cases hm : mask hd
    · simp only [hm, Bool.false_eq_true, if_false]
      exact ih _ (fun k hk hmk => hnot k (List.mem_cons_of_mem hd hk) hmk)
    · simp only [hm, if_true]
      rw [ih _ (fun k hk hmk => hnot k (List.mem_cons_of_mem hd hk) hmk)]
      exact BlockState.writeMem_readMem_of_ne_offset s region (offsetFn hd) (valueFn hd) region o
        (hnot hd (List.mem_cons_self) (by rw [hm])).symm

end Wb

namespace Wb2
open Wb ScratchRms

-- readback of a one-block masked store at outOff offsets, evaluated at global lane k
-- store writes value V[j] at offset (offsets j = outOff (c*B+j)) when mask (c*B+j < N)
-- generic: readback at offset O of an "active-masked" store-foldl, where active lane writes
-- value vfn at offset ofn. If some active member writes O, get its value; else preserved.
-- Active members writing O are unique (offsets distinct among active members at O).
theorem foldl_store_at {α : Type} {region : RegionName}
    (ofn : α → Nat) (vfn : α → ℝ) (mask : α → Bool) (O : Nat) (l : List α)
    (s : BlockState) (a : α) (ha : a ∈ l) (hma : mask a) (hoa : ofn a = O)
    (huniq : ∀ b ∈ l, mask b → ofn b = O → b = a)
    (hnodup : l.Nodup) :
    (l.foldl (fun acc k => if mask k then acc.writeMem region (ofn k) (vfn k) else acc) s).readMem region O
      = vfn a := by
  obtain ⟨l₁, l₂, hl⟩ := List.append_of_mem ha
  subst hl
  rw [List.foldl_append, List.foldl_cons]
  rw [List.nodup_append, List.nodup_cons] at hnodup
  obtain ⟨hnd1, ⟨ha_notin2, hnd2⟩, hdisj⟩ := hnodup
  have h2 : ∀ b ∈ l₂, mask b → ofn b ≠ O := by
    intro b hb hmb heq
    have : b = a := huniq b (by simp [List.mem_append, hb]) hmb heq
    exact ha_notin2 (this ▸ hb)
  rw [foldl_store_preserve ofn vfn mask O l₂ _ (fun b hb hmb => h2 b hb hmb)]
  simp only [hma, if_true]
  rw [hoa]
  rw [BlockState.writeMem_readMem]
  have h1 : ∀ b ∈ l₁, mask b → ofn b ≠ O := by
    intro b hb hmb heq
    have hb' : b = a := huniq b (by simp [List.mem_append, hb]) hmb heq
    exact (hdisj b hb a (List.mem_cons_self)) hb'
  rw [foldl_store_preserve ofn vfn mask O l₁ _ (fun b hb hmb => h1 b hb hmb)]
  simp

end Wb2

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

theorem blockDiv (BLOCK a b : Nat) (hBLOCK : 0 < BLOCK) (hb : b < BLOCK) :
    (a * BLOCK + b) / BLOCK = a := by
  rw [Nat.mul_comm, Nat.mul_add_div hBLOCK, Nat.div_eq_of_lt hb, Nat.add_zero]

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

theorem withBot_sum_some {B : Nat} (g : Fin B → ℝ) :
    @Finset.sum (Fin B) (WithBot ℝ) _ Finset.univ (fun k => (some (g k) : WithBot ℝ))
      = some (Finset.univ.sum g) := by
  show (Finset.univ.sum fun k => ((g k : ℝ) : WithBot ℝ)) = ((Finset.univ.sum g : ℝ) : WithBot ℝ)
  exact (WithBot.coe_sum Finset.univ g).symm

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


theorem foldl_store_regs {α : Type} {region : RegionName}
    (ofn : α → Nat) (vfn : α → ℝ) (mask : α → Bool) (l : List α) (s : BlockState)
    (dtype : TileDType) (shape : TileShape) (name : RegName) :
    (l.foldl (fun acc k => if mask k then acc.writeMem region (ofn k) (vfn k) else acc) s).regs dtype shape name
      = s.regs dtype shape name := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih => rw [List.foldl_cons]; cases mask hd <;> simp [ih]

theorem foldl_store_pids {α : Type} {region : RegionName}
    (ofn : α → Nat) (vfn : α → ℝ) (mask : α → Bool) (l : List α) (s : BlockState) :
    (l.foldl (fun acc k => if mask k then acc.writeMem region (ofn k) (vfn k) else acc) s).pids = s.pids := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih => rw [List.foldl_cons]; cases mask hd <;> simp [ih]

theorem foldl_store_other_region {α : Type} {region : RegionName}
    (ofn : α → Nat) (vfn : α → ℝ) (mask : α → Bool) (l : List α) (s : BlockState)
    (r : RegionName) (ofs : Nat) (hr : r ≠ region) :
    (l.foldl (fun acc k => if mask k then acc.writeMem region (ofn k) (vfn k) else acc) s).readMem r ofs
      = s.readMem r ofs := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
    rw [List.foldl_cons]; cases mask hd
    · simp only [Bool.false_eq_true, if_false]; exact ih _
    · simp only [if_true]; rw [ih]; exact BlockState.writeMem_readMem_of_ne_region s region (ofn hd) (vfn hd) r ofs hr

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
  have hsFpids : sF.pids = s0.pids := by rw [hsF, StoreLemma.foldl_store_pids]; exact hs7pids
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
  · rw [hsF, StoreLemma.foldl_store_regs]; exact hpb7
  · rw [hsF, StoreLemma.foldl_store_regs]; exact hpm7
  · rw [hsF, StoreLemma.foldl_store_regs]; exact hbN7
  · rw [hsF, StoreLemma.foldl_store_regs]; exact hom7
  · rw [hsF, StoreLemma.foldl_store_regs]; exact hrs7
  · funext ofs; rw [hsF, StoreLemma.foldl_store_other_region _ _ _ _ _ _ _ (Ne.symm hox), hs7rmAll]
  · funext ofs; rw [hsF, StoreLemma.foldl_store_other_region _ _ _ _ _ _ _ (Ne.symm how), hs7rmAll]
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
      rw [Wb2.foldl_store_at
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
      rw [Wb.foldl_store_preserve
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

noncomputable def rstdVal (s : BlockState) (x : RegionName) (sxb sxm sxk N : Nat) (eps : ℝ) : ℝ :=
  WithBot.unbotD 0 (WithBot.realRsqrt (some (meanSq s x sxb sxm sxk N + eps)))

-- full spec at global lane k
noncomputable def rmsSpecFull (s : BlockState) (x w : RegionName) (sxb sxm sxk srw N : Nat) (eps : ℝ) (k : Nat) : ℝ :=
  s.readMem x (xOff s sxb sxm sxk k) * (WithBot.unbotD 0 (WithBot.realRsqrt (some (meanSq s x sxb sxm sxk N + eps)))) * s.readMem w (k*srw)

theorem rmsnorm_full_correct
    (x w o : RegionName) (sxb sxm sxk srw sob som sok N B : Nat) (eps : ℝ)
    (s s' : BlockState) (hB : 0 < B) (hNpos : 0 < N)
    (hox : o ≠ x) (how : o ≠ w)
    (hinj : Function.Injective (fun k : Fin N => outOff s sob som sok k.val))
    (hExec : exec (VeriTile.Bench.TritonBenchG.RmsnormTriton.rmsnorm_triton x w o sxb sxm sxk srw sob som sok N B eps) s = some s') :
    ∀ k : Fin N, s'.readMem o (outOff s sob som sok k.val)
      = rmsSpecFull s x w sxb sxm sxk srw N eps k.val := by
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
    (hinj : Function.Injective (fun k : Fin N => outOff s sob som sok k.val)) :
    ComputeCorrect.Realizes
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
  exact rmsnorm_full_correct x w o sxb sxm sxk srw sob som sok N B eps s s' hB hNpos hox how hinj hExec k

/-- **Full output summary**: the surface lowers to the algorithm layer, and the
masked store realizes the genuine multi-block RMS-norm closed form at every
global lane `k < N_SIZE`. The general counterpart of the one-block
`rmsnorm_triton_output_summary` — no `N_SIZE ≤ BLOCK_N_SIZE` hypothesis. -/
theorem rmsnorm_full_output_summary
    (x w o : RegionName) (sxb sxm sxk srw sob som sok N B : Nat) (eps : ℝ)
    (s : BlockState) (hB : 0 < B) (hNpos : 0 < N)
    (hox : o ≠ x) (how : o ≠ w)
    (hinj : Function.Injective (fun k : Fin N => outOff s sob som sok k.val)) :
    (∃ alg, (VeriTile.Bench.TritonBenchG.RmsnormTriton.rmsnorm_triton x w o
        sxb sxm sxk srw sob som sok N B eps).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := VeriTile.Bench.TritonBenchG.RmsnormTriton.rmsnorm_triton x w o
        sxb sxm sxk srw sob som sok N B eps)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin N => True)
        (fun k => (o, outOff s sob som sok k.val)))
      (expected := fun k : Fin N => rmsSpecFull s x w sxb sxm sxk srw N eps k.val) := by
  refine ⟨?_, rmsnorm_full_compute_correct x w o sxb sxm sxk srw sob som sok N B eps s hB hNpos hox how hinj⟩
  simp [VeriTile.Bench.TritonBenchG.RmsnormTriton.rmsnorm_triton,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

end Final2

end Full

end VeriTile.Bench.TritonBenchG.RmsnormTriton
