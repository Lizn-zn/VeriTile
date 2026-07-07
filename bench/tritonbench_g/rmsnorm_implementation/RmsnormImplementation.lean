import VeriTile.Triton

/-!
# `rmsnorm_implementation` — strict per-kernel correctness

`rmsnorm_triton` is a fused RMSNorm forward over a 3D tensor: each program
`(pid_batch, pid_m)` normalizes one row by its root-mean-square, scales by the
per-column RMS weights, and writes `out = (x / sqrt(mean(x²) + eps)) * w`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body, for one program. The host launch (`rmsnorm_triton[(batch, M,)](...)`, the
2D grid over `(batch, M)`, the host-side `BLOCK_N_SIZE = 4096` choice,
scheduling, and how the runtime composes per-program writes into one buffer) is
the *trusted boundary*, not a proof obligation here. Because the two program ids
`pid_batch = tl.program_id(0)` and `pid_m = tl.program_id(1)` are universally
quantified (via `s.pids 0` / `s.pids 1`), the per-program statement covers every
program of the 2D grid.

## Proof architecture

```
rmsnorm_implementation_compute_fullN_correct  ← TOP THEOREM (general N, multi-block)
  └─ rmsnorm_implementation_fullN_correct      ← algorithm-layer readback per column
       ├─ rmsnorm_implementation_staged_fullN_correct_from_preloop
       ├─ rmsVarForRange_context_of_preloop   ← var-loop forRange invariant
       └─ rmsOutForRange_fullN_of_init_stride_pos  ← output-loop forRange invariant
            └─ rmsOutLoopBody_step_output_invariant
rmsnorm_implementation_compute_correct        ← one-block slice (0 < N ≤ BLOCK_N_SIZE)
  └─ rmsnorm_implementation_correct
```

`rmsnorm_implementation_compute_fullN_correct` is the strongest result: it covers
arbitrary `N_SIZE` (var and output loops each tiled over `BLOCK_N_SIZE` and
closed by `forRange` loop invariants). The one-block `*_compute_correct` is the
single-iteration specialization. The RMSNorm row math is defined inline in this
file rather than reusing `VeriTile.Triton.Math.RMSNorm`.

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); the `.to(tl.float32)` cast
on the loaded input reduces to identity at the algorithm layer (post-erasure all
dtypes unify to `ℝ`). The variance reduction `tl.sum(var) / N_SIZE` sums over the
*padded* `BLOCK_N_SIZE` block, but out-of-range lanes are masked to `0` (load
`other=0.0`), so the sum equals the logical row length `N_SIZE`. Unlike the
reciprocal-`rstd` variants, this kernel computes `std = sqrt(meanSq + eps)` and
divides (`x / std`); the spec models this as `x * rmsInvVarFullN` where
`rmsInvVarFullN = (rmsStdFullNSpec)⁻¹` and `rmsStdFullNSpec = sqrt(meanSq + eps)`.
The affine step multiplies by the per-column weight. The fullN theorem requires
`0 < BLOCK_N_SIZE`, `0 < stride_out_k` (column-store injectivity), and
output/input region disjointness (`x_ptr ≠ out_ptr`, `rms_w_ptr ≠ out_ptr`); the
one-block slice instead assumes `0 < N_SIZE ≤ BLOCK_N_SIZE` and output
injectivity. `@triton.autotune` is not modeled.
-/

namespace VeriTile.Bench.TritonBenchG.RmsnormImplementation

open VeriTile.Triton

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

/-- Faithful transcription of `rmsnorm_implementation.py`'s `rmsnorm_triton`.

Allowed mechanical Lean-syntax-only changes:
- Python `N_SIZE: tl.constexpr` / `eps: tl.constexpr` / `BLOCK_N_SIZE: tl.constexpr`
  -> Lean `Nat` / `ℝ` parameters. -/
def rmsnorm_implementation
    (x_ptr rms_w_ptr out_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k : Nat)
    (N_SIZE BLOCK_N_SIZE : Nat) (eps : ℝ) :
    ComputeKernel := triton {
  pid_batch = tl.program_id(0)
  pid_m = tl.program_id(1)
  offset_m = pid_batch * $(stride_x_batch) + pid_m * $(stride_x_m)
  block_n_size = tl.arange(0, $(BLOCK_N_SIZE))
  var = tl.zeros([$(BLOCK_N_SIZE)], tl.float32)
  for block_n_strart_ptr in range(0, $(N_SIZE), $(BLOCK_N_SIZE)) {
    offset_n = block_n_strart_ptr + block_n_size
    x_ptr_mask = offset_n < $(N_SIZE)
    x = tl.load(x_ptr + offset_m + offset_n * $(stride_x_k), mask=x_ptr_mask, other=0.0)
    xf = (x).to(tl.float32)
    var += xf * xf
  }
  var = tl.sum(var, axis=0) / $(N_SIZE)
  std = tl.sqrt(var + $(eps))
  for block_n_strart_ptr in range(0, $(N_SIZE), $(BLOCK_N_SIZE)) {
    offset_n = block_n_strart_ptr + block_n_size
    x_ptr_mask = offset_n < $(N_SIZE)
    rms_w_offset = tl.load(rms_w_ptr + offset_n * $(stride_rms_w), mask=x_ptr_mask)
    x = tl.load(x_ptr + offset_m + offset_n * $(stride_x_k), mask=x_ptr_mask, other=0.0)
    x_new = x / std
    out = x_new * rms_w_offset
    out_offset = pid_batch * $(stride_out_batch) + pid_m * $(stride_out_m) +
      offset_n * $(stride_out_k)
    tl.store(out_ptr + out_offset, out, mask=x_ptr_mask)
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

def outColOffset
    (s : BlockState) (stride_out_batch stride_out_m stride_out_k col : Nat) : Nat :=
  s.pids 0 * stride_out_batch + s.pids 1 * stride_out_m + col * stride_out_k

theorem outColOffset_nat_injective_of_stride_out_k_pos
    (s : BlockState) (stride_out_batch stride_out_m stride_out_k : Nat)
    (hStride : 0 < stride_out_k) :
    Function.Injective
      (fun col : Nat =>
        outColOffset s stride_out_batch stride_out_m stride_out_k col) := by
  intro a b h
  unfold outColOffset at h
  have hmul : a * stride_out_k = b * stride_out_k := Nat.add_left_cancel h
  exact Nat.mul_right_cancel hStride hmul

theorem outColOffset_fin_injective_of_stride_out_k_pos
    (s : BlockState) (stride_out_batch stride_out_m stride_out_k N_SIZE : Nat)
    (hStride : 0 < stride_out_k) :
    Function.Injective
      (fun i : Fin N_SIZE =>
        outColOffset s stride_out_batch stride_out_m stride_out_k i.val) := by
  intro a b h
  apply Fin.ext
  exact outColOffset_nat_injective_of_stride_out_k_pos s stride_out_batch
    stride_out_m stride_out_k hStride h

theorem outColOffset_chunk_injective_of_stride_out_k_pos
    (s : BlockState)
    (stride_out_batch stride_out_m stride_out_k BLOCK_N_SIZE off : Nat)
    (hStride : 0 < stride_out_k) :
    Function.Injective
      (fun idx : TileIndex [BLOCK_N_SIZE] =>
        outColOffset s stride_out_batch stride_out_m stride_out_k
          (off + idx.1.val)) := by
  intro a b h
  have hCol :
      off + a.1.val = off + b.1.val :=
    outColOffset_nat_injective_of_stride_out_k_pos s stride_out_batch
      stride_out_m stride_out_k hStride h
  have hHead : a.1 = b.1 := by
    apply Fin.ext
    omega
  cases a with
  | mk aHead aTail =>
      cases b with
      | mk bHead bTail =>
          cases aTail
          cases bTail
          simp only at hHead
          cases hHead
          rfl

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
    ((Tile.scalar N_SIZE).natToReal.data PUnit.unit)

noncomputable def rmsStdCarrier
    (s : BlockState) (x_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k N_SIZE BLOCK_N_SIZE : Nat)
    (eps : ℝ) : WithBot ℝ :=
  WithBot.realSqrt
    (Option.map (fun a => a + eps)
      (rmsVarCarrier s x_ptr stride_x_batch stride_x_m stride_x_k
        N_SIZE BLOCK_N_SIZE))

/-- Algebraic offset into `x_ptr` for an arbitrary column index `col : Nat`:
row `(batch, m) = (pids 0, pids 1)` at strides `(stride_x_batch, stride_x_m)`,
column `col` at stride `stride_x_k`. -/
def xColOffset
    (s : BlockState) (stride_x_batch stride_x_m stride_x_k col : Nat) : Nat :=
  s.pids 0 * stride_x_batch + s.pids 1 * stride_x_m + col * stride_x_k

/-- Multi-block full-N variance carrier: the algebraic ground truth for
`Σ_{j < N_SIZE} (x[j])²`, independent of any block decomposition. -/
noncomputable def rmsVarFullNCarrier
    (s : BlockState) (x_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k N_SIZE _BLOCK_N_SIZE : Nat) : ℝ :=
  ∑ j : Fin N_SIZE,
    (s.readMem x_ptr
        (xColOffset s stride_x_batch stride_x_m stride_x_k j.val))^2

/-- Multi-block full-N reciprocal-standard-deviation:
`1 / sqrt(Σ x_j² / N_SIZE + eps)`. -/
noncomputable def rmsInvVarFullN
    (s : BlockState) (x_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k N_SIZE BLOCK_N_SIZE : Nat)
    (eps : ℝ) : ℝ :=
  1 / Real.sqrt
    (rmsVarFullNCarrier s x_ptr stride_x_batch stride_x_m stride_x_k
      N_SIZE BLOCK_N_SIZE / (N_SIZE : ℝ) + eps)

/-- Multi-block full-N output spec: `x[i] * rmsInvVarFullN` for each
`i < N_SIZE`, expressed against the algebraic ground truth. -/
noncomputable def rmsnormYFullNSpec
    (s : BlockState) (x_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k N_SIZE BLOCK_N_SIZE : Nat)
    (eps : ℝ) (i : Fin N_SIZE) : ℝ :=
  s.readMem x_ptr
      (s.pids 0 * stride_x_batch + s.pids 1 * stride_x_m + i.val * stride_x_k) *
    rmsInvVarFullN s x_ptr stride_x_batch stride_x_m stride_x_k
      N_SIZE BLOCK_N_SIZE eps

/-- Full-N RMSNorm output including the learned RMS weight. This is the
Python-observable value written by `rmsnorm_triton` for column `i`. -/
noncomputable def rmsnormWeightedYFullNSpec
    (s : BlockState) (x_ptr rms_w_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k stride_rms_w N_SIZE BLOCK_N_SIZE : Nat)
    (eps : ℝ) (i : Fin N_SIZE) : ℝ :=
  rmsnormYFullNSpec s x_ptr stride_x_batch stride_x_m stride_x_k
      N_SIZE BLOCK_N_SIZE eps i *
    s.readMem rms_w_ptr (i.val * stride_rms_w)

noncomputable def rmsStdFullNSpec
    (s : BlockState) (x_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k N_SIZE BLOCK_N_SIZE : Nat)
    (eps : ℝ) : ℝ :=
  Real.sqrt
    (rmsVarFullNCarrier s x_ptr stride_x_batch stride_x_m stride_x_k
      N_SIZE BLOCK_N_SIZE / (N_SIZE : ℝ) + eps)

def rmsnormOutLoopInvariant
    (s0 : BlockState) (x_ptr rms_w_ptr out_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE : Nat)
    (eps : ℝ) (off : Nat) (st : BlockState) : Prop :=
  ∀ i : Fin N_SIZE,
    i.val < off →
      st.readMem out_ptr
          (outColOffset s0 stride_out_batch stride_out_m stride_out_k i.val) =
        rmsnormWeightedYFullNSpec s0 x_ptr rms_w_ptr stride_x_batch stride_x_m
          stride_x_k stride_rms_w N_SIZE BLOCK_N_SIZE eps i

theorem rmsnormOutLoopInvariant_zero
    (s0 st : BlockState) (x_ptr rms_w_ptr out_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE : Nat)
    (eps : ℝ) :
    rmsnormOutLoopInvariant s0 x_ptr rms_w_ptr out_ptr
      stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE eps 0 st := by
  intro i hlt
  omega

/-- The output-loop body statements as expanded by the DSL. -/
def rmsOutLoopBody
    (x_ptr rms_w_ptr out_ptr : RegionName)
    (stride_x_k stride_rms_w stride_out_batch stride_out_m stride_out_k
      N_SIZE BLOCK_N_SIZE : Nat) :
    List Stmt :=
  [ .assign .nat [BLOCK_N_SIZE] "offset_n"
      (.add NumericDType.nat Broadcast.scalarL
        (.ref .nat [] "block_n_strart_ptr")
        (.ref .nat [BLOCK_N_SIZE] "block_n_size"))
  , .assign .bool [BLOCK_N_SIZE] "x_ptr_mask"
      (.lt ComparableDType.nat Broadcast.scalarR
        (.ref .nat [BLOCK_N_SIZE] "offset_n") (.constNat N_SIZE))
  , .assign .real [BLOCK_N_SIZE] "rms_w_offset"
      (.load .real
        (.region rms_w_ptr
          (.mul NumericDType.nat Broadcast.scalarR
            (.ref .nat [BLOCK_N_SIZE] "offset_n") (.constNat stride_rms_w)))
        (.mask (.ref .bool [BLOCK_N_SIZE] "x_ptr_mask")))
  , .assign .real [BLOCK_N_SIZE] "x"
      (.load .real
        (.region x_ptr
          (.add NumericDType.nat Broadcast.scalarL
            (.ref .nat [] "offset_m")
            (.mul NumericDType.nat Broadcast.scalarR
              (.ref .nat [BLOCK_N_SIZE] "offset_n") (.constNat stride_x_k))))
        (.maskOther
          (.ref .bool [BLOCK_N_SIZE] "x_ptr_mask")
          ((Op.const 0.0).broadcast [BLOCK_N_SIZE])))
  , .assign .real [BLOCK_N_SIZE] "x_new"
      (.div NumericDType.real Broadcast.scalarR
        (.ref .real [BLOCK_N_SIZE] "x")
        (.ref .real [] "std"))
  , .assign .real [BLOCK_N_SIZE] "out"
      (.mul NumericDType.real Broadcast.nil.consSame
        (.ref .real [BLOCK_N_SIZE] "x_new")
        (.ref .real [BLOCK_N_SIZE] "rms_w_offset"))
  , .assign .nat [BLOCK_N_SIZE] "out_offset"
      (.add NumericDType.nat Broadcast.scalarL
        (.add NumericDType.nat Broadcast.nil
          (.mul NumericDType.nat Broadcast.nil
            (.ref .nat [] "pid_batch") (.constNat stride_out_batch))
          (.mul NumericDType.nat Broadcast.nil
            (.ref .nat [] "pid_m") (.constNat stride_out_m)))
        (.mul NumericDType.nat Broadcast.scalarR
          (.ref .nat [BLOCK_N_SIZE] "offset_n") (.constNat stride_out_k)))
  , .store .real [BLOCK_N_SIZE]
      (.region out_ptr (.ref .nat [BLOCK_N_SIZE] "out_offset"))
      (.ref .real [BLOCK_N_SIZE] "out")
      (.mask (.ref .bool [BLOCK_N_SIZE] "x_ptr_mask"))
  ]

theorem rmsOutLoopBody_step_write_current
    (s0 st st' : BlockState) (x_ptr rms_w_ptr out_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE off : Nat)
    (eps : ℝ) (i : Fin BLOCK_N_SIZE)
    (hActive : off + i.val < N_SIZE)
    (hPidBatch :
      st.regs .nat [] "pid_batch" = some (Tile.scalar (s0.pids 0)))
    (hPidM :
      st.regs .nat [] "pid_m" = some (Tile.scalar (s0.pids 1)))
    (hOffsetM :
      st.regs .nat [] "offset_m" =
        some (Tile.scalar
          (s0.pids 0 * stride_x_batch + s0.pids 1 * stride_x_m)))
    (hBlockN :
      st.regs .nat [BLOCK_N_SIZE] "block_n_size" =
        some { data := fun idx : TileIndex [BLOCK_N_SIZE] => idx.1.val })
    (hStd :
      st.regs .real [] "std" =
        some (Tile.scalar
          (rmsStdFullNSpec s0 x_ptr stride_x_batch stride_x_m stride_x_k
            N_SIZE BLOCK_N_SIZE eps)))
    (hReadX : ∀ offset, st.readMem x_ptr offset = s0.readMem x_ptr offset)
    (hReadW : ∀ offset, st.readMem rms_w_ptr offset = s0.readMem rms_w_ptr offset)
    (hOutInj :
      Function.Injective
        (fun idx : TileIndex [BLOCK_N_SIZE] =>
          outColOffset s0 stride_out_batch stride_out_m stride_out_k
            (off + idx.1.val)))
    (hStep :
      stepStmts
        (rmsOutLoopBody x_ptr rms_w_ptr out_ptr stride_x_k stride_rms_w
          stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE)
        (st.setReg "block_n_strart_ptr" .nat [] (Tile.scalar off)) = some st') :
    st'.readMem out_ptr
        (outColOffset s0 stride_out_batch stride_out_m stride_out_k
          (off + i.val)) =
      rmsnormWeightedYFullNSpec s0 x_ptr rms_w_ptr stride_x_batch stride_x_m
        stride_x_k stride_rms_w N_SIZE BLOCK_N_SIZE eps ⟨off + i.val, hActive⟩ := by
  unfold rmsOutLoopBody at hStep
  simp [stepStmts, stepStmt, evalOp, evalOp.eq_def, hPidBatch, hPidM, hOffsetM, hBlockN, hStd, Tile.bop,
    Tile.cop, Tile.ptrAdd, Tile.uop, NumericDType.add, NumericDType.mul,
    NumericDType.div, ComparableDType.lt, Option.bind, FloatDType.cast,
    hReadX, hReadW, outColOffset, rmsStdFullNSpec] at hStep
  subst st'
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_N_SIZE] =>
        s0.pids 0 * stride_out_batch + s0.pids 1 * stride_out_m +
          (off + idx.1.val) * stride_out_k) := by
    simpa [outColOffset] using hOutInj
  simp only [outColOffset]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj
    ((i, PUnit.unit) : TileIndex [BLOCK_N_SIZE])]
  simp [hActive, rmsnormWeightedYFullNSpec, rmsnormYFullNSpec, rmsInvVarFullN,
    rmsStdFullNSpec, outColOffset, div_eq_mul_inv]

theorem rmsOutLoopBody_step_preserves_regs_pids
    (s0 st st' : BlockState) (x_ptr rms_w_ptr out_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE off : Nat)
    (eps : ℝ)
    (hPids : st.pids = s0.pids)
    (hPidBatch :
      st.regs .nat [] "pid_batch" = some (Tile.scalar (s0.pids 0)))
    (hPidM :
      st.regs .nat [] "pid_m" = some (Tile.scalar (s0.pids 1)))
    (hOffsetM :
      st.regs .nat [] "offset_m" =
        some (Tile.scalar
          (s0.pids 0 * stride_x_batch + s0.pids 1 * stride_x_m)))
    (hBlockN :
      st.regs .nat [BLOCK_N_SIZE] "block_n_size" =
        some { data := fun idx : TileIndex [BLOCK_N_SIZE] => idx.1.val })
    (hStd :
      st.regs .real [] "std" =
        some (Tile.scalar
          (rmsStdFullNSpec s0 x_ptr stride_x_batch stride_x_m stride_x_k
            N_SIZE BLOCK_N_SIZE eps)))
    (hReadX : ∀ offset, st.readMem x_ptr offset = s0.readMem x_ptr offset)
    (hReadW : ∀ offset, st.readMem rms_w_ptr offset = s0.readMem rms_w_ptr offset)
    (hStep :
      stepStmts
        (rmsOutLoopBody x_ptr rms_w_ptr out_ptr stride_x_k stride_rms_w
          stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE)
        (st.setReg "block_n_strart_ptr" .nat [] (Tile.scalar off)) = some st') :
    st'.pids = s0.pids ∧
      st'.regs .nat [] "pid_batch" = some (Tile.scalar (s0.pids 0)) ∧
      st'.regs .nat [] "pid_m" = some (Tile.scalar (s0.pids 1)) ∧
      st'.regs .nat [] "offset_m" =
        some (Tile.scalar
          (s0.pids 0 * stride_x_batch + s0.pids 1 * stride_x_m)) ∧
      st'.regs .nat [BLOCK_N_SIZE] "block_n_size" =
        some { data := fun idx : TileIndex [BLOCK_N_SIZE] => idx.1.val } ∧
      st'.regs .real [] "std" =
        some (Tile.scalar
          (rmsStdFullNSpec s0 x_ptr stride_x_batch stride_x_m stride_x_k
            N_SIZE BLOCK_N_SIZE eps)) := by
  unfold rmsOutLoopBody at hStep
  simp [stepStmts, stepStmt, evalOp, evalOp.eq_def, hPidBatch, hPidM, hOffsetM, hBlockN, hStd, Tile.bop,
    Tile.cop, Tile.ptrAdd, Tile.uop, NumericDType.add, NumericDType.mul,
    NumericDType.div, ComparableDType.lt, Option.bind, FloatDType.cast,
    hReadX, hReadW, outColOffset, rmsStdFullNSpec] at hStep
  subst st'
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [BlockState.foldl_writeMem_prop_masked_pids]
    simp [BlockState.setReg, hPids]
  · simp [hPidBatch]
  · simp [hPidM]
  · simp [hOffsetM]
  · simp [hBlockN]
  · simp [hStd]

theorem rmsOutLoopBody_step_preserves_reads
    (s0 st st' : BlockState) (x_ptr rms_w_ptr out_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE off : Nat)
    (eps : ℝ)
    (hPidBatch :
      st.regs .nat [] "pid_batch" = some (Tile.scalar (s0.pids 0)))
    (hPidM :
      st.regs .nat [] "pid_m" = some (Tile.scalar (s0.pids 1)))
    (hOffsetM :
      st.regs .nat [] "offset_m" =
        some (Tile.scalar
          (s0.pids 0 * stride_x_batch + s0.pids 1 * stride_x_m)))
    (hBlockN :
      st.regs .nat [BLOCK_N_SIZE] "block_n_size" =
        some { data := fun idx : TileIndex [BLOCK_N_SIZE] => idx.1.val })
    (hStd :
      st.regs .real [] "std" =
        some (Tile.scalar
          (rmsStdFullNSpec s0 x_ptr stride_x_batch stride_x_m stride_x_k
            N_SIZE BLOCK_N_SIZE eps)))
    (hReadX : ∀ offset, st.readMem x_ptr offset = s0.readMem x_ptr offset)
    (hReadW : ∀ offset, st.readMem rms_w_ptr offset = s0.readMem rms_w_ptr offset)
    (hXOutNe : x_ptr ≠ out_ptr)
    (hWOutNe : rms_w_ptr ≠ out_ptr)
    (hStep :
      stepStmts
        (rmsOutLoopBody x_ptr rms_w_ptr out_ptr stride_x_k stride_rms_w
          stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE)
        (st.setReg "block_n_strart_ptr" .nat [] (Tile.scalar off)) = some st') :
    (∀ offset, st'.readMem x_ptr offset = s0.readMem x_ptr offset) ∧
      (∀ offset, st'.readMem rms_w_ptr offset = s0.readMem rms_w_ptr offset) := by
  unfold rmsOutLoopBody at hStep
  simp [stepStmts, stepStmt, evalOp, evalOp.eq_def, hPidBatch, hPidM, hOffsetM, hBlockN, hStd, Tile.bop,
    Tile.cop, Tile.ptrAdd, Tile.uop, NumericDType.add, NumericDType.mul,
    NumericDType.div, ComparableDType.lt, Option.bind, FloatDType.cast,
    hReadX, hReadW, outColOffset, rmsStdFullNSpec] at hStep
  subst st'
  constructor
  · intro offset
    rw [BlockState.foldl_writeMem_const_region_prop_masked_readMem_other
      (region := out_ptr)
      (P := fun lane : TileIndex [BLOCK_N_SIZE] => off + lane.1.val < N_SIZE)
      (R := x_ptr) (off := offset) (hRR := hXOutNe)]
    exact hReadX offset
  · intro offset
    rw [BlockState.foldl_writeMem_const_region_prop_masked_readMem_other
      (region := out_ptr)
      (P := fun lane : TileIndex [BLOCK_N_SIZE] => off + lane.1.val < N_SIZE)
      (R := rms_w_ptr) (off := offset) (hRR := hWOutNe)]
    exact hReadW offset

theorem rmsOutLoopBody_step_preserves_old_output
    (s0 st st' : BlockState) (x_ptr rms_w_ptr out_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE off : Nat)
    (eps : ℝ) (col : Fin N_SIZE)
    (hOld : col.val < off)
    (hPidBatch :
      st.regs .nat [] "pid_batch" = some (Tile.scalar (s0.pids 0)))
    (hPidM :
      st.regs .nat [] "pid_m" = some (Tile.scalar (s0.pids 1)))
    (hOffsetM :
      st.regs .nat [] "offset_m" =
        some (Tile.scalar
          (s0.pids 0 * stride_x_batch + s0.pids 1 * stride_x_m)))
    (hBlockN :
      st.regs .nat [BLOCK_N_SIZE] "block_n_size" =
        some { data := fun idx : TileIndex [BLOCK_N_SIZE] => idx.1.val })
    (hStd :
      st.regs .real [] "std" =
        some (Tile.scalar
          (rmsStdFullNSpec s0 x_ptr stride_x_batch stride_x_m stride_x_k
            N_SIZE BLOCK_N_SIZE eps)))
    (hReadX : ∀ offset, st.readMem x_ptr offset = s0.readMem x_ptr offset)
    (hReadW : ∀ offset, st.readMem rms_w_ptr offset = s0.readMem rms_w_ptr offset)
    (hOutInj : Function.Injective
      (fun i : Fin N_SIZE =>
        outColOffset s0 stride_out_batch stride_out_m stride_out_k i.val))
    (hStep :
      stepStmts
        (rmsOutLoopBody x_ptr rms_w_ptr out_ptr stride_x_k stride_rms_w
          stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE)
        (st.setReg "block_n_strart_ptr" .nat [] (Tile.scalar off)) = some st') :
    st'.readMem out_ptr
        (outColOffset s0 stride_out_batch stride_out_m stride_out_k col.val) =
      st.readMem out_ptr
        (outColOffset s0 stride_out_batch stride_out_m stride_out_k col.val) := by
  unfold rmsOutLoopBody at hStep
  simp [stepStmts, stepStmt, evalOp, evalOp.eq_def, hPidBatch, hPidM, hOffsetM, hBlockN, hStd, Tile.bop,
    Tile.cop, Tile.ptrAdd, Tile.uop, NumericDType.add, NumericDType.mul,
    NumericDType.div, ComparableDType.lt, Option.bind, FloatDType.cast,
    hReadX, hReadW, outColOffset, rmsStdFullNSpec] at hStep
  subst st'
  simp only [outColOffset]
  rw [BlockState.scatter_prop_masked_preserves_other_offset
    (region := out_ptr)
    (P := fun lane : TileIndex [BLOCK_N_SIZE] => off + lane.1.val < N_SIZE)
    (off := s0.pids 0 * stride_out_batch + s0.pids 1 * stride_out_m +
      col.val * stride_out_k)]
  · rfl
  · intro lane hActive hEq
    have hFinEq :
        (⟨off + lane.1.val, hActive⟩ : Fin N_SIZE) = col := by
      apply hOutInj
      simpa [outColOffset] using hEq
    have hVal : off + lane.1.val = col.val := congrArg Fin.val hFinEq
    omega

theorem rmsOutLoopBody_step_output_invariant
    (s0 st st' : BlockState) (x_ptr rms_w_ptr out_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE off : Nat)
    (eps : ℝ)
    (hInv : rmsnormOutLoopInvariant s0 x_ptr rms_w_ptr out_ptr
      stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE eps off st)
    (hPidBatch :
      st.regs .nat [] "pid_batch" = some (Tile.scalar (s0.pids 0)))
    (hPidM :
      st.regs .nat [] "pid_m" = some (Tile.scalar (s0.pids 1)))
    (hOffsetM :
      st.regs .nat [] "offset_m" =
        some (Tile.scalar
          (s0.pids 0 * stride_x_batch + s0.pids 1 * stride_x_m)))
    (hBlockN :
      st.regs .nat [BLOCK_N_SIZE] "block_n_size" =
        some { data := fun idx : TileIndex [BLOCK_N_SIZE] => idx.1.val })
    (hStd :
      st.regs .real [] "std" =
        some (Tile.scalar
          (rmsStdFullNSpec s0 x_ptr stride_x_batch stride_x_m stride_x_k
            N_SIZE BLOCK_N_SIZE eps)))
    (hReadX : ∀ offset, st.readMem x_ptr offset = s0.readMem x_ptr offset)
    (hReadW : ∀ offset, st.readMem rms_w_ptr offset = s0.readMem rms_w_ptr offset)
    (hOutInj : Function.Injective
      (fun i : Fin N_SIZE =>
        outColOffset s0 stride_out_batch stride_out_m stride_out_k i.val))
    (hOutChunkInj :
      Function.Injective
        (fun idx : TileIndex [BLOCK_N_SIZE] =>
          outColOffset s0 stride_out_batch stride_out_m stride_out_k
            (off + idx.1.val)))
    (hStep :
      stepStmts
        (rmsOutLoopBody x_ptr rms_w_ptr out_ptr stride_x_k stride_rms_w
          stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE)
        (st.setReg "block_n_strart_ptr" .nat [] (Tile.scalar off)) = some st') :
    rmsnormOutLoopInvariant s0 x_ptr rms_w_ptr out_ptr
      stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE eps
      (off + BLOCK_N_SIZE) st' := by
  intro col hWritten
  by_cases hOld : col.val < off
  · rw [rmsOutLoopBody_step_preserves_old_output s0 st st' x_ptr rms_w_ptr out_ptr
      stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE off eps
      col hOld hPidBatch hPidM hOffsetM hBlockN hStd hReadX hReadW hOutInj hStep]
    exact hInv col hOld
  · have hOffLe : off ≤ col.val := Nat.le_of_not_gt hOld
    let lane : Fin BLOCK_N_SIZE := ⟨col.val - off, by omega⟩
    have hLaneActive : off + lane.val < N_SIZE := by
      have hcol : off + lane.val = col.val := by
        simp [lane]
        omega
      simp [hcol, col.isLt]
    have hcolEq : off + lane.val = col.val := by
      simp [lane]
      omega
    have hRead := rmsOutLoopBody_step_write_current s0 st st' x_ptr rms_w_ptr out_ptr
      stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE off eps
      lane hLaneActive hPidBatch hPidM hOffsetM hBlockN hStd hReadX hReadW
      hOutChunkInj hStep
    simpa [hcolEq] using hRead

def rmsOutLoopContextInvariant
    (s0 : BlockState) (x_ptr rms_w_ptr out_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE off : Nat)
    (eps : ℝ) (st : BlockState) : Prop :=
  rmsnormOutLoopInvariant s0 x_ptr rms_w_ptr out_ptr
      stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE eps off st ∧
    st.pids = s0.pids ∧
    st.regs .nat [] "pid_batch" = some (Tile.scalar (s0.pids 0)) ∧
    st.regs .nat [] "pid_m" = some (Tile.scalar (s0.pids 1)) ∧
    st.regs .nat [] "offset_m" =
      some (Tile.scalar
        (s0.pids 0 * stride_x_batch + s0.pids 1 * stride_x_m)) ∧
    st.regs .nat [BLOCK_N_SIZE] "block_n_size" =
      some { data := fun idx : TileIndex [BLOCK_N_SIZE] => idx.1.val } ∧
    st.regs .real [] "std" =
      some (Tile.scalar
        (rmsStdFullNSpec s0 x_ptr stride_x_batch stride_x_m stride_x_k
          N_SIZE BLOCK_N_SIZE eps)) ∧
    (∀ offset, st.readMem x_ptr offset = s0.readMem x_ptr offset) ∧
    (∀ offset, st.readMem rms_w_ptr offset = s0.readMem rms_w_ptr offset)

theorem rmsOutLoopContextInvariant_step_of_body
    (s0 st st' : BlockState) (x_ptr rms_w_ptr out_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE off : Nat)
    (eps : ℝ)
    (hCtx : rmsOutLoopContextInvariant s0 x_ptr rms_w_ptr out_ptr
      stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE off eps st)
    (hXOutNe : x_ptr ≠ out_ptr)
    (hWOutNe : rms_w_ptr ≠ out_ptr)
    (hOutInj : Function.Injective
      (fun i : Fin N_SIZE =>
        outColOffset s0 stride_out_batch stride_out_m stride_out_k i.val))
    (hOutChunkInj :
      Function.Injective
        (fun idx : TileIndex [BLOCK_N_SIZE] =>
          outColOffset s0 stride_out_batch stride_out_m stride_out_k
            (off + idx.1.val)))
    (hStep :
      stepStmts
        (rmsOutLoopBody x_ptr rms_w_ptr out_ptr stride_x_k stride_rms_w
          stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE)
        (st.setReg "block_n_strart_ptr" .nat [] (Tile.scalar off)) = some st') :
    rmsOutLoopContextInvariant s0 x_ptr rms_w_ptr out_ptr
      stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE
      (off + BLOCK_N_SIZE) eps st' := by
  rcases hCtx with
    ⟨hOutInv, hPids, hPidBatch, hPidM, hOffsetM, hBlockN, hStd, hReadX, hReadW⟩
  have hOutInv' :=
    rmsOutLoopBody_step_output_invariant s0 st st' x_ptr rms_w_ptr out_ptr
      stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE off eps
      hOutInv hPidBatch hPidM hOffsetM hBlockN hStd hReadX hReadW hOutInj
      hOutChunkInj hStep
  rcases rmsOutLoopBody_step_preserves_regs_pids s0 st st' x_ptr rms_w_ptr out_ptr
      stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE off eps
      hPids hPidBatch hPidM hOffsetM hBlockN hStd hReadX hReadW hStep with
    ⟨hPids', hPidBatch', hPidM', hOffsetM', hBlockN', hStd'⟩
  rcases rmsOutLoopBody_step_preserves_reads s0 st st' x_ptr rms_w_ptr out_ptr
      stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE off eps
      hPidBatch hPidM hOffsetM hBlockN hStd hReadX hReadW hXOutNe hWOutNe hStep with
    ⟨hReadX', hReadW'⟩
  exact ⟨hOutInv', hPids', hPidBatch', hPidM', hOffsetM', hBlockN', hStd', hReadX', hReadW'⟩

theorem rmsOutLoopContextInvariant_body_step_exists
    (s0 st : BlockState) (x_ptr rms_w_ptr out_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE off : Nat)
    (eps : ℝ)
    (hCtx : rmsOutLoopContextInvariant s0 x_ptr rms_w_ptr out_ptr
      stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE off eps st)
    (hXOutNe : x_ptr ≠ out_ptr)
    (hWOutNe : rms_w_ptr ≠ out_ptr)
    (hOutInj : Function.Injective
      (fun i : Fin N_SIZE =>
        outColOffset s0 stride_out_batch stride_out_m stride_out_k i.val))
    (hOutChunkInj :
      Function.Injective
        (fun idx : TileIndex [BLOCK_N_SIZE] =>
          outColOffset s0 stride_out_batch stride_out_m stride_out_k
            (off + idx.1.val))) :
    ∃ st',
      stepStmts
        (rmsOutLoopBody x_ptr rms_w_ptr out_ptr stride_x_k stride_rms_w
          stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE)
        (st.setReg "block_n_strart_ptr" .nat [] (Tile.scalar off)) = some st' ∧
      rmsOutLoopContextInvariant s0 x_ptr rms_w_ptr out_ptr
        stride_x_batch stride_x_m stride_x_k stride_rms_w
        stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE
        (off + BLOCK_N_SIZE) eps st' := by
  rcases hCtx with
    ⟨hOutInv, hPids, hPidBatch, hPidM, hOffsetM, hBlockN, hStd, hReadX, hReadW⟩
  cases hStep :
      stepStmts
        (rmsOutLoopBody x_ptr rms_w_ptr out_ptr stride_x_k stride_rms_w
          stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE)
        (st.setReg "block_n_strart_ptr" .nat [] (Tile.scalar off)) with
  | none =>
      unfold rmsOutLoopBody at hStep
      simp [stepStmts, stepStmt, evalOp, evalOp.eq_def, hPidBatch, hPidM, hOffsetM, hBlockN, hStd, Tile.bop,
        Tile.cop, Tile.ptrAdd, Tile.uop, NumericDType.add, NumericDType.mul,
        NumericDType.div, ComparableDType.lt, Option.bind, FloatDType.cast,
        hReadX, hReadW, hPids, outColOffset, rmsStdFullNSpec] at hStep
  | some st' =>
      refine ⟨st', rfl, ?_⟩
      exact rmsOutLoopContextInvariant_step_of_body s0 st st' x_ptr rms_w_ptr
        out_ptr stride_x_batch stride_x_m stride_x_k stride_rms_w
        stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE off eps
        ⟨hOutInv, hPids, hPidBatch, hPidM, hOffsetM, hBlockN, hStd, hReadX, hReadW⟩
        hXOutNe hWOutNe hOutInj hOutChunkInj hStep

theorem rmsOutForRange_context
    (s0 stInit stLoop : BlockState) (x_ptr rms_w_ptr out_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE : Nat)
    (eps : ℝ)
    (hStepNe : BLOCK_N_SIZE ≠ 0)
    (hInit : rmsOutLoopContextInvariant s0 x_ptr rms_w_ptr out_ptr
      stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE 0 eps stInit)
    (hXOutNe : x_ptr ≠ out_ptr)
    (hWOutNe : rms_w_ptr ≠ out_ptr)
    (hOutInj : Function.Injective
      (fun i : Fin N_SIZE =>
        outColOffset s0 stride_out_batch stride_out_m stride_out_k i.val))
    (hOutChunkInj :
      ∀ off,
        off < N_SIZE →
          Function.Injective
            (fun idx : TileIndex [BLOCK_N_SIZE] =>
              outColOffset s0 stride_out_batch stride_out_m stride_out_k
                (off + idx.1.val)))
    (hLoop :
      stepStmt (.forRange "block_n_strart_ptr" 0 N_SIZE BLOCK_N_SIZE
        (rmsOutLoopBody x_ptr rms_w_ptr out_ptr stride_x_k stride_rms_w
          stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE))
        stInit = some stLoop) :
    ∃ final,
      N_SIZE ≤ final ∧
        rmsOutLoopContextInvariant s0 x_ptr rms_w_ptr out_ptr
          stride_x_batch stride_x_m stride_x_k stride_rms_w
          stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE
          final eps stLoop := by
  obtain ⟨final, stFinal, hFor, hFinal, hCtx⟩ :=
    forRange_inv
      (idx := "block_n_strart_ptr") (start := 0) (stop := N_SIZE)
      (step := BLOCK_N_SIZE)
      (body := rmsOutLoopBody x_ptr rms_w_ptr out_ptr stride_x_k stride_rms_w
        stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE)
      (P := fun off st =>
        rmsOutLoopContextInvariant s0 x_ptr rms_w_ptr out_ptr
          stride_x_batch stride_x_m stride_x_k stride_rms_w
          stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE off
          eps st)
      (s_init := stInit)
      hStepNe hInit
      (by
        intro off st hlt hCtx
        exact rmsOutLoopContextInvariant_body_step_exists s0 st x_ptr
          rms_w_ptr out_ptr stride_x_batch stride_x_m stride_x_k stride_rms_w
          stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE off eps
          hCtx hXOutNe hWOutNe hOutInj (hOutChunkInj off hlt))
  have hEq : stFinal = stLoop := by
    rw [hLoop] at hFor
    injection hFor with h
    exact h.symm
  subst hEq
  exact ⟨final, hFinal, hCtx⟩

theorem rmsOutLoopContextInvariant_readout_fullN
    (s0 st : BlockState) (x_ptr rms_w_ptr out_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE final : Nat)
    (eps : ℝ)
    (hFinal : N_SIZE ≤ final)
    (hCtx : rmsOutLoopContextInvariant s0 x_ptr rms_w_ptr out_ptr
      stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE
      final eps st) :
    ∀ i : Fin N_SIZE,
      st.readMem out_ptr
          (outColOffset s0 stride_out_batch stride_out_m stride_out_k i.val) =
        rmsnormWeightedYFullNSpec s0 x_ptr rms_w_ptr stride_x_batch stride_x_m
          stride_x_k stride_rms_w N_SIZE BLOCK_N_SIZE eps i := by
  intro i
  exact hCtx.1 i (Nat.lt_of_lt_of_le i.isLt hFinal)

theorem rmsOutForRange_fullN_of_init
    (s0 stInit stLoop : BlockState) (x_ptr rms_w_ptr out_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE : Nat)
    (eps : ℝ)
    (hStepNe : BLOCK_N_SIZE ≠ 0)
    (hInit : rmsOutLoopContextInvariant s0 x_ptr rms_w_ptr out_ptr
      stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE 0 eps stInit)
    (hXOutNe : x_ptr ≠ out_ptr)
    (hWOutNe : rms_w_ptr ≠ out_ptr)
    (hOutInj : Function.Injective
      (fun i : Fin N_SIZE =>
        outColOffset s0 stride_out_batch stride_out_m stride_out_k i.val))
    (hOutChunkInj :
      ∀ off,
        off < N_SIZE →
          Function.Injective
            (fun idx : TileIndex [BLOCK_N_SIZE] =>
              outColOffset s0 stride_out_batch stride_out_m stride_out_k
                (off + idx.1.val)))
    (hLoop :
      stepStmt (.forRange "block_n_strart_ptr" 0 N_SIZE BLOCK_N_SIZE
        (rmsOutLoopBody x_ptr rms_w_ptr out_ptr stride_x_k stride_rms_w
          stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE))
        stInit = some stLoop) :
    ∀ i : Fin N_SIZE,
      stLoop.readMem out_ptr
          (outColOffset s0 stride_out_batch stride_out_m stride_out_k i.val) =
        rmsnormWeightedYFullNSpec s0 x_ptr rms_w_ptr stride_x_batch stride_x_m
          stride_x_k stride_rms_w N_SIZE BLOCK_N_SIZE eps i := by
  obtain ⟨final, hFinal, hCtx⟩ :=
    rmsOutForRange_context s0 stInit stLoop x_ptr rms_w_ptr out_ptr
      stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE eps
      hStepNe hInit hXOutNe hWOutNe hOutInj hOutChunkInj hLoop
  exact rmsOutLoopContextInvariant_readout_fullN s0 stLoop x_ptr rms_w_ptr
    out_ptr stride_x_batch stride_x_m stride_x_k stride_rms_w
    stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE final eps
    hFinal hCtx

theorem rmsOutForRange_fullN_of_init_stride_pos
    (s0 stInit stLoop : BlockState) (x_ptr rms_w_ptr out_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE : Nat)
    (eps : ℝ)
    (hStepNe : BLOCK_N_SIZE ≠ 0)
    (hInit : rmsOutLoopContextInvariant s0 x_ptr rms_w_ptr out_ptr
      stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE 0 eps stInit)
    (hXOutNe : x_ptr ≠ out_ptr)
    (hWOutNe : rms_w_ptr ≠ out_ptr)
    (hStrideOutKPos : 0 < stride_out_k)
    (hLoop :
      stepStmt (.forRange "block_n_strart_ptr" 0 N_SIZE BLOCK_N_SIZE
        (rmsOutLoopBody x_ptr rms_w_ptr out_ptr stride_x_k stride_rms_w
          stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE))
        stInit = some stLoop) :
    ∀ i : Fin N_SIZE,
      stLoop.readMem out_ptr
          (outColOffset s0 stride_out_batch stride_out_m stride_out_k i.val) =
        rmsnormWeightedYFullNSpec s0 x_ptr rms_w_ptr stride_x_batch stride_x_m
          stride_x_k stride_rms_w N_SIZE BLOCK_N_SIZE eps i := by
  exact rmsOutForRange_fullN_of_init s0 stInit stLoop x_ptr rms_w_ptr out_ptr
    stride_x_batch stride_x_m stride_x_k stride_rms_w
    stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE eps
    hStepNe hInit hXOutNe hWOutNe
    (outColOffset_fin_injective_of_stride_out_k_pos s0 stride_out_batch
      stride_out_m stride_out_k N_SIZE hStrideOutKPos)
    (fun off _ =>
      outColOffset_chunk_injective_of_stride_out_k_pos s0 stride_out_batch
        stride_out_m stride_out_k BLOCK_N_SIZE off hStrideOutKPos)
    hLoop

noncomputable def rmsnormSpec
    (s : BlockState) (x_ptr rms_w_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k stride_rms_w N_SIZE BLOCK_N_SIZE : Nat)
    (eps : ℝ) (i : Fin BLOCK_N_SIZE) : ℝ :=
  WithBot.unbotD 0
    (Option.map₂ (fun scaled w => scaled * w)
      (Option.map₂ (fun x std => x / std)
        (some (s.readMem x_ptr
          (xOffset s stride_x_batch stride_x_m stride_x_k i)))
        (rmsStdCarrier s x_ptr stride_x_batch stride_x_m stride_x_k
          N_SIZE BLOCK_N_SIZE eps))
      (some (s.readMem rms_w_ptr (wOffset stride_rms_w i))))

/-- Algorithm-layer correctness for the one-block RMSNorm implementation slice. -/
theorem rmsnorm_implementation_correct
    (x_ptr rms_w_ptr out_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE : Nat)
    (eps : ℝ) (s s' : BlockState)
    (hNpos : 0 < N_SIZE) (hNle : N_SIZE ≤ BLOCK_N_SIZE)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N_SIZE =>
        outOffset s stride_out_batch stride_out_m stride_out_k i))
    (hExec : exec (rmsnorm_implementation x_ptr rms_w_ptr out_ptr
        stride_x_batch stride_x_m stride_x_k stride_rms_w
        stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE eps) s =
        some s') :
    ∀ i : Fin BLOCK_N_SIZE,
      s'.readMem out_ptr
          (outOffset s stride_out_batch stride_out_m stride_out_k i) =
        if i.val < N_SIZE then
          rmsnormSpec s x_ptr rms_w_ptr stride_x_batch stride_x_m stride_x_k
            stride_rms_w N_SIZE BLOCK_N_SIZE eps i
        else s.readMem out_ptr
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
    simp [exec, rmsnorm_implementation, stepStmts, stepStmt, evalOp, evalOp.eq_def,
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
      simp [hi, rmsnormSpec, rmsStdCarrier, rmsVarCarrier, rmsInputTile,
            xOffset, wOffset, Tile.reduceSum, Tile.reduceSumDrop, Tile.select,
            TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
            WithBot.realSqrt, Tile.natToReal, NumericDType.mul, NumericDType.div,
            FloatDType.cast]
      rfl
    · simp [hi]
  · exact False.elim (hB (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))
/-- Compute-facing correctness for the one-block RMSNorm implementation slice. -/
theorem rmsnorm_implementation_compute_correct
    (x_ptr rms_w_ptr out_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE : Nat)
    (eps : ℝ) (s : BlockState)
    (hNpos : 0 < N_SIZE) (hNle : N_SIZE ≤ BLOCK_N_SIZE)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N_SIZE =>
        outOffset s stride_out_batch stride_out_m stride_out_k i)) :
    ComputeCorrect.Realizes
      (kernel := rmsnorm_implementation x_ptr rms_w_ptr out_ptr
        stride_x_batch stride_x_m stride_x_k stride_rms_w
        stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE eps)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N_SIZE => i.val < N_SIZE)
        (fun i => (out_ptr,
          outOffset s stride_out_batch stride_out_m stride_out_k i)))
      (expected := fun i =>
        rmsnormSpec s x_ptr rms_w_ptr stride_x_batch stride_x_m stride_x_k
          stride_rms_w N_SIZE BLOCK_N_SIZE eps i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [rmsnorm_implementation, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := rmsnorm_implementation_correct x_ptr rms_w_ptr out_ptr
    stride_x_batch stride_x_m stride_x_k stride_rms_w
    stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE eps
    s s' hNpos hNle hOutInj hExec i
  simpa [hActive] using h
/-! ## Phase B: var-loop carriers and forRange invariant.

We isolate the first `for block_n_strart_ptr in range(0, N_SIZE, BLOCK_N_SIZE)`
loop and prove that its `var` register holds the per-lane prefix sum of
squares carrier indexed by `Fin BLOCK_N_SIZE`. -/

/-- Per-lane partial sum of squares after columns `0..off`. -/
noncomputable def rmsVarLanePrefix
    (s : BlockState) (x_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k N_SIZE BLOCK_N_SIZE off : Nat)
    (j : Fin BLOCK_N_SIZE) : ℝ :=
  ((Finset.range off).filter fun col => col < N_SIZE ∧ col % BLOCK_N_SIZE = j.val).sum
    fun col =>
      (s.readMem x_ptr
        (xColOffset s stride_x_batch stride_x_m stride_x_k col))^2

/-- Tile-valued accumulator spec at chunk offset `off`. -/
noncomputable def rmsVarAccumulatorSpec
    (s : BlockState) (x_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k N_SIZE BLOCK_N_SIZE off : Nat) :
    Tile .real [BLOCK_N_SIZE] :=
  { data := fun idx =>
      some (rmsVarLanePrefix s x_ptr stride_x_batch stride_x_m stride_x_k
        N_SIZE BLOCK_N_SIZE off idx.1) }

theorem rmsVarLanePrefix_final_eq_class_sum
    (s : BlockState) (x_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k N_SIZE BLOCK_N_SIZE off : Nat)
    (j : Fin BLOCK_N_SIZE) (hFinal : N_SIZE ≤ off) :
    rmsVarLanePrefix s x_ptr stride_x_batch stride_x_m stride_x_k
        N_SIZE BLOCK_N_SIZE off j =
      ((Finset.range N_SIZE).filter
        fun col => col % BLOCK_N_SIZE = j.val).sum
        fun col =>
          (s.readMem x_ptr
            (xColOffset s stride_x_batch stride_x_m stride_x_k col))^2 := by
  classical
  unfold rmsVarLanePrefix
  apply Finset.sum_congr
  · ext col
    simp only [Finset.mem_filter, Finset.mem_range, and_assoc]
    constructor
    · intro h
      exact ⟨h.2.1, h.2.2⟩
    · intro h
      exact ⟨Nat.lt_of_lt_of_le h.1 hFinal, h.1, h.2⟩
  · intro col _; rfl

theorem rmsVarFullNCarrier_partition_by_lane
    (s : BlockState) (x_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k N_SIZE BLOCK_N_SIZE : Nat)
    (hB : 0 < BLOCK_N_SIZE) :
    rmsVarFullNCarrier s x_ptr stride_x_batch stride_x_m stride_x_k
        N_SIZE BLOCK_N_SIZE =
      ∑ j : Fin BLOCK_N_SIZE,
        ((Finset.range N_SIZE).filter
          fun col => col % BLOCK_N_SIZE = j.val).sum
          fun col =>
            (s.readMem x_ptr
              (xColOffset s stride_x_batch stride_x_m stride_x_k col))^2 := by
  classical
  unfold rmsVarFullNCarrier xColOffset
  rw [Finset.sum_fin_eq_sum_range]
  rw [← Finset.sum_biUnion]
  · apply Finset.sum_congr
    · ext col
      simp only [Finset.mem_biUnion, Finset.mem_univ, true_and,
        Finset.mem_filter, Finset.mem_range]
      constructor
      · intro hcol
        exact ⟨⟨col % BLOCK_N_SIZE, Nat.mod_lt col hB⟩, hcol, rfl⟩
      · rintro ⟨j, hcol, _hmod⟩
        exact hcol
    · intro col hmem
      have hcol : col < N_SIZE := by
        simp only [Finset.mem_biUnion, Finset.mem_univ, true_and,
          Finset.mem_filter, Finset.mem_range] at hmem
        rcases hmem with ⟨j, hcol, _hmod⟩
        exact hcol
      simp [hcol]
  · intro i _ j _ hij
    apply Finset.disjoint_left.mpr
    intro col hi hj
    simp only [Finset.mem_filter, Finset.mem_range] at hi hj
    have hval : i.val = j.val := by
      rw [← hi.2, ← hj.2]
    exact hij (Fin.ext hval)

theorem rmsVarAccumulatorSpec_final_sum_eq_fullN
    (s : BlockState) (x_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k N_SIZE BLOCK_N_SIZE off : Nat)
    (hB : 0 < BLOCK_N_SIZE) (hFinal : N_SIZE ≤ off) :
    (∑ j : Fin BLOCK_N_SIZE,
      WithBot.unbotD 0
        ((rmsVarAccumulatorSpec s x_ptr stride_x_batch stride_x_m stride_x_k
          N_SIZE BLOCK_N_SIZE off).data (j, PUnit.unit))) =
      rmsVarFullNCarrier s x_ptr stride_x_batch stride_x_m stride_x_k
        N_SIZE BLOCK_N_SIZE := by
  classical
  rw [rmsVarFullNCarrier_partition_by_lane s x_ptr stride_x_batch stride_x_m
    stride_x_k N_SIZE BLOCK_N_SIZE hB]
  apply Finset.sum_congr rfl
  intro j _
  simp [rmsVarAccumulatorSpec,
    rmsVarLanePrefix_final_eq_class_sum s x_ptr stride_x_batch stride_x_m
      stride_x_k N_SIZE BLOCK_N_SIZE off j hFinal]

theorem rmsVarAccumulatorSpec_reduceSum
    (s : BlockState) (x_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k N_SIZE BLOCK_N_SIZE off : Nat) :
    (Tile.reduceSum (shape := [BLOCK_N_SIZE]) ⟨0, by simp⟩ Bool.false
      (rmsVarAccumulatorSpec s x_ptr stride_x_batch stride_x_m stride_x_k
        N_SIZE BLOCK_N_SIZE off)).data PUnit.unit =
      some
        (∑ j : Fin BLOCK_N_SIZE,
          WithBot.unbotD 0
            ((rmsVarAccumulatorSpec s x_ptr stride_x_batch stride_x_m
              stride_x_k N_SIZE BLOCK_N_SIZE off).data (j, PUnit.unit))) := by
  simp [Tile.reduceSum, Tile.reduceSumDrop, TileShape.axisDim,
    TileShape.eraseAxis, TileShape.insertAxisIndex, rmsVarAccumulatorSpec]
  rfl

/-- Tile-valued spec for the squared chunk loaded at chunk offset `off`:
the lane-`j` entry is `(x[off+j])^2` when `off + j < N_SIZE`, else `0`. -/
noncomputable def rmsVarChunkSquareSpec
    (s : BlockState) (x_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k N_SIZE BLOCK_N_SIZE off : Nat) :
    Tile .real [BLOCK_N_SIZE] :=
  { data := fun idx =>
      some
        (if off + idx.1.val < N_SIZE then
          (s.readMem x_ptr
            (xColOffset s stride_x_batch stride_x_m stride_x_k
              (off + idx.1.val)))^2
        else
          0) }

@[simp] theorem rmsVarLanePrefix_zero
    (s : BlockState) (x_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k N_SIZE BLOCK_N_SIZE : Nat)
    (j : Fin BLOCK_N_SIZE) :
    rmsVarLanePrefix s x_ptr stride_x_batch stride_x_m stride_x_k
        N_SIZE BLOCK_N_SIZE 0 j = 0 := by
  simp [rmsVarLanePrefix]

theorem rmsVarAccumulatorSpec_zero
    (s : BlockState) (x_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k N_SIZE BLOCK_N_SIZE : Nat) :
    rmsVarAccumulatorSpec s x_ptr stride_x_batch stride_x_m stride_x_k
        N_SIZE BLOCK_N_SIZE 0 =
      { data := fun _ : TileIndex [BLOCK_N_SIZE] => some 0 } := by
  ext idx
  simp [rmsVarAccumulatorSpec]

theorem rmsVarLoopOffset_mod_step
    (off BLOCK_N_SIZE : Nat) (hOff : off % BLOCK_N_SIZE = 0) :
    (off + BLOCK_N_SIZE) % BLOCK_N_SIZE = 0 := by
  rw [Nat.add_mod, hOff]
  simp

theorem rmsVarChunkLane_mod
    (off BLOCK_N_SIZE : Nat) (j : Fin BLOCK_N_SIZE)
    (hOff : off % BLOCK_N_SIZE = 0) :
    (off + j.val) % BLOCK_N_SIZE = j.val := by
  rw [Nat.add_mod, hOff]
  simp [Nat.mod_eq_of_lt j.isLt]

theorem rmsVarChunkLane_not_mem_current
    (N_SIZE off BLOCK_N_SIZE : Nat) (j : Fin BLOCK_N_SIZE) :
    off + j.val ∉ (Finset.range off).filter
      (fun col => col < N_SIZE ∧ col % BLOCK_N_SIZE = j.val) := by
  simp

theorem rmsVarLanePrefix_step
    (s : BlockState) (x_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k N_SIZE BLOCK_N_SIZE off : Nat)
    (j : Fin BLOCK_N_SIZE) (hOff : off % BLOCK_N_SIZE = 0) :
    rmsVarLanePrefix s x_ptr stride_x_batch stride_x_m stride_x_k
        N_SIZE BLOCK_N_SIZE (off + BLOCK_N_SIZE) j =
      rmsVarLanePrefix s x_ptr stride_x_batch stride_x_m stride_x_k
          N_SIZE BLOCK_N_SIZE off j +
        if off + j.val < N_SIZE then
          (s.readMem x_ptr
            (xColOffset s stride_x_batch stride_x_m stride_x_k
              (off + j.val)))^2
        else
          0 := by
  classical
  let pred : Nat → Prop := fun col => col < N_SIZE ∧ col % BLOCK_N_SIZE = j.val
  let f : Nat → ℝ := fun col =>
    (s.readMem x_ptr
      (xColOffset s stride_x_batch stride_x_m stride_x_k col))^2
  have hunique :
      ∀ col, off ≤ col → col < off + BLOCK_N_SIZE → col % BLOCK_N_SIZE = j.val →
        col = off + j.val := by
    intro col hle hlt hmod
    obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_le hle
    have hklt : k < BLOCK_N_SIZE := by omega
    have hmodk : (off + k) % BLOCK_N_SIZE = k := by
      rw [Nat.add_mod, hOff, Nat.mod_eq_of_lt hklt]
      simpa using Nat.mod_eq_of_lt hklt
    rw [hmodk] at hmod
    omega
  by_cases hjN : off + j.val < N_SIZE
  · have hset :
        (Finset.range (off + BLOCK_N_SIZE)).filter pred =
          insert (off + j.val) ((Finset.range off).filter pred) := by
      ext col
      simp only [Finset.mem_filter, Finset.mem_range, Finset.mem_insert]
      constructor
      · intro h
        by_cases hltOff : col < off
        · exact Or.inr ⟨hltOff, h.2⟩
        · exact Or.inl (hunique col (Nat.le_of_not_gt hltOff) h.1 h.2.2)
      · intro h
        rcases h with h | h
        · subst h
          exact ⟨by omega, hjN, rmsVarChunkLane_mod off BLOCK_N_SIZE j hOff⟩
        · exact ⟨by omega, h.2⟩
    unfold rmsVarLanePrefix
    simp [hjN]
    change ((Finset.range (off + BLOCK_N_SIZE)).filter pred).sum f =
      ((Finset.range off).filter pred).sum f + f (off + j.val)
    rw [hset]
    rw [Finset.sum_insert]
    · ring
    · exact rmsVarChunkLane_not_mem_current N_SIZE off BLOCK_N_SIZE j
  · have hset :
        (Finset.range (off + BLOCK_N_SIZE)).filter pred =
          (Finset.range off).filter pred := by
      ext col
      simp only [Finset.mem_filter, Finset.mem_range]
      constructor
      · intro h
        by_cases hltOff : col < off
        · exact ⟨hltOff, h.2⟩
        · have hcol : col = off + j.val :=
            hunique col (Nat.le_of_not_gt hltOff) h.1 h.2.2
          omega
      · intro h
        exact ⟨by omega, h.2⟩
    unfold rmsVarLanePrefix
    simp [hjN]
    change ((Finset.range (off + BLOCK_N_SIZE)).filter pred).sum f =
      ((Finset.range off).filter pred).sum f
    rw [hset]

theorem rmsVarAccumulatorSpec_step
    (s : BlockState) (x_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k N_SIZE BLOCK_N_SIZE off : Nat)
    (hOff : off % BLOCK_N_SIZE = 0) :
    rmsVarAccumulatorSpec s x_ptr stride_x_batch stride_x_m stride_x_k
        N_SIZE BLOCK_N_SIZE (off + BLOCK_N_SIZE) =
      { data := fun idx : TileIndex [BLOCK_N_SIZE] =>
          some
            (WithBot.unbotD 0
                ((rmsVarAccumulatorSpec s x_ptr stride_x_batch stride_x_m stride_x_k
                  N_SIZE BLOCK_N_SIZE off).data idx) +
              WithBot.unbotD 0
                ((rmsVarChunkSquareSpec s x_ptr stride_x_batch stride_x_m stride_x_k
                  N_SIZE BLOCK_N_SIZE off).data idx)) } := by
  ext idx
  by_cases hcol : off + idx.1.val < N_SIZE
  · simp [rmsVarAccumulatorSpec, rmsVarChunkSquareSpec, hcol,
      rmsVarLanePrefix_step s x_ptr stride_x_batch stride_x_m stride_x_k
        N_SIZE BLOCK_N_SIZE off idx.1 hOff]
  · simp [rmsVarAccumulatorSpec, rmsVarChunkSquareSpec, hcol,
      rmsVarLanePrefix_step s x_ptr stride_x_batch stride_x_m stride_x_k
        N_SIZE BLOCK_N_SIZE off idx.1 hOff]

/-! ### Var-loop body as a list of statements. -/

/-- The pre-loop statements that establish the var register and context. -/
def rmsVarPreLoop
    (stride_x_batch stride_x_m BLOCK_N_SIZE : Nat) :
    List Stmt :=
  [ .assign .nat [] "pid_batch" (.programId 0)
  , .assign .nat [] "pid_m" (.programId 1)
  , .assign .nat [] "offset_m"
      (.add NumericDType.nat Broadcast.nil
        (.mul NumericDType.nat Broadcast.nil
          (.ref .nat [] "pid_batch") (.constNat stride_x_batch))
        (.mul NumericDType.nat Broadcast.nil
          (.ref .nat [] "pid_m") (.constNat stride_x_m)))
  , .assign .nat [BLOCK_N_SIZE] "block_n_size" (.arange BLOCK_N_SIZE)
  , .assign .real [BLOCK_N_SIZE] "var"
      (.full [BLOCK_N_SIZE] (.const 0))
  ]

/-- The var-loop body statements as expanded by the DSL. -/
def rmsVarLoopBody
    (x_ptr : RegionName) (stride_x_k N_SIZE BLOCK_N_SIZE : Nat) :
    List Stmt :=
  [ .assign .nat [BLOCK_N_SIZE] "offset_n"
      (.add NumericDType.nat Broadcast.scalarL
        (.ref .nat [] "block_n_strart_ptr")
        (.ref .nat [BLOCK_N_SIZE] "block_n_size"))
  , .assign .bool [BLOCK_N_SIZE] "x_ptr_mask"
      (.lt ComparableDType.nat Broadcast.scalarR
        (.ref .nat [BLOCK_N_SIZE] "offset_n") (.constNat N_SIZE))
  , .assign .real [BLOCK_N_SIZE] "x"
      (.load .real
        (.region x_ptr
          (.add NumericDType.nat Broadcast.scalarL
            (.ref .nat [] "offset_m")
            (.mul NumericDType.nat Broadcast.scalarR
              (.ref .nat [BLOCK_N_SIZE] "offset_n") (.constNat stride_x_k))))
        (.maskOther
          (.ref .bool [BLOCK_N_SIZE] "x_ptr_mask")
          ((Op.const 0.0).broadcast [BLOCK_N_SIZE])))
  , .assign .real [BLOCK_N_SIZE] "xf"
      (.castFloat FloatDType.real FloatDType.real
        (.ref .real [BLOCK_N_SIZE] "x"))
  , .assign .real [BLOCK_N_SIZE] "var"
      (.add NumericDType.real Broadcast.nil.consSame
        (.ref .real [BLOCK_N_SIZE] "var")
        (.mul NumericDType.real Broadcast.nil.consSame
          (.ref .real [BLOCK_N_SIZE] "xf")
          (.ref .real [BLOCK_N_SIZE] "xf")))
  ]

/-- Pre-loop context invariant: after executing the pre-loop statements,
the `var` register is the zero tile and the auxiliary registers hold the
expected scalar/arange values, with memory unchanged. -/
theorem rmsVarPreLoop_step_regs
    (s st : BlockState) (x_ptr : RegionName)
    (stride_x_batch stride_x_m BLOCK_N_SIZE : Nat)
    (hStep :
      stepStmts (rmsVarPreLoop stride_x_batch stride_x_m BLOCK_N_SIZE) s
        = some st) :
    st.regs .real [BLOCK_N_SIZE] "var" =
        some { data := fun _ : TileIndex [BLOCK_N_SIZE] => some 0 } ∧
      st.regs .nat [] "offset_m" =
        some (Tile.scalar
          (s.pids 0 * stride_x_batch + s.pids 1 * stride_x_m)) ∧
      st.regs .nat [BLOCK_N_SIZE] "block_n_size" =
        some { data := fun idx : TileIndex [BLOCK_N_SIZE] => idx.1.val } ∧
      (∀ offset, st.readMem x_ptr offset = s.readMem x_ptr offset) := by
  unfold rmsVarPreLoop at hStep
  simp [stepStmts, stepStmt, evalOp, evalOp.eq_def, Tile.bop, NumericDType.add,
    NumericDType.mul, BlockState.setReg, Option.bind] at hStep
  subst st
  refine ⟨?_, ?_, ?_, ?_⟩
  · simp
  · simp [BlockState.setReg]
  · simp [BlockState.setReg]
    rfl
  · intro offset
    rfl

/-- Per-iteration accumulator update for the var-loop body.

Given context registers `offset_m`, `block_n_size`, and `var` set to the
expected values, executing the body with `block_n_strart_ptr = off` updates
`var` to its post-iteration value: the previous accumulator plus the squared
chunk-load tile. -/
theorem rmsVarLoopBody_step_accumulator_update
    (s0 st st' : BlockState) (x_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k N_SIZE BLOCK_N_SIZE off : Nat)
    (hAcc :
      st.regs .real [BLOCK_N_SIZE] "var" =
        some (rmsVarAccumulatorSpec s0 x_ptr stride_x_batch stride_x_m
          stride_x_k N_SIZE BLOCK_N_SIZE off))
    (hOffsetM :
      st.regs .nat [] "offset_m" =
        some (Tile.scalar
          (s0.pids 0 * stride_x_batch + s0.pids 1 * stride_x_m)))
    (hBlockN :
      st.regs .nat [BLOCK_N_SIZE] "block_n_size" =
        some { data := fun idx : TileIndex [BLOCK_N_SIZE] => idx.1.val })
    (hRead : ∀ offset, st.readMem x_ptr offset = s0.readMem x_ptr offset)
    (hStep :
      stepStmts (rmsVarLoopBody x_ptr stride_x_k N_SIZE BLOCK_N_SIZE)
        (st.setReg "block_n_strart_ptr" .nat [] (Tile.scalar off)) = some st') :
    st'.regs .real [BLOCK_N_SIZE] "var" =
      some
        { data := fun idx : TileIndex [BLOCK_N_SIZE] =>
            some
              (WithBot.unbotD 0
                  ((rmsVarAccumulatorSpec s0 x_ptr stride_x_batch stride_x_m
                    stride_x_k N_SIZE BLOCK_N_SIZE off).data idx) +
                WithBot.unbotD 0
                  ((rmsVarChunkSquareSpec s0 x_ptr stride_x_batch stride_x_m
                    stride_x_k N_SIZE BLOCK_N_SIZE off).data idx)) } := by
  unfold rmsVarLoopBody at hStep
  simp [stepStmts, stepStmt, evalOp, evalOp.eq_def, hAcc, hOffsetM, hBlockN, Tile.bop, Tile.cop,
    Tile.ptrAdd, Tile.uop, NumericDType.add, NumericDType.mul, ComparableDType.lt,
    Option.bind, FloatDType.cast, hRead, xColOffset] at hStep
  subst st'
  simp [BlockState.setReg]
  ext idx
  by_cases hcol : off + idx.1.val < N_SIZE
  · simp [rmsVarAccumulatorSpec, rmsVarChunkSquareSpec, hcol, xColOffset, sq]
  · simp [rmsVarAccumulatorSpec, rmsVarChunkSquareSpec, hcol, xColOffset]
    norm_num

/-- Body step preserves the context registers (offset_m, block_n_size) and
the x_ptr memory contents. -/
theorem rmsVarLoopBody_step_preserves_context
    (s0 st st' : BlockState) (x_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k N_SIZE BLOCK_N_SIZE off : Nat)
    (hAcc :
      st.regs .real [BLOCK_N_SIZE] "var" =
        some (rmsVarAccumulatorSpec s0 x_ptr stride_x_batch stride_x_m
          stride_x_k N_SIZE BLOCK_N_SIZE off))
    (hOffsetM :
      st.regs .nat [] "offset_m" =
        some (Tile.scalar
          (s0.pids 0 * stride_x_batch + s0.pids 1 * stride_x_m)))
    (hBlockN :
      st.regs .nat [BLOCK_N_SIZE] "block_n_size" =
        some { data := fun idx : TileIndex [BLOCK_N_SIZE] => idx.1.val })
    (hStep :
      stepStmts (rmsVarLoopBody x_ptr stride_x_k N_SIZE BLOCK_N_SIZE)
        (st.setReg "block_n_strart_ptr" .nat [] (Tile.scalar off)) = some st') :
    st'.regs .nat [] "offset_m" =
        some (Tile.scalar
          (s0.pids 0 * stride_x_batch + s0.pids 1 * stride_x_m)) ∧
      st'.regs .nat [BLOCK_N_SIZE] "block_n_size" =
        some { data := fun idx : TileIndex [BLOCK_N_SIZE] => idx.1.val } ∧
      (∀ offset, st'.readMem x_ptr offset = st.readMem x_ptr offset) := by
  unfold rmsVarLoopBody at hStep
  simp [stepStmts, stepStmt, evalOp, evalOp.eq_def, hAcc, hOffsetM, hBlockN, Tile.bop, Tile.cop,
    Tile.ptrAdd, Tile.uop, NumericDType.add, NumericDType.mul, ComparableDType.lt,
    Option.bind, FloatDType.cast] at hStep
  subst st'
  refine ⟨?_, ?_, ?_⟩
  · simp [hOffsetM]
  · simp [hBlockN]
  · intro offset
    rfl

/-- The accumulator-only loop invariant: `var` holds the carrier for chunk
offset `off`, and `off` is a multiple of `BLOCK_N_SIZE`. -/
def rmsVarLoopInvariant
    (s0 : BlockState) (x_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k N_SIZE BLOCK_N_SIZE off : Nat)
    (st : BlockState) : Prop :=
  off % BLOCK_N_SIZE = 0 ∧
    st.regs .real [BLOCK_N_SIZE] "var" =
      some (rmsVarAccumulatorSpec s0 x_ptr stride_x_batch stride_x_m
        stride_x_k N_SIZE BLOCK_N_SIZE off)

theorem rmsVarLoopInvariant_init_of_zero_reg
    (s0 st : BlockState) (x_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k N_SIZE BLOCK_N_SIZE : Nat)
    (hReg :
      st.regs .real [BLOCK_N_SIZE] "var" =
        some { data := fun _ : TileIndex [BLOCK_N_SIZE] => some 0 }) :
    rmsVarLoopInvariant s0 x_ptr stride_x_batch stride_x_m stride_x_k
      N_SIZE BLOCK_N_SIZE 0 st := by
  refine ⟨?_, ?_⟩
  · simp
  · simpa [rmsVarAccumulatorSpec_zero] using hReg

theorem rmsVarLoopInvariant_step_of_accumulator_update
    (s0 st' : BlockState) (x_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k N_SIZE BLOCK_N_SIZE off : Nat)
    (hOff : off % BLOCK_N_SIZE = 0)
    (hUpdate :
      st'.regs .real [BLOCK_N_SIZE] "var" =
        some
          { data := fun idx : TileIndex [BLOCK_N_SIZE] =>
              some
                (WithBot.unbotD 0
                    ((rmsVarAccumulatorSpec s0 x_ptr stride_x_batch stride_x_m
                      stride_x_k N_SIZE BLOCK_N_SIZE off).data idx) +
                  WithBot.unbotD 0
                    ((rmsVarChunkSquareSpec s0 x_ptr stride_x_batch stride_x_m
                      stride_x_k N_SIZE BLOCK_N_SIZE off).data idx)) }) :
    rmsVarLoopInvariant s0 x_ptr stride_x_batch stride_x_m stride_x_k
      N_SIZE BLOCK_N_SIZE (off + BLOCK_N_SIZE) st' := by
  refine ⟨?_, ?_⟩
  · exact rmsVarLoopOffset_mod_step off BLOCK_N_SIZE hOff
  · simpa [rmsVarAccumulatorSpec_step s0 x_ptr stride_x_batch stride_x_m
      stride_x_k N_SIZE BLOCK_N_SIZE off hOff] using hUpdate

/-- Context invariant for the var-loop: accumulator carrier plus auxiliary
context registers and memory preservation. -/
def rmsVarLoopContextInvariant
    (s0 : BlockState) (x_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k N_SIZE BLOCK_N_SIZE off : Nat)
    (st : BlockState) : Prop :=
  rmsVarLoopInvariant s0 x_ptr stride_x_batch stride_x_m stride_x_k
      N_SIZE BLOCK_N_SIZE off st ∧
    st.regs .nat [] "offset_m" =
      some (Tile.scalar
        (s0.pids 0 * stride_x_batch + s0.pids 1 * stride_x_m)) ∧
    st.regs .nat [BLOCK_N_SIZE] "block_n_size" =
      some { data := fun idx : TileIndex [BLOCK_N_SIZE] => idx.1.val } ∧
    (∀ offset, st.readMem x_ptr offset = s0.readMem x_ptr offset)

theorem rmsVarLoopContextInvariant_init_of_preloop
    (s0 st : BlockState) (x_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k N_SIZE BLOCK_N_SIZE : Nat)
    (hStep :
      stepStmts (rmsVarPreLoop stride_x_batch stride_x_m BLOCK_N_SIZE) s0
        = some st) :
    rmsVarLoopContextInvariant s0 x_ptr stride_x_batch stride_x_m stride_x_k
      N_SIZE BLOCK_N_SIZE 0 st := by
  rcases rmsVarPreLoop_step_regs s0 st x_ptr stride_x_batch stride_x_m
      BLOCK_N_SIZE hStep with ⟨hZero, hOffsetM, hBlockN, hRead⟩
  exact ⟨rmsVarLoopInvariant_init_of_zero_reg s0 st x_ptr stride_x_batch
      stride_x_m stride_x_k N_SIZE BLOCK_N_SIZE hZero,
    hOffsetM, hBlockN, hRead⟩

theorem rmsVarLoopContextInvariant_step_of_body
    (s0 st st' : BlockState) (x_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k N_SIZE BLOCK_N_SIZE off : Nat)
    (hCtx : rmsVarLoopContextInvariant s0 x_ptr stride_x_batch stride_x_m
      stride_x_k N_SIZE BLOCK_N_SIZE off st)
    (hStep :
      stepStmts (rmsVarLoopBody x_ptr stride_x_k N_SIZE BLOCK_N_SIZE)
        (st.setReg "block_n_strart_ptr" .nat [] (Tile.scalar off)) = some st') :
    rmsVarLoopContextInvariant s0 x_ptr stride_x_batch stride_x_m stride_x_k
      N_SIZE BLOCK_N_SIZE (off + BLOCK_N_SIZE) st' := by
  rcases hCtx with ⟨hInv, hOffsetM, hBlockN, hRead⟩
  have hUpdate :=
    rmsVarLoopBody_step_accumulator_update s0 st st' x_ptr
      stride_x_batch stride_x_m stride_x_k N_SIZE BLOCK_N_SIZE off
      hInv.2 hOffsetM hBlockN hRead hStep
  rcases rmsVarLoopBody_step_preserves_context s0 st st' x_ptr
    stride_x_batch stride_x_m stride_x_k N_SIZE BLOCK_N_SIZE off
    hInv.2 hOffsetM hBlockN hStep with ⟨hOffsetM', hBlockN', hReadStep⟩
  refine ⟨?_, hOffsetM', hBlockN', ?_⟩
  · exact rmsVarLoopInvariant_step_of_accumulator_update s0 st' x_ptr
      stride_x_batch stride_x_m stride_x_k N_SIZE BLOCK_N_SIZE off
      hInv.1 hUpdate
  · intro offset
    rw [hReadStep offset, hRead offset]

theorem rmsVarLoopContextInvariant_body_step_exists
    (s0 st : BlockState) (x_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k N_SIZE BLOCK_N_SIZE off : Nat)
    (hCtx : rmsVarLoopContextInvariant s0 x_ptr stride_x_batch stride_x_m
      stride_x_k N_SIZE BLOCK_N_SIZE off st) :
    ∃ st',
      stepStmts (rmsVarLoopBody x_ptr stride_x_k N_SIZE BLOCK_N_SIZE)
        (st.setReg "block_n_strart_ptr" .nat [] (Tile.scalar off)) = some st' ∧
      rmsVarLoopContextInvariant s0 x_ptr stride_x_batch stride_x_m stride_x_k
        N_SIZE BLOCK_N_SIZE (off + BLOCK_N_SIZE) st' := by
  rcases hCtx with ⟨hInv, hOffsetM, hBlockN, hRead⟩
  cases hStep :
      stepStmts (rmsVarLoopBody x_ptr stride_x_k N_SIZE BLOCK_N_SIZE)
        (st.setReg "block_n_strart_ptr" .nat [] (Tile.scalar off)) with
  | none =>
      unfold rmsVarLoopBody at hStep
      simp [stepStmts, stepStmt, evalOp, evalOp.eq_def, hInv.2, hOffsetM, hBlockN, Tile.bop,
        Tile.cop, Tile.ptrAdd, Tile.uop, NumericDType.add, NumericDType.mul,
        ComparableDType.lt, Option.bind, FloatDType.cast] at hStep
  | some st' =>
      refine ⟨st', rfl, ?_⟩
      exact rmsVarLoopContextInvariant_step_of_body s0 st st' x_ptr
        stride_x_batch stride_x_m stride_x_k N_SIZE BLOCK_N_SIZE off
        ⟨hInv, hOffsetM, hBlockN, hRead⟩ hStep

/-- After the var-loop completes, the context invariant holds at some
final offset `≥ N_SIZE`. This is the Phase-B headline lemma. -/
theorem rmsVarForRange_context_of_preloop
    (s0 stPre stLoop : BlockState) (x_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k N_SIZE BLOCK_N_SIZE : Nat)
    (hStepNe : BLOCK_N_SIZE ≠ 0)
    (hPre :
      stepStmts (rmsVarPreLoop stride_x_batch stride_x_m BLOCK_N_SIZE) s0
        = some stPre)
    (hLoop :
      stepStmt (.forRange "block_n_strart_ptr" 0 N_SIZE BLOCK_N_SIZE
        (rmsVarLoopBody x_ptr stride_x_k N_SIZE BLOCK_N_SIZE))
        stPre = some stLoop) :
    ∃ final,
      N_SIZE ≤ final ∧
        rmsVarLoopContextInvariant s0 x_ptr stride_x_batch stride_x_m
          stride_x_k N_SIZE BLOCK_N_SIZE final stLoop := by
  have hInit :
      rmsVarLoopContextInvariant s0 x_ptr stride_x_batch stride_x_m
        stride_x_k N_SIZE BLOCK_N_SIZE 0 stPre :=
    rmsVarLoopContextInvariant_init_of_preloop s0 stPre x_ptr stride_x_batch
      stride_x_m stride_x_k N_SIZE BLOCK_N_SIZE hPre
  obtain ⟨final, stFinal, hFor, hFinal, hCtx⟩ :=
    forRange_inv
      (idx := "block_n_strart_ptr") (start := 0) (stop := N_SIZE)
      (step := BLOCK_N_SIZE)
      (body := rmsVarLoopBody x_ptr stride_x_k N_SIZE BLOCK_N_SIZE)
      (P := rmsVarLoopContextInvariant s0 x_ptr stride_x_batch stride_x_m
        stride_x_k N_SIZE BLOCK_N_SIZE)
      (s_init := stPre)
      hStepNe hInit
      (by
        intro off st _hlt hCtx
        exact rmsVarLoopContextInvariant_body_step_exists s0 st x_ptr
          stride_x_batch stride_x_m stride_x_k N_SIZE BLOCK_N_SIZE off hCtx)
  have hEq : stFinal = stLoop := by
    rw [hLoop] at hFor
    injection hFor with h
    exact h.symm
  subst hEq
  exact ⟨final, hFinal, hCtx⟩

theorem rmsVarLoopInvariant_reduceSum_to_fullN
    (s0 st : BlockState) (x_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k N_SIZE BLOCK_N_SIZE off : Nat)
    (hInv : rmsVarLoopContextInvariant s0 x_ptr stride_x_batch stride_x_m
      stride_x_k N_SIZE BLOCK_N_SIZE off st)
    (hB : 0 < BLOCK_N_SIZE) (hFinal : N_SIZE ≤ off) :
    ∃ acc : Tile .real [BLOCK_N_SIZE],
      st.regs .real [BLOCK_N_SIZE] "var" = some acc ∧
        WithBot.unbotD 0
          ((Tile.reduceSum (shape := [BLOCK_N_SIZE]) ⟨0, by simp⟩ Bool.false
            acc).data PUnit.unit) / (N_SIZE : ℝ) =
          rmsVarFullNCarrier s0 x_ptr stride_x_batch stride_x_m stride_x_k
            N_SIZE BLOCK_N_SIZE / (N_SIZE : ℝ) := by
  refine ⟨rmsVarAccumulatorSpec s0 x_ptr stride_x_batch stride_x_m stride_x_k
      N_SIZE BLOCK_N_SIZE off, hInv.1.2, ?_⟩
  rw [rmsVarAccumulatorSpec_reduceSum]
  simp [rmsVarAccumulatorSpec_final_sum_eq_fullN s0 x_ptr stride_x_batch
    stride_x_m stride_x_k N_SIZE BLOCK_N_SIZE off hB hFinal]

def rmsStdPostLoop (N_SIZE BLOCK_N_SIZE : Nat) (eps : ℝ) : List Stmt :=
  [ .assign .real [] "var"
      (.div NumericDType.real Broadcast.nil
        (.reduceSum (⟨0, by simp⟩ : Fin [BLOCK_N_SIZE].length) Bool.false
          (.ref .real [BLOCK_N_SIZE] "var"))
        (.const (N_SIZE : ℝ)))
  , .assign .real [] "std"
      (.sqrt
        (.add NumericDType.real Broadcast.nil
          (.ref .real [] "var")
          (.const eps)))
  ]

theorem rmsStdPostLoop_step_to_out_init
    (s0 stVar stStd : BlockState) (x_ptr rms_w_ptr out_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE final : Nat)
    (eps : ℝ)
    (hVarCtx : rmsVarLoopContextInvariant s0 x_ptr stride_x_batch stride_x_m
      stride_x_k N_SIZE BLOCK_N_SIZE final stVar)
    (hFinal : N_SIZE ≤ final)
    (hBlockPos : 0 < BLOCK_N_SIZE)
    (hPids : stVar.pids = s0.pids)
    (hPidBatch :
      stVar.regs .nat [] "pid_batch" = some (Tile.scalar (s0.pids 0)))
    (hPidM :
      stVar.regs .nat [] "pid_m" = some (Tile.scalar (s0.pids 1)))
    (hReadW : ∀ offset, stVar.readMem rms_w_ptr offset = s0.readMem rms_w_ptr offset)
    (hStep :
      stepStmts (rmsStdPostLoop N_SIZE BLOCK_N_SIZE eps) stVar = some stStd) :
    rmsOutLoopContextInvariant s0 x_ptr rms_w_ptr out_ptr
      stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE 0 eps stStd := by
  rcases hVarCtx with ⟨hVarInv, hOffsetM, hBlockN, hReadX⟩
  rcases rmsVarLoopInvariant_reduceSum_to_fullN s0 stVar x_ptr
      stride_x_batch stride_x_m stride_x_k N_SIZE BLOCK_N_SIZE final
      ⟨hVarInv, hOffsetM, hBlockN, hReadX⟩ hBlockPos hFinal with
    ⟨acc, hVarReg, hReduce⟩
  have hAccEq :
      acc =
        rmsVarAccumulatorSpec s0 x_ptr stride_x_batch stride_x_m stride_x_k
          N_SIZE BLOCK_N_SIZE final := by
    rw [hVarInv.2] at hVarReg
    injection hVarReg with h
    exact h.symm
  subst acc
  have hSum :
      (∑ i : Fin BLOCK_N_SIZE,
        rmsVarLanePrefix s0 x_ptr stride_x_batch stride_x_m stride_x_k
          N_SIZE BLOCK_N_SIZE final i) =
        rmsVarFullNCarrier s0 x_ptr stride_x_batch stride_x_m stride_x_k
          N_SIZE BLOCK_N_SIZE := by
    simpa [rmsVarAccumulatorSpec] using
      rmsVarAccumulatorSpec_final_sum_eq_fullN s0 x_ptr stride_x_batch
        stride_x_m stride_x_k N_SIZE BLOCK_N_SIZE final hBlockPos hFinal
  have hStdArg :
      (∑ i : Fin BLOCK_N_SIZE,
          rmsVarLanePrefix s0 x_ptr stride_x_batch stride_x_m stride_x_k
            N_SIZE BLOCK_N_SIZE final i) / (N_SIZE : ℝ) + eps =
        rmsVarFullNCarrier s0 x_ptr stride_x_batch stride_x_m stride_x_k
            N_SIZE BLOCK_N_SIZE / (N_SIZE : ℝ) + eps := by
    rw [hSum]
  unfold rmsStdPostLoop at hStep
  simp [stepStmts, stepStmt, evalOp, evalOp.eq_def, hVarReg, Tile.bop, Tile.uop,
    NumericDType.add, NumericDType.div, Option.bind, WithBot.realSqrt] at hStep
  subst stStd
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · exact rmsnormOutLoopInvariant_zero s0 _ x_ptr rms_w_ptr out_ptr
      stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE eps
  · simp [BlockState.setReg, hPids]
  · simp [BlockState.setReg, hPidBatch]
  · simp [BlockState.setReg, hPidM]
  · simp [BlockState.setReg, hOffsetM]
  · simp [BlockState.setReg, hBlockN]
  · simp [BlockState.setReg, rmsStdFullNSpec, Tile.reduceSum, Tile.reduceSumDrop,
      TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
      rmsVarAccumulatorSpec]
    exact congrArg
      (fun x : ℝ =>
        Tile.scalar (dtype := .real) ((Real.sqrt x : ℝ) : WithBot ℝ))
      hStdArg
  · intro offset
    change stVar.readMem x_ptr offset = s0.readMem x_ptr offset
    exact hReadX offset
  · intro offset
    change stVar.readMem rms_w_ptr offset = s0.readMem rms_w_ptr offset
    exact hReadW offset

theorem rmsnorm_implementation_staged_fullN_correct
    (s0 stPre stVar stStd stOut : BlockState)
    (x_ptr rms_w_ptr out_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE : Nat)
    (eps : ℝ)
    (hBlockPos : 0 < BLOCK_N_SIZE)
    (hStrideOutKPos : 0 < stride_out_k)
    (hXOutNe : x_ptr ≠ out_ptr)
    (hWOutNe : rms_w_ptr ≠ out_ptr)
    (hPidsVar : stVar.pids = s0.pids)
    (hPidBatchVar :
      stVar.regs .nat [] "pid_batch" = some (Tile.scalar (s0.pids 0)))
    (hPidMVar :
      stVar.regs .nat [] "pid_m" = some (Tile.scalar (s0.pids 1)))
    (hReadWVar : ∀ offset, stVar.readMem rms_w_ptr offset = s0.readMem rms_w_ptr offset)
    (hPre :
      stepStmts (rmsVarPreLoop stride_x_batch stride_x_m BLOCK_N_SIZE) s0
        = some stPre)
    (hVarLoop :
      stepStmt (.forRange "block_n_strart_ptr" 0 N_SIZE BLOCK_N_SIZE
        (rmsVarLoopBody x_ptr stride_x_k N_SIZE BLOCK_N_SIZE))
        stPre = some stVar)
    (hStd :
      stepStmts (rmsStdPostLoop N_SIZE BLOCK_N_SIZE eps) stVar = some stStd)
    (hOutLoop :
      stepStmt (.forRange "block_n_strart_ptr" 0 N_SIZE BLOCK_N_SIZE
        (rmsOutLoopBody x_ptr rms_w_ptr out_ptr stride_x_k stride_rms_w
          stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE))
        stStd = some stOut) :
    ∀ i : Fin N_SIZE,
      stOut.readMem out_ptr
          (outColOffset s0 stride_out_batch stride_out_m stride_out_k i.val) =
        rmsnormWeightedYFullNSpec s0 x_ptr rms_w_ptr stride_x_batch stride_x_m
          stride_x_k stride_rms_w N_SIZE BLOCK_N_SIZE eps i := by
  have hStepNe : BLOCK_N_SIZE ≠ 0 := Nat.ne_of_gt hBlockPos
  obtain ⟨final, hFinal, hVarCtx⟩ :=
    rmsVarForRange_context_of_preloop s0 stPre stVar x_ptr stride_x_batch
      stride_x_m stride_x_k N_SIZE BLOCK_N_SIZE hStepNe hPre hVarLoop
  have hOutInit :
      rmsOutLoopContextInvariant s0 x_ptr rms_w_ptr out_ptr
        stride_x_batch stride_x_m stride_x_k stride_rms_w
        stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE 0 eps
        stStd :=
    rmsStdPostLoop_step_to_out_init s0 stVar stStd x_ptr rms_w_ptr out_ptr
      stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE final eps
      hVarCtx hFinal hBlockPos hPidsVar hPidBatchVar hPidMVar hReadWVar hStd
  exact rmsOutForRange_fullN_of_init_stride_pos s0 stStd stOut x_ptr rms_w_ptr
    out_ptr stride_x_batch stride_x_m stride_x_k stride_rms_w
    stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE eps
    hStepNe hOutInit hXOutNe hWOutNe hStrideOutKPos hOutLoop

theorem rmsVarPreLoop_step_preserves_pids_read
    (s st : BlockState) (R : RegionName)
    (stride_x_batch stride_x_m BLOCK_N_SIZE : Nat)
    (hStep :
      stepStmts (rmsVarPreLoop stride_x_batch stride_x_m BLOCK_N_SIZE) s
        = some st) :
    st.pids = s.pids ∧
      st.regs .nat [] "pid_batch" = some (Tile.scalar (s.pids 0)) ∧
      st.regs .nat [] "pid_m" = some (Tile.scalar (s.pids 1)) ∧
      ∀ offset, st.readMem R offset = s.readMem R offset := by
  unfold rmsVarPreLoop at hStep
  simp [stepStmts, stepStmt, evalOp, evalOp.eq_def, Tile.bop, NumericDType.add,
    NumericDType.mul, Option.bind] at hStep
  subst st
  refine ⟨?_, ?_, ?_, ?_⟩
  · simp [BlockState.setReg]
  · simp [BlockState.setReg]
  · simp [BlockState.setReg]
  · intro offset
    rfl

theorem rmsVarLoopBody_step_preserves_pids_read
    (s0 st st' : BlockState) (x_ptr R : RegionName)
    (stride_x_batch stride_x_m stride_x_k N_SIZE BLOCK_N_SIZE off : Nat)
    (hCtx : rmsVarLoopContextInvariant s0 x_ptr stride_x_batch stride_x_m
      stride_x_k N_SIZE BLOCK_N_SIZE off st)
    (hStep :
      stepStmts (rmsVarLoopBody x_ptr stride_x_k N_SIZE BLOCK_N_SIZE)
        (st.setReg "block_n_strart_ptr" .nat [] (Tile.scalar off)) = some st') :
    st'.pids = st.pids ∧
      st'.regs .nat [] "pid_batch" = st.regs .nat [] "pid_batch" ∧
      st'.regs .nat [] "pid_m" = st.regs .nat [] "pid_m" ∧
      ∀ offset, st'.readMem R offset = st.readMem R offset := by
  rcases hCtx with ⟨hInv, hOffsetM, hBlockN, _hReadX⟩
  unfold rmsVarLoopBody at hStep
  simp [stepStmts, stepStmt, evalOp, evalOp.eq_def, hInv.2, hOffsetM, hBlockN, Tile.bop,
    Tile.cop, Tile.ptrAdd, Tile.uop, NumericDType.add, NumericDType.mul,
    ComparableDType.lt, Option.bind, FloatDType.cast] at hStep
  subst st'
  refine ⟨?_, ?_, ?_, ?_⟩
  · simp [BlockState.setReg]
  · simp [BlockState.setReg]
  · simp [BlockState.setReg]
  · intro offset
    rfl

theorem rmsVarForRange_context_pids_read_of_preloop
    (s0 stPre stLoop : BlockState) (x_ptr R : RegionName)
    (stride_x_batch stride_x_m stride_x_k N_SIZE BLOCK_N_SIZE : Nat)
    (hStepNe : BLOCK_N_SIZE ≠ 0)
    (hPre :
      stepStmts (rmsVarPreLoop stride_x_batch stride_x_m BLOCK_N_SIZE) s0
        = some stPre)
    (hLoop :
      stepStmt (.forRange "block_n_strart_ptr" 0 N_SIZE BLOCK_N_SIZE
        (rmsVarLoopBody x_ptr stride_x_k N_SIZE BLOCK_N_SIZE))
        stPre = some stLoop) :
    ∃ final,
      N_SIZE ≤ final ∧
        rmsVarLoopContextInvariant s0 x_ptr stride_x_batch stride_x_m
          stride_x_k N_SIZE BLOCK_N_SIZE final stLoop ∧
        stLoop.pids = s0.pids ∧
        stLoop.regs .nat [] "pid_batch" = some (Tile.scalar (s0.pids 0)) ∧
        stLoop.regs .nat [] "pid_m" = some (Tile.scalar (s0.pids 1)) ∧
        (∀ offset, stLoop.readMem R offset = s0.readMem R offset) := by
  have hInitVar :
      rmsVarLoopContextInvariant s0 x_ptr stride_x_batch stride_x_m
        stride_x_k N_SIZE BLOCK_N_SIZE 0 stPre :=
    rmsVarLoopContextInvariant_init_of_preloop s0 stPre x_ptr stride_x_batch
      stride_x_m stride_x_k N_SIZE BLOCK_N_SIZE hPre
  rcases rmsVarPreLoop_step_preserves_pids_read s0 stPre R
      stride_x_batch stride_x_m BLOCK_N_SIZE hPre with
    ⟨hPidsInit, hPidBatchInit, hPidMInit, hReadInit⟩
  obtain ⟨final, stFinal, hFor, hFinal, hCtx⟩ :=
    forRange_inv
      (idx := "block_n_strart_ptr") (start := 0) (stop := N_SIZE)
      (step := BLOCK_N_SIZE)
      (body := rmsVarLoopBody x_ptr stride_x_k N_SIZE BLOCK_N_SIZE)
      (P := fun off st =>
          rmsVarLoopContextInvariant s0 x_ptr stride_x_batch stride_x_m
          stride_x_k N_SIZE BLOCK_N_SIZE off st ∧
          st.pids = s0.pids ∧
          st.regs .nat [] "pid_batch" = some (Tile.scalar (s0.pids 0)) ∧
          st.regs .nat [] "pid_m" = some (Tile.scalar (s0.pids 1)) ∧
          (∀ offset, st.readMem R offset = s0.readMem R offset))
      (s_init := stPre)
      hStepNe ⟨hInitVar, hPidsInit, hPidBatchInit, hPidMInit, hReadInit⟩
      (by
        intro off st _hlt hCtx
        rcases hCtx with ⟨hVarCtx, hPids, hPidBatch, hPidM, hRead⟩
        obtain ⟨st', hStep, hVarCtx'⟩ :=
          rmsVarLoopContextInvariant_body_step_exists s0 st x_ptr
            stride_x_batch stride_x_m stride_x_k N_SIZE BLOCK_N_SIZE off hVarCtx
        rcases rmsVarLoopBody_step_preserves_pids_read s0 st st' x_ptr R
            stride_x_batch stride_x_m stride_x_k N_SIZE BLOCK_N_SIZE off
            hVarCtx hStep with
          ⟨hPidsStep, hPidBatchStep, hPidMStep, hReadStep⟩
        refine ⟨st', hStep, hVarCtx', ?_, ?_, ?_, ?_⟩
        · rw [hPidsStep, hPids]
        · rw [hPidBatchStep, hPidBatch]
        · rw [hPidMStep, hPidM]
        · intro offset
          rw [hReadStep offset, hRead offset])
  have hEq : stFinal = stLoop := by
    rw [hLoop] at hFor
    injection hFor with h
    exact h.symm
  subst hEq
  exact ⟨final, hFinal, hCtx.1, hCtx.2.1, hCtx.2.2.1, hCtx.2.2.2.1,
    hCtx.2.2.2.2⟩

theorem rmsnorm_implementation_staged_fullN_correct_from_preloop
    (s0 stPre stVar stStd stOut : BlockState)
    (x_ptr rms_w_ptr out_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE : Nat)
    (eps : ℝ)
    (hBlockPos : 0 < BLOCK_N_SIZE)
    (hStrideOutKPos : 0 < stride_out_k)
    (hXOutNe : x_ptr ≠ out_ptr)
    (hWOutNe : rms_w_ptr ≠ out_ptr)
    (hPre :
      stepStmts (rmsVarPreLoop stride_x_batch stride_x_m BLOCK_N_SIZE) s0
        = some stPre)
    (hVarLoop :
      stepStmt (.forRange "block_n_strart_ptr" 0 N_SIZE BLOCK_N_SIZE
        (rmsVarLoopBody x_ptr stride_x_k N_SIZE BLOCK_N_SIZE))
        stPre = some stVar)
    (hStd :
      stepStmts (rmsStdPostLoop N_SIZE BLOCK_N_SIZE eps) stVar = some stStd)
    (hOutLoop :
      stepStmt (.forRange "block_n_strart_ptr" 0 N_SIZE BLOCK_N_SIZE
        (rmsOutLoopBody x_ptr rms_w_ptr out_ptr stride_x_k stride_rms_w
          stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE))
        stStd = some stOut) :
    ∀ i : Fin N_SIZE,
      stOut.readMem out_ptr
          (outColOffset s0 stride_out_batch stride_out_m stride_out_k i.val) =
        rmsnormWeightedYFullNSpec s0 x_ptr rms_w_ptr stride_x_batch stride_x_m
          stride_x_k stride_rms_w N_SIZE BLOCK_N_SIZE eps i := by
  have hStepNe : BLOCK_N_SIZE ≠ 0 := Nat.ne_of_gt hBlockPos
  obtain ⟨_final, _hFinal, _hVarCtx, hPids, hPidBatch, hPidM, hReadW⟩ :=
    rmsVarForRange_context_pids_read_of_preloop s0 stPre stVar x_ptr
      rms_w_ptr stride_x_batch stride_x_m stride_x_k N_SIZE BLOCK_N_SIZE
      hStepNe hPre hVarLoop
  exact rmsnorm_implementation_staged_fullN_correct s0 stPre stVar stStd stOut
    x_ptr rms_w_ptr out_ptr stride_x_batch stride_x_m stride_x_k stride_rms_w
    stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE eps
    hBlockPos hStrideOutKPos hXOutNe hWOutNe hPids hPidBatch hPidM hReadW
    hPre hVarLoop hStd hOutLoop

theorem rmsnorm_implementation_toAlg_body
    (x_ptr rms_w_ptr out_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE : Nat)
    (eps : ℝ) :
    (rmsnorm_implementation x_ptr rms_w_ptr out_ptr
      stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE eps).toAlgKernel.body =
      rmsVarPreLoop stride_x_batch stride_x_m BLOCK_N_SIZE ++
      [ .forRange "block_n_strart_ptr" 0 N_SIZE BLOCK_N_SIZE
          (rmsVarLoopBody x_ptr stride_x_k N_SIZE BLOCK_N_SIZE) ] ++
      rmsStdPostLoop N_SIZE BLOCK_N_SIZE eps ++
      [ .forRange "block_n_strart_ptr" 0 N_SIZE BLOCK_N_SIZE
          (rmsOutLoopBody x_ptr rms_w_ptr out_ptr stride_x_k stride_rms_w
            stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE) ] := by
  simp [rmsnorm_implementation, rmsVarPreLoop, rmsVarLoopBody, rmsStdPostLoop,
    rmsOutLoopBody, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  constructor <;> rfl

theorem rmsnorm_implementation_fullN_correct
    (x_ptr rms_w_ptr out_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE : Nat)
    (eps : ℝ) (s s' : BlockState)
    (hBlockPos : 0 < BLOCK_N_SIZE)
    (hStrideOutKPos : 0 < stride_out_k)
    (hXOutNe : x_ptr ≠ out_ptr)
    (hWOutNe : rms_w_ptr ≠ out_ptr)
    (hExec : exec (rmsnorm_implementation x_ptr rms_w_ptr out_ptr
        stride_x_batch stride_x_m stride_x_k stride_rms_w
        stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE eps) s =
        some s') :
    ∀ i : Fin N_SIZE,
      s'.readMem out_ptr
          (outColOffset s stride_out_batch stride_out_m stride_out_k i.val) =
        rmsnormWeightedYFullNSpec s x_ptr rms_w_ptr stride_x_batch stride_x_m
          stride_x_k stride_rms_w N_SIZE BLOCK_N_SIZE eps i := by
  unfold exec at hExec
  rw [rmsnorm_implementation_toAlg_body] at hExec
  rw [stepStmts.append_some_iff] at hExec
  rcases hExec with ⟨stStd, hBeforeOut, hOutLoopList⟩
  rw [stepStmts.append_some_iff] at hBeforeOut
  rcases hBeforeOut with ⟨stVar, hBeforeStd, hStd⟩
  rw [stepStmts.append_some_iff] at hBeforeStd
  rcases hBeforeStd with ⟨stPre, hPre, hVarLoopList⟩
  have hVarLoop :
      stepStmt (.forRange "block_n_strart_ptr" 0 N_SIZE BLOCK_N_SIZE
        (rmsVarLoopBody x_ptr stride_x_k N_SIZE BLOCK_N_SIZE))
        stPre = some stVar := by
    unfold stepStmts at hVarLoopList
    unfold stepStmt
    cases hAux :
        stepForRangeAux "block_n_strart_ptr" 0 N_SIZE BLOCK_N_SIZE
          (rmsVarLoopBody x_ptr stride_x_k N_SIZE BLOCK_N_SIZE) stPre <;>
      simp [hAux] at hVarLoopList ⊢
    exact hVarLoopList
  have hOutLoop :
      stepStmt (.forRange "block_n_strart_ptr" 0 N_SIZE BLOCK_N_SIZE
        (rmsOutLoopBody x_ptr rms_w_ptr out_ptr stride_x_k stride_rms_w
          stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE))
        stStd = some s' := by
    unfold stepStmts at hOutLoopList
    unfold stepStmt
    cases hAux :
        stepForRangeAux "block_n_strart_ptr" 0 N_SIZE BLOCK_N_SIZE
          (rmsOutLoopBody x_ptr rms_w_ptr out_ptr stride_x_k stride_rms_w
            stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE) stStd <;>
      simp [hAux] at hOutLoopList ⊢
    exact hOutLoopList
  exact rmsnorm_implementation_staged_fullN_correct_from_preloop s stPre stVar
    stStd s' x_ptr rms_w_ptr out_ptr stride_x_batch stride_x_m stride_x_k
    stride_rms_w stride_out_batch stride_out_m stride_out_k N_SIZE
    BLOCK_N_SIZE eps hBlockPos hStrideOutKPos hXOutNe hWOutNe hPre hVarLoop
    hStd hOutLoop

/-- Compute-facing correctness for the full-N RMSNorm implementation. -/
theorem rmsnorm_implementation_compute_fullN_correct
    (x_ptr rms_w_ptr out_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE : Nat)
    (eps : ℝ) (s : BlockState)
    (hBlockPos : 0 < BLOCK_N_SIZE)
    (hStrideOutKPos : 0 < stride_out_k)
    (hXOutNe : x_ptr ≠ out_ptr)
    (hWOutNe : rms_w_ptr ≠ out_ptr) :
    ComputeCorrect.Realizes
      (kernel := rmsnorm_implementation x_ptr rms_w_ptr out_ptr
        stride_x_batch stride_x_m stride_x_k stride_rms_w
        stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE eps)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin N_SIZE => True)
        (fun i => (out_ptr,
          outColOffset s stride_out_batch stride_out_m stride_out_k i.val)))
      (expected := fun i =>
        rmsnormWeightedYFullNSpec s x_ptr rms_w_ptr stride_x_batch stride_x_m
          stride_x_k stride_rms_w N_SIZE BLOCK_N_SIZE eps i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [rmsnorm_implementation, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i _hActive
  exact rmsnorm_implementation_fullN_correct x_ptr rms_w_ptr out_ptr
    stride_x_batch stride_x_m stride_x_k stride_rms_w
    stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE eps
    s s' hBlockPos hStrideOutKPos hXOutNe hWOutNe hExec i

/-- Per-kernel output summary for `rmsnorm_triton`: the DSL surface lowers to the
algorithm layer, and the masked store to `out_ptr` is compute-correct for
arbitrary `N_SIZE` — every output column holds the full-`N` RMSNorm spec
`rmsnormWeightedYFullNSpec`. Built on the multi-block `*_compute_fullN_correct`
result; requires `0 < BLOCK_N_SIZE`, `0 < stride_out_k`, and output/input
disjointness. -/
theorem rmsnorm_implementation_output_summary
    (x_ptr rms_w_ptr out_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE : Nat)
    (eps : ℝ) (s : BlockState)
    (hBlockPos : 0 < BLOCK_N_SIZE)
    (hStrideOutKPos : 0 < stride_out_k)
    (hXOutNe : x_ptr ≠ out_ptr)
    (hWOutNe : rms_w_ptr ≠ out_ptr) :
    (∃ alg, (rmsnorm_implementation x_ptr rms_w_ptr out_ptr
        stride_x_batch stride_x_m stride_x_k stride_rms_w
        stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE
        eps).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := rmsnorm_implementation x_ptr rms_w_ptr out_ptr
        stride_x_batch stride_x_m stride_x_k stride_rms_w
        stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE eps)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin N_SIZE => True)
        (fun i => (out_ptr,
          outColOffset s stride_out_batch stride_out_m stride_out_k i.val)))
      (expected := fun i =>
        rmsnormWeightedYFullNSpec s x_ptr rms_w_ptr stride_x_batch stride_x_m
          stride_x_k stride_rms_w N_SIZE BLOCK_N_SIZE eps i) := by
  refine ⟨?_, ?_⟩
  · simp only [rmsnorm_implementation, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
    exact ⟨_, rfl⟩
  · exact rmsnorm_implementation_compute_fullN_correct x_ptr rms_w_ptr out_ptr
      stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE eps
      s hBlockPos hStrideOutKPos hXOutNe hWOutNe

end VeriTile.Bench.TritonBenchG.RmsnormImplementation
