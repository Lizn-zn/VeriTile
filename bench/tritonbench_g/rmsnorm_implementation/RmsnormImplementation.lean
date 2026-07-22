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
    ComputeCorrect.Realizes_without_Rounding
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
    ComputeCorrect.Realizes_without_Rounding
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
specification rmsnorm_implementation_output_summary
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
    ComputeCorrect.Realizes_without_Rounding
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

/-! ## The `⊨[R]` streaming headline (wave-5 S3 per-step emit genre)

Everything below is purely additive; the exact surface above is untouched.
This kernel is the near-twin of `rmsnorm_triton` and rides the same skin,
`StreamEmitMasked2DKernelIO₂` (streaming genre, style S3): the store sits
**inside** the second pass, so the output is a per-step `BLOCK_N_SIZE`-lane
window family rather than one terminal tile, and the kernel's spec `f t j`
is the genre's *two-pass* shape — the step-`t` tile combined with a fold
over the entire stream. Unlike the reciprocal-`rstd` twin, this kernel
computes `std = sqrt(var + eps)` and **divides** (`x / std`), so the stream
spec is stated in division form.

Structure of the `execR R` story: this kernel has **zero rounding events**.
Every load and store is at `.real`, and the only `castFloat` is the erased
`.to(tl.float32)` on `xf`, i.e. `.real → .real` — exact under every `R` by
the model's defining `round_real` (`Rcast_real_real` below). Both passes
therefore collapse verbatim onto the exact stepper
(`stepForRangeAuxR_castFree`), and the whole proven
`rmsVarPreLoop` / `rmsVarLoopContextInvariant` / `rmsStdPostLoop` /
`rmsOutLoopContextInvariant` invariant stack above is reused unchanged; the
`⊨[R]` face adds only the `TraceSafeR` walk, the per-cell memory frame, and
the stream-lane spec bridge. The skin's readback contract at the default
`outDType := .real` grid carries `R.round .real`, the identity by
`round_real` — the ∀-`R` face is the exact streaming contract via the
model's `.real` identity fields, not a `.triv` special case. -/

section IOFace

open Finset
open scoped VeriTile.Triton.StreamEmitMasked2DKernelIO₂

set_option maxHeartbeats 4000000
set_option linter.unusedVariables false

/-! ### Stream geometry: trip count and windows -/

/-- Trip count of both `for block_n_strart_ptr in range(0, N_SIZE,
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

/-- **Streaming IO signature** of `rmsnorm_implementation` on the two-stream
per-step emit skin (S3: in-loop store). Step `t` of either pass (at
`block_n_strart_ptr = t·BLOCK_N_SIZE`) reads the `BLOCK_N_SIZE`-lane `x`
tile (`read1`, both passes read the same addresses) and the `rms_w` tile
(`read2`, pass 2 only); step `t` of pass 2 stores the `BLOCK_N_SIZE`-lane
output window (`write`) at the **`.real`** grid (`outDType` default — the
kernel's store is untyped `tl.store(out_ptr + out_offset, out)` at `.real`,
so the per-step stores have no quantization event). The windows transcribe
the kernel's pointer arithmetic verbatim
(`offset_n = t·BLOCK_N_SIZE + j`):

* `read1` step `t`, lane `j`:
  `pid₀·stride_x_batch + pid₁·stride_x_m + (t·BLOCK_N_SIZE + j)·stride_x_k`
  — the kernel's `x_ptr + offset_m + offset_n * stride_x_k`.
* `read2` step `t`, lane `j`: `(t·BLOCK_N_SIZE + j)·stride_rms_w` — the
  kernel's `rms_w_ptr + offset_n * stride_rms_w`.
* `write` step `t`, lane `j`:
  `pid₀·stride_out_batch + pid₁·stride_out_m + (t·BLOCK_N_SIZE + j)·stride_out_k`
  — the kernel's `out_offset`.

All three masks are the kernel's single `x_ptr_mask`:
`t·BLOCK_N_SIZE + j < N_SIZE`. -/
def rmsnormImplementationKernelIO (x w o : RegionName)
    (sxb sxm sxk srw sob som sok N B : Nat) (eps : ℝ) :
    StreamEmitMasked2DKernelIO₂ where
  kernel := rmsnorm_implementation x w o sxb sxm sxk srw sob som sok N B eps
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

/-- The guarded stream-level sum of squares: the pass-1 fold `var += xf*xf`
over the whole curried `x` stream, guarded by the kernel's window
(`t·B + e < N`) — the contract only pins `xs` on masked lanes, so the spec
must not read unmasked lanes. -/
noncomputable def rmsImplStreamSumSq (N B : Nat)
    (xs : Fin (rmsNumSteps N B) → Fin B → ℝ) : ℝ :=
  ∑ u : Fin (rmsNumSteps N B), ∑ e : Fin B,
    if u.val * B + e.val < N then xs u e ^ 2 else 0

/-- The stream-level RMS-norm spec (the genre's two-pass shape, in this
kernel's **division** spelling): output window `(t, j)` holds the step-`t`
`x` value divided by `std = √(Σ x²/N_SIZE + eps)` — the fold over the
*entire* stream — times the step-`t` `rms_w` value. Algebraically
`rmsnormWeightedYFullNSpec` with the `Fin N_SIZE` sum re-blocked to the
guarded stream double sum. -/
noncomputable def rmsImplStreamSpec (N B : Nat) (eps : ℝ)
    (xs ws : Fin (rmsNumSteps N B) → Fin B → ℝ)
    (t : Fin (rmsNumSteps N B)) (j : Fin B) : ℝ :=
  xs t j / Real.sqrt (rmsImplStreamSumSq N B xs / (N : ℝ) + eps) * ws t j

/-! ### The stream-lane spec bridge -/

private theorem rmsImpl_sum_blocks_lanes (BLOCK c : Nat) (hBLOCK : 0 < BLOCK)
    (H : Nat → ℝ) :
    (∑ b : Fin c, ∑ j : Fin BLOCK, H (b.val * BLOCK + j.val))
      = ∑ k : Fin (c * BLOCK), H k.val := by
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

private theorem rmsImpl_sum_fin_extend (N M : Nat) (hNM : N ≤ M) (f : Nat → ℝ) :
    (∑ k : Fin M, (if k.val < N then f k.val else 0)) = ∑ k : Fin N, f k.val := by
  rw [Fin.sum_univ_eq_sum_range (fun k => if k < N then f k else 0) M,
      Fin.sum_univ_eq_sum_range (fun k => f k) N]
  rw [← Finset.sum_subset (s₁ := range N) (s₂ := range M)
        (fun x hx => Finset.mem_range.mpr (lt_of_lt_of_le (Finset.mem_range.mp hx) hNM))
        (by intro k hk hknotN; simp only [Finset.mem_range] at hk hknotN; simp [Nat.not_lt.mp hknotN])]
  apply Finset.sum_congr rfl
  intro k hk; simp only [Finset.mem_range] at hk; simp [hk]

private theorem rmsImpl_sum_sq_mean (BLOCK c N : Nat) (hBLOCK : 0 < BLOCK)
    (hge : N ≤ c * BLOCK) (f : Nat → ℝ) :
    (∑ j : Fin BLOCK, ∑ b : Fin c, (if (b.val * BLOCK + j.val) < N then f (b.val * BLOCK + j.val) else 0))
      = ∑ k : Fin N, f k.val := by
  rw [Finset.sum_comm]
  rw [rmsImpl_sum_blocks_lanes BLOCK c hBLOCK (fun m => if m < N then f m else 0)]
  exact rmsImpl_sum_fin_extend N (c*BLOCK) hge f

/-- Under the stream pin, the guarded stream double sum **is** the exact
stack's `rmsVarFullNCarrier` (`Σ_{k<N} x[k]²`), re-blocking
`k ↔ (k/B, k%B)` via `rmsImpl_sum_sq_mean`. -/
private theorem rmsImplStreamSumSq_eq_carrier (x : RegionName) (s₀ : BlockState)
    (sxb sxm sxk N B : Nat) (hB : 0 < B)
    (xs : Fin (rmsNumSteps N B) → Fin B → ℝ)
    (hx : ∀ (t : Fin (rmsNumSteps N B)) (e : Fin B), t.val * B + e.val < N →
      s₀.readMem x (s₀.pids 0 * sxb + s₀.pids 1 * sxm + (t.val * B + e.val) * sxk)
        = xs t e) :
    rmsImplStreamSumSq N B xs
      = rmsVarFullNCarrier s₀ x sxb sxm sxk N B := by
  unfold rmsImplStreamSumSq rmsVarFullNCarrier
  rw [Finset.sum_comm,
    ← rmsImpl_sum_sq_mean B (rmsNumSteps N B) N hB (rmsNumSteps_mul_ge N B hB)
      (fun k => (s₀.readMem x (xColOffset s₀ sxb sxm sxk k)) ^ 2)]
  refine Finset.sum_congr rfl fun e _ => Finset.sum_congr rfl fun t _ => ?_
  by_cases h : t.val * B + e.val < N
  · rw [if_pos h, if_pos h, ← hx t e h]
    rfl
  · rw [if_neg h, if_neg h]

/-- Per-lane spec bridge: at a masked window `(t, j)` the stream spec **is**
the exact stack's `rmsnormWeightedYFullNSpec` at global lane `t·B + j`
(division form: `x / √(…)` matches the spec's `x · (1/√(…))` via
`div_eq_mul_inv` / `one_div`). -/
private theorem rmsImplStreamSpec_eq_fullNSpec (x w : RegionName)
    (s₀ : BlockState) (sxb sxm sxk srw N B : Nat) (eps : ℝ) (hB : 0 < B)
    (xs ws : Fin (rmsNumSteps N B) → Fin B → ℝ)
    (hx : ∀ (t : Fin (rmsNumSteps N B)) (e : Fin B), t.val * B + e.val < N →
      s₀.readMem x (s₀.pids 0 * sxb + s₀.pids 1 * sxm + (t.val * B + e.val) * sxk)
        = xs t e)
    (hw : ∀ (t : Fin (rmsNumSteps N B)) (e : Fin B), t.val * B + e.val < N →
      s₀.readMem w ((t.val * B + e.val) * srw) = ws t e)
    (t : Fin (rmsNumSteps N B)) (j : Fin B) (hj : t.val * B + j.val < N) :
    rmsImplStreamSpec N B eps xs ws t j
      = rmsnormWeightedYFullNSpec s₀ x w sxb sxm sxk srw N B eps
          ⟨t.val * B + j.val, hj⟩ := by
  unfold rmsImplStreamSpec rmsnormWeightedYFullNSpec rmsnormYFullNSpec rmsInvVarFullN
  rw [rmsImplStreamSumSq_eq_carrier x s₀ sxb sxm sxk N B hB xs hx,
    ← hx t j hj, ← hw t j hj, div_eq_mul_inv, one_div]

/-! ### Cast-free collapses and the covered fragment -/

/-- The erased `.to(tl.float32)` on `xf` is a `.real → .real` cast: exact
under every `R` — `R.cast .real .real` is the exact cast by the model's
defining `round_real`. The kernel's only `castFloat` is of this shape, so
the collapse lemmas below rewrite it to the exact cast. -/
private theorem Rcast_real_real (R : RoundingModel) :
    R.cast .real .real = FloatDType.cast .real .real := by
  funext v
  simp [RoundingModel.cast, FloatDType.cast]

/-- The prologue is cast-free: it steps identically under `stepStmtsR R`. -/
private theorem rmsImplPre_castFree (R : RoundingModel) (sxb sxm B : Nat)
    (t : BlockState) :
    stepStmtsR R (rmsVarPreLoop sxb sxm B) t
      = stepStmts (rmsVarPreLoop sxb sxm B) t := by
  simp only [rmsVarPreLoop, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def]
  rfl

/-- The pass-1 body is cast-free (masked `.real` load, the `.real → .real`
`xf` cast, a real multiply-add): it steps identically under `stepStmtsR R`,
so the exact var-loop invariant stack transports to `execR`. -/
private theorem rmsImplVarBody_castFree (R : RoundingModel) (x : RegionName)
    (sxk N B : Nat) (t : BlockState) :
    stepStmtsR R (rmsVarLoopBody x sxk N B) t
      = stepStmts (rmsVarLoopBody x sxk N B) t := by
  simp only [rmsVarLoopBody, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def, Rcast_real_real]
  rfl

/-- The reduce/sqrt tail is cast-free. -/
private theorem rmsImplPost_castFree (R : RoundingModel) (N B : Nat) (eps : ℝ)
    (t : BlockState) :
    stepStmtsR R (rmsStdPostLoop N B eps) t
      = stepStmts (rmsStdPostLoop N B eps) t := by
  simp only [rmsStdPostLoop, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def]
  rfl

/-- The pass-2 body is cast-free **including its in-loop masked `.real`
store**: `stepStmtR` delegates a `.real`-typed store to the exact
`writeMemTyped` (`writeMemTypedR R .real` is definitionally the exact
write), so the whole storing loop steps identically under `stepStmtsR R`
and the exact output-loop invariant stack transports to `execR`. -/
private theorem rmsImplOutBody_castFree (R : RoundingModel) (x w o : RegionName)
    (sxk srw sob som sok N B : Nat) (t : BlockState) :
    stepStmtsR R (rmsOutLoopBody x w o sxk srw sob som sok N B) t
      = stepStmts (rmsOutLoopBody x w o sxk srw sob som sok N B) t := by
  simp only [rmsOutLoopBody, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def, Rcast_real_real, BlockState.writeMemTypedR]
  rfl

/-- The full two-pass surface sits inside the flat-memory bridge's covered
fragment (`FlattenOk`; the two `forRange` clauses recurse into the
cast-free bodies). -/
theorem rmsnorm_implementation_flattenOk (x w o : RegionName)
    (sxb sxm sxk srw sob som sok N B : Nat) (eps : ℝ) :
    ((rmsnorm_implementation x w o sxb sxm sxk srw sob som sok N B eps).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  rw [rmsnorm_implementation_toAlg_body]
  simp [rmsVarPreLoop, rmsVarLoopBody, rmsStdPostLoop, rmsOutLoopBody,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

/-! ### Cell-level memory frames

The exact stack proves per-lane `readMem` values; the `⊨[R]` Hoare triple
additionally needs a per-**cell** (`BlockState.mem`) frame, so the
register-only segments get a generic assigns-don't-touch-memory lemma and
the storing pass-2 body gets a cell-level frame twin of
`rmsOutLoopBody_step_preserves_old_output`. -/

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

/-- Cell-level frame of a `Prop`-masked exact `writeMem` scatter `foldl`:
every cell not hit by an active lane is untouched (the cell-level sibling
of the library's `scatter_prop_masked_preserves_other_offset`, phrased at
`BlockState.mem` instead of `readMem`). -/
private theorem foldl_writeMem_prop_preserve_cell {α : Type}
    (region : RegionName) (ofn : α → Nat) (vfn : α → ℝ) (P : α → Prop)
    [DecidablePred P] (r : RegionName) (oo : Nat) (l : List α) (s : BlockState)
    (hnot : ∀ k ∈ l, P k → ¬(r = region ∧ oo = ofn k)) :
    (l.foldl (fun acc k =>
        if P k then acc.writeMem region (ofn k) (vfn k) else acc) s).mem r oo
      = s.mem r oo := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons]
      by_cases hP : P hd
      · rw [if_pos hP,
          ih _ (fun k hk hPk => hnot k (List.mem_cons_of_mem hd hk) hPk),
          BlockState.writeMem_mem]
        exact if_neg (hnot hd List.mem_cons_self hP)
      · rw [if_neg hP]
        exact ih _ (fun k hk hPk => hnot k (List.mem_cons_of_mem hd hk) hPk)

set_option maxHeartbeats 8000000 in
/-- **Cell-level frame of one pass-2 iteration** (the `mem` twin of
`rmsOutLoopBody_step_preserves_old_output`, same mega-`simp` walk): from
the output-loop invariant register pins, one storing body iteration leaves
every memory **cell** off the `{(out, outColOffset s0 col) : col < N_SIZE}`
write window untouched — the masked scatter store only hits active lanes
`i + j < N_SIZE`, whose offsets are `outColOffset s0 (i + j)`. -/
private theorem rmsImplOutBody_step_frame
    (s0 st st' : BlockState) (x w o : RegionName)
    (sxb sxm sxk srw sob som sok N B i : Nat) (eps : ℝ)
    (hPidBatch : st.regs .nat [] "pid_batch" = some (Tile.scalar (s0.pids 0)))
    (hPidM : st.regs .nat [] "pid_m" = some (Tile.scalar (s0.pids 1)))
    (hOffsetM : st.regs .nat [] "offset_m" =
      some (Tile.scalar (s0.pids 0 * sxb + s0.pids 1 * sxm)))
    (hBlockN : st.regs .nat [B] "block_n_size" =
      some { data := fun idx : TileIndex [B] => idx.1.val })
    (hStd : st.regs .real [] "std" =
      some (Tile.scalar (rmsStdFullNSpec s0 x sxb sxm sxk N B eps)))
    (hReadX : ∀ offset, st.readMem x offset = s0.readMem x offset)
    (hReadW : ∀ offset, st.readMem w offset = s0.readMem w offset)
    (hStep : stepStmts (rmsOutLoopBody x w o sxk srw sob som sok N B)
        (st.setReg "block_n_strart_ptr" .nat [] (Tile.scalar i)) = some st')
    (r : RegionName) (oo : Nat)
    (hcond : r ≠ o ∨ ∀ col : Fin N, oo ≠ outColOffset s0 sob som sok col.val) :
    st'.mem r oo = st.mem r oo := by
  unfold rmsOutLoopBody at hStep
  simp [stepStmts, stepStmt, evalOp, evalOp.eq_def, hPidBatch, hPidM, hOffsetM,
    hBlockN, hStd, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, NumericDType.add,
    NumericDType.mul, NumericDType.div, ComparableDType.lt, Option.bind,
    FloatDType.cast, hReadX, hReadW, outColOffset, rmsStdFullNSpec] at hStep
  subst st'
  refine Eq.trans (foldl_writeMem_prop_preserve_cell o _ _ _ r oo _ _ ?_) rfl
  intro lane _ hActive hbad
  rcases hcond with hne | hno
  · exact hne hbad.1
  · exact hno ⟨i + lane.1.val, hActive⟩ hbad.2

/-! ### Segment termination from an arbitrary launch state

The exact headline consumes an `exec = some` hypothesis; the `⊨[R]` Hoare
triple must *prove* termination, so the two straight-line segments get
none-case-refuting existence lemmas (the loops terminate through
`forRange_inv`). -/

/-- The prologue always steps (register-only assigns of total ops). -/
private theorem rmsImplPre_exists (sxb sxm B : Nat) (s : BlockState) :
    ∃ st, stepStmts (rmsVarPreLoop sxb sxm B) s = some st := by
  cases hStep : stepStmts (rmsVarPreLoop sxb sxm B) s with
  | none =>
      exfalso
      unfold rmsVarPreLoop at hStep
      simp [stepStmts, stepStmt, evalOp, evalOp.eq_def, Tile.bop,
        NumericDType.add, NumericDType.mul, BlockState.setReg, Option.bind] at hStep
  | some st => exact ⟨st, rfl⟩

/-- The reduce/sqrt tail always steps once the `var` register holds a tile. -/
private theorem rmsImplPost_exists (N B : Nat) (eps : ℝ) (stVar : BlockState)
    (acc : Tile .real [B])
    (hVarReg : stVar.regs .real [B] "var" = some acc) :
    ∃ st, stepStmts (rmsStdPostLoop N B eps) stVar = some st := by
  cases hStep : stepStmts (rmsStdPostLoop N B eps) stVar with
  | none =>
      exfalso
      unfold rmsStdPostLoop at hStep
      simp [stepStmts, stepStmt, evalOp, evalOp.eq_def, hVarReg, Tile.bop,
        Tile.uop, NumericDType.add, NumericDType.div, Option.bind,
        WithBot.realSqrt] at hStep
  | some st => exact ⟨st, rfl⟩

/-! ### The `TraceSafeR` walk: cast-free index ops -/

/-- `evalOpR` = `evalOp` on the cast-free nat/bool index ops of the two
passes (register refs and nat arithmetic — `R` never enters). -/
private theorem rmsImpl_offsnR_eq (R : RoundingModel) (B : Nat) (s : BlockState) :
    evalOpR R (Op.add NumericDType.nat Broadcast.scalarL
        (Op.ref .nat [] "block_n_strart_ptr") (Op.ref .nat [B] "block_n_size")) s
      = evalOp (Op.add NumericDType.nat Broadcast.scalarL
        (Op.ref .nat [] "block_n_strart_ptr") (Op.ref .nat [B] "block_n_size")) s := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

private theorem rmsImpl_maskR_eq (R : RoundingModel) (N B : Nat) (s : BlockState) :
    evalOpR R (Op.lt ComparableDType.nat Broadcast.scalarR
        (Op.ref .nat [B] "offset_n") (Op.constNat N)) s
      = evalOp (Op.lt ComparableDType.nat Broadcast.scalarR
        (Op.ref .nat [B] "offset_n") (Op.constNat N)) s := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

private theorem rmsImpl_xaddrR_eq (R : RoundingModel) (sxk B : Nat) (s : BlockState) :
    evalOpR R (Op.add NumericDType.nat Broadcast.scalarL (Op.ref .nat [] "offset_m")
        (Op.mul NumericDType.nat Broadcast.scalarR (Op.ref .nat [B] "offset_n") (Op.constNat sxk))) s
      = evalOp (Op.add NumericDType.nat Broadcast.scalarL (Op.ref .nat [] "offset_m")
        (Op.mul NumericDType.nat Broadcast.scalarR (Op.ref .nat [B] "offset_n") (Op.constNat sxk))) s := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

private theorem rmsImpl_waddrR_eq (R : RoundingModel) (srw B : Nat) (s : BlockState) :
    evalOpR R (Op.mul NumericDType.nat Broadcast.scalarR
        (Op.ref .nat [B] "offset_n") (Op.constNat srw)) s
      = evalOp (Op.mul NumericDType.nat Broadcast.scalarR
        (Op.ref .nat [B] "offset_n") (Op.constNat srw)) s := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

private theorem rmsImpl_outoffR_eq (R : RoundingModel) (sob som sok B : Nat) (s : BlockState) :
    evalOpR R (Op.add NumericDType.nat Broadcast.scalarL
        (Op.add NumericDType.nat Broadcast.nil
          (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "pid_batch") (Op.constNat sob))
          (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "pid_m") (Op.constNat som)))
        (Op.mul NumericDType.nat Broadcast.scalarR (Op.ref .nat [B] "offset_n") (Op.constNat sok))) s
      = evalOp (Op.add NumericDType.nat Broadcast.scalarL
        (Op.add NumericDType.nat Broadcast.nil
          (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "pid_batch") (Op.constNat sob))
          (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "pid_m") (Op.constNat som)))
        (Op.mul NumericDType.nat Broadcast.scalarR (Op.ref .nat [B] "offset_n") (Op.constNat sok))) s := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

/-! ### The `TraceSafeR` walk: per-lane index-op values -/

/-- The `offset_n` tile at raw counter `i`: `i + j` per lane. -/
private theorem rmsImpl_offsn_eval (B i : Nat) (s : BlockState)
    (hbN : s.regs .nat [B] "block_n_size" = some (Tile.vec (fun j : Fin B => j.val))) :
    evalOp (Op.add NumericDType.nat Broadcast.scalarL
        (Op.ref .nat [] "block_n_strart_ptr") (Op.ref .nat [B] "block_n_size"))
        (s.setReg "block_n_strart_ptr" .nat [] (Tile.scalar i))
      = some (Tile.vec (fun j : Fin B => i + j.val)) := by
  rw [evalOp_add, evalOp_ref_setReg_same,
    evalOp_ref_setReg_ne_name _ _ _ _ _ _ _ _
      (show ("block_n_size":RegName) ≠ "block_n_strart_ptr" by decide),
    evalOp_ref, hbN]
  apply congrArg some
  ext j
  simp only [Tile.bop_data, Broadcast.leftIndex_scalarL, Broadcast.rightIndex_scalarL,
    Tile.scalar, Tile.vec, NumericDType.add]

/-- The `x_ptr_mask` tile at raw counter `i`: `decide (i + j < N)` per lane. -/
private theorem rmsImpl_mask_eval (N B i : Nat) (s : BlockState)
    (hoffsn : s.regs .nat [B] "offset_n" = some (Tile.vec (fun j : Fin B => i + j.val))) :
    evalOp (Op.lt ComparableDType.nat Broadcast.scalarR
        (Op.ref .nat [B] "offset_n") (Op.constNat N)) s
      = some (Tile.vec (fun j : Fin B => decide (i + j.val < N))) := by
  rw [evalOp_lt, evalOp_ref, hoffsn, evalOp_constNat]
  apply congrArg some
  ext j
  simp only [Tile.cop_data, Tile.vec_data, Tile.scalar_data, Broadcast.leftIndex_scalarR,
    Broadcast.rightIndex_scalarR, ComparableDType.lt, decide_eq_decide]

/-- The per-lane value of the `x` load's address op (`offset_m + offset_n * sxk`). -/
private theorem rmsImpl_xaddr_eval (sxk B base i : Nat) (s : BlockState)
    (hom : s.regs .nat [] "offset_m" = some (Tile.scalar base))
    (hoffsn : s.regs .nat [B] "offset_n" = some (Tile.vec (fun j : Fin B => i + j.val))) :
    evalOp (Op.add NumericDType.nat Broadcast.scalarL (Op.ref .nat [] "offset_m")
        (Op.mul NumericDType.nat Broadcast.scalarR (Op.ref .nat [B] "offset_n") (Op.constNat sxk))) s
      = some (Tile.vec (fun j : Fin B => base + (i + j.val) * sxk)) := by
  rw [evalOp_add, evalOp_ref, hom, evalOp_mul, evalOp_ref, hoffsn, evalOp_constNat]
  apply congrArg some
  ext j
  simp only [Tile.bop_data, Tile.vec, Tile.scalar, Broadcast.leftIndex_scalarL,
    Broadcast.rightIndex_scalarL, Broadcast.leftIndex_scalarR,
    Broadcast.rightIndex_scalarR, NumericDType.add, NumericDType.mul]

/-- The per-lane value of the `rms_w` load's address op (`offset_n * srw`). -/
private theorem rmsImpl_waddr_eval (srw B i : Nat) (s : BlockState)
    (hoffsn : s.regs .nat [B] "offset_n" = some (Tile.vec (fun j : Fin B => i + j.val))) :
    evalOp (Op.mul NumericDType.nat Broadcast.scalarR
        (Op.ref .nat [B] "offset_n") (Op.constNat srw)) s
      = some (Tile.vec (fun j : Fin B => (i + j.val) * srw)) := by
  rw [evalOp_mul, evalOp_ref, hoffsn, evalOp_constNat]
  apply congrArg some
  ext j
  simp only [Tile.bop_data, Tile.vec, Tile.scalar, Broadcast.leftIndex_scalarR,
    Broadcast.rightIndex_scalarR, NumericDType.mul]

/-- The per-lane value of the store's `out_offset` op, directly in launch
(`s0`)-anchored form via the invariant's `pid_batch` / `pid_m` pins. -/
private theorem rmsImpl_outoff_eval (sob som sok B i : Nat) (s0 s : BlockState)
    (hpb : s.regs .nat [] "pid_batch" = some (Tile.scalar (s0.pids 0)))
    (hpm : s.regs .nat [] "pid_m" = some (Tile.scalar (s0.pids 1)))
    (hoffsn : s.regs .nat [B] "offset_n" = some (Tile.vec (fun j : Fin B => i + j.val))) :
    evalOp (Op.add NumericDType.nat Broadcast.scalarL
        (Op.add NumericDType.nat Broadcast.nil
          (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "pid_batch") (Op.constNat sob))
          (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "pid_m") (Op.constNat som)))
        (Op.mul NumericDType.nat Broadcast.scalarR (Op.ref .nat [B] "offset_n") (Op.constNat sok))) s
      = some (Tile.vec (fun j : Fin B => outColOffset s0 sob som sok (i + j.val))) := by
  rw [evalOp_add, evalOp_add, evalOp_mul, evalOp_ref, hpb, evalOp_constNat,
    evalOp_mul, evalOp_ref, hpm, evalOp_constNat, evalOp_mul, evalOp_ref,
    hoffsn, evalOp_constNat]
  apply congrArg some
  ext j
  simp only [Tile.bop_data, Tile.vec, Tile.scalar, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.add, NumericDType.mul, outColOffset]

set_option maxHeartbeats 4000000 in
/-- Per-iteration `TraceSafeListR` for the pass-1 body: the index/mask
assigns, the `xf` cast and the multiply-add are register-only; the masked
`x` load's **active** lanes are exactly the skin's `mask1` window at step
`i / B`, in bounds by the `read1` window bound (instantiated at raw counter
`i`). -/
private theorem rmsImplVarBodySafeR (R : RoundingModel) (bounds : RegionBounds)
    (x : RegionName) (sxb sxm sxk N B : Nat)
    (s0 st : BlockState) (i : Nat)
    (hbN : st.regs .nat [B] "block_n_size" = some (Tile.vec (fun j : Fin B => j.val)))
    (hom : st.regs .nat [] "offset_m" = some (Tile.scalar (s0.pids 0 * sxb + s0.pids 1 * sxm)))
    (hbx : ∀ j : Fin B, i + j.val < N →
      s0.pids 0 * sxb + s0.pids 1 * sxm + (i + j.val) * sxk < bounds x) :
    Stmt.TraceSafeListR R bounds (rmsVarLoopBody x sxk N B)
      (st.setReg "block_n_strart_ptr" .nat [] (Tile.scalar i)) := by
  unfold rmsVarLoopBody
  -- offset_n
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun t1 ht1 => ?_)
  rw [stepStmtR_assign_eq_some
    ((rmsImpl_offsnR_eq R B _).trans (rmsImpl_offsn_eval B i st hbN))] at ht1
  obtain rfl := Option.some.inj ht1
  set q1 := (st.setReg "block_n_strart_ptr" .nat [] (Tile.scalar i)).setReg
    "offset_n" .nat [B] (Tile.vec (fun j : Fin B => i + j.val)) with hq1
  have hoffsn1 : q1.regs .nat [B] "offset_n"
      = some (Tile.vec (fun j : Fin B => i + j.val)) := by simp [hq1]
  -- x_ptr_mask
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun t2 ht2 => ?_)
  rw [stepStmtR_assign_eq_some
    ((rmsImpl_maskR_eq R N B _).trans (rmsImpl_mask_eval N B i q1 hoffsn1))] at ht2
  obtain rfl := Option.some.inj ht2
  set q2 := q1.setReg "x_ptr_mask" .bool [B]
    (Tile.vec (fun j : Fin B => decide (i + j.val < N))) with hq2
  have hoffsn2 : q2.regs .nat [B] "offset_n"
      = some (Tile.vec (fun j : Fin B => i + j.val)) := by
    rw [hq2]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
      (show ("offset_n":RegName) ≠ "x_ptr_mask" by decide)]
    exact hoffsn1
  have hmask2 : q2.regs .bool [B] "x_ptr_mask"
      = some (Tile.vec (fun j : Fin B => decide (i + j.val < N))) := by
    rw [hq2]; simp
  have hom2 : q2.regs .nat [] "offset_m"
      = some (Tile.scalar (s0.pids 0 * sxb + s0.pids 1 * sxm)) := by
    rw [hq2, hq1]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("offset_m":RegName) ≠ "x_ptr_mask" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("offset_m":RegName) ≠ "offset_n" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("offset_m":RegName) ≠ "block_n_strart_ptr" by decide)]
    exact hom
  -- the masked x load
  refine Stmt.TraceSafeListR.cons_intro ?_ (fun t3 ht3 => ?_)
  · simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR,
      MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR,
      and_true, true_and, and_self]
    intro offsets hoffsets idx hactive
    rw [rmsImpl_xaddrR_eq,
      rmsImpl_xaddr_eval sxk B (s0.pids 0 * sxb + s0.pids 1 * sxm) i q2 hom2 hoffsn2] at hoffsets
    obtain rfl := Option.some.inj hoffsets
    obtain ⟨masks, hm, hmi⟩ := hactive
    rw [evalOpR_ref, hmask2] at hm
    obtain rfl := Option.some.inj hm
    have hlt : i + idx.1.val < N := by simpa [Tile.vec] using hmi
    simpa [Region.cast_id, Tile.vec] using hbx idx.1 hlt
  · obtain ⟨v3, -, rfl⟩ := stepStmtR_assign_inv ht3
    -- xf cast: register-only
    refine Stmt.TraceSafeListR.cons_intro
      (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun t4 ht4 => ?_)
    obtain ⟨v4, -, rfl⟩ := stepStmtR_assign_inv ht4
    -- var accumulate: register-only
    exact Stmt.TraceSafeListR.cons_intro
      (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
      (fun _ _ => Stmt.TraceSafeListR.nil_intro)

set_option maxHeartbeats 8000000 in
/-- Per-iteration `TraceSafeListR` for the pass-2 body: the index/mask/
scaling assigns are register-only; the masked `rms_w_offset` / `x` loads'
and the masked store's **active** lanes are the skin's `mask2` / `mask1` /
`writeMask` windows at step `i / B`, in bounds by the corresponding window
bounds (instantiated at raw counter `i`). -/
private theorem rmsImplOutBodySafeR (R : RoundingModel) (bounds : RegionBounds)
    (x w o : RegionName) (sxb sxm sxk srw sob som sok N B : Nat)
    (s0 st : BlockState) (i : Nat)
    (hpb : st.regs .nat [] "pid_batch" = some (Tile.scalar (s0.pids 0)))
    (hpm : st.regs .nat [] "pid_m" = some (Tile.scalar (s0.pids 1)))
    (hbN : st.regs .nat [B] "block_n_size" = some (Tile.vec (fun j : Fin B => j.val)))
    (hom : st.regs .nat [] "offset_m" = some (Tile.scalar (s0.pids 0 * sxb + s0.pids 1 * sxm)))
    (hbx : ∀ j : Fin B, i + j.val < N →
      s0.pids 0 * sxb + s0.pids 1 * sxm + (i + j.val) * sxk < bounds x)
    (hbw : ∀ j : Fin B, i + j.val < N → (i + j.val) * srw < bounds w)
    (hbo : ∀ j : Fin B, i + j.val < N →
      s0.pids 0 * sob + s0.pids 1 * som + (i + j.val) * sok < bounds o) :
    Stmt.TraceSafeListR R bounds (rmsOutLoopBody x w o sxk srw sob som sok N B)
      (st.setReg "block_n_strart_ptr" .nat [] (Tile.scalar i)) := by
  unfold rmsOutLoopBody
  -- offset_n
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun t1 ht1 => ?_)
  rw [stepStmtR_assign_eq_some
    ((rmsImpl_offsnR_eq R B _).trans (rmsImpl_offsn_eval B i st hbN))] at ht1
  obtain rfl := Option.some.inj ht1
  set q1 := (st.setReg "block_n_strart_ptr" .nat [] (Tile.scalar i)).setReg
    "offset_n" .nat [B] (Tile.vec (fun j : Fin B => i + j.val)) with hq1
  have hoffsn1 : q1.regs .nat [B] "offset_n"
      = some (Tile.vec (fun j : Fin B => i + j.val)) := by simp [hq1]
  -- x_ptr_mask
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun t2 ht2 => ?_)
  rw [stepStmtR_assign_eq_some
    ((rmsImpl_maskR_eq R N B _).trans (rmsImpl_mask_eval N B i q1 hoffsn1))] at ht2
  obtain rfl := Option.some.inj ht2
  set q2 := q1.setReg "x_ptr_mask" .bool [B]
    (Tile.vec (fun j : Fin B => decide (i + j.val < N))) with hq2
  have hoffsn2 : q2.regs .nat [B] "offset_n"
      = some (Tile.vec (fun j : Fin B => i + j.val)) := by
    rw [hq2]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
      (show ("offset_n":RegName) ≠ "x_ptr_mask" by decide)]
    exact hoffsn1
  have hmask2 : q2.regs .bool [B] "x_ptr_mask"
      = some (Tile.vec (fun j : Fin B => decide (i + j.val < N))) := by
    rw [hq2]; simp
  have hom2 : q2.regs .nat [] "offset_m"
      = some (Tile.scalar (s0.pids 0 * sxb + s0.pids 1 * sxm)) := by
    rw [hq2, hq1]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("offset_m":RegName) ≠ "x_ptr_mask" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("offset_m":RegName) ≠ "offset_n" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("offset_m":RegName) ≠ "block_n_strart_ptr" by decide)]
    exact hom
  have hpb2 : q2.regs .nat [] "pid_batch" = some (Tile.scalar (s0.pids 0)) := by
    rw [hq2, hq1]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("pid_batch":RegName) ≠ "x_ptr_mask" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("pid_batch":RegName) ≠ "offset_n" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("pid_batch":RegName) ≠ "block_n_strart_ptr" by decide)]
    exact hpb
  have hpm2 : q2.regs .nat [] "pid_m" = some (Tile.scalar (s0.pids 1)) := by
    rw [hq2, hq1]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("pid_m":RegName) ≠ "x_ptr_mask" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("pid_m":RegName) ≠ "offset_n" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("pid_m":RegName) ≠ "block_n_strart_ptr" by decide)]
    exact hpm
  -- the masked rms_w_offset load
  refine Stmt.TraceSafeListR.cons_intro ?_ (fun t3 ht3 => ?_)
  · simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR,
      MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR,
      and_true, true_and, and_self]
    intro offsets hoffsets idx hactive
    rw [rmsImpl_waddrR_eq, rmsImpl_waddr_eval srw B i q2 hoffsn2] at hoffsets
    obtain rfl := Option.some.inj hoffsets
    obtain ⟨masks, hm, hmi⟩ := hactive
    rw [evalOpR_ref, hmask2] at hm
    obtain rfl := Option.some.inj hm
    have hlt : i + idx.1.val < N := by simpa [Tile.vec] using hmi
    simpa [Region.cast_id, Tile.vec] using hbw idx.1 hlt
  · obtain ⟨v3, -, rfl⟩ := stepStmtR_assign_inv ht3
    set q3 := q2.setReg "rms_w_offset" .real [B] v3 with hq3
    have hoffsn3 : q3.regs .nat [B] "offset_n"
        = some (Tile.vec (fun j : Fin B => i + j.val)) := by
      rw [hq3]
      simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("offset_n":RegName) ≠ "rms_w_offset" by decide)]
      exact hoffsn2
    have hmask3 : q3.regs .bool [B] "x_ptr_mask"
        = some (Tile.vec (fun j : Fin B => decide (i + j.val < N))) := by
      rw [hq3]
      simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("x_ptr_mask":RegName) ≠ "rms_w_offset" by decide)]
      exact hmask2
    have hom3 : q3.regs .nat [] "offset_m"
        = some (Tile.scalar (s0.pids 0 * sxb + s0.pids 1 * sxm)) := by
      rw [hq3]
      simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("offset_m":RegName) ≠ "rms_w_offset" by decide)]
      exact hom2
    -- the masked x load
    refine Stmt.TraceSafeListR.cons_intro ?_ (fun t4 ht4 => ?_)
    · simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR,
        MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR,
        and_true, true_and, and_self]
      intro offsets hoffsets idx hactive
      rw [rmsImpl_xaddrR_eq,
        rmsImpl_xaddr_eval sxk B (s0.pids 0 * sxb + s0.pids 1 * sxm) i q3 hom3 hoffsn3] at hoffsets
      obtain rfl := Option.some.inj hoffsets
      obtain ⟨masks, hm, hmi⟩ := hactive
      rw [evalOpR_ref, hmask3] at hm
      obtain rfl := Option.some.inj hm
      have hlt : i + idx.1.val < N := by simpa [Tile.vec] using hmi
      simpa [Region.cast_id, Tile.vec] using hbx idx.1 hlt
    · obtain ⟨v4, -, rfl⟩ := stepStmtR_assign_inv ht4
      set q4 := q3.setReg "x" .real [B] v4 with hq4
      -- x_new (register-only)
      refine Stmt.TraceSafeListR.cons_intro
        (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun t5 ht5 => ?_)
      obtain ⟨v5, -, rfl⟩ := stepStmtR_assign_inv ht5
      set q5 := q4.setReg "x_new" .real [B] v5 with hq5
      -- out (register-only)
      refine Stmt.TraceSafeListR.cons_intro
        (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun t6 ht6 => ?_)
      obtain ⟨v6, -, rfl⟩ := stepStmtR_assign_inv ht6
      set q6 := q5.setReg "out" .real [B] v6 with hq6
      -- out_offset
      have hpb6 : q6.regs .nat [] "pid_batch" = some (Tile.scalar (s0.pids 0)) := by
        rw [hq6, hq5, hq4, hq3]
        simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
            (show ("pid_batch":RegName) ≠ "out" by decide),
          BlockState.setReg_ne_name _ _ _ _ _ _ _ _
            (show ("pid_batch":RegName) ≠ "x_new" by decide),
          BlockState.setReg_ne_name _ _ _ _ _ _ _ _
            (show ("pid_batch":RegName) ≠ "x" by decide),
          BlockState.setReg_ne_name _ _ _ _ _ _ _ _
            (show ("pid_batch":RegName) ≠ "rms_w_offset" by decide)]
        exact hpb2
      have hpm6 : q6.regs .nat [] "pid_m" = some (Tile.scalar (s0.pids 1)) := by
        rw [hq6, hq5, hq4, hq3]
        simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
            (show ("pid_m":RegName) ≠ "out" by decide),
          BlockState.setReg_ne_name _ _ _ _ _ _ _ _
            (show ("pid_m":RegName) ≠ "x_new" by decide),
          BlockState.setReg_ne_name _ _ _ _ _ _ _ _
            (show ("pid_m":RegName) ≠ "x" by decide),
          BlockState.setReg_ne_name _ _ _ _ _ _ _ _
            (show ("pid_m":RegName) ≠ "rms_w_offset" by decide)]
        exact hpm2
      have hoffsn6 : q6.regs .nat [B] "offset_n"
          = some (Tile.vec (fun j : Fin B => i + j.val)) := by
        rw [hq6, hq5, hq4]
        simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
            (show ("offset_n":RegName) ≠ "out" by decide),
          BlockState.setReg_ne_name _ _ _ _ _ _ _ _
            (show ("offset_n":RegName) ≠ "x_new" by decide),
          BlockState.setReg_ne_name _ _ _ _ _ _ _ _
            (show ("offset_n":RegName) ≠ "x" by decide)]
        exact hoffsn3
      have hmask6 : q6.regs .bool [B] "x_ptr_mask"
          = some (Tile.vec (fun j : Fin B => decide (i + j.val < N))) := by
        rw [hq6, hq5, hq4]
        simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
            (show ("x_ptr_mask":RegName) ≠ "out" by decide),
          BlockState.setReg_ne_name _ _ _ _ _ _ _ _
            (show ("x_ptr_mask":RegName) ≠ "x_new" by decide),
          BlockState.setReg_ne_name _ _ _ _ _ _ _ _
            (show ("x_ptr_mask":RegName) ≠ "x" by decide)]
        exact hmask3
      refine Stmt.TraceSafeListR.cons_intro
        (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun t7 ht7 => ?_)
      rw [stepStmtR_assign_eq_some ((rmsImpl_outoffR_eq R sob som sok B q6).trans
        (rmsImpl_outoff_eval sob som sok B i s0 q6 hpb6 hpm6 hoffsn6))] at ht7
      obtain rfl := Option.some.inj ht7
      set q7 := q6.setReg "out_offset" .nat [B]
        (Tile.vec (fun j : Fin B => outColOffset s0 sob som sok (i + j.val))) with hq7
      have houtoff7 : q7.regs .nat [B] "out_offset"
          = some (Tile.vec (fun j : Fin B => outColOffset s0 sob som sok (i + j.val))) := by
        rw [hq7]; simp
      have hmask7 : q7.regs .bool [B] "x_ptr_mask"
          = some (Tile.vec (fun j : Fin B => decide (i + j.val < N))) := by
        rw [hq7]
        simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
          (show ("x_ptr_mask":RegName) ≠ "out_offset" by decide)]
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
      have hlt : i + idx.1.val < N := by simpa [Tile.vec] using hmi
      simpa [Region.cast_id, Tile.vec, outColOffset] using hbo idx.1 hlt

set_option maxHeartbeats 8000000 in
/-- **The `TraceSafeR` walk for the whole two-pass kernel** — driven by
`Stmt.forRangeTraceSafeR_inv` over the proven `rmsVarLoopContextInvariant`
for pass 1 and `rmsOutLoopContextInvariant` (enriched with the counter
alignment `off % B = 0`) for pass 2, with the counter advancing by the
loops' stride `BLOCK_N_SIZE`. The three bound groups are the skin's
`read1`/`read2`/`write` windows; the alignment converts the raw counter
into the step index `i / B < ⌈N/B⌉` the windows are phrased over, and
`hxo`/`hwo`/`hsok` feed the output-loop step (its readback clause needs the
store not to clobber the streamed inputs and the write window to be
injective). -/
private theorem rmsImpl_traceSafeR (R : RoundingModel) (bounds : RegionBounds)
    (x w o : RegionName) (sxb sxm sxk srw sob som sok N B : Nat) (eps : ℝ)
    (hB : 0 < B) (hxo : x ≠ o) (hwo : w ≠ o) (hsok : 0 < sok)
    (s : BlockState)
    (hbx : ∀ (t : Fin (rmsNumSteps N B)) (j : Fin B), t.val * B + j.val < N →
      s.pids 0 * sxb + s.pids 1 * sxm + (t.val * B + j.val) * sxk < bounds x)
    (hbw : ∀ (t : Fin (rmsNumSteps N B)) (j : Fin B), t.val * B + j.val < N →
      (t.val * B + j.val) * srw < bounds w)
    (hbo : ∀ (t : Fin (rmsNumSteps N B)) (j : Fin B), t.val * B + j.val < N →
      s.pids 0 * sob + s.pids 1 * som + (t.val * B + j.val) * sok < bounds o) :
    ((rmsnorm_implementation x w o sxb sxm sxk srw sob som sok N B eps).toAlgKernel).TraceSafeR R bounds s := by
  have hStepNe : B ≠ 0 := Nat.ne_of_gt hB
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
  rw [rmsnorm_implementation_toAlg_body]
  simp only [List.append_assoc, List.singleton_append]
  refine Stmt.TraceSafeListR.append_intro _ _ ?_ ?_
  · -- prologue: register-only assigns, safe at every state
    refine Stmt.TraceSafeListR.of_forall _ _ ?_
    intro stmt hst s'
    simp only [rmsVarPreLoop, List.mem_cons, List.not_mem_nil, or_false] at hst
    rcases hst with rfl | rfl | rfl | rfl | rfl <;>
      simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]
  · intro s1 hs1
    rw [rmsImplPre_castFree R sxb sxm B s] at hs1
    have hCtx0 :=
      rmsVarLoopContextInvariant_init_of_preloop s s1 x sxb sxm sxk N B hs1
    obtain ⟨hPids0, hPb0, hPm0, hRw0⟩ :=
      rmsVarPreLoop_step_preserves_pids_read s s1 w sxb sxm B hs1
    refine Stmt.TraceSafeListR.cons_intro ?_ ?_
    · -- pass 1 (invariant principle over `rmsVarLoopContextInvariant`)
      simp only [Stmt.TraceSafeR]
      refine Stmt.forRangeTraceSafeR_inv R bounds "block_n_strart_ptr" N B
        (rmsVarLoopBody x sxk N B)
        (fun off st => rmsVarLoopContextInvariant s x sxb sxm sxk N B off st)
        ?_ 0 s1 hCtx0
      intro i stt hi hP
      have hbN' : stt.regs .nat [B] "block_n_size"
          = some (Tile.vec (fun j : Fin B => j.val)) := hP.2.2.1
      refine ⟨rmsImplVarBodySafeR R bounds x sxb sxm sxk N B s stt i hbN' hP.2.1
        (fun j hj => hbx' i (Nat.dvd_of_mod_eq_zero hP.1.1) j hj), ?_⟩
      obtain ⟨st', hstep, hCtx'⟩ :=
        rmsVarLoopContextInvariant_body_step_exists s stt x sxb sxm sxk N B i hP
      exact ⟨st', by rw [rmsImplVarBody_castFree]; exact hstep, hCtx'⟩
    · intro s2 hs2
      -- identify the pass-1 exit state via the exact loop run
      obtain ⟨finalV, sv, hsv, hVarFinal, hPV⟩ :=
        forRange_inv (idx := "block_n_strart_ptr") (start := 0) (stop := N) (step := B)
          (body := rmsVarLoopBody x sxk N B)
          (P := fun off st =>
            rmsVarLoopContextInvariant s x sxb sxm sxk N B off st ∧
            st.pids = s.pids ∧
            st.regs .nat [] "pid_batch" = some (Tile.scalar (s.pids 0)) ∧
            st.regs .nat [] "pid_m" = some (Tile.scalar (s.pids 1)) ∧
            (∀ offset, st.readMem w offset = s.readMem w offset))
          hStepNe ⟨hCtx0, hPids0, hPb0, hPm0, hRw0⟩
          (fun i stt hlt hP => by
            obtain ⟨st', hstep, hCtx'⟩ :=
              rmsVarLoopContextInvariant_body_step_exists s stt x sxb sxm sxk N B i hP.1
            obtain ⟨hPidsS, hPbS, hPmS, hRwS⟩ :=
              rmsVarLoopBody_step_preserves_pids_read s stt st' x w sxb sxm sxk N B i hP.1 hstep
            refine ⟨st', hstep, hCtx', ?_, ?_, ?_, ?_⟩
            · rw [hPidsS]; exact hP.2.1
            · rw [hPbS]; exact hP.2.2.1
            · rw [hPmS]; exact hP.2.2.2.1
            · intro offset; rw [hRwS offset]; exact hP.2.2.2.2 offset)
      rw [stepStmtR_forRange,
        stepForRangeAuxR_castFree R _ (rmsImplVarBody_castFree R x sxk N B) "block_n_strart_ptr",
        ← stepForRangeAux.forRange_unfold, hsv] at hs2
      obtain rfl := Option.some.inj hs2
      obtain ⟨hVarCtx, hVarPids, hVarPb, hVarPm, hVarRw⟩ := hPV
      refine Stmt.TraceSafeListR.append_intro _ _ ?_ ?_
      · -- reduce/sqrt tail: register-only assigns
        refine Stmt.TraceSafeListR.of_forall _ _ ?_
        intro stmt hst s'
        simp only [rmsStdPostLoop, List.mem_cons, List.not_mem_nil, or_false] at hst
        rcases hst with rfl | rfl <;> simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]
      · intro s3 hs3
        rw [rmsImplPost_castFree R N B eps sv] at hs3
        have hOutBase :=
          rmsStdPostLoop_step_to_out_init s sv s3 x w o sxb sxm sxk srw sob som sok
            N B finalV eps hVarCtx hVarFinal hB hVarPids hVarPb hVarPm hVarRw hs3
        -- pass 2 (invariant principle over `rmsOutLoopContextInvariant` + alignment)
        refine Stmt.TraceSafeListR.cons_intro ?_ (fun s4 _ => Stmt.TraceSafeListR.nil_intro)
        simp only [Stmt.TraceSafeR]
        refine Stmt.forRangeTraceSafeR_inv R bounds "block_n_strart_ptr" N B
          (rmsOutLoopBody x w o sxk srw sob som sok N B)
          (fun off st => off % B = 0 ∧
            rmsOutLoopContextInvariant s x w o sxb sxm sxk srw sob som sok N B off eps st)
          ?_ 0 s3 ⟨Nat.zero_mod B, hOutBase⟩
        intro i stt hi hQ
        obtain ⟨hMod, hCtx⟩ := hQ
        have hQd := hCtx
        obtain ⟨hOutInv, hPidsQ, hPbQ, hPmQ, hOmQ, hBnQ, hStdQ, hRXQ, hRWQ⟩ := hQd
        have hbN' : stt.regs .nat [B] "block_n_size"
            = some (Tile.vec (fun j : Fin B => j.val)) := hBnQ
        refine ⟨rmsImplOutBodySafeR R bounds x w o sxb sxm sxk srw sob som sok N B s stt i
            hPbQ hPmQ hbN' hOmQ
            (fun j hj => hbx' i (Nat.dvd_of_mod_eq_zero hMod) j hj)
            (fun j hj => hbw' i (Nat.dvd_of_mod_eq_zero hMod) j hj)
            (fun j hj => hbo' i (Nat.dvd_of_mod_eq_zero hMod) j hj), ?_⟩
        obtain ⟨st', hstep, hCtx'⟩ :=
          rmsOutLoopContextInvariant_body_step_exists s stt x w o sxb sxm sxk srw
            sob som sok N B i eps hCtx hxo hwo
            (outColOffset_fin_injective_of_stride_out_k_pos s sob som sok N hsok)
            (outColOffset_chunk_injective_of_stride_out_k_pos s sob som sok B i hsok)
        exact ⟨st', by rw [rmsImplOutBody_castFree]; exact hstep,
          rmsVarLoopOffset_mod_step i B hMod, hCtx'⟩

/-! ### The rounded Hoare triple (`hrun`) -/

set_option maxHeartbeats 8000000 in
/-- Termination, per-lane values and the per-cell frame of the whole
two-pass kernel under `execR R`, from an **arbitrary** launch state: the
exact `rmsVarPreLoop` / `rmsVarLoopContextInvariant` / `rmsStdPostLoop` /
`rmsOutLoopContextInvariant` stack runs verbatim (both passes are
cast-free, so `execR R` collapses onto the exact stepper), extended with
the per-segment memory frames. -/
private theorem rmsImpl_runR (R : RoundingModel) (x w o : RegionName)
    (sxb sxm sxk srw sob som sok N B : Nat) (eps : ℝ)
    (hB : 0 < B) (hxo : x ≠ o) (hwo : w ≠ o) (hsok : 0 < sok)
    (s₀ : BlockState) :
    ∃ sfin,
      execR R (rmsnorm_implementation x w o sxb sxm sxk srw sob som sok N B eps).toAlgKernel s₀
        = some sfin
      ∧ (∀ i : Fin N, sfin.readMem o (outColOffset s₀ sob som sok i.val)
          = rmsnormWeightedYFullNSpec s₀ x w sxb sxm sxk srw N B eps i)
      ∧ (∀ r oo, (r ≠ o ∨ ∀ i : Fin N, oo ≠ outColOffset s₀ sob som sok i.val) →
          sfin.mem r oo = s₀.mem r oo) := by
  have hStepNe : B ≠ 0 := Nat.ne_of_gt hB
  -- prologue
  obtain ⟨stPre, hPre⟩ := rmsImplPre_exists sxb sxm B s₀
  have hPreCtx :=
    rmsVarLoopContextInvariant_init_of_preloop s₀ stPre x sxb sxm sxk N B hPre
  obtain ⟨hPrePids, hPrePb, hPrePm, hPreRw⟩ :=
    rmsVarPreLoop_step_preserves_pids_read s₀ stPre w sxb sxm B hPre
  have hPreMem : stPre.mem = s₀.mem :=
    stepStmts_assigns_mem (rmsVarPreLoop sxb sxm B)
      (by
        intro stmt hst
        simp only [rmsVarPreLoop, List.mem_cons, List.not_mem_nil, or_false] at hst
        rcases hst with rfl | rfl | rfl | rfl | rfl <;> exact ⟨_, _, _, _, rfl⟩)
      hPre
  -- pass 1 with the memory frame carried alongside the context invariant
  obtain ⟨finalV, stVar, hVarFor, hVarFinal, hVarP⟩ :=
    forRange_inv (idx := "block_n_strart_ptr") (start := 0) (stop := N) (step := B)
      (body := rmsVarLoopBody x sxk N B)
      (P := fun off st =>
        rmsVarLoopContextInvariant s₀ x sxb sxm sxk N B off st ∧
        st.pids = s₀.pids ∧
        st.regs .nat [] "pid_batch" = some (Tile.scalar (s₀.pids 0)) ∧
        st.regs .nat [] "pid_m" = some (Tile.scalar (s₀.pids 1)) ∧
        (∀ offset, st.readMem w offset = s₀.readMem w offset) ∧
        st.mem = s₀.mem)
      hStepNe
      ⟨hPreCtx, hPrePids, hPrePb, hPrePm, hPreRw, hPreMem⟩
      (fun i stt hlt hP => by
        obtain ⟨st', hstep, hCtx'⟩ :=
          rmsVarLoopContextInvariant_body_step_exists s₀ stt x sxb sxm sxk N B i hP.1
        obtain ⟨hPidsS, hPbS, hPmS, hRwS⟩ :=
          rmsVarLoopBody_step_preserves_pids_read s₀ stt st' x w sxb sxm sxk N B i hP.1 hstep
        have hMemS : st'.mem
            = (stt.setReg "block_n_strart_ptr" .nat [] (Tile.scalar i)).mem :=
          stepStmts_assigns_mem (rmsVarLoopBody x sxk N B)
            (by
              intro stmt hst
              simp only [rmsVarLoopBody, List.mem_cons, List.not_mem_nil, or_false] at hst
              rcases hst with rfl | rfl | rfl | rfl | rfl <;> exact ⟨_, _, _, _, rfl⟩)
            hstep
        refine ⟨st', hstep, hCtx', ?_, ?_, ?_, ?_, ?_⟩
        · rw [hPidsS]; exact hP.2.1
        · rw [hPbS]; exact hP.2.2.1
        · rw [hPmS]; exact hP.2.2.2.1
        · intro offset; rw [hRwS offset]; exact hP.2.2.2.2.1 offset
        · rw [hMemS]; exact hP.2.2.2.2.2)
  obtain ⟨hVarCtx, hVarPids, hVarPb, hVarPm, hVarRw, hVarMem⟩ := hVarP
  -- reduce/sqrt tail
  obtain ⟨stStd, hStd⟩ := rmsImplPost_exists N B eps stVar _ hVarCtx.1.2
  have hOutBase :=
    rmsStdPostLoop_step_to_out_init s₀ stVar stStd x w o sxb sxm sxk srw sob som sok
      N B finalV eps hVarCtx hVarFinal hB hVarPids hVarPb hVarPm hVarRw hStd
  have hStdMem : stStd.mem = s₀.mem := by
    rw [stepStmts_assigns_mem (rmsStdPostLoop N B eps)
      (by
        intro stmt hst
        simp only [rmsStdPostLoop, List.mem_cons, List.not_mem_nil, or_false] at hst
        rcases hst with rfl | rfl <;> exact ⟨_, _, _, _, rfl⟩)
      hStd]
    exact hVarMem
  -- pass 2 with the conditional cell frame carried alongside the invariant
  have hOutInj := outColOffset_fin_injective_of_stride_out_k_pos s₀ sob som sok N hsok
  obtain ⟨finalO, stOut, hOutFor, hOutFinal, hOutP⟩ :=
    forRange_inv (idx := "block_n_strart_ptr") (start := 0) (stop := N) (step := B)
      (body := rmsOutLoopBody x w o sxk srw sob som sok N B)
      (P := fun off st =>
        rmsOutLoopContextInvariant s₀ x w o sxb sxm sxk srw sob som sok N B off eps st ∧
        ∀ r oo, (r ≠ o ∨ ∀ i : Fin N, oo ≠ outColOffset s₀ sob som sok i.val) →
          st.mem r oo = s₀.mem r oo)
      hStepNe
      ⟨hOutBase, fun r oo _ => by rw [hStdMem]⟩
      (fun i stt hlt hQ => by
        obtain ⟨st', hstep, hQ'⟩ :=
          rmsOutLoopContextInvariant_body_step_exists s₀ stt x w o sxb sxm sxk srw
            sob som sok N B i eps hQ.1 hxo hwo hOutInj
            (outColOffset_chunk_injective_of_stride_out_k_pos s₀ sob som sok B i hsok)
        refine ⟨st', hstep, hQ', ?_⟩
        intro r oo hcond
        obtain ⟨hOutInv, hPidsQ, hPbQ, hPmQ, hOmQ, hBnQ, hStdQ, hRXQ, hRWQ⟩ := hQ.1
        rw [rmsImplOutBody_step_frame s₀ stt st' x w o sxb sxm sxk srw sob som sok N B i eps
          hPbQ hPmQ hOmQ hBnQ hStdQ hRXQ hRWQ hstep r oo hcond]
        exact hQ.2 r oo hcond)
  -- assemble the `execR` run through the cast-free collapses
  have hVarR : stepStmtR R (Stmt.forRange "block_n_strart_ptr" 0 N B
      (rmsVarLoopBody x sxk N B)) stPre = some stVar := by
    rw [stepStmtR_forRange,
      stepForRangeAuxR_castFree R _ (rmsImplVarBody_castFree R x sxk N B) "block_n_strart_ptr",
      ← stepForRangeAux.forRange_unfold]
    exact hVarFor
  have hOutR : stepStmtR R (Stmt.forRange "block_n_strart_ptr" 0 N B
      (rmsOutLoopBody x w o sxk srw sob som sok N B)) stStd = some stOut := by
    rw [stepStmtR_forRange,
      stepForRangeAuxR_castFree R _ (rmsImplOutBody_castFree R x w o sxk srw sob som sok N B) "block_n_strart_ptr",
      ← stepForRangeAux.forRange_unfold]
    exact hOutFor
  refine ⟨stOut, ?_, ?_, ?_⟩
  · show execR R (rmsnorm_implementation x w o sxb sxm sxk srw sob som sok N B eps).toAlgKernel s₀
        = some stOut
    unfold execR
    rw [rmsnorm_implementation_toAlg_body]
    simp only [List.append_assoc, List.singleton_append, List.cons_append,
      List.nil_append]
    rw [stepStmtsR_append R (rmsVarPreLoop sxb sxm B) _ s₀,
      rmsImplPre_castFree R sxb sxm B s₀, hPre, Option.bind_some,
      stepStmtsR_cons_some hVarR,
      stepStmtsR_append R (rmsStdPostLoop N B eps)
        [Stmt.forRange "block_n_strart_ptr" 0 N B (rmsOutLoopBody x w o sxk srw sob som sok N B)] stVar,
      rmsImplPost_castFree R N B eps stVar, hStd, Option.bind_some,
      stepStmtsR_cons_some hOutR, stepStmtsR_nil]
  · intro i
    exact hOutP.1.1 i (Nat.lt_of_lt_of_le i.isLt hOutFinal)
  · exact hOutP.2

/-! ### The headline -/

set_option maxHeartbeats 4000000 in
/-- **The `⊨[R]` streaming headline (wave-5 S3 per-step emit genre).** For
every rounding model `R`, the faithful `rmsnorm_implementation` surface
implements, on its `StreamEmitMasked2DKernelIO₂` signature, the **ideal ℝ
two-pass RMS-norm** over the streamed tiles: emitted window `(t, j)` holds
`x[t,j] / √(Σ_guarded x²/N_SIZE + eps) · w[t,j]`, where the guarded double
sum folds the *entire* `x` stream — the spec `f` is exact real arithmetic
in the kernel's own division spelling. The kernel has **zero rounding
events** (loads, both passes' arithmetic and the per-step stores are all at
`.real`; the erased `.to(tl.float32)` is `.real → .real`), so the skin's
boundary quantization degenerates: the readback contract's `R.round .real`
is the identity by the model's defining `round_real`, and the `.real`
in-loop stores are exact under `execR R` — the ∀-`R` face holds via the
`RoundingModel` `.real` identity fields, not as a `.triv` special case.

Layer map: both passes are cast-free, so under `execR R` they collapse
verbatim onto the exact stepper and the proven
`rmsVarLoopContextInvariant` / `rmsStdPostLoop_step_to_out_init` /
`rmsOutLoopContextInvariant` invariant stack above is reused unchanged; the
`⊨[R]` face adds the `TraceSafeR` walk, the per-cell memory frame
(`rmsImplOutBody_step_frame`, the `mem` twin of
`rmsOutLoopBody_step_preserves_old_output`), and the stream-lane spec
bridge (`rmsImpl_sum_sq_mean` re-blocking `k ↔ (k/B, k%B)`).

All four hypotheses are truth-forced (exactly the exact headline
`rmsnorm_implementation_output_summary`'s side conditions):

* `hBlockPos : 0 < BLOCK_N_SIZE` — both loops step by `BLOCK_N_SIZE`
  (`range(0, N_SIZE, BLOCK_N_SIZE)`); at `BLOCK_N_SIZE = 0` and
  `N_SIZE > 0` neither `forRange` advances, `execR` cannot terminate the
  way the invariant stack requires, and the step index `i / B` is
  meaningless. It holds for every real launch.
* `hXOutNe : x_ptr ≠ out_ptr`, `hWOutNe : rms_w_ptr ≠ out_ptr` — pass 2
  stores into `out` **between** its re-reads of `x` and `rms_w`; if `out`
  aliased either input, later blocks would re-read already-overwritten
  values and the closed form would be false.
* `hStrideOutKPos : 0 < stride_out_k` — the per-lane write window
  `pid₀·sob + pid₁·som + k·sok` is injective over global lanes only when
  the output column stride is nonzero
  (`outColOffset_fin_injective_of_stride_out_k_pos`); with
  `stride_out_k = 0` all lanes collide and the per-lane readback would be
  last-writer-wins. Always true for a real tensor.

Relation to the exact surface: the exact headline
`rmsnorm_implementation_output_summary` (`Realizes_without_Rounding`) above
is retained unchanged; this `⊨[R]` face restates the same RMS-norm content
on the streaming emit skin, for every `R` at once (at the `.real` grid the
two faces carry the same exact cell). Both faces are kept per the
rounding-as-default doctrine. -/
specification rmsnorm_implementation_io_correctness (R : RoundingModel)
    (x_ptr rms_w_ptr out_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE : Nat)
    (eps : ℝ)
    (hBlockPos : 0 < BLOCK_N_SIZE)
    (hXOutNe : x_ptr ≠ out_ptr) (hWOutNe : rms_w_ptr ≠ out_ptr)
    (hStrideOutKPos : 0 < stride_out_k) :
    rmsnormImplementationKernelIO x_ptr rms_w_ptr out_ptr stride_x_batch
        stride_x_m stride_x_k stride_rms_w stride_out_batch stride_out_m
        stride_out_k N_SIZE BLOCK_N_SIZE eps ⊨[R]
      fun _ _ xs ws t j =>
        rmsImplStreamSpec N_SIZE BLOCK_N_SIZE eps xs ws t j := by
  refine StreamEmitMasked2DKernelIO₂.ImplementsR.intro _ ?_ ?_ ?_
  · exact rmsnorm_implementation_flattenOk x_ptr rms_w_ptr out_ptr
      stride_x_batch stride_x_m stride_x_k stride_rms_w stride_out_batch
      stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE eps
  · -- safety walk
    intro bounds s xs ws _hx _hw hbr1 hbr2 hbw
    simp only [rmsnormImplementationKernelIO] at hbr1 hbr2 hbw ⊢
    exact rmsImpl_traceSafeR R bounds x_ptr rms_w_ptr out_ptr stride_x_batch
      stride_x_m stride_x_k stride_rms_w stride_out_batch stride_out_m
      stride_out_k N_SIZE BLOCK_N_SIZE eps hBlockPos hXOutNe hWOutNe
      hStrideOutKPos s hbr1 hbr2 hbw
  · -- the rounded Hoare triple
    intro s₀ xs ws _hundef hx hw
    simp only [rmsnormImplementationKernelIO] at hx hw ⊢
    obtain ⟨sfin, hexec, hval, hframe⟩ :=
      rmsImpl_runR R x_ptr rms_w_ptr out_ptr stride_x_batch stride_x_m
        stride_x_k stride_rms_w stride_out_batch stride_out_m stride_out_k
        N_SIZE BLOCK_N_SIZE eps hBlockPos hXOutNe hWOutNe hStrideOutKPos s₀
    refine ⟨sfin, hexec, ?_, ?_⟩
    · intro t j hj
      have hk : t.val * BLOCK_N_SIZE + j.val < N_SIZE := hj
      have hval' : sfin.readMem out_ptr
          (s₀.pids 0 * stride_out_batch + s₀.pids 1 * stride_out_m
            + (t.val * BLOCK_N_SIZE + j.val) * stride_out_k)
          = rmsnormWeightedYFullNSpec s₀ x_ptr rms_w_ptr stride_x_batch
              stride_x_m stride_x_k stride_rms_w N_SIZE BLOCK_N_SIZE eps
              ⟨t.val * BLOCK_N_SIZE + j.val, hk⟩ :=
        hval ⟨t.val * BLOCK_N_SIZE + j.val, hk⟩
      rw [BlockState.readMemAs_real, hval',
        ← rmsImplStreamSpec_eq_fullNSpec x_ptr rms_w_ptr s₀ stride_x_batch
            stride_x_m stride_x_k stride_rms_w N_SIZE BLOCK_N_SIZE eps
            hBlockPos xs ws hx hw t j hj]
      simp [FloatDType.ofReal]
    · intro r oo hcond
      refine hframe r oo ?_
      rcases hcond with hne | hno
      · exact Or.inl hne
      · refine Or.inr fun k => ?_
        have hdm : k.val / BLOCK_N_SIZE * BLOCK_N_SIZE + k.val % BLOCK_N_SIZE
            = k.val := by
          rw [Nat.mul_comm]
          exact Nat.div_add_mod k.val BLOCK_N_SIZE
        have h := hno
          ⟨k.val / BLOCK_N_SIZE,
            rmsStep_lt_numSteps N_SIZE BLOCK_N_SIZE k.val hBlockPos k.isLt⟩
          ⟨k.val % BLOCK_N_SIZE, Nat.mod_lt _ hBlockPos⟩
          (by simp [hdm, k.isLt])
        simpa [hdm, outColOffset] using h

end IOFace

end VeriTile.Bench.TritonBenchG.RmsnormImplementation
