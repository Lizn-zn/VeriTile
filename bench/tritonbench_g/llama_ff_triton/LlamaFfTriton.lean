import VeriTile.Triton
import VeriTile.Examples.Common

/-!
# `llama_ff_triton` — RMSNorm-fused SwiGLU dual-GEMM correctness

`llama_ff_triton.py`'s `ff_llama` is the llama FFN first half: per output tile
the kernel streams K-blocks accumulating THREE running values — the RMS moment
`a_sum += pow(a, 2)` (shape `[BLOCK_M, BLOCK_K]`), and the two gated-GEMM
accumulators `acc1 += tl.dot(a * rms_w, w1_block)` /
`acc2 += tl.dot(a * rms_w, w3_block)` — then the tail computes
`a_norm = rsqrt(sum(a_sum, axis=1)/K + EPS)`, scales both accumulators by
`a_norm[:, None]`, gates `accumulator = (acc1 * tl.sigmoid(acc1)) * acc2`
(SiLU-gated SwiGLU), and masked-stores the fp16 tile. Loads are UNMASKED (the
host guarantees the K-walk; row/col indices are `% M`/`% N` wrapped exactly like
the `matmul_triton_autotune` template); the linear `pid` is split by the plain
`pid // cdiv(N, BLOCK_N)` / `pid % cdiv(N, BLOCK_N)` schedule.

This file proves the **full K-loop + RMS/SiLU tail** correct against a genuine
closed form: every active output cell `(i, j)` holds the fp16-cast of

  `silu(n_i · S1[i,j]) * (n_i · S2[i,j])`

over ℝ, where over the wrapped row `r(i) = (pid_m·BM + i) % M` and wrapped
column `c(j) = (pid_n·BN + j) % N`:
- `S1[i,j] = Σ_{t<numKBlocks} Σ_{e<BK} A[r(i), t·BK+e] · RMS[t·BK+e] · W1[t·BK+e, c(j)]`
  (with the strides in the addressing), similarly `S2` with `W3`;
- `n_i = rsqrt( (Σ_{t<numKBlocks} Σ_{e<BK} A[r(i), t·BK+e]²) / K + EPS )`;
- `silu(x) = x · sigmoid(x)` (the shared `TiledActivation.silu`).

This is NOT the kernel's own emitted value — the double sums, the RMS moment,
and the SiLU gate are derived independently of the kernel from the loaded
`A`/`W1`/`W3`/`RMS` cells.

## Proof architecture

```
ff_llama_closed_form_correct            ← TOP THEOREM (ComputeCorrect.Realizes_without_Rounding)
  └─ ff_llama_exec_closed_form          ← exec-side closed form (every active cell)
       ├─ preLoop      (P 0: acc1/acc2/a_sum zeroed, four pointers seeded, pid split derived)
       ├─ ff_step      (one K-block: unmasked loads advance the three partials + four pointers)
       ├─ ff_postLoop  (RMS tail: mean/rsqrt + a_norm scaling + SiLU gate + fp16 cast + masked store)
       └─ forRange_inv (loop-invariant principle, drives the K-loop)
```

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); the three accumulators
are over `ℝ` and the store's `tl.float16` cast is the placeholder
`FloatDType.real.cast .fp16`. The contracted dimension is presented as
`K = BLOCK_SIZE_K · numKBlocks` so the loop trip count
`cdiv(K, BLOCK_SIZE_K) = numKBlocks` is exact — the kernel's loads are
UNMASKED, so this exact-multiple presentation is required for the loads to be
meaningful (disclosed; the host launches with `dim` a multiple of the K block).
`num_stages`/`num_warps` are not modeled. The host launch (grid, linear-pid
scheduling) is trusted; the per-program statement is universally quantified
over `s`, covering every program of the grid. The `pid → (pid_m, pid_n)` split
and the `% M`/`% N` index wraps are transcribed exactly as the kernel computes
them and the spec references the same derived indices, so they are not separate
proof obligations. Output-offset injectivity is discharged from the row-major
hypotheses `soutn = 1` and `BN ≤ soutm` (the host's contiguous fp16 output).

## Translation-surface blocker

Translation-surface blocker: (a) the kernel's `USE_FP8` constexpr arm is
dropped entirely — the parameter and its three `if USE_FP8:` branches; that
path bit-reinterprets int8 weight bytes as `tl.float8e5` via
`.to(tl.float8e5, bitcast=True)`, a value-level bit reinterpretation with no
transcription in the ℝ-valued model (the `sgmv_expand_slice` dropped-constexpr
precedent) — this file ports the `USE_FP8 = False` (fp16 weights) arm ONLY;
(b) the loop trip count `tl.cdiv(K, BLOCK_SIZE_K)` is supplied as the
antiquoted `numKBlocks` binder (so that `tl.cdiv` call does not appear as a
surface statement; `K = BLOCK_SIZE_K · numKBlocks`, the unmasked-loads
presentation, template precedent); (c) the store's implicit fp16 cast is
spelled `(accumulator).to(tl.float16)` — Python's store carries no cast because
the host allocates `out` as fp16 and Triton casts implicitly at the typed
pointer, while the DSL types stores by value (the `f8_conversion_utils`
precedent); (d) the loop counter is spelled `_i` where Python spells it `_`.
The textual py↔lean scans in `bench/audit_tritonbench_g.sh` exempt this port on
this marker (registration in `proof_blockers.md` is left to the landing PR).
-/

namespace VeriTile.Bench.TritonBenchG.LlamaFfTriton

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Faithful transcription of `llama_ff_triton.py`'s `ff_llama`
(`USE_FP8 = False` arm; see the Translation-surface blocker in the module
docstring for the dropped constexpr arm, the `numKBlocks` presentation, the
spelled-out store cast, and the `_i` loop counter). -/
def ff_llama_surface
    (A W1 W3 OUT RMS : RegionName)
    (M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K numKBlocks : Nat) (EPS : ℝ) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  pid_m = pid // tl.cdiv($(N), $(BLOCK_SIZE_N))
  pid_n = pid % tl.cdiv($(N), $(BLOCK_SIZE_N))
  offs_am = (pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))) % $(M)
  offs_bn = (pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))) % $(N)
  offs_k = tl.arange(0, $(BLOCK_SIZE_K))
  a_ptrs = A + offs_am[:, None] * $(sam) + offs_k[None, :] * $(sak)
  w1_ptrs = W1 + offs_k[:, None] * $(sw1k) + offs_bn[None, :] * $(sw1n)
  w3_ptrs = W3 + offs_k[:, None] * $(sw3k) + offs_bn[None, :] * $(sw3n)
  acc1 = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_N)], dtype=tl.float32)
  acc2 = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_N)], dtype=tl.float32)
  rms_w_ptrs = RMS + tl.arange(0, $(BLOCK_SIZE_K))[None, :] * $(srms)
  a_sum = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_K)], dtype=tl.float32)
  for _i in range($(0), $(numKBlocks), $(1)) {
    a = tl.load(a_ptrs)
    a_sum += tl.extra.cuda.libdevice.pow((a).to(tl.float32), $((2 : ℝ)))
    rms_w = tl.load(rms_w_ptrs)
    a = a * rms_w
    b = tl.load(w1_ptrs)
    acc1 += tl.dot(a, b)
    c = tl.load(w3_ptrs)
    acc2 += tl.dot(a, c)
    a_ptrs += $(BLOCK_SIZE_K) * $(sak)
    w1_ptrs += $(BLOCK_SIZE_K) * $(sw1k)
    w3_ptrs += $(BLOCK_SIZE_K) * $(sw3k)
    rms_w_ptrs += $(BLOCK_SIZE_K) * $(srms)
  }
  a_mean = tl.sum(a_sum, axis=1) / $(K) + $(EPS)
  a_norm = tl.math.rsqrt(a_mean)
  acc1 = acc1 * a_norm[:, None]
  acc2 = acc2 * a_norm[:, None]
  accumulator = (acc1 * tl.sigmoid(acc1)) * acc2
  offs_outm = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_outn = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  out_ptrs = OUT + $(soutm) * offs_outm[:, None] + $(soutn) * offs_outn[None, :]
  out_mask = (offs_outm[:, None] < $(M)) & (offs_outn[None, :] < $(N))
  tl.store(out_ptrs, (accumulator).to(tl.float16), mask=out_mask)
}

/-- The full surface lowers to the algorithm layer. -/
theorem ff_llama_surface_toAlgorithm_supported
    (A W1 W3 OUT RMS : RegionName)
    (M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK numKBlocks : Nat) (EPS : ℝ) :
    ∃ alg, (ff_llama_surface A W1 W3 OUT RMS M N K sam sak sw1k sw1n sw3k sw3n
      soutm soutn srms BM BN BK numKBlocks EPS).toAlgorithm? = Except.ok alg := by
  simp [ff_llama_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-! ## exec-stepping helpers -/

@[simp] theorem evalOp_expandDim_zero_nat {D : Nat} (name : RegName) (s : BlockState) :
    @evalOp .nat [1, D] (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] name)) s =
      (s.regs .nat [D] name).bind (fun v =>
        some ({ data := fun i : TileIndex [1, D] => v.data (i.2.1, PUnit.unit) } : Tile .nat [1, D])) := by
  unfold evalOp; simp [Tile.expandDim]; rfl

@[simp] theorem evalOp_expandDim_one_nat {M : Nat} (name : RegName) (s : BlockState) :
    @evalOp .nat [M, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] name)) s =
      (s.regs .nat [M] name).bind (fun v =>
        some ({ data := fun i : TileIndex [M, 1] => v.data (i.1, PUnit.unit) } : Tile .nat [M, 1])) := by
  unfold evalOp; simp [Tile.expandDim]; rfl

/-- `tl.arange(0, D)[None, :]` eval (the inline `rms_w_ptrs` index vector). -/
theorem evalOp_expandDim_zero_arange {D : Nat} (s : BlockState) :
    @evalOp .nat [1, D] (Op.expandDim ⟨0, by simp⟩ (Op.arange D)) s =
      some ({ data := fun i : TileIndex [1, D] => i.2.1.val } : Tile .nat [1, D]) := by
  unfold evalOp; simp [Tile.expandDim]; rfl

theorem evalOp_boolAnd {a b shape} (bc : Broadcast a b shape)
    (x : Op .bool a) (y : Op .bool b) (s : BlockState) :
    evalOp (.boolAnd bc x y) s = (do
      let vx ← evalOp x s; let vy ← evalOp y s; some (Tile.bop (· && ·) bc vx vy)) := by
  simp [evalOp]

theorem evalOp_sigmoid {shape : TileShape} (a : Op .real shape) (s : BlockState) :
    evalOp (.sigmoid a) s = (do
      let va ← evalOp a s; some (Tile.uop WithBot.realSigmoid va)) := by
  simp [evalOp]

theorem evalOp_rsqrt {shape : TileShape} (a : Op .real shape) (s : BlockState) :
    evalOp (.rsqrt a) s = (do
      let va ← evalOp a s; some (Tile.uop WithBot.realRsqrt va)) := by
  simp [evalOp]

/-- Unmasked `.ptr` load of a `.real` tile reads the clean `readMem` cells. -/
theorem load_ptr_none_real {shape : TileShape} (name : RegName) (st : BlockState)
    (ptrs : Tile .ptr shape)
    (hp : st.regs .ptr shape name = some ptrs) :
    evalOp (Op.load .real (.ptr (Op.ref .ptr shape name)) MaskOpt.none) st
      = some ⟨fun i => some (st.readMem (ptrs.data i).1 (ptrs.data i).2)⟩ := by
  simp only [evalOp, evalOp_ref, hp, bind, Option.bind]
  refine congrArg some ?_
  ext i
  simp [BlockState.readMemValue_real]

/-- The identity `(a).to(tl.float32)` cast on a `.real` register. -/
theorem cast_real_real_eval {shape : TileShape} (name : RegName) (st : BlockState)
    (v : Tile .real shape) (h : st.regs .real shape name = some v) :
    evalOp (dtype := .real) (Op.castFloat FloatDType.real FloatDType.real (Op.ref .real shape name)) st
      = some v := by
  unfold evalOp
  erw [evalOp_ref, h]
  refine congrArg some ?_
  ext i
  rfl

/-- `a_ptrs` eval: cell `(i,e) = (A, offs_am i · sam + offs_k e · sak)`. -/
theorem aptrs_eval (s : BlockState) (A : RegionName) (M K sam sak : Nat) (gm : Fin M → Nat)
    (hm : s.regs .nat [M] "offs_am" = some (Tile.vec gm))
    (hk : s.regs .nat [K] "offs_k" = some (Tile.vec (fun e : Fin K => e.val))) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase A)
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "offs_am")) (Op.constNat sam))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [K] "offs_k")) (Op.constNat sak)))) s
      = some (⟨fun idx : TileIndex [M, K] => (A.cast, gm idx.1 * sam + idx.2.1.val * sak)⟩ : Tile .ptr [M, K]) := by
  rw [evalOp_ptrAdd, evalOp_ptrBase]
  simp only [evalOp_add, evalOp_mul, evalOp_constNat, evalOp_ref, evalOp_expandDim_one_nat,
    evalOp_expandDim_zero_nat, hm, hk, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec]
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec, NumericDType.add, NumericDType.mul]

/-- `w1_ptrs`/`w3_ptrs` eval: cell `(e,j) = (W, offs_k e · swk + offs_bn j · swn)`. -/
theorem wptrs_eval (s : BlockState) (W : RegionName) (K N swk swn : Nat) (gn : Fin N → Nat)
    (hk : s.regs .nat [K] "offs_k" = some (Tile.vec (fun e : Fin K => e.val)))
    (hn : s.regs .nat [N] "offs_bn" = some (Tile.vec gn)) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase W)
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [K] "offs_k")) (Op.constNat swk))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [N] "offs_bn")) (Op.constNat swn)))) s
      = some (⟨fun idx : TileIndex [K, N] => (W.cast, idx.1.val * swk + gn idx.2.1 * swn)⟩ : Tile .ptr [K, N]) := by
  rw [evalOp_ptrAdd, evalOp_ptrBase]
  simp only [evalOp_add, evalOp_mul, evalOp_constNat, evalOp_ref, evalOp_expandDim_one_nat,
    evalOp_expandDim_zero_nat, hk, hn, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec]
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec, NumericDType.add, NumericDType.mul]

/-- `rms_w_ptrs` eval (inline arange): cell `(0,e) = (RMS, e · srms)`. -/
theorem rmsptrs_eval (s : BlockState) (RMS : RegionName) (BK srms : Nat) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase RMS)
      (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.arange BK)) (Op.constNat srms))) s
      = some (⟨fun idx : TileIndex [1, BK] => (RMS.cast, idx.2.1.val * srms)⟩ : Tile .ptr [1, BK]) := by
  rw [evalOp_ptrAdd, evalOp_ptrBase]
  simp only [evalOp_mul, evalOp_constNat, evalOp_expandDim_zero_arange,
    Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  · simp [Tile.ptrAdd, Tile.bop]
  · simp [Tile.ptrAdd, Tile.bop, NumericDType.mul]

/-- `tl.zeros` eval → the all-`0` tile. -/
theorem zeros_eval (sh : TileShape) (st : BlockState) :
    evalOp (Op.full sh (Op.const 0)) st
      = some (⟨fun _ => some (0 : ℝ)⟩ : Tile .real sh) := by
  simp [evalOp_full, evalOp_const, Option.bind]

/-- `ptr += BLOCK_K · stride` eval (shared by all four advancing pointers). -/
theorem ptr_adv_eval {n : Nat} {r : TileShape} (st : BlockState) (name : RegName)
    (BK stride : Nat) (p : Tile .ptr (n :: r))
    (hp : st.regs .ptr (n :: r) name = some p) :
    evalOp (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr (n :: r) name)
      (Op.mul .nat Broadcast.nil (Op.constNat BK) (Op.constNat stride))) st
      = some (Tile.ptrAdd Broadcast.scalarR p (Tile.scalar (BK * stride))) := by
  rw [evalOp_ptrAdd]
  simp [evalOp_ref, hp, evalOp_mul, evalOp_constNat, NumericDType.mul, Tile.bop]

/-- `acc += tl.dot(a, b)` statement eval (accumulator on the LEFT). -/
theorem accdot_eval (M K N : Nat) (st : BlockState) (accName bName : RegName)
    (zt : Tile .real [M, N]) (xt : Tile .real [M, K]) (yt : Tile .real [K, N])
    (hz : st.regs .real [M, N] accName = some zt)
    (hx : st.regs .real [M, K] "a" = some xt)
    (hy : st.regs .real [K, N] bName = some yt) :
    evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [M, N] accName)
        (Op.dot (batch := []) (Op.ref .real [M, K] "a") (Op.ref .real [K, N] bName))) st
      = some (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
          zt (Tile.dot [] xt yt)) := by
  have hd : evalOp (Op.dot (batch := []) (Op.ref .real [M, K] "a")
        (Op.ref .real [K, N] bName)) st = some (Tile.dot [] xt yt) := by
    rw [evalOp_dot]; simp [hx, hy]
  rw [evalOp_add]
  simp only [evalOp_ref, hz, bind, Option.bind_some]
  erw [hd]
  rfl

/-- `acc + dot` lane `(i,j)`: `some (zv + dv)`. -/
theorem adddot_lane (M N : Nat) (zt dt : Tile .real [M, N]) (i : Fin M) (j : Fin N) (zv dv : ℝ)
    (hz : zt.data (i, j, PUnit.unit) = some zv) (hd : dt.data (i, j, PUnit.unit) = some dv) :
    (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil)) zt dt).data
        (i, j, PUnit.unit) = some (zv + dv) := by
  simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, hz, hd, NumericDType.add,
    WithBot.realAdd, Option.map₂, Option.bind, Option.map]

/-- `Real.rpow x 2 = x ^ 2` (the `pow(_, 2)` bridging). -/
theorem rpow_two_eq (x : ℝ) : Real.rpow x 2 = x ^ 2 := Real.rpow_two x

/-- `offs_outm` / `offs_outn` eval (the un-wrapped global index, no `% M`). -/
theorem offsout_eval (s : BlockState) (M BM : Nat) (pm : Nat) (pidReg : RegName)
    (hpm : s.regs .nat [] pidReg = some (Tile.scalar pm)) :
    evalOp (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] pidReg) (Op.constNat BM)) (Op.arange M)) s
      = some (Tile.vec (fun i : Fin M => pm * BM + i.val)) := by
  simp only [evalOp_add, evalOp_mul, evalOp_ref, evalOp_constNat, evalOp_arange, hpm,
    Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext i
  simp only [Tile.bop_data, Tile.scalar_data, Tile.vec_data, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.add, NumericDType.mul]

/-- `out_ptrs` eval: cell `(i,j) = (OUT, soutm · offs_outm i + soutn · offs_outn j)`
(strides on the **left** of the products). -/
theorem outptrs_eval (s : BlockState) (OUT : RegionName) (M N som son : Nat)
    (gm : Fin M → Nat) (gn : Fin N → Nat)
    (hm : s.regs .nat [M] "offs_outm" = some (Tile.vec gm))
    (hn : s.regs .nat [N] "offs_outn" = some (Tile.vec gn)) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase OUT)
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.mul .nat Broadcast.scalarL (Op.constNat som) (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "offs_outm")))
        (Op.mul .nat Broadcast.scalarL (Op.constNat son) (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [N] "offs_outn"))))) s
      = some (⟨fun idx : TileIndex [M, N] => (OUT.cast, som * gm idx.1 + son * gn idx.2.1)⟩ : Tile .ptr [M, N]) := by
  rw [evalOp_ptrAdd, evalOp_ptrBase]
  simp only [evalOp_add, evalOp_mul, evalOp_constNat, evalOp_ref, evalOp_expandDim_one_nat,
    evalOp_expandDim_zero_nat, hm, hn, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec]
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec, NumericDType.add, NumericDType.mul]

/-- `out_mask` eval: the `(offs_outm < M) & (offs_outn < N)` boolean tile. -/
theorem outmask_eval (s : BlockState) (M N BM BN : Nat) (gm : Fin BM → Nat) (gn : Fin BN → Nat)
    (hm : s.regs .nat [BM] "offs_outm" = some (Tile.vec gm))
    (hn : s.regs .nat [BN] "offs_outn" = some (Tile.vec gn)) :
    evalOp (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_outm")) (Op.constNat M))
        (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_outn")) (Op.constNat N))) s
      = some ⟨fun idx : TileIndex [BM, BN] => (decide (gm idx.1 < M) && decide (gn idx.2.1 < N))⟩ := by
  rw [evalOp_boolAnd, evalOp_lt, evalOp_lt, evalOp_expandDim_one_nat, evalOp_expandDim_zero_nat]
  simp only [hm, hn, evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  simp [Tile.bop_data, Tile.cop_data, Tile.expandDim, Tile.vec, Tile.scalar_data, ComparableDType.lt,
    Broadcast.leftIndex, Broadcast.rightIndex]

/-! ## Scheduling + closed-form spec -/

/-- The kernel's `pid_m` derivation: `pid // cdiv(N, BLOCK_N)`. -/
def pidM (pid N BN : Nat) : Nat := pid / cdiv N BN

/-- The kernel's `pid_n` derivation: `pid % cdiv(N, BLOCK_N)`. -/
def pidN (pid N BN : Nat) : Nat := pid % cdiv N BN

/-- Global output row of tile lane `i`: `pid_m · BLOCK_M + i`, **before** the
`% M` wrap (the kernel's `offs_outm`). -/
def rowGlobal (s : BlockState) (N BM BN : Nat) (i : Fin BM) : Nat :=
  pidM (s.pids 0) N BN * BM + i.val

/-- Global output column of tile lane `j`: `pid_n · BLOCK_N + j`, before wrap. -/
def colGlobal (s : BlockState) (N BN : Nat) (j : Fin BN) : Nat :=
  pidN (s.pids 0) N BN * BN + j.val

/-- The `% M`-wrapped A-row index of tile lane `i` (the kernel's `offs_am`). -/
def rowIndex (s : BlockState) (M N BM BN : Nat) (i : Fin BM) : Nat :=
  rowGlobal s N BM BN i % M

/-- The `% N`-wrapped W-column index of tile lane `j` (the kernel's `offs_bn`). -/
def colIndex (s : BlockState) (N BN : Nat) (j : Fin BN) : Nat :=
  colGlobal s N BN j % N

/-- `A[i, k] = readMem A (offs_am i · sam + k · sak)`. -/
noncomputable def aElem (s : BlockState) (A : RegionName) (M N BM BN sam sak : Nat)
    (i : Fin BM) (k : Nat) : ℝ :=
  s.readMem A (rowIndex s M N BM BN i * sam + k * sak)

/-- `RMS[k] = readMem RMS (k · srms)`. -/
noncomputable def rmsElem (s : BlockState) (RMS : RegionName) (srms : Nat) (k : Nat) : ℝ :=
  s.readMem RMS (k * srms)

/-- `W[k, j] = readMem W (k · swk + offs_bn j · swn)` (shared by `W1`/`W3`). -/
noncomputable def wElem (s : BlockState) (W : RegionName) (N BN swk swn : Nat)
    (j : Fin BN) (k : Nat) : ℝ :=
  s.readMem W (k * swk + colIndex s N BN j * swn)

/-- Partial gated-GEMM accumulator after `c` K-blocks (shared by `acc1`/`acc2`):
`Σ_{t<c} Σ_{e<BK} A[r(i), t·BK+e] · RMS[t·BK+e] · W[t·BK+e, c(j)]`. -/
noncomputable def accPartial (s : BlockState) (A W RMS : RegionName)
    (M N BM BN sam sak swk swn srms BK : Nat) (i : Fin BM) (j : Fin BN) (c : Nat) : ℝ :=
  ∑ t : Fin c, ∑ e : Fin BK,
    aElem s A M N BM BN sam sak i (t.val * BK + e.val)
      * rmsElem s RMS srms (t.val * BK + e.val)
      * wElem s W N BN swk swn j (t.val * BK + e.val)

/-- Partial RMS moment after `c` K-blocks (the `a_sum` lane `(i,e)`):
`Σ_{t<c} A[r(i), t·BK+e]²`. -/
noncomputable def aSumPartial (s : BlockState) (A : RegionName)
    (M N BM BN sam sak BK : Nat) (i : Fin BM) (e : Fin BK) (c : Nat) : ℝ :=
  ∑ t : Fin c, aElem s A M N BM BN sam sak i (t.val * BK + e.val) ^ 2

theorem accPartial_succ (s : BlockState) (A W RMS : RegionName)
    (M N BM BN sam sak swk swn srms BK : Nat) (i : Fin BM) (j : Fin BN) (c : Nat) :
    accPartial s A W RMS M N BM BN sam sak swk swn srms BK i j (c + 1)
      = accPartial s A W RMS M N BM BN sam sak swk swn srms BK i j c
        + (Finset.univ.sum fun e : Fin BK =>
            aElem s A M N BM BN sam sak i (c * BK + e.val)
              * rmsElem s RMS srms (c * BK + e.val)
              * wElem s W N BN swk swn j (c * BK + e.val)) := by
  unfold accPartial
  rw [Fin.sum_univ_castSucc]
  simp

theorem aSumPartial_succ (s : BlockState) (A : RegionName)
    (M N BM BN sam sak BK : Nat) (i : Fin BM) (e : Fin BK) (c : Nat) :
    aSumPartial s A M N BM BN sam sak BK i e (c + 1)
      = aSumPartial s A M N BM BN sam sak BK i e c
        + aElem s A M N BM BN sam sak i (c * BK + e.val) ^ 2 := by
  unfold aSumPartial
  rw [Fin.sum_univ_castSucc]
  simp

/-- **The RMS normalization factor** (over ℝ):
`n_i = rsqrt( (Σ_{t<numKBlocks} Σ_{e<BK} A[r(i), t·BK+e]²) / K + EPS )`. -/
noncomputable def normSpec (s : BlockState) (A : RegionName)
    (M N BM BN sam sak BK numKBlocks K : Nat) (EPS : ℝ) (i : Fin BM) : ℝ :=
  1 / Real.sqrt
    ((∑ t : Fin numKBlocks, ∑ e : Fin BK,
        aElem s A M N BM BN sam sak i (t.val * BK + e.val) ^ 2) / (K : ℝ) + EPS)

/-- **Genuine RMSNorm-fused SwiGLU spec** (over ℝ):
`OUT[i,j] = silu(n_i · S1[i,j]) * (n_i · S2[i,j])` with `S1`/`S2` the full-loop
gated-GEMM sums against `W1`/`W3` and `n_i` the RMS factor. -/
noncomputable def ffSpec (s : BlockState) (A W1 W3 RMS : RegionName)
    (M N BM BN sam sak sw1k sw1n sw3k sw3n srms BK numKBlocks K : Nat) (EPS : ℝ)
    (i : Fin BM) (j : Fin BN) : ℝ :=
  TiledActivation.silu
      (normSpec s A M N BM BN sam sak BK numKBlocks K EPS i
        * accPartial s A W1 RMS M N BM BN sam sak sw1k sw1n srms BK i j numKBlocks)
    * (normSpec s A M N BM BN sam sak BK numKBlocks K EPS i
        * accPartial s A W3 RMS M N BM BN sam sak sw3k sw3n srms BK i j numKBlocks)

/-- The SiLU gate lane identity: the kernel's `(v·σ(v))·w` at `v = S·n`, `w = S'·n`
is the spec's `silu(n·S) · (n·S')`. -/
theorem gate_lane_eq (n s1 s2 : ℝ) :
    (s1 * n * Real.sigmoid (s1 * n)) * (s2 * n)
      = TiledActivation.silu (n * s1) * (n * s2) := by
  rw [mul_comm n s1, mul_comm n s2]
  simp only [TiledActivation.silu]

/-- The output store address for tile lane `(i,j)`:
`soutm · offs_outm i + soutn · offs_outn j` (un-wrapped global indices). -/
def outOffset (s : BlockState) (N BM BN soutm soutn : Nat) (idx : TileIndex [BM, BN]) : Nat :=
  soutm * rowGlobal s N BM BN idx.1 + soutn * colGlobal s N BN idx.2.1

/-- The boundary predicate `(row < M) & (col < N)` for tile lane `(i,j)`. -/
def active (s : BlockState) (M N BM BN : Nat) (idx : TileIndex [BM, BN]) : Prop :=
  rowGlobal s N BM BN idx.1 < M ∧ colGlobal s N BN idx.2.1 < N

instance activeDecidable (s : BlockState) (M N BM BN : Nat)
    (idx : TileIndex [BM, BN]) : Decidable (active s M N BM BN idx) := by
  unfold active; infer_instance

/-! ## Body decomposition -/

/-- The 13-statement prologue: pid split, index vectors, the four pointer
seeds, and the three zero accumulators. -/
def ffPrologue (A W1 W3 RMS : RegionName)
    (M N BM BN BK sam sak sw1k sw1n sw3k sw3n srms : Nat) : List Stmt :=
  [ Stmt.assign .nat [] "pid" (Op.programId 0),
    Stmt.assign .nat [] "pid_m"
      (Op.floorDiv .nat Broadcast.nil (Op.ref .nat [] "pid")
        (Op.div .nat Broadcast.nil
          (Op.sub .nat Broadcast.nil
            (Op.add .nat Broadcast.nil (Op.constNat N) (Op.constNat BN)) (Op.constNat 1))
          (Op.constNat BN))),
    Stmt.assign .nat [] "pid_n"
      (Op.mod .nat Broadcast.nil (Op.ref .nat [] "pid")
        (Op.div .nat Broadcast.nil
          (Op.sub .nat Broadcast.nil
            (Op.add .nat Broadcast.nil (Op.constNat N) (Op.constNat BN)) (Op.constNat 1))
          (Op.constNat BN))),
    Stmt.assign .nat [BM] "offs_am"
      (Op.mod .nat Broadcast.scalarR
        (Op.add .nat Broadcast.scalarL
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_m") (Op.constNat BM))
          (Op.arange BM))
        (Op.constNat M)),
    Stmt.assign .nat [BN] "offs_bn"
      (Op.mod .nat Broadcast.scalarR
        (Op.add .nat Broadcast.scalarL
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_n") (Op.constNat BN))
          (Op.arange BN))
        (Op.constNat N)),
    Stmt.assign .nat [BK] "offs_k" (Op.arange BK),
    Stmt.assign .ptr [BM, BK] "a_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase A)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_am")) (Op.constNat sam))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BK] "offs_k")) (Op.constNat sak)))),
    Stmt.assign .ptr [BK, BN] "w1_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase W1)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BK] "offs_k")) (Op.constNat sw1k))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_bn")) (Op.constNat sw1n)))),
    Stmt.assign .ptr [BK, BN] "w3_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase W3)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BK] "offs_k")) (Op.constNat sw3k))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_bn")) (Op.constNat sw3n)))),
    Stmt.assign .real [BM, BN] "acc1" (Op.full [BM, BN] (Op.const 0)),
    Stmt.assign .real [BM, BN] "acc2" (Op.full [BM, BN] (Op.const 0)),
    Stmt.assign .ptr [1, BK] "rms_w_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase RMS)
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.arange BK)) (Op.constNat srms))),
    Stmt.assign .real [BM, BK] "a_sum" (Op.full [BM, BK] (Op.const 0)) ]

/-- The 12-statement K-loop body (unmasked loads + RMS moment + two fused
dots + four pointer advances). -/
def ffLoopBody (BM BN BK sak sw1k sw3k srms : Nat) : List Stmt :=
  [ Stmt.assign .real [BM, BK] "a"
      (Op.load .real (.ptr (Op.ref .ptr [BM, BK] "a_ptrs")) MaskOpt.none),
    Stmt.assign .real [BM, BK] "a_sum"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BM, BK] "a_sum")
        (Op.pow Broadcast.scalarR
          (Op.castFloat FloatDType.real FloatDType.real (Op.ref .real [BM, BK] "a"))
          (Op.const 2))),
    Stmt.assign .real [1, BK] "rms_w"
      (Op.load .real (.ptr (Op.ref .ptr [1, BK] "rms_w_ptrs")) MaskOpt.none),
    Stmt.assign .real [BM, BK] "a"
      (Op.mul .real (Broadcast.consR (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BM, BK] "a") (Op.ref .real [1, BK] "rms_w")),
    Stmt.assign .real [BK, BN] "b"
      (Op.load .real (.ptr (Op.ref .ptr [BK, BN] "w1_ptrs")) MaskOpt.none),
    Stmt.assign .real [BM, BN] "acc1"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BM, BN] "acc1")
        (Op.dot (batch := []) (Op.ref .real [BM, BK] "a") (Op.ref .real [BK, BN] "b"))),
    Stmt.assign .real [BK, BN] "c"
      (Op.load .real (.ptr (Op.ref .ptr [BK, BN] "w3_ptrs")) MaskOpt.none),
    Stmt.assign .real [BM, BN] "acc2"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BM, BN] "acc2")
        (Op.dot (batch := []) (Op.ref .real [BM, BK] "a") (Op.ref .real [BK, BN] "c"))),
    Stmt.assign .ptr [BM, BK] "a_ptrs"
      (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BM, BK] "a_ptrs")
        (Op.mul .nat Broadcast.nil (Op.constNat BK) (Op.constNat sak))),
    Stmt.assign .ptr [BK, BN] "w1_ptrs"
      (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BK, BN] "w1_ptrs")
        (Op.mul .nat Broadcast.nil (Op.constNat BK) (Op.constNat sw1k))),
    Stmt.assign .ptr [BK, BN] "w3_ptrs"
      (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BK, BN] "w3_ptrs")
        (Op.mul .nat Broadcast.nil (Op.constNat BK) (Op.constNat sw3k))),
    Stmt.assign .ptr [1, BK] "rms_w_ptrs"
      (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [1, BK] "rms_w_ptrs")
        (Op.mul .nat Broadcast.nil (Op.constNat BK) (Op.constNat srms))) ]

/-- The 10-statement post-loop tail: mean/rsqrt, the two `a_norm` scalings, the
SiLU gate, the two `offs_out*` vectors, `out_ptrs`, `out_mask`, and the masked
fp16 store. -/
def ffStoreTail (OUT : RegionName) (M N BM BN BK K soutm soutn : Nat) (EPS : ℝ) : List Stmt :=
  [ Stmt.assign .real [BM] "a_mean"
      (Op.add .real Broadcast.scalarR
        (Op.div .real Broadcast.scalarR
          (Op.reduceSum (⟨1, by simp⟩ : Fin [BM, BK].length) Bool.false
            (Op.ref .real [BM, BK] "a_sum"))
          (Op.const (K : ℝ)))
        (Op.const EPS)),
    Stmt.assign .real [BM] "a_norm" (Op.rsqrt (Op.ref .real [BM] "a_mean")),
    Stmt.assign .real [BM, BN] "acc1"
      (Op.mul .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [BM, BN] "acc1")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BM] "a_norm"))),
    Stmt.assign .real [BM, BN] "acc2"
      (Op.mul .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [BM, BN] "acc2")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BM] "a_norm"))),
    Stmt.assign .real [BM, BN] "accumulator"
      (Op.mul .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.mul .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
          (Op.ref .real [BM, BN] "acc1")
          (Op.sigmoid (Op.ref .real [BM, BN] "acc1")))
        (Op.ref .real [BM, BN] "acc2")),
    Stmt.assign .nat [BM] "offs_outm"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_m") (Op.constNat BM))
        (Op.arange BM)),
    Stmt.assign .nat [BN] "offs_outn"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_n") (Op.constNat BN))
        (Op.arange BN)),
    Stmt.assign .ptr [BM, BN] "out_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase OUT)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarL (Op.constNat soutm) (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_outm")))
          (Op.mul .nat Broadcast.scalarL (Op.constNat soutn) (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_outn"))))),
    Stmt.assign .bool [BM, BN] "out_mask"
      (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_outm")) (Op.constNat M))
        (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_outn")) (Op.constNat N))),
    Stmt.store .fp16 [BM, BN] (.ptr (Op.ref .ptr [BM, BN] "out_ptrs"))
      (Op.castFloat FloatDType.real FloatDType.fp16 (Op.ref .real [BM, BN] "accumulator"))
      (.mask (Op.ref .bool [BM, BN] "out_mask")) ]

/-- Body decomposition: prologue (13) ++ [for-loop] ++ store-tail (10). By `rfl`. -/
theorem ff_body_split (A W1 W3 OUT RMS : RegionName)
    (M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK numKBlocks : Nat) (EPS : ℝ) :
    (ff_llama_surface A W1 W3 OUT RMS M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms
        BM BN BK numKBlocks EPS).toAlgKernel.body
      = ffPrologue A W1 W3 RMS M N BM BN BK sam sak sw1k sw1n sw3k sw3n srms
        ++ (Stmt.forRange "_i" 0 numKBlocks 1 (ffLoopBody BM BN BK sak sw1k sw3k srms)
            :: ffStoreTail OUT M N BM BN BK K soutm soutn EPS) := by
  rfl

/-! ## Loop invariant -/

/-- **Loop invariant** (counter `c = block index`, step `1`).

After `c` K-blocks: program ids and `mem`/`undef` fixed; `pid_m`/`pid_n`
seeded; the three accumulators `acc1`/`acc2`/`a_sum` hold their partials; the
four pointers `a_ptrs`/`w1_ptrs`/`w3_ptrs`/`rms_w_ptrs` advanced by `c` blocks. -/
noncomputable def ffInvariant (A W1 W3 RMS : RegionName) (s0 : BlockState)
    (M N BM BN BK sam sak sw1k sw1n sw3k sw3n srms numKBlocks : Nat)
    (c : Nat) (s : BlockState) : Prop :=
  s.pids = s0.pids ∧ c ≤ numKBlocks ∧
  (s.regs .nat [] "pid_m" = some (Tile.scalar (pidM (s0.pids 0) N BN))) ∧
  (s.regs .nat [] "pid_n" = some (Tile.scalar (pidN (s0.pids 0) N BN))) ∧
  (s.regs .real [BM, BN] "acc1" = some ⟨fun idx : TileIndex [BM, BN] =>
      some (accPartial s0 A W1 RMS M N BM BN sam sak sw1k sw1n srms BK idx.1 idx.2.1 c)⟩) ∧
  (s.regs .real [BM, BN] "acc2" = some ⟨fun idx : TileIndex [BM, BN] =>
      some (accPartial s0 A W3 RMS M N BM BN sam sak sw3k sw3n srms BK idx.1 idx.2.1 c)⟩) ∧
  (s.regs .real [BM, BK] "a_sum" = some ⟨fun idx : TileIndex [BM, BK] =>
      some (aSumPartial s0 A M N BM BN sam sak BK idx.1 idx.2.1 c)⟩) ∧
  (s.regs .ptr [BM, BK] "a_ptrs" = some ⟨fun idx : TileIndex [BM, BK] =>
      (A.cast, rowIndex s0 M N BM BN idx.1 * sam + idx.2.1.val * sak + c * BK * sak)⟩) ∧
  (s.regs .ptr [BK, BN] "w1_ptrs" = some ⟨fun idx : TileIndex [BK, BN] =>
      (W1.cast, idx.1.val * sw1k + colIndex s0 N BN idx.2.1 * sw1n + c * BK * sw1k)⟩) ∧
  (s.regs .ptr [BK, BN] "w3_ptrs" = some ⟨fun idx : TileIndex [BK, BN] =>
      (W3.cast, idx.1.val * sw3k + colIndex s0 N BN idx.2.1 * sw3n + c * BK * sw3k)⟩) ∧
  (s.regs .ptr [1, BK] "rms_w_ptrs" = some ⟨fun idx : TileIndex [1, BK] =>
      (RMS.cast, idx.2.1.val * srms + c * BK * srms)⟩) ∧
  (∀ rg o, s.undef rg o = 0) ∧ (s.mem = s0.mem)

/-- **preLoop scalars** (statements 0–5): the pid split and the 3 index vectors. -/
theorem preLoop_scalars (s : BlockState) (M N BM BN BK : Nat) :
    ∃ s6, stepStmts
      [ Stmt.assign .nat [] "pid" (Op.programId 0),
        Stmt.assign .nat [] "pid_m"
          (Op.floorDiv .nat Broadcast.nil (Op.ref .nat [] "pid")
            (Op.div .nat Broadcast.nil
              (Op.sub .nat Broadcast.nil
                (Op.add .nat Broadcast.nil (Op.constNat N) (Op.constNat BN)) (Op.constNat 1))
              (Op.constNat BN))),
        Stmt.assign .nat [] "pid_n"
          (Op.mod .nat Broadcast.nil (Op.ref .nat [] "pid")
            (Op.div .nat Broadcast.nil
              (Op.sub .nat Broadcast.nil
                (Op.add .nat Broadcast.nil (Op.constNat N) (Op.constNat BN)) (Op.constNat 1))
              (Op.constNat BN))),
        Stmt.assign .nat [BM] "offs_am"
          (Op.mod .nat Broadcast.scalarR
            (Op.add .nat Broadcast.scalarL
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_m") (Op.constNat BM))
              (Op.arange BM))
            (Op.constNat M)),
        Stmt.assign .nat [BN] "offs_bn"
          (Op.mod .nat Broadcast.scalarR
            (Op.add .nat Broadcast.scalarL
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_n") (Op.constNat BN))
              (Op.arange BN))
            (Op.constNat N)),
        Stmt.assign .nat [BK] "offs_k" (Op.arange BK) ] s = some s6
      ∧ s6.pids = s.pids
      ∧ s6.regs .nat [] "pid_m" = some (Tile.scalar (pidM (s.pids 0) N BN))
      ∧ s6.regs .nat [] "pid_n" = some (Tile.scalar (pidN (s.pids 0) N BN))
      ∧ s6.regs .nat [BM] "offs_am" = some (Tile.vec (fun i : Fin BM => rowIndex s M N BM BN i))
      ∧ s6.regs .nat [BN] "offs_bn" = some (Tile.vec (fun j : Fin BN => colIndex s N BN j))
      ∧ s6.regs .nat [BK] "offs_k" = some (Tile.vec (fun e : Fin BK => e.val))
      ∧ s6.undef = s.undef
      ∧ s6.mem = s.mem := by
  simp only [pidM, pidN, rowIndex, colIndex, rowGlobal, colGlobal, cdiv]
  simp [stepStmts, stepStmt, evalOp_floorDiv, evalOp_mod, Option.bind, BlockState.setReg,
    Tile.bop, Tile.vec, NumericDType.add, NumericDType.mul, NumericDType.div, NumericDType.sub,
    IntegralDType.floorDiv, IntegralDType.mod]

set_option maxHeartbeats 1000000 in
/-- **preLoop** (statements 0–12): from a clean input state (`undef = 0`), the
prologue steps to a state satisfying `ffInvariant … 0`. -/
theorem preLoop (A W1 W3 OUT RMS : RegionName) (s : BlockState)
    (M N sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK numKBlocks : Nat) (K : Nat) (EPS : ℝ)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ∃ s', stepStmts ((ff_llama_surface A W1 W3 OUT RMS M N K sam sak sw1k sw1n sw3k sw3n
        soutm soutn srms BM BN BK numKBlocks EPS).toAlgKernel.body.take 13) s = some s'
      ∧ ffInvariant A W1 W3 RMS s M N BM BN BK sam sak sw1k sw1n sw3k sw3n srms numKBlocks 0 s' := by
  obtain ⟨s6, h6, hpids, hpm, hpn, hm, hn, hk, huf, hmem⟩ := preLoop_scalars s M N BM BN BK
  rw [show ((ff_llama_surface A W1 W3 OUT RMS M N K sam sak sw1k sw1n sw3k sw3n
        soutm soutn srms BM BN BK numKBlocks EPS).toAlgKernel.body.take 13)
      = [ Stmt.assign .nat [] "pid" (Op.programId 0),
          Stmt.assign .nat [] "pid_m"
            (Op.floorDiv .nat Broadcast.nil (Op.ref .nat [] "pid")
              (Op.div .nat Broadcast.nil
                (Op.sub .nat Broadcast.nil
                  (Op.add .nat Broadcast.nil (Op.constNat N) (Op.constNat BN)) (Op.constNat 1))
                (Op.constNat BN))),
          Stmt.assign .nat [] "pid_n"
            (Op.mod .nat Broadcast.nil (Op.ref .nat [] "pid")
              (Op.div .nat Broadcast.nil
                (Op.sub .nat Broadcast.nil
                  (Op.add .nat Broadcast.nil (Op.constNat N) (Op.constNat BN)) (Op.constNat 1))
                (Op.constNat BN))),
          Stmt.assign .nat [BM] "offs_am"
            (Op.mod .nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarL
                (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_m") (Op.constNat BM))
                (Op.arange BM))
              (Op.constNat M)),
          Stmt.assign .nat [BN] "offs_bn"
            (Op.mod .nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarL
                (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_n") (Op.constNat BN))
                (Op.arange BN))
              (Op.constNat N)),
          Stmt.assign .nat [BK] "offs_k" (Op.arange BK) ]
      ++ [ Stmt.assign .ptr [BM, BK] "a_ptrs"
            (Op.ptrAdd Broadcast.scalarL (Op.ptrBase A)
              (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
                (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_am")) (Op.constNat sam))
                (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BK] "offs_k")) (Op.constNat sak)))),
          Stmt.assign .ptr [BK, BN] "w1_ptrs"
            (Op.ptrAdd Broadcast.scalarL (Op.ptrBase W1)
              (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
                (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BK] "offs_k")) (Op.constNat sw1k))
                (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_bn")) (Op.constNat sw1n)))),
          Stmt.assign .ptr [BK, BN] "w3_ptrs"
            (Op.ptrAdd Broadcast.scalarL (Op.ptrBase W3)
              (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
                (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BK] "offs_k")) (Op.constNat sw3k))
                (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_bn")) (Op.constNat sw3n)))),
          Stmt.assign .real [BM, BN] "acc1" (Op.full [BM, BN] (Op.const 0)),
          Stmt.assign .real [BM, BN] "acc2" (Op.full [BM, BN] (Op.const 0)),
          Stmt.assign .ptr [1, BK] "rms_w_ptrs"
            (Op.ptrAdd Broadcast.scalarL (Op.ptrBase RMS)
              (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.arange BK)) (Op.constNat srms))),
          Stmt.assign .real [BM, BK] "a_sum" (Op.full [BM, BK] (Op.const 0)) ] from rfl,
    stepStmts.append_some h6,
    stepStmts.cons_some (stepStmt_assign_eq_some
      (aptrs_eval s6 A BM BK sam sak (fun i => rowIndex s M N BM BN i) (by simpa using hm) (by simpa using hk))),
    stepStmts.cons_some (stepStmt_assign_eq_some
      (wptrs_eval _ W1 BK BN sw1k sw1n (fun j => colIndex s N BN j) (by simp [hk]) (by simp [hn]))),
    stepStmts.cons_some (stepStmt_assign_eq_some
      (wptrs_eval _ W3 BK BN sw3k sw3n (fun j => colIndex s N BN j) (by simp [hk]) (by simp [hn]))),
    stepStmts.cons_some (stepStmt_assign_eq_some (zeros_eval [BM, BN] _)),
    stepStmts.cons_some (stepStmt_assign_eq_some (zeros_eval [BM, BN] _)),
    stepStmts.cons_some (stepStmt_assign_eq_some (rmsptrs_eval _ RMS BK srms)),
    stepStmts.cons_some (stepStmt_assign_eq_some (zeros_eval [BM, BK] _)),
    stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  refine ⟨by simp [hpids], by simp, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp [hpm]
  · simp [hpn]
  · simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx
    simp [accPartial]
  · simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx
    simp [accPartial]
  · simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx
    simp [aSumPartial]
  · simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx <;> simp [Nat.zero_mul]
  · simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx <;> simp [Nat.zero_mul]
  · simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx <;> simp [Nat.zero_mul]
  · simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx <;> simp [Nat.zero_mul]
  · intro rg o; simp [huf, hundef]
  · exact hmem

set_option maxHeartbeats 4000000 in
/-- **Step lemma**: one K-loop body iteration advances the invariant by one
block. The unmasked loads read the genuine `A`/`RMS`/`W1`/`W3` cells, the RMS
moment and the two fused dots advance the three partials, and the four
pointers advance one step. -/
theorem ff_step (A W1 W3 RMS : RegionName) (s0 : BlockState)
    (M N BM BN BK sam sak sw1k sw1n sw3k sw3n srms numKBlocks : Nat)
    (t : Nat) (s : BlockState) (hclt : t < numKBlocks)
    (hinv : ffInvariant A W1 W3 RMS s0 M N BM BN BK sam sak sw1k sw1n sw3k sw3n srms numKBlocks t s) :
    ∃ s', stepStmts (ffLoopBody BM BN BK sak sw1k sw3k srms)
        (s.setReg "_i" .nat [] (Tile.scalar t)) = some s'
      ∧ ffInvariant A W1 W3 RMS s0 M N BM BN BK sam sak sw1k sw1n sw3k sw3n srms numKBlocks (t + 1) s' := by
  simp only [ffInvariant] at hinv
  obtain ⟨hpids, hcle, hpm, hpn, hacc1, hacc2, hasum, hap, hw1p, hw3p, hrmsp, hundef, hmem⟩ := hinv
  set apT : Tile .ptr [BM, BK] :=
    ⟨fun idx : TileIndex [BM, BK] =>
      (A.cast, rowIndex s0 M N BM BN idx.1 * sam + idx.2.1.val * sak + t * BK * sak)⟩ with hapT
  set w1pT : Tile .ptr [BK, BN] :=
    ⟨fun idx : TileIndex [BK, BN] =>
      (W1.cast, idx.1.val * sw1k + colIndex s0 N BN idx.2.1 * sw1n + t * BK * sw1k)⟩ with hw1pT
  set w3pT : Tile .ptr [BK, BN] :=
    ⟨fun idx : TileIndex [BK, BN] =>
      (W3.cast, idx.1.val * sw3k + colIndex s0 N BN idx.2.1 * sw3n + t * BK * sw3k)⟩ with hw3pT
  set rmspT : Tile .ptr [1, BK] :=
    ⟨fun idx : TileIndex [1, BK] => (RMS.cast, idx.2.1.val * srms + t * BK * srms)⟩ with hrmspT
  set acc1T : Tile .real [BM, BN] :=
    ⟨fun idx : TileIndex [BM, BN] =>
      some (accPartial s0 A W1 RMS M N BM BN sam sak sw1k sw1n srms BK idx.1 idx.2.1 t)⟩ with hacc1T
  set acc2T : Tile .real [BM, BN] :=
    ⟨fun idx : TileIndex [BM, BN] =>
      some (accPartial s0 A W3 RMS M N BM BN sam sak sw3k sw3n srms BK idx.1 idx.2.1 t)⟩ with hacc2T
  set sk := s.setReg "_i" .nat [] (Tile.scalar t) with hsk
  have hrmem : ∀ (R : RegionName) (o : Nat), sk.readMem R o = s0.readMem R o := by
    intro R o; simp only [hsk, BlockState.setReg_readMem]
    unfold BlockState.readMem; rw [hmem]
  have hapk : sk.regs .ptr [BM, BK] "a_ptrs" = some apT := by simp [hsk, hap, hapT]
  have hw1pk : sk.regs .ptr [BK, BN] "w1_ptrs" = some w1pT := by simp [hsk, hw1p, hw1pT]
  have hw3pk : sk.regs .ptr [BK, BN] "w3_ptrs" = some w3pT := by simp [hsk, hw3p, hw3pT]
  have hrmspk : sk.regs .ptr [1, BK] "rms_w_ptrs" = some rmspT := by simp [hsk, hrmsp, hrmspT]
  -- canonical loaded / derived tiles
  set aT0 : Tile .real [BM, BK] :=
    ⟨fun idx : TileIndex [BM, BK] =>
      some (aElem s0 A M N BM BN sam sak idx.1 (t * BK + idx.2.1.val))⟩ with haT0
  set rmsT : Tile .real [1, BK] :=
    ⟨fun idx : TileIndex [1, BK] =>
      some (rmsElem s0 RMS srms (t * BK + idx.2.1.val))⟩ with hrmsT
  set aT1 : Tile .real [BM, BK] :=
    ⟨fun idx : TileIndex [BM, BK] =>
      some (aElem s0 A M N BM BN sam sak idx.1 (t * BK + idx.2.1.val)
        * rmsElem s0 RMS srms (t * BK + idx.2.1.val))⟩ with haT1
  set w1T : Tile .real [BK, BN] :=
    ⟨fun idx : TileIndex [BK, BN] =>
      some (wElem s0 W1 N BN sw1k sw1n idx.2.1 (t * BK + idx.1.val))⟩ with hw1T
  set w3T : Tile .real [BK, BN] :=
    ⟨fun idx : TileIndex [BK, BN] =>
      some (wElem s0 W3 N BN sw3k sw3n idx.2.1 (t * BK + idx.1.val))⟩ with hw3T
  set asumT1 : Tile .real [BM, BK] :=
    ⟨fun idx : TileIndex [BM, BK] =>
      some (aSumPartial s0 A M N BM BN sam sak BK idx.1 idx.2.1 (t + 1))⟩ with hasumT1
  unfold ffLoopBody
  -- 1: a = tl.load(a_ptrs)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (show evalOp (Op.load .real (.ptr (Op.ref .ptr [BM, BK] "a_ptrs")) MaskOpt.none) sk
          = some aT0 from by
        rw [load_ptr_none_real "a_ptrs" sk apT hapk]
        refine congrArg some ?_
        ext idx
        simp only [hapT, hrmem, Region.cast_id, haT0, aElem]
        rw [show rowIndex s0 M N BM BN idx.1 * sam + idx.2.1.val * sak + t * BK * sak
              = rowIndex s0 M N BM BN idx.1 * sam + (t * BK + idx.2.1.val) * sak from by ring]))]
  -- 2: a_sum += pow(a.to(fp32), 2)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (show evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
            (Op.ref .real [BM, BK] "a_sum")
            (Op.pow Broadcast.scalarR
              (Op.castFloat FloatDType.real FloatDType.real (Op.ref .real [BM, BK] "a"))
              (Op.const 2)))
          (sk.setReg "a" .real [BM, BK] aT0) = some asumT1 from by
        rw [evalOp_add]
        simp only [evalOp_pow, evalOp_ref, evalOp_const, Option.bind_eq_bind]
        rw [cast_real_real_eval "a" _ aT0 (by simp)]
        rw [show (sk.setReg "a" .real [BM, BK] aT0).regs .real [BM, BK] "a_sum"
              = some (⟨fun idx : TileIndex [BM, BK] =>
                  some (aSumPartial s0 A M N BM BN sam sak BK idx.1 idx.2.1 t)⟩ : Tile .real [BM, BK]) from by
            simp [hsk, hasum]]
        simp only [Option.bind_some]
        refine congrArg some ?_
        ext idx
        simp only [hasumT1, haT0, Tile.bop_data, Tile.scalar_data,
          Broadcast.leftIndex_consSame, Broadcast.rightIndex_consSame,
          Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR,
          Broadcast.leftIndex_nil, Broadcast.rightIndex_nil,
          NumericDType.add, WithBot.realAdd, WithBot.realPow_some,
          Option.map₂, Option.bind, Option.map]
        rw [aSumPartial_succ, rpow_two_eq]))]
  -- 3: rms_w = tl.load(rms_w_ptrs)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (show evalOp (Op.load .real (.ptr (Op.ref .ptr [1, BK] "rms_w_ptrs")) MaskOpt.none)
          ((sk.setReg "a" .real [BM, BK] aT0).setReg "a_sum" .real [BM, BK] asumT1)
          = some rmsT from by
        rw [load_ptr_none_real "rms_w_ptrs" _ rmspT (by simp [hrmspk])]
        refine congrArg some ?_
        ext idx
        simp only [hrmspT, BlockState.setReg_readMem, hrmem, Region.cast_id, hrmsT, rmsElem]
        rw [show idx.2.1.val * srms + t * BK * srms
              = (t * BK + idx.2.1.val) * srms from by ring]))]
  -- 4: a = a * rms_w
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (show evalOp (Op.mul .real (Broadcast.consR (Broadcast.consSame Broadcast.nil))
            (Op.ref .real [BM, BK] "a") (Op.ref .real [1, BK] "rms_w")) _
          = some aT1 from by
        rw [evalOp_mul]
        simp only [evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
          String.reduceEq, not_false_eq_true, Option.bind_eq_bind, Option.bind_some]
        refine congrArg some ?_
        ext idx
        simp only [haT0, hrmsT, haT1, Tile.bop_data,
          Broadcast.leftIndex_consR, Broadcast.leftIndex_consSame, Broadcast.leftIndex_nil,
          Broadcast.rightIndex_consR, Broadcast.rightIndex_consSame, Broadcast.rightIndex_nil,
          NumericDType.mul, WithBot.realMul, Option.map₂, Option.bind, Option.map]))]
  -- 5: b = tl.load(w1_ptrs)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (show evalOp (Op.load .real (.ptr (Op.ref .ptr [BK, BN] "w1_ptrs")) MaskOpt.none) _
          = some w1T from by
        rw [load_ptr_none_real "w1_ptrs" _ w1pT (by simp [hw1pk])]
        refine congrArg some ?_
        ext idx
        simp only [hw1pT, BlockState.setReg_readMem, hrmem, Region.cast_id, hw1T, wElem]
        rw [show idx.1.val * sw1k + colIndex s0 N BN idx.2.1 * sw1n + t * BK * sw1k
              = (t * BK + idx.1.val) * sw1k + colIndex s0 N BN idx.2.1 * sw1n from by ring]))]
  -- 6: acc1 += tl.dot(a, b)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (show evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
            (Op.ref .real [BM, BN] "acc1")
            (Op.dot (batch := []) (Op.ref .real [BM, BK] "a") (Op.ref .real [BK, BN] "b"))) _
          = some (⟨fun idx : TileIndex [BM, BN] =>
              some (accPartial s0 A W1 RMS M N BM BN sam sak sw1k sw1n srms BK idx.1 idx.2.1 (t + 1))⟩
              : Tile .real [BM, BN]) from by
        rw [accdot_eval BM BK BN _ "acc1" "b" acc1T aT1 w1T
            (by simp [hsk, hacc1, hacc1T]) (by simp) (by simp)]
        refine congrArg some ?_
        ext idx
        rw [adddot_lane BM BN acc1T (Tile.dot [] aT1 w1T) idx.1 idx.2.1
            (accPartial s0 A W1 RMS M N BM BN sam sak sw1k sw1n srms BK idx.1 idx.2.1 t)
            (Finset.univ.sum fun e : Fin BK =>
              (aElem s0 A M N BM BN sam sak idx.1 (t * BK + e.val)
                * rmsElem s0 RMS srms (t * BK + e.val))
                * wElem s0 W1 N BN sw1k sw1n idx.2.1 (t * BK + e.val))
            (by rw [hacc1T])
            (tile_dot_data BM BK BN aT1 w1T idx.1 idx.2.1 _ _
              (fun e => by rw [haT1]) (fun e => by rw [hw1T]))]
        show _ = some (accPartial s0 A W1 RMS M N BM BN sam sak sw1k sw1n srms BK idx.1 idx.2.1 (t + 1))
        rw [accPartial_succ]))]
  -- 7: c = tl.load(w3_ptrs)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (show evalOp (Op.load .real (.ptr (Op.ref .ptr [BK, BN] "w3_ptrs")) MaskOpt.none) _
          = some w3T from by
        rw [load_ptr_none_real "w3_ptrs" _ w3pT (by simp [hw3pk])]
        refine congrArg some ?_
        ext idx
        simp only [hw3pT, BlockState.setReg_readMem, hrmem, Region.cast_id, hw3T, wElem]
        rw [show idx.1.val * sw3k + colIndex s0 N BN idx.2.1 * sw3n + t * BK * sw3k
              = (t * BK + idx.1.val) * sw3k + colIndex s0 N BN idx.2.1 * sw3n from by ring]))]
  -- 8: acc2 += tl.dot(a, c)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (show evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
            (Op.ref .real [BM, BN] "acc2")
            (Op.dot (batch := []) (Op.ref .real [BM, BK] "a") (Op.ref .real [BK, BN] "c"))) _
          = some (⟨fun idx : TileIndex [BM, BN] =>
              some (accPartial s0 A W3 RMS M N BM BN sam sak sw3k sw3n srms BK idx.1 idx.2.1 (t + 1))⟩
              : Tile .real [BM, BN]) from by
        rw [accdot_eval BM BK BN _ "acc2" "c" acc2T aT1 w3T
            (by simp [hsk, hacc2, hacc2T]) (by simp) (by simp)]
        refine congrArg some ?_
        ext idx
        rw [adddot_lane BM BN acc2T (Tile.dot [] aT1 w3T) idx.1 idx.2.1
            (accPartial s0 A W3 RMS M N BM BN sam sak sw3k sw3n srms BK idx.1 idx.2.1 t)
            (Finset.univ.sum fun e : Fin BK =>
              (aElem s0 A M N BM BN sam sak idx.1 (t * BK + e.val)
                * rmsElem s0 RMS srms (t * BK + e.val))
                * wElem s0 W3 N BN sw3k sw3n idx.2.1 (t * BK + e.val))
            (by rw [hacc2T])
            (tile_dot_data BM BK BN aT1 w3T idx.1 idx.2.1 _ _
              (fun e => by rw [haT1]) (fun e => by rw [hw3T]))]
        show _ = some (accPartial s0 A W3 RMS M N BM BN sam sak sw3k sw3n srms BK idx.1 idx.2.1 (t + 1))
        rw [accPartial_succ]))]
  -- 9–12: the four pointer advances
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (ptr_adv_eval _ "a_ptrs" BK sak apT (by simp [hapk])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (ptr_adv_eval _ "w1_ptrs" BK sw1k w1pT (by simp [hw1pk])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (ptr_adv_eval _ "w3_ptrs" BK sw3k w3pT (by simp [hw3pk])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (ptr_adv_eval _ "rms_w_ptrs" BK srms rmspT (by simp [hrmspk])))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  simp only [ffInvariant]
  refine ⟨by simp [hsk, hpids], by omega, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp [hsk, hpm]
  · simp [hsk, hpn]
  · simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
  · simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
  · simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    rw [hasumT1]
  · simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx <;> simp only [Tile.ptrAdd, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
      Tile.scalar, hapT, NumericDType.add]
    ring
  · simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx <;> simp only [Tile.ptrAdd, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
      Tile.scalar, hw1pT, NumericDType.add]
    ring
  · simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx <;> simp only [Tile.ptrAdd, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
      Tile.scalar, hw3pT, NumericDType.add]
    ring
  · simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx <;> simp only [Tile.ptrAdd, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
      Tile.scalar, hrmspT, NumericDType.add]
    ring
  · intro rg o; simp [hsk, hundef]
  · show _ = s0.mem
    rw [← hmem, hsk]; rfl

/-! ## Post-loop: RMS tail + SiLU gate + fp16 cast + masked store -/

/-- The rank-2 `tl.sum(_, axis=1)` lane collapse: row `i` of the dropped-axis
sum of an all-`some` tile is `some (Σ_e f i e)`. -/
theorem reduceSumDrop_row_lane (BM BK : Nat) (f : Fin BM → Fin BK → ℝ) (idx : TileIndex [BM]) :
    (Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [BM, BK].length)
        (⟨fun idx : TileIndex [BM, BK] => some (f idx.1 idx.2.1)⟩ : Tile .real [BM, BK])).data idx
      = some (∑ e : Fin BK, f idx.1 e) := by
  show (Finset.univ.sum fun e : Fin BK => ((f idx.1 e : ℝ) : WithBot ℝ)) = _
  rw [← WithBot.coe_sum]
  rfl

/-- `a_mean = tl.sum(a_sum, axis=1) / K + EPS` eval (result shape pinned to the
literal `[BM]` so the assign statement matches the body decomposition). -/
theorem amean_eval (BM BK K : Nat) (EPS : ℝ) (st : BlockState) (f : Fin BM → Fin BK → ℝ)
    (hsum : st.regs .real [BM, BK] "a_sum"
      = some ⟨fun idx : TileIndex [BM, BK] => some (f idx.1 idx.2.1)⟩) :
    @evalOp .real [BM] (Op.add .real Broadcast.scalarR
        (Op.div .real Broadcast.scalarR
          (Op.reduceSum (⟨1, by simp⟩ : Fin [BM, BK].length) Bool.false
            (Op.ref .real [BM, BK] "a_sum"))
          (Op.const (K : ℝ)))
        (Op.const EPS)) st
      = some ⟨fun idx : TileIndex [BM] => some ((∑ e : Fin BK, f idx.1 e) / (K : ℝ) + EPS)⟩ := by
  rw [evalOp_add]
  erw [evalOp_div]
  erw [evalOp_reduceSum]
  rw [evalOp_ref, hsum]
  simp only [evalOp_const, Option.bind_eq_bind, Option.bind_some, Tile.reduceSum_false]
  refine congrArg some ?_
  ext idx
  simp only [Tile.bop_data, Tile.scalar_data,
    Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR]
  erw [reduceSumDrop_row_lane BM BK f idx]
  simp only [NumericDType.div, NumericDType.add, WithBot.realDiv, WithBot.realAdd,
    Option.map₂, Option.bind, Option.map]

/-- `a_norm = tl.math.rsqrt(a_mean)` eval. -/
theorem anorm_eval (BM : Nat) (st : BlockState) (g : Fin BM → ℝ)
    (h : st.regs .real [BM] "a_mean" = some ⟨fun idx : TileIndex [BM] => some (g idx.1)⟩) :
    evalOp (Op.rsqrt (Op.ref .real [BM] "a_mean")) st
      = some ⟨fun idx : TileIndex [BM] => some (1 / Real.sqrt (g idx.1))⟩ := by
  rw [evalOp_rsqrt, evalOp_ref, h]
  simp only [Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  simp only [Tile.uop_data, WithBot.realRsqrt_some]

@[simp] theorem evalOp_expandDim_one_real {M : Nat} (name : RegName) (s : BlockState) :
    @evalOp .real [M, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [M] name)) s =
      (s.regs .real [M] name).bind (fun v =>
        some ({ data := fun i : TileIndex [M, 1] => v.data (i.1, PUnit.unit) } : Tile .real [M, 1])) := by
  unfold evalOp; simp [Tile.expandDim]; rfl

/-- `acc = acc * a_norm[:, None]` eval. -/
theorem accscale_eval (BM BN : Nat) (st : BlockState) (name : RegName)
    (f : TileIndex [BM, BN] → ℝ) (g : Fin BM → ℝ)
    (hacc : st.regs .real [BM, BN] name = some ⟨fun idx => some (f idx)⟩)
    (hn : st.regs .real [BM] "a_norm" = some ⟨fun idx : TileIndex [BM] => some (g idx.1)⟩) :
    evalOp (Op.mul .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [BM, BN] name)
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BM] "a_norm"))) st
      = some ⟨fun idx : TileIndex [BM, BN] => some (f idx * g idx.1)⟩ := by
  rw [evalOp_mul]
  simp only [evalOp_expandDim_one_real, evalOp_ref, hacc, hn,
    Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  simp only [Tile.bop_data, Broadcast.leftIndex_consSame, Broadcast.leftIndex_consR,
    Broadcast.leftIndex_nil, Broadcast.rightIndex_consSame, Broadcast.rightIndex_consR,
    Broadcast.rightIndex_nil,
    NumericDType.mul, WithBot.realMul, Option.map₂, Option.bind, Option.map]

/-- `accumulator = (acc1 * tl.sigmoid(acc1)) * acc2` eval. -/
theorem gate_eval (BM BN : Nat) (st : BlockState)
    (f1 f2 : TileIndex [BM, BN] → ℝ)
    (h1 : st.regs .real [BM, BN] "acc1" = some ⟨fun idx => some (f1 idx)⟩)
    (h2 : st.regs .real [BM, BN] "acc2" = some ⟨fun idx => some (f2 idx)⟩) :
    evalOp (Op.mul .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.mul .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
          (Op.ref .real [BM, BN] "acc1")
          (Op.sigmoid (Op.ref .real [BM, BN] "acc1")))
        (Op.ref .real [BM, BN] "acc2")) st
      = some ⟨fun idx : TileIndex [BM, BN] =>
          some ((f1 idx * Real.sigmoid (f1 idx)) * f2 idx)⟩ := by
  rw [evalOp_mul, evalOp_mul, evalOp_sigmoid]
  simp only [evalOp_ref, h1, h2, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  simp only [Tile.bop_data, Tile.uop_data,
    Broadcast.leftIndex_consSame, Broadcast.rightIndex_consSame,
    Broadcast.leftIndex_nil, Broadcast.rightIndex_nil,
    NumericDType.mul, WithBot.realMul, WithBot.realSigmoid_some,
    Option.map₂, Option.bind, Option.map]

set_option maxHeartbeats 4000000 in
/-- **postLoop**: from the invariant at `numKBlocks` blocks, the RMS tail +
SiLU gate + fp16 cast + masked store writes the genuine closed form
`fp16( silu(n·S1) * (n·S2) )` at every active output lane (given the
output-offset map is injective). -/
theorem ff_postLoop (A W1 W3 OUT RMS : RegionName) (s0 : BlockState)
    (M N BM BN BK sam sak sw1k sw1n sw3k sw3n srms soutm soutn numKBlocks K : Nat) (EPS : ℝ)
    (hInj : Function.Injective (outOffset s0 N BM BN soutm soutn))
    (st : BlockState)
    (hinv : ffInvariant A W1 W3 RMS s0 M N BM BN BK sam sak sw1k sw1n sw3k sw3n srms
      numKBlocks numKBlocks st) :
    ∃ sfin, stepStmts (ffStoreTail OUT M N BM BN BK K soutm soutn EPS) st = some sfin
      ∧ ∀ idx : TileIndex [BM, BN],
          sfin.mem OUT (outOffset s0 N BM BN soutm soutn idx)
            = if active s0 M N BM BN idx then
                MemCell.of .fp16
                  (FloatDType.real.cast FloatDType.fp16
                    (some (ffSpec s0 A W1 W3 RMS M N BM BN sam sak sw1k sw1n sw3k sw3n srms
                      BK numKBlocks K EPS idx.1 idx.2.1)))
              else
                st.mem OUT (outOffset s0 N BM BN soutm soutn idx) := by
  simp only [ffInvariant] at hinv
  obtain ⟨hpids, hcle, hpm, hpn, hacc1, hacc2, hasum, hap, hw1p, hw3p, hrmsp, hundef, hmem⟩ := hinv
  set nrm : Fin BM → ℝ :=
    fun i => normSpec s0 A M N BM BN sam sak BK numKBlocks K EPS i with hnrm
  set s1F : TileIndex [BM, BN] → ℝ :=
    fun idx => accPartial s0 A W1 RMS M N BM BN sam sak sw1k sw1n srms BK idx.1 idx.2.1 numKBlocks
    with hs1F
  set s3F : TileIndex [BM, BN] → ℝ :=
    fun idx => accPartial s0 A W3 RMS M N BM BN sam sak sw3k sw3n srms BK idx.1 idx.2.1 numKBlocks
    with hs3F
  set accT : Tile .real [BM, BN] :=
    ⟨fun idx : TileIndex [BM, BN] =>
      some (ffSpec s0 A W1 W3 RMS M N BM BN sam sak sw1k sw1n sw3k sw3n srms
        BK numKBlocks K EPS idx.1 idx.2.1)⟩ with haccT
  set cT : Tile .fp16 [BM, BN] :=
    ⟨fun idx => FloatDType.real.cast FloatDType.fp16 (accT.data idx)⟩ with hcT
  set cpT : Tile .ptr [BM, BN] :=
    ⟨fun idx : TileIndex [BM, BN] => (OUT.cast, outOffset s0 N BM BN soutm soutn idx)⟩ with hcpT
  set cmaskT : Tile .bool [BM, BN] :=
    ⟨fun idx : TileIndex [BM, BN] =>
      (decide (rowGlobal s0 N BM BN idx.1 < M) && decide (colGlobal s0 N BN idx.2.1 < N))⟩
    with hcmaskT
  unfold ffStoreTail
  -- a_mean
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (amean_eval BM BK K EPS st
        (fun i e => aSumPartial s0 A M N BM BN sam sak BK i e numKBlocks) (by rw [hasum])))]
  -- a_norm (canonicalized to `normSpec` via `Finset.sum_comm`)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (show evalOp (Op.rsqrt (Op.ref .real [BM] "a_mean")) _
          = some (⟨fun idx : TileIndex [BM] => some (nrm idx.1)⟩ : Tile .real [BM]) from by
        rw [anorm_eval BM _
          (fun i => (∑ e : Fin BK, aSumPartial s0 A M N BM BN sam sak BK i e numKBlocks)
            / (K : ℝ) + EPS) (by simp)]
        refine congrArg some ?_
        ext idx
        simp only [hnrm, normSpec, aSumPartial]
        rw [Finset.sum_comm]))]
  -- acc1 = acc1 * a_norm[:, None]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (accscale_eval BM BN _ "acc1" s1F nrm (by simp [hacc1, hs1F]) (by simp)))]
  -- acc2 = acc2 * a_norm[:, None]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (accscale_eval BM BN _ "acc2" s3F nrm (by simp [hacc2, hs3F]) (by simp)))]
  -- accumulator = (acc1 * tl.sigmoid(acc1)) * acc2  (canonicalized to `ffSpec`)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (show evalOp (Op.mul .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
            (Op.mul .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
              (Op.ref .real [BM, BN] "acc1")
              (Op.sigmoid (Op.ref .real [BM, BN] "acc1")))
            (Op.ref .real [BM, BN] "acc2")) _
          = some accT from by
        rw [gate_eval BM BN _
          (fun idx => s1F idx * nrm idx.1) (fun idx => s3F idx * nrm idx.1)
          (by simp) (by simp)]
        refine congrArg some ?_
        ext idx
        simp only [haccT, ffSpec, hs1F, hs3F, hnrm]
        exact congrArg some (gate_lane_eq
          (normSpec s0 A M N BM BN sam sak BK numKBlocks K EPS idx.1)
          (accPartial s0 A W1 RMS M N BM BN sam sak sw1k sw1n srms BK idx.1 idx.2.1 numKBlocks)
          (accPartial s0 A W3 RMS M N BM BN sam sak sw3k sw3n srms BK idx.1 idx.2.1 numKBlocks))))]
  -- offs_outm / offs_outn
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (offsout_eval _ BM BM (pidM (s0.pids 0) N BN) "pid_m" (by simp [hpm])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (offsout_eval _ BN BN (pidN (s0.pids 0) N BN) "pid_n" (by simp [hpn])))]
  -- out_ptrs
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (show evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase OUT)
          (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
            (Op.mul .nat Broadcast.scalarL (Op.constNat soutm) (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_outm")))
            (Op.mul .nat Broadcast.scalarL (Op.constNat soutn) (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_outn"))))) _
          = some cpT from by
        rw [outptrs_eval _ OUT BM BN soutm soutn
          (fun i => rowGlobal s0 N BM BN i) (fun j => colGlobal s0 N BN j)
          (by simp [rowGlobal]) (by simp [colGlobal])]
        rfl))]
  -- out_mask
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (show evalOp (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_outm")) (Op.constNat M))
          (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_outn")) (Op.constNat N))) _
          = some cmaskT from by
        rw [outmask_eval _ M N BM BN
          (fun i => rowGlobal s0 N BM BN i) (fun j => colGlobal s0 N BN j)
          (by simp [rowGlobal]) (by simp [colGlobal])]))]
  -- abstract the post-assign state
  generalize hst9 : (((((((((st.setReg "a_mean" .real [BM]
      (⟨fun idx : TileIndex [BM] =>
        some ((∑ e : Fin BK, aSumPartial s0 A M N BM BN sam sak BK idx.1 e numKBlocks)
          / (K : ℝ) + EPS)⟩ : Tile .real [BM])).setReg "a_norm" .real [BM]
      (⟨fun idx : TileIndex [BM] => some (nrm idx.1)⟩ : Tile .real [BM])).setReg
      "acc1" .real [BM, BN]
      (⟨fun idx : TileIndex [BM, BN] => some (s1F idx * nrm idx.1)⟩ : Tile .real [BM, BN])).setReg
      "acc2" .real [BM, BN]
      (⟨fun idx : TileIndex [BM, BN] => some (s3F idx * nrm idx.1)⟩ : Tile .real [BM, BN])).setReg
      "accumulator" .real [BM, BN] accT).setReg "offs_outm" .nat [BM]
      (Tile.vec fun i : Fin BM => pidM (s0.pids 0) N BN * BM + i.val)).setReg "offs_outn" .nat [BN]
      (Tile.vec fun j : Fin BN => pidN (s0.pids 0) N BN * BN + j.val)).setReg
      "out_ptrs" .ptr [BM, BN] cpT).setReg "out_mask" .bool [BM, BN] cmaskT) = st9
  have hacc9 : st9.regs .real [BM, BN] "accumulator" = some accT := by rw [← hst9]; simp
  have hcp9 : st9.regs .ptr [BM, BN] "out_ptrs" = some cpT := by rw [← hst9]; simp
  have hcm9 : st9.regs .bool [BM, BN] "out_mask" = some cmaskT := by rw [← hst9]; simp
  have hmem9 : st9.mem = st.mem := by
    rw [← hst9]; funext region offset; simp only [BlockState.setReg_mem]
  -- the masked store
  have hstore : stepStmt (Stmt.store .fp16 [BM, BN] (.ptr (Op.ref .ptr [BM, BN] "out_ptrs"))
        (Op.castFloat FloatDType.real FloatDType.fp16 (Op.ref .real [BM, BN] "accumulator"))
        (.mask (Op.ref .bool [BM, BN] "out_mask"))) st9
      = some ((TileShape.allIndices [BM, BN]).foldl
          (fun acc i =>
            if cmaskT.data i then
              acc.writeMemTyped .fp16 OUT (outOffset s0 N BM BN soutm soutn i) (cT.data i)
            else acc) st9) := by
    simp only [stepStmt]
    erw [show evalOp (Op.castFloat FloatDType.real FloatDType.fp16 (Op.ref .real [BM, BN] "accumulator")) st9
          = some cT from by erw [evalOp_castFloat, evalOp_ref, hacc9]; rw [hcT]; rfl]
    rw [show evalOp (Op.ref .ptr [BM, BN] "out_ptrs") st9 = some cpT from by rw [evalOp_ref, hcp9]]
    rw [show evalOp (Op.ref .bool [BM, BN] "out_mask") st9 = some cmaskT from by rw [evalOp_ref, hcm9]]
    simp only [bind, Option.bind_some]
    refine congrArg some (List.foldl_ext _ _ _ (fun acc i _ => ?_))
    by_cases hmask : cmaskT.data i
    · simp only [hmask, if_true, cpT, hcpT, Region.cast_id]
    · simp only [hmask, Bool.false_eq_true, if_false]
  rw [stepStmts.cons_some hstore, stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  intro idx
  rw [scatter_memcell_fp16_prop_masked_nd (region := OUT) (s := st9)
        (offsetFn := outOffset s0 N BM BN soutm soutn)
        (valueFn := fun i => cT.data i)
        (P := fun i => cmaskT.data i = Bool.true) hInj idx]
  by_cases hact : active s0 M N BM BN idx
  · have hmasktrue : cmaskT.data idx = Bool.true := by
      simp only [hcmaskT]
      obtain ⟨hr, hcc⟩ := hact
      simp [hr, hcc]
    simp only [hmasktrue, if_true, if_pos hact]
    simp only [hcT, haccT, FloatDType.cast, FloatDType.ofReal, FloatDType.storeValue,
      FloatDType.real_toWithBot, FloatDType.fp16_ofWithBot, FloatDType.fp16_toWithBot,
      WithBot.unbotD_some]
  · have hmaskfalse : ¬ (cmaskT.data idx = Bool.true) := by
      simp only [hcmaskT, Bool.and_eq_true, decide_eq_true_eq, not_and]
      intro hr hcc
      exact hact ⟨hr, hcc⟩
    rw [if_neg hmaskfalse, if_neg hact, hmem9]

/-! ## Composition: full exec closed form -/

set_option maxHeartbeats 4000000 in
set_option linter.unusedVariables false in
/-- **Top exec reduction**: composes `preLoop` + `ff_step` (driven by
`forRange_inv`) + `ff_postLoop` into the full `exec` result. Every active
output lane's memory cell equals the genuine closed form
`fp16( silu(n·S1) * (n·S2) )`. (`hK` is the presentation hypothesis that makes
the spec's `/ K` mean the full-loop moment — the loads are unmasked, so it is
carried as an honest disclosure rather than consumed by a mask argument.) -/
theorem ff_llama_exec_closed_form (A W1 W3 OUT RMS : RegionName) (s : BlockState)
    (M N sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK numKBlocks : Nat) (K : Nat)
    (hK : K = BK * numKBlocks) (EPS : ℝ)
    (hInj : Function.Injective (outOffset s N BM BN soutm soutn))
    (hundef : ∀ rg o, s.undef rg o = 0)
    (idx : TileIndex [BM, BN]) :
    (match exec (ff_llama_surface A W1 W3 OUT RMS M N K sam sak sw1k sw1n sw3k sw3n
        soutm soutn srms BM BN BK numKBlocks EPS) s with
      | some s' => s'.mem OUT (outOffset s N BM BN soutm soutn idx)
      | none => (0 : MemCell)) =
      (if active s M N BM BN idx then
        MemCell.of .fp16
          (FloatDType.real.cast FloatDType.fp16
            (some (ffSpec s A W1 W3 RMS M N BM BN sam sak sw1k sw1n sw3k sw3n srms
              BK numKBlocks K EPS idx.1 idx.2.1)))
      else
        s.mem OUT (outOffset s N BM BN soutm soutn idx)) := by
  obtain ⟨s0, hpre_eq, hP0⟩ := preLoop A W1 W3 OUT RMS s M N sam sak sw1k sw1n sw3k sw3n
    soutm soutn srms BM BN BK numKBlocks K EPS hundef
  obtain ⟨final, sLoop, hLoopStmt, hfinal, hPLoop⟩ :=
    forRange_inv (idx := "_i") (start := 0) (stop := numKBlocks) (step := 1)
      (by omega) hP0
      (fun t st hlt hinv => by
        have := ff_step A W1 W3 RMS s M N BM BN BK sam sak sw1k sw1n sw3k sw3n srms
          numKBlocks t st hlt hinv
        simpa using this)
  have hfinalEq : final = numKBlocks := by
    have hle : final ≤ numKBlocks := by
      simp only [ffInvariant] at hPLoop
      exact hPLoop.2.1
    exact le_antisymm hle hfinal
  rw [hfinalEq] at hPLoop
  obtain ⟨sfin, hTail, hpost⟩ :=
    ff_postLoop A W1 W3 OUT RMS s M N BM BN BK sam sak sw1k sw1n sw3k sw3n srms
      soutm soutn numKBlocks K EPS hInj sLoop hPLoop
  have hpre_eq' : stepStmts (ffPrologue A W1 W3 RMS M N BM BN BK sam sak sw1k sw1n sw3k sw3n srms)
      s = some s0 := hpre_eq
  have hexec : exec (ff_llama_surface A W1 W3 OUT RMS M N K sam sak sw1k sw1n sw3k sw3n
      soutm soutn srms BM BN BK numKBlocks EPS) s = some sfin := by
    rw [exec, ff_body_split A W1 W3 OUT RMS M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms
        BM BN BK numKBlocks EPS,
      stepStmts.append_some hpre_eq', stepStmts.cons_some hLoopStmt, hTail]
  rw [hexec]
  have hsloopmem : sLoop.mem = s.mem := by
    simp only [ffInvariant] at hPLoop
    exact hPLoop.2.2.2.2.2.2.2.2.2.2.2.2
  have := hpost idx
  rw [hsloopmem] at this
  exact this

/-- **Closed-form correctness for `llama_ff_triton` (general statement).**

For arbitrary linear program id `pid`, tile dims `BM`/`BN`, K-block size `BK`,
and K-block count `numKBlocks` (so the contracted dimension is
`K = BK · numKBlocks`; the kernel's loads are unmasked, so this exact-multiple
presentation is required), every **active** output cell `(i, j)` of the
computed `BM × BN` tile equals

  `fp16( silu(n_i · S1[i,j]) * (n_i · S2[i,j]) )`

over ℝ — where `S1[i,j] = Σ_{t<numKBlocks} Σ_{e<BK} A[r(i), t·BK+e] ·
RMS[t·BK+e] · W1[t·BK+e, c(j)]` (similarly `S2` with `W3`),
`n_i = rsqrt((Σ_{t,e} A[r(i), t·BK+e]²)/K + EPS)`, `silu x = x · sigmoid x`,
`r(i) = (pid_m·BM + i) % M`, `c(j) = (pid_n·BN + j) % N` — the genuine
RMSNorm-fused SwiGLU closed form of the loaded `A`/`RMS`/`W1`/`W3` cells, NOT
the kernel's own executed value; inactive lanes are left untouched.

Layout: `A[i,k]` at `A + offs_am(i)·sam + k·sak`, `W[k,j]` at
`W + k·swk + offs_bn(j)·swn`, `RMS[k]` at `RMS + k·srms`, `OUT[i,j]` at
`OUT + soutm·offs_outm(i) + soutn·offs_outn(j)`, with `pid_m = pid //
cdiv(N,BN)`, `pid_n = pid % cdiv(N,BN)`. Preconditions: the row-major output
bounds `soutn = 1` / `BN ≤ soutm` (give output-offset injectivity) and clean
initial `undef`. -/
specification ff_llama_closed_form_correct
    (A W1 W3 OUT RMS : RegionName) (s : BlockState)
    (M N sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK numKBlocks : Nat) (K : Nat)
    (hK : K = BK * numKBlocks) (EPS : ℝ)
    (hcn : soutn = 1) (hbnle : BN ≤ soutm)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := ff_llama_surface A W1 W3 OUT RMS M N K sam sak sw1k sw1n sw3k sw3n
        soutm soutn srms BM BN BK numKBlocks EPS)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s M N BM BN)
        (fun idx => (OUT, outOffset s N BM BN soutm soutn idx)))
      (expected := fun idx : TileIndex [BM, BN] =>
        MemCell.of .fp16
          (FloatDType.real.cast FloatDType.fp16
            (some (ffSpec s A W1 W3 RMS M N BM BN sam sak sw1k sw1n sw3k sw3n srms
              BK numKBlocks K EPS idx.1 idx.2.1)))) := by
  subst hcn
  have hInj : Function.Injective (outOffset s N BM BN soutm 1) := by
    have heq : outOffset s N BM BN soutm 1
        = fun idx : TileIndex [BM, BN] =>
            (soutm * (pidM (s.pids 0) N BN * BM) + pidN (s.pids 0) N BN * BN)
              + idx.1.val * soutm + idx.2.1.val := by
      funext idx; simp only [outOffset, rowGlobal, colGlobal]; ring
    rw [heq]; exact rowMajor2D_inj _ soutm hbnle
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [ff_llama_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst hs0
  intro idx hActive
  have hmain := ff_llama_exec_closed_form A W1 W3 OUT RMS s0 M N sam sak sw1k sw1n sw3k sw3n
    soutm 1 srms BM BN BK numKBlocks K hK EPS hInj hundef idx
  have hExec2 : exec (ff_llama_surface A W1 W3 OUT RMS M N K sam sak sw1k sw1n sw3k sw3n
      soutm 1 srms BM BN BK numKBlocks EPS) s0 = some s' := hExec
  rw [hExec2] at hmain
  rw [if_pos hActive] at hmain
  simpa only [ComputeCorrect.OutputReadable.read_memcell] using hmain

end VeriTile.Bench.TritonBenchG.LlamaFfTriton
