import VeriTile.Triton
import VeriTile.Examples.Common

/-!
# `rms_matmul_rbe` — batched RMSNorm-fused GEMM + its QKV cross-JIT caller

`rms_matmul_rbe.py` contains TWO `@triton.jit` kernels and this file ports
BOTH:

1. **`rms_matmul_rbe`** (the file's first kernel) — a batched RMSNorm-fused
   GEMM, byte-identical to the kernel of the same name already ported from
   `rms_rbe_matmul.py` (where it is the second kernel): per output tile,
   `pid_batch = tl.program_id(axis=0)` picks the batch slab and the linear
   `pid = tl.program_id(axis=1)` is split into `(pid_m, pid_n)` by
   `pid // cdiv(N, BLOCK_N)` / `pid % cdiv(N, BLOCK_N)`; the row/col index
   vectors are `% M`/`% N` wrapped; the K-loop streams the RMS moment
   `x_sum += pow(x, 2)`, scales `x = x * rms_w`, and accumulates
   `accumulator += tl.dot(x, w)` while THREE pointers advance (`x_ptrs`,
   `w_ptrs`, `rms_w_ptrs`); the tail computes
   `x_norm = rsqrt(sum(x_sum, axis=1)/K + EPS)`, scales
   `accumulator * x_norm[:, None]`, re-derives the UNwrapped
   `offs_m`/`offs_n`, and masked-stores the fp16 tile. Loads are UNMASKED;
   `x_ptrs`/`out_ptrs` carry the batch offset.

2. **`rms_matmul_rbe_qkv`** (the file's second kernel; the ONLY one any host
   function in this file launches — `rms_matmul_rbe_qkv_wrapper` launches it
   over the `(batch, cdiv(M,BM)·cdiv(N,BN))` grid) — a **cross-JIT caller**:
   its whole body is three sequential calls of `rms_matmul_rbe`, at
   `(q_weight, q out)`, `(k_weight, k out)`, `(v_weight, v out)`, all sharing
   `x_ptr` and `rms_w_ptr` and the same `M`/`N`/`K`/block sizes; the X strides
   repeat across the calls and each output carries its own batch/m/n strides.

This file proves the callee correct against a genuine closed form
(`rms_matmul_rbe_closed_form_correct`, the exact twin of the
`rms_rbe_matmul` port's headline) and then the qkv caller correct as a
conjunction of three per-output realizations
(`rms_matmul_rbe_qkv_closed_form_correct`): every active Q/K/V output cell
`(b, i, j)` holds the fp16-cast of

  `n_i · S_w[i,j]`

over ℝ, where `S_w` is the RMS-scaled GEMM double sum against the pass's OWN
weight matrix (`QW`/`KW`/`VW`) and `n_i` is the SHARED RMS normalizer — with
the wrapped row `r(i) = (pid_m·BM + i) % M`, wrapped column
`c(j) = (pid_n·BN + j) % N`, and batch offset `b·sxb` in every X read:
- `S_w[i,j] = Σ_{t<numKBlocks} Σ_{e<BK} X[b, r(i), t·BK+e] · RMS[t·BK+e]
    · W[t·BK+e, c(j)]`;
- `n_i = 1/√( (Σ_{t<numKBlocks} Σ_{e<BK} X[b, r(i), t·BK+e]²) / K + EPS )`.

These are NOT the kernels' own emitted values — the double sums and the RMS
moment are derived independently of the kernels from the loaded
`X`/`W`/`RMS` cells.

## Proof architecture

```
rms_matmul_rbe_closed_form_correct       ← callee headline (Realizes_without_Rounding)
rms_matmul_rbe_qkv_closed_form_correct   ← QKV headline (∧ of three Realizes_without_Rounding,
  │                                         one per output region Q / KOut / V)
  └─ qkv_exec_closed_form                ← three-pass exec closed form
       ├─ rrm_pass ×3                    ← ONE generic inlined pass (exists + pids/undef
       │    ├─ preLoop                       preserved + cross-region frame + masked
       │    ├─ rrm_step  (via forRange_inv)  fp16 store characterization); the same
       │    └─ rrm_postLoop                  pass lemma also discharges the callee headline
       └─ state-congruence transport     ← rmsSpec/outOffset/active depend only on
            (rmsSpec_congr, outOffset_congr,  pids + the read regions, so pass 2/3
             active_congr)                    results transport to the initial state
                                              across the earlier passes' Q/K stores
```

Frame transport between passes: pass 2 (resp. 3) recomputes EVERY register
from `tl.program_id`, so only `pids` (preserved), `undef` (untouched), and
the memory of the read regions `X`/`RMS`/weight (untouched because the
earlier passes store only into their own distinct output region) need to be
carried; `rrm_pass`'s cross-region frame conjunct provides exactly that.

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); the accumulators are
over `ℝ` and the stores' `tl.float16` casts are the placeholder
`FloatDType.real.cast .fp16`. The contracted dimension is presented as
`K = BLOCK_SIZE_K · numKBlocks` so the loop trip count
`cdiv(K, BLOCK_SIZE_K) = numKBlocks` is exact — the loads are UNMASKED, so
this exact-multiple presentation is required for the loads to be meaningful
(disclosed; the host launches with `K` a multiple of the K block).
`num_stages`/`num_warps` are not modeled. The host launch (grid, batch axis,
linear-pid scheduling) is trusted; the per-program statement is universally
quantified over `s`, covering every program of the grid (`s.pids 0` is the
batch id, `s.pids 1` the linear tile pid). The `pid → (pid_m, pid_n)` split
and the `% M`/`% N` index wraps are transcribed exactly as the kernel
computes them and the spec references the same derived indices, so they are
not separate proof obligations. Output-offset injectivity is discharged from
the row-major hypotheses `son = 1` / `BN ≤ som` (resp. `sqn`/`sqm`,
`skn`/`skm`, `svn`/`svm` for the three qkv outputs; the host allocates all
three outputs contiguous fp16). The qkv headline additionally assumes the
written regions `Q`/`KOut`/`V` pairwise distinct and distinct from the read
regions `X`/`RMS` and the weight regions each later pass reads — the host
passes eight distinct buffers (disclosed); these are exactly the
no-aliasing facts the cross-pass frame transport consumes.

## Translation-surface blocker

Translation-surface blocker: (a) the second kernel `rms_matmul_rbe_qkv` is a
cross-JIT caller — its body is three sequential calls
`rms_matmul_rbe(...)`; the DSL has no jit-to-jit call surface, so the callee
body is INLINED three times in `rms_matmul_rbe_qkv_surface` (the
`attn_fwd`/`_attn_fwd_inner` inline precedent), with the per-pass weight
region/strides and output region/strides substituted (pass 1:
`q_weight`/`q`, pass 2: `k_weight`/`k`, pass 3: `v_weight`/`v`); each
inlined pass recomputes everything from `tl.program_id` — the register names
are simply reassigned across passes; (b) the callee's `USE_FP8` constexpr
arm is dropped entirely — the parameter and its two `if USE_FP8:` branches;
that path bit-reinterprets int8 bytes as `tl.float8e5` via
`.to(tl.float8e5, bitcast=True)`, a value-level bit reinterpretation with no
transcription in the ℝ-valued model (the `llama_ff_triton` dropped-constexpr
precedent) — this file ports the `USE_FP8 = False` (fp16 weights) arm ONLY;
(c) the callee's parameters `start_token_position`, `RBE_EPILOGUE`, and
`THETA` are declared but unused in its body — the caller passes them
(`RBE_EPILOGUE=True/True/False` across the three calls) but they vanish
under the inlining, and the caller's own `start_token_position`/`USE_FP8`/
`THETA` parameters are dropped as unused with them (the
`sgmv_expand_slice` unused-parameter precedent); (d) the loop trip count
`tl.cdiv(K, BLOCK_SIZE_K)` is supplied as the antiquoted `numKBlocks` binder
(so that `tl.cdiv` call does not appear as a surface statement;
`K = BLOCK_SIZE_K · numKBlocks`, the unmasked-loads presentation,
`llama_ff_triton` precedent); (e) the stores' implicit fp16 casts are
spelled `(accumulator).to(tl.float16)` — Python's stores carry no cast
because the host allocates the outputs at the weight dtype (fp16 on the
ported arm) and Triton casts implicitly at the typed pointer, while the DSL
types stores by value (the `f8_conversion_utils` precedent); (f) the loop
counters are spelled `_i` where Python spells them `_`. The textual py↔lean
scans in `bench/audit_tritonbench_g.sh` exempt this port on this marker
(registration in `proof_blockers.md` is left to the landing PR).
-/

namespace VeriTile.Bench.TritonBenchG.RmsMatmulRbe

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Faithful transcription of `rms_matmul_rbe.py`'s FIRST kernel
`rms_matmul_rbe` (`USE_FP8 = False` arm; see the Translation-surface blocker
in the module docstring for the dropped constexpr arm, the dropped unused
parameters, the `numKBlocks` presentation, the spelled-out store cast, and
the `_i` loop counter). Byte-identical to the `rms_rbe_matmul` port's target
JIT. -/
def rms_matmul_rbe_surface
    (X W RMS OUT : RegionName)
    (M N K sxb sxm sxk swk swn srms sob som son
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K numKBlocks : Nat) (EPS : ℝ) :
    ComputeKernel := triton {
  pid_batch = tl.program_id(axis=0)
  pid = tl.program_id(axis=1)
  pid_m = pid // tl.cdiv($(N), $(BLOCK_SIZE_N))
  pid_n = pid % tl.cdiv($(N), $(BLOCK_SIZE_N))
  offs_m = (pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))) % $(M)
  offs_n = (pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))) % $(N)
  offs_k = tl.arange(0, $(BLOCK_SIZE_K))
  x_ptrs = X + pid_batch * $(sxb) + offs_m[:, None] * $(sxm) + offs_k[None, :] * $(sxk)
  w_ptrs = W + offs_k[:, None] * $(swk) + offs_n[None, :] * $(swn)
  accumulator = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_N)], dtype=tl.float32)
  rms_w_ptrs = RMS + tl.arange(0, $(BLOCK_SIZE_K))[None, :] * $(srms)
  x_sum = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_K)], dtype=tl.float32)
  for _i in range($(0), $(numKBlocks), $(1)) {
    x = tl.load(x_ptrs)
    x_sum += tl.extra.cuda.libdevice.pow((x).to(tl.float32), $((2 : ℝ)))
    rms_w = tl.load(rms_w_ptrs)
    x = x * rms_w
    w = tl.load(w_ptrs)
    accumulator += tl.dot(x, w)
    x_ptrs += $(BLOCK_SIZE_K) * $(sxk)
    w_ptrs += $(BLOCK_SIZE_K) * $(swk)
    rms_w_ptrs += $(BLOCK_SIZE_K) * $(srms)
  }
  x_mean = tl.sum(x_sum, axis=1) / $(K) + $(EPS)
  x_norm = tl.math.rsqrt(x_mean)
  accumulator = accumulator * x_norm[:, None]
  offs_m = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_n = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  out_ptrs = OUT + pid_batch * $(sob) + offs_m[:, None] * $(som) + offs_n[None, :] * $(son)
  out_mask = (offs_m[:, None] < $(M)) & (offs_n[None, :] < $(N))
  tl.store(out_ptrs, (accumulator).to(tl.float16), mask=out_mask)
}

/-- Faithful transcription of `rms_matmul_rbe_qkv` (the file's SECOND kernel
and the only one launched by a host function): the callee body of
`rms_matmul_rbe` inlined three times — pass 1 at `(QW, Q)` with strides
`sqwk`/`sqwn` and `sqb`/`sqm`/`sqn`, pass 2 at `(KW, KOut)` with
`skwk`/`skwn` and `skb`/`skm`/`skn`, pass 3 at `(VW, V)` with `svwk`/`svwn`
and `svb`/`svm`/`svn`; `X`/`RMS` and the X strides `sxb`/`sxm`/`sxk` and
`srms` are shared. See the Translation-surface blocker for the cross-JIT
inline, the dropped `USE_FP8` arm, and the vanished unused parameters. -/
def rms_matmul_rbe_qkv_surface
    (X QW KW VW RMS Q KOut V : RegionName)
    (M N K sxb sxm sxk sqwk sqwn skwk skwn svwk svwn srms
      sqb sqm sqn skb skm skn svb svm svn BM BN BK numKBlocks : Nat) (EPS : ℝ) :
    ComputeKernel := triton {
  pid_batch = tl.program_id(axis=0)
  pid = tl.program_id(axis=1)
  pid_m = pid // tl.cdiv($(N), $(BN))
  pid_n = pid % tl.cdiv($(N), $(BN))
  offs_m = (pid_m * $(BM) + tl.arange(0, $(BM))) % $(M)
  offs_n = (pid_n * $(BN) + tl.arange(0, $(BN))) % $(N)
  offs_k = tl.arange(0, $(BK))
  x_ptrs = X + pid_batch * $(sxb) + offs_m[:, None] * $(sxm) + offs_k[None, :] * $(sxk)
  w_ptrs = QW + offs_k[:, None] * $(sqwk) + offs_n[None, :] * $(sqwn)
  accumulator = tl.zeros([$(BM), $(BN)], dtype=tl.float32)
  rms_w_ptrs = RMS + tl.arange(0, $(BK))[None, :] * $(srms)
  x_sum = tl.zeros([$(BM), $(BK)], dtype=tl.float32)
  for _i in range($(0), $(numKBlocks), $(1)) {
    x = tl.load(x_ptrs)
    x_sum += tl.extra.cuda.libdevice.pow((x).to(tl.float32), $((2 : ℝ)))
    rms_w = tl.load(rms_w_ptrs)
    x = x * rms_w
    w = tl.load(w_ptrs)
    accumulator += tl.dot(x, w)
    x_ptrs += $(BK) * $(sxk)
    w_ptrs += $(BK) * $(sqwk)
    rms_w_ptrs += $(BK) * $(srms)
  }
  x_mean = tl.sum(x_sum, axis=1) / $(K) + $(EPS)
  x_norm = tl.math.rsqrt(x_mean)
  accumulator = accumulator * x_norm[:, None]
  offs_m = pid_m * $(BM) + tl.arange(0, $(BM))
  offs_n = pid_n * $(BN) + tl.arange(0, $(BN))
  out_ptrs = Q + pid_batch * $(sqb) + offs_m[:, None] * $(sqm) + offs_n[None, :] * $(sqn)
  out_mask = (offs_m[:, None] < $(M)) & (offs_n[None, :] < $(N))
  tl.store(out_ptrs, (accumulator).to(tl.float16), mask=out_mask)
  pid_batch = tl.program_id(axis=0)
  pid = tl.program_id(axis=1)
  pid_m = pid // tl.cdiv($(N), $(BN))
  pid_n = pid % tl.cdiv($(N), $(BN))
  offs_m = (pid_m * $(BM) + tl.arange(0, $(BM))) % $(M)
  offs_n = (pid_n * $(BN) + tl.arange(0, $(BN))) % $(N)
  offs_k = tl.arange(0, $(BK))
  x_ptrs = X + pid_batch * $(sxb) + offs_m[:, None] * $(sxm) + offs_k[None, :] * $(sxk)
  w_ptrs = KW + offs_k[:, None] * $(skwk) + offs_n[None, :] * $(skwn)
  accumulator = tl.zeros([$(BM), $(BN)], dtype=tl.float32)
  rms_w_ptrs = RMS + tl.arange(0, $(BK))[None, :] * $(srms)
  x_sum = tl.zeros([$(BM), $(BK)], dtype=tl.float32)
  for _i in range($(0), $(numKBlocks), $(1)) {
    x = tl.load(x_ptrs)
    x_sum += tl.extra.cuda.libdevice.pow((x).to(tl.float32), $((2 : ℝ)))
    rms_w = tl.load(rms_w_ptrs)
    x = x * rms_w
    w = tl.load(w_ptrs)
    accumulator += tl.dot(x, w)
    x_ptrs += $(BK) * $(sxk)
    w_ptrs += $(BK) * $(skwk)
    rms_w_ptrs += $(BK) * $(srms)
  }
  x_mean = tl.sum(x_sum, axis=1) / $(K) + $(EPS)
  x_norm = tl.math.rsqrt(x_mean)
  accumulator = accumulator * x_norm[:, None]
  offs_m = pid_m * $(BM) + tl.arange(0, $(BM))
  offs_n = pid_n * $(BN) + tl.arange(0, $(BN))
  out_ptrs = KOut + pid_batch * $(skb) + offs_m[:, None] * $(skm) + offs_n[None, :] * $(skn)
  out_mask = (offs_m[:, None] < $(M)) & (offs_n[None, :] < $(N))
  tl.store(out_ptrs, (accumulator).to(tl.float16), mask=out_mask)
  pid_batch = tl.program_id(axis=0)
  pid = tl.program_id(axis=1)
  pid_m = pid // tl.cdiv($(N), $(BN))
  pid_n = pid % tl.cdiv($(N), $(BN))
  offs_m = (pid_m * $(BM) + tl.arange(0, $(BM))) % $(M)
  offs_n = (pid_n * $(BN) + tl.arange(0, $(BN))) % $(N)
  offs_k = tl.arange(0, $(BK))
  x_ptrs = X + pid_batch * $(sxb) + offs_m[:, None] * $(sxm) + offs_k[None, :] * $(sxk)
  w_ptrs = VW + offs_k[:, None] * $(svwk) + offs_n[None, :] * $(svwn)
  accumulator = tl.zeros([$(BM), $(BN)], dtype=tl.float32)
  rms_w_ptrs = RMS + tl.arange(0, $(BK))[None, :] * $(srms)
  x_sum = tl.zeros([$(BM), $(BK)], dtype=tl.float32)
  for _i in range($(0), $(numKBlocks), $(1)) {
    x = tl.load(x_ptrs)
    x_sum += tl.extra.cuda.libdevice.pow((x).to(tl.float32), $((2 : ℝ)))
    rms_w = tl.load(rms_w_ptrs)
    x = x * rms_w
    w = tl.load(w_ptrs)
    accumulator += tl.dot(x, w)
    x_ptrs += $(BK) * $(sxk)
    w_ptrs += $(BK) * $(svwk)
    rms_w_ptrs += $(BK) * $(srms)
  }
  x_mean = tl.sum(x_sum, axis=1) / $(K) + $(EPS)
  x_norm = tl.math.rsqrt(x_mean)
  accumulator = accumulator * x_norm[:, None]
  offs_m = pid_m * $(BM) + tl.arange(0, $(BM))
  offs_n = pid_n * $(BN) + tl.arange(0, $(BN))
  out_ptrs = V + pid_batch * $(svb) + offs_m[:, None] * $(svm) + offs_n[None, :] * $(svn)
  out_mask = (offs_m[:, None] < $(M)) & (offs_n[None, :] < $(N))
  tl.store(out_ptrs, (accumulator).to(tl.float16), mask=out_mask)
}

/-- The callee surface lowers to the algorithm layer. -/
theorem rms_matmul_rbe_surface_toAlgorithm_supported
    (X W RMS OUT : RegionName)
    (M N K sxb sxm sxk swk swn srms sob som son BM BN BK numKBlocks : Nat) (EPS : ℝ) :
    ∃ alg, (rms_matmul_rbe_surface X W RMS OUT M N K sxb sxm sxk swk swn srms sob som son
      BM BN BK numKBlocks EPS).toAlgorithm? = Except.ok alg := by
  simp [rms_matmul_rbe_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- The full three-pass qkv surface lowers to the algorithm layer. -/
theorem rms_matmul_rbe_qkv_surface_toAlgorithm_supported
    (X QW KW VW RMS Q KOut V : RegionName)
    (M N K sxb sxm sxk sqwk sqwn skwk skwn svwk svwn srms
      sqb sqm sqn skb skm skn svb svm svn BM BN BK numKBlocks : Nat) (EPS : ℝ) :
    ∃ alg, (rms_matmul_rbe_qkv_surface X QW KW VW RMS Q KOut V M N K sxb sxm sxk
      sqwk sqwn skwk skwn svwk svwn srms sqb sqm sqn skb skm skn svb svm svn
      BM BN BK numKBlocks EPS).toAlgorithm? = Except.ok alg := by
  simp [rms_matmul_rbe_qkv_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

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

/-- The identity `(x).to(tl.float32)` cast on a `.real` register. -/
theorem cast_real_real_eval {shape : TileShape} (name : RegName) (st : BlockState)
    (v : Tile .real shape) (h : st.regs .real shape name = some v) :
    evalOp (dtype := .real) (Op.castFloat FloatDType.real FloatDType.real (Op.ref .real shape name)) st
      = some v := by
  unfold evalOp
  erw [evalOp_ref, h]
  refine congrArg some ?_
  ext i
  rfl

/-- Batched row×col pointer grid eval (shared by `x_ptrs` and `out_ptrs`):
cell `(i,j) = (R, pid_batch·sb + gm i · sm + gk j · sk)`. -/
theorem batchptrs_eval (s : BlockState) (R : RegionName) (P Q sb sm sk : Nat) (pb : Nat)
    (mName kName : RegName) (gm : Fin P → Nat) (gk : Fin Q → Nat)
    (hpb : s.regs .nat [] "pid_batch" = some (Tile.scalar pb))
    (hm : s.regs .nat [P] mName = some (Tile.vec gm))
    (hk : s.regs .nat [Q] kName = some (Tile.vec gk)) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase R)
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.add .nat Broadcast.scalarL
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_batch") (Op.constNat sb))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [P] mName)) (Op.constNat sm)))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [Q] kName)) (Op.constNat sk)))) s
      = some (⟨fun idx : TileIndex [P, Q] => (R.cast, pb * sb + gm idx.1 * sm + gk idx.2.1 * sk)⟩ : Tile .ptr [P, Q]) := by
  rw [evalOp_ptrAdd, evalOp_ptrBase]
  simp only [evalOp_add, evalOp_mul, evalOp_constNat, evalOp_ref, evalOp_expandDim_one_nat,
    evalOp_expandDim_zero_nat, hpb, hm, hk, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec]
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec, Tile.scalar_data, Broadcast.leftIndex,
      Broadcast.rightIndex, NumericDType.add, NumericDType.mul]

/-- `w_ptrs` eval: cell `(e,j) = (W, offs_k e · swk + offs_n j · swn)`. -/
theorem wptrs_eval (s : BlockState) (W : RegionName) (K N swk swn : Nat) (gn : Fin N → Nat)
    (hk : s.regs .nat [K] "offs_k" = some (Tile.vec (fun e : Fin K => e.val)))
    (hn : s.regs .nat [N] "offs_n" = some (Tile.vec gn)) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase W)
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [K] "offs_k")) (Op.constNat swk))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [N] "offs_n")) (Op.constNat swn)))) s
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

/-- `ptr += BLOCK_K · stride` eval (shared by all three advancing pointers). -/
theorem ptr_adv_eval {n : Nat} {r : TileShape} (st : BlockState) (name : RegName)
    (BK stride : Nat) (p : Tile .ptr (n :: r))
    (hp : st.regs .ptr (n :: r) name = some p) :
    evalOp (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr (n :: r) name)
      (Op.mul .nat Broadcast.nil (Op.constNat BK) (Op.constNat stride))) st
      = some (Tile.ptrAdd Broadcast.scalarR p (Tile.scalar (BK * stride))) := by
  rw [evalOp_ptrAdd]
  simp [evalOp_ref, hp, evalOp_mul, evalOp_constNat, NumericDType.mul, Tile.bop]

/-- `acc += tl.dot(x, w)` statement eval (accumulator on the LEFT). -/
theorem accdot_eval (M K N : Nat) (st : BlockState) (accName wName : RegName)
    (zt : Tile .real [M, N]) (xt : Tile .real [M, K]) (yt : Tile .real [K, N])
    (hz : st.regs .real [M, N] accName = some zt)
    (hx : st.regs .real [M, K] "x" = some xt)
    (hy : st.regs .real [K, N] wName = some yt) :
    evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [M, N] accName)
        (Op.dot (batch := []) (Op.ref .real [M, K] "x") (Op.ref .real [K, N] wName))) st
      = some (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
          zt (Tile.dot [] xt yt)) := by
  have hd : evalOp (Op.dot (batch := []) (Op.ref .real [M, K] "x")
        (Op.ref .real [K, N] wName)) st = some (Tile.dot [] xt yt) := by
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

/-- `offs_m` / `offs_n` tail re-derivation eval (the un-wrapped global index,
no `% M`). -/
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

/-- `out_mask` eval: the `(offs_m < M) & (offs_n < N)` boolean tile. -/
theorem outmask_eval (s : BlockState) (M N BM BN : Nat) (gm : Fin BM → Nat) (gn : Fin BN → Nat)
    (hm : s.regs .nat [BM] "offs_m" = some (Tile.vec gm))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec gn)) :
    evalOp (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m")) (Op.constNat M))
        (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_n")) (Op.constNat N))) s
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
`% M` wrap (the kernel's tail `offs_m`). The tile pid is `s.pids 1` (axis 1;
axis 0 is the batch). -/
def rowGlobal (s : BlockState) (N BM BN : Nat) (i : Fin BM) : Nat :=
  pidM (s.pids 1) N BN * BM + i.val

/-- Global output column of tile lane `j`: `pid_n · BLOCK_N + j`, before wrap. -/
def colGlobal (s : BlockState) (N BN : Nat) (j : Fin BN) : Nat :=
  pidN (s.pids 1) N BN * BN + j.val

/-- The `% M`-wrapped X-row index of tile lane `i` (the kernel's loop `offs_m`). -/
def rowIndex (s : BlockState) (M N BM BN : Nat) (i : Fin BM) : Nat :=
  rowGlobal s N BM BN i % M

/-- The `% N`-wrapped W-column index of tile lane `j` (the kernel's loop `offs_n`). -/
def colIndex (s : BlockState) (N BN : Nat) (j : Fin BN) : Nat :=
  colGlobal s N BN j % N

/-- `X[b, i, k] = readMem X (pid_batch · sxb + offs_m i · sxm + k · sxk)`
(the batch offset `s.pids 0 · sxb` is in every X read). -/
noncomputable def xElem (s : BlockState) (X : RegionName) (M N BM BN sxb sxm sxk : Nat)
    (i : Fin BM) (k : Nat) : ℝ :=
  s.readMem X (s.pids 0 * sxb + rowIndex s M N BM BN i * sxm + k * sxk)

/-- `RMS[k] = readMem RMS (k · srms)`. -/
noncomputable def rmsElem (s : BlockState) (RMS : RegionName) (srms : Nat) (k : Nat) : ℝ :=
  s.readMem RMS (k * srms)

/-- `W[k, j] = readMem W (k · swk + offs_n j · swn)`. -/
noncomputable def wElem (s : BlockState) (W : RegionName) (N BN swk swn : Nat)
    (j : Fin BN) (k : Nat) : ℝ :=
  s.readMem W (k * swk + colIndex s N BN j * swn)

/-- Partial RMS-scaled GEMM accumulator after `c` K-blocks:
`Σ_{t<c} Σ_{e<BK} X[b, r(i), t·BK+e] · RMS[t·BK+e] · W[t·BK+e, c(j)]`. -/
noncomputable def accPartial (s : BlockState) (X W RMS : RegionName)
    (M N BM BN sxb sxm sxk swk swn srms BK : Nat) (i : Fin BM) (j : Fin BN) (c : Nat) : ℝ :=
  ∑ t : Fin c, ∑ e : Fin BK,
    xElem s X M N BM BN sxb sxm sxk i (t.val * BK + e.val)
      * rmsElem s RMS srms (t.val * BK + e.val)
      * wElem s W N BN swk swn j (t.val * BK + e.val)

/-- Partial RMS moment after `c` K-blocks (the `x_sum` lane `(i,e)`):
`Σ_{t<c} X[b, r(i), t·BK+e]²`. -/
noncomputable def xSumPartial (s : BlockState) (X : RegionName)
    (M N BM BN sxb sxm sxk BK : Nat) (i : Fin BM) (e : Fin BK) (c : Nat) : ℝ :=
  ∑ t : Fin c, xElem s X M N BM BN sxb sxm sxk i (t.val * BK + e.val) ^ 2

theorem accPartial_succ (s : BlockState) (X W RMS : RegionName)
    (M N BM BN sxb sxm sxk swk swn srms BK : Nat) (i : Fin BM) (j : Fin BN) (c : Nat) :
    accPartial s X W RMS M N BM BN sxb sxm sxk swk swn srms BK i j (c + 1)
      = accPartial s X W RMS M N BM BN sxb sxm sxk swk swn srms BK i j c
        + (Finset.univ.sum fun e : Fin BK =>
            xElem s X M N BM BN sxb sxm sxk i (c * BK + e.val)
              * rmsElem s RMS srms (c * BK + e.val)
              * wElem s W N BN swk swn j (c * BK + e.val)) := by
  unfold accPartial
  rw [Fin.sum_univ_castSucc]
  simp

theorem xSumPartial_succ (s : BlockState) (X : RegionName)
    (M N BM BN sxb sxm sxk BK : Nat) (i : Fin BM) (e : Fin BK) (c : Nat) :
    xSumPartial s X M N BM BN sxb sxm sxk BK i e (c + 1)
      = xSumPartial s X M N BM BN sxb sxm sxk BK i e c
        + xElem s X M N BM BN sxb sxm sxk i (c * BK + e.val) ^ 2 := by
  unfold xSumPartial
  rw [Fin.sum_univ_castSucc]
  simp

/-- **The RMS normalization factor** (over ℝ):
`n_i = rsqrt( (Σ_{t<numKBlocks} Σ_{e<BK} X[b, r(i), t·BK+e]²) / K + EPS )`. -/
noncomputable def normSpec (s : BlockState) (X : RegionName)
    (M N BM BN sxb sxm sxk BK numKBlocks K : Nat) (EPS : ℝ) (i : Fin BM) : ℝ :=
  1 / Real.sqrt
    ((∑ t : Fin numKBlocks, ∑ e : Fin BK,
        xElem s X M N BM BN sxb sxm sxk i (t.val * BK + e.val) ^ 2) / (K : ℝ) + EPS)

/-- **Genuine batched RMSNorm-fused GEMM spec** (over ℝ):
`OUT[b,i,j] = n_i · S[i,j]` with `S` the full-loop RMS-scaled GEMM sum and
`n_i` the RMS factor. -/
noncomputable def rmsSpec (s : BlockState) (X W RMS : RegionName)
    (M N BM BN sxb sxm sxk swk swn srms BK numKBlocks K : Nat) (EPS : ℝ)
    (i : Fin BM) (j : Fin BN) : ℝ :=
  normSpec s X M N BM BN sxb sxm sxk BK numKBlocks K EPS i
    * accPartial s X W RMS M N BM BN sxb sxm sxk swk swn srms BK i j numKBlocks

/-- The output store address for tile lane `(i,j)`:
`pid_batch · sob + offs_m i · som + offs_n j · son` (un-wrapped global indices). -/
def outOffset (s : BlockState) (N BM BN sob som son : Nat) (idx : TileIndex [BM, BN]) : Nat :=
  s.pids 0 * sob + rowGlobal s N BM BN idx.1 * som + colGlobal s N BN idx.2.1 * son

/-- The boundary predicate `(row < M) & (col < N)` for tile lane `(i,j)`. -/
def active (s : BlockState) (M N BM BN : Nat) (idx : TileIndex [BM, BN]) : Prop :=
  rowGlobal s N BM BN idx.1 < M ∧ colGlobal s N BN idx.2.1 < N

instance activeDecidable (s : BlockState) (M N BM BN : Nat)
    (idx : TileIndex [BM, BN]) : Decidable (active s M N BM BN idx) := by
  unfold active; infer_instance

/-! ## State-congruence transport

The spec layer depends on the state only through `pids` and the memory of
the read regions; the address/mask layer only through `pids`. These
congruence lemmas transport a later pass's spec/addresses (expressed at that
pass's entry state) back to the initial state, given the earlier passes only
stored into distinct regions. -/

theorem xElem_congr {sA sB : BlockState} {X : RegionName}
    {M N BM BN sxb sxm sxk : Nat} {i : Fin BM} {k : Nat}
    (hpids : sA.pids = sB.pids) (hX : ∀ o, sA.mem X o = sB.mem X o) :
    xElem sA X M N BM BN sxb sxm sxk i k = xElem sB X M N BM BN sxb sxm sxk i k := by
  unfold xElem rowIndex rowGlobal pidM
  rw [hpids]
  unfold BlockState.readMem
  rw [hX]

theorem rmsElem_congr {sA sB : BlockState} {RMS : RegionName} {srms k : Nat}
    (hRMS : ∀ o, sA.mem RMS o = sB.mem RMS o) :
    rmsElem sA RMS srms k = rmsElem sB RMS srms k := by
  unfold rmsElem
  unfold BlockState.readMem
  rw [hRMS]

theorem wElem_congr {sA sB : BlockState} {W : RegionName}
    {N BN swk swn : Nat} {j : Fin BN} {k : Nat}
    (hpids : sA.pids = sB.pids) (hW : ∀ o, sA.mem W o = sB.mem W o) :
    wElem sA W N BN swk swn j k = wElem sB W N BN swk swn j k := by
  unfold wElem colIndex colGlobal pidN
  rw [hpids]
  unfold BlockState.readMem
  rw [hW]

theorem accPartial_congr {sA sB : BlockState} {X W RMS : RegionName}
    {M N BM BN sxb sxm sxk swk swn srms BK : Nat} {i : Fin BM} {j : Fin BN} {c : Nat}
    (hpids : sA.pids = sB.pids)
    (hX : ∀ o, sA.mem X o = sB.mem X o)
    (hW : ∀ o, sA.mem W o = sB.mem W o)
    (hRMS : ∀ o, sA.mem RMS o = sB.mem RMS o) :
    accPartial sA X W RMS M N BM BN sxb sxm sxk swk swn srms BK i j c
      = accPartial sB X W RMS M N BM BN sxb sxm sxk swk swn srms BK i j c := by
  unfold accPartial
  refine Finset.sum_congr rfl fun t _ => Finset.sum_congr rfl fun e _ => ?_
  rw [xElem_congr hpids hX, rmsElem_congr hRMS, wElem_congr hpids hW]

theorem normSpec_congr {sA sB : BlockState} {X : RegionName}
    {M N BM BN sxb sxm sxk BK numKBlocks K : Nat} {EPS : ℝ} {i : Fin BM}
    (hpids : sA.pids = sB.pids) (hX : ∀ o, sA.mem X o = sB.mem X o) :
    normSpec sA X M N BM BN sxb sxm sxk BK numKBlocks K EPS i
      = normSpec sB X M N BM BN sxb sxm sxk BK numKBlocks K EPS i := by
  unfold normSpec
  have hsum : (∑ t : Fin numKBlocks, ∑ e : Fin BK,
        xElem sA X M N BM BN sxb sxm sxk i (t.val * BK + e.val) ^ 2)
      = ∑ t : Fin numKBlocks, ∑ e : Fin BK,
          xElem sB X M N BM BN sxb sxm sxk i (t.val * BK + e.val) ^ 2 :=
    Finset.sum_congr rfl fun t _ => Finset.sum_congr rfl fun e _ => by
      rw [xElem_congr hpids hX]
  rw [hsum]

theorem rmsSpec_congr {sA sB : BlockState} {X W RMS : RegionName}
    {M N BM BN sxb sxm sxk swk swn srms BK numKBlocks K : Nat} {EPS : ℝ}
    {i : Fin BM} {j : Fin BN}
    (hpids : sA.pids = sB.pids)
    (hX : ∀ o, sA.mem X o = sB.mem X o)
    (hW : ∀ o, sA.mem W o = sB.mem W o)
    (hRMS : ∀ o, sA.mem RMS o = sB.mem RMS o) :
    rmsSpec sA X W RMS M N BM BN sxb sxm sxk swk swn srms BK numKBlocks K EPS i j
      = rmsSpec sB X W RMS M N BM BN sxb sxm sxk swk swn srms BK numKBlocks K EPS i j := by
  unfold rmsSpec
  rw [normSpec_congr hpids hX, accPartial_congr hpids hX hW hRMS]

theorem outOffset_congr {sA sB : BlockState} {N BM BN sob som son : Nat}
    (hpids : sA.pids = sB.pids) (idx : TileIndex [BM, BN]) :
    outOffset sA N BM BN sob som son idx = outOffset sB N BM BN sob som son idx := by
  unfold outOffset rowGlobal colGlobal pidM pidN
  rw [hpids]

theorem active_congr {sA sB : BlockState} {M N BM BN : Nat}
    (hpids : sA.pids = sB.pids) (idx : TileIndex [BM, BN]) :
    active sA M N BM BN idx ↔ active sB M N BM BN idx := by
  unfold active rowGlobal colGlobal pidM pidN
  rw [hpids]

/-! ## Body decomposition -/

/-- The 12-statement prologue: batch pid, pid split, index vectors, the three
pointer seeds, and the two zero accumulators. -/
def rrmPrologue (X W RMS : RegionName)
    (M N BM BN BK sxb sxm sxk swk swn srms : Nat) : List Stmt :=
  [ Stmt.assign .nat [] "pid_batch" (Op.programId 0),
    Stmt.assign .nat [] "pid" (Op.programId 1),
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
    Stmt.assign .nat [BM] "offs_m"
      (Op.mod .nat Broadcast.scalarR
        (Op.add .nat Broadcast.scalarL
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_m") (Op.constNat BM))
          (Op.arange BM))
        (Op.constNat M)),
    Stmt.assign .nat [BN] "offs_n"
      (Op.mod .nat Broadcast.scalarR
        (Op.add .nat Broadcast.scalarL
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_n") (Op.constNat BN))
          (Op.arange BN))
        (Op.constNat N)),
    Stmt.assign .nat [BK] "offs_k" (Op.arange BK),
    Stmt.assign .ptr [BM, BK] "x_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase X)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.add .nat Broadcast.scalarL
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_batch") (Op.constNat sxb))
            (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m")) (Op.constNat sxm)))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BK] "offs_k")) (Op.constNat sxk)))),
    Stmt.assign .ptr [BK, BN] "w_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase W)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BK] "offs_k")) (Op.constNat swk))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_n")) (Op.constNat swn)))),
    Stmt.assign .real [BM, BN] "accumulator" (Op.full [BM, BN] (Op.const 0)),
    Stmt.assign .ptr [1, BK] "rms_w_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase RMS)
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.arange BK)) (Op.constNat srms))),
    Stmt.assign .real [BM, BK] "x_sum" (Op.full [BM, BK] (Op.const 0)) ]

/-- The 9-statement K-loop body (unmasked loads + RMS moment + fused dot +
three pointer advances). -/
def rrmLoopBody (BM BN BK sxk swk srms : Nat) : List Stmt :=
  [ Stmt.assign .real [BM, BK] "x"
      (Op.load .real (.ptr (Op.ref .ptr [BM, BK] "x_ptrs")) MaskOpt.none),
    Stmt.assign .real [BM, BK] "x_sum"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BM, BK] "x_sum")
        (Op.pow Broadcast.scalarR
          (Op.castFloat FloatDType.real FloatDType.real (Op.ref .real [BM, BK] "x"))
          (Op.const 2))),
    Stmt.assign .real [1, BK] "rms_w"
      (Op.load .real (.ptr (Op.ref .ptr [1, BK] "rms_w_ptrs")) MaskOpt.none),
    Stmt.assign .real [BM, BK] "x"
      (Op.mul .real (Broadcast.consR (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BM, BK] "x") (Op.ref .real [1, BK] "rms_w")),
    Stmt.assign .real [BK, BN] "w"
      (Op.load .real (.ptr (Op.ref .ptr [BK, BN] "w_ptrs")) MaskOpt.none),
    Stmt.assign .real [BM, BN] "accumulator"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BM, BN] "accumulator")
        (Op.dot (batch := []) (Op.ref .real [BM, BK] "x") (Op.ref .real [BK, BN] "w"))),
    Stmt.assign .ptr [BM, BK] "x_ptrs"
      (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BM, BK] "x_ptrs")
        (Op.mul .nat Broadcast.nil (Op.constNat BK) (Op.constNat sxk))),
    Stmt.assign .ptr [BK, BN] "w_ptrs"
      (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BK, BN] "w_ptrs")
        (Op.mul .nat Broadcast.nil (Op.constNat BK) (Op.constNat swk))),
    Stmt.assign .ptr [1, BK] "rms_w_ptrs"
      (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [1, BK] "rms_w_ptrs")
        (Op.mul .nat Broadcast.nil (Op.constNat BK) (Op.constNat srms))) ]

/-- The 8-statement post-loop tail: mean/rsqrt, the `x_norm` scaling, the two
re-derived `offs_m`/`offs_n` vectors, `out_ptrs` (with the batch offset),
`out_mask`, and the masked fp16 store. -/
def rrmStoreTail (OUT : RegionName) (M N BM BN BK K sob som son : Nat) (EPS : ℝ) : List Stmt :=
  [ Stmt.assign .real [BM] "x_mean"
      (Op.add .real Broadcast.scalarR
        (Op.div .real Broadcast.scalarR
          (Op.reduceSum (⟨1, by simp⟩ : Fin [BM, BK].length) Bool.false
            (Op.ref .real [BM, BK] "x_sum"))
          (Op.const (K : ℝ)))
        (Op.const EPS)),
    Stmt.assign .real [BM] "x_norm" (Op.rsqrt (Op.ref .real [BM] "x_mean")),
    Stmt.assign .real [BM, BN] "accumulator"
      (Op.mul .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [BM, BN] "accumulator")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BM] "x_norm"))),
    Stmt.assign .nat [BM] "offs_m"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_m") (Op.constNat BM))
        (Op.arange BM)),
    Stmt.assign .nat [BN] "offs_n"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_n") (Op.constNat BN))
        (Op.arange BN)),
    Stmt.assign .ptr [BM, BN] "out_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase OUT)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.add .nat Broadcast.scalarL
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_batch") (Op.constNat sob))
            (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m")) (Op.constNat som)))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_n")) (Op.constNat son)))),
    Stmt.assign .bool [BM, BN] "out_mask"
      (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m")) (Op.constNat M))
        (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_n")) (Op.constNat N))),
    Stmt.store .fp16 [BM, BN] (.ptr (Op.ref .ptr [BM, BN] "out_ptrs"))
      (Op.castFloat FloatDType.real FloatDType.fp16 (Op.ref .real [BM, BN] "accumulator"))
      (.mask (Op.ref .bool [BM, BN] "out_mask")) ]

/-- One full inlined pass of the callee body:
prologue (12) ++ [K-loop] ++ store-tail (8). -/
def rrmPassBody (X W RMS OUT : RegionName)
    (M N K sxb sxm sxk swk swn srms sob som son BM BN BK numKBlocks : Nat) (EPS : ℝ) :
    List Stmt :=
  rrmPrologue X W RMS M N BM BN BK sxb sxm sxk swk swn srms
    ++ (Stmt.forRange "_i" 0 numKBlocks 1 (rrmLoopBody BM BN BK sxk swk srms)
        :: rrmStoreTail OUT M N BM BN BK K sob som son EPS)

/-- Callee body decomposition: one pass. By `rfl`. -/
theorem rrm_body_split (X W RMS OUT : RegionName)
    (M N K sxb sxm sxk swk swn srms sob som son BM BN BK numKBlocks : Nat) (EPS : ℝ) :
    (rms_matmul_rbe_surface X W RMS OUT M N K sxb sxm sxk swk swn srms sob som son
        BM BN BK numKBlocks EPS).toAlgKernel.body
      = rrmPassBody X W RMS OUT M N K sxb sxm sxk swk swn srms sob som son
          BM BN BK numKBlocks EPS := by
  rfl

/-- qkv body decomposition: three substituted passes. By `rfl`. -/
theorem qkv_body_split (X QW KW VW RMS Q KOut V : RegionName)
    (M N K sxb sxm sxk sqwk sqwn skwk skwn svwk svwn srms
      sqb sqm sqn skb skm skn svb svm svn BM BN BK numKBlocks : Nat) (EPS : ℝ) :
    (rms_matmul_rbe_qkv_surface X QW KW VW RMS Q KOut V M N K sxb sxm sxk
        sqwk sqwn skwk skwn svwk svwn srms sqb sqm sqn skb skm skn svb svm svn
        BM BN BK numKBlocks EPS).toAlgKernel.body
      = rrmPassBody X QW RMS Q M N K sxb sxm sxk sqwk sqwn srms sqb sqm sqn
          BM BN BK numKBlocks EPS
        ++ (rrmPassBody X KW RMS KOut M N K sxb sxm sxk skwk skwn srms skb skm skn
              BM BN BK numKBlocks EPS
            ++ rrmPassBody X VW RMS V M N K sxb sxm sxk svwk svwn srms svb svm svn
                BM BN BK numKBlocks EPS) := by
  rfl

/-! ## Loop invariant -/

/-- **Loop invariant** (counter `c = block index`, step `1`).

After `c` K-blocks: program ids and `mem`/`undef` fixed; `pid_batch`/`pid_m`/
`pid_n` seeded; the two accumulators `accumulator`/`x_sum` hold their partials;
the three pointers `x_ptrs`/`w_ptrs`/`rms_w_ptrs` advanced by `c` blocks. -/
noncomputable def rrmInvariant (X W RMS : RegionName) (s0 : BlockState)
    (M N BM BN BK sxb sxm sxk swk swn srms numKBlocks : Nat)
    (c : Nat) (s : BlockState) : Prop :=
  s.pids = s0.pids ∧ c ≤ numKBlocks ∧
  (s.regs .nat [] "pid_batch" = some (Tile.scalar (s0.pids 0))) ∧
  (s.regs .nat [] "pid_m" = some (Tile.scalar (pidM (s0.pids 1) N BN))) ∧
  (s.regs .nat [] "pid_n" = some (Tile.scalar (pidN (s0.pids 1) N BN))) ∧
  (s.regs .real [BM, BN] "accumulator" = some ⟨fun idx : TileIndex [BM, BN] =>
      some (accPartial s0 X W RMS M N BM BN sxb sxm sxk swk swn srms BK idx.1 idx.2.1 c)⟩) ∧
  (s.regs .real [BM, BK] "x_sum" = some ⟨fun idx : TileIndex [BM, BK] =>
      some (xSumPartial s0 X M N BM BN sxb sxm sxk BK idx.1 idx.2.1 c)⟩) ∧
  (s.regs .ptr [BM, BK] "x_ptrs" = some ⟨fun idx : TileIndex [BM, BK] =>
      (X.cast, s0.pids 0 * sxb + rowIndex s0 M N BM BN idx.1 * sxm + idx.2.1.val * sxk
        + c * BK * sxk)⟩) ∧
  (s.regs .ptr [BK, BN] "w_ptrs" = some ⟨fun idx : TileIndex [BK, BN] =>
      (W.cast, idx.1.val * swk + colIndex s0 N BN idx.2.1 * swn + c * BK * swk)⟩) ∧
  (s.regs .ptr [1, BK] "rms_w_ptrs" = some ⟨fun idx : TileIndex [1, BK] =>
      (RMS.cast, idx.2.1.val * srms + c * BK * srms)⟩) ∧
  (∀ rg o, s.undef rg o = 0) ∧ (s.mem = s0.mem)

/-- **preLoop scalars** (statements 0–6): batch pid, the pid split, and the 3
index vectors. -/
theorem preLoop_scalars (s : BlockState) (M N BM BN BK : Nat) :
    ∃ s7, stepStmts
      [ Stmt.assign .nat [] "pid_batch" (Op.programId 0),
        Stmt.assign .nat [] "pid" (Op.programId 1),
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
        Stmt.assign .nat [BM] "offs_m"
          (Op.mod .nat Broadcast.scalarR
            (Op.add .nat Broadcast.scalarL
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_m") (Op.constNat BM))
              (Op.arange BM))
            (Op.constNat M)),
        Stmt.assign .nat [BN] "offs_n"
          (Op.mod .nat Broadcast.scalarR
            (Op.add .nat Broadcast.scalarL
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_n") (Op.constNat BN))
              (Op.arange BN))
            (Op.constNat N)),
        Stmt.assign .nat [BK] "offs_k" (Op.arange BK) ] s = some s7
      ∧ s7.pids = s.pids
      ∧ s7.regs .nat [] "pid_batch" = some (Tile.scalar (s.pids 0))
      ∧ s7.regs .nat [] "pid_m" = some (Tile.scalar (pidM (s.pids 1) N BN))
      ∧ s7.regs .nat [] "pid_n" = some (Tile.scalar (pidN (s.pids 1) N BN))
      ∧ s7.regs .nat [BM] "offs_m" = some (Tile.vec (fun i : Fin BM => rowIndex s M N BM BN i))
      ∧ s7.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => colIndex s N BN j))
      ∧ s7.regs .nat [BK] "offs_k" = some (Tile.vec (fun e : Fin BK => e.val))
      ∧ s7.undef = s.undef
      ∧ s7.mem = s.mem := by
  simp only [pidM, pidN, rowIndex, colIndex, rowGlobal, colGlobal, cdiv]
  simp [stepStmts, stepStmt, evalOp_floorDiv, evalOp_mod, Option.bind, BlockState.setReg,
    Tile.bop, Tile.vec, NumericDType.add, NumericDType.mul, NumericDType.div, NumericDType.sub,
    IntegralDType.floorDiv, IntegralDType.mod]

set_option maxHeartbeats 1000000 in
/-- **preLoop** (statements 0–11): from a clean input state (`undef = 0`), the
prologue steps to a state satisfying `rrmInvariant … 0`. -/
theorem preLoop (X W RMS : RegionName) (s : BlockState)
    (M N sxb sxm sxk swk swn srms BM BN BK numKBlocks : Nat)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ∃ s', stepStmts (rrmPrologue X W RMS M N BM BN BK sxb sxm sxk swk swn srms) s = some s'
      ∧ rrmInvariant X W RMS s M N BM BN BK sxb sxm sxk swk swn srms numKBlocks 0 s' := by
  obtain ⟨s7, h7, hpids, hpb, hpm, hpn, hm, hn, hk, huf, hmem⟩ := preLoop_scalars s M N BM BN BK
  rw [show rrmPrologue X W RMS M N BM BN BK sxb sxm sxk swk swn srms
      = [ Stmt.assign .nat [] "pid_batch" (Op.programId 0),
          Stmt.assign .nat [] "pid" (Op.programId 1),
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
          Stmt.assign .nat [BM] "offs_m"
            (Op.mod .nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarL
                (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_m") (Op.constNat BM))
                (Op.arange BM))
              (Op.constNat M)),
          Stmt.assign .nat [BN] "offs_n"
            (Op.mod .nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarL
                (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_n") (Op.constNat BN))
                (Op.arange BN))
              (Op.constNat N)),
          Stmt.assign .nat [BK] "offs_k" (Op.arange BK) ]
      ++ [ Stmt.assign .ptr [BM, BK] "x_ptrs"
            (Op.ptrAdd Broadcast.scalarL (Op.ptrBase X)
              (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
                (Op.add .nat Broadcast.scalarL
                  (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_batch") (Op.constNat sxb))
                  (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m")) (Op.constNat sxm)))
                (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BK] "offs_k")) (Op.constNat sxk)))),
          Stmt.assign .ptr [BK, BN] "w_ptrs"
            (Op.ptrAdd Broadcast.scalarL (Op.ptrBase W)
              (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
                (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BK] "offs_k")) (Op.constNat swk))
                (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_n")) (Op.constNat swn)))),
          Stmt.assign .real [BM, BN] "accumulator" (Op.full [BM, BN] (Op.const 0)),
          Stmt.assign .ptr [1, BK] "rms_w_ptrs"
            (Op.ptrAdd Broadcast.scalarL (Op.ptrBase RMS)
              (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.arange BK)) (Op.constNat srms))),
          Stmt.assign .real [BM, BK] "x_sum" (Op.full [BM, BK] (Op.const 0)) ] from rfl,
    stepStmts.append_some h7,
    stepStmts.cons_some (stepStmt_assign_eq_some
      (batchptrs_eval s7 X BM BK sxb sxm sxk (s.pids 0) "offs_m" "offs_k"
        (fun i => rowIndex s M N BM BN i) (fun e : Fin BK => e.val)
        (by simpa using hpb) (by simpa using hm) (by simpa using hk))),
    stepStmts.cons_some (stepStmt_assign_eq_some
      (wptrs_eval _ W BK BN swk swn (fun j => colIndex s N BN j) (by simp [hk]) (by simp [hn]))),
    stepStmts.cons_some (stepStmt_assign_eq_some (zeros_eval [BM, BN] _)),
    stepStmts.cons_some (stepStmt_assign_eq_some (rmsptrs_eval _ RMS BK srms)),
    stepStmts.cons_some (stepStmt_assign_eq_some (zeros_eval [BM, BK] _)),
    stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  refine ⟨by simp [hpids], by simp, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp [hpb]
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
    simp [xSumPartial]
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
block. The unmasked loads read the genuine `X`/`RMS`/`W` cells, the RMS moment
and the fused dot advance the two partials, and the three pointers advance one
step. -/
theorem rrm_step (X W RMS : RegionName) (s0 : BlockState)
    (M N BM BN BK sxb sxm sxk swk swn srms numKBlocks : Nat)
    (t : Nat) (s : BlockState) (hclt : t < numKBlocks)
    (hinv : rrmInvariant X W RMS s0 M N BM BN BK sxb sxm sxk swk swn srms numKBlocks t s) :
    ∃ s', stepStmts (rrmLoopBody BM BN BK sxk swk srms)
        (s.setReg "_i" .nat [] (Tile.scalar t)) = some s'
      ∧ rrmInvariant X W RMS s0 M N BM BN BK sxb sxm sxk swk swn srms numKBlocks (t + 1) s' := by
  simp only [rrmInvariant] at hinv
  obtain ⟨hpids, hcle, hpb, hpm, hpn, hacc, hxsum, hxp, hwp, hrmsp, hundef, hmem⟩ := hinv
  set xpT : Tile .ptr [BM, BK] :=
    ⟨fun idx : TileIndex [BM, BK] =>
      (X.cast, s0.pids 0 * sxb + rowIndex s0 M N BM BN idx.1 * sxm + idx.2.1.val * sxk
        + t * BK * sxk)⟩ with hxpT
  set wpT : Tile .ptr [BK, BN] :=
    ⟨fun idx : TileIndex [BK, BN] =>
      (W.cast, idx.1.val * swk + colIndex s0 N BN idx.2.1 * swn + t * BK * swk)⟩ with hwpT
  set rmspT : Tile .ptr [1, BK] :=
    ⟨fun idx : TileIndex [1, BK] => (RMS.cast, idx.2.1.val * srms + t * BK * srms)⟩ with hrmspT
  set accT : Tile .real [BM, BN] :=
    ⟨fun idx : TileIndex [BM, BN] =>
      some (accPartial s0 X W RMS M N BM BN sxb sxm sxk swk swn srms BK idx.1 idx.2.1 t)⟩ with haccT
  set sk := s.setReg "_i" .nat [] (Tile.scalar t) with hsk
  have hrmem : ∀ (R : RegionName) (o : Nat), sk.readMem R o = s0.readMem R o := by
    intro R o; simp only [hsk, BlockState.setReg_readMem]
    unfold BlockState.readMem; rw [hmem]
  have hxpk : sk.regs .ptr [BM, BK] "x_ptrs" = some xpT := by simp [hsk, hxp, hxpT]
  have hwpk : sk.regs .ptr [BK, BN] "w_ptrs" = some wpT := by simp [hsk, hwp, hwpT]
  have hrmspk : sk.regs .ptr [1, BK] "rms_w_ptrs" = some rmspT := by simp [hsk, hrmsp, hrmspT]
  -- canonical loaded / derived tiles
  set xT0 : Tile .real [BM, BK] :=
    ⟨fun idx : TileIndex [BM, BK] =>
      some (xElem s0 X M N BM BN sxb sxm sxk idx.1 (t * BK + idx.2.1.val))⟩ with hxT0
  set rmsT : Tile .real [1, BK] :=
    ⟨fun idx : TileIndex [1, BK] =>
      some (rmsElem s0 RMS srms (t * BK + idx.2.1.val))⟩ with hrmsT
  set xT1 : Tile .real [BM, BK] :=
    ⟨fun idx : TileIndex [BM, BK] =>
      some (xElem s0 X M N BM BN sxb sxm sxk idx.1 (t * BK + idx.2.1.val)
        * rmsElem s0 RMS srms (t * BK + idx.2.1.val))⟩ with hxT1
  set wT : Tile .real [BK, BN] :=
    ⟨fun idx : TileIndex [BK, BN] =>
      some (wElem s0 W N BN swk swn idx.2.1 (t * BK + idx.1.val))⟩ with hwT
  set xsumT1 : Tile .real [BM, BK] :=
    ⟨fun idx : TileIndex [BM, BK] =>
      some (xSumPartial s0 X M N BM BN sxb sxm sxk BK idx.1 idx.2.1 (t + 1))⟩ with hxsumT1
  unfold rrmLoopBody
  -- 1: x = tl.load(x_ptrs)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (show evalOp (Op.load .real (.ptr (Op.ref .ptr [BM, BK] "x_ptrs")) MaskOpt.none) sk
          = some xT0 from by
        rw [load_ptr_none_real "x_ptrs" sk xpT hxpk]
        refine congrArg some ?_
        ext idx
        simp only [hxpT, hrmem, Region.cast_id, hxT0, xElem]
        rw [show s0.pids 0 * sxb + rowIndex s0 M N BM BN idx.1 * sxm + idx.2.1.val * sxk
                + t * BK * sxk
              = s0.pids 0 * sxb + rowIndex s0 M N BM BN idx.1 * sxm
                + (t * BK + idx.2.1.val) * sxk from by ring]))]
  -- 2: x_sum += pow(x.to(fp32), 2)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (show evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
            (Op.ref .real [BM, BK] "x_sum")
            (Op.pow Broadcast.scalarR
              (Op.castFloat FloatDType.real FloatDType.real (Op.ref .real [BM, BK] "x"))
              (Op.const 2)))
          (sk.setReg "x" .real [BM, BK] xT0) = some xsumT1 from by
        rw [evalOp_add]
        simp only [evalOp_pow, evalOp_ref, evalOp_const, Option.bind_eq_bind]
        rw [cast_real_real_eval "x" _ xT0 (by simp)]
        rw [show (sk.setReg "x" .real [BM, BK] xT0).regs .real [BM, BK] "x_sum"
              = some (⟨fun idx : TileIndex [BM, BK] =>
                  some (xSumPartial s0 X M N BM BN sxb sxm sxk BK idx.1 idx.2.1 t)⟩ : Tile .real [BM, BK]) from by
            simp [hsk, hxsum]]
        simp only [Option.bind_some]
        refine congrArg some ?_
        ext idx
        simp only [hxsumT1, hxT0, Tile.bop_data, Tile.scalar_data,
          Broadcast.leftIndex_consSame, Broadcast.rightIndex_consSame,
          Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR,
          Broadcast.leftIndex_nil, Broadcast.rightIndex_nil,
          NumericDType.add, WithBot.realAdd, WithBot.realPow_some,
          Option.map₂, Option.bind, Option.map]
        rw [xSumPartial_succ, rpow_two_eq]))]
  -- 3: rms_w = tl.load(rms_w_ptrs)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (show evalOp (Op.load .real (.ptr (Op.ref .ptr [1, BK] "rms_w_ptrs")) MaskOpt.none)
          ((sk.setReg "x" .real [BM, BK] xT0).setReg "x_sum" .real [BM, BK] xsumT1)
          = some rmsT from by
        rw [load_ptr_none_real "rms_w_ptrs" _ rmspT (by simp [hrmspk])]
        refine congrArg some ?_
        ext idx
        simp only [hrmspT, BlockState.setReg_readMem, hrmem, Region.cast_id, hrmsT, rmsElem]
        rw [show idx.2.1.val * srms + t * BK * srms
              = (t * BK + idx.2.1.val) * srms from by ring]))]
  -- 4: x = x * rms_w
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (show evalOp (Op.mul .real (Broadcast.consR (Broadcast.consSame Broadcast.nil))
            (Op.ref .real [BM, BK] "x") (Op.ref .real [1, BK] "rms_w")) _
          = some xT1 from by
        rw [evalOp_mul]
        simp only [evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
          String.reduceEq, not_false_eq_true, Option.bind_eq_bind, Option.bind_some]
        refine congrArg some ?_
        ext idx
        simp only [hxT0, hrmsT, hxT1, Tile.bop_data,
          Broadcast.leftIndex_consR, Broadcast.leftIndex_consSame, Broadcast.leftIndex_nil,
          Broadcast.rightIndex_consR, Broadcast.rightIndex_consSame, Broadcast.rightIndex_nil,
          NumericDType.mul, WithBot.realMul, Option.map₂, Option.bind, Option.map]))]
  -- 5: w = tl.load(w_ptrs)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (show evalOp (Op.load .real (.ptr (Op.ref .ptr [BK, BN] "w_ptrs")) MaskOpt.none) _
          = some wT from by
        rw [load_ptr_none_real "w_ptrs" _ wpT (by simp [hwpk])]
        refine congrArg some ?_
        ext idx
        simp only [hwpT, BlockState.setReg_readMem, hrmem, Region.cast_id, hwT, wElem]
        rw [show idx.1.val * swk + colIndex s0 N BN idx.2.1 * swn + t * BK * swk
              = (t * BK + idx.1.val) * swk + colIndex s0 N BN idx.2.1 * swn from by ring]))]
  -- 6: accumulator += tl.dot(x, w)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (show evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
            (Op.ref .real [BM, BN] "accumulator")
            (Op.dot (batch := []) (Op.ref .real [BM, BK] "x") (Op.ref .real [BK, BN] "w"))) _
          = some (⟨fun idx : TileIndex [BM, BN] =>
              some (accPartial s0 X W RMS M N BM BN sxb sxm sxk swk swn srms BK idx.1 idx.2.1 (t + 1))⟩
              : Tile .real [BM, BN]) from by
        rw [accdot_eval BM BK BN _ "accumulator" "w" accT xT1 wT
            (by simp [hsk, hacc, haccT]) (by simp) (by simp)]
        refine congrArg some ?_
        ext idx
        rw [adddot_lane BM BN accT (Tile.dot [] xT1 wT) idx.1 idx.2.1
            (accPartial s0 X W RMS M N BM BN sxb sxm sxk swk swn srms BK idx.1 idx.2.1 t)
            (Finset.univ.sum fun e : Fin BK =>
              (xElem s0 X M N BM BN sxb sxm sxk idx.1 (t * BK + e.val)
                * rmsElem s0 RMS srms (t * BK + e.val))
                * wElem s0 W N BN swk swn idx.2.1 (t * BK + e.val))
            (by rw [haccT])
            (tile_dot_data BM BK BN xT1 wT idx.1 idx.2.1 _ _
              (fun e => by rw [hxT1]) (fun e => by rw [hwT]))]
        show _ = some (accPartial s0 X W RMS M N BM BN sxb sxm sxk swk swn srms BK idx.1 idx.2.1 (t + 1))
        rw [accPartial_succ]))]
  -- 7–9: the three pointer advances
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (ptr_adv_eval _ "x_ptrs" BK sxk xpT (by simp [hxpk])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (ptr_adv_eval _ "w_ptrs" BK swk wpT (by simp [hwpk])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (ptr_adv_eval _ "rms_w_ptrs" BK srms rmspT (by simp [hrmspk])))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  simp only [rrmInvariant]
  refine ⟨by simp [hsk, hpids], by omega, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp [hsk, hpb]
  · simp [hsk, hpm]
  · simp [hsk, hpn]
  · simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
  · simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    rw [hxsumT1]
  · simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx <;> simp only [Tile.ptrAdd, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
      Tile.scalar, hxpT, NumericDType.add]
    ring
  · simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx <;> simp only [Tile.ptrAdd, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
      Tile.scalar, hwpT, NumericDType.add]
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

/-! ## Post-loop: RMS tail + fp16 cast + masked store -/

/-- The rank-2 `tl.sum(_, axis=1)` lane collapse: row `i` of the dropped-axis
sum of an all-`some` tile is `some (Σ_e f i e)`. -/
theorem reduceSumDrop_row_lane (BM BK : Nat) (f : Fin BM → Fin BK → ℝ) (idx : TileIndex [BM]) :
    (Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [BM, BK].length)
        (⟨fun idx : TileIndex [BM, BK] => some (f idx.1 idx.2.1)⟩ : Tile .real [BM, BK])).data idx
      = some (∑ e : Fin BK, f idx.1 e) := by
  show (Finset.univ.sum fun e : Fin BK => ((f idx.1 e : ℝ) : WithBot ℝ)) = _
  rw [← WithBot.coe_sum]
  rfl

/-- `x_mean = tl.sum(x_sum, axis=1) / K + EPS` eval (result shape pinned to the
literal `[BM]` so the assign statement matches the body decomposition). -/
theorem xmean_eval (BM BK K : Nat) (EPS : ℝ) (st : BlockState) (f : Fin BM → Fin BK → ℝ)
    (hsum : st.regs .real [BM, BK] "x_sum"
      = some ⟨fun idx : TileIndex [BM, BK] => some (f idx.1 idx.2.1)⟩) :
    @evalOp .real [BM] (Op.add .real Broadcast.scalarR
        (Op.div .real Broadcast.scalarR
          (Op.reduceSum (⟨1, by simp⟩ : Fin [BM, BK].length) Bool.false
            (Op.ref .real [BM, BK] "x_sum"))
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

/-- `x_norm = tl.math.rsqrt(x_mean)` eval. -/
theorem xnorm_eval (BM : Nat) (st : BlockState) (g : Fin BM → ℝ)
    (h : st.regs .real [BM] "x_mean" = some ⟨fun idx : TileIndex [BM] => some (g idx.1)⟩) :
    evalOp (Op.rsqrt (Op.ref .real [BM] "x_mean")) st
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

/-- `accumulator = accumulator * x_norm[:, None]` eval. -/
theorem accscale_eval (BM BN : Nat) (st : BlockState) (name : RegName)
    (f : TileIndex [BM, BN] → ℝ) (g : Fin BM → ℝ)
    (hacc : st.regs .real [BM, BN] name = some ⟨fun idx => some (f idx)⟩)
    (hn : st.regs .real [BM] "x_norm" = some ⟨fun idx : TileIndex [BM] => some (g idx.1)⟩) :
    evalOp (Op.mul .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [BM, BN] name)
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BM] "x_norm"))) st
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

/-! ### Masked fp16 store-foldl state projections

The store `foldl` writes only `(OUT, offset)` cells; program ids, `undef`,
and every other region's memory pass through untouched. These three walks
feed the strengthened postLoop conjuncts that the qkv cross-pass frame
transport consumes. -/

theorem foldl_maskstore_pids {ι : Type} (region : RegionName)
    (ofn : ι → Nat) (vfn : ι → TileCarrier TileDType.fp16) (m : ι → Bool) (l : List ι) :
    ∀ s : BlockState,
      (l.foldl (fun acc i =>
        if m i then acc.writeMemTyped .fp16 region (ofn i) (vfn i) else acc) s).pids
        = s.pids := by
  induction l with
  | nil => intro s; rfl
  | cons hd tl ih =>
      intro s
      rw [List.foldl_cons, ih]
      by_cases h : m hd = Bool.true
      · rw [if_pos h, BlockState.writeMemTyped_pids]
      · rw [if_neg h]

theorem foldl_maskstore_undef {ι : Type} (region : RegionName)
    (ofn : ι → Nat) (vfn : ι → TileCarrier TileDType.fp16) (m : ι → Bool) (l : List ι)
    (rg : RegionName) (o : Nat) :
    ∀ s : BlockState,
      (l.foldl (fun acc i =>
        if m i then acc.writeMemTyped .fp16 region (ofn i) (vfn i) else acc) s).undef rg o
        = s.undef rg o := by
  induction l with
  | nil => intro s; rfl
  | cons hd tl ih =>
      intro s
      rw [List.foldl_cons, ih]
      by_cases h : m hd = Bool.true
      · rw [if_pos h, BlockState.writeMemTyped_undef]
      · rw [if_neg h]

theorem foldl_maskstore_mem_ne {ι : Type} {region R : RegionName} (hne : R ≠ region)
    (ofn : ι → Nat) (vfn : ι → TileCarrier TileDType.fp16) (m : ι → Bool) (l : List ι)
    (o : Nat) :
    ∀ s : BlockState,
      (l.foldl (fun acc i =>
        if m i then acc.writeMemTyped .fp16 region (ofn i) (vfn i) else acc) s).mem R o
        = s.mem R o := by
  induction l with
  | nil => intro s; rfl
  | cons hd tl ih =>
      intro s
      rw [List.foldl_cons, ih]
      by_cases h : m hd = Bool.true
      · rw [if_pos h]
        unfold BlockState.writeMemTyped BlockState.writeMemAs
        change (if R = region ∧ o = ofn hd then _ else s.mem R o) = s.mem R o
        rw [if_neg (fun hc => hne hc.1)]
      · rw [if_neg h]

set_option maxHeartbeats 4000000 in
/-- **postLoop**: from the invariant at `numKBlocks` blocks, the RMS tail +
fp16 cast + masked store writes the genuine closed form `fp16( n · S )` at
every active output lane (given the output-offset map is injective); program
ids and `undef` pass through and every region other than `OUT` is untouched
(the frame the qkv cross-pass transport consumes). -/
theorem rrm_postLoop (X W RMS OUT : RegionName) (s0 : BlockState)
    (M N BM BN BK sxb sxm sxk swk swn srms sob som son numKBlocks K : Nat) (EPS : ℝ)
    (hInj : Function.Injective (outOffset s0 N BM BN sob som son))
    (st : BlockState)
    (hinv : rrmInvariant X W RMS s0 M N BM BN BK sxb sxm sxk swk swn srms
      numKBlocks numKBlocks st) :
    ∃ sfin, stepStmts (rrmStoreTail OUT M N BM BN BK K sob som son EPS) st = some sfin
      ∧ sfin.pids = st.pids
      ∧ (∀ rg o, sfin.undef rg o = st.undef rg o)
      ∧ (∀ R o, R ≠ OUT → sfin.mem R o = st.mem R o)
      ∧ ∀ idx : TileIndex [BM, BN],
          sfin.mem OUT (outOffset s0 N BM BN sob som son idx)
            = if active s0 M N BM BN idx then
                MemCell.of .fp16
                  (FloatDType.real.cast FloatDType.fp16
                    (some (rmsSpec s0 X W RMS M N BM BN sxb sxm sxk swk swn srms
                      BK numKBlocks K EPS idx.1 idx.2.1)))
              else
                st.mem OUT (outOffset s0 N BM BN sob som son idx) := by
  simp only [rrmInvariant] at hinv
  obtain ⟨hpids, hcle, hpb, hpm, hpn, hacc, hxsum, hxp, hwp, hrmsp, hundef, hmem⟩ := hinv
  set nrm : Fin BM → ℝ :=
    fun i => normSpec s0 X M N BM BN sxb sxm sxk BK numKBlocks K EPS i with hnrm
  set sF : TileIndex [BM, BN] → ℝ :=
    fun idx => accPartial s0 X W RMS M N BM BN sxb sxm sxk swk swn srms BK idx.1 idx.2.1 numKBlocks
    with hsF
  set accT : Tile .real [BM, BN] :=
    ⟨fun idx : TileIndex [BM, BN] =>
      some (rmsSpec s0 X W RMS M N BM BN sxb sxm sxk swk swn srms
        BK numKBlocks K EPS idx.1 idx.2.1)⟩ with haccT
  set cT : Tile .fp16 [BM, BN] :=
    ⟨fun idx => FloatDType.real.cast FloatDType.fp16 (accT.data idx)⟩ with hcT
  set cpT : Tile .ptr [BM, BN] :=
    ⟨fun idx : TileIndex [BM, BN] => (OUT.cast, outOffset s0 N BM BN sob som son idx)⟩ with hcpT
  set cmaskT : Tile .bool [BM, BN] :=
    ⟨fun idx : TileIndex [BM, BN] =>
      (decide (rowGlobal s0 N BM BN idx.1 < M) && decide (colGlobal s0 N BN idx.2.1 < N))⟩
    with hcmaskT
  unfold rrmStoreTail
  -- x_mean
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (xmean_eval BM BK K EPS st
        (fun i e => xSumPartial s0 X M N BM BN sxb sxm sxk BK i e numKBlocks) (by rw [hxsum])))]
  -- x_norm (canonicalized to `normSpec` via `Finset.sum_comm`)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (show evalOp (Op.rsqrt (Op.ref .real [BM] "x_mean")) _
          = some (⟨fun idx : TileIndex [BM] => some (nrm idx.1)⟩ : Tile .real [BM]) from by
        rw [xnorm_eval BM _
          (fun i => (∑ e : Fin BK, xSumPartial s0 X M N BM BN sxb sxm sxk BK i e numKBlocks)
            / (K : ℝ) + EPS) (by simp)]
        refine congrArg some ?_
        ext idx
        simp only [hnrm, normSpec, xSumPartial]
        rw [Finset.sum_comm]))]
  -- accumulator = accumulator * x_norm[:, None]  (canonicalized to `rmsSpec`)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (show evalOp (Op.mul .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
            (Op.ref .real [BM, BN] "accumulator")
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BM] "x_norm"))) _
          = some accT from by
        rw [accscale_eval BM BN _ "accumulator" sF nrm (by simp [hacc, hsF]) (by simp)]
        refine congrArg some ?_
        ext idx
        simp only [haccT, rmsSpec, hsF, hnrm]
        exact congrArg some (mul_comm _ _)))]
  -- offs_m / offs_n (tail re-derivation, un-wrapped)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (offsout_eval _ BM BM (pidM (s0.pids 1) N BN) "pid_m" (by simp [hpm])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (offsout_eval _ BN BN (pidN (s0.pids 1) N BN) "pid_n" (by simp [hpn])))]
  -- out_ptrs (with the batch offset)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (show evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase OUT)
          (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
            (Op.add .nat Broadcast.scalarL
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_batch") (Op.constNat sob))
              (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m")) (Op.constNat som)))
            (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_n")) (Op.constNat son)))) _
          = some cpT from by
        rw [batchptrs_eval _ OUT BM BN sob som son (s0.pids 0) "offs_m" "offs_n"
          (fun i => rowGlobal s0 N BM BN i) (fun j => colGlobal s0 N BN j)
          (by simp [hpb]) (by simp [rowGlobal]) (by simp [colGlobal])]
        rfl))]
  -- out_mask
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (show evalOp (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m")) (Op.constNat M))
          (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_n")) (Op.constNat N))) _
          = some cmaskT from by
        rw [outmask_eval _ M N BM BN
          (fun i => rowGlobal s0 N BM BN i) (fun j => colGlobal s0 N BN j)
          (by simp [rowGlobal]) (by simp [colGlobal])]))]
  -- abstract the post-assign state
  generalize hst8 : (((((((st.setReg "x_mean" .real [BM]
      (⟨fun idx : TileIndex [BM] =>
        some ((∑ e : Fin BK, xSumPartial s0 X M N BM BN sxb sxm sxk BK idx.1 e numKBlocks)
          / (K : ℝ) + EPS)⟩ : Tile .real [BM])).setReg "x_norm" .real [BM]
      (⟨fun idx : TileIndex [BM] => some (nrm idx.1)⟩ : Tile .real [BM])).setReg
      "accumulator" .real [BM, BN] accT).setReg "offs_m" .nat [BM]
      (Tile.vec fun i : Fin BM => pidM (s0.pids 1) N BN * BM + i.val)).setReg "offs_n" .nat [BN]
      (Tile.vec fun j : Fin BN => pidN (s0.pids 1) N BN * BN + j.val)).setReg
      "out_ptrs" .ptr [BM, BN] cpT).setReg "out_mask" .bool [BM, BN] cmaskT) = st8
  have hacc8 : st8.regs .real [BM, BN] "accumulator" = some accT := by rw [← hst8]; simp
  have hcp8 : st8.regs .ptr [BM, BN] "out_ptrs" = some cpT := by rw [← hst8]; simp
  have hcm8 : st8.regs .bool [BM, BN] "out_mask" = some cmaskT := by rw [← hst8]; simp
  have hmem8 : st8.mem = st.mem := by
    rw [← hst8]; funext region offset; simp only [BlockState.setReg_mem]
  -- the masked store
  have hstore : stepStmt (Stmt.store .fp16 [BM, BN] (.ptr (Op.ref .ptr [BM, BN] "out_ptrs"))
        (Op.castFloat FloatDType.real FloatDType.fp16 (Op.ref .real [BM, BN] "accumulator"))
        (.mask (Op.ref .bool [BM, BN] "out_mask"))) st8
      = some ((TileShape.allIndices [BM, BN]).foldl
          (fun acc i =>
            if cmaskT.data i then
              acc.writeMemTyped .fp16 OUT (outOffset s0 N BM BN sob som son i) (cT.data i)
            else acc) st8) := by
    simp only [stepStmt]
    erw [show evalOp (Op.castFloat FloatDType.real FloatDType.fp16 (Op.ref .real [BM, BN] "accumulator")) st8
          = some cT from by erw [evalOp_castFloat, evalOp_ref, hacc8]; rw [hcT]; rfl]
    rw [show evalOp (Op.ref .ptr [BM, BN] "out_ptrs") st8 = some cpT from by rw [evalOp_ref, hcp8]]
    rw [show evalOp (Op.ref .bool [BM, BN] "out_mask") st8 = some cmaskT from by rw [evalOp_ref, hcm8]]
    simp only [bind, Option.bind_some]
    refine congrArg some (List.foldl_ext _ _ _ (fun acc i _ => ?_))
    by_cases hmask : cmaskT.data i
    · simp only [hmask, if_true, cpT, hcpT, Region.cast_id]
    · simp only [hmask, Bool.false_eq_true, if_false]
  rw [stepStmts.cons_some hstore, stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_⟩
  · -- pids
    refine (foldl_maskstore_pids OUT (fun i => outOffset s0 N BM BN sob som son i)
        (fun i => cT.data i) (fun i => cmaskT.data i) _ st8).trans ?_
    rw [← hst8]
    rfl
  · -- undef
    intro rg o
    refine (foldl_maskstore_undef OUT (fun i => outOffset s0 N BM BN sob som son i)
        (fun i => cT.data i) (fun i => cmaskT.data i) _ rg o st8).trans ?_
    rw [← hst8]
    rfl
  · -- frame: regions other than OUT untouched
    intro R o hne
    refine (foldl_maskstore_mem_ne hne (fun i => outOffset s0 N BM BN sob som son i)
        (fun i => cT.data i) (fun i => cmaskT.data i) _ o st8).trans ?_
    rw [hmem8]
  · -- the masked store characterization
    intro idx
    rw [scatter_memcell_fp16_prop_masked_nd (region := OUT) (s := st8)
          (offsetFn := outOffset s0 N BM BN sob som son)
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
      rw [if_neg hmaskfalse, if_neg hact, hmem8]

/-! ## One generic inlined pass -/

set_option maxHeartbeats 4000000 in
set_option linter.unusedVariables false in
/-- **One full pass** (prologue + K-loop + RMS tail + masked fp16 store) from
an arbitrary clean entry state: composes `preLoop` + `rrm_step` (driven by
`forRange_inv`) + `rrm_postLoop`. Produces the final state together with the
four facts the sequential three-pass composition needs: `pids` preserved,
`undef` still clean, every region other than `OUT` untouched, and the active
output lanes holding the genuine closed form `fp16( n · S )`. (`hK` is the
presentation hypothesis that makes the spec's `/ K` mean the full-loop
moment — the loads are unmasked, so it is carried as an honest disclosure
rather than consumed by a mask argument.) -/
theorem rrm_pass (X W RMS OUT : RegionName) (s : BlockState)
    (M N sxb sxm sxk swk swn srms sob som son BM BN BK numKBlocks : Nat) (K : Nat)
    (hK : K = BK * numKBlocks) (EPS : ℝ)
    (hInj : Function.Injective (outOffset s N BM BN sob som son))
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ∃ s', stepStmts (rrmPassBody X W RMS OUT M N K sxb sxm sxk swk swn srms sob som son
        BM BN BK numKBlocks EPS) s = some s'
      ∧ s'.pids = s.pids
      ∧ (∀ rg o, s'.undef rg o = 0)
      ∧ (∀ R o, R ≠ OUT → s'.mem R o = s.mem R o)
      ∧ ∀ idx : TileIndex [BM, BN],
          s'.mem OUT (outOffset s N BM BN sob som son idx)
            = if active s M N BM BN idx then
                MemCell.of .fp16
                  (FloatDType.real.cast FloatDType.fp16
                    (some (rmsSpec s X W RMS M N BM BN sxb sxm sxk swk swn srms
                      BK numKBlocks K EPS idx.1 idx.2.1)))
              else
                s.mem OUT (outOffset s N BM BN sob som son idx) := by
  obtain ⟨s0, hpre_eq, hP0⟩ := preLoop X W RMS s M N sxb sxm sxk swk swn srms
    BM BN BK numKBlocks hundef
  obtain ⟨final, sLoop, hLoopStmt, hfinal, hPLoop⟩ :=
    forRange_inv (idx := "_i") (start := 0) (stop := numKBlocks) (step := 1)
      (by omega) hP0
      (fun t st hlt hinv => by
        have := rrm_step X W RMS s M N BM BN BK sxb sxm sxk swk swn srms
          numKBlocks t st hlt hinv
        simpa using this)
  have hfinalEq : final = numKBlocks := by
    have hle : final ≤ numKBlocks := by
      simp only [rrmInvariant] at hPLoop
      exact hPLoop.2.1
    exact le_antisymm hle hfinal
  rw [hfinalEq] at hPLoop
  have hPLoop' := hPLoop
  simp only [rrmInvariant] at hPLoop'
  obtain ⟨hpidsL, _hcle, _hpb, _hpm, _hpn, _hacc, _hxsum, _hxp, _hwp, _hrmsp,
    hundefL, hmemL⟩ := hPLoop'
  obtain ⟨sfin, hTail, hpidsT, hundefT, hframeT, hpost⟩ :=
    rrm_postLoop X W RMS OUT s M N BM BN BK sxb sxm sxk swk swn srms
      sob som son numKBlocks K EPS hInj sLoop hPLoop
  refine ⟨sfin, ?_, ?_, ?_, ?_, ?_⟩
  · show stepStmts (rrmPrologue X W RMS M N BM BN BK sxb sxm sxk swk swn srms
        ++ (Stmt.forRange "_i" 0 numKBlocks 1 (rrmLoopBody BM BN BK sxk swk srms)
            :: rrmStoreTail OUT M N BM BN BK K sob som son EPS)) s = some sfin
    rw [stepStmts.append_some hpre_eq, stepStmts.cons_some hLoopStmt, hTail]
  · rw [hpidsT, hpidsL]
  · intro rg o
    rw [hundefT rg o]
    exact hundefL rg o
  · intro R o hne
    rw [hframeT R o hne, hmemL]
  · intro idx
    have h := hpost idx
    rw [hmemL] at h
    exact h

/-! ## Callee headline -/

/-- **Closed-form correctness for `rms_matmul_rbe` (general statement).**

For arbitrary batch id (`s.pids 0`), linear tile pid (`s.pids 1`), tile dims
`BM`/`BN`, K-block size `BK`, and K-block count `numKBlocks` (so the
contracted dimension is `K = BK · numKBlocks`; the kernel's loads are
unmasked, so this exact-multiple presentation is required), every **active**
output cell `(i, j)` of the computed `BM × BN` tile equals

  `fp16( n_i · S[i,j] )`

over ℝ — where `S[i,j] = Σ_{t<numKBlocks} Σ_{e<BK} X[b, r(i), t·BK+e] ·
RMS[t·BK+e] · W[t·BK+e, c(j)]`,
`n_i = rsqrt((Σ_{t,e} X[b, r(i), t·BK+e]²)/K + EPS)`,
`r(i) = (pid_m·BM + i) % M`, `c(j) = (pid_n·BN + j) % N`, and every X read
carries the batch offset `b·sxb` (b = `pid_batch = s.pids 0`) — the genuine
batched RMSNorm-fused GEMM closed form of the loaded `X`/`RMS`/`W` cells, NOT
the kernel's own executed value; inactive lanes are left untouched.

Layout: `X[b,i,k]` at `X + b·sxb + offs_m(i)·sxm + k·sxk`, `W[k,j]` at
`W + k·swk + offs_n(j)·swn`, `RMS[k]` at `RMS + k·srms`, `OUT[b,i,j]` at
`OUT + b·sob + offs_m(i)·som + offs_n(j)·son`, with `pid_m = pid //
cdiv(N,BN)`, `pid_n = pid % cdiv(N,BN)` over `pid = s.pids 1`. Preconditions:
the row-major output bounds `son = 1` / `BN ≤ som` (give output-offset
injectivity; the batch term is a per-program constant) and clean initial
`undef`. -/
specification rms_matmul_rbe_closed_form_correct
    (X W RMS OUT : RegionName) (s : BlockState)
    (M N sxb sxm sxk swk swn srms sob som son BM BN BK numKBlocks : Nat) (K : Nat)
    (hK : K = BK * numKBlocks) (EPS : ℝ)
    (hcn : son = 1) (hbnle : BN ≤ som)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := rms_matmul_rbe_surface X W RMS OUT M N K sxb sxm sxk swk swn srms
        sob som son BM BN BK numKBlocks EPS)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s M N BM BN)
        (fun idx => (OUT, outOffset s N BM BN sob som son idx)))
      (expected := fun idx : TileIndex [BM, BN] =>
        MemCell.of .fp16
          (FloatDType.real.cast FloatDType.fp16
            (some (rmsSpec s X W RMS M N BM BN sxb sxm sxk swk swn srms
              BK numKBlocks K EPS idx.1 idx.2.1)))) := by
  subst hcn
  have hInj : Function.Injective (outOffset s N BM BN sob som 1) := by
    have heq : outOffset s N BM BN sob som 1
        = fun idx : TileIndex [BM, BN] =>
            (s.pids 0 * sob + som * (pidM (s.pids 1) N BN * BM) + pidN (s.pids 1) N BN * BN)
              + idx.1.val * som + idx.2.1.val := by
      funext idx; simp only [outOffset, rowGlobal, colGlobal]; ring
    rw [heq]; exact rowMajor2D_inj _ som hbnle
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [rms_matmul_rbe_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst hs0
  intro idx hActive
  obtain ⟨sP, hP, _hpids, _hund, _hfr, hstore⟩ := rrm_pass X W RMS OUT s0 M N sxb sxm sxk
    swk swn srms sob som 1 BM BN BK numKBlocks K hK EPS hInj hundef
  have hExec2 : exec (rms_matmul_rbe_surface X W RMS OUT M N K sxb sxm sxk swk swn srms
      sob som 1 BM BN BK numKBlocks EPS) s0 = some s' := hExec
  have hexecP : exec (rms_matmul_rbe_surface X W RMS OUT M N K sxb sxm sxk swk swn srms
      sob som 1 BM BN BK numKBlocks EPS) s0 = some sP := by
    rw [exec, rrm_body_split X W RMS OUT M N K sxb sxm sxk swk swn srms sob som 1
      BM BN BK numKBlocks EPS]
    exact hP
  have hs' : s' = sP := by
    rw [hExec2] at hexecP
    exact Option.some.inj hexecP
  subst hs'
  have hmain := hstore idx
  rw [if_pos hActive] at hmain
  simpa only [ComputeCorrect.OutputReadable.read_memcell] using hmain

/-! ## QKV: three-pass exec closed form -/

set_option maxHeartbeats 4000000 in
set_option linter.unusedVariables false in
/-- **Three-pass exec reduction for `rms_matmul_rbe_qkv`**: runs `rrm_pass`
at `(QW, Q)`, `(KW, KOut)`, `(VW, V)` in sequence. Pass 2/3 recompute every
register from `tl.program_id`, so their entry facts are only `pids` (equal to
the initial `pids`), clean `undef`, and the untouched read regions; their
store characterizations — expressed at their own entry states — transport
back to the initial state `s` via the congruence lemmas and the pairwise
region distinctness. Every active lane of each output holds the fp16-cast of
`n_i · S_w[i,j]` with the pass's own weight matrix and the shared RMS
normalizer. -/
theorem qkv_exec_closed_form (X QW KW VW RMS Q KOut V : RegionName) (s : BlockState)
    (M N sxb sxm sxk sqwk sqwn skwk skwn svwk svwn srms
      sqb sqm sqn skb skm skn svb svm svn BM BN BK numKBlocks : Nat) (K : Nat)
    (hK : K = BK * numKBlocks) (EPS : ℝ)
    (hInjQ : Function.Injective (outOffset s N BM BN sqb sqm sqn))
    (hInjK : Function.Injective (outOffset s N BM BN skb skm skn))
    (hInjV : Function.Injective (outOffset s N BM BN svb svm svn))
    (hXQ : X ≠ Q) (hRMSQ : RMS ≠ Q) (hKWQ : KW ≠ Q) (hVWQ : VW ≠ Q)
    (hXK : X ≠ KOut) (hRMSK : RMS ≠ KOut) (hVWK : VW ≠ KOut)
    (hQK : Q ≠ KOut) (hQV : Q ≠ V) (hKV : KOut ≠ V)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    (match exec (rms_matmul_rbe_qkv_surface X QW KW VW RMS Q KOut V M N K sxb sxm sxk
        sqwk sqwn skwk skwn svwk svwn srms sqb sqm sqn skb skm skn svb svm svn
        BM BN BK numKBlocks EPS) s with
      | some sF =>
          (∀ idx : TileIndex [BM, BN], active s M N BM BN idx →
              sF.mem Q (outOffset s N BM BN sqb sqm sqn idx)
                = MemCell.of .fp16
                    (FloatDType.real.cast FloatDType.fp16
                      (some (rmsSpec s X QW RMS M N BM BN sxb sxm sxk sqwk sqwn srms
                        BK numKBlocks K EPS idx.1 idx.2.1))))
          ∧ (∀ idx : TileIndex [BM, BN], active s M N BM BN idx →
              sF.mem KOut (outOffset s N BM BN skb skm skn idx)
                = MemCell.of .fp16
                    (FloatDType.real.cast FloatDType.fp16
                      (some (rmsSpec s X KW RMS M N BM BN sxb sxm sxk skwk skwn srms
                        BK numKBlocks K EPS idx.1 idx.2.1))))
          ∧ (∀ idx : TileIndex [BM, BN], active s M N BM BN idx →
              sF.mem V (outOffset s N BM BN svb svm svn idx)
                = MemCell.of .fp16
                    (FloatDType.real.cast FloatDType.fp16
                      (some (rmsSpec s X VW RMS M N BM BN sxb sxm sxk svwk svwn srms
                        BK numKBlocks K EPS idx.1 idx.2.1))))
      | none => False) := by
  -- pass 1: (QW, Q)
  obtain ⟨s1, h1, hpids1, hundef1, hframe1, hstore1⟩ :=
    rrm_pass X QW RMS Q s M N sxb sxm sxk sqwk sqwn srms sqb sqm sqn
      BM BN BK numKBlocks K hK EPS hInjQ hundef
  -- pass 2: (KW, KOut), from s1
  have hInjK1 : Function.Injective (outOffset s1 N BM BN skb skm skn) := by
    have hfe : outOffset s1 N BM BN skb skm skn = outOffset s N BM BN skb skm skn :=
      funext fun idx => outOffset_congr hpids1 idx
    rw [hfe]; exact hInjK
  obtain ⟨s2, h2, hpids2, hundef2, hframe2, hstore2⟩ :=
    rrm_pass X KW RMS KOut s1 M N sxb sxm sxk skwk skwn srms skb skm skn
      BM BN BK numKBlocks K hK EPS hInjK1 hundef1
  have hpids2' : s2.pids = s.pids := hpids2.trans hpids1
  -- pass 3: (VW, V), from s2
  have hInjV2 : Function.Injective (outOffset s2 N BM BN svb svm svn) := by
    have hfe : outOffset s2 N BM BN svb svm svn = outOffset s N BM BN svb svm svn :=
      funext fun idx => outOffset_congr hpids2' idx
    rw [hfe]; exact hInjV
  obtain ⟨s3, h3, hpids3, hundef3, hframe3, hstore3⟩ :=
    rrm_pass X VW RMS V s2 M N sxb sxm sxk svwk svwn srms svb svm svn
      BM BN BK numKBlocks K hK EPS hInjV2 hundef2
  -- the full three-pass execution
  have hexec : exec (rms_matmul_rbe_qkv_surface X QW KW VW RMS Q KOut V M N K sxb sxm sxk
      sqwk sqwn skwk skwn svwk svwn srms sqb sqm sqn skb skm skn svb svm svn
      BM BN BK numKBlocks EPS) s = some s3 := by
    rw [exec, qkv_body_split X QW KW VW RMS Q KOut V M N K sxb sxm sxk
        sqwk sqwn skwk skwn svwk svwn srms sqb sqm sqn skb skm skn svb svm svn
        BM BN BK numKBlocks EPS,
      stepStmts.append_some h1, stepStmts.append_some h2, h3]
  rw [hexec]
  refine ⟨?_, ?_, ?_⟩
  · -- Q output: pass 1's store, framed through passes 2–3
    intro idx hact
    rw [hframe3 Q _ hQV, hframe2 Q _ hQK]
    have h := hstore1 idx
    rw [if_pos hact] at h
    exact h
  · -- K output: pass 2's store (transported from s1 to s), framed through pass 3
    intro idx hact
    rw [hframe3 KOut _ hKV]
    have h := hstore2 idx
    rw [outOffset_congr hpids1 idx] at h
    have hact1 : active s1 M N BM BN idx := (active_congr hpids1 idx).mpr hact
    rw [if_pos hact1] at h
    rw [rmsSpec_congr hpids1 (fun o => hframe1 X o hXQ) (fun o => hframe1 KW o hKWQ)
      (fun o => hframe1 RMS o hRMSQ)] at h
    exact h
  · -- V output: pass 3's store (transported from s2 to s)
    intro idx hact
    have h := hstore3 idx
    rw [outOffset_congr hpids2' idx] at h
    have hact2 : active s2 M N BM BN idx := (active_congr hpids2' idx).mpr hact
    rw [if_pos hact2] at h
    have hX2 : ∀ o, s2.mem X o = s.mem X o :=
      fun o => (hframe2 X o hXK).trans (hframe1 X o hXQ)
    have hVW2 : ∀ o, s2.mem VW o = s.mem VW o :=
      fun o => (hframe2 VW o hVWK).trans (hframe1 VW o hVWQ)
    have hRMS2 : ∀ o, s2.mem RMS o = s.mem RMS o :=
      fun o => (hframe2 RMS o hRMSK).trans (hframe1 RMS o hRMSQ)
    rw [rmsSpec_congr hpids2' hX2 hVW2 hRMS2] at h
    exact h

/-! ## QKV headline -/

set_option maxHeartbeats 4000000 in
/-- **Closed-form correctness for `rms_matmul_rbe_qkv` (general statement).**

The qkv kernel is the callee `rms_matmul_rbe` inlined three times — at
`(q_weight, q)`, `(k_weight, k)`, `(v_weight, v)` — all sharing `X`, `RMS`,
and the launch geometry. The conclusion is the standard multi-output bundle:
a conjunction of three `Realizes_without_Rounding`, one per stored output.
For arbitrary batch id (`s.pids 0`) and linear tile pid (`s.pids 1`), every
**active** output cell `(i, j)` of the computed `BM × BN` tile of each
output equals

  `fp16( n_i · S_w[i,j] )`

over ℝ — where `S_w[i,j] = Σ_{t<numKBlocks} Σ_{e<BK} X[b, r(i), t·BK+e] ·
RMS[t·BK+e] · W[t·BK+e, c(j)]` against the pass's OWN weight matrix
(`W = QW`/`KW`/`VW` with its own strides) and
`n_i = rsqrt((Σ_{t,e} X[b, r(i), t·BK+e]²)/K + EPS)` is the SHARED RMS
normalizer; `r(i) = (pid_m·BM + i) % M`, `c(j) = (pid_n·BN + j) % N`, batch
offset `b·sxb` in every X read — the genuine closed form of the loaded
`X`/`RMS`/weight cells, NOT the kernel's own executed value; inactive lanes
impose no obligation.

Preconditions: `K = BK · numKBlocks` (unmasked loads); per-output row-major
bounds `sqn = 1`/`BN ≤ sqm`, `skn = 1`/`BN ≤ skm`, `svn = 1`/`BN ≤ svm`
(output-offset injectivity); pairwise distinctness of the written regions
`Q`/`KOut`/`V` from each other and from the read regions each later pass
consumes (`X`, `RMS`, and the pass's weight region — the host passes eight
distinct buffers), which is exactly what the cross-pass frame transport
needs; and clean initial `undef`. -/
specification rms_matmul_rbe_qkv_closed_form_correct
    (X QW KW VW RMS Q KOut V : RegionName) (s : BlockState)
    (M N sxb sxm sxk sqwk sqwn skwk skwn svwk svwn srms
      sqb sqm sqn skb skm skn svb svm svn BM BN BK numKBlocks : Nat) (K : Nat)
    (hK : K = BK * numKBlocks) (EPS : ℝ)
    (hqn : sqn = 1) (hqm : BN ≤ sqm)
    (hkn : skn = 1) (hkm : BN ≤ skm)
    (hvn : svn = 1) (hvm : BN ≤ svm)
    (hXQ : X ≠ Q) (hRMSQ : RMS ≠ Q) (hKWQ : KW ≠ Q) (hVWQ : VW ≠ Q)
    (hXK : X ≠ KOut) (hRMSK : RMS ≠ KOut) (hVWK : VW ≠ KOut)
    (hQK : Q ≠ KOut) (hQV : Q ≠ V) (hKV : KOut ≠ V)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := rms_matmul_rbe_qkv_surface X QW KW VW RMS Q KOut V M N K sxb sxm sxk
        sqwk sqwn skwk skwn svwk svwn srms sqb sqm sqn skb skm skn svb svm svn
        BM BN BK numKBlocks EPS)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s M N BM BN)
        (fun idx => (Q, outOffset s N BM BN sqb sqm sqn idx)))
      (expected := fun idx : TileIndex [BM, BN] =>
        MemCell.of .fp16
          (FloatDType.real.cast FloatDType.fp16
            (some (rmsSpec s X QW RMS M N BM BN sxb sxm sxk sqwk sqwn srms
              BK numKBlocks K EPS idx.1 idx.2.1))))
    ∧ ComputeCorrect.Realizes_without_Rounding
        (kernel := rms_matmul_rbe_qkv_surface X QW KW VW RMS Q KOut V M N K sxb sxm sxk
          sqwk sqwn skwk skwn svwk svwn srms sqb sqm sqn skb skm skn svb svm svn
          BM BN BK numKBlocks EPS)
        (initialState := s)
        (write := ComputeCorrect.WriteMap.writeIf
          (active s M N BM BN)
          (fun idx => (KOut, outOffset s N BM BN skb skm skn idx)))
        (expected := fun idx : TileIndex [BM, BN] =>
          MemCell.of .fp16
            (FloatDType.real.cast FloatDType.fp16
              (some (rmsSpec s X KW RMS M N BM BN sxb sxm sxk skwk skwn srms
                BK numKBlocks K EPS idx.1 idx.2.1))))
    ∧ ComputeCorrect.Realizes_without_Rounding
        (kernel := rms_matmul_rbe_qkv_surface X QW KW VW RMS Q KOut V M N K sxb sxm sxk
          sqwk sqwn skwk skwn svwk svwn srms sqb sqm sqn skb skm skn svb svm svn
          BM BN BK numKBlocks EPS)
        (initialState := s)
        (write := ComputeCorrect.WriteMap.writeIf
          (active s M N BM BN)
          (fun idx => (V, outOffset s N BM BN svb svm svn idx)))
        (expected := fun idx : TileIndex [BM, BN] =>
          MemCell.of .fp16
            (FloatDType.real.cast FloatDType.fp16
              (some (rmsSpec s X VW RMS M N BM BN sxb sxm sxk svwk svwn srms
                BK numKBlocks K EPS idx.1 idx.2.1)))) := by
  subst hqn
  subst hkn
  subst hvn
  have hInjQ : Function.Injective (outOffset s N BM BN sqb sqm 1) := by
    have heq : outOffset s N BM BN sqb sqm 1
        = fun idx : TileIndex [BM, BN] =>
            (s.pids 0 * sqb + sqm * (pidM (s.pids 1) N BN * BM) + pidN (s.pids 1) N BN * BN)
              + idx.1.val * sqm + idx.2.1.val := by
      funext idx; simp only [outOffset, rowGlobal, colGlobal]; ring
    rw [heq]; exact rowMajor2D_inj _ sqm hqm
  have hInjK : Function.Injective (outOffset s N BM BN skb skm 1) := by
    have heq : outOffset s N BM BN skb skm 1
        = fun idx : TileIndex [BM, BN] =>
            (s.pids 0 * skb + skm * (pidM (s.pids 1) N BN * BM) + pidN (s.pids 1) N BN * BN)
              + idx.1.val * skm + idx.2.1.val := by
      funext idx; simp only [outOffset, rowGlobal, colGlobal]; ring
    rw [heq]; exact rowMajor2D_inj _ skm hkm
  have hInjV : Function.Injective (outOffset s N BM BN svb svm 1) := by
    have heq : outOffset s N BM BN svb svm 1
        = fun idx : TileIndex [BM, BN] =>
            (s.pids 0 * svb + svm * (pidM (s.pids 1) N BN * BM) + pidN (s.pids 1) N BN * BN)
              + idx.1.val * svm + idx.2.1.val := by
      funext idx; simp only [outOffset, rowGlobal, colGlobal]; ring
    rw [heq]; exact rowMajor2D_inj _ svm hvm
  refine ⟨?_, ?_, ?_⟩
  · -- Q realization
    rw [ComputeCorrect.realizes_writeIf_iff]
    apply ComputeKernel.computeCorrect_of_toAlgKernel
    · simp [rms_matmul_rbe_qkv_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
    intro s0 s' hExec hs0
    subst hs0
    intro idx hActive
    have hmain := qkv_exec_closed_form X QW KW VW RMS Q KOut V s0 M N sxb sxm sxk
      sqwk sqwn skwk skwn svwk svwn srms sqb sqm 1 skb skm 1 svb svm 1
      BM BN BK numKBlocks K hK EPS hInjQ hInjK hInjV
      hXQ hRMSQ hKWQ hVWQ hXK hRMSK hVWK hQK hQV hKV hundef
    have hExec2 : exec (rms_matmul_rbe_qkv_surface X QW KW VW RMS Q KOut V M N K sxb sxm sxk
        sqwk sqwn skwk skwn svwk svwn srms sqb sqm 1 skb skm 1 svb svm 1
        BM BN BK numKBlocks EPS) s0 = some s' := hExec
    rw [hExec2] at hmain
    have hcell := hmain.1 idx hActive
    simpa only [ComputeCorrect.OutputReadable.read_memcell] using hcell
  · -- K realization
    rw [ComputeCorrect.realizes_writeIf_iff]
    apply ComputeKernel.computeCorrect_of_toAlgKernel
    · simp [rms_matmul_rbe_qkv_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
    intro s0 s' hExec hs0
    subst hs0
    intro idx hActive
    have hmain := qkv_exec_closed_form X QW KW VW RMS Q KOut V s0 M N sxb sxm sxk
      sqwk sqwn skwk skwn svwk svwn srms sqb sqm 1 skb skm 1 svb svm 1
      BM BN BK numKBlocks K hK EPS hInjQ hInjK hInjV
      hXQ hRMSQ hKWQ hVWQ hXK hRMSK hVWK hQK hQV hKV hundef
    have hExec2 : exec (rms_matmul_rbe_qkv_surface X QW KW VW RMS Q KOut V M N K sxb sxm sxk
        sqwk sqwn skwk skwn svwk svwn srms sqb sqm 1 skb skm 1 svb svm 1
        BM BN BK numKBlocks EPS) s0 = some s' := hExec
    rw [hExec2] at hmain
    have hcell := hmain.2.1 idx hActive
    simpa only [ComputeCorrect.OutputReadable.read_memcell] using hcell
  · -- V realization
    rw [ComputeCorrect.realizes_writeIf_iff]
    apply ComputeKernel.computeCorrect_of_toAlgKernel
    · simp [rms_matmul_rbe_qkv_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
    intro s0 s' hExec hs0
    subst hs0
    intro idx hActive
    have hmain := qkv_exec_closed_form X QW KW VW RMS Q KOut V s0 M N sxb sxm sxk
      sqwk sqwn skwk skwn svwk svwn srms sqb sqm 1 skb skm 1 svb svm 1
      BM BN BK numKBlocks K hK EPS hInjQ hInjK hInjV
      hXQ hRMSQ hKWQ hVWQ hXK hRMSK hVWK hQK hQV hKV hundef
    have hExec2 : exec (rms_matmul_rbe_qkv_surface X QW KW VW RMS Q KOut V M N K sxb sxm sxk
        sqwk sqwn skwk skwn svwk svwn srms sqb sqm 1 skb skm 1 svb svm 1
        BM BN BK numKBlocks EPS) s0 = some s' := hExec
    rw [hExec2] at hmain
    have hcell := hmain.2.2 idx hActive
    simpa only [ComputeCorrect.OutputReadable.read_memcell] using hcell

end VeriTile.Bench.TritonBenchG.RmsMatmulRbe
