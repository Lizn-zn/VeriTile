import VeriTile.Triton
import VeriTile.Examples.FlashAttention1
import VeriTile.Examples.AttentionForwardClosedForm

/-!
# `attention_llama` — strict per-kernel correctness

Target JIT: `_fwd_kernel` — the file's ONLY `@triton.jit` kernel, a
FlashAttention-1-style streaming-softmax forward (natural `exp`, in-loop
normalization `l_rcp = 1/l_curr; p *= l_rcp; acc *= (l_prev*l_rcp)`), with
plain masked pointer loads, in-loop pointer advance
(`k_ptrs += BLOCK_N*stride_kn` / `v_ptrs += BLOCK_N*stride_vk`), and one
masked `Out` store at the end. The launcher `triton_fa` uses
`grid = (cdiv(m_size, BLOCK), head_size·batch)` with
`BLOCK_M = BLOCK_N = BLOCK = 64`, `BLOCK_DMODEL = Lk ∈ {16,32,64,128}`.

## Parameter aliasing at launch (H is the QUERY length here)

The launcher passes `N_HEAD := head_size`, `H := m_size`, `N_CTX := n_size`.
So the `offs_m[:, None] < H` mask on the `q` load and the `Out` store is a
row bound on the QUERY length — `H` is NOT a head count in this kernel,
despite its name. The head/batch decomposition is
`batch_id = tl.program_id(1) // N_HEAD`, `off_hz = tl.program_id(1) % N_HEAD`.

## Twin surfaces (`IS_CAUSAL` constexpr)

`IS_CAUSAL` is a compile-time constexpr selecting between two statement
lists, so this port keeps TWO faithful surfaces (the `triton_matmul`
twin-surface precedent):

* `attention_llama_fwd_surface` — `IS_CAUSAL = False`: keeps the scalar
  assignment `block_n_end = N_CTX` faithfully; the loop is
  `for start_n in range(0, block_n_end, BLOCK_N)` over a RUNTIME register
  bound (`forRangeDyn`).
* `attention_llama_fwd_causal_surface` — `IS_CAUSAL = True`: keeps BOTH
  assignments (`block_n_end = N_CTX`, then
  `block_n_end = (start_m + 1) * BLOCK_N + start_position`) plus the causal
  masking statement
  `qk = tl.where(offs_m[:, None] >= (block_n_offs[None, :] + start_position), qk, -inf)`.

Every other statement is identical between the twins and byte-faithful to
the Python (including the post-loop `start_m`/`offs_m`/`offs_d`
rematerializations, which are statements).

## Causal-mask direction quirk

The causal predicate is `query_row ≥ key_index + start_position`, i.e.
`start_position` SHRINKS visibility (key `j` is visible to query row `r`
iff `j + start_position ≤ r`). It does NOT model "attend to a prefix
cache" (that would be `j ≤ r + start_position`). The port models what the
kernel computes.

## Ghost-lane analysis

* The unconditional `qk = tl.where(offs_n[None, :] < N_CTX, qk, -inf)`
  uses `offs_n` (`0..BLOCK_N-1`), NOT `block_n_offs` — under
  `BLOCK_N ≤ N_CTX` (implied by both headlines' side conditions) it is an
  identity no-op. The surface keeps the statement faithfully; the proof
  discharges it as the identity.
* Non-causal, non-divisible `N_CTX`: tail-block ghost lanes
  (`block_n_offs ≥ N_CTX`) load `k = 0`, so their `qk` row is `0` and they
  contribute `exp(0 - m)` mass to the softmax denominator — an upstream
  quirk that skews the result. That is exactly why the non-causal headline
  carries `N_CTX = BLOCK_N · numKVBlocks`.
* Causal arm: ghost keys of the final partial block (`j ≥ block_n_end`)
  satisfy `j + start_position > query_row` whenever `BLOCK_M = BLOCK_N`
  (since `query_row < (pid₀+1)·BLOCK_M` and
  `j ≥ (pid₀+1)·BLOCK_N + start_position`), so the causal `where` masks
  them to `-inf` — harmless. Keys beyond `N_CTX` are excluded by
  `span ≤ N_CTX`. The causal arm therefore tolerates arbitrary
  `N_CTX ≥ span` (no divisibility of `N_CTX` needed).

## Positional-`other` load spelling

The Python spells the K/V loads with a POSITIONAL `other`
(`tl.load(k_ptrs, block_n_offs[:, None] < N_CTX, 0.)`); the DSL's
`tl.load` grammar has kwarg `other=` only, and spelling it as a kwarg
would break the audit's kwarg-sequence scan against the Python. The port
spells these two loads as plain masked loads (`MaskOpt.mask`), whose
masked-off lanes read the `undef` carrier — pinned to `0` (the exact
value the Python supplies) by the headline hypothesis `hundef`. The `q`
load's `other=0.0` is a Python kwarg and is kept as `MaskOpt.maskOther`.

## Proof architecture

```
attention_llama_fwd_closed_form_correct          ← ★ non-causal headline
  └─ al_exec (preLoop walk → forRangeDyn_inv over alInv → postLoop walk)
       └─ StreamingAccumulator streaming bridge (streaming_eq_attentionReal)
attention_llama_fwd_causal_closed_form_correct   ← ★ causal headline
  └─ alc_exec (preLoop + causal block_n_end → forRangeDyn_inv over alcInv → postLoop)
       └─ FA1MathCausal streaming bridge (streaming_eq_attentionRealCausalBlock)
```

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE); `float("-inf")` is the
`⊥ : WithBot ℝ` carrier; `p = p.to(Q.dtype.element_ty)` erases to the
identity over `ℝ` (`Q.dtype = float16`; the `element_ty` erasure
precedent); the `Out` store is `.real` — the kernel stores the fp32 `acc`
WITHOUT a cast, so both headlines are exact-ℝ statements about the stored
cells. `num_warps`/`num_stages` and the host launch (grid, the
`assert Lk in {16,32,64,128}`) are the trusted boundary; program ids are
universally quantified via `s`, covering every program of the grid.

## Translation-surface blocker

Translation-surface blocker: the `USE_FP8 = True` arm
(`k = k.to(tl.float8e5, bitcast=True); k = k.to(tl.float16)`, same for
`v`) is an int8-byte → float8e5 BIT-REINTERPRETATION (`bitcast=True`) —
the established ℝ-model-limit family (the `llama_ff_triton` /
`sgmv_expand_slice` dropped-arm precedent): a bit pattern reinterpretation
has no ℝ-level semantics. This port models `USE_FP8 = False` only
(upstream test cases 1/2); test cases 3/4 (int8 K/V) launch the fp8 arm
and are OUT of the modeled surface. The twin surfaces therefore drop the
two `if USE_FP8:` blocks, and the textual py↔lean scans in
`bench/audit_tritonbench_g.sh` exempt this port on this marker.
-/

namespace VeriTile.Bench.TritonBenchG.AttentionLlama

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-! **★ Main theorems:** `attention_llama_fwd_closed_form_correct` (non-causal),
`attention_llama_fwd_causal_closed_form_correct` (causal) -/

/-! # ══════════ CORRECT — genuine / dimension-general (review this) ══════════ -/

section Correct_without_Rounding

/-- DSL port of `attention_llama.py`'s `_fwd_kernel`, `IS_CAUSAL = False`
arm (`USE_FP8 = False`; see the header blocker for the dropped fp8 arm).
`block_n_end = N_CTX` is kept faithfully as a runtime scalar register, so
the streaming loop is a `forRangeDyn` over it. `start_position` is passed
but (faithfully) unused in this arm. -/
def attention_llama_fwd_surface
    (Q K V Out : RegionName) (sm_scale : ℝ)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      N_HEAD H N_CTX _start_position
      BLOCK_M BLOCK_N BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)

  head_idx = tl.program_id(1)
  batch_id = head_idx // $(N_HEAD)
  off_hz = head_idx % $(N_HEAD)

  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_n = tl.arange(0, $(BLOCK_N))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  off_q = batch_id * $(stride_qz) + off_hz * $(stride_qh) + offs_m[:, None] * $(stride_qm) + offs_d[None, :] * $(stride_qk)
  off_k = batch_id * $(stride_kz) + off_hz * $(stride_kh) + offs_n[:, None] * $(stride_kn) + offs_d[None, :] * $(stride_kk)
  off_v = batch_id * $(stride_vz) + off_hz * $(stride_vh) + offs_n[:, None] * $(stride_vk) + offs_d[None, :] * $(stride_vn)
  q_ptrs = Q + off_q
  k_ptrs = K + off_k
  v_ptrs = V + off_v
  m_prev = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_prev = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)
  q = tl.load(q_ptrs, offs_m[:, None] < $(H), other=0.0)
  block_n_end = $(N_CTX)
  for start_n in range($(0), block_n_end, $(BLOCK_N)) {
    block_n_offs = start_n + offs_n
    k = tl.load(k_ptrs, block_n_offs[:, None] < $(N_CTX))
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    qk += tl.dot(q, tl.trans(k))
    qk = tl.where(offs_n[None, :] < $(N_CTX), qk, float("-inf"))
    qk *= $((sm_scale : ℝ))
    m_curr = tl.maximum(tl.max(qk, 1), m_prev)
    l_prev *= tl.exp(m_prev - m_curr)
    p = tl.exp(qk - m_curr[:, None])
    l_curr = tl.sum(p, 1) + l_prev
    l_rcp = 1.0 / l_curr
    p *= l_rcp[:, None]
    acc *= (l_prev * l_rcp)[:, None]
    p = (p).to(Q.dtype.element_ty)
    v = tl.load(v_ptrs, block_n_offs[:, None] < $(N_CTX))
    acc += tl.dot(p, v)
    l_prev = l_curr
    m_prev = m_curr
    k_ptrs += $(BLOCK_N) * $(stride_kn)
    v_ptrs += $(BLOCK_N) * $(stride_vk)
  }
  start_m = tl.program_id(0)
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))

  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  off_o = batch_id * $(stride_oz) + off_hz * $(stride_oh) + offs_m[:, None] * $(stride_om) + offs_d[None, :] * $(stride_on)
  out_ptrs = Out + off_o
  tl.store(out_ptrs, acc, offs_m[:, None] < $(H))
}

/-- DSL port of `attention_llama.py`'s `_fwd_kernel`, `IS_CAUSAL = True`
arm (`USE_FP8 = False`). Keeps BOTH `block_n_end` assignments and the
causal `tl.where` statement faithfully; everything else is identical to
the non-causal twin. -/
def attention_llama_fwd_causal_surface
    (Q K V Out : RegionName) (sm_scale : ℝ)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      N_HEAD H N_CTX start_position
      BLOCK_M BLOCK_N BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)

  head_idx = tl.program_id(1)
  batch_id = head_idx // $(N_HEAD)
  off_hz = head_idx % $(N_HEAD)

  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_n = tl.arange(0, $(BLOCK_N))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  off_q = batch_id * $(stride_qz) + off_hz * $(stride_qh) + offs_m[:, None] * $(stride_qm) + offs_d[None, :] * $(stride_qk)
  off_k = batch_id * $(stride_kz) + off_hz * $(stride_kh) + offs_n[:, None] * $(stride_kn) + offs_d[None, :] * $(stride_kk)
  off_v = batch_id * $(stride_vz) + off_hz * $(stride_vh) + offs_n[:, None] * $(stride_vk) + offs_d[None, :] * $(stride_vn)
  q_ptrs = Q + off_q
  k_ptrs = K + off_k
  v_ptrs = V + off_v
  m_prev = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_prev = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)
  q = tl.load(q_ptrs, offs_m[:, None] < $(H), other=0.0)
  block_n_end = $(N_CTX)
  block_n_end = (start_m + $(1)) * $(BLOCK_N) + $(start_position)
  for start_n in range($(0), block_n_end, $(BLOCK_N)) {
    block_n_offs = start_n + offs_n
    k = tl.load(k_ptrs, block_n_offs[:, None] < $(N_CTX))
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    qk += tl.dot(q, tl.trans(k))
    qk = tl.where(offs_n[None, :] < $(N_CTX), qk, float("-inf"))
    qk *= $((sm_scale : ℝ))
    qk = tl.where(offs_m[:, None] >= (block_n_offs[None, :] + $(start_position)), qk, float("-inf"))
    m_curr = tl.maximum(tl.max(qk, 1), m_prev)
    l_prev *= tl.exp(m_prev - m_curr)
    p = tl.exp(qk - m_curr[:, None])
    l_curr = tl.sum(p, 1) + l_prev
    l_rcp = 1.0 / l_curr
    p *= l_rcp[:, None]
    acc *= (l_prev * l_rcp)[:, None]
    p = (p).to(Q.dtype.element_ty)
    v = tl.load(v_ptrs, block_n_offs[:, None] < $(N_CTX))
    acc += tl.dot(p, v)
    l_prev = l_curr
    m_prev = m_curr
    k_ptrs += $(BLOCK_N) * $(stride_kn)
    v_ptrs += $(BLOCK_N) * $(stride_vk)
  }
  start_m = tl.program_id(0)
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))

  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  off_o = batch_id * $(stride_oz) + off_hz * $(stride_oh) + offs_m[:, None] * $(stride_om) + offs_d[None, :] * $(stride_on)
  out_ptrs = Out + off_o
  tl.store(out_ptrs, acc, offs_m[:, None] < $(H))
}

/-- The non-causal surface lowers to the algorithm layer (streaming
`forRangeDyn` loop and masked loads/store included). -/
theorem attention_llama_fwd_surface_toAlgorithm_supported
    (Q K V Out : RegionName) (sm_scale : ℝ)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      N_HEAD H N_CTX start_position
      BLOCK_M BLOCK_N BLOCK_DMODEL : Nat) :
    ∃ alg, (attention_llama_fwd_surface Q K V Out sm_scale
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn
      stride_kk stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh
      stride_om stride_on N_HEAD H N_CTX start_position
      BLOCK_M BLOCK_N BLOCK_DMODEL).toAlgorithm? = Except.ok alg := by
  simp [attention_llama_fwd_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- The causal surface lowers to the algorithm layer. -/
theorem attention_llama_fwd_causal_surface_toAlgorithm_supported
    (Q K V Out : RegionName) (sm_scale : ℝ)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      N_HEAD H N_CTX start_position
      BLOCK_M BLOCK_N BLOCK_DMODEL : Nat) :
    ∃ alg, (attention_llama_fwd_causal_surface Q K V Out sm_scale
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn
      stride_kk stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh
      stride_om stride_on N_HEAD H N_CTX start_position
      BLOCK_M BLOCK_N BLOCK_DMODEL).toAlgorithm? = Except.ok alg := by
  simp [attention_llama_fwd_causal_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-! ## Named lowering (`Stmt` lists) + body splits

The two surfaces lower to a shared 18-statement preamble, a `forRangeDyn`
streaming loop (body = shared 6-statement prefix + [causal `where`] +
shared 14-statement tail), and a shared 6-statement postLoop ending in the
masked `.real` `Out` store. The causal arm inserts exactly two extra
statements: the second `block_n_end` assignment (preamble) and the causal
`tl.where` (loop body). -/

/-- The 18 lowered preamble statements shared by both arms: pids, the
`batch_id`/`off_hz` split, the three index vectors, the flat 2D offset
tiles, the three pointer tiles, the `⊥`/zero seeds, the H-masked `q` load
(`other=0.0`), and `block_n_end = N_CTX`. -/
def alPreLoopG (Q K V : RegionName)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn
      N_HEAD H N_CTX BM BN BD : Nat) : List Stmt :=
  [ Stmt.assign .nat [] "start_m" (Op.programId 0),
    Stmt.assign .nat [] "head_idx" (Op.programId 1),
    Stmt.assign .nat [] "batch_id"
      (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.ref .nat [] "head_idx") (Op.constNat N_HEAD)),
    Stmt.assign .nat [] "off_hz"
      (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] "head_idx") (Op.constNat N_HEAD)),
    Stmt.assign .nat [BM] "offs_m"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat BM)) (Op.arange BM)),
    Stmt.assign .nat [BN] "offs_n" (Op.arange BN),
    Stmt.assign .nat [BD] "offs_d" (Op.arange BD),
    Stmt.assign .nat [BM, BD] "off_q"
      (Op.add .nat Broadcast.nil.consL.consR
        (Op.add .nat Broadcast.scalarL
          (Op.add .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "batch_id") (Op.constNat sqz))
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat sqh)))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m")) (Op.constNat sqm)))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BD] "offs_d")) (Op.constNat sqk))),
    Stmt.assign .nat [BN, BD] "off_k"
      (Op.add .nat Broadcast.nil.consL.consR
        (Op.add .nat Broadcast.scalarL
          (Op.add .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "batch_id") (Op.constNat skz))
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat skh)))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BN] "offs_n")) (Op.constNat skn)))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BD] "offs_d")) (Op.constNat skk))),
    Stmt.assign .nat [BN, BD] "off_v"
      (Op.add .nat Broadcast.nil.consL.consR
        (Op.add .nat Broadcast.scalarL
          (Op.add .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "batch_id") (Op.constNat svz))
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat svh)))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BN] "offs_n")) (Op.constNat svk)))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BD] "offs_d")) (Op.constNat svn))),
    Stmt.assign .ptr [BM, BD] "q_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Q) (Op.ref .nat [BM, BD] "off_q")),
    Stmt.assign .ptr [BN, BD] "k_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase K) (Op.ref .nat [BN, BD] "off_k")),
    Stmt.assign .ptr [BN, BD] "v_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase V) (Op.ref .nat [BN, BD] "off_v")),
    Stmt.assign .real [BM] "m_prev"
      (Op.add .real Broadcast.scalarR (Op.full [BM] (Op.const 0)) Op.negInf),
    Stmt.assign .real [BM] "l_prev" (Op.full [BM] (Op.const 0)),
    Stmt.assign .real [BM, BD] "acc" (Op.full [BM, BD] (Op.const 0)),
    Stmt.assign .real [BM, BD] "q"
      (Op.load .real (MemAccess.ptr (Op.ref .ptr [BM, BD] "q_ptrs"))
        (MaskOpt.maskOther
          (Op.remap [BM, BD] Broadcast.nil.consL.consSame.leftIndex
            (Op.lt ComparableDType.nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m")) (Op.constNat H)))
          (Op.broadcast (Op.const (0.0 : ℝ)) [BM, BD]))),
    Stmt.assign .nat [] "block_n_end" (Op.constNat N_CTX) ]

/-- The causal arm's second `block_n_end` assignment
(`block_n_end = (start_m + 1) * BLOCK_N + start_position`). -/
def alBnEndCausalStmt (SP BN : Nat) : Stmt :=
  Stmt.assign .nat [] "block_n_end"
    (Op.add .nat Broadcast.nil
      (Op.mul .nat Broadcast.nil
        (Op.add .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat 1)) (Op.constNat BN))
      (Op.constNat SP))

/-- The 6 lowered loop-body prefix statements shared by both arms:
`block_n_offs`, the masked `k` load, `qk` zeros, `qk += q·kᵀ`, the
(identity) `offs_n < N_CTX` `where`, and the `sm_scale` scaling. -/
def alLoopPrefixG (sm_scale : ℝ) (N_CTX BM BN BD : Nat) : List Stmt :=
  [ Stmt.assign .nat [BN] "block_n_offs"
      (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n") (Op.ref .nat [BN] "offs_n")),
    Stmt.assign .real [BN, BD] "k"
      (Op.load .real (MemAccess.ptr (Op.ref .ptr [BN, BD] "k_ptrs"))
        (MaskOpt.mask
          (Op.remap [BN, BD] Broadcast.nil.consL.consSame.leftIndex
            (Op.lt ComparableDType.nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BN] "block_n_offs")) (Op.constNat N_CTX))))),
    Stmt.assign .real [BM, BN] "qk" (Op.full [BM, BN] (Op.const 0)),
    Stmt.assign .real [BM, BN] "qk"
      (Op.add .real Broadcast.nil.consSame.consSame (Op.ref .real [BM, BN] "qk")
        (Op.dot (batch := []) (Op.ref .real [BM, BD] "q")
          (Op.transpose (batch := []) (Op.ref .real [BN, BD] "k")))),
    Stmt.assign .real [BM, BN] "qk"
      (Op.where
        (Op.remap [BM, BN] Broadcast.nil.consSame.consL.leftIndex
          (Op.lt ComparableDType.nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_n")) (Op.constNat N_CTX)))
        (Op.ref .real [BM, BN] "qk") (Op.broadcast Op.negInf [BM, BN])),
    Stmt.assign .real [BM, BN] "qk"
      (Op.mul .real Broadcast.scalarR (Op.ref .real [BM, BN] "qk") (Op.const sm_scale)) ]

/-- The causal masking statement
(`qk = tl.where(offs_m[:, None] >= (block_n_offs[None, :] + start_position), qk, -inf)`). -/
def alCausalWhereStmt (SP BM BN : Nat) : Stmt :=
  Stmt.assign .real [BM, BN] "qk"
    (Op.where
      (Op.ge ComparableDType.nat Broadcast.nil.consL.consR
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m"))
        (Op.add .nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "block_n_offs")) (Op.constNat SP)))
      (Op.ref .real [BM, BN] "qk") (Op.broadcast Op.negInf [BM, BN]))

/-- The 14 lowered loop-body tail statements shared by both arms: the
`m_curr`/`l_prev`/`p`/`l_curr`/`l_rcp` recurrence, the two rescales, the
(erased) `p = p.to(Q.dtype.element_ty)` re-assign, the masked `v` load,
`acc += p·v`, the `l_prev`/`m_prev` carries, and the two pointer advances. -/
def alLoopTailG (skn svk N_CTX BM BN BD : Nat) : List Stmt :=
  [ Stmt.assign .real [BM] "m_curr"
      (Op.where
        (Op.gt .real Broadcast.nil.consSame
          (Op.reduceMax (⟨1, by simp⟩ : Fin [BM, BN].length) Bool.false (Op.ref .real [BM, BN] "qk"))
          (Op.ref .real [BM] "m_prev"))
        (Op.reduceMax (⟨1, by simp⟩ : Fin [BM, BN].length) Bool.false (Op.ref .real [BM, BN] "qk"))
        (Op.ref .real [BM] "m_prev")),
    Stmt.assign .real [BM] "l_prev"
      (Op.mul .real Broadcast.nil.consSame (Op.ref .real [BM] "l_prev")
        (Op.exp (Op.sub .real Broadcast.nil.consSame
          (Op.ref .real [BM] "m_prev") (Op.ref .real [BM] "m_curr")))),
    Stmt.assign .real [BM, BN] "p"
      (Op.exp (Op.sub .real Broadcast.nil.consR.consSame (Op.ref .real [BM, BN] "qk")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BM] "m_curr")))),
    Stmt.assign .real [BM] "l_curr"
      (Op.add .real Broadcast.nil.consSame
        (Op.reduceSum (⟨1, by simp⟩ : Fin [BM, BN].length) Bool.false (Op.ref .real [BM, BN] "p"))
        (Op.ref .real [BM] "l_prev")),
    Stmt.assign .real [BM] "l_rcp"
      (Op.div .real Broadcast.scalarL (Op.const (1.0 : ℝ)) (Op.ref .real [BM] "l_curr")),
    Stmt.assign .real [BM, BN] "p"
      (Op.mul .real Broadcast.nil.consR.consSame (Op.ref .real [BM, BN] "p")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BM] "l_rcp"))),
    Stmt.assign .real [BM, BD] "acc"
      (Op.mul .real Broadcast.nil.consR.consSame (Op.ref .real [BM, BD] "acc")
        (Op.expandDim ⟨1, by simp⟩
          (Op.mul .real Broadcast.nil.consSame (Op.ref .real [BM] "l_prev") (Op.ref .real [BM] "l_rcp")))),
    Stmt.assign .real [BM, BN] "p" (Op.ref .real [BM, BN] "p"),
    Stmt.assign .real [BN, BD] "v"
      (Op.load .real (MemAccess.ptr (Op.ref .ptr [BN, BD] "v_ptrs"))
        (MaskOpt.mask
          (Op.remap [BN, BD] Broadcast.nil.consL.consSame.leftIndex
            (Op.lt ComparableDType.nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BN] "block_n_offs")) (Op.constNat N_CTX))))),
    Stmt.assign .real [BM, BD] "acc"
      (Op.add .real Broadcast.nil.consSame.consSame (Op.ref .real [BM, BD] "acc")
        (Op.dot (batch := []) (Op.ref .real [BM, BN] "p") (Op.ref .real [BN, BD] "v"))),
    Stmt.assign .real [BM] "l_prev" (Op.ref .real [BM] "l_curr"),
    Stmt.assign .real [BM] "m_prev" (Op.ref .real [BM] "m_curr"),
    Stmt.assign .ptr [BN, BD] "k_ptrs"
      (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BN, BD] "k_ptrs")
        (Op.mul .nat Broadcast.nil (Op.constNat BN) (Op.constNat skn))),
    Stmt.assign .ptr [BN, BD] "v_ptrs"
      (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BN, BD] "v_ptrs")
        (Op.mul .nat Broadcast.nil (Op.constNat BN) (Op.constNat svk))) ]

/-- Non-causal loop body = shared prefix ++ shared tail (20 statements). -/
def alLoopBodyG (sm_scale : ℝ) (skn svk N_CTX BM BN BD : Nat) : List Stmt :=
  alLoopPrefixG sm_scale N_CTX BM BN BD ++ alLoopTailG skn svk N_CTX BM BN BD

/-- Causal loop body = shared prefix ++ causal `where` ++ shared tail
(21 statements). -/
def alLoopBodyCausalG (sm_scale : ℝ) (skn svk N_CTX SP BM BN BD : Nat) : List Stmt :=
  alLoopPrefixG sm_scale N_CTX BM BN BD
    ++ alCausalWhereStmt SP BM BN :: alLoopTailG skn svk N_CTX BM BN BD

/-- The 6 lowered postLoop statements shared by both arms: the
`start_m`/`offs_m`/`offs_d` rematerializations, the `off_o` offset tile,
`out_ptrs`, and the H-masked `.real` `Out` store of `acc`. -/
def alPostLoopG (Out : RegionName) (soz soh som son H BM BD : Nat) : List Stmt :=
  [ Stmt.assign .nat [] "start_m" (Op.programId 0),
    Stmt.assign .nat [BM] "offs_m"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat BM)) (Op.arange BM)),
    Stmt.assign .nat [BD] "offs_d" (Op.arange BD),
    Stmt.assign .nat [BM, BD] "off_o"
      (Op.add .nat Broadcast.nil.consL.consR
        (Op.add .nat Broadcast.scalarL
          (Op.add .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "batch_id") (Op.constNat soz))
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat soh)))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m")) (Op.constNat som)))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BD] "offs_d")) (Op.constNat son))),
    Stmt.assign .ptr [BM, BD] "out_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Out) (Op.ref .nat [BM, BD] "off_o")),
    Stmt.store .real [BM, BD] (MemAccess.ptr (Op.ref .ptr [BM, BD] "out_ptrs"))
      (Op.ref .real [BM, BD] "acc")
      (MaskOpt.mask
        (Op.remap [BM, BD] Broadcast.nil.consL.consSame.leftIndex
          (Op.lt ComparableDType.nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m")) (Op.constNat H)))) ]

set_option maxRecDepth 8000 in
/-- The lowered non-causal body is exactly
`alPreLoopG ++ forRangeDyn (prefix ++ tail) :: alPostLoopG`. -/
theorem al_body_split (Q K V Out : RegionName) (sm_scale : ℝ)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      N_HEAD H N_CTX SP BM BN BD : Nat) :
    (attention_llama_fwd_surface Q K V Out sm_scale
      sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      N_HEAD H N_CTX SP BM BN BD).toAlgKernel.body
      = alPreLoopG Q K V sqz sqh sqm sqk skz skh skn skk svz svh svk svn N_HEAD H N_CTX BM BN BD
        ++ (Stmt.forRangeDyn "start_n" (Op.constNat 0) (Op.ref .nat [] "block_n_end")
              (Op.constNat BN) (alLoopBodyG sm_scale skn svk N_CTX BM BN BD)
            :: alPostLoopG Out soz soh som son H BM BD) := by
  rfl

set_option maxRecDepth 8000 in
/-- The lowered causal body is exactly
`alPreLoopG ++ [causal block_n_end] ++ forRangeDyn (prefix ++ where ++ tail) :: alPostLoopG`. -/
theorem alc_body_split (Q K V Out : RegionName) (sm_scale : ℝ)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      N_HEAD H N_CTX SP BM BN BD : Nat) :
    (attention_llama_fwd_causal_surface Q K V Out sm_scale
      sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      N_HEAD H N_CTX SP BM BN BD).toAlgKernel.body
      = alPreLoopG Q K V sqz sqh sqm sqk skz skh skn skk svz svh svk svn N_HEAD H N_CTX BM BN BD
        ++ (alBnEndCausalStmt SP BN
            :: Stmt.forRangeDyn "start_n" (Op.constNat 0) (Op.ref .nat [] "block_n_end")
                 (Op.constNat BN) (alLoopBodyCausalG sm_scale skn svk N_CTX SP BM BN BD)
            :: alPostLoopG Out soz soh som son H BM BD) := by
  rfl

/-! ## Genuine input-memory spec layer

All tiles are read from the INITIAL state's memory through the kernel's own
offset expressions (`batch_id·stride_z + off_hz·stride_h` base + row/col
strides). `alQTileG` carries the kernel's own `row < H` load guard (the
`paQGuarded` precedent); on active output rows (`alActive`) the guard is
true, so the headline value is the pure `readMem` attention there. -/

/-- The program's batch/head base offset for a region with batch stride
`sz` and head stride `sh`: `batch_id·sz + off_hz·sh` with
`batch_id = pids 1 / N_HEAD`, `off_hz = pids 1 % N_HEAD`. -/
def alBase (s : BlockState) (N_HEAD sz sh : Nat) : Nat :=
  (s.pids 1 / N_HEAD) * sz + (s.pids 1 % N_HEAD) * sh

/-- Global query row of local row `i`. -/
def alRow (s : BlockState) (BM : Nat) (i : Fin BM) : Nat :=
  s.pids 0 * BM + i.val

/-- Active (stored) output lanes: the kernel's `offs_m < H` store mask
(`H` = the launcher's `m_size`, the QUERY length). -/
def alActive {BD : Nat} (s : BlockState) (H BM : Nat)
    (idx : TileIndex [BM, BD]) : Prop :=
  alRow s BM idx.1 < H

instance {BD : Nat} (s : BlockState) (H BM : Nat) :
    DecidablePred (alActive (BD := BD) s H BM) := fun idx =>
  inferInstanceAs (Decidable (alRow s BM idx.1 < H))

/-- Flat `Out` offset of output lane `idx` (the kernel's own
`off_o = batch_id·stride_oz + off_hz·stride_oh + row·stride_om + col·stride_on`). -/
def alOutOffset (s : BlockState) (N_HEAD soz soh som son BM : Nat)
    {BD : Nat} (idx : TileIndex [BM, BD]) : Nat :=
  alBase s N_HEAD soz soh + alRow s BM idx.1 * som + idx.2.1.val * son

/-- The loaded (H-guarded) Q tile: the kernel's masked
`q = tl.load(q_ptrs, offs_m[:,None] < H, other=0.0)`. -/
noncomputable def alQTileG (s : BlockState) (Q : RegionName)
    (N_HEAD sqz sqh sqm sqk H BM BD : Nat) :
    TileIndex [BM, BD] → ℝ :=
  fun idx =>
    if alRow s BM idx.1 < H then
      s.readMem Q (alBase s N_HEAD sqz sqh + alRow s BM idx.1 * sqm + idx.2.1.val * sqk)
    else 0

/-- The K tile over the streamed key span `SEQ` (every loaded lane is
in-range under the headline's side conditions, so no guard). -/
noncomputable def alKTileG (s : BlockState) (K : RegionName)
    (N_HEAD skz skh skn skk SEQ BD : Nat) :
    TileIndex [SEQ, BD] → ℝ :=
  fun idx =>
    s.readMem K (alBase s N_HEAD skz skh + idx.1.val * skn + idx.2.1.val * skk)

/-- The V tile over the streamed key span `SEQ`. -/
noncomputable def alVTileG (s : BlockState) (V : RegionName)
    (N_HEAD svz svh svk svn SEQ BD : Nat) :
    TileIndex [SEQ, BD] → ℝ :=
  fun idx =>
    s.readMem V (alBase s N_HEAD svz svh + idx.1.val * svk + idx.2.1.val * svn)

/-- Literal 2D pointer tile at row offset `rowOff`: lane `(i, d)` points at
`(R, base + (rowOff + i)·srow + d·scol)`. Used for `k_ptrs`/`v_ptrs`
(rowOff = the streaming counter) and `q_ptrs`/`out_ptrs` (rowOff = the
block's global row base, srow/scol = row/col strides). -/
def alPtrTile (R : RegionName) (base srow scol rowOff : Nat) (A B : Nat) :
    Tile .ptr [A, B] :=
  ⟨fun idx : TileIndex [A, B] =>
    (R, base + (rowOff + idx.1.val) * srow + idx.2.1.val * scol)⟩

/-! ## Per-statement op-eval recipes (RECIPE LAYER)

Standalone `evalOp` reductions with abstract register-readback hypotheses,
the plain-pointer analogues of `triton_attention`'s `ta_*` recipe family
(same loop-body math statements) plus the `remap`-broadcast mask forms of
`context_attn_fwd`. -/

/-- Eval helper for `floorDiv` (the `batch_id = head_idx // N_HEAD` split). -/
theorem al_evalOp_floorDiv {dtype a b shape} (h : IntegralDType dtype)
    (bc : Broadcast a b shape) (x : Op dtype a) (y : Op dtype b) (s : BlockState) :
    evalOp (.floorDiv h bc x y) s = (do
      let vx ← evalOp x s; let vy ← evalOp y s; some (Tile.bop h.floorDiv bc vx vy)) := by
  simp [evalOp]

/-- Eval helper for `mod` (the `off_hz = head_idx % N_HEAD` split). -/
theorem al_evalOp_mod {dtype a b shape} (h : IntegralDType dtype)
    (bc : Broadcast a b shape) (x : Op dtype a) (y : Op dtype b) (s : BlockState) :
    evalOp (.mod h bc x y) s = (do
      let vx ← evalOp x s; let vy ← evalOp y s; some (Tile.bop h.mod bc vx vy)) := by
  simp [evalOp]

/-- Eval helper for `remap` (the row/column broadcast of the load/store
masks). -/
theorem al_evalOp_remap {dtype outShape inShape}
    (map : TileIndex outShape → TileIndex inShape) (a : Op dtype inShape) (s : BlockState) :
    evalOp (.remap outShape map a) s = (do
      let v ← evalOp a s; some (Tile.remap map v)) := by
  simp [evalOp]

/-- Eval helper for the `>=` causal predicate (`Op.ge`), which has no
dedicated `@[simp]` form. -/
theorem al_evalOp_ge {dtype a b shape} (h : ComparableDType dtype)
    (bc : Broadcast a b shape) (x : Op dtype a) (y : Op dtype b) (s : BlockState) :
    evalOp (.ge h bc x y) s = (do
      let vx ← evalOp x s; let vy ← evalOp y s; some (Tile.cop h.ge bc vx vy)) := by
  simp [evalOp]

/-- **Row mask eval** (`(offs_m[:, None] < H)` broadcast over a `[BM, S2]`
tile via `remap`): `mask[r, c] = decide (gm r < H)`. Shared by the `q`
load and the `Out` store. -/
theorem al_rowmask_eval {BM S2 : Nat} (s : BlockState) (gm : Fin BM → Nat) (H : Nat)
    (hm : s.regs .nat [BM] "offs_m" = some (Tile.vec gm)) :
    evalOp (Op.remap [BM, S2] Broadcast.nil.consL.consSame.leftIndex
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m")) (Op.constNat H))) s
      = some (⟨fun idx : TileIndex [BM, S2] => decide (gm idx.1 < H)⟩ : Tile .bool [BM, S2]) := by
  have hexp : @evalOp TileDType.nat [BM, 1]
      (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m")) s
        = some (Tile.expandDim ⟨1, by simp⟩ (Tile.vec gm)) :=
    evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hm
  rw [al_evalOp_remap, evalOp_lt]
  simp only [hexp, evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
  rfl

/-- **Key-lane mask eval** (`(block_n_offs[:, None] < N_CTX)` broadcast over
a `[BN, BD]` tile via `remap`): `mask[j, d] = decide (SN + j < N_CTX)`.
Shared by the `k` and `v` loads. -/
theorem al_colmask_eval {BN BD : Nat} (s : BlockState) (SN N_CTX : Nat)
    (hb : s.regs .nat [BN] "block_n_offs" = some (Tile.vec (fun j : Fin BN => SN + j.val))) :
    evalOp (Op.remap [BN, BD] Broadcast.nil.consL.consSame.leftIndex
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BN] "block_n_offs")) (Op.constNat N_CTX))) s
      = some (⟨fun idx : TileIndex [BN, BD] => decide (SN + idx.1.val < N_CTX)⟩ : Tile .bool [BN, BD]) := by
  have hexp : @evalOp TileDType.nat [BN, 1]
      (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BN] "block_n_offs")) s
        = some (Tile.expandDim ⟨1, by simp⟩ (Tile.vec (fun j : Fin BN => SN + j.val))) :=
    evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hb
  rw [al_evalOp_remap, evalOp_lt]
  simp only [hexp, evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
  rfl

/-- **Masked pointer-load with `other` characterization** (the `q` load).
Per lane: `some (if mask then readMem else other)`. The `MaskOpt.maskOther`
sibling of `AttentionForwardClosedForm.load_ptr_mask_real`. -/
theorem al_load_ptr_maskOther_real {shape : TileShape}
    (ptrOp : Op .ptr shape) (maskOp : Op .bool shape) (otherOp : Op .real shape)
    (s : BlockState) (ptrs : Tile .ptr shape) (masks : Tile .bool shape)
    (others : Tile .real shape)
    (hp : evalOp ptrOp s = some ptrs) (hm : evalOp maskOp s = some masks)
    (ho : evalOp otherOp s = some others) :
    evalOp (.load .real (.ptr ptrOp) (.maskOther maskOp otherOp)) s
      = some ⟨fun i => if masks.data i then
          some (s.readMem (ptrs.data i).1 (ptrs.data i).2) else others.data i⟩ := by
  simp only [evalOp, hp, hm, ho, Option.bind]
  refine congrArg some ?_
  ext i
  simp only [BlockState.readMemValue_real]
  cases hmi : masks.data i <;> simp [hmi]

/-- **`qk = tl.zeros([BM, BN])` eval** (all-`0` tile). -/
theorem al_qkzeros_eval (s : BlockState) (BM BN : Nat) :
    evalOp (Op.full [BM, BN] (Op.const 0)) s
      = some (⟨fun _ : TileIndex [BM, BN] => some (0 : ℝ)⟩ : Tile .real [BM, BN]) := by
  simp [evalOp_full, evalOp_const, Option.bind]

/-- **`qk += tl.dot(q, tl.trans(k))` eval**. -/
theorem al_qk_dot_eval (s : BlockState) (BM BN BD : Nat)
    (qktile : Tile .real [BM, BN]) (qtile : Tile .real [BM, BD]) (ktile : Tile .real [BN, BD])
    (hqk : s.regs .real [BM, BN] "qk" = some qktile)
    (hq : s.regs .real [BM, BD] "q" = some qtile)
    (hk : s.regs .real [BN, BD] "k" = some ktile) :
    evalOp (Op.add .real Broadcast.nil.consSame.consSame
        (Op.ref .real [BM, BN] "qk")
        (Op.dot (batch := []) (Op.ref .real [BM, BD] "q")
          (Op.transpose (batch := []) (Op.ref .real [BN, BD] "k")))) s
      = some (Tile.bop NumericDType.real.add Broadcast.nil.consSame.consSame
          qktile (Tile.dot [] qtile (Tile.transpose [] ktile))) := by
  have hkr : evalOp (Op.ref .real [BN, BD] "k") s = some ktile := by rw [evalOp_ref, hk]
  have hqkr : evalOp (Op.ref .real [BM, BD] "q") s = some qtile := by rw [evalOp_ref, hq]
  have htr : evalOp (Op.transpose (batch := []) (Op.ref .real [BN, BD] "k")) s
      = some (Tile.transpose [] ktile) := by
    erw [evalOp_transpose [] (Op.ref .real [BN, BD] "k"), hkr]; rfl
  have hdot : @evalOp TileDType.real [BM, BN]
      (Op.dot (batch := []) (Op.ref .real [BM, BD] "q")
        (Op.transpose (batch := []) (Op.ref .real [BN, BD] "k"))) s
      = some (Tile.dot [] qtile (Tile.transpose [] ktile)) := by
    erw [evalOp_dot [] (Op.ref .real [BM, BD] "q")
      (Op.transpose (batch := []) (Op.ref .real [BN, BD] "k")), hqkr, htr]; rfl
  rw [evalOp_add]
  simp only [evalOp_ref, hqk, hdot, Option.bind_eq_bind, Option.bind_some]; rfl

/-- **Identity `where` eval** (`qk = tl.where(offs_n[None, :] < N_CTX, qk, -inf)`):
under `BLOCK_N ≤ N_CTX` every lane satisfies `offs_n < N_CTX`, so the tile
is unchanged — the kernel's `offs_n`-vs-`block_n_offs` misuse is a no-op. -/
theorem al_wheren_id_eval (s : BlockState) (BM BN N_CTX : Nat)
    (hBNle : BN ≤ N_CTX) (qktile : Tile .real [BM, BN])
    (hon : s.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => j.val)))
    (hqk : s.regs .real [BM, BN] "qk" = some qktile) :
    evalOp (Op.where
        (Op.remap [BM, BN] Broadcast.nil.consSame.consL.leftIndex
          (Op.lt ComparableDType.nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_n")) (Op.constNat N_CTX)))
        (Op.ref .real [BM, BN] "qk") (Op.broadcast Op.negInf [BM, BN])) s
      = some qktile := by
  have hexp : @evalOp TileDType.nat [1, BN]
      (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_n")) s
        = some (Tile.expandDim ⟨0, by simp⟩ (Tile.vec (fun j : Fin BN => j.val))) :=
    evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hon
  have hmask : evalOp (Op.remap [BM, BN] Broadcast.nil.consSame.consL.leftIndex
      (Op.lt ComparableDType.nat Broadcast.scalarR
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_n")) (Op.constNat N_CTX))) s
      = some (⟨fun idx : TileIndex [BM, BN] => decide (idx.2.1.val < N_CTX)⟩ : Tile .bool [BM, BN]) := by
    rw [al_evalOp_remap, evalOp_lt]
    simp only [hexp, evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
    rfl
  have hbcast : @evalOp TileDType.real [BM, BN] (Op.broadcast Op.negInf [BM, BN]) s
      = some (⟨fun _ : TileIndex [BM, BN] => (⊥ : WithBot ℝ)⟩ : Tile .real [BM, BN]) := by
    simp only [evalOp, evalOp_negInf, Option.bind_eq_bind, Option.bind_some]; rfl
  rw [evalOp_where]
  simp only [hmask, evalOp_ref, hqk, hbcast, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  have hlt : idx.2.1.val < N_CTX := lt_of_lt_of_le idx.2.1.isLt hBNle
  simp only [Tile.select_data, hlt, decide_true, if_true]

/-- **`qk *= sm_scale` eval**. -/
theorem al_qk_scale_eval (s : BlockState) (BM BN : Nat) (sc : ℝ)
    (qktile : Tile .real [BM, BN]) (hqk : s.regs .real [BM, BN] "qk" = some qktile) :
    evalOp (Op.mul .real Broadcast.scalarR (Op.ref .real [BM, BN] "qk") (Op.const sc)) s
      = some (Tile.bop NumericDType.real.mul Broadcast.scalarR qktile
          (Tile.scalar (some sc : WithBot ℝ))) := by
  rw [evalOp_mul]; simp [evalOp_ref, evalOp_const, hqk]

/-- **Causal `where` eval**
(`qk = tl.where(offs_m[:, None] >= (block_n_offs[None, :] + SP), qk, -inf)`):
kept lanes satisfy `SN + j + SP ≤ gm i`, others become `⊥`. -/
theorem al_where_causal_eval (s : BlockState) (BM BN SN SP : Nat)
    (gm : Fin BM → Nat) (qktile : Tile .real [BM, BN])
    (hom : s.regs .nat [BM] "offs_m" = some (Tile.vec gm))
    (hbn : s.regs .nat [BN] "block_n_offs" = some (Tile.vec (fun j : Fin BN => SN + j.val)))
    (hqk : s.regs .real [BM, BN] "qk" = some qktile) :
    evalOp (Op.where
        (Op.ge ComparableDType.nat Broadcast.nil.consL.consR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m"))
          (Op.add .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "block_n_offs")) (Op.constNat SP)))
        (Op.ref .real [BM, BN] "qk") (Op.broadcast Op.negInf [BM, BN])) s
      = some ⟨fun idx : TileIndex [BM, BN] =>
          if SN + idx.2.1.val + SP ≤ gm idx.1 then qktile.data idx else (⊥ : WithBot ℝ)⟩ := by
  have hexpM : @evalOp TileDType.nat [BM, 1]
      (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m")) s
        = some (Tile.expandDim ⟨1, by simp⟩ (Tile.vec gm)) :=
    evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hom
  have hexpN : @evalOp TileDType.nat [1, BN]
      (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "block_n_offs")) s
        = some (Tile.expandDim ⟨0, by simp⟩ (Tile.vec (fun j : Fin BN => SN + j.val))) :=
    evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hbn
  have haddN : @evalOp TileDType.nat [1, BN]
      (Op.add .nat Broadcast.scalarR
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "block_n_offs")) (Op.constNat SP)) s
        = some (Tile.bop NumericDType.nat.add Broadcast.scalarR
            (Tile.expandDim ⟨0, by simp⟩ (Tile.vec (fun j : Fin BN => SN + j.val)))
            (Tile.scalar SP)) := by
    rw [evalOp_add, hexpN]
    simp only [evalOp_constNat, Option.bind_eq_bind, Option.bind_some]; rfl
  have hbcast : @evalOp TileDType.real [BM, BN] (Op.broadcast Op.negInf [BM, BN]) s
      = some (⟨fun _ : TileIndex [BM, BN] => (⊥ : WithBot ℝ)⟩ : Tile .real [BM, BN]) := by
    simp only [evalOp, evalOp_negInf, Option.bind_eq_bind, Option.bind_some]; rfl
  rw [evalOp_where, al_evalOp_ge]
  simp only [hexpM, haddN, evalOp_ref, hqk, hbcast, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  simp only [Tile.select_data, Tile.cop_data, Tile.bop_data, Broadcast.leftIndex,
    Broadcast.rightIndex, Tile.expandDim_data, Tile.scalar, Tile.vec,
    ComparableDType.nat, NumericDType.add]
  by_cases h : SN + idx.2.1.val + SP ≤ gm idx.1
  · rw [if_pos (by simpa using h)]; simp [h]
  · rw [if_neg (by simpa using h)]; simp [h]

/-- **`m_curr = tl.maximum(tl.max(qk, 1), m_prev)` eval** (lowered to
`where(reduceMax(qk,1) > m_prev, reduceMax(qk,1), m_prev)`). -/
theorem al_mij_eval (s : BlockState) (BM BN : Nat)
    (mtile : Tile .real [BM]) (qktile : Tile .real [BM, BN]) (rmaxT : Tile .real [BM])
    (hmp : s.regs .real [BM] "m_prev" = some mtile)
    (hqk : s.regs .real [BM, BN] "qk" = some qktile)
    (hrm : Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [BM, BN].length) qktile = some rmaxT) :
    evalOp (Op.where
        (Op.gt .real Broadcast.nil.consSame
          (Op.reduceMax (⟨1, by simp⟩ : Fin [BM, BN].length) Bool.false (Op.ref .real [BM, BN] "qk"))
          (Op.ref .real [BM] "m_prev"))
        (Op.reduceMax (⟨1, by simp⟩ : Fin [BM, BN].length) Bool.false (Op.ref .real [BM, BN] "qk"))
        (Op.ref .real [BM] "m_prev")) s
      = some (Tile.select
          (Tile.cop ComparableDType.real.gt Broadcast.nil.consSame rmaxT mtile)
          rmaxT mtile) := by
  have hrmaxN : evalOp (Op.reduceMax (⟨1, by simp⟩ : Fin [BM, BN].length) Bool.false
      (Op.ref .real [BM, BN] "qk")) s = some rmaxT := by
    rw [evalOp_reduceMax]; simp only [evalOp_ref, hqk]; exact hrm
  have hrmax : @evalOp TileDType.real [BM]
      (Op.reduceMax (⟨1, by simp⟩ : Fin [BM, BN].length) Bool.false
        (Op.ref .real [BM, BN] "qk")) s = some rmaxT := hrmaxN
  rw [evalOp_where]
  simp only [evalOp_gt, evalOp_ref, hmp, hrmax, Option.bind_eq_bind, Option.bind_some]

/-- **`l_prev *= tl.exp(m_prev - m_curr)` eval** (the α rescale, natural exp). -/
theorem al_alpha_eval (s : BlockState) (BM : Nat) (ltile mp mc : Tile .real [BM])
    (hl : s.regs .real [BM] "l_prev" = some ltile)
    (hmp : s.regs .real [BM] "m_prev" = some mp)
    (hmc : s.regs .real [BM] "m_curr" = some mc) :
    evalOp (Op.mul .real Broadcast.nil.consSame
        (Op.ref .real [BM] "l_prev")
        (Op.exp (Op.sub .real Broadcast.nil.consSame
          (Op.ref .real [BM] "m_prev") (Op.ref .real [BM] "m_curr")))) s
      = some (Tile.bop NumericDType.real.mul Broadcast.nil.consSame ltile
          (Tile.uop WithBot.realExp
            (Tile.bop NumericDType.real.sub Broadcast.nil.consSame mp mc))) := by
  rw [evalOp_mul]
  simp only [evalOp_ref, evalOp_exp, evalOp_sub, hl, hmp, hmc,
    Option.bind_eq_bind, Option.bind_some]

/-- **`p = tl.exp(qk - m_curr[:, None])` eval** (natural-exp numerator shift). -/
theorem al_p_eval (s : BlockState) (BM BN : Nat) (hax : 1 < [BM].length.succ)
    (qktile : Tile .real [BM, BN]) (mc : Tile .real [BM])
    (hqk : s.regs .real [BM, BN] "qk" = some qktile)
    (hmc : s.regs .real [BM] "m_curr" = some mc) :
    evalOp (Op.exp (Op.sub .real Broadcast.nil.consR.consSame
        (Op.ref .real [BM, BN] "qk") (Op.expandDim ⟨1, hax⟩ (Op.ref .real [BM] "m_curr")))) s
      = some (Tile.uop WithBot.realExp
          (Tile.bop NumericDType.real.sub Broadcast.nil.consR.consSame
            qktile (Tile.expandDim ⟨1, hax⟩ mc))) := by
  have hexp : @evalOp TileDType.real [BM, 1]
      (Op.expandDim ⟨1, hax⟩ (Op.ref .real [BM] "m_curr")) s
      = some (Tile.expandDim ⟨1, hax⟩ mc) := evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hmc
  rw [evalOp_exp, evalOp_sub]
  simp only [evalOp_ref, hqk, hexp, Option.bind_eq_bind, Option.bind_some]; rfl

/-- **`l_curr = tl.sum(p, 1) + l_prev` eval**. -/
theorem al_lij_eval (s : BlockState) (BM BN : Nat)
    (ptile : Tile .real [BM, BN]) (ltile : Tile .real [BM])
    (hp : s.regs .real [BM, BN] "p" = some ptile)
    (hl : s.regs .real [BM] "l_prev" = some ltile) :
    evalOp (Op.add .real Broadcast.nil.consSame
        (Op.reduceSum (⟨1, by simp⟩ : Fin [BM, BN].length) Bool.false (Op.ref .real [BM, BN] "p"))
        (Op.ref .real [BM] "l_prev")) s
      = some (Tile.bop NumericDType.real.add Broadcast.nil.consSame
          (Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [BM, BN].length) ptile) ltile) := by
  have hpr : evalOp (Op.ref .real [BM, BN] "p") s = some ptile := by rw [evalOp_ref, hp]
  have hsum : @evalOp TileDType.real [BM]
      (Op.reduceSum (⟨1, by simp⟩ : Fin [BM, BN].length) Bool.false (Op.ref .real [BM, BN] "p")) s
      = some (Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [BM, BN].length) ptile) := by
    erw [evalOp_reduceSum (⟨1, by simp⟩ : Fin [BM, BN].length) Bool.false
      (Op.ref .real [BM, BN] "p"), hpr]; rfl
  rw [evalOp_add]
  simp only [evalOp_ref, hl, hsum, Option.bind_eq_bind, Option.bind_some]; rfl

/-- **`l_rcp = 1.0 / l_curr` eval**. -/
theorem al_lrcp_eval (s : BlockState) (BM : Nat) (lc : Tile .real [BM])
    (hlc : s.regs .real [BM] "l_curr" = some lc) :
    evalOp (Op.div .real Broadcast.scalarL (Op.const (1.0 : ℝ)) (Op.ref .real [BM] "l_curr")) s
      = some (Tile.bop NumericDType.real.div Broadcast.scalarL
          (Tile.scalar (some (1.0 : ℝ) : WithBot ℝ)) lc) := by
  rw [evalOp_div]
  simp only [evalOp_const, evalOp_ref, hlc, Option.bind_eq_bind, Option.bind_some]

/-- **`p *= l_rcp[:, None]` eval**. -/
theorem al_p_rcp_eval (s : BlockState) (BM BN : Nat) (hax : 1 < [BM].length.succ)
    (ptile : Tile .real [BM, BN]) (lr : Tile .real [BM])
    (hp : s.regs .real [BM, BN] "p" = some ptile)
    (hlr : s.regs .real [BM] "l_rcp" = some lr) :
    evalOp (Op.mul .real Broadcast.nil.consR.consSame
        (Op.ref .real [BM, BN] "p") (Op.expandDim ⟨1, hax⟩ (Op.ref .real [BM] "l_rcp"))) s
      = some (Tile.bop NumericDType.real.mul Broadcast.nil.consR.consSame
          ptile (Tile.expandDim ⟨1, hax⟩ lr)) := by
  have hexp : @evalOp TileDType.real [BM, 1]
      (Op.expandDim ⟨1, hax⟩ (Op.ref .real [BM] "l_rcp")) s
      = some (Tile.expandDim ⟨1, hax⟩ lr) := evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hlr
  rw [evalOp_mul]
  simp only [evalOp_ref, hp, hexp, Option.bind_eq_bind, Option.bind_some]; rfl

/-- **`acc *= (l_prev * l_rcp)[:, None]` eval**. -/
theorem al_acc_rescale_eval (s : BlockState) (BM BD : Nat) (hax : 1 < [BM].length.succ)
    (acctile : Tile .real [BM, BD]) (lp lr : Tile .real [BM])
    (hacc : s.regs .real [BM, BD] "acc" = some acctile)
    (hlp : s.regs .real [BM] "l_prev" = some lp)
    (hlr : s.regs .real [BM] "l_rcp" = some lr) :
    evalOp (Op.mul .real Broadcast.nil.consR.consSame
        (Op.ref .real [BM, BD] "acc")
        (Op.expandDim ⟨1, hax⟩
          (Op.mul .real Broadcast.nil.consSame (Op.ref .real [BM] "l_prev") (Op.ref .real [BM] "l_rcp")))) s
      = some (Tile.bop NumericDType.real.mul Broadcast.nil.consR.consSame
          acctile (Tile.expandDim ⟨1, hax⟩
            (Tile.bop NumericDType.real.mul Broadcast.nil.consSame lp lr))) := by
  have hmul : @evalOp TileDType.real [BM]
      (Op.mul .real Broadcast.nil.consSame (Op.ref .real [BM] "l_prev") (Op.ref .real [BM] "l_rcp")) s
      = some (Tile.bop NumericDType.real.mul Broadcast.nil.consSame lp lr) := by
    rw [evalOp_mul]; simp only [evalOp_ref, hlp, hlr, Option.bind_eq_bind, Option.bind_some]
  have hexp : @evalOp TileDType.real [BM, 1]
      (Op.expandDim ⟨1, hax⟩
        (Op.mul .real Broadcast.nil.consSame (Op.ref .real [BM] "l_prev") (Op.ref .real [BM] "l_rcp"))) s
      = some (Tile.expandDim ⟨1, hax⟩ (Tile.bop NumericDType.real.mul Broadcast.nil.consSame lp lr)) := by
    erw [evalOp_expandDim, hmul]; rfl
  rw [evalOp_mul]
  simp only [evalOp_ref, hacc, hexp, Option.bind_eq_bind, Option.bind_some]; rfl

/-- **`acc += tl.dot(p, v)` eval** (`p` stays `.real`: the
`p.to(Q.dtype.element_ty)` cast erased to a re-assign). -/
theorem al_acc_pv_eval (s : BlockState) (BM BN BD : Nat)
    (acctile : Tile .real [BM, BD]) (ptile : Tile .real [BM, BN]) (vtile : Tile .real [BN, BD])
    (hacc : s.regs .real [BM, BD] "acc" = some acctile)
    (hp : s.regs .real [BM, BN] "p" = some ptile)
    (hv : s.regs .real [BN, BD] "v" = some vtile) :
    evalOp (Op.add .real Broadcast.nil.consSame.consSame
        (Op.ref .real [BM, BD] "acc")
        (Op.dot (batch := []) (Op.ref .real [BM, BN] "p") (Op.ref .real [BN, BD] "v"))) s
      = some (Tile.bop NumericDType.real.add Broadcast.nil.consSame.consSame
          acctile (Tile.dot [] ptile vtile)) := by
  have hpr : evalOp (Op.ref .real [BM, BN] "p") s = some ptile := by rw [evalOp_ref, hp]
  have hvr : evalOp (Op.ref .real [BN, BD] "v") s = some vtile := by rw [evalOp_ref, hv]
  have hdot : @evalOp TileDType.real [BM, BD]
      (Op.dot (batch := []) (Op.ref .real [BM, BN] "p") (Op.ref .real [BN, BD] "v")) s
      = some (Tile.dot [] ptile vtile) := by
    erw [evalOp_dot [] (Op.ref .real [BM, BN] "p") (Op.ref .real [BN, BD] "v"), hpr, hvr]; rfl
  rw [evalOp_add]
  simp only [evalOp_ref, hacc, hdot, Option.bind_eq_bind, Option.bind_some]; rfl

/-- **Pointer advance eval** (`k_ptrs += BLOCK_N * stride`): every lane's
offset gains `BN·srow`, i.e. `alPtrTile … rowOff → alPtrTile … (rowOff + BN)`. -/
theorem al_ptr_advance_eval (s : BlockState) (R : RegionName) (name : RegName)
    (base srow scol rowOff BN A B : Nat)
    (hp : s.regs .ptr [A, B] name = some (alPtrTile R base srow scol rowOff A B)) :
    evalOp (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [A, B] name)
        (Op.mul .nat Broadcast.nil (Op.constNat BN) (Op.constNat srow))) s
      = some (alPtrTile R base srow scol (rowOff + BN) A B) := by
  rw [evalOp_ptrAdd, evalOp_mul]
  simp only [evalOp_ref, hp, evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some (Tile.ext fun idx => ?_)
  show (R, base + (rowOff + idx.1.val) * srow + idx.2.1.val * scol + BN * srow)
    = (R, base + (rowOff + BN + idx.1.val) * srow + idx.2.1.val * scol)
  refine congrArg _ ?_
  ring

/-! ## Loop-body prefix/tail step lemmas (shared between the twins)

`alLoopPrefixG` produces the scaled score tile `qk0` (the identity
`offs_n < N_CTX` `where` discharged under `BN ≤ N_CTX`); `alLoopTailG`
runs the streaming-softmax recurrence from an ARBITRARY `qk` register
value `qkT`, so the causal arm reuses it verbatim after its extra
`where`. -/

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
open VeriTile.Examples.AttentionForwardClosedForm in
/-- **Prefix steps.** The 6 shared statements step `sin` (loop-body entry,
`start_n = SN`) to a state whose `qk` holds the scaled dot
`sc·(0 + q·kᵀ)` of the loaded (all-in-range, by `hin`) K block, with
`block_n_offs = SN + arange` set and every other register framed. -/
theorem al_loopPrefix_steps (sc : ℝ) (N_CTX BM BN BD SN : Nat) (sin : BlockState)
    (K : RegionName) (kBase skn skk : Nat)
    (qtile : Tile .real [BM, BD]) (kTfn : TileIndex [BN, BD] → ℝ)
    (hBNle : BN ≤ N_CTX)
    (hin : ∀ j : Fin BN, SN + j.val < N_CTX)
    (hsn : sin.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hoffn : sin.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => j.val)))
    (hq : sin.regs .real [BM, BD] "q" = some qtile)
    (hkp : sin.regs .ptr [BN, BD] "k_ptrs" = some (alPtrTile K kBase skn skk SN BN BD))
    (hkload : ∀ idx : TileIndex [BN, BD],
        kTfn idx = sin.readMem K (kBase + (SN + idx.1.val) * skn + idx.2.1.val * skk))
    (hundef : ∀ rg o, sin.undef rg o = 0) :
    ∃ sPre, stepStmts (alLoopPrefixG sc N_CTX BM BN BD) sin = some sPre
      ∧ sPre.pids = sin.pids ∧ sPre.mem = sin.mem ∧ (∀ rg o, sPre.undef rg o = 0)
      ∧ sPre.regs .real [BM, BN] "qk" = some (Tile.bop NumericDType.real.mul Broadcast.scalarR
          (Tile.bop NumericDType.real.add Broadcast.nil.consSame.consSame
            (⟨fun _ : TileIndex [BM, BN] => some (0 : ℝ)⟩ : Tile .real [BM, BN])
            (Tile.dot [] qtile (Tile.transpose []
              (⟨fun idx => some (kTfn idx)⟩ : Tile .real [BN, BD]))))
          (Tile.scalar (some sc : WithBot ℝ)))
      ∧ sPre.regs .nat [BN] "block_n_offs" = some (Tile.vec (fun j : Fin BN => SN + j.val))
      ∧ sPre.regs .real [BM] "m_prev" = sin.regs .real [BM] "m_prev"
      ∧ sPre.regs .real [BM] "l_prev" = sin.regs .real [BM] "l_prev"
      ∧ sPre.regs .real [BM, BD] "acc" = sin.regs .real [BM, BD] "acc"
      ∧ sPre.regs .real [BM, BD] "q" = sin.regs .real [BM, BD] "q"
      ∧ sPre.regs .nat [BM] "offs_m" = sin.regs .nat [BM] "offs_m"
      ∧ sPre.regs .nat [BN] "offs_n" = sin.regs .nat [BN] "offs_n"
      ∧ sPre.regs .nat [] "batch_id" = sin.regs .nat [] "batch_id"
      ∧ sPre.regs .nat [] "off_hz" = sin.regs .nat [] "off_hz"
      ∧ sPre.regs .ptr [BN, BD] "k_ptrs" = sin.regs .ptr [BN, BD] "k_ptrs"
      ∧ sPre.regs .ptr [BN, BD] "v_ptrs" = sin.regs .ptr [BN, BD] "v_ptrs" := by
  set kT : Tile .real [BN, BD] := ⟨fun idx => some (kTfn idx)⟩ with hkT
  set qkdot : Tile .real [BM, BN] :=
    Tile.bop NumericDType.real.add Broadcast.nil.consSame.consSame
      (⟨fun _ : TileIndex [BM, BN] => some (0 : ℝ)⟩ : Tile .real [BM, BN])
      (Tile.dot [] qtile (Tile.transpose [] kT)) with hqkdot
  unfold alLoopPrefixG
  -- P1: block_n_offs = start_n + offs_n
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
        (Op.ref .nat [BN] "offs_n")) sin
        = some (Tile.vec (fun j : Fin BN => SN + j.val)) from by
      rw [evalOp_add]
      simp only [evalOp_ref, hsn, hoffn, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext j
      simp only [Tile.bop_data, Tile.scalar_data, Tile.vec_data, castTile_self,
        Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add]))]
  -- P2: k = masked load (all lanes in range under `hin`)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.load .real (MemAccess.ptr (Op.ref .ptr [BN, BD] "k_ptrs"))
        (MaskOpt.mask
          (Op.remap [BN, BD] Broadcast.nil.consL.consSame.leftIndex
            (Op.lt ComparableDType.nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BN] "block_n_offs")) (Op.constNat N_CTX))))) _
        = some kT from by
      rw [load_ptr_mask_real (Op.ref .ptr [BN, BD] "k_ptrs") _ _
        (alPtrTile K kBase skn skk SN BN BD)
        (⟨fun idx : TileIndex [BN, BD] => decide (SN + idx.1.val < N_CTX)⟩ : Tile .bool [BN, BD])
        (by rw [evalOp_ref]; simp [hkp])
        (al_colmask_eval _ SN N_CTX (by simp))
        (by intro rg o; simp [hundef])]
      refine congrArg some ?_; ext idx
      have hi : SN + idx.1.val < N_CTX := hin idx.1
      simp only [hi, decide_true, if_true, hkT, hkload idx, BlockState.setReg_readMem]
      rfl))]
  -- P3: qk = zeros
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (al_qkzeros_eval _ BM BN))]
  -- P4: qk += dot(q, trans k)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (al_qk_dot_eval _ BM BN BD (⟨fun _ => some (0:ℝ)⟩) qtile kT
      (by simp) (by simp [hq]) (by simp [hkT])))]
  -- P5: identity where (offs_n < N_CTX)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (al_wheren_id_eval _ BM BN N_CTX hBNle qkdot (by simp [hoffn]) (by simp [hqkdot])))]
  -- P6: qk *= sc
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (al_qk_scale_eval _ BM BN sc qkdot (by simp [hqkdot])))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp
  · funext region offset; simp
  · intro rg o; simp [hundef]
  · simp [hqkdot, hkT]
  · simp [BlockState.setReg_ne_name]
  · simp [BlockState.setReg_ne_name]
  · simp [BlockState.setReg_ne_name]
  · simp [BlockState.setReg_ne_name]
  · simp [BlockState.setReg_ne_name]
  · simp [BlockState.setReg_ne_name]
  · simp [BlockState.setReg_ne_name]
  · simp [BlockState.setReg_ne_name]
  · simp [BlockState.setReg_ne_name]
  · simp [BlockState.setReg_ne_name]
  · simp [BlockState.setReg_ne_name]

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
open VeriTile.Examples.AttentionForwardClosedForm in
/-- **Tail steps.** The 14 shared statements run the natural-exp streaming
recurrence with in-loop `l_rcp` normalization from an ARBITRARY `qk`
register value `qkT`, load the (all-in-range) V block, and advance both
pointers by `BLOCK_N` rows. The `∃`-package exposes the recurrence tiles
exactly as the invariant-step lemmas consume them. -/
theorem al_loopTail_steps (skn svk N_CTX BM BN BD SN : Nat) (sin : BlockState)
    (hBN : 0 < BN)
    (qkT : Tile .real [BM, BN]) (mtile ltile : Tile .real [BM])
    (acctile : Tile .real [BM, BD])
    (K V : RegionName) (kBase vBase skk svn : Nat)
    (vTfn : TileIndex [BN, BD] → ℝ)
    (hin : ∀ j : Fin BN, SN + j.val < N_CTX)
    (hqk : sin.regs .real [BM, BN] "qk" = some qkT)
    (hbn : sin.regs .nat [BN] "block_n_offs" = some (Tile.vec (fun j : Fin BN => SN + j.val)))
    (hmp : sin.regs .real [BM] "m_prev" = some mtile)
    (hlp : sin.regs .real [BM] "l_prev" = some ltile)
    (hacc : sin.regs .real [BM, BD] "acc" = some acctile)
    (hkp : sin.regs .ptr [BN, BD] "k_ptrs" = some (alPtrTile K kBase skn skk SN BN BD))
    (hvp : sin.regs .ptr [BN, BD] "v_ptrs" = some (alPtrTile V vBase svk svn SN BN BD))
    (hvload : ∀ idx : TileIndex [BN, BD],
        vTfn idx = sin.readMem V (vBase + (SN + idx.1.val) * svk + idx.2.1.val * svn))
    (hundef : ∀ rg o, sin.undef rg o = 0) :
    ∃ rmaxT : Tile .real [BM],
      Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [BM, BN].length) qkT = some rmaxT ∧
      ∃ sF, stepStmts (alLoopTailG skn svk N_CTX BM BN BD) sin = some sF
        ∧ sF.pids = sin.pids ∧ sF.mem = sin.mem ∧ (∀ rg o, sF.undef rg o = 0)
        ∧ ∃ (mcurrT lcurrT lrcpT : Tile .real [BM]) (pT : Tile .real [BM, BN])
              (acc1T : Tile .real [BM, BD]),
            mcurrT = Tile.select
                (Tile.cop ComparableDType.real.gt Broadcast.nil.consSame rmaxT mtile) rmaxT mtile
            ∧ pT = Tile.bop NumericDType.real.mul Broadcast.nil.consR.consSame
                (Tile.uop WithBot.realExp
                  (Tile.bop NumericDType.real.sub Broadcast.nil.consR.consSame
                    qkT (Tile.expandDim ⟨1, by simp⟩ mcurrT)))
                (Tile.expandDim ⟨1, by simp⟩ lrcpT)
            ∧ lcurrT = Tile.bop NumericDType.real.add Broadcast.nil.consSame
                (Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [BM, BN].length)
                  (Tile.uop WithBot.realExp
                    (Tile.bop NumericDType.real.sub Broadcast.nil.consR.consSame
                      qkT (Tile.expandDim ⟨1, by simp⟩ mcurrT))))
                (Tile.bop NumericDType.real.mul Broadcast.nil.consSame ltile
                  (Tile.uop WithBot.realExp
                    (Tile.bop NumericDType.real.sub Broadcast.nil.consSame mtile mcurrT)))
            ∧ lrcpT = Tile.bop NumericDType.real.div Broadcast.scalarL
                (Tile.scalar (some (1.0 : ℝ) : WithBot ℝ)) lcurrT
            ∧ acc1T = Tile.bop NumericDType.real.mul Broadcast.nil.consR.consSame acctile
                (Tile.expandDim ⟨1, by simp⟩
                  (Tile.bop NumericDType.real.mul Broadcast.nil.consSame
                    (Tile.bop NumericDType.real.mul Broadcast.nil.consSame ltile
                      (Tile.uop WithBot.realExp
                        (Tile.bop NumericDType.real.sub Broadcast.nil.consSame mtile mcurrT)))
                    lrcpT))
            ∧ sF.regs .real [BM] "m_prev" = some mcurrT
            ∧ sF.regs .real [BM] "l_prev" = some lcurrT
            ∧ sF.regs .real [BM, BD] "acc" = some
                (Tile.bop NumericDType.real.add Broadcast.nil.consSame.consSame
                  acc1T (Tile.dot [] pT (⟨fun idx => some (vTfn idx)⟩ : Tile .real [BN, BD])))
            ∧ sF.regs .ptr [BN, BD] "k_ptrs" = some (alPtrTile K kBase skn skk (SN + BN) BN BD)
            ∧ sF.regs .ptr [BN, BD] "v_ptrs" = some (alPtrTile V vBase svk svn (SN + BN) BN BD)
            ∧ sF.regs .real [BM, BD] "q" = sin.regs .real [BM, BD] "q"
            ∧ sF.regs .nat [BM] "offs_m" = sin.regs .nat [BM] "offs_m"
            ∧ sF.regs .nat [BN] "offs_n" = sin.regs .nat [BN] "offs_n"
            ∧ sF.regs .nat [] "batch_id" = sin.regs .nat [] "batch_id"
            ∧ sF.regs .nat [] "off_hz" = sin.regs .nat [] "off_hz" := by
  set vT : Tile .real [BN, BD] := ⟨fun idx => some (vTfn idx)⟩ with hvT
  obtain ⟨rmaxT, hrm⟩ : ∃ t, Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [BM, BN].length) qkT = some t :=
    ⟨_, by
      unfold Tile.reduceMaxDrop
      rw [dif_pos (show 0 < TileShape.axisDim [BM, BN] (⟨1, by simp⟩ : Fin [BM, BN].length) from hBN)]⟩
  set mcurrT : Tile .real [BM] := Tile.select
      (Tile.cop ComparableDType.real.gt Broadcast.nil.consSame rmaxT mtile) rmaxT mtile
    with hmc
  set alphaT : Tile .real [BM] := Tile.uop WithBot.realExp
      (Tile.bop NumericDType.real.sub Broadcast.nil.consSame mtile mcurrT) with hal
  set lprev1T : Tile .real [BM] := Tile.bop NumericDType.real.mul Broadcast.nil.consSame
      ltile alphaT with hlp1
  set pexpT : Tile .real [BM, BN] := Tile.uop WithBot.realExp
      (Tile.bop NumericDType.real.sub Broadcast.nil.consR.consSame
        qkT (Tile.expandDim ⟨1, by simp⟩ mcurrT)) with hpexp
  set lcurrT : Tile .real [BM] := Tile.bop NumericDType.real.add Broadcast.nil.consSame
      (Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [BM, BN].length) pexpT) lprev1T with hlc
  set lrcpT : Tile .real [BM] := Tile.bop NumericDType.real.div Broadcast.scalarL
      (Tile.scalar (some (1.0 : ℝ) : WithBot ℝ)) lcurrT with hlr
  set pT : Tile .real [BM, BN] := Tile.bop NumericDType.real.mul Broadcast.nil.consR.consSame
      pexpT (Tile.expandDim ⟨1, by simp⟩ lrcpT) with hpT
  set acc1T : Tile .real [BM, BD] := Tile.bop NumericDType.real.mul Broadcast.nil.consR.consSame
      acctile (Tile.expandDim ⟨1, by simp⟩
        (Tile.bop NumericDType.real.mul Broadcast.nil.consSame lprev1T lrcpT)) with hacc1
  refine ⟨rmaxT, hrm, ?_⟩
  unfold alLoopTailG
  -- T1: m_curr = maximum(max(qk,1), m_prev)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (al_mij_eval _ BM BN mtile qkT rmaxT hmp hqk hrm))]
  -- T2: l_prev *= exp(m_prev - m_curr)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (al_alpha_eval _ BM ltile mtile mcurrT (by simp [hlp]) (by simp [hmp]) (by simp [hmc])))]
  -- T3: p = exp(qk - m_curr[:,None])
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (al_p_eval _ BM BN Nat.one_lt_two qkT mcurrT (by simp [hqk]) (by simp [hmc])))]
  -- T4: l_curr = sum(p,1) + l_prev
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (al_lij_eval _ BM BN pexpT lprev1T (by simp [hpexp]) (by simp [hlp1, hal])))]
  -- T5: l_rcp = 1/l_curr
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (al_lrcp_eval _ BM lcurrT (by simp [hlc])))]
  -- T6: p *= l_rcp[:,None]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (al_p_rcp_eval _ BM BN Nat.one_lt_two pexpT lrcpT (by simp [hpexp]) (by simp [hlr])))]
  -- T7: acc *= (l_prev * l_rcp)[:,None]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (al_acc_rescale_eval _ BM BD Nat.one_lt_two acctile lprev1T lrcpT
      (by simp [hacc]) (by simp [hlp1, hal]) (by simp [hlr])))]
  -- T8: p = p (the erased element_ty cast)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ref .real [BM, BN] "p") _ = some pT from by
      rw [evalOp_ref]; simp [hpT]))]
  -- T9: v = masked load (all lanes in range under `hin`)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.load .real (MemAccess.ptr (Op.ref .ptr [BN, BD] "v_ptrs"))
        (MaskOpt.mask
          (Op.remap [BN, BD] Broadcast.nil.consL.consSame.leftIndex
            (Op.lt ComparableDType.nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BN] "block_n_offs")) (Op.constNat N_CTX))))) _
        = some vT from by
      rw [load_ptr_mask_real (Op.ref .ptr [BN, BD] "v_ptrs") _ _
        (alPtrTile V vBase svk svn SN BN BD)
        (⟨fun idx : TileIndex [BN, BD] => decide (SN + idx.1.val < N_CTX)⟩ : Tile .bool [BN, BD])
        (by rw [evalOp_ref]; simp [hvp])
        (al_colmask_eval _ SN N_CTX (by simp [hbn]))
        (by intro rg o; simp [hundef])]
      refine congrArg some ?_; ext idx
      have hi : SN + idx.1.val < N_CTX := hin idx.1
      simp only [hi, decide_true, if_true, hvT, hvload idx, BlockState.setReg_readMem]
      rfl))]
  -- T10: acc += dot(p, v)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (al_acc_pv_eval _ BM BN BD acc1T pT vT (by simp [hacc1]) (by simp [hpT]) (by simp [hvT])))]
  -- T11: l_prev = l_curr
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ref .real [BM] "l_curr") _ = some lcurrT from by
      rw [evalOp_ref]; simp [hlc]))]
  -- T12: m_prev = m_curr
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ref .real [BM] "m_curr") _ = some mcurrT from by
      rw [evalOp_ref]; simp [hmc]))]
  -- T13: k_ptrs += BN * skn
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (al_ptr_advance_eval _ K "k_ptrs" kBase skn skk SN BN BN BD (by simp [hkp])))]
  -- T14: v_ptrs += BN * svk
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (al_ptr_advance_eval _ V "v_ptrs" vBase svk svn SN BN BN BD (by simp [hvp])))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, mcurrT, lcurrT, lrcpT, pT, acc1T, rfl, rfl, rfl, rfl, rfl,
    ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp
  · funext region offset; simp
  · intro rg o; simp [hundef]
  · simp [hmc]
  · simp [hlc]
  · simp [hacc1, hpT, hvT]
  · simp
  · simp
  · simp [BlockState.setReg_ne_name]
  · simp [BlockState.setReg_ne_name]
  · simp [BlockState.setReg_ne_name]
  · simp [BlockState.setReg_ne_name]
  · simp [BlockState.setReg_ne_name]

/-! ## Streaming-math bridge cells (shared row-max; non-causal SA layer)

The non-causal arm binds the running registers to the library
`StreamingAccumulator` (`SA`) accumulators (every score lane is a real
`some` under `N_CTX = BLOCK_N·numKVBlocks`); the causal arm (below) binds
to `FA1MathCausal`. -/

/-- Row max via `reduceMaxDrop`: cell `r` is the `Finset.sup` of row `r`
(valid for arbitrary `WithBot`-valued rows; both arms reuse it). -/
theorem al_rmax_cell {BM BN : Nat} (hBN : 0 < BN)
    (qkT : Tile .real [BM, BN]) (rmaxT : Tile .real [BM])
    (hrm : Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [BM, BN].length) qkT = some rmaxT)
    (r : Fin BM) :
    rmaxT.data (r, PUnit.unit)
      = (Finset.univ : Finset (Fin BN)).sup (fun j => qkT.data (r, j, PUnit.unit)) := by
  unfold Tile.reduceMaxDrop at hrm
  rw [dif_pos (show 0 < TileShape.axisDim [BM, BN] (⟨1, by simp⟩ : Fin [BM, BN].length) from hBN)] at hrm
  rw [← Option.some.inj hrm]
  show (Finset.univ : Finset (Fin BN)).sup' ⟨⟨0, hBN⟩, Finset.mem_univ _⟩
      (fun k => qkT.data ((r, k, PUnit.unit) : TileIndex [BM, BN])) = _
  rw [Finset.sup'_eq_sup]

open StreamingAccumulator in
/-- Non-causal scaled-score cell: the prefix's `qk0` lane `(r, j)` is the
real `StreamingAccumulator.scaledScore` at the block-`c` global key index. -/
theorem al_qk0_cell (BM BN BD numKV : Nat)
    (qT : TileIndex [BM, BD] → ℝ) (kT : TileIndex [BN * numKV, BD] → ℝ) (sc : ℝ)
    (c : Nat) (hc : c < numKV) (r : Fin BM) (j : Fin BN) :
    (Tile.bop NumericDType.real.mul Broadcast.scalarR
      (Tile.bop NumericDType.real.add Broadcast.nil.consSame.consSame
        (⟨fun _ : TileIndex [BM, BN] => some (0 : ℝ)⟩ : Tile .real [BM, BN])
        (Tile.dot [] (⟨fun idx => some (qT idx)⟩ : Tile .real [BM, BD])
          (Tile.transpose [] (⟨fun idx : TileIndex [BN, BD] =>
            some (kT (blockIndex BN numKV c (Nat.succ_le_iff.mpr hc) idx.1, idx.2.1, PUnit.unit))⟩
              : Tile .real [BN, BD]))))
      (Tile.scalar (some sc : WithBot ℝ))).data (r, j, PUnit.unit)
      = some (StreamingAccumulator.scaledScore qT kT sc r (blockIndex BN numKV c (Nat.succ_le_iff.mpr hc) j)) := by
  rw [Tile.bop_data, Tile.bop_data]
  simp only [Broadcast.leftIndex, Broadcast.rightIndex, Broadcast.scalarR, Tile.scalar_data,
    NumericDType.mul, NumericDType.add]
  rw [show (Tile.dot [] (⟨fun idx => some (qT idx)⟩ : Tile .real [BM, BD])
        (Tile.transpose [] (⟨fun idx : TileIndex [BN, BD] =>
          some (kT (blockIndex BN numKV c (Nat.succ_le_iff.mpr hc) idx.1, idx.2.1, PUnit.unit))⟩
            : Tile .real [BN, BD]))).data (r, j, PUnit.unit)
      = some (Finset.univ.sum (fun e : Fin BD => qT (r, e, PUnit.unit)
          * kT (blockIndex BN numKV c (Nat.succ_le_iff.mpr hc) j, e, PUnit.unit))) from by
    rw [Tile.dot_nil_data]
    rw [show (@Finset.sum (Fin BD) (WithBot ℝ) _ Finset.univ
          (fun e => Option.map₂ (· * ·)
            ((⟨fun idx => some (qT idx)⟩ : Tile .real [BM, BD]).data (r, e, PUnit.unit))
            ((Tile.transpose [] (⟨fun idx : TileIndex [BN, BD] =>
              some (kT (blockIndex BN numKV c (Nat.succ_le_iff.mpr hc) idx.1, idx.2.1, PUnit.unit))⟩
                : Tile .real [BN, BD])).data (e, j, PUnit.unit))))
        = @Finset.sum (Fin BD) (WithBot ℝ) _ Finset.univ
            (fun e => (some (qT (r, e, PUnit.unit)
              * kT (blockIndex BN numKV c (Nat.succ_le_iff.mpr hc) j, e, PUnit.unit)) : WithBot ℝ))
        from Finset.sum_congr rfl (fun e _ => by
          rw [Tile.transpose_nil_data]; rfl)]
    rw [WithBot.sum_someTerm_eq_some]]
  simp only [WithBot.realAdd, WithBot.realMul, Option.map₂, Option.bind, Option.map]
  refine congrArg some ?_
  simp only [StreamingAccumulator.scaledScore]
  ring

/-- Real row-sum of `exp(qk − mNew)` when every `qk` lane is a real
`some (g r j)` and `mNew` is a real `some (μ r)` (non-causal `l_curr`). -/
theorem al_pexp_rowsum_real {BM BN : Nat}
    (qkT : Tile .real [BM, BN]) (mNewT : Tile .real [BM])
    (g : Fin BM → Fin BN → ℝ) (μ : Fin BM → ℝ)
    (hg : ∀ r j, qkT.data (r, j, PUnit.unit) = some (g r j))
    (hμ : ∀ r, mNewT.data (r, PUnit.unit) = some (μ r)) (r : Fin BM) :
    (Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [BM, BN].length)
        (Tile.uop WithBot.realExp
          (Tile.bop NumericDType.real.sub Broadcast.nil.consR.consSame
            qkT (Tile.expandDim ⟨1, by simp⟩ mNewT)))).data (r, PUnit.unit)
      = some ((Finset.univ : Finset (Fin BN)).sum (fun j => Real.exp (g r j - μ r))) := by
  rw [Tile.reduceSumDrop_data]
  have hcell : ∀ k : Fin (TileShape.axisDim [BM, BN] (⟨1, by simp⟩ : Fin [BM, BN].length)),
      (Tile.uop WithBot.realExp
        (Tile.bop NumericDType.real.sub Broadcast.nil.consR.consSame
          qkT (Tile.expandDim ⟨1, by simp⟩ mNewT))).data
        (TileShape.insertAxisIndex [BM, BN] (⟨1, by simp⟩ : Fin [BM, BN].length) (r, PUnit.unit) k)
      = (some (Real.exp (g r k - μ r)) : WithBot ℝ) := by
    intro k
    rw [show (TileShape.insertAxisIndex [BM, BN] (⟨1, by simp⟩ : Fin [BM, BN].length)
          (r, PUnit.unit) k) = (r, k, PUnit.unit) from rfl]
    rw [Tile.uop_data, Tile.bop_data]
    simp only [Broadcast.leftIndex, Broadcast.rightIndex, Tile.expandDim_data,
      TileShape.dropInsertedIndex, NumericDType.sub]
    rw [hg r k, hμ r]
    rfl
  rw [Finset.sum_congr rfl (fun k _ => hcell k)]
  rw [WithBot.sum_someTerm_eq_some]
  rfl

/-- Real `p·v` dot cell (non-causal): with all-real `qk`/`mNew`/`l_rcp`/`v`,
lane `(r, d)` of `(exp(qk−mNew)·l_rcp[:,None]) · v` is
`l_rcp_r · Σ_j exp(g r j − μ r) · v (j, d)`. -/
theorem al_pv_dot_real {BM BN BD : Nat}
    (qkT : Tile .real [BM, BN]) (mNewT lrcpT : Tile .real [BM]) (vT : Tile .real [BN, BD])
    (g : Fin BM → Fin BN → ℝ) (μ : Fin BM → ℝ) (vfn : TileIndex [BN, BD] → ℝ)
    (hg : ∀ r j, qkT.data (r, j, PUnit.unit) = some (g r j))
    (hμ : ∀ r, mNewT.data (r, PUnit.unit) = some (μ r))
    (hv : ∀ idx, vT.data idx = some (vfn idx))
    (r : Fin BM) (d : Fin BD) (lv : ℝ)
    (hlr : lrcpT.data (r, PUnit.unit) = some lv) :
    (Tile.dot []
        (Tile.bop NumericDType.real.mul Broadcast.nil.consR.consSame
          (Tile.uop WithBot.realExp
            (Tile.bop NumericDType.real.sub Broadcast.nil.consR.consSame
              qkT (Tile.expandDim ⟨1, by simp⟩ mNewT)))
          (Tile.expandDim ⟨1, by simp⟩ lrcpT))
        vT).data (r, d, PUnit.unit)
      = some (lv * (Finset.univ : Finset (Fin BN)).sum (fun j =>
          Real.exp (g r j - μ r) * vfn (j, d, PUnit.unit))) := by
  rw [Tile.dot_nil_data]
  refine Eq.trans (b := @Finset.sum (Fin BN) (WithBot ℝ) _ Finset.univ
      (fun j => (some (lv * (Real.exp (g r j - μ r) * vfn (j, d, PUnit.unit))) : WithBot ℝ)))
    (Finset.sum_congr rfl (fun j (_ : j ∈ Finset.univ) => ?_)) ?_
  · rw [Tile.bop_data]
    simp only [Broadcast.leftIndex, Broadcast.rightIndex, Tile.expandDim_data,
      TileShape.dropInsertedIndex, NumericDType.mul]
    rw [Tile.uop_data, Tile.bop_data]
    simp only [Broadcast.leftIndex, Broadcast.rightIndex, Tile.expandDim_data,
      TileShape.dropInsertedIndex, NumericDType.sub]
    rw [hg r j, hμ r, hlr, hv (j, d, PUnit.unit)]
    show Option.map₂ (· * ·) (WithBot.realMul (WithBot.realExp
        (WithBot.realSub (some (g r j)) (some (μ r)))) (some lv)) (some (vfn (j, d, PUnit.unit)))
      = some (lv * (Real.exp (g r j - μ r) * vfn (j, d, PUnit.unit)))
    simp only [WithBot.realSub, WithBot.realExp, WithBot.realMul, Option.map₂_some_some,
      Option.map₂, Option.bind, Option.map]
    refine congrArg some ?_
    ring
  · rw [WithBot.sum_someTerm_eq_some, ← Finset.mul_sum]

open StreamingAccumulator in
/-- `lFree` is non-negative at every block count (sums of exponentials). -/
theorem al_lFree_nonneg {BM BD BN numKV : Nat}
    (qT : TileIndex [BM, BD] → ℝ) (kT : TileIndex [BN * numKV, BD] → ℝ) (sc : ℝ)
    (k : Nat) (hk : k ≤ numKV) (r : Fin BM) :
    0 ≤ lFree qT kT sc k hk r := by
  unfold lFree
  exact Finset.sum_nonneg (fun n _ => Finset.sum_nonneg (fun jL _ =>
    le_of_lt (Real.exp_pos _)))

open StreamingAccumulator in
/-- `lFree` is strictly positive once at least one block has been seen. -/
theorem al_lFree_pos {BM BD BN numKV : Nat} (hBN : 0 < BN)
    (qT : TileIndex [BM, BD] → ℝ) (kT : TileIndex [BN * numKV, BD] → ℝ) (sc : ℝ)
    (k : Nat) (hk : k + 1 ≤ numKV) (r : Fin BM) :
    0 < lFree qT kT sc (k + 1) hk r := by
  rw [lFree_succ qT kT sc k hk r]
  have hblock : 0 < Finset.univ.sum (fun jL : Fin BN =>
      Real.exp (StreamingAccumulator.scaledScore qT kT sc r (blockIndex BN numKV k hk jL))) := by
    refine Finset.sum_pos (fun jL _ => Real.exp_pos _) ?_
    exact ⟨⟨0, hBN⟩, Finset.mem_univ _⟩
  have hprev := al_lFree_nonneg qT kT sc k (Nat.le_of_succ_le hk) r
  linarith

open StreamingAccumulator in
/-- `lPartial` is non-zero once at least one block has been seen. -/
theorem al_lPartial_succ_ne_zero {BM BD BN numKV : Nat} (hBN : 0 < BN)
    (qT : TileIndex [BM, BD] → ℝ) (kT : TileIndex [BN * numKV, BD] → ℝ) (sc : ℝ)
    (k : Nat) (hk : k + 1 ≤ numKV) (r : Fin BM) :
    lPartial qT numKV kT sc (k + 1) r ≠ 0 := by
  rw [lPartial_eq_mShifted hBN qT numKV kT sc (k + 1) hk r]
  exact mul_ne_zero (Real.exp_ne_zero _) (ne_of_gt (al_lFree_pos hBN qT kT sc k hk r))

open StreamingAccumulator in
/-- **Acc-rescale cancel (non-causal).**
`(oPartial c / lPartial c) · lPartial c = oPartial c`, valid also at
`c = 0` (`0/0·0 = 0 = oPartial 0`). -/
theorem al_oPartial_div_lPartial_cancel {BM BD BN numKV : Nat} (hBN : 0 < BN)
    (qT : TileIndex [BM, BD] → ℝ) (kT vT : TileIndex [BN * numKV, BD] → ℝ) (sc : ℝ)
    (c : Nat) (hc : c ≤ numKV) (idx : TileIndex [BM, BD]) :
    (oPartial qT numKV kT vT sc c idx / lPartial qT numKV kT sc c idx.1)
      * lPartial qT numKV kT sc c idx.1
      = oPartial qT numKV kT vT sc c idx := by
  cases c with
  | zero =>
      show (0 / lPartial qT numKV kT sc 0 idx.1) * lPartial qT numKV kT sc 0 idx.1 = (0 : ℝ)
      rw [zero_div, zero_mul]
  | succ k =>
      exact div_mul_cancel₀ _ (al_lPartial_succ_ne_zero hBN qT kT sc k hc idx.1)

/-! ## Non-causal invariant + step -/

open StreamingAccumulator in
/-- **Non-causal loop invariant.** Counter `i = c·BLOCK_N` over `numKV`
blocks of `BLOCK_N` keys (`N_CTX = BLOCK_N·numKV`). The running registers
bind to the library `StreamingAccumulator` accumulators at block count
`c = i / BLOCK_N`; `k_ptrs`/`v_ptrs` sit at row offset `i`; `batch_id`/
`off_hz` scalars persist for the postLoop `Out` offset. -/
noncomputable def alInv (Q K V : RegionName) (s0 : BlockState) (sc : ℝ)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn
      N_HEAD H BM BN BD numKV : Nat) (i : Nat) (st : BlockState) : Prop :=
  let qT := alQTileG s0 Q N_HEAD sqz sqh sqm sqk H BM BD
  let kT := alKTileG s0 K N_HEAD skz skh skn skk (BN * numKV) BD
  let vT := alVTileG s0 V N_HEAD svz svh svk svn (BN * numKV) BD
  st.pids = s0.pids ∧ i % BN = 0 ∧ i ≤ BN * numKV ∧
  (st.regs .real [BM] "m_prev" = some ⟨fun r : TileIndex [BM] =>
      mPartial BN qT numKV kT sc (i / BN) r.1⟩) ∧
  (st.regs .real [BM] "l_prev" = some ⟨fun r : TileIndex [BM] =>
      ((lPartial qT numKV kT sc (i / BN) r.1 : ℝ) : WithBot ℝ)⟩) ∧
  (st.regs .real [BM, BD] "acc" = some ⟨fun idx : TileIndex [BM, BD] =>
      ((oPartial qT numKV kT vT sc (i / BN) idx
          / lPartial qT numKV kT sc (i / BN) idx.1 : ℝ) : WithBot ℝ)⟩) ∧
  (st.regs .real [BM, BD] "q" = some ⟨fun idx : TileIndex [BM, BD] => some (qT idx)⟩) ∧
  (st.regs .nat [BM] "offs_m" = some (Tile.vec (fun r : Fin BM => s0.pids 0 * BM + r.val))) ∧
  (st.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => j.val))) ∧
  (st.regs .nat [] "batch_id" = some (Tile.scalar (s0.pids 1 / N_HEAD))) ∧
  (st.regs .nat [] "off_hz" = some (Tile.scalar (s0.pids 1 % N_HEAD))) ∧
  (st.regs .ptr [BN, BD] "k_ptrs"
    = some (alPtrTile K (alBase s0 N_HEAD skz skh) skn skk i BN BD)) ∧
  (st.regs .ptr [BN, BD] "v_ptrs"
    = some (alPtrTile V (alBase s0 N_HEAD svz svh) svk svn i BN BD)) ∧
  (∀ rg o, st.undef rg o = 0) ∧ (st.mem = s0.mem)

open StreamingAccumulator in
/-- **Invariant base case** at `c = 0` (`mPartial 0 = ⊥`, `lPartial 0 = 0`,
`oPartial 0 / lPartial 0 = 0/0 = 0`), from the preamble outputs. -/
theorem alInv_zero (Q K V : RegionName) (sp : BlockState) (sc : ℝ)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn
      N_HEAD H BM BN BD numKV : Nat)
    (hm : sp.regs .real [BM] "m_prev" = some ⟨fun _ : TileIndex [BM] => (⊥ : WithBot ℝ)⟩)
    (hl : sp.regs .real [BM] "l_prev" = some ⟨fun _ : TileIndex [BM] => some (0 : ℝ)⟩)
    (hacc : sp.regs .real [BM, BD] "acc" = some ⟨fun _ : TileIndex [BM, BD] => some (0 : ℝ)⟩)
    (hq : sp.regs .real [BM, BD] "q" = some ⟨fun idx : TileIndex [BM, BD] =>
        some (alQTileG sp Q N_HEAD sqz sqh sqm sqk H BM BD idx)⟩)
    (hoffm : sp.regs .nat [BM] "offs_m" = some (Tile.vec (fun r : Fin BM => sp.pids 0 * BM + r.val)))
    (hoffn : sp.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => j.val)))
    (hbi : sp.regs .nat [] "batch_id" = some (Tile.scalar (sp.pids 1 / N_HEAD)))
    (hoh : sp.regs .nat [] "off_hz" = some (Tile.scalar (sp.pids 1 % N_HEAD)))
    (hkp : sp.regs .ptr [BN, BD] "k_ptrs"
      = some (alPtrTile K (alBase sp N_HEAD skz skh) skn skk 0 BN BD))
    (hvp : sp.regs .ptr [BN, BD] "v_ptrs"
      = some (alPtrTile V (alBase sp N_HEAD svz svh) svk svn 0 BN BD))
    (hundef : ∀ rg o, sp.undef rg o = 0) :
    alInv Q K V sp sc sqz sqh sqm sqk skz skh skn skk svz svh svk svn
      N_HEAD H BM BN BD numKV 0 sp := by
  unfold alInv
  refine ⟨rfl, by simp, by simp, ?_, ?_, ?_, hq, hoffm, hoffn, hbi, hoh, hkp, hvp, hundef, rfl⟩
  · rw [hm]; refine congrArg some ?_; ext r
    show (⊥ : WithBot ℝ) = mPartial BN _ numKV _ sc (0 / BN) r.1
    rw [Nat.zero_div]; rfl
  · rw [hl]; refine congrArg some ?_; ext r
    show some (0 : ℝ) = ((lPartial _ numKV _ sc (0 / BN) r.1 : ℝ) : WithBot ℝ)
    rw [Nat.zero_div]; rfl
  · rw [hacc]; refine congrArg some ?_; ext idx
    show some (0 : ℝ) = ((oPartial (alQTileG sp Q N_HEAD sqz sqh sqm sqk H BM BD) numKV
        (alKTileG sp K N_HEAD skz skh skn skk (BN * numKV) BD)
        (alVTileG sp V N_HEAD svz svh svk svn (BN * numKV) BD) sc (0 / BN) idx
        / lPartial (alQTileG sp Q N_HEAD sqz sqh sqm sqk H BM BD) numKV
          (alKTileG sp K N_HEAD skz skh skn skk (BN * numKV) BD) sc (0 / BN) idx.1 : ℝ) : WithBot ℝ)
    rw [Nat.zero_div]
    show some (0 : ℝ) = (((0 : ℝ) / (0 : ℝ) : ℝ) : WithBot ℝ)
    rw [div_zero]
    rfl

set_option maxHeartbeats 8000000 in
set_option maxRecDepth 8000 in
open StreamingAccumulator in
/-- **Non-causal loop step.** Advancing the counter from `i = c·BLOCK_N` to
`i + BLOCK_N` carries the SA running registers from block count `c` to
`c+1` (in-loop `l_rcp` normalization made precise through the
`div`/`mul` cancel). -/
theorem al_attn_step (Q K V : RegionName) (s0 : BlockState) (sc : ℝ)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn
      N_HEAD H BM BN BD numKV : Nat)
    (_hBM : 0 < BM) (hBN : 0 < BN)
    (i : Nat) (st : BlockState) (hilt : i < BN * numKV)
    (hinv : alInv Q K V s0 sc sqz sqh sqm sqk skz skh skn skk svz svh svk svn
      N_HEAD H BM BN BD numKV i st) :
    ∃ s', stepStmts (alLoopBodyG sc skn svk (BN * numKV) BM BN BD)
        (st.setReg "start_n" .nat [] (Tile.scalar i)) = some s'
      ∧ alInv Q K V s0 sc sqz sqh sqm sqk skz skh skn skk svz svh svk svn
          N_HEAD H BM BN BD numKV (i + BN) s' := by
  simp only [alInv] at hinv
  obtain ⟨hpids, hmod, hile, hmp, hlp, hacc, hq, hoffm, hoffn, hbi, hoh, hkp, hvp, hundef, hmem⟩ := hinv
  set c := i / BN with hcdef
  have hic : i = c * BN := by
    rw [hcdef, Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hmod)]
  have hclt : c < numKV := by
    rw [hcdef]
    exact (Nat.div_lt_iff_lt_mul hBN).mpr (by rw [Nat.mul_comm numKV BN]; exact hilt)
  have hcle : c ≤ numKV := le_of_lt hclt
  have hc1 : (i + BN) / BN = c + 1 := by
    rw [hcdef, hic, Nat.add_div_right _ hBN, Nat.mul_div_cancel _ hBN]
  set qT := alQTileG s0 Q N_HEAD sqz sqh sqm sqk H BM BD with hqT
  set kT := alKTileG s0 K N_HEAD skz skh skn skk (BN * numKV) BD with hkT
  set vT := alVTileG s0 V N_HEAD svz svh svk svn (BN * numKV) BD with hvT
  set se := st.setReg "start_n" .nat [] (Tile.scalar i) with hse
  set kTfn : TileIndex [BN, BD] → ℝ :=
    (fun idx => kT (blockIndex BN numKV c (Nat.succ_le_iff.mpr hclt) idx.1, idx.2.1, PUnit.unit)) with hkTfn
  set vTfn : TileIndex [BN, BD] → ℝ :=
    (fun idx => vT (blockIndex BN numKV c (Nat.succ_le_iff.mpr hclt) idx.1, idx.2.1, PUnit.unit)) with hvTfn
  have hin : ∀ j : Fin BN, i + j.val < BN * numKV := by
    intro j
    have hjv : j.val < BN := j.isLt
    calc i + j.val < i + BN := by omega
      _ = (c + 1) * BN := by rw [hic]; ring
      _ ≤ numKV * BN := Nat.mul_le_mul_right _ hclt
      _ = BN * numKV := Nat.mul_comm _ _
  have hmem_se : ∀ (R : RegionName) (o : Nat), se.readMem R o = s0.readMem R o := by
    intro R o; rw [hse]; simp only [BlockState.setReg_readMem]
    unfold BlockState.readMem; rw [hmem]
  have hkload : ∀ idx : TileIndex [BN, BD],
      kTfn idx = se.readMem K (alBase s0 N_HEAD skz skh + (i + idx.1.val) * skn + idx.2.1.val * skk) := by
    intro idx; rw [hmem_se]
    simp only [hkTfn, hkT, alKTileG, hic, blockIndex]
  have hvload : ∀ idx : TileIndex [BN, BD],
      vTfn idx = se.readMem V (alBase s0 N_HEAD svz svh + (i + idx.1.val) * svk + idx.2.1.val * svn) := by
    intro idx; rw [hmem_se]
    simp only [hvTfn, hvT, alVTileG, hic, blockIndex]
  -- run the shared prefix
  obtain ⟨sPre, hpre, hprepids, hpremem, hpreundef, hpreqk, hprebn, hpremp, hprelp, hpreacc,
      hpreq, hpreoffm, hpreoffn, hprebi, hpreoh, hprekp, hprevp⟩ :=
    al_loopPrefix_steps sc (BN * numKV) BM BN BD i se K (alBase s0 N_HEAD skz skh) skn skk
      (⟨fun idx => some (qT idx)⟩) kTfn
      (Nat.le_mul_of_pos_right BN (lt_of_le_of_lt (Nat.zero_le c) hclt))
      hin
      (by rw [hse, BlockState.setReg_same])
      (by rw [hse, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hoffn)
      (by rw [hse, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hq)
      (by rw [hse, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hkp)
      hkload
      (by intro rg o; rw [hse, BlockState.setReg_undef]; exact hundef rg o)
  set qk0 : Tile .real [BM, BN] := Tile.bop NumericDType.real.mul Broadcast.scalarR
      (Tile.bop NumericDType.real.add Broadcast.nil.consSame.consSame
        (⟨fun _ : TileIndex [BM, BN] => some (0 : ℝ)⟩ : Tile .real [BM, BN])
        (Tile.dot [] (⟨fun idx => some (qT idx)⟩ : Tile .real [BM, BD])
          (Tile.transpose [] (⟨fun idx => some (kTfn idx)⟩ : Tile .real [BN, BD]))))
      (Tile.scalar (some sc : WithBot ℝ)) with hqk0def
  -- run the shared tail
  obtain ⟨rmaxT, hrm, sF, htail, htailpids, htailmem, htailundef,
      mcurrT, lcurrT, lrcpT, pT, acc1T, hmcd, hpTd, hlcd, hlrd, hacc1d,
      hmF, hlF, haccF, hkpF, hvpF, hqF, hoffmF, hoffnF, hbiF, hohF⟩ :=
    al_loopTail_steps skn svk (BN * numKV) BM BN BD i sPre hBN qk0
      (⟨fun r : TileIndex [BM] => mPartial BN qT numKV kT sc c r.1⟩)
      (⟨fun r : TileIndex [BM] => ((lPartial qT numKV kT sc c r.1 : ℝ) : WithBot ℝ)⟩)
      (⟨fun idx : TileIndex [BM, BD] =>
        ((oPartial qT numKV kT vT sc c idx
            / lPartial qT numKV kT sc c idx.1 : ℝ) : WithBot ℝ)⟩)
      K V (alBase s0 N_HEAD skz skh) (alBase s0 N_HEAD svz svh) skk svn vTfn
      hin
      (by rw [hpreqk])
      hprebn
      (by rw [hpremp, hse, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hmp)
      (by rw [hprelp, hse, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hlp)
      (by rw [hpreacc, hse, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hacc)
      (by rw [hprekp, hse, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hkp)
      (by rw [hprevp, hse, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hvp)
      (by intro idx
          rw [show sPre.readMem V (alBase s0 N_HEAD svz svh + (i + idx.1.val) * svk + idx.2.1.val * svn)
              = se.readMem V (alBase s0 N_HEAD svz svh + (i + idx.1.val) * svk + idx.2.1.val * svn) from by
            unfold BlockState.readMem; rw [hpremem]]
          exact hvload idx)
      hpreundef
  refine ⟨sF, ?_, ?_⟩
  · show stepStmts (alLoopPrefixG sc (BN * numKV) BM BN BD ++ alLoopTailG skn svk (BN * numKV) BM BN BD) se = some sF
    rw [stepStmts.append_some hpre]
    exact htail
  -- ══ re-establish the invariant at i + BN ══
  -- per-cell score bridge
  have hqk0cell : ∀ (r : Fin BM) (j : Fin BN),
      qk0.data (r, j, PUnit.unit)
        = some (StreamingAccumulator.scaledScore qT kT sc r (blockIndex BN numKV c (Nat.succ_le_iff.mpr hclt) j)) := by
    intro r j
    rw [hqk0def]
    exact al_qk0_cell BM BN BD numKV qT kT sc c hclt r j
  have hrmaxCell : ∀ r : Fin BM,
      rmaxT.data (r, PUnit.unit)
        = (Finset.univ : Finset (Fin BN)).sup (fun j =>
            ((StreamingAccumulator.scaledScore qT kT sc r (blockIndex BN numKV c (Nat.succ_le_iff.mpr hclt) j) : ℝ) : WithBot ℝ)) := by
    intro r
    rw [al_rmax_cell hBN qk0 rmaxT hrm r]
    exact Finset.sup_congr rfl (fun j _ => hqk0cell r j)
  have hmcurrCell : ∀ r : Fin BM,
      mcurrT.data (r, PUnit.unit) = mPartial BN qT numKV kT sc (c + 1) r := by
    intro r
    rw [hmcd, Tile.select_data, Tile.cop_data]
    simp only [Broadcast.leftIndex, Broadcast.rightIndex, ComparableDType.gt, hrmaxCell r]
    rw [mPartial_succ_of_lt qT numKV kT sc c hclt r]
    by_cases hgt : (Finset.univ : Finset (Fin BN)).sup (fun j =>
        ((StreamingAccumulator.scaledScore qT kT sc r (blockIndex BN numKV c (Nat.succ_le_iff.mpr hclt) j) : ℝ) : WithBot ℝ))
        > mPartial BN qT numKV kT sc c r
    · rw [decide_eq_true hgt]
      simp only [if_true]
      rw [max_eq_right (le_of_lt hgt)]
    · rw [decide_eq_false hgt]
      simp only [Bool.false_eq_true, if_false]
      rw [max_eq_left (not_lt.mp hgt)]
  have hmcurrTile : mcurrT = ⟨fun r : TileIndex [BM] => mPartial BN qT numKV kT sc (c + 1) r.1⟩ := by
    ext r; exact hmcurrCell r.1
  have hne : ∀ r : Fin BM, mPartial BN qT numKV kT sc (c + 1) r ≠ ⊥ :=
    fun r => mPartial_succ_ne_bot hBN qT numKV kT sc c (Nat.succ_le_iff.mpr hclt) r
  set μ : Fin BM → ℝ := fun r => (mPartial BN qT numKV kT sc (c + 1) r).unbotD 0 with hμdef
  have hμcell : ∀ r : Fin BM, mcurrT.data (r, PUnit.unit) = some (μ r) := by
    intro r
    rw [hmcurrCell r]
    simp only [hμdef]
    cases hmp' : mPartial BN qT numKV kT sc (c + 1) r with
    | bot => exact absurd hmp' (hne r)
    | coe m => rfl
  -- alpha cell: exp(mPartial c − mPartial (c+1)) = alphaPartial c (as a real some)
  have halphaCell : ∀ r : Fin BM,
      (Tile.uop WithBot.realExp
        (Tile.bop NumericDType.real.sub Broadcast.nil.consSame
          (⟨fun r : TileIndex [BM] => mPartial BN qT numKV kT sc c r.1⟩ : Tile .real [BM])
          mcurrT)).data (r, PUnit.unit)
        = ((alphaPartial qT numKV kT sc c r : ℝ) : WithBot ℝ) := by
    intro r
    rw [Tile.uop_data, Tile.bop_data]
    simp only [Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.sub]
    rw [hmcurrTile]
    exact alphaPartial_toWithBot qT numKV kT sc c r
  -- l_curr cell = lPartial (c+1)
  have hlcurrCell : ∀ r : Fin BM,
      lcurrT.data (r, PUnit.unit) = some (lPartial qT numKV kT sc (c + 1) r) := by
    intro r
    rw [hlcd, Tile.bop_data]
    simp only [Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add]
    erw [al_pexp_rowsum_real qk0 mcurrT
      (fun r j => StreamingAccumulator.scaledScore qT kT sc r (blockIndex BN numKV c (Nat.succ_le_iff.mpr hclt) j)) μ
      hqk0cell hμcell r]
    rw [show (Tile.bop NumericDType.real.mul Broadcast.nil.consSame
          (⟨fun r : TileIndex [BM] => ((lPartial qT numKV kT sc c r.1 : ℝ) : WithBot ℝ)⟩ : Tile .real [BM])
          (Tile.uop WithBot.realExp
            (Tile.bop NumericDType.real.sub Broadcast.nil.consSame
              (⟨fun r : TileIndex [BM] => mPartial BN qT numKV kT sc c r.1⟩ : Tile .real [BM])
              mcurrT))).data (r, PUnit.unit)
        = some (lPartial qT numKV kT sc c r * alphaPartial qT numKV kT sc c r) from by
      rw [Tile.bop_data]
      simp only [Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.mul]
      rw [halphaCell r]
      rfl]
    show WithBot.realAdd (some _) (some _) = some _
    rw [WithBot.realAdd, Option.map₂_some_some]
    refine congrArg some ?_
    rw [lPartial_succ_of_lt qT numKV kT sc c hclt r]
    simp only [hμdef]
    ring
  -- l_rcp cell = 1 / lPartial (c+1)
  have hlrcpCell : ∀ r : Fin BM,
      lrcpT.data (r, PUnit.unit) = some (1 / lPartial qT numKV kT sc (c + 1) r) := by
    intro r
    rw [hlrd, Tile.bop_data]
    simp only [Broadcast.leftIndex, Broadcast.rightIndex, Broadcast.scalarL, Tile.scalar_data,
      NumericDType.div]
    rw [hlcurrCell r]
    show WithBot.realDiv (some (1.0 : ℝ)) (some (lPartial qT numKV kT sc (c + 1) r))
      = some (1 / lPartial qT numKV kT sc (c + 1) r)
    rw [WithBot.realDiv, Option.map₂_some_some]
    norm_num
  -- final acc cell = oPartial (c+1) / lPartial (c+1)
  have haccCell : ∀ idx : TileIndex [BM, BD],
      (Tile.bop NumericDType.real.add Broadcast.nil.consSame.consSame
        acc1T (Tile.dot [] pT (⟨fun idx => some (vTfn idx)⟩ : Tile .real [BN, BD]))).data idx
        = some (oPartial qT numKV kT vT sc (c + 1) idx
            / lPartial qT numKV kT sc (c + 1) idx.1) := by
    intro idx
    obtain ⟨r, d, u⟩ := idx; cases u
    rw [Tile.bop_data]
    simp only [Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add]
    have hacc1V : acc1T.data (r, d, PUnit.unit)
        = some (oPartial qT numKV kT vT sc c (r, d, PUnit.unit)
              / lPartial qT numKV kT sc c r
            * (lPartial qT numKV kT sc c r * alphaPartial qT numKV kT sc c r
                * (1 / lPartial qT numKV kT sc (c + 1) r))) := by
      rw [hacc1d, Tile.bop_data]
      simp only [Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.mul, NumericDType.sub,
        Tile.expandDim_data, TileShape.dropInsertedIndex, Tile.bop_data]
      rw [halphaCell r, hlrcpCell r]
      show WithBot.realMul
          (some (oPartial qT numKV kT vT sc c (r, d, PUnit.unit) / lPartial qT numKV kT sc c r))
          (WithBot.realMul
            (WithBot.realMul (some (lPartial qT numKV kT sc c r))
              (some (alphaPartial qT numKV kT sc c r)))
            (some (1 / lPartial qT numKV kT sc (c + 1) r))) = _
      simp only [WithBot.realMul, Option.map₂_some_some]
    rw [hacc1V]
    rw [hpTd]
    erw [al_pv_dot_real qk0 mcurrT lrcpT (⟨fun idx => some (vTfn idx)⟩ : Tile .real [BN, BD])
      (fun r j => StreamingAccumulator.scaledScore qT kT sc r (blockIndex BN numKV c (Nat.succ_le_iff.mpr hclt) j)) μ vTfn
      hqk0cell hμcell (fun idx => rfl) r d
      (1 / lPartial qT numKV kT sc (c + 1) r) (hlrcpCell r)]
    rw [show WithBot.realAdd (some (oPartial qT numKV kT vT sc c (r, d, PUnit.unit)
            / lPartial qT numKV kT sc c r
          * (lPartial qT numKV kT sc c r * alphaPartial qT numKV kT sc c r
              * (1 / lPartial qT numKV kT sc (c + 1) r))))
        (some (1 / lPartial qT numKV kT sc (c + 1) r
          * (Finset.univ : Finset (Fin BN)).sum (fun j =>
              Real.exp (StreamingAccumulator.scaledScore qT kT sc r (blockIndex BN numKV c (Nat.succ_le_iff.mpr hclt) j) - μ r)
                * vTfn (j, d, PUnit.unit))))
      = some (oPartial qT numKV kT vT sc c (r, d, PUnit.unit)
            / lPartial qT numKV kT sc c r
          * (lPartial qT numKV kT sc c r * alphaPartial qT numKV kT sc c r
              * (1 / lPartial qT numKV kT sc (c + 1) r))
        + 1 / lPartial qT numKV kT sc (c + 1) r
          * (Finset.univ : Finset (Fin BN)).sum (fun j =>
              Real.exp (StreamingAccumulator.scaledScore qT kT sc r (blockIndex BN numKV c (Nat.succ_le_iff.mpr hclt) j) - μ r)
                * vTfn (j, d, PUnit.unit))) from by
      rw [WithBot.realAdd, Option.map₂_some_some]]
    refine congrArg some ?_
    have hcancel := al_oPartial_div_lPartial_cancel hBN qT kT vT sc c hcle (r, d, PUnit.unit)
    rw [oPartial_succ_of_lt qT numKV kT vT sc c hclt (r, d, PUnit.unit)]
    simp only [hμdef, hvTfn]
    rw [show oPartial qT numKV kT vT sc c (r, d, PUnit.unit)
            / lPartial qT numKV kT sc c r
          * (lPartial qT numKV kT sc c r * alphaPartial qT numKV kT sc c r
              * (1 / lPartial qT numKV kT sc (c + 1) r))
        = (oPartial qT numKV kT vT sc c (r, d, PUnit.unit)
            / lPartial qT numKV kT sc c r
          * lPartial qT numKV kT sc c r)
          * (alphaPartial qT numKV kT sc c r * (1 / lPartial qT numKV kT sc (c + 1) r)) from by ring]
    rw [hcancel]
    rw [add_div]
    congr 1
    · ring
    · rw [Finset.mul_sum, Finset.sum_div]
      refine Finset.sum_congr rfl (fun j _ => ?_)
      ring
  -- assemble
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [htailpids, hprepids, hse, BlockState.setReg_pids, hpids]
  · rw [hic, show c * BN + BN = (c + 1) * BN from by ring, Nat.mul_mod_left]
  · rw [hic, show c * BN + BN = (c + 1) * BN from by ring]
    have h1 : (c + 1) * BN ≤ numKV * BN := Nat.mul_le_mul_right _ hclt
    rw [Nat.mul_comm numKV BN] at h1
    exact h1
  · rw [hmF, hc1, hmcurrTile]
  · rw [hlF, hc1]; refine congrArg some ?_; ext r; rw [hlcurrCell r.1]; rfl
  · rw [haccF, hc1]; refine congrArg some ?_; ext idx; rw [haccCell idx]; rfl
  · rw [hqF, hpreq, hse, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hq
  · rw [hoffmF, hpreoffm, hse, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hoffm
  · rw [hoffnF, hpreoffn, hse, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hoffn
  · rw [hbiF, hprebi, hse, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hbi
  · rw [hohF, hpreoh, hse, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hoh
  · rw [hkpF]
  · rw [hvpF]
  · exact htailundef
  · rw [htailmem, hpremem, hse]
    funext region offset
    rw [BlockState.setReg_mem]
    exact congrFun (congrFun hmem region) offset

/-! ## Preamble walk (shared) -/

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
open VeriTile.Examples.AttentionForwardClosedForm in
/-- **Preamble execution.** The 18 shared statements step a clean state to
loop entry: pids split, index vectors, the three pointer tiles, the
`⊥`/zero seeds, the H-guarded `q` load, and `block_n_end = N_CTX`. -/
theorem alPreLoop_evalG (s : BlockState) (Q K V : RegionName)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn
      N_HEAD H N_CTX BM BN BD : Nat)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ∃ s0, stepStmts (alPreLoopG Q K V sqz sqh sqm sqk skz skh skn skk svz svh svk svn
        N_HEAD H N_CTX BM BN BD) s = some s0
      ∧ s0.pids = s.pids ∧ s0.mem = s.mem ∧ (∀ rg o, s0.undef rg o = 0)
      ∧ s0.regs .nat [] "start_m" = some (Tile.scalar (s.pids 0))
      ∧ s0.regs .nat [] "batch_id" = some (Tile.scalar (s.pids 1 / N_HEAD))
      ∧ s0.regs .nat [] "off_hz" = some (Tile.scalar (s.pids 1 % N_HEAD))
      ∧ s0.regs .nat [BM] "offs_m" = some (Tile.vec (fun r : Fin BM => s.pids 0 * BM + r.val))
      ∧ s0.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => j.val))
      ∧ s0.regs .real [BM] "m_prev" = some ⟨fun _ : TileIndex [BM] => (⊥ : WithBot ℝ)⟩
      ∧ s0.regs .real [BM] "l_prev" = some ⟨fun _ : TileIndex [BM] => some (0 : ℝ)⟩
      ∧ s0.regs .real [BM, BD] "acc" = some ⟨fun _ : TileIndex [BM, BD] => some (0 : ℝ)⟩
      ∧ s0.regs .real [BM, BD] "q" = some ⟨fun idx : TileIndex [BM, BD] =>
          some (alQTileG s Q N_HEAD sqz sqh sqm sqk H BM BD idx)⟩
      ∧ s0.regs .ptr [BN, BD] "k_ptrs"
          = some (alPtrTile K (alBase s N_HEAD skz skh) skn skk 0 BN BD)
      ∧ s0.regs .ptr [BN, BD] "v_ptrs"
          = some (alPtrTile V (alBase s N_HEAD svz svh) svk svn 0 BN BD)
      ∧ s0.regs .nat [] "block_n_end" = some (Tile.scalar N_CTX) := by
  unfold alPreLoopG
  -- S1: start_m = pid0
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 0 s))]
  -- S2: head_idx = pid1
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 1 _))]
  -- S3: batch_id = head_idx // N_HEAD
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.floorDiv IntegralDType.nat Broadcast.nil
        (Op.ref .nat [] "head_idx") (Op.constNat N_HEAD)) _
        = some (Tile.scalar (s.pids 1 / N_HEAD)) from by
      rw [al_evalOp_floorDiv]
      simp only [evalOp_ref, evalOp_constNat, BlockState.setReg_same, BlockState.setReg_pids,
        Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext u; rfl))]
  -- S4: off_hz = head_idx % N_HEAD
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.mod IntegralDType.nat Broadcast.nil
        (Op.ref .nat [] "head_idx") (Op.constNat N_HEAD)) _
        = some (Tile.scalar (s.pids 1 % N_HEAD)) from by
      rw [al_evalOp_mod]
      simp only [evalOp_ref, evalOp_constNat, BlockState.setReg_same, BlockState.setReg_pids,
        BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
        Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext u; rfl))]
  -- S5: offs_m = start_m * BM + arange BM
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat BM)) (Op.arange BM)) _
        = some (Tile.vec (fun r : Fin BM => s.pids 0 * BM + r.val)) from by
      rw [evalOp_add, evalOp_arange]
      simp only [evalOp_mul, evalOp_ref, evalOp_constNat, BlockState.setReg_same,
        BlockState.setReg_pids, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
        not_false_eq_true, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext r
      simp only [Tile.bop_data, Tile.scalar_data, Tile.vec_data, castTile_self,
        Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul]))]
  -- S6: offs_n = arange BN
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_arange BN _))]
  -- S7: offs_d = arange BD
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_arange BD _))]
  -- S8-S10: the three flat 2D offset tiles
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .nat Broadcast.nil.consL.consR
        (Op.add .nat Broadcast.scalarL
          (Op.add .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "batch_id") (Op.constNat sqz))
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat sqh)))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m")) (Op.constNat sqm)))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BD] "offs_d")) (Op.constNat sqk))) _
        = some (⟨fun idx : TileIndex [BM, BD] =>
            alBase s N_HEAD sqz sqh + (s.pids 0 * BM + idx.1.val) * sqm + idx.2.1.val * sqk⟩
              : Tile .nat [BM, BD]) from by
      rw [evalOp_add, evalOp_add, evalOp_add, evalOp_mul, evalOp_mul, evalOp_mul, evalOp_mul]
      rw [show @evalOp TileDType.nat [BM, 1]
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m")) _
          = some (Tile.expandDim ⟨1, by simp⟩ (Tile.vec (fun r : Fin BM => s.pids 0 * BM + r.val)))
        from evalOp_expandDim_ref_of_regs _ _ _ _ _ _ (by
          simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
            String.reduceEq, not_false_eq_true])]
      rw [show @evalOp TileDType.nat [1, BD]
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BD] "offs_d")) _
          = some (Tile.expandDim ⟨0, by simp⟩ (Tile.vec (fun d : Fin BD => d.val)))
        from evalOp_expandDim_ref_of_regs _ _ _ _ _ _ (by
          simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
            String.reduceEq, not_false_eq_true])]
      simp only [evalOp_ref, evalOp_constNat, BlockState.setReg_same, BlockState.setReg_ne_name,
        ne_eq, String.reduceEq, not_false_eq_true, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp only [Tile.bop_data, Tile.scalar_data, Tile.vec_data, Tile.expandDim_data,
        castTile_self, Broadcast.leftIndex, Broadcast.rightIndex, TileShape.dropInsertedIndex,
        NumericDType.add, NumericDType.mul]
      rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .nat Broadcast.nil.consL.consR
        (Op.add .nat Broadcast.scalarL
          (Op.add .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "batch_id") (Op.constNat skz))
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat skh)))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BN] "offs_n")) (Op.constNat skn)))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BD] "offs_d")) (Op.constNat skk))) _
        = some (⟨fun idx : TileIndex [BN, BD] =>
            alBase s N_HEAD skz skh + (0 + idx.1.val) * skn + idx.2.1.val * skk⟩
              : Tile .nat [BN, BD]) from by
      rw [evalOp_add, evalOp_add, evalOp_add, evalOp_mul, evalOp_mul, evalOp_mul, evalOp_mul]
      rw [show @evalOp TileDType.nat [BN, 1]
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BN] "offs_n")) _
          = some (Tile.expandDim ⟨1, by simp⟩ (Tile.vec (fun j : Fin BN => j.val)))
        from evalOp_expandDim_ref_of_regs _ _ _ _ _ _ (by
          simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
            String.reduceEq, not_false_eq_true])]
      rw [show @evalOp TileDType.nat [1, BD]
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BD] "offs_d")) _
          = some (Tile.expandDim ⟨0, by simp⟩ (Tile.vec (fun d : Fin BD => d.val)))
        from evalOp_expandDim_ref_of_regs _ _ _ _ _ _ (by
          simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
            String.reduceEq, not_false_eq_true])]
      simp only [evalOp_ref, evalOp_constNat, BlockState.setReg_same, BlockState.setReg_ne_name,
        ne_eq, String.reduceEq, not_false_eq_true, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp only [Tile.bop_data, Tile.scalar_data, Tile.vec_data, Tile.expandDim_data,
        castTile_self, Broadcast.leftIndex, Broadcast.rightIndex, TileShape.dropInsertedIndex,
        NumericDType.add, NumericDType.mul]
      show (s.pids 1 / N_HEAD) * skz + (s.pids 1 % N_HEAD) * skh + idx.1.val * skn + idx.2.1.val * skk
        = alBase s N_HEAD skz skh + (0 + idx.1.val) * skn + idx.2.1.val * skk
      simp [alBase, Nat.zero_add]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .nat Broadcast.nil.consL.consR
        (Op.add .nat Broadcast.scalarL
          (Op.add .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "batch_id") (Op.constNat svz))
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat svh)))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BN] "offs_n")) (Op.constNat svk)))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BD] "offs_d")) (Op.constNat svn))) _
        = some (⟨fun idx : TileIndex [BN, BD] =>
            alBase s N_HEAD svz svh + (0 + idx.1.val) * svk + idx.2.1.val * svn⟩
              : Tile .nat [BN, BD]) from by
      rw [evalOp_add, evalOp_add, evalOp_add, evalOp_mul, evalOp_mul, evalOp_mul, evalOp_mul]
      rw [show @evalOp TileDType.nat [BN, 1]
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BN] "offs_n")) _
          = some (Tile.expandDim ⟨1, by simp⟩ (Tile.vec (fun j : Fin BN => j.val)))
        from evalOp_expandDim_ref_of_regs _ _ _ _ _ _ (by
          simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
            String.reduceEq, not_false_eq_true])]
      rw [show @evalOp TileDType.nat [1, BD]
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BD] "offs_d")) _
          = some (Tile.expandDim ⟨0, by simp⟩ (Tile.vec (fun d : Fin BD => d.val)))
        from evalOp_expandDim_ref_of_regs _ _ _ _ _ _ (by
          simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
            String.reduceEq, not_false_eq_true])]
      simp only [evalOp_ref, evalOp_constNat, BlockState.setReg_same, BlockState.setReg_ne_name,
        ne_eq, String.reduceEq, not_false_eq_true, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp only [Tile.bop_data, Tile.scalar_data, Tile.vec_data, Tile.expandDim_data,
        castTile_self, Broadcast.leftIndex, Broadcast.rightIndex, TileShape.dropInsertedIndex,
        NumericDType.add, NumericDType.mul]
      show (s.pids 1 / N_HEAD) * svz + (s.pids 1 % N_HEAD) * svh + idx.1.val * svk + idx.2.1.val * svn
        = alBase s N_HEAD svz svh + (0 + idx.1.val) * svk + idx.2.1.val * svn
      simp [alBase, Nat.zero_add]))]
  -- S11-S13: the three pointer tiles
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Q) (Op.ref .nat [BM, BD] "off_q")) _
        = some (alPtrTile Q (alBase s N_HEAD sqz sqh) sqm sqk (s.pids 0 * BM) BM BD) from by
      rw [evalOp_ptrAdd, evalOp_ptrBase]
      simp only [evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
        String.reduceEq, not_false_eq_true, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some (Tile.ext fun idx => ?_)
      show (Q, 0 + (alBase s N_HEAD sqz sqh + (s.pids 0 * BM + idx.1.val) * sqm + idx.2.1.val * sqk))
        = (Q, alBase s N_HEAD sqz sqh + (s.pids 0 * BM + idx.1.val) * sqm + idx.2.1.val * sqk)
      rw [Nat.zero_add]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase K) (Op.ref .nat [BN, BD] "off_k")) _
        = some (alPtrTile K (alBase s N_HEAD skz skh) skn skk 0 BN BD) from by
      rw [evalOp_ptrAdd, evalOp_ptrBase]
      simp only [evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
        String.reduceEq, not_false_eq_true, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some (Tile.ext fun idx => ?_)
      show (K, 0 + (alBase s N_HEAD skz skh + (0 + idx.1.val) * skn + idx.2.1.val * skk))
        = (K, alBase s N_HEAD skz skh + (0 + idx.1.val) * skn + idx.2.1.val * skk)
      rw [Nat.zero_add]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase V) (Op.ref .nat [BN, BD] "off_v")) _
        = some (alPtrTile V (alBase s N_HEAD svz svh) svk svn 0 BN BD) from by
      rw [evalOp_ptrAdd, evalOp_ptrBase]
      simp only [evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
        String.reduceEq, not_false_eq_true, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some (Tile.ext fun idx => ?_)
      show (V, 0 + (alBase s N_HEAD svz svh + (0 + idx.1.val) * svk + idx.2.1.val * svn))
        = (V, alBase s N_HEAD svz svh + (0 + idx.1.val) * svk + idx.2.1.val * svn)
      rw [Nat.zero_add]))]
  -- S14: m_prev = zeros - inf = ⊥
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .real Broadcast.scalarR (Op.full [BM] (Op.const 0)) Op.negInf) _
        = some (⟨fun _ : TileIndex [BM] => (⊥ : WithBot ℝ)⟩ : Tile .real [BM]) from by
      rw [evalOp_add]
      simp only [evalOp_full, evalOp_const, evalOp_negInf, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext r
      simp only [Tile.bop_data, Tile.scalar_data, castTile_self, Broadcast.leftIndex,
        Broadcast.rightIndex, NumericDType.add, WithBot.realAdd, Option.map₂_none_right]
      rfl))]
  -- S15: l_prev = zeros
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.full [BM] (Op.const 0)) _
        = some (⟨fun _ : TileIndex [BM] => some (0 : ℝ)⟩ : Tile .real [BM]) from by
      simp only [evalOp_full, evalOp_const, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext r; rfl))]
  -- S16: acc = zeros
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.full [BM, BD] (Op.const 0)) _
        = some (⟨fun _ : TileIndex [BM, BD] => some (0 : ℝ)⟩ : Tile .real [BM, BD]) from by
      simp only [evalOp_full, evalOp_const, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext r; rfl))]
  -- S17: q = masked load with other=0.0
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.load .real (MemAccess.ptr (Op.ref .ptr [BM, BD] "q_ptrs"))
        (MaskOpt.maskOther
          (Op.remap [BM, BD] Broadcast.nil.consL.consSame.leftIndex
            (Op.lt ComparableDType.nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m")) (Op.constNat H)))
          (Op.broadcast (Op.const (0.0 : ℝ)) [BM, BD]))) _
        = some ⟨fun idx : TileIndex [BM, BD] =>
            some (alQTileG s Q N_HEAD sqz sqh sqm sqk H BM BD idx)⟩ from by
      rw [al_load_ptr_maskOther_real (Op.ref .ptr [BM, BD] "q_ptrs") _ _ _
        (alPtrTile Q (alBase s N_HEAD sqz sqh) sqm sqk (s.pids 0 * BM) BM BD)
        (⟨fun idx : TileIndex [BM, BD] => decide (s.pids 0 * BM + idx.1.val < H)⟩ : Tile .bool [BM, BD])
        (⟨fun _ : TileIndex [BM, BD] => some (0.0 : ℝ)⟩ : Tile .real [BM, BD])
        (by rw [evalOp_ref]
            simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
              String.reduceEq, not_false_eq_true])
        (al_rowmask_eval _ (fun r : Fin BM => s.pids 0 * BM + r.val) H (by
          simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
            String.reduceEq, not_false_eq_true]))
        (by simp only [evalOp, evalOp_const, Option.bind_eq_bind, Option.bind_some]; rfl)]
      refine congrArg some ?_; ext idx
      by_cases hr : s.pids 0 * BM + idx.1.val < H
      · simp only [hr, decide_true, if_true]
        unfold alQTileG alRow
        rw [if_pos hr]
        simp only [alPtrTile, BlockState.setReg_readMem]
      · simp only [hr, decide_false, Bool.false_eq_true, if_false]
        have hq0 : alQTileG s Q N_HEAD sqz sqh sqm sqk H BM BD idx = 0 := by
          unfold alQTileG alRow
          rw [if_neg hr]
        rw [hq0]
        exact congrArg some (by norm_num)))]
  -- S18: block_n_end = N_CTX
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_constNat N_CTX _))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp
  · funext region offset; simp
  · intro rg o; simp [hundef]
  · simp [BlockState.setReg_ne_name]
  · simp [BlockState.setReg_ne_name]
  · simp [BlockState.setReg_ne_name]
  · simp [BlockState.setReg_ne_name]
  · simp [BlockState.setReg_ne_name]
  · simp [BlockState.setReg_ne_name]
  · simp [BlockState.setReg_ne_name]
  · simp [BlockState.setReg_ne_name]
  · simp [BlockState.setReg_ne_name]
  · simp [BlockState.setReg_ne_name]
  · simp [BlockState.setReg_ne_name]
  · simp

/-! ## PostLoop walk: the masked `.real` `Out` store -/

/-- The masked scatter of the final `acc` into `Out` (the store's foldl
normal form): lane `(i, d)` writes `g (i, d)` at
`oBase + (row0 + i)·som + d·son` iff `row0 + i < H`. -/
noncomputable def alOutStoreState (sin : BlockState) (Out : RegionName)
    (oBase som son row0 H BM BD : Nat) (g : TileIndex [BM, BD] → ℝ) : BlockState :=
  (TileShape.allIndices [BM, BD]).foldl
    (fun acc a => if row0 + a.1.val < H
        then acc.writeMem Out (oBase + (row0 + a.1.val) * som + a.2.1.val * son) (g a) else acc) sin

set_option maxHeartbeats 4000000 in
/-- **Out store step (eq).** -/
theorem alOutStore_step_eq (sin : BlockState) (Out : RegionName)
    (oBase som son row0 H BM BD : Nat) (g : TileIndex [BM, BD] → ℝ)
    (accT : Tile .real [BM, BD])
    (hgf : ∀ a : TileIndex [BM, BD], accT.data a = some (g a))
    (hb : sin.regs .real [BM, BD] "acc" = some accT)
    (hp : sin.regs .ptr [BM, BD] "out_ptrs" = some (alPtrTile Out oBase som son row0 BM BD))
    (hom : sin.regs .nat [BM] "offs_m" = some (Tile.vec (fun r : Fin BM => row0 + r.val))) :
    stepStmt (Stmt.store .real [BM, BD] (MemAccess.ptr (Op.ref .ptr [BM, BD] "out_ptrs"))
        (Op.ref .real [BM, BD] "acc")
        (MaskOpt.mask
          (Op.remap [BM, BD] Broadcast.nil.consL.consSame.leftIndex
            (Op.lt ComparableDType.nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m")) (Op.constNat H))))) sin
      = some (alOutStoreState sin Out oBase som son row0 H BM BD g) := by
  have hmask := al_rowmask_eval (S2 := BD) sin (fun r : Fin BM => row0 + r.val) H hom
  unfold stepStmt alOutStoreState
  simp only [evalOp_ref, hb, hp, hmask]
  simp only [Option.bind_eq_bind, Option.bind_some, Option.map_some]
  refine congrArg some
    (congrArg (fun gg => List.foldl gg sin (TileShape.allIndices [BM, BD])) ?_)
  funext acc a
  by_cases h : row0 + a.1.val < H
  · simp only [h, decide_true, if_true, BlockState.writeMemTyped_real, hgf]
    rfl
  · simp only [h, decide_false, Bool.false_eq_true, if_false]

set_option maxHeartbeats 4000000 in
/-- **Out store readback + pids frame.** -/
theorem alOutStore_props (sin : BlockState) (Out : RegionName)
    (oBase som son row0 H BM BD : Nat) (g : TileIndex [BM, BD] → ℝ)
    (houtinj : Function.Injective (fun idx : TileIndex [BM, BD] =>
        (row0 + idx.1.val) * som + idx.2.1.val * son)) :
    (alOutStoreState sin Out oBase som son row0 H BM BD g).pids = sin.pids
      ∧ (∀ a : TileIndex [BM, BD], row0 + a.1.val < H →
          (alOutStoreState sin Out oBase som son row0 H BM BD g).readMem Out
              (oBase + (row0 + a.1.val) * som + a.2.1.val * son)
            = g a) := by
  classical
  unfold alOutStoreState
  constructor
  · rw [BlockState.foldl_writeMem_prop_masked_pids]
  · intro a ha
    have hInj : Function.Injective (fun i : TileIndex [BM, BD] =>
        oBase + (row0 + i.1.val) * som + i.2.1.val * son) := by
      intro x y hxy
      apply houtinj
      have hxy' : oBase + ((row0 + x.1.val) * som + x.2.1.val * son)
          = oBase + ((row0 + y.1.val) * som + y.2.1.val * son) := by
        simpa [Nat.add_assoc] using hxy
      exact Nat.add_left_cancel hxy'
    have h := BlockState.scatter_readback_prop_masked_nd (region := Out) sin
      (fun i : TileIndex [BM, BD] => oBase + (row0 + i.1.val) * som + i.2.1.val * son)
      (fun i => g i) (fun i => row0 + i.1.val < H) hInj a
    rw [h, if_pos ha]

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **PostLoop walk (both arms).** From ANY loop-exit state whose `acc`
register holds a real tile `g` and whose `batch_id`/`off_hz` scalars
survive, the 6 postLoop statements rematerialize the offsets and store
`g` at every active (`row < H`) `Out` lane. -/
theorem alPostLoop_run (Out : RegionName) (s0 : BlockState)
    (soz soh som son N_HEAD H BM BD : Nat) (sL : BlockState)
    (g : TileIndex [BM, BD] → ℝ)
    (houtinj : Function.Injective (fun idx : TileIndex [BM, BD] =>
        (s0.pids 0 * BM + idx.1.val) * som + idx.2.1.val * son))
    (hpids : sL.pids = s0.pids)
    (hacc : sL.regs .real [BM, BD] "acc"
      = some ⟨fun idx : TileIndex [BM, BD] => ((g idx : ℝ) : WithBot ℝ)⟩)
    (hbi : sL.regs .nat [] "batch_id" = some (Tile.scalar (s0.pids 1 / N_HEAD)))
    (hoh : sL.regs .nat [] "off_hz" = some (Tile.scalar (s0.pids 1 % N_HEAD))) :
    ∃ sP, stepStmts (alPostLoopG Out soz soh som son H BM BD) sL = some sP
      ∧ ∀ idx : TileIndex [BM, BD], alActive s0 H BM idx →
          sP.readMem Out (alOutOffset s0 N_HEAD soz soh som son BM idx) = g idx := by
  unfold alPostLoopG
  -- W1: start_m = pid0
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.programId 0) sL = some (Tile.scalar (s0.pids 0)) from by
      rw [evalOp_programId, hpids]))]
  -- W2: offs_m = start_m * BM + arange BM
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat BM)) (Op.arange BM)) _
        = some (Tile.vec (fun r : Fin BM => s0.pids 0 * BM + r.val)) from by
      rw [evalOp_add, evalOp_arange]
      simp only [evalOp_mul, evalOp_ref, evalOp_constNat, BlockState.setReg_same,
        Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext r
      simp only [Tile.bop_data, Tile.scalar_data, Tile.vec_data, castTile_self,
        Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul]))]
  -- W3: offs_d = arange BD
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_arange BD _))]
  -- W4: off_o
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .nat Broadcast.nil.consL.consR
        (Op.add .nat Broadcast.scalarL
          (Op.add .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "batch_id") (Op.constNat soz))
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat soh)))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m")) (Op.constNat som)))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BD] "offs_d")) (Op.constNat son))) _
        = some (⟨fun idx : TileIndex [BM, BD] =>
            alBase s0 N_HEAD soz soh + (s0.pids 0 * BM + idx.1.val) * som + idx.2.1.val * son⟩
              : Tile .nat [BM, BD]) from by
      rw [evalOp_add, evalOp_add, evalOp_add, evalOp_mul, evalOp_mul, evalOp_mul, evalOp_mul]
      rw [show @evalOp TileDType.nat [BM, 1]
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m")) _
          = some (Tile.expandDim ⟨1, by simp⟩ (Tile.vec (fun r : Fin BM => s0.pids 0 * BM + r.val)))
        from evalOp_expandDim_ref_of_regs _ _ _ _ _ _ (by
          simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
            String.reduceEq, not_false_eq_true])]
      rw [show @evalOp TileDType.nat [1, BD]
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BD] "offs_d")) _
          = some (Tile.expandDim ⟨0, by simp⟩ (Tile.vec (fun d : Fin BD => d.val)))
        from evalOp_expandDim_ref_of_regs _ _ _ _ _ _ (by
          simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
            String.reduceEq, not_false_eq_true])]
      rw [show evalOp (Op.ref .nat [] "batch_id") _ = some (Tile.scalar (s0.pids 1 / N_HEAD)) from by
        rw [evalOp_ref]
        simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]
        exact hbi]
      rw [show evalOp (Op.ref .nat [] "off_hz") _ = some (Tile.scalar (s0.pids 1 % N_HEAD)) from by
        rw [evalOp_ref]
        simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]
        exact hoh]
      simp only [evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp only [Tile.bop_data, Tile.scalar_data, Tile.vec_data, Tile.expandDim_data,
        castTile_self, Broadcast.leftIndex, Broadcast.rightIndex, TileShape.dropInsertedIndex,
        NumericDType.add, NumericDType.mul]
      rfl))]
  -- W5: out_ptrs = Out + off_o
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Out) (Op.ref .nat [BM, BD] "off_o")) _
        = some (alPtrTile Out (alBase s0 N_HEAD soz soh) som son (s0.pids 0 * BM) BM BD) from by
      rw [evalOp_ptrAdd, evalOp_ptrBase]
      simp only [evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
        String.reduceEq, not_false_eq_true, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some (Tile.ext fun idx => ?_)
      show (Out, 0 + (alBase s0 N_HEAD soz soh + (s0.pids 0 * BM + idx.1.val) * som + idx.2.1.val * son))
        = (Out, alBase s0 N_HEAD soz soh + (s0.pids 0 * BM + idx.1.val) * som + idx.2.1.val * son)
      rw [Nat.zero_add]))]
  -- W6: the masked store
  rw [stepStmts.cons_some (alOutStore_step_eq _ Out (alBase s0 N_HEAD soz soh) som son
    (s0.pids 0 * BM) H BM BD g
    (⟨fun idx : TileIndex [BM, BD] => ((g idx : ℝ) : WithBot ℝ)⟩ : Tile .real [BM, BD])
    (fun a => rfl)
    (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]
        rw [hacc])
    (by simp only [BlockState.setReg_same])
    (by simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
          String.reduceEq, not_false_eq_true]))]
  rw [stepStmts.nil]
  obtain ⟨_hpidsP, hread⟩ := alOutStore_props _ Out (alBase s0 N_HEAD soz soh) som son
    (s0.pids 0 * BM) H BM BD g houtinj
  refine ⟨_, rfl, ?_⟩
  intro idx hActive
  have h := hread idx hActive
  exact h

/-! ## Non-causal exec + headline -/

set_option maxHeartbeats 8000000 in
set_option maxRecDepth 8000 in
open StreamingAccumulator in
/-- **Non-causal full execution.** Preamble walk → `forRangeDyn_inv` over
`alInv` → postLoop store walk. Every active `Out` lane holds the fully
streamed `oPartial numKV / lPartial numKV` over the initial state's
Q/K/V tiles. -/
theorem al_exec (Q K V Out : RegionName) (s : BlockState) (sc : ℝ)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      N_HEAD H SP BM BN BD numKV : Nat)
    (hBM : 0 < BM) (hBN : 0 < BN)
    (houtinj : Function.Injective (fun idx : TileIndex [BM, BD] =>
        (s.pids 0 * BM + idx.1.val) * som + idx.2.1.val * son))
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ∃ sF, exec (attention_llama_fwd_surface Q K V Out sc
        sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
        N_HEAD H (BN * numKV) SP BM BN BD) s = some sF
      ∧ ∀ idx : TileIndex [BM, BD], alActive s H BM idx →
          sF.readMem Out (alOutOffset s N_HEAD soz soh som son BM idx)
            = oPartial (alQTileG s Q N_HEAD sqz sqh sqm sqk H BM BD) numKV
                (alKTileG s K N_HEAD skz skh skn skk (BN * numKV) BD)
                (alVTileG s V N_HEAD svz svh svk svn (BN * numKV) BD) sc numKV idx
              / lPartial (alQTileG s Q N_HEAD sqz sqh sqm sqk H BM BD) numKV
                (alKTileG s K N_HEAD skz skh skn skk (BN * numKV) BD) sc numKV idx.1 := by
  have hbody : exec (attention_llama_fwd_surface Q K V Out sc
        sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
        N_HEAD H (BN * numKV) SP BM BN BD) s
      = stepStmts (alPreLoopG Q K V sqz sqh sqm sqk skz skh skn skk svz svh svk svn
            N_HEAD H (BN * numKV) BM BN BD
          ++ (Stmt.forRangeDyn "start_n" (Op.constNat 0) (Op.ref .nat [] "block_n_end")
                (Op.constNat BN) (alLoopBodyG sc skn svk (BN * numKV) BM BN BD)
              :: alPostLoopG Out soz soh som son H BM BD)) s := by
    show stepStmts (attention_llama_fwd_surface Q K V Out sc
        sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
        N_HEAD H (BN * numKV) SP BM BN BD).toAlgKernel.body s = _
    rw [al_body_split]
  rw [hbody]
  -- preamble
  obtain ⟨sp, hpre, hsppids, hspmem, hspundef, _hsm, hbi, hoh, hoffm, hoffn, hmp, hlp, hacc,
      hqreg, hkp, hvp, hbne⟩ :=
    alPreLoop_evalG s Q K V sqz sqh sqm sqk skz skh skn skk svz svh svk svn
      N_HEAD H (BN * numKV) BM BN BD hundef
  rw [stepStmts.append_some hpre]
  -- tile transport `s → sp` (the invariant is stated over sp)
  have htileQ : alQTileG sp Q N_HEAD sqz sqh sqm sqk H BM BD
      = alQTileG s Q N_HEAD sqz sqh sqm sqk H BM BD := by
    funext idx
    simp only [alQTileG, alRow, alBase, hsppids]
    unfold BlockState.readMem
    rw [hspmem]
  have htileK : alKTileG sp K N_HEAD skz skh skn skk (BN * numKV) BD
      = alKTileG s K N_HEAD skz skh skn skk (BN * numKV) BD := by
    funext idx
    simp only [alKTileG, alBase, hsppids]
    unfold BlockState.readMem
    rw [hspmem]
  have htileV : alVTileG sp V N_HEAD svz svh svk svn (BN * numKV) BD
      = alVTileG s V N_HEAD svz svh svk svn (BN * numKV) BD := by
    funext idx
    simp only [alVTileG, alBase, hsppids]
    unfold BlockState.readMem
    rw [hspmem]
  have hbaseK : alBase sp N_HEAD skz skh = alBase s N_HEAD skz skh := by
    simp only [alBase, hsppids]
  have hbaseV : alBase sp N_HEAD svz svh = alBase s N_HEAD svz svh := by
    simp only [alBase, hsppids]
  -- invariant base case at counter 0 (over sp)
  have hinv0 : alInv Q K V sp sc sqz sqh sqm sqk skz skh skn skk svz svh svk svn
      N_HEAD H BM BN BD numKV 0 sp := by
    refine alInv_zero Q K V sp sc sqz sqh sqm sqk skz skh skn skk svz svh svk svn
      N_HEAD H BM BN BD numKV hmp hlp hacc ?_ ?_ hoffn ?_ ?_ ?_ ?_ hspundef
    · rw [hqreg, htileQ]
    · rw [hoffm, hsppids]
    · rw [hbi, hsppids]
    · rw [hoh, hsppids]
    · rw [hkp, hbaseK]
    · rw [hvp, hbaseV]
  -- the streaming loop
  obtain ⟨final, sL, hloop, hfin, hinvL⟩ :=
    forRangeDyn_inv (idx := "start_n") (startOp := Op.constNat 0)
      (stopOp := Op.ref .nat [] "block_n_end")
      (stepOp := Op.constNat BN)
      (P := fun i st => alInv Q K V sp sc sqz sqh sqm sqk skz skh skn skk svz svh svk svn
        N_HEAD H BM BN BD numKV i st)
      (s_init := sp)
      (by rw [evalOp_constNat])
      (by rw [evalOp_ref, hbne])
      (by rw [evalOp_constNat])
      (by omega)
      hinv0
      (fun i st hi hP => al_attn_step Q K V sp sc sqz sqh sqm sqk skz skh skn skk svz svh svk svn
        N_HEAD H BM BN BD numKV hBM hBN i st hi hP)
  rw [stepStmts.cons_some hloop]
  -- final counter = BN * numKV
  simp only [alInv] at hinvL
  obtain ⟨hpidsL, _hmodL, hleL, _hmpL, _hlpL, haccL, _hqL, _hoffmL, _hoffnL, hbiL, hohL,
    _hkpL, _hvpL, _hundefL, _hmemL⟩ := hinvL
  have hfinal : final = BN * numKV := by omega
  subst hfinal
  rw [Nat.mul_div_cancel_left _ hBN] at haccL
  -- postLoop
  obtain ⟨sP, hpost, hread⟩ :=
    alPostLoop_run Out sp soz soh som son N_HEAD H BM BD sL
      (fun idx : TileIndex [BM, BD] =>
        oPartial (alQTileG sp Q N_HEAD sqz sqh sqm sqk H BM BD) numKV
            (alKTileG sp K N_HEAD skz skh skn skk (BN * numKV) BD)
            (alVTileG sp V N_HEAD svz svh svk svn (BN * numKV) BD) sc numKV idx
          / lPartial (alQTileG sp Q N_HEAD sqz sqh sqm sqk H BM BD) numKV
            (alKTileG sp K N_HEAD skz skh skn skk (BN * numKV) BD) sc numKV idx.1)
      (by simpa only [hsppids] using houtinj) hpidsL haccL hbiL hohL
  refine ⟨sP, hpost, ?_⟩
  intro idx hActive
  have hActive' : alActive sp H BM idx := by
    simpa only [alActive, alRow, hsppids] using hActive
  have hoff : alOutOffset sp N_HEAD soz soh som son BM idx
      = alOutOffset s N_HEAD soz soh som son BM idx := by
    simp only [alOutOffset, alRow, alBase, hsppids]
  have h := hread idx hActive'
  rw [hoff, htileQ, htileK, htileV] at h
  exact h

set_option maxHeartbeats 1600000 in
open StreamingAccumulator in
/-- **★ MAIN (non-causal).** Every ACTIVE `Out` cell (query row `< H`, the
kernel's own masked store) holds the FULL natural-exp softmax attention
over the `N_CTX` keys at scale `sm_scale`, read from input memory through
the kernel's own offset expressions: the H-guarded Q tile (`alQTileG` —
the guard is the kernel's own `offs_m < H` load mask, true on every
active row) against the K/V tiles at the batch/head base. Side
conditions: `N_CTX = BLOCK_N · numKVBlocks` (the ghost-lane divisibility;
see the header), `0 < numKVBlocks`, output-offset injectivity, and the
clean-undef carrier (which also pins the k/v masked loads' dead lanes to
the Python's positional `other=0.`). -/
specification attention_llama_fwd_closed_form_correct
    (Q K V Out : RegionName) (s : BlockState) (sm_scale : ℝ)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      N_HEAD H N_CTX start_position BLOCK_M BLOCK_N BLOCK_DMODEL numKVBlocks : Nat)
    (hBM : 0 < BLOCK_M) (hBN : 0 < BLOCK_N)
    (hSEQ : N_CTX = BLOCK_N * numKVBlocks) (hnum : 0 < numKVBlocks)
    (houtinj : Function.Injective (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        (s.pids 0 * BLOCK_M + idx.1.val) * stride_om + idx.2.1.val * stride_on))
    (hundef : ∀ rg o, s.undef rg o = 0) :
    (∃ alg, (attention_llama_fwd_surface Q K V Out sm_scale
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      N_HEAD H N_CTX start_position BLOCK_M BLOCK_N BLOCK_DMODEL).toAlgorithm?
        = Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := attention_llama_fwd_surface Q K V Out sm_scale
        stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
        stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
        N_HEAD H N_CTX start_position BLOCK_M BLOCK_N BLOCK_DMODEL)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => alActive s H BLOCK_M idx)
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          (Out, alOutOffset s N_HEAD stride_oz stride_oh stride_om stride_on BLOCK_M idx)))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        attentionReal
          (alQTileG s Q N_HEAD stride_qz stride_qh stride_qm stride_qk H BLOCK_M BLOCK_DMODEL)
          (alKTileG s K N_HEAD stride_kz stride_kh stride_kn stride_kk N_CTX BLOCK_DMODEL)
          (alVTileG s V N_HEAD stride_vz stride_vh stride_vk stride_vn N_CTX BLOCK_DMODEL)
          sm_scale idx) := by
  subst hSEQ
  obtain ⟨sF, hstep, hO⟩ :=
    al_exec Q K V Out s sm_scale stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on N_HEAD H start_position
      BLOCK_M BLOCK_N BLOCK_DMODEL numKVBlocks hBM hBN houtinj hundef
  refine ⟨?_, ?_⟩
  · exact attention_llama_fwd_surface_toAlgorithm_supported Q K V Out sm_scale
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      N_HEAD H (BLOCK_N * numKVBlocks) start_position BLOCK_M BLOCK_N BLOCK_DMODEL
  · rw [ComputeCorrect.realizes_writeIf_iff]
    apply ComputeKernel.computeCorrect_of_toAlgKernel
    · simp [attention_llama_fwd_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
    intro s0 s' hExec hs0
    subst s0
    intro idx hActive
    rw [hstep] at hExec
    obtain rfl : sF = s' := Option.some.inj hExec
    simp only [ComputeCorrect.OutputReadable.read_real]
    rw [hO idx hActive]
    set qT := alQTileG s Q N_HEAD stride_qz stride_qh stride_qm stride_qk H BLOCK_M BLOCK_DMODEL
    set kT := alKTileG s K N_HEAD stride_kz stride_kh stride_kn stride_kk
      (BLOCK_N * numKVBlocks) BLOCK_DMODEL
    set vT := alVTileG s V N_HEAD stride_vz stride_vh stride_vk stride_vn
      (BLOCK_N * numKVBlocks) BLOCK_DMODEL
    have hl : lPartial qT numKVBlocks kT sm_scale numKVBlocks idx.1 ≠ 0 := by
      obtain ⟨k, rfl⟩ : ∃ k, numKVBlocks = k + 1 := ⟨numKVBlocks - 1, by omega⟩
      exact al_lPartial_succ_ne_zero hBN qT kT sm_scale k (le_refl _) idx.1
    exact streaming_eq_attentionReal hBN qT numKVBlocks hnum kT vT sm_scale idx hl

/-! ## Causal-arm streaming bridges (`FA1MathCausal` layer)

The causal arm binds the running registers to the `FA1MathCausal`
accumulators at `qStart = pid₀·BLOCK_M − start_position`: the kernel's
`offs_m ≥ block_n_offs + start_position` predicate is
`c·BN + j + SP ≤ row`, i.e. `j_global ≤ qStart + i` under
`SP ≤ pid₀·BLOCK_M` (`hsp`). -/

open VeriTile.Examples.FA1MathCausal in
/-- Causal masked qk cell: the kernel's shifted predicate matches
`maskedScore (qRow0 − SP)` at the block-`c` global key index. -/
theorem alc_qk1_cell (BM BN BD numC : Nat)
    (qT : TileIndex [BM, BD] → ℝ) (kT : TileIndex [BN * numC, BD] → ℝ) (sc : ℝ)
    (qRow0 SP : Nat) (hsp : SP ≤ qRow0)
    (c : Nat) (hc : c < numC) (r : Fin BM) (j : Fin BN) :
    (if c * BN + j.val + SP ≤ qRow0 + r.val then
        (Tile.bop NumericDType.real.mul Broadcast.scalarR
          (Tile.bop NumericDType.real.add Broadcast.nil.consSame.consSame
            (⟨fun _ : TileIndex [BM, BN] => some (0 : ℝ)⟩ : Tile .real [BM, BN])
            (Tile.dot [] (⟨fun idx => some (qT idx)⟩ : Tile .real [BM, BD])
              (Tile.transpose [] (⟨fun idx : TileIndex [BN, BD] =>
                some (kT (StreamingAccumulator.blockIndex BN numC c (Nat.succ_le_iff.mpr hc) idx.1, idx.2.1, PUnit.unit))⟩
                  : Tile .real [BN, BD]))))
          (Tile.scalar (some sc : WithBot ℝ))).data (r, j, PUnit.unit)
      else (⊥ : WithBot ℝ))
      = maskedScore (qRow0 - SP) qT kT sc r
          (StreamingAccumulator.blockIndex BN numC c (Nat.succ_le_iff.mpr hc) j) := by
  set jg := StreamingAccumulator.blockIndex BN numC c (Nat.succ_le_iff.mpr hc) j with hjg
  have hjgval : jg.val = c * BN + j.val := rfl
  by_cases h : c * BN + j.val + SP ≤ qRow0 + r.val
  · rw [if_pos h]
    rw [al_qk0_cell BM BN BD numC qT kT sc c hc r j]
    rw [maskedScore_of_le (qRow0 - SP) qT kT sc r jg (by rw [hjgval]; omega)]
    rfl
  · rw [if_neg h, maskedScore_of_not_le (qRow0 - SP) qT kT sc r jg (by rw [hjgval]; omega)]

open VeriTile.Examples.FA1MathCausal in
/-- Causal `pexp` cell `exp(qk1(r,j) − mNew r)`. -/
theorem alc_pexp_cell (BM BN BD numC : Nat)
    (qT : TileIndex [BM, BD] → ℝ) (kT : TileIndex [BN * numC, BD] → ℝ) (sc : ℝ)
    (qRow0 SP : Nat) (hsp : SP ≤ qRow0)
    (c : Nat) (hc : c < numC) (mNew : Fin BM → WithBot ℝ) (r : Fin BM) (j : Fin BN) :
    WithBot.realExp (WithBot.realSub
        (if c * BN + j.val + SP ≤ qRow0 + r.val then
            (Tile.bop NumericDType.real.mul Broadcast.scalarR
              (Tile.bop NumericDType.real.add Broadcast.nil.consSame.consSame
                (⟨fun _ : TileIndex [BM, BN] => some (0 : ℝ)⟩ : Tile .real [BM, BN])
                (Tile.dot [] (⟨fun idx => some (qT idx)⟩ : Tile .real [BM, BD])
                  (Tile.transpose [] (⟨fun idx : TileIndex [BN, BD] =>
                    some (kT (StreamingAccumulator.blockIndex BN numC c (Nat.succ_le_iff.mpr hc) idx.1, idx.2.1, PUnit.unit))⟩
                      : Tile .real [BN, BD]))))
              (Tile.scalar (some sc : WithBot ℝ))).data (r, j, PUnit.unit)
          else (⊥ : WithBot ℝ))
        (mNew r))
      = some ((WithBot.realExp (Option.map₂ (fun x y : ℝ => x - y)
          (maskedScore (qRow0 - SP) qT kT sc r
            (StreamingAccumulator.blockIndex BN numC c (Nat.succ_le_iff.mpr hc) j))
          (mNew r))).unbotD 0) := by
  rw [alc_qk1_cell BM BN BD numC qT kT sc qRow0 SP hsp c hc r j]
  exact realExp_eq_some_unbotD _

set_option maxHeartbeats 1600000 in
open VeriTile.Examples.FA1MathCausal in
/-- Causal masked exp block-sum cell `Σ_j pexp(r,j)`. -/
theorem alc_pexp_block_sum (BM BN BD numC : Nat)
    (qT : TileIndex [BM, BD] → ℝ) (kT : TileIndex [BN * numC, BD] → ℝ) (sc : ℝ)
    (qRow0 SP : Nat) (hsp : SP ≤ qRow0)
    (c : Nat) (hc : c < numC) (mNew : Fin BM → WithBot ℝ) (r : Fin BM) :
    (Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [BM, BN].length)
        (Tile.uop WithBot.realExp
          (Tile.bop NumericDType.real.sub Broadcast.nil.consR.consSame
            (⟨fun idx : TileIndex [BM, BN] =>
              if c * BN + idx.2.1.val + SP ≤ qRow0 + idx.1.val then
                (Tile.bop NumericDType.real.mul Broadcast.scalarR
                  (Tile.bop NumericDType.real.add Broadcast.nil.consSame.consSame
                    (⟨fun _ : TileIndex [BM, BN] => some (0 : ℝ)⟩ : Tile .real [BM, BN])
                    (Tile.dot [] (⟨fun idx => some (qT idx)⟩ : Tile .real [BM, BD])
                      (Tile.transpose [] (⟨fun idx : TileIndex [BN, BD] =>
                        some (kT (StreamingAccumulator.blockIndex BN numC c (Nat.succ_le_iff.mpr hc) idx.1, idx.2.1, PUnit.unit))⟩
                          : Tile .real [BN, BD]))))
                  (Tile.scalar (some sc : WithBot ℝ))).data idx
              else (⊥ : WithBot ℝ)⟩ : Tile .real [BM, BN])
            (Tile.expandDim ⟨1, by simp⟩
              (⟨fun r : TileIndex [BM] => mNew r.1⟩ : Tile .real [BM]))))).data
        (r, PUnit.unit)
      = some ((Finset.univ : Finset (Fin BN)).sum (fun jLocal =>
          (WithBot.realExp (Option.map₂ (fun x y : ℝ => x - y)
            (maskedScore (qRow0 - SP) qT kT sc r
              (StreamingAccumulator.blockIndex BN numC c (Nat.succ_le_iff.mpr hc) jLocal))
            (mNew r))).unbotD 0)) := by
  rw [Tile.reduceSumDrop_data]
  have hcell : ∀ k : Fin (TileShape.axisDim [BM, BN] (⟨1, by simp⟩ : Fin [BM, BN].length)),
      (Tile.uop WithBot.realExp
        (Tile.bop NumericDType.real.sub Broadcast.nil.consR.consSame
          (⟨fun idx : TileIndex [BM, BN] =>
            if c * BN + idx.2.1.val + SP ≤ qRow0 + idx.1.val then
              (Tile.bop NumericDType.real.mul Broadcast.scalarR
                (Tile.bop NumericDType.real.add Broadcast.nil.consSame.consSame
                  (⟨fun _ : TileIndex [BM, BN] => some (0 : ℝ)⟩ : Tile .real [BM, BN])
                  (Tile.dot [] (⟨fun idx => some (qT idx)⟩ : Tile .real [BM, BD])
                    (Tile.transpose [] (⟨fun idx : TileIndex [BN, BD] =>
                      some (kT (StreamingAccumulator.blockIndex BN numC c (Nat.succ_le_iff.mpr hc) idx.1, idx.2.1, PUnit.unit))⟩
                        : Tile .real [BN, BD]))))
                (Tile.scalar (some sc : WithBot ℝ))).data idx
            else (⊥ : WithBot ℝ)⟩ : Tile .real [BM, BN])
          (Tile.expandDim ⟨1, by simp⟩
            (⟨fun r : TileIndex [BM] => mNew r.1⟩ : Tile .real [BM])))).data
          (TileShape.insertAxisIndex [BM, BN] (⟨1, by simp⟩ : Fin [BM, BN].length) (r, PUnit.unit) k)
        = (some ((WithBot.realExp (Option.map₂ (fun x y : ℝ => x - y)
            (maskedScore (qRow0 - SP) qT kT sc r
              (StreamingAccumulator.blockIndex BN numC c (Nat.succ_le_iff.mpr hc) k))
            (mNew r))).unbotD 0) : WithBot ℝ) := by
    intro k
    rw [show (TileShape.insertAxisIndex [BM, BN] (⟨1, by simp⟩ : Fin [BM, BN].length)
          (r, PUnit.unit) k) = (r, k, PUnit.unit) from rfl]
    rw [Tile.uop_data, Tile.bop_data]
    simp only [Broadcast.leftIndex, Broadcast.rightIndex, Tile.expandDim_data,
      TileShape.dropInsertedIndex, NumericDType.sub]
    exact alc_pexp_cell BM BN BD numC qT kT sc qRow0 SP hsp c hc mNew r k
  rw [Finset.sum_congr rfl (fun k _ => hcell k)]
  rw [WithBot.sum_someTerm_eq_some]
  rfl

set_option maxHeartbeats 1600000 in
open VeriTile.Examples.FA1MathCausal in
/-- Causal `p·v` dot cell. -/
theorem alc_pv_dot_block (BM BN BD numC : Nat)
    (qT : TileIndex [BM, BD] → ℝ) (kT vT : TileIndex [BN * numC, BD] → ℝ) (sc : ℝ)
    (qRow0 SP : Nat) (hsp : SP ≤ qRow0)
    (c : Nat) (hc : c < numC) (mNew : Fin BM → WithBot ℝ)
    (r : Fin BM) (d : Fin BD) (lrcpT : Tile .real [BM]) (lrcpVal : ℝ)
    (hlrcp : lrcpT.data (r, PUnit.unit) = some lrcpVal) :
    (Tile.dot []
        (Tile.bop NumericDType.real.mul Broadcast.nil.consR.consSame
          (Tile.uop WithBot.realExp
            (Tile.bop NumericDType.real.sub Broadcast.nil.consR.consSame
              (⟨fun idx : TileIndex [BM, BN] =>
                if c * BN + idx.2.1.val + SP ≤ qRow0 + idx.1.val then
                  (Tile.bop NumericDType.real.mul Broadcast.scalarR
                    (Tile.bop NumericDType.real.add Broadcast.nil.consSame.consSame
                      (⟨fun _ : TileIndex [BM, BN] => some (0 : ℝ)⟩ : Tile .real [BM, BN])
                      (Tile.dot [] (⟨fun idx => some (qT idx)⟩ : Tile .real [BM, BD])
                        (Tile.transpose [] (⟨fun idx : TileIndex [BN, BD] =>
                          some (kT (StreamingAccumulator.blockIndex BN numC c (Nat.succ_le_iff.mpr hc) idx.1, idx.2.1, PUnit.unit))⟩
                            : Tile .real [BN, BD]))))
                    (Tile.scalar (some sc : WithBot ℝ))).data idx
                else (⊥ : WithBot ℝ)⟩ : Tile .real [BM, BN])
              (Tile.expandDim ⟨1, by simp⟩
                (⟨fun r : TileIndex [BM] => mNew r.1⟩ : Tile .real [BM]))))
          (Tile.expandDim ⟨1, by simp⟩ lrcpT))
        (⟨fun idx : TileIndex [BN, BD] =>
          some (vT (StreamingAccumulator.blockIndex BN numC c (Nat.succ_le_iff.mpr hc) idx.1, idx.2.1, PUnit.unit))⟩
            : Tile .real [BN, BD])).data (r, d, PUnit.unit)
      = some (lrcpVal * (Finset.univ : Finset (Fin BN)).sum (fun jLocal =>
          (WithBot.realExp (Option.map₂ (fun x y : ℝ => x - y)
            (maskedScore (qRow0 - SP) qT kT sc r
              (StreamingAccumulator.blockIndex BN numC c (Nat.succ_le_iff.mpr hc) jLocal))
            (mNew r))).unbotD 0
            * vT (StreamingAccumulator.blockIndex BN numC c (Nat.succ_le_iff.mpr hc) jLocal, d, PUnit.unit))) := by
  rw [Tile.dot_nil_data]
  refine Eq.trans (b := @Finset.sum (Fin BN) (WithBot ℝ) _ Finset.univ
      (fun j => (some (lrcpVal *
        ((WithBot.realExp (Option.map₂ (fun x y : ℝ => x - y)
          (maskedScore (qRow0 - SP) qT kT sc r
            (StreamingAccumulator.blockIndex BN numC c (Nat.succ_le_iff.mpr hc) j))
          (mNew r))).unbotD 0
          * vT (StreamingAccumulator.blockIndex BN numC c (Nat.succ_le_iff.mpr hc) j, d, PUnit.unit))) : WithBot ℝ)))
    (Finset.sum_congr rfl (fun j (_ : j ∈ Finset.univ) => ?_)) ?_
  · rw [Tile.bop_data]
    simp only [Broadcast.leftIndex, Broadcast.rightIndex, Tile.expandDim_data,
      TileShape.dropInsertedIndex, NumericDType.mul]
    rw [hlrcp]
    rw [Tile.uop_data, Tile.bop_data]
    simp only [Broadcast.leftIndex, Broadcast.rightIndex, Tile.expandDim_data,
      TileShape.dropInsertedIndex, NumericDType.sub]
    rw [alc_pexp_cell BM BN BD numC qT kT sc qRow0 SP hsp c hc mNew r j]
    simp only [WithBot.realMul, Option.map₂_some_some]
    refine congrArg some ?_
    ring
  · rw [WithBot.sum_someTerm_eq_some, ← Finset.mul_sum]

open VeriTile.Examples.FA1MathCausal in
/-- If the causal m-free normalizer over the first `k` blocks is zero, so is
the m-free output accumulator (a zero normalizer forces every causal
indicator to zero). -/
theorem alc_lFree_zero_imp_oFree_zero {M D Bk N : Nat}
    (qStart : Nat) (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [Bk * N, D] → ℝ)
    (scale : ℝ) (k : Nat) (hk : k ≤ N) (idx : TileIndex [M, D])
    (h0 : lFree qStart Q K scale k hk idx.1 = 0) :
    oFree qStart Q K V scale k hk idx = 0 := by
  induction k with
  | zero => exact oFree_zero qStart Q K V scale idx
  | succ k ih =>
      have hk' : k ≤ N := Nat.le_of_succ_le hk
      rw [lFree_succ qStart Q K scale k hk idx.1] at h0
      have hterm_nonneg : ∀ jL : Fin Bk,
          0 ≤ (if (StreamingAccumulator.blockIndex Bk N k hk jL).val ≤ qStart + idx.1.val then
                Real.exp (StreamingAccumulator.scaledScore Q K scale idx.1
                  (StreamingAccumulator.blockIndex Bk N k hk jL))
              else 0) := by
        intro jL
        by_cases hv : (StreamingAccumulator.blockIndex Bk N k hk jL).val ≤ qStart + idx.1.val
        · simp [hv, le_of_lt (Real.exp_pos _)]
        · simp [hv]
      have hblock_nonneg : 0 ≤ Finset.univ.sum (fun jL : Fin Bk =>
            if (StreamingAccumulator.blockIndex Bk N k hk jL).val ≤ qStart + idx.1.val then
              Real.exp (StreamingAccumulator.scaledScore Q K scale idx.1
                (StreamingAccumulator.blockIndex Bk N k hk jL))
            else 0) :=
        Finset.sum_nonneg (fun jL _ => hterm_nonneg jL)
      have hlfree_nonneg : 0 ≤ lFree qStart Q K scale k hk' idx.1 := by
        unfold lFree
        exact Finset.sum_nonneg (fun n _ => Finset.sum_nonneg (fun jL _ => by
          by_cases hv : (StreamingAccumulator.blockIndex Bk N n.val (Nat.lt_of_lt_of_le n.isLt hk') jL).val ≤ qStart + idx.1.val
          · simp [hv, le_of_lt (Real.exp_pos _)]
          · simp [hv]))
      have hlf0 : lFree qStart Q K scale k hk' idx.1 = 0 := by linarith
      have hblock0 : Finset.univ.sum (fun jL : Fin Bk =>
            if (StreamingAccumulator.blockIndex Bk N k hk jL).val ≤ qStart + idx.1.val then
              Real.exp (StreamingAccumulator.scaledScore Q K scale idx.1
                (StreamingAccumulator.blockIndex Bk N k hk jL))
            else 0) = 0 := by linarith
      have hterm0 : ∀ jL : Fin Bk,
          (if (StreamingAccumulator.blockIndex Bk N k hk jL).val ≤ qStart + idx.1.val then
            Real.exp (StreamingAccumulator.scaledScore Q K scale idx.1
              (StreamingAccumulator.blockIndex Bk N k hk jL))
          else 0) = 0 := fun jL =>
        (Finset.sum_eq_zero_iff_of_nonneg (fun jL' _ => hterm_nonneg jL')).mp hblock0 jL
          (Finset.mem_univ _)
      rw [oFree_succ qStart Q K V scale k hk idx, ih hk' hlf0]
      rw [zero_add]
      refine Finset.sum_eq_zero (fun jL _ => ?_)
      have := hterm0 jL
      by_cases hv : (StreamingAccumulator.blockIndex Bk N k hk jL).val ≤ qStart + idx.1.val
      · exfalso
        rw [if_pos hv] at this
        exact (Real.exp_ne_zero _) this
      · simp [hv]

open VeriTile.Examples.FA1MathCausal in
/-- **Acc-rescale cancel (causal).**
`(oPartial c / lPartial c) · lPartial c = oPartial c`, valid also when
`lPartial c = 0` (then `oPartial c = 0` too). -/
theorem alc_oPartial_div_lPartial_cancel {M D Bk : Nat} (hBk : 0 < Bk)
    (qStart : Nat) (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ)
    (c : Nat) (hc : c ≤ numKVBlocks) (idx : TileIndex [M, D]) :
    (oPartial Bk qStart Q numKVBlocks K V scale c idx
        / lPartial Bk qStart Q numKVBlocks K scale c idx.1)
      * lPartial Bk qStart Q numKVBlocks K scale c idx.1
      = oPartial Bk qStart Q numKVBlocks K V scale c idx := by
  by_cases hl : lPartial Bk qStart Q numKVBlocks K scale c idx.1 = 0
  · rw [hl, mul_zero]
    rw [lPartial_eq_mShifted hBk qStart Q numKVBlocks K scale c hc idx.1] at hl
    have hlFree0 : lFree qStart Q K scale c hc idx.1 = 0 :=
      (mul_eq_zero.mp hl).resolve_left (Real.exp_ne_zero _)
    rw [oPartial_eq_mShifted hBk qStart Q numKVBlocks K V scale c hc idx,
        alc_lFree_zero_imp_oFree_zero qStart Q K V scale c hc idx hlFree0, mul_zero]
  · rw [div_mul_cancel₀ _ hl]

/-! ## Causal invariant + step -/

open VeriTile.Examples.FA1MathCausal in
/-- **Causal loop invariant.** Counter `i = c·BLOCK_N` over `numC` causal
blocks (`span = BLOCK_N·numC = (pid₀+1)·BLOCK_N + start_position`). The
running registers bind to the `FA1MathCausal` accumulators at
`qStart = pid₀·BLOCK_M − start_position`. -/
noncomputable def alcInv (Q K V : RegionName) (s0 : BlockState) (sc : ℝ)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn
      N_HEAD H SP BM BN BD numC : Nat) (i : Nat) (st : BlockState) : Prop :=
  let qS := s0.pids 0 * BM - SP
  let qT := alQTileG s0 Q N_HEAD sqz sqh sqm sqk H BM BD
  let kT := alKTileG s0 K N_HEAD skz skh skn skk (BN * numC) BD
  let vT := alVTileG s0 V N_HEAD svz svh svk svn (BN * numC) BD
  st.pids = s0.pids ∧ i % BN = 0 ∧ i ≤ BN * numC ∧
  (st.regs .real [BM] "m_prev" = some ⟨fun r : TileIndex [BM] =>
      mPartial BN qS qT numC kT sc (i / BN) r.1⟩) ∧
  (st.regs .real [BM] "l_prev" = some ⟨fun r : TileIndex [BM] =>
      ((lPartial BN qS qT numC kT sc (i / BN) r.1 : ℝ) : WithBot ℝ)⟩) ∧
  (st.regs .real [BM, BD] "acc" = some ⟨fun idx : TileIndex [BM, BD] =>
      ((oPartial BN qS qT numC kT vT sc (i / BN) idx
          / lPartial BN qS qT numC kT sc (i / BN) idx.1 : ℝ) : WithBot ℝ)⟩) ∧
  (st.regs .real [BM, BD] "q" = some ⟨fun idx : TileIndex [BM, BD] => some (qT idx)⟩) ∧
  (st.regs .nat [BM] "offs_m" = some (Tile.vec (fun r : Fin BM => s0.pids 0 * BM + r.val))) ∧
  (st.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => j.val))) ∧
  (st.regs .nat [] "batch_id" = some (Tile.scalar (s0.pids 1 / N_HEAD))) ∧
  (st.regs .nat [] "off_hz" = some (Tile.scalar (s0.pids 1 % N_HEAD))) ∧
  (st.regs .ptr [BN, BD] "k_ptrs"
    = some (alPtrTile K (alBase s0 N_HEAD skz skh) skn skk i BN BD)) ∧
  (st.regs .ptr [BN, BD] "v_ptrs"
    = some (alPtrTile V (alBase s0 N_HEAD svz svh) svk svn i BN BD)) ∧
  (∀ rg o, st.undef rg o = 0) ∧ (st.mem = s0.mem)

open VeriTile.Examples.FA1MathCausal in
/-- **Causal invariant base case** at `c = 0`. -/
theorem alcInv_zero (Q K V : RegionName) (sp : BlockState) (sc : ℝ)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn
      N_HEAD H SP BM BN BD numC : Nat)
    (hm : sp.regs .real [BM] "m_prev" = some ⟨fun _ : TileIndex [BM] => (⊥ : WithBot ℝ)⟩)
    (hl : sp.regs .real [BM] "l_prev" = some ⟨fun _ : TileIndex [BM] => some (0 : ℝ)⟩)
    (hacc : sp.regs .real [BM, BD] "acc" = some ⟨fun _ : TileIndex [BM, BD] => some (0 : ℝ)⟩)
    (hq : sp.regs .real [BM, BD] "q" = some ⟨fun idx : TileIndex [BM, BD] =>
        some (alQTileG sp Q N_HEAD sqz sqh sqm sqk H BM BD idx)⟩)
    (hoffm : sp.regs .nat [BM] "offs_m" = some (Tile.vec (fun r : Fin BM => sp.pids 0 * BM + r.val)))
    (hoffn : sp.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => j.val)))
    (hbi : sp.regs .nat [] "batch_id" = some (Tile.scalar (sp.pids 1 / N_HEAD)))
    (hoh : sp.regs .nat [] "off_hz" = some (Tile.scalar (sp.pids 1 % N_HEAD)))
    (hkp : sp.regs .ptr [BN, BD] "k_ptrs"
      = some (alPtrTile K (alBase sp N_HEAD skz skh) skn skk 0 BN BD))
    (hvp : sp.regs .ptr [BN, BD] "v_ptrs"
      = some (alPtrTile V (alBase sp N_HEAD svz svh) svk svn 0 BN BD))
    (hundef : ∀ rg o, sp.undef rg o = 0) :
    alcInv Q K V sp sc sqz sqh sqm sqk skz skh skn skk svz svh svk svn
      N_HEAD H SP BM BN BD numC 0 sp := by
  unfold alcInv
  refine ⟨rfl, by simp, by simp, ?_, ?_, ?_, hq, hoffm, hoffn, hbi, hoh, hkp, hvp, hundef, rfl⟩
  · rw [hm]; refine congrArg some ?_; ext r
    show (⊥ : WithBot ℝ) = mPartial BN _ _ numC _ sc (0 / BN) r.1
    rw [Nat.zero_div]; rfl
  · rw [hl]; refine congrArg some ?_; ext r
    show some (0 : ℝ) = ((lPartial BN _ _ numC _ sc (0 / BN) r.1 : ℝ) : WithBot ℝ)
    rw [Nat.zero_div]; rfl
  · rw [hacc]; refine congrArg some ?_; ext idx
    show some (0 : ℝ) = ((oPartial BN (sp.pids 0 * BM - SP) (alQTileG sp Q N_HEAD sqz sqh sqm sqk H BM BD) numC
        (alKTileG sp K N_HEAD skz skh skn skk (BN * numC) BD)
        (alVTileG sp V N_HEAD svz svh svk svn (BN * numC) BD) sc (0 / BN) idx
        / lPartial BN (sp.pids 0 * BM - SP) (alQTileG sp Q N_HEAD sqz sqh sqm sqk H BM BD) numC
          (alKTileG sp K N_HEAD skz skh skn skk (BN * numC) BD) sc (0 / BN) idx.1 : ℝ) : WithBot ℝ)
    rw [Nat.zero_div]
    show some (0 : ℝ) = (((0 : ℝ) / (0 : ℝ) : ℝ) : WithBot ℝ)
    rw [div_zero]
    rfl

set_option maxHeartbeats 8000000 in
set_option maxRecDepth 8000 in
open VeriTile.Examples.FA1MathCausal in
/-- **Causal loop step.** Advancing the counter from `i = c·BLOCK_N` to
`i + BLOCK_N` carries the `FA1MathCausal` running registers from block
count `c` to `c+1` (prefix → causal `where` → shared tail). -/
theorem alc_attn_step (Q K V : RegionName) (s0 : BlockState) (sc : ℝ)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn
      N_HEAD H N_CTX SP BM BN BD numC : Nat)
    (_hBM : 0 < BM) (hBN : 0 < BN)
    (hspanle : BN * numC ≤ N_CTX)
    (hsp : SP ≤ s0.pids 0 * BM)
    (i : Nat) (st : BlockState) (hilt : i < BN * numC)
    (hinv : alcInv Q K V s0 sc sqz sqh sqm sqk skz skh skn skk svz svh svk svn
      N_HEAD H SP BM BN BD numC i st) :
    ∃ s', stepStmts (alLoopBodyCausalG sc skn svk N_CTX SP BM BN BD)
        (st.setReg "start_n" .nat [] (Tile.scalar i)) = some s'
      ∧ alcInv Q K V s0 sc sqz sqh sqm sqk skz skh skn skk svz svh svk svn
          N_HEAD H SP BM BN BD numC (i + BN) s' := by
  simp only [alcInv] at hinv
  obtain ⟨hpids, hmod, hile, hmp, hlp, hacc, hq, hoffm, hoffn, hbi, hoh, hkp, hvp, hundef, hmem⟩ := hinv
  set c := i / BN with hcdef
  have hic : i = c * BN := by
    rw [hcdef, Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hmod)]
  have hclt : c < numC := by
    rw [hcdef]
    exact (Nat.div_lt_iff_lt_mul hBN).mpr (by rw [Nat.mul_comm numC BN]; exact hilt)
  have hcle : c ≤ numC := le_of_lt hclt
  have hc1 : (i + BN) / BN = c + 1 := by
    rw [hcdef, hic, Nat.add_div_right _ hBN, Nat.mul_div_cancel _ hBN]
  set qS := s0.pids 0 * BM - SP with hqS
  set qT := alQTileG s0 Q N_HEAD sqz sqh sqm sqk H BM BD with hqT
  set kT := alKTileG s0 K N_HEAD skz skh skn skk (BN * numC) BD with hkT
  set vT := alVTileG s0 V N_HEAD svz svh svk svn (BN * numC) BD with hvT
  set se := st.setReg "start_n" .nat [] (Tile.scalar i) with hse
  set kTfn : TileIndex [BN, BD] → ℝ :=
    (fun idx => kT (StreamingAccumulator.blockIndex BN numC c (Nat.succ_le_iff.mpr hclt) idx.1, idx.2.1, PUnit.unit)) with hkTfn
  set vTfn : TileIndex [BN, BD] → ℝ :=
    (fun idx => vT (StreamingAccumulator.blockIndex BN numC c (Nat.succ_le_iff.mpr hclt) idx.1, idx.2.1, PUnit.unit)) with hvTfn
  have hnumCpos : 0 < numC := lt_of_le_of_lt (Nat.zero_le c) hclt
  have hin : ∀ j : Fin BN, i + j.val < N_CTX := by
    intro j
    have hjv : j.val < BN := j.isLt
    calc i + j.val < i + BN := by omega
      _ = (c + 1) * BN := by rw [hic]; ring
      _ ≤ numC * BN := Nat.mul_le_mul_right _ hclt
      _ = BN * numC := Nat.mul_comm _ _
      _ ≤ N_CTX := hspanle
  have hBNle : BN ≤ N_CTX := by
    calc BN = 1 * BN := (Nat.one_mul _).symm
      _ ≤ numC * BN := Nat.mul_le_mul_right _ hnumCpos
      _ = BN * numC := Nat.mul_comm _ _
      _ ≤ N_CTX := hspanle
  have hmem_se : ∀ (R : RegionName) (o : Nat), se.readMem R o = s0.readMem R o := by
    intro R o; rw [hse]; simp only [BlockState.setReg_readMem]
    unfold BlockState.readMem; rw [hmem]
  have hkload : ∀ idx : TileIndex [BN, BD],
      kTfn idx = se.readMem K (alBase s0 N_HEAD skz skh + (i + idx.1.val) * skn + idx.2.1.val * skk) := by
    intro idx; rw [hmem_se]
    simp only [hkTfn, hkT, alKTileG, hic, StreamingAccumulator.blockIndex]
  have hvload : ∀ idx : TileIndex [BN, BD],
      vTfn idx = se.readMem V (alBase s0 N_HEAD svz svh + (i + idx.1.val) * svk + idx.2.1.val * svn) := by
    intro idx; rw [hmem_se]
    simp only [hvTfn, hvT, alVTileG, hic, StreamingAccumulator.blockIndex]
  -- run the shared prefix
  obtain ⟨sPre, hpre, hprepids, hpremem, hpreundef, hpreqk, hprebn, hpremp, hprelp, hpreacc,
      hpreq, hpreoffm, hpreoffn, hprebi, hpreoh, hprekp, hprevp⟩ :=
    al_loopPrefix_steps sc N_CTX BM BN BD i se K (alBase s0 N_HEAD skz skh) skn skk
      (⟨fun idx => some (qT idx)⟩) kTfn
      hBNle
      hin
      (by rw [hse, BlockState.setReg_same])
      (by rw [hse, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hoffn)
      (by rw [hse, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hq)
      (by rw [hse, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hkp)
      hkload
      (by intro rg o; rw [hse, BlockState.setReg_undef]; exact hundef rg o)
  set qk0 : Tile .real [BM, BN] := Tile.bop NumericDType.real.mul Broadcast.scalarR
      (Tile.bop NumericDType.real.add Broadcast.nil.consSame.consSame
        (⟨fun _ : TileIndex [BM, BN] => some (0 : ℝ)⟩ : Tile .real [BM, BN])
        (Tile.dot [] (⟨fun idx => some (qT idx)⟩ : Tile .real [BM, BD])
          (Tile.transpose [] (⟨fun idx => some (kTfn idx)⟩ : Tile .real [BN, BD]))))
      (Tile.scalar (some sc : WithBot ℝ)) with hqk0def
  set qk1 : Tile .real [BM, BN] := (⟨fun idx : TileIndex [BM, BN] =>
      if i + idx.2.1.val + SP ≤ s0.pids 0 * BM + idx.1.val then qk0.data idx
      else (⊥ : WithBot ℝ)⟩ : Tile .real [BM, BN]) with hqk1def
  -- the causal where evaluation at sPre
  have hwhere : evalOp (Op.where
      (Op.ge ComparableDType.nat Broadcast.nil.consL.consR
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m"))
        (Op.add .nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "block_n_offs")) (Op.constNat SP)))
      (Op.ref .real [BM, BN] "qk") (Op.broadcast Op.negInf [BM, BN])) sPre = some qk1 := by
    rw [al_where_causal_eval sPre BM BN i SP (fun r : Fin BM => s0.pids 0 * BM + r.val) qk0
      (by rw [hpreoffm, hse, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hoffm)
      hprebn
      (by rw [hpreqk])]
  set sW := sPre.setReg "qk" .real [BM, BN] qk1 with hsW
  -- run the shared tail from sW
  obtain ⟨rmaxT, hrm, sF, htail, htailpids, htailmem, htailundef,
      mcurrT, lcurrT, lrcpT, pT, acc1T, hmcd, hpTd, hlcd, hlrd, hacc1d,
      hmF, hlF, haccF, hkpF, hvpF, hqF, hoffmF, hoffnF, hbiF, hohF⟩ :=
    al_loopTail_steps skn svk N_CTX BM BN BD i sW hBN qk1
      (⟨fun r : TileIndex [BM] => mPartial BN qS qT numC kT sc c r.1⟩)
      (⟨fun r : TileIndex [BM] => ((lPartial BN qS qT numC kT sc c r.1 : ℝ) : WithBot ℝ)⟩)
      (⟨fun idx : TileIndex [BM, BD] =>
        ((oPartial BN qS qT numC kT vT sc c idx
            / lPartial BN qS qT numC kT sc c idx.1 : ℝ) : WithBot ℝ)⟩)
      K V (alBase s0 N_HEAD skz skh) (alBase s0 N_HEAD svz svh) skk svn vTfn
      hin
      (by rw [hsW, BlockState.setReg_same])
      (by rw [hsW, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
          exact hprebn)
      (by rw [hsW, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
          rw [hpremp, hse, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hmp)
      (by rw [hsW, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
          rw [hprelp, hse, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hlp)
      (by rw [hsW, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
          rw [hpreacc, hse, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hacc)
      (by rw [hsW, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
          rw [hprekp, hse, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hkp)
      (by rw [hsW, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
          rw [hprevp, hse, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hvp)
      (by intro idx
          rw [show sW.readMem V (alBase s0 N_HEAD svz svh + (i + idx.1.val) * svk + idx.2.1.val * svn)
              = se.readMem V (alBase s0 N_HEAD svz svh + (i + idx.1.val) * svk + idx.2.1.val * svn) from by
            rw [hsW]
            simp only [BlockState.setReg_readMem]
            unfold BlockState.readMem; rw [hpremem]]
          exact hvload idx)
      (by intro rg o; rw [hsW, BlockState.setReg_undef]; exact hpreundef rg o)
  refine ⟨sF, ?_, ?_⟩
  · show stepStmts (alLoopPrefixG sc N_CTX BM BN BD
        ++ alCausalWhereStmt SP BM BN :: alLoopTailG skn svk N_CTX BM BN BD) se = some sF
    rw [stepStmts.append_some hpre]
    rw [show alCausalWhereStmt SP BM BN = Stmt.assign .real [BM, BN] "qk"
        (Op.where
          (Op.ge ComparableDType.nat Broadcast.nil.consL.consR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m"))
            (Op.add .nat Broadcast.scalarR
              (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "block_n_offs")) (Op.constNat SP)))
          (Op.ref .real [BM, BN] "qk") (Op.broadcast Op.negInf [BM, BN])) from rfl]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some hwhere)]
    exact htail
  -- ══ re-establish the invariant at i + BN ══
  have hqk1cell : ∀ (r : Fin BM) (j : Fin BN),
      qk1.data (r, j, PUnit.unit)
        = maskedScore qS qT kT sc r (StreamingAccumulator.blockIndex BN numC c (Nat.succ_le_iff.mpr hclt) j) := by
    intro r j
    show (if i + j.val + SP ≤ s0.pids 0 * BM + r.val then qk0.data (r, j, PUnit.unit)
        else (⊥ : WithBot ℝ)) = _
    rw [hic, hqk0def]
    exact alc_qk1_cell BM BN BD numC qT kT sc (s0.pids 0 * BM) SP hsp c hclt r j
  have hrmaxCell : ∀ r : Fin BM,
      rmaxT.data (r, PUnit.unit)
        = (Finset.univ : Finset (Fin BN)).sup (fun j =>
            maskedScore qS qT kT sc r (StreamingAccumulator.blockIndex BN numC c (Nat.succ_le_iff.mpr hclt) j)) := by
    intro r
    rw [al_rmax_cell hBN qk1 rmaxT hrm r]
    exact Finset.sup_congr rfl (fun j _ => hqk1cell r j)
  have hmcurrCell : ∀ r : Fin BM,
      mcurrT.data (r, PUnit.unit) = mPartial BN qS qT numC kT sc (c + 1) r := by
    intro r
    rw [hmcd, Tile.select_data, Tile.cop_data]
    simp only [Broadcast.leftIndex, Broadcast.rightIndex, ComparableDType.gt, hrmaxCell r]
    rw [mPartial_succ_of_lt qS qT numC kT sc c hclt r]
    by_cases hgt : (Finset.univ : Finset (Fin BN)).sup (fun j =>
        maskedScore qS qT kT sc r (StreamingAccumulator.blockIndex BN numC c (Nat.succ_le_iff.mpr hclt) j))
        > mPartial BN qS qT numC kT sc c r
    · rw [decide_eq_true hgt]
      simp only [if_true]
      rw [max_eq_right (le_of_lt hgt)]
    · rw [decide_eq_false hgt]
      simp only [Bool.false_eq_true, if_false]
      rw [max_eq_left (not_lt.mp hgt)]
  have hmcurrTile : mcurrT = ⟨fun r : TileIndex [BM] => mPartial BN qS qT numC kT sc (c + 1) r.1⟩ := by
    ext r; exact hmcurrCell r.1
  set mNew : Fin BM → WithBot ℝ := fun r => mPartial BN qS qT numC kT sc (c + 1) r with hmNew
  have hqk1canon : qk1 = (⟨fun idx : TileIndex [BM, BN] =>
      if c * BN + idx.2.1.val + SP ≤ s0.pids 0 * BM + idx.1.val then
        (Tile.bop NumericDType.real.mul Broadcast.scalarR
          (Tile.bop NumericDType.real.add Broadcast.nil.consSame.consSame
            (⟨fun _ : TileIndex [BM, BN] => some (0 : ℝ)⟩ : Tile .real [BM, BN])
            (Tile.dot [] (⟨fun idx => some (qT idx)⟩ : Tile .real [BM, BD])
              (Tile.transpose [] (⟨fun idx : TileIndex [BN, BD] =>
                some (kT (StreamingAccumulator.blockIndex BN numC c (Nat.succ_le_iff.mpr hclt) idx.1, idx.2.1, PUnit.unit))⟩
                  : Tile .real [BN, BD]))))
          (Tile.scalar (some sc : WithBot ℝ))).data idx
      else (⊥ : WithBot ℝ)⟩ : Tile .real [BM, BN]) := by
    rw [hqk1def]
    ext idx
    simp only [hqk0def, hic, hkTfn]
  have hlcurrCell : ∀ r : Fin BM,
      lcurrT.data (r, PUnit.unit) = some (lPartial BN qS qT numC kT sc (c + 1) r) := by
    intro r
    rw [hlcd, Tile.bop_data]
    simp only [Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add]
    rw [hqk1canon, hmcurrTile]
    erw [alc_pexp_block_sum BM BN BD numC qT kT sc (s0.pids 0 * BM) SP hsp c hclt mNew r]
    rw [show (Tile.bop NumericDType.real.mul Broadcast.nil.consSame
          (⟨fun r : TileIndex [BM] => ((lPartial BN qS qT numC kT sc c r.1 : ℝ) : WithBot ℝ)⟩ : Tile .real [BM])
          (Tile.uop WithBot.realExp
            (Tile.bop NumericDType.real.sub Broadcast.nil.consSame
              (⟨fun r : TileIndex [BM] => mPartial BN qS qT numC kT sc c r.1⟩ : Tile .real [BM])
              (⟨fun r : TileIndex [BM] => mPartial BN qS qT numC kT sc (c + 1) r.1⟩ : Tile .real [BM])))).data (r, PUnit.unit)
        = some ((WithBot.realExp (Option.map₂ (fun x y : ℝ => x - y)
              (mPartial BN qS qT numC kT sc c r) (mPartial BN qS qT numC kT sc (c + 1) r))).unbotD 0
            * lPartial BN qS qT numC kT sc c r) from by
      rw [Tile.bop_data, Tile.uop_data, Tile.bop_data]
      simp only [Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.mul, NumericDType.sub]
      rw [realExp_eq_some_unbotD (WithBot.realSub (mPartial BN qS qT numC kT sc c r)
            (mPartial BN qS qT numC kT sc (c + 1) r))]
      show WithBot.realMul (some (lPartial BN qS qT numC kT sc c r)) (some _) = some _
      rw [WithBot.realMul, Option.map₂_some_some]
      refine congrArg some ?_
      rw [mul_comm]
      rfl]
    show WithBot.realAdd (some _) (some _) = some _
    rw [WithBot.realAdd, Option.map₂_some_some]
    refine congrArg some ?_
    rw [lPartial_succ_of_lt qS qT numC kT sc c hclt r]
    simp only [hmNew]
    ring
  have hlrcpCell : ∀ r : Fin BM,
      lrcpT.data (r, PUnit.unit) = some (1 / lPartial BN qS qT numC kT sc (c + 1) r) := by
    intro r
    rw [hlrd, Tile.bop_data]
    simp only [Broadcast.leftIndex, Broadcast.rightIndex, Broadcast.scalarL, Tile.scalar_data,
      NumericDType.div]
    rw [hlcurrCell r]
    show WithBot.realDiv (some (1.0 : ℝ)) (some (lPartial BN qS qT numC kT sc (c + 1) r))
      = some (1 / lPartial BN qS qT numC kT sc (c + 1) r)
    rw [WithBot.realDiv, Option.map₂_some_some]
    norm_num
  have haccCell : ∀ idx : TileIndex [BM, BD],
      (Tile.bop NumericDType.real.add Broadcast.nil.consSame.consSame
        acc1T (Tile.dot [] pT (⟨fun idx => some (vTfn idx)⟩ : Tile .real [BN, BD]))).data idx
        = some (oPartial BN qS qT numC kT vT sc (c + 1) idx
            / lPartial BN qS qT numC kT sc (c + 1) idx.1) := by
    intro idx
    obtain ⟨r, d, u⟩ := idx; cases u
    rw [Tile.bop_data]
    simp only [Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add]
    set alphaV : ℝ := (WithBot.realExp (Option.map₂ (fun x y : ℝ => x - y)
        (mPartial BN qS qT numC kT sc c r)
        (mPartial BN qS qT numC kT sc (c + 1) r))).unbotD 0 with halphaV
    have hacc1V : acc1T.data (r, d, PUnit.unit)
        = some (oPartial BN qS qT numC kT vT sc c (r, d, PUnit.unit)
              / lPartial BN qS qT numC kT sc c r
            * (lPartial BN qS qT numC kT sc c r * alphaV
                * (1 / lPartial BN qS qT numC kT sc (c + 1) r))) := by
      rw [hacc1d, Tile.bop_data]
      simp only [Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.mul, NumericDType.sub,
        Tile.expandDim_data, TileShape.dropInsertedIndex, Tile.bop_data, Tile.uop_data]
      rw [hlrcpCell r, hmcurrCell r]
      rw [realExp_eq_some_unbotD (WithBot.realSub (mPartial BN qS qT numC kT sc c r)
            (mPartial BN qS qT numC kT sc (c + 1) r))]
      show WithBot.realMul
          (some (oPartial BN qS qT numC kT vT sc c (r, d, PUnit.unit)
            / lPartial BN qS qT numC kT sc c r))
          (WithBot.realMul
            (WithBot.realMul (some (lPartial BN qS qT numC kT sc c r))
              (some ((WithBot.realExp (WithBot.realSub (mPartial BN qS qT numC kT sc c r)
                  (mPartial BN qS qT numC kT sc (c + 1) r))).unbotD 0)))
            (some (1 / lPartial BN qS qT numC kT sc (c + 1) r))) = _
      simp only [WithBot.realMul, Option.map₂_some_some]
      refine congrArg some ?_
      rw [show (WithBot.realExp (WithBot.realSub (mPartial BN qS qT numC kT sc c r)
              (mPartial BN qS qT numC kT sc (c + 1) r))).unbotD 0 = alphaV from by
        rw [halphaV]; rfl]
    rw [hacc1V]
    rw [hpTd, hqk1canon, hmcurrTile]
    erw [alc_pv_dot_block BM BN BD numC qT kT vT sc (s0.pids 0 * BM) SP hsp c hclt mNew r d lrcpT
      (1 / lPartial BN qS qT numC kT sc (c + 1) r) (hlrcpCell r)]
    rw [show WithBot.realAdd (some (oPartial BN qS qT numC kT vT sc c (r, d, PUnit.unit)
            / lPartial BN qS qT numC kT sc c r
          * (lPartial BN qS qT numC kT sc c r * alphaV
              * (1 / lPartial BN qS qT numC kT sc (c + 1) r))))
        (some (1 / lPartial BN qS qT numC kT sc (c + 1) r
          * (Finset.univ : Finset (Fin BN)).sum (fun jLocal =>
              (WithBot.realExp (Option.map₂ (fun x y : ℝ => x - y)
                (maskedScore (s0.pids 0 * BM - SP) qT kT sc r
                  (StreamingAccumulator.blockIndex BN numC c (Nat.succ_le_iff.mpr hclt) jLocal))
                (mNew r))).unbotD 0
                * vT (StreamingAccumulator.blockIndex BN numC c (Nat.succ_le_iff.mpr hclt) jLocal, d, PUnit.unit))))
      = some (oPartial BN qS qT numC kT vT sc c (r, d, PUnit.unit)
            / lPartial BN qS qT numC kT sc c r
          * (lPartial BN qS qT numC kT sc c r * alphaV
              * (1 / lPartial BN qS qT numC kT sc (c + 1) r))
        + 1 / lPartial BN qS qT numC kT sc (c + 1) r
          * (Finset.univ : Finset (Fin BN)).sum (fun jLocal =>
              (WithBot.realExp (Option.map₂ (fun x y : ℝ => x - y)
                (maskedScore (s0.pids 0 * BM - SP) qT kT sc r
                  (StreamingAccumulator.blockIndex BN numC c (Nat.succ_le_iff.mpr hclt) jLocal))
                (mNew r))).unbotD 0
                * vT (StreamingAccumulator.blockIndex BN numC c (Nat.succ_le_iff.mpr hclt) jLocal, d, PUnit.unit))) from by
      rw [WithBot.realAdd, Option.map₂_some_some]]
    refine congrArg some ?_
    have hcancel := alc_oPartial_div_lPartial_cancel hBN qS qT numC kT vT sc c hcle (r, d, PUnit.unit)
    rw [oPartial_succ_of_lt qS qT numC kT vT sc c hclt (r, d, PUnit.unit)]
    simp only [hmNew, halphaV, hqS]
    rw [show oPartial BN (s0.pids 0 * BM - SP) qT numC kT vT sc c (r, d, PUnit.unit)
            / lPartial BN (s0.pids 0 * BM - SP) qT numC kT sc c r
          * (lPartial BN (s0.pids 0 * BM - SP) qT numC kT sc c r
              * (WithBot.realExp (Option.map₂ (fun x y : ℝ => x - y)
                  (mPartial BN (s0.pids 0 * BM - SP) qT numC kT sc c r)
                  (mPartial BN (s0.pids 0 * BM - SP) qT numC kT sc (c + 1) r))).unbotD 0
              * (1 / lPartial BN (s0.pids 0 * BM - SP) qT numC kT sc (c + 1) r))
        = (oPartial BN (s0.pids 0 * BM - SP) qT numC kT vT sc c (r, d, PUnit.unit)
            / lPartial BN (s0.pids 0 * BM - SP) qT numC kT sc c r
          * lPartial BN (s0.pids 0 * BM - SP) qT numC kT sc c r)
          * ((WithBot.realExp (Option.map₂ (fun x y : ℝ => x - y)
              (mPartial BN (s0.pids 0 * BM - SP) qT numC kT sc c r)
              (mPartial BN (s0.pids 0 * BM - SP) qT numC kT sc (c + 1) r))).unbotD 0
            * (1 / lPartial BN (s0.pids 0 * BM - SP) qT numC kT sc (c + 1) r)) from by ring]
    rw [show oPartial BN (s0.pids 0 * BM - SP) qT numC kT vT sc c (r, d, PUnit.unit)
            / lPartial BN (s0.pids 0 * BM - SP) qT numC kT sc c r
          * lPartial BN (s0.pids 0 * BM - SP) qT numC kT sc c r
        = oPartial BN (s0.pids 0 * BM - SP) qT numC kT vT sc c (r, d, PUnit.unit) from hcancel]
    rw [add_div]
    congr 1
    · ring
    · rw [Finset.mul_sum, Finset.sum_div]
      refine Finset.sum_congr rfl (fun j _ => ?_)
      ring
  -- assemble
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [htailpids, hsW, BlockState.setReg_pids, hprepids, hse, BlockState.setReg_pids, hpids]
  · rw [hic, show c * BN + BN = (c + 1) * BN from by ring, Nat.mul_mod_left]
  · rw [hic, show c * BN + BN = (c + 1) * BN from by ring]
    have h1 : (c + 1) * BN ≤ numC * BN := Nat.mul_le_mul_right _ hclt
    rw [Nat.mul_comm numC BN] at h1
    exact h1
  · rw [hmF, hc1, hmcurrTile]
  · rw [hlF, hc1]; refine congrArg some ?_; ext r; rw [hlcurrCell r.1]; rfl
  · rw [haccF, hc1]; refine congrArg some ?_; ext idx; rw [haccCell idx]; rfl
  · rw [hqF, hsW, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hpreq, hse,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
    exact hq
  · rw [hoffmF, hsW, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hpreoffm, hse,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
    exact hoffm
  · rw [hoffnF, hsW, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hpreoffn, hse,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
    exact hoffn
  · rw [hbiF, hsW, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hprebi, hse,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
    exact hbi
  · rw [hohF, hsW, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hpreoh, hse,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
    exact hoh
  · rw [hkpF]
  · rw [hvpF]
  · exact htailundef
  · rw [htailmem, hsW]
    funext region offset
    rw [BlockState.setReg_mem]
    rw [show sPre.mem region offset = se.mem region offset from
      congrFun (congrFun hpremem region) offset]
    rw [hse, BlockState.setReg_mem]
    exact congrFun (congrFun hmem region) offset

/-! ## Causal exec + headline -/

set_option maxHeartbeats 8000000 in
set_option maxRecDepth 8000 in
open VeriTile.Examples.FA1MathCausal in
/-- **Causal full execution.** Preamble walk → causal `block_n_end` →
`forRangeDyn_inv` over `alcInv` → postLoop store walk. Every active `Out`
lane holds the fully streamed causal `oPartial numC / lPartial numC`. -/
theorem alc_exec (Q K V Out : RegionName) (s : BlockState) (sc : ℝ)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      N_HEAD H N_CTX SP BM BN BD numC : Nat)
    (hBM : 0 < BM) (hBN : 0 < BN)
    (hspanEq : (s.pids 0 + 1) * BN + SP = BN * numC)
    (hspanle : BN * numC ≤ N_CTX)
    (hsp : SP ≤ s.pids 0 * BM)
    (houtinj : Function.Injective (fun idx : TileIndex [BM, BD] =>
        (s.pids 0 * BM + idx.1.val) * som + idx.2.1.val * son))
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ∃ sF, exec (attention_llama_fwd_causal_surface Q K V Out sc
        sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
        N_HEAD H N_CTX SP BM BN BD) s = some sF
      ∧ ∀ idx : TileIndex [BM, BD], alActive s H BM idx →
          sF.readMem Out (alOutOffset s N_HEAD soz soh som son BM idx)
            = oPartial BN (s.pids 0 * BM - SP)
                (alQTileG s Q N_HEAD sqz sqh sqm sqk H BM BD) numC
                (alKTileG s K N_HEAD skz skh skn skk (BN * numC) BD)
                (alVTileG s V N_HEAD svz svh svk svn (BN * numC) BD) sc numC idx
              / lPartial BN (s.pids 0 * BM - SP)
                (alQTileG s Q N_HEAD sqz sqh sqm sqk H BM BD) numC
                (alKTileG s K N_HEAD skz skh skn skk (BN * numC) BD) sc numC idx.1 := by
  have hbody : exec (attention_llama_fwd_causal_surface Q K V Out sc
        sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
        N_HEAD H N_CTX SP BM BN BD) s
      = stepStmts (alPreLoopG Q K V sqz sqh sqm sqk skz skh skn skk svz svh svk svn
            N_HEAD H N_CTX BM BN BD
          ++ (alBnEndCausalStmt SP BN
              :: Stmt.forRangeDyn "start_n" (Op.constNat 0) (Op.ref .nat [] "block_n_end")
                   (Op.constNat BN) (alLoopBodyCausalG sc skn svk N_CTX SP BM BN BD)
              :: alPostLoopG Out soz soh som son H BM BD)) s := by
    show stepStmts (attention_llama_fwd_causal_surface Q K V Out sc
        sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
        N_HEAD H N_CTX SP BM BN BD).toAlgKernel.body s = _
    rw [alc_body_split]
  rw [hbody]
  -- preamble
  obtain ⟨sp, hpre, hsppids, hspmem, hspundef, hsm, hbi, hoh, hoffm, hoffn, hmp, hlp, hacc,
      hqreg, hkp, hvp, _hbne⟩ :=
    alPreLoop_evalG s Q K V sqz sqh sqm sqk skz skh skn skk svz svh svk svn
      N_HEAD H N_CTX BM BN BD hundef
  rw [stepStmts.append_some hpre]
  -- the causal block_n_end assignment
  rw [show alBnEndCausalStmt SP BN = Stmt.assign .nat [] "block_n_end"
      (Op.add .nat Broadcast.nil
        (Op.mul .nat Broadcast.nil
          (Op.add .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat 1)) (Op.constNat BN))
        (Op.constNat SP)) from rfl]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .nat Broadcast.nil
        (Op.mul .nat Broadcast.nil
          (Op.add .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat 1)) (Op.constNat BN))
        (Op.constNat SP)) sp = some (Tile.scalar (BN * numC)) from by
      rw [evalOp_add, evalOp_mul, evalOp_add]
      simp only [evalOp_ref, hsm, evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_
      ext u
      show (s.pids 0 + 1) * BN + SP = BN * numC
      exact hspanEq))]
  set spc := sp.setReg "block_n_end" .nat [] (Tile.scalar (BN * numC)) with hspc
  have hspcpids : spc.pids = s.pids := by rw [hspc, BlockState.setReg_pids, hsppids]
  have hspcmem : spc.mem = s.mem := by
    rw [hspc]; funext rg o; rw [BlockState.setReg_mem]
    exact congrFun (congrFun hspmem rg) o
  have hspcundef : ∀ rg o, spc.undef rg o = 0 := by
    intro rg o; rw [hspc, BlockState.setReg_undef]; exact hspundef rg o
  -- tile transport `s → spc`
  have htileQ : alQTileG spc Q N_HEAD sqz sqh sqm sqk H BM BD
      = alQTileG s Q N_HEAD sqz sqh sqm sqk H BM BD := by
    funext idx
    simp only [alQTileG, alRow, alBase, hspcpids]
    unfold BlockState.readMem
    rw [hspcmem]
  have htileK : alKTileG spc K N_HEAD skz skh skn skk (BN * numC) BD
      = alKTileG s K N_HEAD skz skh skn skk (BN * numC) BD := by
    funext idx
    simp only [alKTileG, alBase, hspcpids]
    unfold BlockState.readMem
    rw [hspcmem]
  have htileV : alVTileG spc V N_HEAD svz svh svk svn (BN * numC) BD
      = alVTileG s V N_HEAD svz svh svk svn (BN * numC) BD := by
    funext idx
    simp only [alVTileG, alBase, hspcpids]
    unfold BlockState.readMem
    rw [hspcmem]
  have hbaseK : alBase spc N_HEAD skz skh = alBase s N_HEAD skz skh := by
    simp only [alBase, hspcpids]
  have hbaseV : alBase spc N_HEAD svz svh = alBase s N_HEAD svz svh := by
    simp only [alBase, hspcpids]
  -- invariant base case at counter 0 (over spc)
  have hinv0 : alcInv Q K V spc sc sqz sqh sqm sqk skz skh skn skk svz svh svk svn
      N_HEAD H SP BM BN BD numC 0 spc := by
    refine alcInv_zero Q K V spc sc sqz sqh sqm sqk skz skh skn skk svz svh svk svn
      N_HEAD H SP BM BN BD numC ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ hspcundef
    · rw [hspc, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hmp
    · rw [hspc, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hlp
    · rw [hspc, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hacc
    · rw [hspc, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
      rw [show alQTileG (sp.setReg "block_n_end" .nat [] (Tile.scalar (BN * numC))) Q
            N_HEAD sqz sqh sqm sqk H BM BD
          = alQTileG s Q N_HEAD sqz sqh sqm sqk H BM BD from by rw [← hspc]; exact htileQ]
      exact hqreg
    · rw [hspc, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
      rw [show (sp.setReg "block_n_end" .nat [] (Tile.scalar (BN * numC))).pids = s.pids from by
        rw [← hspc]; exact hspcpids]
      exact hoffm
    · rw [hspc, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hoffn
    · rw [hspc, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
      rw [show (sp.setReg "block_n_end" .nat [] (Tile.scalar (BN * numC))).pids = s.pids from by
        rw [← hspc]; exact hspcpids]
      exact hbi
    · rw [hspc, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
      rw [show (sp.setReg "block_n_end" .nat [] (Tile.scalar (BN * numC))).pids = s.pids from by
        rw [← hspc]; exact hspcpids]
      exact hoh
    · rw [hspc, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
      rw [show alBase (sp.setReg "block_n_end" .nat [] (Tile.scalar (BN * numC))) N_HEAD skz skh
          = alBase s N_HEAD skz skh from by rw [← hspc]; exact hbaseK]
      exact hkp
    · rw [hspc, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
      rw [show alBase (sp.setReg "block_n_end" .nat [] (Tile.scalar (BN * numC))) N_HEAD svz svh
          = alBase s N_HEAD svz svh from by rw [← hspc]; exact hbaseV]
      exact hvp
  -- the streaming loop
  obtain ⟨final, sL, hloop, hfin, hinvL⟩ :=
    forRangeDyn_inv (idx := "start_n") (startOp := Op.constNat 0)
      (stopOp := Op.ref .nat [] "block_n_end")
      (stepOp := Op.constNat BN)
      (P := fun i st => alcInv Q K V spc sc sqz sqh sqm sqk skz skh skn skk svz svh svk svn
        N_HEAD H SP BM BN BD numC i st)
      (s_init := spc)
      (by rw [evalOp_constNat])
      (by rw [evalOp_ref, hspc, BlockState.setReg_same])
      (by rw [evalOp_constNat])
      (by omega)
      hinv0
      (fun i st hi hP => alc_attn_step Q K V spc sc sqz sqh sqm sqk skz skh skn skk svz svh svk svn
        N_HEAD H N_CTX SP BM BN BD numC hBM hBN hspanle
        (by rw [hspcpids]; exact hsp) i st hi hP)
  rw [stepStmts.cons_some hloop]
  -- final counter = BN * numC
  simp only [alcInv] at hinvL
  obtain ⟨hpidsL, _hmodL, hleL, _hmpL, _hlpL, haccL, _hqL, _hoffmL, _hoffnL, hbiL, hohL,
    _hkpL, _hvpL, _hundefL, _hmemL⟩ := hinvL
  have hfinal : final = BN * numC := by omega
  subst hfinal
  rw [Nat.mul_div_cancel_left _ hBN] at haccL
  -- postLoop
  obtain ⟨sP, hpost, hread⟩ :=
    alPostLoop_run Out spc soz soh som son N_HEAD H BM BD sL
      (fun idx : TileIndex [BM, BD] =>
        oPartial BN (spc.pids 0 * BM - SP)
            (alQTileG spc Q N_HEAD sqz sqh sqm sqk H BM BD) numC
            (alKTileG spc K N_HEAD skz skh skn skk (BN * numC) BD)
            (alVTileG spc V N_HEAD svz svh svk svn (BN * numC) BD) sc numC idx
          / lPartial BN (spc.pids 0 * BM - SP)
            (alQTileG spc Q N_HEAD sqz sqh sqm sqk H BM BD) numC
            (alKTileG spc K N_HEAD skz skh skn skk (BN * numC) BD) sc numC idx.1)
      (by simpa only [hspcpids] using houtinj) hpidsL haccL hbiL hohL
  refine ⟨sP, hpost, ?_⟩
  intro idx hActive
  have hActive' : alActive spc H BM idx := by
    simpa only [alActive, alRow, hspcpids] using hActive
  have hoff : alOutOffset spc N_HEAD soz soh som son BM idx
      = alOutOffset s N_HEAD soz soh som son BM idx := by
    simp only [alOutOffset, alRow, alBase, hspcpids]
  have h := hread idx hActive'
  rw [hoff, htileQ, htileK, htileV, hspcpids] at h
  exact h

/-- The `VeriTile.Examples` and `VeriTile.Triton` spellings of the causal
block attention closed form coincide (definitional unfold). -/
theorem alc_attentionRealCausalBlock_eq {M S D : Nat} (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ) (scale : ℝ)
    (idx : TileIndex [M, D]) :
    VeriTile.Examples.attentionRealCausalBlock qStart Q K V scale idx
      = attentionRealCausalBlock qStart Q K V scale idx := by
  obtain ⟨i, d, u⟩ := idx; cases u
  unfold VeriTile.Examples.attentionRealCausalBlock VeriTile.Triton.attentionRealCausalBlock
    VeriTile.Triton.scaledScore
  rfl

set_option maxHeartbeats 1600000 in
open VeriTile.Examples.FA1MathCausal in
/-- **★ MAIN (causal).** Every ACTIVE `Out` cell (query row `< H`) holds
the natural-exp CAUSAL softmax attention over the streamed key span
`span = (pid₀+1)·BLOCK_N + start_position = BLOCK_N·numCausalBlocks`
(`hspanEq` — equivalent to `BLOCK_N ∣ start_position` at this pid): the
visible keys of query row `row = pid₀·BLOCK_M + i` are exactly
`{j < span : j ≤ (pid₀·BLOCK_M − start_position) + i}`, i.e.
`j + start_position ≤ row` under `hsp : start_position ≤ pid₀·BLOCK_M` —
the kernel's own `offs_m ≥ block_n_offs + start_position` predicate
(`start_position` SHRINKS visibility; see the header quirk note). Side
conditions: `hspanEq`, `span ≤ N_CTX` (`hspanle`, which makes every
loaded key lane in-range — the causal arm tolerates arbitrary
`N_CTX ≥ span`), `hsp`, output-offset injectivity, and the clean-undef
carrier. -/
specification attention_llama_fwd_causal_closed_form_correct
    (Q K V Out : RegionName) (s : BlockState) (sm_scale : ℝ)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      N_HEAD H N_CTX start_position BLOCK_M BLOCK_N BLOCK_DMODEL numCausalBlocks : Nat)
    (hBM : 0 < BLOCK_M) (hBN : 0 < BLOCK_N)
    (hspanEq : (s.pids 0 + 1) * BLOCK_N + start_position = BLOCK_N * numCausalBlocks)
    (hspanle : BLOCK_N * numCausalBlocks ≤ N_CTX)
    (hsp : start_position ≤ s.pids 0 * BLOCK_M)
    (houtinj : Function.Injective (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        (s.pids 0 * BLOCK_M + idx.1.val) * stride_om + idx.2.1.val * stride_on))
    (hundef : ∀ rg o, s.undef rg o = 0) :
    (∃ alg, (attention_llama_fwd_causal_surface Q K V Out sm_scale
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      N_HEAD H N_CTX start_position BLOCK_M BLOCK_N BLOCK_DMODEL).toAlgorithm?
        = Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := attention_llama_fwd_causal_surface Q K V Out sm_scale
        stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
        stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
        N_HEAD H N_CTX start_position BLOCK_M BLOCK_N BLOCK_DMODEL)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => alActive s H BLOCK_M idx)
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          (Out, alOutOffset s N_HEAD stride_oz stride_oh stride_om stride_on BLOCK_M idx)))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        attentionRealCausalBlock (s.pids 0 * BLOCK_M - start_position)
          (alQTileG s Q N_HEAD stride_qz stride_qh stride_qm stride_qk H BLOCK_M BLOCK_DMODEL)
          (alKTileG s K N_HEAD stride_kz stride_kh stride_kn stride_kk
            (BLOCK_N * numCausalBlocks) BLOCK_DMODEL)
          (alVTileG s V N_HEAD stride_vz stride_vh stride_vk stride_vn
            (BLOCK_N * numCausalBlocks) BLOCK_DMODEL)
          sm_scale idx) := by
  have hnumC : 0 < numCausalBlocks := by
    rcases Nat.eq_zero_or_pos numCausalBlocks with h0 | h
    · exfalso
      rw [h0, Nat.mul_zero] at hspanEq
      have h1 : (s.pids 0 + 1) * BLOCK_N = 0 := by omega
      rcases Nat.mul_eq_zero.mp h1 with h2 | h2 <;> omega
    · exact h
  obtain ⟨sF, hstep, hO⟩ :=
    alc_exec Q K V Out s sm_scale stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on N_HEAD H N_CTX start_position
      BLOCK_M BLOCK_N BLOCK_DMODEL numCausalBlocks hBM hBN hspanEq hspanle hsp houtinj hundef
  refine ⟨?_, ?_⟩
  · exact attention_llama_fwd_causal_surface_toAlgorithm_supported Q K V Out sm_scale
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      N_HEAD H N_CTX start_position BLOCK_M BLOCK_N BLOCK_DMODEL
  · rw [ComputeCorrect.realizes_writeIf_iff]
    apply ComputeKernel.computeCorrect_of_toAlgKernel
    · simp [attention_llama_fwd_causal_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
    intro s0 s' hExec hs0
    subst s0
    intro idx hActive
    rw [hstep] at hExec
    obtain rfl : sF = s' := Option.some.inj hExec
    simp only [ComputeCorrect.OutputReadable.read_real]
    rw [hO idx hActive]
    rw [streaming_eq_attentionRealCausalBlock hBN (s.pids 0 * BLOCK_M - start_position)
      (alQTileG s Q N_HEAD stride_qz stride_qh stride_qm stride_qk H BLOCK_M BLOCK_DMODEL)
      numCausalBlocks hnumC
      (alKTileG s K N_HEAD stride_kz stride_kh stride_kn stride_kk
        (BLOCK_N * numCausalBlocks) BLOCK_DMODEL)
      (alVTileG s V N_HEAD stride_vz stride_vh stride_vk stride_vn
        (BLOCK_N * numCausalBlocks) BLOCK_DMODEL)
      sm_scale idx]
    exact alc_attentionRealCausalBlock_eq (s.pids 0 * BLOCK_M - start_position)
      (alQTileG s Q N_HEAD stride_qz stride_qh stride_qm stride_qk H BLOCK_M BLOCK_DMODEL)
      (alKTileG s K N_HEAD stride_kz stride_kh stride_kn stride_kk
        (BLOCK_N * numCausalBlocks) BLOCK_DMODEL)
      (alVTileG s V N_HEAD stride_vz stride_vh stride_vk stride_vn
        (BLOCK_N * numCausalBlocks) BLOCK_DMODEL)
      sm_scale idx

end Correct_without_Rounding

end VeriTile.Bench.TritonBenchG.AttentionLlama
