import VeriTile.Triton

/-!
# `chunk_linear_attn` — strict per-kernel correctness

The upstream file holds **four** `@triton.jit` kernels: the state-recurrence
forward `chunk_linear_attn_fwd_kernel_h`, the output forward
`chunk_linear_attn_fwd_kernel_o`, the state-recurrence backward
`chunk_linear_attn_bwd_kernel_dh`, and the fused gradient backward
`chunk_linear_attn_bwd_kernel_dqkv`. This file covers the two **state-recurrence**
kernels — the file's first kernel, `chunk_linear_attn.py`'s
`chunk_linear_attn_fwd_kernel_h`, and its descending mirror `bwd_kernel_dh` —
which share one proof skeleton: a `[BK, BV]` running state is **stored to memory
at every chunk** and then advanced by one `tl.dot`.

```
fwd_h:  h[·,·,t] = H_t,   H_0 = (h0 or 0),  H_{t+1} = H_t + k_tᵀ · v_t
        (and ht = H_NT when STORE_FINAL_STATE)
bwd_dh: dh[·,·,t] = D_t,  D_NT = 0,         D_{t-1} = D_t + (scale·q_t) · do_t
```

The store-then-accumulate order matters and is carried by the spec: chunk `t`
receives the state *before* chunk `t`'s own contribution.

## The descending loop, spelled ascending

`bwd_kernel_dh` iterates `for i_t in range(NT - 1, -1, -1)`. The DSL's `forRange`
counts up (`stepForRangeAux` advances while `cur < stop`), so the faithful surface
spells the change of variable explicitly:

    for j in range(0, NT, 1) { i_t = NT - $(1) - j; … }

— the identical iteration sequence, with the descending index computed from the
ascending counter as the body's first statement. This is the established
respelling (`triton_linear_activation`'s "descending K-loop as antiquoted
ascending trip count", `completion_audit.md`), and it is observable in the
surface rather than hidden in a spec.

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); the host launch (the 3-D
grid `(NK, NV, B*H)` and the host-computed `NT`) is the *trusted boundary*. Every
dimension and stride stays a symbolic parameter, as do the two `constexpr` gates
`USE_INITIAL_STATE` / `STORE_FINAL_STATE` — one theorem covers all four
configurations. The `.to(...)` dtype round-trips erase to the identity at the
algorithm layer.

Spelling notes, per `bench/MAIN_THEOREM_CONVENTIONS.md`, all surface syntax
rather than semantics: integer literals inside index arithmetic are written
`$(n)`; `boundary_check=(0, 1)` is written `boundary_check=([0, 1] : List Nat)`.
-/

namespace VeriTile.Bench.TritonBenchG.ChunkLinearAttn

open VeriTile.Triton

section Correct_without_Rounding

/-- Faithful transcription of `chunk_linear_attn_fwd_kernel_h`. -/
def cla_fwd_h_surface
    (k v h h0 ht : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat)
    (T K V BT BK BV NT : Nat)
    (USE_INITIAL_STATE STORE_FINAL_STATE : Bool) :
    ComputeKernel := triton {
  i_k = tl.program_id(0)
  i_v = tl.program_id(1)
  i_bh = tl.program_id(2)
  b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
  if USE_INITIAL_STATE {
    p_h0 = tl.make_block_ptr(base=h0 + i_bh * $(K) * $(V),
      shape=($(K), $(V)), strides=($(V), $(1)),
      offsets=(i_k * $(BK), i_v * $(BV)), block_shape=($(BK), $(BV)), order=(1, 0))
    b_h = (tl.load(p_h0, boundary_check=([0, 1] : List Nat))).to(tl.float32)
  }
  for i_t in range($(0), $(NT), $(1)) {
    p_k = tl.make_block_ptr(base=k + i_bh * $(s_qk_h),
      shape=($(K), $(T)), strides=($(s_qk_d), $(s_qk_t)),
      offsets=(i_k * $(BK), i_t * $(BT)), block_shape=($(BK), $(BT)), order=(0, 1))
    p_v = tl.make_block_ptr(base=v + i_bh * $(s_vo_h),
      shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)),
      offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
    p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V),
      shape=($(K), $(V)), strides=($(s_h_t), $(1)),
      offsets=(i_k * $(BK), i_v * $(BV)), block_shape=($(BK), $(BV)), order=(1, 0))
    tl.store(p_h, (b_h).to(p_h.dtype.element_ty), boundary_check=([0, 1] : List Nat))
    b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
    b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
    b_h += tl.dot(b_k, b_v, allow_tf32=false)
  }
  if STORE_FINAL_STATE {
    p_ht = tl.make_block_ptr(base=ht + i_bh * $(K) * $(V),
      shape=($(K), $(V)), strides=($(V), $(1)),
      offsets=(i_k * $(BK), i_v * $(BV)), block_shape=($(BK), $(BV)), order=(1, 0))
    tl.store(p_ht, (b_h).to(p_ht.dtype.element_ty), boundary_check=([0, 1] : List Nat))
  }
}

/-- The forward state surface lowers to the algorithm layer. -/
theorem cla_fwd_h_surface_toAlgorithm_supported
    (k v h h0 ht : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat)
    (T K V BT BK BV NT : Nat) (USE_INITIAL_STATE STORE_FINAL_STATE : Bool) :
    ∃ alg, (cla_fwd_h_surface k v h h0 ht s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
      s_h_h s_h_t T K V BT BK BV NT
      USE_INITIAL_STATE STORE_FINAL_STATE).toAlgorithm? = Except.ok alg := by
  simp [cla_fwd_h_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- Faithful transcription of `chunk_linear_attn_bwd_kernel_dh`, with the
descending loop spelled as its ascending change of variable (see the preamble). -/
def cla_bwd_dh_surface
    (q do_ dh : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat)
    (scale : ℝ) (T K V BT BK BV NT : Nat) :
    ComputeKernel := triton {
  i_k = tl.program_id(0)
  i_v = tl.program_id(1)
  i_bh = tl.program_id(2)
  b_dh = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
  for j in range($(0), $(NT), $(1)) {
    i_t = $(NT) - $(1) - j
    p_q = tl.make_block_ptr(base=q + i_bh * $(s_qk_h),
      shape=($(K), $(T)), strides=($(s_qk_d), $(s_qk_t)),
      offsets=(i_k * $(BK), i_t * $(BT)), block_shape=($(BK), $(BT)), order=(0, 1))
    p_do = tl.make_block_ptr(base=do_ + i_bh * $(s_vo_h),
      shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)),
      offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
    p_dh = tl.make_block_ptr(base=dh + i_bh * $(s_h_h) + i_t * $(K) * $(V),
      shape=($(K), $(V)), strides=($(s_h_t), $(1)),
      offsets=(i_k * $(BK), i_v * $(BV)), block_shape=($(BK), $(BV)), order=(1, 0))
    tl.store(p_dh, (b_dh).to(p_dh.dtype.element_ty), boundary_check=([0, 1] : List Nat))
    b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat))
    b_q = (b_q * $(scale)).to(b_q.dtype)
    b_do = tl.load(p_do, boundary_check=([0, 1] : List Nat))
    b_dh += tl.dot(b_q, (b_do).to(b_q.dtype), allow_tf32=false)
  }
}

/-- The backward state surface lowers to the algorithm layer. -/
theorem cla_bwd_dh_surface_toAlgorithm_supported
    (q do_ dh : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat)
    (scale : ℝ) (T K V BT BK BV NT : Nat) :
    ∃ alg, (cla_bwd_dh_surface q do_ dh s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
      s_h_h s_h_t scale T K V BT BK BV NT).toAlgorithm? = Except.ok alg := by
  simp [cla_bwd_dh_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-! ## Compiled body decomposition

The algorithm-lowered statement lists, checked against the macro output by `rfl`.
One lowering worth naming: an assignment-position `.to(tl.float32)` on a load
lowers to a real `Op.castFloat FloatDType.real FloatDType.real` (not to erasure —
only *expression-interior* casts erase), whose `.real → .real` semantics is the
identity on lanes. -/

/-- The `USE_INITIAL_STATE` branch: the `h0` block pointer and the seeded load. -/
def claInitBranch (h0 : RegionName) (K V BK BV : Nat) : List Stmt :=
  [ Stmt.assign .blockPtr [BK, BV] "p_h0"
      (Op.makeBlockPtrDynOffsets h0
        (Op.mul .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat K))
          (Op.constNat V))
        [K, V] [BK, BV] [V, 1]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat BK),
          Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_v") (Op.constNat BV)]),
    Stmt.assign .real [BK, BV] "b_h"
      (Op.castFloat FloatDType.real FloatDType.real
        (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [BK, BV] "p_h0") [0, 1])
          MaskOpt.none)) ]

/-- The forward chunk body: three block pointers, the state **store**, the `k` and
`v` loads, and the recurrence `b_h += tl.dot(b_k, b_v)`. The store-before-load
order is the whole content of the spec's "state *before* chunk `t`". -/
def claFwdBody (k v h : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat)
    (T K V BT BK BV : Nat) : List Stmt :=
  [ Stmt.assign .blockPtr [BK, BT] "p_k"
      (Op.makeBlockPtrDynOffsets k
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_qk_h))
        [K, T] [BK, BT] [s_qk_d, s_qk_t]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat BK),
          Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat BT)]),
    Stmt.assign .blockPtr [BT, BV] "p_v"
      (Op.makeBlockPtrDynOffsets v
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_vo_h))
        [T, V] [BT, BV] [s_vo_t, s_vo_d]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat BT),
          Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_v") (Op.constNat BV)]),
    Stmt.assign .blockPtr [BK, BV] "p_h"
      (Op.makeBlockPtrDynOffsets h
        (Op.add .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_h_h))
          (Op.mul .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat K))
            (Op.constNat V)))
        [K, V] [BK, BV] [s_h_t, 1]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat BK),
          Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_v") (Op.constNat BV)]),
    Stmt.store .real [BK, BV]
      (MemAccess.blockPtr (Op.ref .blockPtr [BK, BV] "p_h") [0, 1])
      (Op.ref .real [BK, BV] "b_h") MaskOpt.none,
    Stmt.assign .real [BK, BT] "b_k"
      (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [BK, BT] "p_k") [0, 1])
        MaskOpt.none),
    Stmt.assign .real [BT, BV] "b_v"
      (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [BT, BV] "p_v") [0, 1])
        MaskOpt.none),
    Stmt.assign .real [BK, BV] "b_h"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BK, BV] "b_h")
        (Op.dot (batch := []) (Op.ref .real [BK, BT] "b_k")
          (Op.ref .real [BT, BV] "b_v"))) ]

/-- The `STORE_FINAL_STATE` branch: the `ht` block pointer and the final store. -/
def claFinalBranch (ht : RegionName) (K V BK BV : Nat) : List Stmt :=
  [ Stmt.assign .blockPtr [BK, BV] "p_ht"
      (Op.makeBlockPtrDynOffsets ht
        (Op.mul .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat K))
          (Op.constNat V))
        [K, V] [BK, BV] [V, 1]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat BK),
          Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_v") (Op.constNat BV)]),
    Stmt.store .real [BK, BV]
      (MemAccess.blockPtr (Op.ref .blockPtr [BK, BV] "p_ht") [0, 1])
      (Op.ref .real [BK, BV] "b_h") MaskOpt.none ]

set_option maxRecDepth 8000 in
/-- **Forward body split (by `rfl`).** Seven top-level statements. -/
theorem cla_fwd_body_eq (k v h h0 ht : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat)
    (T K V BT BK BV NT : Nat) (UIS SFS : Bool) :
    (cla_fwd_h_surface k v h h0 ht s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
        s_h_h s_h_t T K V BT BK BV NT UIS SFS).toAlgKernel.body
      = [ Stmt.assign .nat [] "i_k" (Op.programId 0),
          Stmt.assign .nat [] "i_v" (Op.programId 1),
          Stmt.assign .nat [] "i_bh" (Op.programId 2),
          Stmt.assign .real [BK, BV] "b_h" (Op.full [BK, BV] (Op.const 0)),
          Stmt.ifThen (Op.constBool UIS) (claInitBranch h0 K V BK BV),
          Stmt.forRange "i_t" 0 NT 1
            (claFwdBody k v h s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t
              T K V BT BK BV),
          Stmt.ifThen (Op.constBool SFS) (claFinalBranch ht K V BK BV) ] := by
  rfl

/-- The backward chunk body: the descending index as its first statement, then the
same store-then-accumulate skeleton over `q` / `do`. -/
def claBwdBody (q do_ dh : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat) (scale : ℝ)
    (T K V BT BK BV NT : Nat) : List Stmt :=
  [ Stmt.assign .nat [] "i_t"
      (Op.sub .nat Broadcast.nil
        (Op.sub .nat Broadcast.nil (Op.constNat NT) (Op.constNat 1))
        (Op.ref .nat [] "j")),
    Stmt.assign .blockPtr [BK, BT] "p_q"
      (Op.makeBlockPtrDynOffsets q
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_qk_h))
        [K, T] [BK, BT] [s_qk_d, s_qk_t]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat BK),
          Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat BT)]),
    Stmt.assign .blockPtr [BT, BV] "p_do"
      (Op.makeBlockPtrDynOffsets do_
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_vo_h))
        [T, V] [BT, BV] [s_vo_t, s_vo_d]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat BT),
          Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_v") (Op.constNat BV)]),
    Stmt.assign .blockPtr [BK, BV] "p_dh"
      (Op.makeBlockPtrDynOffsets dh
        (Op.add .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_h_h))
          (Op.mul .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat K))
            (Op.constNat V)))
        [K, V] [BK, BV] [s_h_t, 1]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat BK),
          Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_v") (Op.constNat BV)]),
    Stmt.store .real [BK, BV]
      (MemAccess.blockPtr (Op.ref .blockPtr [BK, BV] "p_dh") [0, 1])
      (Op.ref .real [BK, BV] "b_dh") MaskOpt.none,
    Stmt.assign .real [BK, BT] "b_q"
      (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [BK, BT] "p_q") [0, 1])
        MaskOpt.none),
    Stmt.assign .real [BK, BT] "b_q"
      (Op.mul .real Broadcast.scalarR (Op.ref .real [BK, BT] "b_q") (Op.const scale)),
    Stmt.assign .real [BT, BV] "b_do"
      (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [BT, BV] "p_do") [0, 1])
        MaskOpt.none),
    Stmt.assign .real [BK, BV] "b_dh"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BK, BV] "b_dh")
        (Op.dot (batch := []) (Op.ref .real [BK, BT] "b_q")
          (Op.ref .real [BT, BV] "b_do"))) ]

set_option maxRecDepth 8000 in
/-- **Backward body split (by `rfl`).** Five top-level statements. -/
theorem cla_bwd_body_eq (q do_ dh : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat)
    (scale : ℝ) (T K V BT BK BV NT : Nat) :
    (cla_bwd_dh_surface q do_ dh s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
        s_h_h s_h_t scale T K V BT BK BV NT).toAlgKernel.body
      = [ Stmt.assign .nat [] "i_k" (Op.programId 0),
          Stmt.assign .nat [] "i_v" (Op.programId 1),
          Stmt.assign .nat [] "i_bh" (Op.programId 2),
          Stmt.assign .real [BK, BV] "b_dh" (Op.full [BK, BV] (Op.const 0)),
          Stmt.forRange "j" 0 NT 1
            (claBwdBody q do_ dh s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h
              s_h_t scale T K V BT BK BV NT) ] := by
  rfl

/-! ## Closed-form specification

All accessors read the **launch** state's memory at the addresses the block
pointers compute, with the boundary checks factored into guarded forms. The chunk
index is `t`, the K lane `e`, the T lane `c`, the V lane `p`. -/

/-- `k[i_k·BK + e, t·BT + c]` (parent `(K, T)`, strides `(s_qk_d, s_qk_t)`). -/
noncomputable def kElem (s : BlockState) (k : RegionName)
    (s_qk_h s_qk_t s_qk_d BT BK : Nat) (t c e : Nat) : ℝ :=
  s.readMem k (s.pids 2 * s_qk_h + (s.pids 0 * BK + e) * s_qk_d
    + (t * BT + c) * s_qk_t)

/-- `v[t·BT + c, i_v·BV + p]` (parent `(T, V)`, strides `(s_vo_t, s_vo_d)`). -/
noncomputable def vElem (s : BlockState) (v : RegionName)
    (s_vo_h s_vo_t s_vo_d BT BV : Nat) (t c p : Nat) : ℝ :=
  s.readMem v (s.pids 2 * s_vo_h + (t * BT + c) * s_vo_t
    + (s.pids 1 * BV + p) * s_vo_d)

/-- `h0[i_k·BK + e, i_v·BV + p]` (parent `(K, V)`, strides `(V, 1)`). -/
noncomputable def h0Elem (s : BlockState) (h0 : RegionName)
    (K V BK BV : Nat) (e p : Nat) : ℝ :=
  s.readMem h0 (s.pids 2 * K * V + (s.pids 0 * BK + e) * V
    + (s.pids 1 * BV + p) * 1)

/-- The guarded `k` lane, as `b_k` holds it. -/
noncomputable def kGuarded (s : BlockState) (k : RegionName)
    (s_qk_h s_qk_t s_qk_d : Nat) (T K BT BK : Nat) (t c e : Nat) : ℝ :=
  if s.pids 0 * BK + e < K ∧ t * BT + c < T then
    kElem s k s_qk_h s_qk_t s_qk_d BT BK t c e
  else 0

/-- The guarded `v` lane, as `b_v` holds it. -/
noncomputable def vGuarded (s : BlockState) (v : RegionName)
    (s_vo_h s_vo_t s_vo_d : Nat) (T V BT BV : Nat) (t c p : Nat) : ℝ :=
  if t * BT + c < T ∧ s.pids 1 * BV + p < V then
    vElem s v s_vo_h s_vo_t s_vo_d BT BV t c p
  else 0

/-- The guarded initial state, as the `USE_INITIAL_STATE` branch leaves `b_h`. -/
noncomputable def h0Guarded (s : BlockState) (h0 : RegionName)
    (K V BK BV : Nat) (e p : Nat) : ℝ :=
  if s.pids 0 * BK + e < K ∧ s.pids 1 * BV + p < V then
    h0Elem s h0 K V BK BV e p
  else 0

/-- One chunk's contribution to state lane `(e, p)`: `Σ_c k[e, c] · v[c, p]`. -/
noncomputable def claContrib (s : BlockState) (k v : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat)
    (T K V BT BK BV : Nat) (t e p : Nat) : ℝ :=
  ∑ c : Fin BT,
    kGuarded s k s_qk_h s_qk_t s_qk_d T K BT BK t c.val e
      * vGuarded s v s_vo_h s_vo_t s_vo_d T V BT BV t c.val p

/-- **The forward state at chunk `t`** — what the kernel stores into `h[·,·,t]`:
the (gated) initial state plus every chunk *before* `t`. -/
noncomputable def claHState (s : BlockState) (k v h0 : RegionName)
    (UIS : Bool) (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat)
    (T K V BT BK BV : Nat) (t e p : Nat) : ℝ :=
  (if UIS then h0Guarded s h0 K V BK BV e p else 0)
    + ∑ u : Fin t, claContrib s k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
        T K V BT BK BV u.val e p

/-- One chunk's contribution to the backward state lane `(e, p)`:
`Σ_c (scale · q)[e, c] · do[c, p]` — `q` shares `k`'s layout, `do` shares `v`'s. -/
noncomputable def claBContrib (s : BlockState) (q do_ : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (T K V BT BK BV : Nat) (t e p : Nat) : ℝ :=
  ∑ c : Fin BT,
    (kGuarded s q s_qk_h s_qk_t s_qk_d T K BT BK t c.val e * scale)
      * vGuarded s do_ s_vo_h s_vo_t s_vo_d T V BT BV t c.val p

/-- **The backward state at chunk `t`** — what the kernel stores into `dh[·,·,t]`:
every chunk *after* `t` (the loop descends, and each chunk is stored before it is
accumulated). -/
noncomputable def claDhState (s : BlockState) (q do_ : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (T K V BT BK BV NT : Nat) (t e p : Nat) : ℝ :=
  ∑ u : Fin NT, if t < u.val then
    claBContrib s q do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
      T K V BT BK BV u.val e p
  else 0

/-- The `h` / `dh` chunk-`t` store address for lane `(e, p)`. -/
def claHOffset (s : BlockState) (s_h_h s_h_t K V BK BV : Nat) (t : Nat)
    (idx : TileIndex [BK, BV]) : Nat :=
  s.pids 2 * s_h_h + t * K * V + (s.pids 0 * BK + idx.1.val) * s_h_t
    + (s.pids 1 * BV + idx.2.1.val) * 1

/-- The `ht` store address for lane `(e, p)`. -/
def claHtOffset (s : BlockState) (K V BK BV : Nat)
    (idx : TileIndex [BK, BV]) : Nat :=
  s.pids 2 * K * V + (s.pids 0 * BK + idx.1.val) * V
    + (s.pids 1 * BV + idx.2.1.val) * 1

/-- A state lane is *active* when it maps inside the `K × V` window. -/
def claActive (s : BlockState) (K V BK BV : Nat)
    (idx : TileIndex [BK, BV]) : Prop :=
  s.pids 0 * BK + idx.1.val < K ∧ s.pids 1 * BV + idx.2.1.val < V

/-! ## Eval recipes

Local copies (bench files never import each other), plus the two lowerings this
pair adds: the `.real → .real` `castFloat` identity and the descending-index
arithmetic `NT - 1 - j`. -/

private theorem cla_makeBlockPtr_2d_eval (rg : RegionName) (s : BlockState)
    (baseOp : Op .nat []) (rowOffOp colOffOp : Op .nat [])
    (parentShape blockShape strides : List Nat)
    (base rowOff colOff : Nat)
    (hbase : evalOp baseOp s = some (Tile.scalar base))
    (hrow : evalOp rowOffOp s = some (Tile.scalar rowOff))
    (hcol : evalOp colOffOp s = some (Tile.scalar colOff)) :
    evalOp (Op.makeBlockPtrDynOffsets rg baseOp parentShape blockShape strides
        [rowOffOp, colOffOp]) s
      = some (⟨fun _ => BlockPtr.mk rg base parentShape blockShape strides
          [rowOff, colOff]⟩ : Tile .blockPtr blockShape) := by
  simp only [evalOp, hbase, hrow, hcol, List.mapM, List.mapM.loop, bind, Option.bind,
    Tile.scalar, List.reverse_cons, List.reverse_nil, List.nil_append,
    List.cons_append]

/-- No-mask 2D block-pointer load through a bound register: lane `(i,j)` reads the
genuine memory cell when in-bounds, else `0`. -/
private theorem cla_load_bp_2d (rg : RegionName) (s : BlockState) (name : RegName)
    (base rows cols BR BS strideT strideS rowOff colOff : Nat)
    (hreg : s.regs .blockPtr [BR, BS] name = some
      ⟨fun _ => BlockPtr.mk rg base [rows, cols] [BR, BS] [strideT, strideS]
        [rowOff, colOff]⟩) :
    evalOp (Op.load .real
      (MemAccess.blockPtr (Op.ref .blockPtr [BR, BS] name) [0, 1]) MaskOpt.none) s
    = some ⟨fun idx : TileIndex [BR, BS] =>
        if (rowOff + idx.1.val < rows ∧ colOff + idx.2.1.val < cols) then
          some (s.readMem rg (base + (rowOff + idx.1.val) * strideT
            + (colOff + idx.2.1.val) * strideS))
        else some 0⟩ := by
  simp only [evalOp, evalOp_ref, hreg, bind, Option.bind]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, j, rest⟩ := idx
  simp only [TileShape.indexToList, BlockPtr.address_2d_offsets,
    BlockPtr.inBounds_2d_offsets, BlockState.readMemValue_real]
  by_cases h : rowOff + i.val < rows ∧ colOff + j.val < cols
  · simp only [h, and_self, decide_true, if_true]
  · simp only [h, decide_false, if_false, BlockState.defaultCarrier]
    rfl

/-- Scalar `name * c`. -/
private theorem cla_mulConst_eval (s : BlockState) (name : RegName) (val c : Nat)
    (hr : s.regs .nat [] name = some (Tile.scalar val)) :
    evalOp (Op.mul .nat Broadcast.nil (Op.ref .nat [] name) (Op.constNat c)) s
      = some (Tile.scalar (val * c)) := by
  rw [evalOp_mul]
  simp only [evalOp_ref, evalOp_constNat, hr, Option.bind_eq_bind, Option.bind_some]
  rfl

/-- Scalar `(name * cB) * cC` — the `h0` / `ht` base `i_bh * K * V`. -/
private theorem cla_mulMulConst_eval (s : BlockState) (name : RegName)
    (val cB cC : Nat) (hr : s.regs .nat [] name = some (Tile.scalar val)) :
    evalOp (Op.mul .nat Broadcast.nil
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] name) (Op.constNat cB))
        (Op.constNat cC)) s
      = some (Tile.scalar (val * cB * cC)) := by
  rw [evalOp_mul, cla_mulConst_eval s name val cB hr]
  simp only [evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
  rfl

/-- The `p_h` / `p_dh` base: `i_bh * s_h_h + i_t * K * V`. -/
private theorem cla_hBase_eval (s : BlockState) (s_h_h K V : Nat) (ibh it : Nat)
    (hibh : s.regs .nat [] "i_bh" = some (Tile.scalar ibh))
    (hit : s.regs .nat [] "i_t" = some (Tile.scalar it)) :
    evalOp (Op.add .nat Broadcast.nil
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_h_h))
        (Op.mul .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat K))
          (Op.constNat V))) s
      = some (Tile.scalar (ibh * s_h_h + it * K * V)) := by
  rw [evalOp_add, cla_mulConst_eval s "i_bh" ibh s_h_h hibh, evalOp_mul,
    cla_mulConst_eval s "i_t" it K hit]
  simp only [evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
  rfl

/-- The descending change of variable: `i_t = NT - 1 - j` on `nat` scalars. -/
private theorem cla_itIdx_eval (t : BlockState) (NT c : Nat)
    (hj : t.regs .nat [] "j" = some (Tile.scalar c)) :
    evalOp (Op.sub .nat Broadcast.nil
        (Op.sub .nat Broadcast.nil (Op.constNat NT) (Op.constNat 1))
        (Op.ref .nat [] "j")) t
      = some (Tile.scalar (NT - 1 - c)) := by
  have hx : evalOp (Op.sub .nat Broadcast.nil (Op.constNat NT) (Op.constNat 1)) t
      = some (Tile.scalar (NT - 1)) := by
    rw [evalOp_sub]
    simp only [evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
    rfl
  rw [evalOp_sub, hx]
  simp only [evalOp_ref, hj, Option.bind_eq_bind, Option.bind_some]
  rfl

/-- The assignment-position `.to(tl.float32)` on an already-`.real` operand: a
genuine `Op.castFloat .real .real` node whose semantics is the identity. -/
private theorem cla_castFloat_eval {sh : TileShape} (x : Op .real sh)
    (t : BlockState) (vx : Tile .real sh) (hx : evalOp x t = some vx) :
    @evalOp TileDType.real sh (Op.castFloat FloatDType.real FloatDType.real x) t
      = some vx := by
  show @evalOp FloatDType.real.toTileDType sh
    (Op.castFloat FloatDType.real FloatDType.real x) t = some vx
  rw [evalOp_castFloat,
    show @evalOp FloatDType.real.toTileDType sh x t = some vx from hx]
  rfl

/-- `tl.zeros([BK, BV], dtype=tl.float32)`. -/
private theorem cla_zeros_eval (BK BV : Nat) (t : BlockState) :
    evalOp (Op.full [BK, BV] (Op.const 0)) t
      = some (⟨fun _ => some (0 : ℝ)⟩ : Tile .real [BK, BV]) := by
  simp [evalOp_full, evalOp_const]

/-- `tl.dot` at rank 2, `erw`-only shapes. -/
private theorem cla_dot_eval {M K N : Nat} (x : Op .real [M, K]) (y : Op .real [K, N])
    (t : BlockState) (vx : Tile .real [M, K]) (vy : Tile .real [K, N])
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.dot (batch := []) x y) t = some (Tile.dot [] vx vy) := by
  erw [evalOp_dot, hx, hy]
  rfl

/-- A `WithBot ℝ` sum of `some`s is `some` of the real sum. -/
private theorem cla_withBot_sum_some {N : Nat} (g : Fin N → ℝ) :
    @Finset.sum (Fin N) (WithBot ℝ) _ Finset.univ (fun k => (some (g k) : WithBot ℝ))
      = some (Finset.univ.sum g) := by
  show (Finset.univ.sum fun k => ((g k : ℝ) : WithBot ℝ))
    = ((Finset.univ.sum g : ℝ) : WithBot ℝ)
  exact (WithBot.coe_sum Finset.univ g).symm

/-- 2D dot element collapse for all-`some` operands. -/
private theorem cla_dot2d_elem {M K N : Nat} (a : Tile .real [M, K])
    (b : Tile .real [K, N]) (m : Fin M) (n : Fin N) (fa fb : Fin K → ℝ)
    (ha : ∀ e : Fin K, a.data (m, e, PUnit.unit) = some (fa e))
    (hb : ∀ e : Fin K, b.data (e, n, PUnit.unit) = some (fb e)) :
    (Tile.dot [] a b).data (m, n, PUnit.unit)
      = some (Finset.univ.sum fun e : Fin K => fa e * fb e) := by
  rw [Tile.dot_nil_data]
  rw [show (@Finset.sum (Fin K) (WithBot ℝ) _ Finset.univ
        (fun e => Option.map₂ (· * ·) (a.data (m, e, PUnit.unit))
          (b.data (e, n, PUnit.unit))))
      = @Finset.sum (Fin K) (WithBot ℝ) _ Finset.univ
          (fun e => (some (fa e * fb e) : WithBot ℝ))
      from Finset.sum_congr rfl (fun e _ => by rw [ha e, hb e]; rfl)]
  exact cla_withBot_sum_some _

/-- `Stmt.ifThen`'s step equation (well-founded recursion, so named). -/
private theorem cla_ifThen_step (cond : Op .bool []) (body : List Stmt)
    (t : BlockState) :
    stepStmt (Stmt.ifThen cond body) t
      = (evalOp cond t).bind
          (fun c => if c.data PUnit.unit then stepStmts body t else some t) := by
  unfold stepStmt
  cases evalOp cond t <;> rfl

/-- `setReg` leaves memory alone, at function level. -/
private theorem cla_setReg_mem {dtype : TileDType} {sh : TileShape}
    (s : BlockState) (nm : RegName) (v : Tile dtype sh) :
    (s.setReg nm dtype sh v).mem = s.mem := rfl

/-! ## Value tiles and load bridges

The four loads (`k`/`v` forward; `q`/`do` backward reuse the same two bridges —
`q` shares `k`'s layout and `do` shares `v`'s, with only the region name and
launch strides substituted). -/

/-- The loaded `b_k` tile of chunk `t`: cell `(e, c)` holds the guarded `k` lane. -/
noncomputable def claBkTile (s : BlockState) (k : RegionName)
    (s_qk_h s_qk_t s_qk_d T K BT BK : Nat) (t : Nat) : Tile .real [BK, BT] :=
  ⟨fun idx => some (kGuarded s k s_qk_h s_qk_t s_qk_d T K BT BK t
    idx.2.1.val idx.1.val)⟩

/-- The loaded `b_v` tile of chunk `t`: cell `(c, p)` holds the guarded `v` lane. -/
noncomputable def claBvTile (s : BlockState) (v : RegionName)
    (s_vo_h s_vo_t s_vo_d T V BT BV : Nat) (t : Nat) : Tile .real [BT, BV] :=
  ⟨fun idx => some (vGuarded s v s_vo_h s_vo_t s_vo_d T V BT BV t
    idx.1.val idx.2.1.val)⟩

/-- `b_k` (or `b_q`) load lands on `claBkTile` (memory matched to launch `s`). -/
private theorem cla_kLoad_eq (s sin : BlockState) (k : RegionName) (name : RegName)
    (s_qk_h s_qk_t s_qk_d T K BT BK : Nat) (t : Nat)
    (hmem : ∀ off, sin.readMem k off = s.readMem k off)
    (hpk : sin.regs .blockPtr [BK, BT] name = some
      ⟨fun _ => BlockPtr.mk k (s.pids 2 * s_qk_h) [K, T] [BK, BT] [s_qk_d, s_qk_t]
        [s.pids 0 * BK, t * BT]⟩) :
    evalOp (Op.load .real
        (MemAccess.blockPtr (Op.ref .blockPtr [BK, BT] name) [0, 1]) MaskOpt.none) sin
      = some (claBkTile s k s_qk_h s_qk_t s_qk_d T K BT BK t) := by
  rw [cla_load_bp_2d k sin name (s.pids 2 * s_qk_h) K T BK BT s_qk_d s_qk_t
    (s.pids 0 * BK) (t * BT) hpk]
  refine congrArg some ?_
  ext idx
  obtain ⟨e, c, u⟩ := idx
  simp only [claBkTile, kGuarded, kElem, hmem]
  by_cases hb : s.pids 0 * BK + e.val < K ∧ t * BT + c.val < T
  · rw [if_pos hb, if_pos hb]
  · rw [if_neg hb, if_neg hb]

/-- `b_v` (or `b_do`) load lands on `claBvTile`. -/
private theorem cla_vLoad_eq (s sin : BlockState) (v : RegionName) (name : RegName)
    (s_vo_h s_vo_t s_vo_d T V BT BV : Nat) (t : Nat)
    (hmem : ∀ off, sin.readMem v off = s.readMem v off)
    (hpv : sin.regs .blockPtr [BT, BV] name = some
      ⟨fun _ => BlockPtr.mk v (s.pids 2 * s_vo_h) [T, V] [BT, BV] [s_vo_t, s_vo_d]
        [t * BT, s.pids 1 * BV]⟩) :
    evalOp (Op.load .real
        (MemAccess.blockPtr (Op.ref .blockPtr [BT, BV] name) [0, 1]) MaskOpt.none) sin
      = some (claBvTile s v s_vo_h s_vo_t s_vo_d T V BT BV t) := by
  rw [cla_load_bp_2d v sin name (s.pids 2 * s_vo_h) T V BT BV s_vo_t s_vo_d
    (t * BT) (s.pids 1 * BV) hpv]
  refine congrArg some ?_
  ext idx
  obtain ⟨c, p, u⟩ := idx
  simp only [claBvTile, vGuarded, vElem, hmem]
  by_cases hb : t * BT + c.val < T ∧ s.pids 1 * BV + p.val < V
  · rw [if_pos hb, if_pos hb]
  · rw [if_neg hb, if_neg hb]

/-- The seeded `b_h` tile: cell `(e, p)` holds the guarded `h0` lane. -/
noncomputable def claH0Tile (s : BlockState) (h0 : RegionName)
    (K V BK BV : Nat) : Tile .real [BK, BV] :=
  ⟨fun idx => some (h0Guarded s h0 K V BK BV idx.1.val idx.2.1.val)⟩

/-- The `h0` load lands on `claH0Tile`. -/
private theorem cla_h0Load_eq (s sin : BlockState) (h0 : RegionName)
    (K V BK BV : Nat)
    (hmem : ∀ off, sin.readMem h0 off = s.readMem h0 off)
    (hph0 : sin.regs .blockPtr [BK, BV] "p_h0" = some
      ⟨fun _ => BlockPtr.mk h0 (s.pids 2 * K * V) [K, V] [BK, BV] [V, 1]
        [s.pids 0 * BK, s.pids 1 * BV]⟩) :
    evalOp (Op.load .real
        (MemAccess.blockPtr (Op.ref .blockPtr [BK, BV] "p_h0") [0, 1]) MaskOpt.none) sin
      = some (claH0Tile s h0 K V BK BV) := by
  rw [cla_load_bp_2d h0 sin "p_h0" (s.pids 2 * K * V) K V BK BV V 1
    (s.pids 0 * BK) (s.pids 1 * BV) hph0]
  refine congrArg some ?_
  ext idx
  obtain ⟨e, p, u⟩ := idx
  simp only [claH0Tile, h0Guarded, h0Elem, hmem]
  by_cases hb : s.pids 0 * BK + e.val < K ∧ s.pids 1 * BV + p.val < V
  · rw [if_pos hb, if_pos hb]
  · rw [if_neg hb, if_neg hb]

/-! ## The generic state-block store

All three stores (`h[·,·,t]` per chunk, `ht` once, `dh[·,·,t]` per chunk) are the
same shape: a `[BK, BV]` block of a `(K, V)` parent at strides `(σ, 1)` and
offsets `(i_k·BK, i_v·BV)`, differing only in base and stride, so one lemma pair
serves all three. -/

/-- The state-block store address at lane `(e, p)`: `base + row·σ + col`. -/
def claStoreAddr (s : BlockState) (base σ BK BV : Nat)
    (idx : TileIndex [BK, BV]) : Nat :=
  base + (s.pids 0 * BK + idx.1.val) * σ + (s.pids 1 * BV + idx.2.1.val) * 1

theorem claHOffset_eq_storeAddr (s : BlockState) (s_h_h s_h_t K V BK BV t : Nat)
    (idx : TileIndex [BK, BV]) :
    claHOffset s s_h_h s_h_t K V BK BV t idx
      = claStoreAddr s (s.pids 2 * s_h_h + t * K * V) s_h_t BK BV idx := rfl

theorem claHtOffset_eq_storeAddr (s : BlockState) (K V BK BV : Nat)
    (idx : TileIndex [BK, BV]) :
    claHtOffset s K V BK BV idx
      = claStoreAddr s (s.pids 2 * K * V) V BK BV idx := rfl

/-- The explicit post-store state: the masked lane-by-lane scatter of cell fn `f`
over input state `sin` (value and mask built from launch `s`). -/
noncomputable def claStoreState (s sin : BlockState) (rg : RegionName)
    (base σ K V BK BV : Nat) (f : Nat → Nat → ℝ) : BlockState :=
  (TileShape.allIndices [BK, BV]).foldl
    (fun acc i => if (s.pids 0 * BK + i.1.val < K ∧ s.pids 1 * BV + i.2.1.val < V)
        then acc.writeMem rg (claStoreAddr s base σ BK BV i)
          (f i.1.val i.2.1.val) else acc) sin

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **State-block store step (eq).** Stepping the block-ptr store of tile `bT`
(cell fn `f`) through pointer register `pname` yields `claStoreState`. -/
theorem claStore_step_eq (s sin : BlockState) (rg : RegionName)
    (bname pname : RegName) (base σ K V BK BV : Nat)
    (f : Nat → Nat → ℝ) (bT : Tile .real [BK, BV])
    (hbf : ∀ e p, bT.data (e, p, PUnit.unit) = some (f e.val p.val))
    (hb : sin.regs .real [BK, BV] bname = some bT)
    (hp : sin.regs .blockPtr [BK, BV] pname = some
      ⟨fun _ => BlockPtr.mk rg base [K, V] [BK, BV] [σ, 1]
        [s.pids 0 * BK, s.pids 1 * BV]⟩) :
    stepStmt (Stmt.store .real [BK, BV]
        (MemAccess.blockPtr (Op.ref .blockPtr [BK, BV] pname) [0, 1])
        (Op.ref .real [BK, BV] bname) MaskOpt.none) sin
      = some (claStoreState s sin rg base σ K V BK BV f) := by
  unfold stepStmt claStoreState
  simp only [evalOp_ref, hb, hp]
  refine congrArg some
    (congrArg (fun g => List.foldl g sin (TileShape.allIndices [BK, BV])) ?_)
  funext acc i
  obtain ⟨e, p, u⟩ := i
  simp only [TileShape.indexToList, BlockPtr.address_2d_offsets,
    BlockPtr.inBounds_2d_offsets, Bool.true_and, claStoreAddr]
  by_cases hbnd : s.pids 0 * BK + e.val < K ∧ s.pids 1 * BV + p.val < V
  · simp only [hbnd, BlockState.writeMemTyped_real, hbf, Nat.mul_one]
    rfl
  · simp only [hbnd, decide_false, Bool.false_eq_true, if_false]

/-- If two `Q`-blocks with in-block offsets `A, B < Q` collide, the block indices
agree. -/
private theorem cla_block_index_inj {Q j c A B : Nat} (hA : A < Q) (hB : B < Q)
    (heq : j * Q + A = c * Q + B) : j = c := by
  have hQ : 0 < Q := Nat.lt_of_le_of_lt (Nat.zero_le A) hA
  have hj : (j * Q + A) / Q = j := by
    rw [Nat.mul_comm, Nat.mul_add_div hQ, Nat.div_eq_of_lt hA, Nat.add_zero]
  have hc : (c * Q + B) / Q = c := by
    rw [Nat.mul_comm, Nat.mul_add_div hQ, Nat.div_eq_of_lt hB, Nat.add_zero]
  rw [← hj, heq, hc]

/-- The store addresses of one block are pairwise distinct once a whole `BV` row
segment fits under the row stride. -/
theorem claStoreAddr_injective (s : BlockState) (base σ BK BV : Nat)
    (hBVσ : BV ≤ σ) :
    Function.Injective (claStoreAddr s base σ BK BV) := by
  rintro ⟨e, p, u⟩ ⟨e', p', u'⟩ heq
  simp only [claStoreAddr] at heq
  have hp := p.isLt
  have hp' := p'.isLt
  have h2 : (s.pids 0 * BK + e.val) * σ + p.val
      = (s.pids 0 * BK + e'.val) * σ + p'.val := by omega
  have hlt : p.val < σ := by omega
  have hlt' : p'.val < σ := by omega
  have hjj : s.pids 0 * BK + e.val = s.pids 0 * BK + e'.val :=
    cla_block_index_inj hlt hlt' h2
  have he : e = e' := Fin.ext (by omega)
  have hpv : p = p' := Fin.ext (by
    have hσ : (s.pids 0 * BK + e.val) * σ = (s.pids 0 * BK + e'.val) * σ := by
      rw [hjj]
    omega)
  subst he
  subst hpv
  rfl

set_option maxHeartbeats 4000000 in
/-- **State-block store readback.** `claStoreState` writes `f` at every active
lane, leaves other regions alone, and leaves same-region offsets alone when they
avoid every *active*-lane address (mask-restricted, so no pid pinning needed). -/
theorem claStore_step_props (s sin : BlockState) (rg : RegionName)
    (base σ K V BK BV : Nat) (f : Nat → Nat → ℝ)
    (hInj : Function.Injective (claStoreAddr s base σ BK BV)) :
    (claStoreState s sin rg base σ K V BK BV f).pids = sin.pids
      ∧ (claStoreState s sin rg base σ K V BK BV f).regs = sin.regs
      ∧ (∀ idx : TileIndex [BK, BV], claActive s K V BK BV idx →
          (claStoreState s sin rg base σ K V BK BV f).readMem rg
              (claStoreAddr s base σ BK BV idx)
            = f idx.1.val idx.2.1.val)
      ∧ (∀ rg' off, rg' ≠ rg →
          (claStoreState s sin rg base σ K V BK BV f).readMem rg' off
            = sin.readMem rg' off)
      ∧ (∀ off, (∀ idx : TileIndex [BK, BV], claActive s K V BK BV idx →
            off ≠ claStoreAddr s base σ BK BV idx) →
          (claStoreState s sin rg base σ K V BK BV f).readMem rg off
            = sin.readMem rg off) := by
  classical
  unfold claStoreState
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · rw [BlockState.foldl_writeMem_prop_masked_pids]
  · funext dtype shape name
    rw [BlockState.foldl_writeMem_prop_masked_regs]
  · intro idx hidx
    obtain ⟨h1, h2⟩ := hidx
    have h := BlockState.scatter_readback_prop_masked_nd (region := rg) sin
      (claStoreAddr s base σ BK BV) (fun i => f i.1.val i.2.1.val)
      (fun i => s.pids 0 * BK + i.1.val < K ∧ s.pids 1 * BV + i.2.1.val < V)
      hInj idx
    rw [h, if_pos ⟨h1, h2⟩]
  · intro rg' off hrg
    exact BlockState.foldl_writeMem_const_region_prop_masked_readMem_other
      rg (claStoreAddr s base σ BK BV) (fun i => f i.1.val i.2.1.val)
      (fun i => s.pids 0 * BK + i.1.val < K ∧ s.pids 1 * BV + i.2.1.val < V)
      _ sin rg' off hrg
  · intro off hoff
    exact BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
      rg (claStoreAddr s base σ BK BV) (fun i => f i.1.val i.2.1.val)
      (fun i => s.pids 0 * BK + i.1.val < K ∧ s.pids 1 * BV + i.2.1.val < V)
      _ sin off (fun i _ hPi => hoff i hPi)

/-! ### Cross-chunk block disjointness -/

/-- Distinct chunks write disjoint `K*V` blocks of `h` / `dh`: active lanes of
chunk `j` never collide with active lanes of chunk `c ≠ j`. The block-fit
hypothesis `(K-1)·s_h_t + V ≤ K·V` is an equality under the launcher's
contiguous `s_h_t = V`. -/
theorem claHOffset_chunk_disjoint (s : BlockState)
    (s_h_h s_h_t K V BK BV : Nat) (hFit : (K - 1) * s_h_t + V ≤ K * V)
    (j c : Nat) (hjc : j ≠ c) (idxj idxc : TileIndex [BK, BV])
    (hja : claActive s K V BK BV idxj) (hca : claActive s K V BK BV idxc) :
    claHOffset s s_h_h s_h_t K V BK BV j idxj
      ≠ claHOffset s s_h_h s_h_t K V BK BV c idxc := by
  obtain ⟨ej, pj, _⟩ := idxj
  obtain ⟨ec, pc, _⟩ := idxc
  obtain ⟨hjK, hjV⟩ := hja
  obtain ⟨hcK, hcV⟩ := hca
  simp only [claHOffset]
  intro heq
  apply hjc
  have hbound : ∀ R C : Nat, R < K → C < V → R * s_h_t + C < K * V := by
    intro R C hR hC
    have h1 : R * s_h_t ≤ (K - 1) * s_h_t := Nat.mul_le_mul_right s_h_t (by omega)
    omega
  have hjm : j * (K * V) = j * K * V := by rw [Nat.mul_assoc]
  have hcm : c * (K * V) = c * K * V := by rw [Nat.mul_assoc]
  have heq2 : j * (K * V) + ((s.pids 0 * BK + ej.val) * s_h_t + (s.pids 1 * BV + pj.val))
      = c * (K * V) + ((s.pids 0 * BK + ec.val) * s_h_t + (s.pids 1 * BV + pc.val)) := by
    omega
  exact cla_block_index_inj
    (by have := hbound _ _ hjK hjV; omega)
    (by have := hbound _ _ hcK hcV; omega)
    heq2

/-! ## Spec-layer recurrences -/

theorem claHState_zero (s : BlockState) (k v h0 : RegionName) (UIS : Bool)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat)
    (T K V BT BK BV : Nat) (e p : Nat) :
    claHState s k v h0 UIS s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
        T K V BT BK BV 0 e p
      = (if UIS then h0Guarded s h0 K V BK BV e p else 0) := by
  simp [claHState]

theorem claHState_succ (s : BlockState) (k v h0 : RegionName) (UIS : Bool)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat)
    (T K V BT BK BV : Nat) (t e p : Nat) :
    claHState s k v h0 UIS s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
        T K V BT BK BV (t + 1) e p
      = claHState s k v h0 UIS s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
          T K V BT BK BV t e p
        + claContrib s k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
            T K V BT BK BV t e p := by
  simp [claHState, Fin.sum_univ_castSucc, add_assoc]

/-- The backward running state after `c` descending iterations: every chunk from
`NT - c` up. Stored (before advancing) at chunk `NT - 1 - c`. -/
noncomputable def claDhCarry (s : BlockState) (q do_ : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (T K V BT BK BV NT : Nat) (c e p : Nat) : ℝ :=
  ∑ u : Fin NT, if NT - c ≤ u.val then
    claBContrib s q do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
      T K V BT BK BV u.val e p
  else 0

theorem claDhCarry_zero (s : BlockState) (q do_ : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (T K V BT BK BV NT : Nat) (e p : Nat) :
    claDhCarry s q do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
        T K V BT BK BV NT 0 e p = 0 := by
  refine Finset.sum_eq_zero fun u _ => ?_
  rw [if_neg]
  have := u.isLt
  omega

theorem claDhCarry_succ (s : BlockState) (q do_ : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (T K V BT BK BV NT : Nat) (c e p : Nat) (hc : c < NT) :
    claDhCarry s q do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
        T K V BT BK BV NT (c + 1) e p
      = claDhCarry s q do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
          T K V BT BK BV NT c e p
        + claBContrib s q do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
            T K V BT BK BV (NT - 1 - c) e p := by
  classical
  unfold claDhCarry
  set B : Nat → ℝ := fun uv =>
    claBContrib s q do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
      T K V BT BK BV uv e p with hB
  have hpt : ∀ u : Fin NT,
      (if NT - (c + 1) ≤ u.val then B u.val else 0)
        = (if NT - c ≤ u.val then B u.val else 0)
          + (if u = (⟨NT - 1 - c, by omega⟩ : Fin NT) then B u.val else 0) := by
    intro u
    by_cases h1 : NT - c ≤ u.val
    · rw [if_pos (by omega), if_pos h1, if_neg (by
        intro hu
        have : u.val = NT - 1 - c := by rw [hu]
        omega), add_zero]
    · by_cases h2 : u = (⟨NT - 1 - c, by omega⟩ : Fin NT)
      · have hval : u.val = NT - 1 - c := by rw [h2]
        rw [if_pos (by omega), if_neg h1, if_pos h2, zero_add]
      · have hval : u.val ≠ NT - 1 - c := by
          intro hv
          exact h2 (Fin.ext hv)
        rw [if_neg (by omega), if_neg h1, if_neg h2, add_zero]
  rw [Finset.sum_congr rfl (fun u _ => hpt u), Finset.sum_add_distrib,
    Finset.sum_ite_eq' Finset.univ (⟨NT - 1 - c, by omega⟩ : Fin NT)
      (fun u => B u.val), if_pos (Finset.mem_univ _)]

/-- What the backward kernel stores at chunk `NT - 1 - c` is exactly the spec's
`claDhState` there: the strictly-later chunks. -/
theorem claDhState_eq_carry (s : BlockState) (q do_ : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (T K V BT BK BV NT : Nat) (c e p : Nat) (hc : c < NT) :
    claDhState s q do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
        T K V BT BK BV NT (NT - 1 - c) e p
      = claDhCarry s q do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
          T K V BT BK BV NT c e p := by
  unfold claDhState claDhCarry
  exact Finset.sum_congr rfl fun u _ => if_congr (by omega) rfl rfl

/-! ## The forward walk -/

/-- Tile `+` with both operand values known. -/
private theorem cla_addTile_eval {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op .real a) (y : Op .real b) (t : BlockState)
    (vx : Tile .real a) (vy : Tile .real b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.add .real bc x y) t
      = some (Tile.bop NumericDType.real.add bc vx vy) := by
  rw [evalOp_add, hx, hy]
  rfl

/-- The carried forward state tile at chunk `t` — cell `(e, p)` is `claHState`. -/
noncomputable def claHTile (s : BlockState) (k v h0 : RegionName) (UIS : Bool)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat)
    (T K V BT BK BV : Nat) (t : Nat) : Tile .real [BK, BV] :=
  ⟨fun idx => some (claHState s k v h0 UIS s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
    T K V BT BK BV t idx.1.val idx.2.1.val)⟩

/-- The recurrence statement: `b_h + tl.dot(b_k, b_v)` advances the state tile by
one chunk. -/
private theorem cla_fwd_advance_eval (s sin : BlockState) (k v h0 : RegionName)
    (UIS : Bool) (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat)
    (T K V BT BK BV : Nat) (t : Nat)
    (hbh : sin.regs .real [BK, BV] "b_h" = some (claHTile s k v h0 UIS
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d T K V BT BK BV t))
    (hbk : sin.regs .real [BK, BT] "b_k"
      = some (claBkTile s k s_qk_h s_qk_t s_qk_d T K BT BK t))
    (hbv : sin.regs .real [BT, BV] "b_v"
      = some (claBvTile s v s_vo_h s_vo_t s_vo_d T V BT BV t)) :
    evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BK, BV] "b_h")
        (Op.dot (batch := []) (Op.ref .real [BK, BT] "b_k")
          (Op.ref .real [BT, BV] "b_v"))) sin
      = some (claHTile s k v h0 UIS s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
          T K V BT BK BV (t + 1)) := by
  rw [cla_addTile_eval (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
    (Op.ref .real [BK, BV] "b_h")
    (Op.dot (batch := []) (Op.ref .real [BK, BT] "b_k")
      (Op.ref .real [BT, BV] "b_v")) sin
    (claHTile s k v h0 UIS s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
      T K V BT BK BV t)
    (Tile.dot [] (claBkTile s k s_qk_h s_qk_t s_qk_d T K BT BK t)
      (claBvTile s v s_vo_h s_vo_t s_vo_d T V BT BV t))
    (by rw [evalOp_ref]; exact hbh)
    (cla_dot_eval _ _ sin _ _ (by rw [evalOp_ref]; exact hbk)
      (by rw [evalOp_ref]; exact hbv))]
  refine congrArg some (Tile.ext fun idx => ?_)
  obtain ⟨e, p, u⟩ := idx
  simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, claHTile,
    NumericDType.add, WithBot.realAdd]
  rw [cla_dot2d_elem _ _ e p
    (fun c => kGuarded s k s_qk_h s_qk_t s_qk_d T K BT BK t c.val e.val)
    (fun c => vGuarded s v s_vo_h s_vo_t s_vo_d T V BT BV t c.val p.val)
    (fun c => rfl) (fun c => rfl)]
  rw [show ∀ x y : ℝ, Option.map₂ (· + ·) (some x : WithBot ℝ) (some y) = some (x + y)
    from fun _ _ => rfl]
  refine congrArg some ?_
  rw [claHState_succ]
  rfl

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **One forward chunk.** Stores the carried state at chunk `t`'s block of `h`,
then advances the state by chunk `t`'s `kᵀ·v`. -/
theorem claFwdBody_step (s sin : BlockState) (k v h h0 : RegionName) (UIS : Bool)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat)
    (T K V BT BK BV : Nat) (t : Nat)
    (hHk : h ≠ k) (hHv : h ≠ v) (hσ : BV ≤ s_h_t)
    (hmemK : ∀ off, sin.readMem k off = s.readMem k off)
    (hmemV : ∀ off, sin.readMem v off = s.readMem v off)
    (hik : sin.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0)))
    (hiv : sin.regs .nat [] "i_v" = some (Tile.scalar (s.pids 1)))
    (hibh : sin.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2)))
    (hit : sin.regs .nat [] "i_t" = some (Tile.scalar t))
    (hbh : sin.regs .real [BK, BV] "b_h" = some (claHTile s k v h0 UIS
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d T K V BT BK BV t)) :
    ∃ s', stepStmts (claFwdBody k v h s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
        s_h_h s_h_t T K V BT BK BV) sin = some s'
      ∧ s'.pids = sin.pids
      ∧ s'.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0))
      ∧ s'.regs .nat [] "i_v" = some (Tile.scalar (s.pids 1))
      ∧ s'.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2))
      ∧ s'.regs .real [BK, BV] "b_h" = some (claHTile s k v h0 UIS
          s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d T K V BT BK BV (t + 1))
      ∧ (∀ rg off, rg ≠ h → s'.readMem rg off = sin.readMem rg off)
      ∧ (∀ idx : TileIndex [BK, BV], claActive s K V BK BV idx →
          s'.readMem h (claHOffset s s_h_h s_h_t K V BK BV t idx)
            = claHState s k v h0 UIS s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
                T K V BT BK BV t idx.1.val idx.2.1.val)
      ∧ (∀ off, (∀ idx : TileIndex [BK, BV], claActive s K V BK BV idx →
            off ≠ claHOffset s s_h_h s_h_t K V BK BV t idx) →
          s'.readMem h off = sin.readMem h off) := by
  unfold claFwdBody
  -- p_k
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cla_makeBlockPtr_2d_eval k sin _ _ _ [K, T] [BK, BT] [s_qk_d, s_qk_t]
      (s.pids 2 * s_qk_h) (s.pids 0 * BK) (t * BT)
      (cla_mulConst_eval sin "i_bh" (s.pids 2) s_qk_h hibh)
      (cla_mulConst_eval sin "i_k" (s.pids 0) BK hik)
      (cla_mulConst_eval sin "i_t" t BT hit)))]
  -- p_v
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cla_makeBlockPtr_2d_eval v _ _ _ _ [T, V] [BT, BV] [s_vo_t, s_vo_d]
      (s.pids 2 * s_vo_h) (t * BT) (s.pids 1 * BV)
      (cla_mulConst_eval _ "i_bh" (s.pids 2) s_vo_h (by simpa using hibh))
      (cla_mulConst_eval _ "i_t" t BT (by simpa using hit))
      (cla_mulConst_eval _ "i_v" (s.pids 1) BV (by simpa using hiv))))]
  -- p_h
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cla_makeBlockPtr_2d_eval h _ _ _ _ [K, V] [BK, BV] [s_h_t, 1]
      (s.pids 2 * s_h_h + t * K * V) (s.pids 0 * BK) (s.pids 1 * BV)
      (cla_hBase_eval _ s_h_h K V (s.pids 2) t (by simpa using hibh)
        (by simpa using hit))
      (cla_mulConst_eval _ "i_k" (s.pids 0) BK (by simpa using hik))
      (cla_mulConst_eval _ "i_v" (s.pids 1) BV (by simpa using hiv))))]
  -- the state store into h's chunk-t block
  rw [stepStmts.cons_some (claStore_step_eq s _ h "b_h" "p_h"
    (s.pids 2 * s_h_h + t * K * V) s_h_t K V BK BV
    (fun e p => claHState s k v h0 UIS s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
      T K V BT BK BV t e p)
    (claHTile s k v h0 UIS s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
      T K V BT BK BV t)
    (fun e p => rfl)
    (by simpa using hbh)
    (by simp))]
  obtain ⟨hSpids, hSregs, hSact, hSoth, hSoff⟩ :=
    claStore_step_props s
      (((sin.setReg "p_k" .blockPtr [BK, BT]
          (⟨fun _ => BlockPtr.mk k (s.pids 2 * s_qk_h) [K, T] [BK, BT]
            [s_qk_d, s_qk_t] [s.pids 0 * BK, t * BT]⟩ : Tile .blockPtr [BK, BT])).setReg
        "p_v" .blockPtr [BT, BV]
          (⟨fun _ => BlockPtr.mk v (s.pids 2 * s_vo_h) [T, V] [BT, BV]
            [s_vo_t, s_vo_d] [t * BT, s.pids 1 * BV]⟩ : Tile .blockPtr [BT, BV])).setReg
        "p_h" .blockPtr [BK, BV]
          (⟨fun _ => BlockPtr.mk h (s.pids 2 * s_h_h + t * K * V) [K, V] [BK, BV]
            [s_h_t, 1] [s.pids 0 * BK, s.pids 1 * BV]⟩ : Tile .blockPtr [BK, BV]))
      h (s.pids 2 * s_h_h + t * K * V) s_h_t K V BK BV
      (fun e p => claHState s k v h0 UIS s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
        T K V BT BK BV t e p)
      (claStoreAddr_injective s (s.pids 2 * s_h_h + t * K * V) s_h_t BK BV hσ)
  set sSt := claStoreState s
      (((sin.setReg "p_k" .blockPtr [BK, BT]
          (⟨fun _ => BlockPtr.mk k (s.pids 2 * s_qk_h) [K, T] [BK, BT]
            [s_qk_d, s_qk_t] [s.pids 0 * BK, t * BT]⟩ : Tile .blockPtr [BK, BT])).setReg
        "p_v" .blockPtr [BT, BV]
          (⟨fun _ => BlockPtr.mk v (s.pids 2 * s_vo_h) [T, V] [BT, BV]
            [s_vo_t, s_vo_d] [t * BT, s.pids 1 * BV]⟩ : Tile .blockPtr [BT, BV])).setReg
        "p_h" .blockPtr [BK, BV]
          (⟨fun _ => BlockPtr.mk h (s.pids 2 * s_h_h + t * K * V) [K, V] [BK, BV]
            [s_h_t, 1] [s.pids 0 * BK, s.pids 1 * BV]⟩ : Tile .blockPtr [BK, BV]))
      h (s.pids 2 * s_h_h + t * K * V) s_h_t K V BK BV
      (fun e p => claHState s k v h0 UIS s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
        T K V BT BK BV t e p) with hsSt
  -- b_k load
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cla_kLoad_eq s sSt k "p_k" s_qk_h s_qk_t s_qk_d T K BT BK t
      (fun off => by
        rw [hSoth k off (Ne.symm hHk)]
        simpa using hmemK off)
      (by rw [hSregs]; simp)))]
  -- b_v load
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cla_vLoad_eq s _ v "p_v" s_vo_h s_vo_t s_vo_d T V BT BV t
      (fun off => by
        rw [BlockState.setReg_readMem, hSoth v off (Ne.symm hHv)]
        simpa using hmemV off)
      (by
        rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hSregs]
        simp)))]
  -- b_h advance
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cla_fwd_advance_eval s _ k v h0 UIS s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
      T K V BT BK BV t
      (by
        rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
          BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hSregs]
        simpa using hbh)
      (by
        rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
        simp)
      (by simp)))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · -- pids
    simp only [BlockState.setReg_pids, hSpids]
  · -- i_k
    rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hSregs]
    simpa using hik
  · -- i_v
    rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hSregs]
    simpa using hiv
  · -- i_bh
    rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hSregs]
    simpa using hibh
  · -- b_h
    rw [BlockState.setReg_same]
  · -- other regions unchanged
    intro rg off hrg
    rw [BlockState.setReg_readMem, BlockState.setReg_readMem,
      BlockState.setReg_readMem, hSoth rg off hrg]
    simp
  · -- chunk-t active readback
    intro idx hidx
    rw [BlockState.setReg_readMem, BlockState.setReg_readMem,
      BlockState.setReg_readMem, claHOffset_eq_storeAddr]
    exact hSact idx hidx
  · -- h off the active block unchanged
    intro off hoff
    rw [BlockState.setReg_readMem, BlockState.setReg_readMem,
      BlockState.setReg_readMem,
      hSoff off (fun idx hidx => by
        rw [← claHOffset_eq_storeAddr]
        exact hoff idx hidx)]
    simp

/-- An `ifThen` with a `false` constexpr condition is a no-op. -/
private theorem cla_ifThen_false_noop (body : List Stmt) (X : BlockState) :
    stepStmt (Stmt.ifThen (Op.constBool Bool.false) body) X = some X := by
  simp [stepStmt, evalOp]

/-- An `ifThen` with a `true` constexpr condition runs its body. -/
private theorem cla_ifThen_true_run (body : List Stmt) (X : BlockState) :
    stepStmt (Stmt.ifThen (Op.constBool Bool.true) body) X = stepStmts body X := by
  simp [stepStmt, evalOp]

/-- **The `USE_INITIAL_STATE` gate.** Both configurations leave `b_h` holding the
chunk-`0` state (the gated seed): the `true` branch loads `h0`, the `false`
branch keeps the zeros. No memory is written either way. -/
theorem claInitBranch_run (s s0 : BlockState) (k v h0 : RegionName) (UIS : Bool)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (T K V BT BK BV : Nat)
    (hpids : s0.pids = s.pids)
    (hik : s0.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0)))
    (hiv : s0.regs .nat [] "i_v" = some (Tile.scalar (s.pids 1)))
    (hibh : s0.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2)))
    (hbh : s0.regs .real [BK, BV] "b_h"
      = some (⟨fun _ => some 0⟩ : Tile .real [BK, BV]))
    (hmem : ∀ off, s0.readMem h0 off = s.readMem h0 off) :
    ∃ s1, stepStmt (Stmt.ifThen (Op.constBool UIS) (claInitBranch h0 K V BK BV)) s0
        = some s1
      ∧ s1.pids = s.pids
      ∧ (∀ rg off, s1.readMem rg off = s0.readMem rg off)
      ∧ s1.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0))
      ∧ s1.regs .nat [] "i_v" = some (Tile.scalar (s.pids 1))
      ∧ s1.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2))
      ∧ s1.regs .real [BK, BV] "b_h" = some (claHTile s k v h0 UIS
          s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d T K V BT BK BV 0) := by
  cases UIS
  · refine ⟨s0, cla_ifThen_false_noop _ _, hpids, fun _ _ => rfl,
        hik, hiv, hibh, ?_⟩
    rw [hbh]
    refine congrArg some (Tile.ext fun idx => ?_)
    simp [claHTile, claHState]
  ·
      rw [cla_ifThen_true_run]
      unfold claInitBranch
      rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (cla_makeBlockPtr_2d_eval h0 s0 _ _ _ [K, V] [BK, BV] [V, 1]
          (s.pids 2 * K * V) (s.pids 0 * BK) (s.pids 1 * BV)
          (cla_mulMulConst_eval s0 "i_bh" (s.pids 2) K V hibh)
          (cla_mulConst_eval s0 "i_k" (s.pids 0) BK hik)
          (cla_mulConst_eval s0 "i_v" (s.pids 1) BV hiv)))]
      rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (cla_castFloat_eval _ _ _
          (cla_h0Load_eq s _ h0 K V BK BV
            (fun off => by simpa using hmem off)
            (by simp))))]
      rw [stepStmts.nil]
      refine ⟨_, rfl, by simp [hpids], fun rg off => by simp, ?_, ?_, ?_, ?_⟩
      · simpa using hik
      · simpa using hiv
      · simpa using hibh
      · rw [BlockState.setReg_same]
        refine congrArg some (Tile.ext fun idx => ?_)
        simp [claH0Tile, claHTile, claHState]

/-- **The `STORE_FINAL_STATE` gate.** When set, flushes the post-loop state
`claHState NT` to `ht`; either way, every other region is untouched. -/
theorem claFinalBranch_run (s sL : BlockState) (k v h0 ht : RegionName)
    (UIS SFS : Bool) (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat)
    (T K V BT BK BV NT : Nat) (hBVV : BV ≤ V)
    (hik : sL.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0)))
    (hiv : sL.regs .nat [] "i_v" = some (Tile.scalar (s.pids 1)))
    (hibh : sL.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2)))
    (hbh : sL.regs .real [BK, BV] "b_h" = some (claHTile s k v h0 UIS
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d T K V BT BK BV NT)) :
    ∃ sF, stepStmt (Stmt.ifThen (Op.constBool SFS) (claFinalBranch ht K V BK BV)) sL
        = some sF
      ∧ (∀ rg off, rg ≠ ht → sF.readMem rg off = sL.readMem rg off)
      ∧ (SFS = Bool.true → ∀ idx : TileIndex [BK, BV], claActive s K V BK BV idx →
          sF.readMem ht (claHtOffset s K V BK BV idx)
            = claHState s k v h0 UIS s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
                T K V BT BK BV NT idx.1.val idx.2.1.val) := by
  cases SFS
  · exact ⟨sL, cla_ifThen_false_noop _ _, fun _ _ _ => rfl,
      fun hSFS => nomatch hSFS⟩
  ·
      rw [cla_ifThen_true_run]
      unfold claFinalBranch
      rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (cla_makeBlockPtr_2d_eval ht sL _ _ _ [K, V] [BK, BV] [V, 1]
          (s.pids 2 * K * V) (s.pids 0 * BK) (s.pids 1 * BV)
          (cla_mulMulConst_eval sL "i_bh" (s.pids 2) K V hibh)
          (cla_mulConst_eval sL "i_k" (s.pids 0) BK hik)
          (cla_mulConst_eval sL "i_v" (s.pids 1) BV hiv)))]
      rw [stepStmts.cons_some (claStore_step_eq s _ ht "b_h" "p_ht"
        (s.pids 2 * K * V) V K V BK BV
        (fun e p => claHState s k v h0 UIS s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t
          s_vo_d T K V BT BK BV NT e p)
        (claHTile s k v h0 UIS s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
          T K V BT BK BV NT)
        (fun e p => rfl)
        (by simpa using hbh)
        (by simp))]
      rw [stepStmts.nil]
      obtain ⟨hFpids, hFregs, hFact, hFoth, hFoff⟩ :=
        claStore_step_props s
          (sL.setReg "p_ht" .blockPtr [BK, BV]
            (⟨fun _ => BlockPtr.mk ht (s.pids 2 * K * V) [K, V] [BK, BV] [V, 1]
              [s.pids 0 * BK, s.pids 1 * BV]⟩ : Tile .blockPtr [BK, BV]))
          ht (s.pids 2 * K * V) V K V BK BV
          (fun e p => claHState s k v h0 UIS s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t
            s_vo_d T K V BT BK BV NT e p)
          (claStoreAddr_injective s (s.pids 2 * K * V) V BK BV hBVV)
      refine ⟨_, rfl, ?_, ?_⟩
      · intro rg off hrg
        rw [hFoth rg off hrg]
        simp
      · intro _ idx hidx
        rw [claHtOffset_eq_storeAddr]
        exact hFact idx hidx

set_option maxHeartbeats 4000000 in
/-- **The collapsed chunk loop.** `forRange_inv` over the carried state, the
per-chunk store history, and the untouched-regions frame. -/
theorem claFwdLoop_run (s sPre : BlockState) (k v h h0 : RegionName) (UIS : Bool)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat)
    (T K V BT BK BV NT : Nat)
    (hHk : h ≠ k) (hHv : h ≠ v) (hσ : BV ≤ s_h_t)
    (hFit : (K - 1) * s_h_t + V ≤ K * V)
    (hpids : sPre.pids = s.pids)
    (hik : sPre.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0)))
    (hiv : sPre.regs .nat [] "i_v" = some (Tile.scalar (s.pids 1)))
    (hibh : sPre.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2)))
    (hbh : sPre.regs .real [BK, BV] "b_h" = some (claHTile s k v h0 UIS
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d T K V BT BK BV 0))
    (hmem : ∀ rg off, sPre.readMem rg off = s.readMem rg off) :
    ∃ sL, stepStmt (Stmt.forRange "i_t" 0 NT 1
        (claFwdBody k v h s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t
          T K V BT BK BV)) sPre = some sL
      ∧ sL.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0))
      ∧ sL.regs .nat [] "i_v" = some (Tile.scalar (s.pids 1))
      ∧ sL.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2))
      ∧ sL.regs .real [BK, BV] "b_h" = some (claHTile s k v h0 UIS
          s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d T K V BT BK BV NT)
      ∧ (∀ rg off, rg ≠ h → sL.readMem rg off = s.readMem rg off)
      ∧ (∀ j (idx : TileIndex [BK, BV]), j < NT → claActive s K V BK BV idx →
          sL.readMem h (claHOffset s s_h_h s_h_t K V BK BV j idx)
            = claHState s k v h0 UIS s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
                T K V BT BK BV j idx.1.val idx.2.1.val) := by
  have hres := forRange_inv (idx := "i_t") (start := 0) (stop := NT) (step := 1)
    (body := claFwdBody k v h s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t
      T K V BT BK BV)
    (P := fun c tS =>
      c ≤ NT
      ∧ tS.pids = s.pids
      ∧ tS.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0))
      ∧ tS.regs .nat [] "i_v" = some (Tile.scalar (s.pids 1))
      ∧ tS.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2))
      ∧ tS.regs .real [BK, BV] "b_h" = some (claHTile s k v h0 UIS
          s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d T K V BT BK BV c)
      ∧ (∀ rg off, rg ≠ h → tS.readMem rg off = s.readMem rg off)
      ∧ (∀ j (idx : TileIndex [BK, BV]), j < c → claActive s K V BK BV idx →
          tS.readMem h (claHOffset s s_h_h s_h_t K V BK BV j idx)
            = claHState s k v h0 UIS s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
                T K V BT BK BV j idx.1.val idx.2.1.val))
    (s_init := sPre) (by decide)
    ⟨Nat.zero_le _, hpids, hik, hiv, hibh, hbh,
      fun rg off _ => hmem rg off,
      fun j idx hj _ => absurd hj (Nat.not_lt_zero j)⟩
    (by
      intro i tS hi hP
      obtain ⟨hle, hPpids, hPik, hPiv, hPibh, hPbh, hPoth, hPhist⟩ := hP
      obtain ⟨s', hstep, hspids, hsik, hsiv, hsibh, hsbh, hsoth, hsact, hsoff⟩ :=
        claFwdBody_step s (tS.setReg "i_t" .nat [] (Tile.scalar i)) k v h h0 UIS
          s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t T K V BT BK BV i
          hHk hHv hσ
          (fun off => by
            rw [BlockState.setReg_readMem]
            exact hPoth k off (Ne.symm hHk))
          (fun off => by
            rw [BlockState.setReg_readMem]
            exact hPoth v off (Ne.symm hHv))
          (by simpa using hPik)
          (by simpa using hPiv)
          (by simpa using hPibh)
          (by simp)
          (by simpa using hPbh)
      refine ⟨s', hstep, by omega,
        by rw [hspids, BlockState.setReg_pids]; exact hPpids,
        hsik, hsiv, hsibh, hsbh, ?_, ?_⟩
      · intro rg off hrg
        rw [hsoth rg off hrg, BlockState.setReg_readMem]
        exact hPoth rg off hrg
      · intro j idx hj hidx
        rcases Nat.lt_or_ge j i with hji | hji
        · -- an older chunk: preserved across chunk i's store
          rw [hsoff (claHOffset s s_h_h s_h_t K V BK BV j idx)
            (fun idx' hidx' => claHOffset_chunk_disjoint s s_h_h s_h_t K V BK BV
              hFit j i (by omega) idx idx' hidx hidx'),
            BlockState.setReg_readMem]
          exact hPhist j idx hji hidx
        · -- j = i: the fresh store
          have hji' : j = i := by omega
          subst hji'
          exact hsact idx hidx)
  obtain ⟨final, sL, hstep, hfin, hle, hLpids, hLik, hLiv, hLibh, hLbh, hLoth,
    hLhist⟩ := hres
  have hfinal : final = NT := Nat.le_antisymm hle hfin
  subst hfinal
  exact ⟨sL, hstep, hLik, hLiv, hLibh, hLbh, hLoth, fun j idx hj hidx =>
    hLhist j idx hj hidx⟩

/-- **The shared prologue.** Both kernels open with the three program ids and a
zeroed `[BK, BV]` state register (`b_h` forward, `b_dh` backward). -/
private theorem claPreLoop_run (s : BlockState) (BK BV : Nat) (bname : RegName)
    (hb1 : "i_k" ≠ bname) (hb2 : "i_v" ≠ bname) (hb3 : "i_bh" ≠ bname) :
    ∃ sP, (∀ rest : List Stmt, stepStmts
        (Stmt.assign .nat [] "i_k" (Op.programId 0)
          :: Stmt.assign .nat [] "i_v" (Op.programId 1)
          :: Stmt.assign .nat [] "i_bh" (Op.programId 2)
          :: Stmt.assign .real [BK, BV] bname (Op.full [BK, BV] (Op.const 0))
          :: rest) s = stepStmts rest sP)
      ∧ sP.pids = s.pids
      ∧ sP.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0))
      ∧ sP.regs .nat [] "i_v" = some (Tile.scalar (s.pids 1))
      ∧ sP.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2))
      ∧ sP.regs .real [BK, BV] bname = some (⟨fun _ => some 0⟩ : Tile .real [BK, BV])
      ∧ (∀ rg off, sP.readMem rg off = s.readMem rg off) := by
  refine ⟨(((s.setReg "i_k" .nat [] (Tile.scalar (s.pids 0))).setReg
        "i_v" .nat [] (Tile.scalar (s.pids 1))).setReg
        "i_bh" .nat [] (Tile.scalar (s.pids 2))).setReg
        bname .real [BK, BV] (⟨fun _ => some 0⟩ : Tile .real [BK, BV]),
    fun rest => ?_, ?_, ?_, ?_, ?_, ?_, fun rg off => ?_⟩
  · rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 0 s)),
      stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 1 _)),
      stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 2 _)),
      stepStmts.cons_some (stepStmt_assign_eq_some (cla_zeros_eval BK BV _))]
    rfl
  · rw [BlockState.setReg_pids, BlockState.setReg_pids, BlockState.setReg_pids,
      BlockState.setReg_pids]
  · rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hb1,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_same]
  · rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hb2,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_same]
  · rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hb3,
      BlockState.setReg_same]
  · rw [BlockState.setReg_same]
  · rw [BlockState.setReg_readMem, BlockState.setReg_readMem,
      BlockState.setReg_readMem, BlockState.setReg_readMem]

set_option maxHeartbeats 1000000 in
/-- **Genuine, dimension-general correctness** of the file's first kernel,
`chunk_linear_attn_fwd_kernel_h`. For every launch state the kernel runs to
completion; every active lane of every chunk block `h[·,·,t]` (`t < NT`) holds
the **pre-chunk** state `claHState t` — the gated `h0` seed plus every chunk
before `t` — and, when `STORE_FINAL_STATE` is set, `ht` holds the post-loop
state `claHState NT`. One theorem covers all four gate configurations, with
every dimension, stride, and the chunk count symbolic.

The three layout hypotheses are equalities/immediate under the launcher's
contiguous `h` (`s_h_t = V`): `BV ≤ s_h_t` makes one block's store lanes
injective, and the block-fit `(K-1)·s_h_t + V ≤ K·V` makes distinct chunks'
active lanes land in disjoint `K·V` blocks. -/
specification cla_fwd_h_exec_genuine
    (k v h h0 ht : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat)
    (T K V BT BK BV NT : Nat) (USE_INITIAL_STATE STORE_FINAL_STATE : Bool)
    (s : BlockState)
    (hHk : h ≠ k) (hHv : h ≠ v) (hHtH : ht ≠ h)
    (hσ : BV ≤ s_h_t) (hFit : (K - 1) * s_h_t + V ≤ K * V) (hBVV : BV ≤ V) :
    ∃ sF, exec (cla_fwd_h_surface k v h h0 ht s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t
        s_vo_d s_h_h s_h_t T K V BT BK BV NT
        USE_INITIAL_STATE STORE_FINAL_STATE).toAlgKernel s = some sF
      ∧ (∀ t (idx : TileIndex [BK, BV]), t < NT → claActive s K V BK BV idx →
          sF.readMem h (claHOffset s s_h_h s_h_t K V BK BV t idx)
            = claHState s k v h0 USE_INITIAL_STATE s_qk_h s_qk_t s_qk_d s_vo_h
                s_vo_t s_vo_d T K V BT BK BV t idx.1.val idx.2.1.val)
      ∧ (STORE_FINAL_STATE = Bool.true →
          ∀ idx : TileIndex [BK, BV], claActive s K V BK BV idx →
          sF.readMem ht (claHtOffset s K V BK BV idx)
            = claHState s k v h0 USE_INITIAL_STATE s_qk_h s_qk_t s_qk_d s_vo_h
                s_vo_t s_vo_d T K V BT BK BV NT idx.1.val idx.2.1.val) := by
  obtain ⟨sP, hpre, hPpids, hPik, hPiv, hPibh, hPbh, hPmem⟩ :=
    claPreLoop_run s BK BV "b_h" (by decide) (by decide) (by decide)
  rw [exec, cla_fwd_body_eq, hpre]
  obtain ⟨s1, hinit, h1pids, h1mem, h1ik, h1iv, h1ibh, h1bh⟩ :=
    claInitBranch_run s sP k v h0 USE_INITIAL_STATE s_qk_h s_qk_t s_qk_d s_vo_h
      s_vo_t s_vo_d T K V BT BK BV hPpids hPik hPiv hPibh hPbh
      (fun off => hPmem h0 off)
  rw [stepStmts.cons_some hinit]
  obtain ⟨sL, hloop, hLik, hLiv, hLibh, hLbh, hLoth, hLhist⟩ :=
    claFwdLoop_run s s1 k v h h0 USE_INITIAL_STATE s_qk_h s_qk_t s_qk_d s_vo_h
      s_vo_t s_vo_d s_h_h s_h_t T K V BT BK BV NT hHk hHv hσ hFit
      h1pids h1ik h1iv h1ibh h1bh
      (fun rg off => by rw [h1mem rg off, hPmem rg off])
  rw [stepStmts.cons_some hloop]
  obtain ⟨sF, hfin, hFoth, hFht⟩ :=
    claFinalBranch_run s sL k v h0 ht USE_INITIAL_STATE STORE_FINAL_STATE
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d T K V BT BK BV NT hBVV
      hLik hLiv hLibh hLbh
  rw [stepStmts.cons_some hfin, stepStmts.nil]
  refine ⟨sF, rfl, ?_, hFht⟩
  intro t idx htNT hidx
  rw [hFoth h _ (Ne.symm hHtH)]
  exact hLhist t idx htNT hidx

/-! ## The backward walk

Same skeleton with the descending change of variable: iteration `c` handles
chunk `NT - 1 - c`, stores the carry (`claDhCarry c` = every chunk after it),
then accumulates that chunk's `(scale·q)ᵀ·do`. -/

/-- The carried backward state tile after `c` descending iterations. -/
noncomputable def claDhTile (s : BlockState) (q do_ : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (T K V BT BK BV NT : Nat) (c : Nat) : Tile .real [BK, BV] :=
  ⟨fun idx => some (claDhCarry s q do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
    scale T K V BT BK BV NT c idx.1.val idx.2.1.val)⟩

/-- The scaled query tile `b_q * scale` of chunk `t`: cell `(e, c)`. -/
noncomputable def claBqTile (s : BlockState) (q : RegionName)
    (s_qk_h s_qk_t s_qk_d : Nat) (scale : ℝ) (T K BT BK : Nat) (t : Nat) :
    Tile .real [BK, BT] :=
  ⟨fun idx => some (kGuarded s q s_qk_h s_qk_t s_qk_d T K BT BK t
    idx.2.1.val idx.1.val * scale)⟩

/-- The `b_q * scale` statement lands on `claBqTile`. -/
private theorem cla_scale_eval (s sin : BlockState) (q : RegionName)
    (s_qk_h s_qk_t s_qk_d : Nat) (scale : ℝ) (T K BT BK : Nat) (t : Nat)
    (hbq : sin.regs .real [BK, BT] "b_q"
      = some (claBkTile s q s_qk_h s_qk_t s_qk_d T K BT BK t)) :
    evalOp (Op.mul .real Broadcast.scalarR (Op.ref .real [BK, BT] "b_q")
        (Op.const scale)) sin
      = some (claBqTile s q s_qk_h s_qk_t s_qk_d scale T K BT BK t) := by
  rw [evalOp_mul]
  simp only [evalOp_ref, hbq, evalOp_const, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some (Tile.ext fun idx => ?_)
  obtain ⟨e, c, u⟩ := idx
  simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, claBkTile,
    claBqTile]
  rfl

/-- The recurrence statement: `b_dh + tl.dot(b_q, b_do)` advances the carry by
one (descending) chunk. -/
private theorem cla_bwd_advance_eval (s sin : BlockState) (q do_ : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (T K V BT BK BV NT : Nat) (c : Nat) (hc : c < NT)
    (hbdh : sin.regs .real [BK, BV] "b_dh" = some (claDhTile s q do_
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale T K V BT BK BV NT c))
    (hbq : sin.regs .real [BK, BT] "b_q" = some (claBqTile s q
      s_qk_h s_qk_t s_qk_d scale T K BT BK (NT - 1 - c)))
    (hbdo : sin.regs .real [BT, BV] "b_do" = some (claBvTile s do_
      s_vo_h s_vo_t s_vo_d T V BT BV (NT - 1 - c))) :
    evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BK, BV] "b_dh")
        (Op.dot (batch := []) (Op.ref .real [BK, BT] "b_q")
          (Op.ref .real [BT, BV] "b_do"))) sin
      = some (claDhTile s q do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
          T K V BT BK BV NT (c + 1)) := by
  rw [cla_addTile_eval (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
    (Op.ref .real [BK, BV] "b_dh")
    (Op.dot (batch := []) (Op.ref .real [BK, BT] "b_q")
      (Op.ref .real [BT, BV] "b_do")) sin
    (claDhTile s q do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
      T K V BT BK BV NT c)
    (Tile.dot [] (claBqTile s q s_qk_h s_qk_t s_qk_d scale T K BT BK (NT - 1 - c))
      (claBvTile s do_ s_vo_h s_vo_t s_vo_d T V BT BV (NT - 1 - c)))
    (by rw [evalOp_ref]; exact hbdh)
    (cla_dot_eval _ _ sin _ _ (by rw [evalOp_ref]; exact hbq)
      (by rw [evalOp_ref]; exact hbdo))]
  refine congrArg some (Tile.ext fun idx => ?_)
  obtain ⟨e, p, u⟩ := idx
  simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, claDhTile,
    NumericDType.add, WithBot.realAdd]
  rw [cla_dot2d_elem _ _ e p
    (fun cc => kGuarded s q s_qk_h s_qk_t s_qk_d T K BT BK (NT - 1 - c)
      cc.val e.val * scale)
    (fun cc => vGuarded s do_ s_vo_h s_vo_t s_vo_d T V BT BV (NT - 1 - c)
      cc.val p.val)
    (fun cc => rfl) (fun cc => rfl)]
  rw [show ∀ x y : ℝ, Option.map₂ (· + ·) (some x : WithBot ℝ) (some y) = some (x + y)
    from fun _ _ => rfl]
  refine congrArg some ?_
  rw [claDhCarry_succ s q do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
    T K V BT BK BV NT c e.val p.val hc]
  rfl

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **One backward chunk** (ascending counter `c`, descending chunk `NT-1-c`).
Stores the carry at chunk `NT-1-c`'s block of `dh` — which is exactly the
spec's `claDhState` there — then advances by that chunk's contribution. -/
theorem claBwdBody_step (s sin : BlockState) (q do_ dh : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat) (scale : ℝ)
    (T K V BT BK BV NT : Nat) (c : Nat) (hc : c < NT)
    (hDq : dh ≠ q) (hDdo : dh ≠ do_) (hσ : BV ≤ s_h_t)
    (hmemQ : ∀ off, sin.readMem q off = s.readMem q off)
    (hmemDo : ∀ off, sin.readMem do_ off = s.readMem do_ off)
    (hik : sin.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0)))
    (hiv : sin.regs .nat [] "i_v" = some (Tile.scalar (s.pids 1)))
    (hibh : sin.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2)))
    (hj : sin.regs .nat [] "j" = some (Tile.scalar c))
    (hbdh : sin.regs .real [BK, BV] "b_dh" = some (claDhTile s q do_
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale T K V BT BK BV NT c)) :
    ∃ s', stepStmts (claBwdBody q do_ dh s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t
        s_vo_d s_h_h s_h_t scale T K V BT BK BV NT) sin = some s'
      ∧ s'.pids = sin.pids
      ∧ s'.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0))
      ∧ s'.regs .nat [] "i_v" = some (Tile.scalar (s.pids 1))
      ∧ s'.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2))
      ∧ s'.regs .real [BK, BV] "b_dh" = some (claDhTile s q do_
          s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale T K V BT BK BV NT (c + 1))
      ∧ (∀ rg off, rg ≠ dh → s'.readMem rg off = sin.readMem rg off)
      ∧ (∀ idx : TileIndex [BK, BV], claActive s K V BK BV idx →
          s'.readMem dh (claHOffset s s_h_h s_h_t K V BK BV (NT - 1 - c) idx)
            = claDhState s q do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
                T K V BT BK BV NT (NT - 1 - c) idx.1.val idx.2.1.val)
      ∧ (∀ off, (∀ idx : TileIndex [BK, BV], claActive s K V BK BV idx →
            off ≠ claHOffset s s_h_h s_h_t K V BK BV (NT - 1 - c) idx) →
          s'.readMem dh off = sin.readMem dh off) := by
  unfold claBwdBody
  -- i_t = NT - 1 - j
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (cla_itIdx_eval sin NT c hj))]
  -- p_q
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cla_makeBlockPtr_2d_eval q _ _ _ _ [K, T] [BK, BT] [s_qk_d, s_qk_t]
      (s.pids 2 * s_qk_h) (s.pids 0 * BK) ((NT - 1 - c) * BT)
      (cla_mulConst_eval _ "i_bh" (s.pids 2) s_qk_h (by simpa using hibh))
      (cla_mulConst_eval _ "i_k" (s.pids 0) BK (by simpa using hik))
      (cla_mulConst_eval _ "i_t" (NT - 1 - c) BT (by simp))))]
  -- p_do
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cla_makeBlockPtr_2d_eval do_ _ _ _ _ [T, V] [BT, BV] [s_vo_t, s_vo_d]
      (s.pids 2 * s_vo_h) ((NT - 1 - c) * BT) (s.pids 1 * BV)
      (cla_mulConst_eval _ "i_bh" (s.pids 2) s_vo_h (by simpa using hibh))
      (cla_mulConst_eval _ "i_t" (NT - 1 - c) BT (by simp))
      (cla_mulConst_eval _ "i_v" (s.pids 1) BV (by simpa using hiv))))]
  -- p_dh
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cla_makeBlockPtr_2d_eval dh _ _ _ _ [K, V] [BK, BV] [s_h_t, 1]
      (s.pids 2 * s_h_h + (NT - 1 - c) * K * V) (s.pids 0 * BK) (s.pids 1 * BV)
      (cla_hBase_eval _ s_h_h K V (s.pids 2) (NT - 1 - c) (by simpa using hibh)
        (by simp))
      (cla_mulConst_eval _ "i_k" (s.pids 0) BK (by simpa using hik))
      (cla_mulConst_eval _ "i_v" (s.pids 1) BV (by simpa using hiv))))]
  -- the carry store into dh's chunk block
  rw [stepStmts.cons_some (claStore_step_eq s _ dh "b_dh" "p_dh"
    (s.pids 2 * s_h_h + (NT - 1 - c) * K * V) s_h_t K V BK BV
    (fun e p => claDhCarry s q do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
      scale T K V BT BK BV NT c e p)
    (claDhTile s q do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
      T K V BT BK BV NT c)
    (fun e p => rfl)
    (by simpa using hbdh)
    (by simp))]
  obtain ⟨hSpids, hSregs, hSact, hSoth, hSoff⟩ :=
    claStore_step_props s
      ((((sin.setReg "i_t" .nat [] (Tile.scalar (NT - 1 - c))).setReg
          "p_q" .blockPtr [BK, BT]
          (⟨fun _ => BlockPtr.mk q (s.pids 2 * s_qk_h) [K, T] [BK, BT]
            [s_qk_d, s_qk_t] [s.pids 0 * BK, (NT - 1 - c) * BT]⟩ :
              Tile .blockPtr [BK, BT])).setReg
          "p_do" .blockPtr [BT, BV]
          (⟨fun _ => BlockPtr.mk do_ (s.pids 2 * s_vo_h) [T, V] [BT, BV]
            [s_vo_t, s_vo_d] [(NT - 1 - c) * BT, s.pids 1 * BV]⟩ :
              Tile .blockPtr [BT, BV])).setReg
          "p_dh" .blockPtr [BK, BV]
          (⟨fun _ => BlockPtr.mk dh (s.pids 2 * s_h_h + (NT - 1 - c) * K * V)
            [K, V] [BK, BV] [s_h_t, 1] [s.pids 0 * BK, s.pids 1 * BV]⟩ :
              Tile .blockPtr [BK, BV]))
      dh (s.pids 2 * s_h_h + (NT - 1 - c) * K * V) s_h_t K V BK BV
      (fun e p => claDhCarry s q do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
        scale T K V BT BK BV NT c e p)
      (claStoreAddr_injective s (s.pids 2 * s_h_h + (NT - 1 - c) * K * V)
        s_h_t BK BV hσ)
  set sSt := claStoreState s
      ((((sin.setReg "i_t" .nat [] (Tile.scalar (NT - 1 - c))).setReg
          "p_q" .blockPtr [BK, BT]
          (⟨fun _ => BlockPtr.mk q (s.pids 2 * s_qk_h) [K, T] [BK, BT]
            [s_qk_d, s_qk_t] [s.pids 0 * BK, (NT - 1 - c) * BT]⟩ :
              Tile .blockPtr [BK, BT])).setReg
          "p_do" .blockPtr [BT, BV]
          (⟨fun _ => BlockPtr.mk do_ (s.pids 2 * s_vo_h) [T, V] [BT, BV]
            [s_vo_t, s_vo_d] [(NT - 1 - c) * BT, s.pids 1 * BV]⟩ :
              Tile .blockPtr [BT, BV])).setReg
          "p_dh" .blockPtr [BK, BV]
          (⟨fun _ => BlockPtr.mk dh (s.pids 2 * s_h_h + (NT - 1 - c) * K * V)
            [K, V] [BK, BV] [s_h_t, 1] [s.pids 0 * BK, s.pids 1 * BV]⟩ :
              Tile .blockPtr [BK, BV]))
      dh (s.pids 2 * s_h_h + (NT - 1 - c) * K * V) s_h_t K V BK BV
      (fun e p => claDhCarry s q do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
        scale T K V BT BK BV NT c e p) with hsSt
  -- b_q load
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cla_kLoad_eq s sSt q "p_q" s_qk_h s_qk_t s_qk_d T K BT BK (NT - 1 - c)
      (fun off => by
        rw [hSoth q off (Ne.symm hDq)]
        simpa using hmemQ off)
      (by rw [hSregs]; simp)))]
  -- b_q *= scale
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cla_scale_eval s _ q s_qk_h s_qk_t s_qk_d scale T K BT BK (NT - 1 - c)
      (by simp)))]
  -- b_do load
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cla_vLoad_eq s _ do_ "p_do" s_vo_h s_vo_t s_vo_d T V BT BV (NT - 1 - c)
      (fun off => by
        rw [BlockState.setReg_readMem, BlockState.setReg_readMem,
          hSoth do_ off (Ne.symm hDdo)]
        simpa using hmemDo off)
      (by
        rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
          BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hSregs]
        simp)))]
  -- b_dh advance
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cla_bwd_advance_eval s _ q do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
      scale T K V BT BK BV NT c hc
      (by
        rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
          BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
          BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hSregs]
        simpa using hbdh)
      (by
        rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
        simp)
      (by simp)))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · -- pids
    simp only [BlockState.setReg_pids, hSpids]
  · -- i_k
    rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hSregs]
    simpa using hik
  · -- i_v
    rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hSregs]
    simpa using hiv
  · -- i_bh
    rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hSregs]
    simpa using hibh
  · -- b_dh
    rw [BlockState.setReg_same]
  · -- other regions unchanged
    intro rg off hrg
    rw [BlockState.setReg_readMem, BlockState.setReg_readMem,
      BlockState.setReg_readMem, BlockState.setReg_readMem, hSoth rg off hrg]
    simp
  · -- chunk-(NT-1-c) active readback = claDhState there
    intro idx hidx
    rw [BlockState.setReg_readMem, BlockState.setReg_readMem,
      BlockState.setReg_readMem, BlockState.setReg_readMem,
      claHOffset_eq_storeAddr, hSact idx hidx,
      claDhState_eq_carry s q do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
        scale T K V BT BK BV NT c idx.1.val idx.2.1.val hc]
  · -- dh off the active block unchanged
    intro off hoff
    rw [BlockState.setReg_readMem, BlockState.setReg_readMem,
      BlockState.setReg_readMem, BlockState.setReg_readMem,
      hSoff off (fun idx hidx => by
        rw [← claHOffset_eq_storeAddr]
        exact hoff idx hidx)]
    simp

set_option maxHeartbeats 4000000 in
/-- **The collapsed backward loop.** After all `NT` descending iterations, every
chunk block of `dh` holds the strictly-later-chunks sum `claDhState`. -/
theorem claBwdLoop_run (s sPre : BlockState) (q do_ dh : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat) (scale : ℝ)
    (T K V BT BK BV NT : Nat)
    (hDq : dh ≠ q) (hDdo : dh ≠ do_) (hσ : BV ≤ s_h_t)
    (hFit : (K - 1) * s_h_t + V ≤ K * V)
    (hpids : sPre.pids = s.pids)
    (hik : sPre.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0)))
    (hiv : sPre.regs .nat [] "i_v" = some (Tile.scalar (s.pids 1)))
    (hibh : sPre.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2)))
    (hbdh : sPre.regs .real [BK, BV] "b_dh" = some (claDhTile s q do_
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale T K V BT BK BV NT 0))
    (hmem : ∀ rg off, sPre.readMem rg off = s.readMem rg off) :
    ∃ sL, stepStmt (Stmt.forRange "j" 0 NT 1
        (claBwdBody q do_ dh s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h
          s_h_t scale T K V BT BK BV NT)) sPre = some sL
      ∧ (∀ t (idx : TileIndex [BK, BV]), t < NT → claActive s K V BK BV idx →
          sL.readMem dh (claHOffset s s_h_h s_h_t K V BK BV t idx)
            = claDhState s q do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
                T K V BT BK BV NT t idx.1.val idx.2.1.val) := by
  have hres := forRange_inv (idx := "j") (start := 0) (stop := NT) (step := 1)
    (body := claBwdBody q do_ dh s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h
      s_h_t scale T K V BT BK BV NT)
    (P := fun c tS =>
      c ≤ NT
      ∧ tS.pids = s.pids
      ∧ tS.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0))
      ∧ tS.regs .nat [] "i_v" = some (Tile.scalar (s.pids 1))
      ∧ tS.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2))
      ∧ tS.regs .real [BK, BV] "b_dh" = some (claDhTile s q do_
          s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale T K V BT BK BV NT c)
      ∧ (∀ rg off, rg ≠ dh → tS.readMem rg off = s.readMem rg off)
      ∧ (∀ j' (idx : TileIndex [BK, BV]), j' < c → claActive s K V BK BV idx →
          tS.readMem dh (claHOffset s s_h_h s_h_t K V BK BV (NT - 1 - j') idx)
            = claDhState s q do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
                T K V BT BK BV NT (NT - 1 - j') idx.1.val idx.2.1.val))
    (s_init := sPre) (by decide)
    ⟨Nat.zero_le _, hpids, hik, hiv, hibh, hbdh,
      fun rg off _ => hmem rg off,
      fun j' idx hj' _ => absurd hj' (Nat.not_lt_zero j')⟩
    (by
      intro i tS hi hP
      obtain ⟨hle, hPpids, hPik, hPiv, hPibh, hPbdh, hPoth, hPhist⟩ := hP
      obtain ⟨s', hstep, hspids, hsik, hsiv, hsibh, hsbdh, hsoth, hsact, hsoff⟩ :=
        claBwdBody_step s (tS.setReg "j" .nat [] (Tile.scalar i)) q do_ dh
          s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t scale
          T K V BT BK BV NT i hi hDq hDdo hσ
          (fun off => by
            rw [BlockState.setReg_readMem]
            exact hPoth q off (Ne.symm hDq))
          (fun off => by
            rw [BlockState.setReg_readMem]
            exact hPoth do_ off (Ne.symm hDdo))
          (by simpa using hPik)
          (by simpa using hPiv)
          (by simpa using hPibh)
          (by simp)
          (by simpa using hPbdh)
      refine ⟨s', hstep, by omega,
        by rw [hspids, BlockState.setReg_pids]; exact hPpids,
        hsik, hsiv, hsibh, hsbdh, ?_, ?_⟩
      · intro rg off hrg
        rw [hsoth rg off hrg, BlockState.setReg_readMem]
        exact hPoth rg off hrg
      · intro j' idx hj' hidx
        rcases Nat.lt_or_ge j' i with hji | hji
        · -- an older iteration's chunk: preserved across this iteration's store
          rw [hsoff (claHOffset s s_h_h s_h_t K V BK BV (NT - 1 - j') idx)
            (fun idx' hidx' => claHOffset_chunk_disjoint s s_h_h s_h_t K V BK BV
              hFit (NT - 1 - j') (NT - 1 - i) (by omega) idx idx' hidx hidx'),
            BlockState.setReg_readMem]
          exact hPhist j' idx hji hidx
        · have hji' : j' = i := by omega
          subst hji'
          exact hsact idx hidx)
  obtain ⟨final, sL, hstep, hfin, hle, _, _, _, _, _, _, hLhist⟩ := hres
  have hfinal : final = NT := Nat.le_antisymm hle hfin
  rw [hfinal] at hLhist
  refine ⟨sL, hstep, ?_⟩
  intro t idx htNT hidx
  have h := hLhist (NT - 1 - t) idx (by omega) hidx
  rwa [show NT - 1 - (NT - 1 - t) = t from by omega] at h

set_option maxHeartbeats 1000000 in
/-- **Genuine, dimension-general correctness** of `chunk_linear_attn_bwd_kernel_dh`
(the descending loop, verified through its ascending change of variable). For
every launch state the kernel runs to completion and every active lane of every
chunk block `dh[·,·,t]` (`t < NT`) holds `claDhState t`: the sum of
`(scale·q_u)ᵀ·do_u` over the strictly-later chunks `u > t` — the state *before*
chunk `t`'s own contribution, exactly as the descending store-then-accumulate
order produces. -/
specification cla_bwd_dh_exec_genuine
    (q do_ dh : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat)
    (scale : ℝ) (T K V BT BK BV NT : Nat) (s : BlockState)
    (hDq : dh ≠ q) (hDdo : dh ≠ do_)
    (hσ : BV ≤ s_h_t) (hFit : (K - 1) * s_h_t + V ≤ K * V) :
    ∃ sF, exec (cla_bwd_dh_surface q do_ dh s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t
        s_vo_d s_h_h s_h_t scale T K V BT BK BV NT).toAlgKernel s = some sF
      ∧ ∀ t (idx : TileIndex [BK, BV]), t < NT → claActive s K V BK BV idx →
          sF.readMem dh (claHOffset s s_h_h s_h_t K V BK BV t idx)
            = claDhState s q do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
                T K V BT BK BV NT t idx.1.val idx.2.1.val := by
  obtain ⟨sP, hpre, hPpids, hPik, hPiv, hPibh, hPbdh, hPmem⟩ :=
    claPreLoop_run s BK BV "b_dh" (by decide) (by decide) (by decide)
  rw [exec, cla_bwd_body_eq, hpre]
  obtain ⟨sL, hloop, hLhist⟩ :=
    claBwdLoop_run s sP q do_ dh s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
      s_h_h s_h_t scale T K V BT BK BV NT hDq hDdo hσ hFit
      hPpids hPik hPiv hPibh
      (by
        rw [hPbdh]
        refine congrArg some (Tile.ext fun idx => ?_)
        simp [claDhTile, claDhCarry_zero])
      hPmem
  rw [stepStmts.cons_some hloop, stepStmts.nil]
  exact ⟨sL, rfl, hLhist⟩

end Correct_without_Rounding

end VeriTile.Bench.TritonBenchG.ChunkLinearAttn

