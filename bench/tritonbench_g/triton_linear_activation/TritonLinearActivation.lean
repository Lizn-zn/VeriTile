import VeriTile.Triton

/-!
# `triton_linear_activation` — closed-form fused linear+activation correctness

`triton_linear_activation.py`'s `kernel_fma` is the xformers-style fused linear
layer `Out = activation(A × Wᵀ + bias)`: a single linear `program_idx` is split
into a `(block_m_idx, block_n_idx)` tile coordinate by the L2-grouping
schedule, the `%M` / `%N`-wrapped row/column index vectors avoid load masking,
a `BLOCK_M × BLOCK_N` accumulator tile is seeded with the (optional) masked
bias row, accumulated over K by `acc += tl.dot(a, b)`, optionally spilled to
`ACT_INPUTS` (the pre-activation values saved for backward), passed through the
constexpr-selected activation (`tanh` / `gelu` / `fast_gelu` / `relu` /
identity), and stored to `C` under the `(m_offs < M) & (n_offs < N)` mask.

This file proves the **full K-loop + bias + activation + both stores** correct
against a genuine closed form: every output cell `C[i,j]` of the computed tile
equals `act( bias[n_offs j] + Σ_{k < K} A[m_offs i, k] · B[k, n_offs j] )` over
`ℝ`, and (when `SHOULD_SAVE_ACT_INPUTS`) every `ACT_INPUTS[i,j]` cell equals
the un-activated `bias + Σ_k A·B`. These are NOT the kernel's own emitted
values — the GEMM reference is an independent `Finset.sum` (`gemmSum`) over
the loaded `A`/`B` cells and the activation is the corresponding real function.

## Proof architecture

```
triton_linear_activation_output_summary_general   ← TOP THEOREM (bundled, general)
  ├─ kernel_fma_surface_toAlgorithm_supported     surface lowers to the algorithm layer
  ├─ kernel_fma_C_compute_correct                 C output (Realizes_without_Rounding)
  │    └─ kernel_fma_exec_closed_form             exec-side closed forms (C + ACT_INPUTS)
  │         ├─ preLoop        (schedule scalars + pointers + acc = bias seed)
  │         │    ├─ preLoop_scalars   (L2 schedule + %-wrapped index vectors)
  │         │    └─ biasIf_steps      (HAS_BIAS constexpr gate: masked bias row)
  │         ├─ kernel_fma_step        (one K-block: acc += tl.dot(a, b))
  │         ├─ forRange_inv           (loop-invariant principle, drives the K-loop)
  │         └─ kernel_fma_postLoop    (save-store + activation gates + masked C store)
  │              ├─ saveIf_steps      (SHOULD_SAVE_ACT_INPUTS gate)
  │              └─ actTail_steps     (the four ACTIVATION == "…" gates)
  └─ kernel_fma_act_inputs_compute_correct        ACT_INPUTS output (Realizes_without_Rounding)
```

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float): `tl.float32` accumulation
and the implicit fp16 output round-trip are the identity at the algorithm
layer; `tl.extra.cuda.libdevice.erf` / `tanh` are the exact real
`VeriTile.Math.realErf` / `Real.tanh`. `@triton.autotune`, `@triton.heuristics`,
`num_warps`/`num_stages`, and the config choice are NOT modeled: the surface is
one concrete config, the trusted boundary. The host launch (grid, linear-pid
scheduling) is trusted; the per-program statement is universally quantified
over `s`, covering every program of the grid. The L2-grouping index math
(`program_idx → (block_m_idx, block_n_idx)`) is transcribed exactly as the
kernel computes it, and the spec's layout references the same derived
`m_offs`/`n_offs`, so it is not a separate proof obligation. The kernel's
store masks `(m_offs < M) & (n_offs < N)` test the **already-`%`-wrapped**
offsets, so an overhanging tile wraps around and overwrites in-range rows
rather than masking them off; the headline is therefore honestly scoped to
programs whose tile fits (`hFitM`/`hFitN`, always true when
`BLOCK_M ∣ M ∧ BLOCK_N ∣ N`, e.g. every benchmark launch), which also
discharges store-footprint injectivity.

## Translation-surface blocker

Translation-surface blocker: the `K_LOAD_MASK_NEEDED` constexpr branch is
specialized to its `@triton.heuristics` value `True` (`K % (BLOCK_K·SPLIT_K)
== 0`, i.e. exact-multiple `K`; `SPLIT_K = 1` is the only launched value), so
the kernel's unmasked-load arm is the transcribed surface and the descending
Python loop `for k in range(K, 0, -BLOCK_K)` is presented as the ascending
trip-count loop `range(0, numKBlocks, 1)` with `K = BLOCK_K · numKBlocks`
supplied as the antiquoted `numKBlocks` binder (the loop variable is dead in
the unmasked arm). The `@triton.jit` helper functions `tanh` / `relu` /
`gelu` / `fast_gelu` are inlined at their call sites (`tl.extra.cuda.libdevice.tanh`
is spelled through the DSL's `tanh(...)`, the helpers' call-site spelling); the
module-level constants `sqrt2` / `sqrt2pi` are the antiquoted reals
`Real.sqrt 2` / `Real.sqrt (2.0 / Real.pi)`. The unused `CACHE_KEY_M/N/K` and
`SPLIT_K` parameters are dropped. The Python string constexpr `ACTIVATION` is
the Lean `String` parameter `ACTIVATION` (the four `ACTIVATION == "…"` gates
are transcribed verbatim). The textual py↔lean scans in
`bench/audit_tritonbench_g.sh` exempt this port on this marker (registered in
`proof_blockers.md`).
-/

namespace VeriTile.Bench.TritonBenchG.TritonLinearActivation

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Faithful transcription of `triton_linear_activation.py`'s `kernel_fma`
(the `K_LOAD_MASK_NEEDED = True` heuristics arm; see the module docstring's
translation-surface blocker for the presented loop bound and inlined helpers). -/
noncomputable def kernel_fma_surface
    (C ACT_INPUTS A B bias : RegionName)
    (M N output_m_stride output_n_stride act_inputs_m_stride act_inputs_n_stride
      a_m_stride a_k_stride b_n_stride b_k_stride
      BLOCK_M GROUP_M BLOCK_N BLOCK_K numKBlocks : Nat)
    (HAS_BIAS SHOULD_SAVE_ACT_INPUTS : Bool) (ACTIVATION : String) :
    ComputeKernel := triton {
  program_idx = tl.program_id(axis=0)
  grid_m = ($((M : Nat)) + $(BLOCK_M) - $(1)) // $(BLOCK_M)
  grid_n = ($((N : Nat)) + $(BLOCK_N) - $(1)) // $(BLOCK_N)
  width = $(GROUP_M) * grid_n
  group_idx = program_idx // width
  group_size = min(grid_m - group_idx * $(GROUP_M), $(GROUP_M))
  block_m_idx = group_idx * $(GROUP_M) + (program_idx % group_size)
  block_n_idx = (program_idx % width) // group_size
  m_offs_untagged = block_m_idx * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  n_offs_untagged = block_n_idx * $(BLOCK_N) + tl.arange(0, $(BLOCK_N))
  m_offs = tl.max_contiguous(tl.multiple_of(m_offs_untagged % $(M), $(BLOCK_M)), $(BLOCK_M))
  n_offs = tl.max_contiguous(tl.multiple_of(n_offs_untagged % $(N), $(BLOCK_N)), $(BLOCK_N))
  k_range_offs = tl.arange(0, $(BLOCK_K))
  A = A + (m_offs[:, None] * $(a_m_stride) + k_range_offs[None, :] * $(a_k_stride))
  B = B + (k_range_offs[:, None] * $(b_k_stride) + n_offs[None, :] * $(b_n_stride))
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
  if HAS_BIAS {
    bias = tl.load(bias + n_offs, mask=n_offs < $(N), other=0.0).to(tl.float32)
    acc += bias[None, :]
  }
  for k in range($(0), $(numKBlocks), $(1)) {
    a = tl.load(A)
    b = tl.load(B)
    acc += tl.dot(a, b)
    A += $(BLOCK_K) * $(a_k_stride)
    B += $(BLOCK_K) * $(b_k_stride)
  }
  if SHOULD_SAVE_ACT_INPUTS {
    act_in_ptrs = ACT_INPUTS + m_offs[:, None] * $(act_inputs_m_stride) + n_offs[None, :] * $(act_inputs_n_stride)
    tl.store(act_in_ptrs, acc)
  }
  if ACTIVATION == "tanh" {
    acc = tanh(acc)
  }
  if ACTIVATION == "gelu" {
    acc = acc * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(acc / $((Real.sqrt 2 : ℝ))))
  }
  if ACTIVATION == "fast_gelu" {
    acc = 0.5 * acc * (1 + tanh($((Real.sqrt (2.0 / Real.pi) : ℝ)) * (acc + 0.044715 * acc * acc * acc)))
  }
  if ACTIVATION == "relu" {
    acc = tl.maximum(0, acc)
  }
  C = C + m_offs[:, None] * $(output_m_stride) + n_offs[None, :] * $(output_n_stride)
  c_ptr_mask = (m_offs < $(M))[:, None] & (n_offs < $(N))[None, :]
  tl.store(C, acc, mask=c_ptr_mask)
}

/-- The full `kernel_fma` surface lowers to the algorithm layer. -/
theorem kernel_fma_surface_toAlgorithm_supported
    (C ACT_INPUTS A B bias : RegionName)
    (M N smo sno saim sain sam sak sbn sbk BM GM BN BK numKBlocks : Nat)
    (HAS_BIAS SHOULD_SAVE_ACT_INPUTS : Bool) (ACTIVATION : String) :
    ∃ alg, (kernel_fma_surface C ACT_INPUTS A B bias M N smo sno saim sain
      sam sak sbn sbk BM GM BN BK numKBlocks
      HAS_BIAS SHOULD_SAVE_ACT_INPUTS ACTIVATION).toAlgorithm? = Except.ok alg := by
  simp [kernel_fma_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

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

@[simp] theorem evalOp_expandDim_zero_real {D : Nat} (name : RegName) (s : BlockState) :
    @evalOp .real [1, D] (Op.expandDim ⟨0, by simp⟩ (Op.ref .real [D] name)) s =
      (s.regs .real [D] name).bind (fun v =>
        some ({ data := fun i : TileIndex [1, D] => v.data (i.2.1, PUnit.unit) } : Tile .real [1, D])) := by
  unfold evalOp; simp [Tile.expandDim]; rfl

/-- No-mask `.ptr` load: reads `readMem` at each pointer (clean `undef`). -/
theorem load_ptr_none_real {shape : TileShape}
    (ptrOp : Op .ptr shape) (s : BlockState) (ptrs : Tile .ptr shape)
    (hp : evalOp ptrOp s = some ptrs) :
    evalOp (.load .real (.ptr ptrOp) .none) s
      = some ⟨fun i => some (s.readMem (ptrs.data i).1 (ptrs.data i).2)⟩ := by
  simp only [evalOp, hp]
  refine congrArg some ?_
  ext i
  simp [BlockState.readMemValue_real]

/-- Row×column pointer chunk `R + gx[:, None]·sx + gy[None, :]·sy` (the kernel's
`A`-seed / `act_in_ptrs` / `C` shape): cell `(i,j) = (R, gx i · sx + gy j · sy)`. -/
theorem pairptrs_eval (s : BlockState) (R : RegionName) (X Y sx sy : Nat)
    (nx ny : RegName) (gx : Fin X → Nat) (gy : Fin Y → Nat)
    (hx : s.regs .nat [X] nx = some (Tile.vec gx))
    (hy : s.regs .nat [Y] ny = some (Tile.vec gy)) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase R)
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [X] nx)) (Op.constNat sx))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [Y] ny)) (Op.constNat sy)))) s
      = some (⟨fun idx : TileIndex [X, Y] => (R.cast, gx idx.1 * sx + gy idx.2.1 * sy)⟩ : Tile .ptr [X, Y]) := by
  rw [evalOp_ptrAdd, evalOp_ptrBase]
  simp only [evalOp_add, evalOp_mul, evalOp_constNat, evalOp_ref, evalOp_expandDim_one_nat,
    evalOp_expandDim_zero_nat, hx, hy, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec]
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec, NumericDType.add, NumericDType.mul]

/-- `acc` init eval: `tl.zeros` → the all-`0` tile. -/
theorem acc_init_eval (s : BlockState) (M N : Nat) :
    evalOp (Op.full [M, N] (Op.const 0)) s
      = some (⟨fun _ : TileIndex [M, N] => some (0 : ℝ)⟩ : Tile .real [M, N]) := by
  simp [evalOp_full, evalOp_const, Option.bind]

/-- Pointer advance `P += BK · stride` eval. -/
theorem ptr_adv_eval (s : BlockState) (X Y BK st : Nat) (nm : RegName) (pt : Tile .ptr [X, Y])
    (hp : s.regs .ptr [X, Y] nm = some pt) :
    evalOp (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [X, Y] nm)
      (Op.mul .nat Broadcast.nil (Op.constNat BK) (Op.constNat st))) s
      = some (Tile.ptrAdd Broadcast.scalarR pt (Tile.scalar (BK * st))) := by
  rw [evalOp_ptrAdd]
  simp [evalOp_ref, hp, evalOp_mul, evalOp_constNat, NumericDType.mul, Tile.bop]

/-- **`acc += tl.dot(a, b)` statement eval** (`acc + dot a b`). -/
theorem accdot_op_eval (M K N : Nat) (st : BlockState)
    (zt : Tile .real [M, N]) (xt : Tile .real [M, K]) (yt : Tile .real [K, N])
    (hz : st.regs .real [M, N] "acc" = some zt)
    (hx : st.regs .real [M, K] "a" = some xt)
    (hy : st.regs .real [K, N] "b" = some yt) :
    evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [M, N] "acc")
        (Op.dot (batch := []) (Op.ref .real [M, K] "a") (Op.ref .real [K, N] "b"))) st
      = some (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
          zt (Tile.dot [] xt yt)) := by
  have hd : evalOp (Op.dot (batch := []) (Op.ref .real [M, K] "a")
        (Op.ref .real [K, N] "b")) st = some (Tile.dot [] xt yt) := by
    rw [evalOp_dot]; simp [hx, hy]
  rw [evalOp_add]
  simp only [evalOp_ref, hz, bind, Option.bind_some]
  erw [hd]
  rfl

/-- `acc + dot` lane `(i,j)`: `some (zv + dv)`. -/
theorem dotadd_eval (M N : Nat) (zt dt : Tile .real [M, N]) (i : Fin M) (j : Fin N) (zv dv : ℝ)
    (hz : zt.data (i, j, PUnit.unit) = some zv) (hd : dt.data (i, j, PUnit.unit) = some dv) :
    (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil)) zt dt).data
        (i, j, PUnit.unit) = some (zv + dv) := by
  simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, hd, hz, NumericDType.add,
    WithBot.realAdd, Option.map₂, Option.bind, Option.map]

/-- The masked bias-row load `tl.load(bias + n_offs, mask=n_offs < N, other=0.0)`:
lane `j` reads `bias[n_offs j]` when `n_offs j < N` and pads `0` otherwise. -/
theorem bias_load_eval (st : BlockState) (biasR : RegionName) (N BN : Nat)
    (gn : Fin BN → Nat)
    (hn : st.regs .nat [BN] "n_offs" = some (Tile.vec gn)) :
    evalOp (Op.load .real (MemAccess.region biasR (Op.ref .nat [BN] "n_offs"))
        (MaskOpt.maskOther
          (Op.lt ComparableDType.nat Broadcast.scalarR (Op.ref .nat [BN] "n_offs") (Op.constNat N))
          ((Op.const 0.0).broadcast [BN]))) st
      = some ⟨fun j : TileIndex [BN] =>
          some (if gn j.1 < N then st.readMem biasR (gn j.1) else 0)⟩ := by
  have hmask : evalOp (Op.lt ComparableDType.nat Broadcast.scalarR
      (Op.ref .nat [BN] "n_offs") (Op.constNat N)) st
      = some (Tile.cop ComparableDType.nat.lt Broadcast.scalarR (Tile.vec gn) (Tile.scalar N)) := by
    rw [evalOp_lt]
    simp [hn]
  have hother : evalOp ((Op.const (0.0 : ℝ)).broadcast [BN]) st
      = some (⟨fun _ => some (0.0 : ℝ)⟩ : Tile .real [BN]) := by
    simp only [evalOp, evalOp_const, Option.bind_some]
    rfl
  simp only [evalOp, evalOp_ref, hn, hmask, hother, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext j
  by_cases h : gn j.1 < N
  · simp only [Tile.cop_data, Tile.vec_data, Tile.scalar_data, Broadcast.leftIndex,
      Broadcast.rightIndex, ComparableDType.lt, h, decide_true, if_true,
      BlockState.readMemValue_real, Region.cast_id]
  · simp only [Tile.cop_data, Tile.vec_data, Tile.scalar_data, Broadcast.leftIndex,
      Broadcast.rightIndex, ComparableDType.lt, h, decide_false, Bool.false_eq_true, if_false]
    norm_num

/-! ## Body decomposition (prefix ++ for-loop ++ save/activation/store tail) -/

/-- The `HAS_BIAS` constexpr gate (statement 16): masked bias-row load +
`acc += bias[None, :]`. -/
def biasIfStmt (biasR : RegionName) (N BM BN : Nat) (HAS_BIAS : Bool) : Stmt :=
  Stmt.ifThen (Op.constBool HAS_BIAS)
    [ Stmt.assign .real [BN] "bias"
        (Op.load .real (MemAccess.region biasR (Op.ref .nat [BN] "n_offs"))
          (MaskOpt.maskOther
            (Op.lt ComparableDType.nat Broadcast.scalarR (Op.ref .nat [BN] "n_offs") (Op.constNat N))
            ((Op.const 0.0).broadcast [BN]))),
      Stmt.assign .real [BM, BN] "acc"
        (Op.add .real (Broadcast.consR (Broadcast.consSame Broadcast.nil))
          (Op.ref .real [BM, BN] "acc")
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .real [BN] "bias"))) ]

/-- The 5-statement K-loop body: two unmasked loads, `acc += tl.dot(a, b)`,
two pointer advances. -/
def fmaLoopBody (BM BN BK sak sbk : Nat) : List Stmt :=
  [ Stmt.assign .real [BM, BK] "a"
      (Op.load .real (.ptr (Op.ref .ptr [BM, BK] "A")) .none),
    Stmt.assign .real [BK, BN] "b"
      (Op.load .real (.ptr (Op.ref .ptr [BK, BN] "B")) .none),
    Stmt.assign .real [BM, BN] "acc"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BM, BN] "acc")
        (Op.dot (batch := []) (Op.ref .real [BM, BK] "a") (Op.ref .real [BK, BN] "b"))),
    Stmt.assign .ptr [BM, BK] "A"
      (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BM, BK] "A")
        (Op.mul .nat Broadcast.nil (Op.constNat BK) (Op.constNat sak))),
    Stmt.assign .ptr [BK, BN] "B"
      (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BK, BN] "B")
        (Op.mul .nat Broadcast.nil (Op.constNat BK) (Op.constNat sbk))) ]

/-- The `SHOULD_SAVE_ACT_INPUTS` constexpr gate (statement 18): pointer chunk
+ unmasked store of the pre-activation accumulator. -/
def saveIfStmt (ACT_INPUTS : RegionName) (saim sain BM BN : Nat)
    (SHOULD_SAVE_ACT_INPUTS : Bool) : Stmt :=
  Stmt.ifThen (Op.constBool SHOULD_SAVE_ACT_INPUTS)
    [ Stmt.assign .ptr [BM, BN] "act_in_ptrs"
        (Op.ptrAdd Broadcast.scalarL (Op.ptrBase ACT_INPUTS)
          (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
            (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "m_offs")) (Op.constNat saim))
            (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "n_offs")) (Op.constNat sain)))),
      Stmt.store .real [BM, BN] (.ptr (Op.ref .ptr [BM, BN] "act_in_ptrs"))
        (Op.ref .real [BM, BN] "acc") .none ]

/-- The four sequential `ACTIVATION == "…"` constexpr gates (statements 19–22),
with the `@triton.jit` helpers inlined. -/
noncomputable def actStmts (BM BN : Nat) (ACTIVATION : String) : List Stmt :=
  [ Stmt.ifThen (Op.constBool (ACTIVATION == "tanh"))
      [ Stmt.assign .real [BM, BN] "acc" (Op.tanh (Op.ref .real [BM, BN] "acc")) ],
    Stmt.ifThen (Op.constBool (ACTIVATION == "gelu"))
      [ Stmt.assign .real [BM, BN] "acc"
          (Op.mul .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
            (Op.mul .real Broadcast.scalarR (Op.ref .real [BM, BN] "acc") (Op.const 0.5))
            (Op.add .real Broadcast.scalarL (Op.const 1.0)
              (Op.erf (Op.div .real Broadcast.scalarR (Op.ref .real [BM, BN] "acc")
                (Op.const (Real.sqrt 2)))))) ],
    Stmt.ifThen (Op.constBool (ACTIVATION == "fast_gelu"))
      [ Stmt.assign .real [BM, BN] "acc"
          (Op.mul .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
            (Op.mul .real Broadcast.scalarL (Op.const 0.5) (Op.ref .real [BM, BN] "acc"))
            (Op.add .real Broadcast.scalarL (Op.const 1)
              (Op.tanh (Op.mul .real Broadcast.scalarL (Op.const (Real.sqrt (2.0 / Real.pi)))
                (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
                  (Op.ref .real [BM, BN] "acc")
                  (Op.mul .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
                    (Op.mul .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
                      (Op.mul .real Broadcast.scalarL (Op.const 0.044715) (Op.ref .real [BM, BN] "acc"))
                      (Op.ref .real [BM, BN] "acc"))
                    (Op.ref .real [BM, BN] "acc"))))))) ],
    Stmt.ifThen (Op.constBool (ACTIVATION == "relu"))
      [ Stmt.assign .real [BM, BN] "acc"
          ((Op.gt ComparableDType.real Broadcast.scalarL (Op.const 0) (Op.ref .real [BM, BN] "acc")).where
            ((Op.const 0).broadcast [BM, BN]) (Op.ref .real [BM, BN] "acc")) ] ]

/-- The 3-statement store tail (statements 23–25): the `C` pointer chunk, the
`(m_offs < M) & (n_offs < N)` mask, and the masked store. -/
def storeTail (C : RegionName) (M N smo sno BM BN : Nat) : List Stmt :=
  [ Stmt.assign .ptr [BM, BN] "C"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase C)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "m_offs")) (Op.constNat smo))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "n_offs")) (Op.constNat sno)))),
    Stmt.assign .bool [BM, BN] "c_ptr_mask"
      (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.expandDim ⟨1, by simp⟩
          (Op.lt ComparableDType.nat Broadcast.scalarR (Op.ref .nat [BM] "m_offs") (Op.constNat M)))
        (Op.expandDim ⟨0, by simp⟩
          (Op.lt ComparableDType.nat Broadcast.scalarR (Op.ref .nat [BN] "n_offs") (Op.constNat N)))),
    Stmt.store .real [BM, BN] (.ptr (Op.ref .ptr [BM, BN] "C"))
      (Op.ref .real [BM, BN] "acc") (.mask (Op.ref .bool [BM, BN] "c_ptr_mask")) ]

/-- Body decomposition: prefix (17) ++ [for-loop] ++ save-gate ++ activation
gates (4) ++ store tail (3). By `rfl`. -/
theorem fma_body_split (C ACT_INPUTS A B bias : RegionName)
    (M N smo sno saim sain sam sak sbn sbk BM GM BN BK numKBlocks : Nat)
    (HAS_BIAS SHOULD_SAVE_ACT_INPUTS : Bool) (ACTIVATION : String) :
    (kernel_fma_surface C ACT_INPUTS A B bias M N smo sno saim sain
        sam sak sbn sbk BM GM BN BK numKBlocks
        HAS_BIAS SHOULD_SAVE_ACT_INPUTS ACTIVATION).toAlgKernel.body
      = (kernel_fma_surface C ACT_INPUTS A B bias M N smo sno saim sain
          sam sak sbn sbk BM GM BN BK numKBlocks
          HAS_BIAS SHOULD_SAVE_ACT_INPUTS ACTIVATION).toAlgKernel.body.take 17
        ++ (Stmt.forRange "k" 0 numKBlocks 1 (fmaLoopBody BM BN BK sak sbk)
            :: saveIfStmt ACT_INPUTS saim sain BM BN SHOULD_SAVE_ACT_INPUTS
            :: (actStmts BM BN ACTIVATION ++ storeTail C M N smo sno BM BN)) := by
  rfl

/-- `acc += bias[None, :]` eval: lane `(i,j)` becomes `z (i,j) + w j`. -/
theorem acc_bias_eval (st : BlockState) (BM BN : Nat)
    (z : TileIndex [BM, BN] → ℝ) (w : Fin BN → ℝ)
    (hz : st.regs .real [BM, BN] "acc" = some ⟨fun idx => some (z idx)⟩)
    (hw : st.regs .real [BN] "bias" = some ⟨fun j => some (w j.1)⟩) :
    evalOp (Op.add .real (Broadcast.consR (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BM, BN] "acc")
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .real [BN] "bias"))) st
      = some ⟨fun idx : TileIndex [BM, BN] => some (z idx + w idx.2.1)⟩ := by
  rw [evalOp_add]
  simp only [evalOp_ref, hz, evalOp_expandDim_zero_real, hw, Option.bind_eq_bind,
    Option.bind_some]
  refine congrArg some ?_
  ext idx
  simp [Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, WithBot.realAdd]

/-! ## Schedule + closed-form spec

The spec mirrors the kernel's own L2-grouping schedule and `%`-wrapped index
derivation exactly (so it is not a separate proof obligation), then states the
genuine `bias + Σ_k A·B` linear form and the real activation. -/

/-- `min` as the kernel's `tl.where(a < b, a, b)` spells it. -/
def kernelMin (a b : Nat) : Nat := if a < b then a else b

/-- The kernel's `block_m_idx` derivation from the linear `program_idx`. -/
def blockMIdx (pid M N BM BN GM : Nat) : Nat :=
  let grid_m := (M + BM - 1) / BM
  let grid_n := (N + BN - 1) / BN
  let width := GM * grid_n
  let group_idx := pid / width
  let group_size := kernelMin (grid_m - group_idx * GM) GM
  group_idx * GM + pid % group_size

/-- The kernel's `block_n_idx` derivation from the linear `program_idx`. -/
def blockNIdx (pid M N BM BN GM : Nat) : Nat :=
  let grid_m := (M + BM - 1) / BM
  let grid_n := (N + BN - 1) / BN
  let width := GM * grid_n
  let group_idx := pid / width
  let group_size := kernelMin (grid_m - group_idx * GM) GM
  (pid % width) / group_size

/-- Global output row of tile lane `i`: `block_m_idx · BLOCK_M + i`, **before**
the `% M` wrap (the kernel's `m_offs_untagged`). -/
def rowGlobal (s : BlockState) (M N BM BN GM : Nat) (i : Fin BM) : Nat :=
  blockMIdx (s.pids 0) M N BM BN GM * BM + i.val

/-- Global output column of tile lane `j`, before the `% N` wrap. -/
def colGlobal (s : BlockState) (M N BM BN GM : Nat) (j : Fin BN) : Nat :=
  blockNIdx (s.pids 0) M N BM BN GM * BN + j.val

/-- The `% M`-wrapped row index of tile lane `i` (the kernel's `m_offs`). -/
def rowIndex (s : BlockState) (M N BM BN GM : Nat) (i : Fin BM) : Nat :=
  rowGlobal s M N BM BN GM i % M

/-- The `% N`-wrapped column index of tile lane `j` (the kernel's `n_offs`). -/
def colIndex (s : BlockState) (M N BM BN GM : Nat) (j : Fin BN) : Nat :=
  colGlobal s M N BM BN GM j % N

/-- `A[i, k] = readMem A (m_offs i · a_m_stride + k · a_k_stride)`. -/
noncomputable def aElem (s : BlockState) (A : RegionName) (M N BM BN GM sam sak : Nat)
    (i : Fin BM) (k : Nat) : ℝ :=
  s.readMem A (rowIndex s M N BM BN GM i * sam + k * sak)

/-- `B[k, j] = readMem B (k · b_k_stride + n_offs j · b_n_stride)`. -/
noncomputable def bElem (s : BlockState) (B : RegionName) (M N BM BN GM sbk sbn : Nat)
    (j : Fin BN) (k : Nat) : ℝ :=
  s.readMem B (k * sbk + colIndex s M N BM BN GM j * sbn)

/-- The bias contribution of column lane `j`: the masked bias-row load when
`HAS_BIAS` (`bias[n_offs j]` if `n_offs j < N`, else the load's `other=0.0`),
`0` otherwise. -/
noncomputable def biasBase (s : BlockState) (biasR : RegionName) (M N BM BN GM : Nat)
    (HAS_BIAS : Bool) (j : Fin BN) : ℝ :=
  if HAS_BIAS then
    (if colIndex s M N BM BN GM j < N then s.readMem biasR (colIndex s M N BM BN GM j) else 0)
  else 0

/-- **Genuine pre-activation linear spec** (over ℝ):
`bias[j] + Σ_{k < BLOCK_K·numKBlocks} A[i,k] · B[k,j]`. -/
noncomputable def linearSpec (s : BlockState) (A B biasR : RegionName)
    (M N BM BN GM sam sak sbk sbn BLOCK_K numKBlocks : Nat) (HAS_BIAS : Bool)
    (i : Fin BM) (j : Fin BN) : ℝ :=
  biasBase s biasR M N BM BN GM HAS_BIAS j
    + gemmSum (aElem s A M N BM BN GM sam sak i) (bElem s B M N BM BN GM sbk sbn j)
        (BLOCK_K * numKBlocks)

/-- Partial accumulator after `c` K-blocks: `bias[j] + Σ_{k < c·BLOCK_K} A·B`. -/
noncomputable def accPartial (s : BlockState) (A B biasR : RegionName)
    (M N BM BN GM sam sak sbk sbn BLOCK_K : Nat) (HAS_BIAS : Bool)
    (i : Fin BM) (j : Fin BN) (c : Nat) : ℝ :=
  biasBase s biasR M N BM BN GM HAS_BIAS j
    + gemmSum (aElem s A M N BM BN GM sam sak i) (bElem s B M N BM BN GM sbk sbn j)
        (c * BLOCK_K)

/-- One-block step of the partial accumulator (the shared `gemmSum_blockSucc`). -/
theorem accPartial_succ (s : BlockState) (A B biasR : RegionName)
    (M N BM BN GM sam sak sbk sbn BLOCK_K : Nat) (HAS_BIAS : Bool)
    (i : Fin BM) (j : Fin BN) (c : Nat) :
    accPartial s A B biasR M N BM BN GM sam sak sbk sbn BLOCK_K HAS_BIAS i j (c + 1)
      = accPartial s A B biasR M N BM BN GM sam sak sbk sbn BLOCK_K HAS_BIAS i j c
        + (Finset.univ.sum fun e : Fin BLOCK_K =>
            aElem s A M N BM BN GM sam sak i (c * BLOCK_K + e.val)
              * bElem s B M N BM BN GM sbk sbn j (c * BLOCK_K + e.val)) := by
  unfold accPartial
  rw [gemmSum_blockSucc (aElem s A M N BM BN GM sam sak i)
    (bElem s B M N BM BN GM sbk sbn j) BLOCK_K c]
  ring

/-- Real GELU as the kernel's inlined `gelu` helper spells it:
`x · 0.5 · (1 + erf(x / √2))` with the exact real error function. -/
noncomputable def geluRef (x : ℝ) : ℝ :=
  x * 0.5 * (1.0 + VeriTile.Math.realErf (x / Real.sqrt 2))

/-- Real fast-GELU as the kernel's inlined `fast_gelu` helper spells it:
`0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))`. -/
noncomputable def fastGeluRef (x : ℝ) : ℝ :=
  0.5 * x * (1 + Real.tanh (Real.sqrt (2.0 / Real.pi) * (x + 0.044715 * x * x * x)))

/-- The activation tail exactly as the kernel applies it: the four sequential
`ACTIVATION == "…"` constexpr gates (mutually exclusive for any one string;
every other string — including the benchmark's `""` — is the identity). -/
noncomputable def applyActivation (ACTIVATION : String) (x : ℝ) : ℝ :=
  let x1 := if ACTIVATION == "tanh" then Real.tanh x else x
  let x2 := if ACTIVATION == "gelu" then geluRef x1 else x1
  let x3 := if ACTIVATION == "fast_gelu" then fastGeluRef x2 else x2
  if ACTIVATION == "relu" then max 0 x3 else x3

/-- The `C` store address of tile lane `(i,j)`:
`m_offs i · output_m_stride + n_offs j · output_n_stride` (the kernel stores
through the **`%`-wrapped** `m_offs`/`n_offs`). -/
def cOffset (s : BlockState) (M N BM BN GM smo sno : Nat) (idx : TileIndex [BM, BN]) : Nat :=
  rowIndex s M N BM BN GM idx.1 * smo + colIndex s M N BM BN GM idx.2.1 * sno

/-- The `ACT_INPUTS` store address of tile lane `(i,j)`. -/
def actOffset (s : BlockState) (M N BM BN GM saim sain : Nat) (idx : TileIndex [BM, BN]) : Nat :=
  rowIndex s M N BM BN GM idx.1 * saim + colIndex s M N BM BN GM idx.2.1 * sain

/-! ## Loop invariant -/

@[simp] theorem evalOp_constBool (b : Bool) (s : BlockState) :
    evalOp (.constBool b) s = some (Tile.scalar b) := by simp [evalOp]

/-- `stepStmt` on a constexpr `ifThen` gate: run the body when the flag is
true, skip otherwise. -/
theorem stepStmt_ifThen_constBool (b : Bool) (body : List Stmt) (s : BlockState) :
    stepStmt (Stmt.ifThen (Op.constBool b) body) s
      = if b then stepStmts body s else some s := by
  simp only [stepStmt, evalOp_constBool, Option.bind_eq_bind, Option.bind_some, Tile.scalar_data]
  rfl

/-- **Loop invariant** (counter `c` = K-block index, step `1`).

After `c` K-blocks: the `m_offs`/`n_offs` index vectors seeded (from the L2
schedule, `%`-wrapped); `acc` equals `bias + Σ_{k < c·BLOCK_K} A·B`; the
`A`/`B` pointers advanced by `c` blocks; memory untouched. -/
noncomputable def fmaInvariant (A B biasR : RegionName) (s0 : BlockState)
    (M N BM BN GM sam sak sbk sbn BLOCK_K numKBlocks : Nat) (HAS_BIAS : Bool)
    (c : Nat) (s : BlockState) : Prop :=
  c ≤ numKBlocks ∧
  (s.regs .nat [BM] "m_offs" = some (Tile.vec (fun i : Fin BM => rowIndex s0 M N BM BN GM i))) ∧
  (s.regs .nat [BN] "n_offs" = some (Tile.vec (fun j : Fin BN => colIndex s0 M N BM BN GM j))) ∧
  (s.regs .real [BM, BN] "acc" = some ⟨fun idx : TileIndex [BM, BN] =>
      some (accPartial s0 A B biasR M N BM BN GM sam sak sbk sbn BLOCK_K HAS_BIAS idx.1 idx.2.1 c)⟩) ∧
  (s.regs .ptr [BM, BLOCK_K] "A" = some ⟨fun idx : TileIndex [BM, BLOCK_K] =>
      (A.cast, rowIndex s0 M N BM BN GM idx.1 * sam + idx.2.1.val * sak + c * BLOCK_K * sak)⟩) ∧
  (s.regs .ptr [BLOCK_K, BN] "B" = some ⟨fun idx : TileIndex [BLOCK_K, BN] =>
      (B.cast, idx.1.val * sbk + colIndex s0 M N BM BN GM idx.2.1 * sbn + c * BLOCK_K * sbk)⟩) ∧
  (s.mem = s0.mem)

/-- **preLoop scalars** (statements 0–12): the 8 L2-schedule scalars and the
index vectors `m_offs_untagged`/`n_offs_untagged`/`m_offs`/`n_offs`/
`k_range_offs`. -/
theorem preLoop_scalars (s : BlockState) (M N BM BN BK GM : Nat) :
    ∃ s13, stepStmts
      [ Stmt.assign .nat [] "program_idx" (Op.programId 0),
        Stmt.assign .nat [] "grid_m"
          (Op.floorDiv IntegralDType.nat Broadcast.nil
            (Op.sub .nat Broadcast.nil
              (Op.add .nat Broadcast.nil (Op.constNat M) (Op.constNat BM)) (Op.constNat 1))
            (Op.constNat BM)),
        Stmt.assign .nat [] "grid_n"
          (Op.floorDiv IntegralDType.nat Broadcast.nil
            (Op.sub .nat Broadcast.nil
              (Op.add .nat Broadcast.nil (Op.constNat N) (Op.constNat BN)) (Op.constNat 1))
            (Op.constNat BN)),
        Stmt.assign .nat [] "width"
          (Op.mul .nat Broadcast.nil (Op.constNat GM) (Op.ref .nat [] "grid_n")),
        Stmt.assign .nat [] "group_idx"
          (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.ref .nat [] "program_idx")
            (Op.ref .nat [] "width")),
        Stmt.assign .nat [] "group_size"
          ((Op.lt ComparableDType.nat Broadcast.nil
              (Op.sub .nat Broadcast.nil (Op.ref .nat [] "grid_m")
                (Op.mul .nat Broadcast.nil (Op.ref .nat [] "group_idx") (Op.constNat GM)))
              (Op.constNat GM)).where
            (Op.sub .nat Broadcast.nil (Op.ref .nat [] "grid_m")
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "group_idx") (Op.constNat GM)))
            (Op.constNat GM)),
        Stmt.assign .nat [] "block_m_idx"
          (Op.add .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "group_idx") (Op.constNat GM))
            (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] "program_idx")
              (Op.ref .nat [] "group_size"))),
        Stmt.assign .nat [] "block_n_idx"
          (Op.floorDiv IntegralDType.nat Broadcast.nil
            (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] "program_idx")
              (Op.ref .nat [] "width"))
            (Op.ref .nat [] "group_size")),
        Stmt.assign .nat [BM] "m_offs_untagged"
          (Op.add .nat Broadcast.scalarL
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "block_m_idx") (Op.constNat BM))
            (Op.arange BM)),
        Stmt.assign .nat [BN] "n_offs_untagged"
          (Op.add .nat Broadcast.scalarL
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "block_n_idx") (Op.constNat BN))
            (Op.arange BN)),
        Stmt.assign .nat [BM] "m_offs"
          (Op.mod IntegralDType.nat Broadcast.scalarR (Op.ref .nat [BM] "m_offs_untagged")
            (Op.constNat M)),
        Stmt.assign .nat [BN] "n_offs"
          (Op.mod IntegralDType.nat Broadcast.scalarR (Op.ref .nat [BN] "n_offs_untagged")
            (Op.constNat N)),
        Stmt.assign .nat [BK] "k_range_offs" (Op.arange BK) ] s = some s13
      ∧ s13.pids = s.pids
      ∧ s13.regs .nat [BM] "m_offs" = some (Tile.vec (fun i : Fin BM => rowIndex s M N BM BN GM i))
      ∧ s13.regs .nat [BN] "n_offs" = some (Tile.vec (fun j : Fin BN => colIndex s M N BM BN GM j))
      ∧ s13.regs .nat [BK] "k_range_offs" = some (Tile.vec (fun e : Fin BK => e.val))
      ∧ s13.mem = s.mem := by
  simp only [rowIndex, colIndex, rowGlobal, colGlobal, blockMIdx, blockNIdx, kernelMin]
  simp [stepStmts, stepStmt, evalOp_floorDiv, evalOp_mod, Option.bind, BlockState.setReg,
    Tile.bop, Tile.vec, NumericDType.add, NumericDType.mul, NumericDType.sub,
    IntegralDType.floorDiv, IntegralDType.mod, Tile.select, Tile.cop, ComparableDType.lt]

/-- **`HAS_BIAS` gate stepping**: from a state whose `n_offs` holds the wrapped
column indices and `acc` the all-zero tile, the gate produces
`acc = biasBase` (the masked bias row when `HAS_BIAS`, still `0` otherwise);
memory, pids, and all registers except `acc`/`bias` are untouched. -/
theorem biasIf_steps (st : BlockState) (biasR : RegionName) (s0 : BlockState)
    (M N BM BN GM : Nat) (HAS_BIAS : Bool)
    (hn : st.regs .nat [BN] "n_offs" = some (Tile.vec (fun j : Fin BN => colIndex s0 M N BM BN GM j)))
    (hacc : st.regs .real [BM, BN] "acc"
      = some ⟨fun _ : TileIndex [BM, BN] => some (0 : ℝ)⟩)
    (hrm : ∀ o : Nat, st.readMem biasR o = s0.readMem biasR o) :
    ∃ st', stepStmt (biasIfStmt biasR N BM BN HAS_BIAS) st = some st'
      ∧ st'.regs .real [BM, BN] "acc" = some ⟨fun idx : TileIndex [BM, BN] =>
          some (biasBase s0 biasR M N BM BN GM HAS_BIAS idx.2.1)⟩
      ∧ (∀ {dt} {sh} (nm : RegName), nm ≠ "acc" → nm ≠ "bias" →
          st'.regs dt sh nm = st.regs dt sh nm)
      ∧ st'.mem = st.mem ∧ st'.pids = st.pids := by
  unfold biasIfStmt
  rw [stepStmt_ifThen_constBool]
  cases HAS_BIAS
  · -- HAS_BIAS = false
    refine ⟨st, rfl, ?_, fun _ _ _ => rfl, rfl, rfl⟩
    rw [hacc]
    refine congrArg some ?_
    ext idx
    simp [biasBase]
  · -- HAS_BIAS = true
    rw [if_pos rfl]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (bias_load_eval st biasR N BN (fun j => colIndex s0 M N BM BN GM j) hn))]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (acc_bias_eval _ BM BN (fun _ => 0)
        (fun j => if colIndex s0 M N BM BN GM j < N
          then st.readMem biasR (colIndex s0 M N BM BN GM j) else 0)
        (by simp [hacc]) (by simp)))]
    rw [stepStmts.nil]
    refine ⟨_, rfl, ?_, ?_, ?_, ?_⟩
    · simp only [BlockState.setReg_same]
      refine congrArg some ?_
      ext idx
      simp [biasBase, hrm]
    · intro dt sh nm h1 h2
      simp [BlockState.setReg_ne_name, h1, h2]
    · rfl
    · rfl

set_option maxHeartbeats 1000000 in
/-- **preLoop** (statements 0–16): the schedule scalars, index vectors,
pointer seeds, `acc = 0`, and the `HAS_BIAS` gate step to a state satisfying
`fmaInvariant … 0`. -/
theorem preLoop (C ACT_INPUTS A B biasR : RegionName) (s : BlockState)
    (M N smo sno saim sain sam sak sbn sbk BM GM BN BK numKBlocks : Nat)
    (HAS_BIAS SHOULD_SAVE_ACT_INPUTS : Bool) (ACTIVATION : String) :
    ∃ s', stepStmts ((kernel_fma_surface C ACT_INPUTS A B biasR M N smo sno saim sain
        sam sak sbn sbk BM GM BN BK numKBlocks
        HAS_BIAS SHOULD_SAVE_ACT_INPUTS ACTIVATION).toAlgKernel.body.take 17) s = some s'
      ∧ fmaInvariant A B biasR s M N BM BN GM sam sak sbk sbn BK numKBlocks HAS_BIAS 0 s' := by
  obtain ⟨s13, h13, hpids, hm, hn, hk, hmem⟩ := preLoop_scalars s M N BM BN BK GM
  set sA : BlockState := s13.setReg "A" .ptr [BM, BK]
    (⟨fun idx : TileIndex [BM, BK] =>
      (A.cast, rowIndex s M N BM BN GM idx.1 * sam + idx.2.1.val * sak)⟩ : Tile .ptr [BM, BK])
    with hsA
  set sB : BlockState := sA.setReg "B" .ptr [BK, BN]
    (⟨fun idx : TileIndex [BK, BN] =>
      (B.cast, idx.1.val * sbk + colIndex s M N BM BN GM idx.2.1 * sbn)⟩ : Tile .ptr [BK, BN])
    with hsB
  set sAcc : BlockState := sB.setReg "acc" .real [BM, BN]
    (⟨fun _ : TileIndex [BM, BN] => some (0 : ℝ)⟩ : Tile .real [BM, BN]) with hsAcc
  have hstepA : stepStmt (Stmt.assign .ptr [BM, BK] "A"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase A)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "m_offs")) (Op.constNat sam))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BK] "k_range_offs")) (Op.constNat sak))))) s13
      = some sA :=
    stepStmt_assign_eq_some (pairptrs_eval s13 A BM BK sam sak "m_offs" "k_range_offs"
      (fun i => rowIndex s M N BM BN GM i) (fun e : Fin BK => e.val) hm hk)
  have hstepB : stepStmt (Stmt.assign .ptr [BK, BN] "B"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase B)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BK] "k_range_offs")) (Op.constNat sbk))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "n_offs")) (Op.constNat sbn))))) sA
      = some sB :=
    stepStmt_assign_eq_some (pairptrs_eval sA B BK BN sbk sbn "k_range_offs" "n_offs"
      (fun e : Fin BK => e.val) (fun j => colIndex s M N BM BN GM j)
      (by simp [hsA, hk]) (by simp [hsA, hn]))
  have hstepAcc : stepStmt (Stmt.assign .real [BM, BN] "acc" (Op.full [BM, BN] (Op.const 0))) sB
      = some sAcc :=
    stepStmt_assign_eq_some (acc_init_eval sB BM BN)
  obtain ⟨s17, hbias, hacc17, hpres17, hmem17, hpids17⟩ :=
    biasIf_steps sAcc biasR s M N BM BN GM HAS_BIAS
      (by simp [hsAcc, hsB, hsA, hn]) (by simp [hsAcc])
      (by
        intro o
        simp only [hsAcc, hsB, hsA, BlockState.setReg_readMem]
        unfold BlockState.readMem
        rw [hmem])
  rw [show ((kernel_fma_surface C ACT_INPUTS A B biasR M N smo sno saim sain
        sam sak sbn sbk BM GM BN BK numKBlocks
        HAS_BIAS SHOULD_SAVE_ACT_INPUTS ACTIVATION).toAlgKernel.body.take 17)
      = [ Stmt.assign .nat [] "program_idx" (Op.programId 0),
          Stmt.assign .nat [] "grid_m"
            (Op.floorDiv IntegralDType.nat Broadcast.nil
              (Op.sub .nat Broadcast.nil
                (Op.add .nat Broadcast.nil (Op.constNat M) (Op.constNat BM)) (Op.constNat 1))
              (Op.constNat BM)),
          Stmt.assign .nat [] "grid_n"
            (Op.floorDiv IntegralDType.nat Broadcast.nil
              (Op.sub .nat Broadcast.nil
                (Op.add .nat Broadcast.nil (Op.constNat N) (Op.constNat BN)) (Op.constNat 1))
              (Op.constNat BN)),
          Stmt.assign .nat [] "width"
            (Op.mul .nat Broadcast.nil (Op.constNat GM) (Op.ref .nat [] "grid_n")),
          Stmt.assign .nat [] "group_idx"
            (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.ref .nat [] "program_idx")
              (Op.ref .nat [] "width")),
          Stmt.assign .nat [] "group_size"
            ((Op.lt ComparableDType.nat Broadcast.nil
                (Op.sub .nat Broadcast.nil (Op.ref .nat [] "grid_m")
                  (Op.mul .nat Broadcast.nil (Op.ref .nat [] "group_idx") (Op.constNat GM)))
                (Op.constNat GM)).where
              (Op.sub .nat Broadcast.nil (Op.ref .nat [] "grid_m")
                (Op.mul .nat Broadcast.nil (Op.ref .nat [] "group_idx") (Op.constNat GM)))
              (Op.constNat GM)),
          Stmt.assign .nat [] "block_m_idx"
            (Op.add .nat Broadcast.nil
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "group_idx") (Op.constNat GM))
              (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] "program_idx")
                (Op.ref .nat [] "group_size"))),
          Stmt.assign .nat [] "block_n_idx"
            (Op.floorDiv IntegralDType.nat Broadcast.nil
              (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] "program_idx")
                (Op.ref .nat [] "width"))
              (Op.ref .nat [] "group_size")),
          Stmt.assign .nat [BM] "m_offs_untagged"
            (Op.add .nat Broadcast.scalarL
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "block_m_idx") (Op.constNat BM))
              (Op.arange BM)),
          Stmt.assign .nat [BN] "n_offs_untagged"
            (Op.add .nat Broadcast.scalarL
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "block_n_idx") (Op.constNat BN))
              (Op.arange BN)),
          Stmt.assign .nat [BM] "m_offs"
            (Op.mod IntegralDType.nat Broadcast.scalarR (Op.ref .nat [BM] "m_offs_untagged")
              (Op.constNat M)),
          Stmt.assign .nat [BN] "n_offs"
            (Op.mod IntegralDType.nat Broadcast.scalarR (Op.ref .nat [BN] "n_offs_untagged")
              (Op.constNat N)),
          Stmt.assign .nat [BK] "k_range_offs" (Op.arange BK) ]
      ++ [ Stmt.assign .ptr [BM, BK] "A"
            (Op.ptrAdd Broadcast.scalarL (Op.ptrBase A)
              (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
                (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "m_offs")) (Op.constNat sam))
                (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BK] "k_range_offs")) (Op.constNat sak)))),
          Stmt.assign .ptr [BK, BN] "B"
            (Op.ptrAdd Broadcast.scalarL (Op.ptrBase B)
              (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
                (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BK] "k_range_offs")) (Op.constNat sbk))
                (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "n_offs")) (Op.constNat sbn)))),
          Stmt.assign .real [BM, BN] "acc" (Op.full [BM, BN] (Op.const 0)),
          biasIfStmt biasR N BM BN HAS_BIAS ] from rfl,
    stepStmts.append_some h13,
    stepStmts.cons_some hstepA,
    stepStmts.cons_some hstepB,
    stepStmts.cons_some hstepAcc,
    stepStmts.cons_some hbias, stepStmts.nil]
  refine ⟨_, rfl, by simp, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hpres17 "m_offs" (by decide) (by decide)]
    simp [hsAcc, hsB, hsA, hm]
  · rw [hpres17 "n_offs" (by decide) (by decide)]
    simp [hsAcc, hsB, hsA, hn]
  · rw [hacc17]
    refine congrArg some ?_
    ext idx
    simp only [accPartial, Nat.zero_mul, gemmSum_zero, add_zero]
  · rw [hpres17 "A" (by decide) (by decide)]
    rw [show sAcc.regs .ptr [BM, BK] "A"
        = some (⟨fun idx : TileIndex [BM, BK] =>
            (A.cast, rowIndex s M N BM BN GM idx.1 * sam + idx.2.1.val * sak)⟩ : Tile .ptr [BM, BK])
      from by simp [hsAcc, hsB, hsA]]
    refine congrArg some ?_
    ext idx <;> simp [Nat.zero_mul]
  · rw [hpres17 "B" (by decide) (by decide)]
    rw [show sAcc.regs .ptr [BK, BN] "B"
        = some (⟨fun idx : TileIndex [BK, BN] =>
            (B.cast, idx.1.val * sbk + colIndex s M N BM BN GM idx.2.1 * sbn)⟩ : Tile .ptr [BK, BN])
      from by simp [hsAcc, hsB]]
    refine congrArg some ?_
    ext idx <;> simp [Nat.zero_mul]
  · rw [hmem17]
    simp only [hsAcc, hsB, hsA, BlockState.setReg_mem]
    exact hmem

set_option maxHeartbeats 2000000 in
/-- **Step lemma**: one K-loop body iteration advances the invariant by one
block (`acc += tl.dot(a, b)` adds the `c`-th block's dot to the partial
accumulator; the `A`/`B` pointers advance one step). -/
theorem kernel_fma_step (A B biasR : RegionName) (s0 : BlockState)
    (M N BM BN GM sam sak sbk sbn BLOCK_K numKBlocks : Nat) (HAS_BIAS : Bool)
    (c : Nat) (s : BlockState) (hclt : c < numKBlocks)
    (hinv : fmaInvariant A B biasR s0 M N BM BN GM sam sak sbk sbn BLOCK_K numKBlocks HAS_BIAS c s) :
    ∃ s', stepStmts (fmaLoopBody BM BN BLOCK_K sak sbk)
        (s.setReg "k" .nat [] (Tile.scalar c)) = some s'
      ∧ fmaInvariant A B biasR s0 M N BM BN GM sam sak sbk sbn BLOCK_K numKBlocks HAS_BIAS (c + 1) s' := by
  simp only [fmaInvariant] at hinv
  obtain ⟨hcle, hm, hn, hz, hap, hbp, hmem⟩ := hinv
  set apT : Tile .ptr [BM, BLOCK_K] :=
    ⟨fun idx : TileIndex [BM, BLOCK_K] =>
      (A.cast, rowIndex s0 M N BM BN GM idx.1 * sam + idx.2.1.val * sak + c * BLOCK_K * sak)⟩ with hapT
  set bpT : Tile .ptr [BLOCK_K, BN] :=
    ⟨fun idx : TileIndex [BLOCK_K, BN] =>
      (B.cast, idx.1.val * sbk + colIndex s0 M N BM BN GM idx.2.1 * sbn + c * BLOCK_K * sbk)⟩ with hbpT
  set zT : Tile .real [BM, BN] :=
    ⟨fun idx : TileIndex [BM, BN] =>
      some (accPartial s0 A B biasR M N BM BN GM sam sak sbk sbn BLOCK_K HAS_BIAS idx.1 idx.2.1 c)⟩ with hzT
  set sk := s.setReg "k" .nat [] (Tile.scalar c) with hsk
  have hrmem : ∀ (R : RegionName) (o : Nat), sk.readMem R o = s0.readMem R o := by
    intro R o; simp only [hsk, BlockState.setReg_readMem]; unfold BlockState.readMem; rw [hmem]
  have hapk : sk.regs .ptr [BM, BLOCK_K] "A" = some apT := by simp [hsk, hap, hapT]
  have hbpk : sk.regs .ptr [BLOCK_K, BN] "B" = some bpT := by simp [hsk, hbp, hbpT]
  have hzk : sk.regs .real [BM, BN] "acc" = some zT := by simp [hsk, hz, hzT]
  set asub : Tile .real [BM, BLOCK_K] :=
    ⟨fun idx => some (sk.readMem (apT.data idx).1 (apT.data idx).2)⟩ with hasub
  set bsub : Tile .real [BLOCK_K, BN] :=
    ⟨fun idx => some (sk.readMem (bpT.data idx).1 (bpT.data idx).2)⟩ with hbsub
  unfold fmaLoopBody
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (load_ptr_none_real (Op.ref .ptr [BM, BLOCK_K] "A") _ apT (by rw [evalOp_ref]; simp [hapk])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (load_ptr_none_real (Op.ref .ptr [BLOCK_K, BN] "B") _ bpT (by rw [evalOp_ref]; simp [hbpk])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (accdot_op_eval BM BLOCK_K BN _ zT asub bsub
          (by simp [hzk, hasub, hbsub, BlockState.setReg_readMem])
          (by simp [hasub, BlockState.setReg_readMem])
          (by simp [hbsub, BlockState.setReg_readMem])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (ptr_adv_eval _ BM BLOCK_K BLOCK_K sak "A" apT (by simp [hapk])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (ptr_adv_eval _ BLOCK_K BN BLOCK_K sbk "B" bpT (by simp [hbpk])))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  simp only [fmaInvariant]
  refine ⟨by omega, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp [hsk, hm]
  · simp [hsk, hn]
  · -- acc = accPartial (c+1)
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx
    have has : ∀ e : Fin BLOCK_K, asub.data (idx.1, e, PUnit.unit)
        = some (aElem s0 A M N BM BN GM sam sak idx.1 (c * BLOCK_K + e.val)) := by
      intro e
      simp only [hasub, hapT, hrmem, aElem, Region.cast_id]
      rw [show rowIndex s0 M N BM BN GM idx.1 * sam + e.val * sak + c * BLOCK_K * sak
            = rowIndex s0 M N BM BN GM idx.1 * sam + (c * BLOCK_K + e.val) * sak from by ring]
    have hbs : ∀ e : Fin BLOCK_K, bsub.data (e, idx.2.1, PUnit.unit)
        = some (bElem s0 B M N BM BN GM sbk sbn idx.2.1 (c * BLOCK_K + e.val)) := by
      intro e
      simp only [hbsub, hbpT, hrmem, bElem, Region.cast_id]
      rw [show e.val * sbk + colIndex s0 M N BM BN GM idx.2.1 * sbn + c * BLOCK_K * sbk
            = (c * BLOCK_K + e.val) * sbk + colIndex s0 M N BM BN GM idx.2.1 * sbn from by ring]
    rw [dotadd_eval BM BN zT (Tile.dot [] asub bsub) idx.1 idx.2.1
        (accPartial s0 A B biasR M N BM BN GM sam sak sbk sbn BLOCK_K HAS_BIAS idx.1 idx.2.1 c)
        (Finset.univ.sum fun e : Fin BLOCK_K =>
          aElem s0 A M N BM BN GM sam sak idx.1 (c * BLOCK_K + e.val)
            * bElem s0 B M N BM BN GM sbk sbn idx.2.1 (c * BLOCK_K + e.val))
        (by rw [hzT])
        (tile_dot_data BM BLOCK_K BN asub bsub idx.1 idx.2.1 _ _ has hbs)]
    show some _ = some (accPartial s0 A B biasR M N BM BN GM sam sak sbk sbn BLOCK_K HAS_BIAS idx.1 idx.2.1 (c + 1))
    rw [accPartial_succ]
  · -- A advanced
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx <;> simp only [Tile.ptrAdd, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
      Tile.scalar, hapT, NumericDType.add]
    ring
  · -- B advanced
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx <;> simp only [Tile.ptrAdd, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
      Tile.scalar, hbpT, NumericDType.add]
    ring
  · show _ = s0.mem
    rw [← hmem, hsk]; rfl

/-! ## Post-loop: save-store + activation gates + masked C store -/

theorem evalOp_erf {shape : TileShape} (x : Op .real shape) (s : BlockState) :
    evalOp (.erf x) s = (do
      let v ← evalOp x s
      some (Tile.uop WithBot.realErf v)) := by
  simp [evalOp]

theorem evalOp_boolAnd {a b shape} (bc : Broadcast a b shape)
    (x : Op .bool a) (y : Op .bool b) (s : BlockState) :
    evalOp (.boolAnd bc x y) s = (do
      let vx ← evalOp x s; let vy ← evalOp y s; some (Tile.bop (· && ·) bc vx vy)) := by
  simp [evalOp]

/-- A masked `writeMem` scatter never changes the program ids. -/
theorem foldl_writeMem_masked_pids {α : Type} (region : RegionName)
    (offsetFn : α → Nat) (valueFn : α → ℝ) (P : α → Prop) [DecidablePred P] :
    ∀ (l : List α) (t : BlockState),
      (l.foldl (fun acc k =>
        if P k then acc.writeMem region (offsetFn k) (valueFn k) else acc) t).pids = t.pids := by
  intro l
  induction l with
  | nil => intro t; rfl
  | cons hd tl ih =>
      intro t
      rw [List.foldl_cons]
      by_cases h : P hd
      · rw [if_pos h, ih]; rfl
      · rw [if_neg h, ih]

set_option maxHeartbeats 1000000 in
/-- **`SHOULD_SAVE_ACT_INPUTS` gate stepping**: when enabled, the pre-activation
accumulator is scattered (unmasked) to `ACT_INPUTS + m_offs·saim + n_offs·sain`;
registers other than `act_in_ptrs`, pids, and every other region's memory are
untouched. -/
theorem saveIf_steps (st : BlockState) (ACT_INPUTS : RegionName) (s0 : BlockState)
    (M N BM BN GM saim sain : Nat) (SHOULD_SAVE_ACT_INPUTS : Bool)
    (v : TileIndex [BM, BN] → ℝ)
    (hm : st.regs .nat [BM] "m_offs" = some (Tile.vec (fun i : Fin BM => rowIndex s0 M N BM BN GM i)))
    (hn : st.regs .nat [BN] "n_offs" = some (Tile.vec (fun j : Fin BN => colIndex s0 M N BM BN GM j)))
    (hacc : st.regs .real [BM, BN] "acc" = some ⟨fun idx => some (v idx)⟩) :
    ∃ st', stepStmt (saveIfStmt ACT_INPUTS saim sain BM BN SHOULD_SAVE_ACT_INPUTS) st = some st'
      ∧ (∀ {dt} {sh} (nm : RegName), nm ≠ "act_in_ptrs" → st'.regs dt sh nm = st.regs dt sh nm)
      ∧ st'.pids = st.pids
      ∧ (∀ (R : RegionName) (o : Nat), R ≠ ACT_INPUTS → st'.readMem R o = st.readMem R o)
      ∧ (SHOULD_SAVE_ACT_INPUTS = Bool.false → st'.mem = st.mem)
      ∧ (SHOULD_SAVE_ACT_INPUTS = Bool.true →
          Function.Injective (actOffset s0 M N BM BN GM saim sain) →
          ∀ idx : TileIndex [BM, BN],
            st'.readMem ACT_INPUTS (actOffset s0 M N BM BN GM saim sain idx) = v idx) := by
  unfold saveIfStmt
  rw [stepStmt_ifThen_constBool]
  cases SHOULD_SAVE_ACT_INPUTS
  · -- gate off
    exact ⟨st, rfl, fun _ _ => rfl, rfl, fun _ _ _ => rfl, fun _ => rfl, fun h => nomatch h⟩
  · -- gate on
    rw [if_pos rfl]
    set apT : Tile .ptr [BM, BN] :=
      ⟨fun idx : TileIndex [BM, BN] =>
        (ACT_INPUTS.cast, actOffset s0 M N BM BN GM saim sain idx)⟩ with hapT
    have hstepP : stepStmt (Stmt.assign .ptr [BM, BN] "act_in_ptrs"
        (Op.ptrAdd Broadcast.scalarL (Op.ptrBase ACT_INPUTS)
          (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
            (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "m_offs")) (Op.constNat saim))
            (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "n_offs")) (Op.constNat sain))))) st
        = some (st.setReg "act_in_ptrs" .ptr [BM, BN] apT) :=
      stepStmt_assign_eq_some (pairptrs_eval st ACT_INPUTS BM BN saim sain "m_offs" "n_offs"
        (fun i => rowIndex s0 M N BM BN GM i) (fun j => colIndex s0 M N BM BN GM j) hm hn)
    rw [stepStmts.cons_some hstepP]
    set stP := st.setReg "act_in_ptrs" .ptr [BM, BN] apT with hstP
    set accT : Tile .real [BM, BN] := ⟨fun idx => some (v idx)⟩ with haccT
    have hstore : stepStmt (Stmt.store .real [BM, BN]
          (.ptr (Op.ref .ptr [BM, BN] "act_in_ptrs")) (Op.ref .real [BM, BN] "acc") .none) stP
        = some ((TileShape.allIndices [BM, BN]).foldl
            (fun acc i =>
              if (fun _ : TileIndex [BM, BN] => True) i then
                acc.writeMem ACT_INPUTS (actOffset s0 M N BM BN GM saim sain i) (v i)
              else acc) stP) := by
      simp only [stepStmt]
      rw [show evalOp (Op.ref .real [BM, BN] "acc") stP = some accT from by
            rw [evalOp_ref]; simp [hstP, hacc, haccT]]
      rw [show evalOp (Op.ref .ptr [BM, BN] "act_in_ptrs") stP = some apT from by
            rw [evalOp_ref]; simp [hstP]]
      simp only [bind, Option.bind_some]
      refine congrArg some (List.foldl_ext _ _ _ (fun acc i _ => ?_))
      simp only [hapT, haccT, if_true, Region.cast_id, BlockState.writeMemTyped_real,
        FloatDType.real_storeValue]
      rfl
    rw [stepStmts.cons_some hstore, stepStmts.nil]
    refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_⟩
    · intro dt sh nm hne
      rw [BlockState.foldl_writeMem_prop_masked_regs]
      simp [hstP, hne]
    · rw [foldl_writeMem_masked_pids ACT_INPUTS
          (fun idx : TileIndex [BM, BN] => actOffset s0 M N BM BN GM saim sain idx)
          (fun idx : TileIndex [BM, BN] => v idx)
          (fun _ : TileIndex [BM, BN] => True)]
      simp [hstP]
    · intro R o hR
      rw [BlockState.scatter_prop_masked_preserves_other_region
        (region := ACT_INPUTS)
        (fun idx : TileIndex [BM, BN] => actOffset s0 M N BM BN GM saim sain idx)
        (fun idx : TileIndex [BM, BN] => v idx)
        (fun _ : TileIndex [BM, BN] => True) R hR o
        (TileShape.allIndices [BM, BN]) stP]
      simp [hstP]
    · intro h; exact nomatch h
    · intro _ hInj idx
      rw [BlockState.scatter_readback_prop_masked_nd stP
        (fun idx : TileIndex [BM, BN] => actOffset s0 M N BM BN GM saim sain idx)
        (fun idx : TileIndex [BM, BN] => v idx)
        (fun _ : TileIndex [BM, BN] => True) hInj idx]
      simp

/-- The `tanh` gate body eval. -/
theorem tanh_op_eval (st : BlockState) (BM BN : Nat) (v : TileIndex [BM, BN] → ℝ)
    (hacc : st.regs .real [BM, BN] "acc" = some ⟨fun idx => some (v idx)⟩) :
    evalOp (Op.tanh (Op.ref .real [BM, BN] "acc")) st
      = some ⟨fun idx => some (Real.tanh (v idx))⟩ := by
  rw [evalOp_tanh]
  simp only [evalOp_ref, hacc, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  simp [Tile.uop]

/-- The inlined `gelu` gate body eval: `acc * 0.5 * (1.0 + erf(acc / √2))`. -/
theorem gelu_op_eval (st : BlockState) (BM BN : Nat) (v : TileIndex [BM, BN] → ℝ)
    (hacc : st.regs .real [BM, BN] "acc" = some ⟨fun idx => some (v idx)⟩) :
    evalOp (Op.mul .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.mul .real Broadcast.scalarR (Op.ref .real [BM, BN] "acc") (Op.const 0.5))
        (Op.add .real Broadcast.scalarL (Op.const 1.0)
          (Op.erf (Op.div .real Broadcast.scalarR (Op.ref .real [BM, BN] "acc")
            (Op.const (Real.sqrt 2)))))) st
      = some ⟨fun idx => some (geluRef (v idx))⟩ := by
  simp only [evalOp_mul, evalOp_add, evalOp_div, evalOp_erf, evalOp_const, evalOp_ref, hacc,
    Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  simp only [Tile.bop_data, Tile.uop, Tile.scalar_data, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.mul, NumericDType.add, NumericDType.div, WithBot.realMul, WithBot.realAdd,
    WithBot.realDiv, WithBot.realErf, Option.map₂, Option.bind, Option.map, geluRef]

/-- The inlined `fast_gelu` gate body eval:
`0.5 * acc * (1 + tanh(√(2/π) * (acc + 0.044715 * acc * acc * acc)))`. -/
theorem fast_gelu_op_eval (st : BlockState) (BM BN : Nat) (v : TileIndex [BM, BN] → ℝ)
    (hacc : st.regs .real [BM, BN] "acc" = some ⟨fun idx => some (v idx)⟩) :
    evalOp (Op.mul .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.mul .real Broadcast.scalarL (Op.const 0.5) (Op.ref .real [BM, BN] "acc"))
        (Op.add .real Broadcast.scalarL (Op.const 1)
          (Op.tanh (Op.mul .real Broadcast.scalarL (Op.const (Real.sqrt (2.0 / Real.pi)))
            (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
              (Op.ref .real [BM, BN] "acc")
              (Op.mul .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
                (Op.mul .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
                  (Op.mul .real Broadcast.scalarL (Op.const 0.044715) (Op.ref .real [BM, BN] "acc"))
                  (Op.ref .real [BM, BN] "acc"))
                (Op.ref .real [BM, BN] "acc"))))))) st
      = some ⟨fun idx => some (fastGeluRef (v idx))⟩ := by
  simp only [evalOp_mul, evalOp_add, evalOp_tanh, evalOp_const, evalOp_ref, hacc,
    Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  simp only [Tile.bop_data, Tile.uop, Tile.scalar_data, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.mul, NumericDType.add, WithBot.realMul, WithBot.realAdd, WithBot.realTanh,
    Option.map₂, Option.bind, Option.map, fastGeluRef]

/-- The inlined `relu` gate body eval: `tl.maximum(0, acc)` =
`tl.where(0 > acc, 0, acc)` = `max 0 acc` lane-wise. -/
theorem relu_op_eval (st : BlockState) (BM BN : Nat) (v : TileIndex [BM, BN] → ℝ)
    (hacc : st.regs .real [BM, BN] "acc" = some ⟨fun idx => some (v idx)⟩) :
    evalOp ((Op.gt ComparableDType.real Broadcast.scalarL (Op.const 0) (Op.ref .real [BM, BN] "acc")).where
        ((Op.const 0).broadcast [BM, BN]) (Op.ref .real [BM, BN] "acc")) st
      = some ⟨fun idx => some (max 0 (v idx))⟩ := by
  have hzero : evalOp ((Op.const (0 : ℝ)).broadcast [BM, BN]) st
      = some (⟨fun _ => some (0 : ℝ)⟩ : Tile .real [BM, BN]) := by
    simp only [evalOp, evalOp_const, Option.bind_some]
    rfl
  rw [evalOp_where, evalOp_gt]
  simp only [evalOp_ref, evalOp_const, hacc, hzero, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  simp only [Tile.select_data, Tile.cop_data, Tile.scalar_data, Broadcast.leftIndex,
    Broadcast.rightIndex]
  by_cases h : (0 : ℝ) > v idx
  · rw [if_pos, max_eq_left h.le]
    simp only [ComparableDType.real_gt_eq_true, gt_iff_lt]
    exact WithBot.coe_lt_coe.mpr h
  · rw [if_neg, max_eq_right (not_lt.mp h)]
    intro hc
    exact h (WithBot.coe_lt_coe.mp
      ((ComparableDType.real_gt_eq_true (some 0) (some (v idx))).mp hc))

set_option maxHeartbeats 1000000 in
/-- **Activation-tail stepping**: the four sequential `ACTIVATION == "…"` gates
map `acc` through `applyActivation ACTIVATION`; everything else untouched. -/
theorem actTail_steps (st : BlockState) (BM BN : Nat) (ACTIVATION : String)
    (v : TileIndex [BM, BN] → ℝ)
    (hacc : st.regs .real [BM, BN] "acc" = some ⟨fun idx => some (v idx)⟩) :
    ∃ st', stepStmts (actStmts BM BN ACTIVATION) st = some st'
      ∧ st'.regs .real [BM, BN] "acc"
          = some ⟨fun idx => some (applyActivation ACTIVATION (v idx))⟩
      ∧ (∀ {dt} {sh} (nm : RegName), nm ≠ "acc" → st'.regs dt sh nm = st.regs dt sh nm)
      ∧ st'.mem = st.mem ∧ st'.pids = st.pids := by
  unfold actStmts
  -- gate 1: tanh
  obtain ⟨s1, h1, hacc1, hpres1, hmem1, hpids1⟩ :
      ∃ s1, stepStmt (Stmt.ifThen (Op.constBool (ACTIVATION == "tanh"))
          [Stmt.assign .real [BM, BN] "acc" (Op.tanh (Op.ref .real [BM, BN] "acc"))]) st = some s1
        ∧ s1.regs .real [BM, BN] "acc" = some ⟨fun idx =>
            some (if ACTIVATION == "tanh" then Real.tanh (v idx) else v idx)⟩
        ∧ (∀ {dt} {sh} (nm : RegName), nm ≠ "acc" → s1.regs dt sh nm = st.regs dt sh nm)
        ∧ s1.mem = st.mem ∧ s1.pids = st.pids := by
    rw [stepStmt_ifThen_constBool]
    cases hb : (ACTIVATION == "tanh")
    · refine ⟨st, rfl, ?_, fun _ _ => rfl, rfl, rfl⟩
      rw [hacc]
      exact congrArg some (by ext idx; simp)
    · rw [stepStmts.cons_some (stepStmt_assign_eq_some (tanh_op_eval st BM BN v hacc)),
        stepStmts.nil]
      exact ⟨_, rfl, by simp, fun nm h => by simp [BlockState.setReg_ne_name, h], rfl, rfl⟩
  rw [stepStmts.cons_some h1]
  -- gate 2: gelu
  obtain ⟨s2, h2, hacc2, hpres2, hmem2, hpids2⟩ :
      ∃ s2, stepStmt (Stmt.ifThen (Op.constBool (ACTIVATION == "gelu"))
          [Stmt.assign .real [BM, BN] "acc"
            (Op.mul .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
              (Op.mul .real Broadcast.scalarR (Op.ref .real [BM, BN] "acc") (Op.const 0.5))
              (Op.add .real Broadcast.scalarL (Op.const 1.0)
                (Op.erf (Op.div .real Broadcast.scalarR (Op.ref .real [BM, BN] "acc")
                  (Op.const (Real.sqrt 2))))))]) s1 = some s2
        ∧ s2.regs .real [BM, BN] "acc" = some ⟨fun idx =>
            some (if ACTIVATION == "gelu"
              then geluRef (if ACTIVATION == "tanh" then Real.tanh (v idx) else v idx)
              else (if ACTIVATION == "tanh" then Real.tanh (v idx) else v idx))⟩
        ∧ (∀ {dt} {sh} (nm : RegName), nm ≠ "acc" → s2.regs dt sh nm = s1.regs dt sh nm)
        ∧ s2.mem = s1.mem ∧ s2.pids = s1.pids := by
    rw [stepStmt_ifThen_constBool]
    cases hb : (ACTIVATION == "gelu")
    · refine ⟨s1, rfl, ?_, fun _ _ => rfl, rfl, rfl⟩
      rw [hacc1]
      exact congrArg some (by ext idx; simp)
    · rw [stepStmts.cons_some (stepStmt_assign_eq_some (gelu_op_eval s1 BM BN _ hacc1)),
        stepStmts.nil]
      exact ⟨_, rfl, by simp, fun nm h => by simp [BlockState.setReg_ne_name, h], rfl, rfl⟩
  rw [stepStmts.cons_some h2]
  -- gate 3: fast_gelu
  obtain ⟨s3, h3, hacc3, hpres3, hmem3, hpids3⟩ :
      ∃ s3, stepStmt (Stmt.ifThen (Op.constBool (ACTIVATION == "fast_gelu"))
          [Stmt.assign .real [BM, BN] "acc"
            (Op.mul .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
              (Op.mul .real Broadcast.scalarL (Op.const 0.5) (Op.ref .real [BM, BN] "acc"))
              (Op.add .real Broadcast.scalarL (Op.const 1)
                (Op.tanh (Op.mul .real Broadcast.scalarL (Op.const (Real.sqrt (2.0 / Real.pi)))
                  (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
                    (Op.ref .real [BM, BN] "acc")
                    (Op.mul .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
                      (Op.mul .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
                        (Op.mul .real Broadcast.scalarL (Op.const 0.044715) (Op.ref .real [BM, BN] "acc"))
                        (Op.ref .real [BM, BN] "acc"))
                      (Op.ref .real [BM, BN] "acc")))))))]) s2 = some s3
        ∧ s3.regs .real [BM, BN] "acc" = some ⟨fun idx =>
            some (if ACTIVATION == "fast_gelu"
              then fastGeluRef (if ACTIVATION == "gelu"
                then geluRef (if ACTIVATION == "tanh" then Real.tanh (v idx) else v idx)
                else (if ACTIVATION == "tanh" then Real.tanh (v idx) else v idx))
              else (if ACTIVATION == "gelu"
                then geluRef (if ACTIVATION == "tanh" then Real.tanh (v idx) else v idx)
                else (if ACTIVATION == "tanh" then Real.tanh (v idx) else v idx)))⟩
        ∧ (∀ {dt} {sh} (nm : RegName), nm ≠ "acc" → s3.regs dt sh nm = s2.regs dt sh nm)
        ∧ s3.mem = s2.mem ∧ s3.pids = s2.pids := by
    rw [stepStmt_ifThen_constBool]
    cases hb : (ACTIVATION == "fast_gelu")
    · refine ⟨s2, rfl, ?_, fun _ _ => rfl, rfl, rfl⟩
      rw [hacc2]
      exact congrArg some (by ext idx; simp)
    · rw [stepStmts.cons_some (stepStmt_assign_eq_some (fast_gelu_op_eval s2 BM BN _ hacc2)),
        stepStmts.nil]
      exact ⟨_, rfl, by simp, fun nm h => by simp [BlockState.setReg_ne_name, h], rfl, rfl⟩
  rw [stepStmts.cons_some h3]
  -- gate 4: relu
  obtain ⟨s4, h4, hacc4, hpres4, hmem4, hpids4⟩ :
      ∃ s4, stepStmt (Stmt.ifThen (Op.constBool (ACTIVATION == "relu"))
          [Stmt.assign .real [BM, BN] "acc"
            ((Op.gt ComparableDType.real Broadcast.scalarL (Op.const 0) (Op.ref .real [BM, BN] "acc")).where
              ((Op.const 0).broadcast [BM, BN]) (Op.ref .real [BM, BN] "acc"))]) s3 = some s4
        ∧ s4.regs .real [BM, BN] "acc" = some ⟨fun idx =>
            some (applyActivation ACTIVATION (v idx))⟩
        ∧ (∀ {dt} {sh} (nm : RegName), nm ≠ "acc" → s4.regs dt sh nm = s3.regs dt sh nm)
        ∧ s4.mem = s3.mem ∧ s4.pids = s3.pids := by
    rw [stepStmt_ifThen_constBool]
    cases hb : (ACTIVATION == "relu")
    · refine ⟨s3, rfl, ?_, fun _ _ => rfl, rfl, rfl⟩
      rw [hacc3]
      refine congrArg some ?_
      ext idx
      simp only [applyActivation, hb, Bool.false_eq_true, if_false]
    · rw [stepStmts.cons_some (stepStmt_assign_eq_some (relu_op_eval s3 BM BN _ hacc3)),
        stepStmts.nil]
      refine ⟨_, rfl, ?_, fun nm h => by simp [BlockState.setReg_ne_name, h], rfl, rfl⟩
      simp only [BlockState.setReg_same]
      refine congrArg some ?_
      ext idx
      simp only [applyActivation, hb, if_true]
  rw [stepStmts.cons_some h4, stepStmts.nil]
  refine ⟨s4, rfl, hacc4, ?_, ?_, ?_⟩
  · intro dt sh nm h
    rw [hpres4 nm h, hpres3 nm h, hpres2 nm h, hpres1 nm h]
  · rw [hmem4, hmem3, hmem2, hmem1]
  · rw [hpids4, hpids3, hpids2, hpids1]

/-- `c_ptr_mask` eval: the `(m_offs < M)[:, None] & (n_offs < N)[None, :]`
boolean tile. -/
theorem cmask_eval (st : BlockState) (M N BM BN : Nat)
    (gm : Fin BM → Nat) (gn : Fin BN → Nat)
    (hm : st.regs .nat [BM] "m_offs" = some (Tile.vec gm))
    (hn : st.regs .nat [BN] "n_offs" = some (Tile.vec gn)) :
    evalOp (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.expandDim ⟨1, by simp⟩
          (Op.lt ComparableDType.nat Broadcast.scalarR (Op.ref .nat [BM] "m_offs") (Op.constNat M)))
        (Op.expandDim ⟨0, by simp⟩
          (Op.lt ComparableDType.nat Broadcast.scalarR (Op.ref .nat [BN] "n_offs") (Op.constNat N)))) st
      = some ⟨fun idx : TileIndex [BM, BN] =>
          (decide (gm idx.1 < M) && decide (gn idx.2.1 < N))⟩ := by
  have hx0 : evalOp (Op.expandDim ⟨1, by simp⟩
      (Op.lt ComparableDType.nat Broadcast.scalarR (Op.ref .nat [BM] "m_offs") (Op.constNat M))) st
      = some (Tile.expandDim ⟨1, by simp⟩
          (Tile.cop ComparableDType.nat.lt Broadcast.scalarR (Tile.vec gm) (Tile.scalar M))) := by
    rw [evalOp_expandDim, evalOp_lt]
    simp [hm]
  have hx : evalOp (shape := [BM, 1]) (Op.expandDim ⟨1, by simp⟩
      (Op.lt ComparableDType.nat Broadcast.scalarR (Op.ref .nat [BM] "m_offs") (Op.constNat M))) st
      = some (Tile.expandDim ⟨1, by simp⟩
          (Tile.cop ComparableDType.nat.lt Broadcast.scalarR (Tile.vec gm) (Tile.scalar M))) := hx0
  have hy0 : evalOp (Op.expandDim ⟨0, by simp⟩
      (Op.lt ComparableDType.nat Broadcast.scalarR (Op.ref .nat [BN] "n_offs") (Op.constNat N))) st
      = some (Tile.expandDim ⟨0, by simp⟩
          (Tile.cop ComparableDType.nat.lt Broadcast.scalarR (Tile.vec gn) (Tile.scalar N))) := by
    rw [evalOp_expandDim, evalOp_lt]
    simp [hn]
  have hy : evalOp (shape := [1, BN]) (Op.expandDim ⟨0, by simp⟩
      (Op.lt ComparableDType.nat Broadcast.scalarR (Op.ref .nat [BN] "n_offs") (Op.constNat N))) st
      = some (Tile.expandDim ⟨0, by simp⟩
          (Tile.cop ComparableDType.nat.lt Broadcast.scalarR (Tile.vec gn) (Tile.scalar N))) := hy0
  rw [evalOp_boolAnd]
  simp only [hx, hy, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  simp only [Tile.bop_data, Tile.cop_data, Tile.expandDim, Tile.vec, Tile.scalar_data,
    ComparableDType.lt, Broadcast.leftIndex, Broadcast.rightIndex]
  rfl

set_option maxHeartbeats 2000000 in
/-- **postLoop**: from the invariant at `numKBlocks` blocks, the save-gate +
activation gates + masked `C` store produce, at every in-mask lane, the genuine
closed form `applyActivation(linearSpec)` in `C`, and (when saving, with
`ACT_INPUTS ≠ C`) the un-activated `linearSpec` in `ACT_INPUTS`. -/
theorem kernel_fma_postLoop (C ACT_INPUTS A B biasR : RegionName) (s0 : BlockState)
    (M N smo sno saim sain sam sak sbk sbn BM BN GM BLOCK_K numKBlocks : Nat)
    (HAS_BIAS SHOULD_SAVE_ACT_INPUTS : Bool) (ACTIVATION : String)
    (hInjC : Function.Injective (cOffset s0 M N BM BN GM smo sno))
    (st : BlockState)
    (hinv : fmaInvariant A B biasR s0 M N BM BN GM sam sak sbk sbn BLOCK_K numKBlocks HAS_BIAS numKBlocks st) :
    ∃ sfin, stepStmts (saveIfStmt ACT_INPUTS saim sain BM BN SHOULD_SAVE_ACT_INPUTS
        :: (actStmts BM BN ACTIVATION ++ storeTail C M N smo sno BM BN)) st = some sfin
      ∧ (∀ idx : TileIndex [BM, BN],
          rowIndex s0 M N BM BN GM idx.1 < M → colIndex s0 M N BM BN GM idx.2.1 < N →
          sfin.readMem C (cOffset s0 M N BM BN GM smo sno idx)
            = applyActivation ACTIVATION
                (linearSpec s0 A B biasR M N BM BN GM sam sak sbk sbn BLOCK_K numKBlocks HAS_BIAS idx.1 idx.2.1))
      ∧ (SHOULD_SAVE_ACT_INPUTS = Bool.true → ACT_INPUTS ≠ C →
          Function.Injective (actOffset s0 M N BM BN GM saim sain) →
          ∀ idx : TileIndex [BM, BN],
            sfin.readMem ACT_INPUTS (actOffset s0 M N BM BN GM saim sain idx)
              = linearSpec s0 A B biasR M N BM BN GM sam sak sbk sbn BLOCK_K numKBlocks HAS_BIAS idx.1 idx.2.1) := by
  simp only [fmaInvariant] at hinv
  obtain ⟨hcle, hm, hn, hz, hap, hbp, hmem⟩ := hinv
  -- the pre-activation value carried through the tail
  set w : TileIndex [BM, BN] → ℝ := fun idx =>
    accPartial s0 A B biasR M N BM BN GM sam sak sbk sbn BLOCK_K HAS_BIAS idx.1 idx.2.1 numKBlocks
    with hw
  have hwspec : ∀ idx : TileIndex [BM, BN],
      w idx = linearSpec s0 A B biasR M N BM BN GM sam sak sbk sbn BLOCK_K numKBlocks HAS_BIAS idx.1 idx.2.1 := by
    intro idx
    simp only [hw, accPartial, linearSpec, Nat.mul_comm numKBlocks BLOCK_K]
  -- 1. the save gate
  obtain ⟨s18, hsave, hpres18, hpids18, hmemO18, hmemF18, hsaved⟩ :=
    saveIf_steps st ACT_INPUTS s0 M N BM BN GM saim sain SHOULD_SAVE_ACT_INPUTS w hm hn
      (by rw [hz])
  rw [stepStmts.cons_some hsave]
  -- 2. the activation gates
  have hacc18 : s18.regs .real [BM, BN] "acc" = some ⟨fun idx => some (w idx)⟩ := by
    rw [hpres18 "acc" (by decide)]
    rw [hz]
  obtain ⟨s22, hact, hacc22, hpres22, hmem22, hpids22⟩ :=
    actTail_steps s18 BM BN ACTIVATION w hacc18
  rw [stepStmts.append_some hact]
  -- registers surviving to the store tail
  have hm22 : s22.regs .nat [BM] "m_offs" = some (Tile.vec (fun i : Fin BM => rowIndex s0 M N BM BN GM i)) := by
    rw [hpres22 "m_offs" (by decide), hpres18 "m_offs" (by decide)]; exact hm
  have hn22 : s22.regs .nat [BN] "n_offs" = some (Tile.vec (fun j : Fin BN => colIndex s0 M N BM BN GM j)) := by
    rw [hpres22 "n_offs" (by decide), hpres18 "n_offs" (by decide)]; exact hn
  -- 3. the C pointer chunk
  set cpT : Tile .ptr [BM, BN] :=
    ⟨fun idx : TileIndex [BM, BN] => (C.cast, cOffset s0 M N BM BN GM smo sno idx)⟩ with hcpT
  have hstepC : stepStmt (Stmt.assign .ptr [BM, BN] "C"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase C)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "m_offs")) (Op.constNat smo))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "n_offs")) (Op.constNat sno))))) s22
      = some (s22.setReg "C" .ptr [BM, BN] cpT) :=
    stepStmt_assign_eq_some (pairptrs_eval s22 C BM BN smo sno "m_offs" "n_offs"
      (fun i => rowIndex s0 M N BM BN GM i) (fun j => colIndex s0 M N BM BN GM j) hm22 hn22)
  unfold storeTail
  rw [stepStmts.cons_some hstepC]
  set s23 := s22.setReg "C" .ptr [BM, BN] cpT with hs23
  -- 4. the store mask
  set cmT : Tile .bool [BM, BN] :=
    ⟨fun idx : TileIndex [BM, BN] =>
      (decide (rowIndex s0 M N BM BN GM idx.1 < M) && decide (colIndex s0 M N BM BN GM idx.2.1 < N))⟩
    with hcmT
  have hstepMask : stepStmt (Stmt.assign .bool [BM, BN] "c_ptr_mask"
      (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.expandDim ⟨1, by simp⟩
          (Op.lt ComparableDType.nat Broadcast.scalarR (Op.ref .nat [BM] "m_offs") (Op.constNat M)))
        (Op.expandDim ⟨0, by simp⟩
          (Op.lt ComparableDType.nat Broadcast.scalarR (Op.ref .nat [BN] "n_offs") (Op.constNat N))))) s23
      = some (s23.setReg "c_ptr_mask" .bool [BM, BN] cmT) :=
    stepStmt_assign_eq_some (cmask_eval s23 M N BM BN
      (fun i => rowIndex s0 M N BM BN GM i) (fun j => colIndex s0 M N BM BN GM j)
      (by simp [hs23, hm22]) (by simp [hs23, hn22]))
  rw [stepStmts.cons_some hstepMask]
  set s24 := s23.setReg "c_ptr_mask" .bool [BM, BN] cmT with hs24
  -- 5. the masked store
  set accT : Tile .real [BM, BN] :=
    ⟨fun idx => some (applyActivation ACTIVATION (w idx))⟩ with haccT
  have hstore : stepStmt (Stmt.store .real [BM, BN] (.ptr (Op.ref .ptr [BM, BN] "C"))
        (Op.ref .real [BM, BN] "acc") (.mask (Op.ref .bool [BM, BN] "c_ptr_mask"))) s24
      = some ((TileShape.allIndices [BM, BN]).foldl
          (fun acc i =>
            if cmT.data i then
              acc.writeMem C (cOffset s0 M N BM BN GM smo sno i)
                (applyActivation ACTIVATION (w i))
            else acc) s24) := by
    simp only [stepStmt]
    rw [show evalOp (Op.ref .real [BM, BN] "acc") s24 = some accT from by
          rw [evalOp_ref]; simp [hs24, hs23, hacc22, haccT]]
    rw [show evalOp (Op.ref .ptr [BM, BN] "C") s24 = some cpT from by
          rw [evalOp_ref]; simp [hs24, hs23]]
    rw [show evalOp (Op.ref .bool [BM, BN] "c_ptr_mask") s24 = some cmT from by
          rw [evalOp_ref]; simp [hs24]]
    simp only [bind, Option.bind_some]
    refine congrArg some (List.foldl_ext _ _ _ (fun acc i _ => ?_))
    by_cases hmask : cmT.data i
    · simp only [hmask, if_true, hcpT, haccT, Region.cast_id, BlockState.writeMemTyped_real,
        FloatDType.real_storeValue]
      rfl
    · simp only [hmask, Bool.false_eq_true, if_false]
  rw [stepStmts.cons_some hstore, stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_⟩
  · -- C readback at in-mask lanes
    intro idx hrow hcol
    rw [BlockState.scatter_readback_prop_masked_nd s24
      (fun i : TileIndex [BM, BN] => cOffset s0 M N BM BN GM smo sno i)
      (fun i : TileIndex [BM, BN] => applyActivation ACTIVATION (w i))
      (fun i : TileIndex [BM, BN] => cmT.data i = Bool.true) hInjC idx]
    rw [if_pos (by simp [hcmT, hrow, hcol])]
    rw [hwspec]
  · -- ACT_INPUTS readback
    intro hSave hne hInjAct idx
    have hpresC := BlockState.scatter_prop_masked_preserves_other_region
      (region := C)
      (fun i : TileIndex [BM, BN] => cOffset s0 M N BM BN GM smo sno i)
      (fun i : TileIndex [BM, BN] => applyActivation ACTIVATION (w i))
      (fun i : TileIndex [BM, BN] => cmT.data i = Bool.true)
      ACT_INPUTS hne (actOffset s0 M N BM BN GM saim sain idx)
      (TileShape.allIndices [BM, BN]) s24
    rw [hpresC]
    have hs24read : s24.readMem ACT_INPUTS (actOffset s0 M N BM BN GM saim sain idx)
        = s18.readMem ACT_INPUTS (actOffset s0 M N BM BN GM saim sain idx) := by
      simp only [hs24, hs23, BlockState.setReg_readMem]
      unfold BlockState.readMem
      rw [hmem22]
    rw [hs24read, hsaved hSave hInjAct idx, hwspec]

set_option maxHeartbeats 2000000 in
/-- **Top exec reduction**: composes `preLoop` + `kernel_fma_step` (driven by
`forRange_inv`) + `kernel_fma_postLoop` into the full `exec` result. -/
theorem kernel_fma_exec_closed_form
    (C ACT_INPUTS A B biasR : RegionName) (s : BlockState)
    (M N smo sno saim sain sam sak sbn sbk BM GM BN BLOCK_K numKBlocks : Nat)
    (HAS_BIAS SHOULD_SAVE_ACT_INPUTS : Bool) (ACTIVATION : String)
    (hInjC : Function.Injective (cOffset s M N BM BN GM smo sno)) :
    ∃ s', exec (kernel_fma_surface C ACT_INPUTS A B biasR M N smo sno saim sain
        sam sak sbn sbk BM GM BN BLOCK_K numKBlocks
        HAS_BIAS SHOULD_SAVE_ACT_INPUTS ACTIVATION) s = some s'
      ∧ (∀ idx : TileIndex [BM, BN],
          rowIndex s M N BM BN GM idx.1 < M → colIndex s M N BM BN GM idx.2.1 < N →
          s'.readMem C (cOffset s M N BM BN GM smo sno idx)
            = applyActivation ACTIVATION
                (linearSpec s A B biasR M N BM BN GM sam sak sbk sbn BLOCK_K numKBlocks HAS_BIAS idx.1 idx.2.1))
      ∧ (SHOULD_SAVE_ACT_INPUTS = Bool.true → ACT_INPUTS ≠ C →
          Function.Injective (actOffset s M N BM BN GM saim sain) →
          ∀ idx : TileIndex [BM, BN],
            s'.readMem ACT_INPUTS (actOffset s M N BM BN GM saim sain idx)
              = linearSpec s A B biasR M N BM BN GM sam sak sbk sbn BLOCK_K numKBlocks HAS_BIAS idx.1 idx.2.1) := by
  obtain ⟨s17, hpre_eq, hP0⟩ := preLoop C ACT_INPUTS A B biasR s M N smo sno saim sain
    sam sak sbn sbk BM GM BN BLOCK_K numKBlocks HAS_BIAS SHOULD_SAVE_ACT_INPUTS ACTIVATION
  obtain ⟨final, sLoop, hLoopStmt, hfinal, hPLoop⟩ :=
    forRange_inv (idx := "k") (start := 0) (stop := numKBlocks) (step := 1)
      (by omega) hP0
      (fun c st hlt hinv => by
        have := kernel_fma_step A B biasR s M N BM BN GM sam sak sbk sbn BLOCK_K numKBlocks
          HAS_BIAS c st hlt hinv
        simpa using this)
  have hfinalEq : final = numKBlocks := by
    have hle : final ≤ numKBlocks := by
      simp only [fmaInvariant] at hPLoop
      exact hPLoop.1
    exact le_antisymm hle hfinal
  rw [hfinalEq] at hPLoop
  obtain ⟨sfin, hTail, hC, hAct⟩ :=
    kernel_fma_postLoop C ACT_INPUTS A B biasR s M N smo sno saim sain sam sak sbk sbn
      BM BN GM BLOCK_K numKBlocks HAS_BIAS SHOULD_SAVE_ACT_INPUTS ACTIVATION hInjC sLoop hPLoop
  refine ⟨sfin, ?_, hC, hAct⟩
  rw [exec, fma_body_split C ACT_INPUTS A B biasR M N smo sno saim sain
      sam sak sbn sbk BM GM BN BLOCK_K numKBlocks HAS_BIAS SHOULD_SAVE_ACT_INPUTS ACTIVATION,
    stepStmts.append_some hpre_eq, stepStmts.cons_some hLoopStmt, hTail]

/-- Store-footprint injectivity of the wrapped `rowIndex·stride_m + colIndex`
map for a **fitting** tile (unit column stride, `BN ≤` the row stride): the
`%` wraps are the identity, so the map is a row-major encoding of `(i, j)`. -/
theorem cOffset_injective_of_fit (s : BlockState) (M N BM BN GM smo : Nat)
    (hFitM : blockMIdx (s.pids 0) M N BM BN GM * BM + BM ≤ M)
    (hFitN : blockNIdx (s.pids 0) M N BM BN GM * BN + BN ≤ N)
    (hble : BN ≤ smo) :
    Function.Injective (cOffset s M N BM BN GM smo 1) := by
  have hrow : ∀ i : Fin BM, rowIndex s M N BM BN GM i
      = blockMIdx (s.pids 0) M N BM BN GM * BM + i.val := by
    intro i
    unfold rowIndex rowGlobal
    exact Nat.mod_eq_of_lt (by have := i.isLt; omega)
  have hcol : ∀ j : Fin BN, colIndex s M N BM BN GM j
      = blockNIdx (s.pids 0) M N BM BN GM * BN + j.val := by
    intro j
    unfold colIndex colGlobal
    exact Nat.mod_eq_of_lt (by have := j.isLt; omega)
  have heq : cOffset s M N BM BN GM smo 1
      = fun idx : TileIndex [BM, BN] =>
          (blockMIdx (s.pids 0) M N BM BN GM * BM * smo + blockNIdx (s.pids 0) M N BM BN GM * BN)
            + idx.1.val * smo + idx.2.1.val := by
    funext idx
    unfold cOffset
    rw [hrow, hcol]
    ring
  rw [heq]
  exact rowMajor2D_inj _ smo hble

/-- A fitting tile's wrapped row index is in range (`m_offs i < M`). -/
theorem rowIndex_lt_of_fit (s : BlockState) (M N BM BN GM : Nat)
    (hFitM : blockMIdx (s.pids 0) M N BM BN GM * BM + BM ≤ M) (i : Fin BM) :
    rowIndex s M N BM BN GM i < M := by
  unfold rowIndex rowGlobal
  have hlt : blockMIdx (s.pids 0) M N BM BN GM * BM + i.val < M := by
    have := i.isLt; omega
  rw [Nat.mod_eq_of_lt hlt]
  exact hlt

/-- A fitting tile's wrapped column index is in range (`n_offs j < N`). -/
theorem colIndex_lt_of_fit (s : BlockState) (M N BM BN GM : Nat)
    (hFitN : blockNIdx (s.pids 0) M N BM BN GM * BN + BN ≤ N) (j : Fin BN) :
    colIndex s M N BM BN GM j < N := by
  unfold colIndex colGlobal
  have hlt : blockNIdx (s.pids 0) M N BM BN GM * BN + j.val < N := by
    have := j.isLt; omega
  rw [Nat.mod_eq_of_lt hlt]
  exact hlt

/-- Compute-facing correctness of the `C` output. -/
theorem kernel_fma_C_compute_correct
    (C ACT_INPUTS A B biasR : RegionName) (s : BlockState)
    (M N smo sno saim sain sam sak sbn sbk BM GM BN BLOCK_K numKBlocks : Nat)
    (HAS_BIAS SHOULD_SAVE_ACT_INPUTS : Bool) (ACTIVATION : String)
    (hFitM : blockMIdx (s.pids 0) M N BM BN GM * BM + BM ≤ M)
    (hFitN : blockNIdx (s.pids 0) M N BM BN GM * BN + BN ≤ N)
    (hsno : sno = 1) (hble : BN ≤ smo) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := kernel_fma_surface C ACT_INPUTS A B biasR M N smo sno saim sain
        sam sak sbn sbk BM GM BN BLOCK_K numKBlocks
        HAS_BIAS SHOULD_SAVE_ACT_INPUTS ACTIVATION)
      (initialState := s)
      (write := fun idx : TileIndex [BM, BN] =>
        some (C, cOffset s M N BM BN GM smo sno idx))
      (expected := fun idx : TileIndex [BM, BN] =>
        applyActivation ACTIVATION
          (linearSpec s A B biasR M N BM BN GM sam sak sbk sbn BLOCK_K numKBlocks HAS_BIAS idx.1 idx.2.1)) := by
  subst hsno
  have hInjC : Function.Injective (cOffset s M N BM BN GM smo 1) :=
    cOffset_injective_of_fit s M N BM BN GM smo hFitM hFitN hble
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [kernel_fma_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst hs0
  intro idx
  obtain ⟨s'', hexec, hC, _⟩ := kernel_fma_exec_closed_form C ACT_INPUTS A B biasR s0
    M N smo 1 saim sain sam sak sbn sbk BM GM BN BLOCK_K numKBlocks
    HAS_BIAS SHOULD_SAVE_ACT_INPUTS ACTIVATION hInjC
  have hs'' : s' = s'' := by
    have h2 := hExec
    rw [hexec] at h2
    exact (Option.some.inj h2).symm
  subst hs''
  exact hC idx (rowIndex_lt_of_fit s0 M N BM BN GM hFitM idx.1)
    (colIndex_lt_of_fit s0 M N BM BN GM hFitN idx.2.1)

/-- Compute-facing correctness of the optional `ACT_INPUTS` output. -/
theorem kernel_fma_act_inputs_compute_correct
    (C ACT_INPUTS A B biasR : RegionName) (s : BlockState)
    (M N smo sno saim sain sam sak sbn sbk BM GM BN BLOCK_K numKBlocks : Nat)
    (HAS_BIAS : Bool) (ACTIVATION : String)
    (hFitM : blockMIdx (s.pids 0) M N BM BN GM * BM + BM ≤ M)
    (hFitN : blockNIdx (s.pids 0) M N BM BN GM * BN + BN ≤ N)
    (hsno : sno = 1) (hble : BN ≤ smo)
    (hne : ACT_INPUTS ≠ C)
    (hsain : sain = 1) (hale : BN ≤ saim) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := kernel_fma_surface C ACT_INPUTS A B biasR M N smo sno saim sain
        sam sak sbn sbk BM GM BN BLOCK_K numKBlocks
        HAS_BIAS Bool.true ACTIVATION)
      (initialState := s)
      (write := fun idx : TileIndex [BM, BN] =>
        some (ACT_INPUTS, actOffset s M N BM BN GM saim sain idx))
      (expected := fun idx : TileIndex [BM, BN] =>
        linearSpec s A B biasR M N BM BN GM sam sak sbk sbn BLOCK_K numKBlocks HAS_BIAS idx.1 idx.2.1) := by
  subst hsno
  subst hsain
  have hInjC : Function.Injective (cOffset s M N BM BN GM smo 1) :=
    cOffset_injective_of_fit s M N BM BN GM smo hFitM hFitN hble
  have hInjAct : Function.Injective (actOffset s M N BM BN GM saim 1) := by
    have h : actOffset s M N BM BN GM saim 1 = cOffset s M N BM BN GM saim 1 := rfl
    rw [h]
    exact cOffset_injective_of_fit s M N BM BN GM saim hFitM hFitN hale
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [kernel_fma_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst hs0
  intro idx
  obtain ⟨s'', hexec, _, hAct⟩ := kernel_fma_exec_closed_form C ACT_INPUTS A B biasR s0
    M N smo 1 saim 1 sam sak sbn sbk BM GM BN BLOCK_K numKBlocks
    HAS_BIAS Bool.true ACTIVATION hInjC
  have hs'' : s' = s'' := by
    have h2 := hExec
    rw [hexec] at h2
    exact (Option.some.inj h2).symm
  subst hs''
  exact hAct rfl hne hInjAct idx

/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/

/-- **Dimension-general bundled correctness summary for
`triton_linear_activation.py`'s `kernel_fma`.**

For arbitrary shapes `M`/`N`, contracted dimension `K = BLOCK_K · numKBlocks`
(the `K_LOAD_MASK_NEEDED` heuristics arm — exact-multiple `K`), strides, tile
dims `BLOCK_M`/`BLOCK_N`, L2-group size `GROUP_M`, linear program id, constexpr
flags `HAS_BIAS` / `SHOULD_SAVE_ACT_INPUTS`, and **any** `ACTIVATION` string,
it packages:

* the surface lowers to the algorithm layer;
* the `C` output: every cell of the computed tile equals
  `applyActivation(ACTIVATION, bias[n_offs j] + Σ_{k<K} A[m_offs i, k]·B[k, n_offs j])`
  — the genuine linear form over **input** memory (a `gemmSum` `Finset.sum`)
  composed with the exact real activation (`Real.tanh`, erf-GELU via
  `VeriTile.Math.realErf`, tanh fast-GELU, `max 0 ·` ReLU, identity for every
  other string, e.g. the benchmark's `""`), never a read-back of the kernel's
  own output;
* the `ACT_INPUTS` output (`SHOULD_SAVE_ACT_INPUTS = true`, with its own
  layout side-conditions and `ACT_INPUTS ≠ C`): every cell equals the
  **un-activated** `bias + Σ_k A·B` pre-activation value.

Honest side-conditions: `hFitM`/`hFitN` — this program's tile fits the output
(`block_m_idx·BLOCK_M + BLOCK_M ≤ M`, `block_n_idx·BLOCK_N + BLOCK_N ≤ N`, true
for every program of the launch grid whenever `BLOCK_M ∣ M ∧ BLOCK_N ∣ N`,
e.g. all benchmark shapes); without them the kernel's `%`-wrapped store offsets
genuinely collide (the store mask tests the already-wrapped offsets, so it
never masks an overhanging lane). Unit minor stride and `BLOCK_N ≤` major
stride (`output_n_stride = 1`, `BLOCK_N ≤ output_m_stride`; the wrapper's
contiguous row-major outputs) give store-footprint injectivity. -/
specification triton_linear_activation_output_summary_general
    (C ACT_INPUTS A B bias : RegionName) (s : BlockState)
    (M N output_m_stride output_n_stride act_inputs_m_stride act_inputs_n_stride
      a_m_stride a_k_stride b_n_stride b_k_stride
      BLOCK_M GROUP_M BLOCK_N BLOCK_K numKBlocks : Nat)
    (HAS_BIAS SHOULD_SAVE_ACT_INPUTS : Bool) (ACTIVATION : String)
    (hFitM : blockMIdx (s.pids 0) M N BLOCK_M BLOCK_N GROUP_M * BLOCK_M + BLOCK_M ≤ M)
    (hFitN : blockNIdx (s.pids 0) M N BLOCK_M BLOCK_N GROUP_M * BLOCK_N + BLOCK_N ≤ N)
    (hsno : output_n_stride = 1) (hble : BLOCK_N ≤ output_m_stride) :
    -- (1) the surface lowers to the algorithm layer
    (∃ alg, (kernel_fma_surface C ACT_INPUTS A B bias M N output_m_stride output_n_stride
      act_inputs_m_stride act_inputs_n_stride a_m_stride a_k_stride b_n_stride b_k_stride
      BLOCK_M GROUP_M BLOCK_N BLOCK_K numKBlocks
      HAS_BIAS SHOULD_SAVE_ACT_INPUTS ACTIVATION).toAlgorithm? = Except.ok alg) ∧
    -- (2) C: genuine fused linear + activation
    ComputeCorrect.Realizes_without_Rounding
      (kernel := kernel_fma_surface C ACT_INPUTS A B bias M N output_m_stride output_n_stride
        act_inputs_m_stride act_inputs_n_stride a_m_stride a_k_stride b_n_stride b_k_stride
        BLOCK_M GROUP_M BLOCK_N BLOCK_K numKBlocks
        HAS_BIAS SHOULD_SAVE_ACT_INPUTS ACTIVATION)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
        some (C, cOffset s M N BLOCK_M BLOCK_N GROUP_M output_m_stride output_n_stride idx))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
        applyActivation ACTIVATION
          (linearSpec s A B bias M N BLOCK_M BLOCK_N GROUP_M a_m_stride a_k_stride
            b_k_stride b_n_stride BLOCK_K numKBlocks HAS_BIAS idx.1 idx.2.1)) ∧
    -- (3) ACT_INPUTS (when saving): the genuine un-activated pre-activation values
    (SHOULD_SAVE_ACT_INPUTS = Bool.true → ACT_INPUTS ≠ C →
      act_inputs_n_stride = 1 → BLOCK_N ≤ act_inputs_m_stride →
      ComputeCorrect.Realizes_without_Rounding
        (kernel := kernel_fma_surface C ACT_INPUTS A B bias M N output_m_stride output_n_stride
          act_inputs_m_stride act_inputs_n_stride a_m_stride a_k_stride b_n_stride b_k_stride
          BLOCK_M GROUP_M BLOCK_N BLOCK_K numKBlocks
          HAS_BIAS SHOULD_SAVE_ACT_INPUTS ACTIVATION)
        (initialState := s)
        (write := fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
          some (ACT_INPUTS, actOffset s M N BLOCK_M BLOCK_N GROUP_M
            act_inputs_m_stride act_inputs_n_stride idx))
        (expected := fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
          linearSpec s A B bias M N BLOCK_M BLOCK_N GROUP_M a_m_stride a_k_stride
            b_k_stride b_n_stride BLOCK_K numKBlocks HAS_BIAS idx.1 idx.2.1)) := by
  refine ⟨kernel_fma_surface_toAlgorithm_supported C ACT_INPUTS A B bias M N
      output_m_stride output_n_stride act_inputs_m_stride act_inputs_n_stride
      a_m_stride a_k_stride b_n_stride b_k_stride BLOCK_M GROUP_M BLOCK_N BLOCK_K
      numKBlocks HAS_BIAS SHOULD_SAVE_ACT_INPUTS ACTIVATION,
    kernel_fma_C_compute_correct C ACT_INPUTS A B bias s M N output_m_stride output_n_stride
      act_inputs_m_stride act_inputs_n_stride a_m_stride a_k_stride b_n_stride b_k_stride
      BLOCK_M GROUP_M BLOCK_N BLOCK_K numKBlocks HAS_BIAS SHOULD_SAVE_ACT_INPUTS ACTIVATION
      hFitM hFitN hsno hble,
    ?_⟩
  intro hSave hne hsain hale
  subst hSave
  exact kernel_fma_act_inputs_compute_correct C ACT_INPUTS A B bias s M N
    output_m_stride output_n_stride act_inputs_m_stride act_inputs_n_stride
    a_m_stride a_k_stride b_n_stride b_k_stride BLOCK_M GROUP_M BLOCK_N BLOCK_K numKBlocks
    HAS_BIAS ACTIVATION hFitM hFitN hsno hble hne hsain hale

/-! # ══════════ The `⊨[R]` io face (`StreamMasked3DKernelIO₃ₓ₂` + `pre`) ══════════

The streaming-skin restatement of the exact stack above. Three streamed input
channels (`A`, `B` and the *degenerate static* `bias` row), two terminal
stores (`C` and `ACT_INPUTS`), one static `forRange` of `numKBlocks` steps.
Everything below is additive: the exact headline
`triton_linear_activation_output_summary_general` is untouched.
-/

/-! ## Body decomposition with the prologue named -/

/-- The 13 register-only schedule/index statements (statements 0–12): the L2
grouping scalars and the `%`-wrapped index vectors. -/
private def tlaScalars (M N BM BN BK GM : Nat) : List Stmt :=
  [ Stmt.assign .nat [] "program_idx" (Op.programId 0),
    Stmt.assign .nat [] "grid_m"
      (Op.floorDiv IntegralDType.nat Broadcast.nil
        (Op.sub .nat Broadcast.nil
          (Op.add .nat Broadcast.nil (Op.constNat M) (Op.constNat BM)) (Op.constNat 1))
        (Op.constNat BM)),
    Stmt.assign .nat [] "grid_n"
      (Op.floorDiv IntegralDType.nat Broadcast.nil
        (Op.sub .nat Broadcast.nil
          (Op.add .nat Broadcast.nil (Op.constNat N) (Op.constNat BN)) (Op.constNat 1))
        (Op.constNat BN)),
    Stmt.assign .nat [] "width"
      (Op.mul .nat Broadcast.nil (Op.constNat GM) (Op.ref .nat [] "grid_n")),
    Stmt.assign .nat [] "group_idx"
      (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.ref .nat [] "program_idx")
        (Op.ref .nat [] "width")),
    Stmt.assign .nat [] "group_size"
      ((Op.lt ComparableDType.nat Broadcast.nil
          (Op.sub .nat Broadcast.nil (Op.ref .nat [] "grid_m")
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "group_idx") (Op.constNat GM)))
          (Op.constNat GM)).where
        (Op.sub .nat Broadcast.nil (Op.ref .nat [] "grid_m")
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "group_idx") (Op.constNat GM)))
        (Op.constNat GM)),
    Stmt.assign .nat [] "block_m_idx"
      (Op.add .nat Broadcast.nil
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "group_idx") (Op.constNat GM))
        (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] "program_idx")
          (Op.ref .nat [] "group_size"))),
    Stmt.assign .nat [] "block_n_idx"
      (Op.floorDiv IntegralDType.nat Broadcast.nil
        (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] "program_idx")
          (Op.ref .nat [] "width"))
        (Op.ref .nat [] "group_size")),
    Stmt.assign .nat [BM] "m_offs_untagged"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "block_m_idx") (Op.constNat BM))
        (Op.arange BM)),
    Stmt.assign .nat [BN] "n_offs_untagged"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "block_n_idx") (Op.constNat BN))
        (Op.arange BN)),
    Stmt.assign .nat [BM] "m_offs"
      (Op.mod IntegralDType.nat Broadcast.scalarR (Op.ref .nat [BM] "m_offs_untagged")
        (Op.constNat M)),
    Stmt.assign .nat [BN] "n_offs"
      (Op.mod IntegralDType.nat Broadcast.scalarR (Op.ref .nat [BN] "n_offs_untagged")
        (Op.constNat N)),
    Stmt.assign .nat [BK] "k_range_offs" (Op.arange BK) ]

/-- The 4 seed statements (statements 13–16): the `A`/`B` pointer chunks,
`acc = tl.zeros(...)`, and the `HAS_BIAS` constexpr gate. -/
private def tlaSeeds (A B biasR : RegionName) (N sam sak sbk sbn BM BN BK : Nat)
    (HAS_BIAS : Bool) : List Stmt :=
  [ Stmt.assign .ptr [BM, BK] "A"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase A)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "m_offs")) (Op.constNat sam))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BK] "k_range_offs")) (Op.constNat sak)))),
    Stmt.assign .ptr [BK, BN] "B"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase B)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BK] "k_range_offs")) (Op.constNat sbk))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "n_offs")) (Op.constNat sbn)))),
    Stmt.assign .real [BM, BN] "acc" (Op.full [BM, BN] (Op.const 0)),
    biasIfStmt biasR N BM BN HAS_BIAS ]

/-- `body.take 17` is the named prologue. By `rfl`. -/
private theorem tla_take17 (C ACT_INPUTS A B biasR : RegionName)
    (M N smo sno saim sain sam sak sbn sbk BM GM BN BK numKBlocks : Nat)
    (HAS_BIAS SHOULD_SAVE_ACT_INPUTS : Bool) (ACTIVATION : String) :
    (kernel_fma_surface C ACT_INPUTS A B biasR M N smo sno saim sain
        sam sak sbn sbk BM GM BN BK numKBlocks
        HAS_BIAS SHOULD_SAVE_ACT_INPUTS ACTIVATION).toAlgKernel.body.take 17
      = tlaScalars M N BM BN BK GM ++ tlaSeeds A B biasR N sam sak sbk sbn BM BN BK HAS_BIAS :=
  rfl

/-- Body split with the prologue named. By `rfl`. -/
private theorem tla_body_split (C ACT_INPUTS A B biasR : RegionName)
    (M N smo sno saim sain sam sak sbn sbk BM GM BN BK numKBlocks : Nat)
    (HAS_BIAS SHOULD_SAVE_ACT_INPUTS : Bool) (ACTIVATION : String) :
    (kernel_fma_surface C ACT_INPUTS A B biasR M N smo sno saim sain
        sam sak sbn sbk BM GM BN BK numKBlocks
        HAS_BIAS SHOULD_SAVE_ACT_INPUTS ACTIVATION).toAlgKernel.body
      = (tlaScalars M N BM BN BK GM ++ tlaSeeds A B biasR N sam sak sbk sbn BM BN BK HAS_BIAS)
        ++ (Stmt.forRange "k" 0 numKBlocks 1 (fmaLoopBody BM BN BK sak sbk)
            :: saveIfStmt ACT_INPUTS saim sain BM BN SHOULD_SAVE_ACT_INPUTS
            :: (actStmts BM BN ACTIVATION ++ storeTail C M N smo sno BM BN)) :=
  rfl

/-! ## Cast-free degeneration (`execR R` = `exec`, statement by statement)

The kernel contains **no** `Op.castFloat` and both terminal stores are
`Stmt.store .real`, so every statement steps identically under `stepStmtR R`
and `stepStmt`; the exact invariant stack above transports to `execR R`
verbatim and the headline needs no hypothesis on `R`. -/

set_option maxHeartbeats 1000000 in
/-- The 13 schedule statements are cast-free. -/
private theorem tlaScalars_castFree (R : RoundingModel) (M N BM BN BK GM : Nat)
    (t : BlockState) :
    stepStmtsR R (tlaScalars M N BM BN BK GM) t = stepStmts (tlaScalars M N BM BN BK GM) t := by
  simp only [tlaScalars, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def]
  rfl

set_option maxHeartbeats 1000000 in
/-- The 4 seed statements (including the `HAS_BIAS` gate's masked bias load)
are cast-free. -/
private theorem tlaSeeds_castFree (R : RoundingModel) (A B biasR : RegionName)
    (N sam sak sbk sbn BM BN BK : Nat) (HAS_BIAS : Bool) (t : BlockState) :
    stepStmtsR R (tlaSeeds A B biasR N sam sak sbk sbn BM BN BK HAS_BIAS) t
      = stepStmts (tlaSeeds A B biasR N sam sak sbk sbn BM BN BK HAS_BIAS) t := by
  simp only [tlaSeeds, biasIfStmt, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def]
  rfl

set_option maxHeartbeats 1000000 in
/-- The 5-statement K-loop body is cast-free. -/
private theorem tlaLoopBody_castFree (R : RoundingModel) (BM BN BK sak sbk : Nat)
    (t : BlockState) :
    stepStmtsR R (fmaLoopBody BM BN BK sak sbk) t = stepStmts (fmaLoopBody BM BN BK sak sbk) t := by
  simp only [fmaLoopBody, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def]
  rfl

set_option maxHeartbeats 2000000 in
/-- The whole tail (save gate, the four activation gates, the store tail) is
cast-free — both stores are `.real`. -/
private theorem tlaTail_castFree (R : RoundingModel) (C ACT_INPUTS : RegionName)
    (M N smo sno saim sain BM BN : Nat) (SHOULD_SAVE_ACT_INPUTS : Bool)
    (ACTIVATION : String) (t : BlockState) :
    stepStmtsR R (saveIfStmt ACT_INPUTS saim sain BM BN SHOULD_SAVE_ACT_INPUTS
        :: (actStmts BM BN ACTIVATION ++ storeTail C M N smo sno BM BN)) t
      = stepStmts (saveIfStmt ACT_INPUTS saim sain BM BN SHOULD_SAVE_ACT_INPUTS
        :: (actStmts BM BN ACTIVATION ++ storeTail C M N smo sno BM BN)) t := by
  simp only [saveIfStmt, actStmts, storeTail, List.cons_append, List.nil_append,
    stepStmtsR, stepStmts, stepStmtR, stepStmt, evalOpR.eq_def, evalOp.eq_def]
  rfl

set_option maxHeartbeats 2000000 in
/-- The full surface sits inside the flat-memory bridge's covered fragment
(`FlattenOk`): no `ptrSub`, no pointer-dtype element loads. -/
private theorem tla_flattenOk (C ACT_INPUTS A B biasR : RegionName)
    (M N smo sno saim sain sam sak sbn sbk BM GM BN BK numKBlocks : Nat)
    (HAS_BIAS SHOULD_SAVE_ACT_INPUTS : Bool) (ACTIVATION : String) :
    ((kernel_fma_surface C ACT_INPUTS A B biasR M N smo sno saim sain
        sam sak sbn sbk BM GM BN BK numKBlocks
        HAS_BIAS SHOULD_SAVE_ACT_INPUTS ACTIVATION).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  rw [tla_body_split C ACT_INPUTS A B biasR M N smo sno saim sain
    sam sak sbn sbk BM GM BN BK numKBlocks HAS_BIAS SHOULD_SAVE_ACT_INPUTS ACTIVATION]
  simp [tlaScalars, tlaSeeds, biasIfStmt, fmaLoopBody, saveIfStmt, actStmts, storeTail,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

/-! ## The io signature

The kernel's grid is **1-D** (`tl.program_id(0)` only): `pid₁`/`pid₂` are
quantified but unused, exactly as in the `matmulKernelIO` precedent. The
schedule-derived tile coordinate is transcribed in pid form so the io windows
can be stated without a `BlockState`. -/

/-- Pid form of `rowIndex`: the `% M`-wrapped global row of tile lane `i`
(`(block_m_idx·BLOCK_M + i) % M`). Definitionally `rowIndex` at
`pid₀ = s.pids 0`. -/
def tlaRowIdx (p₀ M N BM BN GM i : Nat) : Nat := (blockMIdx p₀ M N BM BN GM * BM + i) % M

/-- Pid form of `colIndex`: the `% N`-wrapped global column of tile lane `j`. -/
def tlaColIdx (p₀ M N BM BN GM j : Nat) : Nat := (blockNIdx p₀ M N BM BN GM * BN + j) % N

/-- `rowIndex` **is** `tlaRowIdx` at `s.pids 0`. By `rfl`. -/
private theorem rowIndex_eq_tlaRowIdx (s : BlockState) (M N BM BN GM : Nat) (i : Fin BM) :
    rowIndex s M N BM BN GM i = tlaRowIdx (s.pids 0) M N BM BN GM i.val := rfl

/-- `colIndex` **is** `tlaColIdx` at `s.pids 0`. By `rfl`. -/
private theorem colIndex_eq_tlaColIdx (s : BlockState) (M N BM BN GM : Nat) (j : Fin BN) :
    colIndex s M N BM BN GM j = tlaColIdx (s.pids 0) M N BM BN GM j.val := rfl

/-- The `A`-stream lane feeding output lane `l` at inner key `e`: the row of
`l` (row-major over the `[BLOCK_M, BLOCK_N]` output tile) paired with `e`
over the `[BLOCK_M, BLOCK_K]` per-step `A`-tile. -/
def tlaALane (BM BN BK : Nat) (l : Fin (BM * BN)) (e : Fin BK) : Fin (BM * BK) :=
  Lane2D.encode ((Lane2D.decode l).1, e, PUnit.unit)

/-- The `B`-stream lane feeding output lane `l` at inner key `e`: `e` paired
with the column of `l` over the `[BLOCK_K, BLOCK_N]` per-step `B`-tile. -/
def tlaBLane (BM BN BK : Nat) (l : Fin (BM * BN)) (e : Fin BK) : Fin (BK * BN) :=
  Lane2D.encode (e, (Lane2D.decode l).2.1, PUnit.unit)

/-- The bias seed of output lane `l` read off the **degenerate static
stream** `zs` (the `bias` channel ignores `t`, so only step `0` is ever
consulted; `T = 0` has no step to read and the pre-loop load is unreachable
from the streams — the headline therefore assumes `0 < numKBlocks`). The two
guards transcribe `biasBase` verbatim: the `HAS_BIAS` constexpr gate and the
load's own `mask = n_offs < N, other = 0.0`. -/
noncomputable def tlaBiasIO (p₀ M N BM BN GM T : Nat) (HAS_BIAS : Bool)
    (zs : Fin T → Fin BN → ℝ) (l : Fin (BM * BN)) : ℝ :=
  if HAS_BIAS = Bool.true then
    (if tlaColIdx p₀ M N BM BN GM (Lane2D.decode l).2.1.val < N then
      (if h : 0 < T then zs ⟨0, h⟩ (Lane2D.decode l).2.1 else 0)
     else 0)
  else 0

/-- The **pre-activation** streaming spec of output lane `l`: the bias seed
plus the ideal-ℝ double fold `∑ t, ∑ e, A-tile[t](i,e) · B-tile[t](e,j)`
(`linearSpec` restated on the streams). -/
noncomputable def tlaLinIO (p₀ M N BM BN GM BK T : Nat) (HAS_BIAS : Bool)
    (xs : Fin T → Fin (BM * BK) → ℝ) (ys : Fin T → Fin (BK * BN) → ℝ)
    (zs : Fin T → Fin BN → ℝ) (l : Fin (BM * BN)) : ℝ :=
  tlaBiasIO p₀ M N BM BN GM T HAS_BIAS zs l
    + ∑ t : Fin T, ∑ e : Fin BK,
        xs t (tlaALane BM BN BK l e) * ys t (tlaBLane BM BN BK l e)

/-- **Streaming IO signature** of `kernel_fma` on the three-stream,
two-terminal-store fold skin.

Channels: `inp1 = A` and `inp2 = B` are the genuine per-step streams (step
`t` reads the `[BLOCK_M, BLOCK_K]` `A`-tile and the `[BLOCK_K, BLOCK_N]`
`B`-tile at the invariant's advanced pointers); `inp3 = bias` is the genre's
**degenerate static stream** — its window ignores `t` and reads the
`[BLOCK_N]` bias row at the wrapped column indices. Outputs: `out1 = C`
(activated) and `out2 = ACT_INPUTS` (pre-activation), both `[BLOCK_M ·
BLOCK_N]` lanes on the `.real` grid — the port models no narrow-float store
(both statements are `Stmt.store .real`; Python's `.to(tl.float32)` on the
bias load and the host's fp16 output dtype are not modeled), so declaring
anything but `.real` here would be false against this surface.

Masks: the two GEMM streams are unmasked (`K_LOAD_MASK_NEEDED = True` arm);
`mask3` carries **both** of the bias load's guards — the `HAS_BIAS` constexpr
gate and the load's own `n_offs < N` mask. `writeMask1 = True` (under `pre`
the `c_ptr_mask` is all-true); `writeMask2` carries the
`SHOULD_SAVE_ACT_INPUTS` constexpr gate, so the `ACT_INPUTS` claim is vacuous
when the flag is off.

`pre` is the port's **trusted-launch boundary**: this program's tile fits the
output (`hFitM`/`hFitN`). It cannot be folded into `writeMask1` — a
non-fitting program still *writes*, at `%`-wrapped addresses, so the skin's
frame clause ("every cell outside the write-active window is untouched")
would be false. -/
noncomputable def tlaIO (C ACT_INPUTS A B biasR : RegionName)
    (M N smo sno saim sain sam sak sbn sbk BM GM BN BK numKBlocks : Nat)
    (HAS_BIAS SHOULD_SAVE_ACT_INPUTS : Bool) (ACTIVATION : String) :
    StreamMasked3DKernelIO₃ₓ₂ where
  kernel := kernel_fma_surface C ACT_INPUTS A B biasR M N smo sno saim sain
    sam sak sbn sbk BM GM BN BK numKBlocks HAS_BIAS SHOULD_SAVE_ACT_INPUTS ACTIVATION
  inp1 := A
  inp2 := B
  inp3 := biasR
  out1 := C
  out2 := ACT_INPUTS
  T := numKBlocks
  B1 := BM * BK
  B2 := BK * BN
  B3 := BN
  C1 := BM * BN
  C2 := BM * BN
  out1DType := .real
  out2DType := .real
  pre := fun p₀ _ _ =>
    blockMIdx p₀ M N BM BN GM * BM + BM ≤ M ∧ blockNIdx p₀ M N BM BN GM * BN + BN ≤ N
  read1 := fun p₀ _ _ t l => tlaRowIdx p₀ M N BM BN GM (l.val / BK) * sam + l.val % BK * sak + t.val * BK * sak
  read2 := fun p₀ _ _ t l => l.val / BN * sbk + tlaColIdx p₀ M N BM BN GM (l.val % BN) * sbn + t.val * BK * sbk
  read3 := fun p₀ _ _ _ j => tlaColIdx p₀ M N BM BN GM j.val
  write1 := fun p₀ _ _ l => tlaRowIdx p₀ M N BM BN GM (l.val / BN) * smo + tlaColIdx p₀ M N BM BN GM (l.val % BN) * sno
  write2 := fun p₀ _ _ l => tlaRowIdx p₀ M N BM BN GM (l.val / BN) * saim + tlaColIdx p₀ M N BM BN GM (l.val % BN) * sain
  mask1 := fun _ _ _ _ _ => True
  mask2 := fun _ _ _ _ _ => True
  mask3 := fun p₀ _ _ _ j => HAS_BIAS = Bool.true ∧ tlaColIdx p₀ M N BM BN GM j.val < N
  writeMask1 := fun _ _ _ _ => True
  writeMask2 := fun _ _ _ _ => SHOULD_SAVE_ACT_INPUTS = Bool.true

/-! ## Spec bridge: the exact `linearSpec` **is** the streaming `tlaLinIO` -/

/-- Under the three stream pins, the exact closed form `linearSpec` at the
decoded output lane is the skin-level `tlaLinIO`: the bias seed comes off the
static `bias` stream at step `0` and the `gemmSum` contraction is the double
fold over the streamed tiles (`gemmSum_blocks` + address identity of the io
windows with the invariant's pointer shapes). -/
private theorem tlaLinIO_eq (A B biasR : RegionName) (s₀ : BlockState)
    (M N BM BN GM sam sak sbk sbn BK T : Nat) (HAS_BIAS : Bool) (hT : 0 < T)
    (xs : Fin T → Fin (BM * BK) → ℝ) (ys : Fin T → Fin (BK * BN) → ℝ)
    (zs : Fin T → Fin BN → ℝ)
    (hx : ∀ (t : Fin T) (l : Fin (BM * BK)),
      s₀.readMem A (tlaRowIdx (s₀.pids 0) M N BM BN GM (l.val / BK) * sam
          + l.val % BK * sak + t.val * BK * sak) = xs t l)
    (hy : ∀ (t : Fin T) (l : Fin (BK * BN)),
      s₀.readMem B (l.val / BN * sbk
          + tlaColIdx (s₀.pids 0) M N BM BN GM (l.val % BN) * sbn + t.val * BK * sbk) = ys t l)
    (hz : ∀ (t : Fin T) (j : Fin BN), HAS_BIAS = Bool.true →
      tlaColIdx (s₀.pids 0) M N BM BN GM j.val < N →
      s₀.readMem biasR (tlaColIdx (s₀.pids 0) M N BM BN GM j.val) = zs t j)
    (l : Fin (BM * BN)) :
    linearSpec s₀ A B biasR M N BM BN GM sam sak sbk sbn BK T HAS_BIAS
        (Lane2D.decode l).1 (Lane2D.decode l).2.1
      = tlaLinIO (s₀.pids 0) M N BM BN GM BK T HAS_BIAS xs ys zs l := by
  unfold linearSpec tlaLinIO
  congr 1
  · -- the bias seed
    unfold biasBase tlaBiasIO
    cases hb : HAS_BIAS
    · simp
    · rw [if_pos rfl, if_pos rfl, colIndex_eq_tlaColIdx]
      by_cases hc : tlaColIdx (s₀.pids 0) M N BM BN GM (Lane2D.decode l).2.1.val < N
      · rw [if_pos hc, if_pos hc, dif_pos hT]
        exact hz ⟨0, hT⟩ (Lane2D.decode l).2.1 hb hc
      · rw [if_neg hc, if_neg hc]
  · -- the GEMM contraction
    rw [Nat.mul_comm BK T, gemmSum_blocks]
    refine Finset.sum_congr rfl fun t _ => Finset.sum_congr rfl fun e _ => ?_
    have hxa : aElem s₀ A M N BM BN GM sam sak (Lane2D.decode l).1 (t.val * BK + e.val)
        = xs t (tlaALane BM BN BK l e) := by
      rw [← hx t (tlaALane BM BN BK l e)]
      unfold aElem
      simp only [tlaALane, Lane2D.encode_div, Lane2D.encode_mod]
      rw [show rowIndex s₀ M N BM BN GM (Lane2D.decode l).1 * sam + (t.val * BK + e.val) * sak
            = tlaRowIdx (s₀.pids 0) M N BM BN GM (Lane2D.decode l).1.val * sam
              + e.val * sak + t.val * BK * sak from by
        rw [rowIndex_eq_tlaRowIdx]; ring]
    have hyb : bElem s₀ B M N BM BN GM sbk sbn (Lane2D.decode l).2.1 (t.val * BK + e.val)
        = ys t (tlaBLane BM BN BK l e) := by
      rw [← hy t (tlaBLane BM BN BK l e)]
      unfold bElem
      simp only [tlaBLane, Lane2D.encode_div, Lane2D.encode_mod]
      rw [show (t.val * BK + e.val) * sbk
              + colIndex s₀ M N BM BN GM (Lane2D.decode l).2.1 * sbn
            = e.val * sbk + tlaColIdx (s₀.pids 0) M N BM BN GM (Lane2D.decode l).2.1.val * sbn
              + t.val * BK * sbk from by
        rw [colIndex_eq_tlaColIdx]; ring]
    rw [hxa, hyb]

/-- The io's `out1` window **is** `cOffset` at the decoded lane. By `rfl`. -/
private theorem tla_write1_eq (s₀ : BlockState) (M N BM BN GM smo sno : Nat)
    (l : Fin (BM * BN)) :
    tlaRowIdx (s₀.pids 0) M N BM BN GM (l.val / BN) * smo
        + tlaColIdx (s₀.pids 0) M N BM BN GM (l.val % BN) * sno
      = cOffset s₀ M N BM BN GM smo sno (Lane2D.decode l) := rfl

/-- The io's `out2` window **is** `actOffset` at the decoded lane. By `rfl`. -/
private theorem tla_write2_eq (s₀ : BlockState) (M N BM BN GM saim sain : Nat)
    (l : Fin (BM * BN)) :
    tlaRowIdx (s₀.pids 0) M N BM BN GM (l.val / BN) * saim
        + tlaColIdx (s₀.pids 0) M N BM BN GM (l.val % BN) * sain
      = actOffset s₀ M N BM BN GM saim sain (Lane2D.decode l) := rfl

/-! ## The missing per-lane frame

The exact stack above proves the two readbacks but only an *other-region*
preservation statement; the skin's Hoare triple additionally demands a
**cell-level frame** — every flat cell outside the two write windows is
untouched. These two scatter-frame lemmas and the tail re-walk supply it. -/

/-- Cell-level frame for an unmasked `writeMem` scatter (the `ACT_INPUTS`
store). -/
private theorem tla_foldl_frame {α : Type} {region : RegionName}
    (offFn : α → Nat) (valFn : α → ℝ) :
    ∀ (l : List α) (s : BlockState) (r : RegionName) (o : Nat),
      (r = region → ∀ k ∈ l, offFn k ≠ o) →
      ((l.foldl (fun acc k => acc.writeMem region (offFn k) (valFn k)) s).mem r o = s.mem r o)
  | [], _, _, _, _ => rfl
  | k :: rest, s, r, o, h => by
      rw [List.foldl_cons,
        tla_foldl_frame offFn valFn rest _ r o
          (fun hr k' hk' => h hr k' (List.mem_cons_of_mem _ hk')),
        BlockState.writeMem_mem,
        if_neg (fun hc => h hc.1 k List.mem_cons_self hc.2.symm)]

/-- Cell-level frame for a Bool-masked `writeMem` scatter (the `C` store). -/
private theorem tla_foldl_frame_masked {α : Type} {region : RegionName}
    (offFn : α → Nat) (valFn : α → ℝ) (mk : α → Bool) :
    ∀ (l : List α) (s : BlockState) (r : RegionName) (o : Nat),
      (r = region → ∀ k ∈ l, offFn k ≠ o) →
      ((l.foldl (fun acc k => if mk k then acc.writeMem region (offFn k) (valFn k) else acc)
          s).mem r o = s.mem r o)
  | [], _, _, _, _ => rfl
  | k :: rest, s, r, o, h => by
      rw [List.foldl_cons,
        tla_foldl_frame_masked offFn valFn mk rest _ r o
          (fun hr k' hk' => h hr k' (List.mem_cons_of_mem _ hk'))]
      by_cases hp : mk k = Bool.true
      · rw [if_pos hp, BlockState.writeMem_mem,
          if_neg (fun hc => h hc.1 k List.mem_cons_self hc.2.symm)]
      · rw [if_neg hp]

set_option maxHeartbeats 1000000 in
/-- **Save-gate frame**: the `SHOULD_SAVE_ACT_INPUTS` gate touches only the
`ACT_INPUTS` cells of its own window (and nothing at all when the flag is
off); the registers the tail still needs survive. -/
private theorem tla_saveIf_frame (ACT_INPUTS : RegionName) (s0 : BlockState)
    (M N BM BN GM saim sain : Nat) (SAVE : Bool)
    (v : TileIndex [BM, BN] → ℝ) (st : BlockState)
    (hm : st.regs .nat [BM] "m_offs"
      = some (Tile.vec (fun i : Fin BM => rowIndex s0 M N BM BN GM i)))
    (hn : st.regs .nat [BN] "n_offs"
      = some (Tile.vec (fun j : Fin BN => colIndex s0 M N BM BN GM j)))
    (hacc : st.regs .real [BM, BN] "acc" = some ⟨fun idx => some (v idx)⟩) :
    ∃ st', stepStmt (saveIfStmt ACT_INPUTS saim sain BM BN SAVE) st = some st'
      ∧ (∀ {dt} {sh} (nm : RegName), nm ≠ "act_in_ptrs" → st'.regs dt sh nm = st.regs dt sh nm)
      ∧ (∀ r o, (r = ACT_INPUTS → SAVE = Bool.true →
            ∀ idx : TileIndex [BM, BN], o ≠ actOffset s0 M N BM BN GM saim sain idx) →
          st'.mem r o = st.mem r o) := by
  unfold saveIfStmt
  rw [stepStmt_ifThen_constBool]
  cases SAVE
  · exact ⟨st, rfl, fun _ _ => rfl, fun _ _ _ => rfl⟩
  · rw [if_pos rfl]
    set apT : Tile .ptr [BM, BN] :=
      ⟨fun idx : TileIndex [BM, BN] =>
        (ACT_INPUTS.cast, actOffset s0 M N BM BN GM saim sain idx)⟩ with hapT
    have hstepP : stepStmt (Stmt.assign .ptr [BM, BN] "act_in_ptrs"
        (Op.ptrAdd Broadcast.scalarL (Op.ptrBase ACT_INPUTS)
          (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
            (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "m_offs")) (Op.constNat saim))
            (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "n_offs")) (Op.constNat sain))))) st
        = some (st.setReg "act_in_ptrs" .ptr [BM, BN] apT) :=
      stepStmt_assign_eq_some (pairptrs_eval st ACT_INPUTS BM BN saim sain "m_offs" "n_offs"
        (fun i => rowIndex s0 M N BM BN GM i) (fun j => colIndex s0 M N BM BN GM j) hm hn)
    rw [stepStmts.cons_some hstepP]
    set stP := st.setReg "act_in_ptrs" .ptr [BM, BN] apT with hstP
    set accT : Tile .real [BM, BN] := ⟨fun idx => some (v idx)⟩ with haccT
    have hstore : stepStmt (Stmt.store .real [BM, BN]
          (.ptr (Op.ref .ptr [BM, BN] "act_in_ptrs")) (Op.ref .real [BM, BN] "acc") .none) stP
        = some ((TileShape.allIndices [BM, BN]).foldl
            (fun acc i => acc.writeMemTyped .real ACT_INPUTS
              (actOffset s0 M N BM BN GM saim sain i) (accT.data i)) stP) := by
      simp only [stepStmt]
      rw [show evalOp (Op.ref .real [BM, BN] "acc") stP = some accT from by
            rw [evalOp_ref]; simp [hstP, hacc, haccT]]
      rw [show evalOp (Op.ref .ptr [BM, BN] "act_in_ptrs") stP = some apT from by
            rw [evalOp_ref]; simp [hstP]]
      simp only [bind, Option.bind_some]
      refine congrArg some (List.foldl_ext _ _ _ (fun acc i _ => ?_))
      simp only [hapT, if_true, Region.cast_id]
    rw [stepStmts.cons_some hstore, stepStmts.nil]
    refine ⟨_, rfl, ?_, ?_⟩
    · intro dt sh nm hne
      simp only [BlockState.writeMemTyped_real, BlockState.foldl_writeMem_regs, hstP,
        BlockState.setReg_ne_name, hne, ne_eq, not_false_eq_true]
    · intro r o hguard
      rw [show ((TileShape.allIndices [BM, BN]).foldl
            (fun acc i => acc.writeMemTyped .real ACT_INPUTS
              (actOffset s0 M N BM BN GM saim sain i) (accT.data i)) stP).mem r o
          = stP.mem r o from by
        simp only [BlockState.writeMemTyped_real]
        exact tla_foldl_frame (region := ACT_INPUTS)
          (fun i : TileIndex [BM, BN] => actOffset s0 M N BM BN GM saim sain i)
          (fun i => FloatDType.real.storeValue (accT.data i))
          (TileShape.allIndices [BM, BN]) stP r o
          (fun hr idx _ => fun hc => hguard hr rfl idx hc.symm)]
      simp only [hstP, BlockState.setReg_mem]

set_option maxHeartbeats 2000000 in
/-- **Store-tail frame**: the masked `C` store touches only cells of its own
`cOffset` window. -/
private theorem tla_storeTail_frame (C : RegionName) (s0 : BlockState)
    (M N smo sno BM BN GM : Nat) (w : TileIndex [BM, BN] → ℝ) (st : BlockState)
    (hm : st.regs .nat [BM] "m_offs"
      = some (Tile.vec (fun i : Fin BM => rowIndex s0 M N BM BN GM i)))
    (hn : st.regs .nat [BN] "n_offs"
      = some (Tile.vec (fun j : Fin BN => colIndex s0 M N BM BN GM j)))
    (hacc : st.regs .real [BM, BN] "acc" = some ⟨fun idx => some (w idx)⟩) :
    ∃ sfin, stepStmts (storeTail C M N smo sno BM BN) st = some sfin
      ∧ ∀ r o,
          (r = C → ∀ idx : TileIndex [BM, BN], o ≠ cOffset s0 M N BM BN GM smo sno idx) →
          sfin.mem r o = st.mem r o := by
  set cpT : Tile .ptr [BM, BN] :=
    ⟨fun idx : TileIndex [BM, BN] => (C.cast, cOffset s0 M N BM BN GM smo sno idx)⟩ with hcpT
  have hstepC : stepStmt (Stmt.assign .ptr [BM, BN] "C"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase C)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "m_offs")) (Op.constNat smo))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "n_offs")) (Op.constNat sno))))) st
      = some (st.setReg "C" .ptr [BM, BN] cpT) :=
    stepStmt_assign_eq_some (pairptrs_eval st C BM BN smo sno "m_offs" "n_offs"
      (fun i => rowIndex s0 M N BM BN GM i) (fun j => colIndex s0 M N BM BN GM j) hm hn)
  unfold storeTail
  rw [stepStmts.cons_some hstepC]
  set s23 := st.setReg "C" .ptr [BM, BN] cpT with hs23
  set cmT : Tile .bool [BM, BN] :=
    ⟨fun idx : TileIndex [BM, BN] =>
      (decide (rowIndex s0 M N BM BN GM idx.1 < M) && decide (colIndex s0 M N BM BN GM idx.2.1 < N))⟩
    with hcmT
  have hstepMask : stepStmt (Stmt.assign .bool [BM, BN] "c_ptr_mask"
      (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.expandDim ⟨1, by simp⟩
          (Op.lt ComparableDType.nat Broadcast.scalarR (Op.ref .nat [BM] "m_offs") (Op.constNat M)))
        (Op.expandDim ⟨0, by simp⟩
          (Op.lt ComparableDType.nat Broadcast.scalarR (Op.ref .nat [BN] "n_offs") (Op.constNat N))))) s23
      = some (s23.setReg "c_ptr_mask" .bool [BM, BN] cmT) :=
    stepStmt_assign_eq_some (cmask_eval s23 M N BM BN
      (fun i => rowIndex s0 M N BM BN GM i) (fun j => colIndex s0 M N BM BN GM j)
      (by simp [hs23, hm]) (by simp [hs23, hn]))
  rw [stepStmts.cons_some hstepMask]
  set s24 := s23.setReg "c_ptr_mask" .bool [BM, BN] cmT with hs24
  set accT : Tile .real [BM, BN] := ⟨fun idx => some (w idx)⟩ with haccT
  have hstore : stepStmt (Stmt.store .real [BM, BN] (.ptr (Op.ref .ptr [BM, BN] "C"))
        (Op.ref .real [BM, BN] "acc") (.mask (Op.ref .bool [BM, BN] "c_ptr_mask"))) s24
      = some ((TileShape.allIndices [BM, BN]).foldl
          (fun acc i =>
            if cmT.data i then acc.writeMem C (cOffset s0 M N BM BN GM smo sno i) (w i)
            else acc) s24) := by
    simp only [stepStmt]
    rw [show evalOp (Op.ref .real [BM, BN] "acc") s24 = some accT from by
          rw [evalOp_ref]; simp [hs24, hs23, hacc, haccT]]
    rw [show evalOp (Op.ref .ptr [BM, BN] "C") s24 = some cpT from by
          rw [evalOp_ref]; simp [hs24, hs23]]
    rw [show evalOp (Op.ref .bool [BM, BN] "c_ptr_mask") s24 = some cmT from by
          rw [evalOp_ref]; simp [hs24]]
    simp only [bind, Option.bind_some]
    refine congrArg some (List.foldl_ext _ _ _ (fun acc i _ => ?_))
    by_cases hmask : cmT.data i
    · simp only [hmask, if_true, hcpT, haccT, Region.cast_id, BlockState.writeMemTyped_real,
        FloatDType.real_storeValue]
      rfl
    · simp only [hmask, Bool.false_eq_true, if_false]
  rw [stepStmts.cons_some hstore, stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  intro r o hguard
  rw [tla_foldl_frame_masked (region := C)
    (fun i : TileIndex [BM, BN] => cOffset s0 M N BM BN GM smo sno i)
    (fun i : TileIndex [BM, BN] => w i) (fun i : TileIndex [BM, BN] => cmT.data i)
    (TileShape.allIndices [BM, BN]) s24 r o
    (fun hr idx _ => fun hc => hguard hr idx hc.symm)]
  simp only [hs24, hs23, BlockState.setReg_mem]

set_option maxHeartbeats 2000000 in
/-- **Whole-tail frame**: the save gate, the four activation gates and the
masked `C` store together leave every cell outside the two output windows
untouched. -/
private theorem tla_tail_frame (C ACT_INPUTS : RegionName) (s0 : BlockState)
    (M N smo sno saim sain BM BN GM : Nat) (SAVE : Bool) (ACTIVATION : String)
    (v : TileIndex [BM, BN] → ℝ) (st : BlockState)
    (hm : st.regs .nat [BM] "m_offs"
      = some (Tile.vec (fun i : Fin BM => rowIndex s0 M N BM BN GM i)))
    (hn : st.regs .nat [BN] "n_offs"
      = some (Tile.vec (fun j : Fin BN => colIndex s0 M N BM BN GM j)))
    (hacc : st.regs .real [BM, BN] "acc" = some ⟨fun idx => some (v idx)⟩) :
    ∃ sfin, stepStmts (saveIfStmt ACT_INPUTS saim sain BM BN SAVE
        :: (actStmts BM BN ACTIVATION ++ storeTail C M N smo sno BM BN)) st = some sfin
      ∧ ∀ r o,
          (r = ACT_INPUTS → SAVE = Bool.true →
            ∀ idx : TileIndex [BM, BN], o ≠ actOffset s0 M N BM BN GM saim sain idx) →
          (r = C → ∀ idx : TileIndex [BM, BN], o ≠ cOffset s0 M N BM BN GM smo sno idx) →
          sfin.mem r o = st.mem r o := by
  obtain ⟨s18, hsave, hpres18, hframe18⟩ :=
    tla_saveIf_frame ACT_INPUTS s0 M N BM BN GM saim sain SAVE v st hm hn hacc
  rw [stepStmts.cons_some hsave]
  have hm18 : s18.regs .nat [BM] "m_offs"
      = some (Tile.vec (fun i : Fin BM => rowIndex s0 M N BM BN GM i)) := by
    rw [hpres18 "m_offs" (by decide)]; exact hm
  have hn18 : s18.regs .nat [BN] "n_offs"
      = some (Tile.vec (fun j : Fin BN => colIndex s0 M N BM BN GM j)) := by
    rw [hpres18 "n_offs" (by decide)]; exact hn
  have hacc18 : s18.regs .real [BM, BN] "acc" = some ⟨fun idx => some (v idx)⟩ := by
    rw [hpres18 "acc" (by decide)]; exact hacc
  obtain ⟨s22, hact, hacc22, hpres22, hmem22, _⟩ := actTail_steps s18 BM BN ACTIVATION v hacc18
  rw [stepStmts.append_some hact]
  have hm22 : s22.regs .nat [BM] "m_offs"
      = some (Tile.vec (fun i : Fin BM => rowIndex s0 M N BM BN GM i)) := by
    rw [hpres22 "m_offs" (by decide)]; exact hm18
  have hn22 : s22.regs .nat [BN] "n_offs"
      = some (Tile.vec (fun j : Fin BN => colIndex s0 M N BM BN GM j)) := by
    rw [hpres22 "n_offs" (by decide)]; exact hn18
  obtain ⟨sfin, hstore, hframeS⟩ :=
    tla_storeTail_frame C s0 M N smo sno BM BN GM
      (fun idx => applyActivation ACTIVATION (v idx)) s22 hm22 hn22 hacc22
  refine ⟨sfin, hstore, ?_⟩
  intro r o g1 g2
  rw [hframeS r o g2, show s22.mem r o = s18.mem r o from by rw [hmem22], hframe18 r o g1]

/-! ## The `TraceSafeR R` safety walk

Every memory event of the kernel — the two unmasked GEMM loads, the masked
bias-row load, the unmasked `ACT_INPUTS` store and the masked `C` store — is
shown in bounds along the actual `stepStmtR R` execution. All five bound
groups are the io's own windows. -/

@[simp] private theorem evalOpR_constBool (R : RoundingModel) (b : Bool) (s : BlockState) :
    evalOpR R (Op.constBool b) s = some (Tile.scalar b) := by
  simp [evalOpR]

/-- A constexpr `ifThen` whose body is register-only is trace-safe. -/
private theorem tla_ifThen_regOnly_safeR (R : RoundingModel) (bounds : RegionBounds)
    (b : Bool) (body : List Stmt)
    (hb : ∀ st ∈ body, ∀ u, Stmt.TraceSafeR R bounds st u) (s : BlockState) :
    Stmt.TraceSafeR R bounds (Stmt.ifThen (Op.constBool b) body) s := by
  rw [Stmt.TraceSafeR]
  refine ⟨by simp [Op.SafeAtR], ?_⟩
  rw [evalOpR_constBool]
  simp only [Tile.scalar_data]
  cases b
  · simp
  · simpa using Stmt.TraceSafeListR.of_forall body s hb

/-- The row×column pointer chunk evaluates under `R` exactly as it does
exactly (cast-free). -/
private theorem pairptrs_evalR (R : RoundingModel) (s : BlockState) (Rg : RegionName)
    (X Y sx sy : Nat) (nx ny : RegName) (gx : Fin X → Nat) (gy : Fin Y → Nat)
    (hx : s.regs .nat [X] nx = some (Tile.vec gx))
    (hy : s.regs .nat [Y] ny = some (Tile.vec gy)) :
    evalOpR R (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Rg)
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [X] nx)) (Op.constNat sx))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [Y] ny)) (Op.constNat sy)))) s
      = some (⟨fun idx : TileIndex [X, Y] => (Rg.cast, gx idx.1 * sx + gy idx.2.1 * sy)⟩ : Tile .ptr [X, Y]) := by
  rw [show evalOpR R (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Rg)
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [X] nx)) (Op.constNat sx))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [Y] ny)) (Op.constNat sy)))) s
      = evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Rg)
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [X] nx)) (Op.constNat sx))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [Y] ny)) (Op.constNat sy)))) s
    from by simp only [evalOpR.eq_def, evalOp.eq_def]]
  exact pairptrs_eval s Rg X Y sx sy nx ny gx gy hx hy

/-- The bias load's `n_offs < N` mask evaluates under `R` exactly as it does
exactly (cast-free). -/
private theorem tla_ltmask_castFreeR (R : RoundingModel) (N BN : Nat) (s : BlockState) :
    evalOpR R (Op.lt ComparableDType.nat Broadcast.scalarR
        (Op.ref .nat [BN] "n_offs") (Op.constNat N)) s
      = evalOp (Op.lt ComparableDType.nat Broadcast.scalarR
        (Op.ref .nat [BN] "n_offs") (Op.constNat N)) s := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

set_option maxHeartbeats 1000000 in
/-- **Per-iteration body safety**: the two unmasked GEMM loads read at the
invariant's pointer shapes, in bounds by the io's `read1`/`read2` windows
instantiated at step `c`; the other three statements are register-only. -/
private theorem tla_bodySafeR (R : RoundingModel) (bounds : RegionBounds)
    (A B : RegionName) (s0 : BlockState)
    (M N BM BN GM sam sak sbk sbn BK T : Nat)
    (c : Nat) (hc : c < T) (sk : BlockState)
    (hap : sk.regs .ptr [BM, BK] "A" = some ⟨fun idx : TileIndex [BM, BK] =>
        (A.cast, rowIndex s0 M N BM BN GM idx.1 * sam + idx.2.1.val * sak + c * BK * sak)⟩)
    (hbp : sk.regs .ptr [BK, BN] "B" = some ⟨fun idx : TileIndex [BK, BN] =>
        (B.cast, idx.1.val * sbk + colIndex s0 M N BM BN GM idx.2.1 * sbn + c * BK * sbk)⟩)
    (hbA : ∀ (t : Fin T) (l : Fin (BM * BK)),
      tlaRowIdx (s0.pids 0) M N BM BN GM (l.val / BK) * sam + l.val % BK * sak
        + t.val * BK * sak < bounds A)
    (hbB : ∀ (t : Fin T) (l : Fin (BK * BN)),
      l.val / BN * sbk + tlaColIdx (s0.pids 0) M N BM BN GM (l.val % BN) * sbn
        + t.val * BK * sbk < bounds B) :
    Stmt.TraceSafeListR R bounds (fmaLoopBody BM BN BK sak sbk) sk := by
  unfold fmaLoopBody
  refine Stmt.TraceSafeListR.cons_intro ?_ ?_
  · -- load a
    simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR, MemAccess.ActiveAddressSafeR,
      memAccessActiveAddressSafeR]
    refine ⟨trivial, trivial, ?_⟩
    intro ptrs hptrs i _
    rw [evalOpR_ref, hap] at hptrs
    obtain rfl := Option.some.inj hptrs
    show rowIndex s0 M N BM BN GM i.1 * sam + i.2.1.val * sak + c * BK * sak < bounds (Region.cast A)
    have h' := hbA ⟨c, hc⟩ (Lane2D.encode (i.1, i.2.1, PUnit.unit))
    rw [Lane2D.encode_div, Lane2D.encode_mod] at h'
    simpa only [Region.cast_id, rowIndex_eq_tlaRowIdx] using h'
  · intro s1 h1
    obtain ⟨v1, -, rfl⟩ := stepStmtR_assign_inv h1
    refine Stmt.TraceSafeListR.cons_intro ?_ ?_
    · -- load b
      simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR, MemAccess.ActiveAddressSafeR,
        memAccessActiveAddressSafeR]
      refine ⟨trivial, trivial, ?_⟩
      intro ptrs hptrs i _
      rw [evalOpR_ref] at hptrs
      rw [show (sk.setReg "a" .real [BM, BK] v1).regs .ptr [BK, BN] "B"
          = some (⟨fun idx : TileIndex [BK, BN] =>
            (B.cast, idx.1.val * sbk + colIndex s0 M N BM BN GM idx.2.1 * sbn
              + c * BK * sbk)⟩ : Tile .ptr [BK, BN]) from by simp [hbp]] at hptrs
      obtain rfl := Option.some.inj hptrs
      show i.1.val * sbk + colIndex s0 M N BM BN GM i.2.1 * sbn + c * BK * sbk
          < bounds (Region.cast B)
      have h' := hbB ⟨c, hc⟩ (Lane2D.encode (i.1, i.2.1, PUnit.unit))
      rw [Lane2D.encode_div, Lane2D.encode_mod] at h'
      simpa only [Region.cast_id, colIndex_eq_tlaColIdx] using h'
    · intro s2 h2
      obtain ⟨v2, -, rfl⟩ := stepStmtR_assign_inv h2
      refine Stmt.TraceSafeListR.of_forall _ _ ?_
      intro st hst s'
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hst
      rcases hst with rfl | rfl | rfl <;>
        simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]

set_option maxHeartbeats 1000000 in
/-- **`HAS_BIAS` gate safety**: when the gate is live, the masked bias-row
load's active lanes (`n_offs < N`) are in bounds by the io's `read3`/`mask3`
window; when it is off, nothing is read. -/
private theorem tla_biasIfSafeR (R : RoundingModel) (bounds : RegionBounds)
    (biasR : RegionName) (s0 : BlockState) (M N BM BN GM : Nat) (HAS_BIAS : Bool)
    (st : BlockState)
    (hn : st.regs .nat [BN] "n_offs"
      = some (Tile.vec (fun j : Fin BN => colIndex s0 M N BM BN GM j)))
    (hbBias : ∀ j : Fin BN, HAS_BIAS = Bool.true →
      tlaColIdx (s0.pids 0) M N BM BN GM j.val < N →
      tlaColIdx (s0.pids 0) M N BM BN GM j.val < bounds biasR) :
    Stmt.TraceSafeR R bounds (biasIfStmt biasR N BM BN HAS_BIAS) st := by
  unfold biasIfStmt
  cases hb : HAS_BIAS
  · rw [Stmt.TraceSafeR]
    exact ⟨by simp [Op.SafeAtR], by simp⟩
  · rw [Stmt.TraceSafeR]
    refine ⟨by simp [Op.SafeAtR], ?_⟩
    rw [evalOpR_constBool]
    simp only [Tile.scalar_data, if_true]
    refine Stmt.TraceSafeListR.cons_intro ?_ (fun s' _ => Stmt.TraceSafeListR.of_forall _ _ ?_)
    · -- the masked bias-row load
      simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR, MemAccess.ActiveAddressSafeR,
        memAccessActiveAddressSafeR]
      refine ⟨trivial, ⟨⟨trivial, trivial⟩, trivial⟩, ?_⟩
      intro offsets hoff j hact
      rw [evalOpR_ref, hn] at hoff
      obtain rfl := Option.some.inj hoff
      obtain ⟨masks, hmasks, hmj⟩ := hact
      rw [tla_ltmask_castFreeR, evalOp_lt] at hmasks
      simp only [evalOp_ref, hn, evalOp_constNat, Option.bind_eq_bind, Option.bind_some] at hmasks
      obtain rfl := Option.some.inj hmasks
      have hlt : colIndex s0 M N BM BN GM j.1 < N := by
        simp only [Tile.cop_data, Tile.vec_data, Tile.scalar_data, Broadcast.leftIndex,
          Broadcast.rightIndex, ComparableDType.lt, decide_eq_true_eq] at hmj
        exact hmj
      simp only [Tile.vec_data, Region.cast_id, colIndex_eq_tlaColIdx]
      exact hbBias j.1 hb (by rw [← colIndex_eq_tlaColIdx]; exact hlt)
    · -- `acc += bias[None, :]` is register-only
      intro stx hstx u
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hstx
      rcases hstx with rfl
      simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]

set_option maxHeartbeats 1000000 in
/-- **Save-gate safety**: when `SHOULD_SAVE_ACT_INPUTS` is live, the unmasked
`ACT_INPUTS` store's addresses are the io's `write2` window. -/
private theorem tla_saveIfSafeR (R : RoundingModel) (bounds : RegionBounds)
    (ACT_INPUTS : RegionName) (s0 : BlockState) (M N BM BN GM saim sain : Nat)
    (SAVE : Bool) (st : BlockState)
    (hm : st.regs .nat [BM] "m_offs"
      = some (Tile.vec (fun i : Fin BM => rowIndex s0 M N BM BN GM i)))
    (hn : st.regs .nat [BN] "n_offs"
      = some (Tile.vec (fun j : Fin BN => colIndex s0 M N BM BN GM j)))
    (hbAct : ∀ l : Fin (BM * BN), SAVE = Bool.true →
      tlaRowIdx (s0.pids 0) M N BM BN GM (l.val / BN) * saim
        + tlaColIdx (s0.pids 0) M N BM BN GM (l.val % BN) * sain < bounds ACT_INPUTS) :
    Stmt.TraceSafeR R bounds (saveIfStmt ACT_INPUTS saim sain BM BN SAVE) st := by
  unfold saveIfStmt
  cases hb : SAVE
  · rw [Stmt.TraceSafeR]
    exact ⟨by simp [Op.SafeAtR], by simp⟩
  · rw [Stmt.TraceSafeR]
    refine ⟨by simp [Op.SafeAtR], ?_⟩
    rw [evalOpR_constBool]
    simp only [Tile.scalar_data, if_true]
    refine Stmt.TraceSafeListR.cons_intro (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) ?_
    intro s1 h1
    obtain ⟨v1, hv1, rfl⟩ := stepStmtR_assign_inv h1
    rw [pairptrs_evalR R st ACT_INPUTS BM BN saim sain "m_offs" "n_offs"
      (fun i => rowIndex s0 M N BM BN GM i) (fun j => colIndex s0 M N BM BN GM j) hm hn] at hv1
    obtain rfl := Option.some.inj hv1
    refine Stmt.TraceSafeListR.cons_intro ?_ (fun s' _ => Stmt.TraceSafeListR.nil_intro)
    simp only [Stmt.TraceSafeR, MemAccess.SafeAtR, MaskOpt.SafeAtR, Op.SafeAtR.eq_def,
      MaskOpt.ActiveR, MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
    refine ⟨trivial, trivial, trivial, ?_⟩
    intro ptrs hptrs i _
    rw [evalOpR_ref] at hptrs
    simp only [BlockState.setReg_same] at hptrs
    obtain rfl := Option.some.inj hptrs
    show rowIndex s0 M N BM BN GM i.1 * saim + colIndex s0 M N BM BN GM i.2.1 * sain
        < bounds (Region.cast ACT_INPUTS)
    have h' := hbAct (Lane2D.encode (i.1, i.2.1, PUnit.unit)) hb
    rw [Lane2D.encode_div, Lane2D.encode_mod] at h'
    simpa only [Region.cast_id, rowIndex_eq_tlaRowIdx, colIndex_eq_tlaColIdx] using h'

set_option maxHeartbeats 1000000 in
/-- **Store-tail safety**: the masked `C` store's addresses are the io's
`write1` window (`writeMask1 = True`, so the bound is supplied at every lane
and the mask is not consulted). -/
private theorem tla_storeTailSafeR (R : RoundingModel) (bounds : RegionBounds)
    (C : RegionName) (s0 : BlockState) (M N smo sno BM BN GM : Nat) (st : BlockState)
    (hm : st.regs .nat [BM] "m_offs"
      = some (Tile.vec (fun i : Fin BM => rowIndex s0 M N BM BN GM i)))
    (hn : st.regs .nat [BN] "n_offs"
      = some (Tile.vec (fun j : Fin BN => colIndex s0 M N BM BN GM j)))
    (hbC : ∀ l : Fin (BM * BN),
      tlaRowIdx (s0.pids 0) M N BM BN GM (l.val / BN) * smo
        + tlaColIdx (s0.pids 0) M N BM BN GM (l.val % BN) * sno < bounds C) :
    Stmt.TraceSafeListR R bounds (storeTail C M N smo sno BM BN) st := by
  unfold storeTail
  refine Stmt.TraceSafeListR.cons_intro (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) ?_
  intro s1 h1
  obtain ⟨v1, hv1, rfl⟩ := stepStmtR_assign_inv h1
  rw [pairptrs_evalR R st C BM BN smo sno "m_offs" "n_offs"
    (fun i => rowIndex s0 M N BM BN GM i) (fun j => colIndex s0 M N BM BN GM j) hm hn] at hv1
  obtain rfl := Option.some.inj hv1
  refine Stmt.TraceSafeListR.cons_intro (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) ?_
  intro s2 h2
  obtain ⟨v2, -, rfl⟩ := stepStmtR_assign_inv h2
  refine Stmt.TraceSafeListR.cons_intro ?_ (fun s' _ => Stmt.TraceSafeListR.nil_intro)
  simp only [Stmt.TraceSafeR, MemAccess.SafeAtR, MaskOpt.SafeAtR, Op.SafeAtR.eq_def,
    MaskOpt.ActiveR, MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
  refine ⟨trivial, trivial, trivial, ?_⟩
  intro ptrs hptrs i _
  rw [evalOpR_ref] at hptrs
  simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq,
    not_false_eq_true, reduceCtorEq] at hptrs
  obtain rfl := Option.some.inj hptrs
  show rowIndex s0 M N BM BN GM i.1 * smo + colIndex s0 M N BM BN GM i.2.1 * sno
      < bounds (Region.cast C)
  have h' := hbC (Lane2D.encode (i.1, i.2.1, PUnit.unit))
  rw [Lane2D.encode_div, Lane2D.encode_mod] at h'
  simpa only [Region.cast_id, rowIndex_eq_tlaRowIdx, colIndex_eq_tlaColIdx] using h'

set_option maxHeartbeats 1000000 in
/-- The save gate alone is cast-free. -/
private theorem tla_saveIf_castFree (R : RoundingModel) (ACT_INPUTS : RegionName)
    (saim sain BM BN : Nat) (SAVE : Bool) (t : BlockState) :
    stepStmtR R (saveIfStmt ACT_INPUTS saim sain BM BN SAVE) t
      = stepStmt (saveIfStmt ACT_INPUTS saim sain BM BN SAVE) t := by
  simp only [saveIfStmt, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def]
  rfl

set_option maxHeartbeats 2000000 in
/-- The four activation gates are cast-free. -/
private theorem tlaActStmts_castFree (R : RoundingModel) (BM BN : Nat) (ACTIVATION : String)
    (t : BlockState) :
    stepStmtsR R (actStmts BM BN ACTIVATION) t = stepStmts (actStmts BM BN ACTIVATION) t := by
  simp only [actStmts, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def]
  rfl

set_option maxHeartbeats 4000000 in
/-- **The `TraceSafeR R` walk for the whole kernel.** The prologue's 13
schedule statements are register-only; the `A`/`B`/`acc` seeds are
register-only; the `HAS_BIAS` gate, the K-loop (driven by
`Stmt.forRangeTraceSafeR_inv` over the exact `fmaInvariant`), the save gate,
the four activation gates and the store tail are discharged by the lemmas
above. The five bound groups are exactly the io's five windows. -/
private theorem tla_traceSafeR (R : RoundingModel) (bounds : RegionBounds)
    (C ACT_INPUTS A B biasR : RegionName)
    (M N smo sno saim sain sam sak sbn sbk BM GM BN BK numKBlocks : Nat)
    (HAS_BIAS SAVE : Bool) (ACTIVATION : String) (s : BlockState)
    (hbA : ∀ (t : Fin numKBlocks) (l : Fin (BM * BK)),
      tlaRowIdx (s.pids 0) M N BM BN GM (l.val / BK) * sam + l.val % BK * sak
        + t.val * BK * sak < bounds A)
    (hbB : ∀ (t : Fin numKBlocks) (l : Fin (BK * BN)),
      l.val / BN * sbk + tlaColIdx (s.pids 0) M N BM BN GM (l.val % BN) * sbn
        + t.val * BK * sbk < bounds B)
    (hbBias : ∀ j : Fin BN, HAS_BIAS = Bool.true →
      tlaColIdx (s.pids 0) M N BM BN GM j.val < N →
      tlaColIdx (s.pids 0) M N BM BN GM j.val < bounds biasR)
    (hbC : ∀ l : Fin (BM * BN),
      tlaRowIdx (s.pids 0) M N BM BN GM (l.val / BN) * smo
        + tlaColIdx (s.pids 0) M N BM BN GM (l.val % BN) * sno < bounds C)
    (hbAct : ∀ l : Fin (BM * BN), SAVE = Bool.true →
      tlaRowIdx (s.pids 0) M N BM BN GM (l.val / BN) * saim
        + tlaColIdx (s.pids 0) M N BM BN GM (l.val % BN) * sain < bounds ACT_INPUTS) :
    ((kernel_fma_surface C ACT_INPUTS A B biasR M N smo sno saim sain
      sam sak sbn sbk BM GM BN BK numKBlocks
      HAS_BIAS SAVE ACTIVATION).toAlgKernel).TraceSafeR R bounds s := by
  unfold Kernel.TraceSafeR
  rw [tla_body_split C ACT_INPUTS A B biasR M N smo sno saim sain
      sam sak sbn sbk BM GM BN BK numKBlocks HAS_BIAS SAVE ACTIVATION,
    List.append_assoc]
  obtain ⟨s13, h13, hpids13, hm13, hn13, hk13, hmem13⟩ := preLoop_scalars s M N BM BN BK GM
  obtain ⟨sPre, hpre, hP0⟩ := preLoop C ACT_INPUTS A B biasR s M N smo sno saim sain
    sam sak sbn sbk BM GM BN BK numKBlocks HAS_BIAS SAVE ACTIVATION
  have h13' : stepStmts (tlaScalars M N BM BN BK GM) s = some s13 := h13
  have hseeds : stepStmts (tlaSeeds A B biasR N sam sak sbk sbn BM BN BK HAS_BIAS) s13
      = some sPre := by
    have hp := hpre
    rw [tla_take17 C ACT_INPUTS A B biasR M N smo sno saim sain
        sam sak sbn sbk BM GM BN BK numKBlocks HAS_BIAS SAVE ACTIVATION,
      stepStmts.append_some h13'] at hp
    exact hp
  refine Stmt.TraceSafeListR.append_intro _ s ?_ ?_
  · -- the 13 schedule statements are register-only
    refine Stmt.TraceSafeListR.of_forall _ _ ?_
    intro st hst u
    simp only [tlaScalars, List.mem_cons, List.not_mem_nil, or_false] at hst
    rcases hst with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
      simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]
  intro sx hsx
  rw [tlaScalars_castFree R M N BM BN BK GM s, h13'] at hsx
  obtain rfl := Option.some.inj hsx
  refine Stmt.TraceSafeListR.append_intro _ _ ?_ ?_
  · -- the 4 seed statements
    unfold tlaSeeds
    refine Stmt.TraceSafeListR.cons_intro (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) ?_
    intro sA hsA
    obtain ⟨vA, -, rfl⟩ := stepStmtR_assign_inv hsA
    refine Stmt.TraceSafeListR.cons_intro (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) ?_
    intro sB hsB
    obtain ⟨vB, -, rfl⟩ := stepStmtR_assign_inv hsB
    refine Stmt.TraceSafeListR.cons_intro (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) ?_
    intro sAcc hsAcc
    obtain ⟨vAcc, -, rfl⟩ := stepStmtR_assign_inv hsAcc
    refine Stmt.TraceSafeListR.cons_intro ?_ (fun _ _ => Stmt.TraceSafeListR.nil_intro)
    exact tla_biasIfSafeR R bounds biasR s M N BM BN GM HAS_BIAS _ (by simp [hn13]) hbBias
  intro sy hsy
  rw [tlaSeeds_castFree R A B biasR N sam sak sbk sbn BM BN BK HAS_BIAS s13, hseeds] at hsy
  obtain rfl := Option.some.inj hsy
  refine Stmt.TraceSafeListR.cons_intro ?_ ?_
  · -- the K-loop
    simp only [Stmt.TraceSafeR]
    refine Stmt.forRangeTraceSafeR_inv R bounds "k" numKBlocks 1 (fmaLoopBody BM BN BK sak sbk)
      (fmaInvariant A B biasR s M N BM BN GM sam sak sbk sbn BK numKBlocks HAS_BIAS) ?_ 0 _ hP0
    intro c st hc hP
    have hP' := hP
    simp only [fmaInvariant] at hP'
    obtain ⟨-, -, -, -, hap, hbp, -⟩ := hP'
    refine ⟨tla_bodySafeR R bounds A B s M N BM BN GM sam sak sbk sbn BK numKBlocks c hc
      (st.setReg "k" .nat [] (Tile.scalar c)) (by simp [hap]) (by simp [hbp]) hbA hbB, ?_⟩
    obtain ⟨s'', hs'', hP''⟩ := kernel_fma_step A B biasR s M N BM BN GM sam sak sbk sbn
      BK numKBlocks HAS_BIAS c st hc hP
    exact ⟨s'', by rw [tlaLoopBody_castFree]; exact hs'', by simpa using hP''⟩
  · -- the tail
    intro sLoop hsLoop
    obtain ⟨final, sL, hLoopStmt, hfinal, hPL⟩ :=
      forRange_inv (idx := "k") (start := 0) (stop := numKBlocks) (step := 1)
        (body := fmaLoopBody BM BN BK sak sbk)
        (by omega) hP0
        (fun c st hlt hinv => by
          have := kernel_fma_step A B biasR s M N BM BN GM sam sak sbk sbn BK numKBlocks
            HAS_BIAS c st hlt hinv
          simpa using this)
    rw [stepStmtR_forRange,
      stepForRangeAuxR_castFree R _ (tlaLoopBody_castFree R BM BN BK sak sbk) "k",
      ← stepForRangeAux.forRange_unfold, hLoopStmt] at hsLoop
    obtain rfl := Option.some.inj hsLoop
    have hfinalEq : final = numKBlocks := by
      have hle : final ≤ numKBlocks := by
        simp only [fmaInvariant] at hPL
        exact hPL.1
      exact le_antisymm hle hfinal
    rw [hfinalEq] at hPL
    have hPL' := hPL
    simp only [fmaInvariant] at hPL'
    obtain ⟨-, hmL, hnL, hzL, -, -, -⟩ := hPL'
    refine Stmt.TraceSafeListR.cons_intro
      (tla_saveIfSafeR R bounds ACT_INPUTS s M N BM BN GM saim sain SAVE _ hmL hnL hbAct) ?_
    intro s18 h18
    obtain ⟨s18', hsave', hpres18, -⟩ :=
      tla_saveIf_frame ACT_INPUTS s M N BM BN GM saim sain SAVE
        (fun idx => accPartial s A B biasR M N BM BN GM sam sak sbk sbn BK HAS_BIAS
          idx.1 idx.2.1 numKBlocks) sL hmL hnL hzL
    rw [tla_saveIf_castFree, hsave'] at h18
    obtain rfl := Option.some.inj h18
    have hm18 := (hpres18 (dt := .nat) (sh := [BM]) "m_offs" (by decide)).trans hmL
    have hn18 := (hpres18 (dt := .nat) (sh := [BN]) "n_offs" (by decide)).trans hnL
    have hacc18 := (hpres18 (dt := .real) (sh := [BM, BN]) "acc" (by decide)).trans hzL
    refine Stmt.TraceSafeListR.append_intro _ _ ?_ ?_
    · -- the four activation gates: register-only bodies
      refine Stmt.TraceSafeListR.of_forall _ _ ?_
      intro st hst u
      simp only [actStmts, List.mem_cons, List.not_mem_nil, or_false] at hst
      rcases hst with rfl | rfl | rfl | rfl <;>
        exact tla_ifThen_regOnly_safeR R bounds _ _ (fun st' hst' u' => by
          simp only [List.mem_cons, List.not_mem_nil, or_false] at hst'
          rcases hst' with rfl
          simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) u
    · intro s22 h22
      obtain ⟨s22', hact', hacc22, hpres22, -, -⟩ := actTail_steps s18' BM BN ACTIVATION _ hacc18
      rw [tlaActStmts_castFree, hact'] at h22
      obtain rfl := Option.some.inj h22
      exact tla_storeTailSafeR R bounds C s M N smo sno BM BN GM _
        ((hpres22 (dt := .nat) (sh := [BM]) "m_offs" (by decide)).trans hm18)
        ((hpres22 (dt := .nat) (sh := [BN]) "n_offs" (by decide)).trans hn18) hbC

set_option maxHeartbeats 2000000 in
/-- The whole 17-statement prologue is cast-free. -/
private theorem tlaPrologue_castFree (R : RoundingModel) (A B biasR : RegionName)
    (M N sam sak sbk sbn BM GM BN BK : Nat) (HAS_BIAS : Bool) (t : BlockState) :
    stepStmtsR R (tlaScalars M N BM BN BK GM
        ++ tlaSeeds A B biasR N sam sak sbk sbn BM BN BK HAS_BIAS) t
      = stepStmts (tlaScalars M N BM BN BK GM
        ++ tlaSeeds A B biasR N sam sak sbk sbn BM BN BK HAS_BIAS) t := by
  simp only [tlaScalars, tlaSeeds, biasIfStmt, List.cons_append, List.nil_append,
    stepStmtsR, stepStmts, stepStmtR, stepStmt, evalOpR.eq_def, evalOp.eq_def]
  rfl

/-! ### ════════ ★ THE `⊨[R]` IO HEADLINE ★ ════════ -/

open scoped VeriTile.Triton.StreamMasked3DKernelIO₃ₓ₂

set_option maxHeartbeats 4000000 in
/-- **The `⊨[R]` io headline for `triton_linear_activation.py`'s `kernel_fma`**
(round-12; the second `StreamMasked3DKernelIO₃ₓ₂`-with-`pre` consumer, and
the genre's first *cast-free GEMM* face).

For **every** rounding model `R` — with no hypothesis on `R` whatsoever, the
kernel being entirely cast-free (no `Op.castFloat`, both terminal stores
`Stmt.store .real`) — the faithful `kernel_fma` surface implements, on its
`StreamMasked3DKernelIO₃ₓ₂` signature, the **ideal-ℝ fused linear +
activation** over the three streamed channels:

* `C` lane `l = (i, j)` holds
  `applyActivation ACTIVATION (bias-seed + ∑ t, ∑ e, A-tile[t](i,e) · B-tile[t](e,j))`;
* `ACT_INPUTS` lane `l` holds the **un-activated** same value (the claim is
  vacuous unless `SHOULD_SAVE_ACT_INPUTS`, which `writeMask2` carries).

The whole constexpr mix stays **parametric**, exactly as the port has it: the
`String` parameter `ACTIVATION` enters `f` through `applyActivation` — itself
the same four *sequential, non-exclusive* `ACTIVATION == "…"` gates the kernel
runs, so even pathological strings are modelled faithfully and every
unrecognised string (e.g. the benchmark's `""`) is the identity; the `Bool`
parameter `HAS_BIAS` enters through `tlaBiasIO`'s outer guard and through
`mask3`; the `Bool` parameter `SHOULD_SAVE_ACT_INPUTS` enters through
`writeMask2`. Nothing is specialised to a configuration.

**`pre` = the trusted-launch boundary.** `io.pre pid₀ _ _` is instantiated
with exactly the exact headline's `hFitM ∧ hFitN` (this program's tile fits
the output). It cannot be demoted into `writeMask1`: a non-fitting program
still *writes*, at `%`-wrapped addresses, so the skin's frame clause would be
false. Under `pre` the kernel's `c_ptr_mask` is all-true, which is why
`writeMask1 = True`.

**Hypothesis provenance** (all inherited from the exact stack, all
truth-forced):
* `hT : 0 < numKBlocks` — the `bias` row is a *degenerate static stream* read
  before the loop; with `T = 0` there is no step for the skin to carry it on.
* `hsno`/`hble` (`output_n_stride = 1`, `BLOCK_N ≤ output_m_stride`) — the
  exact headline's store-footprint injectivity for `C`
  (`cOffset_injective_of_fit`); `hsain`/`hale` are the same facts for
  `ACT_INPUTS`, and `hne : ACT_INPUTS ≠ C` keeps the `C` store from
  clobbering the saved pre-activation tile. These are the wrapper's
  contiguous row-major layouts.

**Disclosed scope.** (1) The io face is stated for a **5-region disjoint**
allocation `[A, B, bias, C, ACT_INPUTS]`; the real launcher aliases
`bias := x` when `bias is None` and passes `ACT_INPUTS := None` when not
saving, so those two flag-off configurations fall outside this allocation
shape (their claims are still available from the exact headline, which needs
no allocation). (2) The ℕ-truncated `grid_m - group_idx·GROUP_M` of the
L2 schedule is transcribed as the kernel computes it: for a `program_idx`
past the grid the truncation can make `group_size = 0`, and Lean's
`pid % 0 = pid` / `_ / 0 = 0` then silently diverge from Triton's UB. This is
*not* repaired here — `blockMIdx`/`blockNIdx` are the kernel's own arithmetic,
`pre` merely scopes the claim to whatever tile those produce. (3) `.real`
output grids: the port models no narrow-float store, so at every `R` the two
terminal cells carry the exact fold values.

The exact headline `triton_linear_activation_output_summary_general` above is
retained unchanged; this face strictly restates its content on the streaming
skin (at `R := .triv` the readback contract degenerates to it). -/
specification triton_linear_activation_io_correctness (R : RoundingModel)
    (C ACT_INPUTS A B bias : RegionName)
    (M N output_m_stride output_n_stride act_inputs_m_stride act_inputs_n_stride
      a_m_stride a_k_stride b_n_stride b_k_stride
      BLOCK_M GROUP_M BLOCK_N BLOCK_K numKBlocks : Nat)
    (HAS_BIAS SHOULD_SAVE_ACT_INPUTS : Bool) (ACTIVATION : String)
    (hT : 0 < numKBlocks)
    (hsno : output_n_stride = 1) (hble : BLOCK_N ≤ output_m_stride)
    (hsain : act_inputs_n_stride = 1) (hale : BLOCK_N ≤ act_inputs_m_stride)
    (hne : ACT_INPUTS ≠ C) :
    tlaIO C ACT_INPUTS A B bias M N output_m_stride output_n_stride
        act_inputs_m_stride act_inputs_n_stride a_m_stride a_k_stride b_n_stride b_k_stride
        BLOCK_M GROUP_M BLOCK_N BLOCK_K numKBlocks
        HAS_BIAS SHOULD_SAVE_ACT_INPUTS ACTIVATION ⊨[R]
      fun p₀ _ _ xs ys zs =>
        (fun l => applyActivation ACTIVATION
            (tlaLinIO p₀ M N BLOCK_M BLOCK_N GROUP_M BLOCK_K numKBlocks HAS_BIAS xs ys zs l),
         fun l => tlaLinIO p₀ M N BLOCK_M BLOCK_N GROUP_M BLOCK_K numKBlocks HAS_BIAS xs ys zs l) := by
  subst hsno
  subst hsain
  refine StreamMasked3DKernelIO₃ₓ₂.ImplementsR.intro _ ?_ ?_ ?_
  · exact tla_flattenOk C ACT_INPUTS A B bias M N output_m_stride 1
      act_inputs_m_stride 1 a_m_stride a_k_stride b_n_stride b_k_stride
      BLOCK_M GROUP_M BLOCK_N BLOCK_K numKBlocks HAS_BIAS SHOULD_SAVE_ACT_INPUTS ACTIVATION
  · -- the safety walk
    intro bounds s xs ys zs _hlaunch _hx _hy _hz hbr1 hbr2 hbr3 hbw1 hbw2
    simp only [tlaIO] at hbr1 hbr2 hbr3 hbw1 hbw2 ⊢
    exact tla_traceSafeR R bounds C ACT_INPUTS A B bias M N output_m_stride 1
      act_inputs_m_stride 1 a_m_stride a_k_stride b_n_stride b_k_stride
      BLOCK_M GROUP_M BLOCK_N BLOCK_K numKBlocks HAS_BIAS SHOULD_SAVE_ACT_INPUTS ACTIVATION s
      (fun t l => hbr1 t l trivial) (fun t l => hbr2 t l trivial)
      (fun j hb hlt => hbr3 ⟨0, hT⟩ j ⟨hb, hlt⟩)
      (fun l => hbw1 l trivial) (fun l hb => hbw2 l hb)
  · -- the rounded Hoare triple: exact invariant stack + cast-free collapse
    intro s₀ xs ys zs hlaunch _hu hx hy hz
    simp only [tlaIO] at hlaunch hx hy hz ⊢
    obtain ⟨hFitM, hFitN⟩ := hlaunch
    have hInjC : Function.Injective
        (cOffset s₀ M N BLOCK_M BLOCK_N GROUP_M output_m_stride 1) :=
      cOffset_injective_of_fit s₀ M N BLOCK_M BLOCK_N GROUP_M output_m_stride hFitM hFitN hble
    have hInjAct : Function.Injective
        (actOffset s₀ M N BLOCK_M BLOCK_N GROUP_M act_inputs_m_stride 1) :=
      cOffset_injective_of_fit s₀ M N BLOCK_M BLOCK_N GROUP_M act_inputs_m_stride
        hFitM hFitN hale
    -- the exact stack: prologue, K-loop, tail
    obtain ⟨sPre, hpre, hP0⟩ := preLoop C ACT_INPUTS A B bias s₀ M N output_m_stride 1
      act_inputs_m_stride 1 a_m_stride a_k_stride b_n_stride b_k_stride
      BLOCK_M GROUP_M BLOCK_N BLOCK_K numKBlocks HAS_BIAS SHOULD_SAVE_ACT_INPUTS ACTIVATION
    have hpre' : stepStmts (tlaScalars M N BLOCK_M BLOCK_N BLOCK_K GROUP_M
        ++ tlaSeeds A B bias N a_m_stride a_k_stride b_k_stride b_n_stride
            BLOCK_M BLOCK_N BLOCK_K HAS_BIAS) s₀ = some sPre := by
      rw [← tla_take17 C ACT_INPUTS A B bias M N output_m_stride 1
        act_inputs_m_stride 1 a_m_stride a_k_stride b_n_stride b_k_stride
        BLOCK_M GROUP_M BLOCK_N BLOCK_K numKBlocks HAS_BIAS SHOULD_SAVE_ACT_INPUTS ACTIVATION]
      exact hpre
    obtain ⟨final, sL, hLoopStmt, hfinal, hPL⟩ :=
      forRange_inv (idx := "k") (start := 0) (stop := numKBlocks) (step := 1)
        (body := fmaLoopBody BLOCK_M BLOCK_N BLOCK_K a_k_stride b_k_stride)
        (by omega) hP0
        (fun c st hlt hinv => by
          have := kernel_fma_step A B bias s₀ M N BLOCK_M BLOCK_N GROUP_M a_m_stride a_k_stride
            b_k_stride b_n_stride BLOCK_K numKBlocks HAS_BIAS c st hlt hinv
          simpa using this)
    have hfinalEq : final = numKBlocks := by
      have hle : final ≤ numKBlocks := by
        simp only [fmaInvariant] at hPL
        exact hPL.1
      exact le_antisymm hle hfinal
    rw [hfinalEq] at hPL
    have hPL' := hPL
    simp only [fmaInvariant] at hPL'
    obtain ⟨-, hmL, hnL, hzL, -, -, hmemL⟩ := hPL'
    obtain ⟨sfin, hTail, hC, hAct⟩ :=
      kernel_fma_postLoop C ACT_INPUTS A B bias s₀ M N output_m_stride 1
        act_inputs_m_stride 1 a_m_stride a_k_stride b_k_stride b_n_stride
        BLOCK_M BLOCK_N GROUP_M BLOCK_K numKBlocks HAS_BIAS SHOULD_SAVE_ACT_INPUTS ACTIVATION
        hInjC sL hPL
    obtain ⟨sfin2, hTail2, hframe⟩ :=
      tla_tail_frame C ACT_INPUTS s₀ M N output_m_stride 1 act_inputs_m_stride 1
        BLOCK_M BLOCK_N GROUP_M SHOULD_SAVE_ACT_INPUTS ACTIVATION _ sL hmL hnL hzL
    have hfineq : sfin2 = sfin := Option.some.inj (hTail2.symm.trans hTail)
    rw [hfineq] at hframe
    -- the streaming spec bridge
    have hspec : ∀ l : Fin (BLOCK_M * BLOCK_N),
        linearSpec s₀ A B bias M N BLOCK_M BLOCK_N GROUP_M a_m_stride a_k_stride
            b_k_stride b_n_stride BLOCK_K numKBlocks HAS_BIAS
            (Lane2D.decode l).1 (Lane2D.decode l).2.1
          = tlaLinIO (s₀.pids 0) M N BLOCK_M BLOCK_N GROUP_M BLOCK_K numKBlocks
              HAS_BIAS xs ys zs l := fun l =>
      tlaLinIO_eq A B bias s₀ M N BLOCK_M BLOCK_N GROUP_M a_m_stride a_k_stride
        b_k_stride b_n_stride BLOCK_K numKBlocks HAS_BIAS hT xs ys zs
        (fun t l' => hx t l' trivial) (fun t l' => hy t l' trivial)
        (fun t j hb hlt => hz t j ⟨hb, hlt⟩) l
    refine ⟨sfin, ?_, ?_, ?_, ?_⟩
    · -- termination under `execR R` (the whole kernel is cast-free)
      unfold execR
      rw [tla_body_split C ACT_INPUTS A B bias M N output_m_stride 1
          act_inputs_m_stride 1 a_m_stride a_k_stride b_n_stride b_k_stride
          BLOCK_M GROUP_M BLOCK_N BLOCK_K numKBlocks HAS_BIAS SHOULD_SAVE_ACT_INPUTS ACTIVATION,
        stepStmtsR_append,
        tlaPrologue_castFree R A B bias M N a_m_stride a_k_stride b_k_stride b_n_stride
          BLOCK_M GROUP_M BLOCK_N BLOCK_K HAS_BIAS s₀,
        hpre', Option.bind_some,
        stepStmtsR_cons_some
          (show stepStmtR R (Stmt.forRange "k" 0 numKBlocks 1
              (fmaLoopBody BLOCK_M BLOCK_N BLOCK_K a_k_stride b_k_stride)) sPre = some sL from by
            rw [stepStmtR_forRange,
              stepForRangeAuxR_castFree R _
                (tlaLoopBody_castFree R BLOCK_M BLOCK_N BLOCK_K a_k_stride b_k_stride) "k",
              ← stepForRangeAux.forRange_unfold]
            exact hLoopStmt),
        tlaTail_castFree]
      exact hTail
    · -- `C` readback
      intro l _
      rw [BlockState.readMemAs_real, tla_write1_eq,
        hC (Lane2D.decode l)
          (rowIndex_lt_of_fit s₀ M N BLOCK_M BLOCK_N GROUP_M hFitM (Lane2D.decode l).1)
          (colIndex_lt_of_fit s₀ M N BLOCK_M BLOCK_N GROUP_M hFitN (Lane2D.decode l).2.1),
        hspec l]
      simp [FloatDType.ofReal]
    · -- `ACT_INPUTS` readback (vacuous unless the save flag is on)
      intro l hsave
      rw [BlockState.readMemAs_real, tla_write2_eq,
        hAct hsave hne hInjAct (Lane2D.decode l), hspec l]
      simp [FloatDType.ofReal]
    · -- the frame
      intro r oo hcond
      obtain ⟨h1, h2⟩ := hcond
      rw [hframe r oo ?_ ?_, hmemL]
      · intro hr hsave idx
        have h := h2 (Lane2D.encode (idx.1, idx.2.1, PUnit.unit)) hsave hr
        simpa only [Lane2D.encode_div, Lane2D.encode_mod] using h
      · intro hr idx
        have h := h1 (Lane2D.encode (idx.1, idx.2.1, PUnit.unit)) trivial hr
        simpa only [Lane2D.encode_div, Lane2D.encode_mod] using h

end VeriTile.Bench.TritonBenchG.TritonLinearActivation
