import VeriTile.Triton

/-!
# `rowwise_quantization_triton` — strict per-kernel correctness

`_quantize_rowwise` is a per-row int8 quantizer: program `pid` loads one row of
`x` (a `P2`-wide tile masked to the first `BLOCK_SIZE` lanes), computes
`max_val = max(|x|)` over the row, rounds `127.0 · (x / max_val)` to int8 via
CUDA `llrint`, stores the row to `output_ptr`, and writes `max_val` to
`output_maxs + pid`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`_quantize_rowwise[grid](...)`, the grid size
`x.shape[0]`, the host-side `BLOCK_SIZE`/`P2` choice, and how the runtime
composes per-program writes into one buffer) is the *trusted boundary*, not a
proof obligation here. Because `pid` is universally quantified, the per-program
statement covers every program of the grid.

## Proof architecture

```
quantize_rowwise_correctness                     ← TOP THEOREM (quantizeRowwiseIO ⊨ (row-spec, max-spec))
  ├─ quantize_rowwise_real_surface_toAlgorithm_blocked   faithful surface blocked at erasure (llrint)
  ├─ quantize_rowwise_flattenOk                  bridge fragment membership
  ├─ quantize_rowwise_traceSafe                  per-execution lane-wise safety walk
  └─ quantize_rowwise_region_run                 region-model masked Hoare triple
       ├─ quantize_rowwise_exec_isSome           termination (needs `0 < P2`)
       ├─ quantize_rowwise_out_correct           ← masked row store readback
       ├─ quantize_rowwise_max_correct           ← scalar `output_maxs` store readback
       ├─ qrMaxSpec_eq_of_loaded / qrOutSpec_eq_of_loaded   pure-spec bridges
       └─ quantize_rowwise_frame                 masked-scatter + scalar-store frame
```

The headline is the masked two-output Hoare-triple combinator
`quantizeRowwiseIO … ⊨ (qrOutSpec, qrMaxSpec)`
(`Masked2DKernelIO₂ₓ₂.Implements`): the row max **is computed by the verified
kernel** (`qrMaxCarrier` = `sup'` of `tl.where(row_mask, |x|, 0)` over the whole
`P2` tile), the active output lanes hold `127 · (x / max_val)`, and
`output_maxs[pid]` holds `max_val`.

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); `@triton.autotune` /
`num_warps` / `num_stages` are not modeled. The **faithful surface**
(`quantize_rowwise_real_surface`) is *blocked* at algorithm erasure: the CUDA
`tl.extra.cuda.libdevice.llrint` rounding and the int8 cast are outside
VeriTile's real-tile arithmetic layer, so `toAlgorithm?` returns
`Except.error`. The verified kernel `quantize_rowwise_scaled` is the Python
kernel with exactly that final rounding erased — every other statement
(addressing, masking, `tl.abs`, the masked max reduction, both stores) is
transcribed 1:1. The `llrint` rounding / int8 cast is therefore the honest,
unmodeled blocker, recorded as its own theorem.

Masked lanes of the `x` load are `⊥` (the Python `tl.load` has no `other=`),
and `tl.where(row_mask, abs_x, 0.0)` replaces them by `0` before the reduction,
so the reduction over the padded `P2` block matches the upstream semantics.
The output row store is masked, so `⊥` never reaches memory.

The IO skin is the two-input / two-output masked family; this kernel has a
single input, so the signature's **second input slot is a declared-inert
allocation slot**: its read gate is `False`, so it carries no obligation on
either side (it only makes the headline quantify over allocations that also
place a fourth, untouched buffer). `0 < P2` is genuinely forced (see the
headline docstring), and `output_ptr ≠ output_maxs` is required because the
scalar max store is unmasked and must not alias the row store.
-/

namespace VeriTile.Bench.TritonBenchG.RowwiseQuantizationTriton

open VeriTile.Triton
open scoped VeriTile.Triton.Masked2DKernelIO₂ₓ₂

set_option linter.unusedSimpArgs false

set_option maxHeartbeats 4000000

/-- Faithful 1:1 transcription of `rowwise_quantization_triton.py`'s
`_quantize_rowwise`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` / `P2: tl.constexpr` → Lean `Nat`
  parameters. -/
def quantize_rowwise_real_surface
    (x_ptr output_ptr output_maxs : RegionName)
    (_n_elements BLOCK_SIZE P2 : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  arange = tl.arange(0, $(P2))
  offsets = block_start + arange
  row_mask = arange < $(BLOCK_SIZE)
  x = tl.load(x_ptr + offsets, mask=row_mask)
  abs_x = tl.abs(x)
  max_val = tl.max(tl.where(row_mask, abs_x, 0.0), axis=0)
  output = tl.extra.cuda.libdevice.llrint(127.0 * (x / max_val))
  tl.store(output_ptr + offsets, output, mask=row_mask)
  tl.store(output_maxs + pid, max_val)
}

/-- The faithful rowwise quantization surface is blocked at algorithm erasure by
the CUDA `llrint` rounding operation. The verified kernel below is this surface
with exactly that rounding erased. -/
theorem quantize_rowwise_real_surface_toAlgorithm_blocked
    (x_ptr output_ptr output_maxs : RegionName)
    (_n_elements BLOCK_SIZE P2 : Nat) :
    ∃ err,
      (quantize_rowwise_real_surface x_ptr output_ptr output_maxs _n_elements
        BLOCK_SIZE P2).toAlgorithm? = Except.error err := by
  simp [quantize_rowwise_real_surface, ComputeExpr.toAlgorithm?]

/-- The verified kernel: `_quantize_rowwise` with the CUDA `llrint` rounding
(and the implicit int8 cast) erased. Every other statement — the row
addressing, the row mask, `tl.abs`, the masked max reduction, the scaled row
store and the scalar `output_maxs` store — is transcribed 1:1 from the Python
kernel. -/
def quantize_rowwise_scaled
    (x_ptr output_ptr output_maxs : RegionName)
    (_n_elements BLOCK_SIZE P2 : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  arange = tl.arange(0, $(P2))
  offsets = block_start + arange
  row_mask = arange < $(BLOCK_SIZE)
  x = tl.load(x_ptr + offsets, mask=row_mask)
  abs_x = tl.abs(x)
  max_val = tl.max(tl.where(row_mask, abs_x, 0.0), axis=0)
  output = 127.0 * (x / max_val)
  tl.store(output_ptr + offsets, output, mask=row_mask)
  tl.store(output_maxs + pid, max_val)
}

/-! ### The pure row specs

All specs below are pure functions of the **active** row lanes `xs j`
(`j < BLOCK_SIZE`); inactive lanes of the `P2`-wide tile carry no content. -/

/-- The reduced tile `tl.where(row_mask, tl.abs(x), 0.0)`: active lanes hold
`|xs j|`, the padded lanes hold `0`. -/
noncomputable def qrAbsTile (BLOCK_SIZE P2 : Nat) (xs : Fin P2 → ℝ) :
    Tile .real [P2] :=
  { data := fun idx =>
      if idx.1.val < BLOCK_SIZE then
        some (if xs idx.1 < 0 then -xs idx.1 else xs idx.1)
      else some (0 : ℝ) }

/-- The row maximum carrier `max_val = tl.max(tl.where(row_mask, |x|, 0))`. -/
noncomputable def qrMaxCarrier (BLOCK_SIZE P2 : Nat) (xs : Fin P2 → ℝ) :
    Option (Tile .real (TileShape.eraseAxis [P2] ⟨0, by simp⟩)) :=
  Tile.reduceMaxDrop (shape := [P2]) ⟨0, by simp⟩ (qrAbsTile BLOCK_SIZE P2 xs)

/-- The scalar written to `output_maxs[pid]`. -/
noncomputable def qrMaxSpec (BLOCK_SIZE P2 : Nat) (xs : Fin P2 → ℝ) : ℝ :=
  match qrMaxCarrier BLOCK_SIZE P2 xs with
  | some m => WithBot.unbotD 0 (m.data PUnit.unit)
  | none => 0

/-- The value written to the active output row lane `i`: `127 · (x i / max)`. -/
noncomputable def qrOutSpec (BLOCK_SIZE P2 : Nat) (xs : Fin P2 → ℝ)
    (i : Fin P2) : ℝ :=
  match qrMaxCarrier BLOCK_SIZE P2 xs with
  | some m =>
      WithBot.unbotD 0
        (Option.map (fun b => 127.0 * (xs i / b)) (m.data PUnit.unit))
  | none => 0


private theorem quantize_rowwise_exec_isSome
    (x_ptr output_ptr output_maxs : RegionName)
    (n_elements BLOCK_SIZE P2 : Nat) (hP : 0 < P2) (s : BlockState) :
    ∃ s1, exec ((quantize_rowwise_scaled x_ptr output_ptr output_maxs
        n_elements BLOCK_SIZE P2).toAlgKernel) s = some s1 := by
  simp [exec, quantize_rowwise_scaled, ComputeKernel.toAlgKernel, stepStmts,
        stepStmt, evalOp, evalOp.eq_def,
        Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.select,
        Tile.reduceMax, Tile.reduceMaxDrop, TileShape.axisDim,
        TileShape.eraseAxis, TileShape.insertAxisIndex,
        NumericDType.add, NumericDType.mul, NumericDType.sub,
        NumericDType.div, ComparableDType.lt,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, hP]


noncomputable def qrRow (s : BlockState) (x_ptr : RegionName) (BLOCK_SIZE P2 : Nat)
    (j : Fin P2) : ℝ :=
  s.readMem x_ptr (s.pid * BLOCK_SIZE + j.val)

theorem quantize_rowwise_max_correct
    (x_ptr output_ptr output_maxs : RegionName)
    (n_elements BLOCK_SIZE P2 : Nat) (hP : 0 < P2) (s s1 : BlockState)
    (hExec : exec ((quantize_rowwise_scaled x_ptr output_ptr output_maxs
        n_elements BLOCK_SIZE P2).toAlgKernel) s = some s1) :
    s1.readMem output_maxs s.pid
      = qrMaxSpec BLOCK_SIZE P2 (qrRow s x_ptr BLOCK_SIZE P2) := by
  simp [exec, quantize_rowwise_scaled, ComputeKernel.toAlgKernel, stepStmts,
        stepStmt, evalOp, evalOp.eq_def,
        Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.select,
        Tile.reduceMax, Tile.reduceMaxDrop, TileShape.axisDim,
        TileShape.eraseAxis, TileShape.insertAxisIndex,
        NumericDType.add, NumericDType.mul, NumericDType.sub,
        NumericDType.div, ComparableDType.lt,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, hP] at hExec
  subst hExec
  simp [qrMaxSpec, qrMaxCarrier, qrAbsTile, qrRow, Tile.reduceMaxDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex, hP]
  congr 1
  congr 1
  funext k
  by_cases hk : (k : Nat) < BLOCK_SIZE
  · by_cases hv : s.readMem x_ptr (s.pids 0 * BLOCK_SIZE + (k : Nat)) < 0
    · simp [hk, hv]
      exact fun h => absurd (WithBot.coe_le_coe.mp h) (not_le.mpr hv)
    · simp [hk, hv]
      exact fun h => absurd (WithBot.coe_lt_coe.mp h) hv
  · simp [hk]
    norm_num


theorem quantize_rowwise_out_correct
    (x_ptr output_ptr output_maxs : RegionName)
    (n_elements BLOCK_SIZE P2 : Nat) (hP : 0 < P2)
    (hRegions : output_ptr ≠ output_maxs) (s s1 : BlockState)
    (hExec : exec ((quantize_rowwise_scaled x_ptr output_ptr output_maxs
        n_elements BLOCK_SIZE P2).toAlgKernel) s = some s1) :
    ∀ i : Fin P2,
      s1.readMem output_ptr (s.pid * BLOCK_SIZE + i.val)
        = if i.val < BLOCK_SIZE then
            qrOutSpec BLOCK_SIZE P2 (qrRow s x_ptr BLOCK_SIZE P2) i
          else s.readMem output_ptr (s.pid * BLOCK_SIZE + i.val) := by
  intro i
  simp [exec, quantize_rowwise_scaled, ComputeKernel.toAlgKernel, stepStmts,
        stepStmt, evalOp, evalOp.eq_def,
        Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.select,
        Tile.reduceMax, Tile.reduceMaxDrop, TileShape.axisDim,
        TileShape.eraseAxis, TileShape.insertAxisIndex,
        NumericDType.add, NumericDType.mul, NumericDType.sub,
        NumericDType.div, ComparableDType.lt,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, hP] at hExec
  subst hExec
  simp [BlockState.pid_eq, hRegions]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
        (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
  by_cases hi : i.val < BLOCK_SIZE
  · simp [hi, qrOutSpec, qrMaxCarrier, qrAbsTile, qrRow, Tile.reduceMaxDrop,
          TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex, hP]
    congr 1
    congr 1
    congr 1
    funext k
    by_cases hk : (k : Nat) < BLOCK_SIZE
    · by_cases hv : s.readMem x_ptr (s.pids 0 * BLOCK_SIZE + (k : Nat)) < 0
      · simp [hk, hv]
        exact fun h => absurd (WithBot.coe_le_coe.mp h) (not_le.mpr hv)
      · simp [hk, hv]
        exact fun h => absurd (WithBot.coe_lt_coe.mp h) hv
    · simp [hk]
      norm_num
  · simp [hi]


/-! ### Pure-spec bridges

`qrAbsTile` reads its argument only at the **active** lanes, so a launch state
whose active `x` lanes hold `xs` computes the pure specs of `xs`. -/

theorem qrAbsTile_congr (BLOCK_SIZE P2 : Nat) (xs ys : Fin P2 → ℝ)
    (h : ∀ j : Fin P2, j.val < BLOCK_SIZE → xs j = ys j) :
    qrAbsTile BLOCK_SIZE P2 xs = qrAbsTile BLOCK_SIZE P2 ys := by
  unfold qrAbsTile
  congr 1
  funext idx
  by_cases hj : idx.1.val < BLOCK_SIZE
  · simp only [if_pos hj, h idx.1 hj]
  · simp only [if_neg hj]

theorem qrMaxSpec_congr (BLOCK_SIZE P2 : Nat) (xs ys : Fin P2 → ℝ)
    (h : ∀ j : Fin P2, j.val < BLOCK_SIZE → xs j = ys j) :
    qrMaxSpec BLOCK_SIZE P2 xs = qrMaxSpec BLOCK_SIZE P2 ys := by
  unfold qrMaxSpec qrMaxCarrier
  rw [qrAbsTile_congr BLOCK_SIZE P2 xs ys h]

theorem qrOutSpec_congr (BLOCK_SIZE P2 : Nat) (xs ys : Fin P2 → ℝ)
    (h : ∀ j : Fin P2, j.val < BLOCK_SIZE → xs j = ys j)
    (i : Fin P2) (hi : i.val < BLOCK_SIZE) :
    qrOutSpec BLOCK_SIZE P2 xs i = qrOutSpec BLOCK_SIZE P2 ys i := by
  unfold qrOutSpec qrMaxCarrier
  rw [qrAbsTile_congr BLOCK_SIZE P2 xs ys h, h i hi]

/-! ### Frame, bridge membership and the safety walk -/

/-- A masked scatter-store `foldl` leaves every memory cell it does not
actively hit unchanged (cell-level frame for the masked row store). -/
private theorem foldl_store_preserve_cell {α : Type} {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → ℝ) (P : α → Prop) [DecidablePred P]
    (ρ : RegionName) (o : Nat) (l : List α) (s : BlockState)
    (hnot : ∀ k ∈ l, P k → ¬(region = ρ ∧ offsetFn k = o)) :
    (l.foldl (fun acc k =>
        if P k then acc.writeMem region (offsetFn k) (valueFn k) else acc)
      s).mem ρ o = s.mem ρ o := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons]
      by_cases hP : P hd
      · rw [if_pos hP,
          ih _ (fun k hk => hnot k (List.mem_cons_of_mem hd hk)),
          BlockState.writeMem_mem]
        exact if_neg (fun hc =>
          hnot hd List.mem_cons_self hP ⟨hc.1.symm, hc.2.symm⟩)
      · rw [if_neg hP]
        exact ih _ (fun k hk => hnot k (List.mem_cons_of_mem hd hk))

/-- Frame half: every memory cell not actively written — every cell outside the
active output-row lanes and off the scalar `output_maxs[pid]` cell — is
preserved by the run. -/
private theorem quantize_rowwise_frame
    (x_ptr output_ptr output_maxs : RegionName)
    (n_elements BLOCK_SIZE P2 : Nat) (hP : 0 < P2) (s s1 : BlockState)
    (hExec : exec ((quantize_rowwise_scaled x_ptr output_ptr output_maxs
        n_elements BLOCK_SIZE P2).toAlgKernel) s = some s1)
    (ρ : RegionName) (o : Nat)
    (hmissOut : ∀ i : Fin P2, i.val < BLOCK_SIZE →
      ¬(output_ptr = ρ ∧ s.pid * BLOCK_SIZE + i.val = o))
    (hmissMax : ¬(output_maxs = ρ ∧ s.pid = o)) :
    s1.mem ρ o = s.mem ρ o := by
  simp [exec, quantize_rowwise_scaled, ComputeKernel.toAlgKernel, stepStmts,
        stepStmt, evalOp, evalOp.eq_def,
        Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.select,
        Tile.reduceMax, Tile.reduceMaxDrop, TileShape.axisDim,
        TileShape.eraseAxis, TileShape.insertAxisIndex,
        NumericDType.add, NumericDType.mul, NumericDType.sub,
        NumericDType.div, ComparableDType.lt,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, hP] at hExec
  subst hExec
  simp only [BlockState.writeMem_mem, BlockState.pid_eq] at hmissMax ⊢
  rw [if_neg (fun hc => hmissMax ⟨hc.1.symm, hc.2.symm⟩)]
  refine Eq.trans (foldl_store_preserve_cell _ _ _ ρ o _ _ ?_) rfl
  intro k _ hmk hc
  exact hmissOut k.1 (by simpa using hmk) (by simpa [BlockState.pid_eq] using hc)

/-- The kernel sits inside the flat-memory bridge's covered fragment (pointer
arithmetic, masked load, `tl.where`, the max reduction, masked store, scalar
store). -/
theorem quantize_rowwise_flattenOk
    (x_ptr output_ptr output_maxs : RegionName)
    (n_elements BLOCK_SIZE P2 : Nat) :
    ((quantize_rowwise_scaled x_ptr output_ptr output_maxs n_elements
        BLOCK_SIZE P2).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [quantize_rowwise_scaled, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

/-- Per-execution safety walk: the masked `x` load and the masked row store
reduce to **lane-wise** bounds at the active lanes; the **unmasked** scalar
`output_maxs` store reduces to the single-cell bound `pid < bounds
output_maxs`. -/
theorem quantize_rowwise_traceSafe
    (x_ptr output_ptr output_maxs : RegionName)
    (n_elements BLOCK_SIZE P2 : Nat) (hP : 0 < P2)
    (bounds : RegionBounds) (s : BlockState)
    (hin : ∀ j : Fin P2, j.val < BLOCK_SIZE →
      s.pid * BLOCK_SIZE + j.val < bounds x_ptr)
    (hout : ∀ j : Fin P2, j.val < BLOCK_SIZE →
      s.pid * BLOCK_SIZE + j.val < bounds output_ptr)
    (hmax : s.pid < bounds output_maxs) :
    Kernel.TraceSafe bounds
      ((quantize_rowwise_scaled x_ptr output_ptr output_maxs n_elements
        BLOCK_SIZE P2).toAlgKernel) s := by
  unfold Kernel.TraceSafe
  simp only [BlockState.pid_eq] at hin hout hmax
  simp [quantize_rowwise_scaled, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, evalOp.eq_def,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    MaskOpt.Active, Op.PointerAddressesSafeOn, Op.MemorySafe,
    BlockState.setReg,
    Tile.bop, Tile.cop, Tile.uop, Tile.ptrAdd, Tile.select,
    NumericDType.add, NumericDType.mul, NumericDType.sub,
    NumericDType.div, ComparableDType.lt,
    Tile.reduceMax, Tile.reduceMaxDrop,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex, hP]
  exact ⟨fun a ha => hin a ha, fun a ha => hout a ha, hmax⟩

/-- **The region-model masked Hoare triple** — termination, active row-lane
output values, the scalar `output_maxs` value, and frame off both written
windows, from any launch state whose active `x` lanes hold `xs`. This is the
`hrun` obligation of the `⊨` headline. -/
theorem quantize_rowwise_region_run
    (x_ptr output_ptr output_maxs : RegionName)
    (n_elements BLOCK_SIZE P2 : Nat) (hP : 0 < P2)
    (hRegions : output_ptr ≠ output_maxs)
    (s₀ : BlockState) (xs : Fin P2 → ℝ)
    (hx : ∀ j : Fin P2, j.val < BLOCK_SIZE →
      s₀.readMem x_ptr (s₀.pid * BLOCK_SIZE + j.val) = xs j) :
    ∃ s1, exec ((quantize_rowwise_scaled x_ptr output_ptr output_maxs
          n_elements BLOCK_SIZE P2).toAlgKernel) s₀ = some s1
      ∧ (∀ j : Fin P2, j.val < BLOCK_SIZE →
          s1.readMem output_ptr (s₀.pid * BLOCK_SIZE + j.val)
            = qrOutSpec BLOCK_SIZE P2 xs j)
      ∧ s1.readMem output_maxs s₀.pid = qrMaxSpec BLOCK_SIZE P2 xs
      ∧ (∀ ρ o,
          (ρ ≠ output_ptr ∨ ∀ j : Fin P2, j.val < BLOCK_SIZE →
            o ≠ s₀.pid * BLOCK_SIZE + j.val) →
          (ρ ≠ output_maxs ∨ o ≠ s₀.pid) →
          s1.mem ρ o = s₀.mem ρ o) := by
  obtain ⟨s1, hs1⟩ := quantize_rowwise_exec_isSome x_ptr output_ptr output_maxs
    n_elements BLOCK_SIZE P2 hP s₀
  refine ⟨s1, hs1, fun j hj => ?_, ?_, fun ρ o h1 h2 => ?_⟩
  · have h := quantize_rowwise_out_correct x_ptr output_ptr output_maxs
      n_elements BLOCK_SIZE P2 hP hRegions s₀ s1 hs1 j
    rw [h, if_pos hj]
    exact qrOutSpec_congr BLOCK_SIZE P2 _ xs (fun k hk => hx k hk) j hj
  · have h := quantize_rowwise_max_correct x_ptr output_ptr output_maxs
      n_elements BLOCK_SIZE P2 hP s₀ s1 hs1
    rw [h]
    exact qrMaxSpec_congr BLOCK_SIZE P2 _ xs (fun k hk => hx k hk)
  · refine quantize_rowwise_frame x_ptr output_ptr output_maxs n_elements
      BLOCK_SIZE P2 hP s₀ s1 hs1 ρ o (fun i hi hc => ?_) (fun hc => ?_)
    · rcases h1 with hne | hno
      · exact hne hc.1.symm
      · exact hno i hi hc.2.symm
    · rcases h2 with hne | hno
      · exact hne hc.1.symm
      · exact hno hc.2.symm

/-- `_quantize_rowwise`'s masked two-output **IO signature** — the whole
kernel-specific audit surface of the `⊨` headline:

* `in1`/`out1`/`out2` — the input row buffer `x`, the quantized row buffer
  `output`, and the per-row maxima vector `output_maxs`;
* `in2` — a **declared-inert allocation slot**. The Python kernel has a single
  input; this two-input skin's second read channel is switched off
  (`read2Mask := fun _ _ _ => False`), so it carries no obligation on either
  side — the headline merely also quantifies over a fourth, untouched buffer;
* `B = P2` — the padded row window each program owns
  (`P2 = next_power_of_2(n_cols)` on the host side);
* `read1`/`write1` — program `pid` reads and writes its row at
  `pid · BLOCK_SIZE + j` (the host-side one-program-per-row launch);
* `write2` — the **scalar** cell `output_maxs[pid]`, the same for every lane;
* `mask` — the active lanes `j < BLOCK_SIZE`, i.e. the Python `row_mask`; the
  load mask and the row-store mask coincide, so `writeMask1` keeps its `mask`
  default;
* `writeMask2` — lane `0` carries the scalar row maximum; the other lanes are
  write-inactive.

The grid is 1-D, so the second program-id axis is an unused parameter: windows
and masks are constant in `pid₁`. The windows and masks are declared, not
parsed from the kernel; the headline **proves** the kernel's actual addressing
and masking match them. -/
def quantizeRowwiseIO (x_ptr inert output_ptr output_maxs : RegionName)
    (n_elements BLOCK_SIZE P2 : Nat) : Masked2DKernelIO₂ₓ₂ where
  kernel := quantize_rowwise_scaled x_ptr output_ptr output_maxs n_elements
    BLOCK_SIZE P2
  in1 := x_ptr
  in2 := inert
  out1 := output_ptr
  out2 := output_maxs
  B := P2
  read1 := fun pid _ j => pid * BLOCK_SIZE + j.val
  read2 := fun _ _ _ => 0
  write1 := fun pid _ j => pid * BLOCK_SIZE + j.val
  write2 := fun pid _ _ => pid
  mask := fun _ _ j => j.val < BLOCK_SIZE
  read2Mask := fun _ _ _ => False
  writeMask2 := fun _ _ j => j.val = 0

/-- **The headline**: `_quantize_rowwise` (with the CUDA `llrint` rounding
erased — see `quantize_rowwise_real_surface_toAlgorithm_blocked` for the honest
blocker) implements the exact rowwise-quantization pair on its masked
two-output IO signature. For every disjoint flat placement of the buffers,
every program id whose active lanes and scalar max cell are in bounds, and
every launch state whose active `x`-row lanes hold `xs`, the translated pointer
kernel terminates, every active output lane `j` holds
`qrOutSpec … = 127 · (xs j / max_val)`, the cell `output_maxs[pid]` holds
`qrMaxSpec … = max_val`, and every other memory cell is unchanged. The row
maximum `max_val` is **computed by the verified kernel**: it is the `sup'` of
`tl.where(row_mask, |x|, 0)` over the whole `P2` tile.

Side conditions, both genuinely forced: `0 < P2` (the `max` reduce, like
`Finset.sup'`, is only defined on a non-empty tile, and the unmasked scalar
store's bound and frame exclusion are carried by the lane-`0` gate
`writeMask2`, which needs a lane) and `output_ptr ≠ output_maxs` (the unmasked
scalar max store must not alias the masked row store). Proof:
`Masked2DKernelIO₂ₓ₂.Implements.intro` assembles the region-model masked triple
with the flat-memory bridge side conditions. -/
specification quantize_rowwise_correctness
    (x_ptr inert output_ptr output_maxs : RegionName)
    (n_elements BLOCK_SIZE P2 : Nat) (hP : 0 < P2)
    (hRegions : output_ptr ≠ output_maxs) :
    quantizeRowwiseIO x_ptr inert output_ptr output_maxs n_elements
        BLOCK_SIZE P2 ⊨
      fun _ _ xs _ =>
        (fun i => qrOutSpec BLOCK_SIZE P2 xs i,
         fun _ => qrMaxSpec BLOCK_SIZE P2 xs) := by
  refine Masked2DKernelIO₂ₓ₂.Implements.intro _ ?_ ?_ ?_
  · exact quantize_rowwise_flattenOk x_ptr output_ptr output_maxs n_elements
      BLOCK_SIZE P2
  · intro bounds s h1 _h2 h3 h4
    exact quantize_rowwise_traceSafe x_ptr output_ptr output_maxs n_elements
      BLOCK_SIZE P2 hP bounds s h1 h3 (h4 ⟨0, hP⟩ rfl)
  · intro s₀ xs _ys hx _hy
    obtain ⟨s1, hexec, hval1, hval2, hframe⟩ :=
      quantize_rowwise_region_run x_ptr output_ptr output_maxs n_elements
        BLOCK_SIZE P2 hP hRegions s₀ xs hx
    refine ⟨s1, hexec, hval1, fun j _ => hval2, fun ρ o hc1 hc2 => ?_⟩
    refine hframe ρ o hc1 ?_
    rcases hc2 with hne | hno
    · exact Or.inl hne
    · exact Or.inr (hno ⟨0, hP⟩ rfl)

end VeriTile.Bench.TritonBenchG.RowwiseQuantizationTriton
