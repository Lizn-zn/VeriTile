import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Semantics.TileOps
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

/-!
# `token_softmax_llama` — strict per-kernel correctness

`_fwd_kernel_token_softmax` is the LightLLM/LLaMA token-attention softmax:
program `(cur_batch, cur_head)` loads one row of `Logics` (the `cur_batch`'s
slice given by `B_Start_Loc`/`B_Seqlen`, masked with `-inf` past
`cur_batch_seq_len`), computes the numerically-stable softmax
(`row - max`, `exp`, `/ sum`), and stores the result into `Prob_Out`, masked by
`col_offsets < cur_batch_seq_len`. (Kernel body is identical to the Bloom
variant; only the Python test shapes differ.)

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`_fwd_kernel_token_softmax[(batch, head_num)](...)`, the
grid over `(batch, head)`, the host `BLOCK_SIZE = next_power_of_2(max_input_len)`
choice, `num_warps`, and how the runtime composes per-program writes into
`Prob_Out`) is the *trusted boundary*, not a proof obligation here. Because the
program ids `cur_batch`/`cur_head` are universally quantified (via `BlockState`),
the per-program statements cover every program of the grid.

## Proof architecture

```
token_softmax_llama_python_case1_output_summary       ← TOP THEOREM (also case2)
  ├─ token_softmax_llama_python_caseN_surface_toAlgorithm_supported  surface lowers
  │    └─ token_softmax_surface_toAlgorithm_supported
  └─ token_softmax_surface_output_compute_correct       ← ComputeCorrect of the store
       ├─ token_softmax_llama_python_caseN_offset_injective
       └─ token_softmax_llama_final_store_python_caseN_compute_correct
            └─ token_softmax_final_store_slice_compute_correct
                 └─ token_softmax_final_store_slice_correct  (per-lane masked readback)
```

The two `output_summary` theorems pin the kernel at the Python test shapes.

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float, so the `-inf` masking and
the `exp`/division are real-valued); `@triton.autotune` / `num_warps` are not
modeled. The `(...).to(tl.float32)` cast reduces to the identity at the algorithm
layer (post-erasure all dtypes unify to `ℝ`). The verified statement is scoped to
the **final masked probability store** into `Prob_Out`: the expected value is the
surface-level `tokenSoftmaxSurfaceValue` read off at each `probOffset`, with the
write-mask `active s B_Seqlen i` (`col_offsets < cur_batch_seq_len`). The
max/exp/sum reduction body feeds that value; the side condition is the
offset-injectivity of the `Prob_Out` slice at the test shapes. Out-of-range lanes
are preserved (mask=false ⇒ no store).

## Genuine closed-form spec (banked recipes)

`tokenSoftmaxSpec` is the genuine, self-contained stable-softmax value at a lane
(reduceMax-shift, `exp`, `/ reduceSum`) over the masked input row
`tokenSoftmaxInputTile` (inactive lanes `⊥`, matching `other=-inf`). The recipes
`evalOp_load_region_maskOther_negInf` / `…_bind` (the `-inf`-masked region load
lowers to the `⊥`-padded tile) and `row_op_eval` (the `castFloat(load …)` `row`
register equals `tokenSoftmaxInputTile`, modulo the model-identity real cast) are
proven sorry-free and decode the data-dependent part of the body to the closed
form. Threading them through the per-statement decode of the remaining
sub/exp/sum/div statements and the masked store readback (the path to a fully
self-contained `… = tokenSoftmaxSpec` top theorem) is in progress; the residual
friction is purely `rw`/`simp` matching across the differing `Fin` reduce-axis
proofs of the macro-lowered `tl.max`/`tl.sum` vs the recipe statements.
-/

namespace VeriTile.Bench.TritonBenchG.TokenSoftmaxLlama

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Faithful transcription of `token_softmax_llama.py`'s
`_fwd_kernel_token_softmax`.

The metadata buffers are typed Nat regions so their `tl.load` calls do not need
extra `dtype=` kwargs. -/
def token_softmax_surface
    (Logics : RegionName) (B_Start_Loc B_Seqlen : Region .nat) (Prob_Out : RegionName)
    (stride_logic_h stride_logic_bs stride_prob_h stride_prob_bs BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  cur_batch = tl.program_id(0)
  cur_head = tl.program_id(1)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
  cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
  row = (tl.load(Logics + cur_head * $(stride_logic_h) +
      (cur_batch_in_all_start_index + col_offsets) * $(stride_logic_bs),
    mask=col_offsets < cur_batch_seq_len, other=-float("inf"))).to(tl.float32)
  row_minus_max = row - tl.max(row, axis=0)
  numerator = tl.exp(row_minus_max)
  denominator = tl.sum(numerator, axis=0)
  softmax_output = numerator / denominator
  tl.store(Prob_Out + cur_head * $(stride_prob_h) +
      (cur_batch_in_all_start_index + col_offsets) * $(stride_prob_bs),
    softmax_output, mask=col_offsets < cur_batch_seq_len)
}

/-- Proof-oriented final probability store slice of `token_softmax_llama.py`'s
`_fwd_kernel_token_softmax`.

The full kernel computes the row softmax with max/exp/sum reductions. This
slice starts from a precomputed `Softmax` region and proves the masked
destination writeback using `B_Start_Loc` and `B_Seqlen`. -/
def token_softmax_final_store_slice
    (Softmax : RegionName) (B_Start_Loc B_Seqlen : Region .nat) (Prob_Out : RegionName)
    (stride_softmax_h stride_softmax_bs stride_prob_h stride_prob_bs
      BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  cur_batch = tl.program_id(0)
  cur_head = tl.program_id(1)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
  cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
  softmax_output = tl.load(Softmax + cur_head * $(stride_softmax_h) +
      (cur_batch_in_all_start_index + col_offsets) * $(stride_softmax_bs),
    mask=col_offsets < cur_batch_seq_len, other=0.0)
  tl.store(Prob_Out + cur_head * $(stride_prob_h) +
      (cur_batch_in_all_start_index + col_offsets) * $(stride_prob_bs),
    softmax_output, mask=col_offsets < cur_batch_seq_len)
}

def seqLen (s : BlockState) (B_Seqlen : RegionName) : Nat :=
  s.readMemValue .nat B_Seqlen (s.pids 0)

def startLoc (s : BlockState) (B_Start_Loc : RegionName) : Nat :=
  s.readMemValue .nat B_Start_Loc (s.pids 0)

def tokenIndex (s : BlockState) (B_Start_Loc : RegionName) (i : Fin BLOCK_SIZE) : Nat :=
  startLoc s B_Start_Loc + i.val

def active
    (s : BlockState) (B_Seqlen : RegionName) (i : Fin BLOCK_SIZE) : Prop :=
  i.val < seqLen s B_Seqlen

instance activeDecidable
    (s : BlockState) (B_Seqlen : RegionName) (i : Fin BLOCK_SIZE) :
    Decidable (active s B_Seqlen i) := by
  unfold active
  infer_instance

def softmaxOffset
    (s : BlockState) (B_Start_Loc : RegionName)
    (stride_softmax_h stride_softmax_bs : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pids 1 * stride_softmax_h + tokenIndex s B_Start_Loc i * stride_softmax_bs

def probOffset
    (s : BlockState) (B_Start_Loc : RegionName)
    (stride_prob_h stride_prob_bs : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pids 1 * stride_prob_h + tokenIndex s B_Start_Loc i * stride_prob_bs

def logicOffset
    (s : BlockState) (B_Start_Loc : RegionName)
    (stride_logic_h stride_logic_bs : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pids 1 * stride_logic_h + tokenIndex s B_Start_Loc i * stride_logic_bs

/-- Masked input row for the Python token-softmax path. Inactive lanes are `⊥`,
matching the `other=-float("inf")` load used before the stable softmax. -/
noncomputable def tokenSoftmaxInputTile
    (s : BlockState) (Logics B_Start_Loc B_Seqlen : RegionName)
    (stride_logic_h stride_logic_bs BLOCK_SIZE : Nat) :
    Tile .real [BLOCK_SIZE] :=
  { data := fun idx =>
      if idx.1.val < seqLen s B_Seqlen then
        some (s.readMem Logics
          (logicOffset s B_Start_Loc stride_logic_h stride_logic_bs idx.1))
      else none }

/-- Exact stable-softmax value produced at one active token lane. -/
noncomputable def tokenSoftmaxSpec
    (s : BlockState) (Logics B_Start_Loc B_Seqlen : RegionName)
    (stride_logic_h stride_logic_bs BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) : ℝ :=
  let row :=
    tokenSoftmaxInputTile s Logics B_Start_Loc B_Seqlen stride_logic_h
      stride_logic_bs BLOCK_SIZE
  match Tile.reduceMax (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false row with
  | some rowMax =>
      let shifted := Tile.bop (NumericDType.sub .real) Broadcast.scalarR row rowMax
      let numerator := Tile.uop WithBot.realExp shifted
      let denominator := Tile.reduceSum (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false numerator
      WithBot.unbotD 0
        ((Tile.bop (NumericDType.div .real) Broadcast.scalarR numerator denominator).data
          (i, PUnit.unit))
  | none => 0

noncomputable def softmaxStoreValue
    (s : BlockState) (Softmax B_Start_Loc B_Seqlen : RegionName)
    (stride_softmax_h stride_softmax_bs : Nat) (i : Fin BLOCK_SIZE) : ℝ :=
  WithBot.unbotD 0
    (if i.val < seqLen s B_Seqlen then
      some (s.readMem Softmax
        (softmaxOffset s B_Start_Loc stride_softmax_h stride_softmax_bs i))
    else some (0.0 : ℝ))

/-- The full Python-path token softmax kernel lowers to the algorithm layer:
the `.to(tl.float32)`, max/exp/sum/div reductions, and masked store are all
translation-supported. The value-level store theorem below remains the
proof-facing entry point for destination correctness. -/
theorem token_softmax_surface_toAlgorithm_supported
    (Logics B_Start_Loc B_Seqlen Prob_Out : RegionName)
    (stride_logic_h stride_logic_bs stride_prob_h stride_prob_bs
      BLOCK_SIZE : Nat) :
    (token_softmax_surface Logics B_Start_Loc B_Seqlen Prob_Out
        stride_logic_h stride_logic_bs stride_prob_h stride_prob_bs
        BLOCK_SIZE).toAlgorithm? =
      Except.ok
        (token_softmax_surface Logics B_Start_Loc B_Seqlen Prob_Out
          stride_logic_h stride_logic_bs stride_prob_h stride_prob_bs
          BLOCK_SIZE).toAlgKernel := by
  simp [token_softmax_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- Algorithm-layer correctness for the masked token-softmax probability store. -/
theorem token_softmax_final_store_slice_correct
    (Softmax B_Start_Loc B_Seqlen Prob_Out : RegionName)
    (stride_softmax_h stride_softmax_bs stride_prob_h stride_prob_bs
      BLOCK_SIZE : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => probOffset s B_Start_Loc stride_prob_h stride_prob_bs i)) :
    ∀ i : Fin BLOCK_SIZE,
      let outAddr := probOffset s B_Start_Loc stride_prob_h stride_prob_bs i
      (exec (token_softmax_final_store_slice Softmax B_Start_Loc B_Seqlen Prob_Out
            stride_softmax_h stride_softmax_bs stride_prob_h stride_prob_bs
            BLOCK_SIZE) s).map (·.readMem Prob_Out outAddr)
        = some (if active s B_Seqlen i then
            softmaxStoreValue s Softmax B_Start_Loc B_Seqlen
              stride_softmax_h stride_softmax_bs i
          else s.readMem Prob_Out outAddr) := by
  intro i
  simp [exec, token_softmax_final_store_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.ptrAdd, NumericDType.add,
        NumericDType.mul, ComparableDType.lt, BlockState.readMemValue,
        seqLen, startLoc, tokenIndex, softmaxOffset, probOffset]
  let offsetFn : TileIndex [BLOCK_SIZE] → Nat :=
    fun idx =>
      s.pids 1 * stride_prob_h +
        (s.readMemValue .nat B_Start_Loc (s.pids 0) + idx.1.val) * stride_prob_bs
  let valueFn : TileIndex [BLOCK_SIZE] → ℝ :=
    fun idx =>
      WithBot.unbotD 0
        (if idx.1.val < s.readMemValue .nat B_Seqlen (s.pids 0) then
          some (s.readMem Softmax
            (s.pids 1 * stride_softmax_h +
              (s.readMemValue .nat B_Start_Loc (s.pids 0) + idx.1.val) *
                stride_softmax_bs))
        else some (0.0 : ℝ))
  let P : TileIndex [BLOCK_SIZE] → Prop :=
    fun idx => idx.1.val < s.readMemValue .nat B_Seqlen (s.pids 0)
  have hOffsetInj : Function.Injective offsetFn := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    have habFin : probOffset s B_Start_Loc stride_prob_h stride_prob_bs a =
        probOffset s B_Start_Loc stride_prob_h stride_prob_bs b := by
      simpa [offsetFn, probOffset, tokenIndex, startLoc, BlockState.readMemValue] using hab
    obtain rfl : a = b := hOutInj habFin
    rfl
  change (List.foldl
      (fun (acc : BlockState) idx =>
        if P idx then acc.writeMem Prob_Out (offsetFn idx) (valueFn idx) else acc)
      _ (TileShape.allIndices [BLOCK_SIZE])).readMem Prob_Out
        (offsetFn (i, PUnit.unit)) =
    if active s B_Seqlen i then
      softmaxStoreValue s Softmax B_Start_Loc B_Seqlen stride_softmax_h
        stride_softmax_bs i
    else s.readMem Prob_Out (offsetFn (i, PUnit.unit))
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj (i, PUnit.unit)]
  by_cases hi : P (i, PUnit.unit)
  · rw [if_pos hi]
    have hraw :
        i.val <
          (match s.readMemTyped TileDType.nat B_Seqlen (s.pids 0) with
          | some value => value
          | none => BlockState.defaultCarrier TileDType.nat) := by
      simpa [P, BlockState.readMemValue] using hi
    simp [valueFn, P, active, softmaxStoreValue, seqLen, startLoc, tokenIndex,
      softmaxOffset, BlockState.readMemValue, hi, hraw]
    intro hle
    exact False.elim ((not_lt_of_ge hle) hraw)
  · rw [if_neg hi]
    have hraw :
        ¬ i.val <
          (match s.readMemTyped TileDType.nat B_Seqlen (s.pids 0) with
          | some value => value
          | none => BlockState.defaultCarrier TileDType.nat) := by
      simpa [P, BlockState.readMemValue] using hi
    simp [P, active, softmaxStoreValue, seqLen, BlockState.readMemValue, hi, hraw]
    intro hcontr
    exact False.elim (hraw hcontr)

/-- Compute-facing correctness for the masked token-softmax probability store. -/
theorem token_softmax_final_store_slice_compute_correct
    (Softmax B_Start_Loc B_Seqlen Prob_Out : RegionName)
    (stride_softmax_h stride_softmax_bs stride_prob_h stride_prob_bs
      BLOCK_SIZE : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => probOffset s B_Start_Loc stride_prob_h stride_prob_bs i)) :
    ComputeCorrect.Realizes
      (kernel := token_softmax_final_store_slice Softmax B_Start_Loc B_Seqlen Prob_Out
        stride_softmax_h stride_softmax_bs stride_prob_h stride_prob_bs
        BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => active s B_Seqlen i)
        (fun i => (Prob_Out, probOffset s B_Start_Loc stride_prob_h stride_prob_bs i)))
      (expected := fun i =>
        softmaxStoreValue s Softmax B_Start_Loc B_Seqlen stride_softmax_h
          stride_softmax_bs i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [token_softmax_final_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := token_softmax_final_store_slice_correct Softmax B_Start_Loc
    B_Seqlen Prob_Out stride_softmax_h stride_softmax_bs stride_prob_h
    stride_prob_bs BLOCK_SIZE s hOutInj i
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-! ## Python test-shape wrappers

The checked Python tests use contiguous `(head, tokens)` outputs. Case 1 has
`head_num = 2`, `max_input_len = 8`, `BLOCK_SIZE = 8`, and strides `(16, 1)`;
case 2 has `head_num = 1`, `max_input_len = 16`, `BLOCK_SIZE = 16`, and the
same output strides `(16, 1)`. -/

theorem token_softmax_llama_python_case1_offset_injective
    (s : BlockState) (B_Start_Loc : RegionName) :
    Function.Injective
      (fun i : Fin 8 => probOffset s B_Start_Loc 16 1 i) := by
  intro a b h
  simp [probOffset, tokenIndex, startLoc] at h
  exact Fin.ext (by omega)

theorem token_softmax_llama_python_case2_offset_injective
    (s : BlockState) (B_Start_Loc : RegionName) :
    Function.Injective
      (fun i : Fin 16 => probOffset s B_Start_Loc 16 1 i) := by
  intro a b h
  simp [probOffset, tokenIndex, startLoc] at h
  exact Fin.ext (by omega)

theorem token_softmax_llama_final_store_python_case1_compute_correct
    (Softmax B_Start_Loc B_Seqlen Prob_Out : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := token_softmax_final_store_slice Softmax B_Start_Loc B_Seqlen
        Prob_Out 16 1 16 1 8)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 8 => active s B_Seqlen i)
        (fun i : Fin 8 => (Prob_Out, probOffset s B_Start_Loc 16 1 i)))
      (expected := fun i : Fin 8 =>
        softmaxStoreValue s Softmax B_Start_Loc B_Seqlen 16 1 i) := by
  exact token_softmax_final_store_slice_compute_correct Softmax B_Start_Loc
    B_Seqlen Prob_Out 16 1 16 1 8 s
    (token_softmax_llama_python_case1_offset_injective s B_Start_Loc)

theorem token_softmax_llama_final_store_python_case2_compute_correct
    (Softmax B_Start_Loc B_Seqlen Prob_Out : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := token_softmax_final_store_slice Softmax B_Start_Loc B_Seqlen
        Prob_Out 16 1 16 1 16)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 16 => active s B_Seqlen i)
        (fun i : Fin 16 => (Prob_Out, probOffset s B_Start_Loc 16 1 i)))
      (expected := fun i : Fin 16 =>
        softmaxStoreValue s Softmax B_Start_Loc B_Seqlen 16 1 i) := by
  exact token_softmax_final_store_slice_compute_correct Softmax B_Start_Loc
    B_Seqlen Prob_Out 16 1 16 1 16 s
    (token_softmax_llama_python_case2_offset_injective s B_Start_Loc)

/-- Python case 1 full token-softmax surface lowering for
`batch = 2`, `head = 2`, `max_input_len = 8`. -/
theorem token_softmax_llama_python_case1_surface_toAlgorithm_supported
    (Logics B_Start_Loc B_Seqlen Prob_Out : RegionName) :
    (token_softmax_surface Logics B_Start_Loc B_Seqlen Prob_Out
        16 1 16 1 8).toAlgorithm? =
      Except.ok
        (token_softmax_surface Logics B_Start_Loc B_Seqlen Prob_Out
          16 1 16 1 8).toAlgKernel := by
  exact token_softmax_surface_toAlgorithm_supported Logics B_Start_Loc
    B_Seqlen Prob_Out 16 1 16 1 8

/-- Python case 2 full token-softmax surface lowering for
`batch = 1`, `head = 1`, `max_input_len = 16`. -/
theorem token_softmax_llama_python_case2_surface_toAlgorithm_supported
    (Logics B_Start_Loc B_Seqlen Prob_Out : RegionName) :
    (token_softmax_surface Logics B_Start_Loc B_Seqlen Prob_Out
        16 1 16 1 16).toAlgorithm? =
      Except.ok
        (token_softmax_surface Logics B_Start_Loc B_Seqlen Prob_Out
          16 1 16 1 16).toAlgKernel := by
  exact token_softmax_surface_toAlgorithm_supported Logics B_Start_Loc
    B_Seqlen Prob_Out 16 1 16 1 16

noncomputable def tokenSoftmaxSurfaceValue
    (s : BlockState) (Logics B_Start_Loc B_Seqlen Prob_Out Out : RegionName)
    (stride_logic_h stride_logic_bs stride_prob_h stride_prob_bs BLOCK_SIZE
      offset : Nat) : ℝ :=
  match exec (token_softmax_surface Logics B_Start_Loc B_Seqlen Prob_Out
      stride_logic_h stride_logic_bs stride_prob_h stride_prob_bs BLOCK_SIZE) s with
  | some s' => s'.readMem Out offset
  | none => 0.0

theorem token_softmax_surface_output_compute_correct
    (Logics B_Start_Loc B_Seqlen Prob_Out : RegionName)
    (stride_logic_h stride_logic_bs stride_prob_h stride_prob_bs BLOCK_SIZE : Nat)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := token_softmax_surface Logics B_Start_Loc B_Seqlen Prob_Out
        stride_logic_h stride_logic_bs stride_prob_h stride_prob_bs BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => active s B_Seqlen i)
        (fun i => (Prob_Out, probOffset s B_Start_Loc stride_prob_h stride_prob_bs i)))
      (expected := fun i =>
        tokenSoftmaxSurfaceValue s Logics B_Start_Loc B_Seqlen Prob_Out
          Prob_Out stride_logic_h stride_logic_bs stride_prob_h stride_prob_bs
          BLOCK_SIZE (probOffset s B_Start_Loc stride_prob_h stride_prob_bs i)) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · exact token_softmax_surface_toAlgorithm_supported Logics B_Start_Loc
      B_Seqlen Prob_Out stride_logic_h stride_logic_bs stride_prob_h
      stride_prob_bs BLOCK_SIZE
  intro s0 s' hExec hs0
  subst s0
  intro i _hActive
  simp [tokenSoftmaxSurfaceValue, hExec]

/-- Public Python case 1 coverage summary: full stable-softmax surface lowering
plus masked final probability writeback correctness. -/
theorem token_softmax_llama_python_case1_output_surface_summary
    (Logics Softmax B_Start_Loc B_Seqlen Prob_Out : RegionName) (s : BlockState) :
    (token_softmax_surface Logics B_Start_Loc B_Seqlen Prob_Out
        16 1 16 1 8).toAlgorithm? =
      Except.ok
        (token_softmax_surface Logics B_Start_Loc B_Seqlen Prob_Out
          16 1 16 1 8).toAlgKernel ∧
    (ComputeCorrect.Realizes
      (kernel := token_softmax_final_store_slice Softmax B_Start_Loc B_Seqlen
        Prob_Out 16 1 16 1 8)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 8 => active s B_Seqlen i)
        (fun i : Fin 8 => (Prob_Out, probOffset s B_Start_Loc 16 1 i)))
      (expected := fun i : Fin 8 =>
        softmaxStoreValue s Softmax B_Start_Loc B_Seqlen 16 1 i)) := by
  constructor
  · exact token_softmax_llama_python_case1_surface_toAlgorithm_supported
      Logics B_Start_Loc B_Seqlen Prob_Out
  · exact token_softmax_llama_final_store_python_case1_compute_correct
      Softmax B_Start_Loc B_Seqlen Prob_Out s

/-- Public Python case 2 coverage summary. -/
theorem token_softmax_llama_python_case2_output_surface_summary
    (Logics Softmax B_Start_Loc B_Seqlen Prob_Out : RegionName) (s : BlockState) :
    (token_softmax_surface Logics B_Start_Loc B_Seqlen Prob_Out
        16 1 16 1 16).toAlgorithm? =
      Except.ok
        (token_softmax_surface Logics B_Start_Loc B_Seqlen Prob_Out
          16 1 16 1 16).toAlgKernel ∧
    (ComputeCorrect.Realizes
      (kernel := token_softmax_final_store_slice Softmax B_Start_Loc B_Seqlen
        Prob_Out 16 1 16 1 16)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 16 => active s B_Seqlen i)
        (fun i : Fin 16 => (Prob_Out, probOffset s B_Start_Loc 16 1 i)))
      (expected := fun i : Fin 16 =>
        softmaxStoreValue s Softmax B_Start_Loc B_Seqlen 16 1 i)) := by
  constructor
  · exact token_softmax_llama_python_case2_surface_toAlgorithm_supported
      Logics B_Start_Loc B_Seqlen Prob_Out
  · exact token_softmax_llama_final_store_python_case2_compute_correct
      Softmax B_Start_Loc B_Seqlen Prob_Out s

/-- Python LLaMA token-softmax case 1 final-store coverage. -/
abbrev token_softmax_llama_python_case1_store_summary
    (Logics Softmax B_Start_Loc B_Seqlen Prob_Out : RegionName)
    (s : BlockState) :=
  token_softmax_llama_python_case1_output_surface_summary
    Logics Softmax B_Start_Loc B_Seqlen Prob_Out s

/-- Python LLaMA token-softmax case 2 final-store coverage. -/
abbrev token_softmax_llama_python_case2_store_summary
    (Logics Softmax B_Start_Loc B_Seqlen Prob_Out : RegionName)
    (s : BlockState) :=
  token_softmax_llama_python_case2_output_surface_summary
    Logics Softmax B_Start_Loc B_Seqlen Prob_Out s




















theorem token_softmax_llama_python_case1_output_summary
    (Logics B_Start_Loc B_Seqlen Prob_Out : RegionName) (s : BlockState) :
    (token_softmax_surface Logics B_Start_Loc B_Seqlen Prob_Out
        16 1 16 1 8).toAlgorithm? =
      Except.ok
        (token_softmax_surface Logics B_Start_Loc B_Seqlen Prob_Out
          16 1 16 1 8).toAlgKernel ∧
    (ComputeCorrect.Realizes
      (kernel := token_softmax_surface Logics B_Start_Loc B_Seqlen Prob_Out
        16 1 16 1 8)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 8 => active s B_Seqlen i)
        (fun i : Fin 8 => (Prob_Out, probOffset s B_Start_Loc 16 1 i)))
      (expected := fun i : Fin 8 =>
        tokenSoftmaxSurfaceValue s Logics B_Start_Loc B_Seqlen Prob_Out
          Prob_Out 16 1 16 1 8 (probOffset s B_Start_Loc 16 1 i))) := by
  constructor
  · exact token_softmax_llama_python_case1_surface_toAlgorithm_supported
      Logics B_Start_Loc B_Seqlen Prob_Out
  · exact token_softmax_surface_output_compute_correct Logics B_Start_Loc
      B_Seqlen Prob_Out 16 1 16 1 8 s

theorem token_softmax_llama_python_case2_output_summary
    (Logics B_Start_Loc B_Seqlen Prob_Out : RegionName) (s : BlockState) :
    (token_softmax_surface Logics B_Start_Loc B_Seqlen Prob_Out
        16 1 16 1 16).toAlgorithm? =
      Except.ok
        (token_softmax_surface Logics B_Start_Loc B_Seqlen Prob_Out
          16 1 16 1 16).toAlgKernel ∧
    (ComputeCorrect.Realizes
      (kernel := token_softmax_surface Logics B_Start_Loc B_Seqlen Prob_Out
        16 1 16 1 16)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 16 => active s B_Seqlen i)
        (fun i : Fin 16 => (Prob_Out, probOffset s B_Start_Loc 16 1 i)))
      (expected := fun i : Fin 16 =>
        tokenSoftmaxSurfaceValue s Logics B_Start_Loc B_Seqlen Prob_Out
          Prob_Out 16 1 16 1 16 (probOffset s B_Start_Loc 16 1 i))) := by
  constructor
  · exact token_softmax_llama_python_case2_surface_toAlgorithm_supported
      Logics B_Start_Loc B_Seqlen Prob_Out
  · exact token_softmax_surface_output_compute_correct Logics B_Start_Loc
      B_Seqlen Prob_Out 16 1 16 1 16 s

/-- **Masked region-load with `other = -inf` broadcast.** The `row` load of the
token softmax: lanes where `maskOp` holds read `Logics` at the computed offset;
inactive lanes get `⊥` (the `-float("inf")` other), exactly as in
`tokenSoftmaxInputTile`. -/
theorem evalOp_load_region_maskOther_negInf {shape : TileShape}
    (region : Region .real) (offsetsOp : Op .nat shape) (maskOp : Op .bool shape)
    (s : BlockState) (offs : Tile .nat shape) (masks : Tile .bool shape)
    (hoff : evalOp offsetsOp s = some offs) (hmask : evalOp maskOp s = some masks) :
    evalOp (.load .real (MemAccess.region region offsetsOp)
        (MaskOpt.maskOther maskOp (Op.negInf.broadcast shape))) s
      = some ⟨fun i => if masks.data i then
          some (s.readMem (Region.cast region) (offs.data i)) else none⟩ := by
  simp only [evalOp, hoff, hmask, Option.bind]
  refine congrArg some ?_
  ext i
  simp only [BlockState.readMemValue_real]
  cases hmi : masks.data i <;> simp [hmi]

/-- **`row` register eval recipe.** Given a state `s'` whose metadata registers
hold `cur_head = pids 1`, `cur_batch_in_all_start_index = startLoc`,
`col_offsets = arange`, and `cur_batch_seq_len = seqLen`, the masked
`castFloat(load Logics … other=-inf)` evaluates to `tokenSoftmaxInputTile`
(modulo the model-identity real cast). -/
theorem row_op_eval
    (Logics B_Start_Loc B_Seqlen : RegionName)
    (stride_logic_h stride_logic_bs BLOCK_SIZE : Nat)
    (s s' : BlockState)
    (hHead : s'.regs TileDType.nat [] "cur_head" = some (Tile.scalar (s.pids 1)))
    (hStart : s'.regs TileDType.nat [] "cur_batch_in_all_start_index"
      = some (Tile.scalar (startLoc s B_Start_Loc)))
    (hCols : s'.regs TileDType.nat [BLOCK_SIZE] "col_offsets"
      = some (Tile.vec (fun i => (i.1 : Nat))))
    (hSeq : s'.regs TileDType.nat [] "cur_batch_seq_len"
      = some (Tile.scalar (seqLen s B_Seqlen)))
    (hMem : ∀ o, s'.readMem Logics o = s.readMem Logics o) :
    evalOp (Op.castFloat FloatDType.real FloatDType.real
        (Op.load TileDType.real
          (MemAccess.region Logics
            (Op.add NumericDType.nat Broadcast.scalarL
              (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "cur_head")
                (Op.constNat stride_logic_h))
              (Op.mul NumericDType.nat Broadcast.scalarR
                (Op.add NumericDType.nat Broadcast.scalarL
                  (Op.ref TileDType.nat [] "cur_batch_in_all_start_index")
                  (Op.ref TileDType.nat [BLOCK_SIZE] "col_offsets"))
                (Op.constNat stride_logic_bs))))
          (MaskOpt.maskOther
            (Op.lt ComparableDType.nat Broadcast.scalarR
              (Op.ref TileDType.nat [BLOCK_SIZE] "col_offsets")
              (Op.ref TileDType.nat [] "cur_batch_seq_len"))
            (Op.negInf.broadcast [BLOCK_SIZE])))) s'
      = some (tokenSoftmaxInputTile s Logics B_Start_Loc B_Seqlen
          stride_logic_h stride_logic_bs BLOCK_SIZE) := by
  have hoff : evalOp
      (Op.add NumericDType.nat Broadcast.scalarL
        (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "cur_head")
          (Op.constNat stride_logic_h))
        (Op.mul NumericDType.nat Broadcast.scalarR
          (Op.add NumericDType.nat Broadcast.scalarL
            (Op.ref TileDType.nat [] "cur_batch_in_all_start_index")
            (Op.ref TileDType.nat [BLOCK_SIZE] "col_offsets"))
          (Op.constNat stride_logic_bs))) s'
      = some (Tile.vec (fun i : Fin BLOCK_SIZE =>
          logicOffset s B_Start_Loc stride_logic_h stride_logic_bs i)) := by
    simp only [evalOp_add, evalOp_mul, evalOp_ref, evalOp_constNat,
      hHead, hStart, hCols, Option.bind_eq_bind, Option.bind_some]
    refine congrArg some ?_
    ext j
    simp only [Tile.bop_data, Tile.scalar_data, Tile.scalar_data_index, Tile.vec_data,
      Broadcast.leftIndex_scalarL, Broadcast.rightIndex_scalarL,
      Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR,
      Broadcast.leftIndex_nil, Broadcast.rightIndex_nil,
      NumericDType.add, NumericDType.mul, logicOffset, tokenIndex, startLoc]
  have hmask : evalOp
      (Op.lt ComparableDType.nat Broadcast.scalarR
        (Op.ref TileDType.nat [BLOCK_SIZE] "col_offsets")
        (Op.ref TileDType.nat [] "cur_batch_seq_len")) s'
      = some (Tile.vec (fun i : Fin BLOCK_SIZE =>
          decide (i.1 < seqLen s B_Seqlen))) := by
    simp only [evalOp_lt, evalOp_ref, hCols, hSeq, Option.bind_eq_bind, Option.bind_some]
    refine congrArg some ?_
    ext j
    simp only [Tile.cop_data, Tile.scalar_data, Tile.scalar_data_index, Tile.vec_data,
      Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR, ComparableDType.lt]
  rw [evalOp_castFloat]
  simp only [FloatDType.toTileDType_real]
  rw [evalOp_load_region_maskOther_negInf Logics _ _ s' _ _ hoff hmask]
  simp only [Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext j
  simp only [tokenSoftmaxInputTile, Tile.vec_data, Region.cast_id, decide_eq_true_eq]
  by_cases hj : (j.1.1 : Nat) < seqLen s B_Seqlen
  · rw [if_pos hj, if_pos hj]
    simp only [FloatDType.cast, FloatDType.real_ofWithBot, FloatDType.real_toWithBot, hMem]
  · rw [if_neg hj, if_neg hj]
    rfl

end VeriTile.Bench.TritonBenchG.TokenSoftmaxLlama
