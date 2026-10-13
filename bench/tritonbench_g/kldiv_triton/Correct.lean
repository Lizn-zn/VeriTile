import bench.tritonbench_g.kldiv_triton.Proofs.LegacyReal
import bench.tritonbench_g.kldiv_triton.Proofs.Contracts
import bench.tritonbench_g.kldiv_triton.Proofs.Precision

namespace VeriTile.Bench.TritonBenchG.KldivTriton
open VeriTile.Triton
open scoped VeriTile.Triton.StreamEmitMasked2DKernelIO₂

/-- Compute-facing correctness for the default backward kernel under the
Python-tested single-chunk regime `0 < n_cols ≤ BLOCK_SIZE`. -/
specification kldiv_backward_default_compute_correct
    (input_ptr target_ptr : RegionName)
    (input_stride target_stride n_cols BLOCK_SIZE : Nat)
    (s : BlockState)
    (hBS : 0 < BLOCK_SIZE)
    (hLen : n_cols ≤ BLOCK_SIZE)
    (hLenPos : 0 < n_cols)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => linearOffset s input_stride i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := kldiv_backward_default input_ptr target_ptr
        input_stride target_stride n_cols BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < n_cols)
        (fun i => (input_ptr, linearOffset s input_stride i)))
      (expected := fun i => defaultSpec s target_ptr target_stride i) := by
  exact kldiv_backward_default_compute_correct_bridge input_ptr target_ptr input_stride target_stride n_cols BLOCK_SIZE s hBS hLen hLenPos hOutInj

/-- Compute-facing correctness for the log-target backward kernel under the
Python-tested single-chunk regime `0 < n_cols ≤ BLOCK_SIZE`. -/
specification kldiv_backward_log_target_compute_correct
    (input_ptr target_ptr : RegionName)
    (input_stride target_stride n_cols BLOCK_SIZE : Nat)
    (s : BlockState)
    (hBS : 0 < BLOCK_SIZE)
    (hLen : n_cols ≤ BLOCK_SIZE)
    (hLenPos : 0 < n_cols)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => linearOffset s input_stride i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := kldiv_backward_log_target input_ptr target_ptr
        input_stride target_stride n_cols BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < n_cols)
        (fun i => (input_ptr, linearOffset s input_stride i)))
      (expected := fun i => logTargetSpec s target_ptr target_stride i) := by
  exact kldiv_backward_log_target_compute_correct_bridge input_ptr target_ptr input_stride target_stride n_cols BLOCK_SIZE s hBS hLen hLenPos hOutInj

/-- Compute-facing correctness for the forward `log_target=False`,
`reduction=0` kernel under the Python-tested single-chunk regime
`0 < n_cols ≤ BLOCK_SIZE`. -/
specification kldiv_forward_default_none_compute_correct
    (y_ptr gt_ptr loss_ptr : RegionName)
    (y_stride gt_stride loss_stride n_cols BLOCK_SIZE : Nat)
    (s : BlockState)
    (hBS : 0 < BLOCK_SIZE)
    (hLen : n_cols ≤ BLOCK_SIZE)
    (hLenPos : 0 < n_cols)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => linearOffset s loss_stride i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := kldiv_forward_default_none y_ptr gt_ptr loss_ptr
        y_stride gt_stride loss_stride n_cols BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < n_cols)
        (fun i => (loss_ptr, linearOffset s loss_stride i)))
      (expected := fun i =>
        forwardDefaultSpec s y_ptr gt_ptr y_stride gt_stride i) := by
  exact kldiv_forward_default_none_compute_correct_bridge y_ptr gt_ptr loss_ptr y_stride gt_stride loss_stride n_cols BLOCK_SIZE s hBS hLen hLenPos hOutInj

/-- Compute-facing correctness for the forward log-target kernel under the
Python-tested single-chunk regime `0 < n_cols ≤ BLOCK_SIZE`. -/
specification kldiv_forward_log_target_none_compute_correct
    (y_ptr gt_ptr loss_ptr : RegionName)
    (y_stride gt_stride loss_stride n_cols BLOCK_SIZE : Nat)
    (s : BlockState)
    (hBS : 0 < BLOCK_SIZE)
    (hLen : n_cols ≤ BLOCK_SIZE)
    (hLenPos : 0 < n_cols)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => linearOffset s loss_stride i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := kldiv_forward_log_target_none y_ptr gt_ptr loss_ptr
        y_stride gt_stride loss_stride n_cols BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < n_cols)
        (fun i => (loss_ptr, linearOffset s loss_stride i)))
      (expected := fun i =>
        forwardLogTargetSpec s y_ptr gt_ptr y_stride gt_stride i) := by
  exact kldiv_forward_log_target_none_compute_correct_bridge y_ptr gt_ptr loss_ptr y_stride gt_stride loss_stride n_cols BLOCK_SIZE s hBS hLen hLenPos hOutInj

/-- **The `⊨[R]` streaming headline (wave-5 S3 per-step emit genre).** For
every rounding model `R`, the faithful `kldiv_forward_default_none` surface
(the `log_target = False`, `reduction = 0` (None) forward branch) implements,
on its `StreamEmitMasked2DKernelIO₂` signature, the **ideal ℝ elementwise
forward KL divergence** over the streamed tiles: emitted window `(t, j)` holds
`y_true[t,j] · (log y_true[t,j] − y[t,j])`, with `y` the log-space prediction
stream and `y_true` the ground-truth stream. The spec `f` is exact real
arithmetic — this port has **no `eps` clamp** (unlike the `kldiv_ops` port).

Rounding story: the kernel has **zero rounding events**. Both `tl.load`s and
the in-loop `tl.store` are at `.real` (the lowered store is
`Stmt.store TileDType.real … (MaskOpt.mask …)` — no cast on the stored value),
and the kernel contains no `castFloat` at all; the only casts are the
`int`-flavoured `pid = tl.program_id(0).to(tl.int64)` pointer arithmetic, which
is exact. So the skin's boundary quantization degenerates: at the declared grid
`outDType := .real` the readback contract's `R.round .real` is the identity by
the model's defining `round_real`, and the `.real` in-loop stores are exact
under `execR R`. The ∀-`R` face therefore holds via the `RoundingModel` `.real`
identity fields, **not** as a `.triv` special case.

Layer map: the prologue and the loop body are cast-free
(`klPrefix_castFree` / `klLossBody_castFree`), so under `execR R` they collapse
verbatim onto the exact stepper; the loop is driven by the `forRange`
invariant `klInv` (`kl_preLoop` + `klInv_step` + `kl_runR`), the safety walk by
`Stmt.forRangeTraceSafeR_inv` over the same invariant (`kl_lossBodySafeR` +
`kldiv_traceSafeR`), and the stream indices are matched to the memory-level
closed form by `klStreamSpec_eq_klCell`.

**Scope: this face is dimension-general — the full multi-chunk stream.** `T`
is `⌈n_cols / BLOCK_SIZE⌉`, so the loop is proved for an arbitrary number of
iterations; there is **no** `n_cols ≤ BLOCK_SIZE` regime hypothesis and no
`0 < n_cols` hypothesis. (The retained exact surfaces below/above are the
narrower single-chunk statements; see "Relation to the exact surfaces".)

Every hypothesis is truth-forced:

* `hBS : 0 < BLOCK_SIZE` — the loop steps by `BLOCK_SIZE`
  (`range(0, n_cols, BLOCK_SIZE)`); at `BLOCK_SIZE = 0` with `n_cols > 0` the
  `forRange` fold does not advance, `execR` cannot terminate the way the
  invariant requires, and the step index `i / BLOCK_SIZE` is meaningless. It
  holds for every real launch (the host picks
  `BLOCK_SIZE = min(16384, next_power_of_2(S))`).
* `hly : loss_ptr ≠ y_ptr`, `hlg : loss_ptr ≠ gt_ptr` — the store sits
  **inside** the loop, so iteration `t + 1` loads *after* iteration `t` has
  already stored. If the loss buffer aliased either input buffer, later chunks
  would re-read overwritten values and the closed form would be false. (In the
  single-chunk regime these are vacuous; they are exactly the price of the
  general multi-chunk face.) The host allocates `output_tensor` fresh, so both
  hold for every real launch.

Relation to the exact surfaces: the file's exact headline
`kldiv_forward_default_none_compute_correct`
(`Realizes_without_Rounding`, scoped to the Python-tested single-chunk regime
`0 < n_cols ≤ BLOCK_SIZE`, with the offset-injectivity side condition) is
retained unchanged; this `⊨[R]` face restates the same KL content on the
streaming emit skin, for every `R` at once and for an arbitrary number of
chunks (at the `.real` grid the two faces carry the same exact cell on the
single-chunk overlap). Both faces are kept per the rounding-as-default
doctrine. The `log_target = True` branch and the `reduction ≠ 0` reduction
modes keep their existing (exact, partial) treatment — the reduction-mode
surface `kldiv_forward_surface` is still only proved to lower. -/
specification kldiv_forward_default_none_io_correctness (R : RoundingModel)
    (y_ptr gt_ptr loss_ptr : RegionName)
    (y_stride gt_stride loss_stride n_cols BLOCK_SIZE : Nat)
    (hBS : 0 < BLOCK_SIZE)
    (hly : loss_ptr ≠ y_ptr) (hlg : loss_ptr ≠ gt_ptr) :
    kldivForwardDefaultNoneIO y_ptr gt_ptr loss_ptr y_stride gt_stride loss_stride
        n_cols BLOCK_SIZE ⊨[R]
      fun _ _ xs ys t j => klStreamSpec n_cols BLOCK_SIZE xs ys t j := by
  exact kldiv_forward_default_none_io_correctness_bridge R y_ptr gt_ptr loss_ptr y_stride gt_stride loss_stride n_cols BLOCK_SIZE hBS hly hlg


/-- Independent full forward loss formula, successful flat IO and exact raw-cell frame.
Every nonzero reduction mode writes one sum per executed chunk; N=0 writes nothing.
Both source variants are proved separately through their mathematical projection. -/
specification typed_forward_real_correct
    (pred : TypedSource.Storage) (target : TypedSource.MathStorage)
    (output : TypedSource.Storage) (v : Bool) (Y T O : RegionName)
    (sy st so N B : Nat) (flag : Bool) (reduction : Nat)
    (hB : 0 < B) (hYO : Y ≠ O) (hTO : T ≠ O) :
    Contracts.ForwardIO
      (TypedSource.forward pred target output v Y T O sy st so N B flag reduction)
      Y T O sy st so N B flag reduction :=
  Contracts.forward_io pred target output v Y T O sy st so N B flag reduction hB hYO hTO

/-- Full default-backward gradients and flat-memory frame for the source-supported
weak-literal target types. Output storage remains independently selectable. -/
specification typed_backward_default_real_correct
    (target output : TypedSource.Storage) (v : Bool) (T O : RegionName)
    (st so N B : Nat) (hSource : TypedSource.BackwardSourceSupported target)
    (hB : 0 < B) (hTO : T ≠ O) :
    Contracts.BackwardIO (TypedSource.backwardDefault target output v T O st so N B)
      T O st so N Bool.false :=
  Contracts.backward_default_io target output v T O st so N B hB hTO

/-- Full logarithmic-target backward gradients, successful flat execution and raw frame. -/
specification typed_backward_log_real_correct
    (target : TypedSource.MathStorage) (output : TypedSource.Storage) (v : Bool)
    (T O : RegionName) (st so N B : Nat) (hB : 0 < B) (hTO : T ≠ O) :
    Contracts.BackwardIO (TypedSource.backwardLog target output v T O st so N B)
      T O st so N Bool.true :=
  Contracts.backward_log_io target output v T O st so N B hB hTO

#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.TritonBenchG.KldivTriton
