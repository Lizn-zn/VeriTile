/- Countermodels for the algebraic coverage of the frozen admission table.
These are interpretations of scalar symbols, not simulations of IEEE values
or numerical rejection results. They show which conclusions the admitted
equations alone do not force. No assumption is added to a kernel proof. -/
import VeriTile.Triton.Float.ReportedAdmission
import VeriTile.Meta.StatementAudit
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring

namespace FPAdmissionCoverageTests
open VeriTile Triton

def reportedIDs : List String := (FP.ReportedAdmission.all.map (·.ruleID)).eraseDups

/-- Keep the countermodel's coverage tied to the actual accepted table. A new
admitted family changes this obligation instead of being silently ignored. -/
theorem reported_ids : reportedIDs =
    ["ADD-COMMUTE", "MUL-COMMUTE", "ROUND-IDEM", "BF16-WIDEN-RETURN",
     "ADD-ASSOC", "MUL-ASSOC", "MUL-DISTRIB", "FMA-CONTRACT", "CANCEL",
     "DIV-RCP", "CAST-MOVE", "CAST-REMOVE", "ACC-WIDEN"] := by decide

/-- Every floating format is interpreted by the same carrier in these
countermodels. Casts are identities, including the intermediate cast in the
cast-removal law. These interpretations do not assert IEEE behavior. -/
structure ScalarModel where
  add : ℚ → ℚ → ℚ
  sub : ℚ → ℚ → ℚ
  mul : ℚ → ℚ → ℚ
  divRN : ℚ → ℚ → ℚ
  divPlain : ℚ → ℚ → ℚ
  fma : ℚ → ℚ → ℚ → ℚ
  cast : ℚ → ℚ
  exp : ℚ → ℚ
  log : ℚ → ℚ
  sqrt : ℚ → ℚ

/-- The scalar expression pairs in the report. The identity-cast models below
validate all three precision instances whenever a family was admitted.
Ordinary division remains distinct from the tested `tl.div_rn` intrinsic. -/
def LawFor (M : ScalarModel) : String → Prop
  | "ADD-COMMUTE" => ∀ a b, M.cast (M.add a b) = M.cast (M.add b a)
  | "MUL-COMMUTE" => ∀ a b, M.cast (M.mul a b) = M.cast (M.mul b a)
  | "ROUND-IDEM" => ∀ a, M.cast (M.cast a) = M.cast a
  | "BF16-WIDEN-RETURN" => ∀ a, M.cast (M.cast (M.cast a)) = M.cast a
  | "ADD-ASSOC" => ∀ a b c,
      M.cast (M.add (M.cast (M.add a b)) c) = M.cast (M.add a (M.cast (M.add b c)))
  | "MUL-ASSOC" => ∀ a b c,
      M.cast (M.mul (M.cast (M.mul a b)) c) = M.cast (M.mul a (M.cast (M.mul b c)))
  | "MUL-DISTRIB" => ∀ a b c,
      M.cast (M.mul a (M.cast (M.add b c))) =
        M.cast (M.add (M.cast (M.mul a b)) (M.cast (M.mul a c)))
  | "FMA-CONTRACT" => ∀ a b c,
      M.cast (M.add (M.cast (M.mul a b)) c) = M.cast (M.fma a b c)
  | "CANCEL" => ∀ a b, M.cast (M.add (M.cast (M.sub a b)) b) = M.cast a
  | "DIV-RCP" => ∀ a b, M.cast (M.divRN a b) = M.cast (M.mul a (M.cast (M.divRN 1 b)))
  | "CAST-MOVE" => ∀ a b,
      M.cast (M.add (M.cast a) (M.cast b)) = M.cast (M.add a b)
  | "CAST-REMOVE" => ∀ a b c,
      M.cast (M.mul (M.cast (M.add a b)) c) = M.cast (M.mul (M.add a b) c)
  | "ACC-WIDEN" => ∀ a b c,
      M.cast (M.add (M.cast (M.add a b)) c) = M.cast (M.add (M.add a b) c)
  | _ => False

def ModelsReport (M : ScalarModel) : Prop := ∀ name ∈ reportedIDs, LawFor M name

def shiftedAdd (a b : ℚ) := a + b + 1
def shiftedSub (a b : ℚ) := a - b - 1
def shiftedMul (a b : ℚ) := (a + 1) * (b + 1) - 1

/-- Addition/multiplication form a translated ring. The literal 0 is not its
additive identity. Both division symbols are equal here, so merely identifying
ordinary division with div_rn cannot repair the Welford initialization gap. -/
def shifted : ScalarModel where
  add := shiftedAdd
  sub := shiftedSub
  mul := shiftedMul
  divRN := fun _ _ => -1
  divPlain := fun _ _ => -1
  fma := fun a b c => shiftedAdd (shiftedMul a b) c
  cast := id
  exp := fun a => a + 2
  log := id
  sqrt := fun _ => 1

theorem shifted_models_report : ModelsReport shifted := by
  simp [ModelsReport, reported_ids, LawFor, shifted, shiftedAdd, shiftedSub, shiftedMul]
  repeat' apply And.intro
  all_goals (intros; ring)

theorem add_zero_is_not_forced : ¬ (∀ M, ModelsReport M → ∀ a, M.add a 0 = a) := by
  intro h
  have bad := h shifted shifted_models_report 0
  norm_num [shifted, shiftedAdd] at bad

/-- A valid singleton reduction has its input as its only leaf. -/
def singletonBatchMean (M : ScalarModel) (x : ℚ) := M.divPlain x 1

/-- The original recurrence at i=0, including its literal initialization and
the arithmetic in the denominator. Integer conversion is exact in this model. -/
def singletonOnlineMean (M : ScalarModel) (x : ℚ) :=
  M.add 0 (M.divPlain (M.sub x 0) (M.add 0 1))

theorem welford_initialization_is_not_forced :
    ¬ (∀ M, ModelsReport M → M.divPlain = M.divRN →
      ∀ x, singletonBatchMean M x = singletonOnlineMean M x) := by
  intro h
  have bad := h shifted shifted_models_report rfl 2
  norm_num [singletonBatchMean, singletonOnlineMean, shifted, shiftedAdd] at bad

/-- Exact arithmetic except for the untested ordinary division primitive. -/
def distinctDivision : ScalarModel where
  add := (· + ·)
  sub := (· - ·)
  mul := (· * ·)
  divRN := (· / ·)
  divPlain := fun a _ => a + 1
  fma := fun a b c => a * b + c
  cast := id
  exp := fun a => a + 2
  log := id
  sqrt := fun _ => 1

theorem distinct_division_models_report : ModelsReport distinctDivision := by
  simp [ModelsReport, reported_ids, LawFor, distinctDivision, div_eq_mul_inv]
  repeat' apply And.intro
  all_goals (intros; ring)

theorem div_rn_admission_does_not_cover_plain_division :
    ¬ (∀ M, ModelsReport M → ∀ a b,
      M.divPlain a b = M.mul a (M.divPlain 1 b)) := by
  intro h
  have bad := h distinctDivision distinct_division_models_report 2 3
  norm_num [distinctDivision] at bad

/-- No exp/log law was selected, even if both division symbols use exact
division and every cast is an identity. -/
def opaqueExpLog : ScalarModel := { distinctDivision with divPlain := (· / ·) }

theorem opaque_exp_log_models_report : ModelsReport opaqueExpLog :=
  distinct_division_models_report

def naiveSoftmaxFirst (M : ScalarModel) (x y : ℚ) :=
  M.divPlain (M.exp x) (M.add (M.exp x) (M.exp y))

def stableSoftmaxFirst (M : ScalarModel) (x y : ℚ) :=
  let m := max x y
  M.divPlain (M.exp (M.sub x m)) (M.add (M.exp (M.sub x m)) (M.exp (M.sub y m)))

theorem softmax_shift_is_not_forced :
    ¬ (∀ M, ModelsReport M → ∀ x y, naiveSoftmaxFirst M x y = stableSoftmaxFirst M x y) := by
  intro h
  have bad := h opaqueExpLog opaque_exp_log_models_report 0 1
  norm_num [naiveSoftmaxFirst, stableSoftmaxFirst, opaqueExpLog, distinctDivision] at bad

def directLogSumExp (M : ScalarModel) (x y : ℚ) := M.log (M.add (M.exp x) (M.exp y))

def shiftedLogSumExp (M : ScalarModel) (x y : ℚ) :=
  let m := max x y
  M.add m (M.log (M.add (M.exp (M.sub x m)) (M.exp (M.sub y m))))

theorem logsumexp_shift_is_not_forced :
    ¬ (∀ M, ModelsReport M → ∀ x y, directLogSumExp M x y = shiftedLogSumExp M x y) := by
  intro h
  have bad := h opaqueExpLog opaque_exp_log_models_report 0 1
  norm_num [directLogSumExp, shiftedLogSumExp, opaqueExpLog, distinctDivision] at bad

def shiftedPlain : ScalarModel := { shifted with divPlain := (· / ·) }

theorem shifted_plain_models_report : ModelsReport shiftedPlain := shifted_models_report

def singletonBatchVariance (M : ScalarModel) (x : ℚ) :=
  let d := M.sub x (singletonBatchMean M x)
  M.divPlain (M.mul d d) 1

def singletonOnlineVariance (M : ScalarModel) (x : ℚ) :=
  let delta := M.sub x 0
  let delta₂ := M.sub x (singletonOnlineMean M x)
  M.divPlain (M.add 0 (M.mul delta delta₂)) 1

def normalize (M : ScalarModel) (x mean variance gamma beta epsilon : ℚ) :=
  M.add (M.mul (M.mul (M.sub x mean) (M.divPlain 1 (M.sqrt (M.add variance epsilon)))) gamma) beta

theorem layernorm_outputs_are_not_forced :
    ¬ (∀ M, ModelsReport M → ∀ x g b e,
      normalize M x (singletonBatchMean M x) (singletonBatchVariance M x) g b e =
        normalize M x (singletonOnlineMean M x) (singletonOnlineVariance M x) g b e) := by
  intro h
  have bad := h shiftedPlain shifted_plain_models_report 2 1 0 1
  norm_num [normalize, singletonBatchMean, singletonOnlineMean,
    singletonBatchVariance, singletonOnlineVariance, shiftedPlain, shifted,
    shiftedAdd, shiftedSub, shiftedMul] at bad

theorem no_fp64_compute_admission :
    (FP.ReportedAdmission.all.map (·.compute)).eraseDups = ["bf16", "fp32"] := by decide

/-- A future inverse-cancellation atom needs its nonzero domain. Making it
unconditional alongside multiplication by zero would identify 0 and 1. This
is a constraint on future rule design, not a defect in the existing table. -/
theorem unconditional_inverse_cancellation_is_inconsistent (M : ScalarModel)
    (hzero : ∀ a, M.mul 0 a = 0)
    (hinverse : ∀ a, M.mul a (M.divPlain 1 a) = 1) : False := by
  have bad : (0 : ℚ) = 1 := (hzero (M.divPlain 1 0)).symm.trans (hinverse 0)
  norm_num at bad

#guard_msgs (drop info) in
run_cmd do
  for name in #[``reported_ids, ``shifted_models_report, ``add_zero_is_not_forced,
      ``welford_initialization_is_not_forced, ``distinct_division_models_report,
      ``div_rn_admission_does_not_cover_plain_division, ``opaque_exp_log_models_report,
      ``softmax_shift_is_not_forced, ``logsumexp_shift_is_not_forced,
      ``shifted_plain_models_report, ``layernorm_outputs_are_not_forced,
      ``no_fp64_compute_admission, ``unconditional_inverse_cancellation_is_inconsistent] do
    VeriTile.Meta.auditAxioms name
end FPAdmissionCoverageTests
