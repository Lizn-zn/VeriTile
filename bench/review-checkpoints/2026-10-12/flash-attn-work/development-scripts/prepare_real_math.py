from pathlib import Path
w=Path(__file__).resolve().parent
s=Path('bench/tritonbench_g/flash_attn/FlashAttn.lean').read_text()
def between(a,b): return s[s.index(a):s.index(b)]
parts=[between('noncomputable def osStepBot (','/-- The faithful'),] if '/-- The faithful' in s[s.index('noncomputable def osStepBot ('):s.index('noncomputable def flashStateBot')] else [s[s.index('noncomputable def osStepBot ('):s.index('noncomputable def flashStateBot')].rsplit('/--',1)[0]]
parts.append(between('theorem flashStateBot_fst\n','/-- The ⊥-seeded running `max` of'))
parts.append(between('theorem osStepBot_block_fst (','/-- **Reduce-max row bridge.'))
header='''import VeriTile.Triton.Math.Attention
import VeriTile.Triton.Semantics
import VeriTile.Meta.StatementAudit
/-! Pure real online-softmax lemmas, independent of kernel execution and layout.
Adapted from the existing flash_attn mathematical derivation. -/
namespace VeriTile.Bench.TritonBenchG.FlashAttn.RealMath
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false
'''
(w/'FlashAttnDraft/RealMath.lean').write_text(header+'\n'.join(parts)+'''
/-- Mathematical reference sums, with no recurrence or execution in their definition. -/
def denominator (xs : List (ℝ × ℝ)) : ℝ := (xs.map (fun p => pow2 p.1)).sum
def numerator (xs : List (ℝ × ℝ)) : ℝ := (xs.map (fun p => pow2 p.1 * p.2)).sum
def running (xs : List (ℝ × ℝ)) : WithBot ℝ × ℝ × ℝ := xs.foldl osStepBot (⊥,0,0)
def maximum (xs : List (ℝ × ℝ)) : WithBot ℝ :=
  (xs.map (fun p => (p.1 : WithBot ℝ))).foldr (· ⊔ ·) ⊥

theorem running_max (xs : List (ℝ × ℝ)) : (running xs).1=maximum xs := by
  exact (flashStateBot_fst xs ⊥ 0 0).trans (foldl_sup_bot_eq_foldr _)

theorem anchored (xs : List (ℝ × ℝ)) :
    (running xs).2.1=(maximum xs).elim 0 (fun r => pow2 (-r)) * denominator xs ∧
    (running xs).2.2=(maximum xs).elim 0 (fun r => pow2 (-r)) * numerator xs := by
  have h := osStepBot_foldl_consistent xs ⊥ 0 0 0 0 (by simp) (by simp) (fun _ => rfl) (fun _ => rfl)
  change (running xs).2.1 = (running xs).1.elim 0 (fun r => pow2 (-r)) * (0 + denominator xs) ∧
    (running xs).2.2 = (running xs).1.elim 0 (fun r => pow2 (-r)) * (0 + numerator xs) at h
  simpa only [running_max,zero_add] using h

theorem denominator_positive (xs : List (ℝ × ℝ)) (hne : xs≠[]) :
    0 < denominator xs := by
  obtain ⟨p,hp⟩ := List.exists_mem_of_ne_nil xs hne
  have hn : ∀ y ∈ xs.map (fun p => pow2 p.1), (0:ℝ)≤y := by
    intro y hy
    obtain ⟨q,_,rfl⟩ := List.mem_map.mp hy
    exact (pow2_pos _).le
  exact lt_of_lt_of_le (pow2_pos p.1) (List.single_le_sum hn _ (List.mem_map_of_mem hp))

theorem maximum_finite (xs : List (ℝ × ℝ)) (hne : xs≠[]) :
    ∃ r : ℝ, maximum xs=(r : WithBot ℝ) := by
  cases xs with
  | nil => exact False.elim (hne rfl)
  | cons p ps =>
    cases h : maximum ps with
    | bot => exact ⟨p.1,by simp [maximum] at h ⊢; rw [h]; simp⟩
    | coe r => exact ⟨max p.1 r,by simp only [maximum,List.map_cons,List.foldr_cons]; change (p.1 : WithBot ℝ) ⊔ maximum ps = _; rw [h,← WithBot.coe_max]⟩

theorem ratio_eq_reference (xs : List (ℝ × ℝ)) (hne : xs≠[]) :
    (running xs).2.2/(running xs).2.1=numerator xs/denominator xs := by
  obtain ⟨r,hr⟩ := maximum_finite xs hne
  rw [(anchored xs).1,(anchored xs).2,hr]
  exact mul_div_mul_left _ _ (ne_of_gt (pow2_pos (-r)))

theorem logsumexp_eq_reference (xs : List (ℝ × ℝ)) (hne : xs≠[]) :
    (maximum xs).unbotD 0 + Real.log ((running xs).2.1)/Real.log 2 =
      Real.log (denominator xs)/Real.log 2 := by
  obtain ⟨r,hr⟩ := maximum_finite xs hne
  rw [(anchored xs).1,hr,WithBot.unbotD_coe]
  change r + Real.log (pow2 (-r) * denominator xs)/Real.log 2 = _
  rw [Real.log_mul (ne_of_gt (pow2_pos _)) (ne_of_gt (denominator_positive xs hne))]
  rw [pow2,Real.log_exp]
  field_simp
  ring

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FlashAttn.RealMath
''')
