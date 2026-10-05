# Numerical rule results

7 instances; 7 replayed; 2 accepted.

Each nonempty replicate contributes one mean across its in-domain IID scalar instances; R counts these replicates.
Out-of-domain input tuples are skipped without resampling; valid/skipped counts describe scalar tuples, not R.
z = |mean| / SE across replicate means (diagnostic only). U uses the configured magnitude gate.
B = abs(mean) + se_multiplier * SE; bias PASS requires B <= tau, in local ULPs.
Bias FAIL means an interval lies outside tolerance; INCONCLUSIVE means a boundary is crossed.
The SE bands are engineering criteria, not calibrated simultaneous or optional-stopping confidence guarantees.
Bias differences are normalized by each rounded golden value's output-format ULP before averaging.
The magnitude gate uses K=Ec/Er, the ratio of peak absolute oracle errors, without ULP normalization or an additive allowance.
Both errors zero gives K=0; a zero reference error with positive candidate error gives infinity.
K follows the FlashAttention maximum-error metric; tail extrapolation and the bias gate are additional criteria.
`empirical_max` means an observed maximum, not a fitted tail confidence bound.
Acceptance is statistical under the configured profile, not proof of strict floating-point equivalence.
Accept is pending until CPU replay. Missing statistics are shown as —, never zero.

| Rule | Format | R | z | B (ULP) | tau (ULP) | U | U type | Bias | Vars | Accept | State | Valid | Skipped |
|---|---|---:|---:|---:|---:|---:|---|---|---|---|---|---:|---:|
| LOG-MUL | fp32 | 4096 | 2.924915 | 0.1679353 | 0.05 | 6.68556 | pot_pwm | INCONCLUSIVE | PASS | no | COMPLETE | 48643849868 | 20075626868 |
| LOG-MUL-LIBDEVICE | fp32 | 4096 | 2.924915 | 0.1679353 | 0.05 | 6.68556 | pot_pwm | INCONCLUSIVE | PASS | no | COMPLETE | 48643849868 | 20075626868 |
| LOG-EXP-LIBDEVICE | fp32 | 4096 | 68.5974 | 0.7300322 | 0.05 | 0 | empirical_max | FAIL | PASS | no | COMPLETE | 68719476736 | 0 |
| LOG-EXP-FULL-LIBDEVICE | fp32 | 4096 | 68.5974 | 0.7300322 | 0.05 | 0 | empirical_max | FAIL | PASS | no | COMPLETE | 68719476736 | 0 |
| LOG-EXP-GUARDED | fp32 | 4096 | 28014.77 | 0.04578995 | 0.05 | 0.625 | empirical_max | PASS | PASS | yes | COMPLETE | 68719476736 | 0 |
| LOG-MUL-LOG1P | fp32 | 4096 | 2.061441 | 0.1240316 | 0.05 | 6.68556 | pot_pwm | INCONCLUSIVE | PASS | no | COMPLETE | 48643849868 | 20075626868 |
| LOG-MUL-GUARDED | fp32 | 4096 | 296.6221 | 0.000624869 | 0.05 | 6.68556 | pot_pwm | PASS | PASS | yes | COMPLETE | 48643849868 | 20075626868 |
