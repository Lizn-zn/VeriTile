# Numerical rule results

56 instances; 40 replayed; 35 accepted.

Each replicate contributes one mean across its IID scalar instances; R counts replicates.
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

| Rule | Format | R | z | B (ULP) | tau (ULP) | U | U type | Bias | Vars | Accept | State |
|---|---|---:|---:|---:|---:|---:|---|---|---|---|---|
| ADD-ZERO | bf16 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| MUL-ONE | bf16 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| DIV-ONE | bf16 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| DIV-MUL-RCP | bf16 | 4096 | 4314.946 | 0.008102938 | 0.05 | 22.242 | pot_pwm | PASS | FAIL | no | COMPLETE |
| MUL-RCP-CANCEL | bf16 | 4096 | 90572.22 | 0.05362001 | 0.05 | 0 | empirical_max | FAIL | PASS | no | COMPLETE |
| EXP-SUB | bf16 | 4096 | 2207.84 | 0.006433089 | 0.05 | 1.35752 | pot_pwm | PASS | PASS | yes | COMPLETE |
| EXP-ZERO | bf16 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| LOG-MUL | bf16 | — | — | — | — | — | — | — | — | no | NUMERIC_EVENT |
| LOG-EXP | bf16 | 4096 | 1906.302 | 0.05977666 | 0.05 | 0 | empirical_max | FAIL | PASS | no | COMPLETE |
| MAX-COMMUTE | bf16 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| MAX-ASSOC | bf16 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| MAX-IDEM | bf16 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| MAX-NEG-INF | bf16 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| EXP-NEG-INF-SUB | bf16 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| ADD-ZERO | bf16_fp32 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| MUL-ONE | bf16_fp32 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| DIV-ONE | bf16_fp32 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| DIV-MUL-RCP | bf16_fp32 | 4096 | 0 | 0 | 0.05 | 1 | empirical_max | PASS | PASS | yes | COMPLETE |
| MUL-RCP-CANCEL | bf16_fp32 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| EXP-SUB | bf16_fp32 | 4096 | 314.8478 | 6.225613e-06 | 0.05 | 1.000388 | pot_pwm | PASS | PASS | yes | COMPLETE |
| EXP-ZERO | bf16_fp32 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| LOG-MUL | bf16_fp32 | — | — | — | — | — | — | — | — | no | NUMERIC_EVENT |
| LOG-EXP | bf16_fp32 | 4096 | 0.1212198 | 7.832305e-07 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| MAX-COMMUTE | bf16_fp32 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| MAX-ASSOC | bf16_fp32 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| MAX-IDEM | bf16_fp32 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| MAX-NEG-INF | bf16_fp32 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| EXP-NEG-INF-SUB | bf16_fp32 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| ADD-ZERO | fp32 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| MUL-ONE | fp32 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| DIV-ONE | fp32 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| DIV-MUL-RCP | fp32 | 4096 | 0 | 0 | 0.05 | 1 | empirical_max | PASS | PASS | yes | COMPLETE |
| MUL-RCP-CANCEL | fp32 | 4096 | 49081.95 | 0.04271403 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| EXP-SUB | fp32 | 8192 | 8838.004 | 0.02519245 | 0.05 | 2.54112 | pot_pwm | PASS | WARN | no | COMPLETE |
| EXP-ZERO | fp32 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| LOG-MUL | fp32 | — | — | — | — | — | — | — | — | no | NUMERIC_EVENT |
| LOG-EXP | fp32 | 4096 | 1.916659 | 0.06895381 | 0.05 | 0 | empirical_max | INCONCLUSIVE | PASS | no | COMPLETE |
| MAX-COMMUTE | fp32 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| MAX-ASSOC | fp32 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| MAX-IDEM | fp32 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| MAX-NEG-INF | fp32 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| EXP-NEG-INF-SUB | fp32 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| ADD-ZERO | fp64_fp64_fp32 | — | — | — | — | — | — | — | — | no | UNSUPPORTED |
| MUL-ONE | fp64_fp64_fp32 | — | — | — | — | — | — | — | — | no | UNSUPPORTED |
| DIV-ONE | fp64_fp64_fp32 | — | — | — | — | — | — | — | — | no | UNSUPPORTED |
| DIV-MUL-RCP | fp64_fp64_fp32 | 4096 | 0.7558894 | 4.432353e-10 | 0.05 | 1 | empirical_max | PASS | PASS | yes | COMPLETE |
| MUL-RCP-CANCEL | fp64_fp64_fp32 | — | — | — | — | — | — | — | — | no | UNSUPPORTED |
| EXP-SUB | fp64_fp64_fp32 | — | — | — | — | — | — | — | — | no | UNSUPPORTED |
| EXP-ZERO | fp64_fp64_fp32 | — | — | — | — | — | — | — | — | no | UNSUPPORTED |
| LOG-MUL | fp64_fp64_fp32 | — | — | — | — | — | — | — | — | no | UNSUPPORTED |
| LOG-EXP | fp64_fp64_fp32 | — | — | — | — | — | — | — | — | no | UNSUPPORTED |
| MAX-COMMUTE | fp64_fp64_fp32 | — | — | — | — | — | — | — | — | no | UNSUPPORTED |
| MAX-ASSOC | fp64_fp64_fp32 | — | — | — | — | — | — | — | — | no | UNSUPPORTED |
| MAX-IDEM | fp64_fp64_fp32 | — | — | — | — | — | — | — | — | no | UNSUPPORTED |
| MAX-NEG-INF | fp64_fp64_fp32 | — | — | — | — | — | — | — | — | no | UNSUPPORTED |
| EXP-NEG-INF-SUB | fp64_fp64_fp32 | — | — | — | — | — | — | — | — | no | UNSUPPORTED |

## Notes

- LOG-MUL / bf16: log requires a > 0 and b > 0; no abs, truncation or resampling
- LOG-MUL / bf16_fp32: log requires a > 0 and b > 0; no abs, truncation or resampling
- LOG-MUL / fp32: log requires a > 0 and b > 0; no abs, truncation or resampling
- ADD-ZERO / fp64_fp64_fp32: fp64-work supplement covers only ordinary division with fp32 output
- MUL-ONE / fp64_fp64_fp32: fp64-work supplement covers only ordinary division with fp32 output
- DIV-ONE / fp64_fp64_fp32: fp64-work supplement covers only ordinary division with fp32 output
- MUL-RCP-CANCEL / fp64_fp64_fp32: fp64-work supplement covers only ordinary division with fp32 output
- EXP-SUB / fp64_fp64_fp32: fp64-work supplement covers only ordinary division with fp32 output
- EXP-ZERO / fp64_fp64_fp32: fp64-work supplement covers only ordinary division with fp32 output
- LOG-MUL / fp64_fp64_fp32: fp64-work supplement covers only ordinary division with fp32 output
- LOG-EXP / fp64_fp64_fp32: fp64-work supplement covers only ordinary division with fp32 output
- MAX-COMMUTE / fp64_fp64_fp32: fp64-work supplement covers only ordinary division with fp32 output
- MAX-ASSOC / fp64_fp64_fp32: fp64-work supplement covers only ordinary division with fp32 output
- MAX-IDEM / fp64_fp64_fp32: fp64-work supplement covers only ordinary division with fp32 output
- MAX-NEG-INF / fp64_fp64_fp32: fp64-work supplement covers only ordinary division with fp32 output
- EXP-NEG-INF-SUB / fp64_fp64_fp32: fp64-work supplement covers only ordinary division with fp32 output
