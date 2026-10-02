# Numerical rule results

56 instances; 40 replayed; 37 accepted.

z = max |z| over output buckets (diagnostic only). U uses the configured magnitude gate.
B = max(abs(mean) + se_multiplier * SE); bias PASS requires B <= tau, in local ULPs.
Bias FAIL means an interval lies outside tolerance; INCONCLUSIVE means a boundary is crossed.
The SE bands are engineering criteria, not calibrated simultaneous or optional-stopping confidence guarantees.
Errors are normalized per element by the output-format ULP at the rounded golden value before aggregation.
The magnitude gate uses peak normalized oracle errors and an additive allowance of 1 local ULP.
`empirical_max` means an observed maximum, not a fitted tail confidence bound.
Acceptance is statistical under the configured profile, not proof of strict floating-point equivalence.
Accept is pending until CPU replay. Missing statistics are shown as —, never zero.

| Rule | Format | R | z | B (ULP) | tau (ULP) | U | U type | Bias | Vars | Accept | State |
|---|---|---:|---:|---:|---:|---:|---|---|---|---|---|
| ADD-ZERO | bf16 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| MUL-ONE | bf16 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| DIV-ONE | bf16 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| DIV-MUL-RCP | bf16 | 4096 | 71.86937 | 0.009086923 | 0.05 | 0.6904547 | empirical_max | PASS | PASS | yes | COMPLETE |
| MUL-RCP-CANCEL | bf16 | 4096 | 1486.478 | 0.05393987 | 0.05 | 0 | empirical_max | FAIL | PASS | no | COMPLETE |
| EXP-SUB | bf16 | 4096 | 39.29068 | 0.00800256 | 0.05 | 0.3348518 | pot_pwm | PASS | PASS | yes | COMPLETE |
| EXP-ZERO | bf16 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| LOG-MUL | bf16 | — | — | — | — | — | — | — | — | no | NUMERIC_EVENT |
| LOG-EXP | bf16 | 4096 | 33.10437 | 0.07726643 | 0.05 | 0 | empirical_max | FAIL | PASS | no | COMPLETE |
| MAX-COMMUTE | bf16 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| MAX-ASSOC | bf16 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| MAX-IDEM | bf16 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| MAX-NEG-INF | bf16 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| EXP-NEG-INF-SUB | bf16 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| ADD-ZERO | bf16_fp32 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| MUL-ONE | bf16_fp32 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| DIV-ONE | bf16_fp32 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| DIV-MUL-RCP | bf16_fp32 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| MUL-RCP-CANCEL | bf16_fp32 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| EXP-SUB | bf16_fp32 | 4096 | 8.246203 | 1.652825e-05 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| EXP-ZERO | bf16_fp32 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| LOG-MUL | bf16_fp32 | — | — | — | — | — | — | — | — | no | NUMERIC_EVENT |
| LOG-EXP | bf16_fp32 | 4096 | 3.647023 | 0.0001622913 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| MAX-COMMUTE | bf16_fp32 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| MAX-ASSOC | bf16_fp32 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| MAX-IDEM | bf16_fp32 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| MAX-NEG-INF | bf16_fp32 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| EXP-NEG-INF-SUB | bf16_fp32 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| ADD-ZERO | fp32 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| MUL-ONE | fp32 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| DIV-ONE | fp32 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| DIV-MUL-RCP | fp32 | 4096 | 0 | 0 | 0.05 | 0.5495944 | pot_pwm | PASS | PASS | yes | COMPLETE |
| MUL-RCP-CANCEL | fp32 | 4096 | 788.7775 | 0.04319556 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| EXP-SUB | fp32 | 4096 | 103.6311 | 0.02767728 | 0.05 | 0.8998777 | pot_pwm | PASS | PASS | yes | COMPLETE |
| EXP-ZERO | fp32 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| LOG-MUL | fp32 | — | — | — | — | — | — | — | — | no | NUMERIC_EVENT |
| LOG-EXP | fp32 | 4096 | 3.565205 | 12.23416 | 0.05 | 0 | empirical_max | INCONCLUSIVE | PASS | no | COMPLETE |
| MAX-COMMUTE | fp32 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| MAX-ASSOC | fp32 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| MAX-IDEM | fp32 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| MAX-NEG-INF | fp32 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| EXP-NEG-INF-SUB | fp32 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| ADD-ZERO | fp64_fp64_fp32 | — | — | — | — | — | — | — | — | no | UNSUPPORTED |
| MUL-ONE | fp64_fp64_fp32 | — | — | — | — | — | — | — | — | no | UNSUPPORTED |
| DIV-ONE | fp64_fp64_fp32 | — | — | — | — | — | — | — | — | no | UNSUPPORTED |
| DIV-MUL-RCP | fp64_fp64_fp32 | 4096 | 1 | 3.576279e-07 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
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
