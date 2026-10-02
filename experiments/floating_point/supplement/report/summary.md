# Numerical rule results

56 instances; 40 replayed; 32 accepted.

z = max |z| over output buckets. U uses the configured magnitude gate.
`empirical_max` means an observed maximum, not a fitted tail confidence bound.
Acceptance is statistical under the configured profile, not proof of strict floating-point equivalence.
Accept is pending until CPU replay. Missing statistics are shown as —, never zero.

| Rule | Format | R | z | U | U type | Bias | Vars | Accept | State |
|---|---|---:|---:|---:|---|---|---|---|---|
| ADD-ZERO | bf16 | 4096 | 0 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| MUL-ONE | bf16 | 4096 | 0 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| DIV-ONE | bf16 | 4096 | 0 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| DIV-MUL-RCP | bf16 | 4096 | 2.887259 | 1.843361 | pot_pwm | PASS | PASS | yes | COMPLETE |
| MUL-RCP-CANCEL | bf16 | 4096 | 1481.767 | 0 | empirical_max | WARN | PASS | no | COMPLETE |
| EXP-SUB | bf16 | 4096 | 7.746623 | 0.6240578 | pot_pwm | WARN | PASS | no | COMPLETE |
| EXP-ZERO | bf16 | 4096 | 0 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| LOG-MUL | bf16 | — | — | — | — | — | — | no | NUMERIC_EVENT |
| LOG-EXP | bf16 | 4096 | 32.90097 | 0 | empirical_max | WARN | PASS | no | COMPLETE |
| MAX-COMMUTE | bf16 | 4096 | 0 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| MAX-ASSOC | bf16 | 4096 | 0 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| MAX-IDEM | bf16 | 4096 | 0 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| MAX-NEG-INF | bf16 | 4096 | 0 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| EXP-NEG-INF-SUB | bf16 | 4096 | 0 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| ADD-ZERO | bf16_fp32 | 4096 | 0 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| MUL-ONE | bf16_fp32 | 4096 | 0 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| DIV-ONE | bf16_fp32 | 4096 | 0 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| DIV-MUL-RCP | bf16_fp32 | 4096 | 0 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| MUL-RCP-CANCEL | bf16_fp32 | 4096 | 0 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| EXP-SUB | bf16_fp32 | 4096 | 13.34112 | 0 | empirical_max | WARN | PASS | no | COMPLETE |
| EXP-ZERO | bf16_fp32 | 4096 | 0 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| LOG-MUL | bf16_fp32 | — | — | — | — | — | — | no | NUMERIC_EVENT |
| LOG-EXP | bf16_fp32 | 4096 | 5.868474 | 0 | empirical_max | WARN | PASS | no | COMPLETE |
| MAX-COMMUTE | bf16_fp32 | 4096 | 0 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| MAX-ASSOC | bf16_fp32 | 4096 | 0 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| MAX-IDEM | bf16_fp32 | 4096 | 0 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| MAX-NEG-INF | bf16_fp32 | 4096 | 0 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| EXP-NEG-INF-SUB | bf16_fp32 | 4096 | 0 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| ADD-ZERO | fp32 | 4096 | 0 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| MUL-ONE | fp32 | 4096 | 0 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| DIV-ONE | fp32 | 4096 | 0 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| DIV-MUL-RCP | fp32 | 4096 | 0 | 0.9847333 | pot_pwm | PASS | PASS | yes | COMPLETE |
| MUL-RCP-CANCEL | fp32 | 4096 | 790.0049 | 0 | empirical_max | WARN | PASS | no | COMPLETE |
| EXP-SUB | fp32 | 4096 | 133.8035 | 2.998301 | pot_pwm | WARN | WARN | no | COMPLETE |
| EXP-ZERO | fp32 | 4096 | 0 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| LOG-MUL | fp32 | — | — | — | — | — | — | no | NUMERIC_EVENT |
| LOG-EXP | fp32 | 4096 | 467.881 | 0 | empirical_max | WARN | PASS | no | COMPLETE |
| MAX-COMMUTE | fp32 | 4096 | 0 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| MAX-ASSOC | fp32 | 4096 | 0 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| MAX-IDEM | fp32 | 4096 | 0 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| MAX-NEG-INF | fp32 | 4096 | 0 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| EXP-NEG-INF-SUB | fp32 | 4096 | 0 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| ADD-ZERO | fp64_fp64_fp32 | — | — | — | — | — | — | no | UNSUPPORTED |
| MUL-ONE | fp64_fp64_fp32 | — | — | — | — | — | — | no | UNSUPPORTED |
| DIV-ONE | fp64_fp64_fp32 | — | — | — | — | — | — | no | UNSUPPORTED |
| DIV-MUL-RCP | fp64_fp64_fp32 | 4096 | 1 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| MUL-RCP-CANCEL | fp64_fp64_fp32 | — | — | — | — | — | — | no | UNSUPPORTED |
| EXP-SUB | fp64_fp64_fp32 | — | — | — | — | — | — | no | UNSUPPORTED |
| EXP-ZERO | fp64_fp64_fp32 | — | — | — | — | — | — | no | UNSUPPORTED |
| LOG-MUL | fp64_fp64_fp32 | — | — | — | — | — | — | no | UNSUPPORTED |
| LOG-EXP | fp64_fp64_fp32 | — | — | — | — | — | — | no | UNSUPPORTED |
| MAX-COMMUTE | fp64_fp64_fp32 | — | — | — | — | — | — | no | UNSUPPORTED |
| MAX-ASSOC | fp64_fp64_fp32 | — | — | — | — | — | — | no | UNSUPPORTED |
| MAX-IDEM | fp64_fp64_fp32 | — | — | — | — | — | — | no | UNSUPPORTED |
| MAX-NEG-INF | fp64_fp64_fp32 | — | — | — | — | — | — | no | UNSUPPORTED |
| EXP-NEG-INF-SUB | fp64_fp64_fp32 | — | — | — | — | — | — | no | UNSUPPORTED |

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
