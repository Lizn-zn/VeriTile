# 补充基础原子实验

本目录验证补充的基础标量关系，与主实验共用局部 ULP 偏差和最大绝对误差比两个 gate。
这里没有 softmax、Welford、LayerNorm 或整个 reduction 的准入原子。
主目录与本目录使用相同的平均偏差预算和 U gate。通过项分别导出到
`ReportedAdmission` 和 `SupplementalAdmission`；总数从当前报告计算。

## 当前结果

当前 bias 协议将同分布的标量实例汇总成每个 replicate 的一个均值。当前结果
使用固定种子 20261003 在 H200（运行时显示 NVIDIA L20X）重新采样，并完成独立
CPU 回放。已发布表绑定其记录的源代码哈希和聚合协议。

配置固定 `tau=0.05 local ULP`、`se_multiplier=5`。EXP-SUB 的 exp 使用 FP32
`libdevice.exp`，保留原有输入、减法、除法、中间 cast 和输出 cast。
DLC 名称为 `traces_kernel_equivalence_testing`，任务 ID 为 `dlchk3ifersdpxho`。

[完整 z / B / tau / U / accept 表](./report/summary.md) 同时提供
[全精度 CSV](./report/summary.csv) 和 [JSON](./report/summary.json)。
[运行配置](./report/experiment.json) 记录源代码、设备、任务和独立 CPU 回放信息，
[未接受项明细](./report/warning_audit.json) 区分偏差超预算、区间尚不足以确认和 U gate 状态。
表中每个配置都单独报告；未测或不适用的统计不填写为零。

跨 replicate 均值的 `abs(mean) + 5*SE <= 0.05` 才满足 bias 预算；z 仅作诊断。
偏差区间完全位于容差外就是 FAIL；没有 FAIL、但区间跨过边界时是
INCONCLUSIVE。两者都不能准入。
U gate 的幅度阈值为 10/100：`U <= 10` 为 PASS，`10 < U <= 100` 为 WARN，
`U > 100` 为 FAIL。五个标准误是工程判据，不宣称已经校准多桶或
自适应停止覆盖率。统计接受不等于无条件 IEEE 等式证明。

原始观测、PTX 和完整统计不放入 Git；当前表、配置和审核摘要随代码维护。

## 直接运行

### Paired log and exp implementations

Every ordinary log and exp relation has separate intrinsic and libdevice versions.
Use `from triton.language.extra.cuda import libdevice`; the intrinsic calls are
`tl.log` and `tl.exp`. Each comparison holds all other intrinsics, intermediate
rounding, branch bounds, domains, draws and gates fixed. Both unconditional and
guarded log-exp cover the complete two-by-two log/exp matrix.
The compiler uses `enable_fp_fusion=False` and its default math settings.

| Relation / fixed log | tl.exp version | libdevice.exp version |
|---|---|---|
| EXP-SUB-INTRINSIC / — | EXP-SUB-INTRINSIC | EXP-SUB |
| EXP-ZERO / — | EXP-ZERO | EXP-ZERO-LIBDEVICE |
| EXP-NEG-INF-SUB / — | EXP-NEG-INF-SUB | EXP-NEG-INF-SUB-LIBDEVICE |
| LOG-EXP / tl.log | LOG-EXP | LOG-EXP-LIBDEVICE |
| LOG-EXP-LOG-LIBDEVICE / libdevice.log | LOG-EXP-LOG-LIBDEVICE | LOG-EXP-FULL-LIBDEVICE |
| LOG-EXP-GUARDED-FULL-INTRINSIC / tl.log | LOG-EXP-GUARDED-FULL-INTRINSIC | LOG-EXP-GUARDED-INTRINSIC |
| LOG-EXP-GUARDED-EXP-INTRINSIC / libdevice.log | LOG-EXP-GUARDED-EXP-INTRINSIC | LOG-EXP-GUARDED |

The log-only pair definitions are in `scripts/supplement_numerics.py::LOG_PAIRS`;
`EXP_PAIRS` records the exp-only comparisons above. All eight log-exp variants
share draws, as do both EXP-SUB variants. The `LOG-EXP-LIBDEVICE` identifier means
**tl.log with libdevice.exp**. `LOG-EXP-GUARDED-INTRINSIC` changes only log;
`-EXP-INTRINSIC` changes only exp; `-FULL-INTRINSIC` uses both tl calls.
Contracts and Lean fragments select exact implementations independently.

Triton has no `tl.log1p` in this environment. Both LOG1P diagnostics retain
`libdevice.log1p(tl.fma(a,b,-1))` near one and switch only ordinary logs.
They change the reference and cannot establish equivalence to its unchanged
implementation; these two probes remain experiment-only.

All four guarded log-exp variants use the same FP32 formula:

```python
reference = log_impl(exp_impl(a))
candidate = a if 0.5 < abs(a) <= 80 else log_impl(exp_impl(a))
```

Each candidate preserves its own reference in the fallback. Branches select
implementations without filtering input samples. Whole identity blocks skip both
calls; mixed blocks mask inactive fallback arguments. The Lean expressions
model each lane, without proving GPU compilation or block scheduling.
Guarded log-product still retains `log_impl(fp32(a*b))` for
`0.5 <= fp32(a*b) <= 2` and splits the logs elsewhere.

H200 job `dlczuyolms2zwn68`, named `traces_kernel_equivalence_testing`, runs 14 log
variants and 6 pure-exp variants under each of seeds 20261003 and 20261005.
Each case uses 4096 replicates of shape `[4096,4096]`, Normal(1,1), a 0.05
local-ULP bias budget and U thresholds 10/100. Independent CPU replay reproduces
all four complete tables. Constant EXP-ZERO and EXP-NEG-INF-SUB probes may compile
to constants; repeated identical outputs do not expand the tested input domain.

Primary results:

| Rule | log | exp | z | B (local ULP) | U | Accept |
|---|---|---|---:|---:|---:|---|
| LOG-MUL | tl.log | — | 2.924915168 | 0.1679352504 | 6.685560237 | No: INCONCLUSIVE |
| LOG-MUL-LIBDEVICE | libdevice.log | — | 2.924915168 | 0.1679352504 | 6.685560237 | No: INCONCLUSIVE |
| LOG-EXP | tl.log | tl.exp | 3.905770501 | 0.08849077969 | 0 | No: INCONCLUSIVE |
| LOG-EXP-LOG-LIBDEVICE | libdevice.log | tl.exp | 3.905770501 | 0.08849077969 | 0 | No: INCONCLUSIVE |
| LOG-EXP-LIBDEVICE | tl.log | libdevice.exp | 68.59740095 | 0.7300322166 | 0 | No: FAIL |
| LOG-EXP-FULL-LIBDEVICE | libdevice.log | libdevice.exp | 68.59740095 | 0.7300322166 | 0 | No: FAIL |
| LOG-MUL-GUARDED-INTRINSIC | tl.log | — | 296.6221498 | 0.0006248690033 | 6.685560237 | Yes |
| LOG-MUL-GUARDED | libdevice.log | — | 296.6221498 | 0.0006248690033 | 6.685560237 | Yes |
| LOG-EXP-GUARDED-INTRINSIC | tl.log | libdevice.exp | 28014.76841 | 0.04578995059 | 0.625 | Yes |
| LOG-EXP-GUARDED | libdevice.log | libdevice.exp | 28014.76841 | 0.04578995059 | 0.625 | Yes |
| LOG-MUL-LOG1P-INTRINSIC | tl.log | — | 2.061440595 | 0.1240315873 | 6.685560237 | No: INCONCLUSIVE |
| LOG-MUL-LOG1P | libdevice.log | — | 2.061440595 | 0.1240315873 | 6.685560237 | No: INCONCLUSIVE |
| LOG-EXP-GUARDED-FULL-INTRINSIC | tl.log | tl.exp | 39092.26596 | 0.07841065488 | 0.3125 | No: FAIL |
| LOG-EXP-GUARDED-EXP-INTRINSIC | libdevice.log | tl.exp | 39092.26596 | 0.07841065488 | 0.3125 | No: FAIL |
| EXP-SUB-INTRINSIC | — | tl.exp | 32906.95855 | 0.1609050508 | 3.688724142 | No: FAIL |
| EXP-SUB | — | libdevice.exp | 6284.440943 | 0.0251993381 | 2.644559637 | Yes |
| EXP-ZERO | — | tl.exp | 0 | 0 | 0 | Yes |
| EXP-ZERO-LIBDEVICE | — | libdevice.exp | 0 | 0 | 0 | Yes |
| EXP-NEG-INF-SUB | — | tl.exp | 0 | 0 | 0 | Yes |
| EXP-NEG-INF-SUB-LIBDEVICE | — | libdevice.exp | 0 | 0 | 0 | Yes |

Independent-seed confirmation:

| Rule | log | exp | z | B (local ULP) | U | Accept |
|---|---|---|---:|---:|---:|---|
| LOG-MUL | tl.log | — | 1.016401743 | 0.1629233926 | 7.418400148 | No: INCONCLUSIVE |
| LOG-MUL-LIBDEVICE | libdevice.log | — | 1.016401743 | 0.1629233926 | 7.418400148 | No: INCONCLUSIVE |
| LOG-EXP | tl.log | tl.exp | 4.597276651 | 0.09486808173 | 0 | No: INCONCLUSIVE |
| LOG-EXP-LOG-LIBDEVICE | libdevice.log | tl.exp | 4.597276651 | 0.09486808173 | 0 | No: INCONCLUSIVE |
| LOG-EXP-LIBDEVICE | tl.log | libdevice.exp | 68.23167626 | 0.7236907333 | 0 | No: FAIL |
| LOG-EXP-FULL-LIBDEVICE | libdevice.log | libdevice.exp | 68.23167626 | 0.7236907333 | 0 | No: FAIL |
| LOG-MUL-GUARDED-INTRINSIC | tl.log | — | 296.0360072 | 0.0006251552164 | 7.418400148 | Yes |
| LOG-MUL-GUARDED | libdevice.log | — | 296.0360072 | 0.0006251552164 | 7.418400148 | Yes |
| LOG-EXP-GUARDED-INTRINSIC | tl.log | libdevice.exp | 28421.27843 | 0.04579159812 | 0.625 | Yes |
| LOG-EXP-GUARDED | libdevice.log | libdevice.exp | 28421.27843 | 0.04579159812 | 0.625 | Yes |
| LOG-MUL-LOG1P-INTRINSIC | tl.log | — | 1.79346894 | 0.1644764832 | 7.418400148 | No: INCONCLUSIVE |
| LOG-MUL-LOG1P | libdevice.log | — | 1.79346894 | 0.1644764832 | 7.418400148 | No: INCONCLUSIVE |
| LOG-EXP-GUARDED-FULL-INTRINSIC | tl.log | tl.exp | 39224.66494 | 0.0784103817 | 0.3125 | No: FAIL |
| LOG-EXP-GUARDED-EXP-INTRINSIC | libdevice.log | tl.exp | 39224.66494 | 0.0784103817 | 0.3125 | No: FAIL |
| EXP-SUB-INTRINSIC | — | tl.exp | 33437.05898 | 0.1609043644 | 4.049359137 | No: FAIL |
| EXP-SUB | — | libdevice.exp | 6204.438842 | 0.02520213491 | 2.638695181 | Yes |
| EXP-ZERO | — | tl.exp | 0 | 0 | 0 | Yes |
| EXP-ZERO-LIBDEVICE | — | libdevice.exp | 0 | 0 | 0 | Yes |
| EXP-NEG-INF-SUB | — | tl.exp | 0 | 0 | 0 | Yes |
| EXP-NEG-INF-SUB-LIBDEVICE | — | libdevice.exp | 0 | 0 | 0 | Yes |

[Log comparison data](./log_report/comparison.json) and
[exp comparison data](./exp_report/comparison.json) record paired observations
and PTX comparisons under Triton 3.7.1 / CUDA 13.0 on sm_90. Compiler equality
is measured per pair and configuration, never assumed across APIs.
Every row's complete z/B/U/accept data is also available in each report's CSV/JSON.

The generated admission contains four guarded variants: `log_mul_split .tl`,
`log_mul_split .libdevice`, `log_exp_cancel .tl .libdevice`, and
`log_exp_cancel .libdevice .libdevice`. In `log_exp_cancel log exp`, the second
backend defaults to `.libdevice`, preserving the existing one-argument calls.
The two additional tl.exp candidates retain independent fragments and report IDs;
they fail the bias budget under both seeds while their libdevice.exp counterparts
pass. The common guard does not grant admission across exp APIs.
Unconditional LOG-MUL/LOG-EXP and both LOG1P probes remain unadmitted. The LOG1P
pair is experiment-only because its fused operation still needs a corresponding
Lean execution model.

The generated `LogAdmission` includes only accepted log relations. `ExpAdmission`
selects five of the six separately defined fp32 candidates in `Float/Exponential`
from `exp_report`, with decisions confirmed by `exp_validation_report`.
`exp_sub exp`, `exp_zero exp` and `exp_neg_inf_sub exp` default to `.tl`;
exp-sub requires an explicit `.libdevice` selection under these results.
The two constant relations admit either backend. Assumption output retains
the API, for example `exp_zero(tl.exp)` or `exp_sub(libdevice.exp)`.
An unchanged exp prefix needs no exp assumption: the reciprocal-softmax examples
use tl.exp. Shift/cancellation examples retain libdevice.exp where their required
relation fails with tl.exp. There is no equality assumed between the two APIs.
Unaccepted variants cannot be selected through the typed Lean catalog. Numerical
acceptance remains conditional on the recorded experiment and trust contract.

Twenty-five boundary inputs exercise all eight log-exp implementations; sixteen
product boundary inputs cover both log implementations. Both sides preserve
FP32 computation and contain no FP64 instructions. Nonfinite outputs on valid
inputs remain failures of the statistical experiment. Boundary checks explicitly
cover matching underflow/overflow behavior rather than admitting those outputs.
[Boundary data](./log_report/boundaries.json) contains separate descriptive timings
for all four guarded variants on all-fast, all-fallback and mixed inputs.
Acceptance is not a speedup claim. U's `empirical_max` fallback, where reported,
is an observed maximum ratio rather than a fitted tail confidence bound.

Reproduce the current reports:

```bash
python3 scripts/check_log_accuracy.py --output Logs/fp-log-boundaries
python3 scripts/check_log_product.py --output Logs/fp-log-product-diagnostics
python3 scripts/check_numerics_supplement.py run --profile experiments/floating_point/supplement/log_product_config.py --output Logs/fp-log
python3 scripts/check_numerics_supplement.py report Logs/fp-log --output-dir Logs/fp-log-report
python3 scripts/check_numerics_supplement.py run --profile experiments/floating_point/supplement/log_product_validation_config.py --output Logs/fp-log-validation
python3 scripts/check_numerics_supplement.py report Logs/fp-log-validation --output-dir Logs/fp-log-validation-report
python3 scripts/check_numerics_supplement.py run --profile experiments/floating_point/supplement/exp_config.py --output Logs/fp-exp
python3 scripts/check_numerics_supplement.py report Logs/fp-exp --output-dir Logs/fp-exp-report
python3 scripts/check_numerics_supplement.py run --profile experiments/floating_point/supplement/exp_validation_config.py --output Logs/fp-exp-validation
python3 scripts/check_numerics_supplement.py report Logs/fp-exp-validation --output-dir Logs/fp-exp-validation-report
python3 scripts/export_supplemental_rules.py --trust-report --report experiments/floating_point/supplement/log_report --namespace LogAdmission --output VeriTile/Triton/Float/LogAdmission.lean
python3 scripts/export_supplemental_rules.py --trust-report --report experiments/floating_point/supplement/exp_report --namespace ExpAdmission --output VeriTile/Triton/Float/ExpAdmission.lean
```

### 全部补充实验

使用 NVIDIA CUDA 环境。新环境安装：

```bash
python3 -m pip install -r experiments/floating_point/requirements.txt
```

在仓库根目录执行：

```bash
python3 scripts/check_numerics_supplement.py check
python3 scripts/check_numerics_supplement.py run --smoke --output Logs/fp-supplement-smoke
python3 scripts/check_numerics_supplement.py run --output Logs/fp-supplement
python3 scripts/check_numerics_supplement.py report Logs/fp-supplement --output-dir Logs/fp-supplement-report
```

先确认 smoke 没有 `ERROR` 再运行正式实验。Smoke 用 `32×33`、4 次整张量采样，
即使 PASS 也只记为 `SMOKE_ONLY`，不准入。`LOG-MUL` 跳过非正输入；fp64 profile
中不适用的关系仍记为 `UNSUPPORTED`。正式实验不会自动缩小 shape。

中断后，在相同代码、配置和 GPU/软件环境下继续：

```bash
python3 scripts/check_numerics_supplement.py run --output Logs/fp-supplement --resume
```

完成的记录先检查后保留；中断或 `ERROR` 实例从相同 seed 重新开始。
编译/CUDA 错误会记录原因并使进程返回非零；数值 REJECT、WARN、INCONCLUSIVE
是实验结果。输出目录不得复用为另一轮实验，报告也写入新目录。

跑完把 `Logs/fp-supplement-report/` 的五个文件带回来即可查看完整汇总和准入候选：
`summary.md`、`summary.csv`、`summary.json`、`admission.json`、`experiment.json`。
保留原始 bundle（NPZ、record.json、PTX 和源码快照），需要归档时：

```bash
tar -czf fp-supplement-results.tar.gz -C Logs fp-supplement fp-supplement-report
```

`report` 在运行机器上用 NumPy 校验并重算统计量，不会重新执行 GPU kernel。
也可以在同一实验代码版本的 CPU 机器运行该命令。它不会执行 bundle 内的 Python。
补充报告使用独立 schema，不能直接覆盖主目录 `report/` 或送入主目录的 Lean 导出器。
结果回来后，再把通过的**准确表达式、精度和必要定义域**接入 Lean。

## 测什么

[rules.json](./rules.json) 列出每对实际表达式和定义域；[kernels.py](./kernels.py)
实现它们。下表省略每个节点的舍入 `Q` 和末尾的输出转换，源码及报告均保留。

| 原子 | 参考 → 候选 | 用途 |
|---|---|---|
| add-zero | `a + 0 → a` | 初值、零项 |
| mul-one | `a * 1 → a` | 单位元 |
| div-one | `a / 1 → a` | 除法初值 |
| div-mul-rcp | `a / b → a * (1 / b)` | 普通 `/` 的倒数改写，`b ≠ 0` |
| mul-rcp-cancel | `a * (1 / a) → 1` | 消去非零因子，`a ≠ 0` |
| exp-sub | `libdevice.exp(a - b) → libdevice.exp(a) / libdevice.exp(b)` | Shift and scale rewrite |
| exp-sub-intrinsic | `tl.exp(a - b) → tl.exp(a) / tl.exp(b)` | Paired exp implementation; identical intermediate rounding |
| exp-zero | `exp(0) → 1` | 指数初值 |
| exp-zero-libdevice | `libdevice.exp(0) → 1` | Constant initialization |
| log-mul | `log(a * b) → log(a) + log(b)` | 对数因子分解，`a,b > 0` |
| log-mul-libdevice | `libdevice.log(a * b) → libdevice.log(a) + libdevice.log(b)` | 配对比较 log 实现，`a,b > 0` |
| log-mul-log1p | 乘积近 1 时参考用 `log1p(fma(a,b,-1))`，候选两个 log 相加 | 仅 FP32；两种子均未准入 |
| log-mul-guarded | 候选在 `0.5 <= fp32(a*b) <= 2` 时保留参考，范围外拆分 log | 仅 FP32；只准入条件式 |
| log-exp | `tl.log(tl.exp(a)) → a` | 原 intrinsic 组合 |
| log-exp-log-libdevice | `libdevice.log(tl.exp(a)) → a` | Log-only libdevice variant |
| log-exp-libdevice | `tl.log(libdevice.exp(a)) → a` | 当前 fp32 实验因 bias 拒绝 |
| log-exp-full-libdevice | `libdevice.log(libdevice.exp(a)) → a` | 两个函数都使用 libdevice，当前 fp32 实验因 bias 拒绝 |
| log-mul-guarded-intrinsic | Same product guard, with tl.log on every path | FP32 conditional split |
| log-mul-log1p-intrinsic | Same log1p/FMA reference near one; tl.log for ordinary logs | FP32 diagnostic, not admitted |
| log-exp-guarded-intrinsic | Same log-exp guard, with tl.log and libdevice.exp fallback | FP32 conditional identity |
| log-exp-guarded | Fixed `log(exp(a))` reference; candidate returns `a` for `0.5 < abs(a) <= 80`, otherwise the reference | FP32 libdevice fallback |
| log-exp-guarded-full-intrinsic | Same guard, with `tl.log(tl.exp(a))` reference and fallback | FP32; rejected by the bias gate |
| log-exp-guarded-exp-intrinsic | Same guard, with `libdevice.log(tl.exp(a))` reference and fallback | FP32; rejected by the bias gate |
| max-commute | `max(a,b) → max(b,a)` | max 标量换序 |
| max-assoc | `max(max(a,b),c) → max(a,max(b,c))` | max 标量重组 |
| max-idem | `max(a,a) → a` | 消去重复 max 项 |
| max-neg-inf | `max(-inf,a) → a` | online max 的初值 |
| exp-neg-inf-sub | `exp(-inf - a) → 0` | online 指数权重的初值，`a` 有限 |
| exp-neg-inf-sub-libdevice | `libdevice.exp(-inf - a) → 0` | Finite `a`; explicit negative-infinity literal |

每条关系只有固定数量的标量操作。最后一条保留 `-inf` 的局部表达式，
避免把无穷大当成普通有限数套进 exp-sub；它不是 online-softmax 整体关系。
常数和恒等式可能被编译器折叠，保存的 PTX 反映实际执行图。

The catalog contains 31 relations. The default floating-point profile has
**72 executable cases and 144 reference/candidate kernel specializations**,
plus one FP64 residual oracle. The complete Cartesian table has 124 rows;
unsupported combinations are marked `UNSUPPORTED`.
The guarded and LOG1P relations support only FP32 inputs, computation and
outputs. Both EXP-SUB versions support the same BF16 and FP32 precision profiles. COUNT-ZERO and
COUNT-SUCCESSOR use a separate int32 input profile; see the
[primitive configuration and results](../primitives/README.md).

## 配置与精度

编辑 [config.py](./config.py)：默认 shape `[4096,4096]`，独立 `Normal(1,1²)`，
`std` 是 σ。每次 replicate 生成全新的 a/b/c 整张量；常数原子不使用这些操作数。
shape 表示局部表达式的采样批次，不是未来 Lean kernel 的固定尺寸。

| 名称 | 输入 | 每节点计算 | 输出 |
|---|---|---|---|
| bf16 | bf16 | fp32 执行后逐节点舍入到 bf16 | bf16 |
| bf16_fp32 | bf16 | fp32 | bf16 |
| fp32 | fp32 | fp32 | fp32 |
| fp64_fp64_fp32 | fp64 | fp64 | fp32 |

这些原子不做 reduction，所以没有实际累加操作；profile 保留 accumulator 字段，
具体 contract 的 accumulator map 为空。不能据此宣称验证了任何求和累加精度。

fp64 实例仅检查 `fp32(a64 / b64)` 与 `fp32(a64 * (1 / b64))`。
这是 FloatDTypeSoftmax 所需局部表达式的一种实验实例，**不能删除最终 fp32 cast，
也不能作为任意 fp64 exp/log 或裸 fp64 等式的证据**。这里 a、b 直接采样为 fp64，
对应改写位置的 fp64 中间值；整个 softmax 的原始输入仍可为 fp32。
采样分布不声称等于完整 softmax 中间值的实际分布。

一般实例的 oracle 沿用同量化输入上的 torch fp64 数学求值。fp64-work 除法
用 `|fma(-output,b,a)/b|` 计算误差，避免把同精度舍入后的商误认为精确真值：
显式 fp64 FMA 将乘法与减法合并为一次舍入，保留接近正确商时的小残差；
最后的误差除法在 fp64 舍入。oracle 的 PTX 也保存并绑定哈希。
oracle 仍是受信数值计算，不是精确实数证明。

误差尺度固定为每个元素 golden 值在输出 dtype 下的 ULP。先对每个元素计算
`(candidate-reference)/ULP(golden)`，再对本次 replicate 的全部同分布标量实例
中的有效样本取一个均值。保存的 delta 形状为 `[R, 1]`，跨 R 个非空 replicate 均值计算标准误
和诊断 z，偏差区间须落在 ±tau 内。不将元素数计入 R；具有不同分布或语义的
channel/head 不能直接沿用这种合并方式。
幅度 gate 分别取两侧最大绝对 oracle 误差 Er、Ec，计算 `K = Ec/Er` 并拟合 U。
这里不除以 ULP，也没有加性容差；两侧误差都为零时 K=0，Er=0 且 Ec>0 时
K 为无穷大。fp64-work 的残差误差同样保留绝对单位。
K 对齐 FlashAttention 的最大误差比较；尾部外推和 bias gate 是额外的要求，
不能把整个 two-gates 判定说成与 FA 的样本测试完全相同。
bias 的零值尺度使用最小 subnormal 间距；输出 dtype 无法表示的 golden 尺度触发失败。
不使用输出峰值或跨 replicate 的最大 ULP 作为 bias 容差。

`/` 是普通 Triton division，**不是**原 `DIV-RCP` 的 `tl.div_rn`。
EXP-SUB、LOG-EXP-LIBDEVICE、LOG-EXP-FULL-LIBDEVICE 和 LOG-EXP-GUARDED 使用 `libdevice.exp`；
其他 exp 原子使用 `tl.exp`。LOG-MUL-LIBDEVICE 和 LOG-EXP-FULL-LIBDEVICE 使用
`libdevice.log`，其他 log 原子使用 `tl.log`，max 使用 `tl.maximum`。
LOG-EXP-GUARDED uses libdevice.log/exp in the fixed reference and fallback; its fast path returns a.
LOG-MUL-LOG1P 使用显式 `tl.fma` 和 `libdevice.log1p`；LOG-MUL-GUARDED 使用 `libdevice.log`。
intrinsic 身份保存在每条规则的契约中；禁止隐式 FMA fusion。

只跑某组关系可以显式选择：

```bash
python3 scripts/check_numerics_supplement.py run --rules ADD-ZERO,MUL-ONE,DIV-ONE,DIV-MUL-RCP,MUL-RCP-CANCEL --formats fp32 --output Logs/fp-supplement-arithmetic
```

## 定义域与后续证明

只按指定分布采样，在输入量化后跳过定义域外的标量元组：log-mul 要求
`a > 0 && b > 0`，除法要求分母非零，参与运算的输入还须有限。
两侧表达式与 oracle 使用同一份输入掩码统计。原始输入和 kernel 的完整 shape
保持不变，不取绝对值，不补样或重采样。统计对象就是指定分布限制在该关系
定义域内的样本；有效输入上产生的非有限输出/误差仍由 gates 拒绝。

每个非空 replicate 只在有效样本上求均值和误差最大值，空 replicate 跳过，
不能当作零误差。R 只计非空 replicate；向上取整到 batch 的 `replicates_max`
同时限制抽样次数，预算内有效 replicate 不足则为 `INCONCLUSIVE`。
NPZ 的 `valid_samples` 保存每次抽样的有效元组数，record 保存尝试次数、空批次、
有效及跳过总数，报告的 Valid / Skipped 两列显示元组总数。

当前定义域过滤及配对 log 精度实验使用 `scalar-supplement-11`，
须使用新的输出目录。已提交的 PR #11
报告仍保留旧策略及其 `NUMERIC_EVENT` 结果，不能当作新策略已通过的证据。
导出器保留对该报告准确源码标识的识别。当前过滤策略的结果见
[log_report](./log_report/summary.md)；如需单独复现 LOG-MUL：

```bash
python3 scripts/check_numerics_supplement.py run --rules LOG-MUL --formats fp32 --output Logs/fp-log-mul-domain
python3 scripts/check_numerics_supplement.py report Logs/fp-log-mul-domain --output-dir Logs/fp-log-mul-domain-report
```

准入的是该实验对应的原子假设；带非零或正值要求的关系在 Lean 中必须保留这些要求。
exp 中间值的有限性、log 输入的正性等适用条件仍需在使用处处理。
这不要求在证明第二步固定实验 shape。

两个 reciprocal 例子和 stable softmax 的 FP 证明已完成，其余算法仍需以下工作：

- reciprocal 两例已将普通除法、精度、定义域和最终 cast 与对应原子衔接，完成 FP 证明。
- stable softmax 已用通过准入的 `libdevice.exp` EXP-SUB 和基础算术原子完成证明。
  PR #12 的 fp32 `tl.exp` EXP-SUB-INTRINSIC 测得 B=0.1608954387 > 0.05，
  不满足当前准入条件；不能用 libdevice 的结果替代它的结果。
- logsumexp 的 libdevice EXP-SUB 已接入；本轮 LOG-MUL 为 bias INCONCLUSIVE，
  `tl.log(libdevice.exp(a)) = a` 为 bias FAIL，两个 log 前提均未准入。
- online softmax 已完成公开 FP specification，比较 batch 输出和实际在线 m/l
  寄存器的归一化值，沿用原 Correct 的观察范围。
- Welford/LayerNorm 已通过专用导出器接入 count 零转换和有界 successor，
  完成 FP 证明并保留 `N <= 2^24`。Welford 要求非空行；LayerNorm 的空行没有
  输出写入，仍被覆盖。空行的实数总除法行为没有用作 FP 的 `0/0` 定律。
- 原 online softmax 只维护 m/l 寄存器，没有输出 store；仍保留其原有作用范围。

sub-zero 等可以从已选加法/CANCEL 和 add-zero 推导的关系，不重复登记。
现有 atom 的 fp32 实例与其最终转换必须对应；bf16 输出后的等式不能冒充 fp32 中间值等式。

## 本地验证

```bash
python3 -m unittest scripts.test_numerics_supplement -v
python3 -m unittest scripts.test_numerical_domains -v
TRITON_INTERPRET=1 python3 -m unittest scripts.test_numerics_supplement -v
python3 scripts/check_supplement_kernels.py
python3 scripts/export_numerical_rules.py --trust-report --check
python3 scripts/export_supplemental_rules.py --trust-report --check
```

默认浮点 profile 有 62 对表达式。CPU 解释器检查其中 41 对的精度和非整块矩形索引；
十五个 libdevice 组合明确跳过，因为解释器不支持 CUDA `extern_elementwise`，
这些组合用离线编译和真实 GPU 实验验证，不替换其函数实现。另有 profile、定义域、
fp64 除法残差、报告、源/PTX/观测/配置/统计篡改检测测试。离线编译不需要 GPU，
默认目标 sm_80。解释器和离线编译结果均不是 GPU two-gates 准入结果。

## Lean 接入

`python3 scripts/export_supplemental_rules.py --trust-report` 根据本报告生成
`VeriTile/Triton/Float/SupplementalAdmission.lean`，只收录当前报告的 ACCEPT 实例。
总数同时读取主目录和补充目录，避免硬编码某次实验的通过数。导出不重放 GPU，也不生成 `EvidenceValidated`
证明；沿用用户信任报告、在证明中显式提供原子假设的接口。

`SoftmaxReciprocalFPEquiv.lean` 和 `FloatDTypeSoftmaxFPEquiv.lean` 使用各自精度的
`div_mul_rcp` 原子，保留有限/非零定义域和最终输出转换。第二个例子只在 fp32
输出处应用 fp64-work 结果。默认打印仍只有实际使用的原子名称。
