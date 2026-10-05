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

### log 实现、expm1/log1p 与条件式乘积分解

[log_config.py](./log_config.py) 保留 `4096×4096`、独立 `Normal(1,1)`、
fp32 输入/计算/输出及现有 two-gates 参数，只运行：

- `LOG-MUL`：`tl.log(a*b)` 与 `tl.log(a)+tl.log(b)`，跳过 `a<=0` 或 `b<=0` 的输入对。
- `LOG-EXP-LIBDEVICE`：`tl.log(libdevice.exp(a))` 与 `a`，保留负的有限 `a`。

第二条是新原子，不能复用旧 `LOG-EXP`（`tl.exp`）的实验结果。
PR #12 的 `tl.exp` exp-sub 不满足当前偏差条件，因此相关例子使用 `libdevice.exp`。
配对配置 [libdevice_log_config.py](./libdevice_log_config.py) 同时运行上述两项和
`LOG-MUL-LIBDEVICE`、`LOG-EXP-FULL-LIBDEVICE`。后两项将 log 改为
`libdevice.log`，分别复用对应原子的输入种子，其他设置相同；全部运算仍为 FP32。
当前配置 [log_accuracy_config.py](./log_accuracy_config.py) 另加入 `LOG-EXP-EXPM1`：
在 `abs(a) <= 0.5` 时计算 `libdevice.log1p(libdevice.expm1(a))`，范围外保留
`libdevice.log(libdevice.exp(a))`，候选仍为 `a`。阈值在运行前固定。
它保留所有有限输入，并复用 LOG-EXP-LIBDEVICE 的输入种子；不是只测试近零样本。
`tl.where` 两侧均会求值，因此未选路径的输入先置零，避免无用分支溢出或产生 `log1p(-1)`。
这是独立的新表达式，结果不能作为原 LOG-EXP 的准入证据。
完整配置 [log_product_config.py](./log_product_config.py) 同时运行七项，其中新增两项：

- `LOG-MUL-LOG1P`：参考侧在 `0.5 <= fp32(a*b) <= 1.5` 时用 `log1p(fma(a,b,-1))`；候选仍是两个 log 相加。
- `LOG-MUL-GUARDED`：参考侧保留 `log(fp32(a*b))`；候选在 `0.5 <= fp32(a*b) <= 2` 时也保留该表达式，范围外才拆为两个 log 相加。

两项均只使用 FP32，沿用同一正输入定义域、采样和 gate；分支范围在运行前固定。
本配置已在 H200 完成，DLC 任务 `dlc1hczsayjv5apb`，任务名
`traces_kernel_equivalence_testing`；独立 CPU 重放与 GPU 环境的表格完全一致。
有效输入上出现非有限输出仍算失败，不会被过滤。

| Rule | R | z | B（local ULP） | U | Accept |
|---|---:|---:|---:|---:|---|
| LOG-MUL / fp32 | 4096 | 2.924915168 | 0.1679352504 | 6.685560237 | 否：bias INCONCLUSIVE |
| LOG-MUL-LIBDEVICE / fp32 | 4096 | 2.924915168 | 0.1679352504 | 6.685560237 | 否：bias INCONCLUSIVE |
| LOG-EXP-LIBDEVICE / fp32 | 4096 | 68.59740095 | 0.7300322166 | 0 | 否：bias FAIL |
| LOG-EXP-FULL-LIBDEVICE / fp32 | 4096 | 68.59740095 | 0.7300322166 | 0 | 否：bias FAIL |
| LOG-EXP-EXPM1 / fp32 | 4096 | 25775.46400 | 0.0494428110 | 0 | 是 |
| LOG-MUL-LOG1P / fp32 | 4096 | 2.061440595 | 0.1240315873 | 6.685560237 | 否：bias INCONCLUSIVE |
| LOG-MUL-GUARDED / fp32 | 4096 | 296.6221498 | 0.0006248690 | 6.685560237 | 是：条件式 |

换成 `libdevice.log` 没有改变结果。在本次 Triton 3.7.1 / CUDA 13.0、sm_90
和记录的编译设置下，两种 log 的 PTX 都包含 `__nv_logf` 实现；每对两侧的 PTX
仅去掉 `.file` / `.loc` 调试指令后完全一致，没有 `lg2.approx`。
每次 replicate 的 delta、两侧最大绝对误差和有效样本数也逐项相同。
[comparison.json](./log_report/comparison.json) 保存配对种子、观测比较及 PTX 哈希。
这只描述本次编译配置，不代表所有 Triton 版本或编译选项都如此。

LOG-MUL 已完成有效统计，保留 48,643,849,868 个正输入对，跳过
20,075,626,868 个定义域外输入对；bias 区间跨过 0.05 的预算边界。
LOG-EXP-LIBDEVICE 的平均偏差为 -0.6804358853 local ULP，五个标准误区间
完全落在容差范围外。它的 U=0 是因为候选 `a` 对 oracle 没有误差，
并不表示参考 `tl.log(libdevice.exp(a))` 和候选相等。

七项的 U 都通过；LOG-EXP-EXPM1 和 LOG-MUL-GUARDED 满足 bias 预算。前者的平均偏差为
-0.04943322176 local ULP，SE=0.000001917840228，B=0.04944281096，
距离 0.05 预算约 0.00055718904 ULP。z 很大表示偏差稳定可测，不等于超预算。
这不是零偏差或所有输入上的精确恒等式，也没有准入原 LOG-EXP 表达式。

[GPU 边界检查](./log_report/boundaries.json) 验证 15 个输入：`a=±2^-25` 时
原实现得到 0，新实现恢复 `a`；±0.5 的相邻 FP32 数检查分段边界，`a=-20`
使用原路径，`a=90` 的溢出仍保留为非有限结果。PTX 确认新表达式没有 FP64 运算。
条件式 LOG-MUL 的主实验均值为 -0.0006145105 local ULP。独立种子 20261005
复核同样通过：z=296.0360072、B=0.0006251552、U=7.418400148。两次各有
4096 个 replicate；两组完整表均经独立 CPU 回放核对。
[独立种子配置](./log_product_validation_config.py) 与 [复核结果](./log_product_validation_report/summary.md)
属于同一版本的独立验证，不是历史结果。FMA/log1p 方案在复核中仍为 INCONCLUSIVE。

[乘积诊断](./log_report/log_product_diagnostics.json) 包含 16 个 GPU 边界输入，
以及最初 32 批采样的分支统计：43.6002% 的有效输入保留乘积路径，56.3998%
仍使用两个 log 相加。原式在保留区间观察到 33,554,432 ULP 的单元素差值；
区间外的观察最大值只有 4 ULP。这些诊断不代替正式准入实验。
条件式不修改输入掩码，也不证明可以无条件拆分 log；未更改 Lean 原子绑定。

原 LOG-MUL 仍未准入，原 StableLogSumExp 实现的 log-exp 前提也未满足，
因此 StableLogSumExp 仍未完成。
完整当前结果见 [log_report/summary.md](./log_report/summary.md)，
数值审核见 [log_report/warning_audit.json](./log_report/warning_audit.json)。

当前报告的 Lean 数据位于
[`LogAdmission.lean`](../../../VeriTile/Triton/Float/LogAdmission.lean)，
原子绑定位于 [`LogExp.lean`](../../../VeriTile/Triton/Float/LogExp.lean)。独立表包含
`fp32_log_exp_expm1` 与 `fp32_log_mul_guarded` 两项；条件式乘积的数据保留正有限输入、
FP32 精度和 `0.5 <= fp32(a*b) <= 2` 的分支表达式，尚未接入 Lean 原子绑定。
已绑定的 `log_exp_expm1` 保留 fp32 精度、有限输入定义域、`abs(a) <= 0.5`
及未选分支传零的源表达式。`libdevice.log`、`libdevice.expm1`、
`libdevice.log1p` 都是独立的 FP 运算符，不会和 `tl.log` 混用。
`log_exp_expm1_equiv` 使用原有 `≡[R]` 接口；打印的假设只有
`log_exp_expm1`。`apply_rule` 从成功执行的标量片段导出该规则，比较运算
也保留精度；未提供比较解释时，执行失败，不能据此推出数值恒等式。

[`LogExpCounterexample.lean`](../../../VeriTile/Triton/Float/LogExpCounterexample.lean)
把反例范围分开写清：给定报告中的 `logExp(2^-25)=0`，可否定普通
log-exp 的逐点精确恒等式；再给定 `logExp(0)=0`，单元素 LSE 的 direct
和 stable 公式在 bf16 输出后分别为 `0` 与 `2^-25`，仍不相等。
Lean 检查位模式、减法、加法及转换；libdevice 的两次复合求值显式作为
实验前提。边界报告测的是 `libdevice.log(libdevice.exp(a))`，没有直接
测完整 LSE kernel，也没有将 `tl.log` 与 `libdevice.log` 全局等同。
这些反例否定精确恒等式，不否定原实数正确性，也不单独否定整个 LSE
在某个分布下的 two-gates 准入。新分段表达式自身也不是精确恒等式：
报告中的 `a=-0.4999999701976776` 得到 `-0.5`，但整体统计满足准入条件。

按已提交报告重新生成新表（无需重放 GPU 实验）：

```bash
python3 scripts/export_supplemental_rules.py --trust-report \
  --report experiments/floating_point/supplement/log_report \
  --namespace LogAdmission --output VeriTile/Triton/Float/LogAdmission.lean
```

在仓库根目录、安装下述依赖后执行：

```bash
python3 scripts/check_numerics_supplement.py check --profile experiments/floating_point/supplement/log_product_config.py
python3 scripts/check_log_accuracy.py --output Logs/fp-log-accuracy-boundaries
python3 scripts/check_log_product.py --output Logs/fp-log-product-diagnostics
python3 scripts/check_numerics_supplement.py run --profile experiments/floating_point/supplement/log_product_config.py --smoke --output Logs/fp-log-accuracy-smoke
python3 scripts/check_numerics_supplement.py run --profile experiments/floating_point/supplement/log_product_config.py --output Logs/fp-log-accuracy
python3 scripts/check_numerics_supplement.py report Logs/fp-log-accuracy --output-dir Logs/fp-log-accuracy-report
```

复现时先确认 smoke 没有 `ERROR`，再运行正式实验并带回 `Logs/fp-log-accuracy-report/`。
本轮使用 `scalar-supplement-11`，请用新目录；当前 log 报告维护这七项配对结果。

独立种子复核：

```bash
python3 scripts/check_numerics_supplement.py run --profile experiments/floating_point/supplement/log_product_validation_config.py --output Logs/fp-log-product-validation
python3 scripts/check_numerics_supplement.py report Logs/fp-log-product-validation --output-dir Logs/fp-log-product-validation-report
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
| exp-sub | `exp(a - b) → exp(a) / exp(b)` | 推导移位与缩放 |
| exp-zero | `exp(0) → 1` | 指数初值 |
| log-mul | `log(a * b) → log(a) + log(b)` | 对数因子分解，`a,b > 0` |
| log-mul-libdevice | `libdevice.log(a * b) → libdevice.log(a) + libdevice.log(b)` | 配对比较 log 实现，`a,b > 0` |
| log-mul-log1p | 乘积近 1 时参考用 `log1p(fma(a,b,-1))`，候选两个 log 相加 | 仅 FP32；两种子均未准入 |
| log-mul-guarded | 候选在 `0.5 <= fp32(a*b) <= 2` 时保留参考，范围外拆分 log | 仅 FP32；只准入条件式 |
| log-exp | `tl.log(tl.exp(a)) → a` | 原 intrinsic 组合 |
| log-exp-libdevice | `tl.log(libdevice.exp(a)) → a` | 当前 fp32 实验因 bias 拒绝 |
| log-exp-full-libdevice | `libdevice.log(libdevice.exp(a)) → a` | 两个函数都使用 libdevice，当前 fp32 实验因 bias 拒绝 |
| log-exp-expm1 | `abs(a) <= 0.5` 时 `log1p(expm1(a))`，其余 `log(exp(a))`，与 `a` 比较 | 所有函数用 libdevice，独立 FP32 表达式 |
| max-commute | `max(a,b) → max(b,a)` | max 标量换序 |
| max-assoc | `max(max(a,b),c) → max(a,max(b,c))` | max 标量重组 |
| max-idem | `max(a,a) → a` | 消去重复 max 项 |
| max-neg-inf | `max(-inf,a) → a` | online max 的初值 |
| exp-neg-inf-sub | `exp(-inf - a) → 0` | online 指数权重的初值，`a` 有限 |

每条关系只有固定数量的标量操作。最后一条保留 `-inf` 的局部表达式，
避免把无穷大当成普通有限数套进 exp-sub；它不是 online-softmax 整体关系。
常数和恒等式可能被编译器折叠，保存的 PTX 反映实际执行图。

目录现在包含 23 条关系。默认浮点 profile 有 **56 个可执行实例、112 个左右两侧 kernel 特化**，
外加 1 个 fp64 残差 oracle；完整笛卡尔表有 92 行，其余组合明确标为 `UNSUPPORTED`。
其中 EXP-SUB-INTRINSIC、LOG-EXP-EXPM1、LOG-MUL-LOG1P 和 LOG-MUL-GUARDED 只支持 fp32 输入、计算和输出。COUNT-ZERO、COUNT-SUCCESSOR
使用独立的 int32 输入配置，不能用正态浮点输入替代，见
[三个新增 primitive 的配置与当前结果](../primitives/README.md)。

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
EXP-SUB、LOG-EXP-LIBDEVICE、LOG-EXP-FULL-LIBDEVICE 和 LOG-EXP-EXPM1 使用 `libdevice.exp`；
其他 exp 原子使用 `tl.exp`。LOG-MUL-LIBDEVICE 和 LOG-EXP-FULL-LIBDEVICE 使用
`libdevice.log`，其他 log 原子使用 `tl.log`，max 使用 `tl.maximum`。
LOG-EXP-EXPM1 使用 `libdevice.log` 及 `libdevice.log1p` / `libdevice.expm1` 分段路径。
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

默认浮点 profile 有 56 对表达式。CPU 解释器检查其中 41 对的精度和非整块矩形索引；
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
