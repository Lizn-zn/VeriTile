# 补充基础原子实验

在 `codex/fp-example` 分支运行。本目录补充剩余 FP 例子缺少的标量关系；
现有 30 条准入实例、原实验源码和 two-gates 算法保持不变。
这里没有 softmax、Welford、LayerNorm 或整个 reduction 的准入原子。
当前尚无本批 GPU 结果，Lean 可用假设仍是原来的 30 条。

## 直接运行

沿用上次的 NVIDIA CUDA 环境。新环境安装：

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
即使 PASS 也只记为 `SMOKE_ONLY`，不准入。`LOG-MUL` 的域事件及 fp64 profile
中不适用的关系是预期状态。正式实验不会自动缩小 shape。

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
新报告使用独立 schema，不能直接覆盖旧 `report/` 或送入旧 Lean 导出器。
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
| log-exp | `log(exp(a)) → a` | 对数与指数消去 |
| max-commute | `max(a,b) → max(b,a)` | max 标量换序 |
| max-assoc | `max(max(a,b),c) → max(a,max(b,c))` | max 标量重组 |
| max-idem | `max(a,a) → a` | 消去重复 max 项 |
| max-neg-inf | `max(-inf,a) → a` | online max 的初值 |
| exp-neg-inf-sub | `exp(-inf - a) → 0` | online 指数权重的初值，`a` 有限 |

每条关系只有固定数量的标量操作。最后一条保留 `-inf` 的局部表达式，
避免把无穷大当成普通有限数套进 exp-sub；它不是 online-softmax 整体关系。
常数和恒等式可能被编译器折叠，保存的 PTX 反映实际执行图。

14 条关系均测原来的三个精度 profile；额外测一条 fp64-work 的 div-mul-rcp。
因此是 **43 个可执行实例、86 个左右两侧 kernel 特化**，外加 1 个 fp64 残差 oracle。完整笛卡尔表有 56 行，
其中另外 13 个 fp64 组合明确标为 `UNSUPPORTED`，不会制造准入记录。

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

`/` 是普通 Triton division，**不是**原 `DIV-RCP` 的 `tl.div_rn`。
exp/log/max 分别使用 `tl.exp`、`tl.log`、`tl.maximum`；禁止隐式 FMA fusion。

只跑某组关系可以显式选择：

```bash
python3 scripts/check_numerics_supplement.py run --rules ADD-ZERO,MUL-ONE,DIV-ONE,DIV-MUL-RCP,MUL-RCP-CANCEL --formats fp32 --output Logs/fp-supplement-arithmetic
```

## 定义域与后续证明

只按指定分布采样，不添加非对称探针，不取绝对值、截断或重采样。
**默认 Normal(1,1) 会让 log-mul 很容易在首个 replicate 遇到非正输入**，
该实例将是 `INCONCLUSIVE`，不是通过；需要另一组分布时由用户显式修改配置并开新实验。
采样到零分母也记录域事件。实际计算产生非有限输出/误差则由原 gates 拒绝。

准入的是该实验对应的原子假设；带非零或正值要求的关系在 Lean 中必须保留这些要求。
exp 中间值的有限性、log 输入的正性等适用条件仍需在使用处处理。
这不要求在证明第二步固定实验 shape。

这批对应剩余七个例子的数值缺口，但不保证每个原子会通过，也不自动完成剩余证明：

- reciprocal 两例需要将普通除法和 dtype/cast 与准确实验关系对应。
- stable softmax、logsumexp、online softmax 还需要从基础关系推导 reduction/loop 不变量。
- Welford/LayerNorm 还需要 count 的自然数转换及非零/可表示性证明。
  正态浮点采样不能证明任意整数 count 的 cast-successor 恒等式；本批没有伪造此类原子。
  空行的实数总除法行为也不能直接当作 IEEE 浮点的 `0/0` 行为。
- 原 online softmax 只维护 m/l 寄存器，没有输出 store；仍保留其原有作用范围。

sub-zero 等可以从已选加法/CANCEL 和 add-zero 推导的关系，不重复登记。
现有 atom 的 fp32 实例与其最终转换必须对应；bf16 输出后的等式不能冒充 fp32 中间值等式。

## 本地验证

```bash
python3 -m unittest scripts.test_numerics_supplement -v
TRITON_INTERPRET=1 python3 -m unittest scripts.test_numerics_supplement -v
python3 scripts/check_supplement_kernels.py
python3 scripts/export_numerical_rules.py --trust-report --check
```

解释器检查涵盖全部 43 对表达式、精度和非整块矩形索引；另有 profile、定义域、
fp64 除法残差、报告、源/PTX/观测/配置/统计篡改检测测试。离线编译不需要 GPU，
默认目标 sm_80。解释器和离线编译结果均不是 GPU two-gates 准入结果。
