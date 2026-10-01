# 从已接受原语到 TritonBench 等价性证明

本例使用 PR #9 的已发布结果，按用户选择直接信任报告，不重新运行或重放 GPU 实验。
`scripts/export_numerical_rules.py` 检查配置、源码哈希、报告完整性和接受状态，将 42 个实例中的 30 个接受项导出到
[ReportedAdmission.lean](../../VeriTile/Triton/Float/ReportedAdmission.lean)。WARN、REJECT、定义域事件及不适用条目均无对应的导出规则。

## 用户看到的 specification

[TritonBenchVectorAdditionFP.lean](../../bench/examples/TritonBenchVectorAdditionFP.lean)
复用 TritonBench `vector_addition` 的结构，具体化为 `4096×4096` 个元素、block=1024、fp32 输入/计算/输出。
两个 kernel 唯一的差异是 `output = x + y` 与 `output = y + x`：

```lean
specification vector_addition_fp_equiv (R : Rules) :
    originalKernel ≡[R] optimizedKernel := by
  refine ⟨rfl, ?_⟩
  change Spec.Derivation R.assumptions (body originalKernel) (body optimizedKernel)
  rw [original_decomposition, optimized_decomposition]
  exact .frame beforeAdd afterAdd (admitted_add_commute R)
```

`R` 固定引用导出的 `fp32_add_commute`；用户不再提供任意 contract/evidence，也不手填 PASS。
它只包含一条数值模型假设：信任该报告中的 ADD-COMMUTE 及其与指定 fp32 加法片段的对应关系。
这个前提保留为 `Rules.add_comm`，它正是“实验通过后 assume 该原子关系”的逻辑表达。
Lean 证明只交换加法输入，检查加载、地址、mask、store 和程序签名保持一致。
Lean 不证明 GPU 确实运行过，也不把统计接受转换成 IEEE 位值等式；本例没有引入全局公理。

`real_projection` 检查 fp32 版本的数学投影等于原来的 TritonBench kernel，已有实数正确性证明继续适用。
实验 shape 是完整数据的形状；kernel 以 block=1024 遍历其连续展平存储。每个被重写的加法直接读取原始独立正态输入，未把中间值擅自当成同分布输入。

## 运行和检查

```bash
python3 scripts/export_numerical_rules.py --trust-report
lake build VeriTile.Meta.StatementAudit VeriTile.Triton.Float.Equivalence VeriTile.Triton.Float.ReportedAdmission TritonBenchSpecExamples
lake env lean bench/examples/TritonBenchVectorAdditionFP.lean
```

最后一条命令完成 Lean 检查并运行 `#print_spec`，其中应看到：

```text
Kind: FLOATING_POINT_EQUIVALENCE
Atom 1:
  rule ID: "ADD-COMMUTE"
  configuration summary: shape=[4096, 4096]; block=1024;
    input/compute/accumulator/output=fp32/fp32/fp32/fp32;
    independent Normal(mean=1.0, std=1.0); replicates=4096;
    bias=PASS; vars=PASS; U kind=empirical_max; trust=published report
  artifact: some "experiments/floating_point/report/summary.json#fp32/ADD-COMMUTE"
  bias gate: PASS
  vars gate: PASS
```

报告同时打印具体的报告快照标识、`Rules.add_comm` 前提和传递公理依赖。
`empirical_max` 保留原报告含义，不被改写为尾部置信上界。

冻结输入后的回归和现有官方 comparator 检查：

```bash
python3 scripts/export_numerical_rules.py --trust-report --check
python3 -m unittest scripts.test_export_numerical_rules scripts.test_specification_surface -v
python3 scripts/check_comparator.py --file bench/examples/TritonBenchVectorAdditionFP.lean --trust
```

comparator 使用独立输入快照；报告生成和原子绑定必须在证明任务开始之前完成。
如需调用已有证明 agent，入口仍是 `scripts/prove.sh`，目标为本例的完整 specification 名称，无需另一套证明器。

## 范围

导出表已有全部 30 个接受实例，本例完成了其中 fp32 ADD-COMMUTE 到实际 TritonBench 语句的绑定与组合证明。
其他原子的参数化语法绑定应按同一方式明确表达舍入/cast/FMA，不能把一个接受行任意套到不同片段。
导出的 `report:...` 标识绑定发布的 JSON 快照，不伪造仓库外原始 bundle 的 instance key、观测或 PTX 哈希。
更改报告会产生新的标识；更改数值实现会使源码哈希检查失败；更改本例的 shape、block 或精度会使 `report_matches` 编译失败。
任意复合算法仍需由已接受的局部关系推导，不能借这个导出入口把整个算法直接假设为等价。
