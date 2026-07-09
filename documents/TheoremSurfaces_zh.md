# Examples 定理 surface 风格

`VeriTile/Examples/` 中面向用户的定理 surface 都应当从公开的 compute-facing
API 出发。完整 user guide 见
[`CorrectnessSurfaces.md`](./CorrectnessSurfaces.md)。

- 单 kernel 对照数学或算法 spec 的 correctness 用
  `ComputeCorrect.Realizes`(*一个 kernel realize 某个 spec*)、
  `ComputeCorrect.Post` 或 `ComputeCorrect.General`。
- 双 kernel 等价或 rewrite refinement 用 `ComputeRefine.Refines`
  (*一个 kernel refine 另一个* —— 两个终态 memory 在所有非 scratch region 上
  writes-equality)、pointwise 的 `ComputeRefine.RefinesAt`、
  `ComputeRefine.Post` 或 `ComputeRefine.General`。
- 窄浮点 kernel 的 correctness 落在舍入 surface 上:不带限定词的
  `ComputeRefine.Realizes`(单 kernel)和 `Refines` / `RefinesAt`(双 kernel)都
  接受一个 `RoundingModel R`,在 `execR` 下执行;对应的精确实数理想化是它们的
  `*_without_Rounding` 变体。见 [`CorrectnessSurfaces.md`](./CorrectnessSurfaces.md)
  及 showcase `bench/examples/FusedSwiglu.lean`。

投影后的算法 lemma 仍然可以提到 `Kernel.Correct_without_Rounding` 或 `Kernel.Refine`,
但仅限于明确属于内部 bridge lemma。这些 lemma 不应当作为
`scripts/kernel-manifest.tsv` 里登记的公开 example 定理。

## 命名

- 单 kernel correctness 定理:`<name>_correct_view`
- 双 kernel refinement 定理:`<name>_refinement_view`

仅供执行的 helper lemma 可以使用 `_exec_view` 后缀,允许直接陈述
`exec` 等式。公开定理如果是输出观察定理,应当把这种 helper 用
`ComputeCorrect.Realizes`(单 kernel spec)或 `ComputeRefine.Refines`
(双 kernel writes-equality)包装起来。

非普通单 kernel / 双 kernel example view 的领域专用定理 surface,
比如 whole-grid launch 事实或 FlashAttention 专用的数学/trace 陈述,
可以保留已有的描述性名字。当公开 artifact 定理被改名以遵循上述风格时,
`scripts/kernel-manifest.tsv` 必须在同一个 commit 里更新。
