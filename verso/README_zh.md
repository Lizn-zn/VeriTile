# VeriTile 架构概览 —— Verso 源

用 [Verso](https://github.com/leanprover/verso) 把架构概览渲染为
self-contained 的 HTML "deck"。`VeriTileOverview.lean` 中每个顶层
`# Heading` 在渲染输出中成为一张幻灯;arrow keys / `PageUp` /
`PageDown` / `Space` 翻页,`f` 切换全屏。

这是一个 **独立的子项目**,pin 自己的 Lean toolchain
(`v4.30.0-rc2`,匹配 Verso `main`)—— 与父 VeriTile 项目的
`v4.29.0` 独立。它 **不** 导入任何 VeriTile Lean 代码;架构内容是
普通的 Verso markdown 加自定义 block 组件。

## 布局

```text
verso/
├── lakefile.toml                       ← Lake 项目 + verso 依赖
├── lean-toolchain                      ← v4.30.0-rc2
├── VeriTileOverview.lean               ← #doc (Page) "..." => …  (deck 内容)
├── VeriTileSite.lean                   ← Site + slide 风格主题
├── Main.lean                           ← `blogMain` 入口
├── VeriTile_Architecture_Overview.html ← 预渲染 deck,便于快速预览
└── static_files/
    ├── slides.css                      ← 幻灯展示 CSS
    └── slides.js                       ← 方向键导航
```

## 构建

在 `verso/` 下:

```bash
# 第一次:fetch Verso(下载较大,需要几分钟)
lake update
lake build

# 渲染站点到 ./_site/
lake exe veritile-overview --output _site
```

然后用浏览器打开 `_site/index.html`。Arrow keys / `PageUp` /
`PageDown` / `Space` 翻页;按 `f` 切换全屏。

## 迭代

编辑 `VeriTileOverview.lean` 改内容 —— 每个 `# Heading` 控制 section 级
结构与幻灯断点。视觉 / 布局调整改 `static_files/slides.css`。
导航行为改 `static_files/slides.js`。改 `.lean` 文件后重新跑
`lake build`;static file 在 site-render 时拷贝,所以你只需要重新跑
renderer。

## 自定义 block 组件

定义在 `VeriTileOverview.lean` 顶部:

| Directive       | 渲染为                                  |
|-----------------|-----------------------------------------|
| `:::cardBlue`   | `<div class="card card-blue">…</div>`   |
| `:::cardOrange` | `<div class="card card-orange">…</div>` |
| `:::cardGreen`  | `<div class="card card-green">…</div>`  |
| `:::cardPurple` | `<div class="card card-purple">…</div>` |
| `:::cardMuted`  | `<div class="card card-muted">…</div>`  |
| `:::cols`       | `<div class="cols cols-2">…</div>`      |
| `:::cols3`      | `<div class="cols cols-3">…</div>`      |
| `:::pipeline`   | `<div class="pipeline">…</div>`         |
| `:::numgrid`    | `<div class="numgrid">…</div>`          |

在 card 里,第一句 `**bold sentence**` 通过 CSS 成为带标签的 cap
("INPUT" / "WHAT WE DID" / "OUTPUT")。

## 备注

- Verso 的 `lean4:autoprove` 风格高亮要求片段是真实的 Lean 程序。
  本 deck 中的片段是示意性的,使用未带语言标签的 ` ``` ` 代码块
  (无语言 tag),让 Verso 把它们 emit 为普通的 `<pre><code>`。
- 如果 Verso 站点主题改了顶层 wrapper class,可能需要调整 CSS 规则
  `header, footer { display: none; }`。
- 与本 README 同目录有一个预渲染的单文件 deck
  `VeriTile_Architecture_Overview.html` —— 同样的内容,不需要 Lean
  toolchain。
