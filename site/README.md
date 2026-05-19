# VeriTile docs site

Astro + [Starlight](https://starlight.astro.build/) docs site for VeriTile.
Bilingual (English root, `/zh-cn/` Chinese mirror), deployed to
[lizn-zn.github.io/VeriTile/](https://lizn-zn.github.io/VeriTile/) via the
`Deploy docs site` GitHub Actions workflow.

## Quick start

```bash
./site/scripts/dev.sh           # dev server on http://localhost:4321/VeriTile/
./site/scripts/dev.sh build     # one-shot production build to dist/
./site/scripts/dev.sh preview   # build + serve the production output
```

Requires [Bun](https://bun.sh). No Node / npm needed.

## Layout

```text
site/
├── astro.config.mjs            ← site URL, base path, sidebar, i18n
├── package.json                ← dev / build / preview / astro scripts
├── public/                     ← static files copied as-is
├── scripts/
│   ├── dev.sh                  ← one-touch dev server (entry point)
│   └── migrate-docs.sh         ← re-sync content from ../documents/
└── src/
    ├── components/Hero.astro   ← custom homepage hero
    ├── styles/theme.css        ← engineering-notebook theme tokens
    └── content/docs/
        ├── (root)              ← English (no locale prefix)
        └── zh-cn/              ← Chinese mirror
```

## Authoring

Each page in `src/content/docs/` is a Markdown / MDX file with frontmatter:

```yaml
---
title: Page title
description: One-line summary used in nav previews and SEO.
---
```

Internal links use absolute paths **with the `/VeriTile/` base prefix**:

```markdown
[Cookbook](/VeriTile/cookbook/)
```

The base prefix is currently hard-coded so the same Markdown serves dev
(`http://localhost:4321/VeriTile/...`) and prod
(`https://lizn-zn.github.io/VeriTile/...`).

Bench / scripts links should point to GitHub:

```markdown
[`bench/check_ports.sh`](https://github.com/Lizn-zn/VeriTile/blob/main/bench/check_ports.sh)
```

## Re-syncing from `documents/`

`documents/*.md` is the source of truth for the design notes; the
architecture & proofs sections of this site are copies. To re-sync after
upstream edits:

```bash
cd site && ./scripts/migrate-docs.sh
```

The script extracts the H1 as `title:`, drops the bilingual switcher line,
and re-rewrites cross-doc links to site paths. Idempotent — safe to re-run.

## Deployment

Pushing to `main` with changes under `site/**` triggers
`.github/workflows/site.yml`, which runs `bun install` + `bun run build`
and uploads `site/dist/` as a Pages artifact. First-time setup needs the
repo's **Settings → Pages → Source** set to **"GitHub Actions"**; after
that, every push is automatic.

The workflow can also be triggered manually from the Actions tab
(`workflow_dispatch`).
