#!/usr/bin/env bash
# One-shot migration: documents/*.md → site/src/content/docs/{bucket}/{slug}.md
# Idempotent — safe to re-run after edits to documents/.

set -euo pipefail
cd "$(dirname "$0")/.."

SRC_DIR="../documents"
DEST_EN="src/content/docs"
DEST_ZH="src/content/docs/zh-cn"

# name → "bucket slug"
declare -a DOCS=(
  "CodeOrganization      architecture code-organization"
  "TritonSubset          architecture triton-subset"
  "GpuMemoryModel        architecture gpu-memory-model"
  "EraseDType            architecture erase-dtype"
  "ConcurrencySemantics  architecture concurrency-semantics"
  "MemorySafety          architecture memory-safety"
  "ProofConventions      proofs       proof-conventions"
  "TheoremSurfaces       proofs       theorem-surfaces"
  "CorrectnessSurfaces   proofs       correctness-surfaces"
  "ApproxGeluPhiStrategy proofs       approx-gelu-phi-strategy"
  "KernelManifest        proofs       kernel-manifest"
)

# Build sed expressions to rewrite intra-doc relative links
build_link_rewrites () {
  local out=""
  for row in "${DOCS[@]}"; do
    read -r name bucket slug <<< "$row"
    out+="s|](${name}.md)|](/${bucket}/${slug}/)|g;"
    out+="s|](${name}_zh.md)|](/zh-cn/${bucket}/${slug}/)|g;"
    out+="s|](./${name}.md)|](/${bucket}/${slug}/)|g;"
    out+="s|](./${name}_zh.md)|](/zh-cn/${bucket}/${slug}/)|g;"
  done
  printf '%s' "$out"
}

REWRITES="$(build_link_rewrites)"

migrate_one () {
  local src="$1"          # path to source markdown
  local dst="$2"          # path to destination
  local lang="$3"         # 'en' or 'zh'
  local fallback_title="$4"

  # Extract H1 as title (strip leading '# ' and trailing whitespace)
  local title
  title=$(awk '/^# / { sub(/^# /, ""); print; exit }' "$src" || true)
  [[ -z "$title" ]] && title="$fallback_title"
  # Escape any double quotes in title for YAML
  title=${title//\"/\\\"}

  mkdir -p "$(dirname "$dst")"

  # Build the file:
  # 1. Frontmatter
  # 2. Body — drop:
  #    - first H1 line
  #    - the bilingual switcher line (anywhere in the first ~6 lines that
  #      contains both **English** and 中文 markers, or **中文**)
  #    - leading blank lines before the first content paragraph
  # 3. Rewrite cross-doc relative links
  {
    printf -- '---\n'
    printf 'title: "%s"\n' "$title"
    printf -- '---\n\n'
    awk '
      BEGIN { dropped_h1 = 0; dropped_switcher = 0 }
      NR == 1 && /^# / { dropped_h1 = 1; next }
      !dropped_switcher && NR <= 8 && /(\*\*English\*\*.*中文|English\].*\*\*中文\*\*)/ {
        dropped_switcher = 1; next
      }
      { print }
    ' "$src" | sed -E "$REWRITES" | awk 'BEGIN{blank=1} NF { blank=0 } { if (!blank || NR>1) print }'
  } > "$dst"
}

for row in "${DOCS[@]}"; do
  read -r name bucket slug <<< "$row"
  echo "→ $name → $bucket/$slug"
  migrate_one "$SRC_DIR/${name}.md"      "$DEST_EN/$bucket/$slug.md" en "$name"
  if [[ -f "$SRC_DIR/${name}_zh.md" ]]; then
    migrate_one "$SRC_DIR/${name}_zh.md" "$DEST_ZH/$bucket/$slug.md" zh "$name"
  fi
done

echo
echo "migrated $(printf '%s\n' "${DOCS[@]}" | wc -l | tr -d ' ') docs × 2 langs."
