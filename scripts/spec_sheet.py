#!/usr/bin/env python3
"""spec_sheet.py — emit one self-contained "spec sheet" per tritonbench_g kernel.

For each kernel .lean file, find the public summary theorem(s), pull in the
transitive closure of the spec `def`s its statement references (with their
docstrings), and render a single markdown card so a reviewer can audit the
*specification* — "what does `expected` claim the output is?" — without
spelunking a 1500-line proof file.

Audit flags surfaced per sheet:
  - SELF-REF RISK: a referenced spec def (or the theorem conclusion's RHS)
    mentions exec / toAlgKernel / *SurfaceValue / produced*Value, i.e. the
    closed form may be defined in terms of the executed kernel output rather
    than purely over INPUT memory.
  - The theorem's hypotheses (binders) are listed so layout/shape contracts
    are visible in one place.

Usage:
  scripts/spec_sheet.py                      # all kernels -> out dir
  scripts/spec_sheet.py <file.lean> ...      # specific files, print to stdout
  scripts/spec_sheet.py --out DIR            # write *.md sheets to DIR
"""
from __future__ import annotations
import re, sys, os, glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(ROOT, "bench", "tritonbench_g")

DECL_KW = r"(?:noncomputable\s+|private\s+|protected\s+|partial\s+|@\[[^\]]*\]\s*)*" \
          r"(def|denotation|theorem|specification|lemma|abbrev|instance)"
# A top-level decl starts at column 0 (these files put decls at col 0 even
# inside namespace/section).
DECL_RE = re.compile(r"^(" + DECL_KW + r")\s+([A-Za-z_][A-Za-z0-9_'?!.]*)", re.M)
IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_'?!]*")
SELFREF_TOKENS = ("exec", "toAlgKernel", "toAlgorithm", "SurfaceValue",
                  "produced", "readMem")  # readMem alone is fine on INPUT s;
# we only flag readMem when applied to an *executed* state below.


def split_decls(text):
    """Return list of (name, kind, header_start, body_start, end, docstring)."""
    lines = text.split("\n")
    # offsets of each line start
    offs, p = [], 0
    for ln in lines:
        offs.append(p); p += len(ln) + 1
    decls = []
    matches = list(DECL_RE.finditer(text))
    for i, m in enumerate(matches):
        kind = m.group(2)
        name = m.group(3)
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        # grab preceding /-- ... -/ docstring if directly above
        doc = ""
        pre = text[:start].rstrip()
        if pre.endswith("-/"):
            ds = pre.rfind("/--")
            db = pre.rfind("/-!")
            anchor = max(ds, db)
            if anchor != -1:
                doc = pre[anchor:].strip()
        decls.append(dict(name=name, kind=kind, start=start, end=end,
                          text=trim_decl(text[start:end]), doc=doc))
    return decls


def trim_decl(t):
    """Drop the next decl's leading docstring / set_option / attribute prelude
    that the column-0 span heuristic pulls into the tail of this decl."""
    t = t.rstrip()
    changed = True
    while changed:
        changed = False
        if t.endswith("-/"):
            a = max(t.rfind("/--"), t.rfind("/-!"))
            if a != -1:
                t = t[:a].rstrip(); changed = True
        m = re.search(r"\n[ \t]*(set_option|attribute|@\[|open\b|section\b|namespace\b|end\b)[^\n]*$", t)
        if m:
            t = t[:m.start()].rstrip(); changed = True
    return t


def split_statement(decl_text):
    """For a theorem, split into (signature_up_to_first_:=, proof)."""
    # find ':=' that begins the proof (top-level). Heuristic: first ' := ' or
    # ':= by' at the top — adequate for these files.
    m = re.search(r":=\s*by\b", decl_text)
    if not m:
        m = re.search(r"\n\s*:=", decl_text) or re.search(r":=", decl_text)
    if m:
        return decl_text[:m.start()].rstrip(), decl_text[m.start():]
    return decl_text.rstrip(), ""


def strip_comments(t):
    """Remove /- … -/ (nested) and -- line comments. Used for SCANNING only;
    display keeps comments (the inline `-- (1)` notes aid review)."""
    out, i, depth, n = [], 0, 0, len(t)
    while i < n:
        if t[i:i + 2] == "/-":
            depth += 1; i += 2
            while i < n and depth > 0:
                if t[i:i + 2] == "/-": depth += 1; i += 2
                elif t[i:i + 2] == "-/": depth -= 1; i += 2
                else: i += 1
            continue
        if t[i:i + 2] == "--":
            j = t.find("\n", i); i = j if j != -1 else n
            continue
        out.append(t[i]); i += 1
    return "".join(out)


def body_of(decl):
    """Return the def's value (after :=) or theorem statement for closure scan.
    Comments stripped so trailing docstrings of the *next* decl can't leak
    identifiers/self-ref tokens into this decl's scan span."""
    if decl["kind"] in ("def", "abbrev", "instance"):
        # value is after the first := (skip proof heuristic; defs use term :=)
        m = re.search(r":=", decl["text"])
        raw = decl["text"][m.end():] if m else decl["text"]
        return strip_comments(raw)
    sig, _ = split_statement(decl["text"])
    return strip_comments(sig)  # theorems: scan the statement only


def is_documented_accessor(decl):
    """A docstring'd `def` whose body is a single `readMem`, optionally under
    a boundary mask (`if in-bounds then read else 0`) — a named element
    accessor (`ktElem`/`l2Load`-style, PRs #435/#443). Its docstring states
    the logical tensor indexing, so its one flat read *is* the layout
    documentation rather than a readability tax. Reductions (symbol or word
    form), `let`-bound addresses, value arithmetic wrappers (`Real.exp` etc.),
    or a second read disqualify (address arithmetic inside the single read
    and a trivial mask are fine)."""
    if decl["kind"] != "def" or not decl.get("doc"):
        return False
    body = body_of(decl)
    if len(re.findall(r"\breadMem\b", body)) != 1:
        return False
    return not re.search(
        r"[∑∏]|\blet\b|\bmatch\b|WithBot|Real\.|Tile\.|Finset\.|\.sum\b|\.prod\b",
        body)


MANI = os.path.join(BENCH, "proof_gap_manifest.tsv")


def load_manifest():
    """file (abs) -> list of declared public theorem names."""
    m = {}
    if not os.path.exists(MANI):
        return m
    import csv
    for r in csv.DictReader(open(MANI), delimiter="\t"):
        f = os.path.join(ROOT, r["file"])
        m.setdefault(os.path.abspath(f), []).append(r["declaration"])
    return m


_PY_INFIX = re.compile(r"_python(_test_shape|_test_case\d+|_case\d+|_block\d*)?")


def resolve_decl(d, names):
    """Map a (possibly stale, pinned) manifest decl to live theorem name(s)."""
    if d in names:
        return [d]
    stem = _PY_INFIX.sub("", d)
    for c in (stem, stem + "_general",
              stem.replace("_output_summary", "_output_summary_general")):
        if c in names:
            return [c]
    pre = d.split("_python")[0]
    suf = _PY_INFIX.sub("", "_python" + d.split("_python", 1)[1]) if "_python" in d else ""
    cand = [n for n in names if n.startswith(pre) and (not suf or n.endswith(suf))]
    return sorted(set(cand), key=len)[:1] if cand else []


# Headline discovery keys on the `specification` declaration keyword
# (VeriTile/Meta/Specification.lean) — a headline IS a `specification` decl;
# the name-suffix tier heuristics are retired. The suffix patterns below only
# feed the "also present" listing (candidate-looking theorems that are NOT
# declared as specifications).
HEADLINE_SUFFIX_RE = [
    lambda n: "summary" in n,
    lambda n: bool(re.search(r"_compute_correct(_general)?$", n)),
    lambda n: bool(re.search(r"_closed_form_correct(_general)?$", n)),
    lambda n: bool(re.search(r"_correct_general$", n)),
]


def is_headline(n):
    return any(t(n) for t in HEADLINE_SUFFIX_RE)


def find_summary_theorems(decls, manifest_decls):
    thms = {d["name"]: d for d in decls
            if d["kind"] in ("theorem", "specification", "lemma")}
    headline_names = [n for n, d in thms.items() if d["kind"] == "specification"]
    # fallback for files whose public theorem matches no tier: manifest seed
    if not headline_names:
        for d in manifest_decls:
            for live in resolve_decl(d, thms):
                if live not in headline_names:
                    headline_names.append(live)
    headline = [thms[n] for n in headline_names if n in thms]
    also = [thms[n] for n in thms
            if is_headline(n) and n not in headline_names]
    return headline, also


def closure(seed_text, defmap):
    """Transitive set of local def names referenced from seed_text."""
    seen, stack = set(), [t for t in IDENT_RE.findall(seed_text) if t in defmap]
    order = []
    while stack:
        n = stack.pop(0)
        if n in seen:
            continue
        seen.add(n); order.append(n)
        for t in IDENT_RE.findall(body_of(defmap[n])):
            if t in defmap and t not in seen:
                stack.append(t)
    return order


def selfref_flags(text):
    """Self-reference markers for a def BODY. The ONLY fakery signal is defining
    a value by executing the kernel and reading back the result state
    (`match exec … | some s' => s'.readMem …`). The mere mention of `exec`,
    `toAlgKernel`, or a `produced`-prefixed NAME is not self-reference — those
    appear in genuine lowering props and genuine closed forms named `produced*`
    (e.g. ChunkGated's `producedChunkGatedAttentionHStateValue = hClosed …`),
    so name-based heuristics produce false positives and are not used."""
    if re.search(r"\b(sF|sOut|sFin|sExec|sfinal|sfin|s')\.readMem", text) or \
       (re.search(r"match\s+exec\b", text) and ".readMem" in text):
        return ["readMem(executed-state)"]
    return []


def hypotheses(sig):
    """Pull binders that look like hypotheses (Prop-typed / h-named)."""
    hyps = []
    # `:(?!=)` so named arguments like `(kernel := …)` / `(expected := …)` are
    # not mistaken for `name : type` binders.
    for m in re.finditer(r"\(([^():]+):(?!=)\s*([^()]*(?:\([^()]*\)[^()]*)*)\)", sig):
        names, ty = m.group(1).strip(), m.group(2).strip()
        if any(s in ty for s in ("=", "≠", "<", "≤", "∣", "∈", "Prop", "→")) \
           or all(n.startswith("h") for n in names.split()):
            hyps.append(f"{names} : {ty}")
    return hyps


def py_source(file_path):
    d = os.path.dirname(file_path)
    pys = glob.glob(os.path.join(d, "*.py"))
    return [os.path.relpath(p, ROOT) for p in pys]


def make_sheet(file_path, manifest):
    text = open(file_path, encoding="utf-8").read()
    decls = split_decls(text)
    defmap = {d["name"]: d for d in decls if d["kind"] in ("def", "abbrev", "instance")}
    mdecls = manifest.get(os.path.abspath(file_path), [])
    headline, pinned = find_summary_theorems(decls, mdecls)
    rel = os.path.relpath(file_path, ROOT)
    out = [f"# Spec sheet — `{rel}`", ""]
    pys = py_source(file_path)
    if pys:
        out.append("**Python source:** " + ", ".join(f"`{p}`" for p in pys))
        out.append("")
    if not headline:
        out.append("> ⚠ no `*summary*` theorem found — public spec not located.")
        return "\n".join(out), {"file": rel, "headline": 0, "selfref": [], "defs": 0}

    all_selfref, all_defs = set(), set()
    for thm in headline:
        sig, _ = split_statement(thm["text"])
        sig_scan = strip_comments(sig)
        out.append(f"## Public theorem: `{thm['name']}`")
        out.append("")
        if thm["doc"]:
            out.append("<details><summary>docstring</summary>\n")
            out.append("```")
            out.append(thm["doc"]); out.append("```\n</details>\n")
        out.append("**Statement:**")
        out.append("```lean")
        out.append(sig.strip())
        out.append("```")
        hyps = hypotheses(sig)
        if hyps:
            out.append("\n**Assumptions / layout contracts:**")
            for h in hyps:
                out.append(f"- `{h}`")
        # closure of spec defs referenced from the statement
        order = closure(sig_scan, defmap)
        all_defs.update(order)
        # self-ref scan. NOTE: every correctness statement mentions
        # `exec(...)`/`sF.readMem` on its LHS — that is normal, not fakery. The
        # real signal is a pulled `expected` DEF whose body reads back the
        # executed state; scanned per-def below.
        sref = []
        out.append("")
        if order:
            out.append("**Closed-form spec defs (transitive):** "
                       + ", ".join(f"`{n}`" for n in order))
        else:
            out.append("> ⚠ statement references **no local spec def** — "
                       "spec may be inlined or stated against an opaque value.")
        out.append("")
        for n in order:
            d = defmap[n]
            df = selfref_flags(body_of(d))
            sref += [f"{n}:{f}" for f in df]
            tag = "  ⚠ SELF-REF" if df else ""
            out.append(f"<details><summary><code>{n}</code>{tag}</summary>\n")
            if d["doc"]:
                out.append("```")
                out.append(d["doc"]); out.append("```")
            out.append("```lean")
            out.append(d["text"].strip())
            out.append("```\n</details>\n")
        if sref:
            all_selfref.update(sref)
            out.append(f"> ⚠ **self-ref tokens:** {', '.join(sorted(set(sref)))}")
            out.append("")

    if pinned:
        out.append("## Also present (pinned special-case summaries)")
        for d in pinned:
            # Secondary public summaries are scanned for self-reference too —
            # a non-headline `_compute_correct`/`_summary` can still smuggle a
            # `produced*Value`/exec-readback `expected` (e.g. cross-launch
            # resume carriers). Flag them so the audit isn't headline-only.
            psig = strip_comments(split_statement(d["text"])[0])
            psref = []
            for n in closure(psig, defmap):
                psref += [f"{n}:{fl}" for fl in selfref_flags(body_of(defmap[n]))]
            tag = (" ⚠ SELF-REF " + ",".join(sorted(set(psref)))) if psref else ""
            if psref:
                all_selfref.update(psref)
            out.append(f"- `{d['name']}`{tag}")
        out.append("")
    # review-cost proxy: how hard is this spec to read?
    # A **documented element accessor** — a def with a docstring whose body is
    # a single readMem (its address arithmetic IS the documented layout
    # statement) — is a readability *aid*, not a tax: its read is exempt from
    # flat_reads and it scores 1 instead of 3 in the defs term. Otherwise
    # one-line accessor layers (PR #435) would *raise* the review-cost score.
    accessors = {n for n in all_defs if is_documented_accessor(defmap[n])}
    spec_text = "\n".join(defmap[n]["text"] for n in all_defs - accessors)
    flat_reads = len(re.findall(r"readMem\s+\w+\s*\([^()]*\*[^()]*\+", spec_text))
    stmt_lines = sum(len(split_statement(t["text"])[0].split("\n")) for t in headline)
    n_hyps = sum(len(hypotheses(split_statement(t["text"])[0])) for t in headline)
    score = (len(all_defs) - len(accessors)) * 3 + len(accessors) \
        + flat_reads + stmt_lines + n_hyps
    stat = {"file": rel, "headline": len(headline), "selfref": sorted(all_selfref),
            "defs": len(all_defs), "flat_reads": flat_reads,
            "stmt_lines": stmt_lines, "hyps": n_hyps, "score": score}
    return "\n".join(out), stat


def main():
    args = sys.argv[1:]
    outdir = None
    if "--out" in args:
        i = args.index("--out"); outdir = args[i + 1]; del args[i:i + 2]
    files = [a for a in args if a.endswith(".lean")]
    if not files:
        files = sorted(glob.glob(os.path.join(BENCH, "*", "*.lean")))
        if outdir is None:
            outdir = os.path.join(BENCH, "_spec-sheets")
    manifest = load_manifest()
    stats = []
    for f in files:
        sheet, stat = make_sheet(f, manifest)
        stats.append(stat)
        if outdir:
            os.makedirs(outdir, exist_ok=True)
            name = os.path.splitext(os.path.basename(f))[0] + ".md"
            open(os.path.join(outdir, name), "w", encoding="utf-8").write(sheet)
        else:
            print(sheet); print("\n" + "=" * 80 + "\n")
    if outdir:
        # index + summary of audit flags
        idx = ["# Spec-sheet index", ""]
        risky = [s for s in stats if s["selfref"]]
        nohit = [s for s in stats if s["headline"] == 0]
        idx.append(f"- {len(stats)} kernels, "
                   f"{sum(s['headline'] for s in stats)} headline theorems, "
                   f"{len(risky)} with self-ref tokens, "
                   f"{len(nohit)} with no summary theorem found.")
        idx.append("- Ranked by **review-cost** proxy "
                   "`score = 3·defs + flat_offset_reads + stmt_lines + hyps` "
                   "(hardest specs to audit first). Documented single-read "
                   "element accessors (docstring + body = one `readMem`, "
                   "optionally boundary-masked) are exempt from "
                   "`flat_offset_reads` and count 1 (not 3) in the `defs` "
                   "term — they are readability aids, not tax.\n")
        idx.append("| score | kernel | defs | flat-reads | stmt-lines | hyps | flags |")
        idx.append("|---:|---|---:|---:|---:|---:|---|")
        for s in sorted(stats, key=lambda x: (-x["score"], x["file"])):
            base = os.path.splitext(os.path.basename(s["file"]))[0] + ".md"
            flag = ("⚠" + ",".join(s["selfref"])) if s["selfref"] else ""
            if s["headline"] == 0:
                flag = (flag + " ❓no-summary").strip()
            idx.append(f"| {s['score']} | [{base}]({base}) | {s['defs']} "
                       f"| {s['flat_reads']} | {s['stmt_lines']} | {s['hyps']} | {flag} |")
        open(os.path.join(outdir, "INDEX.md"), "w", encoding="utf-8").write("\n".join(idx))
        print(f"wrote {len(files)} sheets to {outdir}")
        print(f"  self-ref-flagged: {len(risky)} | no-summary: {len(nohit)}")


if __name__ == "__main__":
    main()
