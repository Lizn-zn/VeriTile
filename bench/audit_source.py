"""Shared source extraction for the TritonBench-G structural audit.

These helpers preserve the audit's source-text conventions. They do not discover
proof obligations or certify translations; Lean inventories and the official
comparator own proof validation. Rule-specific kernel and call selection is
explicit so changes cannot silently alter the scope of another audit rule.
"""
import ast
import re


def strip_lean_comments(text: str) -> str:
    """Blank out `--` line comments and (nested) `/- ... -/` block comments,
    preserving line structure so reported line numbers stay accurate."""
    out = []
    i = 0
    n = len(text)
    depth = 0
    while i < n:
        two = text[i:i + 2]
        if depth == 0 and two == "--":
            j = text.find("\n", i)
            i = n if j < 0 else j
            continue
        if two == "/-":
            depth += 1
            out.append("  ")
            i += 2
            continue
        if depth > 0 and two == "-/":
            depth -= 1
            out.append("  ")
            i += 2
            continue
        if depth == 0:
            out.append(text[i])
        else:
            out.append(text[i] if text[i] == "\n" else " ")
        i += 1
    return "".join(out)


def is_triton_jit(dec: ast.expr) -> bool:
    if isinstance(dec, ast.Call):
        dec = dec.func
    return (
        isinstance(dec, ast.Attribute) and dec.attr == "jit"
    ) or (
        isinstance(dec, ast.Name) and dec.id == "jit"
    )


def python_jit_names(text: str) -> set[str]:
    tree = ast.parse(text)
    return {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and any(is_triton_jit(dec) for dec in node.decorator_list)
    }


def python_jit_names_text(text: str) -> set[str]:
    """Read JIT names with the control-flow audit's textual body-selection policy."""
    names, pending = set(), False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("@triton.jit"):
            pending = True
            continue
        if pending and stripped.startswith("def "):
            names.add(stripped[4:].split("(", 1)[0].strip())
            pending = False
            continue
        if pending and stripped and not stripped.startswith("@"):
            pending = False
    return names


def lean_first_preamble(text: str) -> str:
    idx = text.find("triton {")
    return text[:idx] if idx >= 0 else text


def target_kernel_candidates(preamble: str):
    """Every ``<file>.py`'s `<name>`'' phrase in the preamble, in order."""
    return re.findall(r"\.py`'s[^`]*`([^`]+)`", preamble, re.S)


def target_kernel_name(preamble: str, jit_names):
    """The declared target JIT: the LAST preamble phrase that names a real
    `@triton.jit` kernel. Prose legitimately reuses the same phrasing for
    non-kernel snippets (a transcription note quoting `labels_ptr += row_idx`,
    say); such a phrase must not shadow the declaration, which is what taking
    the last phrase unconditionally used to do."""
    named = [name for name in target_kernel_candidates(preamble) if name in jit_names]
    return named[-1] if named else None


def python_first_kernel_body(text: str, *, include_block_pointers: bool = False) -> str:
    lines = text.splitlines()

    def body_for_name(target: str) -> str:
        i = 0
        while i < len(lines):
            if not lines[i].strip().startswith("@triton.jit"):
                i += 1
                continue
            i += 1
            while i < len(lines) and lines[i].strip().startswith("@"):
                i += 1
            if i >= len(lines) or not lines[i].strip().startswith("def " + target + "("):
                i += 1
                continue

            start = None
            parens = 0
            for j in range(i, len(lines)):
                parens += lines[j].count("(") - lines[j].count(")")
                if parens <= 0 and lines[j].rstrip().endswith(":"):
                    start = j + 1
                    break
            if start is None:
                return ""

            body = []
            k = start
            while k < len(lines):
                line = lines[k]
                if line and not line.startswith((" ", "\t")):
                    break
                body.append(line)
                k += 1
            body_text = "\n".join(body)
            if any(token in body_text for token in ("tl.program_id", "tl.load", "tl.store")):
                return body_text
            i = k
        return ""

    # Preserve the primary-kernel preference used by the surface checks.
    # Block-pointer-only bodies participate in the fallback scan only.
    fallback_tokens = ("tl.program_id", "tl.load", "tl.store")
    if include_block_pointers:
        fallback_tokens += ("tl.make_block_ptr",)

    preferred = body_for_name("_attn_fwd")
    if preferred:
        return preferred
    i = 0
    while i < len(lines):
        if not lines[i].strip().startswith("@triton.jit"):
            i += 1
            continue
        i += 1
        while i < len(lines) and lines[i].strip().startswith("@"):
            i += 1
        if i >= len(lines) or not lines[i].strip().startswith("def "):
            continue
        start = None
        parens = 0
        for j in range(i, len(lines)):
            parens += lines[j].count("(") - lines[j].count(")")
            if parens <= 0 and lines[j].rstrip().endswith(":"):
                start = j + 1
                break
        if start is None:
            return ""
        body = []
        k = start
        while k < len(lines):
            line = lines[k]
            if line and not line.startswith((" ", "\t")):
                break
            body.append(line)
            k += 1
        body_text = "\n".join(body)
        if any(token in body_text for token in fallback_tokens):
            return body_text
        i = k
    return ""


def python_jit_kernel_bodies(text: str, *, all_jit: bool = False) -> list[str]:
    """Select memory kernels, preferring `_attn_fwd`, or every JIT body.

    Helper-coverage checks need all bodies; sequence checks use the primary
    kernel policy. These choices intentionally remain distinct.
    """
    lines = text.splitlines()
    bodies = []
    i = 0
    while i < len(lines):
        if not lines[i].strip().startswith("@triton.jit"):
            i += 1
            continue
        i += 1
        while i < len(lines) and lines[i].strip().startswith("@"):
            i += 1
        if i >= len(lines) or not lines[i].strip().startswith("def "):
            continue

        def_i = i
        start = None
        parens = 0
        for j in range(def_i, len(lines)):
            line = lines[j]
            parens += line.count("(") - line.count(")")
            if parens <= 0 and line.rstrip().endswith(":"):
                start = j + 1
                break
        if start is None:
            break

        body = []
        k = start
        while k < len(lines):
            line = lines[k]
            if line and not line.startswith((" ", "\t")):
                break
            body.append(line)
            k += 1
        body_text = "\n".join(body)
        if all_jit or any(token in body_text for token in ("tl.program_id", "tl.load", "tl.store")):
            name = lines[def_i].strip().split("def ", 1)[1].split("(", 1)[0]
            if not all_jit and name == "_attn_fwd":
                return [body_text]
            bodies.append(body_text)
        i = k
    return bodies


def lean_first_triton_body(text: str) -> str:
    idx = text.find("triton {")
    if idx < 0:
        return ""
    start = text.find("{", idx)
    depth = 0
    out = []
    for ch in text[start:]:
        if ch == "{":
            depth += 1
            if depth == 1:
                continue
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        if depth >= 1:
            out.append(ch)
    return "".join(out)


def lean_triton_bodies(text: str) -> list[str]:
    bodies = []
    pos = 0
    while True:
        idx = text.find("triton {", pos)
        if idx < 0:
            break
        start = text.find("{", idx)
        depth = 0
        out = []
        end = start
        for off, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
                if depth == 1:
                    continue
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = off + 1
                    break
            if depth >= 1:
                out.append(ch)
        bodies.append("".join(out))
        pos = end
    return bodies


def strip_comments(text: str) -> str:
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())


def tl_calls(text: str, fn: str, *, arguments_only: bool = False,
             normalize_whitespace: bool = False) -> list[str]:
    text = strip_comments(text)
    pattern = re.compile(rf"tl\s*\.\s*{fn}\s*\(")
    calls = []
    pos = 0
    while True:
        match = pattern.search(text, pos)
        if not match:
            break
        k = match.end()
        depth = 1
        while k < len(text) and depth > 0:
            if text[k] == "(":
                depth += 1
            elif text[k] == ")":
                depth -= 1
            k += 1
        call = text[match.end():k - 1] if arguments_only else text[match.start():k]
        if normalize_whitespace:
            call = re.sub(r"\s+", " ", call)
        calls.append(call)
        pos = k
    return calls


def split_top_level_args(args_text: str, fn: str | None = None) -> list[str]:
    """Split an argument list, or a complete `tl.<fn>(...)` call.

    Nested parentheses, brackets, and braces do not delimit arguments.
    """
    if fn is not None:
        match = re.search(rf"tl\s*\.\s*{fn}\s*\(", args_text)
        if match is None:
            raise ValueError(f"Expected a tl.{fn} call")
        args_text = args_text[match.end():-1]
    args = []
    cur = []
    depth = 0
    for ch in args_text:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            arg = "".join(cur).strip()
            if arg:
                args.append(arg)
            cur = []
            continue
        cur.append(ch)
    arg = "".join(cur).strip()
    if arg:
        args.append(arg)
    return args
