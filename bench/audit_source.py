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


def python_block_load_padding(text: str) -> list[dict]:
    """Inventory checked block-pointer loads in JIT bodies without executing
    Python. Omitted/empty padding is unspecified in Triton, not zero. This is
    a source inventory, not a proof that an out-of-bounds lane is reached.
    Dynamic boundary checks remain visible; literal empty checks are harmless.
    """
    tree = ast.parse(text)
    loads = []
    aliases = {"tl"}
    for node in tree.body:
        if isinstance(node, ast.Import):
            aliases.update(a.asname or a.name for a in node.names
                           if a.name == "triton.language")
    for kernel in tree.body:
        if not isinstance(kernel, ast.FunctionDef) or not any(
            is_triton_jit(dec) for dec in kernel.decorator_list
        ):
            continue
        for node in ast.walk(kernel):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "load" or ast.unparse(node.func.value) not in aliases:
                continue
            kwargs = {kw.arg: kw.value for kw in node.keywords}
            boundary = kwargs.get("boundary_check")
            if boundary is None:
                continue
            if isinstance(boundary, (ast.Tuple, ast.List)) and not boundary.elts:
                continue
            padding = kwargs.get("padding_option")
            if padding is None or (isinstance(padding, ast.Constant) and padding.value == ""):
                policy = "unspecified"
            elif isinstance(padding, ast.Constant) and padding.value in ("zero", "nan"):
                policy = padding.value
            else:
                policy = "dynamic_or_invalid"
            loads.append({"kernel": kernel.name, "line": node.lineno,
                          "boundary_check": ast.unparse(boundary), "padding": policy})
    return sorted(loads, key=lambda load: load["line"])


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
    legacy = re.findall(r"\.py`'s[^`]*`([^`]+)`", preamble, re.S)
    if legacy:
        return legacy
    named = re.findall(r"Source:\s*\w+\.py,\s*(\w+)", preamble)
    named += re.findall(r"Complete typed source for `([^`]+)` in\s*`[^`]+\.py`", preamble)
    return named


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


def _quoted_statement(text: str, start: int) -> tuple[str, int]:
    marker = '`(tritonStmt|'
    if not text.startswith(marker, start):
        raise ValueError('Expected a quoted Triton statement')
    depth = 1
    for end in range(start + len(marker), len(text)):
        if text[end] == '(':
            depth += 1
        elif text[end] == ')':
            depth -= 1
            if depth == 0:
                return text[start + len(marker):end].strip(), end + 1
    raise ValueError('Unterminated quoted Triton statement')


def original_statement_splices(body: str, preamble: str) -> str:
    """Expand a literal two-arm statement macro at version=original.

    This deliberately supports only the quoted-statement form used by source
    variants. Unrecognized metaprograms remain visible to the textual checks.
    The Lean AST regressions and proof replay still check actual expansion.
    """
    code = strip_lean_comments(preamble)
    pattern = re.compile(
        r'let\s+(\w+)\s*:\s*TSyntax\s+`tritonStmt\s*←\s*'
        r'if\s+(\w+)\.getId\s*==\s*`(\w+)\s+then\s*')
    for match in pattern.finditer(code):
        yes, end = _quoted_statement(code, match.end())
        other = re.match(r'\s*else\s*', code[end:])
        if other is None:
            raise ValueError('Statement macro needs an explicit else arm')
        no, _ = _quoted_statement(code, end + other.end())
        selected = yes if match[3] == 'original' else no
        body = re.sub(r'\$' + re.escape(match[1]) + r'\b', lambda _: "\n" + selected + "\n", body)
    return body


def strip_region_declarations(body: str) -> str:
    """Omit type declarations, including comma-continued lines, from statements."""
    out, continued = [], False
    for line in body.splitlines():
        if continued or line.strip().startswith('tl.region '):
            continued = line.rstrip().endswith(',')
            out.append('')
        else:
            out.append(line)
    return '\n'.join(out)


def quoted_kernel_signature(text: str) -> str:
    """The lambda that constructs the first macro-generated source kernel."""
    preamble = lean_first_preamble(strip_lean_comments(text))
    starts = list(re.finditer(r'`\(fun\s+', preamble))
    if not starts:
        return ''
    signature = preamble[starts[-1].end():]
    return signature.split('=>', 1)[0] if 'ComputeKernel.mk' in signature else ''


def declared_region_dtypes(lean_body: str) -> dict[str, set[str]]:
    regions = {}
    in_region = False
    for line in strip_lean_comments(lean_body).splitlines():
        line = line.strip()
        if line.startswith('tl.region '):
            in_region = True
            line = line[len('tl.region '):]
        elif not in_region:
            continue
        elif not re.match(r'\w+\s*=\s*(?:tl\.\w+|[A-Z_]+|\$\w+)\s*(?:,|$)', line):
            in_region = False
            continue
        for name, dtype in re.findall(r'(\w+)\s*=\s*(tl\.\w+|[A-Z_]+|\$\w+)', line):
            regions.setdefault(name, set()).add(dtype)
    return regions


def resolve_source_dtype_attributes(python_body: str, lean_body: str) -> str:
    """Resolve Python pointer/tensor dtype attributes from explicit region types.

    Only unambiguous pointer aliases, block pointers and direct loads participate. Unknown
    attributes and literal dtypes stay unchanged, so mismatches remain errors.
    This normalizes names; it does not validate dtype inference or conversions.
    """
    regions = declared_region_dtypes(lean_body)
    types = dict(regions)
    bindings = {}
    for match in re.finditer(r'^\s*(\w+)\s*([+*/-]?=)(?!=)([^\n]*)',
                             strip_region_declarations(strip_lean_comments(lean_body)), re.M):
        name, op, rhs = match.groups()
        # Pointer addition/subtraction preserves its element type. Other
        # reassignments and updates still make the alias ambiguous.
        if op in ('+=', '-='):
            continue
        alias = re.fullmatch(r'\s*(\w+)\s*\+[^\n]+', rhs) if op == '=' else None
        bindings.setdefault(name, []).append(alias[1] if alias else None)
    block_pointers = {}
    for pointer, region in re.findall(
            r'(\w+)\s*=\s*tl\.make_block_ptr\s*\(\s*base\s*=\s*(\w+)\b', lean_body):
        block_pointers.setdefault(pointer, set()).update(regions.get(region, set()))

    def pointer_types(pointer, seen=frozenset()):
        if pointer in seen:
            return set()
        entries = bindings.get(pointer)
        if entries is None:
            return regions.get(pointer, set())
        if len(entries) != 1:
            return set()
        if entries[0] is not None:
            return pointer_types(entries[0], seen | {pointer})
        return block_pointers.get(pointer, set())

    for pointer in bindings:
        values = pointer_types(pointer)
        if values:
            types[pointer] = values
    for match in re.finditer(r'(\w+)\s*=\s*tl\.load\s*\(\s*(\w+)\b', lean_body):
        tensor, pointer = match.groups()
        tail = lean_body[match.end():].split('\n', 1)[0]
        if '.to(' in tail:
            # A converted load is not a direct storage-typed tensor binding.
            types.pop(tensor, None)
            continue
        types.setdefault(tensor, set()).update(types.get(pointer, set()))
    def resolve(match):
        values = types.get(match[1], set())
        return next(iter(values)) if len(values) == 1 else match[0]
    return re.sub(r'\b(\w+)\.dtype(?:\.element_ty)?\b', resolve, python_body)


def cast_sequence(body: str) -> list[str]:
    """Keep literal and macro-selected cast dtypes visible to source checks."""
    casts = re.findall(r'\.to\(\s*(?:tl\.)?(\$?[A-Za-z0-9_.]+)'
                       r'(?:\s*:\s*tritonDType)?\s*\)', strip_comments(body))
    return [cast.replace('_', '').lower() for cast in casts]


def store_calls_with_explicit_destination_casts(python_body: str, lean_body: str) -> list[str]:
    """Match an explicit store cast only to the declared destination dtype.

    Triton converts a stored value to its pointer element type. A typed port
    may spell that conversion explicitly. Resolve only unique, simple pointer
    aliases in the Python body; unknown/reassigned aliases and casts to other
    precisions remain mismatches. All value expressions and masks stay visible.
    This is source normalization, not a proof of numerical cast identities.
    """
    regions = declared_region_dtypes(lean_body)
    bindings: dict[str, list[str | None]] = {}
    for match in re.finditer(r'^\s*(\w+)\s*([+*/-]?=)(?!=)([^\n]*)',
                             strip_comments(python_body), re.M):
        name, op, rhs = match.groups()
        # Pointer addition/subtraction preserves its element type. Other
        # reassignments and updates still make the alias ambiguous.
        if op in ('+=', '-='):
            continue
        alias = re.fullmatch(r'\s*(\w+)\s*\+[^\n]+', rhs) if op == '=' else None
        bindings.setdefault(name, []).append(alias[1] if alias else None)

    def dtype(pointer):
        # Pointer plus/minus an offset retains the known base element type.
        # Other operators and unknown roots remain unresolved.
        try:
            node = ast.parse(pointer, mode="eval").body
        except SyntaxError:
            return None
        while isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            node = node.left
        if not isinstance(node, ast.Name):
            return None
        pointer = node.id
        seen = set()
        while pointer not in seen:
            seen.add(pointer)
            if pointer in bindings:
                aliases = bindings[pointer]
                if len(aliases) != 1 or aliases[0] is None:
                    return None
                if aliases[0] == pointer:
                    # A single self-advance preserves the declared pointer type.
                    values = regions.get(pointer, set())
                    return next(iter(values)) if len(values) == 1 else None
                pointer = aliases[0]
            else:
                values = regions.get(pointer, set())
                return next(iter(values)) if len(values) == 1 else None
        return None

    py_calls, lean_calls = tl_calls(python_body, 'store'), tl_calls(lean_body, 'store')
    if len(py_calls) != len(lean_calls):
        return py_calls
    result = []
    def compact(value):
        value = re.sub(r'\s+', '', value)
        value = re.sub(r'\$\((\w+)\)', r'\1', value)
        while value.startswith('(') and value.endswith(')'):
            depth = 0
            encloses = True
            for i, char in enumerate(value):
                depth += (char == '(') - (char == ')')
                if depth == 0 and i != len(value)-1:
                    encloses = False
                    break
            if not encloses or depth != 0:
                break
            value = value[1:-1]
        return value
    for py_call, lean_call in zip(py_calls, lean_calls):
        py_args = split_top_level_args(py_call, 'store')
        lean_args = split_top_level_args(lean_call, 'store')
        if len(py_args) < 2 or len(lean_args) < 2 or compact(py_args[0]) != compact(lean_args[0]):
            result.append(py_call)
            continue
        target = dtype(py_args[0].strip())
        cast = re.fullmatch(r'(.+)\.to\(\s*(tl\.\w+|\$\w+|[A-Z_]+)(?:\s*:\s*tritonDType)?\s*\)',
                            lean_args[1].strip(), re.S)
        if target is not None and cast and cast[2] == target and compact(cast[1]) == compact(py_args[1]):
            value = py_args[1].strip()
            if not re.fullmatch(r'\w+', value):
                value = '(' + value + ')'
            py_args[1] = value + '.to(' + target + ')'
            result.append('tl.store(' + ', '.join(py_args) + ')')
        else:
            result.append(py_call)
    return result


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
    return original_statement_splices("".join(out), text[:idx])


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
        bodies.append(original_statement_splices("".join(out), text[:idx]))
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


def bare_true_proof(line: str) -> bool:
    """Detect the old bare-True placeholder spelling, not a proved P = True.

    This is a lexical hint; elaboration/trust replay determines proof validity.
    """
    match = re.search(r"True\s*:=\s*by\b", line)
    return bool(match and not re.search(r"(?:=|↔)\s*$", line[:match.start()]))


def python_named_jit_body(text: str, name: str) -> str:
    """Body of the explicitly documented JIT, excluding other JITs and host code."""
    lines = text.splitlines()
    for node in ast.parse(text).body:
        if (isinstance(node, ast.FunctionDef) and node.name == name
                and any(is_triton_jit(d) for d in node.decorator_list)):
            return "\n".join(lines[node.body[0].lineno - 1:node.end_lineno])
    raise ValueError(f"Missing documented JIT: {name}")
