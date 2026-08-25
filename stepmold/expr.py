"""The expression language `assert` nodes and node-level `assert:` are written in.

This exists because a pack is data: an assert has to travel inside `graph.yaml` as a
string, and a string that gets `eval()`-ed is a remote code execution hole in something
designed to run untrusted, compiler-generated packs. So expressions are parsed with
`ast.parse` and walked against a whitelist — anything not on the list is refused by name.

The language is deliberately boring: names from state, dotted lookup into mappings,
comparisons, `and`/`or`/`not`, `in`, arithmetic, indexing, literals, and a fixed set of
helper functions. No attribute access on non-mappings, no method calls, no lambdas, no
comprehensions, no assignment. If an assert needs more than this, it wants to be a
deterministic node in the graph, not a one-liner in YAML.

The other half of the contract is that *every* way an expression can fail arrives as
`ExprError`. `verify._check_assert` catches `ExprError` and turns it into a `Rejected`,
which the retry ladder and `on_fail` edges know how to route; anything else — a raw
`TypeError` from a bad operand, a `RecursionError` from a deeply nested expression, a
`MemoryError` from a huge repetition — escapes that handler and kills the whole run. So
the walker converts what it can and refuses, up front, what it cannot survive: nesting
past `_MAX_DEPTH` and sequence repetition past `_MAX_REPEAT`.

There is a third rule, and it is about *what those errors say*. An assert exists to read
the candidate object, so every failure here happens with model-authored values in hand —
`_check_assert` puts `str(exc)` straight into a `Rejected`, whose feedback is shown to
the model on the next rung. So `str(exc)` is built only from text the pack wrote (the
expression, a helper name, a type name); the offending values live on `exc.detail`, for
logs and for a caller that asks. See `_error`.
"""

import ast
import operator

from .errors import ExprError

__all__ = ["evaluate", "is_true"]

_LITERALS = {"true": True, "false": False, "null": None, "none": None}

# How much of a value a *diagnostic* may show. A ticket is a megabyte often enough that
# an unclipped repr is a real hazard, and nothing here needs more than a glimpse.
_SHOW_LIMIT = 120

# How deep a single expression may nest. `_eval` recurses once per AST level, so without
# a ceiling `1+1+1+...` walks straight into CPython's own recursion limit and raises a
# RecursionError the caller is not expecting. The ceiling is far above anything a
# readable one-line assert reaches, and well under the interpreter's own.
_MAX_DEPTH = 100

# How long a sequence `*` may produce. `"a" * n * n` is the one operator here that can
# amplify a small expression into gigabytes, and MemoryError is not an ExprError. The
# limit is a refusal threshold, not a measurement: anything a sane assert compares
# against is orders of magnitude below it.
_MAX_REPEAT = 1_000_000

_HELPERS = {
    "len": len,
    "abs": abs,
    "min": min,
    "max": max,
    "sum": sum,
    "round": round,
    "sorted": sorted,
    "any": any,
    "all": all,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "lower": lambda value: str(value).lower(),
    "upper": lambda value: str(value).upper(),
    "strip": lambda value: str(value).strip(),
    "startswith": lambda value, prefix: str(value).startswith(prefix),
    "endswith": lambda value, suffix: str(value).endswith(suffix),
    "contains": lambda haystack, needle: needle in haystack,
}

_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}

_COMPARE = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
}


def _error(safe, detail=None):
    """An `ExprError` whose message is safe to put in front of the model.

    `verify._check_assert` turns `str(exc)` into the `Rejected` a retry prompt is built
    from, and an assert reads exactly the values that were just rejected — so a message
    quoting one would carry the rejected generation into the next prompt, which is the
    self-conditioning spiral stepmold is designed around (docs/ARCHITECTURE.md §3). `str(exc)` is
    therefore pack-authored text only; `exc.detail` keeps the whole story for the
    operator, because bytes in a log cannot condition anything.
    """
    exc = ExprError(safe)
    exc.detail = safe if detail is None else detail
    return exc


def _show(value):
    """A value, rendered for `detail` only — bounded, because the value is not ours."""
    return _bound(repr(value))


def _why(exc):
    """A builtin's own message, bounded: several of them quote the whole operand."""
    return _bound(str(exc))


def _bound(text):
    text = " ".join(text.split())
    return text if len(text) <= _SHOW_LIMIT else text[:_SHOW_LIMIT] + "..."


def evaluate(expression, state):
    """Evaluate `expression` against `state`. Raises `ExprError` on anything unsafe."""
    if not isinstance(expression, str) or not expression.strip():
        raise _error("expression is empty")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise _error("could not parse expression %r (%s)" % (expression, exc.msg))
    except RecursionError:
        # Which inputs exhaust the parser rather than the walker is a CPython detail that
        # moves between versions, so cover it here too rather than assume a version.
        raise _error("expression %r is nested too deeply to parse" % expression)
    return _eval(tree.body, state, expression, 0)


def is_true(expression, state):
    """Evaluate `expression` and coerce the result to a bool."""
    return bool(evaluate(expression, state))


def _eval(node, state, source, depth):
    """Walk one AST node. `depth` is how many levels of expression are already open."""
    if depth > _MAX_DEPTH:
        raise _error(
            "expression %r is nested more than %d levels deep, which stepmold refuses to "
            "evaluate" % (source, _MAX_DEPTH)
        )
    inner = depth + 1
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return _name(node.id, state)
    if isinstance(node, ast.Attribute):
        return _attribute(node, state, source)
    if isinstance(node, ast.Subscript):
        return _subscript(node, state, source, inner)
    if isinstance(node, ast.BoolOp):
        return _bool_op(node, state, source, inner)
    if isinstance(node, ast.UnaryOp):
        return _unary(node, state, source, inner)
    if isinstance(node, ast.BinOp):
        return _binary(node, state, source, inner)
    if isinstance(node, ast.Compare):
        return _compare(node, state, source, inner)
    if isinstance(node, ast.Call):
        return _call(node, state, source, inner)
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return [_eval(item, state, source, inner) for item in node.elts]
    if isinstance(node, ast.Dict):
        return _dict(node, state, source, inner)
    if isinstance(node, ast.IfExp):
        if _eval(node.test, state, source, inner):
            return _eval(node.body, state, source, inner)
        return _eval(node.orelse, state, source, inner)
    raise _error(_refuse(node, source))


def _refuse(node, source):
    name = type(node).__name__
    if name in ("ListComp", "SetComp", "DictComp", "GeneratorExp"):
        name = "Comprehension"
    return "%s is not allowed in a stepmold expression (in %r)" % (name, source)


def _name(identifier, state):
    if identifier in _LITERALS:
        return _LITERALS[identifier]
    if identifier.startswith("__"):
        raise _error("name %r is not allowed in a stepmold expression" % identifier)
    if identifier in state:
        return state[identifier]
    if identifier in _HELPERS:
        return _HELPERS[identifier]
    # The key list is diagnostics only: in merge mode the candidate's own property names
    # *are* state keys, so listing them to the model would echo the rejected output back.
    raise _error(
        "expression references %r, which is not in state" % identifier,
        detail="expression references %r, which is not in state (state has: %s)"
        % (identifier, _bound(", ".join(sorted(state))) or "nothing"),
    )


def _path(node):
    """Flatten `a.b.c` into "a.b.c", or return None if it is not a plain path."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def _attribute(node, state, source):
    path = _path(node)
    if path is None:
        raise _error(_refuse(node, source))
    if path.startswith("__") or ".__" in path:
        raise _error("name %r is not allowed in a stepmold expression" % path)
    parts = path.split(".")
    current = _name(parts[0], state)
    for part in parts[1:]:
        if not isinstance(current, dict) or part not in current:
            raise _error(
                "expression references %r, which is not a mapping key in state" % path
            )
        current = current[part]
    return current


def _subscript(node, state, source, depth):
    container = _eval(node.value, state, source, depth)
    key = _eval(node.slice, state, source, depth)
    try:
        return container[key]
    except (KeyError, IndexError, TypeError) as exc:
        # The key is routinely a value out of the candidate — `queues[category]` is the
        # shape every routing assert takes — so it belongs in `detail`, never in `str`.
        raise _error(
            "cannot index by the value %r asks for" % source,
            detail="cannot index %s in %r (%s)" % (_show(key), source, _why(exc)),
        )


def _dict(node, state, source, depth):
    result = {}
    for key_node, value_node in zip(node.keys, node.values):
        if key_node is None:
            # `{**other}`. ast puts a None key there; refuse it by name rather than let
            # the walker report the placeholder's type.
            raise _error(
                "** unpacking is not allowed in a stepmold expression (in %r)" % source
            )
        key = _eval(key_node, state, source, depth)
        value = _eval(value_node, state, source, depth)
        try:
            result[key] = value
        except TypeError as exc:
            raise _error(
                "cannot use that value as a dict key in %r" % source,
                detail="cannot use %s as a dict key in %r (%s)"
                % (_show(key), source, _why(exc)),
            )
    return result


def _bool_op(node, state, source, depth):
    """`and` / `or`, short-circuiting exactly like Python.

    Evaluating every operand up front would break the one idiom pack authors reach for
    most — `x is not null and len(x) > 2` — by running the guarded half against the
    value the guard exists to exclude. So stop at the first operand that decides the
    result and never look at the rest.
    """
    wants_truthy = isinstance(node.op, ast.And)
    result = None
    for value_node in node.values:
        result = _eval(value_node, state, source, depth)
        if bool(result) is not wants_truthy:
            return result
    # Every operand agreed; Python yields the last one, whatever its type.
    return result


def _unary(node, state, source, depth):
    value = _eval(node.operand, state, source, depth)
    if isinstance(node.op, ast.Not):
        return not value
    if not isinstance(node.op, (ast.USub, ast.UAdd)):
        raise _error(_refuse(node.op, source))
    try:
        return -value if isinstance(node.op, ast.USub) else +value
    except TypeError as exc:
        # `-"text"`: a mis-written assert, and the caller only handles ExprError.
        raise _error(
            "cannot evaluate %r against this value" % source,
            detail="cannot evaluate %r (%s)" % (source, _why(exc)),
        )


def _binary(node, state, source, depth):
    handler = _BINARY.get(type(node.op))
    if handler is None:
        raise _error(_refuse(node.op, source))
    left = _eval(node.left, state, source, depth)
    right = _eval(node.right, state, source, depth)
    if isinstance(node.op, ast.Mult):
        _check_repeat(left, right, source)
    try:
        return handler(left, right)
    except (TypeError, ZeroDivisionError) as exc:
        raise _error(
            "cannot evaluate %r against these values" % source,
            detail="cannot evaluate %r (%s)" % (source, _why(exc)),
        )


def _check_repeat(left, right, source):
    """Refuse `sequence * n` before it allocates, if the result would be huge.

    Sequence repetition is the only whitelisted operator that turns a short expression
    into an arbitrarily large object, and the MemoryError it would raise is not an
    ExprError, so it would escape the caller's handler instead of failing the node.
    """
    for sequence, count in ((left, right), (right, left)):
        if not isinstance(sequence, (str, bytes, list, tuple)):
            continue
        if not isinstance(count, int):
            continue
        if len(sequence) * count > _MAX_REPEAT:
            raise _error(
                "repetition in %r would build a sequence that is too large (limit %d)"
                % (source, _MAX_REPEAT),
                detail="repetition in %r would build a sequence of %d items, which is "
                "too large (limit %d)"
                % (source, len(sequence) * count, _MAX_REPEAT),
            )


def _compare(node, state, source, depth):
    left = _eval(node.left, state, source, depth)
    for op, comparator in zip(node.ops, node.comparators):
        handler = _COMPARE.get(type(op))
        if handler is None:
            raise _error(_refuse(op, source))
        right = _eval(comparator, state, source, depth)
        try:
            if not handler(left, right):
                return False
        except TypeError as exc:
            raise _error(
                "cannot compare these values in %r" % source,
                detail="cannot compare in %r (%s)" % (source, _why(exc)),
            )
        left = right
    return True


def _call(node, state, source, depth):
    if not isinstance(node.func, ast.Name) or node.func.id not in _HELPERS:
        called = getattr(node.func, "id", None) or _path(node.func) or "<expression>"
        raise _error(
            "%r is not a stepmold expression helper (allowed: %s)"
            % (called, ", ".join(sorted(_HELPERS)))
        )
    if node.keywords:
        raise _error("stepmold expression helpers take positional arguments only")
    arguments = [_eval(argument, state, source, depth) for argument in node.args]
    try:
        return _HELPERS[node.func.id](*arguments)
    except Exception as exc:
        # `int("<a ticket>")` puts the whole argument in the ValueError it raises, so the
        # builtin's own message is exactly the thing that must not reach a prompt.
        raise _error(
            "helper %s() failed in %r" % (node.func.id, source),
            detail="helper %s() failed in %r (%s)"
            % (node.func.id, source, _why(exc)),
        )
