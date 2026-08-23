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
"""

import ast
import operator

from .errors import ExprError

__all__ = ["evaluate", "is_true"]

_LITERALS = {"true": True, "false": False, "null": None, "none": None}

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


def evaluate(expression, state):
    """Evaluate `expression` against `state`. Raises `ExprError` on anything unsafe."""
    if not isinstance(expression, str) or not expression.strip():
        raise ExprError("expression is empty")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise ExprError("could not parse expression %r (%s)" % (expression, exc.msg))
    return _eval(tree.body, state, expression)


def is_true(expression, state):
    """Evaluate `expression` and coerce the result to a bool."""
    return bool(evaluate(expression, state))


def _eval(node, state, source):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return _name(node.id, state)
    if isinstance(node, ast.Attribute):
        return _attribute(node, state, source)
    if isinstance(node, ast.Subscript):
        return _subscript(node, state, source)
    if isinstance(node, ast.BoolOp):
        return _bool_op(node, state, source)
    if isinstance(node, ast.UnaryOp):
        return _unary(node, state, source)
    if isinstance(node, ast.BinOp):
        return _binary(node, state, source)
    if isinstance(node, ast.Compare):
        return _compare(node, state, source)
    if isinstance(node, ast.Call):
        return _call(node, state, source)
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return [_eval(item, state, source) for item in node.elts]
    if isinstance(node, ast.Dict):
        return {
            _eval(key, state, source): _eval(value, state, source)
            for key, value in zip(node.keys, node.values)
        }
    if isinstance(node, ast.IfExp):
        if _eval(node.test, state, source):
            return _eval(node.body, state, source)
        return _eval(node.orelse, state, source)
    raise ExprError(_refuse(node, source))


def _refuse(node, source):
    name = type(node).__name__
    if name in ("ListComp", "SetComp", "DictComp", "GeneratorExp"):
        name = "Comprehension"
    return "%s is not allowed in a jig expression (in %r)" % (name, source)


def _name(identifier, state):
    if identifier in _LITERALS:
        return _LITERALS[identifier]
    if identifier.startswith("__"):
        raise ExprError("name %r is not allowed in a jig expression" % identifier)
    if identifier in state:
        return state[identifier]
    if identifier in _HELPERS:
        return _HELPERS[identifier]
    raise ExprError(
        "expression references %r, which is not in state (state has: %s)"
        % (identifier, ", ".join(sorted(state)) or "nothing")
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
        raise ExprError(_refuse(node, source))
    if path.startswith("__") or ".__" in path:
        raise ExprError("name %r is not allowed in a jig expression" % path)
    parts = path.split(".")
    current = _name(parts[0], state)
    for part in parts[1:]:
        if not isinstance(current, dict) or part not in current:
            raise ExprError(
                "expression references %r, which is not a mapping key in state" % path
            )
        current = current[part]
    return current


def _subscript(node, state, source):
    container = _eval(node.value, state, source)
    key = _eval(node.slice, state, source)
    try:
        return container[key]
    except (KeyError, IndexError, TypeError) as exc:
        raise ExprError("cannot index %r in %r (%s)" % (key, source, exc))


def _bool_op(node, state, source):
    values = [_eval(value, state, source) for value in node.values]
    if isinstance(node.op, ast.And):
        result = True
        for value in values:
            if not value:
                return value
            result = value
        return result
    for value in values:
        if value:
            return value
    return values[-1]


def _unary(node, state, source):
    value = _eval(node.operand, state, source)
    if isinstance(node.op, ast.Not):
        return not value
    if isinstance(node.op, ast.USub):
        return -value
    if isinstance(node.op, ast.UAdd):
        return +value
    raise ExprError(_refuse(node.op, source))


def _binary(node, state, source):
    handler = _BINARY.get(type(node.op))
    if handler is None:
        raise ExprError(_refuse(node.op, source))
    try:
        return handler(_eval(node.left, state, source), _eval(node.right, state, source))
    except (TypeError, ZeroDivisionError) as exc:
        raise ExprError("cannot evaluate %r (%s)" % (source, exc))


def _compare(node, state, source):
    left = _eval(node.left, state, source)
    for op, comparator in zip(node.ops, node.comparators):
        handler = _COMPARE.get(type(op))
        if handler is None:
            raise ExprError(_refuse(op, source))
        right = _eval(comparator, state, source)
        try:
            if not handler(left, right):
                return False
        except TypeError as exc:
            raise ExprError("cannot compare in %r (%s)" % (source, exc))
        left = right
    return True


def _call(node, state, source):
    if not isinstance(node.func, ast.Name) or node.func.id not in _HELPERS:
        called = getattr(node.func, "id", None) or _path(node.func) or "<expression>"
        raise ExprError(
            "%r is not a jig expression helper (allowed: %s)"
            % (called, ", ".join(sorted(_HELPERS)))
        )
    if node.keywords:
        raise ExprError("jig expression helpers take positional arguments only")
    arguments = [_eval(argument, state, source) for argument in node.args]
    try:
        return _HELPERS[node.func.id](*arguments)
    except Exception as exc:
        raise ExprError("helper %s() failed in %r (%s)" % (node.func.id, source, exc))
