"""The node contract: a JSON Schema subset, validated by hand.

Two jobs, and they are deliberately separate:

* `schema_to_grammar` turns a node's schema into the backend-neutral struct that gets
  handed to `Model.generate`. It is a pass-through today — real backends (T11) translate
  it into `response_format` / `grammar` on the way out. It also *checks* the schema, so a
  typo in a pack fails at `stepmold validate` time rather than at 3am in production.
* `validate_against` is the runtime half of verify-before-commit (docs/ARCHITECTURE.md §3): even
  with a constrained decoder, stepmold never trusts output it has not checked itself. It also
  refuses what `json.loads` accepts but JSON does not have — NaN, Infinity, and nesting
  deep enough to exhaust the interpreter — because every one of those is fatal *after*
  the commit, where no retry and no `on_fail` edge can reach it.

The supported subset is `type`, `properties`, `required`, `enum`, `items`, and
`additionalProperties: false` — enough to express every node contract in the example
packs, small enough to read in one sitting. Anything outside it raises `SchemaError`
rather than being quietly ignored, because a silently-ignored constraint is a
constraint you think you have and don't.
"""

import copy
import math

__all__ = [
    "SchemaError",
    "ValidationError",
    "schema_to_grammar",
    "validate_against",
]

GRAMMAR_KIND = "json_schema"

# How deep a candidate value may nest before stepmold refuses it. `json.loads` will happily
# build a structure thousands of levels deep, and everything downstream of the commit
# walks it recursively — `state._check`, `json.dumps`, the checkpoint — so a deep enough
# object dies with a `RecursionError`, which is not a `StepmoldError`, is not caught by the
# CLI, and does not take the node's `on_fail` edge. The ceiling is far above any node
# contract a pack expresses and far below the interpreter's own limit.
_MAX_DEPTH = 64

_TYPES = {
    "object": dict,
    "array": list,
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "null": type(None),
}
_KEYWORDS = {
    "type", "properties", "required", "enum", "items", "additionalProperties",
    "description", "title",
}


class SchemaError(ValueError):
    """The schema itself is not something stepmold can enforce."""


class ValidationError(ValueError):
    """An object does not satisfy its node's schema.

    `path` is the dotted location of the offending value (`""` for the root), so
    callers can attribute a failure without parsing the message.
    """

    def __init__(self, path, message, safe=None, safe_path=None):
        self.path = path
        self.message = message
        # `safe` is the model-facing half of the message. It may describe the constraint
        # (which comes from the pack's own schema) but never the offending value (which
        # came from the model). Quoting a rejected value back into a retry prompt is the
        # self-conditioning spiral stepmold/verify.py exists to prevent.
        self.safe = message if safe is None else safe
        # The *location* is model-authored text too, whenever the last segment is a key
        # the model invented: `additionalProperties: false` names the offending property,
        # and so does any check that walks a value the schema never declared. Those
        # errors pass their own safe location — the nearest one built from the pack's own
        # names — instead of letting `path` carry a smuggled instruction into a prompt.
        self.safe_path = path if safe_path is None else safe_path
        self.safe_text = "%s: %s" % (_clip(self.safe_path) or "<root>", self.safe)
        ValueError.__init__(self, "%s: %s" % (_clip(path) or "<root>", message))


def schema_to_grammar(schema):
    """Wrap a validated schema in the struct a backend knows how to translate."""
    check_schema(schema)
    return {"kind": GRAMMAR_KIND, "schema": copy.deepcopy(schema)}


def check_schema(schema, path=""):
    """Raise `SchemaError` unless `schema` uses only the supported subset."""
    if not isinstance(schema, dict):
        raise SchemaError(
            "%s: schema must be a mapping, got %s"
            % (path or "<root>", type(schema).__name__)
        )
    unknown = set(schema) - _KEYWORDS
    if unknown:
        raise SchemaError(
            "%s: unsupported schema keyword(s): %s"
            % (path or "<root>", ", ".join(sorted(unknown)))
        )
    for name in _declared_types(schema, path):
        if name not in _TYPES:
            raise SchemaError("%s: unknown type %r" % (path or "<root>", name))

    properties = schema.get("properties")
    if properties is not None:
        if not isinstance(properties, dict):
            raise SchemaError("%s: 'properties' must be a mapping" % (path or "<root>"))
        for key, sub in properties.items():
            check_schema(sub, _join(path, key))

    required = schema.get("required")
    if required is not None:
        if not isinstance(required, list) or not all(
            isinstance(item, str) for item in required
        ):
            raise SchemaError(
                "%s: 'required' must be a list of property names" % (path or "<root>")
            )

    enum = schema.get("enum")
    if enum is not None and (not isinstance(enum, list) or not enum):
        raise SchemaError("%s: 'enum' must be a non-empty list" % (path or "<root>"))

    items = schema.get("items")
    if items is not None:
        check_schema(items, path + "[]")

    extra = schema.get("additionalProperties")
    if extra is not None and not isinstance(extra, bool):
        raise SchemaError(
            "%s: 'additionalProperties' must be true or false" % (path or "<root>")
        )


def validate_against(schema, obj, path=""):
    """Return None if `obj` satisfies `schema`; raise `ValidationError` otherwise."""
    # Shape before schema. `json.loads` hands back two things that are not JSON — NaN /
    # Infinity, and nesting deep enough to exhaust the interpreter — and both are lethal
    # *after* the commit, where no retry and no `on_fail` edge can reach them. Checking
    # them where the value first enters makes them ordinary rejections.
    _check_shape(obj, path, 0)
    return _validate(schema, obj, path)


def _check_shape(value, path, depth):
    """Refuse a value that is JSON-shaped but not JSON, whatever the schema says.

    Walks the whole candidate, not just the parts the schema declares: the schema a
    compiler emits for a free-form field is `{"type": "object"}`, which pins nothing, and
    a NaN or a 3000-deep array under such a field is exactly the case that gets past
    validation and dies at checkpoint time.
    """
    if depth > _MAX_DEPTH:
        raise ValidationError(
            path,
            "value nests more than %d levels deep, which stepmold refuses to commit"
            % _MAX_DEPTH,
            safe="value nests more than %d levels deep" % _MAX_DEPTH,
            # Every segment past the root here may be a key the model invented.
            safe_path="",
        )
    if isinstance(value, float) and not math.isfinite(value):
        raise ValidationError(
            path,
            "%s is not a JSON number — strict JSON has no NaN or Infinity" % _show(value),
            safe="value is not a JSON number — JSON has no NaN or Infinity",
            safe_path="",
        )
    if isinstance(value, dict):
        for key, item in value.items():
            _check_shape(item, _join(path, key), depth + 1)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _check_shape(item, "%s[%d]" % (path, index), depth + 1)


def _validate(schema, obj, path):
    if not isinstance(schema, dict):
        raise SchemaError("%s: schema must be a mapping" % (path or "<root>"))

    declared = _declared_types(schema, path)
    if declared and not any(_is_type(obj, name) for name in declared):
        raise ValidationError(
            path,
            "expected %s, got %s"
            % (" or ".join(declared), type(obj).__name__),
        )

    if "enum" in schema and obj not in schema["enum"]:
        choices = ", ".join(repr(choice) for choice in schema["enum"])
        raise ValidationError(
            path,
            "%s is not one of %s" % (_show(obj), choices),
            safe="value is not one of %s" % choices,
        )

    if isinstance(obj, dict):
        _validate_object(schema, obj, path)
    elif isinstance(obj, list) and schema.get("items") is not None:
        for index, item in enumerate(obj):
            _validate(schema["items"], item, "%s[%d]" % (path, index))
    return None


def _validate_object(schema, obj, path):
    properties = schema.get("properties") or {}
    for name in schema.get("required") or []:
        if name not in obj:
            raise ValidationError(_join(path, name), "required property is missing")
    if schema.get("additionalProperties") is False:
        for name in obj:
            if name not in properties:
                allowed = ", ".join(sorted(properties)) or "none"
                raise ValidationError(
                    _join(path, name),
                    "unexpected property (schema sets additionalProperties: false)",
                    safe="unexpected property; the schema declares only: %s" % allowed,
                    # `name` came from the model, so the model-facing half locates the
                    # failure at the object instead of quoting the invented key.
                    safe_path=path,
                )
    for name, subschema in properties.items():
        if name in obj:
            _validate(subschema, obj[name], _join(path, name))


def _declared_types(schema, path):
    declared = schema.get("type")
    if declared is None:
        return []
    if isinstance(declared, str):
        return [declared]
    if isinstance(declared, list) and declared and all(
        isinstance(name, str) for name in declared
    ):
        return list(declared)
    raise SchemaError(
        "%s: 'type' must be a name or a list of names" % (path or "<root>")
    )


def _is_type(obj, name):
    expected = _TYPES.get(name)
    if expected is None:
        raise SchemaError("unknown type %r" % name)
    if name in ("integer", "number") and isinstance(obj, bool):
        return False  # JSON says booleans are not numbers; Python disagrees
    return isinstance(obj, expected)


def _join(path, name):
    return "%s.%s" % (path, name) if path else name


def _show(value):
    """A value, rendered for a *diagnostic* — bounded, because the value is not ours."""
    return _clip(repr(value))


def _clip(text, limit=120):
    """Keep a message bounded. A megabyte of ticket must not become a megabyte of log."""
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + "..."
