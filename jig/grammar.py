"""The node contract: a JSON Schema subset, validated by hand.

Two jobs, and they are deliberately separate:

* `schema_to_grammar` turns a node's schema into the backend-neutral struct that gets
  handed to `Model.generate`. It is a pass-through today — real backends (T11) translate
  it into `response_format` / `grammar` on the way out. It also *checks* the schema, so a
  typo in a pack fails at `jig validate` time rather than at 3am in production.
* `validate_against` is the runtime half of verify-before-commit (docs/PLAN.md §3): even
  with a constrained decoder, jig never trusts output it has not checked itself.

The supported subset is `type`, `properties`, `required`, `enum`, `items`, and
`additionalProperties: false` — enough to express every node contract in the example
packs, small enough to read in one sitting. Anything outside it raises `SchemaError`
rather than being quietly ignored, because a silently-ignored constraint is a
constraint you think you have and don't.
"""

import copy

__all__ = [
    "SchemaError",
    "ValidationError",
    "schema_to_grammar",
    "validate_against",
]

GRAMMAR_KIND = "json_schema"

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
    """The schema itself is not something jig can enforce."""


class ValidationError(ValueError):
    """An object does not satisfy its node's schema.

    `path` is the dotted location of the offending value (`""` for the root), so
    callers can attribute a failure without parsing the message.
    """

    def __init__(self, path, message):
        self.path = path
        self.message = message
        where = path or "<root>"
        ValueError.__init__(self, "%s: %s" % (where, message))


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
        raise ValidationError(
            path,
            "%r is not one of %s"
            % (obj, ", ".join(repr(choice) for choice in schema["enum"])),
        )

    if isinstance(obj, dict):
        _validate_object(schema, obj, path)
    elif isinstance(obj, list) and schema.get("items") is not None:
        for index, item in enumerate(obj):
            validate_against(schema["items"], item, "%s[%d]" % (path, index))
    return None


def _validate_object(schema, obj, path):
    properties = schema.get("properties") or {}
    for name in schema.get("required") or []:
        if name not in obj:
            raise ValidationError(_join(path, name), "required property is missing")
    if schema.get("additionalProperties") is False:
        for name in obj:
            if name not in properties:
                raise ValidationError(
                    _join(path, name),
                    "unexpected property (schema sets additionalProperties: false)",
                )
    for name, sub in properties.items():
        if name in obj:
            validate_against(sub, obj[name], _join(path, name))


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
