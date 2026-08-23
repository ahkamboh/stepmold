"""A YAML subset parser, because jig may not install PyYAML.

`TASKS.md` specifies the pack format as `manifest.yaml` / `graph.yaml`, and the
project rule is standard library only. The standard library has no YAML parser, so
this is the minimal piece implemented by hand (see NIGHT-LOG.md, T2).

Supported: block mappings, block sequences, nesting by indentation, `#` comments,
single-line flow collections (`[a, b]`, `{a: 1}`), quoted and plain scalars, and the
scalar types null / bool / int / float / str.

Deliberately unsupported, each with a clear error: anchors and aliases, tags,
block scalars (`|`, `>`), multiple documents, and complex keys. Pack files do not
need them, and every unsupported construct fails loudly rather than silently.
"""

import re

__all__ = ["YamlError", "parse"]


class YamlError(ValueError):
    """A YAML document jig cannot parse, with the line it gave up on."""


_INT = re.compile(r"^[-+]?[0-9][0-9_]*$")
_FLOAT = re.compile(r"^[-+]?(?:[0-9][0-9_]*)?\.[0-9_]+(?:[eE][-+]?[0-9]+)?$|"
                    r"^[-+]?[0-9][0-9_]*[eE][-+]?[0-9]+$")
_NULL = ("", "~", "null", "Null", "NULL")
_TRUE = ("true", "True", "TRUE", "yes", "Yes", "YES", "on", "On", "ON")
_FALSE = ("false", "False", "FALSE", "no", "No", "NO", "off", "Off", "OFF")
_UNSUPPORTED = {
    "|": "block scalars (`|`)",
    ">": "folded scalars (`>`)",
    "&": "anchors (`&`)",
    "*": "aliases (`*`)",
    "!": "tags (`!`)",
}


class _Line(object):
    __slots__ = ("indent", "text", "lineno")

    def __init__(self, indent, text, lineno):
        self.indent = indent
        self.text = text
        self.lineno = lineno


def parse(text, filename="<yaml>"):
    """Parse a YAML subset document into plain Python data."""
    lines = _lex(text, filename)
    if not lines:
        return None
    if lines[0].indent != 0:
        raise _err(filename, lines[0], "document must not start indented")
    value, index = _parse_block(lines, 0, 0, filename)
    if index != len(lines):
        raise _err(filename, lines[index], "unexpected content")
    return value


# --------------------------------------------------------------------------- lexing


def _lex(text, filename):
    out = []
    for number, raw in enumerate(text.splitlines(), start=1):
        if "\t" in raw[: len(raw) - len(raw.lstrip(" \t"))]:
            raise YamlError(
                "%s:%d: tabs cannot be used for indentation" % (filename, number)
            )
        stripped = _strip_comment(raw)
        if not stripped.strip():
            continue
        if stripped.strip() in ("---", "..."):
            continue
        indent = len(stripped) - len(stripped.lstrip(" "))
        out.append(_Line(indent, stripped.strip(), number))
    return out


def _strip_comment(raw):
    quote = None
    for index, char in enumerate(raw):
        if quote:
            if char == quote:
                quote = None
        elif char in "'\"":
            quote = char
        elif char == "#" and (index == 0 or raw[index - 1] in " \t"):
            return raw[:index]
    return raw


# --------------------------------------------------------------------------- blocks


def _parse_block(lines, index, indent, filename):
    if _is_item(lines[index].text):
        return _parse_sequence(lines, index, indent, filename)
    return _parse_mapping(lines, index, indent, filename)


def _is_item(text):
    return text == "-" or text.startswith("- ")


def _parse_mapping(lines, index, indent, filename):
    out = {}
    while index < len(lines) and lines[index].indent == indent:
        line = lines[index]
        if _is_item(line.text):
            raise _err(filename, line, "sequence item where a mapping key was expected")
        split = _split_key(line.text)
        if split is None:
            raise _err(filename, line, "expected 'key: value'")
        key_text, rest = split
        key = _scalar(key_text, filename, line)
        if not isinstance(key, (str, int, float, bool)) and key is not None:
            raise _err(filename, line, "unsupported complex key")
        if key in out:
            raise _err(filename, line, "duplicate key %r" % (key,))
        if rest:
            out[key] = _value(rest, filename, line)
            index += 1
        else:
            child = index + 1
            if child < len(lines) and lines[child].indent > indent:
                out[key], index = _parse_block(
                    lines, child, lines[child].indent, filename
                )
            else:
                out[key] = None
                index += 1
    if index < len(lines) and lines[index].indent > indent:
        raise _err(filename, lines[index], "unexpected indentation")
    return out, index


def _parse_sequence(lines, index, indent, filename):
    out = []
    while (
        index < len(lines)
        and lines[index].indent == indent
        and _is_item(lines[index].text)
    ):
        line = lines[index]
        if line.text == "-":
            child = index + 1
            if child < len(lines) and lines[child].indent > indent:
                value, index = _parse_block(lines, child, lines[child].indent, filename)
            else:
                value, index = None, index + 1
            out.append(value)
            continue
        rest = line.text[2:]
        offset = 2 + (len(rest) - len(rest.lstrip(" ")))
        rest = rest.strip()
        if _split_key(rest) is not None or _is_item(rest):
            # A collection opened on the dash line: re-read it as its own block,
            # indented to the column the content actually starts at.
            lines[index] = _Line(indent + offset, rest, line.lineno)
            value, index = _parse_block(lines, index, indent + offset, filename)
        else:
            value, index = _value(rest, filename, line), index + 1
        out.append(value)
    if index < len(lines) and lines[index].indent > indent:
        raise _err(filename, lines[index], "unexpected indentation")
    return out, index


def _split_key(text):
    """Split `key: value` at the first colon outside quotes and flow collections."""
    quote = None
    depth = 0
    for index, char in enumerate(text):
        if quote:
            if char == quote:
                quote = None
        elif char in "'\"":
            quote = char
        elif char in "[{":
            depth += 1
        elif char in "]}":
            depth -= 1
        elif char == ":" and depth == 0:
            after = text[index + 1:]
            if after and not after.startswith(" "):
                continue
            return text[:index].strip(), after.strip()
    return None


# --------------------------------------------------------------------------- scalars


def _value(text, filename, line):
    if text.startswith("[") or text.startswith("{"):
        value, rest = _flow(text, filename, line)
        if rest.strip():
            raise _err(filename, line, "trailing content after flow collection")
        return value
    return _scalar(text, filename, line)


def _flow(text, filename, line):
    if text.startswith("["):
        return _flow_collection(text, "]", filename, line, mapping=False)
    return _flow_collection(text, "}", filename, line, mapping=True)


def _flow_collection(text, closer, filename, line, mapping):
    out = {} if mapping else []
    rest = text[1:].lstrip()
    if rest.startswith(closer):
        return out, rest[1:]
    while True:
        if not rest:
            raise _err(filename, line, "unterminated flow collection")
        if rest[0] in "[{":
            item, rest = _flow(rest, filename, line)
            key = None
        else:
            token, rest = _flow_token(rest, closer, filename, line)
            item = None
            key = token
        if mapping:
            if key is None:
                raise _err(filename, line, "flow mapping keys must be scalars")
            if not key.endswith(":") and not rest.startswith(":"):
                raise _err(filename, line, "expected 'key: value' in flow mapping")
            if rest.startswith(":"):
                rest = rest[1:].lstrip()
            else:
                key = key[:-1].strip()
            if key.endswith(":"):
                key = key[:-1].strip()
            if rest[:1] in ("[", "{"):
                value, rest = _flow(rest, filename, line)
            else:
                token, rest = _flow_token(rest, closer, filename, line)
                value = _scalar(token, filename, line)
            out[_scalar(key, filename, line)] = value
        else:
            out.append(item if key is None else _scalar(key, filename, line))
        rest = rest.lstrip()
        if rest.startswith(","):
            rest = rest[1:].lstrip()
            continue
        if rest.startswith(closer):
            return out, rest[1:]
        raise _err(filename, line, "expected ',' or '%s' in flow collection" % closer)


def _flow_token(text, closer, filename, line):
    if text[:1] in ("'", '"'):
        quote = text[0]
        end = _closing_quote(text, quote)
        if end < 0:
            raise _err(filename, line, "unterminated quoted string")
        return text[: end + 1], text[end + 1:].lstrip()
    stop = len(text)
    for index, char in enumerate(text):
        if char in ",:" or char in "]}":
            stop = index
            break
    return text[:stop].strip(), text[stop:]


def _closing_quote(text, quote):
    index = 1
    while index < len(text):
        if text[index] == "\\" and quote == '"':
            index += 2
            continue
        if text[index] == quote:
            if quote == "'" and text[index + 1:index + 2] == "'":
                index += 2  # '' is an escaped quote inside a single-quoted scalar
                continue
            return index
        index += 1
    return -1


def _scalar(text, filename, line):
    text = text.strip()
    if text[:1] in _UNSUPPORTED and text[:1] not in ("-", "+"):
        raise _err(filename, line, "jig's YAML subset does not support %s"
                   % _UNSUPPORTED[text[0]])
    if text[:1] == "'":
        return _unquote_single(text, filename, line)
    if text[:1] == '"':
        return _unquote_double(text, filename, line)
    if text in _NULL:
        return None
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    if _INT.match(text):
        return int(text.replace("_", ""))
    if _FLOAT.match(text):
        return float(text.replace("_", ""))
    return text


def _unquote_single(text, filename, line):
    end = _closing_quote(text, "'")
    if end != len(text) - 1:
        raise _err(filename, line, "unterminated quoted string")
    return text[1:end].replace("''", "'")


_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "0": "\0", '"': '"', "\\": "\\", "/": "/"}


def _unquote_double(text, filename, line):
    end = _closing_quote(text, '"')
    if end != len(text) - 1:
        raise _err(filename, line, "unterminated quoted string")
    body = text[1:end]
    out = []
    index = 0
    while index < len(body):
        char = body[index]
        if char == "\\" and index + 1 < len(body):
            following = body[index + 1]
            if following == "u":
                out.append(chr(int(body[index + 2:index + 6], 16)))
                index += 6
                continue
            if following not in _ESCAPES:
                raise _err(filename, line, "unknown escape '\\%s'" % following)
            out.append(_ESCAPES[following])
            index += 2
            continue
        out.append(char)
        index += 1
    return "".join(out)


def _err(filename, line, message):
    return YamlError("%s:%d: %s" % (filename, line.lineno, message))
