"""A YAML subset parser, because jig may not install PyYAML.

`TASKS.md` specifies the pack format as `manifest.yaml` / `graph.yaml`, and the
project rule is standard library only. The standard library has no YAML parser, so
this is the minimal piece implemented by hand (see NIGHT-LOG.md, T2).

Supported: block mappings, block sequences, nesting by indentation, `#` comments,
single-line flow collections (`[a, b]`, `{a: 1}`), quoted and plain scalars, block and
folded scalars (`|`, `>`, with `-`/`+` chomping), and the scalar types
null / bool / int / float / str.

Deliberately unsupported, each with a clear error: anchors and aliases, tags, multiple
documents, complex keys, block scalar lines indented less than the block they are in,
`key:value` without a space in a flow mapping, empty flow entries, and the C0 controls
`str.splitlines()` would mistake for line breaks. Pack files do not need them, and every
unsupported construct fails loudly rather than silently: where this parser cannot match
real YAML it refuses the input by name rather than guessing a value, because the pack
format is meant to read the same in a future Go or TypeScript runtime (docs/PLAN.md §7).
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
    "&": "anchors (`&`)",
    "*": "aliases (`*`)",
    "!": "tags (`!`)",
}
_BLOCK_HEADER = re.compile(r"^([|>])([-+]?)$")
# str.splitlines() breaks on these, so a document containing one would be silently
# re-flowed into extra lines. Real YAML rejects them outright, so jig does too. (NEL
# and U+2028/U+2029 are left alone: real YAML treats those as line breaks as well.)
_CONTROLS = "\x0b\x0c\x1c\x1d\x1e"


class _Line(object):
    __slots__ = ("indent", "text", "lineno", "block")

    def __init__(self, indent, text, lineno, block=None):
        self.indent = indent
        self.text = text
        self.lineno = lineno
        self.block = block  # resolved text when this line opened a `|` or `>` scalar


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
    _reject_controls(text, filename)
    raw_lines = text.splitlines()
    out = []
    index = 0
    started = False  # a `---` has opened the one document jig accepts
    ended = False  # a `...` has closed it
    while index < len(raw_lines):
        raw = raw_lines[index]
        number = index + 1
        if "\t" in raw[: len(raw) - len(raw.lstrip(" \t"))]:
            raise YamlError(
                "%s:%d: tabs cannot be used for indentation" % (filename, number)
            )
        stripped = _strip_comment(raw)
        if not stripped.strip():
            index += 1
            continue
        marker = stripped.strip()
        if marker in ("---", "..."):
            # One document is fine to frame; a second one would be silently merged
            # into the first, so refuse it by name.
            if marker == "---" and (started or out or ended):
                raise YamlError(
                    "%s:%d: jig's YAML subset does not support multiple "
                    "documents (`---`)" % (filename, number)
                )
            started = started or marker == "---"
            ended = ended or marker == "..."
            index += 1
            continue
        if ended:
            raise YamlError(
                "%s:%d: content after `...`; jig's YAML subset does not support "
                "multiple documents" % (filename, number)
            )
        indent = len(stripped) - len(stripped.lstrip(" "))
        content = stripped.strip()

        header = _block_header(content)
        if header is None:
            out.append(_Line(indent, content, number))
            index += 1
            continue

        style, chomp = header
        if chomp == "+":
            raise YamlError(
                "%s:%d: jig's YAML subset does not support keep chomping (`%s+`)"
                % (filename, number, style)
            )
        parent = _block_parent_indent(indent, content)
        body, index = _block_body(raw_lines, index + 1, parent, filename, number)
        out.append(_Line(indent, content, number, block=_fold(body, style, chomp)))
    return out


def _reject_controls(text, filename):
    """Refuse the C0 controls `str.splitlines()` would treat as line breaks."""
    hits = [found for found in (text.find(char) for char in _CONTROLS) if found >= 0]
    if not hits:
        return
    at = min(hits)
    # The sentinel keeps a break that ends the prefix from being dropped by
    # splitlines(), so a control on a fresh line reports that line, not the one before.
    number = len((text[:at] + ".").splitlines())
    raise YamlError(
        "%s:%d: control character %r is not allowed" % (filename, number, text[at])
    )


def _block_parent_indent(indent, content):
    """The column a line must out-indent to stay inside the block scalar in `content`.

    A block scalar's body belongs to its parent node. For `- |` the parent is the
    sequence, so the body only has to clear the dash. For `- key: |` the parent is the
    mapping that starts after the dash, so a sibling key in that mapping ends the body
    instead of being swallowed and re-sliced into the value.
    """
    while _is_item(content):
        rest = content[2:]
        offset = 2 + (len(rest) - len(rest.lstrip(" ")))
        rest = rest.strip()
        if _split_key(rest) is None and not _is_item(rest):
            break
        indent += offset
        content = rest
    return indent


def _block_header(content):
    """Return (style, chomp) when `content` ends in a `|`/`>` scalar header."""
    marker = content
    if ":" in content:
        split = _split_key(content)
        if split is None:
            return None
        marker = split[1]
    elif content.startswith("- "):
        marker = content[2:].strip()
    match = _BLOCK_HEADER.match(marker)
    return (match.group(1), match.group(2)) if match else None


def _block_body(raw_lines, index, indent, filename, opened_at):
    """Collect the raw lines belonging to a block scalar opened at `indent`."""
    body = []
    block_indent = None
    while index < len(raw_lines):
        raw = raw_lines[index]
        if raw.strip():
            current = len(raw) - len(raw.lstrip(" "))
            if current <= indent:
                break
            if block_indent is None:
                block_indent = current
            if current < block_indent:
                # Slicing at block_indent would delete leading characters and hand
                # back a plausible-looking wrong value, so refuse instead.
                raise YamlError(
                    "%s:%d: line is indented less than the block scalar opened on "
                    "line %d" % (filename, index + 1, opened_at)
                )
            body.append(raw[block_indent:])
        else:
            body.append("")
        index += 1
    while body and not body[-1].strip():
        body.pop()  # trailing blank lines are not content
    return body, index


def _fold(body, style, chomp):
    """Apply literal/folded joining and `-`/`+` chomping to a block scalar body."""
    if not body:
        return ""  # an empty block scalar is the empty string, not a lone break
    if style == "|":
        text = "\n".join(body)
    else:
        text = _fold_lines(body)
    return text.rstrip("\n") if chomp == "-" else text + "\n"


def _fold_lines(body):
    """Join a folded (`>`) body: breaks become spaces only between plain lines.

    A run of blank lines becomes that many breaks, and a line more indented than the
    block is kept verbatim with the breaks around it, because YAML folds only the
    lines it can safely re-flow.
    """
    parts = []
    previous = None  # the last non-blank line, or None before any content
    blanks = 0
    for line in body:
        if not line.strip():
            blanks += 1
            continue
        if previous is None:
            parts.append("\n" * blanks)
        elif previous.startswith(" ") or line.startswith(" "):
            parts.append("\n" * (blanks + 1))
        else:
            parts.append("\n" * blanks if blanks else " ")
        parts.append(line.rstrip() if line.startswith(" ") else line.strip())
        previous = line
        blanks = 0
    return "".join(parts)


def _strip_comment(raw):
    quote = None
    for index, char in enumerate(raw):
        if quote:
            if char == quote:
                quote = None
        elif char in "'\"" and (index == 0 or raw[index - 1] in " \t[{,"):
            # A quote only opens a string where a token may start. Inside a plain
            # scalar (`o'brien`, `6" pipe`, `b:'c`) it is just a character, and
            # treating it as a string would swallow the `#` of any comment after it.
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
        if line.block is not None:
            out[key] = line.block
            index += 1
        elif rest:
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
            # indented to the column the content actually starts at. Any block scalar
            # belongs to that inner mapping's key (`- when: |`), so it travels along.
            lines[index] = _Line(indent + offset, rest, line.lineno, line.block)
            value, index = _parse_block(lines, index, indent + offset, filename)
        elif line.block is not None:
            value, index = line.block, index + 1
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
    while True:
        if not rest:
            raise _err(filename, line, "unterminated flow collection")
        if rest.startswith(closer):
            # The collection is empty, or a comma just ended the last entry: real
            # YAML allows a trailing comma, and it must not invent an entry.
            return out, rest[1:]
        if rest.startswith(","):
            raise _err(filename, line, "empty entry in flow collection")
        if rest[0] in "[{":
            item, rest = _flow(rest, filename, line)
            key = None
            quoted = False
        else:
            quoted = rest[0] in "'\""
            token, rest = _flow_token(rest, closer, filename, line)
            item = None
            key = token
        if mapping:
            if key is None:
                raise _err(filename, line, "flow mapping keys must be scalars")
            if not key.endswith(":") and not rest.startswith(":"):
                raise _err(filename, line, "expected 'key: value' in flow mapping")
            if rest.startswith(":"):
                if not quoted and rest[1:2] not in ("", " ", ",", closer):
                    # Real YAML reads `{key:value}` as the single scalar 'key:value',
                    # so guessing a pair here would diverge from every other reader.
                    raise _err(filename, line,
                               "flow mapping needs a space after ':' (real YAML reads "
                               "'key:value' as one scalar)")
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
