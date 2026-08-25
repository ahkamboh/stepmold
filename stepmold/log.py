"""Structured logging for the runtime — silent until an operator asks for it.

A stepmold run used to print its final JSON and nothing else. Every fact an operator needs at
3am already existed inside the runtime — `graph` tracks the path and the attempts,
`verify` knows every rejection and why, `openai_compat` knows the tokens and the
milliseconds — and none of it was written down. This module is where it gets written.

Three rules hold it together, and two of them are about what must *not* come out.

**1. A library does not configure logging for its host.**
Importing `stepmold` must not touch the root logger, must not call `basicConfig`, and must not
put a byte on stdout or stderr. So the package logger gets a `NullHandler` at import and
nothing else. That single line does more than swallow records: `logging.lastResort` only
fires when a record reaches the end of the chain having found *no* handler at all, so
without the NullHandler a lone `logger.warning(...)` would print to stderr from a library
that was never switched on. `propagate` is deliberately left alone, so a host that has
configured logging of its own still sees stepmold's records; `configure` — which only the CLI
calls — is the explicit opt-in that installs stepmold's own handler.

**2. Secrets and model text do not go in a log.**
`redact` is the same key-shaped filter `stepmold.backends.openai_compat` has always run over
upstream error bodies; it lives here now because "text that is about to be written down"
is this module's subject, and the backend imports it back. Every formatter runs it over
every string it emits, so redaction is a property of the *sink* rather than a discipline
at two hundred call sites. Rejected model output is the other half, and the rule is in
`stepmold.verify`: the INFO path carries `Rejected.feedback` (what was wrong), never
`Rejected.detail` (what the model said). The detail is DEBUG-only — an operator who asks
for DEBUG is asking to read what came back, and bytes in a log file cannot condition a
model. Prompts and state never appear at all: sizes and digests instead.

**3. Disabled logging costs nothing.**
The long-horizon suite walks 50-node packs, so no call site may build a string, a dict of
computed values, or a JSON blob that a filtered-out level will throw away. `event` checks
`isEnabledFor` before it does anything, and the call sites that would have to *compute* a
field — a digest, a token summary — guard themselves with `isEnabledFor` first.

Two output shapes, because two audiences read logs. `TextFormatter` is for a terminal;
`JsonFormatter` writes one JSON object per line with stable field names, for anything
that ships logs somewhere.
"""

import hashlib
import json
import logging
import re
import sys
import time

__all__ = [
    "BEARER",
    "CLIP",
    "DEBUG",
    "ERROR",
    "INFO",
    "KEY_SHAPED",
    "LEVELS",
    "ROOT",
    "WARNING",
    "JsonFormatter",
    "TextFormatter",
    "clip",
    "configure",
    "digest",
    "event",
    "get_logger",
    "level_named",
    "redact",
    "reset",
    "size_of",
]

#: The package logger every stepmold module hangs off. `logging.getLogger("stepmold")` in a host
#: application reaches the same object, which is the whole point of naming it.
ROOT = "stepmold"

DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR

#: What `--log-level` accepts, plus "off" for "do not configure anything".
LEVELS = ("off", "debug", "info", "warning", "error")

#: How much of any single user-controlled string may reach a log line. A support ticket
#: can be a megabyte; a log line about a support ticket may not be.
CLIP = 200

# Credentials, as they appear in text somebody else wrote: a header a gateway echoed back
# at us, or a key pasted into an upstream error message. These moved here from
# `stepmold.backends.openai_compat` when logging arrived — the backend still imports them, and
# there is still exactly one of them.
BEARER = re.compile(r"(?i)(bearer\s+)[^\s\"',]+")
# The separator is part of the vendor's format, not a detail: Groq issues `gsk_...` and
# GitHub issues `ghp_...`, so a pattern that only accepted a hyphen matched neither of the
# two prefixes most likely to show up in a real 401 body. Both separators, and the GitHub
# token family in full, because `ghp_` is only one of five.
KEY_SHAPED = re.compile(
    r"\b(?:c?sk|gsk|xai|xox[abpsr]|gh[pousr]|AKIA|ASIA)[-_][A-Za-z0-9_-]{8,}"
)

#: LogRecord attributes the formatters read. Prefixed because `extra=` writes straight
#: onto the record and a collision with a stdlib attribute raises at emit time.
EVENT_ATTR = "stepmold_event"
FIELDS_ATTR = "stepmold_fields"

_HANDLER_MARK = "_stepmold_handler"

_package_logger = logging.getLogger(ROOT)
# The one line that makes stepmold a well-behaved library: no output, and no `lastResort`.
_package_logger.addHandler(logging.NullHandler())


def get_logger(name):
    """The logger for one stepmold module — `get_logger("graph")` -> `stepmold.graph`.

    Named for the module rather than passed `__name__` so the hierarchy stays flat and
    predictable: an operator filters on `stepmold.backend` without knowing that it is spelled
    `stepmold.backends.openai_compat` on disk.
    """
    return logging.getLogger("%s.%s" % (ROOT, name))


def event(logger, level, name, /, **fields):
    """Emit one structured event, or do nothing if `level` is filtered out.

    The three parameters are positional-only so that `level=`, `name=` and `logger=` are
    usable as *field* names — `event(_log, INFO, "run.start", level="p1")` has to mean
    what it looks like, and a logging helper that quietly stole three field names would
    be found the hard way by whoever needed one of them.

    `name` is the event, `fields` are its data — one flat mapping of stable field names,
    because a log line an operator can grep and a log line a collector can parse are the
    same line here. Nothing is formatted at the call site: the record carries the fields
    as objects and the formatter decides what a line looks like, which is what keeps a
    filtered-out DEBUG event down to one `isEnabledFor` check.

    Callers whose fields cost something to *compute* — a digest, a token summary, a
    length over a large structure — must guard with `logger.isEnabledFor(...)` themselves
    before calling, because Python evaluates the arguments before the check in here.
    """
    if not logger.isEnabledFor(level):
        return
    logger.log(level, name, extra={EVENT_ATTR: name, FIELDS_ATTR: fields})


# ------------------------------------------------------------------ safe field values


def redact(text):
    """Strip anything credential-shaped out of text before it is written down.

    Unchanged from where it used to live in the backend, and for the same reason: error
    bodies are written by somebody else and read by our logs, and a gateway that echoes
    the offending request back puts the caller's own `Authorization` header in it. stepmold is
    not the leaker there, but without this it is the amplifier.
    """
    if not text:
        return text
    return KEY_SHAPED.sub("<redacted>", BEARER.sub(r"\1<redacted>", text))


def clip(text, limit=CLIP):
    """One line, bounded. Whatever a log line is about, it is not a megabyte long."""
    collapsed = " ".join(str(text).split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[:limit] + "...(%d chars)" % len(collapsed)


def size_of(value):
    """The JSON size of a value, in bytes — what may be said about state at INFO."""
    try:
        return len(json.dumps(value, sort_keys=True, default=repr))
    except (TypeError, ValueError):  # pragma: no cover - default=repr covers the rest
        return -1


def digest(value):
    """A short stable fingerprint of a value, for correlating without quoting.

    Two runs whose `state_digest` matches saw the same state; nobody reading the log
    learns what was in it. That is the trade this function exists to make.
    """
    try:
        blob = json.dumps(value, sort_keys=True, default=repr).encode("utf-8")
    except (TypeError, ValueError):  # pragma: no cover - default=repr covers the rest
        blob = repr(value).encode("utf-8", "replace")
    return hashlib.sha256(blob).hexdigest()[:12]


def _safe(value):
    """A field value as it may appear in a log: redacted, clipped, JSON-representable."""
    if isinstance(value, str):
        # Redact first, THEN clip. The other order truncates a credential that
        # straddles the clip boundary until KEY_SHAPED no longer matches it, and the
        # surviving prefix is printed verbatim. Reproduced at offsets 185-196.
        return clip(redact(value))
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, dict):
        return dict((str(key), _safe(item)) for key, item in value.items())
    return redact(clip(repr(value)))


# ----------------------------------------------------------------------- formatters


class TextFormatter(logging.Formatter):
    """One event per line, for a human at a terminal.

        14:02:11.402 INFO  stepmold.graph node.ok node=classify attempt=1 duration_ms=0.8

    Values are printed bare when they are a single word and JSON-quoted when they are
    not, so a field containing a space cannot be misread as two fields.
    """

    def format(self, record):
        fields = getattr(record, FIELDS_ATTR, None) or {}
        rendered = " ".join(
            "%s=%s" % (key, _as_text(value)) for key, value in fields.items()
        )
        line = "%s %-5s %s %s" % (
            _timestamp(record), record.levelname, record.name, _event_of(record)
        )
        return line + (" " + rendered if rendered else "")


class JsonFormatter(logging.Formatter):
    """One JSON object per line, for anything that ships logs.

    `ts`, `level`, `logger` and `event` come first and always exist; the event's own
    fields follow at the top level, because `jq '.attempt'` is what somebody will
    actually type. A field that collides with one of the four reserved names is prefixed
    rather than dropped — losing data to make a line tidy is the wrong trade in the file
    somebody reads after an incident.
    """

    RESERVED = ("ts", "level", "logger", "event")

    def format(self, record):
        payload = {
            "ts": _timestamp(record, date=True),
            "level": record.levelname,
            "logger": record.name,
            "event": _event_of(record),
        }
        for key, value in (getattr(record, FIELDS_ATTR, None) or {}).items():
            name = "field_%s" % key if key in self.RESERVED else key
            payload[name] = _safe(value)
        if record.exc_info:
            payload["exc_type"] = record.exc_info[0].__name__
        # ensure_ascii=False: a log file is UTF-8, and escaping an em dash to
        # \\u2014 makes the one field an operator is trying to read unreadable.
        return json.dumps(payload, default=str, ensure_ascii=False)


def _event_of(record):
    """The event name, or a plain record's message — redacted either way.

    Only `stepmold_fields` used to be filtered, so a record logged through the ordinary
    stdlib API on a `stepmold.*` logger reached the output unfiltered. Redaction is supposed
    to be a property of the sink, not a discipline every call site remembers, so the
    message goes through the same filter as the fields.
    """
    name = getattr(record, EVENT_ATTR, None)
    if name is not None:
        return redact(str(name))
    return clip(redact(record.getMessage()))


def _timestamp(record, date=False):
    shape = "%Y-%m-%dT%H:%M:%S" if date else "%H:%M:%S"
    stamp = time.strftime(shape, time.gmtime(record.created))
    return "%s.%03d%s" % (stamp, record.msecs, "Z" if date else "")


_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


def _as_text(value):
    """One field's value as a terminal line spells it.

    A missing value prints as `-` rather than vanishing, because a field that
    disappears when it is empty makes two runs' log lines impossible to diff. Lists
    print comma-joined — `inputs=ticket,account` — which is what an eye reads; the JSON
    formatter is where a machine goes for the real structure.
    """
    safe = _safe(value)
    if safe is None:
        return "-"
    if isinstance(safe, str) and _CONTROL.search(safe):
        # An unescaped ESC from upstream text would run as a terminal command in the
        # operator's console. json.dumps turns it into \u001b, which is readable and inert.
        return json.dumps(safe, ensure_ascii=False)
    if isinstance(safe, bool):
        return "true" if safe else "false"
    if isinstance(safe, (int, float)):
        return repr(safe)
    if isinstance(safe, list):
        safe = ",".join(_as_text(item) for item in safe)
    elif not isinstance(safe, str):
        safe = json.dumps(safe, default=str, sort_keys=True, ensure_ascii=False)
    if safe == "" or any(char.isspace() for char in safe) or '"' in safe:
        return json.dumps(safe, ensure_ascii=False)
    return safe


# --------------------------------------------------------------------- configuration


def configure(level="info", fmt="text", stream=None):
    """Turn stepmold's logging on. Only an application — the CLI — may call this.

    Everything lands on **stderr** by default, never stdout: `stepmold run` prints its result
    as JSON on stdout and a caller pipes it onward, so a log line there would corrupt the
    output rather than describe it.

    `propagate` is switched off once stepmold owns a handler of its own. A host that had
    already configured the root logger would otherwise get every record twice, which is
    the classic library-logging bug and reads as a runtime that repeats itself.
    """
    logger = logging.getLogger(ROOT)
    for handler in list(logger.handlers):
        if getattr(handler, _HANDLER_MARK, False):
            logger.removeHandler(handler)
            handler.close()
    handler = logging.StreamHandler(sys.stderr if stream is None else stream)
    handler.setFormatter(JsonFormatter() if fmt == "json" else TextFormatter())
    setattr(handler, _HANDLER_MARK, True)
    logger.addHandler(handler)
    logger.setLevel(level_named(level))
    logger.propagate = False
    return logger


def reset():
    """Undo `configure` — back to a library that emits nothing on its own.

    Here for tests and for a host embedding stepmold that wants its own logging back; the
    NullHandler stays, because rule 1 above holds whether or not anyone configured
    anything.
    """
    logger = logging.getLogger(ROOT)
    for handler in list(logger.handlers):
        if getattr(handler, _HANDLER_MARK, False):
            logger.removeHandler(handler)
            handler.close()
    logger.setLevel(logging.NOTSET)
    logger.propagate = True
    return logger


def level_named(level):
    """`"debug"` / `"DEBUG"` / `logging.DEBUG` -> a level number. `"off"` silences."""
    if isinstance(level, int):
        return level
    name = str(level).strip().lower()
    if name in ("off", "none", "silent"):
        return logging.CRITICAL + 1
    number = logging.getLevelName(name.upper())
    if not isinstance(number, int):
        raise ValueError(
            "unknown log level %r (known: %s)" % (level, ", ".join(LEVELS))
        )
    return number
