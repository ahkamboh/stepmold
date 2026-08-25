"""`python3 -m jig` — run a pack, score it, or check it.

Three commands, because a pack has exactly three things you do to it:

    jig validate <pack>                     is this pack well-formed?
    jig run <pack> --input '<json>'         execute it once
    jig eval <pack>                         score it against its contract

Exit codes are the contract with CI: **0** success, **1** the thing failed (invalid pack,
failed run, evalset not fully passed), **2** you called it wrong (argparse's own code).
`jig eval` exiting 1 on a single failed case is the point — that is what makes an evalset
a gate rather than a report.

argparse only, per the stdlib rule. ARCHITECTURE.md §7 names Typer + Rich; both are dependencies,
so the help text here is plain and the report text is plain.
"""

import argparse
import json
import os
import sys

from . import __version__, log
from .errors import JigError
from .eval import evaluate
from .grammar import ValidationError
from .pack import PackError, _resolve_inside, load_pack

__all__ = ["main", "resolve_model"]

MODEL_SCHEMES = ("fake", "openai")
LOG_FORMATS = ("text", "json")


def main(argv=None):
    """Run the CLI. Returns an exit code rather than calling sys.exit, so it is testable."""
    parser = build_parser()
    args = parser.parse_args(argv)
    _start_logging(args)
    try:
        return args.handler(args)
    except PackError as exc:
        return _fail("pack error: %s" % exc)
    except _build_error() as exc:
        return _fail("build error: %s" % exc)
    except JigError as exc:
        return _fail("%s: %s" % (type(exc).__name__, exc))
    except (ValidationError, ValueError) as exc:
        return _fail(str(exc))


def _build_error():
    """BuildError, or a class that never matches when the compiler is not installed.

    Importing jig.build from the top of this module would defeat the separation the
    compiler is built around, so the exception type is fetched only when it is needed.
    """
    try:
        from .build.spec import BuildError
    except Exception:  # pragma: no cover - the runtime may ship without jig.build
        class BuildError(Exception):
            pass
    return BuildError


def build_parser():
    parser = argparse.ArgumentParser(
        prog="jig",
        description="Compile your agent once with a frontier model. "
                    "Run it forever on a small one.",
    )
    parser.add_argument("--version", action="version", version="jig %s" % __version__)
    commands = parser.add_subparsers(dest="command")
    observability = _observability_options()

    validate = commands.add_parser("validate", parents=[observability],
                                   help="check that a pack is well-formed")
    validate.add_argument("pack", help="path to the pack directory")
    validate.set_defaults(handler=command_validate)

    run = commands.add_parser("run", parents=[observability], help="execute a pack once")
    run.add_argument("pack", help="path to the pack directory")
    run.add_argument("--input", default="{}", help="run inputs as a JSON object")
    run.add_argument("--model", help="model spec (default: the pack manifest's)")
    run.add_argument("--allow-pack-model", dest="allow_pack_model", action="store_true",
                     help="accept a network endpoint chosen by the pack's manifest")
    run.add_argument("--run-id", dest="run_id", help="name this run")
    run.add_argument("--store", help="SQLite file to checkpoint into")
    run.add_argument("--resume", help="continue a previous run id (needs --store)")
    run.add_argument("--state", action="store_true",
                     help="print the whole final state, not the end node's projection")
    run.set_defaults(handler=command_run)

    build = commands.add_parser(
        "build", help="compile a pack from a task description and gold examples")
    build.add_argument("spec", help="directory holding task.md and examples.jsonl")
    build.add_argument("-o", "--out", required=True, help="where to write the pack")
    build.add_argument("--model", required=True,
                       help="the planning model, e.g. openai:https://host/v1#model")
    build.add_argument("--name", help="pack name (default: the output directory's name)")
    build.add_argument("--attempts", type=int, default=3,
                       help="how many times to re-plan on a failing eval (default: 3)")
    build.add_argument("--overwrite", action="store_true",
                       help="replace the output directory if it already exists")
    build.set_defaults(handler=command_build)

    evaluate_command = commands.add_parser(
        "eval", parents=[observability],
        help="score a pack against its evalset (exit 1 if any case fails)"
    )
    evaluate_command.add_argument("pack", help="path to the pack directory")
    evaluate_command.add_argument("--model", help="model spec (default: the manifest's)")
    evaluate_command.add_argument("--json", action="store_true",
                                  help="emit the report as JSON")
    evaluate_command.set_defaults(handler=command_eval)

    parser.set_defaults(handler=lambda args: _usage(parser))
    return parser


def _observability_options():
    """`--log-level` / `--log-format`, shared by every subcommand.

    A parent parser rather than three copies, and rather than options on the top-level
    parser: argparse only accepts a top-level option *before* the subcommand, and nobody
    types `jig --log-level info run pack`.

    The default is `off`, and that is the contract: with no flag, jig configures no
    handler, sets no level, and prints exactly what it printed before this existed.
    Logging is an operator's explicit request, not a thing that happens to them.
    """
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--log-level", dest="log_level", default="off", choices=log.LEVELS,
        help="emit run events on stderr at this level (default: off)",
    )
    shared.add_argument(
        "--log-format", dest="log_format", default="text", choices=LOG_FORMATS,
        help="'text' for a terminal, 'json' for one JSON object per line",
    )
    return shared


def _start_logging(args):
    """Turn logging on if this invocation asked for it. Only an application may.

    stderr, always. `jig run` prints its result as JSON on stdout and callers pipe it
    onward; a log line there would corrupt the output instead of describing it.
    """
    level = getattr(args, "log_level", "off")
    if not level or level == "off":
        return
    log.configure(level=level, fmt=getattr(args, "log_format", "text"),
                  stream=sys.stderr)


# ------------------------------------------------------------------------ commands


def command_validate(args):
    pack = load_pack(args.pack)
    _check_output_shapes(pack)
    print(
        "%s v%s: %s, %s, %s, entry %r"
        % (
            pack.name,
            pack.version,
            _count(len(pack.nodes), "node"),
            _count(len(pack.edges), "edge"),
            _count(len(pack.evalset), "evalset case"),
            pack.entry,
        )
    )
    return 0


def _count(number, noun):
    return "%d %s%s" % (number, noun, "" if number == 1 else "s")


def command_run(args):
    from .graph import run as run_pack
    from .state import Store, resume

    pack = load_pack(args.pack)
    _check_output_shapes(pack)
    if args.resume and not args.store:
        return _fail("--resume needs --store: checkpoints live in the store")

    store = Store(args.store) if args.store else None
    try:
        if args.resume:
            result = resume(pack, resolve_model(args.model, pack, _allow(args)), args.resume, store)
        else:
            result = run_pack(
                pack,
                resolve_model(args.model, pack, _allow(args)),
                _parse_input(args.input),
                run_id=args.run_id,
                store=store,
            )
    finally:
        if store is not None:
            store.close()

    payload = result.state if args.state else result.output
    if not args.state and not payload and result.state:
        # `{}` printed on stdout with exit 0 reads as "the run produced nothing", when
        # what happened is that the end node projected nothing out of a state that has
        # content. Say which, rather than let a caller pipe an empty result onward.
        return _fail(
            "end node %r projected nothing: its 'output' names no key that exists in "
            "state (state has: %s). Fix the node's 'output', or pass --state to print "
            "the whole state." % (result.end_node, ", ".join(sorted(result.state)))
        )
    print(json.dumps(payload, sort_keys=True))
    return 0


def command_eval(args):
    pack = load_pack(args.pack)
    _check_output_shapes(pack)
    report = evaluate(pack, resolve_model(args.model, pack, _allow(args)))
    print(json.dumps(_report_json(report), sort_keys=True) if args.json
          else report.summary())
    return 0 if report.passed_all else 1


def _check_output_shapes(pack):
    """Refuse a pack that uses `output:` in the shape the other node type wants.

    One word, two meanings: on a `generate` node `output` is the single state key to
    commit the result under, on an `end` node it is the list of keys to project. Nothing
    checks which one was written — `output: ticket` on an end node makes the projection
    iterate the *string*, match no state key, and return an empty object while the state
    still holds the data. That belongs in `pack.load_pack` at load time; until it lives
    there, the CLI refuses the pack instead of running it and printing the silence.
    """
    problems = []
    for name in sorted(pack.nodes):
        node = pack.nodes[name]
        if node.output is None:
            continue
        if node.type == "end" and not _is_key_list(node.output):
            hint = ""
            if isinstance(node.output, str):
                hint = " — write 'output: [%s]' if you meant that one key" % node.output
            problems.append(
                "graph.yaml: end node %r: 'output' must be a list of state keys to "
                "project, got %r%s" % (name, node.output, hint)
            )
        elif node.type == "generate" and not isinstance(node.output, str):
            problems.append(
                "graph.yaml: generate node %r: 'output' must be a single state key to "
                "commit the result under (a string), got %r" % (name, node.output)
            )
    if problems:
        raise ValueError("\n     ".join(problems))


def _is_key_list(output):
    return isinstance(output, list) and all(isinstance(key, str) for key in output)


# -------------------------------------------------------------------------- models


def _allow(args):
    """Whether this invocation accepts a network endpoint chosen by the pack."""
    return bool(getattr(args, "allow_pack_model", False))


def command_build(args):
    """Compile a pack. The only subcommand that needs a model at all times.

    jig.build is imported here rather than at module scope on purpose: the runtime ships
    to a client box and must not carry the compiler, and a test asserts that importing
    jig.cli does not pull jig.build in behind it.
    """
    from .build.compile import compile_pack, load_build_spec

    description, cases = load_build_spec(args.spec)

    # A build model is always given explicitly. There is no pack to read a default from,
    # and a compile is the one moment where quietly choosing a model for someone would be
    # the wrong kind of helpful.
    model = _build_model(args.model)

    result = compile_pack(
        args.out, description, cases, model,
        name=args.name, attempts=args.attempts, overwrite=args.overwrite,
        on_event=lambda message: print(message, file=sys.stderr),
    )
    print(result.directory)
    return 0


def _build_model(spec):
    """Resolve --model for a build. Same spec grammar as `run`, minus the pack."""
    scheme, _, rest = spec.partition(":")
    if scheme == "openai":
        return _openai_model(rest)
    if scheme == "fake":
        from .model import FakeModel

        full = os.path.abspath(rest)
        if not os.path.isfile(full):
            raise ValueError("fake: no such script %r" % full)
        with open(full) as handle:
            return FakeModel(json.load(handle))
    raise ValueError(
        "unknown model scheme %r for build (known: openai, fake)" % scheme
    )


def resolve_model(spec, pack, allow_pack_model=False):
    """Turn a model spec into a `Model`.

    Two schemes:

        fake:<path>                       a scripted FakeModel from JSON — a list of
                                          responses, or an object keyed by prompt
                                          substring. A relative path resolves inside the
                                          pack, which is what lets a pack ship its own
                                          offline model so CI needs no GPU.
        openai:<base_url>#<model>[#<grammar_mode>[#<reasoning_reserve>]]
                                          an OpenAI-compatible server (llama.cpp-server,
                                          vLLM, SGLang). Constructing it opens no
                                          connection; the first generate does.
    """
    from_pack = not spec
    spec = spec or pack.model
    if not spec:
        raise ValueError(
            "no model: pass --model or set 'model:' in the pack manifest"
        )
    scheme, _, rest = spec.partition(":")
    if scheme == "fake":
        return _fake_model(rest, pack)
    if scheme == "openai":
        # A pack must not be able to aim a credentialed client at a host it names: the
        # request carries the rendered prompt and the ambient API key. `fake:` is exempt
        # because it is local and contained (see _fake_model), which is what lets a pack
        # ship its own offline model for CI.
        if from_pack and not allow_pack_model:
            raise ValueError(
                "this pack's manifest selects a network endpoint (%r). Pass --model to "
                "choose the endpoint yourself, or --allow-pack-model to accept the "
                "pack's choice." % spec
            )
        return _openai_model(rest)
    raise ValueError(
        "unknown model scheme %r in %r (known: %s)"
        % (scheme, spec, ", ".join(MODEL_SCHEMES))
    )


def _openai_model(rest):
    from .backends.openai_compat import OpenAICompatModel

    parts = rest.split("#")
    base_url = parts[0].strip()
    name = parts[1].strip() if len(parts) > 1 else ""
    if not base_url or not name:
        raise ValueError(
            "openai: needs a base url and a model name, "
            "e.g. openai:http://localhost:8000#qwen3-8b"
        )
    options = {"base_url": base_url, "model": name}
    if len(parts) > 2 and parts[2].strip():
        options["grammar_mode"] = parts[2].strip()
    if len(parts) > 3 and parts[3].strip():
        # Reasoning headroom belongs to the backend, not the pack: a pack budgets the
        # answer and stays portable, while this says how much extra THIS model needs to
        # think. See openai_compat.build_payload.
        try:
            options["reasoning_reserve"] = int(parts[3].strip())
        except ValueError:
            raise ValueError(
                "openai: reasoning reserve must be an integer, got %r" % parts[3].strip()
            )
    return OpenAICompatModel(**options)


def _fake_model(path, pack):
    from .model import FakeModel

    if not path:
        raise ValueError("fake: needs a path to a JSON script, e.g. fake:fakes/script.json")
    try:
        full = _resolve_inside(pack.path, path)
    except PackError as exc:
        raise ValueError("fake: %s" % exc)
    if not os.path.isfile(full):
        raise ValueError("fake model script not found: %s" % full)
    with open(full, "r") as handle:
        try:
            return FakeModel(json.load(handle))
        except ValueError as exc:
            raise ValueError("%s is not valid JSON (%s)" % (full, exc))


# --------------------------------------------------------------------------- output


def _report_json(report):
    return {
        "pack": report.pack,
        "passed": report.passed,
        "failed": report.failed,
        "total": report.total,
        "by_node": report.by_node,
        "cases": [
            {
                "name": case.name,
                "passed": case.passed,
                "node": case.node,
                "error": case.error,
                "expected": case.expected,
                "actual": case.actual,
                "mismatches": [
                    {
                        "field": mismatch.field,
                        "expected": mismatch.expected,
                        "actual": mismatch.actual,
                        "node": mismatch.node,
                        "note": mismatch.note,
                    }
                    for mismatch in case.mismatches
                ],
            }
            for case in report.cases
        ],
    }


def _parse_input(text):
    try:
        value = json.loads(text)
    except ValueError as exc:
        raise ValueError("--input is not valid JSON (%s)" % exc)
    if not isinstance(value, dict):
        raise ValueError("--input must be a JSON object, got %s" % type(value).__name__)
    return value


def _usage(parser):
    parser.print_usage(sys.stderr)
    sys.stderr.write("jig: a command is required (run, eval, validate)\n")
    return 2


def _fail(message):
    sys.stderr.write("jig: %s\n" % message)
    return 1
