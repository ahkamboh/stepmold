"""`python3 -m jig` — run a pack, score it, or check it.

Three commands, because a pack has exactly three things you do to it:

    jig validate <pack>                     is this pack well-formed?
    jig run <pack> --input '<json>'         execute it once
    jig eval <pack>                         score it against its contract

Exit codes are the contract with CI: **0** success, **1** the thing failed (invalid pack,
failed run, evalset not fully passed), **2** you called it wrong (argparse's own code).
`jig eval` exiting 1 on a single failed case is the point — that is what makes an evalset
a gate rather than a report.

argparse only, per the stdlib rule. PLAN.md §7 names Typer + Rich; both are dependencies,
so the help text here is plain and the report text is plain.
"""

import argparse
import json
import os
import sys

from . import __version__
from .errors import JigError
from .eval import evaluate
from .grammar import ValidationError
from .pack import PackError, load_pack

__all__ = ["main", "resolve_model"]

MODEL_SCHEMES = ("fake",)


def main(argv=None):
    """Run the CLI. Returns an exit code rather than calling sys.exit, so it is testable."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except PackError as exc:
        return _fail("pack error: %s" % exc)
    except JigError as exc:
        return _fail("%s: %s" % (type(exc).__name__, exc))
    except (ValidationError, ValueError) as exc:
        return _fail(str(exc))


def build_parser():
    parser = argparse.ArgumentParser(
        prog="jig",
        description="Compile your agent once with a frontier model. "
                    "Run it forever on a small one.",
    )
    parser.add_argument("--version", action="version", version="jig %s" % __version__)
    commands = parser.add_subparsers(dest="command")

    validate = commands.add_parser("validate", help="check that a pack is well-formed")
    validate.add_argument("pack", help="path to the pack directory")
    validate.set_defaults(handler=command_validate)

    run = commands.add_parser("run", help="execute a pack once")
    run.add_argument("pack", help="path to the pack directory")
    run.add_argument("--input", default="{}", help="run inputs as a JSON object")
    run.add_argument("--model", help="model spec (default: the pack manifest's)")
    run.add_argument("--run-id", dest="run_id", help="name this run")
    run.add_argument("--store", help="SQLite file to checkpoint into")
    run.add_argument("--resume", help="continue a previous run id (needs --store)")
    run.add_argument("--state", action="store_true",
                     help="print the whole final state, not the end node's projection")
    run.set_defaults(handler=command_run)

    evaluate_command = commands.add_parser(
        "eval", help="score a pack against its evalset (exit 1 if any case fails)"
    )
    evaluate_command.add_argument("pack", help="path to the pack directory")
    evaluate_command.add_argument("--model", help="model spec (default: the manifest's)")
    evaluate_command.add_argument("--json", action="store_true",
                                  help="emit the report as JSON")
    evaluate_command.set_defaults(handler=command_eval)

    parser.set_defaults(handler=lambda args: _usage(parser))
    return parser


# ------------------------------------------------------------------------ commands


def command_validate(args):
    pack = load_pack(args.pack)
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
    if args.resume and not args.store:
        return _fail("--resume needs --store: checkpoints live in the store")

    store = Store(args.store) if args.store else None
    try:
        if args.resume:
            result = resume(pack, resolve_model(args.model, pack), args.resume, store)
        else:
            result = run_pack(
                pack,
                resolve_model(args.model, pack),
                _parse_input(args.input),
                run_id=args.run_id,
                store=store,
            )
    finally:
        if store is not None:
            store.close()

    print(json.dumps(result.state if args.state else result.output, sort_keys=True))
    return 0


def command_eval(args):
    pack = load_pack(args.pack)
    report = evaluate(pack, resolve_model(args.model, pack))
    print(json.dumps(_report_json(report), sort_keys=True) if args.json
          else report.summary())
    return 0 if report.passed_all else 1


# -------------------------------------------------------------------------- models


def resolve_model(spec, pack):
    """Turn a model spec into a `Model`.

    `fake:<path>` loads a scripted `FakeModel` from JSON (a list of responses, or an
    object keyed by prompt substring); a relative path is resolved inside the pack, which
    is what lets a pack ship an offline model for CI. T11 adds the `openai:` scheme for a
    real backend.
    """
    spec = spec or pack.model
    if not spec:
        raise ValueError(
            "no model: pass --model or set 'model:' in the pack manifest"
        )
    scheme, _, rest = spec.partition(":")
    if scheme == "fake":
        return _fake_model(rest, pack)
    raise ValueError(
        "unknown model scheme %r in %r (known: %s)"
        % (scheme, spec, ", ".join(MODEL_SCHEMES))
    )


def _fake_model(path, pack):
    from .model import FakeModel

    if not path:
        raise ValueError("fake: needs a path to a JSON script, e.g. fake:fakes/script.json")
    full = path if os.path.isabs(path) else os.path.join(pack.path, path)
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
