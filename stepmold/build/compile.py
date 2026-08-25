"""The compile loop: a task description and gold examples in, a working pack out.

    analyze  ->  induce  ->  write_prompts  ->  script  ->  assemble  ->  verify
                    ^                                                        |
                    +--- retry, told exactly which node failed and why ------+

The loop is the reason this is a compiler rather than a code generator. A generator emits
files and hopes; a compiler refuses to emit something that does not satisfy its contract.
Here the contract is the pack's own evalset, and `verify_pack` is what enforces it — so a
pack that scores less than full marks against its gold cases is never returned as a
success, no matter how plausible its prompts look.

Two rules the loop will not break, both of which sound obvious and are the exact corners a
compiler is tempted to cut:

* **The evalset is never edited.** It is the contract. A compiler that adjusts the test
  until it passes has done nothing except launder its own failure.
* **A failed attempt teaches the next one.** `stepmold eval` already names which node caused
  each failure, so the retry says "node `extract` produced sentiment=calm where the gold
  says frustrated" rather than "try again". That per-node attribution is the whole reason
  the loop can converge at all.

The frontier model is used at compile time only, and only by `induce` and `write_prompts`.
`analyze` and `script` are arithmetic over the examples — no model, no GPU, no network —
and between them they are over half the lines of a hand-written pack.
"""

import os
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .analyze import analyze
from .assemble import compile_report, verify_pack, write_pack
from .induce import induce, write_prompts
from .script import check_script, script_for
from .spec import BuildError

__all__ = ["CompileResult", "compile_pack", "load_build_spec"]

DEFAULT_ATTEMPTS = 3


@dataclass
class Attempt:
    """One pass through the loop, kept whether it succeeded or not."""

    number: int
    passed: int = 0
    total: int = 0
    blamed_nodes: List[str] = field(default_factory=list)
    error: Optional[str] = None
    lint: List[str] = field(default_factory=list)

    @property
    def clean(self):
        return self.error is None and self.total > 0 and self.passed == self.total

    def __str__(self):
        if self.error:
            return "attempt %d: %s" % (self.number, self.error)
        line = "attempt %d: %d/%d cases%s" % (
            self.number, self.passed, self.total,
            (" — blamed %s" % ", ".join(self.blamed_nodes)) if self.blamed_nodes else "",
        )
        # check_script names the exact spec key a plan cannot honour — a `rescued` case
        # with no `on_fail` in the plan, say. Saying "11/12" and keeping that to itself
        # would leave the reader to rediscover what the compiler already knows.
        for note in self.lint:
            line += "\n    note: %s" % note
        return line


@dataclass
class CompileResult:
    """What a compile leaves behind, successful or not."""

    ok: bool
    directory: Optional[str]
    attempts: List[Attempt] = field(default_factory=list)
    task: Any = None
    plan: Any = None
    report: Any = None

    def summary(self):
        lines = ["compiled to %s" % self.directory if self.ok
                 else "compile failed after %d attempt(s)" % len(self.attempts)]
        lines.extend("  " + str(a) for a in self.attempts)
        return "\n".join(lines)


def load_build_spec(directory):
    """Read a build directory: `task.md` for the description, `examples.jsonl` for gold.

    Deliberately the same JSON-lines shape as a pack's `evalset.jsonl`, so the examples a
    reader already wrote to test a pack are the examples that compile one — and so the
    compiler's input can be lifted straight out of a pack that already exists.
    """
    task_path = os.path.join(directory, "task.md")
    examples_path = os.path.join(directory, "examples.jsonl")
    for path in (task_path, examples_path):
        if not os.path.isfile(path):
            raise BuildError(
                "%s is missing. A build directory needs task.md (what the workflow does) "
                "and examples.jsonl (gold cases, one JSON object per line, each with "
                "'input' and 'expect')." % path
            )
    with open(task_path) as handle:
        description = handle.read().strip()
    if not description:
        raise BuildError("%s is empty — the planner has nothing to work from" % task_path)

    import json

    cases = []
    with open(examples_path) as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                cases.append(json.loads(line))
            except ValueError as exc:
                raise BuildError(
                    "%s line %d is not valid JSON (%s)" % (examples_path, number, exc)
                )
    if not cases:
        raise BuildError("%s has no cases — there is nothing to compile against"
                         % examples_path)
    return description, cases


def compile_pack(directory, description, cases, model, name=None,
                 attempts=DEFAULT_ATTEMPTS, overwrite=False, on_event=None):
    """Compile a pack into `directory`, or raise `BuildError` explaining why not.

    `model` is only consulted by the planning stages. Each attempt is built in a scratch
    directory and moved into place only once it has scored full marks, so a failed compile
    never leaves a half-written pack where a working one used to be — the same
    verify-before-commit rule the runtime applies to a node, applied to a whole pack.
    """
    say = on_event or (lambda message: None)
    name = name or os.path.basename(os.path.abspath(directory).rstrip(os.sep)) or "pack"

    # Before anything is paid for. _install refuses to clobber an existing pack, and that
    # refusal used to arrive after every attempt had run — inside the try, where it was
    # caught as a failed attempt, re-planned against, and charged for again. A compile that
    # cannot install what it builds has no business building it.
    _check_destination(directory, overwrite)

    say("analyzing %d gold case(s)" % len(cases))
    task = analyze(description, cases, name)
    say("  %d field(s), %d input(s), %d enum(s) inferred"
        % (len(task.fields), len(task.inputs),
           sum(1 for f in task.fields if f.enum)))

    history = []
    feedback = None
    last_plan = None

    for number in range(1, attempts + 1):
        attempt = Attempt(number=number)
        installed = False
        scratch = tempfile.mkdtemp(prefix="stepmold-build-")
        try:
            say("attempt %d: planning" % number)
            plan = induce(task, _guided(model, feedback))
            last_plan = plan
            say("  %d node(s): %s" % (len(plan.nodes),
                                      ", ".join(n.name for n in plan.nodes)))

            prompts = write_prompts(task, plan, model)
            script = script_for(task, plan, prompts=prompts)
            attempt.lint = check_script(script, task, plan, prompts=prompts) or []

            emitted = os.path.join(scratch, "pack")
            write_pack(emitted, task, plan, prompts, script, overwrite=True)

            report = verify_pack(emitted)
            attempt.passed = report.passed
            attempt.total = len(report.cases)
            attempt.blamed_nodes = sorted(
                {case.node for case in report.cases if not case.passed and case.node}
            )
            history.append(attempt)
            say("  %s" % attempt)

            if attempt.clean:
                installed = True
                final = _install(emitted, directory, overwrite)
                return CompileResult(ok=True, directory=final, attempts=history,
                                     task=task, plan=plan, report=report)
            feedback = _feedback(report)
            if attempt.lint:
                feedback = "%s\n%s" % ("\n".join(attempt.lint), feedback)
        except BuildError as exc:
            if installed:
                # The pack compiled and scored full marks; only putting it in place
                # failed. Re-planning cannot fix a filesystem, and telling the planner
                # about it would ask a model to repair a decomposition that was right.
                raise
            attempt.error = str(exc)
            history.append(attempt)
            say("  %s" % attempt)
            feedback = str(exc)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    raise BuildError(
        "could not compile a pack that satisfies its own examples after %d attempt(s).\n%s"
        % (attempts, "\n".join("  " + str(a) for a in history))
    )


def _feedback(report):
    """Turn a failing eval into an instruction the next plan can act on.

    This is where the runtime's per-node blame pays for itself. Without it the retry knows
    only that something was wrong; with it the planner is told which step produced which
    wrong field, which is the difference between a loop that converges and one that
    wanders.
    """
    lines = ["The previous plan scored %d/%d against the gold cases. What went wrong:"
             % (report.passed, len(report.cases))]
    for case in report.cases:
        if case.passed:
            continue
        where = (" (node %s)" % case.node) if case.node else ""
        if case.error:
            lines.append("- case %r%s: %s" % (case.name, where, case.error))
            continue
        for mismatch in case.mismatches[:3]:
            lines.append("- case %r%s: field %s expected %r, got %r"
                         % (case.name, where, mismatch.field,
                            mismatch.expected, mismatch.actual))
    lines.append(
        "Revise the decomposition. A field that came back wrong is usually a node doing "
        "too much at once — split it, or move it after the field it depends on."
    )
    return "\n".join(lines)


def _guided(model, feedback):
    """The planner, with the previous failure prepended to its prompt.

    Wrapping rather than threading a parameter through `induce` keeps the feedback out of
    that stage's signature: `induce` plans from a TaskSpec and knows nothing about loops.
    """
    if not feedback:
        return model

    class _Guided:
        def generate(self, prompt, grammar=None, max_tokens=1024, **kwargs):
            return model.generate(
                feedback + "\n\n" + prompt, grammar=grammar,
                max_tokens=max_tokens, **kwargs
            )

    return _Guided()


def _check_destination(directory, overwrite):
    """Raise if the output directory is occupied and we were not told to replace it.

    Called once before planning as well as from `_install`, because the cost of learning
    this late is measured in frontier-model calls.
    """
    target = os.path.abspath(directory)
    if os.path.exists(target) and os.listdir(target) and not overwrite:
        raise BuildError(
            "%s already exists and is not empty. Pass --overwrite to replace it — a "
            "compile that silently destroyed a hand-tuned pack would be a bad neighbour."
            % target
        )
    return target


def _install(emitted, directory, overwrite):
    """Move a verified pack into place, refusing to clobber unless told to."""
    target = _check_destination(directory, overwrite)
    if os.path.exists(target):
        shutil.rmtree(target)
    parent = os.path.dirname(target)
    if parent:
        os.makedirs(parent, exist_ok=True)
    shutil.move(emitted, target)
    return target
