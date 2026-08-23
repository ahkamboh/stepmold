"""T12 — the README's quickstart is executed, not just written.

A quickstart that has drifted from the code is worse than no quickstart, so every
`$ ` line in the README's console blocks is run here, and its documented output is
compared against what actually comes back. The one exception is the block under
"Running the tests", which would recurse into this suite.
"""

import os
import re
import shlex
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README = os.path.join(ROOT, "README.md")

BLOCK = re.compile(r"```console\n(.*?)```", re.DOTALL)
SECTION = re.compile(r"^## (.+)$", re.MULTILINE)


def read():
    with open(README) as handle:
        return handle.read()


def sections():
    """The README split into {heading: body}."""
    text = read()
    found = {}
    marks = list(SECTION.finditer(text))
    for index, mark in enumerate(marks):
        end = marks[index + 1].start() if index + 1 < len(marks) else len(text)
        found[mark.group(1).strip()] = text[mark.end():end]
    return found


def commands(body):
    """Every `$ ...` command in a body's console blocks, with its documented output."""
    found = []
    for block in BLOCK.findall(body):
        current = None
        for line in block.splitlines():
            if line.startswith("$ "):
                current = [line[2:], []]
                found.append(current)
            elif line.startswith("    ") and current and not current[1]:
                current[0] += " " + line.strip()  # a wrapped continuation line
            elif current is not None and line.strip():
                current[1].append(line)
    return [(command, "\n".join(output)) for command, output in found]


def run(command):
    argv = shlex.split(command)
    if argv[0] == "python3":
        argv[0] = sys.executable
    return subprocess.run(
        argv, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        universal_newlines=True,
    )


class TestReadmeExists(unittest.TestCase):
    def test_the_readme_is_there(self):
        self.assertTrue(os.path.isfile(README))

    def test_it_has_the_sections_the_task_asks_for(self):
        headings = sections()
        for required in ("The problem", "Quickstart", "Benchmarks"):
            self.assertIn(required, headings)


def is_deliberate_failure(command):
    """The quickstart shows one command on purpose failing — the CI-gate demonstration."""
    return "wrong.json" in command


def is_illustrative_only(command):
    """Documented for shape; running it would need a live inference server."""
    return "localhost" in command


class TestQuickstartActuallyWorks(unittest.TestCase):
    def test_every_quickstart_command_succeeds(self):
        found = commands(sections()["Quickstart"])
        self.assertTrue(found, "the quickstart has no runnable commands")
        ran = 0
        for command, _ in found:
            if is_illustrative_only(command) or is_deliberate_failure(command):
                continue
            completed = run(command)
            self.assertEqual(
                completed.returncode, 0,
                "%s failed:\n%s%s" % (command, completed.stdout, completed.stderr),
            )
            ran += 1
        self.assertGreaterEqual(ran, 3, "the quickstart stopped exercising the example")

    def test_the_documented_output_is_the_real_output(self):
        """Including the failing one — its printed report is a claim like any other."""
        for command, expected in commands(sections()["Quickstart"]):
            if not expected or is_illustrative_only(command):
                continue
            completed = run(command)
            self.assertEqual(
                completed.stdout.strip(), expected.strip(),
                "%s printed something other than the README claims" % command,
            )

    def test_the_failure_example_really_does_fail(self):
        """The README shows a non-zero exit; prove it, since it is the CI-gate claim."""
        for command, _ in commands(sections()["Quickstart"]):
            if is_deliberate_failure(command):
                self.assertEqual(run(command).returncode, 1)
                return
        self.fail("the README no longer demonstrates a failing eval")

    def test_the_test_command_is_documented_and_runnable_shaped(self):
        found = commands(sections()["Running the tests"])
        self.assertEqual([command for command, _ in found], ["python3 -m pytest -q"])


class TestNoInventedNumbers(unittest.TestCase):
    def test_the_benchmark_table_is_marked_unmeasured(self):
        benchmarks = sections()["Benchmarks"]
        self.assertIn("TODO: measure", benchmarks)

    def test_every_benchmark_cell_is_a_todo(self):
        for line in sections()["Benchmarks"].splitlines():
            if not line.startswith("|") or set(line) <= set("| -"):
                continue
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            for cell in cells[1:]:
                self.assertIn(
                    cell, ("TODO: measure", "jig + small model", "Frontier baseline"),
                    "benchmark cell %r is not a TODO" % cell,
                )

    def test_the_benchmark_section_states_no_numbers_at_all(self):
        prose = sections()["Benchmarks"]
        self.assertFalse(
            re.search(r"\d", prose),
            "the benchmark section contains a digit — nothing here has been measured",
        )
