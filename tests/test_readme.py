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



# Numbers spell out, and a comparison needs no number at all — a digit check alone is
# not a guard. These are the shapes a fabricated benchmark takes.
NUMBER_WORDS = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty "
    "sixty seventy eighty ninety hundred thousand million billion dozen "
    "half twice thrice double triple quadruple percent"
).split()

CLAIM_PHRASES = (
    "faster", "slower", "cheaper", "costlier", "outperform", "beats", "beat the",
    "we measured", "we ran", "measured at", "as fast as", "orders of magnitude",
    "on par with", "matched the", "matches the", "scored", "achiev", "up to",
    "improved", "improvement", "speedup", "in our runs", "in practice we",
)

DIGIT_OR_PERCENT = re.compile(r"\d|%")
NUMBER_WORD = re.compile(
    r"\b(?:%s)(?:s|fold)?\b" % "|".join(NUMBER_WORDS), re.IGNORECASE
)
CLAIM = re.compile("|".join(re.escape(phrase) for phrase in CLAIM_PHRASES), re.IGNORECASE)


def _benchmarks_document():
    """The benchmark table lives in docs/BENCHMARKS.md, not the README.

    The guard follows it there. What is being prevented is a number presented as measured
    with no way to reproduce it — not the presence of a table in any particular file.
    """
    path = os.path.join(os.path.dirname(README), "docs", "BENCHMARKS.md")
    with open(path) as handle:
        return handle.read()


def unmeasured_claims(text):
    """Every fragment of `text` that reads as a measured result. Empty means honest."""
    found = [match.group(0) for match in DIGIT_OR_PERCENT.finditer(text)]
    found += [match.group(0) for match in NUMBER_WORD.finditer(text)]
    found += [match.group(0) for match in CLAIM.finditer(text)]
    return found


class TestReadmeExists(unittest.TestCase):
    def test_the_readme_is_there(self):
        self.assertTrue(os.path.isfile(README))

    def test_it_has_the_sections_the_task_asks_for(self):
        headings = sections()
        for required in ("Why", "Quickstart", "Results", "Documentation"):
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
    def test_every_benchmark_number_carries_its_provenance(self):
        """The guard changed shape when the first numbers were measured.

        It used to ban every digit, because nothing had been run. Numbers now exist, so
        banning them would be wrong — but an unattributed number is still the thing to
        prevent. A reader must be able to re-run any figure here, so the section has to
        name the date, the endpoint, the model, and the exact command.
        """
        benchmarks = _benchmarks_document()
        for required in ("Measured", "2026-08-24", "api.cerebras.ai",
                         "gpt-oss-120b", "python3 -m jig eval"):
            self.assertIn(
                required, benchmarks,
                "benchmark section states numbers without %r — no provenance" % required,
            )

    def test_the_benchmark_section_says_what_is_not_established(self):
        """A measured number invites an unearned conclusion. Say the limit out loud."""
        benchmarks = _benchmarks_document()
        self.assertIn("do not establish", benchmarks.lower())

    def test_no_section_claims_a_result_without_provenance(self):
        """Only Benchmarks may state results, because only it cites how to reproduce them.

        Elsewhere the thing to catch is not a number — "two halves", "Three rules" and
        "§3" are ordinary prose — but the LANGUAGE of a measured win: faster, cheaper,
        outperforms, we measured. That is what a fabricated benchmark reads like when it
        escapes the table.
        """
        for name, body in sections().items():
            if name in ("Benchmarks", "Results"):
                continue
            self.assertEqual(
                [match.group(0) for match in CLAIM.finditer(body)], [],
                "section %r claims a measured result but cites no provenance; "
                "move it to Benchmarks with the command that reproduces it" % name,
            )


class TestTheGuardCatchesFabrications(unittest.TestCase):
    """The guard is only worth having if it survives someone spelling the number out.

    A digit check alone is evaded by writing "forty-eight of fifty" or by dropping the
    number entirely and keeping the comparison ("cheaper than the frontier baseline").
    Both are still claims about a measurement nobody has run, which is exactly what this
    section must not contain.
    """

    def test_a_digit_is_caught(self):
        self.assertTrue(unmeasured_claims("jig scores 48/50 on the gold cases."))

    def test_a_spelled_out_number_is_caught(self):
        self.assertTrue(
            unmeasured_claims(
                "jig plus a small model passed forty-eight of the fifty gold cases."
            )
        )

    def test_a_comparison_with_no_number_at_all_is_caught(self):
        self.assertTrue(
            unmeasured_claims("Running it this way is cheaper than the frontier baseline.")
        )

    def test_a_multiplier_written_as_a_word_is_caught(self):
        self.assertTrue(unmeasured_claims("Throughput improved tenfold on the target GPU."))

    def test_a_percentage_sign_is_caught(self):
        self.assertTrue(unmeasured_claims("Pass rate: 96%."))

    def test_the_honest_disclaimer_is_not_caught(self):
        """Over-strictness would push the next author into deleting the disclaimer."""
        self.assertEqual(
            unmeasured_claims(
                "TODO: measure. Every number that belongs here is a number nobody has "
                "measured yet, and someone still has to run it on a GPU."
            ),
            [],
        )


def flatten(suite):
    """Every TestCase in a suite tree, in discovery order."""
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            for test in flatten(item):
                yield test
        else:
            yield item


class TestTheOracleCollectsTheWholeSuite(unittest.TestCase):
    """`python3 -m pytest -q` is the only oracle this project has, and a green run only
    means something if it ran everything.

    Discovery is by filename pattern, so a file that stops matching — renamed, moved,
    or made un-importable — takes its tests out of the count without turning the run
    red. That is the failure mode this guards: not a wrong answer, a missing question.
    """

    # The suite must not fall below the count the project commits to (see the repo's
    # test requirement). It is a floor, not a target: adding tests raises it.
    MINIMUM_TESTS = 360

    def discovered(self):
        loader = unittest.TestLoader()
        suite = loader.discover(
            os.path.join(ROOT, "tests"), pattern="test_*.py", top_level_dir=ROOT
        )
        return list(flatten(suite))

    def files_on_disk(self):
        names = os.listdir(os.path.join(ROOT, "tests"))
        return {
            "tests." + os.path.splitext(name)[0]
            for name in names
            if name.startswith("test_") and name.endswith(".py")
        }

    def test_every_test_file_on_disk_is_collected(self):
        collected = {type(test).__module__ for test in self.discovered()}
        missing = sorted(self.files_on_disk() - collected)
        self.assertEqual(
            missing, [],
            "these test files exist but contribute no tests to the run: %s" % missing,
        )

    def test_no_test_file_failed_to_import(self):
        """An import error is collected as a placeholder; it must not pass unnoticed."""
        broken = [
            test.id() for test in self.discovered()
            if type(test).__name__ == "_FailedTest"
        ]
        self.assertEqual(broken, [], "these modules could not be imported: %s" % broken)

    def test_the_suite_has_not_shrunk(self):
        count = len(self.discovered())
        self.assertGreaterEqual(
            count, self.MINIMUM_TESTS,
            "discovery found %d tests, below the floor of %d — something stopped being "
            "collected" % (count, self.MINIMUM_TESTS),
        )
