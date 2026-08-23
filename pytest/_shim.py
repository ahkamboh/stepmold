"""The tiny subset of the pytest API that jig's suite actually uses."""

import unittest


class Failed(AssertionError):
    """Raised by ``pytest.fail``."""


def fail(msg="failed"):
    raise Failed(msg)


def skip(msg=""):
    raise unittest.SkipTest(msg)


def raises(expected, *args, **kwargs):
    """``with pytest.raises(E) as e:`` — ``e.value`` holds the exception."""
    if args:
        func, rest = args[0], args[1:]
        with _Raises(expected):
            func(*rest, **kwargs)
        return None
    return _Raises(expected)


class _Raises:
    def __init__(self, expected):
        self.expected = expected
        self.value = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            raise AssertionError("DID NOT RAISE %r" % (self.expected,))
        if not issubclass(exc_type, self.expected):
            return False
        self.value = exc
        return True


def main(args=None):
    """Discover and run ``tests/test_*.py``. Returns a process exit code."""
    args = list(args or [])
    quiet = "-q" in args or "--quiet" in args
    targets = [a for a in args if not a.startswith("-")]

    loader = unittest.TestLoader()
    if targets:
        suite = unittest.TestSuite()
        for target in targets:
            suite.addTests(_load_target(loader, target))
    else:
        suite = loader.discover("tests", pattern="test_*.py", top_level_dir=".")

    runner = unittest.TextTestRunner(verbosity=1 if quiet else 2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


def _load_target(loader, target):
    import os

    if os.path.isdir(target):
        return loader.discover(target, pattern="test_*.py", top_level_dir=".")
    path = target.split("::")[0]
    name = os.path.splitext(path)[0].replace(os.sep, ".")
    return loader.loadTestsFromName(name)
