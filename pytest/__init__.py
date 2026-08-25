"""Minimal stdlib stand-in for pytest, so `python3 -m pytest -q` works anywhere.

stepmold has a hard zero-dependency rule (stdlib only), but the project's test oracle is
`python3 -m pytest -q`. The whole suite is written as plain `unittest`, so it runs
identically under the real pytest. This package only exists so the oracle command
works on a machine with no pytest installed.

If the real pytest *is* installed, this package steps aside and delegates to it.
See NIGHT-LOG.md (2026-08-24, T0) for the reasoning.
"""

from ._compat import delegate as _delegate

_real = _delegate()

if _real is None:
    from ._shim import fail, main, raises, skip  # noqa: F401

    __all__ = ["fail", "main", "raises", "skip"]
