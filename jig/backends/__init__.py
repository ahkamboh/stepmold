"""Adapters from jig's `Model` protocol to a real inference server.

Nothing in here is exercised by the test suite against a live server — the whole suite
runs offline against `FakeModel`, and these modules are tested with the HTTP layer
mocked. See NIGHT-LOG.md, T11.
"""
