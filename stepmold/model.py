"""The model interface stepmold executes against, and the FakeModel every test uses.

stepmold never calls a model directly from the graph walker; it calls whatever satisfies
`Model`. That is what keeps the entire test suite offline — `FakeModel` is a drop-in.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Sequence, Union, runtime_checkable


class ModelExhausted(RuntimeError):
    """A FakeModel was asked for a response its script does not contain."""


@runtime_checkable
class Model(Protocol):
    """Anything stepmold can generate with."""

    def generate(
        self,
        prompt: str,
        grammar: Optional[dict] = None,
        max_tokens: int = 512,
    ) -> str:
        """Return a completion for `prompt`, shaped by `grammar` when one is given."""
        ...


@dataclass(frozen=True)
class Call:
    """One recorded generation, so tests can assert on how the model was used."""

    prompt: str
    grammar: Optional[dict]
    max_tokens: int


Script = Union[Sequence[str], Dict[str, Union[str, Sequence[str]]]]


@dataclass
class FakeModel:
    """A scripted stand-in for a real model.

    Two modes:

    * **ordered** — `FakeModel(["a", "b"])` returns the responses in order and raises
      `ModelExhausted` once past the end.
    * **keyed** — `FakeModel({"classify": "billing"})` returns the response whose key is
      a substring of the prompt. The *longest* matching key wins, so a specific key can
      override a general one. A key's value may be a list, which is consumed in order
      (this is how a multi-case evalset scripts one node across many inputs).
    """

    scripted: Script
    calls: List[Call] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.scripted, dict):
            if not self.scripted:
                raise ValueError("FakeModel needs at least one scripted response")
            self._keyed_script = {
                k: (v if isinstance(v, str) else list(v))
                for k, v in self.scripted.items()
            }
        elif isinstance(self.scripted, (list, tuple)):
            if not self.scripted:
                raise ValueError("FakeModel needs at least one scripted response")
            self._ordered = list(self.scripted)
        else:
            raise TypeError(
                "FakeModel script must be a list of responses or a dict keyed by "
                "prompt substring, not %s" % type(self.scripted).__name__
            )

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def generate(
        self,
        prompt: str,
        grammar: Optional[dict] = None,
        max_tokens: int = 512,
    ) -> str:
        self.calls.append(Call(prompt=prompt, grammar=grammar, max_tokens=max_tokens))
        if isinstance(self.scripted, dict):
            return self._keyed(prompt)
        return self._next_ordered()

    def _next_ordered(self) -> str:
        index = len(self.calls) - 1
        if index >= len(self._ordered):
            raise ModelExhausted(
                "FakeModel script has %d responses; call %d has nothing to return"
                % (len(self._ordered), len(self.calls))
            )
        return self._ordered[index]

    def _keyed(self, prompt: str) -> str:
        matches = [k for k in self._keyed_script if k in prompt]
        if not matches:
            raise ModelExhausted(
                "FakeModel has no scripted response matching prompt: %r" % prompt
            )
        key = max(matches, key=len)
        responses = self._keyed_script[key]
        if isinstance(responses, str):
            return responses  # a plain-string key answers every matching prompt
        if not responses:
            raise ModelExhausted(
                "FakeModel ran out of scripted responses for key %r" % key
            )
        return responses.pop(0)
