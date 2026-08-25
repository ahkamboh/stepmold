# refund_desk — the pack that acts

Every other pack in `examples/` decides. This one decides and then *does*: it reads a real
order out of a store, judges the claim, and — only when the judgement holds up — issues
the refund. It is the worked example for two capabilities that have tests but had no pack:
the `tool` node, and the confidence gate.

Five working nodes and two endings:

```
classify -> lookup -> assess -> approve -> refund -> done
generate    tool      generate  generate   tool
              |                    |
              +---> needs_human <--+
                    on_fail        on_fail / on_unsure
```

| node | type | what it does | how it can go wrong |
| --- | --- | --- | --- |
| `classify` | generate | refund request, question, or complaint | a question or complaint leaves on an ordinary edge — `needs_human` |
| `lookup` | tool `fetch_order` | reads the order from the host's store | an unknown order id raises; `on_fail: needs_human` |
| `assess` | generate | is the refund justified, given what the store said | burns its ladder; `on_fail` is not set, so this one would abort the run |
| `approve` | generate | the judgement the money moves on | `assert` catches a contradiction, `on_unsure` catches a coin flip; both go to `needs_human` |
| `refund` | tool `issue_refund` | moves the money | `on_fail: needs_human` |
| `done`, `needs_human` | end | the two endings the evalset holds under contract | — |

## The gate comes before the action

`approve` is the gated node: it draws several independent answers and commits only if
enough of them agree, and a node that cannot agree with itself takes `on_unsure` to
`needs_human`. `refund` — the node that moves money — sits *downstream* of that edge, and
there is no other way into it. Read the ordering off the graph rather than off this
sentence: the only edge whose target is `refund` is the one `approve` takes when it is
sure (`tests/test_example_refund.py:test_the_only_edge_into_refund_comes_from_approve`).

(One honest caveat before the argument: the *exit* from the gate, `on_unsure:`, is in this
pack's graph.yaml. The *entrance*, `samples:`/`agree:`, cannot be written there yet — see
"What does not work" below, and `gate_demo.py`, which runs the real gate on this real pack
from Python.)

That ordering is the whole design, and reversing it is the obvious-looking mistake. A gate
placed *after* the action is a review: the money has already moved, the email has already
gone, and being unsure now buys an apology and a reversal — assuming the action was
reversible at all, which the interesting ones are not. A gate placed *before* the action is
a decision that never happened. Disagreement costs three generations and a row in a human
queue; that is the entire bill, because nothing downstream ever ran. A tool call is, in
`stepmold/graph.py`'s own words, "the one thing in a run that retrying cannot undo" — so the
only place a doubt is cheap is upstream of the first side effect. Spend the confidence on
the decision, and let the action be the thing that happens when there is nothing left to
doubt.

## How to run it

The tools are the host's, so the operator hands them over on the command line. There is
deliberately no manifest key for this (`stepmold/cli.py:_add_tools_option`): a pack you did not
write must not be able to choose which code its tool names resolve to.

```
$ python3 -m stepmold eval examples/refund_desk --tools examples/refund_desk/tools.py
refund_desk: 12/12 cases passed
```

```
$ python3 -m stepmold eval examples/refund_desk --tools examples/refund_desk/tools.py --tiers
refund_desk: 12/12 cases passed
refund_desk: 12 cases — 10 auto, 2 escalated, 0 failed
  auto        83.3%   accuracy 100.0% (10/10 correct)
  escalated   16.7%   at approve=1, lookup=1
  failed       0.0%
```

One message at a time. Unlike the other example packs, the scripted model here is keyed on
the order id rather than on case order, so any single case answers itself:

```
$ python3 -m stepmold run examples/refund_desk --tools examples/refund_desk/tools.py \
    --input '{"message":"The mug arrived smashed in three pieces.","order_id":"R-1001"}'
{"approved": true, "justified": true, "kind": "refund", "order_status": "delivered", "refund_amount": 49.99, "refund_id": "RF-R-1001"}

$ python3 -m stepmold run examples/refund_desk --tools examples/refund_desk/tools.py \
    --input '{"message":"The lamp works fine, I just do not want it any more.","order_id":"R-1007"}'
{"approved": false, "justified": false, "kind": "refund", "order_status": "delivered"}

$ python3 -m stepmold run examples/refund_desk --tools examples/refund_desk/tools.py \
    --input '{"message":"I never got order R-9999 and I want a refund.","order_id":"R-9999"}'
{"kind": "refund", "order_id": "R-9999"}
```

The declined run has no `refund_id` and no `refund_amount` in it. That is not a null — an
end node projects the keys state *has* (`graph._project`), and a run that never entered
`refund` never wrote them.

Two scripts in this directory print the things a transcript shows better than prose:

```
$ python3 examples/refund_desk/gate_demo.py     # the gate, three times over
$ python3 examples/refund_desk/once_only.py     # crash mid-refund, resume, count the money
```

## The two tools

`tools.py` is the host half: an in-memory order store, the two functions, and the
`ToolRegistry` that exposes them. The pack names `fetch_order` and `issue_refund` and
contains neither. They are deliberately not the same kind of thing:

| | `fetch_order` | `issue_refund` |
| --- | --- | --- |
| what it does | reads one order | moves money, appends to the ledger |
| `reads` | `order_id` | `order_id`, `order_total` |
| `writes` | `order_total`, `order_status`, `days_since_delivery` | `refund_id`, `refund_amount` |
| `idempotent=` | `True` | `False` |
| in the checkpoint | nothing | the node, the tool, the arguments, the result |
| a resumed run | reads again | replays the recorded result |

The bookkeeping is the price of not being repeatable, and only the tool that is not
repeatable pays it. `python3 examples/refund_desk/once_only.py` kills a run inside the few
microseconds between `issue_refund` returning and the walk leaving its node, then resumes
it:

```
crashed: worker killed after the refund, before the edge
ledger after the crash    [{'order_id': 'R-1001', 'refund_id': 'RF-R-1001', 'amount': 49.99}]
checkpoint next_node      refund
checkpoint tool_calls     [{'args': {'order_id': 'R-1001', 'order_total': 49.99}, 'node': 'refund', 'result': {'refund_amount': 49.99, 'refund_id': 'RF-R-1001'}, 'tool': 'issue_refund'}]

resumed to                done
ledger after the resume   [{'order_id': 'R-1001', 'refund_id': 'RF-R-1001', 'amount': 49.99}]
refunds for R-1001        1
output                    {'kind': 'refund', 'order_status': 'delivered', 'justified': True, 'approved': True, 'refund_id': 'RF-R-1001', 'refund_amount': 49.99}
```

`next_node` still says `refund` because the row is written *before* the commit and before
the edge — everything after that line may die, and the resume lands back on the node and
replays. Committed state could not have answered this: it records what a call returned,
never that it happened. The ledger counts calls, which is why it is the assertion.

## What does not work

The most useful part of this file. Most lines below are pinned by a test in tests/test_example_refund.py, so they cannot rot into a false claim. The exception is noted where it appears.
`tests/test_example_refund.py`, so it cannot rot into a false claim.

**`samples:` and `agree:` cannot be written in graph.yaml.** This is the big one, and it is why the gate below is demonstrated by a script rather than by a line in this pack's graph.
`stepmold.pack._NODE_KEYS` does not carry them and `stepmold.pack.Node` has no field for them, so the
graph in this directory declares only the gate's *exit*:

```
$ cp -r examples/refund_desk /tmp/rd-samples
$ python3 -c "
import pathlib
p = pathlib.Path('/tmp/rd-samples/graph.yaml')
p.write_text(p.read_text().replace(
    '    on_unsure: needs_human',
    '    samples: 3\n    agree: 2\n    on_unsure: needs_human'))
"
$ python3 -m stepmold validate /tmp/rd-samples
stepmold: pack error: graph.yaml: node 'approve' has unknown key(s): agree, samples
```

The gate itself is real and shipped: `stepmold.verify.gate_for` reads the two keys off the node
with `getattr`, `run_node` draws and compares, and `stepmold.graph` routes `Unsure` to
`on_unsure` (which *is* a pack key today). What is missing is one line of the pack format.
So `gate_demo.py` attaches the keys from Python — the same stand-in
`tests/test_verify.py:GatedNode` uses, for the same stated reason — and runs the real gate
against this real pack:

```
AGREED  (draw 1 == draw 2)
  draws       true, true, false
  log         WARNING stepmold.verify node.samples.blind node=approve samples=3 model=FakeModel reason="backend takes no sampling hint, so extra draws repeat the first"
  log         INFO  stepmold.verify node.agreed node=approve agreed=2 of=2 required=2 asked=3 generations=2
  path        classify -> lookup -> assess -> approve -> refund -> done
  end node    done
  ledger      [{'order_id': 'R-1001', 'refund_id': 'RF-R-1001', 'amount': 49.99}]
  refunded?   True

UNSURE  (a real disagreement)
  draws       true, false, true
  log         WARNING stepmold.verify node.samples.blind node=approve samples=3 model=FakeModel reason="backend takes no sampling hint, so extra draws repeat the first"
  log         WARNING stepmold.verify node.unsure node=approve agreed=1 of=3 required=2 asked=3 distinct=3 generations=3
  log         INFO  stepmold.graph edge.on_unsure run_id=d6b459e49e7e4565964f3ab648474dc9 node=approve to=needs_human
  path        classify -> lookup -> assess -> approve -> needs_human
  end node    needs_human
  ledger      []
  refunded?   False

UNSURE  (same decision, three wordings)
  draws       true, true, true
  log         WARNING stepmold.verify node.samples.blind node=approve samples=3 model=FakeModel reason="backend takes no sampling hint, so extra draws repeat the first"
  log         WARNING stepmold.verify node.unsure node=approve agreed=1 of=3 required=2 asked=3 distinct=3 generations=3
  log         INFO  stepmold.graph edge.on_unsure run_id=532fed4bba0642efa98b7c0da5e08c22 node=approve to=needs_human
  path        classify -> lookup -> assess -> approve -> needs_human
  end node    needs_human
  ledger      []
  refunded?   False
```

Because the entrance is Python-only, `evalset.jsonl` has **no case that ends at
`needs_human` by way of `on_unsure`** — the four that land there get there by an ordinary
edge (a question, a complaint) or by `on_fail` (an unknown order, a burnt ladder). Read
that gap as a limit of the pack format, not of the gate.

**All three draws said `approved: true` and the gate still refused.** Look at the third
attempt above: `distinct=3`, and no refund. Agreement is on the *whole object the node
would commit* (`verify._canonical`), not on the field you care about, and the `rationale`
string differed three ways. This is the trap to know about before putting a gate on a wide
node. A gate that should be about a decision needs a node that commits the decision alone —
narrow the grammar, or split the node.

**`node.samples.blind` fires on every sampled run here.** `FakeModel.generate` has no
`sampling` keyword, so `codegen.accepts_sampling` is false and stepmold warns that the extra
draws would be the first draw repeated. It is right to warn: against a real greedy backend
they would be. The draws in `gate_demo.py` differ only because the script says they do —
scripted disagreement, not measured disagreement. Nothing here is evidence about how often
a real model agrees with itself.

**Without `--tools` the pack does not run at all**, and it fails ten cases rather than
saying so once:

```
$ python3 -m stepmold eval examples/refund_desk
refund_desk: 2/12 cases passed
  FAIL damaged on arrival -> refunded [lookup]
    error: ToolsNotAvailable: node 'lookup' is a tool node, and this run was given no tools: this pack needs tools; pass tools= to run()
  ...
  failures by node: lookup=10
```

Exit status 1. The two that "pass" are the question and the complaint, which never reach a
tool node. `--tools` exists on `run` and `eval` only.

**`stepmold validate` now takes `--tools`**, so the load-time tool checks — is this name
registered, and can its `reads` be satisfied by this graph — never run from the command
line. `stepmold.pack.check_tools` is reachable only as `load_pack(path, tools=registry)` from
Python. A clean `stepmold validate` on a pack with tool nodes says nothing about its wiring.

**Scoring this pack moves money.** `stepmold eval` is not a dry run: five of the twelve cases
reach `refund`, and a scored run leaves five entries in the ledger. The store is in memory
and dies with the process, which is the only reason this is safe to ship.

**`issue_refund` raises `AlreadyRefunded` on a repeat.** Nothing in the evalset or the
tests ever triggers it — that is the point of it. It is the alarm that would go off if
stepmold's exactly-once record stopped working, instead of a quiet second row in the ledger.

**The model is a script, not a model.** `fakes/script.json` is keyed on the footer line
each prompt ends with (`Refund desk / approve / order R-1001`), which is what lets a single
`stepmold run` answer its own case instead of the first one. The footer means nothing to a real
backend. Point `--model` at one to run real messages, and expect the numbers above to
change.

## Files

| file | what it is |
| --- | --- |
| `manifest.yaml` | name, entry, the scripted model, and the two inputs |
| `graph.yaml` | five working nodes, two endings, and the comment explaining the missing gate keys |
| `prompts/*.txt` | one per generate node; each ends with the footer the script keys on |
| `grammars/*.json` | the schema each generate node is verified against |
| `fakes/script.json` | 30 entries keyed by node and order id; 29 hold one response, and `approve / order R-1010` holds the three that burn a ladder |
| `evalset.jsonl` | 12 cases: 5 refunded, 3 declined, 4 to a human, 2 of them `rescued: true` |
| `tools.py` | the host: the order store, the two actions, and the `ToolRegistry` |
| `gate_demo.py` | the confidence gate, three times, with the ledger printed |
| `once_only.py` | crash mid-refund, resume, and count the money |
