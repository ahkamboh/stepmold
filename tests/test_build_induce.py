"""Build stages 2 and 3 — the decomposition, and the prompt for each node.

Everything here is scripted against `FakeModel`. The tests that matter are the ones
where the planner answers *plausibly and wrongly*: a field written twice, a field left
out, a gate that reproduces most of the gold endings but not all of them. A compiler
that accepts those produces a pack that fails at 3am instead of at build time.
"""

import json
import unittest

from jig.build.induce import (
    ATTEMPTS,
    MAX_PROMPT_CHARS,
    MAX_WRITES_PER_NODE,
    induce,
    write_prompts,
)
from jig.build.spec import BuildError, FieldSpec, GraphPlan, NodePlan, TaskSpec
from jig.model import FakeModel

# --------------------------------------------------------------------------- fixtures

FIELDS = [
    FieldSpec("category", "string", enum=["billing", "technical", "account", "other"],
              examples=["billing", "technical"]),
    FieldSpec("order_id", "string", optional=True, examples=["A-1001", "B-77"]),
    FieldSpec("amount_usd", "number", optional=True, examples=[49.99, 12.0]),
    FieldSpec("sentiment", "string", enum=["calm", "frustrated", "angry"],
              examples=["frustrated"]),
    FieldSpec("priority", "string", enum=["p0", "p1", "p2", "p3"], examples=["p1", "p0"]),
    FieldSpec("reason", "string", examples=["charged twice for one order"]),
    FieldSpec("queue", "string",
              enum=["billing-ops", "eng-support", "identity", "general"],
              examples=["billing-ops"]),
    FieldSpec("summary", "string", examples=["Customer was charged twice."]),
    FieldSpec("escalate", "boolean", examples=[False, True]),
]


def case(name, ticket, end, **expect):
    full = {
        "category": "billing",
        "order_id": None,
        "amount_usd": None,
        "sentiment": "frustrated",
        "priority": "p1",
        "reason": "one customer is blocked",
        "queue": "billing-ops",
        "summary": "A customer needs help.",
        "escalate": False,
    }
    full.update(expect)
    return {"name": name, "input": {"ticket": ticket}, "expect": full, "end": end}


CASES = [
    case("double charge", "I was charged twice for A-1001.", "done",
         order_id="A-1001", amount_usd=49.99),
    case("cannot log in", "I can't log in.", "done", category="account",
         queue="identity"),
    case("checkout is down", "Checkout is down for everyone.", "escalated",
         category="technical", priority="p0", queue="eng-support", escalate=True,
         sentiment="angry"),
    case("how do refunds work", "How do refunds work?", "done", priority="p3",
         sentiment="calm"),
]


def task(**overrides):
    options = {
        "name": "support_triage",
        "description": "Triage a support ticket and route it to a queue.",
        "inputs": ["ticket"],
        "fields": FIELDS,
        "cases": CASES,
    }
    options.update(overrides)
    return TaskSpec(**options)


def node(name, writes, reads=(), purpose="Decide it.", two_stage=False):
    return {
        "name": name,
        "writes": list(writes),
        "purpose": purpose,
        "two_stage": two_stage,
        "reads": list(reads),
    }


GOOD_NODES = [
    node("classify", ["category"], purpose="Pick the ticket category."),
    node("extract", ["order_id", "amount_usd", "sentiment"], reads=["category"],
         purpose="Pull the stated facts out of the ticket."),
    node("priority", ["priority", "reason"], reads=["category", "sentiment"],
         purpose="Weigh how urgent the ticket is, which is a judgement call.",
         two_stage=True),
    node("emit", ["queue", "summary", "escalate"], reads=["category", "priority"],
         purpose="Route the ticket to a queue."),
]

GOOD_ENDINGS = [
    {"name": "escalated", "when_field": "priority", "when_equals": "p0"},
    {"name": "done", "when_field": "", "when_equals": ""},
]


def plan(nodes=None, endings=None):
    return json.dumps({
        "nodes": GOOD_NODES if nodes is None else nodes,
        "endings": GOOD_ENDINGS if endings is None else endings,
    })


def last_prompt(model):
    return model.calls[-1].prompt


# --------------------------------------------------------------------------- induce


class TestTheProposedPlan(unittest.TestCase):
    """A well-formed answer becomes a GraphPlan with the edges derived from the order."""

    def setUp(self):
        self.model = FakeModel([plan()])
        self.plan = induce(task(), self.model)

    def test_it_is_a_graph_plan_entered_at_the_first_node(self):
        self.assertIsInstance(self.plan, GraphPlan)
        self.assertEqual(self.plan.entry, "classify")
        self.assertEqual([n.name for n in self.plan.nodes],
                         ["classify", "extract", "priority", "emit"])
        self.assertIsInstance(self.plan.nodes[0], NodePlan)

    def test_every_field_is_written_exactly_once(self):
        written = self.plan.written_fields
        self.assertEqual(sorted(written), sorted(spec.name for spec in FIELDS))
        self.assertEqual(len(written), len(set(written)))

    def test_the_nodes_are_chained_in_the_order_given(self):
        chain = [(e["from"], e["to"]) for e in self.plan.edges if "when" not in e]
        self.assertEqual(chain[:3],
                         [("classify", "extract"), ("extract", "priority"),
                          ("priority", "emit")])

    def test_the_conditional_edge_comes_before_the_fallthrough(self):
        # graph.py takes the first matching edge, so an unconditional edge placed first
        # would swallow the branch entirely.
        out = [e for e in self.plan.edges if e["from"] == "emit"]
        self.assertEqual(out[0], {"from": "emit", "to": "escalated",
                                  "when": {"priority": "p0"}})
        self.assertEqual(out[-1], {"from": "emit", "to": "done"})
        self.assertEqual(self.plan.endings, ["escalated", "done"])

    def test_reads_and_two_stage_survive(self):
        self.assertEqual(self.plan.node_named("extract").reads, ["category"])
        self.assertTrue(self.plan.node_named("priority").two_stage)
        self.assertFalse(self.plan.node_named("emit").two_stage)

    def test_the_planner_is_asked_under_a_grammar(self):
        call = self.model.calls[0]
        self.assertEqual(call.grammar["kind"], "json_schema")
        self.assertEqual(call.grammar["schema"]["required"], ["nodes", "endings"])

    def test_the_prompt_carries_the_observed_values_and_the_endings(self):
        prompt = self.model.calls[0].prompt
        self.assertIn("billing, technical, account, other", prompt)
        self.assertIn("sometimes null", prompt)      # order_id is null in most cases
        self.assertIn("escalated (1 case)", prompt)
        self.assertIn("done (3 cases)", prompt)


class TestTheDerivedEdgesAreAGraph(unittest.TestCase):
    """The planner never writes an edge, so these hold for every accepted plan."""

    def setUp(self):
        self.plan = induce(task(), FakeModel([plan()]))
        self.names = [n.name for n in self.plan.nodes]

    def test_every_node_has_an_outgoing_edge_and_no_ending_does(self):
        sources = {edge["from"] for edge in self.plan.edges}
        self.assertEqual(sorted(sources), sorted(self.names))
        self.assertEqual(sources & set(self.plan.endings), set())

    def test_every_edge_points_at_something_that_exists(self):
        known = set(self.names) | set(self.plan.endings)
        for edge in self.plan.edges:
            self.assertIn(edge["to"], known)
            self.assertIn(edge["from"], known)

    def test_every_ending_is_reachable(self):
        reached = {edge["to"] for edge in self.plan.edges}
        for ending in self.plan.endings:
            self.assertIn(ending, reached)


class TestRejectionsThePlannerHasToFix(unittest.TestCase):
    """Each of these is a plausible answer. None of them is silently repaired."""

    def reject_then_accept(self, bad, task_spec=None):
        """Assert `bad` is rejected, and return the re-ask prompt the model saw."""
        model = FakeModel([bad, plan()])
        result = induce(task_spec or task(), model)
        self.assertEqual(model.call_count, 2)
        self.assertIsInstance(result, GraphPlan)
        return last_prompt(model)

    def test_a_field_written_twice(self):
        nodes = [
            node("classify", ["category"]),
            node("extract", ["order_id", "amount_usd", "sentiment"], reads=["category"]),
            node("priority", ["priority", "reason"]),
            # `category` again — the ambiguity assemble would otherwise inherit.
            node("emit", ["queue", "summary", "category"]),
        ]
        reask = self.reject_then_accept(plan(nodes=nodes))
        self.assertIn("'category' is written twice", reask)
        self.assertIn("'classify'", reask)
        self.assertIn("'emit'", reask)

    def test_one_node_that_lists_the_same_field_twice(self):
        nodes = list(GOOD_NODES)
        nodes[1] = node("extract", ["order_id", "order_id", "sentiment"],
                        reads=["category"])
        reask = self.reject_then_accept(plan(nodes=nodes))
        self.assertIn("node 'extract' lists 'order_id' twice", reask)

    def test_a_field_no_node_writes(self):
        nodes = [n for n in GOOD_NODES if n["name"] != "classify"]
        nodes[0] = node("extract", ["order_id", "amount_usd", "sentiment"])
        nodes[1] = node("priority", ["priority", "reason"], reads=["sentiment"])
        reask = self.reject_then_accept(plan(nodes=nodes))
        self.assertIn("no node writes 'category'", reask)

    def test_a_node_writing_a_field_the_task_does_not_have(self):
        nodes = list(GOOD_NODES)
        nodes[0] = node("classify", ["category", "urgency_score"])
        reask = self.reject_then_accept(plan(nodes=nodes))
        self.assertIn("'urgency_score'", reask)
        self.assertIn("not one of the task's fields", reask)

    def test_a_node_that_reads_something_nobody_wrote(self):
        nodes = list(GOOD_NODES)
        nodes[1] = node("extract", ["order_id", "amount_usd", "sentiment"],
                        reads=["customer_tier"])
        reask = self.reject_then_accept(plan(nodes=nodes))
        self.assertIn("'customer_tier'", reask)
        self.assertIn("neither a run input", reask)

    def test_a_node_that_reads_a_field_written_after_it(self):
        nodes = [
            # `classify` wants the sentiment `extract` has not produced yet.
            node("classify", ["category"], reads=["sentiment"]),
            node("extract", ["order_id", "amount_usd", "sentiment"], reads=["category"]),
            node("priority", ["priority", "reason"]),
            node("emit", ["queue", "summary", "escalate"]),
        ]
        reask = self.reject_then_accept(plan(nodes=nodes))
        self.assertIn("does not run before it", reask)
        self.assertIn("'sentiment'", reask)

    def test_one_node_that_writes_everything(self):
        nodes = [node("triage", [spec.name for spec in FIELDS])]
        reask = self.reject_then_accept(plan(nodes=nodes))
        self.assertIn("writes 9 fields", reask)
        self.assertIn("at most %d per node" % MAX_WRITES_PER_NODE, reask)

    def test_a_node_that_writes_nothing(self):
        nodes = list(GOOD_NODES) + [node("review", [])]
        reask = self.reject_then_accept(plan(nodes=nodes))
        self.assertIn("writes no field", reask)

    def test_a_node_name_that_is_not_an_identifier(self):
        nodes = list(GOOD_NODES)
        nodes[0] = node("Classify The Ticket", ["category"])
        reask = self.reject_then_accept(plan(nodes=nodes))
        self.assertIn("not a lowercase identifier", reask)

    def test_two_nodes_with_the_same_name(self):
        nodes = list(GOOD_NODES)
        nodes[3] = node("classify", ["queue", "summary", "escalate"])
        reask = self.reject_then_accept(plan(nodes=nodes))
        self.assertIn("two nodes are both called 'classify'", reask)

    def test_an_empty_purpose(self):
        nodes = list(GOOD_NODES)
        nodes[0] = node("classify", ["category"], purpose="   ")
        reask = self.reject_then_accept(plan(nodes=nodes))
        self.assertIn("empty purpose", reask)

    def test_a_plan_where_everything_thinks(self):
        nodes = [dict(n, two_stage=True) for n in GOOD_NODES]
        reask = self.reject_then_accept(plan(nodes=nodes))
        self.assertIn("4 of 4 nodes are two_stage", reask)

    def test_an_answer_that_is_not_json_at_all(self):
        reask = self.reject_then_accept("I'd start by classifying the ticket, then...")
        self.assertIn("not valid JSON", reask)
        # The rejected text is never quoted back: that is the self-conditioning spiral
        # jig/verify.py exists to prevent, and the compiler holds to its own rule.
        self.assertNotIn("I'd start by classifying", reask)

    def test_an_answer_that_is_json_but_not_the_schema(self):
        reask = self.reject_then_accept(
            json.dumps({"nodes": [dict(GOOD_NODES[0], two_stage="yes")],
                        "endings": GOOD_ENDINGS})
        )
        self.assertIn("expected boolean", reask)


class TestBranching(unittest.TestCase):
    """Endings are checked against the gold cases, not against how sensible they read."""

    def test_a_gate_that_misroutes_a_gold_case_is_rejected(self):
        # Sentiment "angry" is true of the escalated case *and* of nothing else in this
        # fixture, so it looks right — until the whole gold set is replayed.
        endings = [
            {"name": "escalated", "when_field": "sentiment", "when_equals": "calm"},
            {"name": "done", "when_field": "", "when_equals": ""},
        ]
        model = FakeModel([plan(endings=endings), plan()])
        induce(task(), model)
        reask = last_prompt(model)
        # It names the first gold case the gate gets wrong, and there is always one.
        self.assertIn("gold case 'checkout is down'", reask)
        self.assertIn("ends in 'escalated'", reask)
        self.assertIn("route it to 'done'", reask)

    def test_a_gate_on_a_value_no_case_shows(self):
        endings = [
            {"name": "escalated", "when_field": "priority", "when_equals": "p9"},
            {"name": "done", "when_field": "", "when_equals": ""},
        ]
        model = FakeModel([plan(endings=endings), plan()])
        induce(task(), model)
        self.assertIn("a value no gold example has", last_prompt(model))

    def test_a_boolean_gate_is_coerced_before_it_is_compared(self):
        # `when` is `==` against committed state, so "true" and True are different gates.
        endings = [
            {"name": "escalated", "when_field": "escalate", "when_equals": "true"},
            {"name": "done", "when_field": "", "when_equals": ""},
        ]
        result = induce(task(), FakeModel([plan(endings=endings)]))
        edge = [e for e in result.edges if e.get("when")][0]
        self.assertEqual(edge["when"], {"escalate": True})

    def test_a_boolean_gate_that_is_not_a_boolean(self):
        endings = [
            {"name": "escalated", "when_field": "escalate", "when_equals": "yes"},
            {"name": "done", "when_field": "", "when_equals": ""},
        ]
        model = FakeModel([plan(endings=endings), plan()])
        induce(task(), model)
        self.assertIn('must be "true" or "false"', last_prompt(model))

    def test_the_endings_must_be_the_ones_the_cases_show(self):
        endings = [{"name": "done", "when_field": "", "when_equals": ""}]
        model = FakeModel([plan(endings=endings), plan()])
        induce(task(), model)
        self.assertIn("the gold examples end in done, escalated", last_prompt(model))

    def test_a_branch_nobody_asked_for(self):
        # Cases with no `end` key at all: one ending is the only right answer.
        flat = task(cases=[{k: v for k, v in c.items() if k != "end"} for c in CASES])
        one = json.dumps({"nodes": GOOD_NODES,
                          "endings": [{"name": "done", "when_field": "",
                                       "when_equals": ""}]})
        model = FakeModel([plan(), one])
        result = induce(flat, model)
        self.assertIn("the graph needs exactly one", last_prompt(model))
        self.assertEqual(result.endings, ["done"])
        self.assertEqual([e for e in result.edges if e.get("when")], [])

    def test_the_prompt_says_not_to_invent_one(self):
        flat = task(cases=[{k: v for k, v in c.items() if k != "end"} for c in CASES])
        one = json.dumps({"nodes": GOOD_NODES,
                          "endings": [{"name": "done", "when_field": "",
                                       "when_equals": ""}]})
        model = FakeModel([one])
        induce(flat, model)
        self.assertIn("do not invent a branch", model.calls[0].prompt)

    def test_the_last_ending_has_to_be_unconditional(self):
        endings = [
            {"name": "done", "when_field": "priority", "when_equals": "p1"},
            {"name": "escalated", "when_field": "priority", "when_equals": "p0"},
        ]
        model = FakeModel([plan(endings=endings), plan()])
        induce(task(), model)
        self.assertIn("must be the unconditional fallthrough", last_prompt(model))

    def test_an_ending_that_is_also_a_node(self):
        endings = [
            {"name": "emit", "when_field": "priority", "when_equals": "p0"},
            {"name": "done", "when_field": "", "when_equals": ""},
        ]
        model = FakeModel([plan(endings=endings), plan()])
        induce(task(), model)
        self.assertIn("both a node and an ending", last_prompt(model))

    def test_a_gate_on_a_field_no_node_writes(self):
        endings = [
            {"name": "escalated", "when_field": "sla_breach", "when_equals": "true"},
            {"name": "done", "when_field": "", "when_equals": ""},
        ]
        model = FakeModel([plan(endings=endings), plan()])
        induce(task(), model)
        self.assertIn("which no node writes", last_prompt(model))


class TestTheLadderEnds(unittest.TestCase):

    def test_three_bad_answers_raise_rather_than_repair(self):
        bad = plan(nodes=[node("triage", [spec.name for spec in FIELDS])])
        model = FakeModel([bad] * ATTEMPTS)
        with self.assertRaises(BuildError) as caught:
            induce(task(), model)
        self.assertEqual(model.call_count, ATTEMPTS)
        self.assertIn("in %d attempts" % ATTEMPTS, str(caught.exception))
        self.assertIn("at most %d per node" % MAX_WRITES_PER_NODE, str(caught.exception))

    def test_the_first_attempt_carries_no_feedback(self):
        model = FakeModel([plan()])
        induce(task(), model)
        self.assertNotIn("rejected", model.calls[0].prompt)

    def test_a_task_with_no_fields_never_reaches_the_model(self):
        model = FakeModel(["never asked"])
        with self.assertRaises(BuildError) as caught:
            induce(task(fields=[], cases=[]), model)
        self.assertEqual(model.call_count, 0)
        self.assertIn("no output fields", str(caught.exception))

    def test_an_empty_examples_file_still_names_the_task(self):
        with self.assertRaises(BuildError) as caught:
            induce(task(fields=[], cases=[]), FakeModel(["x"]))
        self.assertIn("support_triage", str(caught.exception))


class TestNoGoldCasesAtAll(unittest.TestCase):
    """Fields but an empty examples file: still compilable, and never branched."""

    def setUp(self):
        self.model = FakeModel([plan(endings=[{"name": "done", "when_field": "",
                                               "when_equals": ""}])])
        self.plan = induce(task(cases=[]), self.model)

    def test_the_plan_has_one_ending(self):
        self.assertEqual(self.plan.endings, ["done"])
        self.assertEqual([e for e in self.plan.edges if e.get("when")], [])

    def test_the_planner_is_told_there_are_none(self):
        prompt = self.model.calls[0].prompt
        self.assertIn("observed in 0 gold examples", prompt)
        self.assertIn("do not invent a branch", prompt)

    def test_a_branch_with_nothing_to_replay_it_against_is_still_refused(self):
        model = FakeModel([plan(), plan(endings=[{"name": "done", "when_field": "",
                                                  "when_equals": ""}])])
        induce(task(cases=[]), model)
        self.assertIn("the graph needs exactly one", last_prompt(model))


class TestASingleFieldTask(unittest.TestCase):
    """The small end of the range: one field, one node, one ending, no branch."""

    def setUp(self):
        self.task = TaskSpec(
            name="tiny",
            description="Say whether a message is a question.",
            inputs=["message"],
            fields=[FieldSpec("kind", "string", enum=["question", "complaint"],
                              examples=["question"])],
            cases=[{"name": "a", "input": {"message": "when do you open?"},
                    "expect": {"kind": "question"}}],
        )
        answer = json.dumps({
            "nodes": [node("classify", ["kind"], purpose="Say which kind it is.")],
            "endings": [{"name": "done", "when_field": "", "when_equals": ""}],
        })
        self.plan = induce(self.task, FakeModel([answer]))

    def test_one_node_and_one_edge_into_the_ending(self):
        self.assertEqual(self.plan.entry, "classify")
        self.assertEqual(self.plan.edges, [{"from": "classify", "to": "done"}])

    def test_a_lone_node_may_still_think(self):
        # The "most of the plan is two_stage" rule cannot apply to a one-node plan:
        # there is no other step for the judgement to move into.
        answer = json.dumps({
            "nodes": [node("classify", ["kind"], purpose="A real judgement call.",
                           two_stage=True)],
            "endings": [{"name": "done", "when_field": "", "when_equals": ""}],
        })
        plan_ = induce(self.task, FakeModel([answer]))
        self.assertTrue(plan_.nodes[0].two_stage)


# --------------------------------------------------------------------------- prompts


CLASSIFY_PROMPT = (
    "You are the classify step of a support-ticket triage workflow.\n\n"
    "Choose exactly one category for the ticket:\n"
    "  billing   - charges, refunds, invoices\n"
    "  technical - crashes, bugs, outages\n"
    "  account   - login, passwords, access\n"
    "  other     - anything else\n\n"
    "Ticket: {ticket}\n"
)

EXTRACT_PROMPT = (
    "You are the extract step of a support-ticket triage workflow.\n\n"
    "Pull these out of the ticket. If it does not say, use null.\n"
    "  order_id   - the order reference as written, e.g. A-1001, or null\n"
    "  amount_usd - the dollar amount as a number, e.g. 49.99, or null\n"
    "  sentiment  - calm, frustrated, or angry\n\n"
    "The ticket was classified as: {category}\n\n"
    "Ticket: {ticket}\n"
)

PRIORITY_PROMPT = (
    "You are the priority step of a support-ticket triage workflow.\n\n"
    "Assign a priority and give a one-sentence reason.\n"
    "  p0 - production is down\n"
    "  p1 - one customer is blocked\n"
    "  p2 - broken but survivable\n"
    "  p3 - questions and requests\n\n"
    "Category: {category}\nSentiment: {sentiment}\nNotes: {scratchpad}\n\n"
    "Ticket: {ticket}\n"
)

EMIT_PROMPT = (
    "You are the emit step of a support-ticket triage workflow.\n\n"
    "Route the ticket:\n"
    "  queue    - billing-ops, eng-support, identity, or general\n"
    "  summary  - one plain sentence, at most 20 words\n"
    "  escalate - true only when the priority is p0\n\n"
    "Category: {category}\nPriority: {priority}\n\n"
    "Ticket: {ticket}\n"
)

GOOD_PROMPTS = {
    "'classify'": json.dumps({"prompt": CLASSIFY_PROMPT}),
    "'extract'": json.dumps({"prompt": EXTRACT_PROMPT}),
    "'priority'": json.dumps({"prompt": PRIORITY_PROMPT}),
    "'emit'": json.dumps({"prompt": EMIT_PROMPT}),
}


def good_plan():
    return induce(task(), FakeModel([plan()]))


class TestOnePromptPerNode(unittest.TestCase):

    def setUp(self):
        self.plan = good_plan()
        self.model = FakeModel(dict(GOOD_PROMPTS))
        self.prompts = write_prompts(task(), self.plan, self.model)

    def test_one_entry_per_node_and_nothing_else(self):
        self.assertEqual(sorted(self.prompts),
                         ["classify", "emit", "extract", "priority"])

    def test_each_prompt_is_asked_for_under_a_grammar(self):
        self.assertEqual(self.model.call_count, 4)
        for call in self.model.calls:
            self.assertEqual(call.grammar["schema"]["required"], ["prompt"])

    def test_the_prompts_render_against_the_state_the_node_will_have(self):
        from jig.render import render
        state = {"ticket": "t", "category": "billing", "sentiment": "angry",
                 "priority": "p0", "scratchpad": "notes"}
        for name, text in self.prompts.items():
            render(text, state)  # raises MissingVariable if it names anything else

    def test_the_scribe_is_told_the_fields_and_the_readable_state(self):
        asked = [c.prompt for c in self.model.calls if "'extract'" in c.prompt][0]
        self.assertIn("order_id (string; sometimes null) e.g. A-1001", asked)
        self.assertIn("{category} - written by an earlier step", asked)
        self.assertIn("{ticket} - the run's input", asked)
        # `priority` is written two nodes later; extract must not be offered it.
        self.assertNotIn("{priority}", asked)

    def test_only_a_two_stage_node_is_offered_a_scratchpad(self):
        asked = {c.prompt.split("This step is ")[1].split(":")[0]: c.prompt
                 for c in self.model.calls}
        self.assertIn("{scratchpad}", asked["'priority'"])
        self.assertNotIn("{scratchpad}", asked["'emit'"])


class TestPromptsThatComeBackWrong(unittest.TestCase):

    def reject_then_accept(self, node_key, bad, plan_=None, task_spec=None):
        """Script `bad` first for one node, then the good prompt. Return the re-ask."""
        script = dict(GOOD_PROMPTS)
        script[node_key] = [bad, script[node_key]]
        model = FakeModel(script)
        write_prompts(task_spec or task(), plan_ or good_plan(), model)
        reasks = [c.prompt for c in model.calls if node_key in c.prompt]
        self.assertEqual(len(reasks), 2)
        return reasks[-1]

    def test_a_prompt_that_names_a_variable_the_node_cannot_see(self):
        bad = json.dumps({"prompt": CLASSIFY_PROMPT + "\nPriority: {priority}\n"})
        reask = self.reject_then_accept("'classify'", bad)
        self.assertIn("{priority}", reask)
        self.assertIn("may only name ticket", reask)

    def test_a_prompt_that_ignores_a_field_the_plan_says_it_reads(self):
        # The plan says `extract` reads `category`; a prompt that never names it means
        # the node is not the node the plan described.
        bad = json.dumps({"prompt": EXTRACT_PROMPT.replace(
            "The ticket was classified as: {category}\n", "")})
        reask = self.reject_then_accept("'extract'", bad)
        self.assertIn("reads 'category'", reask)
        self.assertIn("never names {category}", reask)

    def test_a_prompt_that_names_no_state_at_all(self):
        bad = json.dumps({"prompt":
                          "Choose exactly one category: billing, technical, account, "
                          "other. Answer with the category and nothing else at all."})
        reask = self.reject_then_accept("'classify'", bad)
        self.assertIn("names no state value", reask)

    def test_a_prompt_that_never_mentions_a_field_it_has_to_produce(self):
        bad = json.dumps({"prompt": EMIT_PROMPT.replace(
            "  summary  - one plain sentence, at most 20 words\n", "")})
        reask = self.reject_then_accept("'emit'", bad)
        self.assertIn("never mentions 'summary'", reask)

    def test_a_prompt_that_leaves_an_enum_value_out(self):
        bad = json.dumps({"prompt": CLASSIFY_PROMPT.replace(
            "  account   - login, passwords, access\n", "")})
        reask = self.reject_then_accept("'classify'", bad)
        self.assertIn("'account'", reask)
        self.assertIn("belongs in the prompt in full", reask)

    def test_a_prompt_that_asks_for_chain_of_thought(self):
        bad = json.dumps({"prompt": CLASSIFY_PROMPT + "\nThink step by step first.\n"})
        reask = self.reject_then_accept("'classify'", bad)
        self.assertIn("step by step", reask)
        self.assertIn("grammar handles the format", reask)

    def test_a_prompt_that_is_barely_a_prompt(self):
        bad = json.dumps({"prompt": "Classify {ticket}. billing technical account other"})
        reask = self.reject_then_accept("'classify'", bad)
        self.assertIn("characters; state the job", reask)

    def test_a_prompt_that_is_an_essay(self):
        bad = json.dumps({"prompt": CLASSIFY_PROMPT + "x" * MAX_PROMPT_CHARS})
        reask = self.reject_then_accept("'classify'", bad)
        self.assertIn("keep it under %d" % MAX_PROMPT_CHARS, reask)

    def test_a_single_stage_node_may_not_reach_for_the_scratchpad(self):
        bad = json.dumps({"prompt": EMIT_PROMPT + "\nNotes: {scratchpad}\n"})
        reask = self.reject_then_accept("'emit'", bad)
        self.assertIn("only exists on a two_stage node", reask)

    def test_an_answer_that_is_not_json(self):
        reask = self.reject_then_accept("'classify'", "Sure! Here is a good prompt.")
        self.assertIn("not valid JSON", reask)

    def test_the_ladder_ends_in_a_build_error_naming_the_node(self):
        script = dict(GOOD_PROMPTS)
        script["'emit'"] = "not json"
        with self.assertRaises(BuildError) as caught:
            write_prompts(task(), good_plan(), FakeModel(script))
        self.assertIn("prompt for node 'emit'", str(caught.exception))


class TestWhatIsNotRejected(unittest.TestCase):
    """False rejections cost a build. These shapes are all fine."""

    def prompt_for(self, task_spec, node_plan, text):
        plan_ = GraphPlan(entry=node_plan.name, nodes=[node_plan], endings=["done"],
                          edges=[{"from": node_plan.name, "to": "done"}])
        model = FakeModel([json.dumps({"prompt": text})])
        return write_prompts(task_spec, plan_, model)[node_plan.name]

    def test_a_free_text_field_need_not_quote_any_value(self):
        # `summary` has observed examples but no enum: there is no closed set to teach.
        spec = TaskSpec(name="t", description="d", inputs=["ticket"],
                        fields=[FieldSpec("summary", "string",
                                          examples=["Customer was charged twice."])],
                        cases=[{"input": {"ticket": "x"},
                                "expect": {"summary": "y"}}])
        text = ("Write a summary of the ticket below. It must be one plain sentence of "
                "at most twenty words, with no greeting and no advice.\n\n"
                "Ticket: {ticket}\n")
        self.assertIn("summary", self.prompt_for(spec, NodePlan("write", ["summary"],
                                                               "Summarise it."), text))

    def test_a_wide_enum_only_has_to_show_a_few_values(self):
        values = ["a%d" % index for index in range(12)]
        spec = TaskSpec(name="t", description="d", inputs=["ticket"],
                        fields=[FieldSpec("bucket", "string", enum=values,
                                          examples=values[:3])],
                        cases=[{"input": {"ticket": "x"}, "expect": {"bucket": "a0"}}])
        text = ("Put the ticket in exactly one bucket. The buckets are named a0 through "
                "a11 — for example a0, a1 or a2 — and you must answer with one of "
                "them.\n\nTicket: {ticket}\n")
        self.assertTrue(self.prompt_for(spec, NodePlan("bucket", ["bucket"],
                                                       "Bucket it."), text))

    def test_a_field_that_is_null_in_half_the_cases_is_still_just_a_field(self):
        spec = TaskSpec(
            name="t", description="d", inputs=["ticket"],
            fields=[FieldSpec("order_id", "string", optional=True,
                              examples=["A-1001"])],
            cases=[{"input": {"ticket": "a"}, "expect": {"order_id": "A-1001"}},
                   {"input": {"ticket": "b"}, "expect": {"order_id": None}}],
        )
        text = ("Find the order reference in the ticket, exactly as written, for "
                "example A-1001. If the ticket does not give one, answer with null for "
                "order_id.\n\nTicket: {ticket}\n")
        self.assertIn("order_id", self.prompt_for(spec, NodePlan("find", ["order_id"],
                                                                "Find it."), text))

    def test_a_prompt_may_show_a_literal_json_object(self):
        # render leaves `{"a": 1}` alone, so the variable check must too.
        spec = TaskSpec(name="t", description="d", inputs=["ticket"],
                        fields=[FieldSpec("kind", "string", enum=["q", "c"],
                                          examples=["q"])],
                        cases=[{"input": {"ticket": "x"}, "expect": {"kind": "q"}}])
        text = ('Say whether the ticket is a question (q) or a complaint (c). '
                'Answer like {{"kind": "q"}} or {{"kind": "c"}} and nothing more.\n\n'
                'Ticket: {ticket}\n')
        self.assertIn("kind", self.prompt_for(spec, NodePlan("say", ["kind"],
                                                             "Say it."), text))


if __name__ == "__main__":
    unittest.main()
