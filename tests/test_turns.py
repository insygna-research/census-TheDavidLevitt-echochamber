"""Turn loop: native tool transport, sentinel fallback, budgets."""

from helpers import FakeProvider, FakeSearchTool, tool_call_response

from echochamber.core.agent import Agent, Role
from echochamber.core.turns import run_agent_turn
from echochamber.providers import Message


def make_agent(provider):
    return Agent(name="Tester", role=Role.PROSECUTION, provider=provider)


MESSAGES = [Message(role="user", content="Present your argument.")]


def test_plain_turn_without_search():
    provider = FakeProvider(["my argument"])
    text = run_agent_turn(make_agent(provider), MESSAGES)
    assert text == "my argument"
    assert len(provider.calls) == 1


def test_native_search_feeds_results_back():
    provider = FakeProvider(
        [
            tool_call_response("web_search", {"query": "pizza history"}),
            "final argument citing results",
        ],
        supports_tools=True,
    )
    search = FakeSearchTool()

    text = run_agent_turn(
        make_agent(provider), MESSAGES, search_tool=search, max_searches=2
    )

    assert text == "final argument citing results"
    assert search.queries == ["pizza history"]
    # First call offered the web_search tool
    assert provider.calls[0]["tools"] and provider.calls[0]["tools"][0].name == "web_search"
    # Second call included the tool result in the conversation
    roles = [m.role for m in provider.calls[1]["messages"]]
    assert "tool" in roles


def test_native_search_budget_enforced():
    provider = FakeProvider(
        [
            tool_call_response("web_search", {"query": "q1"}, call_id="c1"),
            tool_call_response("web_search", {"query": "q2"}, call_id="c2"),
            "done",
        ],
        supports_tools=True,
    )
    search = FakeSearchTool()

    text = run_agent_turn(
        make_agent(provider), MESSAGES, search_tool=search, max_searches=1
    )

    assert text == "done"
    assert search.queries == ["q1"]  # second search denied
    # After the budget is spent, the tool is no longer offered
    assert provider.calls[2]["tools"] is None


def test_sentinel_search_feeds_results_back():
    provider = FakeProvider(
        ["I need data [SEARCH: remote work stats]", "updated argument with data"],
        supports_tools=False,
    )
    search = FakeSearchTool()

    text = run_agent_turn(
        make_agent(provider), MESSAGES, search_tool=search, max_searches=2
    )

    assert text == "updated argument with data"
    assert search.queries == ["remote work stats"]
    # The follow-up prompt contained the search results
    followup = provider.calls[1]["messages"][-1].content
    assert "Canned result" in followup


def test_sentinel_without_tags_returns_first_response():
    provider = FakeProvider(["no searches needed"], supports_tools=False)
    search = FakeSearchTool()
    text = run_agent_turn(
        make_agent(provider), MESSAGES, search_tool=search, max_searches=2
    )
    assert text == "no searches needed"
    assert search.queries == []
    assert len(provider.calls) == 1


def test_native_failure_falls_back_to_sentinel():
    class ToolAllergicProvider(FakeProvider):
        def complete(self, messages, tools=None, **kwargs):
            if tools:
                raise RuntimeError("this endpoint rejects tools")
            return super().complete(messages, **kwargs)

    provider = ToolAllergicProvider(["plain answer"], supports_tools=True)
    text = run_agent_turn(
        make_agent(provider), MESSAGES, search_tool=FakeSearchTool(), max_searches=1
    )
    assert text == "plain answer"
