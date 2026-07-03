"""Search tag extraction and query normalization."""

from echochamber.tools.search import (
    WEB_SEARCH_TOOL,
    extract_search_queries,
    normalize_query,
)


def test_extract_single_query():
    assert extract_search_queries("text [SEARCH: foo bar] more") == ["foo bar"]


def test_extract_multiple_queries():
    text = "[SEARCH: first] middle [search: second]"
    assert extract_search_queries(text) == ["first", "second"]


def test_extract_no_queries():
    assert extract_search_queries("no tags here") == []


def test_normalize_strips_accents():
    assert normalize_query("réticence dolosive") == "reticence dolosive"


def test_web_search_tool_schema():
    assert WEB_SEARCH_TOOL.name == "web_search"
    assert WEB_SEARCH_TOOL.parameters["required"] == ["query"]
