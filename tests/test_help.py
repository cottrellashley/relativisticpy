"""The documentation, offline, as ``relativisticpy help`` prints it."""

import pytest

from relativisticpy import help as documentation
from relativisticpy.errors import UsageError


def test_every_topic_has_a_page_and_a_title():
    topics = dict(documentation.topics())
    assert topics["checking"] == "Checking papers"
    for topic in topics:
        assert documentation.page(topic).startswith("# ")


def test_a_topic_by_another_name():
    assert documentation.resolve("check") == "checking"
    assert documentation.resolve("Checking") == "checking"
    assert documentation.page("check") == documentation.page("checking")
    assert documentation.resolve("nothing") == "nothing"


def test_every_other_name_names_a_page():
    topics = dict(documentation.topics())
    for alias, topic in documentation.ALIASES.items():
        if topic in topics:  # the pages a copy has, as the docs are shipped
            assert documentation.page(alias) == documentation.page(topic)


def test_a_topic_that_does_not_exist_is_told_what_it_may_have_meant():
    with pytest.raises(UsageError, match="there is no help on 'tensers'") as error:
        documentation.page("tensers")
    assert error.value.hint.startswith("did you mean tensors? the topics are ")
    with pytest.raises(UsageError) as error:
        documentation.page("zzz")
    assert error.value.hint.startswith("the topics are ")


def test_without_the_documentation_it_says_where_it_is(monkeypatch):
    monkeypatch.setattr(documentation, "_folder", lambda: None)
    assert [topic for topic, _ in documentation.topics()] == ["reference"]  # always there
    with pytest.raises(UsageError, match="the documentation is not installed") as error:
        documentation.page("checking")
    assert "https://relativisticpy.org" in error.value.hint
