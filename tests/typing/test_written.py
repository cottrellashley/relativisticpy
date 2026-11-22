r"""Index notation as written: the rule of one term, which the typer keeps as notation is
compiled and a check keeps as it judges a paper, and what surely breaks the rules, read
from the syntax alone."""

import pytest

from relativisticpy.language import parse
from relativisticpy.typing.written import BrokenRule, Repeated, broken_rule, contraction

SPHERICAL = ("t", "r", "theta", "phi")


def statement(text):
    [found] = parse(text).statements
    return found


class TestTheRuleOfATerm:
    def test_an_index_once_is_free_and_twice_once_up_and_once_down_is_summed(self):
        free, summed = contraction([("a", True), ("b", False), ("a", False)])
        assert free == {("b", False)}
        assert summed == {"a"}

    def test_a_term_without_indices_has_none(self):
        assert contraction([]) == (frozenset(), frozenset())

    def test_an_index_three_times_breaks_it(self):
        with pytest.raises(Repeated) as raised:
            contraction([("a", True), ("a", False), ("a", True)])
        assert (raised.value.label, raised.value.positions) == ("a", (True, False, True))

    def test_the_first_index_twice_in_one_position_is_the_one_said(self):
        with pytest.raises(Repeated) as raised:
            contraction([("b", False), ("a", True), ("b", False), ("a", True)])
        assert (raised.value.label, raised.value.positions) == ("b", (False, False))


class TestWhatIsWritten:
    def test_a_broken_rule_says_which_how_to_keep_it_and_where(self):
        found = broken_rule(statement(r"R_{\mu\nu} + g^{\mu\nu} = 0"))
        assert isinstance(found, BrokenRule)
        assert found.message == (
            r"the terms of a sum have different free indices: _{\mu\nu} and ^{\mu\nu}"
        )
        assert found.hint == "every term of a sum has the same free indices"
        assert found.span.text == r"R_{\mu\nu} + g^{\mu\nu}"

    @pytest.mark.parametrize(
        ("text", "message"),
        [
            (r"A_\mu B^\mu C_\mu = 0", r"the index \mu appears 3 times in one term"),
            (r"V_{\mu} W_{\mu} = 0", r"the index \mu appears twice in one term, both times down"),
            (
                "V^a = W^b",
                "the two sides have different free indices: ^{a} on the left, ^{b} on the right",
            ),
        ],
    )
    def test_each_rule_is_said_in_words_of_its_own(self, text, message):
        assert broken_rule(statement(text)).message == message

    @pytest.mark.parametrize(
        "text",
        [
            r"R_{\mu\nu} = 8\pi G T_{\mu\nu}",  # well formed
            "a_i b_i = 1",  # the Euclidean convention
            r"\nabla_\mu V^\mu{}_\mu = 0",  # a derivative, which may be read otherwise
            r"T^{\mu\nu}_{\rm dust} = \rho U^\mu U^\nu",  # words set in roman, not indices
        ],
    )
    def test_what_breaks_no_rule_or_may_be_read_otherwise_is_not_blamed(self, text):
        assert broken_rule(statement(text)) is None

    def test_a_definition_is_not_judged(self):
        assert broken_rule(statement(r"T_{\mu\mu} := 0")) is None

    def test_a_coordinate_names_a_component(self):
        repeated = statement(r"R_{\theta\theta} = 0")
        assert broken_rule(repeated, SPHERICAL) is None  # the component R_{θθ}
        assert broken_rule(repeated) is None  # without coordinates, θ may name one
        found = broken_rule(repeated, ("t", "x"))  # θ is no coordinate, so an index
        assert found.message == r"the index \theta appears twice in one term, both times down"
