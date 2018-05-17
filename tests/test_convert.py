import pytest
from ucca import layer0

from semstr import convert

"""Tests convert module correctness and API."""


@pytest.mark.parametrize("converter, lines", (
        (convert.from_conllu,
         ["# sent_id = 120",
          "1	1	_	Word	Word	_	0	root	_	_",
          "2	2	_	Word	Word	_	1	nsubj	_	_",
          ""]
         ),
        (convert.from_sdp,
         ["#120",
          "1	1	_	Word	-	+	_	_",
          "2	2	_	Word	-	-	_	arg0",
          ""]),
        (convert.from_amr,
         ["# ::id 120",
          "# ::snt a b",
          "(a / a-01~e.0",
          "      :ARG0 (p / person :name (n / name :op1 \"b\"~e.1)))",
          ""]),
))
@pytest.mark.parametrize("num_passages", range(3))
@pytest.mark.parametrize("trailing_newlines", range(3))
def test_from(converter, lines, num_passages, trailing_newlines):
    lines = num_passages * lines
    lines[-1:] = trailing_newlines * [""]
    passages = list(converter(lines, "test"))
    assert len(passages) == num_passages, "Number of passages"
    for passage in passages:
        assert 2 == len(passage.layer(layer0.LAYER_ID).all), "Number of terminals"
        assert passage.ID.startswith("120"), "Passage ID"
