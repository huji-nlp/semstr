import pytest
from ucca import layer0, textutil

from semstr import convert

"""Tests convert module correctness and API."""

UD2_SIMPLE = ["# sent_id = 120",
              "1	1	!	VERB	VBZ	_	0	root	_	_",
              "2	2	@	NOUN	NN	_	1	nsubj	_	_",
              ""]
UD2_ORPHAN = ["# sent_id = 120",
              "1	1	_	VERB	VBZ	_	0	root	_	_",
              "2	2	_	NOUN	NN	_	_	_	_	_",
              ""]
SDP_SIMPLE = ["#120",
              "1	1	_	Word	-	+	_	_",
              "2	2	_	Word	-	-	_	arg0",
              ""]
AMR_SIMPLE = ["# ::id 120",
              "# ::snt a b",
              "(a / a-01~e.0",
              "      :ARG0 (p / person :name (n / name :op1 \"b\"~e.1)))",
              ""]


@pytest.mark.parametrize("converter, lines", (
        (convert.from_conllu, UD2_SIMPLE),
        (convert.from_conllu, UD2_ORPHAN),
        (convert.from_sdp,    SDP_SIMPLE),
        (convert.from_amr,    AMR_SIMPLE),
))
@pytest.mark.parametrize("num_passages", range(3))
@pytest.mark.parametrize("trailing_newlines", range(3))
def test_from(converter, lines, num_passages, trailing_newlines):
    lines = num_passages * lines
    lines[-1:] = trailing_newlines * [""]
    passages = list(converter(lines, "test", annotate=True, mark_aux=True, metadata=True, wikification=True))
    assert len(passages) == num_passages, "Number of passages"
    for passage in passages:
        assert 2 == len(passage.layer(layer0.LAYER_ID).all), "Number of terminals"
        assert passage.ID.startswith("120"), "Passage ID"


@pytest.mark.parametrize("converter, lines", (
        (convert.from_conllu, UD2_SIMPLE),
))
def test_annotate(converter, lines):
    for passage in converter(lines, "test", annotate=True):
        t1, t2 = passage.layer(layer0.LAYER_ID).all
        assert textutil.Attr.DEP(t1.tok[textutil.Attr.DEP.value]) == "root"
        assert t1.tok[textutil.Attr.HEAD.value] == 0
        assert textutil.Attr.TAG(t1.tok[textutil.Attr.TAG.value]) == "VBZ"
        assert textutil.Attr.POS(t1.tok[textutil.Attr.POS.value]) == "VERB"
        assert textutil.Attr.LEMMA(t1.tok[textutil.Attr.LEMMA.value]) == "!"
        assert textutil.Attr.ORTH(t1.tok[textutil.Attr.ORTH.value]) == "1"

        assert textutil.Attr.DEP(t2.tok[textutil.Attr.DEP.value]) == "nsubj"
        assert t2.tok[textutil.Attr.HEAD.value] == -1
        assert textutil.Attr.TAG(t2.tok[textutil.Attr.TAG.value]) == "NN"
        assert textutil.Attr.POS(t2.tok[textutil.Attr.POS.value]) == "NOUN"
        assert textutil.Attr.LEMMA(t2.tok[textutil.Attr.LEMMA.value]) == "@"
        assert textutil.Attr.ORTH(t2.tok[textutil.Attr.ORTH.value]) == "2"
