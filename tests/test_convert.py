import pytest
from ucca import layer0, textutil

from semstr import convert

"""Tests convert module correctness and API."""


def ud_simple(): return "conllu", \
                        ["# sent_id = 120",
                         "1	1	!	VERB	VBZ	_	0	root	_	_",
                         "2	2	@	NOUN	NN	_	1	nsubj	_	_",
                         ""]


def ud_orphan(): return "conllu", \
                        ["# sent_id = 120",
                         "1	1	_	VERB	VBZ	_	0	root	_	_",
                         "2	2	_	NOUN	NN	_	_	_	_	_",
                         ""]


def ud_multitoken(): return "conllu", \
                            ["# sent_id = 120",
                             "1-2	1-2	_	_	_	_	_	_	_	_",
                             "1	1	_	VERB	VBZ	_	0	root	_	_",
                             "2	2	_	NOUN	NN	_	1	nsubj	_	_",
                             ""]


def sdp_simple(): return "sdp", \
                         ["#120",
                          "1	1	_	Word	-	+	_	_",
                          "2	2	_	Word	-	-	_	arg0",
                          ""]


def amr_simple(): return "amr", \
                         ["# ::id 120",
                          "# ::snt a b",
                          "(a / a-01~e.0",
                          "      :ARG0 (p / person :name (n / name :op1 \"b\"~e.1)))",
                          ""]


@pytest.mark.parametrize("create", (ud_simple, ud_orphan, ud_multitoken, sdp_simple, amr_simple))
@pytest.mark.parametrize("num_passages", range(3))
@pytest.mark.parametrize("trailing_newlines", range(3))
def test_from(create, num_passages, trailing_newlines):
    f, original = create()
    converter1, converter2 = convert.CONVERTERS[f]
    lines = num_passages * original
    lines[-1:] = trailing_newlines * [""]
    passages = list(converter1(lines, "test", annotate=True, mark_aux=True, metadata=True, wikification=True))
    assert len(passages) == num_passages, "Number of passages"
    for passage in passages:
        assert 2 == len(passage.layer(layer0.LAYER_ID).all), "Number of terminals"
        assert passage.ID.startswith("120"), "Passage ID"
        # assert original == list(converter2(passage))


@pytest.mark.parametrize("create", (ud_simple,))
def test_annotate(create):
    f, lines = create()
    converter1, converter2 = convert.CONVERTERS[f]
    for passage in converter1(lines, "test", annotate=True):
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
