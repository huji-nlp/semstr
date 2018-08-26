import pytest
from ucca import layer0, textutil
from ucca.ioutil import read_files_and_dirs

from semstr import convert

"""Tests convert module correctness and API."""

textutil.models["en"] = "en_core_web_sm"


# def simple():
#     p = core.Passage("120")
#     l0 = layer0.Layer0(p)
#     l1 = layer1.Layer1(p)
#     terms = [l0.add_terminal(text=str(i), punct=False) for i in range(1, 3)]
#     p1 = l1.add_fnode(None, layer1.EdgeTags.Process)
#     a1 = l1.add_fnode(None, layer1.EdgeTags.Participant)
#     p1.add(layer1.EdgeTags.Terminal, terms[0])
#     a1.add(layer1.EdgeTags.Terminal, terms[1])
#     return p


def loaded(filename=None):
    return next(iter(read_files_and_dirs(filename or "test_files/conversion/120.xml")))


def conll_simple(): return "conll", ["# sent_id = 120",
                                     "1	1	_	Word	Word	_	0	ROOT	_	_",
                                     "2	2	_	Word	Word	_	1	A	_	_",
                                     ""]


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


# @pytest.mark.parametrize("create_passage, create_lines", (
#         (simple, conll_simple),
#         (simple, sdp_simple),
# ))
# @pytest.mark.parametrize("num_passages", range(3))
# @pytest.mark.parametrize("trailing_newlines", range(3))
# def test_from_dep(create_passage, create_lines, num_passages, trailing_newlines):
#     p = create_passage()
#     f, lines = create_lines()
#     converter = convert.FROM_FORMAT[f]
#     lines = num_passages * lines
#     lines[-1:] = trailing_newlines * [""]
#     passages = list(converter(lines, "test"))
#     assert len(passages) == num_passages
#     for passage in passages:
#         assert passage.equals(p), "%s: %s != %s" % (converter, str(passage), str(p))


def test_to_conll():
    passage = loaded()
    converted = convert.to_conll(passage)
    with open("test_files/conversion/120.conll", encoding="utf-8") as f:
        # f.write("\n".join(converted))
        assert converted == f.read().splitlines() + [""]
    converted_passage = next(convert.from_conll(converted, passage.ID))
    # ioutil.passage2file(converted_passage, "test_files/conversion/120.conll.xml")
    ref = loaded("test_files/conversion/120.conll.xml")
    assert_converted_equal_ref(passage, converted_passage, ref)
    # Put the same sentence twice and try converting again
    for converted_passage in convert.from_conll(converted * 2, passage.ID):
        ref = loaded("test_files/conversion/120.conll.xml")
    assert_converted_equal_ref(passage, converted_passage, ref)


def test_to_sdp():
    passage = loaded()
    converted = convert.to_sdp(passage)
    with open("test_files/conversion/120.sdp", encoding="utf-8") as f:
        # f.write("\n".join(converted))
        assert converted == f.read().splitlines() + [""]
    converted_passage = next(convert.from_sdp(converted, passage.ID))
    # ioutil.passage2file(converted_passage, "test_files/conversion/120.sdp.xml")
    ref = loaded("test_files/conversion/120.sdp.xml")
    assert_converted_equal_ref(passage, converted_passage, ref)


def test_to_export():
    passage = loaded()
    converted = convert.to_export(passage)
    with open("test_files/conversion/120.export", encoding="utf-8") as f:
        # f.write("\n".join(converted))
        assert converted == f.read().splitlines()
    converted_passage = next(convert.from_export(converted, passage.ID))
    # ioutil.passage2file(converted_passage, "test_files/conversion/120.export.xml")
    ref = loaded("test_files/conversion/120.export.xml")
    assert_converted_equal_ref(passage, converted_passage, ref)


def assert_converted_equal_ref(passage, converted_passage, ref):
    assert converted_passage.equals(ref), "Passage does not match expected:" \
                                          "\npassage:   %s\nconverted: %s\nexpected:  %s" % \
                                          (passage, converted_passage, ref)
