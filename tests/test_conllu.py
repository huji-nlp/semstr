"""Testing code for the conllu format, unit-testing only."""

import pytest
from ucca.convert import split2sentences

from semstr.convert import from_conllu, to_conllu
from semstr.evaluation.conllu import evaluate


def test_convert():
    """Test that converting an Universal Dependencies tree to UCCA and back retains perfect LAS F1"""
    for passage, ref, _ in read_test_conllu():
        convert_and_evaluate(passage, ref)


def test_split():
    """Test that splitting a single-sentence Universal Dependencies tree converted to UCCA returns the same tree"""
    for passage, ref, _ in read_test_conllu():
        sentences = split2sentences(passage)
        assert len(sentences) == 1, "Should be one sentence: %s" % passage
        convert_and_evaluate(sentences[0], ref)


def test_evaluate():
    """Test that comparing an Universal Dependencies graph against itself returns perfect LAS F1"""
    for _, ref, conllu_id in read_test_conllu():
        assert evaluate(ref, ref).average_f1() == pytest.approx(1, 0.1)


def convert_and_evaluate(passage, ref):
    converted = to_conllu(passage)
    assert evaluate(converted, ref).average_f1() == pytest.approx(1, 0.1), format_lines(converted)


def read_test_conllu():
    with open("test_files/UD_English.conllu") as f:
        yield from from_conllu(f, return_original=True)
    with open("test_files/UD_German.conllu") as f:
        yield from from_conllu(f, return_original=True)


def format_lines(lines):
    lines = [l.split("\t") for l in lines if l and not l.startswith("#")]
    width = [max(map(len, l)) for l in zip(*lines)]
    return "\n" + "\n".join(" ".join("%-*s" % (w, t) for t, w in zip(l, width)) for l in lines)
