import os

from ucca import evaluation
from ucca.constructions import PRIMARY

from ..conversion.conllu import ConlluConverter

EVAL_TYPES = (evaluation.LABELED, evaluation.UNLABELED)


def get_scores(s1, s2, eval_type, verbose=False, units=False):
    converter = ConlluConverter()
    n1, n2 = list(map(list, list(map(converter.build_nodes, (s1, s2)))))
    t1, t2 = list(map(join_tokens, (n1, n2)))
    assert t1 == t2, "Tokens do not match: '%s' != '%s'" % diff(t1, t2)
    edges = [[e for nodes, _ in dep_nodes for n in nodes for e in n.outgoing] for dep_nodes in (n1, n2)]
    for es in edges:
        for e in es:
            e.rel = None if eval_type == evaluation.UNLABELED else e.rel.partition(":")[0]
    g, r = map(set, edges)
    res = evaluation.EvaluatorResults({PRIMARY: evaluation.SummaryStatistics(len(g & r), len(g - r), len(r - g))},
                                      default={PRIMARY.name: PRIMARY})
    if verbose:
        print()
        print("Evaluation type: (" + eval_type + ")")
        if units:
            print("==> Mutual Units:")
            print(g & r)
            print("==> Only in guessed:")
            print(g - r)
            print("==> Only in reference:")
            print(r - g)
        res.print()
    return res


def join_tokens(dep_nodes):
    return "".join((n.parent_multi_word.token.text if n.position == n.parent_multi_word.span[0] else "")
                   if n.parent_multi_word else n.token.text for nodes, _ in dep_nodes for n in nodes[1:])


def diff(s1, s2):
    start, end = [len(os.path.commonprefix((s1[::d], s2[::d]))) for d in (1, -1)]
    return tuple(s[start - 1:-end] for s in (s1, s2))


def evaluate(guessed, ref, converter=None, verbose=False, eval_types=EVAL_TYPES, units=False, **kwargs):
    del kwargs
    if converter is not None:
        guessed = converter(guessed)
        ref = converter(ref)
    return ConlluScores((eval_type, get_scores(guessed, ref, eval_type, verbose, units)) for eval_type in eval_types)


class ConlluScores(evaluation.Scores):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "CoNLL-U"
        self.format = "conllu"
