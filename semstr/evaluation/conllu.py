from ucca import evaluation
from ucca.constructions import PRIMARY

from ..conversion.conllu import ConlluConverter

EVAL_TYPES = (evaluation.LABELED, evaluation.UNLABELED)


def get_scores(s1, s2, eval_type, verbose=False, units=False):
    converter = ConlluConverter()
    edges = [[e for nodes, _ in converter.build_nodes(s) for n in nodes for e in n.outgoing] for s in (s1, s2)]
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
