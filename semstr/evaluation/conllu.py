import os
from collections import OrderedDict

from ucca.constructions import PRIMARY, DEFAULT, Candidate, create_passage_yields
from ucca.evaluation import LABELED, UNLABELED, Scores, Evaluator, EvaluatorResults, SummaryStatistics, ALL_EDGES

from ..conversion.conllu import ConlluConverter

EVAL_TYPES = (LABELED, UNLABELED)


class ConlluEvaluator(Evaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, fscore=True, errors=False, **kwargs)
        self.reference_yield_tags = None

    def get_scores(self, s1, s2, eval_type, r=None, verbose=False, units=False):
        """
        :param s1: sentence to compare
        :param s2: reference sentence
        :param eval_type: evaluation type to use, out of EVAL_TYPES
        1. UNLABELED: disregard dependency relation labels.
        2. LABELED: also requires relation match
        :param r: reference passage for fine-grained evaluation
        :param verbose: print extra information
        :param units: print all matches and mismatches
        :returns EvaluatorResults
        """
        self.reference_yield_tags = None if r is None else create_passage_yields(r)[ALL_EDGES.name]
        g1, g2 = list(map(list, list(map(ConlluConverter().generate_graphs, (s1, s2)))))
        t1, t2 = list(map(join_tokens, (g1, g2)))
        assert t1 == t2, "Tokens do not match: '%s' != '%s'" % diff(t1, t2)
        edges = [[e for g in gs for n in g.nodes for e in n] for gs in (g1, g2)]
        for es in edges:
            for e in es:
                e.rel = e.rel.partition(":")[0]  # Ignore dep rel subtype
        maps = list(map(self.get_edges_by_construction, edges))
        if eval_type == UNLABELED:
            for es in edges:
                for e in es:
                    e.rel = None
        matches = OrderedDict()
        for construction in self.get_ordered_constructions(maps):
            g, r = [m.get(construction, set()) for m in maps]
            matches[construction] = (g & r, g - r, r - g)
        res = EvaluatorResults((c, SummaryStatistics(*list(map(len, m)))) for c, m in matches.items())
        if verbose or units:
            print()
            print("Evaluation type: (" + eval_type + ")")
            if units:
                for c, ms in matches.items():
                    print(c.description + ":")
                    for title, m in zip(("Mutual Units", "Only in guessed", "Only in reference"), ms):
                        print("==> %s:" % title)
                        print(m)
                    print()
            if verbose:
                res.print()
        return res

    def get_edges_by_construction(self, edges):
        edges_by_construction = OrderedDict()
        for edge in edges:
            candidate = Candidate(edge, reference_yield_tags=self.reference_yield_tags)
            for construction in candidate.constructions(self.constructions):
                edges_by_construction.setdefault(construction, set()).add(edge)
        return edges_by_construction

    def get_ordered_constructions(self, maps):
        ordered_constructions = [c for c in self.constructions if c in maps[0] or c in maps[1] or c == PRIMARY]
        ordered_constructions += [c for c in maps[1] if c not in ordered_constructions]
        ordered_constructions += [c for c in maps[0] if c not in ordered_constructions]
        return ordered_constructions


def join_tokens(graphs):
    return "".join((n.parent_multi_word.token.text if n.position == n.parent_multi_word.span[0] else "")
                   if n.parent_multi_word else n.token.text for g in graphs for n in g.nodes)


def diff(s1, s2):
    start, end = [len(os.path.commonprefix((s1[::d], s2[::d]))) for d in (1, -1)]
    return tuple(s[start - 1:-end] for s in (s1, s2))


def evaluate(guessed, ref, ref_yield_tags=None, converter=None, verbose=False, eval_types=EVAL_TYPES, units=False,
             constructions=DEFAULT, **kwargs):
    del kwargs
    if converter is not None:
        guessed = converter(guessed)
        ref = converter(ref)
    evaluator = ConlluEvaluator(verbose, constructions, units)
    return ConlluScores((eval_type, evaluator.get_scores(guessed, ref, eval_type, r=ref_yield_tags))
                        for eval_type in eval_types)


class ConlluScores(Scores):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "CoNLL-U"
        self.format = "conllu"
