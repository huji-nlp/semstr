from ucca import convert, layer0, textutil

from .dep import DependencyConverter

ATTR_GETTERS = {
    textutil.Attr.DEP: lambda n: n.incoming[0].rel,
    textutil.Attr.HEAD: lambda n: n.incoming[0].head_index - n.position if n.incoming[0].head_index else 0,
    textutil.Attr.TAG: lambda n: n.token.tag,
    textutil.Attr.ORTH: lambda n: n.token.text,
}


class ConlluConverter(DependencyConverter, convert.ConllConverter):
    PUNCT_TAG = "PUNCT"
    PUNCT = "punct"
    FLAT = "flat"
    PARATAXIS = "parataxis"
    CC = "cc"
    CONJ = "conj"

    def __init__(self, *args, **kwargs):
        DependencyConverter.__init__(self, *args, tree=True, punct_tag=self.PUNCT_TAG, punct_rel=self.PUNCT,
                                     flat_rel=self.FLAT, scene_rel=self.PARATAXIS, connector_rel=self.CC,
                                     conj_rel=self.CONJ, **kwargs)

    def modify_passage(self, passage):
        passage.extra["format"] = "conllu"

    def read_line(self, *args, **kwargs):
        return self.read_line_and_append(super().read_line, *args, **kwargs)

    def strip_suffix(self, rel):
        rel, *_ = rel.partition(":")
        return rel

    def from_format(self, lines, passage_id, split=False, return_original=False, annotate=False):
        for dep_nodes, sentence_id in self.build_nodes(lines, split):
            try:
                passage = self.build_passage(dep_nodes, sentence_id or passage_id)
            except (AttributeError, IndexError) as e:
                raise RuntimeError("Failed converting '%s'" % sentence_id) from e
            if annotate:
                docs = passage.layer(layer0.LAYER_ID).extra.setdefault("doc", [[]])
                for dep_node in dep_nodes[1:]:
                    paragraph = dep_node.token.paragraph
                    while len(docs) < paragraph:
                        docs.append([])
                    docs[paragraph - 1].append([ATTR_GETTERS.get(a, {}.get)(dep_node) for a in textutil.Attr])
            yield (passage, self.lines_read, passage.ID) if return_original else passage
            self.lines_read = []


def from_conllu(lines, passage_id=None, split=True, return_original=False, annotate=False, *args, **kwargs):
    """Converts from parsed text in Universal Dependencies format to a Passage object.

    :param lines: iterable of lines in Universal Dependencies format, describing a single passage.
    :param passage_id: ID to set for passage
    :param split: split each sentence to its own passage?
    :param return_original: return triple of (UCCA passage, Universal Dependencies string, sentence ID)
    :param annotate: whether to save dependency annotations in "extra" dict of layer 0

    :return generator of Passage objects
    """
    del args, kwargs
    return ConlluConverter().from_format(lines, passage_id, split, return_original=return_original, annotate=annotate)


def to_conllu(passage, test=False, tree=False, constituency=False, *args, **kwargs):
    """ Convert from a Passage object to a string in Universal Dependencies format (conllu)

    :param passage: the Passage object to convert
    :param test: whether to omit the head and deprel columns. Defaults to False
    :param tree: whether to omit columns for non-primary parents. Defaults to True
    :param constituency: use UCCA conversion that introduces intermediate non-terminals

    :return list of lines representing the semantic dependencies in the passage
    """
    del args, kwargs
    return ConlluConverter(constituency=constituency).to_format(passage, test, tree)
