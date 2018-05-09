from ucca import convert, layer0, layer1, textutil

from .dep import DependencyConverter

ATTR_GETTERS = {
    textutil.Attr.DEP: lambda n: n.incoming[0].rel,
    textutil.Attr.HEAD: lambda n: n.incoming[0].head_index - n.position if n.incoming[0].head_index else 0,
    textutil.Attr.TAG: lambda n: n.token.tag,
    textutil.Attr.ORTH: lambda n: n.token.text,
}

PUNCT_TAG = "PUNCT"
PUNCT = "punct"
FLAT = "flat"
PARATAXIS = "parataxis"
CC = "cc"
CONJ = "conj"
AUX = "aux"
MARK = "mark"
ADVCL = "advcl"
XCOMP = "xcomp"


HIGH_ATTACHING = (
    (lambda e: e.rel in (layer1.EdgeTags.Connector, CC), lambda e: e.rel == CONJ),
    (lambda e: e.rel == MARK, lambda e: e.rel in (ADVCL, XCOMP)),
)
REL_REPLACEMENTS = (
    (FLAT, layer1.EdgeTags.Terminal),
    (PUNCT, layer1.EdgeTags.Punctuation),
)


class ConlluConverter(DependencyConverter, convert.ConllConverter):

    def __init__(self, *args, **kwargs):
        DependencyConverter.__init__(self, *args, tree=True, punct_tag=PUNCT_TAG, punct_rel=PUNCT, scene_rel=PARATAXIS,
                                     **kwargs)

    def modify_passage(self, passage):
        passage.extra["format"] = "conllu"

    def read_line(self, *args, **kwargs):
        return self.read_line_and_append(super().read_line, *args, **kwargs)

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

    def to_format(self, *args, **kwargs):
        self.TAG_PRIORITY = [
            self.TOP,
            self.HEAD,
            self.scene_rel,
            self.punct_rel,
            CONJ,
            CC,
            AUX,
            FLAT,
        ]
        return super().to_format(*args, **kwargs)

    def add_node(self, dep_node, edge, l1):
        if self.is_flat(edge):  # Unanalyzable unit
            return edge.head.preterminal, edge.head.node
        if edge.rel == AUX:  # Auxiliary is attached as sibling of main predicate TODO revert due to new guidelines
            preterminal = edge.head.preterminal
            edge.head.preterminal = l1.add_fnode(preterminal, self.HEAD)
            return l1.add_fnode(preterminal, edge.rel)
        return super().add_node(dep_node, edge, l1)

    def preprocess_edges(self, edges, reverse=False):
        super().preprocess_edges(edges)

        def _forward_key(e):
            return e.dependent.position + ((max(e1.dependent.position for e1 in edges) + 1)
                                           if e.dependent.position < edge.dependent.position else 0)
        for edge in edges:
            edge.rel = edge.rel.partition(":")[0]  # Strip suffix
            for source, target in REL_REPLACEMENTS:
                if reverse:
                    source, target = target, source
                if edge.rel == source:
                    edge.rel = target
            for trigger, attach_to in HIGH_ATTACHING:
                if trigger(edge):
                    if reverse:
                        for attach_to_edge in sorted(filter(attach_to, edge.head.outgoing), key=_forward_key)[:1]:
                            edge.head = attach_to_edge.dependent
                    else:
                        for attach_to_edge in sorted(filter(attach_to, edge.head.incoming), key=_forward_key)[:1]:
                            edge.head = attach_to_edge.head
                    edge.head_index = edge.head.position - 1

    def is_flat(self, edge):
        return edge.rel == FLAT


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
