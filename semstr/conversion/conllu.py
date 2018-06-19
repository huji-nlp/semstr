from operator import attrgetter

from ucca import layer0, layer1, textutil

from .conll import ConllConverter


def head_rel(n):
    return n.incoming[0].rel if n.incoming else None


def head_position(n):
    return n.incoming[0].head.position if n.incoming else 0


ATTR_GETTERS = {
    textutil.Attr.DEP: head_rel,
    textutil.Attr.HEAD: lambda n: head_position(n) - n.position if head_position(n) else 0,
    textutil.Attr.TAG: attrgetter("token.tag"),
    textutil.Attr.POS: attrgetter("token.pos"),
    textutil.Attr.LEMMA: attrgetter("token.lemma"),
    textutil.Attr.ORTH: attrgetter("token.text"),
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
APPOS = "appos"


HIGH_ATTACHING = {layer1.EdgeTags.Connector: (CONJ,), CC: (CONJ,), MARK: (ADVCL, XCOMP)}  # trigger: attach to rel
TOP_RELS = (layer1.EdgeTags.ParallelScene, PARATAXIS)
PUNCT_RELS = (PUNCT, layer1.EdgeTags.Punctuation)
FLAT_RELS = (FLAT, layer1.EdgeTags.Terminal)
REL_REPLACEMENTS = (FLAT_RELS, PUNCT_RELS)


class ConlluConverter(ConllConverter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, tree=True, punct_tag=PUNCT_TAG, punct_rel=PUNCT,
                         tag_priority=[self.TOP, PARATAXIS, CONJ, ADVCL, XCOMP], **kwargs)

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
        return super().to_format(*args, **kwargs)

    def generate_header_lines(self, passage_id, dep_nodes):
        yield from super().generate_header_lines(passage_id, dep_nodes)
        yield ["# text = " + " ".join(dep_node.token.text for dep_node in dep_nodes)]
        yield ["# doc_id = " + passage_id.rpartition(".")[0]]

    def add_node(self, dep_node, edge, l1):
        if self.is_flat(edge):  # Unanalyzable unit
            dep_node.preterminal = edge.head.preterminal
            dep_node.node = edge.head.node
        elif edge.rel == AUX and edge.head.preterminal:  # Attached aux as sibling of main predicate
            # TODO update to UCCA guidelines v1.0.6
            dep_node.preterminal = dep_node.node = l1.add_fnode(edge.head.preterminal, edge.rel)
            edge.head.preterminal = l1.add_fnode(edge.head.preterminal, self.HEAD)
        else:
            super().add_node(dep_node, edge, l1)

    def preprocess(self, dep_nodes, to_dep=True):
        max_pos = (max(d.position for d in dep_nodes) if dep_nodes else 0) + 1
        for dep_node in reversed(dep_nodes):
            def _attach_forward_sort_key(e):
                return e.dependent.position + (max_pos if e.dependent.position < dep_node.position else 0)
            for edge in dep_node.incoming:
                edge.rel = edge.rel.partition(":")[0]  # Strip suffix
                for source, target in REL_REPLACEMENTS:
                    if edge.rel == (source, target)[to_dep]:
                        edge.rel = (target, source)[to_dep]
                rels = HIGH_ATTACHING.get(edge.rel)
                if rels:
                    candidates = [e for e in (edge.head.incoming, edge.head.outgoing)[to_dep] if e.rel in rels]
                    if candidates:
                        head_edge = min(candidates, key=_attach_forward_sort_key)
                        head = (head_edge.head, head_edge.dependent)[to_dep]
                        if not any(e.rel == edge.rel for e in head.outgoing):
                            edge.head = head
        if to_dep:
            for dep_node in dep_nodes:
                for edge in dep_node.incoming:
                    if edge.rel in PUNCT_RELS:
                            heads = [d for d in dep_nodes if self.between(dep_node, d.incoming, CONJ)
                                     and not any(e.rel in (PUNCT_RELS + (CC,)) and
                                                 dep_node.position < e.dependent.position < d.position
                                                 for e in d.outgoing)] or \
                                    [d for d in dep_nodes if self.between(dep_node, d.outgoing, APPOS)]
                            if heads:
                                edge.head = heads[0]
        super().preprocess(dep_nodes, to_dep=to_dep)
        if to_dep:
            for dep_node in dep_nodes:
                if dep_node.incoming:
                    dep_node.enhanced = "|".join("%d:%s" % (e.head_index + 1, e.rel) for e in dep_node.incoming)

    @staticmethod
    def between(dep_node, edges, *rels):
        return any(e.rel in rels and e.head.position < dep_node.position < e.dependent.position for e in edges)

    def is_flat(self, edge):
        return edge.rel in FLAT_RELS

    def is_scene(self, edge):
        return edge.rel in TOP_RELS
