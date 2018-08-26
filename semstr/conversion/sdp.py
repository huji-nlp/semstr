from ucca.layer1 import EdgeTags

from .dep import DependencyConverter


class SdpConverter(DependencyConverter):
    def __init__(self, **kwargs):
        super().__init__(punct_tag="_", punct_rel="_", format="sdp", **kwargs)

    def read_line(self, *args, **kwargs):
        return self.read_line_and_append(self._read_line, *args, **kwargs)

    def _read_line(self, line, previous_node, copy_of):
        fields = self.split_line(line)
        # id, form, lemma, pos, top, pred, frame, arg1, arg2, ...
        position, text, lemma, tag, top, pred, frame = fields[:7]
        # incoming: (head positions, dependency relations, is remote for each one)
        return DependencyConverter.Node(int(position),
                                        [DependencyConverter.Edge.create(i, rel)
                                         for i, rel in enumerate(fields[7:], start=1) if rel != "_"] or
                                        self.edges_for_orphan(top == "+"),
                                        token=DependencyConverter.Token(text, tag, lemma), is_head=(pred == "+"),
                                        is_top=(top == "+"), frame=frame)

    def edges_for_orphan(self, top):
        return [self.Edge(0, self.TOP, False)] if top else []

    def generate_lines(self, graph, test):
        yield from super().generate_lines(graph, test)
        # id, form, lemma, pos, top, pred, frame, arg1, arg2, ...
        preds = sorted({e.head_index for dep_node in graph.nodes for e in dep_node.incoming if e.head_index != 0})
        for position, dep_node in enumerate(graph.nodes, start=1):
            assert position == dep_node.position
            tag = dep_node.token.tag
            pred = "+" if position in preds else "-"
            fields = [position, dep_node.token.text, dep_node.token.lemma, tag]
            if not test:
                heads = {e.head_index: e.rel + ("*" if e.remote else "") for e in dep_node.incoming}
                head_preds = [heads.get(p, "_") for p in preds]
                if self.tree and head_preds:
                    head_preds = [head_preds[0]]
                fields += ["+" if dep_node.is_top else "-", pred, "_"] + head_preds  # rel for each pred
            yield fields

    def omit_edge(self, edge, linkage=False):
        return (self.tree or not linkage) and edge.tag == EdgeTags.LinkArgument \
               or self.mark_aux and edge.tag.startswith("#")
