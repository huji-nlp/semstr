from ucca.layer1 import EdgeTags

from .dep import DependencyConverter


class SdpConverter(DependencyConverter):
    def __init__(self, **kwargs):
        super().__init__(punct_tag="_", punct_rel="_", **kwargs)

    def modify_passage(self, passage):
        passage.extra["format"] = "sdp"

    def read_line(self, *args, **kwargs):
        return self.read_line_and_append(self._read_line, *args, **kwargs)

    def _read_line(self, line, previous_node, copy_of):
        fields = self.split_line(line)
        # id, form, lemma, pos, top, pred, frame, arg1, arg2, ...
        position, text, lemma, tag, top, pred, frame = fields[:7]
        # incoming: (head positions, dependency relations, is remote for each one)
        return DependencyConverter.Node(
            int(position), [DependencyConverter.Edge.create(i + 1, rel)
                            for i, rel in enumerate(fields[7:]) if rel != "_"] or self.edges_for_orphan(top == "+"),
            token=DependencyConverter.Token(text, tag, lemma), is_head=(pred == "+"), is_top=(top == "+"), frame=frame)

    def edges_for_orphan(self, top):
        return [self.Edge(0, self.TOP, False)] if top else []

    def generate_lines(self, passage_id, dep_nodes, test, tree):
        # id, form, lemma, pos, top, pred, frame, arg1, arg2, ...
        preds = sorted({e.head_index for dep_node in dep_nodes for e in dep_node.incoming})
        for i, dep_node in enumerate(dep_nodes):
            heads = {e.head_index: e.rel + ("*" if e.remote else "") for e in dep_node.incoming}
            position = i + 1
            assert position == dep_node.position
            tag = dep_node.token.tag
            pred = "+" if i in preds else "-"
            fields = [position, dep_node.token.text, dep_node.token.lemma, tag]
            if not test:
                head_preds = [heads.get(pred, "_") for pred in preds]
                if tree and head_preds:
                    head_preds = [head_preds[0]]
                fields += ["+" if dep_node.is_top else "-", pred, "_"] + head_preds  # rel for each pred
            yield fields

    def omit_edge(self, edge, tree, linkage=False):
        return (tree or not linkage) and edge.tag == EdgeTags.LinkArgument or self.mark_aux and edge.tag.startswith("#")
