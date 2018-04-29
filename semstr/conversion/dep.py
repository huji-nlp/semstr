from ucca import convert, layer1


class DependencyConverter(convert.DependencyConverter):
    """
    Alternative converter to the one in UCCA - instead of introducing centers etc. to get a proper constituency
    structure, just copy the exact structure from the dependency graph, with all edges being between terminals (+root)
    """
    TOP = "TOP"
    HEAD = "head"

    def __init__(self, *args, constituency=False, tree=False, punct_tag=None, punct_rel=None, flat_rel=None,
                 scene_rel=None, connector_rel=None, conj_rel=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.constituency = constituency
        self.tree = tree
        self.punct_tag = punct_tag
        self.punct_rel = punct_rel
        self.flat_rel = flat_rel
        self.scene_rel = scene_rel
        self.conj_rel = conj_rel
        self.connector_rel = connector_rel
        self.lines_read = []

    def read_line_and_append(self, read_line, line, *args, **kwargs):
        self.lines_read.append(line)
        try:
            return read_line(line, *args, **kwargs)
        except ValueError as e:
            raise ValueError("Failed reading line:\n" + line) from e

    def split_line(self, line):
        return line.split("\t")

    def create_non_terminals(self, dep_nodes, l1):
        if self.constituency:
            super().create_non_terminals(dep_nodes, l1)
        if not self.tree:
            for dep_node in dep_nodes:  # Create top nodes
                if dep_node.position != 0 and not dep_node.incoming and dep_node.outgoing:
                    dep_node.node = dep_node.preterminal = l1.add_fnode(None, (self.ROOT, self.TOP)[dep_node.is_top])
        remote_edges = []
        sorted_dep_nodes = self._topological_sort(dep_nodes)
        for dep_node in sorted_dep_nodes:  # Create all other nodes
            incoming = list(dep_node.incoming)
            if dep_node.is_top and incoming[0].head_index != 0:
                top_edge = self.Edge(head_index=0, rel=self.TOP, remote=False)
                top_edge.head = dep_nodes[0]
                incoming[:0] = [top_edge]
            edge, *remotes = incoming
            if self.is_flat(edge.rel):  # Unanalyzable unit
                dep_node.preterminal = edge.head.preterminal
                dep_node.node = edge.head.node
            else:  # Add top-level edge (like UCCA H) if top-level, otherwise add child to head's node
                dep_node.node = dep_node.preterminal = \
                    l1.add_fnode(dep_node.preterminal, self.scene_rel) if edge.rel.upper() == self.ROOT else (
                        l1.add_fnode(None, self.scene_rel) if self.is_scene(edge.rel) else
                        l1.add_fnode(edge.head.node.fparent if self.is_connector(edge.rel) and edge.head.node else
                                     edge.head.node, self.strip_suffix(edge.rel)))
            if dep_node.outgoing:  # Add intermediate head non-terminal for hierarchical structure
                dep_node.preterminal = l1.add_fnode(dep_node.preterminal, self.HEAD)
            remote_edges += remotes
        for edge in remote_edges:
            parent = edge.head.node or l1.heads[0]
            child = edge.dependent.node
            if child not in parent.children and parent not in child.iter():  # Avoid cycles and multi-edges
                l1.add_remote(parent, self.strip_suffix(edge.rel), child)

    def from_format(self, lines, passage_id, split=False, return_original=False):
        for passage in super().from_format(lines, passage_id, split=split):
            yield (passage, self.lines_read, passage.ID) if return_original else passage
            self.lines_read = []

    def find_head_terminal(self, unit):
        while unit.outgoing:  # still non-terminal
            heads = [e.child for e in unit.outgoing if e.tag == self.HEAD]
            try:
                unit = heads[0] if heads else next(iter(e.child for e in unit.outgoing if not e.attrib.get("remote") and
                                                        not e.child.attrib.get("implicit")))
            except StopIteration:
                unit = unit.children[0]
        return unit

    def find_top_headed_edges(self, unit):
        return [e for e in self.find_headed_unit(unit).incoming if e.tag not in (self.ROOT, self.TOP)]

    def preprocess(self, dep_nodes):
        for dep_node in dep_nodes:
            if dep_node.incoming:
                for edge in dep_node.incoming:
                    edge.remote = False
                    if edge.rel == layer1.EdgeTags.Terminal and self.flat_rel:
                        edge.rel = self.flat_rel
                    elif edge.rel == layer1.EdgeTags.Punctuation and self.punct_rel:
                        edge.rel = self.punct_rel
                    elif self.is_connector(edge.rel):
                        for edge1 in edge.head.outgoing:
                            if edge1.rel == self.conj_rel and \
                                    not any(self.is_connector(e.rel) for e in edge1.dependent.outgoing):
                                edge.head_index = edge1.dependent.position - 1
                                edge.head = edge1.dependent
                                break
                    edge.rel = self.strip_suffix(edge.rel)
            elif self.tree:
                dep_node.incoming = [(self.Edge(head_index=-1, rel=self.ROOT.lower(), remote=False))]

    def is_top(self, unit):
        return any(e.tag == self.TOP for e in self.find_headed_unit(unit).incoming)

    def find_headed_unit(self, unit):
        while unit.incoming and (not unit.outgoing or unit.incoming[0].tag == self.HEAD) and \
                not (unit.incoming[0].tag == layer1.EdgeTags.Terminal and unit != unit.parents[0].children[0]):
            unit = unit.parents[0]
        return unit

    def is_punct(self, dep_node):
        return super().is_punct(dep_node) or dep_node.token.tag == self.punct_tag

    def is_flat(self, tag):
        return self.strip_suffix(tag) == self.flat_rel

    def is_scene(self, tag):
        return self.strip_suffix(tag) in (layer1.EdgeTags.ParallelScene, self.scene_rel)

    def is_connector(self, tag):
        return self.strip_suffix(tag) in (layer1.EdgeTags.Connector, self.connector_rel)

    def is_conj(self, tag):
        return self.strip_suffix(tag) == self.conj_rel

    def strip_suffix(self, rel):
        return rel
