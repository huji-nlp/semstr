import re
import sys
from collections import defaultdict
from itertools import islice
from operator import attrgetter

from ucca import core, layer0, layer1
from ucca.layer1 import EdgeTags

from .format import FormatConverter


class ExportConverter(FormatConverter):
    MIN_TERMINAL_ID = 500
    MAX_TERMINAL_ID = 999

    class _IdGenerator:
        def __init__(self):
            self._id = ExportConverter.MIN_TERMINAL_ID - 1

        def __call__(self):
            self._id += 1
            assert self._id <= ExportConverter.MAX_TERMINAL_ID, \
                "More than %d nodes found" % ExportConverter.MAX_TERMINAL_ID
            return str(self._id)

    def __init__(self):
        self.passage_id = self.sentence_id = self.node_by_id = self.lines_read = None

    def _init_nodes(self, line):
        m = re.match("#BOS\s+(\d+).*", line)
        assert m, "Invalid first line: " + line
        self.sentence_id = m.group(1)
        self.node_by_id = {}
        self.pending_nodes = []
        self.remotes = []
        self.linkages = defaultdict(list)
        self.terminals = []
        self.node_ids_with_children = set()
        self.in_terminals_section = True
        self.generate_id = self._IdGenerator()
        self.lines_read = []

    @staticmethod
    def _split_tags(tag, edge_tag):
        # UPARSE writes both into the node tag field, separated by "-", and the edge tag as "--"
        if edge_tag == "--":
            tag, _, edge_tag = tag.partition("-")
        return tag, edge_tag

    def _read_line(self, line):
        self.lines_read.append(line)
        fields = self.split_line(line)
        text, tag = fields[:2]
        m = re.match("#(\d+)", text)
        if m and int(m.group(1)) == ExportConverter.MIN_TERMINAL_ID:
            self.in_terminals_section = False
        if self.in_terminals_section:
            parent_id = fields[4]
            self.node_ids_with_children.add(parent_id)
            edge_tag, parent_id = fields[3:5]
            tag, edge_tag = self._split_tags(tag, edge_tag)
            self.terminals.append((text, tag, edge_tag, parent_id))
        else:
            node_id = m.group(1)
            assert node_id == self.generate_id(), "Node ID does not match order: " + node_id
            for edge_tag, parent_id in zip(fields[3::2], fields[4::2]):
                _, edge_tag = self._split_tags(tag, edge_tag)
                self.node_ids_with_children.add(parent_id)
                if parent_id == "0":
                    self.node_by_id[node_id] = None  # root node: to add to it, we add to None
                elif edge_tag.endswith("*"):
                    self.remotes.append((parent_id, edge_tag.rstrip("*"), node_id))
                elif edge_tag in (EdgeTags.LinkArgument, EdgeTags.LinkRelation):
                    self.linkages[parent_id].append((node_id, edge_tag))
                else:
                    self.pending_nodes.append((parent_id, edge_tag, node_id))

    def _build_passage(self):
        p = core.Passage(self.sentence_id or self.passage_id)
        l0 = layer0.Layer0(p)
        l1 = layer1.Layer1(p)
        paragraph = 1

        # add normal nodes
        while self.pending_nodes:
            for i in reversed(range(len(self.pending_nodes))):
                parent_id, edge_tag, node_id = self.pending_nodes[i]
                parent = self.node_by_id.get(parent_id, -1)
                if parent != -1:
                    del self.pending_nodes[i]
                    implicit = node_id not in self.node_ids_with_children
                    node = l1.add_fnode(parent, edge_tag, implicit=implicit)
                    if edge_tag == EdgeTags.Punctuation:
                        node.tag = layer1.NodeTags.Punctuation
                    self.node_by_id[node_id] = node

        # add remotes
        for parent_id, edge_tag, node_id in self.remotes:
            l1.add_remote(self.node_by_id[parent_id], edge_tag, self.node_by_id[node_id])

        # add linkages
        for node_id, children in self.linkages.items():
            link_relation = next(self.node_by_id[i] for i, t in children if t == EdgeTags.LinkRelation)
            link_arguments = [self.node_by_id[i] for i, t in children if t == EdgeTags.LinkArgument]
            l1.add_linkage(link_relation, *link_arguments)

        # add terminals
        for text, tag, edge_tag, parent_id in self.terminals:
            punctuation = (tag == layer0.NodeTags.Punct)
            terminal = l0.add_terminal(text=text, punct=punctuation, paragraph=paragraph)
            try:
                parent = self.node_by_id[parent_id]
            except KeyError as e:
                raise ValueError("Terminal ('%s') with bad parent (%s) in passage %s" % (text, parent_id, p.ID)) from e
            if parent is None:
                print("Terminal is a child of the root: '%s'" % text, file=sys.stderr)
                parent = l1.add_fnode(parent, edge_tag)
            if edge_tag != EdgeTags.Terminal:
                print("Terminal with incoming %s edge: '%s'" % (edge_tag, text), file=sys.stderr)
            parent.add(EdgeTags.Terminal, terminal)

        return p

    def from_format(self, lines, passage_id, return_original=False, **kwargs):
        self.passage_id = passage_id
        self.node_by_id = None
        for line in filter(str.strip, lines):
            if self.node_by_id is None:
                self._init_nodes(line)
            elif line.startswith("#EOS"):  # finished reading input for a passage
                passage = self._build_passage()
                passage.extra["format"] = kwargs.get("format") or "export"
                yield (passage, self.lines_read, passage.ID) if return_original else passage
                self.node_by_id = None
                self.lines_read = []
            else:  # read input line
                self._read_line(line)

    def to_format(self, passage, test=False, tree=False, **kwargs):
        lines = ["#BOS %s" % passage.ID]  # list of output lines to return
        entries = []
        nodes = list(passage.layer(layer0.LAYER_ID).all)
        node_to_id = defaultdict(self._IdGenerator())
        while nodes:
            next_nodes = []
            for node in nodes:
                if node.ID in node_to_id:
                    continue
                children = sorted((child for child in node.children if
                                   child.layer.ID != layer0.LAYER_ID and child.ID not in node_to_id and
                                   not (tree and child.attrib.get("implicit"))), key=attrgetter("ID"))
                if children:
                    next_nodes += children
                    continue
                incoming = list(islice(sorted(node.incoming,  # non-remote non-linkage first
                                              key=lambda e: (e.attrib.get("remote", False),
                                                             e.tag in (EdgeTags.LinkRelation,
                                                                       EdgeTags.LinkArgument))),
                                       1 if tree else None))  # all or just one
                next_nodes += sorted((e.parent for e in incoming), key=attrgetter("ID"))
                # word/id, (POS) tag, morph tag, edge, parent, [second edge, second parent]*
                identifier = node.text if node.layer.ID == layer0.LAYER_ID else ("#" + node_to_id[node.ID])
                fields = [identifier, node.tag, "--"]
                if not test:  # append two elements for each edge: (edge tag, parent ID)
                    fields += sum([(e.tag + ("*" if e.attrib.get("remote") else ""), e.parent.ID)
                                   for e in incoming], ()) or ("--", 0)
                entries.append(fields)
            if test:  # do not print non-terminal nodes
                break
            nodes = next_nodes
        node_to_id = dict(node_to_id)
        for fields in entries:  # correct from source standard ID to generated node IDs
            for i in range(4, len(fields), 2):
                fields[i] = node_to_id.get(fields[i], 0)
        lines += ["\t".join(str(field) for field in entry) for entry in entries] + \
                 ["#EOS %s" % passage.ID]
        return lines
