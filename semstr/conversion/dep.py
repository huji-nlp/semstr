import re
import sys
from collections import defaultdict
from operator import attrgetter

from ucca import core, layer0, layer1
from ucca.layer1 import EdgeTags

from .format import FormatConverter


class DependencyConverter(FormatConverter):
    """
    Abstract class for converters for dependency formats - instead of introducing UCCA centers etc., create a simple
    hierarchy with "head" edges introduced for each edge head.
    """
    ROOT = "ROOT"
    TOP = "TOP"
    HEAD = "head"
    ORPHAN = "orphan"
    MULTI_WORD_TEXT_ATTRIB = "multi_word_text"

    class Node:
        def __init__(self, position=0, incoming=None, token=None, terminal=None, is_head=True, is_top=False,
                     is_multi_word=False, parent_multi_word=None, frame=None, enhanced=None, misc=None, span=None):
            self.position = position
            self.incoming = []
            if incoming is not None:
                self.add_edges(incoming)
            self.outgoing = []
            self.token = token
            self.terminal = terminal
            self.is_head = is_head
            self.is_top = is_top
            self.is_multi_word = is_multi_word
            self.parent_multi_word = parent_multi_word
            self.node = self.level = self.preterminal = None
            self.heads_visited = set()  # for topological sort
            self.frame = "_" if frame is None else frame
            self.enhanced = "_" if enhanced is None else enhanced
            self.misc = "_" if misc is None else misc
            self.span = span

        def add_edges(self, edges):
            for edge in edges:
                edge.dependent = self

        def __repr__(self):
            return self.token.text if self.token else DependencyConverter.ROOT

        def __eq__(self, other):
            return self.position == other.position and self.span == other.span

        def __hash__(self):
            return hash((self.position, tuple(self.span or ())))

    class Edge:
        def __init__(self, head_index, rel, remote):
            self.head_index = head_index
            self.rel = rel
            self.remote = remote
            self._head = None
            self._dependent = None

        @property
        def head(self):
            return self._head

        @head.setter
        def head(self, head):
            if self._head is not None:
                self._head.outgoing.remove(self)
            self._head = head
            if head is not None:
                head.outgoing.append(self)
                self.head_index = head.position - 1

        @property
        def dependent(self):
            return self._dependent

        @dependent.setter
        def dependent(self, dependent):
            if self._dependent is not None:
                self._dependent.incoming.remove(self)
            self._dependent = dependent
            if dependent is not None:
                dependent.incoming.append(self)

        @classmethod
        def create(cls, head_position, rel):
            return cls(int(head_position), rel.rstrip("*"), rel.endswith("*"))

        def link_head(self, heads, copy_of=None):
            if isinstance(self.head_index, str):
                self.head_index = int((copy_of or {}).get(self.head_index, re.sub(r"\..*", "", self.head_index)))
            self.head = heads[self.head_index]

        def remove(self):
            self.head = self.dependent = None

        def __repr__(self):
            return (str(self.head_index) if self.head is None else repr(self.head)) + \
                   "-[" + (self.rel or "") + ("*" if self.remote else "") + "]->" + repr(self.dependent)

        def __eq__(self, other):
            return self.head_index == other.head_index and self.dependent == other.dependent and self.rel == other.rel \
                   and self.remote == other.remote

        def __hash__(self):
            return hash((self.head_index, self.dependent, self.rel, self.remote))

    class Token:
        def __init__(self, text, tag, lemma=None, pos=None, features=None, paragraph=None):
            self.text = text
            self.tag = tag
            self.lemma = "_" if lemma is None else lemma
            self.pos = tag if pos is None else pos
            self.features = "_" if features is None else features
            self.paragraph = paragraph

    def __init__(self, mark_aux=False, tree=False, punct_tag=None, punct_rel=None, tag_priority=(), **kwargs):
        del kwargs
        self.mark_aux = mark_aux
        self.tree = tree
        self.punct_tag = punct_tag
        self.punct_rel = punct_rel
        self.lines_read = []
        # noinspection PyTypeChecker
        self.tag_priority = [self.HEAD] + list(tag_priority) + self.TAG_PRIORITY + [None]

    def read_line(self, line, previous_node, copy_of):
        return self.Node()

    def generate_lines(self, passage_id, dep_nodes, test, tree):
        yield ""

    @staticmethod
    def _link_heads(dep_nodes, multi_word_nodes=(), copy_of=None):
        heads = [n for n in dep_nodes if n.is_head]
        for dep_node in dep_nodes:
            for edge in dep_node.incoming:
                edge.link_head(heads, copy_of)
        for dep_node in multi_word_nodes:
            start, end = dep_node.span
            for position in range(start, end + 1):
                dep_nodes[position].parent_multi_word = dep_node

    def omit_edge(self, edge, tree):
        return False

    def modify_passage(self, passage):
        pass

    @staticmethod
    def _topological_sort(nodes):
        # sort into topological ordering to create parents before children
        levels = defaultdict(set)  # levels start from 0 (root)
        remaining = [n for n in nodes if not n.outgoing]  # leaves
        while remaining:
            node = remaining.pop()
            if node.level is not None:  # done already
                continue
            if node.incoming:
                heads = [e.head for e in node.incoming if e.head.level is None and e.head not in node.heads_visited]
                if heads:
                    node.heads_visited.update(heads)  # to avoid cycles
                    remaining += [node] + heads
                    continue
                node.level = 1 + max(e.head.level or 0 for e in node.incoming)  # done with heads
            else:  # root
                node.level = 0
            levels[node.level].add(node)

        return [n for level, level_nodes in sorted(levels.items())
                if level > 0  # omit dummy root
                for n in sorted(level_nodes, key=lambda x: x.terminal.position)]

    @staticmethod
    def _label(node):
        dependent_rels = {e.rel for e in node.outgoing}
        if layer0.is_punct(node.terminal):
            return EdgeTags.Punctuation
        elif EdgeTags.ParallelScene in dependent_rels:
            return EdgeTags.ParallelScene
        elif EdgeTags.Participant in dependent_rels:
            return EdgeTags.Process
        else:
            return EdgeTags.Center

    def _label_edge(self, node):
        return ("#" if self.mark_aux else "") + self._label(node)

    def build_nodes(self, lines, split=False):
        # read dependencies and terminals from lines and create nodes
        sentence_id = dep_nodes = multi_word_nodes = previous_node = None
        copy_of = {}
        paragraph = 1
        for line in lines:
            line = line.strip()
            if line.startswith("#"):  # comment
                m = re.match("#\s*(\d+).*", line) or re.match("#\s*sent_id\s*=\s*(\S+)", line)
                if m:  # comment may optionally contain the sentence ID
                    sentence_id = m.group(1)
            elif line:
                if dep_nodes is None:
                    dep_nodes = [self.Node()]  # dummy root
                    multi_word_nodes = []
                dep_node = self.read_line(line, previous_node, copy_of)  # different implementation for each subclass
                if dep_node is not None:
                    previous_node = dep_node
                    dep_node.token.paragraph = paragraph  # mark down which paragraph this is in
                    (multi_word_nodes if dep_node.is_multi_word else dep_nodes).append(dep_node)
            elif split and dep_nodes:
                try:
                    self._link_heads(dep_nodes, multi_word_nodes, copy_of)
                    yield dep_nodes, sentence_id
                except Exception as e:
                    print("Skipped passage '%s': %s" % (sentence_id, e), file=sys.stderr)
                sentence_id = dep_nodes = previous_node = None
                paragraph = 1
            else:
                paragraph += 1
        if not split or dep_nodes:
            self._link_heads(dep_nodes, multi_word_nodes, copy_of)
            yield dep_nodes, sentence_id

    def build_passage(self, dep_nodes, passage_id):
        p = core.Passage(passage_id)
        self.create_terminals(dep_nodes, layer0.Layer0(p))
        self.create_non_terminals(dep_nodes, layer1.Layer1(p))
        self.link_pre_terminals(dep_nodes)
        self.modify_passage(p)
        return p

    def create_non_terminals(self, dep_nodes, l1):
        for dep_node in dep_nodes:
            if dep_node.outgoing:
                if not self.tree and dep_node.position and not dep_node.incoming:  # Create top node
                    dep_node.node = dep_node.preterminal = l1.add_fnode(None,
                                                                        (self.ROOT, self.TOP)[dep_node.is_top])
                if self.is_punct(dep_node):  # Avoid outgoing edges from punctuation by flipping edges
                    head = dep_node.incoming[0].head if dep_node.incoming else dep_nodes[0]
                    outgoing = list(dep_node.outgoing)
                    for edge in outgoing:
                        edge.head = head
                    for edge in dep_node.incoming:
                        edge.head = outgoing[0].head
        remote_edges = []
        sorted_dep_nodes = self._topological_sort(dep_nodes)
        self.preprocess(sorted_dep_nodes, to_dep=False)
        for dep_node in sorted_dep_nodes:  # Create all other nodes
            incoming = list(dep_node.incoming)
            if dep_node.is_top and incoming[0].head_index != 0:
                top_edge = self.Edge(head_index=0, rel=self.TOP, remote=False)
                top_edge.head = dep_nodes[0]
                incoming[:0] = [top_edge]
            edge, *remotes = incoming
            self.add_node(dep_node, edge, l1)
            if dep_node.outgoing and not any(map(self.is_flat, dep_node.incoming)):
                dep_node.preterminal = l1.add_fnode(dep_node.preterminal,
                                                    self.HEAD)  # Intermediate head for hierarchy
            remote_edges += remotes
        for edge in remote_edges:
            parent = edge.head.node or l1.heads[0]
            child = edge.dependent.node or l1.heads[0]
            if child not in parent.children and parent not in child.iter():  # Avoid cycles and multi-edges
                l1.add_remote(parent, edge.rel, child)

        # create nodes starting from the root and going down to pre-terminals
        # linkages = defaultdict(list)
        # for dep_node in self._topological_sort(dep_nodes):
        #     incoming_rels = {e.rel for e in dep_node.incoming}
        #     if incoming_rels == {self.ROOT}:
        #         # keep dep_node.node as None so that dependents are attached to the root
        #         dep_node.preterminal = l1.add_fnode(None, self._label_edge(dep_node))
        #     elif incoming_rels == {EdgeTags.Terminal}:  # part of non-analyzable expression
        #         head = dep_node.incoming[0].head
        #         if layer0.is_punct(head.terminal) and head.incoming and \
        #                 head.incoming[0].head.incoming:
        #             head = head.incoming[0].head  # do not put terminals and punctuation together
        #         if head.preterminal is None:
        #             head.preterminal = l1.add_fnode(None, self._label_edge(head))
        #         dep_node.preterminal = head.preterminal  # only edges to layer 0 can be Terminal
        #     else:  # usual case
        #         remotes = []
        #         for edge in dep_node.incoming:
        #             if edge.rel == EdgeTags.LinkArgument:
        #                 linkages[edge.head].append(dep_node)
        #             elif edge.remote and any(not e.remote for e in dep_node.incoming):
        #                 remotes.append(edge)
        #             elif dep_node.node is None:
        #                 dep_node.node = l1.add_fnode(edge.head.node, edge.rel)
        #                 dep_node.preterminal = l1.add_fnode(
        #                     dep_node.node, self._label_edge(dep_node)) \
        #                     if dep_node.outgoing else dep_node.node
        #             else:
        #                 # print("More than one non-remote non-linkage head for '%s': %s"
        #                 #       % (dep_node.node, dep_node.incoming), file=sys.stderr)
        #                 pass
        #
        #         # link remote edges
        #         for edge in remotes:
        #             if edge.head.node is None:  # add intermediate parent node
        #                 if edge.head.preterminal is None:
        #                     edge.head.preterminal = l1.add_fnode(None, self._label_edge(edge.head))
        #                 edge.head.node = edge.head.preterminal
        #                 edge.head.preterminal = l1.add_fnode(edge.head.node,
        #                                                      self._label_edge(edge.head))
        #             l1.add_remote(edge.head.node, edge.rel, dep_node.node)
        #
        # # link linkage arguments to relations
        # for link_relation, link_arguments in linkages.items():
        #     args = []
        #     for arg in link_arguments:
        #         if arg.node is None:  # add argument node
        #             arg.node = arg.preterminal = l1.add_fnode(None, self._label_edge(arg))
        #         args.append(arg.node)
        #     if link_relation.node is None:
        #         link_relation.node = link_relation.preterminal = l1.add_fnode(None, EdgeTags.Linker)
        #     l1.add_linkage(link_relation.node, *args)

    def create_terminals(self, dep_nodes, l0):
        for dep_node in dep_nodes:
            if dep_node.token and not dep_node.terminal:  # not the root
                dep_node.terminal = l0.add_terminal(
                    text=dep_node.token.text,
                    punct=self.is_punct(dep_node),
                    paragraph=dep_node.token.paragraph)
                dep_node.terminal.extra.update(tag=dep_node.token.tag, pos=dep_node.token.pos,
                                               lemma=dep_node.token.lemma, features=dep_node.token.features,
                                               enhanced=dep_node.enhanced, frame=dep_node.frame)
                if dep_node.parent_multi_word:  # part of a multi-word token (e.g. zum = zu + dem)
                    dep_node.terminal.extra[self.MULTI_WORD_TEXT_ATTRIB] = dep_node.parent_multi_word.token.text

    @staticmethod
    def link_pre_terminals(dep_nodes):
        preterminals = []
        for dep_node in dep_nodes:
            if dep_node.preterminal is not None:  # link pre-terminal to terminal
                dep_node.preterminal.add(EdgeTags.Terminal, dep_node.terminal)
                preterminals.append(dep_node.preterminal)
        for preterminal in preterminals:  # update tag to PNCT when necessary
            if all(map(layer0.is_punct, preterminal.children)):
                preterminal.tag = layer1.NodeTags.Punctuation

    def from_format(self, lines, passage_id, split=False, return_original=False):
        """Converts from parsed text in dependency format to a Passage object.

        :param lines: an iterable of lines in dependency format, describing a single passage.
        :param passage_id: ID to set for passage, in case no ID is specified in the file
        :param split: split each sentence to its own passage?
        :param return_original: return original passage in addition to converted one

        :return generator of Passage objects.
        """
        for dep_nodes, sentence_id in self.build_nodes(lines, split):
            passage = self.build_passage(dep_nodes, sentence_id or passage_id)
            yield (passage, self.lines_read, passage.ID) if return_original else passage
            self.lines_read = []

    TAG_PRIORITY = [  # ordered list of edge labels for head selection
        EdgeTags.Center,
        EdgeTags.Connector,
        EdgeTags.ParallelScene,
        EdgeTags.Process,
        EdgeTags.State,
        EdgeTags.Participant,
        EdgeTags.Adverbial,
        EdgeTags.Time,
        EdgeTags.Quantifier,
        EdgeTags.Elaborator,
        EdgeTags.Relator,
        EdgeTags.Function,
        EdgeTags.Linker,
        EdgeTags.LinkRelation,
        EdgeTags.LinkArgument,
        EdgeTags.Ground,
        EdgeTags.Terminal,
        EdgeTags.Punctuation,
    ]

    def find_head_child_edge(self, unit):
        """ find the outgoing edge to the head child of this unit.
        The child of the returned edge is referred to as h(u) in the paper.
        :param unit: unit to find the edges from
        :return the head outgoing edge
        """
        try:
            return next(e for tag in self.TAG_PRIORITY  # head selection by priority
                        for e in unit.outgoing if e.tag == tag and not e.child.attrib.get("implicit"))
        except StopIteration:
            # edge tags are not in the priority list, so use a simple heuristic:
            # find the child with the highest number of terminals in the yield
            return max(unit.outgoing, key=lambda e: len(e.child.get_terminals()))

    def find_head_terminal(self, unit):
        """ find the head terminal of this unit, by recursive descent.
        Referred to as h*(u) in the paper.
        :param unit: unit to find the terminal of
        :return the unit itself if it is a terminal, otherwise recursively applied to child
        """
        while unit.outgoing:  # still non-terminal
            unit = self.find_head_child(unit)
        if unit.layer.ID != layer0.LAYER_ID:
            raise ValueError("Implicit unit in conversion to dependencies (%s): %s" % (unit.ID, unit.root))
        return unit
        # while unit.outgoing:
        #     unit = self.find_head_child_edge(unit).child
        # if unit.layer.ID != layer0.LAYER_ID:
        #     raise ValueError("Implicit unit in conversion to dependencies (%s): %s" % (unit.ID, unit.root))
        # return unit

    def find_top_headed_edges(self, unit):
        """ find uppermost edges above here, to a head child from its parent.
        Referred to as N(t) in the paper.
        :param unit: unit to start from
        :return generator of edges
        """
        return [e for e in self.find_headed_unit(unit).incoming if e.tag not in (self.ROOT, self.TOP)]
        # This iterative implementation has a bug... find it and re-enable
        # remaining = list(unit.incoming)
        # ret = []
        # while remaining:
        #     edge = remaining.pop()
        #     if edge is find_head_child_edge(edge.parent):
        #         remaining += edge.parent.incoming
        #     else:
        #         ret.append(edge)
        # return ret
        # for e in unit.incoming:
        #     if e == self.find_head_child_edge(e.parent):
        #         yield from self.find_top_headed_edges(e.parent)
        #     elif self.find_head_terminal(e.parent).layer.ID == layer0.LAYER_ID:
        #         yield e

    def find_cycle(self, n, v, p):
        if n in v:
            return False
        v.add(n)
        p.add(n)
        for e in n.incoming:
            if e.head in p or self.find_cycle(e.head, v, p):
                return True
        p.remove(n)
        return False

    def break_cycles(self, dep_nodes):
        # find cycles and remove them
        while True:
            path = set()
            visited = set()
            if not any(self.find_cycle(dep_node, visited, path) for dep_node in dep_nodes):
                break
            # remove edges from cycle in priority order: first remote edges, then linker edges
            edge = min((e for dep_node in path for e in dep_node.incoming),
                       key=lambda e: (not e.remote, e.rel != EdgeTags.Linker))
            edge.remove()

    def preprocess(self, dep_nodes, to_dep=True):
        roots = self.roots(dep_nodes)
        if to_dep and self.tree and len(roots) > 1:
            for root in roots[1:]:
                root.incoming = [e for e in root.incoming if e.rel != self.ROOT.lower() and e.head_index >= 0]
            roots = [roots[0]]
        for dep_node in dep_nodes:
            is_parentless = True
            for edge in dep_node.incoming:
                if edge.remote:
                    if self.is_flat(edge):  # Unanalyzable remote is not possible
                        edge.remove()
                    else:  # Avoid * marking in CoNLL-U
                        edge.remote = False
                else:  # Found primary parent
                    is_parentless = False
            if is_parentless and self.tree:  # Must have exactly one root
                if roots:  # Root already exist, so attach as its child
                    dep_node.incoming = [self.Edge(head_index=roots[0].position - 1, rel=self.ORPHAN, remote=False)]
                else:  # This is the first root
                    roots = [dep_node]
                    dep_node.incoming = [self.Edge(head_index=-1, rel=self.ROOT.lower(), remote=False)]
        # self.break_cycles(dep_nodes)

    def to_format(self, passage, test=False, tree=True):
        """ Convert from a Passage object to a string in dependency format.

        :param passage: the Passage object to convert
        :param test: whether to omit the head and deprel columns. Defaults to False
        :param tree: whether to omit columns for non-primary parents. Defaults to True

        :return a list of strings representing the dependencies in the passage
        """
        lines = []  # list of output lines to return
        terminals = passage.layer(layer0.LAYER_ID).all  # terminal units from the passage
        multi_words = [None]
        dep_nodes = [self.Node(terminal.position, self.incoming_edges(terminal, test, tree), terminal=terminal,
                               is_top=self.is_top(terminal),
                               token=self.Token(terminal.text, terminal.extra.get("tag", terminal.tag),
                                                lemma=terminal.extra.get("lemma"),
                                                pos=terminal.extra.get("pos"),
                                                features=terminal.extra.get("features"),
                                                paragraph=terminal.paragraph),
                               parent_multi_word=self.parent_multi_word(terminal, multi_words),
                               enhanced=terminal.extra.get("enhanced"), misc=terminal.extra.get("misc"))
                     for terminal in sorted(terminals, key=attrgetter("position"))]
        self._link_heads(dep_nodes)
        self.preprocess(dep_nodes)
        lines += ["\t".join(map(str, entry)) for entry in self.generate_lines(passage.ID, dep_nodes, test, tree)] + [""]
        return lines

    def incoming_edges(self, terminal, test, tree):
        if test:
            return []
        edges = list(self.find_top_headed_edges(terminal))
        head_indices = [self.find_head_terminal(e.parent).position - 1 for e in edges]
        # (head positions, dependency relations, is remote for each one)
        return {self.Edge(head_index, e.tag, e.attrib.get("remote", False))
                for e, head_index in zip(edges, head_indices)
                if head_index != terminal.position - 1 and  # avoid self loops
                not self.omit_edge(e, tree)}  # different implementation for each subclass

    def parent_multi_word(self, terminal, multi_words):
        multi_word_text = terminal.extra.get(self.MULTI_WORD_TEXT_ATTRIB)
        if multi_word_text is None:
            multi_words[0] = None
        elif multi_words[0] is None or multi_word_text != multi_words[0].token.text:
            multi_words[0] = self.Node(terminal.position, token=self.Token(multi_word_text, tag="_"),
                                       span=2 * [terminal.position])
        else:
            multi_words[0].span[-1] = terminal.position
        return multi_words[0]

    def read_line_and_append(self, read_line, line, *args, **kwargs):
        self.lines_read.append(line)
        try:
            return read_line(line, *args, **kwargs)
        except ValueError as e:
            raise ValueError("Failed reading line:\n" + line) from e

    def split_line(self, line):
        return line.split("\t")

    def add_node(self, dep_node, edge, l1):
        # Add top-level edge (like UCCA H) if top-level, otherwise add child to head's node
        dep_node.preterminal = dep_node.node = \
            l1.add_fnode(dep_node.preterminal, self.HEAD) if edge.rel.upper() == self.ROOT else (
                l1.add_fnode(None if self.is_scene(edge) else edge.head.node, edge.rel))

    @staticmethod
    def primary_edges(unit, tag=None):
        return (e for e in unit if not e.attrib.get("remote") and not e.child.attrib.get("implicit")
                and (tag is None or e.tag == tag))

    def find_head_child(self, unit):
        try:
            # noinspection PyTypeChecker
            return next(e.child for tag in self.tag_priority for e in self.primary_edges(unit, tag))
        except StopIteration:
            raise RuntimeError("Could not find head child for unit (%s): %s" % (unit.ID, unit))

    def roots(self, dep_nodes):
        return [n for n in dep_nodes if any(e.rel == self.ROOT.lower() for e in n.incoming)]

    def find_headed_unit(self, unit):
        while unit.incoming and (not unit.outgoing or unit.incoming[0].tag == self.HEAD) and \
                not (unit.incoming[0].tag == layer1.EdgeTags.Terminal and unit != unit.parents[0].children[0]):
            unit = unit.parents[0]
        return unit

    def is_top(self, unit):
        return any(e.tag == self.TOP for e in self.find_headed_unit(unit).incoming)

    def is_punct(self, dep_node):
        return dep_node.token and {dep_node.token.tag, dep_node.token.pos} & {layer0.NodeTags.Punct, self.punct_tag}

    def is_flat(self, edge):
        return False

    def is_scene(self, edge):
        return False
