import re
import sys
from collections import defaultdict
from itertools import groupby
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
    PUNCT = "punct"
    MULTI_WORD_TEXT_ATTRIB = "multi_word_text"
    MULTI_WORD_START_ATTRIB = "multi_word_start"
    MULTI_WORD_END_ATTRIB = "multi_word_end"

    class Graph:
        def __init__(self, dep_nodes, sentence_id, original_format=None):
            self.nodes = dep_nodes
            self.id = sentence_id
            self.format = original_format
            self.root = DependencyConverter.Node()

        @property
        def ID(self):
            return self.id

        def by_id(self, i):
            return self.by_position(int(i.split(".")[-1]))

        def by_position(self, i):
            return self.nodes[i]

        def link_pre_terminals(self):
            preterminals = []
            for dep_node in self.nodes:
                if dep_node.terminal and dep_node.preterminal is not None:  # link pre-terminal to terminal
                    dep_node.preterminal.add(EdgeTags.Terminal, dep_node.terminal)
                    preterminals.append(dep_node.preterminal)
            for preterminal in preterminals:  # update tag to PNCT when necessary
                if all(map(layer0.is_punct, preterminal.children)):
                    preterminal.tag = layer1.NodeTags.Punctuation

        def link_heads(self, multi_word_nodes=(), copy_of=None):
            heads = [self.root] + [n for n in self.nodes if n.is_head]
            for dep_node in self.nodes:
                for edge in dep_node.incoming:
                    edge.link_head(heads, copy_of)
            for dep_node in multi_word_nodes:
                start, end = dep_node.span
                for position in range(start - 1, end):
                    self.nodes[position].parent_multi_word = dep_node

        def insert_root(self):
            self.nodes.insert(0, self.root)

        def layer(self, *args, **kwargs):
            del args, kwargs
            return self

        def is_punct(self, node):
            return node.is_punct

        @property
        def all(self):
            return self

        def __iter__(self):
            return iter(self.nodes)

        def __str__(self):
            return " ".join(self.root.get_terminals()) if self.root else None

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
            self.is_punct = None

        @property
        def tag(self):
            return ""

        @property
        def text(self):
            return self.token.text

        @property
        def attrib(self):
            return {}

        @property
        def ID(self):
            return "0.%d" % self.position

        @property
        def punct(self):
            return self.is_punct

        def add_edges(self, edges):
            for remote in (False, True):
                for edge in edges:
                    if edge.remote == remote:
                        edge.dependent = self

        def get_terminals(self, punct=True, remotes=False, visited=None):
            """Returns a list of all terminals under the span of this Node.
            :param punct: whether to include punctuation nodes, defaults to True
            :param remotes: whether to include nodes from remote nodes, defaults to False
            :param visited: used to detect cycles
            :return a list of Node objects
            """
            if visited is None:
                return sorted(self.get_terminals(punct=punct, remotes=remotes, visited=set()),
                              key=attrgetter("position"))
            outgoing = {e for e in set(self) - visited if remotes or not e.remote}
            return ([self] if punct or not self.is_punct else []) + \
                [t for e in outgoing for t in e.dependent.get_terminals(
                    punct=punct, remotes=remotes, visited=visited | outgoing)]

        def __repr__(self):
            return self.token.text if self.token else DependencyConverter.ROOT

        def __eq__(self, other):
            return self.position == other.position and self.span == other.span

        def __hash__(self):
            return hash((self.position, tuple(self.span or ())))

        def __iter__(self):
            return iter(self.outgoing)

    class Edge:
        def __init__(self, head_index=None, rel=None, remote=False, head=None, dependent=None):
            self.head_index = head_index
            self._rel = self.stripped_rel = self.subtype = self._head = self._dependent = None
            self.rel = rel
            self.remote = remote
            self.head = head  # use setter
            self.dependent = dependent

        @property
        def rel(self):
            return self._rel

        @rel.setter
        def rel(self, value):
            self._rel = value
            self.stripped_rel, _, self.subtype = (None, None, None) if self._rel is None else self._rel.partition(":")

        @property
        def tag(self):
            return self.stripped_rel

        @property
        def tags(self):
            return [self.tag]

        @property
        def parent(self):
            return self.head

        @property
        def child(self):
            return self.dependent

        @property
        def attrib(self):
            return dict(remote=self.remote)

        @property
        def head(self):
            return self._head

        @head.setter
        def head(self, head):
            if self._head is not None and self in self._head.outgoing:
                self._head.outgoing.remove(self)
            self._head = head
            if head is not None:
                head.outgoing.append(self)
                self.head_index = head.position

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
            head = str(self.head_index) if self.head is None else repr(self.head)
            label = (self.rel or "") + ("*" if self.remote else "")
            if label:
                label = "-[" + label + "]"
            return head + label + "->" + repr(self.dependent)

        def __eq__(self, other):
            return self.head_index == other.head_index and self.dependent == other.dependent and \
                   self.stripped_rel == other.stripped_rel and self.remote == other.remote

        def __hash__(self):
            return hash((self.head_index, self.dependent, self.stripped_rel, self.remote))

    class Token:
        def __init__(self, text, tag, lemma=None, pos=None, features=None, paragraph=None):
            self.text = text
            self.tag = tag
            self.lemma = "_" if lemma is None else lemma
            self.pos = tag if pos is None else pos
            self.features = "_" if features is None else features
            self.paragraph = paragraph

    def __init__(self, mark_aux=False, tree=False, enhanced=True, strip_suffixes=True, punct_tag=None, punct_rel=None,
                 tag_priority=(), **kwargs):
        self.mark_aux = mark_aux
        self.tree = tree
        self.enhanced = enhanced
        self.strip_suffixes = strip_suffixes
        self.punct_tag = punct_tag
        self.punct_rel = punct_rel
        self.lines_read = []
        # noinspection PyTypeChecker
        self.tag_priority = [self.HEAD] + list(tag_priority) + self.TAG_PRIORITY + [None]
        self.format = kwargs["format"]
        self.is_ucca = self.multi_words = None

    def read_line(self, line, previous_node, copy_of):
        raise NotImplementedError()

    def generate_lines(self, graph, test):
        yield from self.generate_header_lines(graph)

    def generate_header_lines(self, graph):
        if graph.format:
            yield ["# format = " + graph.format]

    def omit_edge(self, edge):
        return False

    def modify_passage(self, passage, graph):
        pass

    @staticmethod
    def _topological_sort(graph):
        # sort into topological ordering to create parents before children
        levels = defaultdict(set)  # levels start from 0 (leaves)
        graph.root.level = 0
        remaining = [n for n in graph.nodes if not n.outgoing]  # leaves
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
                for n in sorted(level_nodes, key=lambda x: x.terminal.position)]

    @staticmethod
    def _label(dep_edge, top=False):
        dependent_rels = {e.stripped_rel for e in dep_edge.dependent}
        if dep_edge.dependent.terminal and layer0.is_punct(dep_edge.dependent.terminal):
            return EdgeTags.Punctuation
        elif top or EdgeTags.ParallelScene in dependent_rels:
            return EdgeTags.ParallelScene
        elif dependent_rels.intersection((EdgeTags.Participant, EdgeTags.Adverbial)):
            return EdgeTags.Process  # May be State but we can't tell
        else:
            return EdgeTags.Center

    def label_edge(self, dep_edge, top=False):
        return (("#" if self.mark_aux else "") + self._label(dep_edge, top=top)) if self.is_ucca else self.HEAD

    def generate_graphs(self, lines):
        # read dependencies and terminals from lines and create nodes
        sentence_id = previous_node = original_format = None
        dep_nodes = []
        multi_word_nodes = []
        copy_of = {}
        paragraph = 1

        def _graph():
            graph = self.Graph(dep_nodes, sentence_id, original_format=original_format)
            graph.link_heads(multi_word_nodes, copy_of)
            graph.insert_root()
            return graph

        for line in lines:
            line = line.strip()
            if line.startswith("#"):  # comment
                self.lines_read.append(line)
                m = re.match(r"#\s*(\d+).*", line) or re.match(r"#\s*sent_id\s*=\s*(\S+)", line)
                if m:  # comment may optionally contain the sentence ID
                    sentence_id = m.group(1)
                else:
                    m = re.match(r"#\s*format\s*=\s*(\S+)", line)
                    if m:  # comment may alternatively contain the original format
                        original_format = m.group(1)
            elif line:
                dep_node = self.read_line(line, previous_node, copy_of)  # different implementation for each subclass
                if dep_node is not None:
                    if dep_node.position and previous_node and previous_node.position:
                        assert dep_node.position == previous_node.position + 1, "'%d %s' follows '%d %s' in %s" % (
                            dep_node.position, dep_node, previous_node.position, previous_node, sentence_id)
                    previous_node = dep_node
                    dep_node.token.paragraph = paragraph  # mark down which paragraph this is in
                    (multi_word_nodes if dep_node.is_multi_word else dep_nodes).append(dep_node)
            elif dep_nodes:
                try:
                    yield _graph()
                except Exception as e:
                    print("Skipped passage '%s': %s" % (sentence_id, e), file=sys.stderr)
                sentence_id = previous_node = None
                dep_nodes = []
                multi_word_nodes = []
                paragraph = 1
            else:
                paragraph += 1
        if dep_nodes:
            yield _graph()

    def build_passage(self, graph, terminals_only=False):
        passage = core.Passage(graph.id)
        self.is_ucca = (graph.format == "ucca")
        if graph.format is None or graph.format == self.format:
            passage.extra["format"] = self.format
        self.create_terminals(graph, layer0.Layer0(passage))
        if not terminals_only:
            self.create_non_terminals(graph, layer1.Layer1(passage))
            graph.link_pre_terminals()
        return passage

    def create_non_terminals(self, graph, l1):
        for dep_node in graph.nodes:
            if dep_node.outgoing and dep_node.token:  # not the root
                if not self.is_ucca and not self.tree and dep_node.position and not dep_node.incoming:  # Top node
                    self.top_edge(graph, dep_node, rel=self.TOP if dep_node.is_top else self.ROOT)
                if self.is_punct(dep_node):  # Avoid outgoing edges from punctuation by flipping edges
                    head = dep_node.incoming[0].head if dep_node.incoming else graph.root
                    outgoing = list(dep_node)
                    for edge in outgoing:
                        edge.head = head
                    for edge in dep_node.incoming:
                        edge.head = outgoing[0].head
        remote_edges = []
        sorted_nodes = self._topological_sort(graph)
        for dep_node in sorted_nodes:  # Other nodes
            incoming = list(dep_node.incoming)
            if incoming:
                if dep_node.is_top and incoming[0].head_index != 0:
                    incoming[:0] = [self.top_edge(graph, dep_node)]
                edge, *remotes = incoming
                self.add_fnode(edge, l1)
                remote_edges += remotes
            if self.requires_head(dep_node):
                dep_node.preterminal = l1.add_fnode(  # Intermediate head for hierarchy
                    dep_node.preterminal, self.label_edge(
                        dep_node.incoming[0] if dep_node.incoming else self.top_edge(graph, dep_node)))
        for edge in remote_edges:
            parent = edge.head.node or l1.heads[0]
            child = edge.dependent.node or l1.heads[0]
            if child not in parent.children and parent not in child.iter():  # Avoid cycles and multi-edges
                l1.add_remote(parent, edge.stripped_rel if self.strip_suffixes else edge.rel, child)
        self.break_cycles(l1.heads)

    def top_edge(self, graph, dep_node, rel=TOP):
        top_edge = self.Edge(head_index=0, rel=rel, remote=False)
        top_edge.head = graph.root
        top_edge.dependent = dep_node
        return top_edge

    def create_terminals(self, graph, l0):
        for dep_node in graph.nodes:
            if dep_node.token and not dep_node.terminal:  # not the root
                dep_node.terminal = l0.add_terminal(
                    text=dep_node.token.text,
                    punct=self.is_punct(dep_node),
                    paragraph=dep_node.token.paragraph)
                extra = dep_node.terminal.extra
                extra.update(tag=dep_node.token.tag, pos=dep_node.token.pos, lemma=dep_node.token.lemma,
                             features=dep_node.token.features, enhanced=dep_node.enhanced, frame=dep_node.frame)
                multi_word = dep_node.parent_multi_word
                if multi_word:  # part of a multi-word token (e.g. zum = zu + dem)
                    extra[self.MULTI_WORD_TEXT_ATTRIB] = multi_word.token.text
                    (extra[self.MULTI_WORD_START_ATTRIB], extra[self.MULTI_WORD_END_ATTRIB]) = multi_word.span

    def from_format(self, lines, passage_id, return_original=False, terminals_only=False, dep=False,
                    preprocess=True, **kwargs):
        """Converts from parsed text in dependency format to a Passage object.

        :param lines: an iterable of lines in dependency format, describing a single passage.
        :param passage_id: ID to set for passage, in case no ID is specified in the file
        :param return_original: return original passage in addition to converted one
        :param terminals_only: create only terminals (with any annotation if specified), no non-terminals
        :param dep: return dependency graph rather than converted UCCA passage
        :param preprocess: preprocess the dependency graph before converting to UCCA (or returning it)?

        :return generator of Passage objects.
        """
        for graph in self.generate_graphs(lines):
            if not graph.id:
                graph.id = passage_id
            graph.format = kwargs.get("format") or graph.format
            if preprocess:
                self.preprocess(graph, to_dep=False)
            passage = graph if dep else self.build_passage(graph, terminals_only=terminals_only)
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
                        for e in unit if e.tag == tag and not e.child.attrib.get("implicit"))
        except StopIteration:
            # edge tags are not in the priority list, so use a simple heuristic:
            # find the child with the highest number of terminals in the yield
            return max(unit, key=lambda e: len(e.child.get_terminals()))

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

    def find_top_headed_edges(self, unit):
        """ find uppermost edges above here, to a head child from its parent.
        Referred to as N(t) in the paper.
        :param unit: unit to start from
        :return generator of edges
        """
        return [e for e in self.find_headed_unit(unit).incoming if e.tag.upper() not in (self.ROOT, self.TOP)]
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

    def find_cycle(self, unit, visited, path, remotes=True):
        if unit in visited:
            return False
        visited.add(unit)
        path.add(unit)
        for e in unit.incoming:
            if (remotes or not e.attrib.get("remote")) and \
                    (e.parent in path or self.find_cycle(e.parent, visited, path, remotes=remotes)):
                return True
        path.remove(unit)
        return False

    def break_cycles(self, nodes, remotes=True):
        # find cycles and remove them
        while True:
            path = set()
            visited = set()
            if not any(self.find_cycle(unit, visited, path, remotes=remotes) for unit in nodes):
                break
            # remove edges from cycle in priority order: first remote edges, then linker edges
            edge = min((e for unit in path for e in unit.incoming if remotes or not e.attrib.get("remote")),
                       key=lambda e: (not e.attrib.get("remote"), e.tag != EdgeTags.Linker))
            try:
                edge.remove()
            except AttributeError:
                edge.parent.remove(edge)

    def orphan_label(self, dep_node):
        return self.PUNCT if self.is_punct(dep_node) else self.ORPHAN

    def preprocess(self, graph, to_dep=True):
        if to_dep:
            self.break_cycles(graph.nodes, remotes=False)
        roots = self.roots(graph.nodes)
        if to_dep and self.tree and len(roots) > 1:
            for root in roots[1:]:
                root.incoming = [e for e in root.incoming if e.stripped_rel != self.ROOT.lower() and e.head_index != 0]
            roots = [roots[0]]
        for dep_node in graph.nodes:
            if dep_node.token:
                dep_node.is_punct = self.is_punct(dep_node)
                is_parentless = True
                for edge in list(dep_node.incoming):
                    if edge.remote:
                        if self.is_flat(edge):  # Unanalyzable remote is not possible
                            edge.remove()
                        elif not self.is_ucca:
                            edge.remote = False  # Avoid * marking
                        if edge.stripped_rel == self.ROOT.lower():
                            edge.head_index = 0
                            dep_node.incoming = [edge] + [e for e in dep_node.incoming if e != edge]  # Make root first
                        else:
                            continue
                    is_parentless = False  # Found primary parent
                if to_dep and is_parentless and self.tree:  # Must have exactly one root
                    dep_node.incoming = []
                    if roots:  # Root already exist, so attach as its child
                        edge = self.Edge(head_index=roots[0].position, rel=self.orphan_label(dep_node), remote=False)
                        edge.head = roots[0]
                        edge.dependent = dep_node
                    else:  # This is the first root
                        roots = [dep_node]
                        edge = self.Edge(head_index=0, rel=self.ROOT.lower(), remote=False)
                        edge.head = graph.root
                        edge.dependent = dep_node

    def to_format(self, passage, test=False, preprocess=True, **kwargs):
        """ Convert from a Passage object to a string in dependency format.

        :param passage: the Passage object to convert
        :param test: whether to omit the head and deprel columns. Defaults to False
        :param preprocess: preprocess the converted dependency graph before returning it?

        :return a list of strings representing the dependencies in the passage
        """
        lines = []  # list of output lines to return
        terminals = passage.layer(layer0.LAYER_ID).all  # terminal units from the passage
        original_format = kwargs.get("format") or passage.extra.get("format", "ucca")
        if original_format == self.format:
            original_format = None
        self.is_ucca = original_format == "ucca"
        self.multi_words = {}
        dep_nodes = []
        for terminal in sorted(terminals, key=attrgetter("position")):
            edges = self.incoming_edges(terminal, test)
            dep_nodes.append(self.Node(terminal.position, edges,
                                       terminal=terminal, is_top=self.is_top(terminal),
                                       token=self.Token(terminal.text, terminal.extra.get("tag", terminal.tag),
                                                        lemma=terminal.extra.get("lemma"),
                                                        pos=terminal.extra.get("pos"),
                                                        features=terminal.extra.get("features"),
                                                        paragraph=terminal.paragraph),
                                       parent_multi_word=self.parent_multi_word(terminal),
                                       enhanced=terminal.extra.get("enhanced") if self.enhanced else None,
                                       misc=terminal.extra.get("misc")))
        graph = self.Graph(dep_nodes, passage.ID, original_format=original_format)
        graph.link_heads()
        if preprocess:
            self.preprocess(graph)
        lines += ["\t".join(map(str, entry)) for entry in self.generate_lines(graph, test)] + [""]
        return lines

    def incoming_edges(self, terminal, test):
        if test:
            return []
        dep_edges = []
        for edge in self.find_top_headed_edges(terminal):
            if not self.omit_edge(edge):
                head_index = self.find_head_terminal(edge.parent).position
                dep_edges.append(self.Edge(0, self.root_label(edge), remote=False) if head_index == terminal.position
                                 else self.Edge(head_index, edge.tag, remote=edge.attrib.get("remote", False)))
        dep_edges = sorted(dep_edges, key=attrgetter("head_index"))
        # Avoid multiple edges between the same pair of node; in case of duplicates, prefer non-remote edges:
        return [sorted(es, key=attrgetter("remote"))[-1] for _, es in groupby(dep_edges, key=attrgetter("head_index"))]

    def root_label(self, edge):
        return self.ROOT.lower() if edge.parent.tag == layer1.NodeTags.Foundational and not self.is_ucca else edge.tag

    def parent_multi_word(self, terminal):
        multi_word = self.multi_words.get(terminal.position)
        if multi_word:
            return multi_word
        multi_word_text = terminal.extra.get(self.MULTI_WORD_TEXT_ATTRIB)
        if multi_word_text is None:
            return None
        multi_word_span = list(map(terminal.extra.get, (self.MULTI_WORD_START_ATTRIB, self.MULTI_WORD_END_ATTRIB)))
        span_found = any(multi_word_span)
        multi_word_span = [terminal.position if p is None else p for p in multi_word_span]
        if not span_found:
            multi_word = self.multi_words.get(terminal.position - 1)
            if not multi_word or multi_word.token.text != multi_word_text:
                multi_word = None
        if multi_word is None:
            multi_word = self.Node(terminal.position, token=self.Token(multi_word_text, tag="_"), span=multi_word_span)
        if not span_found:
            multi_word.span[-1] = terminal.position
        for i in range(multi_word_span[0], multi_word_span[1] + 1):
            self.multi_words[i] = multi_word
        return multi_word

    def read_line_and_append(self, read_line, line, *args, **kwargs):
        self.lines_read.append(line)
        try:
            return read_line(line, *args, **kwargs)
        except ValueError as e:
            raise ValueError("Failed reading line:\n" + line) from e

    def split_line(self, line):
        return line.split("\t")

    def add_fnode(self, edge, l1):
        rel = edge.stripped_rel if self.strip_suffixes else edge.rel
        if edge.stripped_rel == self.ROOT.lower():  # Add top edge (like UCCA H) if top-level
            edge.dependent.preterminal = edge.dependent.node = \
                l1.add_fnode(edge.dependent.preterminal, self.label_edge(edge, top=True))
        elif self.is_scene(edge):
            edge.dependent.preterminal = edge.dependent.node = l1.add_fnode(None, rel)
        elif self.is_primary(edge) or self.is_punct(edge.child):  # otherwise add child to head's node
            edge.dependent.preterminal = edge.dependent.node = l1.add_fnode(edge.head.node, rel)
        else:  # Unanalyzable unit, punctuation or remote (enhanced)
            edge.dependent.preterminal = edge.head.preterminal
            edge.dependent.node = edge.head.node

    @staticmethod
    def primary_edges(unit, tag=None):
        return (e for e in unit if not e.attrib.get("remote") and not e.child.attrib.get("implicit")
                and (tag is None or e.tag == tag))

    def find_head_child(self, unit):
        try:
            # noinspection PyTypeChecker
            return next(e.child for tag in self.tag_priority for e in self.primary_edges(unit, tag))
        except StopIteration as e:
            try:
                return unit.children[0]
            except IndexError:
                raise RuntimeError("Could not find head child for unit (%s): %s" % (unit.ID, unit)) from e

    def roots(self, dep_nodes):
        return [n for n in dep_nodes if n.token and any(e.stripped_rel == self.ROOT.lower() for e in n.incoming)]

    def find_headed_unit(self, unit):
        while unit.incoming and unit.parents[0].incoming and \
                (self.is_ucca and unit == self.find_head_child(unit.parents[0]) or
                 not self.is_ucca and (not unit.outgoing or unit.incoming[0].tag == self.HEAD) and
                 not (unit.incoming[0].tag == layer1.EdgeTags.Terminal and
                      unit != unit.parents[0].children[0])):
            unit = unit.parents[0]
        return unit

    def is_top(self, unit):
        return any(e.tag == self.TOP for e in self.find_headed_unit(unit).incoming)

    def is_punct(self, dep_node):
        return bool(dep_node.token and
                    {dep_node.token.tag, dep_node.token.pos} & {layer0.NodeTags.Punct, self.punct_tag})

    def is_flat(self, edge):
        return edge.stripped_rel == EdgeTags.Terminal

    def is_scene(self, edge):
        return False

    def is_primary(self, edge):
        return not self.is_punct(edge.child) and not self.is_flat(edge) and not edge.remote and not self.is_scene(edge)

    def requires_head(self, dep_node):
        return any(map(self.is_primary, dep_node.outgoing)) and not any(map(self.is_flat, dep_node.incoming))
