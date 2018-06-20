from ..constraints import Constraints
from ..util.amr import TERMINAL_DEP, WIKI, POLARITY, CENTURY, DECADE, TERMINAL_TAGS, PREFIXED_RELATION_ENUM, \
    get_node_attr, LABEL_ATTRIB, is_concept, is_valid_arg


class AmrConstraints(Constraints):
    def __init__(self, **kwargs):
        super().__init__(multigraph=True, require_implicit_childless=False, allow_orphan_terminals=True,
                         allow_root_terminal_children=True, possible_multiple_incoming={TERMINAL_DEP},
                         childless_incoming_trigger={WIKI, POLARITY, CENTURY, DECADE, "polite", "li"},
                         childless_outgoing_allowed=TERMINAL_TAGS, **kwargs)

    def allow_action(self, action, history):
        return True

    def allow_edge(self, edge):  # Prevent multiple identical edges between the same pair of nodes
        return edge.tag in PREFIXED_RELATION_ENUM or edge not in edge.parent.outgoing

    def allow_parent(self, node, tag):
        label = get_node_attr(node, LABEL_ATTRIB)
        return ((not get_node_attr(node, "implicit") or tag not in TERMINAL_TAGS) and
                (label is None or is_concept(label) or tag in TERMINAL_TAGS) and
                (not tag or is_valid_arg(node, label, tag)))

    def allow_child(self, node, tag):
        return not tag or is_valid_arg(node, get_node_attr(node, LABEL_ATTRIB), tag, is_parent=False)

    def allow_label(self, node, label):
        return (is_concept(label) or node.outgoing_tags <= TERMINAL_TAGS and not node.is_root) and \
               (not node.parents or
                is_valid_arg(node, label, *node.outgoing_tags) and
                is_valid_arg(node, label, *node.incoming_tags, is_parent=False))
