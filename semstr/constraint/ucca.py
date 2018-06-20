from ucca.layer1 import EdgeTags

from ..constraints import Constraints

LINKAGE_TAGS = {EdgeTags.LinkArgument, EdgeTags.LinkRelation}


class UccaConstraints(Constraints):
    def __init__(self, **kwargs):
        super().__init__(require_implicit_childless=True, allow_orphan_terminals=False,
                         allow_root_terminal_children=False,
                         top_level_allowed={EdgeTags.ParallelScene, EdgeTags.Linker,
                                            EdgeTags.Function, EdgeTags.Ground,
                                            EdgeTags.Punctuation, EdgeTags.LinkRelation, EdgeTags.LinkArgument},
                         possible_multiple_incoming=LINKAGE_TAGS,
                         childless_incoming_trigger=EdgeTags.Function,
                         childless_outgoing_allowed={EdgeTags.Terminal, EdgeTags.Punctuation},
                         unique_incoming={EdgeTags.Function, EdgeTags.Ground,
                                          EdgeTags.ParallelScene, EdgeTags.Linker,
                                          EdgeTags.LinkRelation, EdgeTags.Connector,
                                          EdgeTags.Punctuation, EdgeTags.Terminal},
                         unique_outgoing={EdgeTags.LinkRelation, EdgeTags.Process, EdgeTags.State},
                         mutually_exclusive_outgoing={EdgeTags.Process, EdgeTags.State},
                         exclusive_outgoing=LINKAGE_TAGS, **kwargs)
    # LinkerIncoming = {EdgeTags.Linker, EdgeTags.LinkRelation}
    # TagRule(trigger=(LinkerIncoming, None), allowed=(LinkerIncoming, None)),  # disabled due to passage 106 unit 1.300
