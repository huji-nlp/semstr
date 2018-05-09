from ..constraints import Constraints, EdgeTags
from ..conversion.conllu import ConlluConverter, PARATAXIS, PUNCT, AUX


class ConlluConstraints(Constraints):
    def __init__(self, args):
        super().__init__(args, unique_outgoing={ConlluConverter.HEAD},
                         required_outgoing={ConlluConverter.HEAD, PARATAXIS},
                         childless_incoming_trigger={ConlluConverter.HEAD},
                         childless_outgoing_allowed={EdgeTags.Terminal, EdgeTags.Punctuation, PUNCT,
                                                     ConlluConverter.HEAD, AUX})
