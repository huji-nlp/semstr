from ..constraints import Constraints, EdgeTags
from ..conversion.conllu import ConlluConverter, PUNCT, AUX


class ConlluConstraints(Constraints):
    def __init__(self, args):
        super().__init__(args, unique_outgoing={ConlluConverter.HEAD}, required_outgoing={ConlluConverter.HEAD})
