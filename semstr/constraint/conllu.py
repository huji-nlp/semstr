from ..constraints import Constraints
from ..conversion.conllu import ConlluConverter


class ConlluConstraints(Constraints):
    def __init__(self, **kwargs):
        super().__init__(unique_outgoing={ConlluConverter.HEAD}, required_outgoing={ConlluConverter.HEAD}, **kwargs)
