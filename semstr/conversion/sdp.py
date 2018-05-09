from ucca import convert

from .dep import DependencyConverter


class SdpConverter(DependencyConverter, convert.SdpConverter):
    def __init__(self, *args, **kwargs):
        DependencyConverter.__init__(self, *args, punct_tag="_", punct_rel="_", **kwargs)

    def modify_passage(self, passage):
        passage.extra["format"] = "sdp"

    def read_line(self, *args, **kwargs):
        return self.read_line_and_append(super().read_line, *args, **kwargs)

    def edges_for_orphan(self, top):
        return [self.Edge(0, self.TOP, False)] if top else []
