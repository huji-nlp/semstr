#!/usr/bin/env python3

import argparse
import operator
from tqdm import tqdm
from ucca import layer0, constructions as ucca_constructions
from ucca.normalization import normalize
from ucca.textutil import annotate_all, Attr

from semstr.cfgutil import add_verbose_arg, read_specs, add_specs_args
from semstr.conversion.conllu import ConlluConverter
from semstr.convert import TO_FORMAT, add_convert_args, write_passage, map_labels, FROM_FORMAT
from semstr.evaluate import Scores, EVALUATORS
from semstr.scripts.udpipe import parse_udpipe

desc = """Read passages in any format, extract text, parse using spaCy/UDPipe and save any format.
NOTE: the dependencies output by spaCy depend on the model used.
English models output CLEAR-style dependencies, not Universal Dependencies.
See https://spacy.io/api/annotation#section-dependency-parsing"""


def parse_spacy(passages, lang, verbose=False):
    for passage, in annotate_all(zip(passages), as_array=True, as_tuples=True, lang=lang, verbose=verbose):
        terminals = sorted(passage.layer(layer0.LAYER_ID).all, key=operator.attrgetter("position"))
        dep_nodes = [ConlluConverter.Node(
            t.position, terminal=t, token=ConlluConverter.Token(t.text, t.tag)) for t in terminals]
        for dep_node in dep_nodes:
            dep_node.token.paragraph = dep_node.terminal.paragraph
            head = Attr.HEAD(dep_node.terminal.tok[Attr.HEAD.value])
            if head:
                head += dep_node.position
            rel = Attr.DEP(dep_node.terminal.tok[Attr.DEP.value], lang=passage.attrib.get("lang", lang))
            assert head is not None and rel is not None, \
                "head=%r, rel=%r for token %d in:\n%s" % (head, rel, dep_node.position, " ".join(map(str, terminals)))
            edge = ConlluConverter.Edge(head, rel, remote=False)
            dep_node.terminal = None
            edge.link_head(dep_nodes)
            dep_node.add_edges([edge])
        graph = ConlluConverter.Graph(dep_nodes, passage.ID)
        graph.insert_root()
        parsed = ConlluConverter().build_passage(graph)
        yield passage, parsed


def parse(passages, lang, udpipe, verbose):
    return parse_udpipe(passages, udpipe, verbose) if udpipe else parse_spacy(passages, lang, verbose)


def main(args):
    for spec in read_specs(args, converters=FROM_FORMAT):
        scores = []
        if not args.verbose:
            spec.passages = tqdm(spec.passages, unit=" passages",
                                 desc="Parsing " + (spec.out_dir if spec.out_dir != "." else spec.lang))
        for passage, parsed in parse(spec.passages, spec.lang, spec.udpipe, args.verbose):
            map_labels(parsed, args.label_map)
            normalize(parsed, extra=True)
            if args.write:
                write_passage(parsed, **vars(args))
            if args.evaluate:
                evaluator = EVALUATORS.get(args.output_format)
                converter = TO_FORMAT.get(args.output_format)
                if converter is not None:
                    passage, parsed = map(converter, (passage, parsed))
                if evaluator is not None:
                    scores.append(evaluator(parsed, passage, constructions=args.constructions,
                                            verbose=args.verbose > 1))
        if scores:
            Scores(scores).print()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=desc)
    add_specs_args(argparser)
    argparser.add_argument("--output-format", choices=TO_FORMAT, help="output file format (default: UCCA)")
    add_convert_args(argparser)
    argparser.add_argument("-e", "--evaluate", action="store_true", help="evaluate against original passages")
    ucca_constructions.add_argument(argparser)
    argparser.add_argument("-W", "--no-write", action="store_false", dest="write", help="do not write parsed passages")
    add_verbose_arg(argparser, help="detailed output")
    main(argparser.parse_args())
