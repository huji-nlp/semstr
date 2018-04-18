#!/usr/bin/env python3

import argparse
import operator

from tqdm import tqdm
from ucca import layer0
from ucca.textutil import annotate_all, Attr

from semstr.cfgutil import add_verbose_arg
from semstr.conversion.conllu import ConlluConverter
from semstr.convert import add_convert_args, CONVERTERS, write_passage, map_labels
from semstr.evaluate import Scores, EVALUATORS
from semstr.scripts.annotate import add_specs_args, read_specs

desc = """Read passages in any format, extract text, parse using spaCy and save any format.
NOTE: the dependencies output by spaCy depend on the model used.
English models output CLEAR-style dependencies, not Universal Dependencies.
See https://spacy.io/api/annotation#section-dependency-parsing"""


def main(args):
    for ps, out_dir, lang in read_specs(args):
        scores = []
        for p, in annotate_all(zip(ps if args.verbose else tqdm(ps, unit=" passages", desc="Parsing " + out_dir)),
                               as_array=True, as_tuples=True, lang=lang, verbose=args.verbose):
            parsed = ConlluConverter().build_passage(get_dep_nodes(p, lang=p.attrib.get("lang", lang)), p.ID)
            if args.write:
                write_passage(parsed, args)
            else:
                map_labels(parsed, args.label_map)
            if args.evaluate:
                evaluator = EVALUATORS[args.output_format]
                _, converter = CONVERTERS[args.output_format]
                if converter is not None:
                    parsed = converter(parsed)
                    p = converter(p)
                scores.append(evaluator.evaluate(parsed, p, verbose=args.verbose > 1))
        if scores:
            Scores(scores).print()


def get_dep_nodes(passage, lang):
    terminals = sorted(passage.layer(layer0.LAYER_ID).all, key=operator.attrgetter("position"))
    dep_nodes = [ConlluConverter.Node()] + [ConlluConverter.Node(
        t.position, terminal=t, token=ConlluConverter.Token(t.text, t.tag)) for t in terminals]
    for dep_node in dep_nodes[1:]:
        dep_node.token.paragraph = dep_node.terminal.paragraph
        head = Attr.HEAD(dep_node.terminal.tok[Attr.HEAD.value], lang=lang)
        if head:
            head += dep_node.position
        rel = Attr.DEP(dep_node.terminal.tok[Attr.DEP.value], lang=lang)
        assert head is not None and rel is not None, \
            "head=%r, rel=%r for token %d in %s" % (head, rel, dep_node.position, terminals)
        edge = ConlluConverter.Edge(head, rel, remote=False)
        dep_node.terminal = None
        edge.link_head(dep_nodes)
        dep_node.add_edges([edge])
    return dep_nodes


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=desc)
    add_specs_args(argparser)
    argparser.add_argument("--output-format", choices=CONVERTERS, help="output file format (default: UCCA)")
    add_convert_args(argparser)
    argparser.add_argument("-e", "--evaluate", action="store_true", help="evaluate against original passages")
    argparser.add_argument("-W", "--no-write", action="store_false", dest="write", help="write parsed passages")
    add_verbose_arg(argparser, help="detailed output")
    main(argparser.parse_args())
