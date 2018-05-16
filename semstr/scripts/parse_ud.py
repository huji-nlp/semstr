#!/usr/bin/env python3

import argparse
import operator
from itertools import tee
from time import time

from tqdm import tqdm
from ucca import layer0
from ucca.normalization import normalize
from ucca.textutil import annotate_all, Attr

from semstr.cfgutil import add_verbose_arg
from semstr.conversion.conllu import ConlluConverter
from semstr.convert import from_conllu, to_conllu, add_convert_args, CONVERTERS, write_passage, map_labels
from semstr.evaluate import Scores, EVALUATORS
from semstr.scripts.annotate import add_specs_args, read_specs

desc = """Read passages in any format, extract text, parse using spaCy/UDPipe and save any format.
NOTE: the dependencies output by spaCy depend on the model used.
English models output CLEAR-style dependencies, not Universal Dependencies.
See https://spacy.io/api/annotation#section-dependency-parsing"""


def parse_spacy(passages, lang, verbose=False):
    for passage, in annotate_all(zip(passages), as_array=True, as_tuples=True, lang=lang, verbose=verbose):
        terminals = sorted(passage.layer(layer0.LAYER_ID).all, key=operator.attrgetter("position"))
        dep_nodes = [ConlluConverter.Node()] + [ConlluConverter.Node(
            t.position, terminal=t, token=ConlluConverter.Token(t.text, t.tag)) for t in terminals]
        for dep_node in dep_nodes[1:]:
            dep_node.token.paragraph = dep_node.terminal.paragraph
            head = Attr.HEAD(dep_node.terminal.tok[Attr.HEAD.value])
            if head:
                head += dep_node.position
            rel = Attr.DEP(dep_node.terminal.tok[Attr.DEP.value], lang=passage.attrib.get("lang", lang))
            assert head is not None and rel is not None, \
                "head=%r, rel=%r for token %d in %s" % (head, rel, dep_node.position, terminals)
            edge = ConlluConverter.Edge(head, rel, remote=False)
            dep_node.terminal = None
            edge.link_head(dep_nodes)
            dep_node.add_edges([edge])
        parsed = ConlluConverter().build_passage(dep_nodes, passage.ID)
        yield passage, parsed


def parse_udpipe(passages, model_name, verbose=False):
    from ufal.udpipe import Model, Pipeline, ProcessingError
    model = Model.load(model_name)
    if not model:
        raise ValueError("Invalid model: '%s'" % model_name)
    pipeline = Pipeline(model, "conllu", Pipeline.DEFAULT, Pipeline.DEFAULT, "conllu")
    passages1, passages2 = tee(passages)
    lines1, lines2 = tee(l and (l + "\t_\t_") for p in passages1 for l in to_conllu(p, tree=True, test=True))
    text = "\n".join(lines1)
    error = ProcessingError()
    num_tokens = sum(1 for l in lines2 if l)
    print("Running UDPipe on %d tokens... " % num_tokens, end="", flush=True)
    start = time()
    processed = pipeline.process(text, error)
    duration = time() - start
    print("Done (%.3fs, %.0f tokens/s)" % (duration, num_tokens / duration if duration else 0))
    if verbose:
        print(processed)
    if error.occurred():
        raise RuntimeError(error.message)
    return zip(passages2, from_conllu(processed.splitlines(), passage_id=None))


PARSERS = (SPACY, UDPIPE) = ("spacy", "udpipe")
ANNOTATORS = {SPACY: parse_spacy, UDPIPE: parse_udpipe}


def main(args):
    for passages, out_dir, lang in read_specs(args):
        scores = []
        if not args.verbose:
            passages = tqdm(passages, unit=" passages", desc="Parsing " + (out_dir if out_dir != "." else lang))
        for passage, parsed in ANNOTATORS[args.parser](passages, lang, args.verbose):
            map_labels(parsed, args.label_map)
            normalize(parsed, extra=True)
            if args.write:
                write_passage(parsed, args)
            if args.evaluate:
                evaluator = EVALUATORS[args.output_format]
                _, converter = CONVERTERS[args.output_format]
                if converter is not None:
                    passage, parsed = map(converter, (passage, parsed))
                scores.append(evaluator.evaluate(parsed, passage, verbose=args.verbose > 1))
        if scores:
            Scores(scores).print()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=desc)
    add_specs_args(argparser)
    argparser.add_argument("--parser", choices=PARSERS, default=SPACY, help="dependency parser to use (default: spacy)")
    argparser.add_argument("--output-format", choices=CONVERTERS, help="output file format (default: UCCA)")
    add_convert_args(argparser)
    argparser.add_argument("-e", "--evaluate", action="store_true", help="evaluate against original passages")
    argparser.add_argument("-W", "--no-write", action="store_false", dest="write", help="write parsed passages")
    add_verbose_arg(argparser, help="detailed output")
    main(argparser.parse_args())
