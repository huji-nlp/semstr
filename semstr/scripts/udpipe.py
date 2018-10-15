#!/usr/bin/env python3

from itertools import tee, groupby
from time import time

import argparse
import os
from operator import itemgetter
from tqdm import tqdm
from ucca import layer0, ioutil, core
from ucca.convert import split2paragraphs
from ucca.textutil import Attr, get_vocab

from semstr.cfgutil import add_verbose_arg, read_specs, add_specs_args
from semstr.convert import FROM_FORMAT, to_conllu, from_conllu
from semstr.evaluate import Scores
from semstr.evaluation.conllu import evaluate
from semstr.scripts.join import find_ids

desc = """Parse text to Universal Dependencies using UDPipe."""


def udpipe(sentences, model_name, verbose=False):
    """
    Parse text to Universal Dependencies using UDPipe.
    :param sentences: iterable of iterables of strings (one string per line)
    :param model_name: filename containing UDPipe model to load
    :param verbose: print extra information
    :return: iterable of lines containing parsed output
    """
    from ufal.udpipe import Model, Pipeline, ProcessingError
    model = Model.load(model_name)
    if not model:
        raise ValueError("Invalid model: '%s'" % model_name)
    pipeline = Pipeline(model, "conllu", Pipeline.DEFAULT, Pipeline.DEFAULT, "conllu")
    lines1, lines2 = tee(l for s in sentences for l in s)
    text = "\n".join(lines1)
    error = ProcessingError()
    num_tokens = sum(1 for l in lines2 if l)
    with ioutil.external_write_mode():
        print("Running %s on %d tokens... " % (model_name, num_tokens), end="", flush=True)
    start = time()
    processed = pipeline.process(text, error)
    duration = time() - start
    with ioutil.external_write_mode():
        print("Done (%.3fs, %.0f tokens/s)" % (duration, num_tokens / duration if duration else 0))
        if verbose:
            print(processed)
    if error.occurred():
        raise RuntimeError(error.message)
    return processed.splitlines()


def parse_udpipe(passages, model_name, verbose=False, annotate=False, terminals_only=False):
    passages1, passages2 = tee(passages)
    processed = udpipe((to_conllu_native(p, test=True, enhanced=False) for p in passages1), model_name, verbose)
    return zip(passages2, from_conllu(processed, passage_id=None, annotate=annotate, terminals_only=terminals_only))


def split(passage):
    try:
        return split2paragraphs(passage)
    except KeyError as e:
        raise RuntimeError("Failed splitting passage " + passage.ID) from e


def annotate_udpipe(passages, model_name, as_array=True, verbose=False, lang=None):
    if model_name:
        t1, t2 = tee((paragraph, passage) for passage in passages for paragraph in split(passage))
        paragraphs = map(itemgetter(0), t1)
        passages = map(itemgetter(1), t2)
        for key, group in groupby(zip(passages, parse_udpipe(paragraphs, model_name, verbose,
                                                             annotate=True, terminals_only=True)), key=itemgetter(0)):
            passage = key
            for passage, (paragraph, annotated) in group:
                # noinspection PyUnresolvedReferences
                l0 = annotated.layer(layer0.LAYER_ID)
                if l0.all:
                    i = next(iter(t.extra["orig_paragraph"] for t in paragraph.layer(layer0.LAYER_ID).all))
                    out_l0 = passage.layer(layer0.LAYER_ID)
                    if as_array:
                        out_l0.doc(i)[:] = l0.doc(1)
                    else:
                        for terminal in out_l0.all:
                            if terminal.paragraph == i:
                                copy_tok_to_extra(l0.by_position(terminal.para_pos), terminal, lang=lang)
            yield passage
    else:
        yield from passages


def copy_tok_to_extra(annotated_terminal, terminal, lang=None):
    # noinspection PyTypeChecker
    for attr, value in zip(Attr, annotated_terminal.tok):
        if value:
            terminal.extra[attr.key] = attr(value, get_vocab(lang=lang))


def split_by_empty_lines(lines, *args, **kwargs):
    del args, kwargs
    # noinspection PyTypeChecker
    yield from (list(g) + [""] for k, g in groupby(map(str.strip, lines), bool) if k)


def to_conllu_native(p, **kwargs):
    return to_conllu(p, format="conllu", **kwargs)


def strip_enhanced(p):
    return [l if not l.strip() or l.startswith("#") else
            "\t".join(l.split("\t")[:-2] + ["_", l.rpartition("\t")[-1]]) for l in p]


def open_out_file(spec, name=None):
    if spec.join:
        filename = spec.join
        if not filename.endswith(".conllu"):
            filename += ".conllu"
        return open(os.path.join(spec.out_dir, filename), "a", encoding="utf-8")
    return open(os.path.join(spec.out_dir, name + ".conllu"), "w", encoding="utf-8")


CONVERTERS = {f: lambda l: (to_conllu_native(p) for p in c(l, passage_id=None)) for f, c in FROM_FORMAT.items()}
CONVERTERS["conllu"] = split_by_empty_lines  # If getting CoNLL-U as input, don't bother converting just to convert back


def main(args):
    for spec in read_specs(args, converters=CONVERTERS):
        scores = []
        sentences, to_parse = tee((to_conllu_native(p), to_conllu_native(p, test=True, enhanced=False))
                                  if isinstance(p, core.Passage) else (p, strip_enhanced(p)) for p in spec.passages)
        t = tqdm(zip((x for x, _ in sentences),
                     split_by_empty_lines(udpipe((x for _, x in to_parse), spec.udpipe, args.verbose))),
                 unit=" sentences")
        for sentence, parsed in t:
            sentence = list(sentence)
            if args.write:
                i = next(find_ids(sentence))
                t.set_postfix(id=i)
                with open_out_file(spec, i) as f:
                    for line in parsed:
                        print(line, file=f)
            if args.evaluate:
                scores.append(evaluate(parsed, sentence, verbose=args.verbose > 1))
        if scores:
            Scores(scores).print()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=desc)
    add_specs_args(argparser)
    argparser.add_argument("-e", "--evaluate", action="store_true", help="evaluate against original passages")
    argparser.add_argument("-W", "--no-write", action="store_false", dest="write", help="do not write parsed passages")
    add_verbose_arg(argparser, help="detailed output")
    main(argparser.parse_args())
