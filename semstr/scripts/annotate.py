#!/usr/bin/env python3

import argparse
from functools import partial

from tqdm import tqdm
from ucca import layer0
from ucca.ioutil import write_passage, read_files_and_dirs, external_write_mode
from ucca.textutil import annotate_all, Attr, get_vocab

from semstr.cfgutil import read_specs, add_specs_args
from semstr.convert import FROM_FORMAT, from_conllu
from semstr.scripts.udpipe import annotate_udpipe

desc = """Read passages in any format, and write back with spaCy/UDPipe annotations."""


CONVERTERS = {"conllu": partial(from_conllu, annotate=True)}


def copy_annotation(passages, conllu, as_array=True, verbose=False, lang=None):
    for passage, annotated in zip(passages, read_files_and_dirs(conllu, converters=CONVERTERS)):
        if verbose:
            with external_write_mode():
                print("Reading annotation from '%s'" % annotated.ID)
        if as_array:
            passage.layer(layer0.LAYER_ID).docs()[:] = annotated.layer(layer0.LAYER_ID).docs()
        else:
            for terminal, annotated_terminal in zip(passage.layer(layer0.LAYER_ID).all,
                                                    annotated.layer(layer0.LAYER_ID).all):
                # noinspection PyTypeChecker
                for attr, value in zip(Attr, annotated_terminal.tok):
                    terminal.extra[attr.key] = attr(value, get_vocab(lang=lang))
        yield passage


def main(args):
    for spec in read_specs(args, converters=FROM_FORMAT):
        if spec.udpipe:
            spec.passages = annotate_udpipe(spec.passages, spec.udpipe, as_array=args.as_array, verbose=args.verbose,
                                            lang=spec.lang)
        elif spec.conllu:
            spec.passages = copy_annotation(spec.passages, spec.conllu, as_array=args.as_array, verbose=args.verbose,
                                            lang=spec.lang)
        for passage in annotate_all(spec.passages if args.verbose else
                                    tqdm(spec.passages, unit=" passages", desc="Annotating " + spec.out_dir),
                                    as_array=args.as_array, replace=not spec.udpipe, lang=spec.lang,
                                    verbose=args.verbose):
            write_passage(passage, outdir=spec.out_dir, verbose=args.verbose, binary=args.binary)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=desc)
    add_specs_args(argparser)
    argparser.add_argument("-a", "--as-array", action="store_true", help="save annotations as array in passage level")
    argparser.add_argument("-v", "--verbose", action="store_true", help="print tagged text for each passage")
    main(argparser.parse_args())
