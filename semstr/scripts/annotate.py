#!/usr/bin/env python3

import argparse
from functools import partial

from tqdm import tqdm
from ucca import layer0
from ucca.ioutil import write_passage, read_files_and_dirs, external_write_mode
from ucca.textutil import annotate_all

from semstr.cfgutil import read_specs, add_specs_args
from semstr.convert import FROM_FORMAT, from_conllu, from_amr
from semstr.scripts.udpipe import annotate_udpipe, copy_tok_to_extra

desc = """Read passages in any format, and write back with spaCy/UDPipe annotations."""


CONVERTERS = {"conllu": partial(from_conllu, annotate=True)}

FROM_FORMAT_NO_PLACEHOLDERS = dict(FROM_FORMAT)
FROM_FORMAT_NO_PLACEHOLDERS.update({"amr": partial(from_amr, placeholders=False)})


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
                copy_tok_to_extra(annotated_terminal, terminal, lang=lang)
        yield passage


def main(args):
    for spec in read_specs(args, converters=FROM_FORMAT_NO_PLACEHOLDERS):
        kwargs = dict(as_array=args.as_array, verbose=args.verbose, lang=spec.lang)
        passages = spec.passages
        if spec.conllu:
            passages = copy_annotation(passages, spec.conllu, **kwargs)
        elif spec.udpipe:
            passages = annotate_udpipe(passages, spec.udpipe, **kwargs)
        for passage in annotate_all(passages if args.verbose else
                                    tqdm(passages, unit=" passages", desc="Annotating " + spec.out_dir),
                                    replace=spec.conllu or not spec.udpipe, **kwargs):
            if passage.extra.get("format") == "amr" and args.as_array:
                from semstr.conversion.amr import AmrConverter
                AmrConverter.introduce_placeholders(passage)
            write_passage(passage, outdir=spec.out_dir, verbose=args.verbose, binary=args.binary)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=desc)
    add_specs_args(argparser)
    argparser.add_argument("-a", "--as-array", action="store_true", help="save annotations as array in passage level")
    argparser.add_argument("-v", "--verbose", action="store_true", help="print tagged text for each passage")
    main(argparser.parse_args())
